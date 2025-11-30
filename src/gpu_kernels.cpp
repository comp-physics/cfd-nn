#include "gpu_kernels.hpp"
#include <cmath>
#include <algorithm>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

namespace nncfd {
namespace gpu_kernels {

// ============================================================================
// GPU Kernel: Compute velocity gradients
// ============================================================================
void compute_velocity_gradients_gpu(
    const double* u, const double* v,
    double* dudx, double* dudy,
    double* dvdx, double* dvdy,
    int Nx, int Ny,
    double dx, double dy)
{
#ifdef USE_GPU_OFFLOAD
    const int stride = Nx + 2;  // Total width including ghost cells
    const double inv_2dx = 1.0 / (2.0 * dx);
    const double inv_2dy = 1.0 / (2.0 * dy);
    
    #pragma omp target teams distribute parallel for collapse(2)
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            // Interior index (0-based for output arrays)
            int idx_out = j * Nx + i;
            
            // Full mesh index (1-based for input arrays with ghost cells)
            int ii = i + 1;
            int jj = j + 1;
            int idx_c = jj * stride + ii;
            int idx_ip = jj * stride + (ii + 1);
            int idx_im = jj * stride + (ii - 1);
            int idx_jp = (jj + 1) * stride + ii;
            int idx_jm = (jj - 1) * stride + ii;
            
            // Central differences
            dudx[idx_out] = (u[idx_ip] - u[idx_im]) * inv_2dx;
            dudy[idx_out] = (u[idx_jp] - u[idx_jm]) * inv_2dy;
            dvdx[idx_out] = (v[idx_ip] - v[idx_im]) * inv_2dx;
            dvdy[idx_out] = (v[idx_jp] - v[idx_jm]) * inv_2dy;
        }
    }
#else
    (void)u; (void)v; (void)dudx; (void)dudy; (void)dvdx; (void)dvdy;
    (void)Nx; (void)Ny; (void)dx; (void)dy;
#endif
}

// ============================================================================
// GPU Kernel: Compute TBNN features and tensor basis
// ============================================================================
void compute_tbnn_features_gpu(
    const double* dudx, const double* dudy,
    const double* dvdx, const double* dvdy,
    const double* k, const double* omega,
    const double* wall_distance,
    double* features,
    double* basis,
    int n_cells,
    double nu, double delta)
{
#ifdef USE_GPU_OFFLOAD
    const double C_mu = 0.09;
    
    #pragma omp target teams distribute parallel for
    for (int idx = 0; idx < n_cells; ++idx) {
        // Get gradients
        double dudx_v = dudx[idx];
        double dudy_v = dudy[idx];
        double dvdx_v = dvdx[idx];
        double dvdy_v = dvdy[idx];
        
        // Strain rate tensor components: S_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
        double Sxx = dudx_v;
        double Syy = dvdy_v;
        double Sxy = 0.5 * (dudy_v + dvdx_v);
        
        // Rotation tensor: Omega_ij = 0.5 * (du_i/dx_j - du_j/dx_i)
        double Oxy = 0.5 * (dudy_v - dvdx_v);
        
        // Magnitudes
        double S_mag = sqrt(2.0 * (Sxx*Sxx + Syy*Syy + 2.0*Sxy*Sxy));
        double Omega_mag = sqrt(2.0 * Oxy * Oxy);
        
        // Get k, omega, epsilon
        double k_val = k[idx];
        double omega_val = omega[idx];
        double eps = C_mu * k_val * omega_val;
        
        // Safe values
        double k_safe = (k_val > 1e-10) ? k_val : 1e-10;
        double eps_safe = (eps > 1e-20) ? eps : 1e-20;
        
        // Time scale for normalization
        double tau = k_safe / eps_safe;
        
        // Normalized strain and rotation
        double S_norm = S_mag * tau;
        double Omega_norm = Omega_mag * tau;
        
        // Normalized tensor components
        double Sxx_n = Sxx * tau;
        double Syy_n = Syy * tau;
        double Sxy_n = Sxy * tau;
        double Oxy_n = Oxy * tau;
        
        // ==================== Features (5 values) ====================
        int feat_base = idx * 5;
        features[feat_base + 0] = S_norm * S_norm;           // ~tr(S_norm^2)
        features[feat_base + 1] = Omega_norm * Omega_norm;   // ~tr(Omega_norm^2)
        features[feat_base + 2] = Sxx_n*Sxx_n + Syy_n*Syy_n + 2.0*Sxy_n*Sxy_n;  // tr(S^2)
        features[feat_base + 3] = 2.0 * Oxy_n * Oxy_n;       // tr(Omega^2)
        features[feat_base + 4] = wall_distance[idx] / delta; // Normalized wall distance
        
        // ==================== Tensor Basis (4 tensors × 3 components) ====================
        int basis_base = idx * 12;
        
        // T^(1) = S (normalized)
        basis[basis_base + 0] = Sxx_n;  // T1_xx
        basis[basis_base + 1] = Sxy_n;  // T1_xy
        basis[basis_base + 2] = Syy_n;  // T1_yy
        
        // T^(2) = S*Omega - Omega*S (commutator)
        // [S, Omega] = [[-2*Sxy*Oxy, (Sxx-Syy)*Oxy], [(Sxx-Syy)*Oxy, 2*Sxy*Oxy]]
        basis[basis_base + 3] = -2.0 * Sxy_n * Oxy_n;        // T2_xx
        basis[basis_base + 4] = (Sxx_n - Syy_n) * Oxy_n;     // T2_xy
        basis[basis_base + 5] = 2.0 * Sxy_n * Oxy_n;         // T2_yy
        
        // T^(3) = S^2 - (1/2)*tr(S^2)*I (deviatoric part of S^2)
        double S2xx = Sxx_n*Sxx_n + Sxy_n*Sxy_n;
        double S2yy = Sxy_n*Sxy_n + Syy_n*Syy_n;
        double S2xy = Sxy_n * (Sxx_n + Syy_n);
        double trS2 = S2xx + S2yy;
        basis[basis_base + 6] = S2xx - 0.5 * trS2;           // T3_xx
        basis[basis_base + 7] = S2xy;                         // T3_xy
        basis[basis_base + 8] = S2yy - 0.5 * trS2;           // T3_yy
        
        // T^(4) = 0 in 2D (Omega^2 is proportional to identity)
        basis[basis_base + 9]  = 0.0;   // T4_xx
        basis[basis_base + 10] = 0.0;   // T4_xy
        basis[basis_base + 11] = 0.0;   // T4_yy
    }
    
    (void)nu;  // Not used in current implementation
#else
    (void)dudx; (void)dudy; (void)dvdx; (void)dvdy;
    (void)k; (void)omega; (void)wall_distance;
    (void)features; (void)basis;
    (void)n_cells; (void)nu; (void)delta;
#endif
}

// ============================================================================
// GPU Kernel: Postprocess NN outputs
// ============================================================================
void postprocess_nn_outputs_gpu(
    const double* nn_outputs,
    const double* basis,
    const double* k,
    const double* dudx, const double* dudy,
    const double* dvdx, const double* dvdy,
    double* nu_t,
    double* tau_xx, double* tau_xy, double* tau_yy,
    int n_cells, int output_dim,
    double nu_ref)
{
#ifdef USE_GPU_OFFLOAD
    const int NUM_BASIS = 4;
    const bool compute_tau = (tau_xx != nullptr);
    
    #pragma omp target teams distribute parallel for
    for (int idx = 0; idx < n_cells; ++idx) {
        // Extract G coefficients from NN output
        double G[4] = {0.0, 0.0, 0.0, 0.0};
        int out_base = idx * output_dim;
        for (int n = 0; n < NUM_BASIS && n < output_dim; ++n) {
            G[n] = nn_outputs[out_base + n];
        }
        
        // Get basis tensors
        int basis_base = idx * 12;
        
        // Construct anisotropy tensor: b_ij = sum_n G_n * T^n_ij
        double b_xx = 0.0, b_xy = 0.0, b_yy = 0.0;
        for (int n = 0; n < NUM_BASIS; ++n) {
            b_xx += G[n] * basis[basis_base + n*3 + 0];
            b_xy += G[n] * basis[basis_base + n*3 + 1];
            b_yy += G[n] * basis[basis_base + n*3 + 2];
        }
        
        // Compute Reynolds stresses if requested
        double k_val = k[idx];
        if (compute_tau) {
            double k_safe = (k_val > 0.0) ? k_val : 0.0;
            tau_xx[idx] = 2.0 * k_safe * (b_xx + 1.0/3.0);
            tau_xy[idx] = 2.0 * k_safe * b_xy;
            tau_yy[idx] = 2.0 * k_safe * (b_yy + 1.0/3.0);
        }
        
        // Compute equivalent eddy viscosity
        // nu_t = -b_xy * k / S_xy (when S_xy is significant)
        double dudy_v = dudy[idx];
        double dvdx_v = dvdx[idx];
        double Sxy = 0.5 * (dudy_v + dvdx_v);
        
        double nu_t_val = 0.0;
        if (fabs(Sxy) > 1e-10) {
            nu_t_val = fabs(-b_xy * k_val / Sxy);
        } else {
            // Fallback: use strain magnitude
            double dudx_v = dudx[idx];
            double dvdy_v = dvdy[idx];
            double Sxx = dudx_v;
            double Syy = dvdy_v;
            double S_mag = sqrt(2.0 * (Sxx*Sxx + Syy*Syy + 2.0*Sxy*Sxy));
            if (S_mag > 1e-10) {
                double b_mag = sqrt(b_xx*b_xx + 2.0*b_xy*b_xy + b_yy*b_yy);
                nu_t_val = k_val * b_mag / S_mag;
            }
        }
        
        // Clip to reasonable bounds
        double max_nu_t = 10.0 * nu_ref;
        nu_t_val = (nu_t_val > 0.0) ? nu_t_val : 0.0;
        nu_t_val = (nu_t_val < max_nu_t) ? nu_t_val : max_nu_t;
        
        // Check for NaN/Inf
        if (nu_t_val != nu_t_val || nu_t_val > 1e30) {  // NaN check: x != x is true only for NaN
            nu_t_val = 0.0;
        }
        
        nu_t[idx] = nu_t_val;
    }
#else
    (void)nn_outputs; (void)basis; (void)k;
    (void)dudx; (void)dudy; (void)dvdx; (void)dvdy;
    (void)nu_t; (void)tau_xx; (void)tau_xy; (void)tau_yy;
    (void)n_cells; (void)output_dim; (void)nu_ref;
#endif
}

// ============================================================================
// GPU Kernel: Full TBNN pipeline
// ============================================================================
void tbnn_full_pipeline_gpu(
    const double* u, const double* v,
    const double* k, const double* omega,
    const double* wall_distance,
    const double* nn_weights, const double* nn_biases,
    const int* weight_offsets, const int* bias_offsets,
    const int* layer_dims, const int* activation_types,
    int n_layers,
    const double* input_means, const double* input_stds,
    int scale_size,
    double* workspace,
    double* nu_t,
    double* tau_xx, double* tau_xy, double* tau_yy,
    int Nx, int Ny, double dx, double dy,
    double nu, double delta)
{
#ifdef USE_GPU_OFFLOAD
    const int n_cells = Nx * Ny;
    const int stride = Nx + 2;
    const double inv_2dx = 1.0 / (2.0 * dx);
    const double inv_2dy = 1.0 / (2.0 * dy);
    const double C_mu = 0.09;
    const int NUM_BASIS = 4;
    const int FEATURE_DIM = 5;
    const bool compute_tau = (tau_xx != nullptr);
    const bool do_scaling = (input_means != nullptr && scale_size > 0);
    
    // Get max layer dimension for NN workspace (on host, before entering target region)
    int max_dim = 0;
    for (int l = 0; l < n_layers; ++l) {
        int in_dim = layer_dims[l * 2];
        int out_dim = layer_dims[l * 2 + 1];
        if (in_dim > max_dim) max_dim = in_dim;
        if (out_dim > max_dim) max_dim = out_dim;
    }
    int output_dim = layer_dims[(n_layers - 1) * 2 + 1];
    
    // ==================== FUSED KERNEL ====================
    // All operations in one kernel to maximize data locality
    // Use is_device_ptr for data already mapped to GPU
    #pragma omp target teams distribute parallel for \
        is_device_ptr(u, v, k, omega, wall_distance, \
                      nn_weights, nn_biases, weight_offsets, bias_offsets, \
                      layer_dims, activation_types, \
                      input_means, input_stds, \
                      workspace, nu_t, tau_xx, tau_xy, tau_yy)
    for (int cell_idx = 0; cell_idx < n_cells; ++cell_idx) {
        // Workspace pointers (computed inside kernel since workspace is device ptr)
        double* dudx_arr = workspace;
        double* dudy_arr = workspace + n_cells;
        double* dvdx_arr = workspace + n_cells * 2;
        double* dvdy_arr = workspace + n_cells * 3;
        double* features_arr = workspace + n_cells * 4;
        double* basis_arr = workspace + n_cells * 4 + n_cells * FEATURE_DIM;
        double* nn_workspace_arr = workspace + n_cells * 4 + n_cells * FEATURE_DIM + n_cells * 12;
        double* nn_outputs_arr = nn_workspace_arr + n_cells * max_dim * 2;
        
        // Convert flat index to (i, j) - interior coordinates
        int i = cell_idx % Nx;
        int j = cell_idx / Nx;
        
        // Full mesh indices (with ghost cells)
        int ii = i + 1;
        int jj = j + 1;
        int idx_ip = jj * stride + (ii + 1);
        int idx_im = jj * stride + (ii - 1);
        int idx_jp = (jj + 1) * stride + ii;
        int idx_jm = (jj - 1) * stride + ii;
        
        // ========== Step 1: Compute velocity gradients ==========
        double dudx_v = (u[idx_ip] - u[idx_im]) * inv_2dx;
        double dudy_v = (u[idx_jp] - u[idx_jm]) * inv_2dy;
        double dvdx_v = (v[idx_ip] - v[idx_im]) * inv_2dx;
        double dvdy_v = (v[idx_jp] - v[idx_jm]) * inv_2dy;
        
        // Store for later use
        dudx_arr[cell_idx] = dudx_v;
        dudy_arr[cell_idx] = dudy_v;
        dvdx_arr[cell_idx] = dvdx_v;
        dvdy_arr[cell_idx] = dvdy_v;
        
        // ========== Step 2: Compute strain/rotation ==========
        double Sxx = dudx_v;
        double Syy = dvdy_v;
        double Sxy = 0.5 * (dudy_v + dvdx_v);
        double Oxy = 0.5 * (dudy_v - dvdx_v);
        
        double S_mag = sqrt(2.0 * (Sxx*Sxx + Syy*Syy + 2.0*Sxy*Sxy));
        double Omega_mag = sqrt(2.0 * Oxy * Oxy);
        
        double k_val = k[cell_idx];
        double omega_val = omega[cell_idx];
        double eps = C_mu * k_val * omega_val;
        
        double k_safe = (k_val > 1e-10) ? k_val : 1e-10;
        double eps_safe = (eps > 1e-20) ? eps : 1e-20;
        double tau = k_safe / eps_safe;
        
        double S_norm = S_mag * tau;
        double Omega_norm = Omega_mag * tau;
        double Sxx_n = Sxx * tau;
        double Syy_n = Syy * tau;
        double Sxy_n = Sxy * tau;
        double Oxy_n = Oxy * tau;
        
        // ========== Step 3: Compute features ==========
        double feat[5];
        feat[0] = S_norm * S_norm;
        feat[1] = Omega_norm * Omega_norm;
        feat[2] = Sxx_n*Sxx_n + Syy_n*Syy_n + 2.0*Sxy_n*Sxy_n;
        feat[3] = 2.0 * Oxy_n * Oxy_n;
        feat[4] = wall_distance[cell_idx] / delta;
        
        // Apply input scaling
        if (do_scaling) {
            for (int f = 0; f < FEATURE_DIM && f < scale_size; ++f) {
                feat[f] = (feat[f] - input_means[f]) / input_stds[f];
            }
        }
        
        // Store features
        for (int f = 0; f < FEATURE_DIM; ++f) {
            features_arr[cell_idx * FEATURE_DIM + f] = feat[f];
        }
        
        // ========== Step 4: Compute tensor basis ==========
        double T[4][3];  // 4 basis tensors, 3 components each (xx, xy, yy)
        
        // T^(1) = S
        T[0][0] = Sxx_n; T[0][1] = Sxy_n; T[0][2] = Syy_n;
        
        // T^(2) = [S, Omega]
        T[1][0] = -2.0 * Sxy_n * Oxy_n;
        T[1][1] = (Sxx_n - Syy_n) * Oxy_n;
        T[1][2] = 2.0 * Sxy_n * Oxy_n;
        
        // T^(3) = S^2 - (1/2)*tr(S^2)*I
        double S2xx = Sxx_n*Sxx_n + Sxy_n*Sxy_n;
        double S2yy = Sxy_n*Sxy_n + Syy_n*Syy_n;
        double S2xy = Sxy_n * (Sxx_n + Syy_n);
        double trS2 = S2xx + S2yy;
        T[2][0] = S2xx - 0.5 * trS2;
        T[2][1] = S2xy;
        T[2][2] = S2yy - 0.5 * trS2;
        
        // T^(4) = 0
        T[3][0] = 0.0; T[3][1] = 0.0; T[3][2] = 0.0;
        
        // Store basis
        for (int n = 0; n < NUM_BASIS; ++n) {
            basis_arr[cell_idx * 12 + n*3 + 0] = T[n][0];
            basis_arr[cell_idx * 12 + n*3 + 1] = T[n][1];
            basis_arr[cell_idx * 12 + n*3 + 2] = T[n][2];
        }
        
        // ========== Step 5: NN Forward Pass ==========
        // Each cell has its own workspace slice
        double* buf1 = nn_workspace_arr + cell_idx * max_dim * 2;
        double* buf2 = buf1 + max_dim;
        
        // Copy features to buf1
        for (int f = 0; f < FEATURE_DIM; ++f) {
            buf1[f] = feat[f];
        }
        
        double* current = buf1;
        double* next = buf2;
        
        for (int l = 0; l < n_layers; ++l) {
            int in_dim = layer_dims[l * 2];
            int out_dim_l = layer_dims[l * 2 + 1];
            int w_off = weight_offsets[l];
            int b_off = bias_offsets[l];
            int act_type = activation_types[l];
            
            // Matrix-vector multiply: next = W * current + b
            for (int o = 0; o < out_dim_l; ++o) {
                double sum = nn_biases[b_off + o];
                for (int k_idx = 0; k_idx < in_dim; ++k_idx) {
                    sum += nn_weights[w_off + o * in_dim + k_idx] * current[k_idx];
                }
                
                // Apply activation (matching enum: Linear=0, ReLU=1, Tanh=2, Sigmoid=3, Swish=4, GELU=5)
                if (act_type == 2) {  // Tanh
                    sum = tanh(sum);
                } else if (act_type == 1) {  // ReLU
                    sum = (sum > 0.0) ? sum : 0.0;
                } else if (act_type == 3) {  // Sigmoid
                    sum = 1.0 / (1.0 + exp(-sum));
                } else if (act_type == 4) {  // Swish
                    sum = sum / (1.0 + exp(-sum));
                } else if (act_type == 5) {  // GELU
                    double c = 0.044715;
                    double s3 = sum * sum * sum;
                    sum = 0.5 * sum * (1.0 + tanh(sqrt(2.0/3.14159265358979323846) * (sum + c * s3)));
                }
                // act_type == 0: Linear (no activation)
                
                next[o] = sum;
            }
            
            // Swap buffers
            double* tmp = current;
            current = next;
            next = tmp;
        }
        
        // Copy output (G coefficients)
        double G[4] = {0.0, 0.0, 0.0, 0.0};
        for (int n = 0; n < NUM_BASIS && n < output_dim; ++n) {
            G[n] = current[n];
            nn_outputs_arr[cell_idx * output_dim + n] = current[n];
        }
        
        // ========== Step 6: Construct anisotropy ==========
        double b_xx = 0.0, b_xy = 0.0, b_yy = 0.0;
        for (int n = 0; n < NUM_BASIS; ++n) {
            b_xx += G[n] * T[n][0];
            b_xy += G[n] * T[n][1];
            b_yy += G[n] * T[n][2];
        }
        
        // ========== Step 7: Compute Reynolds stresses (optional) ==========
        if (compute_tau) {
            double k_safe2 = (k_val > 0.0) ? k_val : 0.0;
            tau_xx[cell_idx] = 2.0 * k_safe2 * (b_xx + 1.0/3.0);
            tau_xy[cell_idx] = 2.0 * k_safe2 * b_xy;
            tau_yy[cell_idx] = 2.0 * k_safe2 * (b_yy + 1.0/3.0);
        }
        
        // ========== Step 8: Compute eddy viscosity ==========
        double nu_t_val = 0.0;
        if (fabs(Sxy) > 1e-10) {
            nu_t_val = fabs(-b_xy * k_val / Sxy);
        } else {
            if (S_mag > 1e-10) {
                double b_mag = sqrt(b_xx*b_xx + 2.0*b_xy*b_xy + b_yy*b_yy);
                nu_t_val = k_val * b_mag / S_mag;
            }
        }
        
        // Clip and validate
        double max_nu_t = 10.0 * nu;
        nu_t_val = (nu_t_val > 0.0) ? nu_t_val : 0.0;
        nu_t_val = (nu_t_val < max_nu_t) ? nu_t_val : max_nu_t;
        if (nu_t_val != nu_t_val || nu_t_val > 1e30) {
            nu_t_val = 0.0;
        }
        
        nu_t[cell_idx] = nu_t_val;
    }
#else
    (void)u; (void)v; (void)k; (void)omega; (void)wall_distance;
    (void)nn_weights; (void)nn_biases;
    (void)weight_offsets; (void)bias_offsets;
    (void)layer_dims; (void)activation_types; (void)n_layers;
    (void)input_means; (void)input_stds; (void)scale_size;
    (void)workspace; (void)nu_t;
    (void)tau_xx; (void)tau_xy; (void)tau_yy;
    (void)Nx; (void)Ny; (void)dx; (void)dy;
    (void)nu; (void)delta;
#endif
}

} // namespace gpu_kernels
} // namespace nncfd

