#include "gpu_kernels.hpp"
#include "numerics.hpp"
#include "profiling.hpp"
#include <cmath>
#include <algorithm>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

namespace nncfd {
namespace gpu_kernels {

// ============================================================================
// GPU Kernel: Compute cell-centered gradients from staggered MAC grid
// ============================================================================
void compute_gradients_from_mac_gpu(
    const double* u_face,
    const double* v_face,
    double* dudx_cell,
    double* dudy_cell,
    double* dvdx_cell,
    double* dvdy_cell,
    int Nx, int Ny,
    int Ng,
    double dx, double dy,
    int u_stride,
    int v_stride,
    int cell_stride,
    int u_total_size,
    int v_total_size,
    int cell_total_size)
{
    NVTX_SCOPE_GRADIENT("kernel:gradients_from_mac");

    const double inv_2dx = 1.0 / (2.0 * dx);
    const double inv_2dy = 1.0 / (2.0 * dy);

#ifdef USE_GPU_OFFLOAD
    // GPU path: parallelize over interior cells
    // CRITICAL: map(present:...) indicates these arrays are already mapped by caller
    // NVHPC requires map(present:) not map(present, alloc:) for pre-mapped data
    #pragma omp target teams distribute parallel for collapse(2) \
        map(present: u_face[0:u_total_size], v_face[0:v_total_size], \
                     dudx_cell[0:cell_total_size], dudy_cell[0:cell_total_size], \
                     dvdx_cell[0:cell_total_size], dvdy_cell[0:cell_total_size])
    for (int jj = 0; jj < Ny; ++jj) {
        for (int ii = 0; ii < Nx; ++ii) {
#else
    // CPU path: same logic without GPU offloading
    (void)u_total_size; (void)v_total_size; (void)cell_total_size;
    for (int jj = 0; jj < Ny; ++jj) {
        for (int ii = 0; ii < Nx; ++ii) {
#endif
            // Interior cell indices (i,j) in [Ng, Ng+N-1]
            const int i = ii + Ng;
            const int j = jj + Ng;

            // Cell-centered output index
            const int idx_cell = j * cell_stride + i;

            // For gradients at cell (i,j), need neighboring face values
            // dudx: central difference of u at x-faces
            //   u(i+1,j) - u(i-1,j) gives u_face at i+1 and i-1
            const int u_idx_ip = j * u_stride + (i + 1);
            const int u_idx_im = j * u_stride + (i - 1);
            const int u_idx_jp = (j + 1) * u_stride + i;
            const int u_idx_jm = (j - 1) * u_stride + i;

            // v at y-faces
            const int v_idx_ip = j * v_stride + (i + 1);
            const int v_idx_im = j * v_stride + (i - 1);
            const int v_idx_jp = (j + 1) * v_stride + i;
            const int v_idx_jm = (j - 1) * v_stride + i;

            // Central differences
            dudx_cell[idx_cell] = (u_face[u_idx_ip] - u_face[u_idx_im]) * inv_2dx;
            dudy_cell[idx_cell] = (u_face[u_idx_jp] - u_face[u_idx_jm]) * inv_2dy;
            dvdx_cell[idx_cell] = (v_face[v_idx_ip] - v_face[v_idx_im]) * inv_2dx;
            dvdy_cell[idx_cell] = (v_face[v_idx_jp] - v_face[v_idx_jm]) * inv_2dy;
        }
    }
}

// ============================================================================
// GPU Kernel: Compute scalar MLP features
// ============================================================================
void compute_mlp_scalar_features_gpu(
    const double* dudx, const double* dudy,
    const double* dvdx, const double* dvdy,
    const double* k, const double* omega,
    const double* wall_distance,
    const double* u_face, const double* v_face,
    double* features,
    int Nx, int Ny, int Ng,
    int cell_stride, int u_stride, int v_stride,
    int total_cells, int u_total, int v_total,
    double nu, double delta, double u_ref)
{
    NVTX_SCOPE_TURB("kernel:mlp_features");

#ifdef USE_GPU_OFFLOAD
    // CRITICAL: map(present:...) indicates these arrays are already mapped by caller
    #pragma omp target teams distribute parallel for collapse(2) \
        map(present: dudx[0:total_cells], dudy[0:total_cells], \
                     dvdx[0:total_cells], dvdy[0:total_cells], \
                     k[0:total_cells], omega[0:total_cells], \
                     wall_distance[0:total_cells], \
                     u_face[0:u_total], v_face[0:v_total], \
                     features[0:(Nx*Ny*6)])
    for (int jj = 0; jj < Ny; ++jj) {
        for (int ii = 0; ii < Nx; ++ii) {
            const int i = ii + Ng;
            const int j = jj + Ng;
            const int idx_cell = j * cell_stride + i;
            const int idx_out = jj * Nx + ii;  // Interior cell index for output
            
            // Get gradients
            double dudx_v = dudx[idx_cell];
            double dudy_v = dudy[idx_cell];
            double dvdx_v = dvdx[idx_cell];
            double dvdy_v = dvdy[idx_cell];
            
            // Strain and rotation magnitudes
            double Sxx = dudx_v;
            double Syy = dvdy_v;
            double Sxy = 0.5 * (dudy_v + dvdx_v);
            double Oxy = 0.5 * (dudy_v - dvdx_v);
            
            // Frobenius norm (matches CPU VelocityGradient::S_mag())
            double S_mag = sqrt(Sxx*Sxx + Syy*Syy + 2.0*Sxy*Sxy);
            double Omega_mag = sqrt(2.0 * Oxy * Oxy);

            // Note: k and omega are mapped but not used in current feature set
            // (reserved for future features like turbulent time scale)

            // Velocity magnitude (from staggered grid)
            double u_avg = 0.5 * (u_face[j * u_stride + i] + u_face[j * u_stride + (i+1)]);
            double v_avg = 0.5 * (v_face[j * v_stride + i] + v_face[(j+1) * v_stride + i]);
            double u_mag = sqrt(u_avg*u_avg + v_avg*v_avg);
            
            // Wall distance (cell-centered; same indexing space as gradients)
            double y_wall = wall_distance[idx_cell];
            
            // Features (6 values):
            // 0: Normalized strain rate magnitude
            // 1: Normalized rotation rate magnitude  
            // 2: Normalized wall distance
            // 3: Strain-rotation ratio
            // 4: Local Reynolds number
            // 5: Normalized velocity magnitude
            int feat_base = idx_out * 6;
            features[feat_base + 0] = S_mag * delta / (u_ref + 1e-10);
            features[feat_base + 1] = Omega_mag * delta / (u_ref + 1e-10);
            features[feat_base + 2] = y_wall / delta;
            features[feat_base + 3] = Omega_mag / (S_mag + 1e-10);
            features[feat_base + 4] = S_mag * delta * delta / (nu + 1e-10);
            features[feat_base + 5] = u_mag / (u_ref + 1e-10);
        }
    }
#else
    (void)dudx; (void)dudy; (void)dvdx; (void)dvdy;
    (void)k; (void)omega; (void)wall_distance;
    (void)u_face; (void)v_face; (void)features;
    (void)Nx; (void)Ny; (void)Ng;
    (void)cell_stride; (void)u_stride; (void)v_stride;
    (void)total_cells; (void)u_total; (void)v_total;
    (void)nu; (void)delta; (void)u_ref;
#endif
}

// ============================================================================
// GPU Kernel: Postprocess MLP outputs to ghosted field
// ============================================================================
void postprocess_mlp_outputs_gpu(
    const double* nn_outputs,
    double* nu_t_field,
    int Nx, int Ny, int Ng,
    int stride,
    double nu_t_max)
{
    NVTX_SCOPE_NN("kernel:postprocess_mlp");

#ifdef USE_GPU_OFFLOAD
    const int total_field_size = stride * (Ny + 2*Ng);
    
    // CRITICAL: map(present:...) indicates these arrays are already mapped
    #pragma omp target teams distribute parallel for collapse(2) \
        map(present: nn_outputs[0:(Nx*Ny)], nu_t_field[0:total_field_size])
    for (int jj = 0; jj < Ny; ++jj) {
        for (int ii = 0; ii < Nx; ++ii) {
            const int i = ii + Ng;
            const int j = jj + Ng;
            const int idx_in = jj * Nx + ii;
            const int idx_out = j * stride + i;
            
            // Get NN prediction
            double nu_t_val = nn_outputs[idx_in];
            
            // Apply realizability and clipping
            if (nu_t_val != nu_t_val || nu_t_val < 0.0) {  // NaN or negative
                nu_t_val = 0.0;
            }
            if (nu_t_val > nu_t_max) {
                nu_t_val = nu_t_max;
            }
            
            nu_t_field[idx_out] = nu_t_val;
        }
    }
#else
    (void)nn_outputs; (void)nu_t_field;
    (void)Nx; (void)Ny; (void)Ng; (void)stride; (void)nu_t_max;
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
    int Nx, int Ny, int Ng,
    int cell_stride, int total_cells,
    double nu, double delta)
{
    NVTX_SCOPE_TURB("kernel:tbnn_features");

#ifdef USE_GPU_OFFLOAD
    const double C_mu = 0.09;
    
    // CRITICAL: map(present:...) indicates these arrays are already mapped by solver/turbulence model
    #pragma omp target teams distribute parallel for collapse(2) \
        map(present: dudx[0:total_cells], dudy[0:total_cells], \
                     dvdx[0:total_cells], dvdy[0:total_cells], \
                     k[0:total_cells], omega[0:total_cells], \
                     wall_distance[0:total_cells], \
                     features[0:(Nx*Ny*5)], basis[0:(Nx*Ny*12)])
    for (int jj = 0; jj < Ny; ++jj) {
        for (int ii = 0; ii < Nx; ++ii) {
            const int i = ii + Ng;
            const int j = jj + Ng;
            const int idx_cell = j * cell_stride + i;
            const int idx_out = jj * Nx + ii;  // Interior cell index for output
            
            // Get gradients
            double dudx_v = dudx[idx_cell];
            double dudy_v = dudy[idx_cell];
            double dvdx_v = dvdx[idx_cell];
            double dvdy_v = dvdy[idx_cell];
        
        // Strain rate tensor components: S_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
        double Sxx = dudx_v;
        double Syy = dvdy_v;
        double Sxy = 0.5 * (dudy_v + dvdx_v);
        
        // Rotation tensor: Omega_ij = 0.5 * (du_i/dx_j - du_j/dx_i)
        double Oxy = 0.5 * (dudy_v - dvdx_v);
        
        // Magnitudes (Frobenius norm - matches CPU VelocityGradient)
        double S_mag = sqrt(Sxx*Sxx + Syy*Syy + 2.0*Sxy*Sxy);
        double Omega_mag = sqrt(2.0 * Oxy * Oxy);
        
        // Get k, omega, epsilon
        double k_val = k[idx_cell];
        double omega_val = omega[idx_cell];
        double eps = C_mu * k_val * omega_val;
        
        // Safe values
        double k_safe = (k_val > numerics::K_FLOOR) ? k_val : numerics::K_FLOOR;
        double eps_safe = (eps > numerics::EPS_FLOOR) ? eps : numerics::EPS_FLOOR;
        
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
        int feat_base = idx_out * 5;
        features[feat_base + 0] = S_norm * S_norm;           // ~tr(S_norm^2)
        features[feat_base + 1] = Omega_norm * Omega_norm;   // ~tr(Omega_norm^2)
        features[feat_base + 2] = Sxx_n*Sxx_n + Syy_n*Syy_n + 2.0*Sxy_n*Sxy_n;  // tr(S^2)
        features[feat_base + 3] = 2.0 * Oxy_n * Oxy_n;       // tr(Omega^2)
        features[feat_base + 4] = wall_distance[idx_cell] / delta; // Normalized wall distance
        
        // ==================== Tensor Basis (4 tensors × 3 components) ====================
        int basis_base = idx_out * 12;
        
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
    }
    
    (void)nu;  // Not used in current implementation
#else
    (void)dudx; (void)dudy; (void)dvdx; (void)dvdy;
    (void)k; (void)omega; (void)wall_distance;
    (void)features; (void)basis;
    (void)Nx; (void)Ny; (void)Ng;
    (void)cell_stride; (void)total_cells;
    (void)nu; (void)delta;
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
    int Nx, int Ny, int Ng,
    int cell_stride, int total_cells,
    int output_dim,
    double nu_ref)
{
    NVTX_SCOPE_NN("kernel:postprocess_nn");

#ifdef USE_GPU_OFFLOAD
    const int NUM_BASIS = 4;
    const bool compute_tau = (tau_xx != nullptr);
    
    // CRITICAL: map(present:...) indicates these arrays are already mapped by turbulence model
    // nn_outputs/basis are interior-only (Nx*Ny), k/gradients/nu_t/tau are ghosted (total_cells)
    if (!compute_tau) {
        #pragma omp target teams distribute parallel for collapse(2) \
            map(present: nn_outputs[0:(Nx*Ny*output_dim)], basis[0:(Nx*Ny*12)], \
                         k[0:total_cells], dudx[0:total_cells], dudy[0:total_cells], \
                         dvdx[0:total_cells], dvdy[0:total_cells], nu_t[0:total_cells])
        for (int jj = 0; jj < Ny; ++jj) {
            for (int ii = 0; ii < Nx; ++ii) {
                const int i = ii + Ng;
                const int j = jj + Ng;
                const int idx_cell = j * cell_stride + i;   // ghosted field index
                const int idx_out  = jj * Nx + ii;          // interior index for nn/basis
                
                // Extract G coefficients from NN output
                double G[NUM_BASIS] = {0.0, 0.0, 0.0, 0.0};
                const int out_base = idx_out * output_dim;
                for (int n = 0; n < NUM_BASIS && n < output_dim; ++n) {
                    G[n] = nn_outputs[out_base + n];
                }
                
                // Get basis tensors
                const int basis_base = idx_out * 12;
                
                // Construct anisotropy tensor: b_ij = sum_n G_n * T^n_ij
                double b_xx = 0.0, b_xy = 0.0, b_yy = 0.0;
                for (int n = 0; n < NUM_BASIS; ++n) {
                    b_xx += G[n] * basis[basis_base + n*3 + 0];
                    b_xy += G[n] * basis[basis_base + n*3 + 1];
                    b_yy += G[n] * basis[basis_base + n*3 + 2];
                }
                
                // Compute equivalent eddy viscosity
                const double k_val = k[idx_cell];
                const double dudy_v = dudy[idx_cell];
                const double dvdx_v = dvdx[idx_cell];
                const double Sxy = 0.5 * (dudy_v + dvdx_v);
                
                double nu_t_val = 0.0;
                if (fabs(Sxy) > 1e-10) {
                    nu_t_val = fabs(-b_xy * k_val / Sxy);
                } else {
                    // Fallback: use strain magnitude (Frobenius norm)
                    const double dudx_v = dudx[idx_cell];
                    const double dvdy_v = dvdy[idx_cell];
                    const double Sxx = dudx_v;
                    const double Syy = dvdy_v;
                    const double S_mag = sqrt(Sxx*Sxx + Syy*Syy + 2.0*Sxy*Sxy);
                    if (S_mag > 1e-10) {
                        const double b_mag = sqrt(b_xx*b_xx + 2.0*b_xy*b_xy + b_yy*b_yy);
                        nu_t_val = k_val * b_mag / S_mag;
                    }
                }
                
                // Clip to reasonable bounds
                const double max_nu_t = 10.0 * nu_ref;
                nu_t_val = (nu_t_val > 0.0) ? nu_t_val : 0.0;
                nu_t_val = (nu_t_val < max_nu_t) ? nu_t_val : max_nu_t;
                
                // Check for NaN/Inf
                if (nu_t_val != nu_t_val || nu_t_val > 1e30) {
                    nu_t_val = 0.0;
                }
                
                nu_t[idx_cell] = nu_t_val;
            }
        }
    } else {
        #pragma omp target teams distribute parallel for collapse(2) \
            map(present: nn_outputs[0:(Nx*Ny*output_dim)], basis[0:(Nx*Ny*12)], \
                         k[0:total_cells], dudx[0:total_cells], dudy[0:total_cells], \
                         dvdx[0:total_cells], dvdy[0:total_cells], nu_t[0:total_cells], \
                         tau_xx[0:total_cells], tau_xy[0:total_cells], tau_yy[0:total_cells])
        for (int jj = 0; jj < Ny; ++jj) {
            for (int ii = 0; ii < Nx; ++ii) {
                const int i = ii + Ng;
                const int j = jj + Ng;
                const int idx_cell = j * cell_stride + i;
                const int idx_out  = jj * Nx + ii;
                
                // Extract G coefficients from NN output
                double G[NUM_BASIS] = {0.0, 0.0, 0.0, 0.0};
                const int out_base = idx_out * output_dim;
                for (int n = 0; n < NUM_BASIS && n < output_dim; ++n) {
                    G[n] = nn_outputs[out_base + n];
                }
                
                // Get basis tensors
                const int basis_base = idx_out * 12;
                
                // Construct anisotropy tensor: b_ij = sum_n G_n * T^n_ij
                double b_xx = 0.0, b_xy = 0.0, b_yy = 0.0;
                for (int n = 0; n < NUM_BASIS; ++n) {
                    b_xx += G[n] * basis[basis_base + n*3 + 0];
                    b_xy += G[n] * basis[basis_base + n*3 + 1];
                    b_yy += G[n] * basis[basis_base + n*3 + 2];
                }
                
                // Compute Reynolds stresses
                const double k_val = k[idx_cell];
                const double k_safe = (k_val > 0.0) ? k_val : 0.0;
                tau_xx[idx_cell] = 2.0 * k_safe * (b_xx + 1.0/3.0);
                tau_xy[idx_cell] = 2.0 * k_safe * b_xy;
                tau_yy[idx_cell] = 2.0 * k_safe * (b_yy + 1.0/3.0);
                
                // Compute equivalent eddy viscosity
                const double dudy_v = dudy[idx_cell];
                const double dvdx_v = dvdx[idx_cell];
                const double Sxy = 0.5 * (dudy_v + dvdx_v);
                
                double nu_t_val = 0.0;
                if (fabs(Sxy) > 1e-10) {
                    nu_t_val = fabs(-b_xy * k_val / Sxy);
                } else {
                    // Fallback: use strain magnitude (Frobenius norm)
                    const double dudx_v = dudx[idx_cell];
                    const double dvdy_v = dvdy[idx_cell];
                    const double Sxx = dudx_v;
                    const double Syy = dvdy_v;
                    const double S_mag = sqrt(Sxx*Sxx + Syy*Syy + 2.0*Sxy*Sxy);
                    if (S_mag > 1e-10) {
                        const double b_mag = sqrt(b_xx*b_xx + 2.0*b_xy*b_xy + b_yy*b_yy);
                        nu_t_val = k_val * b_mag / S_mag;
                    }
                }
                
                // Clip to reasonable bounds
                const double max_nu_t = 10.0 * nu_ref;
                nu_t_val = (nu_t_val > 0.0) ? nu_t_val : 0.0;
                nu_t_val = (nu_t_val < max_nu_t) ? nu_t_val : max_nu_t;
                
                // Check for NaN/Inf
                if (nu_t_val != nu_t_val || nu_t_val > 1e30) {
                    nu_t_val = 0.0;
                }
                
                nu_t[idx_cell] = nu_t_val;
            }
        }
    }
#else
    (void)nn_outputs; (void)basis; (void)k;
    (void)dudx; (void)dudy; (void)dvdx; (void)dvdy;
    (void)nu_t; (void)tau_xx; (void)tau_xy; (void)tau_yy;
    (void)Nx; (void)Ny; (void)Ng; (void)cell_stride; (void)total_cells;
    (void)output_dim; (void)nu_ref;
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
    NVTX_SCOPE_NN("kernel:tbnn_full_pipeline");

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

        // Frobenius norm (matches features.hpp::VelocityGradient::S_mag for ML invariants)
        double S_mag = sqrt(Sxx*Sxx + Syy*Syy + 2.0*Sxy*Sxy);
        double Omega_mag = sqrt(2.0 * Oxy * Oxy);
        
        double k_val = k[cell_idx];
        double omega_val = omega[cell_idx];
        double eps = C_mu * k_val * omega_val;
        
        double k_safe = (k_val > numerics::K_FLOOR) ? k_val : numerics::K_FLOOR;
        double eps_safe = (eps > numerics::EPS_FLOOR) ? eps : numerics::EPS_FLOOR;
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

// ============================================================================
// GPU Kernel: Boussinesq k-omega closure
// ============================================================================
void compute_boussinesq_closure_gpu(
    const double* k,
    const double* omega,
    double* nu_t,
    int Nx, int Ny,
    int Ng,
    int stride,
    int total_size,
    double nu,
    double k_min, double omega_min,
    double nu_t_max)
{
    NVTX_SCOPE_CLOSURE("kernel:boussinesq_closure");

#ifdef USE_GPU_OFFLOAD
    const int n_cells = Nx * Ny;
    
    // CRITICAL: map(present:...) for solver-managed device buffers
    #pragma omp target teams distribute parallel for \
        map(present: k[0:total_size], omega[0:total_size], nu_t[0:total_size])
    for (int idx = 0; idx < n_cells; ++idx) {
        // Convert flat index to (i,j) including ghost cells
        const int i = idx % Nx + Ng;
        const int j = idx / Nx + Ng;
        const int cell_idx = j * stride + i;
        
        // Read k and omega
        double k_val = k[cell_idx];
        double omega_val = omega[cell_idx];
        
        // Clamp to minimum values
        k_val = (k_val > k_min) ? k_val : k_min;
        omega_val = (omega_val > omega_min) ? omega_val : omega_min;
        
        // Boussinesq closure: ν_t = k / ω
        double nu_t_val = k_val / omega_val;
        
        // Realizability constraint: ν_t ≥ 0
        nu_t_val = (nu_t_val > 0.0) ? nu_t_val : 0.0;
        
        // Upper limit: ν_t ≤ nu_t_max * ν
        double max_val = nu_t_max * nu;
        nu_t_val = (nu_t_val < max_val) ? nu_t_val : max_val;
        
        nu_t[cell_idx] = nu_t_val;
    }
#else
    (void)k; (void)omega; (void)nu_t;
    (void)Nx; (void)Ny; (void)Ng; (void)stride; (void)total_size;
    (void)nu; (void)k_min; (void)omega_min; (void)nu_t_max;
#endif
}

// ============================================================================
// GPU Kernel: SST k-omega closure
// ============================================================================
void compute_sst_closure_gpu(
    const double* k,
    const double* omega,
    const double* dudx,
    const double* dudy,
    const double* dvdx,
    const double* dvdy,
    const double* wall_distance,
    double* nu_t,
    int Nx, int Ny,
    int Ng,
    int stride,
    int total_size,
    int wall_dist_size,
    double nu,
    double a1,
    double beta_star,
    double k_min, double omega_min,
    double nu_t_max)
{
    NVTX_SCOPE_CLOSURE("kernel:sst_closure");

#ifdef USE_GPU_OFFLOAD
    const int n_cells = Nx * Ny;
    
    // CRITICAL: map(present:...) for solver-managed device buffers
    // wall_distance has same layout as k/omega/nu_t (full field with ghosts)
    #pragma omp target teams distribute parallel for \
        map(present: k[0:total_size], omega[0:total_size], \
                     dudx[0:total_size], dudy[0:total_size], \
                     dvdx[0:total_size], dvdy[0:total_size], \
                     wall_distance[0:wall_dist_size], \
                     nu_t[0:total_size])
    for (int idx = 0; idx < n_cells; ++idx) {
        // Convert flat index to (i,j) including ghost cells
        const int i = idx % Nx + Ng;
        const int j = idx / Nx + Ng;
        const int cell_idx = j * stride + i;
        
        // Read fields (all use stride-based indexing with ghosts)
        double k_val = k[cell_idx];
        double omega_val = omega[cell_idx];
        double y_wall = wall_distance[cell_idx];  // Wall distance uses same indexing as k/omega
        
        // Clamp to minimum values
        k_val = (k_val > k_min) ? k_val : k_min;
        omega_val = (omega_val > omega_min) ? omega_val : omega_min;
        double y_safe = (y_wall > 1e-10) ? y_wall : 1e-10;
        
        // Strain rate magnitude from gradients
        double Sxx = dudx[cell_idx];
        double Syy = dvdy[cell_idx];
        double Sxy = 0.5 * (dudy[cell_idx] + dvdx[cell_idx]);
        double S_mag = sqrt(2.0 * (Sxx*Sxx + Syy*Syy + 2.0*Sxy*Sxy));
        
        // F2 blending function for SST
        // arg2 = max(2√k / (β*ωy), 500ν / (y²ω))
        double sqrt_k = sqrt(k_val);
        double term1 = 2.0 * sqrt_k / (beta_star * omega_val * y_safe);
        double term2 = 500.0 * nu / (y_safe * y_safe * omega_val);
        double arg2 = (term1 > term2) ? term1 : term2;
        double F2 = tanh(arg2 * arg2);
        
        // SST eddy viscosity: ν_t = a₁k / max(a₁ω, SF₂)
        double denom = a1 * omega_val;
        double SF2 = S_mag * F2;
        denom = (denom > SF2) ? denom : SF2;
        
        // Prevent division by zero
        denom = (denom > 1e-20) ? denom : 1e-20;
        
        double nu_t_val = a1 * k_val / denom;
        
        // Realizability: ν_t ≥ 0
        nu_t_val = (nu_t_val > 0.0) ? nu_t_val : 0.0;
        
        // Upper limit: ν_t ≤ nu_t_max * ν
        double max_val = nu_t_max * nu;
        nu_t_val = (nu_t_val < max_val) ? nu_t_val : max_val;
        
        nu_t[cell_idx] = nu_t_val;
    }
#else
    (void)k; (void)omega; (void)dudx; (void)dudy; (void)dvdx; (void)dvdy;
    (void)wall_distance; (void)nu_t;
    (void)Nx; (void)Ny; (void)Ng; (void)stride; (void)total_size; (void)wall_dist_size;
    (void)nu; (void)a1; (void)beta_star;
    (void)k_min; (void)omega_min; (void)nu_t_max;
#endif
}

// ============================================================================
// GPU Kernel: k-omega transport step
// ============================================================================
void komega_transport_step_gpu(
    const double* u, const double* v,
    double* k, double* omega,
    const double* nu_t_prev,
    int Nx, int Ny, int Ng,
    int stride,
    int u_stride, int v_stride,
    int total_size,
    int vel_u_size, int vel_v_size,
    double dx, double dy, double dt,
    double nu, double sigma_k, double sigma_omega,
    double beta, double beta_star, double alpha,
    double k_min, double k_max,
    double omega_min, double omega_max)
{
    NVTX_SCOPE_TURB("kernel:komega_transport");

#ifdef USE_GPU_OFFLOAD
    const int n_cells = Nx * Ny;
    const double inv_2dx = 1.0 / (2.0 * dx);
    const double inv_2dy = 1.0 / (2.0 * dy);
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    
    // CRITICAL: map(present:...) for all solver-managed device buffers
    #pragma omp target teams distribute parallel for \
        map(present: u[0:vel_u_size], v[0:vel_v_size], \
                     k[0:total_size], omega[0:total_size], \
                     nu_t_prev[0:total_size])
    for (int idx = 0; idx < n_cells; ++idx) {
        // Convert flat index to (i,j) including ghost cells
        const int i = idx % Nx + Ng;
        const int j = idx / Nx + Ng;
        const int cell_idx = j * stride + i;
        
        // Read current values (nu_t_prev now uses same ghost+stride layout as k/omega)
        double k_val = k[cell_idx];
        double omega_val = omega[cell_idx];
        double nu_t_val = nu_t_prev[cell_idx];  // Now uses ghost+stride indexing
        
        // Clamp to valid range
        k_val = (k_val > k_min) ? k_val : k_min;
        omega_val = (omega_val > omega_min) ? omega_val : omega_min;
        nu_t_val = (nu_t_val > 0.0) ? nu_t_val : 0.0;
        
        // Compute velocity gradients from staggered MAC grid
        // u is at x-faces, v is at y-faces
        const int u_idx_ip = j * u_stride + (i + 1);
        const int u_idx_im = j * u_stride + (i - 1);
        const int u_idx_jp = (j + 1) * u_stride + i;
        const int u_idx_jm = (j - 1) * u_stride + i;
        
        const int v_idx_ip = j * v_stride + (i + 1);
        const int v_idx_im = j * v_stride + (i - 1);
        const int v_idx_jp = (j + 1) * v_stride + i;
        const int v_idx_jm = (j - 1) * v_stride + i;
        
        double dudx_val = (u[u_idx_ip] - u[u_idx_im]) * inv_2dx;
        double dudy_val = (u[u_idx_jp] - u[u_idx_jm]) * inv_2dy;
        double dvdx_val = (v[v_idx_ip] - v[v_idx_im]) * inv_2dx;
        double dvdy_val = (v[v_idx_jp] - v[v_idx_jm]) * inv_2dy;
        
        // Strain rate magnitude
        double Sxx = dudx_val;
        double Syy = dvdy_val;
        double Sxy = 0.5 * (dudy_val + dvdx_val);
        double S2 = 2.0 * (Sxx*Sxx + Syy*Syy + 2.0*Sxy*Sxy);
        
        // Production term: P_k = 2 ν_t |S|²
        double P_k = 2.0 * nu_t_val * S2;
        
        // Effective diffusivities
        double nu_k = nu + sigma_k * nu_t_val;
        double nu_omega_eff = nu + sigma_omega * nu_t_val;
        
        // Get velocity at cell center (approximate from faces)
        double u_c = 0.5 * (u[j * u_stride + i] + u[j * u_stride + (i+1)]);
        double v_c = 0.5 * (v[j * v_stride + i] + v[(j+1) * v_stride + i]);
        
        // Advection terms (upwind scheme)
        double adv_k, adv_omega;
        const int k_idx_ip = j * stride + (i + 1);
        const int k_idx_im = j * stride + (i - 1);
        const int k_idx_jp = (j + 1) * stride + i;
        const int k_idx_jm = (j - 1) * stride + i;
        
        if (u_c >= 0.0) {
            adv_k = u_c * (k_val - k[k_idx_im]) / dx;
            adv_omega = u_c * (omega_val - omega[k_idx_im]) / dx;
        } else {
            adv_k = u_c * (k[k_idx_ip] - k_val) / dx;
            adv_omega = u_c * (omega[k_idx_ip] - omega_val) / dx;
        }
        
        if (v_c >= 0.0) {
            adv_k += v_c * (k_val - k[k_idx_jm]) / dy;
            adv_omega += v_c * (omega_val - omega[k_idx_jm]) / dy;
        } else {
            adv_k += v_c * (k[k_idx_jp] - k_val) / dy;
            adv_omega += v_c * (omega[k_idx_jp] - omega_val) / dy;
        }
        
        // Diffusion terms (central differences)
        double diff_k = nu_k * ((k[k_idx_ip] - 2.0*k_val + k[k_idx_im]) * inv_dx2 +
                                (k[k_idx_jp] - 2.0*k_val + k[k_idx_jm]) * inv_dy2);
        
        double diff_omega = nu_omega_eff * ((omega[k_idx_ip] - 2.0*omega_val + omega[k_idx_im]) * inv_dx2 +
                                            (omega[k_idx_jp] - 2.0*omega_val + omega[k_idx_jm]) * inv_dy2);
        
        // Source/sink terms
        // k equation: ∂k/∂t = P_k - β*kω + diff - adv
        double rhs_k = P_k - beta_star * k_val * omega_val + diff_k - adv_k;
        
        // ω equation: ∂ω/∂t = α(ω/k)P_k - βω² + diff - adv
        double rhs_omega = alpha * (omega_val / k_val) * P_k - beta * omega_val * omega_val + diff_omega - adv_omega;
        
        // Explicit Euler time integration
        double k_new = k_val + dt * rhs_k;
        double omega_new = omega_val + dt * rhs_omega;
        
        // Clip to valid range
        k_new = (k_new > k_min) ? k_new : k_min;
        k_new = (k_new < k_max) ? k_new : k_max;
        omega_new = (omega_new > omega_min) ? omega_new : omega_min;
        omega_new = (omega_new < omega_max) ? omega_new : omega_max;
        
        // Write back
        k[cell_idx] = k_new;
        omega[cell_idx] = omega_new;
    }
#else
    (void)u; (void)v; (void)k; (void)omega; (void)nu_t_prev;
    (void)Nx; (void)Ny; (void)Ng; (void)stride;
    (void)u_stride; (void)v_stride;
    (void)total_size; (void)vel_u_size; (void)vel_v_size;
    (void)dx; (void)dy; (void)dt;
    (void)nu; (void)sigma_k; (void)sigma_omega;
    (void)beta; (void)beta_star; (void)alpha;
    (void)k_min; (void)k_max; (void)omega_min; (void)omega_max;
#endif
}

} // namespace gpu_kernels
} // namespace nncfd

