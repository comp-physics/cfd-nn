#include "turbulence_nn_tbnn.hpp"
#include "gpu_kernels.hpp"
#include "timing.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

namespace nncfd {

TurbulenceNNTBNN::TurbulenceNNTBNN()
    : feature_computer_(Mesh()) {}

TurbulenceNNTBNN::~TurbulenceNNTBNN() {
    free_full_gpu_buffers();
    free_gpu_buffers();
}

void TurbulenceNNTBNN::load(const std::string& weights_dir, const std::string& scaling_dir) {
    mlp_.load_weights(weights_dir);
    
    try {
        mlp_.load_scaling(scaling_dir + "/input_means.txt",
                         scaling_dir + "/input_stds.txt");
    } catch (const std::exception& e) {
        // Scaling files optional
    }
}

void TurbulenceNNTBNN::upload_to_gpu() {
#ifdef USE_GPU_OFFLOAD
    int num_devices = omp_get_num_devices();

    if (num_devices == 0) {
        throw std::runtime_error(
            "GPU build (USE_GPU_OFFLOAD=ON) requires GPU device at runtime.\n"
            "Found 0 devices. Either run on GPU-enabled node or rebuild with USE_GPU_OFFLOAD=OFF."
        );
    }
    
    mlp_.upload_to_gpu();
    gpu_ready_ = mlp_.is_on_gpu();
    full_gpu_ready_ = gpu_ready_;  // Full pipeline also ready
#endif
}

void TurbulenceNNTBNN::initialize_gpu_buffers(const Mesh& mesh) {
#ifdef USE_GPU_OFFLOAD
    if (omp_get_num_devices() == 0) {
        gpu_ready_ = false;
        full_gpu_ready_ = false;
        return;
    }
    
    const int n_cells = mesh.Nx * mesh.Ny;
    upload_to_gpu();  // Upload MLP weights if not already done
    allocate_gpu_buffers(n_cells);
    allocate_full_gpu_buffers(mesh);
#else
    (void)mesh;
    gpu_ready_ = false;
    full_gpu_ready_ = false;
#endif
}

void TurbulenceNNTBNN::cleanup_gpu_buffers() {
    free_full_gpu_buffers();
    free_gpu_buffers();
    gpu_ready_ = false;
    full_gpu_ready_ = false;
}

void TurbulenceNNTBNN::allocate_gpu_buffers(int n_cells) {
#ifdef USE_GPU_OFFLOAD
    if (n_cells == cached_n_cells_ && !features_flat_.empty() && buffers_on_gpu_) {
        return;  // Already allocated and mapped
    }
    
    free_gpu_buffers();
    
    int feature_dim = mlp_.input_dim();
    int output_dim = mlp_.output_dim();
    size_t workspace_size = mlp_.workspace_size(n_cells);
    
    // Allocate CPU buffers
    features_flat_.resize(n_cells * feature_dim);
    outputs_flat_.resize(n_cells * output_dim);
    workspace_.resize(workspace_size);
    
    // Map to GPU only if we have valid data
    if (!features_flat_.empty() && !outputs_flat_.empty() && !workspace_.empty()) {
        double* feat_ptr = features_flat_.data();
        double* out_ptr = outputs_flat_.data();
        double* work_ptr = workspace_.data();
        size_t feat_size = features_flat_.size();
        size_t out_size = outputs_flat_.size();
        size_t work_size = workspace_.size();
        
        #pragma omp target enter data \
            map(alloc: feat_ptr[0:feat_size]) \
            map(alloc: out_ptr[0:out_size]) \
            map(alloc: work_ptr[0:work_size])
        
        buffers_on_gpu_ = true;
    }
    
    cached_n_cells_ = n_cells;
#else
    (void)n_cells;
#endif
}

void TurbulenceNNTBNN::allocate_full_gpu_buffers(const Mesh& mesh) {
#ifdef USE_GPU_OFFLOAD
    int n_interior = mesh.Nx * mesh.Ny;
    int n_total = (mesh.Nx + 2) * (mesh.Ny + 2);
    
    if (n_interior == cached_n_cells_ && n_total == cached_total_cells_ && !u_flat_.empty()) {
        return;  // Already allocated
    }
    
    free_full_gpu_buffers();
    
    // Allocate CPU buffers
    u_flat_.resize(n_total);
    v_flat_.resize(n_total);
    k_flat_.resize(n_interior);
    omega_flat_.resize(n_interior);
    wall_dist_flat_.resize(n_interior);
    nu_t_flat_.resize(n_interior);
    tau_xx_flat_.resize(n_interior);
    tau_xy_flat_.resize(n_interior);
    tau_yy_flat_.resize(n_interior);
    
    // Workspace for full pipeline:
    // - gradients: 4 * n_interior
    // - features: 5 * n_interior
    // - basis: 12 * n_interior
    // - NN workspace: 2 * max_dim * n_interior
    // - NN outputs: output_dim * n_interior
    int max_dim = 64;  // Reasonable upper bound for layer dimensions
    for (int l = 0; l < mlp_.num_layers(); ++l) {
        max_dim = std::max(max_dim, mlp_.layer(l).in_dim);
        max_dim = std::max(max_dim, mlp_.layer(l).out_dim);
    }
    
    size_t workspace_size = n_interior * (4 + 5 + 12 + 2 * max_dim + mlp_.output_dim());
    full_workspace_.resize(workspace_size);
    
    // Map to GPU
    double* u_ptr = u_flat_.data();
    double* v_ptr = v_flat_.data();
    double* k_ptr = k_flat_.data();
    double* omega_ptr = omega_flat_.data();
    double* wall_ptr = wall_dist_flat_.data();
    double* nu_t_ptr = nu_t_flat_.data();
    double* tau_xx_ptr = tau_xx_flat_.data();
    double* tau_xy_ptr = tau_xy_flat_.data();
    double* tau_yy_ptr = tau_yy_flat_.data();
    double* work_ptr = full_workspace_.data();
    
    #pragma omp target enter data \
        map(alloc: u_ptr[0:n_total]) \
        map(alloc: v_ptr[0:n_total]) \
        map(alloc: k_ptr[0:n_interior]) \
        map(alloc: omega_ptr[0:n_interior]) \
        map(alloc: wall_ptr[0:n_interior]) \
        map(alloc: nu_t_ptr[0:n_interior]) \
        map(alloc: tau_xx_ptr[0:n_interior]) \
        map(alloc: tau_xy_ptr[0:n_interior]) \
        map(alloc: tau_yy_ptr[0:n_interior]) \
        map(alloc: work_ptr[0:workspace_size])
    
    full_buffers_on_gpu_ = true;  // Mark buffers as mapped to GPU
    cached_n_cells_ = n_interior;
    cached_total_cells_ = n_total;
#else
    (void)mesh;
#endif
}

void TurbulenceNNTBNN::free_gpu_buffers() {
#ifdef USE_GPU_OFFLOAD
    // Only free GPU buffers if they were actually mapped to GPU
    if (buffers_on_gpu_) {
        // Check vectors are non-empty before unmapping
        if (!features_flat_.empty() && !outputs_flat_.empty() && !workspace_.empty()) {
            // Set flag FIRST to prevent re-entry
            buffers_on_gpu_ = false;
            
            double* feat_ptr = features_flat_.data();
            double* out_ptr = outputs_flat_.data();
            double* work_ptr = workspace_.data();
            size_t feat_size = features_flat_.size();
            size_t out_size = outputs_flat_.size();
            size_t work_size = workspace_.size();
            
            #pragma omp target exit data \
                map(delete: feat_ptr[0:feat_size]) \
                map(delete: out_ptr[0:out_size]) \
                map(delete: work_ptr[0:work_size])
        } else {
            buffers_on_gpu_ = false;  // Clear flag even if vectors are empty
        }
    }
#endif
    features_flat_.clear();
    outputs_flat_.clear();
    workspace_.clear();
}

void TurbulenceNNTBNN::free_full_gpu_buffers() {
#ifdef USE_GPU_OFFLOAD
    // Only free GPU buffers if they were actually mapped to GPU
    if (full_buffers_on_gpu_) {
        // Check vectors are non-empty before unmapping
        if (!u_flat_.empty() && !k_flat_.empty() && !full_workspace_.empty()) {
            // Set flag FIRST to prevent re-entry
            full_buffers_on_gpu_ = false;
            
            double* u_ptr = u_flat_.data();
            double* v_ptr = v_flat_.data();
            double* k_ptr = k_flat_.data();
            double* omega_ptr = omega_flat_.data();
            double* wall_ptr = wall_dist_flat_.data();
            double* nu_t_ptr = nu_t_flat_.data();
            double* tau_xx_ptr = tau_xx_flat_.data();
            double* tau_xy_ptr = tau_xy_flat_.data();
            double* tau_yy_ptr = tau_yy_flat_.data();
            double* work_ptr = full_workspace_.data();
            
            size_t n_total = u_flat_.size();
            size_t n_interior = k_flat_.size();
            size_t work_size = full_workspace_.size();
            
            #pragma omp target exit data \
                map(delete: u_ptr[0:n_total]) \
                map(delete: v_ptr[0:n_total]) \
                map(delete: k_ptr[0:n_interior]) \
                map(delete: omega_ptr[0:n_interior]) \
                map(delete: wall_ptr[0:n_interior]) \
                map(delete: nu_t_ptr[0:n_interior]) \
                map(delete: tau_xx_ptr[0:n_interior]) \
                map(delete: tau_xy_ptr[0:n_interior]) \
                map(delete: tau_yy_ptr[0:n_interior]) \
                map(delete: work_ptr[0:work_size])
        } else {
            full_buffers_on_gpu_ = false;  // Clear flag even if vectors are empty
        }
    }
#endif
    u_flat_.clear();
    v_flat_.clear();
    k_flat_.clear();
    omega_flat_.clear();
    wall_dist_flat_.clear();
    nu_t_flat_.clear();
    tau_xx_flat_.clear();
    tau_xy_flat_.clear();
    tau_yy_flat_.clear();
    full_workspace_.clear();
    cached_total_cells_ = 0;
}

void TurbulenceNNTBNN::ensure_initialized(const Mesh& mesh) {
    if (!initialized_) {
        feature_computer_ = FeatureComputer(mesh);
        feature_computer_.set_reference(nu_, delta_, u_ref_);
        
        int n_interior = mesh.Nx * mesh.Ny;
        features_.resize(n_interior);
        basis_.resize(n_interior);
        
        initialized_ = true;
    }
}

void TurbulenceNNTBNN::estimate_k(const Mesh& mesh, const VectorField& velocity, 
                                  ScalarField& k) {
    const double C_mu = 0.09;
    
    double u_tau = 0.0;
    {
        int j = mesh.j_begin();
        double dudy_avg = 0.0;
        int count = 0;
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            if (j+1 < mesh.j_end() && j-1 >= mesh.j_begin()) {
                double dudy = (velocity.u(i, j+1) - velocity.u(i, j-1)) / (2.0 * mesh.dy);
                dudy_avg += std::abs(dudy);
                ++count;
            }
        }
        if (count > 0) {
            dudy_avg /= count;
            u_tau = std::sqrt(nu_ * dudy_avg);
        }
    }
    
    u_tau = std::max(u_tau, 1e-6);
    
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double y_wall = mesh.wall_distance(i, j);
            double y_plus = y_wall * u_tau / (nu_ + 1e-20);
            
            double f_mu = 1.0 - std::exp(-std::min(y_plus / 26.0, 20.0));
            
            double k_est = (u_tau * u_tau / std::sqrt(C_mu)) * f_mu * f_mu;
            k(i, j) = std::max(k_min_, std::min(k_est, 10.0 * u_tau * u_tau));
        }
    }
}

void TurbulenceNNTBNN::update_full_gpu(
    const Mesh& mesh,
    const VectorField& velocity,
    const ScalarField& k_in,
    const ScalarField& omega_in,
    ScalarField& nu_t,
    TensorField* tau_ij)
{
#ifdef USE_GPU_OFFLOAD
    TIMED_SCOPE("nn_tbnn_full_gpu");
    
    const int Nx = mesh.Nx;
    const int Ny = mesh.Ny;
    const int n_cells = Nx * Ny;
    const int n_total = (Nx + 2) * (Ny + 2);
    const int stride = Nx + 2;
    const double dx = mesh.dx;
    const double dy = mesh.dy;
    const double inv_2dx = 1.0 / (2.0 * dx);
    const double inv_2dy = 1.0 / (2.0 * dy);
    const double C_mu = 0.09;
    const double delta_val = delta_;
    const double nu_val = nu_;
    const double k_min_val = k_min_;
    
    // Allocate GPU buffers if needed
    allocate_full_gpu_buffers(mesh);
    
    // Get NN parameters (these are already on GPU)
    const int n_layers = mlp_.num_layers();
    const int input_dim = mlp_.input_dim();
    const int output_dim_val = mlp_.output_dim();
    const bool has_scaling = mlp_.has_scaling();
    const int scale_size = mlp_.scale_size();
    
    // Get max layer dimension
    int max_dim = 0;
    for (int l = 0; l < n_layers; ++l) {
        max_dim = std::max(max_dim, mlp_.layer(l).in_dim);
        max_dim = std::max(max_dim, mlp_.layer(l).out_dim);
    }
    
    // ========== Step 1: Flatten input data ==========
    {
        TIMED_SCOPE("nn_tbnn_flatten_inputs");
        
        // Flatten velocity (with ghost cells)
        const auto& u_data = velocity.u_data();
        const auto& v_data = velocity.v_data();
        std::copy(u_data.begin(), u_data.end(), u_flat_.begin());
        std::copy(v_data.begin(), v_data.end(), v_flat_.begin());
        
        // Flatten k, omega, wall_distance (interior only)
        int idx = 0;
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                k_flat_[idx] = k_in(i, j);
                omega_flat_[idx] = omega_in(i, j);
                wall_dist_flat_[idx] = mesh.wall_distance(i, j);
                ++idx;
            }
        }
    }
    
    // Get all the pointers we need
    double* u_ptr = u_flat_.data();
    double* v_ptr = v_flat_.data();
    double* k_ptr = k_flat_.data();
    double* omega_ptr = omega_flat_.data();
    double* wall_ptr = wall_dist_flat_.data();
    double* work_ptr = full_workspace_.data();
    double* nu_t_ptr = nu_t_flat_.data();
    double* tau_xx_ptr = tau_ij ? tau_xx_flat_.data() : nullptr;
    double* tau_xy_ptr = tau_ij ? tau_xy_flat_.data() : nullptr;
    double* tau_yy_ptr = tau_ij ? tau_yy_flat_.data() : nullptr;
    
    // NN weights (already mapped to GPU via MLP::upload_to_gpu)
    const double* weights_ptr = mlp_.weights_gpu();
    const double* biases_ptr = mlp_.biases_gpu();
    const int* w_offsets_ptr = mlp_.weight_offsets_gpu();
    const int* b_offsets_ptr = mlp_.bias_offsets_gpu();
    const int* dims_ptr = mlp_.layer_dims_gpu();
    const int* act_ptr = mlp_.activation_types_gpu();
    const double* means_ptr = mlp_.input_means_gpu();
    const double* stds_ptr = mlp_.input_stds_gpu();
    
    const bool compute_tau = (tau_ij != nullptr);
    size_t work_size = full_workspace_.size();
    
    // ========== Step 2: Upload to GPU ==========
    {
        TIMED_SCOPE("nn_tbnn_upload");
        #pragma omp target update to(u_ptr[0:n_total]) \
                                   to(v_ptr[0:n_total]) \
                                   to(k_ptr[0:n_cells]) \
                                   to(omega_ptr[0:n_cells]) \
                                   to(wall_ptr[0:n_cells])
    }
    
    // ========== Step 3: Run full GPU pipeline ==========
    {
        TIMED_SCOPE("nn_tbnn_kernel");
        
        // Run fused GPU kernel
        #pragma omp target teams distribute parallel for
        for (int cell_idx = 0; cell_idx < n_cells; ++cell_idx) {
            // Workspace layout within work_ptr (all already on device)
            const int FEATURE_DIM = 5;
            const int NUM_BASIS = 4;
            
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
            double dudx_v = (u_ptr[idx_ip] - u_ptr[idx_im]) * inv_2dx;
            double dudy_v = (u_ptr[idx_jp] - u_ptr[idx_jm]) * inv_2dy;
            double dvdx_v = (v_ptr[idx_ip] - v_ptr[idx_im]) * inv_2dx;
            double dvdy_v = (v_ptr[idx_jp] - v_ptr[idx_jm]) * inv_2dy;
            
            // ========== Step 2: Compute strain/rotation ==========
            double Sxx = dudx_v;
            double Syy = dvdy_v;
            double Sxy = 0.5 * (dudy_v + dvdx_v);
            double Oxy = 0.5 * (dudy_v - dvdx_v);
            
            double S_mag = sqrt(2.0 * (Sxx*Sxx + Syy*Syy + 2.0*Sxy*Sxy));
            double Omega_mag = sqrt(2.0 * Oxy * Oxy);
            
            double k_val = k_ptr[cell_idx];
            double omega_val = omega_ptr[cell_idx];
            
            // Ensure k and omega are positive and reasonable
            k_val = (k_val > k_min_val) ? k_val : k_min_val;
            k_val = (k_val < 100.0) ? k_val : 100.0;  // Cap at reasonable max
            omega_val = (omega_val > 1e-10) ? omega_val : 1e-10;
            omega_val = (omega_val < 1e6) ? omega_val : 1e6;  // Cap at reasonable max
            
            double eps = C_mu * k_val * omega_val;
            double eps_safe = (eps > 1e-20) ? eps : 1e-20;
            double tau_scale = k_val / eps_safe;
            
            double S_norm = S_mag * tau_scale;
            double Omega_norm = Omega_mag * tau_scale;
            double Sxx_n = Sxx * tau_scale;
            double Syy_n = Syy * tau_scale;
            double Sxy_n = Sxy * tau_scale;
            double Oxy_n = Oxy * tau_scale;
            
            // ========== Step 3: Compute features ==========
            double feat[5];
            feat[0] = S_norm * S_norm;
            feat[1] = Omega_norm * Omega_norm;
            feat[2] = Sxx_n*Sxx_n + Syy_n*Syy_n + 2.0*Sxy_n*Sxy_n;
            feat[3] = 2.0 * Oxy_n * Oxy_n;
            feat[4] = wall_ptr[cell_idx] / delta_val;
            
            // Apply input scaling if available
            if (has_scaling && means_ptr != nullptr && stds_ptr != nullptr) {
                for (int f = 0; f < FEATURE_DIM && f < scale_size; ++f) {
                    feat[f] = (feat[f] - means_ptr[f]) / stds_ptr[f];
                }
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
            
            // T^(4) = 0 in 2D
            T[3][0] = 0.0; T[3][1] = 0.0; T[3][2] = 0.0;
            
            // ========== Step 5: NN Forward Pass ==========
            // Use workspace for ping-pong buffers
            // Skip past gradients (4*n_cells), features (5*n_cells), and basis (12*n_cells)
            double* nn_work_start = work_ptr + n_cells * (4 + 5 + 12);
            double* buf1 = nn_work_start + cell_idx * max_dim * 2;
            double* buf2 = buf1 + max_dim;
            
            // Copy features to buf1
            for (int f = 0; f < FEATURE_DIM; ++f) {
                buf1[f] = feat[f];
            }
            
            double* current = buf1;
            double* next = buf2;
            
            for (int l = 0; l < n_layers; ++l) {
                int in_dim = dims_ptr[l * 2];
                int out_dim_l = dims_ptr[l * 2 + 1];
                int w_off = w_offsets_ptr[l];
                int b_off = b_offsets_ptr[l];
                int act_type = act_ptr[l];
                
                // Matrix-vector multiply: next = W * current + b
                for (int o = 0; o < out_dim_l; ++o) {
                    double sum = biases_ptr[b_off + o];
                    for (int k_idx = 0; k_idx < in_dim; ++k_idx) {
                        sum += weights_ptr[w_off + o * in_dim + k_idx] * current[k_idx];
                    }
                    
                    // Apply activation (Linear=0, ReLU=1, Tanh=2, Sigmoid=3, Swish=4, GELU=5)
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
            
            // ========== Step 6: Construct anisotropy ==========
            double G[4] = {0.0, 0.0, 0.0, 0.0};
            for (int n = 0; n < NUM_BASIS && n < output_dim_val; ++n) {
                G[n] = current[n];
            }
            
            double b_xx = 0.0, b_xy = 0.0, b_yy = 0.0;
            for (int n = 0; n < NUM_BASIS; ++n) {
                b_xx += G[n] * T[n][0];
                b_xy += G[n] * T[n][1];
                b_yy += G[n] * T[n][2];
            }
            
            // ========== Step 7: Compute Reynolds stresses (optional) ==========
            if (compute_tau && tau_xx_ptr != nullptr) {
                double k_safe2 = (k_val > 0.0) ? k_val : 0.0;
                tau_xx_ptr[cell_idx] = 2.0 * k_safe2 * (b_xx + 1.0/3.0);
                tau_xy_ptr[cell_idx] = 2.0 * k_safe2 * b_xy;
                tau_yy_ptr[cell_idx] = 2.0 * k_safe2 * (b_yy + 1.0/3.0);
            }
            
            // ========== Step 8: Compute eddy viscosity ==========
            double nu_t_val = 0.0;
            double abs_Sxy = (Sxy >= 0.0) ? Sxy : -Sxy;
            if (abs_Sxy > 1e-10) {
                double tmp = -b_xy * k_val / Sxy;
                nu_t_val = (tmp >= 0.0) ? tmp : -tmp;
            } else {
                if (S_mag > 1e-10) {
                    double b_mag = sqrt(b_xx*b_xx + 2.0*b_xy*b_xy + b_yy*b_yy);
                    nu_t_val = k_val * b_mag / S_mag;
                }
            }
            
            // Clip and validate
            double max_nu_t = 10.0 * nu_val;
            nu_t_val = (nu_t_val > 0.0) ? nu_t_val : 0.0;
            nu_t_val = (nu_t_val < max_nu_t) ? nu_t_val : max_nu_t;
            if (nu_t_val != nu_t_val || nu_t_val > 1e30) {  // NaN check
                nu_t_val = 0.0;
            }
            
            nu_t_ptr[cell_idx] = nu_t_val;
        }
    }
    
    // ========== Step 4: Download results ==========
    {
        TIMED_SCOPE("nn_tbnn_download");
        #pragma omp target update from(nu_t_ptr[0:n_cells])
        
        if (tau_ij) {
            #pragma omp target update from(tau_xx_ptr[0:n_cells]) \
                                       from(tau_xy_ptr[0:n_cells]) \
                                       from(tau_yy_ptr[0:n_cells])
        }
    }
    
    // ========== Step 5: Unflatten results ==========
    {
        TIMED_SCOPE("nn_tbnn_unflatten");
        
        int idx = 0;
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                nu_t(i, j) = nu_t_flat_[idx];
                if (tau_ij) {
                    tau_ij->xx(i, j) = tau_xx_flat_[idx];
                    tau_ij->xy(i, j) = tau_xy_flat_[idx];
                    tau_ij->yy(i, j) = tau_yy_flat_[idx];
                }
                ++idx;
            }
        }
    }
#else
    (void)mesh; (void)velocity; (void)k_in; (void)omega_in; (void)nu_t; (void)tau_ij;
#endif
}

void TurbulenceNNTBNN::update(
    const Mesh& mesh,
    const VectorField& velocity,
    const ScalarField& k_in,
    const ScalarField& omega_in,
    ScalarField& nu_t,
    TensorField* tau_ij,
    const TurbulenceDeviceView* device_view) {
    (void)device_view;  // Not yet implemented for NN-TBNN
    
    TIMED_SCOPE("nn_tbnn_update");
    
    ensure_initialized(mesh);
    feature_computer_.set_reference(nu_, delta_, u_ref_);
    
    // Use provided k/omega or estimate
    ScalarField k_local(mesh);
    ScalarField omega_local(mesh);
    
    // Check if k is provided (non-zero values)
    bool k_provided = false;
    for (int j = mesh.j_begin(); j < mesh.j_end() && !k_provided; ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end() && !k_provided; ++i) {
            if (k_in(i, j) > k_min_) {
                k_provided = true;
            }
        }
    }
    
    if (k_provided) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                k_local(i, j) = k_in(i, j);
                omega_local(i, j) = omega_in(i, j);
            }
        }
    } else {
        estimate_k(mesh, velocity, k_local);
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double y_wall = mesh.wall_distance(i, j);
                omega_local(i, j) = std::sqrt(k_local(i, j)) / (0.41 * std::max(y_wall, 1e-10));
            }
        }
    }
    
#ifdef USE_GPU_OFFLOAD
    // NOTE: Full GPU pipeline has numerical issues - using proven partial GPU approach
    // TODO: Debug full pipeline workspace/data issues
    // if (full_gpu_ready_) {
    //     update_full_gpu(mesh, velocity, k_local, omega_local, nu_t, tau_ij);
    //     return;
    // }
    
    // GPU path with CPU feature computation (proven stable and fast)
    if (gpu_ready_) {
        // Compute features and tensor basis (CPU)
        {
            TIMED_SCOPE("nn_tbnn_features");
            feature_computer_.compute_tbnn_features(velocity, k_local, omega_local, 
                                                    features_, basis_);
        }
        
        int n_cells = mesh.Nx * mesh.Ny;
        int feature_dim = mlp_.input_dim();
        int output_dim = mlp_.output_dim();
        
        TIMED_SCOPE("nn_tbnn_inference_gpu");
        
        // Ensure GPU buffers are allocated
        allocate_gpu_buffers(n_cells);
        
        // Flatten features for GPU (CPU side)
        {
            TIMED_SCOPE("nn_tbnn_flatten");
            for (int idx = 0; idx < n_cells; ++idx) {
                for (int f = 0; f < feature_dim; ++f) {
                    features_flat_[idx * feature_dim + f] = features_[idx].values[f];
                }
            }
        }
        
        // Upload features to GPU
        {
            TIMED_SCOPE("nn_tbnn_upload");
            double* feat_ptr = features_flat_.data();
            size_t feat_size = features_flat_.size();
            #pragma omp target update to(feat_ptr[0:feat_size])
        }
        
        // Run batched NN inference on GPU
        {
            TIMED_SCOPE("nn_tbnn_kernel");
            double* feat_ptr = features_flat_.data();
            double* out_ptr = outputs_flat_.data();
            double* work_ptr = workspace_.data();
            mlp_.forward_batch_gpu(feat_ptr, out_ptr, n_cells, work_ptr);
        }
        
        // Download outputs from GPU
        {
            TIMED_SCOPE("nn_tbnn_download");
            double* out_ptr = outputs_flat_.data();
            size_t out_size = outputs_flat_.size();
            #pragma omp target update from(out_ptr[0:out_size])
        }
        
        // Post-process: construct anisotropy and compute nu_t (CPU)
        {
            TIMED_SCOPE("nn_tbnn_postprocess");
            int idx = 0;
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    // Extract G coefficients from NN output
                    std::array<double, TensorBasis::NUM_BASIS> G;
                    for (int n = 0; n < TensorBasis::NUM_BASIS; ++n) {
                        G[n] = (n < output_dim) ? outputs_flat_[idx * output_dim + n] : 0.0;
                    }
                    
                    // Construct anisotropy tensor
                    double b_xx, b_xy, b_yy;
                    TensorBasis::construct_anisotropy(G, basis_[idx], b_xx, b_xy, b_yy);
                    
                    // Convert to Reynolds stresses if requested
                    if (tau_ij) {
                        double k_val = k_local(i, j);
                        double tau_xx, tau_xy, tau_yy;
                        TensorBasis::anisotropy_to_reynolds_stress(b_xx, b_xy, b_yy, k_val,
                                                                  tau_xx, tau_xy, tau_yy);
                        tau_ij->xx(i, j) = tau_xx;
                        tau_ij->xy(i, j) = tau_xy;
                        tau_ij->yy(i, j) = tau_yy;
                    }
                    
                    // Compute equivalent eddy viscosity (MAC-aware gradients)
                    const double inv_2dx = 1.0 / (2.0 * mesh.dx);
                    const double inv_2dy = 1.0 / (2.0 * mesh.dy);
                    VelocityGradient grad;
                    grad.dudx = (velocity.u(i + 1, j) - velocity.u(i - 1, j)) * inv_2dx;
                    grad.dudy = (velocity.u(i, j + 1) - velocity.u(i, j - 1)) * inv_2dy;
                    grad.dvdx = (velocity.v(i + 1, j) - velocity.v(i - 1, j)) * inv_2dx;
                    grad.dvdy = (velocity.v(i, j + 1) - velocity.v(i, j - 1)) * inv_2dy;
                    
                    double Sxy = grad.Sxy();
                    double k_val = k_local(i, j);
                    
                    if (std::abs(Sxy) > 1e-10) {
                        nu_t(i, j) = std::abs(-b_xy * k_val / Sxy);
                    } else {
                        double S_mag = grad.S_mag();
                        if (S_mag > 1e-10) {
                            double b_mag = std::sqrt(b_xx*b_xx + 2.0*b_xy*b_xy + b_yy*b_yy);
                            nu_t(i, j) = k_val * b_mag / S_mag;
                        } else {
                            nu_t(i, j) = 0.0;
                        }
                    }
                    
                    // Ensure positivity and clip to reasonable bounds
                    nu_t(i, j) = std::max(0.0, std::min(nu_t(i, j), 10.0 * nu_));
                    
                    if (std::isnan(nu_t(i, j)) || std::isinf(nu_t(i, j))) {
                        nu_t(i, j) = 0.0;
                    }
                    
                    ++idx;
                }
            }
        }
        return;
    }
#endif
    
    // CPU fallback path: sequential inference
    {
        TIMED_SCOPE("nn_tbnn_inference_cpu");
        
        // Compute features and tensor basis
        feature_computer_.compute_tbnn_features(velocity, k_local, omega_local, 
                                                features_, basis_);
        
        std::vector<double> buffer1, buffer2;
        
        int idx = 0;
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                // NN forward pass
                std::vector<double> output = mlp_.forward(features_[idx].values);
                
                // Output should be G coefficients (NUM_BASIS values)
                std::array<double, TensorBasis::NUM_BASIS> G;
                for (int n = 0; n < TensorBasis::NUM_BASIS; ++n) {
                    G[n] = (n < static_cast<int>(output.size())) ? output[n] : 0.0;
                }
                
                // Construct anisotropy tensor
                double b_xx, b_xy, b_yy;
                TensorBasis::construct_anisotropy(G, basis_[idx], b_xx, b_xy, b_yy);
                
                // Convert to Reynolds stresses if requested
                if (tau_ij) {
                    double k_val = k_local(i, j);
                    double tau_xx, tau_xy, tau_yy;
                    TensorBasis::anisotropy_to_reynolds_stress(b_xx, b_xy, b_yy, k_val,
                                                              tau_xx, tau_xy, tau_yy);
                    tau_ij->xx(i, j) = tau_xx;
                    tau_ij->xy(i, j) = tau_xy;
                    tau_ij->yy(i, j) = tau_yy;
                }
                
                // Also compute equivalent eddy viscosity (MAC-aware gradients)
                const double inv_2dx = 1.0 / (2.0 * mesh.dx);
                const double inv_2dy = 1.0 / (2.0 * mesh.dy);
                VelocityGradient grad;
                grad.dudx = (velocity.u(i + 1, j) - velocity.u(i - 1, j)) * inv_2dx;
                grad.dudy = (velocity.u(i, j + 1) - velocity.u(i, j - 1)) * inv_2dy;
                grad.dvdx = (velocity.v(i + 1, j) - velocity.v(i - 1, j)) * inv_2dx;
                grad.dvdy = (velocity.v(i, j + 1) - velocity.v(i, j - 1)) * inv_2dy;
                
                double Sxy = grad.Sxy();
                double k_val = k_local(i, j);
                
                if (std::abs(Sxy) > 1e-10) {
                    nu_t(i, j) = std::abs(-b_xy * k_val / Sxy);
                } else {
                    double S_mag = grad.S_mag();
                    if (S_mag > 1e-10) {
                        double b_mag = std::sqrt(b_xx*b_xx + 2.0*b_xy*b_xy + b_yy*b_yy);
                        nu_t(i, j) = k_val * b_mag / S_mag;
                    } else {
                        nu_t(i, j) = 0.0;
                    }
                }
                
                // Ensure positivity and clip to reasonable bounds
                nu_t(i, j) = std::max(0.0, std::min(nu_t(i, j), 10.0 * nu_));
                
                if (std::isnan(nu_t(i, j)) || std::isinf(nu_t(i, j))) {
                    nu_t(i, j) = 0.0;
                }
                
                ++idx;
            }
        }
    }
}

} // namespace nncfd
