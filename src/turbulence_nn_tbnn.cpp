#include "turbulence_nn_tbnn.hpp"
#include "gpu_kernels.hpp"
#include "timing.hpp"
#include "numerics.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

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

void TurbulenceNNTBNN::sync_weights_to_gpu() {
#ifdef USE_GPU_OFFLOAD
    // MLP sync_weights_to_gpu() will verify device availability and throw if not available
    mlp_.sync_weights_to_gpu();
    gpu_ready_ = mlp_.is_on_gpu();
    full_gpu_ready_ = gpu_ready_;  // Full pipeline also ready
#endif
}

void TurbulenceNNTBNN::initialize_gpu_buffers(const Mesh& mesh) {
#ifdef USE_GPU_OFFLOAD
    // Fail fast if no GPU device available (GPU build requires GPU)
    gpu::verify_device_available();
    
    const int n_cells = mesh.Nx * mesh.Ny * mesh.Nz;
    sync_weights_to_gpu();  // Upload MLP weights if not already done
    allocate_gpu_buffers(n_cells);
    gpu_ready_ = (mlp_.is_on_gpu() && buffers_on_gpu_);  // Set gpu_ready after successful allocation
#else
    (void)mesh;
    gpu_ready_ = false;
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
    basis_flat_.resize(n_cells * TensorBasis::NUM_BASIS * TensorBasis::NUM_COMPONENTS);  // 10 tensors × 6 components
    outputs_flat_.resize(n_cells * output_dim);
    workspace_.resize(workspace_size);
    
    // Map to GPU only if we have valid data
    if (!features_flat_.empty() && !basis_flat_.empty() && !outputs_flat_.empty() && !workspace_.empty()) {
        double* feat_ptr = features_flat_.data();
        double* basis_ptr = basis_flat_.data();
        double* out_ptr = outputs_flat_.data();
        double* work_ptr = workspace_.data();
        size_t feat_size = features_flat_.size();
        size_t basis_size = basis_flat_.size();
        size_t out_size = outputs_flat_.size();
        size_t work_size = workspace_.size();
        
        #pragma omp target enter data \
            map(alloc: feat_ptr[0:feat_size]) \
            map(alloc: basis_ptr[0:basis_size]) \
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
    int n_interior = mesh.Nx * mesh.Ny * mesh.Nz;
    int n_total = mesh.total_cells();
    
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
    tau_xz_flat_.resize(n_interior);
    tau_yy_flat_.resize(n_interior);
    tau_yz_flat_.resize(n_interior);
    tau_zz_flat_.resize(n_interior);

    // Workspace for full pipeline:
    // - gradients: 9 * n_interior
    // - features: 5 * n_interior
    // - basis: 60 * n_interior
    // - NN workspace: 2 * max_dim * n_interior
    // - NN outputs: output_dim * n_interior
    int max_dim = 64;  // Reasonable upper bound for layer dimensions
    for (int l = 0; l < mlp_.num_layers(); ++l) {
        max_dim = std::max(max_dim, mlp_.layer(l).in_dim);
        max_dim = std::max(max_dim, mlp_.layer(l).out_dim);
    }
    
    size_t workspace_size = n_interior * (9 + 5 + 60 + 2 * max_dim + mlp_.output_dim());
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
    double* tau_xz_ptr = tau_xz_flat_.data();
    double* tau_yy_ptr = tau_yy_flat_.data();
    double* tau_yz_ptr = tau_yz_flat_.data();
    double* tau_zz_ptr = tau_zz_flat_.data();
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
        map(alloc: tau_xz_ptr[0:n_interior]) \
        map(alloc: tau_yy_ptr[0:n_interior]) \
        map(alloc: tau_yz_ptr[0:n_interior]) \
        map(alloc: tau_zz_ptr[0:n_interior]) \
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
        if (!features_flat_.empty() && !basis_flat_.empty() && !outputs_flat_.empty() && !workspace_.empty()) {
            // Set flag FIRST to prevent re-entry
            buffers_on_gpu_ = false;
            
            double* feat_ptr = features_flat_.data();
            double* basis_ptr = basis_flat_.data();
            double* out_ptr = outputs_flat_.data();
            double* work_ptr = workspace_.data();
            size_t feat_size = features_flat_.size();
            size_t basis_size = basis_flat_.size();
            size_t out_size = outputs_flat_.size();
            size_t work_size = workspace_.size();
            
            #pragma omp target exit data \
                map(delete: feat_ptr[0:feat_size]) \
                map(delete: basis_ptr[0:basis_size]) \
                map(delete: out_ptr[0:out_size]) \
                map(delete: work_ptr[0:work_size])
        } else {
            buffers_on_gpu_ = false;  // Clear flag even if vectors are empty
        }
    }
#endif
    features_flat_.clear();
    basis_flat_.clear();
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
            double* tau_xz_ptr = tau_xz_flat_.data();
            double* tau_yy_ptr = tau_yy_flat_.data();
            double* tau_yz_ptr = tau_yz_flat_.data();
            double* tau_zz_ptr = tau_zz_flat_.data();
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
                map(delete: tau_xz_ptr[0:n_interior]) \
                map(delete: tau_yy_ptr[0:n_interior]) \
                map(delete: tau_yz_ptr[0:n_interior]) \
                map(delete: tau_zz_ptr[0:n_interior]) \
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
    tau_xz_flat_.clear();
    tau_yy_flat_.clear();
    tau_yz_flat_.clear();
    tau_zz_flat_.clear();
    full_workspace_.clear();
    cached_total_cells_ = 0;
}

void TurbulenceNNTBNN::ensure_initialized(const Mesh& mesh) {
    if (!initialized_) {
        feature_computer_ = FeatureComputer(mesh);
        feature_computer_.set_reference(nu_, delta_, u_ref_);
        
        int n_interior = mesh.Nx * mesh.Ny * mesh.Nz;
        features_.resize(n_interior);
        basis_.resize(n_interior);
        
        initialized_ = true;
    }
}

void TurbulenceNNTBNN::estimate_k(const Mesh& mesh, const VectorField& velocity,
                                  ScalarField& k) {
    using numerics::C_MU;

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
            
            double k_est = (u_tau * u_tau / std::sqrt(C_MU)) * f_mu * f_mu;
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
    const int Nz = mesh.Nz;
    const int n_cells = Nx * Ny * Nz;
    const int n_total = mesh.total_cells();
    const int stride = mesh.total_Nx();
    const int plane_stride = mesh.total_Nx() * mesh.total_Ny();
    const double dx = mesh.dx;
    const double dy = mesh.dy;
    const double inv_2dx = 1.0 / (2.0 * dx);
    const double inv_2dy = 1.0 / (2.0 * dy);
    const double C_mu = numerics::C_MU;
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

        // Flatten k, omega, wall_distance (interior only, all z-planes)
        int idx = 0;
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    k_flat_[idx] = k_in(i, j, k);
                    omega_flat_[idx] = omega_in(i, j, k);
                    wall_dist_flat_[idx] = mesh.wall_distance(i, j, k);
                    ++idx;
                }
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
    double* tau_xz_ptr = tau_ij ? tau_xz_flat_.data() : nullptr;
    double* tau_yy_ptr = tau_ij ? tau_yy_flat_.data() : nullptr;
    double* tau_yz_ptr = tau_ij ? tau_yz_flat_.data() : nullptr;
    double* tau_zz_ptr = tau_ij ? tau_zz_flat_.data() : nullptr;
    
    // NN weights (already mapped to GPU via MLP::sync_weights_to_gpu)
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
            const int NUM_BASIS = 10;

            // Convert flat index to (i, j, k) - interior coordinates
            int i = cell_idx % Nx;
            int j = (cell_idx / Nx) % Ny;
            int kz = cell_idx / (Nx * Ny);

            // Full mesh indices (with ghost cells)
            int ii = i + 1;
            int jj = j + 1;
            int kk = kz + 1;
            int base = kk * plane_stride;
            int idx_ip = base + jj * stride + (ii + 1);
            int idx_im = base + jj * stride + (ii - 1);
            int idx_jp = base + (jj + 1) * stride + ii;
            int idx_jm = base + (jj - 1) * stride + ii;

            // ========== Step 1: Compute velocity gradients ==========
            double dudx_v = (u_ptr[idx_ip] - u_ptr[idx_im]) * inv_2dx;
            double dudy_v = (u_ptr[idx_jp] - u_ptr[idx_jm]) * inv_2dy;
            double dvdx_v = (v_ptr[idx_ip] - v_ptr[idx_im]) * inv_2dx;
            double dvdy_v = (v_ptr[idx_jp] - v_ptr[idx_jm]) * inv_2dy;
            // Note: w gradients not available in this fused path (only u,v uploaded).
            // For full 3D, use the device_view path. Here we set z-gradients to zero.
            double dudz_v = 0.0, dvdz_v = 0.0;
            double dwdx_v = 0.0, dwdy_v = 0.0, dwdz_v = 0.0;

            // ========== Step 2: Compute 3D strain and rotation tensors ==========
            // S_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
            double S[3][3];
            S[0][0] = dudx_v;
            S[1][1] = dvdy_v;
            S[2][2] = dwdz_v;
            S[0][1] = 0.5 * (dudy_v + dvdx_v);
            S[0][2] = 0.5 * (dudz_v + dwdx_v);
            S[1][2] = 0.5 * (dvdz_v + dwdy_v);
            S[1][0] = S[0][1];
            S[2][0] = S[0][2];
            S[2][1] = S[1][2];

            // Omega_ij = 0.5 * (du_i/dx_j - du_j/dx_i)
            double O[3][3];
            O[0][0] = 0.0; O[1][1] = 0.0; O[2][2] = 0.0;
            O[0][1] = 0.5 * (dudy_v - dvdx_v);
            O[0][2] = 0.5 * (dudz_v - dwdx_v);
            O[1][2] = 0.5 * (dvdz_v - dwdy_v);
            O[1][0] = -O[0][1];
            O[2][0] = -O[0][2];
            O[2][1] = -O[1][2];

            // S_mag and Omega_mag (Frobenius norms)
            double S_mag = 0.0;
            double Omega_mag = 0.0;
            for (int p = 0; p < 3; ++p) {
                for (int q = 0; q < 3; ++q) {
                    S_mag += S[p][q] * S[p][q];
                    Omega_mag += O[p][q] * O[p][q];
                }
            }
            S_mag = sqrt(S_mag);
            Omega_mag = sqrt(Omega_mag);

            double k_val = k_ptr[cell_idx];
            double omega_val = omega_ptr[cell_idx];

            // Ensure k and omega are positive and reasonable
            k_val = (k_val > k_min_val) ? k_val : k_min_val;
            k_val = (k_val < 100.0) ? k_val : 100.0;
            omega_val = (omega_val > 1e-10) ? omega_val : 1e-10;
            omega_val = (omega_val < 1e6) ? omega_val : 1e6;

            double eps = C_mu * k_val * omega_val;
            double eps_safe = (eps > 1e-20) ? eps : 1e-20;
            double tau_scale = k_val / eps_safe;

            double S_norm = S_mag * tau_scale;
            double Omega_norm = Omega_mag * tau_scale;

            // Normalize S and O by tau_scale
            double Sn[3][3], On[3][3];
            for (int p = 0; p < 3; ++p) {
                for (int q = 0; q < 3; ++q) {
                    Sn[p][q] = S[p][q] * tau_scale;
                    On[p][q] = O[p][q] * tau_scale;
                }
            }

            // ========== Step 3: Compute features ==========
            double feat[5];
            // tr(S^2)
            double trSnSn = 0.0;
            for (int p = 0; p < 3; ++p)
                for (int q = 0; q < 3; ++q)
                    trSnSn += Sn[p][q] * Sn[q][p];
            // tr(O^2)
            double trOnOn = 0.0;
            for (int p = 0; p < 3; ++p)
                for (int q = 0; q < 3; ++q)
                    trOnOn += On[p][q] * On[q][p];

            feat[0] = S_norm * S_norm;
            feat[1] = Omega_norm * Omega_norm;
            feat[2] = trSnSn;
            feat[3] = -trOnOn;  // tr(O^2) is negative; store positive magnitude
            feat[4] = wall_ptr[cell_idx] / delta_val;

            // Apply input scaling if available
            if (has_scaling && means_ptr != nullptr && stds_ptr != nullptr) {
                for (int f = 0; f < FEATURE_DIM && f < scale_size; ++f) {
                    feat[f] = (feat[f] - means_ptr[f]) / stds_ptr[f];
                }
            }

            // ========== Step 4: Compute Pope (1975) tensor basis ==========
            // Helper: 3x3 matrix multiply C = A * B (inline, no lambdas on GPU)
            // We'll compute products into local arrays as needed.
            double T[10][6];  // 10 basis tensors, 6 symmetric components each

            // Precompute needed matrix products
            // S2 = S*S
            double S2[3][3];
            for (int p = 0; p < 3; ++p)
                for (int q = 0; q < 3; ++q) {
                    S2[p][q] = 0.0;
                    for (int r = 0; r < 3; ++r)
                        S2[p][q] += Sn[p][r] * Sn[r][q];
                }

            // O2 = O*O
            double O2[3][3];
            for (int p = 0; p < 3; ++p)
                for (int q = 0; q < 3; ++q) {
                    O2[p][q] = 0.0;
                    for (int r = 0; r < 3; ++r)
                        O2[p][q] += On[p][r] * On[r][q];
                }

            // SO = S*O
            double SO[3][3];
            for (int p = 0; p < 3; ++p)
                for (int q = 0; q < 3; ++q) {
                    SO[p][q] = 0.0;
                    for (int r = 0; r < 3; ++r)
                        SO[p][q] += Sn[p][r] * On[r][q];
                }

            // OS = O*S
            double OS[3][3];
            for (int p = 0; p < 3; ++p)
                for (int q = 0; q < 3; ++q) {
                    OS[p][q] = 0.0;
                    for (int r = 0; r < 3; ++r)
                        OS[p][q] += On[p][r] * Sn[r][q];
                }

            // Traces
            double trS2 = S2[0][0] + S2[1][1] + S2[2][2];
            double trO2 = O2[0][0] + O2[1][1] + O2[2][2];

            // T1 = S
            T[0][0] = Sn[0][0]; T[0][1] = Sn[0][1]; T[0][2] = Sn[0][2];
            T[0][3] = Sn[1][1]; T[0][4] = Sn[1][2]; T[0][5] = Sn[2][2];

            // T2 = SO - OS
            T[1][0] = SO[0][0] - OS[0][0]; T[1][1] = SO[0][1] - OS[0][1]; T[1][2] = SO[0][2] - OS[0][2];
            T[1][3] = SO[1][1] - OS[1][1]; T[1][4] = SO[1][2] - OS[1][2]; T[1][5] = SO[2][2] - OS[2][2];

            // T3 = S^2 - (1/3)*tr(S^2)*I
            double trS2_3 = trS2 / 3.0;
            T[2][0] = S2[0][0] - trS2_3; T[2][1] = S2[0][1]; T[2][2] = S2[0][2];
            T[2][3] = S2[1][1] - trS2_3;  T[2][4] = S2[1][2]; T[2][5] = S2[2][2] - trS2_3;

            // T4 = O^2 - (1/3)*tr(O^2)*I
            double trO2_3 = trO2 / 3.0;
            T[3][0] = O2[0][0] - trO2_3; T[3][1] = O2[0][1]; T[3][2] = O2[0][2];
            T[3][3] = O2[1][1] - trO2_3;  T[3][4] = O2[1][2]; T[3][5] = O2[2][2] - trO2_3;

            // T5 = OS^2 - S^2O
            // OS2 = O*S^2
            double OS2[3][3];
            for (int p = 0; p < 3; ++p)
                for (int q = 0; q < 3; ++q) {
                    OS2[p][q] = 0.0;
                    for (int r = 0; r < 3; ++r)
                        OS2[p][q] += On[p][r] * S2[r][q];
                }
            // S2O = S^2*O
            double S2O[3][3];
            for (int p = 0; p < 3; ++p)
                for (int q = 0; q < 3; ++q) {
                    S2O[p][q] = 0.0;
                    for (int r = 0; r < 3; ++r)
                        S2O[p][q] += S2[p][r] * On[r][q];
                }
            T[4][0] = OS2[0][0] - S2O[0][0]; T[4][1] = OS2[0][1] - S2O[0][1]; T[4][2] = OS2[0][2] - S2O[0][2];
            T[4][3] = OS2[1][1] - S2O[1][1]; T[4][4] = OS2[1][2] - S2O[1][2]; T[4][5] = OS2[2][2] - S2O[2][2];

            // T6 = O^2*S + S*O^2 - (2/3)*tr(S*O^2)*I
            // O2S = O^2*S
            double O2S[3][3];
            for (int p = 0; p < 3; ++p)
                for (int q = 0; q < 3; ++q) {
                    O2S[p][q] = 0.0;
                    for (int r = 0; r < 3; ++r)
                        O2S[p][q] += O2[p][r] * Sn[r][q];
                }
            // SO2 = S*O^2
            double SO2[3][3];
            for (int p = 0; p < 3; ++p)
                for (int q = 0; q < 3; ++q) {
                    SO2[p][q] = 0.0;
                    for (int r = 0; r < 3; ++r)
                        SO2[p][q] += Sn[p][r] * O2[r][q];
                }
            double trSO2 = SO2[0][0] + SO2[1][1] + SO2[2][2];
            double trSO2_23 = 2.0 * trSO2 / 3.0;
            T[5][0] = O2S[0][0] + SO2[0][0] - trSO2_23;
            T[5][1] = O2S[0][1] + SO2[0][1];
            T[5][2] = O2S[0][2] + SO2[0][2];
            T[5][3] = O2S[1][1] + SO2[1][1] - trSO2_23;
            T[5][4] = O2S[1][2] + SO2[1][2];
            T[5][5] = O2S[2][2] + SO2[2][2] - trSO2_23;

            // T7 = O*S*O^2 - O^2*S*O
            // OSO2 = O*(S*O^2) — note SO2 already computed
            double OSO2[3][3];
            for (int p = 0; p < 3; ++p)
                for (int q = 0; q < 3; ++q) {
                    OSO2[p][q] = 0.0;
                    for (int r = 0; r < 3; ++r)
                        OSO2[p][q] += On[p][r] * SO2[r][q];
                }
            // O2SO = O^2*(S*O) — note OS already = O*S, we need (O^2)*(SO)
            // But formula is O^2*S*O. O2S already computed, so O2SO = O2S*O
            double O2SO[3][3];
            for (int p = 0; p < 3; ++p)
                for (int q = 0; q < 3; ++q) {
                    O2SO[p][q] = 0.0;
                    for (int r = 0; r < 3; ++r)
                        O2SO[p][q] += O2S[p][r] * On[r][q];
                }
            T[6][0] = OSO2[0][0] - O2SO[0][0]; T[6][1] = OSO2[0][1] - O2SO[0][1]; T[6][2] = OSO2[0][2] - O2SO[0][2];
            T[6][3] = OSO2[1][1] - O2SO[1][1]; T[6][4] = OSO2[1][2] - O2SO[1][2]; T[6][5] = OSO2[2][2] - O2SO[2][2];

            // T8 = S*O*S^2 - S^2*O*S
            // SOS2 = (S*O)*S^2 — SO already computed
            double SOS2[3][3];
            for (int p = 0; p < 3; ++p)
                for (int q = 0; q < 3; ++q) {
                    SOS2[p][q] = 0.0;
                    for (int r = 0; r < 3; ++r)
                        SOS2[p][q] += SO[p][r] * S2[r][q];
                }
            // S2OS = (S^2*O)*S — S2O already computed
            double S2OS[3][3];
            for (int p = 0; p < 3; ++p)
                for (int q = 0; q < 3; ++q) {
                    S2OS[p][q] = 0.0;
                    for (int r = 0; r < 3; ++r)
                        S2OS[p][q] += S2O[p][r] * Sn[r][q];
                }
            T[7][0] = SOS2[0][0] - S2OS[0][0]; T[7][1] = SOS2[0][1] - S2OS[0][1]; T[7][2] = SOS2[0][2] - S2OS[0][2];
            T[7][3] = SOS2[1][1] - S2OS[1][1]; T[7][4] = SOS2[1][2] - S2OS[1][2]; T[7][5] = SOS2[2][2] - S2OS[2][2];

            // T9 = O^2*S^2 + S^2*O^2 - (2/3)*tr(S^2*O^2)*I
            // O2S2 = O^2*S^2
            double O2S2[3][3];
            for (int p = 0; p < 3; ++p)
                for (int q = 0; q < 3; ++q) {
                    O2S2[p][q] = 0.0;
                    for (int r = 0; r < 3; ++r)
                        O2S2[p][q] += O2[p][r] * S2[r][q];
                }
            // S2O2 = S^2*O^2
            double S2O2[3][3];
            for (int p = 0; p < 3; ++p)
                for (int q = 0; q < 3; ++q) {
                    S2O2[p][q] = 0.0;
                    for (int r = 0; r < 3; ++r)
                        S2O2[p][q] += S2[p][r] * O2[r][q];
                }
            double trS2O2 = S2O2[0][0] + S2O2[1][1] + S2O2[2][2];
            double trS2O2_23 = 2.0 * trS2O2 / 3.0;
            T[8][0] = O2S2[0][0] + S2O2[0][0] - trS2O2_23;
            T[8][1] = O2S2[0][1] + S2O2[0][1];
            T[8][2] = O2S2[0][2] + S2O2[0][2];
            T[8][3] = O2S2[1][1] + S2O2[1][1] - trS2O2_23;
            T[8][4] = O2S2[1][2] + S2O2[1][2];
            T[8][5] = O2S2[2][2] + S2O2[2][2] - trS2O2_23;

            // T10 = O*S^2*O^2 - O^2*S^2*O
            // OS2O2 = O*(S^2*O^2) — S2O2 already computed
            double OS2O2[3][3];
            for (int p = 0; p < 3; ++p)
                for (int q = 0; q < 3; ++q) {
                    OS2O2[p][q] = 0.0;
                    for (int r = 0; r < 3; ++r)
                        OS2O2[p][q] += On[p][r] * S2O2[r][q];
                }
            // O2S2O = (O^2*S^2)*O — O2S2 already computed
            double O2S2O[3][3];
            for (int p = 0; p < 3; ++p)
                for (int q = 0; q < 3; ++q) {
                    O2S2O[p][q] = 0.0;
                    for (int r = 0; r < 3; ++r)
                        O2S2O[p][q] += O2S2[p][r] * On[r][q];
                }
            T[9][0] = OS2O2[0][0] - O2S2O[0][0]; T[9][1] = OS2O2[0][1] - O2S2O[0][1]; T[9][2] = OS2O2[0][2] - O2S2O[0][2];
            T[9][3] = OS2O2[1][1] - O2S2O[1][1]; T[9][4] = OS2O2[1][2] - O2S2O[1][2]; T[9][5] = OS2O2[2][2] - O2S2O[2][2];

            // ========== Step 5: NN Forward Pass ==========
            // Use workspace for ping-pong buffers
            // Skip past gradients (9*n_cells), features (5*n_cells), and basis (60*n_cells)
            double* nn_work_start = work_ptr + n_cells * (9 + 5 + 60);
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
            double G[10] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            for (int n = 0; n < NUM_BASIS && n < output_dim_val; ++n) {
                G[n] = current[n];
            }

            double b_xx = 0.0, b_xy = 0.0, b_xz = 0.0;
            double b_yy = 0.0, b_yz = 0.0, b_zz = 0.0;
            for (int n = 0; n < NUM_BASIS; ++n) {
                b_xx += G[n] * T[n][0]; b_xy += G[n] * T[n][1]; b_xz += G[n] * T[n][2];
                b_yy += G[n] * T[n][3]; b_yz += G[n] * T[n][4]; b_zz += G[n] * T[n][5];
            }

            // ========== Step 7: Compute Reynolds stresses (optional) ==========
            if (compute_tau && tau_xx_ptr != nullptr) {
                double k_safe2 = (k_val > 0.0) ? k_val : 0.0;
                tau_xx_ptr[cell_idx] = 2.0 * k_safe2 * (b_xx + 1.0/3.0);
                tau_xy_ptr[cell_idx] = 2.0 * k_safe2 * b_xy;
                tau_xz_ptr[cell_idx] = 2.0 * k_safe2 * b_xz;
                tau_yy_ptr[cell_idx] = 2.0 * k_safe2 * (b_yy + 1.0/3.0);
                tau_yz_ptr[cell_idx] = 2.0 * k_safe2 * b_yz;
                tau_zz_ptr[cell_idx] = 2.0 * k_safe2 * (b_zz + 1.0/3.0);
            }

            // ========== Step 8: Compute eddy viscosity ==========
            // Use full 3D Sxy for nu_t extraction
            double Sxy = S[0][1];
            double nu_t_val = 0.0;
            double abs_Sxy = (Sxy >= 0.0) ? Sxy : -Sxy;
            if (abs_Sxy > 1e-10) {
                double tmp = -b_xy * k_val / Sxy;
                nu_t_val = (tmp >= 0.0) ? tmp : -tmp;
            } else {
                if (S_mag > 1e-10) {
                    double b_mag = sqrt(b_xx*b_xx + 2.0*b_xy*b_xy + 2.0*b_xz*b_xz
                                      + b_yy*b_yy + 2.0*b_yz*b_yz + b_zz*b_zz);
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
                                       from(tau_xz_ptr[0:n_cells]) \
                                       from(tau_yy_ptr[0:n_cells]) \
                                       from(tau_yz_ptr[0:n_cells]) \
                                       from(tau_zz_ptr[0:n_cells])
        }
    }
    
    // ========== Step 5: Unflatten results ==========
    {
        TIMED_SCOPE("nn_tbnn_unflatten");

        int idx = 0;
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    nu_t(i, j, k) = nu_t_flat_[idx];
                    if (tau_ij) {
                        tau_ij->xx(i, j, k) = tau_xx_flat_[idx];
                        tau_ij->xy(i, j, k) = tau_xy_flat_[idx];
                        tau_ij->xz(i, j, k) = tau_xz_flat_[idx];
                        tau_ij->yy(i, j, k) = tau_yy_flat_[idx];
                        tau_ij->yz(i, j, k) = tau_yz_flat_[idx];
                        tau_ij->zz(i, j, k) = tau_zz_flat_[idx];
                    }
                    ++idx;
                }
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
    TIMED_SCOPE("nn_tbnn_update");
    
    (void)device_view;  // avoid -Wunused-parameter in CPU builds
    
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
                omega_local(i, j) = std::sqrt(k_local(i, j)) / (numerics::KAPPA * std::max(y_wall, numerics::Y_WALL_FLOOR));
            }
        }
    }
    
    const int Nx = mesh.Nx;
    const int Ny = mesh.Ny;
    const int Nz = mesh.Nz;
    [[maybe_unused]] const int n_cells = Nx * Ny * Nz;
    [[maybe_unused]] const int Ng = mesh.Nghost;
    
#ifdef USE_GPU_OFFLOAD
    // GPU path: require device_view and gpu_ready (host fallback forbidden)
    if (!device_view || !gpu_ready_) {
        throw std::runtime_error("NN-TBNN GPU pipeline requires device_view and GPU buffers initialized");
    }
    
    // Validate device_view has all required buffers
    if (!device_view->u_face || !device_view->v_face ||
        !device_view->k || !device_view->omega ||
        !device_view->dudx || !device_view->dudy || !device_view->dvdx || !device_view->dvdy ||
        !device_view->wall_distance ||
        !device_view->nu_t) {
        throw std::runtime_error("NN-TBNN GPU pipeline: device_view missing required buffers");
    }
    if (tau_ij && (!device_view->tau_xx || !device_view->tau_xy || !device_view->tau_yy)) {
        throw std::runtime_error("NN-TBNN GPU pipeline: tau_ij requested but tau buffers not provided in device_view");
    }
    
    {
        TIMED_SCOPE("nn_tbnn_full_gpu");
        
        // Ensure GPU buffers are allocated
        allocate_gpu_buffers(n_cells);

        const int total_cells = mesh.total_cells();
        const int u_total = velocity.u_total_size();
        const int v_total = velocity.v_total_size();
        const int w_total = velocity.w_total_size();
        const int cell_stride = mesh.total_Nx();
        const int cell_plane_stride = mesh.total_Nx() * mesh.total_Ny();
        const int u_stride = velocity.u_stride();
        const int u_plane_stride = velocity.u_plane_stride();
        const int v_stride = velocity.v_stride();
        const int v_plane_stride = velocity.v_plane_stride();
        const int w_stride = velocity.w_stride();
        const int w_plane_stride = velocity.w_plane_stride();

        // Step 1: Compute gradients on GPU
        {
            TIMED_SCOPE("nn_tbnn_gradients_gpu");
            gpu_kernels::compute_gradients_from_mac_gpu(
                device_view->u_face, device_view->v_face, device_view->w_face,
                device_view->dudx, device_view->dudy,
                device_view->dvdx, device_view->dvdy,
                device_view->dudz, device_view->dvdz,
                device_view->dwdx, device_view->dwdy, device_view->dwdz,
                Nx, Ny, Nz, Ng,
                mesh.dx, mesh.dy, mesh.dz,
                u_stride, v_stride, cell_stride,
                u_plane_stride, v_plane_stride,
                w_stride, w_plane_stride, cell_plane_stride,
                u_total, v_total, w_total, total_cells,
                device_view->dyc,
                device_view->dyc_size
            );
        }

        // Step 2: Compute TBNN features + basis on GPU
        {
            TIMED_SCOPE("nn_tbnn_features_gpu");
            double* feat_ptr = features_flat_.data();
            double* basis_ptr = basis_flat_.data();

            gpu_kernels::compute_tbnn_features_gpu(
                device_view->dudx, device_view->dudy,
                device_view->dvdx, device_view->dvdy,
                device_view->k, device_view->omega,
                device_view->wall_distance,
                feat_ptr, basis_ptr,
                Nx, Ny, Nz, Ng,
                cell_stride, cell_plane_stride,
                total_cells,
                nu_, delta_
            );
        }

        // Step 3: Run NN inference on GPU
        {
            TIMED_SCOPE("nn_tbnn_inference_gpu");
            double* feat_ptr = features_flat_.data();
            double* out_ptr = outputs_flat_.data();
            double* work_ptr = workspace_.data();
            mlp_.forward_batch_gpu(feat_ptr, out_ptr, n_cells, work_ptr);
        }

        // Step 4: Postprocess on GPU (anisotropy -> nu_t)
        {
            TIMED_SCOPE("nn_tbnn_postprocess_gpu");
            double* out_ptr = outputs_flat_.data();
            double* basis_ptr = basis_flat_.data();
            double* k_ptr = device_view->k;
            double* dudx_ptr = device_view->dudx;
            double* dudy_ptr = device_view->dudy;
            double* dvdx_ptr = device_view->dvdx;
            double* dvdy_ptr = device_view->dvdy;

            double* nu_t_ptr = device_view->nu_t;
            double* tau_xx_ptr = device_view->tau_xx;
            double* tau_xy_ptr = device_view->tau_xy;
            double* tau_yy_ptr = device_view->tau_yy;

            gpu_kernels::postprocess_nn_outputs_gpu(
                out_ptr, basis_ptr,
                k_ptr, dudx_ptr, dudy_ptr, dvdx_ptr, dvdy_ptr,
                nu_t_ptr,
                tau_xx_ptr, tau_xy_ptr, tau_yy_ptr,
                Nx, Ny, Nz, Ng,
                cell_stride, cell_plane_stride,
                total_cells,
                mlp_.output_dim(),
                nu_
            );
        }
        return;
    }
#else
    // CPU path (only for CPU builds)
    {
        // Compute features and tensor basis
        TIMED_SCOPE("nn_tbnn_features");
        feature_computer_.compute_tbnn_features(velocity, k_local, omega_local, 
                                                features_, basis_);
    }
    
    // Sequential inference (host build)
    {
        TIMED_SCOPE("nn_tbnn_inference_cpu");
        
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
                
                // Construct anisotropy tensor (full 3D: 6 components)
                double b_xx, b_xy, b_xz, b_yy, b_yz, b_zz;
                TensorBasis::construct_anisotropy(G, basis_[idx],
                                                  b_xx, b_xy, b_xz, b_yy, b_yz, b_zz);

                // Convert to Reynolds stresses if requested
                if (tau_ij) {
                    double k_val = k_local(i, j);
                    double tau_xx, tau_xy, tau_xz, tau_yy, tau_yz, tau_zz;
                    TensorBasis::anisotropy_to_reynolds_stress(
                        b_xx, b_xy, b_xz, b_yy, b_yz, b_zz, k_val,
                        tau_xx, tau_xy, tau_xz, tau_yy, tau_yz, tau_zz);
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
                        double b_mag = std::sqrt(b_xx*b_xx + 2.0*b_xy*b_xy + 2.0*b_xz*b_xz
                                               + b_yy*b_yy + 2.0*b_yz*b_yz + b_zz*b_zz);
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
#endif  // USE_GPU_OFFLOAD
}

} // namespace nncfd
