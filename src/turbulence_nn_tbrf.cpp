/// @file turbulence_nn_tbrf.cpp
/// @brief TBRF (Tensor Basis Random Forest) turbulence model implementation
///
/// Implements Kaandorp & Dwight (2020) random forest approach for predicting
/// Reynolds stress anisotropy coefficients g_n in the tensor basis expansion:
///   b_ij = sum_n g_n(lambda) * T^(n)_ij(S_hat, Omega_hat)
///
/// GPU strategy: tree structure arrays are uploaded once and kept on GPU.
/// A fused kernel computes features, tree traversal, anisotropy, and nu_t
/// per cell in a single pass.

#include "turbulence_nn_tbrf.hpp"
#include "gpu_kernels.hpp"
#include "timing.hpp"
#include "numerics.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>

namespace nncfd {

TurbulenceNNTBRF::TurbulenceNNTBRF()
    : feature_computer_(Mesh()) {}

TurbulenceNNTBRF::~TurbulenceNNTBRF() {
    free_work_buffers_gpu();
    free_tree_buffers_gpu();
}

// ============================================================================
// Binary file loading
// ============================================================================

void TurbulenceNNTBRF::load(const std::string& model_dir) {
    TIMED_SCOPE("nn_tbrf_load");

    // Load trees.bin
    {
        const std::string bin_path = model_dir + "/trees.bin";
        std::ifstream fin(bin_path, std::ios::binary);
        if (!fin.is_open()) {
            throw std::runtime_error("TBRF: cannot open " + bin_path);
        }

        // Read header: [total_nodes: int32] [n_basis: int32] [n_trees: int32]
        int32_t header[3];
        fin.read(reinterpret_cast<char*>(header), sizeof(header));
        if (!fin.good()) {
            throw std::runtime_error("TBRF: failed to read header from " + bin_path);
        }
        total_nodes_ = static_cast<int>(header[0]);
        n_basis_ = static_cast<int>(header[1]);
        n_trees_ = static_cast<int>(header[2]);

        if (total_nodes_ <= 0 || n_basis_ <= 0 || n_trees_ <= 0) {
            throw std::runtime_error("TBRF: invalid header values in " + bin_path);
        }

        // Allocate arrays
        children_left_.resize(total_nodes_);
        children_right_.resize(total_nodes_);
        feature_.resize(total_nodes_);
        threshold_.resize(total_nodes_);
        value_.resize(total_nodes_);

        // Read data arrays in order
        fin.read(reinterpret_cast<char*>(children_left_.data()),
                 total_nodes_ * sizeof(int32_t));
        fin.read(reinterpret_cast<char*>(children_right_.data()),
                 total_nodes_ * sizeof(int32_t));
        fin.read(reinterpret_cast<char*>(feature_.data()),
                 total_nodes_ * sizeof(int32_t));
        fin.read(reinterpret_cast<char*>(threshold_.data()),
                 total_nodes_ * sizeof(float));
        fin.read(reinterpret_cast<char*>(value_.data()),
                 total_nodes_ * sizeof(float));

        if (!fin.good()) {
            throw std::runtime_error("TBRF: incomplete read from " + bin_path);
        }

        std::cerr << "TBRF: loaded " << total_nodes_ << " nodes, "
                  << n_basis_ << " basis functions, "
                  << n_trees_ << " trees per basis\n";
    }

    // Load tree_offsets.txt
    {
        const std::string offsets_path = model_dir + "/tree_offsets.txt";
        std::ifstream fin(offsets_path);
        if (!fin.is_open()) {
            throw std::runtime_error("TBRF: cannot open " + offsets_path);
        }

        tree_offsets_.clear();
        std::string line;
        while (std::getline(fin, line)) {
            // Skip comments and empty lines
            if (line.empty() || line[0] == '#') {
                continue;
            }
            std::istringstream iss(line);
            TreeOffset offset;
            if (iss >> offset.basis_idx >> offset.tree_idx
                    >> offset.start_node >> offset.n_nodes) {
                tree_offsets_.push_back(offset);
            }
        }

        if (tree_offsets_.empty()) {
            throw std::runtime_error("TBRF: no valid entries in " + offsets_path);
        }

        std::cerr << "TBRF: loaded " << tree_offsets_.size()
                  << " tree offset entries\n";
    }

    // Build flat tree_starts_ array for GPU-friendly access
    {
        tree_starts_.resize(n_basis_ * n_trees_, -1);
        for (const auto& offset : tree_offsets_) {
            int idx = offset.basis_idx * n_trees_ + offset.tree_idx;
            if (idx >= 0 && idx < static_cast<int>(tree_starts_.size())) {
                tree_starts_[idx] = offset.start_node;
            }
        }
    }

    // Load input normalization (optional)
    {
        const std::string means_path = model_dir + "/input_means.txt";
        const std::string stds_path = model_dir + "/input_stds.txt";

        std::ifstream means_fin(means_path);
        std::ifstream stds_fin(stds_path);

        if (means_fin.is_open() && stds_fin.is_open()) {
            input_means_.clear();
            input_stds_.clear();

            double val;
            while (means_fin >> val) {
                input_means_.push_back(val);
            }
            while (stds_fin >> val) {
                input_stds_.push_back(val);
            }

            if (!input_means_.empty() && input_means_.size() == input_stds_.size()) {
                has_scaling_ = true;
                std::cerr << "TBRF: loaded input scaling ("
                          << input_means_.size() << " features)\n";
            }
        }
    }
}

// ============================================================================
// Tree traversal (CPU)
// ============================================================================

float TurbulenceNNTBRF::traverse_tree(const double* features,
                                       int start_node) const {
    int node = start_node;
    while (true) {
        int left = children_left_[node];
        int right = children_right_[node];

        // Leaf node: children_left == children_right == -1
        if (left == -1) {
            return value_[node];
        }

        // Internal node: branch on feature < threshold
        int feat_idx = feature_[node];
        double feat_val = features[feat_idx];
        if (feat_val <= static_cast<double>(threshold_[node])) {
            node = left;
        } else {
            node = right;
        }
    }
}

double TurbulenceNNTBRF::predict_coefficient(int basis_idx,
                                              const double* features) const {
    double sum = 0.0;
    int count = 0;

    for (const auto& offset : tree_offsets_) {
        if (offset.basis_idx == basis_idx) {
            sum += static_cast<double>(
                traverse_tree(features, offset.start_node));
            ++count;
        }
    }

    return (count > 0) ? sum / count : 0.0;
}

// ============================================================================
// GPU buffer management
// ============================================================================

void TurbulenceNNTBRF::map_trees_to_gpu() {
#ifdef USE_GPU_OFFLOAD
    if (tree_buffers_on_gpu_) {
        return;  // Already mapped
    }

    int32_t* cl_ptr = children_left_.data();
    int32_t* cr_ptr = children_right_.data();
    int32_t* feat_ptr = feature_.data();
    float* thresh_ptr = threshold_.data();
    float* val_ptr = value_.data();
    int* starts_ptr = tree_starts_.data();
    size_t n_nodes = static_cast<size_t>(total_nodes_);
    size_t n_starts = tree_starts_.size();

    #pragma omp target enter data \
        map(to: cl_ptr[0:n_nodes]) \
        map(to: cr_ptr[0:n_nodes]) \
        map(to: feat_ptr[0:n_nodes]) \
        map(to: thresh_ptr[0:n_nodes]) \
        map(to: val_ptr[0:n_nodes]) \
        map(to: starts_ptr[0:n_starts])

    // Map scaling arrays if available
    if (has_scaling_ && !input_means_.empty()) {
        double* means_ptr = input_means_.data();
        double* stds_ptr = input_stds_.data();
        size_t scale_size = input_means_.size();

        #pragma omp target enter data \
            map(to: means_ptr[0:scale_size]) \
            map(to: stds_ptr[0:scale_size])
    }

    tree_buffers_on_gpu_ = true;
#endif
}

void TurbulenceNNTBRF::free_tree_buffers_gpu() {
#ifdef USE_GPU_OFFLOAD
    if (!tree_buffers_on_gpu_) {
        return;
    }

    tree_buffers_on_gpu_ = false;

    if (!children_left_.empty()) {
        int32_t* cl_ptr = children_left_.data();
        int32_t* cr_ptr = children_right_.data();
        int32_t* feat_ptr = feature_.data();
        float* thresh_ptr = threshold_.data();
        float* val_ptr = value_.data();
        int* starts_ptr = tree_starts_.data();
        size_t n_nodes = static_cast<size_t>(total_nodes_);
        size_t n_starts = tree_starts_.size();

        #pragma omp target exit data \
            map(delete: cl_ptr[0:n_nodes]) \
            map(delete: cr_ptr[0:n_nodes]) \
            map(delete: feat_ptr[0:n_nodes]) \
            map(delete: thresh_ptr[0:n_nodes]) \
            map(delete: val_ptr[0:n_nodes]) \
            map(delete: starts_ptr[0:n_starts])
    }

    if (has_scaling_ && !input_means_.empty()) {
        double* means_ptr = input_means_.data();
        double* stds_ptr = input_stds_.data();
        size_t scale_size = input_means_.size();

        #pragma omp target exit data \
            map(delete: means_ptr[0:scale_size]) \
            map(delete: stds_ptr[0:scale_size])
    }
#endif
}

void TurbulenceNNTBRF::allocate_work_buffers_gpu(int n_cells) {
#ifdef USE_GPU_OFFLOAD
    if (n_cells == cached_n_cells_ && !features_flat_.empty() && work_buffers_on_gpu_) {
        return;  // Already allocated and mapped
    }

    free_work_buffers_gpu();

    features_flat_.resize(n_cells * 5);
    basis_flat_.resize(n_cells * TensorBasis::NUM_BASIS * 3);
    nu_t_flat_.resize(n_cells);
    tau_xx_flat_.resize(n_cells);
    tau_xy_flat_.resize(n_cells);
    tau_yy_flat_.resize(n_cells);

    double* feat_ptr = features_flat_.data();
    double* basis_ptr = basis_flat_.data();
    double* nu_t_ptr = nu_t_flat_.data();
    double* txx_ptr = tau_xx_flat_.data();
    double* txy_ptr = tau_xy_flat_.data();
    double* tyy_ptr = tau_yy_flat_.data();
    size_t feat_sz = features_flat_.size();
    size_t basis_sz = basis_flat_.size();

    #pragma omp target enter data \
        map(alloc: feat_ptr[0:feat_sz]) \
        map(alloc: basis_ptr[0:basis_sz]) \
        map(alloc: nu_t_ptr[0:n_cells]) \
        map(alloc: txx_ptr[0:n_cells]) \
        map(alloc: txy_ptr[0:n_cells]) \
        map(alloc: tyy_ptr[0:n_cells])

    work_buffers_on_gpu_ = true;
    cached_n_cells_ = n_cells;
#else
    (void)n_cells;
#endif
}

void TurbulenceNNTBRF::free_work_buffers_gpu() {
#ifdef USE_GPU_OFFLOAD
    if (!work_buffers_on_gpu_) {
        return;
    }

    work_buffers_on_gpu_ = false;

    if (!features_flat_.empty()) {
        double* feat_ptr = features_flat_.data();
        double* basis_ptr = basis_flat_.data();
        double* nu_t_ptr = nu_t_flat_.data();
        double* txx_ptr = tau_xx_flat_.data();
        double* txy_ptr = tau_xy_flat_.data();
        double* tyy_ptr = tau_yy_flat_.data();
        size_t feat_sz = features_flat_.size();
        size_t basis_sz = basis_flat_.size();
        size_t n_cells = nu_t_flat_.size();

        #pragma omp target exit data \
            map(delete: feat_ptr[0:feat_sz]) \
            map(delete: basis_ptr[0:basis_sz]) \
            map(delete: nu_t_ptr[0:n_cells]) \
            map(delete: txx_ptr[0:n_cells]) \
            map(delete: txy_ptr[0:n_cells]) \
            map(delete: tyy_ptr[0:n_cells])
    }
#endif
    features_flat_.clear();
    basis_flat_.clear();
    nu_t_flat_.clear();
    tau_xx_flat_.clear();
    tau_xy_flat_.clear();
    tau_yy_flat_.clear();
}

void TurbulenceNNTBRF::initialize_gpu_buffers(const Mesh& mesh) {
#ifdef USE_GPU_OFFLOAD
    gpu::verify_device_available();

    const int n_cells = mesh.Nx * mesh.Ny * mesh.Nz;

    // Upload tree structure to GPU (once)
    map_trees_to_gpu();

    // Allocate work buffers
    allocate_work_buffers_gpu(n_cells);

    gpu_ready_ = (tree_buffers_on_gpu_ && work_buffers_on_gpu_);
    if (gpu_ready_) {
        std::cerr << "TBRF: GPU buffers initialized ("
                  << total_nodes_ << " tree nodes, "
                  << n_cells << " work cells)\n";
    }
#else
    (void)mesh;
    gpu_ready_ = false;
#endif
}

void TurbulenceNNTBRF::cleanup_gpu_buffers() {
    free_work_buffers_gpu();
    free_tree_buffers_gpu();
    gpu_ready_ = false;
}

// ============================================================================
// Initialization helpers
// ============================================================================

void TurbulenceNNTBRF::ensure_initialized(const Mesh& mesh) {
    if (!initialized_) {
        feature_computer_ = FeatureComputer(mesh);
        feature_computer_.set_reference(nu_, delta_, u_ref_);

        int n_interior = mesh.Nx * mesh.Ny * mesh.Nz;
        features_.resize(n_interior);
        basis_.resize(n_interior);

        initialized_ = true;
    }
}

void TurbulenceNNTBRF::estimate_k(const Mesh& mesh,
                                    const VectorField& velocity,
                                    ScalarField& k) {
    using numerics::C_MU;

    double u_tau = 0.0;
    {
        int j = mesh.j_begin();
        double dudy_avg = 0.0;
        int count = 0;
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            if (j + 1 < mesh.j_end() && j - 1 >= mesh.j_begin()) {
                double dudy = (velocity.u(i, j + 1) - velocity.u(i, j - 1))
                              / (2.0 * mesh.dy);
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

// ============================================================================
// Main update
// ============================================================================

void TurbulenceNNTBRF::update(
    const Mesh& mesh,
    const VectorField& velocity,
    const ScalarField& k_in,
    const ScalarField& omega_in,
    ScalarField& nu_t,
    TensorField* tau_ij,
    const TurbulenceDeviceView* device_view) {
    TIMED_SCOPE("nn_tbrf_update");

    const int Nx = mesh.Nx;
    const int Ny = mesh.Ny;
    const int Nz = mesh.Nz;
    [[maybe_unused]] const int n_cells = Nx * Ny * Nz;
    [[maybe_unused]] const int Ng = mesh.Nghost;

#ifdef USE_GPU_OFFLOAD
    // GPU path: use device_view directly (k/omega already on GPU)
    if (device_view && gpu_ready_) {

        // Validate device_view has required buffers
        if (!device_view->u_face || !device_view->v_face ||
        !device_view->k || !device_view->omega ||
        !device_view->dudx || !device_view->dudy || !device_view->dvdx || !device_view->dvdy ||
        !device_view->wall_distance ||
        !device_view->nu_t) {
        throw std::runtime_error("NN-TBRF GPU pipeline: device_view missing required buffers");
    }
    if (tau_ij && (!device_view->tau_xx || !device_view->tau_xy || !device_view->tau_yy)) {
        throw std::runtime_error("NN-TBRF GPU pipeline: tau_ij requested but tau buffers not provided in device_view");
    }

    {
        TIMED_SCOPE("nn_tbrf_full_gpu");

        // Ensure GPU buffers are allocated
        allocate_work_buffers_gpu(n_cells);

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

        // Step 1: Compute gradients on GPU (reuse TBNN gradient kernel)
        {
            TIMED_SCOPE("nn_tbrf_gradients_gpu");
            gpu_kernels::compute_gradients_from_mac_gpu(
                device_view->u_face, device_view->v_face, device_view->w_face,
                device_view->dudx, device_view->dudy,
                device_view->dvdx, device_view->dvdy,
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

        // Step 2: Compute TBNN features + basis on GPU (reuse TBNN feature kernel)
        {
            TIMED_SCOPE("nn_tbrf_features_gpu");
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

        // Step 3: Tree traversal + postprocessing (fused GPU kernel)
        {
            TIMED_SCOPE("nn_tbrf_tree_kernel");

            // Get pointers to GPU-resident data
            double* feat_ptr = features_flat_.data();
            double* basis_ptr = basis_flat_.data();
            double* k_dev = device_view->k;
            double* dudx_dev = device_view->dudx;
            double* dudy_dev = device_view->dudy;
            double* dvdx_dev = device_view->dvdx;
            double* dvdy_dev = device_view->dvdy;
            double* nu_t_dev = device_view->nu_t;
            double* tau_xx_dev = device_view->tau_xx;
            double* tau_xy_dev = device_view->tau_xy;
            double* tau_yy_dev = device_view->tau_yy;

            // Tree structure pointers (already on GPU from map_trees_to_gpu)
            int32_t* cl_ptr = children_left_.data();
            int32_t* cr_ptr = children_right_.data();
            int32_t* ft_ptr = feature_.data();
            float* th_ptr = threshold_.data();
            float* vl_ptr = value_.data();
            int* starts_ptr = tree_starts_.data();

            // Scaling pointers
            double* means_ptr = has_scaling_ ? input_means_.data() : nullptr;
            double* stds_ptr = has_scaling_ ? input_stds_.data() : nullptr;

            // Copy member vars to locals for GPU (nvc++ this-transfer workaround)
            const int n_basis_local = n_basis_;
            const int n_trees_local = n_trees_;
            [[maybe_unused]] const int n_nodes_local = total_nodes_;
            const bool has_scale = has_scaling_;
            const int scale_size = has_scaling_ ? static_cast<int>(input_means_.size()) : 0;
            const double nu_val = nu_;
            const bool compute_tau = (tau_ij != nullptr);

            // Array sizes for map(present:) clauses
            [[maybe_unused]] const size_t feat_sz = features_flat_.size();
            [[maybe_unused]] const size_t basis_sz = basis_flat_.size();
            [[maybe_unused]] const size_t n_nd = static_cast<size_t>(n_nodes_local);
            [[maybe_unused]] const size_t n_st = tree_starts_.size();
            [[maybe_unused]] const size_t sc_sz = static_cast<size_t>(scale_size);
            [[maybe_unused]] const int tc = total_cells;

            #pragma omp target teams distribute parallel for \
                map(present: feat_ptr[0:feat_sz], basis_ptr[0:basis_sz]) \
                map(present: cl_ptr[0:n_nd], cr_ptr[0:n_nd], ft_ptr[0:n_nd]) \
                map(present: th_ptr[0:n_nd], vl_ptr[0:n_nd], starts_ptr[0:n_st]) \
                map(present: k_dev[0:tc], dudx_dev[0:tc], dudy_dev[0:tc]) \
                map(present: dvdx_dev[0:tc], dvdy_dev[0:tc], nu_t_dev[0:tc])
            for (int cell_idx = 0; cell_idx < n_cells; ++cell_idx) {
                const int FEATURE_DIM = 5;
                const int NUM_BASIS = 4;

                // Convert flat index to (i, j, kz) interior coordinates
                int i = cell_idx % Nx;
                int j = (cell_idx / Nx) % Ny;
                int kz = cell_idx / (Nx * Ny);

                // Full mesh indices (with ghost cells)
                int ii = i + Ng;
                int jj = j + Ng;
                int kk = kz + Ng;
                int cell_flat = kk * cell_plane_stride + jj * cell_stride + ii;

                // ====== Read precomputed features (5 invariants) ======
                double feat[5];
                for (int f = 0; f < FEATURE_DIM; ++f) {
                    feat[f] = feat_ptr[cell_idx * FEATURE_DIM + f];
                }

                // Apply input scaling
                if (has_scale && means_ptr != nullptr && stds_ptr != nullptr) {
                    for (int f = 0; f < FEATURE_DIM && f < scale_size; ++f) {
                        feat[f] = (feat[f] - means_ptr[f]) / stds_ptr[f];
                    }
                }

                // ====== Read precomputed tensor basis (4 x 3 components) ======
                double T[4][3];
                for (int n = 0; n < NUM_BASIS; ++n) {
                    for (int c = 0; c < 3; ++c) {
                        T[n][c] = basis_ptr[cell_idx * 12 + n * 3 + c];
                    }
                }

                // ====== Tree traversal: predict G coefficients ======
                double G[4] = {0.0, 0.0, 0.0, 0.0};
                int n_basis_use = (n_basis_local < NUM_BASIS) ? n_basis_local : NUM_BASIS;

                for (int b = 0; b < n_basis_use; ++b) {
                    double sum = 0.0;
                    int count = 0;

                    for (int t = 0; t < n_trees_local; ++t) {
                        int start = starts_ptr[b * n_trees_local + t];
                        if (start < 0) {
                            continue;  // Invalid tree
                        }

                        // Traverse tree to leaf
                        int node = start;
                        while (cl_ptr[node] != -1) {
                            int feat_idx = ft_ptr[node];
                            double feat_val = feat[feat_idx];
                            if (feat_val <= static_cast<double>(th_ptr[node])) {
                                node = cl_ptr[node];
                            } else {
                                node = cr_ptr[node];
                            }
                        }

                        sum += static_cast<double>(vl_ptr[node]);
                        count = count + 1;
                    }

                    G[b] = (count > 0) ? sum / count : 0.0;
                }

                // ====== Construct anisotropy b_ij = sum_n G[n] * T^(n)_ij ======
                double b_xx = 0.0, b_xy = 0.0, b_yy = 0.0;
                for (int n = 0; n < NUM_BASIS; ++n) {
                    b_xx += G[n] * T[n][0];
                    b_xy += G[n] * T[n][1];
                    b_yy += G[n] * T[n][2];
                }

                // ====== Reynolds stresses (optional) ======
                if (compute_tau && tau_xx_dev != nullptr) {
                    double k_val = k_dev[cell_flat];
                    double k_safe = (k_val > 0.0) ? k_val : 0.0;
                    tau_xx_dev[cell_flat] = 2.0 * k_safe * (b_xx + 1.0 / 3.0);
                    tau_xy_dev[cell_flat] = 2.0 * k_safe * b_xy;
                    tau_yy_dev[cell_flat] = 2.0 * k_safe * (b_yy + 1.0 / 3.0);
                }

                // ====== Compute eddy viscosity ======
                double Sxy = 0.5 * (dudy_dev[cell_flat] + dvdx_dev[cell_flat]);
                double k_val = k_dev[cell_flat];

                double nu_t_val = 0.0;
                double abs_Sxy = (Sxy >= 0.0) ? Sxy : -Sxy;
                if (abs_Sxy > 1e-10) {
                    double tmp = -b_xy * k_val / Sxy;
                    nu_t_val = (tmp >= 0.0) ? tmp : -tmp;
                } else {
                    double Sxx = dudx_dev[cell_flat];
                    double Syy = dvdy_dev[cell_flat];
                    double S_mag = sqrt(2.0 * (Sxx * Sxx + Syy * Syy + 2.0 * Sxy * Sxy));
                    if (S_mag > 1e-10) {
                        double b_mag = sqrt(b_xx * b_xx + 2.0 * b_xy * b_xy + b_yy * b_yy);
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

                nu_t_dev[cell_flat] = nu_t_val;
            }
        }
    }  // end nn_tbrf_full_gpu TIMED_SCOPE
        return;
    }
#else
    // CPU path (only for CPU builds)
    {
        (void)device_view;

        ensure_initialized(mesh);
        feature_computer_.set_reference(nu_, delta_, u_ref_);

        // Use provided k/omega or estimate (CPU only — GPU uses device_view->k/omega)
        ScalarField k_local(mesh);
        ScalarField omega_local(mesh);

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
                    omega_local(i, j) = std::sqrt(k_local(i, j))
                        / (numerics::KAPPA * std::max(y_wall, numerics::Y_WALL_FLOOR));
                }
            }
        }

        // Step 1: Compute features and tensor basis (reuses TBNN code)
        {
            TIMED_SCOPE("nn_tbrf_features");
            feature_computer_.compute_tbnn_features(velocity, k_local, omega_local,
                                                    features_, basis_);
        }

        // Step 2: Tree traversal + anisotropy construction (CPU)
        {
            TIMED_SCOPE("nn_tbrf_inference");

            const int n_basis_local = std::min(n_basis_, TensorBasis::NUM_BASIS);

            int idx = 0;
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    // Prepare scaled features for tree traversal
                    const auto& feat = features_[idx];
                    std::vector<double> scaled_feat(feat.values);

                    if (has_scaling_) {
                        int n_feat = std::min(static_cast<int>(scaled_feat.size()),
                                              static_cast<int>(input_means_.size()));
                        for (int f = 0; f < n_feat; ++f) {
                            scaled_feat[f] = (scaled_feat[f] - input_means_[f])
                                             / input_stds_[f];
                        }
                    }

                    // Traverse forests to get g_n coefficients
                    std::array<double, TensorBasis::NUM_BASIS> G = {};
                    for (int n = 0; n < n_basis_local; ++n) {
                        G[n] = predict_coefficient(n, scaled_feat.data());
                    }

                    // Construct anisotropy tensor: b_ij = sum_n G_n * T^(n)_ij
                    double b_xx, b_xy, b_yy;
                    TensorBasis::construct_anisotropy(G, basis_[idx],
                                                      b_xx, b_xy, b_yy);

                    // Convert to Reynolds stresses if requested
                    if (tau_ij) {
                        double k_val = k_local(i, j);
                        double tau_xx, tau_xy, tau_yy;
                        TensorBasis::anisotropy_to_reynolds_stress(
                            b_xx, b_xy, b_yy, k_val,
                            tau_xx, tau_xy, tau_yy);
                        tau_ij->xx(i, j) = tau_xx;
                        tau_ij->xy(i, j) = tau_xy;
                        tau_ij->yy(i, j) = tau_yy;
                    }

                    // Compute equivalent eddy viscosity from anisotropy
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
                            double b_mag = std::sqrt(b_xx * b_xx + 2.0 * b_xy * b_xy
                                                     + b_yy * b_yy);
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
#endif  // USE_GPU_OFFLOAD
}

} // namespace nncfd
