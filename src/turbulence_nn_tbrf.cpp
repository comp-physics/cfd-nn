/// @file turbulence_nn_tbrf.cpp
/// @brief TBRF (Tensor Basis Random Forest) turbulence model implementation
///
/// Implements Kaandorp & Dwight (2020) random forest approach for predicting
/// Reynolds stress anisotropy coefficients g_n in the tensor basis expansion:
///   b_ij = sum_n g_n(lambda) * T^(n)_ij(S_hat, Omega_hat)
///
/// Tree traversal is CPU-only (branching is poorly suited for GPU).
/// Feature computation reuses the same invariants and tensor basis as TBNN.

#include "turbulence_nn_tbrf.hpp"
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

TurbulenceNNTBRF::~TurbulenceNNTBRF() = default;

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
// Tree traversal
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
// Initialization helpers
// ============================================================================

void TurbulenceNNTBRF::ensure_initialized(const Mesh& mesh) {
    if (!initialized_) {
        feature_computer_ = FeatureComputer(mesh);
        feature_computer_.set_reference(nu_, delta_, u_ref_);

        int n_interior = mesh.Nx * mesh.Ny;
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

    (void)device_view;  // TBRF is CPU-only

    ensure_initialized(mesh);
    feature_computer_.set_reference(nu_, delta_, u_ref_);

    // Use provided k/omega or estimate
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

} // namespace nncfd
