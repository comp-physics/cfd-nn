#pragma once

#include "turbulence_model.hpp"
#include "features.hpp"
#include <memory>
#include <vector>
#include <string>
#include <cstdint>

namespace nncfd {

/// Tree offset entry: maps (basis_idx, tree_idx) to a contiguous node range
struct TreeOffset {
    int basis_idx;
    int tree_idx;
    int start_node;
    int n_nodes;
};

/// TBRF (Tensor Basis Random Forest) turbulence model
/// b_ij = sum_n g_n(lambda) * T^(n)_ij(S, Omega)
/// where g_n are random forest predictions and T^(n) are tensor basis functions.
///
/// Following Kaandorp & Dwight (2020), Computers & Fluids 202.
///
/// CPU-only tree traversal: decision trees are branch-heavy and poorly
/// suited for GPU execution. Feature computation (invariants + tensor basis)
/// reuses the same code as TBNN.
class TurbulenceNNTBRF : public TurbulenceModel {
public:
    TurbulenceNNTBRF();
    ~TurbulenceNNTBRF() override;

    // Delete copy and move to prevent double-free
    TurbulenceNNTBRF(const TurbulenceNNTBRF&) = delete;
    TurbulenceNNTBRF& operator=(const TurbulenceNNTBRF&) = delete;
    TurbulenceNNTBRF(TurbulenceNNTBRF&&) = delete;
    TurbulenceNNTBRF& operator=(TurbulenceNNTBRF&&) = delete;

    /// Load tree weights and scaling from directory
    /// Expects: trees.bin, tree_offsets.txt, input_means.txt, input_stds.txt
    void load(const std::string& model_dir);

    void update(
        const Mesh& mesh,
        const VectorField& velocity,
        const ScalarField& k,
        const ScalarField& omega,
        ScalarField& nu_t,
        TensorField* tau_ij = nullptr,
        const TurbulenceDeviceView* device_view = nullptr
    ) override;

    std::string name() const override { return "NNTBRF"; }

    bool provides_reynolds_stresses() const override { return true; }

    /// Configuration
    void set_delta(double delta) { delta_ = delta; }
    void set_u_ref(double u_ref) { u_ref_ = u_ref; }
    void set_k_min(double k_min) { k_min_ = k_min; }

    /// GPU buffer management (no-op for TBRF: tree traversal is CPU-only)
    void initialize_gpu_buffers(const Mesh& mesh) override { (void)mesh; }
    void cleanup_gpu_buffers() override {}
    bool is_gpu_ready() const override { return false; }

    /// Access tree metadata
    int total_nodes() const { return total_nodes_; }
    int n_basis() const { return n_basis_; }
    int n_trees() const { return n_trees_; }

private:
    // Tree data arrays (flat, loaded from trees.bin)
    std::vector<int32_t> children_left_;
    std::vector<int32_t> children_right_;
    std::vector<int32_t> feature_;
    std::vector<float> threshold_;
    std::vector<float> value_;

    // Tree offset table
    std::vector<TreeOffset> tree_offsets_;

    // Header metadata
    int total_nodes_ = 0;
    int n_basis_ = 0;
    int n_trees_ = 0;

    // Input normalization
    std::vector<double> input_means_;
    std::vector<double> input_stds_;
    bool has_scaling_ = false;

    // Feature computation
    FeatureComputer feature_computer_;
    bool initialized_ = false;

    // Reference quantities
    double delta_ = 1.0;
    double u_ref_ = 1.0;
    double k_min_ = 1e-10;

    // Work buffers
    std::vector<Features> features_;
    std::vector<std::array<std::array<double, 3>, TensorBasis::NUM_BASIS>> basis_;

    void ensure_initialized(const Mesh& mesh);

    /// Estimate k field from velocity gradient (same as TBNN)
    void estimate_k(const Mesh& mesh, const VectorField& velocity, ScalarField& k);

    /// Traverse a single tree starting at start_node, return leaf value
    float traverse_tree(const double* features, int start_node) const;

    /// Predict g_n coefficient for a given basis index by averaging all trees
    double predict_coefficient(int basis_idx, const double* features) const;
};

} // namespace nncfd
