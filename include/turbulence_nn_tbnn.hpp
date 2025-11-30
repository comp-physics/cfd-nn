#pragma once

#include "turbulence_model.hpp"
#include "nn_core.hpp"
#include "features.hpp"
#include <memory>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

namespace nncfd {

/// TBNN-style neural network for Reynolds stress anisotropy
/// b_ij = sum_n G_n(lambda) * T^(n)_ij(S, Omega)
/// where G_n are NN outputs and T^(n) are tensor basis functions
///
/// GPU Strategy:
/// - NN weights uploaded once and kept on GPU
/// - Features computed on CPU, uploaded in batch
/// - NN inference runs on GPU for all grid cells in parallel
/// - Results downloaded for post-processing
class TurbulenceNNTBNN : public TurbulenceModel {
public:
    TurbulenceNNTBNN();
    ~TurbulenceNNTBNN();
    
    /// Load network weights and scaling from directory
    void load(const std::string& weights_dir, const std::string& scaling_dir);
    
    void update(
        const Mesh& mesh,
        const VectorField& velocity,
        const ScalarField& k,
        const ScalarField& omega,
        ScalarField& nu_t,
        TensorField* tau_ij = nullptr
    ) override;
    
    std::string name() const override { return "NNTBNN"; }
    
    bool provides_reynolds_stresses() const override { return true; }
    
    /// Configuration
    void set_delta(double delta) { delta_ = delta; }
    void set_u_ref(double u_ref) { u_ref_ = u_ref; }
    void set_k_min(double k_min) { k_min_ = k_min; }
    
    /// Access the MLP
    const MLP& mlp() const { return mlp_; }
    
    /// Upload NN weights to GPU (call after load())
    void upload_to_gpu();
    
    /// Check if GPU is ready
    bool is_gpu_ready() const { return gpu_ready_; }
    
private:
    MLP mlp_;
    FeatureComputer feature_computer_;
    
    double delta_ = 1.0;
    double u_ref_ = 1.0;
    double k_min_ = 1e-10;  // Minimum k to avoid division by zero
    
    // Work buffers (CPU)
    std::vector<Features> features_;
    std::vector<std::array<std::array<double, 3>, TensorBasis::NUM_BASIS>> basis_;
    
    // Flattened buffers for GPU batching
    std::vector<double> features_flat_;
    std::vector<double> outputs_flat_;
    std::vector<double> workspace_;
    
    // GPU state
    bool gpu_ready_ = false;
    bool initialized_ = false;
    int cached_n_cells_ = 0;
    
    void ensure_initialized(const Mesh& mesh);
    void allocate_gpu_buffers(int n_cells);
    void free_gpu_buffers();
    
    /// Estimate k field from velocity gradient (simple algebraic model)
    void estimate_k(const Mesh& mesh, const VectorField& velocity, ScalarField& k);
};

} // namespace nncfd
