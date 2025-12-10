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
/// GPU Strategy (Full Offload):
/// - NN weights uploaded once and kept on GPU
/// - Velocity field uploaded per timestep
/// - ALL computation on GPU: gradients, features, tensor basis, NN inference, postprocessing
/// - Only nu_t (and optionally tau_ij) downloaded back
class TurbulenceNNTBNN : public TurbulenceModel {
public:
    TurbulenceNNTBNN();
    ~TurbulenceNNTBNN();
    
    // Delete copy and move to prevent double-free of GPU buffers
    TurbulenceNNTBNN(const TurbulenceNNTBNN&) = delete;
    TurbulenceNNTBNN& operator=(const TurbulenceNNTBNN&) = delete;
    TurbulenceNNTBNN(TurbulenceNNTBNN&&) = delete;
    TurbulenceNNTBNN& operator=(TurbulenceNNTBNN&&) = delete;
    
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
    
    // Work buffers (CPU fallback)
    std::vector<Features> features_;
    std::vector<std::array<std::array<double, 3>, TensorBasis::NUM_BASIS>> basis_;
    
    // Flattened buffers for GPU batching (legacy path)
    std::vector<double> features_flat_;
    std::vector<double> outputs_flat_;
    std::vector<double> workspace_;
    
    // Full GPU pipeline buffers
    std::vector<double> u_flat_;           // Velocity u (with ghost cells)
    std::vector<double> v_flat_;           // Velocity v (with ghost cells)
    std::vector<double> k_flat_;           // TKE (interior only)
    std::vector<double> omega_flat_;       // Omega (interior only)
    std::vector<double> wall_dist_flat_;   // Wall distance (interior only)
    std::vector<double> nu_t_flat_;        // Output nu_t (interior only)
    std::vector<double> tau_xx_flat_;      // Output tau_xx (interior only)
    std::vector<double> tau_xy_flat_;      // Output tau_xy (interior only)
    std::vector<double> tau_yy_flat_;      // Output tau_yy (interior only)
    std::vector<double> full_workspace_;   // GPU workspace for full pipeline
    
    // GPU state
    bool gpu_ready_ = false;
    bool full_gpu_ready_ = false;  // Reserved for future GPU optimization
    bool buffers_on_gpu_ = false;  // Track if feature buffers are mapped to GPU
    bool full_buffers_on_gpu_ = false;  // Track if full pipeline buffers are mapped to GPU
    bool initialized_ = false;
    int cached_n_cells_ = 0;  // Reserved for future GPU optimization
    int cached_total_cells_ = 0;
    
    void ensure_initialized(const Mesh& mesh);
    void allocate_gpu_buffers(int n_cells);
    void allocate_full_gpu_buffers(const Mesh& mesh);
    void free_gpu_buffers();
    void free_full_gpu_buffers();
    
    /// Update using full GPU pipeline (all computation on GPU)
    void update_full_gpu(
        const Mesh& mesh,
        const VectorField& velocity,
        const ScalarField& k,
        const ScalarField& omega,
        ScalarField& nu_t,
        TensorField* tau_ij
    );
    
    /// Estimate k field from velocity gradient (simple algebraic model)
    void estimate_k(const Mesh& mesh, const VectorField& velocity, ScalarField& k);
};

} // namespace nncfd
