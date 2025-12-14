#pragma once

#include "turbulence_model.hpp"
#include "nn_core.hpp"
#include "features.hpp"
#include <memory>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

namespace nncfd {

/// Neural network scalar eddy viscosity model
/// nu_t = NN(features)
/// 
/// GPU Strategy:
/// - NN weights uploaded once and kept on GPU
/// - Features computed on CPU, uploaded in batch
/// - NN inference runs on GPU for all grid cells in parallel
class TurbulenceNNMLP : public TurbulenceModel {
public:
    TurbulenceNNMLP();
    ~TurbulenceNNMLP();
    
    // Delete copy and move to prevent double-free of GPU buffers
    TurbulenceNNMLP(const TurbulenceNNMLP&) = delete;
    TurbulenceNNMLP& operator=(const TurbulenceNNMLP&) = delete;
    TurbulenceNNMLP(TurbulenceNNMLP&&) = delete;
    TurbulenceNNMLP& operator=(TurbulenceNNMLP&&) = delete;
    
    /// Load network weights and scaling from directory
    void load(const std::string& weights_dir, const std::string& scaling_dir);
    
    void update(
        const Mesh& mesh,
        const VectorField& velocity,
        const ScalarField& k,
        const ScalarField& omega,
        ScalarField& nu_t,
        TensorField* tau_ij = nullptr,
        const TurbulenceDeviceView* device_view = nullptr
    ) override;
    
    std::string name() const override { return "NNMLP"; }
    
    /// Configuration
    void set_nu_t_max(double val) { nu_t_max_ = val; }
    void set_blend_alpha(double alpha) { blend_alpha_ = alpha; }
    void set_delta(double delta) { delta_ = delta; }
    void set_u_ref(double u_ref) { u_ref_ = u_ref; }
    
    /// Enable baseline blending
    void enable_baseline_blend(bool enable) { blend_with_baseline_ = enable; }
    
    /// Access the MLP
    const MLP& mlp() const { return mlp_; }
    
    /// Upload NN weights to GPU (call after load())
    void upload_to_gpu();
    
    /// GPU buffer management (override from base)
    void initialize_gpu_buffers(const Mesh& mesh) override;
    void cleanup_gpu_buffers() override;
    bool is_gpu_ready() const override { return gpu_ready_; }
    
private:
    MLP mlp_;
    FeatureComputer feature_computer_;
    
    double nu_t_max_ = 1.0;
    double blend_alpha_ = 1.0;  // 1 = pure NN, 0 = pure baseline
    double delta_ = 1.0;
    double u_ref_ = 1.0;
    bool blend_with_baseline_ = false;
    
    // Baseline model for blending
    std::unique_ptr<TurbulenceModel> baseline_;
    
    // Work buffers to avoid allocation in update()
    std::vector<Features> features_;
    std::vector<double> buffer1_, buffer2_;
    ScalarField baseline_nu_t_;
    
    // GPU batching buffers
    std::vector<double> features_flat_;
    std::vector<double> outputs_flat_;
    std::vector<double> workspace_;
    
    // GPU state
    bool gpu_ready_ = false;
    bool initialized_ = false;
    [[maybe_unused]] bool buffers_on_gpu_ = false;  // Track if buffers are actually mapped to GPU
    [[maybe_unused]] int cached_n_cells_ = 0;
    
    void ensure_initialized(const Mesh& mesh);
    void allocate_gpu_buffers(int n_cells);
    void free_gpu_buffers();
};

} // namespace nncfd
