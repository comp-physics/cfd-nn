#pragma once

#include "turbulence_model.hpp"
#include "nn_core.hpp"
#include "features.hpp"
#include <memory>

namespace nncfd {

/// Neural network scalar eddy viscosity model
/// nu_t = NN(features)
class TurbulenceNNMLP : public TurbulenceModel {
public:
    TurbulenceNNMLP();
    
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
    
    void ensure_initialized(const Mesh& mesh);
};

} // namespace nncfd


