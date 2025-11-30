#pragma once

#include "turbulence_model.hpp"
#include "nn_core.hpp"
#include "features.hpp"
#include <memory>

namespace nncfd {

/// TBNN-style neural network for Reynolds stress anisotropy
/// b_ij = sum_n G_n(lambda) * T^(n)_ij(S, Omega)
/// where G_n are NN outputs and T^(n) are tensor basis functions
class TurbulenceNNTBNN : public TurbulenceModel {
public:
    TurbulenceNNTBNN();
    
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
    
private:
    MLP mlp_;
    FeatureComputer feature_computer_;
    
    double delta_ = 1.0;
    double u_ref_ = 1.0;
    double k_min_ = 1e-10;  // Minimum k to avoid division by zero
    
    // Work buffers
    std::vector<Features> features_;
    std::vector<std::array<std::array<double, 3>, TensorBasis::NUM_BASIS>> basis_;
    std::vector<double> buffer1_, buffer2_;
    
    void ensure_initialized(const Mesh& mesh);
    
    /// Estimate k field from velocity gradient (simple algebraic model)
    void estimate_k(const Mesh& mesh, const VectorField& velocity, ScalarField& k);
};

} // namespace nncfd


