#pragma once

#include "turbulence_model.hpp"
#include "features.hpp"

namespace nncfd {

/// Simple algebraic eddy viscosity model (mixing length)
/// nu_t = (kappa * y)^2 * |S| with van Driest damping
class MixingLengthModel : public TurbulenceModel {
public:
    MixingLengthModel();
    
    void update(
        const Mesh& mesh,
        const VectorField& velocity,
        const ScalarField& k,
        const ScalarField& omega,
        ScalarField& nu_t,
        TensorField* tau_ij = nullptr
    ) override;
    
    std::string name() const override { return "MixingLength"; }
    
    /// Set model constants
    void set_kappa(double kappa) { kappa_ = kappa; }
    void set_A_plus(double A_plus) { A_plus_ = A_plus; }
    void set_delta(double delta) { delta_ = delta; }
    
private:
    double kappa_ = 0.41;   ///< von Karman constant
    double A_plus_ = 26.0;  ///< van Driest damping constant
    double delta_ = 1.0;    ///< Channel half-height
    
    ScalarField dudx_, dudy_, dvdx_, dvdy_;
};

/// Simple k-omega model (without transport equations for now)
/// Uses algebraic relations to estimate k and omega, then computes nu_t
class AlgebraicKOmegaModel : public TurbulenceModel {
public:
    AlgebraicKOmegaModel();
    
    void update(
        const Mesh& mesh,
        const VectorField& velocity,
        const ScalarField& k,
        const ScalarField& omega,
        ScalarField& nu_t,
        TensorField* tau_ij = nullptr
    ) override;
    
    std::string name() const override { return "AlgebraicKOmega"; }
    
    void set_delta(double delta) { delta_ = delta; }
    void set_C_mu(double C_mu) { C_mu_ = C_mu; }
    
private:
    double delta_ = 1.0;
    double C_mu_ = 0.09;
    
    ScalarField dudx_, dudy_, dvdx_, dvdy_;
};

} // namespace nncfd


