#pragma once

#include "mesh.hpp"
#include "fields.hpp"
#include "config.hpp"
#include <memory>
#include <string>

namespace nncfd {

/// Abstract base class for turbulence closures
class TurbulenceModel {
public:
    virtual ~TurbulenceModel() = default;
    
    /// Update turbulent quantities given current mean flow
    /// @param mesh      Computational mesh
    /// @param velocity  Mean velocity field
    /// @param k         Turbulent kinetic energy (optional, may be unused)
    /// @param omega     Specific dissipation rate (optional, may be unused)
    /// @param nu_t      [out] Eddy viscosity field
    /// @param tau_ij    [out] Reynolds stress tensor (optional, for TBNN)
    virtual void update(
        const Mesh& mesh,
        const VectorField& velocity,
        const ScalarField& k,
        const ScalarField& omega,
        ScalarField& nu_t,
        TensorField* tau_ij = nullptr
    ) = 0;
    
    /// Get model name
    virtual std::string name() const = 0;
    
    /// Check if model provides explicit Reynolds stresses (vs. eddy viscosity only)
    virtual bool provides_reynolds_stresses() const { return false; }
    
    /// Does this model solve transport equations (e.g., k-ω, k-ε)?
    /// If true, advance_turbulence() should be called each time step.
    virtual bool uses_transport_equations() const { return false; }
    
    /// Advance turbulence transport variables (k, ω, etc.) by one time step.
    /// Default: no-op (for algebraic / NN models).
    /// @param mesh       Computational mesh
    /// @param velocity   Mean velocity field
    /// @param dt         Time step size
    /// @param k          [in/out] Turbulent kinetic energy
    /// @param omega      [in/out] Specific dissipation rate
    /// @param nu_t_prev  Eddy viscosity from previous step (for diffusion coefficients)
    virtual void advance_turbulence(
        const Mesh& mesh,
        const VectorField& velocity,
        double dt,
        ScalarField& k,
        ScalarField& omega,
        const ScalarField& nu_t_prev
    ) {
        (void)mesh;
        (void)velocity;
        (void)dt;
        (void)k;
        (void)omega;
        (void)nu_t_prev;
    }
    
    /// Initialize any model-specific fields (e.g., k, omega for transport models)
    virtual void initialize(const Mesh& /*mesh*/, const VectorField& /*velocity*/) {}
    
    /// Set laminar viscosity (needed for some models)
    void set_nu(double nu) { nu_ = nu; }
    double nu() const { return nu_; }
    
    /// Set reference length scale (channel half-height, etc.)
    void set_delta(double delta) { delta_ = delta; }
    double delta() const { return delta_; }
    
protected:
    double nu_ = 0.001;    ///< Laminar viscosity
    double delta_ = 1.0;   ///< Reference length scale
};

/// Factory function to create turbulence model from config
std::unique_ptr<TurbulenceModel> create_turbulence_model(
    TurbulenceModelType type,
    const std::string& weights_path = "",
    const std::string& scaling_path = ""
);

} // namespace nncfd

