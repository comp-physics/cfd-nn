#pragma once

#include "turbulence_model.hpp"
#include "features.hpp"

namespace nncfd {

/// Gene Expression Programming (GEP) algebraic turbulence model
/// Based on Weatheritt & Sandberg (2016), JFM 807
/// "A novel evolutionary algorithm applied to algebraic modifications 
/// of the RANS stress-strain relationship"
///
/// This model uses algebraic expressions discovered via GEP to compute
/// corrections to the eddy viscosity. Unlike neural networks, this requires
/// no pre-trained weights - the formulas are fixed.
///
/// The model computes:
///   nu_t = C_mu * k^2 / epsilon * f(S, Omega)
/// where f is a correction function based on strain and rotation invariants.
class TurbulenceGEP : public TurbulenceModel {
public:
    TurbulenceGEP();
    ~TurbulenceGEP();
    
    // Delete copy/move to prevent double-free of GPU buffers
    TurbulenceGEP(const TurbulenceGEP&) = delete;
    TurbulenceGEP& operator=(const TurbulenceGEP&) = delete;
    TurbulenceGEP(TurbulenceGEP&&) = delete;
    TurbulenceGEP& operator=(TurbulenceGEP&&) = delete;
    
    std::string name() const override { return "GEP (Weatheritt-Sandberg)"; }
    
    void initialize(const Mesh& mesh, const VectorField& velocity) override;
    void initialize_gpu_buffers(const Mesh& mesh) override;
    void cleanup_gpu_buffers() override;
    bool is_gpu_ready() const override { return true; }  // GEP uses device_view, no internal GPU state
    
    void update(const Mesh& mesh,
                const VectorField& velocity,
                const ScalarField& k,
                const ScalarField& omega,
                ScalarField& nu_t,
                TensorField* tau_ij = nullptr,
                const TurbulenceDeviceView* device_view = nullptr) override;
    
    bool provides_reynolds_stresses() const override { return false; }
    
    void set_nu(double nu) { nu_ = nu; }
    void set_delta(double delta) { delta_ = delta; }
    void set_u_ref(double u_ref) { u_ref_ = u_ref; }
    void set_nu_t_max(double nu_t_max) { nu_t_max_ = nu_t_max; }
    
    /// Model variant selection
    enum class Variant {
        WS2016_Channel,    ///< Weatheritt-Sandberg 2016 channel flow model
        WS2016_PeriodicHill, ///< Weatheritt-Sandberg 2016 periodic hill model
        Simple             ///< Simplified algebraic model
    };
    
    void set_variant(Variant v) { variant_ = v; }
    
private:
    double nu_ = 0.001;
    double delta_ = 1.0;
    double u_ref_ = 1.0;
    double nu_t_max_ = 1.0;
    Variant variant_ = Variant::Simple;
    
    FeatureComputer feature_computer_;
    bool initialized_ = false;
    
    void ensure_initialized(const Mesh& mesh);
    
    /// Compute GEP correction factor based on invariants
    double compute_gep_factor(double I1_S, double I2_S, double I1_Omega, double I2_Omega,
                              double y_plus, double Re_d) const;
};

} // namespace nncfd

