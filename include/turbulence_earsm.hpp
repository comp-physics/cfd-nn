#pragma once

/// Explicit Algebraic Reynolds Stress Models (EARSM)
/// 
/// EARSM models compute the Reynolds stress anisotropy tensor as:
///   b_ij = Σ_n G_n(invariants) × T^(n)_ij(S, Ω)
/// 
/// where:
///   - G_n are scalar coefficients (functions of tensor invariants)
///   - T^(n) are tensor basis functions
/// 
/// This gives a nonlinear eddy viscosity model that captures:
///   - Streamline curvature effects
///   - Secondary flows
///   - Anisotropy in normal stresses
/// 
/// Implemented models:
///   - Wallin-Johansson (2000) - derived from pressure-strain model
///   - Gatski-Speziale (1993) - regularized version
///   - Pope (1975) - quadratic model

#include "turbulence_model.hpp"
#include "turbulence_transport.hpp"
#include "features.hpp"
#include <array>
#include <functional>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

namespace nncfd {

// ============================================================================
// EARSM Model Types
// ============================================================================

enum class EARSMType {
    WallinJohansson2000,    // Full WJ EARSM (recommended)
    GatskiSpeziale1993,     // Regularized EARSM
    Pope1975,               // Quadratic model
    Shih1993,               // Nonlinear k-ε
    Custom                  // User-defined G_n functions
};

// ============================================================================
// EARSM Constants
// ============================================================================

/// Constants for Wallin-Johansson EARSM
struct WJConstants {
    // Pressure-strain model constants (from SSG model)
    double C1 = 1.8;
    double C1_star = 0.5;
    double C2 = 0.36;
    double C3 = 1.25;
    double C3_star = 0.4;
    double C4 = 0.4;
    double C5 = 1.88;
    
    // Derived constants for explicit solution
    double A1() const { return 4.0/5.0 - C2/2.0; }
    double A2() const { return 2.0 - C4/2.0; }
    double A3() const { return 2.0 - C3/2.0; }
    double A4() const { return 2.0 * C5 - 1.0; }
};

/// Constants for Gatski-Speziale EARSM
struct GSConstants {
    double C_mu = 0.09;
    double C1 = 1.8;
    double C2 = 0.6;
    double C3 = 0.6;
    double C4 = 0.6;
    double C5 = 0.2;
    
    // Regularization parameter
    double eta_max = 10.0;
};

// ============================================================================
// EARSM Closure Base Class
// ============================================================================

/// Abstract EARSM closure: computes G_n coefficients from invariants
class EARSMClosure : public TurbulenceClosure {
public:
    EARSMClosure();
    virtual ~EARSMClosure() = default;
    
    // Delete copy and move to prevent double-free of GPU buffers
    EARSMClosure(const EARSMClosure&) = delete;
    EARSMClosure& operator=(const EARSMClosure&) = delete;
    EARSMClosure(EARSMClosure&&) = delete;
    EARSMClosure& operator=(EARSMClosure&&) = delete;
    
    bool provides_reynolds_stresses() const override { return true; }
    
    void initialize_gpu_buffers(const Mesh& mesh);
    void cleanup_gpu_buffers();
    bool is_gpu_ready() const { return buffers_on_gpu_; }
    
    void compute_nu_t(
        const Mesh& mesh,
        const VectorField& velocity,
        const ScalarField& k,
        const ScalarField& omega,
        ScalarField& nu_t,
        TensorField* tau_ij = nullptr
    ) override;
    
    /// Get EARSM type
    virtual EARSMType type() const = 0;
    
protected:
    /// Compute G coefficients given invariants
    /// @param eta   Normalized strain invariant: η = (k/ε) × |S|
    /// @param zeta  Normalized rotation invariant: ζ = (k/ε) × |Ω|
    /// @param G     [out] Tensor basis coefficients (4 for 2D)
    virtual void compute_G(
        double eta, double zeta,
        std::array<double, TensorBasis::NUM_BASIS>& G
    ) const = 0;
    
    // Work arrays (initialized lazily)
    std::unique_ptr<FeatureComputer> feature_computer_;
    std::vector<Features> features_;
    std::vector<std::array<std::array<double, 3>, TensorBasis::NUM_BASIS>> basis_;
    bool initialized_ = false;
    
    void ensure_initialized(const Mesh& mesh);
    
    // GPU buffers (always present for ABI stability)
    std::vector<double> k_flat_, omega_flat_;
    std::vector<double> u_flat_, v_flat_;
    std::vector<double> wall_dist_flat_;
    std::vector<double> nu_t_flat_;
    std::vector<double> tau_flat_;
    std::vector<double> work_flat_;
    bool gpu_ready_ = false;
    bool buffers_on_gpu_ = false;  // Track if buffers are actually mapped to GPU
    
    void allocate_gpu_buffers(const Mesh& mesh);
    void free_gpu_buffers();
    void compute_nu_t_gpu(
        const Mesh& mesh,
        const VectorField& velocity,
        const ScalarField& k,
        const ScalarField& omega,
        ScalarField& nu_t,
        TensorField* tau_ij
    );
};

// ============================================================================
// Wallin-Johansson EARSM (2000)
// ============================================================================

/// Wallin-Johansson Explicit Algebraic Reynolds Stress Model
/// 
/// Derived from the SSG pressure-strain model with weak equilibrium assumption.
/// Provides improved predictions for:
///   - Rotating channel flow
///   - Backward-facing step
///   - Separated flows
/// 
/// Reference: Wallin & Johansson (2000), Phys. Fluids 12(11)
class WallinJohanssonEARSM : public EARSMClosure {
public:
    explicit WallinJohanssonEARSM(const WJConstants& constants = WJConstants())
        : constants_(constants) {}
    
    std::string name() const override { return "WJ-EARSM"; }
    EARSMType type() const override { return EARSMType::WallinJohansson2000; }
    
    WJConstants& constants() { return constants_; }
    const WJConstants& constants() const { return constants_; }
    
protected:
    void compute_G(
        double eta, double zeta,
        std::array<double, TensorBasis::NUM_BASIS>& G
    ) const override;
    
private:
    WJConstants constants_;
    
    /// Solve cubic equation for N (effective viscosity parameter)
    double solve_for_N(double II_S, double II_Omega) const;
};

// ============================================================================
// Gatski-Speziale EARSM (1993)
// ============================================================================

/// Gatski-Speziale Regularized EARSM
/// 
/// A regularized version that avoids singularities in the cubic solution.
/// Good for general-purpose use.
/// 
/// Reference: Gatski & Speziale (1993), J. Fluid Mech. 254
class GatskiSpezialeEARSM : public EARSMClosure {
public:
    explicit GatskiSpezialeEARSM(const GSConstants& constants = GSConstants())
        : constants_(constants) {}
    
    std::string name() const override { return "GS-EARSM"; }
    EARSMType type() const override { return EARSMType::GatskiSpeziale1993; }
    
    GSConstants& constants() { return constants_; }
    const GSConstants& constants() const { return constants_; }
    
protected:
    void compute_G(
        double eta, double zeta,
        std::array<double, TensorBasis::NUM_BASIS>& G
    ) const override;
    
private:
    GSConstants constants_;
};

// ============================================================================
// Pope Quadratic Model (1975)
// ============================================================================

/// Pope's Quadratic Nonlinear Eddy Viscosity Model
/// 
/// A simple quadratic extension of the Boussinesq hypothesis:
///   b_ij = -C_μ (k/ε) S_ij + C_1 (k/ε)² [S_ik S_kj - (1/3)S²δ_ij]
///        + C_2 (k/ε)² [Ω_ik S_kj + S_ik Ω_kj]
/// 
/// Reference: Pope (1975), J. Fluid Mech. 72
class PopeQuadraticEARSM : public EARSMClosure {
public:
    PopeQuadraticEARSM(double C1 = 0.1, double C2 = 0.1)
        : C1_(C1), C2_(C2) {}
    
    std::string name() const override { return "Pope-Quadratic"; }
    EARSMType type() const override { return EARSMType::Pope1975; }
    
    void set_C1(double C1) { C1_ = C1; }
    void set_C2(double C2) { C2_ = C2; }
    
protected:
    void compute_G(
        double eta, double zeta,
        std::array<double, TensorBasis::NUM_BASIS>& G
    ) const override;
    
private:
    double C1_, C2_;
};

// ============================================================================
// Custom EARSM (user-defined)
// ============================================================================

/// Custom EARSM with user-defined G_n functions
/// 
/// Allows defining custom coefficient functions for research/experimentation.
class CustomEARSM : public EARSMClosure {
public:
    using GFunction = std::function<void(double eta, double zeta, 
                                         std::array<double, TensorBasis::NUM_BASIS>&)>;
    
    explicit CustomEARSM(GFunction g_func, const std::string& name = "Custom-EARSM")
        : g_func_(std::move(g_func)), name_(name) {}
    
    std::string name() const override { return name_; }
    EARSMType type() const override { return EARSMType::Custom; }
    
protected:
    void compute_G(
        double eta, double zeta,
        std::array<double, TensorBasis::NUM_BASIS>& G
    ) const override {
        g_func_(eta, zeta, G);
    }
    
private:
    GFunction g_func_;
    std::string name_;
};

// ============================================================================
// EARSM + Transport Model Combination
// ============================================================================

/// SST k-ω transport with EARSM closure
/// 
/// Combines the robustness of SST k-ω transport equations with
/// the improved physics of EARSM closure.
class SSTWithEARSM : public TurbulenceModel {
public:
    explicit SSTWithEARSM(
        EARSMType earsm_type = EARSMType::WallinJohansson2000,
        const SSTConstants& sst_constants = SSTConstants()
    );
    
    void initialize(const Mesh& mesh, const VectorField& velocity) override;
    bool uses_transport_equations() const override { return true; }
    bool provides_reynolds_stresses() const override { return true; }
    
    void advance_turbulence(
        const Mesh& mesh,
        const VectorField& velocity,
        double dt,
        ScalarField& k,
        ScalarField& omega,
        const ScalarField& nu_t_prev,
        const TurbulenceDeviceView* device_view = nullptr
    ) override;
    
    void update(
        const Mesh& mesh,
        const VectorField& velocity,
        const ScalarField& k,
        const ScalarField& omega,
        ScalarField& nu_t,
        TensorField* tau_ij = nullptr,
        const TurbulenceDeviceView* device_view = nullptr
    ) override;
    
    std::string name() const override;
    
    /// Access underlying models
    SSTKOmegaTransport& transport() { return transport_; }
    EARSMClosure& closure() { return *closure_; }
    
private:
    SSTKOmegaTransport transport_;
    std::unique_ptr<EARSMClosure> closure_;
};

// ============================================================================
// GPU Kernels for EARSM
// ============================================================================

namespace earsm_kernels {

/// Compute EARSM G coefficients for all cells (Wallin-Johansson)
void compute_wj_coefficients_gpu(
    const double* dudx, const double* dudy,
    const double* dvdx, const double* dvdy,
    const double* k, const double* omega,
    double* G,                              // Output: n_cells * 4
    int n_cells,
    const WJConstants& constants
);

/// Compute EARSM G coefficients for all cells (Gatski-Speziale)
void compute_gs_coefficients_gpu(
    const double* dudx, const double* dudy,
    const double* dvdx, const double* dvdy,
    const double* k, const double* omega,
    double* G,                              // Output: n_cells * 4
    int n_cells,
    const GSConstants& constants
);

/// Full EARSM pipeline: gradients → invariants → G → b_ij → ν_t
void earsm_full_pipeline_gpu(
    // Velocity (with ghost cells)
    const double* u, const double* v,
    // Turbulence quantities (interior only)
    const double* k, const double* omega,
    // Workspace (pre-allocated)
    double* workspace,
    // Output
    double* nu_t,
    double* tau_xx, double* tau_xy, double* tau_yy,  // Can be nullptr
    // Mesh parameters
    int Nx, int Ny, double dx, double dy,
    // Model parameters
    double nu, double delta,
    EARSMType type
);

} // namespace earsm_kernels

// ============================================================================
// Factory Functions
// ============================================================================

/// Create an EARSM closure by type
std::unique_ptr<EARSMClosure> create_earsm_closure(EARSMType type);

/// Create an EARSM closure by name
std::unique_ptr<EARSMClosure> create_earsm_closure(const std::string& name);

} // namespace nncfd

