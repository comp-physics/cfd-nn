#pragma once

/// Turbulence transport equation solvers (k-ω, k-ε, etc.)
/// 
/// Design: Modular transport + closure architecture
/// - TransportModel: Solves PDEs for k, ω (or ε)
/// - ClosureModel: Computes ν_t (and optionally τ_ij) from k, ω, velocity gradients
/// 
/// This allows mixing different transport models with different closures:
/// - SST k-ω transport + linear Boussinesq closure
/// - SST k-ω transport + EARSM closure
/// - k-ε transport + nonlinear closure
/// etc.

#include "turbulence_model.hpp"
#include "features.hpp"
#include "gpu_utils.hpp"
#include <array>
#include <functional>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

namespace nncfd {

// ============================================================================
// Model Constants Structures (easily swappable)
// ============================================================================

/// SST k-ω model constants (Menter 1994)
struct SSTConstants {
    // Inner layer (k-ω) constants
    double sigma_k1 = 0.85;
    double sigma_omega1 = 0.5;
    double beta1 = 0.075;
    double alpha1 = 5.0 / 9.0;
    
    // Outer layer (k-ε transformed) constants  
    double sigma_k2 = 1.0;
    double sigma_omega2 = 0.856;
    double beta2 = 0.0828;
    double alpha2 = 0.44;
    
    // Common constants
    double beta_star = 0.09;    // C_mu equivalent
    double a1 = 0.31;           // SST limiter constant
    double kappa = 0.41;        // von Karman constant
    
    // Blending function constants
    double CD_omega_min = 1e-10;
    
    // Numerical limits
    double k_min = 1e-10;
    double omega_min = 1e-10;
    double k_max = 100.0;
    double omega_max = 1e8;
};

/// Standard k-ω model constants (Wilcox 1988)
struct KOmegaConstants {
    double sigma_k = 0.5;
    double sigma_omega = 0.5;
    double beta = 0.075;
    double beta_star = 0.09;
    double alpha = 5.0 / 9.0;
    
    double k_min = 1e-10;
    double omega_min = 1e-10;
    double k_max = 100.0;
    double omega_max = 1e8;
};

/// Standard k-ε model constants
struct KEpsilonConstants {
    double C_mu = 0.09;
    double C_eps1 = 1.44;
    double C_eps2 = 1.92;
    double sigma_k = 1.0;
    double sigma_eps = 1.3;
    
    double k_min = 1e-10;
    double eps_min = 1e-10;
    double k_max = 100.0;
    double eps_max = 1e8;
};

// ============================================================================
// Closure Model Interface (how ν_t is computed from k, ω)
// ============================================================================

/// Abstract closure model: given (k, ω, velocity gradients) → ν_t
class TurbulenceClosure {
public:
    virtual ~TurbulenceClosure() = default;
    
    /// Compute eddy viscosity (and optionally Reynolds stresses)
    /// @param mesh       Computational mesh
    /// @param velocity   Mean velocity field
    /// @param k          Turbulent kinetic energy
    /// @param omega      Specific dissipation rate
    /// @param nu_t       [out] Eddy viscosity
    /// @param tau_ij     [out] Reynolds stress tensor (optional)
    virtual void compute_nu_t(
        const Mesh& mesh,
        const VectorField& velocity,
        const ScalarField& k,
        const ScalarField& omega,
        ScalarField& nu_t,
        TensorField* tau_ij = nullptr
    ) = 0;
    
    /// Get closure name
    virtual std::string name() const = 0;
    
    /// Does this closure provide explicit Reynolds stresses?
    virtual bool provides_reynolds_stresses() const { return false; }
    
    /// Set laminar viscosity
    void set_nu(double nu) { nu_ = nu; }
    double nu() const { return nu_; }
    
    /// Set reference length scale
    void set_delta(double delta) { delta_ = delta; }
    double delta() const { return delta_; }

protected:
    double nu_ = 0.001;
    double delta_ = 1.0;
};

// ============================================================================
// Linear Eddy Viscosity Closures
// ============================================================================

/// Standard Boussinesq closure: ν_t = k / ω (or C_μ k² / ε)
class BoussinesqClosure : public TurbulenceClosure {
public:
    void compute_nu_t(
        const Mesh& mesh,
        const VectorField& velocity,
        const ScalarField& k,
        const ScalarField& omega,
        ScalarField& nu_t,
        TensorField* tau_ij = nullptr
    ) override;
    
    std::string name() const override { return "Boussinesq"; }
    
    void set_C_mu(double C_mu) { C_mu_ = C_mu; }
    
private:
    double C_mu_ = 0.09;
};

/// SST closure with strain-rate limiter: ν_t = a₁k / max(a₁ω, SF₂)
class SSTClosure : public TurbulenceClosure {
public:
    explicit SSTClosure(const SSTConstants& constants = SSTConstants())
        : constants_(constants) {}
    
    void compute_nu_t(
        const Mesh& mesh,
        const VectorField& velocity,
        const ScalarField& k,
        const ScalarField& omega,
        ScalarField& nu_t,
        TensorField* tau_ij = nullptr
    ) override;
    
    std::string name() const override { return "SST"; }
    
    /// Access constants for tuning
    SSTConstants& constants() { return constants_; }
    const SSTConstants& constants() const { return constants_; }
    
private:
    SSTConstants constants_;
    
    // Work arrays for GPU
    ScalarField dudx_, dudy_, dvdx_, dvdy_;
    bool initialized_ = false;
    
    void ensure_initialized(const Mesh& mesh);
    
    /// Compute F2 blending function
    double compute_F2(double k, double omega, double y_wall) const;
};

// ============================================================================
// SST k-ω Transport Model
// ============================================================================

/// SST k-ω two-equation transport model
/// Solves:
///   ∂k/∂t + u·∇k = P_k - β*kω + ∇·[(ν + σ_k ν_t)∇k]
///   ∂ω/∂t + u·∇ω = αS² - βω² + ∇·[(ν + σ_ω ν_t)∇ω] + CD_ω
class SSTKOmegaTransport : public TurbulenceModel {
public:
    explicit SSTKOmegaTransport(const SSTConstants& constants = SSTConstants());
    ~SSTKOmegaTransport();
    
    // Delete copy and move to prevent double-free of GPU buffers
    SSTKOmegaTransport(const SSTKOmegaTransport&) = delete;
    SSTKOmegaTransport& operator=(const SSTKOmegaTransport&) = delete;
    SSTKOmegaTransport(SSTKOmegaTransport&&) = delete;
    SSTKOmegaTransport& operator=(SSTKOmegaTransport&&) = delete;
    
    // TurbulenceModel interface
    void initialize(const Mesh& mesh, const VectorField& velocity) override;
    void initialize_gpu_buffers(const Mesh& mesh) override;
    void cleanup_gpu_buffers() override;
    bool is_gpu_ready() const override { return buffers_on_gpu_; }
    
    bool uses_transport_equations() const override { return true; }
    
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
    
    std::string name() const override { return "SSTKOmega"; }
    
    /// Set custom closure (default is SSTClosure)
    void set_closure(std::unique_ptr<TurbulenceClosure> closure);
    
    /// Access constants for tuning
    SSTConstants& constants() { return constants_; }
    const SSTConstants& constants() const { return constants_; }
    
    /// Wall boundary condition type
    enum class WallBC {
        Automatic,      // Use wall function or low-Re based on y+
        LowReynolds,    // Resolve viscous sublayer
        WallFunction    // Use wall functions
    };
    
    void set_wall_bc(WallBC bc) { wall_bc_ = bc; }
    
private:
    SSTConstants constants_;
    std::unique_ptr<TurbulenceClosure> closure_;
    WallBC wall_bc_ = WallBC::Automatic;
    
    // Work arrays
    ScalarField dudx_, dudy_, dvdx_, dvdy_;
    ScalarField P_k_;           // Production term
    ScalarField F1_, F2_;       // Blending functions
    ScalarField adv_k_, adv_omega_;
    ScalarField diff_k_, diff_omega_;
    ScalarField nu_k_, nu_omega_;  // Effective diffusivities
    ScalarField k_old_, omega_old_;  // Snapshot for Jacobi iteration (CPU/GPU consistency)
    ScalarField wall_dist_;     // Pre-computed wall distance for unified CPU/GPU kernel
    
    bool initialized_ = false;
    
    // GPU buffers (always present for ABI stability)
    std::vector<double> k_flat_, omega_flat_, nu_t_flat_;
    std::vector<double> u_flat_, v_flat_;
    std::vector<double> work_flat_;
    std::vector<double> wall_dist_flat_;
    bool gpu_ready_ = false;
    [[maybe_unused]] bool buffers_on_gpu_ = false;  // Track if buffers are actually mapped to GPU
    [[maybe_unused]] int cached_n_cells_ = 0;
    
    void allocate_gpu_buffers(const Mesh& mesh);
    void free_gpu_buffers();
    
    void ensure_initialized(const Mesh& mesh);
    void compute_velocity_gradients(const Mesh& mesh, const VectorField& velocity);
    void compute_blending_functions(const Mesh& mesh, const ScalarField& k, 
                                    const ScalarField& omega);
    void compute_production(const Mesh& mesh, const ScalarField& nu_t);
    
    void apply_wall_bc_k(const Mesh& mesh, ScalarField& k);
    void apply_wall_bc_omega(const Mesh& mesh, ScalarField& omega,
                             const ScalarField& k);
};

// ============================================================================
// Standard k-ω Transport Model (Wilcox)
// ============================================================================

/// Standard k-ω model (Wilcox 1988)
class KOmegaTransport : public TurbulenceModel {
public:
    explicit KOmegaTransport(const KOmegaConstants& constants = KOmegaConstants());
    
    void initialize(const Mesh& mesh, const VectorField& velocity) override;
    void initialize_gpu_buffers(const Mesh& mesh) override;
    void cleanup_gpu_buffers() override;
    bool is_gpu_ready() const override { return gpu_ready_; }
    
    bool uses_transport_equations() const override { return true; }
    
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
    
    std::string name() const override { return "KOmega"; }
    
    void set_closure(std::unique_ptr<TurbulenceClosure> closure);
    
private:
    KOmegaConstants constants_;
    std::unique_ptr<TurbulenceClosure> closure_;
    
    ScalarField dudx_, dudy_, dvdx_, dvdy_;
    ScalarField P_k_;
    bool initialized_ = false;
    bool gpu_ready_ = false;
    
    void ensure_initialized(const Mesh& mesh);
};

// ============================================================================
// GPU Kernels for Transport Equations
// ============================================================================

namespace transport_kernels {

/// Compute scalar advection term: u·∇φ
void compute_scalar_advection_gpu(
    const double* u, const double* v,      // Velocity (with ghost cells)
    const double* phi,                      // Scalar field (with ghost cells)
    double* adv,                            // Output advection term (interior only)
    int Nx, int Ny, int stride,
    double dx, double dy,
    bool use_upwind = true
);

/// Compute scalar diffusion term: ∇·(D∇φ)
void compute_scalar_diffusion_gpu(
    const double* phi,                      // Scalar field (with ghost cells)
    const double* D,                        // Diffusivity field (interior only)
    double* diff,                           // Output diffusion term (interior only)
    int Nx, int Ny, int stride,
    double dx, double dy
);

/// Compute SST production term: P_k = min(2ν_t|S|², 10β*kω)
void compute_production_gpu(
    const double* dudx, const double* dudy,
    const double* dvdx, const double* dvdy,
    const double* nu_t, const double* k, const double* omega,
    double* P_k,
    int n_cells,
    double beta_star
);

/// Compute SST blending functions F1, F2
void compute_blending_functions_gpu(
    const double* k, const double* omega,
    const double* wall_distance,
    const double* dudx, const double* dudy,
    const double* dvdx, const double* dvdy,
    double* F1, double* F2,
    int n_cells,
    double nu, double beta_star,
    double sigma_omega2, double CD_omega_min
);

/// Full SST k-ω transport step (fused kernel)
void sst_transport_step_gpu(
    // Velocity (with ghost cells)
    const double* u, const double* v,
    // Current k, omega (with ghost cells)
    double* k, double* omega,
    // Previous nu_t (interior only)
    const double* nu_t_prev,
    // Wall distance (interior only)
    const double* wall_distance,
    // Workspace (pre-allocated)
    double* workspace,
    // Output nu_t (interior only)
    double* nu_t,
    // Mesh parameters
    int Nx, int Ny, double dx, double dy,
    // Time step
    double dt,
    // Model constants
    double nu, double delta,
    const SSTConstants& constants
);

} // namespace transport_kernels

// ============================================================================
// Factory Functions
// ============================================================================

/// Create a transport model by name
std::unique_ptr<TurbulenceModel> create_transport_model(
    const std::string& name,
    const std::string& closure_name = "SST"
);

/// Create a closure by name
std::unique_ptr<TurbulenceClosure> create_closure(
    const std::string& name
);

} // namespace nncfd

