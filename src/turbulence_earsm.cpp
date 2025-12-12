#include "turbulence_earsm.hpp"
#include "timing.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <complex>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

namespace nncfd {

// ============================================================================
// EARSM Closure Base Implementation
// ============================================================================

EARSMClosure::EARSMClosure() = default;

void EARSMClosure::initialize_gpu_buffers(const Mesh& mesh) {
#ifdef USE_GPU_OFFLOAD
    if (omp_get_num_devices() == 0) {
        buffers_on_gpu_ = false;
        return;
    }
    
    // Check if already allocated
    if (buffers_on_gpu_ && !k_flat_.empty()) {
        return;
    }
    
    // Free old buffers if they exist
    free_gpu_buffers();
    
    // Allocate GPU buffers
    allocate_gpu_buffers(mesh);
#else
    (void)mesh;
    buffers_on_gpu_ = false;
#endif
}

void EARSMClosure::cleanup_gpu_buffers() {
#ifdef USE_GPU_OFFLOAD
    free_gpu_buffers();
#endif
    buffers_on_gpu_ = false;
}

void EARSMClosure::ensure_initialized(const Mesh& mesh) {
    if (!initialized_) {
        feature_computer_ = std::make_unique<FeatureComputer>(mesh);
        feature_computer_->set_reference(nu_, delta_, 1.0);
        
        int n_cells = mesh.Nx * mesh.Ny;
        features_.resize(n_cells);
        basis_.resize(n_cells);
        
        // GPU buffers are initialized explicitly through initialize_gpu_buffers()
        
        initialized_ = true;
    }
}

#ifdef USE_GPU_OFFLOAD
void EARSMClosure::allocate_gpu_buffers(const Mesh& mesh) {
    int n_interior = mesh.Nx * mesh.Ny;
    int n_total = (mesh.Nx + 2) * (mesh.Ny + 2);
    
    // Check if already allocated
    if (buffers_on_gpu_ && !k_flat_.empty()) {
        return;  // Already allocated and mapped
    }
    
    // Free old buffers if they exist
    free_gpu_buffers();
    
    // Allocate CPU buffers
    k_flat_.resize(n_interior);
    omega_flat_.resize(n_interior);
    u_flat_.resize(n_total);
    v_flat_.resize(n_total);
    wall_dist_flat_.resize(n_interior);
    nu_t_flat_.resize(n_interior);
    tau_flat_.resize(n_interior * 3);  // xx, xy, yy
    
    // Workspace: gradients (4), features (5), basis (12), G (4), b (3)
    work_flat_.resize(n_interior * 28);
    
    // Precompute wall distances
    int idx = 0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            wall_dist_flat_[idx++] = mesh.wall_distance(i, j);
        }
    }
    
    // Map buffers to GPU persistently - individual pragmas like RANSSolver
    if (!k_flat_.empty() && !work_flat_.empty()) {
        double* k_ptr = k_flat_.data();
        double* omega_ptr = omega_flat_.data();
        double* u_ptr = u_flat_.data();
        double* v_ptr = v_flat_.data();
        double* wall_ptr = wall_dist_flat_.data();
        double* nu_t_ptr = nu_t_flat_.data();
        double* tau_ptr = tau_flat_.data();
        double* work_ptr = work_flat_.data();
        
        size_t k_size = k_flat_.size();
        size_t omega_size = omega_flat_.size();
        size_t u_size = u_flat_.size();
        size_t v_size = v_flat_.size();
        size_t wall_size = wall_dist_flat_.size();
        size_t nu_t_size = nu_t_flat_.size();
        size_t tau_size = tau_flat_.size();
        size_t work_size = work_flat_.size();
        
        // Map buffers to GPU - use single pragma with multiple arrays (like NN models)
        #pragma omp target enter data \
            map(alloc: k_ptr[0:k_size]) \
            map(alloc: omega_ptr[0:omega_size]) \
            map(alloc: u_ptr[0:u_size]) \
            map(alloc: v_ptr[0:v_size]) \
            map(alloc: wall_ptr[0:wall_size]) \
            map(alloc: nu_t_ptr[0:nu_t_size]) \
            map(alloc: tau_ptr[0:tau_size]) \
            map(alloc: work_ptr[0:work_size])
        
        buffers_on_gpu_ = true;  // Mark as mapped
    }
}

void EARSMClosure::free_gpu_buffers() {
    // Unmap GPU buffers if they were mapped
    if (buffers_on_gpu_) {
        // Check vectors are non-empty before unmapping
        if (!k_flat_.empty() && !work_flat_.empty()) {
            buffers_on_gpu_ = false;  // Set flag FIRST to prevent re-entry
            
            double* k_ptr = k_flat_.data();
            double* omega_ptr = omega_flat_.data();
            double* u_ptr = u_flat_.data();
            double* v_ptr = v_flat_.data();
            double* wall_ptr = wall_dist_flat_.data();
            double* nu_t_ptr = nu_t_flat_.data();
            double* tau_ptr = tau_flat_.data();
            double* work_ptr = work_flat_.data();
            
            size_t k_size = k_flat_.size();
            size_t omega_size = omega_flat_.size();
            size_t u_size = u_flat_.size();
            size_t v_size = v_flat_.size();
            size_t wall_size = wall_dist_flat_.size();
            size_t nu_t_size = nu_t_flat_.size();
            size_t tau_size = tau_flat_.size();
            size_t work_size = work_flat_.size();
            
            // Unmap buffers from GPU - use single pragma with multiple arrays (like NN models)
            #pragma omp target exit data \
                map(delete: k_ptr[0:k_size]) \
                map(delete: omega_ptr[0:omega_size]) \
                map(delete: u_ptr[0:u_size]) \
                map(delete: v_ptr[0:v_size]) \
                map(delete: wall_ptr[0:wall_size]) \
                map(delete: nu_t_ptr[0:nu_t_size]) \
                map(delete: tau_ptr[0:tau_size]) \
                map(delete: work_ptr[0:work_size])
        } else {
            buffers_on_gpu_ = false;
        }
    }
    
    // Clear CPU buffers (always, regardless of GPU offloading)
    k_flat_.clear();
    omega_flat_.clear();
    u_flat_.clear();
    v_flat_.clear();
    wall_dist_flat_.clear();
    nu_t_flat_.clear();
    tau_flat_.clear();
    work_flat_.clear();
}
#else
// No-op implementations when GPU offloading is disabled
void EARSMClosure::allocate_gpu_buffers(const Mesh& mesh) {
    (void)mesh;
    buffers_on_gpu_ = false;
}
#endif

void EARSMClosure::compute_nu_t(
    const Mesh& mesh,
    const VectorField& velocity,
    const ScalarField& k,
    const ScalarField& omega,
    ScalarField& nu_t,
    TensorField* tau_ij)
{
    TIMED_SCOPE("earsm_closure");
    
    ensure_initialized(mesh);
    
#ifdef USE_GPU_OFFLOAD_DISABLED_FOR_EARSM
    // TEMPORARILY DISABLED: GPU path for EARSM has "partially present" memory issues
    // Pointer aliasing causes conflicts with already-mapped GPU buffers
    // TODO: Debug and re-enable GPU path for EARSM models
    if (gpu_ready_ && omp_get_num_devices() > 0) {
        compute_nu_t_gpu(mesh, velocity, k, omega, nu_t, tau_ij);
        return;
    }
#endif
    
    // CPU implementation
    feature_computer_->set_reference(nu_, delta_, 1.0);
    feature_computer_->compute_tbnn_features(velocity, k, omega, features_, basis_);
    
    const double C_mu = 0.09;
    
    int idx = 0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i, ++idx) {
            double k_loc = std::max(1e-10, k(i, j));
            double omega_loc = std::max(1e-10, omega(i, j));
            double eps = C_mu * k_loc * omega_loc;
            double tau = k_loc / std::max(eps, 1e-20);
            
            // Get velocity gradient for strain/rotation magnitudes
            VelocityGradient grad = compute_velocity_gradient(mesh, velocity, i, j);
            double S_mag = grad.S_mag();
            double Omega_mag = grad.Omega_mag();
            
            // Normalized invariants
            double eta = tau * S_mag;       // η = (k/ε)|S|
            double zeta = tau * Omega_mag;  // ζ = (k/ε)|Ω|
            
            // Compute G coefficients from derived class
            std::array<double, TensorBasis::NUM_BASIS> G;
            compute_G(eta, zeta, G);
            
            // Construct anisotropy tensor
            double b_xx, b_xy, b_yy;
            TensorBasis::construct_anisotropy(G, basis_[idx], b_xx, b_xy, b_yy);
            
            // Compute Reynolds stresses if requested
            if (tau_ij) {
                double tau_xx, tau_xy, tau_yy;
                TensorBasis::anisotropy_to_reynolds_stress(
                    b_xx, b_xy, b_yy, k_loc, tau_xx, tau_xy, tau_yy);
                tau_ij->xx(i, j) = tau_xx;
                tau_ij->xy(i, j) = tau_xy;
                tau_ij->yy(i, j) = tau_yy;
            }
            
            // Compute equivalent eddy viscosity
            double Sxy = grad.Sxy();
            double nu_t_loc = 0.0;
            
            if (std::abs(Sxy) > 1e-10) {
                // Match τ_xy = -2k b_xy with Boussinesq: τ_xy = -2ν_t S_xy
                nu_t_loc = std::abs(-b_xy * k_loc / Sxy);
            } else if (S_mag > 1e-10) {
                // Fallback: use magnitude
                double b_mag = std::sqrt(b_xx*b_xx + 2.0*b_xy*b_xy + b_yy*b_yy);
                nu_t_loc = k_loc * b_mag / S_mag;
            }
            
            // Clipping
            nu_t_loc = std::max(0.0, nu_t_loc);
            nu_t_loc = std::min(nu_t_loc, 100.0 * nu_);
            
            if (!std::isfinite(nu_t_loc)) {
                nu_t_loc = 0.0;
            }
            
            nu_t(i, j) = nu_t_loc;
        }
    }
}

#ifdef USE_GPU_OFFLOAD
void EARSMClosure::compute_nu_t_gpu(
    const Mesh& mesh,
    const VectorField& velocity,
    const ScalarField& k,
    const ScalarField& omega,
    ScalarField& nu_t,
    TensorField* tau_ij)
{
    // For now, fall back to CPU for EARSM
    // GPU implementation would need to encode compute_G as a GPU kernel
    // This is model-specific, so we'd need virtual dispatch on GPU (complex)
    
    // Instead, compute on CPU
    feature_computer_->set_reference(nu_, delta_, 1.0);
    feature_computer_->compute_tbnn_features(velocity, k, omega, features_, basis_);
    
    const double C_mu = 0.09;
    
    int idx = 0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i, ++idx) {
            double k_loc = std::max(1e-10, k(i, j));
            double omega_loc = std::max(1e-10, omega(i, j));
            double eps = C_mu * k_loc * omega_loc;
            double tau = k_loc / std::max(eps, 1e-20);
            
            VelocityGradient grad = compute_velocity_gradient(mesh, velocity, i, j);
            double S_mag = grad.S_mag();
            double Omega_mag = grad.Omega_mag();
            
            double eta = tau * S_mag;
            double zeta = tau * Omega_mag;
            
            std::array<double, TensorBasis::NUM_BASIS> G;
            compute_G(eta, zeta, G);
            
            double b_xx, b_xy, b_yy;
            TensorBasis::construct_anisotropy(G, basis_[idx], b_xx, b_xy, b_yy);
            
            if (tau_ij) {
                double tau_xx, tau_xy, tau_yy;
                TensorBasis::anisotropy_to_reynolds_stress(
                    b_xx, b_xy, b_yy, k_loc, tau_xx, tau_xy, tau_yy);
                tau_ij->xx(i, j) = tau_xx;
                tau_ij->xy(i, j) = tau_xy;
                tau_ij->yy(i, j) = tau_yy;
            }
            
            double Sxy = grad.Sxy();
            double nu_t_loc = 0.0;
            
            if (std::abs(Sxy) > 1e-10) {
                nu_t_loc = std::abs(-b_xy * k_loc / Sxy);
            } else if (S_mag > 1e-10) {
                double b_mag = std::sqrt(b_xx*b_xx + 2.0*b_xy*b_xy + b_yy*b_yy);
                nu_t_loc = k_loc * b_mag / S_mag;
            }
            
            nu_t_loc = std::max(0.0, nu_t_loc);
            nu_t_loc = std::min(nu_t_loc, 100.0 * nu_);
            
            if (!std::isfinite(nu_t_loc)) {
                nu_t_loc = 0.0;
            }
            
            nu_t(i, j) = nu_t_loc;
        }
    }
}
#else
// No-op implementation when GPU offloading is disabled
void EARSMClosure::compute_nu_t_gpu(
    const Mesh& mesh,
    const VectorField& velocity,
    const ScalarField& k,
    const ScalarField& omega,
    ScalarField& nu_t,
    TensorField* tau_ij)
{
    (void)mesh; (void)velocity; (void)k; (void)omega; (void)nu_t; (void)tau_ij;
    // CPU path used instead
}
#endif

// ============================================================================
// Wallin-Johansson EARSM Implementation
// ============================================================================

double WallinJohanssonEARSM::solve_for_N(double II_S, double II_Omega) const {
    // Solve the cubic equation for N (effective viscosity parameter)
    // From Wallin & Johansson (2000), the cubic is:
    // N³ + p N + q = 0
    // where p and q depend on II_S, II_Omega and model constants
    
    const double A1 = constants_.A1();
    // A2 used in more complex 3D formulation
    const double A3 = constants_.A3();
    const double A4 = constants_.A4();
    
    // For 2D, the cubic simplifies significantly
    // In the limit of 2D incompressible flow:
    // N = -A1 / (1 + A3 II_S + A4 II_Omega)
    
    // More general form from the paper:
    // p = (2/27)(A1² - 9A3 A4 II_Omega) + (9/20)A2² II_S
    // q = (1/27)(-A1³ + 9A1 A3 A4 II_Omega) - (9/20)A1 A2² II_S
    
    double denom = 1.0 + A3 * II_S + A4 * II_Omega;
    
    // Regularization for stability
    denom = std::max(std::abs(denom), 0.1) * (denom >= 0 ? 1.0 : -1.0);
    
    double N = -A1 / denom;
    
    // Limit N to prevent extreme values
    N = std::max(-10.0, std::min(N, 10.0));
    
    return N;
}

void WallinJohanssonEARSM::compute_G(
    double eta, double zeta,
    std::array<double, TensorBasis::NUM_BASIS>& G) const
{
    // Wallin-Johansson EARSM coefficients
    // 
    // The anisotropy tensor is:
    // b_ij = β₁ S*_ij + β₂ (S*_ik Ω*_kj - Ω*_ik S*_kj) + β₃ (S*_ik S*_kj - ⅓ II_S δ_ij)
    //      + β₄ (Ω*_ik Ω*_kj - ⅓ II_Ω δ_ij) + ...
    //
    // where S* = (k/ε) S, Ω* = (k/ε) Ω
    // 
    // The β coefficients are functions of the invariants II_S = tr(S*²), II_Ω = tr(Ω*²)
    
    // Invariants (η² ≈ II_S, ζ² ≈ II_Ω for 2D)
    double II_S = eta * eta;
    double II_Omega = -zeta * zeta;  // Note: tr(Ω²) is negative
    
    // Solve for N
    double N = solve_for_N(II_S, std::abs(II_Omega));
    
    const double A1 = constants_.A1();
    const double A2 = constants_.A2();
    const double A3 = constants_.A3();
    // A4 used in 3D formulation for Ω² term
    
    // Compute β coefficients
    // From Wallin & Johansson (2000):
    // β₁ = -N / (A1 + N)
    // β₂ = ...
    
    double denom = A1 + N;
    if (std::abs(denom) < 0.01) {
        denom = 0.01 * (denom >= 0 ? 1.0 : -1.0);
    }
    
    double beta1 = -N / denom;
    
    // For the commutator term [S, Ω]
    double beta2 = 0.0;
    if (std::abs(II_Omega) > 1e-10) {
        beta2 = A2 * N * N / (denom * denom);
    }
    
    // For the S² term
    double beta3 = 0.0;
    if (II_S > 1e-10) {
        beta3 = A3 * N / denom;
    }
    
    // Map to tensor basis coefficients
    // T^(1) = S*        → G[0] = β₁
    // T^(2) = [S*, Ω*]  → G[1] = β₂
    // T^(3) = S*² - ⅓tr(S*²)I → G[2] = β₃
    // T^(4) = 0 in 2D   → G[3] = 0
    
    G[0] = beta1;
    G[1] = beta2;
    G[2] = beta3;
    G[3] = 0.0;
    
    // Limit coefficients for stability
    for (int n = 0; n < TensorBasis::NUM_BASIS; ++n) {
        G[n] = std::max(-10.0, std::min(G[n], 10.0));
    }
}

// ============================================================================
// Gatski-Speziale EARSM Implementation
// ============================================================================

void GatskiSpezialeEARSM::compute_G(
    double eta, double zeta,
    std::array<double, TensorBasis::NUM_BASIS>& G) const
{
    // Gatski-Speziale regularized EARSM
    // 
    // Uses a regularized form to avoid singularities in the cubic solution.
    // The effective C_μ is:
    // C_μ = C_μ0 / (1 + η²/η_max²)
    //
    // And the anisotropy is approximated as:
    // b_ij ≈ -C_μ S*_ij + C₁ [S*, Ω*]_ij + C₂ (S*² - ⅓ II_S I)_ij
    
    const double C_mu0 = constants_.C_mu;
    const double C1 = constants_.C1;
    const double C2 = constants_.C2;
    const double eta_max = constants_.eta_max;
    
    // Regularization factor
    double reg = 1.0 + (eta * eta) / (eta_max * eta_max);
    double C_mu_eff = C_mu0 / reg;
    
    // Rotation correction
    double rot_factor = 1.0;
    if (eta > 1e-10) {
        double ratio = zeta / eta;
        rot_factor = 1.0 / (1.0 + 0.1 * ratio * ratio);
    }
    
    // Coefficients
    G[0] = -C_mu_eff * rot_factor;  // Linear term (Boussinesq-like)
    G[1] = C1 * C_mu_eff * C_mu_eff;  // Rotation-strain interaction
    G[2] = C2 * C_mu_eff;              // Quadratic strain term
    G[3] = 0.0;                         // Zero in 2D
    
    // Stability limits
    for (int n = 0; n < TensorBasis::NUM_BASIS; ++n) {
        G[n] = std::max(-5.0, std::min(G[n], 5.0));
    }
}

// ============================================================================
// Pope Quadratic Model Implementation
// ============================================================================

void PopeQuadraticEARSM::compute_G(
    double eta, double zeta,
    std::array<double, TensorBasis::NUM_BASIS>& G) const
{
    // Pope's quadratic nonlinear eddy viscosity model (1975)
    //
    // b_ij = -C_μ S*_ij + C₁ (S*² - ⅓ II_S I)_ij + C₂ [S*, Ω*]_ij
    //
    // This is a simple quadratic extension that captures some
    // anisotropy effects while remaining computationally cheap.
    
    const double C_mu = 0.09;
    
    // Regularization for high strain rates
    double reg = 1.0 + 0.01 * eta * eta;
    double C_mu_eff = C_mu / reg;
    
    G[0] = -C_mu_eff;     // Linear Boussinesq term
    G[1] = C2_ * eta;     // Rotation-strain interaction (scaled by η)
    G[2] = C1_ * eta;     // Quadratic strain term (scaled by η)
    G[3] = 0.0;           // Zero in 2D
    
    (void)zeta;  // Not used in simple Pope model
}

// ============================================================================
// SST + EARSM Combined Model
// ============================================================================

SSTWithEARSM::SSTWithEARSM(EARSMType earsm_type, const SSTConstants& sst_constants)
    : transport_(sst_constants)
{
    closure_ = create_earsm_closure(earsm_type);
    
    // Set the EARSM closure on the transport model
    // Note: We keep a separate copy for the update() call
}

void SSTWithEARSM::initialize(const Mesh& mesh, const VectorField& velocity) {
    transport_.set_nu(nu_);
    transport_.set_delta(delta_);
    transport_.initialize(mesh, velocity);
    
    if (closure_) {
        closure_->set_nu(nu_);
        closure_->set_delta(delta_);
    }
}

void SSTWithEARSM::advance_turbulence(
    const Mesh& mesh,
    const VectorField& velocity,
    double dt,
    ScalarField& k,
    ScalarField& omega,
    const ScalarField& nu_t_prev)
{
    // Use SST transport for k, ω evolution
    transport_.set_nu(nu_);
    transport_.advance_turbulence(mesh, velocity, dt, k, omega, nu_t_prev);
}

void SSTWithEARSM::update(
    const Mesh& mesh,
    const VectorField& velocity,
    const ScalarField& k,
    const ScalarField& omega,
    ScalarField& nu_t,
    TensorField* tau_ij,
    const TurbulenceDeviceView* device_view)
{
    (void)device_view;  // Not yet implemented for EARSM
    // Use EARSM closure for ν_t computation
    if (closure_) {
        closure_->set_nu(nu_);
        closure_->set_delta(delta_);
        closure_->compute_nu_t(mesh, velocity, k, omega, nu_t, tau_ij);
    } else {
        // Fallback to SST closure
        transport_.update(mesh, velocity, k, omega, nu_t, tau_ij);
    }
}

std::string SSTWithEARSM::name() const {
    std::string base = "SST+";
    if (closure_) {
        return base + closure_->name();
    }
    return base + "EARSM";
}

// ============================================================================
// Factory Functions
// ============================================================================

std::unique_ptr<EARSMClosure> create_earsm_closure(EARSMType type) {
    switch (type) {
        case EARSMType::WallinJohansson2000:
            return std::make_unique<WallinJohanssonEARSM>();
        case EARSMType::GatskiSpeziale1993:
            return std::make_unique<GatskiSpezialeEARSM>();
        case EARSMType::Pope1975:
            return std::make_unique<PopeQuadraticEARSM>();
        default:
            return std::make_unique<WallinJohanssonEARSM>();
    }
}

std::unique_ptr<EARSMClosure> create_earsm_closure(const std::string& name) {
    if (name == "WJ" || name == "WallinJohansson" || name == "wj" || name == "WJ-EARSM") {
        return std::make_unique<WallinJohanssonEARSM>();
    } else if (name == "GS" || name == "GatskiSpeziale" || name == "gs" || name == "GS-EARSM") {
        return std::make_unique<GatskiSpezialeEARSM>();
    } else if (name == "Pope" || name == "pope" || name == "quadratic" || name == "Pope-Quadratic") {
        return std::make_unique<PopeQuadraticEARSM>();
    }
    
    // Default to Wallin-Johansson
    return std::make_unique<WallinJohanssonEARSM>();
}

// ============================================================================
// GPU Kernels for EARSM (Specialized Versions)
// ============================================================================

namespace earsm_kernels {

void compute_wj_coefficients_gpu(
    const double* dudx, const double* dudy,
    const double* dvdx, const double* dvdy,
    const double* k, const double* omega,
    double* G,
    int n_cells,
    const WJConstants& constants)
{
#ifdef USE_GPU_OFFLOAD
    const double A1 = constants.A1();
    const double A2 = constants.A2();
    const double A3 = constants.A3();
    const double A4 = constants.A4();
    const double C_mu = 0.09;
    
    // k and omega are already on GPU from solver; gradients and G are local
    #pragma omp target teams distribute parallel for \
        map(to: dudx[0:n_cells], dudy[0:n_cells], \
                dvdx[0:n_cells], dvdy[0:n_cells]) \
        map(present: k[0:n_cells], omega[0:n_cells]) \
        map(from: G[0:n_cells*4])
    for (int idx = 0; idx < n_cells; ++idx) {
        // Strain and rotation
        double Sxx = dudx[idx];
        double Syy = dvdy[idx];
        double Sxy = 0.5 * (dudy[idx] + dvdx[idx]);
        double Oxy = 0.5 * (dudy[idx] - dvdx[idx]);
        
        double S_mag = sqrt(2.0 * (Sxx*Sxx + Syy*Syy + 2.0*Sxy*Sxy));
        double Omega_mag = sqrt(2.0 * Oxy * Oxy);
        
        double k_loc = k[idx];
        double omega_loc = omega[idx];
        k_loc = (k_loc > 1e-10) ? k_loc : 1e-10;
        omega_loc = (omega_loc > 1e-10) ? omega_loc : 1e-10;
        
        double eps = C_mu * k_loc * omega_loc;
        double tau = k_loc / eps;
        
        double eta = tau * S_mag;
        double zeta = tau * Omega_mag;
        
        double II_S = eta * eta;
        double II_Omega = zeta * zeta;
        
        // Solve for N
        double denom = 1.0 + A3 * II_S + A4 * II_Omega;
        denom = (fabs(denom) > 0.1) ? denom : 0.1 * (denom >= 0 ? 1.0 : -1.0);
        double N = -A1 / denom;
        N = (N > -10.0) ? N : -10.0;
        N = (N < 10.0) ? N : 10.0;
        
        // Compute β coefficients
        double denom2 = A1 + N;
        denom2 = (fabs(denom2) > 0.01) ? denom2 : 0.01 * (denom2 >= 0 ? 1.0 : -1.0);
        
        double beta1 = -N / denom2;
        double beta2 = (II_Omega > 1e-10) ? A2 * N * N / (denom2 * denom2) : 0.0;
        double beta3 = (II_S > 1e-10) ? A3 * N / denom2 : 0.0;
        
        // Store G coefficients
        int base = idx * 4;
        G[base + 0] = beta1;
        G[base + 1] = beta2;
        G[base + 2] = beta3;
        G[base + 3] = 0.0;
    }
#else
    (void)dudx; (void)dudy; (void)dvdx; (void)dvdy;
    (void)k; (void)omega; (void)G; (void)n_cells; (void)constants;
#endif
}

void compute_gs_coefficients_gpu(
    const double* dudx, const double* dudy,
    const double* dvdx, const double* dvdy,
    const double* k, const double* omega,
    double* G,
    int n_cells,
    const GSConstants& constants)
{
#ifdef USE_GPU_OFFLOAD
    const double C_mu0 = constants.C_mu;
    const double C1 = constants.C1;
    const double C2 = constants.C2;
    const double eta_max = constants.eta_max;
    const double C_mu_eps = 0.09;
    
    // k and omega are already on GPU from solver; gradients and G are local
    #pragma omp target teams distribute parallel for \
        map(to: dudx[0:n_cells], dudy[0:n_cells], \
                dvdx[0:n_cells], dvdy[0:n_cells]) \
        map(present: k[0:n_cells], omega[0:n_cells]) \
        map(from: G[0:n_cells*4])
    for (int idx = 0; idx < n_cells; ++idx) {
        double Sxx = dudx[idx];
        double Syy = dvdy[idx];
        double Sxy = 0.5 * (dudy[idx] + dvdx[idx]);
        double Oxy = 0.5 * (dudy[idx] - dvdx[idx]);
        
        double S_mag = sqrt(2.0 * (Sxx*Sxx + Syy*Syy + 2.0*Sxy*Sxy));
        double Omega_mag = sqrt(2.0 * Oxy * Oxy);
        
        double k_loc = k[idx];
        double omega_loc = omega[idx];
        k_loc = (k_loc > 1e-10) ? k_loc : 1e-10;
        omega_loc = (omega_loc > 1e-10) ? omega_loc : 1e-10;
        
        double eps = C_mu_eps * k_loc * omega_loc;
        double tau = k_loc / eps;
        
        double eta = tau * S_mag;
        double zeta = tau * Omega_mag;
        
        // Regularization
        double reg = 1.0 + (eta * eta) / (eta_max * eta_max);
        double C_mu_eff = C_mu0 / reg;
        
        double rot_factor = 1.0;
        if (eta > 1e-10) {
            double ratio = zeta / eta;
            rot_factor = 1.0 / (1.0 + 0.1 * ratio * ratio);
        }
        
        int base = idx * 4;
        G[base + 0] = -C_mu_eff * rot_factor;
        G[base + 1] = C1 * C_mu_eff * C_mu_eff;
        G[base + 2] = C2 * C_mu_eff;
        G[base + 3] = 0.0;
    }
#else
    (void)dudx; (void)dudy; (void)dvdx; (void)dvdy;
    (void)k; (void)omega; (void)G; (void)n_cells; (void)constants;
#endif
}

} // namespace earsm_kernels

} // namespace nncfd

