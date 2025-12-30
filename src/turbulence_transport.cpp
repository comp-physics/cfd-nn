/// @file turbulence_transport.cpp
/// @brief Implementation of transport equation turbulence models (SST k-ω, k-ω)
///
/// This file implements two-equation RANS models that solve transport PDEs:
/// - SST k-ω (Menter 1994): Blended k-ω/k-ε with strain-rate limiter
/// - Standard k-ω (Wilcox 1988): Original k-ω formulation
///
/// These models provide good accuracy for complex flows including separation,
/// adverse pressure gradients, and streamline curvature. Both include:
/// - Production, dissipation, and diffusion terms
/// - Wall boundary conditions (low-Re formulation)
/// - GPU-accelerated transport step and closure
/// - Positivity enforcement and realizability constraints

#include "turbulence_transport.hpp"
#include "gpu_kernels.hpp"
#include "timing.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

// ============================================================================
// Device-callable helper functions (work on both CPU and GPU)
// ============================================================================

#ifdef USE_GPU_OFFLOAD
#pragma omp declare target
#endif

/// Return the smaller of two values
inline double dmin(double a, double b) { return a < b ? a : b; }

/// Return the larger of two values
inline double dmax(double a, double b) { return a > b ? a : b; }

/// Clamp a value to [lo, hi] range
inline double dclamp(double val, double lo, double hi) {
    return val < lo ? lo : (val > hi ? hi : val);
}

/// Ensure value is at least min_val
inline double dmax0(double val, double min_val) { return val > min_val ? val : min_val; }

#ifdef USE_GPU_OFFLOAD
#pragma omp end declare target
#endif

namespace nncfd {

// ============================================================================
// Boussinesq Closure Implementation
// ============================================================================

void BoussinesqClosure::compute_nu_t(
    const Mesh& mesh,
    const VectorField& velocity,
    const ScalarField& k,
    const ScalarField& omega,
    ScalarField& nu_t,
    TensorField* tau_ij)
{
    (void)velocity;
    (void)tau_ij;  // Boussinesq doesn't compute explicit stresses
    
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double k_loc = std::max(1e-10, k(i, j));
            double omega_loc = std::max(1e-10, omega(i, j));
            
            // ν_t = k / ω (for k-ω models)
            // Equivalent to ν_t = C_μ k² / ε where ε = C_μ k ω
            double nu_t_loc = k_loc / omega_loc;
            
            // Clipping
            nu_t_loc = std::max(0.0, nu_t_loc);
            nu_t_loc = std::min(nu_t_loc, 1000.0 * nu_);
            
            nu_t(i, j) = nu_t_loc;
        }
    }
}

// ============================================================================
// SST Closure Implementation
// ============================================================================

void SSTClosure::ensure_initialized(const Mesh& mesh) {
    if (!initialized_) {
        dudx_ = ScalarField(mesh);
        dudy_ = ScalarField(mesh);
        dvdx_ = ScalarField(mesh);
        dvdy_ = ScalarField(mesh);
        initialized_ = true;
    }
}

double SSTClosure::compute_F2(double k, double omega, double y_wall) const {
    double k_safe = std::max(constants_.k_min, k);
    double omega_safe = std::max(constants_.omega_min, omega);
    double y_safe = std::max(1e-10, y_wall);
    
    // arg2 = max(2√k / (β*ωy), 500ν / (y²ω))
    double sqrt_k = std::sqrt(k_safe);
    double term1 = 2.0 * sqrt_k / (constants_.beta_star * omega_safe * y_safe);
    double term2 = 500.0 * nu_ / (y_safe * y_safe * omega_safe);
    double arg2 = std::max(term1, term2);
    
    // F2 = tanh(arg2²)
    return std::tanh(arg2 * arg2);
}

void SSTClosure::compute_nu_t(
    const Mesh& mesh,
    const VectorField& velocity,
    const ScalarField& k,
    const ScalarField& omega,
    ScalarField& nu_t,
    TensorField* tau_ij)
{
    (void)tau_ij;  // SST closure is still LEVM
    
    ensure_initialized(mesh);
    
    // Compute velocity gradients (MAC-aware for CPU/GPU consistency)
    compute_gradients_from_mac(mesh, velocity, dudx_, dudy_, dvdx_, dvdy_);
    
    const double a1 = constants_.a1;
    
    // NOTE: SSTClosure GPU path is intentionally disabled to avoid pointer aliasing
    // issues with RANSSolver's GPU buffers. The caller (RANSSolver or SSTKOmegaTransport)
    // handles GPU synchronization when needed. This host path is acceptable because
    // the SST closure computation is relatively cheap compared to transport equation solves.
    
    // CPU path
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double k_loc = std::max(constants_.k_min, k(i, j));
            double omega_loc = std::max(constants_.omega_min, omega(i, j));
            double y_wall = mesh.wall_distance(i, j);
            
            // Strain rate magnitude
            double Sxx = dudx_(i, j);
            double Syy = dvdy_(i, j);
            double Sxy = 0.5 * (dudy_(i, j) + dvdx_(i, j));
            double S_mag = std::sqrt(2.0 * (Sxx*Sxx + Syy*Syy + 2.0*Sxy*Sxy));
            
            // F2 blending function
            double F2 = compute_F2(k_loc, omega_loc, y_wall);
            
            // SST eddy viscosity: ν_t = a₁k / max(a₁ω, SF₂)
            double denom = std::max(a1 * omega_loc, S_mag * F2);
            double nu_t_loc = a1 * k_loc / denom;
            
            // Clipping
            nu_t_loc = std::max(0.0, nu_t_loc);
            nu_t_loc = std::min(nu_t_loc, 1000.0 * nu_);
            
            nu_t(i, j) = nu_t_loc;
        }
    }
}

// ============================================================================
// SST k-ω Transport Implementation
// ============================================================================

SSTKOmegaTransport::SSTKOmegaTransport(const SSTConstants& constants)
    : constants_(constants)
{
    // Default to SST closure
    closure_ = std::make_unique<SSTClosure>(constants_);
}

SSTKOmegaTransport::~SSTKOmegaTransport() {
    cleanup_gpu_buffers();
}

void SSTKOmegaTransport::initialize_gpu_buffers(const Mesh& mesh) {
#ifdef USE_GPU_OFFLOAD
    // Fail fast if no GPU device available (GPU build requires GPU)
    gpu::verify_device_available();
    
    int n_interior = mesh.Nx * mesh.Ny;
    
    // Check if already allocated for this mesh size
    if (buffers_on_gpu_ && cached_n_cells_ == n_interior) {
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

void SSTKOmegaTransport::cleanup_gpu_buffers() {
#ifdef USE_GPU_OFFLOAD
    free_gpu_buffers();
#endif
    buffers_on_gpu_ = false;
}

void SSTKOmegaTransport::set_closure(std::unique_ptr<TurbulenceClosure> closure) {
    closure_ = std::move(closure);
    if (closure_) {
        closure_->set_nu(nu_);
        closure_->set_delta(delta_);
    }
}

void SSTKOmegaTransport::ensure_initialized(const Mesh& mesh) {
    if (!initialized_) {
        dudx_ = ScalarField(mesh);
        dudy_ = ScalarField(mesh);
        dvdx_ = ScalarField(mesh);
        dvdy_ = ScalarField(mesh);
        P_k_ = ScalarField(mesh);
        F1_ = ScalarField(mesh);
        F2_ = ScalarField(mesh);
        adv_k_ = ScalarField(mesh);
        adv_omega_ = ScalarField(mesh);
        diff_k_ = ScalarField(mesh);
        diff_omega_ = ScalarField(mesh);
        nu_k_ = ScalarField(mesh);
        nu_omega_ = ScalarField(mesh);
        k_old_ = ScalarField(mesh);
        omega_old_ = ScalarField(mesh);
        wall_dist_ = ScalarField(mesh);

        // Pre-compute wall distance for unified CPU/GPU kernel
        for (int j = 0; j < mesh.total_Ny(); ++j) {
            for (int i = 0; i < mesh.total_Nx(); ++i) {
                wall_dist_(i, j) = mesh.wall_distance(i, j);
            }
        }
        
#ifdef USE_GPU_OFFLOAD
        allocate_gpu_buffers(mesh);
#endif
        
        initialized_ = true;
    }
}

#ifdef USE_GPU_OFFLOAD
void SSTKOmegaTransport::allocate_gpu_buffers(const Mesh& mesh) {
    int n_interior = mesh.Nx * mesh.Ny;
    int n_total = (mesh.Nx + 2) * (mesh.Ny + 2);
    
    // Check if already allocated
    if (n_interior == cached_n_cells_ && buffers_on_gpu_) {
        return;  // Already allocated and mapped
    }
    
    // Free old buffers if they exist
    free_gpu_buffers();
    
    // Allocate CPU buffers
    k_flat_.resize(n_total);
    omega_flat_.resize(n_total);
    nu_t_flat_.resize(n_interior);
    u_flat_.resize(n_total);
    v_flat_.resize(n_total);
    wall_dist_flat_.resize(n_interior);
    
    // Workspace: gradients (4), P_k, F1, F2, adv_k, adv_omega, diff_k, diff_omega, nu_k, nu_omega
    work_flat_.resize(n_interior * 13);
    
    // Precompute wall distances
    int idx = 0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            wall_dist_flat_[idx++] = mesh.wall_distance(i, j);
        }
    }
    
    // Map buffers to GPU persistently - use individual pragmas like RANSSolver
    if (!k_flat_.empty() && !work_flat_.empty()) {
        double* k_ptr = k_flat_.data();
        double* omega_ptr = omega_flat_.data();
        double* nu_t_ptr = nu_t_flat_.data();
        double* u_ptr = u_flat_.data();
        double* v_ptr = v_flat_.data();
        double* wall_ptr = wall_dist_flat_.data();
        double* work_ptr = work_flat_.data();
        
        size_t k_size = k_flat_.size();
        size_t omega_size = omega_flat_.size();
        size_t nu_t_size = nu_t_flat_.size();
        size_t u_size = u_flat_.size();
        size_t v_size = v_flat_.size();
        size_t wall_size = wall_dist_flat_.size();
        size_t work_size = work_flat_.size();
        
        // Map buffers to GPU - use 'to' for k/omega to upload initial values
        // k and omega are initialized by RANSSolver::initialize() before this is called
        // so we MUST upload them. Other arrays are computed on GPU, so use alloc.
        #pragma omp target enter data \
            map(to: k_ptr[0:k_size]) \
            map(to: omega_ptr[0:omega_size]) \
            map(alloc: nu_t_ptr[0:nu_t_size]) \
            map(alloc: u_ptr[0:u_size]) \
            map(alloc: v_ptr[0:v_size]) \
            map(to: wall_ptr[0:wall_size]) \
            map(alloc: work_ptr[0:work_size])
        
        buffers_on_gpu_ = true;  // Mark as mapped (separate from gpu_ready_)
    }
    
    cached_n_cells_ = n_interior;
}

void SSTKOmegaTransport::free_gpu_buffers() {
    // Unmap GPU buffers if they were mapped (check buffers_on_gpu_ flag)
    if (buffers_on_gpu_) {
        // Check vectors are non-empty before unmapping
        if (!k_flat_.empty() && !work_flat_.empty()) {
            buffers_on_gpu_ = false;  // Set flag FIRST to prevent re-entry
            
            double* k_ptr = k_flat_.data();
            double* omega_ptr = omega_flat_.data();
            double* nu_t_ptr = nu_t_flat_.data();
            double* u_ptr = u_flat_.data();
            double* v_ptr = v_flat_.data();
            double* wall_ptr = wall_dist_flat_.data();
            double* work_ptr = work_flat_.data();
            
            size_t k_size = k_flat_.size();
            size_t omega_size = omega_flat_.size();
            size_t nu_t_size = nu_t_flat_.size();
            size_t u_size = u_flat_.size();
            size_t v_size = v_flat_.size();
            size_t wall_size = wall_dist_flat_.size();
            size_t work_size = work_flat_.size();
            
            // Unmap buffers from GPU - use release instead of delete for robustness
            #pragma omp target exit data \
                map(release: k_ptr[0:k_size]) \
                map(release: omega_ptr[0:omega_size]) \
                map(release: nu_t_ptr[0:nu_t_size]) \
                map(release: u_ptr[0:u_size]) \
                map(release: v_ptr[0:v_size]) \
                map(release: wall_ptr[0:wall_size]) \
                map(release: work_ptr[0:work_size])
        } else {
            buffers_on_gpu_ = false;
        }
    }
    
    // Clear CPU buffers (always, regardless of GPU offloading)
    k_flat_.clear();
    omega_flat_.clear();
    nu_t_flat_.clear();
    u_flat_.clear();
    v_flat_.clear();
    wall_dist_flat_.clear();
    work_flat_.clear();
}
#else
// No-op implementations when GPU offloading is disabled
void SSTKOmegaTransport::allocate_gpu_buffers(const Mesh& mesh) {
    (void)mesh;
    buffers_on_gpu_ = false;
}

void SSTKOmegaTransport::free_gpu_buffers() {
    // No-op for host builds
    buffers_on_gpu_ = false;
}
#endif

void SSTKOmegaTransport::initialize(const Mesh& mesh, const VectorField& velocity) {
    ensure_initialized(mesh);
    
    if (closure_) {
        closure_->set_nu(nu_);
        closure_->set_delta(delta_);
    }
    
    // Initialize GPU buffers if available
    initialize_gpu_buffers(mesh);
    
    // Estimate initial friction velocity from velocity gradient at wall
    double u_tau = 0.0;
    {
        int j = mesh.j_begin();
        double dudy_wall = 0.0;
        int count = 0;
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            if (j+1 < mesh.j_end()) {
                double dudy = (velocity.u(i, j+1) - velocity.u(i, j)) / mesh.dy;
                dudy_wall += std::abs(dudy);
                ++count;
            }
        }
        if (count > 0) {
            dudy_wall /= count;
            u_tau = std::sqrt(nu_ * dudy_wall);
        }
    }
    u_tau = std::max(u_tau, 0.01);  // Minimum value
}

void SSTKOmegaTransport::compute_velocity_gradients(
    const Mesh& mesh, const VectorField& velocity)
{
    compute_gradients_from_mac(mesh, velocity, dudx_, dudy_, dvdx_, dvdy_);
}

void SSTKOmegaTransport::compute_blending_functions(
    const Mesh& mesh,
    const ScalarField& k,
    const ScalarField& omega)
{
    const double beta_star = constants_.beta_star;
    const double sigma_omega2 = constants_.sigma_omega2;
    const double CD_min = constants_.CD_omega_min;
    
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double k_loc = std::max(constants_.k_min, k(i, j));
            double omega_loc = std::max(constants_.omega_min, omega(i, j));
            double y_wall = std::max(1e-10, mesh.wall_distance(i, j));
            
            // Cross-diffusion term for F1
            double dkdx = (k(i+1, j) - k(i-1, j)) / (2.0 * mesh.dx);
            double dkdy = (k(i, j+1) - k(i, j-1)) / (2.0 * mesh.dy);
            double domegadx = (omega(i+1, j) - omega(i-1, j)) / (2.0 * mesh.dx);
            double domegady = (omega(i, j+1) - omega(i, j-1)) / (2.0 * mesh.dy);
            
            double CD_omega = std::max(
                2.0 * sigma_omega2 / omega_loc * (dkdx * domegadx + dkdy * domegady),
                CD_min
            );
            
            // F1 blending function
            double sqrt_k = std::sqrt(k_loc);
            double arg1_1 = sqrt_k / (beta_star * omega_loc * y_wall);
            double arg1_2 = 500.0 * nu_ / (y_wall * y_wall * omega_loc);
            double arg1_3 = 4.0 * sigma_omega2 * k_loc / (CD_omega * y_wall * y_wall);
            double arg1 = std::min(std::max(arg1_1, arg1_2), arg1_3);
            F1_(i, j) = std::tanh(arg1 * arg1 * arg1 * arg1);
            
            // F2 blending function
            double arg2_1 = 2.0 * sqrt_k / (beta_star * omega_loc * y_wall);
            double arg2_2 = 500.0 * nu_ / (y_wall * y_wall * omega_loc);
            double arg2 = std::max(arg2_1, arg2_2);
            F2_(i, j) = std::tanh(arg2 * arg2);
        }
    }
}

void SSTKOmegaTransport::compute_production(
    const Mesh& mesh, const ScalarField& nu_t)
{
    // Note: beta_star used for production limiting in advance_turbulence
    
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double Sxx = dudx_(i, j);
            double Syy = dvdy_(i, j);
            double Sxy = 0.5 * (dudy_(i, j) + dvdx_(i, j));
            double S2 = 2.0 * (Sxx*Sxx + Syy*Syy + 2.0*Sxy*Sxy);
            
            // Production: P_k = 2 ν_t |S|²
            // Limited to prevent excessive production: P_k ≤ 10 β* k ω
            double P_k = 2.0 * nu_t(i, j) * S2;
            
            P_k_(i, j) = P_k;
        }
    }
}

void SSTKOmegaTransport::apply_wall_bc_k(const Mesh& mesh, ScalarField& k) {
    int Ng = mesh.Nghost;
    
    // Bottom wall: k = 0
    for (int i = 0; i < mesh.total_Nx(); ++i) {
        for (int g = 0; g < Ng; ++g) {
            int j_ghost = g;
            int j_interior = Ng;
            k(i, j_ghost) = -k(i, j_interior);  // Extrapolate to give k=0 at wall
        }
    }
    
    // Top wall: k = 0
    for (int i = 0; i < mesh.total_Nx(); ++i) {
        for (int g = 0; g < Ng; ++g) {
            int j_ghost = mesh.Ny + Ng + g;
            int j_interior = mesh.Ny + Ng - 1;
            k(i, j_ghost) = -k(i, j_interior);
        }
    }
}

void SSTKOmegaTransport::apply_wall_bc_omega(
    const Mesh& mesh, ScalarField& omega, const ScalarField& k)
{
    (void)k;
    int Ng = mesh.Nghost;
    const double beta1 = constants_.beta1;
    
    // Bottom wall: ω_wall = 10 × 6ν / (β₁ y₁²)
    for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
        int j_first = mesh.j_begin();
        double y1 = mesh.wall_distance(i, j_first);
        double omega_wall = 10.0 * 6.0 * nu_ / (beta1 * y1 * y1);
        
        for (int g = 0; g < Ng; ++g) {
            int j_ghost = g;
            omega(i, j_ghost) = 2.0 * omega_wall - omega(i, j_first);
        }
    }
    
    // Top wall
    for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
        int j_last = mesh.j_end() - 1;
        double y1 = mesh.wall_distance(i, j_last);
        double omega_wall = 10.0 * 6.0 * nu_ / (beta1 * y1 * y1);
        
        for (int g = 0; g < Ng; ++g) {
            int j_ghost = mesh.Ny + Ng + g;
            omega(i, j_ghost) = 2.0 * omega_wall - omega(i, j_last);
        }
    }
}

// ============================================================================
// Device-callable helper functions for unified CPU/GPU kernel
// ============================================================================

#ifdef USE_GPU_OFFLOAD
#pragma omp declare target
#endif

/// Compute F1 blending function at a single cell
/// This is a pure function that depends only on local cell data
/// Used by both CPU and GPU paths to ensure identical arithmetic
static inline double compute_F1_local(
    double k_val, double omega_val, double y_wall,
    double dkdx, double dkdy, double domegadx, double domegady,
    double nu, double beta_star, double sigma_omega2, double CD_min,
    double k_min, double omega_min)
{
    // Clamp inputs
    double k_safe = dmax(k_val, k_min);
    double omega_safe = dmax(omega_val, omega_min);
    double y_safe = dmax(y_wall, 1e-10);

    // Cross-diffusion term CD_omega
    double CD_omega = dmax(2.0 * sigma_omega2 / omega_safe * (dkdx * domegadx + dkdy * domegady), CD_min);

    // F1 = tanh(arg1^4) where arg1 = min(max(arg1_1, arg1_2), arg1_3)
    double sqrt_k = sqrt(k_safe);
    double arg1_1 = sqrt_k / (beta_star * omega_safe * y_safe);
    double arg1_2 = 500.0 * nu / (y_safe * y_safe * omega_safe);
    double arg1_3 = 4.0 * sigma_omega2 * k_safe / (CD_omega * y_safe * y_safe);

    double arg1 = dmin(dmax(arg1_1, arg1_2), arg1_3);

    return tanh(arg1 * arg1 * arg1 * arg1);
}

#ifdef USE_GPU_OFFLOAD
#pragma omp end declare target
#endif

void SSTKOmegaTransport::advance_turbulence(
    const Mesh& mesh,
    const VectorField& velocity,
    double dt,
    ScalarField& k,
    ScalarField& omega,
    const ScalarField& nu_t_prev,
    const TurbulenceDeviceView* device_view)
{
    TIMED_SCOPE("sst_transport");

    ensure_initialized(mesh);

    // ========================================================================
    // UNIFIED SST k-ω TRANSPORT IMPLEMENTATION
    // ========================================================================
    // This implementation uses IDENTICAL arithmetic for CPU and GPU.
    // The only difference is the OpenMP target pragma for GPU offloading.
    //
    // SST k-ω equations (Menter 1994):
    //   ∂k/∂t + u·∇k = P_k - β*kω + ∇·[(ν + σ_k ν_t)∇k]
    //   ∂ω/∂t + u·∇ω = α(ω/k)P_k - βω² + ∇·[(ν + σ_ω ν_t)∇ω] + CD
    //
    // Key points for correctness:
    // 1. Face velocities used for advection (MAC grid)
    // 2. F1 computed at each cell using that cell's local data
    // 3. Diffusion uses face-averaged coefficients (conservative form)
    // 4. Each face coefficient uses the F1/nu_t of its respective cell
    // ========================================================================

    const int Nx = mesh.Nx;
    const int Ny = mesh.Ny;
    const int Ng = mesh.Nghost;
    const int n_cells = Nx * Ny;
    const int cell_stride = mesh.total_Nx();

    const double dx = mesh.dx;
    const double dy = mesh.dy;
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double inv_2dx = 1.0 / (2.0 * dx);
    const double inv_2dy = 1.0 / (2.0 * dy);

    // Copy model constants to local variables (required for GPU capture)
    const double nu = nu_;
    const double beta_star = constants_.beta_star;
    const double beta1 = constants_.beta1;
    const double beta2 = constants_.beta2;
    const double alpha1 = constants_.alpha1;
    const double alpha2 = constants_.alpha2;
    const double sigma_k1 = constants_.sigma_k1;
    const double sigma_k2 = constants_.sigma_k2;
    const double sigma_omega1 = constants_.sigma_omega1;
    const double sigma_omega2 = constants_.sigma_omega2;
    const double k_min = constants_.k_min;
    const double k_max = constants_.k_max;
    const double omega_min = constants_.omega_min;
    const double omega_max = constants_.omega_max;
    const double CD_min = constants_.CD_omega_min;

    // ========================================================================
    // UNIFIED CPU/GPU IMPLEMENTATION
    // ========================================================================
    // Both CPU and GPU use the same loop body with raw pointer arithmetic.
    // - GPU: uses device_view pointers (already on GPU), parallel loop = Jacobi
    // - CPU: uses host pointers + snapshot arrays for Jacobi iteration
    // ========================================================================

    // Set up pointers based on whether we're using GPU or CPU
    const double* u_ptr = nullptr;
    const double* v_ptr = nullptr;
    const double* k_read_ptr = nullptr;      // Read from (snapshot for CPU Jacobi)
    const double* omega_read_ptr = nullptr;
    double* k_write_ptr = nullptr;           // Write to
    double* omega_write_ptr = nullptr;
    const double* nu_t_ptr = nullptr;
    const double* wall_dist_ptr = nullptr;
    int u_stride_local = 0;
    int v_stride_local = 0;
    size_t cell_total_size = (size_t)mesh.total_Nx() * mesh.total_Ny();

#ifdef USE_GPU_OFFLOAD
    // GPU: use device pointers (parallel loop = Jacobi automatically)
    size_t u_total_size = (size_t)mesh.total_Ny() * (mesh.total_Nx() + 1);
    size_t v_total_size = (size_t)(mesh.total_Ny() + 1) * mesh.total_Nx();
    u_ptr = device_view->u_face;
    v_ptr = device_view->v_face;
    k_read_ptr = device_view->k;
    omega_read_ptr = device_view->omega;
    k_write_ptr = device_view->k;
    omega_write_ptr = device_view->omega;
    nu_t_ptr = device_view->nu_t;
    wall_dist_ptr = device_view->wall_distance;
    u_stride_local = device_view->u_stride;
    v_stride_local = device_view->v_stride;
#else
    (void)device_view;
    // CPU: Jacobi snapshot - copy k, omega to snapshot arrays first
    for (int jj = 0; jj < mesh.total_Ny(); ++jj) {
        for (int ii = 0; ii < mesh.total_Nx(); ++ii) {
            k_old_(ii, jj) = k(ii, jj);
            omega_old_(ii, jj) = omega(ii, jj);
        }
    }
    u_ptr = velocity.u_data().data();
    v_ptr = velocity.v_data().data();
    k_read_ptr = k_old_.data().data();
    omega_read_ptr = omega_old_.data().data();
    k_write_ptr = k.data().data();
    omega_write_ptr = omega.data().data();
    nu_t_ptr = nu_t_prev.data().data();
    wall_dist_ptr = wall_dist_.data().data();
    u_stride_local = velocity.u_stride();
    v_stride_local = velocity.v_stride();
#endif

    // ========================================================================
    // UNIFIED KERNEL LOOP - Single code path for CPU and GPU
    // ========================================================================
#ifdef USE_GPU_OFFLOAD
    #pragma omp target teams distribute parallel for \
        map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], \
                     k_read_ptr[0:cell_total_size], omega_read_ptr[0:cell_total_size], \
                     k_write_ptr[0:cell_total_size], omega_write_ptr[0:cell_total_size], \
                     nu_t_ptr[0:cell_total_size], wall_dist_ptr[0:cell_total_size])
#endif
        for (int cell_idx = 0; cell_idx < n_cells; ++cell_idx) {
            // Convert flat index to (ii, jj) with ghost offset
            const int i = cell_idx % Nx;
            const int j = cell_idx / Nx;
            const int ii = i + Ng;
            const int jj = j + Ng;

            // Cell indices for scalar fields
            const int idx_c  = jj * cell_stride + ii;
            const int idx_ip = jj * cell_stride + (ii + 1);
            const int idx_im = jj * cell_stride + (ii - 1);
            const int idx_jp = (jj + 1) * cell_stride + ii;
            const int idx_jm = (jj - 1) * cell_stride + ii;
            const int idx_ip2 = jj * cell_stride + (ii + 2);
            const int idx_im2 = jj * cell_stride + (ii - 2);
            const int idx_jp2 = (jj + 2) * cell_stride + ii;
            const int idx_jm2 = (jj - 2) * cell_stride + ii;

            // 1. GET CELL VALUES
            double k_c_raw = k_read_ptr[idx_c];
            double omega_c_raw = omega_read_ptr[idx_c];
            double y_c = wall_dist_ptr[idx_c];
            double nu_t_c_raw = nu_t_ptr[idx_c];
            double k_c = dmax(k_c_raw, k_min);
            double omega_c = dmax(omega_c_raw, omega_min);
            double nu_t_c = dmax(nu_t_c_raw, 0.0);

            // 2. VELOCITY GRADIENTS
            const int u_idx_c = jj * u_stride_local + ii;
            const int u_idx_ip = jj * u_stride_local + (ii + 1);
            const int u_idx_im = jj * u_stride_local + (ii - 1);
            const int u_idx_jp = (jj + 1) * u_stride_local + ii;
            const int u_idx_jm = (jj - 1) * u_stride_local + ii;
            const int v_idx_c = jj * v_stride_local + ii;
            const int v_idx_ip = jj * v_stride_local + (ii + 1);
            const int v_idx_im = jj * v_stride_local + (ii - 1);
            const int v_idx_jp = (jj + 1) * v_stride_local + ii;
            const int v_idx_jm = (jj - 1) * v_stride_local + ii;

            double dudx = (u_ptr[u_idx_ip] - u_ptr[u_idx_im]) * inv_2dx;
            double dudy = (u_ptr[u_idx_jp] - u_ptr[u_idx_jm]) * inv_2dy;
            double dvdx = (v_ptr[v_idx_ip] - v_ptr[v_idx_im]) * inv_2dx;
            double dvdy = (v_ptr[v_idx_jp] - v_ptr[v_idx_jm]) * inv_2dy;
            double u_face = u_ptr[u_idx_c];
            double v_face = v_ptr[v_idx_c];

            // 3. STRAIN RATE AND PRODUCTION
            double Sxx = dudx, Syy = dvdy, Sxy = 0.5 * (dudy + dvdx);
            double S2 = 2.0 * (Sxx*Sxx + Syy*Syy + 2.0*Sxy*Sxy);
            double P_k = dmin(2.0 * nu_t_c * S2, 10.0 * beta_star * k_c * omega_c);

            // 4. F1 AT CENTER
            double dkdx_c = (k_read_ptr[idx_ip] - k_read_ptr[idx_im]) * inv_2dx;
            double dkdy_c = (k_read_ptr[idx_jp] - k_read_ptr[idx_jm]) * inv_2dy;
            double domegadx_c = (omega_read_ptr[idx_ip] - omega_read_ptr[idx_im]) * inv_2dx;
            double domegady_c = (omega_read_ptr[idx_jp] - omega_read_ptr[idx_jm]) * inv_2dy;
            double F1_c = compute_F1_local(k_c_raw, omega_c_raw, y_c, dkdx_c, dkdy_c, domegadx_c, domegady_c, nu, beta_star, sigma_omega2, CD_min, k_min, omega_min);
            double beta_c = F1_c * beta1 + (1.0 - F1_c) * beta2;
            double alpha_c = F1_c * alpha1 + (1.0 - F1_c) * alpha2;
            double sigma_k_c = F1_c * sigma_k1 + (1.0 - F1_c) * sigma_k2;
            double sigma_omega_c = F1_c * sigma_omega1 + (1.0 - F1_c) * sigma_omega2;
            double nu_k_c = nu + sigma_k_c * nu_t_c;
            double nu_omega_c = nu + sigma_omega_c * nu_t_c;

            // 5. NEIGHBORS
            double k_ip_raw = k_read_ptr[idx_ip], omega_ip_raw = omega_read_ptr[idx_ip], y_ip = wall_dist_ptr[idx_ip];
            double nu_t_ip = dmax(nu_t_ptr[idx_ip], 0.0);
            double dkdx_ip = (k_read_ptr[idx_ip2] - k_read_ptr[idx_c]) * inv_2dx;
            double dkdy_ip = (k_read_ptr[(jj+1)*cell_stride+(ii+1)] - k_read_ptr[(jj-1)*cell_stride+(ii+1)]) * inv_2dy;
            double domegadx_ip = (omega_read_ptr[idx_ip2] - omega_read_ptr[idx_c]) * inv_2dx;
            double domegady_ip = (omega_read_ptr[(jj+1)*cell_stride+(ii+1)] - omega_read_ptr[(jj-1)*cell_stride+(ii+1)]) * inv_2dy;
            double F1_ip = compute_F1_local(k_ip_raw, omega_ip_raw, y_ip, dkdx_ip, dkdy_ip, domegadx_ip, domegady_ip, nu, beta_star, sigma_omega2, CD_min, k_min, omega_min);
            double nu_k_ip = nu + (F1_ip * sigma_k1 + (1.0 - F1_ip) * sigma_k2) * nu_t_ip;
            double nu_omega_ip = nu + (F1_ip * sigma_omega1 + (1.0 - F1_ip) * sigma_omega2) * nu_t_ip;

            double k_im_raw = k_read_ptr[idx_im], omega_im_raw = omega_read_ptr[idx_im], y_im = wall_dist_ptr[idx_im];
            double nu_t_im = dmax(nu_t_ptr[idx_im], 0.0);
            double dkdx_im = (k_read_ptr[idx_c] - k_read_ptr[idx_im2]) * inv_2dx;
            double dkdy_im = (k_read_ptr[(jj+1)*cell_stride+(ii-1)] - k_read_ptr[(jj-1)*cell_stride+(ii-1)]) * inv_2dy;
            double domegadx_im = (omega_read_ptr[idx_c] - omega_read_ptr[idx_im2]) * inv_2dx;
            double domegady_im = (omega_read_ptr[(jj+1)*cell_stride+(ii-1)] - omega_read_ptr[(jj-1)*cell_stride+(ii-1)]) * inv_2dy;
            double F1_im = compute_F1_local(k_im_raw, omega_im_raw, y_im, dkdx_im, dkdy_im, domegadx_im, domegady_im, nu, beta_star, sigma_omega2, CD_min, k_min, omega_min);
            double nu_k_im = nu + (F1_im * sigma_k1 + (1.0 - F1_im) * sigma_k2) * nu_t_im;
            double nu_omega_im = nu + (F1_im * sigma_omega1 + (1.0 - F1_im) * sigma_omega2) * nu_t_im;

            double k_jp_raw = k_read_ptr[idx_jp], omega_jp_raw = omega_read_ptr[idx_jp], y_jp = wall_dist_ptr[idx_jp];
            double nu_t_jp = dmax(nu_t_ptr[idx_jp], 0.0);
            double dkdx_jp = (k_read_ptr[(jj+1)*cell_stride+(ii+1)] - k_read_ptr[(jj+1)*cell_stride+(ii-1)]) * inv_2dx;
            double dkdy_jp = (k_read_ptr[idx_jp2] - k_read_ptr[idx_c]) * inv_2dy;
            double domegadx_jp = (omega_read_ptr[(jj+1)*cell_stride+(ii+1)] - omega_read_ptr[(jj+1)*cell_stride+(ii-1)]) * inv_2dx;
            double domegady_jp = (omega_read_ptr[idx_jp2] - omega_read_ptr[idx_c]) * inv_2dy;
            double F1_jp = compute_F1_local(k_jp_raw, omega_jp_raw, y_jp, dkdx_jp, dkdy_jp, domegadx_jp, domegady_jp, nu, beta_star, sigma_omega2, CD_min, k_min, omega_min);
            double nu_k_jp = nu + (F1_jp * sigma_k1 + (1.0 - F1_jp) * sigma_k2) * nu_t_jp;
            double nu_omega_jp = nu + (F1_jp * sigma_omega1 + (1.0 - F1_jp) * sigma_omega2) * nu_t_jp;

            double k_jm_raw = k_read_ptr[idx_jm], omega_jm_raw = omega_read_ptr[idx_jm], y_jm = wall_dist_ptr[idx_jm];
            double nu_t_jm = dmax(nu_t_ptr[idx_jm], 0.0);
            double dkdx_jm = (k_read_ptr[(jj-1)*cell_stride+(ii+1)] - k_read_ptr[(jj-1)*cell_stride+(ii-1)]) * inv_2dx;
            double dkdy_jm = (k_read_ptr[idx_c] - k_read_ptr[idx_jm2]) * inv_2dy;
            double domegadx_jm = (omega_read_ptr[(jj-1)*cell_stride+(ii+1)] - omega_read_ptr[(jj-1)*cell_stride+(ii-1)]) * inv_2dx;
            double domegady_jm = (omega_read_ptr[idx_c] - omega_read_ptr[idx_jm2]) * inv_2dy;
            double F1_jm = compute_F1_local(k_jm_raw, omega_jm_raw, y_jm, dkdx_jm, dkdy_jm, domegadx_jm, domegady_jm, nu, beta_star, sigma_omega2, CD_min, k_min, omega_min);
            double nu_k_jm = nu + (F1_jm * sigma_k1 + (1.0 - F1_jm) * sigma_k2) * nu_t_jm;
            double nu_omega_jm = nu + (F1_jm * sigma_omega1 + (1.0 - F1_jm) * sigma_omega2) * nu_t_jm;

            // 6. FACE-AVERAGED DIFFUSIVITIES
            double nu_k_e = 0.5 * (nu_k_c + nu_k_ip), nu_k_w = 0.5 * (nu_k_c + nu_k_im);
            double nu_k_n = 0.5 * (nu_k_c + nu_k_jp), nu_k_s = 0.5 * (nu_k_c + nu_k_jm);
            double nu_omega_e = 0.5 * (nu_omega_c + nu_omega_ip), nu_omega_w = 0.5 * (nu_omega_c + nu_omega_im);
            double nu_omega_n = 0.5 * (nu_omega_c + nu_omega_jp), nu_omega_s = 0.5 * (nu_omega_c + nu_omega_jm);

            // 7. ADVECTION (upwind)
            double adv_k, adv_omega;
            if (u_face >= 0) { adv_k = u_face * (k_c_raw - k_im_raw) / dx; adv_omega = u_face * (omega_c_raw - omega_im_raw) / dx; }
            else { adv_k = u_face * (k_ip_raw - k_c_raw) / dx; adv_omega = u_face * (omega_ip_raw - omega_c_raw) / dx; }
            if (v_face >= 0) { adv_k += v_face * (k_c_raw - k_jm_raw) / dy; adv_omega += v_face * (omega_c_raw - omega_jm_raw) / dy; }
            else { adv_k += v_face * (k_jp_raw - k_c_raw) / dy; adv_omega += v_face * (omega_jp_raw - omega_c_raw) / dy; }

            // 8. DIFFUSION
            double diff_k = (nu_k_e * (k_ip_raw - k_c_raw) - nu_k_w * (k_c_raw - k_im_raw)) / dx2
                          + (nu_k_n * (k_jp_raw - k_c_raw) - nu_k_s * (k_c_raw - k_jm_raw)) / dy2;
            double diff_omega = (nu_omega_e * (omega_ip_raw - omega_c_raw) - nu_omega_w * (omega_c_raw - omega_im_raw)) / dx2
                              + (nu_omega_n * (omega_jp_raw - omega_c_raw) - nu_omega_s * (omega_c_raw - omega_jm_raw)) / dy2;

            // 9. CROSS-DIFFUSION
            double CD = 2.0 * (1.0 - F1_c) * sigma_omega2 / omega_c * (dkdx_c * domegadx_c + dkdy_c * domegady_c);
            CD = dmax(CD, 0.0);

            // 10. TIME INTEGRATION
            double rhs_k = P_k - beta_star * k_c * omega_c + diff_k - adv_k;
            double rhs_omega = alpha_c * (omega_c / k_c) * P_k - beta_c * omega_c * omega_c + diff_omega - adv_omega + CD;
            double k_new = dclamp(k_c + dt * rhs_k, k_min, k_max);
            double omega_new = dclamp(omega_c + dt * rhs_omega, omega_min, omega_max);
            k_write_ptr[idx_c] = k_new;
            omega_write_ptr[idx_c] = omega_new;
        }

    // Apply wall boundary conditions (with GPU sync if needed)
#ifdef USE_GPU_OFFLOAD
    #pragma omp target update from(k_write_ptr[0:cell_total_size], omega_write_ptr[0:cell_total_size])
#endif
    apply_wall_bc_k(mesh, k);
    apply_wall_bc_omega(mesh, omega, k);
#ifdef USE_GPU_OFFLOAD
    #pragma omp target update to(k_write_ptr[0:cell_total_size], omega_write_ptr[0:cell_total_size])
#endif
}

void SSTKOmegaTransport::update(
    const Mesh& mesh,
    const VectorField& velocity,
    const ScalarField& k,
    const ScalarField& omega,
    ScalarField& nu_t,
    TensorField* tau_ij,
    const TurbulenceDeviceView* device_view)
{
    (void)tau_ij;  // SST closure doesn't compute explicit stresses

    ensure_initialized(mesh);

    // ========================================================================
    // UNIFIED SST CLOSURE IMPLEMENTATION
    // ========================================================================
    // Computes ν_t = a₁k / max(a₁ω, S*F₂) using identical arithmetic for CPU/GPU
    // ========================================================================

    const int Nx = mesh.Nx;
    const int Ny = mesh.Ny;
    const int Ng = mesh.Nghost;
    const int n_cells = Nx * Ny;
    const int cell_stride = mesh.total_Nx();
    const size_t cell_total_size = (size_t)mesh.total_Nx() * mesh.total_Ny();

    // Model constants
    const double nu = nu_;
    const double a1 = constants_.a1;
    const double beta_star = constants_.beta_star;
    const double k_min = constants_.k_min;
    const double omega_min = constants_.omega_min;
    const double nu_t_max = 1000.0 * nu;

    // Set up pointers
    const double* k_ptr = nullptr;
    const double* omega_ptr = nullptr;
    const double* dudx_ptr = nullptr;
    const double* dudy_ptr = nullptr;
    const double* dvdx_ptr = nullptr;
    const double* dvdy_ptr = nullptr;
    const double* wall_dist_ptr = nullptr;
    double* nu_t_ptr = nullptr;

#ifdef USE_GPU_OFFLOAD
    // GPU: compute gradients on GPU and use device pointers
    gpu_kernels::compute_gradients_from_mac_gpu(
        device_view->u_face,
        device_view->v_face,
        device_view->dudx,
        device_view->dudy,
        device_view->dvdx,
        device_view->dvdy,
        Nx, Ny, Ng,
        device_view->dx, device_view->dy,
        device_view->u_stride,
        device_view->v_stride,
        cell_stride,
        velocity.u_total_size(),
        velocity.v_total_size(),
        cell_total_size
    );
    k_ptr = device_view->k;
    omega_ptr = device_view->omega;
    dudx_ptr = device_view->dudx;
    dudy_ptr = device_view->dudy;
    dvdx_ptr = device_view->dvdx;
    dvdy_ptr = device_view->dvdy;
    wall_dist_ptr = device_view->wall_distance;
    nu_t_ptr = device_view->nu_t;
#else
    (void)device_view;
    // CPU: compute gradients on host and use host pointers
    compute_gradients_from_mac(mesh, velocity, dudx_, dudy_, dvdx_, dvdy_);
    k_ptr = k.data().data();
    omega_ptr = omega.data().data();
    dudx_ptr = dudx_.data().data();
    dudy_ptr = dudy_.data().data();
    dvdx_ptr = dvdx_.data().data();
    dvdy_ptr = dvdy_.data().data();
    wall_dist_ptr = wall_dist_.data().data();
    nu_t_ptr = nu_t.data().data();
#endif

    // ========================================================================
    // UNIFIED CLOSURE KERNEL - Single code path for CPU and GPU
    // ========================================================================
#ifdef USE_GPU_OFFLOAD
    #pragma omp target teams distribute parallel for \
        map(present: k_ptr[0:cell_total_size], omega_ptr[0:cell_total_size], \
                     dudx_ptr[0:cell_total_size], dudy_ptr[0:cell_total_size], \
                     dvdx_ptr[0:cell_total_size], dvdy_ptr[0:cell_total_size], \
                     wall_dist_ptr[0:cell_total_size], nu_t_ptr[0:cell_total_size])
#endif
    for (int cell_idx = 0; cell_idx < n_cells; ++cell_idx) {
        const int i = cell_idx % Nx;
        const int j = cell_idx / Nx;
        const int ii = i + Ng;
        const int jj = j + Ng;
        const int idx = jj * cell_stride + ii;

        // Read and clamp fields
        double k_val = dmax(k_ptr[idx], k_min);
        double omega_val = dmax(omega_ptr[idx], omega_min);
        double y_safe = dmax(wall_dist_ptr[idx], 1e-10);

        // Strain rate magnitude
        double Sxx = dudx_ptr[idx];
        double Syy = dvdy_ptr[idx];
        double Sxy = 0.5 * (dudy_ptr[idx] + dvdx_ptr[idx]);
        double S_mag = sqrt(2.0 * (Sxx*Sxx + Syy*Syy + 2.0*Sxy*Sxy));

        // F2 blending function
        double sqrt_k = sqrt(k_val);
        double term1 = 2.0 * sqrt_k / (beta_star * omega_val * y_safe);
        double term2 = 500.0 * nu / (y_safe * y_safe * omega_val);
        double arg2 = dmax(term1, term2);
        double F2 = tanh(arg2 * arg2);

        // SST eddy viscosity: ν_t = a₁k / max(a₁ω, SF₂)
        double denom = dmax(dmax(a1 * omega_val, S_mag * F2), 1e-20);
        double nu_t_val = dclamp(a1 * k_val / denom, 0.0, nu_t_max);

        nu_t_ptr[idx] = nu_t_val;
    }
}

// ============================================================================
// Standard k-ω Transport Implementation
// ============================================================================

KOmegaTransport::KOmegaTransport(const KOmegaConstants& constants)
    : constants_(constants)
{
    closure_ = std::make_unique<BoussinesqClosure>();
}

void KOmegaTransport::set_closure(std::unique_ptr<TurbulenceClosure> closure) {
    closure_ = std::move(closure);
    if (closure_) {
        closure_->set_nu(nu_);
        closure_->set_delta(delta_);
    }
}

void KOmegaTransport::ensure_initialized(const Mesh& mesh) {
    if (!initialized_) {
        dudx_ = ScalarField(mesh);
        dudy_ = ScalarField(mesh);
        dvdx_ = ScalarField(mesh);
        dvdy_ = ScalarField(mesh);
        P_k_ = ScalarField(mesh);
        initialized_ = true;
    }
}

void KOmegaTransport::initialize(const Mesh& mesh, const VectorField& velocity) {
    ensure_initialized(mesh);
    (void)velocity;
    
    if (closure_) {
        closure_->set_nu(nu_);
        closure_->set_delta(delta_);
    }
}

void KOmegaTransport::initialize_gpu_buffers(const Mesh& mesh) {
#ifdef USE_GPU_OFFLOAD
    // KOmegaTransport uses solver-owned GPU buffers via device_view
    // No separate buffers to allocate, but mark as GPU-ready so solver knows
    // we will use device_view GPU kernels and won't need host→device syncs
    gpu::verify_device_available();
    gpu_ready_ = true;
#else
    (void)mesh;
    gpu_ready_ = false;
#endif
}

void KOmegaTransport::cleanup_gpu_buffers() {
    gpu_ready_ = false;
}

void KOmegaTransport::advance_turbulence(
    const Mesh& mesh,
    const VectorField& velocity,
    double dt,
    ScalarField& k,
    ScalarField& omega,
    const ScalarField& nu_t_prev,
    const TurbulenceDeviceView* device_view)
{
    TIMED_SCOPE("komega_transport");
    
    ensure_initialized(mesh);
    
#ifdef USE_GPU_OFFLOAD
    // GPU path using device_view and komega_transport_step_gpu kernel
    if (device_view && device_view->is_valid()) {
        // All pointers are solver-owned and already device-resident
        // No data upload/download needed!
        const int Nx = mesh.Nx;
        const int Ny = mesh.Ny;
        const int Ng = mesh.Nghost;
        const int cell_stride = mesh.total_Nx();
        const size_t cell_total_size = (size_t)mesh.total_Nx() * mesh.total_Ny();
        const size_t u_total_size = (size_t)mesh.total_Ny() * (mesh.total_Nx() + 1);
        const size_t v_total_size = (size_t)(mesh.total_Ny() + 1) * mesh.total_Nx();
        
        gpu_kernels::komega_transport_step_gpu(
            device_view->u_face, device_view->v_face,   // Velocity on GPU
            device_view->k, device_view->omega,          // k, omega on GPU (in/out)
            device_view->nu_t,                           // nu_t_prev on GPU (uses ghost+stride)
            Nx, Ny, Ng,
            cell_stride,
            device_view->u_stride, device_view->v_stride,
            cell_total_size,
            u_total_size, v_total_size,
            mesh.dx, mesh.dy, dt,
            nu_, constants_.sigma_k, constants_.sigma_omega,
            constants_.beta, constants_.beta_star, constants_.alpha,
            constants_.k_min, constants_.k_max,
            constants_.omega_min, constants_.omega_max
        );
        
        // Transport PDE done entirely on GPU - no CPU sync needed
        // k and omega ScalarFields will be synced by solver if needed for output
        return;
    }
#else
    (void)device_view;
#endif
    
    // Host path (only used when GPU offload disabled or device_view invalid)
    compute_gradients_from_mac(mesh, velocity, dudx_, dudy_, dvdx_, dvdy_);
    
    const double dx = mesh.dx;
    const double dy = mesh.dy;
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double u = velocity.u(i, j);
            double v = velocity.v(i, j);
            double k_old = std::max(constants_.k_min, k(i, j));
            double omega_old = std::max(constants_.omega_min, omega(i, j));
            double nu_t_loc = std::max(0.0, nu_t_prev(i, j));
            
            // Strain rate
            double Sxx = dudx_(i, j);
            double Syy = dvdy_(i, j);
            double Sxy = 0.5 * (dudy_(i, j) + dvdx_(i, j));
            double S2 = 2.0 * (Sxx*Sxx + Syy*Syy + 2.0*Sxy*Sxy);
            
            // Production
            double P_k = 2.0 * nu_t_loc * S2;
            
            // Effective viscosities
            double nu_k = nu_ + constants_.sigma_k * nu_t_loc;
            double nu_omega = nu_ + constants_.sigma_omega * nu_t_loc;
            
            // Advection (upwind)
            double dkdx = (u >= 0) ? (k(i,j) - k(i-1,j)) / dx : (k(i+1,j) - k(i,j)) / dx;
            double dkdy = (v >= 0) ? (k(i,j) - k(i,j-1)) / dy : (k(i,j+1) - k(i,j)) / dy;
            double domegadx = (u >= 0) ? (omega(i,j) - omega(i-1,j)) / dx : (omega(i+1,j) - omega(i,j)) / dx;
            double domegady = (v >= 0) ? (omega(i,j) - omega(i,j-1)) / dy : (omega(i,j+1) - omega(i,j)) / dy;
            
            double adv_k = u * dkdx + v * dkdy;
            double adv_omega = u * domegadx + v * domegady;
            
            // Diffusion
            double diff_k = nu_k * ((k(i+1,j) - 2.0*k(i,j) + k(i-1,j)) / dx2
                                  + (k(i,j+1) - 2.0*k(i,j) + k(i,j-1)) / dy2);
            double diff_omega = nu_omega * ((omega(i+1,j) - 2.0*omega(i,j) + omega(i-1,j)) / dx2
                                          + (omega(i,j+1) - 2.0*omega(i,j) + omega(i,j-1)) / dy2);
            
            // RHS
            double rhs_k = P_k - constants_.beta_star * k_old * omega_old + diff_k - adv_k;
            double rhs_omega = constants_.alpha * (omega_old / k_old) * P_k 
                             - constants_.beta * omega_old * omega_old
                             + diff_omega - adv_omega;
            
            // Update
            double k_new = k_old + dt * rhs_k;
            double omega_new = omega_old + dt * rhs_omega;
            
            k(i, j) = std::min(std::max(k_new, constants_.k_min), constants_.k_max);
            omega(i, j) = std::min(std::max(omega_new, constants_.omega_min), constants_.omega_max);
        }
    }
}

void KOmegaTransport::update(
    const Mesh& mesh,
    const VectorField& velocity,
    const ScalarField& k,
    const ScalarField& omega,
    ScalarField& nu_t,
    TensorField* tau_ij,
    const TurbulenceDeviceView* device_view)
{
#ifdef USE_GPU_OFFLOAD
    // GPU path using device_view (Phase 2)
    // Run GPU closure for default Boussinesq or when no custom closure is set
    bool use_gpu_boussinesq = (!closure_ || 
                               (closure_ && closure_->name() == "Boussinesq") ||
                               dynamic_cast<BoussinesqClosure*>(closure_.get()) != nullptr);
    
    if (device_view && device_view->is_valid() && use_gpu_boussinesq) {
        // Direct Boussinesq closure on GPU
        const int Nx = mesh.Nx;
        const int Ny = mesh.Ny;
        const int Ng = mesh.Nghost;
        const int stride = mesh.total_Nx();
        const size_t total_size = (size_t)mesh.total_Nx() * mesh.total_Ny();
        
        gpu_kernels::compute_boussinesq_closure_gpu(
            device_view->k,           // Already on device
            device_view->omega,       // Already on device
            device_view->nu_t,        // Output on device
            Nx, Ny, Ng, stride, total_size,
            nu_,                      // Laminar viscosity
            constants_.k_min, constants_.omega_min,
            1000.0                    // nu_t_max multiplier
        );
        
        // No CPU sync needed - nu_t stays on device for solver
        return;
    }
#else
    (void)device_view;
#endif
    
    // CPU path or custom closure
    if (closure_) {
        closure_->compute_nu_t(mesh, velocity, k, omega, nu_t, tau_ij);
    } else {
        // Fallback: CPU Boussinesq closure
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double k_loc = std::max(constants_.k_min, k(i, j));
                double omega_loc = std::max(constants_.omega_min, omega(i, j));
                nu_t(i, j) = k_loc / omega_loc;
            }
        }
    }
}

// ============================================================================
// Factory Functions
// ============================================================================

std::unique_ptr<TurbulenceModel> create_transport_model(
    const std::string& name,
    const std::string& closure_name)
{
    if (name == "SST" || name == "SSTKOmega" || name == "sst") {
        auto model = std::make_unique<SSTKOmegaTransport>();
        if (closure_name != "SST" && closure_name != "default") {
            model->set_closure(create_closure(closure_name));
        }
        return model;
    } else if (name == "KOmega" || name == "komega" || name == "k-omega") {
        auto model = std::make_unique<KOmegaTransport>();
        if (closure_name != "Boussinesq" && closure_name != "default") {
            model->set_closure(create_closure(closure_name));
        }
        return model;
    }
    
    return nullptr;
}

std::unique_ptr<TurbulenceClosure> create_closure(const std::string& name) {
    if (name == "Boussinesq" || name == "boussinesq" || name == "linear") {
        return std::make_unique<BoussinesqClosure>();
    } else if (name == "SST" || name == "sst") {
        return std::make_unique<SSTClosure>();
    }
    
    return nullptr;
}

} // namespace nncfd

