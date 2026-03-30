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
#include "numerics.hpp"
#include "timing.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

namespace nncfd {

// ============================================================================
// Unified SST k-ω Transport Cell Kernel - compiles for both CPU and GPU
// ============================================================================
#ifdef USE_GPU_OFFLOAD
#pragma omp declare target
#endif

/// Compute SST k-ω transport update for a single cell
/// This kernel computes all SST physics: gradients, F1 blending, production,
/// advection, diffusion, cross-diffusion, and time integration.
///
/// @param cell_idx       Cell indices: center, +x, -x, +y, -y
/// @param u_idx, v_idx   Velocity face indices for gradient computation
/// @param u_ptr, v_ptr   Velocity face arrays (MAC grid)
/// @param k_ptr          TKE array (read)
/// @param omega_ptr      Specific dissipation array (read)
/// @param nu_t_ptr       Eddy viscosity array (read)
/// @param wall_dist_ptr  Wall distance array
/// @param dx, dy, dt     Grid spacing and time step
/// @param inv_2dx, inv_2dy  Precomputed gradient factors
/// @param dx2, dy2       Precomputed dx², dy²
/// @param nu             Molecular viscosity
/// @param beta_star, beta1, beta2, alpha1, alpha2  SST constants
/// @param sigma_k1, sigma_k2, sigma_omega1, sigma_omega2  SST diffusion constants
/// @param k_min, k_max, omega_min, omega_max, CD_min  Limiting constants
/// @param k_new_out, omega_new_out  [out] Updated values
inline void sst_transport_cell_kernel(
    // Cell indices (x/y neighbors + z neighbors)
    int idx_c, int idx_ip, int idx_im, int idx_jp, int idx_jm,
    int idx_kp, int idx_km,
    // Velocity face indices
    int u_idx_ip, int u_idx_im, int u_idx_jp, int u_idx_jm,
    int v_idx_ip, int v_idx_im, int v_idx_jp, int v_idx_jm,
    int u_idx_c, int u_idx_c1, int v_idx_c, int v_idx_c1,
    int w_idx_c, int w_idx_c1,
    // Data pointers
    const double* u_ptr, const double* v_ptr, const double* w_ptr,
    const double* k_ptr, const double* omega_ptr,
    const double* nu_t_ptr, const double* wall_dist_ptr,
    // Grid parameters
    double dx, double dy, double dz, double dt,
    double inv_2dx, double inv_2dy, double inv_2dz,
    double dx2, double dy2, double dz2,
    bool is_3d,
    // Model constants
    double nu, double beta_star, double beta1, double beta2,
    double alpha1, double alpha2, double sigma_k1, double sigma_k2,
    double sigma_omega1, double sigma_omega2,
    double k_min, double k_max, double omega_min, double omega_max, double CD_min,
    // Outputs
    double& k_new_out, double& omega_new_out)
{
    // Velocity gradients from MAC grid
    double dudx_v = (u_ptr[u_idx_ip] - u_ptr[u_idx_im]) * inv_2dx;
    double dudy_v = (u_ptr[u_idx_jp] - u_ptr[u_idx_jm]) * inv_2dy;
    double dvdx_v = (v_ptr[v_idx_ip] - v_ptr[v_idx_im]) * inv_2dx;
    double dvdy_v = (v_ptr[v_idx_jp] - v_ptr[v_idx_jm]) * inv_2dy;

    // Cell-centered velocity (interpolate from faces)
    double u_c = 0.5 * (u_ptr[u_idx_c] + u_ptr[u_idx_c1]);
    double v_c = 0.5 * (v_ptr[v_idx_c] + v_ptr[v_idx_c1]);
    double w_c = 0.0;
    if (is_3d && w_ptr) {
        w_c = 0.5 * (w_ptr[w_idx_c] + w_ptr[w_idx_c1]);
    }

    // Get cell values with limiting
    double k_c = k_ptr[idx_c];
    double omega_c = omega_ptr[idx_c];
    double y_wall = wall_dist_ptr[idx_c];
    double nu_t_c = nu_t_ptr[idx_c];

    k_c = (k_c > k_min) ? k_c : k_min;
    omega_c = (omega_c > omega_min) ? omega_c : omega_min;
    double y_safe = (y_wall > 1e-10) ? y_wall : 1e-10;
    nu_t_c = (nu_t_c > 0.0) ? nu_t_c : 0.0;

    // Strain rate magnitude squared (2D terms)
    double Sxx = dudx_v;
    double Syy = dvdy_v;
    double Sxy = 0.5 * (dudy_v + dvdx_v);
    double S2 = 2.0 * (Sxx*Sxx + Syy*Syy + 2.0*Sxy*Sxy);

    // Add 3D strain rate contribution: Szz diagonal term
    // Cross terms (Sxz, Syz) require additional face indices not available here.
    // The full 3D gradient computation is done in compute_gradients_from_mac_gpu
    // and used by the SST closure. For transport production, Szz is the dominant
    // 3D contribution (captures spanwise compression/expansion).
    if (is_3d && w_ptr) {
        double dwdz_v = (w_ptr[w_idx_c1] - w_ptr[w_idx_c]) / dz;
        double Szz = dwdz_v;
        S2 += 2.0 * Szz * Szz;
    }

    // Gradients for cross-diffusion and F1
    double dkdx = (k_ptr[idx_ip] - k_ptr[idx_im]) * inv_2dx;
    double dkdy = (k_ptr[idx_jp] - k_ptr[idx_jm]) * inv_2dy;
    double domegadx = (omega_ptr[idx_ip] - omega_ptr[idx_im]) * inv_2dx;
    double domegady = (omega_ptr[idx_jp] - omega_ptr[idx_jm]) * inv_2dy;

    // Add z-direction gradients for cross-diffusion in 3D
    double dkdz = 0.0;
    double domegadz = 0.0;
    if (is_3d) {
        dkdz = (k_ptr[idx_kp] - k_ptr[idx_km]) * inv_2dz;
        domegadz = (omega_ptr[idx_kp] - omega_ptr[idx_km]) * inv_2dz;
    }

    // Cross-diffusion term for F1 calculation
    double grad_k_dot_grad_omega = dkdx * domegadx + dkdy * domegady + dkdz * domegadz;
    double CD_omega = 2.0 * sigma_omega2 / omega_c * grad_k_dot_grad_omega;
    CD_omega = (CD_omega > CD_min) ? CD_omega : CD_min;

    // F1 blending function
    double sqrt_k = std::sqrt(k_c);
    double arg1_1 = sqrt_k / (beta_star * omega_c * y_safe);
    double arg1_2 = 500.0 * nu / (y_safe * y_safe * omega_c);
    double arg1_3 = 4.0 * sigma_omega2 * k_c / (CD_omega * y_safe * y_safe);
    double arg1 = arg1_1;
    arg1 = (arg1 > arg1_2) ? arg1 : arg1_2;
    arg1 = (arg1 < arg1_3) ? arg1 : arg1_3;
    double F1 = std::tanh(arg1 * arg1 * arg1 * arg1);

    // Blended constants
    double beta = F1 * beta1 + (1.0 - F1) * beta2;
    double alpha = F1 * alpha1 + (1.0 - F1) * alpha2;
    double sigma_k = F1 * sigma_k1 + (1.0 - F1) * sigma_k2;
    double sigma_omega = F1 * sigma_omega1 + (1.0 - F1) * sigma_omega2;

    // Effective diffusivities
    double nu_k = nu + sigma_k * nu_t_c;
    double nu_omega_eff = nu + sigma_omega * nu_t_c;

    // Production (limited)
    double P_k = 2.0 * nu_t_c * S2;
    double P_k_limit = 10.0 * beta_star * k_c * omega_c;
    P_k = (P_k < P_k_limit) ? P_k : P_k_limit;

    // Advection (upwind)
    double adv_k, adv_omega;
    if (u_c >= 0) {
        adv_k = u_c * (k_c - k_ptr[idx_im]) / dx;
        adv_omega = u_c * (omega_c - omega_ptr[idx_im]) / dx;
    } else {
        adv_k = u_c * (k_ptr[idx_ip] - k_c) / dx;
        adv_omega = u_c * (omega_ptr[idx_ip] - omega_c) / dx;
    }
    if (v_c >= 0) {
        adv_k += v_c * (k_c - k_ptr[idx_jm]) / dy;
        adv_omega += v_c * (omega_c - omega_ptr[idx_jm]) / dy;
    } else {
        adv_k += v_c * (k_ptr[idx_jp] - k_c) / dy;
        adv_omega += v_c * (omega_ptr[idx_jp] - omega_c) / dy;
    }
    // z-advection in 3D
    if (is_3d) {
        if (w_c >= 0) {
            adv_k += w_c * (k_c - k_ptr[idx_km]) / dz;
            adv_omega += w_c * (omega_c - omega_ptr[idx_km]) / dz;
        } else {
            adv_k += w_c * (k_ptr[idx_kp] - k_c) / dz;
            adv_omega += w_c * (omega_ptr[idx_kp] - omega_c) / dz;
        }
    }

    // Diffusion (central difference)
    double diff_k = nu_k * ((k_ptr[idx_ip] - 2.0*k_c + k_ptr[idx_im]) / dx2
                          + (k_ptr[idx_jp] - 2.0*k_c + k_ptr[idx_jm]) / dy2);
    double diff_omega = nu_omega_eff * ((omega_ptr[idx_ip] - 2.0*omega_c + omega_ptr[idx_im]) / dx2
                                       + (omega_ptr[idx_jp] - 2.0*omega_c + omega_ptr[idx_jm]) / dy2);
    // z-diffusion in 3D
    if (is_3d) {
        diff_k += nu_k * (k_ptr[idx_kp] - 2.0*k_c + k_ptr[idx_km]) / dz2;
        diff_omega += nu_omega_eff * (omega_ptr[idx_kp] - 2.0*omega_c + omega_ptr[idx_km]) / dz2;
    }

    // Cross-diffusion term for omega equation
    double CD = 2.0 * (1.0 - F1) * sigma_omega2 / omega_c
              * grad_k_dot_grad_omega;
    CD = (CD > 0.0) ? CD : 0.0;

    // Point-implicit time integration: treat destruction terms implicitly
    // for unconditional stability of stiff source terms at wall cells.
    //   k:     dk/dt = P_k - β* k ω + diff - adv
    //   ω:     dω/dt = α(ω/k)P_k - β ω² + diff - adv + CD
    // Implicit form: k_new = (k + dt*source) / (1 + dt*sink_coeff)
    double source_k = P_k + diff_k - adv_k;
    double sink_k = beta_star * omega_c;  // destruction coefficient for k

    double source_omega = alpha * (omega_c / k_c) * P_k
                        + diff_omega - adv_omega + CD;
    double sink_omega = beta * omega_c;  // destruction coefficient for ω

    double k_new = (k_c + dt * source_k) / (1.0 + dt * sink_k);
    double omega_new = (omega_c + dt * source_omega) / (1.0 + dt * sink_omega);

    // Clipping
    k_new = (k_new > k_min) ? k_new : k_min;
    k_new = (k_new < k_max) ? k_new : k_max;
    omega_new = (omega_new > omega_min) ? omega_new : omega_min;
    omega_new = (omega_new < omega_max) ? omega_new : omega_max;

    k_new_out = k_new;
    omega_new_out = omega_new;
}

#ifdef USE_GPU_OFFLOAD
#pragma omp end declare target
#endif
// ============================================================================

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
    
    using namespace numerics;

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double k_loc = std::max(K_FLOOR, k(i, j));
            double omega_loc = std::max(OMEGA_FLOOR, omega(i, j));

            // ν_t = k / ω (for k-ω models)
            // Equivalent to ν_t = C_μ k² / ε where ε = C_μ k ω
            // Use bounded_ratio to prevent overflow from small omega
            double nu_t_loc = bounded_ratio(k_loc, omega_loc, OMEGA_FLOOR, 1000.0 * nu_);

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

    // NOTE: GPU path for SST closure is handled by the caller
    // (SSTKOmegaTransport::update) which calls compute_sst_closure_gpu()
    // with device_view pointers and map(present:). This CPU path is the
    // fallback for non-GPU builds or custom closures.

    // CPU path
    using namespace numerics;

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
            double nu_t_loc = safe_divide(a1 * k_loc, denom, K_FLOOR);

            // Clipping
            nu_t_loc = std::clamp(nu_t_loc, 0.0, 1000.0 * nu_);

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
    // Clamp to omega_max to prevent overflow on fine grids where y1 → 0
    for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
        int j_first = mesh.j_begin();
        double y1 = std::max(mesh.wall_distance(i, j_first), 1e-10);
        double omega_wall = 10.0 * 6.0 * nu_ / (beta1 * y1 * y1);
        omega_wall = std::min(omega_wall, constants_.omega_max);

        for (int g = 0; g < Ng; ++g) {
            int j_ghost = g;
            omega(i, j_ghost) = 2.0 * omega_wall - omega(i, j_first);
        }
    }
    
    // Top wall
    // Clamp to omega_max to prevent overflow on fine grids where y1 → 0
    for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
        int j_last = mesh.j_end() - 1;
        double y1 = std::max(mesh.wall_distance(i, j_last), 1e-10);
        double omega_wall = 10.0 * 6.0 * nu_ / (beta1 * y1 * y1);
        omega_wall = std::min(omega_wall, constants_.omega_max);

        for (int g = 0; g < Ng; ++g) {
            int j_ghost = mesh.Ny + Ng + g;
            omega(i, j_ghost) = 2.0 * omega_wall - omega(i, j_last);
        }
    }
}

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
    
#ifdef USE_GPU_OFFLOAD
    // GPU path using device_view (no model-owned flat buffers!)
    if (device_view && device_view->is_valid()) {
        // Use the existing fused SST transport GPU implementation,
        // but operate directly on solver-owned device-resident data
        const int Nx = mesh.Nx;
        const int Ny = mesh.Ny;
        const int Nz = mesh.Nz;
        const int Ng = mesh.Nghost;
        const int n_cells = Nx * Ny * Nz;
        const int cell_stride = mesh.total_Nx();
        const int cell_plane_stride = mesh.total_Nx() * mesh.total_Ny();
        [[maybe_unused]] const size_t cell_total_size = (size_t)mesh.total_Nx() * mesh.total_Ny() * mesh.total_Nz();
        [[maybe_unused]] const size_t u_total_size = device_view->u_total;
        [[maybe_unused]] const size_t v_total_size = device_view->v_total;
        [[maybe_unused]] const size_t w_total_size = device_view->w_total;
        const bool is_3d = (Nz > 1);

        const double dx = mesh.dx;
        const double dy = mesh.dy;
        const double dz = mesh.dz;
        const double dx2 = dx * dx;
        const double dy2 = dy * dy;
        const double dz2 = dz * dz;
        const double inv_2dx = 1.0 / (2.0 * dx);
        const double inv_2dy = 1.0 / (2.0 * dy);
        const double inv_2dz = (dz > 0.0) ? 1.0 / (2.0 * dz) : 0.0;

        // Copy model constants to local for GPU capture
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

        // Get device pointers from device_view
        const double* u_ptr = device_view->u_face;
        const double* v_ptr = device_view->v_face;
        const double* w_ptr = device_view->w_face;
        double* k_ptr = device_view->k;
        double* omega_ptr = device_view->omega;
        const double* nu_t_ptr = device_view->nu_t;
        const double* wall_dist_ptr = device_view->wall_distance;
        const int u_stride = device_view->u_stride;
        const int v_stride = device_view->v_stride;
        const int w_stride = device_view->w_stride;
        const int u_plane_stride = device_view->u_plane_stride;
        const int v_plane_stride = device_view->v_plane_stride;
        const int w_plane_stride = device_view->w_plane_stride;

        // GPU kernel: SST k-ω transport using unified kernel (3D-capable)
        // Note: w_ptr mapping only for 3D (w may not be GPU-mapped for 2D Nz=1)
        if (is_3d) {
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], \
                         k_ptr[0:cell_total_size], omega_ptr[0:cell_total_size], \
                         nu_t_ptr[0:cell_total_size], wall_dist_ptr[0:cell_total_size], \
                         w_ptr[0:w_total_size])
        for (int flat_idx = 0; flat_idx < n_cells; ++flat_idx) {
            const int i = flat_idx % Nx;
            const int j = (flat_idx / Nx) % Ny;
            const int kk = flat_idx / (Nx * Ny);
            const int ii = i + Ng;
            const int jj = j + Ng;
            const int kkz = kk + Ng;

            // Cell indices (3D with plane stride)
            const int idx_c  = kkz * cell_plane_stride + jj * cell_stride + ii;
            const int idx_ip = kkz * cell_plane_stride + jj * cell_stride + (ii + 1);
            const int idx_im = kkz * cell_plane_stride + jj * cell_stride + (ii - 1);
            const int idx_jp = kkz * cell_plane_stride + (jj + 1) * cell_stride + ii;
            const int idx_jm = kkz * cell_plane_stride + (jj - 1) * cell_stride + ii;
            const int idx_kp = (kkz + 1) * cell_plane_stride + jj * cell_stride + ii;
            const int idx_km = (kkz - 1) * cell_plane_stride + jj * cell_stride + ii;

            // Velocity face indices for gradients (u at x-faces)
            const int u_idx_ip = kkz * u_plane_stride + jj * u_stride + (ii + 1);
            const int u_idx_im = kkz * u_plane_stride + jj * u_stride + (ii - 1);
            const int u_idx_jp = kkz * u_plane_stride + (jj + 1) * u_stride + ii;
            const int u_idx_jm = kkz * u_plane_stride + (jj - 1) * u_stride + ii;
            // v at y-faces
            const int v_idx_ip = kkz * v_plane_stride + jj * v_stride + (ii + 1);
            const int v_idx_im = kkz * v_plane_stride + jj * v_stride + (ii - 1);
            const int v_idx_jp = kkz * v_plane_stride + (jj + 1) * v_stride + ii;
            const int v_idx_jm = kkz * v_plane_stride + (jj - 1) * v_stride + ii;

            // Velocity face indices for cell-center interpolation
            const int u_idx_c  = kkz * u_plane_stride + jj * u_stride + ii;
            const int u_idx_c1 = kkz * u_plane_stride + jj * u_stride + (ii + 1);
            const int v_idx_c  = kkz * v_plane_stride + jj * v_stride + ii;
            const int v_idx_c1 = kkz * v_plane_stride + (jj + 1) * v_stride + ii;
            // w at z-faces for cell-center interpolation
            const int w_idx_c  = kkz * w_plane_stride + jj * w_stride + ii;
            const int w_idx_c1 = (kkz + 1) * w_plane_stride + jj * w_stride + ii;

            // Call unified kernel
            double k_new, omega_new;
            sst_transport_cell_kernel(
                idx_c, idx_ip, idx_im, idx_jp, idx_jm,
                idx_kp, idx_km,
                u_idx_ip, u_idx_im, u_idx_jp, u_idx_jm,
                v_idx_ip, v_idx_im, v_idx_jp, v_idx_jm,
                u_idx_c, u_idx_c1, v_idx_c, v_idx_c1,
                w_idx_c, w_idx_c1,
                u_ptr, v_ptr, w_ptr, k_ptr, omega_ptr, nu_t_ptr, wall_dist_ptr,
                dx, dy, dz, dt, inv_2dx, inv_2dy, inv_2dz, dx2, dy2, dz2,
                is_3d,
                nu, beta_star, beta1, beta2, alpha1, alpha2,
                sigma_k1, sigma_k2, sigma_omega1, sigma_omega2,
                k_min, k_max, omega_min, omega_max, CD_min,
                k_new, omega_new);

            k_ptr[idx_c] = k_new;
            omega_ptr[idx_c] = omega_new;
        }
        } else {
        // 2D version: no w_ptr in map clause (w may not be GPU-mapped for Nz=1)
        // Use plane 0 indexing (no Ng offset) to match all other 2D GPU kernels.
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], \
                         k_ptr[0:cell_total_size], omega_ptr[0:cell_total_size], \
                         nu_t_ptr[0:cell_total_size], wall_dist_ptr[0:cell_total_size])
        for (int flat_idx = 0; flat_idx < n_cells; ++flat_idx) {
            const int i = flat_idx % Nx;
            const int j = (flat_idx / Nx) % Ny;
            const int ii = i + Ng;
            const int jj = j + Ng;
            // 2D: plane 0 indexing (no z-ghost offset)

            const int idx_c  = jj * cell_stride + ii;
            const int idx_ip = jj * cell_stride + (ii + 1);
            const int idx_im = jj * cell_stride + (ii - 1);
            const int idx_jp = (jj + 1) * cell_stride + ii;
            const int idx_jm = (jj - 1) * cell_stride + ii;
            const int idx_kp = idx_c;  // dummy for 2D
            const int idx_km = idx_c;

            const int u_idx_ip = jj * u_stride + (ii + 1);
            const int u_idx_im = jj * u_stride + (ii - 1);
            const int u_idx_jp = (jj + 1) * u_stride + ii;
            const int u_idx_jm = (jj - 1) * u_stride + ii;
            const int v_idx_ip = jj * v_stride + (ii + 1);
            const int v_idx_im = jj * v_stride + (ii - 1);
            const int v_idx_jp = (jj + 1) * v_stride + ii;
            const int v_idx_jm = (jj - 1) * v_stride + ii;
            const int u_idx_c  = jj * u_stride + ii;
            const int u_idx_c1 = jj * u_stride + (ii + 1);
            const int v_idx_c  = jj * v_stride + ii;
            const int v_idx_c1 = (jj + 1) * v_stride + ii;
            const int w_idx_c = 0;   // dummy
            const int w_idx_c1 = 0;  // dummy

            double k_new, omega_new;
            sst_transport_cell_kernel(
                idx_c, idx_ip, idx_im, idx_jp, idx_jm,
                idx_kp, idx_km,
                u_idx_ip, u_idx_im, u_idx_jp, u_idx_jm,
                v_idx_ip, v_idx_im, v_idx_jp, v_idx_jm,
                u_idx_c, u_idx_c1, v_idx_c, v_idx_c1,
                w_idx_c, w_idx_c1,
                u_ptr, v_ptr, nullptr, k_ptr, omega_ptr, nu_t_ptr, wall_dist_ptr,
                dx, dy, dz, dt, inv_2dx, inv_2dy, inv_2dz, dx2, dy2, dz2,
                false,  // is_3d = false
                nu, beta_star, beta1, beta2, alpha1, alpha2,
                sigma_k1, sigma_k2, sigma_omega1, sigma_omega2,
                k_min, k_max, omega_min, omega_max, CD_min,
                k_new, omega_new);

            k_ptr[idx_c] = k_new;
            omega_ptr[idx_c] = omega_new;
        }
        } // end if (is_3d) else

        // Apply wall BCs directly on GPU — no CPU roundtrip.
        // k BC: ghost = -interior (linear extrapolation gives k=0 at wall)
        // Loop over all x-i and z-k cells (3D)
        const int total_Nx_bc = cell_stride;  // == mesh.total_Nx()
        const int total_Nz_bc = mesh.total_Nz();
        const int bc_count = total_Nx_bc * total_Nz_bc;
        #pragma omp target teams distribute parallel for \
            map(present: k_ptr[0:cell_total_size])
        for (int bc_idx = 0; bc_idx < bc_count; ++bc_idx) {
            const int i = bc_idx % total_Nx_bc;
            const int kz = bc_idx / total_Nx_bc;
            for (int g = 0; g < Ng; ++g) {
                k_ptr[kz * cell_plane_stride + g * cell_stride + i]
                    = -k_ptr[kz * cell_plane_stride + Ng * cell_stride + i];
                k_ptr[kz * cell_plane_stride + (Ny + Ng + g) * cell_stride + i]
                    = -k_ptr[kz * cell_plane_stride + (Ny + Ng - 1) * cell_stride + i];
            }
        }

        // omega BC: ghost = 2*omega_wall - interior, where omega_wall = 60*nu/(beta1*y1^2)
        const double* wall_ptr   = device_view->wall_distance;
        [[maybe_unused]] const int wall_total = device_view->cell_total;
        const double nu_bc       = nu;
        const double beta1_bc    = beta1;
        const double omega_max_bc = omega_max;
        const int omega_bc_count = Nx * total_Nz_bc;
        #pragma omp target teams distribute parallel for \
            map(present: omega_ptr[0:cell_total_size], wall_ptr[0:wall_total])
        for (int bc_idx = 0; bc_idx < omega_bc_count; ++bc_idx) {
            const int ix = bc_idx % Nx;
            const int kz = bc_idx / Nx;
            const int i = ix + Ng;
            for (int g = 0; g < Ng; ++g) {
                // Bottom wall
                {
                    const int idx_ghost = kz * cell_plane_stride + g * cell_stride + i;
                    const int idx_int   = kz * cell_plane_stride + Ng * cell_stride + i;
                    double y1 = wall_ptr[idx_int];
                    y1 = (y1 > 1e-10) ? y1 : 1e-10;
                    double ow = 60.0 * nu_bc / (beta1_bc * y1 * y1);
                    ow = (ow < omega_max_bc) ? ow : omega_max_bc;
                    omega_ptr[idx_ghost] = 2.0 * ow - omega_ptr[idx_int];
                }
                // Top wall
                {
                    const int idx_ghost = kz * cell_plane_stride + (Ny + Ng + g) * cell_stride + i;
                    const int idx_int   = kz * cell_plane_stride + (Ny + Ng - 1) * cell_stride + i;
                    double y1 = wall_ptr[idx_int];
                    y1 = (y1 > 1e-10) ? y1 : 1e-10;
                    double ow = 60.0 * nu_bc / (beta1_bc * y1 * y1);
                    ow = (ow < omega_max_bc) ? ow : omega_max_bc;
                    omega_ptr[idx_ghost] = 2.0 * ow - omega_ptr[idx_int];
                }
            }
        }

        // Transport PDE done — wall BCs applied on GPU, no CPU roundtrip.
        return;
    }
#else
    (void)device_view;
#endif
    
    // CPU implementation using unified kernel (same code path as GPU)
    const int cell_stride = mesh.total_Nx();
    const int cell_plane_stride = mesh.total_Nx() * mesh.total_Ny();
    const int u_stride = mesh.total_Nx() + 1;
    const int v_stride = mesh.total_Nx();
    const int w_stride = mesh.total_Nx();
    const int u_plane_stride = u_stride * mesh.total_Ny();
    const int v_plane_stride = v_stride * (mesh.total_Ny() + 1);
    const int w_plane_stride = w_stride * mesh.total_Ny();
    const bool is_3d = (mesh.Nz > 1);

    const double dx = mesh.dx;
    const double dy = mesh.dy;
    const double dz = mesh.dz;
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double dz2 = dz * dz;
    const double inv_2dx = 1.0 / (2.0 * dx);
    const double inv_2dy = 1.0 / (2.0 * dy);
    const double inv_2dz = (dz > 0.0) ? 1.0 / (2.0 * dz) : 0.0;

    // Get raw pointers from fields
    const double* u_ptr = velocity.u_data().data();
    const double* v_ptr = velocity.v_data().data();
    const double* w_ptr = velocity.w_data().data();
    double* k_ptr = k.data().data();
    double* omega_ptr = omega.data().data();
    const double* nu_t_ptr = nu_t_prev.data().data();

    // Create wall_distance buffer for unified kernel
    const size_t total_cells = (size_t)mesh.total_Nx() * mesh.total_Ny() * mesh.total_Nz();
    std::vector<double> wall_dist_buf(total_cells, 0.0);
    for (int kz = 0; kz < mesh.total_Nz(); ++kz) {
        for (int j = 0; j < mesh.total_Ny(); ++j) {
            for (int i = 0; i < mesh.total_Nx(); ++i) {
                const int idx = kz * cell_plane_stride + j * cell_stride + i;
                wall_dist_buf[idx] = mesh.wall_distance(i, j);
            }
        }
    }
    const double* wall_dist_ptr = wall_dist_buf.data();

    // Model constants
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

    // Single pass using unified kernel (3D-capable)
    // For 2D (Nz==1): use plane 0 indexing (no z-ghost offset) to match
    // all other turbulence models and the solver's 2D kernels.
    // For 3D (Nz>1): use full k_begin()..k_end() loop with ghost offset.
    const int kz_start = is_3d ? mesh.k_begin() : 0;
    const int kz_stop  = is_3d ? mesh.k_end()   : 1;
    for (int kz = kz_start; kz < kz_stop; ++kz) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                // Cell indices (3D)
                const int idx_c  = kz * cell_plane_stride + j * cell_stride + i;
                const int idx_ip = kz * cell_plane_stride + j * cell_stride + (i + 1);
                const int idx_im = kz * cell_plane_stride + j * cell_stride + (i - 1);
                const int idx_jp = kz * cell_plane_stride + (j + 1) * cell_stride + i;
                const int idx_jm = kz * cell_plane_stride + (j - 1) * cell_stride + i;
                const int idx_kp = is_3d ? (kz + 1) * cell_plane_stride + j * cell_stride + i : idx_c;
                const int idx_km = is_3d ? (kz - 1) * cell_plane_stride + j * cell_stride + i : idx_c;

                // Velocity face indices for gradients (u at x-faces)
                const int u_idx_ip = kz * u_plane_stride + j * u_stride + (i + 1);
                const int u_idx_im = kz * u_plane_stride + j * u_stride + (i - 1);
                const int u_idx_jp = kz * u_plane_stride + (j + 1) * u_stride + i;
                const int u_idx_jm = kz * u_plane_stride + (j - 1) * u_stride + i;
                const int v_idx_ip = kz * v_plane_stride + j * v_stride + (i + 1);
                const int v_idx_im = kz * v_plane_stride + j * v_stride + (i - 1);
                const int v_idx_jp = kz * v_plane_stride + (j + 1) * v_stride + i;
                const int v_idx_jm = kz * v_plane_stride + (j - 1) * v_stride + i;

                // Velocity face indices for cell-center interpolation
                const int u_idx_c  = kz * u_plane_stride + j * u_stride + i;
                const int u_idx_c1 = kz * u_plane_stride + j * u_stride + (i + 1);
                const int v_idx_c  = kz * v_plane_stride + j * v_stride + i;
                const int v_idx_c1 = kz * v_plane_stride + (j + 1) * v_stride + i;
                const int w_idx_c  = kz * w_plane_stride + j * w_stride + i;
                const int w_idx_c1 = is_3d ? (kz + 1) * w_plane_stride + j * w_stride + i : 0;

                // Call unified kernel (same code path as GPU)
                double k_new, omega_new;
                sst_transport_cell_kernel(
                    idx_c, idx_ip, idx_im, idx_jp, idx_jm,
                    idx_kp, idx_km,
                    u_idx_ip, u_idx_im, u_idx_jp, u_idx_jm,
                    v_idx_ip, v_idx_im, v_idx_jp, v_idx_jm,
                    u_idx_c, u_idx_c1, v_idx_c, v_idx_c1,
                    w_idx_c, w_idx_c1,
                    u_ptr, v_ptr, w_ptr, k_ptr, omega_ptr, nu_t_ptr, wall_dist_ptr,
                    dx, dy, dz, dt, inv_2dx, inv_2dy, inv_2dz, dx2, dy2, dz2,
                    is_3d,
                    nu, beta_star, beta1, beta2, alpha1, alpha2,
                    sigma_k1, sigma_k2, sigma_omega1, sigma_omega2,
                    k_min, k_max, omega_min, omega_max, CD_min,
                    k_new, omega_new);

                k_ptr[idx_c] = k_new;
                omega_ptr[idx_c] = omega_new;
            }
        }
    }

    // Apply wall boundary conditions
    apply_wall_bc_k(mesh, k);
    apply_wall_bc_omega(mesh, omega, k);
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
    ensure_initialized(mesh);
    
#ifdef USE_GPU_OFFLOAD
    // GPU path using device_view (Phase 2 for closure, Phase 4 for full transport)
    if (device_view && device_view->is_valid()) {
        // Check if we have an SST-type closure or default SST closure
        bool use_sst_closure = (closure_ && 
                                (closure_->name() == "SST" || 
                                 dynamic_cast<SSTClosure*>(closure_.get()) != nullptr));
        
        if (use_sst_closure || !closure_) {
            // Direct SST closure on GPU
            const int Nx = mesh.Nx;
            const int Ny = mesh.Ny;
            const int Nz = mesh.Nz;
            const int Ng = mesh.Nghost;
            const int stride = mesh.total_Nx();
            const int cell_plane_stride_closure = mesh.total_Nx() * mesh.total_Ny();
            const size_t total_size = (size_t)mesh.total_Nx() * mesh.total_Ny() * mesh.total_Nz();

            // CRITICAL: Compute gradients first (SST closure needs them)
            gpu_kernels::compute_gradients_from_mac_gpu(
                device_view->u_face,
                device_view->v_face,
                device_view->w_face,
                device_view->dudx,
                device_view->dudy,
                device_view->dvdx,
                device_view->dvdy,
                device_view->dudz,
                device_view->dvdz,
                device_view->dwdx,
                device_view->dwdy,
                device_view->dwdz,
                Nx, Ny, device_view->Nz, Ng,
                device_view->dx, device_view->dy, device_view->dz,
                device_view->u_stride,
                device_view->v_stride,
                stride,
                device_view->u_plane_stride,
                device_view->v_plane_stride,
                device_view->w_stride,
                device_view->w_plane_stride,
                device_view->cell_plane_stride,
                device_view->u_total,
                device_view->v_total,
                device_view->w_total,
                device_view->cell_total,
                device_view->dyc,
                device_view->dyc_size
            );

            // Now compute SST closure using gradients
            gpu_kernels::compute_sst_closure_gpu(
                device_view->k,              // Already on device
                device_view->omega,          // Already on device
                device_view->dudx,           // Gradients just computed
                device_view->dudy,
                device_view->dvdx,
                device_view->dvdy,
                device_view->wall_distance,  // Wall distance on device (full field with ghosts)
                device_view->nu_t,           // Output on device
                Nx, Ny, Nz, Ng, stride,
                cell_plane_stride_closure,
                total_size, total_size,      // Last arg: wall_dist_size = total_size (not interior!)
                nu_,                         // Laminar viscosity
                constants_.a1,               // SST constant (0.31)
                constants_.beta_star,        // SST constant (0.09)
                constants_.k_min, constants_.omega_min,
                1000.0                       // nu_t_max multiplier
            );
            
            // No CPU sync needed - nu_t stays on device for solver
            return;
        }
    }
#else
    (void)device_view;
#endif
    
    // CPU path or custom closure
    if (closure_) {
        closure_->compute_nu_t(mesh, velocity, k, omega, nu_t, tau_ij);
    } else {
        // Fallback: simple k/omega on CPU
        using namespace numerics;
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double k_loc = std::max(constants_.k_min, k(i, j));
                double omega_loc = std::max(constants_.omega_min, omega(i, j));
                nu_t(i, j) = bounded_ratio(k_loc, omega_loc, OMEGA_FLOOR, NU_T_RATIO_MAX * nu_);
            }
        }
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
        const int Nz = mesh.Nz;
        const int Ng = mesh.Nghost;
        const int cell_stride = mesh.total_Nx();
        const int cell_plane = mesh.total_Nx() * mesh.total_Ny();
        const size_t cell_total_size = (size_t)cell_plane * mesh.total_Nz();
        const size_t u_total_size = (size_t)device_view->u_total;
        const size_t v_total_size = (size_t)device_view->v_total;

        gpu_kernels::komega_transport_step_gpu(
            device_view->u_face, device_view->v_face,   // Velocity on GPU
            device_view->k, device_view->omega,          // k, omega on GPU (in/out)
            device_view->nu_t,                           // nu_t_prev on GPU (uses ghost+stride)
            Nx, Ny, Nz, Ng,
            cell_stride,
            cell_plane,
            device_view->u_stride, device_view->v_stride,
            device_view->u_plane_stride, device_view->v_plane_stride,
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
            
            // Point-implicit: treat destruction implicitly for stability
            double source_k = P_k + diff_k - adv_k;
            double sink_k = constants_.beta_star * omega_old;
            double source_omega = constants_.alpha * (omega_old / k_old) * P_k
                                + diff_omega - adv_omega;
            double sink_omega = constants_.beta * omega_old;

            double k_new = (k_old + dt * source_k) / (1.0 + dt * sink_k);
            double omega_new = (omega_old + dt * source_omega) / (1.0 + dt * sink_omega);
            
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
        const int Nz = mesh.Nz;
        const int Ng = mesh.Nghost;
        const int stride = mesh.total_Nx();
        const int plane_stride = mesh.total_Nx() * mesh.total_Ny();
        const size_t total_size = (size_t)plane_stride * mesh.total_Nz();

        gpu_kernels::compute_boussinesq_closure_gpu(
            device_view->k,           // Already on device
            device_view->omega,       // Already on device
            device_view->nu_t,        // Output on device
            Nx, Ny, Nz, Ng, stride, plane_stride, total_size,
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
        using namespace numerics;
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double k_loc = std::max(constants_.k_min, k(i, j));
                double omega_loc = std::max(constants_.omega_min, omega(i, j));
                nu_t(i, j) = bounded_ratio(k_loc, omega_loc, OMEGA_FLOOR, NU_T_RATIO_MAX * nu_);
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

