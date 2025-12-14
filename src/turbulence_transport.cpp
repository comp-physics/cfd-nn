#include "turbulence_transport.hpp"
#include "gpu_kernels.hpp"
#include "timing.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
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
    compute_gradients_from_mac_cpu(mesh, velocity, dudx_, dudy_, dvdx_, dvdy_);
    
    const double a1 = constants_.a1;
    
    // IMPORTANT: SSTClosure GPU path disabled - causes pointer aliasing with RANSSolver's GPU buffers
    // The caller (RANSSolver or SSTKOmegaTransport) will handle GPU sync if needed
#ifdef USE_GPU_OFFLOAD_DISABLED_FOR_CLOSURE
    if (false && omp_get_num_devices() > 0) {
        const int Nx = mesh.Nx;
        const int Ny = mesh.Ny;
        const int n_cells = Nx * Ny;
        const int stride = mesh.total_Nx();
        const size_t total_size = (size_t)mesh.total_Nx() * mesh.total_Ny();
        
        // Get raw pointers
        const double* dudx_ptr = dudx_.data().data();
        const double* dudy_ptr = dudy_.data().data();
        const double* dvdx_ptr = dvdx_.data().data();
        const double* dvdy_ptr = dvdy_.data().data();
        const double* k_ptr = k.data().data();        // Already on GPU from solver
        const double* omega_ptr = omega.data().data(); // Already on GPU from solver
        double* nu_t_ptr = nu_t.data().data();         // Already on GPU from solver
        
        const double nu = nu_;
        const double beta_star = constants_.beta_star;
        const double k_min = constants_.k_min;
        const double omega_min = constants_.omega_min;
        
        // Precompute wall distances (flatten to match interior cells loop order)
        std::vector<double> wall_dist(n_cells);
        int flat_idx = 0;
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                wall_dist[flat_idx++] = mesh.wall_distance(i, j);
            }
        }
        const double* wall_dist_ptr = wall_dist.data();
        
        // CRITICAL FIX: Use stride-based indexing for arrays with ghost cells
        // Arrays k, omega, nu_t have size (Nx+2)*(Ny+2), not Nx*Ny
        // Don't use map() clause - let runtime handle mapping automatically
        #pragma omp target teams distribute parallel for \
            map(to: dudx_ptr[0:total_size], dudy_ptr[0:total_size], \
                    dvdx_ptr[0:total_size], dvdy_ptr[0:total_size], \
                    wall_dist_ptr[0:n_cells])
        for (int idx = 0; idx < n_cells; ++idx) {
            // Convert flat index to (i,j) including ghost cells
            const int i = idx % Nx + 1;  // +1 to skip ghost cells
            const int j = idx / Nx + 1;
            const int cell_idx = j * stride + i;  // Stride-based index for arrays with ghosts
            
            double k_loc = k_ptr[cell_idx];
            double omega_loc = omega_ptr[cell_idx];
            double y_wall = wall_dist_ptr[idx];  // Wall dist is flat (no ghosts)
            
            k_loc = (k_loc > k_min) ? k_loc : k_min;
            omega_loc = (omega_loc > omega_min) ? omega_loc : omega_min;
            double y_safe = (y_wall > 1e-10) ? y_wall : 1e-10;
            
            // Strain rate magnitude from gradient fields (also have ghosts, use cell_idx)
            double Sxx = dudx_ptr[cell_idx];
            double Syy = dvdy_ptr[cell_idx];
            double Sxy = 0.5 * (dudy_ptr[cell_idx] + dvdx_ptr[cell_idx]);
            double S_mag = sqrt(2.0 * (Sxx*Sxx + Syy*Syy + 2.0*Sxy*Sxy));
            
            // F2 blending function
            double sqrt_k = sqrt(k_loc);
            double term1 = 2.0 * sqrt_k / (beta_star * omega_loc * y_safe);
            double term2 = 500.0 * nu / (y_safe * y_safe * omega_loc);
            double arg2 = (term1 > term2) ? term1 : term2;
            double F2 = tanh(arg2 * arg2);
            
            // SST eddy viscosity: ν_t = a₁k / max(a₁ω, SF₂)
            double denom = a1 * omega_loc;
            double SF2 = S_mag * F2;
            denom = (denom > SF2) ? denom : SF2;
            
            double nu_t_loc = a1 * k_loc / denom;
            
            // Clipping
            nu_t_loc = (nu_t_loc > 0.0) ? nu_t_loc : 0.0;
            double max_nu_t = 1000.0 * nu;
            nu_t_loc = (nu_t_loc < max_nu_t) ? nu_t_loc : max_nu_t;
            
            nu_t_ptr[cell_idx] = nu_t_loc;  // Write to correct cell with ghosts
        }
        
        return;
    }
#endif
    
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
    // Check if GPU is available
    #include <omp.h>
    if (omp_get_num_devices() == 0) {
        buffers_on_gpu_ = false;
        return;
    }
    
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
        
        // Map buffers to GPU - use single pragma with multiple arrays (like NN models)
        #pragma omp target enter data \
            map(alloc: k_ptr[0:k_size]) \
            map(alloc: omega_ptr[0:omega_size]) \
            map(alloc: nu_t_ptr[0:nu_t_size]) \
            map(alloc: u_ptr[0:u_size]) \
            map(alloc: v_ptr[0:v_size]) \
            map(alloc: wall_ptr[0:wall_size]) \
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
    // No-op for CPU-only builds
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
    compute_gradients_from_mac_cpu(mesh, velocity, dudx_, dudy_, dvdx_, dvdy_);
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
        std::cout << "[SSTKOmegaTransport] Using GPU path for advance_turbulence!" << std::endl;
        // Use the existing fused SST transport GPU implementation,
        // but operate directly on solver-owned device-resident data
        const int Nx = mesh.Nx;
        const int Ny = mesh.Ny;
        const int Ng = mesh.Nghost;
        const int n_cells = Nx * Ny;
        const int cell_stride = mesh.total_Nx();
        const size_t cell_total_size = (size_t)mesh.total_Nx() * mesh.total_Ny();
        const size_t u_total_size = (size_t)mesh.total_Ny() * (mesh.total_Nx() + 1);
        const size_t v_total_size = (size_t)(mesh.total_Ny() + 1) * mesh.total_Nx();
        
        const double dx = mesh.dx;
        const double dy = mesh.dy;
        const double dx2 = dx * dx;
        const double dy2 = dy * dy;
        const double inv_2dx = 1.0 / (2.0 * dx);
        const double inv_2dy = 1.0 / (2.0 * dy);
        
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
        double* k_ptr = device_view->k;
        double* omega_ptr = device_view->omega;
        const double* nu_t_ptr = device_view->nu_t;
        const double* wall_dist_ptr = device_view->wall_distance;
        const int u_stride = device_view->u_stride;
        const int v_stride = device_view->v_stride;
        
        // Fused SST transport kernel (all on GPU, no staging)
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], \
                         k_ptr[0:cell_total_size], omega_ptr[0:cell_total_size], \
                         nu_t_ptr[0:cell_total_size], wall_dist_ptr[0:cell_total_size])
        for (int cell_idx = 0; cell_idx < n_cells; ++cell_idx) {
            int i = cell_idx % Nx;
            int j = cell_idx / Nx;
            int ii = i + Ng;
            int jj = j + Ng;
            
            int idx_c = jj * cell_stride + ii;
            int idx_ip = jj * cell_stride + (ii + 1);
            int idx_im = jj * cell_stride + (ii - 1);
            int idx_jp = (jj + 1) * cell_stride + ii;
            int idx_jm = (jj - 1) * cell_stride + ii;
            
            // Velocity gradients from MAC grid
            const int u_idx_ip = jj * u_stride + (ii + 1);
            const int u_idx_im = jj * u_stride + (ii - 1);
            const int u_idx_jp = (jj + 1) * u_stride + ii;
            const int u_idx_jm = (jj - 1) * u_stride + ii;
            const int v_idx_ip = jj * v_stride + (ii + 1);
            const int v_idx_im = jj * v_stride + (ii - 1);
            const int v_idx_jp = (jj + 1) * v_stride + ii;
            const int v_idx_jm = (jj - 1) * v_stride + ii;
            
            double dudx_v = (u_ptr[u_idx_ip] - u_ptr[u_idx_im]) * inv_2dx;
            double dudy_v = (u_ptr[u_idx_jp] - u_ptr[u_idx_jm]) * inv_2dy;
            double dvdx_v = (v_ptr[v_idx_ip] - v_ptr[v_idx_im]) * inv_2dx;
            double dvdy_v = (v_ptr[v_idx_jp] - v_ptr[v_idx_jm]) * inv_2dy;
            
            // Get cell-centered velocity (interpolate from faces)
            double u_c = 0.5 * (u_ptr[jj * u_stride + ii] + u_ptr[jj * u_stride + (ii+1)]);
            double v_c = 0.5 * (v_ptr[jj * v_stride + ii] + v_ptr[(jj+1) * v_stride + ii]);
            
            double k_c = k_ptr[idx_c];
            double omega_c = omega_ptr[idx_c];
            double y_wall = wall_dist_ptr[idx_c];
            double nu_t_c = nu_t_ptr[idx_c];
            
            k_c = (k_c > k_min) ? k_c : k_min;
            omega_c = (omega_c > omega_min) ? omega_c : omega_min;
            double y_safe = (y_wall > 1e-10) ? y_wall : 1e-10;
            nu_t_c = (nu_t_c > 0.0) ? nu_t_c : 0.0;
            
            // Strain rate
            double Sxx = dudx_v;
            double Syy = dvdy_v;
            double Sxy = 0.5 * (dudy_v + dvdx_v);
            double S2 = 2.0 * (Sxx*Sxx + Syy*Syy + 2.0*Sxy*Sxy);
            
            // Cross-diffusion for F1
            double dkdx = (k_ptr[idx_ip] - k_ptr[idx_im]) * inv_2dx;
            double dkdy = (k_ptr[idx_jp] - k_ptr[idx_jm]) * inv_2dy;
            double domegadx = (omega_ptr[idx_ip] - omega_ptr[idx_im]) * inv_2dx;
            double domegady = (omega_ptr[idx_jp] - omega_ptr[idx_jm]) * inv_2dy;
            
            double CD_omega = 2.0 * sigma_omega2 / omega_c * (dkdx * domegadx + dkdy * domegady);
            CD_omega = (CD_omega > CD_min) ? CD_omega : CD_min;
            
            // F1 blending
            double sqrt_k = sqrt(k_c);
            double arg1_1 = sqrt_k / (beta_star * omega_c * y_safe);
            double arg1_2 = 500.0 * nu / (y_safe * y_safe * omega_c);
            double arg1_3 = 4.0 * sigma_omega2 * k_c / (CD_omega * y_safe * y_safe);
            double arg1 = arg1_1;
            arg1 = (arg1 > arg1_2) ? arg1 : arg1_2;
            arg1 = (arg1 < arg1_3) ? arg1 : arg1_3;
            double F1 = tanh(arg1 * arg1 * arg1 * arg1);
            
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
            
            // Diffusion
            double diff_k = nu_k * ((k_ptr[idx_ip] - 2.0*k_c + k_ptr[idx_im]) / dx2
                                  + (k_ptr[idx_jp] - 2.0*k_c + k_ptr[idx_jm]) / dy2);
            double diff_omega = nu_omega_eff * ((omega_ptr[idx_ip] - 2.0*omega_c + omega_ptr[idx_im]) / dx2
                                              + (omega_ptr[idx_jp] - 2.0*omega_c + omega_ptr[idx_jm]) / dy2);
            
            // Cross-diffusion term for omega equation
            double CD = 2.0 * (1.0 - F1) * sigma_omega2 / omega_c 
                      * (dkdx * domegadx + dkdy * domegady);
            CD = (CD > 0.0) ? CD : 0.0;
            
            // RHS
            double rhs_k = P_k - beta_star * k_c * omega_c + diff_k - adv_k;
            double rhs_omega = alpha * (omega_c / k_c) * P_k 
                             - beta * omega_c * omega_c
                             + diff_omega - adv_omega + CD;
            
            // Update
            double k_new = k_c + dt * rhs_k;
            double omega_new = omega_c + dt * rhs_omega;
            
            // Clip
            k_new = (k_new > k_min) ? k_new : k_min;
            k_new = (k_new < k_max) ? k_new : k_max;
            omega_new = (omega_new > omega_min) ? omega_new : omega_min;
            omega_new = (omega_new < omega_max) ? omega_new : omega_max;
            
            k_ptr[idx_c] = k_new;
            omega_ptr[idx_c] = omega_new;
        }
        
        // Transport PDE done entirely on GPU - no CPU sync needed
        return;
    }
#else
    (void)device_view;
#endif
    
    // CPU implementation
    compute_velocity_gradients(mesh, velocity);
    compute_blending_functions(mesh, k, omega);
    compute_production(mesh, nu_t_prev);
    
    // Build effective diffusivities with blending
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double F1 = F1_(i, j);
            double sigma_k = F1 * constants_.sigma_k1 + (1.0 - F1) * constants_.sigma_k2;
            double sigma_omega = F1 * constants_.sigma_omega1 + (1.0 - F1) * constants_.sigma_omega2;
            
            double nu_t_loc = std::max(0.0, nu_t_prev(i, j));
            nu_k_(i, j) = nu_ + sigma_k * nu_t_loc;
            nu_omega_(i, j) = nu_ + sigma_omega * nu_t_loc;
        }
    }
    
    // Compute advection terms (upwind)
    const double dx = mesh.dx;
    const double dy = mesh.dy;
    
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double u = velocity.u(i, j);
            double v = velocity.v(i, j);
            
            // k advection
            double dkdx = (u >= 0) ? (k(i,j) - k(i-1,j)) / dx : (k(i+1,j) - k(i,j)) / dx;
            double dkdy = (v >= 0) ? (k(i,j) - k(i,j-1)) / dy : (k(i,j+1) - k(i,j)) / dy;
            adv_k_(i, j) = u * dkdx + v * dkdy;
            
            // omega advection
            double domegadx = (u >= 0) ? (omega(i,j) - omega(i-1,j)) / dx : (omega(i+1,j) - omega(i,j)) / dx;
            double domegady = (v >= 0) ? (omega(i,j) - omega(i,j-1)) / dy : (omega(i,j+1) - omega(i,j)) / dy;
            adv_omega_(i, j) = u * domegadx + v * domegady;
        }
    }
    
    // Compute diffusion terms
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            // k diffusion
            double nu_e = 0.5 * (nu_k_(i, j) + nu_k_(i+1, j));
            double nu_w = 0.5 * (nu_k_(i, j) + nu_k_(i-1, j));
            double nu_n = 0.5 * (nu_k_(i, j) + nu_k_(i, j+1));
            double nu_s = 0.5 * (nu_k_(i, j) + nu_k_(i, j-1));
            
            diff_k_(i, j) = (nu_e * (k(i+1,j) - k(i,j)) - nu_w * (k(i,j) - k(i-1,j))) / dx2
                          + (nu_n * (k(i,j+1) - k(i,j)) - nu_s * (k(i,j) - k(i,j-1))) / dy2;
            
            // omega diffusion
            nu_e = 0.5 * (nu_omega_(i, j) + nu_omega_(i+1, j));
            nu_w = 0.5 * (nu_omega_(i, j) + nu_omega_(i-1, j));
            nu_n = 0.5 * (nu_omega_(i, j) + nu_omega_(i, j+1));
            nu_s = 0.5 * (nu_omega_(i, j) + nu_omega_(i, j-1));
            
            diff_omega_(i, j) = (nu_e * (omega(i+1,j) - omega(i,j)) - nu_w * (omega(i,j) - omega(i-1,j))) / dx2
                              + (nu_n * (omega(i,j+1) - omega(i,j)) - nu_s * (omega(i,j) - omega(i,j-1))) / dy2;
        }
    }
    
    // Time integration (explicit Euler)
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double F1 = F1_(i, j);
            double k_old = std::max(constants_.k_min, k(i, j));
            double omega_old = std::max(constants_.omega_min, omega(i, j));
            
            // Blended constants
            double beta = F1 * constants_.beta1 + (1.0 - F1) * constants_.beta2;
            double alpha = F1 * constants_.alpha1 + (1.0 - F1) * constants_.alpha2;
            double sigma_omega2 = constants_.sigma_omega2;
            
            // Limit production
            double P_k = std::min(P_k_(i, j), 10.0 * constants_.beta_star * k_old * omega_old);
            
            // k equation: ∂k/∂t = P_k - β*kω + diff - adv
            double rhs_k = P_k - constants_.beta_star * k_old * omega_old 
                         + diff_k_(i, j) - adv_k_(i, j);
            
            // ω equation: ∂ω/∂t = α(ω/k)P_k - βω² + diff - adv + CD
            // Cross-diffusion term
            double dkdx = (k(i+1, j) - k(i-1, j)) / (2.0 * dx);
            double dkdy = (k(i, j+1) - k(i, j-1)) / (2.0 * dy);
            double domegadx = (omega(i+1, j) - omega(i-1, j)) / (2.0 * dx);
            double domegady = (omega(i, j+1) - omega(i, j-1)) / (2.0 * dy);
            double CD = 2.0 * (1.0 - F1) * sigma_omega2 / omega_old 
                      * (dkdx * domegadx + dkdy * domegady);
            CD = std::max(CD, 0.0);  // Only positive cross-diffusion
            
            double rhs_omega = alpha * (omega_old / k_old) * P_k 
                             - beta * omega_old * omega_old
                             + diff_omega_(i, j) - adv_omega_(i, j) + CD;
            
            // Update
            double k_new = k_old + dt * rhs_k;
            double omega_new = omega_old + dt * rhs_omega;
            
            // Clipping
            k(i, j) = std::min(std::max(k_new, constants_.k_min), constants_.k_max);
            omega(i, j) = std::min(std::max(omega_new, constants_.omega_min), constants_.omega_max);
        }
    }
    
    // Apply wall boundary conditions
    apply_wall_bc_k(mesh, k);
    apply_wall_bc_omega(mesh, omega, k);
}

#ifdef USE_GPU_OFFLOAD
void SSTKOmegaTransport::advance_turbulence_gpu(
    const Mesh& mesh,
    const VectorField& velocity,
    double dt,
    ScalarField& k,
    ScalarField& omega,
    const ScalarField& nu_t_prev)
{
    TIMED_SCOPE("sst_transport_gpu");
    
    const int Nx = mesh.Nx;
    const int Ny = mesh.Ny;
    const int n_cells = Nx * Ny;
    const int n_total = (Nx + 2) * (Ny + 2);
    const int stride = Nx + 2;
    const double dx = mesh.dx;
    const double dy = mesh.dy;
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double inv_2dx = 1.0 / (2.0 * dx);
    const double inv_2dy = 1.0 / (2.0 * dy);
    
    // Copy data to flat arrays
    std::copy(velocity.u_data().begin(), velocity.u_data().end(), u_flat_.begin());
    std::copy(velocity.v_data().begin(), velocity.v_data().end(), v_flat_.begin());
    std::copy(k.data().begin(), k.data().end(), k_flat_.begin());
    std::copy(omega.data().begin(), omega.data().end(), omega_flat_.begin());
    
    // Copy nu_t_prev to flat array
    int idx = 0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            nu_t_flat_[idx++] = nu_t_prev(i, j);
        }
    }
    
    // Get pointers (buffers are already persistently mapped to GPU)
    double* u_ptr = u_flat_.data();
    double* v_ptr = v_flat_.data();
    double* k_ptr = k_flat_.data();
    double* omega_ptr = omega_flat_.data();
    const double* nu_t_ptr = nu_t_flat_.data();
    const double* wall_dist_ptr = wall_dist_flat_.data();
    double* work_ptr = work_flat_.data();
    
    // Update GPU with input data (buffers are persistent, just update contents)
    #pragma omp target update to(u_ptr[0:n_total], v_ptr[0:n_total], \
                                 k_ptr[0:n_total], omega_ptr[0:n_total], \
                                 nu_t_ptr[0:n_cells])
    
    // Model constants (copy to local for GPU capture)
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
    
    // GPU kernel - let OpenMP runtime figure out device mapping automatically
    // Data was mapped with target enter data, so runtime will find it in the mapping table
    #pragma omp target teams distribute parallel for
    for (int cell_idx = 0; cell_idx < n_cells; ++cell_idx) {
        int i = cell_idx % Nx;
        int j = cell_idx / Nx;
        int ii = i + 1;  // With ghost cells
        int jj = j + 1;
        
        int idx_c = jj * stride + ii;
        int idx_ip = jj * stride + (ii + 1);
        int idx_im = jj * stride + (ii - 1);
        int idx_jp = (jj + 1) * stride + ii;
        int idx_jm = (jj - 1) * stride + ii;
        
        // Velocity gradients
        double dudx_v = (u_ptr[idx_ip] - u_ptr[idx_im]) * inv_2dx;
        double dudy_v = (u_ptr[idx_jp] - u_ptr[idx_jm]) * inv_2dy;
        double dvdx_v = (v_ptr[idx_ip] - v_ptr[idx_im]) * inv_2dx;
        double dvdy_v = (v_ptr[idx_jp] - v_ptr[idx_jm]) * inv_2dy;
        
        double u_c = u_ptr[idx_c];
        double v_c = v_ptr[idx_c];
        double k_c = k_ptr[idx_c];
        double omega_c = omega_ptr[idx_c];
        double y_wall = wall_dist_ptr[cell_idx];
        double nu_t_c = nu_t_ptr[cell_idx];
        
        k_c = (k_c > k_min) ? k_c : k_min;
        omega_c = (omega_c > omega_min) ? omega_c : omega_min;
        double y_safe = (y_wall > 1e-10) ? y_wall : 1e-10;
        nu_t_c = (nu_t_c > 0.0) ? nu_t_c : 0.0;
        
        // Strain rate
        double Sxx = dudx_v;
        double Syy = dvdy_v;
        double Sxy = 0.5 * (dudy_v + dvdx_v);
        double S2 = 2.0 * (Sxx*Sxx + Syy*Syy + 2.0*Sxy*Sxy);
        
        // Cross-diffusion for F1
        double dkdx = (k_ptr[idx_ip] - k_ptr[idx_im]) * inv_2dx;
        double dkdy = (k_ptr[idx_jp] - k_ptr[idx_jm]) * inv_2dy;
        double domegadx = (omega_ptr[idx_ip] - omega_ptr[idx_im]) * inv_2dx;
        double domegady = (omega_ptr[idx_jp] - omega_ptr[idx_jm]) * inv_2dy;
        
        double CD_omega = 2.0 * sigma_omega2 / omega_c * (dkdx * domegadx + dkdy * domegady);
        CD_omega = (CD_omega > CD_min) ? CD_omega : CD_min;
        
        // F1 blending
        double sqrt_k = sqrt(k_c);
        double arg1_1 = sqrt_k / (beta_star * omega_c * y_safe);
        double arg1_2 = 500.0 * nu / (y_safe * y_safe * omega_c);
        double arg1_3 = 4.0 * sigma_omega2 * k_c / (CD_omega * y_safe * y_safe);
        double arg1 = arg1_1;
        arg1 = (arg1 > arg1_2) ? arg1 : arg1_2;
        arg1 = (arg1 < arg1_3) ? arg1 : arg1_3;
        double F1 = tanh(arg1 * arg1 * arg1 * arg1);
        
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
        
        // Diffusion (simple Laplacian for now - could use face-averaged viscosity)
        double diff_k = nu_k * ((k_ptr[idx_ip] - 2.0*k_c + k_ptr[idx_im]) / dx2
                              + (k_ptr[idx_jp] - 2.0*k_c + k_ptr[idx_jm]) / dy2);
        double diff_omega = nu_omega_eff * ((omega_ptr[idx_ip] - 2.0*omega_c + omega_ptr[idx_im]) / dx2
                                          + (omega_ptr[idx_jp] - 2.0*omega_c + omega_ptr[idx_jm]) / dy2);
        
        // Cross-diffusion term for omega equation
        double CD = 2.0 * (1.0 - F1) * sigma_omega2 / omega_c 
                  * (dkdx * domegadx + dkdy * domegady);
        CD = (CD > 0.0) ? CD : 0.0;
        
        // RHS
        double rhs_k = P_k - beta_star * k_c * omega_c + diff_k - adv_k;
        double rhs_omega = alpha * (omega_c / k_c) * P_k 
                         - beta * omega_c * omega_c
                         + diff_omega - adv_omega + CD;
        
        // Update
        double k_new = k_c + dt * rhs_k;
        double omega_new = omega_c + dt * rhs_omega;
        
        // Clip
        k_new = (k_new > k_min) ? k_new : k_min;
        k_new = (k_new < k_max) ? k_new : k_max;
        omega_new = (omega_new > omega_min) ? omega_new : omega_min;
        omega_new = (omega_new < omega_max) ? omega_new : omega_max;
        
        k_ptr[idx_c] = k_new;
        omega_ptr[idx_c] = omega_new;
    }
    
    // Update CPU with results from GPU
    #pragma omp target update from(k_ptr[0:n_total], omega_ptr[0:n_total])
    
    // Copy back to fields
    std::copy(k_flat_.begin(), k_flat_.end(), k.data().begin());
    std::copy(omega_flat_.begin(), omega_flat_.end(), omega.data().begin());
    
    // Apply wall BCs on CPU (simpler for now)
    apply_wall_bc_k(mesh, k);
    apply_wall_bc_omega(mesh, omega, k);
}
#endif

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
            const int Ng = mesh.Nghost;
            const int stride = mesh.total_Nx();
            const size_t total_size = (size_t)mesh.total_Nx() * mesh.total_Ny();
            
            // CRITICAL: Compute gradients first (SST closure needs them)
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
                stride,
                velocity.u_total_size(),
                velocity.v_total_size(),
                total_size
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
                Nx, Ny, Ng, stride, total_size, total_size,  // Last arg: wall_dist_size = total_size (not interior!)
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
        std::cout << "[KOmegaTransport] Using GPU path for advance_turbulence!" << std::endl;
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
    } else {
        std::cout << "[KOmegaTransport] GPU path NOT taken. device_view=" << device_view 
                  << " valid=" << (device_view ? device_view->is_valid() : false) << std::endl;
    }
#else
    (void)device_view;
#endif
    
    // CPU fallback path
    std::cout << "[KOmegaTransport] Using CPU fallback for advance_turbulence" << std::endl;
    compute_gradients_from_mac_cpu(mesh, velocity, dudx_, dudy_, dvdx_, dvdy_);
    
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

