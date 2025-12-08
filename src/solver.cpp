#include "solver.hpp"
#include "timing.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

namespace nncfd {

// ============================================================================
// Unified CPU/GPU kernels - single source of truth for numerical algorithms
// These kernels are compiled for both host and device when USE_GPU_OFFLOAD is on
// ============================================================================

#ifdef USE_GPU_OFFLOAD
#pragma omp declare target
#endif

// Boundary condition kernel for x-direction
inline void apply_velocity_bc_x_cell(
    int j, int g,
    int Nx, int Ng, int stride,
    bool x_lo_periodic, bool x_lo_noslip,
    bool x_hi_periodic, bool x_hi_noslip,
    double* u_ptr, double* v_ptr)
{
    // Left boundary
    int i_ghost_left     = g;
    int i_interior_left  = Ng;
    int i_periodic_left  = Nx + Ng - 1 - g;

    int idx_ghost_left    = j * stride + i_ghost_left;
    int idx_interior_left = j * stride + i_interior_left;
    int idx_periodic_left = j * stride + i_periodic_left;

    if (x_lo_periodic) {
        u_ptr[idx_ghost_left] = u_ptr[idx_periodic_left];
        v_ptr[idx_ghost_left] = v_ptr[idx_periodic_left];
    } else if (x_lo_noslip) {
        u_ptr[idx_ghost_left] = -u_ptr[idx_interior_left];
        v_ptr[idx_ghost_left] = -v_ptr[idx_interior_left];
    }

    // Right boundary
    int i_ghost_right     = Nx + Ng + g;
    int i_interior_right  = Nx + Ng - 1;
    int i_periodic_right  = Ng + g;

    int idx_ghost_right    = j * stride + i_ghost_right;
    int idx_interior_right = j * stride + i_interior_right;
    int idx_periodic_right = j * stride + i_periodic_right;

    if (x_hi_periodic) {
        u_ptr[idx_ghost_right] = u_ptr[idx_periodic_right];
        v_ptr[idx_ghost_right] = v_ptr[idx_periodic_right];
    } else if (x_hi_noslip) {
        u_ptr[idx_ghost_right] = -u_ptr[idx_interior_right];
        v_ptr[idx_ghost_right] = -v_ptr[idx_interior_right];
    }
}

// Boundary condition kernel for y-direction
inline void apply_velocity_bc_y_cell(
    int i, int g,
    int Ny, int Ng, int stride,
    bool y_lo_periodic, bool y_lo_noslip,
    bool y_hi_periodic, bool y_hi_noslip,
    double* u_ptr, double* v_ptr)
{
    // Bottom wall
    int j_ghost_bot     = g;
    int j_interior_bot  = Ng;
    int j_periodic_bot  = Ny + Ng - 1 - g;

    int idx_ghost_bot    = j_ghost_bot * stride + i;
    int idx_interior_bot = j_interior_bot * stride + i;
    int idx_periodic_bot = j_periodic_bot * stride + i;

    if (y_lo_noslip) {
        u_ptr[idx_ghost_bot] = -u_ptr[idx_interior_bot];
        v_ptr[idx_ghost_bot] = -v_ptr[idx_interior_bot];
    } else if (y_lo_periodic) {
        u_ptr[idx_ghost_bot] = u_ptr[idx_periodic_bot];
        v_ptr[idx_ghost_bot] = v_ptr[idx_periodic_bot];
    }

    // Top wall
    int j_ghost_top     = Ny + Ng + g;
    int j_interior_top  = Ny + Ng - 1;
    int j_periodic_top  = Ng + g;

    int idx_ghost_top    = j_ghost_top * stride + i;
    int idx_interior_top = j_interior_top * stride + i;
    int idx_periodic_top = j_periodic_top * stride + i;

    if (y_hi_noslip) {
        u_ptr[idx_ghost_top] = -u_ptr[idx_interior_top];
        v_ptr[idx_ghost_top] = -v_ptr[idx_interior_top];
    } else if (y_hi_periodic) {
        u_ptr[idx_ghost_top] = u_ptr[idx_periodic_top];
        v_ptr[idx_ghost_top] = v_ptr[idx_periodic_top];
    }
}

// Convective term kernel for a single cell
inline void convective_cell_kernel(
    int cell_idx, int stride, double dx, double dy, bool use_central,
    const double* u_ptr, const double* v_ptr,
    double* conv_u_ptr, double* conv_v_ptr)
{
    double uu = u_ptr[cell_idx];
    double vv = v_ptr[cell_idx];

    double dudx, dudy, dvdx, dvdy;

    if (use_central) {
        // Central differences
        dudx = (u_ptr[cell_idx+1] - u_ptr[cell_idx-1]) / (2.0 * dx);
        dudy = (u_ptr[cell_idx+stride] - u_ptr[cell_idx-stride]) / (2.0 * dy);
        dvdx = (v_ptr[cell_idx+1] - v_ptr[cell_idx-1]) / (2.0 * dx);
        dvdy = (v_ptr[cell_idx+stride] - v_ptr[cell_idx-stride]) / (2.0 * dy);
    } else {
        // First-order upwind
        if (uu >= 0) {
            dudx = (u_ptr[cell_idx] - u_ptr[cell_idx-1]) / dx;
            dvdx = (v_ptr[cell_idx] - v_ptr[cell_idx-1]) / dx;
        } else {
            dudx = (u_ptr[cell_idx+1] - u_ptr[cell_idx]) / dx;
            dvdx = (v_ptr[cell_idx+1] - v_ptr[cell_idx]) / dx;
        }

        if (vv >= 0) {
            dudy = (u_ptr[cell_idx] - u_ptr[cell_idx-stride]) / dy;
            dvdy = (v_ptr[cell_idx] - v_ptr[cell_idx-stride]) / dy;
        } else {
            dudy = (u_ptr[cell_idx+stride] - u_ptr[cell_idx]) / dy;
            dvdy = (v_ptr[cell_idx+stride] - v_ptr[cell_idx]) / dy;
        }
    }

    conv_u_ptr[cell_idx] = uu * dudx + vv * dudy;
    conv_v_ptr[cell_idx] = uu * dvdx + vv * dvdy;
}

// Diffusive term kernel for a single cell
inline void diffusive_cell_kernel(
    int cell_idx, int stride, double dx2, double dy2,
    const double* u_ptr, const double* v_ptr, const double* nu_ptr,
    double* diff_u_ptr, double* diff_v_ptr)
{
    // Face-averaged effective viscosity
    double nu_e = 0.5 * (nu_ptr[cell_idx] + nu_ptr[cell_idx+1]);
    double nu_w = 0.5 * (nu_ptr[cell_idx] + nu_ptr[cell_idx-1]);
    double nu_n = 0.5 * (nu_ptr[cell_idx] + nu_ptr[cell_idx+stride]);
    double nu_s = 0.5 * (nu_ptr[cell_idx] + nu_ptr[cell_idx-stride]);

    // Diffusive flux for u
    double diff_u_x = (nu_e * (u_ptr[cell_idx+1] - u_ptr[cell_idx]) 
                    - nu_w * (u_ptr[cell_idx] - u_ptr[cell_idx-1])) / dx2;
    double diff_u_y = (nu_n * (u_ptr[cell_idx+stride] - u_ptr[cell_idx]) 
                    - nu_s * (u_ptr[cell_idx] - u_ptr[cell_idx-stride])) / dy2;

    // Diffusive flux for v
    double diff_v_x = (nu_e * (v_ptr[cell_idx+1] - v_ptr[cell_idx]) 
                    - nu_w * (v_ptr[cell_idx] - v_ptr[cell_idx-1])) / dx2;
    double diff_v_y = (nu_n * (v_ptr[cell_idx+stride] - v_ptr[cell_idx]) 
                    - nu_s * (v_ptr[cell_idx] - v_ptr[cell_idx-stride])) / dy2;

    diff_u_ptr[cell_idx] = diff_u_x + diff_u_y;
    diff_v_ptr[cell_idx] = diff_v_x + diff_v_y;
}

// Divergence kernel for a single cell
inline void divergence_cell_kernel(
    int cell_idx, int stride, double dx, double dy,
    const double* u_ptr, const double* v_ptr,
    double* div_ptr)
{
    double dudx = (u_ptr[cell_idx + 1] - u_ptr[cell_idx - 1]) / (2.0 * dx);
    double dvdy = (v_ptr[cell_idx + stride] - v_ptr[cell_idx - stride]) / (2.0 * dy);
    div_ptr[cell_idx] = dudx + dvdy;
}

// Velocity correction kernel for a single cell
inline void correct_velocity_cell_kernel(
    int cell_idx, int stride, double dx, double dy, double dt,
    const double* u_star_ptr, const double* v_star_ptr,
    const double* p_corr_ptr,
    double* u_ptr, double* v_ptr, double* p_ptr)
{
    double dp_dx = (p_corr_ptr[cell_idx + 1] - p_corr_ptr[cell_idx - 1]) / (2.0 * dx);
    double dp_dy = (p_corr_ptr[cell_idx + stride] - p_corr_ptr[cell_idx - stride]) / (2.0 * dy);

    u_ptr[cell_idx] = u_star_ptr[cell_idx] - dt * dp_dx;
    v_ptr[cell_idx] = v_star_ptr[cell_idx] - dt * dp_dy;
    p_ptr[cell_idx] += p_corr_ptr[cell_idx];
}

// Poisson boundary condition kernel for x-direction
inline void apply_poisson_bc_x_cell(
    int j, int g,
    int Nx, int Ng, int stride,
    int bc_x_lo, int bc_x_hi,  // 0=Dirichlet, 1=Neumann, 2=Periodic
    double dirichlet_val,
    double* p_ptr)
{
    // Left boundary
    int i_ghost = g;
    int i_interior = Ng;
    int i_periodic = Nx + Ng - 1 - g;
    
    int idx_ghost = j * stride + i_ghost;
    int idx_interior = j * stride + i_interior;
    int idx_periodic = j * stride + i_periodic;
    
    if (bc_x_lo == 2) {  // Periodic
        p_ptr[idx_ghost] = p_ptr[idx_periodic];
    } else if (bc_x_lo == 1) {  // Neumann
        p_ptr[idx_ghost] = p_ptr[idx_interior];
    } else {  // Dirichlet
        p_ptr[idx_ghost] = 2.0 * dirichlet_val - p_ptr[idx_interior];
    }
    
    // Right boundary
    i_ghost = Nx + Ng + g;
    i_interior = Nx + Ng - 1;
    i_periodic = Ng + g;
    
    idx_ghost = j * stride + i_ghost;
    idx_interior = j * stride + i_interior;
    idx_periodic = j * stride + i_periodic;
    
    if (bc_x_hi == 2) {  // Periodic
        p_ptr[idx_ghost] = p_ptr[idx_periodic];
    } else if (bc_x_hi == 1) {  // Neumann
        p_ptr[idx_ghost] = p_ptr[idx_interior];
    } else {  // Dirichlet
        p_ptr[idx_ghost] = 2.0 * dirichlet_val - p_ptr[idx_interior];
    }
}

// Poisson boundary condition kernel for y-direction
inline void apply_poisson_bc_y_cell(
    int i, int g,
    int Ny, int Ng, int stride,
    int bc_y_lo, int bc_y_hi,  // 0=Dirichlet, 1=Neumann, 2=Periodic
    double dirichlet_val,
    double* p_ptr)
{
    // Bottom boundary
    int j_ghost = g;
    int j_interior = Ng;
    int j_periodic = Ny + Ng - 1 - g;
    
    int idx_ghost = j_ghost * stride + i;
    int idx_interior = j_interior * stride + i;
    int idx_periodic = j_periodic * stride + i;
    
    if (bc_y_lo == 2) {  // Periodic
        p_ptr[idx_ghost] = p_ptr[idx_periodic];
    } else if (bc_y_lo == 1) {  // Neumann
        p_ptr[idx_ghost] = p_ptr[idx_interior];
    } else {  // Dirichlet
        p_ptr[idx_ghost] = 2.0 * dirichlet_val - p_ptr[idx_interior];
    }
    
    // Top boundary
    j_ghost = Ny + Ng + g;
    j_interior = Ny + Ng - 1;
    j_periodic = Ng + g;
    
    idx_ghost = j_ghost * stride + i;
    idx_interior = j_interior * stride + i;
    idx_periodic = j_periodic * stride + i;
    
    if (bc_y_hi == 2) {  // Periodic
        p_ptr[idx_ghost] = p_ptr[idx_periodic];
    } else if (bc_y_hi == 1) {  // Neumann
        p_ptr[idx_ghost] = p_ptr[idx_interior];
    } else {  // Dirichlet
        p_ptr[idx_ghost] = 2.0 * dirichlet_val - p_ptr[idx_interior];
    }
}

// Red-black SOR Poisson iteration kernel for a single cell
inline void poisson_sor_cell_kernel(
    int cell_idx, int stride,
    double dx2, double dy2, double omega,
    const double* rhs_ptr,
    double* p_ptr)
{
    const double coeff = 2.0 / dx2 + 2.0 / dy2;
    
    double p_old = p_ptr[cell_idx];
    double p_gs = ((p_ptr[cell_idx+1] + p_ptr[cell_idx-1]) / dx2 +
                   (p_ptr[cell_idx+stride] + p_ptr[cell_idx-stride]) / dy2
                   - rhs_ptr[cell_idx]) / coeff;
    p_ptr[cell_idx] = (1.0 - omega) * p_old + omega * p_gs;
}

// Poisson residual kernel for a single cell
inline double poisson_residual_cell_kernel(
    int cell_idx, int stride,
    double dx2, double dy2,
    const double* rhs_ptr,
    const double* p_ptr)
{
    double laplacian = (p_ptr[cell_idx+1] - 2.0*p_ptr[cell_idx] + p_ptr[cell_idx-1]) / dx2
                     + (p_ptr[cell_idx+stride] - 2.0*p_ptr[cell_idx] + p_ptr[cell_idx-stride]) / dy2;
    double res = laplacian - rhs_ptr[cell_idx];
    return (res < 0.0) ? -res : res;  // abs
}

#ifdef USE_GPU_OFFLOAD
#pragma omp end declare target
#endif

RANSSolver::RANSSolver(const Mesh& mesh, const Config& config)
    : mesh_(&mesh)
    , config_(config)
    , velocity_(mesh)
    , velocity_star_(mesh)
    , pressure_(mesh)
    , pressure_correction_(mesh)
    , nu_t_(mesh)
    , k_(mesh)
    , omega_(mesh)
    , tau_ij_(mesh)
    , rhs_poisson_(mesh)
    , div_velocity_(mesh)
    , nu_eff_(mesh, config.nu)   // Persistent effective viscosity field
    , conv_(mesh)                 // Persistent convective work field
    , diff_(mesh)                 // Persistent diffusive work field
    , poisson_solver_(mesh)
    , mg_poisson_solver_(mesh)
    , use_multigrid_(true)
    , current_dt_(config.dt)
{
    // Set up Poisson solver BCs (periodic in x, Neumann in y for channel)
    poisson_solver_.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                           PoissonBC::Neumann, PoissonBC::Neumann);
    mg_poisson_solver_.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                              PoissonBC::Neumann, PoissonBC::Neumann);

#ifdef USE_GPU_OFFLOAD
    initialize_gpu_buffers();
    // GPU buffers are now mapped and will persist for solver lifetime
    // All kernels (solver and turbulence models) will use is_device_ptr
#endif
}

RANSSolver::~RANSSolver() {
#ifdef USE_GPU_OFFLOAD
    cleanup_gpu_buffers();
#endif
}

void RANSSolver::set_turbulence_model(std::unique_ptr<TurbulenceModel> model) {
    turb_model_ = std::move(model);
    if (turb_model_) {
        turb_model_->set_nu(config_.nu);
    }
}

void RANSSolver::set_velocity_bc(const VelocityBC& bc) {
    velocity_bc_ = bc;
    
    // Update Poisson BCs based on velocity BCs
    PoissonBC p_x_lo = (bc.x_lo == VelocityBC::Periodic) ? PoissonBC::Periodic : PoissonBC::Neumann;
    PoissonBC p_x_hi = (bc.x_hi == VelocityBC::Periodic) ? PoissonBC::Periodic : PoissonBC::Neumann;
    PoissonBC p_y_lo = PoissonBC::Neumann;  // Always Neumann for pressure at walls
    PoissonBC p_y_hi = PoissonBC::Neumann;
    
    // Store for GPU Poisson solver
    poisson_bc_x_lo_ = p_x_lo;
    poisson_bc_x_hi_ = p_x_hi;
    poisson_bc_y_lo_ = p_y_lo;
    poisson_bc_y_hi_ = p_y_hi;
    
    poisson_solver_.set_bc(p_x_lo, p_x_hi, p_y_lo, p_y_hi);
    mg_poisson_solver_.set_bc(p_x_lo, p_x_hi, p_y_lo, p_y_hi);
}

void RANSSolver::set_body_force(double fx, double fy) {
    fx_ = fx;
    fy_ = fy;
}

void RANSSolver::initialize(const VectorField& initial_velocity) {
    velocity_ = initial_velocity;
    apply_velocity_bc();
    
    if (turb_model_) {
        turb_model_->initialize(*mesh_, velocity_);
    }
    
#ifdef USE_GPU_OFFLOAD
    // Data already on GPU from initialize_gpu_buffers() called in constructor
    // which uses map(to:) to copy initial values
#endif
}

void RANSSolver::initialize_uniform(double u0, double v0) {
    velocity_.fill(u0, v0);
    apply_velocity_bc();
    
    // Initialize k, omega for transport models
    if (turb_model_ && turb_model_->uses_transport_equations()) {
        // Estimate initial turbulence from velocity
        double u_ref = std::max(std::abs(u0), 0.01);
        double Ti = 0.05;  // 5% turbulence intensity
        double k_init = 1.5 * (u_ref * Ti) * (u_ref * Ti);
        double omega_init = k_init / (0.09 * config_.nu * 100.0);  // ν_t/ν ≈ 100 initially
        
        k_.fill(k_init);
        omega_.fill(omega_init);
        
        // Set wall values for omega (higher near walls)
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            // Bottom wall
            int j_bot = mesh_->j_begin();
            double y_bot = mesh_->wall_distance(i, j_bot);
            omega_(i, j_bot) = 10.0 * 6.0 * config_.nu / (0.075 * y_bot * y_bot);
            
            // Top wall
            int j_top = mesh_->j_end() - 1;
            double y_top = mesh_->wall_distance(i, j_top);
            omega_(i, j_top) = 10.0 * 6.0 * config_.nu / (0.075 * y_top * y_top);
        }
    }
    
    if (turb_model_) {
        turb_model_->initialize(*mesh_, velocity_);
    }
    
#ifdef USE_GPU_OFFLOAD
    // Data already on GPU from initialize_gpu_buffers() called in constructor
    // which uses map(to:) to copy initial values
#endif
}

void RANSSolver::apply_velocity_bc() {
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Ng = mesh_->Nghost;

    const int total_Nx = mesh_->total_Nx();
    const int total_Ny = mesh_->total_Ny();
    const int stride   = total_Nx;

    const bool x_lo_periodic = (velocity_bc_.x_lo == VelocityBC::Periodic);
    const bool x_lo_noslip   = (velocity_bc_.x_lo == VelocityBC::NoSlip);
    const bool x_hi_periodic = (velocity_bc_.x_hi == VelocityBC::Periodic);
    const bool x_hi_noslip   = (velocity_bc_.x_hi == VelocityBC::NoSlip);

    const bool y_lo_periodic = (velocity_bc_.y_lo == VelocityBC::Periodic);
    const bool y_lo_noslip   = (velocity_bc_.y_lo == VelocityBC::NoSlip);
    const bool y_hi_periodic = (velocity_bc_.y_hi == VelocityBC::Periodic);
    const bool y_hi_noslip   = (velocity_bc_.y_hi == VelocityBC::NoSlip);

#ifdef USE_GPU_OFFLOAD
    if (gpu_ready_ && Nx >= 32 && Ny >= 32) {
        double* u_ptr = velocity_u_ptr_;
        double* v_ptr = velocity_v_ptr_;
        const size_t total_size = field_total_size_;

        // x-direction BCs - use map(present:) for already-mapped data
        const int n_x_bc = total_Ny * Ng;
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:total_size], v_ptr[0:total_size])
        for (int idx = 0; idx < n_x_bc; ++idx) {
            int j = idx / Ng;
            int g = idx % Ng;
            apply_velocity_bc_x_cell(j, g, Nx, Ng, stride,
                                     x_lo_periodic, x_lo_noslip,
                                     x_hi_periodic, x_hi_noslip,
                                     u_ptr, v_ptr);
        }

        // y-direction BCs - use map(present:) for already-mapped data
        const int n_y_bc = total_Nx * Ng;
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:total_size], v_ptr[0:total_size])
        for (int idx = 0; idx < n_y_bc; ++idx) {
            int i = idx / Ng;
            int g = idx % Ng;
            apply_velocity_bc_y_cell(i, g, Ny, Ng, stride,
                                     y_lo_periodic, y_lo_noslip,
                                     y_hi_periodic, y_hi_noslip,
                                     u_ptr, v_ptr);
        }
        return;
    }
#endif

    // CPU path: same kernels, but on host pointers
    double* u_ptr = velocity_.u_field().data().data();
    double* v_ptr = velocity_.v_field().data().data();

    // x-direction (periodic or inflow/outflow)
    for (int j = 0; j < total_Ny; ++j) {
        for (int g = 0; g < Ng; ++g) {
            apply_velocity_bc_x_cell(j, g, Nx, Ng, stride,
                                     x_lo_periodic, x_lo_noslip,
                                     x_hi_periodic, x_hi_noslip,
                                     u_ptr, v_ptr);
        }
    }

    // y-direction (typically no-slip walls for channel)
    for (int i = 0; i < total_Nx; ++i) {
        for (int g = 0; g < Ng; ++g) {
            apply_velocity_bc_y_cell(i, g, Ny, Ng, stride,
                                     y_lo_periodic, y_lo_noslip,
                                     y_hi_periodic, y_hi_noslip,
                                     u_ptr, v_ptr);
        }
    }
}

void RANSSolver::compute_convective_term(const VectorField& vel, VectorField& conv) {
    const double dx = mesh_->dx;
    const double dy = mesh_->dy;
    [[maybe_unused]] const int Nx = mesh_->Nx;
    [[maybe_unused]] const int Ny = mesh_->Ny;
    const int stride = mesh_->total_Nx();
    const bool use_central = (config_.convective_scheme == ConvectiveScheme::Central);

#ifdef USE_GPU_OFFLOAD
    // GPU path: same kernel, different parallelization + data source
    if (gpu_ready_ && Nx >= 32 && Ny >= 32) {
        const int n_cells = Nx * Ny;
        const size_t total_size = field_total_size_;

        const double* u_ptr      = velocity_u_ptr_;
        const double* v_ptr      = velocity_v_ptr_;
        double*       conv_u_ptr = conv_u_ptr_;
        double*       conv_v_ptr = conv_v_ptr_;

        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:total_size], v_ptr[0:total_size],       \
                        conv_u_ptr[0:total_size], conv_v_ptr[0:total_size]) \
            firstprivate(dx, dy, stride, use_central, Nx)
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + 1;  // interior i
            int j = idx / Nx + 1;  // interior j
            int cell_idx = j * stride + i;

            convective_cell_kernel(cell_idx, stride, dx, dy, use_central,
                                   u_ptr, v_ptr, conv_u_ptr, conv_v_ptr);
        }
        return;
    }
#endif

    // CPU path: same kernel, different data source and parallelization
    const double* u_ptr      = vel.u_field().data().data();
    const double* v_ptr      = vel.v_field().data().data();
    double*       conv_u_ptr = conv.u_field().data().data();
    double*       conv_v_ptr = conv.v_field().data().data();

    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            int cell_idx = j * stride + i;
            convective_cell_kernel(cell_idx, stride, dx, dy, use_central,
                                   u_ptr, v_ptr, conv_u_ptr, conv_v_ptr);
        }
    }
}

void RANSSolver::compute_diffusive_term(const VectorField& vel, const ScalarField& nu_eff, 
                                        VectorField& diff) {
    const double dx = mesh_->dx;
    const double dy = mesh_->dy;
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    [[maybe_unused]] const int Nx = mesh_->Nx;
    [[maybe_unused]] const int Ny = mesh_->Ny;
    const int stride = mesh_->total_Nx();

#ifdef USE_GPU_OFFLOAD
    // GPU path: same kernel, different parallelization + data source
    if (gpu_ready_ && Nx >= 32 && Ny >= 32) {
        const int n_cells = Nx * Ny;
        const size_t total_size = field_total_size_;

        const double* u_ptr      = velocity_u_ptr_;
        const double* v_ptr      = velocity_v_ptr_;
        const double* nu_ptr     = nu_eff_ptr_;
        double*       diff_u_ptr = diff_u_ptr_;
        double*       diff_v_ptr = diff_v_ptr_;

        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:total_size], v_ptr[0:total_size], nu_ptr[0:total_size], \
                        diff_u_ptr[0:total_size], diff_v_ptr[0:total_size]) \
            firstprivate(dx2, dy2, stride, Nx)
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + 1;
            int j = idx / Nx + 1;
            int cell_idx = j * stride + i;

            diffusive_cell_kernel(cell_idx, stride, dx2, dy2,
                                  u_ptr, v_ptr, nu_ptr,
                                  diff_u_ptr, diff_v_ptr);
        }
        return;
    }
#endif

    // CPU path: same kernel, different data source and parallelization
    const double* u_ptr      = vel.u_field().data().data();
    const double* v_ptr      = vel.v_field().data().data();
    const double* nu_ptr     = nu_eff.data().data();
    double*       diff_u_ptr = diff.u_field().data().data();
    double*       diff_v_ptr = diff.v_field().data().data();

    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            int cell_idx = j * stride + i;
            diffusive_cell_kernel(cell_idx, stride, dx2, dy2,
                                  u_ptr, v_ptr, nu_ptr,
                                  diff_u_ptr, diff_v_ptr);
        }
    }
}

void RANSSolver::compute_divergence(const VectorField& vel, ScalarField& div) {
    const double dx = mesh_->dx;
    const double dy = mesh_->dy;
    [[maybe_unused]] const int Nx = mesh_->Nx;
    [[maybe_unused]] const int Ny = mesh_->Ny;
    const int stride = mesh_->total_Nx();

#ifdef USE_GPU_OFFLOAD
    // GPU path: same kernel, different parallelization + data source
    if (gpu_ready_ && Nx >= 32 && Ny >= 32) {
        const int n_cells = Nx * Ny;
        const size_t total_size = field_total_size_;

        // Use velocity_star pointers (this is called with velocity_star_)
        const double* u_ptr  = velocity_star_u_ptr_;
        const double* v_ptr  = velocity_star_v_ptr_;
        double*       div_ptr = div_velocity_ptr_;

        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:total_size], v_ptr[0:total_size], div_ptr[0:total_size]) \
            firstprivate(dx, dy, stride, Nx)
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + 1;
            int j = idx / Nx + 1;
            int cell_idx = j * stride + i;

            divergence_cell_kernel(cell_idx, stride, dx, dy,
                                   u_ptr, v_ptr, div_ptr);
        }
    } else
#endif
    {
        // CPU path: same kernel, different data source and parallelization
        const double* u_ptr  = vel.u_field().data().data();
        const double* v_ptr  = vel.v_field().data().data();
        double*       div_ptr = div.data().data();

        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                int cell_idx = j * stride + i;
                divergence_cell_kernel(cell_idx, stride, dx, dy,
                                       u_ptr, v_ptr, div_ptr);
            }
        }
    }
}

void RANSSolver::compute_pressure_gradient(ScalarField& dp_dx, ScalarField& dp_dy) {
    double dx = mesh_->dx;
    double dy = mesh_->dy;
    
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            dp_dx(i, j) = (pressure_(i+1, j) - pressure_(i-1, j)) / (2.0 * dx);
            dp_dy(i, j) = (pressure_(i, j+1) - pressure_(i, j-1)) / (2.0 * dy);
        }
    }
}

void RANSSolver::correct_velocity() {
    const double dx = mesh_->dx;
    const double dy = mesh_->dy;
    const double dt = current_dt_;
    [[maybe_unused]] const int Nx = mesh_->Nx;
    [[maybe_unused]] const int Ny = mesh_->Ny;
    const int stride = mesh_->total_Nx();

#ifdef USE_GPU_OFFLOAD
    // GPU path: same kernel, different parallelization + data source
    if (gpu_ready_ && Nx >= 32 && Ny >= 32) {
        const int n_cells = Nx * Ny;
        const size_t total_size = field_total_size_;

        const double* u_star_ptr = velocity_star_u_ptr_;
        const double* v_star_ptr = velocity_star_v_ptr_;
        const double* p_corr_ptr = pressure_corr_ptr_;
        double*       u_ptr      = velocity_u_ptr_;
        double*       v_ptr      = velocity_v_ptr_;
        double*       p_ptr      = pressure_ptr_;

        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:total_size], v_ptr[0:total_size], p_ptr[0:total_size], \
                        u_star_ptr[0:total_size], v_star_ptr[0:total_size], p_corr_ptr[0:total_size]) \
            firstprivate(dx, dy, dt, stride, Nx)
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + 1;  // +1 for ghost cells
            int j = idx / Nx + 1;
            int cell_idx = j * stride + i;

            correct_velocity_cell_kernel(cell_idx, stride, dx, dy, dt,
                                         u_star_ptr, v_star_ptr, p_corr_ptr,
                                         u_ptr, v_ptr, p_ptr);
        }
    } else {
#endif
        // CPU path: same kernel, different data source and parallelization
        const double* u_star_ptr = velocity_star_.u_field().data().data();
        const double* v_star_ptr = velocity_star_.v_field().data().data();
        const double* p_corr_ptr = pressure_correction_.data().data();
        double*       u_ptr      = velocity_.u_field().data().data();
        double*       v_ptr      = velocity_.v_field().data().data();
        double*       p_ptr      = pressure_.data().data();

        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                int cell_idx = j * stride + i;
                correct_velocity_cell_kernel(cell_idx, stride, dx, dy, dt,
                                             u_star_ptr, v_star_ptr, p_corr_ptr,
                                             u_ptr, v_ptr, p_ptr);
            }
        }
#ifdef USE_GPU_OFFLOAD
    }
#endif
}

double RANSSolver::compute_residual() {
    // Compute residual based on velocity change
    double max_res = 0.0;
    
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            double du = velocity_.u(i, j) - velocity_star_.u(i, j);
            double dv = velocity_.v(i, j) - velocity_star_.v(i, j);
            max_res = std::max(max_res, std::abs(du));
            max_res = std::max(max_res, std::abs(dv));
        }
    }
    
    return max_res;
}

double RANSSolver::step() {
    TIMED_SCOPE("solver_step");
    
    // Store old velocity for convergence check
    VectorField velocity_old(*mesh_);
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            velocity_old.u(i, j) = velocity_.u(i, j);
            velocity_old.v(i, j) = velocity_.v(i, j);
        }
    }
    
    // 1a. Advance turbulence transport equations (if model uses them)
    if (turb_model_ && turb_model_->uses_transport_equations()) {
        TIMED_SCOPE("turbulence_transport");
        turb_model_->advance_turbulence(
            *mesh_,
            velocity_,
            current_dt_,
            k_,          // Updated in-place
            omega_,      // Updated in-place
            nu_t_        // Previous step's nu_t for diffusion coefficients
        );
    }
    
    // 1b. Update turbulence model (compute nu_t and optional tau_ij)
    if (turb_model_) {
        TIMED_SCOPE("turbulence_update");
        turb_model_->update(*mesh_, velocity_, k_, omega_, nu_t_, 
                           turb_model_->provides_reynolds_stresses() ? &tau_ij_ : nullptr);
        
#ifdef USE_GPU_OFFLOAD
        // CRITICAL: Sync nu_t from CPU to GPU after turbulence model update
        // Turbulence models compute nu_t on CPU (turbulence GPU path disabled
        // when solver uses GPU to avoid conflicting device allocations)
        if (gpu_ready_ && mesh_->Nx >= 32 && mesh_->Ny >= 32) {
            #pragma omp target update to(nu_t_ptr_[0:field_total_size_])
        }
#endif
    }
    
    // Effective viscosity: nu_eff_ = nu + nu_t (use persistent field)
    nu_eff_.fill(config_.nu);
    if (turb_model_) {
#ifdef USE_GPU_OFFLOAD
        if (gpu_ready_ && mesh_->Nx >= 32 && mesh_->Ny >= 32) {
            const int Nx = mesh_->Nx;
            const int Ny = mesh_->Ny;
            const int n_cells = Nx * Ny;
            const int stride = Nx + 2;
            const size_t total_size = field_total_size_;
            const double nu = config_.nu;
            double* nu_eff_ptr = nu_eff_ptr_;
            const double* nu_t_ptr = nu_t_ptr_;
            
            #pragma omp target teams distribute parallel for \
                map(present: nu_eff_ptr[0:total_size]) \
                map(present: nu_t_ptr[0:total_size]) \
                firstprivate(nu, stride, Nx)
            for (int idx = 0; idx < n_cells; ++idx) {
                int i = idx % Nx + 1;  // +1 for ghost cells
                int j = idx / Nx + 1;
                int cell_idx = j * stride + i;
                nu_eff_ptr[cell_idx] = nu + nu_t_ptr[cell_idx];
            }
        } else
#endif
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                nu_eff_(i, j) = config_.nu + nu_t_(i, j);
            }
        }
    }

    // 2. Compute convective and diffusive terms (use persistent fields)
    {
        TIMED_SCOPE("convective_term");
        compute_convective_term(velocity_, conv_);
    }
    
    {
        TIMED_SCOPE("diffusive_term");
        compute_diffusive_term(velocity_, nu_eff_, diff_);
    }
    
    // 3. Compute provisional velocity u* (without pressure gradient)
    // u* = u^n + dt * (-conv + diff + body_force)
#ifdef USE_GPU_OFFLOAD
    if (gpu_ready_ && mesh_->Nx >= 32 && mesh_->Ny >= 32) {
        const int Nx = mesh_->Nx;
        const int Ny = mesh_->Ny;
        const int n_cells = Nx * Ny;
        const int stride = Nx + 2;
        const size_t total_size = field_total_size_;
        const double dt = current_dt_;
        const double fx = fx_;
        const double fy = fy_;
        const double* u_ptr = velocity_u_ptr_;
        const double* v_ptr = velocity_v_ptr_;
        double* u_star_ptr = velocity_star_u_ptr_;
        double* v_star_ptr = velocity_star_v_ptr_;
        const double* conv_u_ptr = conv_u_ptr_;
        const double* conv_v_ptr = conv_v_ptr_;
        const double* diff_u_ptr = diff_u_ptr_;
        const double* diff_v_ptr = diff_v_ptr_;
        
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:total_size]) \
            map(present: v_ptr[0:total_size]) \
            map(present: u_star_ptr[0:total_size]) \
            map(present: v_star_ptr[0:total_size]) \
            map(present: conv_u_ptr[0:total_size]) \
            map(present: conv_v_ptr[0:total_size]) \
            map(present: diff_u_ptr[0:total_size]) \
            map(present: diff_v_ptr[0:total_size]) \
            firstprivate(dt, fx, fy, stride, Nx)
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + 1;  // +1 for ghost cells
            int j = idx / Nx + 1;
            int cell_idx = j * stride + i;
            
            u_star_ptr[cell_idx] = u_ptr[cell_idx] + dt * (-conv_u_ptr[cell_idx] + diff_u_ptr[cell_idx] + fx);
            v_star_ptr[cell_idx] = v_ptr[cell_idx] + dt * (-conv_v_ptr[cell_idx] + diff_v_ptr[cell_idx] + fy);
        }
    } else
#endif
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            velocity_star_.u(i, j) = velocity_.u(i, j) + current_dt_ * 
                (-conv_.u(i, j) + diff_.u(i, j) + fx_);
            velocity_star_.v(i, j) = velocity_.v(i, j) + current_dt_ * 
                (-conv_.v(i, j) + diff_.v(i, j) + fy_);
        }
    }
    
    // Apply BCs to provisional velocity (needed for divergence calculation)
    // Temporarily swap velocity_ and velocity_star_ to use apply_velocity_bc
    std::swap(velocity_, velocity_star_);
    apply_velocity_bc();
    std::swap(velocity_, velocity_star_);
    
    // 4. Solve pressure Poisson equation
    // nabla^2p' = (1/dt) nabla*u*
    {
        TIMED_SCOPE("divergence");
        compute_divergence(velocity_star_, div_velocity_);
    }
    
#ifdef USE_GPU_OFFLOAD
    if (gpu_ready_ && mesh_->Nx >= 32 && mesh_->Ny >= 32) {
        const int Nx = mesh_->Nx;
        const int Ny = mesh_->Ny;
        const int n_cells = Nx * Ny;
        const int stride = Nx + 2;
        const size_t total_size = field_total_size_;
        const double dt_inv = 1.0 / current_dt_;
        const double* div_ptr = div_velocity_ptr_;
        double* rhs_ptr = rhs_poisson_ptr_;
        double* p_corr_ptr = pressure_corr_ptr_;
        
        #pragma omp target teams distribute parallel for \
            map(present: div_ptr[0:total_size]) \
            map(present: rhs_ptr[0:total_size]) \
            map(present: p_corr_ptr[0:total_size]) \
            firstprivate(dt_inv, stride, Nx)
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + 1;
            int j = idx / Nx + 1;
            int cell_idx = j * stride + i;
            
            rhs_ptr[cell_idx] = div_ptr[cell_idx] * dt_inv;
            p_corr_ptr[cell_idx] = 0.0;  // Initialize pressure correction
        }
    } else
#endif
    {
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                rhs_poisson_(i, j) = div_velocity_(i, j) / current_dt_;
            }
        }
        
        pressure_correction_.fill(0.0);
    }
    
    // 4b. Solve Poisson equation for pressure correction
    {
        TIMED_SCOPE("poisson_solve");
        PoissonConfig pcfg;
        pcfg.tol = config_.poisson_tol;
        pcfg.max_iter = config_.poisson_max_iter;
        pcfg.omega = config_.poisson_omega;
        
        // Use multigrid solver for both CPU and GPU
        // The multigrid solver internally uses GPU-accelerated kernels when USE_GPU_OFFLOAD is enabled
        // All V-cycle operations (smoothing, restriction, prolongation) run on persistent device arrays
        if (use_multigrid_) {
            // Use multigrid solver - it handles GPU offloading internally
            mg_poisson_solver_.solve(rhs_poisson_, pressure_correction_, pcfg);
        } else {
            poisson_solver_.solve(rhs_poisson_, pressure_correction_, pcfg);
        }
    }
    
    // 5. Correct velocity and pressure
    {
        TIMED_SCOPE("velocity_correction");
        correct_velocity();
    }
    
    // 6. Apply boundary conditions
    apply_velocity_bc();
    
    ++iter_;
    
    // Return max velocity change as convergence criterion
    double max_change = 0.0;
    
#ifdef USE_GPU_OFFLOAD
    if (gpu_ready_ && mesh_->Nx >= 32 && mesh_->Ny >= 32) {
        // Sync current velocity from GPU to compute residual
        #pragma omp target update from(velocity_u_ptr_[0:field_total_size_])
        #pragma omp target update from(velocity_v_ptr_[0:field_total_size_])
    }
#endif
    
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            double du = std::abs(velocity_.u(i, j) - velocity_old.u(i, j));
            double dv = std::abs(velocity_.v(i, j) - velocity_old.v(i, j));
            max_change = std::max(max_change, std::max(du, dv));
        }
    }
    return max_change;
}

std::pair<double, int> RANSSolver::solve_steady() {
    double residual = 1.0;
    
    if (config_.verbose) {
        if (config_.adaptive_dt) {
            std::cout << std::setw(8) << "Iter" 
                      << std::setw(15) << "Residual"
                      << std::setw(15) << "Max |u|"
                      << std::setw(12) << "dt"
                      << "\n";
        } else {
            std::cout << std::setw(8) << "Iter" 
                      << std::setw(15) << "Residual"
                      << std::setw(15) << "Max |u|"
                      << "\n";
        }
    }
    
    for (iter_ = 0; iter_ < config_.max_iter; ++iter_) {
        // Update time step if adaptive
        if (config_.adaptive_dt) {
            current_dt_ = compute_adaptive_dt();
        }
        
        residual = step();
        
        if (config_.verbose && (iter_ + 1) % config_.output_freq == 0) {
            double max_vel = velocity_.max_magnitude();
            if (config_.adaptive_dt) {
                std::cout << std::setw(8) << iter_ + 1
                          << std::setw(15) << std::scientific << std::setprecision(3) << residual
                          << std::setw(15) << std::fixed << max_vel
                          << std::setw(12) << std::scientific << std::setprecision(2) << current_dt_
                          << "\n";
            } else {
                std::cout << std::setw(8) << iter_ + 1
                          << std::setw(15) << std::scientific << std::setprecision(3) << residual
                          << std::setw(15) << std::fixed << max_vel
                          << "\n";
            }
        }
        
        if (residual < config_.tol) {
            if (config_.verbose) {
                std::cout << "Converged at iteration " << iter_ + 1 
                          << " with residual " << residual << "\n";
            }
            break;
        }
        
        // Check for divergence
        if (std::isnan(residual) || std::isinf(residual)) {
            if (config_.verbose) {
                std::cerr << "Solver diverged at iteration " << iter_ + 1 << "\n";
            }
            break;
        }
    }
    
    return {residual, iter_ + 1};
}

std::pair<double, int> RANSSolver::solve_steady_with_snapshots(
    const std::string& output_prefix,
    int num_snapshots,
    int snapshot_freq) 
{
    // Calculate snapshot frequency if not provided
    if (snapshot_freq < 0 && num_snapshots > 0) {
        snapshot_freq = std::max(1, config_.max_iter / num_snapshots);
    }
    
    if (config_.verbose && !output_prefix.empty()) {
        std::cout << "Will output ";
        if (num_snapshots > 0) {
            std::cout << num_snapshots << " VTK snapshots (every " 
                     << snapshot_freq << " iterations)\n";
        } else {
            std::cout << "final VTK snapshot only\n";
        }
    }
    
    double residual = 1.0;
    int snapshot_count = 0;
    
    if (config_.verbose) {
        if (config_.adaptive_dt) {
            std::cout << std::setw(8) << "Iter" 
                      << std::setw(15) << "Residual"
                      << std::setw(15) << "Max |u|"
                      << std::setw(12) << "dt"
                      << "\n";
        } else {
            std::cout << std::setw(8) << "Iter" 
                      << std::setw(15) << "Residual"
                      << std::setw(15) << "Max |u|"
                      << "\n";
        }
    }
    
    for (iter_ = 0; iter_ < config_.max_iter; ++iter_) {
        // Update time step if adaptive
        if (config_.adaptive_dt) {
            current_dt_ = compute_adaptive_dt();
        }
        
        residual = step();
        
        // Write VTK snapshots at regular intervals
        if (!output_prefix.empty() && num_snapshots > 0 && 
            snapshot_freq > 0 && (iter_ + 1) % snapshot_freq == 0) {
            snapshot_count++;
            std::string vtk_file = output_prefix + "_" + 
                                  std::to_string(snapshot_count) + ".vtk";
            try {
                write_vtk(vtk_file);
                if (config_.verbose) {
                    std::cout << "Wrote snapshot " << snapshot_count 
                             << ": " << vtk_file << "\n";
                }
            } catch (const std::exception& e) {
                std::cerr << "Warning: Could not write VTK snapshot: " 
                         << e.what() << "\n";
            }
        }
        
        // Console output
        if (config_.verbose && (iter_ + 1) % config_.output_freq == 0) {
            double max_vel = velocity_.max_magnitude();
            if (config_.adaptive_dt) {
                std::cout << std::setw(8) << iter_ + 1
                          << std::setw(15) << std::scientific << std::setprecision(3) << residual
                          << std::setw(15) << std::fixed << max_vel
                          << std::setw(12) << std::scientific << std::setprecision(2) << current_dt_
                          << "\n";
            } else {
                std::cout << std::setw(8) << iter_ + 1
                          << std::setw(15) << std::scientific << std::setprecision(3) << residual
                          << std::setw(15) << std::fixed << max_vel
                          << "\n";
            }
        }
        
        if (residual < config_.tol) {
            if (config_.verbose) {
                std::cout << "Converged at iteration " << iter_ + 1 
                          << " with residual " << residual << "\n";
            }
            break;
        }
        
        // Check for divergence
        if (std::isnan(residual) || std::isinf(residual)) {
            if (config_.verbose) {
                std::cerr << "Solver diverged at iteration " << iter_ + 1 << "\n";
            }
            break;
        }
    }
    
    // Write final snapshot if output prefix provided
    if (!output_prefix.empty()) {
        std::string final_file = output_prefix + "_final.vtk";
        try {
            write_vtk(final_file);
            if (config_.verbose) {
                std::cout << "Final VTK output: " << final_file << "\n";
                if (num_snapshots > 0) {
                    std::cout << "Total VTK snapshots: " << snapshot_count + 1 
                             << " (" << snapshot_count << " during + 1 final)\n";
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: Could not write final VTK: " 
                     << e.what() << "\n";
        }
    }
    
    return {residual, iter_ + 1};
}

double RANSSolver::bulk_velocity() const {
    // Area-averaged streamwise velocity
    double sum = 0.0;
    int count = 0;
    
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            sum += velocity_.u(i, j);
            ++count;
        }
    }
    
    return sum / count;
}

double RANSSolver::wall_shear_stress() const {
    // Compute du/dy at the bottom wall
    // Using one-sided difference from first interior cell to wall
    double sum = 0.0;
    int count = 0;
    
    int j = mesh_->j_begin();  // First interior row
    double y_cell = mesh_->y(j);
    double y_wall = mesh_->y_min;
    double dist = y_cell - y_wall;
    
    for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
        // u at wall is 0 (no-slip)
        double dudy = velocity_.u(i, j) / dist;
        sum += dudy;
        ++count;
    }
    
    double dudy_avg = sum / count;
    return config_.nu * dudy_avg;  // tau_w = mu * du/dy = rho * nu * du/dy (rho=1)
}

double RANSSolver::friction_velocity() const {
    double tau_w = wall_shear_stress();
    return std::sqrt(std::abs(tau_w));  // u_tau = sqrt(tau_w / rho)
}

double RANSSolver::Re_tau() const {
    double u_tau = friction_velocity();
    double delta = (mesh_->y_max - mesh_->y_min) / 2.0;
    return u_tau * delta / config_.nu;
}

void RANSSolver::print_velocity_profile(double x_loc) const {
    // Find i index closest to x_loc
    int i_loc = mesh_->i_begin();
    double min_dist = std::abs(mesh_->x(i_loc) - x_loc);
    
    for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
        double dist = std::abs(mesh_->x(i) - x_loc);
        if (dist < min_dist) {
            min_dist = dist;
            i_loc = i;
        }
    }
    
    std::cout << "\nVelocity profile at x = " << mesh_->x(i_loc) << ":\n";
    std::cout << std::setw(12) << "y" << std::setw(12) << "u" << std::setw(12) << "v" << "\n";
    
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        std::cout << std::setw(12) << std::fixed << std::setprecision(4) << mesh_->y(j)
                  << std::setw(12) << velocity_.u(i_loc, j)
                  << std::setw(12) << velocity_.v(i_loc, j)
                  << "\n";
    }
}

void RANSSolver::write_fields(const std::string& prefix) const {
    velocity_.write(prefix + "_velocity.dat");
    pressure_.write(prefix + "_pressure.dat");
    
    if (turb_model_) {
        nu_t_.write(prefix + "_nu_t.dat");
    }
}

double RANSSolver::compute_adaptive_dt() const {
    // CFL condition: dt <= CFL * min(dx, dy) / |u_max|
    double u_max = 1e-10;  // Small value to avoid division by zero
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            double u_mag = std::sqrt(velocity_.u(i, j) * velocity_.u(i, j) + 
                                    velocity_.v(i, j) * velocity_.v(i, j));
            u_max = std::max(u_max, u_mag);
        }
    }
    double dt_cfl = config_.CFL_max * std::min(mesh_->dx, mesh_->dy) / u_max;
    
    // Diffusion stability: dt <= factor * min(dx^2, dy^2) / nu_eff_max
    double nu_eff_max = config_.nu;
    if (turb_model_) {
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                nu_eff_max = std::max(nu_eff_max, config_.nu + nu_t_(i, j));
            }
        }
    }
    double dx_min = std::min(mesh_->dx, mesh_->dy);
    double dt_diff = 0.25 * dx_min * dx_min / nu_eff_max;
    
    // Take minimum of both constraints
    return std::min(dt_cfl, dt_diff);
}

void RANSSolver::write_vtk(const std::string& filename) const {
#ifdef USE_GPU_OFFLOAD
    // Download data from GPU for I/O
    const_cast<RANSSolver*>(this)->sync_from_gpu();
#endif
    
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error: Cannot open " << filename << " for writing\n";
        return;
    }
    
    int Nx = mesh_->Nx;
    int Ny = mesh_->Ny;
    
    // VTK header
    file << "# vtk DataFile Version 3.0\n";
    file << "RANS simulation output\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_POINTS\n";
    file << "DIMENSIONS " << Nx << " " << Ny << " 1\n";
    file << "ORIGIN " << mesh_->x_min << " " << mesh_->y_min << " 0\n";
    file << "SPACING " << mesh_->dx << " " << mesh_->dy << " 1\n";
    file << "POINT_DATA " << Nx * Ny << "\n";
    
    // Velocity vector field
    file << "VECTORS velocity double\n";
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            file << velocity_.u(i, j) << " " << velocity_.v(i, j) << " 0\n";
        }
    }
    
    // Pressure scalar field
    file << "SCALARS pressure double 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            file << pressure_(i, j) << "\n";
        }
    }
    
    // Velocity magnitude
    file << "SCALARS velocity_magnitude double 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            double mag = std::sqrt(velocity_.u(i, j) * velocity_.u(i, j) + 
                                  velocity_.v(i, j) * velocity_.v(i, j));
            file << mag << "\n";
        }
    }
    
    // Eddy viscosity (if turbulence model is active)
    if (turb_model_) {
        file << "SCALARS nu_t double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                file << nu_t_(i, j) << "\n";
            }
        }
    }
    
    file.close();
}

#ifdef USE_GPU_OFFLOAD
void RANSSolver::initialize_gpu_buffers() {
    int num_devices = omp_get_num_devices();
    if (num_devices == 0) {
        gpu_ready_ = false;
        std::cout << "No GPU devices found, disabling GPU offload\n";
        return;
    }
    
    // Get raw pointers to all field data
    field_total_size_ = (mesh_->Nx + 2) * (mesh_->Ny + 2);
    
    velocity_u_ptr_ = velocity_.u_field().data().data();
    velocity_v_ptr_ = velocity_.v_field().data().data();
    velocity_star_u_ptr_ = velocity_star_.u_field().data().data();
    velocity_star_v_ptr_ = velocity_star_.v_field().data().data();
    pressure_ptr_ = pressure_.data().data();
    pressure_corr_ptr_ = pressure_correction_.data().data();
    nu_t_ptr_ = nu_t_.data().data();
    nu_eff_ptr_ = nu_eff_.data().data();
    conv_u_ptr_ = conv_.u_field().data().data();
    conv_v_ptr_ = conv_.v_field().data().data();
    diff_u_ptr_ = diff_.u_field().data().data();
    diff_v_ptr_ = diff_.v_field().data().data();
    rhs_poisson_ptr_ = rhs_poisson_.data().data();
    div_velocity_ptr_ = div_velocity_.data().data();
    k_ptr_ = k_.data().data();
    omega_ptr_ = omega_.data().data();
    
    if (config_.verbose) {
        std::cout << "Mapping " << (16 * field_total_size_ * sizeof(double) / 1024.0 / 1024.0) 
                  << " MB to GPU for persistent solver arrays...\n";
    }
    
    // Map all arrays to GPU device and copy initial values
    // Using map(to:) instead of map(alloc:) to transfer initialized data
    // Data will persist on GPU for entire solver lifetime
    #pragma omp target enter data map(to: velocity_u_ptr_[0:field_total_size_])
    #pragma omp target enter data map(to: velocity_v_ptr_[0:field_total_size_])
    #pragma omp target enter data map(to: velocity_star_u_ptr_[0:field_total_size_])
    #pragma omp target enter data map(to: velocity_star_v_ptr_[0:field_total_size_])
    #pragma omp target enter data map(to: pressure_ptr_[0:field_total_size_])
    #pragma omp target enter data map(to: pressure_corr_ptr_[0:field_total_size_])
    #pragma omp target enter data map(to: nu_t_ptr_[0:field_total_size_])
    #pragma omp target enter data map(to: nu_eff_ptr_[0:field_total_size_])
    #pragma omp target enter data map(to: conv_u_ptr_[0:field_total_size_])
    #pragma omp target enter data map(to: conv_v_ptr_[0:field_total_size_])
    #pragma omp target enter data map(to: diff_u_ptr_[0:field_total_size_])
    #pragma omp target enter data map(to: diff_v_ptr_[0:field_total_size_])
    #pragma omp target enter data map(to: rhs_poisson_ptr_[0:field_total_size_])
    #pragma omp target enter data map(to: div_velocity_ptr_[0:field_total_size_])
    #pragma omp target enter data map(to: k_ptr_[0:field_total_size_])
    #pragma omp target enter data map(to: omega_ptr_[0:field_total_size_])
    
    gpu_ready_ = true;
    
    if (config_.verbose) {
        std::cout << "✓ GPU arrays mapped successfully (persistent, data resident on device)\n";
    }
}

void RANSSolver::cleanup_gpu_buffers() {
    if (!gpu_ready_) return;
    
    // Copy final results back from GPU, then free device memory
    // Using map(from:) to get final state back to host
    #pragma omp target exit data map(from: velocity_u_ptr_[0:field_total_size_])
    #pragma omp target exit data map(from: velocity_v_ptr_[0:field_total_size_])
    #pragma omp target exit data map(from: pressure_ptr_[0:field_total_size_])
    #pragma omp target exit data map(from: nu_t_ptr_[0:field_total_size_])
    #pragma omp target exit data map(from: k_ptr_[0:field_total_size_])
    #pragma omp target exit data map(from: omega_ptr_[0:field_total_size_])
    
    // Delete temporary/work arrays without copying back
    #pragma omp target exit data map(delete: velocity_star_u_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: velocity_star_v_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: pressure_corr_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: nu_eff_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: conv_u_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: conv_v_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: diff_u_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: diff_v_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: rhs_poisson_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: div_velocity_ptr_[0:field_total_size_])
    
    gpu_ready_ = false;
}

void RANSSolver::sync_to_gpu() {
    if (!gpu_ready_) return;
    
    // Update GPU with changed fields (typically after CPU-side modifications)
    // In normal operation, this is rarely needed since all computation is on GPU
    #pragma omp target update to(velocity_u_ptr_[0:field_total_size_])
    #pragma omp target update to(velocity_v_ptr_[0:field_total_size_])
    #pragma omp target update to(pressure_ptr_[0:field_total_size_])
    #pragma omp target update to(nu_t_ptr_[0:field_total_size_])
    #pragma omp target update to(k_ptr_[0:field_total_size_])
    #pragma omp target update to(omega_ptr_[0:field_total_size_])
}

void RANSSolver::sync_from_gpu() {
    if (!gpu_ready_) return;
    
    // Download only solution fields needed for I/O (data stays on GPU)
    // This is called before write_vtk, write_fields, or when accessing fields from CPU
    #pragma omp target update from(velocity_u_ptr_[0:field_total_size_])
    #pragma omp target update from(velocity_v_ptr_[0:field_total_size_])
    #pragma omp target update from(pressure_ptr_[0:field_total_size_])
    #pragma omp target update from(nu_t_ptr_[0:field_total_size_])
    #pragma omp target update from(k_ptr_[0:field_total_size_])
    #pragma omp target update from(omega_ptr_[0:field_total_size_])
}
#endif

} // namespace nncfd

