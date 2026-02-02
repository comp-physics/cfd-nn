/// @file solver.cpp
/// @brief Implementation of incompressible RANS solver with fractional-step projection method
///
/// This file implements the RANSSolver class, which solves the Reynolds-Averaged Navier-Stokes
/// equations using a fractional-step projection method. Key features:
/// - Fractional-step time integration (explicit Euler + pressure projection)
/// - Multigrid Poisson solver for pressure correction
/// - Staggered MAC grid discretization (2nd-order central differences)
/// - GPU acceleration via OpenMP target offload
/// - Support for multiple turbulence models (algebraic, transport, EARSM, neural networks)
///
/// The implementation includes unified CPU/GPU kernels that compile for both host and device,
/// ensuring numerical consistency between platforms.

#include "solver.hpp"
#include "timing.hpp"
#include "gpu_utils.hpp"
#include "profiling.hpp"
#include "mpi_check.hpp"
#include "numerics.hpp"
#include "stencil_operators.hpp"
#include "solver_kernels.hpp"
#include "solver_time_kernels.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <cassert>
#include <cstring>

#ifdef GPU_PROFILE_TRANSFERS
#include <chrono>
#endif

// Legacy NVTX macros for backward compatibility
// These are kept for existing code; new code should use profiling.hpp macros
#ifdef GPU_PROFILE_KERNELS
#if __has_include(<nvtx3/nvToolsExt.h>)
    #include <nvtx3/nvToolsExt.h>
    #define NVTX_PUSH(name) nvtxRangePushA(name)
    #define NVTX_POP() nvtxRangePop()
#elif __has_include(<nvToolsExt.h>)
    #include <nvToolsExt.h>
    #define NVTX_PUSH(name) nvtxRangePushA(name)
    #define NVTX_POP() nvtxRangePop()
#else
    #define NVTX_PUSH(name)
    #define NVTX_POP()
#endif
#else
#define NVTX_PUSH(name)
#define NVTX_POP()
#endif

namespace nncfd {


// Import GPU kernels from dedicated header
using namespace nncfd::kernels;

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
    , velocity_old_(mesh)         // GPU-resident old velocity for residual
    , velocity_rk_(mesh)          // RK work buffer for multi-stage methods
    , dudx_(mesh), dudy_(mesh), dvdx_(mesh), dvdy_(mesh)  // Gradient scratch for turbulence
    , wall_distance_(mesh)        // Precomputed wall distance field
    , poisson_solver_(mesh)
    , mg_poisson_solver_(mesh)
    , use_multigrid_(true)
    , current_dt_(config.dt)
{
    // Check for MPI environment - hard fail for GPU builds, warn for CPU
    // (this code uses GPU parallelism, not MPI distribution)
    // Set NNCFD_ALLOW_MULTI_RANK=1 to override (dangerous)
    enforce_single_rank_gpu("RANSSolver");

    // Validate ghost cell requirements for advection schemes
    // Upwind2 requires Nghost >= 2 due to 5-point stencil (i±2)
    if (config_.convective_scheme == ConvectiveScheme::Upwind2 && mesh.Nghost < 2) {
        std::cerr << "[Solver] WARNING: upwind2 scheme requires Nghost >= 2 but Nghost = "
                  << mesh.Nghost << "\n"
                  << "         Falling back to 1st-order upwind scheme.\n";
        config_.convective_scheme = ConvectiveScheme::Upwind;
    }

    // O4 spatial discretization requires Nghost >= 2 for 5-point stencils
    if (config_.space_order == 4 && mesh.Nghost < 2) {
        std::cerr << "[Solver] WARNING: space_order=4 requires Nghost >= 2 but Nghost = "
                  << mesh.Nghost << "\n"
                  << "         Falling back to space_order=2.\n";
        config_.space_order = 2;
    }

    // Precompute wall distance (once, then stays on GPU if enabled)
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            wall_distance_(i, j) = mesh.wall_distance(i, j);
        }
    }
    // Set up Poisson solver BCs (periodic in x, Neumann in y for channel)
    poisson_solver_.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                           PoissonBC::Neumann, PoissonBC::Neumann);
    mg_poisson_solver_.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                              PoissonBC::Neumann, PoissonBC::Neumann);

#ifdef USE_HYPRE
    // Initialize HYPRE PFMG solver
    hypre_poisson_solver_ = std::make_unique<HyprePoissonSolver>(mesh);
    hypre_poisson_solver_->set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                                   PoissonBC::Neumann, PoissonBC::Neumann);
#endif

#ifdef USE_FFT_POISSON
    // Initialize FFT solvers for periodic cases
    // FFT (2D): requires periodic x AND z with uniform spacing - 3D only
    // FFT2D: for 2D meshes with periodic x, walls y
    // FFT1D: requires periodic x OR z (exactly one) with uniform spacing - 3D only
    bool fft_applicable = false;
    bool fft2d_applicable = false;
    bool fft1d_applicable = false;

    // Check which FFT solver is applicable (actual BCs set later via set_velocity_bc)
    // For now, assume defaults: periodic x,z - will be updated in set_velocity_bc
    bool periodic_xz = true;  // Default for channel: periodic x/z
    bool uniform_xz = true;   // Default for channel: uniform x/z spacing

    if (mesh.is2D()) {
        // 2D mesh: try FFT2D solver (periodic x, non-periodic y)
        try {
            fft2d_poisson_solver_ = std::make_unique<FFT2DPoissonSolver>(mesh);
            fft2d_poisson_solver_->set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                                          PoissonBC::Neumann, PoissonBC::Neumann);
            fft2d_applicable = true;
            std::cout << "[Solver] FFT2D solver initialized for 2D mesh\n";
        } catch (const std::exception& e) {
            std::cerr << "[Solver] FFT2D solver initialization failed: " << e.what() << "\n";
            fft2d_applicable = false;
        }
    } else {
        // 3D mesh: try 2D FFT first (periodic x AND z)
        if (periodic_xz && uniform_xz) {
            try {
                fft_poisson_solver_ = std::make_unique<FFTPoissonSolver>(mesh);
                fft_poisson_solver_->set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                                             PoissonBC::Neumann, PoissonBC::Neumann,
                                             PoissonBC::Periodic, PoissonBC::Periodic);
                // Set space order for O4-consistent eigenvalues when using O4 projection
                fft_poisson_solver_->set_space_order(config_.space_order);
                fft_applicable = true;
            } catch (const std::exception& e) {
                std::cerr << "[Solver] FFT solver initialization failed: " << e.what() << "\n";
            }
        }

        // Also initialize 1D FFT solver (for cases like duct flow)
        // Will be used if 2D FFT becomes incompatible after BC update
        try {
            // Default to x-periodic (duct flow typical case)
            fft1d_poisson_solver_ = std::make_unique<FFT1DPoissonSolver>(mesh, 0);
            fft1d_poisson_solver_->set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                                           PoissonBC::Neumann, PoissonBC::Neumann,
                                           PoissonBC::Neumann, PoissonBC::Neumann);
            fft1d_applicable = true;
        } catch (const std::exception& e) {
            std::cerr << "[Solver] FFT1D solver initialization failed: " << e.what() << "\n";
        }
    }
#endif

    // ========================================================================
    // Poisson Solver Auto-Selection
    // Priority: FFT (3D) → FFT2D (2D mesh) → FFT1D (3D 1-periodic) → HYPRE → MG
    // ========================================================================
    PoissonSolverType requested = config.poisson_solver;

    if (requested == PoissonSolverType::Auto) {
        // Auto-select: FFT > FFT2D > FFT1D > HYPRE > MG
#ifdef USE_FFT_POISSON
        if (fft_applicable) {
            selected_solver_ = PoissonSolverType::FFT;
            selection_reason_ ="auto: periodic(x,z) + uniform(dx,dz) + 3D";
        } else if (fft2d_applicable) {
            selected_solver_ = PoissonSolverType::FFT2D;
            selection_reason_ ="auto: 2D mesh + periodic(x) + uniform(dx)";
        } else if (fft1d_applicable) {
            selected_solver_ = PoissonSolverType::FFT1D;
            selection_reason_ ="auto: periodic(x) + uniform(dx) + 3D (1D FFT)";
        } else
#endif
#ifdef USE_HYPRE
        if (hypre_poisson_solver_) {
            selected_solver_ = PoissonSolverType::HYPRE;
            selection_reason_ ="auto: FFT not applicable, HYPRE available";
        } else
#endif
        {
            selected_solver_ = PoissonSolverType::MG;
            selection_reason_ ="auto: fallback to multigrid";
        }
    } else if (requested == PoissonSolverType::FFT) {
#ifdef USE_FFT_POISSON
        if (fft_applicable) {
            selected_solver_ = PoissonSolverType::FFT;
            selection_reason_ ="explicit: user requested FFT";
        } else {
            std::cerr << "[Solver] Warning: FFT requested but not applicable "
                      << "(requires 3D, periodic x/z, uniform dx/dz). Falling back to ";
            if (fft1d_applicable) {
                selected_solver_ = PoissonSolverType::FFT1D;
                std::cerr << "FFT1D.\n";
                selection_reason_ ="fallback from FFT: using FFT1D";
            } else
#ifdef USE_HYPRE
            if (hypre_poisson_solver_) {
                selected_solver_ = PoissonSolverType::HYPRE;
                std::cerr << "HYPRE.\n";
                selection_reason_ ="fallback from FFT: not applicable";
            } else
#endif
            {
                selected_solver_ = PoissonSolverType::MG;
                std::cerr << "MG.\n";
                selection_reason_ ="fallback from FFT: not applicable";
            }
        }
#else
        std::cerr << "[Solver] Warning: FFT requested but USE_FFT_POISSON not built. ";
#ifdef USE_HYPRE
        selected_solver_ = PoissonSolverType::HYPRE;
        std::cerr << "Using HYPRE.\n";
#else
        selected_solver_ = PoissonSolverType::MG;
        std::cerr << "Using MG.\n";
#endif
        selection_reason_ ="fallback from FFT: not built";
#endif
    } else if (requested == PoissonSolverType::FFT1D) {
#ifdef USE_FFT_POISSON
        if (fft1d_applicable) {
            selected_solver_ = PoissonSolverType::FFT1D;
            selection_reason_ ="explicit: user requested FFT1D";
        } else {
            std::cerr << "[Solver] Warning: FFT1D requested but not applicable. ";
            selected_solver_ = PoissonSolverType::MG;
            std::cerr << "Using MG.\n";
            selection_reason_ ="fallback from FFT1D: not applicable";
        }
#else
        std::cerr << "[Solver] Warning: FFT1D requested but USE_FFT_POISSON not built. ";
        selected_solver_ = PoissonSolverType::MG;
        std::cerr << "Using MG.\n";
        selection_reason_ ="fallback from FFT1D: not built";
#endif
    } else if (requested == PoissonSolverType::HYPRE) {
#ifdef USE_HYPRE
        if (hypre_poisson_solver_) {
            selected_solver_ = PoissonSolverType::HYPRE;
            selection_reason_ ="explicit: user requested HYPRE";
        } else {
            std::cerr << "[Solver] Warning: HYPRE initialization failed. Using MG.\n";
            selected_solver_ = PoissonSolverType::MG;
            selection_reason_ ="fallback from HYPRE: init failed";
        }
#else
        std::cerr << "[Solver] Warning: HYPRE requested but USE_HYPRE not built. Using MG.\n";
        selected_solver_ = PoissonSolverType::MG;
        selection_reason_ ="fallback from HYPRE: not built";
#endif
    } else {
        // PoissonSolverType::MG
        selected_solver_ = PoissonSolverType::MG;
        selection_reason_ ="explicit: user requested MG";
    }

    // Log the selection
    const char* solver_name = (selected_solver_ == PoissonSolverType::FFT) ? "FFT" :
                              (selected_solver_ == PoissonSolverType::FFT2D) ? "FFT2D" :
                              (selected_solver_ == PoissonSolverType::FFT1D) ? "FFT1D" :
                              (selected_solver_ == PoissonSolverType::HYPRE) ? "HYPRE" : "MG";
    std::cout << "[Poisson] selected=" << solver_name
              << " reason=" << selection_reason_
              << " dims=" << mesh.Nx << "x" << mesh.Ny << "x" << mesh.Nz << "\n";

    // Safety check: O4 spatial order requires Nghost >= 2, but MG is currently ng=1 only
    // Note: use config_ (member) not config (parameter) since space_order may have been downgraded
    if (config_.space_order == 4 && selected_solver_ == PoissonSolverType::MG) {
        std::cerr << "[Solver] ERROR: space_order=4 requires Nghost >= 2, but MG backend is ng=1 only.\n";
#ifdef USE_HYPRE
        if (hypre_poisson_solver_) {
            std::cerr << "         Falling back to HYPRE.\n";
            selected_solver_ = PoissonSolverType::HYPRE;
            selection_reason_ = "fallback from MG: O4 requires ng>=2";
        } else {
            std::cerr << "         HYPRE not available. Results may be incorrect!\n";
        }
#else
        std::cerr << "         HYPRE not built. Consider using FFT (periodic BCs) or rebuilding with HYPRE.\n";
        std::cerr << "         Results may be incorrect!\n";
#endif
    }

#ifdef USE_GPU_OFFLOAD
    // Fail-fast if GPU offload is enabled but no device is available
    gpu::verify_device_available();
#endif

    // Initialize raw pointers for unified code paths
    // In GPU builds, this also maps data to device; in CPU builds, just sets pointers
    initialize_gpu_buffers();
}

RANSSolver::~RANSSolver() {
    cleanup_gpu_buffers();  // Safe to call unconditionally (no-op when GPU disabled)
}

void RANSSolver::set_turbulence_model(std::unique_ptr<TurbulenceModel> model) {
    turb_model_ = std::move(model);
    if (turb_model_) {
        turb_model_->set_nu(config_.nu);
        
        // Initialize turbulence model GPU buffers if GPU is available and mesh is initialized
        if (mesh_) {
            turb_model_->initialize_gpu_buffers(*mesh_);
        }
    }
}

void RANSSolver::set_velocity_bc(const VelocityBC& bc) {
    // Validate: periodic BCs must be symmetric (both ends must match)
    if ((bc.x_lo == VelocityBC::Periodic) != (bc.x_hi == VelocityBC::Periodic)) {
        throw std::invalid_argument("Periodic BC in x requires both x_lo and x_hi to be Periodic");
    }
    if ((bc.y_lo == VelocityBC::Periodic) != (bc.y_hi == VelocityBC::Periodic)) {
        throw std::invalid_argument("Periodic BC in y requires both y_lo and y_hi to be Periodic");
    }
    if ((bc.z_lo == VelocityBC::Periodic) != (bc.z_hi == VelocityBC::Periodic)) {
        throw std::invalid_argument("Periodic BC in z requires both z_lo and z_hi to be Periodic");
    }

    velocity_bc_ = bc;

    // Update Poisson BCs based on velocity BCs
    PoissonBC p_x_lo = (bc.x_lo == VelocityBC::Periodic) ? PoissonBC::Periodic : PoissonBC::Neumann;
    PoissonBC p_x_hi = (bc.x_hi == VelocityBC::Periodic) ? PoissonBC::Periodic : PoissonBC::Neumann;
    PoissonBC p_y_lo = (bc.y_lo == VelocityBC::Periodic) ? PoissonBC::Periodic : PoissonBC::Neumann;
    PoissonBC p_y_hi = (bc.y_hi == VelocityBC::Periodic) ? PoissonBC::Periodic : PoissonBC::Neumann;
    PoissonBC p_z_lo = (bc.z_lo == VelocityBC::Periodic) ? PoissonBC::Periodic : PoissonBC::Neumann;
    PoissonBC p_z_hi = (bc.z_hi == VelocityBC::Periodic) ? PoissonBC::Periodic : PoissonBC::Neumann;

    // Store for GPU Poisson solver
    poisson_bc_x_lo_ = p_x_lo;
    poisson_bc_x_hi_ = p_x_hi;
    poisson_bc_y_lo_ = p_y_lo;
    poisson_bc_y_hi_ = p_y_hi;
    poisson_bc_z_lo_ = p_z_lo;
    poisson_bc_z_hi_ = p_z_hi;

    // Set BCs on Poisson solvers - use 3D overload for 3D meshes
    if (!mesh_->is2D()) {
        poisson_solver_.set_bc(p_x_lo, p_x_hi, p_y_lo, p_y_hi, p_z_lo, p_z_hi);
        mg_poisson_solver_.set_bc(p_x_lo, p_x_hi, p_y_lo, p_y_hi, p_z_lo, p_z_hi);
#ifdef USE_HYPRE
        if (hypre_poisson_solver_) {
            hypre_poisson_solver_->set_bc(p_x_lo, p_x_hi, p_y_lo, p_y_hi, p_z_lo, p_z_hi);
        }
#endif
    } else {
        poisson_solver_.set_bc(p_x_lo, p_x_hi, p_y_lo, p_y_hi);
        mg_poisson_solver_.set_bc(p_x_lo, p_x_hi, p_y_lo, p_y_hi);
#ifdef USE_HYPRE
        if (hypre_poisson_solver_) {
            hypre_poisson_solver_->set_bc(p_x_lo, p_x_hi, p_y_lo, p_y_hi);
        }
#endif
    }

    // Re-check FFT/FFT1D applicability after BC update
#ifdef USE_FFT_POISSON
    bool periodic_x = (p_x_lo == PoissonBC::Periodic && p_x_hi == PoissonBC::Periodic);
    bool periodic_z = (p_z_lo == PoissonBC::Periodic && p_z_hi == PoissonBC::Periodic);

    // 2D FFT requires periodic in BOTH x and z
    bool fft_compatible = periodic_x && periodic_z && !mesh_->is2D();

    // 1D FFT requires periodic in EXACTLY ONE of x or z
    bool fft1d_compatible = (periodic_x != periodic_z) && !mesh_->is2D();

    if (fft_poisson_solver_) {
        if (fft_compatible) {
            // Update FFT solver BCs
            fft_poisson_solver_->set_bc(p_x_lo, p_x_hi, p_y_lo, p_y_hi, p_z_lo, p_z_hi);
        } else if (selected_solver_ == PoissonSolverType::FFT) {
            // FFT was selected but BCs are now incompatible
            if (fft1d_compatible && fft1d_poisson_solver_) {
                std::cerr << "[Poisson] Warning: FFT solver incompatible with BCs "
                          << "(requires periodic x AND z). Switching to FFT1D.\n";
                selected_solver_ = PoissonSolverType::FFT1D;
            } else {
                std::cerr << "[Poisson] Warning: FFT solver incompatible with BCs "
                          << "(requires periodic x AND z). Switching to MG.\n";
                selected_solver_ = PoissonSolverType::MG;
            }
        }
    }

    if (fft1d_poisson_solver_) {
        if (fft1d_compatible) {
            // Update FFT1D solver BCs
            fft1d_poisson_solver_->set_bc(p_x_lo, p_x_hi, p_y_lo, p_y_hi, p_z_lo, p_z_hi);
        } else if (selected_solver_ == PoissonSolverType::FFT1D) {
            // FFT1D was selected but BCs are now incompatible - switch to MG
            std::cerr << "[Poisson] Warning: FFT1D solver incompatible with BCs "
                      << "(requires periodic in exactly one of x or z). Switching to MG.\n";
            selected_solver_ = PoissonSolverType::MG;
        }
    }

    // FFT2D requires periodic x AND 2D mesh AND non-periodic y
    bool periodic_y = (p_y_lo == PoissonBC::Periodic && p_y_hi == PoissonBC::Periodic);
    bool fft2d_compatible = periodic_x && !periodic_y && mesh_->is2D();

    if (fft2d_poisson_solver_) {
        if (fft2d_compatible) {
            // Update FFT2D solver BCs (y direction can be Neumann or Dirichlet)
            fft2d_poisson_solver_->set_bc(p_x_lo, p_x_hi, p_y_lo, p_y_hi);
        } else if (selected_solver_ == PoissonSolverType::FFT2D) {
            // FFT2D was selected but BCs are now incompatible - switch to MG
            std::cerr << "[Poisson] Warning: FFT2D solver incompatible with BCs "
                      << "(requires periodic x for 2D mesh). Switching to MG.\n";
            selected_solver_ = PoissonSolverType::MG;
        }
    }
#endif

#ifdef USE_HYPRE
    // Check HYPRE compatibility with current BCs
    // KNOWN ISSUE: HYPRE GPU (CUDA) has numerical instability with 2D problems
    // that have periodic Y-direction BCs. Fall back to MG for these cases.
    // The issue manifests as NaN after ~10 time steps.
    // 3D works fine, and 2D with x-periodic + y-walls (channel) works fine.
    // But 2D with y-periodic (spanwise or fully periodic) fails.
    {
        bool hypre_periodic_y = (p_y_lo == PoissonBC::Periodic && p_y_hi == PoissonBC::Periodic);
        bool y_periodic_2d = mesh_->is2D() && hypre_periodic_y;

        if (selected_solver_ == PoissonSolverType::HYPRE && y_periodic_2d) {
#ifdef USE_GPU_OFFLOAD
            // Fall back to MG for 2D with y-periodic on GPU (HYPRE CUDA instability)
            std::cerr << "[Poisson] HYPRE->MG fallback: 2D y-periodic + GPU\n";
            selected_solver_ = PoissonSolverType::MG;
#endif
        }
    }
#endif

    // Initialize recycling inflow if enabled
    // Must be called after BCs are set so we can validate periodic z requirement
    initialize_recycling_inflow();
}

void RANSSolver::set_body_force(double fx, double fy, double fz) {
    fx_ = fx;
    fy_ = fy;
    fz_ = fz;
}

void RANSSolver::print_solver_info() const {
    std::cout << "\n=== Solver Configuration ===\n";

    // Mesh info
    std::cout << "Mesh: " << mesh_->Nx << " x " << mesh_->Ny;
    if (!mesh_->is2D()) {
        std::cout << " x " << mesh_->Nz << " (3D)";
    } else {
        std::cout << " (2D)";
    }
    std::cout << "\n";

    // Poisson solver selection
    std::cout << "Poisson solver: ";
    switch (selected_solver_) {
        case PoissonSolverType::FFT:
            std::cout << "FFT (2D-FFT in x-z + tridiagonal in y)";
            break;
        case PoissonSolverType::FFT2D:
            std::cout << "FFT2D (1D-FFT in x + tridiagonal in y)";
            break;
        case PoissonSolverType::FFT1D:
            std::cout << "FFT1D (1D-FFT in periodic dir + 2D Helmholtz)";
            break;
        case PoissonSolverType::HYPRE:
            std::cout << "HYPRE PFMG (geometric multigrid)";
            break;
        case PoissonSolverType::MG:
            std::cout << "Native Multigrid (V-cycle)";
            break;
        default:
            std::cout << "Unknown";
    }
    std::cout << "\n";

    std::cout << "Selection reason: " << selection_reason_ << "\n";

    // Poisson solver parameters (MG uses these, others may not)
    if (selected_solver_ == PoissonSolverType::MG) {
        std::cout << "MG params: tol=" << config_.poisson_tol
                  << ", max_vcycles=" << config_.poisson_max_vcycles
                  << ", omega=" << config_.poisson_omega << "\n";
    }

    // Boundary conditions
    std::cout << "Velocity BCs: x=";
    std::cout << (velocity_bc_.x_lo == VelocityBC::Periodic ? "periodic" :
                  velocity_bc_.x_lo == VelocityBC::NoSlip ? "wall" : "inflow/outflow");
    std::cout << ", y=";
    std::cout << (velocity_bc_.y_lo == VelocityBC::Periodic ? "periodic" :
                  velocity_bc_.y_lo == VelocityBC::NoSlip ? "wall" : "inflow/outflow");
    if (!mesh_->is2D()) {
        std::cout << ", z=";
        std::cout << (velocity_bc_.z_lo == VelocityBC::Periodic ? "periodic" :
                      velocity_bc_.z_lo == VelocityBC::NoSlip ? "wall" : "inflow/outflow");
    }
    std::cout << "\n";

    // Turbulence model
    std::cout << "Turbulence: ";
    if (turb_model_) {
        std::cout << turb_model_->name();
    } else {
        std::cout << "None (laminar)";
    }
    std::cout << "\n";

    // Discretization info
    std::cout << "Convective scheme: ";
    switch (config_.convective_scheme) {
        case ConvectiveScheme::Central: std::cout << "Central"; break;
        case ConvectiveScheme::Skew:    std::cout << "Skew-symmetric"; break;
        case ConvectiveScheme::Upwind:  std::cout << "Upwind (1st)"; break;
        case ConvectiveScheme::Upwind2: std::cout << "Upwind (2nd)"; break;
    }
    std::cout << ", space_order=" << config_.space_order << "\n";

    // Build info
    std::cout << "Build features: ";
    std::vector<std::string> features;
#ifdef USE_GPU_OFFLOAD
    features.push_back("GPU");
#else
    features.push_back("CPU");
#endif
#ifdef USE_FFT_POISSON
    features.push_back("FFT");
#endif
#ifdef USE_HYPRE
    features.push_back("HYPRE");
#endif
    for (size_t i = 0; i < features.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << features[i];
    }
    std::cout << "\n";

    std::cout << "============================\n\n";
}

void RANSSolver::initialize(const VectorField& initial_velocity) {
    velocity_ = initial_velocity;
    apply_velocity_bc();
    
    // Initialize k, omega for transport models if not already set
    if (turb_model_ && turb_model_->uses_transport_equations()) {
        // Estimate initial turbulence from velocity magnitude
        double u_max = 0.0;
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                double u = 0.5 * (velocity_.u(i, j) + velocity_.u(i+1, j));
                double v = 0.5 * (velocity_.v(i, j) + velocity_.v(i, j+1));
                double u_mag = std::sqrt(u*u + v*v);
                u_max = std::max(u_max, u_mag);
            }
        }
        // Use reasonable reference velocity - minimum 1% of bulk or 0.01 whichever is larger
        // This ensures k/omega stay above the low-turbulence threshold (1e-8) for EARSM
        double u_ref = std::max(u_max, 0.01);
        double Ti = 0.05;  // 5% turbulence intensity
        double k_init = 1.5 * (u_ref * Ti) * (u_ref * Ti);
        // Ensure k_init is physically meaningful (above low-turb threshold)
        k_init = std::max(k_init, 1e-7);
        
        double omega_init = k_init / (numerics::C_MU * config_.nu * 100.0);  // ν_t/ν ≈ 100 initially
        omega_init = std::max(omega_init, 1e-6);  // Ensure omega is also meaningful
        
        k_.fill(k_init);
        omega_.fill(omega_init);
        
        // Set wall values for omega (higher near walls)
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            // Bottom wall
            int j_bot = mesh_->j_begin();
            double y_bot = mesh_->wall_distance(i, j_bot);
            if (y_bot > 1e-10) {
                omega_(i, j_bot) = 10.0 * 6.0 * config_.nu / (0.075 * y_bot * y_bot);
            }
            
            // Top wall
            int j_top = mesh_->j_end() - 1;
            double y_top = mesh_->wall_distance(i, j_top);
            if (y_top > 1e-10) {
                omega_(i, j_top) = 10.0 * 6.0 * config_.nu / (0.075 * y_top * y_top);
            }
        }
    }
    
    if (turb_model_) {
        turb_model_->initialize(*mesh_, velocity_);
    }
    
#ifdef USE_GPU_OFFLOAD
    // Ensure initialized fields are mirrored to device for GPU runs
    sync_to_gpu();
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
        // Ensure k_init is physically meaningful (above low-turb threshold)
        k_init = std::max(k_init, 1e-7);
        
        double omega_init = k_init / (numerics::C_MU * config_.nu * 100.0);  // ν_t/ν ≈ 100 initially
        omega_init = std::max(omega_init, 1e-6);  // Ensure omega is also meaningful
        
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
    // CRITICAL: Sync k_ and omega_ to GPU after CPU-side initialization
    // These were modified at lines 419-420 AFTER initialize_gpu_buffers() ran
    // Without this sync, GPU kernels will use stale zero values instead of proper initial conditions
    if (turb_model_ && turb_model_->uses_transport_equations()) {
        #pragma omp target update to(k_ptr_[0:field_total_size_])
        #pragma omp target update to(omega_ptr_[0:field_total_size_])
    }

    // Also sync velocity to device so GPU path starts from correct ICs
    sync_to_gpu();
#endif
}

void RANSSolver::apply_velocity_bc() {
    NVTX_PUSH("apply_velocity_bc");
    
    // Get unified view (device pointers in GPU build, host pointers in CPU build)
    auto v = get_solver_view();
    
    const int Nx = v.Nx;
    const int Ny = v.Ny;
    const int Ng = v.Ng;
    const int u_stride = v.u_stride;
    const int v_stride = v.v_stride;
    const int u_total_Ny = Ny + 2 * Ng;
    const int v_total_Nx = Nx + 2 * Ng;

    const bool x_lo_periodic = (velocity_bc_.x_lo == VelocityBC::Periodic);
    const bool x_lo_noslip   = (velocity_bc_.x_lo == VelocityBC::NoSlip);
    const bool x_lo_inflow   = (velocity_bc_.x_lo == VelocityBC::Inflow);
    const bool x_hi_periodic = (velocity_bc_.x_hi == VelocityBC::Periodic);
    const bool x_hi_noslip   = (velocity_bc_.x_hi == VelocityBC::NoSlip);
    const bool x_hi_outflow  = (velocity_bc_.x_hi == VelocityBC::Outflow);

    const bool y_lo_periodic = (velocity_bc_.y_lo == VelocityBC::Periodic);
    const bool y_lo_noslip   = (velocity_bc_.y_lo == VelocityBC::NoSlip);
    const bool y_hi_periodic = (velocity_bc_.y_hi == VelocityBC::Periodic);
    const bool y_hi_noslip   = (velocity_bc_.y_hi == VelocityBC::NoSlip);

    // Validate that all BCs are supported
    if (!x_lo_periodic && !x_lo_noslip && !x_lo_inflow) {
        throw std::runtime_error("Unsupported velocity BC type for x_lo");
    }
    if (!x_hi_periodic && !x_hi_noslip && !x_hi_outflow) {
        throw std::runtime_error("Unsupported velocity BC type for x_hi");
    }
    if (!y_lo_periodic && !y_lo_noslip) {
        throw std::runtime_error("Unsupported velocity BC type for y_lo (only Periodic and NoSlip are implemented)");
    }
    if (!y_hi_periodic && !y_hi_noslip) {
        throw std::runtime_error("Unsupported velocity BC type for y_hi (only Periodic and NoSlip are implemented)");
    }

    double* u_ptr = v.u_face;
    double* v_ptr = v.v_face;
    [[maybe_unused]] const size_t u_total_size = velocity_.u_total_size();
    [[maybe_unused]] const size_t v_total_size = velocity_.v_total_size();

    // For 3D, apply x/y BCs to ALL z-planes (interior and ghost)
    // This is necessary because the z-BC code assumes x/y BCs are already applied
    // Nz_total = Nz + 2*Ng for 3D, 1 for 2D
    const int Nz_total = mesh_->is2D() ? 1 : (v.Nz + 2*Ng);
    const int u_plane_stride = mesh_->is2D() ? 0 : v.u_plane_stride;
    const int v_plane_stride = mesh_->is2D() ? 0 : v.v_plane_stride;

    // Apply u BCs in x-direction (for all k-planes including ghosts in 3D)
    const int n_u_x_bc = u_total_Ny * Ng * Nz_total;
    #pragma omp target teams distribute parallel for \
        map(present: u_ptr[0:u_total_size]) \
        firstprivate(Nx, Ny, Ng, u_stride, u_plane_stride, Nz_total, x_lo_periodic, x_lo_noslip, x_hi_periodic, x_hi_noslip)
    for (int idx = 0; idx < n_u_x_bc; ++idx) {
        int j = idx % u_total_Ny;
        int g = (idx / u_total_Ny) % Ng;
        int k = idx / (u_total_Ny * Ng);  // k = 0 to Nz_total-1 covers all planes
        double* u_plane_ptr = u_ptr + k * u_plane_stride;
        apply_u_bc_x_staggered(j, g, Nx, Ng, u_stride,
                              x_lo_periodic, x_lo_noslip,
                              x_hi_periodic, x_hi_noslip, u_plane_ptr);
    }

    // Apply u BCs in y-direction (for all k-planes including ghosts in 3D)
    const int n_u_y_bc = (Nx + 1 + 2 * Ng) * Ng * Nz_total;
    #pragma omp target teams distribute parallel for \
        map(present: u_ptr[0:u_total_size]) \
        firstprivate(Nx, Ny, Ng, u_stride, u_plane_stride, Nz_total, y_lo_periodic, y_lo_noslip, y_hi_periodic, y_hi_noslip)
    for (int idx = 0; idx < n_u_y_bc; ++idx) {
        int u_x_size = Nx + 1 + 2 * Ng;
        int i = idx % u_x_size;
        int g = (idx / u_x_size) % Ng;
        int k = idx / (u_x_size * Ng);
        double* u_plane_ptr = u_ptr + k * u_plane_stride;
        apply_u_bc_y_staggered(i, g, Ny, Ng, u_stride,
                              y_lo_periodic, y_lo_noslip,
                              y_hi_periodic, y_hi_noslip, u_plane_ptr);
    }

    // Apply v BCs in x-direction (for all k-planes including ghosts in 3D)
    const int n_v_x_bc = (Ny + 1 + 2 * Ng) * Ng * Nz_total;
    #pragma omp target teams distribute parallel for \
        map(present: v_ptr[0:v_total_size]) \
        firstprivate(Nx, Ny, Ng, v_stride, v_plane_stride, Nz_total, x_lo_periodic, x_lo_noslip, x_hi_periodic, x_hi_noslip)
    for (int idx = 0; idx < n_v_x_bc; ++idx) {
        int v_y_size = Ny + 1 + 2 * Ng;
        int j = idx % v_y_size;
        int g = (idx / v_y_size) % Ng;
        int k = idx / (v_y_size * Ng);
        double* v_plane_ptr = v_ptr + k * v_plane_stride;
        apply_v_bc_x_staggered(j, g, Nx, Ng, v_stride,
                              x_lo_periodic, x_lo_noslip,
                              x_hi_periodic, x_hi_noslip, v_plane_ptr);
    }

    // Apply v BCs in y-direction (for all k-planes including ghosts in 3D)
    const int n_v_y_bc = v_total_Nx * Ng * Nz_total;
    #pragma omp target teams distribute parallel for \
        map(present: v_ptr[0:v_total_size]) \
        firstprivate(Nx, Ny, Ng, v_stride, v_plane_stride, Nz_total, y_lo_periodic, y_lo_noslip, y_hi_periodic, y_hi_noslip)
    for (int idx = 0; idx < n_v_y_bc; ++idx) {
        int i = idx % v_total_Nx;
        int g = (idx / v_total_Nx) % Ng;
        int k = idx / (v_total_Nx * Ng);
        double* v_plane_ptr = v_ptr + k * v_plane_stride;
        apply_v_bc_y_staggered(i, g, Ny, Ng, v_stride,
                              y_lo_periodic, y_lo_noslip,
                              y_hi_periodic, y_hi_noslip, v_plane_ptr);
    }

    // CORNER FIX: For fully periodic domains, apply x-direction BCs again
    // to ensure corner ghosts are correctly wrapped after y-direction BCs modified them
    if (x_lo_periodic && x_hi_periodic) {
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_total_size]) \
            firstprivate(Nx, Ny, Ng, u_stride, u_plane_stride, Nz_total, x_lo_periodic, x_lo_noslip, x_hi_periodic, x_hi_noslip)
        for (int idx = 0; idx < n_u_x_bc; ++idx) {
            int j = idx % u_total_Ny;
            int g = (idx / u_total_Ny) % Ng;
            int k = idx / (u_total_Ny * Ng);
            double* u_plane_ptr = u_ptr + k * u_plane_stride;
            apply_u_bc_x_staggered(j, g, Nx, Ng, u_stride,
                                  x_lo_periodic, x_lo_noslip,
                                  x_hi_periodic, x_hi_noslip, u_plane_ptr);
        }
    }

    if (y_lo_periodic && y_hi_periodic) {
        #pragma omp target teams distribute parallel for \
            map(present: v_ptr[0:v_total_size]) \
            firstprivate(Nx, Ny, Ng, v_stride, v_plane_stride, Nz_total, y_lo_periodic, y_lo_noslip, y_hi_periodic, y_hi_noslip)
        for (int idx = 0; idx < n_v_y_bc; ++idx) {
            int i = idx % v_total_Nx;
            int g = (idx / v_total_Nx) % Ng;
            int k = idx / (v_total_Nx * Ng);
            double* v_plane_ptr = v_ptr + k * v_plane_stride;
            apply_v_bc_y_staggered(i, g, Ny, Ng, v_stride,
                                  y_lo_periodic, y_lo_noslip,
                                  y_hi_periodic, y_hi_noslip, v_plane_ptr);
        }
    }

    // Outflow BC at x_hi: zero-gradient (Neumann) for ghost cells
    // Copy interior values to ghost cells: ghost[Ng+Nx+1+g] = interior[Ng+Nx-1-g]
    if (x_hi_outflow) {
        // u outflow (normal velocity): apply to all j, all k-planes
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_total_size]) \
            firstprivate(Nx, Ng, u_stride, u_plane_stride, u_total_Ny, Nz_total)
        for (int idx = 0; idx < u_total_Ny * Ng * Nz_total; ++idx) {
            int j = idx % u_total_Ny;
            int g = (idx / u_total_Ny) % Ng;
            int k = idx / (u_total_Ny * Ng);
            double* u_plane_ptr = u_ptr + k * u_plane_stride;
            int i_ghost = Ng + Nx + 1 + g;     // Ghost face beyond outflow
            int i_interior = Ng + Nx - 1 - g;  // Interior face
            u_plane_ptr[j * u_stride + i_ghost] = u_plane_ptr[j * u_stride + i_interior];
        }

        // v outflow (tangential velocity): apply to all j, all k-planes
        const int v_y_total = Ny + 1 + 2 * Ng;
        #pragma omp target teams distribute parallel for \
            map(present: v_ptr[0:v_total_size]) \
            firstprivate(Nx, Ng, v_stride, v_plane_stride, v_y_total, Nz_total)
        for (int idx = 0; idx < v_y_total * Ng * Nz_total; ++idx) {
            int j = idx % v_y_total;
            int g = (idx / v_y_total) % Ng;
            int k = idx / (v_y_total * Ng);
            double* v_plane_ptr = v_ptr + k * v_plane_stride;
            int i_ghost = Ng + Nx + g;         // Ghost cell beyond outflow
            int i_interior = Ng + Nx - 1 - g;  // Interior cell
            v_plane_ptr[j * v_stride + i_ghost] = v_plane_ptr[j * v_stride + i_interior];
        }
    }

    // Inflow BC at x_lo: zero-gradient for ghost cells (inlet face set by recycling)
    if (x_lo_inflow) {
        // u inflow ghost cells: ghost[Ng-1-g] = interior[Ng+g]
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_total_size]) \
            firstprivate(Ng, u_stride, u_plane_stride, u_total_Ny, Nz_total)
        for (int idx = 0; idx < u_total_Ny * Ng * Nz_total; ++idx) {
            int j = idx % u_total_Ny;
            int g = (idx / u_total_Ny) % Ng;
            int k = idx / (u_total_Ny * Ng);
            double* u_plane_ptr = u_ptr + k * u_plane_stride;
            int i_ghost = Ng - 1 - g;     // Ghost face before inlet
            int i_interior = Ng + g;      // Interior face
            u_plane_ptr[j * u_stride + i_ghost] = u_plane_ptr[j * u_stride + i_interior];
        }

        // v inflow ghost cells
        const int v_y_total = Ny + 1 + 2 * Ng;
        #pragma omp target teams distribute parallel for \
            map(present: v_ptr[0:v_total_size]) \
            firstprivate(Ng, v_stride, v_plane_stride, v_y_total, Nz_total)
        for (int idx = 0; idx < v_y_total * Ng * Nz_total; ++idx) {
            int j = idx % v_y_total;
            int g = (idx / v_y_total) % Ng;
            int k = idx / (v_y_total * Ng);
            double* v_plane_ptr = v_ptr + k * v_plane_stride;
            int i_ghost = Ng - 1 - g;     // Ghost cell before inlet
            int i_interior = Ng + g;      // Interior cell
            v_plane_ptr[j * v_stride + i_ghost] = v_plane_ptr[j * v_stride + i_interior];
        }
    }

    // 3D z-direction boundary conditions
    if (!mesh_->is2D()) {
        const int Nz = v.Nz;
        // u_plane_stride and v_plane_stride already defined in outer scope
        const int w_stride = v.w_stride;
        const int w_plane_stride = v.w_plane_stride;
        double* w_ptr = v.w_face;
        [[maybe_unused]] const size_t w_total_size = velocity_.w_total_size();

        const bool z_lo_periodic = (velocity_bc_.z_lo == VelocityBC::Periodic);
        const bool z_hi_periodic = (velocity_bc_.z_hi == VelocityBC::Periodic);
        const bool z_lo_noslip = (velocity_bc_.z_lo == VelocityBC::NoSlip);
        const bool z_hi_noslip = (velocity_bc_.z_hi == VelocityBC::NoSlip);

        // Validate that z-direction BCs are supported (Inflow/Outflow not implemented)
        if (!z_lo_periodic && !z_lo_noslip) {
            throw std::runtime_error("Unsupported velocity BC type for z_lo (only Periodic and NoSlip are implemented)");
        }
        if (!z_hi_periodic && !z_hi_noslip) {
            throw std::runtime_error("Unsupported velocity BC type for z_hi (only Periodic and NoSlip are implemented)");
        }

        // Apply u BCs in z-direction (for all x-faces, all y rows)
        // Each x-face: (Nx+1) i-values, (Ny) j-values, Ng ghost layers at each z-end
        const int n_u_z_bc = (Nx + 1 + 2*Ng) * (Ny + 2*Ng) * Ng;
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_total_size]) \
            firstprivate(Nx, Ny, Ng, Nz, u_stride, u_plane_stride, z_lo_periodic, z_hi_periodic, z_lo_noslip, z_hi_noslip)
        for (int idx = 0; idx < n_u_z_bc; ++idx) {
            int i = idx % (Nx + 1 + 2*Ng);
            int j = (idx / (Nx + 1 + 2*Ng)) % (Ny + 2*Ng);
            int g = idx / ((Nx + 1 + 2*Ng) * (Ny + 2*Ng));
            // z-lo ghost: k = Ng-1-g = Ng-1, Ng-2, ... for g=0,1,...
            // z-hi ghost: k = Ng+Nz+g
            int k_lo = Ng - 1 - g;
            int k_hi = Ng + Nz + g;
            int src_lo = Ng;        // First interior k
            int src_hi = Ng + Nz - 1;  // Last interior k
            int idx_lo = k_lo * u_plane_stride + j * u_stride + i;
            int idx_hi = k_hi * u_plane_stride + j * u_stride + i;
            int idx_src_lo = src_lo * u_plane_stride + j * u_stride + i;
            int idx_src_hi = src_hi * u_plane_stride + j * u_stride + i;

            if (z_lo_periodic && z_hi_periodic) {
                // Periodic: copy from opposite interior boundary
                u_ptr[idx_lo] = u_ptr[(Ng + Nz - 1 - g) * u_plane_stride + j * u_stride + i];
                u_ptr[idx_hi] = u_ptr[(Ng + g) * u_plane_stride + j * u_stride + i];
            } else {
                if (z_lo_noslip) u_ptr[idx_lo] = -u_ptr[idx_src_lo];
                if (z_hi_noslip) u_ptr[idx_hi] = -u_ptr[idx_src_hi];
            }
        }

        // Apply v BCs in z-direction
        const int n_v_z_bc = (Nx + 2*Ng) * (Ny + 1 + 2*Ng) * Ng;
        #pragma omp target teams distribute parallel for \
            map(present: v_ptr[0:v_total_size]) \
            firstprivate(Nx, Ny, Ng, Nz, v_stride, v_plane_stride, z_lo_periodic, z_hi_periodic, z_lo_noslip, z_hi_noslip)
        for (int idx = 0; idx < n_v_z_bc; ++idx) {
            int i = idx % (Nx + 2*Ng);
            int j = (idx / (Nx + 2*Ng)) % (Ny + 1 + 2*Ng);
            int g = idx / ((Nx + 2*Ng) * (Ny + 1 + 2*Ng));
            int k_lo = Ng - 1 - g;
            int k_hi = Ng + Nz + g;
            int src_lo = Ng;
            int src_hi = Ng + Nz - 1;
            int idx_lo = k_lo * v_plane_stride + j * v_stride + i;
            int idx_hi = k_hi * v_plane_stride + j * v_stride + i;
            int idx_src_lo = src_lo * v_plane_stride + j * v_stride + i;
            int idx_src_hi = src_hi * v_plane_stride + j * v_stride + i;

            if (z_lo_periodic && z_hi_periodic) {
                v_ptr[idx_lo] = v_ptr[(Ng + Nz - 1 - g) * v_plane_stride + j * v_stride + i];
                v_ptr[idx_hi] = v_ptr[(Ng + g) * v_plane_stride + j * v_stride + i];
            } else {
                if (z_lo_noslip) v_ptr[idx_lo] = -v_ptr[idx_src_lo];
                if (z_hi_noslip) v_ptr[idx_hi] = -v_ptr[idx_src_hi];
            }
        }

        // Apply w BCs in z-direction (w is at z-faces, so different treatment)
        // For periodic: w at k=Ng and k=Ng+Nz should be same (wrap around)
        const int n_w_z_bc = (Nx + 2*Ng) * (Ny + 2*Ng) * Ng;
        #pragma omp target teams distribute parallel for \
            map(present: w_ptr[0:w_total_size]) \
            firstprivate(Nx, Ny, Ng, Nz, w_stride, w_plane_stride, z_lo_periodic, z_hi_periodic, z_lo_noslip, z_hi_noslip)
        for (int idx = 0; idx < n_w_z_bc; ++idx) {
            int i = idx % (Nx + 2*Ng);
            int j = (idx / (Nx + 2*Ng)) % (Ny + 2*Ng);
            int g = idx / ((Nx + 2*Ng) * (Ny + 2*Ng));
            int k_lo = Ng - 1 - g;
            int k_hi = Ng + Nz + 1 + g;  // Note: Nz+1 z-faces for w
            int idx_lo = k_lo * w_plane_stride + j * w_stride + i;
            int idx_hi = k_hi * w_plane_stride + j * w_stride + i;

            if (z_lo_periodic && z_hi_periodic) {
                // CRITICAL for staggered grid with periodic BCs:
                // The topmost interior face (Ng+Nz) IS the bottommost interior face (Ng)
                // They represent the same physical location in a periodic domain
                if (g == 0) {
                    w_ptr[(Ng + Nz) * w_plane_stride + j * w_stride + i] =
                        w_ptr[Ng * w_plane_stride + j * w_stride + i];
                }
                // For w at z-faces with periodic BC:
                // Ghost at k=Ng-1-g gets value from k=Ng+Nz-1-g (interior near hi)
                // Ghost at k=Ng+Nz+1+g gets value from k=Ng+1+g (interior near lo)
                w_ptr[idx_lo] = w_ptr[(Ng + Nz - 1 - g) * w_plane_stride + j * w_stride + i];
                w_ptr[idx_hi] = w_ptr[(Ng + 1 + g) * w_plane_stride + j * w_stride + i];
            } else {
                // For no-slip: w at boundaries should be zero (normal velocity)
                if (z_lo_noslip) {
                    // w at k=Ng (first interior z-face) = 0 for solid wall
                    if (g == 0) {
                        w_ptr[(Ng) * w_plane_stride + j * w_stride + i] = 0.0;
                    }
                    w_ptr[idx_lo] = -w_ptr[(Ng + g + 1) * w_plane_stride + j * w_stride + i];
                }
                if (z_hi_noslip) {
                    // w at k=Ng+Nz (last interior z-face) = 0 for solid wall
                    if (g == 0) {
                        w_ptr[(Ng + Nz) * w_plane_stride + j * w_stride + i] = 0.0;
                    }
                    w_ptr[idx_hi] = -w_ptr[(Ng + Nz - 1 - g) * w_plane_stride + j * w_stride + i];
                }
            }
        }

        // Apply w BCs in x and y directions
        // w in x-direction
        const int n_w_x_bc = (Ny + 2*Ng) * (Nz + 1 + 2*Ng) * Ng;
        #pragma omp target teams distribute parallel for \
            map(present: w_ptr[0:w_total_size]) \
            firstprivate(Nx, Ng, w_stride, w_plane_stride, x_lo_periodic, x_lo_noslip, x_hi_periodic, x_hi_noslip)
        for (int idx = 0; idx < n_w_x_bc; ++idx) {
            int j = idx % (Ny + 2*Ng);
            int k = (idx / (Ny + 2*Ng)) % (Nz + 1 + 2*Ng);
            int g = idx / ((Ny + 2*Ng) * (Nz + 1 + 2*Ng));
            int i_lo = Ng - 1 - g;
            int i_hi = Ng + Nx + g;
            int idx_lo = k * w_plane_stride + j * w_stride + i_lo;
            int idx_hi = k * w_plane_stride + j * w_stride + i_hi;

            if (x_lo_periodic && x_hi_periodic) {
                w_ptr[idx_lo] = w_ptr[k * w_plane_stride + j * w_stride + (Ng + Nx - 1 - g)];
                w_ptr[idx_hi] = w_ptr[k * w_plane_stride + j * w_stride + (Ng + g)];
            } else {
                if (x_lo_noslip) w_ptr[idx_lo] = -w_ptr[k * w_plane_stride + j * w_stride + (Ng + g)];
                if (x_hi_noslip) w_ptr[idx_hi] = -w_ptr[k * w_plane_stride + j * w_stride + (Ng + Nx - 1 - g)];
            }
        }

        // w inflow/outflow in x-direction (zero-gradient)
        if (x_lo_inflow) {
            #pragma omp target teams distribute parallel for \
                map(present: w_ptr[0:w_total_size]) \
                firstprivate(Nx, Ny, Nz, Ng, w_stride, w_plane_stride)
            for (int idx = 0; idx < n_w_x_bc; ++idx) {
                int j = idx % (Ny + 2*Ng);
                int k = (idx / (Ny + 2*Ng)) % (Nz + 1 + 2*Ng);
                int g = idx / ((Ny + 2*Ng) * (Nz + 1 + 2*Ng));
                int i_ghost = Ng - 1 - g;
                int i_interior = Ng + g;
                w_ptr[k * w_plane_stride + j * w_stride + i_ghost] =
                    w_ptr[k * w_plane_stride + j * w_stride + i_interior];
            }
        }
        if (x_hi_outflow) {
            #pragma omp target teams distribute parallel for \
                map(present: w_ptr[0:w_total_size]) \
                firstprivate(Nx, Ny, Nz, Ng, w_stride, w_plane_stride)
            for (int idx = 0; idx < n_w_x_bc; ++idx) {
                int j = idx % (Ny + 2*Ng);
                int k = (idx / (Ny + 2*Ng)) % (Nz + 1 + 2*Ng);
                int g = idx / ((Ny + 2*Ng) * (Nz + 1 + 2*Ng));
                int i_ghost = Ng + Nx + g;
                int i_interior = Ng + Nx - 1 - g;
                w_ptr[k * w_plane_stride + j * w_stride + i_ghost] =
                    w_ptr[k * w_plane_stride + j * w_stride + i_interior];
            }
        }

        // w in y-direction
        const int n_w_y_bc = (Nx + 2*Ng) * (Nz + 1 + 2*Ng) * Ng;
        #pragma omp target teams distribute parallel for \
            map(present: w_ptr[0:w_total_size]) \
            firstprivate(Ny, Ng, w_stride, w_plane_stride, y_lo_periodic, y_lo_noslip, y_hi_periodic, y_hi_noslip)
        for (int idx = 0; idx < n_w_y_bc; ++idx) {
            int i = idx % (Nx + 2*Ng);
            int k = (idx / (Nx + 2*Ng)) % (Nz + 1 + 2*Ng);
            int g = idx / ((Nx + 2*Ng) * (Nz + 1 + 2*Ng));
            int j_lo = Ng - 1 - g;
            int j_hi = Ng + Ny + g;
            int idx_lo = k * w_plane_stride + j_lo * w_stride + i;
            int idx_hi = k * w_plane_stride + j_hi * w_stride + i;

            if (y_lo_periodic && y_hi_periodic) {
                w_ptr[idx_lo] = w_ptr[k * w_plane_stride + (Ng + Ny - 1 - g) * w_stride + i];
                w_ptr[idx_hi] = w_ptr[k * w_plane_stride + (Ng + g) * w_stride + i];
            } else {
                if (y_lo_noslip) w_ptr[idx_lo] = -w_ptr[k * w_plane_stride + (Ng + g) * w_stride + i];
                if (y_hi_noslip) w_ptr[idx_hi] = -w_ptr[k * w_plane_stride + (Ng + Ny - 1 - g) * w_stride + i];
            }
        }
        // NOTE: x/y BCs for u and v across all k-planes are now handled
        // above (lines 1268-1360) using the staggered kernel functions
        // which properly handle staggered grid periodic BCs.
    }

    NVTX_POP();  // End apply_velocity_bc
}

void RANSSolver::compute_convective_term(const VectorField& vel, VectorField& conv) {
    NVTX_SCOPE_CONVECT("solver:convective_term");

    (void)vel;   // Unused - always operates on velocity_ via view
    (void)conv;  // Unused - always operates on conv_ via view

    // Get unified view
    auto v = get_solver_view();

    const double dx = v.dx;
    const double dy = v.dy;
    const int Nx = v.Nx;
    const int Ny = v.Ny;
    const int Nz = v.Nz;
    const int Ng = v.Ng;
    const int u_stride = v.u_stride;
    const int v_stride = v.v_stride;
    const bool use_central = (config_.convective_scheme == ConvectiveScheme::Central);
    const bool use_skew = (config_.convective_scheme == ConvectiveScheme::Skew);
    const bool use_upwind2 = (config_.convective_scheme == ConvectiveScheme::Upwind2);
    const bool use_O4 = (config_.space_order == 4);

    // Periodic flags for O4 boundary safety checks
    const bool x_periodic = (velocity_bc_.x_lo == VelocityBC::Periodic) &&
                            (velocity_bc_.x_hi == VelocityBC::Periodic);
    const bool y_periodic = (velocity_bc_.y_lo == VelocityBC::Periodic) &&
                            (velocity_bc_.y_hi == VelocityBC::Periodic);
    const bool z_periodic = (velocity_bc_.z_lo == VelocityBC::Periodic) &&
                            (velocity_bc_.z_hi == VelocityBC::Periodic);

    [[maybe_unused]] const size_t u_total_size = velocity_.u_total_size();
    [[maybe_unused]] const size_t v_total_size = velocity_.v_total_size();
    [[maybe_unused]] const size_t w_total_size = velocity_.w_total_size();

    const double* u_ptr      = v.u_face;
    const double* v_ptr      = v.v_face;
    double*       conv_u_ptr = v.conv_u;
    double*       conv_v_ptr = v.conv_v;

    // 3D path
    if (!mesh_->is2D()) {
        const double dz = v.dz;
        const int u_plane_stride = v.u_plane_stride;
        const int v_plane_stride = v.v_plane_stride;
        const int w_stride = v.w_stride;
        const int w_plane_stride = v.w_plane_stride;
        const double* w_ptr = v.w_face;
        double*       conv_w_ptr = v.conv_w;

        const int n_u_faces = (Nx + 1) * Ny * Nz;
        const int n_v_faces = Nx * (Ny + 1) * Nz;
        const int n_w_faces = Nx * Ny * (Nz + 1);
        const int conv_u_stride = u_stride;
        const int conv_u_plane_stride = u_plane_stride;
        const int conv_v_stride = v_stride;
        const int conv_v_plane_stride = v_plane_stride;
        const int conv_w_stride = w_stride;
        const int conv_w_plane_stride = w_plane_stride;

        if (use_O4 && use_skew) {
            // O4 Skew-symmetric advection (energy-conserving with 4th-order advective derivatives)
            #pragma omp target teams distribute parallel for \
                map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], w_ptr[0:w_total_size], conv_u_ptr[0:u_total_size]) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, Ng, Nx, Ny, Nz, x_periodic, y_periodic, z_periodic)
            for (int idx = 0; idx < n_u_faces; ++idx) {
                int i = idx % (Nx + 1) + Ng;
                int j = (idx / (Nx + 1)) % Ny + Ng;
                int k = idx / ((Nx + 1) * Ny) + Ng;

                convective_u_face_kernel_skew_O4_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, u_stride, u_plane_stride,
                    dx, dy, dz, Ng, Nx, Ny, Nz, x_periodic, y_periodic, z_periodic,
                    u_ptr, v_ptr, w_ptr, conv_u_ptr);
            }

            #pragma omp target teams distribute parallel for \
                map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], w_ptr[0:w_total_size], conv_v_ptr[0:v_total_size]) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, Ng, Nx, Ny, Nz, x_periodic, y_periodic, z_periodic)
            for (int idx = 0; idx < n_v_faces; ++idx) {
                int i = idx % Nx + Ng;
                int j = (idx / Nx) % (Ny + 1) + Ng;
                int k = idx / (Nx * (Ny + 1)) + Ng;

                convective_v_face_kernel_skew_O4_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, v_stride, v_plane_stride,
                    dx, dy, dz, Ng, Nx, Ny, Nz, x_periodic, y_periodic, z_periodic,
                    u_ptr, v_ptr, w_ptr, conv_v_ptr);
            }

            #pragma omp target teams distribute parallel for \
                map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], w_ptr[0:w_total_size], conv_w_ptr[0:w_total_size]) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, Ng, Nx, Ny, Nz, x_periodic, y_periodic, z_periodic)
            for (int idx = 0; idx < n_w_faces; ++idx) {
                int i = idx % Nx + Ng;
                int j = (idx / Nx) % Ny + Ng;
                int k = idx / (Nx * Ny) + Ng;

                convective_w_face_kernel_skew_O4_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, w_stride, w_plane_stride,
                    dx, dy, dz, Ng, Nx, Ny, Nz, x_periodic, y_periodic, z_periodic,
                    u_ptr, v_ptr, w_ptr, conv_w_ptr);
            }
        } else if (use_skew) {
            // O2 Skew-symmetric (energy-conserving) advection
            #pragma omp target teams distribute parallel for \
                map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], w_ptr[0:w_total_size], conv_u_ptr[0:u_total_size]) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, conv_u_stride, conv_u_plane_stride, Nx, Ny, Ng)
            for (int idx = 0; idx < n_u_faces; ++idx) {
                int i = idx % (Nx + 1) + Ng;
                int j = (idx / (Nx + 1)) % Ny + Ng;
                int k = idx / ((Nx + 1) * Ny) + Ng;

                convective_u_face_kernel_skew_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, conv_u_stride, conv_u_plane_stride,
                    dx, dy, dz, u_ptr, v_ptr, w_ptr, conv_u_ptr);
            }

            #pragma omp target teams distribute parallel for \
                map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], w_ptr[0:w_total_size], conv_v_ptr[0:v_total_size]) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, conv_v_stride, conv_v_plane_stride, Nx, Ny, Ng)
            for (int idx = 0; idx < n_v_faces; ++idx) {
                int i = idx % Nx + Ng;
                int j = (idx / Nx) % (Ny + 1) + Ng;
                int k = idx / (Nx * (Ny + 1)) + Ng;

                convective_v_face_kernel_skew_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, conv_v_stride, conv_v_plane_stride,
                    dx, dy, dz, u_ptr, v_ptr, w_ptr, conv_v_ptr);
            }

            #pragma omp target teams distribute parallel for \
                map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], w_ptr[0:w_total_size], conv_w_ptr[0:w_total_size]) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, conv_w_stride, conv_w_plane_stride, Nx, Ny, Ng)
            for (int idx = 0; idx < n_w_faces; ++idx) {
                int i = idx % Nx + Ng;
                int j = (idx / Nx) % Ny + Ng;
                int k = idx / (Nx * Ny) + Ng;

                convective_w_face_kernel_skew_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, conv_w_stride, conv_w_plane_stride,
                    dx, dy, dz, u_ptr, v_ptr, w_ptr, conv_w_ptr);
            }
        } else if (use_upwind2) {
            // 2nd-order upwind with minmod limiter
            #pragma omp target teams distribute parallel for \
                map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], w_ptr[0:w_total_size], conv_u_ptr[0:u_total_size]) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, conv_u_stride, conv_u_plane_stride, Nx, Ny, Ng)
            for (int idx = 0; idx < n_u_faces; ++idx) {
                int i = idx % (Nx + 1) + Ng;
                int j = (idx / (Nx + 1)) % Ny + Ng;
                int k = idx / ((Nx + 1) * Ny) + Ng;

                convective_u_face_kernel_upwind2_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, conv_u_stride, conv_u_plane_stride,
                    dx, dy, dz, u_ptr, v_ptr, w_ptr, conv_u_ptr);
            }

            #pragma omp target teams distribute parallel for \
                map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], w_ptr[0:w_total_size], conv_v_ptr[0:v_total_size]) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, conv_v_stride, conv_v_plane_stride, Nx, Ny, Ng)
            for (int idx = 0; idx < n_v_faces; ++idx) {
                int i = idx % Nx + Ng;
                int j = (idx / Nx) % (Ny + 1) + Ng;
                int k = idx / (Nx * (Ny + 1)) + Ng;

                convective_v_face_kernel_upwind2_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, conv_v_stride, conv_v_plane_stride,
                    dx, dy, dz, u_ptr, v_ptr, w_ptr, conv_v_ptr);
            }

            #pragma omp target teams distribute parallel for \
                map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], w_ptr[0:w_total_size], conv_w_ptr[0:w_total_size]) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, conv_w_stride, conv_w_plane_stride, Nx, Ny, Ng)
            for (int idx = 0; idx < n_w_faces; ++idx) {
                int i = idx % Nx + Ng;
                int j = (idx / Nx) % Ny + Ng;
                int k = idx / (Nx * Ny) + Ng;

                convective_w_face_kernel_upwind2_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, conv_w_stride, conv_w_plane_stride,
                    dx, dy, dz, u_ptr, v_ptr, w_ptr, conv_w_ptr);
            }
        } else if (use_O4 && use_central) {
            // O4 Central advection (4th-order derivatives, hybrid O4/O2 near boundaries)
            #pragma omp target teams distribute parallel for \
                map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], w_ptr[0:w_total_size], conv_u_ptr[0:u_total_size]) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, Ng, Nx, Ny, Nz, x_periodic, y_periodic, z_periodic)
            for (int idx = 0; idx < n_u_faces; ++idx) {
                int i = idx % (Nx + 1) + Ng;
                int j = (idx / (Nx + 1)) % Ny + Ng;
                int k = idx / ((Nx + 1) * Ny) + Ng;

                convective_u_face_kernel_central_O4_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, u_stride, u_plane_stride,
                    dx, dy, dz, Ng, Nx, Ny, Nz, x_periodic, y_periodic, z_periodic,
                    u_ptr, v_ptr, w_ptr, conv_u_ptr);
            }

            #pragma omp target teams distribute parallel for \
                map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], w_ptr[0:w_total_size], conv_v_ptr[0:v_total_size]) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, Ng, Nx, Ny, Nz, x_periodic, y_periodic, z_periodic)
            for (int idx = 0; idx < n_v_faces; ++idx) {
                int i = idx % Nx + Ng;
                int j = (idx / Nx) % (Ny + 1) + Ng;
                int k = idx / (Nx * (Ny + 1)) + Ng;

                convective_v_face_kernel_central_O4_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, v_stride, v_plane_stride,
                    dx, dy, dz, Ng, Nx, Ny, Nz, x_periodic, y_periodic, z_periodic,
                    u_ptr, v_ptr, w_ptr, conv_v_ptr);
            }

            #pragma omp target teams distribute parallel for \
                map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], w_ptr[0:w_total_size], conv_w_ptr[0:w_total_size]) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, Ng, Nx, Ny, Nz, x_periodic, y_periodic, z_periodic)
            for (int idx = 0; idx < n_w_faces; ++idx) {
                int i = idx % Nx + Ng;
                int j = (idx / Nx) % Ny + Ng;
                int k = idx / (Nx * Ny) + Ng;

                convective_w_face_kernel_central_O4_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, w_stride, w_plane_stride,
                    dx, dy, dz, Ng, Nx, Ny, Nz, x_periodic, y_periodic, z_periodic,
                    u_ptr, v_ptr, w_ptr, conv_w_ptr);
            }
        } else {
            // Central or 1st-order upwind (O2 path)
            #pragma omp target teams distribute parallel for \
                map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], w_ptr[0:w_total_size], conv_u_ptr[0:u_total_size]) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, use_central, Nx, Ny, Ng)
            for (int idx = 0; idx < n_u_faces; ++idx) {
                int i = idx % (Nx + 1) + Ng;
                int j = (idx / (Nx + 1)) % Ny + Ng;
                int k = idx / ((Nx + 1) * Ny) + Ng;

                convective_u_face_kernel_staggered_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, u_stride, u_plane_stride,
                    dx, dy, dz, use_central, u_ptr, v_ptr, w_ptr, conv_u_ptr);
            }

            #pragma omp target teams distribute parallel for \
                map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], w_ptr[0:w_total_size], conv_v_ptr[0:v_total_size]) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, use_central, Nx, Ny, Ng)
            for (int idx = 0; idx < n_v_faces; ++idx) {
                int i = idx % Nx + Ng;
                int j = (idx / Nx) % (Ny + 1) + Ng;
                int k = idx / (Nx * (Ny + 1)) + Ng;

                convective_v_face_kernel_staggered_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, v_stride, v_plane_stride,
                    dx, dy, dz, use_central, u_ptr, v_ptr, w_ptr, conv_v_ptr);
            }

            #pragma omp target teams distribute parallel for \
                map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], w_ptr[0:w_total_size], conv_w_ptr[0:w_total_size]) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, use_central, Nx, Ny, Ng)
            for (int idx = 0; idx < n_w_faces; ++idx) {
                int i = idx % Nx + Ng;
                int j = (idx / Nx) % Ny + Ng;
                int k = idx / (Nx * Ny) + Ng;

                convective_w_face_kernel_staggered_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, w_stride, w_plane_stride,
                    dx, dy, dz, use_central, u_ptr, v_ptr, w_ptr, conv_w_ptr);
            }
        }
        return;
    }

    // 2D path
    const int n_u_faces = (Nx + 1) * Ny;
    const int n_v_faces = Nx * (Ny + 1);
    const int conv_u_stride = u_stride;
    const int conv_v_stride = v_stride;

    // Warn once if O4 requested but not implemented for 2D advection
    if (use_O4 && (use_skew || use_central)) {
        static bool warned_o4_2d = false;
        if (!warned_o4_2d) {
            std::cerr << "[Solver] WARNING: space_order=4 requested but O4 advection kernels "
                      << "are not implemented for 2D. Using O2 advection.\n";
            warned_o4_2d = true;
        }
    }

    if (use_skew) {
        // Skew-symmetric (energy-conserving) advection - 2D
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], conv_u_ptr[0:u_total_size]) \
            firstprivate(dx, dy, u_stride, v_stride, conv_u_stride, Nx, Ng)
        for (int idx = 0; idx < n_u_faces; ++idx) {
            int i = idx % (Nx + 1) + Ng;
            int j = idx / (Nx + 1) + Ng;

            convective_u_face_kernel_skew_2d(i, j, u_stride, v_stride, conv_u_stride,
                                            dx, dy, u_ptr, v_ptr, conv_u_ptr);
        }

        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], conv_v_ptr[0:v_total_size]) \
            firstprivate(dx, dy, u_stride, v_stride, conv_v_stride, Nx, Ng)
        for (int idx = 0; idx < n_v_faces; ++idx) {
            int i = idx % Nx + Ng;
            int j = idx / Nx + Ng;

            convective_v_face_kernel_skew_2d(i, j, u_stride, v_stride, conv_v_stride,
                                            dx, dy, u_ptr, v_ptr, conv_v_ptr);
        }
    } else if (use_upwind2) {
        // 2nd-order upwind with minmod limiter - 2D
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], conv_u_ptr[0:u_total_size]) \
            firstprivate(dx, dy, u_stride, v_stride, conv_u_stride, Nx, Ng)
        for (int idx = 0; idx < n_u_faces; ++idx) {
            int i = idx % (Nx + 1) + Ng;
            int j = idx / (Nx + 1) + Ng;

            convective_u_face_kernel_upwind2_2d(i, j, u_stride, v_stride, conv_u_stride,
                                               dx, dy, u_ptr, v_ptr, conv_u_ptr);
        }

        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], conv_v_ptr[0:v_total_size]) \
            firstprivate(dx, dy, u_stride, v_stride, conv_v_stride, Nx, Ng)
        for (int idx = 0; idx < n_v_faces; ++idx) {
            int i = idx % Nx + Ng;
            int j = idx / Nx + Ng;

            convective_v_face_kernel_upwind2_2d(i, j, u_stride, v_stride, conv_v_stride,
                                               dx, dy, u_ptr, v_ptr, conv_v_ptr);
        }
    } else {
        // Central or 1st-order upwind (original path) - 2D
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], conv_u_ptr[0:u_total_size]) \
            firstprivate(dx, dy, u_stride, v_stride, use_central, Nx, Ng)
        for (int idx = 0; idx < n_u_faces; ++idx) {
            int i_local = idx % (Nx + 1);
            int j_local = idx / (Nx + 1);
            int i = i_local + Ng;
            int j = j_local + Ng;

            convective_u_face_kernel_staggered(i, j, u_stride, v_stride, u_stride, dx, dy, use_central,
                                              u_ptr, v_ptr, conv_u_ptr);
        }

        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], conv_v_ptr[0:v_total_size]) \
            firstprivate(dx, dy, u_stride, v_stride, use_central, Nx, Ng)
        for (int idx = 0; idx < n_v_faces; ++idx) {
            int i_local = idx % Nx;
            int j_local = idx / Nx;
            int i = i_local + Ng;
            int j = j_local + Ng;

            convective_v_face_kernel_staggered(i, j, u_stride, v_stride, v_stride, dx, dy, use_central,
                                              u_ptr, v_ptr, conv_v_ptr);
        }
    }
}

void RANSSolver::compute_diffusive_term(const VectorField& vel, const ScalarField& nu_eff,
                                        VectorField& diff) {
    NVTX_SCOPE_DIFFUSE("solver:diffusive_term");

    (void)vel;     // Unused - always operates on velocity_ via view
    (void)nu_eff;  // Unused - always operates on nu_eff_ via view
    (void)diff;    // Unused - always operates on diff_ via view

    // Get unified view
    auto v = get_solver_view();

    const double dx = v.dx;
    const double dy = v.dy;
    const int Nx = v.Nx;
    const int Ny = v.Ny;
    const int Nz = v.Nz;
    const int Ng = v.Ng;
    const int u_stride = v.u_stride;
    const int v_stride = v.v_stride;
    const int nu_stride = v.cell_stride;

    [[maybe_unused]] const size_t u_total_size = velocity_.u_total_size();
    [[maybe_unused]] const size_t v_total_size = velocity_.v_total_size();
    [[maybe_unused]] const size_t w_total_size = velocity_.w_total_size();
    [[maybe_unused]] const size_t nu_total_size = field_total_size_;

    const double* u_ptr      = v.u_face;
    const double* v_ptr      = v.v_face;
    const double* nu_ptr     = v.nu_eff;
    double*       diff_u_ptr = v.diff_u;
    double*       diff_v_ptr = v.diff_v;

    // 3D path
    if (!mesh_->is2D()) {
        const double dz = v.dz;
        const int u_plane_stride = v.u_plane_stride;
        const int v_plane_stride = v.v_plane_stride;
        const int w_stride = v.w_stride;
        const int w_plane_stride = v.w_plane_stride;
        const int nu_plane_stride = v.cell_plane_stride;
        const double* w_ptr = v.w_face;
        double*       diff_w_ptr = v.diff_w;

        // Compute u-momentum diffusion at x-faces (3D)
        const int n_u_faces = (Nx + 1) * Ny * Nz;
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_total_size], nu_ptr[0:nu_total_size], diff_u_ptr[0:u_total_size]) \
            firstprivate(dx, dy, dz, u_stride, u_plane_stride, nu_stride, nu_plane_stride, Nx, Ny, Ng)
        for (int idx = 0; idx < n_u_faces; ++idx) {
            int i = idx % (Nx + 1) + Ng;
            int j = (idx / (Nx + 1)) % Ny + Ng;
            int k = idx / ((Nx + 1) * Ny) + Ng;

            diffusive_u_face_kernel_staggered_3d(i, j, k,
                u_stride, u_plane_stride, nu_stride, nu_plane_stride, u_stride, u_plane_stride,
                dx, dy, dz, u_ptr, nu_ptr, diff_u_ptr);
        }

        // Compute v-momentum diffusion at y-faces (3D)
        const int n_v_faces = Nx * (Ny + 1) * Nz;
        #pragma omp target teams distribute parallel for \
            map(present: v_ptr[0:v_total_size], nu_ptr[0:nu_total_size], diff_v_ptr[0:v_total_size]) \
            firstprivate(dx, dy, dz, v_stride, v_plane_stride, nu_stride, nu_plane_stride, Nx, Ny, Ng)
        for (int idx = 0; idx < n_v_faces; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % (Ny + 1) + Ng;
            int k = idx / (Nx * (Ny + 1)) + Ng;

            diffusive_v_face_kernel_staggered_3d(i, j, k,
                v_stride, v_plane_stride, nu_stride, nu_plane_stride, v_stride, v_plane_stride,
                dx, dy, dz, v_ptr, nu_ptr, diff_v_ptr);
        }

        // Compute w-momentum diffusion at z-faces (3D)
        const int n_w_faces = Nx * Ny * (Nz + 1);
        #pragma omp target teams distribute parallel for \
            map(present: w_ptr[0:w_total_size], nu_ptr[0:nu_total_size], diff_w_ptr[0:w_total_size]) \
            firstprivate(dx, dy, dz, w_stride, w_plane_stride, nu_stride, nu_plane_stride, Nx, Ny, Ng)
        for (int idx = 0; idx < n_w_faces; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % Ny + Ng;
            int k = idx / (Nx * Ny) + Ng;

            diffusive_w_face_kernel_staggered_3d(i, j, k,
                w_stride, w_plane_stride, nu_stride, nu_plane_stride, w_stride, w_plane_stride,
                dx, dy, dz, w_ptr, nu_ptr, diff_w_ptr);
        }
        return;
    }

    // 2D path
    // Compute u-momentum diffusion at x-faces
    const int n_u_faces = (Nx + 1) * Ny;
    #pragma omp target teams distribute parallel for \
        map(present: u_ptr[0:u_total_size], nu_ptr[0:nu_total_size], diff_u_ptr[0:u_total_size]) \
        firstprivate(dx, dy, u_stride, nu_stride, Nx, Ng)
    for (int idx = 0; idx < n_u_faces; ++idx) {
        int i_local = idx % (Nx + 1);
        int j_local = idx / (Nx + 1);
        int i = i_local + Ng;
        int j = j_local + Ng;

        diffusive_u_face_kernel_staggered(i, j, u_stride, nu_stride, u_stride, dx, dy,
                                         u_ptr, nu_ptr, diff_u_ptr);
    }

    // Compute v-momentum diffusion at y-faces
    const int n_v_faces = Nx * (Ny + 1);
    #pragma omp target teams distribute parallel for \
        map(present: v_ptr[0:v_total_size], nu_ptr[0:nu_total_size], diff_v_ptr[0:v_total_size]) \
        firstprivate(dx, dy, v_stride, nu_stride, Nx, Ng)
    for (int idx = 0; idx < n_v_faces; ++idx) {
        int i_local = idx % Nx;
        int j_local = idx / Nx;
        int i = i_local + Ng;
        int j = j_local + Ng;

        diffusive_v_face_kernel_staggered(i, j, v_stride, nu_stride, v_stride, dx, dy,
                                         v_ptr, nu_ptr, diff_v_ptr);
    }
}

void RANSSolver::compute_divergence(VelocityWhich which, ScalarField& div) {
    (void)div;  // Unused - always operates on div_velocity_ via view

    // Get unified view
    auto v = get_solver_view();
    auto vel_ptrs = select_face_velocity(v, which);

    const double dx = v.dx;
    const double dy = v.dy;
    const int Nx = v.Nx;
    const int Ny = v.Ny;
    const int Nz = v.Nz;
    const int Ng = v.Ng;
    const int div_stride = v.cell_stride;

    // O4 spatial discretization for divergence (Dfc_O4)
    const bool use_O4 = (config_.space_order == 4);

    // Periodic flags for O4 boundary handling
    const bool x_periodic = (velocity_bc_.x_lo == VelocityBC::Periodic) &&
                            (velocity_bc_.x_hi == VelocityBC::Periodic);
    const bool y_periodic = (velocity_bc_.y_lo == VelocityBC::Periodic) &&
                            (velocity_bc_.y_hi == VelocityBC::Periodic);
    const bool z_periodic = (velocity_bc_.z_lo == VelocityBC::Periodic) &&
                            (velocity_bc_.z_hi == VelocityBC::Periodic);

    const int u_stride = vel_ptrs.u_stride;
    const int v_stride = vel_ptrs.v_stride;

    [[maybe_unused]] const size_t u_total_size = velocity_.u_total_size();
    [[maybe_unused]] const size_t v_total_size = velocity_.v_total_size();
    [[maybe_unused]] const size_t w_total_size = velocity_.w_total_size();
    [[maybe_unused]] const size_t div_total_size = field_total_size_;

    const double* u_ptr = vel_ptrs.u;
    const double* v_ptr = vel_ptrs.v;
    double* div_ptr = v.div;

    // 3D path
    if (!mesh_->is2D()) {
        const double dz = v.dz;
        const int u_plane_stride = vel_ptrs.u_plane_stride;
        const int v_plane_stride = vel_ptrs.v_plane_stride;
        const int w_stride = vel_ptrs.w_stride;
        const int w_plane_stride = vel_ptrs.w_plane_stride;
        const int div_plane_stride = v.cell_plane_stride;
        const double* w_ptr = vel_ptrs.w;

        const int n_cells = Nx * Ny * Nz;
        if (use_O4) {
            // O4 divergence with periodic-aware fallback
            #pragma omp target teams distribute parallel for \
                map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], w_ptr[0:w_total_size], div_ptr[0:div_total_size]) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, div_stride, div_plane_stride, Nx, Ny, Nz, Ng, x_periodic, y_periodic, z_periodic)
            for (int idx = 0; idx < n_cells; ++idx) {
                const int i = idx % Nx + Ng;
                const int j = (idx / Nx) % Ny + Ng;
                const int k = idx / (Nx * Ny) + Ng;

                divergence_cell_kernel_staggered_O4_3d(i, j, k, Ng, Nx, Ny, Nz,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, div_stride, div_plane_stride,
                    dx, dy, dz, x_periodic, y_periodic, z_periodic,
                    u_ptr, v_ptr, w_ptr, div_ptr);
            }
        } else {
            // O2 divergence
            #pragma omp target teams distribute parallel for \
                map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], w_ptr[0:w_total_size], div_ptr[0:div_total_size]) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, div_stride, div_plane_stride, Nx, Ny, Ng)
            for (int idx = 0; idx < n_cells; ++idx) {
                const int i = idx % Nx + Ng;
                const int j = (idx / Nx) % Ny + Ng;
                const int k = idx / (Nx * Ny) + Ng;

                divergence_cell_kernel_staggered_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, div_stride, div_plane_stride,
                    dx, dy, dz, u_ptr, v_ptr, w_ptr, div_ptr);
            }
        }
        return;
    }

    // 2D path
    const int n_cells = Nx * Ny;

    // Use target data for scalar parameters (NVHPC workaround)
    #pragma omp target data map(to: dx, dy, u_stride, v_stride, div_stride, Nx, Ng)
    {
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], div_ptr[0:div_total_size])
        for (int idx = 0; idx < n_cells; ++idx) {
            const int i = idx % Nx + Ng;  // Cell center i index (with ghosts)
            const int j = idx / Nx + Ng;  // Cell center j index (with ghosts)

            // Fully inlined divergence computation
            const int u_right = j * u_stride + (i + 1);
            const int u_left = j * u_stride + i;
            const int v_top = (j + 1) * v_stride + i;
            const int v_bottom = j * v_stride + i;
            const int div_idx = j * div_stride + i;

            const double dudx = (u_ptr[u_right] - u_ptr[u_left]) / dx;
            const double dvdy = (v_ptr[v_top] - v_ptr[v_bottom]) / dy;
            div_ptr[div_idx] = dudx + dvdy;
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
    NVTX_PUSH("correct_velocity");

    // Get unified view (device pointers in GPU build, host pointers in CPU build)
    auto v = get_solver_view();

    const double dx = v.dx;
    const double dy = v.dy;
    const double dt = v.dt;
    const int Nx = v.Nx;
    const int Ny = v.Ny;
    const int Nz = v.Nz;
    const int Ng = v.Ng;
    const int u_stride = v.u_stride;
    const int v_stride = v.v_stride;
    const int p_stride = v.cell_stride;

    // O4 spatial discretization for pressure gradient (Dcf_O4)
    const bool use_O4 = (config_.space_order == 4);

    const bool x_periodic = (velocity_bc_.x_lo == VelocityBC::Periodic) &&
                            (velocity_bc_.x_hi == VelocityBC::Periodic);
    const bool y_periodic = (velocity_bc_.y_lo == VelocityBC::Periodic) &&
                            (velocity_bc_.y_hi == VelocityBC::Periodic);
    const bool z_periodic = (velocity_bc_.z_lo == VelocityBC::Periodic) &&
                            (velocity_bc_.z_hi == VelocityBC::Periodic);

    [[maybe_unused]] const size_t u_total_size = velocity_.u_total_size();
    [[maybe_unused]] const size_t v_total_size = velocity_.v_total_size();
    [[maybe_unused]] const size_t w_total_size = velocity_.w_total_size();
    [[maybe_unused]] const size_t p_total_size = field_total_size_;

    const double* u_star_ptr = v.u_star_face;
    const double* v_star_ptr = v.v_star_face;
    const double* p_corr_ptr = v.p_corr;
    double*       u_ptr      = v.u_face;
    double*       v_ptr      = v.v_face;
    double*       p_ptr      = v.p;

    // 3D path
    if (!mesh_->is2D()) {
        const double dz = v.dz;
        const int u_plane_stride = v.u_plane_stride;
        const int v_plane_stride = v.v_plane_stride;
        const int w_stride = v.w_stride;
        const int w_plane_stride = v.w_plane_stride;
        const int p_plane_stride = v.cell_plane_stride;
        const double* w_star_ptr = v.w_star_face;
        double*       w_ptr      = v.w_face;

        // Correct u-velocities at x-faces (3D)
        const int n_u_faces = (Nx + 1) * Ny * Nz;
        if (use_O4) {
            // O4 pressure gradient with periodic-aware fallback
            #pragma omp target teams distribute parallel for \
                map(present: u_ptr[0:u_total_size], u_star_ptr[0:u_total_size], p_corr_ptr[0:p_total_size]) \
                firstprivate(dx, dt, u_stride, u_plane_stride, p_stride, p_plane_stride, Nx, Ny, Ng, x_periodic)
            for (int idx = 0; idx < n_u_faces; ++idx) {
                int i = idx % (Nx + 1) + Ng;
                int j = (idx / (Nx + 1)) % Ny + Ng;
                int k = idx / ((Nx + 1) * Ny) + Ng;

                correct_u_face_kernel_staggered_O4_3d(i, j, k, Ng, Nx,
                    u_stride, u_plane_stride, p_stride, p_plane_stride,
                    dx, dt, x_periodic, u_star_ptr, p_corr_ptr, u_ptr);
            }
        } else {
            // O2 pressure gradient
            #pragma omp target teams distribute parallel for \
                map(present: u_ptr[0:u_total_size], u_star_ptr[0:u_total_size], p_corr_ptr[0:p_total_size]) \
                firstprivate(dx, dt, u_stride, u_plane_stride, p_stride, p_plane_stride, Nx, Ny, Ng)
            for (int idx = 0; idx < n_u_faces; ++idx) {
                int i = idx % (Nx + 1) + Ng;
                int j = (idx / (Nx + 1)) % Ny + Ng;
                int k = idx / ((Nx + 1) * Ny) + Ng;

                correct_u_face_kernel_staggered_3d(i, j, k,
                    u_stride, u_plane_stride, p_stride, p_plane_stride,
                    dx, dt, u_star_ptr, p_corr_ptr, u_ptr);
            }
        }

        // Enforce x-periodicity (3D)
        if (x_periodic) {
            const int n_u_periodic = Ny * Nz;
            #pragma omp target teams distribute parallel for \
                map(present: u_ptr[0:u_total_size]) \
                firstprivate(u_stride, u_plane_stride, Nx, Ny, Ng)
            for (int idx = 0; idx < n_u_periodic; ++idx) {
                int j = idx % Ny + Ng;
                int k = idx / Ny + Ng;
                int i_left = Ng;
                int i_right = Ng + Nx;
                int idx_left = k * u_plane_stride + j * u_stride + i_left;
                int idx_right = k * u_plane_stride + j * u_stride + i_right;
                double u_avg = 0.5 * (u_ptr[idx_left] + u_ptr[idx_right]);
                u_ptr[idx_left] = u_avg;
                u_ptr[idx_right] = u_avg;
            }
        }

        // Correct v-velocities at y-faces (3D)
        const int n_v_faces = Nx * (Ny + 1) * Nz;
        if (use_O4) {
            // O4 pressure gradient with periodic-aware fallback
            #pragma omp target teams distribute parallel for \
                map(present: v_ptr[0:v_total_size], v_star_ptr[0:v_total_size], p_corr_ptr[0:p_total_size]) \
                firstprivate(dy, dt, v_stride, v_plane_stride, p_stride, p_plane_stride, Nx, Ny, Ng, y_periodic)
            for (int idx = 0; idx < n_v_faces; ++idx) {
                int i = idx % Nx + Ng;
                int j = (idx / Nx) % (Ny + 1) + Ng;
                int k = idx / (Nx * (Ny + 1)) + Ng;

                correct_v_face_kernel_staggered_O4_3d(i, j, k, Ng, Ny,
                    v_stride, v_plane_stride, p_stride, p_plane_stride,
                    dy, dt, y_periodic, v_star_ptr, p_corr_ptr, v_ptr);
            }
        } else {
            // O2 pressure gradient
            #pragma omp target teams distribute parallel for \
                map(present: v_ptr[0:v_total_size], v_star_ptr[0:v_total_size], p_corr_ptr[0:p_total_size]) \
                firstprivate(dy, dt, v_stride, v_plane_stride, p_stride, p_plane_stride, Nx, Ny, Ng)
            for (int idx = 0; idx < n_v_faces; ++idx) {
                int i = idx % Nx + Ng;
                int j = (idx / Nx) % (Ny + 1) + Ng;
                int k = idx / (Nx * (Ny + 1)) + Ng;

                correct_v_face_kernel_staggered_3d(i, j, k,
                    v_stride, v_plane_stride, p_stride, p_plane_stride,
                    dy, dt, v_star_ptr, p_corr_ptr, v_ptr);
            }
        }

        // Enforce y-periodicity (3D)
        if (y_periodic) {
            const int n_v_periodic = Nx * Nz;
            #pragma omp target teams distribute parallel for \
                map(present: v_ptr[0:v_total_size]) \
                firstprivate(v_stride, v_plane_stride, Nx, Ny, Ng)
            for (int idx = 0; idx < n_v_periodic; ++idx) {
                int i = idx % Nx + Ng;
                int k = idx / Nx + Ng;
                int j_bottom = Ng;
                int j_top = Ng + Ny;
                int idx_bottom = k * v_plane_stride + j_bottom * v_stride + i;
                int idx_top = k * v_plane_stride + j_top * v_stride + i;
                double v_avg = 0.5 * (v_ptr[idx_bottom] + v_ptr[idx_top]);
                v_ptr[idx_bottom] = v_avg;
                v_ptr[idx_top] = v_avg;
            }
        }

        // Correct w-velocities at z-faces (3D)
        const int n_w_faces = Nx * Ny * (Nz + 1);
        if (use_O4) {
            // O4 pressure gradient with periodic-aware fallback
            #pragma omp target teams distribute parallel for \
                map(present: w_ptr[0:w_total_size], w_star_ptr[0:w_total_size], p_corr_ptr[0:p_total_size]) \
                firstprivate(dz, dt, w_stride, w_plane_stride, p_stride, p_plane_stride, Nx, Ny, Nz, Ng, z_periodic)
            for (int idx = 0; idx < n_w_faces; ++idx) {
                int i = idx % Nx + Ng;
                int j = (idx / Nx) % Ny + Ng;
                int k = idx / (Nx * Ny) + Ng;

                correct_w_face_kernel_staggered_O4_3d(i, j, k, Ng, Nz,
                    w_stride, w_plane_stride, p_stride, p_plane_stride,
                    dz, dt, z_periodic, w_star_ptr, p_corr_ptr, w_ptr);
            }
        } else {
            // O2 pressure gradient
            #pragma omp target teams distribute parallel for \
                map(present: w_ptr[0:w_total_size], w_star_ptr[0:w_total_size], p_corr_ptr[0:p_total_size]) \
                firstprivate(dz, dt, w_stride, w_plane_stride, p_stride, p_plane_stride, Nx, Ny, Ng)
            for (int idx = 0; idx < n_w_faces; ++idx) {
                int i = idx % Nx + Ng;
                int j = (idx / Nx) % Ny + Ng;
                int k = idx / (Nx * Ny) + Ng;

                correct_w_face_kernel_staggered_3d(i, j, k,
                    w_stride, w_plane_stride, p_stride, p_plane_stride,
                    dz, dt, w_star_ptr, p_corr_ptr, w_ptr);
            }
        }

        // Enforce z-periodicity (3D)
        if (z_periodic) {
            const int n_w_periodic = Nx * Ny;
            #pragma omp target teams distribute parallel for \
                map(present: w_ptr[0:w_total_size]) \
                firstprivate(w_stride, w_plane_stride, Nx, Nz, Ng)
            for (int idx = 0; idx < n_w_periodic; ++idx) {
                int i = idx % Nx + Ng;
                int j = idx / Nx + Ng;
                int k_front = Ng;
                int k_back = Ng + Nz;
                int idx_front = k_front * w_plane_stride + j * w_stride + i;
                int idx_back = k_back * w_plane_stride + j * w_stride + i;
                double w_avg = 0.5 * (w_ptr[idx_front] + w_ptr[idx_back]);
                w_ptr[idx_front] = w_avg;
                w_ptr[idx_back] = w_avg;
            }
        }

        // Update pressure at cell centers (3D)
        const int n_cells = Nx * Ny * Nz;
        #pragma omp target teams distribute parallel for \
            map(present: p_ptr[0:p_total_size], p_corr_ptr[0:p_total_size]) \
            firstprivate(p_stride, p_plane_stride, Nx, Ny, Ng)
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % Ny + Ng;
            int k = idx / (Nx * Ny) + Ng;

            int p_idx = k * p_plane_stride + j * p_stride + i;
            p_ptr[p_idx] += p_corr_ptr[p_idx];
        }

        NVTX_POP();
        return;
    }

    // 2D path
    const int n_cells = Nx * Ny;

    // Correct ALL u-velocities at x-faces (including redundant face if periodic)
    const int n_u_faces = (Nx + 1) * Ny;
    #pragma omp target teams distribute parallel for \
        map(present: u_ptr[0:u_total_size], u_star_ptr[0:u_total_size], p_corr_ptr[0:p_total_size]) \
        firstprivate(dx, dt, u_stride, p_stride, Nx, Ng)
    for (int idx = 0; idx < n_u_faces; ++idx) {
        int i_local = idx % (Nx + 1);
        int j_local = idx / (Nx + 1);
        int i = i_local + Ng;
        int j = j_local + Ng;

        correct_u_face_kernel_staggered(i, j, u_stride, p_stride, dx, dt,
                                       u_star_ptr, p_corr_ptr, u_ptr);
    }

    // Enforce exact x-periodicity: average the left and right edge values
    if (x_periodic) {
        const int n_u_periodic = Ny;
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_total_size]) \
            firstprivate(u_stride, Nx, Ng)
        for (int j_local = 0; j_local < n_u_periodic; ++j_local) {
            int j = j_local + Ng;
            int i_left = Ng;
            int i_right = Ng + Nx;
            double u_avg = 0.5 * (u_ptr[j * u_stride + i_left] + u_ptr[j * u_stride + i_right]);
            u_ptr[j * u_stride + i_left] = u_avg;
            u_ptr[j * u_stride + i_right] = u_avg;
        }
    }

    // Correct ALL v-velocities at y-faces (including redundant face if periodic)
    const int n_v_faces = Nx * (Ny + 1);
    #pragma omp target teams distribute parallel for \
        map(present: v_ptr[0:v_total_size], v_star_ptr[0:v_total_size], p_corr_ptr[0:p_total_size]) \
        firstprivate(dy, dt, v_stride, p_stride, Nx, Ng)
    for (int idx = 0; idx < n_v_faces; ++idx) {
        int i_local = idx % Nx;
        int j_local = idx / Nx;
        int i = i_local + Ng;
        int j = j_local + Ng;

        correct_v_face_kernel_staggered(i, j, v_stride, p_stride, dy, dt,
                                       v_star_ptr, p_corr_ptr, v_ptr);
    }

    // Enforce exact y-periodicity: average the bottom and top edge values
    if (y_periodic) {
        const int n_v_periodic = Nx;
        #pragma omp target teams distribute parallel for \
            map(present: v_ptr[0:v_total_size]) \
            firstprivate(v_stride, Ny, Ng)
        for (int i_local = 0; i_local < n_v_periodic; ++i_local) {
            int i = i_local + Ng;
            int j_bottom = Ng;
            int j_top = Ng + Ny;
            double v_avg = 0.5 * (v_ptr[j_bottom * v_stride + i] + v_ptr[j_top * v_stride + i]);
            v_ptr[j_bottom * v_stride + i] = v_avg;
            v_ptr[j_top * v_stride + i] = v_avg;
        }
    }

    // Update pressure at cell centers
    #pragma omp target teams distribute parallel for \
        map(present: p_ptr[0:p_total_size], p_corr_ptr[0:p_total_size]) \
        firstprivate(p_stride, Nx)
    for (int idx = 0; idx < n_cells; ++idx) {
        int i = idx % Nx + Ng;
        int j = idx / Nx + Ng;

        update_pressure_kernel(i, j, p_stride, p_corr_ptr, p_ptr);
    }

    NVTX_POP();  // End correct_velocity
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
    NVTX_SCOPE_SOLVER("time_step");

    // Store old velocity for convergence check (at face locations for staggered grid)
    const int Ng = mesh_->Nghost;
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    
// Unified CPU/GPU path: copy current velocity to velocity_old using raw pointers
    {
    NVTX_PUSH("velocity_copy");
    [[maybe_unused]] const size_t u_total_size = velocity_.u_total_size();
    [[maybe_unused]] const size_t v_total_size = velocity_.v_total_size();
    const int u_stride = Nx + 2 * Ng + 1;
    const int v_stride = Nx + 2 * Ng;

    if (mesh_->is2D()) {
        time_kernels::copy_2d_uv(velocity_u_ptr_, velocity_old_u_ptr_,
                                 velocity_v_ptr_, velocity_old_v_ptr_,
                                 Nx, Ny, Ng, u_stride, v_stride,
                                 u_total_size, v_total_size);
    } else {
        const int Nz = mesh_->Nz;
        const int u_plane = u_stride * (Ny + 2 * Ng);
        const int v_plane = v_stride * (Ny + 2 * Ng + 1);
        const int w_stride = Nx + 2 * Ng;
        const int w_plane = w_stride * (Ny + 2 * Ng);
        time_kernels::copy_3d_uvw(velocity_u_ptr_, velocity_old_u_ptr_,
                                  velocity_v_ptr_, velocity_old_v_ptr_,
                                  velocity_w_ptr_, velocity_old_w_ptr_,
                                  Nx, Ny, Nz, Ng,
                                  u_stride, v_stride, w_stride,
                                  u_plane, v_plane, w_plane,
                                  u_total_size, v_total_size, velocity_.w_total_size());
    }
    NVTX_POP();
    }
    
    // 1a. Advance turbulence transport equations (if model uses them)
    if (turb_model_ && turb_model_->uses_transport_equations()) {
        TIMED_SCOPE("turbulence_transport");
        NVTX_PUSH("turbulence_transport");
        
        // Get device view for GPU-accelerated transport
        const TurbulenceDeviceView* device_view_ptr = nullptr;
#ifdef USE_GPU_OFFLOAD
        TurbulenceDeviceView device_view = get_device_view();
        if (device_view.is_valid()) {
            device_view_ptr = &device_view;
        }
#endif
        
        turb_model_->advance_turbulence(
            *mesh_,
            velocity_,
            current_dt_,
            k_,          // Updated in-place
            omega_,      // Updated in-place
            nu_t_,       // Previous step's nu_t for diffusion coefficients
            device_view_ptr
        );
        NVTX_POP();
        
#ifdef USE_GPU_OFFLOAD
        // CRITICAL FIX: Sync k and omega to GPU after transport equation update
        // ONLY if model didn't use GPU path (models operating on device_view don't need this)
        if (!turb_model_->is_gpu_ready()) {
            #pragma omp target update to(k_ptr_[0:field_total_size_])
            #pragma omp target update to(omega_ptr_[0:field_total_size_])
        }
#endif
    }
    
    // 1b. Update turbulence model (compute nu_t and optional tau_ij)
    if (turb_model_) {
        TIMED_SCOPE("turbulence_update");
        NVTX_PUSH("turbulence_update");
        
        // PHASE 1 GPU OPTIMIZATION: Pass device view if GPU is ready
        const TurbulenceDeviceView* device_view_ptr = nullptr;
#ifdef USE_GPU_OFFLOAD
        TurbulenceDeviceView device_view = get_device_view();
        if (device_view.is_valid()) {
            device_view_ptr = &device_view;
        }
        
        // GPU simulation: enforce device_view validity (host fallback forbidden)
        if (gpu_ready_ && (!device_view_ptr || !device_view_ptr->is_valid())) {
            throw std::runtime_error("GPU simulation requires valid TurbulenceDeviceView - host fallback forbidden");
        }
#endif
        
        turb_model_->update(*mesh_, velocity_, k_, omega_, nu_t_, 
                           turb_model_->provides_reynolds_stresses() ? &tau_ij_ : nullptr,
                           device_view_ptr);
        NVTX_POP();
        
        // CRITICAL FIX: Only sync nu_t to GPU if the model didn't use GPU path
        // Models that use device_view write directly to GPU nu_t and should NOT be overwritten
        // Models that work on CPU (like NN-MLP) write to host nu_t and MUST be synced to GPU
#ifdef USE_GPU_OFFLOAD
        // If device_view was valid and model is GPU-ready, nu_t is already on device
        // Otherwise (CPU path), sync host nu_t to device
        bool model_used_gpu = (device_view_ptr && device_view_ptr->is_valid() && turb_model_->is_gpu_ready());
        if (!model_used_gpu) {
        #pragma omp target update to(nu_t_ptr_[0:field_total_size_])
        }
#endif
    }
    
    // Effective viscosity: nu_eff_ = nu + nu_t (use persistent field)
    // GPU path: compute directly on GPU without CPU fill
#ifdef USE_GPU_OFFLOAD
    if (gpu_ready_) {
        NVTX_PUSH("nu_eff_computation");
        const int Nx = mesh_->Nx;
        const int Ny = mesh_->Ny;
        const int Nz = mesh_->Nz;
        const int Ng = mesh_->Nghost;
        const int stride = Nx + 2 * Ng;
        const int plane_stride = stride * (Ny + 2 * Ng);
        const size_t total_size = field_total_size_;
        const double nu = config_.nu;
        double* nu_eff_ptr = nu_eff_ptr_;
        const double* nu_t_ptr = nu_t_ptr_;
        const bool is_2d = mesh_->is2D();

        if (is_2d) {
            // 2D path
            const int n_cells = Nx * Ny;
            if (turb_model_) {
                #pragma omp target teams distribute parallel for \
                    map(present: nu_eff_ptr[0:total_size]) \
                    map(present: nu_t_ptr[0:total_size]) \
                    firstprivate(nu, stride, Nx, Ng)
                for (int idx = 0; idx < n_cells; ++idx) {
                    int i = idx % Nx + Ng;
                    int j = idx / Nx + Ng;
                    int cell_idx = j * stride + i;
                    nu_eff_ptr[cell_idx] = nu + nu_t_ptr[cell_idx];
                }
            } else {
                #pragma omp target teams distribute parallel for \
                    map(present: nu_eff_ptr[0:total_size]) \
                    firstprivate(nu, stride, Nx, Ng)
                for (int idx = 0; idx < n_cells; ++idx) {
                    int i = idx % Nx + Ng;
                    int j = idx / Nx + Ng;
                    int cell_idx = j * stride + i;
                    nu_eff_ptr[cell_idx] = nu;
                }
            }
        } else {
            // 3D path
            const int n_cells = Nx * Ny * Nz;
            if (turb_model_) {
                #pragma omp target teams distribute parallel for \
                    map(present: nu_eff_ptr[0:total_size]) \
                    map(present: nu_t_ptr[0:total_size]) \
                    firstprivate(nu, stride, plane_stride, Nx, Ny, Ng)
                for (int idx = 0; idx < n_cells; ++idx) {
                    int i = idx % Nx + Ng;
                    int j = (idx / Nx) % Ny + Ng;
                    int k = idx / (Nx * Ny) + Ng;
                    int cell_idx = k * plane_stride + j * stride + i;
                    nu_eff_ptr[cell_idx] = nu + nu_t_ptr[cell_idx];
                }
            } else {
                #pragma omp target teams distribute parallel for \
                    map(present: nu_eff_ptr[0:total_size]) \
                    firstprivate(nu, stride, plane_stride, Nx, Ny, Ng)
                for (int idx = 0; idx < n_cells; ++idx) {
                    int i = idx % Nx + Ng;
                    int j = (idx / Nx) % Ny + Ng;
                    int k = idx / (Nx * Ny) + Ng;
                    int cell_idx = k * plane_stride + j * stride + i;
                    nu_eff_ptr[cell_idx] = nu;
                }
            }
        }
        NVTX_POP();
    } else
#endif
    {
        // CPU path
        nu_eff_.fill(config_.nu);
        if (turb_model_) {
            if (mesh_->is2D()) {
                for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                    for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                        nu_eff_(i, j) = config_.nu + nu_t_(i, j);
                    }
                }
            } else {
                for (int k = mesh_->k_begin(); k < mesh_->k_end(); ++k) {
                    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                            nu_eff_(i, j, k) = config_.nu + nu_t_(i, j, k);
                        }
                    }
                }
            }
        }
    }

    // Dispatch to appropriate time integrator
    // RK methods handle their own convective/diffusive terms and projection
    if (config_.time_integrator != TimeIntegrator::Euler) {
        if (config_.time_integrator == TimeIntegrator::RK3) {
            ssprk3_step(current_dt_);
        } else if (config_.time_integrator == TimeIntegrator::RK2) {
            ssprk2_step(current_dt_);
        }

        // Compute residual for RK methods
        // Note: Post-step divergence check and NaN guard are still done below
        // Fall through to residual computation
    } else {
    // =========== Euler time integration path (default) ===========
    // 2. Compute convective and diffusive terms (use persistent fields)
    {
        TIMED_SCOPE("convective_term");
        NVTX_PUSH("convection");
        compute_convective_term(velocity_, conv_);
        NVTX_POP();

        // Convective KE production diagnostic: <u, conv(u)>
        // For skew-symmetric form, this should be ~0 (energy conservative)
        // NOTE: CPU-only diagnostic - GPU builds skip this (would need expensive D→H sync)
#ifndef USE_GPU_OFFLOAD
        static bool conv_ke_diagnostics = (std::getenv("NNCFD_CONV_KE_DIAGNOSTICS") != nullptr);
        if (conv_ke_diagnostics && (iter_ % 100 == 0)) {
            double dke_conv = compute_convective_ke_production();
            double ke = 0.0;
            const double dV = mesh_->dx * mesh_->dy * (mesh_->is2D() ? 1.0 : mesh_->dz);
            if (mesh_->is2D()) {
                for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                    for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                        double u = 0.5 * (velocity_.u(i, j) + velocity_.u(i+1, j));
                        double v = 0.5 * (velocity_.v(i, j) + velocity_.v(i, j+1));
                        ke += 0.5 * (u*u + v*v) * dV;
                    }
                }
            } else {
                for (int k = mesh_->k_begin(); k < mesh_->k_end(); ++k) {
                    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                            double u = 0.5 * (velocity_.u(i, j, k) + velocity_.u(i+1, j, k));
                            double v = 0.5 * (velocity_.v(i, j, k) + velocity_.v(i, j+1, k));
                            double w = 0.5 * (velocity_.w(i, j, k) + velocity_.w(i, j, k+1));
                            ke += 0.5 * (u*u + v*v + w*w) * dV;
                        }
                    }
                }
            }
            // Normalize by KE to get fractional rate
            double rel_rate = (ke > 1e-30) ? dke_conv / ke : 0.0;
            std::cout << "[Convection] dKE/dt_conv=" << std::scientific << std::setprecision(6)
                      << dke_conv << " (rel=" << rel_rate << "/s)\n";
        }
#endif  // !USE_GPU_OFFLOAD
    }

    {
        TIMED_SCOPE("diffusive_term");
        NVTX_PUSH("diffusion");
        compute_diffusive_term(velocity_, nu_eff_, diff_);
        NVTX_POP();
    }
    
    // 3. Compute provisional velocity u* (without pressure gradient) at face locations
    // u* = u^n + dt * (-conv + diff + body_force)
    NVTX_PUSH("predictor_step");
    
    // Get unified view (reuse Nx, Ny, Ng from function scope)
    auto v = get_solver_view();
    
    const int u_stride_pred = v.u_stride;
    const int v_stride_pred = v.v_stride;
    const double dt = v.dt;
    const double fx = fx_;
    const double fy = fy_;
    
    const bool x_periodic = (velocity_bc_.x_lo == VelocityBC::Periodic) && 
                            (velocity_bc_.x_hi == VelocityBC::Periodic);
    const bool y_periodic = (velocity_bc_.y_lo == VelocityBC::Periodic) && 
                            (velocity_bc_.y_hi == VelocityBC::Periodic);
    
    [[maybe_unused]] const size_t u_total_size_pred = velocity_.u_total_size();
    [[maybe_unused]] const size_t v_total_size_pred = velocity_.v_total_size();
    
    const double* u_ptr = v.u_face;
    const double* v_ptr = v.v_face;
    double* u_star_ptr = v.u_star_face;
    double* v_star_ptr = v.v_star_face;
    const double* conv_u_ptr = v.conv_u;
    const double* conv_v_ptr = v.conv_v;
    const double* diff_u_ptr = v.diff_u;
    const double* diff_v_ptr = v.diff_v;

    const bool is_2d_pred = mesh_->is2D();
    const int Nz_pred = mesh_->Nz;
    const int Nz_eff_pred = is_2d_pred ? 1 : Nz_pred;
    // Avoid reading uninitialized strides in 2D mode (set to 0 if 2D)
    const int u_plane_stride_pred = is_2d_pred ? 0 : v.u_plane_stride;
    const int v_plane_stride_pred = is_2d_pred ? 0 : v.v_plane_stride;

    // Compute u* at ALL x-faces (including redundant if periodic)
    const int n_u_faces_pred = (Nx + 1) * Ny * Nz_eff_pred;
    #pragma omp target teams distribute parallel for \
        map(present: u_ptr[0:u_total_size_pred], u_star_ptr[0:u_total_size_pred], \
                    conv_u_ptr[0:u_total_size_pred], diff_u_ptr[0:u_total_size_pred]) \
        firstprivate(dt, fx, u_stride_pred, u_plane_stride_pred, Nx, Ny, Nz_eff_pred, Ng, is_2d_pred)
    for (int idx = 0; idx < n_u_faces_pred; ++idx) {
        int i_local = idx % (Nx + 1);
        int j_local = (idx / (Nx + 1)) % Ny;
        int k_local = idx / ((Nx + 1) * Ny);
        int i = i_local + Ng;
        int j = j_local + Ng;
        int k = k_local + Ng;
        int u_idx = is_2d_pred ? (j * u_stride_pred + i)
                               : (k * u_plane_stride_pred + j * u_stride_pred + i);

        u_star_ptr[u_idx] = u_ptr[u_idx] + dt * (-conv_u_ptr[u_idx] + diff_u_ptr[u_idx] + fx);
    }

    // Enforce exact x-periodicity for u*: average left and right edges
    if (x_periodic) {
        const int n_u_periodic = Ny * Nz_eff_pred;
        #pragma omp target teams distribute parallel for \
            map(present: u_star_ptr[0:u_total_size_pred]) \
            firstprivate(u_stride_pred, u_plane_stride_pred, Nx, Ny, Nz_eff_pred, Ng, is_2d_pred)
        for (int idx = 0; idx < n_u_periodic; ++idx) {
            int j_local = idx % Ny;
            int k_local = idx / Ny;
            int j = j_local + Ng;
            int k = k_local + Ng;
            int base = is_2d_pred ? (j * u_stride_pred)
                                  : (k * u_plane_stride_pred + j * u_stride_pred);
            double u_avg = 0.5 * (u_star_ptr[base + Ng] + u_star_ptr[base + (Ng + Nx)]);
            u_star_ptr[base + Ng] = u_avg;
            u_star_ptr[base + (Ng + Nx)] = u_avg;
        }
    }

    // Compute v* at ALL y-faces (including redundant if periodic)
    const int n_v_faces_pred = Nx * (Ny + 1) * Nz_eff_pred;
    #pragma omp target teams distribute parallel for \
        map(present: v_ptr[0:v_total_size_pred], v_star_ptr[0:v_total_size_pred], \
                    conv_v_ptr[0:v_total_size_pred], diff_v_ptr[0:v_total_size_pred]) \
        firstprivate(dt, fy, v_stride_pred, v_plane_stride_pred, Nx, Ny, Nz_eff_pred, Ng, is_2d_pred)
    for (int idx = 0; idx < n_v_faces_pred; ++idx) {
        int i_local = idx % Nx;
        int j_local = (idx / Nx) % (Ny + 1);
        int k_local = idx / (Nx * (Ny + 1));
        int i = i_local + Ng;
        int j = j_local + Ng;
        int k = k_local + Ng;
        int v_idx = is_2d_pred ? (j * v_stride_pred + i)
                               : (k * v_plane_stride_pred + j * v_stride_pred + i);

        v_star_ptr[v_idx] = v_ptr[v_idx] + dt * (-conv_v_ptr[v_idx] + diff_v_ptr[v_idx] + fy);
    }

    // Enforce exact y-periodicity for v*: average bottom and top edges
    if (y_periodic) {
        const int n_v_periodic = Nx * Nz_eff_pred;
        #pragma omp target teams distribute parallel for \
            map(present: v_star_ptr[0:v_total_size_pred]) \
            firstprivate(v_stride_pred, v_plane_stride_pred, Nx, Ny, Nz_eff_pred, Ng, is_2d_pred)
        for (int idx = 0; idx < n_v_periodic; ++idx) {
            int i_local = idx % Nx;
            int k_local = idx / Nx;
            int i = i_local + Ng;
            int k = k_local + Ng;
            int base_lo = is_2d_pred ? (Ng * v_stride_pred + i)
                                     : (k * v_plane_stride_pred + Ng * v_stride_pred + i);
            int base_hi = is_2d_pred ? ((Ng + Ny) * v_stride_pred + i)
                                     : (k * v_plane_stride_pred + (Ng + Ny) * v_stride_pred + i);
            double v_avg = 0.5 * (v_star_ptr[base_lo] + v_star_ptr[base_hi]);
            v_star_ptr[base_lo] = v_avg;
            v_star_ptr[base_hi] = v_avg;
        }
    }

    // 3D: Compute w* at ALL z-faces
    if (!mesh_->is2D()) {
        const int Nz = mesh_->Nz;
        const int w_stride_pred = v.w_stride;
        const int w_plane_stride_pred = v.w_plane_stride;
        const double fz = fz_;
        [[maybe_unused]] const size_t w_total_size_pred = velocity_.w_total_size();

        const double* w_ptr = v.w_face;
        double* w_star_ptr = v.w_star_face;
        const double* conv_w_ptr = v.conv_w;
        const double* diff_w_ptr = v.diff_w;

        const bool z_periodic = (velocity_bc_.z_lo == VelocityBC::Periodic) &&
                                (velocity_bc_.z_hi == VelocityBC::Periodic);

        // Compute w* = w + dt * (-conv_w + diff_w + fz)
        const int n_w_faces_pred = Nx * Ny * (Nz + 1);
        #pragma omp target teams distribute parallel for \
            map(present: w_ptr[0:w_total_size_pred], w_star_ptr[0:w_total_size_pred], \
                        conv_w_ptr[0:w_total_size_pred], diff_w_ptr[0:w_total_size_pred]) \
            firstprivate(dt, fz, w_stride_pred, w_plane_stride_pred, Nx, Ny, Ng)
        for (int idx = 0; idx < n_w_faces_pred; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % Ny + Ng;
            int k = idx / (Nx * Ny) + Ng;
            int w_idx = k * w_plane_stride_pred + j * w_stride_pred + i;

            w_star_ptr[w_idx] = w_ptr[w_idx] + dt * (-conv_w_ptr[w_idx] + diff_w_ptr[w_idx] + fz);
        }

        // Enforce exact z-periodicity for w*: average front and back edges
        if (z_periodic) {
            const int n_w_periodic = Nx * Ny;
            #pragma omp target teams distribute parallel for \
                map(present: w_star_ptr[0:w_total_size_pred]) \
                firstprivate(w_stride_pred, w_plane_stride_pred, Nx, Nz, Ng)
            for (int idx = 0; idx < n_w_periodic; ++idx) {
                int i = idx % Nx + Ng;
                int j = idx / Nx + Ng;
                int idx_back = Ng * w_plane_stride_pred + j * w_stride_pred + i;
                int idx_front = (Ng + Nz) * w_plane_stride_pred + j * w_stride_pred + i;
                double w_avg = 0.5 * (w_star_ptr[idx_back] + w_star_ptr[idx_front]);
                w_star_ptr[idx_back] = w_avg;
                w_star_ptr[idx_front] = w_avg;
            }
        }
    }

    // Apply BCs to provisional velocity (needed for divergence calculation)
    // Temporarily swap velocity_ and velocity_star_ to use apply_velocity_bc
    std::swap(velocity_, velocity_star_);
#ifdef USE_GPU_OFFLOAD
    // CRITICAL: std::swap invalidates GPU pointers - they still point to old memory
    // After swap, velocity_u_ptr_ points to what is now velocity_star_ data!
    // Must swap the pointers too to keep them consistent
    std::swap(velocity_u_ptr_, velocity_star_u_ptr_);
    std::swap(velocity_v_ptr_, velocity_star_v_ptr_);
    if (!mesh_->is2D()) {
        std::swap(velocity_w_ptr_, velocity_star_w_ptr_);
    }
#endif

    // PHASE 1.5 OPTIMIZATION: Skip redundant BC call for fully periodic domains
    // The inline periodic averaging above already handles periodic BCs correctly
    // Only apply BCs if domain has non-periodic boundaries (which need ghost cell updates)
    const bool z_periodic_check = mesh_->is2D() ||
                                  ((velocity_bc_.z_lo == VelocityBC::Periodic) &&
                                   (velocity_bc_.z_hi == VelocityBC::Periodic));
    const bool needs_bc_update = !x_periodic || !y_periodic || !z_periodic_check;
    if (needs_bc_update) {
        apply_velocity_bc();
    }

    std::swap(velocity_, velocity_star_);
#ifdef USE_GPU_OFFLOAD
    // Swap pointers back to restore original mapping
    std::swap(velocity_u_ptr_, velocity_star_u_ptr_);
    std::swap(velocity_v_ptr_, velocity_star_v_ptr_);
    if (!mesh_->is2D()) {
        std::swap(velocity_w_ptr_, velocity_star_w_ptr_);
    }
#endif
    NVTX_POP();  // End predictor_step
    
    // 4. Solve pressure Poisson equation
    // nabla^2p' = (1/dt) nabla*u*
    {
        TIMED_SCOPE("divergence");
        NVTX_PUSH("divergence");
        compute_divergence(VelocityWhich::Star, div_velocity_);
        NVTX_POP();
    }

    // Build RHS on GPU and subtract mean divergence to ensure solvability
    // GPU-RESIDENT OPTIMIZATION: Keep all data on device, only transfer scalars
    NVTX_PUSH("poisson_rhs_build");
    double mean_div = 0.0;
    
#ifdef USE_GPU_OFFLOAD
    if (gpu_ready_) {
        // GPU-resident path: compute mean divergence on device via reduction
        const int Nx = mesh_->Nx;
        const int Ny = mesh_->Ny;
        const int Nz = mesh_->Nz;
        const int Ng = mesh_->Nghost;
        const int i_begin = mesh_->i_begin();
        const int j_begin = mesh_->j_begin();
        const int k_begin = mesh_->k_begin();
        const int stride = Nx + 2 * Ng;
        const int plane_stride = stride * (Ny + 2 * Ng);
        const bool is_2d = mesh_->is2D();

        // Local aliases to avoid implicit 'this' mapping (NVHPC workaround)
        double* div_ptr = div_velocity_ptr_;
        double* rhs_ptr = rhs_poisson_ptr_;
        double* p_corr_ptr = pressure_corr_ptr_;
        const size_t n_field = field_total_size_;

        double sum_div = 0.0;
        int count = is_2d ? (Nx * Ny) : (Nx * Ny * Nz);

        if (is_2d) {
            // 2D path
            #pragma omp target teams distribute parallel for \
                map(present: div_ptr[0:n_field]) \
                reduction(+:sum_div)
            for (int j = 0; j < Ny; ++j) {
                for (int i = 0; i < Nx; ++i) {
                    int ii = i + i_begin;
                    int jj = j + j_begin;
                    int idx = jj * stride + ii;
                    sum_div += div_ptr[idx];
                }
            }
        } else {
            // 3D path
            #pragma omp target teams distribute parallel for collapse(3) \
                map(present: div_ptr[0:n_field]) \
                reduction(+:sum_div)
            for (int k = 0; k < Nz; ++k) {
                for (int j = 0; j < Ny; ++j) {
                    for (int i = 0; i < Nx; ++i) {
                        int ii = i + i_begin;
                        int jj = j + j_begin;
                        int kk = k + k_begin;
                        int idx = kk * plane_stride + jj * stride + ii;
                        sum_div += div_ptr[idx];
                    }
                }
            }
        }

        mean_div = (count > 0) ? sum_div / count : 0.0;

        // Build RHS on GPU: rhs = (div - mean_div) / dt
        const double dt_inv = 1.0 / current_dt_;

        if (is_2d) {
            // 2D path
            #pragma omp target teams distribute parallel for \
                map(present: div_ptr[0:n_field], rhs_ptr[0:n_field])
            for (int j = 0; j < Ny; ++j) {
                for (int i = 0; i < Nx; ++i) {
                    int ii = i + i_begin;
                    int jj = j + j_begin;
                    int idx = jj * stride + ii;
                    rhs_ptr[idx] = (div_ptr[idx] - mean_div) * dt_inv;
                }
            }
        } else {
            // 3D path
            #pragma omp target teams distribute parallel for collapse(3) \
                map(present: div_ptr[0:n_field], rhs_ptr[0:n_field])
            for (int k = 0; k < Nz; ++k) {
                for (int j = 0; j < Ny; ++j) {
                    for (int i = 0; i < Nx; ++i) {
                        int ii = i + i_begin;
                        int jj = j + j_begin;
                        int kk = k + k_begin;
                        int idx = kk * plane_stride + jj * stride + ii;
                        rhs_ptr[idx] = (div_ptr[idx] - mean_div) * dt_inv;
                    }
                }
            }
        }

        // OPTIMIZATION: Warm-start for Poisson solver (device-resident)
        // Zero pressure correction on device on first iteration only
        if (iter_ == 0) {
            #pragma omp target teams distribute parallel for \
                map(present: p_corr_ptr[0:n_field])
            for (size_t idx = 0; idx < n_field; ++idx) {
                p_corr_ptr[idx] = 0.0;
            }
        }
        // Otherwise, reuse previous solution (already on device, no action needed)

    } else
#endif
    {
        // Host path
        double sum_div = 0.0;
        int count = 0;

        if (mesh_->is2D()) {
            // 2D path
            for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                    double div = div_velocity_(i, j);
                    sum_div += div;
                    ++count;
                }
            }
            mean_div = (count > 0) ? sum_div / count : 0.0;

            // Use multiplication by inverse to match GPU arithmetic exactly
            const double dt_inv = 1.0 / current_dt_;
            for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                    rhs_poisson_(i, j) = (div_velocity_(i, j) - mean_div) * dt_inv;
                }
            }
        } else {
            // 3D path
            const int Nz = mesh_->Nz;
            const int Ng = mesh_->Nghost;
            for (int k = Ng; k < Ng + Nz; ++k) {
                for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                    for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                        double div = div_velocity_(i, j, k);
                        sum_div += div;
                        ++count;
                    }
                }
            }
            mean_div = (count > 0) ? sum_div / count : 0.0;

            // Use multiplication by inverse to match GPU arithmetic exactly
            const double dt_inv = 1.0 / current_dt_;
            for (int k = Ng; k < Ng + Nz; ++k) {
                for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                    for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                        rhs_poisson_(i, j, k) = (div_velocity_(i, j, k) - mean_div) * dt_inv;
                    }
                }
            }
        }

        // Warm-start: zero on first iteration
        if (iter_ == 0) {
            pressure_correction_.fill(0.0);
        }
    }
    NVTX_POP();  // End poisson_rhs_build

    // 4b. Solve Poisson equation for pressure correction
    {
        TIMED_SCOPE("poisson_solve");
        NVTX_PUSH("poisson_solve");
        
        // CRITICAL: Use relative tolerance for Poisson solver (standard multigrid practice)
        // When turbulence changes effective viscosity, RHS magnitude varies significantly
        // Absolute tolerance would be too strict for small RHS, too loose for large RHS
        double rhs_norm_sq = 0.0;
        int rhs_count = 0;

// Unified CPU/GPU path: compute RHS norm using raw pointers
        {
            const int Nx = mesh_->Nx;
            const int Ny = mesh_->Ny;
            const int Nz = mesh_->Nz;
            const int Ng = mesh_->Nghost;
            const int i_begin = mesh_->i_begin();
            const int j_begin = mesh_->j_begin();
            const int k_begin = mesh_->k_begin();
            const int stride = Nx + 2 * Ng;
            const int plane_stride = stride * (Ny + 2 * Ng);

            if (mesh_->is2D()) {
#ifdef USE_GPU_OFFLOAD
                #pragma omp target teams distribute parallel for \
                    map(present: rhs_poisson_ptr_[0:field_total_size_]) \
                    reduction(+:rhs_norm_sq, rhs_count)
#endif
                for (int j = 0; j < Ny; ++j) {
                    for (int i = 0; i < Nx; ++i) {
                        int ii = i + i_begin;
                        int jj = j + j_begin;
                        int idx = jj * stride + ii;
                        double rhs_val = rhs_poisson_ptr_[idx];
                        rhs_norm_sq += rhs_val * rhs_val;
                        rhs_count++;
                    }
                }
            } else {
#ifdef USE_GPU_OFFLOAD
                #pragma omp target teams distribute parallel for collapse(3) \
                    map(present: rhs_poisson_ptr_[0:field_total_size_]) \
                    reduction(+:rhs_norm_sq, rhs_count)
#endif
                for (int k = 0; k < Nz; ++k) {
                    for (int j = 0; j < Ny; ++j) {
                        for (int i = 0; i < Nx; ++i) {
                            int ii = i + i_begin;
                            int jj = j + j_begin;
                            int kk = k + k_begin;
                            int idx = kk * plane_stride + jj * stride + ii;
                            double rhs_val = rhs_poisson_ptr_[idx];
                            rhs_norm_sq += rhs_val * rhs_val;
                            rhs_count++;
                        }
                    }
                }
            }
        }
        
        double rhs_rms = std::sqrt(rhs_norm_sq / std::max(rhs_count, 1));
        
        // Configure Poisson solver with robust convergence criteria
        // The MG solver now supports three convergence criteria (any triggers exit):
        //   1. ||r||_∞ ≤ tol_abs  (absolute, usually disabled)
        //   2. ||r||/||b|| ≤ tol_rhs  (RHS-relative, recommended for projection)
        //   3. ||r||/||r0|| ≤ tol_rel  (initial-residual relative, backup)
        PoissonConfig pcfg;
        pcfg.max_vcycles = config_.poisson_max_vcycles;
        pcfg.omega = config_.poisson_omega;
        pcfg.verbose = false;  // Disable per-cycle output (too verbose)

        // New robust tolerance parameters (preferred for MG)
        pcfg.tol_abs = config_.poisson_tol_abs;
        pcfg.tol_rhs = config_.poisson_tol_rhs;
        pcfg.tol_rel = config_.poisson_tol_rel;
        pcfg.check_interval = config_.poisson_check_interval;
        pcfg.use_l2_norm = config_.poisson_use_l2_norm;
        pcfg.linf_safety_factor = config_.poisson_linf_safety;
        pcfg.fixed_cycles = config_.poisson_fixed_cycles;
        pcfg.adaptive_cycles = config_.poisson_adaptive_cycles;
        pcfg.check_after = config_.poisson_check_after;
        pcfg.nu1 = config_.poisson_nu1;
        pcfg.nu2 = config_.poisson_nu2;
        pcfg.chebyshev_degree = config_.poisson_chebyshev_degree;
        pcfg.use_vcycle_graph = config_.poisson_use_vcycle_graph;

        // Legacy tolerance for backward compatibility (non-MG solvers use this)
        double relative_tol = config_.poisson_tol * std::max(rhs_rms, 1e-12);
        double effective_tol = std::max(relative_tol, config_.poisson_abs_tol_floor);
        pcfg.tol = effective_tol;
        
        // Environment variable to enable detailed Poisson cycle diagnostics
        static bool poisson_diagnostics = (std::getenv("NNCFD_POISSON_DIAGNOSTICS") != nullptr);
        static int poisson_diagnostics_interval = []() {
            const char* env = std::getenv("NNCFD_POISSON_DIAGNOSTICS_INTERVAL");
            int v = env ? std::atoi(env) : 1;
            return (v > 0) ? v : 1;
        }();
        
        int cycles = 0;
        double final_residual = 0.0;

        // Dispatch to selected Poisson solver
        // Note: Selection was done at init time; we just execute the selected path here
        static bool solver_logged = false;

#ifdef USE_GPU_OFFLOAD
        if (gpu_ready_) {
            // GPU path based on selected solver
            switch (selected_solver_) {
#ifdef USE_FFT_POISSON
                case PoissonSolverType::FFT:
                    if (fft_poisson_solver_) {
                        if (!solver_logged) {
                            std::cout << "[Poisson] Using FFT solve_device() (cuFFT+cuSPARSE)\n";
                            solver_logged = true;
                        }
                        cycles = fft_poisson_solver_->solve_device(rhs_poisson_ptr_, pressure_corr_ptr_, pcfg);
                        final_residual = fft_poisson_solver_->residual();
                    }
                    break;
                case PoissonSolverType::FFT2D:
                    if (fft2d_poisson_solver_) {
                        if (!solver_logged) {
                            std::cout << "[Poisson] Using FFT2D solve_device() (1D cuFFT + cuSPARSE tridiag)\n";
                            solver_logged = true;
                        }
                        cycles = fft2d_poisson_solver_->solve_device(rhs_poisson_ptr_, pressure_corr_ptr_, pcfg);
                        final_residual = fft2d_poisson_solver_->residual();
                    }
                    break;
                case PoissonSolverType::FFT1D:
                    if (fft1d_poisson_solver_) {
                        if (!solver_logged) {
                            std::cout << "[Poisson] Using FFT1D solve_device() (1D cuFFT + 2D Helmholtz)\n";
                            solver_logged = true;
                        }
                        cycles = fft1d_poisson_solver_->solve_device(rhs_poisson_ptr_, pressure_corr_ptr_, pcfg);
                        final_residual = fft1d_poisson_solver_->residual();
                    }
                    break;
#endif
#ifdef USE_HYPRE
                case PoissonSolverType::HYPRE:
                    if (hypre_poisson_solver_) {
                        if (hypre_poisson_solver_->using_cuda()) {
                            if (!solver_logged) {
                                std::cout << "[Poisson] Using HYPRE solve_device() (CUDA)\n";
                                solver_logged = true;
                            }
                            cycles = hypre_poisson_solver_->solve_device(rhs_poisson_ptr_, pressure_corr_ptr_, pcfg);
                            final_residual = hypre_poisson_solver_->residual();
                        } else {
                            // HYPRE host fallback with GPU staging
                            if (!solver_logged) {
                                std::cout << "[Poisson] Using HYPRE solve() (host, GPU staging)\n";
                                solver_logged = true;
                            }
                            #pragma omp target update from(rhs_poisson_ptr_[0:field_total_size_])
                            std::memcpy(rhs_poisson_.data().data(), rhs_poisson_ptr_, field_total_size_ * sizeof(double));
                            cycles = hypre_poisson_solver_->solve(rhs_poisson_, pressure_correction_, pcfg);
                            final_residual = hypre_poisson_solver_->residual();
                            std::memcpy(pressure_corr_ptr_, pressure_correction_.data().data(), field_total_size_ * sizeof(double));
                            #pragma omp target update to(pressure_corr_ptr_[0:field_total_size_])
                        }
                    }
                    break;
#endif
                case PoissonSolverType::MG:
                default:
                    if (!solver_logged) {
                        std::cout << "[Poisson] Using MG solve_device()\n";
                        solver_logged = true;
                    }
                    cycles = mg_poisson_solver_.solve_device(rhs_poisson_ptr_, pressure_corr_ptr_, pcfg);
                    final_residual = mg_poisson_solver_.residual();
                    break;
            }
        } else
#endif
        {
            // Host path
            if (!solver_logged) {
                std::cout << "[Poisson] Using HOST path\n";
                solver_logged = true;
            }
            if (use_multigrid_) {
                cycles = mg_poisson_solver_.solve(rhs_poisson_, pressure_correction_, pcfg);
                final_residual = mg_poisson_solver_.residual();
            } else {
                cycles = poisson_solver_.solve(rhs_poisson_, pressure_correction_, pcfg);
                final_residual = poisson_solver_.residual();
            }
        }

        // Print cycle count diagnostics if enabled
        if (poisson_diagnostics && (iter_ % poisson_diagnostics_interval == 0)) {
            std::cout << "[Poisson] iter=" << iter_ << " cycles=" << cycles
                      << " residual=" << std::scientific << std::setprecision(6)
                      << final_residual;
            // For MG solver, also print norms and ratios for convergence analysis
            if (selected_solver_ == PoissonSolverType::MG) {
                // Get both L∞ and L2 norms
                double r_inf = mg_poisson_solver_.residual();
                double r_l2 = mg_poisson_solver_.residual_l2();
                double b_inf = mg_poisson_solver_.rhs_norm();
                double b_l2 = mg_poisson_solver_.rhs_norm_l2();
                double r0_inf = mg_poisson_solver_.initial_residual();
                double r0_l2 = mg_poisson_solver_.initial_residual_l2();

                // Show which norm is used for convergence
                const char* norm_type = pcfg.use_l2_norm ? "L2" : "Linf";
                double r_norm = pcfg.use_l2_norm ? r_l2 : r_inf;
                double b_norm = pcfg.use_l2_norm ? b_l2 : b_inf;
                double r0_norm = pcfg.use_l2_norm ? r0_l2 : r0_inf;
                double r_over_b = (b_norm > 1e-30) ? r_norm / b_norm : 0.0;
                double r_over_r0 = (r0_norm > 1e-30) ? r_norm / r0_norm : 0.0;
                std::cout << " [" << norm_type << "] ||b||=" << b_norm
                          << " ||r0||=" << r0_norm
                          << " ||r||/||b||=" << r_over_b
                          << " ||r||/||r0||=" << r_over_r0;
            }
            std::cout << "\n";
        }
        
        NVTX_POP();
    }
    
    // 5. Correct velocity and pressure
    {
        TIMED_SCOPE("velocity_correction");
        NVTX_PUSH("velocity_correction");
        correct_velocity();
        NVTX_POP();
    }

    // 6. Apply boundary conditions
    apply_velocity_bc();

    // 7. Recycling inflow: extract from recycle plane, process, apply at inlet
    if (use_recycling_) {
        NVTX_PUSH("recycling_inflow");
        extract_recycle_plane();      // Sample velocity at recycle plane
        process_recycle_inflow();     // Apply shift, filter, mass-flux correction
        apply_recycling_inlet_bc();   // Override inlet BC with recycled data
        apply_fringe_blending();      // Optional: smooth transition near inlet
        NVTX_POP();
    }

    } // End Euler time integration path

    // Post-projection divergence check (diagnostic only)
    // This is the actual measure of projection quality: max|div(u^{n+1})|
    {
        static bool div_diagnostics = (std::getenv("NNCFD_POISSON_DIAGNOSTICS") != nullptr);
        static int div_diagnostics_interval = []() {
            const char* env = std::getenv("NNCFD_POISSON_DIAGNOSTICS_INTERVAL");
            int v = env ? std::atoi(env) : 1;
            return (v > 0) ? v : 1;
        }();
        if (div_diagnostics && (iter_ % div_diagnostics_interval == 0)) {
            compute_divergence(VelocityWhich::Current, div_velocity_);  // Divergence of corrected velocity
            double max_div = 0.0;
            double sum_div2 = 0.0;  // For L2 norm
            int n_cells = 0;
            if (mesh_->is2D()) {
                const double dV = mesh_->dx * mesh_->dy;  // Cell volume (uniform)
                for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                    for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                        double d = div_velocity_(i, j);
                        max_div = std::max(max_div, std::abs(d));
                        sum_div2 += d * d * dV;
                        n_cells++;
                    }
                }
            } else {
                const double dV = mesh_->dx * mesh_->dy * mesh_->dz;  // Cell volume
                for (int k = mesh_->k_begin(); k < mesh_->k_end(); ++k) {
                    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                            double d = div_velocity_(i, j, k);
                            max_div = std::max(max_div, std::abs(d));
                            sum_div2 += d * d * dV;
                            n_cells++;
                        }
                    }
                }
            }
            double l2_div = std::sqrt(sum_div2);
            std::cout << "[Projection] ||div(u)||_Linf=" << std::scientific << std::setprecision(6)
                      << max_div << " ||div(u)||_L2=" << l2_div
                      << " dt*Linf=" << current_dt_ * max_div << "\n";
        }
    }
    
    // Note: iter_ is managed by the outer solve loop, don't increment here

    // Return max velocity change as convergence criterion (unified view-based)
    NVTX_PUSH("residual_computation");
    auto v_res = get_solver_view();

    [[maybe_unused]] const size_t u_total_size_res = velocity_.u_total_size();
    [[maybe_unused]] const size_t v_total_size_res = velocity_.v_total_size();
    [[maybe_unused]] const size_t w_total_size_res = velocity_.w_total_size();
    const double* u_new_ptr = v_res.u_face;
    const double* v_new_ptr = v_res.v_face;
    const double* u_old_ptr = v_res.u_old_face;
    const double* v_old_ptr = v_res.v_old_face;
    const int u_stride_res = v_res.u_stride;
    const int v_stride_res = v_res.v_stride;
    const int Nz = mesh_->Nz;
    const bool is_2d_res = mesh_->is2D();
    const int Nz_eff = is_2d_res ? 1 : Nz;  // Effective Nz for loop bounds

    // Compute max |u_new - u_old| via reduction
    const int n_u_faces_res = (Nx + 1) * Ny * Nz_eff;
    const int u_plane_stride_res = is_2d_res ? 0 : v_res.u_plane_stride;
    double max_du = 0.0;
    #pragma omp target teams distribute parallel for reduction(max:max_du) \
        map(present: u_new_ptr[0:u_total_size_res], u_old_ptr[0:u_total_size_res]) \
        map(to: Ng, u_stride_res, u_plane_stride_res, Nx, Ny, Nz_eff, is_2d_res)
    for (int idx = 0; idx < n_u_faces_res; ++idx) {
        int i_local = idx % (Nx + 1);
        int j_local = (idx / (Nx + 1)) % Ny;
        int k_local = idx / ((Nx + 1) * Ny);
        int i = i_local + Ng;
        int j = j_local + Ng;
        int k = k_local + Ng;
        int u_idx = is_2d_res ? (j * u_stride_res + i)
                              : (k * u_plane_stride_res + j * u_stride_res + i);
        double du = u_new_ptr[u_idx] - u_old_ptr[u_idx];
        if (du < 0.0) du = -du;
        if (du > max_du) max_du = du;
    }

    // Compute max |v_new - v_old| via reduction
    const int n_v_faces_res = Nx * (Ny + 1) * Nz_eff;
    const int v_plane_stride_res = is_2d_res ? 0 : v_res.v_plane_stride;
    double max_dv = 0.0;
    #pragma omp target teams distribute parallel for reduction(max:max_dv) \
        map(present: v_new_ptr[0:v_total_size_res], v_old_ptr[0:v_total_size_res]) \
        map(to: Ng, v_stride_res, v_plane_stride_res, Nx, Ny, Nz_eff, is_2d_res)
    for (int idx = 0; idx < n_v_faces_res; ++idx) {
        int i_local = idx % Nx;
        int j_local = (idx / Nx) % (Ny + 1);
        int k_local = idx / (Nx * (Ny + 1));
        int i = i_local + Ng;
        int j = j_local + Ng;
        int k = k_local + Ng;
        int v_idx = is_2d_res ? (j * v_stride_res + i)
                              : (k * v_plane_stride_res + j * v_stride_res + i);
        double dv = v_new_ptr[v_idx] - v_old_ptr[v_idx];
        if (dv < 0.0) dv = -dv;
        if (dv > max_dv) max_dv = dv;
    }

    double max_change = (max_du > max_dv) ? max_du : max_dv;

    // For 3D, also check w component
    if (!is_2d_res) {
        const double* w_new_ptr = v_res.w_face;
        const double* w_old_ptr = v_res.w_old_face;
        const int w_stride_res = v_res.w_stride;
        const int w_plane_stride_res = v_res.w_plane_stride;
        const int n_w_faces_res = Nx * Ny * (Nz + 1);
        double max_dw = 0.0;
        #pragma omp target teams distribute parallel for reduction(max:max_dw) \
            map(present: w_new_ptr[0:w_total_size_res], w_old_ptr[0:w_total_size_res]) \
            map(to: Ng, w_stride_res, w_plane_stride_res, Nx, Ny, Nz)
        for (int idx = 0; idx < n_w_faces_res; ++idx) {
            int i_local = idx % Nx;
            int j_local = (idx / Nx) % Ny;
            int k_local = idx / (Nx * Ny);
            int i = i_local + Ng;
            int j = j_local + Ng;
            int k = k_local + Ng;
            int w_idx = k * w_plane_stride_res + j * w_stride_res + i;
            double dw = w_new_ptr[w_idx] - w_old_ptr[w_idx];
            if (dw < 0.0) dw = -dw;
            if (dw > max_dw) max_dw = dw;
        }
        if (max_dw > max_change) max_change = max_dw;
    }

    NVTX_POP();  // End residual_computation

    // NaN/Inf GUARD: Check for numerical stability issues
    // Do this after turbulence update but before next iteration starts
    check_for_nan_inf(step_count_);
    ++step_count_;

    return max_change;
}


std::pair<double, int> RANSSolver::solve_steady() {
    double residual = 1.0;
    
    if (config_.verbose) {
        // Enable line buffering for immediate output visibility (SLURM/redirected stdout)
        std::cout << std::unitbuf;
        
        if (config_.adaptive_dt) {
            std::cout << std::setw(8) << "Iter" 
                      << std::setw(15) << "Residual"
                      << std::setw(15) << "Max |u|"
                      << std::setw(12) << "dt"
                      << std::endl;
        } else {
            std::cout << std::setw(8) << "Iter" 
                      << std::setw(15) << "Residual"
                      << std::setw(15) << "Max |u|"
                      << std::endl;
        }
    }
    
    for (iter_ = 0; iter_ < config_.max_steps; ++iter_) {
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
                          << std::endl;  // Flush for immediate visibility
            } else {
                std::cout << std::setw(8) << iter_ + 1
                          << std::setw(15) << std::scientific << std::setprecision(3) << residual
                          << std::setw(15) << std::fixed << max_vel
                          << std::endl;  // Flush for immediate visibility
            }
        }
        
        if (residual < config_.tol) {
            if (config_.verbose) {
                std::cout << "Converged at iteration " << iter_ + 1 
                          << " with residual " << residual << std::endl;
            }
            break;
        }
        
        // Check for divergence
        if (std::isnan(residual) || std::isinf(residual)) {
            if (config_.verbose) {
                std::cerr << "Solver diverged at iteration " << iter_ + 1 << std::endl;
            }
            break;
        }
    }
    
#ifdef USE_GPU_OFFLOAD
    // Sync solution fields after solve completes for backward compatibility
    // This ensures CPU data is up-to-date for tests and diagnostics
    // Note: solve_steady_with_snapshots() handles syncs during I/O instead
    sync_solution_from_gpu();
#endif
    
    return {residual, iter_ + 1};
}

std::pair<double, int> RANSSolver::solve_steady_with_snapshots(
    const std::string& output_prefix,
    int num_snapshots,
    int snapshot_freq) 
{
    // Calculate snapshot frequency if not provided
    if (snapshot_freq < 0 && num_snapshots > 0) {
        snapshot_freq = std::max(1, config_.max_steps / num_snapshots);
    }
    
    if (config_.verbose && !output_prefix.empty()) {
        std::cout << "Will output ";
        if (num_snapshots > 0) {
            std::cout << num_snapshots << " VTK snapshots (every " 
                     << snapshot_freq << " iterations)" << std::endl;
        } else {
            std::cout << "final VTK snapshot only" << std::endl;
        }
    }
    
    double residual = 1.0;
    int snapshot_count = 0;

    // Progress output interval for CI visibility (always enabled)
    const int progress_interval = std::max(1, config_.max_steps / 10);

    if (config_.verbose) {
        // Enable line buffering for immediate output visibility
        std::cout << std::unitbuf;

        if (config_.adaptive_dt) {
            std::cout << std::setw(8) << "Iter"
                      << std::setw(15) << "Residual"
                      << std::setw(15) << "Max |u|"
                      << std::setw(12) << "dt"
                      << std::endl;
        } else {
            std::cout << std::setw(8) << "Iter"
                      << std::setw(15) << "Residual"
                      << std::setw(15) << "Max |u|"
                      << std::endl;
        }
    }

    for (iter_ = 0; iter_ < config_.max_steps; ++iter_) {
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
                             << ": " << vtk_file << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "Warning: Could not write VTK snapshot: " 
                         << e.what() << std::endl;
            }
        }
        
        // Always show progress every ~10% for CI visibility
        if ((iter_ + 1) % progress_interval == 0 || iter_ == 0) {
            std::cout << "    Iter " << std::setw(6) << iter_ + 1 << " / " << config_.max_steps
                      << "  (" << std::setw(3) << (100 * (iter_ + 1) / config_.max_steps) << "%)"
                      << "  residual = " << std::scientific << std::setprecision(3) << residual
                      << std::fixed << "\n" << std::flush;
        } else if (config_.verbose && (iter_ + 1) % config_.output_freq == 0) {
            // Detailed verbose output
            double max_vel = velocity_.max_magnitude();
            if (config_.adaptive_dt) {
                std::cout << std::setw(8) << iter_ + 1
                          << std::setw(15) << std::scientific << std::setprecision(3) << residual
                          << std::setw(15) << std::fixed << max_vel
                          << std::setw(12) << std::scientific << std::setprecision(2) << current_dt_
                          << std::endl;
            } else {
                std::cout << std::setw(8) << iter_ + 1
                          << std::setw(15) << std::scientific << std::setprecision(3) << residual
                          << std::setw(15) << std::fixed << max_vel
                          << std::endl;
            }
        }
        
        if (residual < config_.tol) {
            if (config_.verbose) {
                std::cout << "Converged at iteration " << iter_ + 1 
                          << " with residual " << residual << std::endl;
            }
            break;
        }
        
        // Check for divergence
        if (std::isnan(residual) || std::isinf(residual)) {
            if (config_.verbose) {
                std::cerr << "Solver diverged at iteration " << iter_ + 1 << std::endl;
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
    
#ifdef USE_GPU_OFFLOAD
    // Sync all fields from GPU after solve completes
    // write_vtk() calls sync_from_gpu(), but if no output was written we still need to sync
    if (output_prefix.empty()) {
        sync_from_gpu();
    }
#endif
    
    return {residual, iter_ + 1};
}

double RANSSolver::bulk_velocity() const {
    // Area-averaged streamwise velocity
    double sum = 0.0;
    int count = 0;
    
    [[maybe_unused]] const int Nx = mesh_->Nx;
    [[maybe_unused]] const int Ny = mesh_->Ny;
    [[maybe_unused]] const int Ng = mesh_->Nghost;
    
#ifdef USE_GPU_OFFLOAD
    if (gpu_ready_) {
        // GPU path: compute sum on device, only transfer scalar
        const size_t u_total_size = velocity_.u_total_size();
        const int u_stride = Nx + 2*Ng + 1;
        
        #pragma omp target teams distribute parallel for \
            map(present: velocity_u_ptr_[0:u_total_size]) \
            reduction(+:sum)
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                int ii = i + Ng;
                int jj = j + Ng;
                sum += velocity_u_ptr_[jj * u_stride + ii];
            }
        }
        count = Nx * Ny;
    } else
#endif
    {
        // Host path
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                sum += velocity_.u(i, j);
                ++count;
            }
        }
    }

    return sum / count;
}

double RANSSolver::wall_shear_stress() const {
    // Compute du/dy at the bottom wall
    // Using one-sided difference from first interior cell to wall
    double sum = 0.0;
    int count = 0;
    
    [[maybe_unused]] const int Nx = mesh_->Nx;
    const int Ng = mesh_->Nghost;
    const int j_wall = Ng;  // First interior row
    const double y_cell = mesh_->y(j_wall);
    const double y_wall = mesh_->y_min;
    const double dist = y_cell - y_wall;
    
#ifdef USE_GPU_OFFLOAD
    if (gpu_ready_) {
        // GPU path: compute sum on device, only transfer scalar
        const size_t u_total_size = velocity_.u_total_size();
        const int u_stride = Nx + 2*Ng + 1;
        
        #pragma omp target teams distribute parallel for \
            map(present: velocity_u_ptr_[0:u_total_size]) \
            reduction(+:sum)
        for (int i = 0; i < Nx; ++i) {
            int ii = i + Ng;
            // u at wall is 0 (no-slip), so dudy = u[j_wall] / dist
            double dudy = velocity_u_ptr_[j_wall * u_stride + ii] / dist;
            sum += dudy;
        }
        count = Nx;
    } else
#endif
    {
        // Host path
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            // u at wall is 0 (no-slip)
            double dudy = velocity_.u(i, j_wall) / dist;
            sum += dudy;
            ++count;
        }
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

double RANSSolver::compute_convective_ke_production() const {
    // Compute <u, conv(u)> = rate of KE change due to advection
    // For skew-symmetric advection with div(u)=0, this should be ~0
    //
    // IMPORTANT: For periodic directions, the last face wraps to the first,
    // so we only sum over Nx (not Nx+1) unique faces to avoid double counting.
    // For non-periodic (wall) directions, all Ny+1 faces are unique.

    double sum = 0.0;
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Ng = mesh_->Nghost;
    const double dV = mesh_->dx * mesh_->dy * (mesh_->is2D() ? 1.0 : mesh_->dz);

    // Determine periodicity from velocity BCs
    const bool x_periodic = (velocity_bc_.x_lo == VelocityBC::Periodic);
    const bool y_periodic = (velocity_bc_.y_lo == VelocityBC::Periodic);
    const bool z_periodic = (velocity_bc_.z_lo == VelocityBC::Periodic);

    // Face counts: periodic dirs have N unique faces, non-periodic have N+1
    const int n_u_x = x_periodic ? Nx : (Nx + 1);  // u-faces in x
    const int n_v_y = y_periodic ? Ny : (Ny + 1);  // v-faces in y
    const int n_w_z = z_periodic ? Nz : (Nz + 1);  // w-faces in z

    if (mesh_->is2D()) {
        // 2D: u-faces + v-faces
        // u-faces: n_u_x faces in x, Ny cells in y
        for (int j = Ng; j < Ng + Ny; ++j) {
            for (int i = Ng; i < Ng + n_u_x; ++i) {
                sum += velocity_.u(i, j) * conv_.u(i, j) * dV;
            }
        }
        // v-faces: Nx cells in x, n_v_y faces in y
        for (int j = Ng; j < Ng + n_v_y; ++j) {
            for (int i = Ng; i < Ng + Nx; ++i) {
                sum += velocity_.v(i, j) * conv_.v(i, j) * dV;
            }
        }
    } else {
        // 3D: u-faces + v-faces + w-faces
        // u-faces: n_u_x in x, Ny in y, Nz in z
        for (int k = Ng; k < Ng + Nz; ++k) {
            for (int j = Ng; j < Ng + Ny; ++j) {
                for (int i = Ng; i < Ng + n_u_x; ++i) {
                    sum += velocity_.u(i, j, k) * conv_.u(i, j, k) * dV;
                }
            }
        }
        // v-faces: Nx in x, n_v_y in y, Nz in z
        for (int k = Ng; k < Ng + Nz; ++k) {
            for (int j = Ng; j < Ng + n_v_y; ++j) {
                for (int i = Ng; i < Ng + Nx; ++i) {
                    sum += velocity_.v(i, j, k) * conv_.v(i, j, k) * dV;
                }
            }
        }
        // w-faces: Nx in x, Ny in y, n_w_z in z
        for (int k = Ng; k < Ng + n_w_z; ++k) {
            for (int j = Ng; j < Ng + Ny; ++j) {
                for (int i = Ng; i < Ng + Nx; ++i) {
                    sum += velocity_.w(i, j, k) * conv_.w(i, j, k) * dV;
                }
            }
        }
    }

    return sum;
}

// ============================================================================
// NaN/Inf Guard: Abort immediately on non-finite values
// ============================================================================

void RANSSolver::check_for_nan_inf(int step) const {
    if (!config_.turb_guard_enabled) {
        return;  // Guard disabled in config
    }
    
    // Only check every guard_interval steps (performance)
    if (step % config_.turb_guard_interval != 0) {
        return;
    }
    
    bool all_finite = true;
    const bool has_transport = turb_model_ && turb_model_->uses_transport_equations();
    
#ifdef USE_GPU_OFFLOAD
    // GPU path: Do NaN/Inf check entirely on device, only transfer 1 scalar
    if (gpu_ready_) {
        int has_bad = 0;

        const size_t u_total = velocity_.u_total_size();
        const size_t v_total = velocity_.v_total_size();
        const size_t field_total = field_total_size_;

        // Local aliases to avoid implicit 'this' mapping (NVHPC workaround)
        const double* u = velocity_u_ptr_;
        const double* v = velocity_v_ptr_;
        const double* p = pressure_ptr_;
        const double* nut = nu_t_ptr_;
        const double* k_arr = k_ptr_;
        const double* omega_arr = omega_ptr_;

        #pragma omp target data map(present: u[0:u_total], v[0:v_total], p[0:field_total], nut[0:field_total])
        {
            // Check u-velocity (x-faces)
            #pragma omp target teams distribute parallel for reduction(|: has_bad)
            for (size_t idx = 0; idx < u_total; ++idx) {
                const double x = u[idx];
                // Use manual NaN/Inf check (x != x for NaN, or x-x != 0 for Inf)
                has_bad |= (x != x || (x - x) != 0.0) ? 1 : 0;
            }

            // Check v-velocity (y-faces)
            #pragma omp target teams distribute parallel for reduction(|: has_bad)
            for (size_t idx = 0; idx < v_total; ++idx) {
                const double x = v[idx];
                has_bad |= (x != x || (x - x) != 0.0) ? 1 : 0;
            }

            // Check pressure and eddy viscosity (cell-centered)
            #pragma omp target teams distribute parallel for reduction(|: has_bad)
            for (size_t idx = 0; idx < field_total; ++idx) {
                const double pval = p[idx];
                const double nutval = nut[idx];
                has_bad |= (pval != pval || (pval - pval) != 0.0 || nutval != nutval || (nutval - nutval) != 0.0) ? 1 : 0;
            }
        }

        // Check transport variables if turbulence model uses them
        if (has_transport) {
            #pragma omp target teams distribute parallel for \
                map(present: k_arr[0:field_total], omega_arr[0:field_total]) \
                reduction(|: has_bad)
            for (size_t idx = 0; idx < field_total; ++idx) {
                const double kval = k_arr[idx];
                const double wval = omega_arr[idx];
                has_bad |= (kval != kval || (kval - kval) != 0.0 || wval != wval || (wval - wval) != 0.0) ? 1 : 0;
            }
        }

        all_finite = (has_bad == 0);
    } else
#endif
    {
        // CPU path: Check host-side fields directly (no GPU sync needed)
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                // Check velocity, pressure, nu_t
                double u = 0.5 * (velocity_.u(i, j) + velocity_.u(i+1, j));
                double v = 0.5 * (velocity_.v(i, j) + velocity_.v(i, j+1));
                double p = pressure_(i, j);
                double nu_t_val = nu_t_(i, j);
                
                if (!std::isfinite(u) || !std::isfinite(v) || 
                    !std::isfinite(p) || !std::isfinite(nu_t_val)) {
                    all_finite = false;
                    break;
                }
                
                // Check transport variables if applicable
                if (has_transport) {
                    double k_val = k_(i, j);
                    double omega_val = omega_(i, j);
                    
                    if (!std::isfinite(k_val) || !std::isfinite(omega_val)) {
                        all_finite = false;
                        break;
                    }
                }
            }
            if (!all_finite) break;
        }
    }
    
    // Abort immediately on non-finite values
    if (!all_finite) {
        std::cerr << "\n========================================\n";
        std::cerr << "NUMERICAL STABILITY GUARD: NaN/Inf DETECTED\n";
        std::cerr << "========================================\n";
        std::cerr << "Step: " << step << "\n";
        std::cerr << "\nOne or more fields contain NaN or Inf:\n";
        std::cerr << "  - Velocity (u, v)\n";
        std::cerr << "  - Pressure (p)\n";
        std::cerr << "  - Eddy viscosity (nu_t)\n";
        if (has_transport) {
            std::cerr << "  - Transport variables (k, omega)\n";
        }
        std::cerr << "\nThis indicates numerical instability.\n";
        std::cerr << "Aborting to prevent garbage propagation.\n";
        std::cerr << "\nPossible causes:\n";
        std::cerr << "  - Time step too large (reduce dt or enable adaptive_dt)\n";
        std::cerr << "  - Turbulence model incompatible with flow regime\n";
        std::cerr << "  - Mesh resolution insufficient\n";
        std::cerr << "  - Boundary conditions inconsistent\n";
        std::cerr << "========================================\n";
        throw std::runtime_error("NaN/Inf detected in solution fields");
    }
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
#ifdef USE_GPU_OFFLOAD
    // Download solution fields from GPU before writing
    const_cast<RANSSolver*>(this)->sync_solution_from_gpu();
    // Transport fields only if turbulence model active
    if (turb_model_ && turb_model_->uses_transport_equations()) {
        const_cast<RANSSolver*>(this)->sync_transport_from_gpu();
    }
#endif
    
    velocity_.write(prefix + "_velocity.dat");
    pressure_.write(prefix + "_pressure.dat");

    if (turb_model_) {
        nu_t_.write(prefix + "_nu_t.dat");
    }
}

double RANSSolver::compute_adaptive_dt() const {
    [[maybe_unused]] const int Nx = mesh_->Nx;
    [[maybe_unused]] const int Ny = mesh_->Ny;
    [[maybe_unused]] const int Ng = mesh_->Nghost;
    const double nu = config_.nu;
    
// Unified CPU/GPU path: compute CFL and diffusive constraints using raw pointers
    double u_max = 1e-10;
    double nu_eff_max = nu;

    [[maybe_unused]] const size_t u_total_size = velocity_.u_total_size();
    [[maybe_unused]] const size_t v_total_size = velocity_.v_total_size();
    [[maybe_unused]] const size_t field_total_size = field_total_size_;
    const int u_stride = Nx + 2*Ng + 1;
    const int v_stride = Nx + 2*Ng;
    const int stride = Nx + 2*Ng;

    if (mesh_->is2D()) {
#ifdef USE_GPU_OFFLOAD
        // Local aliases to avoid implicit 'this' mapping (NVHPC workaround)
        const double* u = velocity_u_ptr_;
        const double* v = velocity_v_ptr_;
        const double* nut = nu_t_ptr_;
        const size_t n_u = u_total_size;
        const size_t n_v = v_total_size;
        const size_t n_f = field_total_size;

        // 2D: Compute max velocity magnitude (for advective CFL)
        #pragma omp target teams distribute parallel for \
            map(present: u[0:n_u], v[0:n_v]) reduction(max:u_max)
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                int ii = i + Ng;
                int jj = j + Ng;
                // Interpolate u and v to cell center for staggered grid
                double u_avg = 0.5 * (u[jj * u_stride + ii] + u[jj * u_stride + ii + 1]);
                double v_avg = 0.5 * (v[jj * v_stride + ii] + v[(jj + 1) * v_stride + ii]);
                double u_mag = sqrt(u_avg*u_avg + v_avg*v_avg);
                if (u_mag > u_max) u_max = u_mag;
            }
        }

        // 2D: Compute max effective viscosity (for diffusive CFL) if turbulence active
        if (turb_model_) {
            #pragma omp target teams distribute parallel for \
                map(present: nut[0:n_f]) reduction(max:nu_eff_max)
            for (int j = 0; j < Ny; ++j) {
                for (int i = 0; i < Nx; ++i) {
                    int ii = i + Ng;
                    int jj = j + Ng;
                    int idx = jj * stride + ii;
                    double nu_eff = nu + nut[idx];
                    if (nu_eff > nu_eff_max) nu_eff_max = nu_eff;
                }
            }
        }
#else
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                int ii = i + Ng;
                int jj = j + Ng;
                double u_avg = 0.5 * (velocity_u_ptr_[jj * u_stride + ii] +
                                      velocity_u_ptr_[jj * u_stride + ii + 1]);
                double v_avg = 0.5 * (velocity_v_ptr_[jj * v_stride + ii] +
                                      velocity_v_ptr_[(jj + 1) * v_stride + ii]);
                double u_mag = sqrt(u_avg*u_avg + v_avg*v_avg);
                if (u_mag > u_max) u_max = u_mag;
            }
        }
        if (turb_model_) {
            for (int j = 0; j < Ny; ++j) {
                for (int i = 0; i < Nx; ++i) {
                    int ii = i + Ng;
                    int jj = j + Ng;
                    int idx = jj * stride + ii;
                    double nu_eff = nu + nu_t_ptr_[idx];
                    if (nu_eff > nu_eff_max) nu_eff_max = nu_eff;
                }
            }
        }
#endif
    } else {
        // 3D case
        const int Nz = mesh_->Nz;
        [[maybe_unused]] const size_t w_total_size = velocity_.w_total_size();
        const int u_plane_stride = u_stride * (Ny + 2*Ng);
        const int v_plane_stride = v_stride * (Ny + 2*Ng + 1);
        const int w_stride = Nx + 2*Ng;
        const int w_plane_stride = w_stride * (Ny + 2*Ng);
        const int plane_stride = stride * (Ny + 2*Ng);

#ifdef USE_GPU_OFFLOAD
        // Local aliases to avoid implicit 'this' mapping (NVHPC workaround)
        const double* u = velocity_u_ptr_;
        const double* v = velocity_v_ptr_;
        const double* w = velocity_w_ptr_;
        const double* nut = nu_t_ptr_;
        const size_t n_u = u_total_size;
        const size_t n_v = v_total_size;
        const size_t n_w = w_total_size;
        const size_t n_f = field_total_size;

        // 3D: Compute max velocity magnitude (for advective CFL)
        #pragma omp target teams distribute parallel for collapse(3) \
            map(present: u[0:n_u], v[0:n_v], w[0:n_w]) reduction(max:u_max)
        for (int k = 0; k < Nz; ++k) {
            for (int j = 0; j < Ny; ++j) {
                for (int i = 0; i < Nx; ++i) {
                    int ii = i + Ng;
                    int jj = j + Ng;
                    int kk = k + Ng;
                    // Interpolate u, v, w to cell center for staggered grid
                    double u_avg = 0.5 * (u[kk * u_plane_stride + jj * u_stride + ii] +
                                          u[kk * u_plane_stride + jj * u_stride + ii + 1]);
                    double v_avg = 0.5 * (v[kk * v_plane_stride + jj * v_stride + ii] +
                                          v[kk * v_plane_stride + (jj + 1) * v_stride + ii]);
                    double w_avg = 0.5 * (w[kk * w_plane_stride + jj * w_stride + ii] +
                                          w[(kk + 1) * w_plane_stride + jj * w_stride + ii]);
                    double u_mag = sqrt(u_avg*u_avg + v_avg*v_avg + w_avg*w_avg);
                    if (u_mag > u_max) u_max = u_mag;
                }
            }
        }

        // 3D: Compute max effective viscosity (for diffusive CFL) if turbulence active
        if (turb_model_) {
            #pragma omp target teams distribute parallel for collapse(3) \
                map(present: nut[0:n_f]) reduction(max:nu_eff_max)
            for (int k = 0; k < Nz; ++k) {
                for (int j = 0; j < Ny; ++j) {
                    for (int i = 0; i < Nx; ++i) {
                        int ii = i + Ng;
                        int jj = j + Ng;
                        int kk = k + Ng;
                        int idx = kk * plane_stride + jj * stride + ii;
                        double nu_eff = nu + nut[idx];
                        if (nu_eff > nu_eff_max) nu_eff_max = nu_eff;
                    }
                }
            }
        }
#else
        for (int k = 0; k < Nz; ++k) {
            for (int j = 0; j < Ny; ++j) {
                for (int i = 0; i < Nx; ++i) {
                    int ii = i + Ng;
                    int jj = j + Ng;
                    int kk = k + Ng;
                    double u_avg = 0.5 * (velocity_u_ptr_[kk * u_plane_stride + jj * u_stride + ii] +
                                          velocity_u_ptr_[kk * u_plane_stride + jj * u_stride + ii + 1]);
                    double v_avg = 0.5 * (velocity_v_ptr_[kk * v_plane_stride + jj * v_stride + ii] +
                                          velocity_v_ptr_[kk * v_plane_stride + (jj + 1) * v_stride + ii]);
                    double w_avg = 0.5 * (velocity_w_ptr_[kk * w_plane_stride + jj * w_stride + ii] +
                                          velocity_w_ptr_[(kk + 1) * w_plane_stride + jj * w_stride + ii]);
                    double u_mag = sqrt(u_avg*u_avg + v_avg*v_avg + w_avg*w_avg);
                    if (u_mag > u_max) u_max = u_mag;
                }
            }
        }
        if (turb_model_) {
            for (int k = 0; k < Nz; ++k) {
                for (int j = 0; j < Ny; ++j) {
                    for (int i = 0; i < Nx; ++i) {
                        int ii = i + Ng;
                        int jj = j + Ng;
                        int kk = k + Ng;
                        int idx = kk * plane_stride + jj * stride + ii;
                        double nu_eff = nu + nu_t_ptr_[idx];
                        if (nu_eff > nu_eff_max) nu_eff_max = nu_eff;
                    }
                }
            }
        }
#endif
    }

    // Compute time step constraints (same for GPU and CPU)
    double dx_min = mesh_->is2D() ? std::min(mesh_->dx, mesh_->dy)
                                  : std::min({mesh_->dx, mesh_->dy, mesh_->dz});
    double dt_cfl = config_.CFL_max * dx_min / u_max;
    
    // Diffusive stability: dt < 0.25 * dx² / ν (hard limit from von Neumann analysis)
    // NOTE: Do NOT scale by CFL_max - this is a stability constant, not a tuning parameter
    double dt_diff = 0.25 * dx_min * dx_min / nu_eff_max;
    
    return std::min(dt_cfl, dt_diff);
}


// ============================================================================
// Shared pointer extraction (used by both CPU and GPU paths)
// ============================================================================

void RANSSolver::extract_field_pointers() {
    field_total_size_ = mesh_->total_cells();

    // Staggered grid velocity fields
    velocity_u_ptr_ = velocity_.u_data().data();
    velocity_v_ptr_ = velocity_.v_data().data();
    velocity_star_u_ptr_ = velocity_star_.u_data().data();
    velocity_star_v_ptr_ = velocity_star_.v_data().data();
    velocity_old_u_ptr_ = velocity_old_.u_data().data();
    velocity_old_v_ptr_ = velocity_old_.v_data().data();
    velocity_rk_u_ptr_ = velocity_rk_.u_data().data();
    velocity_rk_v_ptr_ = velocity_rk_.v_data().data();

    // Cell-centered fields
    pressure_ptr_ = pressure_.data().data();
    pressure_corr_ptr_ = pressure_correction_.data().data();
    nu_t_ptr_ = nu_t_.data().data();
    nu_eff_ptr_ = nu_eff_.data().data();
    rhs_poisson_ptr_ = rhs_poisson_.data().data();
    div_velocity_ptr_ = div_velocity_.data().data();

    // Work arrays
    conv_u_ptr_ = conv_.u_data().data();
    conv_v_ptr_ = conv_.v_data().data();
    diff_u_ptr_ = diff_.u_data().data();
    diff_v_ptr_ = diff_.v_data().data();

    // 3D w-velocity fields
    if (!mesh_->is2D()) {
        velocity_w_ptr_ = velocity_.w_data().data();
        velocity_star_w_ptr_ = velocity_star_.w_data().data();
        velocity_old_w_ptr_ = velocity_old_.w_data().data();
        velocity_rk_w_ptr_ = velocity_rk_.w_data().data();
        conv_w_ptr_ = conv_.w_data().data();
        diff_w_ptr_ = diff_.w_data().data();
    }

    // Turbulence transport fields
    k_ptr_ = k_.data().data();
    omega_ptr_ = omega_.data().data();

    // Reynolds stress tensor components (for EARSM/TBNN)
    tau_xx_ptr_ = tau_ij_.xx_data().data();
    tau_xy_ptr_ = tau_ij_.xy_data().data();
    tau_yy_ptr_ = tau_ij_.yy_data().data();

    // Gradient scratch buffers for turbulence models
    dudx_ptr_ = dudx_.data().data();
    dudy_ptr_ = dudy_.data().data();
    dvdx_ptr_ = dvdx_.data().data();
    dvdy_ptr_ = dvdy_.data().data();
    wall_distance_ptr_ = wall_distance_.data().data();
}

#ifdef USE_GPU_OFFLOAD
void RANSSolver::initialize_gpu_buffers() {
    // Verify GPU is available (throws if not)
    gpu::verify_device_available();

    // Extract all raw pointers (shared with CPU path)
    extract_field_pointers();
    
#ifdef GPU_PROFILE_TRANSFERS
    auto transfer_start = std::chrono::steady_clock::now();
#endif
    
    // Fail fast if no GPU device available (GPU build requires GPU)
    gpu::verify_device_available();
    
    // Map all arrays to GPU device and copy initial values
    // Using map(to:) to transfer initialized data, map(alloc:) for device-only buffers
    // Data will persist on GPU for entire solver lifetime
    // Staggered grid: u and v have different sizes
    const size_t u_total_size = velocity_.u_total_size();
    const size_t v_total_size = velocity_.v_total_size();

    // Consolidated GPU buffer mapping - grouping arrays by size and transfer type
    // Group 1: u-component sized arrays (to: transfer initial data)
    #pragma omp target enter data \
        map(to: velocity_u_ptr_[0:u_total_size], \
                velocity_star_u_ptr_[0:u_total_size], \
                conv_u_ptr_[0:u_total_size], \
                diff_u_ptr_[0:u_total_size])

    // Group 2: v-component sized arrays (to: transfer initial data)
    #pragma omp target enter data \
        map(to: velocity_v_ptr_[0:v_total_size], \
                velocity_star_v_ptr_[0:v_total_size], \
                conv_v_ptr_[0:v_total_size], \
                diff_v_ptr_[0:v_total_size])

    // Group 3: field-sized arrays with initial data (to: transfer)
    #pragma omp target enter data \
        map(to: pressure_ptr_[0:field_total_size_], \
                pressure_corr_ptr_[0:field_total_size_], \
                nu_t_ptr_[0:field_total_size_], \
                nu_eff_ptr_[0:field_total_size_], \
                rhs_poisson_ptr_[0:field_total_size_], \
                div_velocity_ptr_[0:field_total_size_], \
                k_ptr_[0:field_total_size_], \
                omega_ptr_[0:field_total_size_])

    // Group 4: gradient buffers (to: need zero init to prevent NaN in EARSM)
    #pragma omp target enter data \
        map(to: dudx_ptr_[0:field_total_size_], \
                dudy_ptr_[0:field_total_size_], \
                dvdx_ptr_[0:field_total_size_], \
                dvdy_ptr_[0:field_total_size_], \
                wall_distance_ptr_[0:field_total_size_])

    // Group 5: device-only arrays (alloc: will be computed on GPU)
    // velocity_old: device-resident for residual computation (host never used)
    // velocity_rk: work buffer for RK time integration stages
    // tau_*: Reynolds stress components computed by EARSM/TBNN
    #pragma omp target enter data \
        map(alloc: velocity_old_u_ptr_[0:u_total_size], \
                   velocity_old_v_ptr_[0:v_total_size], \
                   velocity_rk_u_ptr_[0:u_total_size], \
                   velocity_rk_v_ptr_[0:v_total_size], \
                   tau_xx_ptr_[0:field_total_size_], \
                   tau_xy_ptr_[0:field_total_size_], \
                   tau_yy_ptr_[0:field_total_size_])

    // 3D w-velocity fields
    if (!mesh_->is2D()) {
        const size_t w_total_size = velocity_.w_total_size();
        #pragma omp target enter data \
            map(to: velocity_w_ptr_[0:w_total_size], \
                    velocity_star_w_ptr_[0:w_total_size], \
                    conv_w_ptr_[0:w_total_size], \
                    diff_w_ptr_[0:w_total_size]) \
            map(alloc: velocity_old_w_ptr_[0:w_total_size], \
                       velocity_rk_w_ptr_[0:w_total_size])
    }

    // Zero-initialize device-only arrays to prevent garbage in first residual computation
    // Arrays allocated with map(alloc:) contain garbage until explicitly written
    #pragma omp target teams distribute parallel for map(present: velocity_old_u_ptr_[0:u_total_size])
    for (size_t i = 0; i < u_total_size; ++i) velocity_old_u_ptr_[i] = 0.0;

    #pragma omp target teams distribute parallel for map(present: velocity_old_v_ptr_[0:v_total_size])
    for (size_t i = 0; i < v_total_size; ++i) velocity_old_v_ptr_[i] = 0.0;

    #pragma omp target teams distribute parallel for map(present: velocity_rk_u_ptr_[0:u_total_size])
    for (size_t i = 0; i < u_total_size; ++i) velocity_rk_u_ptr_[i] = 0.0;

    #pragma omp target teams distribute parallel for map(present: velocity_rk_v_ptr_[0:v_total_size])
    for (size_t i = 0; i < v_total_size; ++i) velocity_rk_v_ptr_[i] = 0.0;

    if (!mesh_->is2D()) {
        const size_t w_total_size = velocity_.w_total_size();
        #pragma omp target teams distribute parallel for map(present: velocity_old_w_ptr_[0:w_total_size])
        for (size_t i = 0; i < w_total_size; ++i) velocity_old_w_ptr_[i] = 0.0;
        #pragma omp target teams distribute parallel for map(present: velocity_rk_w_ptr_[0:w_total_size])
        for (size_t i = 0; i < w_total_size; ++i) velocity_rk_w_ptr_[i] = 0.0;
    }

    // Zero-initialize Reynolds stress tensor components
    #pragma omp target teams distribute parallel for map(present: tau_xx_ptr_[0:field_total_size_])
    for (size_t i = 0; i < field_total_size_; ++i) tau_xx_ptr_[i] = 0.0;

    #pragma omp target teams distribute parallel for map(present: tau_xy_ptr_[0:field_total_size_])
    for (size_t i = 0; i < field_total_size_; ++i) tau_xy_ptr_[i] = 0.0;

    #pragma omp target teams distribute parallel for map(present: tau_yy_ptr_[0:field_total_size_])
    for (size_t i = 0; i < field_total_size_; ++i) tau_yy_ptr_[i] = 0.0;

    // Verify mappings succeeded (fail fast if GPU unavailable despite num_devices>0)
    if (!gpu::is_pointer_present(velocity_u_ptr_)) {
        throw std::runtime_error("GPU mapping failed despite device availability");
    }
    
    gpu_ready_ = true;
    
#ifdef GPU_PROFILE_TRANSFERS
    auto transfer_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> transfer_time = transfer_end - transfer_start;
    double mb_transferred = 16 * field_total_size_ * sizeof(double) / 1024.0 / 1024.0;
    double bandwidth = mb_transferred / transfer_time.count();
    (void)mb_transferred;
    (void)bandwidth;
#endif
}

void RANSSolver::cleanup_gpu_buffers() {
    assert(gpu_ready_ && "GPU must be initialized before cleanup");
    
    // Staggered grid sizes
    const size_t u_total_size = velocity_.u_total_size();
    const size_t v_total_size = velocity_.v_total_size();
    
    // Copy final results back from GPU, then free device memory
    // Using map(from:) to get final state back to host
    #pragma omp target exit data map(from: velocity_u_ptr_[0:u_total_size])
    #pragma omp target exit data map(from: velocity_v_ptr_[0:v_total_size])
    #pragma omp target exit data map(from: pressure_ptr_[0:field_total_size_])
    #pragma omp target exit data map(from: nu_t_ptr_[0:field_total_size_])

    // 3D w-velocity results
    if (!mesh_->is2D()) {
        const size_t w_total_size = velocity_.w_total_size();
        #pragma omp target exit data map(from: velocity_w_ptr_[0:w_total_size])
    }

    // Delete temporary/work arrays without copying back
    #pragma omp target exit data map(delete: velocity_star_u_ptr_[0:u_total_size])
    #pragma omp target exit data map(delete: velocity_star_v_ptr_[0:v_total_size])
    #pragma omp target exit data map(delete: velocity_old_u_ptr_[0:u_total_size])
    #pragma omp target exit data map(delete: velocity_old_v_ptr_[0:v_total_size])
    #pragma omp target exit data map(delete: velocity_rk_u_ptr_[0:u_total_size])
    #pragma omp target exit data map(delete: velocity_rk_v_ptr_[0:v_total_size])
    #pragma omp target exit data map(delete: pressure_corr_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: nu_eff_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: conv_u_ptr_[0:u_total_size])
    #pragma omp target exit data map(delete: conv_v_ptr_[0:v_total_size])
    #pragma omp target exit data map(delete: diff_u_ptr_[0:u_total_size])
    #pragma omp target exit data map(delete: diff_v_ptr_[0:v_total_size])
    #pragma omp target exit data map(delete: rhs_poisson_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: div_velocity_ptr_[0:field_total_size_])

    // 3D temporary arrays
    if (!mesh_->is2D()) {
        const size_t w_total_size = velocity_.w_total_size();
        #pragma omp target exit data map(delete: velocity_star_w_ptr_[0:w_total_size])
        #pragma omp target exit data map(delete: velocity_old_w_ptr_[0:w_total_size])
        #pragma omp target exit data map(delete: velocity_rk_w_ptr_[0:w_total_size])
        #pragma omp target exit data map(delete: conv_w_ptr_[0:w_total_size])
        #pragma omp target exit data map(delete: diff_w_ptr_[0:w_total_size])
    }
    
    // Delete gradient scratch buffers
    #pragma omp target exit data map(delete: dudx_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: dudy_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: dvdx_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: dvdy_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: wall_distance_ptr_[0:field_total_size_])
    
    // Delete transport fields
    #pragma omp target exit data map(delete: k_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: omega_ptr_[0:field_total_size_])
    
    // Delete Reynolds stress tensor buffers
    #pragma omp target exit data map(delete: tau_xx_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: tau_xy_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: tau_yy_ptr_[0:field_total_size_])
    
    gpu_ready_ = false;
}

void RANSSolver::sync_to_gpu() {
    assert(gpu_ready_ && "GPU must be initialized before sync");
    
    // Update GPU with changed fields (typically after CPU-side modifications)
    // Staggered grid: u and v have different sizes
    const size_t u_total_size = velocity_.u_total_size();
    const size_t v_total_size = velocity_.v_total_size();

    #pragma omp target update to(velocity_u_ptr_[0:u_total_size])
    #pragma omp target update to(velocity_v_ptr_[0:v_total_size])
    #pragma omp target update to(pressure_ptr_[0:field_total_size_])
    #pragma omp target update to(nu_t_ptr_[0:field_total_size_])

    // 3D w-velocity
    if (!mesh_->is2D()) {
        const size_t w_total_size = velocity_.w_total_size();
        #pragma omp target update to(velocity_w_ptr_[0:w_total_size])
    }

    // Upload k and omega if turbulence model uses transport equations
    // These are initialized by RANSSolver::initialize() after GPU buffers are allocated
    if (turb_model_ && turb_model_->uses_transport_equations()) {
        #pragma omp target update to(k_ptr_[0:field_total_size_])
        #pragma omp target update to(omega_ptr_[0:field_total_size_])
    }
}

void RANSSolver::sync_from_gpu() {
    // Legacy sync for backward compatibility - downloads everything
    // Prefer using sync_solution_from_gpu() and sync_transport_from_gpu() selectively
    sync_solution_from_gpu();
    sync_transport_from_gpu();
}

void RANSSolver::sync_solution_from_gpu() {
    assert(gpu_ready_ && "GPU must be initialized before sync");
    
    // Download only primary solution fields needed for I/O/analysis
    // Staggered grid: u and v have different sizes
    const size_t u_total_size = velocity_.u_total_size();
    const size_t v_total_size = velocity_.v_total_size();

    #pragma omp target update from(velocity_u_ptr_[0:u_total_size])
    #pragma omp target update from(velocity_v_ptr_[0:v_total_size])
    #pragma omp target update from(pressure_ptr_[0:field_total_size_])
    #pragma omp target update from(nu_t_ptr_[0:field_total_size_])

    // 3D w-velocity
    if (!mesh_->is2D()) {
        const size_t w_total_size = velocity_.w_total_size();
        #pragma omp target update from(velocity_w_ptr_[0:w_total_size])
    }
}

void RANSSolver::sync_transport_from_gpu() {
    assert(gpu_ready_ && "GPU must be initialized before sync");
    
    // Download transport equation fields (k, omega) only if turbulence model uses them
    // For laminar runs (turb_model = none), this saves hundreds of MB on large grids!
    if (turb_model_ && turb_model_->uses_transport_equations()) {
        #pragma omp target update from(k_ptr_[0:field_total_size_])
        #pragma omp target update from(omega_ptr_[0:field_total_size_])
    }
}

TurbulenceDeviceView RANSSolver::get_device_view() const {
    assert(gpu_ready_ && "GPU must be initialized to get device view");
    
    TurbulenceDeviceView view;
    
    // Velocity field (staggered, solver-owned, persistent on GPU)
    view.u_face = velocity_u_ptr_;
    view.v_face = velocity_v_ptr_;
    view.u_stride = velocity_.u_stride();
    view.v_stride = velocity_.v_stride();
    
    // Turbulence fields (cell-centered)
    view.k = k_ptr_;
    view.omega = omega_ptr_;
    view.nu_t = nu_t_ptr_;
    view.cell_stride = mesh_->total_Nx();  // Stride for cell-centered fields
    
    // Reynolds stress tensor
    view.tau_xx = tau_xx_ptr_;
    view.tau_xy = tau_xy_ptr_;
    view.tau_yy = tau_yy_ptr_;
    
    // Gradient scratch buffers
    view.dudx = dudx_ptr_;
    view.dudy = dudy_ptr_;
    view.dvdx = dvdx_ptr_;
    view.dvdy = dvdy_ptr_;
    
    // Wall distance
    view.wall_distance = wall_distance_ptr_;
    
    // Mesh parameters
    view.Nx = mesh_->Nx;
    view.Ny = mesh_->Ny;
    view.Ng = mesh_->Nghost;
    view.dx = mesh_->dx;
    view.dy = mesh_->dy;
    view.delta = (turb_model_ ? turb_model_->delta() : 1.0);
    
    return view;
}

// ============================================================================
// Device-side diagnostic functions (GPU QOI computation)
// ============================================================================

bool RANSSolver::verify_gpu_field_presence() const {
    if (!gpu_ready_) return false;

    int device = omp_get_default_device();
    bool all_present = true;

    // Helper lambda to check and report
    auto check_field = [&](double* ptr, const char* name) {
        if (!omp_target_is_present(ptr, device)) {
            std::fprintf(stderr, "[verify_gpu_field_presence] MISSING: %s (ptr=%p)\n",
                         name, static_cast<void*>(ptr));
            all_present = false;
        }
    };

    // Check all velocity fields (critical for RK stepping)
    check_field(velocity_u_ptr_, "velocity_u");
    check_field(velocity_v_ptr_, "velocity_v");
    check_field(velocity_star_u_ptr_, "velocity_star_u");
    check_field(velocity_star_v_ptr_, "velocity_star_v");
    check_field(velocity_old_u_ptr_, "velocity_old_u");
    check_field(velocity_old_v_ptr_, "velocity_old_v");
    check_field(velocity_rk_u_ptr_, "velocity_rk_u");
    check_field(velocity_rk_v_ptr_, "velocity_rk_v");

    // Check work arrays
    check_field(conv_u_ptr_, "conv_u");
    check_field(conv_v_ptr_, "conv_v");
    check_field(diff_u_ptr_, "diff_u");
    check_field(diff_v_ptr_, "diff_v");

    // Check scalar fields (critical for projection)
    check_field(pressure_ptr_, "pressure");
    check_field(pressure_corr_ptr_, "pressure_correction");
    check_field(rhs_poisson_ptr_, "rhs_poisson");
    check_field(div_velocity_ptr_, "div_velocity");
    check_field(nu_eff_ptr_, "nu_eff");

    // 3D fields
    if (!mesh_->is2D()) {
        check_field(velocity_w_ptr_, "velocity_w");
        check_field(velocity_star_w_ptr_, "velocity_star_w");
        check_field(velocity_old_w_ptr_, "velocity_old_w");
        check_field(velocity_rk_w_ptr_, "velocity_rk_w");
        check_field(conv_w_ptr_, "conv_w");
        check_field(diff_w_ptr_, "diff_w");
    }

    // If presence checks passed, do a write/read sanity check on a critical field
    if (all_present) {
        const int Ng = mesh_->Nghost;
        const int test_idx = Ng * (mesh_->Nx + 2 * Ng) + Ng;
        const double sentinel = 314159.265358979;

        double* rhs_dev = static_cast<double*>(omp_get_mapped_ptr(rhs_poisson_ptr_, device));

        double original = 0.0;
        double readback = 0.0;

        #pragma omp target is_device_ptr(rhs_dev) map(from: original)
        {
            original = rhs_dev[test_idx];
        }

        #pragma omp target is_device_ptr(rhs_dev)
        {
            rhs_dev[test_idx] = sentinel;
        }

        #pragma omp target is_device_ptr(rhs_dev) map(from: readback)
        {
            readback = rhs_dev[test_idx];
        }

        #pragma omp target is_device_ptr(rhs_dev) firstprivate(original)
        {
            rhs_dev[test_idx] = original;
        }

        if (std::abs(readback - sentinel) > 1e-10) {
            std::fprintf(stderr, "[verify_gpu_field_presence] WRITE/READ FAILED: "
                         "wrote %.6f, read %.6f (expected sentinel)\n", sentinel, readback);
            all_present = false;
        }
    }

    return all_present;
}

double RANSSolver::compute_kinetic_energy_device() const {
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Ng = mesh_->Nghost;
    const double dx = mesh_->dx;
    const double dy = mesh_->dy;
    const double dz = mesh_->is2D() ? 1.0 : mesh_->dz;
    const double dV = dx * dy * dz;

    const int u_stride = velocity_.u_stride();
    const int v_stride = velocity_.v_stride();

    double ke = 0.0;

    if (mesh_->is2D()) {
        const int n_cells = Nx * Ny;
        int device = omp_get_default_device();
        const double* u_dev = static_cast<const double*>(omp_get_mapped_ptr(velocity_u_ptr_, device));
        const double* v_dev = static_cast<const double*>(omp_get_mapped_ptr(velocity_v_ptr_, device));

        #pragma omp target teams distribute parallel for reduction(+:ke) is_device_ptr(u_dev, v_dev) \
            firstprivate(Nx, Ny, Ng, u_stride, v_stride, dV)
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + Ng;
            int j = idx / Nx + Ng;
            double u = 0.5 * (u_dev[j * u_stride + i] + u_dev[j * u_stride + (i + 1)]);
            double v = 0.5 * (v_dev[j * v_stride + i] + v_dev[(j + 1) * v_stride + i]);
            ke += 0.5 * (u * u + v * v) * dV;
        }
    } else {
        const int u_plane = velocity_.u_plane_stride();
        const int v_plane = velocity_.v_plane_stride();
        const int w_stride = velocity_.w_stride();
        const int w_plane = velocity_.w_plane_stride();
        const int n_cells = Nx * Ny * Nz;

        int device = omp_get_default_device();
        const double* u_dev = static_cast<const double*>(omp_get_mapped_ptr(velocity_u_ptr_, device));
        const double* v_dev = static_cast<const double*>(omp_get_mapped_ptr(velocity_v_ptr_, device));
        const double* w_dev = static_cast<const double*>(omp_get_mapped_ptr(velocity_w_ptr_, device));

        #pragma omp target teams distribute parallel for reduction(+:ke) is_device_ptr(u_dev, v_dev, w_dev) \
            firstprivate(Nx, Ny, Nz, Ng, u_stride, v_stride, w_stride, u_plane, v_plane, w_plane, dV)
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % Ny + Ng;
            int k = idx / (Nx * Ny) + Ng;
            double u = 0.5 * (u_dev[k * u_plane + j * u_stride + i] + u_dev[k * u_plane + j * u_stride + (i + 1)]);
            double v = 0.5 * (v_dev[k * v_plane + j * v_stride + i] + v_dev[k * v_plane + (j + 1) * v_stride + i]);
            double w = 0.5 * (w_dev[k * w_plane + j * w_stride + i] + w_dev[(k + 1) * w_plane + j * w_stride + i]);
            ke += 0.5 * (u * u + v * v + w * w) * dV;
        }
    }

    return ke;
}

double RANSSolver::compute_max_velocity_device() const {
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Ng = mesh_->Nghost;

    const int u_stride = velocity_.u_stride();
    const int v_stride = velocity_.v_stride();

    double max_vel = 0.0;

    const double* u_dev = gpu::dev_ptr(velocity_u_ptr_);
    const double* v_dev = gpu::dev_ptr(velocity_v_ptr_);

    if (mesh_->is2D()) {
        const int n_u = (Nx + 1) * Ny;
        #pragma omp target teams distribute parallel for is_device_ptr(u_dev) reduction(max:max_vel) \
            firstprivate(Nx, Ny, Ng, u_stride)
        for (int idx = 0; idx < n_u; ++idx) {
            int i = idx % (Nx + 1) + Ng;
            int j = idx / (Nx + 1) + Ng;
            double val = u_dev[j * u_stride + i];
            if (val < 0) val = -val;
            if (val > max_vel) max_vel = val;
        }

        const int n_v = Nx * (Ny + 1);
        #pragma omp target teams distribute parallel for is_device_ptr(v_dev) reduction(max:max_vel) \
            firstprivate(Nx, Ny, Ng, v_stride)
        for (int idx = 0; idx < n_v; ++idx) {
            int i = idx % Nx + Ng;
            int j = idx / Nx + Ng;
            double val = v_dev[j * v_stride + i];
            if (val < 0) val = -val;
            if (val > max_vel) max_vel = val;
        }
    } else {
        const int u_plane = velocity_.u_plane_stride();
        const int v_plane = velocity_.v_plane_stride();
        const int w_stride = velocity_.w_stride();
        const int w_plane = velocity_.w_plane_stride();
        const double* w_dev = gpu::dev_ptr(velocity_w_ptr_);

        const int n_u = (Nx + 1) * Ny * Nz;
        #pragma omp target teams distribute parallel for is_device_ptr(u_dev) reduction(max:max_vel) \
            firstprivate(Nx, Ny, Nz, Ng, u_stride, u_plane)
        for (int idx = 0; idx < n_u; ++idx) {
            int i = idx % (Nx + 1) + Ng;
            int j = (idx / (Nx + 1)) % Ny + Ng;
            int k = idx / ((Nx + 1) * Ny) + Ng;
            double val = u_dev[k * u_plane + j * u_stride + i];
            if (val < 0) val = -val;
            if (val > max_vel) max_vel = val;
        }

        const int n_v = Nx * (Ny + 1) * Nz;
        #pragma omp target teams distribute parallel for is_device_ptr(v_dev) reduction(max:max_vel) \
            firstprivate(Nx, Ny, Nz, Ng, v_stride, v_plane)
        for (int idx = 0; idx < n_v; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % (Ny + 1) + Ng;
            int k = idx / (Nx * (Ny + 1)) + Ng;
            double val = v_dev[k * v_plane + j * v_stride + i];
            if (val < 0) val = -val;
            if (val > max_vel) max_vel = val;
        }

        const int n_w = Nx * Ny * (Nz + 1);
        #pragma omp target teams distribute parallel for is_device_ptr(w_dev) reduction(max:max_vel) \
            firstprivate(Nx, Ny, Nz, Ng, w_stride, w_plane)
        for (int idx = 0; idx < n_w; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % Ny + Ng;
            int k = idx / (Nx * Ny) + Ng;
            double val = w_dev[k * w_plane + j * w_stride + i];
            if (val < 0) val = -val;
            if (val > max_vel) max_vel = val;
        }
    }

    return max_vel;
}

double RANSSolver::compute_divergence_linf_device() const {
    auto* self = const_cast<RANSSolver*>(this);
    self->compute_divergence(VelocityWhich::Current, self->div_velocity_);

    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Ng = mesh_->Nghost;
    const int stride = mesh_->total_Nx();
    const int plane_stride = stride * mesh_->total_Ny();

    double max_div = 0.0;
    const double* div_dev = gpu::dev_ptr(div_velocity_ptr_);

    if (mesh_->is2D()) {
        const int n_cells = Nx * Ny;
        #pragma omp target teams distribute parallel for is_device_ptr(div_dev) reduction(max:max_div) \
            firstprivate(Nx, Ny, Ng, stride)
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + Ng;
            int j = idx / Nx + Ng;
            double val = div_dev[j * stride + i];
            if (val < 0) val = -val;
            if (val > max_div) max_div = val;
        }
    } else {
        const int n_cells = Nx * Ny * Nz;
        #pragma omp target teams distribute parallel for is_device_ptr(div_dev) reduction(max:max_div) \
            firstprivate(Nx, Ny, Nz, Ng, stride, plane_stride)
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % Ny + Ng;
            int k = idx / (Nx * Ny) + Ng;
            double val = div_dev[k * plane_stride + j * stride + i];
            if (val < 0) val = -val;
            if (val > max_div) max_div = val;
        }
    }

    return max_div;
}

double RANSSolver::compute_max_conv_device() const {
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Ng = mesh_->Nghost;

    double max_conv = 0.0;

    const double* conv_u_dev = gpu::dev_ptr(conv_u_ptr_);
    const double* conv_v_dev = gpu::dev_ptr(conv_v_ptr_);

    if (mesh_->is2D()) {
        const int u_stride = conv_.u_stride();
        const int v_stride = conv_.v_stride();
        const int n_cells = Nx * Ny;

        #pragma omp target teams distribute parallel for is_device_ptr(conv_u_dev, conv_v_dev) \
            reduction(max:max_conv) firstprivate(Nx, Ny, Ng, u_stride, v_stride)
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + Ng;
            int j = idx / Nx + Ng;
            double cu = conv_u_dev[j * u_stride + i];
            double cv = conv_v_dev[j * v_stride + i];
            if (cu < 0) cu = -cu;
            if (cv < 0) cv = -cv;
            double val = (cu > cv) ? cu : cv;
            if (val > max_conv) max_conv = val;
        }
    } else {
        const int u_stride = conv_.u_stride();
        const int v_stride = conv_.v_stride();
        const int w_stride = conv_.w_stride();
        const int u_plane = conv_.u_plane_stride();
        const int v_plane = conv_.v_plane_stride();
        const int w_plane = conv_.w_plane_stride();
        const int n_cells = Nx * Ny * Nz;

        const double* conv_w_dev = gpu::dev_ptr(conv_w_ptr_);

        #pragma omp target teams distribute parallel for is_device_ptr(conv_u_dev, conv_v_dev, conv_w_dev) \
            reduction(max:max_conv) firstprivate(Nx, Ny, Nz, Ng, u_stride, v_stride, w_stride, u_plane, v_plane, w_plane)
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % Ny + Ng;
            int k = idx / (Nx * Ny) + Ng;
            double cu = conv_u_dev[k * u_plane + j * u_stride + i];
            double cv = conv_v_dev[k * v_plane + j * v_stride + i];
            double cw = conv_w_dev[k * w_plane + j * w_stride + i];
            if (cu < 0) cu = -cu;
            if (cv < 0) cv = -cv;
            if (cw < 0) cw = -cw;
            double val = cu;
            if (cv > val) val = cv;
            if (cw > val) val = cw;
            if (val > max_conv) max_conv = val;
        }
    }

    return max_conv;
}


SolverDeviceView RANSSolver::get_solver_view() const {
    SolverDeviceView view;

#ifdef USE_GPU_OFFLOAD
    assert(gpu_ready_ && "GPU must be initialized to get solver view");

    // GPU path: return device-present pointers
    view.u_face = velocity_u_ptr_;
    view.v_face = velocity_v_ptr_;
    view.u_star_face = velocity_star_u_ptr_;
    view.v_star_face = velocity_star_v_ptr_;
    view.u_old_face = velocity_old_u_ptr_;
    view.v_old_face = velocity_old_v_ptr_;
    view.u_stride = velocity_.u_stride();
    view.v_stride = velocity_.v_stride();

    // Initialize 3D fields to avoid undefined behavior in 2D mode
    view.w_face = nullptr;
    view.w_star_face = nullptr;
    view.w_old_face = nullptr;
    view.w_stride = 0;
    view.u_plane_stride = 0;
    view.v_plane_stride = 0;
    view.w_plane_stride = 0;

    // 3D velocity fields (overwrite defaults if 3D)
    if (!mesh_->is2D()) {
        view.w_face = velocity_w_ptr_;
        view.w_star_face = velocity_star_w_ptr_;
        view.w_old_face = velocity_old_w_ptr_;
        view.w_stride = velocity_.w_stride();
        view.u_plane_stride = velocity_.u_plane_stride();
        view.v_plane_stride = velocity_.v_plane_stride();
        view.w_plane_stride = velocity_.w_plane_stride();
    }

    view.p = pressure_ptr_;
    view.p_corr = pressure_corr_ptr_;
    view.nu_t = nu_t_ptr_;
    view.nu_eff = nu_eff_ptr_;
    view.rhs = rhs_poisson_ptr_;
    view.div = div_velocity_ptr_;
    view.cell_stride = mesh_->total_Nx();
    view.cell_plane_stride = mesh_->total_Nx() * mesh_->total_Ny();

    view.conv_u = conv_u_ptr_;
    view.conv_v = conv_v_ptr_;
    view.diff_u = diff_u_ptr_;
    view.diff_v = diff_v_ptr_;

    // Initialize 3D work arrays to avoid undefined behavior in 2D mode
    view.conv_w = nullptr;
    view.diff_w = nullptr;

    // 3D work arrays (overwrite defaults if 3D)
    if (!mesh_->is2D()) {
        view.conv_w = conv_w_ptr_;
        view.diff_w = diff_w_ptr_;
    }

    view.Nx = mesh_->Nx;
    view.Ny = mesh_->Ny;
    view.Nz = mesh_->Nz;
    view.Ng = mesh_->Nghost;
    view.dx = mesh_->dx;
    view.dy = mesh_->dy;
    view.dz = mesh_->dz;
    view.dt = current_dt_;
#else
    // CPU build: always return host pointers
    view.u_face = const_cast<double*>(velocity_.u_data().data());
    view.v_face = const_cast<double*>(velocity_.v_data().data());
    view.u_star_face = const_cast<double*>(velocity_star_.u_data().data());
    view.v_star_face = const_cast<double*>(velocity_star_.v_data().data());
    view.u_old_face = const_cast<double*>(velocity_old_.u_data().data());
    view.v_old_face = const_cast<double*>(velocity_old_.v_data().data());
    view.u_stride = velocity_.u_stride();
    view.v_stride = velocity_.v_stride();
    
    view.p = const_cast<double*>(pressure_.data().data());
    view.p_corr = const_cast<double*>(pressure_correction_.data().data());
    view.nu_t = const_cast<double*>(nu_t_.data().data());
    view.nu_eff = const_cast<double*>(nu_eff_.data().data());
    view.rhs = const_cast<double*>(rhs_poisson_.data().data());
    view.div = const_cast<double*>(div_velocity_.data().data());
    view.cell_stride = mesh_->total_Nx();
    
    view.conv_u = const_cast<double*>(conv_.u_data().data());
    view.conv_v = const_cast<double*>(conv_.v_data().data());
    view.diff_u = const_cast<double*>(diff_.u_data().data());
    view.diff_v = const_cast<double*>(diff_.v_data().data());
    
    view.Nx = mesh_->Nx;
    view.Ny = mesh_->Ny;
    view.Ng = mesh_->Nghost;
    view.dx = mesh_->dx;
    view.dy = mesh_->dy;
    view.dt = current_dt_;
#endif
    
    return view;
}
#else
// CPU: Set raw pointers for unified code paths (no GPU mapping)
//
// This function enables the same loop code to work on both CPU and GPU builds.
// In GPU builds, these pointers are mapped to device memory with OpenMP target pragmas.
// In CPU builds, the loops simply use these raw pointers directly (no pragmas applied).
// This unification eliminates divergent CPU/GPU arithmetic and reduces code duplication.
void RANSSolver::initialize_gpu_buffers() {
    extract_field_pointers();
    gpu_ready_ = false;
}

void RANSSolver::cleanup_gpu_buffers() {
    // No-op
}

void RANSSolver::sync_to_gpu() {
    // No-op
}

void RANSSolver::sync_from_gpu() {
    // No-op
}

void RANSSolver::sync_solution_from_gpu() {
    // No-op
}

void RANSSolver::sync_transport_from_gpu() {
    // No-op
}

TurbulenceDeviceView RANSSolver::get_device_view() const {
    // CPU build: return host pointers (same pattern as GPU version)
    TurbulenceDeviceView view;

    // Velocity field (staggered)
    view.u_face = velocity_u_ptr_;
    view.v_face = velocity_v_ptr_;
    view.u_stride = velocity_.u_stride();
    view.v_stride = velocity_.v_stride();

    // Turbulence fields (cell-centered)
    view.k = k_ptr_;
    view.omega = omega_ptr_;
    view.nu_t = nu_t_ptr_;
    view.cell_stride = mesh_->total_Nx();

    // Reynolds stress tensor
    view.tau_xx = tau_xx_ptr_;
    view.tau_xy = tau_xy_ptr_;
    view.tau_yy = tau_yy_ptr_;

    // Gradient scratch buffers
    view.dudx = dudx_ptr_;
    view.dudy = dudy_ptr_;
    view.dvdx = dvdx_ptr_;
    view.dvdy = dvdy_ptr_;

    // Wall distance
    view.wall_distance = wall_distance_ptr_;

    // Mesh parameters
    view.Nx = mesh_->Nx;
    view.Ny = mesh_->Ny;
    view.Ng = mesh_->Nghost;
    view.dx = mesh_->dx;
    view.dy = mesh_->dy;
    view.delta = (turb_model_ ? turb_model_->delta() : 1.0);

    return view;
}

SolverDeviceView RANSSolver::get_solver_view() const {
    // CPU build: always return host pointers
    SolverDeviceView view;

    view.u_face = const_cast<double*>(velocity_.u_data().data());
    view.v_face = const_cast<double*>(velocity_.v_data().data());
    view.u_star_face = const_cast<double*>(velocity_star_.u_data().data());
    view.v_star_face = const_cast<double*>(velocity_star_.v_data().data());
    view.u_old_face = const_cast<double*>(velocity_old_.u_data().data());
    view.v_old_face = const_cast<double*>(velocity_old_.v_data().data());
    view.u_stride = velocity_.u_stride();
    view.v_stride = velocity_.v_stride();

    // Initialize 3D fields to avoid undefined behavior in 2D mode
    view.w_face = nullptr;
    view.w_star_face = nullptr;
    view.w_old_face = nullptr;
    view.w_stride = 0;
    view.u_plane_stride = 0;
    view.v_plane_stride = 0;
    view.w_plane_stride = 0;

    // 3D velocity fields (overwrite defaults if 3D)
    if (!mesh_->is2D()) {
        view.w_face = const_cast<double*>(velocity_.w_data().data());
        view.w_star_face = const_cast<double*>(velocity_star_.w_data().data());
        view.w_old_face = const_cast<double*>(velocity_old_.w_data().data());
        view.w_stride = velocity_.w_stride();
        view.u_plane_stride = velocity_.u_plane_stride();
        view.v_plane_stride = velocity_.v_plane_stride();
        view.w_plane_stride = velocity_.w_plane_stride();
    }

    view.p = const_cast<double*>(pressure_.data().data());
    view.p_corr = const_cast<double*>(pressure_correction_.data().data());
    view.nu_t = const_cast<double*>(nu_t_.data().data());
    view.nu_eff = const_cast<double*>(nu_eff_.data().data());
    view.rhs = const_cast<double*>(rhs_poisson_.data().data());
    view.div = const_cast<double*>(div_velocity_.data().data());
    view.cell_stride = mesh_->total_Nx();
    view.cell_plane_stride = mesh_->total_Nx() * mesh_->total_Ny();

    view.conv_u = const_cast<double*>(conv_.u_data().data());
    view.conv_v = const_cast<double*>(conv_.v_data().data());
    view.diff_u = const_cast<double*>(diff_.u_data().data());
    view.diff_v = const_cast<double*>(diff_.v_data().data());

    // Initialize 3D work arrays to avoid undefined behavior in 2D mode
    view.conv_w = nullptr;
    view.diff_w = nullptr;

    // 3D work arrays (overwrite defaults if 3D)
    if (!mesh_->is2D()) {
        view.conv_w = const_cast<double*>(conv_.w_data().data());
        view.diff_w = const_cast<double*>(diff_.w_data().data());
    }

    view.Nx = mesh_->Nx;
    view.Ny = mesh_->Ny;
    view.Nz = mesh_->Nz;
    view.Ng = mesh_->Nghost;
    view.dx = mesh_->dx;
    view.dy = mesh_->dy;
    view.dz = mesh_->dz;
    view.dt = current_dt_;

    return view;
}
#endif

} // namespace nncfd

