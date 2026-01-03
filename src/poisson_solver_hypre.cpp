#ifdef USE_HYPRE

#include "poisson_solver_hypre.hpp"
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <cstring>
#include <algorithm>  // for std::min

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

namespace nncfd {

// Static flag to track if HYPRE has been initialized globally
static bool hypre_initialized = false;

// Forward declaration for static helper
static void apply_pressure_bc(ScalarField& p, const Mesh& mesh,
                               PoissonBC bc_x_lo, PoissonBC bc_x_hi,
                               PoissonBC bc_y_lo, PoissonBC bc_y_hi,
                               PoissonBC bc_z_lo, PoissonBC bc_z_hi);

HyprePoissonSolver::HyprePoissonSolver(const Mesh& mesh)
    : mesh_(&mesh) {
    initialize_hypre();
}

HyprePoissonSolver::~HyprePoissonSolver() {
    // Host buffers (rhs_host_, x_host_) are automatically cleaned up by vector destructor

    // Clean up HYPRE objects in reverse order of creation
    if (solver_) HYPRE_StructPFMGDestroy(solver_);
    if (x_) HYPRE_StructVectorDestroy(x_);
    if (b_) HYPRE_StructVectorDestroy(b_);
    if (A_) HYPRE_StructMatrixDestroy(A_);
    if (stencil_) HYPRE_StructStencilDestroy(stencil_);
    if (grid_) HYPRE_StructGridDestroy(grid_);

    // Note: We don't call HYPRE_Finalize() here because other solvers
    // might still be using HYPRE. Let it clean up at program exit.
}

void HyprePoissonSolver::set_bc(PoissonBC x_lo, PoissonBC x_hi,
                                 PoissonBC y_lo, PoissonBC y_hi) {
    bc_x_lo_ = x_lo;
    bc_x_hi_ = x_hi;
    bc_y_lo_ = y_lo;
    bc_y_hi_ = y_hi;
    // Keep z defaults for 2D
    matrix_assembled_ = false;  // Need to reassemble matrix
}

void HyprePoissonSolver::set_bc(PoissonBC x_lo, PoissonBC x_hi,
                                 PoissonBC y_lo, PoissonBC y_hi,
                                 PoissonBC z_lo, PoissonBC z_hi) {
    bc_x_lo_ = x_lo;
    bc_x_hi_ = x_hi;
    bc_y_lo_ = y_lo;
    bc_y_hi_ = y_hi;
    bc_z_lo_ = z_lo;
    bc_z_hi_ = z_hi;
    matrix_assembled_ = false;  // Need to reassemble matrix
}

void HyprePoissonSolver::initialize_hypre() {
    // Initialize HYPRE (only once globally)
    if (!hypre_initialized) {
        HYPRE_Init();

        #if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_GPU)
        // Initialize HYPRE's device subsystem
        // KEY: Use HOST memory for vectors, but DEVICE execution for solve.
        // HYPRE handles internal transfers. Using MEMORY_DEVICE with SetBoxValues
        // doesn't work correctly (data isn't copied properly), but HOST+EXEC_DEVICE
        // allows HYPRE to manage GPU execution internally while we use simple host pointers.
        HYPRE_DeviceInitialize();
        HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST);
        HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);
        using_cuda_ = true;
        std::cout << "[HyprePoissonSolver] CUDA backend enabled (HOST memory, DEVICE execution)\n";
        #else
        // Use HOST memory for CPU-based solve
        HYPRE_SetMemoryLocation(HYPRE_MEMORY_HOST);
        HYPRE_SetExecutionPolicy(HYPRE_EXEC_HOST);
        using_cuda_ = false;
        std::cout << "[HyprePoissonSolver] Using CPU backend (HOST memory)\n";
        #endif

        hypre_initialized = true;
    } else {
        // Check which mode we're in
        #if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_GPU)
        using_cuda_ = true;
        #else
        using_cuda_ = false;
        #endif
    }

    // Set grid extents (0-based indexing for interior cells)
    ilower_[0] = 0;
    ilower_[1] = 0;
    ilower_[2] = 0;
    iupper_[0] = mesh_->Nx - 1;
    iupper_[1] = mesh_->Ny - 1;
    iupper_[2] = mesh_->is2D() ? 0 : mesh_->Nz - 1;

    // Allocate device buffers for packed data (no ghost cells)
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->is2D() ? 1 : mesh_->Nz;
    device_buffer_size_ = static_cast<size_t>(Nx) * Ny * Nz;

    // Allocate host-side staging buffers for HYPRE
    // Since we use HYPRE_MEMORY_HOST + HYPRE_EXEC_DEVICE, HYPRE handles
    // the GPU transfers internally - we just need host buffers for SetBoxValues
    rhs_host_.resize(device_buffer_size_);
    x_host_.resize(device_buffer_size_);

    // Create HYPRE objects
    create_grid();
    create_stencil();
    create_matrix();
    create_vectors();
    create_solver();

    initialized_ = true;
}

void HyprePoissonSolver::create_grid() {
    const int ndim = mesh_->is2D() ? 2 : 3;

    // Create the grid
    HYPRE_StructGridCreate(MPI_COMM_SELF, ndim, &grid_);

    // Set the extents (single box for our structured grid)
    HYPRE_StructGridSetExtents(grid_, ilower_, iupper_);

    // Handle periodicity
    // HYPRE uses a period array where period[d] = 0 means non-periodic,
    // period[d] = N means periodic with N cells in that direction
    HYPRE_Int period[3] = {0, 0, 0};

    if (bc_x_lo_ == PoissonBC::Periodic && bc_x_hi_ == PoissonBC::Periodic) {
        period[0] = mesh_->Nx;
    }
    if (bc_y_lo_ == PoissonBC::Periodic && bc_y_hi_ == PoissonBC::Periodic) {
        period[1] = mesh_->Ny;
    }
    if (!mesh_->is2D() &&
        bc_z_lo_ == PoissonBC::Periodic && bc_z_hi_ == PoissonBC::Periodic) {
        period[2] = mesh_->Nz;
    }

    HYPRE_StructGridSetPeriodic(grid_, period);

    // Assemble the grid
    HYPRE_StructGridAssemble(grid_);
}

void HyprePoissonSolver::create_stencil() {
    const int ndim = mesh_->is2D() ? 2 : 3;
    const int stencil_size = mesh_->is2D() ? 5 : 7;  // 5-point 2D, 7-point 3D

    HYPRE_StructStencilCreate(ndim, stencil_size, &stencil_);

    // Define stencil offsets
    // 2D: center, west, east, south, north
    // 3D: center, west, east, south, north, back, front
    HYPRE_Int offsets[7][3] = {
        { 0,  0,  0},  // center
        {-1,  0,  0},  // west (i-1)
        { 1,  0,  0},  // east (i+1)
        { 0, -1,  0},  // south (j-1)
        { 0,  1,  0},  // north (j+1)
        { 0,  0, -1},  // back (k-1)
        { 0,  0,  1},  // front (k+1)
    };

    for (int s = 0; s < stencil_size; ++s) {
        HYPRE_StructStencilSetElement(stencil_, s, offsets[s]);
    }
}

void HyprePoissonSolver::create_matrix() {
    HYPRE_StructMatrixCreate(MPI_COMM_SELF, grid_, stencil_, &A_);
    HYPRE_StructMatrixInitialize(A_);
}

void HyprePoissonSolver::compute_laplacian_coefficients() {
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->is2D() ? 1 : mesh_->Nz;
    const int stencil_size = mesh_->is2D() ? 5 : 7;

    const size_t n_cells = static_cast<size_t>(Nx) * Ny * Nz;
    coeffs_.resize(n_cells * stencil_size);

    // Get mesh spacing
    const double dx = mesh_->dx;
    const double dy = mesh_->dy;
    const double dz = mesh_->is2D() ? 1.0 : mesh_->dz;

    // For uniform grids, coefficients are constant
    const double ax = 1.0 / (dx * dx);  // Coefficient for x neighbors
    const double ay = 1.0 / (dy * dy);  // Coefficient for y neighbors
    const double az = mesh_->is2D() ? 0.0 : 1.0 / (dz * dz);  // Coefficient for z neighbors

    // Loop over all interior cells
    for (int k = 0; k < Nz; ++k) {
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                const size_t cell_idx = (k * Ny + j) * Nx + i;
                double* c = &coeffs_[cell_idx * stencil_size];

                // Default coefficients (interior cell)
                double aW = ax, aE = ax;
                double aS = ay, aN = ay;
                double aB = az, aF = az;

                // Handle x-direction boundaries
                if (i == 0) {
                    if (bc_x_lo_ == PoissonBC::Neumann) {
                        aW = 0.0;  // No flux through boundary
                    } else if (bc_x_lo_ == PoissonBC::Dirichlet) {
                        // Dirichlet: coefficient stays, but handled in RHS
                    }
                    // Periodic: handled by HYPRE grid periodicity
                }
                if (i == Nx - 1) {
                    if (bc_x_hi_ == PoissonBC::Neumann) {
                        aE = 0.0;
                    }
                }

                // Handle y-direction boundaries
                if (j == 0) {
                    if (bc_y_lo_ == PoissonBC::Neumann) {
                        aS = 0.0;
                    }
                }
                if (j == Ny - 1) {
                    if (bc_y_hi_ == PoissonBC::Neumann) {
                        aN = 0.0;
                    }
                }

                // Handle z-direction boundaries (3D only)
                if (!mesh_->is2D()) {
                    if (k == 0) {
                        if (bc_z_lo_ == PoissonBC::Neumann) {
                            aB = 0.0;
                        }
                    }
                    if (k == Nz - 1) {
                        if (bc_z_hi_ == PoissonBC::Neumann) {
                            aF = 0.0;
                        }
                    }
                }

                // Diagonal is negative sum of off-diagonals (for Laplacian)
                double diag = -(aW + aE + aS + aN + aB + aF);

                // Store coefficients in HYPRE order
                c[STENCIL_CENTER] = diag;
                c[STENCIL_WEST] = aW;
                c[STENCIL_EAST] = aE;
                c[STENCIL_SOUTH] = aS;
                c[STENCIL_NORTH] = aN;
                if (!mesh_->is2D()) {
                    c[STENCIL_BACK] = aB;
                    c[STENCIL_FRONT] = aF;
                }
            }
        }
    }

    // NOTE: For HYPRE PFMG, we do NOT pin a cell to handle the null space.
    // PFMG can solve singular systems (all Neumann/Periodic) by finding a
    // particular solution. Pinning a cell breaks the periodic structure and
    // can cause convergence issues with multigrid.
    //
    // The solution is unique up to an additive constant. In practice, the
    // pressure gradient (what we use for velocity correction) is well-defined.
}

void HyprePoissonSolver::assemble_matrix() {
    // Compute Laplacian coefficients for all cells
    compute_laplacian_coefficients();

    const int stencil_size = mesh_->is2D() ? 5 : 7;

    // Use host vectors directly - HYPRE handles transfers internally
    // when using MEMORY_HOST + EXEC_DEVICE mode
    std::vector<HYPRE_Int> stencil_indices(stencil_size);
    for (int s = 0; s < stencil_size; ++s) {
        stencil_indices[s] = s;
    }

    HYPRE_StructMatrixSetBoxValues(A_, ilower_, iupper_,
                                    stencil_size, stencil_indices.data(),
                                    coeffs_.data());

    HYPRE_StructMatrixAssemble(A_);

    matrix_assembled_ = true;
}

void HyprePoissonSolver::create_vectors() {
    HYPRE_StructVectorCreate(MPI_COMM_SELF, grid_, &b_);
    HYPRE_StructVectorCreate(MPI_COMM_SELF, grid_, &x_);

    HYPRE_StructVectorInitialize(b_);
    HYPRE_StructVectorInitialize(x_);

    HYPRE_StructVectorAssemble(b_);
    HYPRE_StructVectorAssemble(x_);
}

void HyprePoissonSolver::create_solver() {
    HYPRE_StructPFMGCreate(MPI_COMM_SELF, &solver_);
}

void HyprePoissonSolver::setup_solver(const PoissonConfig& cfg) {
    // Set solver parameters
    HYPRE_StructPFMGSetMaxIter(solver_, cfg.max_iter);
    HYPRE_StructPFMGSetTol(solver_, cfg.tol);

    // Configure PFMG parameters - different settings for 2D vs 3D
    if (mesh_->is2D()) {
        // 2D: Use settings that match working standalone HYPRE 2D test
        // - RelaxType 2 (RB GS) works on GPU despite not being "officially" recommended
        // - Pre=2, Post=2 for good smoothing
        // - Don't limit levels - let HYPRE decide
        HYPRE_StructPFMGSetRelaxType(solver_, 2);  // RB Gauss-Seidel
        HYPRE_StructPFMGSetNumPreRelax(solver_, 2);
        HYPRE_StructPFMGSetNumPostRelax(solver_, 2);
        // Note: Don't set MaxLevels, SkipRelax, or RAPType for 2D - use defaults
    } else {
        // 3D: Optimized settings for GPU performance
        #if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_GPU)
        HYPRE_StructPFMGSetRelaxType(solver_, 1);  // Weighted Jacobi for GPU
        #else
        HYPRE_StructPFMGSetRelaxType(solver_, 2);  // RB Gauss-Seidel for CPU
        #endif

        HYPRE_StructPFMGSetNumPreRelax(solver_, 1);
        HYPRE_StructPFMGSetNumPostRelax(solver_, 1);

        // Skip fine grid relaxation for 3D (performance)
        HYPRE_StructPFMGSetSkipRelax(solver_, 1);

        // Coarse grid operator: Galerkin
        HYPRE_StructPFMGSetRAPType(solver_, 0);

        // Limit levels for GPU efficiency: coarsen to ~64 cells per dimension
        {
            const int Nx = mesh_->Nx;
            const int Ny = mesh_->Ny;
            const int Nz = mesh_->Nz;
            const int min_dim = std::min({Nx, Ny, Nz});
            int max_levels = 1;
            int dim = min_dim;
            while (dim > 64) {
                dim /= 2;
                max_levels++;
            }
            HYPRE_StructPFMGSetMaxLevels(solver_, max_levels);
        }

        // Set grid spacing for semicoarsening decisions (3D only)
        double dxyz[3] = {mesh_->dx, mesh_->dy, mesh_->dz};
        HYPRE_StructPFMGSetDxyz(solver_, dxyz);
    }

    // Logging: Always enable level 1 to compute residual norms for GetFinalRelativeResidualNorm
    // Without logging=1, HYPRE returns residual=0 which prevents proper convergence monitoring
    HYPRE_StructPFMGSetLogging(solver_, 1);

    // Print level: 0 = no output, 2 = convergence info (verbose only)
    if (cfg.verbose) {
        HYPRE_StructPFMGSetPrintLevel(solver_, 2);
    } else {
        HYPRE_StructPFMGSetPrintLevel(solver_, 0);
    }

    // Setup the solver with the matrix
    HYPRE_StructPFMGSetup(solver_, A_, b_, x_);

    #if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_GPU)
    // Synchronize after setup
    cudaDeviceSynchronize();
    #endif
}

bool HyprePoissonSolver::has_dirichlet_bc() const {
    return bc_x_lo_ == PoissonBC::Dirichlet ||
           bc_x_hi_ == PoissonBC::Dirichlet ||
           bc_y_lo_ == PoissonBC::Dirichlet ||
           bc_y_hi_ == PoissonBC::Dirichlet ||
           bc_z_lo_ == PoissonBC::Dirichlet ||
           bc_z_hi_ == PoissonBC::Dirichlet;
}

bool HyprePoissonSolver::needs_nullspace_handling() const {
    // If no Dirichlet BC, the Laplacian is singular (constant nullspace)
    return !has_dirichlet_bc();
}

int HyprePoissonSolver::solve(const ScalarField& rhs, ScalarField& p,
                               const PoissonConfig& cfg) {
    if (!initialized_) {
        throw std::runtime_error("HyprePoissonSolver not initialized");
    }

    // Assemble matrix if needed (first solve or after BC change)
    if (!matrix_assembled_) {
        assemble_matrix();
        setup_solver(cfg);
    }

    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->is2D() ? 1 : mesh_->Nz;
    const int Ng = mesh_->Nghost;

    // Use pre-allocated host buffers
    // With HYPRE_MEMORY_HOST + HYPRE_EXEC_DEVICE, HYPRE handles GPU transfers internally
    double* rhs_vals = rhs_host_.data();
    double* x_vals = x_host_.data();

    // Copy RHS and initial guess from ScalarField to host staging buffers
    for (int k = 0; k < Nz; ++k) {
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                const size_t hypre_idx = (static_cast<size_t>(k) * Ny + j) * Nx + i;
                rhs_vals[hypre_idx] = rhs(i + Ng, j + Ng, mesh_->is2D() ? 0 : k + Ng);
                x_vals[hypre_idx] = p(i + Ng, j + Ng, mesh_->is2D() ? 0 : k + Ng);
            }
        }
    }

    // NOTE: We don't pin a cell for singular systems - HYPRE PFMG handles this.
    // See comment in compute_laplacian_coefficients().

    // Set vector values using host buffers
    HYPRE_StructVectorSetBoxValues(b_, ilower_, iupper_, rhs_vals);
    HYPRE_StructVectorSetBoxValues(x_, ilower_, iupper_, x_vals);

    // Solve (HYPRE runs on GPU internally when EXEC_DEVICE is set)
    HYPRE_StructPFMGSolve(solver_, A_, b_, x_);

    // Get iteration count and residual
    HYPRE_Int num_iterations;
    HYPRE_StructPFMGGetNumIterations(solver_, &num_iterations);
    HYPRE_StructPFMGGetFinalRelativeResidualNorm(solver_, &residual_);

    // Copy solution back to host buffer
    HYPRE_StructVectorGetBoxValues(x_, ilower_, iupper_, x_vals);

    // Copy solution to ScalarField
    for (int k = 0; k < Nz; ++k) {
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                const size_t hypre_idx = (static_cast<size_t>(k) * Ny + j) * Nx + i;
                p(i + Ng, j + Ng, mesh_->is2D() ? 0 : k + Ng) = x_vals[hypre_idx];
            }
        }
    }

    // Apply boundary conditions to ghost cells
    // (The solver only updates interior cells)
    apply_pressure_bc(p, *mesh_, bc_x_lo_, bc_x_hi_, bc_y_lo_, bc_y_hi_,
                      bc_z_lo_, bc_z_hi_);

    return static_cast<int>(num_iterations);
}

int HyprePoissonSolver::solve_device(double* rhs_ptr, double* p_ptr,
                                      const PoissonConfig& cfg) {
#if !defined(USE_GPU_OFFLOAD) || !(defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_GPU))
    // If GPU offload or HYPRE CUDA not enabled, fall back to host solve
    (void)rhs_ptr;
    (void)p_ptr;
    (void)cfg;
    throw std::runtime_error("solve_device requires HYPRE CUDA backend");
#else
    if (!initialized_) {
        throw std::runtime_error("HyprePoissonSolver not initialized");
    }

    if (!using_cuda_) {
        throw std::runtime_error("solve_device requires HYPRE CUDA backend");
    }

    // Assemble matrix if needed (first solve or after BC change)
    if (!matrix_assembled_) {
        assemble_matrix();
        setup_solver(cfg);
    }

    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->is2D() ? 1 : mesh_->Nz;
    const int Ng = mesh_->Nghost;
    const bool is2D = mesh_->is2D();

    // Full array strides (with ghost cells)
    const int Nx_full = Nx + 2 * Ng;
    const int Ny_full = Ny + 2 * Ng;
    const size_t n_cells = static_cast<size_t>(Nx) * Ny * Nz;
    const size_t total_size = static_cast<size_t>(Nx_full) * Ny_full * (is2D ? 1 : (Nz + 2 * Ng));

    // Host staging buffers for HYPRE (MEMORY_HOST mode)
    double* rhs_vals = rhs_host_.data();
    double* x_vals = x_host_.data();

    // Copy data from GPU to host staging buffers
    // Using omp target update to transfer packed interior values
    #pragma omp target update from(rhs_ptr[0:total_size], p_ptr[0:total_size])

    // Pack RHS and initial guess from ghost-cell layout to packed layout on host
    for (int k = 0; k < Nz; ++k) {
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                const size_t hypre_idx = (static_cast<size_t>(k) * Ny + j) * Nx + i;
                const int kk = is2D ? 0 : k + Ng;
                const size_t full_idx = static_cast<size_t>(kk) * Nx_full * Ny_full +
                                        (j + Ng) * Nx_full + (i + Ng);
                rhs_vals[hypre_idx] = rhs_ptr[full_idx];
                x_vals[hypre_idx] = p_ptr[full_idx];
            }
        }
    }

    // NOTE: We don't pin a cell for singular systems - HYPRE PFMG handles this.
    // See comment in compute_laplacian_coefficients().

    // Set HYPRE vectors from host buffers
    HYPRE_StructVectorSetBoxValues(b_, ilower_, iupper_, rhs_vals);
    HYPRE_StructVectorSetBoxValues(x_, ilower_, iupper_, x_vals);

    // Solve using HYPRE PFMG (GPU execution handled internally by HYPRE)
    HYPRE_StructPFMGSolve(solver_, A_, b_, x_);

    // Get iteration count and residual
    HYPRE_Int num_iterations;
    HYPRE_StructPFMGGetNumIterations(solver_, &num_iterations);
    HYPRE_StructPFMGGetFinalRelativeResidualNorm(solver_, &residual_);

    // Get solution back to host buffer
    HYPRE_StructVectorGetBoxValues(x_, ilower_, iupper_, x_vals);

    // Unpack solution from packed layout to ghost-cell layout on host
    for (int k = 0; k < Nz; ++k) {
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                const size_t hypre_idx = (static_cast<size_t>(k) * Ny + j) * Nx + i;
                const int kk = is2D ? 0 : k + Ng;
                const size_t full_idx = static_cast<size_t>(kk) * Nx_full * Ny_full +
                                        (j + Ng) * Nx_full + (i + Ng);
                p_ptr[full_idx] = x_vals[hypre_idx];
            }
        }
    }

    // Apply boundary conditions on host (simpler than GPU version)
    // X boundaries
    for (int k = 0; k < Nz; ++k) {
        for (int j = 0; j < Ny; ++j) {
            const int kk = is2D ? 0 : k + Ng;
            const int jj = j + Ng;
            // x_lo ghost
            if (bc_x_lo_ == PoissonBC::Periodic) {
                p_ptr[kk * Nx_full * Ny_full + jj * Nx_full + 0] =
                    p_ptr[kk * Nx_full * Ny_full + jj * Nx_full + Nx];
            } else if (bc_x_lo_ == PoissonBC::Neumann) {
                p_ptr[kk * Nx_full * Ny_full + jj * Nx_full + 0] =
                    p_ptr[kk * Nx_full * Ny_full + jj * Nx_full + Ng];
            }
            // x_hi ghost
            if (bc_x_hi_ == PoissonBC::Periodic) {
                p_ptr[kk * Nx_full * Ny_full + jj * Nx_full + (Nx + Ng)] =
                    p_ptr[kk * Nx_full * Ny_full + jj * Nx_full + Ng];
            } else if (bc_x_hi_ == PoissonBC::Neumann) {
                p_ptr[kk * Nx_full * Ny_full + jj * Nx_full + (Nx + Ng)] =
                    p_ptr[kk * Nx_full * Ny_full + jj * Nx_full + (Nx + Ng - 1)];
            }
        }
    }

    // Y boundaries
    for (int k = 0; k < Nz; ++k) {
        for (int i = 0; i < Nx + 2 * Ng; ++i) {
            const int kk = is2D ? 0 : k + Ng;
            // y_lo ghost
            if (bc_y_lo_ == PoissonBC::Periodic) {
                p_ptr[kk * Nx_full * Ny_full + 0 * Nx_full + i] =
                    p_ptr[kk * Nx_full * Ny_full + Ny * Nx_full + i];
            } else if (bc_y_lo_ == PoissonBC::Neumann) {
                p_ptr[kk * Nx_full * Ny_full + 0 * Nx_full + i] =
                    p_ptr[kk * Nx_full * Ny_full + Ng * Nx_full + i];
            }
            // y_hi ghost
            if (bc_y_hi_ == PoissonBC::Periodic) {
                p_ptr[kk * Nx_full * Ny_full + (Ny + Ng) * Nx_full + i] =
                    p_ptr[kk * Nx_full * Ny_full + Ng * Nx_full + i];
            } else if (bc_y_hi_ == PoissonBC::Neumann) {
                p_ptr[kk * Nx_full * Ny_full + (Ny + Ng) * Nx_full + i] =
                    p_ptr[kk * Nx_full * Ny_full + (Ny + Ng - 1) * Nx_full + i];
            }
        }
    }

    // Z boundaries (3D only)
    if (!is2D) {
        for (int j = 0; j < Ny + 2 * Ng; ++j) {
            for (int i = 0; i < Nx + 2 * Ng; ++i) {
                // z_lo ghost
                if (bc_z_lo_ == PoissonBC::Periodic) {
                    p_ptr[0 * Nx_full * Ny_full + j * Nx_full + i] =
                        p_ptr[Nz * Nx_full * Ny_full + j * Nx_full + i];
                } else if (bc_z_lo_ == PoissonBC::Neumann) {
                    p_ptr[0 * Nx_full * Ny_full + j * Nx_full + i] =
                        p_ptr[Ng * Nx_full * Ny_full + j * Nx_full + i];
                }
                // z_hi ghost
                if (bc_z_hi_ == PoissonBC::Periodic) {
                    p_ptr[(Nz + Ng) * Nx_full * Ny_full + j * Nx_full + i] =
                        p_ptr[Ng * Nx_full * Ny_full + j * Nx_full + i];
                } else if (bc_z_hi_ == PoissonBC::Neumann) {
                    p_ptr[(Nz + Ng) * Nx_full * Ny_full + j * Nx_full + i] =
                        p_ptr[(Nz + Ng - 1) * Nx_full * Ny_full + j * Nx_full + i];
                }
            }
        }
    }

    // Copy solution back to GPU
    #pragma omp target update to(p_ptr[0:total_size])

    return static_cast<int>(num_iterations);
#endif // USE_GPU_OFFLOAD && HYPRE_USING_CUDA
}

// Helper function to apply BCs to ghost cells after solve
static void apply_pressure_bc(ScalarField& p, const Mesh& mesh,
                               PoissonBC bc_x_lo, PoissonBC bc_x_hi,
                               PoissonBC bc_y_lo, PoissonBC bc_y_hi,
                               PoissonBC bc_z_lo, PoissonBC bc_z_hi) {
    const int Nx = mesh.Nx;
    const int Ny = mesh.Ny;
    const int Nz = mesh.is2D() ? 1 : mesh.Nz;
    const int Ng = mesh.Nghost;

    // X boundaries
    for (int k = 0; k < Nz; ++k) {
        for (int j = 0; j < Ny; ++j) {
            const int kk = mesh.is2D() ? 0 : k + Ng;
            if (bc_x_lo == PoissonBC::Periodic) {
                p(0, j + Ng, kk) = p(Nx, j + Ng, kk);
            } else if (bc_x_lo == PoissonBC::Neumann) {
                p(0, j + Ng, kk) = p(Ng, j + Ng, kk);
            }

            if (bc_x_hi == PoissonBC::Periodic) {
                p(Nx + Ng, j + Ng, kk) = p(Ng, j + Ng, kk);
            } else if (bc_x_hi == PoissonBC::Neumann) {
                p(Nx + Ng, j + Ng, kk) = p(Nx + Ng - 1, j + Ng, kk);
            }
        }
    }

    // Y boundaries
    for (int k = 0; k < Nz; ++k) {
        for (int i = 0; i < Nx + 2 * Ng; ++i) {
            const int kk = mesh.is2D() ? 0 : k + Ng;
            if (bc_y_lo == PoissonBC::Periodic) {
                p(i, 0, kk) = p(i, Ny, kk);
            } else if (bc_y_lo == PoissonBC::Neumann) {
                p(i, 0, kk) = p(i, Ng, kk);
            }

            if (bc_y_hi == PoissonBC::Periodic) {
                p(i, Ny + Ng, kk) = p(i, Ng, kk);
            } else if (bc_y_hi == PoissonBC::Neumann) {
                p(i, Ny + Ng, kk) = p(i, Ny + Ng - 1, kk);
            }
        }
    }

    // Z boundaries (3D only)
    if (!mesh.is2D()) {
        for (int j = 0; j < Ny + 2 * Ng; ++j) {
            for (int i = 0; i < Nx + 2 * Ng; ++i) {
                if (bc_z_lo == PoissonBC::Periodic) {
                    p(i, j, 0) = p(i, j, Nz);
                } else if (bc_z_lo == PoissonBC::Neumann) {
                    p(i, j, 0) = p(i, j, Ng);
                }

                if (bc_z_hi == PoissonBC::Periodic) {
                    p(i, j, Nz + Ng) = p(i, j, Ng);
                } else if (bc_z_hi == PoissonBC::Neumann) {
                    p(i, j, Nz + Ng) = p(i, j, Nz + Ng - 1);
                }
            }
        }
    }
}

} // namespace nncfd

#endif // USE_HYPRE
