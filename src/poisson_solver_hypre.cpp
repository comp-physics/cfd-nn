#ifdef USE_HYPRE

#include "poisson_solver_hypre.hpp"
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <cstring>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

// For CUDA managed memory allocation (works with unified memory)
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_GPU)
#include <cuda_runtime.h>
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
    // Clean up device buffers (using cudaFree for managed memory)
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_GPU)
    if (rhs_device_) {
        cudaFree(rhs_device_);
        rhs_device_ = nullptr;
    }
    if (x_device_) {
        cudaFree(x_device_);
        x_device_ = nullptr;
    }
#endif

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
        // Ensure CUDA context is ready before HYPRE initialization
        #if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_GPU)
        cudaSetDevice(0);
        cudaFree(0);  // Force context creation
        #endif

        HYPRE_Init();

        #if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_GPU)
        // Initialize HYPRE's device subsystem with unified memory
        // Key: HYPRE was built with HYPRE_USING_UNIFIED_MEMORY=1, so we can
        // pass cudaMallocManaged pointers which work with both OMP and HYPRE CUDA
        HYPRE_DeviceInitialize();
        HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE);
        HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);
        using_cuda_ = true;
        std::cout << "[HyprePoissonSolver] CUDA backend enabled (unified memory)\n";
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

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_GPU)
    if (using_cuda_) {
        // Allocate MANAGED memory - accessible from both host and device
        // Works with both OpenMP target (is_device_ptr) and HYPRE CUDA
        cudaError_t err1 = cudaMallocManaged(&rhs_device_, device_buffer_size_ * sizeof(double));
        cudaError_t err2 = cudaMallocManaged(&x_device_, device_buffer_size_ * sizeof(double));

        if (err1 != cudaSuccess || err2 != cudaSuccess) {
            throw std::runtime_error("Failed to allocate managed memory for HYPRE buffers");
        }
    }
#endif

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

    // Handle singular case (all Neumann/Periodic) by pinning one cell
    if (needs_nullspace_handling()) {
        // Pin cell (0,0,0): set diagonal=1, off-diagonals=0
        double* c = &coeffs_[0];
        c[STENCIL_CENTER] = 1.0;
        c[STENCIL_WEST] = 0.0;
        c[STENCIL_EAST] = 0.0;
        c[STENCIL_SOUTH] = 0.0;
        c[STENCIL_NORTH] = 0.0;
        if (!mesh_->is2D()) {
            c[STENCIL_BACK] = 0.0;
            c[STENCIL_FRONT] = 0.0;
        }
    }
}

void HyprePoissonSolver::assemble_matrix() {
    // Compute Laplacian coefficients for all cells
    compute_laplacian_coefficients();

    const int stencil_size = mesh_->is2D() ? 5 : 7;
    const size_t n_cells = device_buffer_size_;

    #if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_GPU)
    // For CUDA mode, we need to use managed memory for the coefficient arrays
    double* m_coeffs = nullptr;
    HYPRE_Int* m_stencil_indices = nullptr;

    cudaMallocManaged(&m_coeffs, n_cells * stencil_size * sizeof(double));
    cudaMallocManaged(&m_stencil_indices, stencil_size * sizeof(HYPRE_Int));

    // Copy coefficients to managed memory
    std::memcpy(m_coeffs, coeffs_.data(), n_cells * stencil_size * sizeof(double));
    for (int s = 0; s < stencil_size; ++s) {
        m_stencil_indices[s] = s;
    }

    // Ensure data is visible to device
    cudaDeviceSynchronize();

    // Set matrix values using managed memory pointers
    HYPRE_StructMatrixSetBoxValues(A_, ilower_, iupper_,
                                    stencil_size, m_stencil_indices,
                                    m_coeffs);

    HYPRE_StructMatrixAssemble(A_);
    cudaDeviceSynchronize();

    // Clean up temporary managed memory
    cudaFree(m_coeffs);
    cudaFree(m_stencil_indices);
    #else
    // CPU mode: use host vectors directly
    std::vector<HYPRE_Int> stencil_indices(stencil_size);
    for (int s = 0; s < stencil_size; ++s) {
        stencil_indices[s] = s;
    }

    HYPRE_StructMatrixSetBoxValues(A_, ilower_, iupper_,
                                    stencil_size, stencil_indices.data(),
                                    coeffs_.data());

    HYPRE_StructMatrixAssemble(A_);
    #endif

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

    // Relaxation type:
    // - For GPU: use 1 (weighted Jacobi) which is fully parallel and GPU-friendly
    // - For CPU: use 2 (red-black Gauss-Seidel) which converges faster
    #if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_GPU)
    HYPRE_StructPFMGSetRelaxType(solver_, 1);  // Weighted Jacobi for GPU
    #else
    HYPRE_StructPFMGSetRelaxType(solver_, 2);  // RB Gauss-Seidel for CPU
    #endif

    // Number of pre/post relaxation sweeps
    HYPRE_StructPFMGSetNumPreRelax(solver_, 2);
    HYPRE_StructPFMGSetNumPostRelax(solver_, 2);

    // Skip relaxation on fine grid for efficiency
    HYPRE_StructPFMGSetSkipRelax(solver_, 0);

    // Use Galerkin coarse grid operator (0 = Galerkin, 1 = non-Galerkin)
    HYPRE_StructPFMGSetRAPType(solver_, 0);

    // Logging for verbose mode
    if (cfg.verbose) {
        HYPRE_StructPFMGSetLogging(solver_, 1);
        HYPRE_StructPFMGSetPrintLevel(solver_, 2);
    } else {
        HYPRE_StructPFMGSetLogging(solver_, 0);
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
    const size_t n_cells = static_cast<size_t>(Nx) * Ny * Nz;

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_GPU)
    // When HYPRE is in DEVICE mode, we must use managed memory for SetBoxValues
    // Use the pre-allocated managed memory buffers
    double* rhs_vals = rhs_device_;
    double* x_vals = x_device_;

    // Copy RHS and initial guess from ScalarField to managed memory
    for (int k = 0; k < Nz; ++k) {
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                const size_t hypre_idx = (static_cast<size_t>(k) * Ny + j) * Nx + i;
                rhs_vals[hypre_idx] = rhs(i + Ng, j + Ng, mesh_->is2D() ? 0 : k + Ng);
                x_vals[hypre_idx] = p(i + Ng, j + Ng, mesh_->is2D() ? 0 : k + Ng);
            }
        }
    }

    // Handle pinned cell for singular systems
    if (needs_nullspace_handling()) {
        rhs_vals[0] = 0.0;  // Pin cell (0,0,0) to zero
    }

    // Prefetch managed memory to GPU before HYPRE operations
    // This is needed because host-side writes to managed memory stay on CPU pages
    int device;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(rhs_vals, n_cells * sizeof(double), device, 0);
    cudaMemPrefetchAsync(x_vals, n_cells * sizeof(double), device, 0);
    cudaDeviceSynchronize();

    // Set vector values using managed memory
    HYPRE_StructVectorSetBoxValues(b_, ilower_, iupper_, rhs_vals);
    HYPRE_StructVectorSetBoxValues(x_, ilower_, iupper_, x_vals);
    cudaDeviceSynchronize();

    // Solve on GPU
    HYPRE_StructPFMGSolve(solver_, A_, b_, x_);
    cudaDeviceSynchronize();

    // Get iteration count and residual
    HYPRE_Int num_iterations;
    HYPRE_StructPFMGGetNumIterations(solver_, &num_iterations);
    HYPRE_StructPFMGGetFinalRelativeResidualNorm(solver_, &residual_);

    // Copy solution back to managed memory buffer
    HYPRE_StructVectorGetBoxValues(x_, ilower_, iupper_, x_vals);
    cudaDeviceSynchronize();

    // Copy solution to ScalarField
    for (int k = 0; k < Nz; ++k) {
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                const size_t hypre_idx = (static_cast<size_t>(k) * Ny + j) * Nx + i;
                p(i + Ng, j + Ng, mesh_->is2D() ? 0 : k + Ng) = x_vals[hypre_idx];
            }
        }
    }
#else
    // HOST mode: use standard vectors
    std::vector<double> rhs_values(n_cells);
    std::vector<double> x_values(n_cells);

    for (int k = 0; k < Nz; ++k) {
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                const size_t hypre_idx = (static_cast<size_t>(k) * Ny + j) * Nx + i;
                rhs_values[hypre_idx] = rhs(i + Ng, j + Ng, mesh_->is2D() ? 0 : k + Ng);
                x_values[hypre_idx] = p(i + Ng, j + Ng, mesh_->is2D() ? 0 : k + Ng);
            }
        }
    }

    // Handle pinned cell for singular systems
    if (needs_nullspace_handling()) {
        rhs_values[0] = 0.0;  // Pin cell (0,0,0) to zero
    }

    // Set vector values
    HYPRE_StructVectorSetBoxValues(b_, ilower_, iupper_, rhs_values.data());
    HYPRE_StructVectorSetBoxValues(x_, ilower_, iupper_, x_values.data());

    // Solve
    HYPRE_StructPFMGSolve(solver_, A_, b_, x_);

    // Get iteration count and residual
    HYPRE_Int num_iterations;
    HYPRE_StructPFMGGetNumIterations(solver_, &num_iterations);
    HYPRE_StructPFMGGetFinalRelativeResidualNorm(solver_, &residual_);

    // Copy solution back to ScalarField
    HYPRE_StructVectorGetBoxValues(x_, ilower_, iupper_, x_values.data());

    for (int k = 0; k < Nz; ++k) {
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                const size_t hypre_idx = (static_cast<size_t>(k) * Ny + j) * Nx + i;
                p(i + Ng, j + Ng, mesh_->is2D() ? 0 : k + Ng) = x_values[hypre_idx];
            }
        }
    }
#endif

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

    // Get managed memory buffer pointers
    double* rhs_dev = rhs_device_;
    double* x_dev = x_device_;

    // Pack RHS and initial guess from ghost-cell layout to packed layout on GPU
    // Note: rhs_ptr and p_ptr are OMP-mapped arrays (use present)
    //       rhs_dev and x_dev are managed memory (use is_device_ptr)
    const size_t total_size = static_cast<size_t>(Nx_full) * Ny_full * (is2D ? 1 : (Nz + 2 * Ng));
    #pragma omp target teams distribute parallel for collapse(3) \
        map(present: rhs_ptr[0:total_size], p_ptr[0:total_size]) \
        is_device_ptr(rhs_dev, x_dev)
    for (int k = 0; k < Nz; ++k) {
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                const size_t hypre_idx = (static_cast<size_t>(k) * Ny + j) * Nx + i;
                const int kk = is2D ? 0 : k + Ng;
                const size_t full_idx = static_cast<size_t>(kk) * Nx_full * Ny_full +
                                        (j + Ng) * Nx_full + (i + Ng);
                rhs_dev[hypre_idx] = rhs_ptr[full_idx];
                x_dev[hypre_idx] = p_ptr[full_idx];
            }
        }
    }

    // Handle pinned cell for singular systems (on device)
    if (needs_nullspace_handling()) {
        #pragma omp target is_device_ptr(rhs_dev)
        {
            rhs_dev[0] = 0.0;
        }
    }

    // Synchronize to ensure OMP kernels complete before HYPRE
    cudaDeviceSynchronize();

    // Set HYPRE vectors from managed memory buffers
    HYPRE_StructVectorSetBoxValues(b_, ilower_, iupper_, rhs_dev);
    HYPRE_StructVectorSetBoxValues(x_, ilower_, iupper_, x_dev);

    // Solve on GPU using HYPRE PFMG
    HYPRE_StructPFMGSolve(solver_, A_, b_, x_);
    cudaDeviceSynchronize();

    // Get iteration count and residual
    HYPRE_Int num_iterations;
    HYPRE_StructPFMGGetNumIterations(solver_, &num_iterations);
    HYPRE_StructPFMGGetFinalRelativeResidualNorm(solver_, &residual_);

    // Get solution back to managed memory buffer
    HYPRE_StructVectorGetBoxValues(x_, ilower_, iupper_, x_dev);
    cudaDeviceSynchronize();

    // Unpack solution from packed layout to ghost-cell layout on GPU
    #pragma omp target teams distribute parallel for collapse(3) \
        map(present: p_ptr[0:total_size]) is_device_ptr(x_dev)
    for (int k = 0; k < Nz; ++k) {
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                const size_t hypre_idx = (static_cast<size_t>(k) * Ny + j) * Nx + i;
                const int kk = is2D ? 0 : k + Ng;
                const size_t full_idx = static_cast<size_t>(kk) * Nx_full * Ny_full +
                                        (j + Ng) * Nx_full + (i + Ng);
                p_ptr[full_idx] = x_dev[hypre_idx];
            }
        }
    }

    // Apply boundary conditions to ghost cells on GPU
    // X boundaries
    #pragma omp target teams distribute parallel for collapse(2) map(present: p_ptr[0:total_size])
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
    #pragma omp target teams distribute parallel for collapse(2) map(present: p_ptr[0:total_size])
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
        #pragma omp target teams distribute parallel for collapse(2) map(present: p_ptr[0:total_size])
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
