/// @file poisson_solver_multigrid.cpp
/// @brief Geometric multigrid solver for pressure Poisson equation
///
/// This file implements a V-cycle geometric multigrid solver achieving O(N)
/// complexity for the pressure correction equation in the fractional-step method.
/// Key features:
/// - V-cycle algorithm with Chebyshev or Jacobi smoothing
/// - Automatic mesh hierarchy construction (restriction to coarsest level)
/// - Full weighting restriction and bilinear/trilinear prolongation
/// - GPU-accelerated smoothing and residual computation
/// - Fused residual + norm computation for efficient convergence checking
/// - L2 norm convergence with L∞ safety cap for robustness
/// - 10-100x faster than pure SOR iteration for large grids
///
/// The solver constructs a hierarchy of grids by recursive coarsening and solves
/// the system using recursive V-cycles that combine smoothing on each level with
/// coarse-grid correction.
///
/// GPU Synchronization & CUDA Graphs:
/// -----------------------------------
/// The MG kernels have data dependencies (each operation reads results from the
/// previous one). Using OpenMP `nowait` on target regions causes race conditions
/// because nvhpc may execute deferred tasks on different CUDA streams.
///
/// Solution: V-cycle CUDA Graphs (DEFAULT in GPU builds with NVHPC)
/// - Captures entire V-cycle kernel sequence as a single CUDA graph
/// - Replays with one graph launch, eliminating per-kernel sync overhead
/// - See initialize_vcycle_graph() and vcycle_graphed() for implementation
/// - Disable via config.poisson_use_vcycle_graph = false if needed
///
/// Fallback (non-NVHPC compilers or when graphs disabled):
/// - All GPU kernels use synchronous execution (implicit barrier after each)
/// - ~37% of GPU API time spent in cudaStreamSynchronize (from Nsys profiling)
///
/// Alternative approaches (not implemented):
/// - OpenMP depend clauses: Express dependencies explicitly (complex)
/// - Custom CUDA streams: Ensure all kernels use same stream (non-portable)

#include "poisson_solver_multigrid.hpp"
#include "gpu_utils.hpp"
#include "profiling.hpp"
#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#include "mg_cuda_kernels.hpp"
#include <cuda_runtime.h>
// NVHPC provides ompx_get_cuda_stream() to get the CUDA stream used by OpenMP.
// This allows launching CUDA Graphs on the same stream, eliminating sync overhead.
// The function returns void* which we cast to cudaStream_t.

// CUDA error checking macro for synchronization calls
#define CUDA_CHECK_SYNC(call)                                                  \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            throw std::runtime_error(std::string("[MG] CUDA sync error: ") +   \
                                     cudaGetErrorString(err) + " at " +        \
                                     __FILE__ + ":" + std::to_string(__LINE__)); \
        }                                                                      \
    } while (0)
#endif
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <limits>   // for std::numeric_limits (NaN handling)
#include <cstdlib>  // for std::getenv

namespace nncfd {

// ============================================================================
// Chebyshev Eigenvalue Bounds
// ============================================================================
// Conservative eigenvalue bounds for D^{-1}*A where D = diag(A).
// For the 7-point discrete Laplacian, the true eigenvalues are in (0, 2).
// We use slightly narrower bounds [0.05, 1.95] for numerical stability.
// Note: Keep in sync with mg_cuda_kernels.cpp (duplicated for CPU build isolation)
constexpr double CHEBYSHEV_LAMBDA_MIN = 0.05;
constexpr double CHEBYSHEV_LAMBDA_MAX = 1.95;

MultigridPoissonSolver::MultigridPoissonSolver(const Mesh& mesh) : mesh_(&mesh) {
    create_hierarchy();

    // Initialize GPU buffers (maps to device) OR set up raw pointers for CPU
    // This enables unified loops to use cached pointers on both CPU and GPU
    initialize_gpu_buffers();

    // V-cycle CUDA Graph is enabled by default (use_vcycle_graph_ = true in header)
    // Can be disabled via config.poisson_use_vcycle_graph = false
}

MultigridPoissonSolver::~MultigridPoissonSolver() {
    cleanup_gpu_buffers();  // Safe to call unconditionally (no-op when GPU disabled)
}

void MultigridPoissonSolver::set_bc(PoissonBC x_lo, PoissonBC x_hi,
                                     PoissonBC y_lo, PoissonBC y_hi) {
    bc_x_lo_ = x_lo;
    bc_x_hi_ = x_hi;
    bc_y_lo_ = y_lo;
    bc_y_hi_ = y_hi;
    // Keep z BCs at default (periodic) for 2D compatibility

#ifdef USE_GPU_OFFLOAD
    // Invalidate V-cycle graph so it gets recaptured with new BCs
    if (vcycle_graph_) {
        vcycle_graph_->destroy();
        vcycle_graph_.reset();
    }
#endif
}

void MultigridPoissonSolver::set_bc(PoissonBC x_lo, PoissonBC x_hi,
                                     PoissonBC y_lo, PoissonBC y_hi,
                                     PoissonBC z_lo, PoissonBC z_hi) {
    bc_x_lo_ = x_lo;
    bc_x_hi_ = x_hi;
    bc_y_lo_ = y_lo;
    bc_y_hi_ = y_hi;
    bc_z_lo_ = z_lo;
    bc_z_hi_ = z_hi;

#ifdef USE_GPU_OFFLOAD
    // Invalidate V-cycle graph so it gets recaptured with new BCs
    if (vcycle_graph_) {
        vcycle_graph_->destroy();
        vcycle_graph_.reset();
    }
#endif
}

void MultigridPoissonSolver::create_hierarchy() {
    // Create grid hierarchy from fine to coarse
    int Nx = mesh_->Nx;
    int Ny = mesh_->Ny;
    int Nz = mesh_->Nz;
    double dx = mesh_->dx;
    double dy = mesh_->dy;
    double dz = mesh_->dz;
    const bool is_2d = mesh_->is2D();

    // Finest level uses mesh's ghost width (may be >1 for O4 schemes)
    // Coarse levels use Ng=1 since MG internally uses O2 stencils
    const int ng_fine = mesh_->Nghost;
    constexpr int ng_coarse = 1;

    // Finest level
    if (is_2d) {
        levels_.push_back(std::make_unique<GridLevel>(Nx, Ny, dx, dy, ng_fine));
    } else {
        levels_.push_back(std::make_unique<GridLevel>(Nx, Ny, Nz, dx, dy, dz, ng_fine));
    }

    // Coarsen until we reach minimum grid size
    // 8x8 coarsest is GPU-friendly while still giving good MG efficiency
    // For 128x128: gives 4 levels (128→64→32→16→8) instead of 3
    constexpr int MIN_COARSE_SIZE = 8;

    if (is_2d) {
        while (Nx > MIN_COARSE_SIZE && Ny > MIN_COARSE_SIZE) {
            Nx /= 2;
            Ny /= 2;
            dx *= 2.0;
            dy *= 2.0;
            levels_.push_back(std::make_unique<GridLevel>(Nx, Ny, dx, dy, ng_coarse));
        }
    } else {
        while (Nx > MIN_COARSE_SIZE && Ny > MIN_COARSE_SIZE && Nz > MIN_COARSE_SIZE) {
            Nx /= 2;
            Ny /= 2;
            Nz /= 2;
            dx *= 2.0;
            dy *= 2.0;
            dz *= 2.0;
            levels_.push_back(std::make_unique<GridLevel>(Nx, Ny, Nz, dx, dy, dz, ng_coarse));
        }
    }
}

void MultigridPoissonSolver::apply_bc(int level) {
    NVTX_SCOPE_BC("mg:apply_bc");
    // UNIFIED CPU/GPU implementation for boundary conditions
    // Uses raw pointers and identical arithmetic for bitwise consistency
    auto& grid = *levels_[level];
    const int Nx = grid.Nx;
    const int Ny = grid.Ny;
    const int Nz = grid.Nz;
    const int Ng = grid.Ng;  // Ghost cells from level (may be >1 for finest level)
    const bool is_2d = grid.is2D();
    const int stride = grid.stride;
    const int plane_stride = grid.plane_stride;

    // Convert BCs to integers for branchless GPU code
    const int bc_x_lo = static_cast<int>(bc_x_lo_);
    const int bc_x_hi = static_cast<int>(bc_x_hi_);
    const int bc_y_lo = static_cast<int>(bc_y_lo_);
    const int bc_y_hi = static_cast<int>(bc_y_hi_);
    const int bc_z_lo = static_cast<int>(bc_z_lo_);
    const int bc_z_hi = static_cast<int>(bc_z_hi_);
    const double dval = dirichlet_val_;

    // NVHPC WORKAROUND: Use omp_get_mapped_ptr for actual device addresses
#ifdef USE_GPU_OFFLOAD
    int device = omp_get_default_device();
    double* u_ptr = static_cast<double*>(omp_get_mapped_ptr(u_ptrs_[level], device));
    #define BC_TARGET_FOR_X \
        _Pragma("omp target teams distribute parallel for is_device_ptr(u_ptr)")
    #define BC_TARGET_FOR_Y \
        _Pragma("omp target teams distribute parallel for is_device_ptr(u_ptr)")
    #define BC_TARGET_FOR_Z \
        _Pragma("omp target teams distribute parallel for is_device_ptr(u_ptr)")
#else
    double* u_ptr = u_ptrs_[level];
    #define BC_TARGET_FOR_X
    #define BC_TARGET_FOR_Y
    #define BC_TARGET_FOR_Z
#endif

    if (is_2d) {
        // ========== 2D BOUNDARY CONDITIONS ==========
        // Fill ALL Ng ghost layers (important when Ng > 1 for O4 stencils)
        // Pass 1: x-direction boundaries (loop over j and all ghost layers)
        BC_TARGET_FOR_X
        for (int j = 0; j < Ny + 2*Ng; ++j) {
            for (int g = 0; g < Ng; ++g) {
                int idx_lo = j * stride + g;               // Left ghosts (i=0,1,...,Ng-1)
                int idx_hi = j * stride + (Nx + Ng + g);   // Right ghosts (i=Nx+Ng,...,Nx+2Ng-1)

                // Left boundary - periodic wraps to right interior
                if (bc_x_lo == 2) { // Periodic
                    u_ptr[idx_lo] = u_ptr[j * stride + Nx + g];
                } else if (bc_x_lo == 1) { // Neumann (zero gradient)
                    u_ptr[idx_lo] = u_ptr[j * stride + Ng];
                } else { // Dirichlet
                    u_ptr[idx_lo] = 2.0 * dval - u_ptr[j * stride + Ng];
                }

                // Right boundary - periodic wraps to left interior
                if (bc_x_hi == 2) { // Periodic
                    u_ptr[idx_hi] = u_ptr[j * stride + Ng + g];
                } else if (bc_x_hi == 1) { // Neumann (zero gradient)
                    u_ptr[idx_hi] = u_ptr[j * stride + (Nx + Ng - 1)];
                } else { // Dirichlet
                    u_ptr[idx_hi] = 2.0 * dval - u_ptr[j * stride + (Nx + Ng - 1)];
                }
            }
        }

        // Pass 2: y-direction boundaries (loop over i and all ghost layers)
        BC_TARGET_FOR_Y
        for (int i = 0; i < Nx + 2*Ng; ++i) {
            for (int g = 0; g < Ng; ++g) {
                int idx_lo = g * stride + i;               // Bottom ghosts (j=0,1,...,Ng-1)
                int idx_hi = (Ny + Ng + g) * stride + i;   // Top ghosts (j=Ny+Ng,...,Ny+2Ng-1)

                // Bottom boundary - periodic wraps to top interior
                if (bc_y_lo == 2) { // Periodic
                    u_ptr[idx_lo] = u_ptr[(Ny + g) * stride + i];
                } else if (bc_y_lo == 1) { // Neumann (zero gradient)
                    u_ptr[idx_lo] = u_ptr[Ng * stride + i];
                } else { // Dirichlet
                    u_ptr[idx_lo] = 2.0 * dval - u_ptr[Ng * stride + i];
                }

                // Top boundary - periodic wraps to bottom interior
                if (bc_y_hi == 2) { // Periodic
                    u_ptr[idx_hi] = u_ptr[(Ng + g) * stride + i];
                } else if (bc_y_hi == 1) { // Neumann (zero gradient)
                    u_ptr[idx_hi] = u_ptr[(Ny + Ng - 1) * stride + i];
                } else { // Dirichlet
                    u_ptr[idx_hi] = 2.0 * dval - u_ptr[(Ny + Ng - 1) * stride + i];
                }
            }
        }

        // Pass 3: Re-apply x-direction for corner consistency (periodic BCs)
        // This ensures corners get correct values when both x and y are periodic
        const bool needs_corner_fix = (bc_x_lo == 2 || bc_x_hi == 2 || bc_y_lo == 2 || bc_y_hi == 2);
        if (needs_corner_fix) {
            BC_TARGET_FOR_X
            for (int j = 0; j < Ny + 2*Ng; ++j) {
                for (int g = 0; g < Ng; ++g) {
                    int idx_lo = j * stride + g;
                    int idx_hi = j * stride + (Nx + Ng + g);

                    if (bc_x_lo == 2) {
                        u_ptr[idx_lo] = u_ptr[j * stride + Nx + g];
                    } else if (bc_x_lo == 1) {
                        u_ptr[idx_lo] = u_ptr[j * stride + Ng];
                    } else {
                        u_ptr[idx_lo] = 2.0 * dval - u_ptr[j * stride + Ng];
                    }

                    if (bc_x_hi == 2) {
                        u_ptr[idx_hi] = u_ptr[j * stride + Ng + g];
                    } else if (bc_x_hi == 1) {
                        u_ptr[idx_hi] = u_ptr[j * stride + (Nx + Ng - 1)];
                    } else {
                        u_ptr[idx_hi] = 2.0 * dval - u_ptr[j * stride + (Nx + Ng - 1)];
                    }
                }
            }

            BC_TARGET_FOR_Y
            for (int i = 0; i < Nx + 2*Ng; ++i) {
                for (int g = 0; g < Ng; ++g) {
                    int idx_lo = g * stride + i;
                    int idx_hi = (Ny + Ng + g) * stride + i;

                    if (bc_y_lo == 2) {
                        u_ptr[idx_lo] = u_ptr[(Ny + g) * stride + i];
                    } else if (bc_y_lo == 1) {
                        u_ptr[idx_lo] = u_ptr[Ng * stride + i];
                    } else {
                        u_ptr[idx_lo] = 2.0 * dval - u_ptr[Ng * stride + i];
                    }

                    if (bc_y_hi == 2) {
                        u_ptr[idx_hi] = u_ptr[(Ng + g) * stride + i];
                    } else if (bc_y_hi == 1) {
                        u_ptr[idx_hi] = u_ptr[(Ny + Ng - 1) * stride + i];
                    } else {
                        u_ptr[idx_hi] = 2.0 * dval - u_ptr[(Ny + Ng - 1) * stride + i];
                    }
                }
            }
        }
    } else {
        // ========== 3D BOUNDARY CONDITIONS ==========
        // Fill ALL Ng ghost layers (important when Ng > 1 for O4 stencils)
        const int n_jk = (Ny + 2*Ng) * (Nz + 2*Ng);
        const int n_ik = (Nx + 2*Ng) * (Nz + 2*Ng);
        const int n_ij = (Nx + 2*Ng) * (Ny + 2*Ng);

        // Pass 1: x-direction boundaries (loop over j,k faces and all ghost layers)
        BC_TARGET_FOR_X
        for (int jk = 0; jk < n_jk; ++jk) {
            int j = jk % (Ny + 2*Ng);
            int k = jk / (Ny + 2*Ng);
            for (int g = 0; g < Ng; ++g) {
                int idx_lo = k * plane_stride + j * stride + g;
                int idx_hi = k * plane_stride + j * stride + (Nx + Ng + g);

                // Left boundary - periodic wraps to right interior
                if (bc_x_lo == 2) {
                    u_ptr[idx_lo] = u_ptr[k * plane_stride + j * stride + Nx + g];
                } else if (bc_x_lo == 1) {
                    u_ptr[idx_lo] = u_ptr[k * plane_stride + j * stride + Ng];
                } else {
                    u_ptr[idx_lo] = 2.0 * dval - u_ptr[k * plane_stride + j * stride + Ng];
                }

                // Right boundary - periodic wraps to left interior
                if (bc_x_hi == 2) {
                    u_ptr[idx_hi] = u_ptr[k * plane_stride + j * stride + Ng + g];
                } else if (bc_x_hi == 1) {
                    u_ptr[idx_hi] = u_ptr[k * plane_stride + j * stride + (Nx + Ng - 1)];
                } else {
                    u_ptr[idx_hi] = 2.0 * dval - u_ptr[k * plane_stride + j * stride + (Nx + Ng - 1)];
                }
            }
        }

        // Pass 2: y-direction boundaries (loop over i,k faces and all ghost layers)
        BC_TARGET_FOR_Y
        for (int ik = 0; ik < n_ik; ++ik) {
            int i = ik % (Nx + 2*Ng);
            int k = ik / (Nx + 2*Ng);
            for (int g = 0; g < Ng; ++g) {
                int idx_lo = k * plane_stride + g * stride + i;
                int idx_hi = k * plane_stride + (Ny + Ng + g) * stride + i;

                // Bottom boundary - periodic wraps to top interior
                if (bc_y_lo == 2) {
                    u_ptr[idx_lo] = u_ptr[k * plane_stride + (Ny + g) * stride + i];
                } else if (bc_y_lo == 1) {
                    u_ptr[idx_lo] = u_ptr[k * plane_stride + Ng * stride + i];
                } else {
                    u_ptr[idx_lo] = 2.0 * dval - u_ptr[k * plane_stride + Ng * stride + i];
                }

                // Top boundary - periodic wraps to bottom interior
                if (bc_y_hi == 2) {
                    u_ptr[idx_hi] = u_ptr[k * plane_stride + (Ng + g) * stride + i];
                } else if (bc_y_hi == 1) {
                    u_ptr[idx_hi] = u_ptr[k * plane_stride + (Ny + Ng - 1) * stride + i];
                } else {
                    u_ptr[idx_hi] = 2.0 * dval - u_ptr[k * plane_stride + (Ny + Ng - 1) * stride + i];
                }
            }
        }

        // Pass 3: z-direction boundaries (loop over i,j faces and all ghost layers)
        BC_TARGET_FOR_Z
        for (int ij = 0; ij < n_ij; ++ij) {
            int i = ij % (Nx + 2*Ng);
            int j = ij / (Nx + 2*Ng);
            for (int g = 0; g < Ng; ++g) {
                int idx_lo = g * plane_stride + j * stride + i;
                int idx_hi = (Nz + Ng + g) * plane_stride + j * stride + i;

                // Back boundary - periodic wraps to front interior
                if (bc_z_lo == 2) {
                    u_ptr[idx_lo] = u_ptr[(Nz + g) * plane_stride + j * stride + i];
                } else if (bc_z_lo == 1) {
                    u_ptr[idx_lo] = u_ptr[Ng * plane_stride + j * stride + i];
                } else {
                    u_ptr[idx_lo] = 2.0 * dval - u_ptr[Ng * plane_stride + j * stride + i];
                }

                // Front boundary - periodic wraps to back interior
                if (bc_z_hi == 2) {
                    u_ptr[idx_hi] = u_ptr[(Ng + g) * plane_stride + j * stride + i];
                } else if (bc_z_hi == 1) {
                    u_ptr[idx_hi] = u_ptr[(Nz + Ng - 1) * plane_stride + j * stride + i];
                } else {
                    u_ptr[idx_hi] = 2.0 * dval - u_ptr[(Nz + Ng - 1) * plane_stride + j * stride + i];
                }
            }
        }
    }

    // Clean up macros
    #undef BC_TARGET_FOR_X
    #undef BC_TARGET_FOR_Y
    #undef BC_TARGET_FOR_Z
}

void MultigridPoissonSolver::apply_bc_to_residual(int level) {
    NVTX_SCOPE_BC("mg:apply_bc_residual");
    // Apply boundary conditions to the residual array for proper restriction
    // The 9-point restriction stencil reads from ghost cells, so they must be set
    auto& grid = *levels_[level];
    const int Nx = grid.Nx;
    const int Ny = grid.Ny;
    const int Nz = grid.Nz;
    const int Ng = grid.Ng;  // Use level's ghost width
    const bool is_2d = grid.is2D();

#ifdef USE_GPU_OFFLOAD
    // GPU path for residual BCs
    if (gpu_ready_) {
        const int stride = Nx + 2*Ng;

        // NVHPC WORKAROUND: Use omp_get_mapped_ptr for actual device addresses
        int device = omp_get_default_device();
        double* r_ptr = static_cast<double*>(omp_get_mapped_ptr(r_ptrs_[level], device));

        // Convert BCs to integers for GPU
        const int bc_x_lo = static_cast<int>(bc_x_lo_);
        const int bc_x_hi = static_cast<int>(bc_x_hi_);
        const int bc_y_lo = static_cast<int>(bc_y_lo_);
        const int bc_y_hi = static_cast<int>(bc_y_hi_);
        const int bc_z_lo = static_cast<int>(bc_z_lo_);
        const int bc_z_hi = static_cast<int>(bc_z_hi_);

        if (is_2d) {
            // 2D GPU path for residual - fill ALL Ng ghost layers
            // x-direction boundaries
            #pragma omp target teams distribute parallel for is_device_ptr(r_ptr)
            for (int j = 0; j < Ny + 2*Ng; ++j) {
                int idx = j * stride;
                for (int g = 0; g < Ng; ++g) {
                    // Left ghost at position g
                    if (bc_x_lo == 2) { // Periodic
                        r_ptr[idx + g] = r_ptr[idx + Nx + g];
                    } else { // Neumann/Dirichlet: zero ghost
                        r_ptr[idx + g] = 0.0;
                    }
                    // Right ghost at position Nx+Ng+g
                    if (bc_x_hi == 2) { // Periodic
                        r_ptr[idx + Nx + Ng + g] = r_ptr[idx + Ng + g];
                    } else {
                        r_ptr[idx + Nx + Ng + g] = 0.0;
                    }
                }
            }

            // y-direction boundaries
            #pragma omp target teams distribute parallel for is_device_ptr(r_ptr)
            for (int i = 0; i < Nx + 2*Ng; ++i) {
                for (int g = 0; g < Ng; ++g) {
                    // Bottom ghost at position g
                    if (bc_y_lo == 2) { // Periodic
                        r_ptr[g * stride + i] = r_ptr[(Ny + g) * stride + i];
                    } else {
                        r_ptr[g * stride + i] = 0.0;
                    }
                    // Top ghost at position Ny+Ng+g
                    if (bc_y_hi == 2) { // Periodic
                        r_ptr[(Ny + Ng + g) * stride + i] = r_ptr[(Ng + g) * stride + i];
                    } else {
                        r_ptr[(Ny + Ng + g) * stride + i] = 0.0;
                    }
                }
            }
        } else {
            // 3D GPU path for residual - unified kernel matching apply_bc
            const int plane_stride = stride * (Ny + 2*Ng);
            const int Nx_g = Nx + 2*Ng;
            const int Ny_g = Ny + 2*Ng;
            const int Nz_g = Nz + 2*Ng;
            const int n_total_g = Nx_g * Ny_g * Nz_g;

            #pragma omp target teams distribute parallel for is_device_ptr(r_ptr) \
                firstprivate(Nx, Ny, Nz, Ng, stride, plane_stride, bc_x_lo, bc_x_hi, bc_y_lo, bc_y_hi, bc_z_lo, bc_z_hi)
            for (int idx_g = 0; idx_g < n_total_g; ++idx_g) {
                int i = idx_g % Nx_g;
                int j = (idx_g / Nx_g) % Ny_g;
                int k = idx_g / (Nx_g * Ny_g);

                // Skip interior points
                if (i >= Ng && i < Nx + Ng && j >= Ng && j < Ny + Ng && k >= Ng && k < Nz + Ng) {
                    continue;
                }

                int cell_idx = k * plane_stride + j * stride + i;

                // X-direction boundaries (residual: zero for non-periodic)
                if (i < Ng) { // Left boundary
                    if (bc_x_lo == 2) { // Periodic
                        r_ptr[cell_idx] = r_ptr[k * plane_stride + j * stride + (i + Nx)];
                    } else {
                        r_ptr[cell_idx] = 0.0;
                    }
                } else if (i >= Nx + Ng) { // Right boundary
                    if (bc_x_hi == 2) { // Periodic
                        r_ptr[cell_idx] = r_ptr[k * plane_stride + j * stride + (i - Nx)];
                    } else {
                        r_ptr[cell_idx] = 0.0;
                    }
                }

                // Y-direction boundaries (may overwrite x-boundary corners)
                if (j < Ng) { // Bottom boundary
                    if (bc_y_lo == 2) { // Periodic
                        r_ptr[cell_idx] = r_ptr[k * plane_stride + (j + Ny) * stride + i];
                    } else {
                        r_ptr[cell_idx] = 0.0;
                    }
                } else if (j >= Ny + Ng) { // Top boundary
                    if (bc_y_hi == 2) { // Periodic
                        r_ptr[cell_idx] = r_ptr[k * plane_stride + (j - Ny) * stride + i];
                    } else {
                        r_ptr[cell_idx] = 0.0;
                    }
                }

                // Z-direction boundaries (may overwrite x/y-boundary corners)
                if (k < Ng) { // Back boundary
                    if (bc_z_lo == 2) { // Periodic
                        r_ptr[cell_idx] = r_ptr[(k + Nz) * plane_stride + j * stride + i];
                    } else {
                        r_ptr[cell_idx] = 0.0;
                    }
                } else if (k >= Nz + Ng) { // Front boundary
                    if (bc_z_hi == 2) { // Periodic
                        r_ptr[cell_idx] = r_ptr[(k - Nz) * plane_stride + j * stride + i];
                    } else {
                        r_ptr[cell_idx] = 0.0;
                    }
                }
            }
        }
        return;  // GPU path done
    }
#endif

    // CPU fallback path
    // Similar logic to apply_bc but on grid.r instead of grid.u
    // Fill ALL Ng ghost layers for each boundary
    if (is_2d) {
        // 2D CPU path for residual
        // x-direction boundaries
        for (int j = 0; j < Ny + 2*Ng; ++j) {
            for (int g = 0; g < Ng; ++g) {
                // Left ghosts
                switch (bc_x_lo_) {
                    case PoissonBC::Periodic:
                        grid.r(g, j) = grid.r(Nx + g, j);
                        break;
                    case PoissonBC::Neumann:
                    case PoissonBC::Dirichlet:
                        grid.r(g, j) = 0.0;  // Zero ghost for Neumann/Dirichlet residual
                        break;
                }
                // Right ghosts
                switch (bc_x_hi_) {
                    case PoissonBC::Periodic:
                        grid.r(Nx + Ng + g, j) = grid.r(Ng + g, j);
                        break;
                    case PoissonBC::Neumann:
                    case PoissonBC::Dirichlet:
                        grid.r(Nx + Ng + g, j) = 0.0;
                        break;
                }
            }
        }
        // y-direction boundaries
        for (int i = 0; i < Nx + 2*Ng; ++i) {
            for (int g = 0; g < Ng; ++g) {
                // Bottom ghosts
                switch (bc_y_lo_) {
                    case PoissonBC::Periodic:
                        grid.r(i, g) = grid.r(i, Ny + g);
                        break;
                    case PoissonBC::Neumann:
                    case PoissonBC::Dirichlet:
                        grid.r(i, g) = 0.0;
                        break;
                }
                // Top ghosts
                switch (bc_y_hi_) {
                    case PoissonBC::Periodic:
                        grid.r(i, Ny + Ng + g) = grid.r(i, Ng + g);
                        break;
                    case PoissonBC::Neumann:
                    case PoissonBC::Dirichlet:
                        grid.r(i, Ny + Ng + g) = 0.0;
                        break;
                }
            }
        }
    } else {
        // 3D CPU path for residual
        // x-direction
        for (int k = 0; k < Nz + 2*Ng; ++k) {
            for (int j = 0; j < Ny + 2*Ng; ++j) {
                for (int g = 0; g < Ng; ++g) {
                    switch (bc_x_lo_) {
                        case PoissonBC::Periodic:
                            grid.r(g, j, k) = grid.r(Nx + g, j, k);
                            break;
                        default:
                            grid.r(g, j, k) = 0.0;
                            break;
                    }
                    switch (bc_x_hi_) {
                        case PoissonBC::Periodic:
                            grid.r(Nx + Ng + g, j, k) = grid.r(Ng + g, j, k);
                            break;
                        default:
                            grid.r(Nx + Ng + g, j, k) = 0.0;
                            break;
                    }
                }
            }
        }
        // y-direction
        for (int k = 0; k < Nz + 2*Ng; ++k) {
            for (int i = 0; i < Nx + 2*Ng; ++i) {
                for (int g = 0; g < Ng; ++g) {
                    switch (bc_y_lo_) {
                        case PoissonBC::Periodic:
                            grid.r(i, g, k) = grid.r(i, Ny + g, k);
                            break;
                        default:
                            grid.r(i, g, k) = 0.0;
                            break;
                    }
                    switch (bc_y_hi_) {
                        case PoissonBC::Periodic:
                            grid.r(i, Ny + Ng + g, k) = grid.r(i, Ng + g, k);
                            break;
                        default:
                            grid.r(i, Ny + Ng + g, k) = 0.0;
                            break;
                    }
                }
            }
        }
        // z-direction
        for (int j = 0; j < Ny + 2*Ng; ++j) {
            for (int i = 0; i < Nx + 2*Ng; ++i) {
                for (int g = 0; g < Ng; ++g) {
                    switch (bc_z_lo_) {
                        case PoissonBC::Periodic:
                            grid.r(i, j, g) = grid.r(i, j, Nz + g);
                            break;
                        default:
                            grid.r(i, j, g) = 0.0;
                            break;
                    }
                    switch (bc_z_hi_) {
                        case PoissonBC::Periodic:
                            grid.r(i, j, Nz + Ng + g) = grid.r(i, j, Ng + g);
                            break;
                        default:
                            grid.r(i, j, Nz + Ng + g) = 0.0;
                            break;
                    }
                }
            }
        }
    }
}

void MultigridPoissonSolver::smooth_chebyshev(int level, int degree) {
    // Chebyshev polynomial acceleration on Jacobi smoother
    // Uses ping-pong buffers for BC consistency: every A*u sees BC-valid ghosts
    //
    // Algorithm (Richardson-Chebyshev):
    //   For each step k = 0..degree-1:
    //     1. apply_bc(u) - ensure ghost cells are valid
    //     2. Compute Jacobi update: u_jacobi = (neighbors - f) / diag
    //     3. Update with Chebyshev weight: u_new = (1-ω_k)*u + ω_k*u_jacobi
    //     4. Copy to u buffer for next iteration
    //
    // The Chebyshev-optimal weights are: ω_k = 1/(d - c*cos(θ_k))
    // where θ_k = π*(2k+1)/(2*degree), d = (λ_max+λ_min)/2, c = (λ_max-λ_min)/2
    NVTX_SCOPE_POISSON("mg:smooth_chebyshev");

    auto& grid = *levels_[level];
    const double dx2 = grid.dx * grid.dx;
    const double dy2 = grid.dy * grid.dy;
    const double dz2 = grid.dz * grid.dz;
    const bool is_2d = grid.is2D();
    const double coeff = is_2d ? (2.0 / dx2 + 2.0 / dy2)
                               : (2.0 / dx2 + 2.0 / dy2 + 2.0 / dz2);

    const int Ng = grid.Ng;  // Use level's ghost width
    const int Nx = grid.Nx;
    const int Ny = grid.Ny;
    const int Nz = grid.Nz;
    const int stride = grid.stride;
    const int plane_stride = grid.plane_stride;
    const size_t total_size = grid.total_size;

    // Chebyshev eigenvalue bounds (see constants at top of file)
    const double d = (CHEBYSHEV_LAMBDA_MAX + CHEBYSHEV_LAMBDA_MIN) / 2.0;
    const double c = (CHEBYSHEV_LAMBDA_MAX - CHEBYSHEV_LAMBDA_MIN) / 2.0;

    // NVHPC WORKAROUND: Use omp_get_mapped_ptr to get actual device addresses.
    // Local pointer copies with map(present:) get HOST addresses in NVHPC.
#ifdef USE_GPU_OFFLOAD
    int device = omp_get_default_device();
    double* u_ptr = static_cast<double*>(omp_get_mapped_ptr(u_ptrs_[level], device));
    const double* f_ptr = static_cast<const double*>(omp_get_mapped_ptr(f_ptrs_[level], device));
    double* tmp_ptr = tmp_ptrs_[level];  // Already a raw device pointer (omp_target_alloc)
    // Note: Cannot use nowait here - Chebyshev iterations have data dependencies
    // Each iteration reads the result of the previous iteration
    #define CHEBY_TARGET_2D \
        _Pragma("omp target teams distribute parallel for collapse(2) is_device_ptr(u_ptr, f_ptr, tmp_ptr)")
    #define CHEBY_TARGET_3D \
        _Pragma("omp target teams distribute parallel for collapse(3) is_device_ptr(u_ptr, f_ptr, tmp_ptr)")
    #define CHEBY_TARGET_COPY \
        _Pragma("omp target teams distribute parallel for is_device_ptr(u_ptr, tmp_ptr)")
#else
    double* u_ptr = u_ptrs_[level];
    const double* f_ptr = f_ptrs_[level];
    double* tmp_ptr = r_ptrs_[level];  // Reuse r as scratch buffer on CPU
    #define CHEBY_TARGET_2D
    #define CHEBY_TARGET_3D
    #define CHEBY_TARGET_COPY
#endif

    // Chebyshev-Jacobi iteration with ping-pong
    for (int k = 0; k < degree; ++k) {
        // Step 1: Apply BC before operator application (ghost fill)
        apply_bc(level);

        // Chebyshev-optimal weight for step k
        // ω_k = 1/(d - c*cos(θ_k)) where θ_k = π*(2k+1)/(2*degree)
        double theta = M_PI * (2.0 * k + 1.0) / (2.0 * degree);
        double omega = 1.0 / (d - c * std::cos(theta));

        // Steps 2-3: Compute Jacobi update and apply Chebyshev weight
        // Read from u (with valid BCs), write to tmp
        if (is_2d) {
            CHEBY_TARGET_2D
            for (int j = Ng; j < Ny + Ng; ++j) {
                for (int i = Ng; i < Nx + Ng; ++i) {
                    int idx = j * stride + i;
                    double u_old = u_ptr[idx];
                    double u_jacobi = ((u_ptr[idx+1] + u_ptr[idx-1]) / dx2
                                     + (u_ptr[idx+stride] + u_ptr[idx-stride]) / dy2
                                     - f_ptr[idx]) / coeff;
                    tmp_ptr[idx] = (1.0 - omega) * u_old + omega * u_jacobi;
                }
            }
        } else {
            CHEBY_TARGET_3D
            for (int kk = Ng; kk < Nz + Ng; ++kk) {
                for (int j = Ng; j < Ny + Ng; ++j) {
                    for (int i = Ng; i < Nx + Ng; ++i) {
                        int idx = kk * plane_stride + j * stride + i;
                        double u_old = u_ptr[idx];
                        double u_jacobi = ((u_ptr[idx+1] + u_ptr[idx-1]) / dx2
                                         + (u_ptr[idx+stride] + u_ptr[idx-stride]) / dy2
                                         + (u_ptr[idx+plane_stride] + u_ptr[idx-plane_stride]) / dz2
                                         - f_ptr[idx]) / coeff;
                        tmp_ptr[idx] = (1.0 - omega) * u_old + omega * u_jacobi;
                    }
                }
            }
        }

        // Step 4: Copy tmp to u for next iteration
        CHEBY_TARGET_COPY
        for (size_t idx = 0; idx < total_size; ++idx) {
            u_ptr[idx] = tmp_ptr[idx];
        }
    }

    // Final BC application after all iterations
    apply_bc(level);

    #undef CHEBY_TARGET_2D
    #undef CHEBY_TARGET_3D
    #undef CHEBY_TARGET_COPY
}

void MultigridPoissonSolver::smooth_jacobi(int level, int iterations, double omega) {
    // Weighted Jacobi smoother - UNIFIED CPU/GPU implementation
    // Uses ping-pong buffers (u <-> tmp) for identical arithmetic on both paths.
    // GPU uses omp target; CPU uses same algorithm for bitwise consistency.
    NVTX_SCOPE_POISSON("mg:smooth_jacobi");

    auto& grid = *levels_[level];
    const double dx2 = grid.dx * grid.dx;
    const double dy2 = grid.dy * grid.dy;
    const double dz2 = grid.dz * grid.dz;
    const bool is_2d = grid.is2D();
    const double coeff = is_2d ? (2.0 / dx2 + 2.0 / dy2)
                               : (2.0 / dx2 + 2.0 / dy2 + 2.0 / dz2);

    const int Ng = grid.Ng;  // Use level's ghost width
    const int Nx = grid.Nx;
    const int Ny = grid.Ny;
    const int Nz = grid.Nz;
    const int stride = grid.stride;
    const int plane_stride = grid.plane_stride;
    [[maybe_unused]] const size_t total_size = grid.total_size;

#ifdef USE_GPU_OFFLOAD
    // NVHPC WORKAROUND: Use omp_get_mapped_ptr to get actual device addresses.
    // Local pointer copies with map(present:) get HOST addresses in NVHPC.
    int device = omp_get_default_device();
    double* u_ptr = static_cast<double*>(omp_get_mapped_ptr(u_ptrs_[level], device));
    const double* f_ptr = static_cast<const double*>(omp_get_mapped_ptr(f_ptrs_[level], device));
    double* tmp_ptr = tmp_ptrs_[level];  // Already a raw device pointer (omp_target_alloc)
    // All pointers are now device pointers - use is_device_ptr for all
    #define JACOBI_TARGET_U_TO_TMP_2D \
        _Pragma("omp target teams distribute parallel for collapse(2) is_device_ptr(u_ptr, f_ptr, tmp_ptr)")
    #define JACOBI_TARGET_TMP_TO_U_2D \
        _Pragma("omp target teams distribute parallel for collapse(2) is_device_ptr(u_ptr, f_ptr, tmp_ptr)")
    #define JACOBI_TARGET_U_TO_TMP_3D \
        _Pragma("omp target teams distribute parallel for collapse(3) is_device_ptr(u_ptr, f_ptr, tmp_ptr)")
    #define JACOBI_TARGET_TMP_TO_U_3D \
        _Pragma("omp target teams distribute parallel for collapse(3) is_device_ptr(u_ptr, f_ptr, tmp_ptr)")
    #define JACOBI_TARGET_COPY \
        _Pragma("omp target teams distribute parallel for is_device_ptr(u_ptr, tmp_ptr)")
#else
    // CPU path: use host pointers directly
    double* u_ptr = u_ptrs_[level];
    const double* f_ptr = f_ptrs_[level];
    double* tmp_ptr = r_ptrs_[level];  // Reuse r as scratch buffer on CPU
    // CPU: no pragmas needed
    #define JACOBI_TARGET_U_TO_TMP_2D
    #define JACOBI_TARGET_TMP_TO_U_2D
    #define JACOBI_TARGET_U_TO_TMP_3D
    #define JACOBI_TARGET_TMP_TO_U_3D
    #define JACOBI_TARGET_COPY
#endif

    // Ping-pong Jacobi: u -> tmp, then tmp -> u
    // Identical algorithm for CPU and GPU - only pragmas differ
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            // u -> tmp
            if (is_2d) {
                JACOBI_TARGET_U_TO_TMP_2D
                for (int j = Ng; j < Ny + Ng; ++j) {
                    for (int i = Ng; i < Nx + Ng; ++i) {
                        int idx = j * stride + i;
                        double u_old = u_ptr[idx];
                        double u_jacobi = ((u_ptr[idx+1] + u_ptr[idx-1]) / dx2
                                         + (u_ptr[idx+stride] + u_ptr[idx-stride]) / dy2
                                         - f_ptr[idx]) / coeff;
                        tmp_ptr[idx] = (1.0 - omega) * u_old + omega * u_jacobi;
                    }
                }
            } else {
                JACOBI_TARGET_U_TO_TMP_3D
                for (int k = Ng; k < Nz + Ng; ++k) {
                    for (int j = Ng; j < Ny + Ng; ++j) {
                        for (int i = Ng; i < Nx + Ng; ++i) {
                            int idx = k * plane_stride + j * stride + i;
                            double u_old = u_ptr[idx];
                            double u_jacobi = ((u_ptr[idx+1] + u_ptr[idx-1]) / dx2
                                             + (u_ptr[idx+stride] + u_ptr[idx-stride]) / dy2
                                             + (u_ptr[idx+plane_stride] + u_ptr[idx-plane_stride]) / dz2
                                             - f_ptr[idx]) / coeff;
                            tmp_ptr[idx] = (1.0 - omega) * u_old + omega * u_jacobi;
                        }
                    }
                }
            }
        } else {
            // tmp -> u
            if (is_2d) {
                JACOBI_TARGET_TMP_TO_U_2D
                for (int j = Ng; j < Ny + Ng; ++j) {
                    for (int i = Ng; i < Nx + Ng; ++i) {
                        int idx = j * stride + i;
                        double u_old = tmp_ptr[idx];
                        double u_jacobi = ((tmp_ptr[idx+1] + tmp_ptr[idx-1]) / dx2
                                         + (tmp_ptr[idx+stride] + tmp_ptr[idx-stride]) / dy2
                                         - f_ptr[idx]) / coeff;
                        u_ptr[idx] = (1.0 - omega) * u_old + omega * u_jacobi;
                    }
                }
            } else {
                JACOBI_TARGET_TMP_TO_U_3D
                for (int k = Ng; k < Nz + Ng; ++k) {
                    for (int j = Ng; j < Ny + Ng; ++j) {
                        for (int i = Ng; i < Nx + Ng; ++i) {
                            int idx = k * plane_stride + j * stride + i;
                            double u_old = tmp_ptr[idx];
                            double u_jacobi = ((tmp_ptr[idx+1] + tmp_ptr[idx-1]) / dx2
                                             + (tmp_ptr[idx+stride] + tmp_ptr[idx-stride]) / dy2
                                             + (tmp_ptr[idx+plane_stride] + tmp_ptr[idx-plane_stride]) / dz2
                                             - f_ptr[idx]) / coeff;
                            u_ptr[idx] = (1.0 - omega) * u_old + omega * u_jacobi;
                        }
                    }
                }
            }
        }

        // After even iter: result in tmp_ptr; after odd iter: result in u_ptr
        // Copy result to u_ptr for BC application
        if (iter % 2 == 0) {
            JACOBI_TARGET_COPY
            for (size_t idx = 0; idx < total_size; ++idx) {
                u_ptr[idx] = tmp_ptr[idx];
            }
        }
        apply_bc(level);

        // Copy updated BCs back to tmp for next iteration (if not last)
        if (iter % 2 == 0 && iter + 1 < iterations) {
            JACOBI_TARGET_COPY
            for (size_t idx = 0; idx < total_size; ++idx) {
                tmp_ptr[idx] = u_ptr[idx];
            }
        }
    }

    // Clean up macros
    #undef JACOBI_TARGET_U_TO_TMP_2D
    #undef JACOBI_TARGET_TMP_TO_U_2D
    #undef JACOBI_TARGET_U_TO_TMP_3D
    #undef JACOBI_TARGET_TMP_TO_U_3D
    #undef JACOBI_TARGET_COPY
}

void MultigridPoissonSolver::compute_residual(int level) {
    NVTX_SCOPE_RESIDUAL("mg:residual");

    // r = f - L(u) where L is Laplacian operator
    // UNIFIED CODE: Same arithmetic for CPU and GPU, pragma handles offloading
    auto& grid = *levels_[level];
    const double dx2 = grid.dx * grid.dx;
    const double dy2 = grid.dy * grid.dy;
    const double dz2 = grid.dz * grid.dz;
    const bool is_2d = grid.is2D();

    const int Ng = grid.Ng;  // Use level's ghost width
    const int Nx = grid.Nx;
    const int Ny = grid.Ny;
    const int Nz = grid.Nz;
    const int stride = grid.stride;
    const int plane_stride = grid.plane_stride;

#ifdef USE_GPU_OFFLOAD
    // NVHPC WORKAROUND: Use omp_get_mapped_ptr for actual device addresses
    int device = omp_get_default_device();
    const double* u_ptr = static_cast<const double*>(omp_get_mapped_ptr(u_ptrs_[level], device));
    const double* f_ptr = static_cast<const double*>(omp_get_mapped_ptr(f_ptrs_[level], device));
    double* r_ptr = static_cast<double*>(omp_get_mapped_ptr(r_ptrs_[level], device));
#else
    const double* u_ptr = u_ptrs_[level];
    const double* f_ptr = f_ptrs_[level];
    double* r_ptr = r_ptrs_[level];
#endif

    if (is_2d) {
#ifdef USE_GPU_OFFLOAD
        #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(u_ptr, f_ptr, r_ptr)
#endif
        for (int j = Ng; j < Ny + Ng; ++j) {
            for (int i = Ng; i < Nx + Ng; ++i) {
                int idx = j * stride + i;
                double laplacian = (u_ptr[idx+1] - 2.0*u_ptr[idx] + u_ptr[idx-1]) / dx2
                                 + (u_ptr[idx+stride] - 2.0*u_ptr[idx] + u_ptr[idx-stride]) / dy2;
                r_ptr[idx] = f_ptr[idx] - laplacian;
            }
        }
    } else {
        // 3D path
#ifdef USE_GPU_OFFLOAD
        #pragma omp target teams distribute parallel for collapse(3) is_device_ptr(u_ptr, f_ptr, r_ptr)
#endif
        for (int k = Ng; k < Nz + Ng; ++k) {
            for (int j = Ng; j < Ny + Ng; ++j) {
                for (int i = Ng; i < Nx + Ng; ++i) {
                    int idx = k * plane_stride + j * stride + i;
                    double laplacian = (u_ptr[idx+1] - 2.0*u_ptr[idx] + u_ptr[idx-1]) / dx2
                                     + (u_ptr[idx+stride] - 2.0*u_ptr[idx] + u_ptr[idx-stride]) / dy2
                                     + (u_ptr[idx+plane_stride] - 2.0*u_ptr[idx] + u_ptr[idx-plane_stride]) / dz2;
                    r_ptr[idx] = f_ptr[idx] - laplacian;
                }
            }
        }
    }
}

void MultigridPoissonSolver::compute_residual_and_norms(int level, double& r_inf, double& r_l2) {
    NVTX_SCOPE_RESIDUAL("mg:residual+norms");

    // Fused residual computation + norm calculation (single pass over memory)
    // Computes r = f - L(u) AND ||r||_∞ AND ||r||_2 in one kernel
    // Much more efficient than separate compute_residual() + compute_max_residual()
    auto& grid = *levels_[level];
    const double dx2 = grid.dx * grid.dx;
    const double dy2 = grid.dy * grid.dy;
    const double dz2 = grid.dz * grid.dz;
    const bool is_2d = grid.is2D();

    const int Ng = grid.Ng;  // Use level's ghost width
    const int Nx = grid.Nx;
    const int Ny = grid.Ny;
    const int Nz = grid.Nz;
    const int stride = grid.stride;
    const int plane_stride = grid.plane_stride;

    // NVHPC WORKAROUND: Use omp_get_mapped_ptr for actual device addresses.
#ifdef USE_GPU_OFFLOAD
    int device = omp_get_default_device();
    const double* u_ptr = static_cast<const double*>(omp_get_mapped_ptr(u_ptrs_[level], device));
    const double* f_ptr = static_cast<const double*>(omp_get_mapped_ptr(f_ptrs_[level], device));
    double* r_ptr = static_cast<double*>(omp_get_mapped_ptr(r_ptrs_[level], device));
#else
    const double* u_ptr = u_ptrs_[level];
    const double* f_ptr = f_ptrs_[level];
    double* r_ptr = r_ptrs_[level];
#endif

    double max_res = 0.0;
    double sum_sq = 0.0;

    if (is_2d) {
#ifdef USE_GPU_OFFLOAD
        #pragma omp target teams distribute parallel for collapse(2) \
            is_device_ptr(u_ptr, f_ptr, r_ptr) reduction(max: max_res) reduction(+: sum_sq)
#endif
        for (int j = Ng; j < Ny + Ng; ++j) {
            for (int i = Ng; i < Nx + Ng; ++i) {
                int idx = j * stride + i;
                double laplacian = (u_ptr[idx+1] - 2.0*u_ptr[idx] + u_ptr[idx-1]) / dx2
                                 + (u_ptr[idx+stride] - 2.0*u_ptr[idx] + u_ptr[idx-stride]) / dy2;
                double r = f_ptr[idx] - laplacian;
                r_ptr[idx] = r;
                // Compute norms
                double abs_r = (r >= 0.0) ? r : -r;
                if (abs_r > max_res) max_res = abs_r;
                sum_sq += r * r;
            }
        }
    } else {
        // 3D path
#ifdef USE_GPU_OFFLOAD
        #pragma omp target teams distribute parallel for collapse(3) \
            is_device_ptr(u_ptr, f_ptr, r_ptr) reduction(max: max_res) reduction(+: sum_sq)
#endif
        for (int k = Ng; k < Nz + Ng; ++k) {
            for (int j = Ng; j < Ny + Ng; ++j) {
                for (int i = Ng; i < Nx + Ng; ++i) {
                    int idx = k * plane_stride + j * stride + i;
                    double laplacian = (u_ptr[idx+1] - 2.0*u_ptr[idx] + u_ptr[idx-1]) / dx2
                                     + (u_ptr[idx+stride] - 2.0*u_ptr[idx] + u_ptr[idx-stride]) / dy2
                                     + (u_ptr[idx+plane_stride] - 2.0*u_ptr[idx] + u_ptr[idx-plane_stride]) / dz2;
                    double r = f_ptr[idx] - laplacian;
                    r_ptr[idx] = r;
                    // Compute norms
                    double abs_r = (r >= 0.0) ? r : -r;
                    if (abs_r > max_res) max_res = abs_r;
                    sum_sq += r * r;
                }
            }
        }
    }

    r_inf = max_res;
    r_l2 = std::sqrt(sum_sq);
}

void MultigridPoissonSolver::restrict_residual(int fine_level) {
    NVTX_SCOPE_MG("mg:restrict");

    // Full-weighting restriction from fine to coarse grid
    // UNIFIED CODE: Same arithmetic for CPU and GPU, pragma handles offloading
    auto& fine = *levels_[fine_level];
    auto& coarse = *levels_[fine_level + 1];
    const bool is_2d = fine.is2D();

    // Each level has its own ghost width (finest may have Ng>1 for O4, coarse levels use Ng=1)
    const int Ng_f = fine.Ng;    // Fine level ghost width
    const int Ng_c = coarse.Ng;  // Coarse level ghost width
    const int Nx_c = coarse.Nx;
    const int Ny_c = coarse.Ny;
    const int Nz_c = coarse.Nz;
    const int stride_f = fine.stride;
    const int stride_c = coarse.stride;
    const int plane_stride_f = fine.plane_stride;
    const int plane_stride_c = coarse.plane_stride;

#ifdef USE_GPU_OFFLOAD
    // NVHPC WORKAROUND: Use omp_get_mapped_ptr for actual device addresses
    int device = omp_get_default_device();
    const double* r_fine = static_cast<const double*>(omp_get_mapped_ptr(r_ptrs_[fine_level], device));
    double* f_coarse = static_cast<double*>(omp_get_mapped_ptr(f_ptrs_[fine_level + 1], device));
#else
    const double* r_fine = r_ptrs_[fine_level];
    double* f_coarse = f_ptrs_[fine_level + 1];
#endif

    if (is_2d) {
        // 2D: 9-point stencil
#ifdef USE_GPU_OFFLOAD
        #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(r_fine, f_coarse)
#endif
        for (int j_c = Ng_c; j_c < Ny_c + Ng_c; ++j_c) {
            for (int i_c = Ng_c; i_c < Nx_c + Ng_c; ++i_c) {
                // Map coarse index to fine index: i_f = Ng_f + 2*(i_c - Ng_c)
                int i_f = Ng_f + 2 * (i_c - Ng_c);
                int j_f = Ng_f + 2 * (j_c - Ng_c);
                int idx_c = j_c * stride_c + i_c;
                int idx_f = j_f * stride_f + i_f;

                f_coarse[idx_c] = 0.25 * r_fine[idx_f]
                                + 0.125 * (r_fine[idx_f-1] + r_fine[idx_f+1]
                                         + r_fine[idx_f-stride_f] + r_fine[idx_f+stride_f])
                                + 0.0625 * (r_fine[idx_f-1-stride_f] + r_fine[idx_f+1-stride_f]
                                          + r_fine[idx_f-1+stride_f] + r_fine[idx_f+1+stride_f]);
            }
        }
    } else {
        // 3D: 27-point stencil
#ifdef USE_GPU_OFFLOAD
        #pragma omp target teams distribute parallel for collapse(3) is_device_ptr(r_fine, f_coarse)
#endif
        for (int k_c = Ng_c; k_c < Nz_c + Ng_c; ++k_c) {
            for (int j_c = Ng_c; j_c < Ny_c + Ng_c; ++j_c) {
                for (int i_c = Ng_c; i_c < Nx_c + Ng_c; ++i_c) {
                    // Map coarse index to fine index: i_f = Ng_f + 2*(i_c - Ng_c)
                    int i_f = Ng_f + 2 * (i_c - Ng_c);
                    int j_f = Ng_f + 2 * (j_c - Ng_c);
                    int k_f = Ng_f + 2 * (k_c - Ng_c);
                    int idx_c = k_c * plane_stride_c + j_c * stride_c + i_c;
                    int idx_f = k_f * plane_stride_f + j_f * stride_f + i_f;

                    // 27-point full-weighting stencil
                    double sum = 0.0;

                    // Center point (weight = 1/8)
                    sum += 0.125 * r_fine[idx_f];

                    // 6 face neighbors (weight = 1/16 each)
                    sum += 0.0625 * (r_fine[idx_f-1] + r_fine[idx_f+1]
                                   + r_fine[idx_f-stride_f] + r_fine[idx_f+stride_f]
                                   + r_fine[idx_f-plane_stride_f] + r_fine[idx_f+plane_stride_f]);

                    // 12 edge neighbors (weight = 1/32 each)
                    sum += 0.03125 * (r_fine[idx_f-1-stride_f] + r_fine[idx_f+1-stride_f]
                                    + r_fine[idx_f-1+stride_f] + r_fine[idx_f+1+stride_f]
                                    + r_fine[idx_f-1-plane_stride_f] + r_fine[idx_f+1-plane_stride_f]
                                    + r_fine[idx_f-1+plane_stride_f] + r_fine[idx_f+1+plane_stride_f]
                                    + r_fine[idx_f-stride_f-plane_stride_f] + r_fine[idx_f+stride_f-plane_stride_f]
                                    + r_fine[idx_f-stride_f+plane_stride_f] + r_fine[idx_f+stride_f+plane_stride_f]);

                    // 8 corner neighbors (weight = 1/64 each)
                    sum += 0.015625 * (r_fine[idx_f-1-stride_f-plane_stride_f] + r_fine[idx_f+1-stride_f-plane_stride_f]
                                     + r_fine[idx_f-1+stride_f-plane_stride_f] + r_fine[idx_f+1+stride_f-plane_stride_f]
                                     + r_fine[idx_f-1-stride_f+plane_stride_f] + r_fine[idx_f+1-stride_f+plane_stride_f]
                                     + r_fine[idx_f-1+stride_f+plane_stride_f] + r_fine[idx_f+1+stride_f+plane_stride_f]);

                    f_coarse[idx_c] = sum;
                }
            }
        }
    }
}

void MultigridPoissonSolver::prolongate_correction(int coarse_level) {
    NVTX_SCOPE_MG("mg:prolongate");

    // Owner-computes trilinear interpolation: each fine cell computes its own value
    // by reading from neighboring coarse cells. No atomics needed!
    // UNIFIED CODE: Same arithmetic for CPU and GPU, pragma handles offloading
    auto& coarse = *levels_[coarse_level];
    auto& fine = *levels_[coarse_level - 1];
    const bool is_2d = fine.is2D();

    // Each level has its own ghost width (finest may have Ng>1 for O4, coarse levels use Ng=1)
    const int Ng_f = fine.Ng;    // Fine level ghost width
    const int Ng_c = coarse.Ng;  // Coarse level ghost width
    const int Nx_f = fine.Nx;
    const int Ny_f = fine.Ny;
    const int Nz_f = fine.Nz;
    const int stride_f = fine.stride;
    const int stride_c = coarse.stride;
    const int plane_stride_f = fine.plane_stride;
    const int plane_stride_c = coarse.plane_stride;

    // NVHPC WORKAROUND: Use omp_get_mapped_ptr for actual device addresses
#ifdef USE_GPU_OFFLOAD
    int device = omp_get_default_device();
    const double* u_coarse = static_cast<const double*>(omp_get_mapped_ptr(u_ptrs_[coarse_level], device));
    double* u_fine = static_cast<double*>(omp_get_mapped_ptr(u_ptrs_[coarse_level - 1], device));
#else
    const double* u_coarse = u_ptrs_[coarse_level];
    double* u_fine = u_ptrs_[coarse_level - 1];
#endif

    if (is_2d) {
        // 2D owner-computes bilinear interpolation
        // Each fine cell reads from up to 4 coarse neighbors
#ifdef USE_GPU_OFFLOAD
        #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(u_coarse, u_fine)
#endif
        for (int j_f = Ng_f; j_f < Ny_f + Ng_f; ++j_f) {
            for (int i_f = Ng_f; i_f < Nx_f + Ng_f; ++i_f) {
                // Find base coarse cell and position within coarse cell pair
                // Map fine to coarse: i_c = (i_f - Ng_f) / 2 + Ng_c
                int i_c = (i_f - Ng_f) / 2 + Ng_c;
                int j_c = (j_f - Ng_f) / 2 + Ng_c;
                int di = (i_f - Ng_f) & 1;  // 0 = coincident, 1 = midpoint
                int dj = (j_f - Ng_f) & 1;

                // Interpolation weights: 0.5*d gives 0.0 or 0.5
                double wx1 = 0.5 * di;
                double wx0 = 1.0 - wx1;
                double wy1 = 0.5 * dj;
                double wy0 = 1.0 - wy1;

                int idx_c = j_c * stride_c + i_c;

                // Bilinear interpolation from 4 coarse neighbors
                double correction = wx0 * wy0 * u_coarse[idx_c]
                                  + wx1 * wy0 * u_coarse[idx_c + 1]
                                  + wx0 * wy1 * u_coarse[idx_c + stride_c]
                                  + wx1 * wy1 * u_coarse[idx_c + 1 + stride_c];

                int idx_f = j_f * stride_f + i_f;
                u_fine[idx_f] += correction;
            }
        }
    } else {
        // 3D owner-computes trilinear interpolation
        // Each fine cell reads from up to 8 coarse neighbors
#ifdef USE_GPU_OFFLOAD
        #pragma omp target teams distribute parallel for collapse(3) is_device_ptr(u_coarse, u_fine)
#endif
        for (int k_f = Ng_f; k_f < Nz_f + Ng_f; ++k_f) {
            for (int j_f = Ng_f; j_f < Ny_f + Ng_f; ++j_f) {
                for (int i_f = Ng_f; i_f < Nx_f + Ng_f; ++i_f) {
                    // Find base coarse cell and position within coarse cell pair
                    // Map fine to coarse: i_c = (i_f - Ng_f) / 2 + Ng_c
                    int i_c = (i_f - Ng_f) / 2 + Ng_c;
                    int j_c = (j_f - Ng_f) / 2 + Ng_c;
                    int k_c = (k_f - Ng_f) / 2 + Ng_c;
                    int di = (i_f - Ng_f) & 1;  // 0 = coincident, 1 = midpoint
                    int dj = (j_f - Ng_f) & 1;
                    int dk = (k_f - Ng_f) & 1;

                    // Interpolation weights
                    double wx1 = 0.5 * di;
                    double wx0 = 1.0 - wx1;
                    double wy1 = 0.5 * dj;
                    double wy0 = 1.0 - wy1;
                    double wz1 = 0.5 * dk;
                    double wz0 = 1.0 - wz1;

                    int idx_c = k_c * plane_stride_c + j_c * stride_c + i_c;

                    // Trilinear interpolation from 8 coarse neighbors
                    double correction =
                        wx0 * wy0 * wz0 * u_coarse[idx_c]
                      + wx1 * wy0 * wz0 * u_coarse[idx_c + 1]
                      + wx0 * wy1 * wz0 * u_coarse[idx_c + stride_c]
                      + wx1 * wy1 * wz0 * u_coarse[idx_c + 1 + stride_c]
                      + wx0 * wy0 * wz1 * u_coarse[idx_c + plane_stride_c]
                      + wx1 * wy0 * wz1 * u_coarse[idx_c + 1 + plane_stride_c]
                      + wx0 * wy1 * wz1 * u_coarse[idx_c + stride_c + plane_stride_c]
                      + wx1 * wy1 * wz1 * u_coarse[idx_c + 1 + stride_c + plane_stride_c];

                    int idx_f = k_f * plane_stride_f + j_f * stride_f + i_f;
                    u_fine[idx_f] += correction;
                }
            }
        }
    }
}

void MultigridPoissonSolver::solve_coarsest(int iterations) {
    // Direct solve on coarsest grid using multiple Chebyshev iterations
    // Each Chebyshev call performs degree=4 polynomial sweeps
    int coarsest = levels_.size() - 1;
    for (int i = 0; i < iterations; ++i) {
        smooth_chebyshev(coarsest, 4);
    }
}

void MultigridPoissonSolver::vcycle(int level, int nu1, int nu2, int degree) {
    NVTX_SCOPE_POISSON("mg:vcycle");

    // Check environment variable override for smoother type (on first call)
    static bool checked_env = false;
    if (!checked_env) {
        checked_env = true;
        const char* env = std::getenv("MG_SMOOTHER");
        if (env != nullptr) {
            std::string s(env);
            if (s == "jacobi" || s == "JACOBI") {
                smoother_type_ = MGSmootherType::Jacobi;
            } else if (s == "chebyshev" || s == "CHEBYSHEV") {
                smoother_type_ = MGSmootherType::Chebyshev;
            }
        }
    }

    // Helper lambda to call the appropriate smoother
    // nu = number of smoothing passes, degree = Chebyshev polynomial degree per pass
    auto do_smooth = [this, level, degree](int nu) {
        for (int pass = 0; pass < nu; ++pass) {
            if (smoother_type_ == MGSmootherType::Chebyshev) {
                smooth_chebyshev(level, degree);
            } else {
                smooth_jacobi(level, degree, 0.8);  // omega = 0.8 for stability
            }
        }
    };

    if (level == static_cast<int>(levels_.size()) - 1) {
        // Coarsest level - solve approximately
        // With MIN_COARSE_SIZE=8, coarsest is 8x8 (64 points)
        // Use more iterations/higher degree for accurate coarse solve
        if (smoother_type_ == MGSmootherType::Chebyshev) {
            smooth_chebyshev(level, std::max(8, degree * 2));  // Higher degree for coarse
        } else {
            smooth_jacobi(level, 20, 0.8);  // 20 Jacobi iterations
        }
        return;
    }

    // Pre-smoothing (nu1 passes of degree-k Chebyshev)
    do_smooth(nu1);

    // Compute residual
    compute_residual(level);

    // Apply BCs to residual ghost cells for proper 9-point restriction
    // This is essential for periodic BCs where ghost cells must wrap around
    apply_bc_to_residual(level);

    // Restrict to coarse grid
    restrict_residual(level);

    // Zero coarse grid solution
    {
        NVTX_SCOPE_MG("mg:zero_coarse");
        auto& coarse = *levels_[level + 1];

#ifdef USE_GPU_OFFLOAD
        assert(gpu_ready_ && "GPU must be initialized");
        const size_t size_c = level_sizes_[level + 1];
        // NVHPC WORKAROUND: Use omp_get_mapped_ptr for actual device addresses.
        int device = omp_get_default_device();
        double* u_coarse = static_cast<double*>(omp_get_mapped_ptr(u_ptrs_[level + 1], device));

        #pragma omp target teams distribute parallel for is_device_ptr(u_coarse)
        for (int idx = 0; idx < (int)size_c; ++idx) {
            u_coarse[idx] = 0.0;
        }
#else
        // Zero the coarse grid (CPU path)
        for (int k = 0; k < coarse.Sz; ++k) {
            for (int j = 0; j < coarse.Sy; ++j) {
                for (int i = 0; i < coarse.Sx; ++i) {
                    coarse.u(i, j, k) = 0.0;
                }
            }
        }
#endif
    }

    // Recursive call to coarser level
    vcycle(level + 1, nu1, nu2, degree);

    // Prolongate correction and apply boundary conditions
    // (BC needed before post-smoothing reads ghost cells)
    prolongate_correction(level + 1);
    apply_bc(level);

    // Post-smoothing (nu2 passes of degree-k Chebyshev)
    if (nu2 > 0) {
        do_smooth(nu2);
    }
}

double MultigridPoissonSolver::compute_max_residual(int level) {
    // UNIFIED CODE: Same arithmetic for CPU and GPU, pragma handles offloading
    compute_residual(level);

    auto& grid = *levels_[level];
    double max_res = 0.0;
    const int Ng = grid.Ng;  // Use level's ghost width
    const int Nx = grid.Nx;
    const int Ny = grid.Ny;
    const int Nz = grid.Nz;
    const int stride = grid.stride;
    const int plane_stride = grid.plane_stride;

    // NVHPC WORKAROUND: Use omp_get_mapped_ptr for actual device addresses.
#ifdef USE_GPU_OFFLOAD
    int device = omp_get_default_device();
    const double* r_ptr = static_cast<const double*>(omp_get_mapped_ptr(r_ptrs_[level], device));
#else
    const double* r_ptr = r_ptrs_[level];
#endif

    if (Nz == 1) {
        // 2D case
#ifdef USE_GPU_OFFLOAD
        #pragma omp target teams distribute parallel for reduction(max:max_res) is_device_ptr(r_ptr)
#endif
        for (int idx = 0; idx < Nx * Ny; ++idx) {
            int i = idx % Nx + Ng;
            int j = idx / Nx + Ng;
            int ridx = j * stride + i;
            double v = r_ptr[ridx];
            // Explicitly detect NaN and propagate infinity to signal divergence
            if (!(v == v)) {  // NaN check: NaN != NaN
                max_res = 1e308;  // Large value that will "win" max reduction
            } else {
                double abs_v = (v >= 0.0) ? v : -v;
                if (abs_v > max_res) max_res = abs_v;
            }
        }
    } else {
        // 3D case
#ifdef USE_GPU_OFFLOAD
        #pragma omp target teams distribute parallel for reduction(max:max_res) is_device_ptr(r_ptr)
#endif
        for (int idx = 0; idx < Nx * Ny * Nz; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % Ny + Ng;
            int k = idx / (Nx * Ny) + Ng;
            int ridx = k * plane_stride + j * stride + i;
            double v = r_ptr[ridx];
            // Explicitly detect NaN and propagate infinity to signal divergence
            if (!(v == v)) {  // NaN check: NaN != NaN
                max_res = 1e308;  // Large value that will "win" max reduction
            } else {
                double abs_v = (v >= 0.0) ? v : -v;
                if (abs_v > max_res) max_res = abs_v;
            }
        }
    }

    // If max_res is invalid (NaN from the solver), signal divergence to caller
    // This prevents infinite loops in the Poisson solver's convergence check
    if (!std::isfinite(max_res)) {
        return std::numeric_limits<double>::infinity();
    }

    return max_res;
}

bool MultigridPoissonSolver::has_nullspace() const {
    // The Poisson operator with Neumann/Periodic BCs has a nullspace (constants).
    // Dirichlet BC on ANY face pins the solution, eliminating the nullspace.
    //
    // This determines whether subtract_mean() is needed after solving.
    bool has_dirichlet = (bc_x_lo_ == PoissonBC::Dirichlet || bc_x_hi_ == PoissonBC::Dirichlet ||
                          bc_y_lo_ == PoissonBC::Dirichlet || bc_y_hi_ == PoissonBC::Dirichlet ||
                          bc_z_lo_ == PoissonBC::Dirichlet || bc_z_hi_ == PoissonBC::Dirichlet);
    return !has_dirichlet;
}

void MultigridPoissonSolver::fix_nullspace(int level) {
    if (has_nullspace()) {
        subtract_mean(level);
        apply_bc(level);  // Re-apply BCs after mean subtraction (ghost cells now inconsistent)
    }
}

void MultigridPoissonSolver::subtract_mean(int level) {
    // UNIFIED CODE: Same arithmetic for CPU and GPU, pragma handles offloading
    auto& grid = *levels_[level];
    double sum = 0.0;
    const int Ng = grid.Ng;  // Use level's ghost width
    const int Nx = grid.Nx;
    const int Ny = grid.Ny;
    const int Nz = grid.Nz;
    const int stride = grid.stride;
    const int plane_stride = grid.plane_stride;

    // NVHPC WORKAROUND: Use omp_get_mapped_ptr for actual device addresses.
#ifdef USE_GPU_OFFLOAD
    int device = omp_get_default_device();
    double* u_ptr = static_cast<double*>(omp_get_mapped_ptr(u_ptrs_[level], device));
#else
    double* u_ptr = u_ptrs_[level];
#endif

    if (Nz == 1) {
        // 2D case - compute sum
#ifdef USE_GPU_OFFLOAD
        #pragma omp target teams distribute parallel for reduction(+:sum) is_device_ptr(u_ptr)
#endif
        for (int idx = 0; idx < Nx * Ny; ++idx) {
            int i = idx % Nx + Ng;
            int j = idx / Nx + Ng;
            sum += u_ptr[j * stride + i];
        }

        double mean = sum / (Nx * Ny);

        // 2D case - subtract mean
#ifdef USE_GPU_OFFLOAD
        #pragma omp target teams distribute parallel for is_device_ptr(u_ptr)
#endif
        for (int idx = 0; idx < Nx * Ny; ++idx) {
            int i = idx % Nx + Ng;
            int j = idx / Nx + Ng;
            u_ptr[j * stride + i] -= mean;
        }
    } else {
        // 3D case - compute sum
#ifdef USE_GPU_OFFLOAD
        #pragma omp target teams distribute parallel for reduction(+:sum) is_device_ptr(u_ptr)
#endif
        for (int idx = 0; idx < Nx * Ny * Nz; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % Ny + Ng;
            int k = idx / (Nx * Ny) + Ng;
            sum += u_ptr[k * plane_stride + j * stride + i];
        }

        double mean = sum / (Nx * Ny * Nz);

        // 3D case - subtract mean
#ifdef USE_GPU_OFFLOAD
        #pragma omp target teams distribute parallel for is_device_ptr(u_ptr)
#endif
        for (int idx = 0; idx < Nx * Ny * Nz; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % Ny + Ng;
            int k = idx / (Nx * Ny) + Ng;
            u_ptr[k * plane_stride + j * stride + i] -= mean;
        }
    }
}

int MultigridPoissonSolver::solve(const ScalarField& rhs, ScalarField& p, const PoissonConfig& cfg) {
    NVTX_SCOPE_POISSON("poisson:solve");

    // Copy RHS and initial guess to finest level (CPU side)
    auto& finest = *levels_[0];
    const int Ng = finest.Ng;  // Use level's ghost width

    // 3D-aware data copy
    if (mesh_->is2D()) {
        for (int j = Ng; j < finest.Ny + Ng; ++j) {
            for (int i = Ng; i < finest.Nx + Ng; ++i) {
                finest.f(i, j) = rhs(i, j);
                finest.u(i, j) = p(i, j);
            }
        }
    } else {
        for (int k = Ng; k < finest.Nz + Ng; ++k) {
            for (int j = Ng; j < finest.Ny + Ng; ++j) {
                for (int i = Ng; i < finest.Nx + Ng; ++i) {
                    finest.f(i, j, k) = rhs(i, j, k);
                    finest.u(i, j, k) = p(i, j, k);
                }
            }
        }
    }
    
#ifdef USE_GPU_OFFLOAD
    // Upload to GPU once before all V-cycles
    assert(gpu_ready_ && "GPU must be initialized");
    sync_level_to_gpu(0);
#endif

    apply_bc(0);

    // ========================================================================
    // Fixed-cycle mode: run exactly N V-cycles without convergence checks
    // This is the fastest mode for projection - no D→H transfers mid-solve
    // ========================================================================
    if (cfg.fixed_cycles > 0) {
        const int num_cycles = cfg.fixed_cycles;
        // Optimal at 128³ channel: nu1=3, nu2=1 (more pre-smooth for wall BCs)
        // Benchmark: 10× better div_L2 AND 13% faster than nu1=2,nu2=2,cyc=10
        const int nu1 = (cfg.nu1 > 0) ? cfg.nu1 : 3;
        const int nu2 = (cfg.nu2 > 0) ? cfg.nu2 : 1;
        const int degree = cfg.chebyshev_degree;

        for (int cycle = 0; cycle < num_cycles; ++cycle) {
            vcycle(0, nu1, nu2, degree);
        }

        // Set residual to 0 to indicate we didn't compute it
        residual_ = 0.0;
        residual_l2_ = 0.0;
        r0_ = 0.0;
        r0_l2_ = 0.0;
        b_inf_ = 0.0;
        b_l2_ = 0.0;

        // Handle nullspace for singular problems (pure Neumann/Periodic)
        fix_nullspace(0);

#ifdef USE_GPU_OFFLOAD
        sync_level_from_gpu(0);
#endif

        // Copy result back to output field
        if (mesh_->is2D()) {
            for (int j = 0; j < finest.Ny + 2*Ng; ++j) {
                for (int i = 0; i < finest.Nx + 2*Ng; ++i) {
                    p(i, j) = finest.u(i, j);
                }
            }
        } else {
            for (int k = 0; k < finest.Nz + 2*Ng; ++k) {
                for (int j = 0; j < finest.Ny + 2*Ng; ++j) {
                    for (int i = 0; i < finest.Nx + 2*Ng; ++i) {
                        p(i, j, k) = finest.u(i, j, k);
                    }
                }
            }
        }

        return num_cycles;
    }

    // ========================================================================
    // Convergence-based mode: MG V-cycles with tolerance checking
    // ========================================================================
    // Convergence criteria (any one triggers exit):
    //   1. ||r||_∞ ≤ tol_abs  (absolute, usually disabled)
    //   2. ||r||/||b|| ≤ tol_rhs  (RHS-relative, recommended for projection)
    //   3. ||r||/||r0|| ≤ tol_rel  (initial-residual relative, backup)
    // Check every check_interval cycles to reduce overhead.
    assert(cfg.max_vcycles > 0 && "PoissonConfig.max_vcycles must be positive");
    const int max_cycles = cfg.max_vcycles;
    const bool accurate_mode = (max_cycles > 5);
    // Optimal at 128³ channel: nu1=3, nu2=1 (more pre-smooth for wall BCs)
    const int nu1 = (cfg.nu1 > 0) ? cfg.nu1 : (accurate_mode ? 3 : 2);
    const int nu2 = (cfg.nu2 > 0) ? cfg.nu2 : 1;
    const int degree = cfg.chebyshev_degree;
    const int check_interval = std::max(1, cfg.check_interval);

    // Compute reference norms for relative tolerances (CPU path)
    // ||b||_∞ = max|f| and ||b||_2 = sqrt(sum(f^2)) on finest level - store in member for diagnostics
    auto& finest_cpu = *levels_[0];
    b_inf_ = 0.0;
    double b_sum_sq = 0.0;
    if (mesh_->is2D()) {
        for (int j = 1; j <= finest_cpu.Ny; ++j) {
            for (int i = 1; i <= finest_cpu.Nx; ++i) {
                double val = finest_cpu.f(i, j);
                b_inf_ = std::max(b_inf_, std::abs(val));
                b_sum_sq += val * val;
            }
        }
    } else {
        for (int k = 1; k <= finest_cpu.Nz; ++k) {
            for (int j = 1; j <= finest_cpu.Ny; ++j) {
                for (int i = 1; i <= finest_cpu.Nx; ++i) {
                    double val = finest_cpu.f(i, j, k);
                    b_inf_ = std::max(b_inf_, std::abs(val));
                    b_sum_sq += val * val;
                }
            }
        }
    }
    b_l2_ = std::sqrt(b_sum_sq);

    // Initial residual - use fused function to compute residual + both norms in single pass
    compute_residual_and_norms(0, r0_, r0_l2_);
    residual_ = r0_;
    residual_l2_ = r0_l2_;

    int cycles_used = 0;
    for (int cycle = 0; cycle < max_cycles; ++cycle) {
        vcycle(0, nu1, nu2, degree);
        cycles_used = cycle + 1;

        // Check convergence every check_interval cycles (reduces overhead)
        if ((cycle % check_interval) == (check_interval - 1) || cycle == max_cycles - 1) {
            // Fused residual + norm computation (single pass over memory, single GPU reduction)
            compute_residual_and_norms(0, residual_, residual_l2_);

            // Select norm for convergence based on config (L2 is smoother, less sensitive to hot cells)
            const double r_norm = cfg.use_l2_norm ? residual_l2_ : residual_;
            const double b_norm = cfg.use_l2_norm ? b_l2_ : b_inf_;
            const double r0_norm = cfg.use_l2_norm ? r0_l2_ : r0_;

            // Robust convergence check: any criterion triggers exit
            bool converged = false;
            if (cfg.tol_abs > 0.0 && r_norm <= cfg.tol_abs) {
                converged = true;  // Absolute tolerance met
            }
            if (cfg.tol_rhs > 0.0 && r_norm <= cfg.tol_rhs * (b_norm + 1e-30)) {
                converged = true;  // RHS-relative tolerance met
            }
            if (cfg.tol_rel > 0.0 && r_norm <= cfg.tol_rel * (r0_norm + 1e-30)) {
                converged = true;  // Initial-residual relative tolerance met
            }
            // Legacy: also check cfg.tol for backward compatibility
            if (cfg.tol > 0.0 && r_norm <= cfg.tol) {
                converged = true;
            }

            // L∞ safety cap: when using L2, also enforce loose L∞ bound to catch "bad cells"
            // This prevents L2 from hiding localized divergence spikes
            if (converged && cfg.use_l2_norm && cfg.linf_safety_factor > 0.0 && cfg.tol_rhs > 0.0) {
                double linf_ratio = residual_ / (b_inf_ + 1e-30);
                double linf_cap = cfg.tol_rhs * cfg.linf_safety_factor;
                if (linf_ratio > linf_cap) {
                    converged = false;  // L∞ still too high, keep iterating
                }
            }

            if (converged) {
                break;
            }
        }
    }

    // Subtract mean for singular Poisson problems (no Dirichlet BCs)
    // Handle nullspace for singular problems (pure Neumann/Periodic)
    fix_nullspace(0);

#ifdef USE_GPU_OFFLOAD
    // Download from GPU once after all V-cycles
    assert(gpu_ready_ && "GPU must be initialized");
    sync_level_from_gpu(0);
#endif

    // Copy result back to output field (CPU side)
    // CRITICAL: Copy ALL cells including ghost cells to match GPU behavior
    // The ghost cells contain correct BC values that are needed for pressure gradient
    if (mesh_->is2D()) {
        for (int j = 0; j < finest.Ny + 2*Ng; ++j) {
            for (int i = 0; i < finest.Nx + 2*Ng; ++i) {
                p(i, j) = finest.u(i, j);
            }
        }
    } else {
        for (int k = 0; k < finest.Nz + 2*Ng; ++k) {
            for (int j = 0; j < finest.Ny + 2*Ng; ++j) {
                for (int i = 0; i < finest.Nx + 2*Ng; ++i) {
                    p(i, j, k) = finest.u(i, j, k);
                }
            }
        }
    }

    return cycles_used;  // Actual number of V-cycles executed
}

#ifdef USE_GPU_OFFLOAD
/// GPU Multigrid Solver - Device-Resident Implementation
///
/// CONTRACT: All device kernels in this section MUST use one of these patterns:
///   1. gpu::dev_ptr(host_ptr) + is_device_ptr(dev_ptr)
///   2. omp_get_mapped_ptr() + is_device_ptr()
///   3. Member pointers with map(present:) for simple kernels
///
/// FORBIDDEN: Local pointer aliases without explicit device address resolution.
/// See gpu_utils.hpp for full NVHPC workaround documentation.
///
int MultigridPoissonSolver::solve_device(double* rhs_present, double* p_present, const PoissonConfig& cfg) {
    NVTX_SCOPE_POISSON("poisson:solve_device");

    assert(gpu_ready_ && "GPU must be initialized in constructor");

    // Device-resident solve using Model 1 (host pointer + present mapping)
    // Parameters are host pointers that caller has already mapped via `target enter data`.
    // NVHPC WORKAROUND: Use member pointers directly instead of local copies from vectors.
    // Local pointer copies get HOST addresses in NVHPC target regions.

    auto& finest = *levels_[0];
    const int Nx = finest.Nx;
    const int Ny = finest.Ny;
    const int Nz = finest.Nz;
    // Total size includes ghost cells: Sx * Sy * Sz (accounts for level's Ng)
    const size_t total_size = finest.total_size;

    // Copy RHS and initial guess from caller's present-mapped arrays to multigrid level-0 buffers
    // NVHPC WORKAROUND: Use omp_get_mapped_ptr to get actual device addresses, then use is_device_ptr.
    {
        int device = omp_get_default_device();
        double* rhs_dev = static_cast<double*>(omp_get_mapped_ptr(rhs_present, device));
        double* p_dev = static_cast<double*>(omp_get_mapped_ptr(p_present, device));
        double* f_dev = static_cast<double*>(omp_get_mapped_ptr(f_level0_ptr_, device));
        double* u_dev = static_cast<double*>(omp_get_mapped_ptr(u_level0_ptr_, device));

        #pragma omp target teams distribute parallel for is_device_ptr(rhs_dev, p_dev, f_dev, u_dev)
        for (size_t idx = 0; idx < total_size; ++idx) {
            f_dev[idx] = rhs_dev[idx];
            u_dev[idx] = p_dev[idx];
        }
    }

    apply_bc(0);

    // ========================================================================
    // Fixed-cycle mode: run N V-cycles with optional adaptive checking
    // This is the fastest mode for projection - minimal D→H transfers
    // ========================================================================
    if (cfg.fixed_cycles > 0) {
        const int max_cycles = cfg.fixed_cycles;

        // Optimal at 128³ channel: nu1=3, nu2=1 (more pre-smooth for wall BCs)
        // Benchmark: 10× better div_L2 AND 13% faster than nu1=2,nu2=2,cyc=10
        const int nu1 = (cfg.nu1 > 0) ? cfg.nu1 : 3;
        const int nu2 = (cfg.nu2 > 0) ? cfg.nu2 : 1;
        const int degree = cfg.chebyshev_degree;

        // Use full V-cycle graph if available (massive reduction in kernel launches)
        // Respect both config setting and environment variable override
        // NOTE: V-cycle graph is 3D only (2D path not fully tested)
        const bool is_3d = (levels_[0]->Nz > 1);
        const bool use_graph = use_vcycle_graph_ && cfg.use_vcycle_graph && is_3d;
        auto run_cycles = [&](int n) {
            if (use_graph) {
                // Initialize graph on first use or if parameters changed
                if (!vcycle_graph_ || vcycle_graph_nu1_ != nu1 ||
                    vcycle_graph_nu2_ != nu2 || vcycle_graph_degree_ != degree) {
                    initialize_vcycle_graph(nu1, nu2, degree);
                }
                if (vcycle_graph_ && vcycle_graph_->is_valid()) {
                    for (int cycle = 0; cycle < n; ++cycle) {
                        vcycle_graphed();
                    }
                    return;
                }
            }
            // Fall back to non-graphed
            for (int cycle = 0; cycle < n; ++cycle) {
                vcycle(0, nu1, nu2, degree);
            }
        };

        int cycles_run = 0;

        if (cfg.adaptive_cycles) {
            // Adaptive mode: run check_after cycles, then check, add more if needed
            // Pattern: 4 cycles → check → +2 if bad → check → ... → cap at max
            const int check_after = cfg.check_after;
            const double target_tol = cfg.tol_rhs;

            // First batch of cycles
            int batch = std::min(check_after, max_cycles);
            run_cycles(batch);
            cycles_run = batch;

            // CRITICAL: Sync all device work before OpenMP target reduction
            // Both CUDA graph and OpenMP target regions may use different streams.
            // DeviceSynchronize ensures all async GPU work completes before reduction.
            CUDA_CHECK_SYNC(cudaDeviceSynchronize());

            // Compute initial ||b||_2 for relative residual check
            // Use GPU reduction - only transfers 8 bytes (the sum) instead of 17.5 MB array
            {
                auto& finest = *levels_[0];
                const int Ng = finest.Ng;
                const int Nx = finest.Nx;
                const int Ny = finest.Ny;
                const int Nz = finest.Nz;
                const int stride = finest.stride;
                const int plane_stride = finest.plane_stride;
                [[maybe_unused]] const size_t f_size = finest.total_size;
                const double* f_ptr = f_ptrs_[0];
                const bool is_2d = finest.is2D();

                double b_sum_sq = 0.0;

                if (is_2d) {
                    #pragma omp target teams distribute parallel for collapse(2) \
                        map(present: f_ptr[0:f_size]) \
                        reduction(+: b_sum_sq)
                    for (int j = Ng; j < Ny + Ng; ++j) {
                        for (int i = Ng; i < Nx + Ng; ++i) {
                            int idx = j * stride + i;
                            double val = f_ptr[idx];
                            b_sum_sq += val * val;
                        }
                    }
                } else {
                    #pragma omp target teams distribute parallel for collapse(3) \
                        map(present: f_ptr[0:f_size]) \
                        reduction(+: b_sum_sq)
                    for (int k = Ng; k < Nz + Ng; ++k) {
                        for (int j = Ng; j < Ny + Ng; ++j) {
                            for (int i = Ng; i < Nx + Ng; ++i) {
                                int idx = k * plane_stride + j * stride + i;
                                double val = f_ptr[idx];
                                b_sum_sq += val * val;
                            }
                        }
                    }
                }
                b_l2_ = std::sqrt(b_sum_sq);
            }

            // Helper: sync all device work if graph mode was used
            auto sync_if_graphed = [&]() {
                if (use_graph && vcycle_graph_ && vcycle_graph_->is_valid()) {
                    CUDA_CHECK_SYNC(cudaDeviceSynchronize());
                }
            };

            // Check residual and add more cycles if needed
            while (cycles_run < max_cycles) {
                // Compute residual norm (this is the only D→H transfer per check)
                compute_residual_and_norms(0, residual_, residual_l2_);
                double rel_res = (b_l2_ > 0) ? residual_l2_ / b_l2_ : residual_l2_;

                if (rel_res <= target_tol) {
                    break;  // Converged!
                }

                // Run 2 more cycles
                int add = std::min(2, max_cycles - cycles_run);
                run_cycles(add);
                cycles_run += add;

                // Sync before next residual check if graph was used
                sync_if_graphed();
            }
        } else {
            // Pure fixed mode: run exactly max_cycles without checking
            run_cycles(max_cycles);
            cycles_run = max_cycles;
        }

        // Set residual to 0 if we didn't compute it (pure fixed mode)
        if (!cfg.adaptive_cycles) {
            residual_ = 0.0;
            residual_l2_ = 0.0;
            r0_ = 0.0;
            r0_l2_ = 0.0;
            b_inf_ = 0.0;
            b_l2_ = 0.0;
        }

        // Handle nullspace for singular problems (pure Neumann/Periodic)
        fix_nullspace(0);

        // Copy result from multigrid buffer back to caller's present-mapped array (D-to-D)
        // NVHPC WORKAROUND: Use omp_get_mapped_ptr for actual device addresses
        {
            int device = omp_get_default_device();
            double* p_dev = static_cast<double*>(omp_get_mapped_ptr(p_present, device));
            double* u_dev = static_cast<double*>(omp_get_mapped_ptr(u_level0_ptr_, device));
            #pragma omp target teams distribute parallel for is_device_ptr(p_dev, u_dev)
            for (size_t idx = 0; idx < total_size; ++idx) {
                p_dev[idx] = u_dev[idx];
            }
        }

        return cycles_run;
    }

    // ========================================================================
    // Convergence-based mode: MG V-cycles with tolerance checking
    // ========================================================================
    // MG V-cycles with robust tolerance-based early termination (GPU path)
    // Convergence criteria (any one triggers exit):
    //   1. ||r||_∞ ≤ tol_abs  (absolute, usually disabled)
    //   2. ||r||/||b|| ≤ tol_rhs  (RHS-relative, recommended for projection)
    //   3. ||r||/||r0|| ≤ tol_rel  (initial-residual relative, backup)
    // Check every check_interval cycles to reduce overhead.
    assert(cfg.max_vcycles > 0 && "PoissonConfig.max_vcycles must be positive");
    const int max_cycles = cfg.max_vcycles;
    const bool accurate_mode = (max_cycles > 5);
    // Optimal at 128³ channel: nu1=3, nu2=1 (more pre-smooth for wall BCs)
    const int nu1 = (cfg.nu1 > 0) ? cfg.nu1 : (accurate_mode ? 3 : 2);
    const int nu2 = (cfg.nu2 > 0) ? cfg.nu2 : 1;
    const int degree = cfg.chebyshev_degree;
    const int check_interval = std::max(1, cfg.check_interval);

    // Compute ||b||_∞ and ||b||_2 on device via fused reduction - store in members for diagnostics
    auto& finest_gpu = *levels_[0];
    const int Ng = finest_gpu.Ng;  // Use level's ghost width
    const int Nx_g = finest_gpu.Nx;
    const int Ny_g = finest_gpu.Ny;
    const int Nz_g = finest_gpu.Nz;
    const int stride_gpu = finest_gpu.stride;
    const int plane_stride_gpu = finest_gpu.plane_stride;
    const bool is_2d_gpu = finest_gpu.is2D();

    // NVHPC WORKAROUND: Use omp_get_mapped_ptr for actual device addresses
    double b_inf_local = 0.0;
    double b_sum_sq = 0.0;
    {
        int device = omp_get_default_device();
        const double* f_dev = static_cast<const double*>(omp_get_mapped_ptr(f_level0_ptr_, device));
        if (is_2d_gpu) {
            #pragma omp target teams distribute parallel for collapse(2) \
                is_device_ptr(f_dev) reduction(max: b_inf_local) reduction(+: b_sum_sq)
            for (int j = Ng; j < Ny_g + Ng; ++j) {
                for (int i = Ng; i < Nx_g + Ng; ++i) {
                    int idx = j * stride_gpu + i;
                    double val = f_dev[idx];
                    b_inf_local = std::max(b_inf_local, std::abs(val));
                    b_sum_sq += val * val;
                }
            }
        } else {
            #pragma omp target teams distribute parallel for collapse(3) \
                is_device_ptr(f_dev) reduction(max: b_inf_local) reduction(+: b_sum_sq)
            for (int k = Ng; k < Nz_g + Ng; ++k) {
                for (int j = Ng; j < Ny_g + Ng; ++j) {
                    for (int i = Ng; i < Nx_g + Ng; ++i) {
                        int idx = k * plane_stride_gpu + j * stride_gpu + i;
                        double val = f_dev[idx];
                        b_inf_local = std::max(b_inf_local, std::abs(val));
                        b_sum_sq += val * val;
                    }
                }
            }
        }
    }
    b_inf_ = b_inf_local;  // Store for diagnostics
    b_l2_ = std::sqrt(b_sum_sq);

    // Initial residual - use fused function to compute residual + both norms in single pass
    compute_residual_and_norms(0, r0_, r0_l2_);
    residual_ = r0_;
    residual_l2_ = r0_l2_;

    int cycles_used = 0;
    for (int cycle = 0; cycle < max_cycles; ++cycle) {
        vcycle(0, nu1, nu2, degree);
        cycles_used = cycle + 1;

        // Check convergence every check_interval cycles (reduces overhead)
        if ((cycle % check_interval) == (check_interval - 1) || cycle == max_cycles - 1) {
            // Fused residual + norm computation (single pass over memory, single GPU reduction)
            compute_residual_and_norms(0, residual_, residual_l2_);

            // Select norm for convergence based on config (L2 is smoother, less sensitive to hot cells)
            const double r_norm = cfg.use_l2_norm ? residual_l2_ : residual_;
            const double b_norm = cfg.use_l2_norm ? b_l2_ : b_inf_;
            const double r0_norm = cfg.use_l2_norm ? r0_l2_ : r0_;

            // Robust convergence check: any criterion triggers exit
            bool converged = false;
            if (cfg.tol_abs > 0.0 && r_norm <= cfg.tol_abs) {
                converged = true;  // Absolute tolerance met
            }
            if (cfg.tol_rhs > 0.0 && r_norm <= cfg.tol_rhs * (b_norm + 1e-30)) {
                converged = true;  // RHS-relative tolerance met
            }
            if (cfg.tol_rel > 0.0 && r_norm <= cfg.tol_rel * (r0_norm + 1e-30)) {
                converged = true;  // Initial-residual relative tolerance met
            }
            // Legacy: also check cfg.tol for backward compatibility
            if (cfg.tol > 0.0 && r_norm <= cfg.tol) {
                converged = true;
            }

            // L∞ safety cap: when using L2, also enforce loose L∞ bound to catch "bad cells"
            // This prevents L2 from hiding localized divergence spikes
            if (converged && cfg.use_l2_norm && cfg.linf_safety_factor > 0.0 && cfg.tol_rhs > 0.0) {
                double linf_ratio = residual_ / (b_inf_ + 1e-30);
                double linf_cap = cfg.tol_rhs * cfg.linf_safety_factor;
                if (linf_ratio > linf_cap) {
                    converged = false;  // L∞ still too high, keep iterating
                }
            }

            if (converged) {
                break;
            }
        }
    }

    // Subtract mean for singular Poisson problems (no Dirichlet BCs)
    // Handle nullspace for singular problems (pure Neumann/Periodic)
    fix_nullspace(0);

    // Copy result from multigrid level-0 buffer back to caller's present-mapped pointer
    // NVHPC WORKAROUND: Use omp_get_mapped_ptr for actual device addresses
    {
        int device = omp_get_default_device();
        double* p_dev = static_cast<double*>(omp_get_mapped_ptr(p_present, device));
        double* u_dev = static_cast<double*>(omp_get_mapped_ptr(u_level0_ptr_, device));
        #pragma omp target teams distribute parallel for is_device_ptr(p_dev, u_dev)
        for (size_t idx = 0; idx < total_size; ++idx) {
            p_dev[idx] = u_dev[idx];
        }
    }

    return cycles_used;  // Actual number of V-cycles executed
}

void MultigridPoissonSolver::initialize_gpu_buffers() {
    // Force OpenMP runtime initialization before checking devices
    omp_set_default_device(0);
    
    // Verify GPU is available (throws if not)
    gpu::verify_device_available();
    
    // Allocate persistent device storage for all levels
    u_ptrs_.resize(levels_.size());
    f_ptrs_.resize(levels_.size());
    r_ptrs_.resize(levels_.size());
    tmp_ptrs_.resize(levels_.size());  // Scratch buffer for Jacobi
    level_sizes_.resize(levels_.size());
    
    for (size_t lvl = 0; lvl < levels_.size(); ++lvl) {
        auto& grid = *levels_[lvl];
        // Use grid's total_size which accounts for its ghost width (Ng)
        const size_t total_size = grid.total_size;
        level_sizes_[lvl] = total_size;
        
        // Get pointers to CPU data
        u_ptrs_[lvl] = grid.u.data().data();
        f_ptrs_[lvl] = grid.f.data().data();
        r_ptrs_[lvl] = grid.r.data().data();
        
        // Allocate on device
        #pragma omp target enter data map(alloc: u_ptrs_[lvl][0:total_size])
        #pragma omp target enter data map(alloc: f_ptrs_[lvl][0:total_size])
        #pragma omp target enter data map(alloc: r_ptrs_[lvl][0:total_size])

        // Allocate device-only scratch buffer for Jacobi ping-pong
        int device_id = omp_get_default_device();
        tmp_ptrs_[lvl] = static_cast<double*>(
            omp_target_alloc(total_size * sizeof(double), device_id));
        if (tmp_ptrs_[lvl] == nullptr) {
            throw std::runtime_error("Failed to allocate Jacobi scratch buffer on GPU");
        }

        // Zero-initialize residual and scratch arrays to avoid garbage
        // NVHPC WORKAROUND: Use omp_get_mapped_ptr for actual device addresses
        double* r_ptr = static_cast<double*>(omp_get_mapped_ptr(r_ptrs_[lvl], device_id));
        double* tmp_ptr = tmp_ptrs_[lvl];  // Already a device pointer (omp_target_alloc)
        #pragma omp target teams distribute parallel for is_device_ptr(r_ptr, tmp_ptr)
        for (size_t idx = 0; idx < total_size; ++idx) {
            r_ptr[idx] = 0.0;
            tmp_ptr[idx] = 0.0;
        }
    }
    
    // Verify mappings succeeded
    if (!u_ptrs_.empty() && !gpu::is_pointer_present(u_ptrs_[0])) {
        throw std::runtime_error("GPU mapping failed despite device availability");
    }

    // NVHPC WORKAROUND: Set level-0 member pointers for direct use in target regions.
    // Local pointer copies from vectors get HOST addresses in NVHPC target regions.
    if (!u_ptrs_.empty()) {
        u_level0_ptr_ = u_ptrs_[0];
        f_level0_ptr_ = f_ptrs_[0];
        r_level0_ptr_ = r_ptrs_[0];
        level0_size_ = level_sizes_[0];
    }

    gpu_ready_ = true;
}

void MultigridPoissonSolver::cleanup_gpu_buffers() {
    assert(gpu_ready_ && "GPU must be initialized");

    int device_id = omp_get_default_device();
    for (size_t lvl = 0; lvl < levels_.size(); ++lvl) {
        const size_t total_size = level_sizes_[lvl];

        #pragma omp target exit data map(delete: u_ptrs_[lvl][0:total_size])
        #pragma omp target exit data map(delete: f_ptrs_[lvl][0:total_size])
        #pragma omp target exit data map(delete: r_ptrs_[lvl][0:total_size])

        // Free device-only scratch buffer
        if (tmp_ptrs_[lvl] != nullptr) {
            omp_target_free(tmp_ptrs_[lvl], device_id);
            tmp_ptrs_[lvl] = nullptr;
        }
    }

    gpu_ready_ = false;
}

void MultigridPoissonSolver::sync_level_to_gpu(int level) {
    assert(gpu_ready_ && "GPU must be initialized");
    const size_t total_size = level_sizes_[level];
    
    #pragma omp target update to(u_ptrs_[level][0:total_size])
    #pragma omp target update to(f_ptrs_[level][0:total_size])
    #pragma omp target update to(r_ptrs_[level][0:total_size])
}

void MultigridPoissonSolver::sync_level_from_gpu(int level) {
    assert(gpu_ready_ && "GPU must be initialized");
    const size_t total_size = level_sizes_[level];

    #pragma omp target update from(u_ptrs_[level][0:total_size])
    #pragma omp target update from(f_ptrs_[level][0:total_size])
    #pragma omp target update from(r_ptrs_[level][0:total_size])
}

void MultigridPoissonSolver::initialize_vcycle_graph(int nu1, int nu2, int degree) {
    // Initialize full V-cycle CUDA Graph
    // This captures the entire V-cycle (all levels, all operations) as a single graph
    if (!use_vcycle_graph_) return;

    // Convert BCs to CUDA enum
    auto to_cuda_bc = [](PoissonBC bc) -> mg_cuda::BC {
        switch (bc) {
            case PoissonBC::Dirichlet: return mg_cuda::BC::Dirichlet;
            case PoissonBC::Neumann: return mg_cuda::BC::Neumann;
            case PoissonBC::Periodic: return mg_cuda::BC::Periodic;
            default: return mg_cuda::BC::Neumann;
        }
    };

    // Build fingerprint to check if recapture is needed
    mg_cuda::VCycleGraphFingerprint new_fp;
    new_fp.num_levels = levels_.size();
    new_fp.degree = degree;
    new_fp.nu1 = nu1;
    new_fp.nu2 = nu2;
    new_fp.bc_x_lo = to_cuda_bc(bc_x_lo_);
    new_fp.bc_x_hi = to_cuda_bc(bc_x_hi_);
    new_fp.bc_y_lo = to_cuda_bc(bc_y_lo_);
    new_fp.bc_y_hi = to_cuda_bc(bc_y_hi_);
    new_fp.bc_z_lo = to_cuda_bc(bc_z_lo_);
    new_fp.bc_z_hi = to_cuda_bc(bc_z_hi_);
    new_fp.coarse_iters = 8;
    for (size_t lvl = 0; lvl < levels_.size(); ++lvl) {
        auto& grid = *levels_[lvl];
        double dx2 = grid.dx * grid.dx;
        double dy2 = grid.dy * grid.dy;
        double dz2 = grid.dz * grid.dz;
        double coeff = grid.is2D() ? (2.0/dx2 + 2.0/dy2) : (2.0/dx2 + 2.0/dy2 + 2.0/dz2);
        new_fp.level_sizes.push_back(level_sizes_[lvl]);
        new_fp.level_coeffs.push_back(coeff);
        new_fp.level_dx.push_back(dx2);
        new_fp.level_dy.push_back(dy2);
        new_fp.level_dz.push_back(dz2);
    }

    // Check if existing graph is still valid
    if (vcycle_graph_ && !vcycle_graph_->needs_recapture(new_fp)) {
        // Graph is still valid, no recapture needed
        return;
    }

    vcycle_graph_nu1_ = nu1;
    vcycle_graph_nu2_ = nu2;
    vcycle_graph_degree_ = degree;

    // Build level configurations for V-cycle graph
    std::vector<mg_cuda::VCycleLevelConfig> configs;
    for (size_t lvl = 0; lvl < levels_.size(); ++lvl) {
        auto& grid = *levels_[lvl];
        mg_cuda::VCycleLevelConfig cfg;
        cfg.Nx = grid.Nx;
        cfg.Ny = grid.Ny;
        cfg.Nz = grid.Nz;
        cfg.Ng = 1;
        cfg.dx2 = grid.dx * grid.dx;
        cfg.dy2 = grid.dy * grid.dy;
        cfg.dz2 = grid.dz * grid.dz;
        cfg.inv_dx2 = 1.0 / cfg.dx2;
        cfg.inv_dy2 = 1.0 / cfg.dy2;
        cfg.inv_dz2 = (grid.Nz == 1) ? 0.0 : 1.0 / cfg.dz2;  // Zero for 2D
        cfg.coeff = grid.is2D() ? (2.0/cfg.dx2 + 2.0/cfg.dy2)
                                : (2.0/cfg.dx2 + 2.0/cfg.dy2 + 2.0/cfg.dz2);
        cfg.total_size = level_sizes_[lvl];
        // Convert host pointers to device pointers for CUDA kernels
        cfg.u = gpu::get_device_ptr(u_ptrs_[lvl]);
        cfg.f = gpu::get_device_ptr(f_ptrs_[lvl]);
        cfg.r = gpu::get_device_ptr(r_ptrs_[lvl]);
        cfg.tmp = tmp_ptrs_[lvl];  // Already device pointer
        configs.push_back(cfg);
    }

    // Verify device pointers are valid - null indicates a bug in buffer initialization
    for (size_t lvl = 0; lvl < configs.size(); ++lvl) {
        if (!configs[lvl].u || !configs[lvl].f || !configs[lvl].r || !configs[lvl].tmp) {
            throw std::runtime_error(
                "[MG] FATAL: Null device pointer at level " + std::to_string(lvl) +
                ". GPU buffer initialization failed - check that omp target enter data "
                "was called in initialize_gpu_buffers().");
        }
    }

    // Create/recapture the V-cycle graph
    if (!vcycle_graph_) {
        vcycle_graph_ = std::make_unique<mg_cuda::CudaVCycleGraph>();
    }
    vcycle_graph_->initialize(
        configs, degree, nu1, nu2,
        to_cuda_bc(bc_x_lo_), to_cuda_bc(bc_x_hi_),
        to_cuda_bc(bc_y_lo_), to_cuda_bc(bc_y_hi_),
        to_cuda_bc(bc_z_lo_), to_cuda_bc(bc_z_hi_));

    std::cout << "[MG] Full V-cycle CUDA Graph "
              << (vcycle_graph_->is_valid() ? "captured" : "FAILED")
              << " for " << levels_.size()
              << " levels (nu1=" << nu1 << ", nu2=" << nu2 << ")\n";
}

void MultigridPoissonSolver::vcycle_graphed() {
    NVTX_SCOPE_POISSON("mg:vcycle_graphed");

    if (!vcycle_graph_ || !vcycle_graph_->is_valid()) {
        // Fall back to non-graphed V-cycle
        vcycle(0, vcycle_graph_nu1_, vcycle_graph_nu2_, vcycle_graph_degree_);
        return;
    }

#ifdef __NVCOMPILER
    // Get OpenMP's CUDA stream for launching the graph (NVHPC-specific)
    // This avoids cross-stream synchronization overhead
    cudaStream_t omp_stream = reinterpret_cast<cudaStream_t>(
        ompx_get_cuda_stream(omp_get_default_device(), /*sync=*/0));

    // Runtime stream validation with graceful fallback
    // A null stream would launch on CUDA default stream, causing potential race conditions
    // with OpenMP target regions. Instead of crashing in production, fall back to the
    // non-graphed path which is slower but correct.
    if (omp_stream == nullptr) {
        static bool warned = false;
        if (!warned) {
            std::cerr << "[MG] WARNING: OpenMP CUDA stream is null - falling back to non-graphed V-cycle\n"
                      << "    This may indicate a runtime issue. Performance will be degraded.\n";
            warned = true;
        }
        vcycle(0, vcycle_graph_nu1_, vcycle_graph_nu2_, vcycle_graph_degree_);
        return;
    }

    // Single graph launch for entire V-cycle
    vcycle_graph_->execute(omp_stream);
#else
    // Non-NVHPC compilers: fall back to non-graphed V-cycle
    // (ompx_get_cuda_stream is NVHPC-specific)
    vcycle(0, vcycle_graph_nu1_, vcycle_graph_nu2_, vcycle_graph_degree_);
#endif
}
#else
// CPU: Set raw pointers for unified code paths (no GPU mapping)
void MultigridPoissonSolver::initialize_gpu_buffers() {
    // Set up raw pointers so unified loops can use them on CPU
    u_ptrs_.resize(levels_.size());
    f_ptrs_.resize(levels_.size());
    r_ptrs_.resize(levels_.size());
    level_sizes_.resize(levels_.size());

    for (size_t lvl = 0; lvl < levels_.size(); ++lvl) {
        auto& grid = *levels_[lvl];
        // Use grid's total_size which accounts for its ghost width (Ng)
        level_sizes_[lvl] = grid.total_size;
        u_ptrs_[lvl] = grid.u.data().data();
        f_ptrs_[lvl] = grid.f.data().data();
        r_ptrs_[lvl] = grid.r.data().data();
    }
    gpu_ready_ = false;
}

void MultigridPoissonSolver::cleanup_gpu_buffers() {
    // No-op
}

void MultigridPoissonSolver::sync_level_to_gpu(int level) {
    (void)level;
    // No-op
}

void MultigridPoissonSolver::sync_level_from_gpu(int level) {
    (void)level;
    // No-op
}
#endif

} // namespace nncfd

