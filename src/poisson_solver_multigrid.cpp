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
#include <iomanip>  // for std::setprecision, std::fixed
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
    compute_gershgorin_bounds();  // Compute per-level Chebyshev eigenvalue bounds

    // Initialize y-metric pointers for non-uniform y-spacing at level 0
    // (D·G = L consistency for stretched grids in projection step)
    if (mesh_->is_y_stretched()) {
        y_stretched_ = true;
        yLap_aS_ = mesh_->yLap_aS.data();
        yLap_aN_ = mesh_->yLap_aN.data();
        yLap_aP_ = mesh_->yLap_aP.data();
        y_metrics_size_ = mesh_->total_Ny();

#ifdef USE_GPU_OFFLOAD
        // CRITICAL: Disable CUDA Graph for stretched grids
        // The graphed V-cycle uses uniform-y operators that don't have non-uniform y support yet
        use_vcycle_graph_ = false;
#endif

        // Chebyshev smoother is now enabled for stretched grids using Gershgorin bounds.
        // The compute_gershgorin_bounds() function computes λmax per level based on
        // actual operator coefficients, making Chebyshev safe for variable-coefficient Laplacians.
        // Jacobi fallback is available via MG_SMOOTHER=jacobi environment variable if needed.
        std::cout << "[MG] y-stretched mesh: disabled V-cycle CUDA Graph, using Chebyshev with Gershgorin bounds\n";
    }

    // Initialize GPU buffers (maps to device) OR set up raw pointers for CPU
    // This enables unified loops to use cached pointers on both CPU and GPU
    initialize_gpu_buffers();

    // V-cycle CUDA Graph is enabled by default (use_vcycle_graph_ = true in header)
    // Can be disabled via config.poisson_use_vcycle_graph = false or by y-stretching (above)
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
    const bool y_stretched = mesh_->is_y_stretched();

    // Finest level uses mesh's ghost width (may be >1 for O4 schemes)
    // Coarse levels use Ng=1 since MG internally uses O2 stencils
    const int ng_fine = mesh_->Nghost;
    constexpr int ng_coarse = 1;

    // Minimum coarse grid size (8x8 is GPU-friendly while giving good MG efficiency)
    constexpr int MIN_COARSE_SIZE = 8;

    // Finest level
    if (is_2d) {
        levels_.push_back(std::make_unique<GridLevel>(Nx, Ny, dx, dy, ng_fine));
    } else {
        levels_.push_back(std::make_unique<GridLevel>(Nx, Ny, Nz, dx, dy, dz, ng_fine));
    }

    // For y-stretched grids: use semi-coarsening (coarsen x/z only, keep y unchanged)
    // This maintains D·G = L consistency because y-metric coefficients are the same on all levels
    if (y_stretched) {
        semi_coarsening_ = true;
        std::cout << "[MG] Semi-coarsening mode for y-stretched mesh (coarsen x/z only, y unchanged)\n";

        // Semi-coarsening: only reduce Nx and Nz, keep Ny constant
        // dy stays the same (non-uniform), dx and dz double per level
        if (is_2d) {
            while (Nx > MIN_COARSE_SIZE) {
                Nx /= 2;
                dx *= 2.0;
                // Ny and dy unchanged
                levels_.push_back(std::make_unique<GridLevel>(Nx, Ny, dx, dy, ng_coarse));
            }
        } else {
            while (Nx > MIN_COARSE_SIZE && Nz > MIN_COARSE_SIZE) {
                Nx /= 2;
                Nz /= 2;
                dx *= 2.0;
                dz *= 2.0;
                // Ny and dy unchanged
                levels_.push_back(std::make_unique<GridLevel>(Nx, Ny, Nz, dx, dy, dz, ng_coarse));
            }
        }
        return;
    }

    // Standard full coarsening for uniform grids

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

void MultigridPoissonSolver::compute_gershgorin_bounds() {
    // Compute per-level Gershgorin bounds for D⁻¹A (Jacobi-preconditioned operator)
    // For Chebyshev smoothing, we need λmax such that eigenvalues of D⁻¹A ∈ [λmin, λmax]
    //
    // Gershgorin theorem: For each row i, eigenvalues lie in disk centered at
    // diagonal with radius = sum of |off-diagonals|. For D⁻¹A:
    //   - Diagonal of D⁻¹A is 1 (since D⁻¹D = I)
    //   - Off-diagonals of D⁻¹A are |a_ij|/d_i
    //   - λmax ≤ max_i(1 + Σ_j |a_ij|/d_i)
    //
    // For 7-point stencil: λmax ≤ 1 + max_i(off_diag_sum_i / diag_i)

    cheby_lambda_max_.resize(levels_.size());

    for (size_t level = 0; level < levels_.size(); ++level) {
        auto& grid = *levels_[level];
        const double dx2 = grid.dx * grid.dx;
        const double dy2 = grid.dy * grid.dy;
        const double dz2 = grid.dz * grid.dz;
        const bool is_2d = grid.is2D();
        const int Ny = grid.Ny;
        const int Ng = grid.Ng;

        // Check if this level uses non-uniform y (semi-coarsening keeps y-metrics)
        const bool use_nonuniform_y = y_stretched_ && (semi_coarsening_ || level == 0);

        // For semi-coarsening: coarse levels have Ng=1 but y-metrics are indexed for fine mesh
        const int Ng_fine = levels_[0]->Ng;
        const int y_metric_offset = use_nonuniform_y ? (Ng_fine - Ng) : 0;

        double max_ratio = 0.0;

        if (use_nonuniform_y) {
            // Non-uniform y: compute max over all interior cells
            for (int j = Ng; j < Ny + Ng; ++j) {
                int jm = j + y_metric_offset;
                double aS = yLap_aS_[jm];
                double aN = yLap_aN_[jm];

                // Diagonal and off-diagonal sum for this row
                double diag_j = is_2d ? (2.0/dx2 + (aS + aN))
                                      : (2.0/dx2 + (aS + aN) + 2.0/dz2);
                double off_sum = is_2d ? (2.0/dx2 + aS + aN)
                                       : (2.0/dx2 + aS + aN + 2.0/dz2);

                double ratio = off_sum / diag_j;
                max_ratio = std::max(max_ratio, ratio);
            }
        } else {
            // Uniform grid: ratio is exactly 1 for all interior cells
            // (off_sum = diag for 7-point stencil)
            max_ratio = 1.0;
        }

        // λmax = 1 + max_ratio, with safety margin
        // Add 10% margin for numerical safety (boundary modifications, rounding, precision)
        double raw_lambda_max = 1.0 + max_ratio;
        cheby_lambda_max_[level] = raw_lambda_max * 1.10;

        // Floor only (no ceiling) - never underestimate λmax, that causes divergence
        // If true λmax > computed, polynomial becomes conservative (slower but safe)
        // If true λmax < computed (clamp wrongly), we'd diverge
        cheby_lambda_max_[level] = std::max(1.8, cheby_lambda_max_[level]);
    }

    // Report computed bounds (raw values for diagnostics)
    std::cout << "[MG] Chebyshev Gershgorin bounds: ";
    for (size_t level = 0; level < levels_.size(); ++level) {
        std::cout << "L" << level << "=" << std::fixed << std::setprecision(3)
                  << cheby_lambda_max_[level];
        if (level < levels_.size() - 1) std::cout << ", ";
    }
    std::cout << std::defaultfloat << "\n";
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
                } else { // Dirichlet - mirror ghost g through boundary using interior cell Ng+g
                    u_ptr[idx_lo] = 2.0 * dval - u_ptr[j * stride + (Ng + g)];
                }

                // Right boundary - periodic wraps to left interior
                if (bc_x_hi == 2) { // Periodic
                    u_ptr[idx_hi] = u_ptr[j * stride + Ng + g];
                } else if (bc_x_hi == 1) { // Neumann (zero gradient)
                    u_ptr[idx_hi] = u_ptr[j * stride + (Nx + Ng - 1)];
                } else { // Dirichlet - mirror ghost g through boundary using interior cell (Nx+Ng-1-g)
                    u_ptr[idx_hi] = 2.0 * dval - u_ptr[j * stride + (Nx + Ng - 1 - g)];
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
                } else { // Dirichlet - mirror ghost g through boundary using interior cell Ng+g
                    u_ptr[idx_lo] = 2.0 * dval - u_ptr[(Ng + g) * stride + i];
                }

                // Top boundary - periodic wraps to bottom interior
                if (bc_y_hi == 2) { // Periodic
                    u_ptr[idx_hi] = u_ptr[(Ng + g) * stride + i];
                } else if (bc_y_hi == 1) { // Neumann (zero gradient)
                    u_ptr[idx_hi] = u_ptr[(Ny + Ng - 1) * stride + i];
                } else { // Dirichlet - mirror ghost g through boundary using interior cell (Ny+Ng-1-g)
                    u_ptr[idx_hi] = 2.0 * dval - u_ptr[(Ny + Ng - 1 - g) * stride + i];
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
                    } else { // Dirichlet - mirror ghost g through boundary using interior cell Ng+g
                        u_ptr[idx_lo] = 2.0 * dval - u_ptr[j * stride + (Ng + g)];
                    }

                    if (bc_x_hi == 2) {
                        u_ptr[idx_hi] = u_ptr[j * stride + Ng + g];
                    } else if (bc_x_hi == 1) {
                        u_ptr[idx_hi] = u_ptr[j * stride + (Nx + Ng - 1)];
                    } else { // Dirichlet - mirror ghost g through boundary using interior cell (Nx+Ng-1-g)
                        u_ptr[idx_hi] = 2.0 * dval - u_ptr[j * stride + (Nx + Ng - 1 - g)];
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
                    } else { // Dirichlet - mirror ghost g through boundary using interior cell Ng+g
                        u_ptr[idx_lo] = 2.0 * dval - u_ptr[(Ng + g) * stride + i];
                    }

                    if (bc_y_hi == 2) {
                        u_ptr[idx_hi] = u_ptr[(Ng + g) * stride + i];
                    } else if (bc_y_hi == 1) {
                        u_ptr[idx_hi] = u_ptr[(Ny + Ng - 1) * stride + i];
                    } else { // Dirichlet - mirror ghost g through boundary using interior cell (Ny+Ng-1-g)
                        u_ptr[idx_hi] = 2.0 * dval - u_ptr[(Ny + Ng - 1 - g) * stride + i];
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
                } else { // Dirichlet - mirror ghost g through boundary using interior cell Ng+g
                    u_ptr[idx_lo] = 2.0 * dval - u_ptr[k * plane_stride + j * stride + (Ng + g)];
                }

                // Right boundary - periodic wraps to left interior
                if (bc_x_hi == 2) {
                    u_ptr[idx_hi] = u_ptr[k * plane_stride + j * stride + Ng + g];
                } else if (bc_x_hi == 1) {
                    u_ptr[idx_hi] = u_ptr[k * plane_stride + j * stride + (Nx + Ng - 1)];
                } else { // Dirichlet - mirror ghost g through boundary using interior cell (Nx+Ng-1-g)
                    u_ptr[idx_hi] = 2.0 * dval - u_ptr[k * plane_stride + j * stride + (Nx + Ng - 1 - g)];
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
                } else { // Dirichlet - mirror ghost g through boundary using interior cell Ng+g
                    u_ptr[idx_lo] = 2.0 * dval - u_ptr[k * plane_stride + (Ng + g) * stride + i];
                }

                // Top boundary - periodic wraps to bottom interior
                if (bc_y_hi == 2) {
                    u_ptr[idx_hi] = u_ptr[k * plane_stride + (Ng + g) * stride + i];
                } else if (bc_y_hi == 1) {
                    u_ptr[idx_hi] = u_ptr[k * plane_stride + (Ny + Ng - 1) * stride + i];
                } else { // Dirichlet - mirror ghost g through boundary using interior cell (Ny+Ng-1-g)
                    u_ptr[idx_hi] = 2.0 * dval - u_ptr[k * plane_stride + (Ny + Ng - 1 - g) * stride + i];
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
                } else { // Dirichlet - mirror ghost g through boundary using interior cell Ng+g
                    u_ptr[idx_lo] = 2.0 * dval - u_ptr[(Ng + g) * plane_stride + j * stride + i];
                }

                // Front boundary - periodic wraps to back interior
                if (bc_z_hi == 2) {
                    u_ptr[idx_hi] = u_ptr[(Ng + g) * plane_stride + j * stride + i];
                } else if (bc_z_hi == 1) {
                    u_ptr[idx_hi] = u_ptr[(Nz + Ng - 1) * plane_stride + j * stride + i];
                } else { // Dirichlet - mirror ghost g through boundary using interior cell (Nz+Ng-1-g)
                    u_ptr[idx_hi] = 2.0 * dval - u_ptr[(Nz + Ng - 1 - g) * plane_stride + j * stride + i];
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

    // Check for non-uniform y (D·G = L consistency)
    // With semi-coarsening, ALL levels use the same y-metric coefficients
    const bool use_nonuniform_y = y_stretched_ && (semi_coarsening_ || level == 0);

    const int Ng = grid.Ng;  // Use level's ghost width
    const int Nx = grid.Nx;
    const int Ny = grid.Ny;
    const int Nz = grid.Nz;
    const int stride = grid.stride;
    const int plane_stride = grid.plane_stride;
    const size_t total_size = grid.total_size;

    // For semi-coarsening: coarse levels have Ng=1 but y-metrics are indexed for fine mesh (Ng=Ng_fine)
    const int Ng_fine = levels_[0]->Ng;
    const int y_metric_offset = use_nonuniform_y ? (Ng_fine - Ng) : 0;

    // Chebyshev eigenvalue bounds: use per-level Gershgorin bounds for D⁻¹A
    // λmin = 0.1 * λmax (conservative: wider interval → higher ω, but more forgiving of bound errors)
    // Can tune to 0.05 for performance once bounds are validated
    const double lambda_max = cheby_lambda_max_[level];
    const double lambda_min = 0.1 * lambda_max;
    const double d = (lambda_max + lambda_min) / 2.0;
    const double c = (lambda_max - lambda_min) / 2.0;

    // NVHPC WORKAROUND: Use omp_get_mapped_ptr to get actual device addresses.
    // Local pointer copies with map(present:) get HOST addresses in NVHPC.
#ifdef USE_GPU_OFFLOAD
    int device = omp_get_default_device();
    double* u_ptr = static_cast<double*>(omp_get_mapped_ptr(u_ptrs_[level], device));
    const double* f_ptr = static_cast<const double*>(omp_get_mapped_ptr(f_ptrs_[level], device));
    double* tmp_ptr = tmp_ptrs_[level];  // Already a raw device pointer (omp_target_alloc)
    const double* aS_ptr = use_nonuniform_y ? static_cast<const double*>(omp_get_mapped_ptr(const_cast<double*>(yLap_aS_), device)) : nullptr;
    const double* aN_ptr = use_nonuniform_y ? static_cast<const double*>(omp_get_mapped_ptr(const_cast<double*>(yLap_aN_), device)) : nullptr;
    // Note: Cannot use nowait here - Chebyshev iterations have data dependencies
    // Each iteration reads the result of the previous iteration
    #define CHEBY_TARGET_2D \
        _Pragma("omp target teams distribute parallel for collapse(2) is_device_ptr(u_ptr, f_ptr, tmp_ptr)")
    #define CHEBY_TARGET_2D_NONUNIF \
        _Pragma("omp target teams distribute parallel for collapse(2) is_device_ptr(u_ptr, f_ptr, tmp_ptr, aS_ptr, aN_ptr)")
    #define CHEBY_TARGET_3D \
        _Pragma("omp target teams distribute parallel for collapse(3) is_device_ptr(u_ptr, f_ptr, tmp_ptr)")
    #define CHEBY_TARGET_3D_NONUNIF \
        _Pragma("omp target teams distribute parallel for collapse(3) is_device_ptr(u_ptr, f_ptr, tmp_ptr, aS_ptr, aN_ptr)")
    #define CHEBY_TARGET_COPY \
        _Pragma("omp target teams distribute parallel for is_device_ptr(u_ptr, tmp_ptr)")
#else
    double* u_ptr = u_ptrs_[level];
    const double* f_ptr = f_ptrs_[level];
    double* tmp_ptr = r_ptrs_[level];  // Reuse r as scratch buffer on CPU
    const double* aS_ptr = yLap_aS_;
    const double* aN_ptr = yLap_aN_;
    #define CHEBY_TARGET_2D
    #define CHEBY_TARGET_2D_NONUNIF
    #define CHEBY_TARGET_3D
    #define CHEBY_TARGET_3D_NONUNIF
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

        // SAFETY: Clamp ω to prevent catastrophic over-relaxation
        // Even with correct bounds, numerical precision can cause issues at extremes
        constexpr double OMEGA_MAX = 2.0;
        omega = std::min(omega, OMEGA_MAX);

        // Steps 2-3: Compute Jacobi update and apply Chebyshev weight
        // Read from u (with valid BCs), write to tmp
        if (is_2d && use_nonuniform_y) {
            // 2D with non-uniform y-spacing (semi-coarsening in x only)
            CHEBY_TARGET_2D_NONUNIF
            for (int j = Ng; j < Ny + Ng; ++j) {
                for (int i = Ng; i < Nx + Ng; ++i) {
                    int idx = j * stride + i;
                    int jm = j + y_metric_offset;  // Map to fine mesh y-metric index
                    double u_old = u_ptr[idx];
                    double lap_x = (u_ptr[idx+1] + u_ptr[idx-1]) / dx2;
                    double lap_y = aS_ptr[jm] * u_ptr[idx-stride] + aN_ptr[jm] * u_ptr[idx+stride];
                    double diag_j = 2.0 / dx2 + (aS_ptr[jm] + aN_ptr[jm]);
                    double u_jacobi = (lap_x + lap_y - f_ptr[idx]) / diag_j;
                    tmp_ptr[idx] = (1.0 - omega) * u_old + omega * u_jacobi;
                }
            }
        } else if (is_2d) {
            // 2D with uniform y-spacing
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
        } else if (use_nonuniform_y) {
            // 3D with non-uniform y-spacing
            CHEBY_TARGET_3D_NONUNIF
            for (int kk = Ng; kk < Nz + Ng; ++kk) {
                for (int j = Ng; j < Ny + Ng; ++j) {
                    for (int i = Ng; i < Nx + Ng; ++i) {
                        int idx = kk * plane_stride + j * stride + i;
                        int jm = j + y_metric_offset;  // Map to fine mesh y-metric index
                        double u_old = u_ptr[idx];
                        double lap_x = (u_ptr[idx+1] + u_ptr[idx-1]) / dx2;
                        double lap_y = aS_ptr[jm] * u_ptr[idx-stride] + aN_ptr[jm] * u_ptr[idx+stride];
                        double lap_z = (u_ptr[idx+plane_stride] + u_ptr[idx-plane_stride]) / dz2;
                        double diag_j = 2.0 / dx2 + (aS_ptr[jm] + aN_ptr[jm]) + 2.0 / dz2;
                        double u_jacobi = (lap_x + lap_y + lap_z - f_ptr[idx]) / diag_j;
                        tmp_ptr[idx] = (1.0 - omega) * u_old + omega * u_jacobi;
                    }
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
    #undef CHEBY_TARGET_2D_NONUNIF
    #undef CHEBY_TARGET_3D
    #undef CHEBY_TARGET_3D_NONUNIF
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

    // Check for non-uniform y (D·G = L consistency)
    // With semi-coarsening, ALL levels use the same y-metric coefficients
    const bool use_nonuniform_y = y_stretched_ && (semi_coarsening_ || level == 0);

    const int Ng = grid.Ng;  // Use level's ghost width
    const int Nx = grid.Nx;
    const int Ny = grid.Ny;
    const int Nz = grid.Nz;
    const int stride = grid.stride;
    const int plane_stride = grid.plane_stride;
    [[maybe_unused]] const size_t total_size = grid.total_size;

    // For semi-coarsening: coarse levels have Ng=1 but y-metrics are indexed for fine mesh (Ng=Ng_fine)
    const int Ng_fine = levels_[0]->Ng;
    const int y_metric_offset = use_nonuniform_y ? (Ng_fine - Ng) : 0;

#ifdef USE_GPU_OFFLOAD
    // NVHPC WORKAROUND: Use omp_get_mapped_ptr to get actual device addresses.
    // Local pointer copies with map(present:) get HOST addresses in NVHPC.
    int device = omp_get_default_device();
    double* u_ptr = static_cast<double*>(omp_get_mapped_ptr(u_ptrs_[level], device));
    const double* f_ptr = static_cast<const double*>(omp_get_mapped_ptr(f_ptrs_[level], device));
    double* tmp_ptr = tmp_ptrs_[level];  // Already a raw device pointer (omp_target_alloc)
    const double* aS_ptr = use_nonuniform_y ? static_cast<const double*>(omp_get_mapped_ptr(const_cast<double*>(yLap_aS_), device)) : nullptr;
    const double* aN_ptr = use_nonuniform_y ? static_cast<const double*>(omp_get_mapped_ptr(const_cast<double*>(yLap_aN_), device)) : nullptr;
    // All pointers are now device pointers - use is_device_ptr for all
    #define JACOBI_TARGET_U_TO_TMP_2D \
        _Pragma("omp target teams distribute parallel for collapse(2) is_device_ptr(u_ptr, f_ptr, tmp_ptr)")
    #define JACOBI_TARGET_TMP_TO_U_2D \
        _Pragma("omp target teams distribute parallel for collapse(2) is_device_ptr(u_ptr, f_ptr, tmp_ptr)")
    #define JACOBI_TARGET_U_TO_TMP_2D_NONUNIF \
        _Pragma("omp target teams distribute parallel for collapse(2) is_device_ptr(u_ptr, f_ptr, tmp_ptr, aS_ptr, aN_ptr)")
    #define JACOBI_TARGET_TMP_TO_U_2D_NONUNIF \
        _Pragma("omp target teams distribute parallel for collapse(2) is_device_ptr(u_ptr, f_ptr, tmp_ptr, aS_ptr, aN_ptr)")
    #define JACOBI_TARGET_U_TO_TMP_3D \
        _Pragma("omp target teams distribute parallel for collapse(3) is_device_ptr(u_ptr, f_ptr, tmp_ptr)")
    #define JACOBI_TARGET_TMP_TO_U_3D \
        _Pragma("omp target teams distribute parallel for collapse(3) is_device_ptr(u_ptr, f_ptr, tmp_ptr)")
    #define JACOBI_TARGET_U_TO_TMP_3D_NONUNIF \
        _Pragma("omp target teams distribute parallel for collapse(3) is_device_ptr(u_ptr, f_ptr, tmp_ptr, aS_ptr, aN_ptr)")
    #define JACOBI_TARGET_TMP_TO_U_3D_NONUNIF \
        _Pragma("omp target teams distribute parallel for collapse(3) is_device_ptr(u_ptr, f_ptr, tmp_ptr, aS_ptr, aN_ptr)")
    #define JACOBI_TARGET_COPY \
        _Pragma("omp target teams distribute parallel for is_device_ptr(u_ptr, tmp_ptr)")
#else
    // CPU path: use host pointers directly
    double* u_ptr = u_ptrs_[level];
    const double* f_ptr = f_ptrs_[level];
    double* tmp_ptr = r_ptrs_[level];  // Reuse r as scratch buffer on CPU
    const double* aS_ptr = yLap_aS_;
    const double* aN_ptr = yLap_aN_;
    // CPU: no pragmas needed
    #define JACOBI_TARGET_U_TO_TMP_2D
    #define JACOBI_TARGET_TMP_TO_U_2D
    #define JACOBI_TARGET_U_TO_TMP_2D_NONUNIF
    #define JACOBI_TARGET_TMP_TO_U_2D_NONUNIF
    #define JACOBI_TARGET_U_TO_TMP_3D
    #define JACOBI_TARGET_TMP_TO_U_3D
    #define JACOBI_TARGET_U_TO_TMP_3D_NONUNIF
    #define JACOBI_TARGET_TMP_TO_U_3D_NONUNIF
    #define JACOBI_TARGET_COPY
#endif

    // Ping-pong Jacobi: u -> tmp, then tmp -> u
    // Identical algorithm for CPU and GPU - only pragmas differ
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            // u -> tmp
            if (is_2d && use_nonuniform_y) {
                // 2D with non-uniform y-spacing (semi-coarsening in x only)
                JACOBI_TARGET_U_TO_TMP_2D_NONUNIF
                for (int j = Ng; j < Ny + Ng; ++j) {
                    for (int i = Ng; i < Nx + Ng; ++i) {
                        int idx = j * stride + i;
                        int jm = j + y_metric_offset;  // Map to fine mesh y-metric index
                        double u_old = u_ptr[idx];
                        double lap_x = (u_ptr[idx+1] + u_ptr[idx-1]) / dx2;
                        double lap_y = aS_ptr[jm] * u_ptr[idx-stride] + aN_ptr[jm] * u_ptr[idx+stride];
                        double diag_j = 2.0 / dx2 + (aS_ptr[jm] + aN_ptr[jm]);
                        double u_jacobi = (lap_x + lap_y - f_ptr[idx]) / diag_j;
                        tmp_ptr[idx] = (1.0 - omega) * u_old + omega * u_jacobi;
                    }
                }
            } else if (is_2d) {
                // 2D with uniform y-spacing
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
            } else if (use_nonuniform_y) {
                // 3D with non-uniform y-spacing
                // Jacobi stencil: lap_y = aS*u[j-1] + aN*u[j+1], diag = (aS + aN) + 2/dx² + 2/dz²
                JACOBI_TARGET_U_TO_TMP_3D_NONUNIF
                for (int k = Ng; k < Nz + Ng; ++k) {
                    for (int j = Ng; j < Ny + Ng; ++j) {
                        for (int i = Ng; i < Nx + Ng; ++i) {
                            int idx = k * plane_stride + j * stride + i;
                            int jm = j + y_metric_offset;  // Map to fine mesh y-metric index
                            double u_old = u_ptr[idx];
                            double lap_x = (u_ptr[idx+1] + u_ptr[idx-1]) / dx2;
                            double lap_y = aS_ptr[jm] * u_ptr[idx-stride] + aN_ptr[jm] * u_ptr[idx+stride];
                            double lap_z = (u_ptr[idx+plane_stride] + u_ptr[idx-plane_stride]) / dz2;
                            double diag_j = 2.0 / dx2 + (aS_ptr[jm] + aN_ptr[jm]) + 2.0 / dz2;
                            double u_jacobi = (lap_x + lap_y + lap_z - f_ptr[idx]) / diag_j;
                            tmp_ptr[idx] = (1.0 - omega) * u_old + omega * u_jacobi;
                        }
                    }
                }
            } else {
                // 3D with uniform spacing
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
            if (is_2d && use_nonuniform_y) {
                // 2D with non-uniform y-spacing (semi-coarsening in x only)
                JACOBI_TARGET_TMP_TO_U_2D_NONUNIF
                for (int j = Ng; j < Ny + Ng; ++j) {
                    for (int i = Ng; i < Nx + Ng; ++i) {
                        int idx = j * stride + i;
                        int jm = j + y_metric_offset;  // Map to fine mesh y-metric index
                        double u_old = tmp_ptr[idx];
                        double lap_x = (tmp_ptr[idx+1] + tmp_ptr[idx-1]) / dx2;
                        double lap_y = aS_ptr[jm] * tmp_ptr[idx-stride] + aN_ptr[jm] * tmp_ptr[idx+stride];
                        double diag_j = 2.0 / dx2 + (aS_ptr[jm] + aN_ptr[jm]);
                        double u_jacobi = (lap_x + lap_y - f_ptr[idx]) / diag_j;
                        u_ptr[idx] = (1.0 - omega) * u_old + omega * u_jacobi;
                    }
                }
            } else if (is_2d) {
                // 2D with uniform y-spacing
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
            } else if (use_nonuniform_y) {
                // 3D with non-uniform y-spacing
                JACOBI_TARGET_TMP_TO_U_3D_NONUNIF
                for (int k = Ng; k < Nz + Ng; ++k) {
                    for (int j = Ng; j < Ny + Ng; ++j) {
                        for (int i = Ng; i < Nx + Ng; ++i) {
                            int idx = k * plane_stride + j * stride + i;
                            int jm = j + y_metric_offset;  // Map to fine mesh y-metric index
                            double u_old = tmp_ptr[idx];
                            double lap_x = (tmp_ptr[idx+1] + tmp_ptr[idx-1]) / dx2;
                            double lap_y = aS_ptr[jm] * tmp_ptr[idx-stride] + aN_ptr[jm] * tmp_ptr[idx+stride];
                            double lap_z = (tmp_ptr[idx+plane_stride] + tmp_ptr[idx-plane_stride]) / dz2;
                            double diag_j = 2.0 / dx2 + (aS_ptr[jm] + aN_ptr[jm]) + 2.0 / dz2;
                            double u_jacobi = (lap_x + lap_y + lap_z - f_ptr[idx]) / diag_j;
                            u_ptr[idx] = (1.0 - omega) * u_old + omega * u_jacobi;
                        }
                    }
                }
            } else {
                // 3D with uniform spacing
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
    #undef JACOBI_TARGET_U_TO_TMP_2D_NONUNIF
    #undef JACOBI_TARGET_TMP_TO_U_2D_NONUNIF
    #undef JACOBI_TARGET_U_TO_TMP_3D
    #undef JACOBI_TARGET_TMP_TO_U_3D
    #undef JACOBI_TARGET_U_TO_TMP_3D_NONUNIF
    #undef JACOBI_TARGET_TMP_TO_U_3D_NONUNIF
    #undef JACOBI_TARGET_COPY
}

void MultigridPoissonSolver::smooth_xz_plane_rbgs(int level, int iterations) {
    // Red-Black Gauss-Seidel in x-z planes (for each fixed y-row j)
    // Targets modes that are constant in y but oscillatory in x/z
    // These modes pass through y-line smoothing unchanged, causing MG stalls
    //
    // For each plane at y=j, we solve: (Lx + Lz)u = f - Ly*u
    // where Ly*u is frozen from the previous y-line sweep
    // Red: (i+k) % 2 == 0, Black: (i+k) % 2 == 1
    NVTX_SCOPE_POISSON("mg:smooth_xz_rbgs");

    auto& grid = *levels_[level];
    const double dx2 = grid.dx * grid.dx;
    const double dz2 = grid.dz * grid.dz;
    const int Ng = grid.Ng;
    const int Nx = grid.Nx;
    const int Ny = grid.Ny;
    const int Nz = grid.Nz;
    const int stride = grid.stride;
    const int plane_stride = grid.plane_stride;

    // Check for non-uniform y (D·G = L consistency)
    const bool use_nonuniform_y = y_stretched_ && (semi_coarsening_ || level == 0);

    // For semi-coarsening: coarse levels have Ng=1 but y-metrics are indexed for fine mesh
    const int Ng_fine = levels_[0]->Ng;
    const int y_metric_offset = use_nonuniform_y ? (Ng_fine - Ng) : 0;

    // Uniform y coefficients (fallback)
    const double dy2 = grid.dy * grid.dy;

#ifdef USE_GPU_OFFLOAD
    int device = omp_get_default_device();
    double* u_ptr = static_cast<double*>(omp_get_mapped_ptr(u_ptrs_[level], device));
    const double* f_ptr = static_cast<const double*>(omp_get_mapped_ptr(f_ptrs_[level], device));
    const double* aS_ptr = use_nonuniform_y ? static_cast<const double*>(omp_get_mapped_ptr(const_cast<double*>(yLap_aS_), device)) : nullptr;
    const double* aN_ptr = use_nonuniform_y ? static_cast<const double*>(omp_get_mapped_ptr(const_cast<double*>(yLap_aN_), device)) : nullptr;

    for (int iter = 0; iter < iterations; ++iter) {
        // Red sweep: (i + k) % 2 == 0
        if (use_nonuniform_y) {
            #pragma omp target teams distribute parallel for collapse(3) is_device_ptr(u_ptr, f_ptr, aS_ptr, aN_ptr)
            for (int k = Ng; k < Nz + Ng; ++k) {
                for (int j = Ng; j < Ny + Ng; ++j) {
                    for (int i = Ng; i < Nx + Ng; ++i) {
                        if ((i + k) % 2 == 0) {
                            int idx = k * plane_stride + j * stride + i;
                            int jm = j + y_metric_offset;
                            // x/z neighbors (in-place update, but red-black ensures no conflict)
                            double lap_x = (u_ptr[idx+1] + u_ptr[idx-1]) / dx2;
                            double lap_z = (u_ptr[idx+plane_stride] + u_ptr[idx-plane_stride]) / dz2;
                            // y neighbors (frozen from y-line sweep)
                            double lap_y = aS_ptr[jm] * u_ptr[idx-stride] + aN_ptr[jm] * u_ptr[idx+stride];
                            double diag = 2.0/dx2 + (aS_ptr[jm] + aN_ptr[jm]) + 2.0/dz2;
                            u_ptr[idx] = (lap_x + lap_y + lap_z - f_ptr[idx]) / diag;
                        }
                    }
                }
            }
        } else {
            double coeff = 2.0/dx2 + 2.0/dy2 + 2.0/dz2;
            #pragma omp target teams distribute parallel for collapse(3) is_device_ptr(u_ptr, f_ptr)
            for (int k = Ng; k < Nz + Ng; ++k) {
                for (int j = Ng; j < Ny + Ng; ++j) {
                    for (int i = Ng; i < Nx + Ng; ++i) {
                        if ((i + k) % 2 == 0) {
                            int idx = k * plane_stride + j * stride + i;
                            double lap_x = (u_ptr[idx+1] + u_ptr[idx-1]) / dx2;
                            double lap_z = (u_ptr[idx+plane_stride] + u_ptr[idx-plane_stride]) / dz2;
                            double lap_y = (u_ptr[idx+stride] + u_ptr[idx-stride]) / dy2;
                            u_ptr[idx] = (lap_x + lap_y + lap_z - f_ptr[idx]) / coeff;
                        }
                    }
                }
            }
        }

        // Black sweep: (i + k) % 2 == 1
        if (use_nonuniform_y) {
            #pragma omp target teams distribute parallel for collapse(3) is_device_ptr(u_ptr, f_ptr, aS_ptr, aN_ptr)
            for (int k = Ng; k < Nz + Ng; ++k) {
                for (int j = Ng; j < Ny + Ng; ++j) {
                    for (int i = Ng; i < Nx + Ng; ++i) {
                        if ((i + k) % 2 == 1) {
                            int idx = k * plane_stride + j * stride + i;
                            int jm = j + y_metric_offset;
                            double lap_x = (u_ptr[idx+1] + u_ptr[idx-1]) / dx2;
                            double lap_z = (u_ptr[idx+plane_stride] + u_ptr[idx-plane_stride]) / dz2;
                            double lap_y = aS_ptr[jm] * u_ptr[idx-stride] + aN_ptr[jm] * u_ptr[idx+stride];
                            double diag = 2.0/dx2 + (aS_ptr[jm] + aN_ptr[jm]) + 2.0/dz2;
                            u_ptr[idx] = (lap_x + lap_y + lap_z - f_ptr[idx]) / diag;
                        }
                    }
                }
            }
        } else {
            double coeff = 2.0/dx2 + 2.0/dy2 + 2.0/dz2;
            #pragma omp target teams distribute parallel for collapse(3) is_device_ptr(u_ptr, f_ptr)
            for (int k = Ng; k < Nz + Ng; ++k) {
                for (int j = Ng; j < Ny + Ng; ++j) {
                    for (int i = Ng; i < Nx + Ng; ++i) {
                        if ((i + k) % 2 == 1) {
                            int idx = k * plane_stride + j * stride + i;
                            double lap_x = (u_ptr[idx+1] + u_ptr[idx-1]) / dx2;
                            double lap_z = (u_ptr[idx+plane_stride] + u_ptr[idx-plane_stride]) / dz2;
                            double lap_y = (u_ptr[idx+stride] + u_ptr[idx-stride]) / dy2;
                            u_ptr[idx] = (lap_x + lap_y + lap_z - f_ptr[idx]) / coeff;
                        }
                    }
                }
            }
        }

        // Apply BCs after each RB iteration
        apply_bc(level);
    }
#else
    // CPU path
    double* u_ptr = u_ptrs_[level];
    const double* f_ptr = f_ptrs_[level];
    const double* aS_ptr = yLap_aS_;
    const double* aN_ptr = yLap_aN_;
    double coeff = 2.0/dx2 + 2.0/dy2 + 2.0/dz2;

    for (int iter = 0; iter < iterations; ++iter) {
        // Red sweep
        for (int k = Ng; k < Nz + Ng; ++k) {
            for (int j = Ng; j < Ny + Ng; ++j) {
                for (int i = Ng; i < Nx + Ng; ++i) {
                    if ((i + k) % 2 == 0) {
                        int idx = k * plane_stride + j * stride + i;
                        double lap_x = (u_ptr[idx+1] + u_ptr[idx-1]) / dx2;
                        double lap_z = (u_ptr[idx+plane_stride] + u_ptr[idx-plane_stride]) / dz2;
                        double lap_y, diag;
                        if (use_nonuniform_y) {
                            int jm = j + y_metric_offset;
                            lap_y = aS_ptr[jm] * u_ptr[idx-stride] + aN_ptr[jm] * u_ptr[idx+stride];
                            diag = 2.0/dx2 + (aS_ptr[jm] + aN_ptr[jm]) + 2.0/dz2;
                        } else {
                            lap_y = (u_ptr[idx+stride] + u_ptr[idx-stride]) / dy2;
                            diag = coeff;
                        }
                        u_ptr[idx] = (lap_x + lap_y + lap_z - f_ptr[idx]) / diag;
                    }
                }
            }
        }
        // Black sweep
        for (int k = Ng; k < Nz + Ng; ++k) {
            for (int j = Ng; j < Ny + Ng; ++j) {
                for (int i = Ng; i < Nx + Ng; ++i) {
                    if ((i + k) % 2 == 1) {
                        int idx = k * plane_stride + j * stride + i;
                        double lap_x = (u_ptr[idx+1] + u_ptr[idx-1]) / dx2;
                        double lap_z = (u_ptr[idx+plane_stride] + u_ptr[idx-plane_stride]) / dz2;
                        double lap_y, diag;
                        if (use_nonuniform_y) {
                            int jm = j + y_metric_offset;
                            lap_y = aS_ptr[jm] * u_ptr[idx-stride] + aN_ptr[jm] * u_ptr[idx+stride];
                            diag = 2.0/dx2 + (aS_ptr[jm] + aN_ptr[jm]) + 2.0/dz2;
                        } else {
                            lap_y = (u_ptr[idx+stride] + u_ptr[idx-stride]) / dy2;
                            diag = coeff;
                        }
                        u_ptr[idx] = (lap_x + lap_y + lap_z - f_ptr[idx]) / diag;
                    }
                }
            }
        }
        apply_bc(level);
    }
#endif
}

void MultigridPoissonSolver::smooth_y_lines(int level, int iterations) {
    // Y-line relaxation: solve tridiagonal systems along y-lines (Line-Jacobi)
    // This is optimal for anisotropic problems where y-direction dominates.
    // For each (i,k) pair, solve: -aS[j]*u[j-1] + diag[j]*u[j] - aN[j]*u[j+1] = rhs[j]
    // where rhs[j] = f[j] - x_laplacian(u) - z_laplacian(u) (neighbors frozen)
    // Thomas algorithm: O(Ny) per line, Nx*Nz independent lines (massive GPU parallelism)
    NVTX_SCOPE_POISSON("mg:smooth_yline");

    auto& grid = *levels_[level];
    const int Nx = grid.Nx;
    const int Ny = grid.Ny;
    const int Nz = grid.Nz;
    const int Ng = grid.Ng;
    const int stride = grid.stride;
    const int plane_stride = grid.plane_stride;
    const double dx2 = grid.dx * grid.dx;
    const double dz2 = grid.dz * grid.dz;
    const bool is_2d = grid.is2D();

    // Y-metric coefficients
    const int Ng_fine = levels_[0]->Ng;
    const int y_metric_offset = (Ng_fine - Ng);

    const double inv_dx2 = 1.0 / dx2;
    const double inv_dz2 = is_2d ? 0.0 : (1.0 / dz2);

    // BC type affects diagonal modification at walls
    // For Neumann: ghost = interior, so wall-side coefficient is absorbed into diagonal -> modify diagonal
    // For Dirichlet: ghost is fixed, no absorption -> keep full diagonal
    const bool neumann_y_lo = (bc_y_lo_ == PoissonBC::Neumann);
    const bool neumann_y_hi = (bc_y_hi_ == PoissonBC::Neumann);

#ifdef USE_GPU_OFFLOAD
    // Max Ny we support for stack-allocated work arrays (128 is safe for occupancy)
    // Above this, local memory spills can hurt performance
    constexpr int MAX_NY = 128;
    constexpr double PIVOT_EPS = 1e-14;  // Guard against tiny pivots in Thomas
    if (Ny <= MAX_NY) {
        // GPU path: parallel Thomas solve per (i,k) line - NO host transfers
        // Each thread handles one complete y-line with thread-local work arrays
        int device = omp_get_default_device();
        double* u_ptr_gpu = static_cast<double*>(omp_get_mapped_ptr(u_ptrs_[level], device));
        const double* f_ptr_gpu = static_cast<const double*>(omp_get_mapped_ptr(f_ptrs_[level], device));
        const double* aS_ptr_gpu = static_cast<const double*>(omp_get_mapped_ptr(const_cast<double*>(yLap_aS_), device));
        const double* aN_ptr_gpu = static_cast<const double*>(omp_get_mapped_ptr(const_cast<double*>(yLap_aN_), device));

        for (int iter = 0; iter < iterations; ++iter) {
            if (is_2d) {
                // 2D: parallelize over i (Nx lines)
                #pragma omp target teams distribute parallel for is_device_ptr(u_ptr_gpu, f_ptr_gpu, aS_ptr_gpu, aN_ptr_gpu)
                for (int i = Ng; i < Nx + Ng; ++i) {
                    // Thread-local work arrays on GPU stack
                    double c_prime[MAX_NY];
                    double d_prime[MAX_NY];

                    // Forward sweep (Thomas algorithm)
                    // Solving -L(u) = -f with positive diagonal for stability:
                    //   -aS*u[j-1] + (aS+aN+2/dx²)*u[j] - aN*u[j+1] = -f + x_neighbors
                    //
                    // WALL BC HANDLING:
                    // - Neumann (du/dy=0): ghost = interior, so wall-side coeff absorbs into diagonal
                    // - Dirichlet: ghost is fixed BC value, no diagonal modification needed
                    for (int j = Ng; j < Ny + Ng; ++j) {
                        int jm = j + y_metric_offset;
                        int idx = j * stride + i;

                        const bool at_south_wall = (j == Ng);
                        const bool at_north_wall = (j == Ny + Ng - 1);

                        // Off-diagonal: always 0 at walls (ghost not in tridiagonal solve)
                        double a = at_south_wall ? 0.0 : -aS_ptr_gpu[jm];
                        double c = at_north_wall ? 0.0 : -aN_ptr_gpu[jm];
                        // Diagonal: only modify for Neumann BC (absorption into diagonal)
                        double b = ((at_south_wall && neumann_y_lo) ? 0.0 : aS_ptr_gpu[jm])
                                 + ((at_north_wall && neumann_y_hi) ? 0.0 : aN_ptr_gpu[jm])
                                 + 2.0 * inv_dx2;

                        // RHS: -f + x_neighbors (consistent with -L(u) = -f form)
                        double rhs = inv_dx2 * (u_ptr_gpu[idx+1] + u_ptr_gpu[idx-1]) - f_ptr_gpu[idx];
                        // For Dirichlet BC: add ghost contribution to RHS (moved from tridiagonal)
                        if (at_south_wall && !neumann_y_lo) rhs += aS_ptr_gpu[jm] * u_ptr_gpu[idx - stride];
                        if (at_north_wall && !neumann_y_hi) rhs += aN_ptr_gpu[jm] * u_ptr_gpu[idx + stride];

                        int j_local = j - Ng;
                        if (j_local == 0) {
                            // Guard against tiny diagonal (shouldn't happen for valid Poisson)
                            double b_safe = (b > PIVOT_EPS) ? b : PIVOT_EPS;
                            c_prime[0] = c / b_safe;
                            d_prime[0] = rhs / b_safe;
                        } else {
                            double denom = b - a * c_prime[j_local - 1];
                            // Clamp tiny pivots to avoid NaN (system should be diagonally dominant)
                            double denom_safe = (denom > PIVOT_EPS || denom < -PIVOT_EPS) ? denom :
                                                (denom >= 0 ? PIVOT_EPS : -PIVOT_EPS);
                            c_prime[j_local] = c / denom_safe;
                            d_prime[j_local] = (rhs - a * d_prime[j_local - 1]) / denom_safe;
                        }
                    }

                    // Backward substitution
                    for (int j = Ny + Ng - 1; j >= Ng; --j) {
                        int j_local = j - Ng;
                        int idx = j * stride + i;
                        if (j_local == Ny - 1) {
                            u_ptr_gpu[idx] = d_prime[j_local];
                        } else {
                            u_ptr_gpu[idx] = d_prime[j_local] - c_prime[j_local] * u_ptr_gpu[idx + stride];
                        }
                    }
                }
            } else {
                // 3D: parallelize over (i,k) pairs (Nx*Nz lines)
                const int num_lines = Nx * Nz;
                #pragma omp target teams distribute parallel for is_device_ptr(u_ptr_gpu, f_ptr_gpu, aS_ptr_gpu, aN_ptr_gpu)
                for (int line_idx = 0; line_idx < num_lines; ++line_idx) {
                    int i = (line_idx % Nx) + Ng;
                    int k = (line_idx / Nx) + Ng;

                    // Thread-local work arrays on GPU stack
                    double c_prime[MAX_NY];
                    double d_prime[MAX_NY];

                    // Forward sweep (Thomas algorithm)
                    // Solving -L(u) = -f with positive diagonal for stability:
                    //   -aS*u[j-1] + (aS+aN+2/dx²+2/dz²)*u[j] - aN*u[j+1] = -f + x_neighbors + z_neighbors
                    //
                    // WALL BC HANDLING:
                    // - Neumann (du/dy=0): ghost = interior, so wall-side coeff absorbs into diagonal
                    // - Dirichlet: ghost is fixed BC value, no diagonal modification needed
                    for (int j = Ng; j < Ny + Ng; ++j) {
                        int jm = j + y_metric_offset;
                        int idx = k * plane_stride + j * stride + i;

                        const bool at_south_wall = (j == Ng);
                        const bool at_north_wall = (j == Ny + Ng - 1);

                        // Off-diagonal: always 0 at walls (ghost not in tridiagonal solve)
                        double a = at_south_wall ? 0.0 : -aS_ptr_gpu[jm];
                        double c = at_north_wall ? 0.0 : -aN_ptr_gpu[jm];
                        // Diagonal: only modify for Neumann BC (absorption into diagonal)
                        double b = ((at_south_wall && neumann_y_lo) ? 0.0 : aS_ptr_gpu[jm])
                                 + ((at_north_wall && neumann_y_hi) ? 0.0 : aN_ptr_gpu[jm])
                                 + 2.0 * inv_dx2 + 2.0 * inv_dz2;

                        // RHS: -f + x_neighbors + z_neighbors (consistent with -L(u) = -f form)
                        double rhs = inv_dx2 * (u_ptr_gpu[idx+1] + u_ptr_gpu[idx-1])
                                   + inv_dz2 * (u_ptr_gpu[idx+plane_stride] + u_ptr_gpu[idx-plane_stride])
                                   - f_ptr_gpu[idx];
                        // For Dirichlet BC: add ghost contribution to RHS (moved from tridiagonal)
                        if (at_south_wall && !neumann_y_lo) rhs += aS_ptr_gpu[jm] * u_ptr_gpu[idx - stride];
                        if (at_north_wall && !neumann_y_hi) rhs += aN_ptr_gpu[jm] * u_ptr_gpu[idx + stride];

                        int j_local = j - Ng;
                        if (j_local == 0) {
                            double b_safe = (b > PIVOT_EPS) ? b : PIVOT_EPS;
                            c_prime[0] = c / b_safe;
                            d_prime[0] = rhs / b_safe;
                        } else {
                            double denom = b - a * c_prime[j_local - 1];
                            double denom_safe = (denom > PIVOT_EPS || denom < -PIVOT_EPS) ? denom :
                                                (denom >= 0 ? PIVOT_EPS : -PIVOT_EPS);
                            c_prime[j_local] = c / denom_safe;
                            d_prime[j_local] = (rhs - a * d_prime[j_local - 1]) / denom_safe;
                        }
                    }

                    // Backward substitution
                    for (int j = Ny + Ng - 1; j >= Ng; --j) {
                        int j_local = j - Ng;
                        int idx = k * plane_stride + j * stride + i;
                        if (j_local == Ny - 1) {
                            u_ptr_gpu[idx] = d_prime[j_local];
                        } else {
                            u_ptr_gpu[idx] = d_prime[j_local] - c_prime[j_local] * u_ptr_gpu[idx + stride];
                        }
                    }
                }
            }

            apply_bc(level);
        }
        return;  // GPU path complete
    }
#endif

    // CPU fallback path (for non-GPU builds or Ny > MAX_NY)
    constexpr double PIVOT_EPS_CPU = 1e-14;  // Same threshold as GPU
    double* u_ptr = u_ptrs_[level];
    const double* f_ptr = f_ptrs_[level];
    const double* aS_ptr = yLap_aS_;
    const double* aN_ptr = yLap_aN_;

    std::vector<double> c_prime(Ny);
    std::vector<double> d_prime(Ny);

    for (int iter = 0; iter < iterations; ++iter) {
        if (is_2d) {
            for (int i = Ng; i < Nx + Ng; ++i) {
                for (int j = Ng; j < Ny + Ng; ++j) {
                    int jm = j + y_metric_offset;
                    int idx = j * stride + i;

                    const bool at_south_wall = (j == Ng);
                    const bool at_north_wall = (j == Ny + Ng - 1);

                    // Off-diagonal: always 0 at walls (ghost not in tridiagonal solve)
                    double a = at_south_wall ? 0.0 : -aS_ptr[jm];
                    double c = at_north_wall ? 0.0 : -aN_ptr[jm];
                    // Diagonal: only modify for Neumann BC (absorption into diagonal)
                    double b = ((at_south_wall && neumann_y_lo) ? 0.0 : aS_ptr[jm])
                             + ((at_north_wall && neumann_y_hi) ? 0.0 : aN_ptr[jm])
                             + 2.0 * inv_dx2;
                    // RHS: -f + x_neighbors (consistent with -L(u) = -f form)
                    double rhs = inv_dx2 * (u_ptr[idx+1] + u_ptr[idx-1]) - f_ptr[idx];
                    // For Dirichlet BC: add ghost contribution to RHS (moved from tridiagonal)
                    if (at_south_wall && !neumann_y_lo) rhs += aS_ptr[jm] * u_ptr[idx - stride];
                    if (at_north_wall && !neumann_y_hi) rhs += aN_ptr[jm] * u_ptr[idx + stride];

                    int j_local = j - Ng;
                    if (j_local == 0) {
                        double b_safe = (std::abs(b) > PIVOT_EPS_CPU) ? b : PIVOT_EPS_CPU;
                        c_prime[0] = c / b_safe;
                        d_prime[0] = rhs / b_safe;
                    } else {
                        double denom = b - a * c_prime[j_local - 1];
                        double denom_safe = (std::abs(denom) > PIVOT_EPS_CPU) ? denom :
                                            (denom >= 0 ? PIVOT_EPS_CPU : -PIVOT_EPS_CPU);
                        c_prime[j_local] = c / denom_safe;
                        d_prime[j_local] = (rhs - a * d_prime[j_local - 1]) / denom_safe;
                    }
                }

                for (int j = Ny + Ng - 1; j >= Ng; --j) {
                    int j_local = j - Ng;
                    int idx = j * stride + i;
                    if (j_local == Ny - 1) {
                        u_ptr[idx] = d_prime[j_local];
                    } else {
                        u_ptr[idx] = d_prime[j_local] - c_prime[j_local] * u_ptr[idx + stride];
                    }
                }
            }
        } else {
            for (int k = Ng; k < Nz + Ng; ++k) {
                for (int i = Ng; i < Nx + Ng; ++i) {
                    for (int j = Ng; j < Ny + Ng; ++j) {
                        int jm = j + y_metric_offset;
                        int idx = k * plane_stride + j * stride + i;

                        const bool at_south_wall = (j == Ng);
                        const bool at_north_wall = (j == Ny + Ng - 1);

                        // Off-diagonal: always 0 at walls (ghost not in tridiagonal solve)
                        double a = at_south_wall ? 0.0 : -aS_ptr[jm];
                        double c = at_north_wall ? 0.0 : -aN_ptr[jm];
                        // Diagonal: only modify for Neumann BC (absorption into diagonal)
                        double b = ((at_south_wall && neumann_y_lo) ? 0.0 : aS_ptr[jm])
                                 + ((at_north_wall && neumann_y_hi) ? 0.0 : aN_ptr[jm])
                                 + 2.0 * inv_dx2 + 2.0 * inv_dz2;
                        // RHS: -f + x_neighbors + z_neighbors (consistent with -L(u) = -f form)
                        double rhs = inv_dx2 * (u_ptr[idx+1] + u_ptr[idx-1])
                                   + inv_dz2 * (u_ptr[idx+plane_stride] + u_ptr[idx-plane_stride])
                                   - f_ptr[idx];
                        // For Dirichlet BC: add ghost contribution to RHS (moved from tridiagonal)
                        if (at_south_wall && !neumann_y_lo) rhs += aS_ptr[jm] * u_ptr[idx - stride];
                        if (at_north_wall && !neumann_y_hi) rhs += aN_ptr[jm] * u_ptr[idx + stride];

                        int j_local = j - Ng;
                        if (j_local == 0) {
                            double b_safe = (std::abs(b) > PIVOT_EPS_CPU) ? b : PIVOT_EPS_CPU;
                            c_prime[0] = c / b_safe;
                            d_prime[0] = rhs / b_safe;
                        } else {
                            double denom = b - a * c_prime[j_local - 1];
                            double denom_safe = (std::abs(denom) > PIVOT_EPS_CPU) ? denom :
                                                (denom >= 0 ? PIVOT_EPS_CPU : -PIVOT_EPS_CPU);
                            c_prime[j_local] = c / denom_safe;
                            d_prime[j_local] = (rhs - a * d_prime[j_local - 1]) / denom_safe;
                        }
                    }

                    for (int j = Ny + Ng - 1; j >= Ng; --j) {
                        int j_local = j - Ng;
                        int idx = k * plane_stride + j * stride + i;
                        if (j_local == Ny - 1) {
                            u_ptr[idx] = d_prime[j_local];
                        } else {
                            u_ptr[idx] = d_prime[j_local] - c_prime[j_local] * u_ptr[idx + stride];
                        }
                    }
                }
            }
        }

        apply_bc(level);
    }
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

    // Check for non-uniform y (D·G = L consistency)
    // With semi-coarsening, ALL levels use the same y-metric coefficients
    const bool use_nonuniform_y = y_stretched_ && (semi_coarsening_ || level == 0);

    // For semi-coarsening: coarse levels have Ng=1 but y-metrics are indexed for fine mesh (Ng=Ng_fine)
    // Map j -> j_metric: j_metric = j - Ng + Ng_fine
    const int Ng_fine = levels_[0]->Ng;
    const int y_metric_offset = use_nonuniform_y ? (Ng_fine - Ng) : 0;

#ifdef USE_GPU_OFFLOAD
    // NVHPC WORKAROUND: Use omp_get_mapped_ptr for actual device addresses
    int device = omp_get_default_device();
    const double* u_ptr = static_cast<const double*>(omp_get_mapped_ptr(u_ptrs_[level], device));
    const double* f_ptr = static_cast<const double*>(omp_get_mapped_ptr(f_ptrs_[level], device));
    double* r_ptr = static_cast<double*>(omp_get_mapped_ptr(r_ptrs_[level], device));
    const double* aS_ptr = use_nonuniform_y ? static_cast<const double*>(omp_get_mapped_ptr(const_cast<double*>(yLap_aS_), device)) : nullptr;
    const double* aN_ptr = use_nonuniform_y ? static_cast<const double*>(omp_get_mapped_ptr(const_cast<double*>(yLap_aN_), device)) : nullptr;
    const double* aP_ptr = use_nonuniform_y ? static_cast<const double*>(omp_get_mapped_ptr(const_cast<double*>(yLap_aP_), device)) : nullptr;
#else
    const double* u_ptr = u_ptrs_[level];
    const double* f_ptr = f_ptrs_[level];
    double* r_ptr = r_ptrs_[level];
    const double* aS_ptr = yLap_aS_;
    const double* aN_ptr = yLap_aN_;
    const double* aP_ptr = yLap_aP_;
#endif

    if (is_2d) {
        if (use_nonuniform_y) {
            // 2D with non-uniform y-spacing
#ifdef USE_GPU_OFFLOAD
            #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(u_ptr, f_ptr, r_ptr, aS_ptr, aN_ptr, aP_ptr)
#endif
            for (int j = Ng; j < Ny + Ng; ++j) {
                for (int i = Ng; i < Nx + Ng; ++i) {
                    int idx = j * stride + i;
                    int jm = j + y_metric_offset;  // Map to fine mesh y-metric index
                    // Non-uniform y Laplacian: aS*u[j-1] + aP*u[j] + aN*u[j+1]
                    double lap_x = (u_ptr[idx+1] - 2.0*u_ptr[idx] + u_ptr[idx-1]) / dx2;
                    double lap_y = aS_ptr[jm] * u_ptr[idx-stride] + aP_ptr[jm] * u_ptr[idx] + aN_ptr[jm] * u_ptr[idx+stride];
                    r_ptr[idx] = f_ptr[idx] - (lap_x + lap_y);
                }
            }
        } else {
            // 2D with uniform spacing
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
        }
    } else {
        // 3D path
        if (use_nonuniform_y) {
            // 3D with non-uniform y-spacing
#ifdef USE_GPU_OFFLOAD
            #pragma omp target teams distribute parallel for collapse(3) is_device_ptr(u_ptr, f_ptr, r_ptr, aS_ptr, aN_ptr, aP_ptr)
#endif
            for (int k = Ng; k < Nz + Ng; ++k) {
                for (int j = Ng; j < Ny + Ng; ++j) {
                    for (int i = Ng; i < Nx + Ng; ++i) {
                        int idx = k * plane_stride + j * stride + i;
                        int jm = j + y_metric_offset;  // Map to fine mesh y-metric index
                        // Non-uniform y Laplacian: aS*u[j-1] + aP*u[j] + aN*u[j+1]
                        double lap_x = (u_ptr[idx+1] - 2.0*u_ptr[idx] + u_ptr[idx-1]) / dx2;
                        double lap_y = aS_ptr[jm] * u_ptr[idx-stride] + aP_ptr[jm] * u_ptr[idx] + aN_ptr[jm] * u_ptr[idx+stride];
                        double lap_z = (u_ptr[idx+plane_stride] - 2.0*u_ptr[idx] + u_ptr[idx-plane_stride]) / dz2;
                        r_ptr[idx] = f_ptr[idx] - (lap_x + lap_y + lap_z);
                    }
                }
            }
        } else {
            // 3D with uniform spacing
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

    // Check for non-uniform y (D·G = L consistency)
    // With semi-coarsening, ALL levels use the same y-metric coefficients
    const bool use_nonuniform_y = y_stretched_ && (semi_coarsening_ || level == 0);

    const int Ng = grid.Ng;  // Use level's ghost width
    const int Nx = grid.Nx;
    const int Ny = grid.Ny;
    const int Nz = grid.Nz;
    const int stride = grid.stride;
    const int plane_stride = grid.plane_stride;

    // For semi-coarsening: coarse levels have Ng=1 but y-metrics are indexed for fine mesh (Ng=Ng_fine)
    const int Ng_fine = levels_[0]->Ng;
    const int y_metric_offset = use_nonuniform_y ? (Ng_fine - Ng) : 0;

    // NVHPC WORKAROUND: Use omp_get_mapped_ptr for actual device addresses.
#ifdef USE_GPU_OFFLOAD
    int device = omp_get_default_device();
    const double* u_ptr = static_cast<const double*>(omp_get_mapped_ptr(u_ptrs_[level], device));
    const double* f_ptr = static_cast<const double*>(omp_get_mapped_ptr(f_ptrs_[level], device));
    double* r_ptr = static_cast<double*>(omp_get_mapped_ptr(r_ptrs_[level], device));
    const double* aS_ptr = use_nonuniform_y ? static_cast<const double*>(omp_get_mapped_ptr(const_cast<double*>(yLap_aS_), device)) : nullptr;
    const double* aN_ptr = use_nonuniform_y ? static_cast<const double*>(omp_get_mapped_ptr(const_cast<double*>(yLap_aN_), device)) : nullptr;
    const double* aP_ptr = use_nonuniform_y ? static_cast<const double*>(omp_get_mapped_ptr(const_cast<double*>(yLap_aP_), device)) : nullptr;
#else
    const double* u_ptr = u_ptrs_[level];
    const double* f_ptr = f_ptrs_[level];
    double* r_ptr = r_ptrs_[level];
    const double* aS_ptr = yLap_aS_;
    const double* aN_ptr = yLap_aN_;
    const double* aP_ptr = yLap_aP_;
#endif

    double max_res = 0.0;
    double sum_sq = 0.0;

    if (is_2d) {
        if (use_nonuniform_y) {
            // 2D with non-uniform y-spacing
#ifdef USE_GPU_OFFLOAD
            #pragma omp target teams distribute parallel for collapse(2) \
                is_device_ptr(u_ptr, f_ptr, r_ptr, aS_ptr, aN_ptr, aP_ptr) reduction(max: max_res) reduction(+: sum_sq)
#endif
            for (int j = Ng; j < Ny + Ng; ++j) {
                for (int i = Ng; i < Nx + Ng; ++i) {
                    int idx = j * stride + i;
                    int jm = j + y_metric_offset;  // Map to fine mesh y-metric index
                    double lap_x = (u_ptr[idx+1] - 2.0*u_ptr[idx] + u_ptr[idx-1]) / dx2;
                    double lap_y = aS_ptr[jm] * u_ptr[idx-stride] + aP_ptr[jm] * u_ptr[idx] + aN_ptr[jm] * u_ptr[idx+stride];
                    double r = f_ptr[idx] - (lap_x + lap_y);
                    r_ptr[idx] = r;
                    double abs_r = (r >= 0.0) ? r : -r;
                    if (abs_r > max_res) max_res = abs_r;
                    sum_sq += r * r;
                }
            }
        } else {
            // 2D with uniform spacing
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
                    double abs_r = (r >= 0.0) ? r : -r;
                    if (abs_r > max_res) max_res = abs_r;
                    sum_sq += r * r;
                }
            }
        }
    } else {
        // 3D path
        if (use_nonuniform_y) {
            // 3D with non-uniform y-spacing
#ifdef USE_GPU_OFFLOAD
            #pragma omp target teams distribute parallel for collapse(3) \
                is_device_ptr(u_ptr, f_ptr, r_ptr, aS_ptr, aN_ptr, aP_ptr) reduction(max: max_res) reduction(+: sum_sq)
#endif
            for (int k = Ng; k < Nz + Ng; ++k) {
                for (int j = Ng; j < Ny + Ng; ++j) {
                    for (int i = Ng; i < Nx + Ng; ++i) {
                        int idx = k * plane_stride + j * stride + i;
                        int jm = j + y_metric_offset;  // Map to fine mesh y-metric index
                        double lap_x = (u_ptr[idx+1] - 2.0*u_ptr[idx] + u_ptr[idx-1]) / dx2;
                        double lap_y = aS_ptr[jm] * u_ptr[idx-stride] + aP_ptr[jm] * u_ptr[idx] + aN_ptr[jm] * u_ptr[idx+stride];
                        double lap_z = (u_ptr[idx+plane_stride] - 2.0*u_ptr[idx] + u_ptr[idx-plane_stride]) / dz2;
                        double r = f_ptr[idx] - (lap_x + lap_y + lap_z);
                        r_ptr[idx] = r;
                        double abs_r = (r >= 0.0) ? r : -r;
                        if (abs_r > max_res) max_res = abs_r;
                        sum_sq += r * r;
                    }
                }
            }
        } else {
            // 3D with uniform spacing
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
                        double abs_r = (r >= 0.0) ? r : -r;
                        if (abs_r > max_res) max_res = abs_r;
                        sum_sq += r * r;
                    }
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

void MultigridPoissonSolver::restrict_residual_xz(int fine_level) {
    NVTX_SCOPE_MG("mg:restrict_xz");

    // Semi-coarsening restriction: coarsen only in x and z, keep y unchanged
    // For each (i_c, j, k_c): average over x-z neighbors at the same y
    // This maintains y-metric consistency across all MG levels
    auto& fine = *levels_[fine_level];
    auto& coarse = *levels_[fine_level + 1];
    const bool is_2d = fine.is2D();

    const int Ng_f = fine.Ng;
    const int Ng_c = coarse.Ng;
    const int Nx_c = coarse.Nx;
    const int Ny_c = coarse.Ny;  // Same as Ny_f (no y coarsening)
    const int Nz_c = coarse.Nz;
    const int stride_f = fine.stride;
    const int stride_c = coarse.stride;
    const int plane_stride_f = fine.plane_stride;
    const int plane_stride_c = coarse.plane_stride;

#ifdef USE_GPU_OFFLOAD
    int device = omp_get_default_device();
    const double* r_fine = static_cast<const double*>(omp_get_mapped_ptr(r_ptrs_[fine_level], device));
    double* f_coarse = static_cast<double*>(omp_get_mapped_ptr(f_ptrs_[fine_level + 1], device));
#else
    const double* r_fine = r_ptrs_[fine_level];
    double* f_coarse = f_ptrs_[fine_level + 1];
#endif

    if (is_2d) {
        // 2D semi-coarsening: coarsen x only, keep y unchanged
        // Use 3-point averaging in x: 0.25 * r[i-1] + 0.5 * r[i] + 0.25 * r[i+1]
#ifdef USE_GPU_OFFLOAD
        #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(r_fine, f_coarse)
#endif
        for (int j = Ng_c; j < Ny_c + Ng_c; ++j) {
            for (int i_c = Ng_c; i_c < Nx_c + Ng_c; ++i_c) {
                // y is unchanged: j_f = j (same index, different Ng offset)
                int j_f = j - Ng_c + Ng_f;
                // x coarsens: i_f = 2 * (i_c - Ng_c) + Ng_f
                int i_f = Ng_f + 2 * (i_c - Ng_c);

                int idx_f = j_f * stride_f + i_f;
                int idx_c = j * stride_c + i_c;

                // 3-point x-direction averaging (full weighting in x only)
                f_coarse[idx_c] = 0.25 * r_fine[idx_f - 1]
                                + 0.50 * r_fine[idx_f]
                                + 0.25 * r_fine[idx_f + 1];
            }
        }
    } else {
        // 3D semi-coarsening: coarsen x and z only, keep y unchanged
        // Use 9-point averaging in x-z plane
#ifdef USE_GPU_OFFLOAD
        #pragma omp target teams distribute parallel for collapse(3) is_device_ptr(r_fine, f_coarse)
#endif
        for (int k_c = Ng_c; k_c < Nz_c + Ng_c; ++k_c) {
            for (int j = Ng_c; j < Ny_c + Ng_c; ++j) {
                for (int i_c = Ng_c; i_c < Nx_c + Ng_c; ++i_c) {
                    // y is unchanged: j_f = j (same index, different Ng offset)
                    int j_f = j - Ng_c + Ng_f;
                    // x and z coarsen
                    int i_f = Ng_f + 2 * (i_c - Ng_c);
                    int k_f = Ng_f + 2 * (k_c - Ng_c);

                    int idx_f = k_f * plane_stride_f + j_f * stride_f + i_f;
                    int idx_c = k_c * plane_stride_c + j * stride_c + i_c;

                    // 9-point x-z plane averaging (no y neighbors)
                    double sum = 0.0;
                    // Center (weight = 1/4)
                    sum += 0.25 * r_fine[idx_f];
                    // 4 face neighbors in x-z (weight = 1/8 each)
                    sum += 0.125 * (r_fine[idx_f - 1] + r_fine[idx_f + 1]
                                  + r_fine[idx_f - plane_stride_f] + r_fine[idx_f + plane_stride_f]);
                    // 4 corner neighbors in x-z (weight = 1/16 each)
                    sum += 0.0625 * (r_fine[idx_f - 1 - plane_stride_f] + r_fine[idx_f + 1 - plane_stride_f]
                                   + r_fine[idx_f - 1 + plane_stride_f] + r_fine[idx_f + 1 + plane_stride_f]);

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

void MultigridPoissonSolver::prolongate_correction_xz(int coarse_level) {
    NVTX_SCOPE_MG("mg:prolongate_xz");

    // Semi-coarsening prolongation: interpolate only in x and z, keep y unchanged
    // For each fine cell: bilinear interpolation in x-z from coarse cells at same y
    //
    // COARSE CORRECTION DAMPING: For anisotropic problems, the coarse correction
    // can overshoot. With y-line relaxation on coarse levels (which properly handles
    // the y-direction anisotropy), full correction (α=1.0) is stable.
    const double alpha = 1.0;  // No damping - y-line relaxation handles anisotropy

    auto& coarse = *levels_[coarse_level];
    auto& fine = *levels_[coarse_level - 1];
    const bool is_2d = fine.is2D();

    const int Ng_f = fine.Ng;
    const int Ng_c = coarse.Ng;
    const int Nx_f = fine.Nx;
    const int Ny_f = fine.Ny;
    const int Nz_f = fine.Nz;
    const int stride_f = fine.stride;
    const int stride_c = coarse.stride;
    const int plane_stride_f = fine.plane_stride;
    const int plane_stride_c = coarse.plane_stride;

#ifdef USE_GPU_OFFLOAD
    int device = omp_get_default_device();
    const double* u_coarse = static_cast<const double*>(omp_get_mapped_ptr(u_ptrs_[coarse_level], device));
    double* u_fine = static_cast<double*>(omp_get_mapped_ptr(u_ptrs_[coarse_level - 1], device));
#else
    const double* u_coarse = u_ptrs_[coarse_level];
    double* u_fine = u_ptrs_[coarse_level - 1];
#endif

    if (is_2d) {
        // 2D semi-coarsening: interpolate x only, y is identity (injection)
#ifdef USE_GPU_OFFLOAD
        #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(u_coarse, u_fine)
#endif
        for (int j_f = Ng_f; j_f < Ny_f + Ng_f; ++j_f) {
            for (int i_f = Ng_f; i_f < Nx_f + Ng_f; ++i_f) {
                // y: direct mapping (no coarsening)
                int j_c = j_f - Ng_f + Ng_c;
                // x: coarsened
                int i_c = (i_f - Ng_f) / 2 + Ng_c;
                int di = (i_f - Ng_f) & 1;

                // Linear interpolation in x only
                double wx1 = 0.5 * di;
                double wx0 = 1.0 - wx1;

                int idx_c = j_c * stride_c + i_c;
                double correction = wx0 * u_coarse[idx_c] + wx1 * u_coarse[idx_c + 1];

                int idx_f = j_f * stride_f + i_f;
                u_fine[idx_f] += alpha * correction;  // Damped correction
            }
        }
    } else {
        // 3D semi-coarsening: bilinear interpolation in x-z, y is identity
#ifdef USE_GPU_OFFLOAD
        #pragma omp target teams distribute parallel for collapse(3) is_device_ptr(u_coarse, u_fine)
#endif
        for (int k_f = Ng_f; k_f < Nz_f + Ng_f; ++k_f) {
            for (int j_f = Ng_f; j_f < Ny_f + Ng_f; ++j_f) {
                for (int i_f = Ng_f; i_f < Nx_f + Ng_f; ++i_f) {
                    // y: direct mapping (no coarsening)
                    int j_c = j_f - Ng_f + Ng_c;
                    // x and z: coarsened
                    int i_c = (i_f - Ng_f) / 2 + Ng_c;
                    int k_c = (k_f - Ng_f) / 2 + Ng_c;
                    int di = (i_f - Ng_f) & 1;
                    int dk = (k_f - Ng_f) & 1;

                    // Bilinear interpolation weights in x-z
                    double wx1 = 0.5 * di;
                    double wx0 = 1.0 - wx1;
                    double wz1 = 0.5 * dk;
                    double wz0 = 1.0 - wz1;

                    int idx_c = k_c * plane_stride_c + j_c * stride_c + i_c;

                    // Bilinear interpolation from 4 coarse neighbors in x-z plane
                    double correction =
                        wx0 * wz0 * u_coarse[idx_c]
                      + wx1 * wz0 * u_coarse[idx_c + 1]
                      + wx0 * wz1 * u_coarse[idx_c + plane_stride_c]
                      + wx1 * wz1 * u_coarse[idx_c + 1 + plane_stride_c];

                    int idx_f = k_f * plane_stride_f + j_f * stride_f + i_f;
                    u_fine[idx_f] += alpha * correction;  // Damped correction
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

double MultigridPoissonSolver::dot_product_gpu(const double* a, const double* b, size_t n) {
    // GPU-resident dot product with parallel reduction
    double result = 0.0;

#ifdef USE_GPU_OFFLOAD
    #pragma omp target teams distribute parallel for reduction(+:result) is_device_ptr(a, b)
#endif
    for (size_t i = 0; i < n; ++i) {
        result += a[i] * b[i];
    }

    return result;
}

void MultigridPoissonSolver::apply_laplacian_coarse(const double* x, double* y) {
    // Apply Laplacian operator on coarsest level: y = A * x
    // Same stencil as compute_residual but without the f - ... subtraction
    NVTX_SCOPE_MG("mg:apply_laplacian_coarse");

    int coarsest = static_cast<int>(levels_.size()) - 1;
    auto& grid = *levels_[coarsest];
    const double dx2 = grid.dx * grid.dx;
    const double dz2 = grid.dz * grid.dz;
    const bool is_2d = grid.is2D();

    const int Ng = grid.Ng;
    const int Nx = grid.Nx;
    const int Ny = grid.Ny;
    const int Nz = grid.Nz;
    const int stride = grid.stride;
    const int plane_stride = grid.plane_stride;

    // For semi-coarsening: ALL levels use same y-metric coefficients from finest level
    const bool use_nonuniform_y = y_stretched_ && semi_coarsening_;
    const int Ng_fine = levels_[0]->Ng;
    const int y_metric_offset = use_nonuniform_y ? (Ng_fine - Ng) : 0;

#ifdef USE_GPU_OFFLOAD
    int device = omp_get_default_device();
    const double* aS_ptr = use_nonuniform_y ?
        static_cast<const double*>(omp_get_mapped_ptr(const_cast<double*>(yLap_aS_), device)) : nullptr;
    const double* aN_ptr = use_nonuniform_y ?
        static_cast<const double*>(omp_get_mapped_ptr(const_cast<double*>(yLap_aN_), device)) : nullptr;
    const double* aP_ptr = use_nonuniform_y ?
        static_cast<const double*>(omp_get_mapped_ptr(const_cast<double*>(yLap_aP_), device)) : nullptr;
#else
    const double* aS_ptr = yLap_aS_;
    const double* aN_ptr = yLap_aN_;
    const double* aP_ptr = yLap_aP_;
#endif

    if (is_2d) {
        const double dy2 = grid.dy * grid.dy;
        if (use_nonuniform_y) {
#ifdef USE_GPU_OFFLOAD
            #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(x, y, aS_ptr, aN_ptr, aP_ptr)
#endif
            for (int j = Ng; j < Ny + Ng; ++j) {
                for (int i = Ng; i < Nx + Ng; ++i) {
                    int idx = j * stride + i;
                    int jm = j + y_metric_offset;
                    double lap_x = (x[idx+1] - 2.0*x[idx] + x[idx-1]) / dx2;
                    double lap_y = aS_ptr[jm] * x[idx-stride] + aP_ptr[jm] * x[idx] + aN_ptr[jm] * x[idx+stride];
                    y[idx] = lap_x + lap_y;
                }
            }
        } else {
#ifdef USE_GPU_OFFLOAD
            #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(x, y)
#endif
            for (int j = Ng; j < Ny + Ng; ++j) {
                for (int i = Ng; i < Nx + Ng; ++i) {
                    int idx = j * stride + i;
                    double lap = (x[idx+1] - 2.0*x[idx] + x[idx-1]) / dx2
                               + (x[idx+stride] - 2.0*x[idx] + x[idx-stride]) / dy2;
                    y[idx] = lap;
                }
            }
        }
    } else {
        // 3D path
        if (use_nonuniform_y) {
#ifdef USE_GPU_OFFLOAD
            #pragma omp target teams distribute parallel for collapse(3) is_device_ptr(x, y, aS_ptr, aN_ptr, aP_ptr)
#endif
            for (int k = Ng; k < Nz + Ng; ++k) {
                for (int j = Ng; j < Ny + Ng; ++j) {
                    for (int i = Ng; i < Nx + Ng; ++i) {
                        int idx = k * plane_stride + j * stride + i;
                        int jm = j + y_metric_offset;
                        double lap_x = (x[idx+1] - 2.0*x[idx] + x[idx-1]) / dx2;
                        double lap_y = aS_ptr[jm] * x[idx-stride] + aP_ptr[jm] * x[idx] + aN_ptr[jm] * x[idx+stride];
                        double lap_z = (x[idx+plane_stride] - 2.0*x[idx] + x[idx-plane_stride]) / dz2;
                        y[idx] = lap_x + lap_y + lap_z;
                    }
                }
            }
        } else {
            const double dy2 = grid.dy * grid.dy;
#ifdef USE_GPU_OFFLOAD
            #pragma omp target teams distribute parallel for collapse(3) is_device_ptr(x, y)
#endif
            for (int k = Ng; k < Nz + Ng; ++k) {
                for (int j = Ng; j < Ny + Ng; ++j) {
                    for (int i = Ng; i < Nx + Ng; ++i) {
                        int idx = k * plane_stride + j * stride + i;
                        double lap = (x[idx+1] - 2.0*x[idx] + x[idx-1]) / dx2
                                   + (x[idx+stride] - 2.0*x[idx] + x[idx-stride]) / dy2
                                   + (x[idx+plane_stride] - 2.0*x[idx] + x[idx-plane_stride]) / dz2;
                        y[idx] = lap;
                    }
                }
            }
        }
    }
}

void MultigridPoissonSolver::solve_coarse_pcg(int max_iters, double tol) {
    // Preconditioned Conjugate Gradient for coarsest level
    // Preconditioner: 2 sweeps of y-line relaxation (Thomas algorithm)
    //
    // This properly reduces x/z low-frequency modes that y-line smoothing alone misses.
    // The y-line preconditioner captures the dominant y-anisotropy, while CG handles
    // the isotropic x/z components that the semi-coarsening hierarchy struggles with.
    //
    // For Neumann BCs (nullspace present), we project out the nullspace (mean) from
    // the search direction to prevent pAp from going to zero.
    NVTX_SCOPE_POISSON("mg:solve_coarse_pcg");

    int coarsest = static_cast<int>(levels_.size()) - 1;
    auto& grid = *levels_[coarsest];
    const int Ng = grid.Ng;
    const int Nx = grid.Nx;
    const int Ny = grid.Ny;
    const int Nz = grid.Nz;
    const int stride = grid.stride;
    const int plane_stride = grid.plane_stride;
    const size_t total_size = grid.total_size;
    const bool is_2d = grid.is2D();
    const bool needs_nullspace_projection = has_nullspace();
    const int n_interior = Nx * Ny * (is_2d ? 1 : Nz);

    // Allocate PCG scratch buffers on first call
    if (pcg_buf_size_ < total_size) {
#ifdef USE_GPU_OFFLOAD
        if (pcg_p_) {
            int device = omp_get_default_device();
            omp_target_free(pcg_p_, device);
            omp_target_free(pcg_z_, device);
            omp_target_free(pcg_Ap_, device);
            omp_target_free(pcg_x_, device);
            omp_target_free(pcg_f_save_, device);
        }
        int device = omp_get_default_device();
        pcg_p_ = static_cast<double*>(omp_target_alloc(total_size * sizeof(double), device));
        pcg_z_ = static_cast<double*>(omp_target_alloc(total_size * sizeof(double), device));
        pcg_Ap_ = static_cast<double*>(omp_target_alloc(total_size * sizeof(double), device));
        pcg_x_ = static_cast<double*>(omp_target_alloc(total_size * sizeof(double), device));
        pcg_f_save_ = static_cast<double*>(omp_target_alloc(total_size * sizeof(double), device));
#else
        delete[] pcg_p_;
        delete[] pcg_z_;
        delete[] pcg_Ap_;
        delete[] pcg_x_;
        delete[] pcg_f_save_;
        pcg_p_ = new double[total_size];
        pcg_z_ = new double[total_size];
        pcg_Ap_ = new double[total_size];
        pcg_x_ = new double[total_size];
        pcg_f_save_ = new double[total_size];
#endif
        pcg_buf_size_ = total_size;
    }

#ifdef USE_GPU_OFFLOAD
    int device = omp_get_default_device();
    double* u_ptr = static_cast<double*>(omp_get_mapped_ptr(u_ptrs_[coarsest], device));
    double* f_ptr = static_cast<double*>(omp_get_mapped_ptr(f_ptrs_[coarsest], device));
    double* r_ptr = static_cast<double*>(omp_get_mapped_ptr(r_ptrs_[coarsest], device));
#else
    double* u_ptr = u_ptrs_[coarsest];
    double* f_ptr = f_ptrs_[coarsest];
    double* r_ptr = r_ptrs_[coarsest];
#endif
    double* p = pcg_p_;
    double* z = pcg_z_;
    double* Ap = pcg_Ap_;
    double* x = pcg_x_;        // Solution accumulator
    double* f_save = pcg_f_save_;  // Saved RHS

    // Save original RHS (f) - we'll need it for preconditioner
    // Initialize solution x = u (incoming guess)
#ifdef USE_GPU_OFFLOAD
    #pragma omp target teams distribute parallel for is_device_ptr(f_save, f_ptr, x, u_ptr)
    for (size_t i = 0; i < total_size; ++i) {
        f_save[i] = f_ptr[i];
        x[i] = u_ptr[i];
    }
#else
    for (size_t i = 0; i < total_size; ++i) {
        f_save[i] = f_ptr[i];
        x[i] = u_ptr[i];
    }
#endif

    // Compute initial residual r = f - A*x
    // Copy x to u for residual computation (compute_residual uses u)
#ifdef USE_GPU_OFFLOAD
    #pragma omp target teams distribute parallel for is_device_ptr(u_ptr, x)
    for (size_t i = 0; i < total_size; ++i) {
        u_ptr[i] = x[i];
    }
#else
    for (size_t i = 0; i < total_size; ++i) {
        u_ptr[i] = x[i];
    }
#endif

    apply_bc(coarsest);
    compute_residual(coarsest);

    // Compute ||r||^2 for initial norm (only over interior)
    double r_norm_sq_init = 0.0;
#ifdef USE_GPU_OFFLOAD
    if (is_2d) {
        #pragma omp target teams distribute parallel for collapse(2) reduction(+:r_norm_sq_init) is_device_ptr(r_ptr)
        for (int j = Ng; j < Ny + Ng; ++j) {
            for (int i = Ng; i < Nx + Ng; ++i) {
                int idx = j * stride + i;
                r_norm_sq_init += r_ptr[idx] * r_ptr[idx];
            }
        }
    } else {
        #pragma omp target teams distribute parallel for collapse(3) reduction(+:r_norm_sq_init) is_device_ptr(r_ptr)
        for (int k = Ng; k < Nz + Ng; ++k) {
            for (int j = Ng; j < Ny + Ng; ++j) {
                for (int i = Ng; i < Nx + Ng; ++i) {
                    int idx = k * plane_stride + j * stride + i;
                    r_norm_sq_init += r_ptr[idx] * r_ptr[idx];
                }
            }
        }
    }
#else
    for (int k = Ng; k < Nz + Ng; ++k) {
        for (int j = Ng; j < Ny + Ng; ++j) {
            for (int i = Ng; i < Nx + Ng; ++i) {
                int idx = k * plane_stride + j * stride + i;
                r_norm_sq_init += r_ptr[idx] * r_ptr[idx];
            }
        }
    }
#endif

    if (r_norm_sq_init < tol * tol) {
        // Already converged - copy solution back
#ifdef USE_GPU_OFFLOAD
        #pragma omp target teams distribute parallel for is_device_ptr(u_ptr, x)
        for (size_t i = 0; i < total_size; ++i) {
            u_ptr[i] = x[i];
        }
#else
        for (size_t i = 0; i < total_size; ++i) {
            u_ptr[i] = x[i];
        }
#endif
        return;
    }

    // Apply preconditioner: z = M^{-1} r
    // Preconditioner = 2 y-line sweeps solving M*z = r
    // Set f = r (RHS for preconditioner), u = 0 (initial guess for z)
#ifdef USE_GPU_OFFLOAD
    #pragma omp target teams distribute parallel for is_device_ptr(f_ptr, r_ptr, u_ptr)
    for (size_t i = 0; i < total_size; ++i) {
        f_ptr[i] = r_ptr[i];
        u_ptr[i] = 0.0;
    }
#else
    for (size_t i = 0; i < total_size; ++i) {
        f_ptr[i] = r_ptr[i];
        u_ptr[i] = 0.0;
    }
#endif

    smooth_y_lines(coarsest, 2);

    // For Neumann BCs: project out nullspace (mean) from preconditioned residual
    // This ensures p has no component in the nullspace, preventing pAp → 0
    if (needs_nullspace_projection) {
        double z_sum = 0.0;
#ifdef USE_GPU_OFFLOAD
        if (is_2d) {
            #pragma omp target teams distribute parallel for collapse(2) reduction(+:z_sum) is_device_ptr(u_ptr)
            for (int j = Ng; j < Ny + Ng; ++j) {
                for (int i = Ng; i < Nx + Ng; ++i) {
                    z_sum += u_ptr[j * stride + i];
                }
            }
        } else {
            #pragma omp target teams distribute parallel for collapse(3) reduction(+:z_sum) is_device_ptr(u_ptr)
            for (int k = Ng; k < Nz + Ng; ++k) {
                for (int j = Ng; j < Ny + Ng; ++j) {
                    for (int i = Ng; i < Nx + Ng; ++i) {
                        z_sum += u_ptr[k * plane_stride + j * stride + i];
                    }
                }
            }
        }
#else
        for (int k = Ng; k < Nz + Ng; ++k) {
            for (int j = Ng; j < Ny + Ng; ++j) {
                for (int i = Ng; i < Nx + Ng; ++i) {
                    z_sum += u_ptr[k * plane_stride + j * stride + i];
                }
            }
        }
#endif
        double z_mean = z_sum / n_interior;
#ifdef USE_GPU_OFFLOAD
        if (is_2d) {
            #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(u_ptr)
            for (int j = Ng; j < Ny + Ng; ++j) {
                for (int i = Ng; i < Nx + Ng; ++i) {
                    u_ptr[j * stride + i] -= z_mean;
                }
            }
        } else {
            #pragma omp target teams distribute parallel for collapse(3) is_device_ptr(u_ptr)
            for (int k = Ng; k < Nz + Ng; ++k) {
                for (int j = Ng; j < Ny + Ng; ++j) {
                    for (int i = Ng; i < Nx + Ng; ++i) {
                        u_ptr[k * plane_stride + j * stride + i] -= z_mean;
                    }
                }
            }
        }
#else
        for (int k = Ng; k < Nz + Ng; ++k) {
            for (int j = Ng; j < Ny + Ng; ++j) {
                for (int i = Ng; i < Nx + Ng; ++i) {
                    u_ptr[k * plane_stride + j * stride + i] -= z_mean;
                }
            }
        }
#endif
    }

    // z = u (result of preconditioner), p = z
#ifdef USE_GPU_OFFLOAD
    #pragma omp target teams distribute parallel for is_device_ptr(z, p, u_ptr)
    for (size_t i = 0; i < total_size; ++i) {
        z[i] = u_ptr[i];
        p[i] = u_ptr[i];
    }
#else
    for (size_t i = 0; i < total_size; ++i) {
        z[i] = u_ptr[i];
        p[i] = u_ptr[i];
    }
#endif

    // rho = r^T z (only interior)
    double rho = 0.0;
#ifdef USE_GPU_OFFLOAD
    if (is_2d) {
        #pragma omp target teams distribute parallel for collapse(2) reduction(+:rho) is_device_ptr(r_ptr, z)
        for (int j = Ng; j < Ny + Ng; ++j) {
            for (int i = Ng; i < Nx + Ng; ++i) {
                int idx = j * stride + i;
                rho += r_ptr[idx] * z[idx];
            }
        }
    } else {
        #pragma omp target teams distribute parallel for collapse(3) reduction(+:rho) is_device_ptr(r_ptr, z)
        for (int k = Ng; k < Nz + Ng; ++k) {
            for (int j = Ng; j < Ny + Ng; ++j) {
                for (int i = Ng; i < Nx + Ng; ++i) {
                    int idx = k * plane_stride + j * stride + i;
                    rho += r_ptr[idx] * z[idx];
                }
            }
        }
    }
#else
    for (int k = Ng; k < Nz + Ng; ++k) {
        for (int j = Ng; j < Ny + Ng; ++j) {
            for (int i = Ng; i < Nx + Ng; ++i) {
                int idx = k * plane_stride + j * stride + i;
                rho += r_ptr[idx] * z[idx];
            }
        }
    }
#endif

    // Main PCG loop
    for (int iter = 0; iter < max_iters; ++iter) {
        // Apply BC to p for Laplacian stencil (need ghost values)
        // Copy p to u for BC application
#ifdef USE_GPU_OFFLOAD
        #pragma omp target teams distribute parallel for is_device_ptr(u_ptr, p)
        for (size_t i = 0; i < total_size; ++i) {
            u_ptr[i] = p[i];
        }
#else
        for (size_t i = 0; i < total_size; ++i) {
            u_ptr[i] = p[i];
        }
#endif
        apply_bc(coarsest);

        // Ap = A * p (using u which has BCs applied)
        apply_laplacian_coarse(u_ptr, Ap);

        // alpha = rho / (p^T Ap) - use interior only
        double pAp = 0.0;
#ifdef USE_GPU_OFFLOAD
        if (is_2d) {
            #pragma omp target teams distribute parallel for collapse(2) reduction(+:pAp) is_device_ptr(u_ptr, Ap)
            for (int j = Ng; j < Ny + Ng; ++j) {
                for (int i = Ng; i < Nx + Ng; ++i) {
                    int idx = j * stride + i;
                    pAp += u_ptr[idx] * Ap[idx];  // u_ptr has p with BCs
                }
            }
        } else {
            #pragma omp target teams distribute parallel for collapse(3) reduction(+:pAp) is_device_ptr(u_ptr, Ap)
            for (int k = Ng; k < Nz + Ng; ++k) {
                for (int j = Ng; j < Ny + Ng; ++j) {
                    for (int i = Ng; i < Nx + Ng; ++i) {
                        int idx = k * plane_stride + j * stride + i;
                        pAp += u_ptr[idx] * Ap[idx];
                    }
                }
            }
        }
#else
        for (int k = Ng; k < Nz + Ng; ++k) {
            for (int j = Ng; j < Ny + Ng; ++j) {
                for (int i = Ng; i < Nx + Ng; ++i) {
                    int idx = k * plane_stride + j * stride + i;
                    pAp += u_ptr[idx] * Ap[idx];
                }
            }
        }
#endif

        if (std::abs(pAp) < 1e-30) {
            // PCG breakdown - restart with p = z (preconditioned residual)
            // Re-precondition current residual and reset search direction
#ifdef USE_GPU_OFFLOAD
            #pragma omp target teams distribute parallel for is_device_ptr(u_ptr, r_ptr)
            for (size_t ii = 0; ii < total_size; ++ii) u_ptr[ii] = r_ptr[ii];
#else
            for (size_t ii = 0; ii < total_size; ++ii) u_ptr[ii] = r_ptr[ii];
#endif
            apply_bc(coarsest);
            smooth_y_lines(coarsest, 2);
            // Project nullspace from preconditioned residual
            if (needs_nullspace_projection) {
                double z_sum2 = 0.0;
#ifdef USE_GPU_OFFLOAD
                #pragma omp target teams distribute parallel for collapse(3) reduction(+:z_sum2) is_device_ptr(u_ptr)
                for (int kk = Ng; kk < Nz + Ng; ++kk)
                    for (int jj = Ng; jj < Ny + Ng; ++jj)
                        for (int ii = Ng; ii < Nx + Ng; ++ii)
                            z_sum2 += u_ptr[kk * plane_stride + jj * stride + ii];
#else
                for (int kk = Ng; kk < Nz + Ng; ++kk)
                    for (int jj = Ng; jj < Ny + Ng; ++jj)
                        for (int ii = Ng; ii < Nx + Ng; ++ii)
                            z_sum2 += u_ptr[kk * plane_stride + jj * stride + ii];
#endif
                double z_mean2 = z_sum2 / n_interior;
#ifdef USE_GPU_OFFLOAD
                #pragma omp target teams distribute parallel for collapse(3) is_device_ptr(u_ptr) firstprivate(z_mean2)
                for (int kk = Ng; kk < Nz + Ng; ++kk)
                    for (int jj = Ng; jj < Ny + Ng; ++jj)
                        for (int ii = Ng; ii < Nx + Ng; ++ii)
                            u_ptr[kk * plane_stride + jj * stride + ii] -= z_mean2;
#else
                for (int kk = Ng; kk < Nz + Ng; ++kk)
                    for (int jj = Ng; jj < Ny + Ng; ++jj)
                        for (int ii = Ng; ii < Nx + Ng; ++ii)
                            u_ptr[kk * plane_stride + jj * stride + ii] -= z_mean2;
#endif
            }
            // Reset z = u, p = z, rho = r^T z
#ifdef USE_GPU_OFFLOAD
            #pragma omp target teams distribute parallel for is_device_ptr(z, p, u_ptr)
            for (size_t ii = 0; ii < total_size; ++ii) { z[ii] = u_ptr[ii]; p[ii] = u_ptr[ii]; }
#else
            for (size_t ii = 0; ii < total_size; ++ii) { z[ii] = u_ptr[ii]; p[ii] = u_ptr[ii]; }
#endif
            rho = 0.0;
#ifdef USE_GPU_OFFLOAD
            #pragma omp target teams distribute parallel for collapse(3) reduction(+:rho) is_device_ptr(r_ptr, z)
            for (int kk = Ng; kk < Nz + Ng; ++kk)
                for (int jj = Ng; jj < Ny + Ng; ++jj)
                    for (int ii = Ng; ii < Nx + Ng; ++ii)
                        rho += r_ptr[kk * plane_stride + jj * stride + ii] * z[kk * plane_stride + jj * stride + ii];
#else
            for (int kk = Ng; kk < Nz + Ng; ++kk)
                for (int jj = Ng; jj < Ny + Ng; ++jj)
                    for (int ii = Ng; ii < Nx + Ng; ++ii)
                        rho += r_ptr[kk * plane_stride + jj * stride + ii] * z[kk * plane_stride + jj * stride + ii];
#endif
            continue;  // Restart iteration with fresh search direction
        }
        double alpha = rho / pAp;

        // Guard against alpha blowup (rho >> pAp)
        if (std::abs(alpha) > 1e10) {
            std::cerr << "[PCG ALPHA GUARD] alpha=" << alpha << " (rho=" << rho
                      << ", pAp=" << pAp << ") - clamping to 1e10\n";
            alpha = (alpha > 0) ? 1e10 : -1e10;
        }

        // Update: x += alpha * p, r -= alpha * Ap
        // Split from convergence check to avoid reduction sync every iteration
#ifdef USE_GPU_OFFLOAD
        if (is_2d) {
            #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(x, u_ptr, r_ptr, Ap)
            for (int j = Ng; j < Ny + Ng; ++j) {
                for (int i = Ng; i < Nx + Ng; ++i) {
                    int idx = j * stride + i;
                    x[idx] += alpha * u_ptr[idx];  // u_ptr has p with BCs
                    r_ptr[idx] -= alpha * Ap[idx];
                }
            }
        } else {
            #pragma omp target teams distribute parallel for collapse(3) is_device_ptr(x, u_ptr, r_ptr, Ap)
            for (int k = Ng; k < Nz + Ng; ++k) {
                for (int j = Ng; j < Ny + Ng; ++j) {
                    for (int i = Ng; i < Nx + Ng; ++i) {
                        int idx = k * plane_stride + j * stride + i;
                        x[idx] += alpha * u_ptr[idx];
                        r_ptr[idx] -= alpha * Ap[idx];
                    }
                }
            }
        }
#else
        for (int k = Ng; k < Nz + Ng; ++k) {
            for (int j = Ng; j < Ny + Ng; ++j) {
                for (int i = Ng; i < Nx + Ng; ++i) {
                    int idx = k * plane_stride + j * stride + i;
                    x[idx] += alpha * u_ptr[idx];
                    r_ptr[idx] -= alpha * Ap[idx];
                }
            }
        }
#endif

        // Check convergence every 4 iterations (reduces GPU→CPU sync overhead)
        // The reduction for r_norm_sq is the bottleneck — skip it most iterations
        if (iter % 4 == 3 || iter == max_iters - 1) {
            double r_norm_sq = 0.0;
#ifdef USE_GPU_OFFLOAD
            if (is_2d) {
                #pragma omp target teams distribute parallel for collapse(2) reduction(+:r_norm_sq) is_device_ptr(r_ptr)
                for (int j = Ng; j < Ny + Ng; ++j) {
                    for (int i = Ng; i < Nx + Ng; ++i) {
                        int idx = j * stride + i;
                        r_norm_sq += r_ptr[idx] * r_ptr[idx];
                    }
                }
            } else {
                #pragma omp target teams distribute parallel for collapse(3) reduction(+:r_norm_sq) is_device_ptr(r_ptr)
                for (int k = Ng; k < Nz + Ng; ++k) {
                    for (int j = Ng; j < Ny + Ng; ++j) {
                        for (int i = Ng; i < Nx + Ng; ++i) {
                            int idx = k * plane_stride + j * stride + i;
                            r_norm_sq += r_ptr[idx] * r_ptr[idx];
                        }
                    }
                }
            }
#else
            for (int k = Ng; k < Nz + Ng; ++k) {
                for (int j = Ng; j < Ny + Ng; ++j) {
                    for (int i = Ng; i < Nx + Ng; ++i) {
                        int idx = k * plane_stride + j * stride + i;
                        r_norm_sq += r_ptr[idx] * r_ptr[idx];
                    }
                }
            }
#endif
            if (r_norm_sq < tol * tol * r_norm_sq_init) {
                break;
            }
        }

        // Apply preconditioner: z = M^{-1} r
        // Set f = r (RHS), u = 0 (initial guess)
#ifdef USE_GPU_OFFLOAD
        #pragma omp target teams distribute parallel for is_device_ptr(f_ptr, r_ptr, u_ptr)
        for (size_t i = 0; i < total_size; ++i) {
            f_ptr[i] = r_ptr[i];
            u_ptr[i] = 0.0;
        }
#else
        for (size_t i = 0; i < total_size; ++i) {
            f_ptr[i] = r_ptr[i];
            u_ptr[i] = 0.0;
        }
#endif

        smooth_y_lines(coarsest, 2);

        // For Neumann BCs: project out nullspace (mean) from preconditioned residual
        if (needs_nullspace_projection) {
            double z_sum = 0.0;
#ifdef USE_GPU_OFFLOAD
            if (is_2d) {
                #pragma omp target teams distribute parallel for collapse(2) reduction(+:z_sum) is_device_ptr(u_ptr)
                for (int j = Ng; j < Ny + Ng; ++j) {
                    for (int i = Ng; i < Nx + Ng; ++i) {
                        z_sum += u_ptr[j * stride + i];
                    }
                }
            } else {
                #pragma omp target teams distribute parallel for collapse(3) reduction(+:z_sum) is_device_ptr(u_ptr)
                for (int k = Ng; k < Nz + Ng; ++k) {
                    for (int j = Ng; j < Ny + Ng; ++j) {
                        for (int i = Ng; i < Nx + Ng; ++i) {
                            z_sum += u_ptr[k * plane_stride + j * stride + i];
                        }
                    }
                }
            }
#else
            for (int k = Ng; k < Nz + Ng; ++k) {
                for (int j = Ng; j < Ny + Ng; ++j) {
                    for (int i = Ng; i < Nx + Ng; ++i) {
                        z_sum += u_ptr[k * plane_stride + j * stride + i];
                    }
                }
            }
#endif
            double z_mean = z_sum / n_interior;
#ifdef USE_GPU_OFFLOAD
            if (is_2d) {
                #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(u_ptr)
                for (int j = Ng; j < Ny + Ng; ++j) {
                    for (int i = Ng; i < Nx + Ng; ++i) {
                        u_ptr[j * stride + i] -= z_mean;
                    }
                }
            } else {
                #pragma omp target teams distribute parallel for collapse(3) is_device_ptr(u_ptr)
                for (int k = Ng; k < Nz + Ng; ++k) {
                    for (int j = Ng; j < Ny + Ng; ++j) {
                        for (int i = Ng; i < Nx + Ng; ++i) {
                            u_ptr[k * plane_stride + j * stride + i] -= z_mean;
                        }
                    }
                }
            }
#else
            for (int k = Ng; k < Nz + Ng; ++k) {
                for (int j = Ng; j < Ny + Ng; ++j) {
                    for (int i = Ng; i < Nx + Ng; ++i) {
                        u_ptr[k * plane_stride + j * stride + i] -= z_mean;
                    }
                }
            }
#endif
        }

        // z = u (result of preconditioner)
#ifdef USE_GPU_OFFLOAD
        #pragma omp target teams distribute parallel for is_device_ptr(z, u_ptr)
        for (size_t i = 0; i < total_size; ++i) {
            z[i] = u_ptr[i];
        }
#else
        for (size_t i = 0; i < total_size; ++i) {
            z[i] = u_ptr[i];
        }
#endif

        // rho_new = r^T z
        double rho_new = 0.0;
#ifdef USE_GPU_OFFLOAD
        if (is_2d) {
            #pragma omp target teams distribute parallel for collapse(2) reduction(+:rho_new) is_device_ptr(r_ptr, z)
            for (int j = Ng; j < Ny + Ng; ++j) {
                for (int i = Ng; i < Nx + Ng; ++i) {
                    int idx = j * stride + i;
                    rho_new += r_ptr[idx] * z[idx];
                }
            }
        } else {
            #pragma omp target teams distribute parallel for collapse(3) reduction(+:rho_new) is_device_ptr(r_ptr, z)
            for (int k = Ng; k < Nz + Ng; ++k) {
                for (int j = Ng; j < Ny + Ng; ++j) {
                    for (int i = Ng; i < Nx + Ng; ++i) {
                        int idx = k * plane_stride + j * stride + i;
                        rho_new += r_ptr[idx] * z[idx];
                    }
                }
            }
        }
#else
        for (int k = Ng; k < Nz + Ng; ++k) {
            for (int j = Ng; j < Ny + Ng; ++j) {
                for (int i = Ng; i < Nx + Ng; ++i) {
                    int idx = k * plane_stride + j * stride + i;
                    rho_new += r_ptr[idx] * z[idx];
                }
            }
        }
#endif

        // beta = rho_new / rho
        double beta = (std::abs(rho) > 1e-30) ? (rho_new / rho) : 0.0;
        rho = rho_new;

        // p = z + beta * p
#ifdef USE_GPU_OFFLOAD
        #pragma omp target teams distribute parallel for is_device_ptr(p, z)
        for (size_t i = 0; i < total_size; ++i) {
            p[i] = z[i] + beta * p[i];
        }
#else
        for (size_t i = 0; i < total_size; ++i) {
            p[i] = z[i] + beta * p[i];
        }
#endif
    }

    // Copy solution back to u, restore original f
#ifdef USE_GPU_OFFLOAD
    #pragma omp target teams distribute parallel for is_device_ptr(u_ptr, x, f_ptr, f_save)
    for (size_t i = 0; i < total_size; ++i) {
        u_ptr[i] = x[i];
        f_ptr[i] = f_save[i];
    }
#else
    for (size_t i = 0; i < total_size; ++i) {
        u_ptr[i] = x[i];
        f_ptr[i] = f_save[i];
    }
#endif
    // PCG iteration tracking disabled - not needed in production
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
    // For 3D semi-coarsening: use y-line relaxation on ALL levels (including finest)
    // This is critical for stretched y-grids where point-smoothers are ineffective
    auto do_smooth = [this, level, degree](int nu) {
        // For 3D semi-coarsening: use y-line relaxation on ALL levels
        // The y-direction has highly anisotropic coefficients (stretched mesh),
        // so point-smoothers like Chebyshev converge very slowly.
        // Y-line relaxation solves exact tridiagonal systems, capturing the stiff y-modes.
        const bool is_3d = levels_[level]->Nz > 1;
        const bool use_yline_for_anisotropy = semi_coarsening_ && is_3d;
        const bool needs_strong_smooth = has_nullspace();  // Neumann/Periodic needs more sweeps

        for (int pass = 0; pass < nu; ++pass) {
            if (use_yline_for_anisotropy) {
                // Y-line relaxation: solves y-direction exactly (Thomas algorithm)
                // Neumann case needs more sweeps to converge the slowest y-eigenmodes
                const int yline_sweeps = needs_strong_smooth ? 4 : 2;
                smooth_y_lines(level, yline_sweeps);
                // XZ-plane RBGS: attacks modes constant in y but oscillatory in x/z
                // These modes pass through y-line unchanged and cause MG stall without this
                // RBGS is more effective than Jacobi for these plane-like error modes
                // Use more sweeps on finest level where high-frequency content is strongest
                const int rbgs_sweeps = (level == 0 && needs_strong_smooth) ? 4 : 2;
                smooth_xz_plane_rbgs(level, rbgs_sweeps);
            } else if (semi_coarsening_) {
                // 2D semi-coarsening: use more Jacobi iterations with damping
                smooth_jacobi(level, degree * 4, 0.67);
            } else if (smoother_type_ == MGSmootherType::Chebyshev) {
                smooth_chebyshev(level, degree);
            } else {
                smooth_jacobi(level, degree, 0.8);
            }
        }
    };

    const int num_levels = static_cast<int>(levels_.size());
    if (level == num_levels - 1) {
        // Coarsest level - do a REAL solve, not just a few smooths
        // For anisotropic problems, weak coarse solves cause correction overshoot
        // For Neumann BCs (has_nullspace), we need even more iterations
        const bool is_3d = levels_[level]->Nz > 1;
        const bool needs_strong_coarse = has_nullspace();  // Neumann/Periodic needs stronger solve
        if (semi_coarsening_ && is_3d) {
            // 3D semi-coarsening: use PCG with y-line preconditioner
            // Y-line smoothing alone can't reduce x/z low-frequency modes effectively
            // PCG properly handles the isotropic x/z components
            const int pcg_iters = needs_strong_coarse ? 50 : 30;
            solve_coarse_pcg(pcg_iters, 1e-10);
        } else if (semi_coarsening_) {
            // 2D semi-coarsening: use many damped Jacobi iterations
            const int iters = needs_strong_coarse ? 400 : 200;
            smooth_jacobi(level, iters, 0.67);
        } else {
            const int coarse_iters = needs_strong_coarse ? 100 : 50;
            const double coarse_omega = 0.67;
            smooth_jacobi(level, coarse_iters, coarse_omega);
        }
        return;
    }

    // K-cycle-lite: Apply extra smoothing at next-to-coarsest level
    // This helps kill stubborn modes that don't transfer well to the coarsest grid
    const bool is_3d = levels_[level]->Nz > 1;
    if (level == num_levels - 2 && semi_coarsening_ && is_3d && has_nullspace()) {
        // At L2 (e.g., 16×64×16 for 64³ grid), do extra work
        // This is K-cycle-lite: stronger solve at near-coarsest level
        do_smooth(nu1);
        compute_residual(level);
        apply_bc_to_residual(level);
        restrict_residual_xz(level);

        // Zero and solve coarsest (L3)
        {
            auto& coarse = *levels_[level + 1];
            size_t size_c = level_sizes_[level + 1];
#ifdef USE_GPU_OFFLOAD
            int device = omp_get_default_device();
            double* u_coarse = static_cast<double*>(omp_get_mapped_ptr(u_ptrs_[level + 1], device));
            #pragma omp target teams distribute parallel for is_device_ptr(u_coarse)
            for (int idx = 0; idx < (int)size_c; ++idx) u_coarse[idx] = 0.0;
#else
            std::fill(u_ptrs_[level + 1], u_ptrs_[level + 1] + size_c, 0.0);
#endif
        }
        // Strong PCG solve at coarsest
        solve_coarse_pcg(50, 1e-10);

        // Prolongate from level+1 to level and post-smooth
        prolongate_correction_xz(level + 1);  // coarse_level = level + 1
        apply_bc(level);
        do_smooth(nu2);

        // Repeat the coarse solve (K-cycle: two coarse solves)
        compute_residual(level);
        apply_bc_to_residual(level);
        restrict_residual_xz(level);
        {
            size_t size_c = level_sizes_[level + 1];
#ifdef USE_GPU_OFFLOAD
            int device = omp_get_default_device();
            double* u_coarse = static_cast<double*>(omp_get_mapped_ptr(u_ptrs_[level + 1], device));
            #pragma omp target teams distribute parallel for is_device_ptr(u_coarse)
            for (int idx = 0; idx < (int)size_c; ++idx) u_coarse[idx] = 0.0;
#else
            std::fill(u_ptrs_[level + 1], u_ptrs_[level + 1] + size_c, 0.0);
#endif
        }
        solve_coarse_pcg(50, 1e-10);
        prolongate_correction_xz(level + 1);  // coarse_level = level + 1
        apply_bc(level);
        do_smooth(nu2);
        return;
    }

    // Pre-smoothing (nu1 passes of degree-k Chebyshev)
    do_smooth(nu1);

    // Compute residual
    compute_residual(level);

    // Apply BCs to residual ghost cells for proper 9-point restriction
    // This is essential for periodic BCs where ghost cells must wrap around
    apply_bc_to_residual(level);

    // Restrict to coarse grid (use semi-coarsening if y is stretched)
    if (semi_coarsening_) {
        restrict_residual_xz(level);
    } else {
        restrict_residual(level);
    }

    // For Neumann BCs, ensure coarse-level RHS is mean-free after restriction
    // Full-weighting restriction should preserve zero-mean, but numerical precision
    // can accumulate. This ensures compatibility condition at all levels.
    if (has_nullspace()) {
        make_rhs_mean_free(level + 1);
    }

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
    if (semi_coarsening_) {
        prolongate_correction_xz(level + 1);
    } else {
        prolongate_correction(level + 1);
    }
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

void MultigridPoissonSolver::make_rhs_mean_free(int level) {
    // For Neumann/Periodic BCs, the Poisson equation has a nullspace (constants).
    // For a solution to exist, the RHS must satisfy the compatibility condition:
    //   integral(rhs) = 0  (or equivalently, mean(rhs) = 0)
    // This function subtracts the mean from the RHS to ensure compatibility.
    // Call this BEFORE V-cycles, not after - it's the proper Neumann handling.
    auto& grid = *levels_[level];
    double sum = 0.0;
    const int Ng = grid.Ng;
    const int Nx = grid.Nx;
    const int Ny = grid.Ny;
    const int Nz = grid.Nz;
    const int stride = grid.stride;
    const int plane_stride = grid.plane_stride;

#ifdef USE_GPU_OFFLOAD
    int device = omp_get_default_device();
    double* f_ptr = static_cast<double*>(omp_get_mapped_ptr(f_ptrs_[level], device));
#else
    double* f_ptr = f_ptrs_[level];
#endif

    if (Nz == 1) {
        // 2D case - compute sum
#ifdef USE_GPU_OFFLOAD
        #pragma omp target teams distribute parallel for reduction(+:sum) is_device_ptr(f_ptr)
#endif
        for (int idx = 0; idx < Nx * Ny; ++idx) {
            int i = idx % Nx + Ng;
            int j = idx / Nx + Ng;
            sum += f_ptr[j * stride + i];
        }

        double mean = sum / (Nx * Ny);

        // 2D case - subtract mean
#ifdef USE_GPU_OFFLOAD
        #pragma omp target teams distribute parallel for is_device_ptr(f_ptr)
#endif
        for (int idx = 0; idx < Nx * Ny; ++idx) {
            int i = idx % Nx + Ng;
            int j = idx / Nx + Ng;
            f_ptr[j * stride + i] -= mean;
        }
    } else {
        // 3D case - compute sum
#ifdef USE_GPU_OFFLOAD
        #pragma omp target teams distribute parallel for reduction(+:sum) is_device_ptr(f_ptr)
#endif
        for (int idx = 0; idx < Nx * Ny * Nz; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % Ny + Ng;
            int k = idx / (Nx * Ny) + Ng;
            sum += f_ptr[k * plane_stride + j * stride + i];
        }

        double mean = sum / (Nx * Ny * Nz);

        // 3D case - subtract mean
#ifdef USE_GPU_OFFLOAD
        #pragma omp target teams distribute parallel for is_device_ptr(f_ptr)
#endif
        for (int idx = 0; idx < Nx * Ny * Nz; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % Ny + Ng;
            int k = idx / (Nx * Ny) + Ng;
            f_ptr[k * plane_stride + j * stride + i] -= mean;
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
        converged_ = false;       // Not applicable - no convergence check performed
        fixed_cycle_mode_ = true; // Flag that we used fixed-cycle mode

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
    const int Ng_f = finest_cpu.Ng;  // Use actual Ng for correct interior indexing (O4 has Ng=2)
    b_inf_ = 0.0;
    double b_sum_sq = 0.0;
    static int cheby_growth_warnings_cpu = 0;  // Throttle residual-growth warnings
    if (mesh_->is2D()) {
        for (int j = Ng_f; j < Ng_f + finest_cpu.Ny; ++j) {
            for (int i = Ng_f; i < Ng_f + finest_cpu.Nx; ++i) {
                double val = finest_cpu.f(i, j);
                b_inf_ = std::max(b_inf_, std::abs(val));
                b_sum_sq += val * val;
            }
        }
    } else {
        for (int k = Ng_f; k < Ng_f + finest_cpu.Nz; ++k) {
            for (int j = Ng_f; j < Ng_f + finest_cpu.Ny; ++j) {
                for (int i = Ng_f; i < Ng_f + finest_cpu.Nx; ++i) {
                    double val = finest_cpu.f(i, j, k);
                    b_inf_ = std::max(b_inf_, std::abs(val));
                    b_sum_sq += val * val;
                }
            }
        }
    }
    b_l2_ = std::sqrt(b_sum_sq);

    // Apply BCs to initial guess before any computation
    // Critical for y-line smoother which needs correct x/z ghost values for RHS
    apply_bc(0);

    // Initial residual - use fused function to compute residual + both norms in single pass
    compute_residual_and_norms(0, r0_, r0_l2_);
    residual_ = r0_;
    residual_l2_ = r0_l2_;
    converged_ = false;        // Reset: will be set true if tolerance achieved
    fixed_cycle_mode_ = false; // Convergence-based mode

    int cycles_used = 0;
    double prev_residual_cpu = r0_l2_;  // Track for non-contraction detection

    for (int cycle = 0; cycle < max_cycles; ++cycle) {
        vcycle(0, nu1, nu2, degree);
        cycles_used = cycle + 1;

        // Check convergence every check_interval cycles (reduces overhead)
        if ((cycle % check_interval) == (check_interval - 1) || cycle == max_cycles - 1) {
            // Fused residual + norm computation (single pass over memory, single GPU reduction)
            compute_residual_and_norms(0, residual_, residual_l2_);

            // SAFETY: Detect non-contraction (residual growing) - indicates smoother issue
            if (residual_l2_ > 1.05 * prev_residual_cpu && prev_residual_cpu > 1e-30) {
                if (smoother_type_ == MGSmootherType::Chebyshev && cheby_growth_warnings_cpu < 5) {
                    std::cerr << "[MG WARNING] Residual grew: " << std::scientific << prev_residual_cpu
                              << " -> " << residual_l2_ << " (cycle " << cycle
                              << "). Consider MG_SMOOTHER=jacobi\n" << std::defaultfloat;
                    cheby_growth_warnings_cpu++;
                }
            }
            prev_residual_cpu = residual_l2_;

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
                converged_ = true;  // Record that we achieved tolerance
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

    // For Neumann/Periodic BCs, make RHS mean-free BEFORE solving (compatibility condition)
    // This is critical for convergence - without it, MG can "wander" in the nullspace
    if (has_nullspace()) {
        make_rhs_mean_free(0);
    }

    // Optional RHS pre-smoothing on finest level (semi-coarsening + Neumann only)
    // This removes near-grid noise from intermediate velocity divergence
    // without affecting the mean-free constraint. Costs ~1 xz-RBGS sweep.
    // Enable via environment: MG_PRESMOOTH_RHS=1
    static bool presmooth_rhs = (std::getenv("MG_PRESMOOTH_RHS") != nullptr);
    if (presmooth_rhs && semi_coarsening_ && has_nullspace() && levels_[0]->Nz > 1) {
        // Apply mild smoothing to RHS (stored in f_ptrs_[0])
        // This is a Jacobi-like operation on the RHS field itself
        // We use the tmp buffer and xz-RBGS-like structure but on f instead of u
        // For simplicity, just do 1 damped Jacobi sweep on the RHS
        auto& grid = *levels_[0];
        const double dx2 = grid.dx * grid.dx;
        const double dz2 = grid.dz * grid.dz;
        const int Ng = grid.Ng;
        const int Nx = grid.Nx;
        const int Ny = grid.Ny;
        const int Nz = grid.Nz;
        const int stride = grid.stride;
        const int plane_stride = grid.plane_stride;
        const double omega = 0.5;  // Light damping

#ifdef USE_GPU_OFFLOAD
        int device = omp_get_default_device();
        double* f_ptr = static_cast<double*>(omp_get_mapped_ptr(f_ptrs_[0], device));
        double* tmp_ptr = tmp_ptrs_[0];

        // Copy f to tmp
        size_t total = level_sizes_[0];
        #pragma omp target teams distribute parallel for is_device_ptr(f_ptr, tmp_ptr)
        for (size_t idx = 0; idx < total; ++idx) tmp_ptr[idx] = f_ptr[idx];

        // One Jacobi sweep: smooth f in xz-planes (average with neighbors)
        double coeff_xz = 2.0/dx2 + 2.0/dz2;
        #pragma omp target teams distribute parallel for collapse(3) is_device_ptr(f_ptr, tmp_ptr)
        for (int k = Ng; k < Nz + Ng; ++k) {
            for (int j = Ng; j < Ny + Ng; ++j) {
                for (int i = Ng; i < Nx + Ng; ++i) {
                    int idx = k * plane_stride + j * stride + i;
                    double f_avg = (tmp_ptr[idx+1] + tmp_ptr[idx-1]) / dx2 +
                                   (tmp_ptr[idx+plane_stride] + tmp_ptr[idx-plane_stride]) / dz2;
                    f_ptr[idx] = (1.0 - omega) * tmp_ptr[idx] + omega * f_avg / coeff_xz;
                }
            }
        }
#endif
        // Re-enforce mean-free after smoothing
        make_rhs_mean_free(0);
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

            // Compute ||b||_2 BEFORE running any V-cycles (RHS is still pristine)
            // This matches the convergence mode pattern
            // CRITICAL: Sync to ensure data copy is complete before reading
            CUDA_CHECK_SYNC(cudaDeviceSynchronize());
            {
                auto& finest = *levels_[0];
                const int Ng = finest.Ng;
                const int Nx = finest.Nx;
                const int Ny = finest.Ny;
                const int Nz = finest.Nz;
                const int stride = finest.stride;
                const int plane_stride = finest.plane_stride;
                const bool is_2d = finest.is2D();

                // NVHPC WORKAROUND: Use member pointer f_level0_ptr_ instead of vector element f_ptrs_[0]
                // Vector element access can return stale addresses in NVHPC
                int device = omp_get_default_device();
                const double* f_ptr = static_cast<const double*>(omp_get_mapped_ptr(f_level0_ptr_, device));

                double b_sum_sq = 0.0;

                if (is_2d) {
                    // 2D data lives at plane k=Ng (middle z-plane), not k=0
                    // Memory layout: Sz = 1 + 2*Ng, data at plane Ng
                    const int k_plane_offset = Ng * plane_stride;
                    #pragma omp target teams distribute parallel for collapse(2) \
                        is_device_ptr(f_ptr) reduction(+: b_sum_sq)
                    for (int j = Ng; j < Ny + Ng; ++j) {
                        for (int i = Ng; i < Nx + Ng; ++i) {
                            int idx = k_plane_offset + j * stride + i;
                            double val = f_ptr[idx];
                            b_sum_sq += val * val;
                        }
                    }
                } else {
                    #pragma omp target teams distribute parallel for collapse(3) \
                        is_device_ptr(f_ptr) reduction(+: b_sum_sq)
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

                // Sanity check: if ||b||_2 is invalid or garbage, fall back to all cycles
                // This prevents early exit on bad reduction results
                // - NaN/Inf: obviously invalid
                // - Very small (<1e-30): likely zeros from wrong memory location
                // - Very large (>1e15): likely garbage from bad reduction
                if (!std::isfinite(b_l2_) || b_l2_ < 1e-30 || b_l2_ > 1e15) {
                    b_l2_ = 0.0;  // Force rel_res check to use raw residual
                }
            }

            // First batch of cycles
            int batch = std::min(check_after, max_cycles);
            run_cycles(batch);
            cycles_run = batch;

            // Sync after V-cycles before checking residual
            CUDA_CHECK_SYNC(cudaDeviceSynchronize());

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

        // Mark that we used fixed-cycle mode (for status reporting)
        converged_ = false;       // Not applicable - no convergence check performed
        fixed_cycle_mode_ = true; // Flag that we used fixed-cycle mode

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
            // 2D data lives at plane k=Ng (middle z-plane), not k=0
            // Memory layout: Sz = 1 + 2*Ng, data at plane Ng
            const int k_plane_offset = Ng * plane_stride_gpu;
            #pragma omp target teams distribute parallel for collapse(2) \
                is_device_ptr(f_dev) reduction(max: b_inf_local) reduction(+: b_sum_sq)
            for (int j = Ng; j < Ny_g + Ng; ++j) {
                for (int i = Ng; i < Nx_g + Ng; ++i) {
                    int idx = k_plane_offset + j * stride_gpu + i;
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

    // Apply BCs to initial guess before any computation
    // Critical for y-line smoother which needs correct x/z ghost values for RHS
    apply_bc(0);

    // Initial residual - use fused function to compute residual + both norms in single pass
    compute_residual_and_norms(0, r0_, r0_l2_);
    residual_ = r0_;
    residual_l2_ = r0_l2_;
    converged_ = false;        // Reset: will be set true if tolerance achieved
    fixed_cycle_mode_ = false; // Convergence-based mode

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
                converged_ = true;  // Record that we achieved tolerance
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
    
    // Map y-metric arrays for non-uniform y-spacing (level 0 only)
    if (y_stretched_ && yLap_aS_ && yLap_aN_ && yLap_aP_) {
        #pragma omp target enter data map(to: yLap_aS_[0:y_metrics_size_], yLap_aN_[0:y_metrics_size_], yLap_aP_[0:y_metrics_size_])
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

    // Unmap y-metric arrays for non-uniform y-spacing
    if (y_stretched_ && yLap_aS_ && yLap_aN_ && yLap_aP_) {
        #pragma omp target exit data map(delete: yLap_aS_[0:y_metrics_size_], yLap_aN_[0:y_metrics_size_], yLap_aP_[0:y_metrics_size_])
    }

    // Free PCG scratch buffers
    if (pcg_p_) {
        omp_target_free(pcg_p_, device_id);
        omp_target_free(pcg_z_, device_id);
        omp_target_free(pcg_Ap_, device_id);
        omp_target_free(pcg_x_, device_id);
        omp_target_free(pcg_f_save_, device_id);
        pcg_p_ = nullptr;
        pcg_z_ = nullptr;
        pcg_Ap_ = nullptr;
        pcg_x_ = nullptr;
        pcg_f_save_ = nullptr;
        pcg_buf_size_ = 0;
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
        cfg.Ng = grid.Ng;  // Use actual Ng (finest level may have Ng=2 for O4 stencils)
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
    // Free PCG scratch buffers (CPU version)
    delete[] pcg_p_;
    delete[] pcg_z_;
    delete[] pcg_Ap_;
    delete[] pcg_x_;
    delete[] pcg_f_save_;
    pcg_p_ = nullptr;
    pcg_z_ = nullptr;
    pcg_Ap_ = nullptr;
    pcg_x_ = nullptr;
    pcg_f_save_ = nullptr;
    pcg_buf_size_ = 0;
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

