/// @file poisson_solver_multigrid.cpp
/// @brief Geometric multigrid solver for pressure Poisson equation
///
/// This file implements a V-cycle geometric multigrid solver achieving O(N)
/// complexity for the pressure correction equation in the fractional-step method.
/// Key features:
/// - V-cycle algorithm with SOR smoothing
/// - Automatic mesh hierarchy construction (restriction to coarsest level)
/// - Full weighting restriction and bilinear prolongation
/// - GPU-accelerated smoothing and residual computation
/// - 10-100x faster than pure SOR iteration for large grids
///
/// The solver constructs a hierarchy of grids by recursive coarsening and solves
/// the system using recursive V-cycles that combine smoothing on each level with
/// coarse-grid correction.

#include "poisson_solver_multigrid.hpp"
#include "gpu_utils.hpp"
#include "profiling.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstring>  // for memcpy in debug
#include <cassert>
#include <limits>   // for std::numeric_limits (NaN handling)

namespace nncfd {

MultigridPoissonSolver::MultigridPoissonSolver(const Mesh& mesh) : mesh_(&mesh) {
    create_hierarchy();
    
#ifdef USE_GPU_OFFLOAD
    // Initialize GPU buffers immediately - will throw if no GPU available
    initialize_gpu_buffers();
#endif
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

    // Finest level
    if (is_2d) {
        levels_.push_back(std::make_unique<GridLevel>(Nx, Ny, dx, dy));
    } else {
        levels_.push_back(std::make_unique<GridLevel>(Nx, Ny, Nz, dx, dy, dz));
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
            levels_.push_back(std::make_unique<GridLevel>(Nx, Ny, dx, dy));
        }
    } else {
        while (Nx > MIN_COARSE_SIZE && Ny > MIN_COARSE_SIZE && Nz > MIN_COARSE_SIZE) {
            Nx /= 2;
            Ny /= 2;
            Nz /= 2;
            dx *= 2.0;
            dy *= 2.0;
            dz *= 2.0;
            levels_.push_back(std::make_unique<GridLevel>(Nx, Ny, Nz, dx, dy, dz));
        }
    }
}

void MultigridPoissonSolver::apply_bc(int level) {
    // UNIFIED CPU/GPU implementation for boundary conditions
    // Uses raw pointers and identical arithmetic for bitwise consistency
    auto& grid = *levels_[level];
    const int Nx = grid.Nx;
    const int Ny = grid.Ny;
    const int Nz = grid.Nz;
    const int Ng = 1;  // Ghost cells
    const bool is_2d = grid.is2D();
    const int stride = Nx + 2*Ng;
    const int plane_stride = stride * (Ny + 2*Ng);

    // Convert BCs to integers for branchless GPU code
    const int bc_x_lo = static_cast<int>(bc_x_lo_);
    const int bc_x_hi = static_cast<int>(bc_x_hi_);
    const int bc_y_lo = static_cast<int>(bc_y_lo_);
    const int bc_y_hi = static_cast<int>(bc_y_hi_);
    const int bc_z_lo = static_cast<int>(bc_z_lo_);
    const int bc_z_hi = static_cast<int>(bc_z_hi_);
    const double dval = dirichlet_val_;

    // Set up pointers - GPU uses device buffers, CPU uses grid arrays
#ifdef USE_GPU_OFFLOAD
    double* u_ptr = u_ptrs_[level];
    const size_t total_size = level_sizes_[level];
    #define BC_TARGET_FOR_X \
        _Pragma("omp target teams distribute parallel for map(present: u_ptr[0:total_size])")
    #define BC_TARGET_FOR_Y \
        _Pragma("omp target teams distribute parallel for map(present: u_ptr[0:total_size])")
    #define BC_TARGET_FOR_Z \
        _Pragma("omp target teams distribute parallel for map(present: u_ptr[0:total_size])")
#else
    double* u_ptr = grid.u.data().data();
    #define BC_TARGET_FOR_X
    #define BC_TARGET_FOR_Y
    #define BC_TARGET_FOR_Z
#endif

    if (is_2d) {
        // ========== 2D BOUNDARY CONDITIONS ==========
        // Pass 1: x-direction boundaries (loop over j)
        BC_TARGET_FOR_X
        for (int j = 0; j < Ny + 2*Ng; ++j) {
            int idx_lo = j * stride + 0;           // Left ghost (i=0)
            int idx_hi = j * stride + (Nx + Ng);   // Right ghost (i=Nx+1)

            // Left boundary (i=0)
            if (bc_x_lo == 2) { // Periodic
                u_ptr[idx_lo] = u_ptr[j * stride + Nx];
            } else if (bc_x_lo == 1) { // Neumann
                u_ptr[idx_lo] = u_ptr[j * stride + Ng];
            } else { // Dirichlet
                u_ptr[idx_lo] = 2.0 * dval - u_ptr[j * stride + Ng];
            }

            // Right boundary (i=Nx+1)
            if (bc_x_hi == 2) { // Periodic
                u_ptr[idx_hi] = u_ptr[j * stride + Ng];
            } else if (bc_x_hi == 1) { // Neumann
                u_ptr[idx_hi] = u_ptr[j * stride + (Nx + Ng - 1)];
            } else { // Dirichlet
                u_ptr[idx_hi] = 2.0 * dval - u_ptr[j * stride + (Nx + Ng - 1)];
            }
        }

        // Pass 2: y-direction boundaries (loop over i)
        BC_TARGET_FOR_Y
        for (int i = 0; i < Nx + 2*Ng; ++i) {
            int idx_lo = 0 * stride + i;               // Bottom ghost (j=0)
            int idx_hi = (Ny + Ng) * stride + i;       // Top ghost (j=Ny+1)

            // Bottom boundary (j=0)
            if (bc_y_lo == 2) { // Periodic
                u_ptr[idx_lo] = u_ptr[Ny * stride + i];
            } else if (bc_y_lo == 1) { // Neumann
                u_ptr[idx_lo] = u_ptr[Ng * stride + i];
            } else { // Dirichlet
                u_ptr[idx_lo] = 2.0 * dval - u_ptr[Ng * stride + i];
            }

            // Top boundary (j=Ny+1)
            if (bc_y_hi == 2) { // Periodic
                u_ptr[idx_hi] = u_ptr[Ng * stride + i];
            } else if (bc_y_hi == 1) { // Neumann
                u_ptr[idx_hi] = u_ptr[(Ny + Ng - 1) * stride + i];
            } else { // Dirichlet
                u_ptr[idx_hi] = 2.0 * dval - u_ptr[(Ny + Ng - 1) * stride + i];
            }
        }

        // Pass 3: Re-apply x-direction for corner consistency (periodic BCs)
        // This ensures corners get correct values when both x and y are periodic
        const bool needs_corner_fix = (bc_x_lo == 2 || bc_x_hi == 2 || bc_y_lo == 2 || bc_y_hi == 2);
        if (needs_corner_fix) {
            BC_TARGET_FOR_X
            for (int j = 0; j < Ny + 2*Ng; ++j) {
                int idx_lo = j * stride + 0;
                int idx_hi = j * stride + (Nx + Ng);

                if (bc_x_lo == 2) {
                    u_ptr[idx_lo] = u_ptr[j * stride + Nx];
                } else if (bc_x_lo == 1) {
                    u_ptr[idx_lo] = u_ptr[j * stride + Ng];
                } else {
                    u_ptr[idx_lo] = 2.0 * dval - u_ptr[j * stride + Ng];
                }

                if (bc_x_hi == 2) {
                    u_ptr[idx_hi] = u_ptr[j * stride + Ng];
                } else if (bc_x_hi == 1) {
                    u_ptr[idx_hi] = u_ptr[j * stride + (Nx + Ng - 1)];
                } else {
                    u_ptr[idx_hi] = 2.0 * dval - u_ptr[j * stride + (Nx + Ng - 1)];
                }
            }

            BC_TARGET_FOR_Y
            for (int i = 0; i < Nx + 2*Ng; ++i) {
                int idx_lo = 0 * stride + i;
                int idx_hi = (Ny + Ng) * stride + i;

                if (bc_y_lo == 2) {
                    u_ptr[idx_lo] = u_ptr[Ny * stride + i];
                } else if (bc_y_lo == 1) {
                    u_ptr[idx_lo] = u_ptr[Ng * stride + i];
                } else {
                    u_ptr[idx_lo] = 2.0 * dval - u_ptr[Ng * stride + i];
                }

                if (bc_y_hi == 2) {
                    u_ptr[idx_hi] = u_ptr[Ng * stride + i];
                } else if (bc_y_hi == 1) {
                    u_ptr[idx_hi] = u_ptr[(Ny + Ng - 1) * stride + i];
                } else {
                    u_ptr[idx_hi] = 2.0 * dval - u_ptr[(Ny + Ng - 1) * stride + i];
                }
            }
        }
    } else {
        // ========== 3D BOUNDARY CONDITIONS ==========
        // Use face-by-face approach matching 2D structure for consistency
        const int n_jk = (Ny + 2*Ng) * (Nz + 2*Ng);
        const int n_ik = (Nx + 2*Ng) * (Nz + 2*Ng);
        const int n_ij = (Nx + 2*Ng) * (Ny + 2*Ng);

        // Pass 1: x-direction boundaries (loop over j,k faces)
        BC_TARGET_FOR_X
        for (int jk = 0; jk < n_jk; ++jk) {
            int j = jk % (Ny + 2*Ng);
            int k = jk / (Ny + 2*Ng);
            int idx_lo = k * plane_stride + j * stride + 0;
            int idx_hi = k * plane_stride + j * stride + (Nx + Ng);

            // Left boundary (i=0)
            if (bc_x_lo == 2) {
                u_ptr[idx_lo] = u_ptr[k * plane_stride + j * stride + Nx];
            } else if (bc_x_lo == 1) {
                u_ptr[idx_lo] = u_ptr[k * plane_stride + j * stride + Ng];
            } else {
                u_ptr[idx_lo] = 2.0 * dval - u_ptr[k * plane_stride + j * stride + Ng];
            }

            // Right boundary (i=Nx+1)
            if (bc_x_hi == 2) {
                u_ptr[idx_hi] = u_ptr[k * plane_stride + j * stride + Ng];
            } else if (bc_x_hi == 1) {
                u_ptr[idx_hi] = u_ptr[k * plane_stride + j * stride + (Nx + Ng - 1)];
            } else {
                u_ptr[idx_hi] = 2.0 * dval - u_ptr[k * plane_stride + j * stride + (Nx + Ng - 1)];
            }
        }

        // Pass 2: y-direction boundaries (loop over i,k faces)
        BC_TARGET_FOR_Y
        for (int ik = 0; ik < n_ik; ++ik) {
            int i = ik % (Nx + 2*Ng);
            int k = ik / (Nx + 2*Ng);
            int idx_lo = k * plane_stride + 0 * stride + i;
            int idx_hi = k * plane_stride + (Ny + Ng) * stride + i;

            // Bottom boundary (j=0)
            if (bc_y_lo == 2) {
                u_ptr[idx_lo] = u_ptr[k * plane_stride + Ny * stride + i];
            } else if (bc_y_lo == 1) {
                u_ptr[idx_lo] = u_ptr[k * plane_stride + Ng * stride + i];
            } else {
                u_ptr[idx_lo] = 2.0 * dval - u_ptr[k * plane_stride + Ng * stride + i];
            }

            // Top boundary (j=Ny+1)
            if (bc_y_hi == 2) {
                u_ptr[idx_hi] = u_ptr[k * plane_stride + Ng * stride + i];
            } else if (bc_y_hi == 1) {
                u_ptr[idx_hi] = u_ptr[k * plane_stride + (Ny + Ng - 1) * stride + i];
            } else {
                u_ptr[idx_hi] = 2.0 * dval - u_ptr[k * plane_stride + (Ny + Ng - 1) * stride + i];
            }
        }

        // Pass 3: z-direction boundaries (loop over i,j faces)
        BC_TARGET_FOR_Z
        for (int ij = 0; ij < n_ij; ++ij) {
            int i = ij % (Nx + 2*Ng);
            int j = ij / (Nx + 2*Ng);
            int idx_lo = 0 * plane_stride + j * stride + i;
            int idx_hi = (Nz + Ng) * plane_stride + j * stride + i;

            // Back boundary (k=0)
            if (bc_z_lo == 2) {
                u_ptr[idx_lo] = u_ptr[Nz * plane_stride + j * stride + i];
            } else if (bc_z_lo == 1) {
                u_ptr[idx_lo] = u_ptr[Ng * plane_stride + j * stride + i];
            } else {
                u_ptr[idx_lo] = 2.0 * dval - u_ptr[Ng * plane_stride + j * stride + i];
            }

            // Front boundary (k=Nz+1)
            if (bc_z_hi == 2) {
                u_ptr[idx_hi] = u_ptr[Ng * plane_stride + j * stride + i];
            } else if (bc_z_hi == 1) {
                u_ptr[idx_hi] = u_ptr[(Nz + Ng - 1) * plane_stride + j * stride + i];
            } else {
                u_ptr[idx_hi] = 2.0 * dval - u_ptr[(Nz + Ng - 1) * plane_stride + j * stride + i];
            }
        }
    }

    // Clean up macros
    #undef BC_TARGET_FOR_X
    #undef BC_TARGET_FOR_Y
    #undef BC_TARGET_FOR_Z
}

void MultigridPoissonSolver::apply_bc_to_residual(int level) {
    // Apply boundary conditions to the residual array for proper restriction
    // The 9-point restriction stencil reads from ghost cells, so they must be set
    auto& grid = *levels_[level];
    const int Nx = grid.Nx;
    const int Ny = grid.Ny;
    const int Nz = grid.Nz;
    const int Ng = 1;
    const bool is_2d = grid.is2D();

#ifdef USE_GPU_OFFLOAD
    // GPU path for residual BCs
    if (gpu_ready_) {
        const size_t total_size = level_sizes_[level];
        double* r_ptr = r_ptrs_[level];
        const int stride = Nx + 2*Ng;

        // Convert BCs to integers for GPU
        const int bc_x_lo = static_cast<int>(bc_x_lo_);
        const int bc_x_hi = static_cast<int>(bc_x_hi_);
        const int bc_y_lo = static_cast<int>(bc_y_lo_);
        const int bc_y_hi = static_cast<int>(bc_y_hi_);
        const int bc_z_lo = static_cast<int>(bc_z_lo_);
        const int bc_z_hi = static_cast<int>(bc_z_hi_);

        if (is_2d) {
            // 2D GPU path for residual
            // x-direction boundaries
            #pragma omp target teams distribute parallel for \
                map(present: r_ptr[0:total_size])
            for (int j = 0; j < Ny + 2; ++j) {
                int idx = j * stride;
                // Left boundary (i=0)
                if (bc_x_lo == 2) { // Periodic
                    r_ptr[idx] = r_ptr[idx + Nx];
                } else { // Neumann/Dirichlet: zero ghost
                    r_ptr[idx] = 0.0;
                }
                // Right boundary (i=Nx+1)
                if (bc_x_hi == 2) { // Periodic
                    r_ptr[idx + Nx + Ng] = r_ptr[idx + Ng];
                } else {
                    r_ptr[idx + Nx + Ng] = 0.0;
                }
            }

            // y-direction boundaries
            #pragma omp target teams distribute parallel for \
                map(present: r_ptr[0:total_size])
            for (int i = 0; i < Nx + 2; ++i) {
                // Bottom boundary (j=0)
                if (bc_y_lo == 2) { // Periodic
                    r_ptr[i] = r_ptr[Ny * stride + i];
                } else {
                    r_ptr[i] = 0.0;
                }
                // Top boundary (j=Ny+1)
                if (bc_y_hi == 2) { // Periodic
                    r_ptr[(Ny + Ng) * stride + i] = r_ptr[Ng * stride + i];
                } else {
                    r_ptr[(Ny + Ng) * stride + i] = 0.0;
                }
            }
        } else {
            // 3D GPU path for residual - unified kernel matching apply_bc
            const int plane_stride = stride * (Ny + 2*Ng);
            const int Nx_g = Nx + 2*Ng;
            const int Ny_g = Ny + 2*Ng;
            const int Nz_g = Nz + 2*Ng;
            const int n_total_g = Nx_g * Ny_g * Nz_g;

            #pragma omp target teams distribute parallel for \
                map(present: r_ptr[0:total_size]) \
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
    if (is_2d) {
        // 2D CPU path for residual
        // x-direction boundaries
        for (int j = 0; j < Ny + 2*Ng; ++j) {
            // Left boundary
            switch (bc_x_lo_) {
                case PoissonBC::Periodic:
                    grid.r(0, j) = grid.r(Nx, j);
                    break;
                case PoissonBC::Neumann:
                case PoissonBC::Dirichlet:
                    grid.r(0, j) = 0.0;  // Zero ghost for Neumann/Dirichlet residual
                    break;
            }
            // Right boundary
            switch (bc_x_hi_) {
                case PoissonBC::Periodic:
                    grid.r(Nx + Ng, j) = grid.r(Ng, j);
                    break;
                case PoissonBC::Neumann:
                case PoissonBC::Dirichlet:
                    grid.r(Nx + Ng, j) = 0.0;
                    break;
            }
        }
        // y-direction boundaries
        for (int i = 0; i < Nx + 2*Ng; ++i) {
            // Bottom boundary
            switch (bc_y_lo_) {
                case PoissonBC::Periodic:
                    grid.r(i, 0) = grid.r(i, Ny);
                    break;
                case PoissonBC::Neumann:
                case PoissonBC::Dirichlet:
                    grid.r(i, 0) = 0.0;
                    break;
            }
            // Top boundary
            switch (bc_y_hi_) {
                case PoissonBC::Periodic:
                    grid.r(i, Ny + Ng) = grid.r(i, Ng);
                    break;
                case PoissonBC::Neumann:
                case PoissonBC::Dirichlet:
                    grid.r(i, Ny + Ng) = 0.0;
                    break;
            }
        }
    } else {
        // 3D CPU path for residual
        // x-direction
        for (int k = 0; k < Nz + 2*Ng; ++k) {
            for (int j = 0; j < Ny + 2*Ng; ++j) {
                switch (bc_x_lo_) {
                    case PoissonBC::Periodic:
                        grid.r(0, j, k) = grid.r(Nx, j, k);
                        break;
                    default:
                        grid.r(0, j, k) = 0.0;
                        break;
                }
                switch (bc_x_hi_) {
                    case PoissonBC::Periodic:
                        grid.r(Nx + Ng, j, k) = grid.r(Ng, j, k);
                        break;
                    default:
                        grid.r(Nx + Ng, j, k) = 0.0;
                        break;
                }
            }
        }
        // y-direction
        for (int k = 0; k < Nz + 2*Ng; ++k) {
            for (int i = 0; i < Nx + 2*Ng; ++i) {
                switch (bc_y_lo_) {
                    case PoissonBC::Periodic:
                        grid.r(i, 0, k) = grid.r(i, Ny, k);
                        break;
                    default:
                        grid.r(i, 0, k) = 0.0;
                        break;
                }
                switch (bc_y_hi_) {
                    case PoissonBC::Periodic:
                        grid.r(i, Ny + Ng, k) = grid.r(i, Ng, k);
                        break;
                    default:
                        grid.r(i, Ny + Ng, k) = 0.0;
                        break;
                }
            }
        }
        // z-direction
        for (int j = 0; j < Ny + 2*Ng; ++j) {
            for (int i = 0; i < Nx + 2*Ng; ++i) {
                switch (bc_z_lo_) {
                    case PoissonBC::Periodic:
                        grid.r(i, j, 0) = grid.r(i, j, Nz);
                        break;
                    default:
                        grid.r(i, j, 0) = 0.0;
                        break;
                }
                switch (bc_z_hi_) {
                    case PoissonBC::Periodic:
                        grid.r(i, j, Nz + Ng) = grid.r(i, j, Ng);
                        break;
                    default:
                        grid.r(i, j, Nz + Ng) = 0.0;
                        break;
                }
            }
        }
    }
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

    const int Ng = 1;
    const int Nx = grid.Nx;
    const int Ny = grid.Ny;
    const int Nz = grid.Nz;
    const int stride = Nx + 2;
    const int plane_stride = (Nx + 2) * (Ny + 2);
    const size_t total_size = static_cast<size_t>(Nx + 2) *
                              static_cast<size_t>(Ny + 2) *
                              static_cast<size_t>(Nz + 2);

    // Set up pointers - GPU uses device buffers, CPU uses grid arrays
#ifdef USE_GPU_OFFLOAD
    double* u_ptr = u_ptrs_[level];
    double* tmp_ptr = tmp_ptrs_[level];  // Device-only scratch buffer
    const double* f_ptr = f_ptrs_[level];
    // tmp_ptr is device-only (omp_target_alloc), use is_device_ptr
    #define JACOBI_TARGET_U_TO_TMP_2D \
        _Pragma("omp target teams distribute parallel for collapse(2) map(present: u_ptr[0:total_size], f_ptr[0:total_size]) is_device_ptr(tmp_ptr)")
    #define JACOBI_TARGET_TMP_TO_U_2D \
        _Pragma("omp target teams distribute parallel for collapse(2) map(present: u_ptr[0:total_size], f_ptr[0:total_size]) is_device_ptr(tmp_ptr)")
    #define JACOBI_TARGET_U_TO_TMP_3D \
        _Pragma("omp target teams distribute parallel for collapse(3) map(present: u_ptr[0:total_size], f_ptr[0:total_size]) is_device_ptr(tmp_ptr)")
    #define JACOBI_TARGET_TMP_TO_U_3D \
        _Pragma("omp target teams distribute parallel for collapse(3) map(present: u_ptr[0:total_size], f_ptr[0:total_size]) is_device_ptr(tmp_ptr)")
    #define JACOBI_TARGET_COPY \
        _Pragma("omp target teams distribute parallel for map(present: u_ptr[0:total_size]) is_device_ptr(tmp_ptr)")
#else
    double* u_ptr = grid.u.data().data();
    double* tmp_ptr = grid.r.data().data();  // Reuse r as scratch buffer
    const double* f_ptr = grid.f.data().data();
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

void MultigridPoissonSolver::smooth(int level, int iterations, double omega) {
    NVTX_SCOPE_POISSON("mg:smooth");

    // Red-Black Gauss-Seidel with SOR
    // UNIFIED CODE: Same arithmetic for CPU and GPU, pragma handles offloading
    auto& grid = *levels_[level];
    const double dx2 = grid.dx * grid.dx;
    const double dy2 = grid.dy * grid.dy;
    const double dz2 = grid.dz * grid.dz;
    const bool is_2d = grid.is2D();
    // Diagonal coefficient: 4 neighbors for 2D, 6 neighbors for 3D
    const double coeff = is_2d ? (2.0 / dx2 + 2.0 / dy2)
                               : (2.0 / dx2 + 2.0 / dy2 + 2.0 / dz2);

    const int Ng = 1;
    const int Nx = grid.Nx;
    const int Ny = grid.Ny;
    const int Nz = grid.Nz;
    const int stride = Nx + 2;
    const int plane_stride = (Nx + 2) * (Ny + 2);

    // Use raw pointers for unified CPU/GPU code
#ifdef USE_GPU_OFFLOAD
    double* u_ptr = u_ptrs_[level];
    const double* f_ptr = f_ptrs_[level];
    const size_t total_size = level_sizes_[level];
#else
    double* u_ptr = grid.u.data().data();
    const double* f_ptr = grid.f.data().data();
#endif

    if (is_2d) {
        for (int iter = 0; iter < iterations; ++iter) {
            // Red sweep (i + j even)
#ifdef USE_GPU_OFFLOAD
            #pragma omp target teams distribute parallel for collapse(2) \
                map(present: u_ptr[0:total_size], f_ptr[0:total_size])
#endif
            for (int j = Ng; j < Ny + Ng; ++j) {
                for (int i = Ng; i < Nx + Ng; ++i) {
                    if (((i + j) & 1) == 0) {
                        int idx = j * stride + i;
                        double u_old = u_ptr[idx];
                        double u_gs = ((u_ptr[idx+1] + u_ptr[idx-1]) / dx2
                                     + (u_ptr[idx+stride] + u_ptr[idx-stride]) / dy2
                                     - f_ptr[idx]) / coeff;
                        u_ptr[idx] = (1.0 - omega) * u_old + omega * u_gs;
                    }
                }
            }

            // Black sweep (i + j odd)
#ifdef USE_GPU_OFFLOAD
            #pragma omp target teams distribute parallel for collapse(2) \
                map(present: u_ptr[0:total_size], f_ptr[0:total_size])
#endif
            for (int j = Ng; j < Ny + Ng; ++j) {
                for (int i = Ng; i < Nx + Ng; ++i) {
                    if (((i + j) & 1) == 1) {
                        int idx = j * stride + i;
                        double u_old = u_ptr[idx];
                        double u_gs = ((u_ptr[idx+1] + u_ptr[idx-1]) / dx2
                                     + (u_ptr[idx+stride] + u_ptr[idx-stride]) / dy2
                                     - f_ptr[idx]) / coeff;
                        u_ptr[idx] = (1.0 - omega) * u_old + omega * u_gs;
                    }
                }
            }
        }
    } else {
        // 3D path
        for (int iter = 0; iter < iterations; ++iter) {
            // Red sweep (i + j + k even)
#ifdef USE_GPU_OFFLOAD
            #pragma omp target teams distribute parallel for collapse(3) \
                map(present: u_ptr[0:total_size], f_ptr[0:total_size])
#endif
            for (int k = Ng; k < Nz + Ng; ++k) {
                for (int j = Ng; j < Ny + Ng; ++j) {
                    for (int i = Ng; i < Nx + Ng; ++i) {
                        if (((i + j + k) & 1) == 0) {
                            int idx = k * plane_stride + j * stride + i;
                            double u_old = u_ptr[idx];
                            double u_gs = ((u_ptr[idx+1] + u_ptr[idx-1]) / dx2
                                         + (u_ptr[idx+stride] + u_ptr[idx-stride]) / dy2
                                         + (u_ptr[idx+plane_stride] + u_ptr[idx-plane_stride]) / dz2
                                         - f_ptr[idx]) / coeff;
                            u_ptr[idx] = (1.0 - omega) * u_old + omega * u_gs;
                        }
                    }
                }
            }

            // Black sweep (i + j + k odd)
#ifdef USE_GPU_OFFLOAD
            #pragma omp target teams distribute parallel for collapse(3) \
                map(present: u_ptr[0:total_size], f_ptr[0:total_size])
#endif
            for (int k = Ng; k < Nz + Ng; ++k) {
                for (int j = Ng; j < Ny + Ng; ++j) {
                    for (int i = Ng; i < Nx + Ng; ++i) {
                        if (((i + j + k) & 1) == 1) {
                            int idx = k * plane_stride + j * stride + i;
                            double u_old = u_ptr[idx];
                            double u_gs = ((u_ptr[idx+1] + u_ptr[idx-1]) / dx2
                                         + (u_ptr[idx+stride] + u_ptr[idx-stride]) / dy2
                                         + (u_ptr[idx+plane_stride] + u_ptr[idx-plane_stride]) / dz2
                                         - f_ptr[idx]) / coeff;
                            u_ptr[idx] = (1.0 - omega) * u_old + omega * u_gs;
                        }
                    }
                }
            }
        }
    }

    // Apply boundary conditions once after all smoothing iterations
    apply_bc(level);
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

    const int Ng = 1;
    const int Nx = grid.Nx;
    const int Ny = grid.Ny;
    const int Nz = grid.Nz;
    const int stride = Nx + 2;
    const int plane_stride = (Nx + 2) * (Ny + 2);

    // Use raw pointers for unified CPU/GPU code
#ifdef USE_GPU_OFFLOAD
    const double* u_ptr = u_ptrs_[level];
    const double* f_ptr = f_ptrs_[level];
    double* r_ptr = r_ptrs_[level];
    const size_t total_size = level_sizes_[level];
#else
    const double* u_ptr = grid.u.data().data();
    const double* f_ptr = grid.f.data().data();
    double* r_ptr = grid.r.data().data();
#endif

    if (is_2d) {
#ifdef USE_GPU_OFFLOAD
        #pragma omp target teams distribute parallel for collapse(2) \
            map(present: u_ptr[0:total_size], f_ptr[0:total_size], r_ptr[0:total_size])
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
        #pragma omp target teams distribute parallel for collapse(3) \
            map(present: u_ptr[0:total_size], f_ptr[0:total_size], r_ptr[0:total_size])
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

void MultigridPoissonSolver::restrict_residual(int fine_level) {
    NVTX_SCOPE_MG("mg:restrict");

    // Full-weighting restriction from fine to coarse grid
    // UNIFIED CODE: Same arithmetic for CPU and GPU, pragma handles offloading
    auto& fine = *levels_[fine_level];
    auto& coarse = *levels_[fine_level + 1];
    const bool is_2d = fine.is2D();

    const int Ng = 1;
    const int Nx_c = coarse.Nx;
    const int Ny_c = coarse.Ny;
    const int Nz_c = coarse.Nz;
    const int stride_f = fine.Nx + 2;
    const int stride_c = coarse.Nx + 2;
    const int plane_stride_f = (fine.Nx + 2) * (fine.Ny + 2);
    const int plane_stride_c = (coarse.Nx + 2) * (coarse.Ny + 2);

    // Use raw pointers for unified CPU/GPU code
#ifdef USE_GPU_OFFLOAD
    const double* r_fine = r_ptrs_[fine_level];
    double* f_coarse = f_ptrs_[fine_level + 1];
    const size_t size_f = level_sizes_[fine_level];
    const size_t size_c = level_sizes_[fine_level + 1];
#else
    const double* r_fine = fine.r.data().data();
    double* f_coarse = coarse.f.data().data();
#endif

    if (is_2d) {
        // 2D: 9-point stencil
#ifdef USE_GPU_OFFLOAD
        #pragma omp target teams distribute parallel for collapse(2) \
            map(present: r_fine[0:size_f], f_coarse[0:size_c])
#endif
        for (int j_c = Ng; j_c < Ny_c + Ng; ++j_c) {
            for (int i_c = Ng; i_c < Nx_c + Ng; ++i_c) {
                int i_f = 2 * (i_c - Ng) + Ng;
                int j_f = 2 * (j_c - Ng) + Ng;
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
        #pragma omp target teams distribute parallel for collapse(3) \
            map(present: r_fine[0:size_f], f_coarse[0:size_c])
#endif
        for (int k_c = Ng; k_c < Nz_c + Ng; ++k_c) {
            for (int j_c = Ng; j_c < Ny_c + Ng; ++j_c) {
                for (int i_c = Ng; i_c < Nx_c + Ng; ++i_c) {
                    int i_f = 2 * (i_c - Ng) + Ng;
                    int j_f = 2 * (j_c - Ng) + Ng;
                    int k_f = 2 * (k_c - Ng) + Ng;
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

    const int Ng = 1;
    const int Nx_f = fine.Nx;
    const int Ny_f = fine.Ny;
    const int Nz_f = fine.Nz;
    const int stride_f = fine.Nx + 2;
    const int stride_c = coarse.Nx + 2;
    const int plane_stride_f = (fine.Nx + 2) * (fine.Ny + 2);
    const int plane_stride_c = (coarse.Nx + 2) * (coarse.Ny + 2);

    // Use raw pointers for unified CPU/GPU code
#ifdef USE_GPU_OFFLOAD
    const double* u_coarse = u_ptrs_[coarse_level];
    double* u_fine = u_ptrs_[coarse_level - 1];
    const size_t size_f = level_sizes_[coarse_level - 1];
    const size_t size_c = level_sizes_[coarse_level];
#else
    const double* u_coarse = coarse.u.data().data();
    double* u_fine = fine.u.data().data();
#endif

    if (is_2d) {
        // 2D owner-computes bilinear interpolation
        // Each fine cell reads from up to 4 coarse neighbors
#ifdef USE_GPU_OFFLOAD
        #pragma omp target teams distribute parallel for collapse(2) \
            map(present: u_coarse[0:size_c], u_fine[0:size_f])
#endif
        for (int j_f = Ng; j_f < Ny_f + Ng; ++j_f) {
            for (int i_f = Ng; i_f < Nx_f + Ng; ++i_f) {
                // Find base coarse cell and position within coarse cell pair
                int i_c = (i_f - Ng) / 2 + Ng;
                int j_c = (j_f - Ng) / 2 + Ng;
                int di = (i_f - Ng) & 1;  // 0 = coincident, 1 = midpoint
                int dj = (j_f - Ng) & 1;

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
        #pragma omp target teams distribute parallel for collapse(3) \
            map(present: u_coarse[0:size_c], u_fine[0:size_f])
#endif
        for (int k_f = Ng; k_f < Nz_f + Ng; ++k_f) {
            for (int j_f = Ng; j_f < Ny_f + Ng; ++j_f) {
                for (int i_f = Ng; i_f < Nx_f + Ng; ++i_f) {
                    // Find base coarse cell and position within coarse cell pair
                    int i_c = (i_f - Ng) / 2 + Ng;
                    int j_c = (j_f - Ng) / 2 + Ng;
                    int k_c = (k_f - Ng) / 2 + Ng;
                    int di = (i_f - Ng) & 1;  // 0 = coincident, 1 = midpoint
                    int dj = (j_f - Ng) & 1;
                    int dk = (k_f - Ng) & 1;

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
    // Direct solve on coarsest grid using many SOR iterations
    // Use omega=1.0 (pure Gauss-Seidel) for maximum stability on coarse grids
    // Higher omega can cause divergence on small grids with periodic BCs
    int coarsest = levels_.size() - 1;
    smooth(coarsest, iterations, 1.0);
}

void MultigridPoissonSolver::vcycle(int level, int nu1, int nu2) {
    NVTX_SCOPE_POISSON("mg:vcycle");

    if (level == static_cast<int>(levels_.size()) - 1) {
        // Coarsest level - solve approximately
        // With MIN_COARSE_SIZE=8, coarsest is 8x8 (64 points)
        // 20 iterations is sufficient for good convergence
        smooth_jacobi(level, 20, 0.8);
        return;
    }

    // Pre-smoothing with weighted Jacobi
    // Jacobi has 1 target region per iteration (vs 2 for RB-SOR)
    // omega=0.8 (damped Jacobi) for stability
    smooth_jacobi(level, nu1, 0.8);
    
    // Compute residual
    compute_residual(level);

    // Apply BCs to residual ghost cells for proper 9-point restriction
    // This is essential for periodic BCs where ghost cells must wrap around
    apply_bc_to_residual(level);

    // Restrict to coarse grid
    restrict_residual(level);
    
    // Zero coarse grid solution
    auto& coarse = *levels_[level + 1];
    
#ifdef USE_GPU_OFFLOAD
    assert(gpu_ready_ && "GPU must be initialized");
    const size_t size_c = level_sizes_[level + 1];
    double* u_coarse = u_ptrs_[level + 1];
    
    #pragma omp target teams distribute parallel for \
        map(present: u_coarse[0:size_c])
    for (int idx = 0; idx < (int)size_c; ++idx) {
        u_coarse[idx] = 0.0;
    }
#else
    const int Ng = 1;
    for (int k = 0; k < coarse.Nz + 2*Ng; ++k) {
        for (int j = 0; j < coarse.Ny + 2*Ng; ++j) {
            for (int i = 0; i < coarse.Nx + 2*Ng; ++i) {
                coarse.u(i, j, k) = 0.0;
            }
        }
    }
#endif
    
    // Recursive call to coarser level
    vcycle(level + 1, nu1, nu2);

    // Prolongate correction and apply boundary conditions
    // (BC needed before post-smoothing reads ghost cells)
    prolongate_correction(level + 1);
    apply_bc(level);

    // Post-smoothing with weighted Jacobi (projection mode: nu2=0 skips this)
    if (nu2 > 0) {
        smooth_jacobi(level, nu2, 0.8);
    }
}

double MultigridPoissonSolver::compute_max_residual(int level) {
    // UNIFIED CODE: Same arithmetic for CPU and GPU, pragma handles offloading
    compute_residual(level);

    auto& grid = *levels_[level];
    double max_res = 0.0;
    const int Ng = 1;
    const int Nx = grid.Nx;
    const int Ny = grid.Ny;
    const int Nz = grid.Nz;
    const int stride = Nx + 2;
    const int plane_stride = stride * (Ny + 2);

    // Use raw pointers for unified CPU/GPU code
#ifdef USE_GPU_OFFLOAD
    const double* r_ptr = r_ptrs_[level];
    const size_t total_size = level_sizes_[level];
#else
    const double* r_ptr = grid.r.data().data();
#endif

    if (Nz == 1) {
        // 2D case
#ifdef USE_GPU_OFFLOAD
        #pragma omp target teams distribute parallel for reduction(max:max_res) \
            map(present: r_ptr[0:total_size])
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
        #pragma omp target teams distribute parallel for reduction(max:max_res) \
            map(present: r_ptr[0:total_size])
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

void MultigridPoissonSolver::subtract_mean(int level) {
    // UNIFIED CODE: Same arithmetic for CPU and GPU, pragma handles offloading
    auto& grid = *levels_[level];
    double sum = 0.0;
    const int Ng = 1;
    const int Nx = grid.Nx;
    const int Ny = grid.Ny;
    const int Nz = grid.Nz;
    const int stride = Nx + 2;
    const int plane_stride = stride * (Ny + 2);

    // Use raw pointers for unified CPU/GPU code
#ifdef USE_GPU_OFFLOAD
    double* u_ptr = u_ptrs_[level];
    const size_t total_size = level_sizes_[level];
#else
    double* u_ptr = grid.u.data().data();
#endif

    if (Nz == 1) {
        // 2D case - compute sum
#ifdef USE_GPU_OFFLOAD
        #pragma omp target teams distribute parallel for reduction(+:sum) \
            map(present: u_ptr[0:total_size])
#endif
        for (int idx = 0; idx < Nx * Ny; ++idx) {
            int i = idx % Nx + Ng;
            int j = idx / Nx + Ng;
            sum += u_ptr[j * stride + i];
        }

        double mean = sum / (Nx * Ny);

        // 2D case - subtract mean
#ifdef USE_GPU_OFFLOAD
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:total_size])
#endif
        for (int idx = 0; idx < Nx * Ny; ++idx) {
            int i = idx % Nx + Ng;
            int j = idx / Nx + Ng;
            u_ptr[j * stride + i] -= mean;
        }
    } else {
        // 3D case - compute sum
#ifdef USE_GPU_OFFLOAD
        #pragma omp target teams distribute parallel for reduction(+:sum) \
            map(present: u_ptr[0:total_size])
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
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:total_size])
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
    const int Ng = 1;

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

    // MG V-cycles: use cfg.max_iter to control number of V-cycles
    // - Projection mode (max_iter <= 5): nu1=1, nu2=0 for fast projection
    // - Accurate mode (max_iter > 5): nu1=2, nu2=2 for fast convergence
    assert(cfg.max_iter > 0 && "PoissonConfig.max_iter must be positive");
    const int num_cycles = cfg.max_iter;
    const bool accurate_mode = (num_cycles > 5);
    const int nu1 = accurate_mode ? 2 : 1;  // Pre-smoothing sweeps
    const int nu2 = accurate_mode ? 2 : 0;  // Post-smoothing sweeps

    for (int cycle = 0; cycle < num_cycles; ++cycle) {
        vcycle(0, nu1, nu2);
    }

    // Final residual
    residual_ = compute_max_residual(0);
    
    // Subtract mean for singular Poisson problems (no Dirichlet BCs)
    // Whenever all boundaries are Neumann or Periodic, the solution is defined up to a constant
    // and we must fix the nullspace by subtracting the mean
    bool has_dirichlet = (bc_x_lo_ == PoissonBC::Dirichlet || bc_x_hi_ == PoissonBC::Dirichlet ||
                          bc_y_lo_ == PoissonBC::Dirichlet || bc_y_hi_ == PoissonBC::Dirichlet ||
                          bc_z_lo_ == PoissonBC::Dirichlet || bc_z_hi_ == PoissonBC::Dirichlet);

    if (!has_dirichlet) {
        subtract_mean(0);
        // Re-apply BCs after mean subtraction since ghost cells are now inconsistent
        apply_bc(0);
    }

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

    return num_cycles;  // Number of V-cycles executed
}

#ifdef USE_GPU_OFFLOAD
int MultigridPoissonSolver::solve_device(double* rhs_present, double* p_present, const PoissonConfig& cfg) {
    NVTX_SCOPE_POISSON("poisson:solve_device");

    assert(gpu_ready_ && "GPU must be initialized in constructor");
    
    // Device-resident solve using Model 1 (host pointer + present mapping)
    // Parameters are host pointers that caller has already mapped via `target enter data`.
    // We use map(present: ...) to access the device copies without additional transfers.
    
    auto& finest = *levels_[0];
    const int Nx = finest.Nx;
    const int Ny = finest.Ny;
    const int Nz = finest.Nz;
    // Total size includes ghost cells: (Nx+2)*(Ny+2)*(Nz+2) for 3D, or *3 for 2D (Nz=1)
    const size_t total_size = static_cast<size_t>(Nx + 2) *
                              static_cast<size_t>(Ny + 2) *
                              static_cast<size_t>(Nz + 2);

    // Get device pointers for finest level multigrid buffers
    double* u_dev = u_ptrs_[0];
    double* f_dev = f_ptrs_[0];

    // Copy RHS and initial guess from caller's present-mapped arrays to multigrid level-0 buffers
    // This is device-to-device copy via present mappings (no host staging)
    #pragma omp target teams distribute parallel for \
        map(present: rhs_present[0:total_size], p_present[0:total_size], f_dev[0:total_size], u_dev[0:total_size])
    for (size_t idx = 0; idx < total_size; ++idx) {
        f_dev[idx] = rhs_present[idx];
        u_dev[idx] = p_present[idx];
    }

    apply_bc(0);

    // MG V-cycles: use cfg.max_iter to control number of V-cycles
    // - Projection mode (max_iter <= 5): nu1=1, nu2=0 for fast projection
    // - Accurate mode (max_iter > 5): nu1=2, nu2=2 for fast convergence
    assert(cfg.max_iter > 0 && "PoissonConfig.max_iter must be positive");
    const int num_cycles = cfg.max_iter;
    const bool accurate_mode = (num_cycles > 5);
    const int nu1 = accurate_mode ? 2 : 1;  // Pre-smoothing sweeps
    const int nu2 = accurate_mode ? 2 : 0;  // Post-smoothing sweeps

    for (int cycle = 0; cycle < num_cycles; ++cycle) {
        vcycle(0, nu1, nu2);
    }

    // Final residual (always compute at end for diagnostics)
    residual_ = compute_max_residual(0);
    
    // Subtract mean for singular Poisson problems (no Dirichlet BCs)
    // Whenever all boundaries are Neumann or Periodic, the solution is defined up to a constant
    // and we must fix the nullspace by subtracting the mean
    bool has_dirichlet = (bc_x_lo_ == PoissonBC::Dirichlet || bc_x_hi_ == PoissonBC::Dirichlet ||
                          bc_y_lo_ == PoissonBC::Dirichlet || bc_y_hi_ == PoissonBC::Dirichlet ||
                          bc_z_lo_ == PoissonBC::Dirichlet || bc_z_hi_ == PoissonBC::Dirichlet);

    if (!has_dirichlet) {
        subtract_mean(0);
        // Re-apply BCs after mean subtraction since ghost cells are now inconsistent
        apply_bc(0);
    }

    // Copy result from multigrid level-0 buffer back to caller's present-mapped pointer
    // This is device-to-device copy via present mappings (no host staging)
    #pragma omp target teams distribute parallel for \
        map(present: p_present[0:total_size], u_dev[0:total_size])
    for (size_t idx = 0; idx < total_size; ++idx) {
        p_present[idx] = u_dev[idx];
    }

    return num_cycles;  // Number of V-cycles executed
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
        // 3D: include Nz+2 in size calculation
        const size_t total_size = static_cast<size_t>(grid.Nx + 2) *
                                  static_cast<size_t>(grid.Ny + 2) *
                                  static_cast<size_t>(grid.Nz + 2);
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
        double* r_ptr = r_ptrs_[lvl];
        double* tmp_ptr = tmp_ptrs_[lvl];
        #pragma omp target teams distribute parallel for \
            map(present: r_ptr[0:total_size]) is_device_ptr(tmp_ptr)
        for (size_t idx = 0; idx < total_size; ++idx) {
            r_ptr[idx] = 0.0;
            tmp_ptr[idx] = 0.0;
        }
    }
    
    // Verify mappings succeeded
    if (!u_ptrs_.empty() && !gpu::is_pointer_present(u_ptrs_[0])) {
        throw std::runtime_error("GPU mapping failed despite device availability");
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
#else
// No-op implementations when GPU offloading is disabled
void MultigridPoissonSolver::initialize_gpu_buffers() {
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

