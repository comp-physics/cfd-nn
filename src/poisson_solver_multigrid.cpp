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
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstring>  // for memcpy in debug
#include <cassert>

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
}

void MultigridPoissonSolver::create_hierarchy() {
    // Create grid hierarchy from fine to coarse
    int Nx = mesh_->Nx;
    int Ny = mesh_->Ny;
    double dx = mesh_->dx;
    double dy = mesh_->dy;
    
    // Finest level
    levels_.push_back(std::make_unique<GridLevel>(Nx, Ny, dx, dy));
    
    // Coarsen until we reach ~8x8 grid
    while (Nx > 8 && Ny > 8) {
        Nx /= 2;
        Ny /= 2;
        dx *= 2.0;
        dy *= 2.0;
        levels_.push_back(std::make_unique<GridLevel>(Nx, Ny, dx, dy));
    }
}

void MultigridPoissonSolver::apply_bc(int level) {
    auto& grid = *levels_[level];
    int Nx = grid.Nx;
    int Ny = grid.Ny;
    int Ng = 1;  // Ghost cells
    
#ifdef USE_GPU_OFFLOAD
    if (Nx >= 16 && Ny >= 16) {
        const size_t total_size = level_sizes_[level];
        double* u_ptr = u_ptrs_[level];
        const int stride = Nx + 2;
        
        // Convert BCs to integers for GPU
        const int bc_x_lo = static_cast<int>(bc_x_lo_);
        const int bc_x_hi = static_cast<int>(bc_x_hi_);
        const int bc_y_lo = static_cast<int>(bc_y_lo_);
        const int bc_y_hi = static_cast<int>(bc_y_hi_);
        const double dval = dirichlet_val_;
        
        // x-direction boundaries
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:total_size])
        for (int j = 0; j < Ny + 2; ++j) {
            int idx = j * stride;
            
            // Left boundary (i=0)
            if (bc_x_lo == 2) { // Periodic
                u_ptr[idx] = u_ptr[idx + Nx];
            } else if (bc_x_lo == 1) { // Neumann
                u_ptr[idx] = u_ptr[idx + Ng];
            } else { // Dirichlet
                u_ptr[idx] = 2.0 * dval - u_ptr[idx + Ng];
            }
            
            // Right boundary (i=Nx+1)
            if (bc_x_hi == 2) { // Periodic
                u_ptr[idx + Nx + Ng] = u_ptr[idx + Ng];
            } else if (bc_x_hi == 1) { // Neumann
                u_ptr[idx + Nx + Ng] = u_ptr[idx + Nx + Ng - 1];
            } else { // Dirichlet
                u_ptr[idx + Nx + Ng] = 2.0 * dval - u_ptr[idx + Nx + Ng - 1];
            }
        }
        
        // y-direction boundaries
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:total_size])
        for (int i = 0; i < Nx + 2; ++i) {
            // Bottom boundary (j=0)
            if (bc_y_lo == 2) { // Periodic
                u_ptr[i] = u_ptr[Ny * stride + i];
            } else if (bc_y_lo == 1) { // Neumann
                u_ptr[i] = u_ptr[Ng * stride + i];
            } else { // Dirichlet
                u_ptr[i] = 2.0 * dval - u_ptr[Ng * stride + i];
            }
            
            // Top boundary (j=Ny+1)
            if (bc_y_hi == 2) { // Periodic
                u_ptr[(Ny + Ng) * stride + i] = u_ptr[Ng * stride + i];
            } else if (bc_y_hi == 1) { // Neumann
                u_ptr[(Ny + Ng) * stride + i] = u_ptr[(Ny + Ng - 1) * stride + i];
            } else { // Dirichlet
                u_ptr[(Ny + Ng) * stride + i] = 2.0 * dval - u_ptr[(Ny + Ng - 1) * stride + i];
            }
        }
        
        return;
    }
#endif
    
    // CPU path
    // x-direction boundaries
    for (int j = 0; j < Ny + 2*Ng; ++j) {
        // Left boundary
        int i_ghost = 0;
        int i_interior = Ng;
        int i_periodic = Nx;
        
        switch (bc_x_lo_) {
            case PoissonBC::Periodic:
                grid.u(i_ghost, j) = grid.u(i_periodic, j);
                break;
            case PoissonBC::Neumann:
                grid.u(i_ghost, j) = grid.u(i_interior, j);
                break;
            case PoissonBC::Dirichlet:
                grid.u(i_ghost, j) = 2.0 * dirichlet_val_ - grid.u(i_interior, j);
                break;
        }
        
        // Right boundary
        i_ghost = Nx + Ng;
        i_interior = Nx + Ng - 1;
        i_periodic = Ng;
        
        switch (bc_x_hi_) {
            case PoissonBC::Periodic:
                grid.u(i_ghost, j) = grid.u(i_periodic, j);
                break;
            case PoissonBC::Neumann:
                grid.u(i_ghost, j) = grid.u(i_interior, j);
                break;
            case PoissonBC::Dirichlet:
                grid.u(i_ghost, j) = 2.0 * dirichlet_val_ - grid.u(i_interior, j);
                break;
        }
    }
    
    // y-direction boundaries
    for (int i = 0; i < Nx + 2*Ng; ++i) {
        // Bottom boundary
        int j_ghost = 0;
        int j_interior = Ng;
        
        switch (bc_y_lo_) {
            case PoissonBC::Neumann:
                grid.u(i, j_ghost) = grid.u(i, j_interior);
                break;
            case PoissonBC::Dirichlet:
                grid.u(i, j_ghost) = 2.0 * dirichlet_val_ - grid.u(i, j_interior);
                break;
            case PoissonBC::Periodic:
                grid.u(i, j_ghost) = grid.u(i, Ny);
                break;
        }
        
        // Top boundary
        j_ghost = Ny + Ng;
        j_interior = Ny + Ng - 1;
        
        switch (bc_y_hi_) {
            case PoissonBC::Neumann:
                grid.u(i, j_ghost) = grid.u(i, j_interior);
                break;
            case PoissonBC::Dirichlet:
                grid.u(i, j_ghost) = 2.0 * dirichlet_val_ - grid.u(i, j_interior);
                break;
            case PoissonBC::Periodic:
                grid.u(i, j_ghost) = grid.u(i, Ng);
                break;
        }
    }
    
    // Re-apply periodic BCs to fix corner ghost cells
    // When both directions have periodic BCs applied sequentially, corner values
    // can be inconsistent. Re-applying ensures all ghost cells are properly synchronized.
    const bool x_periodic = (bc_x_lo_ == PoissonBC::Periodic) && (bc_x_hi_ == PoissonBC::Periodic);
    const bool y_periodic = (bc_y_lo_ == PoissonBC::Periodic) && (bc_y_hi_ == PoissonBC::Periodic);
    
    if (x_periodic || y_periodic) {
        // Re-apply x-direction boundaries
        for (int j = 0; j < Ny + 2*Ng; ++j) {
            // Left boundary
            int i_ghost = 0;
            int i_interior = Ng;
            int i_periodic = Nx;
            
            switch (bc_x_lo_) {
                case PoissonBC::Periodic:
                    grid.u(i_ghost, j) = grid.u(i_periodic, j);
                    break;
                case PoissonBC::Neumann:
                    grid.u(i_ghost, j) = grid.u(i_interior, j);
                    break;
                case PoissonBC::Dirichlet:
                    grid.u(i_ghost, j) = 2.0 * dirichlet_val_ - grid.u(i_interior, j);
                    break;
            }
            
            // Right boundary
            i_ghost = Nx + Ng;
            i_interior = Nx + Ng - 1;
            i_periodic = Ng;
            
            switch (bc_x_hi_) {
                case PoissonBC::Periodic:
                    grid.u(i_ghost, j) = grid.u(i_periodic, j);
                    break;
                case PoissonBC::Neumann:
                    grid.u(i_ghost, j) = grid.u(i_interior, j);
                    break;
                case PoissonBC::Dirichlet:
                    grid.u(i_ghost, j) = 2.0 * dirichlet_val_ - grid.u(i_interior, j);
                    break;
            }
        }
        
        // Re-apply y-direction boundaries
        for (int i = 0; i < Nx + 2*Ng; ++i) {
            // Bottom boundary
            int j_ghost = 0;
            int j_interior = Ng;
            
            switch (bc_y_lo_) {
                case PoissonBC::Neumann:
                    grid.u(i, j_ghost) = grid.u(i, j_interior);
                    break;
                case PoissonBC::Dirichlet:
                    grid.u(i, j_ghost) = 2.0 * dirichlet_val_ - grid.u(i, j_interior);
                    break;
                case PoissonBC::Periodic:
                    grid.u(i, j_ghost) = grid.u(i, Ny);
                    break;
            }
            
            // Top boundary
            j_ghost = Ny + Ng;
            j_interior = Ny + Ng - 1;
            
            switch (bc_y_hi_) {
                case PoissonBC::Neumann:
                    grid.u(i, j_ghost) = grid.u(i, j_interior);
                    break;
                case PoissonBC::Dirichlet:
                    grid.u(i, j_ghost) = 2.0 * dirichlet_val_ - grid.u(i, j_interior);
                    break;
                case PoissonBC::Periodic:
                    grid.u(i, j_ghost) = grid.u(i, Ng);
                    break;
            }
        }
    }
}

void MultigridPoissonSolver::smooth(int level, int iterations, double omega) {
    // Red-Black Gauss-Seidel with SOR
    auto& grid = *levels_[level];
    const double dx2 = grid.dx * grid.dx;
    const double dy2 = grid.dy * grid.dy;
    const double coeff = 2.0 / dx2 + 2.0 / dy2;
    
    const int Ng = 1;
    const int Nx = grid.Nx;
    const int Ny = grid.Ny;
    
#ifdef USE_GPU_OFFLOAD
    // GPU path: red-black Gauss-Seidel on persistent device arrays
    // Optimized to avoid warp divergence by using strided loops (same as CPU path)
    if (gpu::should_use_gpu_path()) {
        const int stride = Nx + 2;
        const size_t total_size = level_sizes_[level];
        double* u_ptr = u_ptrs_[level];
        const double* f_ptr = f_ptrs_[level];
        
        for (int iter = 0; iter < iterations; ++iter) {
            // Red sweep (i + j even) - uniform loop with parity check
            #pragma omp target teams distribute parallel for collapse(2) \
                map(present: u_ptr[0:total_size], f_ptr[0:total_size])
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
            
            // Black sweep (i + j odd) - uniform loop with parity check
            #pragma omp target teams distribute parallel for collapse(2) \
                map(present: u_ptr[0:total_size], f_ptr[0:total_size])
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
        
        // Apply boundary conditions once after all smoothing iterations
        apply_bc(level);
        
        return;
    }
#endif
    
    // CPU path
    for (int iter = 0; iter < iterations; ++iter) {
        // Red sweep (i + j even)
        for (int j = Ng; j < Ny + Ng; ++j) {
            int start = Ng + ((Ng + j) % 2);
            for (int i = start; i < Nx + Ng; i += 2) {
                double u_old = grid.u(i, j);
                double u_gs = ((grid.u(i+1, j) + grid.u(i-1, j)) / dx2
                             + (grid.u(i, j+1) + grid.u(i, j-1)) / dy2
                             - grid.f(i, j)) / coeff;
                grid.u(i, j) = (1.0 - omega) * u_old + omega * u_gs;
            }
        }
        
        // Black sweep (i + j odd)
        for (int j = Ng; j < Ny + Ng; ++j) {
            int start = Ng + ((Ng + j + 1) % 2);
            for (int i = start; i < Nx + Ng; i += 2) {
                double u_old = grid.u(i, j);
                double u_gs = ((grid.u(i+1, j) + grid.u(i-1, j)) / dx2
                             + (grid.u(i, j+1) + grid.u(i, j-1)) / dy2
                             - grid.f(i, j)) / coeff;
                grid.u(i, j) = (1.0 - omega) * u_old + omega * u_gs;
            }
        }
    }
    
    // Apply boundary conditions once after all smoothing iterations
    apply_bc(level);
}

void MultigridPoissonSolver::compute_residual(int level) {
    // r = f - L(u) where L is Laplacian operator
    auto& grid = *levels_[level];
    const double dx2 = grid.dx * grid.dx;
    const double dy2 = grid.dy * grid.dy;

    const int Ng = 1;
    const int Nx = grid.Nx;
    const int Ny = grid.Ny;

#ifdef USE_GPU_OFFLOAD
    // GPU path with persistent device arrays
    if (gpu::should_use_gpu_path()) {
        const int stride = Nx + 2;
        const size_t total_size = level_sizes_[level];
        const double* u_ptr = u_ptrs_[level];
        const double* f_ptr = f_ptrs_[level];
        double* r_ptr = r_ptrs_[level];

        #pragma omp target teams distribute parallel for collapse(2) \
            map(present: u_ptr[0:total_size], f_ptr[0:total_size], r_ptr[0:total_size])
        for (int j = Ng; j < Ny + Ng; ++j) {
            for (int i = Ng; i < Nx + Ng; ++i) {
                int idx = j * stride + i;
                double laplacian = (u_ptr[idx+1] - 2.0*u_ptr[idx] + u_ptr[idx-1]) / dx2
                                 + (u_ptr[idx+stride] - 2.0*u_ptr[idx] + u_ptr[idx-stride]) / dy2;
                r_ptr[idx] = f_ptr[idx] - laplacian;
            }
        }

        return;
    }
#endif
    
    // CPU path
    for (int j = Ng; j < Ny + Ng; ++j) {
        for (int i = Ng; i < Nx + Ng; ++i) {
            double laplacian = (grid.u(i+1, j) - 2.0*grid.u(i, j) + grid.u(i-1, j)) / dx2
                             + (grid.u(i, j+1) - 2.0*grid.u(i, j) + grid.u(i, j-1)) / dy2;
            grid.r(i, j) = grid.f(i, j) - laplacian;
        }
    }
}

void MultigridPoissonSolver::restrict_residual(int fine_level) {
    // Full-weighting restriction from fine to coarse grid
    auto& fine = *levels_[fine_level];
    auto& coarse = *levels_[fine_level + 1];
    
    const int Ng = 1;
    
#ifdef USE_GPU_OFFLOAD
    if (coarse.Nx >= 16 && coarse.Ny >= 16) {
        const int Nx_c = coarse.Nx;
        const int Ny_c = coarse.Ny;
        const int stride_f = fine.Nx + 2;
        const int stride_c = coarse.Nx + 2;
        const size_t size_f = level_sizes_[fine_level];
        const size_t size_c = level_sizes_[fine_level + 1];
        
        const double* r_fine = r_ptrs_[fine_level];
        double* f_coarse = f_ptrs_[fine_level + 1];
        
        #pragma omp target teams distribute parallel for collapse(2) \
            map(present: r_fine[0:size_f], f_coarse[0:size_c])
        for (int j_c = Ng; j_c < Ny_c + Ng; ++j_c) {
            for (int i_c = Ng; i_c < Nx_c + Ng; ++i_c) {
                int i_f = 2 * (i_c - Ng) + Ng;
                int j_f = 2 * (j_c - Ng) + Ng;
                int idx_c = j_c * stride_c + i_c;
                int idx_f = j_f * stride_f + i_f;
                
                // Full-weighting stencil
                f_coarse[idx_c] = 0.25 * r_fine[idx_f]
                                + 0.125 * (r_fine[idx_f-1] + r_fine[idx_f+1]
                                         + r_fine[idx_f-stride_f] + r_fine[idx_f+stride_f])
                                + 0.0625 * (r_fine[idx_f-1-stride_f] + r_fine[idx_f+1-stride_f]
                                          + r_fine[idx_f-1+stride_f] + r_fine[idx_f+1+stride_f]);
            }
        }
        return;
    }
#endif
    
    for (int j_c = Ng; j_c < coarse.Ny + Ng; ++j_c) {
        for (int i_c = Ng; i_c < coarse.Nx + Ng; ++i_c) {
            int i_f = 2 * (i_c - Ng) + Ng;
            int j_f = 2 * (j_c - Ng) + Ng;
            
            // Full-weighting stencil
            coarse.f(i_c, j_c) = 0.25 * fine.r(i_f, j_f)
                               + 0.125 * (fine.r(i_f-1, j_f) + fine.r(i_f+1, j_f)
                                        + fine.r(i_f, j_f-1) + fine.r(i_f, j_f+1))
                               + 0.0625 * (fine.r(i_f-1, j_f-1) + fine.r(i_f+1, j_f-1)
                                         + fine.r(i_f-1, j_f+1) + fine.r(i_f+1, j_f+1));
        }
    }
}

void MultigridPoissonSolver::prolongate_correction(int coarse_level) {
    // Bilinear interpolation from coarse to fine grid
    auto& coarse = *levels_[coarse_level];
    auto& fine = *levels_[coarse_level - 1];
    
    const int Ng = 1;
    
#ifdef USE_GPU_OFFLOAD
    if (fine.Nx >= 32 && fine.Ny >= 32) {
        const int Nx_c = coarse.Nx;
        const int Ny_c = coarse.Ny;
        const int Nx_f = fine.Nx;
        const int Ny_f = fine.Ny;
        const int stride_f = fine.Nx + 2;
        const int stride_c = coarse.Nx + 2;
        const size_t size_f = level_sizes_[coarse_level - 1];
        const size_t size_c = level_sizes_[coarse_level];
        
        const double* u_coarse = u_ptrs_[coarse_level];
        double* u_fine = u_ptrs_[coarse_level - 1];
        
        #pragma omp target teams distribute parallel for collapse(2) \
            map(present: u_coarse[0:size_c], u_fine[0:size_f])
        for (int j_c = Ng; j_c < Ny_c + Ng; ++j_c) {
            for (int i_c = Ng; i_c < Nx_c + Ng; ++i_c) {
                int i_f = 2 * (i_c - Ng) + Ng;
                int j_f = 2 * (j_c - Ng) + Ng;
                int idx_c = j_c * stride_c + i_c;
                int idx_f = j_f * stride_f + i_f;
                
                double val_c = u_coarse[idx_c];
                
                // Direct injection to coarse points
                u_fine[idx_f] += val_c;
                
                // Interpolate to fine points
                if (i_f + 1 < Nx_f + Ng) {
                    double val_east = u_coarse[idx_c + 1];
                    u_fine[idx_f + 1] += 0.5 * (val_c + val_east);
                }
                if (j_f + 1 < Ny_f + Ng) {
                    double val_north = u_coarse[idx_c + stride_c];
                    u_fine[idx_f + stride_f] += 0.5 * (val_c + val_north);
                }
                if (i_f + 1 < Nx_f + Ng && j_f + 1 < Ny_f + Ng) {
                    double val_east = u_coarse[idx_c + 1];
                    double val_north = u_coarse[idx_c + stride_c];
                    double val_ne = u_coarse[idx_c + 1 + stride_c];
                    u_fine[idx_f + 1 + stride_f] += 0.25 * (val_c + val_east + val_north + val_ne);
                }
            }
        }
        return;
    }
#endif
    
    for (int j_c = Ng; j_c < coarse.Ny + Ng; ++j_c) {
        for (int i_c = Ng; i_c < coarse.Nx + Ng; ++i_c) {
            int i_f = 2 * (i_c - Ng) + Ng;
            int j_f = 2 * (j_c - Ng) + Ng;
            
            // Direct injection to coarse points
            fine.u(i_f, j_f) += coarse.u(i_c, j_c);
            
            // Interpolate to fine points
            if (i_f + 1 < fine.Nx + Ng) {
                fine.u(i_f+1, j_f) += 0.5 * (coarse.u(i_c, j_c) + coarse.u(i_c+1, j_c));
            }
            if (j_f + 1 < fine.Ny + Ng) {
                fine.u(i_f, j_f+1) += 0.5 * (coarse.u(i_c, j_c) + coarse.u(i_c, j_c+1));
            }
            if (i_f + 1 < fine.Nx + Ng && j_f + 1 < fine.Ny + Ng) {
                fine.u(i_f+1, j_f+1) += 0.25 * (coarse.u(i_c, j_c) + coarse.u(i_c+1, j_c)
                                               + coarse.u(i_c, j_c+1) + coarse.u(i_c+1, j_c+1));
            }
        }
    }
}

void MultigridPoissonSolver::solve_coarsest(int iterations) {
    // Direct solve on coarsest grid using many SOR iterations
    int coarsest = levels_.size() - 1;
    smooth(coarsest, iterations, 1.8);
}

void MultigridPoissonSolver::vcycle(int level, int nu1, int nu2) {
    if (level == static_cast<int>(levels_.size()) - 1) {
        // Coarsest level - solve directly
        // Reduced from 100 to 40 iterations for better performance
        solve_coarsest(40);
        return;
    }
    
    // Pre-smoothing
    smooth(level, nu1, 1.8);
    
    // Compute residual
    compute_residual(level);
    
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
    for (int j = 0; j < coarse.Ny + 2*Ng; ++j) {
        for (int i = 0; i < coarse.Nx + 2*Ng; ++i) {
            coarse.u(i, j) = 0.0;
        }
    }
#endif
    
    // Recursive call to coarser level
    vcycle(level + 1, nu1, nu2);
    
    // Prolongate correction
    prolongate_correction(level + 1);
    
    // Apply boundary conditions
    apply_bc(level);
    
    // Post-smoothing (more iterations for better convergence)
    smooth(level, nu2, 1.8);
}

double MultigridPoissonSolver::compute_max_residual(int level) {
    compute_residual(level);
    
    auto& grid = *levels_[level];
    double max_res = 0.0;
    const int Ng = 1;
    
#ifdef USE_GPU_OFFLOAD
    if (grid.Nx >= 32 && grid.Ny >= 32) {
        // Compute max residual on GPU and return scalar to host
        const int Nx = grid.Nx;
        const int Ny = grid.Ny;
        const int stride = Nx + 2;
        const size_t total_size = level_sizes_[level];
        const double* r_ptr = r_ptrs_[level];

        // Reduction over interior cells only
        #pragma omp target teams distribute parallel for reduction(max:max_res) \
            map(present: r_ptr[0:total_size]) \
            map(tofrom: max_res) \
            firstprivate(Nx, Ny, stride, Ng)
        for (int idx = 0; idx < Nx * Ny; ++idx) {
            int i = idx % Nx + Ng;
            int j = idx / Nx + Ng;
            int ridx = j * stride + i;
            double v = std::abs(r_ptr[ridx]);
            if (v > max_res) max_res = v;
        }
    } else
#endif
    {
        for (int j = Ng; j < grid.Ny + Ng; ++j) {
            for (int i = Ng; i < grid.Nx + Ng; ++i) {
                max_res = std::max(max_res, std::abs(grid.r(i, j)));
            }
        }
    }
    
    return max_res;
}

void MultigridPoissonSolver::subtract_mean(int level) {
    auto& grid = *levels_[level];
    double sum = 0.0;
    int count = 0;
    const int Ng = 1;
    
#ifdef USE_GPU_OFFLOAD
    if (grid.Nx >= 32 && grid.Ny >= 32) {
        const size_t total_size = level_sizes_[level];
        double* u_ptr = u_ptrs_[level];
        const int Nx = grid.Nx;
        const int Ny = grid.Ny;
        const int stride = Nx + 2;
        
        // Compute sum on GPU
        #pragma omp target teams distribute parallel for reduction(+:sum) \
            map(present: u_ptr[0:total_size])
        for (int idx = 0; idx < Nx * Ny; ++idx) {
            int i = idx % Nx + Ng;
            int j = idx / Nx + Ng;
            sum += u_ptr[j * stride + i];
        }
        
        count = Nx * Ny;
        double mean = sum / count;
        
        // Subtract mean on GPU
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:total_size])
        for (int idx = 0; idx < Nx * Ny; ++idx) {
            int i = idx % Nx + Ng;
            int j = idx / Nx + Ng;
            u_ptr[j * stride + i] -= mean;
        }
    } else
#endif
    {
    for (int j = Ng; j < grid.Ny + Ng; ++j) {
        for (int i = Ng; i < grid.Nx + Ng; ++i) {
            sum += grid.u(i, j);
            ++count;
        }
    }
    
    double mean = sum / count;
    
    for (int j = Ng; j < grid.Ny + Ng; ++j) {
        for (int i = Ng; i < grid.Nx + Ng; ++i) {
            grid.u(i, j) -= mean;
            }
        }
    }
}

int MultigridPoissonSolver::solve(const ScalarField& rhs, ScalarField& p, const PoissonConfig& cfg) {
    
    // Copy RHS and initial guess to finest level (CPU side)
    auto& finest = *levels_[0];
    const int Ng = 1;
    
    for (int j = Ng; j < finest.Ny + Ng; ++j) {
        for (int i = Ng; i < finest.Nx + Ng; ++i) {
            finest.f(i, j) = rhs(i, j);
            finest.u(i, j) = p(i, j);
        }
    }
    
#ifdef USE_GPU_OFFLOAD
    // Upload to GPU once before all V-cycles
    assert(gpu_ready_ && "GPU must be initialized");
    sync_level_to_gpu(0);
#endif
    
    apply_bc(0);
    
    // Perform V-cycles until converged
    // Interpret cfg.max_iter as "max Poisson iterations per solve".
    // For multigrid, one V-cycle is one iteration, so cfg.max_iter directly
    // bounds the number of cycles. If cfg.max_iter <= 0, fall back to 10.
    const int max_cycles = (cfg.max_iter > 0) ? cfg.max_iter : 10;
    
    int cycle = 0;
    for (; cycle < max_cycles; ++cycle) {
        vcycle(0, 2, 2);  // Reduced from (3,3) to (2,2) for better performance
        
        // Check convergence after each cycle
        residual_ = compute_max_residual(0);
        
        if (residual_ < cfg.tol) {
            break;
        }
    }
    
    // Final residual
    residual_ = compute_max_residual(0);
    
    // Subtract mean for singular Poisson problems (no Dirichlet BCs)
    // Whenever all boundaries are Neumann or Periodic, the solution is defined up to a constant
    // and we must fix the nullspace by subtracting the mean
    bool has_dirichlet = (bc_x_lo_ == PoissonBC::Dirichlet || bc_x_hi_ == PoissonBC::Dirichlet ||
                          bc_y_lo_ == PoissonBC::Dirichlet || bc_y_hi_ == PoissonBC::Dirichlet);
    
    if (!has_dirichlet) {
        subtract_mean(0);
    }
    
#ifdef USE_GPU_OFFLOAD
    // Download from GPU once after all V-cycles
    assert(gpu_ready_ && "GPU must be initialized");
    sync_level_from_gpu(0);
#endif
    
    // Copy result back to output field (CPU side)
    for (int j = Ng; j < finest.Ny + Ng; ++j) {
        for (int i = Ng; i < finest.Nx + Ng; ++i) {
            p(i, j) = finest.u(i, j);
        }
    }
    
    return cycle + 1;
}

#ifdef USE_GPU_OFFLOAD
int MultigridPoissonSolver::solve_device(double* rhs_present, double* p_present, const PoissonConfig& cfg) {
    assert(gpu_ready_ && "GPU must be initialized in constructor");
    
    // Device-resident solve using Model 1 (host pointer + present mapping)
    // Parameters are host pointers that caller has already mapped via `target enter data`.
    // We use map(present: ...) to access the device copies without additional transfers.
    
    auto& finest = *levels_[0];
    const int Nx = finest.Nx;
    const int Ny = finest.Ny;
    const size_t total_size = (Nx + 2) * (Ny + 2);
    
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
    
    // Perform V-cycles until converged (all operations already on device)
    // Interpret cfg.max_iter as "max Poisson iterations per solve".
    // For multigrid, one V-cycle is one iteration, so cfg.max_iter directly
    // bounds the number of cycles. If cfg.max_iter <= 0, fall back to 10.
    const int max_cycles = (cfg.max_iter > 0) ? cfg.max_iter : 10;
    
    // Configurable residual monitoring interval (check every N cycles)
    // Only copy residual back to CPU when checking (reduces DtoH transfers)
    static const char* env_check_interval = std::getenv("NNCFD_POISSON_RESIDUAL_CHECK_INTERVAL");
    const int check_interval = env_check_interval ? std::atoi(env_check_interval) : 1;
    
    int cycle = 0;
    for (; cycle < max_cycles; ++cycle) {
        vcycle(0, 2, 2);  // Reduced from (3,3) to (2,2) for better performance
        
        // Only check convergence every 'check_interval' cycles (or on last cycle)
        bool should_check = (cycle % check_interval == 0) || (cycle == max_cycles - 1);
        
        if (should_check) {
            // Compute residual on GPU, then copy result to CPU for convergence check
            residual_ = compute_max_residual(0);
            
            if (residual_ < cfg.tol) {
                break;
            }
        }
    }
    
    // Final residual (always compute at end for diagnostics)
    residual_ = compute_max_residual(0);
    
    // Subtract mean for singular Poisson problems (no Dirichlet BCs)
    // Whenever all boundaries are Neumann or Periodic, the solution is defined up to a constant
    // and we must fix the nullspace by subtracting the mean
    bool has_dirichlet = (bc_x_lo_ == PoissonBC::Dirichlet || bc_x_hi_ == PoissonBC::Dirichlet ||
                          bc_y_lo_ == PoissonBC::Dirichlet || bc_y_hi_ == PoissonBC::Dirichlet);
    
    if (!has_dirichlet) {
        subtract_mean(0);
    }
    
    // Copy result from multigrid level-0 buffer back to caller's present-mapped pointer
    // This is device-to-device copy via present mappings (no host staging)
    #pragma omp target teams distribute parallel for \
        map(present: p_present[0:total_size], u_dev[0:total_size])
    for (size_t idx = 0; idx < total_size; ++idx) {
        p_present[idx] = u_dev[idx];
    }
    
    return cycle + 1;
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
    level_sizes_.resize(levels_.size());
    
    for (size_t lvl = 0; lvl < levels_.size(); ++lvl) {
        auto& grid = *levels_[lvl];
        const size_t total_size = (grid.Nx + 2) * (grid.Ny + 2);
        level_sizes_[lvl] = total_size;
        
        // Get pointers to CPU data
        u_ptrs_[lvl] = grid.u.data().data();
        f_ptrs_[lvl] = grid.f.data().data();
        r_ptrs_[lvl] = grid.r.data().data();
        
        // Allocate on device
        #pragma omp target enter data map(alloc: u_ptrs_[lvl][0:total_size])
        #pragma omp target enter data map(alloc: f_ptrs_[lvl][0:total_size])
        #pragma omp target enter data map(alloc: r_ptrs_[lvl][0:total_size])
    }
    
    // Verify mappings succeeded
    if (!u_ptrs_.empty() && !gpu::is_pointer_present(u_ptrs_[0])) {
        throw std::runtime_error("GPU mapping failed despite device availability");
    }
    
    gpu_ready_ = true;
}

void MultigridPoissonSolver::cleanup_gpu_buffers() {
    assert(gpu_ready_ && "GPU must be initialized");
    
    for (size_t lvl = 0; lvl < levels_.size(); ++lvl) {
        const size_t total_size = level_sizes_[lvl];
        
        #pragma omp target exit data map(delete: u_ptrs_[lvl][0:total_size])
        #pragma omp target exit data map(delete: f_ptrs_[lvl][0:total_size])
        #pragma omp target exit data map(delete: r_ptrs_[lvl][0:total_size])
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

