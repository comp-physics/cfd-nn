#include "poisson_solver_multigrid.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

namespace nncfd {

MultigridPoissonSolver::MultigridPoissonSolver(const Mesh& mesh) : mesh_(&mesh) {
    create_hierarchy();
    
    // Note: GPU buffers are initialized lazily on first solve() call
    // because OpenMP runtime may not be ready during construction
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
    if (gpu_ready_ && Nx >= 16 && Ny >= 16) {
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
    if (gpu_ready_ && Nx >= 32 && Ny >= 32) {
        const int stride = Nx + 2;
        const size_t total_size = level_sizes_[level];
        double* u_ptr = u_ptrs_[level];
        const double* f_ptr = f_ptrs_[level];
        
        for (int iter = 0; iter < iterations; ++iter) {
            // Red sweep (i + j even)
            #pragma omp target teams distribute parallel for collapse(2) \
                map(present: u_ptr[0:total_size], f_ptr[0:total_size])
            for (int j = Ng; j < Ny + Ng; ++j) {
                for (int i = Ng; i < Nx + Ng; ++i) {
                    // Only update red points (i + j even)
                    if ((i + j) % 2 == 0) {
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
            #pragma omp target teams distribute parallel for collapse(2) \
                map(present: u_ptr[0:total_size], f_ptr[0:total_size])
            for (int j = Ng; j < Ny + Ng; ++j) {
                for (int i = Ng; i < Nx + Ng; ++i) {
                    // Only update black points (i + j odd)
                    if ((i + j) % 2 == 1) {
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
    if (gpu_ready_ && Nx >= 32 && Ny >= 32) {
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
    if (gpu_ready_ && coarse.Nx >= 16 && coarse.Ny >= 16) {
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
    if (gpu_ready_ && fine.Nx >= 32 && fine.Ny >= 32) {
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
        // Coarsest level - solve directly with more iterations
        solve_coarsest(100);
        return;
    }
    
    // Pre-smoothing (more iterations for better convergence)
    smooth(level, nu1, 1.8);
    
    // Compute residual
    compute_residual(level);
    
    // Restrict to coarse grid
    restrict_residual(level);
    
    // Zero coarse grid solution
    auto& coarse = *levels_[level + 1];
    const int Ng = 1;
    
#ifdef USE_GPU_OFFLOAD
    if (gpu_ready_) {
        const size_t size_c = level_sizes_[level + 1];
        double* u_coarse = u_ptrs_[level + 1];
        
        #pragma omp target teams distribute parallel for \
            map(present: u_coarse[0:size_c])
        for (int idx = 0; idx < (int)size_c; ++idx) {
            u_coarse[idx] = 0.0;
        }
    } else
#endif
    {
    for (int j = 0; j < coarse.Ny + 2*Ng; ++j) {
        for (int i = 0; i < coarse.Nx + 2*Ng; ++i) {
            coarse.u(i, j) = 0.0;
            }
        }
    }
    
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
    if (gpu_ready_ && grid.Nx >= 32 && grid.Ny >= 32) {
        const size_t total_size = level_sizes_[level];
        const double* r_ptr = r_ptrs_[level];
        const int Nx = grid.Nx;
        const int Ny = grid.Ny;
        
        #pragma omp target teams distribute parallel for reduction(max:max_res) \
            map(present: r_ptr[0:total_size])
        for (int idx = 0; idx < Nx * Ny; ++idx) {
            int i = idx % Nx + Ng;
            int j = idx / Nx + Ng;
            int stride = Nx + 2;
            int cell_idx = j * stride + i;
            double abs_r = r_ptr[cell_idx];
            if (abs_r < 0.0) abs_r = -abs_r;
            if (abs_r > max_res) max_res = abs_r;
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
    if (gpu_ready_ && grid.Nx >= 32 && grid.Ny >= 32) {
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
#ifdef USE_GPU_OFFLOAD
    // Lazy GPU initialization on first solve
    // Only try once - if gpu_ready_ is false after init, it means no GPU available
    static bool init_attempted = false;
    if (!init_attempted) {
        initialize_gpu_buffers();
        init_attempted = true;
    }
#endif
    
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
    if (gpu_ready_) {
        sync_level_to_gpu(0);
    }
#endif
    
    apply_bc(0);
    
    // Perform V-cycles until converged
    // Multigrid should converge in 3-10 cycles for well-conditioned problems
    // But we allow more cycles (cfg.max_iter / 100) to handle difficult cases
    // A V-cycle is roughly equivalent to 100 SOR iterations
    const int max_cycles = std::max(10, cfg.max_iter / 100);
    
    int cycle = 0;
    for (; cycle < max_cycles; ++cycle) {
        vcycle(0, 3, 3);  // More smoothing iterations per cycle
        
        // Check convergence after each cycle
        residual_ = compute_max_residual(0);
        
        if (cfg.verbose && (cycle + 1) % 2 == 0) {
            std::cout << "MG cycle " << cycle + 1 
                      << ", residual = " << residual_ << "\n";
        }
        
        if (residual_ < cfg.tol) {
            break;
        }
    }
    
    // Final residual
    residual_ = compute_max_residual(0);
    
    // Subtract mean for pure Neumann/periodic problems (singular Poisson)
    bool is_fully_periodic = (bc_x_lo_ == PoissonBC::Periodic && bc_x_hi_ == PoissonBC::Periodic &&
                              bc_y_lo_ == PoissonBC::Periodic && bc_y_hi_ == PoissonBC::Periodic);
    bool is_pure_neumann = (bc_x_lo_ == PoissonBC::Neumann && bc_x_hi_ == PoissonBC::Neumann &&
                            bc_y_lo_ == PoissonBC::Neumann && bc_y_hi_ == PoissonBC::Neumann);
    bool is_mixed_periodic_neumann = (bc_x_lo_ == PoissonBC::Periodic && bc_x_hi_ == PoissonBC::Periodic &&
                                      bc_y_lo_ == PoissonBC::Neumann && bc_y_hi_ == PoissonBC::Neumann);
    
    if (is_fully_periodic || is_pure_neumann || is_mixed_periodic_neumann) {
        subtract_mean(0);
    }
    
#ifdef USE_GPU_OFFLOAD
    // Download from GPU once after all V-cycles
    if (gpu_ready_) {
        sync_level_from_gpu(0);
    }
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
void MultigridPoissonSolver::initialize_gpu_buffers() {
    int num_devices = omp_get_num_devices();
    if (num_devices == 0) {
        gpu_ready_ = false;
        std::cout << "[MultigridPoisson] No GPU devices found, using CPU path\n";
        return;
    }
    
    std::cout << "[MultigridPoisson] Initializing GPU buffers for " << levels_.size() 
              << " levels, finest grid: " << levels_[0]->Nx << "x" << levels_[0]->Ny << "\n";
    
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
    
    gpu_ready_ = true;
    std::cout << "[MultigridPoisson] GPU buffers allocated successfully\n";
}

void MultigridPoissonSolver::cleanup_gpu_buffers() {
    if (!gpu_ready_) return;
    
    for (size_t lvl = 0; lvl < levels_.size(); ++lvl) {
        const size_t total_size = level_sizes_[lvl];
        
        #pragma omp target exit data map(delete: u_ptrs_[lvl][0:total_size])
        #pragma omp target exit data map(delete: f_ptrs_[lvl][0:total_size])
        #pragma omp target exit data map(delete: r_ptrs_[lvl][0:total_size])
    }
    
    gpu_ready_ = false;
}

void MultigridPoissonSolver::sync_level_to_gpu(int level) {
    if (!gpu_ready_) return;
    const size_t total_size = level_sizes_[level];
    
    #pragma omp target update to(u_ptrs_[level][0:total_size])
    #pragma omp target update to(f_ptrs_[level][0:total_size])
    #pragma omp target update to(r_ptrs_[level][0:total_size])
}

void MultigridPoissonSolver::sync_level_from_gpu(int level) {
    if (!gpu_ready_) return;
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

