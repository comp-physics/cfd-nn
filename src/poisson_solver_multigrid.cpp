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
    // GPU path: red-black Gauss-Seidel on GPU
    if (omp_get_num_devices() > 0 && Nx >= 32 && Ny >= 32) {
        const int stride = Nx + 2;
        const int total_size = stride * (Ny + 2);
        double* u_ptr = grid.u.data().data();
        const double* f_ptr = grid.f.data().data();
        
        for (int iter = 0; iter < iterations; ++iter) {
            // Red sweep (i + j even)
            #pragma omp target teams distribute parallel for collapse(2) \
                map(tofrom: u_ptr[0:total_size]) \
                map(to: f_ptr[0:total_size])
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
            
            apply_bc(level);
            
            // Black sweep (i + j odd)
            #pragma omp target teams distribute parallel for collapse(2) \
                map(tofrom: u_ptr[0:total_size]) \
                map(to: f_ptr[0:total_size])
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
            
            apply_bc(level);
        }
        
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
        
        apply_bc(level);
        
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
        
        apply_bc(level);
    }
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
    // GPU path
    if (omp_get_num_devices() > 0 && Nx >= 32 && Ny >= 32) {
        const int stride = Nx + 2;
        const int total_size = stride * (Ny + 2);
        const double* u_ptr = grid.u.data().data();
        const double* f_ptr = grid.f.data().data();
        double* r_ptr = grid.r.data().data();
        
        #pragma omp target teams distribute parallel for collapse(2) \
            map(to: u_ptr[0:total_size], f_ptr[0:total_size]) \
            map(from: r_ptr[0:total_size])
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
    for (int j = 0; j < coarse.Ny + 2*Ng; ++j) {
        for (int i = 0; i < coarse.Nx + 2*Ng; ++i) {
            coarse.u(i, j) = 0.0;
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
    
    for (int j = Ng; j < grid.Ny + Ng; ++j) {
        for (int i = Ng; i < grid.Nx + Ng; ++i) {
            max_res = std::max(max_res, std::abs(grid.r(i, j)));
        }
    }
    
    return max_res;
}

void MultigridPoissonSolver::subtract_mean(int level) {
    auto& grid = *levels_[level];
    double sum = 0.0;
    int count = 0;
    const int Ng = 1;
    
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

int MultigridPoissonSolver::solve(const ScalarField& rhs, ScalarField& p, const PoissonConfig& cfg) {
    // Copy RHS and initial guess to finest level
    auto& finest = *levels_[0];
    const int Ng = 1;
    
    for (int j = Ng; j < finest.Ny + Ng; ++j) {
        for (int i = Ng; i < finest.Nx + Ng; ++i) {
            finest.f(i, j) = rhs(i, j);
            finest.u(i, j) = p(i, j);
        }
    }
    
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
    
    // Subtract mean for pure Neumann/periodic problems
    if ((bc_x_lo_ == PoissonBC::Periodic && bc_y_lo_ == PoissonBC::Neumann && bc_y_hi_ == PoissonBC::Neumann) ||
        (bc_x_lo_ == PoissonBC::Neumann && bc_x_hi_ == PoissonBC::Neumann &&
         bc_y_lo_ == PoissonBC::Neumann && bc_y_hi_ == PoissonBC::Neumann)) {
        subtract_mean(0);
    }
    
    // Copy result back
    for (int j = Ng; j < finest.Ny + Ng; ++j) {
        for (int i = Ng; i < finest.Nx + Ng; ++i) {
            p(i, j) = finest.u(i, j);
        }
    }
    
    return cycle + 1;
}

} // namespace nncfd

