#include "poisson_solver.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace nncfd {

PoissonSolver::PoissonSolver(const Mesh& mesh) : mesh_(&mesh) {}

void PoissonSolver::set_bc(PoissonBC x_lo, PoissonBC x_hi,
                           PoissonBC y_lo, PoissonBC y_hi) {
    bc_x_lo_ = x_lo;
    bc_x_hi_ = x_hi;
    bc_y_lo_ = y_lo;
    bc_y_hi_ = y_hi;
}

void PoissonSolver::apply_bc(ScalarField& p) {
    int Nx = mesh_->Nx;
    int Ny = mesh_->Ny;
    int Ng = mesh_->Nghost;
    
    // x-direction boundaries
    for (int j = 0; j < mesh_->total_Ny(); ++j) {
        // Left boundary (x_lo)
        for (int g = 0; g < Ng; ++g) {
            int i_ghost = g;
            int i_interior = Ng;
            int i_periodic = Nx + Ng - 1 - g;
            
            switch (bc_x_lo_) {
                case PoissonBC::Periodic:
                    p(i_ghost, j) = p(i_periodic, j);
                    break;
                case PoissonBC::Neumann:
                    p(i_ghost, j) = p(i_interior, j);
                    break;
                case PoissonBC::Dirichlet:
                    p(i_ghost, j) = 2.0 * dirichlet_val_ - p(i_interior, j);
                    break;
            }
        }
        
        // Right boundary (x_hi)
        for (int g = 0; g < Ng; ++g) {
            int i_ghost = Nx + Ng + g;
            int i_interior = Nx + Ng - 1;
            int i_periodic = Ng + g;
            
            switch (bc_x_hi_) {
                case PoissonBC::Periodic:
                    p(i_ghost, j) = p(i_periodic, j);
                    break;
                case PoissonBC::Neumann:
                    p(i_ghost, j) = p(i_interior, j);
                    break;
                case PoissonBC::Dirichlet:
                    p(i_ghost, j) = 2.0 * dirichlet_val_ - p(i_interior, j);
                    break;
            }
        }
    }
    
    // y-direction boundaries
    for (int i = 0; i < mesh_->total_Nx(); ++i) {
        // Bottom boundary (y_lo)
        for (int g = 0; g < Ng; ++g) {
            int j_ghost = g;
            int j_interior = Ng;
            int j_periodic = Ny + Ng - 1 - g;
            
            switch (bc_y_lo_) {
                case PoissonBC::Periodic:
                    p(i, j_ghost) = p(i, j_periodic);
                    break;
                case PoissonBC::Neumann:
                    p(i, j_ghost) = p(i, j_interior);
                    break;
                case PoissonBC::Dirichlet:
                    p(i, j_ghost) = 2.0 * dirichlet_val_ - p(i, j_interior);
                    break;
            }
        }
        
        // Top boundary (y_hi)
        for (int g = 0; g < Ng; ++g) {
            int j_ghost = Ny + Ng + g;
            int j_interior = Ny + Ng - 1;
            int j_periodic = Ng + g;
            
            switch (bc_y_hi_) {
                case PoissonBC::Periodic:
                    p(i, j_ghost) = p(i, j_periodic);
                    break;
                case PoissonBC::Neumann:
                    p(i, j_ghost) = p(i, j_interior);
                    break;
                case PoissonBC::Dirichlet:
                    p(i, j_ghost) = 2.0 * dirichlet_val_ - p(i, j_interior);
                    break;
            }
        }
    }
}

double PoissonSolver::compute_residual(const ScalarField& rhs, const ScalarField& p) {
    double dx = mesh_->dx;
    double dy = mesh_->dy;
    double dx2 = dx * dx;
    double dy2 = dy * dy;
    
    double max_res = 0.0;
    
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            // Laplacian using 5-point stencil
            double laplacian = (p(i+1, j) - 2.0*p(i, j) + p(i-1, j)) / dx2
                             + (p(i, j+1) - 2.0*p(i, j) + p(i, j-1)) / dy2;
            
            double res = std::abs(laplacian - rhs(i, j));
            max_res = std::max(max_res, res);
        }
    }
    
    return max_res;
}

void PoissonSolver::sor_iteration(const ScalarField& rhs, ScalarField& p, double omega) {
    double dx = mesh_->dx;
    double dy = mesh_->dy;
    double dx2 = dx * dx;
    double dy2 = dy * dy;
    
    double coeff = 2.0 / dx2 + 2.0 / dy2;
    
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            double p_old = p(i, j);
            double p_gs = ((p(i+1, j) + p(i-1, j)) / dx2 
                         + (p(i, j+1) + p(i, j-1)) / dy2
                         - rhs(i, j)) / coeff;
            p(i, j) = (1.0 - omega) * p_old + omega * p_gs;
        }
    }
}

void PoissonSolver::sor_rb_iteration(const ScalarField& rhs, ScalarField& p, double omega) {
    double dx = mesh_->dx;
    double dy = mesh_->dy;
    double dx2 = dx * dx;
    double dy2 = dy * dy;
    
    double coeff = 2.0 / dx2 + 2.0 / dy2;
    
    // Red sweep (i + j even)
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        int start = mesh_->i_begin() + ((mesh_->i_begin() + j) % 2);
        for (int i = start; i < mesh_->i_end(); i += 2) {
            double p_old = p(i, j);
            double p_gs = ((p(i+1, j) + p(i-1, j)) / dx2 
                         + (p(i, j+1) + p(i, j-1)) / dy2
                         - rhs(i, j)) / coeff;
            p(i, j) = (1.0 - omega) * p_old + omega * p_gs;
        }
    }
    
    apply_bc(p);
    
    // Black sweep (i + j odd)
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        int start = mesh_->i_begin() + ((mesh_->i_begin() + j + 1) % 2);
        for (int i = start; i < mesh_->i_end(); i += 2) {
            double p_old = p(i, j);
            double p_gs = ((p(i+1, j) + p(i-1, j)) / dx2 
                         + (p(i, j+1) + p(i, j-1)) / dy2
                         - rhs(i, j)) / coeff;
            p(i, j) = (1.0 - omega) * p_old + omega * p_gs;
        }
    }
}

int PoissonSolver::solve(const ScalarField& rhs, ScalarField& p, const PoissonConfig& cfg) {
    apply_bc(p);
    
    int iter = 0;
    for (; iter < cfg.max_iter; ++iter) {
        sor_rb_iteration(rhs, p, cfg.omega);
        apply_bc(p);
        
        if ((iter + 1) % 100 == 0 || iter == 0) {
            residual_ = compute_residual(rhs, p);
            if (cfg.verbose && (iter + 1) % 1000 == 0) {
                std::cout << "Poisson iter " << iter + 1 
                          << ", residual = " << residual_ << "\n";
            }
            if (residual_ < cfg.tol) {
                break;
            }
        }
    }
    
    // Final residual check
    residual_ = compute_residual(rhs, p);
    
    // For pure Neumann or fully periodic problems, subtract mean to ensure unique solution
    bool is_fully_periodic = (bc_x_lo_ == PoissonBC::Periodic && bc_x_hi_ == PoissonBC::Periodic &&
                              bc_y_lo_ == PoissonBC::Periodic && bc_y_hi_ == PoissonBC::Periodic);
    bool is_pure_neumann = (bc_x_lo_ == PoissonBC::Neumann && bc_x_hi_ == PoissonBC::Neumann &&
                            bc_y_lo_ == PoissonBC::Neumann && bc_y_hi_ == PoissonBC::Neumann);
    bool is_mixed_periodic_neumann = (bc_x_lo_ == PoissonBC::Periodic && bc_x_hi_ == PoissonBC::Periodic &&
                                      bc_y_lo_ == PoissonBC::Neumann && bc_y_hi_ == PoissonBC::Neumann);
    
    if (is_fully_periodic || is_pure_neumann || is_mixed_periodic_neumann) {
        // Subtract mean
        double sum = 0.0;
        int count = 0;
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                sum += p(i, j);
                ++count;
            }
        }
        double mean = sum / count;
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                p(i, j) -= mean;
            }
        }
        apply_bc(p);
    }
    
    return iter + 1;
}

int PoissonSolver::solve_variable(const ScalarField& alpha, const ScalarField& rhs, 
                                  ScalarField& p, const PoissonConfig& cfg) {
    // Solve: div(alpha * grad(p)) = rhs
    // Using SOR with variable coefficient
    
    apply_bc(p);
    
    double dx = mesh_->dx;
    double dy = mesh_->dy;
    
    int iter = 0;
    for (; iter < cfg.max_iter; ++iter) {
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                // Face-averaged alpha values
                double alpha_e = 0.5 * (alpha(i, j) + alpha(i+1, j));
                double alpha_w = 0.5 * (alpha(i, j) + alpha(i-1, j));
                double alpha_n = 0.5 * (alpha(i, j) + alpha(i, j+1));
                double alpha_s = 0.5 * (alpha(i, j) + alpha(i, j-1));
                
                double ae = alpha_e / (dx * dx);
                double aw = alpha_w / (dx * dx);
                double an = alpha_n / (dy * dy);
                double as = alpha_s / (dy * dy);
                double ap = ae + aw + an + as;
                
                double p_old = p(i, j);
                double p_gs = (ae * p(i+1, j) + aw * p(i-1, j)
                             + an * p(i, j+1) + as * p(i, j-1)
                             - rhs(i, j)) / ap;
                
                p(i, j) = (1.0 - cfg.omega) * p_old + cfg.omega * p_gs;
            }
        }
        apply_bc(p);
        
        if ((iter + 1) % 100 == 0) {
            residual_ = compute_residual(rhs, p);  // Approximate check
            if (residual_ < cfg.tol) {
                break;
            }
        }
    }
    
    return iter + 1;
}

} // namespace nncfd

