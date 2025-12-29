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
    // Keep z BCs at default (Periodic)
}

void PoissonSolver::set_bc(PoissonBC x_lo, PoissonBC x_hi,
                           PoissonBC y_lo, PoissonBC y_hi,
                           PoissonBC z_lo, PoissonBC z_hi) {
    bc_x_lo_ = x_lo;
    bc_x_hi_ = x_hi;
    bc_y_lo_ = y_lo;
    bc_y_hi_ = y_hi;
    bc_z_lo_ = z_lo;
    bc_z_hi_ = z_hi;
}

void PoissonSolver::apply_bc(ScalarField& p) {
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Ng = mesh_->Nghost;
    const bool is_2d = mesh_->is2D();

    // Apply BCs across full allocated z-extent for consistency
    // (in 2D, replicating across all z-planes ensures no uninitialized ghost data)
    const int k_start = 0;
    const int k_stop = mesh_->total_Nz();

    // x-direction boundaries
    for (int k = k_start; k < k_stop; ++k) {
        for (int j = 0; j < mesh_->total_Ny(); ++j) {
            // Left boundary (x_lo)
            for (int g = 0; g < Ng; ++g) {
                int i_ghost = g;
                int i_interior = Ng;
                int i_periodic = Nx + Ng - 1 - g;

                switch (bc_x_lo_) {
                    case PoissonBC::Periodic:
                        p(i_ghost, j, k) = p(i_periodic, j, k);
                        break;
                    case PoissonBC::Neumann:
                        p(i_ghost, j, k) = p(i_interior, j, k);
                        break;
                    case PoissonBC::Dirichlet:
                        p(i_ghost, j, k) = 2.0 * dirichlet_val_ - p(i_interior, j, k);
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
                        p(i_ghost, j, k) = p(i_periodic, j, k);
                        break;
                    case PoissonBC::Neumann:
                        p(i_ghost, j, k) = p(i_interior, j, k);
                        break;
                    case PoissonBC::Dirichlet:
                        p(i_ghost, j, k) = 2.0 * dirichlet_val_ - p(i_interior, j, k);
                        break;
                }
            }
        }
    }

    // y-direction boundaries
    for (int k = k_start; k < k_stop; ++k) {
        for (int i = 0; i < mesh_->total_Nx(); ++i) {
            // Bottom boundary (y_lo)
            for (int g = 0; g < Ng; ++g) {
                int j_ghost = g;
                int j_interior = Ng;
                int j_periodic = Ny + Ng - 1 - g;

                switch (bc_y_lo_) {
                    case PoissonBC::Periodic:
                        p(i, j_ghost, k) = p(i, j_periodic, k);
                        break;
                    case PoissonBC::Neumann:
                        p(i, j_ghost, k) = p(i, j_interior, k);
                        break;
                    case PoissonBC::Dirichlet:
                        p(i, j_ghost, k) = 2.0 * dirichlet_val_ - p(i, j_interior, k);
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
                        p(i, j_ghost, k) = p(i, j_periodic, k);
                        break;
                    case PoissonBC::Neumann:
                        p(i, j_ghost, k) = p(i, j_interior, k);
                        break;
                    case PoissonBC::Dirichlet:
                        p(i, j_ghost, k) = 2.0 * dirichlet_val_ - p(i, j_interior, k);
                        break;
                }
            }
        }
    }

    // z-direction boundaries (3D only)
    if (!is_2d) {
        for (int j = 0; j < mesh_->total_Ny(); ++j) {
            for (int i = 0; i < mesh_->total_Nx(); ++i) {
                // Front boundary (z_lo)
                for (int g = 0; g < Ng; ++g) {
                    int k_ghost = g;
                    int k_interior = Ng;
                    int k_periodic = Nz + Ng - 1 - g;

                    switch (bc_z_lo_) {
                        case PoissonBC::Periodic:
                            p(i, j, k_ghost) = p(i, j, k_periodic);
                            break;
                        case PoissonBC::Neumann:
                            p(i, j, k_ghost) = p(i, j, k_interior);
                            break;
                        case PoissonBC::Dirichlet:
                            p(i, j, k_ghost) = 2.0 * dirichlet_val_ - p(i, j, k_interior);
                            break;
                    }
                }

                // Back boundary (z_hi)
                for (int g = 0; g < Ng; ++g) {
                    int k_ghost = Nz + Ng + g;
                    int k_interior = Nz + Ng - 1;
                    int k_periodic = Ng + g;

                    switch (bc_z_hi_) {
                        case PoissonBC::Periodic:
                            p(i, j, k_ghost) = p(i, j, k_periodic);
                            break;
                        case PoissonBC::Neumann:
                            p(i, j, k_ghost) = p(i, j, k_interior);
                            break;
                        case PoissonBC::Dirichlet:
                            p(i, j, k_ghost) = 2.0 * dirichlet_val_ - p(i, j, k_interior);
                            break;
                    }
                }
            }
        }
    }
}

double PoissonSolver::compute_residual(const ScalarField& rhs, const ScalarField& p) {
    const double dx = mesh_->dx;
    const double dy = mesh_->dy;
    const double dz = mesh_->dz;
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double dz2 = dz * dz;
    const bool is_2d = mesh_->is2D();

    double max_res = 0.0;

    // For 2D, only process k=0 plane (2D data is stored at k=0 by design)
    const int k_start = is_2d ? 0 : mesh_->k_begin();
    const int k_stop = is_2d ? 1 : mesh_->k_end();

    for (int k = k_start; k < k_stop; ++k) {
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                // Laplacian: 5-point stencil for 2D, 7-point for 3D
                double laplacian = (p(i+1, j, k) - 2.0*p(i, j, k) + p(i-1, j, k)) / dx2
                                 + (p(i, j+1, k) - 2.0*p(i, j, k) + p(i, j-1, k)) / dy2;
                if (!is_2d) {
                    laplacian += (p(i, j, k+1) - 2.0*p(i, j, k) + p(i, j, k-1)) / dz2;
                }

                double res = std::abs(laplacian - rhs(i, j, k));
                max_res = std::max(max_res, res);
            }
        }
    }

    return max_res;
}

void PoissonSolver::sor_iteration(const ScalarField& rhs, ScalarField& p, double omega) {
    const double dx = mesh_->dx;
    const double dy = mesh_->dy;
    const double dz = mesh_->dz;
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double dz2 = dz * dz;
    const bool is_2d = mesh_->is2D();

    // Diagonal coefficient: 2D uses 4 neighbors, 3D uses 6 neighbors
    double coeff = 2.0 / dx2 + 2.0 / dy2;
    if (!is_2d) {
        coeff += 2.0 / dz2;
    }

    // For 2D, only process k=0 plane (2D data is stored at k=0 by design)
    const int k_start = is_2d ? 0 : mesh_->k_begin();
    const int k_stop = is_2d ? 1 : mesh_->k_end();

    for (int k = k_start; k < k_stop; ++k) {
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                double p_old = p(i, j, k);
                double p_gs = ((p(i+1, j, k) + p(i-1, j, k)) / dx2
                             + (p(i, j+1, k) + p(i, j-1, k)) / dy2
                             - rhs(i, j, k));
                if (!is_2d) {
                    p_gs += (p(i, j, k+1) + p(i, j, k-1)) / dz2;
                }
                p_gs /= coeff;
                p(i, j, k) = (1.0 - omega) * p_old + omega * p_gs;
            }
        }
    }
}

void PoissonSolver::sor_rb_iteration(const ScalarField& rhs, ScalarField& p, double omega) {
    const double dx = mesh_->dx;
    const double dy = mesh_->dy;
    const double dz = mesh_->dz;
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double dz2 = dz * dz;
    const bool is_2d = mesh_->is2D();

    // Diagonal coefficient: 2D uses 4 neighbors, 3D uses 6 neighbors
    double coeff = 2.0 / dx2 + 2.0 / dy2;
    if (!is_2d) {
        coeff += 2.0 / dz2;
    }

    // For 2D, only process k=0 plane (2D data is stored at k=0 by design)
    const int k_start = is_2d ? 0 : mesh_->k_begin();
    const int k_stop = is_2d ? 1 : mesh_->k_end();

    // Red sweep (i + j + k even)
    for (int k = k_start; k < k_stop; ++k) {
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            int start = mesh_->i_begin() + ((mesh_->i_begin() + j + k) % 2);
            for (int i = start; i < mesh_->i_end(); i += 2) {
                double p_old = p(i, j, k);
                double p_gs = ((p(i+1, j, k) + p(i-1, j, k)) / dx2
                             + (p(i, j+1, k) + p(i, j-1, k)) / dy2
                             - rhs(i, j, k));
                if (!is_2d) {
                    p_gs += (p(i, j, k+1) + p(i, j, k-1)) / dz2;
                }
                p_gs /= coeff;
                p(i, j, k) = (1.0 - omega) * p_old + omega * p_gs;
            }
        }
    }

    apply_bc(p);

    // Black sweep (i + j + k odd)
    for (int k = k_start; k < k_stop; ++k) {
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            int start = mesh_->i_begin() + ((mesh_->i_begin() + j + k + 1) % 2);
            for (int i = start; i < mesh_->i_end(); i += 2) {
                double p_old = p(i, j, k);
                double p_gs = ((p(i+1, j, k) + p(i-1, j, k)) / dx2
                             + (p(i, j+1, k) + p(i, j-1, k)) / dy2
                             - rhs(i, j, k));
                if (!is_2d) {
                    p_gs += (p(i, j, k+1) + p(i, j, k-1)) / dz2;
                }
                p_gs /= coeff;
                p(i, j, k) = (1.0 - omega) * p_old + omega * p_gs;
            }
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
    // An axis is "ok" for nullspace if both boundaries are periodic or both are Neumann
    const bool is_2d = mesh_->is2D();

    auto axis_ok = [](PoissonBC lo, PoissonBC hi) {
        return (lo == PoissonBC::Periodic && hi == PoissonBC::Periodic) ||
               (lo == PoissonBC::Neumann && hi == PoissonBC::Neumann);
    };

    const bool needs_mean_subtraction =
        axis_ok(bc_x_lo_, bc_x_hi_) &&
        axis_ok(bc_y_lo_, bc_y_hi_) &&
        (is_2d || axis_ok(bc_z_lo_, bc_z_hi_));

    if (needs_mean_subtraction) {
        // Subtract mean
        double sum = 0.0;
        int count = 0;
        // For 2D, only process k=0 plane (2D data is stored at k=0 by design)
        const int k_start = is_2d ? 0 : mesh_->k_begin();
        const int k_stop = is_2d ? 1 : mesh_->k_end();

        for (int k = k_start; k < k_stop; ++k) {
            for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                    sum += p(i, j, k);
                    ++count;
                }
            }
        }
        double mean = sum / count;
        for (int k = k_start; k < k_stop; ++k) {
            for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                    p(i, j, k) -= mean;
                }
            }
        }
        apply_bc(p);
    }
    
    return iter + 1;
}

} // namespace nncfd

