#include "solver.hpp"
#include "timing.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>

namespace nncfd {

RANSSolver::RANSSolver(const Mesh& mesh, const Config& config)
    : mesh_(&mesh)
    , config_(config)
    , velocity_(mesh)
    , velocity_star_(mesh)
    , pressure_(mesh)
    , pressure_correction_(mesh)
    , nu_t_(mesh)
    , k_(mesh)
    , omega_(mesh)
    , tau_ij_(mesh)
    , rhs_poisson_(mesh)
    , div_velocity_(mesh)
    , poisson_solver_(mesh)
    , current_dt_(config.dt)
{
    // Set up Poisson solver BCs (periodic in x, Neumann in y for channel)
    poisson_solver_.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                           PoissonBC::Neumann, PoissonBC::Neumann);
}

void RANSSolver::set_turbulence_model(std::unique_ptr<TurbulenceModel> model) {
    turb_model_ = std::move(model);
    if (turb_model_) {
        turb_model_->set_nu(config_.nu);
    }
}

void RANSSolver::set_velocity_bc(const VelocityBC& bc) {
    velocity_bc_ = bc;
    
    // Update Poisson BCs based on velocity BCs
    PoissonBC p_x_lo = (bc.x_lo == VelocityBC::Periodic) ? PoissonBC::Periodic : PoissonBC::Neumann;
    PoissonBC p_x_hi = (bc.x_hi == VelocityBC::Periodic) ? PoissonBC::Periodic : PoissonBC::Neumann;
    PoissonBC p_y_lo = PoissonBC::Neumann;  // Always Neumann for pressure at walls
    PoissonBC p_y_hi = PoissonBC::Neumann;
    
    poisson_solver_.set_bc(p_x_lo, p_x_hi, p_y_lo, p_y_hi);
}

void RANSSolver::set_body_force(double fx, double fy) {
    fx_ = fx;
    fy_ = fy;
}

void RANSSolver::initialize(const VectorField& initial_velocity) {
    velocity_ = initial_velocity;
    apply_velocity_bc();
    
    if (turb_model_) {
        turb_model_->initialize(*mesh_, velocity_);
    }
}

void RANSSolver::initialize_uniform(double u0, double v0) {
    velocity_.fill(u0, v0);
    apply_velocity_bc();
    
    if (turb_model_) {
        turb_model_->initialize(*mesh_, velocity_);
    }
}

void RANSSolver::apply_velocity_bc() {
    int Nx = mesh_->Nx;
    int Ny = mesh_->Ny;
    int Ng = mesh_->Nghost;
    
    // x-direction (periodic or inflow/outflow)
    for (int j = 0; j < mesh_->total_Ny(); ++j) {
        for (int g = 0; g < Ng; ++g) {
            // Left boundary
            int i_ghost = g;
            int i_interior = Ng;
            int i_periodic = Nx + Ng - 1 - g;
            
            if (velocity_bc_.x_lo == VelocityBC::Periodic) {
                velocity_.u(i_ghost, j) = velocity_.u(i_periodic, j);
                velocity_.v(i_ghost, j) = velocity_.v(i_periodic, j);
            } else if (velocity_bc_.x_lo == VelocityBC::NoSlip) {
                velocity_.u(i_ghost, j) = -velocity_.u(i_interior, j);
                velocity_.v(i_ghost, j) = -velocity_.v(i_interior, j);
            }
            
            // Right boundary
            i_ghost = Nx + Ng + g;
            i_interior = Nx + Ng - 1;
            i_periodic = Ng + g;
            
            if (velocity_bc_.x_hi == VelocityBC::Periodic) {
                velocity_.u(i_ghost, j) = velocity_.u(i_periodic, j);
                velocity_.v(i_ghost, j) = velocity_.v(i_periodic, j);
            } else if (velocity_bc_.x_hi == VelocityBC::NoSlip) {
                velocity_.u(i_ghost, j) = -velocity_.u(i_interior, j);
                velocity_.v(i_ghost, j) = -velocity_.v(i_interior, j);
            }
        }
    }
    
    // y-direction (typically no-slip walls for channel)
    for (int i = 0; i < mesh_->total_Nx(); ++i) {
        for (int g = 0; g < Ng; ++g) {
            // Bottom wall
            int j_ghost = g;
            int j_interior = Ng;
            
            if (velocity_bc_.y_lo == VelocityBC::NoSlip) {
                // Linear extrapolation for no-slip: u_wall = 0
                // ghost = -interior (simple first order)
                velocity_.u(i, j_ghost) = -velocity_.u(i, j_interior);
                velocity_.v(i, j_ghost) = -velocity_.v(i, j_interior);
            } else if (velocity_bc_.y_lo == VelocityBC::Periodic) {
                int j_periodic = Ny + Ng - 1 - g;
                velocity_.u(i, j_ghost) = velocity_.u(i, j_periodic);
                velocity_.v(i, j_ghost) = velocity_.v(i, j_periodic);
            }
            
            // Top wall
            j_ghost = Ny + Ng + g;
            j_interior = Ny + Ng - 1;
            
            if (velocity_bc_.y_hi == VelocityBC::NoSlip) {
                velocity_.u(i, j_ghost) = -velocity_.u(i, j_interior);
                velocity_.v(i, j_ghost) = -velocity_.v(i, j_interior);
            } else if (velocity_bc_.y_hi == VelocityBC::Periodic) {
                int j_periodic = Ng + g;
                velocity_.u(i, j_ghost) = velocity_.u(i, j_periodic);
                velocity_.v(i, j_ghost) = velocity_.v(i, j_periodic);
            }
        }
    }
}

void RANSSolver::compute_convective_term(const VectorField& vel, VectorField& conv) {
    // Compute: (u*nabla)u using specified scheme
    // Central difference or upwind
    
    double dx = mesh_->dx;
    double dy = mesh_->dy;
    
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            double uu = vel.u(i, j);
            double vv = vel.v(i, j);
            
            double dudx, dudy, dvdx, dvdy;
            
            if (config_.convective_scheme == ConvectiveScheme::Central) {
                // Central differences
                dudx = (vel.u(i+1, j) - vel.u(i-1, j)) / (2.0 * dx);
                dudy = (vel.u(i, j+1) - vel.u(i, j-1)) / (2.0 * dy);
                dvdx = (vel.v(i+1, j) - vel.v(i-1, j)) / (2.0 * dx);
                dvdy = (vel.v(i, j+1) - vel.v(i, j-1)) / (2.0 * dy);
            } else {
                // First-order upwind
                if (uu >= 0) {
                    dudx = (vel.u(i, j) - vel.u(i-1, j)) / dx;
                    dvdx = (vel.v(i, j) - vel.v(i-1, j)) / dx;
                } else {
                    dudx = (vel.u(i+1, j) - vel.u(i, j)) / dx;
                    dvdx = (vel.v(i+1, j) - vel.v(i, j)) / dx;
                }
                
                if (vv >= 0) {
                    dudy = (vel.u(i, j) - vel.u(i, j-1)) / dy;
                    dvdy = (vel.v(i, j) - vel.v(i, j-1)) / dy;
                } else {
                    dudy = (vel.u(i, j+1) - vel.u(i, j)) / dy;
                    dvdy = (vel.v(i, j+1) - vel.v(i, j)) / dy;
                }
            }
            
            conv.u(i, j) = uu * dudx + vv * dudy;
            conv.v(i, j) = uu * dvdx + vv * dvdy;
        }
    }
}

void RANSSolver::compute_diffusive_term(const VectorField& vel, const ScalarField& nu_eff, 
                                        VectorField& diff) {
    // Compute: nabla*(nu_eff nablau)  (simplified: nu_eff nabla^2u for uniform nu_eff)
    // For variable nu_eff, use proper flux formulation
    
    double dx = mesh_->dx;
    double dy = mesh_->dy;
    double dx2 = dx * dx;
    double dy2 = dy * dy;
    
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            // Face-averaged effective viscosity
            double nu_e = 0.5 * (nu_eff(i, j) + nu_eff(i+1, j));
            double nu_w = 0.5 * (nu_eff(i, j) + nu_eff(i-1, j));
            double nu_n = 0.5 * (nu_eff(i, j) + nu_eff(i, j+1));
            double nu_s = 0.5 * (nu_eff(i, j) + nu_eff(i, j-1));
            
            // Diffusive flux for u
            double diff_u_x = (nu_e * (vel.u(i+1, j) - vel.u(i, j)) 
                            - nu_w * (vel.u(i, j) - vel.u(i-1, j))) / dx2;
            double diff_u_y = (nu_n * (vel.u(i, j+1) - vel.u(i, j)) 
                            - nu_s * (vel.u(i, j) - vel.u(i, j-1))) / dy2;
            
            // Diffusive flux for v
            double diff_v_x = (nu_e * (vel.v(i+1, j) - vel.v(i, j)) 
                            - nu_w * (vel.v(i, j) - vel.v(i-1, j))) / dx2;
            double diff_v_y = (nu_n * (vel.v(i, j+1) - vel.v(i, j)) 
                            - nu_s * (vel.v(i, j) - vel.v(i, j-1))) / dy2;
            
            diff.u(i, j) = diff_u_x + diff_u_y;
            diff.v(i, j) = diff_v_x + diff_v_y;
        }
    }
}

void RANSSolver::compute_divergence(const VectorField& vel, ScalarField& div) {
    double dx = mesh_->dx;
    double dy = mesh_->dy;
    
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            double dudx = (vel.u(i+1, j) - vel.u(i-1, j)) / (2.0 * dx);
            double dvdy = (vel.v(i, j+1) - vel.v(i, j-1)) / (2.0 * dy);
            div(i, j) = dudx + dvdy;
        }
    }
}

void RANSSolver::compute_pressure_gradient(ScalarField& dp_dx, ScalarField& dp_dy) {
    double dx = mesh_->dx;
    double dy = mesh_->dy;
    
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            dp_dx(i, j) = (pressure_(i+1, j) - pressure_(i-1, j)) / (2.0 * dx);
            dp_dy(i, j) = (pressure_(i, j+1) - pressure_(i, j-1)) / (2.0 * dy);
        }
    }
}

void RANSSolver::correct_velocity() {
    double dx = mesh_->dx;
    double dy = mesh_->dy;
    
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            double dp_dx = (pressure_correction_(i+1, j) - pressure_correction_(i-1, j)) / (2.0 * dx);
            double dp_dy = (pressure_correction_(i, j+1) - pressure_correction_(i, j-1)) / (2.0 * dy);
            
            velocity_.u(i, j) = velocity_star_.u(i, j) - current_dt_ * dp_dx;
            velocity_.v(i, j) = velocity_star_.v(i, j) - current_dt_ * dp_dy;
        }
    }
    
    // Update pressure
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            pressure_(i, j) += pressure_correction_(i, j);
        }
    }
}

double RANSSolver::compute_residual() {
    // Compute residual based on velocity change
    double max_res = 0.0;
    
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            double du = velocity_.u(i, j) - velocity_star_.u(i, j);
            double dv = velocity_.v(i, j) - velocity_star_.v(i, j);
            max_res = std::max(max_res, std::abs(du));
            max_res = std::max(max_res, std::abs(dv));
        }
    }
    
    return max_res;
}

double RANSSolver::step() {
    TIMED_SCOPE("solver_step");
    
    // Store old velocity for convergence check
    VectorField velocity_old(*mesh_);
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            velocity_old.u(i, j) = velocity_.u(i, j);
            velocity_old.v(i, j) = velocity_.v(i, j);
        }
    }
    
    // 1. Update turbulence model (if any)
    if (turb_model_) {
        TIMED_SCOPE("turbulence_update");
        turb_model_->update(*mesh_, velocity_, k_, omega_, nu_t_, 
                           turb_model_->provides_reynolds_stresses() ? &tau_ij_ : nullptr);
    }
    
    // Effective viscosity: nu_eff = nu + nu_t
    ScalarField nu_eff(*mesh_, config_.nu);
    if (turb_model_) {
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                nu_eff(i, j) = config_.nu + nu_t_(i, j);
            }
        }
    }
    
    // 2. Compute convective and diffusive terms
    VectorField conv(*mesh_);
    VectorField diff(*mesh_);
    
    {
        TIMED_SCOPE("convective_term");
        compute_convective_term(velocity_, conv);
    }
    
    {
        TIMED_SCOPE("diffusive_term");
        compute_diffusive_term(velocity_, nu_eff, diff);
    }
    
    // 3. Compute provisional velocity u* (without pressure gradient)
    // u* = u^n + dt * (-conv + diff + body_force)
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            velocity_star_.u(i, j) = velocity_.u(i, j) + current_dt_ * 
                (-conv.u(i, j) + diff.u(i, j) + fx_);
            velocity_star_.v(i, j) = velocity_.v(i, j) + current_dt_ * 
                (-conv.v(i, j) + diff.v(i, j) + fy_);
        }
    }
    
    // Apply BCs to provisional velocity (needed for divergence calculation)
    // Temporarily swap velocity_ and velocity_star_ to use apply_velocity_bc
    std::swap(velocity_, velocity_star_);
    apply_velocity_bc();
    std::swap(velocity_, velocity_star_);
    
    // 4. Solve pressure Poisson equation
    // nabla^2p' = (1/dt) nabla*u*
    {
        TIMED_SCOPE("divergence");
        compute_divergence(velocity_star_, div_velocity_);
    }
    
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            rhs_poisson_(i, j) = div_velocity_(i, j) / current_dt_;
        }
    }
    
    pressure_correction_.fill(0.0);
    
    {
        TIMED_SCOPE("poisson_solve");
        PoissonConfig pcfg;
        pcfg.tol = config_.poisson_tol;
        pcfg.max_iter = config_.poisson_max_iter;
        pcfg.omega = config_.poisson_omega;
        poisson_solver_.solve(rhs_poisson_, pressure_correction_, pcfg);
    }
    
    // 5. Correct velocity and pressure
    {
        TIMED_SCOPE("velocity_correction");
        correct_velocity();
    }
    
    // 6. Apply boundary conditions
    apply_velocity_bc();
    
    ++iter_;
    
    // Return max velocity change as convergence criterion
    double max_change = 0.0;
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            double du = std::abs(velocity_.u(i, j) - velocity_old.u(i, j));
            double dv = std::abs(velocity_.v(i, j) - velocity_old.v(i, j));
            max_change = std::max(max_change, std::max(du, dv));
        }
    }
    return max_change;
}

std::pair<double, int> RANSSolver::solve_steady() {
    double residual = 1.0;
    
    if (config_.verbose) {
        if (config_.adaptive_dt) {
            std::cout << std::setw(8) << "Iter" 
                      << std::setw(15) << "Residual"
                      << std::setw(15) << "Max |u|"
                      << std::setw(12) << "dt"
                      << "\n";
        } else {
            std::cout << std::setw(8) << "Iter" 
                      << std::setw(15) << "Residual"
                      << std::setw(15) << "Max |u|"
                      << "\n";
        }
    }
    
    for (iter_ = 0; iter_ < config_.max_iter; ++iter_) {
        // Update time step if adaptive
        if (config_.adaptive_dt) {
            current_dt_ = compute_adaptive_dt();
        }
        
        residual = step();
        
        if (config_.verbose && (iter_ + 1) % config_.output_freq == 0) {
            double max_vel = velocity_.max_magnitude();
            if (config_.adaptive_dt) {
                std::cout << std::setw(8) << iter_ + 1
                          << std::setw(15) << std::scientific << std::setprecision(3) << residual
                          << std::setw(15) << std::fixed << max_vel
                          << std::setw(12) << std::scientific << std::setprecision(2) << current_dt_
                          << "\n";
            } else {
                std::cout << std::setw(8) << iter_ + 1
                          << std::setw(15) << std::scientific << std::setprecision(3) << residual
                          << std::setw(15) << std::fixed << max_vel
                          << "\n";
            }
        }
        
        if (residual < config_.tol) {
            if (config_.verbose) {
                std::cout << "Converged at iteration " << iter_ + 1 
                          << " with residual " << residual << "\n";
            }
            break;
        }
        
        // Check for divergence
        if (std::isnan(residual) || std::isinf(residual)) {
            if (config_.verbose) {
                std::cerr << "Solver diverged at iteration " << iter_ + 1 << "\n";
            }
            break;
        }
    }
    
    return {residual, iter_ + 1};
}

double RANSSolver::bulk_velocity() const {
    // Area-averaged streamwise velocity
    double sum = 0.0;
    int count = 0;
    
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            sum += velocity_.u(i, j);
            ++count;
        }
    }
    
    return sum / count;
}

double RANSSolver::wall_shear_stress() const {
    // Compute du/dy at the bottom wall
    // Using one-sided difference from first interior cell to wall
    double sum = 0.0;
    int count = 0;
    
    int j = mesh_->j_begin();  // First interior row
    double y_cell = mesh_->y(j);
    double y_wall = mesh_->y_min;
    double dist = y_cell - y_wall;
    
    for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
        // u at wall is 0 (no-slip)
        double dudy = velocity_.u(i, j) / dist;
        sum += dudy;
        ++count;
    }
    
    double dudy_avg = sum / count;
    return config_.nu * dudy_avg;  // tau_w = mu * du/dy = rho * nu * du/dy (rho=1)
}

double RANSSolver::friction_velocity() const {
    double tau_w = wall_shear_stress();
    return std::sqrt(std::abs(tau_w));  // u_tau = sqrt(tau_w / rho)
}

double RANSSolver::Re_tau() const {
    double u_tau = friction_velocity();
    double delta = (mesh_->y_max - mesh_->y_min) / 2.0;
    return u_tau * delta / config_.nu;
}

void RANSSolver::print_velocity_profile(double x_loc) const {
    // Find i index closest to x_loc
    int i_loc = mesh_->i_begin();
    double min_dist = std::abs(mesh_->x(i_loc) - x_loc);
    
    for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
        double dist = std::abs(mesh_->x(i) - x_loc);
        if (dist < min_dist) {
            min_dist = dist;
            i_loc = i;
        }
    }
    
    std::cout << "\nVelocity profile at x = " << mesh_->x(i_loc) << ":\n";
    std::cout << std::setw(12) << "y" << std::setw(12) << "u" << std::setw(12) << "v" << "\n";
    
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        std::cout << std::setw(12) << std::fixed << std::setprecision(4) << mesh_->y(j)
                  << std::setw(12) << velocity_.u(i_loc, j)
                  << std::setw(12) << velocity_.v(i_loc, j)
                  << "\n";
    }
}

void RANSSolver::write_fields(const std::string& prefix) const {
    velocity_.write(prefix + "_velocity.dat");
    pressure_.write(prefix + "_pressure.dat");
    
    if (turb_model_) {
        nu_t_.write(prefix + "_nu_t.dat");
    }
}

double RANSSolver::compute_adaptive_dt() const {
    // CFL condition: dt <= CFL * min(dx, dy) / |u_max|
    double u_max = 1e-10;  // Small value to avoid division by zero
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            double u_mag = std::sqrt(velocity_.u(i, j) * velocity_.u(i, j) + 
                                    velocity_.v(i, j) * velocity_.v(i, j));
            u_max = std::max(u_max, u_mag);
        }
    }
    double dt_cfl = config_.CFL_max * std::min(mesh_->dx, mesh_->dy) / u_max;
    
    // Diffusion stability: dt <= factor * min(dx^2, dy^2) / nu_eff_max
    double nu_eff_max = config_.nu;
    if (turb_model_) {
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                nu_eff_max = std::max(nu_eff_max, config_.nu + nu_t_(i, j));
            }
        }
    }
    double dx_min = std::min(mesh_->dx, mesh_->dy);
    double dt_diff = 0.25 * dx_min * dx_min / nu_eff_max;
    
    // Take minimum of both constraints
    return std::min(dt_cfl, dt_diff);
}

void RANSSolver::write_vtk(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error: Cannot open " << filename << " for writing\n";
        return;
    }
    
    int Nx = mesh_->Nx;
    int Ny = mesh_->Ny;
    
    // VTK header
    file << "# vtk DataFile Version 3.0\n";
    file << "RANS simulation output\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_POINTS\n";
    file << "DIMENSIONS " << Nx << " " << Ny << " 1\n";
    file << "ORIGIN " << mesh_->x_min << " " << mesh_->y_min << " 0\n";
    file << "SPACING " << mesh_->dx << " " << mesh_->dy << " 1\n";
    file << "POINT_DATA " << Nx * Ny << "\n";
    
    // Velocity vector field
    file << "VECTORS velocity double\n";
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            file << velocity_.u(i, j) << " " << velocity_.v(i, j) << " 0\n";
        }
    }
    
    // Pressure scalar field
    file << "SCALARS pressure double 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            file << pressure_(i, j) << "\n";
        }
    }
    
    // Velocity magnitude
    file << "SCALARS velocity_magnitude double 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            double mag = std::sqrt(velocity_.u(i, j) * velocity_.u(i, j) + 
                                  velocity_.v(i, j) * velocity_.v(i, j));
            file << mag << "\n";
        }
    }
    
    // Eddy viscosity (if turbulence model is active)
    if (turb_model_) {
        file << "SCALARS nu_t double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                file << nu_t_(i, j) << "\n";
            }
        }
    }
    
    file.close();
}

} // namespace nncfd

