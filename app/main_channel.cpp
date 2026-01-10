/// Channel flow solver with Poiseuille validation
/// Solves incompressible Navier-Stokes in a 2D or 3D channel with:
/// - Periodic boundary conditions in x (streamwise)
/// - No-slip walls at y = y_min and y = y_max
/// - For 3D (Nz > 1): Periodic boundary conditions in z (spanwise)
/// - Constant body force (pressure gradient) driving the flow
///
/// 2D mode: Nz = 1 (default)
/// 3D mode: Nz > 1 (spanwise-periodic channel)

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "config.hpp"
#include "timing.hpp"
#include "turbulence_model.hpp"
#include "turbulence_baseline.hpp"
#include "turbulence_nn_mlp.hpp"
#include "turbulence_nn_tbnn.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <filesystem>

using namespace nncfd;

/// Analytical Poiseuille solution: u(y) = -(dp/dx)/(2*nu) * (y^2 - H^2)
/// where H is the half-height and y is measured from centerline
/// For channel [-H, H]: u(y) = u_max * (1 - y^2/H^2)
/// where u_max = -(dp/dx) * H^2 / (2*nu)
double poiseuille_velocity(double y, double H, double dp_dx, double nu) {
    double u_max = -dp_dx * H * H / (2.0 * nu);
    return u_max * (1.0 - (y * y) / (H * H));
}

/// Compute L2 error against Poiseuille solution (works for 2D and 3D)
double compute_poiseuille_error(const Mesh& mesh, const VectorField& velocity,
                                double dp_dx, double nu) {
    double H = (mesh.y_max - mesh.y_min) / 2.0;
    double y_center = (mesh.y_min + mesh.y_max) / 2.0;

    double sum_error2 = 0.0;
    double sum_exact2 = 0.0;

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            double y = mesh.y(j) - y_center;  // y from centerline
            double u_exact = poiseuille_velocity(y, H, dp_dx, nu);

            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double u_num = mesh.is2D() ? velocity.u(i, j) : velocity.u(i, j, k);

                double err = u_num - u_exact;
                sum_error2 += err * err;
                sum_exact2 += u_exact * u_exact;
            }
        }
    }

    return std::sqrt(sum_error2 / sum_exact2);
}

/// Print comparison with analytical solution (works for 2D and 3D)
void compare_with_analytical(const Mesh& mesh, const VectorField& velocity,
                             double dp_dx, double nu) {
    double H = (mesh.y_max - mesh.y_min) / 2.0;
    double y_center = (mesh.y_min + mesh.y_max) / 2.0;

    std::cout << "\n=== Comparison with Poiseuille Solution ===\n";
    std::cout << std::setw(12) << "y"
              << std::setw(15) << "u_numerical"
              << std::setw(15) << "u_analytical"
              << std::setw(15) << "error"
              << "\n";

    // Sample at center x (and center z for 3D)
    int i = mesh.i_begin() + mesh.Nx / 2;
    int k = mesh.k_begin() + mesh.Nz / 2;

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        double y = mesh.y(j);
        double y_from_center = y - y_center;
        double u_exact = poiseuille_velocity(y_from_center, H, dp_dx, nu);
        double u_num = mesh.is2D() ? velocity.u(i, j) : velocity.u(i, j, k);
        double error = std::abs(u_num - u_exact);

        std::cout << std::setw(12) << std::fixed << std::setprecision(4) << y
                  << std::setw(15) << std::setprecision(6) << u_num
                  << std::setw(15) << u_exact
                  << std::setw(15) << std::scientific << error
                  << "\n";
    }

    double L2_error = compute_poiseuille_error(mesh, velocity, dp_dx, nu);
    std::cout << "\nRelative L2 error: " << std::scientific << L2_error << "\n";
}

/// Write velocity profile to file for plotting (works for 2D and 3D)
void write_profile(const std::string& filename, const Mesh& mesh,
                   const VectorField& velocity, double dp_dx, double nu) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << "\n";
        return;
    }

    double H = (mesh.y_max - mesh.y_min) / 2.0;
    double y_center = (mesh.y_min + mesh.y_max) / 2.0;

    file << "# y  u_numerical  u_analytical\n";

    // Sample at center x (and center z for 3D)
    int i = mesh.i_begin() + mesh.Nx / 2;
    int k = mesh.k_begin() + mesh.Nz / 2;

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        double y = mesh.y(j);
        double y_from_center = y - y_center;
        double u_exact = poiseuille_velocity(y_from_center, H, dp_dx, nu);
        double u_num = mesh.is2D() ? velocity.u(i, j) : velocity.u(i, j, k);

        file << y << " " << u_num << " " << u_exact << "\n";
    }
}

/// Create divergence-free perturbed velocity field via streamfunction
/// ψ(x,y) = A * sin(kx*x) * sin²(π(y+1)/2)
/// u = ∂ψ/∂y, v = -∂ψ/∂x guarantees ∇·u = 0 exactly
/// Wall factor sin²(π(y+1)/2) vanishes at y=±1 (no-slip compatible)
/// For 3D: extends uniformly in z (w = 0, still divergence-free)
VectorField create_perturbed_channel_field(const Mesh& mesh, double amplitude = 1e-3) {
    VectorField vel(mesh);
    const double kx = 2.0 * M_PI / (mesh.x_max - mesh.x_min);

    if (mesh.is2D()) {
        // 2D case
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            double y = mesh.y(j);
            double s = std::sin(0.5 * M_PI * (y + 1.0));
            double c = std::cos(0.5 * M_PI * (y + 1.0));
            double dpsi_dy_factor = M_PI * s * c;

            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                double x = mesh.xf[i];
                vel.u(i, j) = amplitude * std::sin(kx * x) * dpsi_dy_factor;
            }
        }

        for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
            double y = mesh.yf[j];
            double s = std::sin(0.5 * M_PI * (y + 1.0));
            double s2 = s * s;

            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double x = mesh.x(i);
                double dpsi_dx = amplitude * kx * std::cos(kx * x) * s2;
                vel.v(i, j) = -dpsi_dx;
            }
        }
    } else {
        // 3D case: extend 2D streamfunction uniformly in z
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                double y = mesh.y(j);
                double s = std::sin(0.5 * M_PI * (y + 1.0));
                double c = std::cos(0.5 * M_PI * (y + 1.0));
                double dpsi_dy_factor = M_PI * s * c;

                for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                    double x = mesh.xf[i];
                    vel.u(i, j, k) = amplitude * std::sin(kx * x) * dpsi_dy_factor;
                }
            }
        }

        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
                double y = mesh.yf[j];
                double s = std::sin(0.5 * M_PI * (y + 1.0));
                double s2 = s * s;

                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    double x = mesh.x(i);
                    double dpsi_dx = amplitude * kx * std::cos(kx * x) * s2;
                    vel.v(i, j, k) = -dpsi_dx;
                }
            }
        }
        // w = 0 by default (VectorField initializes to zero)
    }

    return vel;
}

int main(int argc, char** argv) {
    std::cout << "=== Channel Flow Solver ===\n\n";

    // Parse configuration
    Config config;

    // Default channel flow settings
    config.Nx = 16;
    config.Ny = 32;
    config.Nz = 1;              // 2D by default, set Nz > 1 for 3D
    config.x_min = 0.0;
    config.x_max = 2.0 * M_PI;  // One wavelength in x
    config.y_min = -1.0;
    config.y_max = 1.0;         // Channel half-height H = 1
    config.z_min = 0.0;
    config.z_max = M_PI;        // Half wavelength in z (for 3D)

    config.nu = 0.01;           // Kinematic viscosity
    config.dp_dx = -1.0;        // Pressure gradient (body force)

    config.dt = 0.001;
    config.max_iter = 50000;
    config.tol = 1e-8;
    config.output_freq = 1000;
    config.verbose = true;

    config.turb_model = TurbulenceModelType::None;  // Laminar by default

    config.poisson_tol = 1e-8;
    config.poisson_max_iter = 20;  // V-cycles for multigrid (not SOR iterations)
    config.poisson_omega = 1.8;

    // Parse command line
    config.parse_args(argc, argv);
    config.print();

    bool is3D = config.Nz > 1;
    if (is3D) {
        std::cout << "Running in 3D mode (spanwise-periodic channel)\n\n";
    }
    
    // Ensure output directory exists
    try {
        std::filesystem::create_directories(config.output_dir);
    } catch (const std::exception& e) {
        std::cerr << "Warning: Could not create output directory: " << e.what() << "\n";
    }
    
    // Compute expected max velocity for Poiseuille flow
    double H = (config.y_max - config.y_min) / 2.0;
    double u_max_expected = -config.dp_dx * H * H / (2.0 * config.nu);
    std::cout << "Expected centerline velocity (Poiseuille): " << u_max_expected << "\n";
    std::cout << "Expected bulk velocity: " << (2.0/3.0) * u_max_expected << "\n\n";
    
    // Create mesh (2D or 3D based on Nz)
    Mesh mesh;
    if (is3D) {
        if (config.stretch_y) {
            mesh.init_stretched_y(config.Nx, config.Ny, config.Nz,
                                  config.x_min, config.x_max,
                                  config.y_min, config.y_max,
                                  config.z_min, config.z_max,
                                  Mesh::tanh_stretching(config.stretch_beta));
        } else {
            mesh.init_uniform(config.Nx, config.Ny, config.Nz,
                              config.x_min, config.x_max,
                              config.y_min, config.y_max,
                              config.z_min, config.z_max);
        }
        std::cout << "Mesh: " << mesh.Nx << " x " << mesh.Ny << " x " << mesh.Nz << " cells (3D)\n";
        std::cout << "dx = " << mesh.dx << ", dy = " << mesh.dy << ", dz = " << mesh.dz << "\n\n";
    } else {
        if (config.stretch_y) {
            mesh.init_stretched_y(config.Nx, config.Ny,
                                  config.x_min, config.x_max,
                                  config.y_min, config.y_max,
                                  Mesh::tanh_stretching(config.stretch_beta));
        } else {
            mesh.init_uniform(config.Nx, config.Ny,
                              config.x_min, config.x_max,
                              config.y_min, config.y_max);
        }
        std::cout << "Mesh: " << mesh.Nx << " x " << mesh.Ny << " cells\n";
        std::cout << "dx = " << mesh.dx << ", dy = " << mesh.dy << "\n\n";
    }
    
    // Create solver (Poisson solver selection handled internally via config.poisson_solver)
    RANSSolver solver(mesh, config);

    // Set boundary conditions
    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;  // Streamwise periodic
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;    // Walls
    bc.y_hi = VelocityBC::NoSlip;
    if (is3D) {
        bc.z_lo = VelocityBC::Periodic;  // Spanwise periodic
        bc.z_hi = VelocityBC::Periodic;
    }
    solver.set_velocity_bc(bc);

    // Set body force (equivalent to pressure gradient)
    solver.set_body_force(-config.dp_dx, 0.0);  // fz = 0 by default
    
    // Set turbulence model if requested
    if (config.turb_model != TurbulenceModelType::None) {
        auto turb_model = create_turbulence_model(config.turb_model,
                                                  config.nn_weights_path,
                                                  config.nn_scaling_path);
        if (turb_model) {
            turb_model->set_nu(config.nu);
            
            // Set additional parameters for baseline models
            if (auto* ml = dynamic_cast<MixingLengthModel*>(turb_model.get())) {
                ml->set_delta(H);
            }
            if (auto* nn_mlp = dynamic_cast<TurbulenceNNMLP*>(turb_model.get())) {
                nn_mlp->set_delta(H);
                nn_mlp->set_nu_t_max(config.nu_t_max);
            }
            if (auto* nn_tbnn = dynamic_cast<TurbulenceNNTBNN*>(turb_model.get())) {
                nn_tbnn->set_delta(H);
            }
            
            solver.set_turbulence_model(std::move(turb_model));
        }
    }

    // Print final solver configuration (after all setup)
    solver.print_solver_info();

    // Branch based on simulation mode
    double final_residual = 0.0;
    int total_iterations = 0;
    
    if (config.simulation_mode == SimulationMode::Unsteady) {
        // ============================================================
        // UNSTEADY MODE: Time-accurate integration
        // ============================================================
        std::cout << "\n=== Running in UNSTEADY mode ===\n";
        std::cout << "Time steps: " << config.max_iter << "\n";
        std::cout << "Initial dt: " << config.dt << "\n\n";
        
        // Force laminar for unsteady developing flow
        config.turb_model = TurbulenceModelType::None;

        // Initialize with divergence-free perturbation
        std::cout << "Perturbation amplitude: " << config.perturbation_amplitude << "\n";
        solver.initialize(create_perturbed_channel_field(mesh, config.perturbation_amplitude));
        
    #ifdef USE_GPU_OFFLOAD
        solver.sync_to_gpu();
    #endif
        
        const std::string prefix = config.write_fields ? (config.output_dir + "developing_channel") : "";
        const int snapshot_freq = (config.num_snapshots > 0) ?
            std::max(1, config.max_iter / config.num_snapshots) : 0;
        
        ScopedTimer total_timer("Total simulation", false);
        
        int snap_count = 0;
        // Progress output interval for CI visibility (always enabled)
        const int progress_interval = std::max(1, config.max_iter / 10);

        for (int step = 1; step <= config.max_iter; ++step) {
            if (config.adaptive_dt) {
                (void)solver.compute_adaptive_dt();
            }
            double residual = solver.step();

            // Reset timers after warmup iterations (excluded from reported timing)
            if (config.warmup_iter > 0 && step == config.warmup_iter) {
                TimingStats::instance().reset();
                if (config.verbose) {
                    std::cout << "    [Warmup complete: " << config.warmup_iter
                              << " iterations, timers reset]\n";
                }
            }

            if (!prefix.empty() && snapshot_freq > 0 && (step % snapshot_freq == 0)) {
                ++snap_count;
                solver.write_vtk(prefix + "_" + std::to_string(snap_count) + ".vtk");
            }

            // Always show progress every ~10% for CI visibility
            if (step % progress_interval == 0 || step == 1) {
                std::cout << "    Step " << std::setw(6) << step << " / " << config.max_iter
                          << "  (" << std::setw(3) << (100 * step / config.max_iter) << "%)"
                          << "  residual = " << std::scientific << std::setprecision(3) << residual
                          << std::fixed << "\n" << std::flush;
            } else if (config.verbose && (step % config.output_freq == 0)) {
                std::cout << "Step " << step << " / " << config.max_iter
                          << ", residual = " << std::scientific << residual << "\n";
            }
            
            if (std::isnan(residual) || std::isinf(residual)) {
                std::cerr << "ERROR: Solver diverged at step " << step << "\n";
                return 1;
            }
            
            final_residual = residual;
        }
        
        if (!prefix.empty()) {
            solver.write_vtk(prefix + "_final.vtk");
        }
        
        total_timer.stop();
        total_iterations = config.max_iter;
        
        std::cout << "\n=== Unsteady simulation complete ===\n";
        
    } else {
        // ============================================================
        // STEADY MODE: Convergence-based solve
        // ============================================================
        std::cout << "\n=== Running in STEADY mode ===\n";
        std::cout << "Convergence tolerance: " << config.tol << "\n";
        std::cout << "Max iterations: " << config.max_iter << "\n\n";
        
        // Initialize with small perturbation (w=0 for 3D handled internally)
        solver.initialize_uniform(0.1 * u_max_expected, 0.0);
        
        // Solve to steady state with automatic VTK snapshots
        ScopedTimer total_timer("Total simulation", false);

        // Benchmark-friendly: allow skipping all file output (snapshots + final fields)
        const std::string output_prefix = config.write_fields ? (config.output_dir + "channel") : "";
        const int num_snapshots = config.write_fields ? config.num_snapshots : 0;
        auto [residual, iterations] = solver.solve_steady_with_snapshots(output_prefix, num_snapshots);
        
        total_timer.stop();
        
        final_residual = residual;
        total_iterations = iterations;
    }
    
    // Results (common to both modes)
    std::cout << "\n=== Results ===\n";
    std::cout << "Final residual: " << std::scientific << final_residual << "\n";
    std::cout << "Iterations/Steps: " << total_iterations << "\n";
    if (config.simulation_mode == SimulationMode::Steady) {
        std::cout << "Converged: " << (final_residual < config.tol ? "YES" : "NO") << "\n";
    }
    std::cout << "Bulk velocity: " << std::fixed << std::setprecision(6) << solver.bulk_velocity() << "\n";
    std::cout << "Wall shear stress: " << solver.wall_shear_stress() << "\n";
    std::cout << "Friction velocity u_tau: " << solver.friction_velocity() << "\n";
    std::cout << "Re_tau: " << solver.Re_tau() << "\n";
    
    // Compare with analytical solution (for laminar case)
    if (config.postprocess && config.turb_model == TurbulenceModelType::None) {
        compare_with_analytical(mesh, solver.velocity(), config.dp_dx, config.nu);
        
        double L2_error = compute_poiseuille_error(mesh, solver.velocity(), 
                                                   config.dp_dx, config.nu);
        
        if (L2_error < 0.01) {
            std::cout << "\n*** VALIDATION PASSED: L2 error < 1% ***\n";
        } else {
            std::cout << "\n*** WARNING: L2 error = " << L2_error * 100 << "% ***\n";
        }
    }
    
    // Write additional output files
    try {
        if (config.postprocess) {
            write_profile(config.output_dir + "velocity_profile.dat", mesh,
                          solver.velocity(), config.dp_dx, config.nu);
        }
        if (config.write_fields) {
            solver.write_fields(config.output_dir + "channel");
        }
    } catch (const std::exception& e) {
        std::cerr << "Warning: Could not write output files: " << e.what() << "\n";
    }
    
    // Print timing summary
    TimingStats::instance().print_summary();
    
    return 0;
}

