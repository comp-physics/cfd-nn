/// Channel flow solver with Poiseuille validation
/// Solves incompressible Navier-Stokes in a 2D channel with:
/// - Periodic boundary conditions in x
/// - No-slip walls at y = y_min and y = y_max
/// - Constant body force (pressure gradient) driving the flow

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

using namespace nncfd;

/// Analytical Poiseuille solution: u(y) = -(dp/dx)/(2*nu) * (y^2 - H^2)
/// where H is the half-height and y is measured from centerline
/// For channel [-H, H]: u(y) = u_max * (1 - y^2/H^2)
/// where u_max = -(dp/dx) * H^2 / (2*nu)
double poiseuille_velocity(double y, double H, double dp_dx, double nu) {
    double u_max = -dp_dx * H * H / (2.0 * nu);
    return u_max * (1.0 - (y * y) / (H * H));
}

/// Compute L2 error against Poiseuille solution
double compute_poiseuille_error(const Mesh& mesh, const VectorField& velocity,
                                double dp_dx, double nu) {
    double H = (mesh.y_max - mesh.y_min) / 2.0;
    double y_center = (mesh.y_min + mesh.y_max) / 2.0;
    
    double sum_error2 = 0.0;
    double sum_exact2 = 0.0;
    
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double y = mesh.y(j) - y_center;  // y from centerline
            double u_exact = poiseuille_velocity(y, H, dp_dx, nu);
            double u_num = velocity.u(i, j);
            
            double err = u_num - u_exact;
            sum_error2 += err * err;
            sum_exact2 += u_exact * u_exact;
        }
    }
    
    return std::sqrt(sum_error2 / sum_exact2);
}

/// Print comparison with analytical solution
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
    
    // Print at center x location
    int i = mesh.i_begin() + mesh.Nx / 2;
    
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        double y = mesh.y(j);
        double y_from_center = y - y_center;
        double u_exact = poiseuille_velocity(y_from_center, H, dp_dx, nu);
        double u_num = velocity.u(i, j);
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

/// Write velocity profile to file for plotting
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
    
    int i = mesh.i_begin() + mesh.Nx / 2;
    
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        double y = mesh.y(j);
        double y_from_center = y - y_center;
        double u_exact = poiseuille_velocity(y_from_center, H, dp_dx, nu);
        double u_num = velocity.u(i, j);
        
        file << y << " " << u_num << " " << u_exact << "\n";
    }
}

int main(int argc, char** argv) {
    std::cout << "=== Channel Flow Solver ===\n\n";
    
    // Parse configuration
    Config config;
    
    // Default channel flow settings
    config.Nx = 16;
    config.Ny = 32;
    config.x_min = 0.0;
    config.x_max = 2.0 * M_PI;  // One wavelength in x
    config.y_min = -1.0;
    config.y_max = 1.0;         // Channel half-height H = 1
    
    config.nu = 0.01;           // Kinematic viscosity
    config.dp_dx = -1.0;        // Pressure gradient (body force)
    
    config.dt = 0.001;
    config.max_iter = 50000;
    config.tol = 1e-8;
    config.output_freq = 1000;
    config.verbose = true;
    
    config.turb_model = TurbulenceModelType::None;  // Laminar by default
    
    config.poisson_tol = 1e-8;
    config.poisson_max_iter = 5000;
    config.poisson_omega = 1.8;
    
    // Parse command line
    config.parse_args(argc, argv);
    config.print();
    
    // Compute expected max velocity for Poiseuille flow
    double H = (config.y_max - config.y_min) / 2.0;
    double u_max_expected = -config.dp_dx * H * H / (2.0 * config.nu);
    std::cout << "Expected centerline velocity (Poiseuille): " << u_max_expected << "\n";
    std::cout << "Expected bulk velocity: " << (2.0/3.0) * u_max_expected << "\n\n";
    
    // Create mesh
    Mesh mesh;
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
    
    // Create solver
    RANSSolver solver(mesh, config);
    
    // Set boundary conditions
    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);
    
    // Set body force (equivalent to pressure gradient)
    solver.set_body_force(-config.dp_dx, 0.0);
    
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
                nn_mlp->set_blend_alpha(config.blend_alpha);
            }
            if (auto* nn_tbnn = dynamic_cast<TurbulenceNNTBNN*>(turb_model.get())) {
                nn_tbnn->set_delta(H);
            }
            
            solver.set_turbulence_model(std::move(turb_model));
        }
    }
    
    // Initialize with small perturbation
    solver.initialize_uniform(0.1 * u_max_expected, 0.0);
    
    // Solve to steady state with automatic VTK snapshots
    ScopedTimer total_timer("Total simulation", true);

    // Benchmark-friendly: allow skipping all file output (snapshots + final fields)
    const std::string output_prefix = config.write_fields ? (config.output_dir + "channel") : "";
    const int num_snapshots = config.write_fields ? config.num_snapshots : 0;
    auto [residual, iterations] = solver.solve_steady_with_snapshots(output_prefix, num_snapshots);
    
    total_timer.stop();
    
    // Results
    std::cout << "\n=== Results ===\n";
    std::cout << "Final residual: " << std::scientific << residual << "\n";
    std::cout << "Iterations: " << iterations << "\n";
    std::cout << "Converged: " << (residual < config.tol ? "YES" : "NO") << "\n";
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

