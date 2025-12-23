/// Unsteady developing channel flow (no turbulence model)
/// Demonstrates time-accurate simulation with divergence-free initialization
/// 
/// Use case: DNS-like unsteady flow, laminar instability, transient dynamics
/// Contrast with main_channel.cpp which solves to steady state with RANS

#include "mesh.hpp"
#include "solver.hpp"
#include "config.hpp"
#include "timing.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <string>

using namespace nncfd;

/// Create divergence-free perturbed velocity field via streamfunction
/// ψ(x,y) = A * sin(kx*x) * sin²(π(y+1)/2)
/// u = ∂ψ/∂y, v = -∂ψ/∂x guarantees ∇·u = 0 exactly
/// Wall factor sin²(π(y+1)/2) vanishes at y=±1 (no-slip compatible)
VectorField create_perturbed_channel_field(const Mesh& mesh, double amplitude = 1e-3) {
    VectorField vel(mesh);
    const double kx = 2.0 * M_PI / (mesh.x_max - mesh.x_min);

    // u = dψ/dy at x-faces
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        double y = mesh.y(j);
        double s = std::sin(0.5 * M_PI * (y + 1.0));
        double c = std::cos(0.5 * M_PI * (y + 1.0));
        double dpsi_dy_factor = M_PI * s * c;

        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            double x = (i < mesh.i_end()) ? (mesh.x(i) + 0.5 * mesh.dx) : mesh.x_max;
            vel.u(i, j) = amplitude * std::sin(kx * x) * dpsi_dy_factor;
        }
    }

    // v = -dψ/dx at y-faces
    for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
        double y = (j < mesh.j_end()) ? (mesh.y(j) + 0.5 * mesh.dy) : mesh.y_max;
        double s = std::sin(0.5 * M_PI * (y + 1.0));
        double s2 = s * s;

        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double x = mesh.x(i);
            double dpsi_dx = amplitude * kx * std::cos(kx * x) * s2;
            vel.v(i, j) = -dpsi_dx;
        }
    }

    return vel;
}

int main(int argc, char** argv) {
    std::cout << "=== Unsteady Developing Channel Flow ===\n\n";
    
    // Parse configuration
    Config config;
    config.parse_args(argc, argv);
    
    // Override: force laminar (no turbulence model)
    config.turb_model = TurbulenceModelType::None;
    
    // Override: treat max_iter as number of time steps (not convergence iterations)
    const int nsteps = config.max_iter;
    config.max_iter = nsteps;  // Ensure consistency
    
    config.print();
    
    std::cout << "\nSimulation mode: Time-accurate unsteady (laminar)\n";
    std::cout << "Time steps: " << nsteps << "\n";
    std::cout << "Initial condition: Divergence-free perturbation\n\n";
    
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
    
    // Set body force (pressure gradient)
    solver.set_body_force(-config.dp_dx, 0.0);
    
    // Initialize with divergence-free perturbation
    std::cout << "Initializing with divergence-free perturbation...\n";
    solver.initialize(create_perturbed_channel_field(mesh, 1e-3));
    
#ifdef USE_GPU_OFFLOAD
    std::cout << "Uploading to GPU...\n";
    solver.sync_to_gpu();
#endif
    
    // Ensure output directory exists
    try {
        std::filesystem::create_directories(config.output_dir);
    } catch (const std::exception& e) {
        std::cerr << "Warning: Could not create output directory: " << e.what() << "\n";
    }
    
    const std::string prefix = config.write_fields ? (config.output_dir + "developing_channel") : "";
    const int snapshot_freq = (config.num_snapshots > 0) ? 
                              std::max(1, nsteps / config.num_snapshots) : 0;
    
    if (config.verbose && !prefix.empty()) {
        std::cout << "\nWill write " << config.num_snapshots 
                  << " VTK snapshots (every " << snapshot_freq << " steps)\n\n";
    }
    
    // Time integration loop
    ScopedTimer total_timer("Total simulation", true);
    
    int snap_count = 0;
    
    if (config.verbose) {
        std::cout << std::unitbuf;  // Line buffering for immediate output
        if (config.adaptive_dt) {
            std::cout << std::setw(8) << "Step" 
                      << std::setw(15) << "Residual"
                      << std::setw(15) << "Max |u|"
                      << std::setw(12) << "dt"
                      << std::endl;
        } else {
            std::cout << std::setw(8) << "Step" 
                      << std::setw(15) << "Residual"
                      << std::setw(15) << "Max |u|"
                      << std::endl;
        }
    }
    
    for (int step = 1; step <= nsteps; ++step) {
        // Update time step if adaptive
        double dt_used = config.dt;
        if (config.adaptive_dt) {
            dt_used = solver.compute_adaptive_dt();
        }
        
        // Advance one step
        double residual = solver.step();
        
        // Write VTK snapshots
        if (!prefix.empty() && snapshot_freq > 0 && (step % snapshot_freq == 0)) {
            ++snap_count;
            std::string vtk_file = prefix + "_" + std::to_string(snap_count) + ".vtk";
            try {
                solver.write_vtk(vtk_file);
                if (config.verbose) {
                    std::cout << "Wrote snapshot " << snap_count << ": " << vtk_file << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "Warning: Could not write VTK snapshot: " << e.what() << std::endl;
            }
        }
        
        // Console output
        if (config.verbose && (step % config.output_freq == 0)) {
            double max_vel = solver.velocity().max_magnitude();
            if (config.adaptive_dt) {
                std::cout << std::setw(8) << step
                          << std::setw(15) << std::scientific << std::setprecision(3) << residual
                          << std::setw(15) << std::fixed << max_vel
                          << std::setw(12) << std::scientific << std::setprecision(2) << dt_used
                          << std::endl;
            } else {
                std::cout << std::setw(8) << step
                          << std::setw(15) << std::scientific << std::setprecision(3) << residual
                          << std::setw(15) << std::fixed << max_vel
                          << std::endl;
            }
        }
        
        // Check for divergence
        if (std::isnan(residual) || std::isinf(residual)) {
            std::cerr << "\nSolver diverged at step " << step << std::endl;
            break;
        }
    }
    
    // Write final snapshot
    if (!prefix.empty()) {
        std::string final_file = prefix + "_final.vtk";
        try {
            solver.write_vtk(final_file);
            if (config.verbose) {
                std::cout << "\nFinal VTK output: " << final_file << "\n";
                if (config.num_snapshots > 0) {
                    std::cout << "Total VTK snapshots: " << snap_count + 1 
                             << " (" << snap_count << " during + 1 final)\n";
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: Could not write final VTK: " << e.what() << "\n";
        }
    }
    
    total_timer.stop();
    
    // Results
    std::cout << "\n=== Results ===\n";
    std::cout << "Time steps completed: " << nsteps << "\n";
    std::cout << "Bulk velocity: " << std::fixed << std::setprecision(6) 
              << solver.bulk_velocity() << "\n";
    std::cout << "Wall shear stress: " << solver.wall_shear_stress() << "\n";
    std::cout << "Friction velocity u_tau: " << solver.friction_velocity() << "\n";
    std::cout << "Re_tau: " << solver.Re_tau() << "\n";
    
    // Print timing summary
    TimingStats::instance().print_summary();
    
    return 0;
}

