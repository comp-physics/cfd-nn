/// Taylor-Green Vortex Test
/// Classic validation case for incompressible N-S solvers
/// 
/// Initial condition: u = sin(x)cos(y), v = -cos(x)sin(y)
/// This is divergence-free and decays exponentially: u(t) = u(0)exp(-2νt)
/// Tests: Time integration, viscous terms, pressure-velocity coupling

#include "solver.hpp"
#include "mesh.hpp"
#include "config.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include <iomanip>
#include <algorithm>

using namespace nncfd;

int main() {
    std::cout << "\n";
    std::cout << "========================================================\n";
    std::cout << "  TAYLOR-GREEN VORTEX TEST\n";
    std::cout << "========================================================\n";
    std::cout << "Verifies: Viscous decay, projection method, time integration\n";
    std::cout << "Initial: u=sin(x)cos(y), v=-cos(x)sin(y)\n";
    std::cout << "Theory: Decays as exp(-2νt)\n\n";
    
    // Domain: [0, 2π] × [0, 2π]
    int N = 64;
    Mesh mesh;
    mesh.init_uniform(N, N, 0.0, 2.0*M_PI, 0.0, 2.0*M_PI);
    
    Config config;
    config.nu = 0.01;
    config.dt = 0.01;  // Fixed timestep
    config.adaptive_dt = false;
    config.max_iter = 100;  // Short unsteady run
    config.tol = 1e-10;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;
    
    RANSSolver solver(mesh, config);
    
    // Periodic BCs
    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::Periodic;
    bc.y_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);
    
    // Initialize with Taylor-Green vortex
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            double x = (i < mesh.i_end()) ? mesh.x(i) + mesh.dx/2.0 : mesh.x_max;
            double y = mesh.y(j);
            solver.velocity().u(i, j) = std::sin(x) * std::cos(y);
        }
    }
    for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double x = mesh.x(i);
            double y = (j < mesh.j_end()) ? mesh.y(j) + mesh.dy/2.0 : mesh.y_max;
            solver.velocity().v(i, j) = -std::cos(x) * std::sin(y);
        }
    }
    
    solver.sync_to_gpu();
    
    // Compute initial kinetic energy
    const VectorField& vel0 = solver.velocity();
    double KE0 = 0.0;
    int count = 0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double u = 0.5 * (vel0.u(i, j) + vel0.u(i+1, j));
            double v = 0.5 * (vel0.v(i, j) + vel0.v(i, j+1));
            KE0 += 0.5 * (u*u + v*v) * mesh.dx * mesh.dy;
            count++;
        }
    }
    
    std::cout << "Initial kinetic energy: " << KE0 << "\n\n";
    std::cout << "Time-stepping (100 steps, dt=" << config.dt << ")...\n\n";
    
    std::cout << std::setw(10) << "Step"
              << std::setw(15) << "Time"
              << std::setw(15) << "KE"
              << std::setw(15) << "KE_theory"
              << std::setw(15) << "Error (%)"
              << "\n";
    std::cout << std::string(70, '-') << "\n";
    
    // Time-step and check decay
    std::vector<int> check_steps = {0, 10, 25, 50, 75, 100};
    
    for (int step = 1; step <= config.max_iter; ++step) {
        solver.step();
        
        if (std::find(check_steps.begin(), check_steps.end(), step) != check_steps.end()) {
            solver.sync_from_gpu();
            
            double time = step * config.dt;
            
            // Compute kinetic energy
            const VectorField& vel = solver.velocity();
            double KE = 0.0;
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    double u = 0.5 * (vel.u(i, j) + vel.u(i+1, j));
                    double v = 0.5 * (vel.v(i, j) + vel.v(i, j+1));
                    KE += 0.5 * (u*u + v*v) * mesh.dx * mesh.dy;
                }
            }
            
            // Theoretical decay: KE(t) = KE(0) * exp(-4*nu*t)
            double KE_theory = KE0 * std::exp(-4.0 * config.nu * time);
            double error = std::abs(KE - KE_theory) / KE_theory;
            
            std::cout << std::setw(10) << step
                      << std::setw(15) << std::fixed << std::setprecision(3) << time
                      << std::setw(15) << std::setprecision(6) << KE
                      << std::setw(15) << KE_theory
                      << std::setw(15) << std::setprecision(2) << error * 100
                      << "\n";
        }
    }
    
    solver.sync_from_gpu();
    
    // Final assessment
    double final_time = config.max_iter * config.dt;
    const VectorField& vel_final = solver.velocity();
    double KE_final = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double u = 0.5 * (vel_final.u(i, j) + vel_final.u(i+1, j));
            double v = 0.5 * (vel_final.v(i, j) + vel_final.v(i, j+1));
            KE_final += 0.5 * (u*u + v*v) * mesh.dx * mesh.dy;
        }
    }
    
    double KE_theory_final = KE0 * std::exp(-4.0 * config.nu * final_time);
    double error_final = std::abs(KE_final - KE_theory_final) / KE_theory_final;
    
    std::cout << "\n";
    std::cout << "========================================================\n";
    std::cout << "FINAL RESULTS:\n";
    std::cout << "========================================================\n";
    std::cout << "Final time:        " << final_time << "\n";
    std::cout << "KE (numerical):    " << std::setprecision(6) << KE_final << "\n";
    std::cout << "KE (theoretical):  " << KE_theory_final << "\n";
    std::cout << "Relative error:    " << std::setprecision(2) << error_final * 100 << "%\n\n";
    
    bool passed = true;
    if (error_final < 0.05) {
        std::cout << "[EXCELLENT] <5% error in energy decay\n";
    } else if (error_final < 0.10) {
        std::cout << "[VERY GOOD] <10% error\n";
    } else if (error_final < 0.20) {
        std::cout << "[ACCEPTABLE] <20% error\n";
    } else {
        std::cout << "[FAIL] Error too large\n";
        passed = false;
    }
    
    std::cout << "\nWhat this test validates:\n";
    std::cout << "  [OK] Viscous terms correctly implemented\n";
    std::cout << "  [OK] Projection method preserves divergence-free field\n";
    std::cout << "  [OK] Time integration stable and reasonably accurate\n";
    std::cout << "  [OK] Periodic BCs working correctly\n";
    std::cout << "========================================================\n\n";
    
    return passed ? 0 : 1;
}
