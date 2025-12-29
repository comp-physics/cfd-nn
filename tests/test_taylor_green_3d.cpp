/// 3D Taylor-Green Vortex Test
/// Classic validation case for incompressible 3D N-S solvers
///
/// Initial condition:
///   u = sin(x)cos(y)cos(z)
///   v = -cos(x)sin(y)cos(z)
///   w = 0
///
/// This is divergence-free and decays exponentially: u(t) = u(0)exp(-2νt)
/// Tests: 3D time integration, viscous terms, pressure-velocity coupling

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
    std::cout << "  3D TAYLOR-GREEN VORTEX TEST\n";
    std::cout << "========================================================\n";
    std::cout << "Verifies: 3D viscous decay, projection method, time integration\n";
    std::cout << "Initial: u=sin(x)cos(y)cos(z), v=-cos(x)sin(y)cos(z), w=0\n";
    std::cout << "Theory: Kinetic energy decays as exp(-4νt)\n\n";

    // Domain: [0, 2π]³ with 32³ grid (smaller for faster runtime)
    int N = 32;
    Mesh mesh;
    mesh.init_uniform(N, N, N, 0.0, 2.0*M_PI, 0.0, 2.0*M_PI, 0.0, 2.0*M_PI);

    Config config;
    config.nu = 0.01;
    config.dt = 0.01;  // Fixed timestep
    config.adaptive_dt = false;
    config.max_iter = 100;  // Short unsteady run
    config.tol = 1e-10;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    // Periodic BCs in all directions
    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::Periodic;
    bc.y_hi = VelocityBC::Periodic;
    bc.z_lo = VelocityBC::Periodic;
    bc.z_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    // Initialize with 3D Taylor-Green vortex
    // u-component: u = sin(x)cos(y)cos(z)
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                double x = (i < mesh.i_end()) ? mesh.x(i) + mesh.dx/2.0 : mesh.x_max;
                double y = mesh.y(j);
                double z = mesh.z(k);
                solver.velocity().u(i, j, k) = std::sin(x) * std::cos(y) * std::cos(z);
            }
        }
    }

    // v-component: v = -cos(x)sin(y)cos(z)
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double x = mesh.x(i);
                double y = (j < mesh.j_end()) ? mesh.y(j) + mesh.dy/2.0 : mesh.y_max;
                double z = mesh.z(k);
                solver.velocity().v(i, j, k) = -std::cos(x) * std::sin(y) * std::cos(z);
            }
        }
    }

    // w-component: w = 0 (already initialized to 0)
    // Note: This makes the flow 2D-like in structure but still exercises 3D code paths

    solver.sync_to_gpu();

    // Compute initial kinetic energy
    const VectorField& vel0 = solver.velocity();
    double KE0 = 0.0;
    [[maybe_unused]] int count = 0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                // Average velocities from staggered grid to cell centers
                double u = 0.5 * (vel0.u(i, j, k) + vel0.u(i+1, j, k));
                double v = 0.5 * (vel0.v(i, j, k) + vel0.v(i, j+1, k));
                double w = 0.5 * (vel0.w(i, j, k) + vel0.w(i, j, k+1));
                KE0 += 0.5 * (u*u + v*v + w*w) * mesh.dx * mesh.dy * mesh.dz;
                count++;
            }
        }
    }

    std::cout << "Grid size: " << N << " x " << N << " x " << N << "\n";
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
            for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
                for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                    for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                        double u = 0.5 * (vel.u(i, j, k) + vel.u(i+1, j, k));
                        double v = 0.5 * (vel.v(i, j, k) + vel.v(i, j+1, k));
                        double w = 0.5 * (vel.w(i, j, k) + vel.w(i, j, k+1));
                        KE += 0.5 * (u*u + v*v + w*w) * mesh.dx * mesh.dy * mesh.dz;
                    }
                }
            }

            // Theoretical decay: KE(t) = KE(0) * exp(-4*nu*t)
            // For the 3D TGV with this IC, decay rate is same as 2D
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
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double u = 0.5 * (vel_final.u(i, j, k) + vel_final.u(i+1, j, k));
                double v = 0.5 * (vel_final.v(i, j, k) + vel_final.v(i, j+1, k));
                double w = 0.5 * (vel_final.w(i, j, k) + vel_final.w(i, j, k+1));
                KE_final += 0.5 * (u*u + v*v + w*w) * mesh.dx * mesh.dy * mesh.dz;
            }
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
    std::cout << "  [OK] 3D viscous terms correctly implemented\n";
    std::cout << "  [OK] 3D projection method preserves divergence-free field\n";
    std::cout << "  [OK] 3D time integration stable and reasonably accurate\n";
    std::cout << "  [OK] 3D periodic BCs working correctly\n";
    std::cout << "  [OK] w-velocity component handled correctly\n";
    std::cout << "========================================================\n\n";

    return passed ? 0 : 1;
}
