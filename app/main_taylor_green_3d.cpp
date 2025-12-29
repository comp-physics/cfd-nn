/// 3D Taylor-Green Vortex Solver
/// Classic validation case for incompressible 3D Navier-Stokes solvers
///
/// Initial condition:
///   u =  V₀ sin(x) cos(y) cos(z)
///   v = -V₀ cos(x) sin(y) cos(z)
///   w = 0
///   p = p₀ + (ρV₀²/16)(cos(2x) + cos(2y))(cos(2z) + 2)
///
/// This is divergence-free and decays exponentially for low Re.
/// For higher Re, vortex stretching leads to turbulence transition.
///
/// Key validation: kinetic energy decay rate matches theory (for low Re)

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "config.hpp"
#include "timing.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <vector>

using namespace nncfd;

/// Compute total kinetic energy
double compute_kinetic_energy(const Mesh& mesh, const VectorField& vel) {
    double KE = 0.0;
    double dV = mesh.dx * mesh.dy * mesh.dz;

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double u = 0.5 * (vel.u(i, j, k) + vel.u(i+1, j, k));
                double v = 0.5 * (vel.v(i, j, k) + vel.v(i, j+1, k));
                double w = 0.5 * (vel.w(i, j, k) + vel.w(i, j, k+1));
                KE += 0.5 * (u*u + v*v + w*w) * dV;
            }
        }
    }

    return KE;
}

/// Compute enstrophy (integral of vorticity squared)
double compute_enstrophy(const Mesh& mesh, const VectorField& vel) {
    double ens = 0.0;
    double dV = mesh.dx * mesh.dy * mesh.dz;
    double idx = 1.0 / mesh.dx;
    double idy = 1.0 / mesh.dy;
    double idz = 1.0 / mesh.dz;

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                // Approximate vorticity at cell centers using central differences
                double dwdy = (vel.w(i, j+1, k) - vel.w(i, j, k)) * idy;
                double dvdz = (vel.v(i, j, k+1) - vel.v(i, j, k)) * idz;
                double omega_x = dwdy - dvdz;

                double dudz = (vel.u(i, j, k+1) - vel.u(i, j, k)) * idz;
                double dwdx = (vel.w(i+1, j, k) - vel.w(i, j, k)) * idx;
                double omega_y = dudz - dwdx;

                double dvdx = (vel.v(i+1, j, k) - vel.v(i, j, k)) * idx;
                double dudy = (vel.u(i, j+1, k) - vel.u(i, j, k)) * idy;
                double omega_z = dvdx - dudy;

                ens += (omega_x*omega_x + omega_y*omega_y + omega_z*omega_z) * dV;
            }
        }
    }

    return 0.5 * ens;
}

/// Compute max velocity magnitude
double compute_max_velocity(const Mesh& mesh, const VectorField& vel) {
    double max_vel = 0.0;

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double u = 0.5 * (vel.u(i, j, k) + vel.u(i+1, j, k));
                double v = 0.5 * (vel.v(i, j, k) + vel.v(i, j+1, k));
                double w = 0.5 * (vel.w(i, j, k) + vel.w(i, j, k+1));
                double mag = std::sqrt(u*u + v*v + w*w);
                max_vel = std::max(max_vel, mag);
            }
        }
    }

    return max_vel;
}

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --N N             Grid cells per direction (default: 32)\n";
    std::cout << "  --Re R            Reynolds number (default: 100)\n";
    std::cout << "  --T T             Final time (default: 10.0)\n";
    std::cout << "  --dt T            Time step (default: 0.01)\n";
    std::cout << "  --CFL C           Max CFL for adaptive dt (default: 0.5)\n";
    std::cout << "  --output DIR      Output directory\n";
    std::cout << "  --num_snapshots N Number of VTK snapshots (default: 10)\n";
    std::cout << "  --no_write_fields Skip VTK output\n";
    std::cout << "  --adaptive_dt     Enable adaptive time stepping\n";
}

int main(int argc, char** argv) {
    std::cout << "=== 3D Taylor-Green Vortex Solver ===\n\n";

    // Configuration
    int N = 32;
    double Re = 100.0;
    double T_final = 10.0;
    double dt = 0.01;
    double CFL_max = 0.5;
    bool adaptive_dt = false;
    bool write_fields = true;
    int num_snapshots = 10;
    std::string output_dir = "output/";

    // Initial velocity amplitude
    double V0 = 1.0;

    // Parse command line
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--N" && i+1 < argc) {
            N = std::stoi(argv[++i]);
        } else if (arg == "--Re" && i+1 < argc) {
            Re = std::stod(argv[++i]);
        } else if (arg == "--T" && i+1 < argc) {
            T_final = std::stod(argv[++i]);
        } else if (arg == "--dt" && i+1 < argc) {
            dt = std::stod(argv[++i]);
        } else if (arg == "--CFL" && i+1 < argc) {
            CFL_max = std::stod(argv[++i]);
        } else if (arg == "--output" && i+1 < argc) {
            output_dir = argv[++i];
            if (output_dir.back() != '/') output_dir += '/';
        } else if (arg == "--num_snapshots" && i+1 < argc) {
            num_snapshots = std::stoi(argv[++i]);
        } else if (arg == "--no_write_fields") {
            write_fields = false;
        } else if (arg == "--adaptive_dt") {
            adaptive_dt = true;
        }
    }

    // Compute viscosity from Re
    // Re = V0 * L / nu, with L = 1 (unit box scaled to 2π)
    double L = 1.0;
    double nu = V0 * L / Re;

    // Estimate number of steps
    int max_iter = static_cast<int>(T_final / dt) + 1;

    std::cout << "=== Configuration ===\n";
    std::cout << "Grid: " << N << "³ cells\n";
    std::cout << "Domain: [0, 2π]³ (periodic)\n";
    std::cout << "Re = " << Re << " (nu = " << nu << ")\n";
    std::cout << "V0 = " << V0 << "\n";
    std::cout << "T_final = " << T_final << ", dt = " << dt << "\n";
    std::cout << "Adaptive dt: " << (adaptive_dt ? "YES" : "NO") << "\n";
    std::cout << "=====================\n\n";

    // Theoretical decay rate for low Re: KE(t) = KE(0) * exp(-2*nu*t)
    double decay_rate = 2.0 * nu;
    std::cout << "Theoretical decay rate (low Re): " << decay_rate << "\n";
    std::cout << "Expected KE(T)/KE(0) = " << std::exp(-decay_rate * T_final) << "\n\n";

    // Create 3D mesh
    Mesh mesh;
    mesh.init_uniform(N, N, N, 0.0, 2.0*M_PI, 0.0, 2.0*M_PI, 0.0, 2.0*M_PI);

    std::cout << "Mesh: " << mesh.Nx << " x " << mesh.Ny << " x " << mesh.Nz << " cells\n";
    std::cout << "dx = dy = dz = " << mesh.dx << "\n\n";

    // Configure solver
    Config config;
    config.nu = nu;
    config.dt = dt;
    config.adaptive_dt = adaptive_dt;
    config.CFL_max = CFL_max;
    config.max_iter = max_iter;
    config.tol = 1e-12;  // Won't converge in unsteady mode
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

    // Initialize with Taylor-Green vortex
    // u = V0 * sin(x) * cos(y) * cos(z)
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                double x = mesh.xf[i];
                double y = mesh.y(j);
                double z = mesh.z(k);
                solver.velocity().u(i, j, k) = V0 * std::sin(x) * std::cos(y) * std::cos(z);
            }
        }
    }

    // v = -V0 * cos(x) * sin(y) * cos(z)
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double x = mesh.x(i);
                double y = mesh.yf[j];
                double z = mesh.z(k);
                solver.velocity().v(i, j, k) = -V0 * std::cos(x) * std::sin(y) * std::cos(z);
            }
        }
    }

    // w = 0 (already initialized)

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    // Compute initial kinetic energy
    double KE0 = compute_kinetic_energy(mesh, solver.velocity());
    double ens0 = compute_enstrophy(mesh, solver.velocity());

    std::cout << "Initial KE = " << KE0 << "\n";
    std::cout << "Initial enstrophy = " << ens0 << "\n\n";

    // Output arrays for time history
    std::vector<double> time_hist, ke_hist, ens_hist;
    time_hist.push_back(0.0);
    ke_hist.push_back(KE0);
    ens_hist.push_back(ens0);

    // Snapshot frequency
    int snapshot_freq = (num_snapshots > 0) ? std::max(1, max_iter / num_snapshots) : 0;

    std::cout << "=== Running unsteady simulation ===\n\n";
    std::cout << std::setw(10) << "Step"
              << std::setw(12) << "Time"
              << std::setw(15) << "KE"
              << std::setw(15) << "KE/KE0"
              << std::setw(15) << "Enstrophy"
              << std::setw(12) << "Max |u|"
              << "\n";

    ScopedTimer total_timer("Total simulation", false);

    double t = 0.0;
    int snap_count = 0;

    for (int step = 1; step <= max_iter && t < T_final; ++step) {
        if (adaptive_dt) {
            dt = solver.compute_adaptive_dt();
        }

        double residual = solver.step();
        t += dt;

        // Check for NaN
        if (std::isnan(residual) || std::isinf(residual)) {
            std::cerr << "ERROR: Solver diverged at step " << step << ", t = " << t << "\n";
            return 1;
        }

        // Periodic output
        if (step % 100 == 0 || step == max_iter || t >= T_final) {
#ifdef USE_GPU_OFFLOAD
            solver.sync_from_gpu();
#endif
            double KE = compute_kinetic_energy(mesh, solver.velocity());
            double ens = compute_enstrophy(mesh, solver.velocity());
            double max_u = compute_max_velocity(mesh, solver.velocity());

            time_hist.push_back(t);
            ke_hist.push_back(KE);
            ens_hist.push_back(ens);

            std::cout << std::setw(10) << step
                      << std::setw(12) << std::fixed << std::setprecision(4) << t
                      << std::setw(15) << std::scientific << std::setprecision(6) << KE
                      << std::setw(15) << std::fixed << std::setprecision(6) << KE/KE0
                      << std::setw(15) << std::scientific << ens
                      << std::setw(12) << std::fixed << std::setprecision(4) << max_u
                      << "\n";
        }

        // Write snapshots
        if (write_fields && snapshot_freq > 0 && step % snapshot_freq == 0) {
            ++snap_count;
#ifdef USE_GPU_OFFLOAD
            solver.sync_from_gpu();
#endif
            try {
                std::filesystem::create_directories(output_dir);
                solver.write_vtk(output_dir + "tg3d_" + std::to_string(snap_count) + ".vtk");
            } catch (...) {}
        }
    }

    total_timer.stop();

#ifdef USE_GPU_OFFLOAD
    solver.sync_from_gpu();
#endif

    // Final results
    double KE_final = compute_kinetic_energy(mesh, solver.velocity());
    double ens_final = compute_enstrophy(mesh, solver.velocity());

    std::cout << "\n=== Results ===\n";
    std::cout << "Final time: " << t << "\n";
    std::cout << "Final KE: " << std::scientific << KE_final << "\n";
    std::cout << "KE(T)/KE(0): " << std::fixed << std::setprecision(6) << KE_final/KE0 << "\n";
    std::cout << "Expected (theory): " << std::exp(-decay_rate * t) << "\n";

    double ke_error = std::abs(KE_final/KE0 - std::exp(-decay_rate * t));
    std::cout << "KE decay error: " << ke_error << "\n";

    if (Re <= 100 && ke_error < 0.05) {
        std::cout << "\n*** VALIDATION PASSED: KE decay matches theory (error < 5%) ***\n";
    } else if (Re > 100) {
        std::cout << "\nNote: High Re run - theory not applicable (vortex stretching)\n";
    }

    // Write time history
    if (write_fields) {
        try {
            std::filesystem::create_directories(output_dir);
            std::ofstream hist_file(output_dir + "tg3d_history.dat");
            hist_file << "# time KE KE/KE0 enstrophy\n";
            for (size_t i = 0; i < time_hist.size(); ++i) {
                hist_file << time_hist[i] << " "
                          << ke_hist[i] << " "
                          << ke_hist[i]/KE0 << " "
                          << ens_hist[i] << "\n";
            }
            std::cout << "\nWrote time history to " << output_dir << "tg3d_history.dat\n";

            // Final VTK
            solver.write_vtk(output_dir + "tg3d_final.vtk");
        } catch (const std::exception& e) {
            std::cerr << "Warning: Could not write output: " << e.what() << "\n";
        }
    }

    // Print timing summary
    TimingStats::instance().print_summary();

    return 0;
}
