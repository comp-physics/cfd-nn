/// @file test_mg_physics_match.cpp
/// @brief Verify fixed-cycle MG produces same physics as converged MG
///
/// Runs Taylor-Green vortex with both modes and compares:
/// - max |div(u)| after projection
/// - KE(t) curve
/// - max|u|

#include <iostream>
#include <cmath>
#include <iomanip>
#include <vector>
#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"

using namespace nncfd;

/// Compute total kinetic energy
double compute_KE(const Mesh& mesh, const VectorField& vel) {
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

/// Compute max divergence (measures projection quality)
double compute_max_div(const Mesh& mesh, const VectorField& vel) {
    double max_div = 0.0;
    double idx = 1.0 / mesh.dx;
    double idy = 1.0 / mesh.dy;
    double idz = 1.0 / mesh.dz;

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double div = (vel.u(i+1, j, k) - vel.u(i, j, k)) * idx
                           + (vel.v(i, j+1, k) - vel.v(i, j, k)) * idy
                           + (vel.w(i, j, k+1) - vel.w(i, j, k)) * idz;
                max_div = std::max(max_div, std::abs(div));
            }
        }
    }
    return max_div;
}

/// Compute max velocity magnitude
double compute_max_vel(const Mesh& mesh, const VectorField& vel) {
    double max_vel = 0.0;

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double u = 0.5 * (vel.u(i, j, k) + vel.u(i+1, j, k));
                double v = 0.5 * (vel.v(i, j, k) + vel.v(i, j+1, k));
                double w = 0.5 * (vel.w(i, j, k) + vel.w(i, j, k+1));
                max_vel = std::max(max_vel, std::sqrt(u*u + v*v + w*w));
            }
        }
    }
    return max_vel;
}

/// Run Taylor-Green for N steps with given config
struct RunResult {
    std::vector<double> KE_hist;
    std::vector<double> div_hist;
    std::vector<double> max_vel_hist;
    double final_KE;
    double final_div;
    double final_max_vel;
};

RunResult run_taylor_green(int N, int num_iter, double dt, bool use_fixed_cycles, int fixed_cycles) {
    const double L = 2 * M_PI;
    const double Re = 100.0;
    const double V0 = 1.0;
    const double nu = V0 / Re;

    Mesh mesh;
    mesh.init_uniform(N, N, N, 0.0, L, 0.0, L, 0.0, L);

    Config config;
    config.Nx = N;
    config.Ny = N;
    config.Nz = N;
    config.nu = nu;
    config.dt = dt;
    config.poisson_solver = PoissonSolverType::MG;
    config.verbose = false;

    // Configure MG solver
    if (use_fixed_cycles) {
        config.poisson_fixed_cycles = fixed_cycles;
    } else {
        config.poisson_fixed_cycles = 0;  // Use convergence mode
        config.tol = 1e-6;
        config.poisson_tol_rhs = 1e-4;
    }

    RANSSolver solver(mesh, config);

    // Periodic BCs
    VelocityBC bc;
    bc.x_lo = bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = bc.y_hi = VelocityBC::Periodic;
    bc.z_lo = bc.z_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    // Initialize Taylor-Green
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

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    RunResult result;

    for (int step = 0; step < num_iter; ++step) {
        solver.step();

#ifdef USE_GPU_OFFLOAD
        solver.sync_from_gpu();
#endif
        double KE = compute_KE(mesh, solver.velocity());
        double div = compute_max_div(mesh, solver.velocity());
        double max_vel = compute_max_vel(mesh, solver.velocity());

        result.KE_hist.push_back(KE);
        result.div_hist.push_back(div);
        result.max_vel_hist.push_back(max_vel);
    }

    result.final_KE = result.KE_hist.back();
    result.final_div = result.div_hist.back();
    result.final_max_vel = result.max_vel_hist.back();

    return result;
}

int main() {
    std::cout << "=== MG Physics Match Verification ===\n\n";

    const int N = 64;
    const int num_iter = 20;
    const double dt = 0.01;
    const int fixed_cycles = 8;

    std::cout << "Grid: " << N << "^3, Steps: " << num_iter << ", dt: " << dt << "\n\n";

    // Run with converged MG
    std::cout << "Running converged MG (tol=1e-6, tol_rhs=1e-4)...\n";
    auto ref = run_taylor_green(N, num_iter, dt, false, 0);
    std::cout << "  Final KE: " << std::scientific << ref.final_KE << "\n";
    std::cout << "  Final max|div|: " << ref.final_div << "\n";
    std::cout << "  Final max|u|: " << std::fixed << std::setprecision(6) << ref.final_max_vel << "\n\n";

    // Run with fixed-cycle MG
    std::cout << "Running fixed-cycle MG (" << fixed_cycles << " cycles)...\n";
    auto fixed = run_taylor_green(N, num_iter, dt, true, fixed_cycles);
    std::cout << "  Final KE: " << std::scientific << fixed.final_KE << "\n";
    std::cout << "  Final max|div|: " << fixed.final_div << "\n";
    std::cout << "  Final max|u|: " << std::fixed << std::setprecision(6) << fixed.final_max_vel << "\n\n";

    // Compare
    std::cout << "=== Comparison ===\n";
    double ke_rel_diff = std::abs(fixed.final_KE - ref.final_KE) / ref.final_KE;
    double div_ratio = fixed.final_div / ref.final_div;
    double vel_rel_diff = std::abs(fixed.final_max_vel - ref.final_max_vel) / ref.final_max_vel;

    std::cout << "KE relative difference: " << std::scientific << ke_rel_diff << "\n";
    std::cout << "max|div| ratio (fixed/ref): " << std::fixed << std::setprecision(2) << div_ratio << "\n";
    std::cout << "max|u| relative difference: " << std::scientific << vel_rel_diff << "\n\n";

    // Step-by-step KE comparison
    std::cout << "=== KE History ===\n";
    std::cout << std::setw(6) << "Step" << std::setw(15) << "KE(ref)" << std::setw(15) << "KE(fixed)"
              << std::setw(15) << "Rel Diff" << "\n";
    for (size_t i = 0; i < ref.KE_hist.size(); i += 5) {
        double diff = std::abs(fixed.KE_hist[i] - ref.KE_hist[i]) / ref.KE_hist[i];
        std::cout << std::setw(6) << i+1
                  << std::setw(15) << std::scientific << ref.KE_hist[i]
                  << std::setw(15) << fixed.KE_hist[i]
                  << std::setw(15) << diff << "\n";
    }

    std::cout << "\n=== max|div| History ===\n";
    std::cout << std::setw(6) << "Step" << std::setw(15) << "div(ref)" << std::setw(15) << "div(fixed)" << "\n";
    for (size_t i = 0; i < ref.div_hist.size(); i += 5) {
        std::cout << std::setw(6) << i+1
                  << std::setw(15) << std::scientific << ref.div_hist[i]
                  << std::setw(15) << fixed.div_hist[i] << "\n";
    }

    // Pass/fail
    bool passed = (ke_rel_diff < 1e-6) && (div_ratio < 10.0) && (vel_rel_diff < 1e-6);

    std::cout << "\n=== RESULT ===\n";
    if (passed) {
        std::cout << "PASSED: Fixed-cycle MG produces equivalent physics\n";
        return 0;
    } else {
        std::cout << "FAILED: Significant physics difference detected\n";
        if (ke_rel_diff >= 1e-6) std::cout << "  - KE differs by " << ke_rel_diff << " (threshold: 1e-6)\n";
        if (div_ratio >= 10.0) std::cout << "  - Divergence ratio too high: " << div_ratio << "\n";
        if (vel_rel_diff >= 1e-6) std::cout << "  - max|u| differs by " << vel_rel_diff << " (threshold: 1e-6)\n";
        return 1;
    }
}
