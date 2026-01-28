/// @file test_projection_trace.cpp
/// @brief Detailed trace of projection step for constant velocity
///
/// For constant velocity, every intermediate should be 0 or constant.

#include <iostream>
#include <iomanip>
#include <cmath>
#include "solver.hpp"
#include "mesh.hpp"
#include "test_utilities.hpp"

using namespace nncfd;
using nncfd::test::create_velocity_bc;
using nncfd::test::BCPattern;

int main() {
    std::cout << "================================================================\n";
    std::cout << "  Projection Step Trace for Constant Velocity\n";
    std::cout << "================================================================\n\n";

    const int N = 8;
    const double u_const = 1.5;
    const double v_const = 0.75;

    Mesh mesh;
    mesh.init_uniform(N, N, 0.0, 2.0*M_PI, 0.0, 2.0*M_PI);

    Config config;
    config.nu = 0.01;
    config.dt = 0.01;
    config.adaptive_dt = false;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = true;

    RANSSolver solver(mesh, config);
    solver.set_velocity_bc(create_velocity_bc(BCPattern::FullyPeriodic));

    // Set zero body force explicitly
    solver.set_body_force(0.0, 0.0);

    // Constant velocity everywhere using fill() to ensure ALL staggered grid
    // faces are initialized (manual loops can miss edge faces due to staggered layout)
    solver.velocity().fill(u_const, v_const);

    std::cout << "Setup:\n";
    std::cout << "  Grid: " << N << "x" << N << "\n";
    std::cout << "  Domain: [0, 2π]²\n";
    std::cout << "  dx = dy = " << mesh.dx << "\n";
    std::cout << "  nu = " << config.nu << ", dt = " << config.dt << "\n";
    std::cout << "  Body force: (0, 0)\n";
    std::cout << "  Velocity: u = " << u_const << ", v = " << v_const << "\n\n";

    // Verify initial velocity
    std::cout << "Initial velocity check:\n";
    int i_mid = mesh.i_begin() + N/2;
    int j_mid = mesh.j_begin() + N/2;
    std::cout << "  u(" << i_mid << "," << j_mid << ") = " << solver.velocity().u(i_mid, j_mid) << "\n";
    std::cout << "  v(" << i_mid << "," << j_mid << ") = " << solver.velocity().v(i_mid, j_mid) << "\n\n";

    // Check initial divergence (discrete)
    double max_div_initial = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double div = (solver.velocity().u(i+1, j) - solver.velocity().u(i, j)) / mesh.dx
                       + (solver.velocity().v(i, j+1) - solver.velocity().v(i, j)) / mesh.dy;
            max_div_initial = std::max(max_div_initial, std::abs(div));
        }
    }
    std::cout << "Initial max|div(u)| = " << std::scientific << max_div_initial << "\n\n";

    // Check u at faces - should all be constant
    std::cout << "U at faces (row j=" << j_mid << "):\n";
    for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
        std::cout << "  u(" << i << "," << j_mid << ") = " << std::fixed << std::setprecision(10)
                  << solver.velocity().u(i, j_mid) << "\n";
    }
    std::cout << "\n";

    solver.sync_to_gpu();

    // Run ONE step
    std::cout << "Running one step...\n\n";
    solver.step();

    solver.sync_from_gpu();

    // Check the pressure correction
    const auto& p_corr = solver.pressure_correction();
    std::cout << "After step - Pressure correction:\n";
    double p_min = 1e30, p_max = -1e30, p_sum = 0.0;
    int p_count = 0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double p = p_corr(i, j);
            p_min = std::min(p_min, p);
            p_max = std::max(p_max, p);
            p_sum += p;
            p_count++;
        }
    }
    double p_mean = p_sum / p_count;
    std::cout << "  min = " << std::scientific << p_min << "\n";
    std::cout << "  max = " << p_max << "\n";
    std::cout << "  mean = " << p_mean << "\n";
    std::cout << "  range = " << (p_max - p_min) << "\n\n";

    // Sample pressure correction along a row
    std::cout << "Pressure correction along row j=" << j_mid << ":\n";
    for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
        std::cout << "  p_corr(" << i << ") = " << std::fixed << std::setprecision(8) << p_corr(i, j_mid) << "\n";
    }
    std::cout << "\n";

    // Check dp/dx
    std::cout << "dp/dx along row j=" << j_mid << ":\n";
    for (int i = mesh.i_begin(); i < mesh.i_end() - 1; ++i) {
        double dpdx = (p_corr(i+1, j_mid) - p_corr(i, j_mid)) / mesh.dx;
        std::cout << "  dp/dx(" << i << ") = " << std::scientific << dpdx << "\n";
    }
    std::cout << "\n";

    // Check final velocity
    std::cout << "Final velocity:\n";
    std::cout << "  u(" << i_mid << "," << j_mid << ") = " << std::fixed << std::setprecision(10)
              << solver.velocity().u(i_mid, j_mid) << " (expected " << u_const << ")\n";
    std::cout << "  v(" << i_mid << "," << j_mid << ") = " << solver.velocity().v(i_mid, j_mid)
              << " (expected " << v_const << ")\n\n";

    double u_drift = solver.velocity().u(i_mid, j_mid) - u_const;
    double v_drift = solver.velocity().v(i_mid, j_mid) - v_const;
    std::cout << "Velocity drift:\n";
    std::cout << "  u_drift = " << std::scientific << u_drift << "\n";
    std::cout << "  v_drift = " << v_drift << "\n\n";

    // Analysis: what dp/dx would explain the drift?
    // u_new = u* - dt * dp/dx
    // drift = u_new - u_const = (u* - dt*dp/dx) - u_const
    // If u* = u_const (no convection/diffusion), then drift = -dt*dp/dx
    // So dp/dx = -drift/dt
    double implied_dpdx = -u_drift / config.dt;
    double implied_dpdy = -v_drift / config.dt;
    std::cout << "If drift is from projection:\n";
    std::cout << "  implied dp/dx = " << implied_dpdx << "\n";
    std::cout << "  implied dp/dy = " << implied_dpdy << "\n\n";

    // Check the RHS of Poisson (access internal state if possible)
    // The RHS should be div(u*)/dt which should be 0 for constant velocity
    // But we can't easily access it without modifying the solver...

    // Let's just verify the divergence is still 0 at the end
    double max_div_final = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double div = (solver.velocity().u(i+1, j) - solver.velocity().u(i, j)) / mesh.dx
                       + (solver.velocity().v(i, j+1) - solver.velocity().v(i, j)) / mesh.dy;
            max_div_final = std::max(max_div_final, std::abs(div));
        }
    }
    std::cout << "Final max|div(u)| = " << max_div_final << "\n\n";

    std::cout << "=== Analysis ===\n\n";
    if (std::abs(u_drift) < 1e-12 && std::abs(v_drift) < 1e-12) {
        std::cout << "PASS: No velocity drift\n";
        return 0;
    } else {
        std::cout << "FAIL: Velocity drifted from constant value\n";
        std::cout << "  The projection step introduced spurious pressure gradient.\n";
        return 1;
    }
}
