/// @file test_conv_diff_constant.cpp
/// @brief Check convection and diffusion terms for constant velocity
///
/// For constant velocity, both convection and diffusion should be exactly 0.

#include <iostream>
#include <iomanip>
#include <cmath>
#include "solver.hpp"
#include "mesh.hpp"
#include "test_utilities.hpp"

using namespace nncfd;
using nncfd::test::BCPattern;
using nncfd::test::create_velocity_bc;

int main() {
    std::cout << "================================================================\n";
    std::cout << "  Convection/Diffusion Check for Constant Velocity\n";
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
    config.verbose = false;
    // Try different convective schemes
    config.convective_scheme = ConvectiveScheme::Skew;  // Default for periodic

    RANSSolver solver(mesh, config);
    solver.set_velocity_bc(create_velocity_bc(BCPattern::FullyPeriodic));
    solver.set_body_force(0.0, 0.0);

    // Set constant velocity everywhere using fill() to ensure ALL staggered grid
    // faces are initialized (manual loops can miss edge faces due to staggered layout)
    solver.velocity().fill(u_const, v_const);

    std::cout << "Setup:\n";
    std::cout << "  Grid: " << N << "x" << N << "\n";
    std::cout << "  Constant velocity: u=" << u_const << ", v=" << v_const << "\n";
    std::cout << "  Convective scheme: Skew\n";
    std::cout << "  nu = " << config.nu << "\n\n";

    // Access internal conv and diff fields (they're private, but we can run a step
    // and then check the pressure correction which tells us what happened)

    // Before step - check velocity
    std::cout << "Velocity before step:\n";
    double u_max = 0, u_min = 1e30;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            double u = solver.velocity().u(i, j);
            u_max = std::max(u_max, u);
            u_min = std::min(u_min, u);
        }
    }
    std::cout << "  u range: [" << u_min << ", " << u_max << "]\n";
    std::cout << "  Expected: all = " << u_const << "\n\n";

    // Run one step
    solver.sync_to_gpu();
    solver.step();
    solver.sync_from_gpu();

    // Check pressure correction - if conv and diff were 0, this should be ~0
    const auto& p_corr = solver.pressure_correction();

    double p_min = 1e30, p_max = -1e30;
    double p_sum = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double p = p_corr(i, j);
            p_min = std::min(p_min, p);
            p_max = std::max(p_max, p);
            p_sum += p;
        }
    }
    double p_mean = p_sum / (N * N);

    std::cout << "Pressure correction after step:\n";
    std::cout << "  Range: [" << std::scientific << p_min << ", " << p_max << "]\n";
    std::cout << "  Mean: " << p_mean << "\n";
    std::cout << "  Spread (max-min): " << (p_max - p_min) << "\n\n";

    // Check velocity after step
    std::cout << "Velocity after step:\n";
    u_max = 0; u_min = 1e30;
    double u_drift_max = 0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            double u = solver.velocity().u(i, j);
            u_max = std::max(u_max, u);
            u_min = std::min(u_min, u);
            u_drift_max = std::max(u_drift_max, std::abs(u - u_const));
        }
    }
    std::cout << "  u range: [" << u_min << ", " << u_max << "]\n";
    std::cout << "  Max |drift| from " << u_const << ": " << u_drift_max << "\n\n";

    // Sample dp/dx
    int j_mid = mesh.j_begin() + N/2;
    std::cout << "Sample dp/dx at j=" << j_mid << ":\n";
    for (int i = mesh.i_begin(); i < mesh.i_end() - 1; ++i) {
        double dpdx = (p_corr(i+1, j_mid) - p_corr(i, j_mid)) / mesh.dx;
        std::cout << "  dp/dx(" << i << ") = " << std::setprecision(4) << dpdx << "\n";
    }
    std::cout << "\n";

    // Check what drift the dp/dx would cause
    double implied_drift = 0;
    for (int i = mesh.i_begin(); i < mesh.i_end() - 1; ++i) {
        double dpdx = (p_corr(i+1, j_mid) - p_corr(i, j_mid)) / mesh.dx;
        implied_drift = -config.dt * dpdx;
        break;
    }
    std::cout << "Implied velocity drift from dp/dx: " << implied_drift << "\n";
    std::cout << "Actual velocity drift: " << u_drift_max << "\n\n";

    // Analysis
    std::cout << "=== Analysis ===\n\n";
    // Use relaxed tolerance to avoid flaky failures across compilers/platforms
    double tol = 1e-8;
    if (std::abs(p_max - p_min) < tol && u_drift_max < tol) {
        std::cout << "PASS: Pressure correction is constant, no velocity drift\n";
        std::cout << "  This confirms convection and diffusion are 0 for constant velocity.\n";
        return 0;
    } else {
        std::cout << "FAIL: Non-zero pressure gradient for constant velocity\n";
        std::cout << "  This means either:\n";
        std::cout << "  1. Convection term is non-zero (shouldn't be for const velocity)\n";
        std::cout << "  2. Diffusion term is non-zero (shouldn't be for const velocity)\n";
        std::cout << "  3. The Poisson solver has a bug\n";
        std::cout << "  4. Something else is modifying the velocity\n\n";

        std::cout << "  Since standalone MG test passes, the issue is likely in\n";
        std::cout << "  how conv/diff terms are computed or how u* is formed.\n";
        return 1;
    }
}
