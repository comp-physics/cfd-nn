/// @file test_term_values.cpp
/// @brief Print actual values of conv, diff, and other terms for constant velocity

#include <iostream>
#include <iomanip>
#include <cmath>
#include "solver.hpp"
#include "mesh.hpp"

using namespace nncfd;

int main() {
    std::cout << "================================================================\n";
    std::cout << "  Term Value Check for Constant Velocity\n";
    std::cout << "================================================================\n\n";

    const int N = 8;  // Very small grid
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

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::Periodic;
    bc.y_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    // Constant velocity everywhere (including ghost cells)
    for (int j = 0; j <= mesh.Ny + 1; ++j) {
        for (int i = 0; i <= mesh.Nx + 1; ++i) {
            solver.velocity().u(i, j) = u_const;
            solver.velocity().v(i, j) = v_const;
        }
    }

    int i_test = N/2 + mesh.i_begin();
    int j_test = N/2 + mesh.j_begin();

    std::cout << "Initial velocity at (" << i_test << "," << j_test << "):\n";
    std::cout << "  u = " << solver.velocity().u(i_test, j_test) << "\n";
    std::cout << "  v = " << solver.velocity().v(i_test, j_test) << "\n\n";

    // Manually trigger convection and diffusion computation
    // This is tricky because these are typically done inside step()
    // Instead, let's just run one step and check the result

    solver.sync_to_gpu();

    // We can't easily intercept mid-step values without modifying the solver
    // But we can check initial vs final velocity

    solver.step();

    solver.sync_from_gpu();

    std::cout << "After 1 step at (" << i_test << "," << j_test << "):\n";
    std::cout << "  u = " << std::fixed << std::setprecision(10) << solver.velocity().u(i_test, j_test) << "\n";
    std::cout << "  v = " << solver.velocity().v(i_test, j_test) << "\n";

    double u_drift = solver.velocity().u(i_test, j_test) - u_const;
    double v_drift = solver.velocity().v(i_test, j_test) - v_const;
    std::cout << "  u_drift = " << std::scientific << u_drift << "\n";
    std::cout << "  v_drift = " << v_drift << "\n\n";

    // Check the pressure correction field
    const auto& p_corr = solver.pressure_correction();
    std::cout << "Pressure correction at (" << i_test << "," << j_test << "):\n";
    std::cout << "  p_corr = " << p_corr(i_test, j_test) << "\n";

    // Check gradient of pressure correction
    // dp/dx at (i, j) = (p[i] - p[i-1]) / dx
    double dpdx = (p_corr(i_test, j_test) - p_corr(i_test - 1, j_test)) / mesh.dx;
    double dpdy = (p_corr(i_test, j_test) - p_corr(i_test, j_test - 1)) / mesh.dy;
    std::cout << "  dp/dx (approx at cell center) = " << dpdx << "\n";
    std::cout << "  dp/dy (approx at cell center) = " << dpdy << "\n\n";

    // What the correction should be:
    // u_new = u_star - dt * dp/dx
    // u_drift = -dt * dp/dx
    // So expected dp/dx = -u_drift / dt
    double expected_dpdx = -u_drift / config.dt;
    std::cout << "Expected dp/dx (to cause drift): " << expected_dpdx << "\n";
    std::cout << "Actual dp/dx: " << dpdx << "\n\n";

    // Now let's check: if conv and diff were 0 (as they should be for constant field),
    // then u_star = u + dt * (0 + 0 + fx) = u (since fx=0)
    // Then the pressure solve has RHS = div(u_star) / dt = 0
    // So p_corr should be constant (0 after mean subtraction)
    // And u_new = u_star - dt * dp/dx = u - 0 = u

    // But if u_new ≠ u, then either:
    // 1. conv or diff was nonzero, OR
    // 2. dp/dx was nonzero (meaning div(u_star) was nonzero), OR
    // 3. Something else is modifying velocity

    // Check if the pressure correction has variation
    double p_max = -1e30, p_min = 1e30;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            p_max = std::max(p_max, p_corr(i, j));
            p_min = std::min(p_min, p_corr(i, j));
        }
    }
    std::cout << "Pressure correction range: [" << p_min << ", " << p_max << "]\n";
    std::cout << "Pressure range (max-min): " << (p_max - p_min) << "\n\n";

    // If pressure is uniform, dp/dx should be 0
    // Let's sample some pressure gradients across the domain
    std::cout << "Sample dp/dx values:\n";
    for (int j = mesh.j_begin(); j < mesh.j_end(); j += 2) {
        std::cout << "  j=" << j << ": ";
        for (int i = mesh.i_begin(); i < mesh.i_end(); i += 2) {
            double grad = (p_corr(i+1, j) - p_corr(i, j)) / mesh.dx;
            std::cout << std::setw(12) << std::scientific << std::setprecision(3) << grad;
        }
        std::cout << "\n";
    }

    return 0;
}
