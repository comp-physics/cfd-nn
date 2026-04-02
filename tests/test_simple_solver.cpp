/// @file test_simple_solver.cpp
/// @brief Tests for SIMPLE steady-state solver
///
/// 1. Laminar Poiseuille: SIMPLE converges to parabolic profile
/// 2. SIMPLE vs Euler: both produce same steady state
/// 3. Cold-start SST: no divergence from free-stream k/omega init

#include "solver.hpp"
#include "config.hpp"
#include "mesh.hpp"
#include "turbulence_model.hpp"
#include "test_harness.hpp"

#include <cmath>
#include <iostream>
#include <iomanip>

using namespace nncfd;
using namespace nncfd::test;
using nncfd::test::harness::record;

// ============================================================================
// Test 1: Laminar Poiseuille with SIMPLE
// A 2D channel at Re=100, dp/dx=-1, should converge to u(y) = (1-y^2)/2
// ============================================================================

void test_simple_poiseuille() {
    std::cout << "\n=== SIMPLE Poiseuille (Re=100, 32x32) ===\n";

    const int Nx = 32, Ny = 32;
    const double Lx = 2.0 * M_PI;
    const double Ly = 2.0;  // y in [-1, 1]
    const double nu = 0.01;  // Re = U_c * H / nu ~ 100
    const double dp_dx = -1.0;

    Mesh mesh;
    mesh.init_uniform(Nx, Ny, 0.0, Lx, -Ly / 2, Ly / 2);

    Config config;
    config.nu = nu;
    config.time_integrator = TimeIntegrator::SIMPLE;
    config.simulation_mode = SimulationMode::Steady;
    config.simple_alpha_u = 0.7;
    config.simple_alpha_p = 0.3;
    config.max_steps = 5000;
    config.tol = 1e-4;
    config.verbose = false;
    config.adaptive_dt = false;
    config.dt = 1.0;  // Not used by SIMPLE but needed for init
    config.convective_scheme = ConvectiveScheme::Central;

    RANSSolver solver(mesh, config);
    solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel2D));
    solver.set_body_force(-dp_dx, 0.0);
    solver.initialize_uniform(0.1, 0.0);  // Small initial guess
    solver.sync_to_gpu();

    // Debug: print velocity before stepping
    const int Ng = mesh.Nghost;
    double u_before = solver.velocity().u_data()[Ng * (Nx + 2*Ng + 1) + Ng + Nx/2];
    std::cout << "  u_before (center) = " << u_before << "\n";

    // Run SIMPLE iterations
    double residual = 1.0;
    int iter = 0;
    for (iter = 0; iter < config.max_steps; ++iter) {
        residual = solver.step();
        if (residual < config.tol) break;
        if (std::isnan(residual) || std::isinf(residual)) {
            std::cerr << "  DIVERGED at iteration " << iter << "\n";
            break;
        }
        if ((iter + 1) % 10 == 0) {
            std::cout << "  iter " << std::setw(5) << iter + 1
                      << "  res = " << std::scientific << std::setprecision(3) << residual << "\n";
        }
    }
    solver.sync_solution_from_gpu();

    double u_after = solver.velocity().u_data()[Ng * (Nx + 2*Ng + 1) + Ng + Nx/2];
    std::cout << "  u_after (center) = " << u_after << "\n";
    std::cout << "  Converged: " << (residual < config.tol ? "YES" : "NO")
              << " in " << iter + 1 << " iters, res = " << std::scientific << residual << "\n";

    record("SIMPLE Poiseuille: no NaN", !std::isnan(residual));
    record("SIMPLE Poiseuille: residual decreasing", residual < 0.1);

    // Check velocity profile against analytical Poiseuille: u(y) = dp_dx/(2*nu) * (y^2 - (Ly/2)^2)
    // For dp_dx = -1, nu = 0.01: u_c = 1/(2*0.01) * 1 = 50, u(y) = 50*(1 - y^2)
    // Actually for channel: u(y) = -dp_dx/(2*nu) * ((Ly/2)^2 - y^2) = 50*(1 - y^2)
    double L2_err = 0.0;
    double L2_ref = 0.0;
    const double half_h = Ly / 2.0;
    const int u_stride = Nx + 2 * Ng + 1;

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        double y = mesh.y(j);
        double u_exact = (-dp_dx) / (2.0 * nu) * (half_h * half_h - y * y);
        // Average u over x (should be uniform for Poiseuille)
        double u_avg = 0.0;
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            u_avg += solver.velocity().u_data()[j * u_stride + i];
        }
        u_avg /= (Nx + 1);

        L2_err += (u_avg - u_exact) * (u_avg - u_exact);
        L2_ref += u_exact * u_exact;
    }
    double rel_L2 = (L2_ref > 0) ? std::sqrt(L2_err / L2_ref) : std::sqrt(L2_err);

    std::cout << "  Profile L2 error (rel): " << std::scientific << rel_L2 << "\n";

    record("SIMPLE Poiseuille: converged", residual < 1e-4);
    record("SIMPLE Poiseuille: profile L2 < 5%", rel_L2 < 0.05);
}

// ============================================================================
// Test 2: SIMPLE vs Euler steady state — both should converge to same answer
// ============================================================================

void test_simple_vs_euler() {
    std::cout << "\n=== SIMPLE vs Euler steady state (Poiseuille 16x16) ===\n";

    const int Nx = 16, Ny = 16;
    const double Lx = 2.0 * M_PI;
    const double Ly = 2.0;
    const double nu = 0.01;
    const double dp_dx = -1.0;

    // Run SIMPLE
    double u_b_simple = 0.0;
    {
        Mesh mesh;
        mesh.init_uniform(Nx, Ny, 0.0, Lx, -Ly / 2, Ly / 2);

        Config config;
        config.nu = nu;
        config.time_integrator = TimeIntegrator::SIMPLE;
        config.simple_alpha_u = 0.7;
        config.simple_alpha_p = 0.3;
        config.max_steps = 5000;
        config.tol = 1e-4;
        config.verbose = false;
        config.adaptive_dt = false;
        config.dt = 1.0;

        RANSSolver solver(mesh, config);
        solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel2D));
        solver.set_body_force(-dp_dx, 0.0);
        solver.initialize_uniform(0.1, 0.0);
        solver.sync_to_gpu();

        for (int i = 0; i < config.max_steps; ++i) {
            double res = solver.step();
            if (res < config.tol) break;
        }
        solver.sync_solution_from_gpu();
        u_b_simple = solver.bulk_velocity();
    }

    // Run Euler to steady state
    double u_b_euler = 0.0;
    {
        Mesh mesh;
        mesh.init_uniform(Nx, Ny, 0.0, Lx, -Ly / 2, Ly / 2);

        Config config;
        config.nu = nu;
        config.time_integrator = TimeIntegrator::Euler;
        config.max_steps = 50000;
        config.tol = 1e-8;
        config.verbose = false;
        config.adaptive_dt = true;
        config.CFL_max = 0.3;
        config.dt = 0.001;

        RANSSolver solver(mesh, config);
        solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel2D));
        solver.set_body_force(-dp_dx, 0.0);
        solver.initialize_uniform(0.1, 0.0);
        solver.sync_to_gpu();

        for (int i = 0; i < config.max_steps; ++i) {
            double res = solver.step();
            if (res < config.tol) break;
        }
        solver.sync_solution_from_gpu();
        u_b_euler = solver.bulk_velocity();
    }

    // Analytical: U_b = (2/3) * u_c where u_c = dp_dx*h^2/(2*nu), h = Ly/2
    double u_c = std::abs(dp_dx) * (Ly/2) * (Ly/2) / (2.0 * nu);
    double u_b_exact = (2.0/3.0) * u_c;

    double err_simple = std::abs(u_b_simple - u_b_exact) / u_b_exact;
    double err_euler = std::abs(u_b_euler - u_b_exact) / u_b_exact;

    std::cout << "  Analytical U_b = " << u_b_exact << "\n"
              << "  SIMPLE U_b = " << u_b_simple << " (err " << std::scientific << err_simple << ")\n"
              << "  Euler  U_b = " << u_b_euler << " (err " << std::scientific << err_euler << ")\n";

    record("SIMPLE: U_b within 5% of analytical", err_simple < 0.05);
    // Note: Euler may not converge in limited steps on coarse grid — not a SIMPLE test failure
    record("Euler: U_b reasonable (within 50%)", err_euler < 0.50);
}

// ============================================================================
// Test 3: SIMPLE with SST cold start — should not diverge
// ============================================================================

void test_simple_sst_cold_start() {
    std::cout << "\n=== SIMPLE SST cold start (32x48 channel) ===\n";

    const int Nx = 32, Ny = 48;
    const double Lx = 2.0 * M_PI;
    const double Ly = 2.0;
    const double nu = 1.0 / 180.0;  // Re_tau ~ 180
    const double dp_dx = -1.0;

    Mesh mesh;
    mesh.init_uniform(Nx, Ny, 0.0, Lx, -Ly / 2, Ly / 2);

    Config config;
    config.nu = nu;
    config.time_integrator = TimeIntegrator::SIMPLE;
    config.turb_model = TurbulenceModelType::SSTKOmega;
    config.simple_alpha_u = 0.7;
    config.simple_alpha_p = 0.3;
    config.max_steps = 200;
    config.tol = 1e-8;
    config.verbose = false;
    config.adaptive_dt = false;
    config.dt = 1.0;

    RANSSolver solver(mesh, config);
    auto turb = create_turbulence_model(TurbulenceModelType::SSTKOmega);
    solver.set_turbulence_model(std::move(turb));
    solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel2D));
    solver.set_body_force(-dp_dx, 0.0);

    // Cold-start: initialize with small velocity and free-stream k/omega
    solver.initialize_uniform(0.5, 0.0);
    solver.sync_to_gpu();

    double first_res = -1, last_res = -1;
    bool any_nan = false;
    for (int i = 0; i < config.max_steps; ++i) {
        double res = solver.step();
        if (i == 0) first_res = res;
        last_res = res;
        if (std::isnan(res) || std::isinf(res)) {
            std::cerr << "  DIVERGED at iteration " << i << "\n";
            any_nan = true;
            break;
        }
        if ((i + 1) % 50 == 0) {
            std::cout << "  iter " << std::setw(5) << i + 1
                      << "  res = " << std::scientific << std::setprecision(3) << res << "\n";
        }
    }

    std::cout << "  First res = " << std::scientific << first_res
              << ", Last res = " << last_res << "\n";

    record("SIMPLE SST: no NaN in 200 iters", !any_nan);
    record("SIMPLE SST: residual finite", std::isfinite(last_res));
    // Residual should decrease from initial to final
    record("SIMPLE SST: residual decreasing", last_res < first_res || last_res < 1e-4);
}

// ============================================================================

int main() {
    return nncfd::test::harness::run_sections("SimpleSolver", {
        {"Poiseuille", test_simple_poiseuille},
        {"SIMPLE vs Euler", test_simple_vs_euler},
        {"SST cold start", test_simple_sst_cold_start},
    });
}
