/// Unit tests for RANS solver - Poiseuille validation
///
/// REFACTORED: Using test_framework.hpp for common helpers
/// Original: 675 lines -> Refactored: ~400 lines

#include "test_framework.hpp"
#include <cmath>
#include <cassert>
#include <vector>
#include <algorithm>

using namespace nncfd;
using namespace nncfd::test;

//=============================================================================
// Test 1: Laminar Poiseuille Flow (Physics Smoke Test)
//=============================================================================
void test_laminar_poiseuille() {
    std::cout << "Testing laminar Poiseuille flow... ";

    Mesh mesh;
    mesh.init_uniform(32, 64, 0.0, 4.0, -1.0, 1.0);

    Config config;
    config.nu = 0.01;
    config.dp_dx = -0.001;
    config.adaptive_dt = true;
    config.max_iter = steady_max_iter();
    config.tol = 1e-8;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;
    config.poisson_max_iter = 50;

    RANSSolver solver(mesh, config);
    solver.set_body_force(-config.dp_dx, 0.0);

    // Initialize close to solution for fast convergence
#ifdef USE_GPU_OFFLOAD
    init_poiseuille(solver, mesh, config.dp_dx, config.nu, 1.0, 0.99);
    solver.sync_to_gpu();
#else
    init_poiseuille(solver, mesh, config.dp_dx, config.nu, 1.0, 0.9);
#endif

    auto [residual, iters] = solver.solve_steady();

    // Analytical solution: u(y) = -(dp/dx)/(2*nu) * (H^2 - y^2)
    double H = 1.0;
    double u_max_analytical = -config.dp_dx / (2.0 * config.nu) * H * H;

    const VectorField& vel = solver.velocity();
    double u_centerline = vel.u(mesh.Nx/2, mesh.Ny/2);
    double error = std::abs(u_centerline - u_max_analytical) / u_max_analytical;

    if (error >= poiseuille_error_limit()) {
        std::cout << "FAILED: error = " << error*100 << "% (limit: " << poiseuille_error_limit()*100 << "%)\n";
        std::exit(1);
    }

    if (residual >= steady_residual_limit()) {
        std::cout << "FAILED: residual = " << residual << " (limit: " << steady_residual_limit() << ")\n";
        std::exit(1);
    }

    std::cout << "PASSED (error=" << error*100 << "%, iters=" << iters << ")\n";
}

//=============================================================================
// Test 2: Convergence Behavior
//=============================================================================
void test_convergence() {
    std::cout << "Testing solver convergence behavior... ";

    Mesh mesh;
    mesh.init_uniform(32, 64, 0.0, 2.0, -1.0, 1.0);

    Config config;
    config.nu = 0.01;
    config.dp_dx = -0.001;
    config.adaptive_dt = true;
    config.max_iter = steady_max_iter();
    config.tol = 1e-8;
    config.verbose = false;
    config.poisson_max_iter = 50;

    RANSSolver solver(mesh, config);
    solver.set_body_force(-config.dp_dx, 0.0);

#ifdef USE_GPU_OFFLOAD
    init_poiseuille(solver, mesh, config.dp_dx, config.nu, 1.0, 0.97);
    solver.sync_to_gpu();
#else
    init_poiseuille(solver, mesh, config.dp_dx, config.nu, 1.0, 0.85);
#endif

    auto [residual, iters] = solver.solve_steady();

    if (residual >= steady_residual_limit()) {
        std::cout << "FAILED: residual = " << std::scientific << residual
                  << " (limit: " << steady_residual_limit() << ")\n";
        std::exit(1);
    }

    std::cout << "PASSED (residual=" << std::scientific << residual
              << ", iters=" << iters << ")\n";
}

//=============================================================================
// Test 3: Divergence-Free Constraint
//=============================================================================
void test_divergence_free() {
    std::cout << "Testing divergence-free constraint (staggered grid)... ";

    Mesh mesh;
    mesh.init_uniform(32, 64, 0.0, 4.0, -1.0, 1.0);

    Config config;
    config.nu = 0.01;
    config.dp_dx = -0.001;
    config.adaptive_dt = true;
    config.max_iter = steady_max_iter();
    config.tol = 1e-7;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;
    config.poisson_max_iter = 50;

    RANSSolver solver(mesh, config);
    solver.set_body_force(-config.dp_dx, 0.0);

    // Initialize with sinusoidal perturbation
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            double x = mesh.xf[i];
            solver.velocity().u(i, j) = 0.01 * (1.0 + 0.1 * std::sin(2.0 * M_PI * x / 4.0));
        }
    }
    for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double x = mesh.x(i);
            solver.velocity().v(i, j) = 0.001 * std::sin(2.0 * M_PI * x / 4.0);
        }
    }

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    for (int step = 0; step < 100; ++step) {
        solver.step();
    }

#ifdef USE_GPU_OFFLOAD
    solver.sync_solution_from_gpu();
#endif

    double max_div = compute_max_divergence(solver.velocity(), mesh);

    double div_limit = 1e-3;
    if (max_div >= div_limit) {
        std::cout << "FAILED: max_div = " << std::scientific << max_div << " (limit: " << div_limit << ")\n";
        std::exit(1);
    }

    std::cout << "PASSED (max_div=" << std::scientific << max_div << ")\n";
}

//=============================================================================
// Test 4: Mass Conservation
//=============================================================================
void test_mass_conservation() {
    std::cout << "Testing incompressibility (periodic flux balance)... ";

    Mesh mesh;
    mesh.init_uniform(32, 64, 0.0, 4.0, -1.0, 1.0);

    Config config;
    config.nu = 0.01;
    config.dp_dx = -0.001;
    config.adaptive_dt = true;
    config.max_iter = 1000;
    config.tol = 1e-6;
    config.verbose = false;
    config.poisson_max_iter = 50;

    RANSSolver solver(mesh, config);
    solver.set_body_force(-config.dp_dx, 0.0);

    // Initialize with Poiseuille + x-perturbation
    double H = 1.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        double y = mesh.y(j);
        double u_base = -config.dp_dx / (2.0 * config.nu) * (H * H - y * y);
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            double x = mesh.xf[i];
            solver.velocity().u(i, j) = 0.9 * u_base * (1.0 + 0.05 * std::sin(2.0 * M_PI * x / 4.0));
        }
    }
    for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            solver.velocity().v(i, j) = 0.0;
        }
    }

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    auto [residual, iters] = solver.solve_steady();

#ifdef USE_GPU_OFFLOAD
    solver.sync_solution_from_gpu();
#endif

    // Check flux at multiple x-planes
    const VectorField& vel = solver.velocity();
    std::vector<double> fluxes;
    for (int i = mesh.i_begin(); i <= mesh.i_end(); i += 4) {
        double flux = 0.0;
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            flux += vel.u(i, j) * mesh.dy;
        }
        fluxes.push_back(flux);
    }

    double mean_flux = 0.0;
    for (double f : fluxes) mean_flux += f;
    mean_flux /= fluxes.size();

    double max_variation = 0.0;
    for (double f : fluxes) {
        max_variation = std::max(max_variation, std::abs(f - mean_flux) / std::abs(mean_flux));
    }

    double var_limit = 0.01;
    if (max_variation >= var_limit) {
        std::cout << "FAILED: flux variation = " << max_variation*100 << "% (limit: " << var_limit*100 << "%)\n";
        std::exit(1);
    }

    std::cout << "PASSED (flux variation=" << max_variation*100 << "%)\n";
}

//=============================================================================
// Test 5: Momentum Balance (via L2 profile error)
//=============================================================================
void test_momentum_balance() {
    std::cout << "Testing momentum balance (Poiseuille)... ";

    Mesh mesh;
    mesh.init_uniform(32, 64, 0.0, 4.0, -1.0, 1.0);

    Config config;
    config.nu = 0.01;
    config.dp_dx = -0.001;
    config.adaptive_dt = true;
    config.max_iter = steady_max_iter();
    config.tol = 1e-8;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;
    config.poisson_max_iter = 50;

    RANSSolver solver(mesh, config);
    solver.set_body_force(-config.dp_dx, 0.0);

#ifdef USE_GPU_OFFLOAD
    init_poiseuille(solver, mesh, config.dp_dx, config.nu, 1.0, 0.99);
    solver.sync_to_gpu();
#else
    init_poiseuille(solver, mesh, config.dp_dx, config.nu, 1.0, 0.9);
#endif

    auto [residual, iters] = solver.solve_steady();

    if (residual >= steady_residual_limit()) {
        std::cout << "FAILED: residual = " << residual << " (limit: " << steady_residual_limit() << ")\n";
        std::exit(1);
    }

    // Check L2 error of velocity profile
    double rel_l2_error = compute_poiseuille_error(solver.velocity(), mesh, config.dp_dx, config.nu);

    std::cout << " residual=" << std::scientific << residual
              << ", iters=" << iters << ", L2_error=" << std::fixed << std::setprecision(2) << rel_l2_error * 100 << "%... " << std::flush;

    if (rel_l2_error >= poiseuille_error_limit()) {
        std::cout << "FAILED\n";
        std::cout << "        L2 error = " << rel_l2_error * 100 << "% (limit: " << poiseuille_error_limit()*100 << "%)\n";
        std::exit(1);
    }

    std::cout << "PASSED\n";
}

//=============================================================================
// Test 6: Energy Dissipation
//=============================================================================
void test_energy_dissipation() {
    std::cout << "Testing kinetic energy dissipation... ";

    Mesh mesh;
    mesh.init_uniform(32, 64, 0.0, 4.0, -1.0, 1.0);

    Config config;
    config.nu = 0.01;
    config.dt = 0.01;
    config.adaptive_dt = false;
    config.max_iter = 100;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    // No forcing - energy should only decrease
    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);

    // Initialize with perturbation away from walls
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        double y = mesh.y(j);
        if (std::abs(y) < 0.8) {
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                solver.velocity().u(i, j) = 0.1 * (1.0 - y*y);
            }
        }
    }

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    double KE_initial = compute_kinetic_energy(mesh, solver.velocity());

    for (int step = 0; step < config.max_iter; ++step) {
        solver.step();
    }

#ifdef USE_GPU_OFFLOAD
    solver.sync_solution_from_gpu();
#endif

    double KE_final = compute_kinetic_energy(mesh, solver.velocity());

    // Energy should decrease (dissipation)
    if (KE_final >= KE_initial) {
        std::cout << "FAILED: energy increased! KE_initial=" << KE_initial << " KE_final=" << KE_final << "\n";
        std::exit(1);
    }

    double dissipation = (KE_initial - KE_final) / KE_initial;
    std::cout << "PASSED (dissipation=" << dissipation*100 << "%)\n";
}

//=============================================================================
// Test 7: Single Timestep Accuracy
//=============================================================================
void test_single_timestep_accuracy() {
    std::cout << "Testing single timestep accuracy... ";

    Mesh mesh;
    mesh.init_uniform(32, 64, 0.0, 4.0, -1.0, 1.0);

    Config config;
    config.nu = 0.01;
    config.dp_dx = -0.001;
    config.dt = 0.001;
    config.adaptive_dt = false;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;
    config.poisson_max_iter = 50;

    RANSSolver solver(mesh, config);
    solver.set_body_force(-config.dp_dx, 0.0);

    // Initialize with exact Poiseuille
    init_poiseuille(solver, mesh, config.dp_dx, config.nu, 1.0, 1.0);

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    double error_before = compute_poiseuille_error(solver.velocity(), mesh, config.dp_dx, config.nu);

    solver.step();

#ifdef USE_GPU_OFFLOAD
    solver.sync_solution_from_gpu();
#endif

    double error_after = compute_poiseuille_error(solver.velocity(), mesh, config.dp_dx, config.nu);

    // Error should stay small (within 1%) for single timestep from exact IC
    // The main goal is to verify solver doesn't blow up
    if (error_after > 0.01) {  // 1% tolerance
        std::cout << "FAILED: error too large after 1 step: " << error_after*100 << "% (limit: 1%)\n";
        std::exit(1);
    }

    std::cout << "PASSED (error: " << std::fixed << std::setprecision(2) << error_before*100
              << "% -> " << error_after*100 << "%)\n";
}

//=============================================================================
// Main
//=============================================================================
int main() {
    std::cout << "=== Solver Unit Tests ===\n\n";
    std::cout << "NOTE: Tests use analytical initialization for fast convergence\n\n";

    test_laminar_poiseuille();
    test_convergence();
    test_divergence_free();
    test_mass_conservation();
    test_single_timestep_accuracy();
    test_momentum_balance();
    test_energy_dissipation();

    std::cout << "\nAll solver tests passed!\n";
    return 0;
}
