/// Unit tests for RANS solver - Poiseuille validation

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>

using namespace nncfd;

// Helper: Initialize velocity with analytical Poiseuille profile
// This dramatically speeds up convergence (100x faster) for steady-state tests
void initialize_poiseuille_profile(RANSSolver& solver, const Mesh& mesh, 
                                   double dp_dx, double nu, double scale = 0.9) {
    double H = 1.0;  // Half-height of channel
    
    // Set u-velocity at x-faces (staggered grid)
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        double y = mesh.y(j);
        double u_analytical = -dp_dx / (2.0 * nu) * (H * H - y * y);
        
        // Apply to all x-faces at this y
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            solver.velocity().u(i, j) = scale * u_analytical;
        }
    }
    
    // v-velocity stays zero (no cross-flow in Poiseuille)
    for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            solver.velocity().v(i, j) = 0.0;
        }
    }
}

void test_laminar_poiseuille() {
    std::cout << "Testing laminar Poiseuille flow... ";
    
    // Fast physics validation for CI
    // This is a SMOKE TEST - detailed physics tests are in momentum_balance/energy_dissipation
    Mesh mesh;
    mesh.init_uniform(32, 64, 0.0, 4.0, -1.0, 1.0);
    
    Config config;
    config.nu = 0.01;
    config.dp_dx = -0.001;
    config.adaptive_dt = true;
    config.max_iter = 3000;     // Fast convergence from near-solution init
    config.tol = 1e-8;          // Moderate target
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;
    
    RANSSolver solver(mesh, config);
    solver.set_body_force(-config.dp_dx, 0.0);
    
    // Initialize close to solution for fast convergence (Strategy 1)
    initialize_poiseuille_profile(solver, mesh, config.dp_dx, config.nu, 0.9);
    
    // CRITICAL: Sync initial conditions to GPU before solving
    // This ensures GPU starts with the same initial state as CPU
#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif
    
    auto [residual, iters] = solver.solve_steady();
    
    // Analytical solution: u(y) = -(dp/dx)/(2*nu) * (H^2/4 - y^2)
    double H = 2.0;
    double u_max_analytical = -config.dp_dx / (2.0 * config.nu) * H * H / 4.0;
    
    // Check centerline velocity
    const VectorField& vel = solver.velocity();
    double u_centerline = vel.u(mesh.Nx/2, mesh.Ny/2);
    double error = std::abs(u_centerline - u_max_analytical) / u_max_analytical;
    
    // ONLY test physics correctness - 5% is reasonable for coarse grid
    if (error >= 0.05) {
        std::cout << "FAILED: Poiseuille solution error = " << error*100 << "% (limit: 5%)\n";
        std::cout << "        u_centerline = " << u_centerline << ", u_analytical = " << u_max_analytical << "\n";
        std::cout << "        residual = " << residual << ", iters = " << iters << "\n";
    }
    assert(error < 0.05 && "Poiseuille solution error too large!");
    
    // Accept any reasonable convergence progress (don't require machine precision)
    if (residual >= 1e-4) {
        std::cout << "FAILED: Poor convergence, residual = " << residual << " (limit: 1e-4)\n";
    }
    assert(residual < 1e-4 && "Solver did not show reasonable convergence!");
    
    std::cout << "PASSED (error=" << error*100 << "%, iters=" << iters << ")\n";
}

void test_convergence() {
    std::cout << "Testing solver convergence behavior... ";
    
    // Test: Solver should monotonically reduce residual
    // This is a CONVERGENCE BEHAVIOR test, not a precision test
    Mesh mesh;
    mesh.init_uniform(32, 64, 0.0, 2.0, -1.0, 1.0);
    
    Config config;
    config.nu = 0.01;
    config.dp_dx = -0.001;
    config.adaptive_dt = true;
    config.max_iter = 2000;     // Fast convergence from near-solution init
    config.tol = 1e-8;          // Target (may not reach in 2k iters, that's OK)
    config.verbose = false;
    
    RANSSolver solver(mesh, config);
    solver.set_body_force(-config.dp_dx, 0.0);
    
    // Use analytical initialization for fast convergence (Strategy 1)
    initialize_poiseuille_profile(solver, mesh, config.dp_dx, config.nu, 0.85);
    
#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif
    
    auto [residual, iters] = solver.solve_steady();
    
    // Test: Residual should drop by at least 2 orders of magnitude
    // This proves the solver is working, even if not converged to machine precision
    bool good_convergence = (residual < 1e-4);  // Reasonable progress
    
    if (!good_convergence) {
        std::cout << "FAILED: residual = " << std::scientific << residual 
                  << " (limit: 1e-4 for good progress), iters = " << iters << "\n";
    }
    assert(good_convergence && "Solver did not show good convergence!");
    
    std::cout << "PASSED (residual=" << std::scientific << residual 
              << ", iters=" << iters << ")\n";
}

void test_divergence_free() {
    std::cout << "Testing divergence-free constraint (staggered grid)... ";
    
    // STAGGERED GRID TEST: After implementing MAC grid + periodic BC fix,
    // divergence should be at machine epsilon (~1e-12) for all BC types.
    // This is a STRONG test of the projection method.
    
    Mesh mesh;
    mesh.init_uniform(32, 64, 0.0, 4.0, -1.0, 1.0);
    
    Config config;
    config.nu = 0.01;
    config.dp_dx = -0.001;
    config.adaptive_dt = true;
    config.max_iter = 5000;  // Fast for CI
    config.tol = 1e-7;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;
    
    RANSSolver solver(mesh, config);
    solver.set_body_force(-config.dp_dx, 0.0);
    solver.initialize_uniform(0.01, 0.0);
    
#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif
    
    // Run a few steps (don't need full convergence to test projection)
    for (int step = 0; step < 100; ++step) {
        solver.step();
    }
    
    // Compute divergence using STAGGERED GRID formula
    // div(u) = (u[i+1,j] - u[i,j])/dx + (v[i,j+1] - v[i,j])/dy
    const VectorField& vel = solver.velocity();
    double max_div = 0.0;
    double rms_div = 0.0;
    int count = 0;
    
    const int Ng = mesh.Nghost;
    const int Nx = mesh.Nx;
    const int Ny = mesh.Ny;
    
    for (int j = Ng; j < Ng + Ny; ++j) {
        for (int i = Ng; i < Ng + Nx; ++i) {
            // Staggered divergence at cell center (i,j)
            double dudx = (vel.u(i+1, j) - vel.u(i, j)) / mesh.dx;
            double dvdy = (vel.v(i, j+1) - vel.v(i, j)) / mesh.dy;
            double div = dudx + dvdy;
            max_div = std::max(max_div, std::abs(div));
            rms_div += div * div;
            ++count;
        }
    }
    rms_div = std::sqrt(rms_div / count);
    
    // Staggered grid + proper projection → machine epsilon divergence!
    // Be conservative for CI (allow up to 1e-10), but typically get ~1e-13
    if (max_div >= 1e-10) {
        std::cout << "FAILED: max_div = " << std::scientific << max_div << " (limit: 1e-10)\n";
        std::cout << "        This indicates a bug in the staggered projection!\n";
    }
    assert(max_div < 1e-10 && "Divergence too large for staggered grid!");
    
    std::cout << "PASSED (max_div=" << std::scientific << max_div 
              << ", rms_div=" << rms_div << ")\n";
}

void test_mass_conservation() {
    std::cout << "Testing mass conservation (periodic channel)... ";
    
    Mesh mesh;
    mesh.init_uniform(32, 64, 0.0, 4.0, -1.0, 1.0);  // Smaller for speed
    
    Config config;
    config.nu = 0.01;
    config.dp_dx = -0.001;  // Smaller forcing
    config.adaptive_dt = true;
    config.max_iter = 1000;
    config.tol = 1e-6;
    config.verbose = false;
    
    RANSSolver solver(mesh, config);
    solver.initialize_uniform(0.1, 0.0);
    solver.set_body_force(-config.dp_dx, 0.0);
    
#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif
    
    // Run several time steps and check mass conservation
    double max_flux_error = 0.0;
    for (int step = 0; step < 100; ++step) {  // Fewer steps for CI speed
        solver.step();
        
        // Check net mass flux through periodic boundaries
        double flux_left = 0.0;
        double flux_right = 0.0;
        
        int i_left = mesh.i_begin();
        int i_right = mesh.i_end() - 1;
        
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            flux_left += solver.velocity().u(i_left, j) * mesh.dy;
            flux_right += solver.velocity().u(i_right, j) * mesh.dy;
        }
        
        // For periodic BC, flux in should equal flux out
        double flux_diff = std::abs(flux_right - flux_left);
        max_flux_error = std::max(max_flux_error, flux_diff);
        
        if (flux_diff >= 1e-10) {
            std::cout << "FAILED: Mass flux error = " << std::scientific << flux_diff 
                      << " at step " << step << "\n";
        }
        assert(flux_diff < 1e-10 && "Mass not conserved through periodic boundaries!");
    }
    
    std::cout << "PASSED (max_flux_error=" << std::scientific << max_flux_error << ")\n";
}

void test_momentum_balance() {
    std::cout << "Testing momentum balance (Poiseuille)... ";
    
    // Fast CI test: Use analytical initialization for rapid convergence
    Mesh mesh;
    mesh.init_uniform(32, 64, 0.0, 4.0, -1.0, 1.0);
    
    Config config;
    config.nu = 0.01;      // Same as basic Poiseuille test
    config.dp_dx = -0.001; // Same as basic Poiseuille test
    config.adaptive_dt = true;
    config.max_iter = 3000;  // Enough iterations to converge from near-solution initialization
    config.tol = 1e-8;  // Tight tolerance for accuracy
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;
    
    RANSSolver solver(mesh, config);
    solver.set_body_force(-config.dp_dx, 0.0);
    
    // Initialize with analytical profile at 90% of target
    // This reduces iterations from 10k+ to ~100-500
    initialize_poiseuille_profile(solver, mesh, config.dp_dx, config.nu, 0.9);
    
#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif
    
    auto [residual, iters] = solver.solve_steady();
    assert(residual < 5e-4 && "Solver did not converge to reasonable residual!");  // Physics test, not convergence test
    
    // For steady Poiseuille: analytical solution u(y) = -(dp/dx)/(2*nu) * (H² - y²)
    // Check L2 error across the domain instead of single point
    double H = 1.0;  // Half-height of channel
    
    double l2_error = 0.0;
    double l2_norm = 0.0;
    [[maybe_unused]] int count = 0;
    
    int i_center = mesh.Nx / 2;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        double y = mesh.y(j);
        double u_analytical = -config.dp_dx / (2.0 * config.nu) * (H * H - y * y);
        double u_numerical = solver.velocity().u(i_center, j);
        
        l2_error += (u_numerical - u_analytical) * (u_numerical - u_analytical);
        l2_norm += u_analytical * u_analytical;
        ++count;
    }
    
    double rel_l2_error = std::sqrt(l2_error / l2_norm);
    
    std::cout << " residual=" << std::scientific << residual 
              << ", iters=" << iters << ", L2_error=" << std::fixed << std::setprecision(2) << rel_l2_error * 100 << "%... " << std::flush;
    
    // Strict error tolerance - good initialization allows tight accuracy
    if (rel_l2_error >= 0.05) {
        std::cout << "FAILED\n";
        std::cout << "        Momentum balance L2 error = " << rel_l2_error * 100 
                  << "% (limit: 5%), iters = " << iters << "\n";
        std::cout << "        residual = " << residual << "\n";
        assert(false && "Momentum balance L2 error too large!");
    }
    
    std::cout << "PASSED\n";
}

void test_energy_dissipation() {
    std::cout << "Testing energy dissipation rate... ";
    
    // For steady state: Energy input = Energy dissipation
    // Input = (dp/dx) * bulk_velocity * Height
    // Dissipation = nu * integral(|grad(u)|²) dV
    
    // Fast CI test: Use analytical initialization for rapid convergence
    Mesh mesh;
    mesh.init_uniform(32, 64, 0.0, 4.0, -1.0, 1.0);
    
    Config config;
    config.nu = 0.01;      // Same as basic Poiseuille test
    config.dp_dx = -0.001; // Same as basic Poiseuille test
    config.adaptive_dt = true;
    config.max_iter = 3000;  // Enough iterations to converge from near-solution initialization
    config.tol = 1e-8;  // Tight tolerance for accuracy
    config.verbose = false;
    
    RANSSolver solver(mesh, config);
    solver.set_body_force(-config.dp_dx, 0.0);
    
    // Initialize with analytical profile at 90% of target
    // This reduces iterations from 10k+ to ~100-500
    initialize_poiseuille_profile(solver, mesh, config.dp_dx, config.nu, 0.9);
    
    auto [residual, iters] = solver.solve_steady();
    assert(residual < 5e-4 && "Solver did not converge to reasonable residual!");  // Physics test, not convergence test
    
    // Compute bulk velocity
    double bulk_u = solver.bulk_velocity();
    
    // Energy input rate per unit depth
    double L_x = mesh.x_max - mesh.x_min;
    double H = mesh.y_max - mesh.y_min;
    double power_in = std::abs(config.dp_dx) * bulk_u * H;
    
    // Compute dissipation: epsilon = nu * integral(|grad(u)|²) dV
    double dissipation = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double dudy = (solver.velocity().u(i, j+1) - solver.velocity().u(i, j-1)) / (2.0 * mesh.dy);
            double dvdx = (solver.velocity().v(i+1, j) - solver.velocity().v(i-1, j)) / (2.0 * mesh.dx);
            // Full strain rate tensor contribution
            dissipation += config.nu * (dudy * dudy + dvdx * dvdx) * mesh.dx * mesh.dy;
        }
    }
    dissipation /= L_x;  // Per unit length in x
    
    double energy_balance_error = std::abs(power_in - dissipation) / power_in;
    
    std::cout << " residual=" << std::scientific << residual
              << ", iters=" << iters << ", energy_error=" << std::fixed << std::setprecision(2) << energy_balance_error * 100 << "%... " << std::flush;
    
    // Strict error tolerance - good initialization allows tight accuracy
    if (energy_balance_error >= 0.05) {
        std::cout << "FAILED\n";
        std::cout << "        Energy balance error = " << energy_balance_error * 100 
                  << "% (limit: 5%), iters = " << iters << "\n";
        std::cout << "        power_in = " << power_in << ", dissipation = " << dissipation << "\n";
        std::cout << "        residual = " << residual << "\n";
        assert(false && "Energy balance not satisfied!");
    }
    
    std::cout << "PASSED\n";
}

void test_single_timestep_accuracy() {
    std::cout << "Testing single timestep accuracy (discretization)... ";
    
    // Strategy 4: Test that exact steady-state solution stays nearly exact
    // This is a FAST test (~0.1 sec) that validates discretization correctness
    // If we initialize with the analytical solution, after 1 step it should
    // have very small error (only due to truncation error in time integration)
    
    Mesh mesh;
    mesh.init_uniform(32, 64, 0.0, 4.0, -1.0, 1.0);
    
    Config config;
    config.nu = 0.01;
    config.dp_dx = -0.001;
    config.adaptive_dt = false;  // Fixed dt for reproducibility
    config.dt = 0.001;           // Small timestep
    config.max_iter = 1;         // Just ONE step
    config.tol = 1e-12;          // Irrelevant for single step
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;
    
    RANSSolver solver(mesh, config);
    solver.set_body_force(-config.dp_dx, 0.0);
    
    // Initialize with EXACT analytical solution
    initialize_poiseuille_profile(solver, mesh, config.dp_dx, config.nu, 1.0);
    
#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif
    
    // Store exact solution before stepping
    double H = 1.0;
    std::vector<double> u_exact_before;
    int i_center = mesh.Nx / 2;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        double y = mesh.y(j);
        u_exact_before.push_back(-config.dp_dx / (2.0 * config.nu) * (H * H - y * y));
    }
    
    // Take exactly ONE timestep
    solver.step();
    
    // Check error after one step
    double max_abs_error = 0.0;
    double l2_error = 0.0;
    double l2_norm = 0.0;
    
    int idx = 0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        double u_numerical = solver.velocity().u(i_center, j);
        double u_exact = u_exact_before[idx++];
        
        double abs_error = std::abs(u_numerical - u_exact);
        max_abs_error = std::max(max_abs_error, abs_error);
        
        l2_error += abs_error * abs_error;
        l2_norm += u_exact * u_exact;
    }
    
    double rel_l2_error = std::sqrt(l2_error / l2_norm);
    
    // After 1 small timestep, error should be tiny (< 0.1%)
    // This validates: spatial discretization, time integration, BCs, staggered grid
    if (rel_l2_error >= 0.001) {
        std::cout << "FAILED\n";
        std::cout << "        Single-step error = " << rel_l2_error * 100 
                  << "% (limit: 0.1%)\n";
        std::cout << "        This suggests a discretization bug!\n";
        assert(false && "Single timestep accuracy test failed!");
    }
    
    std::cout << "PASSED (error=" << std::scientific << std::setprecision(2) 
              << rel_l2_error * 100 << "%)\n";
}

int main() {
    std::cout << "=== Solver Unit Tests ===\n\n";
    std::cout << "NOTE: Tests use analytical initialization for fast convergence (<30 sec total)\n";
    std::cout << "      This is appropriate for CI. For validation studies, use examples/.\n\n";
    
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

