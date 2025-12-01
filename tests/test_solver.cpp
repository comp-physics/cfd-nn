/// Unit tests for RANS solver - Poiseuille validation

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include <iostream>
#include <cmath>
#include <cassert>

using namespace nncfd;

void test_laminar_poiseuille() {
    std::cout << "Testing laminar Poiseuille flow... ";
    
    // Setup - balanced resolution for physics validation in reasonable time
    Mesh mesh;
    mesh.init_uniform(48, 96, 0.0, 4.0, -1.0, 1.0);
    
    Config config;
    config.nu = 0.1;
    config.dp_dx = -1.0;
    config.adaptive_dt = true;
    config.max_iter = 20000;  // Reasonable for CI (converges in ~10-15k)
    config.tol = 1e-6;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;
    // config.use_ssprk3_for_steady = false;  // Use default (true)
    
    RANSSolver solver(mesh, config);
    solver.set_body_force(-config.dp_dx, 0.0);
    solver.initialize_uniform(0.1, 0.0);
    
    auto [residual, iters] = solver.solve_steady();
    
    // Analytical solution: u(y) = -(dp/dx)/(2*nu) * (H^2 - y^2)
    double H = 1.0;
    double u_max_analytical = -config.dp_dx / (2.0 * config.nu) * H * H;
    
    // Check centerline velocity
    const VectorField& vel = solver.velocity();
    double u_centerline = vel.u(mesh.Nx/2, mesh.Ny/2);
    double error = std::abs(u_centerline - u_max_analytical) / u_max_analytical;
    
    // With proper grid and convergence, error should be small
    assert(error < 0.03 && "Poiseuille solution error too large!");
    assert(residual < 1e-5 && "Solver did not converge!");
    
    std::cout << "PASSED (error=" << error*100 << "%, iters=" << iters << ")\n";
}

void test_convergence() {
    std::cout << "Testing solver convergence... ";
    
    Mesh mesh;
    mesh.init_uniform(48, 96, 0.0, 2.0, -1.0, 1.0);
    
    Config config;
    config.nu = 0.05;
    config.adaptive_dt = true;
    config.max_iter = 5000;
    config.tol = 1e-6;
    config.verbose = false;
    // Use default value for use_ssprk3_for_steady
    
    RANSSolver solver(mesh, config);
    solver.set_body_force(-1.0, 0.0);
    solver.initialize_uniform(0.1, 0.0);
    
    auto [residual, iters] = solver.solve_steady();
    
    assert(residual < config.tol && "Solver did not converge!");
    
    std::cout << "PASSED (residual=" << std::scientific << residual 
              << ", iters=" << iters << ")\n";
}

void test_divergence_free() {
    std::cout << "Testing divergence-free constraint... ";
    
    Mesh mesh;
    mesh.init_uniform(64, 128, 0.0, 4.0, -1.0, 1.0);
    
    Config config;
    config.nu = 0.01;
    config.dp_dx = -1.0;
    config.adaptive_dt = true;
    config.max_iter = 20000;
    config.tol = 1e-7;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;
    // Use default value for use_ssprk3_for_steady
    
    RANSSolver solver(mesh, config);
    solver.initialize_uniform(0.1, 0.0);
    solver.set_body_force(-config.dp_dx, 0.0);
    
    auto [residual, iters] = solver.solve_steady();
    assert(residual < 1e-5 && "Solver did not converge!");
    (void)iters;
    
    // Compute divergence of final velocity field
    ScalarField div(mesh);
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double dudx = (solver.velocity().u(i+1, j) - solver.velocity().u(i-1, j)) / (2.0 * mesh.dx);
            double dvdy = (solver.velocity().v(i, j+1) - solver.velocity().v(i, j-1)) / (2.0 * mesh.dy);
            div(i, j) = dudx + dvdy;
        }
    }
    
    // Check max divergence
    double max_div = 0.0;
    double rms_div = 0.0;
    int count = 0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            max_div = std::max(max_div, std::abs(div(i, j)));
            rms_div += div(i, j) * div(i, j);
            ++count;
        }
    }
    rms_div = std::sqrt(rms_div / count);
    
    // Divergence should be very small - incompressibility must be enforced
    assert(max_div < 1e-6 && "Velocity field is not divergence-free!");
    assert(rms_div < 1e-8 && "RMS divergence too large!");
    
    std::cout << "PASSED (max_div=" << std::scientific << max_div << ", rms_div=" << rms_div << ")\n";
}

void test_mass_conservation() {
    std::cout << "Testing mass conservation (periodic channel)... ";
    
    Mesh mesh;
    mesh.init_uniform(64, 128, 0.0, 4.0, -1.0, 1.0);
    
    Config config;
    config.nu = 0.01;
    config.dp_dx = -1.0;
    config.adaptive_dt = true;
    config.max_iter = 1000;
    config.tol = 1e-6;
    config.verbose = false;
    // Use default value for use_ssprk3_for_steady
    
    RANSSolver solver(mesh, config);
    solver.initialize_uniform(0.1, 0.0);
    solver.set_body_force(-config.dp_dx, 0.0);
    
    // Run several time steps
    for (int step = 0; step < 200; ++step) {
        solver.step();
        
        // Check net mass flux through domain
        // For periodic BC in x, mass should be conserved
        // Check flux through a vertical line (should be constant across domain)
        double flux_left = 0.0;
        double flux_right = 0.0;
        
        int i_left = mesh.i_begin();
        int i_right = mesh.i_end() - 1;
        
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            flux_left += solver.velocity().u(i_left, j) * mesh.dy;
            flux_right += solver.velocity().u(i_right, j) * mesh.dy;
        }
        
        // For periodic BC, flux in should equal flux out
        [[maybe_unused]] double flux_diff = std::abs(flux_right - flux_left);
        assert(flux_diff < 1e-10 && "Mass not conserved through periodic boundaries!");
    }
    
    std::cout << "PASSED\n";
}

void test_momentum_balance() {
    std::cout << "Testing momentum balance (Poiseuille)... ";
    
    Mesh mesh;
    mesh.init_uniform(64, 128, 0.0, 4.0, -1.0, 1.0);
    
    Config config;
    config.nu = 0.1;
    config.dp_dx = -1.0;
    config.adaptive_dt = true;
    config.max_iter = 100000;  // Increased for better convergence
    config.tol = 1e-6;  // Relaxed for CI speed while maintaining accuracy
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;
    // Use default value for use_ssprk3_for_steady
    
    RANSSolver solver(mesh, config);
    solver.set_body_force(-config.dp_dx, 0.0);
    solver.initialize_uniform(0.1, 0.0);
    
    auto [residual, iters] = solver.solve_steady();
    assert(residual < 1e-4 && "Solver did not converge!");  // Relaxed slightly
    
    // For steady Poiseuille: analytical solution u(y) = -(dp/dx)/(2*nu) * (H² - y²)
    // Check L2 error across the domain instead of single point
    
    double H = 1.0;
    
    double l2_error = 0.0;
    double l2_norm = 0.0;
    int count = 0;
    
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
    
    // Check that L2 error is reasonable
    assert(rel_l2_error < 0.05 && "Momentum balance L2 error too large!");  // 5% L2 error
    
    std::cout << "PASSED (L2_error=" << rel_l2_error * 100 << "%, iters=" << iters << ")\n";
}

void test_energy_dissipation() {
    std::cout << "Testing energy dissipation rate... ";
    
    // For steady state: Energy input = Energy dissipation
    // Input = (dp/dx) * bulk_velocity * Height
    // Dissipation = nu * integral(|grad(u)|²) dV
    
    Mesh mesh;
    mesh.init_uniform(64, 128, 0.0, 4.0, -1.0, 1.0);
    
    Config config;
    config.nu = 0.1;
    config.dp_dx = -1.0;
    config.adaptive_dt = true;
    config.max_iter = 100000;  // Increased for better convergence
    config.tol = 1e-6;  // Relaxed for CI speed while maintaining accuracy
    config.verbose = false;
    // Use default value for use_ssprk3_for_steady
    
    RANSSolver solver(mesh, config);
    solver.set_body_force(-config.dp_dx, 0.0);
    solver.initialize_uniform(0.1, 0.0);
    
    auto [residual, iters] = solver.solve_steady();
    assert(residual < 1e-4 && "Solver did not converge!");  // Relaxed slightly
    
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
    
    // Energy balance should be good with proper convergence
    assert(energy_balance_error < 0.05 && "Energy balance not satisfied!");
    
    std::cout << "PASSED (error=" << energy_balance_error * 100 << "%, iters=" << iters << ")\n";
}

int main() {
    std::cout << "=== Solver Unit Tests ===\n\n";
    
    test_laminar_poiseuille();
    test_convergence();
    test_divergence_free();
    test_mass_conservation();
    test_momentum_balance();
    test_energy_dissipation();
    
    std::cout << "\nAll solver tests passed!\n";
    return 0;
}

