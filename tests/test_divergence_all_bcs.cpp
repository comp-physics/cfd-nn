/// Comprehensive divergence tests for staggered grid with various boundary conditions
/// Verifies that the periodic BC fix and staggered grid implementation
/// achieve machine-epsilon divergence for all supported BC combinations

#include "solver.hpp"
#include "config.hpp"
#include "mesh.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>

using namespace nncfd;

/// Compute max and RMS divergence using staggered grid formula
void compute_divergence_stats(const Mesh& mesh, const VectorField& vel,
                               double& max_div, double& rms_div) {
    max_div = 0.0;
    rms_div = 0.0;
    int count = 0;
    
    const int Ng = mesh.Nghost;
    const int Nx = mesh.Nx;
    const int Ny = mesh.Ny;
    
    for (int j = Ng; j < Ng + Ny; ++j) {
        for (int i = Ng; i < Ng + Nx; ++i) {
            // Staggered divergence: (u[i+1] - u[i])/dx + (v[j+1] - v[j])/dy
            double dudx = (vel.u(i+1, j) - vel.u(i, j)) / mesh.dx;
            double dvdy = (vel.v(i, j+1) - vel.v(i, j)) / mesh.dy;
            double div = dudx + dvdy;
            
            max_div = std::max(max_div, std::abs(div));
            rms_div += div * div;
            ++count;
        }
    }
    
    rms_div = std::sqrt(rms_div / count);
}

/// Test 1: Fully periodic domain (Taylor-Green)
void test_divergence_periodic_periodic() {
    std::cout << "\n=== Test 1: Fully Periodic BCs (Taylor-Green) ===" << std::endl;
    
    Config config;
    config.Nx = 64;
    config.Ny = 64;
    config.x_min = 0.0;
    config.x_max = 2.0 * M_PI;
    config.y_min = 0.0;
    config.y_max = 2.0 * M_PI;
    config.nu = 0.01;
    config.dt = 0.0001;
    config.adaptive_dt = false;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;
    
    Mesh mesh;
    mesh.init_uniform(config.Nx, config.Ny,
                      config.x_min, config.x_max,
                      config.y_min, config.y_max);
    
    RANSSolver solver(mesh, config);
    VelocityBC bc;
    bc.x_lo = bc.x_hi = bc.y_lo = bc.y_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);
    
    // Initialize with Taylor-Green vortex
    VectorField vel_init(mesh);
    const int Ng = mesh.Nghost;
    
    for (int j = Ng; j < Ng + mesh.Ny; ++j) {
        for (int i = Ng; i <= Ng + mesh.Nx; ++i) {
            double x = mesh.x_min + (i - Ng) * mesh.dx;
            double y = mesh.y(j);
            vel_init.u(i, j) = -std::cos(x) * std::sin(y);
        }
    }
    for (int j = Ng; j <= Ng + mesh.Ny; ++j) {
        for (int i = Ng; i < Ng + mesh.Nx; ++i) {
            double x = mesh.x(i);
            double y = mesh.y_min + (j - Ng) * mesh.dy;
            vel_init.v(i, j) = std::sin(x) * std::cos(y);
        }
    }
    solver.initialize(vel_init);
    
    // Initial divergence should already be machine epsilon
    double max_div_init, rms_div_init;
    compute_divergence_stats(mesh, solver.velocity(), max_div_init, rms_div_init);
    
    std::cout << "  Initial divergence:\n";
    std::cout << "    max: " << std::scientific << std::setprecision(3) << max_div_init << "\n";
    std::cout << "    rms: " << rms_div_init << "\n";
    
    assert(max_div_init < 1e-12 && "Initial divergence should be ~0 for Taylor-Green!");
    
    // Run 10 steps
    std::cout << "  Running 10 time steps...\n";
    for (int step = 0; step < 10; ++step) {
        solver.step();
    }
    
    // Check divergence after evolution
    double max_div, rms_div;
    compute_divergence_stats(mesh, solver.velocity(), max_div, rms_div);
    
    std::cout << "  Divergence after 10 steps:\n";
    std::cout << "    max: " << std::scientific << max_div << "\n";
    std::cout << "    rms: " << rms_div << "\n";
    
    // With staggered grid + periodic BC fix, should be at machine epsilon
    assert(max_div < 1e-10 && "Divergence too large for periodic domain!");
    
    std::cout << "  ✓ PASSED\n";
}

/// Test 2: Periodic-X, Wall-Y (Channel flow)
void test_divergence_periodic_wall() {
    std::cout << "\n=== Test 2: Periodic-X, Wall-Y (Channel) ===" << std::endl;
    
    Config config;
    config.Nx = 64;
    config.Ny = 32;
    config.x_min = 0.0;
    config.x_max = 4.0;
    config.y_min = -1.0;
    config.y_max = 1.0;
    config.nu = 0.01;
    config.dp_dx = -0.001;
    config.dt = 0.001;
    config.adaptive_dt = false;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;
    
    Mesh mesh;
    mesh.init_uniform(config.Nx, config.Ny,
                      config.x_min, config.x_max,
                      config.y_min, config.y_max);
    
    RANSSolver solver(mesh, config);
    VelocityBC bc;
    bc.x_lo = bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);
    solver.set_body_force(-config.dp_dx, 0.0);
    solver.initialize_uniform(0.1, 0.0);
    
    // Run 20 steps
    std::cout << "  Running 20 time steps...\n";
    for (int step = 0; step < 20; ++step) {
        solver.step();
    }
    
    // Check divergence
    double max_div, rms_div;
    compute_divergence_stats(mesh, solver.velocity(), max_div, rms_div);
    
    std::cout << "  Divergence after 20 steps:\n";
    std::cout << "    max: " << std::scientific << max_div << "\n";
    std::cout << "    rms: " << rms_div << "\n";
    
    // Should also be at machine epsilon
    assert(max_div < 1e-10 && "Divergence too large for channel flow!");
    
    std::cout << "  ✓ PASSED\n";
}

/// Test 3: Wall-X, Periodic-Y (Spanwise periodic)
void test_divergence_wall_periodic() {
    std::cout << "\n=== Test 3: Wall-X, Periodic-Y (Spanwise) ===" << std::endl;
    
    Config config;
    config.Nx = 32;
    config.Ny = 64;
    config.x_min = -1.0;
    config.x_max = 1.0;
    config.y_min = 0.0;
    config.y_max = 4.0;
    config.nu = 0.01;
    config.dt = 0.001;
    config.adaptive_dt = false;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;
    
    Mesh mesh;
    mesh.init_uniform(config.Nx, config.Ny,
                      config.x_min, config.x_max,
                      config.y_min, config.y_max);
    
    RANSSolver solver(mesh, config);
    VelocityBC bc;
    bc.x_lo = bc.x_hi = VelocityBC::NoSlip;
    bc.y_lo = bc.y_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);
    solver.set_body_force(0.0, -0.001);  // y-direction forcing
    solver.initialize_uniform(0.0, 0.1);
    
    // Run 20 steps
    std::cout << "  Running 20 time steps...\n";
    for (int step = 0; step < 20; ++step) {
        solver.step();
    }
    
    // Check divergence
    double max_div, rms_div;
    compute_divergence_stats(mesh, solver.velocity(), max_div, rms_div);
    
    std::cout << "  Divergence after 20 steps:\n";
    std::cout << "    max: " << std::scientific << max_div << "\n";
    std::cout << "    rms: " << rms_div << "\n";
    
    assert(max_div < 1e-10 && "Divergence too large for spanwise periodic!");
    
    std::cout << "  ✓ PASSED\n";
}

/// Test 4: All walls (lid-driven cavity-like)
void test_divergence_all_walls() {
    std::cout << "\n=== Test 4: All Walls (Cavity-like) ===" << std::endl;
    
    Config config;
    config.Nx = 32;
    config.Ny = 32;
    config.x_min = 0.0;
    config.x_max = 1.0;
    config.y_min = 0.0;
    config.y_max = 1.0;
    config.nu = 0.01;
    config.dt = 0.001;
    config.adaptive_dt = false;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;
    
    Mesh mesh;
    mesh.init_uniform(config.Nx, config.Ny,
                      config.x_min, config.x_max,
                      config.y_min, config.y_max);
    
    RANSSolver solver(mesh, config);
    VelocityBC bc;
    bc.x_lo = bc.x_hi = VelocityBC::NoSlip;
    bc.y_lo = bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);
    
    // Initialize with some internal circulation
    VectorField vel_init(mesh);
    const int Ng = mesh.Nghost;
    for (int j = Ng; j < Ng + mesh.Ny; ++j) {
        for (int i = Ng; i <= Ng + mesh.Nx; ++i) {
            double x = mesh.x_min + (i - Ng) * mesh.dx;
            double y = mesh.y(j);
            // Small internal perturbation
            vel_init.u(i, j) = 0.01 * std::sin(M_PI * x) * std::cos(M_PI * y);
        }
    }
    for (int j = Ng; j <= Ng + mesh.Ny; ++j) {
        for (int i = Ng; i < Ng + mesh.Nx; ++i) {
            double x = mesh.x(i);
            double y = mesh.y_min + (j - Ng) * mesh.dy;
            vel_init.v(i, j) = -0.01 * std::cos(M_PI * x) * std::sin(M_PI * y);
        }
    }
    solver.initialize(vel_init);
    
    // Run 20 steps
    std::cout << "  Running 20 time steps...\n";
    for (int step = 0; step < 20; ++step) {
        solver.step();
    }
    
    // Check divergence
    double max_div, rms_div;
    compute_divergence_stats(mesh, solver.velocity(), max_div, rms_div);
    
    std::cout << "  Divergence after 20 steps:\n";
    std::cout << "    max: " << std::scientific << max_div << "\n";
    std::cout << "    rms: " << rms_div << "\n";
    
    assert(max_div < 1e-10 && "Divergence too large for all-wall BCs!");
    
    std::cout << "  ✓ PASSED\n";
}

int main() {
    std::cout << "========================================\n";
    std::cout << "Divergence Tests for All BC Types\n";
    std::cout << "Staggered Grid Implementation\n";
    std::cout << "========================================\n";
    std::cout << "\nTesting that staggered grid + projection\n";
    std::cout << "achieves machine-epsilon divergence (~1e-12)\n";
    std::cout << "for all supported boundary condition types.\n";
    
    test_divergence_periodic_periodic();
    test_divergence_periodic_wall();
    test_divergence_wall_periodic();
    test_divergence_all_walls();
    
    std::cout << "\n========================================\n";
    std::cout << "All divergence tests PASSED! ✓\n";
    std::cout << "========================================\n";
    
    return 0;
}








