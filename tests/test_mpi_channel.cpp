/// @file test_mpi_channel.cpp
/// @brief Test MPI integration into solver — verify single-rank matches serial
///
/// Test coverage:
///   1. Single-process with Decomposition: solver works with decomp set
///   2. Poiseuille flow converges correctly with decomposition active
///
/// This test validates that adding Decomposition to the solver doesn't break
/// existing single-process functionality.

#include "solver.hpp"
#include "config.hpp"
#include "mesh.hpp"
#include "decomposition.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>

using namespace nncfd;

/// Test: Solver works correctly with single-process Decomposition
bool test_solver_with_decomp() {
    std::cerr << "[test_solver_with_decomp] starting..." << std::endl;

    const int Nx = 8, Ny = 16;
    const double nu = 0.1;
    const double fx = 1.0;

    Mesh mesh;
    mesh.init_uniform(Nx, Ny, 0.0, 2.0*M_PI, -1.0, 1.0);

    Config config;
    config.Nx = Nx;
    config.Ny = Ny;
    config.Nz = 1;
    config.nu = nu;
    config.dt = 0.001;
    config.max_steps = 100;
    config.tol = 1e-8;
    config.poisson_solver = PoissonSolverType::MG;
    config.time_integrator = TimeIntegrator::Euler;

    Decomposition decomp(1);

    RANSSolver solver(mesh, config);
    solver.set_decomposition(&decomp);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);
    solver.set_body_force(fx, 0.0);
    solver.initialize_uniform(0.0, 0.0);

    for (int i = 0; i < 50; ++i) {
        solver.step();
    }

    double ub = solver.bulk_velocity();
    std::cerr << "[test_solver_with_decomp] ub=" << ub << std::endl;
    if (!(std::abs(ub) > 0.0)) {
        std::cerr << "FAIL: Bulk velocity is zero under body force (ub=" << ub << ")" << std::endl;
        return false;
    }

    double dt = solver.compute_adaptive_dt();
    std::cerr << "[test_solver_with_decomp] dt=" << dt << std::endl;
    if (!(dt > 0.0 && dt < 1.0)) {
        std::cerr << "FAIL: Adaptive dt unreasonable (dt=" << dt << ")" << std::endl;
        return false;
    }

    std::cout << "PASS: Solver with Decomposition (Ub=" << ub
              << ", dt=" << dt << ")" << std::endl;
    return true;
}

/// Test: Poiseuille convergence unaffected by Decomposition
bool test_poiseuille_with_decomp() {
    std::cerr << "[test_poiseuille_with_decomp] starting..." << std::endl;

    const int Nx = 4, Ny = 16;
    const double nu = 0.1;
    const double fx = 1.0;
    const double H = 1.0;

    Mesh mesh;
    mesh.init_uniform(Nx, Ny, 0.0, 2.0*M_PI, -1.0, 1.0);

    Config config;
    config.Nx = Nx;
    config.Ny = Ny;
    config.Nz = 1;
    config.nu = nu;
    config.dt = 0.005;
    config.max_steps = 2000;
    config.tol = 1e-8;
    config.poisson_solver = PoissonSolverType::MG;
    config.time_integrator = TimeIntegrator::Euler;

    Decomposition decomp(1);

    RANSSolver solver(mesh, config);
    solver.set_decomposition(&decomp);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);
    solver.set_body_force(fx, 0.0);
    solver.initialize_uniform(0.0, 0.0);

    double residual = 1.0;
    int steps = 0;
    for (int i = 0; i < 2000 && residual > 1e-8; ++i) {
        residual = solver.step();
        steps = i + 1;
    }

    std::cerr << "[test_poiseuille_with_decomp] residual=" << residual
              << " after " << steps << " steps" << std::endl;

    if (residual >= 1e-6) {
        std::cerr << "FAIL: Poiseuille convergence failed: residual=" << residual
                  << " after " << steps << " steps" << std::endl;
        return false;
    }

    double u_analytical = fx * H * H / (3.0 * nu);
    double u_numerical = solver.bulk_velocity();
    double rel_error = std::abs(u_numerical - u_analytical) / std::abs(u_analytical);

    std::cout << "Poiseuille: U_analytical=" << u_analytical
              << ", U_numerical=" << u_numerical
              << ", rel_error=" << rel_error << std::endl;
    if (rel_error >= 0.10) {
        std::cerr << "FAIL: Poiseuille error too large: " << rel_error << " (limit 10%)" << std::endl;
        return false;
    }

    std::cout << "PASS: Poiseuille convergence with Decomposition" << std::endl;
    return true;
}

int main() {
    int failures = 0;

    if (!test_solver_with_decomp()) failures++;
    if (!test_poiseuille_with_decomp()) failures++;

    if (failures > 0) {
        std::cerr << "\n" << failures << " test(s) FAILED" << std::endl;
        return 1;
    }

    std::cout << "\nAll MPI channel tests PASSED" << std::endl;
    return 0;
}
