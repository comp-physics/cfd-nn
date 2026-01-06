/// Unit tests for 3D boundary condition corner cases
///
/// Tests 3D-specific boundary handling:
/// - Multiple BC combinations
/// - Corner and edge interactions
/// - Divergence-free constraint in 3D
/// - 3D gradient computation near boundaries

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "poisson_solver.hpp"
#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>
#include <tuple>

using namespace nncfd;

// ============================================================================
// BC Combination Tests
// ============================================================================

void test_channel_like_bcs() {
    std::cout << "Testing channel-like BCs (Periodic x, Wall y, Periodic z)... ";

    Mesh mesh;
    mesh.init_uniform(16, 32, 8, 0.0, 2.0, -1.0, 1.0, 0.0, 1.0);

    Config config;
    config.nu = 0.01;
    config.dt = 0.001;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    bc.z_lo = VelocityBC::Periodic;
    bc.z_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    solver.set_body_force(-0.001, 0.0);
    solver.initialize_uniform(0.5, 0.0);

    // Run some steps
    for (int i = 0; i < 20; ++i) {
        solver.step();
    }

    // Check solution is finite
    const VectorField& vel = solver.velocity();
    bool all_finite = true;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                if (!std::isfinite(vel.u(i, j, k)) ||
                    !std::isfinite(vel.v(i, j, k)) ||
                    !std::isfinite(vel.w(i, j, k))) {
                    all_finite = false;
                }
            }
        }
    }
    assert(all_finite);

    std::cout << "PASSED\n";
}

void test_duct_like_bcs() {
    std::cout << "Testing duct-like BCs (Periodic x, Wall y, Wall z)... ";

    Mesh mesh;
    mesh.init_uniform(16, 16, 16, 0.0, 2.0, -1.0, 1.0, -1.0, 1.0);

    Config config;
    config.nu = 0.01;
    config.dt = 0.001;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    bc.z_lo = VelocityBC::NoSlip;
    bc.z_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);

    solver.set_body_force(-0.001, 0.0);
    solver.initialize_uniform(0.5, 0.0);

    for (int i = 0; i < 20; ++i) {
        solver.step();
    }

    // Check wall BCs are enforced (velocity should be zero at walls)
    const VectorField& vel = solver.velocity();
    double max_wall_vel = 0.0;

    // Check y walls
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            // y_lo wall
            max_wall_vel = std::max(max_wall_vel, std::abs(vel.u(i, mesh.j_begin(), k)));
            // y_hi wall
            max_wall_vel = std::max(max_wall_vel, std::abs(vel.u(i, mesh.j_end() - 1, k)));
        }
    }

    // Wall velocity should be very small
    assert(max_wall_vel < 0.1);

    std::cout << "PASSED\n";
}

void test_all_periodic_bcs() {
    std::cout << "Testing all periodic BCs... ";

    Mesh mesh;
    int N = 16;
    double L = 2.0 * M_PI;
    mesh.init_uniform(N, N, N, 0.0, L, 0.0, L, 0.0, L);

    ScalarField rhs(mesh);
    ScalarField p(mesh, 0.0);

    // sin(x)*sin(y)*sin(z) has zero mean
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double x = mesh.x(i);
                double y = mesh.y(j);
                double z = mesh.z(k);
                rhs(i, j, k) = -3.0 * std::sin(x) * std::sin(y) * std::sin(z);
            }
        }
    }

    PoissonSolver solver(mesh);
    solver.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                  PoissonBC::Periodic, PoissonBC::Periodic,
                  PoissonBC::Periodic, PoissonBC::Periodic);

    PoissonConfig cfg;
    cfg.tol = 1e-6;
    cfg.max_iter = 5000;
    cfg.omega = 1.5;

    int iters = solver.solve(rhs, p, cfg);

    assert(solver.residual() < 1e-4);

    std::cout << "PASSED (iters=" << iters << ")\n";
}

void test_mixed_neumann_periodic() {
    std::cout << "Testing mixed Neumann/Periodic BCs... ";

    Mesh mesh;
    mesh.init_uniform(16, 16, 16, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0);

    ScalarField rhs(mesh, 0.0);
    ScalarField p(mesh, 0.0);

    // Small perturbation
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                rhs(i, j, k) = 0.1 * std::sin(M_PI * mesh.x(i) / 2.0);
            }
        }
    }

    PoissonSolver solver(mesh);
    // Periodic in x, Neumann in y and z
    solver.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                  PoissonBC::Neumann, PoissonBC::Neumann,
                  PoissonBC::Neumann, PoissonBC::Neumann);

    PoissonConfig cfg;
    cfg.tol = 1e-6;
    cfg.max_iter = 5000;
    cfg.omega = 1.5;

    int iters = solver.solve(rhs, p, cfg);

    assert(solver.residual() < 1e-4);

    std::cout << "PASSED (iters=" << iters << ")\n";
}

// ============================================================================
// Corner and Edge Tests
// ============================================================================

void test_corner_cells_finite() {
    std::cout << "Testing corner cells remain finite... ";

    Mesh mesh;
    mesh.init_uniform(8, 8, 8, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    Config config;
    config.nu = 0.1;
    config.dt = 0.01;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::NoSlip;
    bc.x_hi = VelocityBC::NoSlip;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    bc.z_lo = VelocityBC::NoSlip;
    bc.z_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);

    solver.set_body_force(-0.01, 0.0);
    solver.initialize_uniform(0.1, 0.0);

    for (int i = 0; i < 10; ++i) {
        solver.step();
    }

    // Check all cells including corners
    const VectorField& vel = solver.velocity();
    bool all_finite = true;

    for (int k = 0; k < mesh.total_Nz(); ++k) {
        for (int j = 0; j < mesh.total_Ny(); ++j) {
            for (int i = 0; i < mesh.total_Nx(); ++i) {
                if (!std::isfinite(vel.u(i, j, k)) ||
                    !std::isfinite(vel.v(i, j, k)) ||
                    !std::isfinite(vel.w(i, j, k))) {
                    all_finite = false;
                }
            }
        }
    }
    assert(all_finite);

    std::cout << "PASSED\n";
}

void test_edge_cell_values() {
    std::cout << "Testing edge cell boundary values... ";

    Mesh mesh;
    mesh.init_uniform(8, 8, 8, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    Config config;
    config.nu = 0.1;
    config.dt = 0.01;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    bc.z_lo = VelocityBC::NoSlip;
    bc.z_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);

    solver.initialize_uniform(1.0, 0.0);
    solver.sync_to_gpu();

    // Take a step to apply boundary conditions
    solver.step();
    solver.sync_from_gpu();

    // After BC application, check edge cells (where y and z walls meet)
    const VectorField& vel = solver.velocity();

    // Check u velocity at y=0, z=0 edge (should be affected by both walls)
    bool edge_reasonable = true;
    for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
        double u_edge = vel.u(i, mesh.j_begin(), mesh.k_begin());
        if (!std::isfinite(u_edge)) {
            edge_reasonable = false;
        }
    }
    assert(edge_reasonable);

    std::cout << "PASSED\n";
}

// ============================================================================
// Divergence-Free Tests
// ============================================================================

void test_divergence_free_3d() {
    std::cout << "Testing divergence-free constraint in 3D... ";

    Mesh mesh;
    mesh.init_uniform(16, 16, 16, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0);

    Config config;
    config.nu = 0.01;
    config.dt = 0.001;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;
    config.poisson_max_iter = 50;  // Accurate solve

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::Periodic;
    bc.y_hi = VelocityBC::Periodic;
    bc.z_lo = VelocityBC::Periodic;
    bc.z_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    // Initialize with divergent velocity field
    solver.initialize_uniform(1.0, 0.5);

    // Step will apply projection
    for (int i = 0; i < 5; ++i) {
        solver.step();
    }

    // Check divergence
    const VectorField& vel = solver.velocity();
    double max_div = 0.0;

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double dudx = (vel.u(i + 1, j, k) - vel.u(i, j, k)) / mesh.dx;
                double dvdy = (vel.v(i, j + 1, k) - vel.v(i, j, k)) / mesh.dy;
                double dwdz = (vel.w(i, j, k + 1) - vel.w(i, j, k)) / mesh.dz;
                double div = dudx + dvdy + dwdz;
                max_div = std::max(max_div, std::abs(div));
            }
        }
    }

    // Divergence should be small
    if (max_div > 1e-4) {
        std::cout << "FAILED: max_div=" << max_div << " (expected < 1e-4)\n";
        std::exit(1);
    }

    std::cout << "PASSED (max_div=" << max_div << ")\n";
}

// ============================================================================
// 3D Poisson Solver BC Tests
// ============================================================================

void test_poisson_3d_dirichlet_all() {
    std::cout << "Testing 3D Poisson with all Dirichlet BCs... ";

    Mesh mesh;
    mesh.init_uniform(16, 16, 16, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    ScalarField rhs(mesh, 1.0);
    ScalarField p(mesh, 0.0);

    PoissonSolver solver(mesh);
    solver.set_bc(PoissonBC::Dirichlet, PoissonBC::Dirichlet,
                  PoissonBC::Dirichlet, PoissonBC::Dirichlet,
                  PoissonBC::Dirichlet, PoissonBC::Dirichlet);
    solver.set_dirichlet_value(0.0);

    PoissonConfig cfg;
    cfg.tol = 1e-6;
    cfg.max_iter = 10000;
    cfg.omega = 1.5;

    int iters = solver.solve(rhs, p, cfg);

    assert(solver.residual() < 1e-4);

    std::cout << "PASSED (iters=" << iters << ")\n";
}

void test_poisson_3d_mixed_bcs() {
    std::cout << "Testing 3D Poisson with mixed BCs... ";

    Mesh mesh;
    mesh.init_uniform(16, 32, 8, 0.0, 2.0, -1.0, 1.0, 0.0, 1.0);

    ScalarField rhs(mesh, 0.0);
    ScalarField p(mesh, 0.0);

    // Perturbation
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                rhs(i, j, k) = 0.1 * std::sin(mesh.x(i));
            }
        }
    }

    PoissonSolver solver(mesh);
    // Periodic x, Neumann y, Periodic z
    solver.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                  PoissonBC::Neumann, PoissonBC::Neumann,
                  PoissonBC::Periodic, PoissonBC::Periodic);

    PoissonConfig cfg;
    cfg.tol = 1e-6;
    cfg.max_iter = 5000;
    cfg.omega = 1.5;

    int iters = solver.solve(rhs, p, cfg);

    assert(solver.residual() < 1e-4);

    std::cout << "PASSED (iters=" << iters << ")\n";
}

// ============================================================================
// Solver Stability with 3D BCs
// ============================================================================

void test_3d_solver_stability_100_steps() {
    std::cout << "Testing 3D solver stability over 100 steps... ";

    Mesh mesh;
    mesh.init_uniform(16, 32, 8, 0.0, 2.0, -1.0, 1.0, 0.0, 1.0);

    Config config;
    config.nu = 0.001;
    config.dt = 1e-4;
    config.adaptive_dt = true;
    config.CFL_max = 0.5;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    bc.z_lo = VelocityBC::Periodic;
    bc.z_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    solver.set_body_force(-0.001, 0.0);
    solver.initialize_uniform(0.5, 0.0);

    // Run 100 steps
    for (int i = 0; i < 100; ++i) {
        solver.step();
    }

    // Check stability
    const VectorField& vel = solver.velocity();
    bool stable = true;
    double max_vel = 0.0;

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                if (!std::isfinite(vel.u(i, j, k)) ||
                    !std::isfinite(vel.v(i, j, k)) ||
                    !std::isfinite(vel.w(i, j, k))) {
                    stable = false;
                }
                max_vel = std::max(max_vel, std::abs(vel.u(i, j, k)));
            }
        }
    }

    assert(stable);
    assert(max_vel < 100.0);  // Velocity should not explode

    std::cout << "PASSED (max_vel=" << max_vel << ")\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== 3D Boundary Corner Cases Tests ===\n\n";

    // BC combination tests
    test_channel_like_bcs();
    test_duct_like_bcs();
    test_all_periodic_bcs();
    test_mixed_neumann_periodic();

    // Corner and edge tests
    test_corner_cells_finite();
    test_edge_cell_values();

    // Divergence-free tests
    test_divergence_free_3d();

    // 3D Poisson tests
    test_poisson_3d_dirichlet_all();
    test_poisson_3d_mixed_bcs();

    // Stability tests
    test_3d_solver_stability_100_steps();

    std::cout << "\nAll tests PASSED!\n";
    return 0;
}
