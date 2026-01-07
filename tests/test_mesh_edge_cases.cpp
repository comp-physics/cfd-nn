/// Unit tests for extreme mesh configurations
///
/// Tests edge cases and numerical stability:
/// - High aspect ratio grids (100:1, 1:100)
/// - Very small grids (4x4, 8x8)
/// - Highly non-uniform stretching
/// - Mixed 2D/3D edge cases

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "poisson_solver.hpp"
#include "test_harness.hpp"
#include <iostream>
#include <cmath>

using namespace nncfd;
using nncfd::test::harness::record;

// ============================================================================
// High Aspect Ratio Tests
// ============================================================================

void test_high_aspect_ratio_100_to_1() {
    Mesh mesh;
    // 200 cells in x, 2 cells in y → aspect ratio 100:1
    mesh.init_uniform(200, 2, 0.0, 10.0, 0.0, 0.1);

    // Create scalar field and verify indexing works
    ScalarField p(mesh);

    for (int j = 0; j < mesh.total_Ny(); ++j) {
        for (int i = 0; i < mesh.total_Nx(); ++i) {
            p(i, j) = mesh.x(i) + mesh.y(j);
        }
    }

    // Verify field values
    bool correct = true;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double expected = mesh.x(i) + mesh.y(j);
            if (std::abs(p(i, j) - expected) > 1e-10) {
                correct = false;
            }
        }
    }

    // Verify mesh dimensions
    bool dims_ok = (std::abs(mesh.dx - 0.05) < 1e-10) && (std::abs(mesh.dy - 0.05) < 1e-10);

    record("High aspect ratio 100:1 (200x2)", correct && dims_ok);
}

void test_high_aspect_ratio_1_to_100() {
    Mesh mesh;
    // 2 cells in x, 200 cells in y → aspect ratio 1:100
    mesh.init_uniform(2, 200, 0.0, 0.1, 0.0, 10.0);

    ScalarField p(mesh);

    for (int j = 0; j < mesh.total_Ny(); ++j) {
        for (int i = 0; i < mesh.total_Nx(); ++i) {
            p(i, j) = mesh.x(i) * mesh.y(j);
        }
    }

    // Verify field values
    bool correct = true;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double expected = mesh.x(i) * mesh.y(j);
            if (std::abs(p(i, j) - expected) > 1e-10) {
                correct = false;
            }
        }
    }

    record("High aspect ratio 1:100 (2x200)", correct);
}

void test_poisson_high_aspect_ratio() {
    Mesh mesh;
    // 64x8 grid - aspect ratio 8:1
    mesh.init_uniform(64, 8, 0.0, 8.0, 0.0, 1.0);

    ScalarField rhs(mesh, 1.0);
    ScalarField p(mesh, 0.0);

    PoissonSolver solver(mesh);
    solver.set_bc(PoissonBC::Dirichlet, PoissonBC::Dirichlet,
                  PoissonBC::Dirichlet, PoissonBC::Dirichlet);
    solver.set_dirichlet_value(0.0);

    PoissonConfig cfg;
    cfg.tol = 1e-6;
    cfg.max_iter = 10000;
    cfg.omega = 1.5;

    solver.solve(rhs, p, cfg);

    // Should converge
    bool converged = solver.residual() < 1e-4;

    // Solution should be finite
    bool all_finite = true;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            if (!std::isfinite(p(i, j))) {
                all_finite = false;
            }
        }
    }

    record("Poisson on high aspect ratio grid", converged && all_finite);
}

// ============================================================================
// Very Small Grid Tests
// ============================================================================

void test_small_grid_4x4() {
    Mesh mesh;
    mesh.init_uniform(4, 4, 0.0, 1.0, 0.0, 1.0);

    Config config;
    config.nu = 0.1;
    config.dt = 0.01;
    config.adaptive_dt = false;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);

    solver.set_body_force(-0.01, 0.0);
    solver.initialize_uniform(0.1, 0.0);

    // Should not crash
    for (int i = 0; i < 10; ++i) {
        solver.step();
    }

    // Solution should be finite
    const VectorField& vel = solver.velocity();
    bool all_finite = true;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            if (!std::isfinite(vel.u(i, j)) || !std::isfinite(vel.v(i, j))) {
                all_finite = false;
            }
        }
    }

    record("Minimum viable grid (4x4)", all_finite);
}

void test_small_grid_8x8() {
    Mesh mesh;
    mesh.init_uniform(8, 8, 0.0, 1.0, 0.0, 1.0);

    ScalarField rhs(mesh);
    ScalarField p(mesh, 0.0);

    // Smooth RHS
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double x = mesh.x(i);
            double y = mesh.y(j);
            rhs(i, j) = std::sin(M_PI * x) * std::sin(M_PI * y);
        }
    }

    PoissonSolver solver(mesh);
    solver.set_bc(PoissonBC::Dirichlet, PoissonBC::Dirichlet,
                  PoissonBC::Dirichlet, PoissonBC::Dirichlet);
    solver.set_dirichlet_value(0.0);

    PoissonConfig cfg;
    cfg.tol = 1e-6;
    cfg.max_iter = 5000;
    cfg.omega = 1.5;

    solver.solve(rhs, p, cfg);

    record("Small grid (8x8)", solver.residual() < 1e-4);
}

void test_small_grid_poisson_convergence() {
    Mesh mesh;
    mesh.init_uniform(4, 4, 0.0, 1.0, 0.0, 1.0);

    ScalarField rhs(mesh, 1.0);
    ScalarField p(mesh, 0.0);

    PoissonSolver solver(mesh);
    solver.set_bc(PoissonBC::Dirichlet, PoissonBC::Dirichlet,
                  PoissonBC::Dirichlet, PoissonBC::Dirichlet);
    solver.set_dirichlet_value(0.0);

    PoissonConfig cfg;
    cfg.tol = 1e-6;
    cfg.max_iter = 1000;
    cfg.omega = 1.2;

    solver.solve(rhs, p, cfg);

    // Even on tiny grid, should converge
    record("Poisson convergence on 4x4 grid", solver.residual() < 1e-3);
}

// ============================================================================
// Mesh Stretching Tests
// ============================================================================

void test_stretched_mesh_moderate() {
    Mesh mesh;
    mesh.init_stretched_y(32, 64, 0.0, 2.0, -1.0, 1.0, Mesh::tanh_stretching(2.0));

    // Verify stretching is applied (cells near wall should be smaller)
    double dy_wall = mesh.y(mesh.j_begin() + 1) - mesh.y(mesh.j_begin());
    double dy_center = mesh.y(mesh.Ny / 2 + 1) - mesh.y(mesh.Ny / 2);

    // Wall cells should be smaller than center cells
    bool stretched = dy_wall < dy_center;

    // Run a simple Poisson solve
    ScalarField rhs(mesh, 1.0);
    ScalarField p(mesh, 0.0);

    PoissonSolver solver(mesh);
    solver.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                  PoissonBC::Dirichlet, PoissonBC::Dirichlet);
    solver.set_dirichlet_value(0.0);

    PoissonConfig cfg;
    cfg.tol = 1e-6;
    cfg.max_iter = 5000;
    cfg.omega = 1.5;

    solver.solve(rhs, p, cfg);

    record("Moderately stretched mesh (beta=2.0)", stretched && solver.residual() < 1e-4);
}

void test_stretched_mesh_aggressive() {
    Mesh mesh;
    mesh.init_stretched_y(32, 64, 0.0, 2.0, -1.0, 1.0, Mesh::tanh_stretching(5.0));

    // Verify strong stretching
    double dy_wall = mesh.y(mesh.j_begin() + 1) - mesh.y(mesh.j_begin());
    double dy_center = mesh.y(mesh.Ny / 2 + 1) - mesh.y(mesh.Ny / 2);

    // Should have significant ratio
    double ratio = dy_center / dy_wall;
    bool good_ratio = ratio > 2.0;

    // Still should produce valid mesh
    bool valid = true;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        double dy = mesh.y(j + 1) - mesh.y(j);
        if (dy <= 0.0 || !std::isfinite(dy)) {
            valid = false;
        }
    }

    record("Aggressively stretched mesh (beta=5.0)", good_ratio && valid);
}

// ============================================================================
// Mixed 2D/3D Tests
// ============================================================================

void test_minimal_3d_nz2() {
    Mesh mesh;
    mesh.init_uniform(16, 32, 2, 0.0, 1.0, -0.5, 0.5, 0.0, 0.1);

    bool pass = (mesh.Nz == 2) && (!mesh.is2D());

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

    solver.set_body_force(-0.01, 0.0);
    solver.initialize_uniform(0.5, 0.0);

    // Should not crash
    for (int i = 0; i < 10; ++i) {
        solver.step();
    }

    record("Minimal 3D grid (Nz=2)", pass);
}

void test_2d_vs_3d_code_path() {
    // 2D mesh (Nz=1)
    Mesh mesh2d;
    mesh2d.init_uniform(16, 32, 0.0, 1.0, -0.5, 0.5);
    bool is2d_ok = mesh2d.is2D() && (mesh2d.Nz == 1);

    // 3D mesh (Nz>1)
    Mesh mesh3d;
    mesh3d.init_uniform(16, 32, 8, 0.0, 1.0, -0.5, 0.5, 0.0, 0.5);
    bool is3d_ok = !mesh3d.is2D() && (mesh3d.Nz == 8);

    record("2D vs 3D code path selection", is2d_ok && is3d_ok);
}

// ============================================================================
// Stress Tests
// ============================================================================

void test_moderate_grid_stability() {
    Mesh mesh;
    mesh.init_uniform(64, 64, 0.0, 2.0, -1.0, 1.0);

    Config config;
    config.nu = 0.01;
    config.dt = 0.001;
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
    solver.set_velocity_bc(bc);

    solver.set_body_force(-0.001, 0.0);
    solver.initialize_uniform(0.5, 0.0);

    // Run 50 steps
    double max_residual = 0.0;
    for (int i = 0; i < 50; ++i) {
        double res = solver.step();
        max_residual = std::max(max_residual, res);
    }

    // Should remain stable (residual bounded)
    bool stable = max_residual < 10.0;

    // Check for NaN/Inf
    const VectorField& vel = solver.velocity();
    bool all_finite = true;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            if (!std::isfinite(vel.u(i, j)) || !std::isfinite(vel.v(i, j))) {
                all_finite = false;
            }
        }
    }

    record("Moderate grid stability (64x64)", stable && all_finite);
}

void test_non_square_domain() {
    Mesh mesh;
    mesh.init_uniform(100, 20, 0.0, 10.0, 0.0, 1.0);

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
    solver.set_velocity_bc(bc);

    solver.set_body_force(-0.001, 0.0);
    solver.initialize_uniform(0.5, 0.0);

    bool pass = true;
    for (int i = 0; i < 20; ++i) {
        solver.step();
    }

    record("Non-square domain (Lx=10, Ly=1)", pass);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    return nncfd::test::harness::run("Mesh Edge Cases Tests", [] {
        // High aspect ratio tests
        test_high_aspect_ratio_100_to_1();
        test_high_aspect_ratio_1_to_100();
        test_poisson_high_aspect_ratio();

        // Small grid tests
        test_small_grid_4x4();
        test_small_grid_8x8();
        test_small_grid_poisson_convergence();

        // Mesh stretching tests
        test_stretched_mesh_moderate();
        test_stretched_mesh_aggressive();

        // Mixed 2D/3D tests
        test_minimal_3d_nz2();
        test_2d_vs_3d_code_path();

        // Stress tests
        test_moderate_grid_stability();
        test_non_square_domain();
    });
}
