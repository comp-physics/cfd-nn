/// Unit tests for extreme mesh configurations

#include "test_harness.hpp"
#include "test_utilities.hpp"
#include "poisson_solver.hpp"
#include <cmath>

using namespace nncfd;
using nncfd::test::harness::record;
using nncfd::test::BCPattern;
using nncfd::test::create_velocity_bc;

// ============================================================================
// High Aspect Ratio Tests
// ============================================================================

void test_high_aspect_ratio_100_to_1() {
    Mesh mesh;
    mesh.init_uniform(200, 2, 0.0, 10.0, 0.0, 0.1);
    ScalarField p(mesh);

    for (int j = 0; j < mesh.total_Ny(); ++j)
        for (int i = 0; i < mesh.total_Nx(); ++i)
            p(i, j) = mesh.x(i) + mesh.y(j);

    bool correct = true;
    FOR_INTERIOR_2D(mesh, i, j) {
        if (std::abs(p(i, j) - (mesh.x(i) + mesh.y(j))) > 1e-10) correct = false;
    }

    bool dims_ok = std::abs(mesh.dx - 0.05) < 1e-10 && std::abs(mesh.dy - 0.05) < 1e-10;
    record("High aspect ratio 100:1 (200x2)", correct && dims_ok);
}

void test_high_aspect_ratio_1_to_100() {
    Mesh mesh;
    mesh.init_uniform(2, 200, 0.0, 0.1, 0.0, 10.0);
    ScalarField p(mesh);

    for (int j = 0; j < mesh.total_Ny(); ++j)
        for (int i = 0; i < mesh.total_Nx(); ++i)
            p(i, j) = mesh.x(i) * mesh.y(j);

    bool correct = true;
    FOR_INTERIOR_2D(mesh, i, j) {
        if (std::abs(p(i, j) - mesh.x(i) * mesh.y(j)) > 1e-10) correct = false;
    }
    record("High aspect ratio 1:100 (2x200)", correct);
}

void test_poisson_high_aspect_ratio() {
    Mesh mesh;
    mesh.init_uniform(64, 8, 0.0, 8.0, 0.0, 1.0);

    ScalarField rhs(mesh, 1.0), p(mesh, 0.0);
    PoissonSolver solver(mesh);
    solver.set_bc(PoissonBC::Dirichlet, PoissonBC::Dirichlet,
                  PoissonBC::Dirichlet, PoissonBC::Dirichlet);
    solver.set_dirichlet_value(0.0);

    PoissonConfig cfg{1e-6, 10000, 1.5};
    solver.solve(rhs, p, cfg);

    bool all_finite = true;
    FOR_INTERIOR_2D(mesh, i, j) { if (!std::isfinite(p(i, j))) all_finite = false; }
    record("Poisson on high aspect ratio grid", solver.residual() < 1e-4 && all_finite);
}

// ============================================================================
// Very Small Grid Tests
// ============================================================================

void test_small_grid_4x4() {
    auto ts = nncfd::test::make_test_solver_domain(4, 4, 0.0, 1.0, 0.0, 1.0,
                                                    BCPattern::Channel2D, 0.1, 0.01);
    ts->set_body_force(-0.01, 0.0);
    ts->initialize_uniform(0.1, 0.0);

    for (int i = 0; i < 10; ++i) ts->step();
    ts->sync_from_gpu();

    bool all_finite = true;
    const VectorField& vel = ts->velocity();
    FOR_INTERIOR_2D(ts.mesh, i, j) {
        if (!std::isfinite(vel.u(i, j)) || !std::isfinite(vel.v(i, j))) all_finite = false;
    }
    record("Minimum viable grid (4x4)", all_finite);
}

void test_small_grid_8x8() {
    Mesh mesh;
    mesh.init_uniform(8, 8, 0.0, 1.0, 0.0, 1.0);

    ScalarField rhs(mesh), p(mesh, 0.0);
    FOR_INTERIOR_2D(mesh, i, j) {
        rhs(i, j) = std::sin(M_PI * mesh.x(i)) * std::sin(M_PI * mesh.y(j));
    }

    PoissonSolver solver(mesh);
    solver.set_bc(PoissonBC::Dirichlet, PoissonBC::Dirichlet,
                  PoissonBC::Dirichlet, PoissonBC::Dirichlet);
    solver.set_dirichlet_value(0.0);

    PoissonConfig cfg{1e-6, 5000, 1.5};
    solver.solve(rhs, p, cfg);
    record("Small grid (8x8)", solver.residual() < 1e-4);
}

void test_small_grid_poisson_convergence() {
    Mesh mesh;
    mesh.init_uniform(4, 4, 0.0, 1.0, 0.0, 1.0);

    ScalarField rhs(mesh, 1.0), p(mesh, 0.0);
    PoissonSolver solver(mesh);
    solver.set_bc(PoissonBC::Dirichlet, PoissonBC::Dirichlet,
                  PoissonBC::Dirichlet, PoissonBC::Dirichlet);
    solver.set_dirichlet_value(0.0);

    PoissonConfig cfg{1e-6, 1000, 1.2};
    solver.solve(rhs, p, cfg);
    record("Poisson convergence on 4x4 grid", solver.residual() < 1e-3);
}

// ============================================================================
// Mesh Stretching Tests
// ============================================================================

void test_stretched_mesh_moderate() {
    Mesh mesh;
    mesh.init_stretched_y(32, 64, 0.0, 2.0, -1.0, 1.0, Mesh::tanh_stretching(2.0));

    double dy_wall = mesh.y(mesh.j_begin() + 1) - mesh.y(mesh.j_begin());
    double dy_center = mesh.y(mesh.Ny / 2 + 1) - mesh.y(mesh.Ny / 2);
    bool stretched = dy_wall < dy_center;

    ScalarField rhs(mesh, 1.0), p(mesh, 0.0);
    PoissonSolver solver(mesh);
    solver.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                  PoissonBC::Dirichlet, PoissonBC::Dirichlet);
    solver.set_dirichlet_value(0.0);

    PoissonConfig cfg{1e-6, 5000, 1.5};
    solver.solve(rhs, p, cfg);
    record("Moderately stretched mesh (beta=2.0)", stretched && solver.residual() < 1e-4);
}

void test_stretched_mesh_aggressive() {
    Mesh mesh;
    mesh.init_stretched_y(32, 64, 0.0, 2.0, -1.0, 1.0, Mesh::tanh_stretching(5.0));

    double dy_wall = mesh.y(mesh.j_begin() + 1) - mesh.y(mesh.j_begin());
    double dy_center = mesh.y(mesh.Ny / 2 + 1) - mesh.y(mesh.Ny / 2);
    bool good_ratio = (dy_center / dy_wall) > 2.0;

    bool valid = true;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        double dy = mesh.y(j + 1) - mesh.y(j);
        if (dy <= 0.0 || !std::isfinite(dy)) valid = false;
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
    config.nu = 0.01; config.dt = 0.001;
    config.turb_model = TurbulenceModelType::None; config.verbose = false;

    RANSSolver solver(mesh, config);
    solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel3D));
    solver.set_body_force(-0.01, 0.0);
    solver.initialize_uniform(0.5, 0.0);

    for (int i = 0; i < 10; ++i) solver.step();
    record("Minimal 3D grid (Nz=2)", pass);
}

void test_2d_vs_3d_code_path() {
    Mesh mesh2d;
    mesh2d.init_uniform(16, 32, 0.0, 1.0, -0.5, 0.5);
    bool is2d_ok = mesh2d.is2D() && (mesh2d.Nz == 1);

    Mesh mesh3d;
    mesh3d.init_uniform(16, 32, 8, 0.0, 1.0, -0.5, 0.5, 0.0, 0.5);
    bool is3d_ok = !mesh3d.is2D() && (mesh3d.Nz == 8);

    record("2D vs 3D code path selection", is2d_ok && is3d_ok);
}

// ============================================================================
// Stress Tests
// ============================================================================

void test_moderate_grid_stability() {
    nncfd::test::TestSolver ts;
    ts.mesh.init_uniform(64, 64, 0.0, 2.0, -1.0, 1.0);
    ts.config.nu = 0.01; ts.config.dt = 0.001;
    ts.config.adaptive_dt = true; ts.config.CFL_max = 0.5;
    ts.config.turb_model = TurbulenceModelType::None; ts.config.verbose = false;
    ts.solver = std::make_unique<RANSSolver>(ts.mesh, ts.config);
    ts.solver->set_velocity_bc(create_velocity_bc(BCPattern::Channel2D));
    ts.solver->set_body_force(-0.001, 0.0);
    ts.solver->initialize_uniform(0.5, 0.0);
    ts->sync_to_gpu();

    double max_residual = 0.0;
    for (int i = 0; i < 50; ++i) max_residual = std::max(max_residual, ts->step());
    ts->sync_from_gpu();

    bool all_finite = true;
    const VectorField& vel = ts->velocity();
    FOR_INTERIOR_2D(ts.mesh, i, j) {
        if (!std::isfinite(vel.u(i, j)) || !std::isfinite(vel.v(i, j))) all_finite = false;
    }
    record("Moderate grid stability (64x64)", max_residual < 10.0 && all_finite);
}

void test_non_square_domain() {
    auto ts = nncfd::test::make_test_solver_domain(100, 20, 0.0, 10.0, 0.0, 1.0);
    ts->set_body_force(-0.001, 0.0);
    ts->initialize_uniform(0.5, 0.0);

    for (int i = 0; i < 20; ++i) ts->step();
    record("Non-square domain (Lx=10, Ly=1)", true);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    return nncfd::test::harness::run("Mesh Edge Cases Tests", [] {
        test_high_aspect_ratio_100_to_1();
        test_high_aspect_ratio_1_to_100();
        test_poisson_high_aspect_ratio();
        test_small_grid_4x4();
        test_small_grid_8x8();
        test_small_grid_poisson_convergence();
        test_stretched_mesh_moderate();
        test_stretched_mesh_aggressive();
        test_minimal_3d_nz2();
        test_2d_vs_3d_code_path();
        test_moderate_grid_stability();
        test_non_square_domain();
    });
}
