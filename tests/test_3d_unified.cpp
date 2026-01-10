/// Unified 3D Tests
/// Consolidates: test_3d_bc_application.cpp, test_3d_gradients.cpp,
///               test_3d_w_velocity.cpp, test_3d_bc_corners.cpp

#include "test_harness.hpp"
#include "test_utilities.hpp"
#include "poisson_solver.hpp"
#include <cmath>

using namespace nncfd;
using nncfd::test::harness::record;
using nncfd::test::BCPattern;
using nncfd::test::create_velocity_bc;
using nncfd::test::make_test_solver_3d_domain;
using nncfd::test::create_unit_cube_mesh;

//=============================================================================
// BC TESTS
//=============================================================================

void test_no_slip_walls() {
    auto ts = make_test_solver_3d_domain(16, 16, 8, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    ts->set_body_force(0.001, 0.0, 0.0);

    for (int k = ts.mesh.k_begin(); k < ts.mesh.k_end(); ++k)
        for (int j = ts.mesh.j_begin(); j < ts.mesh.j_end(); ++j)
            for (int i = ts.mesh.i_begin(); i <= ts.mesh.i_end(); ++i)
                ts->velocity().u(i, j, k) = 0.1;

#ifdef USE_GPU_OFFLOAD
    ts->sync_to_gpu();
#endif
    for (int step = 0; step < 5; ++step) ts->step();
    ts->sync_from_gpu();

    double max_wall_v = 0.0;
    for (int k = ts.mesh.k_begin(); k < ts.mesh.k_end(); ++k)
        for (int i = ts.mesh.i_begin(); i < ts.mesh.i_end(); ++i) {
            max_wall_v = std::max(max_wall_v, std::abs(ts->velocity().v(i, ts.mesh.j_begin(), k)));
            max_wall_v = std::max(max_wall_v, std::abs(ts->velocity().v(i, ts.mesh.j_end(), k)));
        }

    record("No-slip walls enforced on y-boundaries", max_wall_v < 1e-14);
}

void test_periodic_z() {
    auto ts = make_test_solver_3d_domain(16, 16, 8, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    for (int k = ts.mesh.k_begin(); k < ts.mesh.k_end(); ++k) {
        double z = ts.mesh.z(k);
        for (int j = ts.mesh.j_begin(); j < ts.mesh.j_end(); ++j) {
            double y = ts.mesh.y(j) - 0.5;
            for (int i = ts.mesh.i_begin(); i <= ts.mesh.i_end(); ++i)
                ts->velocity().u(i, j, k) = 0.01 * (0.25 - y*y) * (1.0 + 0.1*std::sin(2*M_PI*z));
        }
    }

#ifdef USE_GPU_OFFLOAD
    ts->sync_to_gpu();
#endif
    for (int step = 0; step < 10; ++step) ts->step();
    ts->sync_from_gpu();

    double max_w_diff = 0.0;
    for (int j = ts.mesh.j_begin(); j < ts.mesh.j_end(); ++j)
        for (int i = ts.mesh.i_begin(); i < ts.mesh.i_end(); ++i) {
            double w_lo = ts->velocity().w(i, j, ts.mesh.k_begin());
            double w_hi = ts->velocity().w(i, j, ts.mesh.k_end());
            max_w_diff = std::max(max_w_diff, std::abs(w_lo - w_hi));
        }

    record("Periodic z-direction consistency", max_w_diff < 1e-12);
}

void test_mass_conservation() {
    Mesh mesh;
    mesh.init_uniform(32, 32, 4, 0.0, 2.0, 0.0, 2.0, 0.0, 1.0);

    Config cfg;
    cfg.nu = 0.01; cfg.dp_dx = -0.001;
    cfg.adaptive_dt = true; cfg.max_steps = 500; cfg.tol = 1e-6;
    cfg.turb_model = TurbulenceModelType::None; cfg.verbose = false;

    RANSSolver solver(mesh, cfg);
    solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel3D));
    solver.set_body_force(-cfg.dp_dx, 0.0, 0.0);

    double H = 1.0, y_mid = 1.0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k)
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            double y = mesh.y(j) - y_mid;
            double u_ana = -cfg.dp_dx / (2.0 * cfg.nu) * (H*H - y*y);
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i)
                solver.velocity().u(i, j, k) = 0.9 * u_ana;
        }

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif
    [[maybe_unused]] auto [res, iters] = solver.solve_steady();
    solver.sync_from_gpu();

    double max_div = 0.0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k)
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j)
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double dudx = (solver.velocity().u(i+1,j,k) - solver.velocity().u(i,j,k)) / mesh.dx;
                double dvdy = (solver.velocity().v(i,j+1,k) - solver.velocity().v(i,j,k)) / mesh.dy;
                double dwdz = (solver.velocity().w(i,j,k+1) - solver.velocity().w(i,j,k)) / mesh.dz;
                max_div = std::max(max_div, std::abs(dudx + dvdy + dwdz));
            }

    record("Mass conservation (divergence-free)", max_div < 1e-4);
}

//=============================================================================
// GRADIENT TESTS
//=============================================================================

void test_linear_dudz() {
    Mesh mesh = create_unit_cube_mesh(8);
    VectorField vel(mesh);

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        double z = mesh.z(k);
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j)
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i)
                vel.u(i, j, k) = z;
    }

    double max_err = 0.0;
    for (int k = mesh.k_begin() + 1; k < mesh.k_end() - 1; ++k)
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j)
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double dudz = (vel.u(i, j, k+1) - vel.u(i, j, k-1)) / (2.0 * mesh.dz);
                max_err = std::max(max_err, std::abs(dudz - 1.0));
            }

    record("Linear u=z field (du/dz = 1)", max_err < 1e-10);
}

void test_sinusoidal_dwdx() {
    Mesh mesh;
    mesh.init_uniform(32, 8, 8, 0.0, 2*M_PI, 0.0, 1.0, 0.0, 1.0);
    VectorField vel(mesh);

    for (int k = mesh.k_begin(); k <= mesh.k_end(); ++k)
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j)
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i)
                vel.w(i, j, k) = std::sin(mesh.x(i));

    double max_err = 0.0;
    for (int k = mesh.k_begin(); k <= mesh.k_end(); ++k)
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j)
            for (int i = mesh.i_begin() + 1; i < mesh.i_end() - 1; ++i) {
                double dwdx = (vel.w(i+1,j,k) - vel.w(i-1,j,k)) / (2.0 * mesh.dx);
                max_err = std::max(max_err, std::abs(dwdx - std::cos(mesh.x(i))));
            }

    record("Sinusoidal w=sin(x) (dw/dx = cos(x))", max_err < 0.01);
}

void test_divergence_free_field() {
    Mesh mesh;
    mesh.init_uniform(32, 32, 4, 0.0, 2*M_PI, 0.0, 2*M_PI, 0.0, 1.0);
    VectorField vel(mesh);

    // u = sin(x)*cos(y), v = -cos(x)*sin(y), w = 0 → div = 0
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k)
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j)
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i)
                vel.u(i, j, k) = std::sin(mesh.xf[i]) * std::cos(mesh.y(j));

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k)
        for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j)
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i)
                vel.v(i, j, k) = -std::cos(mesh.x(i)) * std::sin(mesh.yf[j]);

    for (int k = mesh.k_begin(); k <= mesh.k_end(); ++k)
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j)
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i)
                vel.w(i, j, k) = 0.0;

    double max_div = 0.0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k)
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j)
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double dudx = (vel.u(i+1,j,k) - vel.u(i,j,k)) / mesh.dx;
                double dvdy = (vel.v(i,j+1,k) - vel.v(i,j,k)) / mesh.dy;
                double dwdz = (vel.w(i,j,k+1) - vel.w(i,j,k)) / mesh.dz;
                max_div = std::max(max_div, std::abs(dudx + dvdy + dwdz));
            }

    record("Divergence accuracy (div-free field)", max_div < 0.01);
}

//=============================================================================
// W-VELOCITY TESTS
//=============================================================================

void test_w_storage() {
    Mesh mesh = create_unit_cube_mesh(8);
    VectorField vel(mesh);

    for (int k = mesh.k_begin(); k <= mesh.k_end(); ++k)
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j)
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i)
                vel.w(i, j, k) = static_cast<double>(i + 10*j + 100*k);

    double max_err = 0.0;
    for (int k = mesh.k_begin(); k <= mesh.k_end(); ++k)
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j)
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i)
                max_err = std::max(max_err, std::abs(vel.w(i,j,k) - (i + 10*j + 100*k)));

    record("W-velocity storage and indexing", max_err < 1e-14);
}

void test_w_staggering() {
    Mesh mesh;
    mesh.init_uniform(4, 4, 4, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    int num_faces = 0;
    for (int k = mesh.k_begin(); k <= mesh.k_end(); ++k) ++num_faces;

    record("W-velocity staggering (z-faces)", num_faces == mesh.Nz + 1);
}

void test_w_divergence_contribution() {
    Mesh mesh = create_unit_cube_mesh(8);
    VectorField vel(mesh);

    // w = z → dw/dz = 1
    for (int k = mesh.k_begin(); k <= mesh.k_end(); ++k)
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j)
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i)
                vel.w(i, j, k) = mesh.zf[k];

    double max_err = 0.0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k)
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j)
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double dwdz = (vel.w(i,j,k+1) - vel.w(i,j,k)) / mesh.dz;
                max_err = std::max(max_err, std::abs(dwdz - 1.0));
            }

    record("W contribution to divergence", max_err < 1e-10);
}

void test_w_center_interpolation() {
    Mesh mesh = create_unit_cube_mesh(8);
    VectorField vel(mesh);

    for (int k = mesh.k_begin(); k <= mesh.k_end(); ++k)
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j)
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i)
                vel.w(i, j, k) = mesh.zf[k];

    double max_err = 0.0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k)
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j)
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double w_ctr = vel.w_center(i, j, k);
                max_err = std::max(max_err, std::abs(w_ctr - mesh.z(k)));
            }

    record("W-velocity cell-center interpolation", max_err < 1e-10);
}

//=============================================================================
// CORNER/EDGE TESTS
//=============================================================================

void test_channel_like_bcs() {
    auto ts = make_test_solver_3d_domain(16, 32, 8, 0.0, 2.0, -1.0, 1.0, 0.0, 1.0);
    ts->set_body_force(-0.001, 0.0);
    ts->initialize_uniform(0.5, 0.0);

    for (int i = 0; i < 20; ++i) ts->step();
    ts->sync_from_gpu();

    bool all_finite = true;
    for (int k = ts.mesh.k_begin(); k < ts.mesh.k_end() && all_finite; ++k)
        for (int j = ts.mesh.j_begin(); j < ts.mesh.j_end() && all_finite; ++j)
            for (int i = ts.mesh.i_begin(); i < ts.mesh.i_end() && all_finite; ++i)
                if (!std::isfinite(ts->velocity().u(i,j,k))) all_finite = false;

    record("Channel-like BCs (Periodic x, Wall y, Periodic z)", all_finite);
}

void test_duct_like_bcs() {
    auto ts = make_test_solver_3d_domain(16, 16, 16, 0.0, 2.0, -1.0, 1.0, -1.0, 1.0, BCPattern::Duct);
    ts->set_body_force(-0.001, 0.0);
    ts->initialize_uniform(0.5, 0.0);

    for (int i = 0; i < 20; ++i) ts->step();
    ts->sync_from_gpu();

    double max_wall = 0.0;
    for (int k = ts.mesh.k_begin(); k < ts.mesh.k_end(); ++k)
        for (int i = ts.mesh.i_begin(); i < ts.mesh.i_end(); ++i) {
            max_wall = std::max(max_wall, std::abs(ts->velocity().u(i, ts.mesh.j_begin(), k)));
            max_wall = std::max(max_wall, std::abs(ts->velocity().u(i, ts.mesh.j_end()-1, k)));
        }

    record("Duct-like BCs (Periodic x, Wall y, Wall z)", max_wall < 1.0);
}

void test_corner_cells_finite() {
    auto ts = make_test_solver_3d_domain(8, 8, 8, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, BCPattern::AllNoSlip, 0.1, 0.01);
    ts->set_body_force(-0.01, 0.0);
    ts->initialize_uniform(0.1, 0.0);

    for (int i = 0; i < 10; ++i) ts->step();
    ts->sync_from_gpu();

    bool all_finite = true;
    for (int k = 0; k < ts.mesh.total_Nz() && all_finite; ++k)
        for (int j = 0; j < ts.mesh.total_Ny() && all_finite; ++j)
            for (int i = 0; i < ts.mesh.total_Nx() && all_finite; ++i)
                if (!std::isfinite(ts->velocity().u(i,j,k))) all_finite = false;

    record("Corner cells remain finite", all_finite);
}

void test_divergence_free_3d() {
    auto ts = make_test_solver_3d_domain(16, 16, 16, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, BCPattern::FullyPeriodic);
    ts.config.poisson_max_vcycles = 50;
    ts->initialize_uniform(1.0, 0.5);

    for (int i = 0; i < 5; ++i) ts->step();
    ts->sync_from_gpu();

    double max_div = 0.0;
    for (int k = ts.mesh.k_begin(); k < ts.mesh.k_end(); ++k)
        for (int j = ts.mesh.j_begin(); j < ts.mesh.j_end(); ++j)
            for (int i = ts.mesh.i_begin(); i < ts.mesh.i_end(); ++i) {
                double dudx = (ts->velocity().u(i+1,j,k) - ts->velocity().u(i,j,k)) / ts.mesh.dx;
                double dvdy = (ts->velocity().v(i,j+1,k) - ts->velocity().v(i,j,k)) / ts.mesh.dy;
                double dwdz = (ts->velocity().w(i,j,k+1) - ts->velocity().w(i,j,k)) / ts.mesh.dz;
                max_div = std::max(max_div, std::abs(dudx + dvdy + dwdz));
            }

    record("Divergence-free constraint in 3D", max_div < 1e-4);
}

void test_3d_solver_stability() {
    Mesh mesh;
    mesh.init_uniform(16, 32, 8, 0.0, 2.0, -1.0, 1.0, 0.0, 1.0);

    Config cfg;
    cfg.nu = 0.001; cfg.dt = 1e-4;
    cfg.adaptive_dt = true; cfg.CFL_max = 0.5;
    cfg.turb_model = TurbulenceModelType::None; cfg.verbose = false;

    RANSSolver solver(mesh, cfg);
    solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel3D));
    solver.set_body_force(-0.001, 0.0);
    solver.initialize_uniform(0.5, 0.0);

    for (int i = 0; i < 100; ++i) solver.step();
    solver.sync_from_gpu();

    bool stable = true;
    double max_vel = 0.0;
    for (int k = mesh.k_begin(); k < mesh.k_end() && stable; ++k)
        for (int j = mesh.j_begin(); j < mesh.j_end() && stable; ++j)
            for (int i = mesh.i_begin(); i < mesh.i_end() && stable; ++i) {
                if (!std::isfinite(solver.velocity().u(i,j,k))) stable = false;
                max_vel = std::max(max_vel, std::abs(solver.velocity().u(i,j,k)));
            }

    record("3D solver stability over 100 steps", stable && max_vel < 100.0);
}

//=============================================================================
// POISSON 3D TESTS
//=============================================================================

void test_poisson_3d_all_periodic() {
    Mesh mesh;
    int N = 16; double L = 2.0 * M_PI;
    mesh.init_uniform(N, N, N, 0.0, L, 0.0, L, 0.0, L);

    ScalarField rhs(mesh), p(mesh, 0.0);
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k)
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j)
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i)
                rhs(i,j,k) = -3.0 * std::sin(mesh.x(i)) * std::sin(mesh.y(j)) * std::sin(mesh.z(k));

    PoissonSolver solver(mesh);
    solver.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                  PoissonBC::Periodic, PoissonBC::Periodic,
                  PoissonBC::Periodic, PoissonBC::Periodic);

    PoissonConfig cfg;
    cfg.tol = 1e-6; cfg.max_steps = 5000; cfg.omega = 1.5;
    solver.solve(rhs, p, cfg);

    record("3D Poisson all periodic BCs", solver.residual() < 1e-4);
}

void test_poisson_3d_dirichlet() {
    Mesh mesh = create_unit_cube_mesh(16);
    ScalarField rhs(mesh, 1.0), p(mesh, 0.0);

    PoissonSolver solver(mesh);
    solver.set_bc(PoissonBC::Dirichlet, PoissonBC::Dirichlet,
                  PoissonBC::Dirichlet, PoissonBC::Dirichlet,
                  PoissonBC::Dirichlet, PoissonBC::Dirichlet);
    solver.set_dirichlet_value(0.0);

    PoissonConfig cfg;
    cfg.tol = 1e-6; cfg.max_steps = 10000; cfg.omega = 1.5;
    solver.solve(rhs, p, cfg);

    record("3D Poisson all Dirichlet BCs", solver.residual() < 1e-4);
}

//=============================================================================
// MAIN
//=============================================================================

int main() {
    namespace harness = nncfd::test::harness;
    return harness::run_sections("Unified 3D Tests", {
        {"Boundary Condition Tests", [] {
            test_no_slip_walls();
            test_periodic_z();
            test_mass_conservation();
        }},
        {"Gradient Tests", [] {
            test_linear_dudz();
            test_sinusoidal_dwdx();
            test_divergence_free_field();
        }},
        {"W-Velocity Tests", [] {
            test_w_storage();
            test_w_staggering();
            test_w_divergence_contribution();
            test_w_center_interpolation();
        }},
        {"Corner/Edge Tests", [] {
            test_channel_like_bcs();
            test_duct_like_bcs();
            test_corner_cells_finite();
            test_divergence_free_3d();
            test_3d_solver_stability();
        }},
        {"3D Poisson Tests", [] {
            test_poisson_3d_all_periodic();
            test_poisson_3d_dirichlet();
        }}
    });
}
