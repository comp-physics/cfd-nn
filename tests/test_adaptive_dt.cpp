/// Unit tests for adaptive time-stepping

#include "test_harness.hpp"
#include "test_utilities.hpp"
#include <cmath>

using namespace nncfd;
using nncfd::test::harness::record;
using nncfd::test::BCPattern;
using nncfd::test::create_velocity_bc;
using nncfd::test::make_test_solver_domain;
using nncfd::test::make_test_solver_3d_domain;

// Helper to create adaptive solver with specific config
static nncfd::test::TestSolver make_adaptive_solver(int Nx, int Ny, double Lx, double Ly,
                                                      double nu, double CFL_max,
                                                      BCPattern bc = BCPattern::FullyPeriodic) {
    nncfd::test::TestSolver ts;
    ts.mesh.init_uniform(Nx, Ny, 0.0, Lx, 0.0, Ly);
    ts.config.nu = nu;
    ts.config.dt = 0.01;
    ts.config.CFL_max = CFL_max;
    ts.config.adaptive_dt = true;
    ts.config.turb_model = TurbulenceModelType::None;
    ts.config.verbose = false;
    ts.solver = std::make_unique<RANSSolver>(ts.mesh, ts.config);
    ts.solver->set_velocity_bc(create_velocity_bc(bc));
    return ts;
}

// ============================================================================
// CFL Condition Tests
// ============================================================================

void test_cfl_uniform_velocity() {
    double dx = 0.1;
    auto ts = make_adaptive_solver(32, 32, 3.2, 3.2, 1e-6, 0.5);
    ts->initialize_uniform(1.0, 0.0);
#ifdef USE_GPU_OFFLOAD
    ts->sync_to_gpu();
#endif

    double dt_computed = ts->compute_adaptive_dt();
    double dt_expected = 0.5 * dx / 1.0;  // CFL_max * dx / u
    double relative_error = std::abs(dt_computed - dt_expected) / dt_expected;

    record("CFL uniform velocity", relative_error <= 0.1);
}

void test_cfl_different_cfl_max() {
    double dx = 0.1;
    std::vector<double> cfl_values = {0.3, 0.5, 0.8, 1.0};
    bool pass = true;

    for (double cfl : cfl_values) {
        auto ts = make_adaptive_solver(32, 32, 3.2, 3.2, 1e-6, cfl);
        ts->initialize_uniform(1.0, 0.0);
#ifdef USE_GPU_OFFLOAD
        ts->sync_to_gpu();
#endif
        double dt_computed = ts->compute_adaptive_dt();
        double dt_expected = cfl * dx / 1.0;
        if (std::abs(dt_computed - dt_expected) / dt_expected > 0.15) pass = false;
    }

    record("CFL different CFL_max values", pass);
}

// ============================================================================
// Diffusive Stability Tests
// ============================================================================

void test_diffusive_limit() {
    double dx = 0.1;
    auto ts = make_adaptive_solver(32, 32, 3.2, 3.2, 0.1, 0.5);  // High viscosity
    ts->initialize_uniform(0.001, 0.0);  // Small velocity
#ifdef USE_GPU_OFFLOAD
    ts->sync_to_gpu();
#endif

    double dt_computed = ts->compute_adaptive_dt();
    double dt_diff_expected = 0.25 * dx * dx / 0.1;  // 0.25 * dx² / nu
    double relative_error = std::abs(dt_computed - dt_diff_expected) / dt_diff_expected;

    record("Diffusive stability limit", relative_error <= 0.1);
}

void test_turbulent_viscosity_effect() {
    Mesh mesh;
    mesh.init_uniform(32, 64, 0.0, 2.0, -1.0, 1.0);

    Config config;
    config.nu = 0.001;
    config.dt = 1e-4;
    config.CFL_max = 0.5;
    config.adaptive_dt = true;
    config.turb_model = TurbulenceModelType::SSTKOmega;
    config.verbose = false;

    RANSSolver solver(mesh, config);
    solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel2D));
    auto turb_model = create_turbulence_model(TurbulenceModelType::SSTKOmega, "", "");
    solver.set_turbulence_model(std::move(turb_model));
    solver.set_body_force(-0.001, 0.0);
    solver.initialize_uniform(0.5, 0.0);

    for (int i = 0; i < 50; ++i) solver.step();

    double dt_after = solver.compute_adaptive_dt();
    record("Turbulent viscosity effect", dt_after > 0.0 && std::isfinite(dt_after));
}

// ============================================================================
// Minimum Selection Tests
// ============================================================================

void test_minimum_selection_cfl_wins() {
    double dx = 0.1;
    auto ts = make_adaptive_solver(32, 32, 3.2, 3.2, 1e-4, 0.5);  // Small nu
    ts->initialize_uniform(10.0, 0.0);  // High velocity
#ifdef USE_GPU_OFFLOAD
    ts->sync_to_gpu();
#endif

    double dt_computed = ts->compute_adaptive_dt();
    double dt_cfl = 0.5 * dx / 10.0;  // 0.005
    double dt_diff = 0.25 * dx * dx / 1e-4;  // 25

    bool cfl_wins = dt_cfl < dt_diff;
    double relative_error = std::abs(dt_computed - dt_cfl) / dt_cfl;

    record("Minimum selection (CFL wins)", cfl_wins && relative_error <= 0.15);
}

void test_minimum_selection_diffusion_wins() {
    double dx = 0.1;
    auto ts = make_adaptive_solver(32, 32, 3.2, 3.2, 0.5, 0.5);  // Large nu
    ts->initialize_uniform(0.01, 0.0);  // Small velocity
#ifdef USE_GPU_OFFLOAD
    ts->sync_to_gpu();
#endif

    double dt_computed = ts->compute_adaptive_dt();
    double dt_cfl = 0.5 * dx / 0.01;  // 5
    double dt_diff = 0.25 * dx * dx / 0.5;  // 0.005

    bool diff_wins = dt_diff < dt_cfl;
    double relative_error = std::abs(dt_computed - dt_diff) / dt_diff;

    record("Minimum selection (diffusion wins)", diff_wins && relative_error <= 0.15);
}

// ============================================================================
// 2D vs 3D Tests
// ============================================================================

void test_3d_adaptive_dt() {
    double dx = 0.2, dy = 0.1, dz = 0.15;
    int Nx = 16, Ny = 32, Nz = 20;

    nncfd::test::TestSolver ts;
    ts.mesh.init_uniform(Nx, Ny, Nz, 0.0, Nx * dx, 0.0, Ny * dy, 0.0, Nz * dz);
    ts.config.nu = 1e-5;
    ts.config.dt = 0.001;
    ts.config.CFL_max = 0.5;
    ts.config.adaptive_dt = true;
    ts.config.turb_model = TurbulenceModelType::None;
    ts.config.verbose = false;
    ts.solver = std::make_unique<RANSSolver>(ts.mesh, ts.config);
    ts.solver->set_velocity_bc(create_velocity_bc(BCPattern::Channel3D));
    ts->initialize_uniform(1.0, 0.0);

#ifdef USE_GPU_OFFLOAD
    ts->sync_to_gpu();
#endif

    double dt_computed = ts->compute_adaptive_dt();
    double dx_min = std::min({dx, dy, dz});
    double dt_cfl_expected = 0.5 * dx_min / 1.0;
    double relative_error = std::abs(dt_computed - dt_cfl_expected) / dt_cfl_expected;

    record("3D adaptive dt calculation", relative_error <= 0.15);
}

void test_2d_3d_consistency() {
    // 2D case
    auto ts2d = make_adaptive_solver(32, 32, 3.2, 3.2, 0.01, 0.5);
    ts2d->initialize_uniform(1.0, 0.0);
#ifdef USE_GPU_OFFLOAD
    ts2d->sync_to_gpu();
#endif
    double dt_2d = ts2d->compute_adaptive_dt();

    // 3D case with same dx, dy
    nncfd::test::TestSolver ts3d;
    ts3d.mesh.init_uniform(32, 32, 8, 0.0, 3.2, 0.0, 3.2, 0.0, 0.8);
    ts3d.config.nu = 0.01;
    ts3d.config.dt = 0.001;
    ts3d.config.CFL_max = 0.5;
    ts3d.config.adaptive_dt = true;
    ts3d.config.turb_model = TurbulenceModelType::None;
    ts3d.config.verbose = false;
    ts3d.solver = std::make_unique<RANSSolver>(ts3d.mesh, ts3d.config);
    ts3d.solver->set_velocity_bc(create_velocity_bc(BCPattern::FullyPeriodic));
    ts3d->initialize_uniform(1.0, 0.0);

#ifdef USE_GPU_OFFLOAD
    ts3d->sync_to_gpu();
#endif
    double dt_3d = ts3d->compute_adaptive_dt();

    double relative_diff = std::abs(dt_2d - dt_3d) / dt_2d;
    record("2D/3D consistency", relative_diff <= 0.2);
}

// ============================================================================
// Edge Cases
// ============================================================================

void test_very_small_velocity() {
    double dx = 0.1;
    auto ts = make_adaptive_solver(32, 32, 3.2, 3.2, 0.01, 0.5);
    ts->initialize_uniform(1e-12, 0.0);
#ifdef USE_GPU_OFFLOAD
    ts->sync_to_gpu();
#endif

    double dt_computed = ts->compute_adaptive_dt();
    double dt_diff = 0.25 * dx * dx / 0.01;

    bool pass = std::isfinite(dt_computed) && dt_computed > 0.0 && dt_computed <= dt_diff * 1.1;
    record("Very small velocity (no div by zero)", pass);
}

void test_anisotropic_grid() {
    double dx = 3.2 / 16;  // 0.2
    double dy = 3.2 / 64;  // 0.05
    auto ts = make_adaptive_solver(16, 64, 3.2, 3.2, 1e-4, 0.5);
    ts->initialize_uniform(1.0, 0.0);
#ifdef USE_GPU_OFFLOAD
    ts->sync_to_gpu();
#endif

    double dt_computed = ts->compute_adaptive_dt();
    double dx_min = std::min(dx, dy);
    double dt_cfl_expected = 0.5 * dx_min / 1.0;
    double relative_error = std::abs(dt_computed - dt_cfl_expected) / dt_cfl_expected;

    record("Anisotropic grid", relative_error <= 0.15);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    return nncfd::test::harness::run("Adaptive Time-Stepping Tests", [] {
        test_cfl_uniform_velocity();
        test_cfl_different_cfl_max();
        test_diffusive_limit();
        test_turbulent_viscosity_effect();
        test_minimum_selection_cfl_wins();
        test_minimum_selection_diffusion_wins();
        test_3d_adaptive_dt();
        test_2d_3d_consistency();
        test_very_small_velocity();
        test_anisotropic_grid();
    });
}
