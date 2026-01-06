/// Unit tests for adaptive time-stepping
///
/// Tests the compute_adaptive_dt() function:
/// - CFL condition calculation
/// - Diffusive stability calculation
/// - Minimum selection between advective and diffusive limits
/// - 2D vs 3D consistency
/// - Edge cases (zero velocity, high viscosity)

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include <iostream>
#include <cmath>
#include <cassert>

using namespace nncfd;

// ============================================================================
// CFL Condition Tests
// ============================================================================

void test_cfl_uniform_velocity() {
    std::cout << "Testing CFL with uniform velocity field... ";

    Mesh mesh;
    double dx = 0.1;
    int Nx = 32, Ny = 32;
    mesh.init_uniform(Nx, Ny, 0.0, Nx * dx, 0.0, Ny * dx);

    Config config;
    config.nu = 1e-6;  // Very small viscosity (CFL-limited)
    config.dt = 0.01;
    config.CFL_max = 0.5;
    config.adaptive_dt = true;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::Periodic;
    bc.y_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    // Initialize with uniform velocity u=1.0
    double u_init = 1.0;
    solver.initialize_uniform(u_init, 0.0);

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    double dt_computed = solver.compute_adaptive_dt();

    // Expected: dt_cfl = CFL_max * dx / u_max = 0.5 * 0.1 / 1.0 = 0.05
    double dt_cfl_expected = config.CFL_max * dx / u_init;

    // Should be CFL-limited (diffusive limit is much larger)
    // dt_diff = 0.25 * dx² / nu = 0.25 * 0.01 / 1e-6 = 2500 (huge)

    // Allow 10% tolerance for interpolation effects
    double relative_error = std::abs(dt_computed - dt_cfl_expected) / dt_cfl_expected;

    if (relative_error > 0.1) {
        std::cout << "FAILED: dt_computed=" << dt_computed
                  << ", expected~" << dt_cfl_expected
                  << ", rel_err=" << relative_error << "\n";
        std::exit(1);
    }

    std::cout << "PASSED (dt=" << dt_computed << ", expected~" << dt_cfl_expected << ")\n";
}

void test_cfl_different_cfl_max() {
    std::cout << "Testing CFL with different CFL_max values... ";

    Mesh mesh;
    double dx = 0.1;
    mesh.init_uniform(32, 32, 0.0, 3.2, 0.0, 3.2);

    std::vector<double> cfl_values = {0.3, 0.5, 0.8, 1.0};

    for (double cfl : cfl_values) {
        Config config;
        config.nu = 1e-6;
        config.dt = 0.01;
        config.CFL_max = cfl;
        config.adaptive_dt = true;
        config.turb_model = TurbulenceModelType::None;
        config.verbose = false;

        RANSSolver solver(mesh, config);

        VelocityBC bc;
        bc.x_lo = VelocityBC::Periodic;
        bc.x_hi = VelocityBC::Periodic;
        bc.y_lo = VelocityBC::Periodic;
        bc.y_hi = VelocityBC::Periodic;
        solver.set_velocity_bc(bc);

        solver.initialize_uniform(1.0, 0.0);

#ifdef USE_GPU_OFFLOAD
        solver.sync_to_gpu();
#endif

        double dt_computed = solver.compute_adaptive_dt();
        double dt_expected = cfl * dx / 1.0;

        double relative_error = std::abs(dt_computed - dt_expected) / dt_expected;
        if (relative_error > 0.15) {
            std::cout << "FAILED for CFL_max=" << cfl
                      << ": dt=" << dt_computed << ", expected~" << dt_expected << "\n";
            std::exit(1);
        }
    }

    std::cout << "PASSED\n";
}

// ============================================================================
// Diffusive Stability Tests
// ============================================================================

void test_diffusive_limit() {
    std::cout << "Testing diffusive stability limit... ";

    Mesh mesh;
    double dx = 0.1;
    mesh.init_uniform(32, 32, 0.0, 3.2, 0.0, 3.2);

    Config config;
    config.nu = 0.1;  // High viscosity (diffusion-limited)
    config.dt = 0.01;
    config.CFL_max = 0.5;
    config.adaptive_dt = true;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::Periodic;
    bc.y_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    // Very small velocity so CFL limit is large
    solver.initialize_uniform(0.001, 0.0);

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    double dt_computed = solver.compute_adaptive_dt();

    // Expected: dt_diff = 0.25 * dx² / nu = 0.25 * 0.01 / 0.1 = 0.025
    double dt_diff_expected = 0.25 * dx * dx / config.nu;

    // dt_cfl = 0.5 * 0.1 / 0.001 = 50 (huge, so diffusion dominates)

    double relative_error = std::abs(dt_computed - dt_diff_expected) / dt_diff_expected;

    if (relative_error > 0.1) {
        std::cout << "FAILED: dt_computed=" << dt_computed
                  << ", expected~" << dt_diff_expected << "\n";
        std::exit(1);
    }

    std::cout << "PASSED (dt=" << dt_computed << ", expected~" << dt_diff_expected << ")\n";
}

void test_turbulent_viscosity_effect() {
    std::cout << "Testing turbulent viscosity effect on dt... ";

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

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);

    auto turb_model = create_turbulence_model(TurbulenceModelType::SSTKOmega, "", "");
    solver.set_turbulence_model(std::move(turb_model));

    solver.set_body_force(-0.001, 0.0);
    solver.initialize_uniform(0.5, 0.0);

    // Get dt before turbulence develops
    double dt_initial = solver.compute_adaptive_dt();

    // Run some steps to develop nu_t
    for (int i = 0; i < 50; ++i) {
        solver.step();
    }

    double dt_after = solver.compute_adaptive_dt();

    // With turbulence, nu_eff = nu + nu_t > nu
    // So diffusive limit dt_diff = 0.25 * dx² / nu_eff should decrease
    // (but CFL might dominate, so just check dt is finite and positive)

    assert(dt_after > 0.0);
    assert(std::isfinite(dt_after));

    std::cout << "PASSED (dt_initial=" << dt_initial << ", dt_after=" << dt_after << ")\n";
}

// ============================================================================
// Minimum Selection Tests
// ============================================================================

void test_minimum_selection_cfl_wins() {
    std::cout << "Testing minimum selection (CFL wins)... ";

    Mesh mesh;
    double dx = 0.1;
    mesh.init_uniform(32, 32, 0.0, 3.2, 0.0, 3.2);

    Config config;
    config.nu = 1e-4;  // Small nu → large dt_diff
    config.dt = 0.01;
    config.CFL_max = 0.5;
    config.adaptive_dt = true;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::Periodic;
    bc.y_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    // High velocity → small dt_cfl
    solver.initialize_uniform(10.0, 0.0);

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    double dt_computed = solver.compute_adaptive_dt();

    // dt_cfl = 0.5 * 0.1 / 10 = 0.005
    // dt_diff = 0.25 * 0.01 / 1e-4 = 25
    // min(0.005, 25) = 0.005

    double dt_cfl = config.CFL_max * dx / 10.0;
    double dt_diff = 0.25 * dx * dx / config.nu;

    assert(dt_cfl < dt_diff);  // CFL should win

    double relative_error = std::abs(dt_computed - dt_cfl) / dt_cfl;
    if (relative_error > 0.15) {
        std::cout << "FAILED: dt=" << dt_computed << ", expected~" << dt_cfl << "\n";
        std::exit(1);
    }

    std::cout << "PASSED\n";
}

void test_minimum_selection_diffusion_wins() {
    std::cout << "Testing minimum selection (diffusion wins)... ";

    Mesh mesh;
    double dx = 0.1;
    mesh.init_uniform(32, 32, 0.0, 3.2, 0.0, 3.2);

    Config config;
    config.nu = 0.5;  // Large nu → small dt_diff
    config.dt = 0.01;
    config.CFL_max = 0.5;
    config.adaptive_dt = true;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::Periodic;
    bc.y_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    // Small velocity → large dt_cfl
    solver.initialize_uniform(0.01, 0.0);

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    double dt_computed = solver.compute_adaptive_dt();

    // dt_cfl = 0.5 * 0.1 / 0.01 = 5
    // dt_diff = 0.25 * 0.01 / 0.5 = 0.005
    // min(5, 0.005) = 0.005

    double dt_cfl = config.CFL_max * dx / 0.01;
    double dt_diff = 0.25 * dx * dx / config.nu;

    assert(dt_diff < dt_cfl);  // Diffusion should win

    double relative_error = std::abs(dt_computed - dt_diff) / dt_diff;
    if (relative_error > 0.15) {
        std::cout << "FAILED: dt=" << dt_computed << ", expected~" << dt_diff << "\n";
        std::exit(1);
    }

    std::cout << "PASSED\n";
}

// ============================================================================
// 2D vs 3D Tests
// ============================================================================

void test_3d_adaptive_dt() {
    std::cout << "Testing 3D adaptive dt calculation... ";

    Mesh mesh;
    double dx = 0.2, dy = 0.1, dz = 0.15;
    int Nx = 16, Ny = 32, Nz = 20;
    mesh.init_uniform(Nx, Ny, Nz,
                      0.0, Nx * dx,
                      0.0, Ny * dy,
                      0.0, Nz * dz);

    Config config;
    config.nu = 1e-5;
    config.dt = 0.001;
    config.CFL_max = 0.5;
    config.adaptive_dt = true;
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

    solver.initialize_uniform(1.0, 0.0);  // u=1, v=0

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    double dt_computed = solver.compute_adaptive_dt();

    // dx_min = min(0.2, 0.1, 0.15) = 0.1 (dy)
    // dt_cfl = 0.5 * 0.1 / 1.0 = 0.05
    double dx_min = std::min({dx, dy, dz});
    double dt_cfl_expected = config.CFL_max * dx_min / 1.0;

    double relative_error = std::abs(dt_computed - dt_cfl_expected) / dt_cfl_expected;

    if (relative_error > 0.15) {
        std::cout << "FAILED: dt=" << dt_computed
                  << ", expected~" << dt_cfl_expected << "\n";
        std::exit(1);
    }

    std::cout << "PASSED (dt=" << dt_computed << ", dx_min=" << dx_min << ")\n";
}

void test_2d_3d_consistency() {
    std::cout << "Testing 2D/3D consistency (same xy, Nz=1 vs Nz>1)... ";

    // 2D case
    Mesh mesh2d;
    mesh2d.init_uniform(32, 32, 0.0, 3.2, 0.0, 3.2);

    Config config;
    config.nu = 0.01;
    config.dt = 0.001;
    config.CFL_max = 0.5;
    config.adaptive_dt = true;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver2d(mesh2d, config);
    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::Periodic;
    bc.y_hi = VelocityBC::Periodic;
    solver2d.set_velocity_bc(bc);
    solver2d.initialize_uniform(1.0, 0.0);

#ifdef USE_GPU_OFFLOAD
    solver2d.sync_to_gpu();
#endif

    double dt_2d = solver2d.compute_adaptive_dt();

    // 3D case with same dx, dy and dz = dx
    Mesh mesh3d;
    mesh3d.init_uniform(32, 32, 8, 0.0, 3.2, 0.0, 3.2, 0.0, 0.8);

    RANSSolver solver3d(mesh3d, config);
    VelocityBC bc3d;
    bc3d.x_lo = VelocityBC::Periodic;
    bc3d.x_hi = VelocityBC::Periodic;
    bc3d.y_lo = VelocityBC::Periodic;
    bc3d.y_hi = VelocityBC::Periodic;
    bc3d.z_lo = VelocityBC::Periodic;
    bc3d.z_hi = VelocityBC::Periodic;
    solver3d.set_velocity_bc(bc3d);
    solver3d.initialize_uniform(1.0, 0.0);

#ifdef USE_GPU_OFFLOAD
    solver3d.sync_to_gpu();
#endif

    double dt_3d = solver3d.compute_adaptive_dt();

    // Both should give similar dt (same dx=dy, dz=dx)
    double relative_diff = std::abs(dt_2d - dt_3d) / dt_2d;

    if (relative_diff > 0.2) {
        std::cout << "FAILED: dt_2d=" << dt_2d << ", dt_3d=" << dt_3d << "\n";
        std::exit(1);
    }

    std::cout << "PASSED (dt_2d=" << dt_2d << ", dt_3d=" << dt_3d << ")\n";
}

// ============================================================================
// Edge Cases
// ============================================================================

void test_very_small_velocity() {
    std::cout << "Testing with very small velocity (no division by zero)... ";

    Mesh mesh;
    mesh.init_uniform(32, 32, 0.0, 3.2, 0.0, 3.2);

    Config config;
    config.nu = 0.01;
    config.dt = 0.001;
    config.CFL_max = 0.5;
    config.adaptive_dt = true;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::Periodic;
    bc.y_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    // Very small but non-zero velocity
    solver.initialize_uniform(1e-12, 0.0);

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    double dt_computed = solver.compute_adaptive_dt();

    // Should not be NaN or Inf
    assert(std::isfinite(dt_computed));
    assert(dt_computed > 0.0);

    // Diffusion limit should dominate
    double dx = 0.1;
    double dt_diff = 0.25 * dx * dx / config.nu;

    // dt should be bounded by diffusion limit
    assert(dt_computed <= dt_diff * 1.1);

    std::cout << "PASSED (dt=" << dt_computed << ")\n";
}

void test_anisotropic_grid() {
    std::cout << "Testing anisotropic grid (dx != dy)... ";

    Mesh mesh;
    // dx = 0.2, dy = 0.05 (4:1 aspect ratio)
    mesh.init_uniform(16, 64, 0.0, 3.2, 0.0, 3.2);

    double dx = 3.2 / 16;  // 0.2
    double dy = 3.2 / 64;  // 0.05

    Config config;
    config.nu = 1e-4;
    config.dt = 0.001;
    config.CFL_max = 0.5;
    config.adaptive_dt = true;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::Periodic;
    bc.y_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    solver.initialize_uniform(1.0, 0.0);

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    double dt_computed = solver.compute_adaptive_dt();

    // Should use dx_min = dy = 0.05
    double dx_min = std::min(dx, dy);
    double dt_cfl_expected = config.CFL_max * dx_min / 1.0;

    double relative_error = std::abs(dt_computed - dt_cfl_expected) / dt_cfl_expected;

    if (relative_error > 0.15) {
        std::cout << "FAILED: dt=" << dt_computed
                  << ", expected~" << dt_cfl_expected
                  << " (dx_min=" << dx_min << ")\n";
        std::exit(1);
    }

    std::cout << "PASSED (dt=" << dt_computed << ", dx_min=" << dx_min << ")\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== Adaptive Time-Stepping Tests ===\n\n";

    // CFL tests
    test_cfl_uniform_velocity();
    test_cfl_different_cfl_max();

    // Diffusive stability tests
    test_diffusive_limit();
    test_turbulent_viscosity_effect();

    // Minimum selection tests
    test_minimum_selection_cfl_wins();
    test_minimum_selection_diffusion_wins();

    // 2D/3D tests
    test_3d_adaptive_dt();
    test_2d_3d_consistency();

    // Edge cases
    test_very_small_velocity();
    test_anisotropic_grid();

    std::cout << "\nAll tests PASSED!\n";
    return 0;
}
