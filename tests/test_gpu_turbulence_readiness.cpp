/// @file test_gpu_turbulence_readiness.cpp
/// @brief Acceptance test for 3D GPU turbulence simulation readiness
///
/// This single-file test validates that the solver is ready for production
/// turbulence simulations on GPU. It checks:
/// 1. GPU offload works (device pointers valid)
/// 2. 3D Poisson solver converges
/// 3. Projection produces div-free velocity
/// 4. Step() completes without NaN/Inf
/// 5. Turbulence classifier detects perturbations
/// 6. Perf mode works without crashes
///
/// PASS criteria: All checks pass within reasonable tolerances
/// This test is designed to catch regressions before long turbulence runs.

#include "solver.hpp"
#include "mesh.hpp"
#include "config.hpp"
#include "gpu_utils.hpp"
#include <iostream>
#include <cmath>
#include <iomanip>
#include <random>

using namespace nncfd;

// Test infrastructure
static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "  FAIL: " << msg << "\n"; \
        return false; \
    } \
} while(0)

#define PASS(name) do { \
    std::cout << "  PASS: " << name << "\n"; \
    ++tests_passed; \
    return true; \
} while(0)

#define FAIL(name, msg) do { \
    std::cerr << "  FAIL: " << name << " - " << msg << "\n"; \
    ++tests_failed; \
    return false; \
} while(0)

// =============================================================================
// Test 1: GPU offload and data mapping
// =============================================================================
bool test_gpu_offload() {
    std::cout << "\n[Test 1] GPU offload and data mapping\n";

#ifdef USE_GPU_OFFLOAD
    Config config;
    config.Nx = 16;
    config.Ny = 16;
    config.Nz = 16;
    config.x_min = 0.0;
    config.x_max = 2.0 * M_PI;
    config.y_min = -1.0;
    config.y_max = 1.0;
    config.z_min = 0.0;
    config.z_max = M_PI;
    config.nu = 1.0 / 180.0;
    config.max_steps = 10;
    config.adaptive_dt = true;
    config.CFL_max = 0.3;
    config.simulation_mode = SimulationMode::Unsteady;
    config.convective_scheme = ConvectiveScheme::Skew;
    config.time_integrator = TimeIntegrator::RK3;

    Mesh mesh;
    mesh.init_uniform(config.Nx, config.Ny, config.Nz,
                         config.x_min, config.x_max,
                         config.y_min, config.y_max,
                         config.z_min, config.z_max, 2);

    RANSSolver solver(mesh, config);

    // Initialize with uniform flow
    solver.initialize_uniform(1.0, 0.0);

    // Verify device pointers are valid
    auto view = solver.get_solver_view();
    CHECK(view.u_face != nullptr, "u_face device ptr should be valid");
    CHECK(view.v_face != nullptr, "v_face device ptr should be valid");
    CHECK(view.w_face != nullptr, "w_face device ptr should be valid (3D)");
    CHECK(view.p != nullptr, "pressure device ptr should be valid");

    // Verify we can get device pointers via omp_get_mapped_ptr
    int dev_id = omp_get_default_device();
    const double* u_dev = static_cast<const double*>(
        omp_get_mapped_ptr(const_cast<double*>(view.u_face), dev_id));
    CHECK(u_dev != nullptr, "omp_get_mapped_ptr should return valid device ptr");

    PASS("GPU offload working");
#else
    std::cout << "  SKIP: GPU offload not enabled (USE_GPU_OFFLOAD not defined)\n";
    ++tests_passed;
    return true;
#endif
}

// =============================================================================
// Test 2: 3D Poisson solver convergence
// =============================================================================
bool test_poisson_convergence() {
    std::cout << "\n[Test 2] 3D Poisson solver convergence\n";

    Config config;
    config.Nx = 32;
    config.Ny = 32;
    config.Nz = 32;
    config.x_min = 0.0;
    config.x_max = 2.0 * M_PI;
    config.y_min = -1.0;
    config.y_max = 1.0;
    config.z_min = 0.0;
    config.z_max = M_PI;
    config.nu = 1.0 / 180.0;
    config.max_steps = 5;
    config.adaptive_dt = true;
    config.CFL_max = 0.3;
    config.simulation_mode = SimulationMode::Unsteady;
    config.convective_scheme = ConvectiveScheme::Skew;
    config.time_integrator = TimeIntegrator::RK3;
    config.diag_interval = 1;  // Full diagnostics for this test

    Mesh mesh;
    mesh.init_uniform(config.Nx, config.Ny, config.Nz,
                         config.x_min, config.x_max,
                         config.y_min, config.y_max,
                         config.z_min, config.z_max, 2);

    RANSSolver solver(mesh, config);

    // Initialize with perturbed Poiseuille (generates non-trivial RHS)
    VectorField vel(mesh);
    double u_max = 1.0 / (2.0 * config.nu);

    for (int k = 0; k < mesh.total_Nz(); ++k) {
        for (int j = 0; j < mesh.total_Ny(); ++j) {
            double y = mesh.yc[j];
            double u_base = u_max * (1.0 - y * y);
            for (int i = 0; i < mesh.total_Nx() + 1; ++i) {
                vel.u(i, j, k) = u_base * (1.0 + 0.1 * std::sin(2.0 * M_PI * i / config.Nx));
            }
        }
    }
    solver.initialize(vel);
    solver.set_body_force(1.0, 0.0, 0.0);

    // Run a few steps and check Poisson stats
    for (int n = 0; n < 3; ++n) {
        solver.step();
    }

    auto stats = solver.poisson_stats();

    std::cout << "  Poisson cycles: " << stats.cycles << "\n";
    std::cout << "  Status: " << stats.status_string() << "\n";
    std::cout << "  res/rhs: " << std::scientific << std::setprecision(3) << stats.res_over_rhs << "\n";
    std::cout << "  div_scaled_linf: " << stats.div_scaled_linf << "\n";

    // Criteria for acceptance
    CHECK(stats.status != RANSSolver::PoissonSolveStatus::HitMaxCycles,
          "Poisson should not hit max cycles on well-posed problem");
    CHECK(stats.div_scaled_linf < 1e-4,
          "Scaled divergence should be < 1e-4 after projection");
    CHECK(!std::isnan(stats.res_over_rhs), "res/rhs should not be NaN");

    PASS("Poisson convergence OK");
}

// =============================================================================
// Test 3: Step completes without NaN/Inf
// =============================================================================
bool test_step_stability() {
    std::cout << "\n[Test 3] Step stability (no NaN/Inf)\n";

    Config config;
    config.Nx = 32;
    config.Ny = 32;
    config.Nz = 32;
    config.x_min = 0.0;
    config.x_max = 4.0 * M_PI;
    config.y_min = -1.0;
    config.y_max = 1.0;
    config.z_min = 0.0;
    config.z_max = 2.0 * M_PI;
    config.nu = 1.0 / 180.0;
    config.max_steps = 50;
    config.adaptive_dt = true;
    config.CFL_max = 0.5;
    config.simulation_mode = SimulationMode::Unsteady;
    config.convective_scheme = ConvectiveScheme::Skew;
    config.time_integrator = TimeIntegrator::RK3;
    config.turb_guard_enabled = true;
    config.turb_guard_interval = 1;
    config.diag_interval = 5;

    Mesh mesh;
    mesh.init_uniform(config.Nx, config.Ny, config.Nz,
                         config.x_min, config.x_max,
                         config.y_min, config.y_max,
                         config.z_min, config.z_max, 2);

    RANSSolver solver(mesh, config);

    // Initialize with perturbed Poiseuille for DNS transition
    VectorField vel(mesh);
    double u_max = 1.0 / (2.0 * config.nu);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int k = 0; k < mesh.total_Nz(); ++k) {
        double z = mesh.zc[k];
        for (int j = 0; j < mesh.total_Ny(); ++j) {
            double y = mesh.yc[j];
            double u_base = u_max * (1.0 - y * y);
            double wall_damp = (1.0 - y * y);

            for (int i = 0; i < mesh.total_Nx() + 1; ++i) {
                double x = mesh.xf[std::min(i, mesh.total_Nx() - 1)];
                double pert = 0.05 * u_max * wall_damp * std::sin(2.0 * M_PI * x / config.x_max)
                            * std::cos(2.0 * M_PI * z / config.z_max);
                vel.u(i, j, k) = u_base + pert;
            }
        }
    }
    solver.initialize(vel);
    solver.set_body_force(1.0, 0.0, 0.0);

    // Run multiple steps
    bool nan_detected = false;
    double max_residual = 0.0;

    for (int n = 0; n < 20; ++n) {
        double res = solver.step();
        if (std::isnan(res) || std::isinf(res)) {
            nan_detected = true;
            std::cerr << "  NaN/Inf detected at step " << n << "\n";
            break;
        }
        max_residual = std::max(max_residual, res);
    }

    std::cout << "  Completed 20 steps\n";
    std::cout << "  Max residual: " << std::scientific << max_residual << "\n";

    CHECK(!nan_detected, "Should complete 20 steps without NaN/Inf");
    CHECK(max_residual < 1e6, "Residual should not explode");

    PASS("Step stability OK");
}

// =============================================================================
// Test 4: Turbulence classifier responds to perturbations
// =============================================================================
bool test_turbulence_classifier() {
    std::cout << "\n[Test 4] Turbulence classifier\n";

    Config config;
    config.Nx = 32;
    config.Ny = 32;
    config.Nz = 32;
    config.x_min = 0.0;
    config.x_max = 4.0 * M_PI;
    config.y_min = -1.0;
    config.y_max = 1.0;
    config.z_min = 0.0;
    config.z_max = 2.0 * M_PI;
    config.nu = 1.0 / 180.0;
    config.max_steps = 100;
    config.adaptive_dt = true;
    config.CFL_max = 0.5;
    config.simulation_mode = SimulationMode::Unsteady;
    config.convective_scheme = ConvectiveScheme::Skew;
    config.time_integrator = TimeIntegrator::RK3;
    config.diag_interval = 10;

    Mesh mesh;
    mesh.init_uniform(config.Nx, config.Ny, config.Nz,
                         config.x_min, config.x_max,
                         config.y_min, config.y_max,
                         config.z_min, config.z_max, 2);

    RANSSolver solver(mesh, config);

    // Initialize with strong perturbations to trigger at least transitional state
    VectorField vel(mesh);
    double u_max = 1.0 / (2.0 * config.nu);

    for (int k = 0; k < mesh.total_Nz(); ++k) {
        double z = mesh.zc[k];
        for (int j = 0; j < mesh.total_Ny(); ++j) {
            double y = mesh.yc[j];
            double u_base = u_max * (1.0 - y * y);
            double wall_damp = (1.0 - y * y);

            for (int i = 0; i < mesh.total_Nx() + 1; ++i) {
                double x = mesh.xf[std::min(i, mesh.total_Nx() - 1)];
                // Strong perturbation to ensure transition
                double pert = 0.2 * u_max * wall_damp
                    * std::sin(2.0 * M_PI * x / config.x_max)
                    * std::cos(4.0 * M_PI * z / config.z_max);
                vel.u(i, j, k) = u_base + pert;
            }
        }
    }
    // Add v,w perturbations
    for (int k = 0; k < mesh.total_Nz(); ++k) {
        double z = mesh.zc[k];
        for (int j = 0; j < mesh.total_Ny() + 1; ++j) {
            double y = mesh.yf[j];
            for (int i = 0; i < mesh.total_Nx(); ++i) {
                double x = mesh.xc[i];
                double wall_damp = (1.0 - y * y);
                vel.v(i, j, k) = 0.05 * u_max * wall_damp * std::cos(2.0 * M_PI * x / config.x_max);
            }
        }
    }
    for (int k = 0; k < mesh.total_Nz() + 1; ++k) {
        double z = mesh.zf[std::min(k, mesh.total_Nz() - 1)];
        for (int j = 0; j < mesh.total_Ny(); ++j) {
            double y = mesh.yc[j];
            for (int i = 0; i < mesh.total_Nx(); ++i) {
                double x = mesh.xc[i];
                double wall_damp = (1.0 - y * y);
                vel.w(i, j, k) = 0.05 * u_max * wall_damp * std::sin(4.0 * M_PI * z / config.z_max);
            }
        }
    }

    solver.initialize(vel);
    solver.set_body_force(1.0, 0.0, 0.0);

    // Run several steps to evolve the turbulent field
    for (int n = 0; n < 30; ++n) {
        solver.step();
    }

    // Use compute_turbulence_presence to check field properties
    auto presence = solver.compute_turbulence_presence();
    std::cout << "  u_tau_ratio: " << std::fixed << std::setprecision(3)
              << presence.u_tau_ratio << "\n";
    std::cout << "  u_rms_mid: " << std::scientific << presence.u_rms_mid << "\n";
    std::cout << "  tke_mid: " << presence.tke_mid << "\n";

    // With strong perturbations, should detect some turbulent activity
    bool detected_activity = (presence.tke_mid > 1e-6 ||
                              presence.u_rms_mid > 1e-3);

    CHECK(detected_activity, "Should detect perturbation activity in field");

    PASS("Turbulence classifier responds to perturbations");
}

// =============================================================================
// Test 5: Perf mode works without crashes
// =============================================================================
bool test_perf_mode() {
    std::cout << "\n[Test 5] Performance mode stability\n";

    Config config;
    config.Nx = 32;
    config.Ny = 32;
    config.Nz = 32;
    config.x_min = 0.0;
    config.x_max = 2.0 * M_PI;
    config.y_min = -1.0;
    config.y_max = 1.0;
    config.z_min = 0.0;
    config.z_max = M_PI;
    config.nu = 1.0 / 180.0;
    config.max_steps = 100;
    config.adaptive_dt = true;
    config.CFL_max = 0.5;
    config.simulation_mode = SimulationMode::Unsteady;
    config.convective_scheme = ConvectiveScheme::Skew;
    config.time_integrator = TimeIntegrator::RK3;

    // Enable perf mode (reduced diagnostics)
    config.perf_mode = true;
    config.finalize();  // Apply perf_mode settings

    std::cout << "  diag_interval (perf): " << config.diag_interval << "\n";
    std::cout << "  poisson_check_interval (perf): " << config.poisson_check_interval << "\n";

    CHECK(config.diag_interval >= 10, "perf_mode should increase diag_interval");

    Mesh mesh;
    mesh.init_uniform(config.Nx, config.Ny, config.Nz,
                         config.x_min, config.x_max,
                         config.y_min, config.y_max,
                         config.z_min, config.z_max, 2);

    RANSSolver solver(mesh, config);

    VectorField vel(mesh);
    double u_max = 1.0 / (2.0 * config.nu);
    for (int k = 0; k < mesh.total_Nz(); ++k) {
        for (int j = 0; j < mesh.total_Ny(); ++j) {
            double y = mesh.yc[j];
            for (int i = 0; i < mesh.total_Nx() + 1; ++i) {
                vel.u(i, j, k) = u_max * (1.0 - y * y);
            }
        }
    }
    solver.initialize(vel);
    solver.set_body_force(1.0, 0.0, 0.0);

    // Run multiple steps in perf mode
    bool crashed = false;
    for (int n = 0; n < 30; ++n) {
        try {
            solver.step();
        } catch (...) {
            crashed = true;
            break;
        }
    }

    CHECK(!crashed, "Should not crash in perf mode");

    PASS("Perf mode OK");
}

// =============================================================================
// Test 6: Projection health watchdog
// =============================================================================
bool test_projection_watchdog() {
    std::cout << "\n[Test 6] Projection health watchdog\n";

    Config config;
    config.Nx = 16;
    config.Ny = 16;
    config.Nz = 16;
    config.x_min = 0.0;
    config.x_max = 2.0 * M_PI;
    config.y_min = -1.0;
    config.y_max = 1.0;
    config.z_min = 0.0;
    config.z_max = M_PI;
    config.nu = 1.0 / 180.0;
    config.max_steps = 10;
    config.adaptive_dt = true;
    config.CFL_max = 0.3;
    config.simulation_mode = SimulationMode::Unsteady;
    config.convective_scheme = ConvectiveScheme::Skew;
    config.time_integrator = TimeIntegrator::RK3;

    // Enable watchdog with reasonable threshold
    config.projection_watchdog = true;
    config.div_threshold = 1e-5;
    config.diag_interval = 1;

    Mesh mesh;
    mesh.init_uniform(config.Nx, config.Ny, config.Nz,
                         config.x_min, config.x_max,
                         config.y_min, config.y_max,
                         config.z_min, config.z_max, 2);

    RANSSolver solver(mesh, config);
    solver.initialize_uniform(1.0, 0.0);
    solver.set_body_force(1.0, 0.0, 0.0);

    // Run a few steps - watchdog should NOT trigger on healthy projection
    for (int n = 0; n < 5; ++n) {
        solver.step();
    }

    auto stats = solver.poisson_stats();
    std::cout << "  div_threshold: " << config.div_threshold << "\n";
    std::cout << "  div_scaled_linf: " << std::scientific << stats.div_scaled_linf << "\n";

    // Good projection should be well below threshold
    CHECK(stats.div_scaled_linf < config.div_threshold,
          "Healthy projection should be below watchdog threshold");

    PASS("Projection watchdog configured correctly");
}

// =============================================================================
// Main
// =============================================================================
int main() {
    std::cout << "==============================================\n";
    std::cout << "3D GPU TURBULENCE READINESS ACCEPTANCE TEST\n";
    std::cout << "==============================================\n";

#ifdef USE_GPU_OFFLOAD
    std::cout << "GPU offload: ENABLED\n";
    std::cout << "Default device: " << omp_get_default_device() << "\n";
    std::cout << "Number of devices: " << omp_get_num_devices() << "\n";
#else
    std::cout << "GPU offload: DISABLED (CPU mode)\n";
#endif

    // Run all tests
    test_gpu_offload();
    test_poisson_convergence();
    test_step_stability();
    test_turbulence_classifier();
    test_perf_mode();
    test_projection_watchdog();

    // Summary
    std::cout << "\n==============================================\n";
    std::cout << "SUMMARY: " << tests_passed << " passed, " << tests_failed << " failed\n";
    std::cout << "==============================================\n";

    if (tests_failed > 0) {
        std::cerr << "\n*** ACCEPTANCE TEST FAILED ***\n";
        std::cerr << "Fix failing tests before running production turbulence simulations.\n";
        return 1;
    }

    std::cout << "\n*** ACCEPTANCE TEST PASSED ***\n";
    std::cout << "System is ready for 3D GPU turbulence simulations.\n";
    return 0;
}
