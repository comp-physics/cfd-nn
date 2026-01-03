/// @file test_perf_sentinel.cpp
/// @brief Performance regression sentinel test
///
/// NOT a full benchmark suite - just a guard against catastrophic slowdowns.
/// Catches "debug flags shipped to release" or pathological kernel changes.
///
/// For each solver family, runs a representative case and checks:
///   - Time per step doesn't exceed baseline by more than 50%
///   - If it does, test FAILS with a clear warning
///
/// Baseline times are recorded empirically per build type (CPU/GPU).
/// The test is deliberately lenient (+50% tolerance) to avoid flaky CI.
///
/// Usage:
///   - Run on nightly CI or manually when profiling
///   - If test fails, investigate recent changes for performance impact

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "config.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <string>

using namespace nncfd;

// ============================================================================
// Performance test configuration
// ============================================================================

struct PerfTestCase {
    std::string name;
    int Nx, Ny, Nz;
    double Lx, Ly, Lz;
    VelocityBC::Type x_bc, y_bc, z_bc;
    PoissonSolverType solver;
    int warmup_steps;
    int timed_steps;
    // Baseline: time per step in milliseconds (very conservative)
    // These are set high to avoid false positives - we're catching 5x slowdowns, not 10% regressions
    double baseline_cpu_ms;  // Expected CPU time per step
    double baseline_gpu_ms;  // Expected GPU time per step (much faster)
};

// ============================================================================
// Run a performance test
// ============================================================================

struct PerfResult {
    bool passed;
    double time_per_step_ms;
    double baseline_ms;
    double ratio;
    std::string message;
};

PerfResult run_perf_test(const PerfTestCase& tc) {
    PerfResult result;

    // Create mesh
    Mesh mesh;
    if (tc.Nz == 1) {
        mesh.init_uniform(tc.Nx, tc.Ny, 0.0, tc.Lx, 0.0, tc.Ly);
    } else {
        mesh.init_uniform(tc.Nx, tc.Ny, tc.Nz, 0.0, tc.Lx, 0.0, tc.Ly, 0.0, tc.Lz);
    }

    // Create config
    Config config;
    config.Nx = tc.Nx;
    config.Ny = tc.Ny;
    config.Nz = tc.Nz;
    config.x_min = 0.0; config.x_max = tc.Lx;
    config.y_min = 0.0; config.y_max = tc.Ly;
    config.z_min = 0.0; config.z_max = tc.Lz;
    config.dt = 0.001;
    config.max_iter = tc.warmup_steps + tc.timed_steps + 100;
    config.nu = 0.01;
    config.poisson_solver = tc.solver;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    // Set BCs
    VelocityBC bc;
    bc.x_lo = tc.x_bc; bc.x_hi = tc.x_bc;
    bc.y_lo = tc.y_bc; bc.y_hi = tc.y_bc;
    bc.z_lo = tc.z_bc; bc.z_hi = tc.z_bc;
    solver.set_velocity_bc(bc);

    // Initialize
    VectorField vel(mesh);
    vel.fill(1.0, 0.0, 0.0);
    solver.initialize(vel);
    solver.set_body_force(0.001, 0.0, 0.0);

    // Warmup
    for (int i = 0; i < tc.warmup_steps; ++i) {
        solver.step();
    }

    // Timed run
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < tc.timed_steps; ++i) {
        solver.step();
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.time_per_step_ms = total_ms / tc.timed_steps;

    // Select baseline based on build type
#ifdef USE_GPU_OFFLOAD
    result.baseline_ms = tc.baseline_gpu_ms;
#else
    result.baseline_ms = tc.baseline_cpu_ms;
#endif

    result.ratio = result.time_per_step_ms / result.baseline_ms;

    // Pass if within 1.5x baseline (very lenient)
    const double TOLERANCE = 1.5;
    result.passed = (result.ratio <= TOLERANCE);

    if (result.passed) {
        result.message = "within tolerance";
    } else {
        char buf[256];
        snprintf(buf, sizeof(buf), "%.1fx slower than baseline (%.2f ms vs %.2f ms)",
                 result.ratio, result.time_per_step_ms, result.baseline_ms);
        result.message = buf;
    }

    return result;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "================================================================\n";
    std::cout << "  Performance Regression Sentinel Test\n";
    std::cout << "================================================================\n\n";

#ifdef USE_GPU_OFFLOAD
    std::cout << "Build: GPU (USE_GPU_OFFLOAD=ON)\n";
#else
    std::cout << "Build: CPU (USE_GPU_OFFLOAD=OFF)\n";
#endif
#ifdef USE_HYPRE
    std::cout << "HYPRE: enabled\n";
#else
    std::cout << "HYPRE: disabled\n";
#endif
    std::cout << "\n";

    // Define test cases with conservative baselines
    // Baselines are set high (2-3x typical) to avoid flaky tests
    // We're catching catastrophic slowdowns (5x+), not 10% regressions

    std::vector<PerfTestCase> tests;

    // MG solver (always available)
    tests.push_back({
        "3D_channel_MG",
        32, 32, 32,
        2.0 * M_PI, 2.0, 2.0 * M_PI,
        VelocityBC::Periodic, VelocityBC::NoSlip, VelocityBC::Periodic,
        PoissonSolverType::MG,
        5,   // warmup
        20,  // timed
        50.0,  // CPU baseline: 50ms/step (conservative)
        10.0   // GPU baseline: 10ms/step (conservative)
    });

    // 2D test (faster, good for quick validation)
    tests.push_back({
        "2D_channel_MG",
        128, 128, 1,
        2.0 * M_PI, 2.0, 1.0,
        VelocityBC::Periodic, VelocityBC::NoSlip, VelocityBC::Periodic,
        PoissonSolverType::MG,
        5,
        50,
        10.0,  // CPU baseline
        5.0    // GPU baseline
    });

#ifdef USE_HYPRE
    // HYPRE solver test
    tests.push_back({
        "3D_channel_HYPRE",
        32, 32, 32,
        2.0 * M_PI, 2.0, 2.0 * M_PI,
        VelocityBC::Periodic, VelocityBC::NoSlip, VelocityBC::Periodic,
        PoissonSolverType::HYPRE,
        5,
        20,
        20.0,  // CPU baseline: HYPRE is faster than MG
        5.0    // GPU baseline
    });
#endif

#ifdef USE_FFT_POISSON
    // FFT solver test (GPU only)
    tests.push_back({
        "3D_channel_FFT",
        32, 32, 32,
        2.0 * M_PI, 2.0, 2.0 * M_PI,
        VelocityBC::Periodic, VelocityBC::NoSlip, VelocityBC::Periodic,
        PoissonSolverType::FFT,
        5,
        20,
        100.0,  // CPU baseline (will fallback to MG)
        3.0     // GPU baseline: FFT is very fast
    });

    // FFT1D solver test (GPU only)
    tests.push_back({
        "3D_duct_FFT1D",
        32, 32, 32,
        2.0 * M_PI, 2.0, 2.0,
        VelocityBC::Periodic, VelocityBC::NoSlip, VelocityBC::NoSlip,
        PoissonSolverType::Auto,  // Will select FFT1D on GPU
        5,
        20,
        50.0,   // CPU baseline (will use MG or HYPRE)
        4.0     // GPU baseline: FFT1D is fast
    });
#endif

    std::cout << "--- Running " << tests.size() << " performance sentinel tests ---\n";
    std::cout << "    (tolerance: 1.5x baseline)\n\n";

    int passed = 0, failed = 0;

    for (const auto& tc : tests) {
        std::cout << "  " << tc.name << " (" << tc.timed_steps << " steps)... " << std::flush;

        PerfResult r = run_perf_test(tc);

        if (r.passed) {
            std::cout << "[PASS] " << std::fixed << std::setprecision(2)
                      << r.time_per_step_ms << " ms/step "
                      << "(" << std::setprecision(1) << r.ratio << "x baseline)\n";
            ++passed;
        } else {
            std::cout << "[FAIL] " << r.message << "\n";
            ++failed;
        }
    }

    // ========================================================================
    // Summary
    // ========================================================================
    std::cout << "\n================================================================\n";
    std::cout << "Performance Sentinel Summary\n";
    std::cout << "================================================================\n";
    std::cout << "  Passed: " << passed << "/" << (passed + failed) << "\n";
    std::cout << "  Failed: " << failed << "/" << (passed + failed) << "\n";

    if (failed == 0) {
        std::cout << "\n[PASS] All performance within acceptable range\n";
        return 0;
    } else {
        std::cout << "\n[FAIL] " << failed << " test(s) significantly slower than baseline\n";
        std::cout << "       Check recent changes for performance impact!\n";
        std::cout << "       (Debug flags? Accidental O0? Algorithm regression?)\n";
        return 1;
    }
}
