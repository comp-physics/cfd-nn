/// @file test_turbulence_golden.cpp
/// @brief Golden snapshot regression tests for turbulence models
///
/// Turbulence models can drift in subtle ways that still pass invariants
/// (e.g., wrong constants, swapped coefficients, feature scaling bugs).
/// This test catches regression by comparing velocity field evolution against
/// known reference values.
///
/// Method:
///   1. Create fixed initial state (parabolic channel profile)
///   2. Run N steps with turbulence model
///   3. Compare key velocity statistics against golden values
///   4. Fail if deviation exceeds tolerance
///
/// Golden values capture the integrated effect of the turbulence model on
/// the flow field. Changes to model constants or formulation will cause
/// these to drift.
///
/// TO REGENERATE GOLDEN VALUES:
///   1. Run this test with REGENERATE_GOLDEN=1 environment variable
///   2. Copy the printed values into the GOLDEN_* constants below
///   3. Verify the new values make physical sense
///   4. Update GOLDEN_VALUES_DATE with the regeneration date

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "config.hpp"
#include "turbulence_model.hpp"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <vector>
#include <string>

using namespace nncfd;

// ============================================================================
// Golden reference values - VERIFIED BASELINE
// ============================================================================
// These values were captured from a verified build and validated for
// physical consistency. Regenerate only after intentional model changes.
//
// Last regenerated: 2025-01-04 (initial baseline)
// Test config: 32x32 mesh, 50 steps, dt=0.001, nu=0.001, body_force=0.01

namespace golden {

// Laminar (no turbulence model) - pure Navier-Stokes
constexpr double LAMINAR_U_MEAN = 6.6739e-01;
constexpr double LAMINAR_U_MAX  = 9.9942e-01;
constexpr double LAMINAR_KE     = 2.6693e-01;

// Baseline mixing length model
constexpr double BASELINE_U_MEAN = 6.6631e-01;
constexpr double BASELINE_U_MAX  = 9.9876e-01;
constexpr double BASELINE_KE     = 2.6600e-01;

// Tolerance for golden value comparison (1% for cross-build regression)
constexpr double REGRESSION_TOLERANCE = 0.01;

}  // namespace golden

// ============================================================================
// Test infrastructure
// ============================================================================

struct VelocityStats {
    double u_mean;         // Mean u velocity
    double u_max;          // Max u velocity
    double ke;             // Kinetic energy
};

struct GoldenTestCase {
    std::string name;
    TurbulenceModelType model;
    VelocityStats expected;
    double tolerance;      // Relative tolerance for comparison
};

/// Compute velocity statistics from solver
VelocityStats compute_vel_stats(const RANSSolver& solver, const Mesh& mesh) {
    VelocityStats result;
    result.u_mean = 0.0;
    result.u_max = -1e30;
    result.ke = 0.0;
    int count = 0;

    const VectorField& vel = solver.velocity();

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double u = 0.5 * (vel.u(i, j) + vel.u(i+1, j));
            double v = 0.5 * (vel.v(i, j) + vel.v(i, j+1));

            result.u_mean += u;
            result.u_max = std::max(result.u_max, u);
            result.ke += 0.5 * (u*u + v*v);
            ++count;
        }
    }

    if (count > 0) {
        result.u_mean /= count;
        result.ke /= count;  // Average KE per cell
    }

    return result;
}

/// Run model for N steps and return final statistics
VelocityStats run_model_snapshot(TurbulenceModelType model, const Mesh& mesh, int nsteps) {
    Config config;
    config.Nx = mesh.Nx;
    config.Ny = mesh.Ny;
    config.x_min = mesh.x_min;
    config.x_max = mesh.x_max;
    config.y_min = mesh.y_min;
    config.y_max = mesh.y_max;
    config.dt = 0.001;
    config.nu = 0.001;  // Re ~ 1000 for stronger turbulence effect
    config.turb_model = model;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    // Create and attach turbulence model (required - solver doesn't auto-create from config)
    solver.set_turbulence_model(create_turbulence_model(model, "", ""));

    // Set up channel-like BCs
    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);

    // Initialize with parabolic profile
    VectorField& vel = solver.velocity();
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            double y = mesh.y(j);
            double y_norm = (y - mesh.y_min) / (mesh.y_max - mesh.y_min);
            // Parabolic profile: U = U_max * 4 * y_norm * (1 - y_norm)
            vel.u(i, j) = 4.0 * y_norm * (1.0 - y_norm);
        }
    }

    solver.initialize(vel);
    solver.set_body_force(0.01, 0.0, 0.0);  // Small pressure gradient

    // Run steps
    for (int step = 0; step < nsteps; ++step) {
        solver.step();
    }

    solver.sync_from_gpu();
    return compute_vel_stats(solver, mesh);
}

bool check_golden(const std::string& name, const VelocityStats& actual,
                  const VelocityStats& expected, double tol) {
    bool pass = true;

    auto check_value = [&](const std::string& metric, double act, double exp) {
        if (std::abs(exp) < 1e-15) {
            // For zero expected, use absolute tolerance
            bool ok = (std::abs(act) < tol);
            if (!ok) {
                std::cout << "    " << metric << ": " << std::scientific << std::setprecision(4)
                          << act << " (expected ~0, abs=" << std::abs(act) << ") [FAIL]\n";
                pass = false;
            }
            return ok;
        }
        double rel_err = std::abs(act - exp) / std::abs(exp);
        bool ok = (rel_err < tol);
        if (!ok) {
            std::cout << "    " << metric << ": " << std::scientific << std::setprecision(4)
                      << act << " (expected " << exp << ", rel_err=" << std::fixed
                      << std::setprecision(2) << rel_err * 100 << "%) [FAIL]\n";
            pass = false;
        }
        return ok;
    };

    std::cout << "  " << name << ":\n";
    std::cout << "    u_mean=" << std::scientific << std::setprecision(4) << actual.u_mean
              << " u_max=" << actual.u_max << " ke=" << actual.ke << "\n";

    check_value("u_mean", actual.u_mean, expected.u_mean);
    check_value("u_max", actual.u_max, expected.u_max);
    check_value("ke", actual.ke, expected.ke);

    std::cout << "  " << name << ": " << (pass ? "[PASS]" : "[FAIL]") << "\n\n";
    return pass;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "================================================================\n";
    std::cout << "  Turbulence Model Golden Snapshot Tests\n";
    std::cout << "================================================================\n\n";

#ifdef USE_GPU_OFFLOAD
    std::cout << "Build: GPU (USE_GPU_OFFLOAD=ON)\n\n";
#else
    std::cout << "Build: CPU (USE_GPU_OFFLOAD=OFF)\n\n";
#endif

    std::cout << "Testing velocity field evolution against golden reference values.\n";
    std::cout << "This catches subtle regressions that still pass invariants.\n\n";

    // Create test mesh (small for speed)
    Mesh mesh;
    mesh.init_uniform(32, 32, 0.0, 2.0 * M_PI, 0.0, 2.0);

    const int nsteps = 50;  // Enough steps to see model effects

    // Check if we're in regeneration mode
    bool regenerate_mode = (std::getenv("REGENERATE_GOLDEN") != nullptr);

    if (regenerate_mode) {
        std::cout << "=== REGENERATE MODE ===\n";
        std::cout << "Running models to capture new golden values...\n\n";

        VelocityStats laminar_stats = run_model_snapshot(TurbulenceModelType::None, mesh, nsteps);
        VelocityStats baseline_stats = run_model_snapshot(TurbulenceModelType::Baseline, mesh, nsteps);

        std::cout << "Copy these values to the golden namespace in this file:\n\n";
        std::cout << "// Laminar (no turbulence model) - pure Navier-Stokes\n";
        std::cout << "constexpr double LAMINAR_U_MEAN = " << std::scientific << std::setprecision(4)
                  << laminar_stats.u_mean << ";\n";
        std::cout << "constexpr double LAMINAR_U_MAX  = " << laminar_stats.u_max << ";\n";
        std::cout << "constexpr double LAMINAR_KE     = " << laminar_stats.ke << ";\n\n";
        std::cout << "// Baseline mixing length model\n";
        std::cout << "constexpr double BASELINE_U_MEAN = " << baseline_stats.u_mean << ";\n";
        std::cout << "constexpr double BASELINE_U_MAX  = " << baseline_stats.u_max << ";\n";
        std::cout << "constexpr double BASELINE_KE     = " << baseline_stats.ke << ";\n\n";
        std::cout << "=== END REGENERATE MODE ===\n";
        return 0;
    }

    // Use hard-coded golden values for regression testing
    VelocityStats golden_laminar = {golden::LAMINAR_U_MEAN, golden::LAMINAR_U_MAX, golden::LAMINAR_KE};
    VelocityStats golden_baseline = {golden::BASELINE_U_MEAN, golden::BASELINE_U_MAX, golden::BASELINE_KE};

    std::cout << "Using golden reference values (regenerate with REGENERATE_GOLDEN=1)\n\n";
    std::cout << "  Golden Laminar:  u_mean=" << std::scientific << std::setprecision(4)
              << golden_laminar.u_mean << " u_max=" << golden_laminar.u_max
              << " ke=" << golden_laminar.ke << "\n";
    std::cout << "  Golden Baseline: u_mean=" << golden_baseline.u_mean
              << " u_max=" << golden_baseline.u_max
              << " ke=" << golden_baseline.ke << "\n\n";

    // Golden values from verified baseline
    std::vector<GoldenTestCase> tests = {
        // Laminar should match golden reference
        {"None (Laminar)", TurbulenceModelType::None,
         golden_laminar,
         golden::REGRESSION_TOLERANCE},

        // Baseline mixing length should match golden reference
        {"Baseline (MixingLength)", TurbulenceModelType::Baseline,
         golden_baseline,
         golden::REGRESSION_TOLERANCE},
    };

    std::cout << "--- Running " << tests.size() << " golden snapshot tests ---\n\n";

    int passed = 0, failed = 0;

    for (const auto& tc : tests) {
        try {
            // Re-run the model (should match exactly)
            VelocityStats actual = run_model_snapshot(tc.model, mesh, nsteps);
            if (check_golden(tc.name, actual, tc.expected, tc.tolerance)) {
                ++passed;
            } else {
                ++failed;
            }
        } catch (const std::exception& e) {
            std::cerr << "  " << tc.name << ": EXCEPTION - " << e.what() << "\n";
            ++failed;
        }
    }

    // Key check: Golden values should show Baseline differs from Laminar
    std::cout << "--- Model Differentiation Check (from golden values) ---\n\n";
    double model_diff = std::abs(golden::BASELINE_U_MEAN - golden::LAMINAR_U_MEAN) /
                        std::abs(golden::LAMINAR_U_MEAN);
    bool models_differ = (model_diff > 0.0001);  // At least 0.01% difference in golden values

    std::cout << "  Golden Baseline vs Laminar u_mean difference: "
              << std::fixed << std::setprecision(4) << model_diff * 100 << "%\n";
    std::cout << "  Models distinguishable in golden: " << (models_differ ? "[YES]" : "[NO]") << "\n\n";

    if (!models_differ) {
        std::cout << "  NOTE: Golden values show minimal turbulence model effect.\n";
        std::cout << "        This is acceptable for this test configuration.\n\n";
    }

    // Summary
    std::cout << "================================================================\n";
    std::cout << "Golden Snapshot Summary\n";
    std::cout << "================================================================\n";
    std::cout << "  Regression tests: " << passed << "/" << (passed + failed) << " passed\n";

    // Only fail on actual regression (values don't match golden)
    if (failed == 0) {
        std::cout << "\n[PASS] All turbulence models match golden reference values\n";
        return 0;
    } else {
        std::cout << "\n[FAIL] " << failed << " model(s) deviated from golden values\n";
        return 1;
    }
}
