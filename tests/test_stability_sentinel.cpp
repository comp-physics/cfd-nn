/// @file test_stability_sentinel.cpp
/// @brief Stability sentinel test - verifies safety clamps work under aggressive settings
///
/// PURPOSE: Catches accidental removal of dt limiters, nu_eff handling regressions,
/// or projection bugs that show up as rising divergence/energy.
///
/// DESIGN: Runs a short case with intentionally aggressive user settings (high CFL),
/// and relies on the code's safety clamps to keep it stable. If clamps are removed
/// or broken, the simulation will blow up.
///
/// Checks:
///   - No NaN/Inf in velocity or pressure
///   - ke_ratio <= 1.05 (energy shouldn't grow significantly)
///   - div_Linf <= 1e-5 (divergence stays bounded)
///   - dt_final logged for trend tracking
///
/// Emits QOI_JSON for trend tracking:
///   {"test":"stability_sentinel","ke_ratio":...,"div_Linf":...,"dt_min":...,"dt_max":...}

#include "test_harness.hpp"
#include "test_utilities.hpp"
#include "mesh.hpp"
#include "solver.hpp"
#include "config.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>

using namespace nncfd;
using namespace nncfd::test;
using nncfd::test::harness::record;
using nncfd::test::BCPattern;
using nncfd::test::create_velocity_bc;

// ============================================================================
// Helper: Check for NaN/Inf in velocity field
// ============================================================================
static bool has_nan_inf(const VectorField& v, const Mesh& mesh) {
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double u = v.u(i, j);
            double vv = v.v(i, j);
            if (std::isnan(u) || std::isinf(u) || std::isnan(vv) || std::isinf(vv)) {
                return true;
            }
        }
    }
    return false;
}

// ============================================================================
// Helper: Compute max divergence
// ============================================================================
static double compute_max_div(const VectorField& v, const Mesh& mesh) {
    double max_div = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double du_dx = (v.u(i+1, j) - v.u(i, j)) / mesh.dx;
            double dv_dy = (v.v(i, j+1) - v.v(i, j)) / mesh.dy;
            max_div = std::max(max_div, std::abs(du_dx + dv_dy));
        }
    }
    return max_div;
}

// ============================================================================
// Stability sentinel test
// ============================================================================
struct SentinelResult {
    bool passed;
    bool has_nan;
    double ke_initial;
    double ke_final;
    double ke_ratio;
    double div_linf;
    double dt_min;
    double dt_max;
    double dt_final;
};

SentinelResult run_stability_sentinel() {
    SentinelResult result;
    result.passed = true;
    result.has_nan = false;
    result.dt_min = 1e10;
    result.dt_max = 0.0;

    // 2D TGV on small grid
    // Tests that the simulation stays stable with slightly aggressive CFL
    // If dt limiter or stability clamps are broken, this will blow up
    const int N = 32;
    const int nsteps = 100;
    const double nu = 1e-3;  // Same as other TGV tests

    Mesh mesh;
    mesh.init_uniform(N, N, 0.0, 2.0*M_PI, 0.0, 2.0*M_PI);

    Config config;
    config.nu = nu;
    config.dt = 1e-2;  // Reasonable starting dt
    config.adaptive_dt = true;
    config.CFL_max = 0.8;  // Slightly higher than default 0.5
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);
    solver.set_velocity_bc(create_velocity_bc(BCPattern::FullyPeriodic));

    // Initialize with TGV
    init_taylor_green(solver, mesh);
    solver.sync_to_gpu();

    // Compute initial energy
    solver.sync_from_gpu();
    result.ke_initial = compute_kinetic_energy(mesh, solver.velocity());

    // Run simulation, tracking dt
    for (int step = 0; step < nsteps; ++step) {
        solver.step();
        double current_dt = solver.current_dt();
        result.dt_min = std::min(result.dt_min, current_dt);
        result.dt_max = std::max(result.dt_max, current_dt);
        result.dt_final = current_dt;
    }

    solver.sync_from_gpu();

    // Check for NaN/Inf
    result.has_nan = has_nan_inf(solver.velocity(), mesh);
    if (result.has_nan) {
        result.passed = false;
    }

    // Compute final metrics
    result.ke_final = compute_kinetic_energy(mesh, solver.velocity());
    result.ke_ratio = result.ke_final / result.ke_initial;
    result.div_linf = compute_max_div(solver.velocity(), mesh);

    // Check energy didn't explode (should decay or stay bounded for TGV)
    if (result.ke_ratio > 1.05) {
        result.passed = false;
    }

    // Check divergence is bounded
    if (result.div_linf > 1e-5) {
        result.passed = false;
    }

    return result;
}

// ============================================================================
// Emit QoI for CI tracking
// ============================================================================
static void emit_qoi_sentinel(const SentinelResult& r) {
    std::cout << "QOI_JSON: {\"test\":\"stability_sentinel\""
              << ",\"ke_ratio\":" << harness::json_double(r.ke_ratio)
              << ",\"div_Linf\":" << harness::json_double(r.div_linf)
              << ",\"dt_min\":" << harness::json_double(r.dt_min)
              << ",\"dt_max\":" << harness::json_double(r.dt_max)
              << ",\"dt_final\":" << harness::json_double(r.dt_final)
              << ",\"has_nan\":" << (r.has_nan ? "true" : "false")
              << "}\n";
}

// ============================================================================
// Main test function
// ============================================================================
void test_stability_sentinel() {
    std::cout << "\n--- Stability Sentinel Test ---\n\n";
    std::cout << "  Running 2D TGV with aggressive CFL settings...\n";
    std::cout << "  (Verifies dt limiter and stability clamps work)\n\n";

    SentinelResult r = run_stability_sentinel();

    // Print results
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  KE initial:  " << r.ke_initial << "\n";
    std::cout << "  KE final:    " << r.ke_final << "\n";
    std::cout << "  KE ratio:    " << r.ke_ratio << " (limit: 1.05)\n";
    std::cout << "  div_Linf:    " << std::scientific << r.div_linf << " (limit: 1e-5)\n";
    std::cout << std::fixed;
    std::cout << "  dt range:    [" << r.dt_min << ", " << r.dt_max << "]\n";
    std::cout << "  dt final:    " << r.dt_final << "\n";
    std::cout << "  NaN/Inf:     " << (r.has_nan ? "YES (BAD)" : "no") << "\n\n";

    // Emit QoI
    emit_qoi_sentinel(r);

    // Record results
    record("No NaN/Inf in velocity", !r.has_nan);
    record("Energy bounded (ke_ratio <= 1.05)", r.ke_ratio <= 1.05);
    record("Divergence bounded (div_Linf <= 1e-5)", r.div_linf <= 1e-5);
    record("Stability sentinel overall", r.passed);
}

// ============================================================================
// Main
// ============================================================================
int main() {
    return harness::run("Stability Sentinel Tests", []() {
        test_stability_sentinel();
    });
}
