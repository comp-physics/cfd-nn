/// @file test_tgv_repeatability.cpp
/// @brief GPU repeatability canary test
///
/// PURPOSE: Detects non-deterministic GPU reductions or race conditions by
/// running the same simulation twice and comparing results.
///
/// For truly deterministic execution:
///   - E_final should match to machine precision (~1e-15)
///   - max|u| should match to machine precision
///
/// For GPU with non-deterministic reductions (common with atomic adds):
///   - Results may differ at ~1e-10 to 1e-12 level
///   - Test warns but doesn't fail for small differences
///
/// Test cases:
///   - 2D TGV, 32x32, 50 steps (fast)
///   - Runs simulation twice in same process
///   - Compares final kinetic energy and max velocity

#include "test_harness.hpp"
#include "test_utilities.hpp"
#include "solver.hpp"
#include "config.hpp"
#include <cmath>
#include <iomanip>

using namespace nncfd;
using namespace nncfd::test;
using nncfd::test::harness::record;
using nncfd::test::create_velocity_bc;
using nncfd::test::BCPattern;

// ============================================================================
// Helper: Run TGV simulation and return final metrics
// ============================================================================
struct TGVMetrics {
    double kinetic_energy;
    double max_velocity;
    double max_divergence;
};

static TGVMetrics run_tgv_2d(int N, int nsteps, double nu, double dt) {
    Mesh mesh;
    mesh.init_uniform(N, N, 0.0, 2.0*M_PI, 0.0, 2.0*M_PI);

    Config config;
    config.nu = nu;
    config.dt = dt;
    config.adaptive_dt = false;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);
    solver.set_velocity_bc(create_velocity_bc(BCPattern::FullyPeriodic));

    init_taylor_green(solver, mesh);
    solver.sync_to_gpu();

    for (int step = 0; step < nsteps; ++step) {
        solver.step();
    }
    solver.sync_from_gpu();

    // Compute final metrics
    TGVMetrics m;
    m.kinetic_energy = compute_kinetic_energy(mesh, solver.velocity());

    m.max_velocity = 0.0;
    m.max_divergence = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double u = 0.5 * (solver.velocity().u(i, j) + solver.velocity().u(i+1, j));
            double v = 0.5 * (solver.velocity().v(i, j) + solver.velocity().v(i, j+1));
            m.max_velocity = std::max(m.max_velocity, std::sqrt(u*u + v*v));

            double du_dx = (solver.velocity().u(i+1, j) - solver.velocity().u(i, j)) / mesh.dx;
            double dv_dy = (solver.velocity().v(i, j+1) - solver.velocity().v(i, j)) / mesh.dy;
            m.max_divergence = std::max(m.max_divergence, std::abs(du_dx + dv_dy));
        }
    }

    return m;
}

// ============================================================================
// Helper: Compute relative L2 norm of velocity difference
// Re-runs simulations to get full fields for comparison
// ============================================================================
static double compute_velocity_rel_l2(int N, double nu, double dt) {
    // Re-run simulations to get full fields for L2 comparison
    // This is a bit wasteful but keeps the metrics struct simple
    Mesh mesh;
    mesh.init_uniform(N, N, 0.0, 2.0*M_PI, 0.0, 2.0*M_PI);

    Config config;
    config.nu = nu;
    config.dt = dt;
    config.adaptive_dt = false;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    // Run 1
    RANSSolver solver1(mesh, config);
    solver1.set_velocity_bc(create_velocity_bc(BCPattern::FullyPeriodic));
    init_taylor_green(solver1, mesh);
    solver1.sync_to_gpu();
    for (int step = 0; step < 50; ++step) solver1.step();
    solver1.sync_from_gpu();

    // Run 2
    RANSSolver solver2(mesh, config);
    solver2.set_velocity_bc(create_velocity_bc(BCPattern::FullyPeriodic));
    init_taylor_green(solver2, mesh);
    solver2.sync_to_gpu();
    for (int step = 0; step < 50; ++step) solver2.step();
    solver2.sync_from_gpu();

    // Compute relative L2 norm: ||u1 - u2||_2 / ||u1||_2
    double diff_sq = 0.0, norm_sq = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double u1 = 0.5 * (solver1.velocity().u(i, j) + solver1.velocity().u(i+1, j));
            double v1 = 0.5 * (solver1.velocity().v(i, j) + solver1.velocity().v(i, j+1));
            double u2 = 0.5 * (solver2.velocity().u(i, j) + solver2.velocity().u(i+1, j));
            double v2 = 0.5 * (solver2.velocity().v(i, j) + solver2.velocity().v(i, j+1));

            diff_sq += (u1 - u2) * (u1 - u2) + (v1 - v2) * (v1 - v2);
            norm_sq += u1 * u1 + v1 * v1;
        }
    }
    return (norm_sq > 1e-30) ? std::sqrt(diff_sq / norm_sq) : std::sqrt(diff_sq);
}

// ============================================================================
// Test: Repeatability
// Robust strategy:
//   Primary: |E_final(1) - E_final(2)| / E_final(1) < 1e-10
//   Secondary: relL2(u1-u2) < 1e-10
// Avoids flaky pointwise max comparisons with small denominators.
// ============================================================================
void test_tgv_repeatability() {
    std::cout << "\n--- TGV Repeatability Test ---\n\n";

    const int N = 32;
    const int nsteps = 50;
    const double nu = 1e-3;
    const double dt = 1e-2;

    // Run simulation twice
    std::cout << "  Running simulation twice...\n";
    TGVMetrics m1 = run_tgv_2d(N, nsteps, nu, dt);
    TGVMetrics m2 = run_tgv_2d(N, nsteps, nu, dt);

    // Primary check: relative energy difference
    double E_rel = std::abs(m1.kinetic_energy - m2.kinetic_energy) / m1.kinetic_energy;

    // Always compute L2 norm for metrics (not just on failure)
    double u_rel_l2 = compute_velocity_rel_l2(N, nu, dt);

    std::cout << "\n  Run 1: E=" << std::scientific << std::setprecision(12) << m1.kinetic_energy
              << ", max|u|=" << m1.max_velocity << "\n";
    std::cout << "  Run 2: E=" << m2.kinetic_energy
              << ", max|u|=" << m2.max_velocity << "\n";
    std::cout << "\n  relE = |E1-E2|/E1 = " << E_rel << "\n";
    std::cout << "  relL2(u) = " << u_rel_l2 << "\n\n";

    // Tolerance for repeatability (1e-10 catches race conditions, allows FP reassoc)
    const double tol = 1e-10;
    bool E_pass = (E_rel < tol);
    bool L2_pass = (u_rel_l2 < tol);

    // Warn about minor non-determinism (common with GPU)
    if (E_rel > 1e-14 && E_pass) {
        std::cout << "  [INFO] Minor non-determinism: relE=" << std::scientific << E_rel << "\n";
        std::cout << "         This is expected with GPU atomic reductions.\n\n";
    }

    // Record results
    // Primary: energy check (catches gross non-determinism)
    record("Strict repeatability (rel_diff < 1e-12)", E_rel < 1e-12);

    // Fail only if both energy AND L2 checks fail
    if (!E_pass && !L2_pass) {
        record("Repeatability (relE or relL2 < 1e-10)", false);
    }

    // Always check that divergence is low
    bool div_ok = (m1.max_divergence < 1e-6) && (m2.max_divergence < 1e-6);
    record("Both runs divergence-free", div_ok);

    // Emit machine-readable QoI for CI metrics
    harness::emit_qoi_repeatability(E_rel, u_rel_l2);
}

// ============================================================================
// Test: Energy conservation consistency
// ============================================================================
void test_energy_conservation_consistency() {
    std::cout << "\n--- Energy Conservation Consistency ---\n\n";

    // Run a longer simulation and check that energy behavior is consistent
    const int N = 32;
    const int nsteps = 100;
    const double nu = 1e-3;
    const double dt = 1e-2;

    std::cout << "  Running two simulations to check energy decay consistency...\n";

    TGVMetrics m1 = run_tgv_2d(N, nsteps, nu, dt);
    TGVMetrics m2 = run_tgv_2d(N, nsteps, nu, dt);

    // Get initial energy for reference
    TGVMetrics m0 = run_tgv_2d(N, 0, nu, dt);  // 0 steps = initial state

    double decay1 = m1.kinetic_energy / m0.kinetic_energy;
    double decay2 = m2.kinetic_energy / m0.kinetic_energy;
    double decay_diff = std::abs(decay1 - decay2);

    std::cout << "  Decay ratio 1: " << std::fixed << std::setprecision(8) << decay1 << "\n";
    std::cout << "  Decay ratio 2: " << decay2 << "\n";
    std::cout << "  Difference: " << std::scientific << decay_diff << "\n\n";

    record("Energy decay consistent (diff < 1e-10)", decay_diff < 1e-10);
}

// ============================================================================
// Main
// ============================================================================
int main() {
    return harness::run("TGV Repeatability Tests", []() {
        test_tgv_repeatability();
        test_energy_conservation_consistency();
    });
}
