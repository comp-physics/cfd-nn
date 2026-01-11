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

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::Periodic;
    bc.y_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

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
// Test: Repeatability
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

    // Compute differences
    double E_diff = std::abs(m1.kinetic_energy - m2.kinetic_energy);
    double E_rel = E_diff / m1.kinetic_energy;
    double vel_diff = std::abs(m1.max_velocity - m2.max_velocity);

    std::cout << "\n  Run 1: E=" << std::scientific << std::setprecision(12) << m1.kinetic_energy
              << ", max|u|=" << m1.max_velocity << "\n";
    std::cout << "  Run 2: E=" << m2.kinetic_energy
              << ", max|u|=" << m2.max_velocity << "\n";
    std::cout << "\n  E_diff: " << E_diff << " (rel: " << E_rel << ")\n";
    std::cout << "  max|u| diff: " << vel_diff << "\n\n";

    // Strict threshold (true determinism)
    const double strict_tol = 1e-12;
    bool strict_pass = (E_rel < strict_tol) && (vel_diff < strict_tol);

    // Relaxed threshold (allows minor GPU non-determinism)
    const double relaxed_tol = 1e-10;
    bool relaxed_pass = (E_rel < relaxed_tol) && (vel_diff < relaxed_tol);

    if (!strict_pass && relaxed_pass) {
        std::cout << "  [WARN] Minor non-determinism detected (within 1e-10)\n";
        std::cout << "         This is common with GPU atomic reductions\n\n";
    }

    // Record results
    record("Strict repeatability (rel_diff < 1e-12)", strict_pass);

    // Only record relaxed as failure if it fails
    if (!relaxed_pass) {
        record("Relaxed repeatability (rel_diff < 1e-10)", false);
    }

    // Always check that divergence is low
    bool div_ok = (m1.max_divergence < 1e-6) && (m2.max_divergence < 1e-6);
    record("Both runs divergence-free", div_ok);
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
