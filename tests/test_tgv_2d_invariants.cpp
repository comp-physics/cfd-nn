/// @file test_tgv_2d_invariants.cpp
/// @brief 2D Taylor-Green vortex invariant tests for CI
///
/// PURPOSE: Catches projection bugs, wrong operator ordering, GPU stale fields,
/// and indexing mistakes by verifying physical invariants that should hold for
/// any correct incompressible projection method:
///
///   1. Divergence-free: max|div(u)| should remain small after projection
///   2. Energy monotonicity: For nu > 0, kinetic energy should not increase
///
/// This test validates the coupling of:
///   advection + diffusion + pressure projection + BC application + GPU sync
///
/// Test cases:
///   - 32x32 grid, 200 steps (fast, runs on every push)
///   - Fully periodic domain [0, 2π]²
///   - Viscosity nu = 1e-3

#include "test_harness.hpp"
#include "test_utilities.hpp"
#include "solver.hpp"
#include "config.hpp"
#include <cmath>
#include <vector>
#include <iomanip>
#include <algorithm>

using namespace nncfd;
using namespace nncfd::test;
using nncfd::test::harness::record;

// ============================================================================
// Helper: Compute max divergence (L-infinity norm)
// ============================================================================
static double compute_max_divergence_2d(const VectorField& vel, const Mesh& mesh) {
    double max_div = 0.0;
    double dx = mesh.dx;
    double dy = mesh.dy;

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            // MAC grid: u at x-faces, v at y-faces
            double du_dx = (vel.u(i+1, j) - vel.u(i, j)) / dx;
            double dv_dy = (vel.v(i, j+1) - vel.v(i, j)) / dy;
            double div = std::abs(du_dx + dv_dy);
            max_div = std::max(max_div, div);
        }
    }
    return max_div;
}

// ============================================================================
// Test: 2D Taylor-Green Vortex Invariants
// ============================================================================
void test_tgv_2d_invariants() {
    std::cout << "\n--- 2D Taylor-Green Vortex Invariants ---\n\n";

    // Configuration: 32x32 periodic, nu=1e-3, 200 steps
    const int N = 32;
    const int nsteps = 200;
    const double nu = 1e-3;
    const double dt_max = 1e-2;  // Cap dt for determinism
    const double L = 2.0 * M_PI;

    // Thresholds (conservative start, will print actual values)
    const double div_threshold = 1e-6;
    const double energy_growth_tol = 1e-12;

    // Setup mesh and config
    Mesh mesh;
    mesh.init_uniform(N, N, 0.0, L, 0.0, L);

    Config config;
    config.nu = nu;
    config.dt = dt_max;
    config.adaptive_dt = false;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    // Fully periodic BCs
    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::Periodic;
    bc.y_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    // Initialize with Taylor-Green vortex
    init_taylor_green(solver, mesh);
    solver.sync_to_gpu();

    // Track metrics
    double E_prev = compute_kinetic_energy(mesh, solver.velocity());
    double max_div_observed = 0.0;
    bool energy_monotonic = true;
    int energy_violation_step = -1;
    double worst_energy_growth = 0.0;

    std::vector<double> div_history;
    std::vector<double> energy_history;
    energy_history.push_back(E_prev);

    // Run simulation, check invariants every step
    for (int step = 1; step <= nsteps; ++step) {
        solver.step();
        solver.sync_from_gpu();

        // Compute divergence
        double div = compute_max_divergence_2d(solver.velocity(), mesh);
        div_history.push_back(div);
        max_div_observed = std::max(max_div_observed, div);

        // Compute energy
        double E_curr = compute_kinetic_energy(mesh, solver.velocity());
        energy_history.push_back(E_curr);

        // Check energy monotonicity: E(t+dt) <= E(t) * (1 + tol)
        if (E_curr > E_prev * (1.0 + energy_growth_tol)) {
            if (energy_monotonic) {
                energy_monotonic = false;
                energy_violation_step = step;
                worst_energy_growth = (E_curr - E_prev) / E_prev;
            } else {
                double growth = (E_curr - E_prev) / E_prev;
                if (growth > worst_energy_growth) {
                    worst_energy_growth = growth;
                }
            }
        }

        E_prev = E_curr;
    }

    // Print diagnostic info
    std::cout << "  Grid: " << N << "x" << N << ", steps: " << nsteps
              << ", nu: " << nu << ", dt: " << dt_max << "\n";
    std::cout << "  max_div_Linf observed: " << std::scientific << std::setprecision(2)
              << max_div_observed << " (threshold: " << div_threshold << ")\n";
    std::cout << "  KE decay: " << std::fixed << std::setprecision(4)
              << energy_history.back() / energy_history.front()
              << " (initial: " << std::scientific << energy_history.front() << ")\n";

    if (!energy_monotonic) {
        std::cout << "  [WARN] Energy grew at step " << energy_violation_step
                  << ", max growth: " << std::scientific << worst_energy_growth << "\n";
    }
    std::cout << "\n";

    // Record test results
    record("Divergence-free (max|div| < 1e-6)", max_div_observed < div_threshold);
    record("Energy monotonicity (E non-increasing)", energy_monotonic);
    record("Energy bounded (final KE finite)", std::isfinite(energy_history.back()));
    record("Energy decaying (final < initial)", energy_history.back() < energy_history.front());
}

// ============================================================================
// Test: Energy decay rate (optional, validates physics)
// ============================================================================
void test_tgv_2d_decay_rate() {
    std::cout << "\n--- 2D Taylor-Green Decay Rate ---\n\n";

    // For TGV, exact decay rate: E(t) = E(0) * exp(-4*nu*t) for viscous case
    const int N = 48;  // Slightly finer for accuracy
    const double nu = 0.01;
    const double T = 0.5;
    const double dt = 0.005;
    const int nsteps = static_cast<int>(T / dt);

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

    double E0 = compute_kinetic_energy(mesh, solver.velocity());

    for (int step = 0; step < nsteps; ++step) {
        solver.step();
    }
    solver.sync_from_gpu();

    double E_final = compute_kinetic_energy(mesh, solver.velocity());
    double E_theory = E0 * std::exp(-4.0 * nu * T);
    double rel_error = std::abs(E_final - E_theory) / E_theory;

    std::cout << "  T=" << T << ", nu=" << nu << ", steps=" << nsteps << "\n";
    std::cout << "  KE ratio: " << std::fixed << std::setprecision(4) << E_final/E0
              << ", theory: " << E_theory/E0 << "\n";
    std::cout << "  Relative error: " << std::scientific << rel_error * 100 << "%\n\n";

    // 30% tolerance accounts for numerical dissipation on coarse grid
    record("Energy decay rate (within 30% of theory)", rel_error < 0.30);
}

// ============================================================================
// Test: Initial divergence (verify init_taylor_green is div-free)
// ============================================================================
void test_tgv_2d_initial_divergence() {
    std::cout << "\n--- Initial Divergence Check ---\n\n";

    const int N = 32;
    Mesh mesh;
    mesh.init_uniform(N, N, 0.0, 2.0*M_PI, 0.0, 2.0*M_PI);

    Config config;
    config.nu = 0.01;
    config.dt = 0.001;
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

    // Check divergence BEFORE any time stepping
    double initial_div = compute_max_divergence_2d(solver.velocity(), mesh);

    std::cout << "  Initial max|div|: " << std::scientific << initial_div << "\n\n";

    // Taylor-Green is analytically divergence-free, but discrete divergence depends on:
    //   - face-center sampling of u,v
    //   - periodic wrap indexing
    //   - floating-point roundoff
    // Use 1e-8 threshold: stricter than simulation check (1e-6), but allows for
    // minor discrete/roundoff effects. The "during simulation" check is the key invariant.
    record("Initial field divergence-free (< 1e-8)", initial_div < 1e-8);
}

// ============================================================================
// Main
// ============================================================================
int main() {
    return harness::run("2D Taylor-Green Vortex Invariants", []() {
        test_tgv_2d_initial_divergence();
        test_tgv_2d_invariants();
        test_tgv_2d_decay_rate();
    });
}
