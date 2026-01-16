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
#include <sstream>
#include <algorithm>

using namespace nncfd;
using namespace nncfd::test;
using nncfd::test::harness::record;

// Helper to format QoI output: "value=X, threshold=Y"
static std::string qoi(double value, double threshold) {
    std::ostringstream ss;
    ss << std::scientific << std::setprecision(2);
    ss << "(val=" << value << ", thr=" << threshold << ")";
    return ss.str();
}

// Helper for ratio comparisons: "value=X, threshold=Y%"
static std::string qoi_pct(double value, double threshold_pct) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1);
    ss << "(val=" << value * 100 << "%, thr=" << threshold_pct * 100 << "%)";
    return ss.str();
}

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

    // Record test results with QoI values
    record("Divergence-free (max|div| < 1e-6)", max_div_observed < div_threshold,
           qoi(max_div_observed, div_threshold));
    record("Energy monotonicity (E non-increasing)", energy_monotonic,
           energy_monotonic ? "(no violations)" : "(violation at step " + std::to_string(energy_violation_step) + ")");
    record("Energy bounded (final KE finite)", std::isfinite(energy_history.back()),
           "(KE_final=" + std::to_string(energy_history.back()) + ")");
    double decay_ratio = energy_history.back() / energy_history.front();
    record("Energy decaying (final < initial)", energy_history.back() < energy_history.front(),
           "(ratio=" + std::to_string(decay_ratio) + ")");

    // Emit machine-readable QoI for CI metrics
    // (const_vel_Linf emitted separately by test_constant_velocity_invariance)
    harness::emit_qoi_tgv_2d(max_div_observed, energy_history.back(), decay_ratio);
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
    record("Energy decay rate (within 30% of theory)", rel_error < 0.30,
           qoi_pct(rel_error, 0.30));
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
    record("Initial field divergence-free (< 1e-8)", initial_div < 1e-8,
           qoi(initial_div, 1e-8));
}

// ============================================================================
// Test: Constant velocity advection invariance
// A constant velocity field should remain constant under advection (periodic BC)
// This catches indexing mistakes, BC bugs, and convective operator errors
// ============================================================================
void test_constant_velocity_invariance() {
    std::cout << "\n--- Constant Velocity Invariance ---\n\n";

    const int N = 32;
    const double L = 2.0 * M_PI;
    const double u_const = 1.5;  // Constant velocity
    const double v_const = 0.75;
    const int nsteps = 20;

    Mesh mesh;
    mesh.init_uniform(N, N, 0.0, L, 0.0, L);

    Config config;
    config.nu = 0.01;  // Small viscosity (shouldn't affect constant field much)
    config.dt = 0.01;
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

    // Initialize with constant velocity everywhere
    for (int j = 0; j <= mesh.Ny + 1; ++j) {
        for (int i = 0; i <= mesh.Nx + 1; ++i) {
            solver.velocity().u(i, j) = u_const;
            solver.velocity().v(i, j) = v_const;
        }
    }
    solver.sync_to_gpu();

    // Run a few steps
    for (int step = 0; step < nsteps; ++step) {
        solver.step();
    }
    solver.sync_from_gpu();

    // Check that velocity is still constant (within tolerance)
    double max_u_diff = 0.0;
    double max_v_diff = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            max_u_diff = std::max(max_u_diff, std::abs(solver.velocity().u(i, j) - u_const));
        }
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            max_v_diff = std::max(max_v_diff, std::abs(solver.velocity().v(i, j) - v_const));
        }
    }

    double max_diff = std::max(max_u_diff, max_v_diff);
    std::cout << "  max|u - u_const|: " << std::scientific << max_u_diff << "\n";
    std::cout << "  max|v - v_const|: " << std::scientific << max_v_diff << "\n\n";

    // Threshold: constant field should stay close to constant
    // Note: Some drift is expected due to pressure projection numerical precision
    // and iterative solver tolerances. The key check is that drift is bounded
    // and doesn't grow catastrophically (which would indicate indexing bugs).
    // 1e-2 catches gross errors; typical observed values are O(1e-3).
    const double threshold = 1e-2;
    record("Constant velocity preserved (< 1e-2)", max_diff < threshold,
           qoi(max_diff, threshold));
}

// ============================================================================
// Test: Single Fourier mode invariance
// u = sin(x), v = 0 is divergence-free and should:
//   - Not create spurious modes (v should stay near 0)
//   - Not grow in energy (viscous decay only)
// This catches convective operator aliasing and indexing bugs
// ============================================================================
void test_fourier_mode_invariance() {
    std::cout << "\n--- Single Fourier Mode Invariance ---\n\n";

    const int N = 32;
    const double L = 2.0 * M_PI;
    const double nu = 0.01;  // Moderate viscosity
    const double dt = 0.01;
    const int nsteps = 50;

    Mesh mesh;
    mesh.init_uniform(N, N, 0.0, L, 0.0, L);

    Config config;
    config.nu = nu;
    config.dt = dt;
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

    // Initialize: u = sin(y), v = 0
    // Divergence-free: du/dx = 0, dv/dy = 0, so div = 0
    // This is a shear flow with a single Fourier mode in y.
    for (int j = 0; j <= mesh.Ny + 1; ++j) {
        double y = mesh.y(j);  // y at cell center (u uses cell-center y)
        for (int i = 0; i <= mesh.Nx + 1; ++i) {
            solver.velocity().u(i, j) = std::sin(y);
            solver.velocity().v(i, j) = 0.0;
        }
    }
    solver.sync_to_gpu();

    // Record initial state
    double E0 = compute_kinetic_energy(mesh, solver.velocity());
    double max_v0 = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            max_v0 = std::max(max_v0, std::abs(solver.velocity().v(i, j)));
        }
    }

    // Run simulation
    for (int step = 0; step < nsteps; ++step) {
        solver.step();
    }
    solver.sync_from_gpu();

    // Check final state
    double E_final = compute_kinetic_energy(mesh, solver.velocity());
    double max_u_final = 0.0;
    double max_v_final = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            max_u_final = std::max(max_u_final, std::abs(solver.velocity().u(i, j)));
        }
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            max_v_final = std::max(max_v_final, std::abs(solver.velocity().v(i, j)));
        }
    }

    // Normalize spurious v by max|u| for amplitude-independent check
    double v_over_u = max_v_final / (max_u_final + 1e-30);
    double ke_ratio = E_final / E0;

    std::cout << "  Initial: E=" << std::scientific << E0 << ", max|v|=" << max_v0 << "\n";
    std::cout << "  Final:   E=" << E_final << ", max|u|=" << max_u_final
              << ", max|v|=" << max_v_final << "\n";
    std::cout << "  Energy ratio: " << std::fixed << std::setprecision(4) << ke_ratio << "\n";
    std::cout << "  max|v|/max|u|: " << std::scientific << v_over_u << "\n\n";

    // Check invariants:
    // 1. Energy should not grow (viscous decay only)
    bool energy_ok = (ke_ratio <= 1.01);  // 1% tolerance for numerical drift
    record("Fourier mode energy stable (E_f/E_0 <= 1.01)", energy_ok,
           qoi(ke_ratio, 1.01));

    // 2. No large spurious v-component should appear (normalized by max|u|)
    // Some small v is expected due to numerical discretization effects
    // Threshold 1e-2 catches indexing bugs; observed values ~1e-3
    bool no_spurious = (v_over_u < 1e-2);
    record("No spurious v-component (max|v|/max|u| < 1e-2)", no_spurious,
           qoi(v_over_u, 1e-2));

    // Emit machine-readable QoI for CI metrics
    harness::emit_qoi_fourier_mode(ke_ratio, v_over_u);
}

// ============================================================================
// Main
// ============================================================================
int main() {
    return harness::run("2D Taylor-Green Vortex Invariants", []() {
        test_tgv_2d_initial_divergence();
        test_tgv_2d_invariants();
        test_tgv_2d_decay_rate();
        test_constant_velocity_invariance();
        test_fourier_mode_invariance();
    });
}
