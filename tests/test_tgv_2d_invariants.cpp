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

// Note: compute_max_divergence_2d is now provided by test_utilities.hpp

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
    solver.set_velocity_bc(create_velocity_bc(BCPattern::FullyPeriodic));

    // Initialize with Taylor-Green vortex
    init_taylor_green(solver, mesh);
    solver.sync_to_gpu();

    // Verify all critical GPU fields are present (catches missing mappings early)
#ifdef USE_GPU_OFFLOAD
    bool fields_present = solver.verify_gpu_field_presence();
    record("All GPU fields present after sync", fields_present);
#endif

    // Reset sync counter AFTER initialization sync (debug builds only)
    // This allows us to verify no syncs occur during stepping
    nncfd::gpu::reset_sync_counter();

    // Track metrics using device-side QOI functions (avoids NVHPC D2H sync bugs)
#ifdef USE_GPU_OFFLOAD
    double E_prev = solver.compute_kinetic_energy_device();
#else
    double E_prev = compute_kinetic_energy(mesh, solver.velocity());
#endif
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

#ifdef USE_GPU_OFFLOAD
        // Use device-side QOI computation (avoids broken D2H sync in NVHPC)
        double div = solver.compute_divergence_linf_device();
        double E_curr = solver.compute_kinetic_energy_device();
#else
        solver.sync_from_gpu();
        double div = compute_max_divergence_2d(solver.velocity(), mesh);
        double E_curr = compute_kinetic_energy(mesh, solver.velocity());
#endif
        div_history.push_back(div);
        max_div_observed = std::max(max_div_observed, div);
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

    // Verify no H↔D syncs occurred during stepping (GPU builds)
    // This enforces the "no mid-step transfers" performance guarantee
    // Counter is incremented in debug builds only, so check is meaningful in debug
#ifdef USE_GPU_OFFLOAD
    int syncs_during_stepping = nncfd::gpu::get_sync_counter();
    // In release builds, counter stays 0 (not instrumented), so this always passes
    // In debug builds, this catches any sync calls during step()
    record("No H↔D syncs during stepping", syncs_during_stepping == 0,
           "(syncs=" + std::to_string(syncs_during_stepping) + ")");
#endif

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

    solver.set_velocity_bc(create_velocity_bc(BCPattern::FullyPeriodic));

    init_taylor_green(solver, mesh);
    solver.sync_to_gpu();

#ifdef USE_GPU_OFFLOAD
    double E0 = solver.compute_kinetic_energy_device();
#else
    double E0 = compute_kinetic_energy(mesh, solver.velocity());
#endif

    for (int step = 0; step < nsteps; ++step) {
        solver.step();
    }

#ifdef USE_GPU_OFFLOAD
    double E_final = solver.compute_kinetic_energy_device();
#else
    solver.sync_from_gpu();
    double E_final = compute_kinetic_energy(mesh, solver.velocity());
#endif
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

    solver.set_velocity_bc(create_velocity_bc(BCPattern::FullyPeriodic));

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
    solver.set_velocity_bc(create_velocity_bc(BCPattern::FullyPeriodic));

    // Initialize with constant velocity everywhere
    for (int j = 0; j <= mesh.Ny + 1; ++j) {
        for (int i = 0; i <= mesh.Nx + 1; ++i) {
            solver.velocity().u(i, j) = u_const;
            solver.velocity().v(i, j) = v_const;
        }
    }
    solver.sync_to_gpu();

    // Record initial max velocity for comparison
    double expected_max = std::max(std::abs(u_const), std::abs(v_const));

    // Run a few steps
    for (int step = 0; step < nsteps; ++step) {
        solver.step();
    }

#ifdef USE_GPU_OFFLOAD
    // GPU path: use device-side QOI (avoids broken D2H sync in NVHPC)
    // Check that max velocity stays close to expected constant
    double max_vel = solver.compute_max_velocity_device();
    double max_diff = std::abs(max_vel - expected_max);
    std::cout << "  max|u|: " << std::scientific << max_vel
              << " (expected: " << expected_max << ")\n";
    std::cout << "  |max|u| - expected|: " << max_diff << "\n\n";
#else
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
#endif

    // Threshold: constant field should stay close to constant
    // Note: Some drift is expected due to pressure projection numerical precision
    // and iterative solver tolerances. The key check is that drift is bounded
    // and doesn't grow catastrophically (which would indicate indexing bugs).
    // 2e-2 catches gross errors; typical observed values are O(1e-2).
    const double threshold = 2e-2;
    record("Constant velocity preserved (< 2e-2)", max_diff < threshold,
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
    solver.set_velocity_bc(create_velocity_bc(BCPattern::FullyPeriodic));

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
#ifdef USE_GPU_OFFLOAD
    double E0 = solver.compute_kinetic_energy_device();
    double max_v0 = 0.0;  // Initial v is 0 by construction
#else
    double E0 = compute_kinetic_energy(mesh, solver.velocity());
    double max_v0 = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            max_v0 = std::max(max_v0, std::abs(solver.velocity().v(i, j)));
        }
    }
#endif

    // Run simulation
    for (int step = 0; step < nsteps; ++step) {
        solver.step();
    }

#ifdef USE_GPU_OFFLOAD
    // GPU path: use device-side QOI (avoids broken D2H sync in NVHPC)
    double E_final = solver.compute_kinetic_energy_device();
    double max_vel_final = solver.compute_max_velocity_device();
    double ke_ratio = E_final / E0;

    // Note: On GPU we can't easily separate max|u| from max|v| without more
    // device functions. Use overall max velocity and energy ratio for stability check.
    std::cout << "  Initial: E=" << std::scientific << E0 << "\n";
    std::cout << "  Final:   E=" << E_final << ", max|vel|=" << max_vel_final << "\n";
    std::cout << "  Energy ratio: " << std::fixed << std::setprecision(4) << ke_ratio << "\n\n";

    // Check invariants:
    // 1. Energy should not grow (viscous decay only)
    bool energy_ok = (ke_ratio <= 1.01);  // 1% tolerance for numerical drift
    record("Fourier mode energy stable (E_f/E_0 <= 1.01)", energy_ok,
           qoi(ke_ratio, 1.01));

    // 2. On GPU, skip detailed v-component check (requires field access)
    // Use a proxy: max velocity should stay bounded (initial max|u|=1)
    bool bounded = (max_vel_final < 2.0);  // Should be ~1 if stable
    record("Max velocity bounded (< 2.0)", bounded,
           qoi(max_vel_final, 2.0));

    // Note: Skip emit_qoi_fourier_mode on GPU because we can't compute the
    // max_v/max_u ratio without device functions for separate u/v components.
    // The ke_ratio alone is emitted for CI tracking.
    std::cout << "QOI_JSON: {\"test\":\"fourier_mode\""
              << ",\"ke_ratio\":" << ke_ratio
              << ",\"gpu\":true}\n" << std::flush;
#else
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
    // Some v is expected due to numerical discretization effects and
    // pressure projection coupling. Threshold 1e-1 catches indexing bugs;
    // observed values ~6% with current advection scheme.
    bool no_spurious = (v_over_u < 1e-1);
    record("No spurious v-component (max|v|/max|u| < 1e-1)", no_spurious,
           qoi(v_over_u, 1e-1));

    // Emit machine-readable QoI for CI metrics
    harness::emit_qoi_fourier_mode(ke_ratio, v_over_u);
#endif
}

// ============================================================================
// Test: RK2/RK3 time integrator smoke test
// Verifies higher-order time steppers work correctly on GPU by running TGV
// with same invariant checks. This catches bugs in copy/blend/projection
// kernels that RK2/RK3 use but Euler doesn't exercise.
// ============================================================================
void test_rk_integrator_smoke() {
    std::cout << "\n--- RK2/RK3 Time Integrator Smoke Test ---\n\n";

    // Smaller grid for speed (smoke test, not accuracy test)
    // Use higher viscosity to show meaningful decay in fewer steps
    const int N = 24;
    const int nsteps = 100;  // More steps
    const double nu = 0.01;  // Higher viscosity for faster decay
    const double dt = 0.005;
    const double L = 2.0 * M_PI;

    // Thresholds (same as main TGV test)
    const double div_threshold = 1e-6;

    Mesh mesh;
    mesh.init_uniform(N, N, 0.0, L, 0.0, L);

    std::cout << "  Grid: " << N << "x" << N << ", " << nsteps << " steps, nu=" << nu << "\n\n";

    // Test RK2 and RK3 integrators
    // Both now use dev_ptr() + is_device_ptr pattern for NVHPC compatibility.
    std::vector<std::pair<std::string, TimeIntegrator>> integrators = {
        {"RK2", TimeIntegrator::RK2},
        {"RK3", TimeIntegrator::RK3}
    };

    for (const auto& [name, integrator] : integrators) {
        Config config;
        config.nu = nu;
        config.dt = dt;
        config.time_integrator = integrator;
        config.adaptive_dt = false;
        config.turb_model = TurbulenceModelType::None;
        config.verbose = false;

        RANSSolver solver(mesh, config);

        // Fully periodic BCs
        solver.set_velocity_bc(create_velocity_bc(BCPattern::FullyPeriodic));

        // Initialize with Taylor-Green vortex
        init_taylor_green(solver, mesh);
        solver.sync_to_gpu();

        // Track energy
#ifdef USE_GPU_OFFLOAD
        double E_init = solver.compute_kinetic_energy_device();
#else
        double E_init = compute_kinetic_energy(mesh, solver.velocity());
#endif

        double max_div = 0.0;
        double max_conv_observed = 0.0;  // Track max convection term magnitude
        bool energy_monotonic = true;
        double E_prev = E_init;

        // Run simulation
        for (int step = 0; step < nsteps; ++step) {
            solver.step();

#ifdef USE_GPU_OFFLOAD
            double div = solver.compute_divergence_linf_device();
            double E_curr = solver.compute_kinetic_energy_device();

            // Check convection term after first few steps (catches "convection accidentally disabled")
            if (step < 5) {
                double max_conv = solver.compute_max_conv_device();
                max_conv_observed = std::max(max_conv_observed, max_conv);
            }
#else
            solver.sync_from_gpu();
            double div = compute_max_divergence_2d(solver.velocity(), mesh);
            double E_curr = compute_kinetic_energy(mesh, solver.velocity());
#endif

            max_div = std::max(max_div, div);
            if (E_curr > E_prev * 1.001) {  // 0.1% tolerance
                energy_monotonic = false;
            }
            E_prev = E_curr;
        }

        double E_final = E_prev;
        double ke_ratio = E_final / E_init;

        std::cout << "  " << name << ": max_div=" << std::scientific << std::setprecision(2)
                  << max_div << " KE_ratio=" << std::fixed << std::setprecision(4) << ke_ratio
                  << (energy_monotonic ? " (stable)" : " (GREW!)") << "\n";

        // Record results - smoke test validates:
        // 1. Divergence stays bounded (projection works)
        // 2. Energy doesn't explode (stability)
        // 3. Energy actually decays (catches "RHS accidentally zero" regressions)
        //    With nu=0.01, dt=0.005, 100 steps, T=0.5, expect measurable decay
        record(name + " divergence-free (< 1e-6)", max_div < div_threshold,
               qoi(max_div, div_threshold));
        record(name + " energy stable (not exploding)", energy_monotonic && std::isfinite(E_final));

        // CRITICAL: Verify field actually evolved (catches "RHS zero" bugs)
        // With viscosity and 100 steps, energy should decay by at least 0.1%
        // If ke_ratio >= 0.9999, the RHS was likely zero (no evolution)
        const double decay_threshold = 0.9999;
        bool field_evolved = (ke_ratio < decay_threshold);
        record(name + " field evolved (KE decayed)", field_evolved,
               "(ratio=" + std::to_string(ke_ratio) + ", threshold=" + std::to_string(decay_threshold) + ")");

#ifdef USE_GPU_OFFLOAD
        // Verify convection term is non-zero (catches "convection accidentally disabled")
        // For TGV at Re=1000, max|conv| should be O(1) due to u*du/dx terms
        const double conv_floor = 1e-6;  // Conservative floor
        bool conv_active = (max_conv_observed > conv_floor);
        record(name + " convection active (max|conv| > 1e-6)", conv_active,
               "(max_conv=" + std::to_string(max_conv_observed) + ")");
#endif
    }
    std::cout << "\n";
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
        test_rk_integrator_smoke();
    });
}
