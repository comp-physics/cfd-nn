/// @file test_skew_energy_conservation.cpp
/// @brief Validates skew-symmetric advection conserves kinetic energy
///
/// PURPOSE: The skew-symmetric (split) form of advection should satisfy
/// <u, conv(u)> = 0 in the discrete sense for periodic BCs, meaning
/// advection itself does not produce or destroy kinetic energy.
///
/// This is the key property that makes skew-symmetric advection suitable
/// for DNS/LES: it prevents spurious energy accumulation at the grid scale
/// (spectral blocking) that can cause numerical blow-up.
///
/// Test strategy:
///   1. Run TGV with near-zero viscosity (inviscid limit)
///   2. Verify KE is bounded (no catastrophic growth)
///   3. Compare <u, conv(u)> between skew and non-skew schemes
///   4. Verify skew-symmetric has smaller |<u, conv(u)>| than central

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

// Note: compute_kinetic_energy_3d is now provided by test_utilities.hpp

// ============================================================================
// Helper: Initialize 3D Taylor-Green vortex
// ============================================================================
static void init_taylor_green_3d(RANSSolver& solver, const Mesh& mesh) {
    // Classic 3D TGV: u = sin(x)cos(y)cos(z), v = -cos(x)sin(y)cos(z), w = 0
    // u at x-faces
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                double x = mesh.xf[i];
                double y = mesh.y(j);
                double z = mesh.z(k);
                solver.velocity().u(i, j, k) = std::sin(x) * std::cos(y) * std::cos(z);
            }
        }
    }
    // v at y-faces
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double x = mesh.x(i);
                double y = mesh.yf[j];
                double z = mesh.z(k);
                solver.velocity().v(i, j, k) = -std::cos(x) * std::sin(y) * std::cos(z);
            }
        }
    }
    // w at z-faces (zero for classic TGV)
    for (int k = mesh.k_begin(); k <= mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                solver.velocity().w(i, j, k) = 0.0;
            }
        }
    }
}

// Helper to format QoI output
static std::string qoi(double value, double threshold) {
    std::ostringstream ss;
    ss << std::scientific << std::setprecision(2);
    ss << "(val=" << value << ", thr=" << threshold << ")";
    return ss.str();
}

// ============================================================================
// Helper: Run simulation and track energy metrics
// ============================================================================
struct EnergyMetrics {
    double ke_initial;
    double ke_final;
    double ke_max;
    double ke_min;
    double max_conv_ke_prod;      // max |<u, conv(u)>| over all steps
    double avg_conv_ke_prod;      // average |<u, conv(u)>| over all steps
    int steps_run;
    bool exploded;                // KE grew by > 100x
};

static EnergyMetrics run_energy_tracking(
    const Mesh& mesh,
    Config config,
    int nsteps
) {
    EnergyMetrics metrics = {};
    metrics.steps_run = nsteps;
    metrics.exploded = false;

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

    // Initial energy
#ifdef USE_GPU_OFFLOAD
    metrics.ke_initial = solver.compute_kinetic_energy_device();
#else
    metrics.ke_initial = compute_kinetic_energy(mesh, solver.velocity());
#endif
    metrics.ke_max = metrics.ke_initial;
    metrics.ke_min = metrics.ke_initial;

    double sum_conv_ke_prod = 0.0;
    int conv_ke_count = 0;

    // Run simulation
    for (int step = 1; step <= nsteps; ++step) {
        solver.step();

#ifdef USE_GPU_OFFLOAD
        double ke = solver.compute_kinetic_energy_device();
#else
        solver.sync_from_gpu();
        double ke = compute_kinetic_energy(mesh, solver.velocity());
#endif
        metrics.ke_max = std::max(metrics.ke_max, ke);
        metrics.ke_min = std::min(metrics.ke_min, ke);

        // Check for explosion
        if (ke > 100.0 * metrics.ke_initial || !std::isfinite(ke)) {
            metrics.exploded = true;
            metrics.ke_final = ke;
            return metrics;
        }

        // Compute <u, conv(u)> - only on CPU (requires field access)
        // Note: This tests the instantaneous energy production rate from advection
#ifndef USE_GPU_OFFLOAD
        double conv_ke_prod = std::abs(solver.compute_convective_ke_production());
        metrics.max_conv_ke_prod = std::max(metrics.max_conv_ke_prod, conv_ke_prod);
        sum_conv_ke_prod += conv_ke_prod;
        conv_ke_count++;
#endif
    }

#ifdef USE_GPU_OFFLOAD
    metrics.ke_final = solver.compute_kinetic_energy_device();
#else
    metrics.ke_final = compute_kinetic_energy(mesh, solver.velocity());
    if (conv_ke_count > 0) {
        metrics.avg_conv_ke_prod = sum_conv_ke_prod / conv_ke_count;
    }
#endif

    return metrics;
}

// ============================================================================
// Test: Skew-symmetric energy boundedness (inviscid limit)
// ============================================================================
void test_skew_energy_bounded() {
    std::cout << "\n--- Skew-Symmetric Energy Boundedness (Inviscid Limit) ---\n\n";

    // Configuration: Small grid, near-zero viscosity, many steps
    const int N = 32;
    const int nsteps = 500;
    const double nu = 1e-8;  // Near-inviscid
    const double dt = 0.005;
    const double L = 2.0 * M_PI;

    Mesh mesh;
    mesh.init_uniform(N, N, 0.0, L, 0.0, L);

    Config config;
    config.nu = nu;
    config.dt = dt;
    config.adaptive_dt = false;
    config.turb_model = TurbulenceModelType::None;
    config.convective_scheme = ConvectiveScheme::Skew;
    config.verbose = false;

    std::cout << "  Grid: " << N << "x" << N << ", steps: " << nsteps << "\n";
    std::cout << "  nu: " << std::scientific << nu << " (near-inviscid)\n";
    std::cout << "  Scheme: Skew-symmetric\n\n";

    EnergyMetrics m = run_energy_tracking(mesh, config, nsteps);

    double ke_ratio = m.ke_final / m.ke_initial;
    double ke_growth = (m.ke_max - m.ke_initial) / m.ke_initial;

    std::cout << "  KE initial: " << std::scientific << m.ke_initial << "\n";
    std::cout << "  KE final:   " << m.ke_final << "\n";
    std::cout << "  KE max:     " << m.ke_max << "\n";
    std::cout << "  KE ratio (final/init): " << std::fixed << std::setprecision(6) << ke_ratio << "\n";
    std::cout << "  KE max growth: " << std::scientific << ke_growth * 100 << "%\n";
#ifndef USE_GPU_OFFLOAD
    std::cout << "  max|<u,conv>|: " << m.max_conv_ke_prod << "\n";
    std::cout << "  avg|<u,conv>|: " << m.avg_conv_ke_prod << "\n";
#endif
    std::cout << "\n";

    // Key invariants for skew-symmetric in inviscid limit:
    // 1. KE should not explode (bounded)
    // 2. KE should not grow significantly (< 1% growth allowed for roundoff)
    // 3. <u, conv(u)> should be near machine epsilon

    record("Energy bounded (no explosion)", !m.exploded);
    record("Energy stable (KE_max/KE_init < 1.01)", m.ke_max / m.ke_initial < 1.01,
           qoi(m.ke_max / m.ke_initial, 1.01));

#ifndef USE_GPU_OFFLOAD
    // <u, conv(u)> should be very small relative to KE
    // Normalize by KE*dt to get dimensionless rate
    double normalized_rate = m.max_conv_ke_prod / (m.ke_initial * dt);
    record("Convection energy-neutral (|<u,conv>|/KE/dt < 0.01)", normalized_rate < 0.01,
           qoi(normalized_rate, 0.01));
#endif
}

// ============================================================================
// Test: Compare skew vs central advection energy properties
// ============================================================================
void test_skew_vs_central_energy() {
    std::cout << "\n--- Skew vs Central Advection Energy Comparison ---\n\n";

    // Configuration: moderate viscosity to avoid numerical instability
    const int N = 32;
    const int nsteps = 200;
    const double nu = 1e-4;  // Small but non-zero
    const double dt = 0.005;
    const double L = 2.0 * M_PI;

    Mesh mesh;
    mesh.init_uniform(N, N, 0.0, L, 0.0, L);

    std::cout << "  Grid: " << N << "x" << N << ", steps: " << nsteps << "\n";
    std::cout << "  nu: " << std::scientific << nu << "\n\n";

    // Test both schemes
    struct SchemeResult {
        std::string name;
        ConvectiveScheme scheme;
        EnergyMetrics metrics;
    };

    std::vector<SchemeResult> results = {
        {"Skew-symmetric", ConvectiveScheme::Skew, {}},
        {"Central", ConvectiveScheme::Central, {}}
    };

    for (auto& r : results) {
        Config config;
        config.nu = nu;
        config.dt = dt;
        config.adaptive_dt = false;
        config.turb_model = TurbulenceModelType::None;
        config.convective_scheme = r.scheme;
        config.verbose = false;

        r.metrics = run_energy_tracking(mesh, config, nsteps);

        double ke_ratio = r.metrics.ke_final / r.metrics.ke_initial;
        std::cout << "  " << r.name << ":\n";
        std::cout << "    KE ratio (final/init): " << std::fixed << std::setprecision(6)
                  << ke_ratio << "\n";
#ifndef USE_GPU_OFFLOAD
        std::cout << "    max|<u,conv>|: " << std::scientific << r.metrics.max_conv_ke_prod << "\n";
        std::cout << "    avg|<u,conv>|: " << r.metrics.avg_conv_ke_prod << "\n";
#endif
        std::cout << "\n";
    }

    // Both schemes should be stable
    record("Skew-symmetric stable", !results[0].metrics.exploded);
    record("Central stable", !results[1].metrics.exploded);

#ifndef USE_GPU_OFFLOAD
    // Skew should have smaller |<u, conv(u)>| than central
    // This is the key distinguishing property
    bool skew_better = results[0].metrics.max_conv_ke_prod <= results[1].metrics.max_conv_ke_prod;
    record("Skew has smaller |<u,conv>| than Central", skew_better,
           "(skew=" + std::to_string(results[0].metrics.max_conv_ke_prod) +
           ", central=" + std::to_string(results[1].metrics.max_conv_ke_prod) + ")");
#endif
}

// ============================================================================
// Test: Long-time energy stability (drift detection)
// ============================================================================
void test_long_time_energy_stability() {
    std::cout << "\n--- Long-Time Energy Stability ---\n\n";

    // Run for many steps to detect slow energy drift
    const int N = 24;  // Smaller grid for speed
    const int nsteps = 1000;
    const double nu = 1e-3;  // Moderate viscosity
    const double dt = 0.005;
    const double L = 2.0 * M_PI;

    Mesh mesh;
    mesh.init_uniform(N, N, 0.0, L, 0.0, L);

    Config config;
    config.nu = nu;
    config.dt = dt;
    config.adaptive_dt = false;
    config.turb_model = TurbulenceModelType::None;
    config.convective_scheme = ConvectiveScheme::Skew;
    config.verbose = false;

    std::cout << "  Grid: " << N << "x" << N << ", steps: " << nsteps << "\n";
    std::cout << "  T_final: " << nsteps * dt << "\n";
    std::cout << "  nu: " << nu << "\n\n";

    EnergyMetrics m = run_energy_tracking(mesh, config, nsteps);

    // For viscous TGV, expected decay: E(t) ~ E(0) * exp(-4*nu*t)
    // At T = 5.0, nu = 0.001: E/E0 ~ exp(-0.02) ~ 0.98
    // Allow for numerical dissipation to make it decay faster
    double T = nsteps * dt;
    double expected_ratio_upper = 1.0;  // Should not grow
    double expected_ratio_lower = std::exp(-10.0 * nu * T);  // Some extra dissipation OK

    double ke_ratio = m.ke_final / m.ke_initial;

    std::cout << "  KE ratio (final/init): " << std::fixed << std::setprecision(6) << ke_ratio << "\n";
    std::cout << "  Expected range: [" << expected_ratio_lower << ", " << expected_ratio_upper << "]\n";
    std::cout << "  KE max/init: " << m.ke_max / m.ke_initial << "\n";
    std::cout << "\n";

    // Key check: KE should never exceed initial (no energy production)
    record("Long-time energy bounded", m.ke_max / m.ke_initial < 1.001,
           qoi(m.ke_max / m.ke_initial, 1.001));

    // Energy should decay (viscosity present)
    record("Energy decaying (final < initial)", m.ke_final < m.ke_initial * 1.001);

    // Final energy in expected range
    bool in_range = (ke_ratio <= expected_ratio_upper) && (ke_ratio >= expected_ratio_lower);
    record("Energy in expected range", in_range,
           "(ratio=" + std::to_string(ke_ratio) + ")");
}

// ============================================================================
// Test: 3D skew-symmetric energy conservation
// ============================================================================
void test_skew_energy_3d() {
    std::cout << "\n--- 3D Skew-Symmetric Energy Conservation ---\n\n";

    const int N = 16;  // Small for speed
    const int nsteps = 100;
    const double nu = 1e-6;  // Near-inviscid
    const double dt = 0.005;
    const double L = 2.0 * M_PI;

    Mesh mesh;
    mesh.init_uniform(N, N, N, 0.0, L, 0.0, L, 0.0, L);

    Config config;
    config.Nx = N;
    config.Ny = N;
    config.Nz = N;
    config.nu = nu;
    config.dt = dt;
    config.adaptive_dt = false;
    config.turb_model = TurbulenceModelType::None;
    config.convective_scheme = ConvectiveScheme::Skew;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    // Fully periodic BCs
    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::Periodic;
    bc.y_hi = VelocityBC::Periodic;
    bc.z_lo = VelocityBC::Periodic;
    bc.z_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    // Initialize with 3D Taylor-Green vortex
    init_taylor_green_3d(solver, mesh);
    solver.sync_to_gpu();

    std::cout << "  Grid: " << N << "x" << N << "x" << N << ", steps: " << nsteps << "\n";
    std::cout << "  nu: " << std::scientific << nu << " (near-inviscid)\n\n";

#ifdef USE_GPU_OFFLOAD
    double ke_init = solver.compute_kinetic_energy_device();
#else
    double ke_init = compute_kinetic_energy_3d(solver.velocity(), mesh);
#endif
    double ke_max = ke_init;
    double ke = ke_init;

    for (int step = 1; step <= nsteps; ++step) {
        solver.step();

#ifdef USE_GPU_OFFLOAD
        ke = solver.compute_kinetic_energy_device();
#else
        solver.sync_from_gpu();
        ke = compute_kinetic_energy_3d(solver.velocity(), mesh);
#endif
        ke_max = std::max(ke_max, ke);

        if (!std::isfinite(ke) || ke > 100.0 * ke_init) {
            std::cout << "  [FAIL] Explosion at step " << step << "\n";
            record("3D energy bounded", false);
            return;
        }
    }

    double ke_ratio = ke / ke_init;
    double ke_growth = (ke_max - ke_init) / ke_init;

    std::cout << "  KE initial: " << std::scientific << ke_init << "\n";
    std::cout << "  KE final:   " << ke << "\n";
    std::cout << "  KE max growth: " << ke_growth * 100 << "%\n\n";

    // 3D skew-symmetric should also conserve energy
    record("3D energy bounded (no explosion)", std::isfinite(ke));
    record("3D energy stable (KE_max/KE_init < 1.01)", ke_max / ke_init < 1.01,
           qoi(ke_max / ke_init, 1.01));
}

// ============================================================================
// Main
// ============================================================================
int main() {
    return harness::run("Skew-Symmetric Energy Conservation", []() {
        test_skew_energy_bounded();
        test_skew_vs_central_energy();
        test_long_time_energy_stability();
        test_skew_energy_3d();
    });
}
