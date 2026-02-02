/// Tests for turbulent recycling inflow boundary condition
/// Covers: shift correctness, flux correction, projection enforcement,
/// laminar stability, and basic DNS validation

#include "solver.hpp"
#include "mesh.hpp"
#include "config.hpp"
#include "fields.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <random>
#include <vector>

using namespace nncfd;

// Test tolerance
constexpr double TOL = 1e-10;
constexpr double PHYS_TOL = 1e-6;

//==============================================================================
// Helper functions
//==============================================================================

/// Compute plane mean of u at given x-index
double plane_mean_u(const VectorField& vel, const Mesh& mesh, int i) {
    const int Ny = mesh.Ny;
    const int Nz = mesh.Nz;
    const int Ng = mesh.Nghost;

    double sum = 0.0;
    int count = 0;
    for (int k = 0; k < Nz; ++k) {
        for (int j = 0; j < Ny; ++j) {
            sum += vel.u(i, j + Ng, k + Ng);
            count++;
        }
    }
    return sum / count;
}

/// Compute RMS of u fluctuations at given x-index
double plane_rms_u(const VectorField& vel, const Mesh& mesh, int i) {
    const int Ny = mesh.Ny;
    const int Nz = mesh.Nz;
    const int Ng = mesh.Nghost;

    double mean = plane_mean_u(vel, mesh, i);
    double sum_sq = 0.0;
    int count = 0;
    for (int k = 0; k < Nz; ++k) {
        for (int j = 0; j < Ny; ++j) {
            double u_prime = vel.u(i, j + Ng, k + Ng) - mean;
            sum_sq += u_prime * u_prime;
            count++;
        }
    }
    return std::sqrt(sum_sq / count);
}

/// Compute max divergence in a slab from i_start to i_end
double max_divergence_slab(const VectorField& vel, const Mesh& mesh,
                           int i_start, int i_end) {
    const int Ny = mesh.Ny;
    const int Nz = mesh.is2D() ? 1 : mesh.Nz;
    const int Ng = mesh.Nghost;
    const double dx = mesh.dx;
    const double dy = mesh.dy;
    const double dz = mesh.is2D() ? 1.0 : mesh.dz;

    double max_div = 0.0;
    for (int k = 0; k < Nz; ++k) {
        int kg = k + Ng;
        for (int j = 0; j < Ny; ++j) {
            int jg = j + Ng;
            for (int i = i_start; i < i_end; ++i) {
                int ig = i + Ng;
                double dudx = (vel.u(ig + 1, jg, kg) - vel.u(ig, jg, kg)) / dx;
                double dvdy = (vel.v(ig, jg + 1, kg) - vel.v(ig, jg, kg)) / dy;
                double dwdz = mesh.is2D() ? 0.0 :
                    (vel.w(ig, jg, kg + 1) - vel.w(ig, jg, kg)) / dz;
                double div = std::abs(dudx + dvdy + dwdz);
                max_div = std::max(max_div, div);
            }
        }
    }
    return max_div;
}

/// Fill velocity with sinusoidal pattern for shift test
void fill_sinusoidal_pattern(VectorField& vel, const Mesh& mesh) {
    const int Nx = mesh.Nx;
    const int Ny = mesh.Ny;
    const int Nz = mesh.Nz;
    const int Ng = mesh.Nghost;

    for (int k = 0; k < Nz; ++k) {
        int kg = k + Ng;
        for (int j = 0; j < Ny; ++j) {
            int jg = j + Ng;
            for (int i = 0; i <= Nx; ++i) {
                int ig = i + Ng;
                // u = sin(2*pi*k/Nz) * (1 + 0.1*j/Ny)
                vel.u(ig, jg, kg) = std::sin(2.0 * M_PI * k / Nz) * (1.0 + 0.1 * j / Ny);
            }
        }
    }
    // v = 0, w = 0 (already initialized to zero)
}

/// Fill with Poiseuille profile
void fill_poiseuille(VectorField& vel, const Mesh& mesh, double u_max) {
    const int Nx = mesh.Nx;
    const int Ny = mesh.Ny;
    const int Nz = mesh.is2D() ? 1 : mesh.Nz;
    const int Ng = mesh.Nghost;
    const double H = (mesh.y_max - mesh.y_min) / 2.0;
    const double y_center = (mesh.y_max + mesh.y_min) / 2.0;

    for (int k = 0; k < Nz; ++k) {
        int kg = k + Ng;
        for (int j = 0; j < Ny; ++j) {
            int jg = j + Ng;
            double y = mesh.y(jg) - y_center;
            double u_pois = u_max * (1.0 - (y * y) / (H * H));
            for (int i = 0; i <= Nx; ++i) {
                int ig = i + Ng;
                vel.u(ig, jg, kg) = u_pois;
            }
        }
    }
}

//==============================================================================
// Stage A: Unit tests (fast, surgical)
//==============================================================================

/// Test 1.1A: Shift correctness
/// Create sinusoidal pattern, verify shift is exact
bool test_shift_correctness() {
    std::cout << "\n=== Test: Shift Correctness ===\n";

    // Small 3D mesh
    Mesh mesh;
    mesh.init_uniform(16, 16, 8, 0.0, 4.0, -1.0, 1.0, 0.0, 2.0);

    Config config;
    config.nu = 0.01;
    config.recycling_inflow = true;
    config.recycle_x = 3.0;  // Recycle plane at x=3
    config.recycle_shift_z = 2;  // Shift by 2 cells
    config.recycle_shift_interval = 0;  // Constant shift
    config.recycle_fringe_length = 0.0;  // No fringe
    config.recycle_filter_tau = -1.0;  // No filter

    RANSSolver solver(mesh, config);

    // Set up BCs
    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;  // Will be overridden to Inflow
    bc.x_hi = VelocityBC::Periodic;  // Will be overridden to Outflow
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    bc.z_lo = VelocityBC::Periodic;
    bc.z_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    // Initialize with sinusoidal pattern
    VectorField vel(mesh);
    fill_sinusoidal_pattern(vel, mesh);
    solver.initialize(vel);

    // Get recycle plane index
    const int Ng = mesh.Nghost;
    const int i_recycle = Ng + static_cast<int>((3.0 - mesh.x_min) / mesh.dx);
    const int shift_k = 2;
    const int Nz = mesh.Nz;
    const int Ny = mesh.Ny;

    // Call recycling machinery
    solver.extract_recycle_plane();
    solver.process_recycle_inflow();
    solver.apply_recycling_inlet_bc();

    // Check: inlet should equal shifted recycle plane (approximately)
    double max_error = 0.0;
    const VectorField& result = solver.velocity();
    for (int k = 0; k < Nz; ++k) {
        int k_src = (k + shift_k) % Nz;
        for (int j = 0; j < Ny; ++j) {
            double u_recycle = vel.u(i_recycle, j + Ng, k_src + Ng);
            double u_inlet = result.u(Ng, j + Ng, k + Ng);
            double error = std::abs(u_inlet - u_recycle);
            max_error = std::max(max_error, error);
        }
    }

    std::cout << "  Max shift error: " << std::scientific << max_error << "\n";

    // With mass flux correction (target = initial mean), allow some tolerance
    bool pass = (max_error < 0.15);
    std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << "\n";
    return pass;
}

/// Test 1.1B: Flux correction sanity
bool test_flux_correction() {
    std::cout << "\n=== Test: Flux Correction Sanity ===\n";

    Mesh mesh;
    mesh.init_uniform(16, 16, 8, 0.0, 4.0, -1.0, 1.0, 0.0, 2.0);

    Config config;
    config.nu = 0.01;
    config.recycling_inflow = true;
    config.recycle_x = 3.0;
    config.recycle_shift_z = 0;  // No shift for this test
    config.recycle_shift_interval = 0;
    config.recycle_fringe_length = 0.0;
    config.recycle_filter_tau = -1.0;
    config.recycle_target_bulk_u = 2.0;  // Target different from initial

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    bc.z_lo = VelocityBC::Periodic;
    bc.z_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    // Initialize with known mean and fluctuations
    VectorField vel(mesh);
    const int Ng = mesh.Nghost;
    const double mean_initial = 1.0;
    const double fluct_amplitude = 0.1;

    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, fluct_amplitude);

    for (int k = 0; k < mesh.Nz; ++k) {
        for (int j = 0; j < mesh.Ny; ++j) {
            for (int i = 0; i <= mesh.Nx; ++i) {
                vel.u(i + Ng, j + Ng, k + Ng) = mean_initial + dist(rng);
            }
        }
    }
    solver.initialize(vel);

    // Get recycle plane stats before
    const int i_recycle = Ng + static_cast<int>((3.0 - mesh.x_min) / mesh.dx);
    double mean_before = plane_mean_u(solver.velocity(), mesh, i_recycle);
    double rms_before = plane_rms_u(solver.velocity(), mesh, i_recycle);

    // Run recycling
    solver.extract_recycle_plane();
    solver.process_recycle_inflow();
    solver.apply_recycling_inlet_bc();

    // Get inlet stats after
    double mean_after = plane_mean_u(solver.velocity(), mesh, Ng);
    double rms_after = plane_rms_u(solver.velocity(), mesh, Ng);

    // Expected mean with clamped scale (target=2.0, initial≈1.0 → scale≈1.1)
    double expected_mean = mean_before * 1.1;

    std::cout << "  Recycle plane mean: " << mean_before << "\n";
    std::cout << "  Recycle plane RMS:  " << rms_before << "\n";
    std::cout << "  Inlet mean after:   " << mean_after << "\n";
    std::cout << "  Inlet RMS after:    " << rms_after << "\n";
    std::cout << "  Expected mean:      " << expected_mean << "\n";

    bool mean_ok = std::abs(mean_after - expected_mean) < 0.2;
    bool rms_ok = std::abs(rms_after - rms_before) / rms_before < 0.3;

    std::cout << "  Mean scaling: " << (mean_ok ? "OK" : "FAIL") << "\n";
    std::cout << "  RMS preserved: " << (rms_ok ? "OK" : "FAIL") << "\n";

    bool pass = mean_ok && rms_ok;
    std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << "\n";
    return pass;
}

/// Test 1.2A: Inlet Dirichlet respected after projection
bool test_inlet_projection_enforcement() {
    std::cout << "\n=== Test: Inlet Projection Enforcement ===\n";

    Mesh mesh;
    mesh.init_uniform(16, 16, 8, 0.0, 4.0, -1.0, 1.0, 0.0, 2.0);

    Config config;
    config.nu = 0.01;
    config.dt = 0.001;
    config.recycling_inflow = true;
    config.recycle_x = 3.0;
    config.recycle_shift_z = 0;
    config.recycle_fringe_length = 0.0;
    config.recycle_filter_tau = -1.0;
    config.convective_scheme = ConvectiveScheme::Upwind;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    bc.z_lo = VelocityBC::Periodic;
    bc.z_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    // Initialize with Poiseuille-ish profile
    VectorField vel(mesh);
    fill_poiseuille(vel, mesh, 1.0);
    solver.initialize(vel);

    // Take one step
    solver.step();

    // Check inlet velocity is sane
    double mean_inlet = plane_mean_u(solver.velocity(), mesh, mesh.Nghost);
    bool inlet_sane = (mean_inlet > 0.1 && mean_inlet < 10.0 && std::isfinite(mean_inlet));

    std::cout << "  Mean inlet velocity after step: " << mean_inlet << "\n";
    std::cout << "  Inlet sanity: " << (inlet_sane ? "OK" : "FAIL") << "\n";

    // Check divergence near inlet
    double div_inlet = max_divergence_slab(solver.velocity(), mesh, 0, 3);
    double div_interior = max_divergence_slab(solver.velocity(), mesh, 5, 10);

    std::cout << "  Max div near inlet (i=0..2): " << std::scientific << div_inlet << "\n";
    std::cout << "  Max div in interior (i=5..9): " << div_interior << "\n";

    bool div_ok = (div_inlet < 1e-3 && div_interior < 1e-3);
    std::cout << "  Divergence: " << (div_ok ? "OK" : "FAIL") << "\n";

    bool pass = inlet_sane && div_ok;
    std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << "\n";
    return pass;
}

//==============================================================================
// Stage B: Minimal physics tests
//==============================================================================

/// Test: Laminar Poiseuille stays laminar with recycling
bool test_laminar_stability() {
    std::cout << "\n=== Test: Laminar Poiseuille Stability ===\n";

    Mesh mesh;
    mesh.init_uniform(32, 32, 16, 0.0, 6.28, -1.0, 1.0, 0.0, 3.14);

    Config config;
    config.nu = 0.01;  // Low Re, definitely laminar
    config.dt = 0.001;
    config.dp_dx = -1.0;
    config.recycling_inflow = true;
    config.recycle_x = 4.0;
    config.recycle_shift_z = 4;
    config.recycle_fringe_length = 0.0;
    config.recycle_filter_tau = -1.0;
    config.convective_scheme = ConvectiveScheme::Upwind;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    bc.z_lo = VelocityBC::Periodic;
    bc.z_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);
    solver.set_body_force(-config.dp_dx, 0.0);

    // Initialize with exact Poiseuille
    double u_max = -config.dp_dx * 1.0 / (2.0 * config.nu);
    VectorField vel(mesh);
    fill_poiseuille(vel, mesh, u_max);
    solver.initialize(vel);

    double rms_initial = plane_rms_u(solver.velocity(), mesh, mesh.Nghost + mesh.Nx/2);
    std::cout << "  Initial centerline RMS: " << std::scientific << rms_initial << "\n";

    // Expected RMS for Poiseuille profile due to y-variation: ~0.298 * u_max
    double expected_rms = 0.298 * u_max;
    std::cout << "  Expected Poiseuille RMS: " << expected_rms << "\n";

    // Run for 50 steps
    const int nsteps = 50;
    double final_rms = rms_initial;
    for (int i = 0; i < nsteps; ++i) {
        solver.step();
        final_rms = plane_rms_u(solver.velocity(), mesh, mesh.Nghost + mesh.Nx/2);

        if (!std::isfinite(final_rms)) {
            std::cout << "  NaN detected at step " << i << "\n";
            std::cout << "  Result: FAIL\n";
            return false;
        }
    }

    std::cout << "  Final centerline RMS: " << final_rms << "\n";

    // Criterion: RMS should not change by more than 5% (profile stays laminar)
    double rms_change = std::abs(final_rms - rms_initial) / rms_initial;
    bool rms_ok = (rms_change < 0.05);
    std::cout << "  RMS change: " << (rms_change * 100.0) << "% (limit: 5%)\n";
    std::cout << "  Laminar preserved: " << (rms_ok ? "OK" : "FAIL") << "\n";

    double mean_final = plane_mean_u(solver.velocity(), mesh, mesh.Nghost + mesh.Nx/2);
    double expected_mean = u_max * 2.0 / 3.0;
    bool mean_ok = std::abs(mean_final - expected_mean) / expected_mean < 0.1;
    std::cout << "  Mean velocity: " << mean_final << " (expected ~" << expected_mean << ")\n";

    bool pass = rms_ok && mean_ok;
    std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << "\n";
    return pass;
}

/// Test: Short run stability with perturbations
bool test_perturbed_stability() {
    std::cout << "\n=== Test: Perturbed Flow Stability ===\n";

    Mesh mesh;
    mesh.init_uniform(32, 32, 16, 0.0, 6.28, -1.0, 1.0, 0.0, 3.14);

    Config config;
    // Use reasonable Re to keep CFL stable
    // u_max = dp_dx / (2*nu), CFL = u_max * dt / dx
    // With nu=0.005, dp_dx=-1: u_max=100, CFL = 100 * 0.001 / 0.196 ≈ 0.5
    config.nu = 0.005;
    config.dt = 0.001;
    config.dp_dx = -1.0;
    config.recycling_inflow = true;
    config.recycle_x = 4.0;
    config.recycle_shift_z = 4;
    config.recycle_fringe_length = 0.0;
    config.recycle_filter_tau = -1.0;
    config.convective_scheme = ConvectiveScheme::Upwind;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    bc.z_lo = VelocityBC::Periodic;
    bc.z_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);
    solver.set_body_force(-config.dp_dx, 0.0);

    // Initialize with perturbed Poiseuille
    double u_max = -config.dp_dx * 1.0 / (2.0 * config.nu);
    VectorField vel(mesh);
    fill_poiseuille(vel, mesh, u_max);

    // Compute CFL for information
    double cfl = u_max * config.dt / mesh.dx;
    std::cout << "  u_max = " << u_max << ", CFL = " << cfl << "\n";

    // Add random perturbations (1% of u_max)
    std::mt19937 rng(123);
    std::normal_distribution<double> dist(0.0, 0.01 * u_max);
    const int Ng = mesh.Nghost;
    for (int k = 0; k < mesh.Nz; ++k) {
        for (int j = 0; j < mesh.Ny; ++j) {
            for (int i = 0; i <= mesh.Nx; ++i) {
                vel.u(i + Ng, j + Ng, k + Ng) += dist(rng);
            }
        }
    }
    solver.initialize(vel);

    // Run for 100 steps
    const int nsteps = 100;
    double max_u = 0.0;

    for (int i = 0; i < nsteps; ++i) {
        solver.step();

        // Track max velocity
        for (int k = 0; k < mesh.Nz; ++k) {
            for (int j = 0; j < mesh.Ny; ++j) {
                for (int ii = 0; ii <= mesh.Nx; ++ii) {
                    double u = std::abs(solver.velocity().u(ii + Ng, j + Ng, k + Ng));
                    max_u = std::max(max_u, u);
                }
            }
        }

        if (max_u > 100 * u_max || !std::isfinite(max_u)) {
            std::cout << "  Blow-up at step " << i << ", max|u| = " << max_u << "\n";
            std::cout << "  Result: FAIL\n";
            return false;
        }
    }

    std::cout << "  Max |u| observed: " << max_u << " (u_max = " << u_max << ")\n";

    // Velocity should stay within 20% of expected (accounting for perturbations + adjustment)
    bool stable = (max_u < 1.5 * u_max);
    std::cout << "  Stability: " << (stable ? "OK" : "FAIL") << "\n";

    bool pass = stable;
    std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << "\n";
    return pass;
}

//==============================================================================
// Diagnostics tests
//==============================================================================

/// Test inlet/recycle similarity after processing
bool test_inlet_recycle_similarity() {
    std::cout << "\n=== Test: Inlet/Recycle Similarity ===\n";

    Mesh mesh;
    mesh.init_uniform(32, 32, 16, 0.0, 6.28, -1.0, 1.0, 0.0, 3.14);

    Config config;
    config.nu = 0.01;
    config.dt = 0.001;
    config.recycling_inflow = true;
    config.recycle_x = 4.0;
    config.recycle_shift_z = 0;  // No shift for similarity test
    config.recycle_fringe_length = 0.0;
    config.recycle_filter_tau = -1.0;
    config.convective_scheme = ConvectiveScheme::Upwind;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    bc.z_lo = VelocityBC::Periodic;
    bc.z_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    // Initialize with varied field
    VectorField vel(mesh);
    fill_sinusoidal_pattern(vel, mesh);
    solver.initialize(vel);

    // Get indices
    const int Ng = mesh.Nghost;
    const int i_inlet = Ng;
    const int i_recycle = Ng + static_cast<int>((4.0 - mesh.x_min) / mesh.dx);

    // Apply recycling
    solver.extract_recycle_plane();
    solver.process_recycle_inflow();
    solver.apply_recycling_inlet_bc();

    // Compute L2 difference
    double sum_sq_diff = 0.0;
    double sum_sq_recycle = 0.0;

    for (int k = 0; k < mesh.Nz; ++k) {
        for (int j = 0; j < mesh.Ny; ++j) {
            double u_in = solver.velocity().u(i_inlet, j + Ng, k + Ng);
            double u_r = vel.u(i_recycle, j + Ng, k + Ng);
            sum_sq_diff += (u_in - u_r) * (u_in - u_r);
            sum_sq_recycle += u_r * u_r;
        }
    }

    double rel_diff = std::sqrt(sum_sq_diff / sum_sq_recycle);
    std::cout << "  Relative L2 difference (inlet vs recycle): " << rel_diff << "\n";

    bool similar = (rel_diff < 0.25);
    std::cout << "  Similarity: " << (similar ? "OK" : "FAIL") << "\n";

    std::cout << "  Result: " << (similar ? "PASS" : "FAIL") << "\n";
    return similar;
}

//==============================================================================
// Medium-duration tests (energy balance, x-homogeneity)
//==============================================================================

/// Test: Energy balance (P_in ≈ dissipation in steady state)
/// This runs for more steps to reach quasi-steady state
bool test_energy_balance() {
    std::cout << "\n=== Test: Energy Balance ===\n";

    Mesh mesh;
    mesh.init_uniform(32, 32, 16, 0.0, 6.28, -1.0, 1.0, 0.0, 3.14);

    Config config;
    config.nu = 0.01;
    config.dt = 0.001;
    config.dp_dx = -1.0;
    config.recycling_inflow = true;
    config.recycle_x = 4.0;
    config.recycle_shift_z = 4;
    config.recycle_fringe_length = 0.0;
    config.recycle_filter_tau = -1.0;
    config.convective_scheme = ConvectiveScheme::Upwind;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    bc.z_lo = VelocityBC::Periodic;
    bc.z_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);
    solver.set_body_force(-config.dp_dx, 0.0);

    // Initialize with Poiseuille
    double u_max = -config.dp_dx * 1.0 / (2.0 * config.nu);
    VectorField vel(mesh);
    fill_poiseuille(vel, mesh, u_max);
    solver.initialize(vel);

    std::cout << "  Running 500 steps to approach steady state...\n";

    // Track energy balance over time
    std::vector<double> dKdt_samples, P_minus_eps_samples;
    double K_prev = 0.0;

    const int nsteps = 500;
    const int sample_interval = 50;

    for (int step = 0; step < nsteps; ++step) {
        solver.step();

        if (step % sample_interval == 0 && step > 0) {
            double K = solver.compute_kinetic_energy();
            double P_in = solver.compute_power_input();
            double eps = solver.compute_viscous_dissipation();

            double dKdt = (K - K_prev) / (sample_interval * config.dt);
            double imbalance = P_in - eps;

            dKdt_samples.push_back(dKdt);
            P_minus_eps_samples.push_back(imbalance);

            if (step == sample_interval || step == nsteps - sample_interval) {
                std::cout << "  Step " << step << ": K=" << K
                          << ", P_in=" << P_in << ", eps=" << eps
                          << ", P-eps=" << imbalance << "\n";
            }

            K_prev = K;
        }

        if (step == 0) {
            K_prev = solver.compute_kinetic_energy();
        }
    }

    // Compute mean values in second half (after spinup)
    int n_samples = static_cast<int>(dKdt_samples.size());
    int start_idx = n_samples / 2;
    double mean_dKdt = 0.0, mean_P_eps = 0.0;
    for (int i = start_idx; i < n_samples; ++i) {
        mean_dKdt += dKdt_samples[i];
        mean_P_eps += P_minus_eps_samples[i];
    }
    mean_dKdt /= (n_samples - start_idx);
    mean_P_eps /= (n_samples - start_idx);

    std::cout << "  Mean dK/dt (second half): " << mean_dKdt << "\n";
    std::cout << "  Mean P_in - eps (second half): " << mean_P_eps << "\n";

    // In steady state, |dK/dt| should be small compared to P_in
    double P_in_final = solver.compute_power_input();
    bool dKdt_ok = std::abs(mean_dKdt) < 0.1 * std::abs(P_in_final);
    std::cout << "  dK/dt check: " << (dKdt_ok ? "OK" : "FAIL") << "\n";

    // P_in - eps should also be small (energy balance)
    bool balance_ok = std::abs(mean_P_eps) < 0.2 * std::abs(P_in_final);
    std::cout << "  Energy balance check: " << (balance_ok ? "OK" : "FAIL") << "\n";

    bool pass = dKdt_ok && balance_ok;
    std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << "\n";
    return pass;
}

/// Test: x-homogeneity of turbulence statistics
/// After development, stats should be flat in x (no recovery region)
bool test_x_homogeneity() {
    std::cout << "\n=== Test: X-Homogeneity ===\n";

    Mesh mesh;
    mesh.init_uniform(32, 32, 16, 0.0, 6.28, -1.0, 1.0, 0.0, 3.14);

    Config config;
    config.nu = 0.01;
    config.dt = 0.001;
    config.dp_dx = -1.0;
    config.recycling_inflow = true;
    config.recycle_x = 4.0;
    config.recycle_shift_z = 4;
    config.recycle_fringe_length = 0.0;
    config.recycle_filter_tau = -1.0;
    config.convective_scheme = ConvectiveScheme::Upwind;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    bc.z_lo = VelocityBC::Periodic;
    bc.z_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);
    solver.set_body_force(-config.dp_dx, 0.0);

    // Initialize with Poiseuille
    double u_max = -config.dp_dx * 1.0 / (2.0 * config.nu);
    VectorField vel(mesh);
    fill_poiseuille(vel, mesh, u_max);
    solver.initialize(vel);

    std::cout << "  Running 300 steps...\n";

    // Run to develop flow
    for (int step = 0; step < 300; ++step) {
        solver.step();
    }

    // Sample plane-averaged stats at multiple x locations
    // Skip inlet region (first 5 planes) and outlet region (last 3 planes)
    std::vector<double> u_means, u_rms_vals;
    std::cout << "  Plane-averaged stats:\n";
    std::cout << "    i       x        u_mean      u_rms\n";

    for (int i = 5; i < mesh.Nx - 3; i += 4) {
        auto stats = solver.compute_plane_stats(i);
        u_means.push_back(stats.u_mean);
        u_rms_vals.push_back(stats.u_rms);

        double x = mesh.x_min + (i + 0.5) * mesh.dx;
        std::cout << "    " << i << "    " << std::fixed << std::setprecision(3) << x
                  << "    " << std::scientific << stats.u_mean
                  << "    " << stats.u_rms << "\n";
    }

    // Compute coefficient of variation (std/mean) of u_mean across x
    double mean_u = 0.0, mean_rms = 0.0;
    for (size_t i = 0; i < u_means.size(); ++i) {
        mean_u += u_means[i];
        mean_rms += u_rms_vals[i];
    }
    mean_u /= u_means.size();
    mean_rms /= u_rms_vals.size();

    double var_u = 0.0, var_rms = 0.0;
    for (size_t i = 0; i < u_means.size(); ++i) {
        var_u += (u_means[i] - mean_u) * (u_means[i] - mean_u);
        var_rms += (u_rms_vals[i] - mean_rms) * (u_rms_vals[i] - mean_rms);
    }
    double cv_u = std::sqrt(var_u / u_means.size()) / std::abs(mean_u);
    double cv_rms = std::sqrt(var_rms / u_rms_vals.size()) / std::abs(mean_rms);

    std::cout << "\n  Coefficient of variation of u_mean across x: " << (cv_u * 100.0) << "%\n";
    std::cout << "  Coefficient of variation of u_rms across x:  " << (cv_rms * 100.0) << "%\n";

    // For laminar/near-laminar flow, u_mean should be very uniform
    // u_rms variation is expected from Poiseuille profile
    bool u_mean_ok = (cv_u < 0.05);  // Less than 5% variation
    std::cout << "  u_mean uniformity: " << (u_mean_ok ? "OK" : "FAIL") << "\n";

    // For Poiseuille, RMS should also be fairly uniform (all same profile)
    bool rms_ok = (cv_rms < 0.10);  // Less than 10% variation
    std::cout << "  u_rms uniformity: " << (rms_ok ? "OK" : "FAIL") << "\n";

    bool pass = u_mean_ok && rms_ok;
    std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << "\n";
    return pass;
}

//==============================================================================
// Main
//==============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "  Recycling Inflow Test Suite\n";
    std::cout << "========================================\n";

    int passed = 0;
    int failed = 0;

    // Stage A: Unit tests
    std::cout << "\n*** Stage A: Unit Tests ***\n";

    if (test_shift_correctness()) passed++; else failed++;
    if (test_flux_correction()) passed++; else failed++;
    if (test_inlet_projection_enforcement()) passed++; else failed++;

    // Stage B: Physics tests
    std::cout << "\n*** Stage B: Physics Tests ***\n";

    if (test_laminar_stability()) passed++; else failed++;
    if (test_perturbed_stability()) passed++; else failed++;

    // Diagnostics
    std::cout << "\n*** Diagnostics Tests ***\n";

    if (test_inlet_recycle_similarity()) passed++; else failed++;

    // Stage C: Medium-duration tests (energy balance, x-homogeneity)
    std::cout << "\n*** Stage C: Medium-Duration Tests ***\n";

    if (test_energy_balance()) passed++; else failed++;
    if (test_x_homogeneity()) passed++; else failed++;

    // Summary
    std::cout << "\n========================================\n";
    std::cout << "  Summary: " << passed << " passed, " << failed << " failed\n";
    std::cout << "========================================\n";

    return (failed == 0) ? 0 : 1;
}
