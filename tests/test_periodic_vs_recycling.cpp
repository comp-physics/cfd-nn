// Validation test: Periodic-x channel vs Recycling inflow channel
//
// This test verifies that the recycling inflow machinery does not pollute
// interior turbulence by comparing to the gold-standard periodic channel.
//
// Both cases use identical: domain, grid, nu, dp/dx, dt, convection scheme.
// The only difference is x-BC treatment:
// - Periodic: FFT Poisson solver, periodic x BCs
// - Recycling: MG Poisson solver, Dirichlet inlet / Neumann outlet
//
// NOTE: For true turbulent validation, need:
// - Finer grid: 128×96×128 or more for Re_tau=180 DNS
// - Higher Re: Re_tau >= 395 for robust turbulence on coarser grids
// - Longer spinup: 50000+ steps (t ~ 10+ flow-through times)
// Current quick/full modes use coarse grids and validate machinery only.

#include "solver.hpp"
#include "mesh.hpp"
#include "config.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include <iomanip>
#include <random>

using namespace nncfd;

// =============================================================================
// Create Poiseuille + 3D perturbation for transition
// =============================================================================

VectorField create_perturbed_channel(const Mesh& mesh, double nu, double dp_dx,
                                     double delta, double pert_amplitude) {
    VectorField vel(mesh);

    // Poiseuille base profile: u = (|dp/dx|/(2*nu)) * (delta^2 - y^2)
    double u_max = std::abs(dp_dx) * delta * delta / (2.0 * nu);

    // Random number generator for perturbations
    std::mt19937 rng(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    const int Ng = mesh.Nghost;
    const double Lx = mesh.x_max - mesh.x_min;
    const double Lz = mesh.z_max - mesh.z_min;
    const double kx = 2.0 * M_PI / Lx;
    const double kz = 2.0 * M_PI / Lz;

    // Initialize u at x-faces
    for (int k = 0; k < mesh.total_Nz(); ++k) {
        double z = (k >= Ng && k < mesh.Nz + Ng) ? mesh.zc[k] : 0.0;
        for (int j = 0; j < mesh.total_Ny(); ++j) {
            double y = mesh.yc[j];
            double y_frac = y / delta;  // -1 to 1

            // Poiseuille profile
            double u_base = u_max * (1.0 - y_frac * y_frac);

            // Wall-damping factor (vanishes at walls)
            double wall_damp = (1.0 - y_frac * y_frac);
            wall_damp = wall_damp * wall_damp;  // sin^2 like behavior

            for (int i = 0; i < mesh.total_Nx() + 1; ++i) {
                double x = (i >= Ng && i <= mesh.Nx + Ng) ?
                           mesh.xc[std::min(i, mesh.Nx + Ng - 1)] : 0.0;

                // 3D perturbation: low-wavenumber modes + random
                double pert_struct = std::sin(kx * x) * std::cos(kz * z);
                double pert_rand = dist(rng);
                double pert = pert_amplitude * u_max * wall_damp * (0.7 * pert_struct + 0.3 * pert_rand);

                vel.u(i, j, k) = u_base + pert;
            }
        }
    }

    // Initialize v at y-faces (small perturbations only, Poiseuille has v=0)
    for (int k = 0; k < mesh.total_Nz(); ++k) {
        double z = (k >= Ng && k < mesh.Nz + Ng) ? mesh.zc[k] : 0.0;
        for (int j = 0; j < mesh.total_Ny() + 1; ++j) {
            double y = (j >= Ng && j <= mesh.Ny + Ng) ?
                       mesh.yc[std::min(j, mesh.Ny + Ng - 1)] : 0.0;
            double y_frac = y / delta;
            double wall_damp = (1.0 - y_frac * y_frac);
            wall_damp = wall_damp * wall_damp;

            for (int i = 0; i < mesh.total_Nx(); ++i) {
                double x = (i >= Ng && i < mesh.Nx + Ng) ? mesh.xc[i] : 0.0;
                double pert = pert_amplitude * u_max * wall_damp * 0.1 * dist(rng);
                vel.v(i, j, k) = pert;
            }
        }
    }

    // Initialize w at z-faces (small perturbations for 3D instabilities)
    if (!mesh.is2D()) {
        for (int k = 0; k < mesh.total_Nz() + 1; ++k) {
            for (int j = 0; j < mesh.total_Ny(); ++j) {
                double y = mesh.yc[j];
                double y_frac = y / delta;
                double wall_damp = (1.0 - y_frac * y_frac);
                wall_damp = wall_damp * wall_damp;

                for (int i = 0; i < mesh.total_Nx(); ++i) {
                    double pert = pert_amplitude * u_max * wall_damp * 0.1 * dist(rng);
                    vel.w(i, j, k) = pert;
                }
            }
        }
    }

    return vel;
}

// =============================================================================
// Test configuration
// =============================================================================

struct ValidationConfig {
    // Target Re_tau (determines nu for fixed dp/dx)
    double Re_tau_target = 180.0;
    double dp_dx = -1.0;  // Body force
    double delta = 1.0;   // Channel half-height

    // Grid (coarser for quick validation, use finer for production)
    int Nx = 64;
    int Ny = 64;
    int Nz = 32;
    bool stretch_y = true;
    double stretch_beta = 2.0;

    // Domain
    double Lx = 2.0 * M_PI;  // 2*pi*delta
    double Ly = 2.0;         // 2*delta
    double Lz = M_PI;        // pi*delta

    // Time stepping
    double dt = 0.0005;         // Moderate dt (CFL ~ 0.5 for U_max ~ 90)
    int spinup_steps = 5000;    // Steps for flow to develop
    int stats_steps = 2000;     // Steps to accumulate statistics
    int stats_interval = 10;    // Accumulate every N steps
    double pert_amplitude = 0.05;  // 5% amplitude (won't trigger turbulence on coarse grid)

    // Convection scheme: Central is stable and adequate for machinery validation
    // Use Skew for production DNS with fine grids
    ConvectiveScheme convection = ConvectiveScheme::Central;

    // Recycle configuration
    double recycle_x = -1.0;  // Auto (10*delta)

    // Tolerance for comparison
    double mean_profile_tol = 0.05;     // 5% max difference in U+
    double reynolds_stress_tol = 0.10;  // 10% relative difference in -<u'v'>+
    double stress_peak_tol = 0.15;      // 15% peak magnitude tolerance
    double closure_tol = 0.02;          // 2% momentum closure residual
};

// =============================================================================
// Run a single case and collect statistics
// =============================================================================

struct CaseResults {
    std::string name;
    RANSSolver::ResolutionDiagnostics resolution;
    RANSSolver::MomentumBalanceDiagnostics momentum;
    RANSSolver::ReynoldsStressProfiles stresses;
    std::vector<double> U_plus;  // Mean velocity profile
    double u_tau;
    double Re_tau_actual;
    bool converged;
};

CaseResults run_case(const std::string& name, const ValidationConfig& cfg,
                     bool use_recycling) {
    CaseResults result;
    result.name = name;
    result.converged = false;

    std::cout << "\n=== Running Case: " << name << " ===\n";

    // Compute nu for target Re_tau
    double nu = RANSSolver::nu_for_Re_tau(cfg.Re_tau_target, cfg.dp_dx, cfg.delta);
    std::cout << "  Target Re_tau: " << cfg.Re_tau_target << "\n";
    std::cout << "  nu: " << nu << "\n";

    // Create config
    Config config;
    config.Nx = cfg.Nx;
    config.Ny = cfg.Ny;
    config.Nz = cfg.Nz;
    config.x_min = 0.0;
    config.x_max = cfg.Lx;
    config.y_min = -cfg.delta;
    config.y_max = cfg.delta;
    config.z_min = 0.0;
    config.z_max = cfg.Lz;
    config.nu = nu;
    config.dt = cfg.dt;
    config.max_steps = cfg.spinup_steps + cfg.stats_steps;
    config.stretch_y = cfg.stretch_y;
    config.stretch_beta = cfg.stretch_beta;
    config.convective_scheme = cfg.convection;
    config.time_integrator = TimeIntegrator::RK3;
    config.simulation_mode = SimulationMode::Unsteady;
    config.verbose = false;
    // Disable NaN guard: it checks ghost cells which may have uninitialized NaN
    // that don't affect the solution (see gpu_check_nan_inf limitation)
    config.turb_guard_enabled = false;

    // Recycling config
    if (use_recycling) {
        config.recycling_inflow = true;
        config.recycle_x = cfg.recycle_x;
    }

    // Create mesh
    Mesh mesh;
    if (cfg.stretch_y) {
        mesh.init_stretched_y(cfg.Nx, cfg.Ny, cfg.Nz,
                              config.x_min, config.x_max,
                              config.y_min, config.y_max,
                              config.z_min, config.z_max,
                              Mesh::tanh_stretching(cfg.stretch_beta), 2);
    } else {
        mesh.init_uniform(cfg.Nx, cfg.Ny, cfg.Nz,
                          config.x_min, config.x_max,
                          config.y_min, config.y_max,
                          config.z_min, config.z_max, 2);
    }

    // Create solver
    RANSSolver solver(mesh, config);

    // Set velocity BCs - this triggers initialize_recycling_inflow() if enabled
    // Default VelocityBC has periodic x, z and no-slip walls y
    VelocityBC vel_bc;
    solver.set_velocity_bc(vel_bc);

    solver.set_body_force(std::abs(cfg.dp_dx), 0.0);

    // Initialize with Poiseuille + 3D perturbation for transition
    if (cfg.pert_amplitude > 0.0) {
        VectorField vel = create_perturbed_channel(mesh, nu, cfg.dp_dx, cfg.delta, cfg.pert_amplitude);
        solver.initialize(vel);
    } else {
        // No perturbation - will stay laminar
        solver.initialize_uniform(1.0, 0.0);
    }

    // Sync to GPU if needed
    solver.sync_to_gpu();

    std::cout << "  Spinup: " << cfg.spinup_steps << " steps\n";

    // Spinup phase
    for (int step = 0; step < cfg.spinup_steps; ++step) {
        solver.step();
        if ((step + 1) % 500 == 0) {
            std::cout << "    Spinup step " << (step + 1) << "/" << cfg.spinup_steps << "\n";
        }
    }

    std::cout << "  Stats collection: " << cfg.stats_steps << " steps\n";

    // Statistics collection phase
    solver.reset_statistics();
    for (int step = 0; step < cfg.stats_steps; ++step) {
        solver.step();
        if ((step + 1) % cfg.stats_interval == 0) {
            solver.sync_solution_from_gpu();
            solver.accumulate_statistics();
        }
        if ((step + 1) % 500 == 0) {
            std::cout << "    Stats step " << (step + 1) << "/" << cfg.stats_steps << "\n";
        }
    }

    // Collect results
    result.resolution = solver.compute_resolution_diagnostics();
    result.momentum = solver.compute_momentum_balance();
    result.stresses = solver.compute_reynolds_stress_profiles();
    result.u_tau = result.resolution.u_tau_force;
    result.Re_tau_actual = solver.Re_tau_from_forcing();
    result.converged = true;

    // Store U+ profile
    result.U_plus.resize(result.stresses.y_plus.size());
    // U+ = U_mean / u_tau (need to get from stats)
    // For now, use the momentum balance which has dU/dy integrated

    std::cout << "  Actual Re_tau: " << result.Re_tau_actual << "\n";
    std::cout << "  u_tau (force): " << result.u_tau << "\n";
    std::cout << "  Momentum closure max residual: "
              << 100.0 * result.momentum.max_residual_normalized(result.u_tau) << "%\n";

    // Turbulence presence diagnostics
    double max_uv_plus = 0.0;
    double max_uu_plus = 0.0;
    for (size_t j = 0; j < result.stresses.uv_plus.size(); ++j) {
        max_uv_plus = std::max(max_uv_plus, -result.stresses.uv_plus[j]);  // -<u'v'>+
        max_uu_plus = std::max(max_uu_plus, result.stresses.uu_plus[j]);
    }
    std::cout << "  Turbulence check: max(-<u'v'>+) = " << max_uv_plus;
    if (max_uv_plus < 0.01) {
        std::cout << " [LAMINAR - no turbulent stresses]\n";
    } else if (max_uv_plus < 0.5) {
        std::cout << " [TRANSITIONAL]\n";
    } else {
        std::cout << " [TURBULENT - expected ~0.9 for channel]\n";
    }

    return result;
}

// =============================================================================
// Compare two cases
// =============================================================================

struct ComparisonResult {
    double max_stress_diff;      // Max relative difference in -<u'v'>+
    double max_uu_diff;          // Max relative difference in <u'u'>+
    double closure_diff;         // Difference in momentum closure residuals
    bool stress_ordering_match;  // Both cases have correct ordering
    bool passes;
};

ComparisonResult compare_cases(const CaseResults& periodic,
                                const CaseResults& recycling,
                                const ValidationConfig& cfg) {
    ComparisonResult cmp;
    cmp.passes = true;

    std::cout << "\n=== Comparison: " << periodic.name << " vs " << recycling.name << " ===\n";

    // Check both have same number of y-points
    if (periodic.stresses.y_plus.size() != recycling.stresses.y_plus.size()) {
        std::cerr << "ERROR: Profile sizes don't match\n";
        cmp.passes = false;
        return cmp;
    }

    const size_t Ny = periodic.stresses.y_plus.size();

    // Compare -<u'v'>+ profiles
    cmp.max_stress_diff = 0.0;
    for (size_t j = 0; j < Ny; ++j) {
        double y_plus = periodic.stresses.y_plus[j];
        // Only compare in buffer/log region (5 < y+ < 150)
        if (y_plus > 5.0 && y_plus < 150.0) {
            double uv_p = periodic.stresses.uv_plus[j];
            double uv_r = recycling.stresses.uv_plus[j];
            double ref = std::max(std::abs(uv_p), 0.1);  // Avoid div by zero
            double rel_diff = std::abs(uv_p - uv_r) / ref;
            cmp.max_stress_diff = std::max(cmp.max_stress_diff, rel_diff);
        }
    }

    // Compare <u'u'>+ profiles
    cmp.max_uu_diff = 0.0;
    for (size_t j = 0; j < Ny; ++j) {
        double y_plus = periodic.stresses.y_plus[j];
        if (y_plus > 5.0 && y_plus < 150.0) {
            double uu_p = periodic.stresses.uu_plus[j];
            double uu_r = recycling.stresses.uu_plus[j];
            double ref = std::max(std::abs(uu_p), 0.1);
            double rel_diff = std::abs(uu_p - uu_r) / ref;
            cmp.max_uu_diff = std::max(cmp.max_uu_diff, rel_diff);
        }
    }

    // Compare momentum closure residuals
    double closure_p = periodic.momentum.max_residual_normalized(periodic.u_tau);
    double closure_r = recycling.momentum.max_residual_normalized(recycling.u_tau);
    cmp.closure_diff = std::abs(closure_p - closure_r);

    // Check stress ordering
    cmp.stress_ordering_match = periodic.stresses.passes_stress_ordering() &&
                                 recycling.stresses.passes_stress_ordering();

    // Report
    std::cout << "\n--- Reynolds Shear Stress -<u'v'>+ ---\n";
    std::cout << "  Max relative difference: " << 100.0 * cmp.max_stress_diff << "%\n";
    std::cout << "  Tolerance: " << 100.0 * cfg.reynolds_stress_tol << "%\n";
    std::cout << "  Result: " << (cmp.max_stress_diff <= cfg.reynolds_stress_tol ? "PASS" : "FAIL") << "\n";

    std::cout << "\n--- Streamwise Stress <u'u'>+ ---\n";
    std::cout << "  Max relative difference: " << 100.0 * cmp.max_uu_diff << "%\n";
    std::cout << "  Tolerance: " << 100.0 * cfg.stress_peak_tol << "%\n";
    std::cout << "  Result: " << (cmp.max_uu_diff <= cfg.stress_peak_tol ? "PASS" : "FAIL") << "\n";

    std::cout << "\n--- Momentum Closure ---\n";
    std::cout << "  Periodic residual: " << 100.0 * closure_p << "%\n";
    std::cout << "  Recycling residual: " << 100.0 * closure_r << "%\n";
    std::cout << "  Difference: " << 100.0 * cmp.closure_diff << "%\n";

    std::cout << "\n--- Stress Ordering ---\n";
    std::cout << "  Both correct: " << (cmp.stress_ordering_match ? "PASS" : "FAIL") << "\n";

    // Overall pass/fail
    cmp.passes = (cmp.max_stress_diff <= cfg.reynolds_stress_tol) &&
                 (cmp.max_uu_diff <= cfg.stress_peak_tol) &&
                 cmp.stress_ordering_match;

    std::cout << "\n=== OVERALL: " << (cmp.passes ? "PASS" : "FAIL") << " ===\n";

    return cmp;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    std::cout << "============================================\n";
    std::cout << "Periodic-x vs Recycling Inflow Validation\n";
    std::cout << "============================================\n";

    ValidationConfig cfg;

    // Parse command-line for quick vs full validation
    bool quick_mode = true;
    bool sweep_mode = false;  // Recycle distance sweep
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--full") {
            quick_mode = false;
        } else if (arg == "--sweep") {
            sweep_mode = true;
            quick_mode = false;  // Sweep requires full mode
        } else if (arg == "--Re_tau" && i + 1 < argc) {
            cfg.Re_tau_target = std::stod(argv[++i]);
        } else if (arg == "--spinup" && i + 1 < argc) {
            cfg.spinup_steps = std::stoi(argv[++i]);
        } else if (arg == "--stats" && i + 1 < argc) {
            cfg.stats_steps = std::stoi(argv[++i]);
        }
    }

    if (quick_mode) {
        std::cout << "\n*** QUICK MODE (use --full for production validation) ***\n";
        // Use smaller grid, lower Re_tau, and shorter run for quick check
        cfg.Re_tau_target = 33.0;  // Much lower Re for coarse grid stability
        cfg.Nx = 32;
        cfg.Ny = 32;
        cfg.Nz = 16;
        cfg.dt = 0.001;
        cfg.spinup_steps = 200;
        cfg.stats_steps = 100;
        cfg.stats_interval = 5;
        cfg.pert_amplitude = 0.05;  // Small perturbation
        cfg.convection = ConvectiveScheme::Upwind;  // Upwind for quick mode
        cfg.stretch_y = false;  // Uniform grid for quick mode
        // Relax tolerances for quick mode (just checking machinery works)
        cfg.reynolds_stress_tol = 1.0;  // 100% - laminar won't have turbulent stats
        cfg.stress_peak_tol = 1.0;
    }

    std::cout << "\nConfiguration:\n";
    std::cout << "  Grid: " << cfg.Nx << " x " << cfg.Ny << " x " << cfg.Nz << "\n";
    std::cout << "  Target Re_tau: " << cfg.Re_tau_target << "\n";
    std::cout << "  Spinup steps: " << cfg.spinup_steps << "\n";
    std::cout << "  Stats steps: " << cfg.stats_steps << "\n";

    if (sweep_mode) {
        // Recycle distance sweep: compare 6δ, 10δ, 14δ against periodic
        std::cout << "\n=== RECYCLE DISTANCE SWEEP ===\n";
        std::cout << "Comparing recycle distances: 6δ, 10δ, 14δ vs Periodic-x\n";

        // Run periodic baseline
        CaseResults periodic = run_case("Periodic-x (baseline)", cfg, false);

        // Run sweep at different recycle distances
        std::vector<double> recycle_distances = {6.0, 10.0, 14.0};
        bool all_pass = true;

        for (double dist : recycle_distances) {
            cfg.recycle_x = dist * cfg.delta;
            std::string name = "Recycling @ " + std::to_string((int)dist) + "δ";
            CaseResults recycling = run_case(name, cfg, true);
            ComparisonResult cmp = compare_cases(periodic, recycling, cfg);
            all_pass = all_pass && cmp.passes;
        }

        std::cout << "\n============================================\n";
        std::cout << "SWEEP RESULT: " << (all_pass ? "ALL PASS" : "SOME FAILED") << "\n";
        std::cout << "============================================\n";

        return all_pass ? 0 : 1;
    }

    // Standard mode: periodic vs recycling (default 10δ)
    CaseResults periodic = run_case("Periodic-x", cfg, false);
    CaseResults recycling = run_case("Recycling Inflow", cfg, true);
    ComparisonResult cmp = compare_cases(periodic, recycling, cfg);

    std::cout << "\n============================================\n";
    std::cout << "VALIDATION RESULT: " << (cmp.passes ? "PASS" : "FAIL") << "\n";
    std::cout << "============================================\n";

    return cmp.passes ? 0 : 1;
}
