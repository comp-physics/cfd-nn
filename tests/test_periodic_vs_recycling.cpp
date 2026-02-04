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
// Create Poiseuille + low-k near-wall perturbation for transition
// =============================================================================
// Uses low wavenumber modes (k=1,2,3) concentrated near walls where
// turbulence is naturally generated. This is much more effective than
// random noise for triggering transition on coarse grids.

VectorField create_perturbed_channel(const Mesh& mesh, double nu, double dp_dx,
                                     double delta, double pert_amplitude) {
    VectorField vel(mesh);

    // Poiseuille base profile: u = (|dp/dx|/(2*nu)) * (delta^2 - y^2)
    double u_max = std::abs(dp_dx) * delta * delta / (2.0 * nu);

    // Random phases for different modes (fixed seed for reproducibility)
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> phase_dist(0.0, 2.0 * M_PI);
    double phi1 = phase_dist(rng), phi2 = phase_dist(rng), phi3 = phase_dist(rng);
    double psi1 = phase_dist(rng), psi2 = phase_dist(rng), psi3 = phase_dist(rng);

    const int Ng = mesh.Nghost;
    const double Lx = mesh.x_max - mesh.x_min;
    const double Lz = mesh.z_max - mesh.z_min;

    // Low wavenumbers only (k=1,2,3 in each direction)
    const double kx1 = 2.0 * M_PI / Lx;
    const double kx2 = 4.0 * M_PI / Lx;
    const double kz1 = 2.0 * M_PI / Lz;
    const double kz2 = 4.0 * M_PI / Lz;
    const double kz3 = 6.0 * M_PI / Lz;

    // Initialize u at x-faces
    for (int k = 0; k < mesh.total_Nz(); ++k) {
        double z = (k >= Ng && k < mesh.Nz + Ng) ? mesh.zc[k] : 0.0;
        for (int j = 0; j < mesh.total_Ny(); ++j) {
            double y = mesh.yc[j];
            double y_frac = y / delta;  // -1 to 1

            // Poiseuille profile
            double u_base = u_max * (1.0 - y_frac * y_frac);

            // Near-wall weighting: peaks near y/delta ~ 0.3 (where streaks form)
            // Shape: exp(-((|y|-0.7*delta)/(0.2*delta))^2) concentrated near walls
            double y_wall_bot = std::abs(y_frac + 1.0);  // Distance from bottom wall
            double y_wall_top = std::abs(y_frac - 1.0);  // Distance from top wall
            double near_wall = std::exp(-std::pow((y_wall_bot - 0.3) / 0.2, 2))
                             + std::exp(-std::pow((y_wall_top - 0.3) / 0.2, 2));
            // Also need wall damping to satisfy no-slip
            double wall_damp = (1.0 - y_frac * y_frac);

            for (int i = 0; i < mesh.total_Nx() + 1; ++i) {
                double x = (i >= Ng && i <= mesh.Nx + Ng) ?
                           mesh.xc[std::min(i, mesh.Nx + Ng - 1)] : 0.0;

                // Low-k structured perturbation (streak-like in x, varying in z)
                double pert_u = 0.6 * std::sin(kx1 * x + phi1) * std::cos(kz1 * z + psi1)
                              + 0.3 * std::sin(kx1 * x + phi2) * std::cos(kz2 * z + psi2)
                              + 0.1 * std::sin(kx2 * x + phi3) * std::cos(kz3 * z + psi3);
                double pert = pert_amplitude * u_max * wall_damp * near_wall * pert_u;

                vel.u(i, j, k) = u_base + pert;
            }
        }
    }

    // Initialize v at y-faces - use streamfunction-derived form for lower divergence
    // v ~ -d(psi)/dx where psi produces the u perturbation
    for (int k = 0; k < mesh.total_Nz(); ++k) {
        double z = (k >= Ng && k < mesh.Nz + Ng) ? mesh.zc[k] : 0.0;
        for (int j = 0; j < mesh.total_Ny() + 1; ++j) {
            double y = (j >= Ng && j <= mesh.Ny + Ng) ?
                       mesh.yc[std::min(j, mesh.Ny + Ng - 1)] : 0.0;
            double y_frac = y / delta;
            double wall_damp = (1.0 - y_frac * y_frac);
            wall_damp = wall_damp * wall_damp;  // Stronger damping for v

            for (int i = 0; i < mesh.total_Nx(); ++i) {
                double x = (i >= Ng && i < mesh.Nx + Ng) ? mesh.xc[i] : 0.0;
                // v perturbation: cosine in x (derivative of sine)
                double pert_v = 0.6 * kx1 * std::cos(kx1 * x + phi1) * std::cos(kz1 * z + psi1)
                              + 0.3 * kx1 * std::cos(kx1 * x + phi2) * std::cos(kz2 * z + psi2);
                double pert = -0.1 * pert_amplitude * u_max * wall_damp * pert_v / kx1;
                vel.v(i, j, k) = pert;
            }
        }
    }

    // Initialize w at z-faces - streamfunction-derived (sine in z for w)
    if (!mesh.is2D()) {
        for (int k = 0; k < mesh.total_Nz() + 1; ++k) {
            double z = (k >= Ng && k <= mesh.Nz + Ng) ?
                       mesh.zc[std::min(k, mesh.Nz + Ng - 1)] : 0.0;
            for (int j = 0; j < mesh.total_Ny(); ++j) {
                double y = mesh.yc[j];
                double y_frac = y / delta;
                double wall_damp = (1.0 - y_frac * y_frac);
                wall_damp = wall_damp * wall_damp;

                for (int i = 0; i < mesh.total_Nx(); ++i) {
                    double x = (i >= Ng && i < mesh.Nx + Ng) ? mesh.xc[i] : 0.0;
                    // w perturbation: sine in z (derivative of cosine)
                    double pert_w = 0.6 * kz1 * std::sin(kx1 * x + phi1) * std::sin(kz1 * z + psi1)
                                  + 0.3 * kz2 * std::sin(kx1 * x + phi2) * std::sin(kz2 * z + psi2);
                    double pert = -0.1 * pert_amplitude * u_max * wall_damp * pert_w / kz1;
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
    // Re-enable y-stretching for wall resolution
    bool stretch_y = true;
    double stretch_beta = 2.0;

    // Domain
    double Lx = 2.0 * M_PI;  // 2*pi*delta
    double Ly = 2.0;         // 2*delta
    double Lz = M_PI;        // pi*delta

    // Time stepping
    // Note: Stretched grids need much smaller dt due to small cells near walls
    double dt = 0.00002;        // Small dt for stretched grid stability
    int spinup_steps = 5000;    // Steps for flow to develop
    int stats_steps = 2000;     // Steps to accumulate statistics
    int stats_interval = 10;    // Accumulate every N steps
    double pert_amplitude = 0.05;  // 5% amplitude (won't trigger turbulence on coarse grid)

    // Convection scheme: Central for full mode (overridden to Upwind for stability)
    // Quick mode uses Upwind explicitly
    ConvectiveScheme convection = ConvectiveScheme::Central;

    // Recycle configuration
    double recycle_x = -1.0;  // Auto (10*delta)

    // Tolerance for comparison
    double mean_profile_tol = 0.05;     // 5% max difference in U+
    double reynolds_stress_tol = 0.10;  // 10% relative difference in -<u'v'>+
    double stress_peak_tol = 0.15;      // 15% peak magnitude tolerance
    double closure_tol = 0.02;          // 2% momentum closure residual

    // Validation mode: Quick (machinery) or Full (DNS realism)
    RANSSolver::ValidationMode validation_mode = RANSSolver::ValidationMode::Quick;

    // Force ramping during startup (stabilizes transition on coarse grids)
    bool enable_force_ramp = true;
    double force_ramp_tau = 2.0;  // Time constant in bulk time units

    // Initial divergence projection (cleans up perturbed velocity)
    // Temporarily disabled for debugging wall shear issue
    bool project_initial = false;
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
    // Disable CUDA Graph for stretched grids (graphed V-cycle uses uniform-y operators)
    if (cfg.stretch_y) {
        config.poisson_use_vcycle_graph = false;
        std::cout << "  [DEBUG] Disabled CUDA Graph for stretched grid\n";
    }
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
        // Use convergence-based MG for recycling (Dirichlet/Neumann BCs need more iterations)
        config.poisson_fixed_cycles = 0;
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

    // DEBUG: Print y-metrics to verify they're computed correctly
    if (mesh.is_y_stretched()) {
        std::cout << "  [DEBUG Y-Metrics] y_stretched=YES\n";
        std::cout << "    dyv[Ng]=" << mesh.dyv[mesh.Nghost] << " (first interior cell height)\n";
        std::cout << "    dyv[Ng+Ny/2]=" << mesh.dyv[mesh.Nghost + cfg.Ny/2] << " (mid-channel cell height)\n";
        std::cout << "    dyc[Ng]=" << mesh.dyc[mesh.Nghost] << " (first interior center-spacing)\n";
        std::cout << "    dyc[Ng+Ny/2]=" << mesh.dyc[mesh.Nghost + cfg.Ny/2] << " (mid-channel center-spacing)\n";
        std::cout << "    yLap_aS[Ng]=" << mesh.yLap_aS[mesh.Nghost] << " yLap_aN[Ng]=" << mesh.yLap_aN[mesh.Nghost] << "\n";
        std::cout << "    yLap_aP[Ng]=" << mesh.yLap_aP[mesh.Nghost] << " (should be -(aS+aN)=" << -(mesh.yLap_aS[mesh.Nghost]+mesh.yLap_aN[mesh.Nghost]) << ")\n";
    } else {
        std::cout << "  [DEBUG Y-Metrics] y_stretched=NO (uniform)\n";
    }

    // Create solver
    RANSSolver solver(mesh, config);

    // Set velocity BCs - this triggers initialize_recycling_inflow() if enabled
    // Default VelocityBC has periodic x, z and no-slip walls y
    VelocityBC vel_bc;
    solver.set_velocity_bc(vel_bc);

    // Set body force immediately (we want the flow to start with proper mean)
    solver.set_body_force(std::abs(cfg.dp_dx), 0.0);

    // Enable force ramping if requested (helps startup stability)
    if (cfg.enable_force_ramp) {
        solver.enable_force_ramp(cfg.force_ramp_tau);
        std::cout << "  Force ramping enabled (tau=" << cfg.force_ramp_tau << " bulk time units)\n";
    }

    // Initialize with Poiseuille + 3D perturbation for transition
    if (cfg.pert_amplitude > 0.0) {
        VectorField vel = create_perturbed_channel(mesh, nu, cfg.dp_dx, cfg.delta, cfg.pert_amplitude);
        solver.initialize(vel);

        // Project initial velocity to remove divergence from perturbation
        if (cfg.project_initial) {
            std::cout << "  Projecting initial velocity to remove divergence...\n";
            solver.project_initial_velocity();
        }
    } else {
        // No perturbation - will stay laminar
        solver.initialize_uniform(1.0, 0.0);
    }

    // DEBUG: Check near-wall velocity BEFORE GPU sync
    {
        const int Ng = mesh.Nghost;
        int j1 = Ng;
        int j2 = Ng + 1;
        double u1_pre = solver.velocity().u(Ng, j1, Ng);
        double u2_pre = solver.velocity().u(Ng, j2, Ng);
        std::cout << "  [DEBUG] Before GPU sync: u1=" << u1_pre << " u2=" << u2_pre
                  << " (j1=" << j1 << ", j2=" << j2 << ")\n";
    }

    // Sync to GPU if needed
    solver.sync_to_gpu();

    // Prime recycling buffers from initial velocity (avoids zero inlet on first step)
    if (use_recycling && solver.is_recycling_enabled()) {
        solver.extract_recycle_plane();
        solver.process_recycle_inflow();
    }

    // Check initial velocity magnitude
    solver.sync_solution_from_gpu();

    // DEBUG: Check near-wall velocity AFTER GPU sync round-trip
    {
        const int Ng = mesh.Nghost;
        int j1 = Ng;
        int j2 = Ng + 1;
        double u1_post = solver.velocity().u(Ng, j1, Ng);
        double u2_post = solver.velocity().u(Ng, j2, Ng);
        std::cout << "  [DEBUG] After GPU sync: u1=" << u1_post << " u2=" << u2_post << "\n";
    }
    double u_max_init = 0.0;
    double u_sum_init = 0.0;
    int u_count_init = 0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double u_val = solver.velocity().u(i, j, k);
                u_max_init = std::max(u_max_init, std::abs(u_val));
                u_sum_init += u_val;
                ++u_count_init;
            }
        }
    }
    double u_mean_init = u_sum_init / u_count_init;

    std::cout << "  Initial u_max: " << std::fixed << std::setprecision(2)
              << u_max_init << " (expected ~" << (std::abs(cfg.dp_dx) * cfg.delta * cfg.delta / (2.0 * nu))
              << "), u_mean: " << u_mean_init << "\n";

    // Laminar wall shear for reference: tau_w_lam = delta * |dp/dx| (from force balance)
    // u_tau_lam = sqrt(tau_w_lam) = sqrt(delta * |dp/dx|) = 1.0 for delta=1, dp/dx=-1
    const double u_tau_laminar = std::sqrt(cfg.delta * std::abs(cfg.dp_dx));

    std::cout << "  Spinup: " << cfg.spinup_steps << " steps\n";

    // Spinup phase with wall shear and mid-channel RMS monitoring
    // Print interval scales with spinup length
    const int print_interval = std::max(100, cfg.spinup_steps / 10);
    // Wall shear sample interval for history tracking
    const int shear_sample_interval = std::max(10, cfg.spinup_steps / 100);

    // Clear wall shear history and turbulence samples before spinup
    solver.clear_wall_shear_history();
    solver.clear_turbulence_samples();

    for (int step = 0; step < cfg.spinup_steps; ++step) {
        solver.step();

        // DEBUG: Check first step specifically
        if (step == 0) {
            solver.sync_solution_from_gpu();
            const int Ng = mesh.Nghost;
            int j1 = Ng;
            int j2 = Ng + 1;
            double u1 = solver.velocity().u(Ng, j1, Ng);
            double u2 = solver.velocity().u(Ng, j2, Ng);
            std::cout << "  [DEBUG] After step 1: u1=" << u1 << " u2=" << u2;
            if (std::isnan(u1) || std::isnan(u2)) std::cout << " *** NAN ***";
            std::cout << "\n";
        }

        // Record wall shear and turbulence samples for settling detection
        if ((step + 1) % shear_sample_interval == 0) {
            solver.sync_solution_from_gpu();
            solver.record_wall_shear_sample(solver.current_time());
            solver.record_turbulence_sample();  // Update windowed classifier
        }

        if ((step + 1) % print_interval == 0 || step == cfg.spinup_steps - 1) {
            solver.sync_solution_from_gpu();

            // Use new turbulence presence indicators for robust detection
            auto indicators = solver.compute_turbulence_presence();
            const auto& classifier = solver.turbulence_classifier();

            std::cout << "    Step " << std::setw(5) << (step + 1) << "/" << cfg.spinup_steps
                      << "  u_tau/u_tau_ref=" << std::fixed << std::setprecision(3) << indicators.u_tau_ratio
                      << "  TKE_mid=" << std::scientific << std::setprecision(2) << indicators.tke_mid
                      << " [instant:" << indicators.state_string()
                      << ", confirmed:" << classifier.state_string() << "]";

            // Show force ramp status if enabled
            if (solver.is_force_ramp_active()) {
                double ramp_progress = 1.0 - std::exp(-solver.current_time() / cfg.force_ramp_tau);
                std::cout << " (ramp: " << std::fixed << std::setprecision(0) << ramp_progress * 100 << "%)";
            }
            std::cout << std::fixed << "\n";
        }
    }

    // Check if wall shear has settled before collecting statistics
    bool shear_settled = solver.is_wall_shear_settled(10, 0.01);
    std::cout << "  Wall shear settled: " << (shear_settled ? "YES" : "NO (stats may be drifting)") << "\n";
    std::cout << "  Confirmed turbulence state: " << solver.turbulence_classifier().state_string() << "\n";

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

    // Sync GPU data before collecting results
    solver.sync_solution_from_gpu();

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

    // Turbulence presence diagnostics using new robust indicators
    auto presence = solver.compute_turbulence_presence();
    std::cout << "  Turbulence presence indicators:\n";
    std::cout << "    State:           " << presence.state_string() << "\n";
    std::cout << "    u_tau/u_tau_lam: " << std::fixed << std::setprecision(3) << presence.u_tau_ratio << "\n";
    std::cout << "    u_rms_mid:       " << std::scientific << std::setprecision(2) << presence.u_rms_mid << "\n";
    std::cout << "    max(-<u'v'>+):   " << std::fixed << std::setprecision(3) << presence.max_uv_plus;
    if (presence.max_uv_plus < 0.01) {
        std::cout << " [no turbulent stresses]\n";
    } else if (presence.max_uv_plus < 0.5) {
        std::cout << " [weak/transitional]\n";
    } else {
        std::cout << " [strong - expected ~0.9 for channel]\n";
    }
    std::cout << std::fixed;

    // Print full validation report using the appropriate mode
    std::cout << "\n  === Validation Report ===\n";
    auto report = solver.validate_turbulence_realism(cfg.validation_mode);
    report.print();

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
        std::cout << "    This validates machinery works, NOT DNS-quality turbulence.\n";
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
        // Use Quick validation mode (skips resolution gates)
        cfg.validation_mode = RANSSolver::ValidationMode::Quick;
        // Enable force ramping even in quick mode (cheap stabilizer, improves repeatability)
        cfg.enable_force_ramp = true;
        cfg.force_ramp_tau = 1.0;  // Shorter ramp for quick mode
        cfg.project_initial = false;  // Skip projection for speed
    } else {
        // Full mode: DNS-quality validation
        cfg.validation_mode = RANSSolver::ValidationMode::Full;
        // Use Upwind convection for stability on 64x64x32 grid
        cfg.convection = ConvectiveScheme::Upwind;
    }

    std::cout << "\nConfiguration:\n";
    std::cout << "  Grid: " << cfg.Nx << " x " << cfg.Ny << " x " << cfg.Nz << "\n";
    std::cout << "  Target Re_tau: " << cfg.Re_tau_target << "\n";
    std::cout << "  Spinup steps: " << cfg.spinup_steps << "\n";
    std::cout << "  Stats steps: " << cfg.stats_steps << "\n";
    std::cout << "  Validation mode: " << (cfg.validation_mode == RANSSolver::ValidationMode::Quick ? "QUICK" : "FULL") << "\n";
    std::cout << "  Force ramping: " << (cfg.enable_force_ramp ? "ON" : "OFF") << "\n";
    std::cout << "  Project initial: " << (cfg.project_initial ? "ON" : "OFF") << "\n";

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
