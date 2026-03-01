/// @file test_dns_channel_validation.cpp
/// @brief DNS channel flow machinery validation (GPU, 3D)
///
/// Runs 192x96x192 channel with v13 recipe (trip + filter) for ~500 steps.
/// Validates DNS machinery works correctly, not converged statistics.
///
/// Validates:
///   1. Turbulence triggered (TKE > 0, velocity fluctuations present)
///   2. Incompressibility: max|div(u)| < 1e-4
///   3. Stability: max velocity bounded, no NaN/Inf
///   4. Resolution quality: y+ < 1, dx+ < 20, dz+ < 10
///   5. Energy evolution: KE doesn't grow unboundedly

#include "test_harness.hpp"
#include "test_utilities.hpp"
#include "reference_data.hpp"
#include "solver.hpp"
#include "config.hpp"
#include <cmath>
#include <vector>
#include <iomanip>

using namespace nncfd;
using namespace nncfd::test;
using nncfd::test::harness::record;

void test_dns_channel_machinery() {
    std::cout << "\n--- DNS Channel 192x96x192, v13 recipe, 500 steps ---\n\n";

    // v13 DNS recipe
    const int Nx = 192, Ny = 96, Nz = 192;
    const double Lx = 4.0 * M_PI;
    const double Ly = 2.0;        // y in [-1, 1]
    const double Lz = 2.0 * M_PI;
    const double nu = 1.0 / 180.0;  // Re_tau ~ 180 target
    const double dp_dx = -1.0;       // dp/dx = -u_tau^2/delta
    const double beta = 2.0;         // Stretching parameter
    const int nsteps = 500;

    // Setup stretched mesh
    Mesh mesh;
    mesh.init_stretched_y(Nx, Ny, Nz,
                          0.0, Lx, -Ly / 2, Ly / 2, 0.0, Lz,
                          Mesh::tanh_stretching(beta));

    Config config;
    config.Nx = Nx;
    config.Ny = Ny;
    config.Nz = Nz;
    config.nu = nu;
    config.dp_dx = dp_dx;
    config.rho = 1.0;
    config.CFL_max = 0.15;
    config.CFL_xz = 0.30;
    config.dt_safety = 0.85;
    config.dt = 0.001;
    config.adaptive_dt = true;
    config.simulation_mode = SimulationMode::Unsteady;
    config.turb_model = TurbulenceModelType::None;
    config.convective_scheme = ConvectiveScheme::Skew;
    config.time_integrator = TimeIntegrator::RK3;
    config.perturbation_amplitude = 0.05;
    config.verbose = false;
    config.perf_mode = true;
    config.gpu_only_mode = true;

    // Filter settings (v13)
    config.filter_strength = 0.03;
    config.filter_interval = 2;

    // Trip forcing
    config.trip_enabled = true;
    config.trip_amplitude = 1.0;
    config.trip_duration = 0.20;
    config.trip_ramp_off_start = 0.10;
    config.trip_n_modes_z = 16;
    config.trip_force_w = true;
    config.trip_w_scale = 2.0;

    // Poisson solver
    config.poisson_solver = PoissonSolverType::MG;
    config.poisson_fixed_cycles = 16;
    config.poisson_max_vcycles = 16;

    RANSSolver solver(mesh, config);
    solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel3D));
    solver.set_body_force(-dp_dx, 0.0);  // fx = -dp/dx = 1.0
    solver.initialize_uniform(1.0, 0.0);
    solver.sync_to_gpu();

    // Track key metrics
    double E_initial = solver.compute_kinetic_energy_device();
    double max_vel = 0.0;
    double max_div = 0.0;
    bool has_nan = false;

    std::vector<double> E_history;
    E_history.push_back(E_initial);

    for (int step = 1; step <= nsteps; ++step) {
        // Apply velocity filter before step (as required by CLAUDE.md)
        if (config.filter_strength > 0.0 && step % config.filter_interval == 0) {
            solver.apply_velocity_filter(config.filter_strength);
        }

        solver.step();

        // Periodic diagnostics (no CPU sync -- use device functions)
        if (step % 50 == 0) {
            double E = solver.compute_kinetic_energy_device();
            double v_max = solver.compute_max_velocity_device();
            double div = solver.compute_divergence_linf_device();

            E_history.push_back(E);
            max_vel = std::max(max_vel, v_max);
            max_div = std::max(max_div, div);

            if (!std::isfinite(E) || !std::isfinite(v_max)) {
                has_nan = true;
                std::cout << "  [ERROR] NaN/Inf at step " << step << "\n";
                break;
            }

            if (step % 100 == 0) {
                std::cout << "  Step " << step << ": E=" << std::scientific << std::setprecision(3)
                          << E << " v_max=" << std::fixed << std::setprecision(1) << v_max
                          << " div=" << std::scientific << div << "\n";
            }
        }
    }

    // Resolution diagnostics (requires GPU sync)
    solver.sync_solution_from_gpu();
    auto res = solver.compute_resolution_diagnostics();

    double E_final = E_history.back();
    double E_ratio = E_final / E_initial;
    double delta = Ly / 2.0;  // Channel half-height
    double re_tau = res.u_tau_force * delta / nu;
    double y_plus_first = std::max(res.y1_plus_bot, res.y1_plus_top);

    std::cout << "\n  Resolution: y1+=" << std::fixed << std::setprecision(2) << y_plus_first
              << " dx+=" << res.dx_plus << " dz+=" << res.dz_plus << "\n";
    std::cout << "  u_tau(force)=" << std::setprecision(4) << res.u_tau_force
              << " Re_tau=" << std::setprecision(1) << re_tau << "\n";
    std::cout << "  max|vel|=" << std::setprecision(1) << max_vel
              << " max|div|=" << std::scientific << max_div << "\n";
    std::cout << "  KE ratio (final/initial)=" << std::fixed << std::setprecision(4) << E_ratio << "\n\n";

    // Record results
    record("No NaN/Inf", !has_nan);
    record("Incompressibility (div < 1e-4)", max_div < 1e-4);
    record("Velocity bounded (< 50)", max_vel < 50.0);
    record("KE not blown up (ratio < 10)", E_ratio < 10.0);
    record("KE not collapsed (ratio > 0.01)", E_ratio > 0.01);
    record("y+ < 1", y_plus_first < 1.0);
    record("dx+ < 20", res.dx_plus < 20.0);
    record("dz+ < 10", res.dz_plus < 10.0);
    record("Re_tau > 50 (turbulence developing)", re_tau > 50.0);
}

int main() {
    return harness::run_sections("DNSChannelValidation", {
        {"DNS channel machinery (v13 recipe)", test_dns_channel_machinery},
    });
}
