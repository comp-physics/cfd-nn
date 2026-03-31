/// 3D Square Duct Flow Solver
/// Solves incompressible Navier-Stokes in a square duct with:
/// - Periodic boundary conditions in x (streamwise)
/// - No-slip walls at y = y_min/max and z = z_min/max
/// - Constant body force (pressure gradient) driving the flow
///
/// Analytical solution (laminar): Series expansion with max velocity at center
/// u_max = (48/π⁴) * (dp/dx) * a² / (2μ) where a is half-width

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "config.hpp"
#include "timing.hpp"
#include "turbulence_model.hpp"
#include "turbulence_baseline.hpp"
#include "turbulence_earsm.hpp"
#include "decomposition.hpp"
#include "qoi_extraction.hpp"

#include <iostream>
#include <iomanip>
#include <filesystem>
#include <cmath>
#include <fstream>
#include <filesystem>

using namespace nncfd;

/// Analytical solution for square duct laminar flow (truncated series)
/// u(y,z) = Σ Σ A_mn * sin(mπy/2a) * sin(nπz/2a)
/// Approximation using first few terms
double duct_velocity_analytical(double y, double z, double a, double dp_dx, double nu, int nterms = 5) {
    double sum = 0.0;
    double coeff = 16.0 * a * a / (M_PI * M_PI * M_PI * nu);

    for (int m = 1; m <= nterms; m += 2) {
        for (int n = 1; n <= nterms; n += 2) {
            double denom = m * n * (m * m + n * n);
            double term = std::sin(m * M_PI * (y + a) / (2.0 * a))
                        * std::sin(n * M_PI * (z + a) / (2.0 * a)) / denom;
            sum += term;
        }
    }

    return -dp_dx * coeff * sum;
}

/// Compute bulk velocity (volume-averaged)
double compute_bulk_velocity_3d(const Mesh& mesh, const VectorField& velocity) {
    double sum_u = 0.0;
    int count = 0;

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                sum_u += 0.5 * (velocity.u(i, j, k) + velocity.u(i+1, j, k));
                count++;
            }
        }
    }

    return sum_u / count;
}

/// Compute max velocity
double compute_max_velocity_3d(const Mesh& mesh, const VectorField& velocity) {
    double max_u = 0.0;

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double u = 0.5 * (velocity.u(i, j, k) + velocity.u(i+1, j, k));
                max_u = std::max(max_u, u);
            }
        }
    }

    return max_u;
}

/// Create divergence-free perturbed velocity field for DNS
/// Uses vector potential A to ensure div(u) = 0: u = curl(A)
/// Wall factors ensure u = 0 at all walls (y = ±1, z = ±1)
VectorField create_perturbed_duct_field(const Mesh& mesh, double amplitude = 1e-3) {
    VectorField vel(mesh);

    const double Lx = mesh.x_max - mesh.x_min;
    const double Ly = mesh.y_max - mesh.y_min;
    const double Lz = mesh.z_max - mesh.z_min;
    const double kx = 2.0 * M_PI / Lx;  // Streamwise wavenumber

    // Wall factors: sin²(π(y-y_min)/Ly) vanishes at y = y_min and y_max
    // Using vector potential A_z = sin(kx*x) * f_y * f_z
    //                       A_y = cos(kx*x) * f_y * f_z (phase shift for 3D)
    // u = ∂A_z/∂y - ∂A_y/∂z, v = -∂A_z/∂x, w = ∂A_y/∂x

    // u-velocity (on x-faces)
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        double z = mesh.z(k);
        double z_norm = (z - mesh.z_min) / Lz;
        double fz = std::sin(M_PI * z_norm);
        double fz2 = fz * fz;
        double dfz = (M_PI / Lz) * std::sin(2.0 * M_PI * z_norm);  // d(sin²)/dz

        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            double y = mesh.y(j);
            double y_norm = (y - mesh.y_min) / Ly;
            double fy = std::sin(M_PI * y_norm);
            double fy2 = fy * fy;
            double dfy = (M_PI / Ly) * std::sin(2.0 * M_PI * y_norm);  // d(sin²)/dy

            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                double x = mesh.xf[i];
                // u = ∂A_z/∂y - ∂A_y/∂z
                vel.u(i, j, k) = amplitude * (
                    std::sin(kx * x) * dfy * fz2 -
                    std::cos(kx * x) * fy2 * dfz
                );
            }
        }
    }

    // v-velocity (on y-faces)
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        double z = mesh.z(k);
        double z_norm = (z - mesh.z_min) / Lz;
        double fz = std::sin(M_PI * z_norm);
        double fz2 = fz * fz;

        for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
            double y = mesh.yf[j];
            double y_norm = (y - mesh.y_min) / Ly;
            double fy = std::sin(M_PI * y_norm);
            double fy2 = fy * fy;

            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double x = mesh.x(i);
                // v = -∂A_z/∂x = -kx * cos(kx*x) * fy² * fz²
                vel.v(i, j, k) = -amplitude * kx * std::cos(kx * x) * fy2 * fz2;
            }
        }
    }

    // w-velocity (on z-faces)
    for (int k = mesh.k_begin(); k <= mesh.k_end(); ++k) {
        double z = mesh.zf[k];
        double z_norm = (z - mesh.z_min) / Lz;
        double fz = std::sin(M_PI * z_norm);
        double fz2 = fz * fz;

        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            double y = mesh.y(j);
            double y_norm = (y - mesh.y_min) / Ly;
            double fy = std::sin(M_PI * y_norm);
            double fy2 = fy * fy;

            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double x = mesh.x(i);
                // w = ∂A_y/∂x = -kx * sin(kx*x) * fy² * fz²
                vel.w(i, j, k) = -amplitude * kx * std::sin(kx * x) * fy2 * fz2;
            }
        }
    }

    return vel;
}

/// Write velocity profile at duct center (y-z plane)
void write_duct_profile(const std::string& filename, const Mesh& mesh,
                        const VectorField& velocity, double dp_dx, double nu) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << "\n";
        return;
    }

    double a = (mesh.y_max - mesh.y_min) / 2.0;

    file << "# y z u_numerical u_analytical\n";

    // Sample at center x location
    int i = mesh.i_begin() + mesh.Nx / 2;

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            double y = mesh.y(j);
            double z = mesh.z(k);
            double u_num = 0.5 * (velocity.u(i, j, k) + velocity.u(i+1, j, k));
            double u_ana = duct_velocity_analytical(y, z, a, dp_dx, nu);

            file << y << " " << z << " " << u_num << " " << u_ana << "\n";
        }
        file << "\n";  // Blank line for gnuplot splot
    }
}

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --Nx N            Grid cells in x (streamwise)\n";
    std::cout << "  --Ny N            Grid cells in y\n";
    std::cout << "  --Nz N            Grid cells in z\n";
    std::cout << "  --Re R            Reynolds number\n";
    std::cout << "  --nu V            Kinematic viscosity\n";
    std::cout << "  --dp_dx D         Pressure gradient\n";
    std::cout << "  --dt T            Time step\n";
    std::cout << "  --max_steps N      Maximum iterations\n";
    std::cout << "  --tol T           Convergence tolerance\n";
    std::cout << "  --model M         Turbulence model (none, baseline, sst, etc.)\n";
    std::cout << "  --output DIR      Output directory\n";
    std::cout << "  --no_write_fields Skip VTK output\n";
    std::cout << "  --adaptive_dt     Enable adaptive time stepping\n";
}

int main(int argc, char** argv) {
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
#else
    int mpi_rank = 0;
#endif

    if (mpi_rank == 0) {
        std::cout << "=== 3D Square Duct Flow Solver ===\n\n";
    }

    // Solver configuration with duct-specific defaults
    Config config;
    config.Nx = 16;
    config.Ny = 32;
    config.Nz = 32;
    config.x_min = 0.0;
    config.x_max = 2.0 * M_PI;
    config.y_min = -1.0;
    config.y_max = 1.0;
    config.z_min = -1.0;
    config.z_max = 1.0;
    config.nu = 0.01;
    config.dp_dx = -1.0;
    config.dt = 0.001;
    config.max_steps = 10000;
    config.tol = 1e-8;
    config.output_freq = 500;
    config.verbose = true;
    config.turb_model = TurbulenceModelType::None;
    config.poisson_tol = 1e-8;
    config.poisson_max_vcycles = 20;  // V-cycles for multigrid (not SOR iterations)

    // Parse command line (handles all args including --Nx, --Ny, --Nz, etc.)
    config.parse_args(argc, argv);

    // Print configuration
    config.print();

    // Compute expected values (only used for steady-state validation)
    double a = (config.y_max - config.y_min) / 2.0;
    double u_center_approx = duct_velocity_analytical(0.0, 0.0, a, config.dp_dx, config.nu);
    // Only print expected velocities for steady mode - these are meaningless for short unsteady runs
    if (config.simulation_mode == SimulationMode::Steady) {
        std::cout << "Expected centerline velocity (approx): " << u_center_approx << "\n\n";
    }

    // Determine required ghost cells based on scheme requirements
    // O4 spatial operators and Upwind2 both need 5-point stencils (i±2), requiring Nghost >= 2
    int nghost = 1;  // Default for O2 schemes
    if (config.space_order == 4 || config.convective_scheme == ConvectiveScheme::Upwind2) {
        nghost = 2;
        std::cout << "Using Nghost = 2 for higher-order stencils\n";
    }

    // Create 3D mesh
    Mesh mesh;
    mesh.init_uniform(config.Nx, config.Ny, config.Nz,
                      config.x_min, config.x_max,
                      config.y_min, config.y_max,
                      config.z_min, config.z_max, nghost);

    std::cout << "Mesh: " << mesh.Nx << " x " << mesh.Ny << " x " << mesh.Nz << " cells (3D)\n";
    std::cout << "dx = " << mesh.dx << ", dy = " << mesh.dy << ", dz = " << mesh.dz << "\n\n";

    // Create MPI decomposition
#ifdef USE_MPI
    Decomposition decomp(MPI_COMM_WORLD, config.Nz);
#else
    Decomposition decomp(config.Nz);
#endif

    // Create solver
    RANSSolver solver(mesh, config);
    solver.set_decomposition(&decomp);

    // Set 3D boundary conditions
    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;  // Streamwise periodic
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;    // Walls
    bc.y_hi = VelocityBC::NoSlip;
    bc.z_lo = VelocityBC::NoSlip;    // Walls
    bc.z_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);

    // Set body force (pressure gradient)
    solver.set_body_force(-config.dp_dx, 0.0, 0.0);

    // Set initial turbulence model — must be set BEFORE sync_to_gpu()
    // so GPU buffers are mapped for the model that actually runs first.
    // If warm-up is configured and target is non-transport, start with SST warm-up model.
    // Otherwise start with the target model directly.
    bool needs_warmup_swap = false;
    if (config.turb_model != TurbulenceModelType::None) {
        auto probe = create_turbulence_model(config.turb_model, "", "");
        bool target_has_transport = probe && probe->uses_transport_equations();
        bool has_warmup = !config.warmup_model.empty() && config.warmup_time > 0.0;

        if (has_warmup && !target_has_transport) {
            // Non-transport target (NN, GEP): start with SST warm-up model
            TurbulenceModelType warmup_type = TurbulenceModelType::SSTKOmega;
            if (config.warmup_model == "komega") warmup_type = TurbulenceModelType::KOmega;
            auto warmup_turb = create_turbulence_model(warmup_type, "", "");
            if (warmup_turb) {
                warmup_turb->set_nu(config.nu);
                solver.set_turbulence_model(std::move(warmup_turb));
                needs_warmup_swap = true;  // swap to target after warm-up
            }
        } else {
            // Transport target or no warm-up: use target directly
            auto turb_model = create_turbulence_model(config.turb_model,
                                                      config.nn_weights_path,
                                                      config.nn_scaling_path,
                                                      config.pope_C1,
                                                      config.pope_C2);
            if (turb_model) {
                turb_model->set_nu(config.nu);
                solver.set_turbulence_model(std::move(turb_model));
            }
        }
    }

    // Print final solver configuration (after all setup)
    solver.print_solver_info();

    // Determine simulation mode
    bool is_unsteady = (config.simulation_mode == SimulationMode::Unsteady);

    // Initialize based on simulation mode
    if (is_unsteady) {
        // DNS/unsteady: use divergence-free perturbations to seed instability
        std::cout << "Initializing with divergence-free perturbations for DNS...\n";
        std::cout << "Perturbation amplitude: " << config.perturbation_amplitude << "\n";
        solver.initialize(create_perturbed_duct_field(mesh, config.perturbation_amplitude));
    } else {
        // Steady: start with small uniform velocity
        solver.initialize_uniform(0.1 * std::abs(u_center_approx), 0.0);
    }

    // For RANS transport models: seed k/omega to turbulent values.
    // Without this, SST k stays at zero on uniform grids because
    // P_k = nu_t * |S|^2 = 0 when nu_t = 0 (chicken-and-egg problem).
    // Key insight: nu_t = k/omega (times blending), so omega must be LOW
    // enough that nu_t/nu ~ 100 in the freestream. Otherwise the SST
    // destruction term beta_star*k*omega overwhelms production P_k = nu_t*S^2.
    if (config.turb_model != TurbulenceModelType::None) {
        double U_b_est = std::max(std::abs(u_center_approx), 0.1);
        double Ti = 0.10;  // 10% turbulence intensity (higher for sustaining k)
        double k_seed = std::max(1.5 * (U_b_est * Ti) * (U_b_est * Ti), 1e-4);
        double nu_t_ratio = 100.0;  // initial nu_t / nu ~ 100 (RANS freestream)
        double omega_seed = k_seed / (0.09 * config.nu * nu_t_ratio);
        omega_seed = std::max(omega_seed, 1.0);

        std::cout << "Seeding turbulence: k=" << std::scientific << k_seed
                  << " omega=" << omega_seed << " (nu_t/nu~" << std::fixed
                  << std::setprecision(0) << nu_t_ratio << ")\n";

        std::fill(solver.k_mutable().data().begin(), solver.k_mutable().data().end(), k_seed);
        std::fill(solver.omega_mutable().data().begin(), solver.omega_mutable().data().end(), omega_seed);
    }

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    if (is_unsteady) {
        std::cout << "=== Running UNSTEADY simulation (DNS) ===\n";
        std::cout << "Max iterations: " << config.max_steps << "\n\n";
    } else {
        std::cout << "=== Running STEADY solve ===\n";
        std::cout << "Convergence tolerance: " << config.tol << "\n\n";
    }

    std::cout << std::setw(10) << "Iter"
              << std::setw(15) << "Residual"
              << std::setw(15) << "Max |u|"
              << std::setw(15) << "Bulk u"
              << "\n";

    // Warm-up phase: run with SST (or the target model with closure disabled)
    // to develop the flow before enabling the full anisotropic closure.
    //
    // For transport models (SST, EARSM, RSM): initialize with the TARGET model
    // directly so k/omega state is preserved. Disable EARSM closure during warm-up.
    // For non-transport models (NN, GEP): initialize with SST warm-up, then switch
    // to target model with background SST transport to keep k/omega alive.
    if (!config.warmup_model.empty() && config.warmup_time > 0.0) {
        // Model already set above (warm-up SST or target transport).
        // Disable EARSM closure during warm-up (acts as pure SST)
        auto* earsm = dynamic_cast<SSTWithEARSM*>(solver.turb_model_ptr());
        if (earsm) {
            earsm->set_closure_active(false);
        }

        std::cout << "=== Warm-up phase: " << config.warmup_model
                  << " for t=" << config.warmup_time << " ===\n";

        for (int ws = 1; ws <= config.max_steps; ++ws) {
            if (config.adaptive_dt) {
                solver.set_dt(solver.compute_adaptive_dt());
            }
            double res = solver.step();
            double t = solver.current_time();
            if (t >= config.warmup_time) {
                std::cout << "Warm-up complete at step " << ws
                          << ", t=" << t << ", res=" << res << "\n";
                break;
            }
            if (std::isnan(res) || std::isinf(res) || res > 1e10) {
                std::cerr << "Warm-up diverged at step " << ws << "\n";
                break;
            }
        }

        // Switch to target model after warm-up.
        // Transport models (SST, EARSM, RSM) were already initialized at the start
        // and developed during warm-up — skip re-creation to preserve k/omega state.
        // Non-transport models (NN, algebraic, GEP) need fresh creation with
        // background SST to keep k/omega alive.
        if (needs_warmup_swap && config.turb_model != TurbulenceModelType::None) {
            auto target_turb = create_turbulence_model(config.turb_model,
                                                       config.nn_weights_path,
                                                       config.nn_scaling_path,
                                                       config.pope_C1,
                                                       config.pope_C2);
            if (target_turb) {
                auto bg_sst = create_turbulence_model(
                    TurbulenceModelType::SSTKOmega, "", "");
                if (bg_sst) {
                    bg_sst->set_nu(config.nu);
                    solver.set_background_transport(std::move(bg_sst));
                }
                target_turb->set_nu(config.nu);
                solver.set_turbulence_model(std::move(target_turb));
                solver.sync_to_gpu();  // Re-map for new model

                // Ramp tau_div from 0→1 over 200 steps to prevent divergence
                // from sudden anisotropic correction on SST-developed flow.
                if (target_turb == nullptr) {  // moved, check solver
                    // target_turb was moved; check via solver
                }
                solver.start_tau_div_ramp(5000);
                std::cout << "  tau_div ramp: 0→1 over 5000 steps\n";
            }
        }
        // else: transport models keep running from warm-up (no switch needed)

        TimingStats::instance().reset();
        std::cout << "=== Evaluation phase from t="
                  << solver.current_time() << " ===\n";

        // Re-enable EARSM closure after warm-up
        if (earsm) {
            earsm->set_closure_active(true);
            std::cout << "  EARSM closure re-activated\n";
        }
    }

    ScopedTimer total_timer("Total simulation", false);

    // Setup VTK snapshots
    const std::string snapshot_prefix = config.write_fields ? (config.output_dir + "duct") : "";
    const int snapshot_freq = (config.num_snapshots > 0 && config.max_steps > 0) ?
        std::max(1, config.max_steps / config.num_snapshots) : 0;
    int snap_count = 0;

    if (!snapshot_prefix.empty()) {
        std::filesystem::create_directories(config.output_dir);
    }

    double residual = 1.0;
    int iter = 0;

    // Progress output interval for CI visibility (always enabled)
    const int progress_interval = std::max(1, config.max_steps / 10);

    // Unsteady: run all iterations; Steady: check convergence
    for (iter = 1; iter <= config.max_steps; ++iter) {
        // For steady mode, check convergence
        if (!is_unsteady && residual <= config.tol) {
            break;
        }

        if (config.adaptive_dt) {
            (void)solver.compute_adaptive_dt();
        }
        residual = solver.step();

        // Physical time limit
        {
            double time = solver.current_time();
            if (config.T_final > 0.0 && time >= config.T_final) {
                std::cout << "Reached T_final=" << config.T_final
                          << " at step " << iter << ", t=" << time << "\n";
                break;
            }
        }

        // Reset timers after warmup iterations (excluded from reported timing)
        if (config.warmup_steps > 0 && iter == config.warmup_steps) {
            TimingStats::instance().reset();
            std::cout << "=== Timers reset after " << config.warmup_steps
                      << " warmup steps, t=" << solver.current_time() << " ===\n";
        }

        // Write VTK snapshot
        if (!snapshot_prefix.empty() && snapshot_freq > 0 && (iter % snapshot_freq == 0)) {
#ifdef USE_GPU_OFFLOAD
            solver.sync_from_gpu();
#endif
            ++snap_count;
            solver.write_vtk(snapshot_prefix + "_" + std::to_string(snap_count) + ".vtk");
        }

        // Always show progress every ~10% for CI visibility
        if (iter % progress_interval == 0 || iter == 1) {
            std::cout << "    Step " << std::setw(6) << iter << " / " << config.max_steps
                      << "  (" << std::setw(3) << (100 * iter / config.max_steps) << "%)"
                      << "  res=" << std::scientific << std::setprecision(3) << residual
                      << std::fixed << "\n" << std::flush;
        } else if (config.verbose && (iter % config.output_freq == 0)) {
#ifdef USE_GPU_OFFLOAD
            solver.sync_from_gpu();
#endif
            double max_u = compute_max_velocity_3d(mesh, solver.velocity());
            double bulk_u = compute_bulk_velocity_3d(mesh, solver.velocity());

            std::cout << std::setw(10) << iter
                      << std::setw(15) << std::scientific << residual
                      << std::setw(15) << std::fixed << max_u
                      << std::setw(15) << bulk_u
                      << "\n";
        }

        if (std::isnan(residual) || std::isinf(residual)) {
            std::cerr << "ERROR: Solver diverged at iteration " << iter << "\n";
            return 1;
        }
    }

    total_timer.stop();

#ifdef USE_GPU_OFFLOAD
    solver.sync_from_gpu();
#endif

    // QoI extraction: cross-section velocity profiles at mid-x
    // Runs on CPU after sync_from_gpu (above). Cost: ~0.8ms for sync,
    // negligible for CPU extraction. At qoi_freq=50: 0.016 ms/step amortized.
    if (config.qoi_freq > 0) {
        std::filesystem::create_directories(config.qoi_output_dir);

        const auto& vel = solver.velocity();
        const int Nx = mesh.Nx, Ny = mesh.Ny, Nz = mesh.Nz, Ng = mesh.Nghost;
        const int i_mid = Nx / 2;  // Mid-x station

        // Extract U(y,z), V(y,z), W(y,z) cross-section
        std::vector<double> u_yz(Ny * Nz), v_yz(Ny * Nz), w_yz(Ny * Nz);
        nncfd::qoi::extract_cross_section_device(
            vel.u_data().data(), vel.v_data().data(), vel.w_data().data(),
            vel.u_stride(), vel.v_stride(), vel.w_stride(),
            vel.u_plane_stride(), vel.v_plane_stride(), vel.w_plane_stride(),
            i_mid, Nx, Ny, Nz, Ng,
            u_yz.data(), v_yz.data(), w_yz.data());

        // Build coordinate arrays
        std::vector<double> yc_arr(Ny), zc_arr(Nz);
        for (int j = 0; j < Ny; ++j) yc_arr[j] = mesh.yc[j + Ng];
        for (int k = 0; k < Nz; ++k) zc_arr[k] = mesh.zc[k + Ng];

        nncfd::qoi::write_cross_section(
            config.qoi_output_dir + "/duct_cross_section.dat",
            yc_arr.data(), zc_arr.data(), u_yz.data(), v_yz.data(), w_yz.data(),
            Ny, Nz, "y z U V W");

        // Wall shear stress along y-walls (averaged over x)
        std::vector<double> tau_bot(Nz), tau_top(Nz);
        nncfd::qoi::compute_wall_shear_y_device(
            vel.u_data().data(), vel.u_stride(), vel.u_plane_stride(),
            config.nu, mesh.dy,
            Nx, Ny, Nz, Ng,
            tau_bot.data(), tau_top.data());

        nncfd::qoi::write_profile(
            config.qoi_output_dir + "/wall_shear_y.dat",
            zc_arr.data(), tau_bot.data(), Nz, "z tau_w_bottom tau_w_top");

        std::cout << "QoI written to " << config.qoi_output_dir << "/\n";
    }

    // Results
    std::cout << "\n=== Results ===\n";
    std::cout << "Final residual: " << std::scientific << residual << "\n";
    std::cout << "Iterations: " << iter - 1 << "\n";
    if (is_unsteady) {
        std::cout << "Mode: Unsteady (ran all " << config.max_steps << " iterations)\n";
        std::cout << "VTK snapshots written: " << snap_count << "\n";
    } else {
        std::cout << "Converged: " << (residual < config.tol ? "YES" : "NO") << "\n";
    }

    double max_u = compute_max_velocity_3d(mesh, solver.velocity());
    double bulk_u = compute_bulk_velocity_3d(mesh, solver.velocity());

    // Secondary flow diagnostics: max |v| and max |w|
    double max_v = 0.0, max_w = 0.0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k)
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j)
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                max_v = std::max(max_v, std::abs(solver.velocity().v(i, j, k)));
                max_w = std::max(max_w, std::abs(solver.velocity().w(i, j, k)));
            }

    std::cout << "Max velocity: " << std::fixed << std::setprecision(6) << max_u << "\n";
    std::cout << "Bulk velocity: " << bulk_u << "\n";
    std::cout << "Max |v| (secondary): " << std::scientific << std::setprecision(6) << max_v << "\n";
    std::cout << "Max |w| (secondary): " << max_w << "\n";

    // Turbulence diagnostics — use 2D accessors for 2D meshes since
    // turbulence fields are stored in 2D layout (k=0 plane only)
    double max_k = 0.0, max_nut = 0.0;
    if (mesh.is2D()) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j)
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                max_k = std::max(max_k, solver.k()(i, j));
                max_nut = std::max(max_nut, solver.nu_t()(i, j));
            }
    } else {
        for (int kk = mesh.k_begin(); kk < mesh.k_end(); ++kk)
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j)
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    max_k = std::max(max_k, solver.k()(i, j, kk));
                    max_nut = std::max(max_nut, solver.nu_t()(i, j, kk));
                }
    }
    std::cout << "Max k: " << max_k << "\n";
    std::cout << "Max nu_t: " << max_nut << std::fixed << "\n";

    // Only compare against analytical solution for steady-state runs
    if (!is_unsteady && config.turb_model == TurbulenceModelType::None) {
        std::cout << "Expected centerline (approx): " << u_center_approx << "\n";
        double error = std::abs(max_u - u_center_approx) / std::abs(u_center_approx);
        std::cout << "Centerline error: " << error * 100.0 << "%\n";

        if (error < 0.05) {
            std::cout << "\n*** VALIDATION PASSED: Error < 5% ***\n";
        }
    }

    // Write output
    if (config.write_fields) {
        try {
            std::filesystem::create_directories(config.output_dir);
            solver.write_vtk(config.output_dir + "duct_final.vtk");
            write_duct_profile(config.output_dir + "duct_profile.dat", mesh,
                              solver.velocity(), config.dp_dx, config.nu);
            std::cout << "\nWrote output to " << config.output_dir << "\n";
        } catch (const std::exception& e) {
            std::cerr << "Warning: Could not write output: " << e.what() << "\n";
        }
    }

    // Print timing summary
    if (mpi_rank == 0) {
        TimingStats::instance().print_summary();
    }

#ifdef USE_MPI
    MPI_Finalize();
#endif

    return 0;
}
