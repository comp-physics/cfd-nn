/// Cylinder flow solver with immersed boundary method
/// Solves incompressible Navier-Stokes around a 2D/3D cylinder using
/// direct-forcing IBM. Outputs drag and lift coefficients.
///
/// Domain: [0, Lx] x [-Ly/2, Ly/2] x [0, Lz]
/// Cylinder center at (x_c, 0) with radius R
/// Inflow: uniform u = U_inf
/// Outflow: convective (approximated by periodic + sponge)

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "config.hpp"
#include "timing.hpp"
#include "ibm_geometry.hpp"
#include "ibm_forcing.hpp"
#include "decomposition.hpp"
#include "turbulence_model.hpp"
#include "qoi_extraction.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <filesystem>

using namespace nncfd;

int main(int argc, char** argv) {
#ifdef USE_MPI
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        std::cerr << "[MPI] MPI_Init failed\n";
        return 1;
    }
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
#else
    int mpi_rank = 0;
#endif

    if (mpi_rank == 0) {
        std::cout << "=== IBM Flow Solver (cylinder/sphere) ===\n\n";
    }

    // Parse configuration
    Config config;

    // Default cylinder flow settings
    config.Nx = 128;
    config.Ny = 128;
    config.Nz = 1;
    config.x_min = 0.0;
    config.x_max = 30.0;
    config.y_min = -10.0;
    config.y_max = 10.0;
    config.z_min = 0.0;
    config.z_max = M_PI;

    config.nu = 0.01;
    config.dp_dx = 0.0;

    config.dt = 0.001;
    config.max_steps = 10000;
    config.tol = 1e-8;
    config.output_freq = 100;
    config.verbose = true;
    config.simulation_mode = SimulationMode::Unsteady;
    config.adaptive_dt = true;

    config.turb_model = TurbulenceModelType::None;

    config.poisson_tol = 1e-6;
    config.poisson_max_vcycles = 20;

    // Parse command line
    config.parse_args(argc, argv);

    // IBM body parameters: use config values, with sensible defaults
    double body_r = config.ibm_radius;
    double body_cx = config.ibm_cx;
    double body_cy = config.ibm_cy;
    double body_cz = config.ibm_cz;
    double U_inf = 1.0;

    // Auto-center: if cx is 0 (default), place body at ~1/3 of domain
    if (body_cx == 0.0) {
        body_cx = config.x_min + (config.x_max - config.x_min) / 3.0;
    }
    if (body_cy == 0.0) {
        body_cy = (config.y_min + config.y_max) / 2.0;
    }
    if (body_cz == 0.0) {
        body_cz = (config.z_min + config.z_max) / 2.0;
    }

    // Compute Re based on diameter
    double Re = U_inf * (2.0 * body_r) / config.nu;

    if (mpi_rank == 0) {
        config.print();
        std::cout << "\nIBM body: " << config.ibm_body
                  << ", center=(" << body_cx << ", " << body_cy;
        if (config.Nz > 1) std::cout << ", " << body_cz;
        std::cout << "), radius=" << body_r << "\n";
        std::cout << "Re (based on diameter) = " << Re << "\n";
        std::cout << "U_inf = " << U_inf << "\n\n";
    }

    // Ensure output directory exists
    try {
        std::filesystem::create_directories(config.output_dir);
    } catch (const std::exception& e) {
        std::cerr << "Warning: Could not create output directory: " << e.what() << "\n";
    }

    bool is3D = config.Nz > 1;

    // Create mesh
    Mesh mesh;
    if (is3D) {
        mesh.init_uniform(config.Nx, config.Ny, config.Nz,
                          config.x_min, config.x_max,
                          config.y_min, config.y_max,
                          config.z_min, config.z_max);
    } else {
        mesh.init_uniform(config.Nx, config.Ny,
                          config.x_min, config.x_max,
                          config.y_min, config.y_max);
    }

    if (mpi_rank == 0) {
        std::cout << "Mesh: " << mesh.Nx << " x " << mesh.Ny;
        if (is3D) std::cout << " x " << mesh.Nz;
        std::cout << " cells\n";
        std::cout << "dx = " << mesh.dx << ", dy = " << mesh.dy;
        if (is3D) std::cout << ", dz = " << mesh.dz;
        std::cout << "\n\n";
    }

    // Create MPI decomposition
#ifdef USE_MPI
    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    Decomposition decomp(MPI_COMM_WORLD, config.Nz);
#else
    Decomposition decomp(config.Nz);
#endif

    // Create IBM body from config
    std::shared_ptr<IBMBody> body;
    if (config.ibm_body == "sphere") {
        body = std::make_shared<SphereBody>(body_cx, body_cy, body_cz, body_r);
    } else {
        body = std::make_shared<CylinderBody>(body_cx, body_cy, body_r);
    }
    IBMForcing ibm(mesh, body);
    if (config.ibm_eta > 0.0) ibm.set_penalization_eta(config.ibm_eta);
    ibm.set_ghost_cell_ibm(true);
    ibm.recompute_weights();

    // Create solver
    RANSSolver solver(mesh, config);
    solver.set_decomposition(&decomp);
    solver.set_ibm_forcing(&ibm);

    // Boundary conditions: periodic in x and z, periodic in y (large domain)
    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::Periodic;
    bc.y_hi = VelocityBC::Periodic;
    if (is3D) {
        bc.z_lo = VelocityBC::Periodic;
        bc.z_hi = VelocityBC::Periodic;
    }
    solver.set_velocity_bc(bc);

    // Body force: fixed dp/dx drives flow in periodic domain.
    // Cd is computed from momentum balance: Cd = |dp/dx| * V / (0.5 * U_b^2 * A_ref).
    // The dp/dx determines the Reynolds number; measure U_b to compute Re = U_b*D/nu.
    if (config.dp_dx != 0.0) {
        solver.set_body_force(-config.dp_dx, 0.0);
    }

    // Set initial turbulence model — must be set BEFORE sync_to_gpu()
    // so GPU buffers are mapped for the model that actually runs first.
    bool needs_warmup_swap = false;
    if (config.turb_model != TurbulenceModelType::None) {
        auto probe = create_turbulence_model(config.turb_model, "", "");
        bool target_has_transport = probe && probe->uses_transport_equations();
        bool has_warmup = !config.warmup_model.empty() && config.warmup_time > 0.0;

        if (has_warmup && !target_has_transport) {
            // Non-transport target: start with SST warm-up model
            TurbulenceModelType warmup_type = TurbulenceModelType::SSTKOmega;
            if (config.warmup_model == "komega") warmup_type = TurbulenceModelType::KOmega;
            auto warmup_turb = create_turbulence_model(warmup_type, "", "");
            if (warmup_turb) {
                warmup_turb->set_nu(config.nu);
                solver.set_turbulence_model(std::move(warmup_turb));
                needs_warmup_swap = true;
            }
        } else {
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

    solver.print_solver_info();

    // Initialize with uniform freestream velocity.
    // The IBM penalization will damp velocity inside the body to zero
    // during the first few steps. Using ibm_eta > 0 prevents divergence
    // from the pressure transient.
    solver.initialize_uniform(U_inf, 0.0);

    // Add deterministic perturbation to break symmetry for vortex shedding
    // Asymmetric v-velocity kick in the near wake — triggers shedding
    {
        auto& vel = solver.velocity();
        double amp_base = config.perturbation_amplitude * U_inf;
        double wake_len = 5.0 * body_r;  // Perturbation extends 5 radii downstream

        if (is3D) {
            for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
                for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                    for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                        double xc = mesh.xc[i];
                        double yc = mesh.yc[j];
                        double zc = mesh.zc[k];
                        double dx = xc - body_cx;
                        double dy = yc - body_cy;
                        double dz = zc - body_cz;
                        double r = std::sqrt(dx * dx + dy * dy + dz * dz);
                        // Only perturb in near-wake: downstream of body, within a few radii
                        if (dx > 0.0 && dx < wake_len && r > body_r) {
                            double env = std::exp(-dy * dy / (body_r * body_r))
                                       * std::exp(-dz * dz / (body_r * body_r))
                                       * (1.0 - std::exp(-dx / body_r));
                            vel.v(i, j, k) += amp_base * env * std::sin(M_PI * yc / body_r);
                            vel.w(i, j, k) += amp_base * env * std::sin(M_PI * zc / body_r);
                        }
                    }
                }
            }
        } else {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                    double xc = mesh.xc[i];
                    double yc = mesh.yc[j];
                    double dx = xc - body_cx;
                    double dy = yc - body_cy;
                    double r = std::sqrt(dx * dx + dy * dy);
                    // Only perturb in near-wake: downstream of body, within a few radii
                    if (dx > 0.0 && dx < wake_len && r > body_r) {
                        double env = std::exp(-dy * dy / (body_r * body_r))
                                   * (1.0 - std::exp(-dx / body_r));
                        vel.v(i, j) += amp_base * env * std::sin(M_PI * yc / body_r);
                    }
                }
            }
        }
    }

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    // Open drag/lift output file
    std::ofstream force_file;
    if (mpi_rank == 0) {
        force_file.open(config.output_dir + "forces.dat");
        if (!force_file.is_open()) {
            std::cerr << "Warning: Could not open " << config.output_dir << "forces.dat\n";
        } else {
            force_file << "# step  time  Fx  Fy  Cd  Cl\n";
        }
    }

    // VTK snapshot setup
    const std::string vtk_prefix = config.write_fields ?
        (config.output_dir + "cylinder") : "";
    const int snapshot_freq = (config.num_snapshots > 0 && config.write_fields) ?
        std::max(1, config.max_steps / config.num_snapshots) : 0;
    int snap_count = 0;

    // Warm-up phase: model already set above (warm-up SST or target transport).
    // For SIMPLE: always use RK3 during warm-up to develop the flow field.
    // SIMPLE needs a developed velocity (not zero) so div(u*) ≠ 0.
    const bool simple_mode = (config.time_integrator == TimeIntegrator::SIMPLE);
    if (simple_mode) {
        solver.set_time_integrator(TimeIntegrator::RK3);
    }

    if (!config.warmup_model.empty() && config.warmup_time > 0.0) {
        {
            if (mpi_rank == 0) {
                std::cout << "=== Warm-up phase: " << config.warmup_model
                          << " for t=" << config.warmup_time
                          << (simple_mode ? " (RK3 for SIMPLE warm-up)" : "")
                          << " ===\n";
            }

            const int warmup_max_steps = 500000;
            for (int ws = 1; ws <= warmup_max_steps; ++ws) {
                // Always use adaptive dt during warm-up (SIMPLE's dt may be too large)
                solver.set_dt(solver.compute_adaptive_dt());
                double res = solver.step();
                double t = solver.current_time();
                if (t >= config.warmup_time) {
                    if (mpi_rank == 0) {
                        std::cout << "Warm-up complete at step " << ws
                                  << ", t=" << t << ", res=" << res << "\n";
                    }
                    break;
                }
                if (std::isnan(res) || std::isinf(res) || res > 1e10) {
                    if (mpi_rank == 0) {
                        std::cerr << "Warm-up diverged at step " << ws << "\n";
                    }
                    break;
                }
            }

            // Switch to target model after warm-up
            if (needs_warmup_swap && config.turb_model != TurbulenceModelType::None) {
                auto target_turb = create_turbulence_model(config.turb_model,
                                                           config.nn_weights_path,
                                                           config.nn_scaling_path,
                                                           config.pope_C1,
                                                           config.pope_C2);
                if (target_turb) {
                    if (!target_turb->uses_transport_equations()) {
                        auto bg_sst = create_turbulence_model(
                            TurbulenceModelType::SSTKOmega, "", "");
                        if (bg_sst) {
                            bg_sst->set_nu(config.nu);
                            solver.set_background_transport(std::move(bg_sst));
                        }
                    }
                    target_turb->set_nu(config.nu);
                    solver.set_turbulence_model(std::move(target_turb));
                    solver.sync_to_gpu();  // Re-map for new model

                    // Ramp tau_div from 0→1 over 5000 steps to prevent divergence
                    // from sudden anisotropic correction on SST-developed flow.
                    solver.start_tau_div_ramp(5000);
                    std::cout << "  tau_div ramp: 0→1 over 5000 steps\n";
                }
            } else if (config.turb_model == TurbulenceModelType::None) {
                solver.set_turbulence_model(nullptr);
            }

            // Switch back to SIMPLE for evaluation phase
            if (simple_mode) {
                solver.set_time_integrator(TimeIntegrator::SIMPLE);
                if (mpi_rank == 0) {
                    std::cout << "  Switching to SIMPLE for evaluation\n";
                }
            }

            // Reset timing stats for the evaluation phase
            TimingStats::instance().reset();
            if (mpi_rank == 0) {
                std::cout << "=== Evaluation phase from t="
                          << solver.current_time() << " ===\n";
            }
        }
    }

    // Time stepping loop
    ScopedTimer total_timer("Total simulation", false);

    // Store force time-series for Strouhal number computation
    std::vector<double> force_times;
    std::vector<double> cl_history;
    std::vector<double> cd_history;

    double rho = 1.0;  // Density (incompressible)
    // Reference area for force coefficients
    double A_ref;
    if (config.ibm_body == "sphere") {
        A_ref = M_PI * body_r * body_r;  // Frontal area of sphere
    } else {
        A_ref = 2.0 * body_r * (is3D ? (config.z_max - config.z_min) : 1.0);
    }
    double q_inf = 0.5 * rho * U_inf * U_inf;

    for (int step = 1; step <= config.max_steps; ++step) {
        if (config.adaptive_dt) {
            solver.set_dt(solver.compute_adaptive_dt());
        }

        bool need_forces = (step % config.output_freq == 0 || step == 1);
        ibm.set_accumulate_forces(need_forces);

        double residual = solver.step();

        // Reset timers after warmup steps (model stabilization phase)
        if (config.warmup_steps > 0 && step == config.warmup_steps) {
            TimingStats::instance().reset();
            if (mpi_rank == 0) {
                std::cout << "=== Timers reset after " << config.warmup_steps
                          << " warmup steps, t=" << solver.current_time() << " ===\n";
            }
        }

        double time = solver.current_time();

        // Physical time limit
        if (config.T_final > 0.0 && time >= config.T_final) {
            if (mpi_rank == 0) {
                std::cout << "Reached T_final=" << config.T_final
                          << " at step " << step << ", t=" << time << "\n";
            }
            break;
        }

        if (need_forces) {
            auto [Fx, Fy, Fz] = ibm.compute_forces(solver.velocity(), solver.current_dt());

            // Momentum balance Cd: at stat. steady state, dp/dx * V = Drag.
            // Cd = |dp/dx| * V_domain / (0.5 * rho * U_bulk^2 * A_ref)
            // Re = U_bulk * D / nu (computed from actual bulk velocity)
            double U_b = solver.bulk_velocity();
            double Lx = config.x_max - config.x_min;
            double Ly = config.y_max - config.y_min;
            double Lz = is3D ? (config.z_max - config.z_min) : 1.0;
            double V_domain = Lx * Ly * Lz;
            double q_bulk = 0.5 * rho * std::max(U_b, 1e-10) * std::max(U_b, 1e-10);
            double Cd = std::abs(config.dp_dx) * V_domain / (q_bulk * A_ref);
            double Cl = Fy / (q_bulk * A_ref);

            // Store for Strouhal computation
            force_times.push_back(time);
            cl_history.push_back(Cl);
            cd_history.push_back(Cd);

            if (mpi_rank == 0) {
                if (force_file.is_open()) {
                    force_file << step << " " << time << " "
                               << Fx << " " << Fy << " "
                               << Cd << " " << Cl << "\n";
                    force_file.flush();
                }

                std::cout << "Step " << std::setw(6) << step
                          << "  t=" << std::fixed << std::setprecision(4) << time
                          << "  res=" << std::scientific << std::setprecision(3) << residual
                          << "  Cd=" << std::fixed << std::setprecision(4) << Cd
                          << "  Cl=" << std::setprecision(4) << Cl
                          << "\n" << std::flush;
            }
        } else if (mpi_rank == 0 && !config.perf_mode) {
            std::cout << "Step " << std::setw(6) << step
                      << "  t=" << std::fixed << std::setprecision(4) << time
                      << "  res=" << std::scientific << std::setprecision(3) << residual
                      << "\n" << std::flush;
        }

        // Write VTK snapshot at regular intervals
        if (!vtk_prefix.empty() && snapshot_freq > 0 && (step % snapshot_freq == 0)) {
            ++snap_count;
            solver.write_vtk(vtk_prefix + "_" + std::to_string(snap_count) + ".vtk");
        }

        if (std::isnan(residual) || std::isinf(residual) || residual > 1e10) {
            if (mpi_rank == 0) {
                std::cerr << "STOPPING: Solver diverged at step " << step
                          << " (residual=" << residual << ")\n";
            }
            break;
        }

        // Early termination: dt stuck at floor (excessive nu_t from model)
        if (config.dt_min > 0.0 && solver.current_dt() <= config.dt_min * 1.01) {
            static int dt_floor_count = 0;
            dt_floor_count = dt_floor_count + 1;
            if (dt_floor_count >= 100) {
                if (mpi_rank == 0) {
                    std::cerr << "STOPPING: dt stuck at floor (" << config.dt_min
                              << ") for 100 steps at step " << step
                              << " — model producing excessive nu_t\n";
                }
                break;
            }
        }
    }

    // Write final VTK snapshot
    if (!vtk_prefix.empty()) {
        solver.write_vtk(vtk_prefix + "_final.vtk");
    }

    total_timer.stop();

#ifdef USE_GPU_OFFLOAD
    solver.sync_from_gpu();
#endif

    // QoI extraction
    if (config.qoi_freq > 0 && mpi_rank == 0) {
        std::filesystem::create_directories(config.qoi_output_dir);
        const auto& vel = solver.velocity();
        const int Nx = mesh.Nx, Ny = mesh.Ny, Nz = mesh.Nz, Ng = mesh.Nghost;
        double D = 2.0 * body_r;

        // Strouhal number from Cl time-series
        double St = qoi::compute_strouhal(force_times, cl_history, D, U_inf);
        std::cout << "\n--- QoI Summary ---\n";
        if (St > 0.0) {
            std::cout << "  Strouhal number St = " << std::fixed
                      << std::setprecision(4) << St << "\n";
        } else {
            std::cout << "  Strouhal number: not enough oscillation cycles\n";
        }

        // Mean Cd from second half of time series
        if (cd_history.size() > 1) {
            int start = static_cast<int>(cd_history.size()) / 2;
            double cd_mean = 0.0;
            for (int i = start; i < static_cast<int>(cd_history.size()); ++i) {
                cd_mean += cd_history[i];
            }
            cd_mean /= (cd_history.size() - start);
            std::cout << "  Mean Cd = " << std::fixed << std::setprecision(4)
                      << cd_mean << "\n";
        }

        // Write QoI summary file
        {
            std::ofstream qf(config.qoi_output_dir + "/qoi_summary.dat");
            if (qf) {
                qf << "# QoI summary\n";
                qf << "Re " << Re << "\n";
                qf << "St " << St << "\n";
                if (!cd_history.empty()) {
                    int start = static_cast<int>(cd_history.size()) / 2;
                    double cd_mean = 0.0;
                    for (int i = start; i < static_cast<int>(cd_history.size()); ++i)
                        cd_mean += cd_history[i];
                    cd_mean /= (cd_history.size() - start);
                    qf << "Cd_mean " << cd_mean << "\n";
                }
            }
        }

        // Separation angle (sphere only)
        if (config.ibm_body == "sphere") {
            double probe_offset = 1.5 * mesh.dx;
            double sep_angle = qoi::compute_separation_angle_sphere(
                vel.u_data().data(), vel.v_data().data(),
                vel.u_stride(), vel.v_stride(),
                is3D ? vel.u_plane_stride() : 0,
                is3D ? vel.v_plane_stride() : 0,
                body_cx, body_cy, body_r, probe_offset,
                mesh.xf.data(), mesh.yf.data(),
                Nx, Ny, Nz, Ng);
            if (sep_angle > 0.0) {
                std::cout << "  Separation angle = " << std::fixed
                          << std::setprecision(1) << sep_angle << " deg\n";
            } else {
                std::cout << "  Separation angle: not detected (flow may not be developed)\n";
            }

            // Append to summary
            std::ofstream qf(config.qoi_output_dir + "/qoi_summary.dat",
                             std::ios::app);
            if (qf) qf << "sep_angle " << sep_angle << "\n";
        }

        // Wake velocity profiles at x/D = 1, 2, 3, 5 downstream of body center
        double wake_stations[] = {1.0, 2.0, 3.0, 5.0};
        for (double xD : wake_stations) {
            double x_station = body_cx + xD * D;
            if (x_station >= config.x_max) continue;

            std::string fname = config.qoi_output_dir + "/wake_xD"
                                + std::to_string(static_cast<int>(xD * 10)) + ".dat";
            std::string header = "y U V at x/D=" + std::to_string(xD)
                                 + " downstream";
            qoi::extract_wake_profile(
                vel.u_data().data(), vel.v_data().data(),
                vel.u_stride(), vel.v_stride(),
                is3D ? vel.u_plane_stride() : 0,
                is3D ? vel.v_plane_stride() : 0,
                Nx, Ny, Nz, Ng,
                x_station, mesh.xc.data(), mesh.yc.data(),
                fname, header);
        }

        std::cout << "QoI written to " << config.qoi_output_dir << "/\n";
    }

    if (mpi_rank == 0) {
        std::cout << "\n=== Simulation complete ===\n";
        std::cout << "Re = " << Re << "\n";
        TimingStats::instance().print_summary();
    }

#ifdef USE_MPI
    MPI_Finalize();
#endif

    return 0;
}
