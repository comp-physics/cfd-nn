/// Periodic hills solver with immersed boundary method
/// Solves incompressible Navier-Stokes over periodic hills (Breuer et al. 2009)
/// using direct-forcing IBM. Outputs forces, residual, and bulk velocity.
///
/// Domain: [0, 9h] x [0, 3.035h] x [0, 1]
/// Hill height h = 1.0, flow driven by body force (dp/dx)

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "config.hpp"
#include "timing.hpp"
#include "ibm_geometry.hpp"
#include "ibm_forcing.hpp"
#include "decomposition.hpp"
#include "turbulence_model.hpp"
#include "turbulence_earsm.hpp"
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
        std::cout << "=== Periodic Hills Solver (IBM) ===\n\n";
    }

    // Parse configuration
    Config config;

    // Hill height
    double h = 1.0;

    // Default periodic hills settings
    config.Nx = 192;
    config.Ny = 96;
    config.Nz = 1;
    config.x_min = 0.0;
    config.x_max = 9.0 * h;
    config.y_min = 0.0;
    config.y_max = 3.035 * h;
    config.z_min = 0.0;
    config.z_max = 1.0;

    config.nu = 9.438e-5;  // Re_h = 10595
    config.dp_dx = -1.0;

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

    // Compute Re based on hill height
    double Re = 1.0 * h / config.nu;

    if (mpi_rank == 0) {
        config.print();
        std::cout << "\nPeriodic hills: h=" << h << "\n";
        std::cout << "Re (based on hill height) = " << Re << "\n";
        std::cout << "dp/dx = " << config.dp_dx << "\n\n";
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

    // Create IBM body
    auto body = std::make_shared<PeriodicHillBody>(h);
    IBMForcing ibm(mesh, body);
    // Use penalization (ibm_eta) for warm-up stability, then switch to ghost-cell
    // for evaluation accuracy. Ghost-cell is enabled after warm-up completes.
    if (config.ibm_eta > 0.0) ibm.set_penalization_eta(config.ibm_eta);

    // Create solver
    RANSSolver solver(mesh, config);
    solver.set_decomposition(&decomp);
    solver.set_ibm_forcing(&ibm);

    // Boundary conditions: Periodic (x), NoSlip (y), Periodic (z if 3D)
    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    if (is3D) {
        bc.z_lo = VelocityBC::Periodic;
        bc.z_hi = VelocityBC::Periodic;
    }
    solver.set_velocity_bc(bc);

    // Body force: use bulk velocity controller if target specified,
    // otherwise fixed dp_dx (standard for periodic hills)
    if (config.bulk_velocity_target > 0.0) {
        solver.set_body_force(0.0, 0.0);
        solver.enable_bulk_velocity_control(config.bulk_velocity_target);
    } else {
        solver.set_body_force(-config.dp_dx, 0.0);
        // No force ramp: penalization at ibm_eta handles the transient.
        // Force ramp (starting at zero body force) caused warm-up divergence
        // because it combined with penalization to give zero driving force.
    }

    // Set initial turbulence model. For models WITH transport (SST, EARSM, k-omega),
    // initialize with the TARGET model directly — it can bootstrap k/omega from cold start.
    // For models WITHOUT transport (NN, algebraic, GEP), initialize with SST for warm-up
    // so k/omega fields get properly developed before switching.
    if (config.turb_model != TurbulenceModelType::None) {
        auto target_probe = create_turbulence_model(config.turb_model, "", "");
        bool target_has_transport = target_probe && target_probe->uses_transport_equations();

        if (target_has_transport) {
            // Transport models: initialize with target directly
            auto turb_model = create_turbulence_model(config.turb_model,
                                                      config.nn_weights_path,
                                                      config.nn_scaling_path,
                                                      config.pope_C1,
                                                      config.pope_C2);
            if (turb_model) {
                turb_model->set_nu(config.nu);
                solver.set_turbulence_model(std::move(turb_model));
            }
        } else if (!config.warmup_model.empty() && config.warmup_time > 0.0) {
            // Non-transport models: initialize with warm-up model (SST)
            TurbulenceModelType warmup_type = TurbulenceModelType::SSTKOmega;
            if (config.warmup_model == "komega") warmup_type = TurbulenceModelType::KOmega;
            auto warmup_turb = create_turbulence_model(warmup_type, "", "");
            if (warmup_turb) {
                warmup_turb->set_nu(config.nu);
                solver.set_turbulence_model(std::move(warmup_turb));
            }
        } else {
            // No warm-up: initialize with target directly
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

    // Initialize from rest — body force will accelerate the flow.
    // Starting from zero avoids huge pressure transients when the IBM
    // penalization damps the velocity inside the hill body.
    double U_ref = (config.bulk_velocity_target > 0.0) ? config.bulk_velocity_target : 1.0;
    solver.initialize_uniform(0.0, 0.0);

    if (config.perturbation_amplitude > 0.0) {
        const double amp = config.perturbation_amplitude;
        std::srand(42);
        auto& vel = solver.velocity();
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                    double r = 2.0 * static_cast<double>(std::rand()) / RAND_MAX - 1.0;
                    if (is3D) vel.u(i, j, k) += amp * U_ref * r;
                    else      vel.u(i, j)     += amp * U_ref * r;
                }
            }
        }
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    double r = 2.0 * static_cast<double>(std::rand()) / RAND_MAX - 1.0;
                    if (is3D) vel.v(i, j, k) += amp * U_ref * r;
                    else      vel.v(i, j)     += amp * U_ref * r;
                }
            }
        }
        if (mpi_rank == 0) {
            std::cout << "Added random perturbations (amplitude=" << amp << ")\n";
        }
    }

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    // Open force output file
    std::ofstream force_file;
    if (mpi_rank == 0) {
        force_file.open(config.output_dir + "forces.dat");
        if (!force_file.is_open()) {
            std::cerr << "Warning: Could not open " << config.output_dir << "forces.dat\n";
        } else {
            force_file << "# step  time  Fx  Fy  residual  bulk_u\n";
        }
    }

    // VTK snapshot setup
    const std::string vtk_prefix = config.write_fields ?
        (config.output_dir + "hills") : "";
    const int snapshot_freq = (config.num_snapshots > 0 && config.write_fields) ?
        std::max(1, config.max_steps / config.num_snapshots) : 0;
    int snap_count = 0;

    // Warm-up phase: run steps to develop the flow.
    // For EARSM models: disable the algebraic closure during warm-up (run as pure SST).
    // This prevents the EARSM anisotropy from destabilizing the cold-start flow.
    if (!config.warmup_model.empty() && config.warmup_time > 0.0) {
        // Disable EARSM closure during warm-up (if applicable)
        auto* earsm = dynamic_cast<SSTWithEARSM*>(solver.turb_model_ptr());
        if (earsm) {
            earsm->set_closure_active(false);
        }
        {
            if (mpi_rank == 0) {
                std::cout << "=== Warm-up phase: " << config.warmup_model
                          << " for t=" << config.warmup_time << " ===\n";
            }

            // Warm-up has its own step budget (not shared with max_steps)
            const int warmup_max_steps = 500000;
            for (int ws = 1; ws <= warmup_max_steps; ++ws) {
                if (config.adaptive_dt) {
                    solver.set_dt(solver.compute_adaptive_dt());
                }
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

            // Switch to target model after warm-up.
            // Transport models (SST, EARSM, k-omega) were already initialized
            // at the start and developed during warm-up — skip re-creation.
            // Non-transport models (NN, algebraic, GEP) need fresh creation
            // with background SST to keep k/omega alive.
            {
                auto probe = create_turbulence_model(config.turb_model, "", "");
                bool target_has_transport = probe && probe->uses_transport_equations();

                if (!target_has_transport && config.turb_model != TurbulenceModelType::None) {
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

                        // Ramp tau_div from 0→1 over 5000 steps to prevent divergence
                        // from sudden anisotropic correction on SST-developed flow.
                        solver.start_tau_div_ramp(5000);
                        if (mpi_rank == 0)
                            std::cout << "  tau_div ramp: 0→1 over 5000 steps\n";
                    }
                }
                // else: transport models keep running from warm-up (no switch needed)
            }

            TimingStats::instance().reset();
            if (mpi_rank == 0) {
                std::cout << "=== Evaluation phase from t="
                          << solver.current_time() << " ===\n";
            }

            // Re-enable EARSM closure (was disabled during warm-up)
            if (earsm) {
                earsm->set_closure_active(true);
                if (mpi_rank == 0) {
                    std::cout << "  EARSM closure re-activated\n";
                }
            }

            // Keep penalization IBM throughout (ghost-cell causes divergence
            // at the switch point because the penalized flow has nonzero velocity
            // inside the body which the ghost-cell correction amplifies)
            // ibm.set_ghost_cell_ibm(true);
            // ibm.set_penalization_eta(0.0);
            // ibm.recompute_and_remap();
        }
    }

    // Time stepping loop
    ScopedTimer total_timer("Total simulation", false);

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
            double bulk_u = solver.bulk_velocity();

            if (mpi_rank == 0) {
                if (force_file.is_open()) {
                    force_file << step << " " << time << " "
                               << Fx << " " << Fy << " "
                               << residual << " " << bulk_u << "\n";
                    force_file.flush();
                }

                std::cout << "Step " << std::setw(6) << step
                          << "  t=" << std::fixed << std::setprecision(4) << time
                          << "  res=" << std::scientific << std::setprecision(3) << residual
                          << "  Fx=" << std::fixed << std::setprecision(4) << Fx
                          << "  U_b=" << std::setprecision(4) << bulk_u
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

        if (std::isnan(residual) || std::isinf(residual)) {
            std::cerr << "ERROR: Solver diverged at step " << step << "\n";
            break;
        }

        if (residual < config.tol && step > 100) {
            if (mpi_rank == 0) {
                std::cout << "Converged at step " << step
                          << " (residual=" << std::scientific << residual << ")\n";
            }
            break;
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

    // QoI extraction: Cf(x), velocity profiles at x/H stations
    if (config.qoi_freq > 0 && mpi_rank == 0) {
        std::filesystem::create_directories(config.qoi_output_dir);

        const auto& vel = solver.velocity();
        const int Nx = mesh.Nx, Ny = mesh.Ny, Ng = mesh.Nghost;

        // Pre-compute hill geometry: y_hill[i] and first fluid cell index
        std::vector<double> hill_y(Nx);
        std::vector<int> j_first_fluid(Nx);
        for (int i = 0; i < Nx; ++i) {
            double xc_i = mesh.xc[i + Ng];
            hill_y[i] = 0.0;
            // Walk upward from bottom to find first fluid cell
            for (int j = Ng; j < Ny + Ng; ++j) {
                double yc_j = mesh.yc[j];
                if (body->phi(xc_i, yc_j, 0.0) > 0.0) {
                    j_first_fluid[i] = j;
                    hill_y[i] = yc_j - body->phi(xc_i, yc_j, 0.0);  // Surface y
                    break;
                }
            }
        }

        // Skin friction Cf(x)
        double u_ref = 1.0;  // Reference velocity
        std::vector<double> cf(Nx);
        qoi::compute_cf_x_device(
            vel.u_data().data(), vel.u_stride(), 0,
            mesh.yf.data(), Ny + 2 * Ng + 1,
            hill_y.data(), j_first_fluid.data(),
            config.nu, u_ref, mesh.dx, cf.data(),
            Nx, Ny, Ng);

        // Write Cf(x)
        std::vector<double> xc_arr(Nx);
        for (int i = 0; i < Nx; ++i) xc_arr[i] = mesh.xc[i + Ng];
        qoi::write_profile(config.qoi_output_dir + "/cf_x.dat",
                           xc_arr.data(), cf.data(), Nx, "x Cf");

        // Separation/reattachment
        auto [x_sep, x_reattach] = qoi::find_separation_reattachment(
            cf.data(), xc_arr.data(), Nx);
        std::cout << "  Separation:    x/H = " << x_sep / h << "\n";
        std::cout << "  Reattachment:  x/H = " << x_reattach / h << "\n";

        // Velocity profiles at x/H = 0.5, 2, 4, 6, 8
        std::vector<double> yc_arr(Ny);
        for (int j = 0; j < Ny; ++j) yc_arr[j] = mesh.yc[j + Ng];

        double stations[] = {0.5, 2.0, 4.0, 6.0, 8.0};
        for (double xH : stations) {
            double x_target = xH * h;
            // Find closest grid index
            int i_station = 0;
            double min_dist = 1e30;
            for (int i = 0; i < Nx; ++i) {
                double d = std::abs(mesh.xc[i + Ng] - x_target);
                if (d < min_dist) { min_dist = d; i_station = i; }
            }

            std::vector<double> u_prof(Ny), v_prof(Ny);
            qoi::extract_velocity_profile_device(
                vel.u_data().data(), vel.v_data().data(),
                vel.u_stride(), vel.v_stride(), 0, 0,
                i_station, Nx, Ny, 1, Ng,
                u_prof.data(), v_prof.data());

            std::string fname = config.qoi_output_dir + "/profile_xH"
                                + std::to_string((int)(xH * 10)) + ".dat";
            qoi::write_profile_uv(fname, yc_arr.data(),
                                  u_prof.data(), v_prof.data(), Ny,
                                  "y U V at x/H=" + std::to_string(xH));
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
