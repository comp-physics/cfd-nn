/// Forward-facing step solver with immersed boundary method
/// Solves incompressible Navier-Stokes over a forward-facing step using
/// direct-forcing IBM. Outputs forces and residual history.
///
/// Domain: [-10, 20] x [0, 6] x [0, 1]
/// Step at x=0, height=1
/// Inflow: uniform u = U_inf

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "config.hpp"
#include "timing.hpp"
#include "ibm_geometry.hpp"
#include "ibm_forcing.hpp"
#include "decomposition.hpp"
#include "turbulence_model.hpp"

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
        std::cout << "=== Forward-Facing Step Solver (IBM) ===\n\n";
    }

    // Parse configuration
    Config config;

    // Default step flow settings
    config.Nx = 256;
    config.Ny = 128;
    config.Nz = 1;
    config.x_min = -10.0;
    config.x_max = 20.0;
    config.y_min = 0.0;
    config.y_max = 6.0;
    config.z_min = 0.0;
    config.z_max = 1.0;

    config.nu = 0.0002;  // Re_s = 5000 based on step height s=1
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

    // Step parameters
    double step_x = 0.0;        // Step location
    double step_height = 1.0;   // Step height
    double U_inf = 1.0;         // Free-stream velocity

    // Compute Re based on step height
    double Re = U_inf * step_height / config.nu;

    if (mpi_rank == 0) {
        config.print();
        std::cout << "\nStep: x=" << step_x
                  << ", height=" << step_height << "\n";
        std::cout << "Re (based on step height) = " << Re << "\n";
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

    // Create IBM body
    auto body = std::make_shared<StepBody>(step_x, step_height);
    IBMForcing ibm(mesh, body);

    // Create solver
    RANSSolver solver(mesh, config);
    solver.set_decomposition(&decomp);
    solver.set_ibm_forcing(&ibm);

    // Boundary conditions: Inflow (x_lo), Outflow (x_hi), NoSlip (y), Periodic (z if 3D)
    VelocityBC bc;
    bc.x_lo = VelocityBC::Inflow;
    bc.x_hi = VelocityBC::Outflow;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    if (is3D) {
        bc.z_lo = VelocityBC::Periodic;
        bc.z_hi = VelocityBC::Periodic;
    }

    // Set inflow profile: uniform u = U_inf
    bc.u_inflow = [U_inf](double) { return U_inf; };
    bc.v_inflow = [](double) { return 0.0; };

    solver.set_velocity_bc(bc);

    // Body force: use bulk velocity controller if target specified,
    // otherwise no forcing (flow driven by inflow)
    if (config.bulk_velocity_target > 0.0) {
        solver.set_body_force(0.0, 0.0);
        solver.enable_bulk_velocity_control(config.bulk_velocity_target);
    } else {
        solver.set_body_force(-config.dp_dx, 0.0);
    }

    // Set turbulence model if requested
    if (config.turb_model != TurbulenceModelType::None) {
        auto turb_model = create_turbulence_model(config.turb_model,
                                                  config.nn_weights_path,
                                                  config.nn_scaling_path);
        if (turb_model) {
            turb_model->set_nu(config.nu);
            solver.set_turbulence_model(std::move(turb_model));
        }
    }

    solver.print_solver_info();

    // Initialize with uniform flow
    solver.initialize_uniform(U_inf, 0.0);

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
            force_file << "# step  time  Fx  Fy  residual\n";
        }
    }

    // VTK snapshot setup
    const std::string vtk_prefix = config.write_fields ?
        (config.output_dir + "step") : "";
    const int snapshot_freq = (config.num_snapshots > 0 && config.write_fields) ?
        std::max(1, config.max_steps / config.num_snapshots) : 0;
    int snap_count = 0;

    // Time stepping loop
    ScopedTimer total_timer("Total simulation", false);

    for (int step = 1; step <= config.max_steps; ++step) {
        if (config.adaptive_dt) {
            solver.set_dt(solver.compute_adaptive_dt());
        }

        bool need_forces = (step % config.output_freq == 0 || step == 1);
        ibm.set_accumulate_forces(need_forces);

        double residual = solver.step();

        double time = solver.current_time();

        if (need_forces) {
            auto [Fx, Fy, Fz] = ibm.compute_forces(solver.velocity(), solver.current_dt());

            if (mpi_rank == 0) {
                if (force_file.is_open()) {
                    force_file << step << " " << time << " "
                               << Fx << " " << Fy << " "
                               << residual << "\n";
                    force_file.flush();
                }

                std::cout << "Step " << std::setw(6) << step
                          << "  t=" << std::fixed << std::setprecision(4) << time
                          << "  res=" << std::scientific << std::setprecision(3) << residual
                          << "  Fx=" << std::fixed << std::setprecision(4) << Fx
                          << "  Fy=" << std::setprecision(4) << Fy
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
