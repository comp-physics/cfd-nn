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

    // Body force from pressure gradient
    solver.set_body_force(-config.dp_dx, 0.0);

    solver.print_solver_info();

    // Initialize with quiescent flow (body force will drive it)
    solver.initialize_uniform(0.0, 0.0);

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

    // Time stepping loop
    ScopedTimer total_timer("Total simulation", false);

    for (int step = 1; step <= config.max_steps; ++step) {
        if (config.adaptive_dt) {
            solver.set_dt(solver.compute_adaptive_dt());
        }
        double residual = solver.step();

        // Compute forces on the body
        // Must sync velocity from GPU since compute_forces reads host memory
#ifdef USE_GPU_OFFLOAD
        solver.sync_solution_from_gpu();
#endif
        auto [Fx, Fy, Fz] = ibm.compute_forces(solver.velocity(), solver.current_dt());

        double time = solver.current_time();
        double bulk_u = solver.bulk_velocity();

        if (mpi_rank == 0) {
            if (force_file.is_open()) {
                force_file << step << " " << time << " "
                           << Fx << " " << Fy << " "
                           << residual << " " << bulk_u << "\n";
                if (step % config.output_freq == 0) force_file.flush();
            }

            if (step % config.output_freq == 0 || step == 1) {
                std::cout << "Step " << std::setw(6) << step
                          << "  t=" << std::fixed << std::setprecision(4) << time
                          << "  res=" << std::scientific << std::setprecision(3) << residual
                          << "  Fx=" << std::fixed << std::setprecision(4) << Fx
                          << "  U_b=" << std::setprecision(4) << bulk_u
                          << "\n" << std::flush;
            }
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
