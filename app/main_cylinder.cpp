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
        std::cout << "=== Cylinder Flow Solver (IBM) ===\n\n";
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

    // IBM parameters from config or defaults
    double cyl_x = 10.0;   // Cylinder center x
    double cyl_y = 0.0;    // Cylinder center y
    double cyl_r = 0.5;    // Cylinder radius
    double U_inf = 1.0;    // Free-stream velocity

    // Compute Re based on diameter
    double Re = U_inf * (2.0 * cyl_r) / config.nu;

    if (mpi_rank == 0) {
        config.print();
        std::cout << "\nCylinder: center=(" << cyl_x << ", " << cyl_y
                  << "), radius=" << cyl_r << "\n";
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

    // Create IBM body
    auto body = std::make_shared<CylinderBody>(cyl_x, cyl_y, cyl_r);
    IBMForcing ibm(mesh, body);

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

    // No body force (flow driven by initial condition / IBM interaction)
    solver.set_body_force(0.0, 0.0);

    solver.print_solver_info();

    // Initialize with uniform flow
    solver.initialize_uniform(U_inf, 0.0);

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

    // Time stepping loop
    ScopedTimer total_timer("Total simulation", false);

    double rho = 1.0;  // Density (incompressible)
    double A_ref = 2.0 * cyl_r * (is3D ? (config.z_max - config.z_min) : 1.0);
    double q_inf = 0.5 * rho * U_inf * U_inf;

    for (int step = 1; step <= config.max_steps; ++step) {
        if (config.adaptive_dt) {
            solver.set_dt(solver.compute_adaptive_dt());
        }
        double residual = solver.step();

        // Compute forces on the body (IBM forcing applied inside step())
        // Must sync velocity from GPU since compute_forces reads host memory
#ifdef USE_GPU_OFFLOAD
        solver.sync_solution_from_gpu();
#endif
        auto [Fx, Fy, Fz] = ibm.compute_forces(solver.velocity(), solver.current_dt());

        double Cd = Fx / (q_inf * A_ref);
        double Cl = Fy / (q_inf * A_ref);

        double time = solver.current_time();

        if (mpi_rank == 0) {
            if (force_file.is_open()) {
                force_file << step << " " << time << " "
                           << Fx << " " << Fy << " "
                           << Cd << " " << Cl << "\n";
                if (step % config.output_freq == 0) force_file.flush();
            }

            if (step % config.output_freq == 0 || step == 1) {
                std::cout << "Step " << std::setw(6) << step
                          << "  t=" << std::fixed << std::setprecision(4) << time
                          << "  res=" << std::scientific << std::setprecision(3) << residual
                          << "  Cd=" << std::fixed << std::setprecision(4) << Cd
                          << "  Cl=" << std::setprecision(4) << Cl
                          << "\n" << std::flush;
            }
        }

        if (std::isnan(residual) || std::isinf(residual)) {
            std::cerr << "ERROR: Solver diverged at step " << step << "\n";
            break;
        }
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
