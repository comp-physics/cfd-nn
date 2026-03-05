/// NACA airfoil flow solver with immersed boundary method
/// Solves incompressible Navier-Stokes around a NACA 4-digit airfoil using
/// direct-forcing IBM. Outputs drag and lift coefficients.

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
    MPI_Init(&argc, &argv);
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
#else
    int mpi_rank = 0;
#endif

    if (mpi_rank == 0) {
        std::cout << "=== NACA Airfoil Flow Solver (IBM) ===\n\n";
    }

    // Parse configuration
    Config config;

    // Default airfoil flow settings
    config.Nx = 256;
    config.Ny = 128;
    config.Nz = 1;
    config.x_min = -5.0;
    config.x_max = 15.0;
    config.y_min = -5.0;
    config.y_max = 5.0;
    config.z_min = 0.0;
    config.z_max = M_PI;

    config.nu = 0.001;
    config.dp_dx = 0.0;

    config.dt = 0.0005;
    config.max_steps = 20000;
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

    // Airfoil parameters
    double airfoil_x = 0.0;     // Leading edge x
    double airfoil_y = 0.0;     // Leading edge y
    double chord = 1.0;         // Chord length
    double aoa_deg = 0.0;       // Angle of attack (degrees)
    std::string naca_code = "0012";
    double U_inf = 1.0;

    double aoa_rad = aoa_deg * M_PI / 180.0;
    double Re = U_inf * chord / config.nu;

    if (mpi_rank == 0) {
        config.print();
        std::cout << "\nAirfoil: NACA " << naca_code
                  << ", chord=" << chord
                  << ", AoA=" << aoa_deg << " deg\n";
        std::cout << "Leading edge at (" << airfoil_x << ", " << airfoil_y << ")\n";
        std::cout << "Re (based on chord) = " << Re << "\n";
        std::cout << "U_inf = " << U_inf << "\n\n";
    }

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

#ifdef USE_MPI
    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    Decomposition decomp(MPI_COMM_WORLD, config.Nz);
#else
    Decomposition decomp(config.Nz);
#endif

    // Create IBM body
    auto body = std::make_shared<NACABody>(airfoil_x, airfoil_y, chord, aoa_rad, naca_code);
    IBMForcing ibm(mesh, body);

    // Create solver
    RANSSolver solver(mesh, config);
    solver.set_decomposition(&decomp);
    solver.set_ibm_forcing(&ibm);

    // Boundary conditions
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

    ScopedTimer total_timer("Total simulation", false);

    double rho = 1.0;
    double A_ref = chord * (is3D ? (config.z_max - config.z_min) : 1.0);
    double q_inf = 0.5 * rho * U_inf * U_inf;

    for (int step = 1; step <= config.max_steps; ++step) {
        if (config.adaptive_dt) {
            solver.set_dt(solver.compute_adaptive_dt());
        }
        double residual = solver.step();

        // Must sync velocity from GPU since compute_forces reads host memory
#ifdef USE_GPU_OFFLOAD
        solver.sync_solution_from_gpu();
#endif
        auto [Fx, Fy, Fz] = ibm.compute_forces(solver.velocity(), solver.current_dt());

        // Rotate forces to lift/drag coordinates if AoA != 0
        double Fd = Fx * std::cos(aoa_rad) + Fy * std::sin(aoa_rad);
        double Fl = -Fx * std::sin(aoa_rad) + Fy * std::cos(aoa_rad);

        double Cd = Fd / (q_inf * A_ref);
        double Cl = Fl / (q_inf * A_ref);

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
        std::cout << "NACA " << naca_code << " at Re = " << Re
                  << ", AoA = " << aoa_deg << " deg\n";
        TimingStats::instance().print_summary();
    }

#ifdef USE_MPI
    MPI_Finalize();
#endif

    return 0;
}
