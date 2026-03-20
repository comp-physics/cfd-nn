/// @file bench_mpi_poisson.cpp
/// @brief Benchmark distributed FFT_MPI vs MG Schwarz Poisson solver
///
/// Usage: srun -n <nprocs> --gres=gpu:<nprocs> ./bench_mpi_poisson [Nx] [Ny] [Nz_global] [nsolves]
/// Default: 64 64 64 20

#include "solver.hpp"
#include "config.hpp"
#include "mesh.hpp"
#include "decomposition.hpp"
#include "halo_exchange.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <iomanip>

#ifdef USE_MPI
#include <mpi.h>
#endif
#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

using namespace nncfd;
using namespace std::chrono;

/// Benchmark a Poisson solver type for nsolves full solver steps
/// Returns average wall time per step in ms
double benchmark_poisson(const Mesh& mesh, Decomposition& decomp,
                         PoissonSolverType solver_type, int nsolves,
                         int rank) {
    Config config;
    config.nu = 1.0;
    config.dt = 5e-4;
    config.adaptive_dt = false;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;
    config.poisson_solver = solver_type;
    if (solver_type == PoissonSolverType::MG) {
        config.poisson_fixed_cycles = 16;  // More cycles for Schwarz convergence
    }

    RANSSolver solver(mesh, config);
    solver.set_decomposition(&decomp);

    // Channel BCs: periodic x/z, no-slip y
    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    bc.z_lo = VelocityBC::Periodic;
    bc.z_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    // Initialize with gentle Poiseuille (high viscosity → low velocity)
    solver.set_body_force(1.0, 0.0, 0.0);
    double Ly = mesh.yf[mesh.Ny + mesh.Nghost] - mesh.yf[mesh.Nghost];
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                double y = mesh.y(j);
                // u_max = dp_dx * Ly^2 / (8*nu) = 1*4/(8*1) = 0.5
                solver.velocity().u(i, j, k) = 0.5 * y * (Ly - y) / (Ly * Ly / 4.0);
            }
        }
    }

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    // Warmup (2 steps)
    for (int w = 0; w < 2; ++w) {
        solver.step();
    }

#ifdef USE_MPI
    MPI_Barrier(decomp.comm());
#endif

    // Timed region
    auto start = high_resolution_clock::now();
    for (int s = 0; s < nsolves; ++s) {
        solver.step();
    }
#ifdef USE_MPI
    MPI_Barrier(decomp.comm());
#endif
    auto end = high_resolution_clock::now();

    double total_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
    return total_ms / nsolves;
}

int main(int argc, char** argv) {
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
#else
    (void)argc; (void)argv;
    int rank = 0, nprocs = 1;
#endif

#ifdef USE_GPU_OFFLOAD
    {
        int num_devices = omp_get_num_devices();
        if (num_devices > 0 && nprocs > 1) {
            omp_set_default_device(rank % num_devices);
        }
    }
#endif

    int Nx = 64, Ny = 64, Nz_global = 64, nsolves = 20;
    if (argc > 1) Nx = std::atoi(argv[1]);
    if (argc > 2) Ny = std::atoi(argv[2]);
    if (argc > 3) Nz_global = std::atoi(argv[3]);
    if (argc > 4) nsolves = std::atoi(argv[4]);

#ifdef USE_MPI
    Decomposition decomp(MPI_COMM_WORLD, Nz_global);
#else
    Decomposition decomp(Nz_global);
#endif
    int Nz_local = decomp.nz_local();

    double dz = (2.0 * M_PI) / Nz_global;
    double z_lo = decomp.k_global_start() * dz;
    double z_hi = z_lo + Nz_local * dz;

    Mesh mesh;
    mesh.init_uniform(Nx, Ny, Nz_local, 0.0, 2.0*M_PI, 0.0, 2.0, z_lo, z_hi);

    if (rank == 0) {
        std::cout << "=== MPI Poisson Solver Benchmark ===" << std::endl;
        std::cout << "Grid: " << Nx << "x" << Ny << "x" << Nz_global
                  << " (" << nprocs << " ranks, " << Nz_local << " z-cells/rank)"
                  << std::endl;
        std::cout << "Solves: " << nsolves << " steps (+ 2 warmup)" << std::endl;
        std::cout << std::endl;
    }

    // Benchmark MG (Schwarz with halo exchange)
    double mg_ms = benchmark_poisson(mesh, decomp, PoissonSolverType::MG, nsolves, rank);

    if (rank == 0) {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "MG Schwarz:  " << mg_ms << " ms/step" << std::endl;
    }

    // Benchmark FFT_MPI
#if defined(USE_FFT_POISSON) && defined(USE_MPI)
    double fft_ms = benchmark_poisson(mesh, decomp, PoissonSolverType::FFT_MPI, nsolves, rank);

    if (rank == 0) {
        std::cout << "FFT_MPI:     " << fft_ms << " ms/step" << std::endl;
        std::cout << std::endl;
        double speedup = mg_ms / fft_ms;
        std::cout << "Speedup (FFT_MPI / MG): " << std::setprecision(1) << speedup << "x" << std::endl;
    }
#else
    if (rank == 0) {
        std::cout << "FFT_MPI:     [not available — build with USE_FFT_POISSON + USE_MPI]" << std::endl;
    }
#endif

#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}
