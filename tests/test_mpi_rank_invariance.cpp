/// @file test_mpi_rank_invariance.cpp
/// @brief Physics validation: Poiseuille bulk velocity invariant to MPI decomposition
///
/// Channel flow driven by body force fx. At convergence, Ub = fx*H^2/(3*nu) (analytical).
/// This must hold regardless of how many MPI ranks split the z-direction.
/// Non-MPI build: tests 1-rank Decomposition. MPI build: tests at runtime nprocs.

#include "solver.hpp"
#include "config.hpp"
#include "mesh.hpp"
#include "decomposition.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

#ifdef USE_MPI
#include <mpi.h>
#endif

using namespace nncfd;

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

    const int Nx = 8, Ny = 32, Nz = 8;
    const double nu = 0.1;
    const double fx = 1.0;
    const double H = 1.0;
    // Analytical Poiseuille: u(y) = fx/(2*nu)*(1-y^2), Ub = fx*H^2/(3*nu)
    const double Ub_analytic = fx * H * H / (3.0 * nu);

    try {
        Mesh mesh;
        mesh.init_uniform(Nx, Ny, Nz,
                          0.0, 2.0 * M_PI,
                          -1.0, 1.0,
                          0.0, 2.0 * M_PI);

        Config config;
        config.Nx = Nx;
        config.Ny = Ny;
        config.Nz = Nz;
        config.nu = nu;
        config.dt = 0.05;
        config.max_steps = 8000;
        config.tol = 1e-8;
        // Do not force MG: 3D MG on Nz=8 can fail to converge; let solver auto-select
        config.time_integrator = TimeIntegrator::Euler;
        config.adaptive_dt = false;
        config.verbose = false;

#ifdef USE_MPI
        Decomposition decomp(MPI_COMM_WORLD, Nz);
#else
        Decomposition decomp(Nz);
#endif

        RANSSolver solver(mesh, config);
        solver.set_decomposition(&decomp);

        VelocityBC bc;
        bc.x_lo = VelocityBC::Periodic;
        bc.x_hi = VelocityBC::Periodic;
        bc.y_lo = VelocityBC::NoSlip;
        bc.y_hi = VelocityBC::NoSlip;
        bc.z_lo = VelocityBC::Periodic;
        bc.z_hi = VelocityBC::Periodic;
        solver.set_velocity_bc(bc);
        solver.set_body_force(fx, 0.0);
        solver.initialize_uniform(0.0, 0.0);

        double residual = 1.0;
        int steps = 0;
        for (int i = 0; i < 8000 && residual > 1e-7; ++i) {
            residual = solver.step();
            steps = i + 1;
        }

        if (rank == 0) {
            std::cerr << "[test_mpi_rank_invariance] nprocs=" << nprocs
                      << " residual=" << residual
                      << " after " << steps << " steps" << std::endl;
        }

        if (residual >= 1e-5) {
            if (rank == 0) {
                std::cerr << "FAIL: Poiseuille did not converge: residual="
                          << residual << " after " << steps << " steps" << std::endl;
            }
#ifdef USE_MPI
            MPI_Abort(MPI_COMM_WORLD, 1);
#endif
            return 1;
        }

        double Ub = solver.bulk_velocity();
        double err = std::abs(Ub - Ub_analytic) / std::abs(Ub_analytic);

        if (rank == 0) {
            std::cout << "nprocs=" << nprocs
                      << "  Ub_numerical=" << Ub
                      << "  Ub_analytic=" << Ub_analytic
                      << "  rel_err=" << err << std::endl;
        }

        if (err >= 0.05) {
            if (rank == 0) {
                std::cerr << "FAIL: Bulk velocity error exceeds 5% (err=" << err
                          << ", nprocs=" << nprocs << ")" << std::endl;
            }
#ifdef USE_MPI
            MPI_Abort(MPI_COMM_WORLD, 1);
#endif
            return 1;
        }

        if (rank == 0) {
            std::cout << "PASS: Poiseuille bulk velocity invariant to MPI rank count"
                      << " (nprocs=" << nprocs << ", err=" << err << ")" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "[rank " << rank << "] EXCEPTION: " << e.what() << std::endl;
#ifdef USE_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        return 1;
    }

#ifdef USE_MPI
    if (rank == 0) {
        std::cout << "\nAll MPI rank invariance tests PASSED" << std::endl;
    }
    MPI_Finalize();
#else
    std::cout << "\nAll MPI rank invariance tests PASSED (single-process)" << std::endl;
#endif

    return 0;
}
