/// @file test_mpi_ibm_3d.cpp
/// @brief Physics validation: IBM cylinder in 3D with MPI z-slab decomposition
///
/// Infinite cylinder (periodic in z) at Re=100. MPI splits the z-direction.
/// IBM forcing must produce nonzero drag at any rank count.
/// Cd in [0.3, 4.0] — wide tolerance for coarse-grid IBM.
///
/// Non-MPI build: single-process, Nz=8 periodic in z.
/// MPI build: Nz slabs split across nprocs, tested at 1/2/4 ranks by CI.

#include "ibm_forcing.hpp"
#include "ibm_geometry.hpp"
#include "solver.hpp"
#include "config.hpp"
#include "mesh.hpp"
#include "decomposition.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <memory>

#ifdef USE_MPI
#include <mpi.h>
#endif

using namespace nncfd;

int main(int argc, char** argv) {
#ifdef USE_MPI
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        std::cerr << "[MPI] MPI_Init failed\n";
        return 1;
    }
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
#else
    (void)argc; (void)argv;
    int rank = 0, nprocs = 1;
#endif

    const double D      = 1.0;
    const double U_inf  = 1.0;
    const double Re     = 100.0;
    const double nu     = U_inf * D / Re;   // 0.01
    const double radius = D / 2.0;

    // Domain: [0,20] x [-6,6] x [0,pi]
    const double Lx = 20.0, Lz = M_PI;
    const int    Nx = 64, Ny = 48, Nz = 8;

    // Cylinder center at (4,0), extruded through all z-slabs
    const double cx = 4.0, cy = 0.0;
    const double dt = 0.005;
    const int nsteps = 1000, avg_start = 800;

    try {
        Mesh mesh;
        mesh.init_uniform(Nx, Ny, Nz, 0.0, Lx, -6.0, 6.0, 0.0, Lz);

        Config config;
        config.nu          = nu;
        config.dt          = dt;
        config.adaptive_dt = false;
        config.turb_model  = TurbulenceModelType::None;
        config.verbose     = false;

#ifdef USE_MPI
        Decomposition decomp(MPI_COMM_WORLD, Nz);
#else
        Decomposition decomp(Nz);
#endif

        RANSSolver solver(mesh, config);
        solver.set_decomposition(&decomp);

        VelocityBC bc;
        bc.x_lo = VelocityBC::Periodic; bc.x_hi = VelocityBC::Periodic;
        bc.y_lo = VelocityBC::Periodic; bc.y_hi = VelocityBC::Periodic;
        bc.z_lo = VelocityBC::Periodic; bc.z_hi = VelocityBC::Periodic;
        solver.set_velocity_bc(bc);
        solver.initialize_uniform(U_inf, 0.0);

        auto body = std::make_shared<CylinderBody>(cx, cy, radius);
        IBMForcing ibm(mesh, body);
        solver.set_ibm_forcing(&ibm);

        if (rank == 0) {
            std::cout << "=== IBM Cylinder 3D + MPI (Re=" << Re
                      << ", nprocs=" << nprocs << ") ===" << std::endl;
            std::cout << "  Grid: " << Nx << "x" << Ny << "x" << Nz
                      << "  IBM: " << ibm.num_forcing_cells() << " forcing, "
                      << ibm.num_solid_cells() << " solid cells" << std::endl;
        }

        if (ibm.num_forcing_cells() == 0)
            throw std::runtime_error("IBM has no forcing cells");
        if (ibm.num_solid_cells() == 0)
            throw std::runtime_error("IBM has no solid cells");

        const double q_inf = 0.5 * U_inf * U_inf;
        double sum_Cd = 0.0, sum_Cl = 0.0;
        int n_avg = 0;

        for (int step = 1; step <= nsteps; ++step) {
            double res = solver.step();
            if (!std::isfinite(res))
                throw std::runtime_error("Solver diverged at step " + std::to_string(step));

            if (step > avg_start) {
                solver.sync_from_gpu();
                auto [Fx, Fy, Fz] = ibm.compute_forces(solver.velocity(), dt);
                // MPI: each rank holds its z-slab contribution; sum across all ranks
                double Fx_global = decomp.allreduce_sum(Fx);
                double Fy_global = decomp.allreduce_sum(Fy);
                // Normalize by span to get per-unit-span drag, then compute Cd
                double Cd = (Fx_global / Lz) / (q_inf * D);
                double Cl = (Fy_global / Lz) / (q_inf * D);
                sum_Cd += Cd;
                sum_Cl += Cl;
                ++n_avg;
            }

            if (step % 200 == 0 && rank == 0) {
                std::cout << "  Step " << step << ": res=" << res << std::endl;
            }
        }

        if (n_avg == 0)
            throw std::runtime_error("No averaging steps collected");

        const double Cd_mean = sum_Cd / n_avg;
        const double Cl_mean = sum_Cl / n_avg;

        if (rank == 0) {
            std::cout << "  Cd_mean = " << Cd_mean << "  (ref: ~1.35 at Re=100)\n";
            std::cout << "  Cl_mean = " << Cl_mean << "  (expected ~0 on average)\n";
        }

        if (Cd_mean < 0.3 || Cd_mean > 4.0)
            throw std::runtime_error("Cd=" + std::to_string(Cd_mean) + " out of [0.3, 4.0]");
        if (std::abs(Cl_mean) > 1.0)
            throw std::runtime_error("|Cl|=" + std::to_string(std::abs(Cl_mean)) + " > 1.0");

        if (rank == 0)
            std::cout << "PASS: IBM cylinder 3D + MPI (nprocs=" << nprocs
                      << ", Cd=" << Cd_mean << ")" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[rank " << rank << "] FAIL: " << e.what() << std::endl;
#ifdef USE_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        return 1;
    }

#ifdef USE_MPI
    if (rank == 0)
        std::cout << "\nAll MPI IBM 3D tests PASSED" << std::endl;
    MPI_Finalize();
#else
    std::cout << "\nAll MPI IBM 3D tests PASSED (single-process)" << std::endl;
#endif
    return 0;
}
