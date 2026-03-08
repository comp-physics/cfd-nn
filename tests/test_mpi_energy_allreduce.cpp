/// @file test_mpi_energy_allreduce.cpp
/// @brief Physics validation: MPI allreduce_sum gives correct global kinetic energy
///
/// TGV initial condition: E_analytical = 0.125 (per unit volume).
/// Each rank computes its local KE contribution, allreduce_sum gives global KE.
/// Must match analytical value within 1% — tests MPI energy conservation.

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

/// Initialize 3D TGV velocity field on host
static void init_tgv_3d(RANSSolver& solver, const Mesh& mesh) {
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                double x = mesh.xf[i];
                double y = mesh.y(j);
                double z = mesh.z(k);
                solver.velocity().u(i, j, k) = std::sin(x) * std::cos(y) * std::cos(z);
            }
        }
    }
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double x = mesh.x(i);
                double y = mesh.yf[j];
                double z = mesh.z(k);
                solver.velocity().v(i, j, k) = -std::cos(x) * std::sin(y) * std::cos(z);
            }
        }
    }
    for (int k = mesh.k_begin(); k <= mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                solver.velocity().w(i, j, k) = 0.0;
            }
        }
    }
}

/// Compute local kinetic energy sum and cell count for this rank's z-slabs
static void compute_local_ke(const RANSSolver& solver, const Mesh& mesh,
                              double& local_ke, double& local_count) {
    local_ke = 0.0;
    local_count = 0.0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double uc = 0.5 * (solver.velocity().u(i, j, k) + solver.velocity().u(i+1, j, k));
                double vc = 0.5 * (solver.velocity().v(i, j, k) + solver.velocity().v(i, j+1, k));
                double wc = 0.5 * (solver.velocity().w(i, j, k) + solver.velocity().w(i, j, k+1));
                local_ke += 0.5 * (uc*uc + vc*vc + wc*wc);
                local_count += 1.0;
            }
        }
    }
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

    const int N = 16;
    const int Nz = N;
    const double L = 2.0 * M_PI;
    const double nu = 1e-3;
    const double dt = 5e-3;
    const int nsteps = 20;
    // TGV analytical KE per cell: <u^2+v^2+w^2>/2
    // u=sin(x)cos(y)cos(z), v=-cos(x)sin(y)cos(z), w=0
    // <u^2> = <v^2> = 1/8, <w^2>=0 → KE = 0.125
    const double E_analytical = 0.125;

    try {
        Mesh mesh;
        mesh.init_uniform(N, N, Nz, 0.0, L, 0.0, L, 0.0, L);

        Config config;
        config.nu = nu;
        config.dt = dt;
        config.adaptive_dt = false;
        config.turb_model = TurbulenceModelType::None;
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
        bc.y_lo = VelocityBC::Periodic;
        bc.y_hi = VelocityBC::Periodic;
        bc.z_lo = VelocityBC::Periodic;
        bc.z_hi = VelocityBC::Periodic;
        solver.set_velocity_bc(bc);

        // Initialize TGV on host, compute KE before GPU sync
        init_tgv_3d(solver, mesh);

        double local_ke = 0.0, local_count = 0.0;
        compute_local_ke(solver, mesh, local_ke, local_count);

        double global_ke_sum = decomp.allreduce_sum(local_ke);
        double global_count  = decomp.allreduce_sum(local_count);

        if (global_count <= 0.0) {
            if (rank == 0) std::cerr << "FAIL: global cell count is zero" << std::endl;
#ifdef USE_MPI
            MPI_Abort(MPI_COMM_WORLD, 1);
#endif
            return 1;
        }

        double global_ke = global_ke_sum / global_count;
        double err_initial = std::abs(global_ke - E_analytical) / std::abs(E_analytical);

        if (rank == 0) {
            std::cerr << "[test_mpi_energy_allreduce] nprocs=" << nprocs
                      << "  global_ke=" << global_ke
                      << "  E_analytical=" << E_analytical
                      << "  err=" << err_initial << std::endl;
        }

        if (err_initial >= 0.01) {
            if (rank == 0) {
                std::cerr << "FAIL: Initial KE error exceeds 1% (err=" << err_initial
                          << ", nprocs=" << nprocs << ")" << std::endl;
            }
#ifdef USE_MPI
            MPI_Abort(MPI_COMM_WORLD, 1);
#endif
            return 1;
        }

        if (rank == 0) {
            std::cout << "PASS: Initial KE allreduce matches analytical (err=" << err_initial
                      << ", nprocs=" << nprocs << ")" << std::endl;
        }

        // Sync to GPU and run some steps
        solver.sync_to_gpu();

        for (int step = 0; step < nsteps; ++step) {
            solver.step();
        }

        solver.sync_from_gpu();

        double local_ke_after = 0.0, local_count_after = 0.0;
        compute_local_ke(solver, mesh, local_ke_after, local_count_after);

        double global_ke_sum_after = decomp.allreduce_sum(local_ke_after);
        double global_count_after  = decomp.allreduce_sum(local_count_after);
        double global_ke_after = global_ke_sum_after / global_count_after;

        if (rank == 0) {
            std::cerr << "[test_mpi_energy_allreduce] E_after=" << global_ke_after
                      << " E_initial=" << global_ke << std::endl;
        }

        // After nsteps, energy should have decreased (viscous dissipation)
        if (global_ke_after >= global_ke * 0.9999) {
            if (rank == 0) {
                std::cerr << "FAIL: Energy did not decrease after " << nsteps
                          << " steps (E_after=" << global_ke_after
                          << " E_initial=" << global_ke << ")" << std::endl;
            }
#ifdef USE_MPI
            MPI_Abort(MPI_COMM_WORLD, 1);
#endif
            return 1;
        }

        if (global_ke_after <= 0.0) {
            if (rank == 0) {
                std::cerr << "FAIL: Energy reached zero or negative (blow-up)" << std::endl;
            }
#ifdef USE_MPI
            MPI_Abort(MPI_COMM_WORLD, 1);
#endif
            return 1;
        }

        if (rank == 0) {
            std::cout << "PASS: Energy decreased after " << nsteps << " steps"
                      << " (E_initial=" << global_ke
                      << ", E_after=" << global_ke_after << ")" << std::endl;
            std::cout << "\nAll MPI energy allreduce tests PASSED (nprocs=" << nprocs << ")" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "[rank " << rank << "] EXCEPTION: " << e.what() << std::endl;
#ifdef USE_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        return 1;
    }

#ifdef USE_MPI
    MPI_Finalize();
#endif

    return 0;
}
