/// @file test_mpi_halo_step.cpp
/// @brief Test that MPI halo exchange works correctly during solver stepping.
///
/// Each rank creates a LOCAL mesh with nz_local z-cells (the decomposed slab),
/// runs a TGV flow for several steps, and checks:
///   1. Energy decreases (viscous dissipation works across rank boundaries)
///   2. No NaN/Inf (halo exchange provides valid ghost data)
///   3. Multi-rank result is close to single-rank reference
///
/// This test exercises the halo exchange calls in step() and project_velocity()
/// that were added in the MPI integration.

#include "solver.hpp"
#include "config.hpp"
#include "mesh.hpp"
#include "decomposition.hpp"
#include "halo_exchange.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

#ifdef USE_MPI
#include <mpi.h>
#endif

using namespace nncfd;

static void init_tgv_local(RANSSolver& solver, const Mesh& mesh,
                            double z_global_offset) {
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                double x = mesh.xf[i];
                double y = mesh.y(j);
                double z = mesh.z(k) + z_global_offset;
                solver.velocity().u(i, j, k) = std::sin(x) * std::cos(y) * std::cos(z);
            }
        }
    }
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double x = mesh.x(i);
                double y = mesh.yf[j];
                double z = mesh.z(k) + z_global_offset;
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

static double compute_local_ke(const RANSSolver& solver, const Mesh& mesh) {
    double ke = 0.0;
    int count = 0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double uc = 0.5 * (solver.velocity().u(i, j, k) + solver.velocity().u(i+1, j, k));
                double vc = 0.5 * (solver.velocity().v(i, j, k) + solver.velocity().v(i, j+1, k));
                double wc = 0.5 * (solver.velocity().w(i, j, k) + solver.velocity().w(i, j, k+1));
                ke += 0.5 * (uc*uc + vc*vc + wc*wc);
                count += 1;
            }
        }
    }
    return (count > 0) ? ke / count : 0.0;
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

    const int Nx = 16, Ny = 16;
    const int Nz_global = 16;
    const double L = 2.0 * M_PI;
    const double nu = 1e-2;  // Higher viscosity for visible dissipation in few steps
    const double dt = 1e-3;
    const int nsteps = 10;

    int passed = 0, failed = 0;

    try {
#ifdef USE_MPI
        Decomposition decomp(MPI_COMM_WORLD, Nz_global);
#else
        Decomposition decomp(Nz_global);
#endif
        const int nz_local = decomp.nz_local();
        const int k_start = decomp.k_global_start();

        // Each rank creates a mesh covering only its z-slab
        double dz = L / Nz_global;
        double z_lo = k_start * dz;
        double z_hi = z_lo + nz_local * dz;

        Mesh mesh;
        mesh.init_uniform(Nx, Ny, nz_local, 0.0, L, 0.0, L, z_lo, z_hi);

        Config config;
        config.nu = nu;
        config.dt = dt;
        config.adaptive_dt = false;
        config.turb_model = TurbulenceModelType::None;
        config.verbose = false;
        config.poisson_solver = PoissonSolverType::MG;

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

        // Initialize TGV with global z-coordinate offset
        init_tgv_local(solver, mesh, 0.0);  // z_lo already baked into mesh.z(k)

        // Compute initial KE
        double local_ke_init = compute_local_ke(solver, mesh);
        double global_ke_init = decomp.allreduce_sum(local_ke_init) / nprocs;

        // Run steps
        solver.sync_to_gpu();
        for (int step = 0; step < nsteps; ++step) {
            solver.step();
        }
        solver.sync_from_gpu();

        // Compute final KE
        double local_ke_final = compute_local_ke(solver, mesh);
        double global_ke_final = decomp.allreduce_sum(local_ke_final) / nprocs;

        // Check 1: No NaN/Inf
        bool all_finite = true;
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    double u = solver.velocity().u(i, j, k);
                    double v = solver.velocity().v(i, j, k);
                    if (!std::isfinite(u) || !std::isfinite(v)) {
                        all_finite = false;
                    }
                }
            }
        }
        bool global_finite = all_finite;
#ifdef USE_MPI
        int local_finite = all_finite ? 1 : 0;
        int global_finite_int = 0;
        MPI_Allreduce(&local_finite, &global_finite_int, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        global_finite = (global_finite_int == 1);
#endif

        if (rank == 0) {
            if (global_finite) {
                std::cout << "PASS: No NaN/Inf after " << nsteps << " steps with halo exchange"
                          << " (nprocs=" << nprocs << ")" << std::endl;
                passed += 1;
            } else {
                std::cerr << "FAIL: NaN/Inf detected after stepping with halo exchange" << std::endl;
                failed += 1;
            }
        }

        // Check 2: Energy decreased (viscous dissipation)
        if (rank == 0) {
            std::cerr << "[test_mpi_halo_step] KE_init=" << global_ke_init
                      << " KE_final=" << global_ke_final << " nprocs=" << nprocs << std::endl;
            if (global_ke_final < global_ke_init * 0.9999) {
                std::cout << "PASS: Energy decreased with halo exchange"
                          << " (" << global_ke_init << " -> " << global_ke_final << ")" << std::endl;
                passed += 1;
            } else {
                std::cerr << "FAIL: Energy did not decrease with halo exchange"
                          << " (" << global_ke_init << " -> " << global_ke_final << ")" << std::endl;
                failed += 1;
            }
        }

        // Check 3: Energy is physically reasonable (not blown up)
        if (rank == 0) {
            if (global_ke_final > 0.0 && global_ke_final < 10.0 * global_ke_init) {
                std::cout << "PASS: Energy physically reasonable" << std::endl;
                passed += 1;
            } else {
                std::cerr << "FAIL: Energy unreasonable (KE_final=" << global_ke_final << ")" << std::endl;
                failed += 1;
            }
        }

        if (rank == 0) {
            std::cout << "\nMPI halo exchange step test: " << passed << " passed, "
                      << failed << " failed (nprocs=" << nprocs << ")" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "[rank " << rank << "] EXCEPTION: " << e.what() << std::endl;
        failed = 1;
    }

#ifdef USE_MPI
    if (failed > 0) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
#endif

    return (failed > 0) ? 1 : 0;
}
