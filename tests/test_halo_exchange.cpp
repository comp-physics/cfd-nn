/// @file test_halo_exchange.cpp
/// @brief Tests for MPI z-direction halo exchange
///
/// Test coverage:
///   1. Single-process: exchange is a no-op (field unchanged)
///   2. MPI periodic exchange: ghost cells receive neighbor's interior data
///   3. Data integrity: pack → MPI → unpack preserves values exactly
///   4. Multi-field batch exchange
///
/// Without USE_MPI, only single-process no-op tests run.
/// With USE_MPI, run: mpirun -np 2 ./test_halo_exchange
///                     mpirun -np 4 ./test_halo_exchange

#include "halo_exchange.hpp"
#include "decomposition.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace nncfd;

/// Test: single-process exchange is a no-op
void test_single_process_noop() {
    const int Nx = 8, Ny = 8, Nz = 16, Ng = 1;
    Decomposition decomp(Nz);  // single-process

    const int stride = Nx + 2*Ng;
    const int plane_stride = stride * (Ny + 2*Ng);
    const int total = stride * (Ny + 2*Ng) * (Nz + 2*Ng);

    std::vector<double> field(total, 0.0);

    // Fill with pattern
    for (int k = Ng; k < Nz + Ng; ++k)
        for (int j = 0; j < Ny + 2*Ng; ++j)
            for (int i = 0; i < Nx + 2*Ng; ++i)
                field[k * plane_stride + j * stride + i] = k * 1000.0 + j * 10.0 + i;

    // Save copy
    std::vector<double> field_copy = field;

    HaloExchange halo(decomp, Nx, Ny, Nz, Ng);
    halo.exchange(field.data(), stride, plane_stride);

    // Field should be unchanged (no parallel exchange)
    for (int idx = 0; idx < total; ++idx) {
        assert(field[idx] == field_copy[idx]);
    }

    std::cout << "PASS: Single-process halo exchange is no-op" << std::endl;
}

#ifdef USE_MPI
/// Test: MPI periodic z-halo exchange with known pattern
void test_mpi_halo_exchange() {
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const int Nx = 8, Ny = 8, Nz_global = 16, Ng = 1;
    Decomposition decomp(MPI_COMM_WORLD, Nz_global);

    const int Nz_local = decomp.nz_local();
    const int stride = Nx + 2*Ng;
    const int plane_stride = stride * (Ny + 2*Ng);
    const int total = stride * (Ny + 2*Ng) * (Nz_local + 2*Ng);

    std::vector<double> field(total, 0.0);

    // Fill interior with rank-dependent pattern: value = k_global * 1000 + j * 100 + i
    for (int k = 0; k < Nz_local; ++k) {
        int k_global = decomp.k_local_to_global(k);
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                int idx = (k + Ng) * plane_stride + (j + Ng) * stride + (i + Ng);
                field[idx] = k_global * 1000.0 + j * 100.0 + i;
            }
        }
    }

    // Exchange halos
    HaloExchange halo(decomp, Nx, Ny, Nz_local, Ng);
    halo.exchange(field.data(), stride, plane_stride);

    // Verify low-z ghost (k=0) has neighbor's last interior plane
    {
        int k_expected_global = (decomp.k_global_start() - 1 + Nz_global) % Nz_global;
        double max_err = 0.0;
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                int idx = 0 * plane_stride + (j + Ng) * stride + (i + Ng);
                double expected = k_expected_global * 1000.0 + j * 100.0 + i;
                max_err = std::max(max_err, std::abs(field[idx] - expected));
            }
        }
        assert(max_err < 1e-10 && "Low-z ghost must match neighbor's last interior");
    }

    // Verify high-z ghost (k=Nz_local+1) has neighbor's first interior plane
    {
        int k_expected_global = (decomp.k_global_start() + Nz_local) % Nz_global;
        double max_err = 0.0;
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                int idx = (Nz_local + Ng) * plane_stride + (j + Ng) * stride + (i + Ng);
                double expected = k_expected_global * 1000.0 + j * 100.0 + i;
                max_err = std::max(max_err, std::abs(field[idx] - expected));
            }
        }
        assert(max_err < 1e-10 && "High-z ghost must match neighbor's first interior");
    }

    if (rank == 0) {
        std::cout << "PASS: MPI halo exchange with " << nprocs << " ranks" << std::endl;
    }
}

/// Test: batch exchange for multiple fields
void test_mpi_batch_exchange() {
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const int Nx = 4, Ny = 4, Nz_global = 8, Ng = 1;
    Decomposition decomp(MPI_COMM_WORLD, Nz_global);

    const int Nz_local = decomp.nz_local();
    const int stride = Nx + 2*Ng;
    const int plane_stride = stride * (Ny + 2*Ng);
    const int total = stride * (Ny + 2*Ng) * (Nz_local + 2*Ng);

    // Two fields with different patterns
    std::vector<double> field_a(total, 0.0), field_b(total, 0.0);
    for (int k = 0; k < Nz_local; ++k) {
        int kg = decomp.k_local_to_global(k);
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                int idx = (k + Ng) * plane_stride + (j + Ng) * stride + (i + Ng);
                field_a[idx] = kg + 1.0;       // simple pattern
                field_b[idx] = (kg + 1.0) * 2; // scaled pattern
            }
        }
    }

    HaloExchange halo(decomp, Nx, Ny, Nz_local, Ng);
    double* fields[2] = {field_a.data(), field_b.data()};
    halo.exchange_batch(fields, 2, stride, plane_stride);

    // Verify both fields got correct ghosts
    int k_lo_global = (decomp.k_global_start() - 1 + Nz_global) % Nz_global;
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            int idx = 0 * plane_stride + (j + Ng) * stride + (i + Ng);
            assert(std::abs(field_a[idx] - (k_lo_global + 1.0)) < 1e-10);
            assert(std::abs(field_b[idx] - (k_lo_global + 1.0) * 2) < 1e-10);
        }
    }

    if (rank == 0) {
        std::cout << "PASS: Batch halo exchange with " << nprocs << " ranks" << std::endl;
    }
}
#endif // USE_MPI

int main(int argc, char** argv) {
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    (void)argc; (void)argv;
#endif

    test_single_process_noop();

#ifdef USE_MPI
    test_mpi_halo_exchange();
    test_mpi_batch_exchange();

    if (rank == 0) {
        std::cout << "\nAll halo exchange tests PASSED" << std::endl;
    }
    MPI_Finalize();
#else
    std::cout << "\nAll halo exchange tests PASSED (single-process only)" << std::endl;
#endif

    return 0;
}
