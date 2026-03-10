/// @file test_decomposition.cpp
/// @brief Tests for MPI z-slab domain decomposition
///
/// Test coverage:
///   1. Single-process: all cells on one rank, k_global_start=0
///   2. Cell count conservation: sum of local Nz across ranks = Nz_global
///   3. Neighbor topology: periodic wrap (rank 0's lo = nprocs-1)
///   4. k_local_to_global mapping consistency
///   5. Allreduce operations: sum, min, max
///   6. Edge case: Nz_global = nprocs (1 cell per rank)
///
/// When compiled with USE_MPI, run with: mpirun -np 2 ./test_decomposition
/// Without USE_MPI, tests only single-process decomposition.

#include "decomposition.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nncfd;

/// Test single-process decomposition (no MPI)
void test_single_process() {
    const int Nz_global = 192;
    Decomposition decomp(Nz_global);

    auto check = [](bool cond, const char* msg) {
        if (!cond) throw std::runtime_error(std::string("FAIL: ") + msg);
    };

    check(decomp.rank() == 0, "rank != 0");
    check(decomp.nprocs() == 1, "nprocs != 1");
    check(decomp.nz_local() == Nz_global, "nz_local != Nz_global");
    check(decomp.nz_global() == Nz_global, "nz_global != Nz_global");
    check(decomp.k_global_start() == 0, "k_global_start != 0");
    check(!decomp.is_parallel(), "is_parallel should be false");

    // k_local_to_global is identity for single process
    check(decomp.k_local_to_global(0) == 0, "k_local_to_global(0)");
    check(decomp.k_local_to_global(100) == 100, "k_local_to_global(100)");
    check(decomp.k_local_to_global(Nz_global - 1) == Nz_global - 1, "k_local_to_global(Nz-1)");

    // Allreduce is identity for single process
    check(std::abs(decomp.allreduce_sum(3.14) - 3.14) < 1e-15, "allreduce_sum");
    check(std::abs(decomp.allreduce_min(2.71) - 2.71) < 1e-15, "allreduce_min");
    check(std::abs(decomp.allreduce_max(1.41) - 1.41) < 1e-15, "allreduce_max");

    // z_global coordinate: cell center at z = 0.5*dz
    check(std::abs(decomp.z_global(0, 0.0, M_PI) - 0.5 * M_PI / Nz_global) < 1e-14,
          "z_global coordinate");

    std::cout << "PASS: Single-process decomposition" << std::endl;
}

/// Test single-process allreduce_sum on vector
void test_single_process_vector_allreduce() {
    Decomposition decomp(64);

    double data[4] = {1.0, 2.0, 3.0, 4.0};
    decomp.allreduce_sum(data, 4);
    // Should be unchanged for single process
    if (!(data[0] == 1.0 && data[1] == 2.0 && data[2] == 3.0 && data[3] == 4.0))
        throw std::runtime_error("FAIL: allreduce_sum(vector) modified values for single process");

    std::cout << "PASS: Single-process vector allreduce" << std::endl;
}

/// Test decomposition with various Nz sizes
void test_various_sizes() {
    // Even division
    {
        Decomposition decomp(128);
        if (decomp.nz_local() != 128)
            throw std::runtime_error("FAIL: nz_local != 128 for Nz=128");
    }

    // Odd number
    {
        Decomposition decomp(7);
        if (decomp.nz_local() != 7)
            throw std::runtime_error("FAIL: nz_local != 7 for Nz=7");
    }

    // Minimum size
    {
        Decomposition decomp(1);
        if (decomp.nz_local() != 1)
            throw std::runtime_error("FAIL: nz_local != 1 for Nz=1");
    }

    std::cout << "PASS: Various Nz sizes" << std::endl;
}

#ifdef USE_MPI
/// Test MPI decomposition (requires mpirun)
void test_mpi_decomposition() {
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const int Nz_global = 192;
    Decomposition decomp(MPI_COMM_WORLD, Nz_global);

    auto mpi_check_cond = [](bool cond, const char* msg) {
        if (!cond) {
            std::cerr << "FAIL: " << msg << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    };

    // Basic invariants
    mpi_check_cond(decomp.rank() == rank, "decomp.rank() != MPI rank");
    mpi_check_cond(decomp.nprocs() == nprocs, "decomp.nprocs() != MPI nprocs");
    mpi_check_cond(decomp.nz_local() > 0, "nz_local == 0");
    mpi_check_cond(decomp.nz_local() <= Nz_global, "nz_local > Nz_global");
    mpi_check_cond(decomp.nz_global() == Nz_global, "nz_global != Nz_global");

    if (nprocs > 1) {
        mpi_check_cond(decomp.is_parallel(), "is_parallel should be true for nprocs>1");
    }

    // Cell count conservation: sum of all local Nz = Nz_global
    int local_nz = decomp.nz_local();
    int global_nz_sum = 0;
    MPI_Allreduce(&local_nz, &global_nz_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    mpi_check_cond(global_nz_sum == Nz_global, "sum of nz_local != Nz_global");

    // Periodic neighbor topology
    if (nprocs > 1) {
        mpi_check_cond(decomp.rank_lo() >= 0 && decomp.rank_lo() < nprocs, "rank_lo out of range");
        mpi_check_cond(decomp.rank_hi() >= 0 && decomp.rank_hi() < nprocs, "rank_hi out of range");
        if (rank == 0) mpi_check_cond(decomp.rank_lo() == nprocs - 1, "rank 0 rank_lo != nprocs-1");
        if (rank == nprocs - 1) mpi_check_cond(decomp.rank_hi() == 0, "last rank rank_hi != 0");
    }

    // Global z-offset consistency: no gaps, no overlaps
    int k_start = decomp.k_global_start();
    mpi_check_cond(k_start >= 0 && k_start < Nz_global, "k_global_start out of range");

    // k_local_to_global
    mpi_check_cond(decomp.k_local_to_global(0) == k_start, "k_local_to_global(0) != k_start");
    mpi_check_cond(decomp.k_local_to_global(decomp.nz_local() - 1) == k_start + decomp.nz_local() - 1,
                   "k_local_to_global(nz_local-1) wrong");

    // Verify no overlap: gather all k_start/nz_local and check
    std::vector<int> all_starts(nprocs), all_counts(nprocs);
    MPI_Allgather(&k_start, 1, MPI_INT, all_starts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&local_nz, 1, MPI_INT, all_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    for (int r = 0; r < nprocs - 1; ++r) {
        mpi_check_cond(all_starts[r + 1] == all_starts[r] + all_counts[r],
                       "gap or overlap in z-slab partition");
    }
    // Last rank's end should equal Nz_global
    mpi_check_cond(all_starts[nprocs - 1] + all_counts[nprocs - 1] == Nz_global,
                   "partition does not cover full Nz_global");

    // Allreduce sum: each rank contributes its rank number
    double my_val = static_cast<double>(rank);
    double sum = decomp.allreduce_sum(my_val);
    double expected_sum = nprocs * (nprocs - 1) / 2.0;
    if (std::abs(sum - expected_sum) >= 1e-12) {
        std::cerr << "FAIL: allreduce_sum expected " << expected_sum << " got " << sum << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Allreduce min/max
    double min_val = decomp.allreduce_min(static_cast<double>(rank));
    if (std::abs(min_val - 0.0) >= 1e-12) {
        std::cerr << "FAIL: allreduce_min expected 0 got " << min_val << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    double max_val = decomp.allreduce_max(static_cast<double>(rank));
    if (std::abs(max_val - (nprocs - 1)) >= 1e-12) {
        std::cerr << "FAIL: allreduce_max expected " << (nprocs-1) << " got " << max_val << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        std::cout << "PASS: MPI decomposition with " << nprocs << " ranks"
                  << " (each rank has " << decomp.nz_local() << " z-cells)" << std::endl;
    }
}

/// Test edge case: Nz_global = nprocs (1 cell per rank)
void test_mpi_minimal_cells() {
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    Decomposition decomp(MPI_COMM_WORLD, nprocs);
    if (decomp.nz_local() != 1) {
        std::cerr << "FAIL: nz_local != 1 for Nz=nprocs\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (decomp.k_global_start() != rank) {
        std::cerr << "FAIL: k_global_start != rank for Nz=nprocs\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        std::cout << "PASS: Minimal cells (1 per rank)" << std::endl;
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

    try {
        // Single-process tests (always run)
        test_single_process();
        test_single_process_vector_allreduce();
        test_various_sizes();
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
#ifdef USE_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        return 1;
    }

#ifdef USE_MPI
    // MPI tests (only when compiled with MPI)
    test_mpi_decomposition();
    test_mpi_minimal_cells();

    if (rank == 0) {
        std::cout << "\nAll decomposition tests PASSED" << std::endl;
    }
    MPI_Finalize();
#else
    std::cout << "\nAll decomposition tests PASSED (single-process only)" << std::endl;
#endif

    return 0;
}
