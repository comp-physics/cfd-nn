/// @file mpi_check.hpp
/// @brief MPI environment detection and warning utilities
///
/// This codebase is designed for single-node GPU parallelism via OpenMP offload.
/// MPI parallelism is NOT supported. These utilities detect MPI usage and warn
/// the user about unsupported configurations.
///
/// IMPORTANT: HYPRE is built with HYPRE_WITH_MPI=OFF. Running with multiple
/// MPI ranks will NOT distribute the workload and may cause incorrect results.

#pragma once

#include <cstdlib>
#include <iostream>
#include <string>

namespace nncfd {

/// @brief Check if running under MPI and return detected rank count
/// @return Number of detected ranks (0 if not running under MPI, 1 if single rank)
inline int detect_mpi_ranks() {
    // Check common MPI environment variables for world size
    const char* ompi_size = std::getenv("OMPI_COMM_WORLD_SIZE");      // OpenMPI
    const char* mvapich_size = std::getenv("MV2_COMM_WORLD_SIZE");    // MVAPICH
    const char* mpich_size = std::getenv("PMI_SIZE");                 // MPICH/PMI
    const char* slurm_ntasks = std::getenv("SLURM_NTASKS");           // SLURM

    int nranks = 0;
    if (ompi_size) nranks = std::atoi(ompi_size);
    else if (mvapich_size) nranks = std::atoi(mvapich_size);
    else if (mpich_size) nranks = std::atoi(mpich_size);
    else if (slurm_ntasks) nranks = std::atoi(slurm_ntasks);

    return nranks;
}

/// @brief Get the current rank (0-based), or 0 if not running under MPI
inline int detect_mpi_rank() {
    const char* ompi_rank = std::getenv("OMPI_COMM_WORLD_RANK");
    const char* mvapich_rank = std::getenv("MV2_COMM_WORLD_RANK");
    const char* pmi_rank = std::getenv("PMI_RANK");
    const char* slurm_procid = std::getenv("SLURM_PROCID");

    if (ompi_rank) return std::atoi(ompi_rank);
    if (mvapich_rank) return std::atoi(mvapich_rank);
    if (pmi_rank) return std::atoi(pmi_rank);
    if (slurm_procid) return std::atoi(slurm_procid);

    return 0;
}

/// @brief Check for MPI environment and warn if multiple ranks detected
/// @param caller_name Name of the calling component (for logging)
/// @return true if MPI warning was issued
inline bool warn_if_mpi_detected(const std::string& caller_name = "Solver") {
    int nranks = detect_mpi_ranks();
    int rank = detect_mpi_rank();

    if (nranks > 1) {
        // Only warn from rank 0 to avoid spam
        if (rank == 0) {
            std::cerr << "\n"
                << "================================================================\n"
                << "  WARNING: Multiple MPI ranks detected (" << nranks << " ranks)\n"
                << "================================================================\n"
                << "\n"
                << "  This code uses single-node GPU parallelism via OpenMP target.\n"
                << "  MPI parallelism is NOT supported.\n"
                << "\n"
                << "  Detected configuration:\n"
                << "    - Running as " << nranks << " MPI ranks\n"
                << "    - This rank: " << rank << "\n"
                << "\n"
                << "  Problems you may encounter:\n"
                << "    - Each rank runs the FULL simulation (no domain decomposition)\n"
                << "    - Memory usage multiplied by rank count\n"
                << "    - HYPRE solver uses MPI_COMM_SELF (no rank communication)\n"
                << "    - Results may be incorrect or duplicated\n"
                << "\n"
                << "  Recommended action:\n"
                << "    Run without mpirun/mpiexec:\n"
                << "      ./channel input.yaml\n"
                << "\n"
                << "  Or use a single rank:\n"
                << "      mpirun -np 1 ./channel input.yaml\n"
                << "\n"
                << "  Caller: " << caller_name << "\n"
                << "================================================================\n\n";
        }
        return true;
    }
    return false;
}

/// @brief Check MPI environment and exit if multiple ranks (strict mode)
/// @param caller_name Name of the calling component (for logging)
inline void enforce_single_rank(const std::string& caller_name = "Solver") {
    int nranks = detect_mpi_ranks();
    int rank = detect_mpi_rank();

    if (nranks > 1) {
        if (rank == 0) {
            std::cerr << "\n"
                << "================================================================\n"
                << "  ERROR: Multiple MPI ranks not supported (" << nranks << " ranks)\n"
                << "================================================================\n"
                << "\n"
                << "  This code is designed for single-node GPU parallelism.\n"
                << "  Please run without mpirun/mpiexec.\n"
                << "\n"
                << "  Caller: " << caller_name << "\n"
                << "================================================================\n\n";
        }
        std::exit(1);
    }
}

} // namespace nncfd
