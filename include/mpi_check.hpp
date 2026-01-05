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
///
/// Note: Only checks MPI-specific environment variables, NOT generic SLURM vars.
/// SLURM_NTASKS is always set in SLURM allocations but doesn't mean MPI is active.
/// We only want to warn when actually running with mpirun/srun with MPI.
inline int detect_mpi_ranks() {
    // Check MPI-specific environment variables only (set by mpirun/mpiexec)
    const char* ompi_size = std::getenv("OMPI_COMM_WORLD_SIZE");      // OpenMPI
    const char* mvapich_size = std::getenv("MV2_COMM_WORLD_SIZE");    // MVAPICH
    const char* mpich_size = std::getenv("PMI_SIZE");                 // MPICH/PMI

    // Note: SLURM_NTASKS is intentionally NOT checked here.
    // It's set for all SLURM jobs, not just MPI jobs.
    // We only detect actual MPI usage via MPI library env vars.

    int nranks = 0;
    if (ompi_size) nranks = std::atoi(ompi_size);
    else if (mvapich_size) nranks = std::atoi(mvapich_size);
    else if (mpich_size) nranks = std::atoi(mpich_size);

    return nranks;
}

/// @brief Get the current rank (0-based), or 0 if not running under MPI
/// Only uses MPI-specific env vars (not SLURM_PROCID which is always set)
inline int detect_mpi_rank() {
    const char* ompi_rank = std::getenv("OMPI_COMM_WORLD_RANK");
    const char* mvapich_rank = std::getenv("MV2_COMM_WORLD_RANK");
    const char* pmi_rank = std::getenv("PMI_RANK");

    if (ompi_rank) return std::atoi(ompi_rank);
    if (mvapich_rank) return std::atoi(mvapich_rank);
    if (pmi_rank) return std::atoi(pmi_rank);

    return 0;
}

/// @brief Check for MPI environment and warn if multiple ranks detected
/// @param caller_name Name of the calling component (for logging)
/// @return true if MPI warning was issued (world_size > 1)
/// @note Only warns when world_size > 1. A single-rank MPI job (world_size=1)
///       or non-MPI environment (world_size=0) will NOT trigger a warning.
inline bool warn_if_mpi_detected(const std::string& caller_name = "Solver") {
    int nranks = detect_mpi_ranks();
    int rank = detect_mpi_rank();

    // Only warn if world_size > 1 (actual multi-rank job)
    // Single-rank or non-MPI environments are fine
    if (nranks > 1) {
        // Only warn from rank 0 to avoid spam
        if (rank == 0) {
            std::cerr << "\n"
                << "================================================================\n"
                << "  WARNING: Multi-rank MPI not supported (world_size=" << nranks << ")\n"
                << "================================================================\n"
                << "\n"
                << "  This code uses GPU parallelism, not MPI domain decomposition.\n"
                << "  Each rank runs the FULL simulation independently.\n"
                << "\n"
                << "  Detected: world_size=" << nranks << ", rank=" << rank << "\n"
                << "\n"
                << "  Run with single rank:\n"
                << "    ./channel input.yaml\n"
                << "    # or: mpirun -np 1 ./channel input.yaml\n"
                << "\n"
                << "  [" << caller_name << "]\n"
                << "================================================================\n\n";
        }
        return true;
    }
    return false;
}

/// @brief Check MPI environment and exit if multiple ranks (strict mode)
/// @param caller_name Name of the calling component (for logging)
/// @note Only exits when world_size > 1. Single-rank or non-MPI is allowed.
inline void enforce_single_rank(const std::string& caller_name = "Solver") {
    int nranks = detect_mpi_ranks();
    int rank = detect_mpi_rank();

    if (nranks > 1) {
        if (rank == 0) {
            std::cerr << "\n"
                << "================================================================\n"
                << "  ERROR: Multi-rank MPI not supported (world_size=" << nranks << ")\n"
                << "================================================================\n"
                << "\n"
                << "  Run with single rank: ./channel input.yaml\n"
                << "\n"
                << "  [" << caller_name << "]\n"
                << "================================================================\n\n";
        }
        std::exit(1);
    }
}

/// @brief Result of MPI guard check for testability
struct MpiGuardResult {
    bool should_exit;       ///< True if multi-rank detected and not allowed
    bool is_multi_rank;     ///< True if world_size > 1
    bool is_override;       ///< True if NNCFD_ALLOW_MULTI_RANK=1 was set
    int world_size;         ///< Detected world size
    int rank;               ///< Detected rank
};

/// @brief Check MPI guard policy (pure function for testability)
/// @param world_size Number of MPI ranks (0 = not MPI, 1 = single rank)
/// @param allow_override Value of NNCFD_ALLOW_MULTI_RANK env var (nullptr if not set)
/// @param is_gpu_build True if USE_GPU_OFFLOAD is enabled
/// @return MpiGuardResult with decision and metadata
inline MpiGuardResult check_mpi_guard_policy(int world_size, const char* allow_override,
                                              bool is_gpu_build) {
    MpiGuardResult result;
    result.world_size = world_size;
    result.rank = 0;  // Caller should set this separately if needed
    result.is_multi_rank = (world_size > 1);
    result.is_override = (allow_override != nullptr &&
                          (std::string(allow_override) == "1" ||
                           std::string(allow_override) == "true" ||
                           std::string(allow_override) == "TRUE"));

    // Policy:
    // - GPU builds: hard fail on multi-rank unless override
    // - CPU builds: warn only (handled separately)
    if (is_gpu_build && result.is_multi_rank && !result.is_override) {
        result.should_exit = true;
    } else {
        result.should_exit = false;
    }

    return result;
}

/// @brief Hard guard for GPU builds - exit if multi-rank unless NNCFD_ALLOW_MULTI_RANK=1
/// @param caller_name Name of the calling component (for logging)
/// @return true if multi-rank was detected (even if allowed via override)
///
/// For GPU builds, multiple MPI ranks on one node cause:
///   - GPU oversubscription (multiple ranks fight for same GPU)
///   - Silent performance collapse
///   - Potential incorrect results (no domain decomposition)
///
/// Use NNCFD_ALLOW_MULTI_RANK=1 to override if you know what you're doing.
inline bool enforce_single_rank_gpu(const std::string& caller_name = "Solver") {
    int nranks = detect_mpi_ranks();
    int rank = detect_mpi_rank();
    const char* allow_override = std::getenv("NNCFD_ALLOW_MULTI_RANK");

#ifdef USE_GPU_OFFLOAD
    constexpr bool is_gpu_build = true;
#else
    constexpr bool is_gpu_build = false;
#endif

    MpiGuardResult result = check_mpi_guard_policy(nranks, allow_override, is_gpu_build);
    result.rank = rank;

    if (result.is_multi_rank) {
        if (result.should_exit) {
            // GPU build, multi-rank, no override -> hard fail
            if (rank == 0) {
                std::cerr << "\n"
                    << "================================================================\n"
                    << "  ERROR: Multi-rank MPI with GPU offload is not supported\n"
                    << "================================================================\n"
                    << "\n"
                    << "  Detected: world_size=" << nranks << ", rank=" << rank << "\n"
                    << "\n"
                    << "  Problem: Multiple ranks will fight for the same GPU,\n"
                    << "  causing oversubscription and incorrect results.\n"
                    << "\n"
                    << "  Solutions:\n"
                    << "    1. Run single-rank: ./channel input.yaml\n"
                    << "    2. Use srun -n 1 instead of srun -n N\n"
                    << "    3. Override (DANGEROUS): NNCFD_ALLOW_MULTI_RANK=1\n"
                    << "\n"
                    << "  [" << caller_name << " - GPU build]\n"
                    << "================================================================\n\n";
            }
            std::exit(1);
        } else if (result.is_override) {
            // Multi-rank but override set -> warn loudly
            if (rank == 0) {
                std::cerr << "\n"
                    << "================================================================\n"
                    << "  WARNING: Multi-rank MPI override active (DANGEROUS)\n"
                    << "================================================================\n"
                    << "  world_size=" << nranks << ", NNCFD_ALLOW_MULTI_RANK=1\n"
                    << "  Proceeding anyway - expect GPU oversubscription issues!\n"
                    << "  [" << caller_name << "]\n"
                    << "================================================================\n\n";
            }
        } else {
            // CPU build, multi-rank -> just warn
            warn_if_mpi_detected(caller_name);
        }
        return true;
    }

    return false;
}

} // namespace nncfd
