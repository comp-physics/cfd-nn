#pragma once

/// @file decomposition.hpp
/// @brief 1D slab decomposition in z-direction for multi-GPU DNS/LES
///
/// Distributes the z-dimension across MPI ranks for parallel simulation.
/// Each rank owns a contiguous z-slab of Nz_local cells. Periodic neighbor
/// topology wraps rank 0 to rank (nprocs-1).
///
/// Usage:
///   Decomposition decomp(MPI_COMM_WORLD, Nz_global);  // MPI parallel
///   Decomposition decomp(Nz_global);                   // single-process fallback
///
/// The decomposition evenly distributes cells with remainder spread across
/// the first ranks (rank r gets Nz_global/nprocs + (r < remainder ? 1 : 0)).
///
/// @note Requires USE_MPI to be defined for actual MPI communication.
///       Without USE_MPI, only the single-process constructor is available.

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace nncfd {

class Decomposition {
public:
#ifdef USE_MPI
    /// Construct with MPI communicator and global z-cell count
    /// @param comm       MPI communicator (typically MPI_COMM_WORLD)
    /// @param Nz_global  Total number of z-cells across all ranks
    /// @throws std::runtime_error if Nz_global < nprocs
    Decomposition(MPI_Comm comm, int Nz_global);
#endif

    /// Trivial single-process decomposition (no MPI required)
    /// @param Nz_global  Total number of z-cells (all on this process)
    explicit Decomposition(int Nz_global);

    // Accessors
    int rank() const { return rank_; }
    int nprocs() const { return nprocs_; }
    int nz_local() const { return nz_local_; }
    int nz_global() const { return nz_global_; }
    int k_global_start() const { return k_global_start_; }
    int rank_lo() const { return rank_lo_; }
    int rank_hi() const { return rank_hi_; }
    bool is_parallel() const { return nprocs_ > 1; }

#ifdef USE_MPI
    MPI_Comm comm() const { return comm_; }
#endif

    /// Convert local k-index (0-based, no ghost) to global
    int k_local_to_global(int k_local) const { return k_global_start_ + k_local; }

    /// Global z-coordinate for local k-index (cell-center)
    double z_global(int k_local, double z_min, double Lz) const {
        double dz = Lz / nz_global_;
        return z_min + (k_global_start_ + k_local + 0.5) * dz;
    }

    /// Allreduce scalar (sum) — returns local_val if single process
    double allreduce_sum(double local_val) const;

    /// Allreduce scalar (min)
    double allreduce_min(double local_val) const;

    /// Allreduce scalar (max)
    double allreduce_max(double local_val) const;

    /// Allreduce vector (sum, in-place)
    void allreduce_sum(double* data, int count) const;

private:
#ifdef USE_MPI
    MPI_Comm comm_ = MPI_COMM_SELF;
#endif
    int rank_ = 0;
    int nprocs_ = 1;
    int nz_global_ = 0;
    int nz_local_ = 0;
    int k_global_start_ = 0;
    int rank_lo_ = 0;   // Periodic neighbor (z-lo direction)
    int rank_hi_ = 0;   // Periodic neighbor (z-hi direction)
};

} // namespace nncfd
