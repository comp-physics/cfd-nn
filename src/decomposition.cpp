/// @file decomposition.cpp
/// @brief Implementation of 1D z-slab decomposition for multi-GPU DNS/LES

#include "decomposition.hpp"
#include <stdexcept>
#include <string>

namespace nncfd {

#ifdef USE_MPI
Decomposition::Decomposition(MPI_Comm comm, int Nz_global)
    : comm_(comm), nz_global_(Nz_global)
{
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &nprocs_);

    if (Nz_global < nprocs_) {
        throw std::runtime_error("Nz_global (" + std::to_string(Nz_global)
            + ") must be >= nprocs (" + std::to_string(nprocs_) + ")");
    }

    // Even distribution with remainder spread across first ranks
    int base = Nz_global / nprocs_;
    int remainder = Nz_global % nprocs_;

    nz_local_ = base + (rank_ < remainder ? 1 : 0);

    // Compute global start index
    k_global_start_ = 0;
    for (int r = 0; r < rank_; ++r) {
        k_global_start_ += base + (r < remainder ? 1 : 0);
    }

    // Periodic neighbors in z
    rank_lo_ = (rank_ - 1 + nprocs_) % nprocs_;
    rank_hi_ = (rank_ + 1) % nprocs_;
}
#endif

Decomposition::Decomposition(int Nz_global)
    : rank_(0), nprocs_(1),
      nz_global_(Nz_global), nz_local_(Nz_global),
      k_global_start_(0), rank_lo_(0), rank_hi_(0)
{
#ifdef USE_MPI
    comm_ = MPI_COMM_SELF;
#endif
}

double Decomposition::allreduce_sum(double local_val) const {
#ifdef USE_MPI
    if (nprocs_ > 1) {
        double global_val = 0.0;
        MPI_Allreduce(&local_val, &global_val, 1, MPI_DOUBLE, MPI_SUM, comm_);
        return global_val;
    }
#endif
    return local_val;
}

double Decomposition::allreduce_min(double local_val) const {
#ifdef USE_MPI
    if (nprocs_ > 1) {
        double global_val = 0.0;
        MPI_Allreduce(&local_val, &global_val, 1, MPI_DOUBLE, MPI_MIN, comm_);
        return global_val;
    }
#endif
    return local_val;
}

double Decomposition::allreduce_max(double local_val) const {
#ifdef USE_MPI
    if (nprocs_ > 1) {
        double global_val = 0.0;
        MPI_Allreduce(&local_val, &global_val, 1, MPI_DOUBLE, MPI_MAX, comm_);
        return global_val;
    }
#endif
    return local_val;
}

void Decomposition::allreduce_sum(double* data, int count) const {
    (void)data; (void)count;
#ifdef USE_MPI
    if (nprocs_ > 1) {
        MPI_Allreduce(MPI_IN_PLACE, data, count, MPI_DOUBLE, MPI_SUM, comm_);
    }
#endif
}

} // namespace nncfd
