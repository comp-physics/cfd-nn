#pragma once

/// @file halo_exchange.hpp
/// @brief MPI z-direction halo exchange for domain decomposition
///
/// Manages exchange of ghost cell data between neighboring MPI ranks in the
/// z-direction. Supports both CPU (host-staged) and GPU-direct communication.
///
/// For single-process runs (decomp.is_parallel() == false), all exchange
/// methods are no-ops.
///
/// Communication pattern (periodic z):
///   rank_lo <-- send_lo [first Ng interior planes] --> rank_lo receives in hi ghost
///   rank_hi <-- send_hi [last Ng interior planes]  --> rank_hi receives in lo ghost
///
/// GPU-direct mode uses CUDA pack/unpack kernels + MPI on device pointers.
/// Requires MPI implementation with CUDA-aware support (e.g., OpenMPI + UCX).

#include "decomposition.hpp"
#include <vector>

namespace nncfd {

class HaloExchange {
public:
    /// @param decomp    Domain decomposition (provides neighbor ranks)
    /// @param Nx        Interior x-cells
    /// @param Ny        Interior y-cells
    /// @param Nz_local  Local interior z-cells on this rank
    /// @param Ng        Ghost cell width (typically 1)
    HaloExchange(const Decomposition& decomp, int Nx, int Ny, int Nz_local, int Ng);
    ~HaloExchange();

    HaloExchange(const HaloExchange&) = delete;
    HaloExchange& operator=(const HaloExchange&) = delete;

    /// Exchange z-halos for a single field (CPU host memory)
    /// Uses MPI_Isend/Irecv with host-side pack/unpack
    void exchange(double* field, int stride, int plane_stride);

    /// Exchange z-halos for a single field (GPU device memory)
    /// Uses CUDA pack/unpack kernels + MPI on device pointers
    void exchange_device(double* d_field, int stride, int plane_stride);

    /// Exchange z-halos for an OpenMP-mapped field (host pointer, GPU-resident)
    /// Uses target update from/to for GPU↔host sync + CPU MPI exchange.
    /// @param host_ptr  Host pointer to the mapped array (NOT device pointer)
    /// @param total_size Total array size in doubles (for target update bounds)
    void exchange_host_staged(double* host_ptr, int stride, int plane_stride,
                               int total_size);

    /// Exchange z-halos for multiple fields simultaneously (CPU)
    void exchange_batch(double** fields, int num_fields, int stride, int plane_stride);

private:
    const Decomposition& decomp_;
    int Nx_, Ny_, Nz_local_, Ng_;
    int face_size_;  // Max face size across all field types: (Nx+1+2Ng) * (Ny+1+2Ng) * Ng

    // Host buffers for packing/unpacking
    std::vector<double> send_lo_, send_hi_;
    std::vector<double> recv_lo_, recv_hi_;

    // GPU buffers (allocated on first use)
    double* d_send_lo_ = nullptr;
    double* d_send_hi_ = nullptr;
    double* d_recv_lo_ = nullptr;
    double* d_recv_hi_ = nullptr;
    bool gpu_buffers_initialized_ = false;

    void init_gpu_buffers();
    void pack_face_cpu(const double* field, double* buffer,
                       int stride, int plane_stride, int k_start);
    void unpack_face_cpu(double* field, const double* buffer,
                         int stride, int plane_stride, int k_start);
};

} // namespace nncfd
