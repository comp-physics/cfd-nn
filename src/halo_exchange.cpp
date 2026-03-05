/// @file halo_exchange.cpp
/// @brief Implementation of MPI z-direction halo exchange
///
/// CPU path: pack z-face planes into contiguous buffers, MPI_Isend/Irecv,
/// then unpack into ghost layers.
///
/// GPU path: CUDA pack kernels → cudaDeviceSynchronize → MPI on device
/// pointers → CUDA unpack kernels → cudaDeviceSynchronize.
///
/// The exchange is a no-op when decomp.is_parallel() is false.

#include "halo_exchange.hpp"
#include <cstring>
#include <stdexcept>

#ifdef USE_CUDA_KERNELS
#include "cuda_halo.hpp"
#include <cuda_runtime.h>
#endif

namespace nncfd {

HaloExchange::HaloExchange(const Decomposition& decomp,
                           int Nx, int Ny, int Nz_local, int Ng)
    : decomp_(decomp), Nx_(Nx), Ny_(Ny), Nz_local_(Nz_local), Ng_(Ng)
{
    face_size_ = (Nx + 2*Ng) * (Ny + 2*Ng) * Ng;
    send_lo_.resize(face_size_);
    send_hi_.resize(face_size_);
    recv_lo_.resize(face_size_);
    recv_hi_.resize(face_size_);
}

HaloExchange::~HaloExchange() {
#ifdef USE_CUDA_KERNELS
    if (gpu_buffers_initialized_) {
        cudaFree(d_send_lo_);
        cudaFree(d_send_hi_);
        cudaFree(d_recv_lo_);
        cudaFree(d_recv_hi_);
    }
#endif
}

void HaloExchange::pack_face_cpu(const double* field, double* buffer,
                                  int stride, int plane_stride, int k_start)
{
    int buf_idx = 0;
    for (int g = 0; g < Ng_; ++g) {
        int k = k_start + g;
        for (int j = 0; j < Ny_ + 2*Ng_; ++j) {
            for (int i = 0; i < Nx_ + 2*Ng_; ++i) {
                buffer[buf_idx++] = field[k * plane_stride + j * stride + i];
            }
        }
    }
}

void HaloExchange::unpack_face_cpu(double* field, const double* buffer,
                                    int stride, int plane_stride, int k_start)
{
    int buf_idx = 0;
    for (int g = 0; g < Ng_; ++g) {
        int k = k_start + g;
        for (int j = 0; j < Ny_ + 2*Ng_; ++j) {
            for (int i = 0; i < Nx_ + 2*Ng_; ++i) {
                field[k * plane_stride + j * stride + i] = buffer[buf_idx++];
            }
        }
    }
}

void HaloExchange::exchange(double* field, int stride, int plane_stride) {
    (void)field; (void)stride; (void)plane_stride;
    if (!decomp_.is_parallel()) return;

#ifdef USE_MPI
    // Pack: send_lo = first Ng interior planes, send_hi = last Ng interior planes
    pack_face_cpu(field, send_lo_.data(), stride, plane_stride, Ng_);
    pack_face_cpu(field, send_hi_.data(), stride, plane_stride, Nz_local_);

    MPI_Request reqs[4];

    // Send lo interior → neighbor's hi ghost; send hi interior → neighbor's lo ghost
    MPI_Isend(send_lo_.data(), face_size_, MPI_DOUBLE,
              decomp_.rank_lo(), 0, decomp_.comm(), &reqs[0]);
    MPI_Isend(send_hi_.data(), face_size_, MPI_DOUBLE,
              decomp_.rank_hi(), 1, decomp_.comm(), &reqs[1]);
    MPI_Irecv(recv_lo_.data(), face_size_, MPI_DOUBLE,
              decomp_.rank_lo(), 1, decomp_.comm(), &reqs[2]);
    MPI_Irecv(recv_hi_.data(), face_size_, MPI_DOUBLE,
              decomp_.rank_hi(), 0, decomp_.comm(), &reqs[3]);

    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

    // Unpack: lo ghost = planes [0, Ng), hi ghost = planes [Nz_local+Ng, Nz_local+2Ng)
    unpack_face_cpu(field, recv_lo_.data(), stride, plane_stride, 0);
    unpack_face_cpu(field, recv_hi_.data(), stride, plane_stride, Nz_local_ + Ng_);
#endif
}

void HaloExchange::exchange_device(double* d_field, int stride, int plane_stride) {
    (void)d_field; (void)stride; (void)plane_stride;
    if (!decomp_.is_parallel()) return;

#if defined(USE_CUDA_KERNELS) && defined(USE_MPI)
    if (!gpu_buffers_initialized_) init_gpu_buffers();

    // Pack on GPU
    cuda_kernels::launch_pack_z_face(d_field, d_send_lo_, Nx_, Ny_, Nz_local_, Ng_, true);
    cuda_kernels::launch_pack_z_face(d_field, d_send_hi_, Nx_, Ny_, Nz_local_, Ng_, false);
    cudaDeviceSynchronize();

    MPI_Request reqs[4];
    MPI_Isend(d_send_lo_, face_size_, MPI_DOUBLE,
              decomp_.rank_lo(), 0, decomp_.comm(), &reqs[0]);
    MPI_Isend(d_send_hi_, face_size_, MPI_DOUBLE,
              decomp_.rank_hi(), 1, decomp_.comm(), &reqs[1]);
    MPI_Irecv(d_recv_lo_, face_size_, MPI_DOUBLE,
              decomp_.rank_lo(), 1, decomp_.comm(), &reqs[2]);
    MPI_Irecv(d_recv_hi_, face_size_, MPI_DOUBLE,
              decomp_.rank_hi(), 0, decomp_.comm(), &reqs[3]);
    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

    // Unpack on GPU
    cuda_kernels::launch_unpack_z_face(d_field, d_recv_lo_, Nx_, Ny_, Nz_local_, Ng_, true);
    cuda_kernels::launch_unpack_z_face(d_field, d_recv_hi_, Nx_, Ny_, Nz_local_, Ng_, false);
    cudaDeviceSynchronize();
#else
    // No CUDA kernels + MPI: cannot safely exchange GPU-resident data
    throw std::runtime_error("[HaloExchange] exchange_device() requires USE_CUDA_KERNELS + USE_MPI. "
                             "Cannot exchange GPU pointers without CUDA pack/unpack kernels.");
#endif
}

void HaloExchange::exchange_batch(double** fields, int num_fields,
                                   int stride, int plane_stride) {
    for (int f = 0; f < num_fields; ++f) {
        exchange(fields[f], stride, plane_stride);
    }
}

#ifdef USE_CUDA_KERNELS
void HaloExchange::init_gpu_buffers() {
    auto check = [](cudaError_t err, const char* name) {
        if (err != cudaSuccess)
            throw std::runtime_error(std::string("[HaloExchange] cudaMalloc failed for ") +
                                     name + ": " + cudaGetErrorString(err));
    };
    check(cudaMalloc(&d_send_lo_, face_size_ * sizeof(double)), "send_lo");
    check(cudaMalloc(&d_send_hi_, face_size_ * sizeof(double)), "send_hi");
    check(cudaMalloc(&d_recv_lo_, face_size_ * sizeof(double)), "recv_lo");
    check(cudaMalloc(&d_recv_hi_, face_size_ * sizeof(double)), "recv_hi");
    gpu_buffers_initialized_ = true;
}
#endif

} // namespace nncfd
