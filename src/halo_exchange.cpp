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
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_MPI
namespace {
void mpi_check(int rc, const char* call) {
    if (rc != MPI_SUCCESS) {
        char msg[MPI_MAX_ERROR_STRING]; int len;
        MPI_Error_string(rc, msg, &len);
        throw std::runtime_error(std::string("[MPI] ") + call + " failed: " + msg);
    }
}
} // anonymous namespace
#endif

#ifdef USE_CUDA_KERNELS
#include "cuda_halo.hpp"
#include <cuda_runtime.h>
#endif

namespace nncfd {

HaloExchange::HaloExchange(const Decomposition& decomp,
                           int Nx, int Ny, int Nz_local, int Ng)
    : decomp_(decomp), Nx_(Nx), Ny_(Ny), Nz_local_(Nz_local), Ng_(Ng)
{
    // Size for largest face: u has Nx+1 columns, v has Ny+1 rows on staggered grid
    face_size_ = (Nx + 1 + 2*Ng) * (Ny + 1 + 2*Ng) * Ng;
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
    // Pack using caller's stride and plane_stride so all columns are captured.
    // For u-velocity, stride = Nx+1+2*Ng (one extra face column).
    int ny_rows = plane_stride / stride;  // Total y-rows including ghosts
    int buf_idx = 0;
    for (int g = 0; g < Ng_; ++g) {
        int k = k_start + g;
        for (int j = 0; j < ny_rows; ++j) {
            for (int i = 0; i < stride; ++i) {
                buffer[buf_idx++] = field[k * plane_stride + j * stride + i];
            }
        }
    }
}

void HaloExchange::unpack_face_cpu(double* field, const double* buffer,
                                    int stride, int plane_stride, int k_start)
{
    int ny_rows = plane_stride / stride;
    int buf_idx = 0;
    for (int g = 0; g < Ng_; ++g) {
        int k = k_start + g;
        for (int j = 0; j < ny_rows; ++j) {
            for (int i = 0; i < stride; ++i) {
                field[k * plane_stride + j * stride + i] = buffer[buf_idx++];
            }
        }
    }
}

void HaloExchange::exchange(double* field, int stride, int plane_stride) {
    (void)field; (void)stride; (void)plane_stride;
    if (!decomp_.is_parallel()) return;

#ifdef USE_MPI
    // Actual message size: Ng z-planes * plane_stride doubles each
    int msg_size = plane_stride * Ng_;

    // Pack: send_lo = first Ng interior planes, send_hi = last Ng interior planes
    pack_face_cpu(field, send_lo_.data(), stride, plane_stride, Ng_);
    pack_face_cpu(field, send_hi_.data(), stride, plane_stride, Nz_local_);

    MPI_Request reqs[4];

    // Send lo interior → neighbor's hi ghost; send hi interior → neighbor's lo ghost
    mpi_check(MPI_Isend(send_lo_.data(), msg_size, MPI_DOUBLE,
                        decomp_.rank_lo(), 0, decomp_.comm(), &reqs[0]), "MPI_Isend(lo)");
    mpi_check(MPI_Isend(send_hi_.data(), msg_size, MPI_DOUBLE,
                        decomp_.rank_hi(), 1, decomp_.comm(), &reqs[1]), "MPI_Isend(hi)");
    mpi_check(MPI_Irecv(recv_lo_.data(), msg_size, MPI_DOUBLE,
                        decomp_.rank_lo(), 1, decomp_.comm(), &reqs[2]), "MPI_Irecv(lo)");
    mpi_check(MPI_Irecv(recv_hi_.data(), msg_size, MPI_DOUBLE,
                        decomp_.rank_hi(), 0, decomp_.comm(), &reqs[3]), "MPI_Irecv(hi)");

    mpi_check(MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE), "MPI_Waitall");

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
    mpi_check(MPI_Isend(d_send_lo_, face_size_, MPI_DOUBLE,
                        decomp_.rank_lo(), 0, decomp_.comm(), &reqs[0]), "MPI_Isend(lo)");
    mpi_check(MPI_Isend(d_send_hi_, face_size_, MPI_DOUBLE,
                        decomp_.rank_hi(), 1, decomp_.comm(), &reqs[1]), "MPI_Isend(hi)");
    mpi_check(MPI_Irecv(d_recv_lo_, face_size_, MPI_DOUBLE,
                        decomp_.rank_lo(), 1, decomp_.comm(), &reqs[2]), "MPI_Irecv(lo)");
    mpi_check(MPI_Irecv(d_recv_hi_, face_size_, MPI_DOUBLE,
                        decomp_.rank_hi(), 0, decomp_.comm(), &reqs[3]), "MPI_Irecv(hi)");
    mpi_check(MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE), "MPI_Waitall");

    // Unpack on GPU
    cuda_kernels::launch_unpack_z_face(d_field, d_recv_lo_, Nx_, Ny_, Nz_local_, Ng_, true);
    cuda_kernels::launch_unpack_z_face(d_field, d_recv_hi_, Nx_, Ny_, Nz_local_, Ng_, false);
    cudaDeviceSynchronize();
#elif defined(USE_MPI)
    // OpenMP offload fallback: caller must pass the HOST pointer (the mapped
    // copy) and bracket the call with target update from/to for the relevant
    // z-planes. See exchange_host_staged() below.
    //
    // Passing a raw device pointer here would segfault in pack_face_cpu().
    // The solver should call exchange_host_staged() instead.
    throw std::runtime_error(
        "[HaloExchange] exchange_device() without USE_CUDA_KERNELS: "
        "use exchange_host_staged() for OpenMP offload builds.");
#else
    throw std::runtime_error("[HaloExchange] exchange_device() requires USE_MPI.");
#endif
}

void HaloExchange::exchange_host_staged(double* host_ptr, int stride,
                                         int plane_stride, int total_size) {
    if (!decomp_.is_parallel()) return;

#ifdef USE_MPI
    int lo_off = Ng_ * plane_stride;           // First interior z-plane
    int hi_off = Nz_local_ * plane_stride;     // Last Ng interior z-planes
    int ghost_len = Ng_ * plane_stride;        // Doubles per Ng z-planes
    int hi_ghost_off = (Nz_local_ + Ng_) * plane_stride;

    // GPU → host: fetch interior z-face planes that we need to send
    #pragma omp target update from(host_ptr[lo_off : ghost_len])
    #pragma omp target update from(host_ptr[hi_off : ghost_len])

    // CPU exchange: pack, MPI send/recv, unpack — all on host memory
    exchange(host_ptr, stride, plane_stride);

    // Host → GPU: push received ghost z-planes back to device
    #pragma omp target update to(host_ptr[0 : ghost_len])
    #pragma omp target update to(host_ptr[hi_ghost_off : ghost_len])

    (void)total_size;  // Used for future bounds checking
#else
    (void)host_ptr; (void)stride; (void)plane_stride; (void)total_size;
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
