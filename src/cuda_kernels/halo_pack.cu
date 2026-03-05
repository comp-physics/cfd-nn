/// @file halo_pack.cu
/// @brief Fused BC kernel and z-face pack/unpack for MPI halo exchange
///
/// The fused BC kernel handles all 6 faces in a single launch, reducing
/// kernel launch overhead compared to 6 separate launches. Each thread
/// determines which face it belongs to via a flat-index decode.
///
/// The pack/unpack kernels extract or insert a single z-plane from a 3D
/// array into a contiguous buffer suitable for MPI_Isend/Irecv.

#include "cuda_halo.hpp"
#include <cuda_runtime.h>

namespace nncfd {
namespace cuda_kernels {

__global__ void apply_bc_3d_fused_kernel(
    double* __restrict__ u,
    int Nx, int Ny, int Nz, int Ng,
    int bc_x_lo, int bc_x_hi,
    int bc_y_lo, int bc_y_hi,
    int bc_z_lo, int bc_z_hi)
{
    const int stride = Nx + 2 * Ng;
    const int plane_stride = stride * (Ny + 2 * Ng);
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Total boundary cells: 2 faces per direction
    const int yz_face = (Ny + 2*Ng) * (Nz + 2*Ng);  // x-faces
    const int xz_face = (Nx + 2*Ng) * (Nz + 2*Ng);  // y-faces
    const int xy_face = (Nx + 2*Ng) * (Ny + 2*Ng);  // z-faces
    const int total = 2 * yz_face + 2 * xz_face + 2 * xy_face;

    if (tid >= total) return;

    int remaining = tid;

    // X-lo face
    if (remaining < yz_face) {
        int j = remaining / (Nz + 2*Ng);
        int k = remaining % (Nz + 2*Ng);
        for (int g = 0; g < Ng; ++g) {
            int idx_ghost = k * plane_stride + j * stride + g;
            if (bc_x_lo == 0) { // periodic
                int idx_src = k * plane_stride + j * stride + (Nx + g);
                u[idx_ghost] = u[idx_src];
            } else if (bc_x_lo == 1) { // neumann
                int idx_src = k * plane_stride + j * stride + Ng;
                u[idx_ghost] = u[idx_src];
            }
        }
        return;
    }
    remaining -= yz_face;

    // X-hi face
    if (remaining < yz_face) {
        int j = remaining / (Nz + 2*Ng);
        int k = remaining % (Nz + 2*Ng);
        for (int g = 0; g < Ng; ++g) {
            int idx_ghost = k * plane_stride + j * stride + (Nx + Ng + g);
            if (bc_x_hi == 0) { // periodic
                int idx_src = k * plane_stride + j * stride + (Ng + g);
                u[idx_ghost] = u[idx_src];
            } else if (bc_x_hi == 1) { // neumann
                int idx_src = k * plane_stride + j * stride + (Nx + Ng - 1);
                u[idx_ghost] = u[idx_src];
            }
        }
        return;
    }
    remaining -= yz_face;

    // Y-lo face
    if (remaining < xz_face) {
        int i = remaining / (Nz + 2*Ng);
        int k = remaining % (Nz + 2*Ng);
        for (int g = 0; g < Ng; ++g) {
            int idx_ghost = k * plane_stride + g * stride + i;
            if (bc_y_lo == 0) {
                int idx_src = k * plane_stride + (Ny + g) * stride + i;
                u[idx_ghost] = u[idx_src];
            } else if (bc_y_lo == 1) {
                int idx_src = k * plane_stride + Ng * stride + i;
                u[idx_ghost] = u[idx_src];
            }
        }
        return;
    }
    remaining -= xz_face;

    // Y-hi face
    if (remaining < xz_face) {
        int i = remaining / (Nz + 2*Ng);
        int k = remaining % (Nz + 2*Ng);
        for (int g = 0; g < Ng; ++g) {
            int idx_ghost = k * plane_stride + (Ny + Ng + g) * stride + i;
            if (bc_y_hi == 0) {
                int idx_src = k * plane_stride + (Ng + g) * stride + i;
                u[idx_ghost] = u[idx_src];
            } else if (bc_y_hi == 1) {
                int idx_src = k * plane_stride + (Ny + Ng - 1) * stride + i;
                u[idx_ghost] = u[idx_src];
            }
        }
        return;
    }
    remaining -= xz_face;

    // Z-lo face
    if (remaining < xy_face) {
        int i = remaining / (Ny + 2*Ng);
        int j = remaining % (Ny + 2*Ng);
        for (int g = 0; g < Ng; ++g) {
            int idx_ghost = g * plane_stride + j * stride + i;
            if (bc_z_lo == 0) {
                int idx_src = (Nz + g) * plane_stride + j * stride + i;
                u[idx_ghost] = u[idx_src];
            } else if (bc_z_lo == 1) {
                int idx_src = Ng * plane_stride + j * stride + i;
                u[idx_ghost] = u[idx_src];
            }
        }
        return;
    }
    remaining -= xy_face;

    // Z-hi face
    if (remaining < xy_face) {
        int i = remaining / (Ny + 2*Ng);
        int j = remaining % (Ny + 2*Ng);
        for (int g = 0; g < Ng; ++g) {
            int idx_ghost = (Nz + Ng + g) * plane_stride + j * stride + i;
            if (bc_z_hi == 0) {
                int idx_src = (Ng + g) * plane_stride + j * stride + i;
                u[idx_ghost] = u[idx_src];
            } else if (bc_z_hi == 1) {
                int idx_src = (Nz + Ng - 1) * plane_stride + j * stride + i;
                u[idx_ghost] = u[idx_src];
            }
        }
    }
}

__global__ void pack_z_face_kernel(
    const double* __restrict__ field,
    double* __restrict__ buffer,
    int Nx, int Ny, int Ng, int stride, int plane_stride,
    int k_src)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < Nx + 2*Ng && j < Ny + 2*Ng) {
        buffer[j * (Nx + 2*Ng) + i] = field[k_src * plane_stride + j * stride + i];
    }
}

__global__ void unpack_z_face_kernel(
    double* __restrict__ field,
    const double* __restrict__ buffer,
    int Nx, int Ny, int Ng, int stride, int plane_stride,
    int k_dst)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < Nx + 2*Ng && j < Ny + 2*Ng) {
        field[k_dst * plane_stride + j * stride + i] = buffer[j * (Nx + 2*Ng) + i];
    }
}

void launch_apply_bc_3d_fused(
    double* u, int Nx, int Ny, int Nz, int Ng,
    int bc_x_lo, int bc_x_hi,
    int bc_y_lo, int bc_y_hi,
    int bc_z_lo, int bc_z_hi,
    void* stream)
{
    cudaStream_t s = stream ? static_cast<cudaStream_t>(stream) : 0;
    const int yz = (Ny + 2*Ng) * (Nz + 2*Ng);
    const int xz = (Nx + 2*Ng) * (Nz + 2*Ng);
    const int xy = (Nx + 2*Ng) * (Ny + 2*Ng);
    const int total = 2*yz + 2*xz + 2*xy;

    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    apply_bc_3d_fused_kernel<<<blocks, threads, 0, s>>>(
        u, Nx, Ny, Nz, Ng,
        bc_x_lo, bc_x_hi, bc_y_lo, bc_y_hi, bc_z_lo, bc_z_hi);
}

void launch_pack_z_face(
    const double* field, double* buffer,
    int Nx, int Ny, int Nz, int Ng,
    bool pack_lo, void* stream)
{
    cudaStream_t s = stream ? static_cast<cudaStream_t>(stream) : 0;
    const int stride = Nx + 2*Ng;
    const int plane_stride = stride * (Ny + 2*Ng);
    int k_src = pack_lo ? Ng : (Nz + Ng - 1);

    dim3 block(16, 16);
    dim3 grid((Nx + 2*Ng + 15)/16, (Ny + 2*Ng + 15)/16);
    pack_z_face_kernel<<<grid, block, 0, s>>>(
        field, buffer, Nx, Ny, Ng, stride, plane_stride, k_src);
}

void launch_unpack_z_face(
    double* field, const double* buffer,
    int Nx, int Ny, int Nz, int Ng,
    bool unpack_lo, void* stream)
{
    cudaStream_t s = stream ? static_cast<cudaStream_t>(stream) : 0;
    const int stride = Nx + 2*Ng;
    const int plane_stride = stride * (Ny + 2*Ng);
    int k_dst = unpack_lo ? 0 : (Nz + 2*Ng - 1);

    dim3 block(16, 16);
    dim3 grid((Nx + 2*Ng + 15)/16, (Ny + 2*Ng + 15)/16);
    unpack_z_face_kernel<<<grid, block, 0, s>>>(
        field, buffer, Nx, Ny, Ng, stride, plane_stride, k_dst);
}

} // namespace cuda_kernels
} // namespace nncfd
