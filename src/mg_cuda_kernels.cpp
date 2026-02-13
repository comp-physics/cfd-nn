/// @file mg_cuda_kernels.cpp
/// @brief CUDA kernel implementation for multigrid Poisson solver
///
/// This file implements CUDA-native kernels for the MG smoother,
/// with CUDA Graph capture support for reduced launch overhead.

#ifdef USE_GPU_OFFLOAD

#include "mg_cuda_kernels.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            throw std::runtime_error(std::string("CUDA error: ") +             \
                                     cudaGetErrorString(err) + " at " +        \
                                     __FILE__ + ":" + std::to_string(__LINE__)); \
        }                                                                      \
    } while (0)

namespace nncfd {
namespace mg_cuda {

// ============================================================================
// Chebyshev Eigenvalue Bounds
// ============================================================================
// Conservative eigenvalue bounds for D^{-1}*A where D = diag(A).
// For the 7-point discrete Laplacian, the true eigenvalues are in (0, 2).
// We use slightly narrower bounds [0.05, 1.95] for numerical stability.
// These ensure Chebyshev acceleration is stable across all grid sizes and BCs.
constexpr double CHEBYSHEV_LAMBDA_MIN = 0.05;
constexpr double CHEBYSHEV_LAMBDA_MAX = 1.95;

// ============================================================================
// CUDA Kernels
// ============================================================================

/// 3D Chebyshev smoother kernel (requires ghost cells to be set)
/// Each thread computes one interior point
__global__ void chebyshev_3d_kernel(
    double* __restrict__ u,
    const double* __restrict__ f,
    double* __restrict__ tmp,
    int Nx, int Ny, int Nz, int Ng,
    double inv_dx2, double inv_dy2, double inv_dz2,
    double inv_coeff, double omega)
{
    // Global thread indices (interior points only)
    int i = blockIdx.x * blockDim.x + threadIdx.x + Ng;
    int j = blockIdx.y * blockDim.y + threadIdx.y + Ng;
    int k = blockIdx.z * blockDim.z + threadIdx.z + Ng;

    if (i < Nx + Ng && j < Ny + Ng && k < Nz + Ng) {
        int stride = Nx + 2 * Ng;
        int plane_stride = stride * (Ny + 2 * Ng);
        int idx = k * plane_stride + j * stride + i;

        // Read neighbors
        double u_c = u[idx];
        double u_xm = u[idx - 1];
        double u_xp = u[idx + 1];
        double u_ym = u[idx - stride];
        double u_yp = u[idx + stride];
        double u_zm = u[idx - plane_stride];
        double u_zp = u[idx + plane_stride];

        // Jacobi update: u_new = (neighbors/h^2 - f) / diag
        double u_jacobi = ((u_xp + u_xm) * inv_dx2 +
                           (u_yp + u_ym) * inv_dy2 +
                           (u_zp + u_zm) * inv_dz2 - f[idx]) * inv_coeff;

        // Chebyshev weighted update
        tmp[idx] = (1.0 - omega) * u_c + omega * u_jacobi;
    }
}

/// 3D Chebyshev smoother with FUSED periodic BCs (no separate BC kernel needed)
/// Uses wrap indexing for periodic boundaries - eliminates BC kernel overhead
__global__ void chebyshev_3d_periodic_kernel(
    double* __restrict__ u,
    const double* __restrict__ f,
    double* __restrict__ tmp,
    int Nx, int Ny, int Nz, int Ng,
    double inv_dx2, double inv_dy2, double inv_dz2,
    double inv_coeff, double omega)
{
    // Global thread indices (interior points: 1..Nx, 1..Ny, 1..Nz with Ng=1)
    int i = blockIdx.x * blockDim.x + threadIdx.x + Ng;
    int j = blockIdx.y * blockDim.y + threadIdx.y + Ng;
    int k = blockIdx.z * blockDim.z + threadIdx.z + Ng;

    if (i < Nx + Ng && j < Ny + Ng && k < Nz + Ng) {
        const int stride = Nx + 2 * Ng;
        const int plane_stride = stride * (Ny + 2 * Ng);
        const int idx = k * plane_stride + j * stride + i;

        // Wrap indices for periodic BCs (avoids separate BC kernel)
        // i ranges from Ng to Nx+Ng-1 (i.e., 1 to Nx for Ng=1)
        int i_m = (i == Ng) ? (Nx + Ng - 1) : (i - 1);       // Wrap left
        int i_p = (i == Nx + Ng - 1) ? Ng : (i + 1);         // Wrap right
        int j_m = (j == Ng) ? (Ny + Ng - 1) : (j - 1);       // Wrap bottom
        int j_p = (j == Ny + Ng - 1) ? Ng : (j + 1);         // Wrap top
        int k_m = (k == Ng) ? (Nz + Ng - 1) : (k - 1);       // Wrap back
        int k_p = (k == Nz + Ng - 1) ? Ng : (k + 1);         // Wrap front

        // Compute neighbor indices with wrap
        int idx_xm = k * plane_stride + j * stride + i_m;
        int idx_xp = k * plane_stride + j * stride + i_p;
        int idx_ym = k * plane_stride + j_m * stride + i;
        int idx_yp = k * plane_stride + j_p * stride + i;
        int idx_zm = k_m * plane_stride + j * stride + i;
        int idx_zp = k_p * plane_stride + j * stride + i;

        // Read neighbors with periodic wrap
        double u_c = u[idx];
        double u_xm = u[idx_xm];
        double u_xp = u[idx_xp];
        double u_ym = u[idx_ym];
        double u_yp = u[idx_yp];
        double u_zm = u[idx_zm];
        double u_zp = u[idx_zp];

        // Jacobi update: u_new = (neighbors/h^2 - f) / diag
        double u_jacobi = ((u_xp + u_xm) * inv_dx2 +
                           (u_yp + u_ym) * inv_dy2 +
                           (u_zp + u_zm) * inv_dz2 - f[idx]) * inv_coeff;

        // Chebyshev weighted update
        tmp[idx] = (1.0 - omega) * u_c + omega * u_jacobi;
    }
}

/// 3D boundary condition kernel
/// Handles periodic, Neumann, and Dirichlet BCs for ghost cells
__global__ void bc_3d_kernel(
    double* __restrict__ u,
    int Nx, int Ny, int Nz, int Ng,
    int bc_x_lo, int bc_x_hi,
    int bc_y_lo, int bc_y_hi,
    int bc_z_lo, int bc_z_hi,
    double dirichlet_val)
{
    int stride = Nx + 2 * Ng;
    int plane_stride = stride * (Ny + 2 * Ng);

    // Thread covers all boundary cells
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_boundary = (Ny + 2*Ng) * (Nz + 2*Ng) * 2 +  // x faces
                         (Nx + 2*Ng) * (Nz + 2*Ng) * 2 +  // y faces
                         (Nx + 2*Ng) * (Ny + 2*Ng) * 2;   // z faces

    if (tid >= total_boundary) return;

    // Decode which boundary face and which cell
    int remaining = tid;
    int face_yz = (Ny + 2*Ng) * (Nz + 2*Ng);
    int face_xz = (Nx + 2*Ng) * (Nz + 2*Ng);
    int face_xy = (Nx + 2*Ng) * (Ny + 2*Ng);

    // X-low face
    if (remaining < face_yz) {
        int jk = remaining;
        int j = jk % (Ny + 2*Ng);
        int k = jk / (Ny + 2*Ng);
        int idx = k * plane_stride + j * stride + 0;
        int idx_int = k * plane_stride + j * stride + Ng;
        int idx_wrap = k * plane_stride + j * stride + (Nx + Ng - 1);

        if (bc_x_lo == 2) { // Periodic
            u[idx] = u[idx_wrap];
        } else if (bc_x_lo == 1) { // Neumann
            u[idx] = u[idx_int];
        } else { // Dirichlet
            u[idx] = 2.0 * dirichlet_val - u[idx_int];
        }
        return;
    }
    remaining -= face_yz;

    // X-high face
    if (remaining < face_yz) {
        int jk = remaining;
        int j = jk % (Ny + 2*Ng);
        int k = jk / (Ny + 2*Ng);
        int idx = k * plane_stride + j * stride + (Nx + Ng);
        int idx_int = k * plane_stride + j * stride + (Nx + Ng - 1);
        int idx_wrap = k * plane_stride + j * stride + Ng;

        if (bc_x_hi == 2) { // Periodic
            u[idx] = u[idx_wrap];
        } else if (bc_x_hi == 1) { // Neumann
            u[idx] = u[idx_int];
        } else { // Dirichlet
            u[idx] = 2.0 * dirichlet_val - u[idx_int];
        }
        return;
    }
    remaining -= face_yz;

    // Y-low face (skip x-edges to avoid race with x-faces)
    if (remaining < face_xz) {
        int ik = remaining;
        int i = ik % (Nx + 2*Ng);
        int k = ik / (Nx + 2*Ng);
        // Skip cells owned by x-faces (all x-ghost layers, robust for any Ng)
        if (i < Ng || i >= Nx + Ng) return;
        int idx = k * plane_stride + 0 * stride + i;
        int idx_int = k * plane_stride + Ng * stride + i;
        int idx_wrap = k * plane_stride + (Ny + Ng - 1) * stride + i;

        if (bc_y_lo == 2) { // Periodic
            u[idx] = u[idx_wrap];
        } else if (bc_y_lo == 1) { // Neumann
            u[idx] = u[idx_int];
        } else { // Dirichlet
            u[idx] = 2.0 * dirichlet_val - u[idx_int];
        }
        return;
    }
    remaining -= face_xz;

    // Y-high face (skip x-edges to avoid race with x-faces)
    if (remaining < face_xz) {
        int ik = remaining;
        int i = ik % (Nx + 2*Ng);
        int k = ik / (Nx + 2*Ng);
        // Skip cells owned by x-faces (all x-ghost layers, robust for any Ng)
        if (i < Ng || i >= Nx + Ng) return;
        int idx = k * plane_stride + (Ny + Ng) * stride + i;
        int idx_int = k * plane_stride + (Ny + Ng - 1) * stride + i;
        int idx_wrap = k * plane_stride + Ng * stride + i;

        if (bc_y_hi == 2) { // Periodic
            u[idx] = u[idx_wrap];
        } else if (bc_y_hi == 1) { // Neumann
            u[idx] = u[idx_int];
        } else { // Dirichlet
            u[idx] = 2.0 * dirichlet_val - u[idx_int];
        }
        return;
    }
    remaining -= face_xz;

    // Z-low face (skip x/y-edges to avoid races)
    if (remaining < face_xy) {
        int ij = remaining;
        int i = ij % (Nx + 2*Ng);
        int j = ij / (Nx + 2*Ng);
        // Skip cells owned by x-faces or y-faces (all ghost layers, robust for any Ng)
        if (i < Ng || i >= Nx + Ng || j < Ng || j >= Ny + Ng) return;
        int idx = 0 * plane_stride + j * stride + i;
        int idx_int = Ng * plane_stride + j * stride + i;
        int idx_wrap = (Nz + Ng - 1) * plane_stride + j * stride + i;

        if (bc_z_lo == 2) { // Periodic
            u[idx] = u[idx_wrap];
        } else if (bc_z_lo == 1) { // Neumann
            u[idx] = u[idx_int];
        } else { // Dirichlet
            u[idx] = 2.0 * dirichlet_val - u[idx_int];
        }
        return;
    }
    remaining -= face_xy;

    // Z-high face (skip x/y-edges to avoid races)
    if (remaining < face_xy) {
        int ij = remaining;
        int i = ij % (Nx + 2*Ng);
        int j = ij / (Nx + 2*Ng);
        // Skip cells owned by x-faces or y-faces (all ghost layers, robust for any Ng)
        if (i < Ng || i >= Nx + Ng || j < Ng || j >= Ny + Ng) return;
        int idx = (Nz + Ng) * plane_stride + j * stride + i;
        int idx_int = (Nz + Ng - 1) * plane_stride + j * stride + i;
        int idx_wrap = Ng * plane_stride + j * stride + i;

        if (bc_z_hi == 2) { // Periodic
            u[idx] = u[idx_wrap];
        } else if (bc_z_hi == 1) { // Neumann
            u[idx] = u[idx_int];
        } else { // Dirichlet
            u[idx] = 2.0 * dirichlet_val - u[idx_int];
        }
        return;
    }
}

/// Simple array copy kernel
__global__ void copy_kernel(double* __restrict__ dst, const double* __restrict__ src, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = src[idx];
    }
}

/// Simple array zero kernel
__global__ void zero_kernel(double* __restrict__ dst, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = 0.0;
    }
}

/// 3D residual computation kernel: r = f - L(u)
/// Each thread computes one interior point
__global__ void residual_3d_kernel(
    const double* __restrict__ u,
    const double* __restrict__ f,
    double* __restrict__ r,
    int Nx, int Ny, int Nz, int Ng,
    double inv_dx2, double inv_dy2, double inv_dz2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + Ng;
    int j = blockIdx.y * blockDim.y + threadIdx.y + Ng;
    int k = blockIdx.z * blockDim.z + threadIdx.z + Ng;

    if (i < Nx + Ng && j < Ny + Ng && k < Nz + Ng) {
        int stride = Nx + 2 * Ng;
        int plane_stride = stride * (Ny + 2 * Ng);
        int idx = k * plane_stride + j * stride + i;

        // Compute Laplacian: L(u) = (u_{i+1} - 2u_i + u_{i-1})/dx^2 + ...
        double laplacian = (u[idx + 1] - 2.0 * u[idx] + u[idx - 1]) * inv_dx2
                         + (u[idx + stride] - 2.0 * u[idx] + u[idx - stride]) * inv_dy2
                         + (u[idx + plane_stride] - 2.0 * u[idx] + u[idx - plane_stride]) * inv_dz2;

        // Residual: r = f - L(u)
        r[idx] = f[idx] - laplacian;
    }
}

/// 3D 27-point full-weighting restriction kernel
/// Each thread computes one coarse grid point
__global__ void restrict_3d_kernel(
    const double* __restrict__ r_fine,
    double* __restrict__ f_coarse,
    int Nx_f, int Ny_f, int Nz_f,
    int Nx_c, int Ny_c, int Nz_c,
    int Ng)
{
    int i_c = blockIdx.x * blockDim.x + threadIdx.x + Ng;
    int j_c = blockIdx.y * blockDim.y + threadIdx.y + Ng;
    int k_c = blockIdx.z * blockDim.z + threadIdx.z + Ng;

    if (i_c < Nx_c + Ng && j_c < Ny_c + Ng && k_c < Nz_c + Ng) {
        int stride_f = Nx_f + 2 * Ng;
        int stride_c = Nx_c + 2 * Ng;
        int plane_stride_f = stride_f * (Ny_f + 2 * Ng);
        int plane_stride_c = stride_c * (Ny_c + 2 * Ng);

        // Map coarse index to fine index
        int i_f = 2 * (i_c - Ng) + Ng;
        int j_f = 2 * (j_c - Ng) + Ng;
        int k_f = 2 * (k_c - Ng) + Ng;

        int idx_c = k_c * plane_stride_c + j_c * stride_c + i_c;
        int idx_f = k_f * plane_stride_f + j_f * stride_f + i_f;

        // 27-point full-weighting stencil
        double sum = 0.0;

        // Center point (weight = 1/8)
        sum += 0.125 * r_fine[idx_f];

        // 6 face neighbors (weight = 1/16 each)
        sum += 0.0625 * (r_fine[idx_f - 1] + r_fine[idx_f + 1]
                       + r_fine[idx_f - stride_f] + r_fine[idx_f + stride_f]
                       + r_fine[idx_f - plane_stride_f] + r_fine[idx_f + plane_stride_f]);

        // 12 edge neighbors (weight = 1/32 each)
        sum += 0.03125 * (r_fine[idx_f - 1 - stride_f] + r_fine[idx_f + 1 - stride_f]
                        + r_fine[idx_f - 1 + stride_f] + r_fine[idx_f + 1 + stride_f]
                        + r_fine[idx_f - 1 - plane_stride_f] + r_fine[idx_f + 1 - plane_stride_f]
                        + r_fine[idx_f - 1 + plane_stride_f] + r_fine[idx_f + 1 + plane_stride_f]
                        + r_fine[idx_f - stride_f - plane_stride_f] + r_fine[idx_f + stride_f - plane_stride_f]
                        + r_fine[idx_f - stride_f + plane_stride_f] + r_fine[idx_f + stride_f + plane_stride_f]);

        // 8 corner neighbors (weight = 1/64 each)
        sum += 0.015625 * (r_fine[idx_f - 1 - stride_f - plane_stride_f] + r_fine[idx_f + 1 - stride_f - plane_stride_f]
                         + r_fine[idx_f - 1 + stride_f - plane_stride_f] + r_fine[idx_f + 1 + stride_f - plane_stride_f]
                         + r_fine[idx_f - 1 - stride_f + plane_stride_f] + r_fine[idx_f + 1 - stride_f + plane_stride_f]
                         + r_fine[idx_f - 1 + stride_f + plane_stride_f] + r_fine[idx_f + 1 + stride_f + plane_stride_f]);

        f_coarse[idx_c] = sum;
    }
}

/// 3D trilinear prolongation kernel
/// Each thread computes one fine grid point (owner-computes pattern)
__global__ void prolongate_3d_kernel(
    const double* __restrict__ u_coarse,
    double* __restrict__ u_fine,
    int Nx_f, int Ny_f, int Nz_f,
    int Nx_c, int Ny_c, int Nz_c,
    int Ng)
{
    int i_f = blockIdx.x * blockDim.x + threadIdx.x + Ng;
    int j_f = blockIdx.y * blockDim.y + threadIdx.y + Ng;
    int k_f = blockIdx.z * blockDim.z + threadIdx.z + Ng;

    if (i_f < Nx_f + Ng && j_f < Ny_f + Ng && k_f < Nz_f + Ng) {
        int stride_f = Nx_f + 2 * Ng;
        int stride_c = Nx_c + 2 * Ng;
        int plane_stride_f = stride_f * (Ny_f + 2 * Ng);
        int plane_stride_c = stride_c * (Ny_c + 2 * Ng);

        // Find base coarse cell and position within coarse cell pair
        int i_c = (i_f - Ng) / 2 + Ng;
        int j_c = (j_f - Ng) / 2 + Ng;
        int k_c = (k_f - Ng) / 2 + Ng;
        int di = (i_f - Ng) & 1;  // 0 = coincident, 1 = midpoint
        int dj = (j_f - Ng) & 1;
        int dk = (k_f - Ng) & 1;

        // Interpolation weights
        double wx1 = 0.5 * di;
        double wx0 = 1.0 - wx1;
        double wy1 = 0.5 * dj;
        double wy0 = 1.0 - wy1;
        double wz1 = 0.5 * dk;
        double wz0 = 1.0 - wz1;

        int idx_c = k_c * plane_stride_c + j_c * stride_c + i_c;

        // Trilinear interpolation from 8 coarse neighbors
        double correction =
            wx0 * wy0 * wz0 * u_coarse[idx_c]
          + wx1 * wy0 * wz0 * u_coarse[idx_c + 1]
          + wx0 * wy1 * wz0 * u_coarse[idx_c + stride_c]
          + wx1 * wy1 * wz0 * u_coarse[idx_c + 1 + stride_c]
          + wx0 * wy0 * wz1 * u_coarse[idx_c + plane_stride_c]
          + wx1 * wy0 * wz1 * u_coarse[idx_c + 1 + plane_stride_c]
          + wx0 * wy1 * wz1 * u_coarse[idx_c + stride_c + plane_stride_c]
          + wx1 * wy1 * wz1 * u_coarse[idx_c + 1 + stride_c + plane_stride_c];

        int idx_f = k_f * plane_stride_f + j_f * stride_f + i_f;
        u_fine[idx_f] += correction;
    }
}

// ============================================================================
// Kernel Launch Functions
// ============================================================================

void launch_chebyshev_3d(
    cudaStream_t stream,
    double* u, const double* f, double* tmp,
    int Nx, int Ny, int Nz, int Ng,
    double dx2, double dy2, double dz2, double coeff,
    double omega)
{
    // Compute inverse values for efficiency
    double inv_dx2 = 1.0 / dx2;
    double inv_dy2 = 1.0 / dy2;
    // For 2D (Nz=1), zero out z-direction contribution to stencil
    double inv_dz2 = (Nz == 1) ? 0.0 : 1.0 / dz2;
    double inv_coeff = 1.0 / coeff;

    // Thread block and grid dimensions
    dim3 block(8, 8, 8);  // 512 threads per block
    dim3 grid((Nx + block.x - 1) / block.x,
              (Ny + block.y - 1) / block.y,
              (Nz + block.z - 1) / block.z);

    chebyshev_3d_kernel<<<grid, block, 0, stream>>>(
        u, f, tmp, Nx, Ny, Nz, Ng,
        inv_dx2, inv_dy2, inv_dz2, inv_coeff, omega);
}

/// Launch Chebyshev kernel with fused periodic BCs (no separate BC kernel needed)
void launch_chebyshev_3d_periodic(
    cudaStream_t stream,
    double* u, const double* f, double* tmp,
    int Nx, int Ny, int Nz, int Ng,
    double dx2, double dy2, double dz2, double coeff,
    double omega)
{
    double inv_dx2 = 1.0 / dx2;
    double inv_dy2 = 1.0 / dy2;
    double inv_dz2 = (Nz == 1) ? 0.0 : 1.0 / dz2;
    double inv_coeff = 1.0 / coeff;

    dim3 block(8, 8, 8);
    dim3 grid((Nx + block.x - 1) / block.x,
              (Ny + block.y - 1) / block.y,
              (Nz + block.z - 1) / block.z);

    chebyshev_3d_periodic_kernel<<<grid, block, 0, stream>>>(
        u, f, tmp, Nx, Ny, Nz, Ng,
        inv_dx2, inv_dy2, inv_dz2, inv_coeff, omega);
}

void launch_bc_3d(
    cudaStream_t stream,
    double* u,
    int Nx, int Ny, int Nz, int Ng,
    BC bc_x_lo, BC bc_x_hi,
    BC bc_y_lo, BC bc_y_hi,
    BC bc_z_lo, BC bc_z_hi,
    double dirichlet_val)
{
    int total_boundary = (Ny + 2*Ng) * (Nz + 2*Ng) * 2 +
                         (Nx + 2*Ng) * (Nz + 2*Ng) * 2 +
                         (Nx + 2*Ng) * (Ny + 2*Ng) * 2;

    int block_size = 256;
    int grid_size = (total_boundary + block_size - 1) / block_size;

    bc_3d_kernel<<<grid_size, block_size, 0, stream>>>(
        u, Nx, Ny, Nz, Ng,
        static_cast<int>(bc_x_lo), static_cast<int>(bc_x_hi),
        static_cast<int>(bc_y_lo), static_cast<int>(bc_y_hi),
        static_cast<int>(bc_z_lo), static_cast<int>(bc_z_hi),
        dirichlet_val);
}

void launch_copy(cudaStream_t stream, double* dst, const double* src, size_t size) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    copy_kernel<<<grid_size, block_size, 0, stream>>>(dst, src, size);
}

void launch_zero(cudaStream_t stream, double* dst, size_t size) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    zero_kernel<<<grid_size, block_size, 0, stream>>>(dst, size);
}

void launch_residual_3d(
    cudaStream_t stream,
    const double* u, const double* f, double* r,
    int Nx, int Ny, int Nz, int Ng,
    double inv_dx2, double inv_dy2, double inv_dz2)
{
    dim3 block(8, 8, 8);
    dim3 grid((Nx + block.x - 1) / block.x,
              (Ny + block.y - 1) / block.y,
              (Nz + block.z - 1) / block.z);

    residual_3d_kernel<<<grid, block, 0, stream>>>(
        u, f, r, Nx, Ny, Nz, Ng, inv_dx2, inv_dy2, inv_dz2);
}

void launch_restrict_3d(
    cudaStream_t stream,
    const double* r_fine, double* f_coarse,
    int Nx_f, int Ny_f, int Nz_f,
    int Nx_c, int Ny_c, int Nz_c,
    int Ng)
{
    dim3 block(8, 8, 8);
    dim3 grid((Nx_c + block.x - 1) / block.x,
              (Ny_c + block.y - 1) / block.y,
              (Nz_c + block.z - 1) / block.z);

    restrict_3d_kernel<<<grid, block, 0, stream>>>(
        r_fine, f_coarse, Nx_f, Ny_f, Nz_f, Nx_c, Ny_c, Nz_c, Ng);
}

void launch_prolongate_3d(
    cudaStream_t stream,
    const double* u_coarse, double* u_fine,
    int Nx_f, int Ny_f, int Nz_f,
    int Nx_c, int Ny_c, int Nz_c,
    int Ng)
{
    dim3 block(8, 8, 8);
    dim3 grid((Nx_f + block.x - 1) / block.x,
              (Ny_f + block.y - 1) / block.y,
              (Nz_f + block.z - 1) / block.z);

    prolongate_3d_kernel<<<grid, block, 0, stream>>>(
        u_coarse, u_fine, Nx_f, Ny_f, Nz_f, Nx_c, Ny_c, Nz_c, Ng);
}

// ============================================================================
// CudaSmootherGraph Implementation
// ============================================================================

CudaSmootherGraph::~CudaSmootherGraph() {
    destroy();
}

void CudaSmootherGraph::destroy() {
    if (graph_exec_) {
        cudaGraphExecDestroy(graph_exec_);
        graph_exec_ = nullptr;
    }
    if (graph_) {
        cudaGraphDestroy(graph_);
        graph_ = nullptr;
    }
}

void CudaSmootherGraph::initialize(const LevelConfig& config, int degree,
                                    BC bc_x_lo, BC bc_x_hi,
                                    BC bc_y_lo, BC bc_y_hi,
                                    BC bc_z_lo, BC bc_z_hi) {
    destroy();  // Clean up any existing graph

    config_ = config;
    degree_ = degree;
    bc_x_lo_ = bc_x_lo;
    bc_x_hi_ = bc_x_hi;
    bc_y_lo_ = bc_y_lo;
    bc_y_hi_ = bc_y_hi;
    bc_z_lo_ = bc_z_lo;
    bc_z_hi_ = bc_z_hi;

    // Create a temporary stream for graph capture
    cudaStream_t capture_stream;
    CUDA_CHECK(cudaStreamCreate(&capture_stream));

    capture_graph(capture_stream);

    CUDA_CHECK(cudaStreamDestroy(capture_stream));
}

void CudaSmootherGraph::capture_graph(cudaStream_t stream) {
    // Chebyshev eigenvalue bounds (see constants at top of file)
    const double d = (CHEBYSHEV_LAMBDA_MAX + CHEBYSHEV_LAMBDA_MIN) / 2.0;
    const double c = (CHEBYSHEV_LAMBDA_MAX - CHEBYSHEV_LAMBDA_MIN) / 2.0;

    // Check if all BCs are periodic - can use fused kernel (no BC pass needed)
    const bool all_periodic = (bc_x_lo_ == BC::Periodic && bc_x_hi_ == BC::Periodic &&
                                bc_y_lo_ == BC::Periodic && bc_y_hi_ == BC::Periodic &&
                                bc_z_lo_ == BC::Periodic && bc_z_hi_ == BC::Periodic);

    // Begin stream capture
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    // Capture the Chebyshev smoother sequence
    for (int k = 0; k < degree_; ++k) {
        // Chebyshev-optimal weight
        double theta = M_PI * (2.0 * k + 1.0) / (2.0 * degree_);
        double omega = 1.0 / (d - c * std::cos(theta));

        if (all_periodic) {
            // Fused kernel: Chebyshev + periodic BC wrap (no separate BC pass)
            launch_chebyshev_periodic(stream, omega);
        } else {
            // Standard path: BC kernel + Chebyshev kernel
            launch_bc_kernel(stream);
            launch_chebyshev_iteration(stream, k, omega);
        }

        // Copy: tmp -> u
        launch_copy_kernel(stream);
    }

    // Final BC application - ALWAYS needed for MG operations outside the smoother
    // Even with fused periodic kernel, compute_residual/restrict/prolongate read ghost cells
    // via standard neighbor indexing (idx-1, idx+stride, etc.)
    launch_bc_kernel(stream);

    // End capture and create executable graph
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph_));
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0));
}

void CudaSmootherGraph::launch_chebyshev_iteration(cudaStream_t stream, int k, double omega) {
    launch_chebyshev_3d(
        stream,
        config_.u, config_.f, config_.tmp,
        config_.Nx, config_.Ny, config_.Nz, config_.Ng,
        config_.dx2, config_.dy2, config_.dz2, config_.coeff,
        omega);
}

void CudaSmootherGraph::launch_chebyshev_periodic(cudaStream_t stream, double omega) {
    launch_chebyshev_3d_periodic(
        stream,
        config_.u, config_.f, config_.tmp,
        config_.Nx, config_.Ny, config_.Nz, config_.Ng,
        config_.dx2, config_.dy2, config_.dz2, config_.coeff,
        omega);
}

void CudaSmootherGraph::launch_bc_kernel(cudaStream_t stream) {
    launch_bc_3d(
        stream,
        config_.u,
        config_.Nx, config_.Ny, config_.Nz, config_.Ng,
        bc_x_lo_, bc_x_hi_,
        bc_y_lo_, bc_y_hi_,
        bc_z_lo_, bc_z_hi_,
        0.0);  // Dirichlet value (0 for pressure)
}

void CudaSmootherGraph::launch_copy_kernel(cudaStream_t stream) {
    launch_copy(stream, config_.u, config_.tmp, config_.total_size);
}

void CudaSmootherGraph::execute(cudaStream_t stream) {
    if (graph_exec_) {
        CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream));
    }
}

void CudaSmootherGraph::debug_print_pointers() const {
    std::cerr << "[Graph] Captured pointers: u=" << config_.u
              << " f=" << config_.f << " tmp=" << config_.tmp
              << " size=" << config_.total_size << "\n";
}

// ============================================================================
// CudaMGContext Implementation
// ============================================================================

CudaMGContext::CudaMGContext() {
    CUDA_CHECK(cudaStreamCreate(&stream_));
}

CudaMGContext::~CudaMGContext() {
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

void CudaMGContext::synchronize() {
    if (stream_) {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
}

void CudaMGContext::initialize_smoother_graphs(
    const std::vector<LevelConfig>& levels,
    int degree,
    BC bc_x_lo, BC bc_x_hi,
    BC bc_y_lo, BC bc_y_hi,
    BC bc_z_lo, BC bc_z_hi)
{
    smoother_graphs_.resize(levels.size());
    for (size_t i = 0; i < levels.size(); ++i) {
        smoother_graphs_[i].initialize(
            levels[i], degree,
            bc_x_lo, bc_x_hi,
            bc_y_lo, bc_y_hi,
            bc_z_lo, bc_z_hi);
    }
}

void CudaMGContext::smooth(int level) {
    if (level >= 0 && level < static_cast<int>(smoother_graphs_.size())) {
        smoother_graphs_[level].execute(stream_);
    }
}

void CudaMGContext::smooth(int level, cudaStream_t stream) {
    if (level >= 0 && level < static_cast<int>(smoother_graphs_.size())) {
        smoother_graphs_[level].execute(stream);
    }
}

void CudaMGContext::debug_graph_pointers(int level) const {
    if (level >= 0 && level < static_cast<int>(smoother_graphs_.size())) {
        smoother_graphs_[level].debug_print_pointers();
    }
}

// ============================================================================
// CudaVCycleGraph Implementation - Full V-cycle in a single graph
// ============================================================================

CudaVCycleGraph::~CudaVCycleGraph() {
    destroy();
}

void CudaVCycleGraph::destroy() {
    if (graph_exec_) {
        cudaGraphExecDestroy(graph_exec_);
        graph_exec_ = nullptr;
    }
    if (graph_) {
        cudaGraphDestroy(graph_);
        graph_ = nullptr;
    }
}

void CudaVCycleGraph::initialize(
    const std::vector<VCycleLevelConfig>& levels,
    int degree, int nu1, int nu2,
    BC bc_x_lo, BC bc_x_hi,
    BC bc_y_lo, BC bc_y_hi,
    BC bc_z_lo, BC bc_z_hi)
{
    destroy();  // Clean up any existing graph

    levels_ = levels;
    degree_ = degree;
    nu1_ = nu1;
    nu2_ = nu2;
    bc_x_lo_ = bc_x_lo;
    bc_x_hi_ = bc_x_hi;
    bc_y_lo_ = bc_y_lo;
    bc_y_hi_ = bc_y_hi;
    bc_z_lo_ = bc_z_lo;
    bc_z_hi_ = bc_z_hi;

    // Store fingerprint for validity checking
    fingerprint_.num_levels = levels.size();
    fingerprint_.level_sizes.clear();
    fingerprint_.level_coeffs.clear();
    fingerprint_.level_dx.clear();
    fingerprint_.level_dy.clear();
    fingerprint_.level_dz.clear();
    for (const auto& lvl : levels) {
        fingerprint_.level_sizes.push_back(lvl.total_size);
        fingerprint_.level_coeffs.push_back(lvl.coeff);
        fingerprint_.level_dx.push_back(lvl.dx2);  // Store dx^2 (what kernels use)
        fingerprint_.level_dy.push_back(lvl.dy2);
        fingerprint_.level_dz.push_back(lvl.dz2);
    }
    fingerprint_.degree = degree;
    fingerprint_.nu1 = nu1;
    fingerprint_.nu2 = nu2;
    fingerprint_.bc_x_lo = bc_x_lo;
    fingerprint_.bc_x_hi = bc_x_hi;
    fingerprint_.bc_y_lo = bc_y_lo;
    fingerprint_.bc_y_hi = bc_y_hi;
    fingerprint_.bc_z_lo = bc_z_lo;
    fingerprint_.bc_z_hi = bc_z_hi;
    fingerprint_.coarse_iters = 8;  // Hardcoded for now

    // Check if all BCs are periodic - use fused smoother kernel
    all_periodic_ = (bc_x_lo == BC::Periodic && bc_x_hi == BC::Periodic &&
                     bc_y_lo == BC::Periodic && bc_y_hi == BC::Periodic &&
                     bc_z_lo == BC::Periodic && bc_z_hi == BC::Periodic);

    // Create a temporary stream for graph capture
    cudaStream_t capture_stream;
    CUDA_CHECK(cudaStreamCreate(&capture_stream));

    capture_vcycle_graph(capture_stream);

    CUDA_CHECK(cudaStreamDestroy(capture_stream));
}

void CudaVCycleGraph::capture_vcycle_graph(cudaStream_t stream) {
    // Begin stream capture
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    // Capture the full V-cycle starting from level 0
    capture_vcycle_level(stream, 0);

    // End capture and create executable graph
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph_));
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0));
}

void CudaVCycleGraph::capture_vcycle_level(cudaStream_t stream, int level) {
    const auto& cfg = levels_[level];
    const int num_levels = static_cast<int>(levels_.size());

    if (level == num_levels - 1) {
        // Coarsest level - just smooth more
        // Match non-graphed path: coarsest uses degree*2 iterations
        capture_smoother(stream, level, std::max(8, degree_ * 2));
        return;
    }

    // Pre-smoothing: nu1 passes of degree iterations each = nu1*degree total
    // This matches the non-graphed vcycle() which calls smooth_chebyshev(level, degree) nu1 times
    capture_smoother(stream, level, nu1_ * degree_);

    // Compute residual: r = f - L(u)
    launch_residual_3d(stream, cfg.u, cfg.f, cfg.r,
                       cfg.Nx, cfg.Ny, cfg.Nz, cfg.Ng,
                       cfg.inv_dx2, cfg.inv_dy2, cfg.inv_dz2);

    // Apply BCs to residual for proper restriction
    launch_bc_3d(stream, cfg.r, cfg.Nx, cfg.Ny, cfg.Nz, cfg.Ng,
                 bc_x_lo_, bc_x_hi_, bc_y_lo_, bc_y_hi_, bc_z_lo_, bc_z_hi_, 0.0);

    // Restrict residual to coarse grid
    const auto& coarse = levels_[level + 1];
    launch_restrict_3d(stream, cfg.r, coarse.f,
                       cfg.Nx, cfg.Ny, cfg.Nz,
                       coarse.Nx, coarse.Ny, coarse.Nz, cfg.Ng);

    // Zero coarse solution
    launch_zero(stream, coarse.u, coarse.total_size);

    // Recursive call to coarser level
    capture_vcycle_level(stream, level + 1);

    // Prolongate correction: u_fine += interp(u_coarse)
    launch_prolongate_3d(stream, coarse.u, cfg.u,
                         cfg.Nx, cfg.Ny, cfg.Nz,
                         coarse.Nx, coarse.Ny, coarse.Nz, cfg.Ng);

    // Apply BCs after prolongation
    launch_bc_3d(stream, cfg.u, cfg.Nx, cfg.Ny, cfg.Nz, cfg.Ng,
                 bc_x_lo_, bc_x_hi_, bc_y_lo_, bc_y_hi_, bc_z_lo_, bc_z_hi_, 0.0);

    // Post-smoothing: nu2 passes of degree iterations each = nu2*degree total
    capture_smoother(stream, level, nu2_ * degree_);
}

void CudaVCycleGraph::capture_smoother(cudaStream_t stream, int level, int iterations) {
    const auto& cfg = levels_[level];

    // Chebyshev eigenvalue bounds (see constants at top of file)
    const double d = (CHEBYSHEV_LAMBDA_MAX + CHEBYSHEV_LAMBDA_MIN) / 2.0;
    const double c = (CHEBYSHEV_LAMBDA_MAX - CHEBYSHEV_LAMBDA_MIN) / 2.0;

    // Capture the Chebyshev smoother sequence
    for (int k = 0; k < iterations; ++k) {
        // Chebyshev-optimal weight
        double theta = M_PI * (2.0 * k + 1.0) / (2.0 * iterations);
        double omega = 1.0 / (d - c * std::cos(theta));

        if (all_periodic_) {
            // Fused kernel with periodic wrap
            launch_chebyshev_3d_periodic(stream, cfg.u, cfg.f, cfg.tmp,
                                         cfg.Nx, cfg.Ny, cfg.Nz, cfg.Ng,
                                         cfg.dx2, cfg.dy2, cfg.dz2, cfg.coeff, omega);
        } else {
            // BC + smoother
            launch_bc_3d(stream, cfg.u, cfg.Nx, cfg.Ny, cfg.Nz, cfg.Ng,
                         bc_x_lo_, bc_x_hi_, bc_y_lo_, bc_y_hi_, bc_z_lo_, bc_z_hi_, 0.0);
            launch_chebyshev_3d(stream, cfg.u, cfg.f, cfg.tmp,
                                cfg.Nx, cfg.Ny, cfg.Nz, cfg.Ng,
                                cfg.dx2, cfg.dy2, cfg.dz2, cfg.coeff, omega);
        }

        // Copy tmp -> u
        launch_copy(stream, cfg.u, cfg.tmp, cfg.total_size);
    }

    // Final BC application
    launch_bc_3d(stream, cfg.u, cfg.Nx, cfg.Ny, cfg.Nz, cfg.Ng,
                 bc_x_lo_, bc_x_hi_, bc_y_lo_, bc_y_hi_, bc_z_lo_, bc_z_hi_, 0.0);
}

void CudaVCycleGraph::execute(cudaStream_t stream) {
    if (graph_exec_) {
        CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream));
    }
}

} // namespace mg_cuda
} // namespace nncfd

#endif // USE_GPU_OFFLOAD
