#include "cuda_smoother.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

namespace nncfd {
namespace cuda_kernels {

// Tile dimensions for shared memory
static constexpr int TILE_X = 8;
static constexpr int TILE_Y = 8;
static constexpr int TILE_Z = 8;
// Shared memory tile includes 1-cell halo on each side
static constexpr int SMEM_X = TILE_X + 2;
static constexpr int SMEM_Y = TILE_Y + 2;
static constexpr int SMEM_Z = TILE_Z + 2;

__global__ void chebyshev_3d_smem_kernel(
    double* __restrict__ u,
    const double* __restrict__ f,
    double* __restrict__ tmp,
    int Nx, int Ny, int Nz, int Ng,
    double inv_dx2, double inv_dy2, double inv_dz2,
    double inv_diag, double omega)
{
    // Global indices (interior only)
    const int i = blockIdx.x * TILE_X + threadIdx.x + Ng;
    const int j = blockIdx.y * TILE_Y + threadIdx.y + Ng;
    const int k = blockIdx.z * TILE_Z + threadIdx.z + Ng;

    const int stride = Nx + 2 * Ng;
    const int plane_stride = stride * (Ny + 2 * Ng);

    // Shared memory tile
    __shared__ double s_u[SMEM_Z][SMEM_Y][SMEM_X];

    // Local thread indices in shared memory (offset by 1 for halo)
    const int si = threadIdx.x + 1;
    const int sj = threadIdx.y + 1;
    const int sk = threadIdx.z + 1;

    // Load center point
    bool in_bounds = (i < Nx + Ng && j < Ny + Ng && k < Nz + Ng);
    int idx = k * plane_stride + j * stride + i;

    if (in_bounds) {
        s_u[sk][sj][si] = u[idx];
    }

    // Load halo cells (face neighbors only)
    // X-halos
    if (threadIdx.x == 0 && i > 0) {
        s_u[sk][sj][0] = u[idx - 1];
    }
    if (threadIdx.x == TILE_X - 1 || i == Nx + Ng - 1) {
        if (i + 1 < Nx + 2 * Ng) {
            s_u[sk][sj][si + 1] = u[idx + 1];
        }
    }
    // Y-halos
    if (threadIdx.y == 0 && j > 0) {
        s_u[sk][0][si] = u[idx - stride];
    }
    if (threadIdx.y == TILE_Y - 1 || j == Ny + Ng - 1) {
        if (j + 1 < Ny + 2 * Ng) {
            s_u[sk][sj + 1][si] = u[idx + stride];
        }
    }
    // Z-halos
    if (threadIdx.z == 0 && k > 0) {
        s_u[0][sj][si] = u[idx - plane_stride];
    }
    if (threadIdx.z == TILE_Z - 1 || k == Nz + Ng - 1) {
        if (k + 1 < Nz + 2 * Ng) {
            s_u[sk + 1][sj][si] = u[idx + plane_stride];
        }
    }

    __syncthreads();

    if (in_bounds) {
        double lap = (s_u[sk][sj][si - 1] + s_u[sk][sj][si + 1]) * inv_dx2
                   + (s_u[sk][sj - 1][si] + s_u[sk][sj + 1][si]) * inv_dy2
                   + (s_u[sk - 1][sj][si] + s_u[sk + 1][sj][si]) * inv_dz2;

        double u_jacobi = (lap - f[idx]) * inv_diag;
        tmp[idx] = (1.0 - omega) * s_u[sk][sj][si] + omega * u_jacobi;
    }
}

__global__ void chebyshev_3d_smem_nonuniform_kernel(
    double* __restrict__ u,
    const double* __restrict__ f,
    double* __restrict__ tmp,
    int Nx, int Ny, int Nz, int Ng,
    double inv_dx2, double inv_dz2,
    const double* __restrict__ aS,
    const double* __restrict__ aN,
    const double* __restrict__ aP,
    double omega)
{
    const int i = blockIdx.x * TILE_X + threadIdx.x + Ng;
    const int j = blockIdx.y * TILE_Y + threadIdx.y + Ng;
    const int k = blockIdx.z * TILE_Z + threadIdx.z + Ng;

    const int stride = Nx + 2 * Ng;
    const int plane_stride = stride * (Ny + 2 * Ng);

    __shared__ double s_u[SMEM_Z][SMEM_Y][SMEM_X];

    const int si = threadIdx.x + 1;
    const int sj = threadIdx.y + 1;
    const int sk = threadIdx.z + 1;

    bool in_bounds = (i < Nx + Ng && j < Ny + Ng && k < Nz + Ng);
    int idx = k * plane_stride + j * stride + i;

    if (in_bounds) {
        s_u[sk][sj][si] = u[idx];
    }

    // Same halo loading as uniform kernel
    if (threadIdx.x == 0 && i > 0)
        s_u[sk][sj][0] = u[idx - 1];
    if (threadIdx.x == TILE_X - 1 || i == Nx + Ng - 1)
        if (i + 1 < Nx + 2 * Ng)
            s_u[sk][sj][si + 1] = u[idx + 1];
    if (threadIdx.y == 0 && j > 0)
        s_u[sk][0][si] = u[idx - stride];
    if (threadIdx.y == TILE_Y - 1 || j == Ny + Ng - 1)
        if (j + 1 < Ny + 2 * Ng)
            s_u[sk][sj + 1][si] = u[idx + stride];
    if (threadIdx.z == 0 && k > 0)
        s_u[0][sj][si] = u[idx - plane_stride];
    if (threadIdx.z == TILE_Z - 1 || k == Nz + Ng - 1)
        if (k + 1 < Nz + 2 * Ng)
            s_u[sk + 1][sj][si] = u[idx + plane_stride];

    __syncthreads();

    if (in_bounds) {
        double lap_xz = (s_u[sk][sj][si - 1] + s_u[sk][sj][si + 1]) * inv_dx2
                       + (s_u[sk - 1][sj][si] + s_u[sk + 1][sj][si]) * inv_dz2;

        double lap_y = aS[j] * s_u[sk][sj - 1][si] + aN[j] * s_u[sk][sj + 1][si];

        double inv_diag = -1.0 / (2.0 * inv_dx2 + aP[j] + 2.0 * inv_dz2);
        double u_jacobi = (lap_xz + lap_y - f[idx]) * inv_diag;
        tmp[idx] = (1.0 - omega) * s_u[sk][sj][si] + omega * u_jacobi;
    }
}

__global__ void smem_copy_kernel(double* __restrict__ dst,
                            const double* __restrict__ src,
                            int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        dst[idx] = src[idx];
    }
}

// Host-side launch functions

void launch_chebyshev_3d_smem(
    double* u, const double* f, double* tmp,
    int Nx, int Ny, int Nz, int Ng,
    double inv_dx2, double inv_dy2, double inv_dz2,
    int degree, double lambda_min, double lambda_max,
    bool bc_periodic_x, bool bc_periodic_y, bool bc_periodic_z,
    void* stream)
{
    cudaStream_t s = stream ? static_cast<cudaStream_t>(stream) : 0;

    dim3 block(TILE_X, TILE_Y, TILE_Z);
    dim3 grid((Nx + TILE_X - 1) / TILE_X,
              (Ny + TILE_Y - 1) / TILE_Y,
              (Nz + TILE_Z - 1) / TILE_Z);

    const double diag = -(2.0 * inv_dx2 + 2.0 * inv_dy2 + 2.0 * inv_dz2);
    const double inv_diag = 1.0 / diag;

    const int total_size = (Nx + 2*Ng) * (Ny + 2*Ng) * (Nz + 2*Ng);

    for (int iter = 0; iter < degree; ++iter) {
        // Chebyshev weight for this iteration
        double theta = M_PI * (2.0 * iter + 1.0) / (2.0 * degree);
        double sigma = (lambda_max + lambda_min) / 2.0
                     + (lambda_max - lambda_min) / 2.0 * cos(theta);
        double omega = 1.0 / sigma;

        chebyshev_3d_smem_kernel<<<grid, block, 0, s>>>(
            u, f, tmp, Nx, Ny, Nz, Ng,
            inv_dx2, inv_dy2, inv_dz2,
            inv_diag, omega);

        // Copy tmp -> u for next iteration
        int copy_blocks = (total_size + 255) / 256;
        smem_copy_kernel<<<copy_blocks, 256, 0, s>>>(u, tmp, total_size);
    }
}

void launch_chebyshev_3d_smem_nonuniform(
    double* u, const double* f, double* tmp,
    int Nx, int Ny, int Nz, int Ng,
    double inv_dx2, double inv_dz2,
    const double* aS, const double* aN, const double* aP,
    int degree, double lambda_min, double lambda_max,
    bool bc_periodic_x, bool bc_periodic_y, bool bc_periodic_z,
    void* stream)
{
    cudaStream_t s = stream ? static_cast<cudaStream_t>(stream) : 0;

    dim3 block(TILE_X, TILE_Y, TILE_Z);
    dim3 grid((Nx + TILE_X - 1) / TILE_X,
              (Ny + TILE_Y - 1) / TILE_Y,
              (Nz + TILE_Z - 1) / TILE_Z);

    const int total_size = (Nx + 2*Ng) * (Ny + 2*Ng) * (Nz + 2*Ng);

    for (int iter = 0; iter < degree; ++iter) {
        double theta = M_PI * (2.0 * iter + 1.0) / (2.0 * degree);
        double sigma = (lambda_max + lambda_min) / 2.0
                     + (lambda_max - lambda_min) / 2.0 * cos(theta);
        double omega = 1.0 / sigma;

        chebyshev_3d_smem_nonuniform_kernel<<<grid, block, 0, s>>>(
            u, f, tmp, Nx, Ny, Nz, Ng,
            inv_dx2, inv_dz2,
            aS, aN, aP, omega);

        int copy_blocks = (total_size + 255) / 256;
        smem_copy_kernel<<<copy_blocks, 256, 0, s>>>(u, tmp, total_size);
    }
}

bool cuda_smoother_available() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}

} // namespace cuda_kernels
} // namespace nncfd
