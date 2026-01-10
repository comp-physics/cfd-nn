#ifdef USE_GPU_OFFLOAD

#include "poisson_solver_fft1d.hpp"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#include <cuda_runtime.h>
#endif

namespace nncfd {

// ============================================================================
// CUDA Kernels
// ============================================================================

#ifdef USE_GPU_OFFLOAD

// Pack kernel: ghost layout -> packed x-lines
// Thread mapping: tid indexes packed array linearly, decompose to (i, j, k)
// Coalesced: consecutive threads vary i first
__global__ void kernel_pack_ghost_to_lines(
    const double* __restrict__ rhs_ghost,
    double* __restrict__ in_pack,
    double* __restrict__ partial_sums,
    int Nx, int Ny, int Nz,
    int stride, int plane_stride)
{
    extern __shared__ double sdata[];

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = Nx * Ny * Nz;

    double local_sum = 0.0;

    if (tid < total) {
        const int i = tid % Nx;
        const int b = tid / Nx;      // b = j*Nz + k
        const int j = b / Nz;
        const int k = b - j * Nz;

        // Ghost index: (k+1)*plane_stride + (j+1)*stride + (i+1)
        const size_t g = (size_t)(k + 1) * (size_t)plane_stride
                       + (size_t)(j + 1) * (size_t)stride
                       + (size_t)(i + 1);

        double val = rhs_ghost[g];

        // Packed index: b*Nx + i
        in_pack[(size_t)b * (size_t)Nx + (size_t)i] = val;
        local_sum = val;
    }

    // Block reduction for mean computation
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// Final reduction kernel
__global__ void kernel_reduce_sum(
    const double* __restrict__ partial_sums,
    double* __restrict__ sum_out,
    int num_blocks)
{
    extern __shared__ double sdata[];

    double sum = 0.0;
    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        sum += partial_sums[i];
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        sum_out[0] = sdata[0];
    }
}

// Subtract mean from packed array
__global__ void kernel_subtract_mean(
    double* __restrict__ in_pack,
    const double* __restrict__ sum_dev,
    int total)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total) {
        double mean = sum_dev[0] / (double)total;
        in_pack[tid] -= mean;
    }
}

// Unpack kernel: packed x-lines -> ghost layout with normalization
__global__ void kernel_unpack_lines_to_ghost(
    const double* __restrict__ out_pack,
    double* __restrict__ p_ghost,
    int Nx, int Ny, int Nz,
    int stride, int plane_stride,
    double invNx)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = Nx * Ny * Nz;
    if (tid >= total) return;

    const int i = tid % Nx;
    const int b = tid / Nx;      // b = j*Nz + k
    const int j = b / Nz;
    const int k = b - j * Nz;

    const size_t g = (size_t)(k + 1) * (size_t)plane_stride
                   + (size_t)(j + 1) * (size_t)stride
                   + (size_t)(i + 1);

    p_ghost[g] = out_pack[(size_t)b * (size_t)Nx + (size_t)i] * invNx;
}

// Pack kernel for z-periodic: ghost layout -> packed z-lines
// Thread mapping: tid indexes packed array linearly, decompose to (k, i, j)
// Packed layout: in_pack[b*Nz + k] where b = i*Ny + j
__global__ void kernel_pack_ghost_to_zlines(
    const double* __restrict__ rhs_ghost,
    double* __restrict__ in_pack,
    double* __restrict__ partial_sums,
    int Nx, int Ny, int Nz,
    int stride, int plane_stride)
{
    extern __shared__ double sdata[];

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = Nx * Ny * Nz;

    double local_sum = 0.0;

    if (tid < total) {
        // For z-periodic: pack z-lines, so k varies fastest in packed array
        // Packed index: b*Nz + k where b = i*Ny + j
        const int k = tid % Nz;
        const int b = tid / Nz;      // b = i*Ny + j
        const int j = b % Ny;
        const int i = b / Ny;

        // Ghost index: (k+1)*plane_stride + (j+1)*stride + (i+1)
        const size_t g = (size_t)(k + 1) * (size_t)plane_stride
                       + (size_t)(j + 1) * (size_t)stride
                       + (size_t)(i + 1);

        double val = rhs_ghost[g];

        // Packed index: b*Nz + k
        in_pack[(size_t)b * (size_t)Nz + (size_t)k] = val;
        local_sum = val;
    }

    // Block reduction for mean computation
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// Unpack kernel for z-periodic: packed z-lines -> ghost layout with normalization
__global__ void kernel_unpack_zlines_to_ghost(
    const double* __restrict__ out_pack,
    double* __restrict__ p_ghost,
    int Nx, int Ny, int Nz,
    int stride, int plane_stride,
    double invNz)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = Nx * Ny * Nz;
    if (tid >= total) return;

    // For z-periodic: unpack z-lines, so k varies fastest in packed array
    const int k = tid % Nz;
    const int b = tid / Nz;      // b = i*Ny + j
    const int j = b % Ny;
    const int i = b / Ny;

    const size_t g = (size_t)(k + 1) * (size_t)plane_stride
                   + (size_t)(j + 1) * (size_t)stride
                   + (size_t)(i + 1);

    p_ghost[g] = out_pack[(size_t)b * (size_t)Nz + (size_t)k] * invNz;
}

// 2D Helmholtz Jacobi iteration kernel
// Processes all modes in parallel
// Layout: p_hat[m * N_yz + j * Nz + k] where k is fastest
// For Neumann BCs: fold ghost contribution into diagonal
__global__ void kernel_helmholtz_jacobi_2d(
    const double* __restrict__ rhs_real,
    const double* __restrict__ rhs_imag,
    const double* __restrict__ p_real_in,
    const double* __restrict__ p_imag_in,
    double* __restrict__ p_real_out,
    double* __restrict__ p_imag_out,
    const double* __restrict__ lambda,
    int N_modes, int Ny, int Nz,
    double ay, double az,
    double omega,
    int bc_y_lo, int bc_y_hi,
    int bc_z_lo, int bc_z_hi)
{
    // Thread indexes into [m, j, k] space
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N_modes * Ny * Nz;
    if (tid >= total) return;

    const int k = tid % Nz;
    const int rem = tid / Nz;
    const int j = rem % Ny;
    const int m = rem / Ny;

    const int N_yz = Ny * Nz;
    const int idx = m * N_yz + j * Nz + k;

    // Compute diagonal (SPD form)
    // Interior: D = 2*ay + 2*az + lambda[m]
    double D = 2.0 * ay + 2.0 * az + lambda[m];

    // Boundary adjustments (fold ghost into diagonal)
    // Neumann: reduce D by missing neighbor coefficient
    bool at_y_lo = (j == 0);
    bool at_y_hi = (j == Ny - 1);
    bool at_z_lo = (k == 0);
    bool at_z_hi = (k == Nz - 1);

    if (at_y_lo && bc_y_lo == 1) D -= ay;  // Neumann
    if (at_y_hi && bc_y_hi == 1) D -= ay;
    if (at_z_lo && bc_z_lo == 1) D -= az;
    if (at_z_hi && bc_z_hi == 1) D -= az;

    // For m=0 with all-Neumann, D could be very small -> pin p[0,0]=0
    // This is handled separately; here we just ensure D > 0
    if (D < 1e-14) D = 1e-14;

    double inv_D = 1.0 / D;

    // Compute Ap for both real and imaginary parts
    // A = D*I - ay*(N+S) - az*(E+W) in SPD form
    // But we compute: residual = b - Ap, update = omega * residual / D

    double p_r = p_real_in[idx];
    double p_i = p_imag_in[idx];

    // Neighbor contributions (handle boundaries)
    double sum_r = 0.0, sum_i = 0.0;

    // South (j-1)
    if (j > 0) {
        int idx_s = m * N_yz + (j-1) * Nz + k;
        sum_r += ay * p_real_in[idx_s];
        sum_i += ay * p_imag_in[idx_s];
    }
    // North (j+1)
    if (j < Ny - 1) {
        int idx_n = m * N_yz + (j+1) * Nz + k;
        sum_r += ay * p_real_in[idx_n];
        sum_i += ay * p_imag_in[idx_n];
    }
    // West (k-1)
    if (k > 0) {
        int idx_w = m * N_yz + j * Nz + (k-1);
        sum_r += az * p_real_in[idx_w];
        sum_i += az * p_imag_in[idx_w];
    }
    // East (k+1)
    if (k < Nz - 1) {
        int idx_e = m * N_yz + j * Nz + (k+1);
        sum_r += az * p_real_in[idx_e];
        sum_i += az * p_imag_in[idx_e];
    }

    // Ap = D*p - sum_neighbors
    double Ap_r = D * p_r - sum_r;
    double Ap_i = D * p_i - sum_i;

    // Residual = b - Ap
    double res_r = rhs_real[idx] - Ap_r;
    double res_i = rhs_imag[idx] - Ap_i;

    // Jacobi update: p_new = p + omega * res / D
    p_real_out[idx] = p_r + omega * res_r * inv_D;
    p_imag_out[idx] = p_i + omega * res_i * inv_D;
}

// Pin m=0, (j=0, k=0) to zero for gauge
__global__ void kernel_pin_zero_mode(
    double* __restrict__ p_real,
    double* __restrict__ p_imag,
    int N_yz)
{
    // Mode m=0 is at index 0..N_yz-1
    // Pin (j=0, k=0) which is index 0
    p_real[0] = 0.0;
    p_imag[0] = 0.0;
}

// 2D Helmholtz Jacobi iteration for m=0 mode ONLY (needs more iterations)
// This mode has λ=0, making it a pure 2D Poisson with slow convergence
__global__ void kernel_helmholtz_jacobi_m0_only(
    const double* __restrict__ rhs_real,
    const double* __restrict__ rhs_imag,
    const double* __restrict__ p_real_in,
    const double* __restrict__ p_imag_in,
    double* __restrict__ p_real_out,
    double* __restrict__ p_imag_out,
    int Ny, int Nz,
    double ay, double az,
    double omega,
    int bc_y_lo, int bc_y_hi,
    int bc_z_lo, int bc_z_hi)
{
    // Thread indexes into [j, k] space for m=0 only
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int N_yz = Ny * Nz;
    if (tid >= N_yz) return;

    const int k = tid % Nz;
    const int j = tid / Nz;
    const int idx = tid;  // For m=0, idx = j*Nz + k

    // Diagonal for m=0: D = 2*ay + 2*az (no lambda shift)
    double D = 2.0 * ay + 2.0 * az;

    // Boundary adjustments
    bool at_y_lo = (j == 0);
    bool at_y_hi = (j == Ny - 1);
    bool at_z_lo = (k == 0);
    bool at_z_hi = (k == Nz - 1);

    if (at_y_lo && bc_y_lo == 1) D -= ay;  // Neumann
    if (at_y_hi && bc_y_hi == 1) D -= ay;
    if (at_z_lo && bc_z_lo == 1) D -= az;
    if (at_z_hi && bc_z_hi == 1) D -= az;

    if (D < 1e-14) D = 1e-14;
    double inv_D = 1.0 / D;

    double p_r = p_real_in[idx];
    double p_i = p_imag_in[idx];

    // Neighbor contributions
    double sum_r = 0.0, sum_i = 0.0;

    if (j > 0) {
        int idx_s = (j-1) * Nz + k;
        sum_r += ay * p_real_in[idx_s];
        sum_i += ay * p_imag_in[idx_s];
    }
    if (j < Ny - 1) {
        int idx_n = (j+1) * Nz + k;
        sum_r += ay * p_real_in[idx_n];
        sum_i += ay * p_imag_in[idx_n];
    }
    if (k > 0) {
        int idx_w = j * Nz + (k-1);
        sum_r += az * p_real_in[idx_w];
        sum_i += az * p_imag_in[idx_w];
    }
    if (k < Nz - 1) {
        int idx_e = j * Nz + (k+1);
        sum_r += az * p_real_in[idx_e];
        sum_i += az * p_imag_in[idx_e];
    }

    double Ap_r = D * p_r - sum_r;
    double Ap_i = D * p_i - sum_i;

    double res_r = rhs_real[idx] - Ap_r;
    double res_i = rhs_imag[idx] - Ap_i;

    p_real_out[idx] = p_r + omega * res_r * inv_D;
    p_imag_out[idx] = p_i + omega * res_i * inv_D;
}

// Convert complex array to split real/imag
__global__ void kernel_complex_to_split(
    const cufftDoubleComplex* __restrict__ c,
    double* __restrict__ real,
    double* __restrict__ imag,
    int total)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    real[tid] = c[tid].x;
    imag[tid] = c[tid].y;
}

// Convert complex array to split real/imag with negation
__global__ void kernel_complex_to_split_negate(
    const cufftDoubleComplex* __restrict__ c,
    double* __restrict__ real,
    double* __restrict__ imag,
    int total)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    real[tid] = -c[tid].x;
    imag[tid] = -c[tid].y;
}

// Convert split real/imag to complex array
__global__ void kernel_split_to_complex(
    const double* __restrict__ real,
    const double* __restrict__ imag,
    cufftDoubleComplex* __restrict__ c,
    int total)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    c[tid].x = real[tid];
    c[tid].y = imag[tid];
}

// ============================================================================
// 2D Multigrid Kernels for Helmholtz solve
// All kernels process all modes in parallel (batched)
// Layout: [m * N_yz + j * Nz + k] where m=mode, j=y-index, k=z-index
// ============================================================================

// 2D Helmholtz smoother (weighted Jacobi) - processes all modes
// Solves: (∂²/∂y² + ∂²/∂z² - λ_m) p = f with Neumann BCs
__global__ void kernel_mg2d_smooth(
    const double* __restrict__ f_real,
    const double* __restrict__ f_imag,
    const double* __restrict__ p_real_in,
    const double* __restrict__ p_imag_in,
    double* __restrict__ p_real_out,
    double* __restrict__ p_imag_out,
    const double* __restrict__ lambda,
    int N_modes, int Ny, int Nz,
    double ay, double az,
    double omega)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N_modes * Ny * Nz;
    if (tid >= total) return;

    const int k = tid % Nz;
    const int rem = tid / Nz;
    const int j = rem % Ny;
    const int m = rem / Ny;

    const int N_yz = Ny * Nz;
    const int idx = m * N_yz + j * Nz + k;

    // Diagonal: D = 2*ay + 2*az + λ_m (with Neumann BC adjustments)
    double D = 2.0 * ay + 2.0 * az + lambda[m];

    // Neumann BC: reduce diagonal at boundaries
    if (j == 0) D -= ay;
    if (j == Ny - 1) D -= ay;
    if (k == 0) D -= az;
    if (k == Nz - 1) D -= az;

    if (D < 1e-14) D = 1e-14;
    double inv_D = 1.0 / D;

    double p_r = p_real_in[idx];
    double p_i = p_imag_in[idx];

    // Neighbor contributions (Neumann: use same value at boundary)
    double sum_r = 0.0, sum_i = 0.0;

    // South (j-1)
    int idx_s = (j > 0) ? (m * N_yz + (j-1) * Nz + k) : idx;
    sum_r += ay * p_real_in[idx_s];
    sum_i += ay * p_imag_in[idx_s];

    // North (j+1)
    int idx_n = (j < Ny - 1) ? (m * N_yz + (j+1) * Nz + k) : idx;
    sum_r += ay * p_real_in[idx_n];
    sum_i += ay * p_imag_in[idx_n];

    // West (k-1)
    int idx_w = (k > 0) ? (m * N_yz + j * Nz + (k-1)) : idx;
    sum_r += az * p_real_in[idx_w];
    sum_i += az * p_imag_in[idx_w];

    // East (k+1)
    int idx_e = (k < Nz - 1) ? (m * N_yz + j * Nz + (k+1)) : idx;
    sum_r += az * p_real_in[idx_e];
    sum_i += az * p_imag_in[idx_e];

    // Ap = D*p - sum_neighbors
    double Ap_r = D * p_r - sum_r;
    double Ap_i = D * p_i - sum_i;

    // Residual = f - Ap, then Jacobi update
    double res_r = f_real[idx] - Ap_r;
    double res_i = f_imag[idx] - Ap_i;

    p_real_out[idx] = p_r + omega * res_r * inv_D;
    p_imag_out[idx] = p_i + omega * res_i * inv_D;
}

// Compute residual: r = f - Ap for 2D Helmholtz
__global__ void kernel_mg2d_residual(
    const double* __restrict__ f_real,
    const double* __restrict__ f_imag,
    const double* __restrict__ p_real,
    const double* __restrict__ p_imag,
    double* __restrict__ r_real,
    double* __restrict__ r_imag,
    const double* __restrict__ lambda,
    int N_modes, int Ny, int Nz,
    double ay, double az)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N_modes * Ny * Nz;
    if (tid >= total) return;

    const int k = tid % Nz;
    const int rem = tid / Nz;
    const int j = rem % Ny;
    const int m = rem / Ny;

    const int N_yz = Ny * Nz;
    const int idx = m * N_yz + j * Nz + k;

    double D = 2.0 * ay + 2.0 * az + lambda[m];
    if (j == 0) D -= ay;
    if (j == Ny - 1) D -= ay;
    if (k == 0) D -= az;
    if (k == Nz - 1) D -= az;

    double p_r = p_real[idx];
    double p_i = p_imag[idx];

    double sum_r = 0.0, sum_i = 0.0;
    int idx_s = (j > 0) ? (m * N_yz + (j-1) * Nz + k) : idx;
    int idx_n = (j < Ny - 1) ? (m * N_yz + (j+1) * Nz + k) : idx;
    int idx_w = (k > 0) ? (m * N_yz + j * Nz + (k-1)) : idx;
    int idx_e = (k < Nz - 1) ? (m * N_yz + j * Nz + (k+1)) : idx;

    sum_r = ay * (p_real[idx_s] + p_real[idx_n]) + az * (p_real[idx_w] + p_real[idx_e]);
    sum_i = ay * (p_imag[idx_s] + p_imag[idx_n]) + az * (p_imag[idx_w] + p_imag[idx_e]);

    double Ap_r = D * p_r - sum_r;
    double Ap_i = D * p_i - sum_i;

    r_real[idx] = f_real[idx] - Ap_r;
    r_imag[idx] = f_imag[idx] - Ap_i;
}

// Full-weighting restriction: fine -> coarse (9-point stencil for 2D)
__global__ void kernel_mg2d_restrict(
    const double* __restrict__ r_fine_real,
    const double* __restrict__ r_fine_imag,
    double* __restrict__ f_coarse_real,
    double* __restrict__ f_coarse_imag,
    int N_modes, int Ny_f, int Nz_f, int Ny_c, int Nz_c)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_c = N_modes * Ny_c * Nz_c;
    if (tid >= total_c) return;

    const int k_c = tid % Nz_c;
    const int rem = tid / Nz_c;
    const int j_c = rem % Ny_c;
    const int m = rem / Ny_c;

    const int N_yz_f = Ny_f * Nz_f;
    const int N_yz_c = Ny_c * Nz_c;

    // Fine grid indices (2:1 coarsening)
    int j_f = 2 * j_c;
    int k_f = 2 * k_c;

    // Clamp to fine grid bounds
    int j_f0 = j_f;
    int j_f1 = (j_f + 1 < Ny_f) ? j_f + 1 : j_f;
    int k_f0 = k_f;
    int k_f1 = (k_f + 1 < Nz_f) ? k_f + 1 : k_f;

    // 4-point average (simple injection at boundaries, full-weighting interior)
    int idx00 = m * N_yz_f + j_f0 * Nz_f + k_f0;
    int idx01 = m * N_yz_f + j_f0 * Nz_f + k_f1;
    int idx10 = m * N_yz_f + j_f1 * Nz_f + k_f0;
    int idx11 = m * N_yz_f + j_f1 * Nz_f + k_f1;

    int idx_c = m * N_yz_c + j_c * Nz_c + k_c;

    f_coarse_real[idx_c] = 0.25 * (r_fine_real[idx00] + r_fine_real[idx01]
                                  + r_fine_real[idx10] + r_fine_real[idx11]);
    f_coarse_imag[idx_c] = 0.25 * (r_fine_imag[idx00] + r_fine_imag[idx01]
                                  + r_fine_imag[idx10] + r_fine_imag[idx11]);
}

// Bilinear prolongation: coarse -> fine (add correction)
__global__ void kernel_mg2d_prolongate(
    const double* __restrict__ p_coarse_real,
    const double* __restrict__ p_coarse_imag,
    double* __restrict__ p_fine_real,
    double* __restrict__ p_fine_imag,
    int N_modes, int Ny_f, int Nz_f, int Ny_c, int Nz_c)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_f = N_modes * Ny_f * Nz_f;
    if (tid >= total_f) return;

    const int k_f = tid % Nz_f;
    const int rem = tid / Nz_f;
    const int j_f = rem % Ny_f;
    const int m = rem / Ny_f;

    const int N_yz_f = Ny_f * Nz_f;
    const int N_yz_c = Ny_c * Nz_c;

    // Find coarse cell and interpolation weights
    int j_c = j_f / 2;
    int k_c = k_f / 2;
    double wy = (j_f % 2 == 0) ? 0.75 : 0.25;  // Weight for j_c vs j_c+1
    double wz = (k_f % 2 == 0) ? 0.75 : 0.25;

    // Clamp coarse indices
    int j_c1 = (j_c + 1 < Ny_c) ? j_c + 1 : j_c;
    int k_c1 = (k_c + 1 < Nz_c) ? k_c + 1 : k_c;

    int idx00 = m * N_yz_c + j_c * Nz_c + k_c;
    int idx01 = m * N_yz_c + j_c * Nz_c + k_c1;
    int idx10 = m * N_yz_c + j_c1 * Nz_c + k_c;
    int idx11 = m * N_yz_c + j_c1 * Nz_c + k_c1;

    // Bilinear interpolation
    double corr_r = wy * wz * p_coarse_real[idx00]
                  + wy * (1.0-wz) * p_coarse_real[idx01]
                  + (1.0-wy) * wz * p_coarse_real[idx10]
                  + (1.0-wy) * (1.0-wz) * p_coarse_real[idx11];
    double corr_i = wy * wz * p_coarse_imag[idx00]
                  + wy * (1.0-wz) * p_coarse_imag[idx01]
                  + (1.0-wy) * wz * p_coarse_imag[idx10]
                  + (1.0-wy) * (1.0-wz) * p_coarse_imag[idx11];

    int idx_f = m * N_yz_f + j_f * Nz_f + k_f;
    p_fine_real[idx_f] += corr_r;
    p_fine_imag[idx_f] += corr_i;
}

#endif // USE_GPU_OFFLOAD

// ============================================================================
// FFT1DPoissonSolver Implementation
// ============================================================================

FFT1DPoissonSolver::FFT1DPoissonSolver(const Mesh& mesh, int periodic_dir)
    : mesh_(&mesh)
    , periodic_dir_(periodic_dir)
    , Nx_(mesh.Nx)
    , Ny_(mesh.Ny)
    , Nz_(mesh.Nz)
    , dx_(mesh.dx)
    , dy_(mesh.dy)
    , dz_(mesh.dz)
{
    if (mesh.is2D()) {
        throw std::runtime_error("FFT1DPoissonSolver requires 3D mesh");
    }

    if (periodic_dir != 0 && periodic_dir != 2) {
        throw std::runtime_error("periodic_dir must be 0 (x) or 2 (z)");
    }

    // Set up dimensions based on periodic direction
    if (periodic_dir_ == 0) {
        // x is periodic
        N_periodic_ = Nx_;
        d_periodic_ = dx_;
        N_yz_ = Ny_ * Nz_;
    } else {
        // z is periodic
        N_periodic_ = Nz_;
        d_periodic_ = dz_;
        N_yz_ = Nx_ * Ny_;
    }

    N_modes_ = N_periodic_ / 2 + 1;

#ifdef USE_GPU_OFFLOAD
    initialize_fft();
    initialize_eigenvalues();
#endif
}

FFT1DPoissonSolver::~FFT1DPoissonSolver() {
#ifdef USE_GPU_OFFLOAD
    cleanup();
#endif
}

void FFT1DPoissonSolver::set_bc(PoissonBC x_lo, PoissonBC x_hi,
                                 PoissonBC y_lo, PoissonBC y_hi,
                                 PoissonBC z_lo, PoissonBC z_hi) {
    if (periodic_dir_ == 0) {
        // x must be periodic
        if (x_lo != PoissonBC::Periodic || x_hi != PoissonBC::Periodic) {
            throw std::runtime_error("FFT1DPoissonSolver: x must be periodic when periodic_dir=0");
        }
        bc_y_lo_ = y_lo;
        bc_y_hi_ = y_hi;
        bc_z_lo_ = z_lo;
        bc_z_hi_ = z_hi;
    } else {
        // z must be periodic
        if (z_lo != PoissonBC::Periodic || z_hi != PoissonBC::Periodic) {
            throw std::runtime_error("FFT1DPoissonSolver: z must be periodic when periodic_dir=2");
        }
        bc_x_lo_ = x_lo;
        bc_x_hi_ = x_hi;
        bc_y_lo_ = y_lo;
        bc_y_hi_ = y_hi;
    }
}

bool FFT1DPoissonSolver::is_suitable(PoissonBC x_lo, PoissonBC x_hi,
                                      PoissonBC y_lo, PoissonBC y_hi,
                                      PoissonBC z_lo, PoissonBC z_hi,
                                      bool uniform_x, bool uniform_z,
                                      bool is_3d) {
    if (!is_3d) return false;

    bool x_periodic = (x_lo == PoissonBC::Periodic && x_hi == PoissonBC::Periodic);
    bool z_periodic = (z_lo == PoissonBC::Periodic && z_hi == PoissonBC::Periodic);

    // Exactly one of x,z must be periodic
    if (x_periodic == z_periodic) return false;  // Both or neither

    // The periodic direction must have uniform spacing
    if (x_periodic && !uniform_x) return false;
    if (z_periodic && !uniform_z) return false;

    return true;
}

#ifdef USE_GPU_OFFLOAD

void FFT1DPoissonSolver::initialize_fft() {
    // Create CUDA stream
    cudaStreamCreate(&stream_);

    const int batch = N_yz_;
    const int N = N_periodic_;

    // Allocate packed buffers
    size_t pack_size = (size_t)batch * N;
    size_t hat_size = (size_t)N_modes_ * N_yz_;

    cudaMalloc(&in_pack_, pack_size * sizeof(double));
    cudaMalloc(&out_pack_, pack_size * sizeof(double));
    cudaMalloc(&rhs_hat_, hat_size * sizeof(cufftDoubleComplex));
    cudaMalloc(&p_hat_, hat_size * sizeof(cufftDoubleComplex));

    // Allocate work buffers for split real/imag
    cudaMalloc(&work_real_, hat_size * sizeof(double));
    cudaMalloc(&work_imag_, hat_size * sizeof(double));

    // Allocate eigenvalue array
    cudaMalloc(&lambda_, N_modes_ * sizeof(double));

    // Allocate reduction buffers
    int block = 256;
    num_blocks_ = ((int)pack_size + block - 1) / block;
    cudaMalloc(&partial_sums_, num_blocks_ * sizeof(double));
    cudaMalloc(&sum_dev_, sizeof(double));

    // Create cuFFT plans
    // Forward: D2Z (real to complex)
    // Layout: input is [batch][N], output is [mode][batch] (mode-major)

    int rank = 1;
    int n[1] = { N };
    int inembed[1] = { N };
    int onembed[1] = { N_modes_ };

    // Input: istride=1 (contiguous x-lines), idist=N (next batch)
    int istride = 1;
    int idist = N;

    // Output: ostride=N_yz (mode-major), odist=1 (batches contiguous)
    int ostride = N_yz_;
    int odist = 1;

    cufftResult result = cufftPlanMany(&fft_plan_r2c_, rank, n,
                                        inembed, istride, idist,
                                        onembed, ostride, odist,
                                        CUFFT_D2Z, batch);
    if (result != CUFFT_SUCCESS) {
        throw std::runtime_error("Failed to create cuFFT D2Z plan");
    }
    cufftSetStream(fft_plan_r2c_, stream_);

    // Inverse: Z2D (complex to real)
    // Input is mode-major, output is batch-major
    int istride_c = N_yz_;
    int idist_c = 1;
    int ostride_r = 1;
    int odist_r = N;

    result = cufftPlanMany(&fft_plan_c2r_, rank, n,
                           onembed, istride_c, idist_c,
                           inembed, ostride_r, odist_r,
                           CUFFT_Z2D, batch);
    if (result != CUFFT_SUCCESS) {
        throw std::runtime_error("Failed to create cuFFT Z2D plan");
    }
    cufftSetStream(fft_plan_c2r_, stream_);

    plans_created_ = true;

    std::cout << "[FFT1DPoissonSolver] Initialized (periodic_dir=" << periodic_dir_
              << ", N=" << N << ", modes=" << N_modes_ << ", batch=" << batch << ")\n";
}

void FFT1DPoissonSolver::initialize_eigenvalues() {
    // Precompute discrete eigenvalues: lambda[m] = (2 - 2*cos(2*pi*m/N)) / h^2
    const int N = N_periodic_;
    const double h2 = d_periodic_ * d_periodic_;

    std::vector<double> lambda_host(N_modes_);
    for (int m = 0; m < N_modes_; ++m) {
        double theta = 2.0 * M_PI * m / N;
        lambda_host[m] = (2.0 - 2.0 * std::cos(theta)) / h2;
    }

    cudaMemcpy(lambda_, lambda_host.data(), N_modes_ * sizeof(double), cudaMemcpyHostToDevice);
}

void FFT1DPoissonSolver::cleanup() {
    if (plans_created_) {
        cufftDestroy(fft_plan_r2c_);
        cufftDestroy(fft_plan_c2r_);
    }
    if (stream_) cudaStreamDestroy(stream_);
    if (in_pack_) cudaFree(in_pack_);
    if (out_pack_) cudaFree(out_pack_);
    if (rhs_hat_) cudaFree(rhs_hat_);
    if (p_hat_) cudaFree(p_hat_);
    if (work_real_) cudaFree(work_real_);
    if (work_imag_) cudaFree(work_imag_);
    if (lambda_) cudaFree(lambda_);
    if (partial_sums_) cudaFree(partial_sums_);
    if (sum_dev_) cudaFree(sum_dev_);
    cleanup_mg();
}

void FFT1DPoissonSolver::cleanup_mg() {
    if (!mg_initialized_) return;

    for (int l = 0; l < mg_num_levels_; ++l) {
        if (mg_p_real_[l]) cudaFree(mg_p_real_[l]);
        if (mg_p_imag_[l]) cudaFree(mg_p_imag_[l]);
        if (mg_r_real_[l]) cudaFree(mg_r_real_[l]);
        if (mg_r_imag_[l]) cudaFree(mg_r_imag_[l]);
        mg_p_real_[l] = mg_p_imag_[l] = mg_r_real_[l] = mg_r_imag_[l] = nullptr;
    }
    if (mg_tmp_real_) cudaFree(mg_tmp_real_);
    if (mg_tmp_imag_) cudaFree(mg_tmp_imag_);
    mg_tmp_real_ = mg_tmp_imag_ = nullptr;
    mg_initialized_ = false;
}

void FFT1DPoissonSolver::initialize_mg_levels() {
    if (mg_initialized_) return;

    // Determine grid sizes for 2D MG (y-z plane for x-periodic)
    int Ny_base = (periodic_dir_ == 0) ? Ny_ : Nx_;
    int Nz_base = (periodic_dir_ == 0) ? Nz_ : Ny_;

    // Build hierarchy: coarsen by 2 until grid is small enough
    mg_num_levels_ = 0;
    int Ny = Ny_base, Nz = Nz_base;
    while (Ny >= 4 && Nz >= 4 && mg_num_levels_ < MG_MAX_LEVELS) {
        mg_Ny_[mg_num_levels_] = Ny;
        mg_Nz_[mg_num_levels_] = Nz;
        mg_N_yz_[mg_num_levels_] = Ny * Nz;
        mg_num_levels_++;
        Ny /= 2;
        Nz /= 2;
    }

    if (mg_num_levels_ < 2) {
        std::cerr << "[FFT1D-MG] Warning: grid too small for MG, using single level\n";
        mg_num_levels_ = 1;
    }

    // Allocate arrays for each level
    cudaError_t err;
    for (int l = 0; l < mg_num_levels_; ++l) {
        size_t size = static_cast<size_t>(N_modes_) * mg_N_yz_[l] * sizeof(double);
        err = cudaMalloc(&mg_p_real_[l], size);
        err = cudaMalloc(&mg_p_imag_[l], size);
        err = cudaMalloc(&mg_r_real_[l], size);
        err = cudaMalloc(&mg_r_imag_[l], size);
        if (err != cudaSuccess) {
            std::cerr << "[FFT1D-MG] cudaMalloc failed for level " << l << "\n";
            cleanup_mg();
            return;
        }
    }

    // Temp buffer for ping-pong smoothing (size of finest level)
    size_t fine_size = static_cast<size_t>(N_modes_) * mg_N_yz_[0] * sizeof(double);
    cudaMalloc(&mg_tmp_real_, fine_size);
    cudaMalloc(&mg_tmp_imag_, fine_size);

    mg_initialized_ = true;
    std::cout << "[FFT1D-MG] Initialized " << mg_num_levels_ << " levels: ";
    for (int l = 0; l < mg_num_levels_; ++l) {
        std::cout << mg_Ny_[l] << "x" << mg_Nz_[l];
        if (l < mg_num_levels_ - 1) std::cout << " -> ";
    }
    std::cout << "\n";
}

void FFT1DPoissonSolver::mg_smooth_2d(int level, int iterations, double omega) {
    const int Ny = mg_Ny_[level];
    const int Nz = mg_Nz_[level];
    const int total = N_modes_ * Ny * Nz;
    const int block = 256;
    const int grid = (total + block - 1) / block;

    double ay = (periodic_dir_ == 0) ? 1.0 / (dy_ * dy_) : 1.0 / (dx_ * dx_);
    double az = (periodic_dir_ == 0) ? 1.0 / (dz_ * dz_) : 1.0 / (dy_ * dy_);

    // Scale coefficients for coarser grids (spacing doubles each level)
    double scale = 1.0 / (1 << level);  // 1, 0.5, 0.25, ...
    scale = scale * scale;  // spacing squared
    ay *= scale;
    az *= scale;

    double* p_in = mg_p_real_[level];
    double* pi_in = mg_p_imag_[level];
    double* p_out = mg_tmp_real_;
    double* pi_out = mg_tmp_imag_;

    for (int iter = 0; iter < iterations; ++iter) {
        kernel_mg2d_smooth<<<grid, block, 0, stream_>>>(
            mg_r_real_[level], mg_r_imag_[level],  // RHS (f)
            p_in, pi_in,
            p_out, pi_out,
            lambda_, N_modes_, Ny, Nz, ay, az, omega
        );
        // Swap pointers
        std::swap(p_in, p_out);
        std::swap(pi_in, pi_out);
    }

    // If odd iterations, result is in tmp, copy back
    if (iterations % 2 == 1) {
        cudaMemcpyAsync(mg_p_real_[level], mg_tmp_real_,
                        N_modes_ * Ny * Nz * sizeof(double), cudaMemcpyDeviceToDevice, stream_);
        cudaMemcpyAsync(mg_p_imag_[level], mg_tmp_imag_,
                        N_modes_ * Ny * Nz * sizeof(double), cudaMemcpyDeviceToDevice, stream_);
    }
}

void FFT1DPoissonSolver::mg_residual_2d(int level) {
    const int Ny = mg_Ny_[level];
    const int Nz = mg_Nz_[level];
    const int total = N_modes_ * Ny * Nz;
    const int block = 256;
    const int grid = (total + block - 1) / block;

    double ay = (periodic_dir_ == 0) ? 1.0 / (dy_ * dy_) : 1.0 / (dx_ * dx_);
    double az = (periodic_dir_ == 0) ? 1.0 / (dz_ * dz_) : 1.0 / (dy_ * dy_);
    double scale = 1.0 / (1 << level);
    scale = scale * scale;
    ay *= scale;
    az *= scale;

    // Use tmp as scratch for residual output, then copy to r
    kernel_mg2d_residual<<<grid, block, 0, stream_>>>(
        mg_r_real_[level], mg_r_imag_[level],  // f (RHS)
        mg_p_real_[level], mg_p_imag_[level],  // p (solution)
        mg_tmp_real_, mg_tmp_imag_,            // r output (temp)
        lambda_, N_modes_, Ny, Nz, ay, az
    );

    // Copy residual to r array (will be restricted to next level)
    cudaMemcpyAsync(mg_r_real_[level], mg_tmp_real_,
                    total * sizeof(double), cudaMemcpyDeviceToDevice, stream_);
    cudaMemcpyAsync(mg_r_imag_[level], mg_tmp_imag_,
                    total * sizeof(double), cudaMemcpyDeviceToDevice, stream_);
}

void FFT1DPoissonSolver::mg_restrict_2d(int fine_level) {
    const int coarse_level = fine_level + 1;
    const int Ny_f = mg_Ny_[fine_level];
    const int Nz_f = mg_Nz_[fine_level];
    const int Ny_c = mg_Ny_[coarse_level];
    const int Nz_c = mg_Nz_[coarse_level];

    const int total_c = N_modes_ * Ny_c * Nz_c;
    const int block = 256;
    const int grid = (total_c + block - 1) / block;

    kernel_mg2d_restrict<<<grid, block, 0, stream_>>>(
        mg_r_real_[fine_level], mg_r_imag_[fine_level],
        mg_r_real_[coarse_level], mg_r_imag_[coarse_level],
        N_modes_, Ny_f, Nz_f, Ny_c, Nz_c
    );

    // Zero coarse solution for V-cycle
    cudaMemsetAsync(mg_p_real_[coarse_level], 0,
                    N_modes_ * Ny_c * Nz_c * sizeof(double), stream_);
    cudaMemsetAsync(mg_p_imag_[coarse_level], 0,
                    N_modes_ * Ny_c * Nz_c * sizeof(double), stream_);
}

void FFT1DPoissonSolver::mg_prolongate_2d(int coarse_level) {
    const int fine_level = coarse_level - 1;
    const int Ny_f = mg_Ny_[fine_level];
    const int Nz_f = mg_Nz_[fine_level];
    const int Ny_c = mg_Ny_[coarse_level];
    const int Nz_c = mg_Nz_[coarse_level];

    const int total_f = N_modes_ * Ny_f * Nz_f;
    const int block = 256;
    const int grid = (total_f + block - 1) / block;

    kernel_mg2d_prolongate<<<grid, block, 0, stream_>>>(
        mg_p_real_[coarse_level], mg_p_imag_[coarse_level],
        mg_p_real_[fine_level], mg_p_imag_[fine_level],
        N_modes_, Ny_f, Nz_f, Ny_c, Nz_c
    );
}

void FFT1DPoissonSolver::mg_vcycle_2d(int level, int nu1, int nu2) {
    const double omega = 0.8;

    if (level == mg_num_levels_ - 1) {
        // Coarsest level: many smoothing iterations
        mg_smooth_2d(level, 20, omega);
        return;
    }

    // Pre-smoothing
    mg_smooth_2d(level, nu1, omega);

    // Compute residual
    mg_residual_2d(level);

    // Restrict to coarse grid
    mg_restrict_2d(level);

    // Recurse
    mg_vcycle_2d(level + 1, nu1, nu2);

    // Prolongate correction
    mg_prolongate_2d(level + 1);

    // Post-smoothing
    mg_smooth_2d(level, nu2, omega);
}

void FFT1DPoissonSolver::solve_helmholtz_2d_mg(int nu1, int nu2) {
    // Initialize MG levels if needed
    if (!mg_initialized_) {
        initialize_mg_levels();
    }

    const int total = N_modes_ * mg_N_yz_[0];
    const int block = 256;
    const int grid = (total + block - 1) / block;

    // Convert RHS from complex to split real/imag (negated)
    kernel_complex_to_split_negate<<<grid, block, 0, stream_>>>(
        rhs_hat_, mg_r_real_[0], mg_r_imag_[0], total
    );

    // Zero initial solution
    cudaMemsetAsync(mg_p_real_[0], 0, total * sizeof(double), stream_);
    cudaMemsetAsync(mg_p_imag_[0], 0, total * sizeof(double), stream_);

    // Single V-cycle
    mg_vcycle_2d(0, nu1, nu2);

    // Pin m=0, (j=0, k=0) to zero for gauge
    kernel_pin_zero_mode<<<1, 1, 0, stream_>>>(mg_p_real_[0], mg_p_imag_[0], mg_N_yz_[0]);

    // Convert solution back to complex
    kernel_split_to_complex<<<grid, block, 0, stream_>>>(
        mg_p_real_[0], mg_p_imag_[0], p_hat_, total
    );
}

void FFT1DPoissonSolver::solve_helmholtz_2d(int iterations, double omega) {
    const int total = N_modes_ * N_yz_;
    const int block = 256;
    const int grid = (total + block - 1) / block;

    // Grid spacing coefficients
    double ay, az;
    int Ny_solve, Nz_solve;
    int bc_y_lo_int, bc_y_hi_int, bc_z_lo_int, bc_z_hi_int;

    if (periodic_dir_ == 0) {
        // x periodic: solve in (y,z) plane
        ay = 1.0 / (dy_ * dy_);
        az = 1.0 / (dz_ * dz_);
        Ny_solve = Ny_;
        Nz_solve = Nz_;
        bc_y_lo_int = static_cast<int>(bc_y_lo_);
        bc_y_hi_int = static_cast<int>(bc_y_hi_);
        bc_z_lo_int = static_cast<int>(bc_z_lo_);
        bc_z_hi_int = static_cast<int>(bc_z_hi_);
    } else {
        // z periodic: solve in (x,y) plane
        // Reinterpret: "y" in kernel = x, "z" in kernel = y
        ay = 1.0 / (dx_ * dx_);
        az = 1.0 / (dy_ * dy_);
        Ny_solve = Nx_;
        Nz_solve = Ny_;
        bc_y_lo_int = static_cast<int>(bc_x_lo_);
        bc_y_hi_int = static_cast<int>(bc_x_hi_);
        bc_z_lo_int = static_cast<int>(bc_y_lo_);
        bc_z_hi_int = static_cast<int>(bc_y_hi_);
    }

    // Ping-pong buffers for Jacobi iteration
    double* p_real_in = work_real_;
    double* p_imag_in = work_imag_;
    double* p_real_out;
    double* p_imag_out;

    // Initialize solution to zero
    cudaMemsetAsync(work_real_, 0, total * sizeof(double), stream_);
    cudaMemsetAsync(work_imag_, 0, total * sizeof(double), stream_);

    // Allocate temporary buffers
    double* rhs_real = nullptr;
    double* rhs_imag = nullptr;
    cudaError_t err;
    err = cudaMalloc(&rhs_real, total * sizeof(double));
    if (err != cudaSuccess) {
        std::cerr << "[FFT1D] cudaMalloc rhs_real failed: " << cudaGetErrorString(err) << "\n";
        return;
    }
    err = cudaMalloc(&rhs_imag, total * sizeof(double));
    if (err != cudaSuccess) {
        std::cerr << "[FFT1D] cudaMalloc rhs_imag failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(rhs_real);
        return;
    }

    // Convert and NEGATE the RHS: we solve (-L_yz + λ*I) * p = -rhs
    kernel_complex_to_split_negate<<<grid, block, 0, stream_>>>(rhs_hat_, rhs_real, rhs_imag, total);

    // Ping-pong buffer B
    double* p_real_B = nullptr;
    double* p_imag_B = nullptr;
    err = cudaMalloc(&p_real_B, total * sizeof(double));
    if (err != cudaSuccess) {
        std::cerr << "[FFT1D] cudaMalloc p_real_B failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(rhs_real); cudaFree(rhs_imag);
        return;
    }
    err = cudaMalloc(&p_imag_B, total * sizeof(double));
    if (err != cudaSuccess) {
        std::cerr << "[FFT1D] cudaMalloc p_imag_B failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(rhs_real); cudaFree(rhs_imag); cudaFree(p_real_B);
        return;
    }
    cudaMemsetAsync(p_real_B, 0, total * sizeof(double), stream_);
    cudaMemsetAsync(p_imag_B, 0, total * sizeof(double), stream_);

    // Uniform iterations for all modes
    // 15 iterations is a good balance: m>0 converges well, m=0 is acceptable
    // (m=0 has slow convergence but we're doing projection, not exact solve)
    int total_iters = iterations;

    for (int iter = 0; iter < total_iters; ++iter) {
        if (iter % 2 == 0) {
            p_real_in = work_real_;
            p_imag_in = work_imag_;
            p_real_out = p_real_B;
            p_imag_out = p_imag_B;
        } else {
            p_real_in = p_real_B;
            p_imag_in = p_imag_B;
            p_real_out = work_real_;
            p_imag_out = work_imag_;
        }

        kernel_helmholtz_jacobi_2d<<<grid, block, 0, stream_>>>(
            rhs_real, rhs_imag,
            p_real_in, p_imag_in,
            p_real_out, p_imag_out,
            lambda_,
            N_modes_, Ny_solve, Nz_solve,
            ay, az, omega,
            bc_y_lo_int, bc_y_hi_int, bc_z_lo_int, bc_z_hi_int
        );

        kernel_pin_zero_mode<<<1, 1, 0, stream_>>>(p_real_out, p_imag_out, N_yz_);
    }

    // Final result location
    double* final_real = (total_iters % 2 == 1) ? p_real_B : work_real_;
    double* final_imag = (total_iters % 2 == 1) ? p_imag_B : work_imag_;

    // Convert back to complex in p_hat
    kernel_split_to_complex<<<grid, block, 0, stream_>>>(final_real, final_imag, p_hat_, total);

    // Cleanup temporary allocations
    cudaFree(rhs_real);
    cudaFree(rhs_imag);
    cudaFree(p_real_B);
    cudaFree(p_imag_B);
}

int FFT1DPoissonSolver::solve_device(double* rhs_ptr, double* p_ptr, const PoissonConfig& cfg) {
    const int stride = Nx_ + 2;
    const int plane_stride = (Nx_ + 2) * (Ny_ + 2);
    const int total_interior = Nx_ * Ny_ * Nz_;

    const int block = 256;
    const int grid = (total_interior + block - 1) / block;
    const size_t smem = block * sizeof(double);

    // Convert OMP-mapped host pointers to CUDA device pointers
    int device = omp_get_default_device();
    double* rhs_dev = static_cast<double*>(omp_get_mapped_ptr(rhs_ptr, device));
    double* p_dev = static_cast<double*>(omp_get_mapped_ptr(p_ptr, device));

    // Debug: check pointers
    if (!rhs_dev || !p_dev || !in_pack_ || !out_pack_ || !rhs_hat_ || !p_hat_) {
        std::cerr << "[FFT1D] ERROR: null pointer detected\n";
        std::cerr << "  rhs_dev=" << rhs_dev << " p_dev=" << p_dev << "\n";
        std::cerr << "  in_pack_=" << in_pack_ << " out_pack_=" << out_pack_ << "\n";
        std::cerr << "  rhs_hat_=" << rhs_hat_ << " p_hat_=" << p_hat_ << "\n";
        return -1;
    }

    cudaError_t err;

    // Profiling: enabled by NNCFD_FFT1D_PROFILE environment variable
    static bool profile_enabled = (std::getenv("NNCFD_FFT1D_PROFILE") != nullptr);
    static int profile_call = 0;
    static double t_pack = 0, t_fft_fwd = 0, t_helmholtz = 0, t_fft_inv = 0, t_unpack = 0;
    cudaEvent_t ev_start, ev_pack, ev_fft_fwd, ev_helmholtz, ev_fft_inv, ev_unpack;

    if (profile_enabled) {
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_pack);
        cudaEventCreate(&ev_fft_fwd);
        cudaEventCreate(&ev_helmholtz);
        cudaEventCreate(&ev_fft_inv);
        cudaEventCreate(&ev_unpack);
        cudaEventRecord(ev_start, stream_);
    }

    // 1. Pack RHS from ghost layout to contiguous periodic lines + compute sum for mean
    if (periodic_dir_ == 0) {
        // x-periodic: pack x-lines
        kernel_pack_ghost_to_lines<<<grid, block, smem, stream_>>>(
            rhs_dev, in_pack_, partial_sums_,
            Nx_, Ny_, Nz_, stride, plane_stride
        );
    } else {
        // z-periodic: pack z-lines
        kernel_pack_ghost_to_zlines<<<grid, block, smem, stream_>>>(
            rhs_dev, in_pack_, partial_sums_,
            Nx_, Ny_, Nz_, stride, plane_stride
        );
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[FFT1D] kernel_pack failed: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    // 2. Reduce partial sums to get total sum
    kernel_reduce_sum<<<1, 256, 256 * sizeof(double), stream_>>>(
        partial_sums_, sum_dev_, num_blocks_
    );

    // 3. Subtract mean from packed RHS (for compatibility with singular Poisson)
    kernel_subtract_mean<<<grid, block, 0, stream_>>>(
        in_pack_, sum_dev_, total_interior
    );

    if (profile_enabled) cudaEventRecord(ev_pack, stream_);

    // 4. Forward FFT: real -> complex (mode-major output)
    cufftResult fft_result = cufftExecD2Z(fft_plan_r2c_, in_pack_, rhs_hat_);
    if (fft_result != CUFFT_SUCCESS) {
        std::cerr << "[FFT1D] cufftExecD2Z failed: " << fft_result << "\n";
        return -1;
    }

    if (profile_enabled) cudaEventRecord(ev_fft_fwd, stream_);

    // 5. Solve 2D Helmholtz for each mode using 2D Multigrid V-cycle
    // Much better convergence than Jacobi iterations
    solve_helmholtz_2d_mg(2, 1);  // nu1=2 pre-smooth, nu2=1 post-smooth
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[FFT1D] solve_helmholtz_2d_mg failed: " << cudaGetErrorString(err) << "\n";
        return -1;
    }

    if (profile_enabled) cudaEventRecord(ev_helmholtz, stream_);

    // 6. Inverse FFT: complex -> real
    fft_result = cufftExecZ2D(fft_plan_c2r_, p_hat_, out_pack_);
    if (fft_result != CUFFT_SUCCESS) {
        std::cerr << "[FFT1D] cufftExecZ2D failed: " << fft_result << "\n";
        return -1;
    }

    if (profile_enabled) cudaEventRecord(ev_fft_inv, stream_);

    // 7. Unpack from packed layout to ghost layout with normalization
    const double invN = 1.0 / (double)N_periodic_;
    if (periodic_dir_ == 0) {
        // x-periodic: unpack x-lines
        kernel_unpack_lines_to_ghost<<<grid, block, 0, stream_>>>(
            out_pack_, p_dev,
            Nx_, Ny_, Nz_, stride, plane_stride,
            invN
        );
    } else {
        // z-periodic: unpack z-lines
        kernel_unpack_zlines_to_ghost<<<grid, block, 0, stream_>>>(
            out_pack_, p_dev,
            Nx_, Ny_, Nz_, stride, plane_stride,
            invN
        );
    }

    if (profile_enabled) cudaEventRecord(ev_unpack, stream_);

    // 8. Synchronize
    cudaStreamSynchronize(stream_);

    // Profiling: accumulate and report
    if (profile_enabled) {
        float ms;
        cudaEventElapsedTime(&ms, ev_start, ev_pack);     t_pack += ms;
        cudaEventElapsedTime(&ms, ev_pack, ev_fft_fwd);   t_fft_fwd += ms;
        cudaEventElapsedTime(&ms, ev_fft_fwd, ev_helmholtz); t_helmholtz += ms;
        cudaEventElapsedTime(&ms, ev_helmholtz, ev_fft_inv); t_fft_inv += ms;
        cudaEventElapsedTime(&ms, ev_fft_inv, ev_unpack); t_unpack += ms;

        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_pack);
        cudaEventDestroy(ev_fft_fwd);
        cudaEventDestroy(ev_helmholtz);
        cudaEventDestroy(ev_fft_inv);
        cudaEventDestroy(ev_unpack);

        profile_call++;
        if (profile_call % 100 == 0) {
            double total = t_pack + t_fft_fwd + t_helmholtz + t_fft_inv + t_unpack;
            std::cout << "\n[FFT1D Profile] After " << profile_call << " solves (avg per solve):\n"
                      << "  Pack+mean:   " << (t_pack / profile_call) << " ms ("
                      << (100.0 * t_pack / total) << "%)\n"
                      << "  FFT forward: " << (t_fft_fwd / profile_call) << " ms ("
                      << (100.0 * t_fft_fwd / total) << "%)\n"
                      << "  Helmholtz:   " << (t_helmholtz / profile_call) << " ms ("
                      << (100.0 * t_helmholtz / total) << "%)\n"
                      << "  FFT inverse: " << (t_fft_inv / profile_call) << " ms ("
                      << (100.0 * t_fft_inv / total) << "%)\n"
                      << "  Unpack:      " << (t_unpack / profile_call) << " ms ("
                      << (100.0 * t_unpack / total) << "%)\n"
                      << "  TOTAL:       " << (total / profile_call) << " ms\n";
        }
    }

    // Return 1 V-cycle per mode
    return N_modes_;
}

#endif // USE_GPU_OFFLOAD

} // namespace nncfd

#endif // USE_GPU_OFFLOAD (outer guard)
