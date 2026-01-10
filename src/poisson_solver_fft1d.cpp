#ifdef USE_GPU_OFFLOAD

#include "poisson_solver_fft1d.hpp"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <omp.h>
#include <cuda_runtime.h>

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

// ============================================================================
// FP32 (Mixed Precision) 2D Multigrid Kernels for Helmholtz solve
// All kernels process all modes in parallel (batched)
// Layout: [m * N_yz + j * Nz + k] where m=mode, j=y-index, k=z-index
// Using float for MG smoothing gives ~2x memory bandwidth improvement
// ============================================================================

// Convert complex double array to split float arrays (for MG entry)
__global__ void kernel_complex_to_split_float(
    const cufftDoubleComplex* __restrict__ c,
    float* __restrict__ real,
    float* __restrict__ imag,
    int total)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    real[tid] = -static_cast<float>(c[tid].x);  // Negate for Helmholtz RHS
    imag[tid] = -static_cast<float>(c[tid].y);
}

// Convert split float arrays back to complex double (for MG exit)
__global__ void kernel_split_float_to_complex(
    const float* __restrict__ real,
    const float* __restrict__ imag,
    cufftDoubleComplex* __restrict__ c,
    int total)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    c[tid].x = static_cast<double>(real[tid]);
    c[tid].y = static_cast<double>(imag[tid]);
}

// FP32 2D Helmholtz smoother (weighted Jacobi)
__global__ void kernel_mg2d_smooth_f32(
    const float* __restrict__ f_real,
    const float* __restrict__ f_imag,
    const float* __restrict__ p_real_in,
    const float* __restrict__ p_imag_in,
    float* __restrict__ p_real_out,
    float* __restrict__ p_imag_out,
    const float* __restrict__ lambda,
    int N_modes, int Ny, int Nz,
    float ay, float az,
    float omega)
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

    // Diagonal: D = 2*ay + 2*az + λ_m
    float D = 2.0f * ay + 2.0f * az + lambda[m];
    if (j == 0) D -= ay;
    if (j == Ny - 1) D -= ay;
    if (k == 0) D -= az;
    if (k == Nz - 1) D -= az;

    if (D < 1e-7f) D = 1e-7f;
    float inv_D = 1.0f / D;

    float p_r = p_real_in[idx];
    float p_i = p_imag_in[idx];

    // Neighbor contributions (only add if neighbor exists - Neumann BC handling)
    float sum_r = 0.0f, sum_i = 0.0f;

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

    float Ap_r = D * p_r - sum_r;
    float Ap_i = D * p_i - sum_i;

    float res_r = f_real[idx] - Ap_r;
    float res_i = f_imag[idx] - Ap_i;

    p_real_out[idx] = p_r + omega * res_r * inv_D;
    p_imag_out[idx] = p_i + omega * res_i * inv_D;
}

// FP32 prolongation (coarse -> fine, add correction)
__global__ void kernel_mg2d_prolongate_f32(
    const float* __restrict__ p_coarse_real,
    const float* __restrict__ p_coarse_imag,
    float* __restrict__ p_fine_real,
    float* __restrict__ p_fine_imag,
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

    int j_c = j_f / 2;
    int k_c = k_f / 2;
    // Standard bilinear interpolation weights:
    // - Even indices (coincident with coarse): weight 1.0 from that coarse point
    // - Odd indices (midpoint): weight 0.5 from each neighboring coarse point
    float wy = 1.0f - 0.5f * (j_f % 2);
    float wz = 1.0f - 0.5f * (k_f % 2);

    int j_c1 = (j_c + 1 < Ny_c) ? j_c + 1 : j_c;
    int k_c1 = (k_c + 1 < Nz_c) ? k_c + 1 : k_c;

    int idx00 = m * N_yz_c + j_c * Nz_c + k_c;
    int idx01 = m * N_yz_c + j_c * Nz_c + k_c1;
    int idx10 = m * N_yz_c + j_c1 * Nz_c + k_c;
    int idx11 = m * N_yz_c + j_c1 * Nz_c + k_c1;

    float corr_r = wy * wz * p_coarse_real[idx00]
                 + wy * (1.0f-wz) * p_coarse_real[idx01]
                 + (1.0f-wy) * wz * p_coarse_real[idx10]
                 + (1.0f-wy) * (1.0f-wz) * p_coarse_real[idx11];
    float corr_i = wy * wz * p_coarse_imag[idx00]
                 + wy * (1.0f-wz) * p_coarse_imag[idx01]
                 + (1.0f-wy) * wz * p_coarse_imag[idx10]
                 + (1.0f-wy) * (1.0f-wz) * p_coarse_imag[idx11];

    int idx_f = m * N_yz_f + j_f * Nz_f + k_f;
    p_fine_real[idx_f] += corr_r;
    p_fine_imag[idx_f] += corr_i;
}

// FP32 pin zero mode
__global__ void kernel_pin_zero_mode_f32(
    float* __restrict__ p_real,
    float* __restrict__ p_imag,
    int N_yz)
{
    p_real[0] = 0.0f;
    p_imag[0] = 0.0f;
}

// ============================================================================
// Fused MG kernels for reduced memory traffic
// ============================================================================

// Helper: compute residual at a single fine grid point (inline)
__device__ __forceinline__ void compute_residual_at_point(
    const float* __restrict__ f_real, const float* __restrict__ f_imag,
    const float* __restrict__ p_real, const float* __restrict__ p_imag,
    const float* __restrict__ lambda,
    int m, int j, int k, int Ny, int Nz, int N_yz,
    float ay, float az,
    float& res_r, float& res_i)
{
    int idx = m * N_yz + j * Nz + k;

    // Diagonal coefficient
    float D = 2.0f * ay + 2.0f * az + lambda[m];
    if (j == 0) D -= ay;
    if (j == Ny - 1) D -= ay;
    if (k == 0) D -= az;
    if (k == Nz - 1) D -= az;

    float p_r = p_real[idx];
    float p_i = p_imag[idx];

    // Neighbor contributions (only add if neighbor exists - Neumann BC)
    float sum_r = 0.0f, sum_i = 0.0f;

    if (j > 0) {
        int idx_s = m * N_yz + (j-1) * Nz + k;
        sum_r += ay * p_real[idx_s];
        sum_i += ay * p_imag[idx_s];
    }
    if (j < Ny - 1) {
        int idx_n = m * N_yz + (j+1) * Nz + k;
        sum_r += ay * p_real[idx_n];
        sum_i += ay * p_imag[idx_n];
    }
    if (k > 0) {
        int idx_w = m * N_yz + j * Nz + (k-1);
        sum_r += az * p_real[idx_w];
        sum_i += az * p_imag[idx_w];
    }
    if (k < Nz - 1) {
        int idx_e = m * N_yz + j * Nz + (k+1);
        sum_r += az * p_real[idx_e];
        sum_i += az * p_imag[idx_e];
    }

    res_r = f_real[idx] - (D * p_r - sum_r);
    res_i = f_imag[idx] - (D * p_i - sum_i);
}

// Fused residual + restriction: computes residual at fine level and restricts to coarse
// Eliminates the intermediate fine residual write to global memory
__global__ void kernel_mg2d_residual_restrict_fused_f32(
    const float* __restrict__ f_fine_real,  // RHS at fine level
    const float* __restrict__ f_fine_imag,
    const float* __restrict__ p_fine_real,  // Solution at fine level
    const float* __restrict__ p_fine_imag,
    float* __restrict__ f_coarse_real,      // RHS at coarse level (output)
    float* __restrict__ f_coarse_imag,
    const float* __restrict__ lambda,
    int N_modes, int Ny_f, int Nz_f, int Ny_c, int Nz_c,
    float ay, float az)
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

    // Fine grid coordinates (2:1 coarsening)
    int j_f0 = 2 * j_c;
    int k_f0 = 2 * k_c;
    int j_f1 = (j_f0 + 1 < Ny_f) ? j_f0 + 1 : j_f0;
    int k_f1 = (k_f0 + 1 < Nz_f) ? k_f0 + 1 : k_f0;

    // Compute residuals at all 4 fine grid points
    float r00_r, r00_i, r01_r, r01_i, r10_r, r10_i, r11_r, r11_i;
    compute_residual_at_point(f_fine_real, f_fine_imag, p_fine_real, p_fine_imag,
                              lambda, m, j_f0, k_f0, Ny_f, Nz_f, N_yz_f, ay, az, r00_r, r00_i);
    compute_residual_at_point(f_fine_real, f_fine_imag, p_fine_real, p_fine_imag,
                              lambda, m, j_f0, k_f1, Ny_f, Nz_f, N_yz_f, ay, az, r01_r, r01_i);
    compute_residual_at_point(f_fine_real, f_fine_imag, p_fine_real, p_fine_imag,
                              lambda, m, j_f1, k_f0, Ny_f, Nz_f, N_yz_f, ay, az, r10_r, r10_i);
    compute_residual_at_point(f_fine_real, f_fine_imag, p_fine_real, p_fine_imag,
                              lambda, m, j_f1, k_f1, Ny_f, Nz_f, N_yz_f, ay, az, r11_r, r11_i);

    // Restriction: average the 4 residuals
    int idx_c = m * N_yz_c + j_c * Nz_c + k_c;
    f_coarse_real[idx_c] = 0.25f * (r00_r + r01_r + r10_r + r11_r);
    f_coarse_imag[idx_c] = 0.25f * (r00_i + r01_i + r10_i + r11_i);
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
    if (lambda_) cudaFree(lambda_);
    if (partial_sums_) cudaFree(partial_sums_);
    if (sum_dev_) cudaFree(sum_dev_);
    cleanup_mg();
}

void FFT1DPoissonSolver::cleanup_mg() {
    if (!mg_initialized_) return;

    // Destroy full solve CUDA graph
    if (solve_graph_exec_) {
        cudaGraphExecDestroy(solve_graph_exec_);
        solve_graph_exec_ = nullptr;
    }
    if (solve_graph_) {
        cudaGraphDestroy(solve_graph_);
        solve_graph_ = nullptr;
    }
    solve_graph_captured_ = false;
    cached_rhs_dev_ = nullptr;
    cached_p_dev_ = nullptr;

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
    if (lambda_f_) cudaFree(lambda_f_);
    lambda_f_ = nullptr;
    mg_initialized_ = false;
}

void FFT1DPoissonSolver::initialize_mg_levels() {
    if (mg_initialized_) return;

    // Determine grid sizes for 2D MG (y-z plane for x-periodic)
    int Ny_base = (periodic_dir_ == 0) ? Ny_ : Nx_;
    int Nz_base = (periodic_dir_ == 0) ? Nz_ : Ny_;

    // Build hierarchy: coarsen by 2 until grid is small enough
    // Stop at 24x24 to ensure proper MG for all grid sizes
    mg_num_levels_ = 0;
    int Ny = Ny_base, Nz = Nz_base;
    while (Ny >= 24 && Nz >= 24 && mg_num_levels_ < MG_MAX_LEVELS) {
        mg_Ny_[mg_num_levels_] = Ny;
        mg_Nz_[mg_num_levels_] = Nz;
        mg_N_yz_[mg_num_levels_] = Ny * Nz;
        mg_num_levels_++;
        Ny /= 2;
        Nz /= 2;
    }

    if (mg_num_levels_ < 1) {
        // Grid is smaller than 48x48 - use single level with base dimensions
        std::cerr << "[FFT1D-MG] Warning: grid too small for MG (" << Ny_base << "x" << Nz_base << "), using single level\n";
        mg_num_levels_ = 1;
        mg_Ny_[0] = Ny_base;
        mg_Nz_[0] = Nz_base;
        mg_N_yz_[0] = Ny_base * Nz_base;
    }

    // Allocate FP32 arrays for each level (mixed precision MG)
    cudaError_t err;
    for (int l = 0; l < mg_num_levels_; ++l) {
        size_t size = static_cast<size_t>(N_modes_) * mg_N_yz_[l] * sizeof(float);
        err = cudaMalloc(&mg_p_real_[l], size);
        err = cudaMalloc(&mg_p_imag_[l], size);
        err = cudaMalloc(&mg_r_real_[l], size);
        err = cudaMalloc(&mg_r_imag_[l], size);
        if (err != cudaSuccess) {
            std::cerr << "[FFT1D-MG] cudaMalloc failed for level " << l << "\n";
            cleanup_mg();
            return;
        }
        // Zero solution arrays for first solve (warm-start thereafter)
        cudaMemset(mg_p_real_[l], 0, size);
        cudaMemset(mg_p_imag_[l], 0, size);
    }

    // Temp buffer for ping-pong smoothing (size of finest level, FP32)
    size_t fine_size = static_cast<size_t>(N_modes_) * mg_N_yz_[0] * sizeof(float);
    cudaMalloc(&mg_tmp_real_, fine_size);
    cudaMalloc(&mg_tmp_imag_, fine_size);

    // Allocate and copy eigenvalues in FP32
    std::vector<float> lambda_f_host(N_modes_);
    for (int m = 0; m < N_modes_; ++m) {
        double theta = 2.0 * M_PI * m / N_periodic_;
        lambda_f_host[m] = static_cast<float>((2.0 - 2.0 * std::cos(theta)) / (d_periodic_ * d_periodic_));
    }
    cudaMalloc(&lambda_f_, N_modes_ * sizeof(float));
    cudaMemcpy(lambda_f_, lambda_f_host.data(), N_modes_ * sizeof(float), cudaMemcpyHostToDevice);

    mg_initialized_ = true;
    std::cout << "[FFT1D-MG] Initialized " << mg_num_levels_ << " levels: ";
    for (int l = 0; l < mg_num_levels_; ++l) {
        std::cout << mg_Ny_[l] << "x" << mg_Nz_[l];
        if (l < mg_num_levels_ - 1) std::cout << " -> ";
    }
    std::cout << "\n";
}

void FFT1DPoissonSolver::mg_smooth_2d_chebyshev(int level, int degree) {
    // Chebyshev polynomial acceleration for the 2D Helmholtz smoother (FP32)
    // Uses optimal weights that sweep through the eigenvalue spectrum
    constexpr float LAMBDA_MIN = 0.05f;
    constexpr float LAMBDA_MAX = 1.95f;
    const float d = (LAMBDA_MAX + LAMBDA_MIN) / 2.0f;  // 1.0
    const float c = (LAMBDA_MAX - LAMBDA_MIN) / 2.0f;  // 0.95

    const int Ny = mg_Ny_[level];
    const int Nz = mg_Nz_[level];
    const int total = N_modes_ * Ny * Nz;
    const int block = 256;
    const int grid = (total + block - 1) / block;

    float ay = (periodic_dir_ == 0) ? 1.0f / (dy_ * dy_) : 1.0f / (dx_ * dx_);
    float az = (periodic_dir_ == 0) ? 1.0f / (dz_ * dz_) : 1.0f / (dy_ * dy_);

    // Scale coefficients for coarser grids
    float scale = 1.0f / (1 << level);
    scale = scale * scale;
    ay *= scale;
    az *= scale;

    // Ping-pong buffers (FP32)
    float* p_in = mg_p_real_[level];
    float* pi_in = mg_p_imag_[level];
    float* p_out = mg_tmp_real_;
    float* pi_out = mg_tmp_imag_;

    for (int k = 0; k < degree; ++k) {
        // Chebyshev-optimal weight for step k
        float theta = static_cast<float>(M_PI) * (2.0f * k + 1.0f) / (2.0f * degree);
        float omega = 1.0f / (d - c * std::cos(theta));

        kernel_mg2d_smooth_f32<<<grid, block, 0, stream_>>>(
            mg_r_real_[level], mg_r_imag_[level],  // RHS (f)
            p_in, pi_in,
            p_out, pi_out,
            lambda_f_, N_modes_, Ny, Nz, ay, az, omega
        );

        // Swap pointers for next iteration
        std::swap(p_in, p_out);
        std::swap(pi_in, pi_out);
    }

    // If odd number of iterations, result is in tmp - copy back
    if (degree % 2 == 1) {
        cudaMemcpyAsync(mg_p_real_[level], mg_tmp_real_,
                        total * sizeof(float), cudaMemcpyDeviceToDevice, stream_);
        cudaMemcpyAsync(mg_p_imag_[level], mg_tmp_imag_,
                        total * sizeof(float), cudaMemcpyDeviceToDevice, stream_);
    }
}

// Fused residual + restriction: computes residual at fine level and directly restricts to coarse
// Eliminates intermediate fine residual storage (saves 2 global memory passes)
void FFT1DPoissonSolver::mg_residual_restrict_fused_2d(int fine_level) {
    const int coarse_level = fine_level + 1;
    const int Ny_f = mg_Ny_[fine_level];
    const int Nz_f = mg_Nz_[fine_level];
    const int Ny_c = mg_Ny_[coarse_level];
    const int Nz_c = mg_Nz_[coarse_level];

    const int total_c = N_modes_ * Ny_c * Nz_c;
    const int block = 256;
    const int grid = (total_c + block - 1) / block;

    // Grid spacing coefficients at fine level
    float ay = (periodic_dir_ == 0) ? 1.0f / (dy_ * dy_) : 1.0f / (dx_ * dx_);
    float az = (periodic_dir_ == 0) ? 1.0f / (dz_ * dz_) : 1.0f / (dy_ * dy_);
    float scale = 1.0f / (1 << fine_level);
    scale = scale * scale;
    ay *= scale;
    az *= scale;

    // Fused kernel: compute residual at 4 fine points, average to 1 coarse point
    kernel_mg2d_residual_restrict_fused_f32<<<grid, block, 0, stream_>>>(
        mg_r_real_[fine_level], mg_r_imag_[fine_level],   // RHS at fine
        mg_p_real_[fine_level], mg_p_imag_[fine_level],   // Solution at fine
        mg_r_real_[coarse_level], mg_r_imag_[coarse_level], // RHS at coarse (output)
        lambda_f_, N_modes_, Ny_f, Nz_f, Ny_c, Nz_c, ay, az
    );

    // Zero coarse solution for V-cycle (FP32)
    cudaMemsetAsync(mg_p_real_[coarse_level], 0,
                    N_modes_ * Ny_c * Nz_c * sizeof(float), stream_);
    cudaMemsetAsync(mg_p_imag_[coarse_level], 0,
                    N_modes_ * Ny_c * Nz_c * sizeof(float), stream_);
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

    kernel_mg2d_prolongate_f32<<<grid, block, 0, stream_>>>(
        mg_p_real_[coarse_level], mg_p_imag_[coarse_level],
        mg_p_real_[fine_level], mg_p_imag_[fine_level],
        N_modes_, Ny_f, Nz_f, Ny_c, Nz_c
    );
}

void FFT1DPoissonSolver::mg_vcycle_2d(int level, int nu1, int nu2) {
    // Use Chebyshev smoothing for faster convergence
    // Mode-aware: low-λ modes (small m) need more iterations than high-λ modes
    // High modes converge faster due to stronger diagonal dominance

    if (level == mg_num_levels_ - 1) {
        // Coarsest level: need more iterations for larger grids
        // For single-level (no actual MG), use more iterations
        int degree = (mg_num_levels_ == 1) ? 20 : 4;  // 20 for single-level, 4 for proper MG
        mg_smooth_2d_chebyshev(level, degree);
        return;
    }

    // Pre-smoothing: degree 2
    mg_smooth_2d_chebyshev(level, 2);

    // Fused residual + restriction: eliminates intermediate fine residual storage
    // (Previously: mg_residual_2d(level) + mg_restrict_2d(level))
    mg_residual_restrict_fused_2d(level);

    // Recurse
    mg_vcycle_2d(level + 1, nu1, nu2);

    // Prolongate correction
    mg_prolongate_2d(level + 1);

    // Post-smoothing: degree 2
    mg_smooth_2d_chebyshev(level, 2);
}

void FFT1DPoissonSolver::solve_helmholtz_2d_mg(int nu1, int nu2) {
    // Initialize MG levels if needed
    if (!mg_initialized_) {
        initialize_mg_levels();
    }

    const int total = N_modes_ * mg_N_yz_[0];
    const int block = 256;
    const int grid = (total + block - 1) / block;

    // Convert RHS from complex double to split float arrays (negated)
    kernel_complex_to_split_float<<<grid, block, 0, stream_>>>(
        rhs_hat_, mg_r_real_[0], mg_r_imag_[0], total
    );

    // Warm-start: reuse previous solution instead of zeroing
    // First solve needs initialization (done in initialize_mg_levels)
    // Subsequent solves benefit from warm-start since pressure changes smoothly

    // V-cycle (all in FP32)
    mg_vcycle_2d(0, nu1, nu2);

    // Pin m=0, (j=0, k=0) to zero for gauge (FP32)
    kernel_pin_zero_mode_f32<<<1, 1, 0, stream_>>>(mg_p_real_[0], mg_p_imag_[0], mg_N_yz_[0]);

    // Convert solution back from split float to complex double
    kernel_split_float_to_complex<<<grid, block, 0, stream_>>>(
        mg_p_real_[0], mg_p_imag_[0], p_hat_, total
    );
}

void FFT1DPoissonSolver::capture_solve_graph(double* rhs_dev, double* p_dev) {
    if (solve_graph_captured_) return;

    // Store pointers for graph execution
    cached_rhs_dev_ = rhs_dev;
    cached_p_dev_ = p_dev;

    const int stride = Nx_ + 2;
    const int plane_stride = (Nx_ + 2) * (Ny_ + 2);
    const int total_interior = Nx_ * Ny_ * Nz_;

    const int block = 256;
    const int grid = (total_interior + block - 1) / block;
    const size_t smem = block * sizeof(double);

    // Create a separate stream for capture
    cudaStream_t capture_stream;
    cudaStreamCreate(&capture_stream);

    // Associate cuFFT plans with the capture stream for graph capture
    cufftSetStream(fft_plan_r2c_, capture_stream);
    cufftSetStream(fft_plan_c2r_, capture_stream);

    // Begin stream capture
    cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);

    // 1. Pack RHS from ghost layout to contiguous periodic lines + compute sum for mean
    if (periodic_dir_ == 0) {
        kernel_pack_ghost_to_lines<<<grid, block, smem, capture_stream>>>(
            rhs_dev, in_pack_, partial_sums_,
            Nx_, Ny_, Nz_, stride, plane_stride
        );
    } else {
        kernel_pack_ghost_to_zlines<<<grid, block, smem, capture_stream>>>(
            rhs_dev, in_pack_, partial_sums_,
            Nx_, Ny_, Nz_, stride, plane_stride
        );
    }

    // 2. Reduce partial sums to get total sum
    kernel_reduce_sum<<<1, 256, 256 * sizeof(double), capture_stream>>>(
        partial_sums_, sum_dev_, num_blocks_
    );

    // 3. Subtract mean from packed RHS
    kernel_subtract_mean<<<grid, block, 0, capture_stream>>>(
        in_pack_, sum_dev_, total_interior
    );

    // 4. Forward FFT: real -> complex
    cufftExecD2Z(fft_plan_r2c_, in_pack_, rhs_hat_);

    // 5. MG Helmholtz solve (mixed precision: FP32 for MG)
    const int total_modes = N_modes_ * mg_N_yz_[0];
    const int grid_modes = (total_modes + block - 1) / block;

    // Convert RHS from complex double to split float arrays (negated)
    kernel_complex_to_split_float<<<grid_modes, block, 0, capture_stream>>>(
        rhs_hat_, mg_r_real_[0], mg_r_imag_[0], total_modes
    );

    // Warm-start: solution arrays were zeroed during initialization
    // Subsequent solves reuse previous solution for faster convergence

    // V-cycle (temporarily switch stream, all FP32)
    cudaStream_t old_stream = stream_;
    stream_ = capture_stream;
    mg_vcycle_2d(0, 2, 1);
    stream_ = old_stream;

    // Pin m=0, (j=0, k=0) to zero for gauge (FP32)
    kernel_pin_zero_mode_f32<<<1, 1, 0, capture_stream>>>(mg_p_real_[0], mg_p_imag_[0], mg_N_yz_[0]);

    // Convert solution back from split float to complex double
    kernel_split_float_to_complex<<<grid_modes, block, 0, capture_stream>>>(
        mg_p_real_[0], mg_p_imag_[0], p_hat_, total_modes
    );

    // 6. Inverse FFT: complex -> real
    cufftExecZ2D(fft_plan_c2r_, p_hat_, out_pack_);

    // 7. Unpack from packed layout to ghost layout with normalization
    const double invN = 1.0 / (double)N_periodic_;
    if (periodic_dir_ == 0) {
        kernel_unpack_lines_to_ghost<<<grid, block, 0, capture_stream>>>(
            out_pack_, p_dev,
            Nx_, Ny_, Nz_, stride, plane_stride,
            invN
        );
    } else {
        kernel_unpack_zlines_to_ghost<<<grid, block, 0, capture_stream>>>(
            out_pack_, p_dev,
            Nx_, Ny_, Nz_, stride, plane_stride,
            invN
        );
    }

    // End capture and instantiate graph
    cudaError_t err = cudaStreamEndCapture(capture_stream, &solve_graph_);
    if (err != cudaSuccess) {
        std::cerr << "[FFT1D] ERROR: cudaStreamEndCapture failed: " << cudaGetErrorString(err) << "\n";
        cudaStreamDestroy(capture_stream);
        cufftSetStream(fft_plan_r2c_, stream_);
        cufftSetStream(fft_plan_c2r_, stream_);
        return;
    }

    err = cudaGraphInstantiate(&solve_graph_exec_, solve_graph_, nullptr, nullptr, 0);
    if (err != cudaSuccess) {
        std::cerr << "[FFT1D] ERROR: cudaGraphInstantiate failed: " << cudaGetErrorString(err) << "\n";
        cudaGraphDestroy(solve_graph_);
        solve_graph_ = nullptr;
        cudaStreamDestroy(capture_stream);
        cufftSetStream(fft_plan_r2c_, stream_);
        cufftSetStream(fft_plan_c2r_, stream_);
        return;
    }

    cudaStreamDestroy(capture_stream);

    // Restore cuFFT plans to original stream
    cufftSetStream(fft_plan_r2c_, stream_);
    cufftSetStream(fft_plan_c2r_, stream_);

    solve_graph_captured_ = true;

    std::cout << "[FFT1D] CUDA graph captured for full solve (pack->FFT->MG->IFFT->unpack)\n";
}

void FFT1DPoissonSolver::execute_solve_graph() {
    if (!solve_graph_exec_) return;
    cudaGraphLaunch(solve_graph_exec_, stream_);
}

int FFT1DPoissonSolver::solve_device(double* rhs_ptr, double* p_ptr, const PoissonConfig& cfg) {
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

    // Initialize MG levels if needed (required before graph capture)
    if (!mg_initialized_) {
        initialize_mg_levels();
    }

    // Profiling mode: bypass graph to measure individual components
    static bool profile_mode = (std::getenv("NNCFD_FFT1D_PROFILE") != nullptr);
    if (profile_mode) {
        static int call_count = 0;
        static double t_pack = 0, t_fft_fwd = 0, t_helmholtz = 0, t_fft_inv = 0, t_unpack = 0;
        cudaEvent_t ev0, ev1, ev2, ev3, ev4, ev5;
        cudaEventCreate(&ev0); cudaEventCreate(&ev1); cudaEventCreate(&ev2);
        cudaEventCreate(&ev3); cudaEventCreate(&ev4); cudaEventCreate(&ev5);

        const int stride = Nx_ + 2;
        const int plane_stride = (Nx_ + 2) * (Ny_ + 2);
        const int total_interior = Nx_ * Ny_ * Nz_;
        const int block = 256;
        const int grid = (total_interior + block - 1) / block;
        const size_t smem = block * sizeof(double);

        cudaEventRecord(ev0, stream_);

        // Pack + mean
        if (periodic_dir_ == 0) {
            kernel_pack_ghost_to_lines<<<grid, block, smem, stream_>>>(
                rhs_dev, in_pack_, partial_sums_, Nx_, Ny_, Nz_, stride, plane_stride);
        } else {
            kernel_pack_ghost_to_zlines<<<grid, block, smem, stream_>>>(
                rhs_dev, in_pack_, partial_sums_, Nx_, Ny_, Nz_, stride, plane_stride);
        }
        kernel_reduce_sum<<<1, 256, 256 * sizeof(double), stream_>>>(partial_sums_, sum_dev_, num_blocks_);
        kernel_subtract_mean<<<grid, block, 0, stream_>>>(in_pack_, sum_dev_, total_interior);
        cudaEventRecord(ev1, stream_);

        // FFT forward
        cufftExecD2Z(fft_plan_r2c_, in_pack_, rhs_hat_);
        cudaEventRecord(ev2, stream_);

        // Helmholtz solve
        solve_helmholtz_2d_mg(2, 1);
        cudaEventRecord(ev3, stream_);

        // FFT inverse
        cufftExecZ2D(fft_plan_c2r_, p_hat_, out_pack_);
        cudaEventRecord(ev4, stream_);

        // Unpack
        const double invN = 1.0 / (double)N_periodic_;
        if (periodic_dir_ == 0) {
            kernel_unpack_lines_to_ghost<<<grid, block, 0, stream_>>>(
                out_pack_, p_dev, Nx_, Ny_, Nz_, stride, plane_stride, invN);
        } else {
            kernel_unpack_zlines_to_ghost<<<grid, block, 0, stream_>>>(
                out_pack_, p_dev, Nx_, Ny_, Nz_, stride, plane_stride, invN);
        }
        cudaEventRecord(ev5, stream_);
        cudaStreamSynchronize(stream_);

        float ms;
        cudaEventElapsedTime(&ms, ev0, ev1); t_pack += ms;
        cudaEventElapsedTime(&ms, ev1, ev2); t_fft_fwd += ms;
        cudaEventElapsedTime(&ms, ev2, ev3); t_helmholtz += ms;
        cudaEventElapsedTime(&ms, ev3, ev4); t_fft_inv += ms;
        cudaEventElapsedTime(&ms, ev4, ev5); t_unpack += ms;

        cudaEventDestroy(ev0); cudaEventDestroy(ev1); cudaEventDestroy(ev2);
        cudaEventDestroy(ev3); cudaEventDestroy(ev4); cudaEventDestroy(ev5);

        call_count++;
        if (call_count % 50 == 0) {
            double total = t_pack + t_fft_fwd + t_helmholtz + t_fft_inv + t_unpack;
            std::cout << "\n[FFT1D Profile] " << call_count << " calls (avg per solve):\n"
                      << "  Pack+mean:   " << (t_pack / call_count) << " ms (" << (100.0 * t_pack / total) << "%)\n"
                      << "  FFT forward: " << (t_fft_fwd / call_count) << " ms (" << (100.0 * t_fft_fwd / total) << "%)\n"
                      << "  Helmholtz:   " << (t_helmholtz / call_count) << " ms (" << (100.0 * t_helmholtz / total) << "%)\n"
                      << "  FFT inverse: " << (t_fft_inv / call_count) << " ms (" << (100.0 * t_fft_inv / total) << "%)\n"
                      << "  Unpack:      " << (t_unpack / call_count) << " ms (" << (100.0 * t_unpack / total) << "%)\n"
                      << "  TOTAL:       " << (total / call_count) << " ms\n";
        }
        return N_modes_;
    }

    // Use CUDA graph for the entire solve sequence
    // Note: Graph is captured with fixed pointers - if they change, we need to recapture
    if (solve_graph_captured_) {
        if (rhs_dev == cached_rhs_dev_ && p_dev == cached_p_dev_) {
            // Same pointers - just execute the graph
            execute_solve_graph();
            cudaStreamSynchronize(stream_);
            return N_modes_;
        } else {
            // Pointers changed - need to recapture
            // (This shouldn't happen in normal use, but handle it)
            if (solve_graph_exec_) {
                cudaGraphExecDestroy(solve_graph_exec_);
                solve_graph_exec_ = nullptr;
            }
            if (solve_graph_) {
                cudaGraphDestroy(solve_graph_);
                solve_graph_ = nullptr;
            }
            solve_graph_captured_ = false;
        }
    }

    // First call (or pointer change): capture and execute the graph
    capture_solve_graph(rhs_dev, p_dev);

    if (solve_graph_captured_) {
        // Graph captured successfully - execute it
        execute_solve_graph();
        cudaStreamSynchronize(stream_);
    } else {
        // Graph capture failed - fall back to non-graph execution
        std::cerr << "[FFT1D] Warning: Graph capture failed, using fallback path\n";

        const int stride = Nx_ + 2;
        const int plane_stride = (Nx_ + 2) * (Ny_ + 2);
        const int total_interior = Nx_ * Ny_ * Nz_;
        const int block = 256;
        const int grid = (total_interior + block - 1) / block;
        const size_t smem = block * sizeof(double);

        // Pack RHS
        if (periodic_dir_ == 0) {
            kernel_pack_ghost_to_lines<<<grid, block, smem, stream_>>>(
                rhs_dev, in_pack_, partial_sums_, Nx_, Ny_, Nz_, stride, plane_stride);
        } else {
            kernel_pack_ghost_to_zlines<<<grid, block, smem, stream_>>>(
                rhs_dev, in_pack_, partial_sums_, Nx_, Ny_, Nz_, stride, plane_stride);
        }

        // Reduce + subtract mean
        kernel_reduce_sum<<<1, 256, 256 * sizeof(double), stream_>>>(partial_sums_, sum_dev_, num_blocks_);
        kernel_subtract_mean<<<grid, block, 0, stream_>>>(in_pack_, sum_dev_, total_interior);

        // Forward FFT
        cufftExecD2Z(fft_plan_r2c_, in_pack_, rhs_hat_);

        // MG solve
        solve_helmholtz_2d_mg(2, 1);

        // Inverse FFT
        cufftExecZ2D(fft_plan_c2r_, p_hat_, out_pack_);

        // Unpack
        const double invN = 1.0 / (double)N_periodic_;
        if (periodic_dir_ == 0) {
            kernel_unpack_lines_to_ghost<<<grid, block, 0, stream_>>>(
                out_pack_, p_dev, Nx_, Ny_, Nz_, stride, plane_stride, invN);
        } else {
            kernel_unpack_zlines_to_ghost<<<grid, block, 0, stream_>>>(
                out_pack_, p_dev, Nx_, Ny_, Nz_, stride, plane_stride, invN);
        }

        cudaStreamSynchronize(stream_);
    }

    // Return 1 V-cycle per mode
    return N_modes_;
}

#endif // USE_GPU_OFFLOAD

} // namespace nncfd

#endif // USE_GPU_OFFLOAD (outer guard)
