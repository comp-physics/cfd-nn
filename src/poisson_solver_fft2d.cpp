#ifdef USE_GPU_OFFLOAD

#include "poisson_solver_fft2d.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include <stdexcept>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#include <cuda_runtime.h>
#endif

namespace nncfd {

// ============================================================================
// CUDA Kernels for 2D FFT Poisson Solver
// ============================================================================

#ifdef USE_GPU_OFFLOAD

// Block-level reduction using warp shuffle
__device__ __forceinline__ double warpReduceSum_2d(double val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ double blockReduceSum_2d(double val) {
    static __shared__ double shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceSum_2d(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0;
    if (wid == 0) val = warpReduceSum_2d(val);

    return val;
}

// Pack kernel: ghost layout [j+1][i+1] -> packed [j*Nx + i]
__global__ void kernel_pack_2d(
    const double* __restrict__ rhs_ghost,
    double* __restrict__ in_pack,
    double* __restrict__ partial_sums,
    int Nx, int Ny, int stride)
{
    extern __shared__ double sdata[];

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = Nx * Ny;

    double local_sum = 0.0;

    if (tid < total) {
        const int i = tid % Nx;
        const int j = tid / Nx;

        const size_t g = (size_t)(j + 1) * stride + (i + 1);

        double val = rhs_ghost[g];
        in_pack[tid] = val;
        local_sum = val;
    }

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
__global__ void kernel_reduce_sum_2d(
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
__global__ void kernel_subtract_mean_2d(
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

// Unpack kernel with BC application
__global__ void kernel_unpack_and_bc_2d(
    const double* __restrict__ out_pack,
    double* __restrict__ p_ghost,
    int Nx, int Ny, int stride,
    double invNx,
    int bc_y_lo, int bc_y_hi)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = Nx * Ny;
    if (tid >= total) return;

    const int i = tid % Nx;
    const int j = tid / Nx;

    const double val = out_pack[tid] * invNx;

    const size_t g = (size_t)(j + 1) * stride + (i + 1);
    p_ghost[g] = val;

    // x-ghosts (periodic)
    if (i == 0) {
        double src_val = out_pack[j * Nx + (Nx - 1)] * invNx;
        p_ghost[(j + 1) * stride + 0] = src_val;
    }
    if (i == Nx - 1) {
        double src_val = out_pack[j * Nx + 0] * invNx;
        p_ghost[(j + 1) * stride + (Nx + 1)] = src_val;
    }

    // y-ghosts (Neumann: copy, Dirichlet: negate)
    if (j == 0) {
        if (bc_y_lo == 1) {  // Neumann
            p_ghost[0 * stride + (i + 1)] = val;
        } else {
            p_ghost[0 * stride + (i + 1)] = -val;
        }
    }
    if (j == Ny - 1) {
        if (bc_y_hi == 1) {  // Neumann
            p_ghost[(Ny + 1) * stride + (i + 1)] = val;
        } else {
            p_ghost[(Ny + 1) * stride + (i + 1)] = -val;
        }
    }
}

// 1D Jacobi iteration for Helmholtz equation in y-direction
// Solves: (d²/dy² - λ[m]) p = f for all modes m simultaneously
// Layout: data[m * Ny + j] where m = mode, j = y-index
__global__ void kernel_helmholtz_jacobi_1d(
    const double* __restrict__ rhs_real,
    const double* __restrict__ rhs_imag,
    const double* __restrict__ p_real_in,
    const double* __restrict__ p_imag_in,
    double* __restrict__ p_real_out,
    double* __restrict__ p_imag_out,
    const double* __restrict__ lambda,
    int N_modes, int Ny,
    double ay,  // 1/dy²
    double omega,
    int bc_y_lo, int bc_y_hi)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N_modes * Ny;
    if (tid >= total) return;

    const int j = tid % Ny;
    const int m = tid / Ny;
    const int idx = m * Ny + j;

    // Diagonal: D = 2*ay + λ[m]
    double D = 2.0 * ay + lambda[m];

    // Neumann BC: reduce diagonal at boundaries
    if (j == 0 && bc_y_lo == 1) D -= ay;
    if (j == Ny - 1 && bc_y_hi == 1) D -= ay;

    // Protect against singular m=0 mode
    if (D < 1e-14) D = 1e-14;
    double inv_D = 1.0 / D;

    double p_r = p_real_in[idx];
    double p_i = p_imag_in[idx];

    // Neighbor contributions
    double sum_r = 0.0, sum_i = 0.0;

    if (j > 0) {
        sum_r += ay * p_real_in[idx - 1];
        sum_i += ay * p_imag_in[idx - 1];
    }
    if (j < Ny - 1) {
        sum_r += ay * p_real_in[idx + 1];
        sum_i += ay * p_imag_in[idx + 1];
    }

    // Ap = D*p - sum_neighbors
    double Ap_r = D * p_r - sum_r;
    double Ap_i = D * p_i - sum_i;

    // Residual = b - Ap (note: we negate RHS for -Laplacian)
    double res_r = -rhs_real[idx] - Ap_r;
    double res_i = -rhs_imag[idx] - Ap_i;

    // Jacobi update
    p_real_out[idx] = p_r + omega * res_r * inv_D;
    p_imag_out[idx] = p_i + omega * res_i * inv_D;
}

// Pin m=0, j=0 to zero for gauge (singular Neumann case)
__global__ void kernel_pin_zero_mode_2d(
    double* __restrict__ p_real,
    double* __restrict__ p_imag,
    int Ny)
{
    p_real[0] = 0.0;
    p_imag[0] = 0.0;
}

// Convert complex array to split real/imag
__global__ void kernel_complex_to_split_2d(
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

// Convert split real/imag to complex array
__global__ void kernel_split_to_complex_2d(
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

#endif // USE_GPU_OFFLOAD

// ============================================================================
// FFT2DPoissonSolver Implementation
// ============================================================================

FFT2DPoissonSolver::FFT2DPoissonSolver(const Mesh& mesh)
    : mesh_(&mesh)
    , Nx_(mesh.Nx)
    , Ny_(mesh.Ny)
    , dx_(mesh.dx)
    , dy_(mesh.dy)
{
    if (!mesh.is2D()) {
        throw std::runtime_error("FFT2DPoissonSolver requires 2D mesh (Nz=1)");
    }

    N_modes_ = Nx_ / 2 + 1;

#ifdef USE_GPU_OFFLOAD
    initialize_fft();
    initialize_eigenvalues();
#endif
}

FFT2DPoissonSolver::~FFT2DPoissonSolver() {
#ifdef USE_GPU_OFFLOAD
    cleanup();
#endif
}

void FFT2DPoissonSolver::set_bc(PoissonBC x_lo, PoissonBC x_hi,
                                 PoissonBC y_lo, PoissonBC y_hi) {
    if (x_lo != PoissonBC::Periodic || x_hi != PoissonBC::Periodic) {
        throw std::runtime_error("FFT2DPoissonSolver: x must be periodic");
    }
    bc_y_lo_ = y_lo;
    bc_y_hi_ = y_hi;
}

bool FFT2DPoissonSolver::is_suitable(PoissonBC x_lo, PoissonBC x_hi,
                                      PoissonBC y_lo, PoissonBC y_hi,
                                      bool uniform_x, bool is_2d) {
    if (!is_2d) return false;
    if (!uniform_x) return false;
    if (x_lo != PoissonBC::Periodic || x_hi != PoissonBC::Periodic) return false;
    if (y_lo == PoissonBC::Periodic || y_hi == PoissonBC::Periodic) return false;
    return true;
}

#ifdef USE_GPU_OFFLOAD

void FFT2DPoissonSolver::initialize_fft() {
    cudaStreamCreate(&stream_);

    const int total = Nx_ * Ny_;
    const size_t hat_size = (size_t)N_modes_ * Ny_;

    // Allocate packed buffers
    cudaMalloc(&in_pack_, total * sizeof(double));
    cudaMalloc(&out_pack_, total * sizeof(double));
    cudaMalloc(&rhs_hat_, hat_size * sizeof(cufftDoubleComplex));
    cudaMalloc(&p_hat_, hat_size * sizeof(cufftDoubleComplex));

    // Work buffers for split real/imag (for Jacobi iteration)
    cudaMalloc(&work_real_, hat_size * sizeof(double));
    cudaMalloc(&work_imag_, hat_size * sizeof(double));

    // Mean subtraction buffers
    int block = 256;
    num_blocks_ = (total + block - 1) / block;
    cudaMalloc(&partial_sums_, num_blocks_ * sizeof(double));
    cudaMalloc(&sum_dev_, sizeof(double));

    // Create cuFFT plans
    // For 2D: batch of Ny 1D FFTs along x
    // Input:  in_pack[j * Nx + i]  (row-major, i fastest)
    // Output: rhs_hat[m * Ny + j]  (mode-major, j fastest)

    int rank = 1;
    int n[1] = { Nx_ };
    int inembed[1] = { Nx_ };
    int onembed[1] = { N_modes_ };

    int istride = 1;
    int idist = Nx_;
    int ostride = Ny_;
    int odist = 1;

    cufftResult result = cufftPlanMany(&fft_plan_r2c_, rank, n,
                                        inembed, istride, idist,
                                        onembed, ostride, odist,
                                        CUFFT_D2Z, Ny_);
    if (result != CUFFT_SUCCESS) {
        throw std::runtime_error("Failed to create cuFFT D2Z plan for 2D");
    }
    cufftSetStream(fft_plan_r2c_, stream_);

    result = cufftPlanMany(&fft_plan_c2r_, rank, n,
                           onembed, ostride, odist,
                           inembed, istride, idist,
                           CUFFT_Z2D, Ny_);
    if (result != CUFFT_SUCCESS) {
        throw std::runtime_error("Failed to create cuFFT Z2D plan for 2D");
    }
    cufftSetStream(fft_plan_c2r_, stream_);

    plans_created_ = true;

    std::cout << "[FFT2DPoissonSolver] Initialized (Nx=" << Nx_
              << ", Ny=" << Ny_ << ", modes=" << N_modes_ << ")\n";
}

void FFT2DPoissonSolver::initialize_eigenvalues() {
    // Discrete eigenvalues: lambda[m] = (2 - 2*cos(2*pi*m/Nx)) / dx^2
    const double h2 = dx_ * dx_;

    std::vector<double> lambda_host(N_modes_);
    for (int m = 0; m < N_modes_; ++m) {
        double theta = 2.0 * M_PI * m / Nx_;
        lambda_host[m] = (2.0 - 2.0 * std::cos(theta)) / h2;
    }

    cudaMalloc(&lambda_x_, N_modes_ * sizeof(double));
    cudaMemcpy(lambda_x_, lambda_host.data(), N_modes_ * sizeof(double), cudaMemcpyHostToDevice);
}

void FFT2DPoissonSolver::cleanup() {
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
    if (lambda_x_) cudaFree(lambda_x_);
    if (partial_sums_) cudaFree(partial_sums_);
    if (sum_dev_) cudaFree(sum_dev_);
}

int FFT2DPoissonSolver::solve_device(double* rhs_ptr, double* p_ptr, const PoissonConfig& cfg) {
    const int stride = Nx_ + 2;
    const int total = Nx_ * Ny_;
    const int hat_total = N_modes_ * Ny_;

    const int block = 256;
    const int grid = (total + block - 1) / block;
    const int grid_hat = (hat_total + block - 1) / block;
    const size_t smem = block * sizeof(double);

    // Convert OMP-mapped host pointers to CUDA device pointers
    int device = omp_get_default_device();
    double* rhs_dev = static_cast<double*>(omp_get_mapped_ptr(rhs_ptr, device));
    double* p_dev = static_cast<double*>(omp_get_mapped_ptr(p_ptr, device));

    // 1. Pack RHS from ghost layout + compute partial sums
    kernel_pack_2d<<<grid, block, smem, stream_>>>(
        rhs_dev, in_pack_, partial_sums_,
        Nx_, Ny_, stride
    );

    // 2. Reduce to get total sum
    kernel_reduce_sum_2d<<<1, 256, 256 * sizeof(double), stream_>>>(
        partial_sums_, sum_dev_, num_blocks_
    );

    // 3. Subtract mean (for singular Neumann case)
    kernel_subtract_mean_2d<<<grid, block, 0, stream_>>>(
        in_pack_, sum_dev_, total
    );

    // 4. Forward FFT: real -> complex
    cufftResult fft_result = cufftExecD2Z(fft_plan_r2c_, in_pack_, rhs_hat_);
    if (fft_result != CUFFT_SUCCESS) {
        std::cerr << "[FFT2D] cufftExecD2Z failed: " << fft_result << "\n";
        return -1;
    }

    // 5. Split complex RHS to real/imag for Jacobi iteration
    kernel_complex_to_split_2d<<<grid_hat, block, 0, stream_>>>(
        rhs_hat_, work_real_, work_imag_, hat_total
    );

    // 6. Allocate ping-pong buffers for Jacobi
    double* p_real_A = nullptr;
    double* p_imag_A = nullptr;
    double* p_real_B = nullptr;
    double* p_imag_B = nullptr;

    cudaMalloc(&p_real_A, hat_total * sizeof(double));
    cudaMalloc(&p_imag_A, hat_total * sizeof(double));
    cudaMalloc(&p_real_B, hat_total * sizeof(double));
    cudaMalloc(&p_imag_B, hat_total * sizeof(double));

    cudaMemsetAsync(p_real_A, 0, hat_total * sizeof(double), stream_);
    cudaMemsetAsync(p_imag_A, 0, hat_total * sizeof(double), stream_);
    cudaMemsetAsync(p_real_B, 0, hat_total * sizeof(double), stream_);
    cudaMemsetAsync(p_imag_B, 0, hat_total * sizeof(double), stream_);

    // 7. Jacobi iteration to solve 1D Helmholtz
    const double ay = 1.0 / (dy_ * dy_);
    const double omega = 0.8;  // Damped Jacobi
    const int iterations = 10;  // Enough for 1D problem

    int bc_y_lo_int = static_cast<int>(bc_y_lo_);
    int bc_y_hi_int = static_cast<int>(bc_y_hi_);

    double* p_real_in = p_real_A;
    double* p_imag_in = p_imag_A;
    double* p_real_out = p_real_B;
    double* p_imag_out = p_imag_B;

    for (int iter = 0; iter < iterations; ++iter) {
        kernel_helmholtz_jacobi_1d<<<grid_hat, block, 0, stream_>>>(
            work_real_, work_imag_,
            p_real_in, p_imag_in,
            p_real_out, p_imag_out,
            lambda_x_,
            N_modes_, Ny_,
            ay, omega,
            bc_y_lo_int, bc_y_hi_int
        );

        // Pin zero mode for gauge
        kernel_pin_zero_mode_2d<<<1, 1, 0, stream_>>>(p_real_out, p_imag_out, Ny_);

        // Swap buffers
        std::swap(p_real_in, p_real_out);
        std::swap(p_imag_in, p_imag_out);
    }

    // 8. Convert back to complex (result is in p_real_in/p_imag_in after swap)
    kernel_split_to_complex_2d<<<grid_hat, block, 0, stream_>>>(
        p_real_in, p_imag_in, p_hat_, hat_total
    );

    // Cleanup ping-pong buffers
    cudaFree(p_real_A);
    cudaFree(p_imag_A);
    cudaFree(p_real_B);
    cudaFree(p_imag_B);

    // 9. Inverse FFT: complex -> real
    fft_result = cufftExecZ2D(fft_plan_c2r_, p_hat_, out_pack_);
    if (fft_result != CUFFT_SUCCESS) {
        std::cerr << "[FFT2D] cufftExecZ2D failed: " << fft_result << "\n";
        return -1;
    }

    // 10. Unpack to ghost layout with normalization and BC
    const double invNx = 1.0 / (double)Nx_;
    kernel_unpack_and_bc_2d<<<grid, block, 0, stream_>>>(
        out_pack_, p_dev,
        Nx_, Ny_, stride,
        invNx,
        bc_y_lo_int, bc_y_hi_int
    );

    // 11. Synchronize
    cudaStreamSynchronize(stream_);

    residual_ = 0.0;
    return iterations;
}

#endif // USE_GPU_OFFLOAD

} // namespace nncfd

#endif // USE_GPU_OFFLOAD (outer guard)
