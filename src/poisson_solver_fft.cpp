#ifdef USE_GPU_OFFLOAD

#include "poisson_solver_fft.hpp"
#include <iostream>
#include <cmath>
#include <stdexcept>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#include <cuda_runtime.h>
#endif

namespace nncfd {

// ============================================================================
// CUDA Kernels for GPU-resident FFT solver operations
// These run on stream_ to avoid OMP↔CUDA interop overhead
// ============================================================================

#ifdef USE_GPU_OFFLOAD

// Block-level reduction using shared memory + warp shuffle
__device__ __forceinline__ double warpReduceSum(double val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ double blockReduceSum(double val) {
    static __shared__ double shared[32];  // One per warp
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceSum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // First warp reduces across warps
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0;
    if (wid == 0) val = warpReduceSum(val);

    return val;
}

// Kernel: Pack RHS from ghost layout to FFT layout + compute partial sums
// Input layout:  rhs_ptr[k+Ng][j+Ng][i+Ng]  (ghost cells, field ordering)
// Output layout: packed[(i*Nz + k)*Ny + j]  (cuFFT interleaved batches)
// Also: each block writes its partial sum to partial_sums[]
__global__ void kernel_pack_and_partial_sum(
    const double* __restrict__ rhs_ptr,
    double* __restrict__ packed,
    double* __restrict__ partial_sums,
    int Nx, int Ny, int Nz, int Ng, int Nx_full, int Ny_full)
{
    const size_t n_total = static_cast<size_t>(Nx) * Ny * Nz;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    double local_sum = 0.0;

    if (idx < n_total) {
        // Decode linear index to (i, k, j) in FFT order
        // Layout: [(i*Nz + k)][j] with j fastest
        int j = idx % Ny;
        size_t mode = idx / Ny;
        int k = mode % Nz;
        int i = mode / Nz;

        // Source index with ghosts: [k+Ng][j+Ng][i+Ng]
        size_t src_idx = static_cast<size_t>(k + Ng) * Nx_full * Ny_full +
                         (j + Ng) * Nx_full + (i + Ng);

        double val = rhs_ptr[src_idx];
        packed[idx] = val;
        local_sum = val;
    }

    // Block-level reduction
    double block_sum = blockReduceSum(local_sum);

    // Thread 0 writes block partial sum
    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = block_sum;
    }
}

// Kernel: Final reduction of partial sums + compute mean
// Reduces partial_sums[num_blocks] -> sum_dev[0]
__global__ void kernel_final_reduce(
    const double* __restrict__ partial_sums,
    double* __restrict__ sum_dev,
    int num_blocks)
{
    double local_sum = 0.0;
    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        local_sum += partial_sums[i];
    }

    double block_sum = blockReduceSum(local_sum);

    if (threadIdx.x == 0) {
        sum_dev[0] = block_sum;
    }
}

// Kernel: Subtract mean from packed RHS (reads sum from sum_dev)
__global__ void kernel_subtract_mean(
    double* __restrict__ packed,
    const double* __restrict__ sum_dev,
    size_t n_total)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_total) {
        double mean = sum_dev[0] / static_cast<double>(n_total);
        packed[idx] -= mean;
    }
}

// Kernel: Unpack solution + apply all BCs in single pass
// Input layout:  packed[(i*Nz + k)*Ny + j]  (cuFFT interleaved batches)
// Output layout: p_ptr[k+Ng][j+Ng][i+Ng]    (ghost cells, field ordering)
// Also fills: x-ghosts (periodic), y-ghosts (Neumann), z-ghosts (periodic)
__global__ void kernel_unpack_and_bc(
    const double* __restrict__ packed,
    double* __restrict__ p_ptr,
    int Nx, int Ny, int Nz, int Ng, int Nx_full, int Ny_full,
    double norm)
{
    const size_t n_total = static_cast<size_t>(Nx) * Ny * Nz;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_total) {
        // Decode linear index to (i, k, j) in FFT order
        int j = idx % Ny;
        size_t mode = idx / Ny;
        int k = mode % Nz;
        int i = mode / Nz;

        double val = packed[idx] * norm;

        // Destination with ghosts: [k+Ng][j+Ng][i+Ng]
        size_t dst_idx = static_cast<size_t>(k + Ng) * Nx_full * Ny_full +
                         (j + Ng) * Nx_full + (i + Ng);
        p_ptr[dst_idx] = val;

        // Fill x-ghosts (periodic) - boundary threads only
        if (i == 0) {
            // x_lo ghost: copy from x_hi interior (i=Nx-1)
            size_t src_mode = static_cast<size_t>(Nx - 1) * Nz + k;
            double src_val = packed[src_mode * Ny + j] * norm;
            p_ptr[(k + Ng) * Nx_full * Ny_full + (j + Ng) * Nx_full + 0] = src_val;
        }
        if (i == Nx - 1) {
            // x_hi ghost: copy from x_lo interior (i=0)
            size_t src_mode = static_cast<size_t>(0) * Nz + k;
            double src_val = packed[src_mode * Ny + j] * norm;
            p_ptr[(k + Ng) * Nx_full * Ny_full + (j + Ng) * Nx_full + (Nx + Ng)] = src_val;
        }

        // Fill y-ghosts (Neumann: dp/dy=0) - boundary threads only
        if (j == 0) {
            // y_lo ghost: copy from first interior row
            p_ptr[(k + Ng) * Nx_full * Ny_full + 0 * Nx_full + (i + Ng)] = val;
        }
        if (j == Ny - 1) {
            // y_hi ghost: copy from last interior row
            p_ptr[(k + Ng) * Nx_full * Ny_full + (Ny + Ng) * Nx_full + (i + Ng)] = val;
        }

        // Fill z-ghosts (periodic) - boundary threads only
        if (k == 0) {
            // z_lo ghost: copy from z_hi interior (k=Nz-1)
            size_t src_mode = static_cast<size_t>(i) * Nz + (Nz - 1);
            double src_val = packed[src_mode * Ny + j] * norm;
            p_ptr[0 * Nx_full * Ny_full + (j + Ng) * Nx_full + (i + Ng)] = src_val;
        }
        if (k == Nz - 1) {
            // z_hi ghost: copy from z_lo interior (k=0)
            size_t src_mode = static_cast<size_t>(i) * Nz + 0;
            double src_val = packed[src_mode * Ny + j] * norm;
            p_ptr[(Nz + Ng) * Nx_full * Ny_full + (j + Ng) * Nx_full + (i + Ng)] = src_val;
        }
    }
}

#endif // USE_GPU_OFFLOAD

FFTPoissonSolver::FFTPoissonSolver(const Mesh& mesh)
    : mesh_(&mesh) {
#ifdef USE_GPU_OFFLOAD
    using_gpu_ = true;
    initialize_fft();
#else
    using_gpu_ = false;
    throw std::runtime_error("FFTPoissonSolver requires GPU offload support");
#endif
}

FFTPoissonSolver::~FFTPoissonSolver() {
#ifdef USE_GPU_OFFLOAD
    if (plans_created_) {
        cufftDestroy(fft_plan_r2c_);
        cufftDestroy(fft_plan_c2r_);
    }
    if (cusparse_handle_) cusparseDestroy(cusparse_handle_);
    if (ev_pack_done_) cudaEventDestroy(ev_pack_done_);
    if (ev_fft_done_) cudaEventDestroy(ev_fft_done_);
    if (stream_) cudaStreamDestroy(stream_);
    if (sum_dev_) cudaFree(sum_dev_);
    if (partial_sums_) cudaFree(partial_sums_);
    if (fft_work_area_) cudaFree(fft_work_area_);
    if (rhs_packed_) cudaFree(rhs_packed_);
    if (p_packed_) cudaFree(p_packed_);
    if (rhs_hat_) cudaFree(rhs_hat_);
    if (p_hat_) cudaFree(p_hat_);
    if (lambda_x_) cudaFree(lambda_x_);
    if (lambda_z_) cudaFree(lambda_z_);
    if (tri_lower_) cudaFree(tri_lower_);
    if (tri_upper_) cudaFree(tri_upper_);
    if (tri_diag_base_) cudaFree(tri_diag_base_);
    if (tri_dl_) cudaFree(tri_dl_);
    if (tri_d_) cudaFree(tri_d_);
    if (tri_du_) cudaFree(tri_du_);
    if (cusparse_buffer_) cudaFree(cusparse_buffer_);
    if (work_c_) cudaFree(work_c_);
    if (work_d_real_) cudaFree(work_d_real_);
    if (work_d_imag_) cudaFree(work_d_imag_);
#endif
}

void FFTPoissonSolver::set_bc(PoissonBC x_lo, PoissonBC x_hi,
                               PoissonBC y_lo, PoissonBC y_hi,
                               PoissonBC z_lo, PoissonBC z_hi) {
    // Verify x and z are periodic
    if (x_lo != PoissonBC::Periodic || x_hi != PoissonBC::Periodic) {
        throw std::runtime_error("FFTPoissonSolver requires periodic BC in x");
    }
    if (z_lo != PoissonBC::Periodic || z_hi != PoissonBC::Periodic) {
        throw std::runtime_error("FFTPoissonSolver requires periodic BC in z");
    }
    bc_y_lo_ = y_lo;
    bc_y_hi_ = y_hi;
}

bool FFTPoissonSolver::is_suitable(PoissonBC x_lo, PoissonBC x_hi,
                                    PoissonBC y_lo, PoissonBC y_hi,
                                    PoissonBC z_lo, PoissonBC z_hi,
                                    bool uniform_x, bool uniform_z) {
    // Must have periodic x and z with uniform spacing
    bool periodic_xz = (x_lo == PoissonBC::Periodic && x_hi == PoissonBC::Periodic &&
                        z_lo == PoissonBC::Periodic && z_hi == PoissonBC::Periodic);
    return periodic_xz && uniform_x && uniform_z;
}

#ifdef USE_GPU_OFFLOAD

void FFTPoissonSolver::initialize_fft() {
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Nz_complex = Nz / 2 + 1;  // R2C output size

    // Allocate packed real arrays (no ghost cells)
    cudaMallocManaged(&rhs_packed_, sizeof(double) * Nx * Ny * Nz);
    cudaMallocManaged(&p_packed_, sizeof(double) * Nx * Ny * Nz);

    // Allocate complex arrays for FFT output
    // Layout: for each y-plane, we have Nx * Nz_complex complex values
    cudaMallocManaged(&rhs_hat_, sizeof(cufftDoubleComplex) * Nx * Ny * Nz_complex);
    cudaMallocManaged(&p_hat_, sizeof(cufftDoubleComplex) * Nx * Ny * Nz_complex);

    // Allocate eigenvalue arrays
    cudaMallocManaged(&lambda_x_, sizeof(double) * Nx);
    cudaMallocManaged(&lambda_z_, sizeof(double) * Nz_complex);

    // Allocate tridiagonal coefficient arrays
    cudaMallocManaged(&tri_lower_, sizeof(double) * Ny);
    cudaMallocManaged(&tri_upper_, sizeof(double) * Ny);
    cudaMallocManaged(&tri_diag_base_, sizeof(double) * Ny);

    // Allocate workspace for Thomas algorithm
    const size_t work_size = static_cast<size_t>(Nx) * Nz_complex * Ny;
    cudaMallocManaged(&work_c_, sizeof(double) * work_size);
    cudaMallocManaged(&work_d_real_, sizeof(double) * work_size);
    cudaMallocManaged(&work_d_imag_, sizeof(double) * work_size);

    // Create dedicated CUDA stream for entire Poisson solve
    // This allows async execution and avoids default stream ordering constraints
    cudaStreamCreate(&stream_);

    // Create CUDA events for stream-to-stream synchronization (no host blocking)
    // cudaEventDisableTiming avoids GPU timestamp overhead
    cudaEventCreateWithFlags(&ev_pack_done_, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&ev_fft_done_, cudaEventDisableTiming);

    // Allocate device scalar for sum (avoids host transfer during mean computation)
    cudaMalloc(&sum_dev_, sizeof(double));

    // Create cuFFT plans for batched 2D FFT with OPTIMIZED LAYOUT
    //
    // Goal: Produce output in [mode][j] layout where mode = kx*Nz_complex + kz
    // This eliminates the transpose needed for cuSPARSE tridiagonal solves.
    //
    // cuFFT output indexing formula:
    //   out[b*odist + (x*onembed[1] + z)*ostride]
    //
    // We want: out[mode*Ny + j] = out[(kx*Nz_complex + kz)*Ny + j]
    // Setting odist=1, ostride=Ny, onembed[1]=Nz_complex:
    //   out[j*1 + (kx*Nz_complex + kz)*Ny] = out[j + mode*Ny] ✓
    //
    // Similarly for input: idist=1, istride=Ny, inembed[1]=Nz
    //   in[j*1 + (i*Nz + k)*Ny] = in[(i*Nz + k)*Ny + j]
    // This is the "interleaved batches" layout with j varying fastest.

    int n[2] = {Nx, Nz};  // Dimensions to transform (2D FFT over x-z)
    int inembed[2] = {Nx, Nz};
    int onembed[2] = {Nx, Nz_complex};

    // OPTIMIZED: Interleaved layout for direct cuSPARSE compatibility
    // Input:  rhs_packed[(i*Nz + k)*Ny + j]  with j fastest
    // Output: rhs_hat[mode*Ny + j]           with j fastest
    int istride = Ny;   // Stride between consecutive (x,z) elements
    int ostride = Ny;   // Same for output
    int idist = 1;      // Distance between y-batches (j and j+1 differ by 1)
    int odist = 1;
    int batch = Ny;

    // Create plans with auto-allocation DISABLED to prevent per-solve allocation
    cufftResult result = cufftCreate(&fft_plan_r2c_);
    if (result != CUFFT_SUCCESS) {
        throw std::runtime_error("Failed to create cuFFT R2C handle");
    }
    result = cufftSetAutoAllocation(fft_plan_r2c_, 0);  // Disable auto work area
    if (result != CUFFT_SUCCESS) {
        throw std::runtime_error("Failed to disable cuFFT R2C auto allocation");
    }

    result = cufftCreate(&fft_plan_c2r_);
    if (result != CUFFT_SUCCESS) {
        throw std::runtime_error("Failed to create cuFFT C2R handle");
    }
    result = cufftSetAutoAllocation(fft_plan_c2r_, 0);  // Disable auto work area
    if (result != CUFFT_SUCCESS) {
        throw std::runtime_error("Failed to disable cuFFT C2R auto allocation");
    }

    // Create plans and query work sizes
    size_t r2c_work_size = 0, c2r_work_size = 0;
    result = cufftMakePlanMany(fft_plan_r2c_, 2, n,
                                inembed, istride, idist,
                                onembed, ostride, odist,
                                CUFFT_D2Z, batch, &r2c_work_size);
    if (result != CUFFT_SUCCESS) {
        throw std::runtime_error("Failed to make cuFFT R2C plan");
    }

    result = cufftMakePlanMany(fft_plan_c2r_, 2, n,
                                onembed, ostride, odist,
                                inembed, istride, idist,
                                CUFFT_Z2D, batch, &c2r_work_size);
    if (result != CUFFT_SUCCESS) {
        throw std::runtime_error("Failed to make cuFFT C2R plan");
    }

    // Allocate unified work area (max of both plans)
    fft_work_size_ = std::max(r2c_work_size, c2r_work_size);
    if (fft_work_size_ > 0) {
        cudaMalloc(&fft_work_area_, fft_work_size_);
        cufftSetWorkArea(fft_plan_r2c_, fft_work_area_);
        cufftSetWorkArea(fft_plan_c2r_, fft_work_area_);
    }

    // Set stream for cuFFT (both plans use same stream)
    cufftSetStream(fft_plan_r2c_, stream_);
    cufftSetStream(fft_plan_c2r_, stream_);

    plans_created_ = true;
    std::cout << "[FFTPoissonSolver] cuFFT work area: " << fft_work_size_ << " bytes (locked)\n";

    // Compute eigenvalues and tridiagonal coefficients
    compute_eigenvalues();
    compute_tridiagonal_coeffs();

    // Initialize cuSPARSE for reference solver
    initialize_cusparse();

    initialized_ = true;
    std::cout << "[FFTPoissonSolver] Initialized with cuFFT (Nx=" << Nx
              << ", Ny=" << Ny << ", Nz=" << Nz << ")"
              << (use_cusparse_ ? " [cuSPARSE tridiag]" : " [Thomas tridiag]")
              << "\n";
}

void FFTPoissonSolver::compute_eigenvalues() {
    const int Nx = mesh_->Nx;
    const int Nz = mesh_->Nz;
    const int Nz_complex = Nz / 2 + 1;
    const double dx = mesh_->dx;
    const double dz = mesh_->dz;
    const double pi = M_PI;

    // Eigenvalues for x direction: λ_x(kx) = (2 - 2*cos(2π*kx/Nx)) / dx²
    for (int kx = 0; kx < Nx; ++kx) {
        lambda_x_[kx] = (2.0 - 2.0 * std::cos(2.0 * pi * kx / Nx)) / (dx * dx);
    }

    // Eigenvalues for z direction (only positive frequencies for R2C)
    for (int kz = 0; kz < Nz_complex; ++kz) {
        lambda_z_[kz] = (2.0 - 2.0 * std::cos(2.0 * pi * kz / Nz)) / (dz * dz);
    }

    cudaDeviceSynchronize();
}

void FFTPoissonSolver::compute_tridiagonal_coeffs() {
    const int Ny = mesh_->Ny;
    const int Ng = mesh_->Nghost;

    // For stretched grids, compute coefficients from mesh spacing
    // aS(j) = 1 / (dy_south * dy_center)
    // aN(j) = 1 / (dy_north * dy_center)
    // where dy_south = y(j) - y(j-1), dy_north = y(j+1) - y(j)
    // and dy_center = (dy_south + dy_north) / 2

    const double* y = mesh_->yc.data();  // Cell centers with ghosts

    for (int j = 0; j < Ny; ++j) {
        const int jg = j + Ng;  // Index with ghost offset

        // Compute local spacings
        double dy_south = y[jg] - y[jg - 1];
        double dy_north = y[jg + 1] - y[jg];
        double dy_center = 0.5 * (dy_south + dy_north);

        double aS = 1.0 / (dy_south * dy_center);
        double aN = 1.0 / (dy_north * dy_center);

        // Apply Neumann BCs at walls
        if (j == 0 && bc_y_lo_ == PoissonBC::Neumann) {
            aS = 0.0;
        }
        if (j == Ny - 1 && bc_y_hi_ == PoissonBC::Neumann) {
            aN = 0.0;
        }

        tri_lower_[j] = aS;
        tri_upper_[j] = aN;
        tri_diag_base_[j] = -(aS + aN);
    }

    cudaDeviceSynchronize();
}

void FFTPoissonSolver::initialize_cusparse() {
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Nz_complex = Nz / 2 + 1;
    const int n_modes = Nx * Nz_complex;

    // Create cuSPARSE handle
    cusparseStatus_t status = cusparseCreate(&cusparse_handle_);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::cerr << "[FFTPoissonSolver] Warning: cuSPARSE init failed, using Thomas\n";
        use_cusparse_ = false;
        return;
    }

    // Set cuSPARSE to use same stream as cuFFT for proper ordering
    cusparseSetStream(cusparse_handle_, stream_);

    // Allocate complex tridiagonal arrays for cuSPARSE
    // For gtsv2StridedBatch: each batch has m elements, batchStride=Ny
    // Total: n_modes batches, each of size Ny
    cudaMallocManaged(&tri_dl_, sizeof(cufftDoubleComplex) * n_modes * Ny);
    cudaMallocManaged(&tri_d_, sizeof(cufftDoubleComplex) * n_modes * Ny);
    cudaMallocManaged(&tri_du_, sizeof(cufftDoubleComplex) * n_modes * Ny);

    // Query buffer size for gtsv2StridedBatch
    // Parameters: m=Ny, batchCount=n_modes, batchStride=Ny
    status = cusparseZgtsv2StridedBatch_bufferSizeExt(
        cusparse_handle_,
        Ny,              // m: system size
        tri_dl_,         // dl: lower diagonal
        tri_d_,          // d: main diagonal
        tri_du_,         // du: upper diagonal
        rhs_hat_,        // x: RHS/solution (in-place)
        n_modes,         // batchCount
        Ny,              // batchStride
        &cusparse_buffer_size_
    );

    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::cerr << "[FFTPoissonSolver] Warning: cuSPARSE buffer query failed, using Thomas\n";
        use_cusparse_ = false;
        return;
    }

    cudaMalloc(&cusparse_buffer_, cusparse_buffer_size_);
    std::cout << "[FFTPoissonSolver] cuSPARSE buffer: " << cusparse_buffer_size_ << " bytes\n";

    // Precompute tridiagonal matrices - they never change between solves!
    // The eigenvalue shift depends only on mode (kx, kz), not on the RHS
    double* lam_x = lambda_x_;
    double* lam_z = lambda_z_;
    double* aS = tri_lower_;
    double* aN = tri_upper_;
    double* diag_base = tri_diag_base_;
    cufftDoubleComplex* dl = tri_dl_;
    cufftDoubleComplex* d = tri_d_;
    cufftDoubleComplex* du = tri_du_;

    #pragma omp target teams distribute parallel for collapse(3) \
        is_device_ptr(dl, d, du, lam_x, lam_z, aS, aN, diag_base)
    for (int kx = 0; kx < Nx; ++kx) {
        for (int kz = 0; kz < Nz_complex; ++kz) {
            for (int j = 0; j < Ny; ++j) {
                const size_t mode = static_cast<size_t>(kx) * Nz_complex + kz;
                const size_t idx = mode * Ny + j;

                // Eigenvalue shift
                double shift = lam_x[kx] + lam_z[kz];
                bool is_zero_mode = (kx == 0 && kz == 0);

                // For zero mode (0,0), pin j=0 to zero
                if (is_zero_mode && j == 0) {
                    dl[idx].x = 0.0;
                    dl[idx].y = 0.0;
                    d[idx].x = 1.0;
                    d[idx].y = 0.0;
                    du[idx].x = 0.0;
                    du[idx].y = 0.0;
                } else {
                    // Lower diagonal (j > 0)
                    dl[idx].x = (j > 0) ? aS[j] : 0.0;
                    dl[idx].y = 0.0;

                    // Main diagonal with eigenvalue shift
                    d[idx].x = diag_base[j] - shift;
                    d[idx].y = 0.0;

                    // Upper diagonal (j < Ny-1)
                    du[idx].x = (j < Ny - 1) ? aN[j] : 0.0;
                    du[idx].y = 0.0;
                }
            }
        }
    }
    cudaDeviceSynchronize();
    std::cout << "[FFTPoissonSolver] Precomputed tridiagonal matrices\n";
}

void FFTPoissonSolver::solve_tridiagonal_cusparse() {
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Nz_complex = Nz / 2 + 1;
    const int n_modes = Nx * Nz_complex;

    // Tridiagonal matrices (tri_dl_, tri_d_, tri_du_) are precomputed at init

    // OPTIMIZED: Solve in-place in rhs_hat_ - eliminates D2D memcpy!
    // cuSPARSE gtsv2StridedBatch overwrites input with solution.
    // cuFFT C2R will then read solution directly from rhs_hat_.
    cufftDoubleComplex* x = rhs_hat_;

    // Fix zero mode (mode=0, j=0): set x[0] to 0 (pinned value for singularity)
    // cudaMemsetAsync on stream_ to zero out the first element (16 bytes)
    cudaMemsetAsync(x, 0, sizeof(cufftDoubleComplex), stream_);

    // Call cuSPARSE batched tridiagonal solver (runs on stream_)
    cusparseStatus_t status = cusparseZgtsv2StridedBatch(
        cusparse_handle_,
        Ny,              // m: system size
        tri_dl_,         // dl: lower diagonal
        tri_d_,          // d: main diagonal
        tri_du_,         // du: upper diagonal
        x,               // x: RHS/solution (in-place)
        n_modes,         // batchCount
        Ny,              // batchStride
        cusparse_buffer_
    );

    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::cerr << "[FFT Solver] cuSPARSE error code: " << status << "\n";
        std::cerr << "  Ny=" << Ny << ", n_modes=" << n_modes << "\n";
        throw std::runtime_error("cuSPARSE gtsv2StridedBatch failed");
    }

    // OPTIMIZED: Solution is now in rhs_hat_, which cuFFT C2R will read from.
    // No copy needed since we eliminated the intermediate p_hat_ buffer for solve.
}

double FFTPoissonSolver::pack_rhs_with_sum(double* rhs_ptr) {
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Ng = mesh_->Nghost;
    const int Nx_full = Nx + 2 * Ng;
    const int Ny_full = Ny + 2 * Ng;
    const size_t total_size = static_cast<size_t>(Nx_full) * Ny_full * (Nz + 2 * Ng);

    double* packed = rhs_packed_;

    // FUSED: Pack from [k][j][i] with ghosts to [(i*Nz+k)][j] and compute sum
    // This fuses pack + sum into one kernel pass for better performance
    double sum = 0.0;
    #pragma omp target teams distribute parallel for collapse(3) reduction(+:sum) \
        map(present, alloc: rhs_ptr[0:total_size]) is_device_ptr(packed)
    for (int i = 0; i < Nx; ++i) {
        for (int k = 0; k < Nz; ++k) {
            for (int j = 0; j < Ny; ++j) {
                // Source index with ghosts: [k+Ng][j+Ng][i+Ng]
                const size_t src_idx = static_cast<size_t>(k + Ng) * Nx_full * Ny_full +
                                       (j + Ng) * Nx_full + (i + Ng);
                // Destination: [(i*Nz + k)][j] with j fastest
                const size_t dst_idx = static_cast<size_t>(i * Nz + k) * Ny + j;
                double val = rhs_ptr[src_idx];
                packed[dst_idx] = val;
                sum += val;
            }
        }
    }
    return sum;
}

void FFTPoissonSolver::subtract_mean(double mean) {
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const size_t n_total = static_cast<size_t>(Nx) * Ny * Nz;
    double* rhs_p = rhs_packed_;

    #pragma omp target teams distribute parallel for is_device_ptr(rhs_p)
    for (size_t i = 0; i < n_total; ++i) {
        rhs_p[i] -= mean;
    }
}

void FFTPoissonSolver::pack_rhs(double* rhs_ptr) {
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Ng = mesh_->Nghost;
    const int Nx_full = Nx + 2 * Ng;
    const int Ny_full = Ny + 2 * Ng;
    const size_t total_size = static_cast<size_t>(Nx_full) * Ny_full * (Nz + 2 * Ng);

    double* packed = rhs_packed_;

    // OPTIMIZED LAYOUT: Pack from [k][j][i] with ghosts to [(i*Nz+k)][j] without ghosts
    // This matches the cuFFT interleaved input layout: rhs_packed[(i*Nz + k)*Ny + j]
    // with j varying fastest, enabling direct cuSPARSE compatibility after FFT.
    #pragma omp target teams distribute parallel for collapse(3) \
        map(present, alloc: rhs_ptr[0:total_size]) is_device_ptr(packed)
    for (int i = 0; i < Nx; ++i) {
        for (int k = 0; k < Nz; ++k) {
            for (int j = 0; j < Ny; ++j) {
                // Source index with ghosts: [k+Ng][j+Ng][i+Ng]
                const size_t src_idx = static_cast<size_t>(k + Ng) * Nx_full * Ny_full +
                                       (j + Ng) * Nx_full + (i + Ng);
                // Destination: [(i*Nz + k)][j] with j fastest
                const size_t dst_idx = static_cast<size_t>(i * Nz + k) * Ny + j;
                packed[dst_idx] = rhs_ptr[src_idx];
            }
        }
    }
}

void FFTPoissonSolver::unpack_solution(double* p_ptr) {
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Ng = mesh_->Nghost;
    const int Nx_full = Nx + 2 * Ng;
    const int Ny_full = Ny + 2 * Ng;
    const size_t total_size = static_cast<size_t>(Nx_full) * Ny_full * (Nz + 2 * Ng);
    const double norm = 1.0 / (Nx * Nz);  // FFT normalization

    double* packed = p_packed_;

    // OPTIMIZED LAYOUT: Unpack from [(i*Nz+k)][j] back to [k][j][i] with ghosts
    // This matches the cuFFT interleaved output: p_packed[(i*Nz + k)*Ny + j]
    #pragma omp target teams distribute parallel for collapse(3) \
        map(present, alloc: p_ptr[0:total_size]) is_device_ptr(packed)
    for (int i = 0; i < Nx; ++i) {
        for (int k = 0; k < Nz; ++k) {
            for (int j = 0; j < Ny; ++j) {
                // Source: [(i*Nz + k)][j] with j fastest
                const size_t src_idx = static_cast<size_t>(i * Nz + k) * Ny + j;
                // Destination with ghosts: [k+Ng][j+Ng][i+Ng]
                const size_t dst_idx = static_cast<size_t>(k + Ng) * Nx_full * Ny_full +
                                       (j + Ng) * Nx_full + (i + Ng);
                p_ptr[dst_idx] = packed[src_idx] * norm;
            }
        }
    }
}

void FFTPoissonSolver::unpack_and_apply_bc(double* p_ptr) {
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Ng = mesh_->Nghost;
    const int Nx_full = Nx + 2 * Ng;
    const int Ny_full = Ny + 2 * Ng;
    const size_t total_size = static_cast<size_t>(Nx_full) * Ny_full * (Nz + 2 * Ng);
    const double norm = 1.0 / (Nx * Nz);  // FFT normalization

    double* packed = p_packed_;

    // FUSED: Unpack interior + fill all ghost cells in one pass
    // This eliminates 3 separate BC kernels and improves memory access
    #pragma omp target teams distribute parallel for collapse(3) \
        map(present, alloc: p_ptr[0:total_size]) is_device_ptr(packed)
    for (int i = 0; i < Nx; ++i) {
        for (int k = 0; k < Nz; ++k) {
            for (int j = 0; j < Ny; ++j) {
                // Source: [(i*Nz + k)][j] with j fastest
                const size_t src_idx = static_cast<size_t>(i * Nz + k) * Ny + j;
                const double val = packed[src_idx] * norm;

                // Destination with ghosts: [k+Ng][j+Ng][i+Ng]
                const size_t dst_idx = static_cast<size_t>(k + Ng) * Nx_full * Ny_full +
                                       (j + Ng) * Nx_full + (i + Ng);
                p_ptr[dst_idx] = val;

                // Fill x-ghosts for boundary cells (periodic)
                if (i == 0) {
                    // x_lo ghost: copy from x_hi interior (i=Nx-1)
                    p_ptr[(k + Ng) * Nx_full * Ny_full + (j + Ng) * Nx_full + 0] =
                        packed[static_cast<size_t>((Nx - 1) * Nz + k) * Ny + j] * norm;
                }
                if (i == Nx - 1) {
                    // x_hi ghost: copy from x_lo interior (i=0)
                    p_ptr[(k + Ng) * Nx_full * Ny_full + (j + Ng) * Nx_full + (Nx + Ng)] =
                        packed[static_cast<size_t>(0 * Nz + k) * Ny + j] * norm;
                }

                // Fill y-ghosts for boundary cells (Neumann: dp/dy=0)
                if (j == 0) {
                    // y_lo ghost: copy from first interior row
                    p_ptr[(k + Ng) * Nx_full * Ny_full + 0 * Nx_full + (i + Ng)] = val;
                }
                if (j == Ny - 1) {
                    // y_hi ghost: copy from last interior row
                    p_ptr[(k + Ng) * Nx_full * Ny_full + (Ny + Ng) * Nx_full + (i + Ng)] = val;
                }

                // Fill z-ghosts for boundary cells (periodic)
                if (k == 0) {
                    // z_lo ghost: copy from z_hi interior (k=Nz-1)
                    p_ptr[0 * Nx_full * Ny_full + (j + Ng) * Nx_full + (i + Ng)] =
                        packed[static_cast<size_t>(i * Nz + (Nz - 1)) * Ny + j] * norm;
                }
                if (k == Nz - 1) {
                    // z_hi ghost: copy from z_lo interior (k=0)
                    p_ptr[(Nz + Ng) * Nx_full * Ny_full + (j + Ng) * Nx_full + (i + Ng)] =
                        packed[static_cast<size_t>(i * Nz + 0) * Ny + j] * norm;
                }
            }
        }
    }
}

void FFTPoissonSolver::apply_bc_device(double* p_ptr) {
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Ng = mesh_->Nghost;
    const int Nx_full = Nx + 2 * Ng;
    const int Ny_full = Ny + 2 * Ng;
    const size_t total_size = static_cast<size_t>(Nx_full) * Ny_full * (Nz + 2 * Ng);

    // X boundaries (periodic)
    #pragma omp target teams distribute parallel for collapse(2) map(present, alloc: p_ptr[0:total_size])
    for (int k = 0; k < Nz; ++k) {
        for (int j = 0; j < Ny; ++j) {
            const int kk = k + Ng;
            const int jj = j + Ng;
            // x_lo ghost from x_hi interior
            p_ptr[kk * Nx_full * Ny_full + jj * Nx_full + 0] =
                p_ptr[kk * Nx_full * Ny_full + jj * Nx_full + Nx];
            // x_hi ghost from x_lo interior
            p_ptr[kk * Nx_full * Ny_full + jj * Nx_full + (Nx + Ng)] =
                p_ptr[kk * Nx_full * Ny_full + jj * Nx_full + Ng];
        }
    }

    // Y boundaries (Neumann)
    #pragma omp target teams distribute parallel for collapse(2) map(present, alloc: p_ptr[0:total_size])
    for (int k = 0; k < Nz; ++k) {
        for (int i = 0; i < Nx + 2 * Ng; ++i) {
            const int kk = k + Ng;
            // y_lo ghost (Neumann: dp/dy = 0)
            p_ptr[kk * Nx_full * Ny_full + 0 * Nx_full + i] =
                p_ptr[kk * Nx_full * Ny_full + Ng * Nx_full + i];
            // y_hi ghost
            p_ptr[kk * Nx_full * Ny_full + (Ny + Ng) * Nx_full + i] =
                p_ptr[kk * Nx_full * Ny_full + (Ny + Ng - 1) * Nx_full + i];
        }
    }

    // Z boundaries (periodic)
    #pragma omp target teams distribute parallel for collapse(2) map(present, alloc: p_ptr[0:total_size])
    for (int j = 0; j < Ny + 2 * Ng; ++j) {
        for (int i = 0; i < Nx + 2 * Ng; ++i) {
            // z_lo ghost
            p_ptr[0 * Nx_full * Ny_full + j * Nx_full + i] =
                p_ptr[Nz * Nx_full * Ny_full + j * Nx_full + i];
            // z_hi ghost
            p_ptr[(Nz + Ng) * Nx_full * Ny_full + j * Nx_full + i] =
                p_ptr[Ng * Nx_full * Ny_full + j * Nx_full + i];
        }
    }
}

// ============================================================================
// CUDA Kernel Launchers (run on stream_ for GPU-resident operation)
// ============================================================================

void FFTPoissonSolver::launch_pack_and_sum(double* rhs_dev) {
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Ng = mesh_->Nghost;
    const int Nx_full = Nx + 2 * Ng;
    const int Ny_full = Ny + 2 * Ng;
    const size_t n_total = static_cast<size_t>(Nx) * Ny * Nz;

    // PURE CUDA: rhs_dev is already a CUDA device pointer (via omp_get_mapped_ptr)
    const int block_size = 256;
    const int num_blocks = (n_total + block_size - 1) / block_size;

    // Allocate/resize partial sums buffer if needed
    if (partial_sums_size_ < static_cast<size_t>(num_blocks)) {
        if (partial_sums_) cudaFree(partial_sums_);
        cudaMalloc(&partial_sums_, sizeof(double) * num_blocks);
        partial_sums_size_ = num_blocks;
    }

    // Launch pack + partial sum kernel on stream_
    kernel_pack_and_partial_sum<<<num_blocks, block_size, 0, stream_>>>(
        rhs_dev, rhs_packed_, partial_sums_,
        Nx, Ny, Nz, Ng, Nx_full, Ny_full);

    // Launch final reduction kernel to compute total sum into sum_dev_
    kernel_final_reduce<<<1, 256, 0, stream_>>>(
        partial_sums_, sum_dev_, num_blocks);
}

void FFTPoissonSolver::launch_subtract_mean(size_t n_total) {
    const int block_size = 256;
    const int num_blocks = (n_total + block_size - 1) / block_size;

    // Launch subtract mean kernel on stream_
    kernel_subtract_mean<<<num_blocks, block_size, 0, stream_>>>(
        rhs_packed_, sum_dev_, n_total);
}

void FFTPoissonSolver::launch_unpack_and_bc(double* p_dev) {
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Ng = mesh_->Nghost;
    const int Nx_full = Nx + 2 * Ng;
    const int Ny_full = Ny + 2 * Ng;
    const size_t n_total = static_cast<size_t>(Nx) * Ny * Nz;
    const double norm = 1.0 / (Nx * Nz);  // FFT normalization

    // PURE CUDA: p_dev is already a CUDA device pointer (via omp_get_mapped_ptr)
    const int block_size = 256;
    const int num_blocks = (n_total + block_size - 1) / block_size;

    // Launch unpack + BC kernel on stream_
    kernel_unpack_and_bc<<<num_blocks, block_size, 0, stream_>>>(
        p_packed_, p_dev,
        Nx, Ny, Nz, Ng, Nx_full, Ny_full, norm);
}

int FFTPoissonSolver::solve_device(double* rhs_ptr, double* p_ptr, const PoissonConfig& cfg) {
    if (!initialized_) {
        throw std::runtime_error("FFTPoissonSolver not initialized");
    }

    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const size_t n_total = static_cast<size_t>(Nx) * Ny * Nz;

    // ========================================================================
    // PURE CUDA SOLVE: All operations on stream_ (no OMP target regions)
    // ========================================================================
    //
    // Data flow (all on stream_):
    //   rhs_dev → [pack+sum] → [subtract_mean] → [cuFFT R2C] →
    //   [cuSPARSE tridiag] → [cuFFT C2R] → [unpack+BC] → p_dev
    //
    // No inter-stream sync needed - everything runs on stream_
    // ========================================================================

    // Convert OMP-mapped host pointers to CUDA device pointers
    int device = omp_get_default_device();
    double* rhs_dev = static_cast<double*>(omp_get_mapped_ptr(rhs_ptr, device));
    double* p_dev = static_cast<double*>(omp_get_mapped_ptr(p_ptr, device));

    // Step 1: Pack RHS + compute sum (CUDA kernel on stream_)
    launch_pack_and_sum(rhs_dev);

    // Step 2: Subtract mean from packed RHS (CUDA kernel on stream_)
    launch_subtract_mean(n_total);

    // Step 3: Forward 2D FFT (R2C) - runs on stream_
    cufftResult result = cufftExecD2Z(fft_plan_r2c_, rhs_packed_, rhs_hat_);
    if (result != CUFFT_SUCCESS) {
        throw std::runtime_error("cuFFT R2C failed");
    }

    // Step 4: Solve tridiagonal systems in y for each Fourier mode
    // cuSPARSE runs on stream_, ordering is automatic (no sync needed)
    if (use_cusparse_) {
        solve_tridiagonal_cusparse();
    } else {
        // Legacy OMP Thomas algorithm (slower, kept for reference)
        // This path requires syncs and is not recommended
        cudaStreamSynchronize(stream_);

        cufftDoubleComplex* rhs_h = rhs_hat_;
        cufftDoubleComplex* p_h = p_hat_;
        double* lam_x = lambda_x_;
        double* lam_z = lambda_z_;
        double* aS = tri_lower_;
        double* aN = tri_upper_;
        double* diag_base = tri_diag_base_;
        double* work_c = work_c_;
        double* work_d_r = work_d_real_;
        double* work_d_i = work_d_imag_;
        const int Nz_complex = Nz / 2 + 1;

        for (int j = 0; j < Ny; ++j) {
            const int j_local = j;
            #pragma omp target teams distribute parallel for collapse(2) \
                is_device_ptr(rhs_h, lam_x, lam_z, aS, aN, diag_base, work_c, work_d_r, work_d_i)
            for (int kx = 0; kx < Nx; ++kx) {
                for (int kz = 0; kz < Nz_complex; ++kz) {
                    const size_t mode_idx = static_cast<size_t>(kx) * Nz_complex + kz;
                    const size_t work_idx = mode_idx * Ny + j_local;
                    const size_t rhs_idx = static_cast<size_t>(j_local) * Nx * Nz_complex + mode_idx;
                    double shift = lam_x[kx] + lam_z[kz];
                    bool is_zero_mode = (kx == 0 && kz == 0);
                    double diag = diag_base[j_local] - shift;
                    double lower = (j_local > 0) ? aS[j_local] : 0.0;
                    double upper = (j_local < Ny - 1) ? aN[j_local] : 0.0;
                    double rhs_r = rhs_h[rhs_idx].x;
                    double rhs_i = rhs_h[rhs_idx].y;
                    if (is_zero_mode && j_local == 0) {
                        diag = 1.0; lower = 0.0; upper = 0.0; rhs_r = 0.0; rhs_i = 0.0;
                    }
                    if (j_local == 0) {
                        work_c[work_idx] = upper / diag;
                        work_d_r[work_idx] = rhs_r / diag;
                        work_d_i[work_idx] = rhs_i / diag;
                    } else {
                        const size_t prev_idx = mode_idx * Ny + (j_local - 1);
                        double denom = diag - lower * work_c[prev_idx];
                        work_c[work_idx] = upper / denom;
                        work_d_r[work_idx] = (rhs_r - lower * work_d_r[prev_idx]) / denom;
                        work_d_i[work_idx] = (rhs_i - lower * work_d_i[prev_idx]) / denom;
                    }
                }
            }
        }
        for (int j = Ny - 1; j >= 0; --j) {
            const int j_local = j;
            #pragma omp target teams distribute parallel for collapse(2) \
                is_device_ptr(p_h, work_c, work_d_r, work_d_i)
            for (int kx = 0; kx < Nx; ++kx) {
                for (int kz = 0; kz < Nz_complex; ++kz) {
                    const size_t mode_idx = static_cast<size_t>(kx) * Nz_complex + kz;
                    const size_t work_idx = mode_idx * Ny + j_local;
                    const size_t out_idx = static_cast<size_t>(j_local) * Nx * Nz_complex + mode_idx;
                    if (j_local == Ny - 1) {
                        p_h[out_idx].x = work_d_r[work_idx];
                        p_h[out_idx].y = work_d_i[work_idx];
                    } else {
                        const size_t next_out_idx = static_cast<size_t>(j_local + 1) * Nx * Nz_complex + mode_idx;
                        p_h[out_idx].x = work_d_r[work_idx] - work_c[work_idx] * p_h[next_out_idx].x;
                        p_h[out_idx].y = work_d_i[work_idx] - work_c[work_idx] * p_h[next_out_idx].y;
                    }
                }
            }
        }
        cudaDeviceSynchronize();
    }

    // Step 5: Inverse 2D FFT (C2R) - runs on stream_
    // Read from rhs_hat_ (contains solution from in-place cuSPARSE solve)
    result = cufftExecZ2D(fft_plan_c2r_, rhs_hat_, p_packed_);
    if (result != CUFFT_SUCCESS) {
        throw std::runtime_error("cuFFT C2R failed");
    }

    // Step 6: Unpack solution + apply BCs (CUDA kernel on stream_)
    launch_unpack_and_bc(p_dev);

    // Sync stream_ to ensure data is visible to caller (OMP may use it next)
    // This is the only sync point - all work was on stream_
    cudaStreamSynchronize(stream_);

    residual_ = 0.0;  // Direct solver, no residual
    return 1;  // "1 iteration" for a direct solver
}

#endif // USE_GPU_OFFLOAD

} // namespace nncfd

#endif // USE_GPU_OFFLOAD
