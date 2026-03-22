#ifdef USE_GPU_OFFLOAD

#include "poisson_solver_fft.hpp"
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <cstdlib>
#include <vector>

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

// Kernel: Pack RHS from ghost layout to FFT layout + compute volume-weighted partial sums
// Input layout:  rhs_ptr[k+Ng][j+Ng][i+Ng]  (ghost cells, field ordering)
// Output layout: packed[(i*Nz + k)*Ny + j]  (cuFFT interleaved batches)
// Volume-weighted sum: Σ f_{ijk} * dyv[j] for solvability on stretched grids
__global__ void kernel_pack_and_partial_sum(
    const double* __restrict__ rhs_ptr,
    double* __restrict__ packed,
    double* __restrict__ partial_sums,
    const double* __restrict__ dyv,
    int Nx, int Ny, int Nz, int Ng, int Nx_full, int Ny_full)
{
    const size_t n_total = static_cast<size_t>(Nx) * Ny * Nz;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    double local_sum = 0.0;

    if (idx < n_total) {
        // Decode by OUTPUT (FFT) layout: j fastest, then k, then i
        // This makes consecutive threads write consecutive packed addresses (coalesced writes)
        int j = idx % Ny;
        size_t mode = idx / Ny;
        int k = mode % Nz;
        int i = mode / Nz;

        // Source from field layout with ghosts: [k+Ng][j+Ng][i+Ng]
        // This read is scattered (consecutive threads differ in j, stride = Nx_full)
        // but L2 cache absorbs scattered reads better than scattered writes
        size_t src_idx = static_cast<size_t>(k + Ng) * Nx_full * Ny_full +
                         (j + Ng) * Nx_full + (i + Ng);

        double val = rhs_ptr[src_idx];
        packed[idx] = val;
        local_sum = val * dyv[j];  // Volume-weighted for solvability
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

// Kernel: Subtract volume-weighted mean from packed RHS (reads sum from sum_dev)
// mean = Σ(f*dyv) / total_volume ensures solvability on stretched grids
__global__ void kernel_subtract_mean(
    double* __restrict__ packed,
    const double* __restrict__ sum_dev,
    double total_volume,
    size_t n_total)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_total) {
        double mean = sum_dev[0] / total_volume;
        packed[idx] -= mean;
    }
}

// Kernel: Unpack solution from FFT layout to field layout (interior only)
// Threads indexed by OUTPUT layout for coalesced writes.
// Input layout:  packed[(i*Nz + k)*Ny + j]  (cuFFT interleaved batches)
// Output layout: p_ptr[(k+Ng)*Ny_full*Nx_full + (j+Ng)*Nx_full + (i+Ng)]
__global__ void kernel_unpack_transpose(
    const double* __restrict__ packed,
    double* __restrict__ p_ptr,
    int Nx, int Ny, int Nz, int Ng, int Nx_full, int Ny_full,
    double norm)
{
    const size_t n_total = static_cast<size_t>(Nx) * Ny * Nz;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_total) {
        // Decode by OUTPUT layout: i fastest, then j, then k
        // This makes consecutive threads write consecutive i-addresses (coalesced)
        int i = idx % Nx;
        int j = (idx / Nx) % Ny;
        int k = idx / (Nx * Ny);

        // Read from packed FFT layout: [(i*Nz + k)*Ny + j]
        size_t src_idx = (static_cast<size_t>(i) * Nz + k) * Ny + j;
        double val = packed[src_idx] * norm;

        // Write to field layout with ghosts — coalesced in i
        size_t dst_idx = static_cast<size_t>(k + Ng) * Nx_full * Ny_full +
                         (j + Ng) * Nx_full + (i + Ng);
        p_ptr[dst_idx] = val;
    }
}

// Kernel: Fill ghost cells (BCs) after unpack — separate for clarity and no divergence
// x: periodic, y: Neumann (dp/dy=0), z: periodic
__global__ void kernel_fill_ghosts(
    double* __restrict__ p_ptr,
    int Nx, int Ny, int Nz, int Ng, int Nx_full, int Ny_full)
{
    // Total ghost cells to fill: 2*Ng slabs per direction
    // x-ghosts: Ng * Ny * Nz * 2 sides
    // y-ghosts: Nx * Ng * Nz * 2 sides
    // z-ghosts: Nx * Ny * Ng * 2 sides
    // We handle each direction in sequence with simple loops

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t Ny_Nz = static_cast<size_t>(Ny) * Nz;
    const size_t Nx_Nz = static_cast<size_t>(Nx) * Nz;
    const size_t Nx_Ny = static_cast<size_t>(Nx) * Ny;
    const size_t Nf = static_cast<size_t>(Nx_full);
    const size_t plane = Nf * Ny_full;

    // X-periodic ghosts: Ng*Ny*Nz cells per side
    const size_t x_total = static_cast<size_t>(Ng) * Ny * Nz * 2;
    if (idx < x_total) {
        size_t half = static_cast<size_t>(Ng) * Ny * Nz;
        bool is_lo = (idx < half);
        size_t lidx = is_lo ? idx : (idx - half);
        int g = lidx % Ng;
        int j = (lidx / Ng) % Ny;
        int k = lidx / (Ng * Ny);
        size_t base = static_cast<size_t>(k + Ng) * plane + (j + Ng) * Nf;
        if (is_lo) {
            // x_lo ghost[g] = interior[Nx - Ng + g]
            p_ptr[base + g] = p_ptr[base + (Nx - Ng + g + Ng)];
        } else {
            // x_hi ghost[Ng+Nx+g] = interior[g]
            p_ptr[base + (Ng + Nx + g)] = p_ptr[base + (g + Ng)];
        }
        return;
    }
    idx -= x_total;

    // Y-Neumann ghosts: Ng*Nx*Nz cells per side
    const size_t y_total = static_cast<size_t>(Ng) * Nx * Nz * 2;
    if (idx < y_total) {
        size_t half = static_cast<size_t>(Ng) * Nx * Nz;
        bool is_lo = (idx < half);
        size_t lidx = is_lo ? idx : (idx - half);
        int g = lidx % Ng;
        int i = (lidx / Ng) % Nx;
        int k = lidx / (Ng * Nx);
        size_t k_off = static_cast<size_t>(k + Ng) * plane;
        if (is_lo) {
            // y_lo ghost[g] = interior[j=Ng] (Neumann: dp/dy=0)
            p_ptr[k_off + g * Nf + (i + Ng)] = p_ptr[k_off + Ng * Nf + (i + Ng)];
        } else {
            // y_hi ghost[Ng+Ny+g] = interior[j=Ng+Ny-1]
            p_ptr[k_off + (Ng + Ny + g) * Nf + (i + Ng)] = p_ptr[k_off + (Ng + Ny - 1) * Nf + (i + Ng)];
        }
        return;
    }
    idx -= y_total;

    // Z-periodic ghosts: Ng*Nx*Ny cells per side
    const size_t z_total = static_cast<size_t>(Ng) * Nx * Ny * 2;
    if (idx < z_total) {
        size_t half = static_cast<size_t>(Ng) * Nx * Ny;
        bool is_lo = (idx < half);
        size_t lidx = is_lo ? idx : (idx - half);
        int g = lidx % Ng;
        int i = (lidx / Ng) % Nx;
        int j = lidx / (Ng * Nx);
        size_t jrow = static_cast<size_t>(j + Ng) * Nf + (i + Ng);
        if (is_lo) {
            // z_lo ghost[g] = interior[Nz - Ng + g]
            p_ptr[g * plane + jrow] = p_ptr[(Nz - Ng + g + Ng) * plane + jrow];
        } else {
            // z_hi ghost[Ng+Nz+g] = interior[g]
            p_ptr[(Ng + Nz + g) * plane + jrow] = p_ptr[(g + Ng) * plane + jrow];
        }
    }
}

#endif // USE_GPU_OFFLOAD

FFTPoissonSolver::FFTPoissonSolver(const Mesh& mesh)
    : mesh_(&mesh) {
    // FFT solver supports Nghost >= 1 (apply_bc_device fills all ghost layers)
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
    if (dyv_dev_) cudaFree(dyv_dev_);
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

void FFTPoissonSolver::set_space_order(int order) {
    if (order != 2 && order != 4) {
        std::cerr << "[FFTPoissonSolver] Warning: space_order must be 2 or 4, got "
                  << order << ". Using 2.\n";
        order = 2;
    }
    if (order != space_order_) {
        space_order_ = order;
        // Recompute eigenvalues and tridiagonal matrices with new order
        if (initialized_) {
            std::cout << "[FFTPoissonSolver] Recomputing eigenvalues for O" << order << "\n";
            compute_eigenvalues();
            initialize_cusparse();  // Recompute tridiagonal matrices
        }
    }
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

    // Allocate device copy of dyv weights for volume-weighted mean subtraction
    // On stretched grids, solvability requires Σ f*dyv = 0, not Σ f = 0
    // On uniform grids, dyv is empty — use constant dy for all cells
    {
        const int Ng = mesh_->Nghost;
        const bool stretched = !mesh_->dyv.empty();
        std::vector<double> dyv_host(Ny);
        double Ly = 0.0;
        for (int j = 0; j < Ny; ++j) {
            dyv_host[j] = stretched ? mesh_->dyv[j + Ng] : mesh_->dy;
            Ly += dyv_host[j];
        }
        total_volume_ = static_cast<double>(Nx) * Nz * Ly;
        cudaMalloc(&dyv_dev_, sizeof(double) * Ny);
        cudaMemcpy(dyv_dev_, dyv_host.data(), sizeof(double) * Ny, cudaMemcpyHostToDevice);
    }

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

    if (space_order_ == 4) {
        // O4 MAC-consistent eigenvalues for staggered (MAC) grid Laplacian.
        //
        // Derivation: On a MAC grid, the Laplacian is ∇²p = div(grad(p)) where:
        //   - grad (cell→face): Dcf with stencil (1, -27, 27, -1)/(24h)
        //   - div (face→cell):  Dfc with stencil (1, -27, 27, -1)/(24h)
        //
        // The Fourier symbol of the composed operator Dfc ∘ Dcf is:
        //   σ_Dcf(θ) = (1 - 27*e^{-iθ} + 27 - e^{iθ}) / (24h)  [evaluated at half-shift]
        //   σ_Dfc(θ) = (e^{iθ} - 27 + 27*e^{-iθ} - e^{-2iθ}) / (24h)
        //
        // After simplification (product of symbols, using e^{iθ} + e^{-iθ} = 2cos(θ)):
        //   λ(θ) = (1460 - 1566*cos(θ) + 108*cos(2θ) - 2*cos(3θ)) / (576*h²)
        //
        // This matches the exact discrete Laplacian used by the O4 projection,
        // ensuring the FFT Poisson solve is spectrally consistent with the
        // finite-difference operators. See: Morinishi et al. (1998) JCP 143:90-124
        // for MAC-consistent discrete operators on staggered grids.
        std::cout << "[FFTPoissonSolver] Using O4 MAC-consistent eigenvalues\n";

        for (int kx = 0; kx < Nx; ++kx) {
            double theta = 2.0 * pi * kx / Nx;
            double c1 = std::cos(theta);
            double c2 = std::cos(2.0 * theta);
            double c3 = std::cos(3.0 * theta);
            lambda_x_[kx] = (1460.0 - 1566.0*c1 + 108.0*c2 - 2.0*c3) / (576.0 * dx * dx);
        }

        for (int kz = 0; kz < Nz_complex; ++kz) {
            double theta = 2.0 * pi * kz / Nz;
            double c1 = std::cos(theta);
            double c2 = std::cos(2.0 * theta);
            double c3 = std::cos(3.0 * theta);
            lambda_z_[kz] = (1460.0 - 1566.0*c1 + 108.0*c2 - 2.0*c3) / (576.0 * dz * dz);
        }
    } else {
        // O2 eigenvalues: λ(θ) = (2 - 2*cos(θ)) / h² = 4*sin²(θ/2) / h²
        // Standard second-order discrete Laplacian
        for (int kx = 0; kx < Nx; ++kx) {
            lambda_x_[kx] = (2.0 - 2.0 * std::cos(2.0 * pi * kx / Nx)) / (dx * dx);
        }

        for (int kz = 0; kz < Nz_complex; ++kz) {
            lambda_z_[kz] = (2.0 - 2.0 * std::cos(2.0 * pi * kz / Nz)) / (dz * dz);
        }
    }

    cudaDeviceSynchronize();
}

void FFTPoissonSolver::compute_tridiagonal_coeffs() {
    const int Ny = mesh_->Ny;
    const int Ng = mesh_->Nghost;

    // For stretched grids, compute variable-coefficient tridiagonal entries.
    // Uses the mesh's D·G=L-consistent Laplacian coefficients:
    //   aS = 1/(dyv[j] * dyc_south), aN = 1/(dyv[j] * dyc_north)
    // where dyv = face spacing, dyc = center-to-center spacing.
    // For uniform grids, yLap arrays are empty so we fall back to 1/dy^2.
    const bool stretched = mesh_->is_y_stretched();

    for (int j = 0; j < Ny; ++j) {
        const int jg = j + Ng;  // Index with ghost offset

        double aS, aN;
        if (stretched) {
            aS = mesh_->yLap_aS[jg];
            aN = mesh_->yLap_aN[jg];
        } else {
            double invDy2 = 1.0 / (mesh_->dy * mesh_->dy);
            aS = invDy2;
            aN = invDy2;
        }

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

    // Free existing buffers if reinitializing (e.g., after set_space_order)
    if (tri_dl_) { cudaFree(tri_dl_); tri_dl_ = nullptr; }
    if (tri_d_) { cudaFree(tri_d_); tri_d_ = nullptr; }
    if (tri_du_) { cudaFree(tri_du_); tri_du_ = nullptr; }
    if (cusparse_buffer_) { cudaFree(cusparse_buffer_); cusparse_buffer_ = nullptr; }

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
        map(present: rhs_ptr[0:total_size]) is_device_ptr(packed)
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
        map(present: rhs_ptr[0:total_size]) is_device_ptr(packed)
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
        map(present: p_ptr[0:total_size]) is_device_ptr(packed)
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

    // Step 1: Unpack interior cells from FFT result
    #pragma omp target teams distribute parallel for collapse(3) \
        map(present: p_ptr[0:total_size]) is_device_ptr(packed)
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
            }
        }
    }

    // Step 2: Apply boundary conditions (reuse apply_bc_device)
    apply_bc_device(p_ptr);
}

void FFTPoissonSolver::apply_bc_device(double* p_ptr) {
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Ng = mesh_->Nghost;
    const int Nx_full = Nx + 2 * Ng;
    const int Ny_full = Ny + 2 * Ng;
    const size_t total_size = static_cast<size_t>(Nx_full) * Ny_full * (Nz + 2 * Ng);

    // Fill ALL Ng ghost layers for each boundary

    // X boundaries (periodic) - fill all Ng ghost layers
    #pragma omp target teams distribute parallel for collapse(3) map(present: p_ptr[0:total_size])
    for (int g = 0; g < Ng; ++g) {
        for (int k = 0; k < Nz; ++k) {
            for (int j = 0; j < Ny; ++j) {
                const int kk = k + Ng;
                const int jj = j + Ng;
                // x_lo ghost layer g: ghost[g] = interior[Nx + g] (periodic wrap)
                p_ptr[kk * Nx_full * Ny_full + jj * Nx_full + g] =
                    p_ptr[kk * Nx_full * Ny_full + jj * Nx_full + (Nx + g)];
                // x_hi ghost layer g: ghost[Ng + Nx + g] = interior[Ng + g]
                p_ptr[kk * Nx_full * Ny_full + jj * Nx_full + (Ng + Nx + g)] =
                    p_ptr[kk * Nx_full * Ny_full + jj * Nx_full + (Ng + g)];
            }
        }
    }

    // Y boundaries (Neumann dp/dy = 0) - fill all Ng ghost layers
    #pragma omp target teams distribute parallel for collapse(3) map(present: p_ptr[0:total_size])
    for (int g = 0; g < Ng; ++g) {
        for (int k = 0; k < Nz; ++k) {
            for (int i = 0; i < Nx + 2 * Ng; ++i) {
                const int kk = k + Ng;
                // y_lo ghost layer g: all ghost layers copy from first interior cell (Neumann)
                p_ptr[kk * Nx_full * Ny_full + g * Nx_full + i] =
                    p_ptr[kk * Nx_full * Ny_full + Ng * Nx_full + i];
                // y_hi ghost layer g: all ghost layers copy from last interior cell
                p_ptr[kk * Nx_full * Ny_full + (Ng + Ny + g) * Nx_full + i] =
                    p_ptr[kk * Nx_full * Ny_full + (Ng + Ny - 1) * Nx_full + i];
            }
        }
    }

    // Z boundaries (periodic) - fill all Ng ghost layers
    #pragma omp target teams distribute parallel for collapse(3) map(present: p_ptr[0:total_size])
    for (int g = 0; g < Ng; ++g) {
        for (int j = 0; j < Ny + 2 * Ng; ++j) {
            for (int i = 0; i < Nx + 2 * Ng; ++i) {
                // z_lo ghost layer g: ghost[g] = interior[Nz + g] (periodic wrap)
                p_ptr[g * Nx_full * Ny_full + j * Nx_full + i] =
                    p_ptr[(Nz + g) * Nx_full * Ny_full + j * Nx_full + i];
                // z_hi ghost layer g: ghost[Ng + Nz + g] = interior[Ng + g]
                p_ptr[(Ng + Nz + g) * Nx_full * Ny_full + j * Nx_full + i] =
                    p_ptr[(Ng + g) * Nx_full * Ny_full + j * Nx_full + i];
            }
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
    // Uses dyv_dev_ for volume-weighted sum (solvability on stretched grids)
    kernel_pack_and_partial_sum<<<num_blocks, block_size, 0, stream_>>>(
        rhs_dev, rhs_packed_, partial_sums_, dyv_dev_,
        Nx, Ny, Nz, Ng, Nx_full, Ny_full);

    // Launch final reduction kernel to compute total sum into sum_dev_
    kernel_final_reduce<<<1, 256, 0, stream_>>>(
        partial_sums_, sum_dev_, num_blocks);
}

void FFTPoissonSolver::launch_subtract_mean(size_t n_total) {
    const int block_size = 256;
    const int num_blocks = (n_total + block_size - 1) / block_size;

    // Launch subtract mean kernel on stream_
    // Uses total_volume_ for volume-weighted mean (solvability on stretched grids)
    kernel_subtract_mean<<<num_blocks, block_size, 0, stream_>>>(
        rhs_packed_, sum_dev_, total_volume_, n_total);
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

    // Launch transpose kernel (output-coalesced: threads indexed by field layout)
    kernel_unpack_transpose<<<num_blocks, block_size, 0, stream_>>>(
        p_packed_, p_dev,
        Nx, Ny, Nz, Ng, Nx_full, Ny_full, norm);

    // Launch ghost fill kernel (separate for no branch divergence)
    const size_t n_ghosts = static_cast<size_t>(Ng) * (Ny * Nz + Nx * Nz + Nx * Ny) * 2;
    const int ghost_blocks = (n_ghosts + block_size - 1) / block_size;
    kernel_fill_ghosts<<<ghost_blocks, block_size, 0, stream_>>>(
        p_dev, Nx, Ny, Nz, Ng, Nx_full, Ny_full);
}

int FFTPoissonSolver::solve_device(double* rhs_ptr, double* p_ptr, const PoissonConfig& cfg) {
    // Track Poisson solve counts (enable with POISSON_STATS=1)
    static bool print_stats = (std::getenv("POISSON_STATS") != nullptr);
    static int solve_count = 0;
    ++solve_count;
    if (print_stats) {
        std::cerr << "[FFT Poisson] solve #" << solve_count << "\n";
    }

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
