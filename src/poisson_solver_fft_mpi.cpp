/// @file poisson_solver_fft_mpi.cpp
/// @brief Distributed GPU FFT Poisson solver with MPI pencil transpose
///
/// Single-rank: delegates to FFTPoissonSolver for optimal performance.
/// Multi-rank: cuFFT 1D R2C in x (local) → MPI transpose → cuFFT 1D C2C
/// in z (local) → cuSPARSE batched tridiag in y → inverse path.
/// All compute on GPU; MPI uses host-staged buffers for portability.

#include "poisson_solver_fft_mpi.hpp"
#ifdef USE_FFT_POISSON
#include "poisson_solver_fft.hpp"
#endif
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>

#ifdef USE_MPI
#include <mpi.h>
namespace {
void mpi_check(int rc, const char* call) {
    if (rc != MPI_SUCCESS) {
        char msg[MPI_MAX_ERROR_STRING]; int len;
        MPI_Error_string(rc, msg, &len);
        throw std::runtime_error(std::string("[MPI] ") + call + " failed: " + msg);
    }
}
} // namespace
#endif

#ifdef USE_FFT_POISSON
#include <omp.h>
#endif

namespace nncfd {

// ============================================================================
// Construction / destruction
// ============================================================================

FFTMPIPoissonSolver::FFTMPIPoissonSolver(const Mesh& mesh, const Decomposition& decomp)
    : mesh_(&mesh), decomp_(&decomp)
{
    Nx_ = mesh.Nx;
    Ny_ = mesh.Ny;
    Ng_ = mesh.Nghost;
    Nz_global_ = decomp.nz_global();
    Nz_local_ = decomp.nz_local();
    distributed_ = decomp.is_parallel();

    if (!distributed_) {
#ifdef USE_FFT_POISSON
        serial_solver_ = std::make_unique<FFTPoissonSolver>(mesh);
#endif
    }
#ifdef USE_FFT_POISSON
    else {
        Nx_c_ = Nx_ / 2 + 1;
    }
#endif
}

FFTMPIPoissonSolver::~FFTMPIPoissonSolver() {
#ifdef USE_FFT_POISSON
    if (gpu_initialized_) {
        free_gpu();
    }
#endif
}

// ============================================================================
// Configuration
// ============================================================================

void FFTMPIPoissonSolver::set_bc(PoissonBC x_lo, PoissonBC x_hi,
                                  PoissonBC y_lo, PoissonBC y_hi,
                                  PoissonBC z_lo, PoissonBC z_hi) {
    bc_x_lo_ = x_lo; bc_x_hi_ = x_hi;
    bc_y_lo_ = y_lo; bc_y_hi_ = y_hi;
    bc_z_lo_ = z_lo; bc_z_hi_ = z_hi;

#ifdef USE_FFT_POISSON
    if (serial_solver_) {
        serial_solver_->set_bc(x_lo, x_hi, y_lo, y_hi, z_lo, z_hi);
    }
    if (distributed_) gpu_initialized_ = false;  // force re-init
#endif
}

void FFTMPIPoissonSolver::set_space_order(int order) {
    space_order_ = order;
#ifdef USE_FFT_POISSON
    if (serial_solver_) {
        serial_solver_->set_space_order(order);
    }
    if (distributed_) gpu_initialized_ = false;  // force re-init
#endif
}

bool FFTMPIPoissonSolver::is_suitable(PoissonBC x_lo, PoissonBC x_hi,
                                       PoissonBC y_lo, PoissonBC y_hi,
                                       PoissonBC z_lo, PoissonBC z_hi,
                                       bool uniform_x, bool uniform_z) {
    return (x_lo == PoissonBC::Periodic && x_hi == PoissonBC::Periodic &&
            z_lo == PoissonBC::Periodic && z_hi == PoissonBC::Periodic &&
            y_lo != PoissonBC::Periodic && y_hi != PoissonBC::Periodic &&
            uniform_x && uniform_z);
}

bool FFTMPIPoissonSolver::using_gpu() const {
#ifdef USE_FFT_POISSON
    if (serial_solver_) return serial_solver_->using_gpu();
    return distributed_;  // GPU path for distributed
#else
    return false;
#endif
}

// ============================================================================
// Solve entry points
// ============================================================================

int FFTMPIPoissonSolver::solve(const ScalarField& /*rhs*/, ScalarField& /*p*/,
                                const PoissonConfig& /*cfg*/) {
    if (distributed_) {
        throw std::runtime_error(
            "FFTMPIPoissonSolver::solve() not available for distributed case. "
            "Use solve_device() (GPU required for distributed FFT_MPI).");
    }
#ifdef USE_FFT_POISSON
    // Single-rank: serial solver only has solve_device(), no CPU path
    throw std::runtime_error(
        "FFTMPIPoissonSolver::solve() single-rank CPU path not available. "
        "Use solve_device() on GPU.");
#else
    throw std::runtime_error("FFTMPIPoissonSolver requires USE_FFT_POISSON");
#endif
}

int FFTMPIPoissonSolver::solve_device(double* rhs_ptr, double* p_ptr,
                                       const PoissonConfig& cfg) {
#ifdef USE_FFT_POISSON
    if (!distributed_ && serial_solver_) {
        int result = serial_solver_->solve_device(rhs_ptr, p_ptr, cfg);
        residual_ = serial_solver_->residual();
        return result;
    }

    if (distributed_) {
        return solve_device_distributed(rhs_ptr, p_ptr);
    }
#else
    (void)rhs_ptr; (void)p_ptr; (void)cfg;
#endif
    throw std::runtime_error("FFTMPIPoissonSolver: no solver available");
}

// ============================================================================
// GPU implementation (distributed multi-rank)
// ============================================================================

#ifdef USE_FFT_POISSON

void FFTMPIPoissonSolver::free_gpu() {
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
    if (cusparse_handle_) { cusparseDestroy(cusparse_handle_); cusparse_handle_ = nullptr; }

    auto free_managed = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
    free_managed(rhs_packed_);
    free_managed(p_packed_);
    free_managed(hat_x_);
    free_managed(pencil_);
    free_managed(mode_buf_);
    free_managed(lambda_x_);
    free_managed(lambda_z_);
    free_managed(tri_lower_);
    free_managed(tri_upper_);
    free_managed(tri_diag_base_);
    free_managed(dyv_dev_);
    free_managed(tri_dl_);
    free_managed(tri_d_);
    free_managed(tri_du_);
    free_managed(cusparse_buffer_);

    // Destroy cuFFT plans (safe to call even if not created)
    cufftDestroy(x_r2c_); x_r2c_ = 0;
    cufftDestroy(x_c2r_); x_c2r_ = 0;
    cufftDestroy(z_fwd_); z_fwd_ = 0;
    cufftDestroy(z_inv_); z_inv_ = 0;

    gpu_initialized_ = false;
}

void FFTMPIPoissonSolver::initialize_gpu() {
    if (gpu_initialized_) free_gpu();

    const int Nx = Nx_, Ny = Ny_, Ng = Ng_;
    const int Nz_local = Nz_local_, Nz_global = Nz_global_;
    const int Nx_c = Nx_c_;
    const int nprocs = decomp_->nprocs();
    const int rank = decomp_->rank();

    // ---- MPI transpose parameters ----
    int kx_base = Nx_c / nprocs;
    int kx_rem = Nx_c % nprocs;
    my_kx_start_ = rank * kx_base + std::min(rank, kx_rem);
    my_kx_count_ = kx_base + (rank < kx_rem ? 1 : 0);
    n_local_modes_ = my_kx_count_ * Nz_global;

    // Forward transpose: z-slabs → kx-pencils
    fwd_send_counts_.resize(nprocs);
    fwd_send_displs_.resize(nprocs);
    fwd_recv_counts_.resize(nprocs);
    fwd_recv_displs_.resize(nprocs);
    int s_off = 0, r_off = 0;
    for (int r = 0; r < nprocs; ++r) {
        int kx_count_r = kx_base + (r < kx_rem ? 1 : 0);
        int nz_from_r = decomp_->nz_for_rank(r);
        // Send: rank r's kx range × Ny × Nz_local × 2 (real+imag)
        fwd_send_counts_[r] = kx_count_r * Ny * Nz_local * 2;
        fwd_send_displs_[r] = s_off;
        s_off += fwd_send_counts_[r];
        // Recv: my kx range × Ny × nz_from_r × 2
        fwd_recv_counts_[r] = my_kx_count_ * Ny * nz_from_r * 2;
        fwd_recv_displs_[r] = r_off;
        r_off += fwd_recv_counts_[r];
    }

    // Reverse transpose: kx-pencils → z-slabs (swap send/recv roles)
    rev_send_counts_.resize(nprocs);
    rev_send_displs_.resize(nprocs);
    rev_recv_counts_.resize(nprocs);
    rev_recv_displs_.resize(nprocs);
    s_off = r_off = 0;
    for (int r = 0; r < nprocs; ++r) {
        int kx_count_r = kx_base + (r < kx_rem ? 1 : 0);
        int nz_for_r = decomp_->nz_for_rank(r);
        rev_send_counts_[r] = my_kx_count_ * Ny * nz_for_r * 2;
        rev_send_displs_[r] = s_off;
        s_off += rev_send_counts_[r];
        rev_recv_counts_[r] = kx_count_r * Ny * Nz_local * 2;
        rev_recv_displs_[r] = r_off;
        r_off += rev_recv_counts_[r];
    }

    int fwd_send_total = fwd_send_displs_.back() + fwd_send_counts_.back();
    int fwd_recv_total = fwd_recv_displs_.back() + fwd_recv_counts_.back();
    int rev_send_total = rev_send_displs_.back() + rev_send_counts_.back();
    int rev_recv_total = rev_recv_displs_.back() + rev_recv_counts_.back();
    int max_host = std::max({fwd_send_total, fwd_recv_total,
                             rev_send_total, rev_recv_total});
    send_host_.resize(max_host, 0.0);
    recv_host_.resize(max_host, 0.0);

    // ---- CUDA stream ----
    cudaStreamCreate(&stream_);

    // ---- GPU buffer allocation (cudaMallocManaged) ----
    size_t real_slab = static_cast<size_t>(Nx) * Ny * Nz_local;
    size_t cx_slab = static_cast<size_t>(Nx_c) * Ny * Nz_local;
    size_t cx_pencil = static_cast<size_t>(my_kx_count_) * Ny * Nz_global;
    size_t cx_modes = static_cast<size_t>(n_local_modes_) * Ny;

    cudaMallocManaged(&rhs_packed_, sizeof(double) * real_slab);
    cudaMallocManaged(&p_packed_, sizeof(double) * real_slab);
    cudaMallocManaged(&hat_x_, sizeof(cufftDoubleComplex) * cx_slab);
    cudaMallocManaged(&pencil_, sizeof(cufftDoubleComplex) * cx_pencil);
    cudaMallocManaged(&mode_buf_, sizeof(cufftDoubleComplex) * cx_modes);

    // ---- Eigenvalues ----
    cudaMallocManaged(&lambda_x_, sizeof(double) * Nx_c);
    cudaMallocManaged(&lambda_z_, sizeof(double) * Nz_global);

    const double dx = mesh_->dx;
    const double dz = mesh_->dz;
    const double pi = M_PI;
    if (space_order_ == 4) {
        for (int kx = 0; kx < Nx_c; ++kx) {
            double theta = 2.0 * pi * kx / Nx;
            double c1 = std::cos(theta), c2 = std::cos(2.0*theta), c3 = std::cos(3.0*theta);
            lambda_x_[kx] = (1460.0 - 1566.0*c1 + 108.0*c2 - 2.0*c3) / (576.0 * dx * dx);
        }
        for (int kz = 0; kz < Nz_global; ++kz) {
            double theta = 2.0 * pi * kz / Nz_global;
            double c1 = std::cos(theta), c2 = std::cos(2.0*theta), c3 = std::cos(3.0*theta);
            lambda_z_[kz] = (1460.0 - 1566.0*c1 + 108.0*c2 - 2.0*c3) / (576.0 * dz * dz);
        }
    } else {
        for (int kx = 0; kx < Nx_c; ++kx)
            lambda_x_[kx] = (2.0 - 2.0 * std::cos(2.0 * pi * kx / Nx)) / (dx * dx);
        for (int kz = 0; kz < Nz_global; ++kz)
            lambda_z_[kz] = (2.0 - 2.0 * std::cos(2.0 * pi * kz / Nz_global)) / (dz * dz);
    }

    // ---- Tridiagonal coefficients (y-direction) ----
    cudaMallocManaged(&tri_lower_, sizeof(double) * Ny);
    cudaMallocManaged(&tri_upper_, sizeof(double) * Ny);
    cudaMallocManaged(&tri_diag_base_, sizeof(double) * Ny);

    const bool stretched = mesh_->is_y_stretched();
    for (int j = 0; j < Ny; ++j) {
        const int jg = j + Ng;
        double aS, aN;
        if (stretched) {
            aS = mesh_->yLap_aS[jg];
            aN = mesh_->yLap_aN[jg];
        } else {
            double invDy2 = 1.0 / (mesh_->dy * mesh_->dy);
            aS = invDy2;
            aN = invDy2;
        }
        if (j == 0 && bc_y_lo_ == PoissonBC::Neumann) aS = 0.0;
        if (j == Ny - 1 && bc_y_hi_ == PoissonBC::Neumann) aN = 0.0;
        tri_lower_[j] = aS;
        tri_upper_[j] = aN;
        tri_diag_base_[j] = -(aS + aN);
    }

    // ---- Volume weights for mean subtraction ----
    cudaMallocManaged(&dyv_dev_, sizeof(double) * Ny);
    double Ly = 0.0;
    for (int j = 0; j < Ny; ++j) {
        dyv_dev_[j] = (!mesh_->dyv.empty()) ? mesh_->dyv[j + Ng] : mesh_->dy;
        Ly += dyv_dev_[j];
    }
    // Total volume across ALL ranks (global Nz)
    total_volume_ = static_cast<double>(Nx) * Nz_global * Ly;

    // ---- cuFFT plans ----
    // x-direction: 1D R2C, batch = Ny * Nz_local
    // Input:  rhs_packed_[(k*Ny + j)*Nx + i], i fastest
    // Output: hat_x_[(k*Ny + j)*Nx_c + kx], kx fastest
    {
        int n_x[1] = {Nx};
        int x_batch = Ny * Nz_local;

        cufftResult rc = cufftPlanMany(&x_r2c_, 1, n_x,
            n_x, 1, Nx,       // inembed, istride, idist
            n_x, 1, Nx_c,     // onembed, ostride, odist (Nx used as dummy inembed)
            CUFFT_D2Z, x_batch);
        if (rc != CUFFT_SUCCESS) throw std::runtime_error("cuFFT x_r2c plan failed");

        // Inverse x: 1D C2R
        int on_x[1] = {Nx_c};
        rc = cufftPlanMany(&x_c2r_, 1, n_x,
            on_x, 1, Nx_c,    // inembed, istride, idist (complex input)
            n_x, 1, Nx,       // onembed, ostride, odist (real output)
            CUFFT_Z2D, x_batch);
        if (rc != CUFFT_SUCCESS) throw std::runtime_error("cuFFT x_c2r plan failed");

        cufftSetStream(x_r2c_, stream_);
        cufftSetStream(x_c2r_, stream_);
    }

    // z-direction: 1D C2C, batch = my_kx_count * Ny
    // Input:  pencil_[(kx_l*Ny + j)*Nz_global + iz], iz fastest
    // Output: same layout with kz instead of iz
    {
        int n_z[1] = {Nz_global};
        int z_batch = my_kx_count_ * Ny;

        cufftResult rc = cufftPlanMany(&z_fwd_, 1, n_z,
            n_z, 1, Nz_global,
            n_z, 1, Nz_global,
            CUFFT_Z2Z, z_batch);
        if (rc != CUFFT_SUCCESS) throw std::runtime_error("cuFFT z_fwd plan failed");

        rc = cufftPlanMany(&z_inv_, 1, n_z,
            n_z, 1, Nz_global,
            n_z, 1, Nz_global,
            CUFFT_Z2Z, z_batch);
        if (rc != CUFFT_SUCCESS) throw std::runtime_error("cuFFT z_inv plan failed");

        cufftSetStream(z_fwd_, stream_);
        cufftSetStream(z_inv_, stream_);
    }

    // ---- cuSPARSE batched tridiagonal ----
    cusparseCreate(&cusparse_handle_);
    cusparseSetStream(cusparse_handle_, stream_);

    cudaMallocManaged(&tri_dl_, sizeof(cufftDoubleComplex) * cx_modes);
    cudaMallocManaged(&tri_d_,  sizeof(cufftDoubleComplex) * cx_modes);
    cudaMallocManaged(&tri_du_, sizeof(cufftDoubleComplex) * cx_modes);

    // Precompute tridiagonal matrices with eigenvalue shifts
    for (int kx_l = 0; kx_l < my_kx_count_; ++kx_l) {
        int kx = my_kx_start_ + kx_l;
        for (int kz = 0; kz < Nz_global; ++kz) {
            size_t mode = static_cast<size_t>(kx_l) * Nz_global + kz;
            double shift = lambda_x_[kx] + lambda_z_[kz];
            bool zero_mode = (kx == 0 && kz == 0);

            for (int j = 0; j < Ny; ++j) {
                size_t idx = mode * Ny + j;
                if (zero_mode && j == 0) {
                    tri_dl_[idx] = {0.0, 0.0};
                    tri_d_[idx]  = {1.0, 0.0};
                    tri_du_[idx] = {0.0, 0.0};
                } else {
                    tri_dl_[idx] = {(j > 0) ? tri_lower_[j] : 0.0, 0.0};
                    tri_d_[idx]  = {tri_diag_base_[j] - shift, 0.0};
                    tri_du_[idx] = {(j < Ny - 1) ? tri_upper_[j] : 0.0, 0.0};
                }
            }
        }
    }

    // Query cuSPARSE buffer size
    cusparseZgtsv2StridedBatch_bufferSizeExt(
        cusparse_handle_, Ny, tri_dl_, tri_d_, tri_du_,
        mode_buf_, n_local_modes_, Ny, &cusparse_buffer_size_);
    cudaMalloc(&cusparse_buffer_, cusparse_buffer_size_);

    // Sync managed memory to device
    cudaDeviceSynchronize();

    gpu_initialized_ = true;
    std::cout << "[FFT_MPI] GPU initialized: Nx=" << Nx << " Ny=" << Ny
              << " Nz_local=" << Nz_local << " Nz_global=" << Nz_global
              << " Nx_c=" << Nx_c << " my_kx=[" << my_kx_start_
              << "," << my_kx_start_ + my_kx_count_ << ")"
              << " n_modes=" << n_local_modes_
              << " cuSPARSE_buf=" << cusparse_buffer_size_ << "B\n";
}

int FFTMPIPoissonSolver::solve_device_distributed(double* rhs_ptr, double* p_ptr) {
    if (!gpu_initialized_) initialize_gpu();

    const int Nx = Nx_, Ny = Ny_, Ng = Ng_;
    const int Nz_local = Nz_local_, Nz_global = Nz_global_;
    const int Nx_c = Nx_c_;
    const int Nx_full = Nx + 2 * Ng;
    const int Ny_full = Ny + 2 * Ng;

    // Get device pointers for OMP-mapped rhs/p
    int device = omp_get_default_device();
    double* rhs_dev = static_cast<double*>(omp_get_mapped_ptr(rhs_ptr, device));
    double* p_dev = static_cast<double*>(omp_get_mapped_ptr(p_ptr, device));

    // ================================================================
    // Step 1: Pack RHS from ghost layout + volume-weighted mean
    // ================================================================
    double* rhs_pk = rhs_packed_;
    double* dyv = dyv_dev_;
    double local_wsum = 0.0;

    #pragma omp target teams distribute parallel for collapse(3) \
        reduction(+:local_wsum) is_device_ptr(rhs_dev, rhs_pk, dyv)
    for (int k = 0; k < Nz_local; ++k) {
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                int ghost_idx = (k + Ng) * Ny_full * Nx_full
                              + (j + Ng) * Nx_full + (i + Ng);
                int pack_idx = (k * Ny + j) * Nx + i;
                double val = rhs_dev[ghost_idx];
                rhs_pk[pack_idx] = val;
                local_wsum += val * dyv[j];
            }
        }
    }

    // Global mean subtraction (MPI allreduce)
    double global_wsum = decomp_->allreduce_sum(local_wsum);
    double mean = global_wsum / total_volume_;

    int pack_total = Nx * Ny * Nz_local;
    #pragma omp target teams distribute parallel for is_device_ptr(rhs_pk)
    for (int idx = 0; idx < pack_total; ++idx) {
        rhs_pk[idx] -= mean;
    }

    // ================================================================
    // Step 2: cuFFT 1D R2C in x
    // ================================================================
    cudaDeviceSynchronize();  // Ensure OMP target kernels complete
    cufftResult rc = cufftExecD2Z(x_r2c_, rhs_packed_, hat_x_);
    if (rc != CUFFT_SUCCESS) throw std::runtime_error("FFT_MPI: x R2C failed");
    cudaStreamSynchronize(stream_);

    // ================================================================
    // Step 3: Forward MPI transpose (z-slabs → kx-pencils)
    // ================================================================
#ifdef USE_MPI
    {
        const int nprocs = decomp_->nprocs();
        int kx_base = Nx_c / nprocs;
        int kx_rem = Nx_c % nprocs;

        // Pack: for each dest rank, extract their kx range
        // hat_x_ layout: [(k*Ny + j)*Nx_c + kx] complex
        double* cx_host = reinterpret_cast<double*>(hat_x_);
        // Managed memory: sync to host for packing
        cudaDeviceSynchronize();

        int offset = 0;
        for (int r = 0; r < nprocs; ++r) {
            int kx_start_r = r * kx_base + std::min(r, kx_rem);
            int kx_count_r = kx_base + (r < kx_rem ? 1 : 0);
            for (int k = 0; k < Nz_local; ++k) {
                for (int j = 0; j < Ny; ++j) {
                    // Complex values for this (k,j) row, kx range for rank r
                    const double* src = cx_host + 2 * ((k * Ny + j) * Nx_c + kx_start_r);
                    std::memcpy(&send_host_[offset], src,
                                kx_count_r * 2 * sizeof(double));
                    offset += kx_count_r * 2;
                }
            }
        }

        mpi_check(MPI_Alltoallv(
            send_host_.data(), fwd_send_counts_.data(), fwd_send_displs_.data(), MPI_DOUBLE,
            recv_host_.data(), fwd_recv_counts_.data(), fwd_recv_displs_.data(), MPI_DOUBLE,
            decomp_->comm()), "MPI_Alltoallv(fwd)");

        // Unpack: recv_host_ → pencil_ (managed memory, write on host)
        // pencil_ layout: [(kx_l*Ny + j)*Nz_global + iz] complex
        double* pencil_host = reinterpret_cast<double*>(pencil_);
        offset = 0;
        for (int r = 0; r < nprocs; ++r) {
            int nz_from_r = decomp_->nz_for_rank(r);
            int kz_start = decomp_->k_global_start_for_rank(r);
            for (int kl = 0; kl < nz_from_r; ++kl) {
                int iz = kz_start + kl;
                for (int j = 0; j < Ny; ++j) {
                    for (int kx_l = 0; kx_l < my_kx_count_; ++kx_l) {
                        int dst = 2 * ((kx_l * Ny + j) * Nz_global + iz);
                        pencil_host[dst]     = recv_host_[offset++];
                        pencil_host[dst + 1] = recv_host_[offset++];
                    }
                }
            }
        }
    }
#endif

    // ================================================================
    // Step 4: cuFFT 1D C2C forward in z
    // ================================================================
    cudaDeviceSynchronize();  // Sync managed memory writes
    rc = cufftExecZ2Z(z_fwd_, pencil_, pencil_, CUFFT_FORWARD);
    if (rc != CUFFT_SUCCESS) throw std::runtime_error("FFT_MPI: z forward failed");
    cudaStreamSynchronize(stream_);

    // ================================================================
    // Step 5: Transpose j↔kz → cuSPARSE [mode*Ny + j] layout
    // ================================================================
    // pencil_[(kx_l*Ny + j)*Nz_global + kz] → mode_buf_[(kx_l*Nz_global + kz)*Ny + j]
    {
        cufftDoubleComplex* src = pencil_;
        cufftDoubleComplex* dst = mode_buf_;
        int mkc = my_kx_count_;
        #pragma omp target teams distribute parallel for collapse(3) \
            is_device_ptr(src, dst)
        for (int kx_l = 0; kx_l < mkc; ++kx_l) {
            for (int j = 0; j < Ny; ++j) {
                for (int kz = 0; kz < Nz_global; ++kz) {
                    int in_idx = (kx_l * Ny + j) * Nz_global + kz;
                    int out_idx = (kx_l * Nz_global + kz) * Ny + j;
                    dst[out_idx] = src[in_idx];
                }
            }
        }
    }

    // ================================================================
    // Step 6: cuSPARSE batched tridiagonal solve in y
    // ================================================================
    cudaDeviceSynchronize();  // Ensure OMP transpose complete

    // Zero the (0,0) mode RHS (pinned value for singular mode)
    if (my_kx_start_ == 0) {
        cudaMemsetAsync(mode_buf_, 0, sizeof(cufftDoubleComplex), stream_);
    }
    cusparseStatus_t cs = cusparseZgtsv2StridedBatch(
        cusparse_handle_, Ny,
        tri_dl_, tri_d_, tri_du_, mode_buf_,
        n_local_modes_, Ny, cusparse_buffer_);
    if (cs != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("FFT_MPI: cuSPARSE tridiag failed, code=" +
                                 std::to_string(static_cast<int>(cs)));
    }
    cudaStreamSynchronize(stream_);

    // ================================================================
    // Step 7: Transpose back kz↔j → pencil layout
    // ================================================================
    // mode_buf_[(kx_l*Nz_global + kz)*Ny + j] → pencil_[(kx_l*Ny + j)*Nz_global + kz]
    {
        cufftDoubleComplex* src = mode_buf_;
        cufftDoubleComplex* dst = pencil_;
        int mkc = my_kx_count_;
        #pragma omp target teams distribute parallel for collapse(3) \
            is_device_ptr(src, dst)
        for (int kx_l = 0; kx_l < mkc; ++kx_l) {
            for (int kz = 0; kz < Nz_global; ++kz) {
                for (int j = 0; j < Ny; ++j) {
                    int in_idx = (kx_l * Nz_global + kz) * Ny + j;
                    int out_idx = (kx_l * Ny + j) * Nz_global + kz;
                    dst[out_idx] = src[in_idx];
                }
            }
        }
    }

    // ================================================================
    // Step 8: cuFFT 1D C2C inverse in z
    // ================================================================
    cudaDeviceSynchronize();
    rc = cufftExecZ2Z(z_inv_, pencil_, pencil_, CUFFT_INVERSE);
    if (rc != CUFFT_SUCCESS) throw std::runtime_error("FFT_MPI: z inverse failed");
    cudaStreamSynchronize(stream_);

    // ================================================================
    // Step 9: Reverse MPI transpose (kx-pencils → z-slabs)
    // ================================================================
#ifdef USE_MPI
    {
        const int nprocs = decomp_->nprocs();

        // Pack pencil_ for reverse transpose
        double* pencil_host = reinterpret_cast<double*>(pencil_);
        cudaDeviceSynchronize();

        int offset = 0;
        for (int r = 0; r < nprocs; ++r) {
            int kz_start = decomp_->k_global_start_for_rank(r);
            int nz_for_r = decomp_->nz_for_rank(r);
            for (int kl = 0; kl < nz_for_r; ++kl) {
                int iz = kz_start + kl;
                for (int j = 0; j < Ny; ++j) {
                    for (int kx_l = 0; kx_l < my_kx_count_; ++kx_l) {
                        int src = 2 * ((kx_l * Ny + j) * Nz_global + iz);
                        send_host_[offset++] = pencil_host[src];
                        send_host_[offset++] = pencil_host[src + 1];
                    }
                }
            }
        }

        mpi_check(MPI_Alltoallv(
            send_host_.data(), rev_send_counts_.data(), rev_send_displs_.data(), MPI_DOUBLE,
            recv_host_.data(), rev_recv_counts_.data(), rev_recv_displs_.data(), MPI_DOUBLE,
            decomp_->comm()), "MPI_Alltoallv(rev)");

        // Unpack: recv_host_ → hat_x_ (managed)
        double* cx_host = reinterpret_cast<double*>(hat_x_);
        int kx_base = Nx_c / nprocs;
        int kx_rem = Nx_c % nprocs;
        offset = 0;
        for (int r = 0; r < nprocs; ++r) {
            int kx_start_r = r * kx_base + std::min(r, kx_rem);
            int kx_count_r = kx_base + (r < kx_rem ? 1 : 0);
            for (int k = 0; k < Nz_local; ++k) {
                for (int j = 0; j < Ny; ++j) {
                    double* dst = cx_host + 2 * ((k * Ny + j) * Nx_c + kx_start_r);
                    std::memcpy(dst, &recv_host_[offset],
                                kx_count_r * 2 * sizeof(double));
                    offset += kx_count_r * 2;
                }
            }
        }
    }
#endif

    // ================================================================
    // Step 10: cuFFT 1D C2R inverse in x
    // ================================================================
    cudaDeviceSynchronize();
    rc = cufftExecZ2D(x_c2r_, hat_x_, p_packed_);
    if (rc != CUFFT_SUCCESS) throw std::runtime_error("FFT_MPI: x C2R failed");
    cudaStreamSynchronize(stream_);

    // ================================================================
    // Step 11: Unpack to ghost layout + normalize + BCs
    // ================================================================
    // cuFFT normalization: 1 / (Nx * Nz_global)
    double norm = 1.0 / (static_cast<double>(Nx) * Nz_global);
    double* p_pk = p_packed_;

    #pragma omp target teams distribute parallel for collapse(3) \
        is_device_ptr(p_dev, p_pk)
    for (int k = 0; k < Nz_local; ++k) {
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                int ghost_idx = (k + Ng) * Ny_full * Nx_full
                              + (j + Ng) * Nx_full + (i + Ng);
                int pack_idx = (k * Ny + j) * Nx + i;
                p_dev[ghost_idx] = p_pk[pack_idx] * norm;
            }
        }
    }

    // Neumann BCs in y (ghost cells)
    // Copy member BCs to locals for OMP target (avoid this-pointer transfer)
    const bool neumann_lo = (bc_y_lo_ == PoissonBC::Neumann);
    const bool neumann_hi = (bc_y_hi_ == PoissonBC::Neumann);
    if (neumann_lo || neumann_hi) {
        #pragma omp target teams distribute parallel for collapse(2) \
            is_device_ptr(p_dev)
        for (int k = 0; k < Nz_local; ++k) {
            for (int i = 0; i < Nx; ++i) {
                int base = (k + Ng) * Ny_full * Nx_full + (i + Ng);
                if (neumann_lo) {
                    p_dev[(Ng - 1) * Nx_full + base] = p_dev[Ng * Nx_full + base];
                }
                if (neumann_hi) {
                    p_dev[(Ny + Ng) * Nx_full + base] = p_dev[(Ny + Ng - 1) * Nx_full + base];
                }
            }
        }
    }

    residual_ = 0.0;  // Direct solver
    return 1;
}

#endif // USE_FFT_POISSON

} // namespace nncfd
