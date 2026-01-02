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

    // Create cuFFT plans for batched 2D FFT
    // We do 2D FFT in x-z for each y-plane
    // Input layout: rhs_packed[k + Nz * (i + Nx * j)] for (i,j,k) = (x,y,z)
    // But we need contiguous x-z planes for each y, so we'll reorder

    // For the FFT, we want data organized as [y][x][z] so each y-plane is contiguous
    // Current layout is [k][j][i] = [z][y][x], need to transpose

    // Actually, let's use a different approach: create many-rank plan
    // that handles non-contiguous data via advanced interface

    // For simplicity, use batch of 1D FFTs or reorganize data
    // Let's use the standard approach: transpose to [y][x][z], FFT, transpose back

    // Simpler approach: since cuFFT supports strided batched transforms,
    // use cufftPlanMany to handle the striding

    // Data layout after packing: p_packed[k * Nx * Ny + j * Nx + i] for (i,j,k)
    // We want 2D FFT over (i,k) for each j

    // Using cufftPlanMany for 2D R2C with batch over y
    int n[2] = {Nx, Nz};  // Dimensions to transform
    int inembed[2] = {Nx, Nz};
    int onembed[2] = {Nx, Nz_complex};
    int istride = 1;  // Stride between elements in innermost dimension
    int ostride = 1;
    int idist = Nx * Nz;  // Distance between batches (each y-plane)
    int odist = Nx * Nz_complex;
    int batch = Ny;

    // Note: This assumes data is organized as [j][i][k] = [y][x][z]
    // We'll need to reorganize from [k][j][i] during pack

    cufftResult result = cufftPlanMany(&fft_plan_r2c_, 2, n,
                                        inembed, istride, idist,
                                        onembed, ostride, odist,
                                        CUFFT_D2Z, batch);
    if (result != CUFFT_SUCCESS) {
        throw std::runtime_error("Failed to create cuFFT R2C plan");
    }

    result = cufftPlanMany(&fft_plan_c2r_, 2, n,
                            onembed, ostride, odist,
                            inembed, istride, idist,
                            CUFFT_Z2D, batch);
    if (result != CUFFT_SUCCESS) {
        throw std::runtime_error("Failed to create cuFFT C2R plan");
    }

    plans_created_ = true;

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
}

void FFTPoissonSolver::solve_tridiagonal_cusparse() {
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Nz_complex = Nz / 2 + 1;
    const int n_modes = Nx * Nz_complex;

    cufftDoubleComplex* rhs_h = rhs_hat_;
    double* lam_x = lambda_x_;
    double* lam_z = lambda_z_;
    double* aS = tri_lower_;
    double* aN = tri_upper_;
    double* diag_base = tri_diag_base_;
    cufftDoubleComplex* dl = tri_dl_;
    cufftDoubleComplex* d = tri_d_;
    cufftDoubleComplex* du = tri_du_;

    // Step 1: Build complex tridiagonal matrices for each mode (kx, kz)
    // cuSPARSE expects: dl[batch*Ny + j], d[batch*Ny + j], du[batch*Ny + j]
    // where batch = kx * Nz_complex + kz
    #pragma omp target teams distribute parallel for collapse(3) \
        is_device_ptr(dl, d, du, lam_x, lam_z, aS, aN, diag_base)
    for (int kx = 0; kx < Nx; ++kx) {
        for (int kz = 0; kz < Nz_complex; ++kz) {
            for (int j = 0; j < Ny; ++j) {
                const size_t batch = static_cast<size_t>(kx) * Nz_complex + kz;
                const size_t idx = batch * Ny + j;

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

    // Step 2: Reorganize RHS for cuSPARSE layout
    // Current layout: rhs_hat[j * Nx * Nz_complex + kx * Nz_complex + kz]
    // cuSPARSE needs: x[batch * Ny + j] where batch = kx * Nz_complex + kz
    // We'll solve in-place on p_hat which we'll use as work array
    cufftDoubleComplex* x = p_hat_;

    #pragma omp target teams distribute parallel for collapse(3) \
        is_device_ptr(rhs_h, x)
    for (int kx = 0; kx < Nx; ++kx) {
        for (int kz = 0; kz < Nz_complex; ++kz) {
            for (int j = 0; j < Ny; ++j) {
                const size_t batch = static_cast<size_t>(kx) * Nz_complex + kz;
                const size_t src_idx = static_cast<size_t>(j) * Nx * Nz_complex + batch;
                const size_t dst_idx = batch * Ny + j;

                // For zero mode (0,0) at j=0, set RHS to 0 (pinned value)
                bool is_zero_mode = (kx == 0 && kz == 0);
                if (is_zero_mode && j == 0) {
                    x[dst_idx].x = 0.0;
                    x[dst_idx].y = 0.0;
                } else {
                    x[dst_idx] = rhs_h[src_idx];
                }
            }
        }
    }
    cudaDeviceSynchronize();

    // Step 3: Call cuSPARSE batched tridiagonal solver
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
        throw std::runtime_error("cuSPARSE gtsv2StridedBatch failed");
    }
    cudaDeviceSynchronize();

    // Step 4: Reorganize solution back to FFT layout
    // cuSPARSE layout: x[batch * Ny + j]
    // FFT layout: p_hat[j * Nx * Nz_complex + batch]
    // Since we solved in p_hat (which is x), we need to do an in-place transpose
    // Use rhs_hat as temp storage
    cufftDoubleComplex* temp = rhs_hat_;

    #pragma omp target teams distribute parallel for collapse(3) \
        is_device_ptr(x, temp)
    for (int kx = 0; kx < Nx; ++kx) {
        for (int kz = 0; kz < Nz_complex; ++kz) {
            for (int j = 0; j < Ny; ++j) {
                const size_t batch = static_cast<size_t>(kx) * Nz_complex + kz;
                const size_t src_idx = batch * Ny + j;
                const size_t dst_idx = static_cast<size_t>(j) * Nx * Nz_complex + batch;
                temp[dst_idx] = x[src_idx];
            }
        }
    }

    // Copy back to p_hat
    #pragma omp target teams distribute parallel for is_device_ptr(x, temp)
    for (size_t i = 0; i < static_cast<size_t>(Nx) * Nz_complex * Ny; ++i) {
        x[i] = temp[i];
    }
    cudaDeviceSynchronize();
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

    // Reorganize from [k][j][i] with ghosts to [j][i][k] without ghosts
    #pragma omp target teams distribute parallel for collapse(3) \
        map(present: rhs_ptr[0:total_size]) is_device_ptr(packed)
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            for (int k = 0; k < Nz; ++k) {
                // Source index with ghosts: [k+Ng][j+Ng][i+Ng]
                const size_t src_idx = static_cast<size_t>(k + Ng) * Nx_full * Ny_full +
                                       (j + Ng) * Nx_full + (i + Ng);
                // Destination index: [j][i][k] for FFT-friendly layout
                const size_t dst_idx = static_cast<size_t>(j) * Nx * Nz + i * Nz + k;
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

    // Reorganize from [j][i][k] back to [k][j][i] with ghosts
    #pragma omp target teams distribute parallel for collapse(3) \
        map(present: p_ptr[0:total_size]) is_device_ptr(packed)
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            for (int k = 0; k < Nz; ++k) {
                const size_t src_idx = static_cast<size_t>(j) * Nx * Nz + i * Nz + k;
                const size_t dst_idx = static_cast<size_t>(k + Ng) * Nx_full * Ny_full +
                                       (j + Ng) * Nx_full + (i + Ng);
                p_ptr[dst_idx] = packed[src_idx] * norm;
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
    #pragma omp target teams distribute parallel for collapse(2) map(present: p_ptr[0:total_size])
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
    #pragma omp target teams distribute parallel for collapse(2) map(present: p_ptr[0:total_size])
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
    #pragma omp target teams distribute parallel for collapse(2) map(present: p_ptr[0:total_size])
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

int FFTPoissonSolver::solve_device(double* rhs_ptr, double* p_ptr, const PoissonConfig& cfg) {
    if (!initialized_) {
        throw std::runtime_error("FFTPoissonSolver not initialized");
    }

    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Nz_complex = Nz / 2 + 1;

    // Step 1: Pack RHS from ghost layout to FFT-friendly layout [j][i][k]
    pack_rhs(rhs_ptr);
    cudaDeviceSynchronize();

    // Step 1b: Enforce mean(rhs) = 0 for nullspace handling
    // This ensures the system is consistent for pure Neumann/periodic BCs
    {
        const size_t n_total = static_cast<size_t>(Nx) * Ny * Nz;
        double* rhs_p = rhs_packed_;

        // Compute sum using reduction
        double sum = 0.0;
        #pragma omp target teams distribute parallel for reduction(+:sum) \
            is_device_ptr(rhs_p)
        for (size_t i = 0; i < n_total; ++i) {
            sum += rhs_p[i];
        }

        // Subtract mean from all values
        const double mean = sum / static_cast<double>(n_total);
        #pragma omp target teams distribute parallel for is_device_ptr(rhs_p)
        for (size_t i = 0; i < n_total; ++i) {
            rhs_p[i] -= mean;
        }
    }
    cudaDeviceSynchronize();

    // Step 2: Forward 2D FFT (R2C) for each y-plane
    cufftResult result = cufftExecD2Z(fft_plan_r2c_, rhs_packed_, rhs_hat_);
    if (result != CUFFT_SUCCESS) {
        throw std::runtime_error("cuFFT R2C failed");
    }
    cudaDeviceSynchronize();

    // Step 3: Solve tridiagonal systems in y for each Fourier mode (kx, kz)
    // The system for mode (kx, kz) is:
    //   (L_y - (λ_x + λ_z) I) p_hat = rhs_hat
    // where L_y is the tridiagonal second-derivative operator in y

    if (use_cusparse_) {
        // Use cuSPARSE batched tridiagonal solver (reference implementation)
        solve_tridiagonal_cusparse();
    } else {
        // Use custom Thomas algorithm
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

        // Thomas algorithm using global workspace arrays
        // Forward sweep: for each j from 0 to Ny-1, compute c'[j] and d'[j]
        for (int j = 0; j < Ny; ++j) {
            const int j_local = j;
            #pragma omp target teams distribute parallel for collapse(2) \
                is_device_ptr(rhs_h, lam_x, lam_z, aS, aN, diag_base, work_c, work_d_r, work_d_i)
            for (int kx = 0; kx < Nx; ++kx) {
                for (int kz = 0; kz < Nz_complex; ++kz) {
                    // Index for workspace: [mode][j] where mode = kx * Nz_complex + kz
                    const size_t mode_idx = static_cast<size_t>(kx) * Nz_complex + kz;
                    const size_t work_idx = mode_idx * Ny + j_local;
                    // Index for FFT arrays: [j][mode]
                    const size_t rhs_idx = static_cast<size_t>(j_local) * Nx * Nz_complex + mode_idx;

                    // Eigenvalue shift
                    double shift = lam_x[kx] + lam_z[kz];
                    bool is_zero_mode = (kx == 0 && kz == 0);

                    // Tridiagonal coefficients
                    double diag = diag_base[j_local] - shift;
                    double lower = (j_local > 0) ? aS[j_local] : 0.0;
                    double upper = (j_local < Ny - 1) ? aN[j_local] : 0.0;

                    // Get RHS values
                    double rhs_r = rhs_h[rhs_idx].x;
                    double rhs_i = rhs_h[rhs_idx].y;

                    // For zero mode, pin j=0 to zero
                    if (is_zero_mode && j_local == 0) {
                        diag = 1.0;
                        lower = 0.0;
                        upper = 0.0;
                        rhs_r = 0.0;
                        rhs_i = 0.0;
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

        // Backward substitution: for each j from Ny-1 down to 0
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

    // Step 4: Inverse 2D FFT (C2R) to get real solution
    result = cufftExecZ2D(fft_plan_c2r_, p_hat_, p_packed_);
    if (result != CUFFT_SUCCESS) {
        throw std::runtime_error("cuFFT C2R failed");
    }
    cudaDeviceSynchronize();

    // Step 5: Unpack solution to ghost-cell layout (includes FFT normalization)
    unpack_solution(p_ptr);

    // Step 6: Apply boundary conditions to ghost cells
    apply_bc_device(p_ptr);

    residual_ = 0.0;  // Direct solver, no residual
    return 1;  // "1 iteration" for a direct solver
}

#endif // USE_GPU_OFFLOAD

} // namespace nncfd

#endif // USE_GPU_OFFLOAD
