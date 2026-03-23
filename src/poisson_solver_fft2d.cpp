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
    // FFT solver's BC application only fills outermost ghost layer
    if (mesh.Nghost != 1) {
        throw std::runtime_error("FFT2DPoissonSolver requires Nghost == 1");
    }

    N_modes_ = Nx_ / 2 + 1;

#ifdef USE_GPU_OFFLOAD
    initialize_fft();
    initialize_eigenvalues();
    precompute_tridiagonal();
    initialize_cusparse();
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

    fully_periodic_ = (y_lo == PoissonBC::Periodic && y_hi == PoissonBC::Periodic);

    // Check if BCs actually changed
    bool bc_changed = (bc_y_lo_ != y_lo || bc_y_hi_ != y_hi);

    bc_y_lo_ = y_lo;
    bc_y_hi_ = y_hi;

#ifdef USE_GPU_OFFLOAD
    // Initialize y-FFT for fully periodic if not already done
    if (fully_periodic_ && lambda_y_ == nullptr) {
        initialize_eigenvalues();  // Creates y-eigenvalues and C2C plans
    }

    // Recompute tridiagonal matrices if BCs changed and already initialized
    if (!fully_periodic_ && bc_changed && tri_lower_ != nullptr) {
        // Update y-direction base coefficients with new BCs
        const double invDy2 = 1.0 / (dy_ * dy_);
        for (int j = 0; j < Ny_; ++j) {
            double aS = invDy2;
            double aN = invDy2;

            if (j == 0 && bc_y_lo_ == PoissonBC::Neumann) {
                aS = 0.0;
            }
            if (j == Ny_ - 1 && bc_y_hi_ == PoissonBC::Neumann) {
                aN = 0.0;
            }

            tri_lower_[j] = aS;
            tri_upper_[j] = aN;
            tri_diag_base_[j] = -(aS + aN);
        }
        cudaDeviceSynchronize();

        // Rebuild cuSPARSE tridiagonal matrices with new coefficients
        double* lam_x = lambda_x_;
        double* aS = tri_lower_;
        double* aN = tri_upper_;
        double* diag_base = tri_diag_base_;
        cufftDoubleComplex* dl = tri_dl_;
        cufftDoubleComplex* d = tri_d_;
        cufftDoubleComplex* du = tri_du_;
        int N_modes = N_modes_;
        int Ny = Ny_;

        #pragma omp target teams distribute parallel for collapse(2) \
            is_device_ptr(dl, d, du, lam_x, aS, aN, diag_base)
        for (int m = 0; m < N_modes; ++m) {
            for (int j = 0; j < Ny; ++j) {
                const size_t idx = (size_t)m * Ny + j;
                double shift = lam_x[m];
                bool is_zero_mode = (m == 0);

                if (is_zero_mode && j == 0) {
                    dl[idx].x = 0.0; dl[idx].y = 0.0;
                    d[idx].x = 1.0;  d[idx].y = 0.0;
                    du[idx].x = 0.0; du[idx].y = 0.0;
                } else {
                    dl[idx].x = (j > 0) ? aS[j] : 0.0;
                    dl[idx].y = 0.0;
                    d[idx].x = diag_base[j] - shift;
                    d[idx].y = 0.0;
                    du[idx].x = (j < Ny - 1) ? aN[j] : 0.0;
                    du[idx].y = 0.0;
                }
            }
        }
        cudaDeviceSynchronize();
    }
#endif
}

bool FFT2DPoissonSolver::is_suitable(PoissonBC x_lo, PoissonBC x_hi,
                                      PoissonBC y_lo, PoissonBC y_hi,
                                      bool uniform_x, bool is_2d) {
    if (!is_2d) return false;
    if (!uniform_x) return false;
    if (x_lo != PoissonBC::Periodic || x_hi != PoissonBC::Periodic) return false;
    // Accept both wall-bounded y (Neumann/Dirichlet → tridiag) and periodic y (→ FFT)
    return true;
}

#ifdef USE_GPU_OFFLOAD

void FFT2DPoissonSolver::initialize_fft() {
    cudaStreamCreate(&stream_);

    const int total = Nx_ * Ny_;
    const size_t hat_size = (size_t)N_modes_ * Ny_;

    // Allocate packed buffers using cudaMallocManaged for OMP interop
    cudaMallocManaged(&in_pack_, total * sizeof(double));
    cudaMallocManaged(&out_pack_, total * sizeof(double));
    cudaMallocManaged(&rhs_hat_, hat_size * sizeof(cufftDoubleComplex));

    // Create cuFFT plans
    // For 2D: batch of Ny 1D FFTs along x
    // Input:  in_pack[j * Nx + i]  (row-major, i fastest)
    // Output: rhs_hat[m * Ny + j]  (mode-major, j fastest for each mode)

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
    const double h2x = dx_ * dx_;

    cudaMallocManaged(&lambda_x_, N_modes_ * sizeof(double));
    for (int m = 0; m < N_modes_; ++m) {
        double theta = 2.0 * M_PI * m / Nx_;
        lambda_x_[m] = (2.0 - 2.0 * std::cos(theta)) / h2x;
    }

    // Y-eigenvalues and 2D FFT plans for fully periodic case
    if (fully_periodic_) {
        const double h2y = dy_ * dy_;
        cudaMallocManaged(&lambda_y_, Ny_ * sizeof(double));
        for (int n = 0; n < Ny_; ++n) {
            double theta = 2.0 * M_PI * n / Ny_;
            lambda_y_[n] = (2.0 - 2.0 * std::cos(theta)) / h2y;
        }

        // Create 2D R2C and C2R plans
        // Input: real [Ny][Nx], output: complex [Ny][Nx/2+1]
        N_modes_2d_ = Nx_ / 2 + 1;  // same as N_modes_
        int n2d[2] = { Ny_, Nx_ };

        cufftResult result = cufftPlanMany(&fft_2d_r2c_, 2, n2d,
                                            nullptr, 0, 0,   // auto layout
                                            nullptr, 0, 0,
                                            CUFFT_D2Z, 1);   // batch=1
        if (result != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create 2D cuFFT R2C plan");
        }
        cufftSetStream(fft_2d_r2c_, stream_);

        result = cufftPlanMany(&fft_2d_c2r_, 2, n2d,
                               nullptr, 0, 0,
                               nullptr, 0, 0,
                               CUFFT_Z2D, 1);
        if (result != CUFFT_SUCCESS) {
            throw std::runtime_error("Failed to create 2D cuFFT C2R plan");
        }
        cufftSetStream(fft_2d_c2r_, stream_);

        std::cout << "[FFT2DPoissonSolver] Fully periodic: 2D FFT + y-eigenvalues created\n";
    }

    cudaDeviceSynchronize();
}

void FFT2DPoissonSolver::precompute_tridiagonal() {
    // Compute y-direction tridiagonal coefficients
    // Use mesh yLap coefficients for D·G=L consistency on stretched grids.
    // For uniform grids, yLap arrays are empty so we fall back to 1/dy^2.
    const bool stretched = mesh_->is_y_stretched();
    const int Ng = mesh_->Nghost;

    cudaMallocManaged(&tri_lower_, Ny_ * sizeof(double));
    cudaMallocManaged(&tri_upper_, Ny_ * sizeof(double));
    cudaMallocManaged(&tri_diag_base_, Ny_ * sizeof(double));

    for (int j = 0; j < Ny_; ++j) {
        const int jg = j + Ng;

        double aS, aN;
        if (stretched) {
            aS = mesh_->yLap_aS[jg];
            aN = mesh_->yLap_aN[jg];
        } else {
            double invDy2 = 1.0 / (dy_ * dy_);
            aS = invDy2;
            aN = invDy2;
        }

        // Apply Neumann BCs at walls
        if (j == 0 && bc_y_lo_ == PoissonBC::Neumann) {
            aS = 0.0;
        }
        if (j == Ny_ - 1 && bc_y_hi_ == PoissonBC::Neumann) {
            aN = 0.0;
        }

        tri_lower_[j] = aS;
        tri_upper_[j] = aN;
        tri_diag_base_[j] = -(aS + aN);
    }
    cudaDeviceSynchronize();

    std::cout << "[FFT2DPoissonSolver] Precomputed y-direction coefficients"
              << (stretched ? " (stretched)" : " (uniform)") << "\n";
}

void FFT2DPoissonSolver::initialize_cusparse() {
    // Create cuSPARSE handle
    cusparseStatus_t status = cusparseCreate(&cusparse_handle_);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("[FFT2DPoissonSolver] cuSPARSE initialization failed");
    }
    cusparseSetStream(cusparse_handle_, stream_);

    // Allocate complex tridiagonal arrays for cuSPARSE
    // Each mode m has Ny equations, total n_modes batches
    const size_t tri_size = (size_t)N_modes_ * Ny_;
    cudaMallocManaged(&tri_dl_, tri_size * sizeof(cufftDoubleComplex));
    cudaMallocManaged(&tri_d_, tri_size * sizeof(cufftDoubleComplex));
    cudaMallocManaged(&tri_du_, tri_size * sizeof(cufftDoubleComplex));

    // Precompute the full tridiagonal matrices with eigenvalue shifts
    // For each mode m and position j:
    //   dl[m*Ny + j] = aS[j]                    (lower diagonal)
    //   d[m*Ny + j]  = diag_base[j] - lambda[m] (main diagonal)
    //   du[m*Ny + j] = aN[j]                    (upper diagonal)

    double* lam_x = lambda_x_;
    double* aS = tri_lower_;
    double* aN = tri_upper_;
    double* diag_base = tri_diag_base_;
    cufftDoubleComplex* dl = tri_dl_;
    cufftDoubleComplex* d = tri_d_;
    cufftDoubleComplex* du = tri_du_;
    int N_modes = N_modes_;
    int Ny = Ny_;

    #pragma omp target teams distribute parallel for collapse(2) \
        is_device_ptr(dl, d, du, lam_x, aS, aN, diag_base)
    for (int m = 0; m < N_modes; ++m) {
        for (int j = 0; j < Ny; ++j) {
            const size_t idx = (size_t)m * Ny + j;

            // Eigenvalue shift for this mode
            double shift = lam_x[m];
            bool is_zero_mode = (m == 0);

            // For zero mode (m=0), pin j=0 to zero for singularity
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
    cudaDeviceSynchronize();

    // Query buffer size for gtsv2StridedBatch
    status = cusparseZgtsv2StridedBatch_bufferSizeExt(
        cusparse_handle_,
        Ny_,             // m: system size
        tri_dl_,         // dl: lower diagonal
        tri_d_,          // d: main diagonal
        tri_du_,         // du: upper diagonal
        rhs_hat_,        // x: RHS/solution (in-place)
        N_modes_,        // batchCount
        Ny_,             // batchStride
        &cusparse_buffer_size_
    );

    if (status != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error("[FFT2DPoissonSolver] cuSPARSE buffer query failed");
    }

    cudaMalloc(&cusparse_buffer_, cusparse_buffer_size_);
    std::cout << "[FFT2DPoissonSolver] cuSPARSE buffer: " << cusparse_buffer_size_ << " bytes\n";
    std::cout << "[FFT2DPoissonSolver] Precomputed tridiagonal matrices\n";
}

void FFT2DPoissonSolver::cleanup() {
    if (plans_created_) {
        cufftDestroy(fft_plan_r2c_);
        cufftDestroy(fft_plan_c2r_);
    }
    if (stream_) cudaStreamDestroy(stream_);

    if (cusparse_handle_) cusparseDestroy(cusparse_handle_);
    if (cusparse_buffer_) cudaFree(cusparse_buffer_);

    if (in_pack_) cudaFree(in_pack_);
    if (out_pack_) cudaFree(out_pack_);
    if (rhs_hat_) cudaFree(rhs_hat_);
    if (lambda_x_) cudaFree(lambda_x_);
    if (lambda_y_) cudaFree(lambda_y_);
    if (fft_2d_r2c_) cufftDestroy(fft_2d_r2c_);
    if (fft_2d_c2r_) cufftDestroy(fft_2d_c2r_);

    if (tri_dl_) cudaFree(tri_dl_);
    if (tri_d_) cudaFree(tri_d_);
    if (tri_du_) cudaFree(tri_du_);
    if (tri_lower_) cudaFree(tri_lower_);
    if (tri_upper_) cudaFree(tri_upper_);
    if (tri_diag_base_) cudaFree(tri_diag_base_);
}

int FFT2DPoissonSolver::solve_device(double* rhs_ptr, double* p_ptr, const PoissonConfig& cfg) {
    const int Nx = Nx_;
    const int Ny = Ny_;
    const int Ng = mesh_->Nghost;
    const int Nx_full = Nx + 2 * Ng;
    const int Ny_full = Ny + 2 * Ng;
    // 2D solver uses 2D indexing: only accesses [0, Nx_full*Ny_full)
    const size_t plane_size = (size_t)Nx_full * Ny_full;
    // FFT normalization: 1/Nx for 1D x-FFT, 1/(Nx*Ny) for 2D FFT
    const double norm = fully_periodic_ ? 1.0 / (Nx * Ny) : 1.0 / Nx;

    double* packed = in_pack_;
    double* unpacked = out_pack_;

    // IMPORTANT: 2D vs 3D indexing in Mesh class
    // ===========================================
    // For 2D meshes, the solver uses Mesh::index(i,j) which returns:
    //   idx = j * Nx_full + i                    (range: [0, Nx_full*Ny_full-1])
    // NOT Mesh::index(i,j,k) which returns:
    //   idx = k * Nx_full * Ny_full + j * Nx_full + i  (different range!)
    //
    // The allocation IS 3D-sized (Nx_full * Ny_full * Nz_full), but the solver's
    // 2D path only uses the first 2D plane [0, Nx_full*Ny_full-1].
    // FFT2D must match this by using 2D indexing (no k offset).
    //
    // 1. Pack RHS from ghost layout to contiguous array + compute sum for mean removal.
    // Solvability on stretched grids requires Σ f*dyv = 0 (volume-weighted).
    // On uniform grids, dyv is empty and all weights equal dy, so Σ f*dy = dy*Σf.
    // Two separate target regions avoid mapping nullptr for the uniform-grid dyv.
    const bool stretched = mesh_->is_y_stretched();
    double weighted_sum = 0.0;

    if (stretched) {
        const double* dyv_ptr = mesh_->dyv.data();
        #pragma omp target teams distribute parallel for collapse(2) reduction(+:weighted_sum) \
            map(present: rhs_ptr[0:plane_size], dyv_ptr[0:Ny_full]) is_device_ptr(packed)
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                const size_t src_idx = (size_t)(j + Ng) * Nx_full + (i + Ng);
                const size_t dst_idx = (size_t)j * Nx + i;
                double val = rhs_ptr[src_idx];
                packed[dst_idx] = val;
                weighted_sum += val * dyv_ptr[j + Ng];
            }
        }
    } else {
        const double dy_val = mesh_->dy;
        #pragma omp target teams distribute parallel for collapse(2) reduction(+:weighted_sum) \
            map(present: rhs_ptr[0:plane_size]) is_device_ptr(packed)
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                const size_t src_idx = (size_t)(j + Ng) * Nx_full + (i + Ng);
                const size_t dst_idx = (size_t)j * Nx + i;
                double val = rhs_ptr[src_idx];
                packed[dst_idx] = val;
                weighted_sum += val * dy_val;
            }
        }
    }

    // 2. Subtract volume-weighted mean (for singular Neumann case)
    // mean = Σ(f*dyv) / (Nx * Ly) where Ly = Σ dyv[j]
    double Ly = mesh_->y_max - mesh_->y_min;
    double mean = weighted_sum / (Nx * Ly);
    #pragma omp target teams distribute parallel for is_device_ptr(packed)
    for (int idx = 0; idx < Nx * Ny; ++idx) {
        packed[idx] -= mean;
    }

    if (!fully_periodic_) {
        // 3. Forward 1D FFT in x: real -> complex (wall-bounded y path)
        cudaDeviceSynchronize();
        cufftResult fft_result = cufftExecD2Z(fft_plan_r2c_, in_pack_, rhs_hat_);
        if (fft_result != CUFFT_SUCCESS) {
            std::cerr << "[FFT2D] cufftExecD2Z failed: " << fft_result << "\n";
            return -1;
        }
        cudaStreamSynchronize(stream_);
    }

    // 4. Fix zero mode: set rhs_hat_[0] to 0 (pinned value for singularity)
    // For fully periodic, this is done after the 2D R2C FFT in step 5
    if (!fully_periodic_) {
        cudaMemsetAsync(rhs_hat_, 0, sizeof(cufftDoubleComplex), stream_);
    }

    // 5. Solve in y-direction
    if (fully_periodic_) {
        // Fully periodic: use 2D R2C FFT, eigenvalue division, 2D C2R
        // The 1D x-FFT (step 3) already ran. But for fully periodic we need
        // a 2D FFT. Re-do from packed real data using the 2D plan.

        // 2D R2C FFT: real [Ny][Nx] -> complex [Ny][Nx/2+1]
        cudaDeviceSynchronize();  // Ensure pack + mean subtraction done
        cufftResult r2c_result = cufftExecD2Z(fft_2d_r2c_, in_pack_, rhs_hat_);
        if (r2c_result != CUFFT_SUCCESS) {
            std::cerr << "[FFT2D] 2D R2C failed: " << r2c_result << "\n";
            return -1;
        }

        // Divide by -(lambda_x[kx] + lambda_y[ky]) for each (ky, kx)
        // cuFFT 2D R2C output layout: [ky * (Nx/2+1) + kx], row-major
        // Use raw double* cast to avoid nvc++ issues with cufftDoubleComplex in OMP target
        cudaDeviceSynchronize();
        double* hat_raw = reinterpret_cast<double*>(rhs_hat_);
        const double* lx = lambda_x_;
        const double* ly = lambda_y_;
        const int nx_c = N_modes_2d_;  // Nx/2+1
        const int ny = Ny_;
        #pragma omp target teams distribute parallel for collapse(2) \
            is_device_ptr(hat_raw, lx, ly)
        for (int ky = 0; ky < ny; ++ky) {
            for (int kx = 0; kx < nx_c; ++kx) {
                double denom = lx[kx] + ly[ky];
                size_t idx = 2 * ((size_t)ky * nx_c + kx);  // 2 doubles per complex
                if (denom > 1e-30) {
                    hat_raw[idx]     = -hat_raw[idx]     / denom;
                    hat_raw[idx + 1] = -hat_raw[idx + 1] / denom;
                } else {
                    hat_raw[idx]     = 0.0;
                    hat_raw[idx + 1] = 0.0;
                }
            }
        }

        // 2D C2R FFT: complex [Ny][Nx/2+1] -> real [Ny][Nx]
        cufftResult c2r_result = cufftExecZ2D(fft_2d_c2r_, rhs_hat_, out_pack_);
        if (c2r_result != CUFFT_SUCCESS) {
            std::cerr << "[FFT2D] 2D C2R failed: " << c2r_result << "\n";
            return -1;
        }
        cudaStreamSynchronize(stream_);

        // Skip the 1D inverse x-FFT (step 6) — the 2D C2R already did both directions
        // Jump directly to unpack (step 7)
    } else {
        // Wall-bounded y: cuSPARSE batched tridiagonal solve (in-place in rhs_hat_)
        cusparseStatus_t status = cusparseZgtsv2StridedBatch(
            cusparse_handle_,
            Ny_,             // m: system size
            tri_dl_,         // dl: lower diagonal
            tri_d_,          // d: main diagonal
            tri_du_,         // du: upper diagonal
            rhs_hat_,        // x: RHS/solution (in-place)
            N_modes_,        // batchCount
            Ny_,             // batchStride
            cusparse_buffer_
        );

        if (status != CUSPARSE_STATUS_SUCCESS) {
            std::cerr << "[FFT2D] cuSPARSE gtsv2StridedBatch failed: " << status << "\n";
            return -1;
        }
    }

    // 6. Inverse FFT: complex -> real (skip for fully periodic — already done in step 5)
    if (!fully_periodic_) {
        cufftResult ifft_result = cufftExecZ2D(fft_plan_c2r_, rhs_hat_, out_pack_);
        if (ifft_result != CUFFT_SUCCESS) {
            std::cerr << "[FFT2D] cufftExecZ2D failed: " << ifft_result << "\n";
            return -1;
        }
        cudaStreamSynchronize(stream_);
    }

    // 7. Unpack to ghost layout with normalization and BC application
    // NOTE: Use 2D indexing to match solver's 2D path (no k component)
    // BC encoding: 0=Dirichlet, 1=Neumann, 2=Periodic
    int bc_y_lo_int = (bc_y_lo_ == PoissonBC::Periodic) ? 2 : (bc_y_lo_ == PoissonBC::Neumann) ? 1 : 0;
    int bc_y_hi_int = (bc_y_hi_ == PoissonBC::Periodic) ? 2 : (bc_y_hi_ == PoissonBC::Neumann) ? 1 : 0;

    #pragma omp target teams distribute parallel for collapse(2) \
        map(present: p_ptr[0:plane_size]) is_device_ptr(unpacked)
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            // Source: [j * Nx + i] (contiguous FFT output)
            const size_t src_idx = (size_t)j * Nx + i;
            const double val = unpacked[src_idx] * norm;

            // Destination: 2D indexing [j+Ng][i+Ng] - matches solver's 2D path
            const size_t dst_idx = (size_t)(j + Ng) * Nx_full + (i + Ng);
            p_ptr[dst_idx] = val;

            // x-ghosts (periodic) - 2D indexing
            if (i == 0) {
                // x_lo ghost: copy from x_hi interior
                double src_val = unpacked[j * Nx + (Nx - 1)] * norm;
                p_ptr[(j + Ng) * Nx_full + 0] = src_val;
            }
            if (i == Nx - 1) {
                // x_hi ghost: copy from x_lo interior
                double src_val = unpacked[j * Nx + 0] * norm;
                p_ptr[(j + Ng) * Nx_full + (Nx + Ng)] = src_val;
            }

            // y-ghosts - 2D indexing
            if (j == 0) {
                if (bc_y_lo_int == 2) {  // Periodic
                    double src_val = unpacked[(Ny - 1) * Nx + i] * norm;
                    p_ptr[0 * Nx_full + (i + Ng)] = src_val;
                } else if (bc_y_lo_int == 1) {  // Neumann
                    p_ptr[0 * Nx_full + (i + Ng)] = val;
                } else {  // Dirichlet
                    p_ptr[0 * Nx_full + (i + Ng)] = -val;
                }
            }
            if (j == Ny - 1) {
                if (bc_y_hi_int == 2) {  // Periodic
                    double src_val = unpacked[0 * Nx + i] * norm;
                    p_ptr[(Ny + Ng) * Nx_full + (i + Ng)] = src_val;
                } else if (bc_y_hi_int == 1) {  // Neumann
                    p_ptr[(Ny + Ng) * Nx_full + (i + Ng)] = val;
                } else {  // Dirichlet
                    p_ptr[(Ny + Ng) * Nx_full + (i + Ng)] = -val;
                }
            }
        }
    }

    residual_ = 0.0;
    return 1;  // 1 iteration (direct solve)
}

#endif // USE_GPU_OFFLOAD

} // namespace nncfd

#endif // USE_GPU_OFFLOAD (outer guard)
