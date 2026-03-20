#pragma once

/// @file poisson_solver_fft_mpi.hpp
/// @brief Distributed GPU FFT Poisson solver with MPI pencil transpose
///
/// For single-rank: delegates to the existing FFTPoissonSolver (GPU).
/// For multi-rank (z-slab decomposition), all compute stays on GPU:
///   1. cuFFT 1D R2C in x (local, each rank has full x)
///   2. MPI_Alltoallv: z-slabs → kx-pencils (host-staged)
///   3. cuFFT 1D C2C in z (local, each rank now has full z)
///   4. Transpose j↔kz for cuSPARSE layout
///   5. cuSPARSE batched tridiagonal solve in y
///   6. Inverse of steps 4-1
///
/// Requires USE_MPI and USE_FFT_POISSON (implies GPU + CUDAToolkit).

#include "mesh.hpp"
#include "fields.hpp"
#include "poisson_solver.hpp"
#include "decomposition.hpp"

#include <vector>
#include <memory>

#ifdef USE_FFT_POISSON
#include <cufft.h>
#include <cusparse.h>
#include <cuda_runtime.h>
#endif

namespace nncfd {

#ifdef USE_FFT_POISSON
class FFTPoissonSolver;  // Forward declaration
#endif

/// Distributed FFT Poisson solver for MPI z-slab decomposition
class FFTMPIPoissonSolver {
public:
    /// @param mesh    Local mesh (full x/y, local z with ghosts)
    /// @param decomp  Domain decomposition (z-direction)
    FFTMPIPoissonSolver(const Mesh& mesh, const Decomposition& decomp);
    ~FFTMPIPoissonSolver();

    // Non-copyable
    FFTMPIPoissonSolver(const FFTMPIPoissonSolver&) = delete;
    FFTMPIPoissonSolver& operator=(const FFTMPIPoissonSolver&) = delete;

    /// Set boundary conditions
    void set_bc(PoissonBC x_lo, PoissonBC x_hi,
                PoissonBC y_lo, PoissonBC y_hi,
                PoissonBC z_lo, PoissonBC z_hi);

    /// Set spatial order for eigenvalue computation
    void set_space_order(int order);

    /// Check if this solver is suitable for the given configuration
    static bool is_suitable(PoissonBC x_lo, PoissonBC x_hi,
                           PoissonBC y_lo, PoissonBC y_hi,
                           PoissonBC z_lo, PoissonBC z_hi,
                           bool uniform_x, bool uniform_z);

    /// Solve on host memory (not supported for distributed — use solve_device)
    int solve(const ScalarField& rhs, ScalarField& p,
              const PoissonConfig& cfg = PoissonConfig());

    /// Solve on device memory (GPU path)
    int solve_device(double* rhs_ptr, double* p_ptr,
                     const PoissonConfig& cfg = PoissonConfig());

    /// Get final residual
    double residual() const { return residual_; }

    /// Check if using GPU path
    bool using_gpu() const;

private:
    const Mesh* mesh_;
    const Decomposition* decomp_;
    double residual_ = 0.0;
    int space_order_ = 2;

    // BCs
    PoissonBC bc_x_lo_ = PoissonBC::Periodic;
    PoissonBC bc_x_hi_ = PoissonBC::Periodic;
    PoissonBC bc_y_lo_ = PoissonBC::Neumann;
    PoissonBC bc_y_hi_ = PoissonBC::Neumann;
    PoissonBC bc_z_lo_ = PoissonBC::Periodic;
    PoissonBC bc_z_hi_ = PoissonBC::Periodic;

#ifdef USE_FFT_POISSON
    // Single-rank: delegate to serial FFT solver (GPU only)
    std::unique_ptr<FFTPoissonSolver> serial_solver_;
#endif

    bool distributed_ = false;
    int Nx_, Ny_, Nz_local_, Nz_global_, Ng_;

#ifdef USE_FFT_POISSON
    int Nx_c_ = 0;  // Nx/2 + 1 (R2C complex output size in x)

    // Lazy GPU initialization: set up on first solve_device() call
    bool gpu_initialized_ = false;
    void initialize_gpu();
    void free_gpu();

    // cuFFT plans
    cufftHandle x_r2c_ = 0;   // 1D R2C in x, batch = Ny * Nz_local
    cufftHandle x_c2r_ = 0;   // 1D C2R in x, batch = Ny * Nz_local
    cufftHandle z_fwd_ = 0;   // 1D C2C forward in z, batch = my_kx_count * Ny
    cufftHandle z_inv_ = 0;   // 1D C2C inverse in z, batch = my_kx_count * Ny
    cudaStream_t stream_ = nullptr;

    // GPU buffers (cudaMallocManaged)
    double* rhs_packed_ = nullptr;            // Nx * Ny * Nz_local (real)
    double* p_packed_ = nullptr;              // Nx * Ny * Nz_local (real)
    cufftDoubleComplex* hat_x_ = nullptr;     // Nx_c * Ny * Nz_local (after x-FFT)
    cufftDoubleComplex* pencil_ = nullptr;    // my_kx_count * Ny * Nz_global (after transpose)
    cufftDoubleComplex* mode_buf_ = nullptr;  // n_local_modes * Ny (cuSPARSE layout)

    // Eigenvalues and tridiag coefficients (cudaMallocManaged)
    double* lambda_x_ = nullptr;       // size Nx_c
    double* lambda_z_ = nullptr;       // size Nz_global
    double* tri_lower_ = nullptr;      // size Ny (aS)
    double* tri_upper_ = nullptr;      // size Ny (aN)
    double* tri_diag_base_ = nullptr;  // size Ny (-(aS+aN))
    double* dyv_dev_ = nullptr;        // size Ny (volume weights)
    double total_volume_ = 0.0;

    // cuSPARSE for batched tridiagonal solve
    cusparseHandle_t cusparse_handle_ = nullptr;
    cufftDoubleComplex* tri_dl_ = nullptr;  // n_local_modes * Ny
    cufftDoubleComplex* tri_d_ = nullptr;
    cufftDoubleComplex* tri_du_ = nullptr;
    void* cusparse_buffer_ = nullptr;
    size_t cusparse_buffer_size_ = 0;

    // MPI transpose parameters
    int my_kx_start_ = 0;
    int my_kx_count_ = 0;
    int n_local_modes_ = 0;  // my_kx_count * Nz_global
    std::vector<int> fwd_send_counts_, fwd_send_displs_;
    std::vector<int> fwd_recv_counts_, fwd_recv_displs_;
    std::vector<int> rev_send_counts_, rev_send_displs_;
    std::vector<int> rev_recv_counts_, rev_recv_displs_;

    // Host staging for MPI (non-CUDA-aware MPI)
    std::vector<double> send_host_;
    std::vector<double> recv_host_;

    // GPU pipeline steps
    int solve_device_distributed(double* rhs_ptr, double* p_ptr);
#endif
};

} // namespace nncfd
