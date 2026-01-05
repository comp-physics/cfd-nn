#pragma once

#include "mesh.hpp"
#include "poisson_solver.hpp"

#ifdef USE_GPU_OFFLOAD
#include <cuda_runtime.h>
#include <cufft.h>
#include <cusparse.h>
#endif

namespace nncfd {

/// 2D FFT Poisson solver for 2D meshes with periodic x, wall-bounded y
/// Uses 1D FFT in x + cuSPARSE batched tridiagonal solve in y
/// Optimal for 2D channel flows
///
/// Algorithm:
/// 1. Pack RHS from ghost layout to contiguous x-lines
/// 2. Subtract mean (for Neumann-Neumann singularity)
/// 3. 1D R2C FFT along x (batched over y)
/// 4. Batched tridiagonal solve: (d²/dy² - λ_x[m])*p_hat = rhs_hat
/// 5. 1D C2R IFFT
/// 6. Unpack solution to ghost layout with BCs
class FFT2DPoissonSolver {
public:
    explicit FFT2DPoissonSolver(const Mesh& mesh);
    ~FFT2DPoissonSolver();

    // Non-copyable
    FFT2DPoissonSolver(const FFT2DPoissonSolver&) = delete;
    FFT2DPoissonSolver& operator=(const FFT2DPoissonSolver&) = delete;

    /// Set boundary conditions
    /// x must be periodic, y can be Neumann or Dirichlet
    void set_bc(PoissonBC x_lo, PoissonBC x_hi,
                PoissonBC y_lo, PoissonBC y_hi);

    /// Check if this solver is suitable
    static bool is_suitable(PoissonBC x_lo, PoissonBC x_hi,
                           PoissonBC y_lo, PoissonBC y_hi,
                           bool uniform_x, bool is_2d);

#ifdef USE_GPU_OFFLOAD
    /// Solve on device
    int solve_device(double* rhs_ptr, double* p_ptr, const PoissonConfig& cfg = PoissonConfig());
#endif

    double residual() const { return residual_; }

private:
    const Mesh* mesh_;
    int Nx_, Ny_;
    double dx_, dy_;
    int N_modes_;  // Nx/2 + 1

    PoissonBC bc_y_lo_ = PoissonBC::Neumann;
    PoissonBC bc_y_hi_ = PoissonBC::Neumann;

    double residual_ = 0.0;

#ifdef USE_GPU_OFFLOAD
    cudaStream_t stream_ = nullptr;

    // cuFFT plans
    cufftHandle fft_plan_r2c_ = 0;
    cufftHandle fft_plan_c2r_ = 0;
    bool plans_created_ = false;

    // cuSPARSE for batched tridiagonal solve
    cusparseHandle_t cusparse_handle_ = nullptr;
    void* cusparse_buffer_ = nullptr;
    size_t cusparse_buffer_size_ = 0;

    // Precomputed tridiagonal coefficients [N_modes * Ny]
    cufftDoubleComplex* tri_dl_ = nullptr;  // Lower diagonal
    cufftDoubleComplex* tri_d_ = nullptr;   // Main diagonal
    cufftDoubleComplex* tri_du_ = nullptr;  // Upper diagonal

    // Device buffers (cudaMallocManaged for OMP interop)
    double* in_pack_ = nullptr;              // Packed real input [Nx * Ny]
    double* out_pack_ = nullptr;             // Packed real output [Nx * Ny]
    cufftDoubleComplex* rhs_hat_ = nullptr;  // Fourier coefficients [N_modes * Ny]

    // Precomputed eigenvalues for x direction
    double* lambda_x_ = nullptr;  // [N_modes]

    // Y-direction stencil coefficients
    double* tri_lower_ = nullptr;     // [Ny] - lower diagonal (a_j)
    double* tri_upper_ = nullptr;     // [Ny] - upper diagonal (c_j)
    double* tri_diag_base_ = nullptr; // [Ny] - base diagonal (before eigenvalue shift)

    void initialize_fft();
    void initialize_cusparse();
    void initialize_eigenvalues();
    void precompute_tridiagonal();
    void cleanup();
#endif
};

} // namespace nncfd
