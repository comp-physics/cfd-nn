#pragma once

#include "mesh.hpp"
#include "poisson_solver.hpp"

#ifdef USE_GPU_OFFLOAD
#include <cuda_runtime.h>
#include <cufft.h>
#endif

namespace nncfd {

/// 2D FFT Poisson solver for 2D meshes with periodic x, wall-bounded y
/// Uses 1D FFT in x + Jacobi iteration for 1D Helmholtz in y
/// Optimal for 2D channel flows
///
/// Algorithm:
/// 1. Pack RHS from ghost layout to contiguous x-lines
/// 2. 1D R2C FFT along x (batched over y)
/// 3. For each mode m: solve 1D Helmholtz in y via Jacobi iteration
///    (d²p/dy² - λ_x[m]*p = f_hat[m])
/// 4. 1D C2R IFFT
/// 5. Unpack solution to ghost layout
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

    // Device buffers
    double* in_pack_ = nullptr;              // Packed real input [Ny * Nx]
    double* out_pack_ = nullptr;             // Packed real output [Ny * Nx]
    cufftDoubleComplex* rhs_hat_ = nullptr;  // Fourier coefficients [N_modes * Ny]
    cufftDoubleComplex* p_hat_ = nullptr;    // Solution in Fourier space [N_modes * Ny]

    // Work buffers for Jacobi iteration (split real/imag)
    double* work_real_ = nullptr;
    double* work_imag_ = nullptr;

    // Precomputed eigenvalues for x direction
    double* lambda_x_ = nullptr;  // [N_modes]

    // Mean subtraction buffers
    double* partial_sums_ = nullptr;
    double* sum_dev_ = nullptr;
    int num_blocks_ = 0;

    void initialize_fft();
    void initialize_eigenvalues();
    void cleanup();
#endif
};

} // namespace nncfd
