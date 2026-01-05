#pragma once

#include "mesh.hpp"
#include "fields.hpp"
#include "poisson_solver.hpp"

#ifdef USE_GPU_OFFLOAD
#include <cuda_runtime.h>
#include <cufft.h>
#endif

namespace nncfd {

/// 1-Periodic FFT Poisson solver for cases with exactly one periodic direction
/// Uses 1D FFT in the periodic direction + 2D Helmholtz solve per mode
///
/// Supports:
/// - Periodic in x, walls (Neumann) in y and z (e.g., duct flow)
/// - Periodic in z, walls (Neumann) in x and y (alternative orientation)
///
/// Algorithm:
/// 1. Pack RHS from ghost layout to contiguous x-lines
/// 2. 1D R2C FFT along periodic direction (batched over yz)
/// 3. For each mode m: solve (L_yz + lambda_x(m)*I) p_hat = rhs_hat
/// 4. 1D C2R IFFT
/// 5. Unpack solution to ghost layout
class FFT1DPoissonSolver {
public:
    /// Constructor
    /// @param mesh The computational mesh
    /// @param periodic_dir Which direction is periodic: 0=x, 2=z
    FFT1DPoissonSolver(const Mesh& mesh, int periodic_dir = 0);

    /// Destructor
    ~FFT1DPoissonSolver();

    // Non-copyable, non-movable (owns GPU resources)
    FFT1DPoissonSolver(const FFT1DPoissonSolver&) = delete;
    FFT1DPoissonSolver& operator=(const FFT1DPoissonSolver&) = delete;
    FFT1DPoissonSolver(FFT1DPoissonSolver&&) = delete;
    FFT1DPoissonSolver& operator=(FFT1DPoissonSolver&&) = delete;

    /// Set boundary conditions
    /// The periodic direction must have Periodic BCs
    /// The other two directions can be Neumann or Dirichlet
    void set_bc(PoissonBC x_lo, PoissonBC x_hi,
                PoissonBC y_lo, PoissonBC y_hi,
                PoissonBC z_lo, PoissonBC z_hi);

    /// Check if this solver is suitable for the given BCs
    /// Returns true if exactly one of (x,z) is periodic and that direction has uniform spacing
    static bool is_suitable(PoissonBC x_lo, PoissonBC x_hi,
                           PoissonBC y_lo, PoissonBC y_hi,
                           PoissonBC z_lo, PoissonBC z_hi,
                           bool uniform_x, bool uniform_z,
                           bool is_3d);

#ifdef USE_GPU_OFFLOAD
    /// Device-resident solve
    /// @param rhs_ptr Host pointer to RHS (must be present-mapped)
    /// @param p_ptr Host pointer to solution (must be present-mapped)
    /// @param cfg Solver configuration
    /// @return Number of iterations (total across all modes)
    int solve_device(double* rhs_ptr, double* p_ptr, const PoissonConfig& cfg);
#endif

    /// Get final residual
    double residual() const { return residual_; }

private:
    const Mesh* mesh_;
    int periodic_dir_;  // 0 = x periodic, 2 = z periodic

    // Grid dimensions
    int Nx_, Ny_, Nz_;
    int N_periodic_;      // Size in periodic direction
    int N_modes_;         // Number of Fourier modes (N_periodic/2 + 1)
    int N_yz_;            // Ny * Nz (for x-periodic) or Nx * Ny (for z-periodic)

    // Grid spacing
    double dx_, dy_, dz_;
    double d_periodic_;   // Spacing in periodic direction

    // Boundary conditions (for non-periodic directions)
    PoissonBC bc_y_lo_ = PoissonBC::Neumann;
    PoissonBC bc_y_hi_ = PoissonBC::Neumann;
    PoissonBC bc_z_lo_ = PoissonBC::Neumann;
    PoissonBC bc_z_hi_ = PoissonBC::Neumann;
    // For z-periodic case, we also need x BCs
    PoissonBC bc_x_lo_ = PoissonBC::Neumann;
    PoissonBC bc_x_hi_ = PoissonBC::Neumann;

    double residual_ = 0.0;

#ifdef USE_GPU_OFFLOAD
    // CUDA stream for all operations
    cudaStream_t stream_ = nullptr;

    // cuFFT plans
    cufftHandle fft_plan_r2c_ = 0;
    cufftHandle fft_plan_c2r_ = 0;
    bool plans_created_ = false;

    // Device buffers
    double* in_pack_ = nullptr;              // Packed real input [batch * N_periodic]
    double* out_pack_ = nullptr;             // Packed real output [batch * N_periodic]
    cufftDoubleComplex* rhs_hat_ = nullptr;  // Fourier coefficients [N_modes * N_yz]
    cufftDoubleComplex* p_hat_ = nullptr;    // Solution in Fourier space [N_modes * N_yz]

    // Precomputed eigenvalues
    double* lambda_ = nullptr;               // Discrete eigenvalues [N_modes]

    // 2D Helmholtz solver workspace (for Jacobi/Chebyshev)
    double* work_real_ = nullptr;            // Real part workspace [N_modes * N_yz]
    double* work_imag_ = nullptr;            // Imag part workspace [N_modes * N_yz]

    // For mean subtraction
    double* partial_sums_ = nullptr;
    double* sum_dev_ = nullptr;
    int num_blocks_ = 0;

    // Initialization
    void initialize_fft();
    void initialize_eigenvalues();
    void cleanup();

    // 2D Helmholtz solve for all modes
    // Uses weighted Jacobi iteration (baseline) or Chebyshev (optimized)
    void solve_helmholtz_2d(int iterations, double omega);
#endif
};

} // namespace nncfd
