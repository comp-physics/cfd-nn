#pragma once

#include "mesh.hpp"
#include "fields.hpp"
#include "poisson_solver.hpp"

#ifdef USE_GPU_OFFLOAD
#include <cufft.h>
#include <cusparse.h>
#include <cuda_runtime.h>
#endif

namespace nncfd {

/// FFT-hybrid Poisson solver for periodic x/z, wall-bounded y
/// Uses 2D FFT in x-z + batched tridiagonal solves in y
/// Optimal for channel/duct flows with uniform spacing in periodic directions
class FFTPoissonSolver {
public:
    explicit FFTPoissonSolver(const Mesh& mesh);
    ~FFTPoissonSolver();

    // Non-copyable
    FFTPoissonSolver(const FFTPoissonSolver&) = delete;
    FFTPoissonSolver& operator=(const FFTPoissonSolver&) = delete;

    /// Set boundary conditions (must be periodic in x/z, Neumann or Dirichlet in y)
    void set_bc(PoissonBC x_lo, PoissonBC x_hi,
                PoissonBC y_lo, PoissonBC y_hi,
                PoissonBC z_lo, PoissonBC z_hi);

    /// Set spatial order for eigenvalue computation (2 or 4)
    /// Must be called before first solve if using O4
    /// O4 uses MAC-consistent eigenvalues: λ = Dfc_O4 ∘ Dcf_O4 symbol
    void set_space_order(int order);

    /// Check if this solver is suitable for the given BC configuration
    static bool is_suitable(PoissonBC x_lo, PoissonBC x_hi,
                           PoissonBC y_lo, PoissonBC y_hi,
                           PoissonBC z_lo, PoissonBC z_hi,
                           bool uniform_x, bool uniform_z);

    /// Solve nabla^2 p = rhs on device
    /// rhs_ptr and p_ptr are device pointers with ghost cells
    /// Returns 1 (direct solver, no iterations)
    int solve_device(double* rhs_ptr, double* p_ptr, const PoissonConfig& cfg = PoissonConfig());

    /// Get final residual (always 0 for direct solver)
    double residual() const { return residual_; }

    /// Check if solver is using GPU
    bool using_gpu() const { return using_gpu_; }

private:
    const Mesh* mesh_;
    bool using_gpu_ = false;
    bool initialized_ = false;
    double residual_ = 0.0;

    // Boundary conditions (y only - x/z must be periodic)
    PoissonBC bc_y_lo_ = PoissonBC::Neumann;
    PoissonBC bc_y_hi_ = PoissonBC::Neumann;

    // Spatial order for eigenvalue computation (2 or 4)
    // O4 uses MAC-consistent Dcf_O4 ∘ Dfc_O4 eigenvalues
    int space_order_ = 2;

#ifdef USE_GPU_OFFLOAD
    // Dedicated CUDA stream for entire Poisson solve
    cudaStream_t stream_ = nullptr;

    // CUDA events for stream-to-stream synchronization (no host blocking)
    // ev_pack_done_: signals OMP pack kernels completed (stream 0 -> stream_)
    // ev_fft_done_: signals cuFFT/cuSPARSE completed (stream_ -> stream 0)
    cudaEvent_t ev_pack_done_ = nullptr;
    cudaEvent_t ev_fft_done_ = nullptr;

    // Device-resident sum for mean subtraction (avoids host scalar transfer)
    double* sum_dev_ = nullptr;

    // Partial sums buffer for block-level reduction (owned by this instance)
    double* partial_sums_ = nullptr;
    size_t partial_sums_size_ = 0;

    // cuFFT plans
    cufftHandle fft_plan_r2c_;  // Forward: Real to Complex
    cufftHandle fft_plan_c2r_;  // Inverse: Complex to Real
    bool plans_created_ = false;

    // cuFFT work area (locked to prevent per-solve allocation)
    void* fft_work_area_ = nullptr;
    size_t fft_work_size_ = 0;

    // Device buffers
    double* rhs_packed_ = nullptr;      // Packed RHS without ghosts (Nx*Ny*Nz)
    double* p_packed_ = nullptr;        // Packed solution without ghosts (Nx*Ny*Nz)
    cufftDoubleComplex* rhs_hat_ = nullptr;  // FFT of RHS (Nx*(Nz/2+1)*Ny)
    cufftDoubleComplex* p_hat_ = nullptr;    // FFT of solution (Nx*(Nz/2+1)*Ny)

    // Precomputed eigenvalues for x and z directions
    double* lambda_x_ = nullptr;  // Eigenvalues for x (size Nx)
    double* lambda_z_ = nullptr;  // Eigenvalues for z (size Nz/2+1)

    // Tridiagonal coefficients for y-direction (stretched grid support)
    double* tri_lower_ = nullptr;   // Lower diagonal aS(j), size Ny
    double* tri_upper_ = nullptr;   // Upper diagonal aN(j), size Ny
    double* tri_diag_base_ = nullptr;  // Base diagonal -(aS+aN), size Ny

    // Workspace for Thomas algorithm (size Nx * Nz_complex * Ny each)
    double* work_c_ = nullptr;      // c' values
    double* work_d_real_ = nullptr; // d' real part
    double* work_d_imag_ = nullptr; // d' imaginary part

    // cuSPARSE support for reference/debugging
    cusparseHandle_t cusparse_handle_ = nullptr;
    bool use_cusparse_ = true;  // cuSPARSE is 6x faster than custom Thomas
    void* cusparse_buffer_ = nullptr;
    size_t cusparse_buffer_size_ = 0;

    // Complex tridiagonal coefficients for cuSPARSE (per mode)
    // Each mode (kx, kz) has its own diagonal with eigenvalue shift
    cufftDoubleComplex* tri_dl_ = nullptr;  // Lower diagonal (Ny-1 per batch)
    cufftDoubleComplex* tri_d_ = nullptr;   // Main diagonal (Ny per batch)
    cufftDoubleComplex* tri_du_ = nullptr;  // Upper diagonal (Ny-1 per batch)

    // Initialize cuFFT plans and buffers
    void initialize_fft();

    // Initialize cuSPARSE
    void initialize_cusparse();

    // Solve tridiagonal using cuSPARSE (reference implementation)
    void solve_tridiagonal_cusparse();

    // Compute eigenvalues for periodic directions
    void compute_eigenvalues();

    // Compute tridiagonal coefficients for y-direction
    void compute_tridiagonal_coeffs();

    // ==================== CUDA Kernel Launchers ====================
    // These run on stream_ for full GPU-resident operation (no host scalars)

    // Pack RHS from ghost-cell layout to packed layout + compute sum on device
    // Writes sum to sum_dev_ (device scalar) - no host transfer!
    void launch_pack_and_sum(double* rhs_ptr);

    // Subtract mean from packed RHS (reads sum from sum_dev_)
    void launch_subtract_mean(size_t n_total);

    // Unpack solution + apply all BCs in single kernel
    void launch_unpack_and_bc(double* p_ptr);

    // ==================== Legacy OMP Functions (kept for reference) ====================
    // Pack RHS from ghost-cell layout to packed layout (on GPU)
    void pack_rhs(double* rhs_ptr);

    // Pack RHS and return sum (fused pack + reduction for mean subtraction)
    // DEPRECATED: Use launch_pack_and_sum() to avoid host scalar transfer
    double pack_rhs_with_sum(double* rhs_ptr);

    // Subtract mean from packed RHS (for nullspace handling)
    // DEPRECATED: Use launch_subtract_mean() for device-resident mean
    void subtract_mean(double mean);

    // Unpack solution from packed layout to ghost-cell layout (on GPU)
    void unpack_solution(double* p_ptr);

    // Fused unpack + BC application (single kernel)
    // DEPRECATED: Use launch_unpack_and_bc() for stream_ execution
    void unpack_and_apply_bc(double* p_ptr);

    // Apply boundary conditions to ghost cells (on GPU)
    void apply_bc_device(double* p_ptr);
#endif
};

} // namespace nncfd
