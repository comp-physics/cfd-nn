#pragma once

/// @file poisson_solver_fft_mpi.hpp
/// @brief Distributed FFT Poisson solver with MPI pencil transpose
///
/// For single-rank: delegates to the existing FFTPoissonSolver.
/// For multi-rank (z-slab decomposition):
///   1. Forward 1D FFT in x (local, each rank has full x)
///   2. MPI_Alltoallv transpose: z-slabs → z-pencils
///   3. Forward 1D FFT in z (local, each rank now has full z)
///   4. Tridiagonal solve in y for each (kx,kz) mode
///   5. Inverse 1D FFT in z
///   6. MPI_Alltoallv transpose: z-pencils → z-slabs
///   7. Inverse 1D FFT in x
///
/// Requires USE_MPI and USE_FFT_POISSON.

#include "mesh.hpp"
#include "fields.hpp"
#include "poisson_solver.hpp"
#include "decomposition.hpp"

#include <vector>
#include <memory>

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

    /// Solve on host memory (CPU path)
    /// @param rhs  Right-hand side field (local z-slab)
    /// @param p    Pressure solution (local z-slab)
    /// @return Number of iterations (1 for direct solver)
    int solve(const ScalarField& rhs, ScalarField& p,
              const PoissonConfig& cfg = PoissonConfig());

    /// Solve on device memory (GPU path)
    /// @param rhs_ptr  Device pointer to RHS with ghost cells
    /// @param p_ptr    Device pointer to solution with ghost cells
    /// @return 1 (direct solver)
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

    // Multi-rank distributed solve buffers
    bool distributed_ = false;
    int Nx_, Ny_, Nz_local_, Nz_global_, Ng_;

    // MPI transpose buffers (host-staged for CPU path)
    std::vector<double> send_buf_;  // packed local data for alltoallv
    std::vector<double> recv_buf_;  // received pencil data
    std::vector<int> send_counts_;
    std::vector<int> send_displs_;
    std::vector<int> recv_counts_;
    std::vector<int> recv_displs_;

    // Precomputed eigenvalues
    std::vector<double> lambda_x_;  // size Nx
    std::vector<double> lambda_z_;  // size Nz_global

    // Tridiagonal coefficients for y-direction
    std::vector<double> tri_lower_;  // aS(j), size Ny
    std::vector<double> tri_upper_;  // aN(j), size Ny
    std::vector<double> tri_diag_;   // -(aS+aN), size Ny

    // Work arrays for CPU distributed solve
    std::vector<double> work_real_;
    std::vector<double> work_imag_;

    void initialize_distributed();
    void compute_eigenvalues();
    void compute_tridiagonal_coeffs();
    void compute_alltoallv_params();

    // CPU distributed solve implementation
    int solve_distributed_cpu(const ScalarField& rhs, ScalarField& p);
};

} // namespace nncfd
