#pragma once

#ifdef USE_HYPRE

#include "mesh.hpp"
#include "fields.hpp"
#include "poisson_solver.hpp"  // For PoissonBC and PoissonConfig

// HYPRE headers
#include "HYPRE.h"
#include "HYPRE_struct_ls.h"
#include "_hypre_struct_mv.h"

namespace nncfd {

/// HYPRE-based Poisson solver using PFMG (Parallel Semicoarsening Multigrid)
///
/// This solver uses HYPRE's structured grid interface (Struct) with the PFMG
/// preconditioner/solver for optimal performance on structured rectilinear grids.
///
/// Features:
/// - Supports uniform and stretched grids (variable dx, dy, dz)
/// - Handles periodic, Neumann, and Dirichlet boundary conditions
/// - Uses CUDA backend for GPU acceleration when available
/// - Warm-start capability (uses previous solution as initial guess)
///
/// The Laplacian is discretized using standard 2nd-order finite differences:
///   (u_{i+1} - 2u_i + u_{i-1})/dx^2 + (y terms) + (z terms) = f_i
///
/// For stretched grids, coefficients are computed from local cell spacings
/// using a finite-volume consistent formulation.
class HyprePoissonSolver {
public:
    /// Construct solver for the given mesh
    /// @param mesh The computational mesh (uniform or stretched)
    explicit HyprePoissonSolver(const Mesh& mesh);

    /// Destructor - cleans up HYPRE objects
    ~HyprePoissonSolver();

    // Non-copyable, non-movable (HYPRE objects are complex to manage)
    HyprePoissonSolver(const HyprePoissonSolver&) = delete;
    HyprePoissonSolver& operator=(const HyprePoissonSolver&) = delete;
    HyprePoissonSolver(HyprePoissonSolver&&) = delete;
    HyprePoissonSolver& operator=(HyprePoissonSolver&&) = delete;

    /// Set boundary conditions (2D)
    void set_bc(PoissonBC x_lo, PoissonBC x_hi,
                PoissonBC y_lo, PoissonBC y_hi);

    /// Set boundary conditions (3D)
    void set_bc(PoissonBC x_lo, PoissonBC x_hi,
                PoissonBC y_lo, PoissonBC y_hi,
                PoissonBC z_lo, PoissonBC z_hi);

    /// Solve the Poisson equation: ∇²p = rhs (host arrays)
    ///
    /// @param rhs Right-hand side (divergence / dt for projection)
    /// @param p Pressure field - used as initial guess (warm-start) and output
    /// @param cfg Solver configuration (tolerance, max iterations)
    /// @return Number of PFMG iterations performed
    int solve(const ScalarField& rhs, ScalarField& p,
              const PoissonConfig& cfg = PoissonConfig());

    /// Solve the Poisson equation with GPU-resident data
    ///
    /// Takes device pointers directly - data stays on GPU throughout solve.
    /// Arrays must be in (Nx+2*Ng)*(Ny+2*Ng)*(Nz+2*Ng) layout with ghost cells.
    ///
    /// @param rhs_ptr Device pointer to RHS array (with ghost cells)
    /// @param p_ptr Device pointer to pressure array (input: initial guess, output: solution)
    /// @param cfg Solver configuration
    /// @return Number of PFMG iterations performed
    int solve_device(double* rhs_ptr, double* p_ptr,
                     const PoissonConfig& cfg = PoissonConfig());

    /// Get the final residual norm from the last solve
    double residual() const { return residual_; }

    /// Check if CUDA backend is active
    bool using_cuda() const { return using_cuda_; }

    /// Check if solver is properly initialized
    bool is_initialized() const { return initialized_; }

private:
    const Mesh* mesh_;
    double residual_ = 0.0;
    bool initialized_ = false;
    bool matrix_assembled_ = false;
    bool using_cuda_ = false;  // True if CUDA backend is active

    // Host staging buffers for packed data (no ghost cells, Nx*Ny*Nz layout)
    // HYPRE handles GPU transfers internally when using MEMORY_HOST + EXEC_DEVICE
    std::vector<double> rhs_host_;
    std::vector<double> x_host_;
    size_t device_buffer_size_ = 0;

    // Boundary conditions
    PoissonBC bc_x_lo_ = PoissonBC::Periodic;
    PoissonBC bc_x_hi_ = PoissonBC::Periodic;
    PoissonBC bc_y_lo_ = PoissonBC::Neumann;
    PoissonBC bc_y_hi_ = PoissonBC::Neumann;
    PoissonBC bc_z_lo_ = PoissonBC::Periodic;
    PoissonBC bc_z_hi_ = PoissonBC::Periodic;

    // Grid extents (0-based interior cells)
    HYPRE_Int ilower_[3];  // Lower bounds [0, 0, 0]
    HYPRE_Int iupper_[3];  // Upper bounds [Nx-1, Ny-1, Nz-1]

    // HYPRE objects
    HYPRE_StructGrid grid_ = nullptr;
    HYPRE_StructStencil stencil_ = nullptr;
    HYPRE_StructMatrix A_ = nullptr;
    HYPRE_StructVector b_ = nullptr;
    HYPRE_StructVector x_ = nullptr;
    HYPRE_StructSolver solver_ = nullptr;

    // Coefficient storage (for stretched grids)
    std::vector<double> coeffs_;  // [n_cells * 7] for 7-point stencil

    // Setup methods
    void initialize_hypre();
    void create_grid();
    void create_stencil();
    void create_matrix();
    void assemble_matrix();
    void create_vectors();
    void create_solver();
    void setup_solver(const PoissonConfig& cfg);

    // Helper methods
    bool has_dirichlet_bc() const;
    bool needs_nullspace_handling() const;
    void compute_laplacian_coefficients();

    // Stencil indices
    static constexpr int STENCIL_CENTER = 0;
    static constexpr int STENCIL_WEST = 1;   // i-1
    static constexpr int STENCIL_EAST = 2;   // i+1
    static constexpr int STENCIL_SOUTH = 3;  // j-1
    static constexpr int STENCIL_NORTH = 4;  // j+1
    static constexpr int STENCIL_BACK = 5;   // k-1
    static constexpr int STENCIL_FRONT = 6;  // k+1
};

} // namespace nncfd

#endif // USE_HYPRE
