#pragma once

/// @file momentum_solver_hypre.hpp
/// @brief HYPRE-based momentum solver for GPU SIMPLE
///
/// Solves the SIMPLE momentum equation: A*u = b where A includes
/// diffusion (variable nu_eff), upwind convection, and Patankar
/// under-relaxation. Uses HYPRE's Struct interface with PFMG
/// or BiCGSTAB+PFMG preconditioner.
///
/// The same solver is used for both u and v (and w in 3D) momentum,
/// with different RHS and coefficient arrays for each component.

#ifdef USE_HYPRE

#include "mesh.hpp"
#include "HYPRE_config.h"
#include "_hypre_utilities.h"
#include "HYPRE.h"
#include "HYPRE_struct_ls.h"

namespace nncfd {

class HypreMomentumSolver {
public:
    HypreMomentumSolver(const Mesh& mesh, bool is_u_component);
    ~HypreMomentumSolver();

    // Non-copyable, non-movable
    HypreMomentumSolver(const HypreMomentumSolver&) = delete;
    HypreMomentumSolver& operator=(const HypreMomentumSolver&) = delete;

    /// Set the momentum matrix coefficients for this SIMPLE iteration.
    /// a_W, a_E, a_S, a_N: off-diagonal (face-based, positive values)
    /// a_B, a_F: back/front for 3D (nullptr for 2D)
    /// a_P: diagonal (center, positive, includes Patankar scaling)
    void set_coefficients(const double* a_W, const double* a_E,
                          const double* a_S, const double* a_N,
                          const double* a_B, const double* a_F,
                          const double* a_P, int n_cells);

    /// Solve A*x = b.
    /// x: solution (in/out, initial guess on input)
    /// b: right-hand side
    /// Returns: number of iterations used
    int solve(const double* b, double* x, double tol = 1e-4, int max_iter = 20);

    /// Get the final residual norm
    double final_residual() const { return final_residual_; }

private:
    const Mesh* mesh_;
    bool is_u_component_;  // true = u-momentum grid, false = v-momentum grid

    // HYPRE objects
    HYPRE_StructGrid grid_ = nullptr;
    HYPRE_StructStencil stencil_ = nullptr;
    HYPRE_StructMatrix A_ = nullptr;
    HYPRE_StructVector b_ = nullptr;
    HYPRE_StructVector x_ = nullptr;
    HYPRE_StructSolver solver_ = nullptr;
    HYPRE_StructSolver precond_ = nullptr;

    // Grid dimensions
    HYPRE_Int ilower_[3], iupper_[3];
    int nx_, ny_, nz_;  // interior dimensions for this component
    size_t n_cells_;

    // Host staging buffers
    std::vector<double> rhs_host_, x_host_;
    std::vector<double> coeff_host_;  // for matrix values

    double final_residual_ = 0.0;
    bool initialized_ = false;

    void create_grid();
    void create_stencil();
    void create_matrix();
    void create_vectors();
    void create_solver();
};

} // namespace nncfd

#endif // USE_HYPRE
