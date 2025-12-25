#pragma once

#include "mesh.hpp"
#include "fields.hpp"
#include <functional>

namespace nncfd {

/// Boundary condition type for Poisson equation
enum class PoissonBC {
    Dirichlet,
    Neumann,
    Periodic
};

/// Configuration for Poisson solver
struct PoissonConfig {
    double tol = 1e-6;       ///< Convergence tolerance
    int max_iter = 10000;    ///< Maximum Poisson iterations per solve (per time step).
                             ///< For SOR-based PoissonSolver: max SOR sweeps.
                             ///< For MultigridPoissonSolver: max V-cycles.
    double omega = 1.5;      ///< SOR relaxation parameter (1 < omega < 2 for over-relaxation)
    bool verbose = false;    ///< Print convergence info
};

/// Poisson solver for pressure equation
/// Solves: nabla^2p = f with specified boundary conditions
class PoissonSolver {
public:
    explicit PoissonSolver(const Mesh& mesh);
    
    /// Set boundary conditions for each boundary
    /// x_lo/x_hi: left/right boundaries
    /// y_lo/y_hi: bottom/top boundaries
    void set_bc(PoissonBC x_lo, PoissonBC x_hi,
                PoissonBC y_lo, PoissonBC y_hi);
    
    /// Set Dirichlet boundary value (for Dirichlet BC)
    void set_dirichlet_value(double val) { dirichlet_val_ = val; }
    
    /// Solve nabla^2p = rhs
    /// Returns number of iterations taken
    int solve(const ScalarField& rhs, ScalarField& p, const PoissonConfig& cfg = PoissonConfig());
    
    /// Get final residual
    double residual() const { return residual_; }
    
private:
    const Mesh* mesh_;
    PoissonBC bc_x_lo_ = PoissonBC::Periodic;
    PoissonBC bc_x_hi_ = PoissonBC::Periodic;
    PoissonBC bc_y_lo_ = PoissonBC::Neumann;
    PoissonBC bc_y_hi_ = PoissonBC::Neumann;
    double dirichlet_val_ = 0.0;
    double residual_ = 0.0;
    
    /// Apply boundary conditions to pressure field
    void apply_bc(ScalarField& p);
    
    /// Compute residual norm
    double compute_residual(const ScalarField& rhs, const ScalarField& p);
    
    /// Single SOR iteration
    void sor_iteration(const ScalarField& rhs, ScalarField& p, double omega);
    
    /// Red-black SOR iteration (better parallelization potential)
    void sor_rb_iteration(const ScalarField& rhs, ScalarField& p, double omega);
};

} // namespace nncfd


