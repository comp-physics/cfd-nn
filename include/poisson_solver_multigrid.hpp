#pragma once

#include "mesh.hpp"
#include "fields.hpp"
#include "poisson_solver.hpp"
#include <vector>
#include <memory>

namespace nncfd {

/// Multigrid Poisson solver for incompressible flow
/// Uses geometric multigrid with V-cycles for O(N) convergence
class MultigridPoissonSolver {
public:
    /// Constructor
    MultigridPoissonSolver(const Mesh& mesh);
    
    /// Set boundary conditions
    void set_bc(PoissonBC x_lo, PoissonBC x_hi,
                PoissonBC y_lo, PoissonBC y_hi);
    
    /// Solve Poisson equation: nabla^2 p = rhs
    /// Returns number of V-cycles performed
    int solve(const ScalarField& rhs, ScalarField& p, const PoissonConfig& cfg);
    
    /// Get final residual
    double residual() const { return residual_; }
    
private:
    /// Grid level in multigrid hierarchy
    struct GridLevel {
        int Nx, Ny;           // Grid size
        double dx, dy;        // Grid spacing
        Mesh mesh;            // Mesh for this level
        ScalarField u;        // Solution
        ScalarField f;        // RHS
        ScalarField r;        // Residual
        
        GridLevel(int nx, int ny, double dx_, double dy_)
            : Nx(nx), Ny(ny), dx(dx_), dy(dy_), mesh()
        {
            mesh.init_uniform(nx, ny, 0.0, nx*dx_, 0.0, ny*dy_);
            u = ScalarField(mesh);
            f = ScalarField(mesh);
            r = ScalarField(mesh);
        }
    };
    
    const Mesh* mesh_;
    std::vector<std::unique_ptr<GridLevel>> levels_;
    
    // Boundary conditions
    PoissonBC bc_x_lo_ = PoissonBC::Periodic;
    PoissonBC bc_x_hi_ = PoissonBC::Periodic;
    PoissonBC bc_y_lo_ = PoissonBC::Neumann;
    PoissonBC bc_y_hi_ = PoissonBC::Neumann;
    
    double residual_ = 0.0;
    double dirichlet_val_ = 0.0;
    
    // Core multigrid operations
    void create_hierarchy();
    void smooth(int level, int iterations, double omega = 1.8);
    void compute_residual(int level);
    void restrict_residual(int fine_level);
    void prolongate_correction(int coarse_level);
    void apply_bc(int level);
    void vcycle(int level, int nu1 = 2, int nu2 = 2);
    
    // Direct solver for coarsest level
    void solve_coarsest(int iterations = 100);
    
    // Utility
    double compute_max_residual(int level);
    void subtract_mean(int level);
};

} // namespace nncfd

