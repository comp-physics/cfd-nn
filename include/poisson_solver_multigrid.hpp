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
    
    /// Destructor
    ~MultigridPoissonSolver();
    
    /// Set boundary conditions
    void set_bc(PoissonBC x_lo, PoissonBC x_hi,
                PoissonBC y_lo, PoissonBC y_hi);
    
    /// Solve Poisson equation: nabla^2 p = rhs
    /// Returns number of V-cycles performed
    int solve(const ScalarField& rhs, ScalarField& p, const PoissonConfig& cfg);
    
#ifdef USE_GPU_OFFLOAD
    /// Device-resident solve: work directly on GPU pointers (no host staging)
    /// @param rhs_device Device pointer to RHS array (size = Nx+2 * Ny+2)
    /// @param p_device Device pointer to solution array (size = Nx+2 * Ny+2)
    /// @param cfg Solver configuration
    /// @return Number of V-cycles performed
    /// @note Both arrays must already be on device with map(present:)
    int solve_device(double* rhs_device, double* p_device, const PoissonConfig& cfg);
#endif
    
    /// Get final residual
    double residual() const { return residual_; }
    
    /// Sync data to/from GPU for a specific multigrid level (always declared for ABI)
    /// These are public so RANSSolver can control when transfers happen
    void sync_level_to_gpu(int level);
    void sync_level_from_gpu(int level);
    
#ifdef USE_GPU_OFFLOAD
    /// Get device pointers for direct GPU-GPU copies
    double* get_u_device_ptr(int level) { return gpu_ready_ ? u_ptrs_[level] : nullptr; }
    double* get_f_device_ptr(int level) { return gpu_ready_ ? f_ptrs_[level] : nullptr; }
    size_t get_level_size(int level) const { return gpu_ready_ ? level_sizes_[level] : 0; }
#endif
    
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
    
    // GPU buffer management (always present for ABI stability)
    bool gpu_ready_ = false;
    std::vector<double*> u_ptrs_;  // Device pointers for u at each level
    std::vector<double*> f_ptrs_;  // Device pointers for f at each level
    std::vector<double*> r_ptrs_;  // Device pointers for r at each level
    std::vector<size_t> level_sizes_;  // Total size for each level
    
    void initialize_gpu_buffers();
    void cleanup_gpu_buffers();
};

} // namespace nncfd

