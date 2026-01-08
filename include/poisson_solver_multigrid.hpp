#pragma once

#include "mesh.hpp"
#include "fields.hpp"
#include "poisson_solver.hpp"
#include <vector>
#include <memory>

#ifdef USE_GPU_OFFLOAD
// Forward declaration for CUDA Graph context
namespace nncfd { namespace mg_cuda { class CudaMGContext; } }
#endif

namespace nncfd {

/// Smoother type for multigrid
enum class MGSmootherType {
    Jacobi,     // Standard weighted Jacobi (reference, safe)
    Chebyshev   // Chebyshev polynomial acceleration (default, faster)
};

/// Multigrid Poisson solver for incompressible flow
/// Uses geometric multigrid with V-cycles for O(N) convergence
class MultigridPoissonSolver {
public:
    /// Constructor
    MultigridPoissonSolver(const Mesh& mesh);
    
    /// Destructor
    ~MultigridPoissonSolver();
    
    /// Set boundary conditions (2D version for backward compatibility)
    void set_bc(PoissonBC x_lo, PoissonBC x_hi,
                PoissonBC y_lo, PoissonBC y_hi);

    /// Set boundary conditions (3D version)
    void set_bc(PoissonBC x_lo, PoissonBC x_hi,
                PoissonBC y_lo, PoissonBC y_hi,
                PoissonBC z_lo, PoissonBC z_hi);
    
    /// Solve Poisson equation: nabla^2 p = rhs
    /// Returns number of V-cycles performed
    int solve(const ScalarField& rhs, ScalarField& p, const PoissonConfig& cfg);
    
#ifdef USE_GPU_OFFLOAD
    /// Device-resident solve: work directly on present-mapped pointers (no host staging)
    /// 
    /// Model 1 contract: Parameters are HOST pointers that must already be mapped to device
    /// via persistent `#pragma omp target enter data`. The solver uses `map(present: ...)` 
    /// internally, so no additional H↔D transfers occur during the solve.
    /// 
    /// @param rhs_present Host pointer to RHS array (must be present-mapped, size = Nx+2 * Ny+2)
    /// @param p_present Host pointer to solution array (must be present-mapped, size = Nx+2 * Ny+2)
    /// @param cfg Solver configuration
    /// @return Number of V-cycles performed
    /// @note For true device pointers (omp_target_alloc), use is_device_ptr instead (not implemented)
    int solve_device(double* rhs_present, double* p_present, const PoissonConfig& cfg);
#endif
    
    /// Get final residual ||r||_∞
    double residual() const { return residual_; }

    /// Get final residual ||r||_2
    double residual_l2() const { return residual_l2_; }

    /// Get RHS norm ||b||_∞ (computed at start of solve)
    double rhs_norm() const { return b_inf_; }

    /// Get RHS norm ||b||_2 (computed at start of solve)
    double rhs_norm_l2() const { return b_l2_; }

    /// Get initial residual ||r0||_∞
    double initial_residual() const { return r0_; }

    /// Get initial residual ||r0||_2
    double initial_residual_l2() const { return r0_l2_; }

    /// Set smoother type (Jacobi for reference/debugging, Chebyshev for performance)
    /// Can also be set via environment variable MG_SMOOTHER=jacobi|chebyshev
    void set_smoother(MGSmootherType type) { smoother_type_ = type; }
    MGSmootherType smoother_type() const { return smoother_type_; }

    /// Sync data to/from GPU for a specific multigrid level (always declared for ABI)
    /// These are public so RANSSolver can control when transfers happen
    void sync_level_to_gpu(int level);
    void sync_level_from_gpu(int level);
    
private:
    /// Grid level in multigrid hierarchy
    struct GridLevel {
        int Nx, Ny, Nz;           // Grid size (Nz=1 for 2D)
        double dx, dy, dz;        // Grid spacing (dz=1.0 for 2D)
        Mesh mesh;                // Mesh for this level
        ScalarField u;            // Solution
        ScalarField f;            // RHS
        ScalarField r;            // Residual

        /// 2D constructor (backward compatible)
        GridLevel(int nx, int ny, double dx_, double dy_)
            : Nx(nx), Ny(ny), Nz(1), dx(dx_), dy(dy_), dz(1.0), mesh()
        {
            mesh.init_uniform(nx, ny, 0.0, nx*dx_, 0.0, ny*dy_);
            u = ScalarField(mesh);
            f = ScalarField(mesh);
            r = ScalarField(mesh);
        }

        /// 3D constructor
        GridLevel(int nx, int ny, int nz, double dx_, double dy_, double dz_)
            : Nx(nx), Ny(ny), Nz(nz), dx(dx_), dy(dy_), dz(dz_), mesh()
        {
            mesh.init_uniform(nx, ny, nz, 0.0, nx*dx_, 0.0, ny*dy_, 0.0, nz*dz_);
            u = ScalarField(mesh);
            f = ScalarField(mesh);
            r = ScalarField(mesh);
        }

        bool is2D() const { return Nz == 1; }
    };
    
    const Mesh* mesh_;
    std::vector<std::unique_ptr<GridLevel>> levels_;
    
    // Boundary conditions
    PoissonBC bc_x_lo_ = PoissonBC::Periodic;
    PoissonBC bc_x_hi_ = PoissonBC::Periodic;
    PoissonBC bc_y_lo_ = PoissonBC::Neumann;
    PoissonBC bc_y_hi_ = PoissonBC::Neumann;
    PoissonBC bc_z_lo_ = PoissonBC::Periodic;
    PoissonBC bc_z_hi_ = PoissonBC::Periodic;
    
    double residual_ = 0.0;      // Final ||r||_∞
    double residual_l2_ = 0.0;   // Final ||r||_2
    double b_inf_ = 0.0;         // ||b||_∞ from last solve
    double b_l2_ = 0.0;          // ||b||_2 from last solve
    double r0_ = 0.0;            // Initial residual ||r0||_∞ from last solve
    double r0_l2_ = 0.0;         // Initial residual ||r0||_2 from last solve
    double dirichlet_val_ = 0.0;
    MGSmootherType smoother_type_ = MGSmootherType::Chebyshev;  // Default to faster smoother

#ifdef USE_GPU_OFFLOAD
    // CUDA Graph support for reduced kernel launch overhead
    std::unique_ptr<mg_cuda::CudaMGContext> cuda_ctx_;
    bool use_cuda_graphs_ = false;  // Set via environment variable MG_USE_CUDA_GRAPHS=1
    void initialize_cuda_graphs();
#endif

    // Core multigrid operations
    void create_hierarchy();
    void smooth(int level, int iterations, double omega = 1.8);
    void smooth_jacobi(int level, int iterations, double omega = 0.8);  // GPU-optimized Jacobi
    void smooth_chebyshev(int level, int degree = 4);  // Chebyshev polynomial smoother
    void compute_residual(int level);
    void restrict_residual(int fine_level);
    void prolongate_correction(int coarse_level);
    void apply_bc(int level);
    void apply_bc_to_residual(int level);  // Apply periodic BCs to residual for restriction
    void vcycle(int level, int nu1 = 2, int nu2 = 2);
    
    // Direct solver for coarsest level
    void solve_coarsest(int iterations = 100);
    
    // Utility
    double compute_max_residual(int level);

    /// Fused residual computation + norm calculation (single pass over memory)
    /// Returns {||r||_∞, ||r||_2} and stores residual in r array
    /// Much faster than compute_residual() followed by compute_max_residual()
    void compute_residual_and_norms(int level, double& r_inf, double& r_l2);

    void subtract_mean(int level);
    
    // GPU buffer management (always present for ABI stability)
    bool gpu_ready_ = false;
    std::vector<double*> u_ptrs_;  // Device pointers for u at each level
    std::vector<double*> f_ptrs_;  // Device pointers for f at each level
    std::vector<double*> r_ptrs_;  // Device pointers for r at each level
    std::vector<double*> tmp_ptrs_;  // Scratch buffer for Jacobi ping-pong
    std::vector<size_t> level_sizes_;  // Total size for each level
    
    void initialize_gpu_buffers();
    void cleanup_gpu_buffers();
};

} // namespace nncfd

