/// @file mg_cuda_kernels.hpp
/// @brief CUDA kernel infrastructure for multigrid Poisson solver
///
/// This file provides CUDA-native kernels for the multigrid smoother operations,
/// enabling CUDA Graph capture for reduced kernel launch overhead.
///
/// Key components:
/// - Chebyshev smoother kernels (3D Laplacian + weighted update)
/// - Boundary condition kernels (periodic, Neumann, Dirichlet)
/// - CUDA Graph capture/replay for smoother sequences
///
/// Performance rationale:
/// - Nsys profiling shows ~37% of MG time in cudaStreamSynchronize
/// - ~290 kernel launches per V-cycle, median 1-5 µs each
/// - CUDA Graphs reduce launch overhead by batching the entire smoother sequence

#pragma once

#ifdef USE_GPU_OFFLOAD

#include <cuda_runtime.h>
#include <vector>

namespace nncfd {
namespace mg_cuda {

/// Boundary condition types (matches PoissonBC enum)
enum class BC : int {
    Dirichlet = 0,
    Neumann = 1,
    Periodic = 2
};

/// Per-level MG configuration for CUDA kernels
struct LevelConfig {
    int Nx, Ny, Nz;           // Interior grid dimensions
    int Ng;                   // Number of ghost cells (typically 1)
    double dx2, dy2, dz2;     // Grid spacing squared (1/h^2)
    double coeff;             // Diagonal coefficient for Jacobi
    size_t total_size;        // Total array size including ghosts

    // Array pointers (device memory)
    double* u;                // Solution array
    double* f;                // RHS array
    double* r;                // Residual array
    double* tmp;              // Scratch buffer for smoother
};

/// CUDA Graph-based smoother for a single MG level
class CudaSmootherGraph {
public:
    CudaSmootherGraph() = default;
    ~CudaSmootherGraph();

    /// Initialize for a given level configuration
    /// @param config Level parameters (grid size, pointers, etc.)
    /// @param degree Chebyshev polynomial degree
    /// @param bc_x_lo/hi, bc_y_lo/hi, bc_z_lo/hi Boundary conditions
    void initialize(const LevelConfig& config, int degree,
                    BC bc_x_lo, BC bc_x_hi,
                    BC bc_y_lo, BC bc_y_hi,
                    BC bc_z_lo, BC bc_z_hi);

    /// Execute the captured smoother graph
    /// @param stream CUDA stream for execution
    void execute(cudaStream_t stream);

    /// Check if graph is initialized and valid
    bool is_valid() const { return graph_exec_ != nullptr; }

    /// Destroy graph resources
    void destroy();

    /// Debug: print captured pointers
    void debug_print_pointers() const;

private:
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    LevelConfig config_{};
    int degree_ = 0;
    BC bc_x_lo_, bc_x_hi_;
    BC bc_y_lo_, bc_y_hi_;
    BC bc_z_lo_, bc_z_hi_;

    /// Capture the smoother kernel sequence into a CUDA Graph
    void capture_graph(cudaStream_t stream);

    /// Launch a single Chebyshev iteration (for graph capture)
    void launch_chebyshev_iteration(cudaStream_t stream, int k, double omega);

    /// Launch Chebyshev iteration with fused periodic BCs (for graph capture)
    void launch_chebyshev_periodic(cudaStream_t stream, double omega);

    /// Launch BC kernel (for graph capture)
    void launch_bc_kernel(cudaStream_t stream);

    /// Launch copy kernel tmp -> u (for graph capture)
    void launch_copy_kernel(cudaStream_t stream);
};

/// CUDA stream and graph manager for entire MG solver
class CudaMGContext {
public:
    CudaMGContext();
    ~CudaMGContext();

    /// Get the dedicated MG stream
    cudaStream_t stream() const { return stream_; }

    /// Synchronize the MG stream
    void synchronize();

    /// Initialize smoother graphs for all levels
    /// @param levels Vector of level configurations
    /// @param degree Chebyshev degree
    /// @param bc_* Boundary conditions
    void initialize_smoother_graphs(
        const std::vector<LevelConfig>& levels,
        int degree,
        BC bc_x_lo, BC bc_x_hi,
        BC bc_y_lo, BC bc_y_hi,
        BC bc_z_lo, BC bc_z_hi);

    /// Execute smoother for a given level (uses internal stream)
    void smooth(int level);

    /// Execute smoother for a given level on specified stream
    /// Use this with OpenMP's stream to avoid cross-stream sync overhead
    void smooth(int level, cudaStream_t stream);

    /// Debug: print captured pointers for a level
    void debug_graph_pointers(int level) const;

    /// Check if CUDA Graphs are enabled and valid
    bool graphs_enabled() const { return !smoother_graphs_.empty(); }

private:
    cudaStream_t stream_ = nullptr;
    std::vector<CudaSmootherGraph> smoother_graphs_;
};

// ============================================================================
// CUDA Kernel Launch Functions (called during graph capture)
// ============================================================================

/// Launch 3D Chebyshev smoother kernel
/// Computes: tmp = (1-omega)*u + omega * jacobi_update(u)
void launch_chebyshev_3d(
    cudaStream_t stream,
    double* u, const double* f, double* tmp,
    int Nx, int Ny, int Nz, int Ng,
    double dx2, double dy2, double dz2, double coeff,
    double omega);

/// Launch 3D boundary condition kernel
void launch_bc_3d(
    cudaStream_t stream,
    double* u,
    int Nx, int Ny, int Nz, int Ng,
    BC bc_x_lo, BC bc_x_hi,
    BC bc_y_lo, BC bc_y_hi,
    BC bc_z_lo, BC bc_z_hi,
    double dirichlet_val);

/// Launch array copy kernel: dst = src
void launch_copy(
    cudaStream_t stream,
    double* dst, const double* src,
    size_t size);

/// Launch array zero kernel: dst = 0
void launch_zero(
    cudaStream_t stream,
    double* dst,
    size_t size);

// ============================================================================
// V-cycle Operation Kernels (for full V-cycle graphing)
// ============================================================================

/// Launch 3D residual computation kernel: r = f - L(u)
/// Operates on interior points only (ghost cells must be set)
void launch_residual_3d(
    cudaStream_t stream,
    const double* u, const double* f, double* r,
    int Nx, int Ny, int Nz, int Ng,
    double inv_dx2, double inv_dy2, double inv_dz2);

/// Launch 3D 27-point full-weighting restriction kernel
/// Restricts residual from fine to coarse grid
void launch_restrict_3d(
    cudaStream_t stream,
    const double* r_fine, double* f_coarse,
    int Nx_f, int Ny_f, int Nz_f,
    int Nx_c, int Ny_c, int Nz_c,
    int Ng);

/// Launch 3D trilinear prolongation kernel
/// Interpolates correction from coarse to fine and adds to fine solution
void launch_prolongate_3d(
    cudaStream_t stream,
    const double* u_coarse, double* u_fine,
    int Nx_f, int Ny_f, int Nz_f,
    int Nx_c, int Ny_c, int Nz_c,
    int Ng);

// ============================================================================
// Full V-cycle Graph (captures entire V-cycle as a single graph)
// ============================================================================

/// V-cycle level configuration for graph capture
struct VCycleLevelConfig {
    int Nx, Ny, Nz;           // Interior grid dimensions
    int Ng;                   // Number of ghost cells
    double inv_dx2, inv_dy2, inv_dz2;  // Inverse grid spacing squared
    double dx2, dy2, dz2;     // Grid spacing squared (for smoother)
    double coeff;             // Diagonal coefficient for Jacobi
    size_t total_size;        // Total array size including ghosts

    // Array pointers (device memory)
    double* u;                // Solution array
    double* f;                // RHS array
    double* r;                // Residual array
    double* tmp;              // Scratch buffer for smoother
};

/// Fingerprint for V-cycle graph validity checking
/// If any of these parameters change, the graph must be recaptured
struct VCycleGraphFingerprint {
    size_t num_levels = 0;
    std::vector<size_t> level_sizes;  // Total size per level
    std::vector<double> level_coeffs; // Diagonal coefficients per level
    int degree = 0;
    int nu1 = 0;
    int nu2 = 0;
    BC bc_x_lo = BC::Neumann, bc_x_hi = BC::Neumann;
    BC bc_y_lo = BC::Neumann, bc_y_hi = BC::Neumann;
    BC bc_z_lo = BC::Neumann, bc_z_hi = BC::Neumann;
    int coarse_iters = 8;  // Iterations at coarsest level

    bool operator==(const VCycleGraphFingerprint& other) const {
        return num_levels == other.num_levels &&
               level_sizes == other.level_sizes &&
               level_coeffs == other.level_coeffs &&
               degree == other.degree &&
               nu1 == other.nu1 && nu2 == other.nu2 &&
               bc_x_lo == other.bc_x_lo && bc_x_hi == other.bc_x_hi &&
               bc_y_lo == other.bc_y_lo && bc_y_hi == other.bc_y_hi &&
               bc_z_lo == other.bc_z_lo && bc_z_hi == other.bc_z_hi &&
               coarse_iters == other.coarse_iters;
    }
    bool operator!=(const VCycleGraphFingerprint& other) const {
        return !(*this == other);
    }
};

/// Full V-cycle CUDA Graph - captures entire V-cycle for single-launch execution
class CudaVCycleGraph {
public:
    CudaVCycleGraph() = default;
    ~CudaVCycleGraph();

    /// Initialize graph for the given level hierarchy
    /// @param levels Vector of level configurations (fine to coarse)
    /// @param degree Chebyshev polynomial degree
    /// @param nu1 Pre-smoothing iterations
    /// @param nu2 Post-smoothing iterations
    /// @param bc_* Boundary conditions for each direction
    void initialize(
        const std::vector<VCycleLevelConfig>& levels,
        int degree, int nu1, int nu2,
        BC bc_x_lo, BC bc_x_hi,
        BC bc_y_lo, BC bc_y_hi,
        BC bc_z_lo, BC bc_z_hi);

    /// Execute the captured V-cycle graph
    void execute(cudaStream_t stream);

    /// Check if graph is valid
    bool is_valid() const { return graph_exec_ != nullptr; }

    /// Check if graph needs recapture due to parameter changes
    bool needs_recapture(const VCycleGraphFingerprint& fp) const {
        return !is_valid() || fingerprint_ != fp;
    }

    /// Get current fingerprint
    const VCycleGraphFingerprint& fingerprint() const { return fingerprint_; }

    /// Destroy graph resources
    void destroy();

private:
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    std::vector<VCycleLevelConfig> levels_;
    VCycleGraphFingerprint fingerprint_;
    int degree_ = 4;
    int nu1_ = 2;
    int nu2_ = 2;
    BC bc_x_lo_, bc_x_hi_;
    BC bc_y_lo_, bc_y_hi_;
    BC bc_z_lo_, bc_z_hi_;
    bool all_periodic_ = false;

    /// Capture the full V-cycle into a graph
    void capture_vcycle_graph(cudaStream_t stream);

    /// Recursive V-cycle capture (called during graph construction)
    void capture_vcycle_level(cudaStream_t stream, int level);

    /// Capture smoother sequence for a level
    void capture_smoother(cudaStream_t stream, int level, int iterations);
};

} // namespace mg_cuda
} // namespace nncfd

#endif // USE_GPU_OFFLOAD
