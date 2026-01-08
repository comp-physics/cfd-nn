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

    /// Execute smoother for a given level
    void smooth(int level);

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

} // namespace mg_cuda
} // namespace nncfd

#endif // USE_GPU_OFFLOAD
