/// @file mg_cuda_kernels.cpp
/// @brief CUDA kernel implementation for multigrid Poisson solver
///
/// This file implements CUDA-native kernels for the MG smoother,
/// with CUDA Graph capture support for reduced launch overhead.

#ifdef USE_GPU_OFFLOAD

#include "mg_cuda_kernels.hpp"
#include <cmath>
#include <stdexcept>
#include <iostream>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            throw std::runtime_error(std::string("CUDA error: ") +             \
                                     cudaGetErrorString(err) + " at " +        \
                                     __FILE__ + ":" + std::to_string(__LINE__)); \
        }                                                                      \
    } while (0)

namespace nncfd {
namespace mg_cuda {

// ============================================================================
// CUDA Kernels
// ============================================================================

/// 3D Chebyshev smoother kernel (requires ghost cells to be set)
/// Each thread computes one interior point
__global__ void chebyshev_3d_kernel(
    double* __restrict__ u,
    const double* __restrict__ f,
    double* __restrict__ tmp,
    int Nx, int Ny, int Nz, int Ng,
    double inv_dx2, double inv_dy2, double inv_dz2,
    double inv_coeff, double omega)
{
    // Global thread indices (interior points only)
    int i = blockIdx.x * blockDim.x + threadIdx.x + Ng;
    int j = blockIdx.y * blockDim.y + threadIdx.y + Ng;
    int k = blockIdx.z * blockDim.z + threadIdx.z + Ng;

    if (i < Nx + Ng && j < Ny + Ng && k < Nz + Ng) {
        int stride = Nx + 2 * Ng;
        int plane_stride = stride * (Ny + 2 * Ng);
        int idx = k * plane_stride + j * stride + i;

        // Read neighbors
        double u_c = u[idx];
        double u_xm = u[idx - 1];
        double u_xp = u[idx + 1];
        double u_ym = u[idx - stride];
        double u_yp = u[idx + stride];
        double u_zm = u[idx - plane_stride];
        double u_zp = u[idx + plane_stride];

        // Jacobi update: u_new = (neighbors/h^2 - f) / diag
        double u_jacobi = ((u_xp + u_xm) * inv_dx2 +
                           (u_yp + u_ym) * inv_dy2 +
                           (u_zp + u_zm) * inv_dz2 - f[idx]) * inv_coeff;

        // Chebyshev weighted update
        tmp[idx] = (1.0 - omega) * u_c + omega * u_jacobi;
    }
}

/// 3D Chebyshev smoother with FUSED periodic BCs (no separate BC kernel needed)
/// Uses wrap indexing for periodic boundaries - eliminates BC kernel overhead
__global__ void chebyshev_3d_periodic_kernel(
    double* __restrict__ u,
    const double* __restrict__ f,
    double* __restrict__ tmp,
    int Nx, int Ny, int Nz, int Ng,
    double inv_dx2, double inv_dy2, double inv_dz2,
    double inv_coeff, double omega)
{
    // Global thread indices (interior points: 1..Nx, 1..Ny, 1..Nz with Ng=1)
    int i = blockIdx.x * blockDim.x + threadIdx.x + Ng;
    int j = blockIdx.y * blockDim.y + threadIdx.y + Ng;
    int k = blockIdx.z * blockDim.z + threadIdx.z + Ng;

    if (i < Nx + Ng && j < Ny + Ng && k < Nz + Ng) {
        const int stride = Nx + 2 * Ng;
        const int plane_stride = stride * (Ny + 2 * Ng);
        const int idx = k * plane_stride + j * stride + i;

        // Wrap indices for periodic BCs (avoids separate BC kernel)
        // i ranges from Ng to Nx+Ng-1 (i.e., 1 to Nx for Ng=1)
        int i_m = (i == Ng) ? (Nx + Ng - 1) : (i - 1);       // Wrap left
        int i_p = (i == Nx + Ng - 1) ? Ng : (i + 1);         // Wrap right
        int j_m = (j == Ng) ? (Ny + Ng - 1) : (j - 1);       // Wrap bottom
        int j_p = (j == Ny + Ng - 1) ? Ng : (j + 1);         // Wrap top
        int k_m = (k == Ng) ? (Nz + Ng - 1) : (k - 1);       // Wrap back
        int k_p = (k == Nz + Ng - 1) ? Ng : (k + 1);         // Wrap front

        // Compute neighbor indices with wrap
        int idx_xm = k * plane_stride + j * stride + i_m;
        int idx_xp = k * plane_stride + j * stride + i_p;
        int idx_ym = k * plane_stride + j_m * stride + i;
        int idx_yp = k * plane_stride + j_p * stride + i;
        int idx_zm = k_m * plane_stride + j * stride + i;
        int idx_zp = k_p * plane_stride + j * stride + i;

        // Read neighbors with periodic wrap
        double u_c = u[idx];
        double u_xm = u[idx_xm];
        double u_xp = u[idx_xp];
        double u_ym = u[idx_ym];
        double u_yp = u[idx_yp];
        double u_zm = u[idx_zm];
        double u_zp = u[idx_zp];

        // Jacobi update: u_new = (neighbors/h^2 - f) / diag
        double u_jacobi = ((u_xp + u_xm) * inv_dx2 +
                           (u_yp + u_ym) * inv_dy2 +
                           (u_zp + u_zm) * inv_dz2 - f[idx]) * inv_coeff;

        // Chebyshev weighted update
        tmp[idx] = (1.0 - omega) * u_c + omega * u_jacobi;
    }
}

/// 3D boundary condition kernel
/// Handles periodic, Neumann, and Dirichlet BCs for ghost cells
__global__ void bc_3d_kernel(
    double* __restrict__ u,
    int Nx, int Ny, int Nz, int Ng,
    int bc_x_lo, int bc_x_hi,
    int bc_y_lo, int bc_y_hi,
    int bc_z_lo, int bc_z_hi,
    double dirichlet_val)
{
    int stride = Nx + 2 * Ng;
    int plane_stride = stride * (Ny + 2 * Ng);

    // Thread covers all boundary cells
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_boundary = (Ny + 2*Ng) * (Nz + 2*Ng) * 2 +  // x faces
                         (Nx + 2*Ng) * (Nz + 2*Ng) * 2 +  // y faces
                         (Nx + 2*Ng) * (Ny + 2*Ng) * 2;   // z faces

    if (tid >= total_boundary) return;

    // Decode which boundary face and which cell
    int remaining = tid;
    int face_yz = (Ny + 2*Ng) * (Nz + 2*Ng);
    int face_xz = (Nx + 2*Ng) * (Nz + 2*Ng);
    int face_xy = (Nx + 2*Ng) * (Ny + 2*Ng);

    // X-low face
    if (remaining < face_yz) {
        int jk = remaining;
        int j = jk % (Ny + 2*Ng);
        int k = jk / (Ny + 2*Ng);
        int idx = k * plane_stride + j * stride + 0;
        int idx_int = k * plane_stride + j * stride + Ng;
        int idx_wrap = k * plane_stride + j * stride + Nx;

        if (bc_x_lo == 2) { // Periodic
            u[idx] = u[idx_wrap];
        } else if (bc_x_lo == 1) { // Neumann
            u[idx] = u[idx_int];
        } else { // Dirichlet
            u[idx] = 2.0 * dirichlet_val - u[idx_int];
        }
        return;
    }
    remaining -= face_yz;

    // X-high face
    if (remaining < face_yz) {
        int jk = remaining;
        int j = jk % (Ny + 2*Ng);
        int k = jk / (Ny + 2*Ng);
        int idx = k * plane_stride + j * stride + (Nx + Ng);
        int idx_int = k * plane_stride + j * stride + (Nx + Ng - 1);
        int idx_wrap = k * plane_stride + j * stride + Ng;

        if (bc_x_hi == 2) { // Periodic
            u[idx] = u[idx_wrap];
        } else if (bc_x_hi == 1) { // Neumann
            u[idx] = u[idx_int];
        } else { // Dirichlet
            u[idx] = 2.0 * dirichlet_val - u[idx_int];
        }
        return;
    }
    remaining -= face_yz;

    // Y-low face
    if (remaining < face_xz) {
        int ik = remaining;
        int i = ik % (Nx + 2*Ng);
        int k = ik / (Nx + 2*Ng);
        int idx = k * plane_stride + 0 * stride + i;
        int idx_int = k * plane_stride + Ng * stride + i;
        int idx_wrap = k * plane_stride + Ny * stride + i;

        if (bc_y_lo == 2) { // Periodic
            u[idx] = u[idx_wrap];
        } else if (bc_y_lo == 1) { // Neumann
            u[idx] = u[idx_int];
        } else { // Dirichlet
            u[idx] = 2.0 * dirichlet_val - u[idx_int];
        }
        return;
    }
    remaining -= face_xz;

    // Y-high face
    if (remaining < face_xz) {
        int ik = remaining;
        int i = ik % (Nx + 2*Ng);
        int k = ik / (Nx + 2*Ng);
        int idx = k * plane_stride + (Ny + Ng) * stride + i;
        int idx_int = k * plane_stride + (Ny + Ng - 1) * stride + i;
        int idx_wrap = k * plane_stride + Ng * stride + i;

        if (bc_y_hi == 2) { // Periodic
            u[idx] = u[idx_wrap];
        } else if (bc_y_hi == 1) { // Neumann
            u[idx] = u[idx_int];
        } else { // Dirichlet
            u[idx] = 2.0 * dirichlet_val - u[idx_int];
        }
        return;
    }
    remaining -= face_xz;

    // Z-low face
    if (remaining < face_xy) {
        int ij = remaining;
        int i = ij % (Nx + 2*Ng);
        int j = ij / (Nx + 2*Ng);
        int idx = 0 * plane_stride + j * stride + i;
        int idx_int = Ng * plane_stride + j * stride + i;
        int idx_wrap = Nz * plane_stride + j * stride + i;

        if (bc_z_lo == 2) { // Periodic
            u[idx] = u[idx_wrap];
        } else if (bc_z_lo == 1) { // Neumann
            u[idx] = u[idx_int];
        } else { // Dirichlet
            u[idx] = 2.0 * dirichlet_val - u[idx_int];
        }
        return;
    }
    remaining -= face_xy;

    // Z-high face
    if (remaining < face_xy) {
        int ij = remaining;
        int i = ij % (Nx + 2*Ng);
        int j = ij / (Nx + 2*Ng);
        int idx = (Nz + Ng) * plane_stride + j * stride + i;
        int idx_int = (Nz + Ng - 1) * plane_stride + j * stride + i;
        int idx_wrap = Ng * plane_stride + j * stride + i;

        if (bc_z_hi == 2) { // Periodic
            u[idx] = u[idx_wrap];
        } else if (bc_z_hi == 1) { // Neumann
            u[idx] = u[idx_int];
        } else { // Dirichlet
            u[idx] = 2.0 * dirichlet_val - u[idx_int];
        }
        return;
    }
}

/// Simple array copy kernel
__global__ void copy_kernel(double* __restrict__ dst, const double* __restrict__ src, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = src[idx];
    }
}

// ============================================================================
// Kernel Launch Functions
// ============================================================================

void launch_chebyshev_3d(
    cudaStream_t stream,
    double* u, const double* f, double* tmp,
    int Nx, int Ny, int Nz, int Ng,
    double dx2, double dy2, double dz2, double coeff,
    double omega)
{
    // Compute inverse values for efficiency
    double inv_dx2 = 1.0 / dx2;
    double inv_dy2 = 1.0 / dy2;
    // For 2D (Nz=1), zero out z-direction contribution to stencil
    double inv_dz2 = (Nz == 1) ? 0.0 : 1.0 / dz2;
    double inv_coeff = 1.0 / coeff;

    // Thread block and grid dimensions
    dim3 block(8, 8, 8);  // 512 threads per block
    dim3 grid((Nx + block.x - 1) / block.x,
              (Ny + block.y - 1) / block.y,
              (Nz + block.z - 1) / block.z);

    chebyshev_3d_kernel<<<grid, block, 0, stream>>>(
        u, f, tmp, Nx, Ny, Nz, Ng,
        inv_dx2, inv_dy2, inv_dz2, inv_coeff, omega);
}

/// Launch Chebyshev kernel with fused periodic BCs (no separate BC kernel needed)
void launch_chebyshev_3d_periodic(
    cudaStream_t stream,
    double* u, const double* f, double* tmp,
    int Nx, int Ny, int Nz, int Ng,
    double dx2, double dy2, double dz2, double coeff,
    double omega)
{
    double inv_dx2 = 1.0 / dx2;
    double inv_dy2 = 1.0 / dy2;
    double inv_dz2 = (Nz == 1) ? 0.0 : 1.0 / dz2;
    double inv_coeff = 1.0 / coeff;

    dim3 block(8, 8, 8);
    dim3 grid((Nx + block.x - 1) / block.x,
              (Ny + block.y - 1) / block.y,
              (Nz + block.z - 1) / block.z);

    chebyshev_3d_periodic_kernel<<<grid, block, 0, stream>>>(
        u, f, tmp, Nx, Ny, Nz, Ng,
        inv_dx2, inv_dy2, inv_dz2, inv_coeff, omega);
}

void launch_bc_3d(
    cudaStream_t stream,
    double* u,
    int Nx, int Ny, int Nz, int Ng,
    BC bc_x_lo, BC bc_x_hi,
    BC bc_y_lo, BC bc_y_hi,
    BC bc_z_lo, BC bc_z_hi,
    double dirichlet_val)
{
    int total_boundary = (Ny + 2*Ng) * (Nz + 2*Ng) * 2 +
                         (Nx + 2*Ng) * (Nz + 2*Ng) * 2 +
                         (Nx + 2*Ng) * (Ny + 2*Ng) * 2;

    int block_size = 256;
    int grid_size = (total_boundary + block_size - 1) / block_size;

    bc_3d_kernel<<<grid_size, block_size, 0, stream>>>(
        u, Nx, Ny, Nz, Ng,
        static_cast<int>(bc_x_lo), static_cast<int>(bc_x_hi),
        static_cast<int>(bc_y_lo), static_cast<int>(bc_y_hi),
        static_cast<int>(bc_z_lo), static_cast<int>(bc_z_hi),
        dirichlet_val);
}

void launch_copy(cudaStream_t stream, double* dst, const double* src, size_t size) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    copy_kernel<<<grid_size, block_size, 0, stream>>>(dst, src, size);
}

// ============================================================================
// CudaSmootherGraph Implementation
// ============================================================================

CudaSmootherGraph::~CudaSmootherGraph() {
    destroy();
}

void CudaSmootherGraph::destroy() {
    if (graph_exec_) {
        cudaGraphExecDestroy(graph_exec_);
        graph_exec_ = nullptr;
    }
    if (graph_) {
        cudaGraphDestroy(graph_);
        graph_ = nullptr;
    }
}

void CudaSmootherGraph::initialize(const LevelConfig& config, int degree,
                                    BC bc_x_lo, BC bc_x_hi,
                                    BC bc_y_lo, BC bc_y_hi,
                                    BC bc_z_lo, BC bc_z_hi) {
    destroy();  // Clean up any existing graph

    config_ = config;
    degree_ = degree;
    bc_x_lo_ = bc_x_lo;
    bc_x_hi_ = bc_x_hi;
    bc_y_lo_ = bc_y_lo;
    bc_y_hi_ = bc_y_hi;
    bc_z_lo_ = bc_z_lo;
    bc_z_hi_ = bc_z_hi;

    // Create a temporary stream for graph capture
    cudaStream_t capture_stream;
    CUDA_CHECK(cudaStreamCreate(&capture_stream));

    capture_graph(capture_stream);

    CUDA_CHECK(cudaStreamDestroy(capture_stream));
}

void CudaSmootherGraph::capture_graph(cudaStream_t stream) {
    // Chebyshev eigenvalue bounds
    const double lambda_min = 0.05;
    const double lambda_max = 1.95;
    const double d = (lambda_max + lambda_min) / 2.0;
    const double c = (lambda_max - lambda_min) / 2.0;

    // Check if all BCs are periodic - can use fused kernel (no BC pass needed)
    const bool all_periodic = (bc_x_lo_ == BC::Periodic && bc_x_hi_ == BC::Periodic &&
                                bc_y_lo_ == BC::Periodic && bc_y_hi_ == BC::Periodic &&
                                bc_z_lo_ == BC::Periodic && bc_z_hi_ == BC::Periodic);

    // Begin stream capture
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    // Capture the Chebyshev smoother sequence
    for (int k = 0; k < degree_; ++k) {
        // Chebyshev-optimal weight
        double theta = M_PI * (2.0 * k + 1.0) / (2.0 * degree_);
        double omega = 1.0 / (d - c * std::cos(theta));

        if (all_periodic) {
            // Fused kernel: Chebyshev + periodic BC wrap (no separate BC pass)
            launch_chebyshev_periodic(stream, omega);
        } else {
            // Standard path: BC kernel + Chebyshev kernel
            launch_bc_kernel(stream);
            launch_chebyshev_iteration(stream, k, omega);
        }

        // Copy: tmp -> u
        launch_copy_kernel(stream);
    }

    // Final BC application (only needed for non-periodic)
    if (!all_periodic) {
        launch_bc_kernel(stream);
    }

    // End capture and create executable graph
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph_));
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0));
}

void CudaSmootherGraph::launch_chebyshev_iteration(cudaStream_t stream, int k, double omega) {
    launch_chebyshev_3d(
        stream,
        config_.u, config_.f, config_.tmp,
        config_.Nx, config_.Ny, config_.Nz, config_.Ng,
        config_.dx2, config_.dy2, config_.dz2, config_.coeff,
        omega);
}

void CudaSmootherGraph::launch_chebyshev_periodic(cudaStream_t stream, double omega) {
    launch_chebyshev_3d_periodic(
        stream,
        config_.u, config_.f, config_.tmp,
        config_.Nx, config_.Ny, config_.Nz, config_.Ng,
        config_.dx2, config_.dy2, config_.dz2, config_.coeff,
        omega);
}

void CudaSmootherGraph::launch_bc_kernel(cudaStream_t stream) {
    launch_bc_3d(
        stream,
        config_.u,
        config_.Nx, config_.Ny, config_.Nz, config_.Ng,
        bc_x_lo_, bc_x_hi_,
        bc_y_lo_, bc_y_hi_,
        bc_z_lo_, bc_z_hi_,
        0.0);  // Dirichlet value (0 for pressure)
}

void CudaSmootherGraph::launch_copy_kernel(cudaStream_t stream) {
    launch_copy(stream, config_.u, config_.tmp, config_.total_size);
}

void CudaSmootherGraph::execute(cudaStream_t stream) {
    if (graph_exec_) {
        CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream));
    }
}

// ============================================================================
// CudaMGContext Implementation
// ============================================================================

CudaMGContext::CudaMGContext() {
    CUDA_CHECK(cudaStreamCreate(&stream_));
}

CudaMGContext::~CudaMGContext() {
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

void CudaMGContext::synchronize() {
    if (stream_) {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
}

void CudaMGContext::initialize_smoother_graphs(
    const std::vector<LevelConfig>& levels,
    int degree,
    BC bc_x_lo, BC bc_x_hi,
    BC bc_y_lo, BC bc_y_hi,
    BC bc_z_lo, BC bc_z_hi)
{
    smoother_graphs_.resize(levels.size());
    for (size_t i = 0; i < levels.size(); ++i) {
        smoother_graphs_[i].initialize(
            levels[i], degree,
            bc_x_lo, bc_x_hi,
            bc_y_lo, bc_y_hi,
            bc_z_lo, bc_z_hi);
    }
}

void CudaMGContext::smooth(int level) {
    if (level >= 0 && level < static_cast<int>(smoother_graphs_.size())) {
        smoother_graphs_[level].execute(stream_);
    }
}

} // namespace mg_cuda
} // namespace nncfd

#endif // USE_GPU_OFFLOAD
