# DNS/LES State-of-the-Art GPU Upgrade — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade CFD-NN to a multi-GPU DNS/LES platform with custom CUDA kernels, MPI domain decomposition, immersed boundary method, and 5 LES subgrid-scale models.

**Architecture:** Bottom-up GPU-first approach. Phase 1 optimizes the Poisson smoother bottleneck with custom CUDA kernels. Phase 2 adds MPI z-slab decomposition transparent to all physics. Phase 3 adds direct-forcing IBM for cylinder/airfoil flows. Phase 4 adds 5 LES SGS models with fused GPU kernels. Phase 5 adds HDF5 checkpoint/restart and a validation campaign.

**Tech Stack:** C++17, OpenMP target offload (NVHPC), custom CUDA kernels (.cu), cuFFT, cuSPARSE, cuBLAS, MPI, HDF5

---

## Phase 1: GPU Hot-Spot Optimization

### Task 1.1: CUDA Build Infrastructure

**Files:**
- Modify: `CMakeLists.txt`

**Step 1: Add CUDA language support to CMakeLists.txt**

After line 5 (`project(nn_cfd ...)`), add CUDA language enablement:

```cmake
# After project() declaration
if(USE_GPU_OFFLOAD)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES ${GPU_CC})
endif()
```

**Step 2: Add cuda_kernels source directory**

After the GPU-specific sources block (line 266), add:

```cmake
if(USE_GPU_OFFLOAD AND CMAKE_CUDA_COMPILER)
    file(GLOB CUDA_KERNEL_SOURCES "src/cuda_kernels/*.cu")
    target_sources(nn_cfd_core PRIVATE ${CUDA_KERNEL_SOURCES})
    target_link_libraries(nn_cfd_core CUDA::cublas)
    set_source_files_properties(${CUDA_KERNEL_SOURCES} PROPERTIES LANGUAGE CUDA)
endif()
```

**Step 3: Create the cuda_kernels directory**

```bash
mkdir -p src/cuda_kernels
```

**Step 4: Build to verify no regressions**

```bash
cd build && cmake .. -DCMAKE_CXX_COMPILER=nvc++ -DUSE_GPU_OFFLOAD=ON && make -j$(nproc)
```

Expected: Clean build, no new warnings.

**Step 5: Commit**

```bash
git add CMakeLists.txt src/cuda_kernels/
git commit -m "build: add CUDA language support and cuda_kernels directory"
```

---

### Task 1.2: Custom CUDA Chebyshev Smoother Kernel

**Files:**
- Create: `src/cuda_kernels/mg_smoother.cu`
- Create: `include/cuda_smoother.hpp`
- Modify: `src/mg_cuda_kernels.cpp` (add dispatch to CUDA kernel)
- Create: `tests/test_cuda_smoother.cpp`

**Step 1: Write the test**

Create `tests/test_cuda_smoother.cpp`:

```cpp
#include "poisson_solver_multigrid.hpp"
#include "mesh.hpp"
#include <cmath>
#include <vector>
#include <iostream>
#include <cassert>

// Test: CUDA smoother matches OpenMP target smoother to round-off
// Strategy: Run both on same input, compare output

using namespace nncfd;

int main() {
    // Small grid for fast test
    const int Nx = 32, Ny = 32, Nz = 32, Ng = 1;
    const double dx = 2.0 * M_PI / Nx;
    const double dy = 2.0 / Ny;
    const double dz = M_PI / Nz;

    Mesh mesh;
    mesh.init_uniform(Nx, Ny, Nz, 0.0, 2.0*M_PI, -1.0, 1.0, 0.0, M_PI, Ng);

    // Create MG solver with default config
    Config config;
    config.Nx = Nx; config.Ny = Ny; config.Nz = Nz;
    config.poisson_solver = PoissonSolverType::MG;

    MultigridPoissonSolver mg_solver;
    mg_solver.initialize(mesh, config);

    // Fill level 0 with sinusoidal RHS
    auto& level = mg_solver.level(0);
    const int stride = level.stride;
    const int plane_stride = level.plane_stride;
    double* u_data = level.u.data();
    double* f_data = level.f.data();

    for (int k = Ng; k < Nz + Ng; ++k) {
        for (int j = Ng; j < Ny + Ng; ++j) {
            for (int i = Ng; i < Nx + Ng; ++i) {
                int idx = k * plane_stride + j * stride + i;
                double x = (i - Ng + 0.5) * dx;
                double y = -1.0 + (j - Ng + 0.5) * dy;
                double z = (k - Ng + 0.5) * dz;
                f_data[idx] = sin(x) * cos(M_PI * y) * sin(z);
                u_data[idx] = 0.0;
            }
        }
    }

    // Copy initial state
    std::vector<double> u_omp(level.total_size);
    std::vector<double> u_cuda(level.total_size);
    std::copy(u_data, u_data + level.total_size, u_omp.data());
    std::copy(u_data, u_data + level.total_size, u_cuda.data());

    // Run OpenMP target smoother (4 Chebyshev iterations)
    mg_solver.smooth_chebyshev_omp(0, 4, u_omp.data());

    // Run CUDA smoother (4 Chebyshev iterations)
    mg_solver.smooth_chebyshev_cuda(0, 4, u_cuda.data());

    // Compare: should match to round-off (~1e-14)
    double max_diff = 0.0;
    for (int k = Ng; k < Nz + Ng; ++k) {
        for (int j = Ng; j < Ny + Ng; ++j) {
            for (int i = Ng; i < Nx + Ng; ++i) {
                int idx = k * plane_stride + j * stride + i;
                double diff = std::abs(u_omp[idx] - u_cuda[idx]);
                max_diff = std::max(max_diff, diff);
            }
        }
    }

    std::cout << "Max difference (OMP vs CUDA smoother): " << max_diff << std::endl;
    assert(max_diff < 1e-12 && "CUDA smoother must match OMP target to round-off");
    std::cout << "PASS: CUDA smoother matches OMP target" << std::endl;
    return 0;
}
```

Register in CMakeLists.txt (test section):

```cmake
if(USE_GPU_OFFLOAD)
    add_nncfd_test(test_cuda_smoother tests/test_cuda_smoother.cpp "gpu;fast")
endif()
```

**Step 2: Run test to verify it fails**

```bash
cd build && cmake .. -DUSE_GPU_OFFLOAD=ON && make -j$(nproc) 2>&1 | tail -5
```

Expected: Compilation failure — `smooth_chebyshev_omp` and `smooth_chebyshev_cuda` don't exist yet.

**Step 3: Create the CUDA smoother header**

Create `include/cuda_smoother.hpp`:

```cpp
#pragma once

namespace nncfd {
namespace cuda_kernels {

/// Launch optimized CUDA Chebyshev smoother with shared memory tiling
/// @param u       Solution array (device pointer)
/// @param f       RHS array (device pointer)
/// @param tmp     Temporary array (device pointer)
/// @param Nx,Ny,Nz Interior dimensions
/// @param Ng      Ghost cell width
/// @param inv_dx2,inv_dy2,inv_dz2 Inverse squared spacings
/// @param degree  Number of Chebyshev iterations
/// @param lambda_min,lambda_max Chebyshev eigenvalue bounds
/// @param bc_periodic_x,bc_periodic_y,bc_periodic_z BC flags
/// @param stream  CUDA stream
void launch_chebyshev_3d_smem(
    double* u, const double* f, double* tmp,
    int Nx, int Ny, int Nz, int Ng,
    double inv_dx2, double inv_dy2, double inv_dz2,
    int degree, double lambda_min, double lambda_max,
    bool bc_periodic_x, bool bc_periodic_y, bool bc_periodic_z,
    void* stream = nullptr);

/// Launch optimized CUDA Chebyshev smoother for non-uniform y grids
void launch_chebyshev_3d_smem_nonuniform(
    double* u, const double* f, double* tmp,
    int Nx, int Ny, int Nz, int Ng,
    double inv_dx2, double inv_dz2,
    const double* aS, const double* aN, const double* aP,
    int degree, double lambda_min, double lambda_max,
    bool bc_periodic_x, bool bc_periodic_y, bool bc_periodic_z,
    void* stream = nullptr);

/// Check if CUDA smoother is available at runtime
bool cuda_smoother_available();

} // namespace cuda_kernels
} // namespace nncfd
```

**Step 4: Implement the CUDA smoother kernel**

Create `src/cuda_kernels/mg_smoother.cu`:

```cuda
#include "cuda_smoother.hpp"
#include <cuda_runtime.h>
#include <cstdio>

namespace nncfd {
namespace cuda_kernels {

// Tile dimensions for shared memory
static constexpr int TILE_X = 8;
static constexpr int TILE_Y = 8;
static constexpr int TILE_Z = 8;
// Shared memory tile includes 1-cell halo on each side
static constexpr int SMEM_X = TILE_X + 2;
static constexpr int SMEM_Y = TILE_Y + 2;
static constexpr int SMEM_Z = TILE_Z + 2;

__global__ void chebyshev_3d_smem_kernel(
    double* __restrict__ u,
    const double* __restrict__ f,
    double* __restrict__ tmp,
    int Nx, int Ny, int Nz, int Ng,
    double inv_dx2, double inv_dy2, double inv_dz2,
    double inv_diag, double omega)
{
    // Global indices (interior only)
    const int i = blockIdx.x * TILE_X + threadIdx.x + Ng;
    const int j = blockIdx.y * TILE_Y + threadIdx.y + Ng;
    const int k = blockIdx.z * TILE_Z + threadIdx.z + Ng;

    const int stride = Nx + 2 * Ng;
    const int plane_stride = stride * (Ny + 2 * Ng);

    // Shared memory tile
    __shared__ double s_u[SMEM_Z][SMEM_Y][SMEM_X];

    // Local thread indices in shared memory (offset by 1 for halo)
    const int si = threadIdx.x + 1;
    const int sj = threadIdx.y + 1;
    const int sk = threadIdx.z + 1;

    // Load center point
    bool in_bounds = (i < Nx + Ng && j < Ny + Ng && k < Nz + Ng);
    int idx = k * plane_stride + j * stride + i;

    if (in_bounds) {
        s_u[sk][sj][si] = u[idx];
    }

    // Load halo cells (face neighbors only)
    // X-halos
    if (threadIdx.x == 0 && i > 0) {
        s_u[sk][sj][0] = u[idx - 1];
    }
    if (threadIdx.x == TILE_X - 1 || i == Nx + Ng - 1) {
        if (i + 1 < Nx + 2 * Ng) {
            s_u[sk][sj][si + 1] = u[idx + 1];
        }
    }
    // Y-halos
    if (threadIdx.y == 0 && j > 0) {
        s_u[sk][0][si] = u[idx - stride];
    }
    if (threadIdx.y == TILE_Y - 1 || j == Ny + Ng - 1) {
        if (j + 1 < Ny + 2 * Ng) {
            s_u[sk][sj + 1][si] = u[idx + stride];
        }
    }
    // Z-halos
    if (threadIdx.z == 0 && k > 0) {
        s_u[0][sj][si] = u[idx - plane_stride];
    }
    if (threadIdx.z == TILE_Z - 1 || k == Nz + Ng - 1) {
        if (k + 1 < Nz + 2 * Ng) {
            s_u[sk + 1][sj][si] = u[idx + plane_stride];
        }
    }

    __syncthreads();

    if (in_bounds) {
        double lap = (s_u[sk][sj][si - 1] + s_u[sk][sj][si + 1]) * inv_dx2
                   + (s_u[sk][sj - 1][si] + s_u[sk][sj + 1][si]) * inv_dy2
                   + (s_u[sk - 1][sj][si] + s_u[sk + 1][sj][si]) * inv_dz2;

        double u_jacobi = (lap - f[idx]) * inv_diag;
        tmp[idx] = (1.0 - omega) * s_u[sk][sj][si] + omega * u_jacobi;
    }
}

__global__ void chebyshev_3d_smem_nonuniform_kernel(
    double* __restrict__ u,
    const double* __restrict__ f,
    double* __restrict__ tmp,
    int Nx, int Ny, int Nz, int Ng,
    double inv_dx2, double inv_dz2,
    const double* __restrict__ aS,
    const double* __restrict__ aN,
    const double* __restrict__ aP,
    double omega)
{
    const int i = blockIdx.x * TILE_X + threadIdx.x + Ng;
    const int j = blockIdx.y * TILE_Y + threadIdx.y + Ng;
    const int k = blockIdx.z * TILE_Z + threadIdx.z + Ng;

    const int stride = Nx + 2 * Ng;
    const int plane_stride = stride * (Ny + 2 * Ng);

    __shared__ double s_u[SMEM_Z][SMEM_Y][SMEM_X];

    const int si = threadIdx.x + 1;
    const int sj = threadIdx.y + 1;
    const int sk = threadIdx.z + 1;

    bool in_bounds = (i < Nx + Ng && j < Ny + Ng && k < Nz + Ng);
    int idx = k * plane_stride + j * stride + i;

    if (in_bounds) {
        s_u[sk][sj][si] = u[idx];
    }

    // Same halo loading as uniform kernel
    if (threadIdx.x == 0 && i > 0)
        s_u[sk][sj][0] = u[idx - 1];
    if (threadIdx.x == TILE_X - 1 || i == Nx + Ng - 1)
        if (i + 1 < Nx + 2 * Ng)
            s_u[sk][sj][si + 1] = u[idx + 1];
    if (threadIdx.y == 0 && j > 0)
        s_u[sk][0][si] = u[idx - stride];
    if (threadIdx.y == TILE_Y - 1 || j == Ny + Ng - 1)
        if (j + 1 < Ny + 2 * Ng)
            s_u[sk][sj + 1][si] = u[idx + stride];
    if (threadIdx.z == 0 && k > 0)
        s_u[0][sj][si] = u[idx - plane_stride];
    if (threadIdx.z == TILE_Z - 1 || k == Nz + Ng - 1)
        if (k + 1 < Nz + 2 * Ng)
            s_u[sk + 1][sj][si] = u[idx + plane_stride];

    __syncthreads();

    if (in_bounds) {
        double lap_xz = (s_u[sk][sj][si - 1] + s_u[sk][sj][si + 1]) * inv_dx2
                       + (s_u[sk - 1][sj][si] + s_u[sk + 1][sj][si]) * inv_dz2;

        double lap_y = aS[j] * s_u[sk][sj - 1][si] + aN[j] * s_u[sk][sj + 1][si];

        double inv_diag = -1.0 / (2.0 * inv_dx2 + aP[j] + 2.0 * inv_dz2);
        double u_jacobi = (lap_xz + lap_y - f[idx]) * inv_diag;
        tmp[idx] = (1.0 - omega) * s_u[sk][sj][si] + omega * u_jacobi;
    }
}

__global__ void copy_kernel(double* __restrict__ dst,
                            const double* __restrict__ src,
                            int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        dst[idx] = src[idx];
    }
}

// Host-side launch functions

void launch_chebyshev_3d_smem(
    double* u, const double* f, double* tmp,
    int Nx, int Ny, int Nz, int Ng,
    double inv_dx2, double inv_dy2, double inv_dz2,
    int degree, double lambda_min, double lambda_max,
    bool bc_periodic_x, bool bc_periodic_y, bool bc_periodic_z,
    void* stream)
{
    cudaStream_t s = stream ? static_cast<cudaStream_t>(stream) : 0;

    dim3 block(TILE_X, TILE_Y, TILE_Z);
    dim3 grid((Nx + TILE_X - 1) / TILE_X,
              (Ny + TILE_Y - 1) / TILE_Y,
              (Nz + TILE_Z - 1) / TILE_Z);

    const double diag = -(2.0 * inv_dx2 + 2.0 * inv_dy2 + 2.0 * inv_dz2);
    const double inv_diag = 1.0 / diag;

    const int total_size = (Nx + 2*Ng) * (Ny + 2*Ng) * (Nz + 2*Ng);

    for (int iter = 0; iter < degree; ++iter) {
        // Chebyshev weight for this iteration
        double theta = M_PI * (2.0 * iter + 1.0) / (2.0 * degree);
        double sigma = (lambda_max + lambda_min) / 2.0
                     + (lambda_max - lambda_min) / 2.0 * cos(theta);
        double omega = 1.0 / sigma;

        chebyshev_3d_smem_kernel<<<grid, block, 0, s>>>(
            u, f, tmp, Nx, Ny, Nz, Ng,
            inv_dx2, inv_dy2, inv_dz2,
            inv_diag, omega);

        // Copy tmp -> u for next iteration
        int copy_blocks = (total_size + 255) / 256;
        copy_kernel<<<copy_blocks, 256, 0, s>>>(u, tmp, total_size);
    }
}

void launch_chebyshev_3d_smem_nonuniform(
    double* u, const double* f, double* tmp,
    int Nx, int Ny, int Nz, int Ng,
    double inv_dx2, double inv_dz2,
    const double* aS, const double* aN, const double* aP,
    int degree, double lambda_min, double lambda_max,
    bool bc_periodic_x, bool bc_periodic_y, bool bc_periodic_z,
    void* stream)
{
    cudaStream_t s = stream ? static_cast<cudaStream_t>(stream) : 0;

    dim3 block(TILE_X, TILE_Y, TILE_Z);
    dim3 grid((Nx + TILE_X - 1) / TILE_X,
              (Ny + TILE_Y - 1) / TILE_Y,
              (Nz + TILE_Z - 1) / TILE_Z);

    const int total_size = (Nx + 2*Ng) * (Ny + 2*Ng) * (Nz + 2*Ng);

    for (int iter = 0; iter < degree; ++iter) {
        double theta = M_PI * (2.0 * iter + 1.0) / (2.0 * degree);
        double sigma = (lambda_max + lambda_min) / 2.0
                     + (lambda_max - lambda_min) / 2.0 * cos(theta);
        double omega = 1.0 / sigma;

        chebyshev_3d_smem_nonuniform_kernel<<<grid, block, 0, s>>>(
            u, f, tmp, Nx, Ny, Nz, Ng,
            inv_dx2, inv_dz2,
            aS, aN, aP, omega);

        int copy_blocks = (total_size + 255) / 256;
        copy_kernel<<<copy_blocks, 256, 0, s>>>(u, tmp, total_size);
    }
}

bool cuda_smoother_available() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}

} // namespace cuda_kernels
} // namespace nncfd
```

**Step 5: Add dispatch logic in MG solver**

Modify `src/mg_cuda_kernels.cpp` — in the `launch_chebyshev_3d` function, add CUDA smoother dispatch at the top:

```cpp
#include "cuda_smoother.hpp"

// In launch_chebyshev_3d(), before the OpenMP target path:
if (cuda_kernels::cuda_smoother_available() && use_cuda_smoother_) {
    cuda_kernels::launch_chebyshev_3d_smem(
        u_ptr, f_ptr, tmp_ptr,
        Nx, Ny, Nz, Ng,
        inv_dx2, inv_dy2, inv_dz2,
        degree, lambda_min_, lambda_max_,
        bc_periodic_x_, bc_periodic_y_, bc_periodic_z_,
        stream_);
    return;
}
// ... existing OpenMP target code follows as fallback ...
```

**Step 6: Build and run test**

```bash
cd build && cmake .. -DUSE_GPU_OFFLOAD=ON && make -j$(nproc)
OMP_TARGET_OFFLOAD=MANDATORY ctest -R test_cuda_smoother --output-on-failure
```

Expected: PASS — CUDA smoother matches OMP target to <1e-12.

**Step 7: Commit**

```bash
git add include/cuda_smoother.hpp src/cuda_kernels/mg_smoother.cu tests/test_cuda_smoother.cpp
git add -p src/mg_cuda_kernels.cpp CMakeLists.txt
git commit -m "feat: add custom CUDA Chebyshev smoother with shared memory tiling"
```

---

### Task 1.3: cuBLAS Reductions

**Files:**
- Create: `src/cuda_kernels/reductions.cu`
- Create: `include/cuda_reductions.hpp`
- Modify: `src/poisson_solver_multigrid.cpp` (residual norm computation)

**Step 1: Write the test**

Add to `tests/test_cuda_smoother.cpp` (extend existing test file):

```cpp
// Test: cuBLAS norm matches manual reduction
void test_cublas_norm() {
    const int N = 100000;
    std::vector<double> h_data(N);
    for (int i = 0; i < N; ++i) h_data[i] = sin(0.01 * i);

    double expected = 0.0;
    for (int i = 0; i < N; ++i) expected += h_data[i] * h_data[i];
    expected = sqrt(expected);

    double* d_data;
    cudaMalloc(&d_data, N * sizeof(double));
    cudaMemcpy(d_data, h_data.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    double result = nncfd::cuda_kernels::device_norm_l2(d_data, N);
    cudaFree(d_data);

    double rel_err = std::abs(result - expected) / expected;
    std::cout << "cuBLAS norm rel error: " << rel_err << std::endl;
    assert(rel_err < 1e-14 && "cuBLAS norm must match CPU reference");
    std::cout << "PASS: cuBLAS reduction" << std::endl;
}
```

**Step 2: Implement cuBLAS wrappers**

Create `include/cuda_reductions.hpp`:

```cpp
#pragma once

namespace nncfd {
namespace cuda_kernels {

/// L2 norm via cuBLAS (cublasDnrm2)
double device_norm_l2(const double* d_data, int n);

/// Max absolute value via cuBLAS (cublasIdamax)
double device_norm_linf(const double* d_data, int n);

/// Sum via cuBLAS (cublasDasum)
double device_sum_abs(const double* d_data, int n);

/// Initialize/finalize cuBLAS handle (call once at solver init/cleanup)
void init_cublas();
void finalize_cublas();

} // namespace cuda_kernels
} // namespace nncfd
```

Create `src/cuda_kernels/reductions.cu`:

```cuda
#include "cuda_reductions.hpp"
#include <cublas_v2.h>
#include <stdexcept>
#include <cmath>

namespace nncfd {
namespace cuda_kernels {

static cublasHandle_t g_cublas_handle = nullptr;

void init_cublas() {
    if (!g_cublas_handle) {
        cublasStatus_t status = cublasCreate(&g_cublas_handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cuBLAS initialization failed");
        }
    }
}

void finalize_cublas() {
    if (g_cublas_handle) {
        cublasDestroy(g_cublas_handle);
        g_cublas_handle = nullptr;
    }
}

double device_norm_l2(const double* d_data, int n) {
    if (!g_cublas_handle) init_cublas();
    double result = 0.0;
    cublasStatus_t status = cublasDnrm2(g_cublas_handle, n, d_data, 1, &result);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasDnrm2 failed");
    }
    return result;
}

double device_norm_linf(const double* d_data, int n) {
    if (!g_cublas_handle) init_cublas();
    int idx = 0;
    cublasStatus_t status = cublasIdamax(g_cublas_handle, n, d_data, 1, &idx);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasIdamax failed");
    }
    // cuBLAS returns 1-based index; read value from device
    double result = 0.0;
    cudaMemcpy(&result, d_data + idx - 1, sizeof(double), cudaMemcpyDeviceToHost);
    return std::abs(result);
}

double device_sum_abs(const double* d_data, int n) {
    if (!g_cublas_handle) init_cublas();
    double result = 0.0;
    cublasStatus_t status = cublasDasum(g_cublas_handle, n, d_data, 1, &result);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasDasum failed");
    }
    return result;
}

} // namespace cuda_kernels
} // namespace nncfd
```

**Step 3: Wire into MG residual computation**

In `src/poisson_solver_multigrid.cpp`, in the residual norm computation (convergence check), add cuBLAS path:

```cpp
#include "cuda_reductions.hpp"

// Replace OpenMP target reduction for r_norm_sq:
#ifdef USE_GPU_OFFLOAD
if (cuda_kernels::cuda_smoother_available()) {
    double* r_dev = static_cast<double*>(omp_get_mapped_ptr(r_ptrs_[0], device));
    r_norm = cuda_kernels::device_norm_l2(r_dev, interior_size);
    r_norm_sq = r_norm * r_norm;
} else
#endif
{
    // existing OpenMP target reduction fallback
}
```

**Step 4: Build and test**

```bash
cd build && cmake .. -DUSE_GPU_OFFLOAD=ON && make -j$(nproc)
OMP_TARGET_OFFLOAD=MANDATORY ctest -R test_cuda_smoother --output-on-failure
```

**Step 5: Commit**

```bash
git add include/cuda_reductions.hpp src/cuda_kernels/reductions.cu
git add -p src/poisson_solver_multigrid.cpp tests/test_cuda_smoother.cpp
git commit -m "feat: add cuBLAS reductions for MG residual computation"
```

---

### Task 1.4: Fused Halo/BC CUDA Kernel

**Files:**
- Create: `src/cuda_kernels/halo_pack.cu`
- Create: `include/cuda_halo.hpp`
- Modify: `src/mg_cuda_kernels.cpp` (replace 6-kernel BC with fused kernel)

**Step 1: Write the header**

Create `include/cuda_halo.hpp`:

```cpp
#pragma once

namespace nncfd {
namespace cuda_kernels {

/// Fused boundary condition application for all 6 faces
/// Replaces 6 separate kernel launches with 1
void launch_apply_bc_3d_fused(
    double* u,
    int Nx, int Ny, int Nz, int Ng,
    int bc_x_lo, int bc_x_hi,  // 0=periodic, 1=neumann, 2=dirichlet
    int bc_y_lo, int bc_y_hi,
    int bc_z_lo, int bc_z_hi,
    void* stream = nullptr);

/// Pack z-face data into contiguous buffer (for MPI halo exchange)
void launch_pack_z_face(
    const double* field,
    double* buffer,
    int Nx, int Ny, int Nz, int Ng,
    bool pack_lo,  // true = pack low-z face, false = pack high-z face
    void* stream = nullptr);

/// Unpack z-face data from contiguous buffer
void launch_unpack_z_face(
    double* field,
    const double* buffer,
    int Nx, int Ny, int Nz, int Ng,
    bool unpack_lo,
    void* stream = nullptr);

} // namespace cuda_kernels
} // namespace nncfd
```

**Step 2: Implement the fused BC kernel**

Create `src/cuda_kernels/halo_pack.cu`:

```cuda
#include "cuda_halo.hpp"
#include <cuda_runtime.h>

namespace nncfd {
namespace cuda_kernels {

__global__ void apply_bc_3d_fused_kernel(
    double* __restrict__ u,
    int Nx, int Ny, int Nz, int Ng,
    int bc_x_lo, int bc_x_hi,
    int bc_y_lo, int bc_y_hi,
    int bc_z_lo, int bc_z_hi)
{
    const int stride = Nx + 2 * Ng;
    const int plane_stride = stride * (Ny + 2 * Ng);
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Total boundary cells: 2 faces per direction
    const int yz_face = (Ny + 2*Ng) * (Nz + 2*Ng);  // x-faces
    const int xz_face = (Nx + 2*Ng) * (Nz + 2*Ng);  // y-faces
    const int xy_face = (Nx + 2*Ng) * (Ny + 2*Ng);  // z-faces
    const int total = 2 * yz_face + 2 * xz_face + 2 * xy_face;

    if (tid >= total) return;

    int remaining = tid;

    // X-lo face
    if (remaining < yz_face) {
        int j = remaining / (Nz + 2*Ng);
        int k = remaining % (Nz + 2*Ng);
        for (int g = 0; g < Ng; ++g) {
            int idx_ghost = k * plane_stride + j * stride + g;
            if (bc_x_lo == 0) { // periodic
                int idx_src = k * plane_stride + j * stride + (Nx + g);
                u[idx_ghost] = u[idx_src];
            } else if (bc_x_lo == 1) { // neumann
                int idx_src = k * plane_stride + j * stride + Ng;
                u[idx_ghost] = u[idx_src];
            }
        }
        return;
    }
    remaining -= yz_face;

    // X-hi face
    if (remaining < yz_face) {
        int j = remaining / (Nz + 2*Ng);
        int k = remaining % (Nz + 2*Ng);
        for (int g = 0; g < Ng; ++g) {
            int idx_ghost = k * plane_stride + j * stride + (Nx + Ng + g);
            if (bc_x_hi == 0) { // periodic
                int idx_src = k * plane_stride + j * stride + (Ng + g);
                u[idx_ghost] = u[idx_src];
            } else if (bc_x_hi == 1) { // neumann
                int idx_src = k * plane_stride + j * stride + (Nx + Ng - 1);
                u[idx_ghost] = u[idx_src];
            }
        }
        return;
    }
    remaining -= yz_face;

    // Y-lo face
    if (remaining < xz_face) {
        int i = remaining / (Nz + 2*Ng);
        int k = remaining % (Nz + 2*Ng);
        for (int g = 0; g < Ng; ++g) {
            int idx_ghost = k * plane_stride + g * stride + i;
            if (bc_y_lo == 0) {
                int idx_src = k * plane_stride + (Ny + g) * stride + i;
                u[idx_ghost] = u[idx_src];
            } else if (bc_y_lo == 1) {
                int idx_src = k * plane_stride + Ng * stride + i;
                u[idx_ghost] = u[idx_src];
            }
        }
        return;
    }
    remaining -= xz_face;

    // Y-hi face
    if (remaining < xz_face) {
        int i = remaining / (Nz + 2*Ng);
        int k = remaining % (Nz + 2*Ng);
        for (int g = 0; g < Ng; ++g) {
            int idx_ghost = k * plane_stride + (Ny + Ng + g) * stride + i;
            if (bc_y_hi == 0) {
                int idx_src = k * plane_stride + (Ng + g) * stride + i;
                u[idx_ghost] = u[idx_src];
            } else if (bc_y_hi == 1) {
                int idx_src = k * plane_stride + (Ny + Ng - 1) * stride + i;
                u[idx_ghost] = u[idx_src];
            }
        }
        return;
    }
    remaining -= xz_face;

    // Z-lo face
    if (remaining < xy_face) {
        int i = remaining / (Ny + 2*Ng);
        int j = remaining % (Ny + 2*Ng);
        for (int g = 0; g < Ng; ++g) {
            int idx_ghost = g * plane_stride + j * stride + i;
            if (bc_z_lo == 0) {
                int idx_src = (Nz + g) * plane_stride + j * stride + i;
                u[idx_ghost] = u[idx_src];
            } else if (bc_z_lo == 1) {
                int idx_src = Ng * plane_stride + j * stride + i;
                u[idx_ghost] = u[idx_src];
            }
        }
        return;
    }
    remaining -= xy_face;

    // Z-hi face
    if (remaining < xy_face) {
        int i = remaining / (Ny + 2*Ng);
        int j = remaining % (Ny + 2*Ng);
        for (int g = 0; g < Ng; ++g) {
            int idx_ghost = (Nz + Ng + g) * plane_stride + j * stride + i;
            if (bc_z_hi == 0) {
                int idx_src = (Ng + g) * plane_stride + j * stride + i;
                u[idx_ghost] = u[idx_src];
            } else if (bc_z_hi == 1) {
                int idx_src = (Nz + Ng - 1) * plane_stride + j * stride + i;
                u[idx_ghost] = u[idx_src];
            }
        }
    }
}

__global__ void pack_z_face_kernel(
    const double* __restrict__ field,
    double* __restrict__ buffer,
    int Nx, int Ny, int Ng, int stride, int plane_stride,
    int k_src)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < Nx + 2*Ng && j < Ny + 2*Ng) {
        buffer[j * (Nx + 2*Ng) + i] = field[k_src * plane_stride + j * stride + i];
    }
}

__global__ void unpack_z_face_kernel(
    double* __restrict__ field,
    const double* __restrict__ buffer,
    int Nx, int Ny, int Ng, int stride, int plane_stride,
    int k_dst)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < Nx + 2*Ng && j < Ny + 2*Ng) {
        field[k_dst * plane_stride + j * stride + i] = buffer[j * (Nx + 2*Ng) + i];
    }
}

void launch_apply_bc_3d_fused(
    double* u, int Nx, int Ny, int Nz, int Ng,
    int bc_x_lo, int bc_x_hi,
    int bc_y_lo, int bc_y_hi,
    int bc_z_lo, int bc_z_hi,
    void* stream)
{
    cudaStream_t s = stream ? static_cast<cudaStream_t>(stream) : 0;
    const int yz = (Ny + 2*Ng) * (Nz + 2*Ng);
    const int xz = (Nx + 2*Ng) * (Nz + 2*Ng);
    const int xy = (Nx + 2*Ng) * (Ny + 2*Ng);
    const int total = 2*yz + 2*xz + 2*xy;

    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    apply_bc_3d_fused_kernel<<<blocks, threads, 0, s>>>(
        u, Nx, Ny, Nz, Ng,
        bc_x_lo, bc_x_hi, bc_y_lo, bc_y_hi, bc_z_lo, bc_z_hi);
}

void launch_pack_z_face(
    const double* field, double* buffer,
    int Nx, int Ny, int Nz, int Ng,
    bool pack_lo, void* stream)
{
    cudaStream_t s = stream ? static_cast<cudaStream_t>(stream) : 0;
    const int stride = Nx + 2*Ng;
    const int plane_stride = stride * (Ny + 2*Ng);
    int k_src = pack_lo ? Ng : (Nz + Ng - 1);

    dim3 block(16, 16);
    dim3 grid((Nx + 2*Ng + 15)/16, (Ny + 2*Ng + 15)/16);
    pack_z_face_kernel<<<grid, block, 0, s>>>(
        field, buffer, Nx, Ny, Ng, stride, plane_stride, k_src);
}

void launch_unpack_z_face(
    double* field, const double* buffer,
    int Nx, int Ny, int Nz, int Ng,
    bool unpack_lo, void* stream)
{
    cudaStream_t s = stream ? static_cast<cudaStream_t>(stream) : 0;
    const int stride = Nx + 2*Ng;
    const int plane_stride = stride * (Ny + 2*Ng);
    int k_dst = unpack_lo ? 0 : (Nz + 2*Ng - 1);

    dim3 block(16, 16);
    dim3 grid((Nx + 2*Ng + 15)/16, (Ny + 2*Ng + 15)/16);
    unpack_z_face_kernel<<<grid, block, 0, s>>>(
        field, buffer, Nx, Ny, Ng, stride, plane_stride, k_dst);
}

} // namespace cuda_kernels
} // namespace nncfd
```

**Step 3: Build and run existing tests (regression)**

```bash
cd build && cmake .. -DUSE_GPU_OFFLOAD=ON && make -j$(nproc)
OMP_TARGET_OFFLOAD=MANDATORY ctest -L fast --output-on-failure
```

Expected: All existing tests still pass.

**Step 4: Commit**

```bash
git add include/cuda_halo.hpp src/cuda_kernels/halo_pack.cu
git commit -m "feat: add fused BC kernel and z-face pack/unpack for MPI halo exchange"
```

---

## Phase 2: MPI Domain Decomposition

### Task 2.1: Decomposition Class

**Files:**
- Create: `include/decomposition.hpp`
- Create: `src/decomposition.cpp`
- Modify: `CMakeLists.txt` (add MPI find, source file)

**Step 1: Write the test**

Create `tests/test_decomposition.cpp`:

```cpp
#include "decomposition.hpp"
#include <cassert>
#include <iostream>

using namespace nncfd;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Test: 192 z-cells across nprocs ranks
    const int Nz_global = 192;
    Decomposition decomp(MPI_COMM_WORLD, Nz_global);

    // Each rank gets Nz_global/nprocs cells (± 1 for remainder)
    assert(decomp.nz_local() > 0);
    assert(decomp.nz_local() <= Nz_global);

    // Sum of all local Nz must equal global
    int nz_sum = 0;
    MPI_Allreduce(&decomp.nz_local(), &nz_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    assert(nz_sum == Nz_global);

    // Neighbor ranks are correct
    if (nprocs > 1) {
        assert(decomp.rank_lo() >= 0);
        assert(decomp.rank_hi() >= 0);
        // Periodic: rank 0's lo neighbor is nprocs-1
        if (rank == 0) assert(decomp.rank_lo() == nprocs - 1);
        if (rank == nprocs - 1) assert(decomp.rank_hi() == 0);
    }

    // Global z-offset is consistent
    int k_start = decomp.k_global_start();
    assert(k_start >= 0 && k_start < Nz_global);

    if (rank == 0) {
        std::cout << "PASS: Decomposition test with " << nprocs << " ranks" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
```

Register in CMakeLists.txt:

```cmake
if(MPI_FOUND)
    add_nncfd_test(test_decomposition tests/test_decomposition.cpp "fast;mpi")
    # Run with 2 ranks
    set_tests_properties(test_decomposition PROPERTIES
        PROCESSORS 2
        ENVIRONMENT "MPIEXEC_PREFLAGS=-np;2")
endif()
```

**Step 2: Implement Decomposition class**

Create `include/decomposition.hpp`:

```cpp
#pragma once

#include <mpi.h>
#include <vector>

namespace nncfd {

/// 1D slab decomposition in z-direction for multi-GPU DNS
class Decomposition {
public:
    /// Construct with communicator and global z-cell count
    /// For single-process: pass MPI_COMM_SELF or nprocs=1
    Decomposition(MPI_Comm comm, int Nz_global);

    /// Trivial single-process decomposition (no MPI)
    explicit Decomposition(int Nz_global);

    // Accessors
    int rank() const { return rank_; }
    int nprocs() const { return nprocs_; }
    int nz_local() const { return nz_local_; }
    int nz_global() const { return nz_global_; }
    int k_global_start() const { return k_global_start_; }
    int rank_lo() const { return rank_lo_; }
    int rank_hi() const { return rank_hi_; }
    MPI_Comm comm() const { return comm_; }
    bool is_parallel() const { return nprocs_ > 1; }

    /// Convert local k-index (0-based, no ghost) to global
    int k_local_to_global(int k_local) const { return k_global_start_ + k_local; }

    /// Global z-coordinate for local k-index
    double z_global(int k_local, double z_min, double Lz) const {
        double dz = Lz / nz_global_;
        return z_min + (k_global_start_ + k_local + 0.5) * dz;
    }

    /// Allreduce scalar (sum)
    double allreduce_sum(double local_val) const;

    /// Allreduce scalar (min)
    double allreduce_min(double local_val) const;

    /// Allreduce scalar (max)
    double allreduce_max(double local_val) const;

    /// Allreduce vector (sum, in-place)
    void allreduce_sum(double* data, int count) const;

private:
    MPI_Comm comm_;
    int rank_;
    int nprocs_;
    int nz_global_;
    int nz_local_;
    int k_global_start_;
    int rank_lo_;  // Periodic neighbor (z-lo)
    int rank_hi_;  // Periodic neighbor (z-hi)
};

} // namespace nncfd
```

Create `src/decomposition.cpp`:

```cpp
#include "decomposition.hpp"
#include <stdexcept>

namespace nncfd {

Decomposition::Decomposition(MPI_Comm comm, int Nz_global)
    : comm_(comm), nz_global_(Nz_global)
{
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &nprocs_);

    if (Nz_global < nprocs_) {
        throw std::runtime_error("Nz_global (" + std::to_string(Nz_global)
            + ") must be >= nprocs (" + std::to_string(nprocs_) + ")");
    }

    // Even distribution with remainder spread across first ranks
    int base = Nz_global / nprocs_;
    int remainder = Nz_global % nprocs_;

    nz_local_ = base + (rank_ < remainder ? 1 : 0);

    // Compute global start index
    k_global_start_ = 0;
    for (int r = 0; r < rank_; ++r) {
        k_global_start_ += base + (r < remainder ? 1 : 0);
    }

    // Periodic neighbors in z
    rank_lo_ = (rank_ - 1 + nprocs_) % nprocs_;
    rank_hi_ = (rank_ + 1) % nprocs_;
}

Decomposition::Decomposition(int Nz_global)
    : comm_(MPI_COMM_SELF), rank_(0), nprocs_(1),
      nz_global_(Nz_global), nz_local_(Nz_global),
      k_global_start_(0), rank_lo_(0), rank_hi_(0)
{
}

double Decomposition::allreduce_sum(double local_val) const {
    if (nprocs_ == 1) return local_val;
    double global_val = 0.0;
    MPI_Allreduce(&local_val, &global_val, 1, MPI_DOUBLE, MPI_SUM, comm_);
    return global_val;
}

double Decomposition::allreduce_min(double local_val) const {
    if (nprocs_ == 1) return local_val;
    double global_val = 0.0;
    MPI_Allreduce(&local_val, &global_val, 1, MPI_DOUBLE, MPI_MIN, comm_);
    return global_val;
}

double Decomposition::allreduce_max(double local_val) const {
    if (nprocs_ == 1) return local_val;
    double global_val = 0.0;
    MPI_Allreduce(&local_val, &global_val, 1, MPI_DOUBLE, MPI_MAX, comm_);
    return global_val;
}

void Decomposition::allreduce_sum(double* data, int count) const {
    if (nprocs_ == 1) return;
    MPI_Allreduce(MPI_IN_PLACE, data, count, MPI_DOUBLE, MPI_SUM, comm_);
}

} // namespace nncfd
```

**Step 3: Add MPI to CMakeLists.txt**

After the `find_package(CUDAToolkit)` block:

```cmake
# MPI support (optional)
option(USE_MPI "Enable MPI for multi-GPU domain decomposition" OFF)
if(USE_MPI)
    find_package(MPI REQUIRED)
    target_link_libraries(nn_cfd_core MPI::MPI_CXX)
    target_compile_definitions(nn_cfd_core PUBLIC USE_MPI)
endif()
```

Add `src/decomposition.cpp` to `LIB_SOURCES`.

**Step 4: Build and test**

```bash
cd build && cmake .. -DUSE_GPU_OFFLOAD=ON -DUSE_MPI=ON && make -j$(nproc)
mpirun -np 2 ./test_decomposition
```

Expected: PASS with 2 ranks.

**Step 5: Commit**

```bash
git add include/decomposition.hpp src/decomposition.cpp tests/test_decomposition.cpp
git add -p CMakeLists.txt
git commit -m "feat: add MPI z-slab domain decomposition class"
```

---

### Task 2.2: Halo Exchange

**Files:**
- Create: `include/halo_exchange.hpp`
- Create: `src/halo_exchange.cpp`
- Create: `tests/test_halo_exchange.cpp`

**Step 1: Write the test**

Create `tests/test_halo_exchange.cpp`:

```cpp
#include "halo_exchange.hpp"
#include "decomposition.hpp"
#include "mesh.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace nncfd;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const int Nx = 8, Ny = 8, Nz_global = 16, Ng = 1;
    Decomposition decomp(MPI_COMM_WORLD, Nz_global);

    const int Nz_local = decomp.nz_local();
    const int stride = Nx + 2*Ng;
    const int plane_stride = stride * (Ny + 2*Ng);
    const int total = (Nx + 2*Ng) * (Ny + 2*Ng) * (Nz_local + 2*Ng);

    std::vector<double> field(total, 0.0);

    // Fill interior with rank-dependent pattern: value = k_global * 1000 + j * 100 + i
    for (int k = 0; k < Nz_local; ++k) {
        int k_global = decomp.k_local_to_global(k);
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                int idx = (k + Ng) * plane_stride + (j + Ng) * stride + (i + Ng);
                field[idx] = k_global * 1000.0 + j * 100.0 + i;
            }
        }
    }

    // Exchange halos
    HaloExchange halo(decomp, Nx, Ny, Nz_local, Ng);
    halo.exchange(field.data(), stride, plane_stride);

    // Verify ghost cells match neighbor's interior
    // Low-z ghost (k=0) should have neighbor's last interior plane
    {
        int k_ghost = 0;
        int k_expected_global = (decomp.k_global_start() - 1 + Nz_global) % Nz_global;
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                int idx = k_ghost * plane_stride + (j + Ng) * stride + (i + Ng);
                double expected = k_expected_global * 1000.0 + j * 100.0 + i;
                assert(std::abs(field[idx] - expected) < 1e-10);
            }
        }
    }

    // High-z ghost (k=Nz_local+1) should have neighbor's first interior plane
    {
        int k_ghost = Nz_local + 2*Ng - 1;
        int k_expected_global = (decomp.k_global_start() + Nz_local) % Nz_global;
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                int idx = k_ghost * plane_stride + (j + Ng) * stride + (i + Ng);
                double expected = k_expected_global * 1000.0 + j * 100.0 + i;
                assert(std::abs(field[idx] - expected) < 1e-10);
            }
        }
    }

    if (rank == 0) {
        std::cout << "PASS: Halo exchange with " << nprocs << " ranks" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
```

**Step 2: Implement HaloExchange**

Create `include/halo_exchange.hpp`:

```cpp
#pragma once

#include "decomposition.hpp"
#include <vector>

namespace nncfd {

/// Manages z-direction halo exchange between MPI ranks
/// Supports both GPU-direct and host-staged communication
class HaloExchange {
public:
    HaloExchange(const Decomposition& decomp, int Nx, int Ny, int Nz_local, int Ng);
    ~HaloExchange();

    /// Exchange z-halos for a single field (CPU memory)
    void exchange(double* field, int stride, int plane_stride);

    /// Exchange z-halos for a single field (GPU device memory)
    void exchange_device(double* d_field, int stride, int plane_stride);

    /// Exchange z-halos for multiple fields simultaneously
    void exchange_batch(double** fields, int num_fields, int stride, int plane_stride);

private:
    const Decomposition& decomp_;
    int Nx_, Ny_, Nz_local_, Ng_;
    int face_size_;  // (Nx+2Ng) * (Ny+2Ng) * Ng

    // Buffers for packing/unpacking
    std::vector<double> send_lo_, send_hi_;
    std::vector<double> recv_lo_, recv_hi_;

    // GPU buffers (if available)
    double* d_send_lo_ = nullptr;
    double* d_send_hi_ = nullptr;
    double* d_recv_lo_ = nullptr;
    double* d_recv_hi_ = nullptr;
    bool gpu_buffers_initialized_ = false;

    void init_gpu_buffers();
    void pack_face_cpu(const double* field, double* buffer, int stride, int plane_stride, int k_start);
    void unpack_face_cpu(double* field, const double* buffer, int stride, int plane_stride, int k_start);
};

} // namespace nncfd
```

Create `src/halo_exchange.cpp`:

```cpp
#include "halo_exchange.hpp"
#include <cstring>
#include <stdexcept>

#ifdef USE_GPU_OFFLOAD
#include "cuda_halo.hpp"
#include <cuda_runtime.h>
#endif

namespace nncfd {

HaloExchange::HaloExchange(const Decomposition& decomp, int Nx, int Ny, int Nz_local, int Ng)
    : decomp_(decomp), Nx_(Nx), Ny_(Ny), Nz_local_(Nz_local), Ng_(Ng)
{
    face_size_ = (Nx + 2*Ng) * (Ny + 2*Ng) * Ng;
    send_lo_.resize(face_size_);
    send_hi_.resize(face_size_);
    recv_lo_.resize(face_size_);
    recv_hi_.resize(face_size_);
}

HaloExchange::~HaloExchange() {
#ifdef USE_GPU_OFFLOAD
    if (gpu_buffers_initialized_) {
        cudaFree(d_send_lo_);
        cudaFree(d_send_hi_);
        cudaFree(d_recv_lo_);
        cudaFree(d_recv_hi_);
    }
#endif
}

void HaloExchange::pack_face_cpu(const double* field, double* buffer,
                                  int stride, int plane_stride, int k_start)
{
    int buf_idx = 0;
    for (int g = 0; g < Ng_; ++g) {
        int k = k_start + g;
        for (int j = 0; j < Ny_ + 2*Ng_; ++j) {
            for (int i = 0; i < Nx_ + 2*Ng_; ++i) {
                buffer[buf_idx++] = field[k * plane_stride + j * stride + i];
            }
        }
    }
}

void HaloExchange::unpack_face_cpu(double* field, const double* buffer,
                                    int stride, int plane_stride, int k_start)
{
    int buf_idx = 0;
    for (int g = 0; g < Ng_; ++g) {
        int k = k_start + g;
        for (int j = 0; j < Ny_ + 2*Ng_; ++j) {
            for (int i = 0; i < Nx_ + 2*Ng_; ++i) {
                field[k * plane_stride + j * stride + i] = buffer[buf_idx++];
            }
        }
    }
}

void HaloExchange::exchange(double* field, int stride, int plane_stride) {
    if (!decomp_.is_parallel()) return;

    // Pack: send_lo = first Ng interior planes, send_hi = last Ng interior planes
    pack_face_cpu(field, send_lo_.data(), stride, plane_stride, Ng_);
    pack_face_cpu(field, send_hi_.data(), stride, plane_stride, Nz_local_);

    MPI_Request reqs[4];

    // Send lo → recv on neighbor's hi ghost; send hi → recv on neighbor's lo ghost
    MPI_Isend(send_lo_.data(), face_size_, MPI_DOUBLE, decomp_.rank_lo(), 0, decomp_.comm(), &reqs[0]);
    MPI_Isend(send_hi_.data(), face_size_, MPI_DOUBLE, decomp_.rank_hi(), 1, decomp_.comm(), &reqs[1]);
    MPI_Irecv(recv_lo_.data(), face_size_, MPI_DOUBLE, decomp_.rank_lo(), 1, decomp_.comm(), &reqs[2]);
    MPI_Irecv(recv_hi_.data(), face_size_, MPI_DOUBLE, decomp_.rank_hi(), 0, decomp_.comm(), &reqs[3]);

    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

    // Unpack: lo ghost = planes [0, Ng), hi ghost = planes [Nz_local+Ng, Nz_local+2Ng)
    unpack_face_cpu(field, recv_lo_.data(), stride, plane_stride, 0);
    unpack_face_cpu(field, recv_hi_.data(), stride, plane_stride, Nz_local_ + Ng_);
}

void HaloExchange::exchange_device(double* d_field, int stride, int plane_stride) {
    if (!decomp_.is_parallel()) return;

#ifdef USE_GPU_OFFLOAD
    if (!gpu_buffers_initialized_) init_gpu_buffers();

    // Pack on GPU
    cuda_kernels::launch_pack_z_face(d_field, d_send_lo_, Nx_, Ny_, Nz_local_, Ng_, true);
    cuda_kernels::launch_pack_z_face(d_field, d_send_hi_, Nx_, Ny_, Nz_local_, Ng_, false);
    cudaDeviceSynchronize();

    MPI_Request reqs[4];
    MPI_Isend(d_send_lo_, face_size_, MPI_DOUBLE, decomp_.rank_lo(), 0, decomp_.comm(), &reqs[0]);
    MPI_Isend(d_send_hi_, face_size_, MPI_DOUBLE, decomp_.rank_hi(), 1, decomp_.comm(), &reqs[1]);
    MPI_Irecv(d_recv_lo_, face_size_, MPI_DOUBLE, decomp_.rank_lo(), 1, decomp_.comm(), &reqs[2]);
    MPI_Irecv(d_recv_hi_, face_size_, MPI_DOUBLE, decomp_.rank_hi(), 0, decomp_.comm(), &reqs[3]);
    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

    // Unpack on GPU
    cuda_kernels::launch_unpack_z_face(d_field, d_recv_lo_, Nx_, Ny_, Nz_local_, Ng_, true);
    cuda_kernels::launch_unpack_z_face(d_field, d_recv_hi_, Nx_, Ny_, Nz_local_, Ng_, false);
    cudaDeviceSynchronize();
#else
    // Fallback: copy to host, exchange, copy back
    exchange(d_field, stride, plane_stride);
#endif
}

#ifdef USE_GPU_OFFLOAD
void HaloExchange::init_gpu_buffers() {
    cudaMalloc(&d_send_lo_, face_size_ * sizeof(double));
    cudaMalloc(&d_send_hi_, face_size_ * sizeof(double));
    cudaMalloc(&d_recv_lo_, face_size_ * sizeof(double));
    cudaMalloc(&d_recv_hi_, face_size_ * sizeof(double));
    gpu_buffers_initialized_ = true;
}
#endif

} // namespace nncfd
```

**Step 3: Build and test**

```bash
cd build && cmake .. -DUSE_GPU_OFFLOAD=ON -DUSE_MPI=ON && make -j$(nproc)
mpirun -np 2 ./test_halo_exchange
mpirun -np 4 ./test_halo_exchange
```

**Step 4: Commit**

```bash
git add include/halo_exchange.hpp src/halo_exchange.cpp tests/test_halo_exchange.cpp
git commit -m "feat: add MPI z-halo exchange with GPU-direct support"
```

---

### Task 2.3: MPI Integration into Solver

**Files:**
- Modify: `src/solver.cpp` (inject Decomposition, add allreduce calls)
- Modify: `src/solver_time.cpp` (halo exchange after RK stages)
- Modify: `app/main_channel.cpp` (MPI_Init, create Decomposition)
- Create: `tests/test_mpi_channel.cpp`

This task wires MPI into the existing solver. Key insertion points:

1. `RANSSolver` constructor accepts optional `Decomposition*`
2. `compute_adaptive_dt()`: `MPI_Allreduce(MPI_MIN)` on dt
3. `bulk_velocity()` / `compute_kinetic_energy_device()`: `MPI_Allreduce(MPI_SUM)`
4. `accumulate_statistics()`: `MPI_Allreduce(MPI_SUM)` on plane sums
5. After each RK substage velocity update: `halo_exchange.exchange_device(u, v, w)`
6. After Poisson solve: halo exchange on pressure correction
7. `main_channel.cpp`: `MPI_Init`, create Decomposition, pass to solver

**Step 1: Write test (2-rank channel matches 1-rank)**

Create `tests/test_mpi_channel.cpp` — runs a short Poiseuille flow with 2 MPI ranks and verifies the result matches the serial (1-rank) analytical solution.

**Step 2: Modify solver to accept Decomposition** — add `decomp_` member, conditional allreduce wrappers.

**Step 3: Add halo exchange calls** — insert after each RK substage in `solver_time.cpp`.

**Step 4: Build and test**

```bash
mpirun -np 1 ./test_mpi_channel  # baseline
mpirun -np 2 ./test_mpi_channel  # must match
```

**Step 5: Commit**

```bash
git commit -m "feat: integrate MPI domain decomposition into solver pipeline"
```

---

### Task 2.4: Distributed FFT Poisson Solver

**Files:**
- Create: `src/poisson_solver_fft_mpi.cpp`
- Modify: `include/config.hpp` (add `PoissonSolverType::FFT_MPI`)
- Create: `tests/test_mpi_poisson.cpp`

Implements pencil-transpose distributed FFT:
1. z-slabs → x-pencils via `MPI_Alltoallv`
2. 1D cuFFT in x
3. Tridiagonal solve in y (cuSPARSE, local per pencil)
4. x-pencils → z-slabs transpose back
5. 1D cuFFT in z (local, each rank has full z for its slab after transpose)

**Test**: Distributed Poiseuille pressure solve matches serial FFT to machine precision.

**Commit**:
```bash
git commit -m "feat: add distributed FFT Poisson solver with MPI pencil transpose"
```

---

### Task 2.5: MPI for All Physics (Transparency Pass)

**Files:**
- Modify: `src/solver_recycling.cpp` (~30 LOC)
- Modify: `src/solver.cpp` (filter, diagnostics)
- Modify: `app/main_duct.cpp`, `app/main_taylor_green_3d.cpp` (MPI init)
- Modify: `src/solver_vtk.cpp` (parallel VTK output)

Insert allreduce calls at all diagnostic/statistics points. Add MPI_Init to all app entry points. Add `.pvd` manifest writing for parallel VTK.

**Test**: Run all existing tests with `mpirun -np 1` to verify no regression. Then `mpirun -np 2` for channel/TGV.

**Commit**:
```bash
git commit -m "feat: make all physics MPI-transparent (diagnostics, filter, recycling, VTK)"
```

---

## Phase 3: Immersed Boundary Method

### Task 3.1: IBM Geometry — SDF Evaluation

**Files:**
- Create: `include/ibm_geometry.hpp`
- Create: `src/ibm_geometry.cpp`
- Create: `tests/test_ibm_sdf.cpp`

**Step 1: Write the test**

Create `tests/test_ibm_sdf.cpp`:

```cpp
#include "ibm_geometry.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

using namespace nncfd;

int main() {
    // Test cylinder SDF
    {
        CylinderBody cyl(5.0, 0.0, 0.5);  // center=(5,0), radius=0.5

        // On surface: phi = 0
        assert(std::abs(cyl.phi(5.5, 0.0, 0.0)) < 1e-14);
        assert(std::abs(cyl.phi(4.5, 0.0, 0.0)) < 1e-14);

        // Inside: phi < 0
        assert(cyl.phi(5.0, 0.0, 0.0) < 0);

        // Outside: phi > 0
        assert(cyl.phi(6.0, 0.0, 0.0) > 0);
        assert(std::abs(cyl.phi(6.0, 0.0, 0.0) - 0.5) < 1e-14);

        // Normal on surface points outward
        auto [nx, ny, nz] = cyl.normal(5.5, 0.0, 0.0);
        assert(std::abs(nx - 1.0) < 1e-14);
        assert(std::abs(ny) < 1e-14);
    }

    // Test sphere SDF
    {
        SphereBody sph(0.0, 0.0, 0.0, 1.0);  // center=(0,0,0), radius=1

        assert(std::abs(sph.phi(1.0, 0.0, 0.0)) < 1e-14);  // surface
        assert(sph.phi(0.0, 0.0, 0.0) < 0);                 // inside
        assert(std::abs(sph.phi(2.0, 0.0, 0.0) - 1.0) < 1e-14);  // outside
    }

    // Test NACA 0012 SDF (basic sanity)
    {
        NACABody naca(0.0, 0.0, 1.0, 0.0, "0012");  // chord=1, aoa=0

        // Leading edge: on surface
        double phi_le = naca.phi(0.0, 0.0, 0.0);
        assert(std::abs(phi_le) < 0.01);  // approximately on surface

        // Trailing edge: on surface
        double phi_te = naca.phi(1.0, 0.0, 0.0);
        assert(std::abs(phi_te) < 0.01);

        // Well inside: negative
        assert(naca.phi(0.3, 0.0, 0.0) < 0);

        // Well outside: positive
        assert(naca.phi(0.5, 0.5, 0.0) > 0);
    }

    std::cout << "PASS: IBM SDF tests" << std::endl;
    return 0;
}
```

**Step 2: Implement geometry classes**

Create `include/ibm_geometry.hpp`:

```cpp
#pragma once

#include <string>
#include <tuple>
#include <memory>
#include <vector>

namespace nncfd {

/// Base class for immersed boundary geometry
class IBMBody {
public:
    virtual ~IBMBody() = default;

    /// Signed distance function: negative inside, positive outside
    virtual double phi(double x, double y, double z) const = 0;

    /// Outward surface normal (unit vector)
    virtual std::tuple<double, double, double> normal(double x, double y, double z) const;

    /// Closest point on surface (for interpolation)
    virtual std::tuple<double, double, double> closest_point(double x, double y, double z) const;

    /// Name for logging
    virtual std::string name() const = 0;
};

class CylinderBody : public IBMBody {
public:
    CylinderBody(double cx, double cy, double radius);
    double phi(double x, double y, double z) const override;
    std::tuple<double, double, double> normal(double x, double y, double z) const override;
    std::string name() const override { return "Cylinder"; }
private:
    double cx_, cy_, radius_;
};

class SphereBody : public IBMBody {
public:
    SphereBody(double cx, double cy, double cz, double radius);
    double phi(double x, double y, double z) const override;
    std::tuple<double, double, double> normal(double x, double y, double z) const override;
    std::string name() const override { return "Sphere"; }
private:
    double cx_, cy_, cz_, radius_;
};

class NACABody : public IBMBody {
public:
    /// 4-digit NACA airfoil
    /// @param x_le  Leading edge x-coordinate
    /// @param y_le  Leading edge y-coordinate
    /// @param chord Chord length
    /// @param aoa   Angle of attack (radians)
    /// @param digits NACA 4-digit designation (e.g., "0012")
    NACABody(double x_le, double y_le, double chord, double aoa, const std::string& digits);
    double phi(double x, double y, double z) const override;
    std::string name() const override { return "NACA" + digits_; }
private:
    double x_le_, y_le_, chord_, aoa_;
    std::string digits_;
    double max_camber_, camber_pos_, thickness_;

    /// NACA thickness distribution at normalized x in [0,1]
    double thickness_at(double xn) const;
    /// NACA camber line y at normalized x
    double camber_at(double xn) const;
    /// Camber line slope dy/dx at normalized x
    double camber_slope_at(double xn) const;
};

/// Factory: create body from config strings
std::unique_ptr<IBMBody> create_ibm_body(const std::string& type,
    double param1, double param2, double param3,
    double param4 = 0.0, const std::string& extra = "");

} // namespace nncfd
```

Create `src/ibm_geometry.cpp` with implementations of all SDF functions.

**Step 3: Build and test**

```bash
cd build && cmake .. -DUSE_GPU_OFFLOAD=ON && make -j$(nproc)
ctest -R test_ibm_sdf --output-on-failure
```

**Step 4: Commit**

```bash
git add include/ibm_geometry.hpp src/ibm_geometry.cpp tests/test_ibm_sdf.cpp
git commit -m "feat: add IBM geometry classes (cylinder, sphere, NACA airfoil SDF)"
```

---

### Task 3.2: IBM Forcing — Cell Classification and Direct Forcing

**Files:**
- Create: `include/ibm_forcing.hpp`
- Create: `src/ibm_forcing.cpp`
- Create: `tests/test_ibm_forcing.cpp`

**Step 1: Write the test**

Test: Cylinder in Stokes flow — IBM forcing should enforce no-slip. After several steps, velocity inside cylinder should be ~0, drag should converge with grid refinement.

**Step 2: Implement IBMForcing class**

Key methods:
- `classify_cells(mesh, sdf)` → fills `cell_type` array (Fluid/Solid/Forcing)
- `compute_forcing(mesh, u, v, w, sdf, dt)` → fills `f_ibm_u, f_ibm_v, f_ibm_w`
- `compute_forces()` → integrates IBM forcing for Cd/Cl
- GPU kernel: single kernel per velocity component

**Step 3: Integrate into solver_time.cpp**

After predictor, before Poisson solve:
```cpp
if (ibm_forcing_) {
    ibm_forcing_->compute_forcing(*mesh_, u_star, v_star, w_star, current_dt_);
    // Add forcing to predicted velocity
    // apply_ibm_forcing_kernel(u_star, f_ibm_u, dt, ...)
}
```

**Step 4: Build and test**

```bash
ctest -R test_ibm_forcing --output-on-failure
```

**Step 5: Commit**

```bash
git commit -m "feat: add IBM direct forcing with cell classification and drag computation"
```

---

### Task 3.3: IBM App Executables

**Files:**
- Create: `app/main_cylinder.cpp`
- Create: `app/main_airfoil.cpp`
- Create: `examples/11_cylinder_flow/cylinder_re100.cfg`
- Create: `examples/11_cylinder_flow/cylinder_re3900.cfg`
- Create: `examples/12_naca_airfoil/naca0012_re1000.cfg`
- Modify: `CMakeLists.txt` (new executables)

Follow existing `main_channel.cpp` pattern:
1. Parse config
2. Create mesh (larger domain: x=[0,30D], y=[-10D,10D], z=[0,πD])
3. Create IBM body from config
4. Initialize solver with IBM
5. Time-stepping loop with drag/lift output

**Commit**:
```bash
git commit -m "feat: add cylinder and airfoil app executables with example configs"
```

---

### Task 3.4: IBM Poisson Compatibility

**Files:**
- Modify: `src/poisson_solver_multigrid.cpp` (mask solid cells in RHS)
- Modify: `src/poisson_solver_fft.cpp` (fluid-only mean subtraction)

Solid cells get zero RHS. Mean subtraction sums only over fluid cells (weighted by fluid volume fraction for cut cells).

**Test**: Run cylinder_re100 for 100 steps — divergence should be O(1e-6) and pressure field should be smooth around the cylinder.

**Commit**:
```bash
git commit -m "feat: make Poisson solvers IBM-compatible (solid cell masking)"
```

---

## Phase 4: LES Subgrid-Scale Models

### Task 4.1: Velocity Gradient Tensor Kernel

**Files:**
- Create: `src/cuda_kernels/velocity_gradient.cu`
- Create: `include/velocity_gradient.hpp`
- Create: `tests/test_velocity_gradient.cpp`

**Step 1: Write the test**

Create `tests/test_velocity_gradient.cpp`:

```cpp
#include "velocity_gradient.hpp"
#include "mesh.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

using namespace nncfd;

int main() {
    // Test on linear velocity field: u = ax + by, v = cx + dy
    // Gradient should be exact to machine precision for 2nd-order FD
    const int Nx = 16, Ny = 16, Nz = 16, Ng = 1;
    const double dx = 1.0 / Nx, dy = 1.0 / Ny, dz = 1.0 / Nz;

    Mesh mesh;
    mesh.init_uniform(Nx, Ny, Nz, 0, 1, 0, 1, 0, 1, Ng);

    // Allocate velocity (staggered) and gradient tensor (cell-centered)
    const int u_stride = Nx + 2*Ng + 1;
    const int v_stride = Nx + 2*Ng;
    const int w_stride = Nx + 2*Ng;
    const int cell_stride = Nx + 2*Ng;
    const int u_plane = u_stride * (Ny + 2*Ng);
    const int v_plane = v_stride * (Ny + 2*Ng + 1);
    const int w_plane = w_stride * (Ny + 2*Ng);
    const int cell_plane = cell_stride * (Ny + 2*Ng);

    // Fill u = 2x (linear in x): du/dx = 2
    std::vector<double> u((Nx+1+2*Ng)*(Ny+2*Ng)*(Nz+2*Ng), 0);
    for (int k = 0; k < Nz+2*Ng; ++k)
        for (int j = 0; j < Ny+2*Ng; ++j)
            for (int i = 0; i < Nx+1+2*Ng; ++i)
                u[k*u_plane + j*u_stride + i] = 2.0 * i * dx;

    // Zero v, w
    std::vector<double> v((Nx+2*Ng)*(Ny+1+2*Ng)*(Nz+2*Ng), 0);
    std::vector<double> w((Nx+2*Ng)*(Ny+2*Ng)*(Nz+1+2*Ng), 0);

    // Compute gradient tensor
    std::vector<double> g11(Nx*Ny*Nz, 0);
    // ... (9 components)

    VelocityGradient grad;
    grad.compute(mesh, u.data(), v.data(), w.data(),
                 g11.data(), /* ... 8 more components */);

    // Check du/dx ≈ 2.0 at all interior cells
    double max_err = 0;
    for (int idx = 0; idx < Nx*Ny*Nz; ++idx) {
        max_err = std::max(max_err, std::abs(g11[idx] - 2.0));
    }
    assert(max_err < 1e-12 && "du/dx must be exact for linear field");

    std::cout << "PASS: Velocity gradient tensor" << std::endl;
    return 0;
}
```

**Step 2: Implement CUDA velocity gradient kernel**

Single kernel reads staggered u,v,w → outputs 9 cell-centered gradient components. Uses shared memory tiling (same 10x10x10 pattern as smoother). Staggered-to-center interpolation: `u_center = 0.5*(u[i] + u[i+1])`.

**Step 3: Build and test**

```bash
ctest -R test_velocity_gradient --output-on-failure
```

**Step 4: Commit**

```bash
git commit -m "feat: add CUDA velocity gradient tensor kernel with shared memory tiling"
```

---

### Task 4.2: Static Smagorinsky Model

**Files:**
- Create: `include/turbulence_les.hpp`
- Create: `src/turbulence_les.cpp`
- Modify: `include/config.hpp` (add LES enum entries)
- Modify: `src/config.cpp` (factory cases)
- Create: `tests/test_les_sgs.cpp`

**Step 1: Write the test**

```cpp
// Test: Smagorinsky on known strain rate field
// Pure shear u = Sy: |S| = S, nu_sgs = (Cs*delta)^2 * S
// Verify nu_sgs matches analytical
```

**Step 2: Implement LES base + Smagorinsky**

Create `include/turbulence_les.hpp`:

```cpp
#pragma once
#include "turbulence_model.hpp"
#include "velocity_gradient.hpp"

namespace nncfd {

/// Base class for LES SGS models
class LESModel : public TurbulenceModel {
public:
    void update(const Mesh& mesh, const VectorField& velocity,
                const ScalarField& k, const ScalarField& omega,
                ScalarField& nu_t, TensorField* tau_ij,
                const TurbulenceDeviceView* device_view) override;

    bool uses_transport_equations() const override { return false; }
    bool is_gpu_ready() const override { return true; }

protected:
    /// Override in subclass: compute nu_sgs from gradient tensor
    virtual void compute_nu_sgs(
        const Mesh& mesh,
        const double* g11, const double* g12, const double* g13,
        const double* g21, const double* g22, const double* g23,
        const double* g31, const double* g32, const double* g33,
        double* nu_sgs,
        const TurbulenceDeviceView* device_view) = 0;

    VelocityGradient grad_computer_;
};

class SmagorinskyModel : public LESModel {
public:
    explicit SmagorinskyModel(double Cs = 0.17) : Cs_(Cs) {}
    std::string name() const override { return "Smagorinsky"; }
protected:
    void compute_nu_sgs(...) override;
private:
    double Cs_;
};

class WALEModel : public LESModel { /* ... */ };
class VremanModel : public LESModel { /* ... */ };
class SigmaModel : public LESModel { /* ... */ };
class DynamicSmagorinskyModel : public LESModel { /* ... */ };

} // namespace nncfd
```

**Step 3: Add to config enums and factory**

In `include/config.hpp`, add to `TurbulenceModelType`:
```cpp
Smagorinsky, DynamicSmagorinsky, WALE, Vreman, Sigma
```

In `src/config.cpp`, add factory cases:
```cpp
case TurbulenceModelType::Smagorinsky:
    return std::make_unique<SmagorinskyModel>(config.les_Cs);
// ... etc
```

**Step 4: Build and test**

```bash
ctest -R test_les_sgs --output-on-failure
```

**Step 5: Commit**

```bash
git commit -m "feat: add Smagorinsky LES SGS model with CUDA velocity gradient"
```

---

### Task 4.3: WALE Model

**Files:**
- Modify: `src/turbulence_les.cpp` (add WALE implementation)
- Modify: `tests/test_les_sgs.cpp` (add WALE tests)

WALE computes `S^d_ij` (traceless symmetric part of `g_ik * g_kj`). Single CUDA kernel per cell — all thread-local arithmetic, no communication.

**Test**: Verify WALE vanishes in pure shear (u = Sy, v = w = 0) and pure rotation.

**Commit**:
```bash
git commit -m "feat: add WALE LES SGS model"
```

---

### Task 4.4: Vreman Model

**Files:**
- Modify: `src/turbulence_les.cpp`
- Modify: `tests/test_les_sgs.cpp`

Vreman: compute `beta_ij`, `Bbeta`, `alpha_ij`. Vanishes at walls.

**Test**: Verify vanishes in pure shear, correct magnitude on known turbulent field.

**Commit**:
```bash
git commit -m "feat: add Vreman LES SGS model"
```

---

### Task 4.5: Sigma Model

**Files:**
- Modify: `src/turbulence_les.cpp`
- Modify: `tests/test_les_sgs.cpp`

Sigma: compute 3x3 SVD (analytical closed-form for 3x3, ~100 FLOPs). Vanishes for all laminar flows.

**Test**: Verify vanishes for pure shear, solid body rotation, and axisymmetric expansion.

**Commit**:
```bash
git commit -m "feat: add Sigma LES SGS model"
```

---

### Task 4.6: Dynamic Smagorinsky Model

**Files:**
- Modify: `src/turbulence_les.cpp`
- Create: `src/les_test_filter.cpp` (or `src/cuda_kernels/test_filter.cu`)
- Create: `tests/test_dynamic_smag.cpp`

Most complex LES model:
1. Test filter kernel: 3D box filter at 2Δ width (trapezoidal weights for stretched y)
2. Germano identity: L_ij, M_ij computation on GPU
3. Plane averaging (Lilly): GPU reduction per y-plane → Ny-length arrays
4. MPI: `allreduce_sum` on the Ny-length numerator/denominator arrays
5. Optional Lagrangian averaging (config: `dynamic_smag_averaging = lagrangian`)

**Test**: Verify Cs² recovery for known HIT-like field. Verify plane-averaged Cs is positive.

**Commit**:
```bash
git commit -m "feat: add Dynamic Smagorinsky LES model with test filter and plane averaging"
```

---

### Task 4.7: LES Channel Integration Test

**Files:**
- Create: `tests/test_les_channel.cpp`
- Create: `examples/10_les_channel/les_retau590_wale.cfg`

Run 500 steps of channel at Re_tau=590 on coarse grid (64x64x64) with WALE. Verify:
- Stable (no blow-up)
- nu_sgs > 0 everywhere
- Reasonable Re_tau estimate

**Commit**:
```bash
git commit -m "test: add LES channel integration test with WALE model"
```

---

## Phase 5: HDF5 I/O + Validation Campaign

### Task 5.1: HDF5 Checkpoint/Restart

**Files:**
- Create: `include/checkpoint.hpp`
- Create: `src/checkpoint.cpp`
- Modify: `CMakeLists.txt` (find HDF5, optional)
- Create: `tests/test_checkpoint.cpp`

**Step 1: Write the test**

```cpp
// Write checkpoint → restart → verify fields match bitwise
// Test with single-GPU and multi-GPU (if MPI enabled)
```

**Step 2: Implement checkpoint writer/reader**

```cpp
class Checkpoint {
public:
    static void write(const std::string& filename,
                      const RANSSolver& solver,
                      const Decomposition* decomp = nullptr);

    static void read(const std::string& filename,
                     RANSSolver& solver,
                     const Decomposition* decomp = nullptr);
};
```

Uses HDF5 C API. MPI-parallel via `H5Pset_fapl_mpio`. Each rank writes its local z-slab as a hyperslab selection.

**Step 3: Build and test**

```bash
cd build && cmake .. -DUSE_GPU_OFFLOAD=ON && make -j$(nproc)
ctest -R test_checkpoint --output-on-failure
```

**Step 4: Commit**

```bash
git commit -m "feat: add HDF5 checkpoint/restart with MPI-parallel I/O"
```

---

### Task 5.2: Validation Configs + Reference Data

**Files:**
- Create: `examples/02_turbulent_channel/channel_retau395.cfg`
- Create: `examples/02_turbulent_channel/channel_retau590.cfg`
- Create: `examples/10_les_channel/les_retau590_dynamic_smag.cfg`
- Create: `examples/10_les_channel/les_retau590_vreman.cfg`
- Create: `examples/11_cylinder_flow/cylinder_re100.cfg`
- Create: `examples/11_cylinder_flow/cylinder_re300.cfg`
- Create: `examples/11_cylinder_flow/cylinder_re3900_les.cfg`
- Create: `examples/12_naca_airfoil/naca0012_re1000_aoa5.cfg`
- Create: `scripts/download_reference_data.sh`

**Reference data sources:**
- MKM channel: `https://turbulence.oden.utexas.edu/data/MKM/chandata.tar.gz`
- Del Álamo Re_tau=590: `http://torroja.dmt.upm.es/channels/data/`
- Cylinder Re=100: Henderson (1995) — hardcode Cd=1.35, St=0.164 in test
- Cylinder Re=3900: Parnaudeau et al. (2008) — digitized mean wake profiles

**Commit**:
```bash
git commit -m "feat: add validation configs and reference data download script"
```

---

### Task 5.3: Validation Integration Tests

**Files:**
- Create: `tests/test_cylinder_re100.cpp`
- Create: `tests/test_les_tgv.cpp`

**test_cylinder_re100.cpp**: Run 2000 steps, check Cd within 5% of 1.35, St within 5% of 0.164. Label: `medium,gpu`.

**test_les_tgv.cpp**: TGV Re=1600, 64³ with WALE. Run 500 steps, check dissipation rate shape qualitatively (should increase then decrease). Label: `medium,gpu`.

**Commit**:
```bash
git commit -m "test: add cylinder Re=100 and TGV LES validation tests"
```

---

### Task 5.4: Final Integration — Run All Tests

**Step 1: Run full test suite**

```bash
cd build && cmake .. -DUSE_GPU_OFFLOAD=ON -DUSE_MPI=ON && make -j$(nproc)

# All fast tests
ctest -L fast --output-on-failure

# GPU tests
OMP_TARGET_OFFLOAD=MANDATORY ctest -L gpu --output-on-failure

# MPI tests
ctest -L mpi --output-on-failure

# Medium tests (validation)
OMP_TARGET_OFFLOAD=MANDATORY ctest -L medium --output-on-failure
```

**Step 2: CPU-only regression**

```bash
mkdir -p build_cpu && cd build_cpu
cmake .. -DCMAKE_CXX_COMPILER=g++ && make -j$(nproc)
ctest -L fast --output-on-failure
```

Expected: All fast tests pass on CPU (CUDA kernels fall back to OpenMP target which falls back to CPU).

**Step 3: Commit any fixes**

```bash
git commit -m "fix: address test failures from integration"
```

---

## Summary

| Phase | Tasks | New Files | Modified Files | Est. LOC |
|-------|-------|-----------|----------------|----------|
| **1: GPU Optimization** | 1.1–1.4 | 6 | 3 | ~800 |
| **2: MPI Decomposition** | 2.1–2.5 | 8 | 8 | ~1200 |
| **3: IBM** | 3.1–3.4 | 10 | 4 | ~1000 |
| **4: LES SGS** | 4.1–4.7 | 6 | 4 | ~800 |
| **5: HDF5 + Validation** | 5.1–5.4 | 12 | 2 | ~600 |
| **Total** | 24 tasks | ~42 files | ~21 files | ~4400 |

Each phase is independently testable and deployable. Phases can be committed to separate branches if desired.
