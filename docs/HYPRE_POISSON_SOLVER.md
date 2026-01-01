# HYPRE PFMG Poisson Solver

This document describes the HYPRE-based GPU-accelerated Poisson solver integration in NN-CFD.

## Overview

HYPRE's **PFMG (Parallel Semicoarsening Multigrid)** solver provides a high-performance alternative to the built-in multigrid solver for the pressure Poisson equation. When built with CUDA support, HYPRE runs the entire multigrid solve on the GPU, achieving **8-10x speedup** over the OpenMP-offloaded multigrid implementation.

### Key Features

- **GPU-native solving**: Entire multigrid V-cycle runs on GPU via HYPRE's CUDA backend
- **Automatic installation**: CMake FetchContent downloads and builds HYPRE automatically
- **Unified memory integration**: Seamless interoperability with OpenMP target offload
- **Structured grid optimization**: PFMG is optimized for Cartesian grids

## Performance

Benchmark results on NVIDIA H200 GPU (Hopper architecture):

| Grid Size | HYPRE PFMG | Built-in Multigrid | Speedup |
|-----------|------------|-------------------|---------|
| 32³       | 5.4 ms     | 52 ms             | **9.7x**  |
| 64³       | 5.4 ms     | 58 ms             | **10.8x** |
| 128³      | 8.0 ms     | 66 ms             | **8.2x**  |

The speedup comes from HYPRE's GPU-native implementation using CUDA kernels, versus the OpenMP target-offloaded implementation which incurs additional overhead from kernel launches and synchronization.

## Building with HYPRE

### Automatic Installation (Recommended)

HYPRE is automatically downloaded and built when you enable the `USE_HYPRE` option:

```bash
# GPU build with HYPRE (NVIDIA HPC SDK required)
mkdir build_hypre && cd build_hypre
CC=nvc CXX=nvc++ cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_GPU_OFFLOAD=ON \
    -DUSE_HYPRE=ON
make -j8
```

CMake will:
1. Download HYPRE v2.31.0 from GitHub
2. Configure HYPRE with CUDA support and unified memory
3. Build HYPRE as a static library
4. Link it with the NN-CFD library

### Build Options

The following HYPRE-related CMake options are available:

| Option | Default | Description |
|--------|---------|-------------|
| `USE_HYPRE` | OFF | Enable HYPRE PFMG Poisson solver |
| `CMAKE_CUDA_ARCHITECTURES` | 90 | CUDA compute capability (90=H100/H200) |

### Requirements

- **NVIDIA HPC SDK** (nvc/nvc++ compiler with `-cuda` support)
- **CUDA Toolkit** (for CUDA runtime and device libraries)
- **CMake 3.16+** (for FetchContent support)
- **Git** (for automatic HYPRE download)

### Compiler Flags

When `USE_HYPRE=ON`, the following flags are automatically added:
- `-cuda`: Enables OpenMP+CUDA interoperability in nvc++
- CUDA runtime is linked automatically

## Usage

### Enabling HYPRE in Configuration

Set `poisson_solver = hypre` in your configuration file or use the command line:

```bash
./channel --poisson_solver hypre --Nx 64 --Ny 128 --Nz 64
```

### Programmatic Usage

```cpp
#include "config.hpp"
#include "solver.hpp"

Config cfg;
cfg.poisson_solver = PoissonSolverType::HYPRE;  // Use HYPRE PFMG
cfg.Nx = 64;
cfg.Ny = 128;
cfg.Nz = 64;

Solver solver(cfg);
solver.run();
```

### Fallback Behavior

If HYPRE is not available (not compiled with `USE_HYPRE`), the solver automatically falls back to the built-in multigrid implementation.

## Technical Details

### Unified Memory Integration

The key challenge in integrating HYPRE CUDA with OpenMP target offload is memory management. The solution uses **CUDA unified (managed) memory** as an intermediary:

1. **Main solver arrays** use OpenMP `omp_target_alloc()` for device memory
2. **HYPRE interface buffers** use `cudaMallocManaged()` for managed memory
3. **Data transfer** happens on-device via OpenMP kernels with `is_device_ptr`

```
┌─────────────────────────────────────────────────────────────────┐
│                     GPU Device Memory                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐    ┌──────────────────────────────┐   │
│  │  OpenMP Device Mem  │    │  CUDA Managed Memory         │   │
│  │  (omp_target_alloc) │───>│  (cudaMallocManaged)         │   │
│  │                     │    │                              │   │
│  │  - u, v, w velocity │    │  - rhs_device_ (RHS buffer)  │   │
│  │  - pressure         │    │  - x_device_ (solution buf)  │   │
│  │  - RHS array        │    │                              │   │
│  └─────────────────────┘    └──────────────────────────────┘   │
│          │                              │                       │
│          │ OMP kernel with              │ HYPRE CUDA            │
│          │ is_device_ptr                │ operations            │
│          ▼                              ▼                       │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              HYPRE PFMG GPU Solve                           ││
│  │  - Restriction, prolongation on GPU                         ││
│  │  - Weighted Jacobi smoothing (GPU-parallel)                 ││
│  │  - All operations in unified memory space                   ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### OpenMP Pragma Patterns

The solver uses specific OpenMP pragma patterns for correct memory access:

```cpp
// Pack RHS from OMP-mapped arrays to managed memory buffers
#pragma omp target teams distribute parallel for collapse(3) \
    map(present: rhs_ptr[0:total_size]) \   // OMP-mapped: use present
    is_device_ptr(rhs_dev)                   // Managed memory: use is_device_ptr
for (int k = 0; k < Nz; ++k) {
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            rhs_dev[hypre_idx] = rhs_ptr[full_idx];
        }
    }
}
```

### PFMG Configuration

The HYPRE PFMG solver is configured for optimal GPU performance:

| Parameter | Value | Reason |
|-----------|-------|--------|
| RelaxType | 1 (Weighted Jacobi) | Fully parallel, GPU-friendly |
| NumPreRelax | 2 | Pre-smoothing sweeps |
| NumPostRelax | 2 | Post-smoothing sweeps |
| RAPType | 0 (Galerkin) | Standard coarse grid operator |

**Note**: Red-Black Gauss-Seidel (RelaxType=2) converges faster but has data dependencies that reduce GPU parallelism. Weighted Jacobi is preferred for GPU execution.

### Boundary Condition Support

The HYPRE solver supports all boundary condition types:

| BC Type | Support | Notes |
|---------|---------|-------|
| Periodic | Yes | Via `HYPRE_StructGridSetPeriodic()` |
| Neumann | Yes | Modified stencil coefficients |
| Dirichlet | Yes | Direct value enforcement |

For pure Neumann/Periodic problems (singular Laplacian), a single cell is pinned to zero to remove the constant nullspace.

## API Usage

### GPU Path (Recommended)

For GPU builds, use the `solve_device()` method with device pointers:

```cpp
// In solver integration (see src/solver.cpp)
if (use_hypre_ && hypre_poisson_solver_->using_cuda()) {
    // Solve directly on GPU-resident data
    iters = hypre_poisson_solver_->solve_device(rhs_ptr, p_ptr, cfg);
}
```

This path packs data on-device using OpenMP target kernels, then calls HYPRE PFMG which runs entirely on GPU.

### Host Path (Fallback)

The `solve(ScalarField&, ScalarField&, ...)` method is available for non-GPU builds or testing, but has known limitations with CUDA mode. Use the GPU path for production.

## Troubleshooting

### Build Issues

**Error: CUDA not found**
```
CMake Error: enable_language(CUDA) failed
```
Solution: Ensure CUDA toolkit is installed and `nvcc` is in PATH:
```bash
module load cuda  # or appropriate system command
export PATH=/usr/local/cuda/bin:$PATH
```

**Error: HYPRE download failed**
```
FetchContent: error downloading hypre
```
Solution: Check network connectivity or download manually:
```bash
git clone https://github.com/hypre-space/hypre.git _deps/hypre-src
git -C _deps/hypre-src checkout v2.31.0
```

### Runtime Issues

**Error: CUDA out of memory**
```
cudaMallocManaged failed: out of memory
```
Solution: Reduce grid size or check for GPU memory leaks. Use `nvidia-smi` to monitor memory usage.

**Error: Illegal memory access**
```
CUDA error: an illegal memory access was encountered
```
This usually indicates a mismatch between OMP-mapped and managed memory. Ensure:
- Use `map(present:...)` for OMP-mapped arrays
- Use `is_device_ptr(...)` for `cudaMallocManaged` pointers
- Call `cudaDeviceSynchronize()` between OMP kernels and HYPRE calls

**Slow performance or convergence issues**
1. Check that HYPRE CUDA is actually enabled:
   ```
   [HyprePoissonSolver] CUDA backend enabled (unified memory)
   ```
2. Verify GPU is being used: `nvidia-smi` should show GPU utilization
3. For ill-conditioned problems, increase `max_iter` in Poisson config

### Debugging

Enable verbose HYPRE output:
```cpp
PoissonConfig cfg;
cfg.verbose = true;  // Prints iteration counts and residuals
```

## Implementation Files

| File | Description |
|------|-------------|
| `include/poisson_solver_hypre.hpp` | HYPRE solver class declaration |
| `src/poisson_solver_hypre.cpp` | HYPRE solver implementation |
| `CMakeLists.txt` (lines 42-86) | HYPRE build configuration |
| `tests/test_hypre_all_bcs.cpp` | HYPRE initialization test |
| `tests/test_hypre_validation.cpp` | HYPRE vs Multigrid comparison test |

## Testing

### Running HYPRE Tests

HYPRE tests require building with the `USE_HYPRE=ON` flag:

```bash
# Build with HYPRE enabled
mkdir build_hypre && cd build_hypre
CC=nvc CXX=nvc++ cmake .. -DUSE_GPU_OFFLOAD=ON -DUSE_HYPRE=ON -DBUILD_TESTS=ON
make -j8

# Run HYPRE tests
./test_hypre_all_bcs      # Initialization test
./test_hypre_validation   # HYPRE vs Multigrid comparison
```

### CI Integration

The CI script supports HYPRE testing with the `--hypre` flag:

```bash
# Run full test suite with HYPRE
./scripts/ci.sh --hypre full

# Run only HYPRE tests
./scripts/ci.sh --hypre hypre
```

### Test Descriptions

| Test | Description |
|------|-------------|
| `test_hypre_all_bcs` | Verifies HYPRE initialization with different BC configurations |
| `test_hypre_validation` | Compares HYPRE and Multigrid solver results on 3D channel/duct flows |

### Cross-Build Comparison

The `test_hypre_validation` test supports cross-build comparison for CI:

```bash
# Generate CPU HYPRE reference (requires CPU build with USE_HYPRE=ON)
./test_hypre_validation --dump-prefix /path/to/ref

# Compare GPU HYPRE against reference (requires GPU build with USE_HYPRE=ON)
./test_hypre_validation --compare-prefix /path/to/ref
```

### Known Limitations

1. **ScalarField-based `solve()` method**: The `solve(ScalarField&, ScalarField&)` method has known issues in CUDA mode. Use the `solve_device()` path (which is automatically used by the solver integration) for production.

2. **Test tolerance**: The validation tests compare HYPRE and Multigrid solutions with tolerance 1e-6. Both solvers should produce nearly identical results for well-posed problems.

## References

- [HYPRE Documentation](https://hypre.readthedocs.io/)
- [HYPRE GitHub Repository](https://github.com/hypre-space/hypre)
- [HYPRE Struct Interface](https://hypre.readthedocs.io/en/latest/ch-struct.html)
- Falgout, R.D. and Yang, U.M. "hypre: A Library of High Performance Preconditioners." ICCS 2002.
