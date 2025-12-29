# CLAUDE.md - Project Context for AI Assistants

This document provides essential context for working with the CFD-NN codebase.

## Project Overview

CFD-NN is a GPU-accelerated incompressible Navier-Stokes solver with neural network turbulence models. It uses a **unified CPU/GPU codebase** via OpenMP target offloading - there are no separate CPU and GPU implementations.

## Build Commands

```bash
# GPU build (requires nvc++ from NVIDIA HPC SDK)
mkdir -p build && cd build
cmake .. -DUSE_GPU_OFFLOAD=ON
make -j8

# CPU-only build
mkdir -p build_cpu && cd build_cpu
cmake .. -DUSE_GPU_OFFLOAD=OFF
make -j8

# Run local CI tests
./scripts/run_ci_local.sh           # Auto-detect GPU, run fast+medium tests
./scripts/run_ci_local.sh --cpu     # Force CPU build
./scripts/run_ci_local.sh gpu       # GPU-specific tests only
./scripts/run_ci_local.sh fast      # Fast tests only (~1 min)
```

## Directory Structure

```
include/          # Headers (public API)
src/              # Implementation files
app/              # Application entry points (channel, duct, taylor_green_3d, etc.)
tests/            # Unit and integration tests (test_*.cpp)
examples/         # Ready-to-run examples with run.sh scripts
scripts/          # Build, CI, profiling scripts
data/models/      # Neural network model weights
```

## Key Architecture Patterns

### Unified CPU/GPU Code via OpenMP Target

The codebase uses OpenMP target directives for GPU offloading. When `USE_GPU_OFFLOAD` is defined, the same code runs on GPU; otherwise it runs on CPU:

```cpp
#pragma omp target teams distribute parallel for collapse(2) \
    map(present: ptr1, ptr2) if(target: use_gpu)
for (int j = 0; j < Ny; ++j) {
    for (int i = 0; i < Nx; ++i) {
        // Kernel code - runs on GPU or CPU
    }
}
```

**Key pattern**: Use `map(present: ...)` for data that has been persistently mapped via `target enter data`. The solver manages GPU memory lifetime.

### Device Views for GPU Data

The `SolverDeviceView` and `TurbulenceDeviceView` structs hold pointers to GPU-resident data:

```cpp
struct SolverDeviceView {
    double* u_face;      // Velocity at x-faces
    double* v_face;      // Velocity at y-faces
    double* w_face;      // Velocity at z-faces (3D)
    double* pressure;    // Cell-centered pressure
    int u_stride;        // Row stride for u array
    // ... more fields
};
```

These are HOST pointers that have been mapped to GPU via `target enter data`. Pass them to kernels with `map(present: view)`.

### 2D/3D Support

The solver supports both 2D and 3D simulations:
- `mesh.is2D()` returns true for 2D cases (Nz == 1)
- 3D loops use `collapse(3)` instead of `collapse(2)`
- Z-direction boundary conditions default to periodic

**Important**: The `Config` struct does NOT have `Nz`, `z_min`, `z_max` members. For 3D applications, use local variables:

```cpp
// In app/main_duct.cpp:
int Nz = 32;           // Local variable, not config.Nz
double z_min = -1.0;
double z_max = 1.0;
Mesh mesh(Nx, Ny, Nz, x_min, x_max, y_min, y_max, z_min, z_max);
```

## Turbulence Models

Available via `TurbulenceModelType` enum in `config.hpp`:

| Model | Type | Description |
|-------|------|-------------|
| `None` | Algebraic | Laminar (no turbulence) |
| `Baseline` | Algebraic | Van Driest mixing length |
| `GEP` | Algebraic | Gene Expression Programming |
| `NNMLP` | Algebraic | Neural network MLP |
| `NNTBNN` | Algebraic | Tensor Basis Neural Network |
| `SSTKOmega` | Transport | SST k-ω with linear Boussinesq |
| `KOmega` | Transport | Standard k-ω (Wilcox 1988) |
| `EARSM_WJ` | Transport | SST k-ω + Wallin-Johansson EARSM |
| `EARSM_GS` | Transport | SST k-ω + Gatski-Speziale EARSM |
| `EARSM_Pope` | Transport | SST k-ω + Pope quadratic model |

Create via factory: `create_turbulence_model(TurbulenceModelType::SSTKOmega)`

## Timing Infrastructure

Use `TIMED_SCOPE` for performance profiling:

```cpp
#include "timing.hpp"

void my_function() {
    TIMED_SCOPE("my_function");  // Records time to TimingStats singleton
    // ... code ...
}

// At end of program:
TimingStats::instance().print_summary();
```

### GPU Utilization Tracking

For GPU builds, the timing system categorizes operations as GPU or CPU:

```cpp
// Check GPU utilization (for CI validation)
auto& stats = TimingStats::instance();
double gpu_ratio = stats.gpu_utilization_ratio();  // 0.0 to 1.0
stats.assert_gpu_dominant(0.7, "solver test");     // Throws if < 70%
```

Categories ending in `_gpu` are explicitly GPU. Core solver operations (poisson, convective, diffusive, etc.) are classified as GPU when `USE_GPU_OFFLOAD` is defined.

## Common Gotchas

### Bash Scripting with `set -e`

**Problem**: Post-increment `((VAR++))` returns exit status 1 when VAR is 0:
```bash
set -e
VIOLATIONS=0
((VIOLATIONS++))   # EXITS! Returns 1 (old value) which is falsy
```

**Solution**: Use explicit assignment:
```bash
VIOLATIONS=$((VIOLATIONS + 1))   # Always succeeds
```

### Solver Initialization

Always initialize before solving:
```cpp
RANSSolver solver(mesh, config);
solver.set_body_force(-config.dp_dx, 0.0);  // Required for driven flows!
solver.initialize_uniform(0.1, 0.0);
solver.set_velocity_bc(bc);
solver.step();
```

### GPU Memory Management

Data must be mapped before use:
```cpp
#pragma omp target enter data map(to: data[0:size])
// ... use in kernels with map(present: data) ...
#pragma omp target exit data map(delete: data[0:size])
```

The solver handles this automatically for its arrays. Use `OmpDeviceBuffer` RAII wrapper for custom buffers.

## Testing

### Key Test Files

| File | Purpose |
|------|---------|
| `test_physics_validation.cpp` | Gold-standard Navier-Stokes validation |
| `test_taylor_green.cpp` | Energy decay verification |
| `test_taylor_green_3d.cpp` | 3D energy decay verification |
| `test_all_turbulence_models_smoke.cpp` | All 9 turbulence models smoke test |
| `test_gpu_utilization.cpp` | Verify GPU dominates compute (CI) |
| `test_cpu_gpu_consistency.cpp` | CPU/GPU produce identical results |

### Running Tests

```bash
cd build
ctest --output-on-failure          # All tests
./test_solver                       # Single test
./test_physics_validation           # Physics validation suite
```

## Applications

| Binary | Description |
|--------|-------------|
| `channel` | 2D/3D channel flow (various configs) |
| `periodic_hills` | Periodic hills benchmark |
| `duct` | 3D square duct flow |
| `taylor_green_3d` | 3D Taylor-Green vortex decay |

Run examples via:
```bash
./examples/channel/run.sh retau180_sst
./examples/09_taylor_green_3d/run.sh
```

## Profiling

### NVIDIA Nsight Systems

```bash
nsys profile -o profile_output ./channel --config channel/retau180_sst.cfg
nsys stats profile_output.nsys-rep
```

### Built-in Timing

```bash
./scripts/profile_gpu_utilization.sh --builtin channel retau180_sst 5000
```

## Config File Format

Configuration uses `.cfg` files with key-value pairs:

```ini
# Grid
Nx = 64
Ny = 128

# Domain
Lx = 4.0
Ly = 2.0

# Physics
Re = 180.0
nu = 0.00556

# Turbulence
model = sst    # Options: none, baseline, gep, sst, komega, earsm_wj, nn_mlp, nn_tbnn

# Solver
tol = 1e-10
max_iter = 10000
```

## Performance Notes

- **Poisson solver dominates**: Multigrid solver is 99%+ of GPU time
- **Smoothing kernels**: Red-black Gauss-Seidel is the bottleneck (~76%)
- **Memory transfers**: Negligible (<3%) - unified memory model works well
- **3D scaling**: Memory grows as O(N³), performance degrades for large N

## See Also

- `.cursorrules` - Detailed testing requirements and CI guidelines
- `README.md` - User-facing documentation
- `examples/README.md` - Example case documentation
- `docs/` - Additional documentation
