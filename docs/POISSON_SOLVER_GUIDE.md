# Poisson Solver Guide

This document provides a comprehensive overview of all Poisson solvers available in NNCFD, including when to use each, their requirements, and performance characteristics.

## Quick Reference

| Solver | Best For | GPU | Stretched Grid | Requirements |
|--------|----------|-----|----------------|--------------|
| **FFT** | 3D channel flows | Yes | y only | Periodic x AND z, uniform dx/dz |
| **FFT2D** | 2D channel flows | Yes | y only | Periodic x, uniform dx, 2D mesh |
| **FFT1D** | 3D duct flows | Yes | Non-periodic dirs | Periodic x OR z (exactly one), 3D |
| **HYPRE PFMG** | Stretched grids | Yes | All dirs | USE_HYPRE build |
| **Native Multigrid** | General fallback | Yes | Limited | Always available |
| **SOR** | Testing only | No | Yes | Always available |

## Solver Selection

The solver is auto-selected with the following priority:

```text
FFT (3D) → FFT2D (2D mesh) → FFT1D (3D 1-periodic) → HYPRE → MG
```

Or explicitly select via command line:
```bash
./channel --poisson_solver fft      # Force FFT (3D)
./channel --poisson_solver fft2d    # Force FFT2D (2D mesh)
./channel --poisson_solver fft1d    # Force FFT1D (3D)
./channel --poisson_solver hypre    # Force HYPRE
./channel --poisson_solver mg       # Force Multigrid
```

---

## 1. FFT Solver (FFTPoissonSolver) - 3D

### Overview
A direct solver using 2D FFT in periodic directions (x,z) combined with batched tridiagonal solves in the wall-normal direction (y). Uses cuFFT for FFT and cuSPARSE for tridiagonal systems.

### When to Use
- **Ideal for**: 3D channel flow, any 3D configuration with periodic x AND z
- **Required**: Periodic boundary conditions in x AND z directions
- **Required**: Uniform grid spacing in x AND z directions
- **Supported**: Neumann or Dirichlet BC in y direction
- **Supported**: Stretched (non-uniform) grid in y direction

### When NOT to Use
- Non-periodic boundaries in x or z
- Stretched/non-uniform spacing in x or z directions
- 2D simulations (use FFT2D instead)

### Algorithm
1. **Pack RHS**: Reorganize from ghost-cell layout `[k][j][i]` to FFT layout `[j][i][k]`
2. **Nullspace handling**: Subtract mean(RHS) to ensure compatibility
3. **Forward FFT**: 2D R2C transform in x-z for each y-plane (cuFFT batched)
4. **Tridiagonal solve**: For each Fourier mode (kx, kz), solve 1D system in y
5. **Inverse FFT**: 2D C2R transform back to physical space
6. **Unpack**: Reorganize back to ghost-cell layout with FFT normalization

### Implementation Files
- `include/poisson_solver_fft.hpp`
- `src/poisson_solver_fft.cpp`

---

## 2. FFT2D Solver (FFT2DPoissonSolver) - 2D Mesh

### Overview
A direct solver for 2D meshes (Nz=1) using 1D FFT in x combined with batched tridiagonal solves in y. Uses cuFFT for FFT and cuSPARSE for tridiagonal systems. Optimal for 2D channel flows.

### When to Use
- **Ideal for**: 2D channel flow simulations
- **Required**: 2D mesh (Nz = 1)
- **Required**: Periodic boundary conditions in x
- **Required**: Uniform grid spacing in x
- **Supported**: Neumann or Dirichlet BC in y direction
- **Supported**: Stretched (non-uniform) grid in y direction

### When NOT to Use
- 3D simulations (use FFT or FFT1D instead)
- Non-periodic boundaries in x

### Algorithm
1. **Pack RHS**: Reorganize from ghost-cell layout to contiguous x-lines
2. **Nullspace handling**: Subtract mean(RHS) for Neumann-Neumann singularity
3. **Forward FFT**: 1D R2C transform along x (batched over y)
4. **Tridiagonal solve**: For each mode m, solve (d²/dy² - λ_x[m])*p_hat = rhs_hat
5. **Inverse FFT**: 1D C2R transform back to physical space
6. **Unpack**: Reorganize back to ghost-cell layout with BCs

### Known Issues
FFT2D may have flux conservation issues in some configurations (~7.6e-3 vs 1e-3 threshold). If tests fail, the solver will automatically fall back to HYPRE or MG.

### Implementation Files
- `include/poisson_solver_fft2d.hpp`
- `src/poisson_solver_fft2d.cpp`

---

## 3. FFT1D Solver (FFT1DPoissonSolver) - 3D with One Periodic Direction

### Overview
A solver for 3D cases with exactly one periodic direction. Uses 1D FFT in the periodic direction combined with 2D Helmholtz solves for each Fourier mode.

### When to Use
- **Ideal for**: 3D duct flow (periodic x, walls in y and z)
- **Required**: Exactly one of x or z is periodic
- **Required**: Uniform spacing in the periodic direction
- **Required**: 3D mesh (Nz > 1)
- **Supported**: Neumann or Dirichlet BC in non-periodic directions

### When NOT to Use
- 2D simulations (use FFT2D instead)
- Both x and z periodic (use FFT instead)
- Neither x nor z periodic (use HYPRE or MG)

### Algorithm
1. **Pack RHS**: Reorganize from ghost layout to contiguous lines along periodic direction
2. **Forward FFT**: 1D R2C FFT along periodic direction (batched over remaining 2D plane)
3. **2D Helmholtz solve**: For each mode m, solve (L_yz + λ[m]*I) p_hat = rhs_hat
4. **Inverse FFT**: 1D C2R transform back to physical space
5. **Unpack**: Reorganize back to ghost-cell layout

### Implementation Files
- `include/poisson_solver_fft1d.hpp`
- `src/poisson_solver_fft1d.cpp`

---

## 4. HYPRE PFMG Solver (HyprePoissonSolver)

### Overview
GPU-accelerated parallel multigrid using HYPRE's structured grid (Struct) interface. Uses PFMG (Parallel Semicoarsening Multigrid) with native CUDA backend.

### When to Use
- **Ideal for**: Stretched grids in all directions, complex boundary conditions
- **Supports**: All boundary condition combinations
- **Supports**: Fully stretched grids in all directions
- **Future**: MPI parallelization for multi-GPU

### When NOT to Use
- When FFT/FFT2D/FFT1D solver is applicable (FFT variants are faster for periodic cases)

### How to Enable
Requires building with `USE_HYPRE=ON`:
```bash
CC=nvc CXX=nvc++ cmake .. -DUSE_GPU_OFFLOAD=ON -DUSE_HYPRE=ON
```

### PFMG Configuration

| Parameter | Value | Reason |
|-----------|-------|--------|
| MaxLevels | 2 | Limits coarsening depth for efficiency |
| RelaxType | 1 (Weighted Jacobi) | Fully parallel, GPU-friendly |
| NumPreRelax | 1 | Minimal pre-smoothing |
| NumPostRelax | 0 | Skip post-smoothing |
| SkipRelax | 1 | Skip fine grid relaxation |

### Implementation Files
- `include/poisson_solver_hypre.hpp`
- `src/poisson_solver_hypre.cpp`
- See `docs/HYPRE_POISSON_SOLVER.md` for detailed documentation

---

## 5. Native Multigrid Solver (MultigridPoissonSolver)

### Overview
Geometric multigrid with V-cycle and Chebyshev smoothing. Implemented with OpenMP target offload for GPU acceleration. Features **V-cycle CUDA Graph capture** for minimal kernel launch overhead on NVIDIA GPUs.

### When to Use
- **Default solver**: Works for all configurations
- **Fallback**: When specialized solvers don't apply
- **General case**: Mixed boundary conditions, moderate stretching

### When NOT to Use
- When FFT variants or HYPRE is applicable (both are faster for their specific cases)
- Heavily stretched grids (HYPRE handles better)

### V-Cycle Algorithm
```
vcycle(level):
    if level == coarsest:
        smooth(level, 8 iterations)      // Direct solve via Chebyshev
    else:
        smooth(level, nu1 iterations)    // Pre-smoothing (default nu1=3)
        compute_residual(level)
        restrict(level → level+1)        // Full-weighting restriction
        vcycle(level+1)                  // Recursive coarse solve
        prolongate(level+1 → level)      // Bilinear interpolation
        smooth(level, nu2 iterations)    // Post-smoothing (default nu2=1)
```

### V-cycle CUDA Graph Optimization (GPU Builds)

On NVIDIA GPUs with NVHPC compiler, the MG solver captures the **entire V-cycle kernel sequence as a CUDA Graph**. This eliminates per-kernel launch overhead and synchronization costs.

**How it works:**
1. On first solve, the V-cycle is executed normally while CUDA captures all kernel launches
2. The captured graph is stored and validated
3. Subsequent solves replay the graph with a single `cudaGraphLaunch()` call
4. Graph is automatically recaptured if boundary conditions or parameters change

**Performance impact:**
- Eliminates ~37% of GPU API time spent in `cudaStreamSynchronize` (measured with Nsys)
- Reduces kernel launch overhead from O(levels × kernels) to O(1)
- Most beneficial for iterative solves with many V-cycles

**Configuration:**
```cpp
// Enabled by default in GPU builds with NVHPC
config.poisson_use_vcycle_graph = true;   // Enable (default)
config.poisson_use_vcycle_graph = false;  // Disable for debugging
```

**Requirements:**
- NVIDIA GPU with CUDA support
- NVHPC compiler (uses `ompx_get_cuda_stream()` for stream access)
- 3D mesh (2D meshes use non-graphed path)

**Fallback behavior:**
- Non-NVHPC compilers: Automatic fallback to standard OpenMP target path
- 2D meshes: V-cycle graph disabled (not fully optimized for 2D)
- Graph capture failure: Falls back to non-graphed path with warning

### Chebyshev Smoothing

The MG solver uses Chebyshev polynomial smoothing instead of Jacobi/Gauss-Seidel:

**Advantages:**
- Better convergence rate than weighted Jacobi
- Fully parallel (no red-black ordering needed)
- Optimal for GPU execution

**Eigenvalue bounds:**
- For the 7-point discrete Laplacian, eigenvalues of D⁻¹A are in (0, 2)
- Conservative bounds [0.05, 1.95] used for numerical stability
- Defined as `CHEBYSHEV_LAMBDA_MIN` and `CHEBYSHEV_LAMBDA_MAX` constants

### Implementation Files
- `include/poisson_solver_multigrid.hpp`
- `src/poisson_solver_multigrid.cpp`
- `src/mg_cuda_kernels.cpp` (CUDA Graph implementation)

---

## 6. SOR Solver (PoissonSolver - Base Class)

### Overview
Simple Successive Over-Relaxation iterative solver. Primarily used as a smoother within multigrid or for validation/testing.

### When to Use
- **Testing**: Validate other solvers against known-correct implementation
- **Smoothing**: Used internally by Multigrid solver
- **Small problems**: When setup cost of other solvers isn't justified

### When NOT to Use
- Production simulations (too slow - O(N²) complexity)
- Large grids

### Implementation Files
- `include/poisson_solver.hpp`
- `src/poisson_solver.cpp`

---

## Boundary Condition Implementation

### Ghost Cell Layout
All solvers use a ghost-cell approach with `Nghost = 1`:
```
Array size: (Nx + 2*Ng) × (Ny + 2*Ng) × (Nz + 2*Ng)
Interior:   [Ng : Ng+Nx] × [Ng : Ng+Ny] × [Ng : Ng+Nz]
```

### Periodic BC
```cpp
// x-direction
p[0][j][k] = p[Nx][j][k]      // Low ghost = high interior
p[Nx+1][j][k] = p[1][j][k]    // High ghost = low interior
```

### Neumann BC (dp/dn = 0)
```cpp
// y-direction (wall at y_lo)
p[i][0][k] = p[i][1][k]       // Zero gradient: ghost = first interior
```

### Dirichlet BC (p = p_val)
```cpp
// y-direction (p = 0 at wall)
p[i][0][k] = 2*p_val - p[i][1][k]   // Linear extrapolation to enforce value
```

---

## Singular System Handling

For pure Neumann or fully periodic problems, the Poisson equation has a nullspace (constant can be added to solution). All solvers handle this by:

1. **Mean subtraction**: Ensure `mean(RHS) = 0` for compatibility
2. **Pin one value**: Set `p(0,0,0) = 0` to fix the constant

---

## Configuration Reference

### Command Line Flags
```bash
--poisson_solver TYPE   # Solver type: auto, fft, fft2d, fft1d, hypre, mg
```

### Config File Options
```ini
poisson_solver = auto    # auto, fft, fft2d, fft1d, hypre, mg
poisson_tol = 1e-6       # Convergence tolerance
poisson_max_iter = 10    # Max iterations/V-cycles per solve
```

### Automatic Selection
If `poisson_solver = auto` (default), the code automatically selects:
1. **FFT** if 3D + periodic x/z + uniform dx/dz
2. **FFT2D** if 2D mesh + periodic x + uniform dx
3. **FFT1D** if 3D + exactly one of (x,z) periodic + uniform in that direction
4. **HYPRE** if available and FFT variants don't apply
5. **MG** as final fallback
