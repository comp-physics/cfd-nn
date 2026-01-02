# Poisson Solver Guide

This document provides a comprehensive overview of all Poisson solvers available in NNCFD, including when to use each, their requirements, and performance characteristics.

## Quick Reference

| Solver | Best For | Speed (128³) | GPU | Stretched Grid | All BCs |
|--------|----------|--------------|-----|----------------|---------|
| **FFT+cuSPARSE** | Channel/duct flows | **0.72 ms** (26x) | ✓ | y only | Periodic x/z required |
| **HYPRE PFMG** | Stretched grids | 1.0-4.2 ms (4-18x) | ✓ | ✓ | ✓ |
| **Native Multigrid** | General/fallback | 18.5 ms (1x baseline) | ✓ | Limited | ✓ |
| **SOR** | Testing only | ~200 ms | ✗ | ✓ | ✓ |

*Speedup relative to Native Multigrid baseline at 128³ uniform grid*

---

## 1. FFT-Hybrid Solver (FFTPoissonSolver)

### Overview
A direct solver using 2D FFT in periodic directions (x,z) combined with batched tridiagonal solves in the wall-normal direction (y). Uses cuFFT for FFT and cuSPARSE for tridiagonal systems.

### When to Use
- **Ideal for**: Channel flow, duct flow, any configuration with periodic x and z
- **Required**: Periodic boundary conditions in x AND z directions
- **Required**: Uniform grid spacing in x AND z directions
- **Supported**: Neumann or Dirichlet BC in y direction
- **Supported**: Stretched (non-uniform) grid in y direction

### When NOT to Use
- Non-periodic boundaries in x or z
- Stretched/non-uniform spacing in x or z directions
- 2D simulations (currently 3D only)

### Performance
| Grid Size | Time per Solve | Speedup vs MG |
|-----------|----------------|---------------|
| 32³ | 0.23 ms | 49x |
| 128³ | 0.72 ms | 26x |

### How to Enable
```bash
./channel --config myconfig.cfg --use_fft
```

Or in config file:
```
use_fft = true
```

### Implementation Details

**Files:**
- `include/poisson_solver_fft.hpp`
- `src/poisson_solver_fft.cpp`

**Algorithm:**
1. **Pack RHS**: Reorganize from ghost-cell layout `[k][j][i]` to FFT layout `[j][i][k]`
2. **Nullspace handling**: Subtract mean(RHS) to ensure compatibility
3. **Forward FFT**: 2D R2C transform in x-z for each y-plane (cuFFT batched)
4. **Tridiagonal solve**: For each Fourier mode (kx, kz), solve 1D system in y
5. **Inverse FFT**: 2D C2R transform back to physical space
6. **Unpack**: Reorganize back to ghost-cell layout with FFT normalization

**Tridiagonal System:**
For each Fourier mode (kx, kz), solve:
```
[-aS(j)] * p(j-1) + [aS(j) + aN(j) + λx(kx) + λz(kz)] * p(j) + [-aN(j)] * p(j+1) = rhs_hat(j)
```

Where:
- `λx(kx) = (2 - 2*cos(2π*kx/Nx)) / dx²` — x-direction eigenvalue
- `λz(kz) = (2 - 2*cos(2π*kz/Nz)) / dz²` — z-direction eigenvalue
- `aS(j) = 1/(dy_south * dy_center)` — south coefficient (supports stretching)
- `aN(j) = 1/(dy_north * dy_center)` — north coefficient (supports stretching)

**Zero Mode Handling:**
The (kx=0, kz=0) mode has a singular system (nullspace). We pin p(j=0) = 0 by modifying the first row.

**Tridiagonal Solver Options:**
- **cuSPARSE** (default): Uses `cusparseZgtsv2StridedBatch` for batched complex tridiagonal solve. 6x faster than custom Thomas.
- **Custom Thomas**: Fallback implementation using global workspace arrays.

**Memory Layout:**
```cpp
rhs_packed_[Nx * Ny * Nz]           // Packed real RHS
p_packed_[Nx * Ny * Nz]             // Packed real solution
rhs_hat_[Nx * Nz_complex * Ny]      // Complex FFT output
p_hat_[Nx * Nz_complex * Ny]        // Complex solution in Fourier space
lambda_x_[Nx]                       // x-direction eigenvalues
lambda_z_[Nz/2+1]                   // z-direction eigenvalues
tri_dl_[n_modes * Ny]               // cuSPARSE lower diagonal
tri_d_[n_modes * Ny]                // cuSPARSE main diagonal
tri_du_[n_modes * Ny]               // cuSPARSE upper diagonal
```

---

## 2. HYPRE PFMG Solver (HyprePoissonSolver)

### Overview
GPU-accelerated parallel multigrid using HYPRE's structured grid (Struct) interface. Uses PFMG (Parallel Semicoarsening Multigrid) with native CUDA backend.

### When to Use
- **Ideal for**: Stretched grids, complex boundary conditions
- **Supports**: All boundary condition combinations
- **Supports**: Fully stretched grids in all directions
- **Future**: MPI parallelization for multi-GPU

### When NOT to Use
- When FFT solver is applicable (FFT is faster for periodic x/z)
- Current tuning is optimized for stretched grids; may need adjustment for uniform

### Performance
| Configuration | Time per Solve | Speedup vs MG |
|---------------|----------------|---------------|
| 32³ stretched | 1.0 ms | 11x |
| 32³ uniform | 1.0 ms | 11x |
| 128³ stretched | ~4.2 ms | 4x |

### How to Enable
```bash
./channel --config myconfig.cfg --use_hypre
```

Or in config file:
```
use_hypre = true
```

### Implementation Details

**Files:**
- `include/poisson_solver_hypre.hpp`
- `src/poisson_solver_hypre.cpp`
- `docs/HYPRE_POISSON_SOLVER.md` (detailed documentation)

**HYPRE Objects:**
```cpp
HYPRE_StructGrid grid_;        // Domain decomposition
HYPRE_StructStencil stencil_;  // 7-point stencil
HYPRE_StructMatrix A_;         // System matrix
HYPRE_StructVector b_, x_;     // RHS and solution vectors
HYPRE_StructSolver solver_;    // PFMG solver
```

**PFMG Configuration (Optimized for Stretched Grids):**
```cpp
MaxLevels = log2(min_dim / 64) + 1  // Coarsen until ~64 cells per dimension
RelaxType = 1                        // Weighted Jacobi (fully parallel)
NumPreRelax = 1                      // Single pre-smoothing sweep
NumPostRelax = 0                     // Skip post-smoothing
SkipRelax = 1                        // Skip fine-level relaxation
RAPType = 0                          // Galerkin RAP (required for stability)
```

**Stencil Coefficients (Variable for Stretched Grids):**
```cpp
// For each cell (i,j,k):
aW = 1.0 / (dx_west * dx_center)     // West neighbor
aE = 1.0 / (dx_east * dx_center)     // East neighbor
aS = 1.0 / (dy_south * dy_center)    // South neighbor
aN = 1.0 / (dy_north * dy_center)    // North neighbor
aB = 1.0 / (dz_back * dz_center)     // Back neighbor
aF = 1.0 / (dz_front * dz_center)    // Front neighbor
aC = -(aW + aE + aS + aN + aB + aF)  // Center (diagonal)
```

**Boundary Condition Handling:**
- **Periodic**: `HYPRE_StructGridSetPeriodic(grid, periodic)`
- **Neumann**: Zero coefficient for boundary neighbor, adjust diagonal
- **Dirichlet**: Move known value to RHS, set diagonal to 1

**Singular System Handling:**
For pure Neumann/periodic problems, pin cell (0,0,0) to zero to remove nullspace.

**Device Memory:**
Uses `cudaMallocManaged()` for unified memory that works with both OpenMP target and HYPRE's CUDA backend.

---

## 3. Native Multigrid Solver (MultigridPoissonSolver)

### Overview
Geometric multigrid with V-cycle and SOR smoothing. Implemented with OpenMP target offload for GPU acceleration.

### When to Use
- **Default solver**: Works for all configurations
- **Fallback**: When specialized solvers don't apply
- **General case**: Mixed boundary conditions, moderate stretching

### When NOT to Use
- When FFT or HYPRE is applicable (both are faster)
- Heavily stretched grids (HYPRE handles better)

### Performance
| Grid Size | Time per Solve | Notes |
|-----------|----------------|-------|
| 32³ | 11.2 ms | Baseline |
| 128³ | 18.5 ms | Baseline |

### Implementation Details

**Files:**
- `include/poisson_solver_multigrid.hpp`
- `src/poisson_solver_multigrid.cpp` (55 KB, largest solver file)

**Grid Hierarchy:**
```cpp
struct GridLevel {
    int Nx, Ny, Nz;           // Grid dimensions
    double dx, dy, dz;         // Grid spacing (doubles each level)
    Mesh mesh;                 // Mesh object with coordinates
    ScalarField u, f, r;       // Solution, RHS, residual
};
```

**Coarsening Strategy:**
- 2:1 coarsening in all directions: `Nx_coarse = Nx_fine / 2`
- Continue until `min(Nx, Ny, Nz) <= 8`
- Grid spacing doubles: `dx_coarse = 2 * dx_fine`

**V-Cycle Algorithm:**
```
vcycle(level):
    if level == coarsest:
        smooth(level, 100 iterations)  // Direct solve via relaxation
    else:
        smooth(level, nu1 iterations)   // Pre-smoothing (default nu1=2)
        compute_residual(level)
        restrict(level → level+1)        // Full-weighting restriction
        vcycle(level+1)                  // Recursive coarse solve
        prolongate(level+1 → level)      // Bilinear interpolation
        smooth(level, nu2 iterations)    // Post-smoothing (default nu2=2)
```

**Smoothing (SOR with Red-Black Ordering):**
```cpp
// Red points: (i+j+k) % 2 == 0
// Black points: (i+j+k) % 2 == 1
u_new = (1-ω)*u_old + ω*(rhs - Σ neighbors) / diagonal
```
Default ω = 1.8 for optimal convergence.

**Restriction (Full Weighting):**
```cpp
// 2D: Average 4 fine cells
r_coarse[i,j] = 0.25 * (r_fine[2i,2j] + r_fine[2i+1,2j] +
                        r_fine[2i,2j+1] + r_fine[2i+1,2j+1])
// 3D: Average 8 fine cells
```

**Prolongation (Bilinear Interpolation):**
```cpp
// Inject coarse values, interpolate at fine-only locations
u_fine[2i,2j] = u_coarse[i,j]                    // Injection
u_fine[2i+1,2j] = 0.5*(u_coarse[i,j] + u_coarse[i+1,j])  // x-interp
// Similar for y and xy combinations
```

**GPU Implementation:**
- Device-resident pointers at each level: `u_ptrs_[level]`, `f_ptrs_[level]`, `r_ptrs_[level]`
- All operations use `#pragma omp target teams distribute parallel for`
- Owner-computes pattern for restriction/prolongation (commit b7fe2b4)

---

## 4. SOR Solver (PoissonSolver - Base Class)

### Overview
Simple Successive Over-Relaxation iterative solver. Primarily used as a smoother within multigrid or for validation/testing.

### When to Use
- **Testing**: Validate other solvers against known-correct implementation
- **Smoothing**: Used internally by Multigrid solver
- **Small problems**: When setup cost of other solvers isn't justified

### When NOT to Use
- Production simulations (too slow)
- Large grids (O(N²) complexity)

### Performance
| Grid Size | Time per Solve | Notes |
|-----------|----------------|-------|
| 32³ | ~50 ms | Very slow |
| 128³ | ~200 ms | Impractical |

### Implementation Details

**Files:**
- `include/poisson_solver.hpp`
- `src/poisson_solver.cpp`

**Algorithm:**
Standard SOR with red-black ordering:
```cpp
for iter = 1 to max_iter:
    // Red sweep
    for all (i,j,k) where (i+j+k) % 2 == 0:
        u[i,j,k] = (1-ω)*u[i,j,k] + ω*gauss_seidel_update(i,j,k)

    // Black sweep
    for all (i,j,k) where (i+j+k) % 2 == 1:
        u[i,j,k] = (1-ω)*u[i,j,k] + ω*gauss_seidel_update(i,j,k)

    if residual < tolerance:
        break
```

**Optimal Relaxation Parameter:**
For Poisson equation on uniform grid:
```
ω_optimal ≈ 2 / (1 + sin(π*h))
```
Default: ω = 1.5 (general), ω = 1.8 (within multigrid)

---

## Solver Selection Logic

The solver is selected in `src/solver.cpp` with the following priority:

```cpp
// 1. Check if FFT is applicable and enabled
if (use_fft && periodic_xz && uniform_xz && is_3D) {
    use FFTPoissonSolver  // Fastest option
}
// 2. Check if HYPRE is enabled
else if (use_hypre && hypre_available) {
    use HyprePoissonSolver  // Good for stretched grids
}
// 3. Default to native multigrid
else {
    use MultigridPoissonSolver  // Always works
}
```

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

## Performance Summary

### 128³ Uniform Grid (H200 GPU)

| Solver | Time (ms) | Relative | When to Use |
|--------|-----------|----------|-------------|
| FFT+cuSPARSE | 0.72 | 1.0x (fastest) | Periodic x/z channel flows |
| HYPRE PFMG | 4.2* | 5.8x | Stretched grids, complex BCs |
| Native MG | 18.5 | 25.7x | General fallback |
| SOR | ~200 | ~280x | Testing only |

*HYPRE requires tuning for uniform grids; shown value is for stretched grids

### 32³ Grid (H200 GPU)

| Solver | Time (ms) | Relative |
|--------|-----------|----------|
| FFT+cuSPARSE | 0.23 | 1.0x |
| HYPRE PFMG | 1.0 | 4.3x |
| Native MG | 11.2 | 49x |

### Scaling Characteristics

| Solver | Complexity | Memory | Notes |
|--------|------------|--------|-------|
| FFT | O(N log N) | O(N) extra for FFT workspace | Optimal for periodic |
| HYPRE | O(N) | O(N) HYPRE internal buffers | Best parallel scaling |
| Multigrid | O(N) | O(N) for hierarchy | Good general purpose |
| SOR | O(N²) | O(1) | Not practical for large N |

---

## Configuration Reference

### Command Line Flags
```bash
--use_fft        # Enable FFT solver (if applicable)
--use_hypre      # Enable HYPRE PFMG solver
```

### Config File Options
```ini
poisson_tol = 1e-6       # Convergence tolerance
poisson_max_iter = 10    # Max iterations/V-cycles per solve
poisson_omega = 1.8      # SOR relaxation parameter
use_fft = true           # Enable FFT solver
use_hypre = true         # Enable HYPRE solver
```

### Automatic Selection
If no solver is explicitly enabled, the code automatically selects:
1. FFT if periodic x/z and uniform spacing
2. Native Multigrid otherwise
