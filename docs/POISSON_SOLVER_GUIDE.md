# Poisson Solver Guide

This document provides a comprehensive reference for all Poisson solver backends in NNCFD, including selection logic, configuration, performance characteristics, and troubleshooting.

## Overview

The pressure Poisson equation is solved every time step (or every RK stage) as part of the fractional-step projection method. NNCFD provides 6 solver backends:

| Solver | Type | GPU | Stretched Grid | Requirements |
|--------|------|-----|----------------|--------------|
| **FFT** | Direct (3D cuFFT) | Yes | y only | Periodic x AND z, uniform dx/dz |
| **FFT2D** | Direct (2D cuFFT) | Yes | y only | Periodic x, uniform dx, 2D mesh |
| **FFT1D** | Hybrid (1D FFT + 2D MG) | Yes | Non-periodic dirs | Exactly one periodic dir, 3D |
| **HYPRE PFMG** | Iterative (semicoarsening MG) | Yes (CUDA) | All dirs | `USE_HYPRE` build |
| **Native Multigrid** | Iterative (geometric MG) | Yes (OMP offload) | Yes (semi-coarsening) | Always available |
| **FFT_MPI** | Direct (distributed FFT) | Yes | y only | `USE_MPI` build |

### Configuration

Select a solver via config file or CLI:

```ini
poisson_solver = auto       # auto, fft, fft2d, fft1d, hypre, mg, fft_mpi
```

```bash
./channel --config flow.cfg --poisson_solver fft
```

With `auto` (the default), the solver is selected automatically based on boundary conditions and available backends.

---

## Auto-Selection Logic

When `poisson_solver = auto`, the following priority chain is evaluated:

```
FFT (3D) --> FFT2D (2D mesh) --> FFT1D (3D, 1 periodic dir) --> HYPRE --> MG
```

The table below shows which solver is selected for common configurations:

| Geometry | x BC | y BC | z BC | Nz | Selected Solver |
|----------|------|------|------|----|-----------------|
| Channel (3D) | Periodic | Wall | Periodic | >1 | FFT |
| Channel (2D) | Periodic | Wall | -- | 1 | FFT2D |
| Duct | Periodic | Wall | Wall | >1 | FFT1D |
| Box (all periodic) | Periodic | Periodic | Periodic | >1 | FFT |
| Cavity | Wall | Wall | Wall | >1 | HYPRE (if built) or MG |
| Any | Any | Any | Any | Any | MG (final fallback) |

SOR is never auto-selected; it must be explicitly requested.

---

## FFT Solvers

All FFT solvers are **direct** (non-iterative) and solve the Poisson equation to machine precision in a single pass. They use cuFFT for GPU-accelerated FFTs and cuSPARSE for batched tridiagonal solves.

### FFT (3D cuFFT)

**Requirements:** Periodic x AND z, uniform dx/dz, 3D mesh (Nz > 1).

**Stretched-y support:** The y-direction tridiagonal system uses the mesh's precomputed `yLap_lower`, `yLap_diag`, `yLap_upper` coefficients, which encode the D*G=L-consistent discrete Laplacian for arbitrarily stretched grids. This ensures the FFT solver produces results consistent with the staggered-grid operators used elsewhere in the code.

**Algorithm:**

1. **Pack RHS** -- reorganize from ghost-cell layout `[k][j][i]` to FFT layout `[j][i][k]`
2. **Nullspace handling** -- subtract volume-weighted mean of RHS to ensure solvability (see below)
3. **Forward FFT** -- 2D R2C transform in x-z for each y-plane (cuFFT batched)
4. **Tridiagonal solve** -- for each Fourier mode (kx, kz), solve 1D tridiagonal system in y using cuSPARSE
5. **Inverse FFT** -- 2D C2R transform back to physical space
6. **Unpack** -- reorganize back to ghost-cell layout with FFT normalization

**Volume-weighted mean subtraction:** On stretched grids, the discrete Laplacian L is not symmetric; the solvability condition requires the volume-weighted sum `sum(f * dyv[j]) = 0`, not the arithmetic mean `sum(f) = 0`. The FFT solver computes the volume-weighted sum using a device copy of `dyv` (face spacings). Without this, the direct solver amplifies the incompatible RHS component, causing catastrophic blow-up. The native MG solver is unaffected because iterative methods naturally project away incompatible components.

**Implementation files:**
- `include/poisson_solver_fft.hpp`
- `src/poisson_solver_fft.cpp`

### FFT2D (2D cuFFT)

**Requirements:** Periodic x, uniform dx, 2D mesh (Nz = 1).

Optimal for 2D channel flows. Uses 1D FFT in x combined with batched tridiagonal solves in y. Same stretched-y support as the 3D FFT solver (mesh yLap coefficients, volume-weighted mean subtraction).

**Algorithm:**

1. **Pack RHS** -- reorganize from ghost-cell layout to contiguous x-lines
2. **Nullspace handling** -- subtract volume-weighted mean for Neumann-Neumann singularity
3. **Forward FFT** -- 1D R2C transform along x (batched over y)
4. **Tridiagonal solve** -- for each mode m, solve `(d^2/dy^2 - lambda_x[m]) * p_hat = rhs_hat`
5. **Inverse FFT** -- 1D C2R transform back to physical space
6. **Unpack** -- reorganize back to ghost-cell layout with BCs

**Known issues:** FFT2D may have flux conservation issues in some configurations (~7.6e-3 vs 1e-3 threshold). If validation tests fail, the solver automatically falls back to HYPRE or MG.

**Implementation files:**
- `include/poisson_solver_fft2d.hpp`
- `src/poisson_solver_fft2d.cpp`

### FFT1D (1D FFT + 2D Helmholtz)

**Requirements:** Exactly one of x or z is periodic with uniform spacing in that direction, 3D mesh (Nz > 1).

Designed for 3D duct flows (periodic x, walls in y and z). Uses 1D FFT in the periodic direction combined with a 2D Helmholtz solve for each Fourier mode.

**Algorithm:**

1. **Pack RHS** -- reorganize from ghost layout to contiguous lines along periodic direction
2. **Forward FFT** -- 1D R2C FFT along periodic direction (batched over remaining 2D plane)
3. **2D Helmholtz solve** -- for each mode m, solve `(L_yz + lambda[m]*I) p_hat = rhs_hat` using an internal 2D multigrid solver
4. **Inverse FFT** -- 1D C2R transform back to physical space
5. **Unpack** -- reorganize back to ghost-cell layout

**Limitation:** The internal 2D MG solver uses uniform `1/(dy*dy)` coefficients, so FFT1D still requires uniform y spacing. It is NOT selected by auto-selection when the grid is stretched in y.

**Implementation files:**
- `include/poisson_solver_fft1d.hpp`
- `src/poisson_solver_fft1d.cpp`

---

## HYPRE PFMG

HYPRE's Parallel Semicoarsening Multigrid (PFMG) solver provides a GPU-accelerated alternative to the native multigrid. When built with CUDA support, HYPRE runs the entire V-cycle on the GPU via native CUDA kernels.

### Building with HYPRE

HYPRE is automatically downloaded and built via CMake FetchContent:

```bash
mkdir build_hypre && cd build_hypre
CC=nvc CXX=nvc++ cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_GPU_OFFLOAD=ON \
    -DUSE_HYPRE=ON
make -j8
```

CMake will:
1. Download HYPRE v2.31.0 from GitHub
2. Configure with CUDA support and unified memory
3. Build as a static library and link with NNCFD

**Build requirements:**
- NVIDIA HPC SDK (nvc/nvc++ with `-cuda` support)
- CUDA Toolkit
- CMake 3.16+ (for FetchContent)
- Git (for automatic download)

| CMake Option | Default | Description |
|--------------|---------|-------------|
| `USE_HYPRE` | OFF | Enable HYPRE PFMG solver |
| `CMAKE_CUDA_ARCHITECTURES` | 90 | CUDA compute capability (90 = H100/H200) |

If HYPRE is not available at runtime, the solver automatically falls back to native MG.

### PFMG Configuration

The PFMG solver is tuned for optimal single-GPU performance on CFD pressure Poisson problems:

| Parameter | Value | Reason |
|-----------|-------|--------|
| MaxLevels | 2 | Limits coarsening depth; reduces kernel launch overhead |
| RelaxType | 1 (Weighted Jacobi) | Fully parallel, GPU-friendly |
| NumPreRelax | 1 | Minimal pre-smoothing |
| NumPostRelax | 0 | Skip post-smoothing (stable with SkipRelax) |
| SkipRelax | 1 | Skip fine grid relaxation (~23% speedup) |
| RAPType | 0 (Galerkin) | Non-Galerkin (RAPType=1) causes instability |
| SetDxyz | (dx, dy, dz) | Anisotropic grid spacing hints for semicoarsening direction |

**Key tuning decisions:**
- `MaxLevels=2` limits PFMG to 2 levels (e.g., coarsest 64^3 for a 128^3 grid). PFMG semicoarsening can create up to 21 levels for 128^3; limiting depth reduces kernel launch overhead.
- `SkipRelax=1` provides ~23% speedup while maintaining convergence for smooth RHS typical in CFD.
- `NumPreRelax=0` causes numerical instability; at least 1 pre-relaxation sweep is required.
- Red-Black Gauss-Seidel (`RelaxType=2`) converges faster per iteration but has data dependencies that reduce GPU parallelism. Weighted Jacobi is preferred.

### Unified Memory Integration

The integration bridges OpenMP target offload memory (used by the main solver) with HYPRE's CUDA backend via unified (managed) memory:

```
GPU Device Memory
+-----------------------------+-----------------------------+
| OpenMP Device Memory        | CUDA Managed Memory         |
| (omp_target_alloc)          | (cudaMallocManaged)         |
|                             |                             |
| u, v, w, pressure, RHS     | rhs_device_, x_device_      |
+-----------------------------+-----------------------------+
       |                              |
       | OMP kernel with              | HYPRE CUDA
       | is_device_ptr                | operations
       v                              v
+-------------------------------------------------------------+
|              HYPRE PFMG GPU Solve                            |
+-------------------------------------------------------------+
```

Data transfer between OpenMP-mapped and managed memory happens on-device via OpenMP kernels using the `is_device_ptr` clause. Profiling shows data transfer accounts for <3% of total solve time; 97% is in `PFMGSolve` itself.

### Boundary Condition Support

| BC Type | Support | Mechanism |
|---------|---------|-----------|
| Periodic | Yes | `HYPRE_StructGridSetPeriodic()` |
| Neumann | Yes | Modified stencil coefficients |
| Dirichlet | Yes | Direct value enforcement |

For pure Neumann/periodic problems, a single cell is pinned to zero to remove the constant nullspace.

**Known limitation:** HYPRE PFMG has instability issues with 2D meshes that have periodic y BCs. The solver may produce NaN after ~50-100 steps. The auto-selector avoids this combination; the solver falls back to MG automatically.

### Implementation Files

| File | Description |
|------|-------------|
| `include/poisson_solver_hypre.hpp` | Class declaration |
| `src/poisson_solver_hypre.cpp` | Implementation |
| `tests/test_hypre_all_bcs.cpp` | Initialization test (all BC combos) |
| `tests/test_hypre_validation.cpp` | HYPRE vs MG comparison |
| `tests/test_hypre_backend.cpp` | Verifies GPU backend is active |
| `tests/test_hypre_canary.cpp` | Monitors known limitations (informational) |

---

## Native Multigrid

Geometric multigrid with V-cycle, Chebyshev polynomial smoothing, and OpenMP target offload for GPU acceleration. This is the general-purpose fallback solver that works for all BC combinations and grid configurations.

### V-Cycle Algorithm

```
vcycle(level):
    if level == coarsest:
        smooth(level, 8 iterations)       // Direct solve via Chebyshev
    else:
        smooth(level, nu1 iterations)     // Pre-smoothing (default nu1=3)
        compute_residual(level)
        restrict(level -> level+1)        // Full-weighting restriction
        vcycle(level+1)                   // Recursive coarse solve
        prolongate(level+1 -> level)      // Bilinear interpolation
        smooth(level, nu2 iterations)     // Post-smoothing (default nu2=1)
```

### Semi-Coarsening for Stretched Grids

Standard geometric multigrid coarsens uniformly in all directions, which fails for stretched y-grids (dy_min << dx, dz) because coarsening in y destroys the wall-normal resolution the stretching provides.

**Solution:** Semi-coarsening coarsens only x and z, keeping y resolution constant across all levels.

1. If `stretch_y = true`, semi-coarsening is automatically enabled
2. Each coarser level halves Nx and Nz but keeps Ny unchanged
3. Grid spacing: dx and dz double per level; dy stays constant
4. All levels share the same y-metric coefficients (precomputed on finest level)
5. Coarsening stops when Nx or Nz reaches 8 cells

The y-direction is handled by a **y-line smoother** (Thomas algorithm) that solves tridiagonal systems along each y-column exactly:

```
-aS[j]*u[j-1] + diag[j]*u[j] - aN[j]*u[j+1] = rhs[j]
```

where `rhs[j] = f[j] - Lx(u) - Lz(u)` (x/z Laplacian terms frozen from current iterate). The Thomas algorithm is O(Ny) per line and massively parallel on GPU (Nx x Nz independent lines). Thread-local stack arrays limit this to Ny <= 128. A pivot guard (eps = 1e-14) prevents NaN from zero pivots.

### Chebyshev Polynomial Smoothing

The smoother uses Chebyshev polynomials instead of Jacobi or Gauss-Seidel, providing better convergence rate with full GPU parallelism (no red-black ordering).

**Eigenvalue bounds (Gershgorin):** Optimal Chebyshev weights require bounds on the eigenvalues of the Jacobi-preconditioned operator D^{-1}A. These are computed per-level using the Gershgorin Circle Theorem:

```
lambda_max <= max_i (1 + sum_j |a_ij| / d_i)
```

For non-uniform y-grids, this iterates over all y-cells to find the maximum ratio. The implementation applies a 10% safety margin (`lambda_max *= 1.10`), enforces a floor (`lambda_max = max(1.8, lambda_max)`), and sets `lambda_min = 0.1 * lambda_max`.

The Chebyshev weights are:

```
theta_k = pi * (2k + 1) / (2 * degree)
omega_k = 1 / (d - c * cos(theta_k))
```

where `d = (lambda_max + lambda_min) / 2` and `c = (lambda_max - lambda_min) / 2`. Weights are clamped to a maximum of 2.0 for stability.

### PCG Coarse Solver

The coarsest MG level uses Preconditioned Conjugate Gradient (PCG) with 2 sweeps of the y-line smoother as preconditioner.

**Breakdown restart:** PCG can fail when `p^T A p` approaches zero. Instead of aborting, the solver detects `|p^T A p| < 1e-30` and restarts: re-precondition the current residual, project out the nullspace (if Neumann/periodic BCs), reset the search direction `p = z`, and continue.

**Convergence check throttling:** The residual norm requires a GPU-to-CPU reduction. To minimize sync overhead, the residual is only checked every 4th CG iteration, reducing GPU-to-CPU sync by ~75% with at most 3 extra iterations.

### CUDA Graph Optimization

On NVIDIA GPUs with NVHPC compiler, the entire V-cycle kernel sequence can be captured as a CUDA Graph, eliminating per-kernel launch overhead.

**How it works:**
1. On first solve, the V-cycle executes normally while CUDA captures all kernel launches
2. The captured graph is stored and validated
3. Subsequent solves replay the graph with a single `cudaGraphLaunch()` call
4. The graph is automatically recaptured if BCs or parameters change

**Performance impact (128^3 grid):**
- Eliminates ~37% of GPU API time spent in `cudaStreamSynchronize`
- Reduces dispatch overhead from O(levels x kernels) to O(1)
- Provides **4.9x overall speedup** for MG solver (see Performance section)

**Configuration:**

```ini
poisson_use_vcycle_graph = true    # Enable (default in GPU builds)
poisson_use_vcycle_graph = false   # Disable (required for recycling inflow)
```

**Requirements:**
- NVIDIA GPU with CUDA support
- NVHPC compiler (uses `ompx_get_cuda_stream()`)
- 3D mesh (2D uses non-graphed path)
- Fixed-cycle mode (`poisson_fixed_cycles > 0`)

**Automatic disabling:** The graph is disabled for recycling inflow BCs (inlet changes each step), semi-coarsening (different operator structure), and 2D meshes.

**Programmatic control:**

```cpp
auto& mg = dynamic_cast<MultigridPoissonSolver&>(poisson_solver);
mg.disable_vcycle_graph();
```

### Convergence Criteria

The MG solver supports 4 convergence modes (any triggers exit):

| Mode | Criterion | Use Case |
|------|-----------|----------|
| RHS-relative | `\|r\| / \|b\| < tol_rhs` | Default for projection |
| Initial-relative | `\|r\| / \|r_0\| < tol_rel` | General iterative |
| Absolute | `\|r\|_inf < tol_abs` | Known RHS scale |
| Fixed-cycle | Exact N cycles, no checks | Fastest (enables CUDA Graphs) |
| Adaptive fixed-cycle | Check after N cycles, add more if needed | Balance of speed and robustness |

**Warm-start:** The solver reuses the previous `pressure_correction_` as the initial guess (already on GPU, no action needed). The first time step starts from zero.

### Implementation Files

- `include/poisson_solver_multigrid.hpp`
- `src/poisson_solver_multigrid.cpp`
- `include/mg_cuda_kernels.hpp` (CUDA Graph implementation)

---

## Legacy SOR

A simple Successive Over-Relaxation solver exists in `poisson_solver.cpp` but is **not selectable** via `poisson_solver = ...` in config files. It serves only as a reference implementation used internally by the legacy `PoissonSolver` class. The relaxation parameter `poisson_omega` (default 1.8) applies to this class only.

**Implementation files:**
- `include/poisson_solver.hpp`
- `src/poisson_solver.cpp`

---

## Configuration Reference

All Poisson solver parameters available in config files or via CLI (`--parameter value`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `poisson_solver` | `auto` | Solver backend: `auto`, `fft`, `fft2d`, `fft1d`, `hypre`, `mg`, `fft_mpi` |
| `poisson_tol` | `1e-6` | Convergence tolerance (RHS-relative or initial-relative) |
| `poisson_max_vcycles` | `20` | Maximum V-cycles or iterations per solve |
| `poisson_fixed_cycles` | `8` | Fixed V-cycle count (0 = convergence-based; >0 enables CUDA Graphs) |
| `poisson_use_vcycle_graph` | `true` | Enable CUDA Graph capture for MG V-cycles (GPU builds) |
| `poisson_chebyshev_degree` | `4` | Chebyshev polynomial degree per smoothing sweep (3-4 typical) |
| `poisson_nu1` | `0` | Pre-smoothing sweeps (0 = auto: 3 for wall BCs) |
| `poisson_nu2` | `0` | Post-smoothing sweeps (0 = auto: 1) |

---

## Performance Comparison

### 128^3 Benchmark (NVIDIA H200, 500 time steps)

| Solver | ms/step | Mcells/s | Speedup vs MG | Notes |
|--------|---------|----------|---------------|-------|
| MG (convergence) | 24.1 | 103 | 1.0x | Baseline, 10 V-cycles avg |
| MG + CUDA Graph (fixed) | 4.9 | 500 | **4.9x** | Graph eliminates dispatch overhead |
| HYPRE PFMG (optimized) | 4.16 ms/solve | -- | **0.99x** (vs MG per-solve) | Comparable to native MG |
| FFT (all-periodic) | 1.7 | 1267 | **14x** | Fastest; periodic BCs only |

### Timing Breakdown (MG + CUDA Graph, channel flow)

| Phase | ms/step | % of step |
|-------|---------|-----------|
| Poisson solve | 3.40 | 69% |
| apply_velocity_bc | 0.23 | 5% |
| velocity_copy | 0.23 | 5% |
| predictor_step | 0.23 | 5% |
| convection | 0.15 | 3% |
| velocity_correction | 0.13 | 3% |
| diffusion | 0.10 | 2% |
| Other | 0.44 | 8% |

### With vs Without CUDA Graph

| Component | Without Graph | With Graph | Reduction |
|-----------|---------------|------------|-----------|
| Total step | 24.1 ms | 4.9 ms | 4.9x |
| Poisson solve | 22.6 ms | 3.4 ms | 6.7x |
| V-cycle compute | ~3 ms | ~3 ms | Same |
| Dispatch overhead | ~18 ms | ~0.1 ms | 180x |

### When to Use Which Solver

| Scenario | Recommended | Why |
|----------|-------------|-----|
| 3D channel (periodic x/z) | FFT | 14x faster than MG; direct solver |
| 3D channel, stretched y | FFT | Stretched-y support via yLap coefficients |
| 2D channel | FFT2D | Direct solver for 2D |
| 3D duct (periodic x, walls y/z) | FFT1D | Hybrid FFT + 2D MG |
| Any BCs, production | MG + CUDA Graph | 4.9x over plain MG |
| Multi-GPU / MPI (future) | HYPRE | Designed for distributed systems |
| Debugging / validation | MG or SOR | Reference implementations |

---

## Boundary Condition Implementation

### Ghost Cell Layout

All solvers use a ghost-cell approach with `Nghost = 1`:

```
Array size: (Nx + 2*Ng) x (Ny + 2*Ng) x (Nz + 2*Ng)
Interior:   [Ng : Ng+Nx] x [Ng : Ng+Ny] x [Ng : Ng+Nz]
```

### Periodic BC

```cpp
p[0][j][k]    = p[Nx][j][k]     // Low ghost = high interior
p[Nx+1][j][k] = p[1][j][k]      // High ghost = low interior
```

### Neumann BC (dp/dn = 0)

```cpp
p[i][0][k] = p[i][1][k]         // Zero gradient: ghost = first interior
```

### Dirichlet BC (p = p_val)

```cpp
p[i][0][k] = 2*p_val - p[i][1][k]   // Linear extrapolation to enforce value at face
```

### Singular System Handling

For pure Neumann or fully periodic problems, the Laplacian has a nullspace (constant mode). All solvers handle this by:

1. **Mean subtraction** -- ensure compatibility of the RHS (volume-weighted for FFT on stretched grids, arithmetic for MG)
2. **Pin one value** -- set `p(0,0,0) = 0` to fix the constant

---

## Troubleshooting

### Solver Does Not Converge

1. Check that the RHS is compatible: `mean(div(u*))` should be near zero. If not, there may be a mass flux error in the boundary conditions.
2. Increase `poisson_max_vcycles` (default 20). For poorly conditioned problems, 30-50 may be needed.
3. Verify grid quality: extreme stretching ratios (>1.2 between adjacent cells) can slow convergence.
4. Try `poisson_solver = mg` explicitly to rule out solver-specific issues.

### Semi-Coarsening Issues

- Semi-coarsening is automatically enabled when `stretch_y = true`. It coarsens x/z only, keeping y fixed.
- CUDA Graphs are automatically disabled with semi-coarsening (different operator structure at each level).
- The convergence check interval is increased to 4 iterations (from default) when semi-coarsening is active, reducing GPU-to-CPU sync overhead.

### FFT on Stretched Grids

- FFT and FFT2D support stretched y via mesh yLap coefficients. This was added in March 2026 and provides 22-32x speedup over MG for stretched grids.
- FFT1D does NOT support stretched y (its internal 2D MG uses uniform `1/(dy*dy)`).
- If FFT produces NaN or blow-up on a stretched grid, check that volume-weighted mean subtraction is working: the log should show the solver selected as FFT, not falling back.
- The solvability condition on stretched grids is `sum(f * dyv[j]) = 0`, not `sum(f) = 0`.

### HYPRE Build Issues

**CUDA not found:**
```
CMake Error: enable_language(CUDA) failed
```
Ensure CUDA toolkit is installed and `nvcc` is in PATH:
```bash
module load cuda
export PATH=/usr/local/cuda/bin:$PATH
```

**HYPRE download failed:**
```
FetchContent: error downloading hypre
```
Download manually:
```bash
git clone https://github.com/hypre-space/hypre.git _deps/hypre-src
git -C _deps/hypre-src checkout v2.31.0
```

**CUDA out of memory at runtime:**
Reduce grid size or check for GPU memory leaks with `nvidia-smi`.

**Illegal memory access:**
Usually a mismatch between OMP-mapped and managed memory. Ensure `map(present: ...)` for OMP-mapped arrays and `is_device_ptr(...)` for `cudaMallocManaged` pointers.

**Silent CPU fallback:** In GPU builds, if HYPRE CUDA initialization fails, it falls back to CPU execution without warning. Run `test_hypre_backend` to verify the GPU backend is active. Check for the log line:
```
[HyprePoissonSolver] CUDA backend enabled (unified memory)
```

### HYPRE Testing

```bash
# Build with HYPRE and tests
CC=nvc CXX=nvc++ cmake .. -DUSE_GPU_OFFLOAD=ON -DUSE_HYPRE=ON -DBUILD_TESTS=ON
make -j8

# Run HYPRE tests
./test_hypre_all_bcs          # All BC configurations
./test_hypre_validation       # HYPRE vs MG comparison
./test_hypre_backend          # Verify GPU backend
./test_hypre_canary           # Known limitations (informational)

# CI integration
./scripts/ci.sh --hypre full  # Full suite with HYPRE
./scripts/ci.sh --hypre hypre # HYPRE tests only

# Cross-build comparison
./test_hypre_validation --dump-prefix /path/to/ref      # Generate reference (CPU build)
./test_hypre_validation --compare-prefix /path/to/ref    # Compare (GPU build)
```

---

## References

- Falgout, R.D. and Yang, U.M. "hypre: A Library of High Performance Preconditioners." ICCS 2002.
- [HYPRE Documentation](https://hypre.readthedocs.io/)
- [HYPRE Struct Interface](https://hypre.readthedocs.io/en/latest/ch-struct.html)
- [HYPRE GitHub Repository](https://github.com/hypre-space/hypre)
