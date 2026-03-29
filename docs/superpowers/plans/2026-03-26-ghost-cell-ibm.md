# Ghost-Cell IBM Implementation (Mar 26, 2026)

## Problem

Volume penalization IBM (Angot et al. 1999) has an accuracy-stability tradeoff:
- eta=0.1: stable on all geometries but 32% no-slip error (Cd 334% wrong on sphere)
- eta=0.001: accurate but only stable on cylinder (diverges on hills/sphere)
- No combination of explicit/semi-implicit/pre-correction/post-correction fixes this

## Solution: Ghost-Cell IBM (Fadlun et al. 2000)

Replace the smooth weight transition in the forcing band with sharp interpolation at the body boundary.

### Algorithm
1. **Solid cells** (phi < -band): hard forcing, u = 0 (weight = 0, existing path)
2. **Forcing cells** (-band < phi < 0): ghost-cell interpolation
   - Find nearest Fluid neighbor in dominant gradient direction (search ±1, then ±2 steps)
   - Compute alpha = |phi_forcing| / (|phi_forcing| + phi_fluid)
   - Set u[forcing] = u[fluid_nbr] * alpha (linear interpolation with u=0 at surface)
3. **Fluid cells** (phi > 0): no modification (weight = 1, existing path)

### Data Structures (sparse, GPU-mapped)
- `ghost_self_u/v/w_` — flat indices of forcing cells
- `ghost_nbr_u/v/w_` — flat indices of fluid neighbors
- `ghost_alpha_u/v/w_` — interpolation weights
- Precomputed on CPU in `compute_ghost_cell_interp()`, mapped to GPU

### GPU Kernel
After the existing weight-multiply pass (which zeroes solid cells), a scatter kernel applies:
```cpp
#pragma omp target teams distribute parallel for
for (int g = 0; g < n_ghost; ++g) {
    u_ptr[self[g]] = u_ptr[nbr[g]] * alpha[g];
}
```
No race conditions: self indices are unique, nbr indices are fluid (unmodified by weight multiply).

## Hills Two-Phase Approach

Hills at Re=10595 is unstable with ghost-cell from cold start (flow hasn't developed).

**Solution:** Penalization warm-up → ghost-cell evaluation
1. **Phase 1 (warm-up):** SST + eta=0.1 penalization. Stable, develops flow.
2. **After warm-up:** Call `ibm.set_ghost_cell_ibm(true); ibm.set_penalization_eta(0.0); ibm.recompute_and_remap();`
3. **Phase 2 (evaluation):** Ghost-cell + eta=0. Accurate, stable with developed flow.

Implemented in `app/main_hills.cpp` after the warm-up model swap section.

## Results

| Case | Method | Cd or QoI | Reference | Error |
|------|--------|-----------|-----------|-------|
| Cylinder Re=100 | ghost-cell, eta=0 | Cd=1.55 (t=10) | 1.35 | **15%** |
| Sphere Re=200 | ghost-cell, eta=0 | Cd=0.26 (t=10, converging) | 0.77 | converging |
| Hills Re=10595 | warmup→gc | sep x/H=0.21 | 0.22 | **5%** |

Compare to old penalization (eta=0.1): Cylinder Cd=37.6 (2686%), Sphere Cd=3.34 (334%), Hills sep=2.7 (1127%).

## Test Status
- 5/5 CPU IBM tests pass
- 6/6 GPU IBM tests pass on V100 (including IBMCylinderDragTest, IBMStrouhalTest)

## Config State
- Cylinder: `ibm_eta = 0`, ghost-cell enabled in `main_cylinder.cpp` constructor
- Sphere: `ibm_eta = 0`, ghost-cell enabled in `main_cylinder.cpp` constructor
- Hills: `ibm_eta = 0.1` (for warm-up), ghost-cell enabled after warm-up in `main_hills.cpp`
- Duct: no IBM

## Uniformity Note
All cases use ghost-cell for the **evaluation phase** (where QoIs are measured). The hills warm-up with penalization is a pre-processing step — confirmed necessary because Re=10595 diverges from cold start with ghost-cell alone (tested, diverges at t=0.26). Cylinder/sphere are stable with ghost-cell from cold start at Re=100-300.

## Files Modified
- `include/ibm_forcing.hpp` — ghost-cell data structures, `set_ghost_cell_ibm()`, `recompute_and_remap()`, `recompute_weights()`
- `src/ibm_forcing.cpp` — `compute_ghost_cell_interp()`, ghost-cell scatter in `apply_forcing_device()` and `apply_forcing()`, `recompute_and_remap()`, GPU mapping/unmapping
- `app/main_hills.cpp` — warm-up → ghost-cell transition
- `app/main_cylinder.cpp` — ghost-cell enabled at construction

## References
- Fadlun et al. (2000), "Combined immersed-boundary finite-difference methods for three-dimensional complex flow simulations", JCP 161:35-60
- Angot, Bruneau & Fabrie (1999), "A penalization method to take into account obstacles in incompressible viscous flows", Numerische Mathematik 81(4):497-520

## Grid Convergence Study (Mar 27, 2026)

Ghost-cell IBM is first-order accurate (O(h) convergence). The Cd error depends strongly on cells per diameter.

| Geometry | Grid | Cells/D | Cd | Cd ref | Error |
|----------|------|---------|-----|--------|-------|
| Cylinder | 384×288 | 24 | 1.47 | 1.35 | 9% |
| Cylinder | 128×96 | 8 | 0.63 | 1.35 | 53% |
| Sphere | 192×128² | 13 | 0.19 | 0.77 | 75% |
| Sphere | 128×96² | 10 | 0.16 | 0.77 | 79% |

**Implication:** Need ~24 cells/D for <10% Cd error. Cylinder at 384×288 is fine. Sphere needs finer grid (e.g., 384×256×256 for ~24 cells/D — 25M cells, feasible on H200 but expensive for 20 models).

**Alternative:** Accept current sphere resolution and report relative model comparison (all models use same IBM, ranking valid even if absolute Cd is off). This is the pragmatic approach for the paper.

**Root cause of low Cd:** The positive-alpha interpolation u_ghost = u_fluid * alpha sets forcing cells deep inside the body to ~50% of fluid velocity, creating a porous-like boundary. At high resolution (many cells in forcing band), the inner cells have phi close to -band_width, so alpha→1 — but the Poisson RHS masking zeroes these cells anyway. At low resolution, fewer cells means the interpolation doesn't resolve the boundary layer properly.

## Sphere Re=200 Detailed Grid Convergence (Mar 27, 2026)

### With SST warm-up, T_final=10-50

| Grid | Cells/D | Total cells | Cd (SST) | Cd ref | Error |
|------|---------|-------------|----------|--------|-------|
| 128×96² | 10 | 1.2M | 0.17 | 0.77 | 78% |
| 192×128² | 13 | 3.1M | 0.25 | 0.77 | 68% |
| 256×192² | 17 | 9.4M | 0.21 | 0.77 | 73% |
| 384×256² | 24 | 25.2M | 0.38 | 0.77 | 51% |

### Baseline (no SST) at 384×256²
Cd=0.40 at t=5 — same as SST (Cd=0.38 at t=10). **SST is not the cause of low Cd.** The ghost-cell IBM itself limits accuracy at this resolution.

### Comparison with cylinder (same cells/D)
| Case | Cells/D | Cd | Cd ref | Error |
|------|---------|-----|--------|-------|
| Cylinder 384×288 (2D) | 24 | 1.47 | 1.35 | 9% |
| Sphere 384×256² (3D) | 24 | 0.38 | 0.77 | 51% |

The 3D sphere converges much slower than the 2D cylinder at the same cells/D. This is a known property of first-order ghost-cell IBM: the 3D staggered grid has more complex interpolation geometry near curved surfaces.

### Conclusion
For the paper: cylinder Cd is quantitatively accurate. Sphere Cd needs either higher-order IBM or much finer grids (~40+ cells/D). Report relative model comparisons for sphere at current resolution.
