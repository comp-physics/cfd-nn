# Recycling Inflow Boundary Condition Guide

This guide documents the recycling/rescaling turbulent inflow boundary condition, based on the method of Lund, Wu & Squires (1998). This BC generates realistic turbulent inflow data for spatially-developing simulations without requiring a separate precursor simulation.

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Configuration](#configuration)
- [Implementation Details](#implementation-details)
- [GPU Implementation](#gpu-implementation)
- [Testing](#testing)
- [Known Limitations](#known-limitations)
- [References](#references)

---

## Overview

For spatially-developing flows (e.g., boundary layers, jets, wakes), the inlet boundary needs realistic turbulent velocity data. The recycling method works by:

1. **Extracting** a velocity plane from a downstream recycle station
2. **Shifting** it in the spanwise direction (decorrelation)
3. **Filtering** it temporally (AR1 smoothing for stability)
4. **Correcting** mass flux and divergence
5. **Blending** it into the inlet region via a fringe zone

This provides a self-sustaining turbulent inflow as long as the recycle station is in a region of developed turbulence.

### When to Use

- Spatially-developing boundary layers
- Channel flow with inflow/outflow BCs
- Any simulation needing realistic turbulent inlet data without a precursor

### When NOT to Use

- Fully periodic channel flow (use periodic BCs instead — simpler and more accurate)
- Laminar inflows (use a prescribed profile)
- Cases where the recycle station would be in a region of interest (contaminates the solution)

---

## How It Works

### Step 1: Recycle Plane Extraction

At each time step, the solver extracts a 2D (y-z) slice of all three velocity components at the recycle plane location:

```
u_recycle(j, k) = u(i_recycle, j, k)
v_recycle(j, k) = v(i_recycle, j, k)
w_recycle(j, k) = w(i_recycle, j, k)
```

The recycle plane is placed automatically at `x_recycle = x_min + 10*delta` (where delta is the channel half-height), or at a user-specified `recycle_x` location. It is clamped to at least 5 cells from each boundary.

### Step 2: Spanwise Shift

To decorrelate the recycled data from the inlet (preventing artificial periodicity), the extracted plane is circularly shifted in z:

```
u_inlet(j, k) = u_recycle(j, (k + shift_k) % Nz)
```

The shift amount `shift_k` changes every `recycle_shift_interval` steps (default: 100) by a deterministic increment of 1-7 cells, preventing the inlet from locking onto the recycle plane structure.

### Step 3: Temporal Filtering (AR1)

An exponential moving average smooths the recycled data in time to prevent high-frequency noise:

```
u_filtered(n+1) = alpha * u_filtered(n) + (1 - alpha) * u_raw(n+1)
```

where `alpha = exp(-dt / tau)` and `tau` is set by `recycle_filter_tau`. Disabled by default (tau = -1).

### Step 4: Mass Flux Correction

The recycled velocity is adjusted to maintain the target bulk velocity at the inlet:

```
bulk_u = area_weighted_average(u_inlet)
scale = target_bulk_u / bulk_u                    # clamped to +/-10% deviation
u_inlet(j, k) += bulk_u * (scale - 1.0)           # uniform offset preserves fluctuations
```

The uniform offset (rather than multiplicative scaling) preserves all fluctuation structure while correcting only the mean.

### Step 5: Divergence Correction

The inlet u-velocity is adjusted to ensure the discrete divergence is zero at each inlet ghost cell:

```
div_transverse = dv/dy + dw/dz    (computed at inlet face)
u_corrected = u_interior + dx * div_transverse
```

This ensures `du/dx + dv/dy + dw/dz = 0` at the inlet, which is required for the Poisson solver to produce a clean pressure field.

### Step 6: Fringe Blending

The recycled velocity is blended into the domain over a fringe zone near the inlet using a cosine weight:

```
beta(x) = 0.5 * (1 + cos(pi * x_local / L_fringe))
```

where `x_local` is the distance from the inlet and `L_fringe` is the fringe zone length.

- At the inlet (x_local = 0): beta = 1.0 (100% recycled velocity)
- At the fringe end (x_local = L_fringe): beta = 0.0 (100% computed velocity)

The blended velocity is:

```
u_field(i, j, k) = beta * u_inlet(j, k) + (1 - beta) * u_field(i, j, k)
```

Applied to all three velocity components (u, v, w) in the fringe zone.

---

## Configuration

### Required Settings

```ini
recycling_inflow = true
```

This automatically sets the Poisson boundary conditions to:
- **x_lo**: Dirichlet (p = 0 at inlet)
- **x_hi**: Neumann (dp/dn = 0 at outlet)
- **y**: Neumann (walls)
- **z**: Periodic

### Optional Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `recycle_x` | -1.0 | x-location of recycle plane. -1 = auto: `x_min + 10*delta` |
| `recycle_shift_z` | -1 | Spanwise shift in cells. -1 = auto: Nz/4 |
| `recycle_shift_interval` | 100 | Steps between shift updates (0 = constant shift) |
| `recycle_filter_tau` | -1.0 | AR1 filter timescale. -1 = disabled, >0 = enabled |
| `recycle_fringe_length` | -1.0 | Fringe zone length. -1 = auto: 2*delta |
| `recycle_target_bulk_u` | -1.0 | Target bulk velocity. -1 = from initial condition |
| `recycle_remove_transverse_mean` | true | Remove mean v,w at inlet (enforce zero net transverse flow) |
| `recycle_diag_interval` | 0 | Log recycling diagnostics every N steps (0 = disabled) |

### Example Config

```ini
# Spatially-developing channel with recycling inflow
simulation_mode = unsteady
turb_model = none
recycling_inflow = true

Nx = 192
Ny = 96
Nz = 64
x_min = 0.0
x_max = 25.0
y_min = -1.0
y_max = 1.0
z_min = 0.0
z_max = 6.283
stretch_y = true
stretch_beta = 2.0

recycle_x = 20.0              # Recycle from near outlet
recycle_fringe_length = 3.0   # Blend over 3 units
recycle_shift_interval = 50   # Update shift every 50 steps
```

---

## Implementation Details

### Source File

`src/solver_recycling.cpp` (~1400 lines) contains all recycling inflow logic.

### Step Sequence Integration

Recycling operations are integrated into the main time step at two points:

**Before Poisson solve (step 12 in the main sequence):**
1. `apply_recycling_inlet_bc()` — extracts recycle plane, shifts, filters, corrects mass flux
2. `correct_inlet_divergence()` — adjusts inlet u for div-free constraint
3. `apply_fringe_blending()` — blends recycled data into fringe zone

**After projection (step 17):**
1. `extract_recycle_plane()` — extracts updated recycle data for next step
2. `process_recycle_inflow()` — prepares inlet buffers for next step

### Buffer Sizes (Staggered Grid)

Because of the MAC staggered grid, each velocity component has different array dimensions:

| Component | Size | Why |
|-----------|------|-----|
| u | Ny x Nz | u is at x-faces (cell-centered in y, z) |
| v | (Ny+1) x Nz | v is at y-faces (Ny+1 faces in y) |
| w | Ny x (Nz+1) | w is at z-faces (Nz+1 faces in z) |

### CUDA Graph Interaction

The recycling inflow modifies the inlet boundary conditions, which can invalidate a captured CUDA Graph V-cycle. The solver calls `disable_vcycle_graph()` on the Poisson solver when recycling is enabled to prevent stale graph execution. The standard (non-graphed) V-cycle is used instead.

---

## GPU Implementation

### Device Buffers

Six persistent device buffers are allocated via `omp_target_alloc()`:

```
recycle_u_ptr_   (Ny * Nz doubles)       — raw extracted u at recycle plane
recycle_v_ptr_   ((Ny+1) * Nz doubles)   — raw extracted v
recycle_w_ptr_   (Ny * (Nz+1) doubles)   — raw extracted w
inlet_u_ptr_     (Ny * Nz doubles)       — processed u for inlet (after shift/filter/correction)
inlet_v_ptr_     ((Ny+1) * Nz doubles)   — processed v for inlet
inlet_w_ptr_     (Ny * (Nz+1) doubles)   — processed w for inlet
inlet_area_ptr_  (Ny * Nz doubles)       — area weights for mass flux correction
```

These are allocated separately from the main field arrays to avoid "partially present" errors with OpenMP target regions.

### Critical GPU Bug Fix: Divergence Correction

**The problem:** The original GPU implementation of `correct_inlet_divergence()` used `inlet_u_ptr_` as scratch space to compute corrected u-velocities. But `apply_fringe_blending()` subsequently read from `inlet_u_ptr_` for blending. The scratch overwrites corrupted the recycled u values, causing large (~3443%) stress differences between CPU and GPU.

**The fix:** The GPU divergence correction now writes corrected u-velocities **directly to the velocity field** (`u_ptr`) at the inlet face and ghost cells, without touching `inlet_u_ptr_`. This preserves the original recycled values for fringe blending.

```cpp
// GPU path: compute corrected u and write directly to field
#pragma omp target teams distribute parallel for
for (int idx = 0; idx < n_inlet; ++idx) {
    // Decompose flat index to (j, k)
    double u_interior = u_ptr[...first interior face...];
    double dvdy = (v_ptr[j+1] - v_ptr[j]) / dy;
    double dwdz = (w_ptr[k+1] - w_ptr[k]) / dz;
    double u_corrected = u_interior + dx * (dvdy + dwdz);

    // Write directly to velocity field (NOT to inlet_u_ptr_)
    u_ptr[inlet_face] = u_corrected;
    u_ptr[ghost_cells] = u_corrected;
}
```

### Fringe Blending GPU Kernels

All three velocity components (u, v, w) are blended on GPU using separate kernels. An earlier bug only blended u on GPU while the CPU path blended all three, causing divergence between CPU and GPU results.

### initialize() GPU Fix

The `initialize()` function calls `apply_velocity_bc()` after `sync_to_gpu()` to ensure ghost cells are correctly set on the GPU. This is scoped to recycling-only to avoid affecting other solvers:

```cpp
#ifdef USE_GPU_OFFLOAD
    sync_to_gpu();
    if (config_.recycling_inflow) {
        apply_velocity_bc();
    }
#endif
```

---

## Testing

### RecyclingInflowTest (12 checks)

Tests basic recycling inflow functionality in a turbulent channel:

1. U-velocity profile symmetry about channel center
2. Friction velocity u_tau > 0 (flow is being driven)
3. Mass conservation at inlet (flux matches target)
4. Fringe blending weights correct (beta = 1 at inlet, 0 at fringe end)
5. Recycle plane extraction matches field values
6. Spanwise shift produces different values than unshifted
7. Divergence at inlet is near zero after correction
8. v, w blending active in fringe zone
9. Ghost cells match inlet face values
10. u_tau consistency between wall-derived and force-derived
11. No NaN/Inf in any velocity field
12. Bulk velocity within 5% of target

### PeriodicVsRecyclingTest

Runs the same channel flow configuration twice:
1. Fully periodic in x (reference)
2. Recycling inflow in x

After several time steps, compares plane-averaged statistics. The recycling case should match the periodic case within tolerance:

- Shear stress: < 5% difference
- Streamwise stress: < 5% difference

This test verifies that the recycling machinery doesn't introduce spurious turbulence or kill existing turbulence.

Both tests pass on CPU and GPU builds.

---

## Known Limitations

1. **Requires sufficient domain length:** The recycle plane must be far enough downstream that the flow is fully developed. A minimum of ~10*delta from the inlet is recommended.

2. **AR1 filter not on GPU:** The temporal filtering step currently runs on CPU only. For GPU builds, the filter is skipped. This means GPU runs may have slightly noisier inlet data.

3. **Spanwise shift requires periodic z:** The circular shift assumes periodic z-boundary conditions. Using recycling with z-walls would require a different decorrelation strategy.

4. **No inner/outer layer decomposition:** The current implementation uses a simple direct recycling (no van Driest rescaling or inner/outer layer separation). This is sufficient for channels but may need extension for boundary layers with strong pressure gradients.

5. **CUDA Graph incompatibility:** Recycling modifies inlet BCs each step, so the CUDA Graph V-cycle is automatically disabled. This costs some GPU performance (~10-20% for the Poisson solve).

6. **Mass flux correction clamped to +/-10%:** Large deviations from the target bulk velocity are clamped to prevent instability. If the flow deviates more than 10%, the correction may not fully restore the target.

---

## References

- Lund, T. S., Wu, X., & Squires, K. D. "Generation of turbulent inflow data for spatially-developing boundary layer simulations." *J. Comput. Phys.* 140.2 (1998): 233-258
- Spalart, P. R. "Direct simulation of a turbulent boundary layer up to Re_theta = 1410." *J. Fluid Mech.* 187 (1988): 61-98
