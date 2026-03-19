# Multi-GPU MPI Solver Fix — Implementation Plan

**Date**: 2026-03-18
**Goal**: Fix z-slab MPI decomposition so multi-GPU simulations produce correct physics.

## Issues Found (5)

1. **Periodic z-BC conflicts with MPI**: `enforce_periodic_halos_device()` and `apply_velocity_bc()` do local z-wrapping, overwriting ghost cells that should come from neighboring ranks.

2. **velocity_star_ halos not exchanged before divergence**: `compute_divergence(Star)` reads z-ghost cells of velocity_star_ that contain garbage after the predictor step.

3. **velocity_ halos corrupted at step start**: Halo exchange at end of step() fills correct data, but `apply_velocity_bc()` immediately overwrites z-ghost cells with local periodic wrapping.

4. **MG Poisson solver is local per rank**: Each rank runs MG on its own z-slab with local periodic z-BCs. Pressure is not globally coupled.

5. **Mean divergence subtraction is local, not global**: Each rank subtracts its own local mean instead of the global mean, breaking Poisson solvability.

**Latent bug**: HaloExchange buffer sizing uses `(Nx+2*Ng)*(Ny+2*Ng)` but u-component has stride `(Nx+1+2*Ng)`. Pack loop may miss the last element per row.

## Implementation Steps

### Step 1: Fix HaloExchange buffer size (latent bug)
- **File**: `src/halo_exchange.cpp`, `include/halo_exchange.hpp`
- **Change**: Use `max(Nx+1+2*Ng, Nx+2*Ng) * max(Ny+1+2*Ng, Ny+2*Ng) * Ng` for face_size_
- **LOC**: ~10

### Step 2: Skip z-periodic BC under MPI
- **File**: `src/solver_operators.cpp` (z-direction BC section, lines 285-388)
- **Change**: When `decomp_ && decomp_->is_parallel() && mesh_->Nz == decomp_->nz_local()`, skip z-periodic ghost fill for u, v, w. Keep z no-slip path unchanged.
- **LOC**: ~15
- **Effect**: Fixes Issues 1 and 3 together. After halo exchange fills z-ghosts from neighbors, `apply_velocity_bc()` no longer overwrites them.

### Step 3: Exchange velocity_star_ halos before divergence
- **Files**: `src/solver.cpp` (Euler path), `src/solver_time.cpp` (RK path)
- **Change**: After predictor step and BCs for velocity_star_, before `compute_divergence(Star)`, add halo exchange for velocity_star_ u/v/w pointers.
- **LOC**: ~30 (15 per file)
- **Effect**: Fixes Issue 2. Divergence RHS has correct values at z-boundary cells.

### Step 4: Make mean divergence subtraction global
- **Files**: `src/solver.cpp`, `src/solver_time.cpp`
- **Change**: After computing local `sum_div`, allreduce to get global sum. Use `Nx * Ny * decomp_->nz_global()` for global cell count.
- **LOC**: ~20
- **Effect**: Fixes Issue 5. Poisson RHS satisfies global solvability condition.

### Step 5: Fix MG z-BCs + add pressure halo outer loop
- **File**: `src/solver.cpp`
- **Change** (Approach A — minimal):
  1. Override z-BCs to Neumann for local MG when MPI z-decomposed
  2. Wrap MG solve in outer loop: solve → exchange pressure halos → check global convergence
- **LOC**: ~40
- **Effect**: Fixes Issue 4. Local MG with Neumann z-BCs + halo exchange provides global pressure coupling via additive Schwarz.

### Step 6: Add validation tests
- **New file**: `tests/test_mpi_physics.cpp`
- **Tests**:
  1. TGV decay: compare KE(t) from 1-rank vs 4-rank (should match within tolerance)
  2. Divergence: global max|div(u)| after projection should be O(1e-12)
  3. Pressure continuity at rank boundaries
  4. 100+ steps without drift
- **LOC**: ~120

## Order
1 → 2 → 3 → 4 → 5 → 6

## Files Modified
| File | Changes | LOC |
|------|---------|-----|
| `src/halo_exchange.cpp` + `.hpp` | Buffer sizing | ~10 |
| `src/solver_operators.cpp` | Guard z-periodic BC | ~15 |
| `src/solver.cpp` | velocity_star_ halo, global mean_div, MG outer loop | ~60 |
| `src/solver_time.cpp` | velocity_star_ halo, global mean_div | ~30 |
| `tests/test_mpi_physics.cpp` | New validation test | ~120 |
| **Total** | | **~235** |

## Risks
- MG convergence with Neumann z-BCs + halo exchange may need many outer iterations. Start with max_outer=20, tune later. If too slow, upgrade to halo exchange inside V-cycle (Approach B).
- w-component staggered seam face needs verification: `w[Ng+Nz_local]` on rank R should equal `w[Ng]` on rank R+1.
- `exchange_device()` requires `USE_CUDA_KERNELS` — verify it works with OpenMP target offload builds.
