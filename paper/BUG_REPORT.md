# Bug Report — Deep Audit (Apr 3, 2026)

## Critical Bugs Found and Fixed

### BUG 1: RSM never computes velocity gradients (FIXED — ede29ba)
**File**: `src/turbulence_rsm.cpp:555` (GPU) and `src/turbulence_rsm.cpp:762` (CPU)
**Impact**: RSM gives identical results to laminar (None) on ALL cases
**Severity**: CRITICAL — ALL RSM production data is INVALID

RSM's `advance_turbulence()` reads velocity gradients from `device_view->dudx` etc.
but nobody computes them beforehand. GPU path had a wrong comment ("must already be
computed by solver"); CPU path used hardcoded zero arrays. Zero gradients → zero
production → zero Reynolds stresses → RSM = laminar.

### BUG 2: Duct wall shear off by factor of 2 (FIXED — aa340bd)
**File**: `app/main_duct.cpp:598`, `src/qoi_extraction.cpp:124`
**Impact**: All reported duct wall shear stress values are half the correct magnitude
**Severity**: CRITICAL — affects duct Cf data for the paper

`dy_wall = mesh.dy` but the wall-to-cell-center distance is `mesh.dy / 2`.

### BUG 3: tau_div limiter uses nu_t, not nu_eff (FIXED — aa340bd)
**File**: `src/solver_operators.cpp:1150` (2D) and line 1246 (3D)
**Impact**: Limiter inactive in laminar warm-up zones → unconstrained tensor correction
**Severity**: HIGH — may explain tensor model divergence during SST→TBNN transition

The tau_div limiter, which is the sole stability mechanism against anti-diffusion,
used `nu_t + 1e-10` instead of `nu_eff = nu + nu_t`. In warm-up zones where nu_t≈0,
the limiter was zero → tensor corrections passed through unclipped.

### BUG 4: Pressure ghost cells not wrapped (FIXED — 65d3540, earlier session)
**File**: `src/solver_time_simple.cpp`
**Impact**: SIMPLE solver stagnation, also affects standard projection path
**Severity**: HIGH

## Critical Bugs Found — NOT YET FIXED

### BUG 5: Wall distance ignores IBM bodies (UNFIXED)
**File**: `src/mesh.cpp:6-23`
**Impact**: SST/RSM/k-omega wall BCs applied at DOMAIN boundary, not cylinder/sphere surface
**Severity**: CRITICAL — affects ALL turbulence models on IBM cases (cylinder, sphere)

`Mesh::wall_distance()` always computes distance to y_min/y_max (domain boundaries).
For the cylinder domain [-6,6], the "wall distance" at a cell near the cylinder
surface (r=0.5) is ~6, not ~0.01. This means:
- SST F1/F2 blending is reversed (near-wall layer treated as freestream)
- omega wall BC applied at wrong location
- TBNN/TBRF wall-distance input feature is wrong

**This explains k-omega Cd=2.42 on cylinder.** The omega wall BC at the wrong distance
produces excess nu_t everywhere, thickening the boundary layer.

**Fix needed**: Compute IBM-aware wall distance from body's phi() function during init.
This is a ~50 LOC change but affects all IBM cases.

### BUG 6: Hills Cf(x) ignores wall slope (UNFIXED)
**File**: `src/qoi_extraction.cpp:33-38`
**Impact**: ~13% error in Cf on hill slopes
**Severity**: MODERATE — affects hills comparison with DNS Cf(x)

tau_w = nu * u_cell / dy uses vertical distance, not wall-normal distance.
On slopes, the error is cos(θ) where θ is the local hill angle (~30° max).

## Important Bugs (limited production impact)

| # | Bug | Location | Impact |
|---|-----|----------|--------|
| 7 | z-wall ghost BC wrong for Ng≥2 | solver_operators.cpp:327 | Only O4 stencils |
| 8 | sync_transport_from_gpu skips bg_transport | solver.cpp:4427 | Stale CPU k/ω diagnostics |
| 9 | SST GPU buffers 2D-sized for 3D | turbulence_transport.cpp:642 | Memory waste only |
| 10 | Cd/Cl inconsistent methods on cylinder | main_cylinder.cpp:421 | Different estimators |
| 11 | map(to:) during GPU compute for CFL | solver_time.cpp:192 | Violates GPU rules |
| 12 | 2nd-order ghost-cell stencil dead code | ibm_forcing.cpp:234 | Memory waste |

## Production Data Impact Summary

| Model | Cylinder | Hills | Duct | Sphere |
|-------|----------|-------|------|--------|
| None | ✅ Valid | ✅ Valid | ✅ Valid | ✅ Valid |
| Baseline | ⚠️ Rerun (nu_t fix) | ⚠️ Rerun (nu_t fix) | ⚠️ Rerun (nu_t fix) | ⚠️ Rerun |
| SST | ⚠️ Wall dist wrong | ✅ Valid (no IBM) | ✅ Valid (no IBM) | ⚠️ Wall dist wrong |
| k-omega | ❌ Wall dist (Cd=2.42) | N/A | ✅ Valid | ⚠️ Wall dist wrong |
| EARSM (all) | ⚠️ Wall dist | ✅ Valid | ❌ Diverges | ⚠️ Wall dist |
| GEP | ⚠️ Rerun (nu_t fix) | ⚠️ Rerun | ⚠️ Rerun | ⚠️ Rerun |
| RSM-SSG | ❌ Zero gradients | ❌ Zero gradients | ❌ Zero gradients | ❌ Zero gradients |
| MLP (all) | ⚠️ Rerun (nu_t fix) | ⚠️ Rerun | ⚠️ Rerun | ⚠️ Rerun |
| TBNN (all) | ⚠️ Wall dist | ✅ Valid | ✅ Valid | ⚠️ Wall dist |
| PI-TBNN (all) | ⚠️ Wall dist | ✅ Valid | ✅ Valid | ⚠️ Wall dist |
| TBRF (all) | ⚠️ Wall dist | ✅ Valid | ✅ Valid | ⚠️ Wall dist |

**Legend**: ✅ Valid | ⚠️ Needs re-run (fixed bug) | ❌ Invalid (wrong results)

## Action Items (priority order)

1. ✅ Fix RSM gradient bug — DONE
2. ✅ Fix duct wall shear 2x — DONE
3. ✅ Fix tau_div limiter — DONE
4. ⬜ Fix IBM-aware wall distance — NEEDED for cylinder/sphere
5. ⬜ Fix hills Cf slope correction — NEEDED for hills
6. ⬜ Rebuild H200 binaries
7. ⬜ Re-run ALL models on ALL cases with fixed code
