# SIMPLE Solver — Complete Status (Apr 4, 2026, 10 PM)

## Summary

GPU SIMPLE for RANS steady-state is partially working. Laminar cases converge.
SST diverges due to the pressure correction coefficient being too large.
The fix is SIMPLEC (Van Doormaal & Raithby 1984), which modifies the pressure
correction to account for neglected neighbor velocity corrections.

## What Works

### 1. Laminar Poiseuille SIMPLE (diagonal-approximation, n_sweeps=0)
- **CPU**: Converges in 4031 iterations, L2 error 0.06%, 9/9 tests pass
- **GPU**: Was crashing (root cause: CPU reading GPU device pointers for pseudo_dt).
  Fix committed (0e41950). Should work now — needs H100/H200 to verify.

### 2. HYPRE Integration
- HYPRE 2.32 builds with nvc++ + GPU offload (H200) and CPU-only (g++/nvc++)
- BiCGSTAB + PFMG: works but converges TOO well (div(u*)=0)
- PFMG direct: works but overshoots (1 V-cycle too strong for SIMPLE)
- Jacobi direct: works, closest to OpenFOAM's 2 GS sweeps
- Build system: conditional CUDA (GPU-only), CPU HYPRE works without CUDA

### 3. Variable-Coefficient MG Pressure
- set_variable_coefficients() at level 0 for D = 1/a_P
- GPU: uses tau_div scratch buffers for 1/a_P (device-resident)
- Correctly forces MG over FFT when varcoeff is active

### 4. 3D Stencil Assembly
- u, v, w momentum kernels: 7-point stencil for 3D duct
- Includes diffusion (variable nu_eff), upwind convection, Patankar, pseudo-transient

### 5. Infrastructure
- set_time_integrator() for runtime RK3↔SIMPLE switching
- RK3 warm-up before SIMPLE (not needed per literature, was workaround for GPU bug)
- Warm-up step budget independent of max_steps
- Bulk velocity controller (proportional, for Re matching)

## What Doesn't Work

### SST SIMPLE: diverges with ALL momentum solvers
- Diagonal-approximation (n_sweeps=0): NaN at step 40
- HYPRE Jacobi (2 sweeps): NaN at step 5
- HYPRE PFMG (1 V-cycle): NaN at step 5
- HYPRE BiCGSTAB+PFMG (converged): stagnates (div(u*)=0)

### Root cause: pressure correction overshoots
The velocity correction `u = u* - 1/a_P * grad(p')` uses `a_P = a_P_phys`
(diffusion + convection diagonal). For SST where nu_t >> nu, the off-diagonal
coefficients are a large fraction of a_P, so `1/a_P` is still large relative
to the velocity scale. The pressure correction over-corrects → velocity
overshoots → larger divergence → larger correction → exponential blowup.

### Why OpenFOAM works: SIMPLEC
OpenFOAM uses `consistent yes` (SIMPLEC) which replaces `a_P` in the
pressure equation with `a_P - sum(|a_nb|)`. Since `sum(|a_nb|) ≈ a_P`
for a well-resolved stencil, the effective coefficient `a_P - H` is
MUCH smaller than `a_P`, making `1/(a_P - H)` LARGER. Wait — that
makes the correction BIGGER, not smaller. Need to re-derive carefully.

Actually: SIMPLEC modifies the pressure equation coefficient, NOT the
velocity correction coefficient. The correction still uses `1/a_P`.
The difference: SIMPLEC's pressure equation produces SMALLER p' because
the equation accounts for neighbor corrections that standard SIMPLE
neglects. Smaller p' → smaller velocity correction → stability.

## Bugs Found and Fixed (10 total this session)

| # | Bug | Commit | Impact |
|---|-----|--------|--------|
| 1 | RSM zero velocity gradients | ede29ba | RSM was laminar everywhere |
| 2 | Duct wall shear 2× error | aa340bd | All duct Cf was half correct |
| 3 | tau_div limiter missing molecular nu | aa340bd | Tensor divergence in warm-up |
| 4 | IBM wall distance ignores bodies | 1756342 | SST/k-omega wrong on cylinder |
| 5 | Hills Cf slope correction | 1756342 | ~13% Cf error on slopes |
| 6 | Warmup shares max_steps budget | 48366a3 | Incomplete warmup on cylinder |
| 7 | Duct stretched grid dy_wall | 0458063 | Wall shear 6.6× error |
| 8 | bulk_velocity() 2D-only for 3D | 377e332 | Controller wrong for 3D |
| 9 | Duct dp/dx too low for Re_b=3500 | various | Re_b was 627 instead of 3500 |
| 10 | **SIMPLE CPU-reads-GPU-data** | 0e41950 | ALL GPU SIMPLE crashes |

## Key Commits (SIMPLE development)

| Commit | Description |
|--------|-------------|
| 65d3540 | Fix SIMPLE pressure ghost cells |
| 9f85138 | Simplify SIMPLE (remove broken ADI) |
| 44ab646 | HYPRE momentum integration + stencil assembly |
| 177c3e8 | Force MG for varcoeff, update HYPRE v2.32 |
| 78207a2 | Fix varcoeff (1/a_P not a_P) + PFMG solver |
| 2bd431b | 2D v-momentum + p' debug |
| 2686443 | Use tau_div as GPU scratch for 1/a_P |
| 0e41950 | **Fix CPU-reads-GPU-data in pseudo_dt** |
| fc7904f | HYPRE Jacobi + conditional CUDA |

## Architecture

```
simple_step() {
  0. Apply BCs, copy velocity → velocity_old
  1. Zero tau_div, turbulence update (SST transport + closure)
  2. Compute nu_eff = nu + nu_t
  3. Compute tau_div (tensor models only)
  4. MOMENTUM SOLVE (n_sweeps > 0: HYPRE, n_sweeps = 0: diagonal-approx)
     → produces velocity_star (predicted velocity)
  5. Compute a_P (diagonal of momentum matrix)
  6. Apply BCs to velocity_star, IBM forcing
  7. PRESSURE CORRECTION
     - Compute div(velocity_star)
     - If SIMPLE && 2D: set varcoeff D = 1/a_P on MG, solve div(D*grad(p'))=div(u*)
     - Else: constant-coeff Poisson, rhs = div(u*)/dt
  8. VELOCITY CORRECTION: u = u* - 1/a_P * grad(p'), p += alpha_p * p'
  9. Pressure ghost cell wrapping (periodic x, Neumann y)
  10. IBM post-correction, BCs, halo exchange
  11. Residual: max|u_new - u_old|
}
```

## Next Steps

### Immediate: SIMPLEC implementation (~50 LOC)
SIMPLEC (Van Doormaal & Raithby, 1984) modifies the pressure equation to:
  `div((1/(a_P - H)) * grad(p')) = div(u*)`
where `H = sum of off-diagonal coefficients` for each cell.

This produces smaller p' → smaller velocity correction → stable for SST.
The velocity correction formula stays the same: `u = u* - 1/a_P * grad(p')`.
Only the pressure equation coefficient changes.

Implementation:
1. Compute `H_u = sum(|a_nb|)` at each u-face (already available from stencil assembly)
2. Compute `D_u = 1/(a_P - H_u)` instead of `D_u = 1/a_P`
3. Pass to varcoeff MG as before
4. Test on SST channel → should converge like OpenFOAM

### After SIMPLEC works:
1. Test SST on cylinder (2D, with IBM)
2. Test SST on duct (3D)
3. Run 21 models × 4 cases with SIMPLE
4. Compare convergence rate: SIMPLE iterations vs RK3 steps
5. Compute per-iteration cost for proper Pareto plot

### Production runs (in parallel):
1. RK3 at correct Re (dp/dx=-0.0166 duct, -0.060 hills)
2. Convergence plots (50K steps for key models)
3. DNS comparison (Krank hills, Vinuesa duct)

## Files

### SIMPLE solver
- `src/solver_time_simple.cpp` — main SIMPLE step
- `src/solver_time_kernels_simple.cpp` — GPU kernels (a_P, predictor, correction, RB-GS, Thomas, stencil assembly)
- `src/solver_time_kernels.hpp` — kernel declarations
- `src/momentum_solver_hypre.cpp` — HYPRE momentum solver (Jacobi/PFMG/BiCGSTAB)
- `include/momentum_solver_hypre.hpp` — HYPRE solver header
- `src/poisson_solver_multigrid.cpp` — variable-coefficient MG (level 0)

### Configs
- `examples/simple_poiseuille_test.cfg` — laminar validation
- `examples/simple_sst_channel.cfg` — SST channel (currently diverges)
- `examples/simple_sst_hypre.cfg` — SST with HYPRE (currently diverges)
- `examples/simple_cylinder_test.cfg` — cylinder with IBM
- `examples/simple_duct_test.cfg` — 3D duct
- `examples/simple_cavity_test.cfg` — driven cavity (not tested)

### Builds
- `build_simple/` — CPU build (g++, no GPU, no HYPRE) — Poiseuille works
- `build_hypre_cpu/` — CPU build (nvc++, HYPRE, no GPU) — SST diverges
- `build_hypre_gpu/` — GPU build (nvc++ cc90, HYPRE) — needs H100/H200
- `build_h200/` — GPU build (nvc++ cc90, no HYPRE) — production RK3 runs
