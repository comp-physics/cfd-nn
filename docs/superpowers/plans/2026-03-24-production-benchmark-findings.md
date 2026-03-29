# Production Benchmark Findings & Plan (Mar 24, 2026)

## What We Ran

**H200 GPU**, `build_h200` (nvc++ 25.5, cc90, Release), 20 models × 8 configs.

| Case | Steps | Grid | Mode | Domain | Integrator |
|------|-------|------|------|--------|------------|
| Hills Re=10595 | 2000 | 384×192×1 (74K) | steady | [0,9]×[0,3.035] | rk3 |
| Cylinder Re=100 | 2000 | 384×288×1 (111K) | unsteady | [-3,13]×[-6,6] | rk3 |
| Cylinder Re=300 | 2000 | 384×288×1 (111K) | unsteady | [-3,13]×[-6,6] | rk3 |
| Cylinder Re=3900 | 2000 | 384×288×1 (111K) | unsteady | [-3,13]×[-6,6] | rk3 |
| Duct Re_b=3500 | 1000 | 96³ (885K) | steady | uniform | rk3 |
| Sphere Re=100 | 500 | 192×128×128 (3.15M) | unsteady | [-4,11]³ | rk3 |
| Sphere Re=200 | 500 | 192×128×128 (3.15M) | unsteady | [-4,11]³ | rk3 |
| Sphere Re=300 | 500 | 192×128×128 (3.15M) | unsteady | [-4,11]³ | rk3 |

All configs: `convective_scheme = skew`, `adaptive_dt = true`, `gpu_only_mode = true`, `perf_mode = true`, `qoi_freq = 1`.

---

## Problem 1: Cylinder — flow not developing (ALL models)

**Symptom:** `res=0.000e+00`, `Cd=0` for all 20 models at all 3 Re.

**Root cause:** Fully-periodic BCs with `bulk_velocity_target = 1.0` and IBM cylinder. The solver drives bulk velocity to 1.0, but starting from uniform flow with no perturbation, the flow simply stays uniform. The IBM forces are zero because the "solid" region has the same velocity as the fluid.

**Evidence:** 2000 steps reached t=25 for most models — the simulation ran fine, it just never produced any flow features.

**Fix options:**
1. **Add initial perturbation** to break symmetry — random noise or asymmetric initialization
2. **Use non-periodic inflow/outflow BCs** (uniform inflow, convective outflow) — but our IBM setup assumes periodic
3. **Initialize with potential flow** around the cylinder — physically motivated, breaks symmetry

**Recommendation:** Option 1 is simplest. Add small random perturbation to initial velocity field. The `main_cylinder.cpp` currently does `solver.initialize_uniform(U_inf, 0.0)` — need to add noise after this.

---

## Problem 2: Hills — timing varies wildly between models (adaptive dt)

**Symptom:** Wall time to reach t=5.92 varies from 8.9s (EARSM-GS) to 197.5s (TBNN-Large). But turb% is only 2-40% — most of the cost variation comes from different time step sizes, not turbulence model evaluation.

**Key data (sorted by wall time to t=5.92):**

| Model | Wall (s) | Wall/t_phys | Turb ms/call | Turb % | dt effect |
|-------|---------|-------------|-------------|--------|-----------|
| earsm_gs | 8.9 | 1.5 | 0.105 | 2.4% | Large dt (low nu_t) |
| earsm_wj | 11.7 | 2.0 | 0.116 | 2.0% | |
| baseline | 13.8 | 2.3 | 0 | 0% | No turb model |
| earsm_pope | 17.4 | 2.9 | 0.162 | 1.9% | |
| mlp | 19.1 | 3.2 | 0.881 | 9.2% | |
| tbrf_10t | 21.0 | 3.5 | 0.398 | 3.8% | |
| sst | 29.8 | 5.0 | 0.257 | 1.7% | Smaller dt (transport) |
| tbnn_small | 31.9 | 5.4 | 2.880 | 18% | |
| tbrf_5t | 40.6 | 6.9 | 0.589 | 2.9% | |
| tbnn | 56.7 | 9.6 | 6.095 | 21.5% | |
| komega | 70.2 | 41.1 | 0.302 | 0.9% | **Tiny dt** |
| tbrf_1t | 75.6 | 12.8 | 0.981 | 2.6% | |
| mlp_large | 93.0 | 15.7 | 19.324 | 41.6% | |
| mlp_med | 102.6 | 17.3 | 4.340 | 8.5% | Confusing ordering |
| tbnn_large | 197.5 | 33.4 | 21.280 | 21.5% | |

**Anomalies:**
- **k-omega is slowest** (41× real-time) with only 0.9% turb cost — the transport equations produce large omega near walls, driving dt extremely small
- **EARSM-GS faster than baseline** — EARSM adds viscosity → smoother flow → larger dt → fewer steps
- **mlp_med slower than mlp_large** — different nu_t values drive different dt
- **TBRF ordering inverted** (1t > 5t > 10t in wall time) — more trees = different nu_t = different dt
- **Separation not detected** (sep=1.0) — 2000 steps / t=5.92 isn't enough convergence

**This is actually the correct metric for the paper** — the total cost to simulate a given physical time IS the fair comparison. But we need to:
1. Run long enough for QoIs to converge
2. Understand that the dt penalty IS part of the model's cost
3. Separate "model overhead per call" from "model changes dt which changes total steps"

---

## Problem 3: Sphere — MLP produces huge nu_t, crushes dt

**Symptom:** MLP/MLP-med/MLP-large on sphere reach t=0.31 in 200 steps while all other models reach t=4.69. The MLP's nu_t is so large it reduces dt by ~15×.

**This is a real physical effect** — the MLP is predicting too much turbulent viscosity, which is both a cost and accuracy issue. It means the MLP is wrong on this unseen geometry, which is actually a finding for the paper.

---

## Problem 4: Hills residual plateau at 2e-3

**Symptom:** Most models show final residual ≈ 2.019e-03, not decreasing further. k-omega is lower (1.4e-3), SST is 1.7e-3.

**Root cause:** Likely not enough steps for convergence at this grid. The dp_dx driven flow needs many flow-through times to reach steady state.

**Fix:** Run more steps (10K-50K) or accept that at 2e-3 the flow is quasi-converged for QoI extraction.

---

## Problem 5: Duct residuals not extracted

**Symptom:** `res=?` for all duct runs.

**Root cause:** Duct app likely doesn't print residual in the same format as hills/cylinder. Need to check output format.

---

## Plan for Production Runs

### Phase 1: Fix cylinder initialization (code change)

Add random perturbation after `initialize_uniform()` in `main_cylinder.cpp`:
```cpp
solver.initialize_uniform(U_inf, 0.0);
// Add perturbation to break symmetry for vortex shedding
solver.add_velocity_perturbation(0.01 * U_inf);  // 1% noise
```

Need to implement `add_velocity_perturbation()` or add noise directly. Must work on GPU (perturb on CPU before `sync_to_gpu`).

### Phase 2: Determine correct step counts

Target: enough physical time for QoIs to converge.

| Case | Physical time target | Reason |
|------|---------------------|--------|
| Hills | t=50 (5+ flow-throughs) | Steady separation needs many flow-throughs |
| Cylinder Re=100 | t=200 (30+ shedding cycles at St=0.165) | Need stable St and mean Cd |
| Cylinder Re=300 | t=150 | Similar |
| Cylinder Re=3900 | t=150 | Similar |
| Duct | t=50 | Steady secondary flows |
| Sphere Re=100 | t=50 | Steady wake |
| Sphere Re=200 | t=50 | Steady non-axisymmetric |
| Sphere Re=300 | t=100 (15+ shedding cycles) | Need stable St |

Step counts depend on dt, which depends on model. Worst case:
- Sphere + MLP-Large at dt~0.002: t=50 needs 25,000 steps × 650ms = 4.5 hours (too long)
- Sphere + baseline at dt~0.02: t=50 needs 2,500 steps × 2.3ms = 6 seconds

**Solution:** Set `max_steps` high enough for fastest dt, and use `max_time` (physical time limit) instead of step count to ensure fair comparison. OR accept that some models are so expensive they can't reach convergence in 2 hours — that IS the paper's finding.

### Phase 3: Cost metric for the paper

Report **three** cost metrics:
1. **Turb model cost per call** (turb_ms) — intrinsic model cost, independent of flow
2. **Wall time per physical time** (s/s_phys) — total cost including dt effects
3. **Breakdown:** turb%, poisson%, convection%, diffusion% — where does the time go?

The Pareto plot should use metric #2 (wall/phys) on x-axis — that's what a user actually pays.

### Phase 4: Re-run benchmark with fixes

1. Fix cylinder perturbation
2. Increase step counts / add physical time limits
3. Re-run on H200
4. Extract converged QoIs
5. Build Pareto plot

### Open questions

- Do we need a `max_time` config parameter? Currently only `max_steps` and `tol` control termination.
- Should we use fixed dt for fair comparison? (No — adaptive dt IS the realistic scenario)
- Is the MLP's huge nu_t on sphere a bug or a finding? (Finding — MLP doesn't generalize)
- The TBRF inverted cost ordering (1t slower than 10t) needs investigation — may be a nu_t/dt interaction
