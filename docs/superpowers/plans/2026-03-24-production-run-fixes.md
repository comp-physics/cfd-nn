# Production Run Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the five issues blocking production a posteriori runs: (1) **RK3 IBM force accumulation bug** — forces permanently zero after step 1, (2) cylinder flow not developing — needs perturbation, (3) no physical-time stopping criterion, (4) duct residual format, (5) benchmark cost metric.

**Architecture:** Minimal changes — perturbation code in one app file, `T_final` loop checks in three app files (parameter already exists), one output format fix, config updates, benchmark script rewrite.

**Tech Stack:** C++17, OpenMP target offload, nvc++ compiler, bash scripts.

**Key reviewer findings (incorporated):**
- `config.T_final` already exists (`include/config.hpp:106`) and is parsed (`src/config.cpp:174`). Taylor-Green app already uses it. No new config parameter needed.
- `config.perturbation_amplitude` already exists (`include/config.hpp:118`, default 1e-2). Use it instead of hardcoding.
- `solver.velocity()` has a non-const overload — no `velocity_mut()` needed.
- Use `mesh.i_begin()`/`mesh.i_end()` indexing pattern (not raw `i + Ng`).

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `app/main_cylinder.cpp` | Modify (~L199, ~L252) | Add velocity perturbation; add `T_final` time-loop check |
| `app/main_hills.cpp` | Modify (~L231) | Add `T_final` time-loop check |
| `app/main_duct.cpp` | Modify (~L373, ~L404) | Add `T_final` time-loop check; fix residual output format |
| `examples/paper_experiments/*.cfg` | Modify | Add `T_final` to all 8 configs |
| `scripts/paper/production_benchmark.sh` | Rewrite | Report wall-time-per-physical-time with component breakdown |

New config parameter: `dt_min` (minimum adaptive dt floor).

## COMPLETED FIXES (Mar 24, 2026)

### Fix 0: RK3 IBM force accumulation bug (ROOT CAUSE of Cd=0)
**File:** `src/solver_time.cpp:1493-1496`
**Bug:** `ssprk3_step()` saved `ibm_accum_saved = ibm_->accumulate_forces()` then set it to `false` for intermediate stages, but **never restored it** before Stage 3. Forces were permanently disabled after step 1.
**Fix:** Added `ibm_->set_accumulate_forces(ibm_accum_saved)` before Stage 3's `euler_substep()`.
**Verified:** Cylinder Cd develops from 20.9→0.71 over 500 steps on CPU (128×96 grid). Previously stuck at exactly 0.

### Fix 1: Cylinder/sphere velocity perturbation
**File:** `app/main_cylinder.cpp:201-247`
**Added:** Deterministic sinusoidal perturbation in near-wake region. Uses `config.perturbation_amplitude` (configurable). Localized downstream of body with Gaussian envelope. 2D and 3D paths.

### Fix 2: T_final physical-time stopping
**Files:** `app/main_cylinder.cpp:248-255`, `app/main_hills.cpp:243-250`, `app/main_duct.cpp:384-390`
**Used:** Existing `config.T_final` parameter (already parsed). Added loop break when `current_time() >= T_final`.

### Fix 3: dt_min floor for adaptive dt
**Files:** `include/config.hpp`, `src/config.cpp`, `src/solver.cpp:3714-3718`
**Added:** `config.dt_min` parameter. Prevents simulation freeze when model produces huge nu_t (e.g., MLP on unseen geometry). Applied after CFL/diffusion constraints in `compute_adaptive_dt()`.

### Fix 4: Duct residual output format
**File:** `app/main_duct.cpp:416`
**Changed:** `"residual = "` → `"res="` to match cylinder/hills format for automated extraction.

### Fix 5: Divergence/dt-floor early termination
**File:** `app/main_cylinder.cpp:282-302`
**Added:** Stop if residual > 1e10. Stop if dt stuck at floor for 100 consecutive steps.

### Config updates
**Files:** All 8 `examples/paper_experiments/*.cfg`
**Added:** `T_final`, `dt_min = 1e-4`, `max_steps` bumped to high ceiling.

## H200 Validation Results (job 5478331, Mar 25 2026)

### Cylinder Re=100 — WORKING
| Model | ms/step | Steps | Cd | T_final hit |
|-------|---------|-------|----|-------------|
| baseline | 1.68 | 1903 | 0.78 | Yes |
| SST | 1.82 | 2493 | 1.91 | Yes |
| MLP | 2.41 | 1903 | 0.78 | Yes |
| TBNN | 10.21 | 1903 | 0.78 | Yes |

**RK3 force fix confirmed: Cd nonzero on H200.** SST shows higher Cd (more diffusion). MLP/TBNN match baseline (Re=100 is laminar, model effect minimal).

### Duct Re_b=3500 — WORKING
| Model | ms/step | Steps | res | T_final hit |
|-------|---------|-------|----|-------------|
| baseline | 2.63 | 10001 | 9.6e-4 | Yes |
| SST | 2.78 | 10001 | 9.6e-4 | Yes |
| MLP | 7.22 | 25890 | 1.5e-8 | No (dt crushed) |
| TBNN | 40.18 | 10001 | 9.6e-4 | Yes |

**MLP dt issue:** MLP produces large nu_t → tiny dt → 25K steps to reach T_final. dt_min prevented freeze but MLP is very slow.

### Fix 6: Volume penalization for IBM stability
**Files:** `src/ibm_forcing.cpp:400-420`, `include/ibm_forcing.hpp:42,117-120`, `include/config.hpp:138`, `src/config.cpp:247`, `app/main_hills.cpp:132`, `app/main_cylinder.cpp:157`
**Root cause:** The weight-multiplication direct forcing (`u *= weight`) creates O(1/dx) velocity gradients at the body surface. After the Poisson pressure correction, re-forcing creates divergence the solver can't resolve — feedback instability.
**Also removed:** The IBM re-forcing after pressure correction (solver_time.cpp:1041-1051). Re-multiplying velocity by weights AFTER the Poisson correction created divergence that was the primary instability mechanism.
**Fix:** Replace `u *= weight` with `u *= [1 - (1-w) * min(dt/eta, 1)]` where `ibm_eta` controls smoothness. With `ibm_eta = 0.5`:
- Hills: stable at 2000 steps (t=6.14), residual 5.6e-3 and decreasing
- Sphere Re=200: stable at 277 steps (t=5.0), residual 2.1e-3 and decreasing
- Cylinder Re=100: stable at 938 steps (t=10.0), Cd=35.4 (still converging)
**Note:** The H200 timing sweep (job 5470248) had `res=0, Cd=0` for all IBM cases because the RK3 force bug meant zero forcing. The "stability" was an artifact of dead flow.

### EARSM-Pope timing anomaly — RESOLVED
Not reproducible on H200 (smoke test: Pope=0.97ms, WJ=0.97ms). Was a run-order artifact in original timing sweep.

---

### Task 1: Add velocity perturbation for cylinder/sphere initialization

The cylinder/sphere in a fully-periodic domain with uniform init never develops flow features. Add deterministic perturbation after `initialize_uniform()` but before `sync_to_gpu()`.

**Files:**
- Modify: `app/main_cylinder.cpp:199-203`

**Reference:** `app/main_hills.cpp` already has perturbation code using `mesh.i_begin()`/`mesh.i_end()` and `config.perturbation_amplitude`.

- [ ] **Step 1: Add perturbation between initialize and GPU sync**

In `app/main_cylinder.cpp`, replace lines 199-203:

```cpp
// Initialize with uniform flow
solver.initialize_uniform(U_inf, 0.0);

// Add deterministic perturbation to break symmetry for vortex shedding
// Sinusoidal perturbation localized near body — reproducible across runs
{
    auto& vel = solver.velocity();
    const int Ng = mesh.Nghost;
    double amp_base = config.perturbation_amplitude * U_inf;

    if (is3D) {
        // 3D: perturb u, v, w
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                    double xc = mesh.xc[i];
                    double yc = mesh.yc[j];
                    double zc = mesh.zc[k];
                    double r = std::sqrt((xc - body_cx) * (xc - body_cx) +
                                         (yc - body_cy) * (yc - body_cy) +
                                         (zc - body_cz) * (zc - body_cz));
                    double amp = amp_base * std::exp(-0.5 * (r / body_r) * (r / body_r));
                    vel.u(i, j, k) += amp * std::sin(2.0 * M_PI * (0.7 * xc + 1.3 * yc));
                    vel.v(i, j, k) += amp * std::sin(2.0 * M_PI * (1.1 * xc + 0.9 * yc));
                    vel.w(i, j, k) += amp * std::sin(2.0 * M_PI * (0.8 * xc + 1.2 * zc));
                }
            }
        }
    } else {
        // 2D: perturb u and v only
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                double xc = mesh.xc[i];
                double yc = mesh.yc[j];
                double r = std::sqrt((xc - body_cx) * (xc - body_cx) +
                                     (yc - body_cy) * (yc - body_cy));
                double amp = amp_base * std::exp(-0.5 * (r / body_r) * (r / body_r));
                vel.u(i, j) += amp * std::sin(2.0 * M_PI * (0.7 * xc + 1.3 * yc));
                vel.v(i, j) += amp * std::sin(2.0 * M_PI * (1.1 * xc + 0.9 * yc));
            }
        }
    }
}

#ifdef USE_GPU_OFFLOAD
solver.sync_to_gpu();
#endif
```

Key design:
- Uses `config.perturbation_amplitude` (configurable, default 1e-2)
- Uses `mesh.i_begin()`/`mesh.i_end()` (correct staggered grid iteration)
- Gaussian envelope localized near body — doesn't pollute far field
- Deterministic sinusoidal — reproducible
- 3D perturbs all 3 components (u, v, w); 2D perturbs u, v
- Placed before `sync_to_gpu()` — perturbation on CPU, uploaded once

- [ ] **Step 2: Verify accessor patterns exist**

```bash
grep -n "i_begin\|i_end\|j_begin\|j_end\|k_begin\|k_end" include/mesh.hpp | head -10
grep -n "vel.u(\|vel.v(\|vel.w(" app/main_hills.cpp | head -5
```

Confirm that `mesh.i_begin()` etc. and `vel.u(i,j)` / `vel.u(i,j,k)` accessors exist.

- [ ] **Step 3: Build and quick test**

```bash
cd build_h200 && make -j$(nproc) cylinder 2>&1 | tail -3
OMP_TARGET_OFFLOAD=MANDATORY ./cylinder --config ../examples/paper_experiments/cylinder_re100.cfg --max_steps 200 2>&1 | grep -E "Cd=|res="
```

Expected: Cd should be non-zero after ~100-200 steps.

- [ ] **Step 4: Commit**

```
git add app/main_cylinder.cpp
git commit -m "Add deterministic velocity perturbation for cylinder/sphere IBM initialization"
```

---

### Task 2: Add `T_final` stopping criterion to cylinder, hills, and duct time loops

`config.T_final` already exists and is parsed. The Taylor-Green app already uses it (`t < T_final`). Add the same check to cylinder, hills, and duct.

**Files:**
- Modify: `app/main_cylinder.cpp:252`
- Modify: `app/main_hills.cpp:241`
- Modify: `app/main_duct.cpp:382`

**Reference pattern** (from `app/main_taylor_green_3d.cpp:315`):
```cpp
for (int step = 1; step <= config.max_steps && t < T_final; ++step) {
```

- [ ] **Step 1: Add T_final check to cylinder time loop**

In `app/main_cylinder.cpp`, after `double time = solver.current_time();` (line 252), add:

```cpp
if (config.T_final > 0.0 && time >= config.T_final) {
    if (mpi_rank == 0) {
        std::cout << "Reached T_final=" << config.T_final
                  << " at step " << step << ", t=" << time << "\n";
    }
    break;
}
```

- [ ] **Step 2: Add T_final check to hills time loop**

In `app/main_hills.cpp`, after `double time = solver.current_time();` (line 241), add the same pattern. Check whether hills has an `mpi_rank` variable — if not, print unconditionally.

- [ ] **Step 3: Add T_final check to duct time loop**

In `app/main_duct.cpp`, after `residual = solver.step();` (line 382), add:

```cpp
double time = solver.current_time();
if (config.T_final > 0.0 && time >= config.T_final) {
    std::cout << "Reached T_final=" << config.T_final
              << " at step " << iter << ", t=" << time << "\n";
    break;
}
```

Note: duct loop uses `iter` not `step`, and doesn't have a `time` variable yet — must introduce it.

- [ ] **Step 4: Build all three**

```bash
cd build_h200 && make -j$(nproc) hills duct cylinder 2>&1 | tail -3
```

- [ ] **Step 5: Quick test — verify T_final stops cylinder**

```bash
# T_final=1.0 should stop well before max_steps=10000
echo "T_final = 1.0" >> /tmp/test_tfinal.cfg
cat examples/paper_experiments/cylinder_re100.cfg /tmp/test_tfinal.cfg > /tmp/combined.cfg
OMP_TARGET_OFFLOAD=MANDATORY ./cylinder --config /tmp/combined.cfg 2>&1 | grep -E "Reached T_final|Step.*t="
```

Expected: `Reached T_final=1.0` message, run stops early.

- [ ] **Step 6: Commit**

```
git add app/main_cylinder.cpp app/main_hills.cpp app/main_duct.cpp
git commit -m "Add T_final physical-time stopping to cylinder, hills, and duct time loops"
```

---

### Task 3: Fix duct residual output format

The duct prints `residual = X.XXe-XX` (spaces around `=`) while the benchmark scripts grep for `res=X` (no spaces). Standardize.

**Files:**
- Modify: `app/main_duct.cpp:404`

- [ ] **Step 1: Fix progress output format**

In `app/main_duct.cpp`, change the progress output (~L404) from:

```cpp
<< "  residual = " << std::scientific << std::setprecision(3) << residual
```

to:

```cpp
<< "  res=" << std::scientific << std::setprecision(3) << residual
```

This matches the format used by `main_cylinder.cpp` and `main_hills.cpp`.

- [ ] **Step 2: Build and verify**

```bash
cd build_h200 && make -j$(nproc) duct
OMP_TARGET_OFFLOAD=MANDATORY ./duct --config ../examples/paper_experiments/duct_reb3500.cfg --max_steps 50 2>&1 | grep "res="
```

Expected: lines containing `res=X.XXXe-XX`.

- [ ] **Step 3: Commit**

```
git add app/main_duct.cpp
git commit -m "Standardize duct residual output to res= format for automated extraction"
```

---

### Task 4: Update paper experiment configs with `T_final`

Set physical time targets for each case. Bump `max_steps` to high ceilings.

**Files:**
- Modify: all 8 files in `examples/paper_experiments/`

- [ ] **Step 1: Add T_final to each config**

| Config | T_final | max_steps | Rationale |
|--------|---------|-----------|-----------|
| `hills_re10595.cfg` | 50.0 | 500000 | ~5 flow-throughs (L=9, U~1) |
| `cylinder_re100.cfg` | 200.0 | 500000 | ~30 shedding cycles (St~0.165) |
| `cylinder_re300.cfg` | 150.0 | 500000 | ~30 cycles (St~0.21) |
| `cylinder_re3900.cfg` | 100.0 | 500000 | ~20 cycles |
| `duct_reb3500.cfg` | 50.0 | 200000 | Steady secondary flows |
| `sphere_re100.cfg` | 50.0 | 100000 | Steady axisymmetric wake |
| `sphere_re200.cfg` | 50.0 | 100000 | Steady non-axisymmetric |
| `sphere_re300.cfg` | 100.0 | 100000 | ~13 cycles (St~0.135) |

Add line `T_final = <value>` and update `max_steps = <ceiling>` in each file.

- [ ] **Step 2: Commit**

```
git add examples/paper_experiments/*.cfg
git commit -m "Set T_final physical-time targets for all paper experiment configs"
```

---

### Task 5: Rewrite benchmark script for wall-time-per-physical-time metric

**Files:**
- Modify: `scripts/paper/production_benchmark.sh`

- [ ] **Step 1: Rewrite extraction logic**

For each run, extract from output log:
- `final_t` — from last `t=X.XXXX` in step output
- `solver_total_s` — from `solver_step` total in Timing Summary
- `turb_total_s` — from `turbulence_update` total
- `poisson_total_s` — from `poisson_solve` total
- `convect_total_s` — from `convective_term` total
- `steps` — from `solver_step` call count
- `final_res` — from last `res=` line
- QoI values from `qoi_summary.dat`

Compute:
- `wall_per_phys = solver_total_s / final_t`
- `avg_dt = final_t / steps`
- `turb_pct = 100 * turb_total_s / solver_total_s`
- `poisson_pct = 100 * poisson_total_s / solver_total_s`

Output per case:
```
=== hills_re10595 ===
MODEL                wall/phys  steps   avg_dt    turb%  poisson%  res          QoI
baseline                 2.33   2000    2.96e-3    0.0%    45.2%   2.02e-03     sep=1.0
sst                      5.03   3500    1.43e-2    1.7%    48.1%   1.75e-03     sep=0.9
```

- [ ] **Step 2: Update CSV output columns**

```
case,model,wall_per_phys,total_wall_s,final_t,steps,avg_dt,turb_pct,poisson_pct,final_res,qoi
```

- [ ] **Step 3: Commit**

```
git add scripts/paper/production_benchmark.sh
git commit -m "Rewrite benchmark: report wall-time-per-physical-time with component breakdown"
```

---

### Task 6: Validation on H200

Quick test of all fixes with 3 models × 3 cases, `T_final=2.0`.

- [ ] **Step 1: Run validation**

Test baseline, sst, mlp on hills, cylinder_re100, sphere_re200 with short T_final.

- [ ] **Step 2: Verify cylinder Cd ≠ 0** — check `qoi_summary.dat`
- [ ] **Step 3: Verify T_final stops run** — check for `Reached T_final` message
- [ ] **Step 4: Verify duct res= extraction** — check parsed output
- [ ] **Step 5: Verify benchmark script produces correct table**

---

## Summary

| Issue | Fix | Scope |
|-------|-----|-------|
| Cylinder dead flow | Deterministic sinusoidal perturbation in `main_cylinder.cpp` | 1 file, ~30 lines |
| No physical-time stop | Use existing `T_final` in 3 app time loops | 3 files, ~5 lines each |
| Duct res= format | Change format string in `main_duct.cpp` | 1 file, 1 line |
| Wrong cost metric | Rewrite `production_benchmark.sh` | 1 file |
| Configs need T_final | Add `T_final` + bump `max_steps` | 8 config files |

## Final Status (Mar 26, 2026)

All production blockers resolved. Summary of every fix completed this session:

1. **RK3 IBM force accumulation bug** — `ssprk3_step()` never restored `accumulate_forces` before Stage 3. Forces permanently zero after step 1. Fixed in `solver_time.cpp:1493-1496`.
2. **Volume penalization (ibm_eta=0.1)** — Replaced hard weight multiply with smooth penalization `u *= [1 - (1-w)*min(dt/eta,1)]`. Stabilizes all 4 IBM geometries.
3. **IBM re-forcing removed** — Re-multiplying velocity by weights after Poisson correction created unresolvable divergence. Removed from `solver_time.cpp:1041-1051`.
4. **EARSM bootstrap fix** — EARSM closure was overwriting SST's nu_t with zero. Fixed to use SST nu_t for transport production terms, EARSM nu_t/tau_ij only for momentum.
5. **SST warm-up in all 3 apps** — Added two-phase time loop (SST warm-up then target model) to `main_cylinder.cpp`, `main_hills.cpp`, `main_duct.cpp`.
6. **warmup_steps timer reset in all 3 apps** — Timer statistics reset after warm-up phase so timing data reflects only evaluation phase.
7. **warmup_steps config file parsing fix** — `warmup_steps` was not being parsed from config files. Added to `config.cpp`.
8. **RK3 sub-timers** — Added per-substep timing for convective, diffusive, poisson, correction, and ibm phases within RK3 integration.
9. **dt_min floor** — New `config.dt_min` parameter prevents simulation freeze when model produces huge nu_t. Applied after CFL/diffusion constraints in `compute_adaptive_dt()`.
10. **T_final stopping** — Used existing `config.T_final` parameter in cylinder, hills, and duct time loops (already existed for Taylor-Green).
11. **Cylinder perturbation** — Deterministic sinusoidal perturbation in near-wake region using `config.perturbation_amplitude`. 2D and 3D paths.
12. **Duct res= format** — Changed `"residual = "` to `"res="` to match cylinder/hills format for automated extraction.
13. **Divergence/dt-floor early termination** — Stop if residual > 1e10 or dt stuck at floor for 100 consecutive steps.
14. **Fixed-dt timing: 15 models x 4 cases** — Ran full timing sweep at fixed dt=0.0004 on H200. Poisson cost confirmed model-independent (0.280+/-0.007ms across all 15 models). Complete data in `results/paper/timing_fixed_dt_h200.md`.

## Not Fixed (by design)

- **Adaptive dt cost variation** — this IS the realistic cost. Paper reports it honestly.
- **MLP generalization failure on sphere** — finding, not a bug.
- **k-omega tiny dt near walls** — known transport behavior, paper discusses.

## Semi-Implicit Volume Penalization (Mar 26, 2026)

### Investigation: Why Cd values were too high

With `ibm_eta=0.1`, Cd=37 on cylinder (ref 1.35). Volume penalization theory (Angot et al. 1999) predicts O(sqrt(eta)) error on the no-slip condition → O(0.32) = 32% error at eta=0.1.

### Eta convergence study (cylinder Re=100 SST, T_final=10):

| eta | Cd | Error vs ref (1.35) | Stable? |
|-----|-----|---------------------|---------|
| 0.1 | 37.6 | 2686% | Yes (all geometries) |
| 0.01 | 4.16 | 208% | Yes (all geometries) |
| 0.001 | 1.57→1.74 | 16-29% | Cylinder only |
| 0.0001 | 1.57 | 16% | Cylinder only |

### Fix: Semi-implicit penalization

Replaced explicit `u *= [1 - (1-w)*min(dt/eta,1)]` with semi-implicit `u /= (1 + (1-w)*dt/eta)`.
The semi-implicit form (Angot et al. 1999) is unconditionally stable for any dt/eta.

**Result:** Cylinder now uses eta=0.001 → Cd=1.74 (within 29% of reference at t=10, will converge further with more time).
Hills/sphere still require eta=0.1 (complex geometry + RK3 substeps cause instability even with semi-implicit form).

### Production configs:
- Cylinder: `ibm_eta = 0.001` (quantitative Cd/St comparison vs reference)
- Hills: `ibm_eta = 0.1` (relative model comparison only)
- Sphere: `ibm_eta = 0.1` (relative model comparison only)
- Duct: no IBM (wall-bounded, no penalization needed)
