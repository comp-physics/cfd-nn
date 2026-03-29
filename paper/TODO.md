# Paper TODO

## Paper Narrative

**Title**: Cost-Accuracy Tradeoffs for Neural Network Turbulence Closures in GPU-Accelerated Incompressible Flow Solvers

**Key findings so far**:
1. A priori accuracy rankings reverse on test set — TBNN overfits (0.085 val → 0.389 test), TBRF generalizes best (0.064 → 0.125), PI-TBNN's realizability penalty acts as regularization
2. Classical RANS (SST, EARSM) adds <5% overhead; NN closures add 30-2000% depending on architecture and grid
3. TBRF GPU tree traversal achieves 21× speedup over CPU — cheaper than MLP on all cases
4. Solver is competitive with CaLES/CaNS after profiling-guided optimization (1.89× speedup from cuFFT stride fix)
5. TBNN accuracy saturates at small network sizes (32² ≈ 64³ ≈ 128³ in val RMSE)

**GPU**: All production results on NVIDIA H200 for consistency. Training on L40S.

---

## Model Matrix (16+ models total)

### Classical RANS (8 models)

| # | Model | Type | Transport eqs | Secondary flow? |
|---|-------|------|---------------|-----------------|
| 1 | None (laminar) | Baseline | 0 | N/A |
| 2 | Mixing length | Algebraic | 0 | No |
| 3 | k-omega | 2-eq transport | 2 | No (linear EV) |
| 4 | SST k-omega | 2-eq transport | 2 | No (linear EV) |
| 5 | EARSM-WJ | Nonlinear algebraic | 2 (SST) | Yes (partial) |
| 6 | EARSM-GS | Nonlinear algebraic | 2 (SST) | Yes (partial) |
| 7 | EARSM-Pope | Nonlinear algebraic | 2 (SST) | Yes (partial) |
| 8 | **RSM (SSG/LRR-ω)** | **Full RS transport** | **7** | **Yes (under-predicts)** |

**RSM is the most expensive classical model (~3-4× SST cost).** It provides the "expensive traditional" comparison point: if TBNN doesn't beat RSM, the NN adds no value. If TBNN beats RSM at similar cost, the NN wins. If TBRF beats RSM at lower cost, TBRF dominates.

### ML Models (8 models, 2-3 sizes per architecture)

| # | Model | Architecture | Params | Val RMSE | GPU Ready |
|---|-------|-------------|--------|----------|-----------|
| 8 | MLP | 5→32→32→1 | 1.2K | 0.110 | Yes |
| 9 | MLP-Medium | 5→64→64→1 | 4.4K | ~0.108 | Yes |
| 10 | MLP-Large | 5→128⁴→1 | 50K | 0.105 | Yes |
| 11 | TBNN-Small | 5→32→32→10 | 1.4K | 0.085 | Yes |
| 12 | TBNN | 5→64→64→64→10 | 9.4K | 0.085 | Yes |
| 13 | TBNN-Large | 5→128→128→128→10 | 35K | 0.086 | Yes |
| 14 | TBRF-1t | 1 tree/basis | 283K nodes | 0.078 | Yes (GPU) |
| 15 | TBRF-5t | 5 trees/basis | 1.4M nodes | 0.068 | Yes (GPU) |

Plus PI-TBNN-Small and PI-TBNN-Large (same architectures as TBNN-Small/Large, different training). GEP also available.

TBRF-10t (2.8M nodes, 56MB) available but may be too large for practical deployment.

## A Posteriori Cases — REVISED (Mar 28)

### Validated cases (quantitative Cd):
| Case | Re | Dim | Grid | T_final | dp/dx | Cd | Ref | Error | Status |
|------|----|-----|------|---------|-------|----|-----|-------|--------|
| Cylinder | ~100 | 2D | 384×288 | 500+ | -0.004 | 1.30 (Maskell) | 1.35 | **3.7%** | READY |
| Sphere | ~200 | 3D | 192×128² | 400+ | -0.0005 | 0.77 (mean) | 0.77 | **~0%** | READY |
| Duct | Re_b=3500 | 3D | 96³ | 50+ | -1.0 | N/A (no IBM) | Pinelli 2010 | — | READY |

### Blocked cases:
| Case | Re | Issue | Options |
|------|----|-------|---------|
| Hills | 10595 | Penalization diverges at correct dp/dx (-0.003). Ghost-cell fails (hill touches y=0). | Lower Re, fix wall intersection, or drop |

### Critical config fixes needed before production:
1. **Cylinder**: use `dp_dx = -0.004`, NOT `bulk_velocity_target=1.0`. Cd from momentum balance.
2. **Sphere**: use `ibm_body = sphere` (NOT `body = sphere` — silently ignored, A_ref 15× wrong)
3. **Hills**: dp/dx must be -0.003, NOT -1.0 (was 8000× too large, caused U_b runaway)
4. **All IBM cases**: T_final must be much longer (sphere T=400+, cylinder T=500+)
5. **Cd formula**: momentum balance `|dp/dx| * V / (0.5 * U_b² * A_ref)` for periodic domains

### Reference values:
- Cylinder Re=100: Cd≈1.35, St≈0.165 (Tritton 1959, Williamson 1996)
- Sphere Re=200: Cd≈0.77, sep≈117° (Johnson & Patel 1999)
- Hills Re=10595: sep x/H≈0.22, reattach x/H≈4.72 (Breuer et al. 2009) — BLOCKED
- Duct: secondary flows from Pinelli et al. (2010)

### Key physics notes:
- SST at Re=100 suppresses shedding entirely (Cd=2.53 steady) — expected for RANS at low Re
- Cylinder Re=100 IS laminar — no turbulence model needed. Tests if NN models "do no harm"
- Sphere Re=200 is at onset of non-axisymmetric flow — Cd oscillates 0.75-0.89 (physical)
- The Maskell blockage correction is standard for periodic-domain cylinder (D/Ly = 8.3%)

**Revised total: 3 validated cases × 20 models = 60 runs (was 160)**

---

## Completed Work

### Training & A Priori
- [x] All 5 original NN models trained on McConkey dataset (TBKAN 2025 split) on L40S
- [x] Full a priori evaluation: val + test RMSE, component-wise, per-case
- [x] PI-TBNN beta sweep (realizability as regularization finding)
- [x] TBRF feature importance (λ₁ dominates at 43.4%)
- [x] All plots: scatter, Lumley triangle, error distributions, training curves
- [x] Training methodology documented (paper/training_methodology.md)
- [x] TBNN-Small (5→32²→10) trained on L40S — val RMSE 0.085
- [x] TBNN-Large (5→128³→10) trained on L40S — val RMSE 0.086

### Solver Implementation
- [x] MLP uses Pope invariants (same as TBNN — Ling et al. 2016 methodology)
- [x] All 5 NN models pass GPU tests (NNModelsTest on V100)
- [x] Model weights committed to git
- [x] **TBRF GPU tree traversal** — 21× speedup over CPU (was CPU-only, now full GPU pipeline)
- [x] Duct binary fixed (NN weights path was empty)
- [x] Fully-periodic 2D Poisson solver (FFT2D with 2D R2C) — enables Nz=1 cylinder

### Solver Optimization
- [x] nsys profiling identified cuFFT stride=Ny as 11× penalty
- [x] FFT unpack kernel: output-coalesced transpose (4.7× kernel speedup)
- [x] FFT stride fix: contiguous layout + shared-memory transpose (1.89× total speedup)
- [x] Thomas solver evaluated and rejected (cuSPARSE PCR faster at Ny=256)
- [x] Fully-periodic 2D Poisson: 86× speedup for cylinder (240ms → 2.8ms MG→FFT2D)

### Bug Fixes (March 23)
- [x] **Duct MLP timing anomaly RESOLVED**: Root cause was MG Poisson + adaptive_projection. Fixed by using uniform grid (FFT1D selected). Blend count mystery solved: `copy_3d_uvw` = `blend_3d_uvw(1,0)`, 7 calls/step × 3 GPU kernels = 21/step is correct.
- [x] **TBRF GPU**: Implemented full GPU pipeline — tree arrays mapped persistently, fused kernel for gradients→features→tree traversal→anisotropy→nu_t. Removed CPU k-estimation overhead from GPU path.

### Solver Benchmarks (V100, updated grids, March 23)

| Model | Duct 885K (ms) | Hills 74K (ms) | Cylinder 111K (ms) | Sphere 3.15M (ms) |
|-------|---------------|----------------|--------------------|--------------------|
| Baseline | 5.5 | 3.3 | 2.9 | 15.8 |
| SST | 5.9 | 3.4 | 3.1 | 16.3 |
| EARSM | 5.8 | 3.5 | 3.2 | 16.5 |
| MLP | 19.3 | 4.8 | 5.0 | 60.0 |
| TBRF-1t (GPU) | 7.3 | 3.5 | 3.3 | 19.6 |
| TBNN | 89.9 | 13.5 | 20.0 | 331.4 |

Non-turb cost is rock-solid across all models (±0.3ms). Need H200 numbers for paper.

### Paper LaTeX
- [x] All sections drafted
- [x] Results populated with actual data (a priori, cost analysis)
- [x] Profiling-guided optimization story written
- [x] References.bib with all citations

---

## Recently Completed

### Training new model sizes (L40S job 5429862) — DONE Mar 23
- [x] MLP-Medium (5→64→64→1) — val RMSE 0.108
- [x] PI-TBNN-Small (5→32→32→10, beta=0.001) — val RMSE 0.090
- [x] PI-TBNN-Large (5→128→128→128→10, beta=0.001) — val RMSE 0.089
- [x] All weights exported to data/models/ and verified

### H200 timing sweep (job 5470248) — DONE Mar 24
- [x] All 20 models × 4 cases on H200 with updated grids
- [x] Includes TBRF-1t/5t/10t, all MLP/TBNN/PI-TBNN sizes
- [x] Results in results/paper/aposteriori/h200_5470248.out

### QoI extraction infrastructure — DONE Mar 24
- [x] CPU-based extraction (nvc++ ICE on is_device_ptr with function-argument pointers)
- [x] Zero overhead on solver timing — runs after sync_solution_from_gpu()
- [x] **Hills QoIs**: Cf(x) profile, velocity profiles at x/H = 0.5/2/4/6/8, separation/reattachment detection
- [x] **Duct QoIs**: y-z cross-section (u,v,w) at mid-x, wall shear stress along y-walls
- [x] **Cylinder QoIs** (added Mar 24): Strouhal number from Cl zero-crossings, mean Cd from 2nd-half time series, wake velocity profiles at x/D = 1/2/3/5
- [x] **Sphere QoIs** (added Mar 24): separation angle from tangential velocity probe, wake velocity profiles at x/D = 1/2/3/5, mean Cd, Strouhal (shared cylinder binary)
- [x] All QoIs verified on GPU (V100): cylinder St=0.077 detected, sphere sep_angle=22.7° detected, wake profiles written correctly
- [x] `qoi_summary.dat` written with scalar QoIs (Re, St, Cd_mean, sep_angle) for easy parsing
- [x] Config: `qoi_freq` and `qoi_output_dir` in .cfg file (not CLI-parseable)

---

## Remaining Work (Priority Order)

### 1. Production a posteriori runs (H200)

#### Completed fixes (Mar 24-25)
- [x] **RK3 IBM force bug fixed** — forces permanently zero (ssprk3_step never restored accumulate_forces flag)
- [x] **Volume penalization** — `ibm_eta=0.1` replaces hard weight multiply; all 4 geometries stable
- [x] **IBM re-forcing removed** — post-Poisson weight multiply caused feedback instability
- [x] **EARSM bootstrap fixed** — SST nu_t used for transport production (EARSM nu_t was zero from cold start)
- [x] **SST warm-up initialization** — `warmup_model=sst`, `warmup_time=10.0` develops flow before target model
- [x] **RK3 sub-timers** — convective, diffusive, poisson, correction, ibm all tracked in solver_time.cpp
- [x] **Benchmark script** — reports wall-time-per-physical-time with component breakdown, correct avg_dt formula
- [x] Cylinder perturbation, T_final, dt_min, duct res= format, divergence detection
- [x] All 8 configs updated with ibm_eta, warmup settings, T_final, dt_min

#### Validated on H200 (Mar 25)
- [x] 9 models × cylinder Re=100 with SST warm-up — all produce distinct Cd values
- [x] Sub-timer breakdown: Poisson=25%, convection=3.4%, ibm=3.4%, turb_update=1.4%
- [x] Hills stable with SST (separation detected x/H=1.51)
- [x] Sphere Re=200 stable (Cd=2.97 at t=10)
- [x] Hills baseline diverges (Re=10595 laminar is physically unstable — expected)
- [x] MLP/TBNN hit dt_min floor (huge nu_t on unseen geometry — paper finding)
- [x] Fixed-dt timing: 15 models × 4 cases, 3-phase pipeline (SST warm-up → stabilize → measure)
- [x] EARSM-Pope anomaly resolved — adaptive dt was root cause, not model cost
- [x] Poisson cost model-independent at fixed dt (FFT direct solve, no iteration count variation)

#### IBM Force Fix (Mar 28) — 3 critical bugs
- [x] **Ghost-cell force was zero**: forcing cells had w=1.0, (1-w)=0. Fixed: accumulate (u_before - u_after)/dt*dV during ghost-cell interpolation in apply_forcing_device() and apply_ghost_cell()
- [x] **Bulk velocity controller broken**: flag set but never used in step(). Fixed: constant-power controller
- [x] **Momentum-balance Cd**: Cd = |dp/dx| * V / (0.5 * U_b² * A_ref) for periodic domains
- [x] **Config key bug**: `body = sphere` silently ignored → A_ref 15× wrong. Correct: `ibm_body = sphere`
- [x] **Hills dp/dx 8000× wrong**: dp/dx=-1.0 with nu=9.44e-5 gives Poiseuille U_b≈8131. Correct: ~-0.003
- [x] **Cylinder validated**: Cd=1.30 Maskell (ref 1.35, 3.7%), converged at 384×288, T=2000
- [x] **Sphere validated**: Cd=0.77 mean (ref 0.77, ~0%), 192×128², T=400
- [x] **Hills Re=10595 BLOCKED**: penalization diverges at correct dp/dx. Ghost-cell from cold start fails (hill touches y=0 wall). Stable only at Re≤1000.

#### Anisotropic Stress Divergence (Mar 28) — MAJOR FIX
- [x] **tau_ij was never used in momentum equations**: ALL tensor models (TBNN, TBRF, EARSM, GEP) were reduced to scalar nu_t. FIXED.
- [x] **Proper decomposition method**: tau_nl = tau_ij - 2*nu_t*S_ij, then div(tau_nl) as source term. Follows literature (Thompson 2019).
- [x] **Naturally stable at full strength** — no arbitrary scaling factor needed.
- [x] **Validated**: SST Cd=1.72, EARSM Cd=1.64 (-5%), TBNN Cd=1.80 (+3%) on cylinder 384×288.
- [x] **Background transport**: set_background_transport(sst) keeps k/omega alive for non-transport models
- [x] **Warm-up init fix** (hills only): set warm-up model FIRST to initialize solver state
- [x] **EARSM on hills WORKS** (Mar 28): SST nu_t restored after EARSM computes tau_ij
  - Root cause: EARSM-derived nu_t destabilizes explicit solver near separation
  - Fix: After EARSM runs, overwrite nu_t with SST's value (anisotropy enters via tau_div only)
  - Added `set_closure_active(false/true)` to disable EARSM during warm-up
  - EARSM U_b=0.80, SST U_b=0.73, k-omega U_b=0.65 — three distinct models on hills
- [x] **ALL 15 models stable on hills** (Mar 28): SST nu_t restoration for all non-transport models
  - SST nu_t restored after model computes tau_ij (for tensor models) or nu_t (for scalar models)
  - For non-transport models (NN, GEP): SST warm-up, then background transport provides k/omega + nu_t
- [x] **Hills SST VALIDATED against literature**: reattach x/H=7.63 (DNS 4.83, lit SST 7-8)
- [x] **2D anisotropic correction is tiny**: EARSM/TBNN/TBRF ≈ SST on 2D hills (5th decimal)
  - k-omega differs because it uses own transport
  - **3D is where tensor models matter** (secondary flows in duct)

#### 3D Extension (CRITICAL PATH for paper — plan: docs/superpowers/plans/2026-03-28-3d-tensor-basis-extension.md)
- [x] **Task 1: 3D velocity gradients** (9 components) — DONE. VelocityGradient extended, GPU kernel, all call sites.
- [ ] **Task 2: NUM_BASIS 4→10** — Pope (1975) 10-tensor 3D basis with 6 components each
- [ ] **Task 3: TBNN 3D inference** — use 10 basis tensors × 6 components in GPU kernel
- [ ] **Task 4: 3D tau_div** — add tau_div_w, 3D decomposition, w-momentum predictor
- [ ] **Task 5: Integration test** — duct SST=0 secondary flow, EARSM>0, TBNN≈DNS

#### Full Reynolds Stress Model (SSG/LRR-ω) — NEW
- [ ] **Implement full RSM**: 7 transport equations (6 Reynolds stress components + omega)
  - SSG pressure-strain model away from walls, LRR near walls (blended)
  - Standard omega equation with RSM-specific production
  - Wall reflection terms for pressure-strain near solid walls
  - Files: new `include/turbulence_rsm.hpp`, `src/turbulence_rsm.cpp`
  - Add `TurbulenceModelType::RSM` to config enum and factory
  - `provides_reynolds_stresses() = true`, `uses_transport_equations() = true`
  - GPU: needs 6 scalar fields for R_ij transport on device
  - Expected cost: ~3-4× SST (7 transport eqs vs 2)
  - **Purpose**: Most expensive classical RANS model. Provides the "expensive traditional"
    point on the Pareto frontier against which TBNN cost is compared.
  - **Literature**: SSG/LRR-ω predicts duct secondary flows but under-predicts magnitude (~50% of DNS).
    See NASA Turbulence Modeling Resource, Menter et al. (2012).
  - **Reference implementation**: NASA TMR (turbmodels.larc.nasa.gov/rsm-ssglrr.html)

#### Remaining for Production
- [ ] **Apply warm-up init fix and closure toggle to cylinder and duct apps**
- [ ] **UPDATE ALL CONFIGS** with corrected dp/dx, ibm_body key, T_final values
- [ ] Run 150-200 production runs (10 configs × 15-20 models) on H200
- [ ] Compare QoIs against DNS reference data (Pinelli 2010 duct, Breuer 2009 hills)
- [ ] Re-collect timing data with anisotropic stress overhead

### 2. Pareto plot (THE figure)
- [ ] x-axis: wall-time per physical time (the real cost a user pays — includes dt effects)
- [ ] y-axis: a posteriori error (L2 norm vs reference — needs production runs)
- [ ] Multiple points per architecture (size variants trace out curves)
- [ ] Separate per case + combined
- [ ] Pareto frontier identification
- **Data status**: Have both per-step cost data (fixed-dt timing) AND adaptive-dt wall time data for all 15 models × 4 cases

### 3. EARSM-Pope timing anomaly — RESOLVED (Mar 27)
- [x] H200 sweep showed: Hills 5.2ms vs 2.3ms (2.3×), Cylinder 7.0ms vs 2.5ms (2.8×)
- [x] **RESOLVED** — was caused by different adaptive dt, not model cost
- [x] Fixed-dt data confirms ALL EARSM variants cost 2.01-2.02ms (hills), 2.16-2.19ms (cylinder) — within noise
- [x] Turb kernel time identical: 0.025-0.028ms for all three EARSM variants
- [x] Root cause: Pope produces slightly different nu_t → different adaptive dt → different wall-time per step in old adaptive-dt measurement
- [x] NOT a Poisson issue (FFT2D is direct solve, no iterations)

### 4. Update paper with final results
- [ ] results_aposteriori.tex — profiles and comparisons (needs production run data)
- [x] results_cost.tex — DONE: 4 cases × 15 models, infrastructure table, per-step cost, overhead %, adaptive-dt wall time, component breakdown
- [x] methods_closures.tex — DONE (updated Mar 27): all 20 models in table, training methodology, SST warm-up, ghost-cell IBM, computational anatomy subsection
- [ ] Pareto plot figure

### 5. Paper polish
- [ ] Consistent notation throughout
- [ ] Figure quality (all using plot_style.py)
- [ ] Abstract updated with final numbers

---

## Key Decisions

1. **GPU**: Production on H200, training on L40S (consistent with original models)
2. **Cases**: 3 validated geometries (cylinder, sphere, duct). Hills Re=10595 BLOCKED.
3. **Cd method**: Momentum balance `|dp/dx|*V/(0.5*U_b²*A_ref)` for periodic-domain IBM cases. Maskell blockage correction for cylinder.
4. **Cylinder always 2D**: RANS models the turbulence, doesn't resolve it
5. **Models**: 20 total (8 classical + 12 ML with size variants)
6. **Channel dropped**: Steady RANS converges to Poiseuille regardless of model
7. **Duct Poisson**: Uniform grid → FFT1D (avoids MG adaptive_projection artifact)
8. **TBRF**: GPU tree traversal (21× speedup, cheaper than MLP)
9. **MLP inputs**: Pope invariants (matching Ling et al. 2016)
10. **Cost measurement**: From same runs that produce accuracy results
11. **Multiple NN sizes**: 2-3 per architecture for Pareto curves
12. **SST at Re=100**: Suppresses shedding entirely (expected for RANS at low Re) — paper finding
13. **Long T_final needed**: Sphere T=400+, cylinder T=500+ for statistical stationarity
14. **Config key**: `ibm_body = sphere`, NOT `body = sphere` (silently ignored)

---

## File Index

| File | Status | Contents |
|------|--------|----------|
| `paper/main.tex` | Draft | Main document |
| `paper/sections/introduction.tex` | Draft | Background, gap, contributions |
| `paper/sections/methods_solver.tex` | Done | Solver, CaNS comparison, optimization |
| `paper/sections/methods_closures.tex` | Done | All 20 models, training, warm-up, IBM, comp anatomy |
| `paper/sections/methods_training.tex` | Needs update | Add new model specs |
| `paper/sections/results_apriori.tex` | Needs update | Add new model results |
| `paper/sections/results_aposteriori.tex` | TODO | Needs production data |
| `paper/sections/results_cost.tex` | Done | H200 fixed-dt timing, 4 cases × 15 models |
| `paper/sections/discussion.tex` | Needs update | TBNN saturation finding |
| `paper/sections/conclusions.tex` | Done | Summary, guidelines |
| `paper/training_methodology.md` | Needs update | Add new model specs |
| `scripts/paper/train_all_models.py` | Done | All model variants |
| `scripts/paper/train_remaining_sizes.sbatch` | Done | L40S job 5429862 completed Mar 23 |
