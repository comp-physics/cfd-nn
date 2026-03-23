# Paper TODO

## Paper Narrative

**Title**: Cost-Accuracy Tradeoffs for Neural Network Turbulence Closures in GPU-Accelerated Incompressible Flow Solvers

**Key findings so far**:
1. A priori accuracy rankings reverse on test set — TBNN overfits (0.085 val → 0.389 test), TBRF generalizes best (0.064 → 0.125), PI-TBNN's realizability penalty acts as regularization
2. NN closures cost 50-1400% overhead depending on grid size and architecture (SST costs 3%)
3. Our solver is competitive with CaLES/CaNS after profiling-guided optimization (1.89× speedup from cuFFT stride fix)
4. The cost depends strongly on whether the case is 2D (cheap) or 3D (expensive)

**GPU**: All production results on NVIDIA H200 for consistency.

---

## A Posteriori Cases (4 cases × 11 models)

| Case | Dim | Grid | Cells | Poisson | Training | Physics | Re |
|------|-----|------|-------|---------|----------|---------|-----|
| Periodic hills | 2D | 196×96×1 | 18.8K | FFT2D | Trained | Separation | Re_H=5600 |
| Cylinder | 2D | 256×192×1 | 49.2K | FFT2D (fully periodic) | Unseen | Bluff body wake | Re=100 |
| Square duct | 3D | 48×64×64 | 197K | Multigrid | Trained | Secondary flows | Re_b=2000 |
| Sphere | 3D | 128×96×96 | 1.18M | FFT 3D | Unseen | 3D wake | Re=100 |

**Grid resolution notes**:
- Hills: 196×96 gives ~22 cells per hill height, adequate for RANS
- Cylinder: 256×192 in domain 16D×12D gives ~16 cells/D. May need refinement.
- Duct: 48×64×64 with stretching (β=2.0) in y and z. Adequate for Re_b=2000 RANS.
- Sphere: 128×96×96 in domain 15D×8D×8D gives ~8.5 cells/D. May need refinement to 160×128×128.

**All 11 RANS models**:

| # | Model | Type | Category |
|---|-------|------|----------|
| 1 | None (laminar) | — | Baseline |
| 2 | Mixing length | Algebraic | Classical |
| 3 | k-omega | 2-eq transport | Classical |
| 4 | SST k-omega | 2-eq transport | Classical (industry standard) |
| 5 | EARSM-WJ | Nonlinear algebraic | Classical |
| 6 | EARSM-GS | Nonlinear algebraic | Classical |
| 7 | EARSM-Pope | Nonlinear algebraic | Classical |
| 8 | GEP | Symbolic regression | ML (algebraic) |
| 9 | MLP (5→32→32→1) | Neural network | ML (scalar closure) |
| 10 | TBNN (5→64³→10) | Neural network | ML (tensor closure) |
| 11 | TBRF (1-tree) | Random forest | ML (tensor closure) |

---

## Completed Work

### Training & A Priori
- [x] All 5 NN models trained on McConkey dataset (TBKAN 2025 split)
- [x] Full a priori evaluation: val + test RMSE, component-wise, per-case
- [x] PI-TBNN beta sweep (realizability as regularization finding)
- [x] TBRF feature importance (λ₁ dominates at 43.4%)
- [x] All plots: scatter, Lumley triangle, error distributions, training curves
- [x] Training methodology documented (docs/paper/training_methodology.md)

### Solver Implementation
- [x] MLP uses Pope invariants (same as TBNN — Ling et al. 2016 methodology)
- [x] All 5 NN models pass GPU tests (NNModelsTest on V100)
- [x] Model weights committed to git
- [x] TBRF C++ inference (CPU tree traversal)
- [x] Duct binary fixed (NN weights path was empty)
- [x] Fully-periodic 2D Poisson solver (FFT2D with 2D R2C) — enables Nz=1 cylinder

### Solver Optimization
- [x] nsys profiling identified cuFFT stride=Ny as 11× penalty
- [x] FFT unpack kernel: output-coalesced transpose (4.7× kernel speedup)
- [x] FFT stride fix: contiguous layout + shared-memory transpose (1.89× total speedup)
- [x] Thomas solver evaluated and rejected (cuSPARSE PCR faster at Ny=256)
- [x] Per-case nsys profiles: hills, cylinder, duct, sphere kernel breakdown
- [x] Fully-periodic 2D Poisson: 86× speedup for cylinder (240ms → 2.8ms MG→FFT2D)

### Solver Benchmarks
- [x] V100: 253 Mcells/s at 256³ (post-optimization)
- [x] H100: 626 Mcells/s at 256³ (pre-FFT-stride-fix, needs rerun)
- [x] A100: 400 Mcells/s at 256³ (pre-FFT-stride-fix, needs rerun)
- [x] CaNS comparison: 6.5× total throughput advantage vs 4× V100
- [x] CaLES comparison: within 1.3× bandwidth-adjusted
- [x] MPI scaling: 2× H100 slower than 1× (FFT transpose overhead)

### Paper LaTeX
- [x] All sections drafted (intro, methods, results, discussion, conclusions, appendix)
- [x] Results populated with actual data (a priori, cost analysis, solver validation)
- [x] Profiling-guided optimization story written (methods_solver.tex)
- [x] Kernel breakdown and cost anatomy documented (results_cost.tex)
- [x] References.bib with all citations

---

## In Progress

### H200 Smoke Test (job 5369348)
- 4 cases × 11 models × 50 steps = 44 runs
- Validates all models on all cases before production runs
- Using corrected Nz=1 cylinder with FFT2D

### H200 A Posteriori Channel (job 5360870)
- May need to cancel — channel converges to laminar Poiseuille with RANS
- SST gave same result as baseline (no turbulence without instability)
- MLP did change the flow (Re_tau 278 → 250)

---

## Remaining Work (Priority Order)

### 1. Validate smoke test results
- [ ] Check all 44 runs complete on H200
- [ ] Verify NN models produce different results from baseline (especially for 3D cases)
- [ ] Check transport models (SST, EARSM) — may need more steps or turbulent initial condition
- [ ] Verify grid convergence (especially sphere at 8.5 cells/D)

### 2. Production a posteriori runs
- [ ] Run all 11 models × 4 cases at full iteration count (10K-15K steps) on H200
- [ ] Extract velocity profiles, drag/lift coefficients, separation points
- [ ] Compare against reference data:
  - Hills: Breuer et al. (2009) LES — separation/reattachment x
  - Cylinder: Established reference — Cd≈1.35, St≈0.165 at Re=100
  - Duct: Pinelli et al. (2010) DNS — secondary flow patterns
  - Sphere: Johnson & Patel (1999) — Cd, separation angle at Re=100
- [ ] Compute L2 error norms for Pareto plot

### 3. Cost benchmarks (from same production runs)
- [ ] Extract timing from production run logs (all on same H200)
- [ ] Sub-phase breakdown for each NN model
- [ ] Cost vs grid size (2D hills 19K → 3D sphere 1.18M)

### 4. Pareto plot (THE figure)
- [ ] x-axis: computational cost (ms/step or overhead %)
- [ ] y-axis: a posteriori error (L2 norm vs reference)
- [ ] One point per model, labeled
- [ ] Pareto frontier identification
- [ ] Separate plot per case, plus combined

### 5. Rerun solver benchmarks on H200
- [ ] Throughput benchmark at 256³ on H200 (post-FFT-stride-fix)
- [ ] nsys kernel breakdown on H200
- [ ] Update paper tables with H200 numbers (currently V100/H100)

### 6. Update paper with final results
- [ ] results_aposteriori.tex — fill with actual profiles and comparisons
- [ ] results_cost.tex — update profiling tables for H200, add MLP numbers
- [ ] Update all tables to use H200 numbers consistently
- [ ] Pareto plot figure
- [ ] Architecture diagrams (tikz) — Fig 1, Fig 2

### 7. Grid convergence study
- [ ] Run sphere at 160×128×128 (2.6M cells) to verify convergence
- [ ] Run cylinder at 384×256 to verify convergence
- [ ] Report in appendix or methods

### 8. Paper polish
- [ ] CaLES comparison moved to appendix (different GPU)
- [ ] Consistent notation throughout
- [ ] Figure quality check (all using plot_style.py)
- [ ] Abstract updated with final numbers

---

## Key Decisions Made

1. **GPU**: All production results on H200 for consistency
2. **Cases**: 4 cases (hills, cylinder, duct, sphere) — 2 trained + 2 unseen, 2 2D + 2 3D
3. **Models**: All 11 RANS models (7 classical + 4 ML)
4. **MLP inputs**: Pope invariants (same as TBNN, matching Ling et al. 2016)
5. **Cylinder**: True 2D (Nz=1) with fully-periodic FFT2D Poisson
6. **CaLES comparison**: Appendix only (different GPU), main paper uses H200 throughout
7. **Cost measurement**: From the same runs that produce accuracy results (not artificial benchmarks)

---

## File Index

| File | Status | Contents |
|------|--------|----------|
| `paper/main.tex` | Draft | Main document, abstract with key findings |
| `paper/sections/introduction.tex` | Draft | Background, gap, contributions |
| `paper/sections/methods_solver.tex` | Done | Solver architecture, CaNS comparison, optimization story |
| `paper/sections/methods_closures.tex` | Done | All 11 models, FLOPs/cell, computational anatomy |
| `paper/sections/methods_training.tex` | Done | Dataset, features, training procedure |
| `paper/sections/results_apriori.tex` | Done | Val/test RMSE, overfitting finding, PI-TBNN sweep |
| `paper/sections/results_aposteriori.tex` | TODO | Needs production run data |
| `paper/sections/results_cost.tex` | Needs update | Has H100 data, needs H200 + MLP |
| `paper/sections/discussion.tex` | Done | Overfitting, regularization, recommendations |
| `paper/sections/conclusions.tex` | Done | Summary, practical guidelines |
| `paper/sections/appendix.tex` | Draft | Training methodology, CaLES comparison |
| `paper/references.bib` | Done | 14 references |
| `paper/TODO.md` | This file | |
| `paper/roadmap.md` | Outdated | Superseded by this TODO |
| `paper/training_methodology.md` | Done | Full training specification |
| `scripts/paper/` | Done | Training, evaluation, plotting scripts |
| `results/paper/` | Partial | A priori done, a posteriori pending |
| `data/models/*_paper/` | Committed | All model weights in git |
