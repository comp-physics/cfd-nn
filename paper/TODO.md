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

## Model Matrix (15 models total)

### Classical RANS (7 models)

| # | Model | Type |
|---|-------|------|
| 1 | None (laminar) | Baseline |
| 2 | Mixing length | Algebraic |
| 3 | k-omega | 2-eq transport |
| 4 | SST k-omega | 2-eq transport |
| 5 | EARSM-WJ | Nonlinear algebraic |
| 6 | EARSM-GS | Nonlinear algebraic |
| 7 | EARSM-Pope | Nonlinear algebraic |

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

## A Posteriori Cases (4 cases)

| Case | Dim | Grid | Cells | Poisson | Re | Training |
|------|-----|------|-------|---------|-----|----------|
| Periodic hills | 2D | 384×192×1 | 74K | FFT2D | Re_H=10595 | Trained |
| Cylinder | 2D | 384×288×1 | 111K | FFT2D (fully periodic) | Re=100 | Unseen |
| Square duct | 3D | 96×96×96 | 885K | FFT1D (uniform grid) | Re_b=3500 | Trained |
| Sphere | 3D | 192×128×128 | 3.15M | FFT 3D | Re=200 | Unseen |

**Grid changes from initial smoke test (March 23):**
- Hills: 196×96 → 384×192 (4× cells, better resolution)
- Cylinder: 256×192 → 384×288 (2.3×, 24 cells/D)
- Duct: 48×64×64 → 96×96×96 (4.5×, uniform grid for FFT1D, Re matched to training data)
- Sphere: 128×96×96 → 192×128×128 (2.7×, 13 cells/D)

**Duct Re fix**: nu changed from 0.001 (Re_tau~5000, way outside training) to 0.006 (Re_b~3500, top of training range).

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

## In Progress

### Training new model sizes (L40S job 5429862)
- [x] MLP-Medium (5→64→64→1) — training on L40S
- [ ] PI-TBNN-Small (5→32→32→10, beta=0.001) — queued
- [ ] PI-TBNN-Large (5→128→128→128→10, beta=0.001) — queued

---

## Remaining Work (Priority Order)

### 1. Complete new model training
- [ ] Verify MLP-Medium, PI-TBNN-Small, PI-TBNN-Large weights exported
- [ ] Run a priori evaluation on all new models
- [ ] Update training_methodology.md with new model specs

### 2. Full timing sweep on H200
- [ ] Run all 15+ models × 4 cases on H200 with updated grids
- [ ] Include TBRF-5t and TBRF-10t for forest size scaling
- [ ] Standardize: same binary, same H200, same step count

### 3. Production a posteriori runs (H200)
- [ ] Run all models × 4 cases at full iteration count (5K-15K steps)
- [ ] Extract velocity profiles, drag/lift, separation points
- [ ] Compare against reference data:
  - Hills: Breuer et al. (2009) LES
  - Cylinder: Cd≈1.35, St≈0.165 at Re=100
  - Duct: Pinelli et al. (2010) DNS secondary flows
  - Sphere: Johnson & Patel (1999) Cd, separation angle

### 4. Pareto plot (THE figure)
- [ ] x-axis: computational cost (turb model overhead ms/step)
- [ ] y-axis: a posteriori error (L2 norm vs reference)
- [ ] Multiple points per architecture (size variants trace out curves)
- [ ] Separate per case + combined
- [ ] Pareto frontier identification

### 5. Rerun solver throughput benchmarks on H200
- [ ] Throughput at 256³ on H200 (post-FFT-stride-fix)
- [ ] Update paper tables (currently V100/H100)

### 6. Update paper with final results
- [ ] results_aposteriori.tex — profiles and comparisons
- [ ] results_cost.tex — H200 timing, size scaling curves
- [ ] Update training_methodology.md with new models
- [ ] Pareto plot figure
- [ ] Architecture diagrams

### 7. Paper polish
- [ ] Consistent notation throughout
- [ ] Figure quality (all using plot_style.py)
- [ ] Abstract updated with final numbers

---

## Key Decisions

1. **GPU**: Production on H200, training on L40S (consistent with original models)
2. **Cases**: 4 cases — 2 trained + 2 unseen, 2 2D + 2 3D
3. **Models**: 15 total (7 classical + 8 ML with size variants)
4. **Grids**: Updated March 23 — larger grids, correct Re for duct
5. **Duct Poisson**: Uniform grid → FFT1D (avoids MG adaptive_projection artifact)
6. **TBRF**: GPU tree traversal (21× speedup, cheaper than MLP)
7. **MLP inputs**: Pope invariants (matching Ling et al. 2016)
8. **Cost measurement**: From same runs that produce accuracy results
9. **Multiple NN sizes**: 2-3 per architecture for Pareto curves

---

## File Index

| File | Status | Contents |
|------|--------|----------|
| `paper/main.tex` | Draft | Main document |
| `paper/sections/introduction.tex` | Draft | Background, gap, contributions |
| `paper/sections/methods_solver.tex` | Done | Solver, CaNS comparison, optimization |
| `paper/sections/methods_closures.tex` | Needs update | Add new model sizes |
| `paper/sections/methods_training.tex` | Needs update | Add new model specs |
| `paper/sections/results_apriori.tex` | Needs update | Add new model results |
| `paper/sections/results_aposteriori.tex` | TODO | Needs production data |
| `paper/sections/results_cost.tex` | Needs update | H200 + new models + TBRF GPU |
| `paper/sections/discussion.tex` | Needs update | TBNN saturation finding |
| `paper/sections/conclusions.tex` | Done | Summary, guidelines |
| `paper/training_methodology.md` | Needs update | Add new model specs |
| `scripts/paper/train_all_models.py` | Updated | Added 5 new model variants |
| `scripts/paper/train_remaining_sizes.sbatch` | Active | L40S job for remaining 3 models |
