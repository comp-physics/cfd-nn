# Paper TODO

Tracking all remaining work for the NN turbulence closures paper.

## A Priori Evaluation — COMPLETE

- [x] Test set RMSE for all models
- [x] Component-wise RMSE breakdown
- [x] Per-case RMSE — KEY FINDING: TBNN overfits (0.085 val → 0.389 test)
- [x] Realizability violation rates
- [x] Scatter plots, Lumley triangle, error distributions
- [x] TBRF feature importance (λ₁ dominates at 43.4%)
- [x] PI-TBNN beta sweep (negative → positive result on generalization)

## Model Implementation — COMPLETE

- [x] MLP: Pope invariants on GPU (was using wrong 6 physical features — FIXED)
- [x] MLP-Large: same pipeline as MLP
- [x] TBNN: Pope invariants + tensor basis on GPU
- [x] PI-TBNN: same as TBNN, different weights
- [x] TBRF: CPU tree traversal (1/5/10-tree variants)
- [x] All 5 models tested on V100 GPU through full solver pipeline
- [x] Model weights committed to git
- [x] input_dim=5 validation (rejects old 6-feature models)

## Solver Validation — MOSTLY COMPLETE

- [x] V100 benchmark: 253 Mcells/s (6.5x faster than CaNS 4xV100)
- [x] H100 benchmark: 626 Mcells/s (pre-optimization), need to rerun
- [x] CaLES comparison: within 1.3x bandwidth-adjusted
- [x] nsys profiling: identified cuFFT stride bug
- [x] FFT stride fix: 1.89x solver speedup (141→74.5 ms at 256³ on V100)
- [x] Kernel breakdown documented (Poisson 30%, RHS 15%, stepping 28%)
- [x] Thomas solver evaluated and rejected (cuSPARSE PCR faster)
- [x] MPI scaling tested (2 GPUs slower — FFT transpose overhead)
- [ ] A100 benchmark with optimized code (job submitted, pending)
- [ ] H100 benchmark with optimized code (need to resubmit)
- [ ] Update paper tables with post-optimization numbers

## A Posteriori Evaluation — BLOCKER

This is the critical missing piece. Need to run models IN the solver and compare against DNS.

- [ ] **Channel Re_τ=180**: run baseline, SST, MLP, TBNN, TBRF on 128×96×128
- [ ] Compare u⁺(y⁺) and Reynolds stress profiles against MKM DNS
- [ ] **Periodic hills Re_H=5600**: run all models on 128×64×64
- [ ] Compare velocity profiles against Breuer et al.
- [ ] **Cylinder Re=100**: run all models on 192×128×4
- [ ] Compare C_d, C_l, St against reference
- [ ] Scripts exist (aposteriori_channel.sbatch, compare_channel_dns.py)
- [ ] MKM DNS data downloaded (results/paper/reference/)

## Cost Benchmarks — NEED REFRESH

Previous profiling was pre-MLP-fix and pre-FFT-optimization. Need fresh numbers.

- [ ] Reprofile ALL models at 128×96×128 (channel grid) on same GPU
- [ ] Include MLP (now works with Pope invariants!)
- [ ] Sub-phase breakdown for each model
- [ ] Grid scaling: 64³, 128³, 256³
- [ ] Generate cost breakdown stacked bar chart (Fig 13)

## Pareto Plot — BLOCKED

Needs both a posteriori error metrics AND cost benchmarks.

- [ ] Define error metric (L2 error in u⁺ profile vs DNS)
- [ ] Generate Pareto plot: cost (x) vs error (y) for all models
- [ ] This is THE figure of the paper

## Remaining Figures

- [ ] Fig 1: Solver architecture diagram (tikz)
- [ ] Fig 2: Model architecture diagrams (tikz)
- [x] Fig 3: Solver throughput — DONE (in paper, needs post-optimization update)
- [x] Fig 4-7: Training curves, scatter, Lumley, PI sweep — DONE
- [ ] Fig 8-12: A posteriori profiles (blocked on solver runs)
- [ ] Fig 13: Cost breakdown stacked bar
- [ ] Fig 14: **Pareto plot — GPU** (THE figure, blocked)
- [ ] Fig 17: TBRF tree count sweep (accuracy vs size vs cost)

## Remaining Tables

- [ ] Tab 3: A posteriori error metrics
- [ ] Tab 5: CPU profiling
- [ ] Tab 6: Sub-phase timing breakdown (have data, need to format)
- [x] Tab 7: Solver throughput vs CaNS — DONE (in paper)

## Writing — MOSTLY COMPLETE

- [x] All sections drafted
- [x] Results sections populated with actual data
- [ ] Update methods_solver.tex with post-optimization A100 numbers
- [ ] Write results_aposteriori.tex (blocked on runs)
- [ ] Update results_cost.tex with fresh profiling including MLP
- [ ] Final pass: consistency, polish

## Priority Order

1. **A posteriori channel runs** — submit SLURM jobs for all models
2. **Reprofile with MLP** — get fresh cost numbers with the fixed MLP
3. **Pareto plot** — combine a posteriori + cost data
4. **Update paper** — fill in remaining sections
5. Architecture diagrams (tikz) — can do anytime
