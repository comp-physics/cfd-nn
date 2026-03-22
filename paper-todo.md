# Paper TODO

Tracking all remaining work for the NN turbulence closures paper.
Full details in `docs/paper/paper_roadmap.md`.

## A Priori Evaluation

- [x] Test set RMSE for all models (PH α=1.2 + CBFS Re=13700) — DONE, see results/paper/apriori/metrics.json
- [x] Component-wise RMSE breakdown (b_11, b_12, b_13, b_22, b_23, b_33) — DONE
- [x] Per-case RMSE (PH vs CBFS separately) — DONE. KEY FINDING: TBNN overfits (0.085 val → 0.389 test, CBFS=0.457)
- [x] Realizability violation rates — DONE (TBNN 1.65% val, TBRF 2.47% val)
- [x] Write `scripts/paper/evaluate_apriori.py` — DONE
- [x] Download MKM DNS reference data — DONE, results/paper/reference/
- [ ] Scatter plots: predicted vs true b_ij on test set
- [ ] Lumley triangle plots for each model
- [ ] Error distribution histograms
- [ ] TBRF feature importance analysis

## TBRF C++ Inference

- [ ] C++ tree loader for `trees.bin` binary format
- [ ] Tree traversal inference function
- [ ] `TurbulenceModelType::NNTBRF` enum + config parsing + factory
- [ ] Validate C++ predictions match Python (float32 tolerance)
- [ ] Benchmark 1/5/10-tree variants on cylinder/airfoil/hills

## Solver Validation

- [ ] Install CaNS on cluster
- [ ] Run CaNS channel flow benchmark (256³, Re_τ=180, same GPU)
- [ ] Run our solver on identical problem
- [ ] Compare wall-clock per step and Mcells/s
- [ ] Formal throughput benchmarks at 256³ and 512³ grids

## A Posteriori Evaluation — GPU

- [ ] Channel Re_τ=180: all 6 models on grid A, extract u⁺(y⁺) + Reynolds stress profiles
- [ ] Channel grid sensitivity: TBNN + SST on grids A-D
- [ ] Periodic hills Re_H=5600: all 6 models, velocity profiles at key x-stations, separation/reattachment
- [ ] Cylinder Re=100: all 6 models, C_d, C_l, St
- [ ] Download MKM DNS reference data
- [ ] Download/digitize Breuer hills reference data
- [ ] Write profile extraction + comparison scripts

## A Posteriori Evaluation — CPU

- [ ] Channel Re_τ=180: all 6 models on CPU (same grid)
- [ ] Periodic hills: all 6 models on CPU
- [ ] Cylinder: all 6 models on CPU
- [ ] Record wall-clock times for CPU vs GPU comparison

## Cost Analysis

- [ ] Collect sub-phase timings: gradient, features, tensor basis, inference, postprocess
- [ ] Grid scaling runs: baseline + MLP + TBNN + TBRF-10 on 64³, 128³, 256³
- [ ] Peak GPU memory for each model
- [ ] Theoretical FLOPs/cell vs measured throughput → arithmetic intensity
- [ ] CPU vs GPU speedup table by closure type

## Figures

- [ ] Fig 1: Solver architecture diagram (fractional-step pipeline + closure plug-in)
- [ ] Fig 2: Model architecture diagrams (MLP, TBNN, TBRF side-by-side)
- [ ] Fig 3: Solver throughput vs CaNS benchmark
- [ ] Fig 4: Training/validation curves
- [ ] Fig 5: Test set scatter (predicted vs true b_ij)
- [ ] Fig 6: Lumley triangle
- [ ] Fig 7: PI-TBNN beta sweep
- [ ] Fig 8: Channel u⁺(y⁺) profiles vs DNS
- [ ] Fig 9: Channel Reynolds stress profiles vs DNS
- [ ] Fig 10: Hills velocity profiles at x-stations
- [ ] Fig 11: Hills separation bubble contours
- [ ] Fig 12: Cylinder C_d convergence
- [ ] Fig 13: Cost breakdown stacked bar (by phase, each model)
- [ ] Fig 14: **Cost-accuracy Pareto plot — GPU** (THE figure)
- [ ] Fig 15: Cost-accuracy Pareto plot — CPU
- [ ] Fig 16: Inference cost scaling with grid size
- [ ] Fig 17: TBRF tree count sweep (accuracy vs size vs cost)

## Tables

- [ ] Tab 1: Model summary (architecture, params, FLOPs/cell, size, deployability)
- [ ] Tab 2: A priori RMSE (val + test, overall + per-component)
- [ ] Tab 3: A posteriori error metrics (all models × all cases)
- [ ] Tab 4: Wall-clock profiling — GPU
- [ ] Tab 5: Wall-clock profiling — CPU
- [ ] Tab 6: Sub-phase timing breakdown
- [ ] Tab 7: Our throughput vs CaNS
- [ ] Tab 8: TBRF tree count sweep
- [ ] Tab 9: PI-TBNN beta sweep
- [ ] Tab 10: GPU vs CPU speedup by closure type

## Writing

- [ ] Introduction
- [ ] Methods: solver architecture + CaNS comparison
- [ ] Methods: classical closures (with operation counts)
- [ ] Methods: NN architectures
- [ ] Methods: training procedure
- [ ] Methods: solver integration + cost model (computational anatomy)
- [ ] Results: solver validation
- [ ] Results: a priori evaluation
- [ ] Results: a posteriori evaluation
- [ ] Results: computational cost analysis
- [ ] Results: cost-accuracy tradeoff (Pareto)
- [ ] Discussion
- [ ] Conclusions
- [ ] Abstract
- [ ] Appendix: full training methodology
