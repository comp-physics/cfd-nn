# Artifacts

## ci_metrics.json

GPU CI test run metrics from 2026-03-04 on NVIDIA H200 (cc90). Full test suite, 3/3 passed, gpu+hypre build. Git sha `e86f3ea` on branch `checking`.

## profiling_results.md

RANS campaign profiling: wall-clock timings over 50,000 steps for all turbulence models (baseline, k-omega, SST, GEP, EARSM variants, NN-MLP, NN-TBNN) on cylinder (Re=100), airfoil (Re=1000), and hills (Re=10595). Includes pre- and post-SST-GPU-port comparison. Key finding: NN-TBNN inference is 7-13x more expensive than NN-MLP; Poisson solver dominates at 24-30% of step time for all models.
