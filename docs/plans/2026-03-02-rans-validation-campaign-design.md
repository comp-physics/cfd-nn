# RANS Validation Campaign Design

**Date**: 2026-03-02
**Goal**: Produce a self-contained validation report proving that each RANS turbulence model in the CFD-NN solver produces correct results for channel flow at Re_tau ~ 180, with sufficient evidence (convergence, accuracy vs DNS, grid independence) to satisfy an expert reviewer.

## Models Under Test

8 models, all on Re_tau = 180 channel flow (nu = 0.005556, dp_dx = -1.0, h = 1.0):

| # | Model | Type | Validation | Notes |
|---|-------|------|-----------|-------|
| 1 | none | Laminar reference | Analytical Poiseuille | Control case, Re_tau = u_tau/nu |
| 2 | baseline | Algebraic (mixing length) | vs MKM DNS | Fix GPU u_tau bug first |
| 3 | gep | Algebraic (GEP) | vs MKM DNS | Weatheritt-Sandberg 2016 |
| 4 | earsm | Algebraic (EARSM) | vs MKM DNS | Wallin-Johansson 2000 |
| 5 | sst | Transport (k-omega SST) | vs MKM DNS | Menter 1994 |
| 6 | komega | Transport (standard k-omega) | vs MKM DNS | Wilcox 1988 |
| 7 | nn_mlp | Neural network (MLP) | Stability only | Untrained weights |
| 8 | nn_tbnn | Neural network (TBNN) | Stability only | Untrained weights |

## Physics Setup

- 2D channel, periodic x, no-slip walls at y = +/-1
- Grid: 64x128, stretched y (beta = 2.0), Lx = 2*pi
- Upwind convection scheme (stable for steady RANS)
- Adaptive dt with CFL_max = 0.5
- Convergence: residual < 1e-8 or max 50000 steps
- Re_tau = 180: nu = 0.005556, dp_dx = -1.0 => u_tau = 1.0

## Validation Metrics

### Per-model:
1. **Convergence history**: residual vs step (log-scale)
2. **Re_tau**: from wall shear stress, compare to MKM 178.12
3. **u+(y+)**: mean velocity in wall units, overlay on MKM DNS
4. **L2 error**: ||u+_computed - u+_MKM||_2 (interpolated to MKM y+ points)
5. **nu_t/nu profile**: eddy viscosity ratio vs y/h
6. **Momentum balance**: integral of dp/dx across channel = wall shear sum

### Grid convergence (SST only):
- 3 grids: 32x64, 64x128, 128x256
- u+(y+) overlay showing convergence
- Re_tau vs grid size

## Prerequisite Fix

**Baseline GPU u_tau bug**: `turbulence_baseline.cpp` reads host velocity array for wall shear computation. On GPU, this data is stale (unmapped) so u_tau ~ 0 and nu_t ~ 0. Fix: compute u_tau from GPU-resident velocity via the TurbulenceDeviceView.

## Deliverables

### Files created:
1. `src/turbulence_baseline.cpp` — Fix GPU u_tau computation (~50 LOC)
2. `scripts/rans_validation/run_validation.sbatch` — SLURM job script
3. `scripts/rans_validation/configs/*.cfg` — Per-model config files (8 + 2 grid variants)
4. `scripts/rans_validation/analyze.py` — Python analysis + plot generation
5. `docs/RANS_VALIDATION_REPORT.md` — Final validation report

### Output structure:
```
output/rans_validation/
├── logs/             # Raw solver stdout per model
├── plots/            # Generated PNGs
│   ├── u_plus_all_models.png
│   ├── residual_history.png
│   ├── nu_t_profiles.png
│   └── grid_convergence_sst.png
├── data/             # Extracted CSV profiles
└── summary.json      # Machine-readable results table
```

### Report structure:
1. Executive summary (pass/fail table)
2. Setup (grid, BCs, physics parameters)
3. Convergence results (all models)
4. Accuracy results (models 1-6 vs MKM DNS)
5. Grid convergence (SST on 3 grids)
6. Stability results (NN models 7-8)
7. Known limitations
8. Conclusions

## Implementation Steps

1. Fix baseline GPU u_tau bug
2. Create config files for all 8 models + 2 grid variants
3. Create SLURM submission script
4. Create Python analysis script
5. Run validation on GPU cluster
6. Generate plots and write report

## Reference Data

MKM DNS at Re_tau = 178.12 (Moser, Kim & Mansour 1999):
- Location: `data/reference/mkm_retau180/chan180/profiles/`
- `chan180.means`: columns y, y+, Umean, dUmean/dy, ...
- `chan180.reystress`: columns y, y+, R_uu, R_vv, R_ww, R_uv, ...
- Normalization: U_tau and h (half-channel height)
