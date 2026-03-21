# Paper Experiment Matrix: NN vs Traditional RANS Cost-Accuracy Tradeoff

## Goal

Determine whether NN-based turbulence closures on coarse grids can match the accuracy of traditional RANS models on fine grids, and at what computational cost.

## Test Cases

### Case 1: Turbulent Channel Flow (Re_tau = 180)

**Reference:** Moser, Kim & Mansour (1999) DNS — downloadable via `scripts/download_reference_data.sh`

**BCs:** Periodic x/z, no-slip y → **FFT Poisson** (fast)

**Why this case:** Cleanest comparison. Every model should "work" — the question is accuracy vs cost.

### Case 2: Periodic Hills (Re = 5,600)

**Reference:** Breuer et al. DNS/LES — available in McConkey dataset

**BCs:** Periodic x/z, no-slip y + IBM hill → **FFT Poisson** (fast)

**Why this case:** Separation and reattachment. Simple RANS models fail here. Tests whether NN closures generalize to complex physics.

## Quantities of Interest (QoIs)

### Channel flow QoIs (vs MKM DNS):

| QoI | What it tests | Who fails |
|-----|---------------|-----------|
| **Q1: Mean velocity u+(y+)** | Basic momentum balance | Nobody (all RANS get this OK) |
| **Q2: Reynolds shear stress -<u'v'>+(y+)** | Turbulent transport | Mixing length is rough, SST OK |
| **Q3: Normal stress anisotropy <u'u'>+ - <v'v'>+** | Anisotropy prediction | **All eddy-viscosity models fail** (can't predict by construction). Only EARSM and TBNN can. |
| **Q4: Wall shear stress (u_tau / Re_tau)** | Integral accuracy | Discriminates model quality |

### Periodic hills QoIs (vs Breuer DNS/LES):

| QoI | What it tests | Who fails |
|-----|---------------|-----------|
| **Q5: Separation point x_s/H** | Adverse pressure gradient response | Simple RANS overshoots |
| **Q6: Reattachment length x_r/H** | Recovery prediction | Sensitive to closure |
| **Q7: Skin friction Cf(x) profile** | Wall-bounded accuracy everywhere | Discriminates all models |
| **Q8: Mean velocity u(y) at x/H = 2, 4, 6** | Flow field accuracy at key stations | Separation zone accuracy |

### Error metrics:

For each QoI, compute:
- **L2 relative error** vs DNS reference: `||f_RANS - f_DNS||_2 / ||f_DNS||_2`
- Report as percentage

**Accuracy targets:**
- Q1 (u+): < 5% L2 error (easy for all models)
- Q2 (-<u'v'>+): < 10% L2 error
- Q3 (anisotropy): < 20% L2 error (hard — eddy viscosity models will fail)
- Q4 (u_tau): < 3% relative error
- Q5 (x_s): < 10% relative error
- Q6 (x_r): < 10% relative error
- Q7 (Cf): < 15% L2 error
- Q8 (u profiles): < 10% L2 error at each station

## Turbulence Models

| Model | Type | Anisotropy? | Config flag | Pre-trained weights |
|-------|------|-------------|-------------|-------------------|
| `baseline` | Algebraic | No | `--model baseline` | None needed |
| `sst` | Transport (2-eq) | No | `--model sst` | None needed |
| `earsm_pope` | Transport + EARSM | **Yes** | `--model earsm_pope` | None needed |
| `nn_mlp` | NN scalar nu_t | No | `--model nn_mlp --nn_preset mlp_channel_caseholdout` | Channel + PHLL |
| `nn_tbnn` | NN anisotropy | **Yes** | `--model nn_tbnn --nn_preset tbnn_channel_caseholdout` | Channel + PHLL |

**Key comparisons:**
- `baseline` vs `sst`: Does transport equation help?
- `sst` vs `earsm_pope`: Does anisotropy help?
- `sst` vs `nn_mlp`: Does NN improve scalar closure?
- `earsm_pope` vs `nn_tbnn`: NN vs traditional anisotropy model

## Grid Resolution Ladder

### Channel flow grids (2D):

| Grid | Nx × Ny | Total cells | y+ at wall | Role |
|------|---------|-------------|------------|------|
| **A (fine)** | 128 × 256 | 32,768 | ~0.7 | Reference-quality RANS |
| **B (medium)** | 64 × 128 | 8,192 | ~1.4 | Standard RANS |
| **C (coarse)** | 32 × 64 | 2,048 | ~2.8 | Can NN compensate? |

### Periodic hills grids (2D):

| Grid | Nx × Ny | Total cells | Role |
|------|---------|-------------|------|
| **A (fine)** | 256 × 128 | 32,768 | Reference-quality RANS |
| **B (medium)** | 128 × 64 | 8,192 | Standard RANS |
| **C (coarse)** | 64 × 32 | 2,048 | Can NN compensate? |

## Full Experiment Matrix

### Channel flow: 5 models × 3 grids = 15 runs

| | Grid A (128×256) | Grid B (64×128) | Grid C (32×64) |
|---|---|---|---|
| baseline | run | run | run |
| sst | run | run | run |
| earsm_pope | run | run | run |
| nn_mlp | run | run | run |
| nn_tbnn | run | run | run |

### Periodic hills: 5 models × 3 grids = 15 runs

Same matrix with PHLL-trained NN presets.

### Total: 30 runs (manageable on a single GPU in a few hours)

## Timing Methodology

For each run, record:
1. **Wall-clock per step** (ms) — from `TimingStats::print_summary()`
2. **Steps to converge** — until QoIs stabilize within tolerance
3. **Total wall-clock** — steps × cost/step
4. **Component breakdown** — convection, diffusion, Poisson, turbulence model

All runs on **same GPU, same compiler flags, same binary.** Only config changes.

Use `--warmup_iter 20` to exclude GPU init from timing.

## Paper Figures

### Figure 1: Cost vs Accuracy (the money plot)
- x-axis: Total wall-clock time to converged QoI
- y-axis: L2 error vs DNS for each QoI
- Each curve: one model (baseline, SST, EARSM, MLP, TBNN)
- Each point on curve: different grid resolution
- Pareto frontier shows optimal cost-accuracy tradeoff

### Figure 2: Per-step cost breakdown (stacked bar)
- One group per model
- Stacked bars: convection, diffusion, Poisson, turbulence model
- Shows where compute time goes

### Figure 3: Accuracy vs grid resolution (convergence)
- x-axis: 1/N (grid spacing)
- y-axis: L2 error for each QoI
- Separate panels for each QoI
- Shows which models converge faster with refinement

### Figure 4: Reynolds stress profiles
- u+(y+) for all models at Grid B
- -<u'v'>+(y+) for all models at Grid B
- Anisotropy for EARSM and TBNN only (others can't predict it)
- vs MKM DNS reference

### Figure 5: Periodic hills flow field
- Streamlines / separation bubble for SST vs EARSM vs TBNN at Grid B
- Cf(x) comparison vs DNS
- Velocity profiles at x/H = 2, 4, 6

## Solver Configuration (common to all runs)

```ini
scheme = skew
adaptive_dt = true
CFL_max = 0.5
simulation_mode = steady
tol = 1e-8
stretch_y = true
stretch_beta = 2.0
```

FFT Poisson solver auto-selected for both cases (periodic x/z).

## What This Answers

1. **"Is the NN closure overhead worth it?"** — Compare total cost for same accuracy target
2. **"When does NN win?"** — For QoIs that require anisotropy (Q3), NN-TBNN may win over EARSM at coarse grids
3. **"When does NN lose?"** — For simple QoIs (Q1, Q4), traditional models are cheaper
4. **"What's the overhead?"** — Per-step cost breakdown isolates NN inference cost
5. **"Does it generalize?"** — Channel (simple) vs periodic hills (complex) tests robustness
