# Paper Experiment Matrix: NN vs Traditional RANS Cost-Accuracy Tradeoff

## Goal

Determine whether NN-based turbulence closures on coarse grids can match the accuracy of traditional RANS models on fine grids, and at what computational cost.

## Test Cases

### Case 1: Turbulent Channel Flow (Re_tau = 180)

**Reference:** Moser, Kim & Mansour (1999) DNS — downloadable via `scripts/download_reference_data.sh`

**BCs:** Periodic x/z, no-slip y, stretched grid → **FFT Poisson** (fast)

**Domain:** [0, 2pi] x [-1, 1] x [0, pi]

**Why this case:** Cleanest comparison. Every model should "work" — the question is accuracy vs cost.

### Case 2: Periodic Hills (Re_H = 5,600)

**Reference:** Breuer et al. DNS/LES — available in McConkey dataset

**BCs:** Periodic x/z, no-slip y + IBM hill → **FFT Poisson** (fast)

**Domain:** [0, 9H] x [0, 3.035H] x [0, 4.5H]

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
| `nn_mlp` | NN scalar nu_t | No | `--model nn_mlp --nn_preset mlp_*_caseholdout` | Channel + PHLL |
| `nn_tbnn` | NN anisotropy | **Yes** | `--model nn_tbnn --nn_preset tbnn_*_caseholdout` | Channel + PHLL |

**Key comparisons:**
- `baseline` vs `sst`: Does transport equation help?
- `sst` vs `earsm_pope`: Does anisotropy help?
- `sst` vs `nn_mlp`: Does NN improve scalar closure?
- `earsm_pope` vs `nn_tbnn`: NN vs traditional anisotropy model

## Grid Resolution Ladder

### Channel flow grids (3D):

| Grid | Nx x Ny x Nz | Total cells | Role |
|------|-------------|-------------|------|
| **D (extra-fine)** | 256 x 192 x 256 | 12,582,912 | Over-resolved reference |
| **A (fine)** | 128 x 96 x 128 | 1,572,864 | Reference-quality RANS |
| **B (medium)** | 64 x 64 x 64 | 262,144 | Standard RANS |
| **C (coarse)** | 32 x 32 x 32 | 32,768 | Can NN compensate? |

### Periodic hills grids (3D):

| Grid | Nx x Ny x Nz | Total cells | Role |
|------|-------------|-------------|------|
| **D (extra-fine)** | 256 x 128 x 128 | 4,194,304 | Over-resolved reference |
| **A (fine)** | 128 x 64 x 64 | 524,288 | Reference-quality RANS |
| **B (medium)** | 64 x 32 x 32 | 65,536 | Standard RANS |
| **C (coarse)** | 32 x 16 x 16 | 8,192 | Can NN compensate? |

## Full Experiment Matrix

### Channel flow: 5 models x 4 grids = 20 runs

| | Grid D (256x192x256) | Grid A (128x96x128) | Grid B (64x64x64) | Grid C (32x32x32) |
|---|---|---|---|---|
| baseline | run | run | run | run |
| sst | run | run | run | run |
| earsm_pope | run | run | run | run |
| nn_mlp | run | run | run | run |
| nn_tbnn | run | run | run | run |

### Periodic hills: 5 models x 4 grids = 20 runs

Same matrix with PHLL-trained NN presets.

### Total: 40 runs on H200 GPU

## SLURM Configuration

- **Partition:** gpu-h200
- **Account:** gts-sbryngelson3
- **QOS:** embers
- **GPUs:** 1 per job
- **Max concurrent:** 40 jobs

### Time limits per grid level:

| Grid | Time limit | max_steps |
|------|-----------|-----------|
| D | 6 hours | 50,000 |
| A | 3 hours | 30,000 |
| B | 1 hour | 20,000 |
| C | 30 min | 10,000 |

## Runner Infrastructure

### Config files: `examples/paper_experiments/`

8 config files (case x grid), model specified at runtime via `--model` CLI:
- `channel_{D,A,B,C}.cfg`
- `hills_{D,A,B,C}.cfg`

### Submission scripts: `scripts/paper/`

| Script | What it does |
|--------|-------------|
| `submit_channel.sh` | Builds solver + submits 20 channel jobs |
| `submit_hills.sh` | Builds solver + submits 20 hills jobs |
| `submit_all.sh` | Submits all 40 jobs |
| `collect_results.sh` | Parses outputs into `results/paper/experiment_results.csv` |

### Output structure:

```
results/paper/
  channel/
    baseline_D/  baseline_A/  baseline_B/  baseline_C/
    sst_D/       sst_A/       sst_B/       sst_C/
    ...
  hills/
    baseline_D/  ...
  experiment_results.csv
```

### Usage:

```bash
# Submit everything
./scripts/paper/submit_all.sh

# Or submit one case at a time
./scripts/paper/submit_channel.sh
./scripts/paper/submit_hills.sh

# After jobs finish, collect results
./scripts/paper/collect_results.sh
```

## Timing Methodology

For each run, record:
1. **Wall-clock per step** (ms) — from `TimingStats::print_summary()`
2. **Steps to converge** — until QoIs stabilize within tolerance
3. **Total wall-clock** — steps x cost/step
4. **Component breakdown** — convection, diffusion, Poisson, turbulence model

All runs on **same GPU (H200), same compiler flags (nvc++ Release), same binary.** Only config changes.

Use `--warmup_steps 50` to exclude GPU init from timing.

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
verbose = false
perf_mode = true
```

FFT Poisson solver auto-selected for both cases (periodic x/z).

## What This Answers

1. **"Is the NN closure overhead worth it?"** — Compare total cost for same accuracy target
2. **"When does NN win?"** — For QoIs that require anisotropy (Q3), NN-TBNN may win over EARSM at coarse grids
3. **"When does NN lose?"** — For simple QoIs (Q1, Q4), traditional models are cheaper
4. **"What's the overhead?"** — Per-step cost breakdown isolates NN inference cost
5. **"Does it generalize?"** — Channel (simple) vs periodic hills (complex) tests robustness
