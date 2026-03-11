# RANS Turbulence Model Campaign

Systematic evaluation of 9 turbulence closures across 4 IBM flow geometries on GPU (H100).

## Flow Cases

| Case | Re | Domain | Grid | IBM Body |
|------|----|--------|------|----------|
| Cylinder | 100 | 30x20x&pi; | 192x128x4 | Circular cylinder (r=0.5) |
| Airfoil | 1000 | 40x30x&pi; | 256x192x4 | NACA 0012 (chord=1, AoA=5&deg;) |
| Backward-facing step | 5000 | 40x6x&pi; | 384x128x4 | Step (height=1) |
| Periodic hills | 10595 | 9x3.036x&pi; | 192x128x4 | Hill profile (Mellen et al.) |

## Turbulence Models

| Model | Type | Config suffix |
|-------|------|---------------|
| Baseline (mixing length) | Algebraic | `_baseline` |
| k-&omega; | Transport | `_komega` |
| k-&omega; SST | Transport | `_sst` |
| GEP | Algebraic | `_gep` |
| EARSM (Wallin-Johansson) | Algebraic | `_earsm_wj` |
| EARSM (Gatski-Speziale) | Algebraic | `_earsm_gs` |
| EARSM (Pope) | Algebraic | `_earsm_pope` |
| NN-MLP | Neural network | `_nnmlp` |
| NN-TBNN | Neural network | `_nntbnn` |

Hills also has 2 extra configs with physics-informed loss (`_nnmlp_phll`, `_nntbnn_phll`), totaling **38 jobs**.

## Running the Campaign

### 1. Build GPU binaries

```bash
mkdir -p build_gpu_campaign && cd build_gpu_campaign
cmake .. -DCMAKE_CXX_COMPILER=nvc++ -DUSE_GPU_OFFLOAD=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) cylinder airfoil step hills
```

### 2. (Re)generate configs (optional)

```bash
python3 scripts/rans_campaign/generate_configs.py
```

### 3. Submit to SLURM

```bash
# Automated (validates prerequisites, submits full array):
scripts/rans_campaign/submit.sh

# Manual (submit in batches of 20):
export RANS_PROJECT_DIR=$(pwd)
sbatch --array=0-19  scripts/rans_campaign/submit_campaign.sbatch
sbatch --array=20-37 scripts/rans_campaign/submit_campaign.sbatch
```

Each job requests 1 H100 GPU, 2 hours wall time, `embers` QOS.

### 4. Analyze results

```bash
python3 scripts/rans_campaign/analyze_campaign.py
```

## Performance Profile (H100, 50k steps)

After IBM optimization (lazy force accumulation, removal of unnecessary GPU sync):

| Component | Cylinder | Airfoil | Step | Hills |
|-----------|----------|---------|------|-------|
| **IBM total** | 3.4-3.6s | 5.2-5.3s | 3.5-4.0s | 3.7-3.8s |
| **Poisson solve** | 6.8-7.9s | 8.4-8.5s | 34-37s | 8.2-8.6s |
| **Turb (algebraic)** | 1.2-2.8s | 1.3-2.8s | 1.1-2.5s | 1.3-2.8s |
| **Turb (transport)** | 0.5-1.2s | 0.5-1.4s | 0.5-1.1s | 0.6-1.3s |
| **Turb (NN-MLP)** | 12.5s | 42.2s | 12.9s | 12.9s |
| **Turb (NN-TBNN)** | 95.8s | 400.6s | 120.2s | 95.2s |

IBM overhead is negligible (~0.07ms/step) for all non-NN models. Poisson dominates for algebraic/transport models; NN inference dominates for neural network models.

### IBM Optimization Details

The IBM direct-forcing method uses pre-computed weight arrays (0=solid, w=forcing, 1=fluid) mapped to GPU at init. Per-step cost is just element-wise weight multiply (3 kernels for u/v/w).

Two optimizations made force computation negligible:
1. **Lazy force accumulation** &mdash; GPU reduction kernels for drag/lift only run at output intervals (`set_accumulate_forces(true)`), not every step. Eliminates 6 GPU&rarr;CPU syncs per step.
2. **Removed unnecessary `sync_solution_from_gpu()`** &mdash; `compute_forces()` returns cached CPU scalars; the full velocity field transfer was wasted work.

## File Layout

```
examples/13_rans_campaign/
    *.cfg                          38 config files (auto-generated)
    README.md                      This file

scripts/rans_campaign/
    generate_configs.py            Config generator
    submit.sh                      Pre-flight checks + submit
    submit_campaign.sbatch         SLURM array job script
    job_list.txt                   Array index -> executable + config mapping
    analyze_campaign.py            Post-run analysis
```
