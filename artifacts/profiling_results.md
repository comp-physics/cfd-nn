# RANS Campaign Profiling Results

## Current Results (post-optimization)

Timings in seconds over 50,000 steps on NVIDIA L40S.
SST wall BCs ported to GPU (no CPU roundtrip). NN inference uses batched (cell, neuron) parallelization.

### Cylinder Re=100 — NVIDIA L40S

| Model | IBM (s) | Poisson (s) | Turb update (s) | Turb transport (s) | Total step (s) |
|-------|---------|-------------|-----------------|---------------------|----------------|
| baseline | — | 11.4 | 2.33 | — | 45.0 |
| komega | 2.65 | 11.8 | 0.47 | 0.85 | 45.1 |
| sst | 2.66 | 11.4 | 1.26 | 2.22 | 46.9 |
| gep | 2.66 | 11.3 | 1.18 | — | 46.3 |
| earsm_wj | 2.67 | 11.3 | 1.63 | 2.22 | 47.4 |
| earsm_gs | 2.67 | 11.3 | 1.58 | 2.22 | 47.3 |
| earsm_pope | 2.67 | 11.3 | 1.46 | 2.22 | 47.1 |
| nnmlp | 2.65 | 11.3 | 12.78 | — | 55.6 |
| nntbnn | 2.67 | 11.4 | 85.55 | — | 129.0 |

### Airfoil Re=1000 — NVIDIA L40S

| Model | IBM (s) | Poisson (s) | Turb update (s) | Turb transport (s) | Total step (s) |
|-------|---------|-------------|-----------------|---------------------|----------------|
| baseline | — | 26.6 | 3.01 | — | 83.0 |
| komega | 6.99 | 26.6 | 0.58 | 2.03 | 86.0 |
| sst | 7.08 | 26.6 | 2.14 | 4.63 | 90.0 |
| gep | 7.08 | 26.6 | 1.89 | — | 88.5 |
| earsm_wj | 7.15 | 26.6 | 3.65 | 4.59 | 91.7 |
| earsm_gs | 7.15 | 26.6 | 3.30 | 4.65 | 91.4 |
| earsm_pope | 7.12 | 26.6 | 2.80 | 4.63 | 90.8 |
| nnmlp | 7.18 | 26.6 | 40.77 | — | 124.3 |
| nntbnn | 7.20 | 26.7 | 543.91 | — | 628.4 |

### Hills Re=10595 — NVIDIA L40S / H200

| Model | GPU | IBM (s) | Poisson (s) | Turb update (s) | Turb transport (s) |
|-------|-----|---------|-------------|-----------------|---------------------|
| baseline | L40S | — | 11.4 | 2.37 | — |
| komega | H200 | — | — | 0.49 | 0.64 |
| sst | L40S | — | 11.5 | 1.26 | 2.27 |
| gep | H200 | — | — | 1.15 | — |
| earsm_wj | H200 | — | — | 1.23 | 1.60 |
| earsm_gs | H200 | — | — | 1.26 | 1.60 |
| earsm_pope | L40S | — | 11.6 | 1.46 | 2.27 |
| nnmlp | H200 | — | — | 12.48 | — |
| nntbnn | H200 | — | — | 87.55 | — |

## SST/EARSM Speedup Summary

The SST wall BC GPU port eliminated a CPU roundtrip that was the dominant cost:

| Case | Before (H100) | After (L40S) | SST/komega ratio |
|------|--------------|-------------|-----------------|
| Cylinder SST transport | 6.64s | **2.22s** | 10.2x → **2.6x** |
| Airfoil SST transport | 23.63s | **4.63s** | 30.3x → **2.3x** |

The remaining 2.3-2.6x ratio is real physics cost (F1/F2 blending, cross-diffusion).
Turbulence is now ~8% of total step time. Poisson dominates at 24-30%.

## Previous Results (pre-optimization)

Timings from H100 (cylinder/airfoil) and L40S (hills) before SST wall BC GPU port.
SST transport was 10-30x slower than komega due to mandatory GPU→CPU→GPU roundtrip for wall BCs.

### Cylinder Re=100 — NVIDIA H100 80GB HBM3

| Model | IBM (s) | Poisson (s) | Turb update (s) | Turb transport (s) |
|-------|---------|-------------|-----------------|---------------------|
| baseline | 3.43 | 7.86 | 2.79 | 0.00 |
| komega | 3.45 | 7.87 | 0.49 | 0.65 |
| sst | 3.44 | 7.34 | 1.15 | 6.64 |
| earsm_wj | 3.48 | 6.94 | 1.28 | 6.72 |
| nnmlp | 3.53 | 7.52 | 12.49 | 0.00 |
| nntbnn | 3.59 | 7.72 | 95.81 | 0.00 |

### Airfoil Re=1000 — NVIDIA H100 80GB HBM3

| Model | IBM (s) | Poisson (s) | Turb update (s) | Turb transport (s) |
|-------|---------|-------------|-----------------|---------------------|
| baseline | 5.27 | 8.46 | 2.79 | 0.00 |
| komega | 5.24 | 8.38 | 0.51 | 0.78 |
| sst | 5.27 | 8.39 | 1.38 | 23.63 |
| earsm_wj | 5.29 | 8.43 | 1.57 | 23.85 |
| nnmlp | 5.24 | 8.39 | 42.16 | 0.00 |
| nntbnn | 5.27 | 8.40 | 400.56 | 0.00 |

## Notes

- IBM cost is ~3-7s total across 50k steps (0.06-0.14 ms/step) — negligible vs Poisson
- Poisson dominates for all algebraic/transport models (24-30% of step time)
- NN-TBNN inference is 7-13x more expensive than NN-MLP (wider layers: 64 vs 32)
- EARSM update is ~30% more than SST (anisotropy tensor computation)
- Step case (Re=5000) dropped: flow relaminarizes regardless of turbulence model
- Hills results are mixed GPU (L40S for some, H200 for others) due to CI scheduling
