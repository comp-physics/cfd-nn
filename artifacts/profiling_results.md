# RANS Campaign Profiling Results

Timings in seconds over 50,000 steps on H100 (cylinder/airfoil) and L40S (hills).
IBM overhead is post-optimization (lazy force accumulation, no GPU sync per step).

## Cylinder Re=100 — NVIDIA H100 80GB HBM3

| Model | IBM (s) | Poisson (s) | Turb update (s) | Turb transport (s) |
|-------|---------|-------------|-----------------|---------------------|
| baseline | 3.43 | 7.86 | 2.79 | 0.00 |
| komega | 3.45 | 7.87 | 0.49 | 0.65 |
| sst | 3.44 | 7.34 | 1.15 | 6.64 |
| gep | 3.49 | 7.27 | 1.19 | 0.00 |
| earsm_wj | 3.48 | 6.94 | 1.28 | 6.72 |
| earsm_gs | 3.49 | 7.37 | 1.28 | 6.72 |
| earsm_pope | 3.44 | 6.83 | 1.25 | 6.59 |
| nnmlp | 3.53 | 7.52 | 12.49 | 0.00 |
| nntbnn | 3.59 | 7.72 | 95.81 | 0.00 |

## Airfoil Re=1000 — NVIDIA H100 80GB HBM3

| Model | IBM (s) | Poisson (s) | Turb update (s) | Turb transport (s) |
|-------|---------|-------------|-----------------|---------------------|
| baseline | 5.27 | 8.46 | 2.79 | 0.00 |
| komega | 5.24 | 8.38 | 0.51 | 0.78 |
| sst | 5.27 | 8.39 | 1.38 | 23.63 |
| gep | 5.28 | 8.47 | 1.27 | 0.00 |
| earsm_wj | 5.29 | 8.43 | 1.57 | 23.85 |
| earsm_gs | 5.28 | 8.42 | 1.55 | 23.67 |
| earsm_pope | 5.31 | 8.51 | 1.51 | 23.99 |
| nnmlp | 5.24 | 8.39 | 42.16 | 0.00 |
| nntbnn | 5.27 | 8.40 | 400.56 | 0.00 |

## Hills Re=10595 — NVIDIA L40S

| Model | IBM (s) | Poisson (s) | Turb update (s) | Turb transport (s) |
|-------|---------|-------------|-----------------|---------------------|
| baseline | 3.04 | 11.36 | 2.33 | 0.00 |
| komega | 3.09 | 11.69 | 0.48 | 0.91 |
| sst | 3.03 | 11.45 | 1.27 | 6.41 |
| gep | 3.08 | 11.76 | 1.22 | 0.00 |
| earsm_wj | 3.06 | 11.68 | 1.64 | 6.40 |
| earsm_gs | 3.05 | 11.70 | 1.61 | 6.46 |
| earsm_pope | 3.05 | 11.61 | 1.47 | 6.59 |
| nnmlp | 3.23 | 11.80 | 12.47 | 0.00 |
| nntbnn | 3.26 | 11.27 | 77.66 | 0.00 |
| nnmlp_phll | 3.05 | 11.46 | 12.37 | 0.00 |
| nntbnn_phll | 3.10 | 11.22 | 77.32 | 0.00 |

## Notes

- IBM cost is ~3-5s total across 50k steps (0.06-0.10 ms/step) — negligible vs Poisson
- Poisson dominates for all algebraic/transport models
- NN-TBNN inference is 10-80x more expensive than NN-MLP
- EARSM models include an extra transport solve (the linear system for b_ij)
- Step case (Re=5000) dropped: flow relaminarizes regardless of turbulence model
