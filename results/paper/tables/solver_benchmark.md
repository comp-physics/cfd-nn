# Solver Throughput Benchmark

## Our Solver — H100 80GB HBM3 (Single GPU)

Channel flow, RK3, skew scheme, FFT Poisson solver.

| Grid | Cells | ms/step | Mcells/s |
|------|-------|---------|----------|
| 256³ | 16.8M | 27.2 | 616 |

## With Turbulence Models — 256³, RK3, H100

| Model | ms/step | Overhead | Turb (ms) |
|-------|---------|----------|-----------|
| Baseline (laminar) | 27.25 | — | — |
| SST k-omega | 27.61 | +1.3% | 0.061 |

## CaNS Reference (Costa et al. 2021)

Re_tau 590, 512x256x144 (18.9M cells), V100 GPUs, DGX-2.

| Config | ms/step | Mcells/s |
|--------|---------|----------|
| 4x V100 | 481 | 39 |
| 16x V100 | 140 | 135 |

## Comparison Notes

Direct comparison approximate (H100 vs V100, 1 vs 4-16 GPUs).
Accounting for ~3.7x H100/V100 bandwidth ratio, our single-GPU throughput
(616 Mcells/s) is roughly equivalent to CaNS on ~4-5 V100s.
Both: staggered FD, FFT Poisson, RK3, incompressible NS.
