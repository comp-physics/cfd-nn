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

## V100 Benchmark (Direct CaNS Comparison)

Tesla V100-PCIE-16GB, single GPU. Same discretization as CaNS:
2nd-order central FD, RK3, explicit diffusion, FFT Poisson.

| Grid | Cells | ms/step | Mcells/s |
|------|-------|---------|----------|
| 256×128×72 | 2.4M | 13.6 | 174 |
| 256³ | 16.8M | 140.6 | 119 |
| 512×256×144 | 18.9M | 146.4 | 129 |

### Head-to-Head: Same Grid (512×256×144 = 18.9M cells)

| Solver | GPUs | ms/step | Mcells/s | Mcells/s/GPU |
|--------|------|---------|----------|-------------|
| Our solver | 1× V100-16GB | 146.4 | 129 | 129 |
| CaNS | 4× V100-32GB | 481 | 39 | 9.8 |
| CaNS | 16× V100-32GB | 140 | 135 | 8.4 |

Our single V100 achieves 129 Mcells/s — 3.3× faster than CaNS on 4 V100s.
Per-GPU efficiency: 13× higher (129 vs 9.8 Mcells/s/GPU).
Note: CaNS uses 32GB V100s on DGX-2 (NVLink); ours is 16GB PCIe.

## H100 MPI Scaling (512×256×144 = 18.9M cells)

| GPUs | ms/step | Mcells/s total | Mcells/s/GPU |
|------|---------|----------------|-------------|
| 1× H100 | 30.1 | 626 | 626 |
| 2× H100 | 103.1 | 183 | 91.5 |

2 GPUs is 3.4× slower than 1 GPU — distributed FFT transpose
overhead dominates at this grid size. Single-GPU is optimal for
grids that fit in GPU memory.
