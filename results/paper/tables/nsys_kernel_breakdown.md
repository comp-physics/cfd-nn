# nsys Kernel Profile: V100, 256^3, 20 steps

GPU: Tesla V100-PCIE-16GB
Grid: 256^3 = 16.8M cells
Config: central FD, RK3, FFT Poisson, no turbulence model

## Summary

| Category | ms/step | % of kernels |
|----------|---------|-------------|
| **Poisson (FFT+tridiag+pack/unpack)** | **96.7** | **73.3%** |
| Convection + Diffusion | 11.9 | 9.0% |
| RK3 blending + substep | 13.1 | 9.9% |
| Projection + correction | 7.9 | 6.0% |
| Boundary conditions | 2.1 | 1.6% |
| Other | 0.3 | 0.2% |
| **Total kernel** | **131.9** | **100%** |

Wall-clock: ~141 ms/step. Kernel utilization: 94%.

## Poisson Breakdown

| Kernel | ms/step | % of Poisson |
|--------|---------|-------------|
| cuFFT c2r (inverse) | 23.9 | 24.7% |
| cuFFT regular (forward, 2 calls) | 21.4 | 22.1% |
| cuFFT r2c (forward) | 13.9 | 14.4% |
| **kernel_unpack_and_bc** | **21.5** | **22.2%** |
| kernel_pack_and_partial_sum | 5.7 | 5.9% |
| cuSPARSE tridiag loop | 4.8 | 4.9% |
| cuSPARSE tridiag first pass | 4.6 | 4.7% |
| kernel_subtract_mean | 1.0 | 1.0% |

## Key Finding

Our Poisson solver takes **73%** of kernel time vs CaLES's **38%**.
The gap is from the data pack/unpack overhead (27 ms/step combined)
between the FFT transforms and the tridiagonal solve. CaLES's
eigenfunction expansion approach likely fuses these operations.

CaLES comparison (A100, 283M cells): Poisson = 38%, RHS = 30%
Our solver (V100, 16.8M cells): Poisson = 73%, RHS = 9%

The RHS (convection+diffusion) fraction is much lower for us because
the Poisson dominates more, not because our RHS is faster.
