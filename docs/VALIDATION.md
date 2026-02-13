# Validation and Test Results

## Laminar Channel Flow (Poiseuille)

The solver validates against the analytical Poiseuille solution for pressure-driven channel flow:

**Analytical solution:**
```
u(y) = -(dp/dx)/(2nu) (H^2 - y^2)
```

where H is the channel half-height and dp/dx < 0 is the imposed pressure gradient.

### Validation Results

**Test case:** Channel with dp/dx = -1, half-height H = 1

| nu | Grid | dt | Iterations | L2 Error | Bulk Velocity Error |
|---|------|----|-----------|---------|--------------------|
| 0.1 | 32x64 | 0.005 | 10,000 | 0.13% | 0.01% |
| 0.1 | 16x32 | 0.005 | 20,000 | 0.13% | 0.02% |
| 0.01 | 32x64 | 0.0002 | 100,000+ | ~1% | ~3% |

**Key findings:**
- Excellent agreement at moderate viscosity (nu = 0.1)
- Stable and robust convergence
- Lower viscosity requires much smaller timesteps due to diffusion stability constraint

### Recommended Parameters

#### High viscosity (nu >= 0.1):
```bash
./channel --Nx 32 --Ny 64 --nu 0.1 --dt 0.005 --max_steps 10000 --tol 1e-8
```

#### Moderate viscosity (nu ~ 0.01):
```bash
./channel --Nx 32 --Ny 64 --nu 0.01 --dt 0.0002 --max_steps 100000 --tol 1e-8
```

#### Low viscosity (nu < 0.01):
Consider grid stretching and smaller timestep:
```bash
./channel --Nx 64 --Ny 128 --nu 0.001 --dt 0.0001 --max_steps 200000 --stretch
```

### Timestep Selection

**Stability constraints:**

1. **CFL condition (convection):**
   ```
   dt <= CFL_max * min(dx, dy) / |u_max|
   ```
   Typically CFL_max = 0.5

2. **Diffusion stability:**
   ```
   dt <= 0.5 * min(dx^2, dy^2) / (nu + nu_t)
   ```
   This is usually the limiting factor for laminar and low-Re flows

For nu = 0.1, dy = 0.03125, the diffusion limit gives dt_max ~= 0.00049, so dt = 0.005 is stable.

For nu = 0.01, dy = 0.03125, the diffusion limit gives dt_max ~= 0.000049, requiring dt < 0.00005.

## Turbulent Channel Flow

### Baseline Model (Mixing Length)

**Test case:** Re = 10,000 (based on channel height and mean velocity)

```bash
./channel --Nx 64 --Ny 128 --nu 0.001 --model baseline --max_steps 20000
```

**Results:**
- Stable convergence
- Reasonable eddy viscosity distribution
- Non-zero wall shear stress
- Not directly validated against DNS (mixing length is approximate)

**Performance:**
- Baseline model adds ~20% computational cost vs laminar
- Most time still spent in Poisson solver

### Neural Network Models

**Test with example weights (random, untrained):**

```bash
# MLP model
./channel --model nn_mlp --nn_preset example_scalar_nut --Nx 16 --Ny 32

# TBNN model
./channel --model nn_tbnn --nn_preset example_tbnn --Nx 16 --Ny 32
```

**Results:**
- Infrastructure loads correctly
- NN inference executes without errors
- Feature computation works
- Random weights cause divergence (expected)

**Performance (with untrained weights):**
- MLP: ~50x slower than laminar (0.388 ms/iter vs 0.008 ms/iter)
- TBNN: ~265x slower than laminar (2.119 ms/iter)
- Most time in NN inference (larger networks are slower)

**Note:** Real trained weights from published models are needed for meaningful validation.

### DNS Channel Flow (No Turbulence Model)

DNS resolves all turbulence scales directly without any model. See `docs/DNS_CHANNEL_GUIDE.md` for the full guide.

**Target:** Re_tau = 180 channel (Moser, Kim & Mansour 1999)

**Grid:** 192 x 96 x 192, Lx = 4pi, Ly = 2, Lz = 2pi, stretch_beta = 2.0

**Run history:**

| Run | Filter | CFL | Result | Re_tau | Notes |
|-----|--------|-----|--------|--------|-------|
| v9 | none | CFL_xz=0.30, CFL_max=0.15 | Blew up ~step 1700 | N/A | Turbulent before blow-up |
| v10 | strength=0.02, interval=10 (x/z only) | same | Blew up ~step 2000 | N/A | Survived longer |
| v11 | strength=0.05, interval=1 (x/y/z) | same | Stable, 3600+ steps | ~255 | First stable run |
| v13 | strength=0.03, interval=2 (x/y/z) | same | Stable, 2400+ steps | ~278 | Best balance |

**Key results from v11 (first fully stable DNS):**

| Step | Re_tau | v_max | w/v ratio | State |
|------|--------|-------|-----------|-------|
| 1200 | 222 | 16.7 | 0.054 | TURBULENT, trip OFF |
| 1800 | 250 | 15.2 | 0.609 | TURBULENT |
| 2400 | 307 | 15.3 | 1.454 | TURBULENT (peak Re_tau) |
| 3600 | 255 | 20.6 | 1.21 | TURBULENT (stabilizing) |

**Known gap:** The velocity filter adds effective viscosity, so the achieved Re_tau (~255-278) exceeds the target of 180. Reaching the exact target would require a less dissipative convective scheme (e.g., hybrid skew-symmetric/upwind).

### Recycling Inflow Validation

See `docs/RECYCLING_INFLOW_GUIDE.md` for the full guide.

**PeriodicVsRecyclingTest:** Runs identical channel flow with periodic BCs and with recycling inflow BCs, then compares plane-averaged statistics:

| Metric | Tolerance | Achieved |
|--------|-----------|----------|
| Shear stress difference | < 5% | ~0.3% |
| Streamwise stress difference | < 5% | ~3.6% |

**RecyclingInflowTest:** 12 checks covering symmetry, mass conservation, divergence, fringe blending, ghost cells. All 12 passing on both CPU and GPU.

---

## Unit Tests

### Mesh and Fields
```bash
./test_mesh
```
**Tests:**
- Mesh indexing
- Ghost cell handling
- Field operations
- Wall distance computation

### Poisson Solver
```bash
./test_poisson
```
**Tests:**
- Laplacian discretization accuracy
- Convergence for manufactured solution
- Boundary conditions (Dirichlet, Neumann, periodic)

### Multigrid Manufactured Solution
```bash
./test_mg_manufactured_solution
```
**Tests:**
- Standard channel BCs (periodic x/z, Neumann y)
- Duct BCs (periodic x, Neumann y/z)
- Recycling inflow BCs (Dirichlet x_lo, Neumann x_hi, Neumann y, periodic z)
- CPU/GPU consistency (max difference = 0.0 for identical inputs)

### FFT Unified Test
```bash
./test_fft_unified
```
**Tests:**
- FFT 3D solver (periodic x/z channel)
- FFT 1D solver (periodic x duct)
- Grid convergence (error decreases with resolution)
- GPU/CPU consistency

### Recycling Inflow Tests
```bash
./test_recycling_inflow
./test_periodic_vs_recycling
```
**Tests:**
- RecyclingInflowTest: 12 checks (symmetry, u_tau, mass conservation, fringe, divergence, ghost cells, etc.)
- PeriodicVsRecyclingTest: recycling matches periodic within 5% for shear and streamwise stress

### Neural Network Loading
```bash
./test_nn_simple
```
**Tests:**
- MLP weight loading
- Forward pass correctness
- Feature computation
- TurbulenceNNMLP and TurbulenceNNTBNN initialization

---

## Known Issues and Limitations

### Numerical

1. **Explicit time stepping** limits timestep for low viscosity
   - **Mitigation:** Adaptive time stepping with directional CFL is now implemented. See `docs/DNS_CHANNEL_GUIDE.md`.

2. **Central differences require velocity filter for DNS stability**
   - Second-order central schemes have zero numerical dissipation, causing grid-scale blow-up in DNS. The velocity filter (`filter_strength`, `filter_interval`) provides explicit diffusion but adds effective viscosity. See `docs/DNS_CHANNEL_GUIDE.md`.

3. **Filter-limited Re_tau in DNS**
   - The velocity filter prevents reaching the exact target Re_tau = 180. Best achieved: Re_tau ~ 278 with strength=0.03, interval=2. A higher-order or hybrid convective scheme would reduce filter requirements.

### Model-Related

4. **Example NN models use random weights**
   - **Solution:** Add real trained weights from publications

5. **Feature sets may not match published models exactly**
   - **Solution:** Verify feature definitions when adding real models

6. **No automatic feature set detection**
   - **Solution:** Manually configure features per model for now

### GPU-Specific

7. **CPU-side diagnostics require GPU sync**
   - Functions that read velocity data on the CPU (e.g., `accumulate_statistics()`, `validate_turbulence_realism()`) must call `sync_solution_from_gpu()` first. The solver handles this internally for built-in diagnostics, but custom code must sync manually.

8. **Recycling inflow disables CUDA Graph V-cycle**
   - Because recycling modifies inlet BCs each step, the CUDA Graph is invalid. The solver falls back to the standard V-cycle path (~10-20% slower for Poisson solves).

## Convergence Criteria

The solver uses velocity residual to determine convergence:

```
residual = max|u^(n+1) - u^n|
```

Typical convergence:
- **Laminar:** Exponential decay, reaches tol=1e-8 reliably
- **Baseline turbulence:** Slower convergence, reaches tol=1e-6
- **NN models:** Depends on weights (untrained weights diverge)
- **DNS (unsteady):** Residual does not converge to zero; use `max_steps` to control run length

## Performance Summary

**Timing for 10,000 iterations on 16x32 grid:**

| Configuration | Total Time | Per Iteration | Relative |
|--------------|------------|---------------|----------|
| Laminar | 0.08 s | 0.008 ms | 1x |
| Baseline | 0.46 s | 0.046 ms | 5.8x |
| NN-MLP | 3.88 s | 0.388 ms | 48x |
| NN-TBNN | 21.19 s | 2.119 ms | 265x |

**Breakdown for laminar case:**
- Poisson solve: 40%
- Convective/diffusive terms: 50%
- Boundary conditions: 10%

**For NN models:**
- NN inference: 95%
- Rest of solver: 5%

## References

- **Poiseuille flow:** Classical analytical solution
- **Mixing length:** Pope, "Turbulent Flows" (2000)
- **TBNN:** Ling et al., JFM 807 (2016)
- **DNS channel:** Moser, Kim & Mansour, Physics of Fluids 11.4 (1999)
- **Recycling inflow:** Lund, Wu & Squires, J. Comput. Phys. 140.2 (1998)

---

**Last updated:** February 2026 — Added DNS channel flow results, recycling inflow validation, MG manufactured solution tests, FFT unified tests
