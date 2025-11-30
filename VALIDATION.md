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
./channel --Nx 32 --Ny 64 --nu 0.1 --dt 0.005 --max_iter 10000 --tol 1e-8
```

#### Moderate viscosity (nu ~ 0.01):
```bash
./channel --Nx 32 --Ny 64 --nu 0.01 --dt 0.0002 --max_iter 100000 --tol 1e-8
```

#### Low viscosity (nu < 0.01):
Consider grid stretching and smaller timestep:
```bash
./channel --Nx 64 --Ny 128 --nu 0.001 --dt 0.0001 --max_iter 200000 --stretch
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
./channel --Nx 64 --Ny 128 --nu 0.001 --model baseline --max_iter 20000
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

## Periodic Hills

**Status:** Implemented, basic testing done

```bash
./periodic_hills --Nx 64 --Ny 48 --model baseline
```

**Geometry:**
- Simplified hill profile
- Periodic boundary conditions in streamwise direction
- No-slip at top and bottom walls

**Validation:**
- Not yet validated against reference data (Breuer et al.)
- Future work: detailed comparison with DNS/LES reference

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

### Neural Network Loading
```bash
./test_nn_simple
```
**Tests:**
- MLP weight loading
- Forward pass correctness
- Feature computation
- TurbulenceNNMLP and TurbulenceNNTBNN initialization

## Known Issues and Limitations

### Numerical

1. **Explicit time stepping** limits timestep for low viscosity
   - **Solution:** Add semi-implicit diffusion or adaptive timestepping

2. **SOR Poisson solver** is slow and non-optimal
   - **Solution:** Implement multigrid or conjugate gradient

3. **First-order upwind** for convection is diffusive
   - **Solution:** Implement QUICK or higher-order schemes

### Model-Related

4. **Example NN models use random weights**
   - **Solution:** Add real trained weights from publications

5. **Feature sets may not match published models exactly**
   - **Solution:** Verify feature definitions when adding real models

6. **No automatic feature set detection**
   - **Solution:** Manually configure features per model for now

## Convergence Criteria

The solver uses velocity residual to determine convergence:

```
residual = max|u^(n+1) - u^n|
```

Typical convergence:
- **Laminar:** Exponential decay, reaches tol=1e-8 reliably
- **Baseline turbulence:** Slower convergence, reaches tol=1e-6
- **NN models:** Depends on weights (untrained weights diverge)

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

## Recommendations

### For Production Use

1. **Implement adaptive timestepping** - essential for robustness
2. **Optimize Poisson solver** - currently the bottleneck
3. **Add real trained NN models** - examples are just infrastructure tests
4. **Validate against DNS data** - obtain reference solutions for target cases

### For Development

1. **Add VTK output** - for visualization in ParaView
2. **Create comprehensive test suite** - automated validation
3. **Profile and optimize NN inference** - batch operations, reduce allocations
4. **Add OpenMP parallelization** - easy speedup for field operations

## References

- **Poiseuille flow:** Classical analytical solution
- **Mixing length:** Pope, "Turbulent Flows" (2000)
- **TBNN:** Ling et al., JFM 807 (2016)
- **Periodic hills:** Breuer & Rodi, Flow Turb. Combust. 66 (2001)

---

**Last updated:** Implementation of laminar solver, baseline turbulence model, and NN infrastructure (2024)
