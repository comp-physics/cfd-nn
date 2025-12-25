# Example 1: Laminar Channel Flow (Poiseuille)

## Overview

This example validates the CFD solver against the **analytical Poiseuille solution** for pressure-driven laminar flow between parallel plates.

**Purpose**: Verify that the solver correctly implements:
- Pressure-velocity coupling
- Viscous stress terms
- Boundary conditions (periodic in x, no-slip walls in y)
- Steady-state convergence

## Physical Problem

**Geometry**: 2D channel with parallel walls
- Domain: 4.0 x 2.0 (Lx x Ly)
- Walls at y = ±1.0

**Flow Conditions**:
- Reynolds number: Re = 100
- Viscosity: ν = 0.01
- Pressure gradient: dp/dx = -0.001 (drives the flow)
- Turbulence model: None (laminar flow)

**Expected Result**: Parabolic velocity profile matching:

```
u(y) = -(dp/dx)/(2ν) x (H²/4 - y²)
```

where H = 2.0 is the channel height and y is measured from the centerline.

## Analytical Solution

For plane Poiseuille flow, the exact solution is:

**Velocity Profile**:
```
u_max = (dp/dx) x H² / (8ν)
u(y) = u_max x (1 - (2y/H)²)
```

**With our parameters**:
- u_max = 0.001 x 4 / (8 x 0.01) = **0.05**
- Parabolic profile symmetric about centerline

## Running the Example

### Quick Start

```bash
cd examples/01_laminar_channel
./run.sh
```

This will:
1. Run the simulation (should converge in ~1000-2000 iterations)
2. Save VTK output files
3. Automatically analyze results and create comparison plots

### Manual Execution

```bash
# From build directory
cd ../../build

# Run simulation
./channel --config ../examples/01_laminar_channel/poiseuille.cfg \
          --output ../examples/01_laminar_channel/output

# Analyze results
cd ../examples/01_laminar_channel
python3 analyze.py
```

## Expected Results

### Convergence
- **Iterations**: 1000-3000 (depends on initial conditions)
- **Residual**: < 1e-10
- **Time**: ~10-30 seconds on a typical laptop

### Accuracy
- **Velocity error**: < 1% relative to analytical solution
- **Profile**: Smooth parabola with no oscillations
- **Symmetry**: Centerline at u_max ~= 0.05

### Output Files

```
output/
├── velocity_0000.vtk      # Initial condition
├── velocity_0001.vtk      # Snapshot 1
├── ...
├── velocity_final.vtk     # Converged solution
└── poiseuille_validation.png  # Analysis plot
```

## Success Criteria

[OK] **PASS**: Relative error < 1%  
[WARNING] **WARNING**: Error between 1-5% (check grid resolution)  
[FAIL] **FAIL**: Error > 5% (solver issue!)

## Visualization

### ParaView
```bash
paraview output/velocity_final.vtk
```

**Recommended views**:
1. Slice through centerplane
2. Color by velocity magnitude
3. Add glyphs to show velocity vectors

### Python Analysis

The `analyze.py` script generates:
1. **Velocity profile comparison**: Numerical vs analytical
2. **Error distribution**: Shows accuracy across the channel
3. **Quantitative metrics**: Max, RMS, and relative errors

## What This Tests

### Physics
- [OK] Incompressibility (nabla*u = 0)
- [OK] Momentum balance (dp/dx = νnabla²u)
- [OK] No-slip boundary conditions
- [OK] Steady-state convergence

### Numerics
- [OK] Pressure Poisson solver
- [OK] Viscous term discretization
- [OK] Spatial accuracy (should be 2nd order with central differencing)

## Troubleshooting

### Simulation doesn't converge
- **Check**: Is `tol` too strict? Try `tol = 1e-8`
- **Check**: Increase `max_iter` to 20000
- **Check**: Reduce `dt` to 0.005

### Error is too large (> 5%)
- **Cause**: Grid too coarse
- **Fix**: Increase `Ny` to 128 in config file
- **Expected**: Error should decrease as Ny² (2nd order convergence)

### Simulation is too slow
- **Cause**: Time step too small or grid too fine
- **Fix**: Use `adaptive_dt = true` and increase `CFL_max` to 1.0
- **Note**: Laminar flow is stable at high CFL

## Extensions

### 1. Grid Refinement Study
Modify `Ny` in config file and measure error:
```bash
# Edit poiseuille.cfg: Ny = 32, 64, 128, 256
# Run for each and plot error vs grid spacing
```

Expected: 2nd order convergence (error ∝ Δy²)

### 2. Reynolds Number Sweep
Change `Re` and `nu`:
```bash
# Re = 10, 100, 1000
# Observe: Higher Re --> flatter center, steeper gradients at walls
```

### 3. Different Pressure Gradients
Modify `dp_dx`:
```bash
# dp_dx = -0.0001, -0.001, -0.01
# Result: u_max scales linearly with |dp/dx|
```

## References

1. **Poiseuille Flow**: White, F. M. "Viscous Fluid Flow" (3rd ed., 2006), Chapter 3
2. **Channel Flow DNS**: Kim, J., Moin, P., & Moser, R. "Turbulence statistics in fully developed channel flow at low Reynolds number" (1987)
3. **Numerical Methods**: Ferziger, J. H. & Perić, M. "Computational Methods for Fluid Dynamics" (2002)

## Related Examples

- **Example 2**: Turbulent channel (same geometry, high Re)
- **Example 3**: Grid refinement study (quantifies numerical accuracy)
- **Example 4**: Validation suite (includes multiple analytical cases)

---

**Next Steps**: Once this passes, move to turbulent flow cases to test turbulence models!

