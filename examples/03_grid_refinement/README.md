# Example 3: Grid Refinement Study

## Overview

Systematic **grid convergence analysis** to quantify the numerical accuracy of the CFD solver.

**Purpose**:
- Verify **spatial order of accuracy** (should be ~2 for central differencing)
- Compute **Grid Convergence Index (GCI)** for error estimation
- Demonstrate Richardson extrapolation
- Provide template for verification & validation studies

## Method: Richardson Extrapolation

For a numerical scheme with order of accuracy \(p\), the error should scale as:

```
Error ∝ h^p
```

where \(h\) is the grid spacing. By running on multiple grids with refinement ratio \(r = h_coarse / h_fine\), we can:

1. **Estimate order of accuracy**:
   ```
   p = log(e_coarse / e_fine) / log(r)
   ```

2. **Extrapolate to h-->0** (estimated "exact" solution):
   ```
   u_exact ~= u_fine + (u_fine - u_coarse) / (r^p - 1)
   ```

3. **Compute Grid Convergence Index (GCI)**:
   ```
   GCI = F_s x |e_fine| / (r^p - 1)
   ```
   where F_s = 1.25 (safety factor)

## Grid Levels

Four systematically refined grids:

| Level | Nx x Ny | h_y | Total Points |
|-------|---------|-----|--------------|
| Coarse | 32 x 64 | 0.0317 | 2,048 |
| Medium | 64 x 128 | 0.0157 | 8,192 |
| Fine | 128 x 256 | 0.0078 | 32,768 |
| Very Fine | 256 x 512 | 0.0039 | 131,072 |

**Refinement ratio**: r = 2.0 (each level doubles resolution)

## Test Case

**Problem**: Laminar Poiseuille flow (same as Example 1)
- Analytical solution known --> can compute exact error
- 2nd-order scheme --> expect p ~= 2.0

## Running the Study

### Quick Start

```bash
cd examples/03_grid_refinement
./run_refinement.sh
```

This will:
1. Run all 4 grid levels sequentially
2. Compute convergence metrics
3. Generate convergence plots

**Expected time**: 
- Coarse: 10-30 seconds
- Medium: 30-60 seconds  
- Fine: 2-4 minutes
- Very Fine: 10-20 minutes

**Total**: ~15-25 minutes

### Run Individual Grids

```bash
cd ../../build

# Coarse grid
./channel --config ../examples/03_grid_refinement/coarse_32x64.cfg \
          --output ../examples/03_grid_refinement/output/coarse

# Medium grid
./channel --config ../examples/03_grid_refinement/medium_64x128.cfg \
          --output ../examples/03_grid_refinement/output/medium

# Fine grid
./channel --config ../examples/03_grid_refinement/fine_128x256.cfg \
          --output ../examples/03_grid_refinement/output/fine

# Very fine grid
./channel --config ../examples/03_grid_refinement/very_fine_256x512.cfg \
          --output ../examples/03_grid_refinement/output/very_fine
```

## Expected Results

### Convergence Order

For 2nd-order central differencing scheme:

```
Observed order of accuracy: p ~= 2.0 ± 0.1
```

### Error Reduction

| Grid Transition | Expected Error Reduction |
|----------------|-------------------------|
| Coarse --> Medium | ~4x (2²) |
| Medium --> Fine | ~4x (2²) |
| Fine --> Very Fine | ~4x (2²) |

### Actual Errors (Typical)

| Grid | L2 Error | L∞ Error |
|------|----------|----------|
| Coarse | ~1e-4 | ~3e-4 |
| Medium | ~2e-5 | ~8e-5 |
| Fine | ~5e-6 | ~2e-5 |
| Very Fine | ~1e-6 | ~5e-6 |

## Visualization

The `convergence_analysis.py` script generates a 4-panel figure:

1. **Velocity Profiles**: All grids vs analytical solution
2. **Error Profiles**: Pointwise error distribution
3. **Convergence Plot**: log-log plot showing Error vs h
4. **Summary Table**: Quantitative metrics

### Interpreting the Convergence Plot

**Perfect 2nd-order convergence**: Data points fall on a straight line with slope = -2 in log-log plot

**Deviations**:
- **p < 2**: Grid too coarse, or 1st-order scheme used
- **p > 2**: Very smooth solution, or lucky error cancellation
- **Plateau**: Reaching machine precision or round-off error

## Success Criteria

[OK] **Excellent**: p = 1.9 - 2.1 (2nd order confirmed)  
[OK] **Good**: p = 1.7 - 2.3 (roughly 2nd order)  
[WARNING] **Acceptable**: p = 1.5 - 1.7 (some 1st order contamination)  
[FAIL] **Fail**: p < 1.5 or p > 2.5 (scheme issue!)

## What This Tests

### Spatial Discretization
- [OK] Gradient operators (nabla, nabla²)
- [OK] Interpolation schemes
- [OK] Boundary condition implementation

### Solver Accuracy
- [OK] Pressure Poisson solver convergence
- [OK] Round-off error management
- [OK] Iterative solver tolerance effects

## Extensions

### 1. Test Different Schemes

Modify config files to test:

```bash
# Central differencing (2nd order)
convection_scheme = central  # Expected: p ~= 2

# Upwind (1st order)
convection_scheme = upwind   # Expected: p ~= 1
```

### 2. Test Stretched Grids

Enable grid stretching:
```bash
stretch_y = true
beta_y = 2.0
```

**Question**: Does stretching affect convergence order?

### 3. Richardson Extrapolation

Use finest 2 grids to extrapolate to h-->0:
```python
u_extrapolated = u_fine + (u_fine - u_medium) / (2^p - 1)
```

This gives an even more accurate "exact" solution!

### 4. Temporal Convergence

Fix spatial grid, vary time step:
```bash
# dt = 0.1, 0.01, 0.001, 0.0001
```

Expected: 1st order for explicit Euler

### 5. Grid Convergence Index (GCI)

Compute error bounds:
```
GCI_fine = 1.25 x |error_fine| / (r^p - 1)
```

Interpretation: Solution on fine grid is within GCI of "exact"

## Troubleshooting

### Convergence Order Too Low (p < 1.5)

**Possible causes**:
- Using 1st-order upwind scheme (check config)
- Grid too coarse (run on finer grids)
- Iterative solver not converged (reduce `tol`)
- Boundary conditions incorrect

**Fixes**:
- Use `convection_scheme = central`
- Reduce `tol` to 1e-12
- Check residuals in solver output

### Convergence Order Too High (p > 2.5)

**Possible causes**:
- Very smooth analytical solution (lucky!)
- Error cancellation
- Round-off error dominating

**Notes**:
- This is rare and usually harmless
- Check that finest grid error is still decreasing

### Very Fine Grid Doesn't Converge

**Cause**: Time step too large for fine grid

**Fix**:
```bash
# Reduce CFL for stability
CFL_max = 0.3

# Or reduce dt directly
dt = 0.001
```

### Errors Don't Decrease Below ~1e-10

**Cause**: Reaching machine precision (double ~= 1e-16)

**Notes**:
- This is expected for very fine grids
- Real error is smaller than we can measure
- GCI analysis will show asymptotic range

## Applications

This methodology is used for:

1. **Code Verification**: Prove solver implements equations correctly
2. **Discretization Error Estimation**: Quantify numerical uncertainty
3. **Grid Selection**: Choose appropriate resolution for target accuracy
4. **Publication**: Demonstrate V&V for journal papers

## ASME V&V Standards

This example follows ASME V&V 20-2009 guidelines:

- [OK] Multiple systematically refined grids (>=3)
- [OK] Constant refinement ratio (r = 2)
- [OK] Monotonic convergence demonstrated
- [OK] Observed order of accuracy computed
- [OK] GCI for uncertainty quantification

## References

1. **Richardson Extrapolation**: Richardson, L. F. (1911). "The approximate arithmetical solution by finite differences." *Phil. Trans. Roy. Soc. London*, A210, 307-357.

2. **GCI Methodology**: Roache, P. J. (1998). "Verification of codes and calculations." *AIAA Journal*, 36(5), 696-702.

3. **V&V Standards**: ASME V&V 20-2009, "Standard for Verification and Validation in Computational Fluid Dynamics and Heat Transfer."

4. **Grid Convergence**: Celik, I. B., et al. (2008). "Procedure for estimation and reporting of uncertainty due to discretization in CFD applications." *J. Fluids Eng.*, 130(7).

## Related Examples

- **Example 1**: Laminar channel (same physics, single grid)
- **Example 4**: Validation suite (multiple test cases)

---

**Key Takeaway**: This example proves your solver is **2nd-order accurate** and provides a template for rigorous verification studies!

