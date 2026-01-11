# Example 4: Validation Suite - DNS & Analytical Benchmarks

## Overview

Comprehensive **validation test suite** comparing solver predictions against:
- **Analytical solutions** (exact, for verification)
- **DNS benchmarks** (Moser et al. 1999, for turbulence model validation)

**Purpose**: Demonstrate that the solver correctly implements the governing equations and produces physically accurate results across a range of flow conditions.

## Validation Philosophy

**Verification**: Solving the equations correctly (math --> code)  
**Validation**: Solving the correct equations (physics --> math)

This suite provides both:
- [OK] Analytical cases verify numerical implementation
- [OK] DNS cases validate turbulence models

## Test Cases

### 1. Poiseuille Flow - Re = 100 (**Verification**)

**Reference**: Analytical solution

**Physics**: Laminar pressure-driven channel flow

**Expected Error**: < 1%

**Tests**:
- Momentum equation discretization
- Pressure-velocity coupling
- Boundary conditions (periodic + no-slip)

---

### 2. Poiseuille Flow - Re = 1000 (**Verification**)

**Reference**: Analytical solution

**Physics**: Higher Reynolds number, still laminar

**Expected Error**: < 1%

**Tests**:
- Solver stability at higher Re
- Convection-diffusion balance
- Iterative convergence

---

### 3. Turbulent Channel - Re_τ = 180 (**Validation**)

**Reference**: DNS by Moser, Kim & Mansour (1999)

**Physics**: Fully developed turbulent channel flow

**Expected Error**: 10-20% (RANS baseline model)

**Tests**:
- Turbulence model performance
- Law of the wall
- Reynolds stress prediction

**DNS Data**: Available at https://www.flow.kth.se/~pschlatt/DATA/

---

### 4. Turbulent Channel - Re_τ = 395 (**Validation**)

**Reference**: DNS by Moser, Kim & Mansour (1999)

**Physics**: Higher Reynolds number turbulent flow

**Expected Error**: 15-25% (RANS baseline model)

**Tests**:
- Model generalization to higher Re
- Log-layer predictions
- Outer layer accuracy

---

## Running the Suite

### Quick Start

```bash
cd examples/04_validation_suite
./run_validation.sh
```

**Time**: ~15-30 minutes for all 4 cases

**Output**:
- VTK files for each case
- Validation report with pass/fail for each case
- Comparison plots

### Run Individual Cases

```bash
cd ../../build

# Laminar cases (fast)
./channel --config ../examples/04_validation_suite/poiseuille_re100.cfg \
          --output ../examples/04_validation_suite/output/poiseuille_re100

./channel --config ../examples/04_validation_suite/poiseuille_re1000.cfg \
          --output ../examples/04_validation_suite/output/poiseuille_re1000

# Turbulent cases (slower)
./channel --config ../examples/04_validation_suite/channel_re180.cfg \
          --output ../examples/04_validation_suite/output/channel_re180

./channel --config ../examples/04_validation_suite/channel_re395.cfg \
          --output ../examples/04_validation_suite/output/channel_re395
```

## Success Criteria

### Poiseuille Cases (Analytical)

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| Relative Error < 1% | [OK] PASS | Excellent numerical accuracy |
| Error 1-5% | [WARNING] ACCEPTABLE | May need finer grid |
| Error > 5% | [FAIL] FAIL | Solver implementation issue |

### Turbulent Cases (DNS)

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| RMS Error < 15% | [OK] PASS | Good RANS model performance |
| Error 15-25% | [WARNING] ACCEPTABLE | Typical RANS accuracy |
| Error > 25% | [FAIL] FAIL | Model calibration needed |

## Expected Results

### Analytical Cases

Both Poiseuille cases should achieve < 1% error, proving:
- [OK] Navier-Stokes equations implemented correctly
- [OK] Boundary conditions applied properly
- [OK] Iterative solver converges accurately
- [OK] Spatial discretization is 2nd-order accurate

### DNS Cases

Baseline mixing length model typically achieves:
- **Re_τ = 180**: 12-18% RMS error
- **Re_τ = 395**: 18-25% RMS error

This demonstrates:
- [OK] Turbulence model provides reasonable eddy viscosity
- [OK] Mean velocity profile matches DNS trends
- [OK] Log-layer slope approximately correct
- [WARNING] Some deviation in outer layer (expected for RANS)

## Visualization

### Automated Report

```bash
python validate_results.py
```

Generates multi-panel comparison:
- Poiseuille cases: Numerical vs analytical profiles
- Turbulent cases: u+ vs y+ compared to DNS

Each panel shows:
- [PASS] or [FAIL] status
- Quantitative error percentage
- Visual comparison

### ParaView

```bash
paraview output/*/velocity_final.vtk
```

## Interpreting Results

### Laminar Cases (Should PASS)

If Poiseuille cases fail:
1. **Check grid resolution**: Increase Ny to 256
2. **Check convergence**: Reduce `tol` to 1e-12
3. **Check boundary conditions**: Verify periodic/no-slip
4. **Potential bug**: If error > 10%, likely code issue

### Turbulent Cases (More Tolerance)

Baseline model limitations:
- [OK] Good: Mean velocity profile
- [WARNING] Fair: Reynolds stress magnitude
- [FAIL] Poor: Reynolds stress anisotropy (needs TBNN)

To improve turbulent case accuracy:
1. **Use GEP model**: Change `model = gep` in configs
2. **Train NN models**: Use McConkey dataset
3. **Finer grid**: Increase Ny to 256 for better wall resolution

## Output Files

```
output/
├── poiseuille_re100/
│   └── velocity_final.vtk
├── poiseuille_re1000/
│   └── velocity_final.vtk
├── channel_re180/
│   ├── velocity_0000.vtk
│   ├── ...
│   └── velocity_final.vtk
├── channel_re395/
│   ├── velocity_0000.vtk
│   ├── ...
│   └── velocity_final.vtk
└── validation_report.png
```

## Adding New Validation Cases

To add your own benchmark:

1. **Create config file**: `my_case.cfg`
2. **Add to run script**: Append to `cases` array
3. **Add reference data**: Update `validate_results.py` with benchmark solution
4. **Set tolerance**: Define acceptable error threshold
5. **Document**: Add to this README

Example cases to add:
- Couette flow (analytical)
- Backward-facing step (DNS/LES)
- Periodic hills (DNS)
- Taylor-Green vortex (exact decay)

## V&V Standards Compliance

This suite follows **ASME V&V 20-2009** guidelines:

- [OK] Multiple independent test cases
- [OK] Known reference solutions
- [OK] Quantitative error metrics
- [OK] Pass/fail criteria defined a priori
- [OK] Grid-independent solutions (medium resolution)

## Troubleshooting

### Case Doesn't Converge

**Symptoms**: max_steps reached, residual still high

**Fixes**:
```bash
# Increase iterations
max_steps = 100000

# Reduce CFL for stability
CFL_max = 0.3

# Reduce tolerance slightly
tol = 1e-7  # Instead of 1e-8
```

### Large Error on Turbulent Cases

**Expected**: RANS models have inherent 15-25% error

**To improve**:
1. Try GEP model (better than baseline)
2. Train NN model on DNS data
3. Use LES instead of RANS (not implemented)

### DNS Comparison Issues

If you have access to full DNS data files:

```python
# Load your DNS data
dns_data = np.load('moser_re180.npz')
y_dns = dns_data['y']
u_dns = dns_data['u']

# Add to validate_results.py
```

## References

### Analytical Solutions

1. **Poiseuille Flow**: White, F. M. (2006). "Viscous Fluid Flow" (3rd ed.). McGraw-Hill.

### DNS Benchmarks

2. **Channel Flow Re_τ=180,395**: Moser, R. D., Kim, J., & Mansour, N. N. (1999). "Direct numerical simulation of turbulent channel flow up to Re_τ=590." *Physics of Fluids*, 11(4), 943-945.

3. **DNS Database**: https://www.flow.kth.se/~pschlatt/DATA/

### V&V Methodology

4. **ASME Standard**: ASME V&V 20-2009. "Standard for Verification and Validation in Computational Fluid Dynamics and Heat Transfer."

5. **Oberkampf & Roy**: "Verification and Validation in Scientific Computing" (2010). Cambridge University Press.

## Related Examples

- **Example 1**: Laminar channel (same as validation case 1, with more detail)
- **Example 2**: Turbulent channel model comparison (extends validation cases 3-4)
- **Example 3**: Grid refinement (systematic convergence study)

---

**Key Takeaway**: Use this suite as a **regression test** - run before every major code change to ensure nothing breaks!

