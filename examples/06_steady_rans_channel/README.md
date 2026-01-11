# Example 06: Steady RANS Channel Flow

## Overview

This example demonstrates **steady-state RANS simulations** of turbulent channel flow at Re_τ = 180, comparing three turbulence models:

1. **Baseline** - Mixing length with van Driest damping (algebraic)
2. **GEP** - Gene Expression Programming (algebraic, data-driven formulas)
3. **SST k-ω** - Transport equation model (Menter 1994)

**Purpose**: Show how different RANS closures predict mean flow statistics in a canonical turbulent flow.

## Physical Setup

- **Geometry**: 2D channel with periodic streamwise BC, no-slip walls
- **Target**: Re_τ = 180 (friction Reynolds number)
- **Domain**: 2π × 2 (streamwise × wall-normal)
- **Grid**: 64 × 128 with wall-normal stretching
- **Driving force**: Constant pressure gradient (dp/dx = -0.0002)

## Running the Examples

### Quick Start (All Models)

```bash
cd examples/06_steady_rans_channel
./run_all.sh
```

This runs all three models and generates comparison plots.

### Individual Models

```bash
# From build directory
cd ../../build

# Baseline (mixing length)
./channel --config ../examples/06_steady_rans_channel/baseline.cfg

# GEP (algebraic)
./channel --config ../examples/06_steady_rans_channel/gep.cfg

# SST k-omega (transport)
./channel --config ../examples/06_steady_rans_channel/sst.cfg
```

## Expected Results

| Model | Type | Re_τ | Convergence | Notes |
|-------|------|------|-------------|-------|
| **Baseline** | Algebraic | ~172 | Fast (~5k iter) | Simple, robust |
| **GEP** | Algebraic | ~177 | Fast (~5k iter) | Better than baseline |
| **SST k-ω** | Transport | ~178 | Slower (~15k iter) | Most accurate |

**DNS Reference** (Moser et al. 1999): Re_τ = 180

## Output Files

Each model produces:
- `output/<model>/channel_final.vtk` - Final solution for ParaView
- `output/<model>/velocity_profile.dat` - Mean velocity vs y
- `output/<model>/channel_*.vtk` - Intermediate snapshots

## Visualization

### ParaView

```bash
paraview output/*/channel_final.vtk
```

Compare velocity profiles and eddy viscosity distributions.

### Python Analysis

```bash
python analyze_results.py
```

Generates:
- Mean velocity profiles (inner scaling: u+ vs y+)
- Eddy viscosity distributions
- Convergence histories
- Comparison with DNS data

## Key Differences Between Models

### Baseline (Mixing Length)

**Pros**:
- Fast convergence
- No transport equations
- Always stable

**Cons**:
- Fixed length scale (κy)
- Cannot adapt to complex flows
- Underpredicts Re_τ slightly

### GEP (Gene Expression Programming)

**Pros**:
- Data-driven formulas
- Better than mixing length
- Still algebraic (fast)

**Cons**:
- More complex expressions
- Requires careful tuning

### SST k-ω (Transport)

**Pros**:
- Most physically complete
- Predicts turbulence evolution
- Best accuracy

**Cons**:
- Slower convergence
- Two additional PDEs
- Requires initialization

## Modifying the Cases

### Change Reynolds Number

Edit config file:
```ini
nu = 0.002531645569620253  # For Re_τ = 395
dp_dx = -0.00048            # Adjust accordingly
```

### Increase Resolution

```ini
Nx = 128
Ny = 256
stretch_beta = 3.0  # More aggressive stretching
```

### Different Turbulence Model

```ini
turb_model = earsm_wj  # Wallin-Johansson EARSM
turb_model = komega    # Standard k-ω
```

## Validation Metrics

Compare with DNS:
1. **Friction velocity**: u_τ = √(τ_w/ρ)
2. **Mean velocity profile**: U+(y+)
3. **Skin friction coefficient**: C_f = 2(u_τ/U_b)²
4. **Centerline velocity**: U_CL

## Troubleshooting

**SST model doesn't converge:**
- Reduce time step: `dt = 0.0005`
- Increase max iterations: `max_steps = 100000`
- Check k/ω initialization (should be small but positive)

**Results don't match DNS:**
- Increase wall-normal resolution: `Ny = 256`
- Check y+ at first cell (should be < 1)
- Ensure simulation reached steady state

**Baseline/GEP give similar results:**
- Expected! Both are algebraic models
- GEP should be slightly better (~2-3% improvement)

## References

1. **DNS Benchmark**: Moser, Kim & Mansour (1999), JFM 399
2. **Mixing Length**: Van Driest (1956), J. Aero. Sci. 23
3. **GEP**: Weatheritt & Sandberg (2016), JFM 802
4. **SST k-ω**: Menter (1994), AIAA Journal 32(8)

## Related Examples

- **Example 05**: High-resolution SST channel at Re_τ = 180
- **Example 02**: Turbulent channel with NN models
- **Example 07**: Unsteady developing channel (no RANS)


