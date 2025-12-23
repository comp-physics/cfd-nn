# Example 07: Unsteady Developing Channel Flow

## Overview

This example demonstrates **time-accurate unsteady simulation** of laminar channel flow without turbulence modeling. It showcases:

- Divergence-free initialization via streamfunction
- Time-accurate integration (no steady-state assumption)
- Laminar flow dynamics and transients
- Viscous dissipation of perturbations

**Purpose**: Contrast with steady RANS examples (05, 06) to show the difference between time-accurate DNS-like simulation and steady-state turbulence modeling.

## Physical Setup

- **Geometry**: 2D channel with periodic streamwise BC, no-slip walls
- **Reynolds number**: Re_τ ~ 180 (same as RANS examples)
- **Initial condition**: Divergence-free perturbation (∇·u = 0 exactly)
- **Turbulence model**: None (laminar/DNS-like)
- **Time integration**: Explicit with adaptive time stepping

## Key Differences from Steady RANS

| Feature | Steady RANS (Ex 06) | Unsteady Developing (Ex 07) |
|---------|---------------------|------------------------------|
| **Goal** | Steady-state mean flow | Time-accurate dynamics |
| **Turbulence** | RANS closure (ν_t) | None (laminar) |
| **Initialization** | Uniform flow | Divergence-free perturbation |
| **Convergence** | Residual < tolerance | Fixed time steps |
| **Output** | Final steady state | Time series snapshots |

## Running the Example

### Quick Start

```bash
cd examples/07_unsteady_developing_channel
./run.sh
```

### Manual Execution

```bash
# From build directory
cd ../../build

./channel --config ../examples/07_unsteady_developing_channel/laminar.cfg
```

**Note**: The `channel` executable supports both steady and unsteady modes. The `simulation_mode = unsteady` setting in the config file switches to time-accurate integration.

## Configuration Files

### `laminar.cfg` - Standard Run

- **Time steps**: 5000
- **Grid**: 64 × 128
- **Snapshots**: 20 (every 250 steps)
- **Runtime**: ~2-5 minutes

### `laminar_fine.cfg` - High Resolution

- **Time steps**: 10000
- **Grid**: 128 × 256
- **Snapshots**: 40
- **Runtime**: ~15-30 minutes

## Expected Behavior

### Phase 1: Initial Perturbation (t = 0)

- Small-amplitude divergence-free velocity field
- Streamwise vortices from streamfunction
- ∇·u = 0 to machine precision

### Phase 2: Transient Development (t = 0 to ~100)

- Perturbations advect and diffuse
- Viscous dissipation reduces kinetic energy
- Flow adjusts to pressure gradient

### Phase 3: Approach to Steady State (t > 100)

- Perturbations decay exponentially
- Flow approaches laminar Poiseuille profile
- Residual decreases monotonically

## Output Files

```
output/
├── developing_channel_1.vtk    # Snapshot 1
├── developing_channel_2.vtk    # Snapshot 2
├── ...
├── developing_channel_20.vtk   # Snapshot 20
└── developing_channel_final.vtk # Final state
```

## Visualization

### ParaView Animation

```bash
paraview output/developing_channel_*.vtk
```

**Workflow**:
1. Load all VTK files as a time series
2. Apply "Warp By Vector" filter to visualize velocity
3. Color by velocity magnitude
4. Play animation to see flow evolution

### Python Analysis

```bash
python analyze_transient.py
```

Generates:
- Kinetic energy vs time
- Velocity profile evolution
- Divergence history (should stay ~0)
- Comparison with Poiseuille solution

## Physical Insights

### Divergence-Free Constraint

The initialization guarantees ∇·u = 0 exactly:
```
ψ(x,y) = A sin(kx·x) sin²(π(y+1)/2)
u = ∂ψ/∂y,  v = -∂ψ/∂x
```

This is maintained by the projection method to machine precision.

### Energy Dissipation

Kinetic energy decays as:
```
KE(t) ≈ KE(0) exp(-2νk²t)
```

where k is the wavenumber of the perturbation.

### Laminar vs Turbulent

Without turbulence modeling:
- Flow remains laminar (Re not high enough for transition)
- No turbulent mixing (ν_t = 0)
- Converges to Poiseuille profile
- Much slower than turbulent case

## Modifying the Example

### Change Initial Amplitude

Edit `app/main_channel.cpp` (in the unsteady mode section):
```cpp
solver.initialize(create_perturbed_channel_field(mesh, 1e-2));  // Larger perturbation
```

### Different Time Step

```ini
dt = 0.0005          # Smaller dt for stability
adaptive_dt = false  # Fixed time step
```

### More Snapshots

```ini
num_snapshots = 50   # More frequent output
max_iter = 10000     # More time steps
```

## Validation Checks

1. **Divergence**: Should remain < 1e-10 throughout
2. **Energy decay**: Should be monotonic (no growth)
3. **Final profile**: Should match Poiseuille solution
4. **Mass conservation**: Total mass should be constant

## Troubleshooting

**Simulation diverges (NaN/Inf):**
- Reduce time step: `dt = 0.0005`
- Reduce CFL: `CFL_max = 0.3`
- Check initial condition (should be smooth)

**Divergence increases:**
- Bug in projection method (should not happen!)
- Check Poisson solver tolerance: `poisson_tol = 1e-10`

**Flow doesn't evolve:**
- Check body force: `dp_dx = -0.0002`
- Increase time steps: `max_iter = 10000`

## Comparison with RANS

Run both examples and compare:

```bash
# Steady RANS (Example 06)
cd ../06_steady_rans_channel
./run_all.sh

# Unsteady laminar (Example 07)
cd ../07_unsteady_developing_channel
./run.sh

# Compare final states
paraview ../06_steady_rans_channel/output/sst/channel_final.vtk \
         output/developing_channel_final.vtk
```

**Key differences**:
- RANS: Higher velocity (turbulent mixing)
- Laminar: Lower velocity (no turbulent transport)
- RANS: Smooth convergence to steady state
- Laminar: Transient oscillations before settling

## Extensions

### 1. Transition to Turbulence

Increase Reynolds number:
```ini
nu = 0.0001  # Higher Re
```

May observe:
- Longer transient phase
- Instability growth
- Possible transition (if Re high enough)

### 2. Forced Perturbations

Add time-dependent body force to sustain perturbations.

### 3. Different Initial Conditions

Try:
- Random perturbations
- Specific instability modes
- Localized disturbances

## References

1. **Streamfunction method**: Chorin (1968), "Numerical solution of Navier-Stokes"
2. **Projection method**: Kim & Moin (1985), "Application of a fractional-step method"
3. **Channel flow DNS**: Moser et al. (1999), JFM 399

## Related Examples

- **Example 06**: Steady RANS channel (turbulent)
- **Example 01**: Laminar Poiseuille (steady)
- **Example 04**: Validation suite (various cases)


