# Turbulent Channel Flow at Re_τ = 180

This example demonstrates a fully-developed turbulent channel flow simulation that can be directly compared to DNS benchmark data.

## Case Description

- **Geometry**: Plane channel (periodic in streamwise direction)
- **Target Reynolds number**: Re_τ = 180 (friction Reynolds number)
- **Turbulence model**: SST k-ω (Menter, 1994)
- **Benchmark reference**: Moser, Kim & Mansour (1999), JFM 399, 263-291
- **DNS data source**: https://turbulence.oden.utexas.edu/data/MKM/

## Physical Setup

The configuration uses:
- Channel half-height: δ = 1.0 (domain: -1 ≤ y ≤ 1)
- Kinematic viscosity: ν = 1/180 ≈ 0.00556
- Pressure gradient: dp/dx = -1.0
- Density: ρ = 1.0

This gives:
- Friction velocity: u_τ = √(τ_w/ρ) = 1.0
- Friction Reynolds: Re_τ = u_τ δ / ν = 180

## Grid Resolution

- Streamwise (x): 256 cells over domain length 2π
- Wall-normal (y): 512 cells with hyperbolic tangent stretching (β = 2.5)
- Total: 131,072 cells

The stretching concentrates grid points near walls for better resolution of the viscous sublayer.

## Running the Simulation

### 1. Submit to H200 GPU

```bash
cd examples/05_channel_retau180_sst
sbatch run_h200.sbatch
```

The simulation will run for up to 200,000 iterations or until convergence (residual < 1e-8). Expected runtime: 30-60 minutes on H200.

### 2. Monitor progress

```bash
tail -f slurm-<jobid>.out
```

### 3. Check output

Results are written to `output/`:
- `channel_*.vtk` - VTK snapshots for visualization (20 snapshots)
- `velocity_profile.dat` - Mean velocity profile vs y-coordinate
- `channel_velocity.dat` - Full field data
- `channel_pressure.dat` - Pressure field

## Post-Processing

After the simulation completes, compare results to DNS data:

```bash
python3 compare_dns.py
```

This will:
1. Load the computed velocity profile
2. Transform to wall units (y⁺, U⁺)
3. Compare with Moser et al. DNS reference data
4. Generate comparison plot: `output/velocity_comparison.png`
5. Print error metrics

### Expected Results

Good agreement should show:
- **Viscous sublayer** (y⁺ < 5): U⁺ ≈ y⁺
- **Buffer layer** (5 < y⁺ < 30): Transition region
- **Log layer** (30 < y⁺ < 180): U⁺ = (1/κ)ln(y⁺) + B where κ ≈ 0.41, B ≈ 5.2
- **RMS error**: Typically 5-15% for RANS models vs DNS

## Modifying the Case

### Change turbulence model

Edit `channel.cfg`:
```ini
turb_model = sst          # Menter SST k-ω (default)
# turb_model = komega     # Standard k-ω (Wilcox)
# turb_model = baseline   # Simple mixing length
# turb_model = gep        # GEP algebraic model
# turb_model = earsm_wj   # EARSM (Wallin-Johansson)
```

### Change Reynolds number

For Re_τ = 395 (Moser et al. higher Re case):
```ini
nu = 0.002531645569620253  # = 1/395
dp_dx = -1.0               # keep same
Ny = 1024                  # increase resolution
```

### Adjust grid resolution

```ini
Nx = 512    # finer streamwise
Ny = 1024   # finer wall-normal
```

Note: Wall-normal resolution is most critical. Aim for Δy⁺ < 1 at the wall.

## Validation Metrics

Compare these quantities with DNS:
1. **Mean velocity profile** U⁺(y⁺)
2. **Friction Reynolds number** Re_τ (should be ~180)
3. **Skin friction coefficient** C_f = 2(u_τ/U_b)²
4. **Centerline velocity** U_CL⁺
5. **Bulk velocity** U_b

## References

1. **DNS Benchmark**:
   Moser, R.D., Kim, J. & Mansour, N.N. (1999)
   "Direct numerical simulation of turbulent channel flow up to Re_tau=590"
   Journal of Fluid Mechanics, 399, 263-291
   https://doi.org/10.1017/S0022112099006851

2. **SST k-ω Model**:
   Menter, F.R. (1994)
   "Two-equation eddy-viscosity turbulence models for engineering applications"
   AIAA Journal, 32(8), 1598-1605

3. **DNS Data Repository**:
   https://turbulence.oden.utexas.edu/data/MKM/

## Troubleshooting

**Simulation doesn't converge:**
- Reduce time step: `dt = 0.0005`
- Increase max iterations: `max_iter = 500000`
- Check CFL number in output (should be < 0.5)

**Poor agreement with DNS:**
- Increase wall-normal resolution: `Ny = 1024`
- Increase stretching: `stretch_beta = 3.0`
- Try different turbulence model
- Ensure simulation has reached steady state (check residual history)

**GPU memory issues:**
- Reduce grid size: `Nx = 128`, `Ny = 256`
- Check available memory: `nvidia-smi`

**Job times out:**
- Increase time limit: `#SBATCH --time=04:00:00`
- Or reduce max_iter to checkpoint intermediate results

