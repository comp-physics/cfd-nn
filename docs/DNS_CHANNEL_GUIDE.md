# DNS Channel Flow Guide

This guide covers Direct Numerical Simulation (DNS) of turbulent channel flow using the incompressible Navier-Stokes solver. DNS resolves all scales of turbulence without any modeling assumptions, requiring fine grids and careful parameter tuning.

## Table of Contents

- [Overview](#overview)
- [Grid Requirements](#grid-requirements)
- [Time Stepping: Directional CFL](#time-stepping-directional-cfl)
- [Trip Forcing](#trip-forcing)
- [Velocity Filter](#velocity-filter)
- [Turbulence Diagnostics](#turbulence-diagnostics)
- [Step Sequence](#step-sequence)
- [Example Configuration](#example-configuration)
- [Troubleshooting](#troubleshooting)
- [Known Limitations](#known-limitations)

---

## Overview

DNS of channel flow at Re_tau = 180 is a canonical benchmark (Moser, Kim & Mansour 1999). The solver uses:

- **Fractional-step projection** with explicit time integration (Euler, RK2, or RK3)
- **Staggered MAC grid** with second-order central differences
- **Periodic** x/z boundaries, **no-slip walls** at y = +/-1
- **Constant body force** dp/dx driving the flow
- **Adaptive time stepping** with directional CFL constraints
- **Velocity filter** for grid-scale stability
- **Trip forcing** to trigger laminar-to-turbulent transition

Set `simulation_mode = unsteady` and `turb_model = none` for DNS.

---

## Grid Requirements

DNS requires resolving the Kolmogorov scale. For channel flow at Re_tau ~ 180, standard resolution targets are:

| Direction | Target (wall units) | Rule of Thumb |
|-----------|-------------------|---------------|
| Streamwise (x) | dx+ < 15 | ~12 recommended |
| Wall-normal (y) | y1+ < 0.5 | First cell off wall |
| Spanwise (z) | dz+ < 8 | ~6 recommended |

### Canonical Grid: 192 x 96 x 192

```
Nx = 192        # Streamwise
Ny = 96         # Wall-normal (half what you'd expect — stretched grid doubles effective resolution)
Nz = 192        # Spanwise
x_min = 0.0
x_max = 12.566  # 4*pi (or 2*pi for shorter domain)
y_min = -1.0
y_max = 1.0
z_min = 0.0
z_max = 6.283   # 2*pi
stretch_y = true
stretch_beta = 2.0
```

### Wall Units Conversion

Wall unit quantities are computed from the friction velocity:

```
u_tau = sqrt(|tau_wall| / rho) = sqrt(nu * |dU/dy|_wall)
y+    = y * u_tau / nu
dx+   = dx * u_tau / nu
```

For Re_tau = 180 with nu = 1/5600:

| Quantity | Value |
|----------|-------|
| dx+ | ~11.8 |
| dz+ | ~5.9 |
| y1+ | ~0.29 |

### Grid Stretching

The `stretch_y = true` option applies tanh stretching to cluster grid points near walls:

```
y(eta) = tanh(beta * eta) / tanh(beta)
```

where eta is the uniform coordinate and beta controls clustering intensity. Higher beta = more points near walls. `stretch_beta = 2.0` is standard for Re_tau = 180.

---

## Time Stepping: Directional CFL

### The Problem

Stretched grids create a severe aspect ratio mismatch: dy_min near walls can be 100x smaller than dx or dz. A single CFL number forces unnecessarily small time steps because the wall-normal velocity v is typically small near walls.

### Directional CFL Solution

The solver supports separate CFL numbers for different directions:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `CFL_max` | 0.5 | CFL for **y-direction** (wall-normal, strict) |
| `CFL_xz` | -1.0 | CFL for **x/z directions** (-1 = use CFL_max) |
| `dt_safety` | 1.0 | Safety multiplier applied after CFL computation |

The adaptive time step is computed as:

```
dt_cfl_x = CFL_xz * dx / max|u|
dt_cfl_y = CFL_max / max(|v| / dy_local)    # accounts for variable dy
dt_cfl_z = CFL_xz * dz / max|w|
dt_diff  = 0.25 * min(dx, dy_min, dz)^2 / (nu + max(nu_t))

dt = dt_safety * min(dt_cfl_x, dt_cfl_y, dt_cfl_z, dt_diff)
```

### Recommended Values for DNS

```
CFL_max = 0.15      # Strict for y (stretched grid, small dy)
CFL_xz = 0.30       # Relaxed for x/z (large, uniform spacing)
dt_safety = 0.85     # 15% headroom for within-step CFL growth
adaptive_dt = true
```

**Why CFL_max = 0.15?** The wall-normal direction has the smallest grid spacing and is where blow-ups originate. A strict CFL here prevents instability without penalizing the time step as much as a single CFL = 0.15 would (since x/z use the relaxed CFL_xz = 0.30).

**Why dt_safety = 0.85?** Velocities can grow within a time step (especially during transition). The safety factor provides headroom so the CFL constraint isn't violated mid-step.

---

## Trip Forcing

### Purpose

DNS of channel flow starts from a laminar (Poiseuille) profile plus small random perturbations. Without explicit forcing, these perturbations may decay rather than trigger transition. Trip forcing injects energy at the right scales to reliably initiate turbulence.

### How It Works

The trip adds a body force to v and w (wall-normal and spanwise) during the predictor step:

```
f_trip = A * env_x(x) * g_y(y) * F_z(z) * ramp(t)
```

**Components:**

| Function | Formula | Purpose |
|----------|---------|---------|
| `A` | `trip_amplitude * u_tau^2` | Overall amplitude scaled by friction velocity |
| `env_x(x)` | Cosine window: `0.5*(1 - cos(2*pi*xi))` over trip region | Localize in x |
| `g_y(y)` | `y * (1 - y^2)` | Concentrate forcing in buffer layer (zero at walls) |
| `F_z(z)` | Sum of spanwise modes with 1/(m+1) weighting | Multi-scale spanwise perturbation |
| `ramp(t)` | Cosine ramp from 1 to 0 during ramp-off phase | Smooth temporal shutdown |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trip_enabled` | false | Enable trip forcing |
| `trip_amplitude` | 3.0 | Amplitude (1-5 typical). Scaled by u_tau^2 |
| `trip_x_start` | -1.0 | Start of trip region (-1 = auto: 0.1*Lx) |
| `trip_x_end` | -1.0 | End of trip region (-1 = auto: 0.2*Lx) |
| `trip_duration` | 2.0 | Total duration of trip forcing |
| `trip_ramp_off_start` | 1.5 | When ramp-off begins |
| `trip_n_modes_z` | 8 | Number of spanwise Fourier modes |
| `trip_force_w` | true | Also force w (creates vortical structures) |
| `trip_w_scale` | 1.0 | Scale factor for w forcing (>1 boosts 3D structures) |

### CRITICAL: Time Units

**`trip_duration` and `trip_ramp_off_start` are in PHYSICAL time**, compared against `current_time_` in the solver. They are NOT in friction time units (t* = t * u_tau / delta) or step counts. This is a common source of confusion.

Example: With `trip_duration = 0.20` and `trip_ramp_off_start = 0.10`, the trip is:
- Full strength for t in [0, 0.10]
- Ramping off for t in [0.10, 0.20]
- Off for t > 0.20

### Time Ramp-off Formula

```
if t >= trip_duration:
    ramp = 0.0                          # Trip is off
elif t <= trip_ramp_off_start:
    ramp = 1.0                          # Full strength
else:
    frac = (t - trip_ramp_off_start) / (trip_duration - trip_ramp_off_start)
    ramp = 0.5 * (1 + cos(pi * frac))   # Smooth cosine decay: 1 -> 0
```

### Recommended Values

```
trip_enabled = true
trip_amplitude = 1.0
trip_duration = 0.20
trip_ramp_off_start = 0.10
trip_w_scale = 2.0       # Boost w to promote 3D structures
trip_n_modes_z = 8
```

**Tuning tips:**
- If the flow relaminarizes after trip turns off, increase `trip_amplitude` or extend `trip_duration`
- If blow-up occurs during trip, reduce `trip_amplitude` or decrease CFL
- `trip_w_scale > 1` helps create streamwise vortices that sustain turbulence
- The trip region is automatically placed at 10-20% of the domain length

---

## Velocity Filter

### Purpose

Second-order central differences have no numerical dissipation, so grid-scale oscillations (2dx waves) can grow without bound. The velocity filter applies a small amount of explicit diffusion to drain energy from these unresolved modes.

### Formula

The filter applies a discrete Laplacian in all three directions:

```
u_filtered = u + alpha * (Lx + Lz) + alpha_y * Ly
```

where:
- `Lx = u[i-1] - 2*u[i] + u[i+1]` (discrete Laplacian in x)
- `Ly = u[j-1] - 2*u[j] + u[j+1]` (discrete Laplacian in y)
- `Lz = u[k-1] - 2*u[k] + u[k+1]` (discrete Laplacian in z)
- `alpha = filter_strength * 0.25` (x/z coefficient)
- `alpha_y = alpha * 0.5` (y coefficient, reduced for stretched grid)

The y-direction coefficient is halved because stretched grids already have small dy near walls, so applying the same filter strength in y would be excessively diffusive there.

### Wall Boundary Treatment

- **u and w** (cell-centered in y): Skip wall-adjacent cells (j = Ng and j = Ny+Ng-1) because the Laplacian stencil would cross the wall boundary
- **v** (face-centered in y): Only filter interior faces (Ng < j < Ny+Ng); wall faces (j = Ng, j = Ny+Ng) are always v = 0

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `filter_strength` | 0.0 | Filter coefficient (0 = disabled). Range: 0.01 - 0.05 |
| `filter_interval` | 10 | Apply filter every N steps (0 = disabled) |

### CRITICAL: Filter Timing

**The filter MUST be applied BEFORE the projection step**, not after. This is because:

1. The filter introduces a small amount of divergence (it modifies velocities independently)
2. The subsequent pressure projection step removes this divergence, restoring div(u) = 0
3. If the filter were applied AFTER projection, the velocity field would have nonzero divergence until the next time step

The solver applies the filter at the beginning of each time step (before the RHS computation).

### Effective Viscosity

The filter acts like additional viscosity. The effective viscosity added by the filter is approximately:

```
nu_filter ~ alpha * dx^2 / dt_filter
```

where `dt_filter = dt * filter_interval`. This means stronger filters or more frequent application increase the effective Reynolds number.

### Tuning Guide

Results from DNS channel flow at Re_tau = 180 (192x96x192 grid):

| strength | interval | Result | Re_tau | Effective nu added |
|----------|----------|--------|--------|-------------------|
| 0.05 | 1 | Stable, ran 3600+ steps | ~255 | ~55x nu |
| 0.03 | 2 | Stable, ran 2400+ steps | ~278 | ~16x nu |
| 0.01 | 5 | Blew up at step 1955 | N/A | ~2x nu |
| 0.00 | - | Blows up at ~1700 steps | N/A | 0 |

**Key tradeoff:** Stronger filter = more stable but higher effective Re_tau (further from target 180). The filter adds effective viscosity that reduces the friction Reynolds number below the target.

**Recommended starting point:**

```
filter_strength = 0.03
filter_interval = 2
```

This provides stability while keeping the effective viscosity overhead manageable (~16x nu).

---

## Turbulence Diagnostics

### Health Checks

The solver periodically monitors simulation health. Set `diag_interval` to control frequency (default: 1, set higher for performance).

**Checked quantities:**
- Maximum velocity components (u, v, w)
- CFL numbers in each direction
- Divergence norms (L2, L-infinity)
- Re_tau (friction Reynolds number)
- Bulk velocity

**Safety cutoff:** If any velocity exceeds `safety_vel_max` (typically set via the NaN/Inf guard), the simulation aborts with a SAFETY-VEL error. This catches blow-ups before they produce NaN.

### Turbulence Presence Detection

The solver automatically classifies the flow state using plane-averaged statistics:

| Indicator | Turbulent Threshold | What It Measures |
|-----------|-------------------|------------------|
| `u_tau_ratio` | ~ 1.0 | Friction velocity consistency (wall vs body force) |
| `u_rms_mid` | > 0 | Velocity fluctuations at channel center |
| `tke_mid` | > 0 | Turbulent kinetic energy at center |
| `max_uv_plus` | > 0.3 | Peak Reynolds shear stress in 5 < y+ < 150 |

### Stage F Realism Validation

A more thorough check (`validate_turbulence_realism()`) verifies:

1. **Grid resolution**: y1+ < 1.0, dx+ < 15, dz+ < 8
2. **u_tau consistency**: Wall-derived vs force-derived u_tau agree within tolerance
3. **Momentum balance**: Total shear stress matches theoretical profile
4. **Reynolds stress ordering**: `<u'u'> > <w'w'> > <v'v'>` in the log layer (10 < y+ < 100)
5. **Reynolds shear stress shape**: `-<u'v'>+` is small at walls, O(1) in the interior

### Statistics Accumulation

Plane-averaged statistics (mean profiles, Reynolds stresses) are accumulated over time:

```
stats_U_mean[j] = running average of plane-averaged U at each y
stats_uu[j]     = running average of <u'u'> at each y
stats_vv[j]     = running average of <v'v'> at each y
stats_uv[j]     = running average of <u'v'> at each y
```

**GPU sync requirement:** `accumulate_statistics()` reads velocity data on the CPU. It calls `sync_solution_from_gpu()` internally. Without this sync, CPU-side statistics would read stale data.

---

## Step Sequence

Each time step follows this order:

```
1.  Copy u^n -> u_old (for residual computation)
2.  Update body force (if ramping enabled)
3.  Turbulence transport update (if RANS model active — not used in DNS)
4.  Turbulence model update (compute nu_t — zero for DNS)
5.  Compute effective viscosity: nu_eff = nu + nu_t
6.  ** VELOCITY FILTER ** (if enabled and step % filter_interval == 0)
7.  Compute convective terms: conv = nabla . (uu)
8.  Compute diffusive terms: diff = nabla^2 u
9.  Predictor: u* = u + dt * (-conv + diff + f_body)
10. ** TRIP FORCING ** applied to v* and w* (if enabled and ramp > 0)
11. Apply velocity BCs to u*
12. ** RECYCLING INFLOW ** (if enabled): extract, shift, correct, blend
13. Compute divergence: div = nabla . u*
14. Poisson solve: nabla^2 p = div / dt
15. Velocity correction: u = u* - dt * nabla p
16. Apply velocity BCs to corrected u
17. Post-projection recycling extraction (if enabled)
18. Divergence check (if diag_interval met)
19. Compute residual: max|u_new - u_old|
20. NaN/Inf guard check
21. Increment step counter and current_time
```

---

## Example Configuration

Complete working config for DNS channel flow (the "v13 recipe"):

```ini
# DNS Channel Flow - Re_tau ~ 180 target
# Achieves Re_tau ~ 278 (filter adds effective viscosity)

# Simulation mode
simulation_mode = unsteady
turb_model = none

# Grid: 192x96x192, 4pi x 2 x 2pi domain
Nx = 192
Ny = 96
Nz = 192
x_min = 0.0
x_max = 12.566
y_min = -1.0
y_max = 1.0
z_min = 0.0
z_max = 6.283
stretch_y = true
stretch_beta = 2.0

# Physics
nu = 0.000178571    # 1/5600, gives Re_tau = 180 with dp/dx = -1
dp_dx = -1.0

# Time stepping - directional CFL
adaptive_dt = true
CFL_max = 0.15
CFL_xz = 0.30
dt_safety = 0.85

# Initial perturbation
perturbation_amplitude = 0.01

# Trip forcing
trip_enabled = true
trip_amplitude = 1.0
trip_duration = 0.20
trip_ramp_off_start = 0.10
trip_w_scale = 2.0

# Velocity filter
filter_strength = 0.03
filter_interval = 2

# Poisson solver
poisson_solver = auto
poisson_tol_rhs = 1e-6
poisson_fixed_cycles = 8

# Performance
perf_mode = true
gpu_only_mode = true
max_steps = 5000
output_freq = 100
```

Run with:
```bash
./channel --config dns_channel.cfg
```

---

## Troubleshooting

### Blow-Up (SAFETY-VEL or NaN)

**Symptoms:** Velocity suddenly jumps to O(100+), then NaN appears.

**Common causes and fixes:**

| Cause | Symptom | Fix |
|-------|---------|-----|
| CFL too high | Blow-up within first 100 steps | Reduce CFL_max (try 0.10) |
| No filter | Blow-up after trip turns off (~step 1500-2000) | Enable filter (strength=0.03, interval=2) |
| Filter too weak | Blow-up after initial turbulence develops | Increase strength or decrease interval |
| Trip too strong | Blow-up during trip phase | Reduce trip_amplitude |
| Trip too long | Sustained high velocities | Reduce trip_duration |
| dt_safety too high | Intermittent blow-ups | Reduce to 0.80 or lower |
| Grid too coarse | Blow-up at any settings | Increase resolution |

### Re_tau Too High

**Symptom:** Stable simulation but Re_tau ~ 250-300 instead of target 180.

**Cause:** Velocity filter adds effective viscosity, which increases the effective Re_tau. The filter acts like additional physical viscosity that shifts the momentum balance.

**Mitigation:**
- Reduce `filter_strength` (but may cause blow-up)
- Increase `filter_interval` (applies filter less often)
- The fundamental tradeoff: stability vs accuracy. Achieving Re_tau = 180 exactly may require a less dissipative numerical scheme (e.g., hybrid skew-symmetric/upwind convection)

### Relaminarization

**Symptom:** Turbulence develops during trip but decays after trip turns off.

**Fixes:**
- Increase `trip_w_scale` (promotes 3D vortical structures)
- Increase `trip_duration` (let turbulence establish longer)
- Increase `trip_amplitude` (more energy injection)
- Ensure `perturbation_amplitude` is nonzero (seeds initial fluctuations)

### GPU Sync Issues

**Symptom:** Diagnostics show zero or stale values on GPU builds.

**Cause:** CPU-side functions reading GPU data without syncing first.

**Rule:** Any function that reads velocity/pressure data on the CPU must call `sync_solution_from_gpu()` first. The solver does this automatically for `accumulate_statistics()` and `validate_turbulence_realism()`, but custom diagnostic code must do it manually.

---

## Known Limitations

1. **Filter-limited Re_tau:** The velocity filter adds effective viscosity, preventing the simulation from reaching the target Re_tau = 180. Best achieved: Re_tau ~ 278 with strength=0.03, interval=2. Reaching the target would require a less dissipative convective scheme.

2. **Explicit time integration only:** The solver uses explicit Euler, RK2, or RK3. There is no implicit diffusion treatment, so the diffusive stability limit (dt < 0.25 * dx_min^2 / nu) constrains the time step at low viscosity.

3. **No higher-order spatial schemes for DNS:** The solver uses second-order central differences. Higher-order (4th, 6th) schemes would reduce the filter strength needed for stability.

4. **Single-GPU only:** The solver runs on a single GPU. Domain decomposition / MPI parallelism is not implemented.

5. **No restart capability:** The solver does not write/read checkpoint files. Long DNS runs that exceed allocation time must restart from scratch.

---

## References

- Moser, R. D., Kim, J., & Mansour, N. N. "Direct numerical simulation of turbulent channel flow up to Re_tau = 590." *Physics of Fluids* 11.4 (1999): 943-945
- Kim, J., Moin, P., & Moser, R. "Turbulence statistics in fully developed channel flow at low Reynolds number." *J. Fluid Mech.* 177 (1987): 133-166
