# Solver Guide

## Overview

CFD-NN is an incompressible Navier-Stokes solver in C++17 with GPU acceleration via OpenMP target offload. It uses a fractional-step projection method on a staggered MAC grid with second-order finite differences.

**Solver modes:**

| Mode | Turbulence model | Description |
|------|-----------------|-------------|
| DNS | `turb_model = none` | No model; all scales resolved on the grid |
| LES | `turb_model = smagorinsky` (etc.) | Subgrid-scale model computes nu_sgs |
| RANS | `turb_model = komega` (etc.) | Full turbulence closure (algebraic, transport, EARSM, or neural network) |

**Key features:**
- Staggered MAC grid: u at x-faces, v at y-faces, w at z-faces, p at cell centers
- Periodic, no-slip wall, and recycling inflow boundary conditions
- Adaptive time stepping with directional CFL constraints
- 6 Poisson solver backends: FFT, FFT2D, FFT1D, HYPRE, Multigrid, SOR
- Immersed boundary method (direct forcing)
- CUDA Graph optimization for multigrid V-cycles

---

## Solver Step Pipeline

The `RANSSolver::step()` method implements a fractional-step projection. The full pipeline is documented in `CLAUDE.md`; the summary is:

1. **Turbulence update** -- compute nu_t (LES: fused GPU kernel; RANS: transport + algebraic; DNS: skip)
2. **Effective viscosity** -- `nu_eff = nu + nu_t`
3. **Convective + diffusive terms** -- from current velocity
4. **Predictor** -- `u* = u^n + dt*(-conv + diff + f_body)` (explicit)
5. **IBM forcing** (if enabled) -- multiply u* by pre-computed weight arrays
6. **Recycling inlet** (if enabled) -- extract, shift, correct, blend
7. **IBM RHS masking** (if enabled) -- zero Poisson RHS at solid cells (GPU kernel)
8. **Pressure Poisson** -- solve for pressure correction p'
9. **Velocity correction** -- `u^{n+1} = u* - dt*grad(p')`
10. **IBM re-forcing** (if enabled) -- re-apply weight multiplication
11. **Boundary conditions** -- periodic halos, no-slip walls
12. **Residual** -- `max|u^{n+1} - u^n|` via GPU reduction

For RK2/RK3, steps 3--7 repeat per stage with SSP weights. The velocity filter (when enabled) is applied BEFORE step 1 in the application time loop, not inside `step()`.

---

## DNS Channel Flow

DNS of channel flow at Re_tau = 180 is the canonical benchmark (Moser, Kim & Mansour 1999). Set `simulation_mode = unsteady` and `turb_model = none`.

### Grid Requirements

DNS requires resolving the Kolmogorov scale. For Re_tau ~ 180:

| Direction | Target (wall units) | Recommendation |
|-----------|-------------------|----------------|
| Streamwise (x) | dx+ < 15 | ~12 |
| Wall-normal (y) | y1+ < 0.5 | First cell off wall |
| Spanwise (z) | dz+ < 8 | ~6 |

**Canonical grid: 192 x 96 x 192** with tanh stretching (`stretch_beta = 2.0`) in y. Domain: 4pi x 2 x 2pi.

| Quantity | Value (Re_tau = 180, nu = 1/5600) |
|----------|-----------------------------------|
| dx+ | ~11.8 |
| dz+ | ~5.9 |
| y1+ | ~0.29 |

Grid stretching formula: `y(eta) = tanh(beta * eta) / tanh(beta)`, where eta is the uniform coordinate and beta controls wall clustering.

### Directional CFL

Stretched grids create severe aspect ratio mismatch: dy_min near walls can be 100x smaller than dx or dz. The solver supports separate CFL numbers per direction:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `CFL_max` | 0.5 | CFL for y-direction (wall-normal, strict) |
| `CFL_xz` | -1.0 | CFL for x/z directions (-1 = use CFL_max) |
| `dt_safety` | 1.0 | Safety multiplier on computed dt |

The adaptive time step:

```
dt_cfl_x = CFL_xz * dx / max|u|
dt_cfl_y = CFL_max / max(|v| / dy_local)
dt_cfl_z = CFL_xz * dz / max|w|
dt_diff  = 0.25 * min(dx, dy_min, dz)^2 / (nu + max(nu_t))

dt = dt_safety * min(dt_cfl_x, dt_cfl_y, dt_cfl_z, dt_diff)
```

**Recommended DNS values:**

```
CFL_max = 0.15       # Strict for y (stretched grid, small dy)
CFL_xz = 0.30        # Relaxed for x/z (large, uniform spacing)
dt_safety = 0.85      # 15% headroom for within-step CFL growth
adaptive_dt = true
```

CFL_max = 0.15 is strict because the wall-normal direction has the smallest spacing and is where blow-ups originate. dt_safety = 0.85 provides headroom since velocities can grow within a step during transition.

### Trip Forcing

Trip forcing injects energy to trigger laminar-to-turbulent transition. It adds a body force to v and w during the predictor step:

```
f_trip = A * env_x(x) * g_y(y) * F_z(z) * ramp(t)
```

| Component | Formula | Purpose |
|-----------|---------|---------|
| `A` | `trip_amplitude * u_tau^2` | Overall amplitude |
| `env_x(x)` | Cosine window over trip region | Localize in x |
| `g_y(y)` | `y * (1 - y^2)` | Concentrate in buffer layer |
| `F_z(z)` | Spanwise modes with 1/(m+1) weighting | Multi-scale perturbation |
| `ramp(t)` | Cosine ramp from 1 to 0 | Smooth temporal shutdown |

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trip_enabled` | false | Enable trip forcing |
| `trip_amplitude` | 3.0 | Amplitude (1--5 typical), scaled by u_tau^2 |
| `trip_x_start` | -1.0 | Start of trip region (-1 = auto: 0.1*Lx) |
| `trip_x_end` | -1.0 | End of trip region (-1 = auto: 0.2*Lx) |
| `trip_duration` | 2.0 | Total duration of trip forcing |
| `trip_ramp_off_start` | 1.5 | When ramp-off begins |
| `trip_n_modes_z` | 8 | Number of spanwise Fourier modes |
| `trip_force_w` | true | Also force w component |
| `trip_w_scale` | 1.0 | Scale factor for w forcing (>1 boosts 3D structures) |

**CRITICAL: `trip_duration` and `trip_ramp_off_start` are in PHYSICAL time** (compared against `current_time_`). They are NOT in friction time units or step counts.

Ramp-off formula:

```
if t >= trip_duration:
    ramp = 0.0
elif t <= trip_ramp_off_start:
    ramp = 1.0
else:
    frac = (t - trip_ramp_off_start) / (trip_duration - trip_ramp_off_start)
    ramp = 0.5 * (1 + cos(pi * frac))
```

**Recommended values:**

```
trip_enabled = true
trip_amplitude = 1.0
trip_duration = 0.20
trip_ramp_off_start = 0.10
trip_w_scale = 2.0
trip_n_modes_z = 8
```

### Velocity Filter

Second-order central differences have no numerical dissipation, so grid-scale oscillations can grow without bound. The velocity filter applies explicit diffusion to drain energy from unresolved modes.

**Formula:**

```
u_filtered = u + alpha * (Lx + Lz) + alpha_y * Ly
```

where Lx, Ly, Lz are discrete Laplacians, `alpha = filter_strength * 0.25`, and `alpha_y = alpha * 0.5` (reduced for stretched grids).

**Wall boundary treatment:**
- u and w (cell-centered in y): skip wall-adjacent cells
- v (face-centered in y): only filter interior faces; wall faces are always v = 0

**CRITICAL: The filter MUST be applied BEFORE the projection step.** The filter introduces small divergence; the subsequent pressure projection removes it.

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `filter_strength` | 0.0 | Filter coefficient (0 = disabled). Range: 0.01--0.05 |
| `filter_interval` | 10 | Apply filter every N steps (0 = disabled) |

**Tuning results** (192x96x192, Re_tau = 180 target):

| strength | interval | Result | Re_tau | Effective nu added |
|----------|----------|--------|--------|-------------------|
| 0.05 | 1 | Stable, 3600+ steps | ~255 | ~55x nu |
| 0.03 | 2 | Stable, 2400+ steps | ~278 | ~16x nu |
| 0.01 | 5 | Blew up at step 1955 | N/A | ~2x nu |
| 0.00 | -- | Blows up at ~1700 steps | N/A | 0 |

The effective viscosity added by the filter is approximately `nu_filter ~ alpha * dx^2 / (dt * filter_interval)`.

### Turbulence Diagnostics

**Health checks** (controlled by `diag_interval`):
- Maximum velocity components (u, v, w) and directional CFL numbers
- Divergence norms (L2, L-infinity)
- Re_tau (friction Reynolds number) and bulk velocity
- Safety cutoff: velocity exceeding `safety_vel_max` triggers SAFETY-VEL abort

**Turbulence presence detection** (plane-averaged statistics):

| Indicator | Turbulent Threshold | Measures |
|-----------|-------------------|----------|
| `u_tau_ratio` | ~ 1.0 | Friction velocity consistency |
| `u_rms_mid` | > 0 | Velocity fluctuations at center |
| `tke_mid` | > 0 | Turbulent kinetic energy at center |
| `max_uv_plus` | > 0.3 | Peak Reynolds shear stress (5 < y+ < 150) |

**Realism validation** (`validate_turbulence_realism()`):
1. Grid resolution: y1+ < 1.0, dx+ < 15, dz+ < 8
2. u_tau consistency: wall-derived vs force-derived agree within tolerance
3. Momentum balance: total shear stress matches theoretical profile
4. Reynolds stress ordering: `<u'u'> > <w'w'> > <v'v'>` in the log layer
5. Reynolds shear stress shape: small at walls, O(1) in the interior

**GPU sync requirement:** `accumulate_statistics()` and `validate_turbulence_realism()` read velocity data on CPU and call `sync_solution_from_gpu()` internally.

### Example Configuration (v13 Recipe)

```ini
# DNS Channel Flow - Re_tau ~ 180 target
# Achieves Re_tau ~ 278 (filter adds effective viscosity)

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
nu = 0.000178571
dp_dx = -1.0

# Time stepping
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

Run with: `./channel --config dns_channel.cfg`

### Troubleshooting

**Blow-up (SAFETY-VEL or NaN):**

| Cause | Symptom | Fix |
|-------|---------|-----|
| CFL too high | Blow-up within first 100 steps | Reduce CFL_max (try 0.10) |
| No filter | Blow-up after trip turns off (~step 1500--2000) | Enable filter (strength=0.03, interval=2) |
| Filter too weak | Blow-up after turbulence develops | Increase strength or decrease interval |
| Trip too strong | Blow-up during trip phase | Reduce trip_amplitude |
| Trip too long | Sustained high velocities | Reduce trip_duration |
| dt_safety too high | Intermittent blow-ups | Reduce to 0.80 or lower |
| Grid too coarse | Blow-up at any settings | Increase resolution |

**Re_tau too high:** The velocity filter adds effective viscosity, shifting Re_tau above the target. Reduce `filter_strength` or increase `filter_interval`, but this may cause blow-up. Achieving Re_tau = 180 exactly may require a less dissipative convective scheme.

**Relaminarization:** Turbulence develops during trip but decays after trip turns off. Increase `trip_w_scale` (promotes 3D vortices), extend `trip_duration`, or increase `trip_amplitude`.

**Known limitations:**
- Filter-limited Re_tau: best achieved ~278 with strength=0.03, interval=2
- Explicit time integration only (no implicit diffusion)
- Second-order spatial discretization only
- Single GPU (no MPI domain decomposition)
- No restart/checkpoint capability

---

## LES Subgrid-Scale Models

Five LES models are available, all GPU-accelerated with fused kernels:

| Model | Config value | Description |
|-------|-------------|-------------|
| Smagorinsky | `smagorinsky` | Classical eddy viscosity, Cs = 0.1 default |
| WALE | `wale` | Wall-adapting, vanishes at walls naturally |
| Vreman | `vreman` | Based on first invariant of velocity gradient |
| Sigma | `sigma` | Based on singular values of gradient tensor |
| Dynamic Smagorinsky | `dynamic_smag` | Germano procedure for local Cs |

### GPU Architecture

Each model has a fused GPU kernel computing velocity gradients and nu_sgs in a single pass:

```
GPU Kernel (one per model):
  for each cell (i,j,k):
    1. Compute 9-component velocity gradient
    2. Compute filter width from mesh (yf)
    3. Compute nu_sgs (model-specific)
    4. Write to nu_t array
```

Key design decisions:
- **Free functions with `#pragma omp declare target`** -- avoids nvc++ implicit `this` transfer
- **`TurbulenceDeviceView` struct** -- raw pointers + scalars, copied to locals before kernel launch
- **Stretched grid support** -- kernels use `yc[]` (cell centers) and `yf[]` (face positions) for non-uniform dy
- **2D/3D unified** -- same kernel handles both via `is2D` flag

The `TurbulenceDeviceView` struct provides the GPU data interface:

```cpp
struct TurbulenceDeviceView {
    double* u_face, *v_face, *w_face;       // Velocity (staggered faces)
    int u_stride, v_stride, w_stride;
    int u_plane_stride, v_plane_stride, w_plane_stride;
    double* nu_t;                            // Turbulence field (cell-centered)
    int cell_stride, cell_plane_stride;
    const double* yf;                        // Face y-positions
    const double* yc;                        // Cell center y-positions
    int Nx, Ny, Nz, Ng;
    double dx, dy, dz;
    int u_total, v_total, w_total;           // Array sizes for map clauses
    int cell_total, yf_total, yc_total;
};
```

### Stretched Grid Filter Width

LES filter width adapts to local cell size:

```cpp
delta = cbrt(dx * dy_local * dz)    // 3D
delta = sqrt(dx * dy_local)         // 2D
```

where `dy_local = yf[j+1] - yf[j]`. Gradient kernels use `yc[]` for central difference spacing.

### Adding a New LES Model

1. **Add model class** in `include/turbulence_les.hpp`:
```cpp
class MyModel : public LESModel {
public:
    double compute_nu_sgs_cell(const double g[9], double delta) const override;
    void update_gpu(const TurbulenceDeviceView* dv) override;
    std::string name() const override { return "MyModel"; }
};
```

2. **Add device kernel** in `src/turbulence_les.cpp`:
```cpp
#pragma omp declare target
inline double my_model_nu_sgs(const double g[9], double param, double delta) {
    return ...;
}
#pragma omp end declare target
```

3. **Implement `update_gpu()`** following the Smagorinsky pattern -- copy all values to locals, use `map(present: ptr[0:size])` with array sections.

4. **Register** in factory (`src/config.cpp`) and add to `TurbulenceModelType` enum in `include/config.hpp`.

5. **Add source** to `nn_cfd_core` in `CMakeLists.txt` and write tests.

### Key Source Files

| File | Purpose |
|------|---------|
| `include/turbulence_les.hpp` | LES model class hierarchy |
| `src/turbulence_les.cpp` | GPU kernels for all 5 models |
| `include/turbulence_model.hpp` | `TurbulenceDeviceView` struct |
| `tests/bench_les_ibm_gpu.cpp` | 3D benchmark |

---

## Immersed Boundary Method (IBM)

### Direct Forcing with Pre-computed Weights

IBM uses pre-computed weight arrays instead of runtime geometry evaluation:

```
Classification (CPU, once at init):
  body.phi(x,y,z) -> IBMCellType -> weight

Weight values:
  Fluid:   weight = 1.0 (pass-through)
  Solid:   weight = 0.0 (zero velocity)
  Forcing: weight = |phi|/band_width (smooth transition)
```

Three sets of weight arrays exist at staggered locations:
- `weight_u_` at u-face locations
- `weight_v_` at v-face locations
- `weight_w_` at w-face locations (3D only)
- `solid_mask_cell_` at cell centers (for Poisson RHS masking)

### Available Bodies

- `CylinderBody` -- infinite cylinder with signed distance function
- `SphereBody` -- sphere with signed distance function

### GPU Design

Weight arrays are mapped to GPU at initialization via `target enter data map(to:...)` and use `map(present:...)` during compute. The forcing operation is element-wise multiplication:

```
During stepping:
  u[i] *= weight_u[i]
  v[i] *= weight_v[i]
  w[i] *= weight_w[i]
```

Poisson RHS masking runs as a GPU kernel using `solid_mask_cell_` -- no CPU sync required.

### Solver Integration

IBM forcing is applied twice per step:

```
compute u* (predictor)
IBM: u* *= weight              <- apply_forcing_device()
compute RHS = div(u*)/dt
IBM: RHS *= solid_mask         <- mask_rhs_device()
solve nabla^2 p' = RHS
correct u = u* - dt*grad(p')
IBM: u *= weight               <- apply_forcing_device()
```

### GPU Mapping Lifecycle

```
Constructor:
  classify_cells()           <- CPU: evaluate phi at all face locations
  compute_weights()          <- CPU: convert cell types to weight arrays

solver.set_ibm_forcing():
  ibm_->map_to_gpu()         <- target enter data map(to: weights, mask)

During stepping:
  apply_forcing_device()     <- target teams: u *= weight (map present)
  mask_rhs_device()          <- target teams: rhs *= mask (map present)

Cleanup:
  ibm_->unmap_from_gpu()     <- target exit data map(delete: ...)
```

For moving bodies, call `classify_cells()` + `compute_weights()` again, then `target update to(...)` to refresh GPU arrays.

---

## GPU Performance

### 128-cubed Profiling (Poisson Solver Comparison)

Profiled with nsys on NVIDIA GPU, 128x128x128 grid, 10 steps, no I/O:

| Solver Mode | ms/step | Speedup | Notes |
|-------------|---------|---------|-------|
| MG (convergence) | 24.1 | 1.0x | Baseline, 10 iterations avg |
| MG+Graph (fixed) | 4.9 | 4.9x | CUDA Graph eliminates dispatch overhead |
| FFT (all-periodic) | 1.7 | 14x | Fastest, periodic BCs only |

**Detailed configuration matrix (128-cubed):**

| BCs | Poisson | Graph | ms/step | Mcells/s |
|-----|---------|-------|---------|----------|
| Periodic (PPP) | MG | No | 20.4 | 103 |
| Periodic (PPP) | MG | Yes | 4.2 | 500 |
| Channel (PWP) | MG | No | 20.5 | 103 |
| Channel (PWP) | MG | Yes | 4.6 | 452 |
| Duct (PWW) | MG | No | 20.4 | 103 |
| Duct (PWW) | MG | Yes | 4.6 | 454 |
| Periodic (PPP) | FFT | N/A | 1.7 | 1267 |

### CUDA Graph Speedup

| Configuration | MG (ms) | MG+Graph (ms) | Speedup |
|--------------|---------|---------------|---------|
| All-periodic | 20.4 | 4.2 | 4.9x |
| Channel | 20.5 | 4.6 | 4.4x |
| Duct | 20.4 | 4.6 | 4.4x |

Without CUDA Graphs, each MG V-cycle requires ~100 separate OpenMP target launches at ~50 us overhead each. With graphs, a single launch per V-cycle reduces overhead from ~50 ms to ~0.1 ms for 10 V-cycles.

CUDA Graphs require `fixed_cycles > 0` (fixed V-cycle count). They are disabled for convergence-based solving, recycling inflow, and semi-coarsening.

### Channel MG+Graph Timing Breakdown (128-cubed, 4.91 ms/step)

| Phase | ms/step | % of step |
|-------|---------|-----------|
| poisson_solve | 3.40 | 69.2% |
| apply_velocity_bc | 0.23 | 4.7% |
| velocity_copy | 0.23 | 4.7% |
| predictor_step | 0.23 | 4.6% |
| convection | 0.15 | 3.0% |
| velocity_correction | 0.13 | 2.7% |
| diffusion | 0.10 | 2.1% |
| divergence | 0.03 | 0.7% |
| nu_eff_computation | 0.02 | 0.5% |
| **Total accounted** | **4.52** | **92%** |

### LES + IBM Benchmarks (RTX 6000, Turing CC 7.5)

**128x64x128 (1.0M cells), MG Poisson, 20 steps:**

| Configuration | ms/step | Mcells/s | Poisson (ms) | Turb (ms) |
|---------------|---------|----------|--------------|-----------|
| Laminar baseline | 12.70 | 82.6 | 9.20 | -- |
| Smagorinsky LES | 22.28 | 47.1 | 18.95 | 0.77 |
| Smagorinsky LES + IBM | 20.16 | 52.0 | 16.59 | 0.71 |

**256x128x256 (8.4M cells), MG Poisson, 10 steps:**

| Configuration | ms/step | Mcells/s | Poisson (ms) | Turb (ms) |
|---------------|---------|----------|--------------|-----------|
| Laminar baseline | 89.44 | 93.8 | 64.18 | -- |
| Smagorinsky LES | 145.12 | 57.8 | 120.25 | 5.81 |
| Smagorinsky LES + IBM | 143.06 | 58.6 | 118.70 | 5.45 |

**256-cubed LES+IBM timing breakdown:**

| Phase | ms/step | % of step |
|-------|---------|-----------|
| poisson_solve | 118.70 | 82.5% |
| turbulence_update | 5.45 | 3.8% |
| convective_term | 4.71 | 3.3% |
| diffusive_term | 4.61 | 3.2% |
| velocity_correction | 1.52 | 1.1% |
| divergence | 1.15 | 0.8% |
| IBM forcing | <0.5 | <0.3% |
| RHS masking | <0.5 | <0.3% |

### Key Observations

1. **Poisson solver dominates** (~70--83% of step time). Use FFT when BCs allow; use MG+Graph for wall BCs.
2. **CUDA Graphs provide 4.4--4.9x speedup** for MG by eliminating OpenMP target dispatch overhead.
3. **IBM overhead is negligible** (<0.3%) -- element-wise weight multiplication is trivially fast.
4. **LES SGS overhead is small** (~4%) -- fused gradient + nu_sgs kernel avoids intermediate storage.
5. **CPU sync was the old IBM bottleneck** -- before GPU RHS masking, IBM was 2.1x slower due to `target update from` + CPU loop + `target update to` (64 MB transfer per step at 8.4M cells).

### Running Benchmarks

```bash
# Build (GPU required)
cd build_gpu && make bench_les_ibm_gpu

# Run with defaults (128x64x128, 20 steps)
./bench_les_ibm_gpu

# Custom grid and steps
./bench_les_ibm_gpu 256 128 256 10

# Profile with nsys
cd build && make profile_comprehensive
nsys profile --stats=true -t cuda,nvtx -o profile ./profile_comprehensive
nsys stats --report nvtx_pushpop_sum profile.nsys-rep
```

---

## Common Pitfalls

### GPU / OpenMP Target

- **`map(present:)` requires array sections**, not bare pointer names. `map(present: u)` causes silent failures with nvc++; use `map(present: u[0:u_sz])`.
- **All accessed pointers must appear in the map clause.** Missing a pointer (e.g., w in 3D) causes segfaults.
- **Virtual functions cannot run on GPU.** `IBMBody::phi()` is virtual -- all geometry evaluation must happen at init time on CPU with results stored in mapped arrays.
- **Never sync GPU data during stepping** for IBM or LES operations. Use GPU-resident kernels and `map(present:...)`.
- **Set `OMP_TARGET_OFFLOAD=MANDATORY`** for GPU test runs to prevent silent CPU fallback.

### IBM

- **Call `map_to_gpu()` before stepping.** Either set IBM before `initialize_gpu_buffers()`, or `set_ibm_forcing()` will auto-detect `gpu_ready_` and map.
- **Poisson RHS masking must be a GPU kernel** (`mask_rhs_device()`), not a CPU loop with `target update from`/`to`.

### DNS

- **Filter must be applied before the projection step**, not after. Post-filter divergence is cleaned by the subsequent Poisson solve.
- **`trip_duration` is physical time**, not step count or friction time units.
- **`CFL_y` always uses `CFL_max`** (no relaxation) -- relaxed y-CFL caused blow-ups in earlier versions.
- **`accumulate_statistics()` reads CPU data** -- it calls `sync_solution_from_gpu()` internally, but custom diagnostic code must sync manually.
- **CUDA Graphs must be disabled for recycling inflow** (`poisson_use_vcycle_graph = false`).

### General

- **`perf_mode`** suppresses per-step output; only prints every `output_freq` steps after step 50.
- **Config comments** (e.g., `central  # comment`) get parsed as part of the value. Use clean configs without inline comments.
- **Staggered grid indexing**: u-stride = `(Nx+1) + 2*Ng`, v has `(Ny+1)` rows, w-stride = `Nx + 2*Ng`. Always use `mesh_->index()` helpers.

---

## References

- Moser, R. D., Kim, J., & Mansour, N. N. "Direct numerical simulation of turbulent channel flow up to Re_tau = 590." *Physics of Fluids* 11.4 (1999): 943--945.
- Kim, J., Moin, P., & Moser, R. "Turbulence statistics in fully developed channel flow at low Reynolds number." *J. Fluid Mech.* 177 (1987): 133--166.
