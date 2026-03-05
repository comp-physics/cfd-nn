# Getting Started

This guide covers building, running, and configuring the CFD-NN solver.

## Prerequisites

- **C++17 compiler**: nvc++ (NVHPC SDK) for GPU builds, or g++/clang++ for CPU-only
- **CMake** >= 3.18
- **NVIDIA GPU** (optional): Compute capability 7.0+ (V100, A100, H100, H200)
- **Python 3** (optional): numpy, matplotlib for post-processing

On HPC systems with modules:
```bash
module load nvhpc cmake
```

## Building

Use `make.sh` for the simplest experience:

```bash
./make.sh gpu              # GPU build (Release) — primary target
./make.sh cpu              # CPU-only build
./make.sh gpu --debug      # GPU build with debug symbols
./make.sh clean            # Remove build artifacts
```

### Build Options

| Flag | Description |
|------|-------------|
| `--debug` | Debug build (default: Release) |
| `--rebuild` | Force clean rebuild |
| `--jobs N` | Parallel compile jobs (default: auto) |
| `--hdf5` | Enable HDF5 checkpoint/restart |
| `--mpi` | Enable MPI domain decomposition |
| `--hypre` | Enable HYPRE Poisson solver (GPU only) |
| `--gpu-cc N` | GPU compute capability (80=A100, 90=H200) |
| `--build-dir DIR` | Custom build directory |
| `--all-features` | Enable HDF5 + MPI (+ HYPRE for GPU) |

### Common Build Configurations

```bash
# Development (fast iteration)
./make.sh cpu --debug

# Production GPU (H200)
./make.sh gpu --gpu-cc 90

# Full-featured GPU build
./make.sh gpu --all-features --gpu-cc 90

# Separate build directory for experiments
./make.sh gpu --build-dir build_experiment
```

### Manual CMake Build

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_CXX_COMPILER=nvc++ \
         -DUSE_GPU_OFFLOAD=ON \
         -DGPU_CC=90 \
         -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Running Simulations

### Using `run.sh` (Recommended)

The `run.sh` wrapper builds and runs in one command:

```bash
# Build GPU and run a simulation
./run.sh gpu --config examples/01_laminar_channel/poiseuille.cfg

# CPU debug build + run
./run.sh cpu --debug --config examples/10_les_channel/les_retau590_wale.cfg

# Build only (no run)
./run.sh gpu --build-only

# Run only (skip build, use existing binary)
./run.sh --run-only --config examples/01_laminar_channel/poiseuille.cfg

# Dry run (show what would execute)
./run.sh gpu --dry-run --config examples/01_laminar_channel/poiseuille.cfg

# Submit as SLURM job
./run.sh gpu --slurm --config examples/07_unsteady_developing_channel/dns_retau180.cfg

# Override config parameters from CLI
./run.sh gpu --config examples/01_laminar_channel/poiseuille.cfg -- --max_steps 5000 --nu 0.005
```

### Running Directly

```bash
cd build
./channel --config ../examples/01_laminar_channel/poiseuille.cfg
```

All executables require `--config file.cfg`. Positional arguments are **not** supported.

### Available Executables

| Binary | Description |
|--------|-------------|
| `channel` | Channel flow (2D/3D), DNS, LES, RANS |
| `duct` | Square duct flow (3D) |
| `taylor_green_3d` | Taylor-Green vortex (3D, periodic) |
| `cylinder` | Flow past cylinder (IBM) |
| `airfoil` | Flow past airfoil (IBM) |

### CLI Overrides

Any config parameter can be overridden from the command line:

```bash
./channel --config base.cfg --Nx 128 --Ny 256 --nu 0.001 --max_steps 50000
```

## Config File Reference

Config files use INI-style `key = value` format. Lines starting with `#` are comments.

### Grid

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Nx` | 64 | Grid cells in x |
| `Ny` | 64 | Grid cells in y |
| `Nz` | 1 | Grid cells in z (1 = 2D) |
| `x_min`, `x_max` | 0, 2*pi | Domain extent in x |
| `y_min`, `y_max` | -1, 1 | Domain extent in y |
| `z_min`, `z_max` | 0, 1 | Domain extent in z |
| `stretch_y` | false | Enable tanh grid stretching in y |
| `stretch_beta` | 2.0 | Stretching parameter (higher = more clustering at walls) |
| `stretch_z` | false | Enable stretching in z |
| `stretch_beta_z` | 2.0 | Z-stretching parameter |

### Physics

Specify any 2 of `Re`, `nu`, `dp_dx` — the third is computed automatically.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Re` | 1000 | Reynolds number (based on half-height and bulk velocity) |
| `nu` | 0.001 | Kinematic viscosity |
| `dp_dx` | -1.0 | Streamwise pressure gradient (drives the flow) |
| `rho` | 1.0 | Density |

### Time Stepping

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dt` | 0.001 | Initial time step |
| `adaptive_dt` | true | Enable CFL-based adaptive time stepping |
| `CFL_max` | 0.5 | Max CFL number (strict, used for y-direction) |
| `CFL_xz` | -1 | CFL for x/z directions (-1 = use CFL_max) |
| `dt_safety` | 1.0 | Safety factor on adaptive dt (0.85 typical for DNS) |
| `max_steps` | 10000 | Maximum number of time steps |
| `T_final` | -1 | Final physical time (-1 = use max_steps instead) |
| `tol` | 1e-6 | Steady-state convergence tolerance (0 = disable) |
| `integrator` | euler | Time integrator: `euler`, `rk2`, `rk3` |

### Turbulence Model

| Parameter | Default | Description |
|-----------|---------|-------------|
| `turb_model` | none | Model selection (see table below) |
| `nu_t_max` | 1.0 | Maximum eddy viscosity clipping |
| `nn_preset` | (empty) | NN model preset name |
| `nn_weights_path` | (empty) | Path to NN weight file |
| `nn_scaling_path` | (empty) | Path to NN scaling file |

Available turbulence models:

| Value | Type | Description |
|-------|------|-------------|
| `none` | — | Laminar (no model) |
| `baseline` | RANS | Algebraic mixing length (Van Driest) |
| `gep` | RANS | Gene Expression Programming algebraic |
| `nn_mlp` | RANS | Neural network scalar eddy viscosity |
| `nn_tbnn` | RANS | Tensor basis neural network |
| `k_omega` | RANS | Standard k-omega (Wilcox 1988) |
| `sst` | RANS | SST k-omega transport |
| `earsm_wj` | RANS | SST + Wallin-Johansson EARSM |
| `earsm_gs` | RANS | SST + Gatski-Speziale EARSM |
| `earsm_pope` | RANS | SST + Pope quadratic |
| `smagorinsky` | LES | Static Smagorinsky (Cs=0.17) |
| `dynamic_smagorinsky` | LES | Dynamic Smagorinsky (Germano) |
| `wale` | LES | Wall-Adapting Local Eddy-viscosity |
| `vreman` | LES | Vreman model |
| `sigma` | LES | Sigma model |

### Numerical Schemes

| Parameter | Default | Description |
|-----------|---------|-------------|
| `scheme` | central | Convective scheme: `central`, `upwind`, `skew`, `upwind2` |
| `space_order` | 2 | Spatial discretization order (2 or 4) |
| `simulation_mode` | steady | Mode: `steady` or `unsteady` |
| `mode` | steady | Alias for `simulation_mode` |

### Velocity Filter

| Parameter | Default | Description |
|-----------|---------|-------------|
| `filter_strength` | 0 | Filter coefficient (0=off, 0.01-0.05 typical) |
| `filter_interval` | 10 | Apply filter every N steps (0=off) |

### Poisson Solver

| Parameter | Default | Description |
|-----------|---------|-------------|
| `poisson_solver` | auto | Backend: `auto`, `fft`, `fft2d`, `fft1d`, `hypre`, `mg`, `fft_mpi` |
| `poisson_tol` | 1e-6 | Absolute tolerance (legacy, see below) |
| `poisson_tol_rhs` | 1e-6 | RHS-relative tolerance: norm(r)/norm(b) |
| `poisson_tol_rel` | 1e-3 | Initial-residual relative tolerance |
| `poisson_tol_abs` | 0 | Absolute tolerance on L-inf residual (0=disabled) |
| `poisson_max_vcycles` | 20 | Maximum MG V-cycles |
| `poisson_fixed_cycles` | 8 | Fixed V-cycle count (bypass convergence checks) |
| `poisson_adaptive_cycles` | true | Enable adaptive cycle count |
| `poisson_check_after` | 4 | Check residual after this many fixed cycles |
| `poisson_check_interval` | 3 | Check convergence every N V-cycles |
| `poisson_nu1` | 0 | Pre-smoothing sweeps (0=auto) |
| `poisson_nu2` | 0 | Post-smoothing sweeps (0=auto) |
| `poisson_chebyshev_degree` | 4 | Chebyshev smoother polynomial degree |
| `poisson_use_vcycle_graph` | true | Enable CUDA Graph V-cycle optimization |

### Output

| Parameter | Default | Description |
|-----------|---------|-------------|
| `output_dir` | output/ | Output directory |
| `output_freq` | 100 | Console output frequency (steps) |
| `num_snapshots` | 10 | Number of VTK snapshots |
| `verbose` | true | Enable verbose output |
| `write_fields` | true | Write VTK field files |
| `vtk_binary` | true | Binary VTK format |
| `postprocess` | true | Run post-processing (Poiseuille comparison, etc.) |
| `diag_interval` | 1 | Expensive diagnostics frequency |

### Performance

| Parameter | Default | Description |
|-----------|---------|-------------|
| `perf_mode` | false | Reduce diagnostics for production speed |
| `benchmark` | false | Benchmark mode (optimized timing) |
| `warmup_steps` | 0 | Steps before resetting timers |
| `gpu_only_mode` | false | Strict GPU-only (no CPU fallbacks) |

### Trip Forcing (DNS Transition)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trip_enabled` | false | Enable trip region forcing |
| `trip_amplitude` | 3.0 | Trip forcing amplitude |
| `trip_duration` | 2.0 | Duration in physical time |
| `trip_ramp_off_start` | 1.5 | Start of ramp-off in physical time |
| `trip_x_start` | -1 | Trip region start x (-1=auto) |
| `trip_x_end` | -1 | Trip region end x (-1=auto) |
| `trip_n_modes_z` | 8 | Spanwise Fourier modes |
| `trip_force_w` | true | Also force w-velocity |
| `trip_w_scale` | 1.0 | W-forcing scale factor |

### Recycling Inflow

| Parameter | Default | Description |
|-----------|---------|-------------|
| `recycling_inflow` | false | Enable recycling inflow BC |
| `recycle_x` | -1 | Recycle plane x-location (-1=auto) |
| `recycle_shift_z` | -1 | Spanwise shift (-1=auto: Nz/4) |
| `recycle_shift_interval` | 100 | Steps between shift updates |
| `recycle_filter_tau` | -1 | Temporal filter timescale (-1=off) |
| `recycle_fringe_length` | -1 | Fringe zone length (-1=auto) |
| `recycle_target_bulk_u` | -1 | Target bulk velocity (-1=from IC) |

### Projection Health

| Parameter | Default | Description |
|-----------|---------|-------------|
| `projection_watchdog` | true | Enable projection health monitoring |
| `div_threshold` | 1e-5 | Alert if divergence exceeds this |
| `adaptive_projection` | true | Increase cycles when divergence is high |
| `div_target` | 1e-4 | Target max divergence |
| `projection_max_cycles` | 60 | Max cycles for adaptive projection |

## Example Workflows

### 1. Laminar Validation (First Run)

```bash
./run.sh cpu --config examples/01_laminar_channel/poiseuille.cfg
```

Validates against analytical Poiseuille solution. Should complete in ~30 seconds.

### 2. Turbulent Channel (RANS)

```bash
./run.sh gpu --config examples/05_channel_retau180_sst/channel.cfg
```

SST k-omega at Re_tau=180. Runs ~20 minutes on GPU.

### 3. 3D DNS Channel Flow

```bash
./run.sh gpu --config examples/07_unsteady_developing_channel/dns_retau180.cfg
```

Full 3D DNS at Re_tau~180 with trip forcing and velocity filter. Requires GPU for practical runtimes.

### 4. LES Channel Flow

```bash
./run.sh gpu --config examples/10_les_channel/les_retau590_wale.cfg
```

LES with WALE SGS model at Re_tau=590 on 64^3 grid.

### 5. Cylinder Flow (IBM)

```bash
./run.sh gpu --config examples/11_cylinder_flow/cylinder_re3900_les.cfg
```

Flow past cylinder using immersed boundary method with LES.

### 6. SLURM Cluster Submission

```bash
./run.sh gpu --slurm --config examples/07_unsteady_developing_channel/dns_retau180.cfg
```

Builds locally and submits a SLURM job with appropriate GPU settings.

## Running Tests

```bash
cd build
ctest --output-on-failure          # All tests
ctest -L fast --output-on-failure  # Fast tests only (<30s)
ctest -L gpu --output-on-failure   # GPU-specific tests
ctest -LE slow --output-on-failure # Skip slow tests
```

Or via make targets:
```bash
make check-fast     # Fast tests
make check-quick    # All except slow
```

## Viewing Results

**VTK files** (ParaView):
```bash
paraview output/velocity_final.vtk
```

**Profile data** (text):
```bash
cat output/velocity_profile.dat
```

**Python post-processing**:
```bash
pip install numpy matplotlib scipy
python examples/01_laminar_channel/analyze.py
```

## Troubleshooting

### `nvc++ not found`
Load the NVHPC module: `module load nvhpc`

### GPU build but running on CPU
Set `OMP_TARGET_OFFLOAD=MANDATORY` to force GPU execution (will error if no GPU).

### Simulation blows up (NaN)
- Reduce `CFL_max` (try 0.15-0.3)
- Enable velocity filter: `filter_strength = 0.03`, `filter_interval = 2`
- Use `integrator = rk3` and `scheme = skew` for DNS/LES
- Add `dt_safety = 0.85` for extra headroom

### Poisson solver slow
- Use `poisson_solver = fft` for periodic x/z (fastest)
- Use `poisson_solver = mg` with `poisson_fixed_cycles = 8` for walls
- Enable `perf_mode = true` to reduce convergence check overhead

### Config parsing issues
Config comments must be on their own line. Do NOT put inline comments:
```ini
# Correct:
# Use central scheme
scheme = central

# WRONG (parsed as "central  # comment"):
scheme = central  # comment
```
