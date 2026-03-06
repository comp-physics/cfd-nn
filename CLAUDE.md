# CLAUDE.md — CFD-NN Project

## What This Project Is

An incompressible Navier-Stokes solver in C++17 with pluggable turbulence closures (algebraic, transport, EARSM, neural network) and GPU acceleration via OpenMP target offload. Uses a fractional-step projection method on a staggered MAC grid.

Primary binary: `build/channel`. Config files are INI-style `.cfg` passed via `--config file.cfg`.

## Build

GPU build is the primary development target (NVHPC compiler):

```bash
./run.sh gpu --build-only              # GPU build (Release)
./run.sh gpu --debug --build-only      # GPU build (Debug)
./run.sh cpu --build-only              # CPU-only build
./run.sh gpu --config file.cfg         # Build + run
```

Or manually:

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_CXX_COMPILER=nvc++ -DUSE_GPU_OFFLOAD=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

GPU compute capability: `-DGPU_CC=90` for H200, `-DGPU_CC=80` for A100.

## Tests

```bash
cd build && ctest --output-on-failure     # All tests
make check-fast                           # Fast tests only (<30s each)
make check-quick                          # All except slow tests
ctest -L gpu                              # GPU-specific tests
```

**Before pushing**: Run `./test_before_ci.sh` (CPU CI) and `./test_before_ci_gpu.sh` (GPU CI).

CI runs 4 CPU configurations (Ubuntu + macOS) x (Debug + Release) plus GPU validation.

## Project Layout

```
app/            Main executables (main_channel.cpp, main_duct.cpp, etc.)
include/        Headers (.hpp)
src/            Implementation (.cpp)
tests/          Unit/integration tests + benchmarks
examples/       Config files organized by flow case
data/models/    Curated NN model weights
scripts/        Utility scripts
```

Key source files:
- `solver.cpp` — Core solver (projection method, stepping, diagnostics)
- `solver_time.cpp` — Time integration (Euler, RK2, RK3)
- `solver_time_kernels_*.cpp` — GPU compute kernels (convection, diffusion, projection)
- `solver_recycling.cpp` — Recycling inflow BC
- `poisson_solver_multigrid.cpp` — Geometric multigrid
- `poisson_solver_fft*.cpp` — FFT Poisson solvers (cuFFT)
- `turbulence_les.cpp` — LES SGS models with fused GPU kernels (Smagorinsky, WALE, Vreman, Sigma, Dynamic Smagorinsky)
- `turbulence_*.cpp` — Other turbulence closure implementations (RANS, EARSM, NN)
- `ibm_forcing.cpp` — Immersed boundary method (direct forcing, GPU weight arrays)
- `ibm_geometry.cpp` — IBM body definitions (cylinder, sphere, signed distance)
- `config.cpp` — Config file parser and CLI args

## Coding Conventions

### Naming
- **snake_case** everywhere: functions, variables, file names
- **PascalCase** for types/classes only: `RANSSolver`, `VelocityBC`, `TurbulenceModel`
- **UPPER_CASE** for macros/defines only: `USE_GPU_OFFLOAD`, `USE_HYPRE`
- Member variables use trailing underscore: `velocity_`, `current_dt_`, `nu_`
- Namespace: `nncfd`

### Style
- C++17 standard, no extensions
- 4-space indentation (never tabs)
- Opening braces on same line: `for (...) {`, `class Foo {`
- `#pragma once` for header guards
- `///` Doxygen-style doc comments on public API
- `//` inline comments with space after slashes
- Large section separators in source files: `// ====...====`
- `const` correctness: mark read-only references and non-mutating methods
- RAII everywhere: `std::unique_ptr` for ownership, no raw `new`/`delete`
- GPU buffers use `OmpDeviceBuffer` wrapper
- Enum values are PascalCase: `ConvectiveScheme::Skew`, `PoissonSolverType::Auto`
- Templates are minimal — used for `std::function` callbacks, containers, smart pointers only; no generic template classes

### Include Ordering
1. Local project headers: `"solver.hpp"`, `"mesh.hpp"` (double quotes)
2. Standard library headers: `<vector>`, `<cmath>`, `<iostream>` (angle brackets)
3. Conditional/platform headers (separated by blank line): `#ifdef USE_GPU_OFFLOAD` → `<omp.h>`

### Namespace Conventions
- Main namespace: `nncfd`
- Nested namespaces for domain separation: `nncfd::gpu_kernels`, `nncfd::numerics`
- Anonymous namespaces for file-local helpers (in `.cpp` files only)
- `using namespace nncfd;` allowed in `app/` entry points — never in headers

### Error Handling
- `throw std::runtime_error(...)` for file I/O and critical failures
- Parse warnings with safe defaults (e.g., invalid `space_order` falls back to 2 with stderr warning)
- `assert()` for development-time invariants

### What NOT to Do
- Do not add `#ifdef __APPLE__` or platform-specific tolerances in tests
- Do not relax test tolerances to hide numerical issues — fix root causes
- Do not compare floats with `==` — use tolerance-based checks
- Do not check iteration counts in tests — check residual convergence
- Do not add unnecessary abstractions, factories, or design patterns to numerical code
- Do not create separate CPU and GPU compute paths (see GPU rules below)
- Do not suppress compiler warnings with flags — fix them

## Adding New Features

Follow the existing layered pattern — every major component has a header, implementation, config integration, and tests:

### New Turbulence Model Example
1. **Header**: `include/turbulence_new.hpp` — inherit from `TurbulenceModel`, override `update()`, `name()`, `is_gpu_ready()`
2. **Implementation**: `src/turbulence_new.cpp` — compute `nu_t` (and optionally `tau_ij` for anisotropic models)
3. **Config enum**: Add to `TurbulenceModelType` in `include/config.hpp`
4. **Factory**: Add case to model creation in `src/config.cpp`
5. **CMakeLists.txt**: Add source file to `nn_cfd_core` library
6. **Tests**: Add `tests/test_new_model.cpp`, register in CMake with appropriate labels

### New Poisson Solver Example
Same pattern: header in `include/`, impl in `src/`, enum in `PoissonSolverType`, factory in solver init, CMake registration.

### General Rules
- Follow whatever the closest existing feature does for file naming, class structure, and integration points
- Every new `.cpp` gets added to `nn_cfd_core` in CMakeLists.txt
- Every new feature needs at least one test

## Solver Step Pipeline

The `RANSSolver::step()` method implements a fractional-step projection method in this order:

1. **Turbulence update** — `turb_model_->advance_turbulence()` then `turb_model_->update()` (compute `nu_t`). LES models use fused GPU kernels via `update_gpu(TurbulenceDeviceView*)`.
2. **Effective viscosity** — `nu_eff_ = nu + nu_t`
3. **Convective + diffusive terms** — computed from current velocity
4. **Predictor** — `u* = u^n + dt·(-conv + diff + body_force)` (explicit)
5. **IBM forcing (if enabled)** — `apply_forcing_device()`: multiply u* by pre-computed weight arrays (0=solid, 1=fluid)
6. **Recycling inlet** (if enabled) — apply recycling BC, correct divergence
7. **IBM RHS masking (if enabled)** — `mask_rhs_device()`: zero Poisson RHS at solid cells (GPU kernel, no CPU sync)
8. **Pressure Poisson** — `∇²p' = (1/dt)·∇·u*`, warm-started from previous `p'`
9. **Velocity correction** — `u^{n+1} = u* - dt·∇p'`
10. **IBM re-forcing (if enabled)** — re-apply weight multiplication to corrected velocity
11. **Boundary conditions** — periodic halos, no-slip walls
12. **Recycle plane extraction** (if enabled) — save for next step
13. **Residual** — `max|u^{n+1} - u^n|` via GPU reduction

For RK2/RK3, steps 3-7 repeat per stage with SSP weights.

The velocity filter (when enabled) is applied BEFORE step 1 in the application's time loop, not inside `step()` itself.

## GPU Rules (Critical)

The codebase uses OpenMP target offload for GPU acceleration. The `#pragma omp target` directives are present in the source unconditionally — they are silently ignored in CPU builds (no `#ifdef` needed around them). The build system suppresses unknown-pragma warnings for CPU builds automatically.

### Single Code Path
There MUST be one arithmetic path, not two. Never write:
```cpp
// WRONG — divergent CPU/GPU paths
#ifdef USE_GPU_OFFLOAD
  // GPU computation
#else
  // CPU computation with different arithmetic
#endif
```

The only valid uses of `#ifdef USE_GPU_OFFLOAD` are:
- Including `<omp.h>` header
- GPU buffer initialization/cleanup (`target enter data` / `target exit data`)
- Calling `omp_get_num_devices()` for runtime GPU detection
- Guard around GPU-only source files in CMakeLists.txt

### No Unnecessary GPU-CPU Sync
GPU runs must NEVER sync memory to CPU except during:
- I/O dumps (VTK output, profile writing)
- Beginning of simulation (initialization)
- End of simulation (final diagnostics)

Functions that read field data on CPU (like `accumulate_statistics()`, `validate_turbulence_realism()`) must call `sync_solution_from_gpu()` first. This is a common bug source.

### Pragma Discipline
- Do not add `#pragma omp target` or `#pragma omp teams` where not needed
- Do not add `#ifdef USE_GPU_OFFLOAD` guards around pragmas — they fall away naturally in CPU builds
- Use `map(present: ptr[0:size])` with **array sections** for data already on GPU — bare pointer names (e.g., `map(present: ptr)`) do not work with nvc++ and cause silent failures or NaN
- Never use `map(to: ...)` or `map(from: ...)` during compute — data is persistently mapped

### nvc++ Workarounds
The NVHPC compiler transfers `this` for every GPU kernel on a member function. Avoid this by:
- Writing GPU kernels as **free functions** (not methods), passing all data as arguments
- Copying member parameters to locals before kernel loops:
```cpp
// Avoids implicit this transfer to GPU
const int Nx = Nx_p, Ny = Ny_p, Nz_eff = Nz_eff_p, Ng = Ng_p;
```
- Using `is_device_ptr()` clause with pointers obtained from `omp_get_mapped_ptr()`

### GPU Diagnostics Without Sync
These return scalar results via GPU reductions (no field transfer):
- `compute_kinetic_energy_device()`, `compute_max_velocity_device()`
- `compute_divergence_linf_device()`, `compute_max_conv_device()`

Use these instead of syncing entire fields to CPU for a single number.

## Physics Integrity Rules

Changes must not violate:
- **Conservation**: Body force integral = wall shear integral (momentum balance)
- **Incompressibility**: div(u) = 0 after projection (machine precision)
- **Staggered grid consistency**: u at x-faces, v at y-faces, w at z-faces, scalars at cell centers
- **D·G = L compatibility**: Divergence and gradient operators must compose to exact Laplacian on the staggered grid (uses precomputed y-metric arrays `dyv`, `dyc`, `yLap_*` for stretched grids)
- **BC compatibility**: Filter and other operations must skip wall-adjacent cells appropriately
- **Symmetry**: Channel flow must satisfy u(y) = u(-y) to machine precision

### Velocity Filter
The velocity filter applies a 3D Laplacian smoothing and MUST be applied BEFORE the pressure projection step (not after), so the projection cleans up any divergence the filter introduces.

### Trip Forcing
`trip_duration` and `trip_ramp_off_start` are in **physical time** (compared vs `current_time_`), not in t* or step-count units.

## Test Guidelines

- Tests must pass in BOTH Debug and Release builds
- No platform-specific tolerances or `#ifdef` in tests
- Check convergence via residuals, not iteration counts
- Always initialize: `set_body_force()`, `initialize_uniform()`, set turbulence model before stepping
- Tests are labeled: `fast` (<30s), `medium` (30-120s), `slow` (>120s), `gpu`, `hypre`, `fft`
- Each test must be independent — no reliance on execution order

## Staggered Grid (MAC)

Velocities live at cell faces, scalars at cell centers:
- `u(i+1/2, j, k)` at x-faces
- `v(i, j+1/2, k)` at y-faces
- `w(i, j, k+1/2)` at z-faces
- `p(i, j, k)` at cell centers

This arrangement guarantees pressure-velocity coupling without checkerboard modes. The divergence and gradient operators must compose to the exact discrete Laplacian — this is enforced via precomputed y-metric arrays (`dyv`, `dyc`, `yLap_*`) for stretched grids.

Ghost cell layers (default 1) surround the domain for BC implementation. Interior ranges: `mesh->i_begin()` to `mesh->i_end()`.

## Recycling Inflow BC

Implements Lund et al. (1998) for spatially-developing turbulent channel flow:
1. **Extract** velocity at recycle plane (`recycle_x`)
2. **Shift** in spanwise direction for decorrelation (`recycle_shift_z`)
3. **Filter** temporally with optional AR1 smoothing (`recycle_filter_tau`)
4. **Correct** mass flux to match target bulk velocity
5. **Remove** transverse means (enforce `mean(v)=0`, `mean(w)=0`)
6. **Correct** divergence at inlet slab
7. **Blend** via fringe zone between recycled and interior flow

Key gotchas:
- Recycling buffers are GPU-resident; plane extraction uses GPU kernels
- CUDA Graphs must be disabled when recycling is active
- Mass correction preserves fluctuations while scaling the mean

## Config Files

Config uses `--config file.cfg` flag (positional args are NOT supported):
```bash
./channel --config examples/01_laminar_channel/poiseuille.cfg
```

INI-style key-value format. CLI args override config file values. Physics parameters: specify any 2 of `Re`, `nu`, `dp_dx` — the 3rd is computed.

## Stable DNS Recipe (v13)

For channel flow DNS at Re_tau ~ 180:
```ini
Nx = 192
Ny = 96
Nz = 192
CFL_max = 0.15
CFL_xz = 0.30
dt_safety = 0.85
trip_amp = 1.0
trip_duration = 0.20
trip_ramp_off = 0.10
trip_w_scale = 2.0
filter_strength = 0.03
filter_interval = 2
stretch_beta = 2.0
scheme = skew
integrator = rk3
gpu_only_mode = true
perf_mode = true
```

This achieves Re_tau ~ 278 (filter adds ~16x molecular viscosity). Grid: dx+ ~ 11.8, dz+ ~ 5.9, y1+ ~ 0.29.

## Timing Infrastructure

Use `TIMED_SCOPE` for performance profiling:
```cpp
#include "timing.hpp"
void my_function() {
    TIMED_SCOPE("my_function");
    // ... code ...
}
TimingStats::instance().print_summary();  // at end of program
```

For GPU builds, check GPU utilization:
```cpp
auto& stats = TimingStats::instance();
double gpu_ratio = stats.gpu_utilization_ratio();      // 0.0 to 1.0
stats.assert_gpu_dominant(0.7, "solver test");         // throws if < 70%
```

## Performance Notes

- Poisson solver dominates: ~83% of step time at 8.4M cells with MG
- LES SGS models add ~4% overhead (fused GPU kernels compute gradient + nu_sgs in one pass)
- IBM forcing adds <0.3% overhead (element-wise weight multiply, sub-millisecond)
- Memory transfers are negligible (<3%) — persistent mapping works well
- 3D scaling: memory grows as O(N^3)
- Benchmark: `bench_les_ibm_gpu [Nx] [Ny] [Nz] [nsteps]` (see `docs/LES_IBM_GPU_GUIDE.md`)

## Workflow Rules

### Scope Discipline
- Default to **<= 3 files** and **<= 200 LOC** touched per change
- If more is needed, stop and propose the file list + rationale before proceeding
- Keep existing style/conventions; no large renames/reformats unless asked
- Do not create new markdown/doc files unless explicitly asked
- When unsure, ask 1 targeted question instead of exploring widely

### Output Style
- Default output: patch/diff first
- If explanation is needed: <= 4 bullets, 1 sentence each
- Do not use heredocs for git commit messages — use `git commit -m "message"` directly

### Debugging Protocol
Never request full logs. Use diagnostic packets only:
- **GitHub CI**: `./tools/ci_packet.sh <runid>` (omit runid for latest)
- **SLURM**: `./tools/spkt <JOBID>` (omit JOBID for latest)
- **Local**: `./tools/local_packet.sh <command...>`

A result packet contains: (1) exact command, (2) where it ran, (3) exit status, (4) first error block + <= 40 lines context, (5) stderr tail <= 120 lines + stdout tail <= 80 lines, (6) file:line references.

If more info is needed after a packet: ask for **ONE** additional item only.

### SLURM
- Always submit slurm jobs using the **embers** QOS

## Poisson Solver Details

6 backends with auto-selection priority: FFT → FFT2D → FFT1D → HYPRE → Multigrid → SOR.

**Multigrid specifics:**
- Semi-coarsening for stretched y-grids (coarsens x/z, keeps y fixed)
- Chebyshev polynomial smoother with Gershgorin eigenvalue bounds
- PCG coarse solver with breakdown restart: `pAp < 1e-30` triggers restart instead of abort
- Convergence check throttling: `r_norm_sq` only computed every 4th iteration (reduces GPU→CPU sync)
- CUDA Graph optimization: entire V-cycle as single GPU graph (disabled for recycling inflow and semi-coarsening)

**Convergence criteria** (any triggers exit):
- RHS-relative: `‖r‖/‖b‖ < tol_rhs` (default for projection)
- Initial-residual relative: `‖r‖/‖r₀‖ < tol_rel`
- Absolute: `‖r‖_∞ < tol_abs`
- Fixed-cycle mode: exact N cycles without convergence checks (fastest)
- Adaptive fixed-cycle mode: check after N cycles, add more if needed

**Warm-start**: Poisson solver reuses previous `pressure_correction_` as initial guess (already on GPU, no action needed). First step starts from zero.

## Common Gotchas

- `head -N` in a pipe kills the background process via SIGPIPE
- `nohup` blocks stdout buffering — use periodic log file checks instead
- Always set `OMP_TARGET_OFFLOAD=MANDATORY` for GPU test runs to prevent silent CPU fallback
- Poisson solver auto-selects: FFT > HYPRE > Multigrid (based on BCs and available backends)
- The `benchmark` and `perf_mode` flags reduce diagnostics/GPU sync for production speed
- `accumulate_statistics()` and `validate_turbulence_realism()` read fields on CPU — must call `sync_solution_from_gpu()` first
- CUDA Graphs are auto-recaptured if BCs change, but must be manually disabled for recycling inflow (`poisson_use_vcycle_graph = false`)
- Staggered grid indexing: u-stride = `(Nx+1) + 2*Ng`, v has `(Ny+1)` rows, w-stride = `Nx + 2*Ng` — always use `mesh_->index()` helpers
- RHS of Poisson equation subtracts mean divergence to ensure solvability for periodic/all-Neumann BCs
- `CFL_y` always uses `CFL_max` (no relaxation) — this is deliberate after blow-up bugs with relaxed y-CFL
- Bash scripting: `((VAR++))` returns exit status 1 when VAR is 0 (breaks `set -e`) — use `VAR=$((VAR + 1))` instead
- `map(present: ptr)` with bare pointer names does NOT work with nvc++ — always use array sections: `map(present: ptr[0:size])`
- IBM `set_ibm_forcing()` must be called either before or after GPU init — it auto-detects `gpu_ready_` and calls `map_to_gpu()` accordingly
- IBM geometry functions (`phi()`) are virtual and cannot be called on GPU — all geometry evaluation must happen at init time with results stored in pre-computed arrays
- Thomas implicit y-diffusion uses `_stretched` variants when `mesh_->is_y_stretched()`: `alpha_lo = dt*nu/(dyv[j]*dyc[j])`, `alpha_hi = dt*nu/(dyv[j]*dyc[j+1])` for u/w; `alpha_lo = dt*nu/(dyc[j]*dyv[j-1])`, `alpha_hi = dt*nu/(dyc[j]*dyv[j])` for v
