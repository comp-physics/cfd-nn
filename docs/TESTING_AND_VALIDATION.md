# Testing and Validation Guide

Consolidated guide for running, extending, and interpreting the NN-CFD test suite and validation results.

---

## Quick Start

```bash
# Build
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run all tests
ctest --output-on-failure

# Run only fast tests (<30s each)
make check-fast

# Run fast + medium tests (skip slow)
make check-quick

# Run GPU-specific tests
ctest -L gpu

# Run a single test by name
ctest -R PoiseuilleTest --output-on-failure
```

### Pre-Push Checks

Before pushing, run the CI scripts locally:

```bash
./test_before_ci.sh          # CPU CI (fast + medium tests)
./test_before_ci_gpu.sh      # GPU CI (requires GPU node)
```

Or use the unified CI script:

```bash
./scripts/ci.sh fast         # Fast tests only
./scripts/ci.sh --cpu        # CPU-only (no GPU, no HYPRE)
./scripts/ci.sh              # Full suite (GPU + HYPRE if available)
```

---

## Test Labels

Every test has at least one label that controls when it runs:

| Label | Time | When It Runs |
|-------|------|-------------|
| `fast` | < 30s | Every CI run (Debug + Release) |
| `medium` | 30s - 5min | Release CI runs |
| `slow` | > 5min | Full local runs, not default CI |
| `gpu` | varies | GPU CI only; `OMP_TARGET_OFFLOAD=MANDATORY` |
| `hypre` | varies | When built with `-DUSE_HYPRE=ON` |
| `fft` | varies | When FFT backends available (cuFFT on GPU) |
| `cross-build` | varies | GPU CI cross-validation stage |

Filter by label:

```bash
ctest -L fast                    # Only fast
ctest -L medium                  # Only medium
ctest -LE slow                   # Exclude slow (fast + medium)
ctest -L "gpu" -L "medium"      # Tests with BOTH labels
```

---

## Test Harness API

All tests use the unified harness in `tests/test_harness.hpp`. Import it:

```cpp
#include "test_harness.hpp"
using namespace nncfd::testing;
```

### Recording Results

Four recording functions, each serving a different purpose:

#### `record(name, pass)` -- Basic Pass/Fail

```cpp
record("velocity_bounded", max_vel < 50.0);
```

Emits `[PASS]` or `[FAIL]`. Failing a `record()` call fails the test.

#### `record_gate(name, pass, actual, threshold)` -- Hard CI Gate with Values

```cpp
record_gate("L2_error", err < 0.01, err, 0.01);
// Output: [GATE:PASS] L2_error = 0.0034 (threshold: 0.01)
```

Same as `record()` but prints the actual value and threshold for easier debugging. Use for quantitative checks.

#### `record_track(name, actual, goal)` -- Diagnostic Tracking (Never Fails CI)

```cpp
record_track("Re_tau", re_tau, 180.0);
// Output: [TRACK:PASS] Re_tau = 178.5 (goal: 180.0)
// or:     [TRACK:WARN] Re_tau = 220.0 (goal: 180.0)
```

Logs the metric but **never fails CI**. Use for aspirational targets or diagnostics that aren't reliable enough to gate on.

#### `record_ratchet(name, actual, baseline, margin, goal)` -- Regression Detection

```cpp
record_ratchet("galilean_err", err, baseline_err, 0.10, 1e-8);
// Limit = min(baseline * 1.10, baseline + abs_floor)
```

Prevents regression: fails if `actual > limit`, where `limit = min(baseline * (1 + margin), baseline + abs_floor)`. The `abs_floor` prevents flaky failures when the baseline is near zero.

### GATE vs TRACK

| | GATE | TRACK |
|---|------|-------|
| Fails CI? | Yes | Never |
| Use for | Hard requirements (convergence, stability, correctness) | Aspirational targets (Re_tau, accuracy goals) |
| Output prefix | `[GATE:PASS]` / `[GATE:FAIL]` | `[TRACK:PASS]` / `[TRACK:WARN]` |

**Rule of thumb:** If the check failing means the code is *broken*, use GATE. If it means the code isn't *as good as we'd like*, use TRACK.

### Running the Harness

```cpp
int main(int argc, char* argv[]) {
    // Single-section test
    return run("MyTest", argc, argv, []() {
        // ... test code with record() calls ...
    });

    // Multi-section test
    return run_sections("MyTest", argc, argv, {
        {"Section1", []() { /* ... */ }},
        {"Section2", []() { /* ... */ }},
    });
}
```

The `run()` wrapper handles:
- Argument parsing
- GPU canary check (detects if GPU offload is actually working)
- Exception handling with diagnostics dump
- Exit code (0 = all pass, 1 = any fail)

### Diagnostics on Failure

Use `CHECK_OR_DUMP()` for automatic diagnostics on failure:

```cpp
SimDiagnostics diag;
// ... after each step, diag records last 5 steps ...

CHECK_OR_DUMP(max_vel < 50.0, diag, "velocity_bounded");
// On failure, prints rolling window of last 5 steps: dt, residual, CFL, etc.
```

### QoI Extraction

Tests can emit machine-readable metrics via `QOI_JSON:` lines:

```cpp
emit_qoi_rans_channel(model_name, u_tau, re_tau, bulk_vel, l2_err);
// Output: QOI_JSON:{"test":"rans_channel","model":"sst","u_tau":0.98,...}
```

Available emitters:
- `emit_qoi_tgv_2d()` / `emit_qoi_tgv_3d()` -- Taylor-Green vortex metrics
- `emit_qoi_repeatability()` -- Bitwise comparison
- `emit_qoi_cpu_gpu()` -- CPU/GPU validation
- `emit_qoi_hypre()` -- HYPRE vs MG comparison
- `emit_qoi_mms()` -- MMS convergence rates
- `emit_qoi_rans_channel()` -- RANS channel metrics
- `emit_qoi_perf()` -- Performance gate metrics

CI scripts can parse these lines to track metrics over time.

---

## Adding a New Test

### Step 1: Create the Test File

```cpp
// tests/test_my_feature.cpp
#include "test_harness.hpp"
#include "solver.hpp"
#include "config.hpp"

using namespace nncfd;
using namespace nncfd::testing;

int main(int argc, char* argv[]) {
    return run("MyFeatureTest", argc, argv, []() {
        // Setup
        Config config;
        config.set_Nx(32);
        config.set_Ny(64);
        config.set_nu(0.1);
        config.set_dp_dx(-1.0);

        RANSSolver solver(config);
        solver.set_body_force(config.dp_dx(), 0.0, 0.0);
        solver.initialize_uniform(0.0, 0.0, 0.0);

        // Run
        for (int i = 0; i < 100; i++) {
            solver.step();
        }

        // Check
        record_gate("converged", solver.residual() < 1e-4,
                     solver.residual(), 1e-4);
        record("velocity_bounded", solver.max_velocity() < 50.0);
    });
}
```

### Step 2: Register in CMakeLists.txt

Add to the appropriate label section in the root `CMakeLists.txt`:

```cmake
# In the fast tests section (~line 426):
add_nncfd_test(test_my_feature TEST_NAME_SUFFIX MyFeatureTest LABELS fast)

# Or for GPU tests:
add_nncfd_test(test_my_feature TEST_NAME_SUFFIX MyFeatureTest LABELS "gpu;medium")
set_tests_properties(MyFeatureTest PROPERTIES ENVIRONMENT "OMP_TARGET_OFFLOAD=MANDATORY")
```

The `add_nncfd_test()` macro:
1. Creates an executable from `tests/test_my_feature.cpp`
2. Links against `nn_cfd_core` library
3. Registers with CTest under the given name and labels

### Step 3: Choose the Right Label

| If your test... | Label |
|----------------|-------|
| Runs in < 30s, no special hardware | `fast` |
| Runs in 30s-5min, or needs grid convergence | `medium` |
| Runs > 5min, or does exhaustive checking | `slow` |
| Requires GPU hardware | add `gpu` |
| Tests HYPRE solver | add `hypre` |
| Tests FFT solver | add `fft` |

### Step 4: Build and Run

```bash
cd build && make -j$(nproc)
ctest -R MyFeatureTest --output-on-failure
```

---

## GPU Testing

### Environment

GPU tests require `OMP_TARGET_OFFLOAD=MANDATORY` to prevent silent CPU fallback. This is set automatically via `set_tests_properties()` in CMakeLists.txt, but set it manually for ad-hoc runs:

```bash
OMP_TARGET_OFFLOAD=MANDATORY ./test_my_gpu_test
```

### GPU Canary Check

The test harness automatically runs a GPU canary check at startup (inside `run()`). If the canary fails, the test is skipped rather than producing misleading results.

### Writing GPU-Aware Tests

Use device-side diagnostics to avoid GPU-CPU sync:

```cpp
// GOOD -- no sync needed
double ke = solver.compute_kinetic_energy_device();
double div = solver.compute_divergence_linf_device();
double vmax = solver.compute_max_velocity_device();

// BAD -- requires sync (slow, may mask bugs)
solver.sync_solution_from_gpu();
double ke = compute_ke_on_host(solver);
```

If you must read field data on the CPU, sync first:

```cpp
solver.sync_solution_from_gpu();
// Now safe to read velocity/pressure arrays on host
```

### GPU Build

```bash
mkdir -p build_gpu && cd build_gpu
cmake .. -DCMAKE_CXX_COMPILER=nvc++ -DUSE_GPU_OFFLOAD=ON -DGPU_CC=90
make -j$(nproc)
```

Use `-DGPU_CC=90` for H200, `-DGPU_CC=80` for A100.

---

## CI Architecture

### CPU CI (GitHub Actions)

Workflow: `.github/workflows/ci.yml`

```
Push -> Build (Debug + Release) x (Ubuntu + macOS)
     -> Run fast tests (Debug)
     -> Run fast + medium tests (Release)
     -> Check for compiler warnings
```

Runs on every push. Tests use the CPU-only build (no GPU offload, no HYPRE).

### GPU CI (Self-Hosted SLURM)

Workflow: `.github/workflows/gpu-ci.yml`

```
Push -> Build GPU binary (nvc++ + HYPRE)
     -> Submit correctness suite to SLURM (H200)
     -> Submit performance suite to SLURM
     -> Cross-build validation (CPU vs GPU outputs)
     -> Cache HYPRE build for next run
```

Key details:
- **Serialized per-branch**: Only one GPU CI job runs per branch at a time
- **240-minute timeout** with cleanup trap for SLURM jobs
- **HYPRE caching**: HYPRE is slow to compile; cached by version extracted from CMakeLists.txt
- **Failure diagnostics**: On failure, dumps commit info, build dirs, HYPRE cache status, test binary list

### Unified CI Script

`scripts/ci.sh` provides a single entry point for both local and CI use:

```bash
./scripts/ci.sh              # All tests (GPU+HYPRE default)
./scripts/ci.sh fast         # Only fast tests
./scripts/ci.sh --cpu        # CPU-only
./scripts/ci.sh --cpu fast   # CPU-only fast
./scripts/ci.sh --debug      # Debug build (4x timeout)
./scripts/ci.sh -v           # Verbose
```

Auto-detects GPU compute capability from `nvidia-smi` when available.

---

## Validation Results

### Laminar Channel Flow (Poiseuille)

The solver validates against the analytical Poiseuille solution for pressure-driven channel flow:

**Analytical solution:**
```
u(y) = -(dp/dx)/(2nu) (H^2 - y^2)
```

where H is the channel half-height and dp/dx < 0 is the imposed pressure gradient.

**Test case:** Channel with dp/dx = -1, half-height H = 1

| nu | Grid | dt | Iterations | L2 Error | Bulk Velocity Error |
|---|------|----|-----------|---------|--------------------|
| 0.1 | 32x64 | 0.005 | 10,000 | 0.13% | 0.01% |
| 0.1 | 16x32 | 0.005 | 20,000 | 0.13% | 0.02% |
| 0.01 | 32x64 | 0.0002 | 100,000+ | ~1% | ~3% |

**Key findings:**
- Excellent agreement at moderate viscosity (nu = 0.1)
- Stable and robust convergence
- Lower viscosity requires much smaller timesteps due to diffusion stability constraint

**Recommended parameters:**

High viscosity (nu >= 0.1):
```bash
./channel --Nx 32 --Ny 64 --nu 0.1 --dt 0.005 --max_steps 10000 --tol 1e-8
```

Moderate viscosity (nu ~ 0.01):
```bash
./channel --Nx 32 --Ny 64 --nu 0.01 --dt 0.0002 --max_steps 100000 --tol 1e-8
```

Low viscosity (nu < 0.01):
```bash
./channel --Nx 64 --Ny 128 --nu 0.001 --dt 0.0001 --max_steps 200000 --stretch
```

**Timestep selection:**

1. **CFL condition (convection):**
   ```
   dt <= CFL_max * min(dx, dy) / |u_max|
   ```
   Typically CFL_max = 0.5

2. **Diffusion stability:**
   ```
   dt <= 0.5 * min(dx^2, dy^2) / (nu + nu_t)
   ```
   This is usually the limiting factor for laminar and low-Re flows

### Turbulent Channel Flow (Mixing Length Baseline)

**Test case:** Re = 10,000 (based on channel height and mean velocity)

```bash
./channel --Nx 64 --Ny 128 --nu 0.001 --model baseline --max_steps 20000
```

**Results:**
- Stable convergence
- Reasonable eddy viscosity distribution
- Non-zero wall shear stress
- Not directly validated against DNS (mixing length is approximate)

**Performance:**
- Baseline model adds ~20% computational cost vs laminar
- Most time still spent in Poisson solver

### Neural Network Models

**Test with example weights (random, untrained):**

```bash
# MLP model
./channel --model nn_mlp --nn_preset example_scalar_nut --Nx 16 --Ny 32

# TBNN model
./channel --model nn_tbnn --nn_preset example_tbnn --Nx 16 --Ny 32
```

**Results:**
- Infrastructure loads correctly
- NN inference executes without errors
- Feature computation works
- Random weights cause divergence (expected)

**Note:** Real trained weights from published models are needed for meaningful validation.

### DNS Channel Flow (No Turbulence Model)

DNS resolves all turbulence scales directly without any model. See `docs/DNS_CHANNEL_GUIDE.md` for the full guide.

**Target:** Re_tau = 180 channel (Moser, Kim & Mansour 1999)

**Grid:** 192 x 96 x 192, Lx = 4pi, Ly = 2, Lz = 2pi, stretch_beta = 2.0

**Run history:**

| Run | Filter | CFL | Result | Re_tau | Notes |
|-----|--------|-----|--------|--------|-------|
| v9 | none | CFL_xz=0.30, CFL_max=0.15 | Blew up ~step 1700 | N/A | Turbulent before blow-up |
| v10 | strength=0.02, interval=10 (x/z only) | same | Blew up ~step 2000 | N/A | Survived longer |
| v11 | strength=0.05, interval=1 (x/y/z) | same | Stable, 3600+ steps | ~255 | First stable run |
| v13 | strength=0.03, interval=2 (x/y/z) | same | Stable, 2400+ steps | ~278 | Best balance |

**Key results from v11 (first fully stable DNS):**

| Step | Re_tau | v_max | w/v ratio | State |
|------|--------|-------|-----------|-------|
| 1200 | 222 | 16.7 | 0.054 | TURBULENT, trip OFF |
| 1800 | 250 | 15.2 | 0.609 | TURBULENT |
| 2400 | 307 | 15.3 | 1.454 | TURBULENT (peak Re_tau) |
| 3600 | 255 | 20.6 | 1.21 | TURBULENT (stabilizing) |

**Known gap:** The velocity filter adds effective viscosity, so the achieved Re_tau (~255-278) exceeds the target of 180. Reaching the exact target would require a less dissipative convective scheme (e.g., hybrid skew-symmetric/upwind).

### Recycling Inflow Validation

See `docs/RECYCLING_INFLOW_GUIDE.md` for the full guide.

**PeriodicVsRecyclingTest:** Runs identical channel flow with periodic BCs and with recycling inflow BCs, then compares plane-averaged statistics:

| Metric | Tolerance | Achieved |
|--------|-----------|----------|
| Shear stress difference | < 5% | ~0.3% |
| Streamwise stress difference | < 5% | ~3.6% |

**RecyclingInflowTest:** 12 checks covering symmetry, mass conservation, divergence, fringe blending, ghost cells. All 12 passing on both CPU and GPU.

### Unit Tests

**Mesh and Fields** (`test_mesh`):
- Mesh indexing, ghost cell handling, field operations, wall distance computation

**Poisson Solver** (`test_poisson`):
- Laplacian discretization accuracy, convergence for manufactured solution, boundary conditions (Dirichlet, Neumann, periodic)

**Multigrid Manufactured Solution** (`test_mg_manufactured_solution`):
- Standard channel BCs (periodic x/z, Neumann y)
- Duct BCs (periodic x, Neumann y/z)
- Recycling inflow BCs (Dirichlet x_lo, Neumann x_hi, Neumann y, periodic z)
- CPU/GPU consistency (max difference = 0.0 for identical inputs)

**FFT Unified** (`test_fft_unified`):
- FFT 3D solver (periodic x/z channel), FFT 1D solver (periodic x duct)
- Grid convergence (error decreases with resolution), GPU/CPU consistency

**Recycling Inflow** (`test_recycling_inflow`, `test_periodic_vs_recycling`):
- RecyclingInflowTest: 12 checks (symmetry, u_tau, mass conservation, fringe, divergence, ghost cells)
- PeriodicVsRecyclingTest: recycling matches periodic within 5% for shear and streamwise stress

**Neural Network Loading** (`test_nn_simple`):
- MLP weight loading, forward pass correctness, feature computation
- TurbulenceNNMLP and TurbulenceNNTBNN initialization

---

## Performance Summary

**Timing for 10,000 iterations on 16x32 grid:**

| Configuration | Total Time | Per Iteration | Relative |
|--------------|------------|---------------|----------|
| Laminar | 0.08 s | 0.008 ms | 1x |
| Baseline | 0.46 s | 0.046 ms | 5.8x |
| NN-MLP | 3.88 s | 0.388 ms | 48x |
| NN-TBNN | 21.19 s | 2.119 ms | 265x |

**Breakdown for laminar case:**
- Poisson solve: 40%
- Convective/diffusive terms: 50%
- Boundary conditions: 10%

**For NN models:**
- NN inference: 95%
- Rest of solver: 5%

**GPU production runs (192x96x192, H200):**
- Poisson solver dominates: ~83% of step time with multigrid
- LES SGS models add ~4% overhead (fused GPU kernels)
- IBM forcing adds <0.3% overhead
- Memory transfers: <3% (persistent mapping)
- Throughput: ~17 steps/min with MG, dt~1.5e-4 at CFL=0.15

---

## Tier 2 Validation

Tier 2 tests run on GPU clusters via SLURM and take hours. They validate physics accuracy against reference data.

### DNS Channel Validation

```bash
# Submit the full DNS run (6+ hours)
sbatch scripts/run_validation.sh
```

This runs the v13 DNS recipe (192x96x192, 3750 steps) and collects Re_tau history, velocity profiles, and turbulence statistics.

**Tier 2 DNS results (H200):** 192x96x192, 3750 steps. Re_tau peaked at 345, settling to ~284. w/v ratio = 1.0-1.45.

**Poiseuille convergence:** 2nd-order convergence confirmed (order 2.02, 2.10).

**TGV Re=1600:** 64^3 stable through vortex breakdown (under-resolved for accuracy).

### RANS Validation Campaign

```bash
# Submit RANS validation (4 hours)
cd scripts/rans_validation
sbatch run_rans_validation.sbatch

# Analyze results after job completes
python analyze.py --log-dir output/
```

The RANS validation:
1. Runs 6+ RANS models for 10,000 steps each (full convergence)
2. Compares u+ profiles against MKM DNS reference data
3. Generates plots and a markdown summary report

**Note:** RANS baseline.cfg has Re_tau=21 (not 180). Must override with `--nu 0.005556 --dp_dx -1.0`.

### Reference Data

MKM DNS reference: download from `https://turbulence.oden.utexas.edu/data/MKM/chandata.tar.gz`

---

## Convergence Criteria

The solver uses velocity residual to determine convergence:

```
residual = max|u^(n+1) - u^n|
```

Typical convergence:
- **Laminar:** Exponential decay, reaches tol=1e-8 reliably
- **Baseline turbulence:** Slower convergence, reaches tol=1e-6
- **NN models:** Depends on weights (untrained weights diverge)
- **DNS (unsteady):** Residual does not converge to zero; use `max_steps` to control run length

---

## Troubleshooting

### Test fails only on GPU

1. Check `OMP_TARGET_OFFLOAD=MANDATORY` is set -- without it, GPU tests silently run on CPU
2. Run the GPU canary: `ctest -R GPUMappingCanaryTest`
3. Check GPU memory: `nvidia-smi` -- out-of-memory causes silent failures
4. Compare CPU vs GPU output: `ctest -R CrossBackendTest`

### RANS model blows up

1. Check which model: transport models (SST, k-omega) are stiffer than algebraic
2. Run the sanity test: `ctest -R RANSChannelSanityTest`
3. Check for NaN in nu_t: the NaN-masking bug (fixed March 2026) caused silent failures
4. For stretched grids, verify the Poisson solver isn't FFT2D: `poisson_solver = mg` in config

### Poisson solver doesn't converge

1. Check solver selection: add `verbose = true` and look for "Poisson solver:" in output
2. For stretched grids: multigrid with semi-coarsening is the safest choice
3. For recycling inflow: CUDA Graphs must be disabled (`poisson_use_vcycle_graph = false`)
4. Check grid resolution: very coarse grids can stall multigrid

### Test passes locally but fails in CI

1. **Debug vs Release**: Debug builds use 4x timeout multiplier. Some tests are time-sensitive.
2. **Compiler differences**: NVHPC (GPU CI) vs GCC/Clang (CPU CI) may produce different floating-point results
3. **Tolerance**: If a test is flaky, consider using `record_track()` instead of `record_gate()` for soft metrics

### GPU CI timeout

1. GPU CI has 240-minute timeout. Large test suites can hit this.
2. HYPRE build cache miss adds ~20 minutes. Check cache status in CI logs.
3. SLURM queue wait time counts against the timeout. Submit during off-peak hours.

### Cross-build test fails

1. This means CPU and GPU produce different results beyond tolerance
2. Common cause: uninitialized GPU memory (GPU has random data, CPU has zeros)
3. Check if the failing metric is a field value (likely data issue) or a scalar diagnostic (likely reduction issue)

---

## Known Issues and Limitations

### Numerical

1. **Explicit time stepping** limits timestep for low viscosity
   - **Mitigation:** Adaptive time stepping with directional CFL is implemented. See `docs/DNS_CHANNEL_GUIDE.md`.

2. **Central differences require velocity filter for DNS stability**
   - Second-order central schemes have zero numerical dissipation, causing grid-scale blow-up in DNS. The velocity filter (`filter_strength`, `filter_interval`) provides explicit diffusion but adds effective viscosity. See `docs/DNS_CHANNEL_GUIDE.md`.

3. **Filter-limited Re_tau in DNS**
   - The velocity filter prevents reaching the exact target Re_tau = 180. Best achieved: Re_tau ~ 278 with strength=0.03, interval=2. A higher-order or hybrid convective scheme would reduce filter requirements.

### Model-Related

4. **Example NN models use random weights**
   - **Solution:** Add real trained weights from publications

5. **Feature sets may not match published models exactly**
   - **Solution:** Verify feature definitions when adding real models

6. **No automatic feature set detection**
   - **Solution:** Manually configure features per model for now

### GPU-Specific

7. **CPU-side diagnostics require GPU sync**
   - Functions that read velocity data on the CPU (e.g., `accumulate_statistics()`, `validate_turbulence_realism()`) must call `sync_solution_from_gpu()` first. The solver handles this internally for built-in diagnostics, but custom code must sync manually.

8. **Recycling inflow disables CUDA Graph V-cycle**
   - Because recycling modifies inlet BCs each step, the CUDA Graph is invalid. The solver falls back to the standard V-cycle path (~10-20% slower for Poisson solves).

---

## References

- **Poiseuille flow:** Classical analytical solution
- **Mixing length:** Pope, "Turbulent Flows" (2000)
- **TBNN:** Ling et al., JFM 807 (2016)
- **DNS channel:** Moser, Kim & Mansour, Physics of Fluids 11.4 (1999)
- **Recycling inflow:** Lund, Wu & Squires, J. Comput. Phys. 140.2 (1998)

---

**Last updated:** March 2026
