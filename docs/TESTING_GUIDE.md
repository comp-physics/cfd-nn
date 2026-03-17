# Testing Guide

How to run, understand, and extend the NN-CFD test suite.

For validation *results*, see [`TESTING_AND_VALIDATION.md`](TESTING_AND_VALIDATION.md).

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

#### `record(name, pass)` — Basic Pass/Fail

```cpp
record("velocity_bounded", max_vel < 50.0);
```

Emits `[PASS]` or `[FAIL]`. Failing a `record()` call fails the test.

#### `record_gate(name, pass, actual, threshold)` — Hard CI Gate with Values

```cpp
record_gate("L2_error", err < 0.01, err, 0.01);
// Output: [GATE:PASS] L2_error = 0.0034 (threshold: 0.01)
```

Same as `record()` but prints the actual value and threshold for easier debugging. Use for quantitative checks.

#### `record_track(name, actual, goal)` — Diagnostic Tracking (Never Fails CI)

```cpp
record_track("Re_tau", re_tau, 180.0);
// Output: [TRACK:PASS] Re_tau = 178.5 (goal: 180.0)
// or:     [TRACK:WARN] Re_tau = 220.0 (goal: 180.0)
```

Logs the metric but **never fails CI**. Use for aspirational targets or diagnostics that aren't reliable enough to gate on.

#### `record_ratchet(name, actual, baseline, margin, goal)` — Regression Detection

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

---

## QoI Extraction

Tests can emit machine-readable metrics via `QOI_JSON:` lines:

```cpp
emit_qoi_rans_channel(model_name, u_tau, re_tau, bulk_vel, l2_err);
// Output: QOI_JSON:{"test":"rans_channel","model":"sst","u_tau":0.98,...}
```

Available emitters:
- `emit_qoi_tgv_2d()` / `emit_qoi_tgv_3d()` — Taylor-Green vortex metrics
- `emit_qoi_repeatability()` — Bitwise comparison
- `emit_qoi_cpu_gpu()` — CPU/GPU validation
- `emit_qoi_hypre()` — HYPRE vs MG comparison
- `emit_qoi_mms()` — MMS convergence rates
- `emit_qoi_rans_channel()` — RANS channel metrics
- `emit_qoi_perf()` — Performance gate metrics

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
// GOOD — no sync needed
double ke = solver.compute_kinetic_energy_device();
double div = solver.compute_divergence_linf_device();
double vmax = solver.compute_max_velocity_device();

// BAD — requires sync (slow, may mask bugs)
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

## Tier 2 Validation

Tier 2 tests run on GPU clusters via SLURM and take hours. They validate physics accuracy against reference data.

### DNS Channel Validation

```bash
# Submit the full DNS run (6+ hours)
sbatch scripts/run_validation.sh
```

This runs the v13 DNS recipe (192x96x192, 3750 steps) and collects Re_tau history, velocity profiles, and turbulence statistics.

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
2. Compares $u^+$ profiles against MKM DNS reference data
3. Generates plots and a markdown summary report

### Reference Data

MKM DNS reference: download from `https://turbulence.oden.utexas.edu/data/MKM/chandata.tar.gz`

---

## CI Architecture

### CPU CI (GitHub Actions)

Workflow: `.github/workflows/ci.yml`

```
Push → Build (Debug + Release) × (Ubuntu + macOS)
     → Run fast tests (Debug)
     → Run fast + medium tests (Release)
     → Check for compiler warnings
```

Runs on every push. Tests use the CPU-only build (no GPU offload, no HYPRE).

### GPU CI (Self-Hosted SLURM)

Workflow: `.github/workflows/gpu-ci.yml`

```
Push → Build GPU binary (nvc++ + HYPRE)
     → Submit correctness suite to SLURM (H200)
     → Submit performance suite to SLURM
     → Cross-build validation (CPU vs GPU outputs)
     → Cache HYPRE build for next run
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

## Troubleshooting

### Test fails only on GPU

1. Check `OMP_TARGET_OFFLOAD=MANDATORY` is set — without it, GPU tests silently run on CPU
2. Run the GPU canary: `ctest -R GPUMappingCanaryTest`
3. Check GPU memory: `nvidia-smi` — out-of-memory causes silent failures
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

**Last updated:** March 2026
