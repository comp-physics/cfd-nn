# PR1 Audit: GPU Offloading Correctness

## Checklist Status

### 1. GPU Pointer Pattern ✅ COMPLETE
- **Pattern**: `gpu::dev_ptr(host_ptr, "field_name")` + `is_device_ptr(dev_ptr)` used consistently
- **Verification**: ~180 `dev_ptr()` calls, ~196 `is_device_ptr` clauses
- **No unsafe patterns**: Verified by `scripts/lint_gpu_pointers.sh`
- **Centralized helper**: `gpu::dev_ptr()` in `include/gpu_utils.hpp` with:
  - Assertion that mapping exists (aborts with diagnostics if not)
  - Optional context string for debugging
  - `omp_target_is_present()` check in error messages
  - Const overload
  - CPU stub that returns host pointer unchanged

### 2. Mapping Integrity Check ✅ COMPLETE
- **Location**: `verify_mapping_integrity()` in `solver.cpp:7286`
- **Called**: At start of `step()` in debug builds
- **Fields checked**:
  - velocity_u/v/w
  - velocity_star_u/v/w
  - conv_u/v/w
  - diff_u/v/w
  - pressure
  - nu_eff
  - div_velocity

### 3. No H↔D Transfers During Stepping ✅ COMPLETE
- **step() function**: No sync_to/from_gpu calls in main compute path
- **One exception**: `sync_from_gpu()` in convective KE diagnostic (line 4960)
  - Gated by env var `NNCFD_CONV_KE_DIAGNOSTICS` (disabled by default)
  - Only runs every 100 iterations when enabled
- **Scalars computed on device**: max|u|, divergence norms via reductions

### 4. Hard Correctness Gates ✅ VERIFIED PASSING
- **Divergence after projection**: `test_tgv_2d_invariants`, `test_tgv_3d_invariants`
- **Energy monotonicity/bounds**: `test_tgv_2d_invariants`, `test_tgv_3d_invariants`
- **Constant velocity preserved**: `test_step_trace` - PASS on CPU and GPU
- **RHS=0 => p_corr=0**: `test_mg_ramp_diagnostic` - PASS

### 5. Kill-Switch Guards ✅ COMPLETE
- **O4 + MG incompatibility**: Guard exists in config.cpp
- **Solver selection**: Tests use explicit solver types where tolerances matter

### 6. CI Regression Prevention ✅ COMPLETE
- **Lint script**: `scripts/lint_gpu_pointers.sh` catches:
  - `firstprivate` with raw pointers
  - Target regions missing `is_device_ptr`
  - Local pointer aliases in GPU code paths
  - Inconsistent `dev_ptr()` vs `is_device_ptr` usage
- **Mapping canary test**: `test_gpu_mapping_canary.cpp` verifies:
  - Device pointer resolution works
  - Kernels actually execute on device (not host fallback)
  - Reductions work correctly
  - Persistent mappings survive multiple kernel launches

---

## Test Results Summary (GPU Build)

| Test | Status | Notes |
|------|--------|-------|
| test_gpu_mapping_canary | 8/8 PASS | Validates dev_ptr + is_device_ptr pattern |
| test_tgv_2d_invariants | 9/9 PASS | All invariants satisfied |
| test_tgv_3d_invariants | 8/8 PASS | All invariants satisfied |
| test_galilean_stage_breakdown | 22/23 PASS | 1 physics diagnostic expected |
| test_step_trace | PASS | conv=0, diff=0, RHS=0, p_corr=0 |
| test_mg_ramp_diagnostic | PASS | Standalone MG works correctly |
| test_projection_trace | PASS | Constant velocity preserved |
| lint_gpu_pointers.sh | PASS | No unsafe pointer patterns |

---

## Items NOT in PR1 (Per Guidance)

1. ❌ API redesign or call signature changes
2. ❌ New storage layouts (multi-field packed buffers)
3. ❌ Splitting order fixes or pressure extrapolation
4. ❌ Sweeping 20+ kernel conversions

---

## Regression Prevention

### CI Gates
1. `lint_gpu_pointers.sh` - Blocks unsafe pointer patterns
2. `test_gpu_mapping_canary` - Blocks if GPU execution fails
3. `test_tgv_2d_invariants` - Blocks if physics breaks
4. `test_tgv_3d_invariants` - Blocks if 3D indexing breaks

### Contract Documentation
- `include/gpu_utils.hpp` - Comprehensive NVHPC workaround docs
- `src/poisson_solver_multigrid.cpp` - Contract comment at GPU section

---

## Files Touched by GPU Path

Core compute kernels in `src/solver.cpp`:
- `step()` - main time-stepping entry point
- `compute_convective_term()` - advection
- `compute_diffusive_term()` - diffusion
- `predictor_step()` - u* computation
- `correct_velocity()` - projection correction
- `compute_divergence()` - div(u*)
- `apply_velocity_bc()` - periodic BC enforcement

Poisson solver in `src/poisson_solver_multigrid.cpp`:
- `solve_device()` - GPU MG solver
- V-cycle kernels with `is_device_ptr` pattern

---

## New Files in This PR

| File | Purpose |
|------|---------|
| `scripts/lint_gpu_pointers.sh` | CI lint for unsafe pointer patterns |
| `tests/test_gpu_mapping_canary.cpp` | Fast GPU mapping verification |
