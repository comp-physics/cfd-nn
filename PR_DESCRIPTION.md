# PR: GPU Offloading Correctness Hardening

## Problem

NVHPC OpenMP target has a bug where mapped host pointers in target regions sometimes receive the **host address** instead of the device address. This causes:
- Kernels silently reading/writing host memory
- Incorrect physics results
- Random crashes

## The One Rule Going Forward

**All GPU kernels must use:**
```cpp
double* u_dev = gpu::dev_ptr(velocity_u_ptr_, "velocity_u");
#pragma omp target teams distribute parallel for is_device_ptr(u_dev)
for (int i = 0; i < n; ++i) {
    u_dev[i] = ...;
}
```

**Forbidden patterns:**
- `firstprivate(ptr)` in target regions
- Local pointer aliases with `map(present:)`
- Raw mapped pointers without `is_device_ptr`

## Regression Prevention

| Gate | What it catches |
|------|-----------------|
| `scripts/lint_gpu_pointers.sh` | Unsafe pointer patterns in code |
| `test_gpu_mapping_canary` | GPU execution actually works |
| `test_tgv_2d/3d_invariants` | Physics correctness on GPU |

## Performance Guarantees

- **No H↔D transfers during stepping** (except opt-in diagnostic)
- **Scalars-only QoIs** via device reductions
- **Persistent mapping** for solver lifetime

## Scope

This PR does NOT include:
- Data layout refactors
- Numerical method changes
- API changes

## Files Changed

| File | Change |
|------|--------|
| `include/gpu_utils.hpp` | Added NVHPC workaround docs, improved `dev_ptr()` error messages |
| `src/solver_time.cpp` | Removed unsafe debug code |
| `src/poisson_solver_multigrid.cpp` | Added contract comment |
| `scripts/lint_gpu_pointers.sh` | **NEW** - CI lint for unsafe patterns |
| `tests/test_gpu_mapping_canary.cpp` | **NEW** - GPU mapping smoke test |
| `CMakeLists.txt` | Added new test |
| `PR1_AUDIT.md` | **NEW** - Detailed audit checklist |

## Test Results (GPU Build)

```
test_gpu_mapping_canary:     8/8 PASS
test_tgv_2d_invariants:      9/9 PASS
test_tgv_3d_invariants:      8/8 PASS
lint_gpu_pointers.sh:        PASS
```

## Merge Readiness

- [x] GPU canary test verifies device execution
- [x] Lint script blocks unsafe patterns
- [x] TGV invariants pass on GPU
- [x] No unexpected H↔D transfers during stepping
- [x] Contract documented for future contributors
- [x] O4+MG guard remains in place
