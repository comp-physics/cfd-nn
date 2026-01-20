# PR: GPU Offloading Correctness Hardening

## 1. Problem

**NVHPC OpenMP target bug**: Mapped host pointers in target regions sometimes receive the **host address** instead of device address.

**Symptoms:**
- Kernels silently reading/writing host memory
- Incorrect physics results (wrong divergence, energy violations)
- Random crashes on some runs

**Root cause**: NVHPC's OpenMP implementation sometimes fails to translate mapped pointers when they appear in certain target constructs.

---

## 2. Rule

**All GPU kernels must use this pattern:**
```cpp
double* u_dev = gpu::dev_ptr(velocity_u_ptr_, "velocity_u");
#pragma omp target teams distribute parallel for is_device_ptr(u_dev)
for (int i = 0; i < n; ++i) {
    u_dev[i] = ...;
}
```

### Allowed in GPU Kernels

| Pattern | Example |
|---------|---------|
| `dev_ptr()` + `is_device_ptr` | `auto* x_dev = gpu::dev_ptr(x_host, "name");` |
| Scalars in firstprivate | `firstprivate(dt, dx, Nx)` |
| Device reductions | `reduction(+:sum)` |

### Forbidden

| Pattern | Why |
|---------|-----|
| Local pointer alias | `double* x = member_ptr_;` passed into target |
| `firstprivate(ptr)` | Mapped host pointer passed by value |
| `map(present: member_ptr_[...])` | May receive host address in kernel |

---

## 3. Safety

### CI Gates (must be blocking checks)

| Gate | What it catches |
|------|-----------------|
| `scripts/lint_gpu_pointers.sh` | Unsafe pointer patterns in code |
| `test_gpu_mapping_canary` | GPU execution actually works |
| `test_tgv_2d_invariants` | Physics correctness (divergence, energy) |
| `test_tgv_3d_invariants` | 3D indexing and symmetry |

### Runtime Checks (debug builds)

| Check | Location |
|-------|----------|
| `verify_mapping_integrity()` | Start of `step()` |
| `dev_ptr()` assertion | Every kernel launch |
| `sync_counter == 0` during stepping | TGV invariant tests |

---

## 4. Performance

### No H↔D Transfers During Stepping

- **Guarantee**: `sync_to_gpu()` and `sync_from_gpu()` are NOT called inside `step()`
- **Verification**: Debug sync counter in solver, asserted in TGV tests
- **Exception**: Opt-in diagnostic (env var `NNCFD_CONV_KE_DIAGNOSTICS`, disabled by default)

### Scalars-Only QoIs

All quantities of interest computed on device via reductions:
- `compute_kinetic_energy_device()`
- `compute_divergence_linf_device()`
- Max velocity magnitude

### Persistent Mapping

- Fields mapped once at solver construction
- Stay on device for entire solver lifetime
- Downloaded only at end for I/O

---

## 5. Scope

### This PR Does

- Add lint script for unsafe patterns
- Add mapping canary test
- Add NVHPC workaround documentation
- Add sync counter for debug verification
- Fix MG solver to use `dev_ptr` pattern

### This PR Does NOT

- Refactor data layouts
- Change numerical methods
- Modify public API
- Remove O4+MG guard (stays in place)

---

## Files Changed

| File | Change |
|------|--------|
| `include/gpu_utils.hpp` | NVHPC docs, improved `dev_ptr()`, sync counter |
| `src/solver.cpp` | Sync counter instrumentation |
| `src/poisson_solver_multigrid.cpp` | Contract comment, `dev_ptr` pattern |
| `tests/test_tgv_2d_invariants.cpp` | Sync assertion |
| `scripts/lint_gpu_pointers.sh` | **NEW** - CI lint |
| `tests/test_gpu_mapping_canary.cpp` | **NEW** - GPU smoke test |
| `CMakeLists.txt` | Added new test |

---

## Test Results (GPU Build)

```
lint_gpu_pointers.sh:        PASSED
test_gpu_mapping_canary:     8/8 PASS
test_tgv_2d_invariants:      9/9 PASS (+ sync check)
test_tgv_3d_invariants:      8/8 PASS
```

---

## CI Enforcement Requirements

**These must be blocking checks in GitHub branch protection:**

1. `lint_gpu_pointers.sh` - Runs on NVHPC GPU runner
2. `test_gpu_mapping_canary` - Runs on NVHPC GPU runner
3. `test_tgv_2d_invariants` - Runs on NVHPC GPU runner
4. `test_tgv_3d_invariants` - Runs on NVHPC GPU runner

If any are currently "informational", make them **required status checks**.

---

## Merge Checklist

- [x] GPU canary test verifies device execution (`omp_is_initial_device()`)
- [x] Lint script blocks unsafe patterns
- [x] TGV invariants pass on GPU
- [x] No H↔D transfers during stepping (sync counter verified)
- [x] Contract documented for future contributors
- [x] O4+MG guard remains in place
