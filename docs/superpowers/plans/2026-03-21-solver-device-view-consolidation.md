# SolverDeviceView Consolidation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate ~60 redundant `gpu::dev_ptr()` calls in the solver hot path by migrating `solver_time.cpp` and `solver.cpp` GPU kernels to the existing `SolverDeviceView` pattern already used in `solver_operators.cpp`.

**Architecture:** The codebase already has `SolverDeviceView` (POD struct with field pointers + mesh params) and `get_solver_view()` (populates it from solver state). `solver_operators.cpp` and `solver_recycling.cpp` use it correctly with `map(present:)`. But `solver_time.cpp` (time integration — the hottest code path) and parts of `solver.cpp` (diagnostics) still call `gpu::dev_ptr()` per-kernel, adding ~60 unnecessary `omp_get_mapped_ptr()` roundtrips per time step. The fix: replace those raw `dev_ptr()` call sites with `get_solver_view()` + `map(present:)`, matching the pattern that already works.

**Tech Stack:** C++17, OpenMP target offload, nvc++ 25.x

**Risk:** LOW — this is a mechanical refactor from one working GPU pattern to another working GPU pattern already used in the same codebase. Both produce identical results.

---

## Context: Why Two Patterns Exist

The codebase evolved two GPU pointer patterns:

**Pattern A** (old, verbose — `solver_time.cpp`, parts of `solver.cpp`):
```cpp
double* u_dev = gpu::dev_ptr(velocity_u_ptr_);   // omp_get_mapped_ptr per call
double* v_dev = gpu::dev_ptr(velocity_v_ptr_);
double* conv_u_dev = gpu::dev_ptr(conv_u_ptr_);
double* diff_u_dev = gpu::dev_ptr(diff_u_ptr_);
// ... 8-12 dev_ptr calls per kernel launch

#pragma omp target teams distribute parallel for \
    is_device_ptr(u_dev, v_dev, conv_u_dev, diff_u_dev)
for (...) { ... }
```

**Pattern B** (clean — `solver_operators.cpp`, `solver_recycling.cpp`):
```cpp
auto v = get_solver_view();     // ONE call, all pointers ready
double* u_ptr = v.u_face;       // Just struct member access
double* v_ptr = v.v_face;

#pragma omp target teams distribute parallel for \
    map(present: u_ptr[0:u_sz], v_ptr[0:v_sz])
for (...) { ... }
```

Both work correctly. Pattern B is cleaner, already tested, and eliminates per-kernel `omp_get_mapped_ptr()` overhead.

## What Changes

**Only `solver_time.cpp` and diagnostic functions in `solver.cpp` change.** No new structs, no new files, no header changes. The `SolverDeviceView` struct and `get_solver_view()` already exist and don't need modification.

## Inventory of Call Sites to Migrate

### `solver_time.cpp` (~55 `gpu::dev_ptr` calls)

| Function | Lines | dev_ptr calls | Description |
|----------|-------|--------------|-------------|
| `compute_velocity_filter_3d` | 37-39 | 3 | u, v, w velocity filter |
| `compute_cfl_v_only` | 189 | 1 | v-only CFL check |
| `step_euler` (predictor) | 325-332 | 8 | u,v velocity + conv + diff |
| `step_euler` (corrector) | 368-375 | 8 | u,v velocity + conv + diff |
| `step_euler` (update) | 410-415 | 6 | u,v velocity + conv + diff |
| `step_rk` (stage 1) | 475-486 | 12 | u,v,w velocity + conv + diff |
| `step_rk` (stage 2) | 553-564 | 12 | u,v,w velocity + conv + diff |
| `compute_divergence` | 768-770 | 3 | u, v, div |
| `compute_rhs_poisson` | 801-802, 828-829 | 4 | div, rhs |
| `compute_div_diagnostics` | 927-928 | 2 | rhs, p_corr |
| `compute_div_linf` | 1024 | 1 | div |
| `compute_adaptive_dt_rhs` | 1085 | 1 | rhs |

### `solver.cpp` (~10 `gpu::dev_ptr` calls)

| Function | Lines | dev_ptr calls | Description |
|----------|-------|--------------|-------------|
| `compute_kinetic_energy_device` | 4432-4462 | 3 | u, v, w |
| `compute_divergence_linf_device` | 4517 | 1 | div |
| `compute_max_conv_device` | 4556-4585 | 3 | conv_u, conv_v, conv_w |

---

### Task 1: Migrate `solver_time.cpp` time integration kernels

**Files:**
- Modify: `src/solver_time.cpp` (lines 320-570 — Euler and RK integrators)

This is the biggest batch: 6 blocks of ~8-12 `dev_ptr` calls each (Euler predictor, Euler corrector, Euler update, RK stage 1, RK stage 2). All follow the identical pattern.

- [ ] **Step 1: Migrate `step_euler` predictor block (lines 325-365)**

Replace:
```cpp
const double* u_in_dev = gpu::dev_ptr(velocity_u_ptr_);
double* u_out_dev = gpu::dev_ptr(velocity_star_u_ptr_);
const double* conv_u_dev = gpu::dev_ptr(conv_u_ptr_);
const double* diff_u_dev = gpu::dev_ptr(diff_u_ptr_);
// ... 8 total dev_ptr calls

#pragma omp target ... is_device_ptr(u_in_dev, u_out_dev, conv_u_dev, diff_u_dev, ...)
```

With:
```cpp
auto sv = get_solver_view();
double* u_in = sv.u_face;
double* u_out = sv.u_star_face;
double* conv_u = sv.conv_u;
double* diff_u = sv.diff_u;
// ... just struct member access, no dev_ptr calls

[[maybe_unused]] const int u_sz = velocity_.u_total_size();
[[maybe_unused]] const int v_sz = velocity_.v_total_size();

#pragma omp target ... map(present: u_in[0:u_sz], u_out[0:u_sz], conv_u[0:u_sz], diff_u[0:u_sz], ...)
```

- [ ] **Step 2: Migrate `step_euler` corrector block (lines 368-405)**

Same pattern as step 1 but u_in/u_out are swapped (star→velocity).

- [ ] **Step 3: Migrate `step_euler` update block (lines 410-450)**

Same pattern, fewer pointers (no star velocity).

- [ ] **Step 4: Migrate `step_rk` stage 1 (lines 475-530)**

Same pattern but includes w-velocity (3D). 12 dev_ptr calls → struct access.

- [ ] **Step 5: Migrate `step_rk` stage 2 (lines 553-600)**

Same pattern as stage 1 with swapped in/out.

- [ ] **Step 6: Run tests**

Run: `ctest -L fast --output-on-failure -j4`
Expected: All 47 fast tests pass (time integration is exercised by every flow test)

- [ ] **Step 7: Commit**

```
git add src/solver_time.cpp
git commit -m "Migrate time integration kernels to SolverDeviceView pattern

Replace ~44 gpu::dev_ptr() calls in Euler/RK integrators with
get_solver_view() + map(present:). Matches the pattern already
used in solver_operators.cpp. No behavioral change."
```

---

### Task 2: Migrate `solver_time.cpp` projection kernels

**Files:**
- Modify: `src/solver_time.cpp` (lines 768-1090 — divergence, RHS, diagnostics)

- [ ] **Step 1: Migrate `compute_divergence` (lines 768-795)**

Replace 3 `dev_ptr` calls (u, v, div) with `get_solver_view()`.

- [ ] **Step 2: Migrate `compute_rhs_poisson` (lines 801-860)**

Replace 4 `dev_ptr` calls (div, rhs × 2 code paths) with `get_solver_view()`.

- [ ] **Step 3: Migrate `compute_div_diagnostics` (lines 927-960)**

Replace 2 `dev_ptr` calls (rhs, p_corr) with `get_solver_view()`.

- [ ] **Step 4: Migrate remaining small functions (lines 1024, 1085)**

Replace 2 `dev_ptr` calls (div, rhs).

- [ ] **Step 5: Run tests**

Run: `ctest -L fast --output-on-failure -j4`
Expected: All 47 fast tests pass

- [ ] **Step 6: Commit**

```
git add src/solver_time.cpp
git commit -m "Migrate projection kernels to SolverDeviceView pattern

Replace ~11 gpu::dev_ptr() calls in divergence/RHS/diagnostic
functions with get_solver_view() + map(present:)."
```

---

### Task 3: Migrate `solver_time.cpp` filter and CFL kernels

**Files:**
- Modify: `src/solver_time.cpp` (lines 37-39, 189)

- [ ] **Step 1: Migrate `compute_velocity_filter_3d` (lines 37-39)**

Replace 3 `dev_ptr` calls with `get_solver_view()`. Note: the 2D filter in `solver.cpp:4083` already uses `get_solver_view()`.

- [ ] **Step 2: Migrate `compute_cfl_v_only` (line 189)**

Replace 1 `dev_ptr` call.

- [ ] **Step 3: Run tests**

Run: `ctest -L fast --output-on-failure -j4`
Expected: All 47 fast tests pass

- [ ] **Step 4: Commit**

```
git add src/solver_time.cpp
git commit -m "Migrate filter/CFL kernels to SolverDeviceView pattern

Replace 4 remaining gpu::dev_ptr() calls in solver_time.cpp."
```

---

### Task 4: Migrate `solver.cpp` diagnostic GPU functions

**Files:**
- Modify: `src/solver.cpp` (lines 4432-4590)

- [ ] **Step 1: Migrate `compute_kinetic_energy_device` (lines 4432-4470)**

Replace 3 `dev_ptr` calls (u, v, w).

- [ ] **Step 2: Migrate `compute_divergence_linf_device` (line 4517)**

Replace 1 `dev_ptr` call (div).

- [ ] **Step 3: Migrate `compute_max_conv_device` (lines 4556-4590)**

Replace 3 `dev_ptr` calls (conv_u, conv_v, conv_w).

- [ ] **Step 4: Run full test suite**

Run: `ctest -L fast --output-on-failure -j4`
Expected: All 47 fast tests pass

Run GPU-specific: Build with nvc++ + GPU, run `ctest -L gpu --output-on-failure`

- [ ] **Step 5: Commit**

```
git add src/solver.cpp
git commit -m "Migrate solver diagnostic GPU functions to SolverDeviceView

Replace 7 gpu::dev_ptr() calls in KE/divergence/convection
diagnostic functions. All solver GPU code now uses unified
SolverDeviceView pattern."
```

---

### Task 5: Verify zero `gpu::dev_ptr` in solver hot path

**Files:**
- No code changes — verification only

- [ ] **Step 1: Count remaining `gpu::dev_ptr` calls in solver files**

```bash
grep -c 'gpu::dev_ptr(' src/solver.cpp src/solver_time.cpp src/solver_operators.cpp src/solver_recycling.cpp
```

Expected: 0 in all four files. (Poisson solver files and test files still use `dev_ptr` — that's fine, they're separate subsystems.)

- [ ] **Step 2: Run GPU build + MPI test**

Build with nvc++ + GPU + MPI, run:
```bash
mpiexec -n 2 ./test_mpi_poisson
mpiexec -n 2 ./test_mpi_halo_step
ctest -L fast -j4
```
Expected: All pass

- [ ] **Step 3: Final commit — update CLAUDE.md**

Add to GPU Rules section:
```
### SolverDeviceView Pattern
Solver GPU kernels use `get_solver_view()` which returns a `SolverDeviceView`
POD struct with all field pointers and mesh params. Extract pointers to locals
and use `map(present: ptr[0:size])`. Do NOT call `gpu::dev_ptr()` for solver
fields — the view already holds properly-mapped pointers.
```

---

## What This Does NOT Change

- **Poisson solvers** (`poisson_solver_*.cpp`): Keep using `omp_get_mapped_ptr()` directly — they manage their own GPU buffers with different lifecycles.
- **Turbulence models**: Already use `TurbulenceDeviceView` — no change needed.
- **`gpu::dev_ptr()`**: The function stays in `gpu_utils.hpp` — it's still needed for code outside the solver (tests, standalone tools).
- **Free functions pattern**: Some GPU kernels in `solver_time_kernels_*.cpp` are free functions that take raw pointers. These don't change — they receive pointers extracted from the view.
- **`SolverDeviceView` struct**: No changes needed — it already has all required fields.

## Impact Summary

| Metric | Before | After |
|--------|--------|-------|
| `gpu::dev_ptr()` calls in solver hot path | ~65 | 0 |
| `omp_get_mapped_ptr()` roundtrips per step | ~65 | ~5 (one `get_solver_view` per function, not per kernel) |
| Lines changed | — | ~200 (mechanical replacement) |
| New code | — | 0 new files, 0 new structs |
| Risk | — | LOW (both patterns already proven in production) |
