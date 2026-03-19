# TODO — Codebase Audit (March 2026)

Open issues from deep codebase audit + ongoing development. Completed items at bottom.

---

## Critical

### FFT_MPI GPU distributed path unimplemented
- **File**: `src/poisson_solver_fft_mpi.cpp:141-144`
- **Problem**: Multi-rank `solve_device()` throws. Single-rank GPU works. CPU distributed path works.
- **Fix**: Implement distributed GPU FFT using CUDA-aware MPI or cuFFTMp.
- **Note**: The current additive Schwarz approach (local MG + pressure halo exchange) works but converges slower than a global FFT solve would.

---

## High

### DynamicSmagorinsky is a stub (hardcoded Cs=0.17)
- **File**: `src/turbulence_les.cpp:555-572`
- **Problem**: No Germano procedure. Identical to static Smagorinsky with a warning on first call.
- **Fix**: Implement Germano procedure or rename to `smagorinsky_constant`.

### 4th-order spatial discretization is partial
- **File**: `src/solver_operators.cpp`
- **Problem**: O4 convection is 3D-only. O4 diffusion not implemented. O4 divergence 3D uniform-y only. MG errors on O4.
- **Fix**: Document `space_order=4` as 3D-convection-only, or implement missing operators.

### AR1 recycling filter skipped on GPU
- **File**: `src/solver_recycling.cpp:599-602`
- **Problem**: GPU is the primary target but the AR1 temporal filter is skipped in GPU builds.
- **Fix**: Implement AR1 filter in an OpenMP target kernel.

---

## Medium

### FFT1D cannot handle stretched y-grids
- **File**: `include/poisson_solver_fft1d.hpp:156`
- **Problem**: Internal 2D MG uses uniform `1/(dy*dy)`. Auto-selection correctly skips it; explicit request will error.
- **Fix**: Add a clear error message, or implement stretched-y support in FFT1D's 2D MG.

### TBNN training script incomplete
- **File**: `scripts/train_tbnn_mcconkey.py:184`
- **Problem**: `compute_tensor_basis_from_data()` is a stub.
- **Fix**: Implement tensor basis computation from strain/rotation tensors.

### MLP feature definitions mismatch between code and metadata
- **Files**: `src/features.cpp`, `data/models/*/metadata.json`
- **Problem**: C++ computes normalized features but metadata claims raw features.
- **Fix**: Align metadata.json with actual C++ feature definitions.

### Variable viscosity face averaging uses 2-point instead of 4-point
- **File**: `include/solver_kernels.hpp:450,485`
- **Problem**: Should use 4-point corner averaging for sharp nu_t gradients near walls.

### No moving/rotating IBM bodies
- **Files**: `include/ibm_geometry.hpp`, `src/ibm_forcing.cpp`
- **Scope**: Major feature addition, not a bug. Static bodies only.

### No multi-body IBM support
- **File**: `include/solver.hpp`
- **Problem**: Single `ibm_` pointer.
- **Workaround**: Combine via custom IBMBody subclass with `phi = min(body1.phi, body2.phi)`.

---

## Low

### NVHPC workarounds throughout codebase
- **Status**: Required by current compiler. Remove when nvc++ fixes implicit `this` handling.

### Recirculation spike detection stub
- **File**: `src/solver_turbulence_diagnostics.cpp:546-568`
- **Problem**: `has_recirculation_spike()` always returns `false`.
- **Fix**: Remove the function (it's never meaningfully used).

### Pope EARSM constants hardcoded
- **File**: `src/turbulence_earsm.cpp:652`
- **Problem**: C1=0.1, C2=0.1 not configurable.
- **Fix**: Add `pope_C1` and `pope_C2` config parameters.

---

## Completed ✅

| # | Item | Severity | Details |
|---|------|----------|---------|
| 1 | FFT_MPI wired into solver factory | Critical | Factory case, diagnostics, GPU dispatch |
| 2 | Multi-GPU MPI solver fixed | Critical | 5 fixes: z-BC skip, velocity_star_ halo, global mean_div, Schwarz pressure, halo buffer sizing |
| 3 | GPU↔host staged halo exchange | Critical | `exchange_host_staged()` with `target update from/to` — no CUDA-aware MPI needed |
| 4 | Auto GPU assignment for MPI | Critical | `CUDA_VISIBLE_DEVICES=local_rank` inside solver constructor |
| 5 | CUDA Graphs disabled for MPI | Critical | Auto-disabled when Schwarz outer loop is active |
| 6 | MPI halo exchange wired into step | Critical | Euler + RK paths, pressure + velocity halos |
| 7 | FFT_MPI CPU single-rank error | Critical | Clear error (FFTPoissonSolver is GPU-only) |
| 8 | HDF5 checkpoint dead code removed | Critical | Deleted header, impl, test (442 LOC) |
| 9 | SOR solver removed | Medium | Deleted `poisson_solver.cpp` (370 LOC), `PoissonSolver` class, `poisson_omega` config |
| 10 | GPU baseline populated | High | `baseline_gpu.json` from RTX 6000 run |
| 11 | GPU CI perf suite removed | High | Unreliable across GPU types; manual benchmarking via scripts |
| 12 | CI auto-cancel on push | — | `cancel-in-progress: true` on both CPU and GPU CI |
| 13 | CI timeouts increased | — | 240→360 min GitHub Actions, 30→60 sacct retries |
| 14 | Orphaned CI scripts removed | — | gpu_perf_suite.sh, gpu_ci_perf.sbatch.template, gpu_correctness_suite.sh |
| 15 | MPI halo step test added | — | `test_mpi_halo_step`: decomposed mesh, 1/2/4 ranks, verified on 2×L40S GPU |
| 16 | Old recycling commented code removed | Low | Cleaned dead code from `solver_recycling.cpp` |
| 17 | Poisson test thresholds fixed | — | Migrated from SOR to MG with proper `tol_rhs`/`tol_rel` |

---

## Summary

| Severity | Remaining |
|----------|-----------|
| **Critical** | 1 (FFT_MPI distributed GPU) |
| **High** | 3 |
| **Medium** | 6 |
| **Low** | 3 |
| **Total open** | **13** |
| **Completed** | **17** |
