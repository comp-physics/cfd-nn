# TODO — Codebase Audit (March 2026)

Open issues from deep codebase audit + ongoing development. Completed items at bottom.

---

## Critical

*No critical items remaining.*

---

## High

*No high-priority items remaining.*

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
- **Scope**: Major feature addition, not a bug. Static bodies only.

### No multi-body IBM support
- **Workaround**: Combine via custom IBMBody subclass with `phi = min(body1.phi, body2.phi)`.

---

## Low

### NVHPC workarounds throughout codebase
- **Status**: Required by current compiler. Remove when nvc++ fixes implicit `this` handling.

### Recirculation spike detection stub
- **File**: `src/solver_turbulence_diagnostics.cpp:546-568`
- **Fix**: Remove the function (always returns false, never used).

### Pope EARSM constants hardcoded
- **File**: `src/turbulence_earsm.cpp:652`
- **Fix**: Add `pope_C1` and `pope_C2` config parameters.

---

## Completed ✅

| # | Item | Severity | Details |
|---|------|----------|---------|
| 1 | FFT_MPI wired into solver factory | Critical | Factory case, diagnostics, GPU dispatch |
| 2 | Multi-GPU MPI solver fixed | Critical | 5 fixes: z-BC skip, velocity_star_ halo, global mean_div, Schwarz pressure, halo buffer sizing |
| 3 | GPU↔host staged halo exchange | Critical | `exchange_host_staged()` with `target update from/to` |
| 4 | Auto GPU assignment for MPI | Critical | `CUDA_VISIBLE_DEVICES=local_rank` inside solver constructor |
| 5 | CUDA Graphs disabled for MPI | Critical | Auto-disabled when Schwarz outer loop is active |
| 6 | MPI halo exchange wired into step | Critical | Euler + RK paths, pressure + velocity halos |
| 7 | FFT_MPI CPU single-rank error | Critical | Clear error (FFTPoissonSolver is GPU-only) |
| 8 | HDF5 checkpoint dead code removed | Critical | Deleted header, impl, test (442 LOC) |
| 9 | SOR solver removed | Medium | Deleted `poisson_solver.cpp` (370 LOC), `PoissonSolver` class |
| 10 | GPU baseline populated | High | `baseline_gpu.json` from RTX 6000 run |
| 11 | GPU CI perf suite removed | High | Unreliable across GPU types |
| 12 | CI auto-cancel on push | — | `cancel-in-progress: true` on both workflows |
| 13 | CI timeouts increased | — | 240→360 min, 30→60 sacct retries |
| 14 | Orphaned CI scripts removed | — | 3 dead scripts deleted |
| 15 | MPI halo step test added | — | Decomposed mesh, 1/2/4 ranks, verified on 2×L40S + 2×H200 |
| 16 | Old recycling commented code removed | Low | Cleaned dead code |
| 17 | Poisson test thresholds fixed | — | Migrated SOR tests to MG with proper tolerances |
| 18 | Dynamic Smagorinsky implemented | High | Full Germano procedure, 3-pass GPU, plane-averaged Cs² |
| 19 | Dynamic Smag nvc++ workaround | — | Split into 5 files with minimal header to avoid nvc++ crash |
| 20 | Dynamic Smag MPI allreduce | — | LM/MM plane sums allreduced for multi-GPU Cs² |
| 21 | Statistics MPI allreduce | — | Two-pass mean/fluctuation with allreduce between passes |
| 22 | Recycling inflow MPI support | — | MPI_Allgather for global plane assembly, shift with Nz_global |
| 23 | TurbulenceDeviceView extracted | — | Minimal 97-line header avoids 107K preprocessed lines in GPU kernels |
| 24 | FFT_MPI GPU distributed path | Critical | Host-staged: `target update from/to` + existing CPU distributed solve |
| 25 | AR1 recycling filter on GPU | High | GPU kernel with filter state buffers, fixes leak, MPI-compatible |
| 26 | O4 spatial warnings | High | All silent O2 fallbacks now warn; documented as advection-only |

---

## Summary

| Severity | Remaining |
|----------|-----------|
| **Critical** | 0 |
| **High** | 0 |
| **Medium** | 6 |
| **Low** | 3 |
| **Total open** | **9** |
| **Completed** | **26** |
