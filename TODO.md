# TODO — Codebase Audit (March 2026)

Comprehensive list of broken, stub, incomplete, and missing features identified by deep codebase audit. Organized by severity. Completed items moved to the bottom.

---

## Critical — Broken or Stub Code

### FFT_MPI GPU distributed path unimplemented
- **File**: `src/poisson_solver_fft_mpi.cpp:141-144`
- **Problem**: Multi-rank `solve_device()` throws. Single-rank GPU works (delegates to serial FFTPoissonSolver). Single-rank CPU path correctly errors since FFTPoissonSolver has no CPU solve.
- **Fix**: Implement distributed GPU FFT using CUDA-aware MPI or cuFFTMp.

---

## High — Misleading or Incomplete Features

### DynamicSmagorinsky is a stub (hardcoded Cs=0.17)
- **File**: `src/turbulence_les.cpp:555-572`
- **Problem**: No Germano procedure implemented. `update()` unconditionally sets `Cs_dyn_ = 0.17` (identical to static Smagorinsky). Prints a one-time stderr warning, then runs silently with the constant coefficient.
- **Fix**: Implement the Germano procedure (test filter at 2Δ, compute L_ij, solve for Cs²) or rename model to make it clear this is a constant-Cs fallback.

### 4th-order spatial discretization is partial
- **File**: `src/solver_operators.cpp`
- **Problem**:
  - O4 convection: **3D only**. 2D falls back to O2.
  - O4 diffusion: **Not implemented at all**.
  - O4 divergence: 3D uniform-y only.
  - O4 pressure gradient: 3D only.
  - MG Poisson: Explicitly errors if O4 requested.
- **Fix**: Implement missing O4 operators or document `space_order=4` as 3D-convection-only.

### AR1 recycling filter skipped on GPU
- **File**: `src/solver_recycling.cpp:599-602`
- **Problem**: GPU is the primary target but the AR1 temporal filter is CPU-only and skipped in GPU builds.
- **Fix**: Allocate filter buffers on GPU and implement the AR1 filter in an OpenMP target kernel.

### Z-direction BCs restricted to Periodic and NoSlip
- **File**: `src/solver_operators.cpp:277-283`
- **Problem**: Inflow/Outflow BCs in z-direction throw `std::runtime_error`.
- **Fix**: Implement z-direction Inflow/Outflow BCs or document as unsupported.

### Recirculation spike detection is a stub
- **File**: `src/solver_turbulence_diagnostics.cpp:546-568`
- **Problem**: `has_recirculation_spike()` always returns `false`. Parameters unused.
- **Fix**: Implement actual spectral spike detection or remove the function.

---

## Medium — Incomplete but Functional

### FFT1D cannot handle stretched y-grids
- **File**: `include/poisson_solver_fft1d.hpp:156`
- **Problem**: Internal 2D MG uses uniform `1/(dy*dy)`. Auto-selection skips FFT1D for stretched-y, but explicit `--poisson fft1d` on a stretched-y duct will error.
- **Fix**: Implement stretched-y support in FFT1D's 2D MG or add a clear error message.

### TBNN training script incomplete
- **File**: `scripts/train_tbnn_mcconkey.py:184`
- **Problem**: `compute_tensor_basis_from_data()` is a stub. Cannot train new TBNN models on the McConkey dataset.
- **Fix**: Implement tensor basis computation (T¹ through T⁴) from strain/rotation tensors.

### MLP feature definitions mismatch between code and metadata
- **Files**: `src/features.cpp:96-112`, `data/models/*/metadata.json`
- **Problem**: C++ computes normalized features but metadata.json claims raw features. Confusing for new models.
- **Fix**: Align metadata.json with actual C++ feature definitions.

### Pope EARSM constants hardcoded
- **File**: `src/turbulence_earsm.cpp:652`
- **Problem**: C1=0.1, C2=0.1 hardcoded. Not configurable via config file.
- **Fix**: Add `pope_C1` and `pope_C2` config parameters.

### Variable viscosity face averaging uses 2-point instead of 4-point
- **File**: `include/solver_kernels.hpp:450,485`
- **Problem**: Should use 4-point corner averaging at y-faces for sharp nu_t gradients.
- **Fix**: Implement 4-point face-corner averaging for diffusion coefficients.

### No moving/rotating IBM bodies
- **Files**: `include/ibm_geometry.hpp`, `src/ibm_forcing.cpp`
- **Problem**: All IBM bodies are static. No runtime geometry update mechanism.
- **Scope**: Major feature addition, not a bug.

### No multi-body IBM support
- **File**: `include/solver.hpp`
- **Problem**: Single `ibm_` pointer. Only one body at a time.
- **Workaround**: Combine via custom IBMBody subclass with `phi = min(body1.phi, body2.phi)`.

### Q-criterion VTK output recomputes gradients
- **File**: `src/solver_vtk.cpp:649-650`
- **Problem**: Redundant gradient computation in output loop.
- **Fix**: Precompute and cache gradient tensor.

### MG device pointer optimization not used
- **File**: `include/poisson_solver_multigrid.hpp:58`
- **Problem**: Uses `map(present:)` instead of more efficient `is_device_ptr` for `omp_target_alloc` pointers.

---

## Low — Minor / Informational

### NVHPC workarounds throughout codebase
- **Files**: `src/solver_time.cpp`, `src/solver.cpp`, `src/solver_recycling.cpp`
- **Status**: Required by current compiler. Remove when nvc++ fixes implicit `this` handling.

### Simplified 2D invariants in feature computation
- **File**: `src/features.cpp:162,172`
- **Status**: By design — 2D meshes don't have all 3D invariant components.

### Upwind advection simplified near boundaries
- **File**: `include/solver_kernels.hpp:1973`
- **Status**: Standard practice for upwind schemes at domain edges.

---

## Completed ✅

| # | Item | Severity | Details |
|---|------|----------|---------|
| 1 | FFT_MPI wired into solver factory | Critical | Factory case, diagnostics, GPU dispatch, solver name lookup |
| 2 | GPU baseline populated | High | `baseline_gpu.json` from RTX 6000, single baseline across all GPUs |
| 3 | SOR solver removed | Medium | Deleted `poisson_solver.cpp` (370 LOC), removed `PoissonSolver` class and `poisson_omega` config |
| 4 | Old recycling commented code removed | Low | Cleaned dead code from `solver_recycling.cpp` |
| 5 | GPU CI perf gates scaled by CC | High | V100 4x, A100/L40S 2x, H100/H200 1x thresholds |
| 6 | FFT_MPI CPU single-rank error | Critical | Clear error message (architectural limitation: FFTPoissonSolver GPU-only) |
| 7 | HDF5 checkpoint dead code removed | Critical | Deleted header, impl, test (442 LOC total) |
| 8 | MPI halo exchange wired into step | Critical | Euler + RK paths, pressure + velocity halos, CPU + GPU |

---

## Summary

| Severity | Total | Done | Remaining |
|----------|-------|------|-----------|
| **Critical** | 5 | 4 | 1 |
| **High** | 7 | 2 | 5 |
| **Medium** | 11 | 1 | 10 |
| **Low** | 4 | 1 | 3 |
| **Total** | **27** | **8** | **19** |

**Codebase health**: Core solver, turbulence models (14/15), IBM, Poisson solvers (5/6), and NN inference are production-quality. Main gaps: one critical MPI item (FFT_MPI distributed GPU), DynamicSmagorinsky stub, partial O4 spatial, and several medium-priority feature gaps.
