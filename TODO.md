# TODO — Codebase Audit (March 2026)

Comprehensive list of broken, stub, incomplete, and missing features identified by deep codebase audit. Organized by severity.

---

## Critical — Broken or Stub Code

### ~~FFT_MPI Poisson solver never instantiated~~ ✅ DONE
- Wired FFT_MPI into solver factory, diagnostics, and GPU dispatch.

### FFT_MPI GPU path unimplemented
- **File**: `src/poisson_solver_fft_mpi.cpp:133-134`
- **Problem**: `solve_device()` throws `std::runtime_error("GPU path not yet implemented (requires CUDA-aware MPI)")`. Multi-GPU MPI runs cannot use FFT Poisson solver.
- **Fix**: Implement distributed GPU FFT using CUDA-aware MPI or cuFFTMp.

### FFT_MPI CPU single-rank path throws
- **File**: `src/poisson_solver_fft_mpi.cpp:107-117`
- **Problem**: `solve()` on a single rank throws instead of delegating to the serial FFT solver's CPU path. Comment says "placeholder".
- **Fix**: Delegate to serial FFT solver or implement host path.

### HDF5 checkpoint is dead code
- **File**: `src/checkpoint.cpp`
- **Problem**: `write_checkpoint()` and `read_checkpoint()` exist but are **never called** from the solver. No config options for checkpoint/restart. Without `USE_HDF5`, both functions throw. With `USE_HDF5`, the functions exist but nothing invokes them.
- **Fix**: Either integrate into solver (add `checkpoint_interval` config, call from time loop) or remove the dead code.

### MPI halo exchange never called in solver loop
- **Files**: `src/solver.cpp`, `src/halo_exchange.cpp`
- **Problem**: `Decomposition` is set via `set_decomposition()` and MPI allreduces work for scalars (residuals, statistics), but `halo_exchange_->exchange()` / `exchange_device()` are **never invoked** during `step()`. The halo exchange infrastructure is tested and correct but disconnected from the actual solve.
- **Fix**: Add halo exchange calls after BC application in `step()` for z-direction ghost cells when `nprocs > 1`.

---

## High — Misleading or Incomplete Features

### DynamicSmagorinsky is a stub (hardcoded Cs=0.17)
- **File**: `src/turbulence_les.cpp:555-572`
- **Problem**: No Germano procedure implemented. `update()` unconditionally sets `Cs_dyn_ = 0.17` (identical to static Smagorinsky). Prints a one-time stderr warning, then runs silently with the constant coefficient. Header claims "Germano procedure with test filter" but code does not compute test-filtered quantities, Germano identity, or Lilly optimization.
- **Fix**: Implement the Germano procedure (test filter at 2Δ, compute L_ij, solve for Cs²) or rename model to make it clear this is a constant-Cs fallback.

### 4th-order spatial discretization is partial
- **File**: `src/solver_operators.cpp`
- **Problem**:
  - O4 convection: **3D only**. 2D prints a one-time warning and falls back to O2 (line 765-773).
  - O4 diffusion: **Not implemented at all**. Only O2 stencils exist (lines 883-1015).
  - O4 divergence: 3D uniform-y only. Stretched-y falls back to O2 (line 1099).
  - O4 pressure gradient: 3D only. 2D not implemented, no warning.
  - MG Poisson: Explicitly errors if O4 requested (solver.cpp:522).
- **Fix**: Implement O4 diffusion kernels, O4 2D advection, and stretched-y O4 divergence — or document `space_order=4` as 3D-convection-only.

### AR1 recycling filter skipped on GPU
- **File**: `src/solver_recycling.cpp:599-602`
- **Problem**: Comment says "Would need inlet_u_filt_ptr_ etc. on GPU — skipping for now". Since GPU is the primary development target, temporal filtering of the recycled inflow is effectively dead.
- **Fix**: Allocate filter buffers on GPU and implement the AR1 filter in an OpenMP target kernel.

### Z-direction BCs restricted to Periodic and NoSlip
- **File**: `src/solver_operators.cpp:277-283`
- **Problem**: Inflow/Outflow BCs in z-direction throw `std::runtime_error`. Only Periodic and NoSlip are supported.
- **Fix**: Implement z-direction Inflow/Outflow BCs or document as unsupported.

### GPU baseline test data all placeholders
- **File**: `tests/baselines/baseline_gpu.json`
- **Problem**: All 30+ metrics are `"TODO"` strings (`"TODO: populate from GPU build"`). GPU CI regression testing has no baselines to compare against.
- **Fix**: Run GPU test suite once and populate with actual values.

### Recirculation spike detection is a stub
- **File**: `src/solver_turbulence_diagnostics.cpp:546-568`
- **Problem**: `has_recirculation_spike()` always returns `false`. Parameters `x_recycle` and `U_bulk` are unused (marked `/*param*/`). Used in validation but never detects anything.
- **Fix**: Implement actual spectral spike detection or remove the function.

---

## Medium — Incomplete but Functional

### ~~SOR solver is orphaned dead code~~ ✅ DONE
- Deleted `src/poisson_solver.cpp` (370 LOC) and removed `PoissonSolver` class.
- Removed `poisson_omega` from config. Migrated tests to `MultigridPoissonSolver`.

### FFT1D cannot handle stretched y-grids
- **File**: `include/poisson_solver_fft1d.hpp:156`
- **Problem**: Internal 2D MG solver uses uniform `1/(dy*dy)` coefficients. Auto-selection correctly skips FFT1D for stretched-y, but explicitly requesting `--poisson fft1d` on a stretched-y duct will error. No hybrid FFT+stretched-MG path exists.
- **Fix**: Implement stretched-y support in FFT1D's internal 2D MG (use mesh yLap coefficients) or add a clear error message.

### TBNN training script incomplete
- **File**: `scripts/train_tbnn_mcconkey.py:184`
- **Problem**: `compute_tensor_basis_from_data()` is a stub. Users cannot train new TBNN models without implementing this function. The training script generates synthetic data as a fallback but cannot process the real McConkey dataset's velocity gradients into tensor basis functions.
- **Fix**: Implement tensor basis computation (T¹ through T⁴) from strain/rotation tensors.

### MLP feature definitions mismatch between code and metadata
- **Files**: `src/features.cpp:96-112`, `data/models/*/metadata.json`
- **Problem**: C++ computes normalized features (|S|·δ/u_ref, |Ω|·δ/u_ref, y/δ, |Ω|/|S|, |S|·δ²/ν, |u|/u_ref) but metadata.json claims raw features (S_mag, Omega_mag, y_norm, k, omega, u_mag). The training script uses yet another set (raw from dataset). Works because z-score normalization files compensate, but is confusing and error-prone for new models.
- **Fix**: Align metadata.json with actual C++ feature definitions, or add a feature version field that selects the computation.

### Pope EARSM constants hardcoded
- **File**: `src/turbulence_earsm.cpp:652`
- **Problem**: Pope model constants C1=0.1, C2=0.1 are hardcoded in the factory call. TODO comment says "expose via pope_model". Not configurable via config file or CLI.
- **Fix**: Add `pope_C1` and `pope_C2` config parameters.

### Variable viscosity face averaging uses 2-point instead of 4-point
- **File**: `include/solver_kernels.hpp:450,485`
- **Problem**: Two TODO comments note that variable-viscosity diffusion should use 4-point corner averaging at y-faces but currently uses simpler 2-point center averaging. Affects accuracy when nu_t varies sharply (e.g., near walls with RANS models).
- **Fix**: Implement 4-point face-corner averaging for diffusion coefficients.

### No moving/rotating IBM bodies
- **Files**: `include/ibm_geometry.hpp`, `src/ibm_forcing.cpp`
- **Problem**: All IBM bodies are static. `classify_cells()` and `compute_weights()` run once at initialization. No mechanism to update geometry at runtime. Would require re-classification, weight recomputation, and GPU buffer updates each step.
- **Scope**: Major feature addition, not a bug.

### No multi-body IBM support
- **File**: `include/solver.hpp:828`
- **Problem**: Solver stores a single `ibm_` pointer. Only one IBM body can be active at a time.
- **Workaround**: Users can combine geometries into a custom IBMBody subclass using `phi = min(body1.phi, body2.phi)`.
- **Fix**: Support `std::vector<IBMForcing*>` or a composite body class.

### Q-criterion VTK output recomputes gradients
- **File**: `src/solver_vtk.cpp:649-650`
- **Problem**: TODO says "For large grids, consider precomputing the 3x3 gradient tensor per cell to avoid redundant lambda calls in the output loop."
- **Fix**: Precompute and cache gradient tensor for Q-criterion output.

### MG device pointer optimization not used
- **File**: `include/poisson_solver_multigrid.hpp:58`
- **Problem**: Comment says "For true device pointers (omp_target_alloc), use is_device_ptr instead (not implemented)". Current approach uses `map(present:)` which works but is less efficient.
- **Fix**: Use `is_device_ptr` clause for `omp_target_alloc`-allocated pointers.

---

## Low — Minor / Informational

### Old recycling approach commented out
- **File**: `src/solver_recycling.cpp:910-915`
- **Problem**: Old inlet application code commented out with explanation ("creates divergence"). Replaced by current divergence-correcting approach.
- **Fix**: Remove commented code block.

### NVHPC workarounds throughout codebase
- **Files**: `src/solver_time.cpp` (8+), `src/solver.cpp` (4+), `src/solver_recycling.cpp`
- **Problem**: Pattern of copying member variables to locals before GPU kernels to avoid implicit `this` transfer. Documented with "nvc++ workaround" comments. Functional but clutters code.
- **Status**: Required by current compiler. Remove when nvc++ fixes implicit this handling.

### Simplified 2D invariants in feature computation
- **File**: `src/features.cpp:162,172`
- **Problem**: TBNN features use "simplified" 2D invariants (comments note this). Full 3D invariant set not computed for 2D meshes.
- **Status**: By design — 2D meshes don't have all 3D invariant components.

### Upwind advection simplified near boundaries
- **File**: `include/solver_kernels.hpp:1973`
- **Problem**: 3D 2nd-order upwind uses 1st-order near boundaries.
- **Status**: Standard practice for upwind schemes at domain edges.

---

## Summary

| Severity | Count | Examples |
|----------|-------|---------|
| **Critical** | 5 | FFT_MPI not wired, checkpoint dead code, halo exchange unused |
| **High** | 6 | DynamicSmag stub, O4 partial, AR1 filter skipped on GPU |
| **Medium** | 11 | SOR dead code, FFT1D stretched-y, TBNN training stub |
| **Low** | 4 | Commented code, compiler workarounds |
| **Total** | **26** | |

**Codebase health**: ~150 LOC of stubs/dead code out of ~43,000 LOC (0.35%). Core solver, turbulence models (14/15), IBM, Poisson solvers (5/6), and NN inference are production-quality. Main gaps are in MPI integration (infrastructure exists but not connected) and a few misleading feature claims (DynamicSmag, O4, checkpoint).
