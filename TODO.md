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

### No moving/rotating IBM bodies
- **Scope**: Major feature addition, not a bug. Static bodies only.

### No multi-body IBM support
- **Workaround**: Combine via custom IBMBody subclass with `phi = min(body1.phi, body2.phi)`.

---

## Low

### NVHPC workarounds throughout codebase
- **Status**: Required by current compiler. Remove when nvc++ fixes implicit `this` handling.

---

## Future Work (Numerics / Parallelism / Workflow)

### Implicit time stepping
- Fully explicit except y-diffusion Thomas. No large-dt implicit solves (backward Euler, Crank-Nicolson).
- Would enable much larger dt for stiff problems (RANS at high Re).

### Higher-order time integration (RK4+)
- Only Euler, RK2, RK3. No SSP-RK4, SSPRK(5,4), or multi-step methods (AB2/AB3).

### Moving/deforming IBM bodies
- Static grid, static immersed bodies only. No prescribed motion, FSI, or overset.

### Multi-body IBM
- One body per simulation. Workaround: combine via `phi = min(body1, body2)`.

### CPU-only MPI
- MPI currently requires GPU (`FFT_MPI` needs cuFFT/cuSPARSE). No CPU-only distributed solver.
- MG Schwarz works on CPU+MPI but is 200-600x slower than FFT_MPI.

### Multi-node MPI
- Z-slab decomposition tested within a single node only. Inter-node halo exchange not validated.
- Host-staged MPI (no CUDA-aware MPI) should work multi-node but untested.

### Checkpoint/restart
- `USE_HDF5` build flag exists but save/load simulation state is not implemented.
- Restarting long DNS runs requires re-running from scratch.

### In-situ visualization
- No Catalyst, ADIOS, or ParaView Coprocessing integration.
- VTK output only (post-hoc visualization).

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
| 27 | Distributed GPU FFT Poisson solver | Critical | Replaced O(N²) CPU DFT with cuFFT+MPI+cuSPARSE GPU pipeline. 200-635x faster than MG Schwarz on 2×A100. Direct solve, machine precision. |
| 28 | Deferred FFT_MPI init in set_decomposition | Medium | Solver retries FFT_MPI init when decomposition is set after construction |
| 29 | MPI Poisson benchmark added | — | `bench_mpi_poisson.cpp`: compares FFT_MPI vs MG Schwarz across grid sizes |
| 30 | TBNN tensor basis implemented | Medium | Full Pope (1975) 10-tensor basis for 3D, 4-tensor for 2D, consistent with C++ |
| 31 | MLP/TBNN metadata fixed | Medium | All 4 metadata.json files updated to match actual C++ normalized features |
| 32 | 4-point viscosity face averaging | Medium | Cross-direction faces use proper corner average; same-direction use direct cell value |
| 33 | FFT1D stretched y-grid error message | Medium | Clear runtime_error when FFT1D explicitly requested on stretched grid |
| 34 | Pope EARSM constants configurable | Low | `pope_C1`/`pope_C2` config params (default 0.1), passed through factory to EARSM |

---

## Summary

| Severity | Remaining |
|----------|-----------|
| **Critical** | 0 |
| **High** | 0 |
| **Medium** | 2 |
| **Low** | 1 |
| **Future work** | 8 |
| **Total open** | **3** (+ 8 future) |
| **Completed** | **34** |
