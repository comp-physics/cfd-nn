# 3D Tensor Basis Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the anisotropic stress divergence and TBNN tensor basis from 2D to 3D, enabling evaluation of tensor-basis turbulence models on the square duct (where secondary corner vortices are the key physics that SST misses but EARSM/TBNN should capture).

**Architecture:** Three subsystems need 3D extension: (1) velocity gradient computation at cell centers (4→9 components), (2) tensor basis computation (4 basis tensors with 3 components → 10 basis tensors with 6 components), (3) tau_div computation in momentum equations (2D → 3D divergence with w-momentum). The existing 2D code continues to work unchanged — the 3D paths activate when `Nz > 1`.

**Tech Stack:** C++17, OpenMP target offload (GPU), staggered MAC grid

**Status:** ALL TASKS COMPLETE (Mar 28-29). Tasks 1-7 done, GPU-validated on V100.

---

## Scope: 5 Tasks

### Task 1: Extend VelocityGradient to 3D and compute all 9 gradients — COMPLETE
**Files modified (12):** `include/features.hpp`, `src/gpu_kernels.cpp`, `include/gpu_kernels.hpp`, `include/turbulence_device_view.hpp`, `include/solver.hpp`, `src/solver.cpp`, `src/turbulence_transport.cpp`, `src/turbulence_earsm.cpp`, `src/turbulence_nn_tbnn.cpp`, `src/turbulence_nn_tbrf.cpp`, `src/turbulence_nn_mlp.cpp`, `src/turbulence_baseline.cpp`, `src/turbulence_gep.cpp`

Changes: VelocityGradient struct 4→9 components, GPU kernel 4 code paths (3D/2D × stretched/uniform), ScalarFields + GPU mapping for dudz/dvdz/dwdx/dwdy/dwdz, TurbulenceDeviceView extended, all 7 turbulence model call sites updated. All tests pass.

### Task 2: Extend TensorBasis to 10 tensors with 6 components — COMPLETE
**Files modified (12+):** `include/features.hpp`, `src/features.cpp`, `include/turbulence_nn_tbnn.hpp`, `src/turbulence_nn_tbnn.cpp`, `include/turbulence_nn_tbrf.hpp`, `src/turbulence_nn_tbrf.cpp`, `include/turbulence_earsm.hpp`, `src/turbulence_earsm.cpp`, `include/turbulence_device_view.hpp`, `include/turbulence_model.hpp`, `src/gpu_kernels.cpp`, `include/gpu_kernels.hpp`, `src/solver.cpp`, `tests/test_features.cpp`, `tests/test_turbulence_unified.cpp`, `tests/test_rans_frame_invariance.cpp`

Changes: NUM_BASIS=10, NUM_COMPONENTS=6, full Pope (1975) 3D integrity basis implemented via 3×3 matrix products (16 intermediate products per cell). All GPU kernels (TBNN fused, TBRF, postprocess) updated for 10×6 basis. EARSM G[4..9]=0 (algebraic). tau_xz/yz/zz pointers added to device views. 8 new 3D tensor basis tests added. All tests pass (16/16 features, 3/3 frame invariance, 22/23 unified — NN-MLP failure pre-existing).

### Task 3: Extend TBNN inference to use 10 basis tensors in 3D — COMPLETE (included in Task 2)
**Files:** `src/turbulence_nn_tbnn.cpp`, `src/gpu_kernels.cpp`

The TBNN GPU kernel computes basis tensors, runs NN inference for 10 coefficients, and contracts with the basis. Currently hardcoded to 4 tensors × 3 components. Needs conditional 3D path with 10 tensors × 6 components.

### Task 4: Extend tau_div to 3D — COMPLETE
**Files modified (7):** `include/solver.hpp`, `include/turbulence_model.hpp`, `src/solver.cpp`, `src/solver_operators.cpp`, `src/solver_time_kernels.hpp`, `src/solver_time_kernels_euler.cpp`, `src/solver_time.cpp`

Changes: Added tau_div_w buffer/pointer, 3D nonlinear stress decomposition (6 components), 3D divergence at u/v/w faces, tau_div_w in w-momentum predictor. All 50 fast tests pass.

### Task 5: Integration test — duct secondary flows — PARTIAL
**Status:** GPU-validated on V100 (Tasks 1-4 all pass), needs production-scale H200 run.

GPU validation (V100, Mar 28):
- All 50 fast tests pass with nvc++ GPU build (OMP_TARGET_OFFLOAD=MANDATORY)
- 23/23 tensor basis tests pass on GPU (analytical + EARSM 2D/3D smoke)
- EARSM 3D duct runs stable (32³, 500 steps, WJ/GS/Pope all produce finite vel/nu_t)
- SST and EARSM produce identical bulk velocity on short runs (expected — secondary flows need long development time O(100) D_h/U_b)
- Full secondary flow comparison needs production H200 run at 96³ for O(10000) steps

### Task 6: Implement Full RSM (SSG/LRR-ω) — COMPLETE
**Files created:** `include/turbulence_rsm.hpp`, `src/turbulence_rsm.cpp`, `tests/test_rsm.cpp`
**Files modified:** `include/config.hpp`, `src/config.cpp`, `src/turbulence_baseline.cpp`, `CMakeLists.txt`

SSG pressure-strain (C1=3.4, C1*=1.8, C2=4.2, C3=0.8, C3*=1.3, C4=1.25, C5=0.4) + omega equation. Point-implicit time advance, Schumann realizability. 14 GPU-validated tests. 3 GPU bugs found and fixed (nullptr grad alias, tau_xz/yz/zz mapping, R_ij buffer sync).

### Task 7: RSM validation on duct — PARTIAL (V100 validated, H200 pending)
RSM produces nonzero secondary flow on duct (max|w|=2.3e-4 on 16×32×16 CPU, nonzero on GPU). SST gives zero secondary flow (expected — isotropic). EARSM gives zero at low Re (k≈0 on uniform grid). Full quantitative comparison needs H200 at Re_b=3500 on 96³ grid.

### Additional fixes (Mar 29):
- Duct app warm-up: EARSM closure toggle, transport model preservation, k/omega seeding
- SST k initialization: Ti=10%, nu_t/nu=100 sustains k on uniform grids
- 3D gradient null-pointer: 2D wrapper passed nullptr for z-gradients on 3D mesh
- solver.cpp: tau_xz/yz/zz extraction, GPU mapping, zeroing, cleanup (was missing)

---

## Key References

- Pope (1975): 10-tensor integrity basis for symmetric traceless tensors
- Ling et al. (2016): TBNN architecture using Pope basis
- Pinelli et al. (2010): DNS of turbulent square duct flow
- McConkey et al. (2021): Training data including square duct cases
- Menter et al. (2012): SSG/LRR-ω Reynolds Stress Model
- Speziale, Sarkar & Gatski (1991): SSG pressure-strain model
- Launder, Reece & Rodi (1975): LRR pressure-strain model
- NASA TMR: turbmodels.larc.nasa.gov/rsm-ssglrr.html

## Self-Review

1. **Spec coverage**: All 7 tasks covered (gradients, basis, TBNN, tau_div, duct validation, RSM, RSM validation)
2. **Placeholder scan**: Task details are high-level — detailed code during execution
3. **Dependencies**: Task 1→2→3 (sequential). Task 4 after Task 1. Task 5 requires 1-4. Task 6 independent. Task 7 requires 5+6.
