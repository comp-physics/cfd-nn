# 3D Tensor Basis Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the anisotropic stress divergence and TBNN tensor basis from 2D to 3D, enabling evaluation of tensor-basis turbulence models on the square duct (where secondary corner vortices are the key physics that SST misses but EARSM/TBNN should capture).

**Architecture:** Three subsystems need 3D extension: (1) velocity gradient computation at cell centers (4→9 components), (2) tensor basis computation (4 basis tensors with 3 components → 10 basis tensors with 6 components), (3) tau_div computation in momentum equations (2D → 3D divergence with w-momentum). The existing 2D code continues to work unchanged — the 3D paths activate when `Nz > 1`.

**Tech Stack:** C++17, OpenMP target offload (GPU), staggered MAC grid

**Status:** Task 1 COMPLETE (Mar 28). Tasks 2-5 remaining.

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

### Task 6: Implement Full RSM (SSG/LRR-ω)
**Files:** Create `include/turbulence_rsm.hpp`, `src/turbulence_rsm.cpp`. Modify `include/config.hpp`, `src/config.cpp`, `src/turbulence_baseline.cpp` (factory), `CMakeLists.txt`.

The full Reynolds Stress Model solves 7 transport equations: 6 for individual Reynolds stress components (R_xx, R_xy, R_xz, R_yy, R_yz, R_zz) plus 1 for omega (specific dissipation rate). This is the most expensive classical RANS model (~3-4× SST cost).

Key components:
- **Pressure-strain model**: SSG (Speziale-Sarkar-Gatski) away from walls, LRR (Launder-Reece-Rodi) near walls, blended via F1 (same as SST blending)
- **Wall reflection terms**: Needed for pressure-strain near solid walls
- **Production**: Exact (no modeling needed — P_ij = -R_ik dU_j/dx_k - R_jk dU_i/dx_k)
- **Dissipation**: Isotropic (eps_ij = 2/3 eps delta_ij) using omega equation
- **Diffusion**: Generalized gradient diffusion (Daly-Harlow)
- **GPU**: 6 scalar fields for R_ij + 1 for omega, all on GPU
- `provides_reynolds_stresses() = true`, `uses_transport_equations() = true`

**Purpose**: Provides the "expensive classical" point on the Pareto frontier. If TBNN doesn't beat RSM on duct secondary flows, the NN adds no value over classical methods. If TBRF matches RSM at lower cost, TBRF dominates.

**Reference implementation**: NASA Turbulence Modeling Resource (turbmodels.larc.nasa.gov/rsm-ssglrr.html)

### Task 7: RSM validation on duct
**Files:** No code changes — run and compare

Run duct Re_b=3500 with RSM and compare secondary flow magnitude to EARSM, TBNN, and DNS.
Expected hierarchy: SST(0) < EARSM(partial) ≤ RSM(partial) < TBNN(≈DNS?)

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
