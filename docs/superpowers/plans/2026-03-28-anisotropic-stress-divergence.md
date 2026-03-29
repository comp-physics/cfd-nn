# Anisotropic Reynolds Stress Divergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the anisotropic Reynolds stress divergence `div(tau_ij)` to the momentum equations so tensor-basis models (TBNN, TBRF, EARSM, GEP) produce different flow predictions from scalar eddy viscosity (SST). Currently `tau_ij` is computed but never applied.

**Status: IMPLEMENTED (Mar 28, 2026).** All tasks complete for 2D. Key results:
- Cylinder: SST Cd=1.72, EARSM Cd=1.64 (-5%), TBNN Cd=1.80 (+3%), TBRF Cd=1.76, GEP Cd=1.66
- Hills Re=5600: EARSM U_b=0.80, SST U_b=0.73, k-omega U_b=0.65 — three distinct models
- Critical fix for hills: must restore SST nu_t after EARSM computes tau_ij (EARSM-derived nu_t destabilizes explicit solver near separation)

**Architecture:** Proper decomposition method (Thompson 2019): tau_nl = tau_ij - 2*nu_t_SST*S_ij, then div(tau_nl) as explicit source term. SST's nu_t stays in the diffusion operator (stable). Only the nonlinear correction enters as a source. For EARSM on separated flows: restore SST nu_t after closure computes tau_ij; disable closure during warm-up.

**Tech Stack:** C++17, OpenMP target offload (GPU), staggered MAC grid, fractional-step projection

**Tech Stack:** C++17, OpenMP target offload (GPU), staggered MAC grid, fractional-step projection

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `include/solver.hpp` | Modify | Add `tau_div_u_ptr_`, `tau_div_v_ptr_` pointers; add `compute_tau_divergence()` declaration |
| `include/turbulence_model.hpp` | Modify | Add `tau_xx`, `tau_xy`, `tau_yy` to `SolverDeviceView` |
| `src/solver_operators.cpp` | Modify | Add `compute_tau_divergence()` function |
| `src/solver_time.cpp` | Modify | Add `+ tau_div` to the predictor step kernels |
| `src/solver.cpp` | Modify | Initialize tau_div pointers, add to GPU mapping, add to `get_solver_view()` |
| `tests/test_ghost_cell_ibm.cpp` | Modify | Add test: EARSM Cd differs from SST with tau_div enabled |

---

## Task 1: Add tau_div storage and SolverDeviceView fields

**Files:**
- Modify: `include/solver.hpp` — add tau_div pointers and VectorField
- Modify: `include/turbulence_model.hpp` — add tau_ij and tau_div to SolverDeviceView

- [ ] **Step 1: Add tau_div pointers to solver.hpp**

In `include/solver.hpp`, find the work array pointers section (around line 1038, near `conv_u_ptr_`, `diff_u_ptr_`). Add after the diff pointers:

```cpp
    // Anisotropic stress divergence: div(tau_ij) at velocity face locations
    // Only nonzero when turbulence model provides_reynolds_stresses()
    double* tau_div_u_ptr_ = nullptr;
    double* tau_div_v_ptr_ = nullptr;
    double* tau_div_w_ptr_ = nullptr;  // 3D (future)
```

Also add the declaration in the public methods section (near `compute_diffusive_term`):

```cpp
    /// Compute anisotropic stress divergence at velocity faces
    /// from cell-centered tau_ij (for TBNN/EARSM/GEP models)
    void compute_tau_divergence();
```

- [ ] **Step 2: Add tau_ij and tau_div to SolverDeviceView**

In `include/turbulence_model.hpp`, inside `struct SolverDeviceView`, after the `diff_w` field (line ~61), add:

```cpp
    // Reynolds stress tensor (cell-centered) — for anisotropic models
    double* tau_xx = nullptr;
    double* tau_xy = nullptr;
    double* tau_yy = nullptr;

    // Anisotropic stress divergence (at velocity faces)
    double* tau_div_u = nullptr;
    double* tau_div_v = nullptr;
```

- [ ] **Step 3: Build and verify**

```bash
cd build_v100 && module load nvhpc && make -j$(nproc) 2>&1 | tail -5
```

Expected: clean build (new members declared but unused).

- [ ] **Step 4: Commit**

```
git add include/solver.hpp include/turbulence_model.hpp
git commit -m "Add tau_div storage and SolverDeviceView fields for anisotropic stress divergence"
```

---

## Task 2: Initialize tau_div buffers and GPU mapping

**Files:**
- Modify: `src/solver.cpp` — allocate tau_div arrays, map to GPU, populate in get_solver_view()

- [ ] **Step 1: Allocate tau_div arrays alongside diff arrays**

In `src/solver.cpp`, find where `diff_u_ptr_` is initialized (search for `diff_u_ptr_ =`). Add tau_div allocation nearby. The tau_div arrays have the same size as the velocity face arrays (u_total_size for tau_div_u, v_total_size for tau_div_v).

Actually, the simplest approach: reuse existing VectorField storage. Find where `diff_` VectorField is initialized and add a parallel `tau_div_` VectorField. But since we need GPU pointers, let's use raw arrays like diff does.

Find the GPU mapping section in `init_gpu_data()` or `map_fields_to_gpu()` (search for `diff_u_ptr_`). The diff arrays are part of the velocity field work buffers. Add the tau_div pointers using the same pattern:

```cpp
    // Allocate tau_div arrays (same size as velocity face arrays)
    tau_div_u_.resize(velocity_.u_total_size(), 0.0);
    tau_div_v_.resize(velocity_.v_total_size(), 0.0);
    tau_div_u_ptr_ = tau_div_u_.data();
    tau_div_v_ptr_ = tau_div_v_.data();
```

You'll need to add `std::vector<double> tau_div_u_, tau_div_v_;` to the private section of `solver.hpp`.

- [ ] **Step 2: Add tau_div to GPU mapping**

Find the `#pragma omp target enter data map(to: diff_u_ptr_...)` and add:

```cpp
    #pragma omp target enter data map(to: tau_div_u_ptr_[0:u_total_sz])
    #pragma omp target enter data map(to: tau_div_v_ptr_[0:v_total_sz])
```

- [ ] **Step 3: Add tau_ij and tau_div to get_solver_view()**

In `get_solver_view()` (around line 4683), add after the existing diff assignments:

```cpp
    view.tau_xx = tau_xx_ptr_;
    view.tau_xy = tau_xy_ptr_;
    view.tau_yy = tau_yy_ptr_;
    view.tau_div_u = tau_div_u_ptr_;
    view.tau_div_v = tau_div_v_ptr_;
```

- [ ] **Step 4: Build and verify**

```bash
cd build_v100 && make -j$(nproc) 2>&1 | tail -5
```

- [ ] **Step 5: Commit**

```
git add src/solver.cpp include/solver.hpp
git commit -m "Allocate tau_div buffers and add to GPU mapping and SolverDeviceView"
```

---

## Task 3: Implement compute_tau_divergence()

**Files:**
- Modify: `src/solver_operators.cpp` — new function computing div(tau_ij) on the staggered grid

The staggered grid differencing for tau_div is analogous to the pressure gradient:
- `tau_div_u[i+1/2,j] = (tau_xx[i+1,j] - tau_xx[i,j]) / dx + (tau_xy_at_v_face[i,j+1/2] - tau_xy_at_v_face[i,j-1/2]) / dy`

But tau_xy is at cell centers, and we need it at v-face locations (j+1/2). Interpolate:
- `tau_xy_at_j+1/2 = 0.5 * (tau_xy[i,j] + tau_xy[i,j+1])`

For the u-momentum equation at u-face (i+1/2, j):
```
tau_div_u = (tau_xx[i+1,j] - tau_xx[i,j]) / dx
          + (0.5*(tau_xy[i,j] + tau_xy[i,j+1]) - 0.5*(tau_xy[i,j-1] + tau_xy[i,j])) / dy
```

Simplifying the y-term:
```
tau_div_u = (tau_xx[i+1,j] - tau_xx[i,j]) / dx
          + (tau_xy[i,j+1] - tau_xy[i,j-1]) / (2*dy)
```

Wait — on a staggered grid we need to be more careful. The u-face is at (i+1/2, j). The tau_xx values are at cell centers (i, j) and (i+1, j). The x-derivative is straightforward: `(tau_xx[i+1,j] - tau_xx[i,j]) / dx`.

For the y-derivative of tau_xy at the u-face: tau_xy is at cell centers. The u-face at (i+1/2, j) needs d(tau_xy)/dy. We need tau_xy at the u-face's y-boundaries, which are at (i+1/2, j±1/2). Interpolate from centers:
- `tau_xy at (i+1/2, j+1/2) = 0.25 * (tau_xy[i,j] + tau_xy[i+1,j] + tau_xy[i,j+1] + tau_xy[i+1,j+1])`

But this 4-point average is expensive. A simpler approximation (second-order accurate):
- `d(tau_xy)/dy at (i+1/2, j) ≈ 0.5 * ((tau_xy[i,j+1] - tau_xy[i,j-1]) + (tau_xy[i+1,j+1] - tau_xy[i+1,j-1])) / (2*dy)`

Actually the simplest correct approach: average the x-neighbors first, then differentiate in y:
- `tau_xy_avg[j] = 0.5 * (tau_xy[i,j] + tau_xy[i+1,j])`
- `d(tau_xy)/dy = (tau_xy_avg[j+1] - tau_xy_avg[j-1]) / (2*dy)`

This is equivalent to `0.5 * (tau_xy[i,j+1] + tau_xy[i+1,j+1] - tau_xy[i,j-1] - tau_xy[i+1,j-1]) / (2*dy)`.

Let me use the SIMPLER approach that matches how the existing code handles similar center-to-face operations:

- [ ] **Step 1: Add compute_tau_divergence() to solver_operators.cpp**

Add this function after `compute_diffusive_term()`:

```cpp
void RANSSolver::compute_tau_divergence() {
    // Compute div(tau_ij) at velocity face locations from cell-centered tau_ij.
    // This is the anisotropic correction beyond the Boussinesq part (already in diff).
    // Only called when the turbulence model provides_reynolds_stresses().

    if (!turb_model_ || !turb_model_->provides_reynolds_stresses()) return;

    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Ng = mesh_->Nghost;
    const double dx = mesh_->dx;
    const double dy = mesh_->dy;
    const int u_stride = Nx + 2 * Ng + 1;
    const int v_stride = Nx + 2 * Ng;
    const int c_stride = Nx + 2 * Ng;  // cell-centered stride

    [[maybe_unused]] const size_t u_sz = velocity_.u_total_size();
    [[maybe_unused]] const size_t v_sz = velocity_.v_total_size();
    [[maybe_unused]] const size_t c_sz = field_total_size_;

    double* tdu = tau_div_u_ptr_;
    double* tdv = tau_div_v_ptr_;
    double* txx = tau_xx_ptr_;
    double* txy = tau_xy_ptr_;
    double* tyy = tau_yy_ptr_;

    if (mesh_->is2D()) {
        // tau_div_u at u-face (i, j) where i ranges Ng..Nx+Ng, j ranges Ng..Ny+Ng-1
        // tau_div_u = d(tau_xx)/dx + d(tau_xy)/dy
        // d(tau_xx)/dx at u-face i: (txx[i,j] - txx[i-1,j]) / dx  (center i is to the right of face i)
        // d(tau_xy)/dy: average txy at the two cell centers flanking the u-face in x,
        //              then central difference in y
        #pragma omp target teams distribute parallel for collapse(2) \
            map(present: tdu[0:u_sz], txx[0:c_sz], txy[0:c_sz])
        for (int j = Ng; j < Ng + Ny; ++j) {
            for (int i = Ng; i <= Ng + Nx; ++i) {
                int u_idx = j * u_stride + i;
                // Cell centers flanking this u-face: (i-1, j) and (i, j)
                // Note: for the leftmost u-face at i=Ng, i-1=Ng-1 is the ghost cell
                int c_left = j * c_stride + (i - 1);
                int c_right = j * c_stride + i;

                double dtxx_dx = (txx[c_right] - txx[c_left]) / dx;

                // d(tau_xy)/dy: average of the two flanking cell columns, central diff in y
                double txy_top = 0.5 * (txy[(j+1) * c_stride + (i-1)] + txy[(j+1) * c_stride + i]);
                double txy_bot = 0.5 * (txy[(j-1) * c_stride + (i-1)] + txy[(j-1) * c_stride + i]);
                double dtxy_dy = (txy_top - txy_bot) / (2.0 * dy);

                tdu[u_idx] = dtxx_dx + dtxy_dy;
            }
        }

        // tau_div_v at v-face (i, j) where i ranges Ng..Nx+Ng-1, j ranges Ng..Ny+Ng
        // tau_div_v = d(tau_xy)/dx + d(tau_yy)/dy
        #pragma omp target teams distribute parallel for collapse(2) \
            map(present: tdv[0:v_sz], txy[0:c_sz], tyy[0:c_sz])
        for (int j = Ng; j <= Ng + Ny; ++j) {
            for (int i = Ng; i < Ng + Nx; ++i) {
                int v_idx = j * v_stride + i;
                // Cell centers flanking this v-face: (i, j-1) and (i, j)
                int c_below = (j - 1) * c_stride + i;
                int c_above = j * c_stride + i;

                double dtyy_dy = (tyy[c_above] - tyy[c_below]) / dy;

                // d(tau_xy)/dx: average of the two flanking cell rows, central diff in x
                double txy_right = 0.5 * (txy[(j-1) * c_stride + (i+1)] + txy[j * c_stride + (i+1)]);
                double txy_left = 0.5 * (txy[(j-1) * c_stride + (i-1)] + txy[j * c_stride + (i-1)]);
                double dtxy_dx = (txy_right - txy_left) / (2.0 * dx);

                tdv[v_idx] = dtxy_dx + dtyy_dy;
            }
        }
    }
    // 3D: not implemented yet (future task)
}
```

- [ ] **Step 2: Build and verify**

```bash
cd build_v100 && make -j$(nproc) 2>&1 | tail -5
```

- [ ] **Step 3: Commit**

```
git add src/solver_operators.cpp
git commit -m "Implement compute_tau_divergence() for 2D anisotropic stress source term"
```

---

## Task 4: Call compute_tau_divergence() and add to predictor

**Files:**
- Modify: `src/solver_time.cpp` — call compute_tau_divergence after diffusion, add to predictor kernel

- [ ] **Step 1: Call compute_tau_divergence() in the step pipeline**

In `src/solver_time.cpp`, find where `compute_diffusive_term()` is called in `project_velocity()` / `euler_substep()`. Add the tau_div computation right after diffusion:

Find the block that computes diffusion (search for `TIMED_SCOPE("diffusive_term")`). After it, add:

```cpp
    // Compute anisotropic stress divergence (only if model provides tau_ij)
    if (turb_model_ && turb_model_->provides_reynolds_stresses()) {
        compute_tau_divergence();
    }
```

- [ ] **Step 2: Add tau_div to the predictor kernel**

In the 2D predictor kernel (around line 350), change:

```cpp
u_out_ptr[idx] = u_in_ptr[idx] + dt * (-conv_u_ptr[idx] + diff_u_ptr[idx] + fx);
```

To:

```cpp
u_out_ptr[idx] = u_in_ptr[idx] + dt * (-conv_u_ptr[idx] + diff_u_ptr[idx] + fx + tau_div_u_ptr[idx]);
```

And similarly for v:

```cpp
v_out_ptr[idx] = v_in_ptr[idx] + dt * (-conv_v_ptr[idx] + diff_v_ptr[idx] + fy + tau_div_v_ptr[idx]);
```

You need to add `tau_div_u_ptr` and `tau_div_v_ptr` to the local variables section and the `map(present:)` clause:

```cpp
double* tau_div_u_ptr = sv.tau_div_u;
double* tau_div_v_ptr = sv.tau_div_v;
```

And in the pragma:
```cpp
map(present: u_in_ptr[0:u_total], u_out_ptr[0:u_total], conv_u_ptr[0:u_total], diff_u_ptr[0:u_total], tau_div_u_ptr[0:u_total])
```

**IMPORTANT:** Do this for ALL predictor kernels — there are multiple copies for different velocity field pairs (velocity→velocity_star, velocity_star→velocity_rk, etc.). Search for ALL occurrences of the pattern `u_in_ptr[idx] + dt * (-conv_u_ptr[idx] + diff_u_ptr[idx] + fx)` and update them all.

When `tau_div_u_ptr` is nullptr or the model doesn't provide Reynolds stresses, tau_div arrays contain all zeros (initialized that way), so adding them is safe — it's just adding zero.

- [ ] **Step 3: Build and run existing tests**

```bash
cd build_v100 && make -j$(nproc) && cd ..
export OMP_TARGET_OFFLOAD=MANDATORY XALT_EXECUTABLE_TRACKING=no
./build_v100/test_ghost_cell_ibm 2>&1 | grep -c PASS
```

Expected: all 17 tests still pass (tau_div is zero for models without Reynolds stresses).

- [ ] **Step 4: Commit**

```
git add src/solver_time.cpp
git commit -m "Add anisotropic stress divergence to predictor step in momentum equations"
```

---

## Task 5: Integration test — EARSM vs SST

**Files:**
- No code changes — run simulations and compare results

The simplest validation: run cylinder Re=100 with SST and EARSM-WJ. With tau_div active, EARSM should produce a DIFFERENT Cd from SST (previously they were nearly identical because tau_ij was unused).

- [ ] **Step 1: Run SST on cylinder**

```bash
cat > /tmp/cyl_sst_tau.cfg << 'EOF'
Nx = 96
Ny = 72
Nz = 1
x_min = -3.0
x_max = 13.0
y_min = -6.0
y_max = 6.0
nu = 0.01
dp_dx = -0.004
max_steps = 500
CFL_max = 0.3
dt_safety = 0.85
adaptive_dt = true
simulation_mode = unsteady
convective_scheme = skew
time_integrator = rk3
poisson_tol = 1e-6
output_freq = 500
ibm_eta = 0
perturbation_amplitude = 0.05
turb_model = sst
warmup_model = sst
warmup_time = 2.0
warmup_steps = 20
gpu_only_mode = true
write_fields = false
EOF

./build_v100/cylinder --config /tmp/cyl_sst_tau.cfg 2>&1 | grep "Cd="
```

- [ ] **Step 2: Run EARSM-WJ on cylinder**

```bash
sed 's/turb_model = sst/turb_model = earsm_wj/' /tmp/cyl_sst_tau.cfg > /tmp/cyl_earsm_tau.cfg
./build_v100/cylinder --config /tmp/cyl_earsm_tau.cfg 2>&1 | grep "Cd="
```

- [ ] **Step 3: Compare Cd values**

Expected: EARSM Cd should differ from SST Cd by a non-trivial amount (>1%). If they're still identical, the tau_div term is not being applied correctly.

- [ ] **Step 4: Run TBNN on cylinder**

```bash
cat > /tmp/cyl_tbnn_tau.cfg << 'EOF'
# Same as sst config but with nn_tbnn
Nx = 96
Ny = 72
Nz = 1
x_min = -3.0
x_max = 13.0
y_min = -6.0
y_max = 6.0
nu = 0.01
dp_dx = -0.004
max_steps = 500
CFL_max = 0.3
dt_safety = 0.85
adaptive_dt = true
simulation_mode = unsteady
convective_scheme = skew
time_integrator = rk3
poisson_tol = 1e-6
output_freq = 500
ibm_eta = 0
perturbation_amplitude = 0.05
turb_model = nn_tbnn
nn_weights_path = data/models/tbnn_small_paper
nn_scaling_path = data/models/tbnn_small_paper
warmup_model = sst
warmup_time = 2.0
warmup_steps = 20
gpu_only_mode = true
write_fields = false
EOF

./build_v100/cylinder --config /tmp/cyl_tbnn_tau.cfg 2>&1 | grep "Cd="
```

Expected: TBNN Cd should differ from SST. If it diverges, add `nu_t_relaxation = 0.3` to stabilize.

---

## Task 6: Zero tau_div for non-anisotropic models

**Files:**
- Modify: `src/solver.cpp` or `src/solver_time.cpp` — ensure tau_div arrays are zeroed when model doesn't provide tau_ij

- [ ] **Step 1: Zero tau_div at the start of each step**

In `src/solver.cpp` `step()`, at the beginning (near where force ramp is updated), add:

```cpp
    // Zero tau_div arrays. They will be populated by compute_tau_divergence()
    // only if the model provides Reynolds stresses. Otherwise they stay zero
    // and add nothing to the predictor.
    if (tau_div_u_ptr_) {
        [[maybe_unused]] const size_t u_sz = velocity_.u_total_size();
        [[maybe_unused]] const size_t v_sz = velocity_.v_total_size();
        double* tdu = tau_div_u_ptr_;
        double* tdv = tau_div_v_ptr_;
        #pragma omp target teams distribute parallel for map(present: tdu[0:u_sz])
        for (size_t i = 0; i < u_sz; ++i) tdu[i] = 0.0;
        #pragma omp target teams distribute parallel for map(present: tdv[0:v_sz])
        for (size_t i = 0; i < v_sz; ++i) tdv[i] = 0.0;
    }
```

- [ ] **Step 2: Build and verify all tests pass**

- [ ] **Step 3: Commit**

```
git add src/solver.cpp
git commit -m "Zero tau_div arrays each step to ensure clean state for non-anisotropic models"
```

---

## Self-Review Checklist

1. **Spec coverage**: ✅ tau_div added to predictor, compute_tau_divergence() for 2D, integration test, zero for non-aniso models
2. **Placeholder scan**: No TBDs. All code provided. Task 4 step 2 says "ALL predictor kernels" — need to ensure this covers the RK3 substep variants too.
3. **Type consistency**: `tau_div_u_ptr_` used consistently. SolverDeviceView field names match.
4. **Missing**: 3D implementation (explicitly deferred). The stretched-y grid case for d(tau_xy)/dy needs dyv weighting (currently uses uniform dy) — note this for non-uniform grids.
5. **Risk**: The 4-point average for d(tau_xy)/dy at u-faces accesses j-1 and j+1 — needs ghost cell values at domain boundaries. The BCs should fill tau_ij ghost cells (periodic or zero-gradient). This might need explicit handling.
