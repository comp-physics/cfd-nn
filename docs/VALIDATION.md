# Validation and Test Results

This document summarizes the solver's validation status: what has been tested, how well it performs, and what gaps remain.

For how to *run* and *extend* the test suite, see [`TESTING_GUIDE.md`](TESTING_GUIDE.md).

---

## Test Suite Overview

Tests are organized into two tiers:

| Tier | Where | Duration | Purpose |
|------|-------|----------|---------|
| **Tier 1 (CI)** | GitHub Actions (CPU) + self-hosted SLURM (GPU) | < 30 min | Catch regressions on every push |
| **Tier 2 (Validation)** | SLURM batch jobs (H200/A100) | 2-8 hours | Full physics validation against reference data |

### Label System

Each test carries one or more labels that control when it runs:

| Label | Time Limit | Description |
|-------|-----------|-------------|
| `fast` | < 30 s | Unit tests, sanity checks, operator verification |
| `medium` | 30 s - 5 min | Grid convergence, manufactured solutions, short solves |
| `slow` | > 5 min | Recycling inflow, endurance stability, detailed parity |
| `gpu` | varies | Requires GPU; sets `OMP_TARGET_OFFLOAD=MANDATORY` |
| `hypre` | varies | Requires HYPRE build (`-DUSE_HYPRE=ON`) |
| `fft` | varies | Requires FFT build (cuFFT/cuSPARSE on GPU) |
| `cross-build` | varies | Compares CPU and GPU build outputs |

### Running Tests

```bash
cd build && ctest --output-on-failure          # All tests
make check-fast                                # Fast tests only (<30s)
make check-quick                               # All except slow tests
ctest -L gpu                                   # GPU-specific tests
ctest -L medium                                # Medium tests
```

---

## Analytical Benchmarks

### Poiseuille Flow (2D Channel)

Steady laminar flow driven by a constant pressure gradient, with analytical solution:

$$u(y) = -\frac{dp/dx}{2\nu} \, y(H - y)$$

**Tests:**
- `PoiseuilleTest` (fast): 32x64 grid, checks L2 error < 1%, symmetry, divergence-free
- `PoiseuilleValidationTest` (gpu, medium): 32x64 and 64x128 grids on GPU, L2 < 1%, L-inf < 5%, bulk velocity error < 1%
- `PoiseuilleRefinementTest` (medium): Grid convergence from 32 to 64 cells, verifies 2nd-order convergence rate >= 1.5 (measured ~2.0)

| Grid | L2 Error | Convergence Order |
|------|----------|-------------------|
| 32x64 | ~0.13% | — |
| 64x128 | ~0.03% | ~2.0 |

### Poiseuille Flow (3D Duct)

`DuctPoiseuilleTest` (medium): 3D duct with walls on y and z faces. Verifies the 3D solver reproduces the analytical Poiseuille profile.

### Taylor-Green Vortex

Unsteady decaying vortex with initial condition $u = \sin x \cos y \cos z$, $v = -\cos x \sin y \cos z$, $w = 0$.

**Tests:**
- `TGV2DInvariantsTest` (fast): 2D energy decay, symmetry, divergence checks
- `TGV3DInvariantsTest` (medium): 3D 32^3 grid, energy monotonicity, symmetry
- `TGVValidationTest` (gpu, medium): Two cases on GPU:
  - Re = 100, 32^3: Viscous decay matches analytical $E(t) = E_0 e^{-2\nu t}$
  - Re = 1600, 64^3: Stable through vortex breakdown (under-resolved, tests stability not accuracy)
- `TGVRepeatabilityTest` (fast): Bitwise reproducibility across runs

**Key results:**
- Energy monotonically decays (no spurious creation)
- Mean velocities $\langle u \rangle = \langle v \rangle = \langle w \rangle = 0$ (symmetry preserved)
- Incompressibility: $\|\nabla \cdot u\|_\infty < 10^{-10}$

### MMS Convergence

`MMSConvergenceTest` (medium): Method of Manufactured Solutions with exact divergence-free field:

$$u = \sin(kx)\cos(ky), \quad v = -\cos(kx)\sin(ky)$$

Measures L2 error decay across grid refinements. Verifies **2nd-order spatial convergence** (measured rate >= 1.8).

### Fractional-Step Temporal Convergence

`FractionalStepTest` (medium): Verifies 1st-order temporal convergence of the projection method using manufactured solutions with decreasing dt.

---

## RANS Validation

### Bug Fixes (March 2026)

Four critical bugs were fixed that affected all transport-equation RANS models:

1. **GPU residual NaN masking**: The GPU max-reduction used `if (du > max_du)` which silently discards NaN per IEEE 754. Fixed by adding NaN sentinel detection in all three residual functions.

2. **Transport equation stiffness**: Explicit Euler on k-omega/SST blew up at wall cells where $\omega_\text{wall} \sim 840000$ and $\beta^* \omega^2 \sim 5 \times 10^{10}$. Fixed with **point-implicit destruction**: $k_\text{new} = (k + \Delta t \cdot S^+) / (1 + \Delta t \cdot S^-)$.

3. **FFT2D stretched-grid bug**: Auto-selection picked FFT2D for 2D stretched grids on GPU. FFT2D assumes uniform dy in its tridiagonal y-solve, so $D \cdot G \neq L$ on stretched grids, breaking the projection. Fixed by adding a `uniform_y` guard to FFT2D selection.

4. **Baseline mixing-length host read**: The mixing-length model read host-side velocity for $u_\tau$ computation on GPU builds, producing $\nu_t \approx 0$ (stale data). Model now uses GPU-resident data.

### All 10 Models: Stability and Profile Shape

`RANSChannelValidationTest` (gpu, medium): Runs all 10 turbulence closures on a 2D stretched channel (32x48, Re_tau = 180) for 200 steps. Checks:

| Check | Criterion |
|-------|-----------|
| No NaN/Inf | velocity and nu_t finite |
| Velocity bounded | $\|u\|_\infty < 50$ |
| Monotonic profile | $u$ increases wall → center |
| Eddy viscosity | $\nu_t > 0$ (transport and EARSM models) |
| Divergence-free | $\|\nabla \cdot u\|_\infty < 10^{-3}$ |

**Results (all passing):**

| Model | Type | Stable | Profile OK | nu_t > 0 |
|-------|------|--------|-----------|----------|
| None (laminar) | — | Yes | Yes | N/A |
| Baseline | Algebraic | Yes | Yes | Yes |
| GEP | Algebraic | Yes | Yes | Yes |
| SST k-omega | Transport | Yes | Yes | Yes |
| k-omega | Transport | Yes | Yes | Yes |
| EARSM-WJ | EARSM | Yes | Yes | Yes |
| EARSM-GS | EARSM | Yes | Yes | Yes |
| EARSM-Pope | EARSM | Yes | Yes | Yes |
| NN-MLP | Neural Net | Yes | Yes | Yes |
| NN-TBNN | Neural Net | Yes | Yes | Yes |

### Accuracy vs MKM DNS Reference

`RANSAccuracyTest` (medium): Validates $u^+$ profiles against Moser, Kim & Mansour (1999) DNS at Re_tau ~ 178. Uses a 2D channel (32x64), 3000 steps to near-steady-state.

| Model | Log-Layer Error (y+ > 30) | Buffer-Layer Error (5 < y+ < 30) | u_tau |
|-------|--------------------------|----------------------------------|-------|
| Baseline | < 30% | < 50% | within 50% of target |
| GEP | < 30% | < 50% | within 50% of target |
| SST | < 20% | < 40% | within 50% of target |

Transport models (SST) achieve tighter accuracy than algebraic models, as expected.

### Profile Shape and Near-Wall Behavior

`RANSChannelSanityTest` (fast): Catches regressions via:
- No-slip at walls, centerline maximum, monotonic wall-to-center (5% tolerance)
- Near-wall $\nu_t$ bounds: $\min(\nu_t) \geq -10^{-12}$, $\max(\nu_t)/\nu < 10^6$, first-cell $\nu_t/\nu < 10$
- Integral metrics: $U_\text{bulk} > 0$, correct wall shear sign

### Frame Invariance

`RANSFrameInvarianceTest` (fast): Verifies tensor algebra satisfies frame objectivity. Applies random orthogonal rotation $R$ to synthetic velocity gradients and checks:
- Scalar invariants unchanged: $|S|$, $|\Omega|$, $\text{tr}(S^2)$, $\text{tr}(\Omega^2)$
- Tensor basis transforms covariantly: $T'^{(n)} = R \cdot T^{(n)} \cdot R^T$

Catches sign errors, missing transposes, and incorrect tensor formulas.

### 3D RANS Channel

`RANS3DChannelTest` (medium): First 3D RANS test. 16x32x16 uniform grid, 200 steps. Tests Baseline, GEP, SST, EARSM-WJ for:
- Stability (no NaN, $v_\text{max} < 50$)
- Symmetric profile ($< 10\%$ tolerance)
- Eddy viscosity positive (TRACK on both CPU and GPU)

**Known limitation:** Some turbulence models use 2D indexing internally, mapping to the $k=0$ plane only. Interior $k$-planes may have $\nu_t = 0$. This is tracked but does not fail CI.

---

## DNS Channel Flow

### v13 Recipe (Stable DNS)

Direct Numerical Simulation of turbulent channel flow at Re_tau ~ 180, targeting the MKM (1999) benchmark.

**Grid:** 192 x 96 x 192, $L_x = 4\pi$, $L_y = 2$, $L_z = 2\pi$, `stretch_beta = 2.0`

**Resolution:** $\Delta x^+ \approx 11.8$, $\Delta z^+ \approx 5.9$, $y_1^+ \approx 0.29$

**Key parameters:**
```ini
CFL_max = 0.15
CFL_xz = 0.30
dt_safety = 0.85
trip_amp = 1.0
trip_duration = 0.20
trip_ramp_off = 0.10
trip_w_scale = 2.0
filter_strength = 0.03
filter_interval = 2
scheme = skew
integrator = rk3
```

### Run History

| Run | Filter | Result | Re_tau | Notes |
|-----|--------|--------|--------|-------|
| v9 | none | Blew up ~1700 | — | Turbulent before blow-up |
| v10 | 0.02, every 10 (x/z only) | Blew up ~2000 | — | Survived longer |
| v11 | 0.05, every 1 (x/y/z) | **Stable**, 3600+ steps | ~255 | First stable run |
| v13 | 0.03, every 2 (x/y/z) | **Stable**, 2400+ steps | ~278 | Best balance |

### v13 on H200 (3750 steps)

| Step | Re_tau | v_max | w/v Ratio | State |
|------|--------|-------|-----------|-------|
| 1200 | 222 | 16.7 | 0.054 | TURBULENT, trip OFF |
| 1800 | 250 | 15.2 | 0.609 | TURBULENT |
| 2400 | 307 | 15.3 | 1.454 | TURBULENT (peak) |
| 3600 | 255 | 20.6 | 1.21 | TURBULENT (stabilizing) |

### CI DNS Check

`DNSChannelValidationTest` (gpu, medium): Lightweight machinery check — 32x32x32 DNS with v13 recipe for 30 steps. Verifies:
- Incompressibility: $\|\nabla \cdot u\|_\infty < 10^{-3}$
- Stability: $v_\text{max} < 50$
- KE bounded: $0.01 < E/E_0 < 10$
- Uses device-side diagnostics (no GPU-CPU sync)

Full 192x96x192 validation is done in Tier 2 SLURM jobs.

### Known Gap

The velocity filter adds effective viscosity (~16x molecular at strength=0.03, interval=2), so the achieved Re_tau (~278) exceeds the target of 180. Reaching the exact target would require a less dissipative convective scheme (e.g., hybrid skew-symmetric/upwind).

---

## Operator Correctness

### Stretched-Grid D·G = L

`OperatorConsistencyTest` (fast): The staggered-grid divergence and gradient operators must compose to the exact discrete Laplacian. Tests $\max|D(G(\phi)) - L(\phi)| < 10^{-10}$ on stretched y-grids. Validates that the precomputed y-metric arrays (`dyc`, `dyv`, `yLap_aS/aP/aN`) are consistent.

This is critical: if $D \cdot G \neq L$, the pressure projection introduces a splitting error that grows over time.

### Gradient Convergence

`OperatorConvergenceTest` (fast): Verifies spatial derivative operators converge at the expected rate on periodic domains using smooth sine/cosine functions:
- O2 operators: rate >= 1.9
- O4 operators: rate >= 3.8
- Tests same-stagger, center-to-face, and face-to-center derivatives
- Validates div-grad adjoint identity: $\langle p, \nabla \cdot u \rangle = -\langle \nabla p, u \rangle$ (required for discrete energy conservation)

### Stretched Gradient Accuracy

`StretchedGradientTest` (fast): Validates $\partial u / \partial y$ on non-uniform grids. Tests $\cos(\pi y/2)$ gradient and Poiseuille profile on stretched meshes. Grid convergence order ~2, L2 relative error < 5%.

### Galilean Invariance

`GalileanInvarianceTest` (fast): The Navier-Stokes equations are Galilean-invariant — adding a uniform velocity offset should not change the dynamics. Compares a Taylor-Green vortex at rest vs with uniform offset $(U_0, V_0)$. Fluctuating KE and velocity perturbations match to $10^{-6}$.

`ProjectionGalileanTest` (fast): Isolates the projection step (single step, tiny dt and nu). Verifies:
- Mean velocity preserved to $10^{-10}$
- Divergence-free in both frames
- Fluctuations identical regardless of frame

---

## GPU Parity

### CPU/GPU Kernel Comparison

`KernelParityDetailedTest` (slow): Compares CPU and GPU implementations of individual non-Poisson kernels: gradients, advection, diffusion, velocity correction. Tests single-step and 10-step evolution parity.

### Cross-Backend Consistency

`CrossBackendTest` (cross-build): Builds both CPU and GPU versions, generates CPU reference outputs, then compares GPU against CPU for laminar, baseline RANS, and NN model scenarios. Per-metric tolerance matching.

### GPU Utilization Gate

`GPUUtilizationTest` (gpu): CI gate ensuring GPU runs dominate compute time (>= 70% GPU utilization). Tests all model types. Catches accidental CPU fallback.

### GPU Turbulence Readiness

`GPUTurbulenceReadinessTest` (gpu): 6-check acceptance suite:
1. GPU offload valid (device pointers, `omp_get_mapped_ptr` works)
2. 3D Poisson convergence
3. Step stability (20 steps, no NaN/Inf)
4. Turbulence classifier detects perturbations
5. Perf mode stability (30 steps with reduced diagnostics)
6. Projection watchdog ($\|\nabla \cdot u\| < 10^{-5}$)

---

## Recycling Inflow

Implements Lund et al. (1998) for spatially-developing turbulent channel flow.

### PeriodicVsRecyclingTest (slow)

Runs identical channel flow with periodic BCs and with recycling inflow BCs, then compares plane-averaged statistics:

| Metric | Tolerance | Achieved |
|--------|-----------|----------|
| Shear stress difference | < 5% | ~0.3% |
| Streamwise stress difference | < 5% | ~3.6% |

### RecyclingInflowTest (slow)

12 checks covering symmetry, $u_\tau$, mass conservation, fringe blending, divergence, and ghost cells. All passing on both CPU and GPU.

---

## Additional Solver Tests

### Conservation and Energy

- `ConservationAuditTest` (medium): Momentum balance — body force integral = wall shear integral
- `EnergyBudgetTest` (fast): Energy budget closure for channel flow
- `SkewEnergyTest` (medium): Skew-symmetric convection conserves discrete kinetic energy

### Projection Method

- `ProjectionEffectivenessTest` (fast): Divergence reduction by projection step
- `ProjectionInvariantsTest` (medium): Projection preserves expected invariants
- `ProjectionTraceTest` (fast): Trace diagnostics for projection step

### Poisson Solvers

- `PoissonUnifiedTest` (medium): All solver backends on standard test problems
- `MGManufacturedSolutionTest` (medium): Multigrid with manufactured solutions for channel, duct, and recycling BCs. CPU/GPU max difference = 0.0.
- `FFTUnifiedTest` (fft, gpu): FFT 3D, 2D, and 1D solvers. Grid convergence and CPU/GPU consistency.
- `MGRampDiagnosticTest` (fast): Multigrid convergence diagnostics
- `SolverSelectionTest` (medium): Auto-selection picks correct backend for each BC configuration

### HYPRE

- `HypreAllBcsTest` (hypre): HYPRE solver on all BC combinations
- `HypreValidationTest` (hypre): HYPRE vs multigrid solution comparison
- `HypreBackendTest` (hypre): HYPRE backend integration

### Stability and Endurance

- `EnduranceStabilityTest` (slow): Long-running stability check
- `StabilitySentinelTest` (medium): Early warning for stability regression
- `PerfSentinelTest` (medium): Performance regression detection

### Scheme Validation

- `SchemeComprehensiveTest` (medium): Comprehensive convective scheme testing
- `SchemeCombinationsTest` (medium): All scheme + integrator combinations
- `O4IntegrationTest` (medium): 4th-order spatial discretization integration
- `TimeIntegratorsTest` (medium): Euler, RK2, RK3 temporal accuracy

### Other

- `MeshTest` (fast): Mesh indexing, ghost cells, field operations, wall distance
- `ConfigTest` (fast): Config file parsing and CLI arg handling
- `AdaptiveDtTest` (fast): Adaptive time stepping with directional CFL
- `NNCoreTest` (fast): MLP weight loading, forward pass, feature computation
- `NNIntegrationTest` (medium): Full NN model integration with solver
- `ResidualConsistencyTest` (fast): Residual computation consistency
- `VTKOutputTest` (fast): VTK file output correctness

---

## CI Test Matrix

### Tier 1: CPU CI (GitHub Actions)

Runs on every push. 4 configurations:

| OS | Build Type | Tests Run |
|----|-----------|-----------|
| Ubuntu | Debug | `fast` only (4x timeout) |
| Ubuntu | Release | `fast` + `medium` |
| macOS | Debug | `fast` only (4x timeout) |
| macOS | Release | `fast` + `medium` |

### Tier 1: GPU CI (Self-Hosted SLURM)

Runs on every push to branches with open PRs:

| Stage | Description |
|-------|-------------|
| Build | GPU build with NVHPC + HYPRE |
| Correctness | All `gpu`-labeled tests via SLURM (H200, 1 GPU) |
| Performance | Performance regression tests via separate SLURM job |
| Cross-build | CPU build + GPU build, compare outputs |

GPU CI serializes per-branch (no parallel runs) to prevent resource contention. 240-minute timeout with cleanup trap.

### Tier 2: Validation (Manual SLURM)

| Script | Duration | Description |
|--------|----------|-------------|
| `scripts/run_validation.sh` | 6-8 hours | DNS v13 (3750 steps) + Poiseuille convergence + TGV |
| `scripts/rans_validation/run_rans_validation.sbatch` | 4 hours | All RANS models (10k steps each) vs MKM reference |

### Complete Test Count by Label

| Label | Count | Description |
|-------|-------|-------------|
| `fast` | ~35 | Unit tests, operator checks, sanity tests |
| `medium` | ~25 | Grid convergence, manufactured solutions, model validation |
| `slow` | ~10 | Recycling, endurance, detailed parity, physics validation |
| `gpu` | ~8 | GPU-specific (utilization, readiness, TGV, Poiseuille, RANS, DNS) |
| `hypre` | 3 | HYPRE solver tests |
| `fft` | 1 | FFT solver tests |
| `cross-build` | 1 | CPU vs GPU output comparison |
| **Total** | **~79** | |

---

## Known Issues and Limitations

### Numerical

1. **Explicit time stepping limits timestep for low viscosity.** Adaptive time stepping with directional CFL mitigates this, but implicit diffusion in y is only available for select kernels.

2. **Central differences require velocity filter for DNS stability.** Second-order central schemes have zero numerical dissipation, causing grid-scale blow-up in DNS. The velocity filter provides explicit diffusion but adds effective viscosity.

3. **Filter-limited Re_tau in DNS.** Best achieved: Re_tau ~ 278 with strength=0.03, interval=2. Reaching the exact MKM target (180) requires a less dissipative convective scheme.

4. **2D turbulence model indexing in 3D.** Some models use 2D indexing that maps to the $k=0$ plane only, giving $\nu_t = 0$ on interior k-planes. Tracked in CI but not yet fixed.

### GPU-Specific

5. **CPU-side diagnostics require GPU sync.** Functions that read velocity data on CPU (e.g., `accumulate_statistics()`) must call `sync_solution_from_gpu()` first.

6. **Recycling inflow disables CUDA Graph V-cycle.** BCs change each step, invalidating the graph. Falls back to standard V-cycle (~10-20% slower for Poisson).

7. **FFT2D requires uniform y-grid.** Auto-selection now correctly skips FFT2D for stretched grids (fixed March 2026).

### Model-Related

8. **Example NN models use random weights.** Real trained weights from publications are needed for meaningful accuracy. Infrastructure tests pass with random weights.

---

## References

- Moser, R. D., Kim, J., & Mansour, N. N. "Direct numerical simulation of turbulent channel flow up to Re_tau = 590." *Physics of Fluids* 11.4 (1999): 943-945
- Brachet, M. E., et al. "Small-scale structure of the Taylor-Green vortex." *J. Fluid Mech.* 130 (1983): 411-452
- Menter, F. R. "Two-equation eddy-viscosity turbulence models for engineering applications." *AIAA J.* 32.8 (1994): 1598-1605
- Wilcox, D. C. "Reassessment of the scale-determining equation for advanced turbulence models." *AIAA J.* 26.11 (1988): 1299-1310
- Wallin, S., & Johansson, A. V. "An explicit algebraic Reynolds stress model..." *J. Fluid Mech.* 403 (2000): 89-132
- Ling, J., Kurzawski, A., & Templeton, J. "Reynolds averaged turbulence modelling using deep neural networks with embedded invariance." *J. Fluid Mech.* 807 (2016): 155-166
- Weatheritt, J., & Sandberg, R. D. "A novel evolutionary algorithm applied to algebraic modifications of the RANS stress-strain relationship." *J. Comput. Phys.* 325 (2016): 22-37
- Lund, T. S., Wu, X., & Squires, K. D. "Generation of turbulent inflow data for spatially-developing boundary layer simulations." *J. Comput. Phys.* 140.2 (1998): 233-258
- Pope, S. B. "A more general effective-viscosity hypothesis." *J. Fluid Mech.* 72.2 (1975): 331-340

---

**Last updated:** March 2026 — Complete rewrite covering all 79 tests, RANS bug fixes, GPU CI, Tier 2 validation
