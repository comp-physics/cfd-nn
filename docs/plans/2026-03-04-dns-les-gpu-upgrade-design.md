# DNS/LES State-of-the-Art GPU Upgrade — Design Document

**Date**: 2026-03-04
**Status**: Approved

## Summary

Upgrade CFD-NN from a single-GPU DNS/RANS solver to a state-of-the-art multi-GPU DNS/LES platform with immersed boundary support. Targets Re_tau 180–590 channel DNS and turbulent flows around complex geometries (cylinder, airfoil). Uses OpenMP target offload + custom CUDA kernels for hot spots + CUDA math libraries (cuFFT, cuSPARSE, cuBLAS).

## Requirements

| Decision | Choice |
|---|---|
| Target Re | Re_tau 180–590 (single-GPU at 180, multi-GPU at 590) |
| Spatial schemes | Keep FD (2nd/4th skew-symmetric), optimize existing |
| Parallelism | Multi-GPU via MPI + OpenMP target (z-slab decomposition) |
| New physics | Immersed boundary method + LES SGS models |
| New flows | Cylinder (Re 100/300/3900), NACA airfoil |
| Statistics | Minimal in-situ, HDF5 field dumps for offline analysis |
| CUDA scope | cuFFT/cuSPARSE/cuBLAS + custom .cu kernels for hot spots |
| No scalar transport | Removed from scope |

## Phase 1: GPU Hot-Spot Optimization (~800 LOC)

### Problem
Chebyshev MG smoother is 76% of GPU time. Z-stride memory access (~20k stride for 192x96x192) causes poor cache behavior. BC application requires 6 separate kernel launches per V-cycle level.

### Solution

1. **Custom CUDA smoother kernel** (`src/cuda_kernels/mg_smoother.cu`):
   - Shared memory tiling: load 10x10x10 tile (8³ interior + 1 ghost each side)
   - Z-plane batching: multiple z-planes per thread block
   - Fused BC+smooth for all-periodic grids (wrap indices in-kernel)
   - Replaces OpenMP target Chebyshev kernel in `mg_cuda_kernels.cpp`

2. **Custom CUDA halo packing** (`src/cuda_kernels/halo_pack.cu`):
   - Single fused kernel replaces 6-kernel BC application
   - Pack face data into contiguous buffers (future MPI exchange)

3. **cuBLAS reductions**:
   - Replace OpenMP target reductions (residual norm, max divergence) with `cublasDnrm2`/`cublasIdamax`
   - Better warp utilization for large arrays

4. **Build integration**:
   - `enable_language(CUDA)` in CMakeLists.txt
   - `.cu` files compiled by nvcc, linked into `nn_cfd_core`
   - Runtime selection: CUDA kernels when available, fall back to OpenMP target

**Expected speedup**: 2–3x on Poisson solve, ~1.5–2x overall solver.

## Phase 2: MPI Domain Decomposition (~1200 LOC)

### Strategy
1D slab decomposition in z (periodic direction). Each MPI rank owns a contiguous z-slab and one GPU.

### Components

1. **Decomposition layer** (`include/decomposition.hpp`, `src/decomposition.cpp`, ~300 LOC):
   - `Decomposition` class: rank, nprocs, z-range per rank, neighbor ranks
   - Each rank builds local `Mesh` with local Nz + 2*Ng ghost cells
   - Global-to-local index mapping for I/O
   - No-ops for nprocs=1 (single-GPU unchanged)

2. **Halo exchange** (`src/halo_exchange.cpp`, ~200 LOC):
   - GPU-direct: CUDA kernel packs z-face → `MPI_Isend`/`MPI_Irecv` with device pointers → unpack
   - Host-staged fallback if no GPU-direct MPI (`MPI_CUDA_AWARE` detection at configure)
   - Called after each RK stage (velocity), after each smoother iteration (pressure), after prolongation

3. **Distributed FFT Poisson solver** (`src/poisson_solver_fft_mpi.cpp`, ~400 LOC):
   - Pencil transpose: z-slabs → x-pencils via `MPI_Alltoallv` (GPU buffers)
   - 1D cuFFT in x per pencil, tridiagonal solve in y (cuSPARSE, local), 1D cuFFT in z after transpose back
   - Optional cuFFTMp backend if available

4. **Distributed multigrid** (~200 LOC changes to existing MG):
   - Halo exchange per smoothing level
   - Coarse solve: gather to rank 0, solve, scatter

5. **I/O coordination** (~100 LOC):
   - Rank 0 writes headers, all ranks write local slabs
   - Statistics: local plane averages → `MPI_Allreduce` for global means

### Transparency to All Physics

The decomposition is transparent — all existing code sees only its local `Mesh`:

- **Turbulence models (all 11)**: Operate on local fields, no changes. Wall distance uses global z-coords from `Decomposition` for duct z-walls.
- **Recycling inflow**: `MPI_Bcast` if recycle plane on different rank (~30 LOC in `solver_recycling.cpp`).
- **Velocity filter**: z-halo exchange before filter loop (~5 LOC).
- **Trip forcing**: Local computation with global z-coords. No communication.
- **Adaptive dt**: Local CFL → `MPI_Allreduce(MPI_MIN)` (~5 LOC).
- **Diagnostics**: Local GPU reduction → `MPI_Allreduce(MPI_SUM)` per scalar (~3 LOC each).
- **Wall shear stress**: y-derivatives only, local. Unchanged.
- **Poisson backends**: MG gets halo exchange. FFT uses distributed transpose. FFT2D is 2D (Nz=1, no decomp). SOR/HYPRE get MPI wrappers.
- **CUDA graphs**: Disabled for multi-GPU (halo exchange breaks capture). Single-GPU keeps graphs.
- **VTK output**: Each rank writes local slab; rank 0 writes `.pvd` manifest.

## Phase 3: Immersed Boundary Method (~1000 LOC)

### Approach
Direct-forcing IBM (Fadlun et al. 2000 / Uhlmann 2005 hybrid). Signed distance field on Cartesian grid, forcing at near-boundary fluid cells.

### Components

1. **Geometry representation** (`include/ibm_geometry.hpp`, `src/ibm_geometry.cpp`, ~300 LOC):
   - `IBMBody` base class: SDF `phi(x,y,z)`, surface normal `n(x,y,z)`
   - Built-in shapes: `CylinderBody`, `SphereBody`, `NACABody` (4-digit NACA from camber+thickness)
   - SDF precomputed on grid at init, GPU-resident `ScalarField`
   - Multiple bodies: `std::vector<std::unique_ptr<IBMBody>>`

2. **IBM forcing** (`include/ibm_forcing.hpp`, `src/ibm_forcing.cpp`, ~400 LOC):
   - Cell classification: `Fluid` / `Solid` / `Forcing` from SDF sign + distance
   - Trilinear interpolation from fluid cells to boundary intercept point
   - Direct forcing: `f_ibm = (u_boundary - u_interp) / dt`
   - Applied after predictor, before Poisson solve
   - Staggered-grid aware: SDF evaluated at face locations per component
   - Single GPU kernel per component

3. **Solver integration** (~50 LOC in `solver_time.cpp`):
   ```
   u* = u^n + dt*(-conv + diff + body_force)
   u* += dt * f_ibm       ← IBM forcing here
   Poisson solve for p'
   u^{n+1} = u* - dt*∇p'
   Solid cells: hard-mask to zero after correction
   ```

4. **Poisson compatibility** (~30 LOC):
   - Solid cells excluded from divergence RHS
   - Mean subtraction over fluid cells only (volume-weighted)

5. **Geometry generators** (`src/ibm_generators.cpp`, ~200 LOC):
   - `CylinderBody(cx, cy, R)`: SDF = `sqrt((x-cx)²+(y-cy)²) - R`
   - `NACABody(x_le, y_le, chord, aoa, digits)`: SDF via Newton iteration on parametric curve
   - Config: `ibm_body = cylinder`, `ibm_center_x = 5.0`, `ibm_radius = 0.5`, etc.

6. **Force diagnostics** (`src/ibm_diagnostics.cpp`, ~100 LOC):
   - Cd/Cl via momentum balance (integrate IBM forcing)
   - GPU reduction, no field sync

### Compatibility
- **Turbulence models**: Wall distance = `min(mesh_wall_dist, abs(phi))`.
- **MPI**: SDF local per rank (analytical evaluation, no communication).
- **Velocity filter**: Skip forcing/solid cells.
- **Recycling**: Bodies downstream of recycle plane.
- **Statistics**: Plane averages exclude solid cells.
- **VTK**: Output SDF + cell-type mask.

## Phase 4: LES Subgrid-Scale Models (~600 LOC)

### Models

All inherit from `TurbulenceModel`, override `update()` to fill `nu_t_`:

1. **Static Smagorinsky** (~60 LOC):
   - `nu_sgs = (Cs·Δ)²·|S|`, Cs = 0.1–0.2, Δ = (dx·dy·dz)^(1/3)

2. **Dynamic Smagorinsky** (Germano/Lilly) (~200 LOC):
   - Cs² computed dynamically via test filter (2Δ) + Germano identity
   - Test filter: 3D box filter on GPU (trapezoidal weights for stretched y)
   - Plane-averaging numerator/denominator (Lilly modification)
   - Clipping: Cs² ≥ 0, or allow backscatter with magnitude cap
   - Config option: `dynamic_smag_averaging = plane | lagrangian`

3. **WALE** (Nicoud & Ducros 1999) (~80 LOC):
   - Based on traceless symmetric part of squared velocity gradient
   - Naturally vanishes at walls, Cw = 0.325–0.5

4. **Vreman** (2004) (~80 LOC):
   - Based on eigenvalues of β_ij. Vanishes at walls. Cv = 0.07

5. **Sigma** (Nicoud et al. 2011) (~100 LOC):
   - Based on singular values of velocity gradient tensor
   - Vanishes for all laminar flows (most selective)
   - 3x3 analytical SVD per cell (thread-local)
   - Cσ = 1.35–1.5

### Shared Infrastructure

- **Velocity gradient kernel** (`src/cuda_kernels/velocity_gradient.cu`, ~150 LOC):
  - Single CUDA kernel computes all 9 components of g_ij from staggered velocities
  - Shared memory tiling for stencil loads
  - Output: 9 SoA arrays at cell centers for coalesced SGS reads
  - Reusable by EARSM models (replace their current gradient computation)

### GPU Performance

- **Fused SGS kernels**: Each model is one CUDA kernel reading gradient tensor, writing nu_sgs
- **Smagorinsky/WALE/Vreman**: ~20–30 FLOPs/cell, memory-bound, <0.05 ms per kernel
- **Sigma**: ~100 FLOPs/cell (3x3 SVD), still thread-local, ~0.1 ms
- **Dynamic Smagorinsky**: Test filter stencil + reduction, ~0.15 ms. Lagrangian option eliminates plane reduction.
- **Gradient tensor persisted on GPU**: 9 × N doubles (~250 MB at 192³), computed once per RK stage
- **Future optimization**: Fuse gradient into convection kernel (save one grid traversal)
- **Total LES overhead**: <5% of step time (Poisson dominates)

### Compatibility
- Existing RANS models untouched (new enum entries alongside existing)
- MPI: only Dynamic Smagorinsky plane averages need `MPI_Allreduce`
- IBM: nu_sgs at fluid cells only, solid cells masked
- Config: `turb_model = Smagorinsky | DynamicSmagorinsky | WALE | Vreman | Sigma`

## Phase 5: HDF5 I/O + Validation Campaign (~800 LOC)

### HDF5 Checkpoint/Restart

(`include/checkpoint.hpp`, `src/checkpoint.cpp`, ~300 LOC)

- **Writer**: HDF5 C API, fields (u,v,w,p,nu_t,k,omega), metadata (step, time, dt, config, grid)
- **MPI-parallel**: Each rank writes local slab via `H5Pset_fapl_mpio`
- **Single-GPU fallback**: Serial HDF5
- **Restart**: Reads HDF5, redistributes if nprocs changed, warm-starts Poisson
- **Field dumps**: High-frequency output for offline spectral analysis
- **Config**: `checkpoint_interval = 500`, `restart_file = checkpoints/step_5000.h5`
- **Optional dependency**: `find_package(HDF5)`, disabled without HDF5

### Validation Cases

1. **Channel DNS**:
   - Re_tau = 180: 192x128x192, match MKM (1999)
   - Re_tau = 395: 384x192x384, match MKM
   - Re_tau = 590: 576x384x576 (multi-GPU), match Del Álamo & Jiménez (2003)

2. **Channel LES**:
   - Re_tau = 590 on 64x64x64 with WALE/Dynamic Smag/Vreman
   - Compare SGS models against DNS reference

3. **TGV LES**:
   - Re = 1600, 64³ with explicit SGS models
   - Compare dissipation rate vs Brachet et al. spectral DNS

4. **Cylinder**:
   - Re = 100 (2D laminar): Cd ≈ 1.35, St ≈ 0.164 vs Henderson (1995)
   - Re = 300 (3D wake): Mode A/B, Cd ≈ 1.38, St ≈ 0.21
   - Re = 3900 (turbulent LES): Match Parnaudeau et al. (2008) wake profiles

5. **Airfoil**:
   - NACA 0012 at Re = 1000, AoA = 0°/5°/10°: laminar separation bubble, Cl/Cd
   - NACA 0012 at Re = 50000, AoA = 5°: LES, Cl vs experiment

6. **Duct DNS**:
   - Re_b = 3500: match Pinelli et al. (2010) secondary flows

### Test Suite (~15 new tests)

**Unit tests**:
- `test_ibm_sdf.cpp`: SDF correctness (cylinder, sphere, NACA)
- `test_ibm_forcing.cpp`: Stokes flow drag convergence
- `test_les_sgs.cpp`: Each SGS model on known gradient field; verify WALE/Vreman/Sigma vanish in pure shear
- `test_dynamic_smag.cpp`: Cs² recovery for HIT
- `test_halo_exchange.cpp`: MPI halo correctness (rank-dependent fill)
- `test_mpi_poisson.cpp`: Distributed Poisson matches serial (Poiseuille)
- `test_mpi_channel.cpp`: 2-rank step matches 1-rank to machine precision
- `test_checkpoint.cpp`: Write/restart/verify bitwise
- `test_cuda_smoother.cpp`: CUDA smoother matches OpenMP target to round-off
- `test_velocity_gradient.cpp`: Gradient on linear/quadratic fields

**Integration tests**:
- `test_cylinder_re100.cpp`: 1000 steps, Cd within 2% of 1.35
- `test_les_channel.cpp`: 500 steps with WALE, stable, reasonable Re_tau

**CI**:
- CPU CI: unit + fast integration, MPI with `mpirun -np 2`
- GPU CI: CUDA kernel tests, multi-GPU if 2+ GPUs, LES smoke tests

## File Summary

### New Files (~25)

```
src/cuda_kernels/mg_smoother.cu        # Phase 1
src/cuda_kernels/halo_pack.cu          # Phase 1
src/cuda_kernels/velocity_gradient.cu  # Phase 4
include/decomposition.hpp              # Phase 2
src/decomposition.cpp                  # Phase 2
src/halo_exchange.cpp                  # Phase 2
src/poisson_solver_fft_mpi.cpp         # Phase 2
include/ibm_geometry.hpp               # Phase 3
src/ibm_geometry.cpp                   # Phase 3
include/ibm_forcing.hpp                # Phase 3
src/ibm_forcing.cpp                    # Phase 3
src/ibm_generators.cpp                 # Phase 3
src/ibm_diagnostics.cpp                # Phase 3
include/turbulence_les.hpp             # Phase 4
src/turbulence_les.cpp                 # Phase 4
include/checkpoint.hpp                 # Phase 5
src/checkpoint.cpp                     # Phase 5
app/main_cylinder.cpp                  # Phase 3
app/main_airfoil.cpp                   # Phase 3
tests/test_ibm_sdf.cpp                 # Phase 3
tests/test_ibm_forcing.cpp             # Phase 3
tests/test_les_sgs.cpp                 # Phase 4
tests/test_dynamic_smag.cpp            # Phase 4
tests/test_halo_exchange.cpp           # Phase 2
tests/test_mpi_poisson.cpp             # Phase 2
tests/test_mpi_channel.cpp             # Phase 2
tests/test_checkpoint.cpp              # Phase 5
tests/test_cuda_smoother.cpp           # Phase 1
tests/test_velocity_gradient.cpp       # Phase 4
```

### Modified Files (~10)
```
CMakeLists.txt                         # All phases (CUDA, MPI, HDF5)
include/config.hpp                     # New enums (SGS models, IBM config)
src/config.cpp                         # Parse new config options
src/solver_time.cpp                    # IBM forcing insertion point
src/poisson_solver_multigrid.cpp       # CUDA smoother dispatch
src/solver.cpp                         # Decomposition injection, IBM init
src/solver_recycling.cpp               # MPI recycle plane broadcast
app/main_channel.cpp                   # MPI init
app/main_duct.cpp                      # MPI init
app/main_taylor_green_3d.cpp           # MPI init
```

### Total Estimated LOC: ~4400 new + ~300 modified
