# LES and IBM GPU Acceleration Guide

**Date:** March 2026
**Status:** Production-ready, benchmarked on RTX 6000 (CC 7.5)

## Overview

The LES subgrid-scale models and immersed boundary method (IBM) run entirely on GPU with zero CPU↔GPU synchronization during time stepping. This eliminates memory transfer bottlenecks that previously dominated IBM runtime.

**Supported LES models (all GPU-accelerated):**
- Smagorinsky
- WALE
- Vreman
- Sigma
- Dynamic Smagorinsky

**IBM features:**
- Direct-forcing method with pre-computed weight arrays
- Cell classification: Fluid / Forcing / Solid
- GPU-resident weight multiplication (no virtual function calls on device)
- GPU-resident Poisson RHS masking (no CPU sync)

## Architecture

### LES GPU Design

Each LES model has a fused GPU kernel that computes velocity gradients and nu_sgs in a single pass, avoiding intermediate storage:

```
┌─────────────────────────────────────────────┐
│  GPU Kernel (one per model)                 │
│                                             │
│  for each cell (i,j,k):                     │
│    1. Compute 9-component velocity gradient  │
│    2. Compute filter width from mesh (yf)    │
│    3. Compute nu_sgs (model-specific)        │
│    4. Write to nu_t array                    │
└─────────────────────────────────────────────┘
```

Key design decisions:
- **Free functions with `#pragma omp declare target`** — avoids nvc++ implicit `this` transfer
- **All data via `TurbulenceDeviceView`** — struct of raw pointers + scalars, copied to locals before kernel
- **Stretched grid support** — gradient kernels use `yc[]` (cell centers) and `yf[]` (face positions) for non-uniform dy
- **2D/3D unified** — same kernel handles both via `is2D` flag; 2D skips w-gradients and uses 2D filter width

### IBM GPU Design

IBM uses pre-computed weight arrays instead of runtime geometry evaluation:

```
Classification (CPU, once at init):
  body.phi(x,y,z) → IBMCellType → weight

Weight values:
  Fluid:   weight = 1.0 (pass-through)
  Solid:   weight = 0.0 (zero velocity)
  Forcing: weight = |phi|/band_width (smooth transition)

GPU forcing (every step):
  u[i] *= weight_u[i]    // element-wise multiply
  v[i] *= weight_v[i]
  w[i] *= weight_w[i]
```

Three sets of weight arrays exist:
- `weight_u_` — at u-face locations (staggered)
- `weight_v_` — at v-face locations (staggered)
- `weight_w_` — at w-face locations (staggered, 3D only)
- `solid_mask_cell_` — at cell centers (for Poisson RHS masking)

All arrays are mapped to GPU via `target enter data map(to:...)` at initialization and use `map(present:...)` during compute.

### Solver Integration

IBM forcing is applied twice per time step:

1. **After predictor, before Poisson solve** — forces predicted velocity u* toward zero in/near body
2. **After velocity correction** — re-enforces no-slip on corrected velocity u^{n+1}

Between these, the Poisson RHS is masked at solid cells (set to zero) to prevent spurious pressure gradients inside the body. This masking runs as a GPU kernel using the pre-computed `solid_mask_cell_` array.

```
step():
  ...
  compute u* (predictor)
  IBM: u* *= weight          ← apply_forcing_device()
  compute RHS = div(u*)/dt
  IBM: RHS *= solid_mask     ← mask_rhs_device()
  solve ∇²p' = RHS
  correct u = u* - dt·∇p'
  IBM: u *= weight           ← apply_forcing_device()
  ...
```

## Key Source Files

| File | Purpose |
|---|---|
| `include/turbulence_les.hpp` | LES model class hierarchy, `update_gpu()` virtual method |
| `src/turbulence_les.cpp` | GPU kernels for all 5 LES models |
| `include/turbulence_model.hpp` | `TurbulenceDeviceView` struct (GPU data interface) |
| `include/ibm_geometry.hpp` | `IBMBody` base class, `CylinderBody`, `SphereBody` |
| `include/ibm_forcing.hpp` | `IBMForcing` class with GPU methods |
| `src/ibm_forcing.cpp` | Weight computation, GPU mapping, device kernels |
| `tests/bench_les_ibm_gpu.cpp` | 3D benchmark (LES, LES+IBM, laminar baseline) |

## TurbulenceDeviceView

The `TurbulenceDeviceView` struct is the GPU data interface between the solver and turbulence models. It contains raw pointers to GPU-resident arrays and mesh parameters:

```cpp
struct TurbulenceDeviceView {
    // Velocity (staggered faces, solver-owned)
    double* u_face, *v_face, *w_face;
    int u_stride, v_stride, w_stride;
    int u_plane_stride, v_plane_stride, w_plane_stride;

    // Turbulence fields (cell-centered)
    double* nu_t;
    int cell_stride, cell_plane_stride;

    // Mesh coordinates (for stretched grids)
    const double* yf;    // face y-positions
    const double* yc;    // cell center y-positions

    // Mesh parameters (scalars)
    int Nx, Ny, Nz, Ng;
    double dx, dy, dz;

    // Array sizes (for map(present: ptr[0:size]) clauses)
    int u_total, v_total, w_total;
    int cell_total, yf_total, yc_total;
};
```

**Critical rule:** GPU kernels must use `map(present: ptr[0:size])` with array sections — bare pointer names (e.g., `map(present: u)`) do not work with nvc++ and will cause runtime failures or NaN.

## Adding a New LES Model

1. Add model class in `include/turbulence_les.hpp`:
```cpp
class MyModel : public LESModel {
public:
    double compute_nu_sgs_cell(const double g[9], double delta) const override;
    void update_gpu(const TurbulenceDeviceView* dv) override;
    std::string name() const override { return "MyModel"; }
};
```

2. Add device-callable kernel function in `src/turbulence_les.cpp`:
```cpp
#pragma omp declare target
inline double my_model_nu_sgs(const double g[9], double param, double delta) {
    // Compute nu_sgs from velocity gradient tensor g and filter width delta
    return ...;
}
#pragma omp end declare target
```

3. Implement `update_gpu()` following the Smagorinsky pattern:
```cpp
void MyModel::update_gpu(const TurbulenceDeviceView* dv) {
    // Copy all needed values to locals (avoid this transfer)
    const int Nx = dv->Nx, Ny = dv->Ny, Ng = dv->Ng;
    // ... extract all pointers, strides, sizes ...
    const int u_sz = dv->u_total, v_sz = dv->v_total, w_sz = dv->w_total;
    const int nut_sz = dv->cell_total, yf_sz = dv->yf_total, yc_sz = dv->yc_total;

    #pragma omp target teams distribute parallel for \
        map(present: u[0:u_sz], v[0:v_sz], w[0:w_sz], \
                     nu_t_ptr[0:nut_sz], yf[0:yf_sz], yc[0:yc_sz]) \
        firstprivate(Nx, Ny, Nz_eff, Ng, dx, dz, ...)
    for (int idx = 0; idx < total; ++idx) {
        // Decompose linear index → i, j, k
        // Compute gradient
        // Compute nu_sgs
        // Write to nu_t
    }
}
```

4. Register in factory (`src/config.cpp`) and add to `TurbulenceModelType` enum.

## GPU Mapping Lifecycle

```
Constructor:
  classify_cells()         ← CPU: evaluate phi at all face locations
  compute_weights()        ← CPU: convert cell types to weight arrays

solver.set_ibm_forcing():
  ibm_->map_to_gpu()       ← target enter data map(to: weight_u, weight_v, weight_w, solid_mask)

During stepping:
  apply_forcing_device()   ← target teams: u *= weight (map present)
  mask_rhs_device()        ← target teams: rhs *= mask (map present)

Cleanup:
  ibm_->unmap_from_gpu()   ← target exit data map(delete: ...)
```

For moving bodies, call `classify_cells()` + `compute_weights()` again, then `target update to(...)` to refresh GPU arrays.

## Benchmark Results

### RTX 6000 (Quadro RTX 6000, 24 GB, Turing CC 7.5)

**Grid: 128×64×128 (1.0M cells), MG Poisson solver, 20 steps**

| Configuration | ms/step | Mcells/s | Poisson (ms) | Turb (ms) |
|---|---|---|---|---|
| Laminar baseline | 12.70 | 82.6 | 9.20 | — |
| Smagorinsky LES | 22.28 | 47.1 | 18.95 | 0.77 |
| Smagorinsky LES + IBM | 20.16 | 52.0 | 16.59 | 0.71 |

**Grid: 256×128×256 (8.4M cells), MG Poisson solver, 10 steps**

| Configuration | ms/step | Mcells/s | Poisson (ms) | Turb (ms) |
|---|---|---|---|---|
| Laminar baseline | 89.44 | 93.8 | 64.18 | — |
| Smagorinsky LES | 145.12 | 57.8 | 120.25 | 5.81 |
| Smagorinsky LES + IBM | 143.06 | 58.6 | 118.70 | 5.45 |

### Timing Breakdown (256³, LES + IBM)

```
poisson_solve       118.70 ms   82.5%  ← Multigrid V-cycles dominate
turbulence_update     5.45 ms    3.8%  ← Fused gradient + nu_sgs kernel
convective_term       4.71 ms    3.3%
diffusive_term        4.61 ms    3.2%
velocity_correction   1.52 ms    1.1%
divergence            1.15 ms    0.8%
IBM forcing           <0.5 ms   <0.3%  ← Element-wise weight multiply
RHS masking           <0.5 ms   <0.3%  ← Element-wise mask multiply
```

### Key Performance Observations

1. **IBM overhead is negligible** — LES+IBM runs at the same speed as LES alone. The weight multiplication kernels are trivially fast (element-wise multiply on 8M elements).

2. **LES SGS overhead is small** (~4%) — the fused gradient+nu_sgs kernel avoids intermediate storage and computes everything in one pass.

3. **Poisson solver dominates** (83%) — the multigrid V-cycle is the bottleneck. Use FFT Poisson solver when BCs allow (all-periodic or periodic-x-z with no-slip-y).

4. **CPU sync was the old bottleneck** — before the GPU RHS masking optimization, IBM was 2.1× slower due to `target update from` + CPU loop + `target update to` for the Poisson RHS solid cell masking. This transferred 64 MB per step at 8.4M cells.

### Running the Benchmark

```bash
# Build (GPU required)
cd build_gpu && make bench_les_ibm_gpu

# Run with defaults (128×64×128, 20 steps)
./bench_les_ibm_gpu

# Custom grid and steps
./bench_les_ibm_gpu 256 128 256 10

# SLURM submission
sbatch --partition=gpu-rtx6000 --qos=embers --gres=gpu:rtx_6000:1 \
    --wrap="OMP_TARGET_OFFLOAD=MANDATORY ./bench_les_ibm_gpu 256 128 256 10"
```

## Common Pitfalls

### map(present:) requires array sections
```cpp
// WRONG — bare pointer name, undefined behavior with nvc++
map(present: u, v, w)

// CORRECT — array section with size
map(present: u[0:u_sz], v[0:v_sz], w[0:w_sz])
```

### All accessed pointers must be in map clause
```cpp
// WRONG — w missing from map, will segfault in 3D
map(present: u[0:u_sz], v[0:v_sz], nu_t[0:nut_sz])
// ... kernel accesses w ...

// CORRECT
map(present: u[0:u_sz], v[0:v_sz], w[0:w_sz], nu_t[0:nut_sz])
```

### IBM must be set before GPU init or call map_to_gpu() explicitly
```cpp
// Option 1: Set IBM before solver.initialize_gpu_buffers()
solver.set_ibm_forcing(&ibm);   // calls ibm.map_to_gpu() if GPU ready
solver.initialize_gpu_buffers();

// Option 2: Set IBM after — set_ibm_forcing() auto-maps if GPU is ready
solver.initialize_gpu_buffers();
solver.set_ibm_forcing(&ibm);   // detects gpu_ready_, calls map_to_gpu()
```

### Never sync GPU data for IBM operations during stepping
```cpp
// WRONG — transfers entire field to CPU and back
#pragma omp target update from(rhs[0:n])
for (int i = 0; i < n; ++i) { if (solid(i)) rhs[i] = 0; }
#pragma omp target update to(rhs[0:n])

// CORRECT — GPU kernel with pre-computed mask
ibm->mask_rhs_device(rhs_ptr);
```

### Virtual functions cannot be called on GPU
The `IBMBody::phi()` method is virtual — it cannot be called inside `#pragma omp target` regions. All geometry evaluation must happen at initialization time on CPU, with results stored in arrays that are mapped to GPU.

## Stretched Grid Support

LES filter width adapts to local cell size on stretched grids:

```cpp
delta = cbrt(dx * dy_local * dz)    // 3D
delta = sqrt(dx * dy_local)         // 2D
```

where `dy_local = yf[j+1] - yf[j]` is the local cell height from the face position array. The velocity gradient kernels use `yc[]` for central difference spacing (`dy_central = yc[j+1] - yc[j-1]`).
