# GPU Turbulence Verification Suite

This example suite verifies that the solver can produce **real turbulence** on NVIDIA H200 GPUs across all three main 3D geometries.

## Quick Start

```bash
# On login node: build the code
module load nvhpc
./build_gpu.sh

# Submit to H200 queue (runs all cases)
sbatch run_h200.sbatch

# Or run interactively on a GPU node
srun --partition=gpu-h200 --gres=gpu:h200:1 --time=01:00:00 --pty bash
./run.sh all
```

## Test Cases Overview

| Case | Geometry | BCs | Turbulence | Purpose |
|------|----------|-----|------------|---------|
| `tgv_re100` | Cube | All periodic | None (laminar) | Validate solver accuracy |
| `tgv_re1600` | Cube | All periodic | DNS | Transition to turbulence |
| `channel_sst` | Channel | Periodic x/z, walls y | SST k-ω | RANS validation |
| `channel_earsm` | Channel | Periodic x/z, walls y | EARSM | Anisotropic RANS |
| `duct_sst` | Duct | Periodic x, walls y/z | SST k-ω | Isotropic baseline |
| `duct_earsm` | Duct | Periodic x, walls y/z | EARSM | Secondary flow test |

---

## 1. Taylor-Green Vortex (All-Periodic)

The Taylor-Green vortex is THE classic DNS benchmark for transition to turbulence.

### Re = 100 (Validation)

```bash
./run.sh tgv_re100
```

**Purpose**: Verify solver accuracy against analytical solution.

At low Reynolds number, the flow remains laminar with exponential energy decay:
```
KE(t) = KE(0) * exp(-2νt)
```

**Success criterion**: KE decay error < 5%

### Re = 1600 (Transition to Turbulence)

```bash
./run.sh tgv_re1600
```

**Purpose**: Demonstrate vortex stretching and turbulence transition.

**What to look for**:
- Kinetic energy decays slowly initially (t < 5)
- Enstrophy peaks around t = 8-10 (turbulence transition)
- Energy cascade to small scales
- Chaotic vortex structures in VTK output

**Reference**: Brachet et al. (1983), JFM 130, 411-452

---

## 2. Channel Flow (Periodic x/z, Walls y)

Turbulent channel flow is the fundamental wall-bounded turbulence benchmark.

### Re_τ = 180 with SST k-ω

```bash
./run.sh channel_sst
```

**Purpose**: Verify RANS turbulence modeling against DNS data.

**Expected results**:
- Friction Reynolds number: Re_τ ≈ 180
- Viscous sublayer: U⁺ = y⁺ for y⁺ < 5
- Log layer: U⁺ = (1/κ)ln(y⁺) + B where κ ≈ 0.41, B ≈ 5.2
- Bulk velocity: U_b ≈ 15-18

**Reference**: Moser, Kim & Mansour (1999), JFM 399, 263-291

### Re_τ = 180 with EARSM

```bash
./run.sh channel_earsm
```

**Purpose**: Test anisotropic Reynolds stress modeling.

EARSM (Explicit Algebraic Reynolds Stress Model) captures:
- Reynolds stress anisotropy (u'u' ≠ v'v' ≠ w'w')
- Better prediction of normal stresses
- Improved log-law behavior compared to isotropic models

---

## 3. Square Duct Flow (Periodic x, Walls y/z)

The square duct is a **critical test** for anisotropic turbulence models.

### Why Duct Flow Matters

In turbulent square duct flow, there are **secondary flows** (Prandtl's second kind):
- Eight corner vortices that transport momentum
- v and w velocities ~1-2% of bulk streamwise velocity
- These are driven by Reynolds stress anisotropy

**Key insight**: Isotropic models (SST, k-ω) CANNOT predict secondary flows!

### SST k-ω (Baseline)

```bash
./run.sh duct_sst
```

**Expected**: Turbulent velocity profile with NO secondary motion (v ≈ w ≈ 0)

### EARSM (Anisotropic)

```bash
./run.sh duct_earsm
```

**Expected**:
- Secondary corner vortices visible in v-w plane
- v, w magnitudes ~1-2% of bulk u
- Iso-velocity contours bulge toward corners

**This is THE litmus test** for whether anisotropic turbulence modeling is working!

---

## Configuration Parameters

### Channel Flow (`channel_retau180_*.cfg`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Grid | 64 × 128 × 64 | Stretched in y for wall resolution |
| Domain | 2π × 2 × π | Standard minimal channel |
| Re_τ | 180 | ν = 1/180 ≈ 0.00556 |
| dp/dx | -1.0 | Pressure gradient driving flow |
| Stretch β | 2.5 | Tanh stretching parameter |

### Duct Flow (`duct_turbulent_*.cfg`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Grid | 32 × 64 × 64 | Uniform (no stretching) |
| Domain | 4π × 2 × 2 | Square cross-section |
| ν | 0.002 | Moderate Re |
| dp/dx | -1.0 | Pressure gradient |

### Taylor-Green (`tgv_re*.cfg`)

| Parameter | Re=100 | Re=1600 | Notes |
|-----------|--------|---------|-------|
| Grid | 32³ | 64³ | Uniform cubic |
| Domain | [0, 2π]³ | [0, 2π]³ | Periodic cube |
| ν | 0.01 | 0.000625 | ν = 1/Re |
| Scheme | Skew | Skew | Energy-conserving |
| Integrator | RK3 | RK3 | 3rd-order accurate |

---

## Output Files

Each case writes to its own output directory:

```
output/
├── tgv_re100_validation/
│   ├── tg3d_*.vtk          # Velocity snapshots
│   ├── tg3d_history.dat    # Time history (KE, enstrophy)
│   └── tg3d_final.vtk      # Final state
├── tgv_re1600_dns/
│   └── ...
├── channel_retau180_sst/
│   ├── channel_*.vtk       # Snapshots
│   ├── velocity_profile.dat # Mean profile vs y
│   └── channel_final.vtk
├── duct_turbulent_earsm/
│   ├── duct_final.vtk
│   └── duct_profile.dat    # y-z velocity profile
└── *.log                   # Console output logs
```

---

## Visualization

### ParaView

```bash
# View all VTK files from a case
paraview output/tgv_re1600_dns/tg3d_*.vtk

# For duct secondary flow, plot v-w vectors in y-z plane:
# 1. Load duct_final.vtk
# 2. Apply Slice filter (normal to x)
# 3. Apply Glyph filter with v,w velocity components
```

### Gnuplot

```bash
# TGV energy decay
gnuplot -e "plot 'output/tgv_re100_validation/tg3d_history.dat' u 1:3 w l title 'KE/KE0', exp(-0.02*x) title 'Theory'"

# Channel velocity profile
gnuplot -e "plot 'output/channel_retau180_sst/velocity_profile.dat' u 1:2 w l title 'U(y)'"
```

---

## Validation Checklist

After running all cases, verify:

- [ ] **TGV Re=100**: KE decay matches exp(-2νt) within 5%
- [ ] **TGV Re=1600**: Enstrophy peak visible around t=8-10
- [ ] **Channel**: Re_τ computed ≈ 180 (within 10%)
- [ ] **Channel**: Log-law region visible in U⁺(y⁺) plot
- [ ] **Duct SST**: v and w are essentially zero (< 0.1% of u)
- [ ] **Duct EARSM**: v and w are ~1-2% of bulk u (secondary flow!)

---

## Troubleshooting

**Simulation diverges (NaN)**:
- Reduce CFL: `--CFL_max 0.3`
- Reduce time step: `--dt 0.0005`
- Check grid resolution is adequate

**Poor agreement with reference**:
- Increase wall-normal resolution for channel
- Ensure simulation reached steady state
- Check boundary conditions are correct

**No secondary flow in duct EARSM**:
- Ensure using `turb_model = earsm_wj` (not sst)
- Run longer to reach steady state
- Check v,w fields in VTK (not just u)

**GPU out of memory**:
- Reduce grid size
- Check `nvidia-smi` for memory usage

---

## References

1. **Taylor-Green DNS**: Brachet, M.E. et al. (1983), JFM 130, 411-452
2. **Channel DNS**: Moser, R.D., Kim, J. & Mansour, N.N. (1999), JFM 399, 263-291
3. **SST k-ω**: Menter, F.R. (1994), AIAA Journal 32(8), 1598-1605
4. **EARSM**: Wallin, S. & Johansson, A.V. (2000), Physics of Fluids 12(11), 2799-2813
5. **Duct Secondary Flow**: Demuren, A.O. & Rodi, W. (1984), JFM 140, 189-222
