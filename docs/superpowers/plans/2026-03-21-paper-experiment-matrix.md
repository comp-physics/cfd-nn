# Paper Experiment Matrix: NN vs Traditional RANS Cost-Accuracy Tradeoff

**Last updated: Mar 24, 2026**

## Goal

Determine whether NN-based turbulence closures can match or exceed the accuracy of traditional RANS models, and at what computational cost. Central deliverable: Pareto plot of cost vs accuracy across 20 models and 8 flow configurations.

## A Posteriori Cases (4 geometries, 8 configurations)

### Case overview

| Geometry | Re values | Dim | Grid | Cells | Poisson | Training | Config files |
|----------|-----------|-----|------|-------|---------|----------|-------------|
| Periodic hills | 10595 | 2D | 384×192×1 | 74K | FFT2D | Trained | `hills_re10595.cfg` |
| Cylinder | 100, 300, 3900 | 2D | 384×288×1 | 111K | FFT2D | Unseen | `cylinder_re{100,300,3900}.cfg` |
| Square duct | 3500 | 3D | 96×96×96 | 885K | FFT1D | Trained | `duct_reb3500.cfg` |
| Sphere | 100, 200, 300 | 3D | 192×128×128 | 3.15M | FFT 3D | Unseen | `sphere_re{100,200,300}.cfg` |

**Design rationale:**
- 2 trained + 2 unseen geometries (tests generalization)
- 2 2D + 2 3D cases (tests scaling)
- Re sweeps for IBM bodies (tests Reynolds number generalization)
- Cylinder is 2D even at Re=3900 — RANS models the turbulence, doesn't resolve it
- All configs in `examples/paper_experiments/`

### Physics at each Re

**Cylinder:**
- Re=100: laminar vortex shedding (Cd≈1.35, St≈0.165)
- Re=300: transitional (Cd≈1.38, St≈0.21)
- Re=3900: turbulent sub-critical wake (Cd≈0.98, St≈0.21, Parnaudeau et al. 2008)

**Sphere:**
- Re=100: steady axisymmetric (Cd≈1.09, sep≈127°)
- Re=200: steady non-axisymmetric (Cd≈0.77, sep≈117°, Johnson & Patel 1999)
- Re=300: unsteady vortex shedding (Cd≈0.66, St≈0.135)

## Quantities of Interest (QoIs) — ALL IMPLEMENTED

### Hills (in `app/main_hills.cpp`)

| QoI | Function | Output file |
|-----|----------|-------------|
| Separation point x_s/H | `find_separation_reattachment()` | stdout |
| Reattachment point x_r/H | `find_separation_reattachment()` | stdout |
| Skin friction Cf(x) | `compute_cf_x_device()` | `cf_x.dat` |
| Velocity profiles u(y) at x/H=0.5,2,4,6,8 | `extract_velocity_profile_device()` | `profile_xH*.dat` |

### Cylinder (in `app/main_cylinder.cpp`)

| QoI | Function | Output file |
|-----|----------|-------------|
| Mean Cd (2nd half) | IBM `compute_forces()` | `qoi_summary.dat` |
| Strouhal number St | `compute_strouhal()` (Cl zero-crossings) | `qoi_summary.dat` |
| Wake profiles u(y) at x/D=1,2,3,5 | `extract_wake_profile()` | `wake_xD*.dat` |
| Force time series | IBM `compute_forces()` | `forces.dat` |

### Duct (in `app/main_duct.cpp`)

| QoI | Function | Output file |
|-----|----------|-------------|
| Cross-section (u,v,w)(y,z) | `extract_cross_section_device()` | `duct_cross_section.dat` |
| Wall shear tau_w(z) | `compute_wall_shear_y_device()` | `wall_shear_y.dat` |

### Sphere (in `app/main_cylinder.cpp`, `ibm_body = sphere`)

| QoI | Function | Output file |
|-----|----------|-------------|
| Mean Cd (2nd half) | IBM `compute_forces()` | `qoi_summary.dat` |
| Separation angle | `compute_separation_angle_sphere()` | `qoi_summary.dat` |
| Strouhal number St | `compute_strouhal()` | `qoi_summary.dat` |
| Wake profiles u(y) at x/D=1,2,3,5 | `extract_wake_profile()` | `wake_xD*.dat` |
| Force time series | IBM `compute_forces()` | `forces.dat` |

### Error metrics

- **L2 relative error** vs reference: `||f_model - f_ref||_2 / ||f_ref||_2`
- Scalar QoIs (Cd, St, sep angle): relative error `|model - ref| / |ref|`

## Turbulence Models (20 total)

### Classical RANS (8)

| Model | Config `turb_model` value | Anisotropy? |
|-------|--------------------------|-------------|
| None (laminar) | `baseline` | No |
| Mixing length | `mixing_length` | No |
| k-omega | `komega` | No |
| SST k-omega | `sst` | No |
| EARSM-WJ | `earsm_wj` | Yes |
| EARSM-GS | `earsm_gs` | Yes |
| EARSM-Pope | `earsm_pope` | Yes |
| GEP | `gep` | Yes |

### ML (12, size variants)

| Model | Config value | Weights dir |
|-------|-------------|-------------|
| MLP | `nn_mlp` | `data/models/mlp_paper/` |
| MLP-Medium | `nn_mlp` | `data/models/mlp_med_paper/` |
| MLP-Large | `nn_mlp` | `data/models/mlp_large_paper/` |
| TBNN-Small | `nn_tbnn` | `data/models/tbnn_small_paper/` |
| TBNN | `nn_tbnn` | `data/models/tbnn_paper/` |
| TBNN-Large | `nn_tbnn` | `data/models/tbnn_large_paper/` |
| PI-TBNN-Small | `nn_tbnn` | `data/models/pi_tbnn_small_paper/` |
| PI-TBNN | `nn_tbnn` | `data/models/pi_tbnn_paper/` |
| PI-TBNN-Large | `nn_tbnn` | `data/models/pi_tbnn_large_paper/` |
| TBRF-1t | `nn_tbrf` | `data/models/tbrf_paper/` (1 tree) |
| TBRF-5t | `nn_tbrf` | `data/models/tbrf_paper/` (5 trees) |
| TBRF-10t | `nn_tbrf` | `data/models/tbrf_paper/` (10 trees) |

## Full Experiment Matrix

**20 models × 8 configurations = 160 runs**

All on H200 GPU, same compiler (nvc++ Release), same binary per geometry.

Turb model set via `turb_model` in config file (NOT via `--turb_model` CLI — that's silently ignored).

## SLURM Configuration

- **Partition:** gpu-h200
- **Account:** gts-sbryngelson3
- **QOS:** embers
- **GPUs:** 1 per job

### Time estimates per case

| Case | Grid cells | Est. ms/step (baseline) | Steps needed | Wall time |
|------|-----------|------------------------|-------------|-----------|
| Hills 2D | 74K | ~3.7 | 15,000 | ~1 min |
| Cylinder 2D | 111K | ~2.3 | 15,000 | ~1 min |
| Duct 3D | 885K | ~2.7 | 10,000 | ~0.5 min |
| Sphere 3D | 3.15M | ~5.6 | 10,000 | ~1 min |

NN models are slower (up to 650 ms/step for MLP-Large on sphere). Worst case: ~2 hours for MLP-Large × sphere. Total for all 160 runs: ~6-8 hours wall time if serialized, ~1 hour with 20 concurrent jobs.

## Output Structure

```
results/paper/aposteriori/
  hills_re10595/{model_name}/qoi/
  cylinder_re100/{model_name}/qoi/
  cylinder_re300/{model_name}/qoi/
  cylinder_re3900/{model_name}/qoi/
  duct_reb3500/{model_name}/qoi/
  sphere_re100/{model_name}/qoi/
  sphere_re200/{model_name}/qoi/
  sphere_re300/{model_name}/qoi/
```

Each `qoi/` directory contains the case-specific output files listed above.

## H200 Timing Data (ALREADY COLLECTED)

Job 5470248 (Mar 24) collected ms/step for all 20 models × 4 original cases (hills, cylinder Re=100, duct, sphere Re=200). New Re values will produce additional timing data from the production runs.

Results: `results/paper/aposteriori/h200_5470248.out`

## Paper Figures (planned)

1. **Pareto plot** (THE figure): cost vs accuracy, one curve per architecture, points = size variants
2. **Per-step cost breakdown**: stacked bar (convection, diffusion, Poisson, turbulence)
3. **Cd/St vs Re**: cylinder and sphere Re-sweep, all models
4. **Hills Cf(x)**: all models vs Breuer DNS
5. **Hills velocity profiles**: at x/H = 2, 4, 6
6. **Duct cross-section**: secondary flow vectors, all models vs Pinelli DNS
7. **Wake profiles**: cylinder and sphere, downstream evolution
8. **Separation angle vs Re**: sphere, all models vs Johnson & Patel
