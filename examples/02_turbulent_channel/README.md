# Example 2: Turbulent Channel Flow - Model Comparison

## Overview

Compare **5 turbulence closure models** for turbulent channel flow at Re_τ = 180:

1. **None** - No turbulence model (underpredicts)
2. **Baseline** - Mixing length with van Driest damping
3. **GEP** - Gene Expression Programming (Weatheritt & Sandberg inspired)
4. **NN-MLP** - Neural network scalar eddy viscosity
5. **NN-TBNN** - Tensor Basis Neural Network (most sophisticated)

**Purpose**: Demonstrate the performance of different turbulence modeling approaches on a canonical test case with known DNS reference data.

## Physical Problem

**Geometry**: 2D turbulent channel flow
- Domain: 6.28 x 2.0 (~= 2π x H)
- Walls at y = ±1.0

**Flow Conditions**:
- **Re_τ = 180** (friction Reynolds number)
- Bulk Reynolds number: Re ~= 3000
- Viscosity: ν = 0.000667
- Pressure gradient: dp/dx = -0.0002

**DNS Reference**: Moser, Kim, & Mansour (1999) - DNS of turbulent channel flow at Re_τ = 180, 395, 590

## Running the Examples

### Quick Start (Run All Models)

```bash
cd examples/02_turbulent_channel
./run_all.sh
```

This runs all 5 models sequentially and generates comparison plots.

**Expected time**: 10-30 minutes total (2-6 minutes per model)

### Run Individual Models

```bash
cd ../../build

# No turbulence model
./channel --config ../examples/02_turbulent_channel/01_no_model.cfg \
          --output_dir ../examples/02_turbulent_channel/output/no_model

# Baseline (mixing length)
./channel --config ../examples/02_turbulent_channel/02_baseline.cfg \
          --output_dir ../examples/02_turbulent_channel/output/baseline

# GEP
./channel --config ../examples/02_turbulent_channel/03_gep.cfg \
          --output_dir ../examples/02_turbulent_channel/output/gep

# NN-MLP (requires trained weights for real results)
./channel --config ../examples/02_turbulent_channel/04_nnmlp.cfg \
          --output_dir ../examples/02_turbulent_channel/output/nn_mlp

# NN-TBNN (requires trained weights for real results)
./channel --config ../examples/02_turbulent_channel/05_nntbnn.cfg \
          --output_dir ../examples/02_turbulent_channel/output/nn_tbnn
```

## Expected Results

### Physics Models (No Training Needed)

| Model | u_τ | Re_τ | vs DNS | Notes |
|-------|-----|------|--------|-------|
| **DNS (Target)** | 0.120 | 180 | - | Moser et al. 1999 |
| **None** | ~0.08 | ~120 | Poor | Underpredicts (no turbulence) |
| **Baseline** | ~0.115 | ~172 | Fair | Simple algebraic model |
| **GEP** | ~0.118 | ~177 | Good | Improved algebraic formulas |

### Neural Network Models

| Model | With Example Weights | With Trained Weights |
|-------|---------------------|---------------------|
| **NN-MLP** | Random (meaningless) | Good (~Re_τ = 175-180) |
| **NN-TBNN** | Random (meaningless) | Best (~Re_τ = 177-182) |

[WARNING] **Important**: NN models with example weights produce random results! You need to train them first (see below).

## Training Neural Network Models

To get real results from NN models:

```bash
# 1. Download McConkey dataset (one-time)
bash scripts/download_mcconkey_data.sh

# 2. Train MLP model (10-15 min)
python scripts/train_mlp_mcconkey.py \
    --data_dir mcconkey_data \
    --case channel \
    --output data/models/mlp_channel_real \
    --epochs 100

# 3. Train TBNN model (20-30 min)
python scripts/train_tbnn_mcconkey.py \
    --data_dir mcconkey_data \
    --case channel \
    --output data/models/tbnn_channel_real \
    --epochs 100

# 4. Update config files to use trained models
# Edit 04_nnmlp.cfg: nn_preset = mlp_channel_real
# Edit 05_nntbnn.cfg: nn_preset = tbnn_channel_real

# 5. Re-run comparison
./run_all.sh
```

See `docs/TRAINING_GUIDE.md` for full training instructions.

## Visualization

### Automated Comparison

```bash
python compare_models.py
```

Generates:
- **Inner scaling plot** (u+ vs y+) - compares against log law and DNS
- **Physical coordinates plot** - mean velocity profiles
- **Summary statistics** - u_τ, Re_τ for each model

### ParaView Visualization

```bash
# Open all final solutions
paraview output/*/velocity_final.vtk
```

**Recommended workflow**:
1. Load all 5 VTK files
2. Apply "Plot Over Line" filter along y-direction
3. Compare velocity profiles side-by-side
4. Color by eddy viscosity (nu_t) to see turbulence model predictions

## Interpretation Guide

### Velocity Profile (Inner Scaling)

**What to look for**:
- [OK] **Viscous sublayer** (y+ < 5): Should follow u+ = y+
- [OK] **Log layer** (30 < y+ < 100): Should follow u+ = 1/κ ln(y+) + B
- [OK] **Wake region** (y+ > 100): Deviates from log law

**Model Performance**:
- **None**: Underpredicts everywhere (no turbulent mixing)
- **Baseline**: Good in log layer, slightly off in wake
- **GEP**: Better overall, especially in outer layer
- **NN (trained)**: Best match to DNS across all regions

### Eddy Viscosity Distribution

Extract nu_t from VTK files to see:
- **Baseline**: Parabolic distribution (κy)²|S|
- **GEP**: More complex, accounts for strain/rotation
- **NN**: Learns optimal distribution from data

## Success Criteria

[OK] **Excellent**: Re_τ = 178-182 (within 1% of DNS)  
[OK] **Good**: Re_τ = 170-178 (within 5%)  
[WARNING] **Fair**: Re_τ = 160-170 (qualitatively correct)  
[FAIL] **Poor**: Re_τ < 160 or > 190 (model issue)

## Output Files

```
output/
├── no_model/
│   ├── velocity_0000.vtk
│   ├── ...
│   └── velocity_final.vtk
├── baseline/
│   └── ...
├── gep/
│   └── ...
├── nn_mlp/
│   └── ...
├── nn_tbnn/
│   └── ...
└── model_comparison.png
```

## Troubleshooting

### Models Don't Converge

- **Increase max_iter**: Edit `config_base.cfg`, set `max_iter = 100000`
- **Reduce CFL**: Change `CFL_max = 0.3` for more stability
- **Check turbulence model**: NN models with bad weights may not converge

### NN Models Give Nonsense Results

- **Cause**: Using example weights (random initialization)
- **Solution**: Train real models on McConkey dataset
- **Workaround**: Focus comparison on None/Baseline/GEP for now

### Results Don't Match DNS

- **Grid too coarse**: Increase `Ny = 256` for better wall resolution
- **Wrong Re_τ**: Adjust `dp_dx` to tune Re_τ to exactly 180
- **Numerical dissipation**: Try `convection_scheme = central` instead of upwind

## Extensions

### 1. Reynolds Number Sweep

Modify `dp_dx` to achieve different Re_τ:

```bash
# Re_tau = 395 (increase pressure gradient)
# dp_dx = -0.0008

# Re_tau = 590 (even higher)
# dp_dx = -0.002
```

Compare how models perform at higher Reynolds numbers.

### 2. Grid Resolution Study

Test grid sensitivity:
```bash
# Coarse: Ny = 64
# Medium: Ny = 128 (default)
# Fine: Ny = 256
```

### 3. Convection Scheme Comparison

Edit `config_base.cfg`:
```
convection_scheme = central  # 2nd order, may oscillate
convection_scheme = upwind   # 1st order, stable
```

### 4. Custom Turbulence Model

Implement your own model in `src/turbulence_custom.cpp` and add to comparison!

## References

1. **DNS Data**: Moser, R. D., Kim, J., & Mansour, N. N. (1999). "Direct numerical simulation of turbulent channel flow up to Re_τ=590." *Physics of Fluids*, 11(4), 943-945.

2. **Mixing Length**: Van Driest, E. R. (1956). "On turbulent flow near a wall." *J. Aero. Sci.*, 23, 1007-1011.

3. **GEP**: Weatheritt, J., & Sandberg, R. D. (2016). "A novel evolutionary algorithm applied to algebraic modifications of the RANS stress-strain relationship." *JFM*, 802, 421-448.

4. **TBNN**: Ling, J., Kurzawski, A., & Templeton, J. (2016). "Reynolds averaged turbulence modelling using deep neural networks with embedded invariance." *JFM*, 807, 155-166.

5. **McConkey Dataset**: McConkey, R., et al. (2021). "A curated dataset for data-driven turbulence modelling." *Scientific Data*, 8(1), 255.

## Related Examples

- **Example 1**: Laminar channel (same geometry, low Re)
- **Example 3**: Grid refinement (quantify numerical errors)
- **Example 4**: Validation suite (multiple test cases)

---

**Next Steps**: Use this framework to test your own turbulence models or train NN models on DNS data!

