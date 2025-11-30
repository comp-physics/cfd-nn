# Changelog

## Latest Update

### New Features

1. **Adaptive Time Stepping**
   - Automatically computes stable time step based on CFL and diffusion constraints
   - Enable with `--adaptive_dt` flag
   - Configure max CFL with `--CFL VALUE` (default 0.5)
   - No more manual dt tuning required for different Reynolds numbers

2. **VTK Output**
   - Generates ParaView-compatible VTK files
   - Includes: velocity vectors, pressure, velocity magnitude, nu_t (if turbulent)
   - Output: `<output_dir>/channel.vtk`

3. **GEP Turbulence Model**
   - Gene Expression Programming algebraic model (Weatheritt-Sandberg style)
   - No pre-trained weights needed - uses algebraic formulas
   - Enable with `--model gep`
   - Three variants: channel flow, periodic hills, simple

4. **Model Comparison Script**
   - Python script to compare multiple turbulence models
   - Generates velocity profiles and comparison plots
   - Usage: `python scripts/compare_models.py`

### Model Zoo Infrastructure

- `--nn_preset NAME` option to load models from `data/models/<NAME>/`
- Metadata.json schema for documenting models
- Example models: `example_scalar_nut`, `example_tbnn`
- Export scripts for PyTorch and TensorFlow

### Bug Fixes

- Fixed MLP weight loading to auto-detect layers from files
- Added divergence detection in solver

## Available Turbulence Models

| Model | Type | Weights Required | Description |
|-------|------|-----------------|-------------|
| `none` | Laminar | No | No turbulence model |
| `baseline` | Algebraic | No | Mixing length with van Driest damping |
| `gep` | Algebraic | No | GEP-style algebraic corrections |
| `nn_mlp` | Neural Network | Yes | Scalar eddy viscosity prediction |
| `nn_tbnn` | Neural Network | Yes | TBNN anisotropy prediction |

## Usage Examples

```bash
# Laminar with adaptive dt
./channel --Nx 32 --Ny 64 --nu 0.01 --adaptive_dt --max_iter 20000

# Baseline turbulence
./channel --Nx 64 --Ny 128 --nu 0.001 --model baseline --adaptive_dt

# GEP model
./channel --Nx 64 --Ny 128 --nu 0.001 --model gep --adaptive_dt

# NN model with preset
./channel --model nn_mlp --nn_preset example_scalar_nut --adaptive_dt
```

## Public NN Weights Status

After extensive search, **no publicly downloadable pre-trained NN turbulence model weights** were found. Available resources:
- DNS/LES data for training (Johns Hopkins, KTH, etc.)
- Code frameworks requiring you to train models
- OpenFOAM implementations using DNS data directly

The GEP algebraic model provides a data-driven alternative without requiring trained weights.

## Files Changed

### New Files
- `include/turbulence_gep.hpp` - GEP model header
- `src/turbulence_gep.cpp` - GEP model implementation
- `scripts/compare_models.py` - Model comparison script

### Modified Files
- `include/solver.hpp` - Added adaptive dt and VTK methods
- `src/solver.cpp` - Implemented adaptive dt and VTK output
- `include/config.hpp` - Added GEP enum
- `src/config.cpp` - Added CLI options for adaptive dt, CFL, GEP
- `src/turbulence_baseline.cpp` - Added GEP to factory
- `CMakeLists.txt` - Added GEP source
- `app/main_channel.cpp` - Added VTK output call

## Performance Notes

With adaptive time stepping, the solver automatically selects appropriate dt:
- High viscosity (nu=0.1): dt ~ 0.005
- Low viscosity (nu=0.01): dt ~ 0.0003
- Turbulent flows: dt adapts to local nu_t

This eliminates the manual parameter tuning previously required.

