# Neural Network Turbulence Model Zoo

This directory contains pre-trained neural network models from published research that can be used with the RANS solver.

## Directory Structure

Each model has its own subdirectory:
```
data/models/
+-- ling_tbnn_2016/          # TBNN for anisotropy (Ling et al. JFM 2016)
+-- example_scalar_nut/      # Example scalar eddy viscosity model
+-- ...                      # Additional published models
```

Each model directory should contain:
- `layer*.txt` - Neural network weight files
- `input_means.txt` - Feature normalization means
- `input_stds.txt` - Feature normalization standard deviations
- `metadata.json` - Model metadata and documentation

## Using a Model

To use a preset model with the solver:

```bash
# Use TBNN model
./channel --model nn_tbnn --nn_preset ling_tbnn_2016

# Use scalar eddy viscosity model
./channel --model nn_mlp --nn_preset example_scalar_nut
```

## Published Models

### Target Models to Implement

The following published models are good candidates for reproduction:

#### 1. Ling et al. (2016) - TBNN
**Reference:** "Reynolds averaged turbulence modelling using deep neural networks with embedded invariance"  
**DOI:** 10.1017/jfm.2016.615  
**Type:** TBNN (anisotropy tensor)  
**Architecture:** 5 invariants -> 64 -> 64 -> 64 -> 10 coefficients (3D) or 4 (2D)  
**Availability:** Check author GitHub or contact authors  

**Status:** Not yet implemented - need weights from authors

#### 2. Wu et al. (2018) - Physics-Informed ML
**Reference:** "Reynolds-averaged Navier--Stokes equations with explicit data-driven Reynolds stress closure"  
**DOI:** 10.1016/j.jcp.2018.03.037  
**Type:** Scalar eddy viscosity  
**Test cases:** Channel flow, periodic hills  
**Availability:** Contact authors  

**Status:** Not yet implemented

#### 3. Weatheritt & Sandberg (2016) - Gene Expression Programming
**Reference:** "A novel evolutionary algorithm applied to algebraic modifications of the RANS stress--strain relationship"  
**DOI:** 10.1017/jfm.2016.608  
**Type:** Algebraic (no NN - just formulas!)  
**Advantage:** No weights needed, interpretable, very fast  

**Status:** Not yet implemented - would be good first target

#### 4. Zhou et al. (2019) - Subgrid-scale model
**Reference:** "Subgrid-scale model for large-eddy simulation of isotropic turbulent flows using an artificial neural network"  
**DOI:** 10.1016/j.camwa.2018.12.027  
**Type:** SGS stress tensor  

**Status:** Not yet implemented

#### 5. Brener et al. (2021) - Deep learning for RANS
**Reference:** "RANS-PINN based simulation optimization of urban designs"  
**Type:** Scalar eddy viscosity  
**Test cases:** Urban flow  

**Status:** Not yet implemented

## Adding a New Model

### From PyTorch

If you have a trained PyTorch model:

```bash
python scripts/export_pytorch.py model.pth \
    --output data/models/my_model \
    --means means.npy \
    --stds stds.npy
```

### From TensorFlow/Keras

```bash
python scripts/export_tensorflow.py model.h5 \
    --output data/models/my_model \
    --means means.npy \
    --stds stds.npy
```

### Metadata File

Create `metadata.json` in the model directory:

```json
{
  "name": "model_name",
  "type": "nn_mlp" or "nn_tbnn",
  "architecture": {
    "layers": [6, 32, 32, 1],
    "activations": ["tanh", "tanh", "linear"]
  },
  "features": {
    "type": "scalar_nut_v1" or "tbnn_ling2016",
    "inputs": ["S_mag", "Omega_mag", "y_plus", "k", "omega", "wall_dist"],
    "normalization": "z-score"
  },
  "reference": {
    "title": "Paper title",
    "authors": "Author et al.",
    "journal": "Journal name",
    "year": 2020,
    "doi": "10.xxxx/xxxxx"
  },
  "test_cases": [
    {
      "case": "channel_flow",
      "Re_tau": 180,
      "expected_error": "< 5%"
    }
  ],
  "notes": "Any additional information about the model"
}
```

## Validation Protocol

When adding a model, validate it against published results:

1. **Reproduce test case** - Use same geometry and Re number
2. **Compare mean profiles** - Velocity, Reynolds stresses
3. **Document differences** - Any deviations from paper
4. **Performance** - Measure inference time
5. **Add to validation suite** - Create test in `tests/`

## Feature Definitions

Different models may use different feature sets. Current supported:

### Scalar Eddy Viscosity (scalar_nut_v1)
6 features:
1. Strain rate magnitude: |S|
2. Rotation rate magnitude: |Omega|
3. Normalized wall distance: y/delta
4. Turbulent kinetic energy: k
5. Specific dissipation: omega
6. Wall distance: d_wall

### TBNN Ling 2016 (tbnn_ling2016)
5 invariants derived from normalized S and Omega tensors

### Future Feature Sets
Additional feature definitions can be added as needed for specific models.

## Resources

### Data Sources for Training (if you need them later)
- **Johns Hopkins Turbulence Database** - http://turbulence.pha.jhu.edu/
- **RANS Database** - https://turbmodels.larc.nasa.gov/
- **Channel Flow DNS** - https://www.flow.kth.se/~pschlatt/DATA/
- **Periodic Hills** - http://www.kbwiki.ercoftac.org/

### Relevant Papers with Code
- TensorFlowFoam - https://arxiv.org/abs/1910.10878
- NNPred Library - https://arxiv.org/abs/2209.12339
- RoseNNa - https://arxiv.org/abs/2307.16322

## Contributing

When adding a model:
1. Create new directory under `data/models/<model_name>/`
2. Add all weight files and metadata.json
3. Update this README with model details
4. Add validation test case
5. Document any differences from published results

