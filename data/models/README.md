# Neural Network Turbulence Model Zoo

This directory contains trained neural network turbulence models and example models for the RANS solver.

## Directory Structure

Each model has its own subdirectory:
```
data/models/
+-- mlp_channel_caseholdout/    # Trained MLP for channel flow (scalar eddy viscosity)
+-- mlp_phll_caseholdout/       # Trained MLP for periodic hills (scalar eddy viscosity)*
+-- tbnn_channel_caseholdout/   # Trained TBNN for channel flow (anisotropy tensor)
+-- tbnn_phll_caseholdout/      # Trained TBNN for periodic hills (anisotropy tensor)*

*Periodic hills geometry not currently in solver; models kept for future use.
```

Each model directory contains:
- `layer*.txt` - Neural network weight files (W = weights, b = biases)
- `input_means.txt` - Feature normalization means
- `input_stds.txt` - Feature normalization standard deviations
- `metadata.json` - Model metadata (architecture, training details, reference)
- `USAGE.md` (optional) - Detailed usage guide for that specific model

## Using a Model

**All NN models require explicit selection** via `--nn_preset` or `--weights/--scaling`:

```bash
# Use trained TBNN model for channel flow
./channel --model nn_tbnn --nn_preset tbnn_channel_caseholdout

# Or specify paths directly
./channel --model nn_tbnn --weights data/models/tbnn_channel_caseholdout --scaling data/models/tbnn_channel_caseholdout
```

## Available Trained Models

### mlp_channel_caseholdout
**Type:** MLP (Multi-Layer Perceptron) - Scalar Eddy Viscosity  
**Architecture:** 6 features → 32 → 32 → 1 (ν_t)  
**Training Data:** McConkey et al. (2021) channel flow dataset with case-holdout validation  

**Best For:** Fast inference, scalar turbulence modeling, channel flows, GPU acceleration

**Features:** 6 inputs for scalar eddy viscosity prediction:
1. Strain rate magnitude: |S|
2. Rotation rate magnitude: |Ω|
3. Normalized wall distance: y/δ
4. Turbulent kinetic energy: k
5. Specific dissipation: ω
6. Velocity magnitude: |u|

**Usage:**
```bash
./channel --model nn_mlp --nn_preset mlp_channel_caseholdout
```

**Training Details:**
- **Methodology:** Following Ling et al. (2016) training protocol
- **Framework:** PyTorch 2.x
- **Optimizer:** Adam with ReduceLROnPlateau scheduler
- **Learning rate:** 1e-3 (initial), reduced on plateau
- **Epochs:** 500
- **Batch size:** 1024
- **Activation functions:** Tanh (hidden layers), ReLU (output layer)
- **Normalization:** Z-score (mean=0, std=1) for all input features
- **Total parameters:** 1,313
- **Training time:** ~2 minutes on Tesla V100 GPU
- **Training date:** December 18, 2025

**Validation Strategy:**
- **Case-holdout validation:** Model trained on subset of Reynolds numbers, validated on held-out cases
- **Ensures generalization** to unseen flow conditions
- **Dataset split:** Train/val/test from different Re_τ cases

**Performance:**
- **Inference speed:** ~50x faster than TBNN (smaller network)
- **GPU acceleration:** Full GPU offload support via OpenMP target directives
- **Memory footprint:** Minimal (1.3K parameters vs 9K for TBNN)

**Advantages:**
- ✅ Fast training and inference
- ✅ Simple architecture, easy to interpret
- ✅ Excellent GPU performance
- ✅ Guarantees non-negative ν_t (ReLU output)
- ✅ Suitable for real-time applications

**Limitations:**
- ❌ No frame invariance guarantees (unlike TBNN)
- ❌ Cannot predict anisotropic Reynolds stresses
- ❌ Less physically consistent than TBNN
- ❌ May require retraining for different geometries

**Reference:** Training methodology adapted from Ling, J., Kurzawski, A., & Templeton, J. (2016). Reynolds averaged turbulence modelling using deep neural networks with embedded invariance. *Journal of Fluid Mechanics*, 807, 155-166.

**Dataset Reference:** McConkey, R., Yee, E., & Lien, F. S. (2021). A curated dataset for data-driven turbulence modelling. *Scientific Data*, 8(1), 255. DOI: [10.1038/s41597-021-01034-2](https://doi.org/10.1038/s41597-021-01034-2)

---

### mlp_phll_caseholdout
**Type:** MLP (Multi-Layer Perceptron) - Scalar Eddy Viscosity
**Architecture:** 6 features → 32 → 32 → 1 (ν_t)
**Training Data:** McConkey et al. (2021) periodic hills dataset with case-holdout validation

> **Note:** The periodic hills geometry is not currently implemented in the solver. This model is kept for:
> - Future periodic hills implementation
> - Transfer learning to similar separated flow geometries
> - Reference for training methodology on complex flows

**Features:** 6 inputs for scalar eddy viscosity prediction (same as mlp_channel)

**Usage (requires periodic hills geometry implementation):**
```bash
./channel --model nn_mlp --nn_preset mlp_phll_caseholdout
```

**Best For:** Fast inference on separated flows (when geometry is implemented)

**Training Details:**
- Framework: PyTorch
- Architecture: 6 → 32 → 32 → 1
- Activations: Tanh (hidden), ReLU (output)
- Total parameters: 1,313

**Reference:** McConkey, R., Yee, E., & Lien, F. S. (2021). A curated dataset for data-driven turbulence modelling. *Scientific Data*, 8(1), 255.

---

### tbnn_channel_caseholdout
**Type:** TBNN (Tensor Basis Neural Network)  
**Architecture:** 5 invariants → 64 → 64 → 64 → 4 coefficients (2D)  
**Training Data:** McConkey et al. (2021) channel flow dataset with case-holdout validation  

**Best For:** Turbulent channel flows, wall-bounded flows, canonical channel simulations

**Features:** 5 Ling et al. (2016) invariants from normalized strain/rotation tensors  
**Usage:**
```bash
./channel --model nn_tbnn --nn_preset tbnn_channel_caseholdout
```

**Training Details:**
- Framework: PyTorch
- Optimizer: Adam with ReduceLROnPlateau
- Learning rate: 1e-3
- Epochs: 500
- Batch size: 1024
- Best validation loss: 0.0152
- Total parameters: 8,964
- Training time: ~3 minutes on Tesla V100

**Reference:** Ling, J., Kurzawski, A., & Templeton, J. (2016). Reynolds averaged turbulence modelling using deep neural networks with embedded invariance. *Journal of Fluid Mechanics*, 807, 155-166.

---

### tbnn_phll_caseholdout
**Type:** TBNN (Tensor Basis Neural Network)
**Architecture:** 5 invariants → 64 → 64 → 64 → 4 coefficients (2D)
**Training Data:** McConkey et al. (2021) periodic hills dataset with case-holdout validation
**Training Split:**
- Train: case_0p5, case_0p8, case_1p5
- Val: case_1p0
- Test: case_1p2

**Features:** 5 Ling et al. (2016) invariants from normalized strain/rotation tensors

> **Note:** The periodic hills geometry is not currently implemented in the solver. This model is kept for:
> - Future periodic hills implementation
> - Transfer learning to similar separated flow geometries
> - Reference for training methodology on complex flows

**Usage (requires periodic hills geometry implementation):**
```bash
./channel --model nn_tbnn --nn_preset tbnn_phll_caseholdout
```

**Best For:** Separated flows, periodic hills geometry, complex recirculation zones (when geometry is implemented)

**Training Details:**
- Framework: PyTorch
- Optimizer: Adam with ReduceLROnPlateau
- Learning rate: 1e-3
- Epochs: 200
- Batch size: 8192
- Best validation loss: 0.0093
- Total parameters: 8,964

**Reference:** Ling, J., Kurzawski, A., & Templeton, J. (2016). Reynolds averaged turbulence modelling using deep neural networks with embedded invariance. *Journal of Fluid Mechanics*, 807, 155-166.

---

### Model Selection Guide

| Model | Type | Training Data | Best For | Parameters | Training Time | Inference Speed |
|-------|------|---------------|----------|------------|---------------|-----------------|
| `mlp_channel_caseholdout` | MLP | Channel flow | Fast inference, GPU acceleration, scalar ν_t | 1,313 | 2 min | Very Fast |
| `tbnn_channel_caseholdout` | TBNN | Channel flow | Physically consistent, anisotropic stresses | 8,964 | 3 min | Moderate |
| `mlp_phll_caseholdout` | MLP | Periodic hills | Separated flows (fast)* | 1,313 | 2 min | Very Fast |
| `tbnn_phll_caseholdout` | TBNN | Periodic hills | Separated flows* | 8,964 | 6 min | Moderate |

*Periodic hills geometry not currently implemented in solver; models kept for future use.

**Recommendations:**
- **For production/real-time applications:** Use `mlp_channel_caseholdout` (fastest, GPU-optimized)
- **For research/high accuracy:** Use `tbnn_channel_caseholdout` (physically consistent)
- **For separated flows:** `tbnn_phll_caseholdout` available when geometry is implemented

**MLP vs TBNN Trade-offs:**
- **MLP:** Faster (50x), simpler, GPU-friendly, but less physically consistent
- **TBNN:** Slower, more complex, but frame-invariant and can predict full Reynolds stress tensor

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

