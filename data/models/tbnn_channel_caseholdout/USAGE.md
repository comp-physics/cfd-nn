# TBNN Channel Flow Model - Usage Guide

## Model Overview

This is a Tensor Basis Neural Network (TBNN) trained on the McConkey et al. (2021) channel flow dataset using case-holdout validation.

**Architecture:** 5 invariants → 64 → 64 → 64 → 4 basis coefficients (2D)

## Training Details

### Dataset Split (Case-Holdout)
- **Training cases:** Multiple channel flow cases from McConkey dataset
- **Validation case:** Held-out case for generalization testing
- **Test case:** Independent case for final evaluation

This split ensures the model generalizes across different Reynolds numbers and flow conditions rather than just different spatial points.

### Training Configuration
- **Framework:** PyTorch
- **Optimizer:** Adam with ReduceLROnPlateau scheduler
- **Learning rate:** 1e-3
- **Batch size:** 1024
- **Epochs:** 500
- **Total parameters:** 8,964
- **Best validation loss:** 0.015233
- **Training time:** ~3 minutes on Tesla V100

### Features (Ling et al. 2016 Invariants)
The model uses 5 scalar invariants computed from normalized strain (S) and rotation (Ω) tensors:

1. **λ₁** = tr(S²) - Strain intensity
2. **λ₂** = tr(Ω²) - Rotation intensity  
3. **η₁** = tr(S³) - Strain skewness
4. **η₂** = tr(Ω²S) - Strain-rotation interaction
5. **y_norm** - Normalized wall distance

These are z-score normalized using training set statistics.

### Output
The model predicts 4 coefficients (G₁, G₂, G₃, G₄) for the 2D tensor basis:
- **T₁** = S̃ (normalized strain)
- **T₂** = S̃Ω̃ - Ω̃S̃ (commutator)
- **T₃** = S̃² - ½tr(S̃²)I (deviatoric strain squared)
- **T₄** = Ω̃² - ½tr(Ω̃²)I (deviatoric rotation squared)

The Reynolds stress anisotropy is reconstructed as:
```
b = Σᵢ Gᵢ Tᵢ
```

## Usage in Solver

### Basic Usage
```bash
cd /storage/home/hcoda1/6/sbryngelson3/cfd-nn
./channel --model nn_tbnn --nn_preset tbnn_channel_caseholdout
```

### With Custom Parameters
```bash
./channel --model nn_tbnn --nn_preset tbnn_channel_caseholdout \
  --Nx 128 --Ny 256 \
  --Re 2800 \
  --max_iter 50000 \
  --output output/tbnn_channel_run
```

### Using a Config File
Create `config.cfg`:
```
turb_model = nn_tbnn
nn_preset = tbnn_channel_caseholdout
Nx = 128
Ny = 256
Re = 2800
max_iter = 50000
output_dir = output/tbnn_channel_run
```

Then run:
```bash
./channel --config config.cfg
```

## Expected Performance

The model was trained specifically on channel flow geometry and should perform best on:
- Turbulent channel flows
- Moderate to high Reynolds numbers
- 2D planar geometries
- Wall-bounded flows

This model is optimized for channel flow and may perform better than the periodic hills variant for canonical channel simulations.

## Comparison with Periodic Hills Model

| Model | Training Data | Best For | Val Loss |
|-------|---------------|----------|----------|
| `tbnn_channel_caseholdout` | Channel flow | Channel simulations | 0.0152 |
| `tbnn_phll_caseholdout` | Periodic hills | Separated flows | 0.0093 |

Choose the model that best matches your target geometry.

## Files in This Directory

- `layer0_W.txt`, `layer0_b.txt` - Input layer (5 → 64)
- `layer1_W.txt`, `layer1_b.txt` - Hidden layer 1 (64 → 64)
- `layer2_W.txt`, `layer2_b.txt` - Hidden layer 2 (64 → 64)
- `layer3_W.txt`, `layer3_b.txt` - Output layer (64 → 4)
- `input_means.txt` - Feature normalization means (5 values)
- `input_stds.txt` - Feature normalization stds (5 values)
- `metadata.json` - Model metadata and references

## References

**TBNN Architecture:**
Ling, J., Kurzawski, A., & Templeton, J. (2016). Reynolds averaged turbulence modelling using deep neural networks with embedded invariance. *Journal of Fluid Mechanics*, 807, 155-166.  
DOI: 10.1017/jfm.2016.615

**Training Dataset:**
McConkey, R., Yee, E., & Lien, F.-S. (2021). A curated dataset for data-driven turbulence modelling. *Scientific Data*, 8, 255.  
DOI: 10.1038/s41597-021-01034-2  
URL: https://www.kaggle.com/datasets/ryleymcconkey/ml-turbulence-dataset

## Training Location

This model was trained in:
- Workspace: `/storage/home/hcoda1/6/sbryngelson3/scratch/nn-cfd`
- Training script: `scripts/run_channel_train.sbatch`
- Job ID: 2998847
- Dataset generator: `generate_channel_npz.py`

To retrain or modify, see the training workspace for scripts and data preparation tools.

