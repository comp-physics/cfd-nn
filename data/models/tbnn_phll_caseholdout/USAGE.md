# TBNN Periodic Hills Model - Usage Guide

## Model Overview

This is a Tensor Basis Neural Network (TBNN) trained on the McConkey et al. (2021) periodic hills dataset using case-holdout validation.

**Architecture:** 5 invariants → 64 → 64 → 64 → 4 basis coefficients (2D)

## Training Details

### Dataset Split (Case-Holdout)
- **Training cases:** case_0p5, case_0p8, case_1p5
- **Validation case:** case_1p0
- **Test case:** case_1p2

This split ensures the model generalizes across different flow conditions rather than just different spatial points.

### Training Configuration
- **Framework:** PyTorch
- **Optimizer:** Adam with ReduceLROnPlateau scheduler
- **Learning rate:** 1e-3
- **Batch size:** 8192
- **Epochs:** 200
- **Total parameters:** 8,964

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
./channel --model nn_tbnn --nn_preset tbnn_phll_caseholdout
```

### With Custom Parameters
```bash
./channel --model nn_tbnn --nn_preset tbnn_phll_caseholdout \
  --Nx 128 --Ny 256 \
  --Re 2800 \
  --max_iter 50000 \
  --output output/tbnn_phll_run
```

### Using a Config File
Create `config.cfg`:
```
turb_model = nn_tbnn
nn_preset = tbnn_phll_caseholdout
Nx = 128
Ny = 256
Re = 2800
max_iter = 50000
output_dir = output/tbnn_phll_run
```

Then run:
```bash
./channel --config config.cfg
```

## Expected Performance

The model was trained on periodic hills geometry but can be applied to channel flow and other 2D geometries. Performance will be best on flows similar to the training data:
- Separated flows
- Moderate Reynolds numbers
- 2D geometries

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
- Training script: `scripts/run_phll_train.sbatch`
- Dataset generator: `generate_phll_npz.py`

To retrain or modify, see the training workspace for scripts and data preparation tools.

