# Training Guide: Neural Network Turbulence Models

This guide explains how to train neural network turbulence models on the McConkey et al. (2021) dataset and integrate them into the nn-cfd solver.

## Overview

The nn-cfd project supports two types of neural network turbulence closures:

1. **TBNN (Tensor Basis Neural Network)**: Predicts Reynolds stress anisotropy using invariant features
2. **MLP (Multi-Layer Perceptron)**: Directly predicts scalar eddy viscosity

## Dataset: McConkey et al. (2021)

### About the Dataset

McConkey et al. compiled a comprehensive dataset of RANS and DNS/LES data specifically for training data-driven turbulence models:

- **Reference**: "A curated dataset for data-driven turbulence modelling." *Scientific Data* 8, 255 (2021)
- **DOI**: [10.1038/s41597-021-01034-2](https://doi.org/10.1038/s41597-021-01034-2)
- **Kaggle**: https://www.kaggle.com/datasets/ryleymcconkey/ml-turbulence-dataset

### Flow Cases Included

1. **Channel flow** - Fully developed turbulent channel at various Re
2. **Periodic hills** - Separated flow over wavy wall geometry
3. **Square duct** - Secondary flows in non-circular cross-sections
4. **Converging-diverging channel** - Flow with adverse pressure gradient

### What's Provided

The dataset includes **exactly the features needed for TBNN and MLP models**:

- [OK] **5 scalar invariants** (λ₁, λ₂, λ₃, λ₄, λ₅) from strain and rotation tensors
- [OK] **10 tensor basis functions** (Pope's basis: T⁽¹⁾, T⁽²⁾, ..., T⁽¹⁰⁾)
- [OK] **Anisotropy tensor** b_ij from DNS/LES (ground truth labels)
- [OK] **RANS flow fields** (velocity, pressure, k, ω)
- [OK] **Wall distance** and geometric information

This means you can train models **without preprocessing** - all features are ready to use.

## Step 1: Download the Dataset

### Option A: Kaggle (Recommended)

```bash
# Install Kaggle CLI
pip install kaggle

# Setup API credentials (see https://www.kaggle.com/docs/api)
# Download dataset
kaggle datasets download -d ryleymcconkey/ml-turbulence-dataset

# Extract
unzip ml-turbulence-dataset.zip -d mcconkey_data/
```

### Option B: Manual Download

1. Visit https://www.kaggle.com/datasets/ryleymcconkey/ml-turbulence-dataset
2. Click "Download" (requires Kaggle account)
3. Extract to `mcconkey_data/` directory

### Expected Directory Structure

```
mcconkey_data/
├── periodic_hills/
│   ├── train/
│   │   └── data.npz
│   ├── val/
│   │   └── data.npz
│   └── test/
│       └── data.npz
├── channel/
│   └── ...
└── README.md
```

## Step 2: Install Dependencies

```bash
pip install torch numpy pandas scikit-learn matplotlib
```

**Note**: Training works on CPU (no GPU required). A typical training run on 10k samples takes 5-15 minutes on a laptop.

## Step 3: Train a TBNN Model

### Quick Start

```bash
cd scripts/

# Train TBNN on channel flow
python train_tbnn_mcconkey.py \
    --data_dir ../mcconkey_data \
    --case channel \
    --output ../data/models/tbnn_channel \
    --epochs 100

# Train on channel flow
python train_tbnn_mcconkey.py \
    --data_dir ../mcconkey_data \
    --case channel \
    --output ../data/models/tbnn_channel \
    --epochs 100
```

### Advanced Options

```bash
python train_tbnn_mcconkey.py \
    --data_dir ../mcconkey_data \
    --case channel \
    --output ../data/models/tbnn_custom \
    --epochs 200 \
    --batch_size 512 \
    --lr 5e-4 \
    --hidden 64 64 64 \
    --device cuda
```

**Parameters**:
- `--data_dir`: Path to McConkey dataset
- `--case`: Flow case (channel, square_duct)
- `--output`: Where to save trained model
- `--epochs`: Training epochs (default 100)
- `--batch_size`: Batch size (default 256)
- `--lr`: Learning rate (default 1e-3)
- `--hidden`: Hidden layer sizes (default [64, 64, 64])
- `--device`: cpu, cuda, mps, or auto

### What Gets Trained

The TBNN architecture follows **Ling et al. (2016)**:

```
Input:  5 invariants (λ₁, λ₂, η₁, η₂, y/δ)
        ↓
Hidden: 64 --> 64 --> 64 (tanh activation)
        ↓
Output: 4 coefficients (G₁, G₂, G₃, G₄) for 2D tensor basis
```

The model learns to predict:

**b_ij = Σ_n G_n(invariants) x T⁽ⁿ⁾_ij(S, Ω)**

where b_ij is the Reynolds stress anisotropy tensor.

## Step 4: Train an MLP Model (Faster Alternative)

### Overview

The MLP (Multi-Layer Perceptron) model is a simpler, faster alternative to TBNN that directly predicts scalar eddy viscosity. While it lacks the frame invariance of TBNN, it offers significant speed advantages and is ideal for GPU acceleration.

### Training Methodology

**We follow the Ling et al. (2016) training protocol**, adapted for scalar eddy viscosity prediction:

1. **Case-holdout validation** - Train on subset of flow cases, validate on held-out cases
2. **Z-score normalization** - All features normalized to mean=0, std=1
3. **Adam optimizer** with ReduceLROnPlateau scheduler
4. **Early stopping** based on validation loss
5. **Batch training** for efficiency

### Architecture

```
Input:  6 features (|S|, |Ω|, y/δ, k, ω, |u|)
        ↓
Hidden: 32 neurons (tanh activation)
        ↓
Hidden: 32 neurons (tanh activation)
        ↓
Output: 1 value (ν_t, with ReLU to ensure ≥ 0)
```

**Total parameters:** 1,313 (vs 8,964 for TBNN)

### Training Command

**For channel flow:**
```bash
python scripts/train_mlp_mcconkey.py \
    --data_dir /path/to/mcconkey_data_processed \
    --case channel \
    --output data/models/mlp_channel_caseholdout \
    --epochs 500 \
    --batch_size 1024 \
    --lr 1e-3 \
    --hidden 32 32 \
    --device cuda
```

**For square duct:**
```bash
python scripts/train_mlp_mcconkey.py \
    --data_dir /path/to/mcconkey_data_processed \
    --case square_duct \
    --output data/models/mlp_phll_caseholdout \
    --epochs 500 \
    --batch_size 2048 \
    --lr 1e-3 \
    --hidden 32 32 \
    --device cuda
```

### Training on GPU Cluster (SLURM)

For faster training on GPU nodes, use the provided SLURM scripts:

```bash
# Edit train_mlp_gpu.slurm to set correct paths
sbatch train_mlp_gpu.slurm
```

**Typical training time:** 2-3 minutes on Tesla V100 GPU

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Hidden layers | [32, 32] | Balance between capacity and speed |
| Activation | Tanh | Smooth, bounded, works well for normalized inputs |
| Output activation | ReLU | Guarantees non-negative ν_t |
| Learning rate | 1e-3 | Standard for Adam optimizer |
| Batch size | 1024-2048 | Depends on dataset size |
| Epochs | 500 | With early stopping |
| Optimizer | Adam | Adaptive learning rate |
| Scheduler | ReduceLROnPlateau | Reduces LR when validation loss plateaus |

### Input Features

The MLP uses 6 scalar features:

1. **|S|** - Strain rate magnitude: √(2S_ij S_ij)
2. **|Ω|** - Rotation rate magnitude: √(2Ω_ij Ω_ij)
3. **y/δ** - Normalized wall distance (y/channel_half_height)
4. **k** - Turbulent kinetic energy from RANS
5. **ω** - Specific dissipation rate from RANS
6. **|u|** - Velocity magnitude: √(u² + v²)

All features are **z-score normalized** before training:
```
x_norm = (x - mean(x)) / std(x)
```

### Output

Single scalar value: **ν_t** (turbulent eddy viscosity)

The ReLU output activation ensures ν_t ≥ 0 (physical realizability).

### Validation Strategy

**Case-holdout validation** ensures the model generalizes to unseen flow conditions:

- **Training set:** Multiple Reynolds numbers (e.g., Re_τ = 180, 395, 590)
- **Validation set:** Held-out Reynolds number (e.g., Re_τ = 550)
- **Test set:** Additional held-out case for final evaluation

This is more rigorous than random point splitting, as it tests generalization to new flow regimes.

### Training Output

After training, the following files are created in the output directory:

```
data/models/mlp_channel_caseholdout/
├── layer0_W.txt          # First hidden layer weights (6 x 32)
├── layer0_b.txt          # First hidden layer biases (32)
├── layer1_W.txt          # Second hidden layer weights (32 x 32)
├── layer1_b.txt          # Second hidden layer biases (32)
├── layer2_W.txt          # Output layer weights (32 x 1)
├── layer2_b.txt          # Output layer bias (1)
├── input_means.txt       # Feature normalization means (6)
├── input_stds.txt        # Feature normalization stds (6)
└── metadata.json         # Model metadata and training details
```

### Advantages of MLP

- ✅ **Fast training** - Smaller network (1.3K vs 9K parameters)
- ✅ **Fast inference** - ~50x faster than TBNN in CFD solver
- ✅ **GPU-friendly** - Excellent GPU acceleration
- ✅ **Simple architecture** - Easy to understand and debug
- ✅ **Guaranteed realizability** - ReLU ensures ν_t ≥ 0
- ✅ **Low memory footprint** - Suitable for embedded/real-time applications

### Disadvantages

- ❌ **No frame invariance** - Not guaranteed to be coordinate-independent
- ❌ **Scalar output only** - Cannot predict anisotropic Reynolds stresses
- ❌ **Less physically consistent** - No embedded tensor structure
- ❌ **Geometry-specific** - May need retraining for different flow types

### When to Use MLP vs TBNN

**Use MLP when:**
- Speed is critical (real-time, interactive applications)
- GPU acceleration is available
- Scalar eddy viscosity is sufficient
- Training on similar flow geometries

**Use TBNN when:**
- Physical consistency is paramount
- Need full Reynolds stress tensor
- Generalizing across diverse geometries
- Research/publication quality results required

### Validation Results

After training `mlp_channel_caseholdout`:

- **Training time:** ~2 minutes on Tesla V100
- **Total parameters:** 1,313
- **Model size:** ~5 KB (text files)
- **Inference time:** ~0.4 ms per iteration (16x32 grid)
- **GPU speedup:** 10-50x over CPU (depending on grid size)

### Example Training Session

```bash
# 1. Activate Python environment
source venv/bin/activate

# 2. Train the model
python scripts/train_mlp_mcconkey.py \
    --data_dir /storage/scratch1/6/sbryngelson3/data-repo/mcconkey_data_processed \
    --case channel \
    --output data/models/mlp_channel_caseholdout \
    --epochs 500 \
    --batch_size 1024 \
    --lr 1e-3 \
    --device cuda

# Expected output:
# Epoch 1/500: train_loss=0.0234, val_loss=0.0198, lr=1.0e-03
# Epoch 50/500: train_loss=0.0089, val_loss=0.0091, lr=1.0e-03
# ...
# Best validation loss: 0.0087 at epoch 234
# Model saved to: data/models/mlp_channel_caseholdout

# 3. Test in solver
cd build
./channel --model nn_mlp --nn_preset mlp_channel_caseholdout --max_steps 10000
```

## Step 5: Use Trained Model in Solver

After training, the model is automatically exported to C++ format:

```
data/models/tbnn_channel/
├── layer0_W.txt
├── layer0_b.txt
├── layer1_W.txt
├── layer1_b.txt
├── layer2_W.txt
├── layer2_b.txt
├── layer3_W.txt
├── layer3_b.txt
├── input_means.txt
├── input_stds.txt
└── metadata.json
```

### Run the Solver

```bash
cd build/

# Use TBNN model
./channel --model nn_tbnn --nn_preset tbnn_channel

# Use MLP model
./channel --model nn_mlp --nn_preset mlp_channel

# Compare against baseline
./channel --model baseline
```

### Full Comparison

```bash
# Run all models and compare
cd scripts/
python compare_models.py --case channel
```

This will generate comparison plots of velocity profiles, Reynolds stresses, etc.

## Step 6: Validate Results

### A Priori Testing

Check if the model predictions match DNS data:

```python
import numpy as np
import torch

# Load test data
test_data = np.load('mcconkey_data/channel/test/data.npz')
invariants = test_data['invariants']
b_true = test_data['anisotropy']

# Load trained model
model = torch.load('models/tbnn_channel/model.pth')
model.eval()

# Predict
b_pred = model(torch.FloatTensor(invariants))

# Compute error
error = np.mean((b_pred.detach().numpy() - b_true)**2)
print(f"Test MSE: {error:.6e}")
```

### A Posteriori Testing

Run the full CFD solver and compare flow fields against DNS:

1. Run solver with trained model
2. Compare velocity profiles
3. Compare Reynolds stresses
4. Compare secondary flow patterns (for duct flow)

## Expected Results

Based on published literature, you should see:

### TBNN (Ling et al. 2016)

- **Channel flow**: ~10-20% improvement over baseline mixing length
- **Duct flow**: Better prediction of secondary flow patterns
- **Training time**: 10-30 minutes on CPU for 10k samples

### MLP

- **Faster training**: 5-10 minutes
- **Similar or slightly worse accuracy** than TBNN
- **Much faster inference** in CFD solver

## Troubleshooting

### Dataset Not Found

If you don't have the McConkey dataset, the scripts will generate **synthetic dummy data** for testing. This allows you to:

- [OK] Test the training pipeline
- [OK] Verify export to C++ works
- [OK] Run the solver with a "mock" model

But results won't be physically meaningful.

**Solution**: Download the real dataset from Kaggle.

### Training Diverges

If loss increases or becomes NaN:

1. **Reduce learning rate**: Try `--lr 1e-4` instead of `1e-3`
2. **Add gradient clipping**: Edit training script to add `torch.nn.utils.clip_grad_norm_()`
3. **Check data**: Ensure features are normalized properly
4. **Reduce batch size**: Try `--batch_size 128`

### Model Performance Poor

If the trained model doesn't improve over baseline:

1. **Train longer**: Increase `--epochs 200`
2. **More data**: Use larger subset of McConkey dataset
3. **Bigger network**: Try `--hidden 128 128 128`
4. **Different case**: Try different cases in the dataset
5. **Check feature computation**: Ensure C++ features match Python training features

## Alternative: Train Without McConkey Dataset

If you cannot access the McConkey dataset, you can train on your own DNS/LES data:

### Requirements

Your data should include:

- Velocity field (u, v, w)
- Strain rate tensor S_ij
- Rotation rate tensor Ω_ij
- Reynolds stress tensor τ_ij (from DNS/LES)
- Turbulent kinetic energy k

### Create Custom Dataset

```python
# Compute invariants from your data
lambda_1 = np.trace(S @ S)
lambda_2 = np.trace(Omega @ Omega)
# ... etc

# Compute anisotropy
b_ij = tau_ij / (2*k) - 1/3 * delta_ij

# Save in format expected by training script
np.savez('my_data/train/data.npz',
         invariants=invariants,
         anisotropy=anisotropy,
         basis=basis)
```

Then run training with `--data_dir my_data`.

## References

1. **TBNN**: Ling et al., "Reynolds averaged turbulence modelling using deep neural networks with embedded invariance," *JFM* 807 (2016)
2. **Dataset**: McConkey et al., "A curated dataset for data-driven turbulence modelling," *Scientific Data* 8 (2021)
3. **Review**: Duraisamy et al., "Turbulence Modeling in the Age of Data," *Annu. Rev. Fluid Mech.* 51 (2019)

## Next Steps

After training and validating your models:

1. **Document performance** - Add results to `VALIDATION.md`
2. **Compare against GEP** - Run comparison with your existing GEP model
3. **Test on multiple cases** - Train on channel, test on duct (generalization)
4. **Publish results** - You now have a complete ML turbulence modeling framework!

## Questions?

- Check `data/models/README.md` for model zoo documentation
- See `scripts/export_pytorch.py` for manual weight export
- Read source code in `include/turbulence_nn_tbnn.hpp` for implementation details


