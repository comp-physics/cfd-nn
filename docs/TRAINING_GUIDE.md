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

# Train TBNN on periodic hills
python train_tbnn_mcconkey.py \
    --data_dir ../mcconkey_data \
    --case periodic_hills \
    --output ../data/models/tbnn_periodic_hills \
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
    --case periodic_hills \
    --output ../data/models/tbnn_custom \
    --epochs 200 \
    --batch_size 512 \
    --lr 5e-4 \
    --hidden 64 64 64 \
    --device cuda
```

**Parameters**:
- `--data_dir`: Path to McConkey dataset
- `--case`: Flow case (periodic_hills, channel, square_duct)
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

## Step 4: Train an MLP Model (Simpler Alternative)

If you want a faster, simpler model:

```bash
python train_mlp_mcconkey.py \
    --data_dir ../mcconkey_data \
    --case periodic_hills \
    --output ../data/models/mlp_periodic_hills \
    --epochs 100
```

The MLP directly predicts scalar eddy viscosity:

```
Input:  6 features (S, Ω, y/δ, k, ω, |u|)
        ↓
Hidden: 32 --> 32 (tanh)
        ↓
Output: 1 (nu_t, with ReLU to ensure >= 0)
```

**Advantages of MLP**:
- [OK] Faster training (smaller network)
- [OK] Faster inference in CFD solver
- [OK] Easier to interpret

**Disadvantages**:
- [FAIL] Less physically consistent (no frame invariance)
- [FAIL] Cannot predict anisotropic Reynolds stresses

## Step 5: Use Trained Model in Solver

After training, the model is automatically exported to C++ format:

```
data/models/tbnn_periodic_hills/
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
./periodic_hills --model nn_tbnn --nn_preset tbnn_periodic_hills

# Use MLP model
./periodic_hills --model nn_mlp --nn_preset mlp_periodic_hills

# Compare against baseline
./periodic_hills --model baseline
```

### Full Comparison

```bash
# Run all models and compare
cd scripts/
python compare_models.py --case periodic_hills
```

This will generate comparison plots of velocity profiles, Reynolds stresses, etc.

## Step 6: Validate Results

### A Priori Testing

Check if the model predictions match DNS data:

```python
import numpy as np
import torch

# Load test data
test_data = np.load('mcconkey_data/periodic_hills/test/data.npz')
invariants = test_data['invariants']
b_true = test_data['anisotropy']

# Load trained model
model = torch.load('models/tbnn_periodic_hills/model.pth')
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
4. Compare separation/reattachment points (for periodic hills)

## Expected Results

Based on published literature, you should see:

### TBNN (Ling et al. 2016)

- **Channel flow**: ~10-20% improvement over baseline mixing length
- **Periodic hills**: Better prediction of separation bubble and reattachment
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
4. **Different case**: Some flows (channel) are easier than others (periodic hills)
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
3. **Test on multiple cases** - Train on channel, test on periodic hills (generalization)
4. **Publish results** - You now have a complete ML turbulence modeling framework!

## Questions?

- Check `data/models/README.md` for model zoo documentation
- See `scripts/export_pytorch.py` for manual weight export
- Read source code in `include/turbulence_nn_tbnn.hpp` for implementation details


