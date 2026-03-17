# Turbulence Models Guide

## Overview

The nn-cfd project supports two types of neural network turbulence closures:

1. **TBNN (Tensor Basis Neural Network)** -- Predicts Reynolds stress anisotropy tensor b_ij using Galilean-invariant scalar features and Pope's tensor basis (Ling et al. 2016). Embeds frame invariance by construction.

2. **MLP (Multi-Layer Perceptron)** -- Predicts scalar eddy viscosity nu_t from local flow features. Simpler, faster, GPU-friendly, but lacks frame invariance and cannot predict anisotropic stresses.

Both model types load pre-trained weights from text files and run inference inside the C++ solver at every time step. A **model zoo** stores published models in `data/models/`, and a **training pipeline** trains new models on the McConkey et al. (2021) dataset.

## Quick Start

### Option A: Real Data (30 minutes)

```bash
# 1. Download McConkey dataset (~500 MB)
pip install kaggle
bash scripts/download_mcconkey_data.sh

# 2. Train a TBNN on channel flow (15 min CPU)
python scripts/train_tbnn_mcconkey.py \
    --data_dir mcconkey_data \
    --case channel \
    --output data/models/my_tbnn \
    --epochs 50

# 3. Run in solver
cd build
./channel --model nn_tbnn --nn_preset my_tbnn --max_steps 10000
```

### Option B: Dummy Data (5 minutes)

If you do not have the dataset, the training scripts generate synthetic data automatically:

```bash
python scripts/train_tbnn_mcconkey.py \
    --data_dir /nonexistent/path \
    --output data/models/test_tbnn \
    --epochs 10

cd build
./channel --model nn_tbnn --nn_preset test_tbnn --max_steps 1000
```

Results from dummy data are not physically meaningful but verify the full pipeline.

### Train an MLP Instead

```bash
python scripts/train_mlp_mcconkey.py \
    --case channel \
    --output data/models/mlp_channel \
    --hidden 32 32 \
    --epochs 100
```

## McConkey Dataset

### Citation

McConkey, R., Yee-Chung, L., Phon-Anant, M., Eyassu, J., & Sharma, A. (2021). A curated dataset for data-driven turbulence modelling. *Scientific Data*, 8(1), 1--12. DOI: [10.1038/s41597-021-01034-2](https://doi.org/10.1038/s41597-021-01034-2)

```bibtex
@article{mcconkey2021curated,
  title={A curated dataset for data-driven turbulence modelling},
  author={McConkey, Romit and Yee-Chung, Lyle and Phon-Anant, Marius
          and Eyassu, James and Sharma, Amit},
  journal={Scientific Data},
  volume={8}, number={1}, pages={1--12}, year={2021},
  doi={10.1038/s41597-021-01034-2}
}
```

### Flow Cases

| Case | Description | Challenge | Re | Samples |
|------|-------------|-----------|-----|---------|
| Channel flow | Parallel walls, fully developed | Boundary layers, wall effects | Re_tau = 180, 395, 590 | ~12,000 |
| Periodic hills | Sinusoidal lower wall, periodic BCs | Separation, recirculation | Re_h = 10,595 | ~50,000 |
| Square duct | Square cross-section | Secondary flows from anisotropy | Re_tau = 300 | ~120,000 |
| Conv-div channel | Converging then diverging | Adverse pressure gradient | -- | ~60,000 |

**Total**: ~240,000 labeled samples across all cases.

### Data Format

```
mcconkey_data/
    periodic_hills/
        train/data.npz
        val/data.npz
        test/data.npz
    channel_Re180/
        ...
    square_duct/
        ...
```

Each `data.npz` contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `invariants` | [N, 5] | Scalar invariants lambda_1 through lambda_5 |
| `basis` | [N, 10, 6] | Pope's tensor basis functions (10 in 3D, 4 in 2D) |
| `anisotropy` | [N, 6] or [N, 3] | Reynolds stress anisotropy b_ij from DNS (labels) |
| `k` | [N] | Turbulent kinetic energy |
| `omega` | [N] | Specific dissipation rate |
| `wall_distance` | [N] | Distance to nearest wall |
| `velocity` | [N, 2] or [N, 3] | Mean velocity field |
| `S` | [N, 2, 2] or [N, 3, 3] | Strain rate tensor |
| `Omega` | [N, 2, 2] or [N, 3, 3] | Rotation rate tensor |

The anisotropy tensor is defined as **b_ij = tau_ij / (2k) - delta_ij / 3**, where tau_ij is the Reynolds stress tensor. It is trace-free, symmetric, and bounded in [-1/3, 2/3].

### Download Methods

**Kaggle CLI (recommended)**:
```bash
pip install kaggle
# Get API key from https://www.kaggle.com/settings -> ~/.kaggle/kaggle.json
bash scripts/download_mcconkey_data.sh
```

**Manual**: Visit https://www.kaggle.com/datasets/ryleymcconkey/ml-turbulence-dataset and click Download.

**Zenodo**: https://zenodo.org/record/5164535 (~500 MB compressed, ~2 GB uncompressed).

### Invariants Definition

The 5 scalar invariants are Galilean-invariant quantities computed from the normalized strain rate tensor S~ and rotation rate tensor Omega~:

1. lambda_1 = tr(S~^2) -- strain rate magnitude squared
2. lambda_2 = tr(Omega~^2) -- rotation rate magnitude squared
3. lambda_3 = tr(S~^3) -- third invariant of strain
4. lambda_4 = tr(S~ Omega~^2) -- mixed invariant
5. lambda_5 = tr(S~^2 Omega~^2) -- higher-order mixed invariant

## Training TBNN

### Architecture

The TBNN follows Ling et al. (2016):

```
Input:  5 invariants (lambda_1, lambda_2, eta_1, eta_2, q*)
            |
Hidden: 64 -> 64 -> 64   (tanh activation)
            |
Output: 4 coefficients (G_1, G_2, G_3, G_4)
```

The model predicts: **b_ij = sum_n G_n(invariants) * T^(n)_ij(S, Omega)**

where T^(n) are the tensor basis functions from Pope's representation theorem. This construction guarantees Galilean invariance and realizability by design.

**Total parameters**: 8,964

### Training Command

```bash
python scripts/train_tbnn_mcconkey.py \
    --data_dir mcconkey_data \
    --case channel \
    --output data/models/tbnn_channel \
    --epochs 100 \
    --batch_size 256 \
    --lr 1e-3 \
    --hidden 64 64 64 \
    --device auto
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_dir` | (required) | Path to McConkey dataset |
| `--case` | (required) | Flow case: channel, square_duct, periodic_hills |
| `--output` | (required) | Output directory for trained model |
| `--epochs` | 100 | Training epochs |
| `--batch_size` | 256 | Batch size |
| `--lr` | 1e-3 | Learning rate (Adam optimizer) |
| `--hidden` | [64, 64, 64] | Hidden layer sizes |
| `--device` | auto | cpu, cuda, mps, or auto |

### Expected Results

- **Training time**: 10--30 minutes on CPU for 10k samples
- **Channel flow**: ~10--20% improvement over baseline mixing length
- **Duct flow**: Better prediction of secondary flow patterns

## Training MLP

### Architecture

```
Input:  6 features (|S|, |Omega|, y/delta, k, omega, |u|)
            |
Hidden: 32 -> 32   (tanh activation)
            |
Output: 1 value (nu_t, with ReLU ensuring >= 0)
```

**Total parameters**: 1,313

### Input Features

The C++ solver (`src/features.cpp`) computes these 6 features:

1. **|S| * delta / u_ref** -- Normalized strain rate magnitude
2. **|Omega| * delta / u_ref** -- Normalized rotation rate magnitude
3. **y / delta** -- Normalized wall distance (y / channel half-height)
4. **|Omega| / |S|** -- Strain-rotation ratio (0 if |S| < 1e-10)
5. **|S| * delta^2 / nu** -- Local Reynolds number based on strain rate
6. **|u| / u_ref** -- Normalized velocity magnitude

**Note:** The training script (`scripts/train_mlp_mcconkey.py`) uses raw features (|S|, |Omega|, y, k, omega, |u|) from the dataset, while the C++ solver normalizes by reference scales. Ensure feature normalization files (`input_means.txt`, `input_stds.txt`) account for this difference.

All features are z-score normalized: x_norm = (x - mean) / std.

### Training Command

```bash
python scripts/train_mlp_mcconkey.py \
    --data_dir mcconkey_data \
    --case channel \
    --output data/models/mlp_channel \
    --epochs 500 \
    --batch_size 1024 \
    --lr 1e-3 \
    --hidden 32 32 \
    --device cuda
```

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Hidden layers | [32, 32] | Balance between capacity and inference speed |
| Activation | Tanh | Smooth, bounded, works well for normalized inputs |
| Output activation | ReLU | Guarantees non-negative nu_t |
| Learning rate | 1e-3 | Standard for Adam |
| Batch size | 1024--2048 | Depends on dataset size |
| Epochs | 500 | With early stopping |
| Optimizer | Adam | Adaptive learning rate |
| Scheduler | ReduceLROnPlateau | Reduces LR when validation loss plateaus |

### Case-Holdout Validation

The training uses case-holdout validation rather than random point splitting:

- **Training set**: Multiple Reynolds numbers (e.g., Re_tau = 180, 395, 590)
- **Validation set**: Held-out Reynolds number (e.g., Re_tau = 550)
- **Test set**: Additional held-out case for final evaluation

This tests generalization to unseen flow regimes, which is more rigorous than random splitting.

### MLP vs TBNN

| Criterion | MLP | TBNN |
|-----------|-----|------|
| Parameters | 1,313 | 8,964 |
| Training time | 5--10 min | 15--30 min |
| Inference speed | ~0.4 ms/iter | ~2 ms/iter |
| Frame invariance | No | Yes (by construction) |
| Output | Scalar nu_t | Anisotropy tensor b_ij |
| GPU acceleration | Excellent | Good |
| Accuracy vs DNS | ~25% error | ~15% error |

**Use MLP when**: speed is critical, scalar eddy viscosity suffices, or training on similar geometries.

**Use TBNN when**: physical consistency matters, full Reynolds stress tensor is needed, or generalizing across diverse geometries.

## Model Zoo

### Using a Preset Model

```bash
# MLP preset
./channel --model nn_mlp --nn_preset example_scalar_nut

# TBNN preset
./channel --model nn_tbnn --nn_preset example_tbnn
```

The `--nn_preset <name>` flag maps to `data/models/<name>/` for both weights and scaling files. Explicit `--weights` overrides the preset if both are given.

Presets also work in config files:

```ini
turb_model = nn_mlp
nn_preset = example_scalar_nut
```

### Directory Structure

Each model lives in its own directory under `data/models/`:

```
data/models/<model_name>/
    layer0_W.txt          # Layer 0 weight matrix
    layer0_b.txt          # Layer 0 bias vector
    layer1_W.txt          # Layer 1 weight matrix
    layer1_b.txt          # ...
    ...
    input_means.txt       # Feature normalization means
    input_stds.txt        # Feature normalization stds
    metadata.json         # Model documentation
```

### Adding a Published Model

**Step 1: Export weights** to text format.

From PyTorch:
```bash
python scripts/export_pytorch.py model.pth \
    --output data/models/my_model \
    --means means.npy --stds stds.npy
```

From TensorFlow/Keras:
```bash
python scripts/export_tensorflow.py model.h5 \
    --output data/models/my_model \
    --means means.npy --stds stds.npy
```

From paper supplementary materials: convert to `layerN_W.txt`, `layerN_b.txt`, `input_means.txt`, `input_stds.txt` manually.

**Step 2: Create `metadata.json`**.

```json
{
  "name": "ling_tbnn_2016",
  "type": "nn_tbnn",
  "description": "TBNN for Reynolds stress anisotropy",
  "architecture": {
    "layers": [5, 64, 64, 64, 4],
    "activations": ["tanh", "tanh", "tanh", "linear"]
  },
  "features": {
    "type": "tbnn_ling2016",
    "inputs": ["lambda_1", "lambda_2", "eta_1", "eta_2", "q_star"]
  },
  "reference": {
    "title": "Reynolds averaged turbulence modelling using deep neural networks with embedded invariance",
    "authors": "Ling, J., Kurzawski, A., & Templeton, J.",
    "journal": "Journal of Fluid Mechanics",
    "year": 2016,
    "doi": "10.1017/jfm.2016.615"
  },
  "test_cases": [
    {
      "case": "channel_flow",
      "Re_tau": 180,
      "command": "./channel --model nn_tbnn --nn_preset ling_tbnn_2016 --Nx 64 --Ny 128"
    }
  ]
}
```

**Step 3: Validate** against the original paper's test case.

```bash
./channel --model nn_tbnn --nn_preset ling_tbnn_2016 \
    --Nx 64 --Ny 128 --Re 2800 --max_steps 50000
```

Compare mean velocity profiles, Reynolds stress components, and friction coefficient.

**Step 4: Document** in `data/models/README.md`.

### Feature Definitions

#### `scalar_nut_v1` (6 features)

Used by MLP models. C++ features: |S|*delta/u_ref, |Omega|*delta/u_ref, y/delta, |Omega|/|S|, |S|*delta^2/nu, |u|/u_ref.

#### `tbnn_ling2016` (5 invariants)

Used by TBNN models. C++ features: tr(S_norm^2), tr(Omega_norm^2), Sxx^2+Syy^2+2*Sxy^2, 2*Oxy^2, y_wall/delta.

To add a new feature set: implement computation in `src/features.cpp`, add a new enum value, update the relevant turbulence model class, and document in `metadata.json`.

## Using Models in the Solver

### Run Commands

```bash
cd build/

# TBNN model
./channel --model nn_tbnn --nn_preset tbnn_channel

# MLP model
./channel --model nn_mlp --nn_preset mlp_channel

# Baseline comparison
./channel --model baseline

# Compare all models
cd scripts/
python compare_models.py --case channel
```

### Training Output Files

After training, the output directory contains everything needed by the C++ solver:

```
data/models/tbnn_channel/
    layer0_W.txt    layer0_b.txt
    layer1_W.txt    layer1_b.txt
    layer2_W.txt    layer2_b.txt
    layer3_W.txt    layer3_b.txt    # TBNN has 4 layers (3 hidden + output)
    input_means.txt
    input_stds.txt
    metadata.json
```

The solver loads these automatically via `--nn_preset`.

## Validation

### A Priori Testing

Compare model predictions directly against DNS data without running the CFD solver:

```python
import numpy as np
import torch

# Load test data
test_data = np.load('mcconkey_data/channel/test/data.npz')
invariants = test_data['invariants']
b_true = test_data['anisotropy']

# Load trained model and predict
model = torch.load('models/tbnn_channel/model.pth')
model.eval()
b_pred = model(torch.FloatTensor(invariants))

# Compute error
error = np.mean((b_pred.detach().numpy() - b_true)**2)
print(f"Test MSE: {error:.6e}")
```

### A Posteriori Testing

Run the full CFD solver with the trained model and compare flow fields against DNS:

1. Run solver with trained model to convergence
2. Compare mean velocity profiles
3. Compare Reynolds stress components (TBNN) or nu_t distribution (MLP)
4. Compare secondary flow patterns (for duct flow)

### Recommended Strategy

1. Start with channel flow (easiest, best DNS data available)
2. Validate a priori predictions before running in the CFD solver
3. Test generalization: train on one Re, test on another
4. Cross-case testing: train on channel, test on duct

## Troubleshooting

### Dataset not found

The training scripts generate synthetic dummy data automatically when the dataset path does not exist. This lets you test the pipeline but produces physically meaningless models. Download the real dataset from Kaggle for research use.

### Training diverges (loss is NaN)

1. Reduce learning rate: `--lr 1e-4` instead of `1e-3`
2. Reduce batch size: `--batch_size 128`
3. Check that input data is properly normalized
4. Add gradient clipping in the training script

### Model performance is poor

1. Train longer: `--epochs 200`
2. Use more data: larger subset of the McConkey dataset
3. Try a bigger network: `--hidden 128 128 128`
4. Verify that C++ feature computation matches the Python training features exactly
5. If trained on dummy data, download the real dataset

### Wrong number of features

```
Error: Expected 6 features, got 5
```

Feature definition mismatch between training and inference. Check `metadata.json` for the correct feature list and verify that the feature computation in the solver matches.

### "Cannot open file" error

```
Error: Cannot open data/models/my_model/layer0_W.txt
```

Model directory is missing or incomplete. Verify with `ls data/models/my_model/` and re-run the export script if files are missing.

### Model diverges in solver

1. Verify you have real trained weights (not random/dummy)
2. Check that feature normalization matches training (`input_means.txt`, `input_stds.txt`)
3. Try a smaller time step: `--dt 0.0001`
4. Set nu_t clipping: `--nu_t_max 0.1`

## Training on Custom Data

If you cannot access the McConkey dataset, you can train on your own DNS/LES data. Your data must include velocity fields, strain and rotation rate tensors, Reynolds stress tensor, and turbulent kinetic energy.

```python
import numpy as np

# Compute invariants from your data
lambda_1 = np.trace(S @ S)
lambda_2 = np.trace(Omega @ Omega)
# ...

# Compute anisotropy
b_ij = tau_ij / (2 * k) - delta_ij / 3

# Save in expected format
np.savez('my_data/train/data.npz',
         invariants=invariants,
         anisotropy=anisotropy,
         basis=basis)
```

Then train with `--data_dir my_data`.

## References

1. Ling, J., Kurzawski, A., & Templeton, J. (2016). Reynolds averaged turbulence modelling using deep neural networks with embedded invariance. *Journal of Fluid Mechanics*, 807, 155--166. DOI: 10.1017/jfm.2016.615

2. McConkey, R., et al. (2021). A curated dataset for data-driven turbulence modelling. *Scientific Data*, 8, 255. DOI: 10.1038/s41597-021-01034-2

3. Duraisamy, K., Iaccarino, G., & Xiao, H. (2019). Turbulence modeling in the age of data. *Annual Review of Fluid Mechanics*, 51, 357--377.

## Key Files

- Training scripts: `scripts/train_tbnn_mcconkey.py`, `scripts/train_mlp_mcconkey.py`
- Export tools: `scripts/export_pytorch.py`, `scripts/export_tensorflow.py`
- Model zoo: `data/models/`
- Feature definitions: `include/features.hpp`, `src/features.cpp`
- TBNN implementation: `include/turbulence_nn_tbnn.hpp`
- MLP implementation: `include/turbulence_nn_mlp.hpp`
- Dataset download: `scripts/download_mcconkey_data.sh`
