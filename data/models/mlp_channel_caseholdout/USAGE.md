# MLP Channel Flow Model - Usage Guide

## Model Overview

**Type:** Multi-Layer Perceptron (MLP) for Scalar Eddy Viscosity  
**Training Data:** McConkey et al. (2021) channel flow dataset  
**Training Date:** December 18, 2025  
**Architecture:** 6 → 32 → 32 → 1  
**Parameters:** 1,313  
**Training Time:** ~2 minutes on Tesla V100 GPU

## Quick Start

```bash
# From build directory
./channel --model nn_mlp --nn_preset mlp_channel_caseholdout --max_iter 10000

# With custom parameters
./channel --model nn_mlp --nn_preset mlp_channel_caseholdout \
    --Nx 128 --Ny 256 --Re 5000 --adaptive_dt --max_iter 20000
```

## Model Details

### Architecture

```
Input Layer:  6 features
   ↓
Hidden Layer 1: 32 neurons (Tanh activation)
   ↓
Hidden Layer 2: 32 neurons (Tanh activation)
   ↓
Output Layer: 1 neuron (ReLU activation)
   ↓
Output: ν_t (turbulent eddy viscosity)
```

### Input Features

The model expects 6 normalized features:

1. **|S|** - Strain rate magnitude
   - Computed from velocity gradients: √(2S_ij S_ij)
   - Normalized using training data statistics

2. **|Ω|** - Rotation rate magnitude
   - Computed from velocity gradients: √(2Ω_ij Ω_ij)
   - Normalized using training data statistics

3. **y/δ** - Normalized wall distance
   - y = distance from wall
   - δ = channel half-height
   - Range: [0, 1] for channel flow

4. **k** - Turbulent kinetic energy
   - From RANS k-ω model
   - Normalized using training data statistics

5. **ω** - Specific dissipation rate
   - From RANS k-ω model
   - Normalized using training data statistics

6. **|u|** - Velocity magnitude
   - √(u² + v²)
   - Normalized using training data statistics

### Output

Single scalar value: **ν_t** (turbulent eddy viscosity)

- Units: m²/s (same as molecular viscosity ν)
- Physical constraint: ν_t ≥ 0 (enforced by ReLU activation)
- Used in momentum equations: τ_ij = (ν + ν_t) × (∂u_i/∂x_j + ∂u_j/∂x_i)

## Training Details

### Methodology

Follows **Ling et al. (2016)** training protocol:

- **Dataset:** McConkey et al. (2021) channel flow cases
- **Validation:** Case-holdout (trained on subset of Re, validated on held-out Re)
- **Normalization:** Z-score (mean=0, std=1) for all features
- **Optimizer:** Adam with initial LR = 1e-3
- **Scheduler:** ReduceLROnPlateau (patience=20, factor=0.5)
- **Batch size:** 1024
- **Epochs:** 500 (with early stopping)
- **Loss function:** MSE on ν_t

### Hyperparameters

```python
{
    "hidden_layers": [32, 32],
    "activations": ["tanh", "tanh", "relu"],
    "learning_rate": 1e-3,
    "batch_size": 1024,
    "epochs": 500,
    "optimizer": "Adam",
    "scheduler": "ReduceLROnPlateau",
    "normalization": "z-score"
}
```

### Training Command

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

## Performance

### Inference Speed

Measured on 16×32 grid, 10,000 iterations:

| Backend | Time per Iteration | Relative Speed |
|---------|-------------------|----------------|
| CPU (single core) | 0.388 ms | 1x |
| GPU (Tesla V100) | 0.008 ms | **48x faster** |

### Comparison with Other Models

| Model | Parameters | Inference Time | Relative Speed |
|-------|-----------|----------------|----------------|
| **MLP** | 1,313 | 0.388 ms | **1x (fastest)** |
| TBNN | 8,964 | 2.119 ms | 5.5x slower |
| Baseline (mixing length) | 0 | 0.046 ms | 8.4x faster* |

*Note: Baseline is faster but less accurate; MLP provides better predictions

### GPU Acceleration

The MLP model has full GPU offload support:

- **GPU build:** Compiled with `-DUSE_GPU_OFFLOAD=ON`
- **Runtime:** Automatically uses GPU if available
- **Speedup:** 10-50x depending on grid size (larger grids = better speedup)
- **Memory:** Minimal GPU memory usage (~5 KB for weights)

## Usage Examples

### Basic Channel Flow

```bash
./channel --model nn_mlp --nn_preset mlp_channel_caseholdout
```

### High Reynolds Number

```bash
./channel --model nn_mlp --nn_preset mlp_channel_caseholdout \
    --Nx 256 --Ny 512 --Re 10000 --adaptive_dt
```

### With VTK Output

```bash
./channel --model nn_mlp --nn_preset mlp_channel_caseholdout \
    --max_iter 20000 --num_snapshots 20 --output ./output_mlp
```

### Comparing with Other Models

```bash
# Run MLP
./channel --model nn_mlp --nn_preset mlp_channel_caseholdout --output ./mlp_results

# Run TBNN for comparison
./channel --model nn_tbnn --nn_preset tbnn_channel_caseholdout --output ./tbnn_results

# Run baseline
./channel --model baseline --output ./baseline_results
```

## Validation

### Test Cases

The model has been validated on:

1. **Channel flow** at Re_τ = 180, 395, 550, 590
2. **Turbulent statistics:** Mean velocity profile, Reynolds stresses
3. **Convergence:** Steady-state solution achieved in <10,000 iterations
4. **Stability:** No NaN/Inf values, ν_t ≥ 0 always satisfied

### Expected Results

For Re_τ = 180 channel flow:

- **Centerline velocity:** U_cl ≈ 18.2 (in wall units)
- **Friction velocity:** u_τ ≈ 1.0
- **Convergence:** Residual < 1e-6 in ~5,000 iterations

### Limitations

1. **Geometry-specific:** Trained on channel flow, may not generalize to other geometries
2. **No frame invariance:** Not guaranteed to be coordinate-independent
3. **Scalar output:** Cannot predict anisotropic Reynolds stresses (only ν_t)
4. **Reynolds number range:** Best performance within training range (Re_τ = 180-590)

## Troubleshooting

### Model Not Found

```
Error: Could not load model from data/models/mlp_channel_caseholdout
```

**Solution:** Ensure you're running from the correct directory (build/ or repo root)

### GPU Not Available

```
Warning: GPU requested but not available, falling back to CPU
```

**Solution:** 
- Check GPU build: `cmake .. -DUSE_GPU_OFFLOAD=ON`
- Verify GPU: `nvidia-smi` or `rocm-smi`

### Divergence / NaN Values

```
Error: NaN detected in velocity field
```

**Possible causes:**
- Time step too large (use `--adaptive_dt`)
- Reynolds number too high for grid resolution
- Incompatible boundary conditions

**Solution:**
```bash
./channel --model nn_mlp --nn_preset mlp_channel_caseholdout \
    --adaptive_dt --CFL 0.3 --max_iter 50000
```

## File Structure

```
mlp_channel_caseholdout/
├── layer0_W.txt       # Hidden layer 1 weights (6 × 32)
├── layer0_b.txt       # Hidden layer 1 biases (32)
├── layer1_W.txt       # Hidden layer 2 weights (32 × 32)
├── layer1_b.txt       # Hidden layer 2 biases (32)
├── layer2_W.txt       # Output layer weights (32 × 1)
├── layer2_b.txt       # Output layer bias (1)
├── input_means.txt    # Feature means for normalization (6)
├── input_stds.txt     # Feature stds for normalization (6)
├── metadata.json      # Model metadata
└── USAGE.md          # This file
```

## References

### Training Methodology

Ling, J., Kurzawski, A., & Templeton, J. (2016). Reynolds averaged turbulence modelling using deep neural networks with embedded invariance. *Journal of Fluid Mechanics*, 807, 155-166. DOI: [10.1017/jfm.2016.615](https://doi.org/10.1017/jfm.2016.615)

### Training Dataset

McConkey, R., Yee, E., & Lien, F. S. (2021). A curated dataset for data-driven turbulence modelling. *Scientific Data*, 8(1), 255. DOI: [10.1038/s41597-021-01034-2](https://doi.org/10.1038/s41597-021-01034-2)

## Support

For issues or questions:

1. Check the main documentation: `docs/TRAINING_GUIDE.md`
2. Review test cases: `tests/test_backend_execution.cpp`
3. See model zoo: `data/models/README.md`

## Version History

- **v1.0** (Dec 18, 2025): Initial trained model
  - Architecture: 6 → 32 → 32 → 1
  - Training: 500 epochs on McConkey channel data
  - Validation: Case-holdout strategy
  - GPU support: Full OpenMP offload

