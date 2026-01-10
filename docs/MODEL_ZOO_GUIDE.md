# Model Zoo: Integrating Published NN Turbulence Models

## Overview

The model zoo provides infrastructure to easily use published neural network turbulence models. Place pre-trained weights in `data/models/<model_name>/` and load them with `--nn_preset <model_name>`.

## Quick Start

### Using a Preset Model

```bash
# Use a preset model by name
./channel --model nn_mlp --nn_preset example_scalar_nut

# Or for TBNN
./channel --model nn_tbnn --nn_preset example_tbnn
```

### Manual Path (without preset)

```bash
# Still works - specify paths directly
./channel --model nn_mlp --weights path/to/weights --scaling path/to/scaling
```

## How It Works

### 1. Command-Line Interface

The `--nn_preset` option tells the solver to look for a model in `data/models/<NAME>/`:

```bash
--nn_preset example_scalar_nut
```

Internally maps to:
- `nn_weights_path = data/models/example_scalar_nut`
- `nn_scaling_path = data/models/example_scalar_nut`

### 2. Priority

If both `--nn_preset` and `--weights` are given:
```bash
./channel --nn_preset foo --weights bar
```

The explicit `--weights bar` takes precedence (overrides the preset).

### 3. Config File

You can also specify presets in a config file:

```
# config.txt
turb_model = nn_mlp
nn_preset = example_scalar_nut
Nx = 64
Ny = 128
```

Then:
```bash
./channel --config config.txt
```

## Directory Structure

```
data/models/
+-- README.md                    # Overview and list of models
+-- example_scalar_nut/          # Example scalar eddy viscosity
|   +-- layer0_W.txt
|   +-- layer0_b.txt
|   +-- layer1_W.txt
|   +-- layer1_b.txt
|   +-- layer2_W.txt
|   +-- layer2_b.txt
|   +-- input_means.txt
|   +-- input_stds.txt
|   +-- metadata.json
+-- example_tbnn/                # Example TBNN model
|   +-- layer0_W.txt
|   +-- ...
|   +-- layer3_b.txt
|   +-- input_means.txt
|   +-- input_stds.txt
|   +-- metadata.json
+-- <your_model_name>/           # Your published model
    +-- ...
```

## Adding a Published Model

### Step 1: Get the Weights

Three options:

#### A. From PyTorch

```bash
python scripts/export_pytorch.py model.pth \
    --output data/models/my_model \
    --means means.npy \
    --stds stds.npy
```

#### B. From TensorFlow/Keras

```bash
python scripts/export_tensorflow.py model.h5 \
    --output data/models/my_model \
    --means means.npy \
    --stds stds.npy
```

#### C. From Paper Supplementary Materials

Some papers provide weights as text files or HDF5. Convert to the required format:
- `layer0_W.txt`, `layer0_b.txt`, ... (one per layer)
- `input_means.txt` (feature normalization means)
- `input_stds.txt` (feature normalization stds)

### Step 2: Create Metadata

Create `metadata.json` documenting the model:

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

### Step 3: Validate

Run the model on the same test case from the paper:

```bash
./channel --model nn_tbnn --nn_preset ling_tbnn_2016 \
    --Nx 64 --Ny 128 --Re 2800 --max_steps 50000
```

Compare results:
- Mean velocity profile
- Reynolds stress components
- Friction coefficient
- Convergence behavior

Document any differences in the metadata.

### Step 4: Update the Zoo README

Add your model to `data/models/README.md`:

```markdown
#### Ling et al. (2016) - TBNN
**Status:** Implemented and validated
**Location:** `data/models/ling_tbnn_2016/`
**Usage:** `./channel --model nn_tbnn --nn_preset ling_tbnn_2016`
**Validation:** Matches published results within 5% for Re_tau=180 channel flow
```

## Feature Definitions

Different published models use different input features. You may need to adjust feature computation to match.

### Current Feature Sets

#### `scalar_nut_v1` (6 features)
Used by: `example_scalar_nut`

1. Strain rate magnitude: |S|
2. Rotation rate magnitude: |Omega|  
3. Normalized wall distance: y/delta
4. Turbulent kinetic energy: k
5. Specific dissipation: omega
6. Wall distance: d_wall

#### `tbnn_ling2016` (5 invariants)
Used by: `example_tbnn`, Ling et al. 2016

Computed from normalized S~ and Omega~:
1. lambda_1 = tr(S~^2)
2. lambda_2 = tr(Omega~^2)
3. eta_1 = tr(S~^3)
4. eta_2 = tr(Omega~^2S~)
5. q* = q/k (normalized non-dimensionality measure)

### Adding New Feature Sets

If a published model uses different features:

1. Add feature computation in `src/features.cpp`
2. Create new enum value in feature type
3. Update `TurbulenceNNMLP` or `TurbulenceNNTBNN` to support it
4. Document in metadata.json

## Troubleshooting

### Model diverges immediately

**Cause:** Random/untrained weights, feature scaling mismatch, or numerical instability

**Solutions:**
1. Verify you have real trained weights (not dummy/random)
2. Check feature normalization matches training
3. Try smaller time step: `--dt 0.0001`
4. Check `nu_t_max` clipping: `--nu_t_max 0.1`

### "Cannot open file" error

```
Error: Cannot open data/models/my_model/layer0_W.txt
```

**Cause:** Model directory doesn't exist or is incomplete

**Solutions:**
1. Verify directory exists: `ls data/models/my_model/`
2. Check all required files are present
3. Run export script again if files are missing

### Wrong number of features

```
Error: Expected 6 features, got 5
```

**Cause:** Feature definition mismatch between training and inference

**Solutions:**
1. Check model metadata for correct feature list
2. Verify feature computation matches training
3. Update feature computer or retrain model

## Best Practices

### Documentation

- Always include `metadata.json` with full paper reference
- Document exact test case parameters for reproducibility
- Note any deviations from published results
- Include training framework and dataset info

### Validation

- Run at least one test case from the original paper
- Compare quantitative metrics (not just qualitative)
- Document computational cost (inference time)
- Test stability across different Re numbers

### Organization

- One model per directory
- Use descriptive names: `author_type_year` (e.g., `ling_tbnn_2016`)
- Keep original paper's architecture exactly
- Note if you made any modifications

### Version Control

- Don't commit large weight files to main repo
- Use Git LFS for weight files, or
- Store on external service (Zenodo, etc.)
- Include download script if weights are external

## Examples

### Reproduce Ling et al. (2016)

```bash
# After getting weights from authors
./channel --model nn_tbnn --nn_preset ling_tbnn_2016 \
    --Nx 128 --Ny 256 \
    --Re 2800 \
    --max_steps 100000 \
    --tol 1e-8 \
    --stretch
```

### Compare Multiple Models

```bash
# Baseline
./channel --model baseline --output output/baseline/

# Model 1
./channel --model nn_mlp --nn_preset wu_2018 --output output/wu_2018/

# Model 2  
./channel --model nn_tbnn --nn_preset ling_2016 --output output/ling_2016/

# Compare results
python scripts/compare_results.py output/*/
```

## Next Steps

To populate the model zoo with real published models:

1. **Target high-value models** (see `data/models/README.md` for candidates)
2. **Contact authors** for pre-trained weights
3. **Export to text format** using provided scripts
4. **Validate** against published results
5. **Document** in metadata.json

## Resources

- **Export tools:** `scripts/export_pytorch.py`, `scripts/export_tensorflow.py`
- **Target models:** `data/models/README.md`
- **Feature definitions:** `include/features.hpp`
- **Validation protocol:** `VALIDATION.md`

