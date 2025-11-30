# Documentation Index

## Quick Start Guides

- **`README.md`** - Main project documentation and usage
- **`QUICK_START.md`** - Build and run in 60 seconds  
- **`QUICK_TRAIN.md`** - Train a neural network model in 30 minutes

## Training Neural Network Models

- **`docs/TRAINING_GUIDE.md`** - Complete training workflow
- **`docs/DATASET_INFO.md`** - McConkey dataset documentation
- **`TRAINING_SUCCESS.md`** - Verified working setup
- **`requirements.txt`** - Python dependencies

## Validation & Testing

- **`VALIDATION.md`** - Test results and validation against analytical/DNS solutions
- **`TESTING.md`** - Unit testing guide and test suite documentation

## Model Integration

- **`data/models/README.md`** - Model zoo and published models
- **`docs/MODEL_ZOO_GUIDE.md`** - Detailed integration guide

## Documentation by Task

| I want to... | Read this |
|--------------|-----------|
| Get started quickly | `README.md` or `QUICK_START.md` |
| Train my own NN model | `QUICK_TRAIN.md` → `docs/TRAINING_GUIDE.md` |
| Understand the dataset | `docs/DATASET_INFO.md` |
| See validation results | `VALIDATION.md` |
| Run unit tests | `TESTING.md` |
| Add a published model | `data/models/README.md` |
| Understand the code | Header files in `include/` |
| Export PyTorch weights | `scripts/export_pytorch.py` |

## Project Organization

**Documentation:** README.md, QUICK_START.md, QUICK_TRAIN.md, VALIDATION.md, TESTING.md, TRAINING_SUCCESS.md, CHANGELOG.md, docs/

**Code:** include/ (headers), src/ (implementations), app/ (executables), scripts/ (training), tests/ (unit tests)

**Data:** data/models/ (trained weights)

## Available Turbulence Models

| Model | Type | Training Required | Description |
|-------|------|-------------------|-------------|
| `none` | Laminar | No | No turbulence model |
| `baseline` | Algebraic | No | Mixing length |
| `gep` | Symbolic | No | Gene Expression Programming |
| `nn_mlp` | Neural Net | Yes | Scalar eddy viscosity |
| `nn_tbnn` | Neural Net | Yes | Tensor Basis NN (Ling 2016) |

## Quick Reference

### Build
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4
```

### Train Model
```bash
source venv/bin/activate
python scripts/train_tbnn_mcconkey.py --case periodic_hills --epochs 100
```

### Run
```bash
./channel --model nn_tbnn --nn_preset your_model --adaptive_dt
```

## Project Goals

This solver implements:
1. Steady incompressible RANS for canonical flows
2. Finite volume/difference on structured grids
3. Multiple turbulence closures (classical + neural network)
4. Pure C++ NN inference (no runtime dependencies)
5. Complete ML training pipeline (Python + PyTorch)
6. Performance instrumentation

**Focus:** Production-ready CFD solver with state-of-the-art ML turbulence closures
