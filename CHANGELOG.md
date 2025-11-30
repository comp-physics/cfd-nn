# Changelog

## [Current] - Neural Network Turbulence Modeling

### Complete Training Infrastructure

**Training Scripts:**
- `scripts/train_tbnn_mcconkey.py` - Train TBNN (Ling et al. 2016) on DNS/LES data
- `scripts/train_mlp_mcconkey.py` - Train MLP for scalar eddy viscosity
- `scripts/validate_trained_model.py` - Validate against test data
- `scripts/run_all_models.py` - Automated model comparison
- `scripts/download_mcconkey_data.sh` - Dataset download script

**Documentation:**
- `docs/TRAINING_GUIDE.md` - Complete training workflow
- `docs/DATASET_INFO.md` - Dataset documentation
- `QUICK_TRAIN.md` - 30-minute quick start
- `requirements.txt` - Python dependencies

**Dataset Integration:**
- McConkey et al. (2021) curated turbulence dataset
- Pre-computed TBNN features (invariants, tensor basis)
- Multiple flow cases (channel, periodic hills, square duct)
- Automatic synthetic data generation for testing

**Key Features:**
- ✅ Train TBNN following Ling et al. (2016) architecture
- ✅ Export to C++ compatible format
- ✅ A priori and a posteriori validation
- ✅ Comprehensive benchmarking tools
- ✅ Works with or without real dataset

### Core Solver Features

**Turbulence Models:**
- Laminar flow (none)
- Mixing length with van Driest damping (baseline)
- Gene Expression Programming algebraic model (gep)
- Neural network scalar eddy viscosity (nn_mlp)
- Tensor Basis Neural Network (nn_tbnn)

**Numerics:**
- Adaptive time stepping based on CFL and diffusion limits
- Projection method for pressure-velocity coupling
- Second-order finite differences
- SOR solver for pressure Poisson equation

**Output:**
- VTK format for ParaView visualization
- Velocity profiles and field data
- Performance timing analysis

### Usage

```bash
# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4

# Train model
source venv/bin/activate
python scripts/train_tbnn_mcconkey.py --case periodic_hills --epochs 100

# Run solver
./channel --model nn_tbnn --nn_preset your_model --adaptive_dt

# Compare models
python scripts/run_all_models.py --case channel --plot
```

### Files Added

**Scripts (5):** train_tbnn_mcconkey.py, train_mlp_mcconkey.py, validate_trained_model.py, run_all_models.py, download_mcconkey_data.sh

**Documentation (4):** TRAINING_GUIDE.md, DATASET_INFO.md, QUICK_TRAIN.md, TRAINING_SUCCESS.md

**Configuration:** requirements.txt, activate_venv.sh

**Total:** ~3,400 lines of code and documentation
