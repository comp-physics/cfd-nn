# NN-CFD: Neural Network Turbulence Closures for Time-Accurate Incompressible Flow

![CI](https://github.com/comp-physics/nn-cfd/workflows/CI/badge.svg)

A **high-performance C++ solver** for **incompressible turbulence simulations** with **pluggable neural network closures**. Features fractional-step projection method with multigrid Poisson solver and pure C++ NN inference.

## Features

- **Fractional-step projection method** for incompressible Navier-Stokes
  - Explicit Euler time integration with adaptive time stepping
  - **Multigrid Poisson solver** (O(N) complexity, 10-100x faster than SOR)
  - Pressure projection for divergence-free velocity
- **Pseudo-time marching to steady RANS** for canonical flows (channel, periodic hills)
- **Multiple turbulence closures**:
  - Baseline algebraic (mixing length)
  - GEP symbolic regression
  - Scalar eddy viscosity neural network (MLP)
  - Tensor Basis Neural Network (TBNN - Ling et al. 2016)
- **Pure C++ NN inference** - no Python/TensorFlow at runtime
- **Complete training pipeline** - train on real DNS/LES data
- **Performance instrumentation** - detailed timing analysis

## Quick Start

### Build the Solver

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

### Run Examples

```bash
# Steady-state examples
# --------------------
# Laminar channel flow (steady state)
./channel --Nx 32 --Ny 64 --nu 0.01 --adaptive_dt --max_iter 10000

# Same, but using Reynolds number (auto-computes nu)
./channel --Nx 32 --Ny 64 --Re 2000 --adaptive_dt --max_iter 10000

# Turbulent with baseline model
./channel --Nx 64 --Ny 128 --nu 0.001 --model baseline --adaptive_dt

# GEP algebraic model
./channel --model gep --adaptive_dt

# Neural network model
./channel --model nn_tbnn --nn_preset test_tbnn --adaptive_dt

# Periodic hills
./periodic_hills --Nx 64 --Ny 96 --model baseline --adaptive_dt

# Higher Reynolds number turbulent channel
./channel --Nx 128 --Ny 256 --Re 5000 --model baseline --adaptive_dt
```

## Training Neural Network Models

Train your own turbulence models on real DNS/LES data:

```bash
# Setup environment (one-time)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download dataset (~500 MB)
bash scripts/download_mcconkey_data.sh

# Train TBNN model (~30 min on CPU)
python scripts/train_tbnn_mcconkey.py \
    --data_dir mcconkey_data \
    --case periodic_hills \
    --output data/models/tbnn_hills \
    --epochs 100

# Validate against DNS
python scripts/validate_trained_model.py \
    --model data/models/tbnn_hills \
    --test_data mcconkey_data/periodic_hills/test/data.npz \
    --plot

# Use in solver
cd build
./periodic_hills --model nn_tbnn --nn_preset tbnn_hills
```

**See `QUICK_TRAIN.md` or `docs/TRAINING_GUIDE.md` for complete instructions.**

## Project Structure

- `include/` - Headers (mesh, solver, turbulence models, NN)
- `src/` - C++ implementations
- `app/` - Main executables (channel, periodic_hills)
- `scripts/` - Training and comparison tools
- `data/models/` - Trained model weights
- `docs/` - Detailed documentation
- `tests/` - Unit tests

## Command-Line Options

```bash
Grid:
  --Nx N, --Ny N        Grid cells (default: 64 x 64)
  --stretch             Enable y-direction stretching (default: off)

Physics:
  --Re VALUE            Reynolds number (default: 1000)
  --nu VALUE            Kinematic viscosity (default: 0.001)
  --dp_dx VALUE         Pressure gradient (driving force, default: -1.0)
  
  Note: Specify ONLY TWO of (Re, nu, dp_dx); the third is computed automatically:
    - --Re only          → uses default dp_dx, computes nu
    - --Re --nu          → computes dp_dx to achieve desired Re
    - --Re --dp_dx       → computes nu to achieve desired Re
    - --nu --dp_dx       → computes Re from these
    - none specified     → uses defaults (Re=1000, nu=0.001, dp_dx=-1.0)
  
  Specifying all three will error unless they are mutually consistent.

Turbulence Model:
  --model TYPE          none|baseline|gep|nn_mlp|nn_tbnn (default: none)
  --nn_preset NAME      Use model from data/models/<NAME>

Time Stepping:
  --adaptive_dt         Automatic time step (default: on)
  --dt VALUE            Fixed time step (default: 0.001)
  --max_iter N          Maximum iterations (default: 10000)
  --CFL VALUE           Max CFL number (default: 0.5)
  --tol VALUE           Convergence tolerance (default: 1e-6)
  
Output:
  --output DIR          Output directory (default: ./output)
  --num_snapshots N     Number of VTK snapshots (default: 10)
  --verbose / --quiet   Verbosity control (default: verbose)
```

### VTK Visualization Output

The solver automatically writes VTK files for visualization during the simulation:

```bash
# Default: 10 snapshots during simulation + final
./channel --Nx 64 --Ny 128 --nu 0.01 --max_iter 1000
# Outputs: channel_1.vtk through channel_10.vtk + channel_final.vtk

# More snapshots for detailed time evolution
./channel --Nx 64 --Ny 128 --nu 0.01 --max_iter 1000 --num_snapshots 20
# Outputs: 20 intermediate snapshots + final

# Only final state (faster, less disk space)
./channel --Nx 64 --Ny 128 --nu 0.01 --max_iter 1000 --num_snapshots 0
# Outputs: only channel_final.vtk
```

VTK files can be visualized with ParaView, VisIt, or similar tools.
```

## Available Turbulence Models

| Model | Type | Description | Speed | Accuracy |
|-------|------|-------------|-------|----------|
| `none` | Laminar | No turbulence model | ⚡⚡⚡⚡⚡ | N/A |
| `baseline` | Algebraic | Mixing length + van Driest | ⚡⚡⚡⚡ | Moderate |
| `gep` | Symbolic | Gene Expression Programming | ⚡⚡⚡ | Good |
| `nn_mlp` | Neural Net | Scalar eddy viscosity | ⚡⚡ | Good |
| `nn_tbnn` | Neural Net | Anisotropic stress (Ling 2016) | ⚡ | Best |

## Governing Equations

**Incompressible RANS:**
```
∂ū_i/∂t + ū_j ∂ū_i/∂x_j = -(1/ρ) ∂p̄/∂x_i + ∂/∂x_j[(ν + ν_t)(∂ū_i/∂x_j)]

∂ū_i/∂x_i = 0
```

**Numerical Method:**
- Projection method (pressure-velocity decoupling)
- Pseudo-time stepping to steady state
- Second-order finite differences
- SOR pressure solver

## Turbulence Closures

### 1. Mixing Length (Baseline)
```
ν_t = (κy)² |S| (1 - exp(-y⁺/A⁺))²
```
Fast, classical model with wall damping.

### 2. GEP (Gene Expression Programming)
Algebraic corrections learned from data. Fast and interpretable.

### 3. MLP (Multi-Layer Perceptron)
```
ν_t = NN(S, Ω, y/δ, k, ω, |u|)
```
Direct prediction from flow features. 6→32→32→1 architecture.

### 4. TBNN (Tensor Basis Neural Network)
```
b_ij = Σ_n G_n(λ₁,...,λ₅) × T^(n)_ij(S, Ω)
```
Frame-invariant anisotropy prediction. 5→64→64→64→4 architecture following Ling et al. (2016).

## Documentation

- **`QUICK_START.md`** - Build and run in 60 seconds
- **`QUICK_TRAIN.md`** - Train a model in 30 minutes
- **`VALIDATION.md`** - Test results and validation
- **`docs/TRAINING_GUIDE.md`** - Complete training workflow
- **`docs/DATASET_INFO.md`** - McConkey dataset guide
- **`DOCUMENTATION_INDEX.md`** - Full documentation index

## McConkey Dataset

This project integrates with the **McConkey et al. (2021)** dataset:

- **Reference**: *Scientific Data* 8, 255 (2021)
- **Content**: RANS + DNS/LES data for multiple flow cases
- **Features**: Pre-computed TBNN invariants and tensor basis
- **Cases**: Channel flow, periodic hills, square duct
- **Download**: `bash scripts/download_mcconkey_data.sh`

## Performance

Timing on 64×128 grid, 10,000 iterations:

| Model | Time/Iter | vs Baseline | Notes |
|-------|-----------|-------------|-------|
| Laminar | 0.01 ms | 1.0× | Reference |
| Baseline | 0.05 ms | 5× | Algebraic model |
| GEP | 0.08 ms | 8× | Symbolic expressions |
| MLP | 0.4 ms | 40× | Small neural net |
| TBNN | 2.1 ms | 210× | Large neural net |

NN models are slower but provide data-driven accuracy for complex flows.

## Validation

**Laminar channel flow** validates against analytical Poiseuille solution:
- L2 error: 0.13% (nu=0.1, 10k iterations)
- See `VALIDATION.md` for detailed results

**Turbulent flows** compare against:
- DNS/LES data (via McConkey dataset)
- Published TBNN results (Ling et al. 2016)

## Dependencies

**C++ Solver**: Standard library only (no external dependencies)

**Training Pipeline**: 
```bash
pip install torch numpy pandas scikit-learn matplotlib
```
(Optional - only needed for training, not for running solver)

## Technical Details

- **C++ Standard**: C++17
- **Build System**: CMake 3.10+
- **Memory Management**: RAII, smart pointers
- **Parallelization**: None (single-threaded, but fast)

## References

**Neural Network Architecture:**
- Ling, J., Kurzawski, A., & Templeton, J. "Reynolds averaged turbulence modelling using deep neural networks with embedded invariance." *JFM* 807 (2016)

**Dataset:**
- McConkey, R., et al. "A curated dataset for data-driven turbulence modelling." *Scientific Data* 8 (2021)

**Turbulence Modeling:**
- Pope, S.B. "Turbulent Flows." Cambridge University Press (2000)
- Duraisamy, K., Iaccarino, G., & Xiao, H. "Turbulence modeling in the age of data." *Ann. Rev. Fluid Mech.* 51 (2019)

## Citation

If you use this code, please cite:

```bibtex
@software{nncfd2025,
  title={NN-CFD: Neural Network Turbulence Closures for RANS},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/nn-cfd}
}
```

And cite the TBNN paper if using neural network models:

```bibtex
@article{ling2016reynolds,
  title={Reynolds averaged turbulence modelling using deep neural networks with embedded invariance},
  author={Ling, Julia and Kurzawski, Andrew and Templeton, Jeremy},
  journal={Journal of Fluid Mechanics},
  volume={807},
  pages={155--166},
  year={2016}
}
```

## License

MIT License - see `license` file

## Contributing

Contributions welcome! Areas for improvement:
- Additional turbulence models
- More test cases
- GPU acceleration
- Advanced solvers (multigrid, preconditioners)
- 3D extension

---

**For detailed implementation, see header files in `include/` with extensive comments.**
