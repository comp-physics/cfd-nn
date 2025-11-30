# NN-CFD: Neural Network Turbulence Closures for Incompressible RANS

A **small, well-structured C++ codebase** for incompressible RANS simulations with **pluggable turbulence closures**, including neural-network-based models.

## Objectives

This solver implements:

1. **Steady incompressible RANS** for canonical flows (channel flow, periodic hills) in 2D
2. **Finite volume/finite difference** discretization on structured grids
3. **Multiple turbulence closures:**
   - Baseline algebraic eddy-viscosity (mixing length)
   - Scalar eddy-viscosity neural network: nu_t = NN(features)
   - TBNN-style neural network for anisotropy: b_ij = Sum_n G_n T^(n)_ij(features)
4. **Pure C++ NN inference** using pre-exported weights (no Python/TensorFlow/PyTorch at runtime)
5. **Performance instrumentation** to compare online runtime costs

Focus: CFD solver and NN inference infrastructure. Training is done externally in Python.

## Quick Start

```bash
# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

# Run laminar channel flow with adaptive time stepping
./channel --Nx 32 --Ny 64 --nu 0.01 --adaptive_dt --max_iter 20000

# Run with baseline turbulence model
./channel --Nx 64 --Ny 128 --nu 0.001 --model baseline --adaptive_dt

# Run with GEP algebraic model (no weights needed)
./channel --Nx 64 --Ny 128 --nu 0.001 --model gep --adaptive_dt

# Run with neural network model (using preset weights)
./channel --model nn_mlp --nn_preset example_scalar_nut --adaptive_dt

# Run periodic hills
./periodic_hills --Nx 64 --Ny 48 --model baseline --adaptive_dt
```

## Project Structure

```
nn-cfd/
+-- CMakeLists.txt
+-- include/              # All header files
|   +-- mesh.hpp          # Structured grid with ghost cells
|   +-- fields.hpp        # ScalarField, VectorField, TensorField
|   +-- solver.hpp        # RANS solver (projection method)
|   +-- poisson_solver.hpp # Pressure Poisson equation (SOR)
|   +-- turbulence_model.hpp       # Abstract base class
|   +-- turbulence_baseline.hpp    # Mixing length model
|   +-- turbulence_nn_mlp.hpp      # Scalar nu_t NN
|   +-- turbulence_nn_tbnn.hpp     # TBNN anisotropy NN
|   +-- nn_core.hpp       # MLP forward pass
|   +-- features.hpp      # Feature extraction for NNs
|   +-- timing.hpp        # Performance timing
|   +-- config.hpp        # Configuration parsing
+-- src/                  # Implementations
+-- app/
|   +-- main_channel.cpp       # Channel flow executable
|   +-- main_periodic_hills.cpp # Periodic hills executable
|   +-- test_nn_simple.cpp      # NN diagnostic tool
+-- data/
|   +-- models/           # NN model zoo (see below)
|   +-- README.md         # Weight format documentation
+-- scripts/
|   +-- generate_dummy_weights.py  # Random weights for testing
|   +-- export_pytorch.py          # PyTorch -> C++ format
|   +-- export_tensorflow.py       # TensorFlow -> C++ format
+-- tests/
    +-- test_mesh.cpp
    +-- test_poisson.cpp
```

## Command-Line Options

```bash
./channel [options]

Grid:
  --Nx N, --Ny N        Grid cells in x and y
  --stretch             Enable y-direction stretching

Physics:
  --nu VALUE            Kinematic viscosity
  --Re VALUE            Reynolds number (alternative to --nu)

Numerics:
  --dt VALUE            Time step
  --max_iter N          Maximum iterations
  --tol VALUE           Convergence tolerance

Turbulence Model:
  --model TYPE          none|laminar|baseline|gep|nn_mlp|nn_tbnn
  --nn_preset NAME      Use preset from data/models/<NAME>
  --weights DIR         NN weights directory (overrides preset)
  --scaling DIR         NN scaling directory

Adaptive Time Stepping:
  --adaptive_dt         Enable adaptive time stepping (recommended)
  --CFL VALUE           Max CFL number (default 0.5)

Output:
  --output DIR          Output directory
  --verbose / --quiet   Verbosity control
```

## Neural Network Model Zoo

Easily use published NN turbulence models via the preset system:

```bash
# Use a preset model from data/models/
./channel --model nn_mlp --nn_preset <model_name>
```

### Directory Structure

```
data/models/
+-- example_scalar_nut/    # Example MLP (6->32->32->1)
|   +-- layer*.txt         # Network weights
|   +-- input_means.txt    # Feature normalization
|   +-- input_stds.txt
|   +-- metadata.json      # Model documentation
+-- example_tbnn/          # Example TBNN (5->64->64->64->4)
    +-- ...
```

### Adding Published Models

1. **Export weights** from PyTorch or TensorFlow:
   ```bash
   python scripts/export_pytorch.py model.pth --output data/models/my_model
   ```

2. **Create metadata.json** documenting the model (see examples)

3. **Run:**
   ```bash
   ./channel --model nn_tbnn --nn_preset my_model
   ```

See `data/models/README.md` for target published models and validation protocol.

## Governing Equations

**Incompressible RANS:**

du_bar_i/dt + u_bar_j du_bar_i/dx_j = -(1/rho) dp_bar/dx_i + d/dx_j[(nu + nu_t)(du_bar_i/dx_j + du_bar_j/dx_i)] - du'_i u'_j/dx_j

**Continuity:**

du_bar_i/dx_i = 0

**Numerical Method:**
- Projection method for pressure-velocity coupling
- Pseudo-time stepping to steady state
- Second-order finite differences
- SOR solver for pressure Poisson equation

**Turbulence Closures:**

1. **Baseline:** Mixing length with van Driest damping
   - nu_t = (kappa*y)^2 |S| (1 - exp(-y+/A+))^2

2. **GEP:** Gene Expression Programming algebraic model (Weatheritt-Sandberg style)
   - No pre-trained weights needed - uses algebraic formulas

3. **NN-MLP:** Direct eddy viscosity prediction (requires pre-trained weights)
   - nu_t = max(0, min(NN(features), nu_t_max))

4. **NN-TBNN:** Anisotropy prediction (requires pre-trained weights)
   - b_ij = Sum_n G_n(invariants) T^(n)_ij(S, Omega)
   - Reynolds stresses: tau_ij = 2k b_ij

## Validation

**Laminar channel flow** validates against analytical Poiseuille solution:

u(y) = (dp/dx)/(2*nu) * (H^2 - y^2)

With `nu=0.1`, `dt=0.005`, the solver achieves **0.13% L2 error** in 10,000 iterations.

See `VALIDATION.md` for detailed results and recommended parameters.

## Performance

Timing breakdown for channel flow (16x32 grid, 10,000 iterations):

| Component | Time (s) | Time/Iter (ms) | % of Total |
|-----------|----------|----------------|------------|
| Laminar solver | 0.08 | 0.008 | 100% |
| Baseline turbulence | 0.46 | 0.046 | 575% |
| NN-MLP | 3.88 | 0.388 | 4850% |
| NN-TBNN | 21.19 | 2.119 | 26488% |

NN models are slower due to inference overhead but provide data-driven closure.

## Technical Details

**C++ Standard:** C++17  
**Dependencies:** Standard library only (optional BLAS)  
**Memory Management:** RAII, smart pointers (no raw pointers)  
**Build System:** CMake 3.10+  

**Key Classes:**
- `Mesh`: 2D structured grid with ghost cells and indexing
- `ScalarField`, `VectorField`, `TensorField`: Field containers
- `RANSSolver`: Main solver implementing projection method
- `TurbulenceModel`: Abstract base for closures
- `MLP`: Dense neural network forward pass
- `FeatureComputer`: Extracts local RANS features for NNs

## Files

- **`README.md`** (this file): Overview and usage
- **`CHANGELOG.md`**: Recent changes and new features
- **`VALIDATION.md`**: Detailed validation results
- **`data/models/README.md`**: Model zoo guide and target published models
- **`docs/MODEL_ZOO_GUIDE.md`**: Detailed model integration guide

## Next Steps

1. **Add real published models** (Ling TBNN, Weatheritt GEP, Wu MLP)
2. **Implement adaptive time stepping** for automatic dt selection
3. **Add VTK output** for ParaView visualization
4. **Optimize Poisson solver** (multigrid or CG)
5. **Benchmark multiple models** on canonical test cases

## References

Solver design based on standard projection methods for incompressible flow. Neural network architectures follow:

- **TBNN:** Ling et al., "Reynolds averaged turbulence modelling using deep neural networks with embedded invariance," JFM 807 (2016)
- **Mixing length:** Classic turbulence model with van Driest damping
- **Feature engineering:** Standard RANS invariants and wall distance

## License

MIT License - see `license` file

---

**For detailed implementation, see header files in `include/` with extensive comments.**
