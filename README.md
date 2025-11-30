# NN-CFD: Neural Network Turbulence Closures for CFD

A C++ codebase for incompressible RANS simulations with pluggable turbulence closures, including neural-network-based models.

## Features

- Steady incompressible RANS solver using projection method
- Finite volume discretization on structured 2D grids
- Multiple turbulence closures:
  - Laminar (no turbulence model)
  - Baseline algebraic eddy viscosity (mixing length with van Driest damping)
  - Neural network scalar eddy viscosity (MLP-based)
  - TBNN-style neural network for Reynolds stress anisotropy
- Pure C++ neural network inference (no external dependencies)
- Timing instrumentation for performance comparison

## Building

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

### Build Options

- `-DUSE_BLAS=ON`: Enable BLAS for potential linear algebra acceleration
- `-DBUILD_TESTS=ON/OFF`: Enable/disable unit tests (default: ON)

## Running

### Channel Flow (Poiseuille validation)

```bash
./channel --Nx 32 --Ny 64 --nu 0.01 --max_iter 50000 --model laminar
```

Options:
- `--Nx`, `--Ny`: Grid resolution
- `--nu`: Kinematic viscosity
- `--Re`: Reynolds number (alternative to nu)
- `--dt`: Time step
- `--max_iter`: Maximum iterations
- `--tol`: Convergence tolerance
- `--model`: Turbulence model (none/laminar, baseline, nn_mlp, nn_tbnn)
- `--weights`: Path to NN weights directory
- `--scaling`: Path to NN scaling parameters
- `--stretch`: Enable wall-normal grid stretching
- `--verbose/--quiet`: Print progress

### Periodic Hills

```bash
./periodic_hills --Nx 64 --Ny 48 --model baseline
```

## Project Structure

```
nn-cfd/
├── CMakeLists.txt
├── include/
│   ├── mesh.hpp           # Structured grid definition
│   ├── fields.hpp         # Scalar/vector/tensor field storage
│   ├── solver.hpp         # RANS solver interface
│   ├── poisson_solver.hpp # Pressure Poisson solver
│   ├── turbulence_model.hpp    # Abstract turbulence closure
│   ├── turbulence_baseline.hpp # Mixing length model
│   ├── turbulence_nn_mlp.hpp   # NN scalar eddy viscosity
│   ├── turbulence_nn_tbnn.hpp  # TBNN anisotropy model
│   ├── nn_core.hpp        # MLP implementation
│   ├── features.hpp       # RANS feature computation
│   ├── timing.hpp         # Performance instrumentation
│   └── config.hpp         # Configuration handling
├── src/
│   └── *.cpp              # Implementations
├── app/
│   ├── main_channel.cpp       # Channel flow driver
│   └── main_periodic_hills.cpp # Periodic hills driver
├── data/                  # NN weights and scaling parameters
└── tests/
    ├── test_mesh.cpp
    └── test_poisson.cpp
```

## Neural Network Weight Format

For `nn_mlp` and `nn_tbnn` models, weights should be provided in text format:

```
data/
├── layer0_W.txt   # Weight matrix (out_dim x in_dim), space-separated
├── layer0_b.txt   # Bias vector (out_dim), one value per line
├── layer1_W.txt
├── layer1_b.txt
├── ...
├── input_means.txt  # Input feature means (optional)
└── input_stds.txt   # Input feature stds (optional)
```

## Governing Equations

The solver implements the incompressible RANS equations:

$$\frac{\partial \bar{u}_i}{\partial t} + \bar{u}_j \frac{\partial \bar{u}_i}{\partial x_j} = -\frac{1}{\rho}\frac{\partial \bar{p}}{\partial x_i} + \frac{\partial}{\partial x_j}\left[(\nu + \nu_t)\frac{\partial \bar{u}_i}{\partial x_j}\right]$$

with continuity:

$$\frac{\partial \bar{u}_i}{\partial x_i} = 0$$

For TBNN models, the Reynolds stress anisotropy is computed as:

$$b_{ij} = \sum_n G_n(\lambda) T^{(n)}_{ij}(S, \Omega)$$

where $G_n$ are NN-predicted coefficients and $T^{(n)}$ are tensor basis functions.

## Validation

The channel flow solver validates against the analytical Poiseuille solution:

$$u(y) = \frac{-\partial p/\partial x}{2\nu}(H^2 - y^2)$$

where $H$ is the channel half-height.

## License

MIT License


