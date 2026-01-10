# NN-CFD: Neural Network Turbulence Closures for Incompressible Flow

![CI](https://github.com/comp-physics/cfd-nn/workflows/CI/badge.svg)

A **high-performance C++ solver** for **incompressible turbulent flow** with **pluggable turbulence closures** ranging from classical algebraic models to advanced transport equations and data-driven neural networks. Features a fractional-step projection method with multiple Poisson solvers, pure C++ NN inference, and comprehensive GPU acceleration via OpenMP target offload.

## Features

- **Fractional-step projection method** for incompressible Navier-Stokes
  - Explicit Euler time integration with adaptive CFL-based time stepping
  - Multiple Poisson solvers with automatic selection (FFT, Multigrid, HYPRE)
  - Pressure projection for divergence-free velocity
- **Staggered MAC grid** with second-order central finite differences
- **10 turbulence closures**: algebraic, transport, EARSM, and neural network models
- **Pure C++ NN inference** - no Python/TensorFlow at runtime
- **GPU acceleration** via OpenMP target directives for NVIDIA and AMD GPUs

## Table of Contents

- [Quick Start](#quick-start)
- [Governing Equations](#governing-equations)
- [Numerical Methods](#numerical-methods)
- [Boundary Conditions](#boundary-conditions)
- [Poisson Solvers](#poisson-solvers)
- [Turbulence Closures](#turbulence-closures)
- [Supported Flow Configurations](#supported-flow-configurations)
- [Command-Line Options](#command-line-options)
- [GPU Acceleration](#gpu-acceleration)
- [Validation](#validation)
- [References](#references)

---

## Quick Start

### Build the Solver

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

### Run Examples

```bash
# Laminar channel flow (Poiseuille, analytical validation)
./channel --Nx 32 --Ny 64 --nu 0.01 --adaptive_dt --max_iter 10000

# Turbulent channel with SST k-omega
./channel --Nx 64 --Ny 128 --Re 5000 --model sst --adaptive_dt

# Neural network turbulence model
./channel --model nn_tbnn --nn_preset tbnn_channel_caseholdout --adaptive_dt

# 3D Taylor-Green vortex
./taylor_green_3d --Nx 64 --Ny 64 --Nz 64 --Re 100 --max_iter 1000
```

---

## Governing Equations

The solver implements the **incompressible Reynolds-Averaged Navier-Stokes (RANS) equations**:

### Momentum Equation

$$\frac{\partial \bar{u}_i}{\partial t} + \bar{u}_j \frac{\partial \bar{u}_i}{\partial x_j} = -\frac{1}{\rho} \frac{\partial \bar{p}}{\partial x_i} + \frac{\partial}{\partial x_j}\left[(\nu + \nu_t) \frac{\partial \bar{u}_i}{\partial x_j}\right] + f_i$$

### Continuity Equation (Incompressibility)

$$\nabla \cdot \mathbf{u} = 0$$

**Variables:**
| Symbol | Description |
|--------|-------------|
| $\bar{u}_i$ | Mean velocity components (u, v, w) |
| $\bar{p}$ | Mean pressure |
| $\nu$ | Kinematic viscosity |
| $\nu_t$ | Turbulent eddy viscosity (from closure model) |
| $f_i$ | Body force (e.g., pressure gradient driving force) |
| $\rho$ | Density (constant for incompressible flow) |

---

## Numerical Methods

### Fractional-Step Projection Method

The solver uses a **three-step fractional-step method** (Chorin 1968) to decouple pressure and velocity:

**Step 1: Provisional Velocity** (explicit momentum without pressure)

$$\frac{\mathbf{u}^* - \mathbf{u}^n}{\Delta t} = -(\mathbf{u}^n \cdot \nabla)\mathbf{u}^n + \nabla \cdot [(\nu + \nu_t) \nabla \mathbf{u}^n] + \mathbf{f}$$

**Step 2: Pressure Poisson Equation** (enforce incompressibility)

$$\nabla^2 p' = \frac{1}{\Delta t} \nabla \cdot \mathbf{u}^*$$

**Step 3: Velocity Correction** (project onto divergence-free space)

$$\mathbf{u}^{n+1} = \mathbf{u}^* - \Delta t \nabla p'$$

This ensures $\nabla \cdot \mathbf{u}^{n+1} = 0$ to machine precision.

### Spatial Discretization

All spatial derivatives use **second-order central finite differences** on a **staggered Marker-and-Cell (MAC) grid**:

| Variable | Grid Location |
|----------|---------------|
| u-velocity | x-faces (staggered in x) |
| v-velocity | y-faces (staggered in y) |
| w-velocity | z-faces (staggered in z) |
| Pressure, scalars | Cell centers |

**Gradient (central difference):**

$$\left.\frac{\partial u}{\partial x}\right|_{i,j} = \frac{u_{i+1,j} - u_{i-1,j}}{2\Delta x}$$

**Laplacian (5-point stencil in 2D, 7-point in 3D):**

$$\left.\nabla^2 u\right|_{i,j} = \frac{u_{i+1,j} - 2u_{i,j} + u_{i-1,j}}{\Delta x^2} + \frac{u_{i,j+1} - 2u_{i,j} + u_{i,j-1}}{\Delta y^2}$$

**Convective Schemes** (selectable via `convective_scheme` config):
- **Central differences** (default): Second-order accurate, lower numerical dissipation
- **First-order upwind**: More stable at high Reynolds numbers, increased numerical dissipation

**Variable Viscosity Diffusion:**

$$\nabla \cdot [(\nu + \nu_t) \nabla u] = \frac{1}{\Delta x^2}\left[\nu_{e}(u_{i+1,j} - u_{i,j}) - \nu_{w}(u_{i,j} - u_{i-1,j})\right] + \ldots$$

where $\nu_e, \nu_w$ are face-averaged effective viscosities.

### Time Integration

**Explicit Euler** with adaptive time stepping:

$$\Delta t = \text{CFL} \cdot \min\left(\frac{\Delta x}{|u|_{\max}}, \frac{\Delta y}{|v|_{\max}}, \frac{\Delta z}{|w|_{\max}}\right)$$

- **CFL number**: Default 0.5 (configurable via `--CFL`)
- **Steady-state convergence**: Iterate until $\|\mathbf{u}^{n+1} - \mathbf{u}^n\|_\infty < \text{tol}$

---

## Boundary Conditions

### Velocity Boundary Conditions

| Type | Description | Implementation |
|------|-------------|----------------|
| **Periodic** | Flow wraps around at boundaries | Ghost cells copy from opposite boundary |
| **No-slip (Wall)** | Zero velocity at solid surfaces | $u = v = w = 0$ at wall |
| **Inflow** | Prescribed velocity profile | User-defined function callbacks |
| **Outflow** | Convective/zero-gradient outflow | Extrapolation from interior |

**Note:** Inflow/Outflow boundary conditions are defined in the code interface but not yet fully implemented for all solvers.

### Pressure (Poisson) Boundary Conditions

The pressure Poisson equation supports three BC types:

| Type | Description | Formula |
|------|-------------|---------|
| **Periodic** | Pressure wraps around | $p(\text{ghost}) = p(\text{periodic partner})$ |
| **Neumann** | Zero normal gradient | $\partial p / \partial n = 0 \Rightarrow p(\text{ghost}) = p(\text{interior})$ |
| **Dirichlet** | Fixed pressure value | $p(\text{ghost}) = 2 p_{\text{bc}} - p(\text{interior})$ |

**Standard BC Configurations:**

| Configuration | x-direction | y-direction | z-direction | Use Case |
|---------------|-------------|-------------|-------------|----------|
| `channel()` | Periodic | Neumann | Periodic | Channel flow |
| `duct()` | Periodic | Neumann | Neumann | Square duct |
| `cavity()` | Neumann | Neumann | Neumann | Lid-driven cavity |
| `all_periodic()` | Periodic | Periodic | Periodic | Periodic box |

### Gauge Fixing

For problems with all Neumann or periodic pressure boundaries (no Dirichlet BC), the pressure is underdetermined up to a constant. The solver automatically:
1. Detects this condition via `has_nullspace()` check
2. Subtracts the mean pressure after each solve to fix the gauge

---

## Poisson Solvers

The solver provides **6 Poisson solver options** with automatic selection based on grid configuration and boundary conditions:

### Automatic Solver Selection Priority

```
FFT (3D) → FFT2D (2D) → FFT1D (3D partial-periodic) → HYPRE → Multigrid
```

### Available Solvers

| Solver | Complexity | Best For | Requirements |
|--------|------------|----------|--------------|
| **FFT** | O(N log N) | 3D channel flows | Periodic x AND z, uniform grid |
| **FFT2D** | O(N log N) | 2D channel flows | 2D mesh, periodic x |
| **FFT1D** | O(N log N) + 2D solve | 3D duct flows | Periodic x OR z (one only) |
| **HYPRE PFMG** | O(N) | Stretched grids, GPU | `USE_HYPRE` build flag |
| **Multigrid** | O(N) | General fallback | Always available |
| **SOR** | O(N²) | Testing/debugging | Always available |

### Geometric Multigrid (V-Cycle)

The default solver implements a geometric multigrid V-cycle:

1. **Pre-smooth**: Apply smoothing iterations on fine grid (Chebyshev or Jacobi)
2. **Restrict**: Compute residual and transfer to coarse grid (full weighting)
3. **Recurse**: Solve on coarse grid (recursively)
4. **Prolongate**: Interpolate correction back to fine grid (bilinear)
5. **Post-smooth**: Apply smoothing iterations

**Features:**
- O(N) complexity (optimal)
- 5-15 V-cycles to convergence (vs 1000-10000 SOR iterations)
- **CUDA Graph optimization**: Entire V-cycle captured as single GPU graph (NVHPC compilers)
- Chebyshev polynomial smoother (faster than Jacobi)

**Convergence Criteria** (any triggers exit):
- `tol_rhs`: RHS-relative $\|r\|/\|b\| < \epsilon$ (recommended for projection)
- `tol_rel`: Initial-residual relative $\|r\|/\|r_0\| < \epsilon$
- `tol_abs`: Absolute $\|r\|_\infty < \epsilon$

### FFT-Based Solvers

For problems with periodic boundaries, FFT solvers provide spectral accuracy:

- **FFT (3D)**: 2D FFT in x-z + batched tridiagonal solve in y (cuSPARSE)
- **FFT2D**: 1D FFT in x + batched tridiagonal in y
- **FFT1D**: 1D FFT in periodic direction + 2D Helmholtz solve per mode

### HYPRE PFMG

GPU-accelerated parallel semicoarsening multigrid from [HYPRE](https://github.com/hypre-space/hypre):

- Supports uniform AND stretched grids
- Entire solve runs on GPU via CUDA backend
- Automatic download and build via CMake FetchContent

See `docs/POISSON_SOLVER_GUIDE.md` for detailed documentation.

---

## Turbulence Closures

The solver supports **10 turbulence closure options**:

### Summary Table

| Model | Type | Equations | Anisotropic | GPU |
|-------|------|-----------|-------------|-----|
| `none` | Direct | 0 | N/A | Yes |
| `baseline` | Algebraic | 0 | No | Yes |
| `gep` | Algebraic | 0 | No | Yes |
| `komega` | Transport | 2 (k, ω) | No | Yes |
| `sst` | Transport | 2 (k, ω) | No | Yes |
| `earsm_wj` | EARSM | 2 (k, ω) | Yes | Yes |
| `earsm_gs` | EARSM | 2 (k, ω) | Yes | Yes |
| `earsm_pope` | EARSM | 2 (k, ω) | Yes | Yes |
| `nn_mlp` | Neural Net | 0 | No | Yes |
| `nn_tbnn` | Neural Net | 0 | Yes | Yes |

### Algebraic Models (Zero-Equation)

#### 1. Mixing Length Model (`baseline`)

Classical model with van Driest wall damping:

$$\nu_t = (\kappa y)^2 |\mathbf{S}| \left(1 - e^{-y^+/A^+}\right)^2$$

- $\kappa = 0.41$ (von Kármán constant)
- $A^+ \approx 26$ (van Driest damping constant)
- $|\mathbf{S}| = \sqrt{2S_{ij}S_{ij}}$ (strain rate magnitude)

#### 2. GEP Model (`gep`)

Symbolic regression formula discovered by genetic algorithms (Weatheritt & Sandberg 2016):

$$\nu_t = f_{\text{GEP}}(S_{ij}, \Omega_{ij}, y, \text{Re}_\tau, \ldots)$$

### Transport Equation Models (Two-Equation)

#### 3. SST k-ω (`sst`)

Menter's Shear Stress Transport model (1994):

**k-equation:**
$$\frac{\partial k}{\partial t} + \bar{u}_j \frac{\partial k}{\partial x_j} = P_k - \beta^* k \omega + \nabla \cdot [(\nu + \sigma_k \nu_t) \nabla k]$$

**ω-equation (with cross-diffusion):**
$$\frac{\partial \omega}{\partial t} + \bar{u}_j \frac{\partial \omega}{\partial x_j} = \alpha \frac{\omega}{k} P_k - \beta \omega^2 + \nabla \cdot [(\nu + \sigma_\omega \nu_t) \nabla \omega] + CD_\omega$$

**Eddy viscosity:**
$$\nu_t = \frac{a_1 k}{\max(a_1 \omega, S F_2)}$$

- Blending functions F₁, F₂ for k-ε/k-ω transition
- Production limiter for numerical stability
- Wall boundary conditions: k = 0, ω = ω_wall(y)

#### 4. Standard k-ω (`komega`)

Wilcox (1988) formulation without blending:

$$\nu_t = \frac{k}{\omega}$$

### EARSM Models (Explicit Algebraic Reynolds Stress)

EARSM models predict the full Reynolds stress anisotropy tensor using a tensor basis expansion:

$$b_{ij} = \sum_{n=1}^{N} G_n(\eta, \xi) \, T_{ij}^{(n)}(\mathbf{S}, \mathbf{\Omega})$$

where:
- $b_{ij}$ = anisotropy tensor (traceless)
- $T_{ij}^{(n)}$ = integrity basis tensors
- $G_n$ = scalar coefficient functions
- $\eta = Sk/\epsilon$, $\xi = \Omega k/\epsilon$ = normalized invariants

Combined with SST k-ω transport for k and ω evolution.

#### 5. Wallin-Johansson EARSM (`earsm_wj`)

Most sophisticated variant with cubic implicit equation for realizability.

#### 6. Gatski-Speziale EARSM (`earsm_gs`)

Quadratic model without implicit solve.

#### 7. Pope Quadratic EARSM (`earsm_pope`)

Classical weak-equilibrium model using first 3 basis tensors.

**Re_t-Based Blending:**

EARSM models use smooth blending between linear Boussinesq (laminar) and full nonlinear (turbulent):

$$\alpha(\text{Re}_t) = \frac{1}{2}\left(1 + \tanh\left(\frac{\text{Re}_t - \text{Re}_{t,\text{center}}}{\text{Re}_{t,\text{width}}}\right)\right)$$

where $\text{Re}_t = k/(\nu\omega)$. Default transition: center at Re_t = 10, width = 5.

### Neural Network Models

#### 8. MLP (`nn_mlp`)

Multi-layer perceptron for scalar eddy viscosity:

$$\nu_t = \text{NN}_{\text{MLP}}(\lambda_1, \ldots, \lambda_5, y/\delta)$$

**Inputs** (invariants of strain and rotation tensors):
- $\lambda_1 = S_{ij}S_{ij}$, $\lambda_2 = \Omega_{ij}\Omega_{ij}$, $\lambda_3 = S_{ij}S_{jk}S_{ki}$, ...
- $y/\delta$ = normalized wall distance

**Architecture:** 6 → 32 → 32 → 1 (ReLU activations)

#### 9. TBNN (`nn_tbnn`)

Tensor Basis Neural Network (Ling et al. 2016) for anisotropic Reynolds stresses:

$$b_{ij} = \sum_{n=1}^{10} g_n(\lambda_1, \ldots, \lambda_5) \, T_{ij}^{(n)}$$

**Architecture:** 5 → 64 → 64 → 64 → 10 (outputs one coefficient per basis tensor)

**Key Properties:**
- **Frame invariance**: Guaranteed by using invariant inputs + tensor basis
- **Realizability**: Enforced during training
- **Anisotropy**: Captures different normal stresses and off-diagonal components

---

## Supported Flow Configurations

### 2D Channel Flow

Pressure-driven flow between parallel plates.

| Configuration | BCs | Use Case |
|---------------|-----|----------|
| Poiseuille (laminar) | Periodic x, walls y | Analytical validation |
| Turbulent RANS | Periodic x, walls y | Model comparison |

**Analytical solution (Poiseuille):**
$$u(y) = -\frac{1}{2\nu}\frac{dp}{dx}(H^2/4 - y^2)$$

### 3D Square Duct Flow

Pressure-driven flow in square cross-section.

| Configuration | BCs | Use Case |
|---------------|-----|----------|
| Laminar duct | Periodic x, walls y and z | 3D solver validation |
| Turbulent duct | Periodic x, walls y and z | Secondary flow study |

### 3D Taylor-Green Vortex

Classic benchmark for unsteady flow and energy decay.

| Configuration | BCs | Use Case |
|---------------|-----|----------|
| Taylor-Green | All periodic | DNS verification, energy decay |

**Initial condition:**
$$u = \sin(x)\cos(y)\cos(z), \quad v = -\cos(x)\sin(y)\cos(z), \quad w = 0$$

**Energy decay (low Re):**
$$KE(t) = KE(0) \cdot e^{-2\nu t}$$

---

## Command-Line Options

### Grid

| Option | Description | Default |
|--------|-------------|---------|
| `--Nx N` | Grid cells in x | 64 |
| `--Ny N` | Grid cells in y | 64 |
| `--Nz N` | Grid cells in z (1 for 2D) | 1 |
| `--stretch` | Enable wall-normal stretching | off |

### Physics

| Option | Description | Default |
|--------|-------------|---------|
| `--Re VALUE` | Reynolds number | 1000 |
| `--nu VALUE` | Kinematic viscosity | 0.001 |
| `--dp_dx VALUE` | Pressure gradient | -1.0 |

**Note:** Specify only TWO of (Re, nu, dp_dx); the third is computed automatically.

### Turbulence Model

| Option | Description |
|--------|-------------|
| `--model TYPE` | Closure type: `none`, `baseline`, `gep`, `sst`, `komega`, `earsm_wj`, `earsm_gs`, `earsm_pope`, `nn_mlp`, `nn_tbnn` |
| `--nn_preset NAME` | Use NN model from `data/models/<NAME>` |

### Time Stepping

| Option | Description | Default |
|--------|-------------|---------|
| `--adaptive_dt` | Automatic CFL-based time step | on |
| `--dt VALUE` | Fixed time step | 0.001 |
| `--max_iter N` | Maximum iterations | 10000 |
| `--CFL VALUE` | Maximum CFL number | 0.5 |
| `--tol VALUE` | Convergence tolerance | 1e-6 |

### Numerical Schemes

| Option | Description | Default |
|--------|-------------|---------|
| `--convective_scheme TYPE` | Convection: `central`, `upwind` | central |
| `--poisson_solver TYPE` | Solver: `auto`, `fft`, `fft2d`, `fft1d`, `hypre`, `mg` | auto |

### Output

| Option | Description | Default |
|--------|-------------|---------|
| `--output DIR` | Output directory | ./output |
| `--num_snapshots N` | VTK snapshots during run | 10 |
| `--verbose` / `--quiet` | Verbosity control | verbose |

---

## GPU Acceleration

All solver components support GPU offload via OpenMP target directives.

### Build with GPU Support

```bash
# NVIDIA GPUs (NVHPC compiler)
CC=nvc CXX=nvc++ cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU_OFFLOAD=ON

# With HYPRE PFMG (fastest Poisson solver)
CC=nvc CXX=nvc++ cmake .. -DUSE_GPU_OFFLOAD=ON -DUSE_HYPRE=ON
```

### GPU-Accelerated Components

- Momentum equation (convection, diffusion)
- Pressure Poisson solver (multigrid V-cycles or HYPRE PFMG)
- Turbulence transport equations (k, ω)
- EARSM tensor basis computations
- Neural network inference

### CUDA Graph Optimization

On NVIDIA GPUs with NVHPC compiler, the multigrid V-cycle is captured as a **CUDA Graph**:
- Eliminates per-kernel launch overhead
- Single `cudaGraphLaunch()` replaces O(levels × kernels) launches
- Automatically recaptured if boundary conditions change

---

## Validation

### Analytical Benchmarks

| Test Case | Metric | Expected |
|-----------|--------|----------|
| Poiseuille flow | L2 error vs analytical | < 5% on 64×128 grid |
| Taylor-Green energy decay | Energy decay error | < 5% |

### Physics Conservation

| Property | Criterion |
|----------|-----------|
| Divergence-free | $\|\nabla \cdot \mathbf{u}\|_\infty < 10^{-10}$ |
| Momentum balance | Body force = wall shear (< 10% imbalance) |
| Channel symmetry | $u(y) = u(-y)$ (machine precision) |

### DNS Benchmarks

| Test Case | Reference |
|-----------|-----------|
| Channel Re_τ = 180, 395, 590 | Moser, Kim & Mansour (1999) |
| McConkey et al. dataset | Scientific Data 8, 255 (2021) |

---

## Training Neural Network Models

Train custom turbulence models on DNS/LES data:

```bash
# Setup environment
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Download dataset (~500 MB)
bash scripts/download_mcconkey_data.sh

# Train TBNN model
python scripts/train_tbnn_mcconkey.py \
    --data_dir mcconkey_data \
    --case channel \
    --output data/models/tbnn_channel \
    --epochs 100

# Use in solver
./channel --model nn_tbnn --nn_preset tbnn_channel
```

---

## References

### Numerical Methods

- Chorin, A. J. "Numerical solution of the Navier-Stokes equations." *Math. Comput.* 22.104 (1968): 745-762
- Briggs, W. L., Henson, V. E., & McCormick, S. F. *A Multigrid Tutorial*, 2nd ed. SIAM, 2000

### Turbulence Modeling

- Menter, F. R. "Two-equation eddy-viscosity turbulence models for engineering applications." *AIAA J.* 32.8 (1994): 1598-1605
- Wilcox, D. C. "Reassessment of the scale-determining equation for advanced turbulence models." *AIAA J.* 26.11 (1988): 1299-1310
- Wallin, S., & Johansson, A. V. "An explicit algebraic Reynolds stress model..." *J. Fluid Mech.* 403 (2000): 89-132
- Gatski, T. B., & Speziale, C. G. "On explicit algebraic stress models..." *J. Fluid Mech.* 254 (1993): 59-78
- Pope, S. B. "A more general effective-viscosity hypothesis." *J. Fluid Mech.* 72.2 (1975): 331-340

### Neural Network Closures

- Ling, J., Kurzawski, A., & Templeton, J. "Reynolds averaged turbulence modelling using deep neural networks with embedded invariance." *J. Fluid Mech.* 807 (2016): 155-166
- Weatheritt, J., & Sandberg, R. D. "A novel evolutionary algorithm applied to algebraic modifications of the RANS stress-strain relationship." *J. Comput. Phys.* 325 (2016): 22-37

### Dataset

- McConkey, R., et al. "A curated dataset for data-driven turbulence modelling." *Scientific Data* 8 (2021): 255

---

## License

MIT License - see `license` file
