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

## Configuration Reference

All parameters can be set via command-line arguments (`--param value`) or config file (key-value pairs). Command-line arguments override config file values.

### Domain and Mesh

| Parameter | CLI | Default | Description |
|-----------|-----|---------|-------------|
| `Nx` | `--Nx` | 64 | Grid cells in x-direction |
| `Ny` | `--Ny` | 64 | Grid cells in y-direction |
| `Nz` | `--Nz` | 1 | Grid cells in z-direction (1 = 2D simulation) |
| `x_min` | - | 0.0 | Domain minimum in x |
| `x_max` | - | 2π | Domain maximum in x |
| `y_min` | - | -1.0 | Domain minimum in y |
| `y_max` | - | 1.0 | Domain maximum in y |
| `z_min` | `--z_min` | 0.0 | Domain minimum in z |
| `z_max` | `--z_max` | 1.0 | Domain maximum in z |
| `stretch_y` | `--stretch` | false | Enable tanh stretching in y (clusters points near walls) |
| `stretch_beta` | - | 2.0 | Y-stretching parameter (higher = more clustering) |
| `stretch_z` | `--stretch_z` | false | Enable tanh stretching in z (3D only) |
| `stretch_beta_z` | `--stretch_beta_z` | 2.0 | Z-stretching parameter |

### Physics Parameters

| Parameter | CLI | Default | Description |
|-----------|-----|---------|-------------|
| `Re` | `--Re` | 1000.0 | Reynolds number |
| `nu` | `--nu` | 0.001 | Kinematic viscosity |
| `dp_dx` | `--dp_dx` | -1.0 | Pressure gradient (body force driving flow) |
| `rho` | - | 1.0 | Density (constant for incompressible) |

#### Auto-Computation of Physics Parameters

The solver uses the relationship: $\text{Re} = \frac{-dp/dx \cdot \delta^3}{3\nu^2}$ where $\delta$ is the channel half-height.

**You should specify only TWO of (Re, nu, dp_dx)**. The third is computed automatically:

| Specified | Computed | Use Case |
|-----------|----------|----------|
| `--Re` only | nu (using default dp_dx=-1) | Quick setup at desired Re |
| `--Re --nu` | dp_dx | Control both Re and viscosity |
| `--Re --dp_dx` | nu | Control Re and driving force |
| `--nu --dp_dx` | Re | Specify physical parameters directly |
| None | Re (from defaults) | Uses nu=0.001, dp_dx=-1.0 → Re≈1000 |

**If all three are specified**, the solver checks consistency and errors if they don't match (within 1% tolerance).

### Time Stepping

| Parameter | CLI | Default | Description |
|-----------|-----|---------|-------------|
| `dt` | `--dt` | 0.001 | Time step size (when not using adaptive) |
| `adaptive_dt` | `--adaptive_dt` | true | Enable CFL-based adaptive time stepping |
| `CFL_max` | `--CFL` | 0.5 | Maximum CFL number for adaptive dt |
| `max_iter` | `--max_iter` | 10000 | Maximum iterations (steady) or time steps (unsteady) |
| `T_final` | - | -1.0 | Final simulation time (-1 = use max_iter instead) |
| `tol` | `--tol` | 1e-6 | Convergence tolerance for steady-state |

**When adaptive_dt is enabled** (default), the time step is computed each iteration as:
$$\Delta t = \text{CFL} \cdot \min\left(\frac{\Delta x}{|u|_{\max}}, \frac{\Delta y}{|v|_{\max}}, \frac{\Delta z}{|w|_{\max}}\right)$$

### Simulation Mode

| Parameter | CLI | Default | Description |
|-----------|-----|---------|-------------|
| `simulation_mode` | `--simulation_mode` | `steady` | `steady` or `unsteady` |
| `perturbation_amplitude` | `--perturbation_amplitude` | 0.01 | Initial perturbation amplitude for DNS |

- **Steady mode**: Iterates until $\|\mathbf{u}^{n+1} - \mathbf{u}^n\|_\infty < \text{tol}$ or max_iter reached
- **Unsteady mode**: Runs exactly max_iter time steps (or until T_final)

### Numerical Schemes

| Parameter | CLI | Default | Description |
|-----------|-----|---------|-------------|
| `convective_scheme` | `--scheme` | `central` | `central` (2nd-order) or `upwind` (1st-order, more stable) |

### Turbulence Model

| Parameter | CLI | Default | Description |
|-----------|-----|---------|-------------|
| `turb_model` | `--model` | `none` | Turbulence closure (see table below) |
| `nu_t_max` | - | 1.0 | Maximum eddy viscosity (clipping) |
| `nn_preset` | `--nn_preset` | - | NN model preset name (loads from `data/models/<NAME>/`) |
| `nn_weights_path` | `--weights` | - | Custom NN weights directory |
| `nn_scaling_path` | `--scaling` | - | Custom NN scaling directory |

**Available turbulence models:**

| `--model` value | Description |
|-----------------|-------------|
| `none` | Laminar (no turbulence model) |
| `baseline` | Algebraic mixing length with van Driest damping |
| `gep` | Gene Expression Programming (Weatheritt-Sandberg 2016) |
| `sst` | SST k-ω transport model (Menter 1994) |
| `komega` | Standard k-ω (Wilcox 1988) |
| `earsm_wj` | SST k-ω + Wallin-Johansson EARSM |
| `earsm_gs` | SST k-ω + Gatski-Speziale EARSM |
| `earsm_pope` | SST k-ω + Pope quadratic EARSM |
| `nn_mlp` | Neural network scalar eddy viscosity (requires `--nn_preset`) |
| `nn_tbnn` | Tensor Basis NN anisotropy model (requires `--nn_preset`) |

**For NN models**, you must specify either:
- `--nn_preset NAME` (loads from `data/models/<NAME>/`), or
- `--weights DIR --scaling DIR` (explicit paths)

Available presets: `tbnn_channel_caseholdout`, `tbnn_phll_caseholdout`, `example_tbnn`, `example_scalar_nut`

### Poisson Solver

| Parameter | CLI | Default | Description |
|-----------|-----|---------|-------------|
| `poisson_solver` | `--poisson` | `auto` | Solver selection (see table below) |
| `poisson_tol` | `--poisson_tol` | 1e-6 | Legacy absolute tolerance (deprecated) |
| `poisson_max_iter` | `--poisson_max_iter` | 20 | Maximum V-cycles per solve |
| `poisson_omega` | - | 1.8 | SOR relaxation parameter (1 < ω < 2) |
| `poisson_abs_tol_floor` | `--poisson_abs_tol_floor` | 1e-8 | Absolute tolerance floor |

**Poisson solver options:**

| `--poisson` value | Description | Requirements |
|-------------------|-------------|--------------|
| `auto` | Auto-select best solver | (default) |
| `fft` | 2D FFT in x-z + tridiagonal in y | 3D, periodic x AND z, uniform grid |
| `fft2d` | 1D FFT in x + tridiagonal in y | 2D only (Nz=1), periodic x |
| `fft1d` | 1D FFT + 2D Helmholtz per mode | 3D, periodic x OR z (one only) |
| `hypre` | HYPRE PFMG GPU-accelerated | Requires `USE_HYPRE` build |
| `mg` | Native geometric multigrid | Always available |

**Auto-selection priority:** FFT → FFT2D → FFT1D → HYPRE → MG

#### Advanced Multigrid Settings

| Parameter | CLI | Default | Description |
|-----------|-----|---------|-------------|
| `poisson_tol_abs` | - | 0.0 | Absolute tolerance on ‖r‖ (0 = disabled) |
| `poisson_tol_rhs` | - | 1e-3 | RHS-relative: ‖r‖/‖b‖ (recommended) |
| `poisson_tol_rel` | - | 1e-3 | Initial-residual relative: ‖r‖/‖r₀‖ |
| `poisson_check_interval` | - | 1 | Check convergence every N V-cycles |
| `poisson_use_l2_norm` | - | true | Use L2 norm (smoother than L∞) |
| `poisson_linf_safety` | - | 10.0 | L∞ safety cap multiplier |
| `poisson_fixed_cycles` | - | 8 | Fixed V-cycle count (0 = convergence-based) |
| `poisson_adaptive_cycles` | - | false | Enable adaptive checking in fixed-cycle mode |
| `poisson_check_after` | - | 4 | Check residual after this many cycles |
| `poisson_nu1` | - | 0 | Pre-smoothing sweeps (0 = auto: 3 for walls) |
| `poisson_nu2` | - | 0 | Post-smoothing sweeps (0 = auto: 1) |
| `poisson_chebyshev_degree` | - | 4 | Chebyshev polynomial degree (3-4 typical) |
| `poisson_use_vcycle_graph` | - | true | Enable CUDA Graph for V-cycle (GPU only) |

**Convergence criteria** (any triggers exit):
- `tol_rhs`: ‖r‖/‖b‖ < ε (recommended for projection)
- `tol_rel`: ‖r‖/‖r₀‖ < ε
- `tol_abs`: ‖r‖ < ε

### Output

| Parameter | CLI | Default | Description |
|-----------|-----|---------|-------------|
| `output_dir` | `--output` | `output/` | Output directory for VTK files |
| `output_freq` | - | 100 | Console output frequency (iterations) |
| `num_snapshots` | `--num_snapshots` | 10 | Number of VTK snapshots during simulation |
| `verbose` | `--verbose` | true | Enable verbose output |
| `postprocess` | `--no_postprocess` | true | Enable Poiseuille table + profile output |
| `write_fields` | `--no_write_fields` | true | Enable VTK/field output |

### Performance and Diagnostics

| Parameter | CLI | Default | Description |
|-----------|-----|---------|-------------|
| `warmup_iter` | `--warmup_iter` | 0 | Iterations to run before timing (excluded from benchmarks) |
| `turb_guard_enabled` | `--turb_guard_enabled` | true | Enable NaN/Inf guard checks |
| `turb_guard_interval` | `--turb_guard_interval` | 5 | Check for NaN/Inf every N iterations |

### Benchmark Mode

The `--benchmark` flag configures the solver for performance timing with minimal overhead:

```bash
# Run benchmark with defaults (192^3 grid, 20 iterations)
./duct --benchmark

# Override grid size
./duct --benchmark --Nx 256 --Ny 256 --Nz 256

# Override iteration count
./duct --benchmark --max_iter 100
```

**Benchmark mode sets these defaults** (all can be overridden by subsequent flags):

| Setting | Value | Rationale |
|---------|-------|-----------|
| Grid size | 192 × 192 × 192 | Large enough for meaningful timing |
| Domain | 3D duct (periodic x, walls y/z) | Representative wall-bounded flow |
| `verbose` | false | No console output |
| `postprocess` | false | No profile analysis |
| `write_fields` | false | No VTK output |
| `num_snapshots` | 0 | No intermediate snapshots |
| `convective_scheme` | upwind | First-order upwind |
| `poisson_fixed_cycles` | 1 | Single V-cycle per time step |
| `turb_model` | none | No turbulence model |
| `max_iter` | 20 | Default iteration count |
| `adaptive_dt` | false | Fixed time step (dt=0.001) |

### Config File Format

Config files use simple key-value syntax:

```ini
# Comment lines start with #
Nx = 128
Ny = 256
Re = 5000
turb_model = sst
adaptive_dt = true
```

Load a config file with `--config FILE`. Command-line arguments override config file values.

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
