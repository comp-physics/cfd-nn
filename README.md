# NN-CFD: Neural Network Turbulence Closures for Time-Accurate Incompressible Flow

![CI](https://github.com/comp-physics/cfd-nn/workflows/CI/badge.svg)

A **high-performance C++ solver** for **incompressible turbulence simulations** with **pluggable turbulence closures** ranging from classical algebraic models to advanced transport equations and data-driven neural networks. Features a fractional-step projection method with a multigrid Poisson solver, pure C++ NN inference, and comprehensive GPU acceleration.

The solver supports **10 turbulence closure options**: 3 algebraic (baseline, GEP), 2 transport (SST k-ω, k-ω), 3 EARSM variants (Wallin-Johansson, Gatski-Speziale, Pope), and 2 neural network closures (MLP, TBNN). All models support GPU offload via OpenMP target directives for NVIDIA and AMD GPUs, achieving 10-50x speedup on large grids.

## Features

- **Fractional-step projection method** for incompressible Navier-Stokes
  - Explicit Euler time integration with adaptive time stepping
  - **Multigrid Poisson solver** (O(N) complexity, 10-100x faster than SOR)
  - Pressure projection for divergence-free velocity
- **Pseudo-time marching to steady RANS** for canonical flows (channel, duct)
- **Multiple turbulence closures**:
  - **Algebraic models**: Mixing length, GEP symbolic regression
  - **Transport equation models**: SST k-ω, standard k-ω (Wilcox)
  - **EARSM models**: Wallin-Johansson, Gatski-Speziale, Pope Quadratic
  - **Neural networks**: Scalar eddy viscosity (MLP), Tensor Basis NN (TBNN - Ling et al. 2016)
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

# SST k-omega transport model
./channel --model sst --adaptive_dt

# EARSM (Wallin-Johansson)
./channel --model earsm_wj --adaptive_dt

# Neural network models
./channel --model nn_mlp --nn_preset mlp_channel_caseholdout --adaptive_dt  # Fast, GPU-optimized
./channel --model nn_tbnn --nn_preset tbnn_channel_caseholdout --adaptive_dt  # Physically consistent

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
    --case channel \
    --output data/models/tbnn_channel \
    --epochs 100

# Validate against DNS
python scripts/validate_trained_model.py \
    --model data/models/tbnn_channel \
    --test_data mcconkey_data/channel/test/data.npz \
    --plot

# Use in solver
cd build
./channel --model nn_tbnn --nn_preset tbnn_channel
```

## Command-Line Options

**Grid:**
- `--Nx N`, `--Ny N` - Grid cells (default: 64 x 64)
- `--stretch` - Enable y-direction stretching (default: off)

**Physics:**
- `--Re VALUE` - Reynolds number (default: 1000)
- `--nu VALUE` - Kinematic viscosity (default: 0.001)
- `--dp_dx VALUE` - Pressure gradient (driving force, default: -1.0)

Note: Specify ONLY TWO of (Re, nu, dp_dx); the third is computed automatically:
- `--Re` only → uses default dp_dx, computes nu
- `--Re --nu` → computes dp_dx to achieve desired Re
- `--Re --dp_dx` → computes nu to achieve desired Re
- `--nu --dp_dx` → computes Re from these
- none specified → uses defaults (Re=1000, nu=0.001, dp_dx=-1.0)

Specifying all three will error unless they are mutually consistent.

**Turbulence Model:**
- `--model TYPE` - Turbulence closure (default: none)
  - Algebraic: `none`, `baseline`, `gep`
  - Transport: `sst`, `komega`
  - EARSM: `earsm_wj`, `earsm_gs`, `earsm_pope`
  - Neural: `nn_mlp`, `nn_tbnn`
- `--nn_preset NAME` - Use NN model from data/models/<NAME>

**Time Stepping:**
- `--adaptive_dt` - Automatic time step (default: on)
- `--dt VALUE` - Fixed time step (default: 0.001)
- `--max_iter N` - Maximum iterations (default: 10000)
- `--CFL VALUE` - Max CFL number (default: 0.5)
- `--tol VALUE` - Convergence tolerance (default: 1e-6)

**Output:**
- `--output DIR` - Output directory (default: ./output)
- `--num_snapshots N` - Number of VTK snapshots (default: 10)
- `--verbose` / `--quiet` - Verbosity control (default: verbose)

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

## Available Turbulence Models

| Model | Type | Description | Speed | Accuracy |
|-------|------|-------------|-------|----------|
| `none` | Laminar | No turbulence model | ***** | N/A |
| `baseline` | Algebraic | Mixing length + van Driest damping | ***** | Moderate |
| `gep` | Algebraic | Gene Expression Programming (symbolic) | ***** | Good |
| `sst` | Transport (2-eq) | SST k-ω (Menter 1994) | *** | Very good |
| `komega` | Transport (2-eq) | Standard k-ω (Wilcox 1988) | *** | Good |
| `earsm_wj` | EARSM | Wallin-Johansson (2000) + SST transport | **** | Excellent |
| `earsm_gs` | EARSM | Gatski-Speziale (1993) + SST transport | **** | Very good |
| `earsm_pope` | EARSM | Pope Quadratic (1975) + SST transport | **** | Good |
| `nn_mlp` | Neural Net | Scalar eddy viscosity (data-driven) | ** | Data-driven |
| `nn_tbnn` | Neural Net | Anisotropic stress (Ling 2016) | * | Data-driven |

**Notes:**
- **Algebraic models** are fastest but limited to simple flows
- **Transport models** solve PDEs for k and ω, capturing history effects  
- **EARSM models** combine SST k-ω transport with explicit algebraic closures for anisotropic Reynolds stress
- **Neural models** require training data but can learn complex flow physics from DNS/LES data

## Mathematical Formulation

### Governing Equations

The solver implements the **incompressible Reynolds-Averaged Navier-Stokes (RANS) equations** for turbulent flow:

**Momentum equation:**

$$\frac{\partial \bar{u}_i}{\partial t} + \bar{u}_j \frac{\partial \bar{u}_i}{\partial x_j} = -\frac{1}{\rho} \frac{\partial \bar{p}}{\partial x_i} + \frac{\partial}{\partial x_j}\left[(\nu + \nu_t) \frac{\partial \bar{u}_i}{\partial x_j}\right] + f_i$$

**Continuity equation (incompressibility):**

$$\frac{\partial \bar{u}_i}{\partial x_i} = 0$$

where:
- $\bar{u}_i$ = mean velocity components
- $\bar{p}$ = mean pressure
- $\nu$ = kinematic viscosity
- $\nu_t$ = turbulent eddy viscosity (from turbulence model)
- $f_i$ = body force (e.g., pressure gradient)
- $\rho$ = density (constant)

### Fractional-Step Projection Method

The solver uses a **fractional-step method** (Chorin 1968) to decouple pressure and velocity:

**Step 1: Provisional velocity** (without pressure gradient)

$$\frac{\mathbf{u}^* - \mathbf{u}^n}{\Delta t} = -(\mathbf{u}^n \cdot \nabla)\mathbf{u}^n + \nabla \cdot [(\nu + \nu_t) \nabla \mathbf{u}^n] + \mathbf{f}$$

**Step 2: Pressure Poisson equation** (enforce incompressibility)

$$\nabla^2 p' = \frac{1}{\Delta t} \nabla \cdot \mathbf{u}^*$$

**Step 3: Velocity correction** (project onto divergence-free space)

$$\mathbf{u}^{n+1} = \mathbf{u}^* - \Delta t \nabla p'$$

This ensures $\nabla \cdot \mathbf{u}^{n+1} = 0$ to machine precision.

### Spatial Discretization

All spatial derivatives use **second-order central finite differences** on a staggered Cartesian grid:

**Gradient (central difference):**

$$\left.\frac{\partial u}{\partial x}\right|_{i,j} = \frac{u_{i+1,j} - u_{i-1,j}}{2\Delta x}$$

**Laplacian (5-point stencil):**

$$\left.\nabla^2 u\right|_{i,j} = \frac{u_{i+1,j} - 2u_{i,j} + u_{i-1,j}}{\Delta x^2} + \frac{u_{i,j+1} - 2u_{i,j} + u_{i,j-1}}{\Delta y^2}$$

**Convective term** (two schemes available):
- **Central:** $(\mathbf{u} \cdot \nabla)\mathbf{u}$ using central differences (default, more accurate)
- **Upwind:** First-order upwind for stability at high Reynolds numbers

**Diffusive term with variable viscosity:**

$$\nabla \cdot [(\nu + \nu_t) \nabla u] = \frac{1}{\Delta x^2}\left[\nu_{e}(u_{i+1,j} - u_{i,j}) - \nu_{w}(u_{i,j} - u_{i-1,j})\right] + \frac{1}{\Delta y^2}\left[\nu_{n}(u_{i,j+1} - u_{i,j}) - \nu_{s}(u_{i,j} - u_{i,j-1})\right]$$

where $\nu_e, \nu_w, \nu_n, \nu_s$ are face-averaged effective viscosities (e.g., $\nu_e = \frac{1}{2}(\nu_{i,j} + \nu_{i+1,j})$).

### Pressure Poisson Solver

The pressure equation is solved using one of several methods, automatically selected based on boundary conditions and grid configuration:

**Solver Selection Priority:** FFT (3D) → FFT2D (2D) → FFT1D (3D) → HYPRE → Multigrid

| Solver | Best For | Requirements |
|--------|----------|--------------|
| **FFT** | 3D channel flows | Periodic x AND z, uniform dx/dz |
| **FFT2D** | 2D channel flows | 2D mesh, periodic x |
| **FFT1D** | 3D duct flows | Periodic x OR z (one only) |
| **HYPRE** | Stretched grids | `USE_HYPRE` build |
| **Multigrid** | General fallback | Always available |

See `docs/POISSON_SOLVER_GUIDE.md` for detailed documentation on all solvers.

#### 1. FFT-Based Solvers (Fastest)

For periodic boundary conditions, the FFT solvers use cuFFT + cuSPARSE for direct solves:
- **FFT**: 2D FFT in x-z + batched tridiagonal in y (3D, periodic x AND z)
- **FFT2D**: 1D FFT in x + batched tridiagonal in y (2D mesh)
- **FFT1D**: 1D FFT in periodic direction + 2D Helmholtz per mode (3D duct)

Complexity: $\mathcal{O}(N \log N)$ - near-optimal for periodic problems.

#### 2. Successive Over-Relaxation (SOR) - Testing Only

Red-black Gauss-Seidel with relaxation parameter $\omega \in (1, 2)$:

$$p_{i,j}^{k+1} = (1-\omega) p_{i,j}^k + \omega \frac{\displaystyle\frac{p_{i+1,j}^k + p_{i-1,j}^k}{\Delta x^2} + \frac{p_{i,j+1}^k + p_{i,j-1}^k}{\Delta y^2} - f_{i,j}}{\displaystyle\frac{2}{\Delta x^2} + \frac{2}{\Delta y^2}}$$

- Complexity: $\mathcal{O}(N^2)$ for $N$ grid points
- Iterations to convergence: 1000-10000
- Used for small grids or validation

#### 3. Geometric Multigrid (V-Cycle)

**Algorithm:**
1. **Pre-smooth**: Apply $\nu_1$ SOR iterations on fine grid
2. **Restrict**: Compute residual $r = f - L_h p_h$ and restrict to coarse grid: $r_{2h} = I_{2h}^h r_h$
3. **Recurse**: Solve $L_{2h} e_{2h} = r_{2h}$ on coarse grid (recursively)
4. **Prolongate**: Interpolate correction back to fine grid: $e_h = I_h^{2h} e_{2h}$
5. **Correct**: $p_h \leftarrow p_h + e_h$
6. **Post-smooth**: Apply $\nu_2$ SOR iterations

**Restriction operator** (full weighting):

$$r_{2h}^{I,J} = \frac{1}{16}\begin{bmatrix}1 & 2 & 1 \\ 2 & 4 & 2 \\ 1 & 2 & 1\end{bmatrix} * r_h$$

**Prolongation operator** (bilinear interpolation):
- Coarse grid values copied directly
- Fine grid values interpolated from surrounding coarse cells

**Performance:**
- Complexity: $\mathcal{O}(N)$ - optimal!
- V-cycles to convergence: 5-15 (vs 1000-10000 SOR iterations)
- Speedup: **10-100x faster** than SOR for large grids

#### 4. HYPRE PFMG (GPU-Accelerated)

For maximum GPU performance, the solver supports [HYPRE](https://github.com/hypre-space/hypre)'s PFMG (Parallel Semicoarsening Multigrid) solver with CUDA backend:

**Performance (NVIDIA H200 GPU):**

| Grid Size | HYPRE PFMG | Built-in Multigrid | Speedup |
|-----------|------------|-------------------|---------|
| 32³       | 5.4 ms     | 52 ms             | **9.7x**  |
| 64³       | 5.4 ms     | 58 ms             | **10.8x** |
| 128³      | 8.0 ms     | 66 ms             | **8.2x**  |

**Key features:**
- Entire multigrid solve runs on GPU via CUDA kernels
- Automatic download and build via CMake FetchContent
- Unified memory integration with OpenMP target offload
- Weighted Jacobi smoothing (GPU-parallel)

See `docs/HYPRE_POISSON_SOLVER.md` for detailed documentation.

### Time Integration

**Explicit Euler** with adaptive time stepping:

$$\Delta t = \text{CFL} \cdot \min\left(\frac{\Delta x}{|u|_{\max}}, \frac{\Delta y}{|v|_{\max}}\right)$$

where CFL $\in (0, 1]$ is the Courant-Friedrichs-Lewy number (default 0.5).

**Pseudo-time marching** to steady state:
- Iterate until $\|\mathbf{u}^{n+1} - \mathbf{u}^n\| < \text{tol}$
- Convergence criterion: maximum velocity change per iteration
- Typical convergence: 1000-10000 iterations for $\text{tol} = 10^{-6}$

### Boundary Conditions

**Channel flow (default):**
- **Streamwise (x):** Periodic
- **Wall-normal (y):** No-slip walls ($u = v = 0$)
- **Pressure:** Neumann boundaries ($\partial p/\partial n = 0$)

**Periodic boundaries:**

$$u(0, y) = u(L_x, y), \quad p(0, y) = p(L_x, y)$$

**No-slip walls:**

$$u(x, 0) = 0, \quad v(x, 0) = 0$$

## Turbulence Closures

The RANS equations require a **turbulence closure** to model the eddy viscosity $\nu_t$ or Reynolds stresses $\tau_{ij}$.

### 1. Mixing Length Model (Baseline)

Classical **algebraic model** with van Driest wall damping:

$$\nu_t = (\kappa y)^2 |\mathbf{S}| \left(1 - e^{-y^+/A^+}\right)^2$$

where:
- $\kappa = 0.41$ = von Kármán constant
- $y$ = distance from wall
- $|\mathbf{S}| = \sqrt{2S_{ij}S_{ij}}$ = strain rate magnitude
- $S_{ij} = \frac{1}{2}\left(\frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i}\right)$ = strain rate tensor
- $y^+ = \frac{y u_\tau}{\nu}$ = wall units
- $A^+ \approx 26$ = van Driest damping constant

**Pros:** Fast, no additional PDEs, physically motivated  
**Cons:** Struggles with separated flows, adverse pressure gradients

### 2. GEP (Gene Expression Programming)

**Symbolic regression** model learned from DNS/LES data:

$$\nu_t = f_{\text{GEP}}(S_{ij}, \Omega_{ij}, y, \text{Re}_\tau, \ldots)$$

where $\Omega_{ij} = \frac{1}{2}\left(\frac{\partial u_i}{\partial x_j} - \frac{\partial u_j}{\partial x_i}\right)$ is the rotation rate tensor.

- Algebraic formula discovered by genetic algorithms
- Interpretable (human-readable equation)
- Faster than neural networks, more flexible than mixing length

**Example GEP formula** (Weatheritt & Sandberg 2016):

$$\nu_t = \nu \cdot \text{Re}_\tau \cdot \left[c_1 \frac{|\mathbf{S}|}{|\mathbf{\Omega}|} + c_2 \left(\frac{y}{\delta}\right)^2 + \ldots\right]$$

### 3. SST k-ω (Transport Equation Model)

**Two-equation model** solving transport PDEs for turbulent kinetic energy ($k$) and specific dissipation rate ($\omega$):

**k-equation:**

$$\frac{\partial k}{\partial t} + \bar{u}_j \frac{\partial k}{\partial x_j} = P_k - \beta^* k \omega + \frac{\partial}{\partial x_j}\left[\left(\nu + \sigma_k \nu_t\right) \frac{\partial k}{\partial x_j}\right]$$

**ω-equation (SST form with cross-diffusion):**

$$\frac{\partial \omega}{\partial t} + \bar{u}_j \frac{\partial \omega}{\partial x_j} = \alpha \frac{\omega}{k} P_k - \beta \omega^2 + \frac{\partial}{\partial x_j}\left[\left(\nu + \sigma_\omega \nu_t\right) \frac{\partial \omega}{\partial x_j}\right] + 2(1-F_1)\sigma_{\omega2} \frac{1}{\omega}\frac{\partial k}{\partial x_j}\frac{\partial \omega}{\partial x_j}$$

**Eddy viscosity:**

$$\nu_t = \frac{a_1 k}{\max(a_1 \omega, S F_2)}$$

where:
- $P_k = 2\nu_t S_{ij}S_{ij}$ = production of turbulent kinetic energy
- $F_1, F_2$ = blending functions (smooth transition between k-ε and k-ω behavior)
- $\alpha, \beta, \beta^*, \sigma_k, \sigma_\omega, a_1$ = model constants (blended between two sets)

**Implementation details:**
- Explicit Euler time integration for k and ω transport
- Central differences for diffusion terms
- Positivity constraints: $k \geq k_{\min}$, $\omega \geq \omega_{\min}$
- Wall boundary conditions: $k = 0$, $\omega = \omega_{\text{wall}}(y)$
- GPU-accelerated with OpenMP target offload

**Pros:** Captures transport/history effects, excellent for boundary layers and adverse pressure gradients  
**Cons:** More expensive than algebraic models, requires solving 2 extra PDEs

### 4. Standard k-ω (Wilcox 1988)

Simplified version of SST without blending functions:

**ω-equation (standard form, no cross-diffusion):**

$$\frac{\partial \omega}{\partial t} + \bar{u}_j \frac{\partial \omega}{\partial x_j} = \alpha \frac{\omega}{k} P_k - \beta \omega^2 + \frac{\partial}{\partial x_j}\left[\left(\nu + \sigma_\omega \nu_t\right) \frac{\partial \omega}{\partial x_j}\right]$$

**Eddy viscosity:**

$$\nu_t = \frac{k}{\omega}$$

Uses constant model coefficients (no blending). Simpler than SST but less accurate for complex flows.

### 5. EARSM (Explicit Algebraic Reynolds Stress Models)

**Anisotropic closures** that predict the full Reynolds stress tensor using algebraic expressions:

**Tensor basis expansion:**

$$b_{ij} = \sum_{n=1}^{10} G_n(\eta, \xi) \, T_{ij}^{(n)}(\mathbf{S}, \mathbf{\Omega})$$

where:
- $b_{ij} = \frac{\langle u_i' u_j' \rangle}{2k} - \frac{1}{3}\delta_{ij}$ = anisotropy tensor
- $T_{ij}^{(n)}$ = integrity basis tensors (same as TBNN)
- $G_n$ = algebraic coefficient functions (model-specific)
- $\eta = \frac{Sk}{\epsilon}$, $\xi = \frac{\Omega k}{\epsilon}$ = strain and rotation time-scale ratios

**Combined with transport equations:**

EARSM models in this code combine SST k-ω transport equations (for k and ω evolution) with explicit algebraic closures (for anisotropic Reynolds stresses). This gives both temporal accuracy and anisotropic stress predictions.

**Eddy viscosity (for use in momentum equation):**

$$\nu_t = C_\mu \frac{k^2}{\epsilon}$$

where $\epsilon = \beta^* k \omega$ (computed from k-ω model), and $C_\mu$ may be variable depending on the EARSM variant.

**Three EARSM variants implemented:**

#### a) Wallin-Johansson EARSM (2000)

Most sophisticated model with **cubic implicit equation** for parameter $N$:

$$N^3 + c_2 N^2 + c_1 N + c_0 = 0$$

where coefficients $c_0, c_1, c_2$ depend on invariants $\eta$ and $\xi$.

**Coefficient functions:**

$$G_n = f_n(N, \eta, \xi, \beta_1, \ldots, \beta_6)$$

with model constants $\beta_1 = 0.9$, $\beta_3 = 1.8$, $\beta_4 = 0.625$, $\beta_6 = 0.6$.

**Features:**
- Captures complex 3D turbulence
- Realizability enforced through cubic solution
- Variable $C_\mu$ improves accuracy

#### b) Gatski-Speziale EARSM (1993)

Simpler quadratic model:

$$G_n = f_n(\eta, \xi, c_1, c_2, c_3, c_4)$$

with constants $c_1 = 1.8$, $c_2 = 0.36$, $c_3 = 1.25$, $c_4 = 0.4$.

No implicit solve required. Good balance between complexity and accuracy.

#### c) Pope Quadratic EARSM (1975)

Classical weak-equilibrium model:

$$G_1 = -\frac{2C_1}{3 + 2C_1 S^2/\Omega^2}, \quad G_2 = -\frac{4C_1^2}{(3 + 2C_1 S^2/\Omega^2)^2}, \quad G_3 = -\frac{2C_1}{3 + 2C_1 S^2/\Omega^2}$$

with $C_1 = 1.8$. Only uses first 3 basis tensors (quadratic in $S_{ij}$, $\Omega_{ij}$).

**EARSM implementation:**
- Combined with SST k-ω transport for k and ω evolution
- Explicit algebraic closure for Reynolds stress anisotropy
- GPU-accelerated tensor computations
- Automatic realizability enforcement (positive k, bounded eigenvalues)

**EARSM stability and Re_t-based blending:**

EARSM closures compute nonlinear anisotropy corrections that are only physically meaningful when turbulence is actually present. The solver uses **smooth turbulence Reynolds number (Re_t) blending** to gracefully transition between linear Boussinesq and full nonlinear EARSM:

$$\text{Re}_t = \frac{k}{\nu \omega}$$

**Blending factor:**
$$\alpha(Re_t) = \frac{1}{2}\left(1 + \tanh\left(\frac{Re_t - Re_{t,\text{center}}}{Re_{t,\text{width}}}\right)\right)$$

- **α = 0**: Pure linear Boussinesq (laminar-like, Re_t ≪ 1)
- **α = 1**: Full nonlinear EARSM (turbulent, Re_t ≫ 1)
- **Default transition**: Centered at Re_t = 10, width = 5

**Physical reasoning:**
- The linear term (β₁ S*) in EARSM reduces to Boussinesq and is always active
- The nonlinear terms (commutator [S*, Ω*], quadratic S*²) capture anisotropy and are blended: `β_nonlinear *= α`
- When Re_t → 0 (laminar), nonlinear corrections vanish smoothly
- No hard switch → no discontinuities in residuals → better convergence

**Configurable parameters** (via `EARSMThresholds` struct):
- `Re_t_center = 10.0`: Center of blending transition (α = 0.5)
- `Re_t_width = 5.0`: Width controlling steepness
- `k_min = 1e-10`: Floor for k (denominator protection only)
- `omega_min = 1e-10`: Floor for ω (denominator protection only)
- `warn_low_turbulence = false`: Print diagnostic if Re_t < 1 globally

**To properly exercise EARSM (high α):**
- Use driven flows (non-zero body force) to maintain turbulence
- Ensure Re_t > 20 in regions of interest (→ α ≈ 0.99)
- Typical values: Re_t ~ 100-1000 in fully turbulent channel flow
- For decay tests, initialize with sufficient k and low ν

**Pros:** Full anisotropic stress with only algebraic closures, physics-based, more accurate than linear eddy viscosity  
**Cons:** More computationally expensive than pure algebraic models, slightly more complex than standard Boussinesq

### 6. MLP (Multi-Layer Perceptron)

**Neural network** for scalar eddy viscosity:

$$\nu_t = \text{NN}_{\text{MLP}}(\lambda_1, \lambda_2, \lambda_3, \lambda_4, \lambda_5, y/\delta)$$

**Inputs** (invariants of $S_{ij}$ and $\Omega_{ij}$):
- $\lambda_1 = S_{ij}S_{ij}$ - strain magnitude squared
- $\lambda_2 = \Omega_{ij}\Omega_{ij}$ - rotation magnitude squared
- $\lambda_3 = S_{ij}S_{jk}S_{ki}$ - strain triple product
- $\lambda_4 = \Omega_{ij}\Omega_{jk}S_{ki}$ - mixed invariant
- $\lambda_5 = \Omega_{ij}\Omega_{jk}\Omega_{kl}S_{li}$ - higher-order invariant
- $y/\delta$ - normalized wall distance

**Architecture:** 6 → 32 → 32 → 1 (fully connected, ReLU activations)

**Training:** Supervised learning on DNS/LES data to match $\nu_t = -\langle u'v' \rangle / (\partial \bar{u}/\partial y)$

### 7. TBNN (Tensor Basis Neural Network)

Predicts full **Reynolds stress anisotropy tensor** $b_{ij}$:

$$b_{ij} = \frac{\langle u_i' u_j' \rangle}{\langle u_k' u_k' \rangle} - \frac{1}{3}\delta_{ij}$$

**Tensor basis expansion** (Ling et al. 2016):

$$b_{ij} = \sum_{n=1}^{10} g_n(\lambda_1, \ldots, \lambda_5) \, T_{ij}^{(n)}(\mathbf{S}, \mathbf{\Omega})$$

where $T_{ij}^{(n)}$ are the **integrity basis tensors** (symmetric, traceless):

$$\begin{aligned}
T_{ij}^{(1)} &= S_{ij} \\
T_{ij}^{(2)} &= S_{ik}\Omega_{kj} - \Omega_{ik}S_{kj} \\
T_{ij}^{(3)} &= S_{ik}S_{kj} - \frac{1}{3}S_{mn}S_{mn}\delta_{ij} \\
&\vdots \\
T_{ij}^{(10)} &= \Omega_{ik}\Omega_{kl}S_{lm}S_{mj} - S_{ik}S_{kl}\Omega_{lm}\Omega_{mj}
\end{aligned}$$

**Neural network:**

$$g_n = \text{NN}_{\text{TBNN}}(\lambda_1, \lambda_2, \lambda_3, \lambda_4, \lambda_5) \quad \text{for } n = 1, \ldots, 10$$

- **Architecture:** 5 → 64 → 64 → 64 → 10 (outputs one coefficient per basis tensor)
- **Frame invariance:** Guaranteed by using invariant inputs $\lambda_i$ and tensor basis
- **Realizability:** Enforced during training (positive turbulent kinetic energy, Schwarz inequality)

**Reconstruction of Reynolds stresses:**

$$\langle u_i' u_j' \rangle = \frac{2}{3} k \left(\delta_{ij} + b_{ij}\right)$$

where $k = \frac{1}{2}\langle u_i' u_i' \rangle$ is the turbulent kinetic energy.

**Key differences from eddy viscosity models:**
- Captures **anisotropy** (different normal stresses, nonzero shear stresses)
- Mixing length and MLP assume isotropic eddy viscosity: $\langle u_i' u_j' \rangle = \frac{2}{3}k\delta_{ij} - \nu_t S_{ij}$
- TBNN allows complex stress distributions learned from DNS/LES
- **Trade-off:** More expensive to evaluate (210x slower than laminar) but may improve accuracy for complex flows


## McConkey Dataset

This project integrates with the **McConkey et al. (2021)** dataset:

- **Reference**: *Scientific Data* 8, 255 (2021)
- **Content**: RANS + DNS/LES data for multiple flow cases
- **Features**: Pre-computed TBNN invariants and tensor basis
- **Cases**: Channel flow, square duct
- **Download**: `bash scripts/download_mcconkey_data.sh`


## GPU Acceleration

All turbulence models support **GPU offload** via OpenMP target directives:

**Build with GPU support:**
```bash
mkdir build && cd build
CC=nvc CXX=nvc++ cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_GPU_OFFLOAD=ON
make -j8
```

**Build with GPU + HYPRE PFMG (fastest Poisson solver):**
```bash
mkdir build_hypre && cd build_hypre
CC=nvc CXX=nvc++ cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_GPU_OFFLOAD=ON \
    -DUSE_HYPRE=ON
make -j8

# Run with HYPRE solver
./channel --poisson_solver hypre --Nx 64 --Ny 128 --Nz 64 --model baseline
```

HYPRE is automatically downloaded and built by CMake. See `docs/HYPRE_POISSON_SOLVER.md` for details.

**GPU-accelerated components:**
- Momentum equation (advection, diffusion)
- Pressure Poisson solver (multigrid V-cycles, or HYPRE PFMG with 8-10x speedup)
- Turbulence transport equations (k, ω)
- EARSM tensor basis computations
- Feature invariant calculations

## Validation

The solver is validated against both **analytical solutions** and **fundamental physics principles**.

### Physics Conservation Tests

The comprehensive test suite (`tests/test_physics_validation.cpp`) verifies the solver obeys fundamental conservation laws and produces physically correct results:

**1. Poiseuille Flow (Analytical Comparison):**
- Tests viscous diffusion and pressure gradient balance
- Compares to exact parabolic velocity profile
- Pass criterion: L2 error < 5% on 64×128 grid
- Result: ~2% error achieved

**2. Divergence-Free Constraint:**
- Verifies $\nabla \cdot \mathbf{u} = 0$ (incompressibility)
- Pass criterion: Machine precision (< 1e-10)
- Result: Maximum divergence ~1e-12

**3. Momentum Balance (Integral Check):**
- Verifies $\int f_{\text{body}} \, dV = \int \tau_{\text{wall}} \, dA$
- Checks global momentum conservation
- Pass criterion: Imbalance < 10%
- Result: ~5% imbalance

**4. Channel Symmetry:**
- Verifies $u(y) = u(-y)$ about centerline
- Tests boundary condition and discretization correctness
- Pass criterion: Machine precision
- Result: Perfect symmetry

**5. Cross-Model Consistency:**
- All turbulence models should agree in laminar limit
- Pass criterion: < 5% difference between models
- Result: All models produce consistent laminar results

**6. Sanity Checks:**
- No NaN/Inf in solution
- Realizability: $\nu_t \geq 0$, $k \geq 0$, $\omega \geq 0$
- All checks pass

**Taylor-Green Vortex Test** (`tests/test_taylor_green.cpp`):
- Exact solution: $u = \sin(x)\cos(y)e^{-2\nu t}$
- Theory: Kinetic energy decays as $KE(t) = KE(0) \cdot e^{-4\nu t}$
- Pass criterion: < 5% error in energy decay
- Result: 0.5% error over 100 time steps (excellent!)

### Analytical Benchmarks

**Laminar channel flow** (Poiseuille solution):

$$u(y) = -\frac{1}{2\nu}\frac{dp}{dx}(1 - y^2), \quad v(y) = 0$$

- **L2 error:** ~2% on 64×128 grid (laminar Re=180)
- **Maximum error:** < 5% at centerline
- See `docs/VALIDATION.md` for detailed convergence studies

### Turbulence Model Validation

**Turbulent flows** compared against:
- **DNS databases:** Moser et al. channel flow DNS
  - Channel flow at Re_τ = 180, 395, 590
  - Mean velocity profiles, Reynolds stresses
- **McConkey et al. (2021) dataset:**
  - Periodic hills at Re = 10,595
  - Multiple flow configurations for training
- **Published results:** Ling et al. (2016) TBNN methodology
  - Framework for learning anisotropy from data
  - Outperforms standard RANS models on test cases

**Example validation metrics (Re_τ = 180):**
- SST k-ω: Mean velocity profile within 10% of DNS
- Baseline mixing length: Good agreement in log layer
- EARSM models: Improved predictions of secondary flows
- Neural models: Data-driven accuracy depends on training quality

## Code Architecture

The solver is designed with **modularity** and **extensibility** in mind:

### Turbulence Model Hierarchy

All turbulence models inherit from a common base class:

```cpp
class TurbulenceModel {
public:
    // Core interface
    virtual void update(const Mesh& mesh, const Field& u, const Field& v) = 0;
    virtual const Field& get_nu_t() const = 0;
    
    // For transport models (k-ω, SST)
    virtual bool uses_transport_equations() const { return false; }
    virtual void advance_turbulence(...) { }
    
    // For anisotropic models (EARSM, TBNN)
    virtual bool provides_anisotropy() const { return false; }
    virtual const Field& get_anisotropy_component(int i, int j) const;
};
```

**Concrete implementations:**
- `MixingLengthModel` - Algebraic eddy viscosity
- `GEPModel` - Symbolic regression
- `SSTKOmegaTransport` - SST k-ω with transport equations
- `KOmegaTransport` - Standard k-ω
- `SSTWithEARSM` - Composite: SST transport + EARSM closure
- `NNMLPModel` - Neural network (scalar)
- `NNTBNNModel` - Neural network (tensor)

### Factory Pattern for Model Selection

Models are created via a factory function:

```cpp
std::unique_ptr<TurbulenceModel> create_turbulence_model(
    TurbulenceModelType type, const Mesh& mesh, const Config& config);
```

This allows **easy addition of new models** without modifying the main solver.

### GPU Offload Strategy

GPU acceleration uses **persistent device buffers** to minimize data transfer:

```cpp
class OmpDeviceBuffer<T> {
    // RAII wrapper for GPU memory
    T* device_ptr_;
    size_t size_;
public:
    void allocate();
    void upload(const T* host_data);
    void download(T* host_data);
    T* device_ptr() const;
};
```

**Offloaded kernels:**
- Advection, diffusion operators
- Turbulence transport step (k, ω)
- EARSM coefficient computation
- Feature invariant calculations

Data stays on GPU between time steps; only final results are copied back to CPU.

## References

### Numerical Methods

**Fractional-step method:**
- Chorin, A. J. "Numerical solution of the Navier-Stokes equations." *Mathematics of Computation* 22.104 (1968): 745-762

**Multigrid methods:**
- Briggs, W. L., Henson, V. E., & McCormick, S. F. *A Multigrid Tutorial*, 2nd ed. SIAM, 2000

### Turbulence Modeling

**Transport Equation Models:**
- Menter, F. R. "Two-equation eddy-viscosity turbulence models for engineering applications." *AIAA Journal* 32.8 (1994): 1598-1605 (SST k-ω)
- Wilcox, D. C. "Reassessment of the scale-determining equation for advanced turbulence models." *AIAA Journal* 26.11 (1988): 1299-1310 (k-ω)

**EARSM (Explicit Algebraic Reynolds Stress Models):**
- Wallin, S., & Johansson, A. V. "An explicit algebraic Reynolds stress model for incompressible and compressible turbulent flows." *Journal of Fluid Mechanics* 403 (2000): 89-132 (Wallin-Johansson)
- Gatski, T. B., & Speziale, C. G. "On explicit algebraic stress models for complex turbulent flows." *Journal of Fluid Mechanics* 254 (1993): 59-78 (Gatski-Speziale)
- Pope, S. B. "A more general effective-viscosity hypothesis." *Journal of Fluid Mechanics* 72.2 (1975): 331-340 (Pope Quadratic)

**Neural Network Closures:**
- Ling, J., Kurzawski, A., & Templeton, J. "Reynolds averaged turbulence modelling using deep neural networks with embedded invariance." *Journal of Fluid Mechanics* 807 (2016): 155-166 (TBNN)

**GEP Symbolic Regression:**
- Weatheritt, J., & Sandberg, R. D. "A novel evolutionary algorithm applied to algebraic modifications of the RANS stress–strain relationship." *Journal of Computational Physics* 325 (2016): 22-37

**Dataset:**
- McConkey, R., et al. "A curated dataset for data-driven turbulence modelling." *Scientific Data* 8 (2021): 255

### BibTeX

```bibtex
@article{chorin1968numerical,
  title={Numerical solution of the Navier-Stokes equations},
  author={Chorin, Alexandre Joel},
  journal={Mathematics of Computation},
  volume={22},
  number={104},
  pages={745--762},
  year={1968}
}

@article{ling2016reynolds,
  title={Reynolds averaged turbulence modelling using deep neural networks with embedded invariance},
  author={Ling, Julia and Kurzawski, Andrew and Templeton, Jeremy},
  journal={Journal of Fluid Mechanics},
  volume={807},
  pages={155--166},
  year={2016},
  publisher={Cambridge University Press}
}

@article{weatheritt2016novel,
  title={A novel evolutionary algorithm applied to algebraic modifications of the RANS stress--strain relationship},
  author={Weatheritt, Jack and Sandberg, Richard D},
  journal={Journal of Computational Physics},
  volume={325},
  pages={22--37},
  year={2016},
  publisher={Elsevier}
}

@article{mcconkey2021curated,
  title={A curated dataset for data-driven turbulence modelling},
  author={McConkey, Ryley and Yee, Eugene and Lien, Fue-Sang},
  journal={Scientific Data},
  volume={8},
  number={1},
  pages={255},
  year={2021},
  publisher={Nature Publishing Group}
}
```

## License

MIT License - see `license` file
