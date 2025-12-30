# 3D Square Duct Flow

This example demonstrates 3D laminar flow in a square duct, a classic validation case for 3D incompressible flow solvers.

## Physics

- **Geometry**: Square duct with side length 2 (y, z ∈ [-1, 1])
- **Boundary Conditions**:
  - Periodic in x (streamwise direction)
  - No-slip walls on all four sides (y and z boundaries)
- **Driving**: Constant body force (pressure gradient) in x-direction
- **Flow**: Develops to steady laminar Poiseuille-like profile

## Analytical Solution

For laminar flow in a square duct, the velocity profile is given by a double Fourier series:

```
u(y,z) = Σ Σ A_mn * sin(mπ(y+a)/2a) * sin(nπ(z+a)/2a)
```

where `a` is the half-width. The maximum velocity occurs at the center and is approximately:

```
u_max ≈ 0.295 * (-dp/dx) * a² / μ
```

## Running

### Using run.sh wrapper

```bash
# Laminar flow, coarse grid (default)
./run.sh laminar_square

# Laminar flow, fine grid
./run.sh laminar_fine

# Turbulent flow with SST k-omega
./run.sh turbulent_sst
```

### Running the binary directly

```bash
cd build

# With config file
./duct --config ../examples/08_duct_flow/laminar_square.cfg

# Override parameters from command line
./duct --config ../examples/08_duct_flow/laminar_square.cfg --Ny 64 --Nz 64
```

## Available Configurations

| Config | Grid | Description |
|--------|------|-------------|
| `laminar_square.cfg` | 16×32×32 | Coarse grid laminar validation |
| `laminar_fine.cfg` | 32×64×64 | Fine grid laminar (higher accuracy) |
| `turbulent_sst.cfg` | 32×64×64 | Turbulent flow with SST k-ω model |

## Configuration Parameters

Key parameters in the `.cfg` files:

```
Nx = 16             # Grid cells in x (streamwise)
Ny = 32             # Grid cells in y
Nz = 32             # Grid cells in z
nu = 0.01           # Kinematic viscosity
dp_dx = -1.0        # Pressure gradient
turb_model = none   # Turbulence model (none, sst, baseline, etc.)
```

## Output

- `duct_final.vtk` - Final velocity field (VTK format)
- `duct_profile.dat` - Velocity profile at duct center

## Validation

The simulation compares the computed centerline velocity against the analytical solution. For well-resolved simulations (NY, NZ ≥ 32), the error should be < 5%.

## Visualization

```bash
# ParaView
paraview output/laminar_square/duct_final.vtk

# Plot velocity profile (gnuplot)
gnuplot -e "splot 'output/laminar_square/duct_profile.dat' u 1:2:3 w pm3d"
```
