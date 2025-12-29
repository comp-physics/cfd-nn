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

```bash
# Default run (16x32x32 grid)
./run.sh

# Custom grid
NX=32 NY=64 NZ=64 ./run.sh

# More iterations for convergence
MAX_ITER=50000 ./run.sh
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| NX | 16 | Grid cells in x (streamwise) |
| NY | 32 | Grid cells in y |
| NZ | 32 | Grid cells in z |
| MAX_ITER | 10000 | Maximum solver iterations |
| NU | 0.01 | Kinematic viscosity |

## Output

- `duct_final.vtk` - Final velocity field (VTK format)
- `duct_profile.dat` - Velocity profile at duct center

## Validation

The simulation compares the computed centerline velocity against the analytical solution. For well-resolved simulations (NY, NZ ≥ 32), the error should be < 5%.

## Visualization

```bash
# ParaView
paraview output/duct_final.vtk

# Plot velocity profile (gnuplot)
gnuplot -e "splot 'output/duct_profile.dat' u 1:2:3 w pm3d"
```
