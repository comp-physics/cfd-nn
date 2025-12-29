# 3D Taylor-Green Vortex

The Taylor-Green vortex is a classic validation case for incompressible Navier-Stokes solvers, featuring analytical initial conditions and known decay behavior.

## Physics

- **Domain**: [0, 2π]³ with periodic boundaries
- **Initial condition**:
  ```
  u =  V₀ sin(x) cos(y) cos(z)
  v = -V₀ cos(x) sin(y) cos(z)
  w = 0
  ```
- **Behavior**:
  - Low Re (≤100): Viscous decay, KE ∝ exp(-2νt)
  - High Re (>1000): Vortex stretching → turbulence transition

## Validation

For low Reynolds numbers, the kinetic energy decays exponentially:

```
KE(t) = KE(0) * exp(-2νt)
```

The simulation tracks this decay and compares against theory.

## Running

### Using run.sh wrapper

```bash
# Re=100 on 32³ grid (default)
./run.sh tg_re100

# Re=100 on 64³ grid (higher accuracy)
./run.sh tg_re100_fine

# Re=1600 DNS on 64³ (turbulence transition)
./run.sh tg_re1600

# Override parameters
./run.sh tg_re100 --Re 200 --T 20.0
```

### Running the binary directly

```bash
cd build

# With config file
./taylor_green_3d --config ../examples/09_taylor_green_3d/tg_re100.cfg

# Override parameters
./taylor_green_3d --config ../examples/09_taylor_green_3d/tg_re100.cfg --N 64 --T 20.0
```

## Available Configurations

| Config | Grid | Re | Description |
|--------|------|-----|-------------|
| `tg_re100.cfg` | 32³ | 100 | Standard validation case |
| `tg_re100_fine.cfg` | 64³ | 100 | High-resolution validation |
| `tg_re1600.cfg` | 64³ | 1600 | Turbulence transition DNS |

## Configuration Parameters

Key parameters in the `.cfg` files:

```
Nx = 32             # Grid cells per direction
Ny = 32
Nz = 32
nu = 0.01           # Viscosity (sets Re = 1/nu for V₀=L=1)
Re = 100            # Reynolds number
dt = 0.01           # Time step
max_iter = 1000     # Max steps (determines T_final)
```

## Output

- `tg3d_*.vtk` - Velocity field snapshots (VTK format)
- `tg3d_final.vtk` - Final velocity field
- `tg3d_history.dat` - Time history of KE, enstrophy

## Key Metrics

1. **Kinetic Energy (KE)**: Should decay exponentially for low Re
2. **Enstrophy**: Integral of vorticity squared; peaks during turbulence transition
3. **KE Decay Rate**: Compare against theoretical 2ν

## Visualization

```bash
# ParaView - view vortex evolution
paraview output/tg_re100/tg3d_*.vtk

# Plot energy decay
python plot_energy.py output/tg_re100/

# gnuplot
gnuplot -e "plot 'output/tg_re100/tg3d_history.dat' u 1:3 w l title 'KE/KE0', exp(-0.02*x) title 'Theory'"
```

## Reference

Taylor, G.I. and Green, A.E. (1937). "Mechanism of the production of small eddies from large ones." Proc. R. Soc. Lond. A 158, 499-521.
