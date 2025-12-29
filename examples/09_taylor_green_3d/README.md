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

```bash
# Default run (32³, Re=100, T=10)
./run.sh

# Higher resolution
N=64 ./run.sh

# Higher Reynolds number (turbulence transition)
N=64 RE=1600 T_FINAL=20.0 ./run.sh

# Quick test
N=16 RE=100 T_FINAL=1.0 ./run.sh
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| N | 32 | Grid cells per direction (N³ total) |
| RE | 100 | Reynolds number |
| T_FINAL | 10.0 | Simulation end time |
| DT | 0.01 | Time step |
| NUM_SNAPSHOTS | 10 | Number of VTK output files |

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
paraview output/tg3d_*.vtk

# Plot energy decay
gnuplot -e "plot 'output/tg3d_history.dat' u 1:3 w l title 'KE/KE0', exp(-0.02*x) title 'Theory'"
```

## Reference

Taylor, G.I. and Green, A.E. (1937). "Mechanism of the production of small eddies from large ones." Proc. R. Soc. Lond. A 158, 499-521.
