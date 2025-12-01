# Quick Start: Build and Run in 60 Seconds

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

## Run

```bash
# Laminar channel flow
./channel --Nx 32 --Ny 64 --nu 0.01 --adaptive_dt --max_iter 10000

# Turbulent with mixing length
./channel --Nx 64 --Ny 128 --nu 0.001 --model baseline --adaptive_dt

# With GEP model
./channel --model gep --adaptive_dt

# Periodic hills
./periodic_hills --Nx 64 --Ny 96 --model baseline --adaptive_dt
```

## Output

Results saved to `output/`:
- `channel_velocity.dat` - Velocity field
- `channel_pressure.dat` - Pressure field
- `channel.vtk` - ParaView visualization

## Next Steps

- **Train NN models**: See `QUICK_TRAIN.md`
- **Full documentation**: See `README.md`
- **Validation results**: See `VALIDATION.md`
