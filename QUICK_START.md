# Quick Start Guide

## Build and Run in 60 Seconds

```bash
# Clone/navigate to directory
cd nn-cfd

# Build
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4

# Run laminar validation (should see ~0.13% error)
./channel --Nx 32 --Ny 64 --nu 0.1 --dt 0.005 --max_iter 10000

# Run with turbulence model
./channel --Nx 32 --Ny 64 --nu 0.001 --model baseline --max_iter 20000

# Run with NN model (example weights)
./channel --model nn_mlp --nn_preset example_scalar_nut --Nx 16 --Ny 32 --max_iter 100
```

## What You Get

- Incompressible RANS solver with projection method  
- Validated against Poiseuille flow (0.13% error)  
- 3 turbulence closures: laminar, baseline, NN (MLP + TBNN)  
- Pure C++ NN inference (no external dependencies)  
- Model zoo for published models  
- Export tools for PyTorch/TensorFlow weights  

## Project Goals

This is a **research tool** for testing neural network turbulence closures:
- Compare NN models against baseline RANS
- Measure online computational cost
- Use pre-trained weights from published papers
- Focus on canonical flows (channel, hills)

**Not included:** NN training (do this in Python externally)

## Next Steps

1. **Validate:** Check results in `VALIDATION.md`
2. **Add models:** See `data/models/README.md` for target published models
3. **Customize:** Modify `include/*.hpp` headers for your needs

## Documentation

- **README.md** - Full documentation and usage
- **VALIDATION.md** - Test results and recommended parameters
- **data/models/README.md** - Model zoo and integration guide
- **docs/MODEL_ZOO_GUIDE.md** - Detailed model integration workflow

## Questions?

- Check header files in `include/` for implementation details
- See `app/main_channel.cpp` for example usage
- Run `./channel --help` for all options
