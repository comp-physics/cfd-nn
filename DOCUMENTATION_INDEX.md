# Documentation Index

## Getting Started

- **QUICK_START.md** - Build and run in 60 seconds  
- **README.md** - Complete project documentation and usage

## Technical Details

- **VALIDATION.md** - Test results, validation against analytical solutions, recommended parameters

## Model Zoo (Using Published NN Models)

- **data/models/README.md** - Target published models, validation protocol  
- **docs/MODEL_ZOO_GUIDE.md** - Detailed guide for integrating published models

## File Organization

```
Documentation:
+-- QUICK_START.md           # Start here
+-- README.md                # Full documentation
+-- VALIDATION.md            # Results and validation
+-- data/models/README.md    # Model zoo overview
+-- docs/MODEL_ZOO_GUIDE.md  # Integration workflow

Code:
+-- include/                 # All headers (well commented)
+-- src/                     # Implementations
+-- app/                     # Executable drivers
+-- scripts/                 # Weight export tools
+-- tests/                   # Unit tests

Data:
+-- data/                    # NN weights (text format)
+-- data/models/             # Model zoo (presets)
```

## Quick Reference

### Build
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4
```

### Run
```bash
# Laminar validation
./channel --Nx 32 --Ny 64 --nu 0.1 --dt 0.005 --max_iter 10000

# Baseline turbulence
./channel --model baseline --Nx 64 --Ny 128

# Neural network model
./channel --model nn_mlp --nn_preset example_scalar_nut
```

### Documentation by Purpose

| I want to... | Read this |
|--------------|-----------|
| Get started quickly | QUICK_START.md |
| Understand the full project | README.md |
| See validation results | VALIDATION.md |
| Add a published NN model | data/models/README.md |
| Understand model integration | docs/MODEL_ZOO_GUIDE.md |
| Modify the code | include/*.hpp (headers) |
| Export PyTorch/TF weights | scripts/export_*.py |

## Project Goals (Reminder)

This solver implements:
1. Steady incompressible RANS for canonical flows
2. Finite volume/difference on structured grids
3. Multiple turbulence closures (baseline + NN)
4. Pure C++ NN inference (no runtime dependencies)
5. Performance instrumentation

**Focus:** CFD solver + NN inference infrastructure  
**Training:** Done externally in Python
