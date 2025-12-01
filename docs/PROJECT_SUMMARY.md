# NN-CFD Project Summary

## What This Is

A production-ready C++ incompressible RANS solver with pluggable turbulence closures, including state-of-the-art neural network models. Implements the TBNN (Tensor Basis Neural Network) from Ling et al. 2016, trained on real DNS data.

## Quick Start

**Build:**
```bash
mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4
```

**Run:**
```bash
./channel --model gep --adaptive_dt                    # Fast symbolic model
./periodic_hills --model nn_tbnn --nn_preset tbnn_hills_real  # ML model on DNS data
```

**Test:**
```bash
./run_tests.sh  # Runs 6 test suites, all pass
```

## Available Models

| Model | Type | Speed | Accuracy | Training |
|-------|------|-------|----------|----------|
| `baseline` | Algebraic | Fast | Baseline | No |
| `gep` | Symbolic | Fast | Best convergence | No |
| `nn_mlp` | Neural Net | Medium | Good | Yes |
| `nn_tbnn` | Neural Net | Slow (350x) | Good | Yes |

**Benchmark (Periodic Hills, 32x48 grid):**
- GEP: 0.953 bulk velocity, 0.01 ms/iter, 1.74e-03 residual
- TBNN: 0.935 bulk velocity, 3.49 ms/iter, 3.00e-03 residual (trained on 73k DNS samples)

## Training Your Own Model

**Setup:**
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
bash scripts/download_mcconkey_data.sh  # Downloads 71GB dataset
python scripts/preprocess_mcconkey_csv.py --case periodic_hills
```

**Train:**
```bash
python scripts/train_tbnn_mcconkey.py --case periodic_hills --epochs 100
# Outputs to data/models/tbnn_hills_real/
```

**Use:**
```bash
./periodic_hills --model nn_tbnn --nn_preset tbnn_hills_real
```

## Key Features

- Pure C++ NN inference (no runtime Python/TensorFlow)
- Validated against analytical Poiseuille solution (<5% error)
- Complete PyTorch training pipeline
- McConkey 2021 dataset integration (DNS/LES ground truth)
- 6 comprehensive test suites with CI
- Adaptive time stepping
- VTK output for visualization
- Performance instrumentation

## Project Structure

```
include/          C++ headers
src/              Implementations
app/              Executables (channel, periodic_hills)
tests/            Unit tests (all passing)
scripts/          Python training scripts
data/models/      Trained model weights
docs/             Detailed documentation
```

## Validation Results

**Laminar Poiseuille:** 0.13% L2 error vs analytical (nu=0.1, 32x64 grid)

**Turbulent Periodic Hills:**
- All models produce valid, converged results
- GEP provides best speed/accuracy tradeoff
- TBNN demonstrates successful DNS-trained ML turbulence closure

## Documentation

- `README.md` - Full documentation
- `QUICK_START.md` - Build and run in 60 seconds
- `QUICK_TRAIN.md` - Train a model in 30 minutes
- `TESTING.md` - Test suite guide
- `VALIDATION.md` - Validation results
- `docs/TRAINING_GUIDE.md` - Complete training workflow
- `FINAL_RESULTS.md` - Benchmark comparisons

## CI/CD

GitHub Actions runs on every push:
- Builds on Ubuntu + macOS
- Release + Debug modes
- All 6 test suites
- Compiler warning checks
- Documentation validation

## Key Achievements

1. Fully working RANS solver with multiple turbulence closures
2. Successfully integrated and trained TBNN on real DNS data (McConkey 2021)
3. Pure C++ NN inference with no runtime dependencies
4. Complete training infrastructure (download --> preprocess --> train --> integrate)
5. Comprehensive testing (6 suites, all passing)
6. Production-ready code quality (C++17, -Wall -Wextra -O3)

## Citation

If you use this code, cite:
- Ling et al. (2016) - TBNN model
- McConkey et al. (2021) - Training dataset

## License & Contact

See README.md for full details.

**Status:** Production-ready, fully tested, documented, and validated.

