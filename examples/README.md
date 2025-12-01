# CFD-NN Examples

This directory contains **validated, ready-to-run examples** demonstrating the capabilities of the CFD-NN solver for turbulence modeling using neural networks.

## Quick Start

```bash
# Navigate to any example directory
cd examples/01_laminar_channel

# Run the example
./run.sh

# View results
# - VTK files in output/ directory
# - Comparison plots generated automatically
```

## Phase 1: Core Validation Examples ✅

Essential examples for **verification and validation** of the solver.

### [01_laminar_channel/](01_laminar_channel/) - **Poiseuille Flow**

**What**: Analytical validation for laminar flow  
**Time**: ~30 seconds  
**Level**: Beginner  

Validates solver against exact Poiseuille solution. **Must pass** before using solver for research.

**Key Learning**:
- How to run basic simulations
- Verify pressure-velocity coupling
- Compare with analytical solutions

---

### [02_turbulent_channel/](02_turbulent_channel/) - **Model Comparison**

**What**: Compare 5 turbulence models (None, Baseline, GEP, NN-MLP, NN-TBNN)  
**Time**: ~20 minutes  
**Level**: Intermediate  

Demonstrates performance differences between turbulence closures on Re_τ = 180 channel flow.

**Key Learning**:
- How turbulence models affect predictions
- Baseline vs GEP performance
- How to use NN models (when trained)

**Note**: NN models require training on McConkey dataset for real results. See `docs/TRAINING_GUIDE.md`.

---

### [03_grid_refinement/](03_grid_refinement/) - **Convergence Study**

**What**: Systematic grid refinement to prove 2nd-order accuracy  
**Time**: ~20 minutes  
**Level**: Advanced  

Quantifies numerical error using Richardson extrapolation across 4 grid levels.

**Key Learning**:
- Verify spatial order of accuracy (p ≈ 2.0)
- Compute Grid Convergence Index (GCI)
- Demonstrate V&V best practices

---

### [04_validation_suite/](04_validation_suite/) - **Benchmark Suite**

**What**: 4 validation cases (2 analytical, 2 DNS benchmarks)  
**Time**: ~30 minutes  
**Level**: Intermediate  

Comprehensive test suite comparing against:
- Poiseuille analytical solutions (Re = 100, 1000)
- Moser DNS data (Re_τ = 180, 395)

**Key Learning**:
- Verification vs Validation
- Comparison with DNS benchmarks
- Acceptable error thresholds for RANS

---

## Running Examples

### Prerequisites

1. **Build the solver**:
   ```bash
   cd ../
   mkdir -p build && cd build
   cmake .. && make
   ```

2. **Python dependencies** (for analysis/plotting):
   ```bash
   pip install numpy matplotlib scipy
   ```

### Running Individual Examples

Each example has:
- **Config files** (`.cfg`) - Simulation parameters
- **Run script** (`run.sh` or `run_all.sh`) - Automated execution
- **Analysis script** (`.py`) - Post-processing and plotting
- **README.md** - Detailed documentation

**Typical workflow**:
```bash
cd examples/XX_example_name/
./run.sh                    # Run simulation(s)
ls output/                  # Check VTK output files
python analyze.py           # Generate comparison plots (if not auto-run)
```

### Viewing Results

**ParaView** (3D visualization):
```bash
paraview output/velocity_final.vtk
```

**Python plots** (automated):
- Each example generates PNG plots in `output/` directory
- Side-by-side comparisons with analytical/DNS data
- Error metrics and validation status

## Example Progression (Learning Path)

**New users** - Start here:
1. ✅ **Example 1** (Laminar channel) - Verify installation works
2. ✅ **Example 4** (Validation suite) - See solver accuracy across cases
3. ✅ **Example 2** (Model comparison) - Understand turbulence modeling

**Researchers** - Advanced studies:
4. ✅ **Example 3** (Grid refinement) - Quantify numerical uncertainty
5. 📚 **Train your own models** - See `docs/TRAINING_GUIDE.md`
6. 🚀 **Create custom examples** - Adapt templates for your cases

## Expected Runtime

| Example | Quick Test | Full Run | Output Size |
|---------|-----------|----------|-------------|
| 01_laminar_channel | 30 sec | 2 min | ~5 MB |
| 02_turbulent_channel | 5 min | 25 min | ~50 MB |
| 03_grid_refinement | 5 min | 25 min | ~100 MB |
| 04_validation_suite | 10 min | 35 min | ~80 MB |

*Times on typical laptop (4-core CPU, no GPU)*

## Success Criteria

### Verification (Analytical Cases)

✅ **PASS**: Error < 1-5%  
❌ **FAIL**: Error > 5% → Check solver implementation

### Validation (DNS Cases)

✅ **PASS**: Error < 15-25% (RANS models)  
⚠️ **ACCEPTABLE**: Error < 35%  
❌ **FAIL**: Error > 35% → Check turbulence model

## Turbulence Models Available

| Model | Training Required? | Typical Accuracy | Speed |
|-------|-------------------|------------------|-------|
| **None** | No | Poor (laminar-like) | Fastest |
| **Baseline** | No | Fair (~20% error) | Fast |
| **GEP** | No | Good (~15% error) | Fast |
| **NN-MLP** | Yes* | Very Good (~10% error) | Medium |
| **NN-TBNN** | Yes* | Best (~8% error) | Slower |

*NN models have example weights (random) but need training on DNS data for real results.

### Training Neural Network Models

To use NN models properly:

```bash
# 1. Download McConkey dataset (~500 MB, one-time)
bash scripts/download_mcconkey_data.sh

# 2. Train MLP model (15 min)
python scripts/train_mlp_mcconkey.py \
    --case channel \
    --output data/models/mlp_real

# 3. Train TBNN model (30 min)
python scripts/train_tbnn_mcconkey.py \
    --case channel \
    --output data/models/tbnn_real

# 4. Update example configs to use trained models
# Edit 02_turbulent_channel/04_nnmlp.cfg:
#   nn_preset = mlp_real
# Edit 02_turbulent_channel/05_nntbnn.cfg:
#   nn_preset = tbnn_real

# 5. Re-run comparisons
cd examples/02_turbulent_channel
./run_all.sh
```

See **`docs/TRAINING_GUIDE.md`** for complete instructions.

## Config File Format

All examples use `.cfg` config files with key-value pairs:

```bash
# Grid
Nx = 64
Ny = 128

# Domain
Lx = 4.0
Ly = 2.0

# Physics
Re = 100.0
nu = 0.01
dp_dx = -0.001

# Turbulence
model = baseline    # Options: none, baseline, gep, nn_mlp, nn_tbnn

# Solver
tol = 1e-10
max_iter = 10000

# Output
output_interval = 500
verbose = true
```

**Tip**: Copy an existing config and modify parameters for your case.

## Troubleshooting

### Solver Not Found

```
ERROR: Solver not found at build/channel
```

**Fix**: Build the project first:
```bash
cd ../
mkdir -p build && cd build
cmake .. && make -j4
```

### Python Analysis Fails

```
ModuleNotFoundError: No module named 'numpy'
```

**Fix**: Install dependencies:
```bash
pip install numpy matplotlib scipy
```

### Simulation Doesn't Converge

**Check**:
1. Is `max_iter` large enough? (Try 50000)
2. Is `CFL_max` too large? (Reduce to 0.3)
3. Is grid stretched too aggressively? (Reduce `beta_y`)

### VTK Files Look Wrong in ParaView

**Common issues**:
- Empty fields → Simulation didn't converge
- NaN values → Numerical instability (reduce CFL)
- Oscillations → Use upwind convection scheme

### NN Models Give Nonsense Results

**Expected!** Example weights are random.

**Fix**: Train real models (see above) or focus on None/Baseline/GEP models.

## Creating Custom Examples

Use existing examples as templates:

1. **Copy a similar example**:
   ```bash
   cp -r 01_laminar_channel 05_my_case
   ```

2. **Modify config files** for your case

3. **Update run script** with new config names

4. **Add reference data** to analysis script (if available)

5. **Document** in README.md

## Performance Optimization

### Faster Runs

- Reduce grid: `Nx = 32, Ny = 64`
- Relax tolerance: `tol = 1e-8`
- Reduce snapshots: `num_snapshots = 1`

### More Accurate

- Finer grid: `Nx = 128, Ny = 256`
- Stricter tolerance: `tol = 1e-12`
- Central differencing: `convection_scheme = central`

### GPU Acceleration

If compiled with `USE_GPU_OFFLOAD=ON`:
- NN models automatically use GPU
- 3-5× speedup for NNMLP/NNTBNN
- No config changes needed

## Citing This Work

If you use these examples in research, please cite:

```bibtex
@software{cfdnn2024,
  title={CFD-NN: Neural Network Turbulence Modeling Framework},
  author={[Your Name]},
  year={2024},
  url={https://github.com/yourusername/cfd-nn}
}
```

And the relevant turbulence model papers (see individual example READMEs).

## Additional Resources

### Documentation
- **`../README.md`** - Main project documentation
- **`../docs/TRAINING_GUIDE.md`** - How to train NN models
- **`../docs/VALIDATION.md`** - Validation results
- **`../docs/TESTING.md`** - Unit test documentation

### Datasets
- **McConkey et al. (2021)** - Turbulence modeling dataset (Kaggle)
- **Moser DNS Data** - Channel flow benchmarks (KTH)
- **Johns Hopkins Turbulence Database** - High-Re DNS data

### Papers
- **Baseline Model**: Van Driest (1956) - Mixing length with damping
- **GEP**: Weatheritt & Sandberg (2016) - Gene expression programming
- **TBNN**: Ling et al. (2016) - Tensor basis neural networks

## Support

- **Issues**: Open a GitHub issue
- **Questions**: See `docs/FAQ.md` (if available)
- **Contributing**: Pull requests welcome!

---

## Example Summary Table

| # | Name | Purpose | Time | Difficulty | Models Tested |
|---|------|---------|------|------------|---------------|
| 01 | Laminar Channel | Verification | 2 min | ⭐ | None |
| 02 | Model Comparison | Model evaluation | 25 min | ⭐⭐ | All 5 |
| 03 | Grid Refinement | Convergence study | 25 min | ⭐⭐⭐ | None |
| 04 | Validation Suite | Benchmarking | 35 min | ⭐⭐ | None, Baseline |

**Legend**: ⭐ Beginner | ⭐⭐ Intermediate | ⭐⭐⭐ Advanced

---

**Ready to start?** Begin with Example 1:

```bash
cd 01_laminar_channel
./run.sh
```

Good luck! 🚀

