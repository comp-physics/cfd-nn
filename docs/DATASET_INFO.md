# McConkey Dataset Information

## Overview

This document provides details about the McConkey et al. (2021) dataset for data-driven turbulence modeling.

## Citation

```bibtex
@article{mcconkey2021curated,
  title={A curated dataset for data-driven turbulence modelling},
  author={McConkey, Romit and Yee-Chung, Lyle and Phon-Anant, Marius and Eyassu, James and Sharma, Amit},
  journal={Scientific Data},
  volume={8},
  number={1},
  pages={1--12},
  year={2021},
  publisher={Nature Publishing Group},
  doi={10.1038/s41597-021-01034-2}
}
```

## Dataset Description

The McConkey dataset is **specifically designed for training machine learning turbulence models**. It contains:

- RANS simulation results (baseline turbulence models)
- High-fidelity DNS/LES results (ground truth)
- Pre-computed features for ML models
- Multiple canonical flow cases

### Why This Dataset?

1. **Complete Feature Set**: Includes all features needed for TBNN (Ling et al. 2016)
   - 5 scalar invariants (λ₁, λ₂, λ₃, λ₄, λ₅)
   - 10 tensor basis functions (T⁽¹⁾, ..., T⁽¹⁰⁾)
   - Anisotropy tensor b_ij from DNS

2. **Multiple Flow Cases**: Train and test on different geometries
   - Channel flow (fully developed turbulent pipe)
   - Periodic hills (separated flow)
   - Square duct (secondary flows)
   - Converging-diverging channel (adverse pressure gradient)

3. **Ready to Use**: No preprocessing needed
   - Features already normalized
   - Train/validation/test splits provided
   - Documented format

4. **Publicly Available**: Open access via Kaggle
   - No need to contact authors
   - Standardized format for reproducibility

## Flow Cases

### 1. Periodic Hills

**Description**: Flow over a sinusoidal lower wall with periodic boundary conditions

**Challenge**: Separated flow with recirculation bubble

**Why Important**: Classic test case for turbulence models. Standard RANS models struggle with separation and reattachment.

**Grid Size**: ~50,000 cells

**Reynolds Number**: Re_h = 10,595 (based on hill height)

**DNS Reference**: Breuer et al. (2009), Rapp & Manhart (2011)

**Good for**: Testing TBNN's ability to learn anisotropic stresses in separated flows

### 2. Channel Flow

**Description**: Fully developed turbulent channel between two parallel walls

**Challenge**: Turbulent boundary layers, wall effects

**Why Important**: Fundamental turbulence benchmark. Analytical and DNS solutions available.

**Grid Size**: ~10,000-50,000 cells

**Reynolds Number**: Re_τ = 180, 395, 590

**DNS Reference**: Moser et al. (1999), Lee & Moser (2015)

**Good for**: Initial model development and validation

### 3. Square Duct

**Description**: Turbulent flow through a square cross-section duct

**Challenge**: Secondary flows driven by Reynolds stress anisotropy

**Why Important**: Standard eddy-viscosity models (EVMs) cannot predict secondary flows. TBNN should capture them.

**Grid Size**: ~100,000 cells

**Reynolds Number**: Re_τ = 300

**DNS Reference**: Pinelli et al. (2010)

**Good for**: Demonstrating TBNN advantage over scalar eddy viscosity

### 4. Converging-Diverging Channel

**Description**: Channel with converging then diverging sections

**Challenge**: Adverse pressure gradient, near-separation

**Why Important**: Tests model generalization to non-equilibrium flows

**Good for**: Testing model robustness

## Data Format

The dataset is organized as:

```
mcconkey_data/
├── periodic_hills/
│   ├── train/
│   │   └── data.npz          # Training data
│   ├── val/
│   │   └── data.npz          # Validation data
│   └── test/
│       └── data.npz          # Test data
├── channel_Re180/
│   └── ...
├── square_duct/
│   └── ...
└── README.md
```

### Data.npz Contents

Each `data.npz` file contains:

**Input Features (for TBNN)**:
- `invariants`: [N, 5] - Scalar invariants λ₁, λ₂, λ₃, λ₄, λ₅
- `basis`: [N, 10, 6] - Tensor basis functions (10 in 3D, 4 in 2D)

**Labels (Ground Truth)**:
- `anisotropy`: [N, 6] or [N, 3] - Reynolds stress anisotropy b_ij from DNS
- `tau_ij`: [N, 6] or [N, 3] - Reynolds stresses (optional)

**Additional Data**:
- `k`: [N] - Turbulent kinetic energy
- `omega`: [N] - Specific dissipation rate
- `wall_distance`: [N] - Distance to nearest wall
- `velocity`: [N, 2] or [N, 3] - Mean velocity field
- `S`: [N, 2, 2] or [N, 3, 3] - Strain rate tensor
- `Omega`: [N, 2, 2] or [N, 3, 3] - Rotation rate tensor

Where `N` is the number of grid cells.

### Invariants Definition

The 5 scalar invariants are:

1. **λ₁** = tr(S²) - Strain rate magnitude squared
2. **λ₂** = tr(Ω²) - Rotation rate magnitude squared
3. **λ₃** = tr(S³) - Third invariant of strain
4. **λ₄** = tr(S Ω²) - Mixed invariant
5. **λ₅** = tr(S² Ω²) - Higher-order mixed invariant

These are **Galilean invariant** - they don't change with coordinate system rotation/translation.

### Anisotropy Tensor

The Reynolds stress anisotropy is defined as:

**b_ij = τ_ij / (2k) - δ_ij/3**

Where:
- τ_ij = -⟨u'_i u'_j⟩ is the Reynolds stress tensor
- k = ½⟨u'_i u'_i⟩ is turbulent kinetic energy
- δ_ij is the Kronecker delta

**Properties**:
- Trace-free: b_ii = 0
- Symmetric: b_ij = b_ji
- Bounded: -1/3 ≤ b_ij ≤ 2/3

In 2D, only 3 independent components: b_xx, b_xy, b_yy (with b_xx + b_yy = 0).

## Downloading the Dataset

### Method 1: Kaggle CLI (Recommended)

```bash
# Install Kaggle
pip install kaggle

# Setup API credentials (one-time)
# 1. Go to https://www.kaggle.com/settings
# 2. Click "Create New API Token"
# 3. Save kaggle.json to ~/.kaggle/

# Download dataset
bash scripts/download_mcconkey_data.sh
```

### Method 2: Manual Download

1. Visit https://www.kaggle.com/datasets/ryleymcconkey/ml-turbulence-dataset
2. Create free Kaggle account if needed
3. Click "Download" button
4. Extract zip to `mcconkey_data/`

### Method 3: Direct Link

The dataset is also available from the original publication:

- **Zenodo**: https://zenodo.org/record/5164535
- **Size**: ~500 MB compressed, ~2 GB uncompressed

## Dataset Statistics

| Case | Training Samples | Validation | Test | Total |
|------|------------------|------------|------|-------|
| Periodic Hills | ~30,000 | ~10,000 | ~10,000 | ~50,000 |
| Channel Re=180 | ~8,000 | ~2,000 | ~2,000 | ~12,000 |
| Square Duct | ~80,000 | ~20,000 | ~20,000 | ~120,000 |
| Conv-Div Channel | ~40,000 | ~10,000 | ~10,000 | ~60,000 |

**Total**: ~240,000 labeled samples across all cases

## Using the Dataset

### Train TBNN

```bash
python scripts/train_tbnn_mcconkey.py \
    --data_dir mcconkey_data \
    --case periodic_hills \
    --output data/models/tbnn_hills
```

### Train MLP

```bash
python scripts/train_mlp_mcconkey.py \
    --data_dir mcconkey_data \
    --case channel \
    --output data/models/mlp_channel
```

### Custom Data Loading

```python
import numpy as np

# Load data
data = np.load('mcconkey_data/periodic_hills/train/data.npz')

# Get features and labels
invariants = data['invariants']  # [N, 5]
anisotropy = data['anisotropy']  # [N, 6] or [N, 3]
basis = data['basis']            # [N, 10, 6] or [N, 4, 3]

# Your training code here
```

## Comparison with Other Datasets

### Johns Hopkins Turbulence Database (JHTDB)

- **Pros**: High-resolution DNS, many flow types
- **Cons**: No pre-computed ML features, requires significant preprocessing
- **Best for**: Fundamental turbulence research

### NASA Turbulence Modeling Resource

- **Pros**: Standard verification cases, extensive documentation
- **Cons**: Mostly validation data, not ML-ready
- **Best for**: Testing existing turbulence models

### McConkey (This Dataset)

- **Pros**: ML-ready features, train/val/test splits, TBNN-compatible
- **Cons**: Limited to 4 flow cases, relatively small dataset
- **Best for**: Training ML turbulence models (TBNN, MLP, etc.)

## Recommended Training Strategy

1. **Start Simple**: Train on channel flow first (easiest case)
2. **Validate Carefully**: Check a priori predictions before running in CFD solver
3. **Test Generalization**: Train on one Re, test on another
4. **Cross-Case Testing**: Train on channel, test on periodic hills
5. **Ensemble Models**: Train multiple models and average predictions

## Known Issues

1. **2D vs 3D**: Dataset is 3D, but nn-cfd is 2D. You need to:
   - Extract 2D slice from 3D data, or
   - Use only 2D-relevant components (xx, xy, yy)

2. **Coordinate Systems**: Ensure C++ feature computation matches Python exactly

3. **Normalization**: Dataset may already be normalized - check before adding your own

## Support

- **Dataset Issues**: https://www.kaggle.com/datasets/ryleymcconkey/ml-turbulence-dataset/discussion
- **Original Paper**: https://www.nature.com/articles/s41597-021-01034-2
- **Related Code**: Check Kaggle for community notebooks

## Additional Resources

- **Tutorial**: See `docs/TRAINING_GUIDE.md`
- **Examples**: Kaggle has example notebooks for TBNN training
- **Papers Using This Dataset**:
  - Kaandorp & Dwight (2020) - Data-driven modelling of the Reynolds stress tensor
  - Geneva & Zabaras (2020) - Modeling the dynamics of PDE systems

## License

The McConkey dataset is released under **CC BY 4.0** (Creative Commons Attribution).

You are free to:
- ✅ Share - copy and redistribute
- ✅ Adapt - remix, transform, build upon
- ✅ Commercial use allowed

Attribution required: Cite McConkey et al. (2021) in any publications.


