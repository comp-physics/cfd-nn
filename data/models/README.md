# Neural Network Turbulence Model Weights

All models are trained on the McConkey et al. (2021) RANS dataset using 5 Pope
invariant inputs: tr(S^2), tr(Omega^2), tr(S^3), tr(S*Omega^2), y/delta.

Training scripts: `scripts/paper/train_all_models.py`

## Model Inventory (21 models for paper)

### Classical RANS (9 models, no weights needed)
These are built into the solver — no weight files required.

| CLI `--model` | Description |
|---------------|-------------|
| `none` | Laminar (no turbulence model) |
| `baseline` | Mixing-length with van Driest damping |
| `gep` | Gene Expression Programming (Weatheritt & Sandberg 2016) |
| `komega` | Standard k-omega (Wilcox 1988) |
| `sst` | SST k-omega (Menter 1994) |
| `earsm_wj` | EARSM Wallin-Johansson (2000) |
| `earsm_gs` | EARSM Gatski-Speziale (1993) |
| `earsm_pope` | EARSM Pope (1975) |
| `rsm` | Reynolds Stress Model SSG (Speziale et al. 1991) |

### MLP — Scalar eddy viscosity (nu_t)
Predicts a scalar nu_t correction. Architecture: fully-connected with tanh/relu.

| Directory | Architecture | Params | CLI usage |
|-----------|-------------|--------|-----------|
| `mlp_paper` | 5 -> 32 -> 32 -> 1 | ~1.2K | `--model nn_mlp --weights data/models/mlp_paper` |
| `mlp_med_paper` | 5 -> 64 -> 64 -> 1 | ~4.5K | `--model nn_mlp --weights data/models/mlp_med_paper` |
| `mlp_large_paper` | 5 -> 128^4 -> 1 | ~50K | `--model nn_mlp --weights data/models/mlp_large_paper` |

### TBNN — Tensor Basis Neural Network (tau_ij)
Predicts 10 Pope tensor basis coefficients for full Reynolds stress anisotropy.

| Directory | Architecture | Params | CLI usage |
|-----------|-------------|--------|-----------|
| `tbnn_small_paper` | 5 -> 32 -> 32 -> 10 | ~1.4K | `--model nn_tbnn --weights data/models/tbnn_small_paper` |
| `tbnn_paper` | 5 -> 64^3 -> 10 | ~9K | `--model nn_tbnn --weights data/models/tbnn_paper` |
| `tbnn_large_paper` | 5 -> 128^3 -> 10 | ~35K | `--model nn_tbnn --weights data/models/tbnn_large_paper` |

### PI-TBNN — Physics-Informed TBNN (tau_ij)
Same as TBNN but trained with physics-informed loss (Galilean invariance,
realizability).

| Directory | Architecture | Params | CLI usage |
|-----------|-------------|--------|-----------|
| `pi_tbnn_small_paper` | 5 -> 32 -> 32 -> 10 | ~1.4K | `--model nn_tbnn --weights data/models/pi_tbnn_small_paper` |
| `pi_tbnn_paper` | 5 -> 64^3 -> 10 | ~9K | `--model nn_tbnn --weights data/models/pi_tbnn_paper` |
| `pi_tbnn_large_paper` | 5 -> 128^3 -> 10 | ~35K | `--model nn_tbnn --weights data/models/pi_tbnn_large_paper` |

### TBRF — Tensor Basis Random Forest (tau_ij)
Predicts 10 tensor basis coefficients using random forests (Kaandorp & Dwight
2020). Stored as binary `trees.bin`, not neural network weight files.

| Directory | Trees | Binary size | CLI usage |
|-----------|-------|-------------|-----------|
| `tbrf_1t_paper` | 1 | 5.4 MB | `--model nn_tbrf --weights data/models/tbrf_1t_paper` |
| `tbrf_5t_paper` | 5 | 28 MB | `--model nn_tbrf --weights data/models/tbrf_5t_paper` |
| `tbrf_10t_paper` | 10 | 54 MB | `--model nn_tbrf --weights data/models/tbrf_10t_paper` |

The full 200-tree forest (~1.1 GB binary, 3.3 GB pickle) is too large for the
repo. The 1/5/10 tree variants are subsampled to match the MLP/TBNN size sweep.

### Legacy models (case-holdout splits)
Trained with case-holdout split instead of random split. Used by some tests.

| Directory | Type |
|-----------|------|
| `tbnn_channel_caseholdout` | TBNN, channel holdout |
| `tbnn_phll_caseholdout` | TBNN, periodic hills holdout |

## File Formats

### MLP / TBNN / PI-TBNN
```
layer{N}_W.txt    Weight matrix (rows=output_dim, cols=input_dim)
layer{N}_b.txt    Bias vector
input_means.txt   Feature means for z-score normalization
input_stds.txt    Feature stds for z-score normalization
metadata.json     Architecture and training info
```

### TBRF
```
trees.bin         Binary format (see metadata.json for layout)
tree_offsets.txt  Node offset per tree
input_means.txt   Feature means for z-score normalization
input_stds.txt    Feature stds for z-score normalization
metadata.json     n_trees, max_depth, n_basis
```

## References

- McConkey, R., Yee, E., & Lien, F. S. (2021). A curated dataset for
  data-driven turbulence modelling. Scientific Data, 8(1), 255.
- Ling, J., Kurzawski, A., & Templeton, J. (2016). Reynolds averaged
  turbulence modelling using deep neural networks with embedded invariance.
  JFM, 807, 155-166.
- Kaandorp, M. L. A., & Dwight, R. P. (2020). Data-driven modelling of the
  Reynolds stress tensor using random forests with invariance. Computers &
  Fluids, 202, 104497.
- Pope, S. B. (1975). A more general effective-viscosity hypothesis. JFM, 72(2),
  331-340.
