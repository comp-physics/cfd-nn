# Neural Network Turbulence Model Training Methodology

Complete specification of all training procedures, hyperparameters, data processing, and results for reproducibility.

## 1. Dataset

**Source**: McConkey, Yee, & Lien (2021), "A curated dataset for data-driven turbulence modelling," *Scientific Data* 8, 255.

**Download**: https://www.kaggle.com/datasets/ryleymcconkey/ml-turbulence-dataset

**Files used**:
- `komegasst.csv` — RANS k-omega SST fields (73 columns per point)
- `REF.csv` — DNS/LES reference fields (48 columns per point)

**Total data**: 902,601 points across 38 flow cases (after removing 211 points with missing `REF_b_11` values, all in the `convdiv12600` case).

**Row alignment**: RANS and REF CSV files are row-aligned (row i in both files corresponds to the same spatial point). Combined by reading both files in lockstep with `csv.DictReader`.

### 1.1 Flow Cases in Dataset

The dataset contains 4 flow geometries:

| Geometry | Cases | Description |
|----------|-------|-------------|
| Square duct (SD) | 14 Reynolds numbers (Re = 1100–3500) | Fully developed turbulent duct flow |
| Periodic hills (PH) | 5 slope angles (alpha = 0.5, 0.8, 1.0, 1.2, 1.5) | Separated flow over periodic hills |
| Converging-diverging channel (CDC) | 1 case (Re = 12600) | Flow through a bump |
| Curved backward-facing step (CBFS) | 1 case (Re = 13700) | Separated flow over curved step |

### 1.2 Train/Validation/Test Split

Case-holdout protocol following TBKAN (2025). No random point-level splitting — entire flow cases are held out to test geometric generalization.

**Training set (18 cases, 271,924 points)**:
- Square duct: Re = {1100, 1150, 1250, 1300, 1350, 1400, 1500, 1600, 2205, 2400, 2600, 2900, 3200, 3500}
- Periodic hills: alpha = {0.5, 1.0, 1.5}
- Converging-diverging channel: Re = 12600

**Validation set (2 cases, 23,967 points)**:
- Square duct: Re = 2000 (interpolation within training Re range)
- Periodic hills: alpha = 0.8 (interpolation within training alpha range)

**Test set (2 cases, 51,844 points)**:
- Periodic hills: alpha = 1.2 (interpolation, but different from training angles)
- Curved backward-facing step: Re = 13700 (entirely new geometry, not seen in training)

### 1.3 RANS Baseline

All RANS fields computed with **k-omega SST** (Menter 1994). The RANS model provides:
- Mean velocity gradients: `gradU_ij` (9 components)
- Turbulent kinetic energy: `k`
- Turbulent dissipation rate: `epsilon`
- Additional fields not used in this work: `omega`, `nut`, `p`, gradient fields, Reynolds stress tensor `turbR_ij`, wall distance, pre-computed `S_ij` and `R_ij`

### 1.4 DNS/LES Reference Data

Reference high-fidelity data provides:
- Reynolds stress anisotropy tensor: `REF_b_11`, `REF_b_12`, `REF_b_13`, `REF_b_22`, `REF_b_23`, `REF_b_33`
- Defined as: $b_{ij} = \frac{\overline{u'_i u'_j}}{2k} - \frac{1}{3}\delta_{ij}$
- Also available but not used as labels: `REF_tau_ij`, `REF_k`, `REF_a_ij`, `REF_gradU_ij`, `REF_divtau_i`


## 2. Feature Engineering

### 2.1 Input Features: 5 Pope Invariants

Following Ling et al. (2016) and Pope (1975), the inputs to all neural network models are 5 scalar invariants of the non-dimensionalized mean strain rate and rotation rate tensors.

**Step 1: Compute strain and rotation rate from RANS velocity gradients**

$$S_{ij} = \frac{1}{2}\left(\frac{\partial U_i}{\partial x_j} + \frac{\partial U_j}{\partial x_i}\right), \quad
\Omega_{ij} = \frac{1}{2}\left(\frac{\partial U_i}{\partial x_j} - \frac{\partial U_j}{\partial x_i}\right)$$

Computed from `komegasst_gradU_ij` (verified to match the pre-computed `komegasst_S_ij` in the dataset to machine precision).

**Step 2: Non-dimensionalize using turbulence time scale**

$$\hat{S}_{ij} = \frac{k}{\epsilon} S_{ij}, \quad \hat{\Omega}_{ij} = \frac{k}{\epsilon} \Omega_{ij}$$

where $k$ and $\epsilon$ are from the RANS k-omega SST solution. Both $k$ and $\epsilon$ are clamped to a minimum of $10^{-30}$ to prevent division by zero.

**Step 3: Compute 5 scalar invariants**

$$\lambda_1 = \text{tr}(\hat{S}^2), \quad
\lambda_2 = \text{tr}(\hat{\Omega}^2), \quad
\lambda_3 = \text{tr}(\hat{S}^3), \quad
\lambda_4 = \text{tr}(\hat{\Omega}^2 \hat{S}), \quad
\lambda_5 = \text{tr}(\hat{\Omega}^2 \hat{S}^2)$$

These 5 invariants form a complete integrity basis for the independent invariants of a symmetric and an antisymmetric 3x3 tensor (Pope 1975).

**Implementation**: All matrix operations (products, traces) computed on GPU using `torch.bmm` in float64 precision for the full batch simultaneously.

### 2.2 Input Normalization

Z-score standardization applied to the 5 invariants:

$$\hat{\lambda}_i = \frac{\lambda_i - \mu_i}{\sigma_i}$$

where $\mu_i$ and $\sigma_i$ are the mean and standard deviation computed from the **training set only**. The same $\mu$ and $\sigma$ are applied to validation and test sets. Standard deviations below $10^{-12}$ are clamped to 1.0.

The normalization parameters are saved in `input_means.txt` and `input_stds.txt` alongside model weights.

### 2.3 Tensor Basis (for TBNN and TBRF)

The Pope (1975) 10-tensor integrity basis is computed for models that predict anisotropy components:

$$T^{(1)} = \hat{S}$$
$$T^{(2)} = \hat{S}\hat{\Omega} - \hat{\Omega}\hat{S}$$
$$T^{(3)} = \hat{S}^2 - \frac{1}{3}\text{tr}(\hat{S}^2)I$$
$$T^{(4)} = \hat{\Omega}^2 - \frac{1}{3}\text{tr}(\hat{\Omega}^2)I$$
$$T^{(5)} = \hat{\Omega}\hat{S}^2 - \hat{S}^2\hat{\Omega}$$
$$T^{(6)} = \hat{\Omega}^2\hat{S} + \hat{S}\hat{\Omega}^2 - \frac{2}{3}\text{tr}(\hat{S}\hat{\Omega}^2)I$$
$$T^{(7)} = \hat{\Omega}\hat{S}\hat{\Omega}^2 - \hat{\Omega}^2\hat{S}\hat{\Omega}$$
$$T^{(8)} = \hat{S}\hat{\Omega}\hat{S}^2 - \hat{S}^2\hat{\Omega}\hat{S}$$
$$T^{(9)} = \hat{\Omega}^2\hat{S}^2 + \hat{S}^2\hat{\Omega}^2 - \frac{2}{3}\text{tr}(\hat{S}^2\hat{\Omega}^2)I$$
$$T^{(10)} = \hat{\Omega}\hat{S}^2\hat{\Omega}^2 - \hat{\Omega}^2\hat{S}^2\hat{\Omega}$$

Each tensor is symmetric and traceless. Stored as 6 independent components per tensor: (11, 12, 13, 22, 23, 33). Shape: [N, 10, 6] per split.

### 2.4 Training Labels

**TBNN/PI-TBNN/TBRF**: Full anisotropy tensor $b_{ij}$ from DNS, 6 symmetric components: $(b_{11}, b_{12}, b_{13}, b_{22}, b_{23}, b_{33})$. This is the raw DNS anisotropy, **not** the discrepancy $\Delta b = b_\text{DNS} - b_\text{RANS}$ (following Ling et al. 2016).

**MLP/MLP-Large**: Scalar proxy for eddy viscosity — the Frobenius norm of the anisotropy tensor:

$$|b| = \sqrt{\sum_{ij} b_{ij}^2} = \sqrt{b_{11}^2 + 2b_{12}^2 + 2b_{13}^2 + b_{22}^2 + 2b_{23}^2 + b_{33}^2}$$

(Note: the actual code computes $\sqrt{\sum_c b_c^2}$ over the 6 stored components, which double-counts off-diagonal terms. This is equivalent to $\sqrt{b_{11}^2 + b_{12}^2 + b_{13}^2 + b_{22}^2 + b_{23}^2 + b_{33}^2}$ without the factor of 2 on off-diagonals. This is a consistent proxy target across all MLP experiments.)


## 3. Model Architectures

### 3.1 MLP (Scalar Closure)

| Property | Value |
|----------|-------|
| Architecture | 5 → 32 → 32 → 1 |
| Activation | Tanh (hidden layers), Linear (output) |
| Parameters | 1,249 |
| Input | 5 normalized invariants |
| Output | Scalar (anisotropy magnitude) |
| Weight init | PyTorch default (Kaiming uniform) |

### 3.2 MLP-Large (Scalar Closure)

| Property | Value |
|----------|-------|
| Architecture | 5 → 128 → 128 → 128 → 128 → 1 |
| Activation | Tanh (hidden layers), Linear (output) |
| Parameters | 50,049 |
| Input | 5 normalized invariants |
| Output | Scalar (anisotropy magnitude) |
| Weight init | PyTorch default (Kaiming uniform) |

### 3.3 TBNN (Tensor Basis Neural Network)

Following Ling, Kurzawski & Templeton (2016), JFM 807.

| Property | Value |
|----------|-------|
| Architecture | 5 → 64 → 64 → 64 → 10 |
| Activation | Tanh (hidden layers), Linear (output) |
| Parameters | 9,354 |
| Input | 5 normalized invariants |
| Output | 10 tensor basis coefficients $g^{(n)}$ |
| Prediction | $b_{ij} = \sum_{n=1}^{10} g^{(n)} T^{(n)}_{ij}$ |
| Weight init | PyTorch default (Kaiming uniform) |

The TBNN output layer has no activation — the 10 coefficients are unconstrained real numbers. The anisotropy tensor is reconstructed by contracting these coefficients with the pre-computed tensor basis.

### 3.4 PI-TBNN (Physics-Informed TBNN)

Same architecture as TBNN (Section 3.3) but trained with an augmented loss function (Section 4.2).

### 3.5 TBRF (Tensor Basis Random Forest)

Following Kaandorp & Dwight (2020), *Computers & Fluids* 202.

| Property | Value |
|----------|-------|
| Algorithm | scikit-learn `RandomForestRegressor` |
| Trees per coefficient | 200 |
| Number of coefficients | 10 (Pope basis) |
| Total trees | 2,000 |
| Max depth | 20 |
| Min samples split | 2 (default) |
| Min samples leaf | 1 (default) |
| Max features | sqrt (default) |
| Bootstrap | True (default) |
| Random state | 42 |
| n_jobs | -1 (all cores) |

**Coefficient extraction**: Before training the random forests, the 10 tensor basis coefficients $g^{(n)}$ are solved for each training point via the minimum-norm least-squares solution:

$$g = A^T (A A^T)^{-1} b$$

where $A = T^T$ is the $[6 \times 10]$ basis matrix (transposed) and $b$ is the $[6]$ anisotropy vector. This is underdetermined (6 equations, 10 unknowns), so the minimum-norm solution is used. Regularized with $A A^T + 10^{-10} I$ for numerical stability.

Computed in vectorized form using NumPy `einsum` and batched matrix inverse on all 271,924 training points simultaneously.

**Compact export variants**: For C++ solver integration, compact versions with reduced tree counts are exported as flat binary files. The binary format packs `children_left` (int32), `children_right` (int32), `feature` (int32), `threshold` (float32), and `value` (float32) per node, with a 12-byte header (total_nodes, n_basis, n_trees as int32).


## 4. Training Procedure

### 4.1 Common Hyperparameters (All Neural Networks)

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam (Kingma & Ba 2015) |
| Initial learning rate | $10^{-3}$ |
| Adam $\beta_1, \beta_2$ | 0.9, 0.999 (PyTorch defaults) |
| Adam $\epsilon$ | $10^{-8}$ (PyTorch default) |
| LR schedule | Cosine annealing with warm restarts |
| Cosine $T_0$ | 200 epochs |
| Cosine $T_\text{mult}$ | 2 (period doubles each restart) |
| Cosine $\eta_\text{min}$ | $10^{-6}$ |
| Batch size | 256 |
| Max epochs | 1,000 |
| Early stopping patience | 150 epochs |
| Early stopping criterion | Validation MSE (no improvement for 150 consecutive epochs) |
| Best model selection | Checkpoint with lowest validation MSE across all epochs |
| Loss function | MSE (mean squared error) |
| Regularization | None (no L2, no dropout) |
| Data precision | float32 for training, float64 for feature extraction |
| DataLoader shuffle | True |

**Cosine annealing schedule**: Learning rate follows $\eta_t = \eta_\text{min} + \frac{1}{2}(\eta_\text{max} - \eta_\text{min})(1 + \cos(\frac{T_\text{cur}}{T_i}\pi))$ with warm restarts. The period starts at $T_0 = 200$ epochs and doubles after each restart ($T_1 = 400$, $T_2 = 800$, ...). This allows the optimizer to escape local minima by periodically increasing the learning rate.

### 4.2 PI-TBNN Loss Function

$$\mathcal{L} = \text{MSE}(b_\text{pred}, b_\text{DNS}) + \alpha \sum_p w_p^2 + \beta \cdot \text{penalty}$$

where:

- **MSE**: Standard mean squared error on all 6 anisotropy components
- **L2 regularization**: $\alpha = 10^{-6}$ (effectively zero; included for framework consistency)
- **Realizability penalty**: soft constraints on diagonal components:

$$\text{penalty} = \frac{1}{N}\sum_{i=1}^{N} \sum_{c \in \{11,22,33\}} \left[\text{ReLU}(-b_{cc}^{(i)} - \tfrac{1}{3})^2 + \text{ReLU}(b_{cc}^{(i)} - \tfrac{2}{3})^2\right]$$

These enforce $b_{ii} \geq -\frac{1}{3}$ and $b_{ii} \leq \frac{2}{3}$ (necessary conditions for realizability).

- **Beta warmup**: Linear ramp from $\beta = 0$ to the target $\beta$ over the first 100 epochs: $\beta_\text{eff}(t) = \beta \cdot \min(1, t/100)$
- **Validation metric**: Always pure MSE (no penalty terms), for fair comparison across models

**Beta sweep conducted**: $\beta \in \{0.001, 0.01\}$ (see Section 6.2).

### 4.3 TBNN Training Target

The TBNN predicts 10 scalar coefficients $g^{(n)}(\hat{\lambda}_1, \ldots, \hat{\lambda}_5)$. The anisotropy tensor is reconstructed as:

$$b_{ij} = \sum_{n=1}^{10} g^{(n)} T^{(n)}_{ij}$$

The loss is computed on the **reconstructed** $b_{ij}$, not on the coefficients $g^{(n)}$ directly. This means the network learns to minimize the error in physical space, allowing it to distribute the representation across basis tensors freely.

### 4.4 MLP Training Target

The MLP models predict a single scalar — the Frobenius-like norm of the anisotropy tensor (Section 2.4). The loss is MSE between predicted and true scalar values.


## 5. Software and Hardware

### 5.1 Software Versions

| Package | Version |
|---------|---------|
| Python | 3.x (NVIDIA PyTorch container) |
| PyTorch | 2.1.0a0+32f93b1 |
| NumPy | 1.22.2 |
| scikit-learn | 1.2.0 |
| pandas | 1.5.3 |
| CUDA | (container default) |

### 5.2 Hardware

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA L40S |
| CPU | Available via SLURM (4 cores for TBRF) |
| System | Georgia Tech PACE cluster |

### 5.3 Reproducibility

- No fixed random seed for PyTorch (weight initialization and data shuffling are non-deterministic)
- TBRF uses `random_state=42` for deterministic forest construction
- Feature extraction uses float64 precision on GPU; training uses float32
- Training script: `scripts/paper/train_all_models.py`


## 6. Results

### 6.1 Final Model Performance

Validation RMSE computed on pure MSE of all 6 anisotropy components (for TBNN/PI-TBNN/TBRF) or the scalar target (for MLP/MLP-Large).

| Model | Val RMSE(b) | Epochs Run | Stopped By | Parameters | Weights Size |
|-------|------------|------------|------------|------------|-------------|
| TBRF (200 trees) | **0.0637** | n/a | n/a | ~55M nodes | 3.3 GB (pickle) |
| TBNN | **0.0845** | 694 | Early stop | 9,354 | 196 KB |
| PI-TBNN (beta=0.001) | 0.0852 | 729 | Early stop | 9,354 | 196 KB |
| MLP-Large | 0.1045 | 344 | Early stop | 50,049 | 896 KB |
| MLP | 0.1096 | 1,000 | Max epochs | 1,249 | 56 KB |

**Note on MLP/MLP-Large RMSE**: These models predict a scalar proxy (anisotropy magnitude), so their RMSE is not directly comparable to the tensor-predicting models (TBNN, TBRF). The RMSE values for MLP models are reported for the scalar target, while TBNN/TBRF RMSE values are computed over all 6 anisotropy components.

### 6.2 PI-TBNN Beta Sweep

The realizability penalty weight $\beta$ was swept to determine whether physics-informed constraints improve accuracy.

| Beta | Val RMSE(b) | Epochs | Observation |
|------|------------|--------|-------------|
| 0 (TBNN) | 0.0845 | 694 | Baseline |
| 0.001 | 0.0852 | 729 | +0.8% — negligible difference |
| 0.01 | 0.0909 | 676 | +7.6% — penalty degrades accuracy |

**Analysis**: The unconstrained TBNN already produces near-realizable outputs. Post-hoc analysis of TBNN predictions on the validation set shows:
- Only 1.29% of points (309/23,967) violate the $b_{ii} \geq -1/3$ bound
- Only 0.004% of points (1/23,967) violate the $b_{ii} \leq 2/3$ bound
- Worst violation: $b_{ii} = -0.45$ vs limit of $-0.33$
- The total realizability penalty magnitude is 0.25% of the MSE loss

The tensor basis architecture inherently constrains the output space (Galilean invariance, symmetry, zero trace), making additional realizability penalties redundant. This is consistent with the literature's shift toward architectural constraints or post-hoc projection (Wu et al. 2018) rather than loss-based penalties.

### 6.3 TBRF Compact Variants

For solver integration experiments, compact TBRF variants with reduced tree counts were evaluated:

| Trees | Total Nodes | Binary Size | Val RMSE | vs TBNN |
|-------|------------|-------------|----------|---------|
| 1 | 282,902 | 5.7 MB | 0.0778 | 7.9% better |
| 5 | 1,432,612 | 28.7 MB | 0.0678 | 19.8% better |
| 10 | 2,797,130 | 55.9 MB | 0.0650 | 23.1% better |
| 200 | 55,051,026 | 1,101 MB | 0.0637 | 24.6% better |

10 trees capture 96% of the full TBRF accuracy improvement over TBNN. However, even the 1-tree variant requires 5.7 MB and involves branching tree traversals that are poorly suited for GPU execution, compared to the TBNN's 196 KB of dense matrix multiplies.

### 6.4 Training Curves

**MLP (5→32→32→1)**: Trained for full 1000 epochs. Best RMSE at epoch ~970. LR cycled through 3 cosine periods (T=200, T=400, T=400 partial). Continued improving slowly throughout.

**MLP-Large (5→128→128→128→128→1)**: Early-stopped at epoch 344. Converged quickly due to higher capacity. Val loss oscillated between cosine restarts but best checkpoint captured the minimum.

**TBNN (5→64→64→64→10)**: Early-stopped at epoch 694. Steady improvement through 3 cosine periods. Train loss 0.008, val loss 0.007 at best — no significant overfitting.

**PI-TBNN (beta=0.001)**: Early-stopped at epoch 729. Nearly identical trajectory to unconstrained TBNN — the penalty at this weight is effectively zero.

**TBRF**: No iterative training. Forest fitting took ~500s on 4 CPU cores.


## 7. Weight Export Format

### 7.1 Neural Networks (MLP, TBNN)

Weights exported as human-readable text files with 10-digit scientific notation:

```
data/models/<model>_paper/
  layer0_W.txt    # Weight matrix, shape [out_features, in_features]
  layer0_b.txt    # Bias vector, shape [out_features]
  layer1_W.txt    # ...
  layer1_b.txt
  ...
  input_means.txt # Z-score means, shape [5]
  input_stds.txt  # Z-score stds, shape [5]
  metadata.json   # Architecture, training info
```

Layer numbering corresponds to weight layers only (activation functions are not counted). For TBNN with architecture 5→64→64→64→10:
- Layer 0: W=[64,5], b=[64], followed by Tanh
- Layer 1: W=[64,64], b=[64], followed by Tanh
- Layer 2: W=[64,64], b=[64], followed by Tanh
- Layer 3: W=[10,64], b=[10], linear output

### 7.2 TBRF Compact Binary

```
data/models/tbrf_{1,5,10}t_paper/
  trees.bin          # Flat binary, all trees packed
  tree_offsets.txt   # Index: basis_idx tree_idx start_node n_nodes
  input_means.txt    # Z-score means
  input_stds.txt     # Z-score stds
  metadata.json      # Format spec
```

**trees.bin layout**:
```
Header:  [total_nodes: int32] [n_basis: int32] [n_trees: int32]
Data:    [children_left:  int32 x total_nodes]
         [children_right: int32 x total_nodes]
         [feature:        int32 x total_nodes]
         [threshold:      float32 x total_nodes]
         [value:          float32 x total_nodes]
```

Leaf nodes have `children_left = children_right = -1`. The `feature` field at leaf nodes is -2 (scikit-learn convention). Node indices in `children_left` and `children_right` are global (offset by the tree's start position in the flat array).

**Inference pseudocode**:
```
for each basis coefficient n = 0..9:
    predictions = []
    for each tree t in trees for coefficient n:
        node = tree_offset[n][t]
        while children_left[node] != -1:
            if input[feature[node]] <= threshold[node]:
                node = children_left[node]
            else:
                node = children_right[node]
        predictions.append(value[node])
    g[n] = mean(predictions)

b_ij = sum over n: g[n] * T^(n)_ij
```


## 8. References

1. Pope, S.B. (1975). A more general effective-viscosity hypothesis. *Journal of Fluid Mechanics*, 72(2), 331-340.
2. Ling, J., Kurzawski, A., & Templeton, J. (2016). Reynolds averaged turbulence modelling using deep neural networks with embedded invariance. *Journal of Fluid Mechanics*, 807, 155-166.
3. Wu, J.L., Xiao, H., Sun, R., & Wang, Q. (2018). Physics-informed machine learning approach for augmenting turbulence models. *Physical Review Fluids*, 3(7), 074602.
4. Kaandorp, M.L.A. & Dwight, R.P. (2020). Data-driven modelling of the Reynolds stress tensor using random forests with invariance. *Computers & Fluids*, 202, 104497.
5. McConkey, R., Yee, E., & Lien, F.S. (2021). A curated dataset for data-driven turbulence modelling. *Scientific Data*, 8, 255.
6. Menter, F.R. (1994). Two-equation eddy-viscosity turbulence models for engineering applications. *AIAA Journal*, 32(8), 1598-1605.
7. Kingma, D.P. & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR 2015*.
8. Lumley, J.L. (1978). Computational modeling of turbulent flows. *Advances in Applied Mechanics*, 18, 123-176.
