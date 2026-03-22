# Paper Roadmap: NN vs RANS Cost-Accuracy Tradeoff

## Thesis

Neural network turbulence closures (MLP, TBNN) provide a cost-accuracy tradeoff between cheap algebraic RANS models and expensive DNS/LES. We quantify this tradeoff by training 5 model architectures on the McConkey (2021) dataset and evaluating them both a priori (offline accuracy) and a posteriori (coupled CFD accuracy and cost).

---

## Completed Work

### Training Pipeline
- [x] McConkey dataset downloaded and parsed (902K points, 38 cases)
- [x] Case-holdout split (TBKAN 2025 protocol): 18 train, 2 val, 2 test
- [x] Feature engineering: 5 Pope invariants + 10 tensor basis, GPU-accelerated
- [x] 5 models trained: MLP, MLP-Large, TBNN, PI-TBNN, TBRF
- [x] PI-TBNN beta sweep: {0.001, 0.01} — negative result (penalty negligible or harmful)
- [x] TBRF compact variants exported: 1, 5, 10 trees as flat binary
- [x] Full training methodology documented (`docs/paper/training_methodology.md`)
- [x] Training logs preserved (`results/paper/training/`)

### Solver Profiling (Previous Work)
- [x] Wall-clock profiling of all models: baseline, k-omega, SST, GEP, 3 EARSM, MLP, TBNN
- [x] Three flow cases: cylinder Re=100, airfoil Re=1000, hills Re=10595
- [x] GPU: L40S and H200
- [x] Results in `artifacts/profiling_results.md`

### Existing Solver Infrastructure
- [x] MLP C++ inference (GPU-accelerated, ~0.2ms/step)
- [x] TBNN C++ inference (GPU-accelerated, ~0.8ms/step cylinder, ~3.8ms/step airfoil)
- [x] Paper experiment configs: channel (4 grids) and hills (4 grids)

---

## Remaining Work

### 1. A Priori Evaluation (Offline)
**Goal**: Report prediction accuracy on held-out test cases.

- [ ] Compute test set RMSE for all models (PH alpha=1.2 + CBFS Re=13700)
- [ ] Component-wise RMSE breakdown (b_11, b_12, etc.)
- [ ] Scatter plots: predicted vs true b_ij for each model on test set
- [ ] Anisotropy invariant maps (Lumley triangle) for each model
- [ ] Realizability violation rates on test set

**Key figure**: Table of val/test RMSE for all models. Scatter plot grid (5 models × representative components).

### 2. TBRF C++ Inference
**Goal**: Implement tree traversal in the solver for compact TBRF variants.

- [ ] C++ tree loader: read `trees.bin` binary format
- [ ] Tree traversal kernel (CPU first, then GPU if feasible)
- [ ] Integrate into `TurbulenceModel` interface (same as TBNN)
- [ ] Validate C++ predictions match Python predictions
- [ ] Benchmark inference time for 1/5/10 tree variants

**Key question**: Can tree traversal be GPU-accelerated despite branching? Even if slow, it demonstrates the cost-accuracy tradeoff.

### 3. A Posteriori Evaluation (Coupled CFD)
**Goal**: Run trained models in the CFD solver and compare flow predictions.

- [ ] Channel flow Re_tau=180: velocity profiles u(y), Reynolds stress profiles
- [ ] Hills flow Re=10595: separation/reattachment points, velocity profiles
- [ ] Cylinder flow Re=100: drag/lift coefficients, wake profiles
- [ ] Compare against: DNS reference, k-omega SST, EARSM, baseline (no model)
- [ ] Grid sensitivity: run on grids A-D to show grid convergence

**Key figures**:
- Velocity profiles u+ vs y+ (channel)
- Reynolds stress profiles (channel)
- Streamlines / separation bubble (hills)
- Cd/Cl convergence (cylinder)

### 4. Cost-Accuracy Tradeoff Analysis
**Goal**: The central figure of the paper — plotting accuracy vs computational cost.

- [ ] x-axis: wall-clock time per step (or overhead vs baseline)
- [ ] y-axis: a posteriori error metric (e.g., L2 error in velocity profile vs DNS)
- [ ] Each model is a point: baseline, k-omega, SST, EARSM, MLP, MLP-Large, TBNN, TBRF-1, TBRF-5, TBRF-10
- [ ] Pareto frontier identification
- [ ] Annotation: model size (parameters/nodes), memory footprint

**Key figure**: Log-log scatter of cost vs accuracy. This is THE figure of the paper.

### 5. Paper Writing
- [ ] Outline with section structure
- [ ] Introduction: motivation, related work
- [ ] Methods: solver, training pipeline, models
- [ ] Results: a priori, a posteriori, cost-accuracy
- [ ] Discussion: when to use which model, limitations
- [ ] Conclusions

---

## Key Results So Far

### A Priori (Validation Set)

| Model | Val RMSE(b) | Parameters | Weights Size | Deployable |
|-------|------------|------------|-------------|------------|
| TBRF (200 trees) | 0.0637 | ~55M nodes | 3.3 GB | No (offline only) |
| TBRF (10 trees) | 0.0650 | ~2.8M nodes | 56 MB | Experimental |
| TBRF (5 trees) | 0.0678 | ~1.4M nodes | 29 MB | Experimental |
| TBRF (1 tree) | 0.0778 | ~283K nodes | 5.7 MB | Experimental |
| TBNN | 0.0845 | 9,354 | 196 KB | Yes |
| PI-TBNN (beta=0.001) | 0.0852 | 9,354 | 196 KB | Yes (no benefit) |
| MLP-Large | 0.1045 | 50,049 | 896 KB | Yes |
| MLP | 0.1096 | 1,249 | 56 KB | Yes |

### PI-TBNN Finding

Realizability penalty either negligible (beta=0.001, +0.8% vs TBNN) or harmful (beta=0.01, +7.6%). Root cause: tensor basis architecture already produces near-realizable outputs (1.3% violation rate). Previous poor results were caused by an L2 regularization bug (alpha=0.01 was 825x larger than MSE).

### Solver Profiling (50K Steps, L40S)

| Model | Turb Update (s) | Total Step (s) | Overhead vs Baseline |
|-------|----------------|---------------|---------------------|
| Baseline | 2.33 | 45.0 | — |
| k-omega | 0.47 | 45.1 | +0.2% |
| SST | 1.26 | 46.9 | +4.2% |
| EARSM | 1.46-1.63 | 47.1-47.4 | +4.7-5.3% |
| NN-MLP | 12.78 | 55.6 | +23.6% |
| NN-TBNN | 85.55 | 129.0 | +186.7% |

(Cylinder Re=100 on L40S. See `artifacts/profiling_results.md` for all cases.)

---

## File Index

| File | Contents |
|------|----------|
| `docs/paper/training_methodology.md` | Full training specification (dataset, features, architectures, hyperparameters, results) |
| `docs/paper/paper_roadmap.md` | This file |
| `artifacts/profiling_results.md` | Solver timing data for all models |
| `results/paper/training/full_5266424.out` | Final 5-model training log |
| `results/paper/training/pi_sweep_5300440.out` | PI-TBNN beta sweep log |
| `data/models/training_summary.json` | Machine-readable results summary |
| `data/models/*_paper/` | Trained model weights |
| `scripts/paper/train_all_models.py` | Training pipeline |
| `examples/paper_experiments/` | CFD experiment configs (channel/hills, 4 grids each) |
