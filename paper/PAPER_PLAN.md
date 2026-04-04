# Paper Plan: 21 Turbulence Closures on GPU

## Working Title
"Pareto Analysis of Neural Network Turbulence Closures in a GPU-Accelerated Incompressible Flow Solver"

## Target
Journal of Computational Physics (or Computer Methods in Applied Mechanics and Engineering)

---

## Abstract (bullet points for draft)

- We present the first systematic comparison of 21 turbulence closures — 9 classical RANS and 12 neural network models — evaluated live within a GPU-accelerated incompressible Navier-Stokes solver
- Models span: algebraic (mixing-length), transport (k-omega, SST), nonlinear algebraic (EARSM x3, GEP), full Reynolds stress (RSM-SSG), scalar NN (MLP x3), tensor-basis NN (TBNN x3, PI-TBNN x3), and tensor-basis random forest (TBRF x3)
- All models are trained on the same dataset (McConkey et al. 2022) and evaluated on 4 canonical flows: cylinder Re=100, periodic hills Re=5600, square duct Re_b=3500, and sphere Re=200
- We report both a priori accuracy (prediction error on test data) and a posteriori accuracy (flow field error vs DNS), alongside GPU computational cost
- **Key finding 1**: Scalar NN closures (MLP) are strictly dominated in Pareto space — they are more expensive than SST AND less accurate. MLP predicts eddy viscosity 130x larger than SST, producing massively over-diffusive solutions
- **Key finding 2**: Tensor-basis NN closures (TBNN, PI-TBNN) occupy the Pareto frontier on anisotropy-dominated flows (square duct), producing secondary flow absent in Boussinesq models. Cost premium: 2-10x SST
- **Key finding 3**: Tensor closures diverge on separated flows (periodic hills) with explicit time integration, due to anti-diffusive anisotropic corrections. This is a fundamental limitation, not a numerical artifact
- **Key finding 4**: A priori accuracy does not predict a posteriori performance — TBRF has lowest test RMSE but cannot be evaluated online due to inference cost; MLP has reasonable a priori accuracy but catastrophic a posteriori performance
- Practical recommendation: for GPU-accelerated RANS, use SST (cost-effective baseline) or TBNN (when anisotropy matters and flow is attached). Avoid MLP-type scalar corrections entirely.

---

## Section-by-Section Plan

### 1. Introduction [DONE - 67 lines, polished]
- Motivation: data-driven closures promise accuracy but cost is unknown
- Gap: no systematic GPU cost-accuracy comparison across model families
- Contributions (5 items): framework, 21-model comparison, cost anatomy, a priori/a posteriori, Pareto analysis

### 2. Methods

#### 2.1 Governing Equations & Solver [DONE - 315 lines, polished]
- Incompressible NS, fractional-step projection, staggered MAC grid
- GPU acceleration via OpenMP target offload
- FFT Poisson, ghost-cell IBM
- Validation against CaNS/CaLES benchmarks

#### 2.2 Turbulence Closures [DONE - in methods_closures.tex]
- All 21 models described
- Classical: mixing-length, k-omega, SST, EARSM (3 variants), GEP, RSM-SSG
- NN: MLP (3 sizes), TBNN (3 sizes), PI-TBNN (3 sizes), TBRF (3 sizes)
- Pope (1975) tensor basis, invariant inputs

#### 2.3 Training Methodology [DONE - in methods_training.tex]
- McConkey dataset, train/val/test split
- Architecture search, hyperparameter selection
- PI-TBNN realizability penalty (beta sweep finding)

#### 2.4 Steady-State Methodology [NEEDS WRITING]
- Pseudo-transient continuation: run IMEX solver until convergence
- Convergence criterion: max|u^{n+1} - u^n| / U_ref < tol
- Implicit y-diffusion (Thomas) removes diffusion CFL limit
- Warm-up protocol for turbulent cases (SST transport stabilization)
- Cost metric: wall time from initialization to convergence on H200 GPU

### 3. Results

#### 3.1 A Priori Evaluation [DONE - populated with real data]
- Validation + test RMSE for all 12 NN models
- Component-wise accuracy breakdown
- PI-TBNN: realizability penalty acts as regularizer (finding)
- TBRF: best a priori accuracy (RMSE 0.064) but offline only

#### 3.2 Computational Cost [MOSTLY DONE - 264 lines, 1 todo]
- Per-step GPU timing for all 21 models x 4 cases (H200 data)
- Infrastructure cost breakdown (Poisson, convection, diffusion, turbulence)
- NN cost scaling with model size and grid resolution
- **TODO**: Add wall-time-to-convergence data (needs converged runs)
- **TODO**: Add OpenFOAM SST baseline for calibration

#### 3.3 A Posteriori Evaluation [NEEDS DATA - 12+ todos]

**3.3.1 Cylinder Re=100 — "Do No Harm"**
- All 21 models stable (no divergence on unseen geometry)
- Cd comparison: None=1.37, SST=1.55, GEP=1.95, MLP=1.41, TBNN=1.57, RSM=1.37
- DNS reference: Cd=1.35, St=0.165
- Finding: GEP over-predicts (26% above SST), MLP under-predicts, tensor models cluster near SST
- Wake profiles at x/D = 1, 2, 3, 5

**3.3.2 Periodic Hills Re=5600 — Tensor Divergence**
- Boussinesq models (SST, Baseline, GEP, MLP): stable, produce attached/separated flow
- Tensor models (TBNN, PI-TBNN, TBRF, EARSM): diverge at ~4500 steps
- Root cause: anisotropic correction produces anti-diffusion in separated region
- tau_div ramp delays but cannot prevent divergence
- DNS reference: Breuer et al. (2009) — Cf, separation/reattachment
- **This is a fundamental contribution**: quantifies when tensor closures CANNOT be used with explicit solvers

**3.3.3 Square Duct Re_b=3500 — THE Differentiator**
- SST U_b=0.179, TBNN U_b≈0.30, MLP U_b≈0.05, RSM U_b=0.40
- DNS reference: Pinelli & Uhlmann (2010) — secondary flow, wall shear
- TBNN produces secondary flow (max|v|≈0.013) absent in SST (max|v|≈0.009)
- MLP nu_t = 130x SST → kills all velocity → strictly worse than SST
- Finding: only tensor closures improve on SST for anisotropy-dominated flows
- Cross-section plots of secondary flow (y-z plane at mid-x)

**3.3.4 Sphere Re=200 — 3D IBM Validation**
- 192x128x128 grid, ghost-cell IBM
- Cd comparison across models
- DNS reference: Tomboulides et al. (2000) Cd≈0.77
- Tests 3D stability of all closures on unseen geometry

#### 3.4 Pareto Analysis — THE SPLASH FIGURE
- x-axis: wall time to convergence (GPU-seconds on H200)
- y-axis: L2 error vs DNS (duct, the differentiating case)
- 21 data points, color-coded by family:
  * Blue circles: classical RANS (SST, k-omega, RSM, etc.)
  * Red triangles: scalar NN (MLP x3)
  * Green squares: tensor NN (TBNN x3, PI-TBNN x3)
  * Orange diamonds: other (GEP, Baseline, TBRF — offline reference)
- Three visible clusters:
  1. Classical RANS: low cost, moderate accuracy (SST is the knee)
  2. Tensor NN: moderate cost, better accuracy (TBNN is the frontier)
  3. Scalar NN: moderate cost, WORSE accuracy (strictly dominated)
- Annotate: "MLP: more expensive AND less accurate than SST"
- Include OpenFOAM SST as star marker for calibration
- **Secondary Pareto plots**: one per case (cylinder, hills, duct, sphere)

### 4. Discussion [PARTIALLY DONE - 101 lines, 1 todo]
- Why MLP fails: Boussinesq assumption cannot capture anisotropy; overly large nu_t kills velocity
- Why TBNN succeeds on duct but fails on hills: training data is 3D duct DNS, hills is 2D separated
- PI-TBNN: realizability penalty as regularizer (a priori helps, a posteriori unclear)
- TBRF: best a priori but impractical online (tree traversal not GPU-friendly)
- A priori ≠ a posteriori: the disconnect and why it matters for deployment
- **TODO**: Add CPU vs GPU cost comparison
- **TODO**: Compare a priori accuracy to Ling (2016) and Kaandorp (2020)

### 5. Conclusions [NEEDS UPDATE - 60 lines, 4 todos]
- Bullet-point conclusions with quantitative claims:
  1. Scalar NN closures (MLP) add no value over SST: higher cost, lower accuracy
  2. Tensor NN closures (TBNN) improve accuracy on anisotropy-dominated flows at 2-10x cost
  3. Tensor closures are fundamentally limited to attached flows with explicit solvers
  4. A priori accuracy is a poor predictor of a posteriori performance
  5. GPU acceleration makes 21-model comparison feasible in hours, not weeks
- Practical recommendations for practitioners
- Future work: implicit solvers for tensor closures, multi-fidelity training data, online adaptation

---

## Figures Plan (ordered by importance)

1. **Pareto plot** (Fig. 1 or last results fig) — cost vs accuracy, all 21 models, duct case
2. **Duct cross-sections** — secondary flow comparison (SST vs TBNN vs MLP vs DNS)
3. **Hills Cf(x)** — Boussinesq models vs DNS, noting tensor divergence
4. **Cylinder wake profiles** — all models vs DNS
5. **Cost breakdown** — stacked bar chart, per-model GPU time
6. **A priori RMSE** — bar chart, all 12 NN models (already done)
7. **Tensor divergence** — time series showing exponential blowup on hills
8. **Sphere Cd** — bar chart, all stable models vs DNS
9. **Training curves / architecture search** — supplementary

---

## Data Needed (blocking)

| Data | Source | Status | Blocks |
|------|--------|--------|--------|
| Duct DNS (Pinelli) | Download/digitize from paper | NOT STARTED | Pareto y-axis |
| Hills DNS (Breuer) | Download/digitize from paper | NOT STARTED | Hills Cf plot |
| Cylinder DNS | Standard reference data | NOT STARTED | Cylinder Cd table |
| Sphere DNS | Tomboulides/Johnson-Patel | NOT STARTED | Sphere Cd table |
| Converged QoI | Extend production runs | PARTIAL (10K steps) | All results tables |
| OpenFOAM SST | Run simpleFoam | NOT STARTED | Cost calibration |

---

## Timeline Estimate

| Week | Deliverables |
|------|-------------|
| 1 | DNS data collected, convergence runs submitted, OpenFOAM cases set up |
| 2 | Converged results in hand, Pareto plot drafted, results_aposteriori populated |
| 3 | Paper complete draft, all figures, internal review |
| 4 | Polish, submit |

---

## What Makes This Paper Splash

1. **Scale**: 21 models — nobody has compared even 5 this way
2. **The MLP negative result**: field has invested heavily in MLP closures; showing they're strictly dominated is important and somewhat controversial
3. **Quantitative Pareto**: not just "model X is better" but "model X gives Y% accuracy improvement at Z% cost increase"
4. **GPU-native**: relevant to exascale computing push; first live NN closure comparison on GPU
5. **The tensor divergence finding**: provides concrete guidance on when NOT to use tensor closures
6. **A priori ≠ a posteriori**: challenges the common practice of evaluating closures only a priori
