# Paper Roadmap: NN Turbulence Closures — Cost-Accuracy Tradeoff

## 1. Paper Narrative

### Thesis

Data-driven turbulence closures promise improved accuracy over classical RANS models, but their computational cost in a coupled CFD solver is rarely quantified. We train five model architectures on the same dataset with the same protocol, implement them all in a single GPU-accelerated incompressible Navier-Stokes solver, and measure both prediction accuracy and wall-clock cost. The result is a Pareto frontier mapping the cost-accuracy tradeoff from cheap algebraic models through transport-equation RANS to neural network closures.

### Key Questions

1. **How much accuracy do you gain** by replacing a classical RANS closure (k-omega SST, EARSM) with a neural network (MLP, TBNN)?
2. **How much does it cost?** What is the wall-clock overhead per time step, and how does it scale with grid size?
3. **Is there a sweet spot?** Can a small MLP achieve most of the accuracy of a large TBNN at a fraction of the cost?
4. **Do physics-informed constraints help?** Does enforcing realizability via loss penalties improve predictions, or does the tensor basis architecture already embed sufficient physics?
5. **Are random forests viable in a solver?** The TBRF is the most accurate offline, but can it run at acceptable cost?

### Positioning in the Literature

Most existing work evaluates NN closures **a priori only** (offline, on held-out data). A smaller body of work does **a posteriori** evaluation (coupled into a solver), but rarely compares multiple architectures head-to-head in the same solver with the same training data. Our contribution is the systematic, apples-to-apples comparison across the full model zoo — from zero-equation algebraic to random forest — with both offline and online metrics, on GPU hardware.

Key comparisons to prior work:
- **Ling et al. (2016)**: Introduced TBNN, a priori only. We reproduce their architecture and evaluate a posteriori.
- **Kaandorp & Dwight (2020)**: Introduced TBRF, a priori only. We implement it in a solver for the first time.
- **McConkey et al. (2021)**: Provided the dataset. We use their data with a standardized split.
- **Wu et al. (2018)**: PI approach with post-hoc projection. We test loss-based realizability penalties and show they don't help.
- **GEP (Weatheritt & Sandberg 2016)**: Algebraic model from genetic programming. Already in our solver as a classical baseline.

---

## 2. Completed Work

### 2.1 Model Training

| Item | Status | Location |
|------|--------|----------|
| McConkey dataset (902K points, 38 cases) | Done | `mcconkey_data/` |
| Case-holdout split (TBKAN 2025 protocol) | Done | `scripts/paper/train_all_models.py` |
| Feature engineering (5 Pope invariants, 10 tensor basis) | Done | GPU-accelerated via PyTorch |
| MLP (5→32→32→1) trained | Done | `data/models/mlp_paper/` |
| MLP-Large (5→128→128→128→128→1) trained | Done | `data/models/mlp_large_paper/` |
| TBNN (5→64→64→64→10) trained | Done | `data/models/tbnn_paper/` |
| PI-TBNN beta sweep (0.001, 0.01) | Done | `data/models/pi_tbnn_paper/` |
| TBRF (200 trees, depth 20) trained | Done | `data/models/tbrf_paper/` |
| TBRF compact exports (1, 5, 10 trees) | Done | `data/models/tbrf_{1,5,10}t_paper/` |
| Training methodology documented | Done | `docs/paper/training_methodology.md` |

### 2.2 Solver Infrastructure

| Item | Status | Location |
|------|--------|----------|
| MLP C++ inference (GPU) | Done | `src/turbulence_nn.cpp` |
| TBNN C++ inference (GPU) | Done | `src/turbulence_nn.cpp` |
| Solver profiling (9 models × 3 flow cases) | Done | `artifacts/profiling_results.md` |
| Paper experiment configs (channel × 4 grids, hills × 4 grids) | Done | `examples/paper_experiments/` |
| 10 RANS/LES models in solver | Done | See `TurbulenceModelType` enum |

### 2.3 Key Results Available

**A priori (validation set RMSE):**

| Model | Val RMSE(b) | Parameters | Weights Size |
|-------|------------|------------|-------------|
| TBRF (200 trees) | 0.0637 | ~55M nodes | 3.3 GB |
| TBRF (10 trees) | 0.0650 | ~2.8M nodes | 56 MB |
| TBNN | 0.0845 | 9,354 | 196 KB |
| PI-TBNN (β=0.001) | 0.0852 | 9,354 | 196 KB |
| MLP-Large | 0.1045 | 50,049 | 896 KB |
| MLP | 0.1096 | 1,249 | 56 KB |

**Solver profiling (cylinder Re=100, L40S, 50K steps):**

| Model | Turb Update (s) | Total Step (s) | Overhead |
|-------|----------------|---------------|----------|
| Baseline | 2.33 | 45.0 | — |
| k-omega | 0.47 | 45.1 | +0.2% |
| SST | 1.26 | 46.9 | +4.2% |
| EARSM (best) | 1.46 | 47.1 | +4.7% |
| NN-MLP | 12.78 | 55.6 | +23.6% |
| NN-TBNN | 85.55 | 129.0 | +186.7% |

**PI-TBNN finding:** Realizability penalty is either negligible (β=0.001, +0.8% vs TBNN) or harmful (β=0.01, +7.6%). The tensor basis already produces near-realizable outputs (1.3% violation rate on val set). This is a useful negative result.

---

## 3. Remaining Work

### Phase 1: Complete A Priori Evaluation

**Goal**: Finish the offline accuracy story with test set metrics and diagnostic plots.

**Tasks**:

- [ ] **Test set RMSE** for all models on held-out cases (periodic hills α=1.2, curved backward-facing step Re=13700)
- [ ] **Component-wise RMSE** breakdown: b_11, b_12, b_13, b_22, b_23, b_33 individually — shows which stress components each model predicts well/poorly
- [ ] **Per-case RMSE**: separate scores for PH α=1.2 (interpolation) vs CBFS Re=13700 (new geometry) — tests generalization
- [ ] **Scatter plots**: predicted vs true b_ij on test set (5 models × 2 key components, e.g., b_12 and b_11)
- [ ] **Lumley triangle plots**: map predicted anisotropy eigenvalues onto the Lumley triangle for each model — visual check of realizability
- [ ] **Error distribution**: histograms of pointwise error for each model — shows whether errors are uniform or concentrated in specific regions
- [ ] **Feature importance**: for TBRF, which invariants matter most? For TBNN, sensitivity analysis of g_n coefficients

**Estimated effort**: 1 session. Script to compute all metrics + generate plots.

**Output**: `scripts/paper/evaluate_apriori.py` → `results/paper/apriori/`

### Phase 2: TBRF C++ Inference

**Goal**: Implement decision tree traversal in the solver so TBRF variants can be profiled alongside neural networks.

**Tasks**:

- [ ] **C++ tree loader**: read `trees.bin` flat binary format (header + 5 arrays)
- [ ] **Inference function**: traverse trees, average predictions across trees per coefficient, reconstruct b_ij via tensor basis contraction
- [ ] **TurbulenceModel integration**: new `TurbulenceModelType::NNTBRF` enum, config parsing (`model = nn_tbrf`, `nn_tbrf_trees = 10`), factory creation
- [ ] **GPU strategy**: decision trees are branch-heavy and cache-unfriendly. Options: (a) CPU-only with GPU→CPU→GPU sync, (b) flatten to branchless lookup table, (c) GPU with warp divergence (expect poor utilization). Start with (a) to get correct results; measure cost.
- [ ] **Validation**: verify C++ predictions match Python predictions on the same inputs (should agree to float32 precision)
- [ ] **Benchmark**: profile 1/5/10-tree variants on cylinder/airfoil/hills

**Estimated effort**: 1-2 sessions.

**Expected outcome**: TBRF will be much slower than TBNN in the solver (despite better offline accuracy), which is exactly the tradeoff story.

### Phase 3: A Posteriori Evaluation

**Goal**: Run all models in the CFD solver on canonical flow cases and compare against DNS reference data.

**Flow cases**:

| Case | Re | Geometry | Grid | Reference Data | Key Metrics |
|------|----|----------|------|---------------|-------------|
| Channel | Re_τ=180 | Flat walls | 128×96×128 (grid A) | MKM (Moser, Kim, Moin 1999) | u⁺(y⁺), -⟨u'v'⟩(y), Re_τ |
| Periodic hills | Re_H=5600 | IBM hills | 128×64×64 (grid A) | Breuer et al. (2009) | Separation/reattachment x, u(y) profiles |
| Cylinder | Re=100 | IBM cylinder | From profiling runs | Established reference | C_d, C_l, St, wake profiles |

**Models to run** (for each flow case):
- No model (laminar baseline)
- k-omega SST
- EARSM-Pope (best EARSM variant)
- NN-MLP (56 KB, cheap)
- NN-TBNN (196 KB, expensive but accurate)
- NN-TBRF-10 (56 MB, most accurate, presumably most expensive)

**Tasks**:

- [ ] **Channel Re_τ=180**: run all 6 models to steady state on grid A. Extract u⁺(y⁺) profiles and Reynolds stress profiles. Compare against MKM DNS data.
- [ ] **Channel grid sensitivity**: run TBNN on grids A-D to show grid convergence (not all models needed, just one NN + one RANS)
- [ ] **Periodic hills Re_H=5600**: run all 6 models. Extract velocity profiles at key x-stations, measure separation/reattachment. Compare against Breuer et al.
- [ ] **Cylinder Re=100**: run all 6 models. Extract drag/lift coefficients, Strouhal number. Compare against established values.
- [ ] **Convergence monitoring**: for each run, record residual history, wall-clock time, final statistics
- [ ] **Profile extraction scripts**: automated post-processing to extract profiles from solver output

**Estimated effort**: 2-3 sessions (writing scripts, submitting SLURM jobs, post-processing). Each run is ~30 min on GPU for 30K steps.

**Output**: `results/paper/aposteriori/{channel,hills,cylinder}/{model_name}/`

**Critical considerations**:
- The NN models were trained on McConkey data (which includes periodic hills but not our exact Re). Channel and cylinder were not in the training set — this tests true generalization.
- Need to ensure the solver's RANS fields (gradU, k, epsilon) are available to the NN at each step for feature computation. This is already implemented for MLP/TBNN.
- For TBRF, feature computation (invariants + tensor basis) happens at every step — this cost must be included in the timing.

### Phase 4: Cost-Accuracy Synthesis

**Goal**: Produce the central figure and analysis of the paper.

**The Pareto Plot**:
- x-axis: computational cost (wall-clock time per step, or overhead fraction vs laminar baseline)
- y-axis: a posteriori prediction error (e.g., L2 norm of velocity profile error vs DNS)
- Each model is a point, with annotations for model size and architecture
- Identify the Pareto frontier — models that are not dominated by any other
- Show this for multiple flow cases (channel, hills, cylinder) to test robustness of the ranking

**Tasks**:

- [ ] **Define error metric**: L2 error in u⁺(y⁺) profile vs DNS? Integrated Reynolds stress error? Need a single scalar per model per case.
- [ ] **Normalize costs**: all runs on same GPU (L40S or H200), same grid, same number of steps
- [ ] **Generate Pareto plot**: matplotlib script, one plot per flow case, plus a combined plot
- [ ] **Scaling analysis**: how does NN inference cost scale with grid size? Run MLP and TBNN on grids A-D.
- [ ] **Memory analysis**: report peak GPU memory for each model
- [ ] **Table**: comprehensive results table (model, accuracy metrics, cost, memory, parameters)

**Expected narrative**: k-omega SST sits at (cheap, moderate accuracy). EARSM is slightly more expensive, slightly more accurate. MLP is moderately more expensive with potentially better accuracy for some flows. TBNN is expensive but more accurate. TBRF is most expensive and most accurate. The Pareto frontier reveals whether NNs offer a favorable tradeoff or whether classical models dominate.

### Phase 5: Paper Writing

**Target venue**: JCP, JFM, or Physics of Fluids (depending on emphasis — numerics, fluid mechanics, or applied ML).

**Proposed structure**:

#### Title
"Cost-Accuracy Tradeoffs for Neural Network Turbulence Closures in GPU-Accelerated Incompressible Flow Solvers"

#### Abstract (draft)
We systematically compare five data-driven turbulence closure architectures — multilayer perceptron (MLP), tensor basis neural network (TBNN), physics-informed TBNN (PI-TBNN), and tensor basis random forest (TBRF) — against classical RANS models in a GPU-accelerated incompressible Navier-Stokes solver. All models are trained on the same dataset (McConkey et al. 2021) with the same case-holdout protocol and evaluated both a priori (offline prediction accuracy) and a posteriori (coupled CFD accuracy and wall-clock cost). We find that [TBD based on results]. The TBNN offers the best accuracy-cost tradeoff, achieving [X]% lower prediction error than k-omega SST at [Y]% additional computational cost.

#### Section Outline

**1. Introduction** (~1.5 pages)
- RANS closures: hierarchy from algebraic to transport to nonlinear
- Data-driven closures: promise and challenges
- Gap: most evaluations are a priori only; cost rarely measured
- Contribution: systematic cost-accuracy comparison in a single solver

**2. Methods** (~4 pages)

*2.1 Governing equations and solver*
- Incompressible N-S, fractional-step projection, staggered MAC grid
- GPU acceleration via OpenMP target offload
- Poisson solver (FFT on GPU)
- IBM for complex geometries

*2.2 Turbulence closure models*
- Classical models in the solver: baseline, k-omega, SST, GEP, EARSM (3 variants)
- Neural network architectures: MLP, MLP-Large, TBNN, PI-TBNN
- Random forest: TBRF with compact tree variants
- Table of all models with parameter counts and architecture diagrams

*2.3 Training procedure*
- Dataset: McConkey et al. (2021), k-omega SST baseline
- Features: 5 Pope invariants, 10 tensor basis
- Split: TBKAN 2025 case-holdout protocol
- Training details: optimizer, LR schedule, early stopping (reference full spec in appendix/supplement)
- PI-TBNN: realizability penalty formulation and beta sweep

*2.4 Solver integration*
- Feature computation at each time step: gradU → S, Omega → invariants
- MLP/TBNN inference: GPU matrix multiplies
- TBRF inference: tree traversal (CPU with GPU sync, or GPU with warp divergence)
- Timing methodology: TIMED_SCOPE infrastructure, warm-up exclusion

**3. Results** (~5-6 pages)

*3.1 A priori evaluation*
- Validation and test set RMSE (table)
- Component-wise accuracy breakdown
- Scatter plots: predicted vs true b_ij
- Lumley triangle visualization
- PI-TBNN beta sweep: negative result

*3.2 A posteriori evaluation*
- Channel flow: u⁺(y⁺) and Reynolds stress profiles
- Periodic hills: separation/reattachment, velocity profiles
- Cylinder: drag, lift, Strouhal
- Grid sensitivity study

*3.3 Computational cost*
- Wall-clock profiling table (all models × all cases)
- Breakdown: feature computation, inference, total overhead
- Scaling with grid size
- Memory footprint

*3.4 Cost-accuracy tradeoff*
- Pareto plot: THE figure
- Pareto frontier analysis
- Scaling: how does the tradeoff change with problem size?

**4. Discussion** (~1.5 pages)
- When should you use each model class?
- Why PI-TBNN doesn't help (architectural constraints are sufficient)
- TBRF: offline ceiling, impractical online — implications for architecture choice
- Limitations: training data dependence, extrapolation to unseen Re/geometries
- Comparison to prior a priori-only evaluations — does ranking change a posteriori?

**5. Conclusions** (~0.5 pages)
- Summary of key findings
- Practical recommendation: which model for which use case
- Future work: online training, transfer learning, hybrid models

**Appendix / Supplementary Material**
- Full training methodology (from `training_methodology.md`)
- Complete profiling tables for all GPU types
- Additional flow case results
- Convergence histories

#### Figures List

| # | Description | Type | Data Source |
|---|-------------|------|-------------|
| 1 | Model architecture diagrams (MLP, TBNN, TBRF) | Schematic | Hand-drawn / tikz |
| 2 | Training/validation curves for each model | Line plot | Training logs |
| 3 | Test set scatter: predicted vs true b_ij (grid of panels) | Scatter | A priori evaluation |
| 4 | Lumley triangle with predictions from each model | Scatter on triangle | A priori evaluation |
| 5 | PI-TBNN beta sweep: RMSE vs beta | Bar/line | Beta sweep results |
| 6 | Channel u⁺(y⁺) profiles: all models vs DNS | Line plot | A posteriori runs |
| 7 | Channel Reynolds stress profiles: all models vs DNS | Line plot | A posteriori runs |
| 8 | Periodic hills velocity profiles at key x-stations | Line plot (multi-panel) | A posteriori runs |
| 9 | Periodic hills separation bubble visualization | Contour/streamline | A posteriori runs |
| 10 | Cylinder drag coefficient convergence | Line plot | A posteriori runs |
| 11 | **Cost-accuracy Pareto plot** (THE figure) | Scatter with annotations | Combined |
| 12 | Inference cost scaling with grid size | Line/bar | Grid sensitivity runs |
| 13 | Timing breakdown pie charts (feature compute vs inference vs solver) | Pie/stacked bar | Profiling |

#### Tables List

| # | Description | Data Source |
|---|-------------|-------------|
| 1 | Model summary (architecture, parameters, size, deployability) | Training |
| 2 | A priori RMSE: validation and test, overall and per-component | A priori evaluation |
| 3 | A posteriori error metrics for each model × flow case | A posteriori runs |
| 4 | Wall-clock profiling: turb update, total step, overhead % | Profiling |
| 5 | TBRF tree count sweep: accuracy vs size vs cost | Training + profiling |
| 6 | PI-TBNN beta sweep results | Beta sweep |

---

## 4. Potential Reviewer Concerns and Mitigations

| Concern | Mitigation |
|---------|------------|
| "Only tested on simple flows" | Channel, hills (separated), cylinder (bluff body) cover a range of physics. McConkey test set includes CBFS (new geometry). |
| "Training data is from k-omega SST — what about other RANS baselines?" | McConkey also provides k-epsilon data; could add as sensitivity test. SST is the most common industrial RANS model. |
| "NN inference could be faster with TensorRT/ONNX" | True — our comparison uses raw matrix multiplies. Note this as future work. Our costs are upper bounds. |
| "Grid is too coarse for DNS-quality" | We're not doing DNS — we're comparing RANS-level models on RANS-appropriate grids. Grid sensitivity study shows convergence. |
| "Only one GPU type" | We have profiling data on L40S, H100, and H200. Report primary results on one GPU, supplementary on others. |
| "MLP target (anisotropy magnitude) is not standard" | Acknowledge this — MLP predicts a scalar proxy, not the full tensor. This is deliberate: it's the simplest possible NN closure. Compare fairly by noting TBNN/TBRF predict the full tensor. |
| "No online/adaptive training" | Out of scope — we focus on the offline-trained, fixed-weight regime. Note as future work. |
| "PI-TBNN implementation is naive" | Acknowledge the literature has moved to architectural constraints. Our negative result with loss penalties is consistent with Wu et al. (2018). We tested multiple beta values to be thorough. |
| "TBRF in the solver is an unfair comparison (CPU tree traversal vs GPU matmul)" | This IS the point — even if you have the most accurate model, it must be efficient enough to deploy. The cost-accuracy tradeoff is the entire thesis. |
| "No uncertainty quantification" | Out of scope for this paper. Could add ensemble spread for TBRF (free — already have multiple trees). Note as future work. |

---

## 5. Data and Reference Sources

### Reference DNS/LES Data for A Posteriori Comparisons

| Flow Case | Reference | Data Availability |
|-----------|-----------|-------------------|
| Channel Re_τ=180 | Moser, Kim, Moin (1999) | `https://turbulence.oden.utexas.edu/data/MKM/chandata.tar.gz` |
| Periodic hills Re_H=5600 | Breuer et al. (2009) | ERCOFTAC database / paper tables |
| Cylinder Re=100 | Established literature | C_d≈1.35, C_l≈±0.34, St≈0.165 |

### McConkey Dataset Cases

38 total cases across 4 geometries. Our split:

| Split | Cases | Points | Purpose |
|-------|-------|--------|---------|
| Train (18) | SD Re={1100-3500 excl 2000}, PH α={0.5,1.0,1.5}, CDC Re=12600 | 271,924 | Model fitting |
| Val (2) | SD Re=2000, PH α=0.8 | 23,967 | Hyperparameter selection, early stopping |
| Test (2) | PH α=1.2, CBFS Re=13700 | 51,844 | Reported a priori accuracy |
| Unused (16) | Flat plate (10 cases), SD Re=1800, convdiv20580, h20/h26/h31/h38/h42 | ~555K | Available for future work |

Note: 16 cases are unused. The flat plate (fp_*) cases and the additional square duct/hills cases could be used for additional training or testing in future work.

---

## 6. Execution Plan

### Priority Order

| Priority | Phase | Effort | Blocking? |
|----------|-------|--------|-----------|
| 1 | A priori test set evaluation | 1 session | No — can run immediately |
| 2 | TBRF C++ inference | 1-2 sessions | Blocks Phase 3 for TBRF |
| 3 | A posteriori: channel flow | 1 session | Blocks Phase 4 |
| 4 | A posteriori: periodic hills | 1 session | Blocks Phase 4 |
| 5 | A posteriori: cylinder | 1 session | Blocks Phase 4 |
| 6 | Cost-accuracy synthesis + Pareto plot | 1 session | Needs all of above |
| 7 | Paper draft | 2-3 sessions | Needs all results |
| 8 | Revisions + polishing | 1-2 sessions | — |

### Non-Blocking Parallelism

- Phase 1 (a priori) and Phase 2 (TBRF C++) are independent — can run in parallel
- Channel, hills, and cylinder a posteriori runs are independent — can submit all SLURM jobs simultaneously
- Paper writing (intro, methods) can start before all results are in

---

## 7. File Index

| File | Contents |
|------|----------|
| `docs/paper/paper_roadmap.md` | This file |
| `docs/paper/training_methodology.md` | Full training specification |
| `artifacts/profiling_results.md` | Solver timing data |
| `results/paper/training/full_5266424.out` | Final 5-model training log |
| `results/paper/training/pi_sweep_5300440.out` | PI-TBNN beta sweep log |
| `data/models/training_summary.json` | Machine-readable results |
| `data/models/*_paper/` | Trained model weights |
| `scripts/paper/train_all_models.py` | Training pipeline |
| `examples/paper_experiments/` | CFD experiment configs |
