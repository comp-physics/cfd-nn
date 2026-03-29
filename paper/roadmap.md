# Paper Roadmap: NN Turbulence Closures — Cost-Accuracy Tradeoff

> **NOTE (Mar 28, 2026):** This roadmap is partially outdated. The authoritative sources are:
> - **`paper/TODO.md`** — current task list, model matrix, case matrix, remaining work
> - **`memory/ibm_force_fix.md`** — critical IBM bug fixes and validated Cd results
>
> **Major updates (Mar 28):**
> - **ANISOTROPIC STRESS DIVERGENCE**: tau_ij was computed but NEVER used. Fixed with proper decomposition method.
>   - Cylinder: SST=1.72, EARSM=1.64, TBNN=1.80, TBRF=1.76, GEP=1.66 — ALL DIFFERENT
>   - Hills: EARSM=0.80, SST=0.73, k-omega=0.65 — three distinct models, all stable
>   - Key: must use SST nu_t in diffusion, model tau_ij as source (standard decomposition)
> - IBM force bugs fixed (ghost-cell force=0, bulk velocity controller, momentum-balance Cd)
> - dp/dx=-1.0 for hills was 8000× too large; config key `body` vs `ibm_body`
> - Hills Re=5600 (training data Re) now works with SST, k-omega, AND EARSM
> - All 4 cases validated: cylinder, sphere, duct, hills (with proper Re values)
> - **SST validated on hills**: reattach x/H=7.63 (DNS 4.83, lit SST 7-8) — solver matches published results
> - **2D anisotropy tiny**: TBNN/EARSM ≈ SST in 2D. Need 3D for meaningful NN correction.
> - **3D extension in progress**: Task 1 DONE (9 velocity gradients). Tasks 2-7 remaining.
> - **Full RSM (SSG/LRR-ω) to be implemented**: 7-eq transport model, most expensive classical RANS.
>   Provides "expensive traditional" comparison point on Pareto frontier.
> - **Duct is PRIMARY case**: SST(0) < EARSM(partial) ≤ RSM(partial) < TBNN(≈DNS?)
> - **Paper hierarchy**: cheap+wrong(SST) → cheap+partial(EARSM) → expensive+partial(RSM) → expensive+right?(TBNN)

## 1. Paper Narrative

### Thesis

Data-driven turbulence closures promise improved accuracy over classical RANS models, but their computational cost in a coupled CFD solver is rarely quantified. We train five model architectures on the same dataset with the same protocol, implement them all in a single GPU-accelerated incompressible Navier-Stokes solver, and measure both prediction accuracy and wall-clock cost on both CPU and GPU. The result is a Pareto frontier mapping the cost-accuracy tradeoff from cheap algebraic models through transport-equation RANS to neural network closures, with a detailed breakdown of *where* the computational cost comes from.

### Key Questions

1. **How much accuracy do you gain** by replacing a classical RANS closure (k-omega SST, EARSM) with a neural network (MLP, TBNN)?
2. **How much does it cost?** What is the wall-clock overhead per time step, and where does the time go (feature computation vs inference vs solver overhead)?
3. **Is there a sweet spot?** Can a small MLP achieve most of the accuracy of a large TBNN at a fraction of the cost?
4. **Do physics-informed constraints help?** Does enforcing realizability via loss penalties improve predictions, or does the tensor basis architecture already embed sufficient physics?
5. **Are random forests viable in a solver?** The TBRF is the most accurate offline, but can it run at acceptable cost?
6. **How does GPU vs CPU change the story?** Does the cost ranking change when the solver runs on CPU? What is the relative speedup for each closure type?
7. **What specific operations make NN closures expensive?** Can we identify and optimize the bottleneck (matmul, activation, feature computation, memory traffic)?

### Positioning in the Literature

Most existing work evaluates NN closures **a priori only** (offline, on held-out data). A smaller body of work does **a posteriori** evaluation (coupled into a solver), but rarely compares multiple architectures head-to-head in the same solver with the same training data. Our contribution is the systematic, apples-to-apples comparison across the full model zoo — from zero-equation algebraic to random forest — with both offline and online metrics, on GPU hardware.

Key comparisons to prior work:
- **Ling et al. (2016)**: Introduced TBNN, a priori only. We reproduce their architecture and evaluate a posteriori.
- **Kaandorp & Dwight (2020)**: Introduced TBRF, a priori only. We implement it in a solver for the first time.
- **McConkey et al. (2021)**: Provided the dataset. We use their data with a standardized split.
- **Wu et al. (2018)**: PI approach with post-hoc projection. We test loss-based realizability penalties and show they don't help.
- **GEP (Weatheritt & Sandberg 2016)**: Algebraic model from genetic programming. Already in our solver as a classical baseline.

---

## 2. Solver Architecture and Performance Validation

A critical part of the paper is establishing that our solver is well-optimized, so that the cost measurements are meaningful. A timing comparison in a slow solver would overstate the *relative* cost of NN closures (since the base solver time would be artificially inflated). We need to show our solver is competitive with established, heavily optimized codes.

### 2.1 Solver Overview

| Component | Implementation |
|-----------|---------------|
| Equations | Incompressible Navier-Stokes (fractional-step projection) |
| Spatial discretization | 2nd-order finite differences, staggered MAC grid |
| Time integration | Explicit RK3 (SSP), Euler, RK2 |
| Poisson solver | FFT via cuFFT (primary), multigrid (fallback) |
| GPU acceleration | OpenMP target offload (single code path for CPU and GPU) |
| Complex geometry | Immersed boundary method (direct forcing) |
| Compiler | NVHPC (nvc++) for GPU, GCC for CPU |
| Parallelism | MPI z-slab decomposition for multi-GPU |

### 2.2 Comparison Target: CaNS

**CaNS** (Costa 2018, *Computers & Mathematics with Applications*) is the gold-standard comparison for our solver class:
- Same method: staggered FD, FFT Poisson, fractional-step, channel flow DNS
- GPU-accelerated (CUDA Fortran / OpenACC with cuFFT)
- Widely cited, well-optimized, open-source

**Benchmark plan**: Run our solver and CaNS on the same problem (channel flow DNS, 256³ grid) on the same GPU (A100 or H100) and compare wall-clock time per step. This establishes our solver's competitiveness.

Other solvers for context (not head-to-head benchmarks):
- **AFiD** (van der Poel et al. 2015, Zhu et al. 2018): Same class, ~400-700 Mcells/s on V100
- **NekRS** (Fischer et al. 2022): Spectral elements, higher-order — not directly comparable but useful context for state-of-the-art GPU performance
- **STREAmS** (Bernardini et al. 2021): Compressible, ~1-2 Gcells/s on V100

### 2.3 Our Measured Throughput

Preliminary numbers from existing profiling data:

| Flow Case | Grid | GPU | ms/step | Mcells/s |
|-----------|------|-----|---------|----------|
| Cylinder Re=100 (baseline) | 128×64×64 (524K) | L40S | 0.90 | 583 |
| Airfoil Re=1000 (baseline) | 256×128×128 (4.2M) | L40S | 1.66 | 2,527 |
| Hills Re=10595 (baseline) | 128×64×64 (524K) | L40S | ~0.90 | ~583 |

The 2.5 Gcells/s at 4.2M cells is competitive with established solvers. The lower throughput at 524K cells is expected (GPU underutilized at small grids).

**TODO**: Run formal benchmarks at 256³ and 512³ for a clean comparison number.

### 2.4 CPU vs GPU Comparison

We need to report CPU results alongside GPU for two reasons:
1. Many practitioners don't have GPU access — CPU cost is relevant
2. The relative overhead of NN closures may differ on CPU (matmuls have different CPU/GPU scaling than stencil operations)

**Plan**: Run all models on CPU (same grid, same problem) using GCC build without GPU offload. This gives us a full CPU/GPU comparison matrix.

---

## 3. Computational Cost Anatomy

A unique contribution of this paper: not just *how much* each model costs, but *why*. This section breaks down exactly what operations each closure performs and how they translate to wall-clock time.

### 3.1 What Each Closure Does Per Time Step

**Algebraic baseline (mixing length)**:
- Read: velocity field (u, v, w), mesh spacing (dy)
- Compute: |du/dy| at each cell → nu_t = l_mix² |du/dy|
- Cost: 1 division + 1 multiply per cell
- Memory: reads 3 fields, writes 1 field
- GPU character: perfectly parallel, memory-bandwidth-bound

**k-omega transport**:
- Read: velocity (3 fields), k, omega, nu_t, mesh spacing
- Compute: convective + diffusive fluxes for k and omega equations, source/sink terms
- Solve: explicit time advance of 2 transport equations
- Write: k, omega, nu_t
- Cost: ~50 FLOPs/cell (stencil operations for convection/diffusion, source terms)
- GPU character: stencil-heavy, memory-bandwidth-bound

**SST k-omega**:
- Everything in k-omega, plus:
- Compute: F1, F2 blending functions (each requires wall distance, grad(k)·grad(omega))
- Cross-diffusion term: CDkω = 2·(1-F1)·σω2·(1/(omega))·grad(k)·grad(omega)
- Cost: ~80 FLOPs/cell (2.5x k-omega due to blending)
- GPU character: same as k-omega but more arithmetic per cell

**EARSM (Explicit Algebraic Reynolds Stress)**:
- Everything in SST, plus:
- Compute: normalized S_hat and Omega_hat tensors (3×3 matrix operations)
- Compute: Pope/WJ/GS algebraic relation for anisotropy tensor → a_ij
- Cost: ~120 FLOPs/cell (SST transport + 3×3 tensor algebra)
- GPU character: more compute-intensive per cell, better arithmetic intensity

**NN-MLP (5→32→32→1)**:
- Step 1: Compute velocity gradients (gradU, 9 components) — stencil operation
- Step 2: Compute features — S, Omega from gradU (3×3 matrix ops), normalize by k/epsilon, compute 5 invariants (trace of matrix products)
- Step 3: NN inference — 3 layers of batched matrix-vector multiply + tanh activation:
  - Layer 0: [5→32] = 160 multiplies + 32 adds + 32 tanh per cell
  - Layer 1: [32→32] = 1024 multiplies + 32 adds + 32 tanh per cell
  - Layer 2: [32→1] = 32 multiplies + 1 add per cell
  - Total: **1,249 FLOPs/cell** (just matmuls) + 64 tanh evaluations
- Step 4: Postprocess — write nu_t
- Memory: weights (56 KB, fits in L1 cache) + features buffer (5 doubles/cell) + output buffer (1 double/cell)
- GPU character: each cell is independent (embarrassingly parallel), but inner loop per cell is serial (layer-sequential). Parallelized over (cell, neuron) pairs, but the dot product over inputs is sequential per thread.

**NN-TBNN (5→64→64→64→10)**:
- Steps 1-2: Same as MLP (feature computation)
- Step 2b: Compute 10 Pope tensor basis tensors T^(n) — 10 × (3×3 matrix multiply chain) = ~900 FLOPs/cell
- Step 3: NN inference — 4 layers:
  - Layer 0: [5→64] = 320 mults + 64 adds + 64 tanh
  - Layer 1: [64→64] = 4096 mults + 64 adds + 64 tanh
  - Layer 2: [64→64] = 4096 mults + 64 adds + 64 tanh
  - Layer 3: [64→10] = 640 mults + 10 adds
  - Total: **9,354 FLOPs/cell** + 192 tanh evaluations
- Step 4: Tensor basis contraction — b_ij = sum(g_n * T^(n)_ij) = 60 multiplies + 60 adds per cell
- Total: ~10,300 FLOPs/cell (7.5x more than MLP, dominated by the two 64×64 hidden layers)
- Memory: weights (196 KB, fits in L2 cache) + features (5/cell) + basis tensors (60/cell) + output (6/cell)
- GPU character: same as MLP but much more work per cell. The 64×64 matmul has enough arithmetic intensity to potentially be compute-bound rather than memory-bound.

**NN-TBRF (10 trees, depth 20)**:
- Steps 1-2b: Same as TBNN (features + tensor basis)
- Step 3: Tree traversal — for each cell, for each of 10 coefficients, traverse 10 trees (depth ~20):
  - Each tree: ~20 comparisons + 20 branches per cell
  - Total: 10 coefficients × 10 trees × 20 levels = **2,000 branch operations per cell**
  - Memory: 56 MB of tree data — does NOT fit in any cache level
- Step 4: Tensor basis contraction (same as TBNN)
- GPU character: **terrible** — branching causes warp divergence, tree data is scattered in memory (no spatial locality), 56 MB blows through L2 cache. This is fundamentally a cache-miss-dominated workload on GPU.

### 3.2 Why NN Closures Are Expensive on GPU

The core issue is that GPUs are designed for data-parallel, regular workloads with high arithmetic intensity. NN inference per grid cell has two problems:

1. **Layer-sequential execution**: Each MLP layer depends on the previous layer's output. For a TBNN with 4 layers, this means 4 separate kernel launches (or 4 serial phases within one kernel). The parallelism is only over (cell, neuron) — the depth dimension is sequential.

2. **Activation functions**: `tanh()` is expensive on GPU (~20 cycles vs ~4 for multiply). With 192 tanh evaluations per cell for TBNN, this is a significant fraction of the cost.

3. **Feature computation overhead**: Computing 5 invariants and 10 tensor basis tensors requires ~40 3×3 matrix multiplies and 15 traces per cell. This is not trivially parallel — each cell does its own small linear algebra.

4. **Memory traffic**: weights must be broadcast to all threads. For MLP (56 KB), this fits in L1. For TBNN (196 KB), it fits in L2. For TBRF (56 MB), it blows cache entirely.

Contrast with stencil-based RANS closures (k-omega, SST), which naturally match GPU memory access patterns: each thread reads a compact neighborhood of the same large arrays, with perfect spatial locality and coalesced access.

### 3.3 Key Profiling Data (Already Available)

From `artifacts/profiling_results.md` — 50,000 steps on L40S:

**Cylinder Re=100:**

| Model | Turb Update (s) | Total Step (s) | Overhead vs Baseline | Turb % of Total |
|-------|----------------|---------------|---------------------|-----------------|
| baseline | 2.33 | 45.0 | — | 5.2% |
| k-omega | 0.47 | 45.1 | +0.2% | 1.0% |
| SST | 1.26 | 46.9 | +4.2% | 2.7% |
| EARSM-Pope | 1.46 | 47.1 | +4.7% | 3.1% |
| NN-MLP | 12.78 | 55.6 | +23.6% | 23.0% |
| NN-TBNN | 85.55 | 129.0 | +186.7% | 66.3% |

**Airfoil Re=1000 (larger grid):**

| Model | Turb Update (s) | Total Step (s) | Overhead vs Baseline | Turb % of Total |
|-------|----------------|---------------|---------------------|-----------------|
| baseline | 3.01 | 83.0 | — | 3.6% |
| k-omega | 0.58 | 86.0 | +3.6% | 0.7% |
| SST | 2.14 | 90.0 | +8.4% | 2.4% |
| EARSM-Pope | 2.80 | 90.8 | +9.4% | 3.1% |
| NN-MLP | 40.77 | 124.3 | +49.8% | 32.8% |
| NN-TBNN | 543.91 | 628.4 | +657.1% | 86.6% |

Key observations:
- TBNN goes from 66% to 87% of total step time as grid grows — it scales worse than the base solver
- MLP goes from 23% to 33% — also scales worse, but less dramatically
- Transport models (SST, EARSM) add only 3-10% overhead regardless of grid size — they scale with the solver
- **The NN inference is the bottleneck, not the feature computation** (the feature computation uses the same gradU stencil as EARSM)

### 3.4 Detailed Timing Breakdown Needed

**TODO**: Add sub-timers within the NN update to separate:
- Gradient computation (stencil on velocity field)
- Feature extraction (invariants from gradients)
- Tensor basis computation (10 matrix products)
- NN forward pass (layer-by-layer)
- Postprocessing (nu_t / tau_ij write-back)

This is already partially instrumented via `TIMED_SCOPE` (e.g., `nn_mlp_features_gpu`, `nn_mlp_inference_gpu`, `nn_mlp_postprocess_gpu`). Need to collect these per-phase timings in the paper runs.

### 3.5 Scaling Analysis Plan

How does cost scale with grid size? This answers: "If I double my grid, how much more does the NN cost?"

| Grid | Cells | Expected Behavior |
|------|-------|-------------------|
| 64³ | 262K | GPU underutilized, overhead from kernel launch dominates |
| 128³ | 2.1M | Sweet spot for GPU, good occupancy |
| 256³ | 16.8M | Memory-bandwidth-limited, NN weights fit in cache |
| 512³ | 134M | Large — may exceed GPU memory with full TBRF |

Plan: run baseline + MLP + TBNN + TBRF-10 on 64³, 128³, 256³ grids. Plot turb_update_time vs N_cells. Compare slope against theoretical O(N) (linear in cells — should be true if GPU is well-utilized).

---

## 4. Completed Work

### 4.1 Model Training

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

### 4.2 Solver Infrastructure

| Item | Status | Location |
|------|--------|----------|
| MLP C++ inference (GPU) | Done | `src/turbulence_nn_mlp.cpp` |
| TBNN C++ inference (GPU) | Done | `src/turbulence_nn_tbnn.cpp` |
| Solver profiling (9 models × 3 flow cases) | Done | `artifacts/profiling_results.md` |
| Paper experiment configs (channel × 4 grids, hills × 4 grids) | Done | `examples/paper_experiments/` |
| 10 RANS/LES models in solver | Done | See `TurbulenceModelType` enum |
| Sub-step timing instrumentation | Done | `TIMED_SCOPE` in all NN update functions |

### 4.3 Key Results Available

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

## 5. Remaining Work

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
- [ ] **GPU strategy**: decision trees are branch-heavy and cache-unfriendly. Options:
  - (a) CPU-only with GPU→CPU→GPU sync — simplest, correct, expected to be slow
  - (b) GPU with warp divergence — theoretically possible, expect poor SM utilization
  - (c) Flatten shallow trees to branchless lookup tables — only works for small trees
  - Start with (a) to get correct results; measure cost. Try (b) if (a) is impractically slow.
- [ ] **Validation**: verify C++ predictions match Python predictions on same inputs (should agree to float32 precision)
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

**Models to run** (for each flow case, on both CPU and GPU):
- No model (laminar baseline)
- k-omega SST
- EARSM-Pope (best EARSM variant)
- NN-MLP (56 KB, cheap)
- NN-TBNN (196 KB, expensive but accurate)
- NN-TBRF-10 (56 MB, most accurate, presumably most expensive)

**Tasks**:

- [ ] **Channel Re_τ=180**: run all 6 models to steady state on grid A (GPU + CPU). Extract u⁺(y⁺) profiles and Reynolds stress profiles. Compare against MKM DNS data.
- [ ] **Channel grid sensitivity**: run TBNN on grids A-D to show grid convergence (not all models needed, just one NN + one RANS)
- [ ] **Periodic hills Re_H=5600**: run all 6 models (GPU + CPU). Extract velocity profiles at key x-stations, measure separation/reattachment. Compare against Breuer et al.
- [ ] **Cylinder Re=100**: run all 6 models (GPU + CPU). Extract drag/lift coefficients, Strouhal number. Compare against established values.
- [ ] **Convergence monitoring**: for each run, record residual history, wall-clock time, final statistics
- [ ] **Profile extraction scripts**: automated post-processing to extract profiles from solver output

**Estimated effort**: 2-3 sessions (writing scripts, submitting SLURM jobs, post-processing). Each run is ~30 min on GPU for 30K steps. CPU runs will be slower — may need to reduce step count or grid size.

**Output**: `results/paper/aposteriori/{channel,hills,cylinder}/{model_name}/`

**Critical considerations**:
- The NN models were trained on McConkey data (which includes periodic hills but not our exact Re). Channel and cylinder were not in the training set — this tests true generalization.
- Need to ensure the solver's RANS fields (gradU, k, epsilon) are available to the NN at each step for feature computation. This is already implemented for MLP/TBNN.
- For TBRF, feature computation (invariants + tensor basis) happens at every step — this cost must be included in the timing.

### Phase 4: Solver Validation Against CaNS

**Goal**: Demonstrate our solver is competitive with established codes, so cost measurements are credible.

**Tasks**:

- [ ] **Install CaNS** on the same cluster (GPU partition)
- [ ] **Channel flow benchmark**: run both solvers on identical problem (256³, Re_τ=180, 1000 steps) on same GPU (A100 or H100)
- [ ] **Measure**: wall-clock per step, Mcells/s, breakdown (Poisson vs advection vs diffusion)
- [ ] **Report**: our solver achieves X% of CaNS throughput (or exceeds it — CaNS uses Fortran, we use C++/OpenMP)

**Fallback**: if CaNS installation is problematic, cite their published numbers and compare. Their papers report ~500-800 Mcells/s on A100 for channel flow at moderate grids. Our preliminary numbers show ~2.5 Gcells/s at 4.2M cells on L40S (laminar, baseline model). Need to verify on the same GPU/grid size for a fair comparison.

**Why this matters**: reviewers will ask "is this solver representative?" A comparison against CaNS (or at minimum a citation of their numbers with our equivalent measurement) answers this.

### Phase 5: Cost-Accuracy Synthesis

**Goal**: Produce the central figures and analysis of the paper.

**The Pareto Plot (THE figure)**:
- x-axis: computational cost (wall-clock time per step, or overhead fraction vs laminar baseline)
- y-axis: a posteriori prediction error (e.g., L2 norm of velocity profile error vs DNS)
- Each model is a point, with annotations for model size and architecture
- Identify the Pareto frontier — models that are not dominated by any other
- Show for multiple flow cases (channel, hills, cylinder) to test robustness of the ranking
- **Two versions**: one for GPU, one for CPU — show how the frontier shifts

**The Cost Breakdown Figure**:
- Stacked bar chart: for each model, show time spent in each phase
  - Gradient computation (stencil)
  - Feature extraction (invariants)
  - Tensor basis computation (10 matrix products) — only for TBNN/TBRF
  - NN forward pass (matmul + activation)
  - Transport equation solve — only for SST/EARSM
  - Postprocessing (write nu_t / tau_ij)
- This shows exactly *where* the cost comes from for each model class

**The Scaling Plot**:
- NN inference cost vs grid size (64³ to 256³ or 512³)
- Compare scaling slope: does NN cost scale linearly with cells? (should, if GPU is well-utilized)
- Overlay: base solver cost scaling (should be O(N) for stencils, O(N log N) for FFT Poisson)
- This shows whether the NN overhead is constant-fraction or grows with problem size

**Tasks**:

- [ ] **Define error metric**: L2 error in u⁺(y⁺) profile vs DNS? Integrated Reynolds stress error? Need a single scalar per model per case.
- [ ] **Normalize costs**: all runs on same GPU (L40S or H200), same grid, same number of steps. Separate CPU runs.
- [ ] **Generate Pareto plot**: matplotlib script, one plot per flow case, plus a combined plot. GPU and CPU versions.
- [ ] **Generate cost breakdown**: stacked bar from sub-timer data
- [ ] **Scaling analysis**: run baseline + MLP + TBNN + TBRF-10 on grids 64³, 128³, 256³. Plot turb_update_time vs N_cells.
- [ ] **Memory analysis**: report peak GPU memory for each model
- [ ] **Operation count analysis**: compute theoretical FLOPs/cell for each model, compare to measured throughput → derive arithmetic intensity and compare to roofline model
- [ ] **Table**: comprehensive results table (model, accuracy metrics, cost GPU, cost CPU, memory, parameters, FLOPs/cell)

**Expected narrative**:
- k-omega SST sits at (cheap, moderate accuracy) — adds <5% overhead
- EARSM is similar cost, potentially better accuracy for anisotropic flows
- MLP is a moderate step up in cost (+25-50%) with potentially better accuracy
- TBNN is expensive (+200-600%) but much more accurate
- TBRF is most expensive and most accurate — but the cost is prohibitive for online use
- The Pareto frontier identifies whether MLP or TBNN offers a favorable tradeoff
- On CPU, the relative cost picture may change (matmuls have different CPU/GPU scaling)

### Phase 6: Paper Writing

**Target venue**: JCP, JFM, or Physics of Fluids (depending on emphasis — numerics, fluid mechanics, or applied ML).

**Proposed structure**:

#### Title
"Cost-Accuracy Tradeoffs for Neural Network Turbulence Closures in GPU-Accelerated Incompressible Flow Solvers"

#### Abstract (draft)
We systematically compare five data-driven turbulence closure architectures — multilayer perceptron (MLP), tensor basis neural network (TBNN), physics-informed TBNN (PI-TBNN), and tensor basis random forest (TBRF) — against classical RANS models in a GPU-accelerated incompressible Navier-Stokes solver. All models are trained on the same dataset (McConkey et al. 2021) with the same case-holdout protocol and evaluated both a priori (offline prediction accuracy) and a posteriori (coupled CFD accuracy and wall-clock cost). We provide a detailed computational anatomy of each closure, showing exactly which operations — matrix multiplies, activation functions, tree traversals, transport equation solves — dominate the cost. [TBD: key finding]. We find that [TBD]. The TBNN offers the best accuracy-cost tradeoff, achieving [X]% lower prediction error than k-omega SST at [Y]% additional computational cost on GPU and [Z]% on CPU.

#### Section Outline

**1. Introduction** (~1.5 pages)
- RANS closures: hierarchy from algebraic to transport to nonlinear
- Data-driven closures: promise and challenges
- Gap: most evaluations are a priori only; cost rarely measured; cost breakdown never provided
- Contribution: systematic cost-accuracy comparison with detailed computational anatomy

**2. Methods** (~5 pages)

*2.1 Solver architecture*
- Incompressible N-S, fractional-step projection, staggered MAC grid
- GPU acceleration via OpenMP target offload (single code path, no #ifdef for arithmetic)
- Poisson solver: FFT via cuFFT (performance comparison vs multigrid)
- IBM for complex geometries (pre-computed weight arrays, <0.3% overhead)
- Performance validation: comparison against CaNS (Costa 2018) on channel flow
- CPU and GPU build from same source code

*2.2 Classical turbulence closures*
- Algebraic: baseline mixing length, GEP (Weatheritt & Sandberg 2016)
- Transport: k-omega (Wilcox 1988), SST k-omega (Menter 1994)
- Nonlinear: EARSM with 3 stress-strain relations (Pope, WJ, GS)
- Operation count per cell for each model class

*2.3 Neural network architectures*
- MLP (5→32→32→1): scalar eddy viscosity from invariants
- MLP-Large (5→128→128→128→128→1): capacity scaling test
- TBNN (5→64→64→64→10): anisotropy via tensor basis (Ling et al. 2016)
- PI-TBNN: realizability penalty formulation
- TBRF: random forest with Pope basis (Kaandorp & Dwight 2020)
- Table of all models: architecture, parameter count, FLOPs/cell, memory footprint
- Architecture diagrams

*2.4 Training procedure*
- Dataset: McConkey et al. (2021), k-omega SST baseline
- Features: 5 Pope invariants, 10 tensor basis
- Split: TBKAN 2025 case-holdout protocol
- Hyperparameters: optimizer, LR schedule, early stopping (reference appendix for full spec)
- PI-TBNN: realizability penalty and beta sweep

*2.5 Solver integration and cost model*
- Feature computation pipeline: gradU → S, Omega → S_hat, Omega_hat → invariants → NN input
- MLP/TBNN inference: GPU kernel design (parallelized over (cell, neuron) pairs, ping-pong workspace buffers)
- TBRF inference: tree traversal with branching (CPU fallback vs GPU attempt)
- Why NN closures are expensive: layer-sequential execution, activation cost, comparison to stencil-based closures
- Timing methodology: `TIMED_SCOPE` infrastructure, warm-up exclusion, sub-phase breakdown

**3. Results** (~6-7 pages)

*3.1 Solver validation*
- Channel flow Poiseuille test (2nd-order convergence)
- Performance comparison against CaNS on identical benchmark
- CPU vs GPU speedup for base solver

*3.2 A priori evaluation*
- Validation and test set RMSE (table)
- Component-wise accuracy breakdown
- Scatter plots: predicted vs true b_ij
- Lumley triangle visualization
- PI-TBNN beta sweep: negative result (tensor basis already near-realizable)
- TBRF tree count sweep: accuracy vs model size

*3.3 A posteriori evaluation*
- Channel flow: u⁺(y⁺) and Reynolds stress profiles vs MKM DNS
- Periodic hills: separation/reattachment, velocity profiles vs Breuer LES
- Cylinder: drag, lift, Strouhal
- Grid sensitivity study (one NN + one RANS on grids A-D)

*3.4 Computational cost analysis*
- Wall-clock profiling table: all models × all cases × GPU and CPU
- Sub-phase breakdown: gradient, features, tensor basis, inference, postprocess
- Comparison: FLOPs/cell (theoretical) vs measured throughput → arithmetic intensity
- Scaling with grid size (64³ to 256³)
- Memory footprint analysis
- Why TBNN costs 7x more than MLP: dominated by two 64×64 matmul layers (8192 of 9354 FLOPs)
- Why TBRF costs even more: branch-heavy traversal, cache-miss-dominated, poor GPU utilization

*3.5 Cost-accuracy tradeoff*
- Pareto plot (GPU): THE figure
- Pareto plot (CPU): how does the frontier shift?
- Pareto frontier analysis: which models are dominated? Which offer favorable tradeoffs?
- Practical recommendations: model selection guidelines by hardware and accuracy requirement

**4. Discussion** (~2 pages)
- When should you use each model class? (decision tree for practitioners)
- Why PI-TBNN doesn't help: architectural constraints are sufficient, loss penalties add gradient noise
- TBRF: offline accuracy ceiling, impractical online — implications for choosing NN architectures
- CPU vs GPU: the cost ranking may differ — MLP is relatively cheaper on CPU (matmul well-optimized by BLAS), TBRF may be relatively less expensive on CPU (no warp divergence penalty)
- Limitations: training data dependence, extrapolation to unseen Re/geometries, single solver
- Comparison to prior a priori-only evaluations — does accuracy ranking change when coupled to a solver?
- Opportunities: TensorRT/ONNX for faster inference, pruning/quantization, distillation from TBRF to MLP

**5. Conclusions** (~0.5 pages)
- Summary of key findings
- Practical recommendation: which model for which use case
- Future work: online training, transfer learning, hybrid models, inference optimization

**Appendix / Supplementary Material**
- Full training methodology (from `training_methodology.md`)
- Complete profiling tables for all GPU types (L40S, H100, H200)
- Additional flow case results
- Convergence histories
- Roofline model analysis for each closure type

#### Figures List

| # | Description | Type | Data Source |
|---|-------------|------|-------------|
| 1 | Solver architecture diagram (fractional-step pipeline with closure plug-in point) | Schematic | tikz |
| 2 | Model architecture diagrams (MLP, TBNN, TBRF side-by-side) | Schematic | tikz |
| 3 | Solver validation: our throughput vs CaNS on identical benchmark | Bar chart | Benchmark runs |
| 4 | Training/validation curves for each model | Line plot (multi-panel) | Training logs |
| 5 | Test set scatter: predicted vs true b_ij (grid of panels) | Scatter | A priori eval |
| 6 | Lumley triangle with predictions from each model | Scatter on triangle | A priori eval |
| 7 | PI-TBNN beta sweep: RMSE vs beta | Bar chart | Beta sweep |
| 8 | Channel u⁺(y⁺) profiles: all models vs DNS | Line plot | A posteriori |
| 9 | Channel Reynolds stress profiles: all models vs DNS | Line plot | A posteriori |
| 10 | Periodic hills velocity profiles at key x-stations | Line plot (multi-panel) | A posteriori |
| 11 | Periodic hills separation bubble visualization | Contour/streamline | A posteriori |
| 12 | Cylinder drag coefficient convergence | Line plot | A posteriori |
| 13 | **Cost breakdown: stacked bar by phase for each model** | Stacked bar | Profiling |
| 14 | **Cost-accuracy Pareto plot — GPU** (THE figure) | Scatter with annotations | Combined |
| 15 | **Cost-accuracy Pareto plot — CPU** | Scatter with annotations | Combined |
| 16 | Inference cost scaling with grid size | Line plot | Grid scaling runs |
| 17 | TBRF tree count sweep: accuracy vs size vs cost | Multi-axis plot | Training + profiling |

#### Tables List

| # | Description | Data Source |
|---|-------------|-------------|
| 1 | Model summary: architecture, parameters, FLOPs/cell, weight size, deployability | Training + analysis |
| 2 | A priori RMSE: validation and test, overall and per-component | A priori eval |
| 3 | A posteriori error metrics for each model × flow case | A posteriori runs |
| 4 | Wall-clock profiling: turb update, total step, overhead % — GPU | GPU profiling |
| 5 | Wall-clock profiling: turb update, total step, overhead % — CPU | CPU profiling |
| 6 | Sub-phase timing breakdown: gradient, features, basis, inference, postprocess | Detailed profiling |
| 7 | Solver validation: our throughput vs CaNS | Benchmark |
| 8 | TBRF tree count sweep: accuracy vs size vs cost | Training + profiling |
| 9 | PI-TBNN beta sweep results | Beta sweep |
| 10 | GPU vs CPU speedup by closure type | Combined profiling |

---

## 6. Potential Reviewer Concerns and Mitigations

| Concern | Mitigation |
|---------|------------|
| "Is your solver representative / well-optimized?" | CaNS benchmark comparison. Our throughput (2.5 Gcells/s at 4.2M cells on L40S) is competitive with published numbers from established codes. |
| "Only GPU results — what about CPU?" | We include full CPU results for all models. The cost ranking may differ on CPU. |
| "Only tested on simple flows" | Channel, hills (separated), cylinder (bluff body) cover a range of physics. McConkey test set includes CBFS (new geometry). |
| "Training data is from k-omega SST — what about other RANS baselines?" | McConkey also provides k-epsilon data; could add as sensitivity test. SST is the most common industrial RANS model. |
| "NN inference could be faster with TensorRT/ONNX" | True — our comparison uses raw matrix multiplies. Note this as future work. Our costs are upper bounds; the qualitative ordering is unlikely to change. |
| "Grid is too coarse for DNS-quality" | We're not doing DNS — we're comparing RANS-level models on RANS-appropriate grids. Grid sensitivity study shows convergence. |
| "Only one GPU type" | We have profiling data on L40S, H100, and H200. Report primary results on one GPU, supplementary on others. |
| "MLP target (anisotropy magnitude) is not standard" | Acknowledge this — MLP predicts a scalar proxy, not the full tensor. This is deliberate: it's the simplest possible NN closure. |
| "No online/adaptive training" | Out of scope — we focus on the offline-trained, fixed-weight regime. Note as future work. |
| "PI-TBNN implementation is naive" | We tested multiple beta values with proper warmup. Our negative result is consistent with the literature (Wu et al. 2018). |
| "TBRF in the solver is an unfair comparison" | This IS the point — the cost-accuracy tradeoff is the entire thesis. Even if TBRF is the most accurate offline, it must be deployable. |
| "No uncertainty quantification" | Out of scope. Could add ensemble spread for TBRF. Note as future work. |
| "What about the cost of TRAINING the models?" | Include training times in a table. All models train in <1 hour on a single GPU. This is amortized over all solver runs. |

---

## 7. Data and Reference Sources

### Reference DNS/LES Data for A Posteriori Comparisons

| Flow Case | Reference | Data Availability |
|-----------|-----------|-------------------|
| Channel Re_τ=180 | Moser, Kim, Moin (1999) | `https://turbulence.oden.utexas.edu/data/MKM/chandata.tar.gz` |
| Periodic hills Re_H=5600 | Breuer et al. (2009) | ERCOFTAC database / paper tables |
| Cylinder Re=100 | Established literature | C_d≈1.35, C_l≈±0.34, St≈0.165 |

### Solver Comparison Target

| Solver | Method | Published GPU Performance | Reference |
|--------|--------|--------------------------|-----------|
| CaNS | Staggered FD, FFT Poisson | ~500-800 Mcells/s on A100 | Costa 2018 |
| AFiD | Staggered FD, FFT Poisson | ~400-700 Mcells/s on V100 | Zhu et al. 2018 |
| NekRS | Spectral elements | ~0.5-1.0 GDOF/s per V100 | Fischer et al. 2022 |

### McConkey Dataset Cases

38 total cases across 4 geometries. Our split:

| Split | Cases | Points | Purpose |
|-------|-------|--------|---------|
| Train (18) | SD Re={1100-3500 excl 2000}, PH α={0.5,1.0,1.5}, CDC Re=12600 | 271,924 | Model fitting |
| Val (2) | SD Re=2000, PH α=0.8 | 23,967 | Hyperparameter selection, early stopping |
| Test (2) | PH α=1.2, CBFS Re=13700 | 51,844 | Reported a priori accuracy |
| Unused (16) | Flat plate (10 cases), SD Re=1800, convdiv20580, h20/h26/h31/h38/h42 | ~555K | Available for future work |

---

## 8. Execution Plan

### Priority Order

| Priority | Phase | Effort | Blocking? |
|----------|-------|--------|-----------|
| 1 | A priori test set evaluation | 1 session | No — can run immediately |
| 2 | TBRF C++ inference | 1-2 sessions | Blocks Phase 3 for TBRF |
| 3 | CaNS benchmark comparison | 1 session | Independent |
| 4 | A posteriori: channel flow (GPU + CPU) | 1 session | Blocks Phase 6 |
| 5 | A posteriori: periodic hills (GPU + CPU) | 1 session | Blocks Phase 6 |
| 6 | A posteriori: cylinder (GPU + CPU) | 1 session | Blocks Phase 6 |
| 7 | Grid scaling runs | 1 session | Blocks Phase 6 |
| 8 | Cost-accuracy synthesis + Pareto plots | 1 session | Needs 4-7 |
| 9 | Paper draft | 2-3 sessions | Needs all results |
| 10 | Revisions + polishing | 1-2 sessions | — |

### Non-Blocking Parallelism

- Phase 1 (a priori), Phase 2 (TBRF C++), and Phase 3 (CaNS) are all independent — can run in parallel
- Channel, hills, and cylinder a posteriori runs are independent — can submit all SLURM jobs simultaneously
- CPU and GPU runs for the same case are independent — can submit in parallel
- Paper writing (intro, methods, Section 2.5 computational anatomy) can start before all results are in — we already have enough detail

---

## 9. File Index

| File | Contents |
|------|----------|
| `docs/paper/paper_roadmap.md` | This file |
| `docs/paper/training_methodology.md` | Full training specification |
| `artifacts/profiling_results.md` | Solver timing data (9 models × 3 cases) |
| `results/paper/training/full_5266424.out` | Final 5-model training log |
| `results/paper/training/pi_sweep_5300440.out` | PI-TBNN beta sweep log |
| `data/models/training_summary.json` | Machine-readable results |
| `data/models/*_paper/` | Trained model weights |
| `scripts/paper/train_all_models.py` | Training pipeline |
| `examples/paper_experiments/` | CFD experiment configs |
| `src/nn_core.cpp` | MLP forward pass implementation (GPU kernel) |
| `src/turbulence_nn_mlp.cpp` | MLP turbulence model (feature compute + inference) |
| `src/turbulence_nn_tbnn.cpp` | TBNN turbulence model (features + basis + inference) |
