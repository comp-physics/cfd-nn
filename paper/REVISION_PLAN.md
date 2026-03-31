# Paper Revision Plan (Mar 31, 2026)

## The Paper's Thesis (revised)

For GPU-accelerated RANS, the cost-accuracy tradeoff of data-driven turbulence closures depends critically on flow dimensionality. On 2D flows, tensor-basis NN corrections add cost but no accuracy over SST. On 3D flows, they produce genuinely different physics. TBRF-1t on GPU is the practical winner: cheapest data-driven model (3-21% overhead), stable everywhere, produces anisotropic stresses where they matter.

## What's WRONG in the current paper

### Falsified claims (must fix)
1. **"TBRF is impractical online"** — WRONG. TBRF-1t is the CHEAPEST data-driven model (cheaper than all dense NNs). Abstract, intro contribution 5, discussion "TBRF: offline ceiling, online impractical", conclusions finding 4, methods_closures cache miss discussion — ALL must be corrected.
2. **"PI-TBNN realizability penalties do not improve accuracy"** — PARTIALLY WRONG. They hurt validation RMSE but improve test-set generalization (2.7x). The intro contribution 4 and conclusions finding 3 contradict the paper's own a priori test-set analysis.
3. **"5 data-driven architectures"** — Should be "4 architectures × 3 sizes = 12 data-driven + 9 classical = 21 total."
4. **"15 models in production sweep"** — Now 21 (or 18-20 depending on EARSM inclusion).

### Stale/outdated
5. **Results cost table** — missing RSM, TBRF-5t/10t, large variants. Grid sizes corrected but model count wrong.
6. **Discussion limitation "single GPU type L40S"** — production is on H200 now.
7. **Results a posteriori** — entirely unpopulated (51 TODOs).

### Framing problems
8. **Hills framed as accuracy test** — "can NN improve SST reattachment?" Answer is NO (2D correction is tiny). Must reframe as stability test.
9. **Cost analysis buried** — appears as section 7 of 9. This is the paper's unique contribution but reads as an afterthought.
10. **No mention of RSM as reference point** — RSM-SSG is the most expensive classical model and the standard for duct secondary flow. If TBNN can't beat RSM, what's the point?

## Proposed Narrative Arc

1. **Setup**: 21 closures in one GPU solver — largest apples-to-apples comparison ever.
2. **A priori**: Offline accuracy ranks models. Rankings reverse between val/test (PI-TBNN generalizes best). This motivates coupled evaluation.
3. **Stability**: 17/21 stable on all 4 cases. EARSM unstable on 3D duct corners with explicit solver — known limitation, documented with literature.
4. **2D accuracy**: On hills/cylinder, all tensor models ≈ SST. The Boussinesq hypothesis already captures the dominant b_12 component. Hills tests stability, not accuracy.
5. **3D accuracy**: On duct, only anisotropic models produce secondary flow. On sphere, models produce genuinely different Cd. These are the discriminating cases.
6. **Cost**: Classical <2% overhead. TBRF-1t 3-21%. MLP 19-72%. TBNN 51-96%. Overhead grows with grid size (NN scales linearly, solver benefits from FFT).
7. **Pareto**: SST dominates for cost. TBRF-1t dominates for data-driven accuracy at minimal overhead. TBNN is never on the Pareto front.
8. **Conclusion**: For GPU explicit solvers, TBRF-1t is the recommended data-driven closure.

## 9 Big Changes Needed

### Change 1: Rewrite abstract
Lead with 21-model comparison on identical GPU hardware. Emphasize cost finding and 2D/3D dichotomy. TBRF-1t as practical winner. Kill "TBRF impractical" claim.

### Change 2: Rewrite introduction contributions
1. Largest apples-to-apples comparison (21 models, 4 cases, 1 solver, 1 GPU)
2. Cost anatomy with clean decomposition
3. 2D anisotropic correction is negligible
4. PI-TBNN realizability as regularizer (reversal between val and test)
5. TBRF GPU tree traversal: cheapest data-driven model, not most expensive
6. EARSM + explicit solver unstable on 3D corners (literature-backed)

### Change 3: Restructure a posteriori by finding, not by case
Current: hills → cylinder → duct → sphere (reader must synthesize cross-cutting findings)
Proposed: stability → 2D nullity → 3D differentiation → cost
Each case appears where it's most informative.

### Change 4: Rewrite TBRF narrative throughout
Every "impractical/branch-heavy/cache-miss" claim → corrected with measured data showing TBRF-1t is cheapest.

### Change 5: Add RSM as critical comparison point
RSM-SSG is the classical reference for anisotropic RANS. If TBNN can't beat RSM on duct secondary flow, the NN adds no value. This comparison must be explicit.

### Change 6: Reframe hills as stability test
"Can NN improve SST reattachment?" → "Do 21 diverse closures remain stable in separated flow? And: anisotropic corrections vanish in 2D mean flows."

### Change 7: Discussion — honest reckoning
Explicitly state: "Our explicit solver is for cost measurement, not production RANS. An implicit solver would be more appropriate. The value is the comparison on identical hardware."

### Change 8: Kill CPU cost comparison TODO
Distraction. Paper is about GPU cost. CPU numbers double the tables without adding insight.

### Change 9: Populate a posteriori
Blocked on production runs, but structure and framing can be finalized now with placeholder tables that have correct rows/columns/captions.

## What "Definitive" Means for Each Case

### Cylinder Re=100 (unseen, laminar)
- **Success**: All models give Cd within 5% of ref (1.35), no model kills shedding
- **Proves**: "Do no harm" — trained models don't destabilize laminar flow
- **Honest note**: SST suppresses shedding at Re=100 (Cd=2.5 steady). This is expected RANS behavior, not a bug.

### Hills Re=5600 (trained, 2D)
- **Success**: All models stable; reattachment within ±10% of SST
- **Proves**: Stability of diverse closures in separation. Tensor correction negligible in 2D.
- **Does NOT prove**: Accuracy improvement over SST. Do not claim this.

### Duct Re_b=3500 (trained, 3D)
- **Success**: Secondary flow present/absent per model, quantitative comparison vs Pinelli DNS
- **Proves**: Only anisotropic models predict secondary flow. The NN learns 3D correction from invariants despite never seeing coupled feedback.
- **Key question**: Do TBNN/TBRF predict correct secondary flow direction and magnitude?

### Sphere Re=200 (unseen, 3D)
- **Success**: Cd and separation angle for all models
- **Proves**: Generalization to unseen 3D geometry. Models produce genuinely different physics (Cd range: 7.2 to 12.0).
- **Key finding**: EARSM-GS gives Cd=7.2, dramatically different from all others.

## Paper Story: Success or Failure?

**Neither — it's an informative study.** The paper's value is:

1. **For practitioners**: If you have a GPU solver and want data-driven closures, here's exactly what they cost and where they help. TBRF-1t is the practical choice. Dense NNs are too expensive for their marginal benefit. Tensor corrections only matter in 3D.

2. **For researchers**: A priori accuracy does not predict a posteriori usefulness. PI-TBNN's realizability penalty helps generalization but not validation. EARSM + explicit solver is fundamentally unstable on 3D corners. The 2D anisotropic correction is negligible — stop testing on 2D hills and expect improvement.

3. **For the field**: The largest controlled comparison of data-driven closures in a single solver on a single GPU. Reproducible, open-source, all 21 models and 4 cases with configs provided.
