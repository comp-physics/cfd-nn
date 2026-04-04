# Paper Gap Analysis — Comprehensive Review (Apr 4, 2026)

## CRITICAL GAPS (must fix before submission)

### 1. Duct Re Mismatch (SEVERITY: CRITICAL)
Our duct runs use fixed dp/dx=-0.003, giving Re_b≈627 for SST.
Target is Re_b=3500 (Pinelli DNS, McConkey training data).
OpenFOAM's meanVelocityForce gives correct Re_b=3500 with dp/dx≈0.010.

**Impact**: Cannot compare to DNS. NN models trained at Re_b=3500 evaluated at Re_b≈627.
**Fix**: Add `bulk_velocity_target = 1.0` to duct config (feature exists in code).
**Effort**: Config change + re-run all 18 duct models (~3 hours on H200).

### 2. Tensor Models Diverge on Cylinder (SEVERITY: HIGH)
All 9 tensor models (TBNN×3, PI-TBNN×3, TBRF×3) diverge at step ~6725
on cylinder Re=100. Cl explodes to 14,219. The tau_div limiter fix
stabilized hills but NOT cylinder.

**Impact**: Cannot report tensor model Cd on cylinder with confidence.
The Cd=1.586 is from averaging before divergence — not a converged value.
**Fix options**: 
  (a) Report Cd from stable portion only (honest but reviewer will question)
  (b) Increase tau_div_scale to clip harder
  (c) Reduce CFL
  (d) Use more aggressive ramp
**Effort**: Needs investigation + re-runs.

### 3. Duct Runs Incomplete (SEVERITY: HIGH)
Only 10/18 duct models complete. Missing: PI-TBNN×3, TBRF×3, TBNN-large, MLP-large.
**Fix**: Need more H200 time. ~2 hours needed.

### 4. Sphere Not Run (SEVERITY: MEDIUM)
0/17 sphere models run with bugfixed code.
**Impact**: Paper claims 4 cases but only has 3.
**Fix**: Need H200 time. 3D case = slower. ~4 hours needed.

---

## IMPORTANT GAPS (should address)

### 5. Hills Re Mismatch (SEVERITY: MEDIUM)  
SST gives Re_H=4916, target is 5600. 12% off.
Hills uses fixed dp/dx too. Less severe than duct (12% vs 82% error).
Could add `bulk_velocity_target = 1.0` to hills config too.

### 6. Grid Adequacy Not Demonstrated (SEVERITY: MEDIUM)
- Duct: 96³ — is this sufficient for RANS at Re_b=3500? No grid convergence study.
- Cylinder: 384×288 — adequate for Re=100 but IBM adds resolution requirements.
- Hills: 384×192 — adequate for 2D RANS.
A reviewer will ask: "how do you know the grid is fine enough?"
**Fix**: Run one case (duct SST) at 2× resolution (192³) and compare. Or cite prior work.

### 7. Convergence Not Demonstrated (SEVERITY: MEDIUM)
Duct SST residual is 3e-6 at 10K steps and still dropping.
Not clearly converged. Need either:
  (a) More steps until residual plateaus
  (b) A convergence plot showing QoI stabilization
**Fix**: Run duct SST for 30K+ steps, show U_b vs step number.

### 8. k-omega Cd=2.42 on Cylinder (SEVERITY: LOW-MEDIUM)
79% error vs DNS. This is a MODEL limitation (excess freestream eddy viscosity)
but a reviewer might think it's a code bug. The IBM wall distance fix didn't help
because the issue is in the model's behavior at Re=100.
**Fix**: Document clearly as known k-omega limitation. Cite Wilcox's caveat about
freestream sensitivity. Or drop k-omega from the cylinder comparison.

### 9. Cylinder Re=100 Is Laminar (SEVERITY: MEDIUM)
A reviewer will ask: "Why evaluate turbulence models on a laminar flow?"
The answer is "do no harm" testing — but we should articulate this clearly.
At Re=100, the correct answer is None (laminar). Any model that gives Cd≠1.35
is ADDING error, not removing it. This reframes the cylinder as a "penalty test."

### 10. No Error Bars / Uncertainty Quantification (SEVERITY: MEDIUM)
All results are single runs. No ensemble, no grid sensitivity, no statistical
analysis. A reviewer will want to know if the differences between models are
significant or within numerical noise.
**Fix**: At minimum, run duct SST with 2 different grids to show grid independence.

---

## NARRATIVE GAPS (strengthen the paper)

### 11. A Priori vs A Posteriori Comparison
We claim "a priori doesn't predict a posteriori" but haven't shown this
quantitatively in the a posteriori section. Need a scatter plot: 
x = a priori RMSE, y = a posteriori error metric.

### 12. Cost Metric Definition
"Wall time to 10K steps" is not the same as "wall time to convergence."
For the Pareto plot, we need a convergence-based cost metric. This requires
converged runs (gap #7).

### 13. OpenFOAM CPU Comparison Fairness
Our CPU solver matches OpenFOAM at 3.34s/step — good.
But OpenFOAM can also run parallel (8 cores → ~4× speedup → 0.84s/step).
And GPU-accelerated OpenFOAM (AmgX) exists.
We should acknowledge these alternatives in the paper.

### 14. Training Data Coverage
McConkey data covers hills, CDC, CBFS (2D) and duct (3D).
Cylinder and sphere are UNSEEN geometries. This is a strength (generalization test)
but we should be explicit about it. The poor NN performance on cylinder may be
partly due to extrapolation to unseen geometry at unseen Re.

---

## SUMMARY: Priority Actions

| Priority | Gap | Effort | Impact |
|----------|-----|--------|--------|
| **P0** | Fix duct Re (bulk_velocity_target) | Config + re-run | Enables DNS comparison |
| **P0** | Finish duct runs (8 models) | 2 hrs H200 | Complete data |
| **P1** | Investigate cylinder tensor divergence | 1-2 hrs | Fix or document |
| **P1** | Run sphere (17 models) | 4 hrs H200 | Complete 4th case |
| **P1** | Demonstrate convergence (30K steps) | 1 hr H200 | Reviewer defense |
| **P2** | Fix hills Re (bulk_velocity_target) | Config + re-run | More accurate |
| **P2** | Grid convergence check | 4 hrs H200 | Reviewer defense |
| **P2** | A priori vs a posteriori scatter | 1 hr analysis | Strengthen finding |
| **P3** | Error bars / sensitivity | Multiple runs | Nice to have |
