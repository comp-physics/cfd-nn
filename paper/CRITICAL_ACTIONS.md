# Critical Actions Before Paper Submission

Synthesized from: literature search, reviewer simulation, gap analysis, results audit.
Updated: Apr 4, 2026.

---

## Literature Position (confirmed unique)

Our paper fills 3 confirmed gaps in the literature:
1. **No prior work compares >4 closure types a posteriori** (Ling 2016 had 4, we have 21)
2. **No prior work embeds NN closures in a GPU solver** with cost measurement
3. **No Pareto cost-vs-accuracy plot exists** for turbulence closures

Key references to cite: Ling 2016 (TBNN), Kaandorp 2020 (TBRF), McConkey 2021 (dataset),
Duraisamy 2019 (review), Schmelzer 2020 (GEP/SpaRTA), Brener 2024 (realizability),
ERCOFTAC Closure Challenge 2026 (benchmark).

**New competitor to track**: TBKAN (2025, KAN-based TBNN) and Self-Scaling TBNN (2026).

---

## BLOCKERS (must fix, in priority order)

### B1: Fix duct Re (1 hour of H200 time)
- Change: Add `bulk_velocity_target = 1.0` to duct config
- This adjusts dp/dx dynamically to maintain U_b=1.0 → Re_b=3500
- Need `enable_bulk_velocity_control()` call in main_duct.cpp
- Then re-run all 18 duct models at correct Re
- **Without this, cannot compare to DNS and models extrapolate from training Re**

### B2: Run to convergence and plot QoI time history (2 hours H200)
- "The single highest-impact action" — reviewer simulation
- Run duct SST for 50K steps, plot U_b(step). Show plateau.
- Run hills SST for 10K steps, plot U_b(step).
- Need this for at LEAST SST, TBNN-small, MLP on duct
- Store intermediate QoI at qoi_freq=100 (every 100 steps)

### B3: Complete remaining duct models (2 hours H200)
- PI-TBNN ×3, TBRF ×3, TBNN-large, MLP-large
- Run at corrected Re (with bulk_velocity_target)

### B4: Document cylinder tensor divergence honestly (no H200 needed)
- Tensor models diverge at step 6725 on cylinder
- Report Cd from stable portion (first 6000 steps) with caveat
- Reframe cylinder as "do no harm" penalty test (Re=100 is laminar)
- Add sentence: "This case tests model behavior outside training conditions"

---

## SHOULD-FIX (major revision risk if missing)

### S1: Report nu_t/nu ratio for all models on all cases
- Explains WHY MLP kills the flow (nu_t = 130× SST)
- Transforms "MLP is bad" into "MLP extrapolates because [invariant X out of range]"
- Compute Pope invariant ranges: training vs test to show distribution shift

### S2: Validate tensor implementation on simple channel
- Run TBNN on plain channel (no IBM, no separation)
- If it works → hills low U_b is model limitation
- If it fails → code bug
- Critical for distinguishing bug from physics

### S3: Report y+ values for all cases
- Easy to compute from wall shear and first cell height
- Standard reviewer ask for any RANS paper

### S4: Run sphere case (4 hours H200)
- Paper claims 4 cases but only presents 3
- Either run it or reduce claim to 3 cases

### S5: Warmup sensitivity test
- Run TBNN on duct with 2× and 0.5× warmup time
- Show final QoI is insensitive (or report sensitivity as finding)

---

## NICE-TO-HAVE (strengthens but not blocking)

### N1: Grid convergence on duct (64³ vs 96³ vs 128³)
### N2: Multi-core CPU vs GPU comparison (not just 1 core)
### N3: A priori vs a posteriori scatter plot
### N4: Pope invariant distribution analysis (training vs test)
### N5: G-coefficient magnitudes for tensor models (explain cylinder identity)

---

## Available DNS Reference Data

| Case | Source | Re | Data | Status |
|------|--------|-----|------|--------|
| Hills | Krank et al. 2018 | Re_H=5600 | Cf(x), U(y) at 10 stations, full 2D fields | DOWNLOADED |
| Duct | Vinuesa et al. 2018 | Re_tau=180 | U,V,W profiles, Reynolds stresses | DOWNLOADED |
| Cylinder | Standard reference | Re=100 | Cd=1.35, St=0.164 | Scalar values |
| Sphere | Johnson & Patel 1999 | Re=200 | Cd=0.775 | Scalar values |

Note: Duct DNS is Re_tau=180 (Vinuesa), not Re_tau=150 (Pinelli). McConkey training 
data uses Re_tau=150. This is a ~20% Re difference. Vinuesa data may be more 
appropriate since our Re_b=3500 gives Re_tau≈180 (need to verify).

---

## Estimated H200 Time Budget

| Task | Hours |
|------|-------|
| Fix duct Re + re-run 18 models | 3 |
| Convergence plots (50K steps, 3 models) | 2 |
| Sphere 17 models | 4 |
| Validation: TBNN on channel | 0.5 |
| Warmup sensitivity | 0.5 |
| **Total** | **~10 hours** |

At embers QOS with 3-hour wall time limit, this requires 3-4 H200 sessions.
