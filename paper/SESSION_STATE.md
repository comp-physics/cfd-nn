# Session State — Apr 4, 2026

## What's Running RIGHT NOW

### H200 GPU Pipeline (job beivew9n0, started 10:44 AM)
Location: `/storage/scratch1/6/sbryngelson3/cfd-nn/run_test/production_final_v3/`
Binary: `build_h200` (rebuilt with ALL bug fixes EXCEPT bulk_velocity 3D fix)

**Phase 1 (DONE):** Duct SST 50K steps with bulk_velocity_target=1.0
- Result: U_b=1.009, Re_b=3533, |v|=0.026 ← CORRECT Re achieved
- QoI extracted: `production_final_v3/qoi_duct_SST/` (cross-section, wall shear)

**Phase 2 (DONE):** Duct 7 key models, 50K steps
- None: U_b=2.10 (laminar, no turbulence)
- RSM: U_b=0.654, |v|=0.0084 ← RSM gradient fix working
- TBNN-small: TIMED OUT at 1200s, partial U_b≈1.59 (controller overshoot)
- TBNN: TIMED OUT at 1200s, partial U_b≈1.35
- MLP: U_b=0.007 ← still catastrophically over-diffusive
- k-omega: U_b=0.488
- GEP: U_b=0.557

**Phase 3 (RUNNING):** Cylinder 21 models, 10K steps, warmup fix
- None: Cd=1.2909 ✓
- Baseline: Cd=1.5961 ✓
- k-omega: running...
- (19 more to go, ~50s each, ~15 min remaining)
- KEY TEST: Will tensor models survive with full 50s warmup?

**Phase 4 (PENDING):** Sphere 5 key models, 10K steps

### Monitor command:
```bash
cat /storage/home/hcoda1/6/sbryngelson3/tmp_claude/claude-3048356/-storage-scratch1-6-sbryngelson3-cfd-nn/b8d3f8dc-0b9b-4ad4-9def-4c402009599f/tasks/beivew9n0.output
```

### OpenFOAM baseline (COMPLETED earlier)
- Duct SST: 5000 SIMPLE iters, 16702s, dp/dx=0.010, Ubar=1.0
- CPU timing: 3340 ms/step (matches our solver at 3336 ms/step)

---

## Bugs Found and Fixed This Session (9 total)

| # | Bug | Commit | Files Changed |
|---|-----|--------|---------------|
| 1 | RSM zero velocity gradients | ede29ba | turbulence_rsm.cpp |
| 2 | Duct wall shear 2× error | aa340bd | main_duct.cpp |
| 3 | tau_div limiter missing molecular nu | aa340bd | solver_operators.cpp |
| 4 | IBM wall distance ignores bodies | 1756342 | solver.cpp, solver.hpp |
| 5 | Hills Cf slope correction | 1756342 | qoi_extraction.cpp, qoi_extraction.hpp, main_hills.cpp |
| 6 | Warmup shares max_steps budget | 48366a3 | main_cylinder.cpp, main_duct.cpp, main_hills.cpp |
| 7 | Duct stretched grid dy_wall | 0458063 | main_duct.cpp |
| 8 | bulk_velocity() 2D-only for 3D | 377e332 | solver.cpp |
| 9 | Duct dp/dx too low for Re_b=3500 | 9233099→8b7530d | solver.cpp, duct config |

All committed and pushed to `fixup` branch.

---

## Key Findings This Session

### 1. Poisson Dominance is Geometry-Dependent
- **Duct (MG, 1 periodic dir):** Poisson = 81% of step → NN cost amortized
- **Cylinder (FFT2D, 1 periodic dir):** Poisson = 22% → NN cost dominates
- Uniform duct + FFT1D is 3.8× SLOWER (49 separate 2D MG solves)
- Duct fundamentally requires MG (only 1 periodic direction)
- Paper insight: cost analysis depends on flow geometry

### 2. Controller Overshoot for TBNN
- SST: controller converges to U_b=1.009 (correct)
- TBNN-small: controller converges to U_b≈1.61 (overshoot)
- Reason: TBNN has less effective friction → controller keeps pushing dp/dx up
- Solution: Use fixed dp/dx=-0.026 for all models, accept different Re_b per model

### 3. MLP Failure Quantified
- MLP nu_t = 121× SST = 1094× molecular viscosity on duct
- U_b=0.007 (essentially dead flow) at Re_b=3500
- Larger MLP → more over-diffusive (monotonic with network size)
- Root cause: scalar Boussinesq can't represent anisotropy → compensates with huge nu_t

### 4. QoI Extraction Requires Run Completion
- QoI (Cf, profiles, Cd) extracted at END of main() only
- Timeout kills → NO QoI data
- Hills production_final runs: ALL QoI missing (300s timeout killed them)
- Need to re-run hills with enough time OR extract QoI separately

### 5. Cost Data
- CPU (Xeon Gold 6226, 1 core) = OpenFOAM = 3340 ms/step on duct 884K cells
- GPU (H200) = 15.3 ms/step (SST), 39.1 ms (TBNN-small), 19.7 ms (MLP)
- GPU speedup: 218× vs CPU (same code)

---

## What STILL Needs to Be Done

### Immediate (next H200 session, ~3 hours)
1. **Duct re-run with dp/dx=-0.026, NO controller:** All 8 key models × 50K steps
   - Config is updated and committed
   - Need to rebuild H200 binary (dp/dx change is config-only, no rebuild needed!)
   - TBNN needs 50K × 67ms = 56 min → run without timeout
   - Estimated: SST 12min, RSM 5min, k-omega 5min, GEP 14min, MLP 10min, 
     TBNN-small 33min, TBNN 56min, None 2min = ~2.3 hours total

2. **Cylinder Phase 3:** Currently running, should finish in ~15 min
   - Check if tensor models survive with warmup fix

3. **Sphere Phase 4:** Will run after cylinder, ~42 min

4. **Hills re-run for QoI:** Need ALL 21 models with enough time for QoI extraction
   - ~2.5 hours on H200 (separate session)
   - OR: extract QoI from existing checkpoint data if available

### Analysis (no GPU needed)
5. **Hills Cf(x) comparison to Krank DNS** — data exists, need proper extraction
6. **Duct profile comparison to Vinuesa DNS** — from QoI cross-section data
7. **Pareto plot** — cost (ms/step) vs accuracy (error vs DNS)
8. **Fill paper results tables** — 45 \todo items in results_aposteriori.tex
9. **MLP failure analysis** — invariant distribution shift analysis

### Code changes needed for final production
10. **Duct config:** dp/dx=-0.026, no controller ← DONE (committed)
11. **Hills config:** may need longer max_steps or T_final for QoI
12. **QoI intermediate extraction:** Would be nice to extract QoI every N steps,
    not just at the end. Low priority but prevents data loss from timeouts.

---

## File Locations

### Production data
- `run_test/production_final/` — Old binary, 300s timeout (hills/cyl QoI missing)
- `run_test/production_final_v2/` — dp/dx=0.010 duct (wrong Re for TBNN)
- `run_test/production_final_v3/` — Current pipeline (controller, correct Re for SST)
- NEXT: `run_test/production_final_v4/` — Fixed dp/dx=-0.026, no controller

### DNS reference data
- `data/dns_reference/hills/KKW_DNS_*.dat` — Krank 2018, Re_H=5600
- `data/dns_reference/duct/*.prof.txt` — Vinuesa 2018, Re_tau=180
- `data/dns_reference/reference_values.md` — All scalar values

### Paper documents
- `paper/PAPER_PLAN.md` — Section-by-section outline
- `paper/RESULTS_SUMMARY.md` — Current results table
- `paper/BUG_REPORT.md` — All bugs found
- `paper/GAP_ANALYSIS.md` — Reviewer simulation results
- `paper/CRITICAL_ACTIONS.md` — Priority action list
- `paper/BLOCKING_ITEMS.md` — What blocks submission
- `paper/POISSON_DOMINANCE_INSIGHT.md` — Key cost finding
- `paper/MLP_FAILURE_ANALYSIS.md` — Why MLP fails
- `paper/PARETO_FRAMEWORK.md` — Plot design
- `paper/RE_MISMATCH_ANALYSIS.md` — Duct Re analysis
- `paper/CONVERGENCE_STATUS.md` — Old convergence assessment

### OpenFOAM baseline
- `openfoam_baseline/duct_reb3500/` — Complete (5000 iters, log available)
- `openfoam_baseline/cylinder_re100/` — Mesh created, not run yet

### Git state
- Branch: `fixup`
- All changes committed and pushed
- Latest commit: 8b7530d (duct dp/dx fix)

## SIMPLE Progress Update (Apr 4, 3:45 PM)

### HYPRE Integration Complete
- Build: nvc++ 25.5 + HYPRE 2.32 (CPU mode) + GPU offload (H200) ✓
- HYPRE BiCGSTAB+PFMG solves SST momentum in 2-3 iterations ✓
- No divergence for SST (the core achievement) ✓
- Variable-coefficient MG forces MG over FFT for SIMPLE pressure ✓

### Remaining SIMPLE Issues
1. Periodic channel stagnates: div(u*)=0 for x-uniform Poiseuille
   → degenerate test case, not relevant for duct/hills
2. Need 3D stencil assembly for duct (currently 2D only)
3. Need v-momentum solver (currently only u solved)
4. Need to test on non-degenerate geometry (duct with walls)

### Key Commits
- 177c3e8: Force MG for varcoeff, update HYPRE v2.32
- 426a33b: HYPRE momentum solver works for SST
- e1cb0e4: Add pseudo-transient to HYPRE stencil

### Performance
- HYPRE momentum solve: 1-3 BiCGSTAB iters (converges fast)
- Per SIMPLE iteration (2D, 32x48): ~56ms (dominated by MG pressure)
- For comparison: OpenFOAM SIMPLE on duct: 3340ms per iteration

## SIMPLE Status Update (Apr 4, 7:00 PM)

### What Works
- HYPRE PFMG approximate momentum solve (1 V-cycle, like OpenFOAM's 2 GS sweeps)
- Variable-coefficient MG pressure solve (using 1/a_P, not a_P — fixed critical bug)
- 2D u+v momentum with HYPRE
- RK3 warm-up → SIMPLE switching
- Nonzero div(u*) and nonzero p' correction
- Cd changes from 0.67→0.70 in first 100 SIMPLE iterations

### What Doesn't Work Yet
- Cd stagnates at 0.70 (DNS=1.35) after ~100 iterations
- Same result for all alpha_u (0.7, 0.9, 1.0) and alpha_p (0.3, 0.5)
- Eventually goes NaN at ~1500 iterations

### Root Cause Investigation Needed
1. Check IBM interaction with pressure solve (are solid cells handled correctly?)
2. Check that v-momentum is actually updating (res=0 suggests it's trivial)
3. Verify a_P consistency between stencil assembly and compute_aP_2d
4. Try on a case WITHOUT IBM (backward-facing step or driven cavity)

### Key Commits
- 78207a2: Fix varcoeff (1/a_P not a_P) + PFMG solver
- 2bd431b: 2D v-momentum + p' debug
