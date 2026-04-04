# Blocking Items for Paper Submission

## What we need for each case (minimum viable paper)

### 1. DUCT (THE critical case — Pareto plot anchor)

**Status**: 11/18 models at dp/dx=-0.010, 10K steps. NOT CONVERGED (SST U_b=0.625, target ~1.0).

**Blocking items**:
- [ ] **B1**: Rebuild H200 binary with warmup fix + bulk velocity controller
- [ ] **B2**: Run SST at 50K steps to find converged U_b (verify dp/dx=-0.010 gives U_b≈1.0)
  - If U_b ≠ 1.0 at convergence: either use `bulk_velocity_target=1.0` (now implemented)
    or adjust dp/dx iteratively
  - This SINGLE run tells us if the Re is correct
- [ ] **B3**: Once Re is confirmed, run all 18 models at 50K steps
  - Skip EARSM (known duct divergence), skip MLP-large (too slow, MLP-small tells same story)
  - Priority: SST, TBNN-small, TBNN, PI-TBNN, RSM, MLP, k-omega, GEP (8 key models)
  - ~12 min per model × 8 = 96 min
- [ ] **B4**: Convergence plot: U_b vs step number for SST, TBNN-small, MLP (3 models)
  - Extract from qoi_freq=500 output (already saved)
- [ ] **B5**: Compare velocity profiles to Vinuesa DNS (Re_b=2500)
  - Normalize by each model's U_b — standard practice for different Re

**H200 time needed**: ~3 hours (rebuild 10 min + SST 50K 12 min + 8 models 96 min + contingency)

### 2. HILLS (separation test)

**Status**: 21/21 complete. SST residual plateaued at 1.26e-4. Tensor models survive.

**Blocking items**:
- [ ] **B6**: Verify convergence — SST residual flat at 1.26e-4, U_b=0.878. 
  Probably converged (periodic flow, residual constant). Plot U_b vs step to confirm.
- [ ] **B7**: Compare Cf(x) to Krank DNS data (already downloaded)
- [ ] **B8**: Extract velocity profiles at x/H stations and compare to DNS

**H200 time needed**: 0 (data exists, just analysis). Maybe 30 min to re-run SST longer if needed.

### 3. CYLINDER (stability test)

**Status**: 21/21 run but tensor models diverge at step 6725 (warmup bug).

**Blocking items**:
- [ ] **B9**: RE-RUN cylinder with warmup fix (new binary). Need rebuilt H200 binary.
  - 21 models × ~50s each = 18 min total
  - With warmup fix, tensor models may survive (warmup completes at t=50s)
- [ ] **B10**: If tensor models STILL diverge after warmup fix: report Cd from stable portion
  (first 5K steps), clearly documented

**H200 time needed**: 30 min (rebuild already done, just run)

### 4. SPHERE (3D validation)

**Status**: 0/17 models.

**Blocking**: 
- [ ] **B11**: Run at least SST, TBNN-small, MLP, RSM, None (5 key models)
  - 3.1M cells, ~50ms/step, 10K steps = 500s per model × 5 = 42 min
  - Sphere warmup_time=50s, dt≈0.004 → 12500 warmup steps (within budget)

**H200 time needed**: 1 hour

### 5. ANALYSIS & WRITING (no GPU needed)

**Blocking items**:
- [ ] **B12**: Compute Re_b for each model on duct (from converged U_b)
- [ ] **B13**: Compute Cf for duct (using corrected stretched-grid dy_wall)
- [ ] **B14**: Compare hills Cf(x) to Krank DNS
- [ ] **B15**: Pareto plot: cost (ms/step × steps_to_converge) vs accuracy (L2 error vs DNS)
- [ ] **B16**: Fill results_aposteriori.tex \todo items with final numbers
- [ ] **B17**: A priori vs a posteriori comparison (scatter plot)
- [ ] **B18**: Write analysis of why MLP fails (nu_t/nu ratio, invariant distribution)
- [ ] **B19**: Conclusions with quantitative claims

---

## Total H200 budget

| Phase | Time | What |
|-------|------|------|
| Rebuild | 10 min | nvc++ with all fixes |
| Duct SST convergence | 12 min | 50K steps, verify Re |
| Duct 8 key models | 96 min | 50K steps each |
| Cylinder re-run | 18 min | 21 models, warmup fix |
| Sphere 5 models | 42 min | Key models only |
| **Total** | **~3 hours** | Fits in 1 embers session |

## Critical path

```
Rebuild (B1) → Duct SST 50K (B2) → Check Re → {if ok: Duct sweep (B3)}
                                              → {if not: adjust dp/dx, re-run}
     ↓ (parallel on same GPU)
Cylinder re-run (B9) → check tensor stability
     ↓
Sphere 5 models (B11)
     ↓
Analysis (B12-B19) — no GPU, can do while runs complete
```

## What can we skip without weakening the paper much?

- **MLP-large, TBNN-large, PI-TBNN-large**: Same story as medium, just slower. Report timing only.
- **TBRF on duct**: Nice to have but TBRF-1t/5t on hills tells the story.
- **Sphere full sweep**: 5 key models is enough. Large NNs OOM anyway.
- **Grid convergence study**: Mention y+<1, cite CaNS validation. Not critical.
- **Multi-core CPU comparison**: Acknowledge in text, don't run.
