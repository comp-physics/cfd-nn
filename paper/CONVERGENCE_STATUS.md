# Production Run Convergence Status (Apr 3, 2026)

## Summary: MOST RUNS ARE NOT CONVERGED

Almost all runs need to be extended. The 10K-step budget was insufficient
for the 3D duct, and the 500-step sphere runs barely started.

## Duct Re_b=3500 (96x96x96, THE critical case)

| Model | Steps | Last Residual | Trend | Converged? | Action |
|-------|-------|---------------|-------|------------|--------|
| None | 1K | 2.6e-4 | — | NO | Extend to 20K |
| k-omega | 10K | 8.0e-8 | dropping | Nearly | Extend to 15K |
| SST | 10K | 3.0e-6 | dropping | NO | Extend to 30K |
| RSM-SSG | 3K | 1.2e-4 | dropping | NO | Extend to 20K |
| TBNN | 10K | 9.0e-4 | **GROWING** | NO — unstable? | Investigate |
| TBNN-small | 10K | 2.5e-5 | dropping slowly | NO | Extend to 30K |
| TBNN-large | 10K | ? | ? | ? | Check |
| PI-TBNN | 10K | 2.8e-5 | dropping slowly | NO | Extend to 30K |
| PI-TBNN-small | 10K | ? | ? | ? | Check |
| PI-TBNN-large | 10K | ? | ? | ? | Check |
| MLP | 10K | 3.9e-6 | dropping | NO | Extend to 30K |
| MLP-med | 10K | 3.9e-6 | dropping | NO | Extend to 30K |
| MLP-large | 10K | 3.9e-6 | dropping | NO | Extend to 30K |
| Baseline | 10K | 3.9e-6 | dropping | NO | Extend to 30K |
| GEP | 10K | 3.9e-6 | dropping | NO | Extend to 30K |
| TBRF-1t | 5K | ? | ? | ? | Check |
| TBRF-5t | 8K | 2.1e-3 | fluctuating | NO | May not converge |
| TBRF-10t | 5K | ? | ? | ? | Check |
| EARSM-* | — | — | DIVERGE | N/A | Expected (duct instability) |

**Key concern**: TBNN (medium) has GROWING residual — may be slowly diverging.
TBNN-small and PI-TBNN are stable but slow to converge.

## Cylinder Re=100 (384x288x1, vortex shedding)

This is an UNSTEADY flow — vortex shedding at Re=100. Use time-averaged Cd.

| Model | Steps | Cd_mean | St | Source | Notes |
|-------|-------|---------|----|--------|-------|
| None | 10K | 1.374 | 0.115 | orig | Good (ref ~1.35) |
| Baseline | 10K | 1.596 | 0.109 | v2 | Corrected |
| SST | 10K | 1.553 | -1 | orig | No shedding (too diffusive) |
| k-omega | 10K | 2.417 | -1 | orig | **WAY TOO HIGH** — bug? |
| EARSM-WJ | 10K | 1.543 | -1 | orig | OK |
| EARSM-GS | 10K | 1.543 | -1 | orig | OK |
| EARSM-Pope | 10K | 1.545 | -1 | orig | OK |
| GEP | 10K | 1.953 | -1 | v2 | Corrected, high |
| RSM-SSG | 10K | 1.374 | 0.115 | orig | **= None, RSM not activating?** |
| MLP (all 3) | 10K | 1.411 | 0.111 | v2 | Corrected, all identical |
| TBNN (all 3) | 10K | 1.565 | -1 | orig | All identical |
| PI-TBNN (all 3)| 10K | 1.565 | -1 | orig | All identical |
| TBRF-1t | 10K | 1.565 | -1 | orig | OK |

**Issues to investigate**:
- k-omega Cd=2.42 — reference is ~1.35, something is wrong
- RSM = None — RSM doesn't seem to activate on cylinder
- All tensor models give identical Cd=1.565 — expected (tau_nl small at Re=100)

## Hills Re=5600 (384x192x1, separation)

| Model | Steps | Last Residual | U_b | Converged? | Source |
|-------|-------|---------------|-----|------------|--------|
| SST | 5K | 5.3e-4 | 0.476 | Approaching | orig |
| Baseline | 5K | 6.8e-6 | 0.028 | YES | v2 |
| GEP | 5K | 6.7e-6 | 0.028 | YES | v2 |
| MLP | 5K | 5.4e-4 | 0.452 | Approaching | v2 |
| TBNN | 3K | DIVERGED | — | N/A | orig |
| All tensor | ~4.5K | DIVERGED | — | N/A | Expected |

**Baseline v2 U_b=0.028 vs orig U_b=0.476**: The bugfix revealed that Baseline
produces almost no flow (nu_t dominates). This is a result, not a bug.

## Sphere Re=200 (192x128x128, 3D IBM)

**ALL RUNS ONLY 500 STEPS — NOT EVEN STARTED**

Cd values are ~12 (reference ~0.77). Flow hasn't developed at all.
Need 10K-50K steps minimum. This is a 3.1M cell 3D case.

| Model | Steps | Cd | Notes |
|-------|-------|----|-------|
| All | 500 | ~11.5-12.0 | Way too early |

## Corrected Results Needed (v2 re-runs)

The MLP nu_t bug (commit bd25218) invalidated Baseline, GEP, MLP results.
V2 re-runs completed for cylinder and hills. Still needed:
- Duct: Baseline, GEP, MLP (all 3) — **are the 10K duct runs from v2 or orig?**
- Sphere: ALL models need re-running (only 500 steps from orig)

## Action Items

### Priority 1: Duct convergence runs (H100/H200)
- Extend ALL duct models to 30K steps (restart from 10K checkpoint)
- Investigate TBNN growing residual
- ~30 jobs, ~15 min each on H200

### Priority 2: Sphere full sweep
- Re-run ALL 21 models for 20K-50K steps
- Use v2 code (post nu_t bugfix)
- ~21 jobs, ~1 hour each on H200 (3.1M cells)

### Priority 3: Fix anomalies
- k-omega Cd=2.42 on cylinder — investigate
- RSM = None on cylinder — investigate
- Verify duct MLP/GEP/Baseline are from v2 (post-fix)

### Priority 4: Hills extension
- Extend SST and MLP to 10K steps for full convergence
- Already have tensor divergence data (documented)
