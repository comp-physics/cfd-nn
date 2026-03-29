# Model Sweep Findings & Next Steps (Mar 25, 2026)

## What We Ran

15 models × 3 cases on H200 (NVIDIA H200, nvc++ 25.5, cc90, Release).
All with `ibm_eta = 0.1` volume penalization, `time_integrator = rk3`, `convective_scheme = skew`.

| Case | Grid | T_final | max_steps |
|------|------|---------|-----------|
| Cylinder Re=100 | 384×288×1 (111K) | 20.0 | 50000 |
| Hills Re=10595 | 384×192×1 (74K) | 20.0 | 50000 |
| Sphere Re=200 | 192×128×128 (3.15M) | 10.0 | 50000 |

---

## Results

### Cylinder Re=100

| Model | ms/step | Steps to T_final | Cd | Behavior |
|-------|---------|-------------------|-----|----------|
| baseline | 1.61 | 1981 | 7.28 | Laminar — identical to all non-transport models |
| mixing_length | 1.74 | 1981 | 7.28 | Zero nu_t (no gradients to trigger mixing length) |
| **komega** | 1.69 | **50000 (hit cap)** | **1610** | Transport bootstraps k/omega, but drives dt tiny → never converges |
| **SST** | 1.76 | **5421** | **37.1** | Transport works, dt smaller than baseline, Cd still high |
| earsm_wj | 1.74 | 1981 | 7.28 | Zero nu_t (EARSM needs nonzero k/omega from transport) |
| earsm_gs | 1.74 | 1981 | 7.28 | Same |
| earsm_pope | 1.74 | 1981 | 7.28 | Same |
| gep | 1.71 | 1981 | 7.28 | Same |
| mlp | 2.36 | 1981 | 7.28 | Zero nu_t at Re=100 (Pope invariants are zero in uniform flow) |
| mlp_med | 3.89 | 1981 | 7.28 | Same |
| tbnn_small | 7.37 | 1981 | 7.28 | Same |
| tbnn | 10.99 | 1981 | 7.28 | Same |
| pi_tbnn_small | 7.38 | 1981 | 7.28 | Same |
| pi_tbnn | 10.92 | 1981 | 7.28 | Same |
| tbrf_1t | 1.76 | 1981 | 7.28 | Same |

### Hills Re=10595

| Model | Status | Separation x/H |
|-------|--------|----------------|
| **komega** | Stable (49387 steps, t=20) | 1.16 |
| **SST** | Stable (50000 steps, t=17.2) | 1.51 |
| All 13 others | **DIVERGED** (res=1e30 at t≈1.5) | — |

### Sphere Re=200

| Model | ms/step | Steps | Cd | Notes |
|-------|---------|-------|----|-------|
| All 8 classical | 6.0-6.3 | 578 | 2.97 | All identical |
| **MLP** | 22.0 | **11340** | **606.8** | Huge spurious nu_t |
| **MLP-med** | 63.6 | **12100** | **294.0** | Same issue, slightly less extreme |
| TBNN-small | 54.4 | 578 | 2.97 | Same as baseline (zero nu_t) |
| TBNN | 151.6 | 578 | 2.97 | Same |
| PI-TBNN-small | 53.8 | 578 | 2.97 | Same |
| PI-TBNN | 152.0 | 578 | 2.97 | Same |
| TBRF-1t | 7.8 | 578 | 2.97 | Same |

---

## Root Cause Analysis

### Why most models give identical results

The problem is **cold-start initialization**. All simulations start from uniform flow (u=U_inf, v=0) with only a small wake perturbation. The volume penalization (`ibm_eta=0.1`) smoothly transitions from solid to fluid, creating **very weak velocity gradients** near the body. This means:

1. **Mixing length**: Computes `nu_t = l_mix² |du/dy|`. With near-uniform flow, `|du/dy| ≈ 0` → `nu_t = 0`.

2. **EARSM (WJ, GS, Pope)**: Computes `nu_t` from k and omega via SST transport, then adds anisotropy correction. But EARSM is implemented as `SSTWithEARSM` — it **requires the SST transport equations to provide k/omega**. If the transport model starts from initial k/omega that decay to zero (no production), the EARSM closure gets zero inputs. **However**, our implementation should initialize k/omega from turbulence intensity (5%, see `initialize_uniform` in solver.cpp). Need to check why k/omega aren't bootstrapping for EARSM like they do for SST.

3. **GEP**: Same as EARSM — algebraic model that needs k/omega from transport.

4. **NN models (TBNN, PI-TBNN, TBRF)**: Compute features from velocity gradients (Pope invariants: traces of S² and Ω² tensors). In near-uniform flow, S ≈ 0, Ω ≈ 0 → all invariants are zero → NN predicts zero. This is mathematically correct — the model has no information to work with.

5. **MLP**: Also uses Pope invariants, but predicts a **scalar nu_t** instead of tensor coefficients. On unseen geometries (sphere), it produces large spurious nu_t from small numerical noise in the invariants — a generalization failure.

6. **k-omega and SST**: Their transport equations have **source terms proportional to |S|² and destruction terms proportional to omega²**. Even with small initial gradients from the penalization band, production > destruction creates a positive feedback that grows k and omega. This is how transport models "bootstrap" — they amplify small signals.

### Why EARSM doesn't bootstrap like SST

**This is likely a bug.** Both EARSM and SST use the same `SSTKOmegaTransport` for their transport equations. The EARSM model is `SSTWithEARSM` which combines SST transport with EARSM closure. If SST alone can bootstrap k/omega but EARSM can't, the issue is either:

1. The EARSM closure overwrites nu_t with zero (from zero invariants), which then suppresses the transport production terms
2. The EARSM initialization is different from SST's
3. The transport equations aren't being advanced for EARSM (code path issue)

**Action needed:** Check if EARSM's `advance_turbulence()` is actually being called, and whether the EARSM closure's nu_t is overriding the SST transport's nu_t.

---

## What We Need for Rich Paper Results

### Problem 1: Non-transport models need developed flow to function

Models that compute nu_t from velocity gradients (mixing_length, GEP, all NN) need **nonzero velocity gradients** to produce nonzero output. With volume penalization from uniform init, these gradients take a very long time to develop.

**Solution: Two-phase initialization**

1. **Phase 1 (warm-up):** Run with SST for N steps to develop the flow field (velocity gradients, k, omega fields)
2. **Phase 2 (evaluation):** Switch to target model, continue from the developed flow state

This is physically motivated — in practice, RANS models are rarely started from uniform flow. The SST warm-up provides the "background turbulence state" that all models need.

**Implementation:**
- Add `warmup_model` config parameter (e.g., `warmup_model = sst`)
- Add `warmup_steps` or `warmup_time` parameter (already exists: `config.warmup_steps`)
- In the app time loop: run Phase 1 with warmup model, then swap to target model and reset timers
- Save/restore k, omega fields across the model swap

Alternatively, simpler: **save the SST solution to disk after warm-up, then restart all models from that checkpoint.** This ensures all models start from exactly the same developed flow.

### Problem 2: EARSM not bootstrapping (possible bug)

EARSM should behave like SST for the transport part. If the EARSM closure's zero nu_t is suppressing the transport production, the fix is to use SST's nu_t for the transport equations and only use EARSM's nu_t/tau_ij for the momentum equations.

**Action:** Debug `SSTWithEARSM::update()` vs `SSTKOmegaTransport::update()` to find where they diverge.

### Problem 3: k-omega dt collapse

k-omega ran 50K steps on cylinder Re=100 but only reached t=13 (vs t=20 for baseline in 1981 steps). The transport equations produce large omega near walls, driving dt extremely small. This is a known issue with k-omega at low Re.

**Solutions:**
- Use SST instead of raw k-omega (SST has F1/F2 blending that damps near-wall omega)
- Set `dt_min` high enough to prevent collapse (but may cause CFL violation)
- Accept this as a finding: "k-omega is impractical at low Re due to dt collapse"

### Problem 4: Volume penalization makes Cd too high

With `ibm_eta = 0.1`, the no-slip is enforced with O(eta) error. At T_final=20 on cylinder Re=100, Cd=7.28 (ref: 1.35). The penalization needs either:
- Smaller eta (but that's unstable without re-forcing)
- Much longer physical time for the flow to develop through the penalized layer
- A better penalization that's both stable and accurate

**For the paper:** Use relative comparisons between models (all use the same IBM), not absolute Cd values. The reference comparison would be against our own solver's "best achievable" (e.g., SST result) rather than DNS.

---

## Recommended Plan

### Phase 1: Fix EARSM bootstrap (investigate bug)
- Check if `SSTWithEARSM::advance_turbulence()` is being called
- Check if EARSM closure's nu_t overwrites SST transport's nu_t
- Expected: 1-2 hours of debugging

### Phase 2: Implement two-phase warm-up
- Run SST for warm-up period to develop flow
- Switch to target model at developed state
- All models start from identical initial conditions
- Expected: ~100 lines of code in app time loops

### Phase 3: Re-run model sweep with warm-up
- 15 models × 3 cases with SST warm-up
- Expect: all models produce distinct results because they start from nonzero velocity gradients
- Expected: same compute time as current sweep (~30 min on H200)

### Phase 4: Full production sweep
- 20 models × 8 configs with warm-up
- Long T_final for converged QoIs
- Wall-time-per-physical-time metric with breakdown

---

## Timing Results with SST Warm-up (H200, Mar 25)

Cylinder Re=100, SST warm-up t=10, evaluate each model to T_final=30.
All models simulate the same 20s of physical time from the same developed flow state.

| Model | Wall (s) | Steps | turb ms/call | turb% | Cd | Notes |
|-------|---------|-------|-------------|-------|-----|-------|
| baseline | 2.69 | 1686 | 0 | 0% | 18.8 | Reference |
| mixing_length | 2.93 | 1690 | 0.055 | 3.2% | 543.1 | Wrong Cd |
| SST | 10.78 | 6166 | 0.025 | 1.4% | 32.0 | 3.7× more steps |
| EARSM-WJ | 3.08 | 1710 | 0.028 | 1.6% | 37.1 | Distinct from SST |
| GEP | 2.84 | 1687 | 0.026 | 1.5% | 542.0 | Wrong, like mixing_length |
| MLP | 108.5 | 46541 | 0.635 | 27.2% | 340.6 | dt crushed by huge nu_t |
| TBNN-small | 228.0 | 46540 | 3.066 | 62.6% | 340.6 | Same dt as MLP, 2× model cost |

**Key observations:**
1. All models produce distinct results (warm-up works)
2. MLP/TBNN hit dt_min floor → same step count (46K), cost is purely model eval
3. SST takes 3.7× more steps due to transport-driven smaller dt
4. mixing_length and GEP give wrong Cd (~540 vs ref ~1.35) — generalization failures
5. EARSM-WJ gives different Cd from SST (37 vs 32) — EARSM bootstrap fix working
6. Cd values still too high (baseline=18.8 vs ref=1.35) — need more physical time with penalization

**FIXED (Mar 25-26):** RK3 sub-timers added, warm-up adaptive dt bug fixed (was missing `set_dt()` call), `solver_step` total extraction fixed (was including warm-up wall time), `final_t` extraction fixed (was matching scientific notation from T_final line).

### Final Validated Timing Results (H200, Mar 26)

**Note:** Hills and cylinder were rerun at fixed dt=0.0004 for consistent Poisson costs across models. Poisson cost anomaly on cylinder RESOLVED — was caused by different adaptive dt across models. At fixed dt=0.0004, Poisson cost is 0.280+/-0.007ms across all 15 models. See results/paper/timing_fixed_dt_h200.md for complete data.

**Cylinder Re=100** (2D, 111K cells, warm-up=2s, eval=2s of physics):

| Model | Wall (s) | Steps | ms/step | turb% | Slowdown |
|-------|---------|-------|---------|-------|----------|
| baseline | 0.28 | 174 | 1.61 | 0% | 1.0× |
| mixing_length | 0.31 | 178 | 1.72 | 3% | 1.1× |
| GEP | 0.30 | 175 | 1.69 | 2% | 1.1× |
| EARSM-WJ | 0.37 | 196 | 1.88 | 2% | 1.3× |
| SST | 0.87 | 482 | 1.80 | 1% | 3.1× (dt) |
| TBRF-1t | 8.21 | 4659 | 1.76 | 5% | 29× (dt crush) |
| MLP | 11.01 | 4654 | 2.37 | 27% | 39× (dt+model) |
| TBNN-small | 22.87 | 4663 | 4.91 | 62% | 82× (dt+model) |

**Duct Re_b=3500** (3D, 885K cells, warm-up=1s, eval=2s):

| Model | Wall (s) | Steps | ms/step | turb% | Slowdown |
|-------|---------|-------|---------|-------|----------|
| baseline | 0.79 | 298 | 2.65 | 0% | 1.0× |
| classical (4) | 0.82-0.85 | 298 | 2.74-2.85 | 1-2% | 1.0-1.1× |
| TBRF-1t | 0.96 | 298 | 3.21 | 16% | 1.2× |
| TBNN-small | 4.84 | 298 | 16.23 | 83% | 6.1× |
| MLP | 125.51 | 17357 | 7.23 | 63% | 159× (dt crush) |

**Sphere Re=200** (3D, 3.15M cells, warm-up=2s, eval=2s):

| Model | Wall (s) | Steps | ms/step | turb% | Slowdown |
|-------|---------|-------|---------|-------|----------|
| baseline | 0.69 | 114 | 6.02 | 0% | 1.0× |
| classical (4) | 0.71-0.75 | 114 | 6.22-6.54 | 1-2% | 1.0-1.1× |
| TBRF-1t | 0.88 | 114 | 7.75 | 20% | 1.3× |
| TBNN-small | 6.24 | 114 | 54.75 | 89% | 9.1× |
| MLP | 49.65 | 2255 | 22.02 | 72% | 72× (dt crush) |

**Key paper findings:**
1. Classical RANS models add <10% overhead — essentially free
2. TBRF-1t adds 5-20% overhead — cheapest NN model by far
3. TBNN-small is 6-82× slower — dominated by per-step model cost on 3D, dt crush on 2D
4. MLP is 39-159× slower — always dt-crushed (huge nu_t on all geometries)
5. Two cost mechanisms: (a) per-step model eval (turb%), (b) dt reduction from large nu_t (step count)
6. MLP's dt crush on duct (trained geometry) suggests systematic over-prediction, not just generalization failure

### Alternative: Checkpoint-based approach
Instead of two-phase time loop:
1. Run SST once per case config to convergence
2. Save checkpoint (velocity + k + omega + nu_t fields)
3. Restart each model from the SST checkpoint
4. This guarantees identical starting conditions and is simpler to implement

**Advantage:** Each model evaluation is independent — easy to parallelize as separate SLURM jobs.
**Disadvantage:** Need checkpoint save/load infrastructure (may already exist via VTK?).
