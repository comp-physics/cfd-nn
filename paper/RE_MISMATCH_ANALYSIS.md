# Re Mismatch Analysis — Duct Case

## The Issue

Our duct config uses fixed dp/dx = -0.003 with nu = 0.000571.
This gives different Re_b for each turbulence model:

| Model | U_b | Re_b = U_b×2/nu | % of target |
|-------|-----|------------------|-------------|
| None (laminar) | 0.385 | 1,349 | 39% |
| SST | 0.179 | 627 | 18% |
| TBNN-small | 0.192 | 673 | 19% |
| RSM-SSG | 0.185 | 648 | 19% |
| MLP | 0.007 | 25 | 0.7% |

Target: Re_b = 3500 (Pinelli DNS). Actual: 627 for SST.

OpenFOAM uses `meanVelocityForce` which adjusts dp/dx dynamically to maintain
Ubar = 1.0, giving the correct Re_b = 3500. It converged to dp/dx ≈ 0.010.

## Is This a Bug or a Design Choice?

**It's a design choice, but the wrong one for DNS comparison.**

Using fixed dp/dx means each model runs at its "natural" Re for that forcing.
This is physically valid — it's like applying a fixed pressure gradient and seeing
what flow develops. But it means:

1. **Cannot compare to Pinelli DNS at Re_b=3500** — our Re_b is ~627 for SST
2. **Models are compared at different Re** — None runs at Re_b=1349, SST at 627
3. **NN models trained at Re_b=3500** are evaluated at Re_b≈627 — extrapolation issue

## Options

### Option A: Use bulk_velocity_target (like OpenFOAM)
Add `bulk_velocity_target = 1.0` to the config. Our solver has this feature
(implemented in solver_turbulence_diagnostics.cpp). It adjusts dp/dx each step
to maintain U_b = 1.0. This gives Re_b = 3500 for all models, matching DNS.

**Pro**: Correct Re, comparable to DNS and OpenFOAM
**Con**: Different models experience different dp/dx (friction depends on model)

### Option B: Increase dp/dx to match Re (fixed dp/dx)
Set dp/dx = -0.010 (from OpenFOAM's converged value). This gives U_b ≈ 1.0 for SST.
Other models would give different U_b but at least SST matches DNS.

**Pro**: Simple, consistent forcing across models
**Con**: MLP would give even lower U_b; laminar would give unrealistically high U_b

### Option C: Keep fixed dp/dx, don't compare to DNS
Report results as "fixed dp/dx comparison" — valid for model ranking but not for 
absolute accuracy against DNS.

**Pro**: No code changes needed
**Con**: Cannot use the Pareto plot with DNS as reference

## Recommendation

**Use Option A (bulk_velocity_target)** for the final production runs. This is what
OpenFOAM does and what the RANS literature expects. The Re then matches the training
data (Re_b=3500) and the DNS reference (Pinelli Re_b=3500).

The current results at Re_b≈627 are still useful for MODEL RANKING (relative comparison)
but not for QUANTITATIVE DNS COMPARISON.

## Impact on Paper

- **Model ranking likely preserved**: tensor > SST > MLP at any Re
- **Absolute numbers will change**: U_b, |v|, Cf will all be different at Re_b=3500
- **Secondary flow strength will change**: larger Re → stronger secondary flow
- **MLP finding unchanged**: over-diffusive regardless of Re
- **Need to re-run duct with corrected config** before paper submission
