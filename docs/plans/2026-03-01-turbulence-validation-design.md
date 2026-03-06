# Turbulence Validation Design

## Goal

Prove that the CFD-NN solver produces physically faithful DNS and RANS-modeled turbulence by validating against established reference data (MKM DNS, Brachet TGV, analytical Poiseuille).

## Architecture: Two-Tier Validation

### Tier 1 — CI GPU Tests (~10-15 min total)

Fast automated tests that run in every PR. Use embedded reference data subsets (no external dependencies). Catch regressions in solver physics.

**Test A: `test_poiseuille_validation.cpp`** (label: gpu, medium)
- Re=100 channel, 64x32 grid, no turbulence model
- ~500 steps to steady state
- Assert: L2 error vs analytical parabola < 0.1%, mass conservation, dp/dx balance
- ~1 min GPU

**Test B: `test_dns_channel_validation.cpp`** (label: gpu, medium)
- 192x96x192 grid, v13 recipe (filter_strength=0.03, trip forcing)
- ~500 steps (machinery validation, not converged statistics)
- Assert: turbulence triggered (TKE > 0), incompressibility (max|div(u)| < 1e-5), CFL bounded, KE stable, resolution quality (y+ < 1, dx+ < 15, dz+ < 8)
- ~5 min GPU

**Test C: `test_rans_channel_validation.cpp`** (label: gpu, medium)
- 64x64 channel, stretched grid, Re_tau=180
- All 10 models x ~200 steps each
- Assert per model: U+ profile L2 error vs embedded MKM data within bounds, no-slip walls, monotonic profile, centerline symmetry, nu_t > 0 in interior
- Error bounds: Baseline < 20%, SST < 15%, NN < 25% (calibrate during implementation)
- ~3-5 min GPU

**Test D: `test_tgv_validation.cpp`** (label: gpu, medium)
- 64^3 periodic box, Re=1600, no turbulence model
- ~200 steps (initial vortex stretching phase)
- Assert: KE monotonically decreasing, dissipation matches early-time trend, symmetry preserved, incompressibility
- ~2 min GPU

### Tier 2 — Full Validation Report (SLURM, 1-4 hrs)

Long simulations producing publication-quality evidence. Run manually on GPU nodes, not in CI.

**DNS Channel Re_tau=180:**
- 192x96x192 (or 256x128x256), stretched grid
- Target true Re_tau=180 (reduce filter strength or increase resolution)
- ~20k-50k steps, accumulate statistics after transition (~10k+ samples)
- Output: mean U(y), Reynolds stresses (uu, vv, ww, uv) vs y, wall shear history

**RANS Channel (all 10 models):**
- 128x128, stretched, Re_tau=180
- 2000 steps each to fully-converged steady state
- Output: mean U(y), nu_t(y) per model

**TGV Re=1600:**
- 128^3 periodic
- ~5000 steps to capture dissipation peak
- Output: KE(t), dissipation rate -dK/dt vs time

**Poiseuille Re=100, Re=1000:**
- 64x64, 128x128
- Run to steady state
- Output: U(y) profile

**Report Generation** (`scripts/generate_validation_report.py`):
- Reads simulation .dat output files
- Loads full MKM reference profiles from `data/reference/mkm_retau180/`
- Generates plots: U+ vs y+ (log), Reynolds stress profiles, momentum balance, model comparison, TGV dissipation, Poiseuille convergence
- Computes error metrics table (L2, Linf, u_tau error per model)
- Outputs PNGs + summary table to `output/validation_report/`

**SLURM Orchestration** (`scripts/run_validation.sh`):
- Submits DNS (long), RANS (10 parallel), TGV, Poiseuille jobs
- Uses `embers` QOS
- Runs Python report generation after all complete

## Reference Data

### Embedded in C++ (`tests/reference_data.hpp`)
- MKM Re_tau=180: ~19 points (y+, U+), ~20 points each for uu+, vv+, ww+, -uv+
- Brachet TGV Re=1600: dissipation peak (epsilon_max ~ 0.0127 at t ~ 9.0)
- Poiseuille: analytical formula (no data needed)

### Full profiles (`data/reference/`)
- `mkm_retau180/`: Complete MKM profiles (~130 y+ points), all stress components
- `brachet_tgv/`: Digitized dissipation rate curve (~30 time points)
- Sources: https://turbulence.oden.utexas.edu/data/MKM/, Brachet et al. (1983)

## New Files

| File | Purpose | Est. Lines |
|------|---------|-----------|
| `tests/reference_data.hpp` | Embedded reference data arrays | ~150 |
| `tests/test_poiseuille_validation.cpp` | Laminar analytical validation | ~150 |
| `tests/test_dns_channel_validation.cpp` | DNS machinery validation | ~200 |
| `tests/test_rans_channel_validation.cpp` | 10 RANS models vs MKM | ~400 |
| `tests/test_tgv_validation.cpp` | TGV energy decay validation | ~200 |
| `data/reference/mkm_retau180/*.dat` | Full MKM DNS profiles | data |
| `data/reference/brachet_tgv/dissipation.dat` | TGV reference curve | data |
| `scripts/generate_validation_report.py` | Plot + error table generation | ~400 |
| `scripts/run_validation.sh` | SLURM orchestration | ~150 |

## Modified Files

| File | Change |
|------|--------|
| `CMakeLists.txt` | Register 4 tests with `gpu` + `medium` labels |

## Risks and Mitigations

1. **True Re_tau=180 stability**: Reducing velocity filter may cause blow-up. Start with current recipe, gradually reduce filter. May need higher resolution.
2. **RANS convergence in 200 steps**: Some models may not converge. Use relaxed tolerances or longer runs for immature models.
3. **CI time budget**: 4 new GPU tests add ~10-15 min. Acceptable (current GPU CI is ~45 min).
4. **MKM data licensing**: Public domain (US government funded). No issues.

## Success Criteria

- All 4 CI tests pass on GPU node in both Debug and Release
- Poiseuille error < 0.1% (analytical)
- RANS models produce bounded errors vs MKM (model-specific thresholds)
- DNS machinery: turbulence triggers, incompressibility holds, resolution adequate
- TGV: energy decays monotonically, no spurious creation
- Full report: plots visually match MKM reference data
