# GPU Optimization Impact Report

**Date:** Thu Dec 11 11:47:34 AM EST 2025
**Branch:** gpu-optimization
**Commit:** d6d86b1
**Node:** atl1-1-01-004-31-0.pace.gatech.edu
**GPU:** NVIDIA L40S

---

## Executive Summary

### Kernel Launch Reduction

| Metric | Baseline | After Optimization | Reduction |
|--------|----------|-------------------|-----------|
| **BC kernels** | 6,331,608 (72%) | 1803000 (37.7%) | 71.5% |
| **Total kernels** | 8,769,713 | 4778729 | 45.5% |
| **Memory transfers** | 536,495 | 649679 | -21.0% |

### Correctness Validation

- **Tests passed:** 7
- **Tests failed:** 1

❌ Some tests failed

### Performance

- **Average execution time:** 64.40s (5 runs)

---

## Detailed Analysis

### Top 15 Kernel Calls

```
1803000 _ZN5nncfd22MultigridPoissonSolver8apply_bcEi
1800000 _ZN5nncfd22MultigridPoissonSolver6smoothEiid
 301500 _ZN5nncfd22MultigridPoissonSolver16compute_residualEi
 300000 _ZN5nncfd22MultigridPoissonSolver6vcycleEiii
 151500 _ZN5nncfd22MultigridPoissonSolver20compute_max_residualEi
 150000 _ZN5nncfd22MultigridPoissonSolver21prolongate_correctionEi
 150000 _ZN5nncfd22MultigridPoissonSolver17restrict_residualEi
  45616 _ZN5nncfd10RANSSolver17apply_velocity_bcEv
  22804 _ZN5nncfd10RANSSolver4stepEv
  22804 _ZN5nncfd10RANSSolver16correct_velocityEv
  11402 _ZN5nncfd10RANSSolver23compute_convective_termERKNS_11VectorFieldERS1_
  11402 _ZN5nncfd10RANSSolver22compute_diffusive_termERKNS_11VectorFieldERKNS_11ScalarFieldERS1_
   5701 _ZN5nncfd10RANSSolver18compute_divergenceERKNS_11VectorFieldERNS_11ScalarFieldE
   3000 _ZN5nncfd22MultigridPoissonSolver13subtract_meanEi
```

### Optimization Impact Assessment

⚠️ **OPTIMIZATION RESULTS NEED REVIEW**

Results:
- BC reduction: 71.5% (target: >70%) - ✅
- Total reduction: 45.5% (target: >50%) - ❌
- Tests passing: ❌

**Recommendation:** Review logs for unexpected behavior.

---

## Files in This Report

- `build.log` - Build output
- `kernel_launches.log` - Full kernel trace (~2GB)
- `kernel_stats.txt` - Kernel statistics summary
- `all_tests.log` - Complete test output
- `OPTIMIZATION_REPORT.md` - This report

---

**Completed:** Thu Dec 11 11:47:34 AM EST 2025
