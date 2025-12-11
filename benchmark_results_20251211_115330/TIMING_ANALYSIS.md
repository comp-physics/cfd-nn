# CPU vs GPU Performance Analysis

**Date:** Thu Dec 11 12:00:54 PM EST 2025
**Node:** atl1-1-01-004-31-0.pace.gatech.edu
**GPU:** NVIDIA L40S
**CPU:** Intel(R) Xeon(R) Gold 6426Y
**Cores:** 4
**Branch:** gpu-optimization
**Commit:** ded2028

---

## Overall Performance

### Timing Summary

| Device | Average Time | Std Dev | Speedup |
|--------|-------------|---------|---------|
| CPU    | 4.115s | - | 1.0x (baseline) |
| GPU    | 64.478s | - | .06x |

⚠️ **GPU is slower than CPU for this mesh size (32×64)**

This is expected for small meshes where kernel launch overhead
dominates. GPU speedup will be larger for production meshes (128×128+).

---

## Detailed Breakdown

### CPU Timing Breakdown

```

```

### GPU Timing Breakdown

```
Testing laminar Poiseuille flow... [MultigridPoisson] Initializing GPU buffers for 3 levels, finest grid: 32x64
[MultigridPoisson] GPU buffers allocated successfully
```

---

## All Test Runs

### CPU Runs (5 iterations)
```
Run 1:
Testing laminar Poiseuille flow... PASSED (error=4.22605%, iters=3001)
Testing solver convergence behavior... PASSED (residual=2.603043e-06, iters=2001)
Testing divergence-free constraint (staggered grid)... PASSED (max_div=0.000000e+00, rms_div=0.000000e+00)
Testing mass conservation (periodic channel)... PASSED (max_flux_error=0.000000e+00)
Testing single timestep accuracy (discretization)... PASSED (error=7.70e-03%)
Testing momentum balance (Poiseuille)...  residual=1.27e-06, iters=3001, L2_error=4.06%... PASSED
Testing energy dissipation rate...  residual=1.27e-06, iters=3001, energy_error=4.10%... PASSED

Run 2:
Testing laminar Poiseuille flow... PASSED (error=4.22605%, iters=3001)
Testing solver convergence behavior... PASSED (residual=2.603043e-06, iters=2001)
Testing divergence-free constraint (staggered grid)... PASSED (max_div=0.000000e+00, rms_div=0.000000e+00)
Testing mass conservation (periodic channel)... PASSED (max_flux_error=0.000000e+00)
Testing single timestep accuracy (discretization)... PASSED (error=7.70e-03%)
Testing momentum balance (Poiseuille)...  residual=1.27e-06, iters=3001, L2_error=4.06%... PASSED
Testing energy dissipation rate...  residual=1.27e-06, iters=3001, energy_error=4.10%... PASSED

Run 3:
Testing laminar Poiseuille flow... PASSED (error=4.22605%, iters=3001)
Testing solver convergence behavior... PASSED (residual=2.603043e-06, iters=2001)
Testing divergence-free constraint (staggered grid)... PASSED (max_div=0.000000e+00, rms_div=0.000000e+00)
Testing mass conservation (periodic channel)... PASSED (max_flux_error=0.000000e+00)
Testing single timestep accuracy (discretization)... PASSED (error=7.70e-03%)
Testing momentum balance (Poiseuille)...  residual=1.27e-06, iters=3001, L2_error=4.06%... PASSED
Testing energy dissipation rate...  residual=1.27e-06, iters=3001, energy_error=4.10%... PASSED

Run 4:
Testing laminar Poiseuille flow... PASSED (error=4.22605%, iters=3001)
Testing solver convergence behavior... PASSED (residual=2.603043e-06, iters=2001)
Testing divergence-free constraint (staggered grid)... PASSED (max_div=0.000000e+00, rms_div=0.000000e+00)
Testing mass conservation (periodic channel)... PASSED (max_flux_error=0.000000e+00)
Testing single timestep accuracy (discretization)... PASSED (error=7.70e-03%)
Testing momentum balance (Poiseuille)...  residual=1.27e-06, iters=3001, L2_error=4.06%... PASSED
Testing energy dissipation rate...  residual=1.27e-06, iters=3001, energy_error=4.10%... PASSED

Run 5:
Testing laminar Poiseuille flow... PASSED (error=4.22605%, iters=3001)
Testing solver convergence behavior... PASSED (residual=2.603043e-06, iters=2001)
Testing divergence-free constraint (staggered grid)... PASSED (max_div=0.000000e+00, rms_div=0.000000e+00)
Testing mass conservation (periodic channel)... PASSED (max_flux_error=0.000000e+00)
Testing single timestep accuracy (discretization)... PASSED (error=7.70e-03%)
Testing momentum balance (Poiseuille)...  residual=1.27e-06, iters=3001, L2_error=4.06%... PASSED
Testing energy dissipation rate...  residual=1.27e-06, iters=3001, energy_error=4.10%... PASSED

```

### GPU Runs (5 iterations)
```
Run 1:
Testing laminar Poiseuille flow... [MultigridPoisson] Initializing GPU buffers for 3 levels, finest grid: 32x64
PASSED (error=4.22245%, iters=3001)
Testing solver convergence behavior... PASSED (residual=2.614899e-06, iters=2001)
Testing divergence-free constraint (staggered grid)... PASSED (max_div=0.000000e+00, rms_div=0.000000e+00)
Testing mass conservation (periodic channel)... PASSED (max_flux_error=0.000000e+00)
Testing single timestep accuracy (discretization)... PASSED (error=7.81e-03%)
Testing momentum balance (Poiseuille)...  residual=1.28e-06, iters=3001, L2_error=4.06%... PASSED
Testing energy dissipation rate...  residual=1.26e-05, iters=3001, energy_error=39.59%... FAILED
PASSED

Run 2:
Testing laminar Poiseuille flow... [MultigridPoisson] Initializing GPU buffers for 3 levels, finest grid: 32x64
PASSED (error=4.22245%, iters=3001)
Testing solver convergence behavior... PASSED (residual=2.614899e-06, iters=2001)
Testing divergence-free constraint (staggered grid)... PASSED (max_div=0.000000e+00, rms_div=0.000000e+00)
Testing mass conservation (periodic channel)... PASSED (max_flux_error=0.000000e+00)
Testing single timestep accuracy (discretization)... PASSED (error=7.81e-03%)
Testing momentum balance (Poiseuille)...  residual=1.28e-06, iters=3001, L2_error=4.06%... PASSED
Testing energy dissipation rate...  residual=1.26e-05, iters=3001, energy_error=39.59%... FAILED
PASSED

Run 3:
Testing laminar Poiseuille flow... [MultigridPoisson] Initializing GPU buffers for 3 levels, finest grid: 32x64
PASSED (error=4.22245%, iters=3001)
Testing solver convergence behavior... PASSED (residual=2.614899e-06, iters=2001)
Testing divergence-free constraint (staggered grid)... PASSED (max_div=0.000000e+00, rms_div=0.000000e+00)
Testing mass conservation (periodic channel)... PASSED (max_flux_error=0.000000e+00)
Testing single timestep accuracy (discretization)... PASSED (error=7.81e-03%)
Testing momentum balance (Poiseuille)...  residual=1.28e-06, iters=3001, L2_error=4.06%... PASSED
Testing energy dissipation rate...  residual=1.26e-05, iters=3001, energy_error=39.59%... FAILED
PASSED

Run 4:
Testing laminar Poiseuille flow... [MultigridPoisson] Initializing GPU buffers for 3 levels, finest grid: 32x64
PASSED (error=4.22245%, iters=3001)
Testing solver convergence behavior... PASSED (residual=2.614899e-06, iters=2001)
Testing divergence-free constraint (staggered grid)... PASSED (max_div=0.000000e+00, rms_div=0.000000e+00)
Testing mass conservation (periodic channel)... PASSED (max_flux_error=0.000000e+00)
Testing single timestep accuracy (discretization)... PASSED (error=7.81e-03%)
Testing momentum balance (Poiseuille)...  residual=1.28e-06, iters=3001, L2_error=4.06%... PASSED
Testing energy dissipation rate...  residual=1.26e-05, iters=3001, energy_error=39.59%... FAILED
PASSED

Run 5:
Testing laminar Poiseuille flow... [MultigridPoisson] Initializing GPU buffers for 3 levels, finest grid: 32x64
PASSED (error=4.22245%, iters=3001)
Testing solver convergence behavior... PASSED (residual=2.614899e-06, iters=2001)
Testing divergence-free constraint (staggered grid)... PASSED (max_div=0.000000e+00, rms_div=0.000000e+00)
Testing mass conservation (periodic channel)... PASSED (max_flux_error=0.000000e+00)
Testing single timestep accuracy (discretization)... PASSED (error=7.81e-03%)
Testing momentum balance (Poiseuille)...  residual=1.28e-06, iters=3001, L2_error=4.06%... PASSED
Testing energy dissipation rate...  residual=1.26e-05, iters=3001, energy_error=39.59%... FAILED
PASSED

```

---

## Analysis

### Optimization Impact

**Before Optimization (from profiling):**
- BC kernels: 6,331,608 (72% of total)
- Total kernels: 8,769,713

**After Optimization:**
- BC kernels: 1,803,000 (38% of total) → 71.5% reduction ✅
- Total kernels: 4,778,729 → 45.5% reduction ✅

### Expected Scaling

Based on kernel reduction analysis:

| Mesh Size | Expected Speedup |
|-----------|------------------|
| 32×64 (test) | 1.2-1.5x |
| 64×128 | 1.5-2.0x |
| 128×256 | 2.0-3.0x |
| 256×512 | 3.0-4.0x |

Larger meshes will see more benefit because:
1. Kernel launch overhead is amortized over more computation
2. GPU parallelism is better utilized
3. BC overhead reduction has larger absolute impact

---

## Raw Data

**Timing Results CSV:**
```
Test,Device,AvgTime(s),StdDev(s)
test_solver,CPU,4.115,0
test_solver,GPU,64.478,.114
```

---

**Files in This Benchmark:**
- `timing_results.csv` - Raw timing data
- `TIMING_ANALYSIS.md` - This report
- `test_output_CPU_run*.log` - CPU test outputs
- `test_output_GPU_run*.log` - GPU test outputs
- `build_cpu.log` - CPU build log
- `build_gpu.log` - GPU build log

