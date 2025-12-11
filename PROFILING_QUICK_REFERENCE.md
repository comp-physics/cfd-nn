# GPU Profiling Quick Reference

**Date:** December 10, 2025  
**Job ID:** 2653116  
**Results:** `/storage/home/hcoda1/6/sbryngelson3/cfd-nn/gpu_profile_results_20251210_185722/`

---

## 📊 Key Numbers

| Metric | Value |
|--------|-------|
| **Total GPU Kernels Launched** | 8,769,713 |
| **Upload Operations** | 265,440 |
| **Download Operations** | 271,055 |
| **Total Transfers** | 536,495 |
| **Transfer/Kernel Ratio** | 6.1% |
| **Test Duration** | 3:44 minutes |

---

## 🔝 Top 5 Kernels (by launch count)

| Rank | Launches | % Total | Kernel |
|------|----------|---------|--------|
| 1 | 6,331,608 | 72.2% | `MultigridPoissonSolver::apply_bc` |
| 2 | 1,460,448 | 16.7% | `MultigridPoissonSolver::smooth` |
| 3 | 244,908 | 2.8% | `MultigridPoissonSolver::compute_residual` |
| 4 | 243,408 | 2.8% | `MultigridPoissonSolver::vcycle` |
| 5 | 123,204 | 1.4% | `MultigridPoissonSolver::compute_max_residual` |

**Top 5 account for 95.9% of all kernel launches!**

---

## ⚠️ Primary Bottleneck

### **Boundary Condition Over-Application**

```
Metric:                          Observed    Expected    Ratio
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BCs per V-cycle:                   26.0         3.0      8.7x
BCs per smooth:                     4.3         1.0      4.3x
Total BC calls:                 6,331,608     730,000    8.7x
```

**Conclusion:** BCs are being applied 8.7x more than necessary!

---

## 🎯 Optimization Targets

### **By Impact (Estimated):**

| Optimization | Target Kernel | Reduction | Speedup |
|--------------|---------------|-----------|---------|
| **Reduce BC freq** | `apply_bc` | 70% fewer calls | **50%** |
| **GPU residual** | `compute_max_residual` | 100k transfers | **20%** |
| **Kernel fusion** | `smooth` + `apply_bc` | Launch overhead | **15%** |

**Combined potential: 2-3x overall speedup**

---

## 📈 Kernel Efficiency Analysis

### **Multigrid Ratios:**
```
Smooths per V-cycle:     6.0  ✅ Good (typical is 4-8)
BCs per V-cycle:        26.0  ❌ Bad (should be 3)
BCs per smooth:          4.3  ❌ Bad (should be 1)
Residuals per V-cycle:   1.0  ✅ Perfect
```

### **RANS Solver:**
```
Convection calls:     11,402  ✅ Matches timesteps
Diffusion calls:      11,402  ✅ Matches timesteps
Velocity corrections: 22,804  ✅ Correct (2x convection)
Divergence checks:     5,701  ✅ Correct (0.5x convection)
```

**Conclusion:** RANS solver is efficient, Poisson solver needs work.

---

## 🔍 Memory Transfer Breakdown

### **By Component:**
```
Component                  Transfers    % Total
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Poisson residuals:         ~123,000      23%
Turbulence updates:         ~45,000       8%
Other/profiling:           ~368,000      69%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL:                      536,495     100%
```

**Note:** "Other" includes profiling overhead from `NVCOMPILER_ACC_NOTIFY=3`

---

## 📁 Generated Files

| File | Size | Description |
|------|------|-------------|
| `kernel_launches.log` | 2.9 GB | Full kernel launch trace (huge!) |
| `unique_kernels.txt` | 878 B | Kernel statistics |
| `timing_comparison.csv` | 36 B | CPU vs GPU times (incomplete) |
| `build.log` | 1.7 KB | Build output |

---

## 🚦 Status Indicators

### ✅ **Working Correctly:**
- GPU kernels launching
- Results match CPU
- No crashes
- Physics tests pass

### ⚠️ **Needs Optimization:**
- BC application frequency
- Transfer rate (6% vs target <0.1%)
- Convergence check synchronization

### ⏸️ **Not Yet Tested:**
- Large mesh scaling
- Production workloads
- Multi-GPU (future)

---

## 🛠️ How to Re-Run Profiling

### **Full Suite:**
```bash
sbatch gpu_profile_simple.sh
```

### **Quick Kernel Count:**
```bash
# On GPU node:
export NVCOMPILER_ACC_NOTIFY=1
export OMP_TARGET_OFFLOAD=MANDATORY
./test_solver 2>&1 | grep "launch CUDA kernel" | wc -l
```

### **Transfer Count Only:**
```bash
export NVCOMPILER_ACC_NOTIFY=2
./test_solver 2>&1 | grep -E "(upload|download)" | wc -l
```

### **With nsys (detailed):**
```bash
module load cuda/11.8.0
nsys profile -o profile.nsys-rep ./test_solver
nsys stats profile.nsys-rep
```

---

## 📊 Performance Baseline

| Configuration | Status |
|---------------|--------|
| **32x64 mesh, laminar** | ✅ Tested |
| **32x64 mesh, turbulent** | ✅ Tested |
| **128x128 mesh** | ⏸️ Not yet profiled |
| **256x256 mesh** | ⏸️ Not yet profiled |
| **Production cases** | ⏸️ Not yet profiled |

**Next:** Run scaling study to characterize performance vs. mesh size.

---

## 🎯 Action Items

### **Immediate:**
1. [ ] Optimize BC application in `poisson_solver_multigrid.cpp`
2. [ ] Move residual max to GPU
3. [ ] Re-profile to measure impact

### **Short-term:**
4. [ ] Scaling study (64x64, 128x128, 256x256)
5. [ ] Test all turbulence models individually
6. [ ] Production case validation

### **Long-term:**
7. [ ] Kernel fusion optimizations
8. [ ] nsys/ncu micro-optimizations
9. [ ] Multi-GPU support (if needed)

---

## 📞 Quick Commands

```bash
# View profiling results
cd /storage/home/hcoda1/6/sbryngelson3/cfd-nn/gpu_profile_results_20251210_185722
less unique_kernels.txt

# Check kernel count
wc -l unique_kernels.txt

# View latest job output
tail -100 /storage/home/hcoda1/6/sbryngelson3/cfd-nn/gpu_profile_2653116.out

# Re-run profiling
cd /storage/home/hcoda1/6/sbryngelson3/cfd-nn
sbatch gpu_profile_simple.sh

# Monitor job
squeue -u sbryngelson3
```

---

**For detailed analysis, see:** `GPU_PROFILING_ANALYSIS.md`  
**For overall status, see:** `GPU_STATUS_SUMMARY.md`


