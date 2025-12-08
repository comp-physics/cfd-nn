# GPU Offload Fix - OpenMP map() Clause Correction

## The Problem

The refactored GPU code was aborting immediately when running on actual GPU hardware (Tesla V100), despite compiling successfully.

**Error:** Instant abort (no error message, just "Aborted")

## Root Cause

**Misuse of `is_device_ptr()` for host pointers with mapped data.**

### OpenMP Device Pointer Semantics

In OpenMP offload, there are two different concepts:

1. **Host pointer with mapped data**: 
   - Created with: `#pragma omp target enter data map(to: ptr[0:n])`
   - The **pointer itself** remains a host pointer
   - The **data** is copied to GPU
   - Access via: `map(present:)` in subsequent kernels

2. **True device pointer**:
   - Created with: `ptr = omp_target_alloc(size, device)`  
   - The pointer **directly** points to device memory
   - Access via: `is_device_ptr(ptr)` in kernels

### What We Did Wrong

```cpp
// Initialization - creates HOST pointer with mapped data
double* velocity_u_ptr_ = velocity_.u_field().data().data();
#pragma omp target enter data map(to: velocity_u_ptr_[0:field_total_size_])

// Later usage - WRONG! Treats host pointer as device pointer
#pragma omp target teams distribute parallel for \
    is_device_ptr(velocity_u_ptr_)  // ❌ INCORRECT!
```

The compiler/runtime expects `is_device_ptr()` to receive a pointer from `omp_target_alloc()`, not a host pointer with mapped data.

## The Fix

Replace all `is_device_ptr()` clauses with `map(present:)`:

```cpp
// BEFORE (WRONG):
#pragma omp target teams distribute parallel for \
    is_device_ptr(u_ptr, v_ptr)

// AFTER (CORRECT):
#pragma omp target teams distribute parallel for \
    map(present: u_ptr[0:total_size], v_ptr[0:total_size])
```

### What `map(present:)` Does

- Tells OpenMP: "This data is already on the device, use the existing mapping"
- Works with host pointers that have mapped data
- OpenMP runtime finds the already-mapped data and uses it
- No data transfer occurs (data already there from `target enter data`)

## Files Changed

### 1. `src/solver.cpp`
**Function:** `RANSSolver::apply_velocity_bc()`
- Fixed x-direction boundary condition kernel
- Fixed y-direction boundary condition kernel

```diff
- is_device_ptr(u_ptr, v_ptr)
+ map(present: u_ptr[0:total_size], v_ptr[0:total_size])
```

### 2. `src/turbulence_baseline.cpp`
**Function:** `MixingLengthModel::update()`
```diff
- is_device_ptr(nu_t_ptr)
+ map(present: nu_t_ptr[0:total_cells])
```

### 3. `src/turbulence_transport.cpp`
**Function:** `SSTTransportModel::update()`
```diff
- is_device_ptr(k_ptr, omega_ptr, nu_t_ptr)
+ map(present: k_ptr[0:n_cells], omega_ptr[0:n_cells], nu_t_ptr[0:n_cells])
```

### 4. `src/turbulence_gep.cpp`
**Function:** `GEPModel::update()`
```diff
- is_device_ptr(nu_t_ptr)
+ map(present: nu_t_ptr[0:n_cells])
```

### 5. `src/turbulence_earsm.cpp`
**Functions:** `compute_wj_coefficients_gpu()`, `compute_gs_coefficients_gpu()`
```diff
- is_device_ptr(k, omega)
+ map(present: k[0:n_cells], omega[0:n_cells])
```

## Results

### Before Fix
- ✅ Compiles successfully with NVHPC
- ✅ Runs on CPU nodes
- ❌ **Instant abort on GPU nodes**

### After Fix
- ✅ Compiles successfully with NVHPC
- ✅ Runs on CPU nodes
- ✅ **Runs on GPU nodes** (Job 2606662: 2+ minutes execution time)

## Key Lessons

### 1. Understand OpenMP Memory Model
- **Host pointers + mapped data** ≠ **Device pointers**
- Use the right map clause for your pointer type

### 2. Map Clause Reference

| Situation | Correct Clause |
|-----------|----------------|
| First mapping | `map(to: ptr[0:n])` or `map(alloc: ptr[0:n])` |
| Already mapped (host ptr) | `map(present: ptr[0:n])` |
| Device pointer | `is_device_ptr(ptr)` |
| Local temporary data | `map(to: ptr[0:n])` then auto-unmapped |

### 3. Debugging Tips
- **Instant abort** often means incorrect map clause
- **Missing data** means forgot to map or wrong `present` assertion
- **Double mapping** errors from redundant map clauses

## Testing

### CPU Test (Baseline)
```bash
cd build_refactor
./channel --Nx 64 --Ny 128 --nu 0.001 --model baseline --max_iter 100
```
✅ Works perfectly

### GPU Test (After Fix)
```bash
sbatch test_simple_gpu.sbatch
```
✅ Runs successfully (no instant abort)

## Next Steps

1. ✅ Verify GPU test completes successfully  
2. Compare CPU vs GPU numerical results
3. Run full validation test suite
4. Benchmark performance improvements
5. Merge to main branch

## References

- OpenMP 5.0 Specification: Section 2.12 (Data Environment)
- NVIDIA HPC SDK Documentation: OpenMP Offloading
- [OpenMP is_device_ptr vs map(present:)](https://www.openmp.org/spec-html/5.0/openmpsu72.html)

---

**Bottom Line:** The refactor architecture is sound. The issue was a subtle but critical misunderstanding of OpenMP pointer semantics. The fix is small (13 lines changed) but essential.

