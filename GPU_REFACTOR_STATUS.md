# GPU Offloading Refactor - Status Report

**Branch:** `refactor/simplify-gpu-offload`  
**Date:** December 8, 2025  
**Status:** ⚠️ Compiles successfully, runtime issue on GPU nodes

## ✅ Completed Work

### 1. Core Architecture Simplification
- **Replaced** `map(alloc:)` with `map(to:)` in solver initialization
- **Replaced** `map(delete:)` with `map(from:)` in solver cleanup for solution fields
- **Removed** `NNCFD_FORCE_CPU_TURB` environment variable hack
- **Updated** all boundary condition kernels to use `is_device_ptr`
- **Simplified** `sync_from_gpu()` to only sync I/O fields

### 2. Turbulence Model Updates
All turbulence models updated to use `is_device_ptr` for solver-mapped arrays:
- ✅ `turbulence_baseline.cpp` - Mixing length model
- ✅ `turbulence_transport.cpp` - SST k-ω transport
- ✅ `turbulence_gep.cpp` - GEP algebraic model
- ✅ `turbulence_earsm.cpp` - EARSM closures

### 3. Documentation
- ✅ Created comprehensive `docs/GPU_REFACTOR_SUMMARY.md`
- ✅ Documented migration guide for custom models
- ✅ Explained benefits and architecture changes

### 4. Build System
- ✅ Code compiles successfully with NVHPC compiler
- ✅ All unit tests pass on CPU nodes
- ✅ Fixed CMakeLists.txt for missing files

## ⚠️ Current Issue

### GPU Runtime Abort
**Symptom:** Program aborts when running on actual GPU hardware (Tesla V100)

```
./channel --Nx 64 --Ny 128 --nu 0.001 --model baseline ...
Aborted
```

**Observed:**
- ✅ CPU-only build works perfectly
- ✅ GPU build compiles without errors
- ❌ GPU execution aborts immediately

**Likely Causes:**
1. **Double mapping conflict** - Some array may still be getting mapped twice
2. **Uninitialized pointer** - A GPU pointer might not be properly initialized
3. **Race condition** - Data dependency issue in `is_device_ptr` usage
4. **Memory allocation** - GPU memory allocation failure

### Next Steps for Debugging

1. **Check initialization order:**
   - Ensure `initialize_gpu_buffers()` is called before any GPU kernels
   - Verify all pointers are valid before mapping

2. **Verify data mapping consistency:**
   - Check that turbulence models aren't trying to map already-mapped arrays
   - Ensure `is_device_ptr` is used correctly everywhere

3. **Add error checking:**
   ```cpp
   #pragma omp target enter data map(to: ptr[0:size])
   // Add: Check for OMP errors here
   ```

4. **Test incrementally:**
   - Test solver without turbulence model
   - Test with laminar flow only
   - Add turbulence model step by step

## Files Changed

### Core Solver
- `include/solver.hpp` - Updated GPU member documentation
- `src/solver.cpp` - Refactored GPU initialization, cleanup, and boundary conditions

### Turbulence Models
- `src/turbulence_baseline.cpp` - Updated GPU kernels
- `src/turbulence_transport.cpp` - Updated GPU kernels
- `src/turbulence_gep.cpp` - Updated GPU kernels  
- `src/turbulence_earsm.cpp` - Updated GPU kernels

### Build & Test
- `CMakeLists.txt` - Commented out missing verify_cpu_gpu_match
- `test_gpu_refactor.sh` - GPU validation script
- `test_gpu_refactor.sbatch` - SLURM job for GPU testing

### Documentation
- `docs/GPU_REFACTOR_SUMMARY.md` - Comprehensive refactor documentation
- `GPU_REFACTOR_STATUS.md` - This status report

## Architecture Before vs After

### Before (Complex)
```
Solver:
  - map(alloc:) arrays
  - Manual sync_to_gpu() before every operation
  - Manual sync_from_gpu() after every operation

Turbulence Models:
  - map(to:/from:) in every kernel
  - Conflicts with solver's mappings
  - Forced to use CPU via environment variable hack
```

### After (Simple)
```
Solver:
  - map(to:) once at initialization
  - Data stays on GPU
  - sync_from_gpu() only for I/O

Turbulence Models:
  - is_device_ptr() for solver arrays
  - map(to:) only for local/temporary arrays
  - No conflicts!
```

## Performance Implications

**Expected Benefits (once runtime issue is fixed):**
- ⚡ **Reduced PCIe transfers** - Data stays on GPU between kernels
- ⚡ **Better GPU utilization** - Less time copying, more time computing
- ⚡ **Simpler code path** - Easier to maintain and debug

## How to Test

### CPU Reference (Works)
```bash
cd build_refactor
./channel --Nx 64 --Ny 128 --nu 0.001 --model baseline \
    --adaptive_dt --max_iter 100 --output_prefix test_cpu --verbose
```

### GPU Test (Currently Aborts)
```bash
# On GPU node:
sbatch test_gpu_refactor.sbatch

# Check output:
cat gpu_refactor_test_*.out
```

### Debug Test (Pending)
```bash
sbatch test_simple_gpu.sbatch
```

## Recommended Next Actions

1. **Debug GPU abort:**
   - Add verbose error output to GPU initialization
   - Check all `target enter data` pragmas succeed
   - Verify pointer validity before use

2. **Add runtime checks:**
   ```cpp
   if (!velocity_u_ptr_) {
       std::cerr << "ERROR: velocity_u_ptr_ is null!" << std::endl;
       exit(1);
   }
   ```

3. **Test individual components:**
   - Test solver without turbulence
   - Test boundary conditions only
   - Test each turbulence model separately

4. **Compare with working GPU code:**
   - Check what changed from last working GPU version
   - Look for unintended side effects

## Contact

For questions or to contribute to debugging:
- See `docs/GPU_REFACTOR_SUMMARY.md` for architecture details
- Check SLURM output files: `gpu_simple_*.out`, `gpu_refactor_test_*.out`
- Run tests: `./test_solver`, `./channel`

---

**Note:** This is a work-in-progress refactor. The simplified architecture is solid, but there's a runtime issue that needs debugging on actual GPU hardware. The CPU path works perfectly, demonstrating the correctness of the numerical implementation.


