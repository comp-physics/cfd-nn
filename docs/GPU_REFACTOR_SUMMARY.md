# GPU Offloading Refactor - Simplified Architecture

## Summary

This refactor simplifies the GPU offloading strategy from a complex multi-layered approach with manual state tracking to a clean, simple architecture where:

1. **Data is mapped once** at solver initialization
2. **Data stays on GPU** for the entire solver lifetime  
3. **All kernels use `is_device_ptr`** instead of temporary `map(to:/from:)`
4. **No mapping conflicts** between solver and turbulence models
5. **Minimal data transfers** - only for I/O operations

## Problems with Old Approach

### 1. Multiple Conflicting Strategies
- Used `map(alloc:)` for persistent arrays
- Used `map(to:/from:)` for temporary transfers
- Mixed `DeviceArray` wrapper classes with raw pointers
- Manual state tracking (`gpu_ready_`, `on_device_`) prone to errors

### 2. Double Mapping Errors
Turbulence models used temporary `map(to:/from:)` clauses for arrays already mapped by the solver:

```cpp
// PROBLEM: nu_t already mapped by solver!
#pragma omp target teams distribute parallel for \
    map(to: dudx[0:n], ...) \
    map(from: nu_t[0:n])  // ERROR: nu_t already on GPU!
```

This caused runtime errors and the infamous workaround:
```cpp
setenv("NNCFD_FORCE_CPU_TURB", "1", 1);  // HACK!
```

### 3. Excessive Data Movement
- `sync_to_gpu()` copied all fields before every kernel
- `sync_from_gpu()` downloaded everything after kernels
- Wasted bandwidth and GPU time

## New Simplified Architecture

### Initialization (Once)
```cpp
RANSSolver::RANSSolver(...) {
    // ... setup fields on CPU ...
    
#ifdef USE_GPU_OFFLOAD
    initialize_gpu_buffers();  // Map all data with map(to:)
#endif
}

void RANSSolver::initialize_gpu_buffers() {
    // Get pointers once
    velocity_u_ptr_ = velocity_.u_field().data().data();
    // ... etc for all fields
    
    // Map ALL data to GPU and copy initial values
    #pragma omp target enter data map(to: velocity_u_ptr_[0:field_total_size_])
    #pragma omp target enter data map(to: pressure_ptr_[0:field_total_size_])
    // ... etc for all fields
    
    gpu_ready_ = true;
}
```

### Computation (No Data Transfers!)
```cpp
void RANSSolver::apply_velocity_bc() {
#ifdef USE_GPU_OFFLOAD
    if (gpu_ready_) {
        double* u_ptr = velocity_u_ptr_;
        double* v_ptr = velocity_v_ptr_;
        
        // Use is_device_ptr - data already on GPU!
        #pragma omp target teams distribute parallel for \
            is_device_ptr(u_ptr, v_ptr)
        for (int idx = 0; idx < n_bc; ++idx) {
            // Apply BCs - NO data transfer!
        }
        return;
    }
#endif
    // CPU fallback
}
```

### Turbulence Models (Use Mapped Data)
```cpp
void MixingLengthModel::update(..., ScalarField& nu_t) {
#ifdef USE_GPU_OFFLOAD
    if (omp_get_num_devices() > 0) {
        double* nu_t_ptr = nu_t.data().data();  // Already on GPU!
        
        // Local gradients use temporary map(to:), nu_t uses is_device_ptr
        #pragma omp target teams distribute parallel for \
            map(to: dudx_ptr[0:n], dudy_ptr[0:n], ...) \
            is_device_ptr(nu_t_ptr)  // No conflict!
        for (int idx = 0; idx < n_cells; ++idx) {
            // Compute nu_t directly on GPU
            nu_t_ptr[idx] = ...;
        }
        return;
    }
#endif
    // CPU fallback
}
```

### I/O (Minimal Transfers)
```cpp
void RANSSolver::write_vtk(const std::string& filename) {
#ifdef USE_GPU_OFFLOAD
    if (gpu_ready_) {
        // Download ONLY fields needed for output
        // Data stays on GPU!
        #pragma omp target update from(velocity_u_ptr_[0:field_total_size_])
        #pragma omp target update from(velocity_v_ptr_[0:field_total_size_])
        #pragma omp target update from(pressure_ptr_[0:field_total_size_])
    }
#endif
    // Write from CPU arrays (now up-to-date)
}
```

### Cleanup (Once)
```cpp
RANSSolver::~RANSSolver() {
#ifdef USE_GPU_OFFLOAD
    cleanup_gpu_buffers();
#endif
}

void RANSSolver::cleanup_gpu_buffers() {
    // Copy results back, then free
    #pragma omp target exit data map(from: velocity_u_ptr_[0:field_total_size_])
    #pragma omp target exit data map(from: pressure_ptr_[0:field_total_size_])
    // ...
    
    // Delete work arrays without copying
    #pragma omp target exit data map(delete: conv_u_ptr_[0:field_total_size_])
    // ...
}
```

## Key Changes Made

### 1. Solver (`src/solver.cpp`, `include/solver.hpp`)
- ✅ Changed `map(alloc:)` to `map(to:)` in `initialize_gpu_buffers()`
- ✅ Changed `map(delete:)` to `map(from:)` in `cleanup_gpu_buffers()` for solution fields
- ✅ Removed `NNCFD_FORCE_CPU_TURB` environment variable hack
- ✅ Updated `apply_velocity_bc()` to use `is_device_ptr`
- ✅ Simplified `sync_from_gpu()` to only sync I/O fields

### 2. Turbulence Models
- ✅ `turbulence_baseline.cpp`: Use `is_device_ptr(nu_t_ptr)` instead of `map(from:)`
- ✅ `turbulence_transport.cpp`: Use `is_device_ptr(k_ptr, omega_ptr, nu_t_ptr)`
- ✅ `turbulence_gep.cpp`: Use `is_device_ptr(nu_t_ptr)`
- ✅ `turbulence_earsm.cpp`: Use `is_device_ptr(k, omega)` in GPU kernels

### 3. Removed Complexity
- 🗑️ No more `DeviceArray` wrapper usage for solver fields
- 🗑️ No more manual state tracking conflicts
- 🗑️ No more forced CPU path for turbulence models
- 🗑️ No more excessive `sync_to_gpu()` calls

## Benefits

### Performance
- **Eliminates redundant transfers**: Data stays on GPU between kernels
- **Reduces PCIe overhead**: Only transfer for I/O, not every kernel
- **Better GPU utilization**: More time computing, less time copying

### Simplicity
- **Single mapping per array**: No double-mapping conflicts
- **Clear ownership**: Solver owns GPU mappings, turbulence models access via `is_device_ptr`
- **Less code**: Removed wrapper classes, manual tracking, workarounds

### Correctness
- **No mapping conflicts**: Solver and turbulence models work together seamlessly
- **Consistent state**: OpenMP manages GPU memory, not manual tracking
- **Easier debugging**: One clear strategy, not multiple competing approaches

## Testing

The refactored code:
1. ✅ Compiles successfully with NVHPC compiler
2. ✅ Passes all unit tests on CPU nodes
3. ⏳ Needs validation on GPU nodes (CPU vs GPU results comparison)

## Migration Guide

If you have custom turbulence models or kernels:

### Old Pattern (Don't Use)
```cpp
double* nu_t_ptr = nu_t.data().data();
#pragma omp target teams distribute parallel for \
    map(from: nu_t_ptr[0:n])  // BAD: nu_t already mapped by solver!
```

### New Pattern (Use This)
```cpp
double* nu_t_ptr = nu_t.data().data();
#pragma omp target teams distribute parallel for \
    is_device_ptr(nu_t_ptr)  // GOOD: Tell OpenMP it's already on GPU
```

For arrays created locally (gradients, features, etc.):
```cpp
std::vector<double> local_gradients(n);
double* grad_ptr = local_gradients.data();

#pragma omp target teams distribute parallel for \
    map(to: grad_ptr[0:n]) \        // OK: Local array, temporary transfer
    is_device_ptr(nu_t_ptr)          // Solver's array, already mapped
```

## Future Work

1. Consider using structured data regions for related fields
2. Explore async data transfers for overlapping I/O and computation
3. Profile and optimize transfer patterns for snapshot output
4. Add GPU memory usage reporting in verbose mode



