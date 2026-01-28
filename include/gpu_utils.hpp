#pragma once

/// @file gpu_utils.hpp
/// @brief GPU utilities for OpenMP target offloading
///
/// Provides RAII-style memory management, device pointer helpers, and macros.
///
/// ## NVHPC Workaround: Device Pointer Pattern
///
/// NVHPC (25.x) has a bug where mapped host pointers used directly in OpenMP target
/// regions sometimes receive the HOST address instead of the device address. This
/// causes silent memory corruption - kernels read/write wrong memory locations.
///
/// **REQUIRED PATTERN** (all GPU kernels must follow this):
/// ```cpp
/// // 1. Get device pointer explicitly using omp_get_mapped_ptr
/// double* u_dev = gpu::dev_ptr(velocity_u_ptr_);
///
/// // 2. Use is_device_ptr clause to tell compiler this is a device address
/// #pragma omp target teams distribute parallel for is_device_ptr(u_dev)
/// for (int i = 0; i < n; ++i) {
///     u_dev[i] = ...;
/// }
/// ```
///
/// **FORBIDDEN PATTERNS** (will cause silent corruption):
/// ```cpp
/// // BAD: Local pointer alias without dev_ptr
/// double* u = velocity_u_ptr_;
/// #pragma omp target teams distribute parallel for
/// for (...) { u[i] = ...; }  // WRONG - may use host address!
///
/// // BAD: Raw pointer with map(present:)
/// #pragma omp target teams distribute parallel for map(present: u[0:n])
/// for (...) { u[i] = ...; }  // WRONG - may use host address!
///
/// // BAD: firstprivate with pointer
/// #pragma omp target teams distribute parallel for firstprivate(u)
/// for (...) { u[i] = ...; }  // WRONG - passes host address!
/// ```
///
/// The dev_ptr() function aborts with a clear error if the pointer is not mapped,
/// providing fail-fast behavior instead of silent corruption.
///
/// This pattern has no measurable performance impact - it's one pointer lookup per
/// kernel launch, not per element.

#include <vector>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

namespace nncfd {
namespace gpu {

/// Check if GPU offloading is available at runtime
inline bool is_gpu_available() {
#ifdef USE_GPU_OFFLOAD
    return omp_get_num_devices() > 0;
#else
    return false;
#endif
}

/// Get number of GPU devices
inline int num_devices() {
#ifdef USE_GPU_OFFLOAD
    return omp_get_num_devices();
#else
    return 0;
#endif
}

/// Check if GPU execution path should be used
/// If GPU offloading is enabled and a device is available, always use it
inline bool should_use_gpu_path() {
    return is_gpu_available();
}

/// RAII wrapper for GPU-resident array
/// Uploads data on construction, keeps on device, downloads on request
template<typename T>
class DeviceArray {
public:
    DeviceArray() : data_(nullptr), size_(0), on_device_(false) {}
    
    /// Construct from existing vector - uploads to GPU
    explicit DeviceArray(std::vector<T>& host_data) 
        : data_(host_data.data()), size_(host_data.size()), on_device_(false) {
        upload();
    }
    
    /// Construct from raw pointer and size - uploads to GPU
    DeviceArray(T* ptr, size_t n) : data_(ptr), size_(n), on_device_(false) {
        upload();
    }
    
    ~DeviceArray() {
        release();
    }
    
    // Non-copyable
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    
    // Movable
    DeviceArray(DeviceArray&& other) noexcept 
        : data_(other.data_), size_(other.size_), on_device_(other.on_device_) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.on_device_ = false;
    }
    
    DeviceArray& operator=(DeviceArray&& other) noexcept {
        if (this != &other) {
            release();
            data_ = other.data_;
            size_ = other.size_;
            on_device_ = other.on_device_;
            other.data_ = nullptr;
            other.size_ = 0;
            other.on_device_ = false;
        }
        return *this;
    }
    
    /// Upload host data to device
    void upload() {
#ifdef USE_GPU_OFFLOAD
        if (data_ && size_ > 0 && !on_device_) {
            #pragma omp target enter data map(to: data_[0:size_])
            on_device_ = true;
        }
#endif
    }
    
    /// Download device data to host
    void download() {
#ifdef USE_GPU_OFFLOAD
        if (data_ && size_ > 0 && on_device_) {
            #pragma omp target update from(data_[0:size_])
        }
#endif
    }
    
    /// Update device from host (for data that changed on host)
    void update_device() {
#ifdef USE_GPU_OFFLOAD
        if (data_ && size_ > 0 && on_device_) {
            #pragma omp target update to(data_[0:size_])
        }
#endif
    }
    
    /// Release device memory
    void release() {
#ifdef USE_GPU_OFFLOAD
        if (data_ && size_ > 0 && on_device_) {
            #pragma omp target exit data map(delete: data_[0:size_])
            on_device_ = false;
        }
#endif
    }
    
    /// Get raw pointer (valid on both host and device when mapped)
    T* data() { return data_; }
    const T* data() const { return data_; }
    
    size_t size() const { return size_; }
    bool is_on_device() const { return on_device_; }
    
private:
    T* data_;
    size_t size_;
    bool on_device_;
};

/// Allocate device-only memory (not backed by host)
template<typename T>
class DeviceOnlyArray {
public:
    DeviceOnlyArray() : data_(nullptr), size_(0) {}
    
    explicit DeviceOnlyArray(size_t n) : size_(n) {
        allocate(n);
    }
    
    ~DeviceOnlyArray() {
        release();
    }
    
    // Non-copyable
    DeviceOnlyArray(const DeviceOnlyArray&) = delete;
    DeviceOnlyArray& operator=(const DeviceOnlyArray&) = delete;
    
    // Movable
    DeviceOnlyArray(DeviceOnlyArray&& other) noexcept 
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }
    
    DeviceOnlyArray& operator=(DeviceOnlyArray&& other) noexcept {
        if (this != &other) {
            release();
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    void allocate(size_t n) {
#ifdef USE_GPU_OFFLOAD
        if (data_) release();
        size_ = n;
        // Allocate on host first, then map to device
        data_ = new T[n]();
        #pragma omp target enter data map(alloc: data_[0:size_])
#else
        if (data_) delete[] data_;
        size_ = n;
        data_ = new T[n]();
#endif
    }
    
    void release() {
#ifdef USE_GPU_OFFLOAD
        if (data_ && size_ > 0) {
            #pragma omp target exit data map(delete: data_[0:size_])
            delete[] data_;
            data_ = nullptr;
            size_ = 0;
        }
#else
        if (data_) {
            delete[] data_;
            data_ = nullptr;
            size_ = 0;
        }
#endif
    }
    
    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t size() const { return size_; }
    
private:
    T* data_;
    size_t size_;
};

/// Scope guard for GPU data region
/// Ensures data is uploaded at start and downloaded at end
class GPUDataScope {
public:
    template<typename... Arrays>
    explicit GPUDataScope(Arrays&... arrays) {
        upload_all(arrays...);
    }
    
    ~GPUDataScope() {
        // Download handled by individual arrays if needed
    }
    
private:
    template<typename T, typename... Rest>
    void upload_all(DeviceArray<T>& first, Rest&... rest) {
        first.upload();
        if constexpr (sizeof...(rest) > 0) {
            upload_all(rest...);
        }
    }
    
    void upload_all() {}
};

#ifdef USE_GPU_OFFLOAD
/// GPU runtime verification functions (compiled only when USE_GPU_OFFLOAD=ON)
/// These are defined in src/gpu_init.cpp to enforce compile-time separation
void verify_device_available();
bool is_pointer_present(void* ptr);

/// Get device pointer for an OpenMP-mapped host pointer
/// Uses omp_get_mapped_ptr (OpenMP 5.1) to convert host -> device pointer
/// Returns nullptr if pointer is not mapped or host_ptr is nullptr
///
/// NVHPC WORKAROUND: NVHPC sometimes fails to use the device address associated
/// with a mapped host pointer when that pointer appears in certain target constructs.
/// Using omp_get_mapped_ptr + is_device_ptr forces the compiler to use the correct
/// device address.
///
/// Usage pattern:
///   double* u_dev = gpu::dev_ptr(velocity_u_ptr_);  // Asserts mapping exists
///   #pragma omp target teams distribute parallel for is_device_ptr(u_dev)
///   for (...) u_dev[i] = ...
///
template<typename T>
inline T* get_device_ptr(T* host_ptr) {
    if (host_ptr == nullptr) return nullptr;
    int device = omp_get_default_device();
    void* dev_ptr = omp_get_mapped_ptr(host_ptr, device);
    // omp_get_mapped_ptr returns nullptr if pointer is not mapped
    return static_cast<T*>(dev_ptr);
}

/// Get device pointer with assertion that mapping exists
/// Aborts if host_ptr is nullptr or not mapped to device
/// Use this when the pointer MUST be mapped (i.e., in compute kernels)
///
/// @param host_ptr  Host pointer that should be mapped to device
/// @param context   Optional field name for debugging (e.g., "velocity_u")
template<typename T>
inline T* dev_ptr(T* host_ptr, const char* context = nullptr) {
    if (host_ptr == nullptr) {
        if (context) {
            std::fprintf(stderr, "dev_ptr(%s): host_ptr is nullptr\n", context);
        } else {
            std::fprintf(stderr, "dev_ptr: host_ptr is nullptr\n");
        }
        std::abort();
    }
    int device = omp_get_default_device();
    void* dev = omp_get_mapped_ptr(host_ptr, device);
    if (dev == nullptr) {
        // Additional diagnostic: check if pointer is "present" on device
        int is_present = omp_target_is_present(host_ptr, device);

        std::fprintf(stderr, "\n========================================\n");
        std::fprintf(stderr, "GPU POINTER MAPPING FAILURE\n");
        std::fprintf(stderr, "========================================\n");
        if (context) {
            std::fprintf(stderr, "Field: %s\n", context);
        }
        std::fprintf(stderr, "Host pointer:  %p\n", static_cast<void*>(host_ptr));
        std::fprintf(stderr, "Target device: %d\n", device);
        std::fprintf(stderr, "omp_target_is_present: %s\n", is_present ? "YES" : "NO");
        std::fprintf(stderr, "\n");

        if (!is_present) {
            std::fprintf(stderr, "Diagnosis: Pointer was NEVER MAPPED to device.\n");
            std::fprintf(stderr, "  - Check that map_fields_to_gpu() was called before step()\n");
            std::fprintf(stderr, "  - Check that std::vector didn't reallocate (invalidating pointer)\n");
            std::fprintf(stderr, "  - Verify field is included in enter data map() directive\n");
        } else {
            std::fprintf(stderr, "Diagnosis: Pointer IS present but omp_get_mapped_ptr returned NULL.\n");
            std::fprintf(stderr, "  - This may be a compiler/runtime bug\n");
            std::fprintf(stderr, "  - Try recompiling with latest NVHPC version\n");
        }
        std::fprintf(stderr, "========================================\n\n");
        std::abort();
    }
    return static_cast<T*>(dev);
}

/// Const overload for dev_ptr
template<typename T>
inline const T* dev_ptr(const T* host_ptr, const char* context = nullptr) {
    return dev_ptr(const_cast<T*>(host_ptr), context);
}

/// Synchronize OpenMP target tasks (wait for deferred target regions to complete)
/// Use this before reading results back to host after using `nowait` target regions.
/// NOTE: This only synchronizes OpenMP target tasks, NOT direct CUDA kernel launches.
///       CUDA kernels (e.g., in mg_cuda_kernels.cpp) use cudaStreamSynchronize instead.
inline void sync() {
    #pragma omp taskwait
}
#else
/// CPU: sync is a no-op
inline void sync() {}

/// CPU stubs for dev_ptr - just return the host pointer unchanged
/// These allow code using gpu::dev_ptr() to compile on CPU builds
template<typename T>
inline T* get_device_ptr(T* host_ptr) {
    return host_ptr;
}

template<typename T>
inline T* dev_ptr(T* host_ptr, const char* context = nullptr) {
    (void)context;
    return host_ptr;
}

template<typename T>
inline const T* dev_ptr(const T* host_ptr, const char* context = nullptr) {
    (void)context;
    return host_ptr;
}
#endif

//=============================================================================
// Debug Sync Counter (for verifying no H↔D transfers during stepping)
//=============================================================================

/// Global counter for H↔D sync operations during stepping
/// Tests can use this to verify "no mid-step transfers" guarantee
///
/// Usage in solver (debug builds):
///   void sync_from_gpu() {
///       #ifndef NDEBUG
///       gpu::increment_sync_counter();
///       #endif
///       // ... actual sync code ...
///   }
///
/// Usage in tests:
///   gpu::reset_sync_counter();
///   for (int i = 0; i < nsteps; ++i) solver.step();
///   ASSERT(gpu::get_sync_counter() == 0);  // No syncs during stepping
///   gpu::ensure_synced(solver);             // Allowed: one sync for output
///   ASSERT(gpu::get_sync_counter() == 1);

inline int& sync_counter_ref() {
    static int counter = 0;
    return counter;
}

inline void reset_sync_counter() { sync_counter_ref() = 0; }
inline int get_sync_counter() { return sync_counter_ref(); }
inline void increment_sync_counter() { ++sync_counter_ref(); }

/// Assert no syncs occurred (for enforcing no-H↔D-during-stepping)
/// Returns true if counter is zero, false otherwise (with error message)
inline bool assert_no_syncs(const char* context = nullptr) {
    if (sync_counter_ref() != 0) {
        std::fprintf(stderr, "[GPU Sync Guard] FAILURE: Expected 0 sync calls");
        if (context) std::fprintf(stderr, " during %s", context);
        std::fprintf(stderr, ", but got %d\n", sync_counter_ref());
        std::fprintf(stderr, "[GPU Sync Guard] This violates the 'no H↔D during stepping' guarantee!\n");
        return false;
    }
    return true;
}

} // namespace gpu
} // namespace nncfd

// Convenience macros for GPU kernels
#ifdef USE_GPU_OFFLOAD

// Parallel for on GPU with automatic data mapping
#define GPU_PARALLEL_FOR(var, start, end) \
    _Pragma("omp target teams distribute parallel for") \
    for (int var = start; var < end; ++var)

// Parallel for with device pointers (data already on GPU)
#define GPU_PARALLEL_FOR_DEVICE(var, start, end) \
    _Pragma("omp target teams distribute parallel for") \
    for (int var = start; var < end; ++var)

// Async parallel for - kernel launches asynchronously, no host sync
// Use gpu::sync() before reading results back to host
#define GPU_PARALLEL_FOR_ASYNC(var, start, end) \
    _Pragma("omp target teams distribute parallel for nowait") \
    for (int var = start; var < end; ++var)

// Collapsed 2D parallel for
#define GPU_PARALLEL_FOR_2D(i, i_start, i_end, j, j_start, j_end) \
    _Pragma("omp target teams distribute parallel for collapse(2)") \
    for (int j = j_start; j < j_end; ++j) \
    for (int i = i_start; i < i_end; ++i)

// SIMD reduction helper
#define GPU_SIMD_REDUCE(var, op) \
    _Pragma("omp simd reduction(" #op ":" #var ")")

#else

// CPU fallback versions
#define GPU_PARALLEL_FOR(var, start, end) \
    _Pragma("omp parallel for") \
    for (int var = start; var < end; ++var)

#define GPU_PARALLEL_FOR_DEVICE(var, start, end) \
    _Pragma("omp parallel for") \
    for (int var = start; var < end; ++var)

// CPU fallback for async - just a regular parallel for (no async semantics on CPU)
// Note: gpu::sync() is a no-op on CPU, so async/sync pattern still works
#define GPU_PARALLEL_FOR_ASYNC(var, start, end) \
    _Pragma("omp parallel for") \
    for (int var = start; var < end; ++var)

#define GPU_PARALLEL_FOR_2D(i, i_start, i_end, j, j_start, j_end) \
    _Pragma("omp parallel for collapse(2)") \
    for (int j = j_start; j < j_end; ++j) \
    for (int i = i_start; i < i_end; ++i)

#define GPU_SIMD_REDUCE(var, op) \
    _Pragma("omp simd reduction(" #op ":" #var ")")

#endif


