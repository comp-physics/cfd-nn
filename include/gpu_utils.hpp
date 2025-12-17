#pragma once

/// GPU utilities for OpenMP target offloading
/// Provides RAII-style memory management and helper macros

#include <vector>
#include <cstddef>

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
#endif

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

#define GPU_PARALLEL_FOR_2D(i, i_start, i_end, j, j_start, j_end) \
    _Pragma("omp parallel for collapse(2)") \
    for (int j = j_start; j < j_end; ++j) \
    for (int i = i_start; i < i_end; ++i)

#define GPU_SIMD_REDUCE(var, op) \
    _Pragma("omp simd reduction(" #op ":" #var ")")

#endif


