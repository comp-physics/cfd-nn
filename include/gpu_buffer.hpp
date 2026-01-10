/// @file gpu_buffer.hpp
/// @brief RAII wrapper for GPU buffer management with OpenMP target offloading
///
/// Provides:
/// - GPUBuffer<T>: RAII-managed GPU buffer with automatic cleanup
/// - Automatic mapping/unmapping to GPU device
/// - Safe copy semantics (move-only)
/// - Works in both GPU and CPU builds (CPU builds just use std::vector)
///
/// This reduces duplication across turbulence models where identical GPU buffer
/// management code is repeated 6+ times (~300+ LOC saved).
///
/// Example usage:
///   GPUBuffer<double> features(n_cells * feature_dim);
///   features.map_to_device();
///   // ... use features.data() in target regions ...
///   features.update_from_device();
///   // Destructor automatically unmaps

#pragma once

#include <vector>
#include <cstddef>
#include <functional>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

namespace nncfd {

/// RAII wrapper for GPU-managed buffers
/// Automatically handles mapping/unmapping to GPU device via OpenMP target
template<typename T>
class GPUBuffer {
public:
    /// Default constructor - empty buffer
    GPUBuffer() = default;

    /// Construct with initial size (data is zero-initialized)
    explicit GPUBuffer(size_t size) : data_(size) {}

    /// Destructor - automatically unmaps from GPU
    ~GPUBuffer() {
        unmap();
    }

    // Move-only semantics (GPU buffers cannot be safely copied)
    GPUBuffer(const GPUBuffer&) = delete;
    GPUBuffer& operator=(const GPUBuffer&) = delete;

    GPUBuffer(GPUBuffer&& other) noexcept
        : data_(std::move(other.data_)),
          mapped_(other.mapped_) {
        other.mapped_ = false;  // Prevent double-unmap
    }

    GPUBuffer& operator=(GPUBuffer&& other) noexcept {
        if (this != &other) {
            unmap();  // Unmap current buffer
            data_ = std::move(other.data_);
            mapped_ = other.mapped_;
            other.mapped_ = false;
        }
        return *this;
    }

    /// Resize the buffer (unmaps first if mapped)
    void resize(size_t new_size) {
        if (new_size != data_.size()) {
            unmap();
            data_.resize(new_size);
        }
    }

    /// Resize and initialize with a value
    void resize(size_t new_size, T value) {
        if (new_size != data_.size()) {
            unmap();
            data_.resize(new_size, value);
        }
    }

    /// Clear the buffer (unmaps first)
    void clear() {
        unmap();
        data_.clear();
    }

    /// Reserve capacity without changing size
    void reserve(size_t capacity) {
        // Note: reserve may invalidate pointers, so unmap first if mapped
        if (capacity > data_.capacity() && mapped_) {
            unmap();
        }
        data_.reserve(capacity);
    }

    /// Map buffer to GPU device (allocate on device, no data copy)
    void map_to_device() {
#ifdef USE_GPU_OFFLOAD
        if (!mapped_ && !data_.empty()) {
            T* ptr = data_.data();
            size_t sz = data_.size();
            #pragma omp target enter data map(alloc: ptr[0:sz])
            mapped_ = true;
        }
#endif
    }

    /// Map buffer to GPU device and copy data to device
    void map_to_device_with_data() {
#ifdef USE_GPU_OFFLOAD
        if (!mapped_ && !data_.empty()) {
            T* ptr = data_.data();
            size_t sz = data_.size();
            #pragma omp target enter data map(to: ptr[0:sz])
            mapped_ = true;
        }
#endif
    }

    /// Copy data from host to device (buffer must be mapped)
    void update_to_device() {
#ifdef USE_GPU_OFFLOAD
        if (mapped_ && !data_.empty()) {
            T* ptr = data_.data();
            size_t sz = data_.size();
            #pragma omp target update to(ptr[0:sz])
        }
#endif
    }

    /// Copy data from device to host (buffer must be mapped)
    void update_from_device() {
#ifdef USE_GPU_OFFLOAD
        if (mapped_ && !data_.empty()) {
            T* ptr = data_.data();
            size_t sz = data_.size();
            #pragma omp target update from(ptr[0:sz])
        }
#endif
    }

    /// Unmap buffer from GPU device
    void unmap() {
#ifdef USE_GPU_OFFLOAD
        if (mapped_ && !data_.empty()) {
            T* ptr = data_.data();
            size_t sz = data_.size();
            mapped_ = false;  // Set flag first to prevent re-entry
            #pragma omp target exit data map(delete: ptr[0:sz])
        }
#endif
        mapped_ = false;
    }

    /// Check if buffer is currently mapped to GPU
    bool is_mapped() const { return mapped_; }

    /// Check if buffer is empty
    bool empty() const { return data_.empty(); }

    /// Get buffer size
    size_t size() const { return data_.size(); }

    /// Get raw pointer to data
    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }

    /// Array access (host only - not safe in target regions)
    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }

    /// Get reference to underlying vector (for compatibility)
    std::vector<T>& vector() { return data_; }
    const std::vector<T>& vector() const { return data_; }

    /// Iterator support
    typename std::vector<T>::iterator begin() { return data_.begin(); }
    typename std::vector<T>::iterator end() { return data_.end(); }
    typename std::vector<T>::const_iterator begin() const { return data_.begin(); }
    typename std::vector<T>::const_iterator end() const { return data_.end(); }

private:
    std::vector<T> data_;
    bool mapped_ = false;
};

/// Helper to manage multiple GPU buffers as a group
/// All buffers are mapped/unmapped together
class GPUBufferGroup {
public:
    /// Add a buffer to the group (buffer must outlive the group)
    template<typename T>
    void add(GPUBuffer<T>& buffer) {
        map_funcs_.push_back([&buffer]() { buffer.map_to_device(); });
        unmap_funcs_.push_back([&buffer]() { buffer.unmap(); });
    }

    /// Map all buffers to device
    void map_all() {
        for (auto& f : map_funcs_) f();
    }

    /// Unmap all buffers from device
    void unmap_all() {
        for (auto& f : unmap_funcs_) f();
    }

    /// Check if group is empty
    bool empty() const { return map_funcs_.empty(); }

    /// Clear the group (does not unmap buffers)
    void clear() {
        map_funcs_.clear();
        unmap_funcs_.clear();
    }

private:
    std::vector<std::function<void()>> map_funcs_;
    std::vector<std::function<void()>> unmap_funcs_;
};

} // namespace nncfd
