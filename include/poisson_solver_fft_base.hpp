/// @file poisson_solver_fft_base.hpp
/// @brief Common utilities for FFT-based Poisson solvers
///
/// Provides shared functionality used by FFTPoissonSolver, FFT1DPoissonSolver,
/// and FFT2DPoissonSolver:
/// - cuFFT error checking macros
/// - CUDA error checking macros
/// - Common memory management patterns
/// - Tridiagonal solver interface

#pragma once

#include <stdexcept>
#include <string>

#ifdef USE_GPU_OFFLOAD
#include <cuda_runtime.h>
#include <cufft.h>
#endif

namespace nncfd {
namespace fft_poisson {

//=============================================================================
// Error Checking Macros
//=============================================================================

#ifdef USE_GPU_OFFLOAD

/// Check CUDA runtime API call and throw on error
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + \
                                     std::to_string(__LINE__) + ": " + \
                                     cudaGetErrorString(err)); \
        } \
    } while (0)

/// Check cuFFT API call and throw on error
#define CUFFT_CHECK(call) \
    do { \
        cufftResult err = (call); \
        if (err != CUFFT_SUCCESS) { \
            throw std::runtime_error(std::string("cuFFT error at ") + __FILE__ + ":" + \
                                     std::to_string(__LINE__) + ": code " + \
                                     std::to_string(static_cast<int>(err))); \
        } \
    } while (0)

#else
// No-op when GPU offload is disabled
#define CUDA_CHECK(call) (void)(call)
#define CUFFT_CHECK(call) (void)(call)
#endif

//=============================================================================
// Memory Management RAII Wrappers
//=============================================================================

#ifdef USE_GPU_OFFLOAD

/// RAII wrapper for CUDA device memory
template<typename T>
class DeviceBuffer {
public:
    DeviceBuffer() : ptr_(nullptr), size_(0) {}

    explicit DeviceBuffer(size_t count) : ptr_(nullptr), size_(count) {
        if (count > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
        }
    }

    ~DeviceBuffer() {
        if (ptr_) cudaFree(ptr_);
    }

    // Move-only
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // No copy
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t size() const { return size_; }

    void resize(size_t count) {
        if (count != size_) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = nullptr;
            size_ = count;
            if (count > 0) {
                CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
            }
        }
    }

private:
    T* ptr_;
    size_t size_;
};

/// RAII wrapper for cuFFT plan
class CufftPlan {
public:
    CufftPlan() : plan_(0), valid_(false) {}

    ~CufftPlan() {
        destroy();
    }

    // Move-only
    CufftPlan(CufftPlan&& other) noexcept
        : plan_(other.plan_), valid_(other.valid_) {
        other.valid_ = false;
    }

    CufftPlan& operator=(CufftPlan&& other) noexcept {
        if (this != &other) {
            destroy();
            plan_ = other.plan_;
            valid_ = other.valid_;
            other.valid_ = false;
        }
        return *this;
    }

    // No copy
    CufftPlan(const CufftPlan&) = delete;
    CufftPlan& operator=(const CufftPlan&) = delete;

    cufftHandle& handle() { return plan_; }
    bool valid() const { return valid_; }

    void set_valid() { valid_ = true; }

    void destroy() {
        if (valid_) {
            cufftDestroy(plan_);
            valid_ = false;
        }
    }

private:
    cufftHandle plan_;
    bool valid_;
};

#endif // USE_GPU_OFFLOAD

//=============================================================================
// Eigenvalue Computation (for spectral methods)
//=============================================================================

/// Compute modified wavenumber for second derivative (2*cos(k*dx) - 2) / dx^2
/// This is the eigenvalue of the discrete Laplacian in the Fourier basis.
inline double modified_eigenvalue(int mode, int N, double h) {
    double k = M_PI * mode / N;
    return (2.0 * std::cos(k) - 2.0) / (h * h);
}

/// Compute FFT eigenvalue for 1D second derivative
/// lambda_i = -4/dx^2 * sin^2(pi*i/N)
inline double fft_eigenvalue(int i, int N, double dx) {
    double s = std::sin(M_PI * i / N);
    return -4.0 / (dx * dx) * s * s;
}

//=============================================================================
// Tridiagonal Solver (Thomas algorithm)
//=============================================================================

/// Solve tridiagonal system Ax = d using Thomas algorithm (in-place)
/// @param a  Sub-diagonal (indexed 1 to n-1, a[0] unused)
/// @param b  Main diagonal (indexed 0 to n-1)
/// @param c  Super-diagonal (indexed 0 to n-2, c[n-1] unused)
/// @param d  RHS on input, solution on output (indexed 0 to n-1)
/// @param n  System size
inline void solve_tridiagonal(double* a, double* b, double* c, double* d, int n) {
    // Forward sweep
    for (int i = 1; i < n; ++i) {
        double w = a[i] / b[i-1];
        b[i] -= w * c[i-1];
        d[i] -= w * d[i-1];
    }

    // Back substitution
    d[n-1] /= b[n-1];
    for (int i = n-2; i >= 0; --i) {
        d[i] = (d[i] - c[i] * d[i+1]) / b[i];
    }
}

/// Solve cyclic tridiagonal system (periodic BCs) using Sherman-Morrison
/// @param a  Sub-diagonal
/// @param b  Main diagonal
/// @param c  Super-diagonal
/// @param d  RHS on input, solution on output
/// @param n  System size
/// @param alpha, beta  Corner entries for cyclic system
void solve_cyclic_tridiagonal(double* a, double* b, double* c, double* d,
                               int n, double alpha, double beta);

} // namespace fft_poisson
} // namespace nncfd
