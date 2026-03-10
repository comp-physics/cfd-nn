#include "cuda_reductions.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cmath>

namespace nncfd {
namespace cuda_kernels {

static cublasHandle_t g_cublas_handle = nullptr;

void init_cublas() {
    if (!g_cublas_handle) {
        cublasStatus_t status = cublasCreate(&g_cublas_handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cuBLAS initialization failed");
        }
    }
}

void finalize_cublas() {
    if (g_cublas_handle) {
        cublasDestroy(g_cublas_handle);
        g_cublas_handle = nullptr;
    }
}

double device_norm_l2(const double* d_data, int n) {
    if (!g_cublas_handle) init_cublas();
    double result = 0.0;
    cublasStatus_t status = cublasDnrm2(g_cublas_handle, n, d_data, 1, &result);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasDnrm2 failed");
    }
    return result;
}

double device_norm_linf(const double* d_data, int n) {
    if (!g_cublas_handle) init_cublas();
    int idx = 0;
    cublasStatus_t status = cublasIdamax(g_cublas_handle, n, d_data, 1, &idx);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasIdamax failed");
    }
    // cuBLAS returns 1-based index; read value from device
    double result = 0.0;
    cudaMemcpy(&result, d_data + idx - 1, sizeof(double), cudaMemcpyDeviceToHost);
    return std::abs(result);
}

double device_sum_abs(const double* d_data, int n) {
    if (!g_cublas_handle) init_cublas();
    double result = 0.0;
    cublasStatus_t status = cublasDasum(g_cublas_handle, n, d_data, 1, &result);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasDasum failed");
    }
    return result;
}

} // namespace cuda_kernels
} // namespace nncfd
