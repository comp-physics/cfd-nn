#pragma once

namespace nncfd {
namespace cuda_kernels {

/// L2 norm via cuBLAS (cublasDnrm2)
double device_norm_l2(const double* d_data, int n);

/// Max absolute value via cuBLAS (cublasIdamax)
double device_norm_linf(const double* d_data, int n);

/// Sum of absolute values via cuBLAS (cublasDasum)
double device_sum_abs(const double* d_data, int n);

/// Initialize/finalize cuBLAS handle (call once at solver init/cleanup)
void init_cublas();
void finalize_cublas();

} // namespace cuda_kernels
} // namespace nncfd
