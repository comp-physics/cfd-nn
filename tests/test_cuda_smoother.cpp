/// @file test_cuda_smoother.cpp
/// @brief Test CUDA shared-memory Chebyshev smoother against existing kernel

#ifdef USE_CUDA_KERNELS

#include "cuda_smoother.hpp"
#include "cuda_reductions.hpp"
#include "mg_cuda_kernels.hpp"
#include <cmath>
#include <vector>
#include <iostream>
#include <cassert>
#include <cuda_runtime.h>

using namespace nncfd;

// Helper: CUDA error check
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return 1; \
    } \
} while(0)

void test_smem_matches_global() {
    // Small grid for fast test
    const int Nx = 32, Ny = 32, Nz = 32, Ng = 1;
    const double dx = 2.0 * M_PI / Nx;
    const double dy = 2.0 / Ny;
    const double dz = M_PI / Nz;
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double inv_dz2 = 1.0 / (dz * dz);

    const int stride = Nx + 2 * Ng;
    const int plane_stride = stride * (Ny + 2 * Ng);
    const int total = stride * (Ny + 2 * Ng) * (Nz + 2 * Ng);

    // Fill with sinusoidal RHS and zero initial guess
    std::vector<double> u_init(total, 0.0);
    std::vector<double> f_data(total, 0.0);

    for (int k = Ng; k < Nz + Ng; ++k) {
        for (int j = Ng; j < Ny + Ng; ++j) {
            for (int i = Ng; i < Nx + Ng; ++i) {
                int idx = k * plane_stride + j * stride + i;
                double x = (i - Ng + 0.5) * dx;
                double y = -1.0 + (j - Ng + 0.5) * dy;
                double z = (k - Ng + 0.5) * dz;
                f_data[idx] = sin(x) * cos(M_PI * y) * sin(z);
            }
        }
    }

    // Allocate device arrays
    double *d_u_global, *d_u_smem, *d_f, *d_tmp;
    CUDA_CHECK(cudaMalloc(&d_u_global, total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_u_smem, total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_f, total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_tmp, total * sizeof(double)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_u_global, u_init.data(), total * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_smem, u_init.data(), total * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_f, f_data.data(), total * sizeof(double), cudaMemcpyHostToDevice));

    // Run existing global-memory kernel (4 Chebyshev iterations)
    const int degree = 4;
    const double lambda_min = 0.05;
    const double lambda_max = 1.95;
    const double diag = -(2.0 * inv_dx2 + 2.0 * inv_dy2 + 2.0 * inv_dz2);

    for (int iter = 0; iter < degree; ++iter) {
        double theta = M_PI * (2.0 * iter + 1.0) / (2.0 * degree);
        double sigma = (lambda_max + lambda_min) / 2.0
                     + (lambda_max - lambda_min) / 2.0 * cos(theta);
        double omega = 1.0 / sigma;

        mg_cuda::launch_chebyshev_3d(
            0, d_u_global, d_f, d_tmp,
            Nx, Ny, Nz, Ng,
            1.0/inv_dx2, 1.0/inv_dy2, 1.0/inv_dz2, diag, omega);

        mg_cuda::launch_copy(0, d_u_global, d_tmp, total);
    }

    // Run shared-memory kernel (same 4 iterations)
    cuda_kernels::launch_chebyshev_3d_smem(
        d_u_smem, d_f, d_tmp,
        Nx, Ny, Nz, Ng,
        inv_dx2, inv_dy2, inv_dz2,
        degree, lambda_min, lambda_max,
        false, false, false,
        nullptr);

    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back
    std::vector<double> u_global(total), u_smem(total);
    CUDA_CHECK(cudaMemcpy(u_global.data(), d_u_global, total * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(u_smem.data(), d_u_smem, total * sizeof(double), cudaMemcpyDeviceToHost));

    // Compare: should match to round-off (~1e-12)
    double max_diff = 0.0;
    for (int k = Ng; k < Nz + Ng; ++k) {
        for (int j = Ng; j < Ny + Ng; ++j) {
            for (int i = Ng; i < Nx + Ng; ++i) {
                int idx = k * plane_stride + j * stride + i;
                double diff = std::abs(u_global[idx] - u_smem[idx]);
                max_diff = std::max(max_diff, diff);
            }
        }
    }

    std::cout << "Max difference (global vs smem smoother): " << max_diff << std::endl;
    assert(max_diff < 1e-12 && "CUDA smem smoother must match global-memory kernel");
    std::cout << "PASS: CUDA shared-memory smoother matches global-memory kernel" << std::endl;

    cudaFree(d_u_global);
    cudaFree(d_u_smem);
    cudaFree(d_f);
    cudaFree(d_tmp);
}

void test_smem_reduces_residual() {
    // Verify smoother actually reduces residual (not just stability)
    const int Nx = 16, Ny = 16, Nz = 16, Ng = 1;
    const double dx = 1.0 / Nx, dy = 1.0 / Ny, dz = 1.0 / Nz;
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double inv_dz2 = 1.0 / (dz * dz);

    const int stride = Nx + 2 * Ng;
    const int plane_stride = stride * (Ny + 2 * Ng);
    const int total = stride * (Ny + 2 * Ng) * (Nz + 2 * Ng);

    std::vector<double> u(total, 0.0);
    std::vector<double> f(total, 0.0);

    // Set RHS = 1 (constant)
    for (int k = Ng; k < Nz + Ng; ++k)
        for (int j = Ng; j < Ny + Ng; ++j)
            for (int i = Ng; i < Nx + Ng; ++i)
                f[k * plane_stride + j * stride + i] = 1.0;

    double *d_u, *d_f, *d_tmp;
    CUDA_CHECK(cudaMalloc(&d_u, total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_f, total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_tmp, total * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_u, u.data(), total * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_f, f.data(), total * sizeof(double), cudaMemcpyHostToDevice));

    // Run 8 Chebyshev iterations
    cuda_kernels::launch_chebyshev_3d_smem(
        d_u, d_f, d_tmp, Nx, Ny, Nz, Ng,
        inv_dx2, inv_dy2, inv_dz2,
        8, 0.05, 1.95,
        false, false, false, nullptr);

    CUDA_CHECK(cudaDeviceSynchronize());

    // Compute residual on CPU
    CUDA_CHECK(cudaMemcpy(u.data(), d_u, total * sizeof(double), cudaMemcpyDeviceToHost));

    double max_residual = 0.0;
    for (int k = Ng; k < Nz + Ng; ++k) {
        for (int j = Ng; j < Ny + Ng; ++j) {
            for (int i = Ng; i < Nx + Ng; ++i) {
                int idx = k * plane_stride + j * stride + i;
                double lap = (u[idx+1] - 2*u[idx] + u[idx-1]) * inv_dx2
                           + (u[idx+stride] - 2*u[idx] + u[idx-stride]) * inv_dy2
                           + (u[idx+plane_stride] - 2*u[idx] + u[idx-plane_stride]) * inv_dz2;
                double res = std::abs(f[idx] - lap);
                max_residual = std::max(max_residual, res);
            }
        }
    }

    std::cout << "Residual after 8 Chebyshev iterations: " << max_residual << std::endl;
    // Smoother should significantly reduce residual from initial ||f|| ~ 1.0
    assert(max_residual < 0.5 && "Smoother must reduce residual");
    std::cout << "PASS: Smoother reduces residual" << std::endl;

    cudaFree(d_u);
    cudaFree(d_f);
    cudaFree(d_tmp);
}

int main() {
    if (!cuda_kernels::cuda_smoother_available()) {
        std::cout << "SKIP: No CUDA device available" << std::endl;
        return 0;
    }

    test_smem_matches_global();
    test_smem_reduces_residual();

    // cuBLAS reduction tests
    {
        const int N = 100000;
        std::vector<double> h_data(N);
        for (int i = 0; i < N; ++i) h_data[i] = sin(0.01 * i);

        double expected_l2 = 0.0;
        for (int i = 0; i < N; ++i) expected_l2 += h_data[i] * h_data[i];
        expected_l2 = sqrt(expected_l2);

        double* d_data;
        CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), N * sizeof(double), cudaMemcpyHostToDevice));

        double result_l2 = cuda_kernels::device_norm_l2(d_data, N);
        double rel_err = std::abs(result_l2 - expected_l2) / expected_l2;
        std::cout << "cuBLAS L2 norm rel error: " << rel_err << std::endl;
        assert(rel_err < 1e-14 && "cuBLAS L2 norm must match CPU reference");
        std::cout << "PASS: cuBLAS L2 norm" << std::endl;

        double result_linf = cuda_kernels::device_norm_linf(d_data, N);
        double expected_linf = 0.0;
        for (int i = 0; i < N; ++i) expected_linf = std::max(expected_linf, std::abs(h_data[i]));
        double linf_err = std::abs(result_linf - expected_linf) / expected_linf;
        std::cout << "cuBLAS Linf norm rel error: " << linf_err << std::endl;
        assert(linf_err < 1e-14 && "cuBLAS Linf norm must match CPU reference");
        std::cout << "PASS: cuBLAS Linf norm" << std::endl;

        cudaFree(d_data);
        cuda_kernels::finalize_cublas();
    }

    std::cout << "\nAll CUDA smoother + reduction tests PASSED" << std::endl;
    return 0;
}

#else // !USE_CUDA_KERNELS

#include <iostream>
int main() {
    std::cout << "SKIP: CUDA kernels not enabled" << std::endl;
    return 0;
}

#endif
