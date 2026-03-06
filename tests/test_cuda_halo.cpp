/// @file test_cuda_halo.cpp
/// @brief Tests for fused BC kernel and z-face pack/unpack kernels
///
/// Test coverage:
///   1. Periodic BCs: ghost cells match opposite interior boundary
///   2. Neumann BCs: ghost cells mirror first/last interior cell
///   3. Mixed BCs: periodic in x/z, Neumann in y (channel flow config)
///   4. Pack/unpack round-trip: pack z-face → unpack → verify identity
///   5. Pack low vs high face: correct plane selected
///
/// All tests run on GPU via CUDA and verify against CPU reference values.

#ifdef USE_CUDA_KERNELS

#include "cuda_halo.hpp"
#include "cuda_smoother.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <cassert>

using namespace nncfd;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(1); \
    } \
} while(0)

/// Test: All-periodic BCs — ghost cells should wrap around
void test_periodic_bc() {
    const int Nx = 8, Ny = 8, Nz = 8, Ng = 1;
    const int stride = Nx + 2 * Ng;
    const int plane_stride = stride * (Ny + 2 * Ng);
    const int total = stride * (Ny + 2 * Ng) * (Nz + 2 * Ng);

    std::vector<double> field(total, 0.0);

    // Fill interior with unique values: value = i + j*100 + k*10000
    for (int k = Ng; k < Nz + Ng; ++k)
        for (int j = Ng; j < Ny + Ng; ++j)
            for (int i = Ng; i < Nx + Ng; ++i)
                field[k * plane_stride + j * stride + i] =
                    (i - Ng) + (j - Ng) * 100.0 + (k - Ng) * 10000.0;

    // Copy to GPU
    double* d_field;
    CUDA_CHECK(cudaMalloc(&d_field, total * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_field, field.data(), total * sizeof(double), cudaMemcpyHostToDevice));

    // Apply periodic BCs (0 = periodic)
    cuda_kernels::launch_apply_bc_3d_fused(
        d_field, Nx, Ny, Nz, Ng, 0, 0, 0, 0, 0, 0, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back
    CUDA_CHECK(cudaMemcpy(field.data(), d_field, total * sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(d_field);

    // Verify x-lo ghost (i=0) = x-hi interior (i=Nx+Ng-1 = last interior)
    double max_err = 0.0;
    for (int k = Ng; k < Nz + Ng; ++k) {
        for (int j = Ng; j < Ny + Ng; ++j) {
            double ghost = field[k * plane_stride + j * stride + 0];
            // i=0 ghost should wrap to interior i=Nx (i.e., last interior = i=Nx+Ng-1, value=(Nx-1))
            // Periodic: ghost[0] = interior[Nx+0] = value at (Nx-1 + ...)
            double expected = (Nx - 1) + (j - Ng) * 100.0 + (k - Ng) * 10000.0;
            // The kernel does: u[ghost] = u[Nx + g] where g=0, so u[Nx] which is interior[Nx-Ng]=interior[Nx-1]
            max_err = std::max(max_err, std::abs(ghost - expected));
        }
    }
    assert(max_err < 1e-14 && "Periodic x-lo ghost must match x-hi interior");

    // Verify z-hi ghost
    for (int j = Ng; j < Ny + Ng; ++j) {
        for (int i = Ng; i < Nx + Ng; ++i) {
            int k_ghost = Nz + Ng;  // first ghost above interior
            double ghost = field[k_ghost * plane_stride + j * stride + i];
            // Should wrap to first interior z-plane (k=Ng, value has k_global=0)
            double expected = (i - Ng) + (j - Ng) * 100.0 + 0 * 10000.0;
            max_err = std::max(max_err, std::abs(ghost - expected));
        }
    }
    assert(max_err < 1e-14 && "Periodic z-hi ghost must match z-lo interior");

    std::cout << "PASS: Periodic BC (max error = " << max_err << ")" << std::endl;
}

/// Test: All-Neumann BCs — ghost cells mirror first/last interior
void test_neumann_bc() {
    const int Nx = 8, Ny = 8, Nz = 8, Ng = 1;
    const int stride = Nx + 2 * Ng;
    const int plane_stride = stride * (Ny + 2 * Ng);
    const int total = stride * (Ny + 2 * Ng) * (Nz + 2 * Ng);

    std::vector<double> field(total, 0.0);

    for (int k = Ng; k < Nz + Ng; ++k)
        for (int j = Ng; j < Ny + Ng; ++j)
            for (int i = Ng; i < Nx + Ng; ++i)
                field[k * plane_stride + j * stride + i] =
                    (i - Ng) + (j - Ng) * 100.0 + (k - Ng) * 10000.0;

    double* d_field;
    CUDA_CHECK(cudaMalloc(&d_field, total * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_field, field.data(), total * sizeof(double), cudaMemcpyHostToDevice));

    // Apply Neumann BCs (1 = neumann)
    cuda_kernels::launch_apply_bc_3d_fused(
        d_field, Nx, Ny, Nz, Ng, 1, 1, 1, 1, 1, 1, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(field.data(), d_field, total * sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(d_field);

    // Neumann x-lo: ghost[0] = interior[Ng] (first interior cell)
    double max_err = 0.0;
    for (int k = Ng; k < Nz + Ng; ++k) {
        for (int j = Ng; j < Ny + Ng; ++j) {
            double ghost = field[k * plane_stride + j * stride + 0];
            double interior = field[k * plane_stride + j * stride + Ng];
            max_err = std::max(max_err, std::abs(ghost - interior));
        }
    }

    // Neumann y-hi: ghost[Ny+Ng] = interior[Ny+Ng-1]
    for (int k = Ng; k < Nz + Ng; ++k) {
        for (int i = Ng; i < Nx + Ng; ++i) {
            double ghost = field[k * plane_stride + (Ny + Ng) * stride + i];
            double interior = field[k * plane_stride + (Ny + Ng - 1) * stride + i];
            max_err = std::max(max_err, std::abs(ghost - interior));
        }
    }
    assert(max_err < 1e-14 && "Neumann BC ghost must equal adjacent interior");

    std::cout << "PASS: Neumann BC (max error = " << max_err << ")" << std::endl;
}

/// Test: Mixed BCs (periodic x/z, Neumann y) — typical channel flow configuration
void test_mixed_bc() {
    const int Nx = 8, Ny = 8, Nz = 8, Ng = 1;
    const int stride = Nx + 2 * Ng;
    const int plane_stride = stride * (Ny + 2 * Ng);
    const int total = stride * (Ny + 2 * Ng) * (Nz + 2 * Ng);

    std::vector<double> field(total, 0.0);

    for (int k = Ng; k < Nz + Ng; ++k)
        for (int j = Ng; j < Ny + Ng; ++j)
            for (int i = Ng; i < Nx + Ng; ++i)
                field[k * plane_stride + j * stride + i] =
                    sin(2.0 * M_PI * (i - Ng) / Nx) *
                    cos(M_PI * (j - Ng) / Ny) *
                    sin(2.0 * M_PI * (k - Ng) / Nz);

    double* d_field;
    CUDA_CHECK(cudaMalloc(&d_field, total * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_field, field.data(), total * sizeof(double), cudaMemcpyHostToDevice));

    // periodic x/z (0), neumann y (1)
    cuda_kernels::launch_apply_bc_3d_fused(
        d_field, Nx, Ny, Nz, Ng, 0, 0, 1, 1, 0, 0, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(field.data(), d_field, total * sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(d_field);

    // Check x is periodic: ghost[0] = interior[Nx]
    double max_err = 0.0;
    for (int k = Ng; k < Nz + Ng; ++k) {
        for (int j = Ng; j < Ny + Ng; ++j) {
            double ghost_xlo = field[k * plane_stride + j * stride + 0];
            // Periodic wrap: interior at i=Nx (i.e. index Nx+0 in the array, which is Nx_interior-1)
            double wrap_src = field[k * plane_stride + j * stride + Nx];
            max_err = std::max(max_err, std::abs(ghost_xlo - wrap_src));
        }
    }

    // Check y is Neumann: ghost[0] = interior[Ng]
    for (int k = Ng; k < Nz + Ng; ++k) {
        for (int i = Ng; i < Nx + Ng; ++i) {
            double ghost = field[k * plane_stride + 0 * stride + i];
            double interior = field[k * plane_stride + Ng * stride + i];
            max_err = std::max(max_err, std::abs(ghost - interior));
        }
    }
    assert(max_err < 1e-14 && "Mixed BC must be correct");

    std::cout << "PASS: Mixed BC periodic-x/z + Neumann-y (max error = " << max_err << ")" << std::endl;
}

/// Test: Pack/unpack round-trip for z-face
void test_pack_unpack_roundtrip() {
    const int Nx = 8, Ny = 8, Nz = 8, Ng = 1;
    const int stride = Nx + 2 * Ng;
    const int plane_stride = stride * (Ny + 2 * Ng);
    const int total = stride * (Ny + 2 * Ng) * (Nz + 2 * Ng);
    const int face_size = (Nx + 2*Ng) * (Ny + 2*Ng);

    std::vector<double> field(total, 0.0);

    // Fill interior with unique values
    for (int k = Ng; k < Nz + Ng; ++k)
        for (int j = 0; j < Ny + 2*Ng; ++j)
            for (int i = 0; i < Nx + 2*Ng; ++i)
                field[k * plane_stride + j * stride + i] =
                    i + j * 100.0 + k * 10000.0;

    double *d_field, *d_buffer, *d_field2;
    CUDA_CHECK(cudaMalloc(&d_field, total * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_buffer, face_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_field2, total * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_field, field.data(), total * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_field2, 0, total * sizeof(double)));

    // Pack low-z face (first interior plane k=Ng)
    cuda_kernels::launch_pack_z_face(d_field, d_buffer, Nx, Ny, Nz, Ng, true, nullptr);
    // Unpack into low-z ghost of field2 (k=0)
    cuda_kernels::launch_unpack_z_face(d_field2, d_buffer, Nx, Ny, Nz, Ng, true, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify: field2's ghost (k=0) should match field's first interior (k=Ng)
    std::vector<double> result(total, 0.0);
    CUDA_CHECK(cudaMemcpy(result.data(), d_field2, total * sizeof(double), cudaMemcpyDeviceToHost));

    double max_err = 0.0;
    for (int j = 0; j < Ny + 2*Ng; ++j) {
        for (int i = 0; i < Nx + 2*Ng; ++i) {
            double packed = result[0 * plane_stride + j * stride + i];
            double original = field[Ng * plane_stride + j * stride + i];
            max_err = std::max(max_err, std::abs(packed - original));
        }
    }
    assert(max_err < 1e-14 && "Pack/unpack low-z roundtrip must be exact");

    // Also test high-z face
    CUDA_CHECK(cudaMemset(d_field2, 0, total * sizeof(double)));
    cuda_kernels::launch_pack_z_face(d_field, d_buffer, Nx, Ny, Nz, Ng, false, nullptr);
    cuda_kernels::launch_unpack_z_face(d_field2, d_buffer, Nx, Ny, Nz, Ng, false, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(result.data(), d_field2, total * sizeof(double), cudaMemcpyDeviceToHost));
    for (int j = 0; j < Ny + 2*Ng; ++j) {
        for (int i = 0; i < Nx + 2*Ng; ++i) {
            int k_ghost = Nz + 2*Ng - 1;
            int k_interior = Nz + Ng - 1;
            double packed = result[k_ghost * plane_stride + j * stride + i];
            double original = field[k_interior * plane_stride + j * stride + i];
            max_err = std::max(max_err, std::abs(packed - original));
        }
    }
    assert(max_err < 1e-14 && "Pack/unpack high-z roundtrip must be exact");

    std::cout << "PASS: Pack/unpack z-face roundtrip (max error = " << max_err << ")" << std::endl;

    cudaFree(d_field);
    cudaFree(d_buffer);
    cudaFree(d_field2);
}

int main() {
    if (!cuda_kernels::cuda_smoother_available()) {
        std::cout << "SKIP: No CUDA device available" << std::endl;
        return 0;
    }

    test_periodic_bc();
    test_neumann_bc();
    test_mixed_bc();
    test_pack_unpack_roundtrip();

    std::cout << "\nAll CUDA halo/BC tests PASSED" << std::endl;
    return 0;
}

#else // !USE_CUDA_KERNELS

#include <iostream>
int main() {
    std::cout << "SKIP: CUDA kernels not enabled" << std::endl;
    return 0;
}

#endif
