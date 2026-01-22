/// @file solver_time_kernels_copy.cpp
/// @brief Copy kernels for RK time integration

#include "solver_time_kernels.hpp"
#include "gpu_utils.hpp"

namespace nncfd {
namespace time_kernels {

void copy_2d_uv(double* u_src, double* u_dst, double* v_src, double* v_dst,
                int Nx, int Ny, int Ng, int u_stride, int v_stride,
                size_t u_total, size_t v_total) {
    (void)u_total; (void)v_total;  // Reserved for future use (e.g., target map clauses)
    // NVHPC WORKAROUND: Use use_device_ptr to convert host pointers to device pointers
    #pragma omp target data use_device_ptr(u_src, u_dst, v_src, v_dst)
    {
        // Copy u-velocity
        #pragma omp target teams distribute parallel for collapse(2)
        for (int j = Ng; j < Ng + Ny; ++j) {
            for (int i = Ng; i <= Ng + Nx; ++i) {
                int idx = j * u_stride + i;
                u_dst[idx] = u_src[idx];
            }
        }

        // Copy v-velocity
        #pragma omp target teams distribute parallel for collapse(2)
        for (int j = Ng; j <= Ng + Ny; ++j) {
            for (int i = Ng; i < Ng + Nx; ++i) {
                int idx = j * v_stride + i;
                v_dst[idx] = v_src[idx];
            }
        }
    }
}

void copy_3d_uvw(double* u_src, double* u_dst, double* v_src, double* v_dst,
                 double* w_src, double* w_dst,
                 int Nx, int Ny, int Nz, int Ng,
                 int u_stride, int v_stride, int w_stride,
                 int u_plane, int v_plane, int w_plane,
                 size_t u_total, size_t v_total, size_t w_total) {
    (void)u_total; (void)v_total; (void)w_total;  // Reserved for future use
    // NVHPC WORKAROUND: Use use_device_ptr to convert host pointers to device pointers
    #pragma omp target data use_device_ptr(u_src, u_dst, v_src, v_dst, w_src, w_dst)
    {
        // Copy u-velocity
        #pragma omp target teams distribute parallel for collapse(3)
        for (int k = Ng; k < Ng + Nz; ++k) {
            for (int j = Ng; j < Ng + Ny; ++j) {
                for (int i = Ng; i <= Ng + Nx; ++i) {
                    int idx = k * u_plane + j * u_stride + i;
                    u_dst[idx] = u_src[idx];
                }
            }
        }

        // Copy v-velocity
        #pragma omp target teams distribute parallel for collapse(3)
        for (int k = Ng; k < Ng + Nz; ++k) {
            for (int j = Ng; j <= Ng + Ny; ++j) {
                for (int i = Ng; i < Ng + Nx; ++i) {
                    int idx = k * v_plane + j * v_stride + i;
                    v_dst[idx] = v_src[idx];
                }
            }
        }

        // Copy w-velocity
        #pragma omp target teams distribute parallel for collapse(3)
        for (int k = Ng; k <= Ng + Nz; ++k) {
            for (int j = Ng; j < Ng + Ny; ++j) {
                for (int i = Ng; i < Ng + Nx; ++i) {
                    int idx = k * w_plane + j * w_stride + i;
                    w_dst[idx] = w_src[idx];
                }
            }
        }
    }
}

} // namespace time_kernels
} // namespace nncfd
