/// @file solver_time_kernels_blend.cpp
/// @brief Blend kernels for RK time integration

#include "solver_time_kernels.hpp"
#include "gpu_utils.hpp"

namespace nncfd {
namespace time_kernels {

void blend_2d_uv(double* u_a, double* u_b, double* v_a, double* v_b,
                 double a1, double a2,
                 int Nx, int Ny, int Ng, int u_stride, int v_stride,
                 size_t u_total, size_t v_total) {
    (void)u_total; (void)v_total;  // Reserved for future use
    // NVHPC WORKAROUND: Use use_device_ptr to convert host pointers to device pointers
    #pragma omp target data use_device_ptr(u_a, u_b, v_a, v_b)
    {
        // Blend u-velocity: u_b = a1 * u_a + a2 * u_b
        #pragma omp target teams distribute parallel for collapse(2)
        for (int j = Ng; j < Ng + Ny; ++j) {
            for (int i = Ng; i <= Ng + Nx; ++i) {
                int idx = j * u_stride + i;
                u_b[idx] = a1 * u_a[idx] + a2 * u_b[idx];
            }
        }

        // Blend v-velocity: v_b = a1 * v_a + a2 * v_b
        #pragma omp target teams distribute parallel for collapse(2)
        for (int j = Ng; j <= Ng + Ny; ++j) {
            for (int i = Ng; i < Ng + Nx; ++i) {
                int idx = j * v_stride + i;
                v_b[idx] = a1 * v_a[idx] + a2 * v_b[idx];
            }
        }
    }
}

void blend_3d_uvw(double* u_a, double* u_b, double* v_a, double* v_b,
                  double* w_a, double* w_b, double a1, double a2,
                  int Nx, int Ny, int Nz, int Ng,
                  int u_stride, int v_stride, int w_stride,
                  int u_plane, int v_plane, int w_plane,
                  size_t u_total, size_t v_total, size_t w_total) {
    (void)u_total; (void)v_total; (void)w_total;  // Reserved for future use
    // NVHPC WORKAROUND: Use use_device_ptr to convert host pointers to device pointers
    #pragma omp target data use_device_ptr(u_a, u_b, v_a, v_b, w_a, w_b)
    {
        // Blend u-velocity
        #pragma omp target teams distribute parallel for collapse(3)
        for (int k = Ng; k < Ng + Nz; ++k) {
            for (int j = Ng; j < Ng + Ny; ++j) {
                for (int i = Ng; i <= Ng + Nx; ++i) {
                    int idx = k * u_plane + j * u_stride + i;
                    u_b[idx] = a1 * u_a[idx] + a2 * u_b[idx];
                }
            }
        }

        // Blend v-velocity
        #pragma omp target teams distribute parallel for collapse(3)
        for (int k = Ng; k < Ng + Nz; ++k) {
            for (int j = Ng; j <= Ng + Ny; ++j) {
                for (int i = Ng; i < Ng + Nx; ++i) {
                    int idx = k * v_plane + j * v_stride + i;
                    v_b[idx] = a1 * v_a[idx] + a2 * v_b[idx];
                }
            }
        }

        // Blend w-velocity
        #pragma omp target teams distribute parallel for collapse(3)
        for (int k = Ng; k <= Ng + Nz; ++k) {
            for (int j = Ng; j < Ng + Ny; ++j) {
                for (int i = Ng; i < Ng + Nx; ++i) {
                    int idx = k * w_plane + j * w_stride + i;
                    w_b[idx] = a1 * w_a[idx] + a2 * w_b[idx];
                }
            }
        }
    }
}

void blend_to_2d_uv(double* u_a, double* u_b, double* u_c,
                    double* v_a, double* v_b, double* v_c,
                    double a1, double a2,
                    int Nx, int Ny, int Ng, int u_stride, int v_stride,
                    size_t u_total, size_t v_total) {
    (void)u_total; (void)v_total;  // Reserved for future use
    // NVHPC WORKAROUND: Use use_device_ptr to convert host pointers to device pointers
    #pragma omp target data use_device_ptr(u_a, u_b, u_c, v_a, v_b, v_c)
    {
        // Blend u-velocity: u_c = a1 * u_a + a2 * u_b
        #pragma omp target teams distribute parallel for collapse(2)
        for (int j = Ng; j < Ng + Ny; ++j) {
            for (int i = Ng; i <= Ng + Nx; ++i) {
                int idx = j * u_stride + i;
                u_c[idx] = a1 * u_a[idx] + a2 * u_b[idx];
            }
        }

        // Blend v-velocity: v_c = a1 * v_a + a2 * v_b
        #pragma omp target teams distribute parallel for collapse(2)
        for (int j = Ng; j <= Ng + Ny; ++j) {
            for (int i = Ng; i < Ng + Nx; ++i) {
                int idx = j * v_stride + i;
                v_c[idx] = a1 * v_a[idx] + a2 * v_b[idx];
            }
        }
    }
}

void blend_to_3d_uvw(double* u_a, double* u_b, double* u_c,
                     double* v_a, double* v_b, double* v_c,
                     double* w_a, double* w_b, double* w_c,
                     double a1, double a2,
                     int Nx, int Ny, int Nz, int Ng,
                     int u_stride, int v_stride, int w_stride,
                     int u_plane, int v_plane, int w_plane,
                     size_t u_total, size_t v_total, size_t w_total) {
    (void)u_total; (void)v_total; (void)w_total;  // Reserved for future use
    // NVHPC WORKAROUND: Use use_device_ptr to convert host pointers to device pointers
    #pragma omp target data use_device_ptr(u_a, u_b, u_c, v_a, v_b, v_c, w_a, w_b, w_c)
    {
        // Blend u-velocity
        #pragma omp target teams distribute parallel for collapse(3)
        for (int k = Ng; k < Ng + Nz; ++k) {
            for (int j = Ng; j < Ng + Ny; ++j) {
                for (int i = Ng; i <= Ng + Nx; ++i) {
                    int idx = k * u_plane + j * u_stride + i;
                    u_c[idx] = a1 * u_a[idx] + a2 * u_b[idx];
                }
            }
        }

        // Blend v-velocity
        #pragma omp target teams distribute parallel for collapse(3)
        for (int k = Ng; k < Ng + Nz; ++k) {
            for (int j = Ng; j <= Ng + Ny; ++j) {
                for (int i = Ng; i < Ng + Nx; ++i) {
                    int idx = k * v_plane + j * v_stride + i;
                    v_c[idx] = a1 * v_a[idx] + a2 * v_b[idx];
                }
            }
        }

        // Blend w-velocity
        #pragma omp target teams distribute parallel for collapse(3)
        for (int k = Ng; k <= Ng + Nz; ++k) {
            for (int j = Ng; j < Ng + Ny; ++j) {
                for (int i = Ng; i < Ng + Nx; ++i) {
                    int idx = k * w_plane + j * w_stride + i;
                    w_c[idx] = a1 * w_a[idx] + a2 * w_b[idx];
                }
            }
        }
    }
}

} // namespace time_kernels
} // namespace nncfd
