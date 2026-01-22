/// @file solver_time_kernels_periodic.cpp
/// @brief Periodicity enforcement kernels for RK time integration

#include "solver_time_kernels.hpp"
#include "gpu_utils.hpp"

namespace nncfd {
namespace time_kernels {

void periodic_2d(double* u, double* v, bool x_periodic, bool y_periodic,
                 int Nx, int Ny, int Ng, int u_stride, int v_stride,
                 size_t u_total, size_t v_total) {
    (void)u_total; (void)v_total;  // Reserved for future use
    // NVHPC WORKAROUND: Use use_device_ptr to convert host pointers to device pointers
    #pragma omp target data use_device_ptr(u, v)
    {
        // Enforce x-periodicity for u: u[Ng+Nx] = u[Ng]
        if (x_periodic) {
            #pragma omp target teams distribute parallel for
            for (int j = Ng; j < Ng + Ny; ++j) {
                u[j * u_stride + Ng + Nx] = u[j * u_stride + Ng];
            }
        }

        // Enforce y-periodicity for v: v[Ng+Ny] = v[Ng]
        if (y_periodic) {
            #pragma omp target teams distribute parallel for
            for (int i = Ng; i < Ng + Nx; ++i) {
                v[(Ng + Ny) * v_stride + i] = v[Ng * v_stride + i];
            }
        }
    }
}

void periodic_3d(double* u, double* v, double* w,
                 bool x_periodic, bool y_periodic, bool z_periodic,
                 int Nx, int Ny, int Nz, int Ng,
                 int u_stride, int v_stride, int w_stride,
                 int u_plane, int v_plane, int w_plane,
                 size_t u_total, size_t v_total, size_t w_total) {
    (void)u_total; (void)v_total; (void)w_total;  // Reserved for future use
    // NVHPC WORKAROUND: Use use_device_ptr to convert host pointers to device pointers
    #pragma omp target data use_device_ptr(u, v, w)
    {
        // Enforce x-periodicity for u
        if (x_periodic) {
            #pragma omp target teams distribute parallel for collapse(2)
            for (int k = Ng; k < Ng + Nz; ++k) {
                for (int j = Ng; j < Ng + Ny; ++j) {
                    u[k * u_plane + j * u_stride + Ng + Nx] =
                        u[k * u_plane + j * u_stride + Ng];
                }
            }
        }

        // Enforce y-periodicity for v
        if (y_periodic) {
            #pragma omp target teams distribute parallel for collapse(2)
            for (int k = Ng; k < Ng + Nz; ++k) {
                for (int i = Ng; i < Ng + Nx; ++i) {
                    v[k * v_plane + (Ng + Ny) * v_stride + i] =
                        v[k * v_plane + Ng * v_stride + i];
                }
            }
        }

        // Enforce z-periodicity for w
        if (z_periodic) {
            #pragma omp target teams distribute parallel for collapse(2)
            for (int j = Ng; j < Ng + Ny; ++j) {
                for (int i = Ng; i < Ng + Nx; ++i) {
                    w[(Ng + Nz) * w_plane + j * w_stride + i] =
                        w[Ng * w_plane + j * w_stride + i];
                }
            }
        }
    }
}

} // namespace time_kernels
} // namespace nncfd
