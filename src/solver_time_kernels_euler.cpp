/// @file solver_time_kernels_euler.cpp
/// @brief Euler advance kernels for RK time integration

#include "solver_time_kernels.hpp"
#include "gpu_utils.hpp"

namespace nncfd {
namespace time_kernels {

void euler_advance_2d(double* u_in, double* u_out, double* conv_u, double* diff_u,
                      double* v_in, double* v_out, double* conv_v, double* diff_v,
                      int Nx, int Ny, int Ng, int u_stride, int v_stride,
                      double dt, double fx, double fy,
                      size_t u_total, size_t v_total) {
    (void)u_total; (void)v_total;  // Reserved for future use
    // NVHPC WORKAROUND: Use use_device_ptr to convert host pointers to device pointers
    // within this region. The inner target regions then use these converted pointers.
    #pragma omp target data use_device_ptr(u_in, u_out, conv_u, diff_u, v_in, v_out, conv_v, diff_v)
    {
        // Advance u-velocity: u_out = u_in + dt * (-conv_u + diff_u + fx)
        #pragma omp target teams distribute parallel for collapse(2)
        for (int j = Ng; j < Ng + Ny; ++j) {
            for (int i = Ng; i <= Ng + Nx; ++i) {
                int idx = j * u_stride + i;
                u_out[idx] = u_in[idx] + dt * (-conv_u[idx] + diff_u[idx] + fx);
            }
        }

        // Advance v-velocity: v_out = v_in + dt * (-conv_v + diff_v + fy)
        #pragma omp target teams distribute parallel for collapse(2)
        for (int j = Ng; j <= Ng + Ny; ++j) {
            for (int i = Ng; i < Ng + Nx; ++i) {
                int idx = j * v_stride + i;
                v_out[idx] = v_in[idx] + dt * (-conv_v[idx] + diff_v[idx] + fy);
            }
        }
    }
}

void euler_advance_3d(double* u_in, double* u_out, double* conv_u, double* diff_u,
                      double* v_in, double* v_out, double* conv_v, double* diff_v,
                      double* w_in, double* w_out, double* conv_w, double* diff_w,
                      int Nx, int Ny, int Nz, int Ng,
                      int u_stride, int v_stride, int w_stride,
                      int u_plane, int v_plane, int w_plane,
                      double dt, double fx, double fy, double fz,
                      size_t u_total, size_t v_total, size_t w_total) {
    (void)u_total; (void)v_total; (void)w_total;  // Reserved for future use
    // NVHPC WORKAROUND: Use use_device_ptr to convert host pointers to device pointers
    #pragma omp target data use_device_ptr(u_in, u_out, conv_u, diff_u, v_in, v_out, conv_v, diff_v, w_in, w_out, conv_w, diff_w)
    {
        // Advance u-velocity
        #pragma omp target teams distribute parallel for collapse(3)
        for (int k = Ng; k < Ng + Nz; ++k) {
            for (int j = Ng; j < Ng + Ny; ++j) {
                for (int i = Ng; i <= Ng + Nx; ++i) {
                    int idx = k * u_plane + j * u_stride + i;
                    u_out[idx] = u_in[idx] + dt * (-conv_u[idx] + diff_u[idx] + fx);
                }
            }
        }

        // Advance v-velocity
        #pragma omp target teams distribute parallel for collapse(3)
        for (int k = Ng; k < Ng + Nz; ++k) {
            for (int j = Ng; j <= Ng + Ny; ++j) {
                for (int i = Ng; i < Ng + Nx; ++i) {
                    int idx = k * v_plane + j * v_stride + i;
                    v_out[idx] = v_in[idx] + dt * (-conv_v[idx] + diff_v[idx] + fy);
                }
            }
        }

        // Advance w-velocity
        #pragma omp target teams distribute parallel for collapse(3)
        for (int k = Ng; k <= Ng + Nz; ++k) {
            for (int j = Ng; j < Ng + Ny; ++j) {
                for (int i = Ng; i < Ng + Nx; ++i) {
                    int idx = k * w_plane + j * w_stride + i;
                    w_out[idx] = w_in[idx] + dt * (-conv_w[idx] + diff_w[idx] + fz);
                }
            }
        }
    }
}

} // namespace time_kernels
} // namespace nncfd
