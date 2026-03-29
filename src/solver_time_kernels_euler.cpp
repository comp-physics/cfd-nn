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
                      size_t u_total, size_t v_total,
                      double* tau_div_u, double* tau_div_v) {
    (void)u_total; (void)v_total;  // Reserved for future use
    // NVHPC WORKAROUND: Use use_device_ptr to convert host pointers to device pointers
    // within this region. The inner target regions then use these converted pointers.
    if (tau_div_u && tau_div_v) {
        #pragma omp target data use_device_ptr(u_in, u_out, conv_u, diff_u, v_in, v_out, conv_v, diff_v, tau_div_u, tau_div_v)
        {
            #pragma omp target teams distribute parallel for collapse(2)
            for (int j = Ng; j < Ng + Ny; ++j) {
                for (int i = Ng; i <= Ng + Nx; ++i) {
                    int idx = j * u_stride + i;
                    u_out[idx] = u_in[idx] + dt * (-conv_u[idx] + diff_u[idx] + fx + tau_div_u[idx]);
                }
            }
            #pragma omp target teams distribute parallel for collapse(2)
            for (int j = Ng; j <= Ng + Ny; ++j) {
                for (int i = Ng; i < Ng + Nx; ++i) {
                    int idx = j * v_stride + i;
                    v_out[idx] = v_in[idx] + dt * (-conv_v[idx] + diff_v[idx] + fy + tau_div_v[idx]);
                }
            }
        }
    } else {
        #pragma omp target data use_device_ptr(u_in, u_out, conv_u, diff_u, v_in, v_out, conv_v, diff_v)
        {
            #pragma omp target teams distribute parallel for collapse(2)
            for (int j = Ng; j < Ng + Ny; ++j) {
                for (int i = Ng; i <= Ng + Nx; ++i) {
                    int idx = j * u_stride + i;
                    u_out[idx] = u_in[idx] + dt * (-conv_u[idx] + diff_u[idx] + fx);
                }
            }
            #pragma omp target teams distribute parallel for collapse(2)
            for (int j = Ng; j <= Ng + Ny; ++j) {
                for (int i = Ng; i < Ng + Nx; ++i) {
                    int idx = j * v_stride + i;
                    v_out[idx] = v_in[idx] + dt * (-conv_v[idx] + diff_v[idx] + fy);
                }
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
                      size_t u_total, size_t v_total, size_t w_total,
                      double* tau_div_u, double* tau_div_v, double* tau_div_w) {
    (void)u_total; (void)v_total; (void)w_total;  // Reserved for future use
    // NVHPC WORKAROUND: Use use_device_ptr to convert host pointers to device pointers
    if (tau_div_u && tau_div_v && tau_div_w) {
        #pragma omp target data use_device_ptr(u_in, u_out, conv_u, diff_u, v_in, v_out, conv_v, diff_v, w_in, w_out, conv_w, diff_w, tau_div_u, tau_div_v, tau_div_w)
        {
            #pragma omp target teams distribute parallel for collapse(3)
            for (int k = Ng; k < Ng + Nz; ++k) {
                for (int j = Ng; j < Ng + Ny; ++j) {
                    for (int i = Ng; i <= Ng + Nx; ++i) {
                        int idx = k * u_plane + j * u_stride + i;
                        u_out[idx] = u_in[idx] + dt * (-conv_u[idx] + diff_u[idx] + fx + tau_div_u[idx]);
                    }
                }
            }
            #pragma omp target teams distribute parallel for collapse(3)
            for (int k = Ng; k < Ng + Nz; ++k) {
                for (int j = Ng; j <= Ng + Ny; ++j) {
                    for (int i = Ng; i < Ng + Nx; ++i) {
                        int idx = k * v_plane + j * v_stride + i;
                        v_out[idx] = v_in[idx] + dt * (-conv_v[idx] + diff_v[idx] + fy + tau_div_v[idx]);
                    }
                }
            }
            #pragma omp target teams distribute parallel for collapse(3)
            for (int k = Ng; k <= Ng + Nz; ++k) {
                for (int j = Ng; j < Ng + Ny; ++j) {
                    for (int i = Ng; i < Ng + Nx; ++i) {
                        int idx = k * w_plane + j * w_stride + i;
                        w_out[idx] = w_in[idx] + dt * (-conv_w[idx] + diff_w[idx] + fz + tau_div_w[idx]);
                    }
                }
            }
        }
    } else {
        #pragma omp target data use_device_ptr(u_in, u_out, conv_u, diff_u, v_in, v_out, conv_v, diff_v, w_in, w_out, conv_w, diff_w)
        {
            #pragma omp target teams distribute parallel for collapse(3)
            for (int k = Ng; k < Ng + Nz; ++k) {
                for (int j = Ng; j < Ng + Ny; ++j) {
                    for (int i = Ng; i <= Ng + Nx; ++i) {
                        int idx = k * u_plane + j * u_stride + i;
                        u_out[idx] = u_in[idx] + dt * (-conv_u[idx] + diff_u[idx] + fx);
                    }
                }
            }
            #pragma omp target teams distribute parallel for collapse(3)
            for (int k = Ng; k < Ng + Nz; ++k) {
                for (int j = Ng; j <= Ng + Ny; ++j) {
                    for (int i = Ng; i < Ng + Nx; ++i) {
                        int idx = k * v_plane + j * v_stride + i;
                        v_out[idx] = v_in[idx] + dt * (-conv_v[idx] + diff_v[idx] + fy);
                    }
                }
            }
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
}

} // namespace time_kernels
} // namespace nncfd
