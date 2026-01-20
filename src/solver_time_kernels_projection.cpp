/// @file solver_time_kernels_projection.cpp
/// @brief Projection kernels for RK time integration

#include "solver_time_kernels.hpp"
#include "gpu_utils.hpp"

namespace nncfd {
namespace time_kernels {

double compute_mean_divergence_2d(double* div, int Nx, int Ny, int Ng,
                                   int stride, size_t total) {
    double sum = 0.0;
    const int count = Nx * Ny;

    // NVHPC WORKAROUND: Use use_device_ptr to convert host pointer to device pointer
    #pragma omp target data use_device_ptr(div)
    {
        #pragma omp target teams distribute parallel for collapse(2) reduction(+:sum)
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                int idx = (j + Ng) * stride + (i + Ng);
                sum += div[idx];
            }
        }
    }

    return (count > 0) ? sum / count : 0.0;
}

double compute_mean_divergence_3d(double* div, int Nx, int Ny, int Nz, int Ng,
                                   int stride, int plane_stride, size_t total) {
    double sum = 0.0;
    const int count = Nx * Ny * Nz;

    // NVHPC WORKAROUND: Use use_device_ptr to convert host pointer to device pointer
    #pragma omp target data use_device_ptr(div)
    {
        #pragma omp target teams distribute parallel for collapse(3) reduction(+:sum)
        for (int k = 0; k < Nz; ++k) {
            for (int j = 0; j < Ny; ++j) {
                for (int i = 0; i < Nx; ++i) {
                    int idx = (k + Ng) * plane_stride + (j + Ng) * stride + (i + Ng);
                    sum += div[idx];
                }
            }
        }
    }

    return (count > 0) ? sum / count : 0.0;
}

void build_poisson_rhs_2d(double* div, double* rhs, double mean_div, double dt_inv,
                           int Nx, int Ny, int Ng, int stride, size_t total) {
    // NVHPC WORKAROUND: Use use_device_ptr to convert host pointers to device pointers
    #pragma omp target data use_device_ptr(div, rhs)
    {
        #pragma omp target teams distribute parallel for collapse(2)
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                int idx = (j + Ng) * stride + (i + Ng);
                rhs[idx] = (div[idx] - mean_div) * dt_inv;
            }
        }
    }
}

void build_poisson_rhs_3d(double* div, double* rhs, double mean_div, double dt_inv,
                           int Nx, int Ny, int Nz, int Ng,
                           int stride, int plane_stride, size_t total) {
    // NVHPC WORKAROUND: Use use_device_ptr to convert host pointers to device pointers
    #pragma omp target data use_device_ptr(div, rhs)
    {
        #pragma omp target teams distribute parallel for collapse(3)
        for (int k = 0; k < Nz; ++k) {
            for (int j = 0; j < Ny; ++j) {
                for (int i = 0; i < Nx; ++i) {
                    int idx = (k + Ng) * plane_stride + (j + Ng) * stride + (i + Ng);
                    rhs[idx] = (div[idx] - mean_div) * dt_inv;
                }
            }
        }
    }
}

void update_pressure_2d(double* p, double* p_corr, int Nx, int Ny, int Ng,
                         int stride, size_t total) {
    // NVHPC WORKAROUND: Use use_device_ptr to convert host pointers to device pointers
    #pragma omp target data use_device_ptr(p, p_corr)
    {
        #pragma omp target teams distribute parallel for collapse(2)
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                int idx = (j + Ng) * stride + (i + Ng);
                p[idx] += p_corr[idx];
            }
        }
    }
}

void update_pressure_3d(double* p, double* p_corr, int Nx, int Ny, int Nz, int Ng,
                         int stride, int plane_stride, size_t total) {
    // NVHPC WORKAROUND: Use use_device_ptr to convert host pointers to device pointers
    #pragma omp target data use_device_ptr(p, p_corr)
    {
        #pragma omp target teams distribute parallel for collapse(3)
        for (int k = 0; k < Nz; ++k) {
            for (int j = 0; j < Ny; ++j) {
                for (int i = 0; i < Nx; ++i) {
                    int idx = (k + Ng) * plane_stride + (j + Ng) * stride + (i + Ng);
                    p[idx] += p_corr[idx];
                }
            }
        }
    }
}

} // namespace time_kernels
} // namespace nncfd
