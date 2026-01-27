/// @file solver_time_kernels.hpp
/// @brief Kernel declarations for RK time integration
///
/// These kernels handle copy, blend, and time advance operations.
/// Split into multiple files to work around nvc++ compiler limits.

#pragma once
#include <cstddef>

namespace nncfd {
namespace time_kernels {

// ============================================================================
// Blend kernels: b = a1*a + a2*b
// ============================================================================

void blend_2d_uv(double* u_a, double* u_b, double* v_a, double* v_b,
                 double a1, double a2,
                 int Nx, int Ny, int Ng, int u_stride, int v_stride,
                 size_t u_total, size_t v_total);

void blend_3d_uvw(double* u_a, double* u_b, double* v_a, double* v_b,
                  double* w_a, double* w_b, double a1, double a2,
                  int Nx, int Ny, int Nz, int Ng,
                  int u_stride, int v_stride, int w_stride,
                  int u_plane, int v_plane, int w_plane,
                  size_t u_total, size_t v_total, size_t w_total);

// ============================================================================
// Copy kernels: implemented as blend with a1=1, a2=0
// ============================================================================

inline void copy_2d_uv(double* u_src, double* u_dst, double* v_src, double* v_dst,
                       int Nx, int Ny, int Ng, int u_stride, int v_stride,
                       size_t u_total, size_t v_total) {
    blend_2d_uv(u_src, u_dst, v_src, v_dst, 1.0, 0.0,
                Nx, Ny, Ng, u_stride, v_stride, u_total, v_total);
}

inline void copy_3d_uvw(double* u_src, double* u_dst, double* v_src, double* v_dst,
                        double* w_src, double* w_dst,
                        int Nx, int Ny, int Nz, int Ng,
                        int u_stride, int v_stride, int w_stride,
                        int u_plane, int v_plane, int w_plane,
                        size_t u_total, size_t v_total, size_t w_total) {
    blend_3d_uvw(u_src, u_dst, v_src, v_dst, w_src, w_dst, 1.0, 0.0,
                 Nx, Ny, Nz, Ng, u_stride, v_stride, w_stride,
                 u_plane, v_plane, w_plane, u_total, v_total, w_total);
}

// ============================================================================
// Blend-to kernels: c = a1*a + a2*b
// ============================================================================

void blend_to_2d_uv(double* u_a, double* u_b, double* u_c,
                    double* v_a, double* v_b, double* v_c,
                    double a1, double a2,
                    int Nx, int Ny, int Ng, int u_stride, int v_stride,
                    size_t u_total, size_t v_total);

void blend_to_3d_uvw(double* u_a, double* u_b, double* u_c,
                     double* v_a, double* v_b, double* v_c,
                     double* w_a, double* w_b, double* w_c,
                     double a1, double a2,
                     int Nx, int Ny, int Nz, int Ng,
                     int u_stride, int v_stride, int w_stride,
                     int u_plane, int v_plane, int w_plane,
                     size_t u_total, size_t v_total, size_t w_total);

// ============================================================================
// Time advance kernels: out = in + dt * (-conv + diff + f)
// ============================================================================

void euler_advance_2d(double* u_in, double* u_out, double* conv_u, double* diff_u,
                      double* v_in, double* v_out, double* conv_v, double* diff_v,
                      int Nx, int Ny, int Ng, int u_stride, int v_stride,
                      double dt, double fx, double fy,
                      size_t u_total, size_t v_total);

void euler_advance_3d(double* u_in, double* u_out, double* conv_u, double* diff_u,
                      double* v_in, double* v_out, double* conv_v, double* diff_v,
                      double* w_in, double* w_out, double* conv_w, double* diff_w,
                      int Nx, int Ny, int Nz, int Ng,
                      int u_stride, int v_stride, int w_stride,
                      int u_plane, int v_plane, int w_plane,
                      double dt, double fx, double fy, double fz,
                      size_t u_total, size_t v_total, size_t w_total);

// ============================================================================
// Periodicity enforcement kernels
// ============================================================================

void periodic_2d(double* u, double* v, bool x_periodic, bool y_periodic,
                 int Nx, int Ny, int Ng, int u_stride, int v_stride,
                 size_t u_total, size_t v_total);

void periodic_3d(double* u, double* v, double* w,
                 bool x_periodic, bool y_periodic, bool z_periodic,
                 int Nx, int Ny, int Nz, int Ng,
                 int u_stride, int v_stride, int w_stride,
                 int u_plane, int v_plane, int w_plane,
                 size_t u_total, size_t v_total, size_t w_total);

// ============================================================================
// Projection kernels (for RK methods)
// ============================================================================

/// Compute mean divergence via GPU reduction
double compute_mean_divergence_2d(double* div, int Nx, int Ny, int Ng,
                                   int stride, size_t total);

double compute_mean_divergence_3d(double* div, int Nx, int Ny, int Nz, int Ng,
                                   int stride, int plane_stride, size_t total);

/// Build Poisson RHS: rhs = (div - mean_div) / dt
void build_poisson_rhs_2d(double* div, double* rhs, double mean_div, double dt_inv,
                           int Nx, int Ny, int Ng, int stride, size_t total);

void build_poisson_rhs_3d(double* div, double* rhs, double mean_div, double dt_inv,
                           int Nx, int Ny, int Nz, int Ng,
                           int stride, int plane_stride, size_t total);

/// Update pressure: p += p_correction
void update_pressure_2d(double* p, double* p_corr, int Nx, int Ny, int Ng,
                         int stride, size_t total);

void update_pressure_3d(double* p, double* p_corr, int Nx, int Ny, int Nz, int Ng,
                         int stride, int plane_stride, size_t total);

} // namespace time_kernels
} // namespace nncfd
