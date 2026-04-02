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
                      size_t u_total, size_t v_total,
                      double* tau_div_u = nullptr, double* tau_div_v = nullptr);

void euler_advance_3d(double* u_in, double* u_out, double* conv_u, double* diff_u,
                      double* v_in, double* v_out, double* conv_v, double* diff_v,
                      double* w_in, double* w_out, double* conv_w, double* diff_w,
                      int Nx, int Ny, int Nz, int Ng,
                      int u_stride, int v_stride, int w_stride,
                      int u_plane, int v_plane, int w_plane,
                      double dt, double fx, double fy, double fz,
                      size_t u_total, size_t v_total, size_t w_total,
                      double* tau_div_u = nullptr, double* tau_div_v = nullptr,
                      double* tau_div_w = nullptr);

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

// ============================================================================
// Implicit y-diffusion (Thomas algorithm tridiagonal solve per column)
// ============================================================================

void thomas_y_diffusion_2d(double* u, double* v, double* nu_eff,
                           int Nx, int Ny, int Ng,
                           int u_stride, int v_stride, int cell_stride,
                           double dt, double dy);

void thomas_y_diffusion_3d(double* u, double* v, double* w, double* nu_eff,
                           int Nx, int Ny, int Nz, int Ng,
                           int u_stride, int v_stride, int w_stride,
                           int u_plane, int v_plane, int w_plane,
                           int cell_stride, int cell_plane,
                           double dt, double dy);

/// Stretched-grid variants: use per-row dyv[j], dyc[j] instead of uniform dy.
/// dyv[j] = cell height at row j, dyc[j] = center-to-center at face j.
/// Both arrays must be device-resident (mapped via target enter data).
void thomas_y_diffusion_2d_stretched(double* u, double* v, double* nu_eff,
                                     int Nx, int Ny, int Ng,
                                     int u_stride, int v_stride, int cell_stride,
                                     double dt,
                                     const double* dyv, const double* dyc);

void thomas_y_diffusion_3d_stretched(double* u, double* v, double* w, double* nu_eff,
                                     int Nx, int Ny, int Nz, int Ng,
                                     int u_stride, int v_stride, int w_stride,
                                     int u_plane, int v_plane, int w_plane,
                                     int cell_stride, int cell_plane,
                                     double dt,
                                     const double* dyv, const double* dyc);

// ============================================================================
// SIMPLE steady-state solver kernels
// ============================================================================

void simple_compute_aP_2d(double* a_p_u, double* a_p_v,
                          const double* nu_eff,
                          const double* u_face, const double* v_face,
                          int Nx, int Ny, int Ng,
                          int u_stride, int v_stride, int cell_stride,
                          double dx, double dy,
                          double pseudo_dt_inv);

void simple_compute_aP_3d(double* a_p_u, double* a_p_v, double* a_p_w,
                          const double* nu_eff,
                          const double* u_face, const double* v_face, const double* w_face,
                          int Nx, int Ny, int Nz, int Ng,
                          int u_stride, int u_plane,
                          int v_stride, int v_plane,
                          int w_stride, int w_plane,
                          int cell_stride, int cell_plane,
                          double dx, double dy, double dz,
                          double pseudo_dt_inv);

void simple_predictor_2d(double* u_star, double* v_star,
                         const double* u, const double* v,
                         const double* u_old, const double* v_old,
                         const double* conv_u, const double* conv_v,
                         const double* diff_u, const double* diff_v,
                         const double* tau_div_u, const double* tau_div_v,
                         const double* a_p_u, const double* a_p_v,
                         const double* p,
                         double fx, double fy, double dx, double dy,
                         double alpha_u,
                         int Nx, int Ny, int Ng,
                         int u_stride, int v_stride, int cell_stride);

void simple_predictor_3d(double* u_star, double* v_star, double* w_star,
                         const double* u, const double* v, const double* w,
                         const double* u_old, const double* v_old, const double* w_old,
                         const double* conv_u, const double* conv_v, const double* conv_w,
                         const double* diff_u, const double* diff_v, const double* diff_w,
                         const double* tau_div_u, const double* tau_div_v, const double* tau_div_w,
                         const double* a_p_u, const double* a_p_v, const double* a_p_w,
                         const double* p,
                         double fx, double fy, double fz,
                         double dx, double dy, double dz,
                         double alpha_u,
                         int Nx, int Ny, int Nz, int Ng,
                         int u_stride, int u_plane,
                         int v_stride, int v_plane,
                         int w_stride, int w_plane,
                         int cell_stride, int cell_plane);

void simple_correct_velocity_2d(double* u, double* v, double* p,
                                const double* u_star, const double* v_star,
                                const double* p_corr,
                                const double* a_p_u, const double* a_p_v,
                                double alpha_p, double dx, double dy,
                                double pseudo_dt_inv, double vol,
                                int Nx, int Ny, int Ng,
                                int u_stride, int v_stride, int cell_stride);

void simple_correct_velocity_3d(double* u, double* v, double* w, double* p,
                                const double* u_star, const double* v_star,
                                const double* w_star, const double* p_corr,
                                const double* a_p_u, const double* a_p_v,
                                const double* a_p_w,
                                double alpha_p, double dx, double dy, double dz,
                                int Nx, int Ny, int Nz, int Ng,
                                int u_stride, int u_plane,
                                int v_stride, int v_plane,
                                int w_stride, int w_plane,
                                int cell_stride, int cell_plane);

// Variable-coefficient Poisson residual: r = rhs - ∇·(1/a_P · ∇p')
// Used in defect-correction loop with constant-coefficient MG as preconditioner
void simple_varcoeff_residual_2d(
    double* residual, const double* p, const double* rhs,
    const double* a_p_u, const double* a_p_v,
    double dx, double dy,
    int Nx, int Ny, int Ng,
    int cell_stride, int u_stride, int v_stride);

// Jacobi momentum sweep: implicit convection + diffusion + frozen pressure
void simple_jacobi_momentum_2d(
    double* u_new, double* v_new,
    const double* u_iter, const double* v_iter,
    const double* u_frozen, const double* v_frozen,
    const double* nu_eff,
    const double* p_old,
    const double* tau_div_u, const double* tau_div_v,
    double fx, double fy, double dx, double dy,
    double pseudo_dt_inv,
    int Nx, int Ny, int Ng,
    int u_stride, int v_stride, int cell_stride);

} // namespace time_kernels
} // namespace nncfd
