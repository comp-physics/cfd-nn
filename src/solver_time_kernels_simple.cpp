/// @file solver_time_kernels_simple.cpp
/// @brief GPU kernels for SIMPLE steady-state solver
///
/// Diagonal-approximation SIMPLE: compute a_P, momentum predictor,
/// a_P-weighted velocity correction, under-relaxed pressure update.
/// Free functions to avoid NVHPC this-pointer transfer.

#include "solver_time_kernels.hpp"
#include <cstdio>

namespace nncfd {
namespace time_kernels {

// ============================================================================
// Compute diagonal coefficient a_P at velocity faces
// ============================================================================

void simple_compute_aP_2d(double* a_p_u, double* a_p_v,
                          const double* nu_eff,
                          const double* u_face, const double* v_face,
                          int Nx, int Ny, int Ng,
                          int u_stride, int v_stride, int cell_stride,
                          double dx, double dy,
                          double pseudo_dt_inv) {
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double vol = dx * dy;
    // pseudo_dt_inv = 1/dt_pseudo: damping term for pseudo-transient SIMPLE

    [[maybe_unused]] const size_t nu_sz = static_cast<size_t>((Ny + 2 * Ng) * cell_stride);
    [[maybe_unused]] const size_t u_sz = static_cast<size_t>((Ny + 2 * Ng) * u_stride);
    [[maybe_unused]] const size_t v_sz = static_cast<size_t>((Ny + 2 * Ng + 1) * v_stride);

    // a_P for u-faces: diffusion + convective diagonal + pseudo-transient
    #pragma omp target teams distribute parallel for collapse(2) \
        map(present: a_p_u[0:u_sz], nu_eff[0:nu_sz], u_face[0:u_sz], v_face[0:v_sz])
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i <= Nx; ++i) {
            int jg = j + Ng;
            int ig = i + Ng;
            int u_idx = jg * u_stride + ig;

            int c_left  = jg * cell_stride + (ig - 1);
            int c_right = jg * cell_stride + ig;
            int cl = (ig > 0) ? c_left : c_right;
            int cr = (ig < Nx + 2 * Ng - 1) ? c_right : c_left;
            double nu_avg = 0.5 * (nu_eff[cl] + nu_eff[cr]);

            // Diffusion diagonal
            double aP_diff = nu_avg * (2.0 * inv_dx2 + 2.0 * inv_dy2) * vol;

            // Convective diagonal (upwind-like): sum of absolute face fluxes
            // At u-face (ig, jg), the cell around it has faces at ig-1/2, ig+1/2, jg-1/2, jg+1/2
            double u_e = u_face[jg * u_stride + (ig + 1)];
            double u_w = u_face[jg * u_stride + (ig - 1)];
            double v_n = v_face[(jg + 1) * v_stride + ig];
            double v_s = v_face[jg * v_stride + ig];
            // Upwind: max(F,0) + max(-F,0) = |F|/2 for each face, summed
            double aP_conv = 0.5 * ((u_e > 0 ? u_e : -u_e) * dy
                                  + (u_w > 0 ? u_w : -u_w) * dy
                                  + (v_n > 0 ? v_n : -v_n) * dx
                                  + (v_s > 0 ? v_s : -v_s) * dx);

            double aP = aP_diff + aP_conv + vol * pseudo_dt_inv;
            a_p_u[u_idx] = (aP > 1e-20) ? aP : 1e-20;
        }
    }

    // a_P for v-faces
    #pragma omp target teams distribute parallel for collapse(2) \
        map(present: a_p_v[0:v_sz], nu_eff[0:nu_sz], u_face[0:u_sz], v_face[0:v_sz])
    for (int j = 0; j <= Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            int jg = j + Ng;
            int ig = i + Ng;
            int v_idx = jg * v_stride + ig;

            int c_bot = (jg - 1) * cell_stride + ig;
            int c_top = jg * cell_stride + ig;
            int cb = (jg > 0) ? c_bot : c_top;
            int ct = (jg < Ny + 2 * Ng - 1) ? c_top : c_bot;
            double nu_avg = 0.5 * (nu_eff[cb] + nu_eff[ct]);

            double aP_diff = nu_avg * (2.0 * inv_dx2 + 2.0 * inv_dy2) * vol;

            double u_e = u_face[jg * u_stride + (ig + 1)];
            double u_w = u_face[jg * u_stride + ig];
            double v_n = v_face[(jg + 1) * v_stride + ig];
            double v_s = v_face[(jg - 1) * v_stride + ig];
            double aP_conv = 0.5 * ((u_e > 0 ? u_e : -u_e) * dy
                                  + (u_w > 0 ? u_w : -u_w) * dy
                                  + (v_n > 0 ? v_n : -v_n) * dx
                                  + (v_s > 0 ? v_s : -v_s) * dx);

            double aP = aP_diff + aP_conv + vol * pseudo_dt_inv;
            a_p_v[v_idx] = (aP > 1e-20) ? aP : 1e-20;
        }
    }
}

void simple_compute_aP_3d(double* a_p_u, double* a_p_v, double* a_p_w,
                          const double* nu_eff,
                          const double* u_face, const double* v_face, const double* w_face,
                          int Nx, int Ny, int Nz, int Ng,
                          int u_stride, int u_plane,
                          int v_stride, int v_plane,
                          int w_stride, int w_plane,
                          int cell_stride, int cell_plane,
                          double dx, double dy, double dz,
                          double pseudo_dt_inv) {
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double inv_dz2 = 1.0 / (dz * dz);
    const double vol = dx * dy * dz;
    const double diff_diag = 2.0 * inv_dx2 + 2.0 * inv_dy2 + 2.0 * inv_dz2;

    [[maybe_unused]] const size_t nu_sz = static_cast<size_t>((Nz + 2 * Ng) * cell_plane);
    [[maybe_unused]] const size_t u_sz = static_cast<size_t>((Nz + 2 * Ng) * u_plane);
    [[maybe_unused]] const size_t v_sz = static_cast<size_t>((Nz + 2 * Ng) * v_plane);
    [[maybe_unused]] const size_t w_sz = static_cast<size_t>((Nz + 2 * Ng + 1) * w_plane);

    const double Ay = dx * dz;  // face area normal to y
    const double Ax = dy * dz;  // face area normal to x
    const double Az = dx * dy;  // face area normal to z

    // a_P for u-faces (diffusion + convection + pseudo-transient)
    const int n_u = (Nx + 1) * Ny * Nz;
    #pragma omp target teams distribute parallel for \
        map(present: a_p_u[0:u_sz], nu_eff[0:nu_sz], u_face[0:u_sz], v_face[0:v_sz], w_face[0:w_sz])
    for (int idx = 0; idx < n_u; ++idx) {
        int i = idx % (Nx + 1) + Ng;
        int j = (idx / (Nx + 1)) % Ny + Ng;
        int k = idx / ((Nx + 1) * Ny) + Ng;

        int c_left  = k * cell_plane + j * cell_stride + (i - 1);
        int c_right = k * cell_plane + j * cell_stride + i;
        double nu_avg = 0.5 * (nu_eff[c_left] + nu_eff[c_right]);
        double aP_diff = nu_avg * diff_diag * vol;

        // Convective diagonal: sum of |face flux * face area|
        double ue = u_face[k * u_plane + j * u_stride + (i+1)];
        double uw = u_face[k * u_plane + j * u_stride + (i-1)];
        double vn = v_face[k * v_plane + (j+1) * v_stride + i];
        double vs = v_face[k * v_plane + j * v_stride + i];
        double wf = w_face[(k+1) * w_plane + j * w_stride + i];
        double wb = w_face[k * w_plane + j * w_stride + i];
        double aP_conv = 0.5 * ((ue>0?ue:-ue)*Ax + (uw>0?uw:-uw)*Ax
                              + (vn>0?vn:-vn)*Ay + (vs>0?vs:-vs)*Ay
                              + (wf>0?wf:-wf)*Az + (wb>0?wb:-wb)*Az);

        double aP = aP_diff + aP_conv + vol * pseudo_dt_inv;
        a_p_u[k * u_plane + j * u_stride + i] = (aP > 1e-20) ? aP : 1e-20;
    }

    // a_P for v-faces
    const int n_v = Nx * (Ny + 1) * Nz;
    #pragma omp target teams distribute parallel for \
        map(present: a_p_v[0:v_sz], nu_eff[0:nu_sz], u_face[0:u_sz], v_face[0:v_sz], w_face[0:w_sz])
    for (int idx = 0; idx < n_v; ++idx) {
        int i = idx % Nx + Ng;
        int j = (idx / Nx) % (Ny + 1) + Ng;
        int k = idx / (Nx * (Ny + 1)) + Ng;

        int c_bot = k * cell_plane + (j - 1) * cell_stride + i;
        int c_top = k * cell_plane + j * cell_stride + i;
        double nu_avg = 0.5 * (nu_eff[c_bot] + nu_eff[c_top]);
        double aP_diff = nu_avg * diff_diag * vol;

        double ue = u_face[k * u_plane + j * u_stride + (i+1)];
        double uw = u_face[k * u_plane + j * u_stride + i];
        double vn = v_face[k * v_plane + (j+1) * v_stride + i];
        double vs = v_face[k * v_plane + (j-1) * v_stride + i];
        double wf = w_face[(k+1) * w_plane + j * w_stride + i];
        double wb = w_face[k * w_plane + j * w_stride + i];
        double aP_conv = 0.5 * ((ue>0?ue:-ue)*Ax + (uw>0?uw:-uw)*Ax
                              + (vn>0?vn:-vn)*Ay + (vs>0?vs:-vs)*Ay
                              + (wf>0?wf:-wf)*Az + (wb>0?wb:-wb)*Az);

        double aP = aP_diff + aP_conv + vol * pseudo_dt_inv;
        a_p_v[k * v_plane + j * v_stride + i] = (aP > 1e-20) ? aP : 1e-20;
    }

    // a_P for w-faces
    const int n_w = Nx * Ny * (Nz + 1);
    #pragma omp target teams distribute parallel for \
        map(present: a_p_w[0:w_sz], nu_eff[0:nu_sz], u_face[0:u_sz], v_face[0:v_sz], w_face[0:w_sz])
    for (int idx = 0; idx < n_w; ++idx) {
        int i = idx % Nx + Ng;
        int j = (idx / Nx) % Ny + Ng;
        int k = idx / (Nx * Ny) + Ng;

        int c_lo = (k - 1) * cell_plane + j * cell_stride + i;
        int c_hi = k * cell_plane + j * cell_stride + i;
        double nu_avg = 0.5 * (nu_eff[c_lo] + nu_eff[c_hi]);
        double aP_diff = nu_avg * diff_diag * vol;

        double ue = u_face[k * u_plane + j * u_stride + (i+1)];
        double uw = u_face[k * u_plane + j * u_stride + i];
        double vn = v_face[k * v_plane + (j+1) * v_stride + i];
        double vs = v_face[k * v_plane + j * v_stride + i];
        double wf = w_face[(k+1) * w_plane + j * w_stride + i];
        double wb = w_face[(k-1) * w_plane + j * w_stride + i];
        double aP_conv = 0.5 * ((ue>0?ue:-ue)*Ax + (uw>0?uw:-uw)*Ax
                              + (vn>0?vn:-vn)*Ay + (vs>0?vs:-vs)*Ay
                              + (wf>0?wf:-wf)*Az + (wb>0?wb:-wb)*Az);

        double aP = aP_diff + aP_conv + vol * pseudo_dt_inv;
        a_p_w[k * w_plane + j * w_stride + i] = (aP > 1e-20) ? aP : 1e-20;
    }
}

// ============================================================================
// SIMPLE momentum predictor with under-relaxation
// u* = alpha_u * [H(u)/a_P - grad(p)/a_P] + (1-alpha_u) * u_old
// ============================================================================

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
                         int u_stride, int v_stride, int cell_stride) {
    [[maybe_unused]] const size_t u_sz = static_cast<size_t>((Ny + 2 * Ng) * u_stride);
    [[maybe_unused]] const size_t v_sz = static_cast<size_t>((Ny + 2 * Ng + 1) * v_stride);
    [[maybe_unused]] const size_t p_sz = static_cast<size_t>((Ny + 2 * Ng) * cell_stride);
    const double one_m_alpha = 1.0 - alpha_u;

    // u-predictor: u* = u + alpha * (H_u - dp/dx) / a_P, blended with u_old
    #pragma omp target teams distribute parallel for collapse(2) \
        map(present: u_star[0:u_sz], u[0:u_sz], u_old[0:u_sz], conv_u[0:u_sz], diff_u[0:u_sz], \
                     tau_div_u[0:u_sz], a_p_u[0:u_sz], p[0:p_sz])
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i <= Nx; ++i) {
            int jg = j + Ng;
            int ig = i + Ng;
            int u_idx = jg * u_stride + ig;

            double H_u = -conv_u[u_idx] + diff_u[u_idx] + tau_div_u[u_idx] + fx;
            double dp_dx = (p[jg * cell_stride + ig] - p[jg * cell_stride + (ig - 1)]) / dx;
            double u_predicted = u[u_idx] + (H_u - dp_dx) / a_p_u[u_idx];
            u_star[u_idx] = alpha_u * u_predicted + one_m_alpha * u_old[u_idx];
        }
    }

    // v-predictor: v* = v + alpha * (H_v - dp/dy) / a_P, blended with v_old
    #pragma omp target teams distribute parallel for collapse(2) \
        map(present: v_star[0:v_sz], v[0:v_sz], v_old[0:v_sz], conv_v[0:v_sz], diff_v[0:v_sz], \
                     tau_div_v[0:v_sz], a_p_v[0:v_sz], p[0:p_sz])
    for (int j = 0; j <= Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            int jg = j + Ng;
            int ig = i + Ng;
            int v_idx = jg * v_stride + ig;

            double H_v = -conv_v[v_idx] + diff_v[v_idx] + tau_div_v[v_idx] + fy;
            double dp_dy = (p[jg * cell_stride + ig] - p[(jg - 1) * cell_stride + ig]) / dy;
            double v_predicted = v[v_idx] + (H_v - dp_dy) / a_p_v[v_idx];
            v_star[v_idx] = alpha_u * v_predicted + one_m_alpha * v_old[v_idx];
        }
    }
}

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
                         int cell_stride, int cell_plane) {
    [[maybe_unused]] const size_t u_sz = static_cast<size_t>((Nz + 2 * Ng) * u_plane);
    [[maybe_unused]] const size_t v_sz = static_cast<size_t>((Nz + 2 * Ng) * v_plane);
    [[maybe_unused]] const size_t w_sz = static_cast<size_t>((Nz + 2 * Ng + 1) * w_plane);
    [[maybe_unused]] const size_t p_sz = static_cast<size_t>((Nz + 2 * Ng) * cell_plane);
    const double one_m_alpha = 1.0 - alpha_u;

    // u-predictor: u* = u + (H_u - dp/dx) / a_P, blended with u_old
    const int n_u = (Nx + 1) * Ny * Nz;
    #pragma omp target teams distribute parallel for \
        map(present: u_star[0:u_sz], u[0:u_sz], u_old[0:u_sz], conv_u[0:u_sz], diff_u[0:u_sz], \
                     tau_div_u[0:u_sz], a_p_u[0:u_sz], p[0:p_sz])
    for (int idx = 0; idx < n_u; ++idx) {
        int i = idx % (Nx + 1) + Ng;
        int j = (idx / (Nx + 1)) % Ny + Ng;
        int k = idx / ((Nx + 1) * Ny) + Ng;
        int u_idx = k * u_plane + j * u_stride + i;

        double H_u = -conv_u[u_idx] + diff_u[u_idx] + tau_div_u[u_idx] + fx;
        double dp_dx = (p[k * cell_plane + j * cell_stride + i] -
                        p[k * cell_plane + j * cell_stride + (i - 1)]) / dx;
        double u_predicted = u[u_idx] + (H_u - dp_dx) / a_p_u[u_idx];
        u_star[u_idx] = alpha_u * u_predicted + one_m_alpha * u_old[u_idx];
    }

    // v-predictor
    const int n_v = Nx * (Ny + 1) * Nz;
    #pragma omp target teams distribute parallel for \
        map(present: v_star[0:v_sz], v[0:v_sz], v_old[0:v_sz], conv_v[0:v_sz], diff_v[0:v_sz], \
                     tau_div_v[0:v_sz], a_p_v[0:v_sz], p[0:p_sz])
    for (int idx = 0; idx < n_v; ++idx) {
        int i = idx % Nx + Ng;
        int j = (idx / Nx) % (Ny + 1) + Ng;
        int k = idx / (Nx * (Ny + 1)) + Ng;
        int v_idx = k * v_plane + j * v_stride + i;

        double H_v = -conv_v[v_idx] + diff_v[v_idx] + tau_div_v[v_idx] + fy;
        double dp_dy = (p[k * cell_plane + j * cell_stride + i] -
                        p[k * cell_plane + (j - 1) * cell_stride + i]) / dy;
        double v_predicted = v[v_idx] + (H_v - dp_dy) / a_p_v[v_idx];
        v_star[v_idx] = alpha_u * v_predicted + one_m_alpha * v_old[v_idx];
    }

    // w-predictor
    const int n_w = Nx * Ny * (Nz + 1);
    #pragma omp target teams distribute parallel for \
        map(present: w_star[0:w_sz], w[0:w_sz], w_old[0:w_sz], conv_w[0:w_sz], diff_w[0:w_sz], \
                     tau_div_w[0:w_sz], a_p_w[0:w_sz], p[0:p_sz])
    for (int idx = 0; idx < n_w; ++idx) {
        int i = idx % Nx + Ng;
        int j = (idx / Nx) % Ny + Ng;
        int k = idx / (Nx * Ny) + Ng;
        int w_idx = k * w_plane + j * w_stride + i;

        double H_w = -conv_w[w_idx] + diff_w[w_idx] + tau_div_w[w_idx] + fz;
        double dp_dz = (p[k * cell_plane + j * cell_stride + i] -
                        p[(k - 1) * cell_plane + j * cell_stride + i]) / dz;
        double w_predicted = w[w_idx] + (H_w - dp_dz) / a_p_w[w_idx];
        w_star[w_idx] = alpha_u * w_predicted + one_m_alpha * w_old[w_idx];
    }
}

// ============================================================================
// SIMPLE velocity correction: u = u* - (1/a_P) * grad(p')
// and pressure update: p += alpha_p * p'
// ============================================================================

void simple_correct_velocity_2d(double* u, double* v, double* p,
                                const double* u_star, const double* v_star,
                                const double* p_corr,
                                const double* a_p_u, const double* a_p_v,
                                double alpha_p, double dx, double dy,
                                int Nx, int Ny, int Ng,
                                int u_stride, int v_stride, int cell_stride) {
    [[maybe_unused]] const size_t u_sz = static_cast<size_t>((Ny + 2 * Ng) * u_stride);
    [[maybe_unused]] const size_t v_sz = static_cast<size_t>((Ny + 2 * Ng + 1) * v_stride);
    [[maybe_unused]] const size_t p_sz = static_cast<size_t>((Ny + 2 * Ng) * cell_stride);

    // Correct u
    #pragma omp target teams distribute parallel for collapse(2) \
        map(present: u[0:u_sz], u_star[0:u_sz], p_corr[0:p_sz], a_p_u[0:u_sz])
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i <= Nx; ++i) {
            int jg = j + Ng;
            int ig = i + Ng;
            int u_idx = jg * u_stride + ig;
            double dp_dx = (p_corr[jg * cell_stride + ig] -
                            p_corr[jg * cell_stride + (ig - 1)]) / dx;
            u[u_idx] = u_star[u_idx] - dp_dx / a_p_u[u_idx];
        }
    }

    // Correct v
    #pragma omp target teams distribute parallel for collapse(2) \
        map(present: v[0:v_sz], v_star[0:v_sz], p_corr[0:p_sz], a_p_v[0:v_sz])
    for (int j = 0; j <= Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            int jg = j + Ng;
            int ig = i + Ng;
            int v_idx = jg * v_stride + ig;
            double dp_dy = (p_corr[jg * cell_stride + ig] -
                            p_corr[(jg - 1) * cell_stride + ig]) / dy;
            v[v_idx] = v_star[v_idx] - dp_dy / a_p_v[v_idx];
        }
    }

    // Under-relaxed pressure update
    const int n_cells = Nx * Ny;
    #pragma omp target teams distribute parallel for \
        map(present: p[0:p_sz], p_corr[0:p_sz])
    for (int idx = 0; idx < n_cells; ++idx) {
        int i = idx % Nx + Ng;
        int j = idx / Nx + Ng;
        p[j * cell_stride + i] += alpha_p * p_corr[j * cell_stride + i];
    }
}

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
                                int cell_stride, int cell_plane) {
    [[maybe_unused]] const size_t u_sz = static_cast<size_t>((Nz + 2 * Ng) * u_plane);
    [[maybe_unused]] const size_t v_sz = static_cast<size_t>((Nz + 2 * Ng) * v_plane);
    [[maybe_unused]] const size_t w_sz = static_cast<size_t>((Nz + 2 * Ng + 1) * w_plane);
    [[maybe_unused]] const size_t p_sz = static_cast<size_t>((Nz + 2 * Ng) * cell_plane);

    // Correct u
    const int n_u = (Nx + 1) * Ny * Nz;
    #pragma omp target teams distribute parallel for \
        map(present: u[0:u_sz], u_star[0:u_sz], p_corr[0:p_sz], a_p_u[0:u_sz])
    for (int idx = 0; idx < n_u; ++idx) {
        int i = idx % (Nx + 1) + Ng;
        int j = (idx / (Nx + 1)) % Ny + Ng;
        int k = idx / ((Nx + 1) * Ny) + Ng;
        int u_idx = k * u_plane + j * u_stride + i;
        double dp_dx = (p_corr[k * cell_plane + j * cell_stride + i] -
                        p_corr[k * cell_plane + j * cell_stride + (i - 1)]) / dx;
        u[u_idx] = u_star[u_idx] - dp_dx / a_p_u[u_idx];
    }

    // Correct v
    const int n_v = Nx * (Ny + 1) * Nz;
    #pragma omp target teams distribute parallel for \
        map(present: v[0:v_sz], v_star[0:v_sz], p_corr[0:p_sz], a_p_v[0:v_sz])
    for (int idx = 0; idx < n_v; ++idx) {
        int i = idx % Nx + Ng;
        int j = (idx / Nx) % (Ny + 1) + Ng;
        int k = idx / (Nx * (Ny + 1)) + Ng;
        int v_idx = k * v_plane + j * v_stride + i;
        double dp_dy = (p_corr[k * cell_plane + j * cell_stride + i] -
                        p_corr[k * cell_plane + (j - 1) * cell_stride + i]) / dy;
        v[v_idx] = v_star[v_idx] - dp_dy / a_p_v[v_idx];
    }

    // Correct w
    const int n_w = Nx * Ny * (Nz + 1);
    #pragma omp target teams distribute parallel for \
        map(present: w[0:w_sz], w_star[0:w_sz], p_corr[0:p_sz], a_p_w[0:w_sz])
    for (int idx = 0; idx < n_w; ++idx) {
        int i = idx % Nx + Ng;
        int j = (idx / Nx) % Ny + Ng;
        int k = idx / (Nx * Ny) + Ng;
        int w_idx = k * w_plane + j * w_stride + i;
        double dp_dz = (p_corr[k * cell_plane + j * cell_stride + i] -
                        p_corr[(k - 1) * cell_plane + j * cell_stride + i]) / dz;
        w[w_idx] = w_star[w_idx] - dp_dz / a_p_w[w_idx];
    }

    // Under-relaxed pressure update
    const int n_cells = Nx * Ny * Nz;
    #pragma omp target teams distribute parallel for \
        map(present: p[0:p_sz], p_corr[0:p_sz])
    for (int idx = 0; idx < n_cells; ++idx) {
        int i = idx % Nx + Ng;
        int j = (idx / Nx) % Ny + Ng;
        int k = idx / (Nx * Ny) + Ng;
        p[k * cell_plane + j * cell_stride + i] +=
            alpha_p * p_corr[k * cell_plane + j * cell_stride + i];
    }
}

// ============================================================================
// Jacobi momentum sweep: implicit convection + diffusion
// u_new_P = [sum_nb(a_nb * u_iter_nb) + source - dp/dx * vol] / a_P
// Coefficients use frozen mass flux (from velocity_old), iterate values
// from previous Jacobi sweep. a_P recomputed inline for consistency.
// ============================================================================

void simple_jacobi_momentum_2d(
    double* u_new, double* v_new,
    const double* u_iter, const double* v_iter,
    const double* u_frozen, const double* v_frozen,
    const double* p, const double* nu_eff,
    const double* tau_div_u, const double* tau_div_v,
    double fx, double fy, double dx, double dy,
    double pseudo_dt_inv,
    int Nx, int Ny, int Ng,
    int u_stride, int v_stride, int cell_stride) {

    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double vol = dx * dy;

    [[maybe_unused]] const size_t u_sz = static_cast<size_t>((Ny + 2 * Ng) * u_stride);
    [[maybe_unused]] const size_t v_sz = static_cast<size_t>((Ny + 2 * Ng + 1) * v_stride);
    [[maybe_unused]] const size_t nu_sz = static_cast<size_t>((Ny + 2 * Ng) * cell_stride);
    [[maybe_unused]] const size_t p_sz = nu_sz;

    // u-momentum Jacobi sweep
    #pragma omp target teams distribute parallel for collapse(2) \
        map(present: u_new[0:u_sz], u_iter[0:u_sz], u_frozen[0:u_sz], \
                     v_frozen[0:v_sz], p[0:p_sz], nu_eff[0:nu_sz], tau_div_u[0:u_sz])
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i <= Nx; ++i) {
            int jg = j + Ng;
            int ig = i + Ng;
            int u_idx = jg * u_stride + ig;

            // Cell centers left/right of this u-face
            int cl = jg * cell_stride + (ig - 1);
            int cr = jg * cell_stride + ig;

            // Diffusion: nu at faces of u-CV
            double nu_L = nu_eff[cl];
            double nu_R = nu_eff[cr];
            // Corner-averaged nu for y-direction diffusion
            double nu_S = 0.25 * (nu_eff[cl] + nu_eff[cr]
                + nu_eff[(jg-1) * cell_stride + (ig-1)] + nu_eff[(jg-1) * cell_stride + ig]);
            double nu_N = 0.25 * (nu_eff[cl] + nu_eff[cr]
                + nu_eff[(jg+1) * cell_stride + (ig-1)] + nu_eff[(jg+1) * cell_stride + ig]);

            double a_W_diff = nu_L * inv_dx2 * vol;
            double a_E_diff = nu_R * inv_dx2 * vol;
            double a_S_diff = nu_S * inv_dy2 * vol;
            double a_N_diff = nu_N * inv_dy2 * vol;

            // Convection: mass fluxes from FROZEN field (upwind linearization)
            double F_w = u_frozen[jg * u_stride + (ig - 1)] * dy;
            double F_e = u_frozen[jg * u_stride + (ig + 1)] * dy;
            double F_s = 0.5 * (v_frozen[jg * v_stride + (ig-1)] + v_frozen[jg * v_stride + ig]) * dx;
            double F_n = 0.5 * (v_frozen[(jg+1) * v_stride + (ig-1)] + v_frozen[(jg+1) * v_stride + ig]) * dx;

            // Off-diagonal convective: neighbor contributes when flux flows toward P
            double a_W_conv = (F_w > 0.0) ? F_w : 0.0;
            double a_E_conv = (F_e < 0.0) ? -F_e : 0.0;
            double a_S_conv = (F_s > 0.0) ? F_s : 0.0;
            double a_N_conv = (F_n < 0.0) ? -F_n : 0.0;

            // Total off-diagonal
            double a_W = a_W_diff + a_W_conv;
            double a_E = a_E_diff + a_E_conv;
            double a_S = a_S_diff + a_S_conv;
            double a_N = a_N_diff + a_N_conv;

            // Diagonal: sum of off-diag diffusion + convection self-contribution + damping
            double a_P_val = (a_W_diff + a_E_diff + a_S_diff + a_N_diff)
                + ((F_w < 0.0 ? -F_w : 0.0) + (F_e > 0.0 ? F_e : 0.0)
                 + (F_s < 0.0 ? -F_s : 0.0) + (F_n > 0.0 ? F_n : 0.0))
                + vol * pseudo_dt_inv;
            if (a_P_val < 1e-20) a_P_val = 1e-20;

            // Neighbor sum (using ITERATE values, not frozen)
            double sum_nb = a_W * u_iter[jg * u_stride + (ig-1)]
                          + a_E * u_iter[jg * u_stride + (ig+1)]
                          + a_S * u_iter[(jg-1) * u_stride + ig]
                          + a_N * u_iter[(jg+1) * u_stride + ig];

            double source = (tau_div_u[u_idx] + fx) * vol;
            double dp_dx = (p[cr] - p[cl]) / dx * vol;

            double result = (sum_nb + source - dp_dx) / a_P_val;
            // NaN debug trap (CPU only, remove for production)
            if (result != result) {
                // Print first NaN location
                static bool printed = false;
                if (!printed) {
                    printed = false; // always print for debugging
                    std::fprintf(stderr, "[Jacobi NaN] u at ig=%d jg=%d: sum_nb=%.6e source=%.6e dp=%.6e aP=%.6e\n"
                                         "  a_W=%.4e a_E=%.4e a_S=%.4e a_N=%.4e\n"
                                         "  u_W=%.4e u_E=%.4e u_S=%.4e u_N=%.4e\n"
                                         "  F_w=%.4e F_e=%.4e F_s=%.4e F_n=%.4e\n"
                                         "  nu_L=%.4e nu_R=%.4e nu_S=%.4e nu_N=%.4e\n",
                                 ig, jg, sum_nb, source, dp_dx, a_P_val,
                                 a_W, a_E, a_S, a_N,
                                 u_iter[jg*u_stride+(ig-1)], u_iter[jg*u_stride+(ig+1)],
                                 u_iter[(jg-1)*u_stride+ig], u_iter[(jg+1)*u_stride+ig],
                                 F_w, F_e, F_s, F_n,
                                 nu_L, nu_R, nu_S, nu_N);
                    printed = true;
                }
            }
            u_new[u_idx] = result;
        }
    }

    // v-momentum Jacobi sweep
    #pragma omp target teams distribute parallel for collapse(2) \
        map(present: v_new[0:v_sz], v_iter[0:v_sz], v_frozen[0:v_sz], \
                     u_frozen[0:u_sz], p[0:p_sz], nu_eff[0:nu_sz], tau_div_v[0:v_sz])
    for (int j = 0; j <= Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            int jg = j + Ng;
            int ig = i + Ng;
            int v_idx = jg * v_stride + ig;

            int cb = (jg - 1) * cell_stride + ig;
            int ct = jg * cell_stride + ig;

            double nu_B = nu_eff[cb];
            double nu_T = nu_eff[ct];
            double nu_W = 0.25 * (nu_eff[cb] + nu_eff[ct]
                + nu_eff[(jg-1) * cell_stride + (ig-1)] + nu_eff[jg * cell_stride + (ig-1)]);
            double nu_E = 0.25 * (nu_eff[cb] + nu_eff[ct]
                + nu_eff[(jg-1) * cell_stride + (ig+1)] + nu_eff[jg * cell_stride + (ig+1)]);

            double a_S_diff = nu_B * inv_dy2 * vol;
            double a_N_diff = nu_T * inv_dy2 * vol;
            double a_W_diff = nu_W * inv_dx2 * vol;
            double a_E_diff = nu_E * inv_dx2 * vol;

            // Convection from frozen field
            double F_w = 0.5 * (u_frozen[(jg-1) * u_stride + ig] + u_frozen[jg * u_stride + ig]) * dy;
            double F_e = 0.5 * (u_frozen[(jg-1) * u_stride + (ig+1)] + u_frozen[jg * u_stride + (ig+1)]) * dy;
            double F_s = v_frozen[(jg - 1) * v_stride + ig] * dx;
            double F_n = v_frozen[(jg + 1) * v_stride + ig] * dx;

            double a_W_conv = (F_w > 0.0) ? F_w : 0.0;
            double a_E_conv = (F_e < 0.0) ? -F_e : 0.0;
            double a_S_conv = (F_s > 0.0) ? F_s : 0.0;
            double a_N_conv = (F_n < 0.0) ? -F_n : 0.0;

            double a_W = a_W_diff + a_W_conv;
            double a_E = a_E_diff + a_E_conv;
            double a_S = a_S_diff + a_S_conv;
            double a_N = a_N_diff + a_N_conv;

            double a_P_val = (a_W_diff + a_E_diff + a_S_diff + a_N_diff)
                + ((F_w < 0.0 ? -F_w : 0.0) + (F_e > 0.0 ? F_e : 0.0)
                 + (F_s < 0.0 ? -F_s : 0.0) + (F_n > 0.0 ? F_n : 0.0))
                + vol * pseudo_dt_inv;
            if (a_P_val < 1e-20) a_P_val = 1e-20;

            double sum_nb = a_W * v_iter[jg * v_stride + (ig-1)]
                          + a_E * v_iter[jg * v_stride + (ig+1)]
                          + a_S * v_iter[(jg-1) * v_stride + ig]
                          + a_N * v_iter[(jg+1) * v_stride + ig];

            double source = (tau_div_v[v_idx] + fy) * vol;
            double dp_dy = (p[ct] - p[cb]) / dy * vol;

            v_new[v_idx] = (sum_nb + source - dp_dy) / a_P_val;
        }
    }
}

} // namespace time_kernels
} // namespace nncfd
