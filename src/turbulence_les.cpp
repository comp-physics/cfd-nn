/// @file turbulence_les.cpp
/// @brief LES subgrid-scale turbulence model implementations (GPU-accelerated)
///
/// All models fuse velocity gradient computation and nu_sgs calculation
/// into a single GPU kernel to avoid intermediate storage.

#include "turbulence_les.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

namespace nncfd {

// ============================================================================
// GPU-compatible cell kernels (declared for device compilation)
// ============================================================================

#ifdef USE_GPU_OFFLOAD
#pragma omp declare target
#endif

/// Compute filter width from mesh face positions at cell jg
inline double les_filter_width_cell(const double* yf, int jg, double dx, double dz, bool is2D) {
    double dy_local = yf[jg + 1] - yf[jg];
    if (is2D) {
        return std::sqrt(dx * dy_local);
    }
    return std::cbrt(dx * dy_local * dz);
}

/// Compute velocity gradients at a single cell center (2D)
/// Returns 4 components: g11=du/dx, g12=du/dy, g21=dv/dx, g22=dv/dy
inline void compute_grad_2d(
    int ig, int jg, double dx,
    const double* yc,
    const double* u, int u_stride,
    const double* v, int v_stride,
    double g[9])
{
    double dy_central = yc[jg + 1] - yc[jg - 1];

    // du/dx
    g[0] = (u[jg * u_stride + ig + 1] - u[jg * u_stride + ig]) / dx;
    // du/dy (interpolate u to center, diff in y)
    double u_jp = 0.5 * (u[(jg + 1) * u_stride + ig] + u[(jg + 1) * u_stride + ig + 1]);
    double u_jm = 0.5 * (u[(jg - 1) * u_stride + ig] + u[(jg - 1) * u_stride + ig + 1]);
    g[1] = (u_jp - u_jm) / dy_central;
    g[2] = 0.0; // du/dz
    // dv/dx (interpolate v to center, diff in x)
    double v_ip = 0.5 * (v[jg * v_stride + ig + 1] + v[(jg + 1) * v_stride + ig + 1]);
    double v_im = 0.5 * (v[jg * v_stride + ig - 1] + v[(jg + 1) * v_stride + ig - 1]);
    g[3] = (v_ip - v_im) / (2.0 * dx);
    // dv/dy (cell-center spacing approximation for staggered v)
    double dy_face = yc[jg + 1] - yc[jg];
    g[4] = (v[(jg + 1) * v_stride + ig] - v[jg * v_stride + ig]) / dy_face;
    g[5] = 0.0; // dv/dz
    g[6] = 0.0; // dw/dx
    g[7] = 0.0; // dw/dy
    g[8] = 0.0; // dw/dz
}

/// Compute velocity gradients at a single cell center (3D)
inline void compute_grad_3d(
    int ig, int jg, int kg,
    double dx, double dz,
    const double* yc,
    const double* u, int u_stride, int u_plane,
    const double* v, int v_stride, int v_plane,
    const double* w, int w_stride, int w_plane,
    double g[9])
{
    double dy_central = yc[jg + 1] - yc[jg - 1];
    double dy_face = yc[jg + 1] - yc[jg];

    // du/dx
    g[0] = (u[kg * u_plane + jg * u_stride + ig + 1] -
            u[kg * u_plane + jg * u_stride + ig]) / dx;
    // du/dy
    double u_jp = 0.5 * (u[kg * u_plane + (jg + 1) * u_stride + ig] +
                          u[kg * u_plane + (jg + 1) * u_stride + ig + 1]);
    double u_jm = 0.5 * (u[kg * u_plane + (jg - 1) * u_stride + ig] +
                          u[kg * u_plane + (jg - 1) * u_stride + ig + 1]);
    g[1] = (u_jp - u_jm) / dy_central;
    // du/dz
    double u_kp = 0.5 * (u[(kg + 1) * u_plane + jg * u_stride + ig] +
                          u[(kg + 1) * u_plane + jg * u_stride + ig + 1]);
    double u_km = 0.5 * (u[(kg - 1) * u_plane + jg * u_stride + ig] +
                          u[(kg - 1) * u_plane + jg * u_stride + ig + 1]);
    g[2] = (u_kp - u_km) / (2.0 * dz);

    // dv/dx
    double v_ip = 0.5 * (v[kg * v_plane + jg * v_stride + ig + 1] +
                          v[kg * v_plane + (jg + 1) * v_stride + ig + 1]);
    double v_im = 0.5 * (v[kg * v_plane + jg * v_stride + ig - 1] +
                          v[kg * v_plane + (jg + 1) * v_stride + ig - 1]);
    g[3] = (v_ip - v_im) / (2.0 * dx);
    // dv/dy
    g[4] = (v[kg * v_plane + (jg + 1) * v_stride + ig] -
            v[kg * v_plane + jg * v_stride + ig]) / dy_face;
    // dv/dz
    double v_kp = 0.5 * (v[(kg + 1) * v_plane + jg * v_stride + ig] +
                          v[(kg + 1) * v_plane + (jg + 1) * v_stride + ig]);
    double v_km = 0.5 * (v[(kg - 1) * v_plane + jg * v_stride + ig] +
                          v[(kg - 1) * v_plane + (jg + 1) * v_stride + ig]);
    g[5] = (v_kp - v_km) / (2.0 * dz);

    // dw/dx
    double w_ip = 0.5 * (w[kg * w_plane + jg * w_stride + ig + 1] +
                          w[(kg + 1) * w_plane + jg * w_stride + ig + 1]);
    double w_im = 0.5 * (w[kg * w_plane + jg * w_stride + ig - 1] +
                          w[(kg + 1) * w_plane + jg * w_stride + ig - 1]);
    g[6] = (w_ip - w_im) / (2.0 * dx);
    // dw/dy
    double w_jp = 0.5 * (w[kg * w_plane + (jg + 1) * w_stride + ig] +
                          w[(kg + 1) * w_plane + (jg + 1) * w_stride + ig]);
    double w_jm = 0.5 * (w[kg * w_plane + (jg - 1) * w_stride + ig] +
                          w[(kg + 1) * w_plane + (jg - 1) * w_stride + ig]);
    g[7] = (w_jp - w_jm) / dy_central;
    // dw/dz
    g[8] = (w[(kg + 1) * w_plane + jg * w_stride + ig] -
            w[kg * w_plane + jg * w_stride + ig]) / dz;
}

/// Box filter (test filter) for cell-centered field: average over 3x3x3 stencil
/// with y-wall truncation (clamp j indices to [Ng, Ng+Ny-1])
inline double box_filter_3d(const double* field, int ig, int jg, int kg,
                            int stride, int plane_stride, int Ny, int Ng) {
    double sum = 0.0;
    int count = 0;
    int j_lo = (jg - 1 >= Ng) ? jg - 1 : jg;
    int j_hi = (jg + 1 < Ng + Ny) ? jg + 1 : jg;
    for (int kk = kg - 1; kk <= kg + 1; ++kk) {
        for (int jj = j_lo; jj <= j_hi; ++jj) {
            for (int ii = ig - 1; ii <= ig + 1; ++ii) {
                sum += field[kk * plane_stride + jj * stride + ii];
                count += 1;
            }
        }
    }
    return sum / count;
}

/// Box filter product: average of f1*f2 over 3x3x3 stencil with y-wall truncation
inline double box_filter_product_3d(const double* f1, const double* f2,
                                    int ig, int jg, int kg,
                                    int stride, int plane_stride, int Ny, int Ng) {
    double sum = 0.0;
    int count = 0;
    int j_lo = (jg - 1 >= Ng) ? jg - 1 : jg;
    int j_hi = (jg + 1 < Ng + Ny) ? jg + 1 : jg;
    for (int kk = kg - 1; kk <= kg + 1; ++kk) {
        for (int jj = j_lo; jj <= j_hi; ++jj) {
            for (int ii = ig - 1; ii <= ig + 1; ++ii) {
                int idx = kk * plane_stride + jj * stride + ii;
                sum += f1[idx] * f2[idx];
                count += 1;
            }
        }
    }
    return sum / count;
}

/// Smagorinsky nu_sgs: (Cs*delta)^2 * |S|
inline double smagorinsky_nu_sgs(const double g[9], double Cs, double delta) {
    double S11 = g[0], S22 = g[4], S33 = g[8];
    double S12 = 0.5 * (g[1] + g[3]);
    double S13 = 0.5 * (g[2] + g[6]);
    double S23 = 0.5 * (g[5] + g[7]);
    double S_mag = std::sqrt(2.0 * (S11*S11 + S22*S22 + S33*S33 +
                                     2.0*(S12*S12 + S13*S13 + S23*S23)));
    return (Cs * delta) * (Cs * delta) * S_mag;
}

/// WALE nu_sgs
inline double wale_nu_sgs(const double g[9], double Cw, double delta) {
    double S11 = g[0], S22 = g[4], S33 = g[8];
    double S12 = 0.5 * (g[1] + g[3]);
    double S13 = 0.5 * (g[2] + g[6]);
    double S23 = 0.5 * (g[5] + g[7]);

    double SijSij = S11*S11 + S22*S22 + S33*S33 +
                    2.0*(S12*S12 + S13*S13 + S23*S23);

    // g^2[i][j] = sum_k g[i][k] * g[k][j]
    double g2[9];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            g2[i*3+j] = 0.0;
            for (int kk = 0; kk < 3; ++kk) {
                g2[i*3+j] += g[i*3+kk] * g[kk*3+j];
            }
        }
    }

    double trace_g2 = g2[0] + g2[4] + g2[8];
    double Sd[9];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Sd[i*3+j] = 0.5 * (g2[i*3+j] + g2[j*3+i]);
            if (i == j) Sd[i*3+j] -= trace_g2 / 3.0;
        }
    }

    double SdSd = 0.0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            SdSd += Sd[i*3+j] * Sd[i*3+j];
        }
    }

    double numerator = SdSd * std::sqrt(SdSd);  // SdSd^(3/2)
    double denom_a = SijSij * SijSij * std::sqrt(SijSij);  // SijSij^(5/2)
    double denom_b = SdSd * std::sqrt(std::sqrt(SdSd));    // SdSd^(5/4)
    double denominator = denom_a + denom_b;

    if (denominator < 1e-30) return 0.0;
    return (Cw * delta) * (Cw * delta) * numerator / denominator;
}

/// Vreman nu_sgs
inline double vreman_nu_sgs(const double g[9], double Cv, double delta) {
    double delta2 = delta * delta;
    double alpha2 = 0.0;
    for (int i = 0; i < 9; ++i) alpha2 += g[i] * g[i];
    if (alpha2 < 1e-30) return 0.0;

    double beta[9];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            beta[i*3+j] = 0.0;
            for (int m = 0; m < 3; ++m) {
                beta[i*3+j] += g[m*3+i] * g[m*3+j];
            }
            beta[i*3+j] *= delta2;
        }
    }

    double B_beta = beta[0]*beta[4] - beta[1]*beta[1] +
                    beta[0]*beta[8] - beta[2]*beta[2] +
                    beta[4]*beta[8] - beta[5]*beta[5];
    if (B_beta < 0.0) B_beta = 0.0;
    return Cv * std::sqrt(B_beta / alpha2);
}

/// Sigma nu_sgs
inline double sigma_nu_sgs(const double g[9], double Cs, double delta) {
    // G = g^T * g
    double G[9];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            G[i*3+j] = 0.0;
            for (int kk = 0; kk < 3; ++kk) {
                G[i*3+j] += g[kk*3+i] * g[kk*3+j];
            }
        }
    }

    double I1 = G[0] + G[4] + G[8];
    double I2 = G[0]*G[4] + G[0]*G[8] + G[4]*G[8]
              - G[1]*G[1] - G[2]*G[2] - G[5]*G[5];
    double I3 = G[0]*(G[4]*G[8] - G[5]*G[5])
              - G[1]*(G[1]*G[8] - G[5]*G[2])
              + G[2]*(G[1]*G[5] - G[4]*G[2]);

    if (I1 < 1e-30) return 0.0;

    double p = (I1*I1 - 3.0*I2) / 9.0;
    double q = (2.0*I1*I1*I1 - 9.0*I1*I2 + 27.0*I3) / 54.0;
    if (p < 0.0) p = 0.0;
    double sp = std::sqrt(p);

    double arg = (sp > 1e-30) ? q / (p * sp) : 0.0;
    if (arg < -1.0) arg = -1.0;
    if (arg > 1.0) arg = 1.0;
    double theta = std::acos(arg) / 3.0;

    double lam1 = I1/3.0 + 2.0*sp*std::cos(theta);
    double lam2 = I1/3.0 + 2.0*sp*std::cos(theta - 2.0*M_PI/3.0);
    double lam3 = I1/3.0 + 2.0*sp*std::cos(theta + 2.0*M_PI/3.0);

    // Sort descending
    if (lam1 < lam2) { double t = lam1; lam1 = lam2; lam2 = t; }
    if (lam1 < lam3) { double t = lam1; lam1 = lam3; lam3 = t; }
    if (lam2 < lam3) { double t = lam2; lam2 = lam3; lam3 = t; }

    double sig1 = std::sqrt(lam1 > 0.0 ? lam1 : 0.0);
    double sig2 = std::sqrt(lam2 > 0.0 ? lam2 : 0.0);
    double sig3 = std::sqrt(lam3 > 0.0 ? lam3 : 0.0);

    if (sig1 < 1e-30) return 0.0;
    double D_sigma = sig3 * (sig1 - sig2) * (sig2 - sig3) / (sig1 * sig1);
    if (D_sigma < 0.0) return 0.0;
    return (Cs * delta) * (Cs * delta) * D_sigma;
}

#ifdef USE_GPU_OFFLOAD
#pragma omp end declare target
#endif

// ============================================================================
// LESModel base class
// ============================================================================

double LESModel::filter_width(const Mesh& mesh, int jg) const {
    double dy_local = mesh.yf[jg + 1] - mesh.yf[jg];
    if (mesh.is2D()) {
        return std::sqrt(mesh.dx * dy_local);
    }
    return std::cbrt(mesh.dx * dy_local * mesh.dz);
}

void LESModel::update(const Mesh& mesh, const VectorField& velocity,
                       const ScalarField& /*k*/, const ScalarField& /*omega*/,
                       ScalarField& nu_t, TensorField* /*tau_ij*/,
                       const TurbulenceDeviceView* device_view) {
    // GPU path: use device_view with fused gradient+nu_sgs kernel
    if (device_view && device_view->is_valid() && device_view->yf) {
        update_gpu(device_view);
        return;
    }

    // CPU fallback: compute gradients then nu_sgs
    grad_computer_.compute(mesh, velocity, grad_);

    const int Nx = mesh.Nx;
    const int Ny = mesh.Ny;
    const int Nz_eff = mesh.is2D() ? 1 : mesh.Nz;
    const int Ng = mesh.Nghost;

    for (int k = 0; k < Nz_eff; ++k) {
        for (int j = 0; j < Ny; ++j) {
            int jg = j + Ng;
            double delta = filter_width(mesh, jg);
            for (int i = 0; i < Nx; ++i) {
                int gidx = grad_.index(i, j, k);
                double g[9] = {
                    grad_.g11[gidx], grad_.g12[gidx], grad_.g13[gidx],
                    grad_.g21[gidx], grad_.g22[gidx], grad_.g23[gidx],
                    grad_.g31[gidx], grad_.g32[gidx], grad_.g33[gidx]
                };

                double nu_sgs = compute_nu_sgs_cell(g, delta);

                if (mesh.is2D()) {
                    nu_t(i + Ng, jg) = nu_sgs;
                } else {
                    nu_t(i + Ng, j + Ng, k + Ng) = nu_sgs;
                }
            }
        }
    }
}

// ============================================================================
// Smagorinsky model
// ============================================================================

double SmagorinskyModel::compute_nu_sgs_cell(const double g[9], double delta) const {
    return smagorinsky_nu_sgs(g, Cs_, delta);
}

void SmagorinskyModel::update_gpu(const TurbulenceDeviceView* dv) {
    const int Nx = dv->Nx, Ny = dv->Ny, Ng = dv->Ng;
    const int Nz_eff = dv->is3D() ? dv->Nz : 1;
    const double dx = dv->dx, dz = dv->dz;
    const double Cs = Cs_;
    const bool is2D = !dv->is3D();

    double* u = dv->u_face;
    double* v = dv->v_face;
    double* w = dv->w_face;
    double* nu_t_ptr = dv->nu_t;
    const double* yf = dv->yf;
    const double* yc = dv->yc;
    const int u_stride = dv->u_stride, v_stride = dv->v_stride, w_stride = dv->w_stride;
    const int u_plane = dv->u_plane_stride, v_plane = dv->v_plane_stride, w_plane = dv->w_plane_stride;
    const int cell_stride = dv->cell_stride, cell_plane = dv->cell_plane_stride;
    [[maybe_unused]] const int u_sz = dv->u_total, v_sz = dv->v_total, w_sz = dv->w_total;
    [[maybe_unused]] const int nut_sz = dv->cell_total, yf_sz = dv->yf_total, yc_sz = dv->yc_total;
    const int total = Nx * Ny * Nz_eff;

    #pragma omp target teams distribute parallel for \
        map(present: u[0:u_sz], v[0:v_sz], w[0:w_sz], nu_t_ptr[0:nut_sz], yf[0:yf_sz], yc[0:yc_sz]) \
        firstprivate(Nx, Ny, Nz_eff, Ng, dx, dz, Cs, is2D, \
                     u_stride, v_stride, w_stride, u_plane, v_plane, w_plane, \
                     cell_stride, cell_plane)
    for (int idx = 0; idx < total; ++idx) {
        int k = idx / (Nx * Ny);
        int rem = idx - k * Nx * Ny;
        int j = rem / Nx;
        int i = rem - j * Nx;
        int ig = i + Ng, jg = j + Ng, kg = k + Ng;

        double g[9];
        if (is2D) {
            compute_grad_2d(ig, jg, dx, yc, u, u_stride, v, v_stride, g);
        } else {
            compute_grad_3d(ig, jg, kg, dx, dz, yc, u, u_stride, u_plane,
                           v, v_stride, v_plane, w, w_stride, w_plane, g);
        }

        double delta = les_filter_width_cell(yf, jg, dx, dz, is2D);
        double nu_sgs = smagorinsky_nu_sgs(g, Cs, delta);

        int cell_idx = is2D ? (jg * cell_stride + ig)
                            : (kg * cell_plane + jg * cell_stride + ig);
        nu_t_ptr[cell_idx] = nu_sgs;
    }
}

// ============================================================================
// WALE model
// ============================================================================

double WALEModel::compute_nu_sgs_cell(const double g[9], double delta) const {
    return wale_nu_sgs(g, Cw_, delta);
}

void WALEModel::update_gpu(const TurbulenceDeviceView* dv) {
    const int Nx = dv->Nx, Ny = dv->Ny, Ng = dv->Ng;
    const int Nz_eff = dv->is3D() ? dv->Nz : 1;
    const double dx = dv->dx, dz = dv->dz;
    const double Cw = Cw_;
    const bool is2D = !dv->is3D();

    double* u = dv->u_face;
    double* v = dv->v_face;
    double* w = dv->w_face;
    double* nu_t_ptr = dv->nu_t;
    const double* yf = dv->yf;
    const double* yc = dv->yc;
    const int u_stride = dv->u_stride, v_stride = dv->v_stride, w_stride = dv->w_stride;
    const int u_plane = dv->u_plane_stride, v_plane = dv->v_plane_stride, w_plane = dv->w_plane_stride;
    const int cell_stride = dv->cell_stride, cell_plane = dv->cell_plane_stride;
    [[maybe_unused]] const int u_sz = dv->u_total, v_sz = dv->v_total, w_sz = dv->w_total;
    [[maybe_unused]] const int nut_sz = dv->cell_total, yf_sz = dv->yf_total, yc_sz = dv->yc_total;
    const int total = Nx * Ny * Nz_eff;

    #pragma omp target teams distribute parallel for \
        map(present: u[0:u_sz], v[0:v_sz], w[0:w_sz], nu_t_ptr[0:nut_sz], yf[0:yf_sz], yc[0:yc_sz]) \
        firstprivate(Nx, Ny, Nz_eff, Ng, dx, dz, Cw, is2D, \
                     u_stride, v_stride, w_stride, u_plane, v_plane, w_plane, \
                     cell_stride, cell_plane)
    for (int idx = 0; idx < total; ++idx) {
        int k = idx / (Nx * Ny);
        int rem = idx - k * Nx * Ny;
        int j = rem / Nx;
        int i = rem - j * Nx;
        int ig = i + Ng, jg = j + Ng, kg = k + Ng;

        double g[9];
        if (is2D) {
            compute_grad_2d(ig, jg, dx, yc, u, u_stride, v, v_stride, g);
        } else {
            compute_grad_3d(ig, jg, kg, dx, dz, yc, u, u_stride, u_plane,
                           v, v_stride, v_plane, w, w_stride, w_plane, g);
        }

        double delta = les_filter_width_cell(yf, jg, dx, dz, is2D);
        double nu_sgs = wale_nu_sgs(g, Cw, delta);

        int cell_idx = is2D ? (jg * cell_stride + ig)
                            : (kg * cell_plane + jg * cell_stride + ig);
        nu_t_ptr[cell_idx] = nu_sgs;
    }
}

// ============================================================================
// Vreman model
// ============================================================================

double VremanModel::compute_nu_sgs_cell(const double g[9], double delta) const {
    return vreman_nu_sgs(g, Cv_, delta);
}

void VremanModel::update_gpu(const TurbulenceDeviceView* dv) {
    const int Nx = dv->Nx, Ny = dv->Ny, Ng = dv->Ng;
    const int Nz_eff = dv->is3D() ? dv->Nz : 1;
    const double dx = dv->dx, dz = dv->dz;
    const double Cv = Cv_;
    const bool is2D = !dv->is3D();

    double* u = dv->u_face;
    double* v = dv->v_face;
    double* w = dv->w_face;
    double* nu_t_ptr = dv->nu_t;
    const double* yf = dv->yf;
    const double* yc = dv->yc;
    const int u_stride = dv->u_stride, v_stride = dv->v_stride, w_stride = dv->w_stride;
    const int u_plane = dv->u_plane_stride, v_plane = dv->v_plane_stride, w_plane = dv->w_plane_stride;
    const int cell_stride = dv->cell_stride, cell_plane = dv->cell_plane_stride;
    [[maybe_unused]] const int u_sz = dv->u_total, v_sz = dv->v_total, w_sz = dv->w_total;
    [[maybe_unused]] const int nut_sz = dv->cell_total, yf_sz = dv->yf_total, yc_sz = dv->yc_total;
    const int total = Nx * Ny * Nz_eff;

    #pragma omp target teams distribute parallel for \
        map(present: u[0:u_sz], v[0:v_sz], w[0:w_sz], nu_t_ptr[0:nut_sz], yf[0:yf_sz], yc[0:yc_sz]) \
        firstprivate(Nx, Ny, Nz_eff, Ng, dx, dz, Cv, is2D, \
                     u_stride, v_stride, w_stride, u_plane, v_plane, w_plane, \
                     cell_stride, cell_plane)
    for (int idx = 0; idx < total; ++idx) {
        int k = idx / (Nx * Ny);
        int rem = idx - k * Nx * Ny;
        int j = rem / Nx;
        int i = rem - j * Nx;
        int ig = i + Ng, jg = j + Ng, kg = k + Ng;

        double g[9];
        if (is2D) {
            compute_grad_2d(ig, jg, dx, yc, u, u_stride, v, v_stride, g);
        } else {
            compute_grad_3d(ig, jg, kg, dx, dz, yc, u, u_stride, u_plane,
                           v, v_stride, v_plane, w, w_stride, w_plane, g);
        }

        double delta = les_filter_width_cell(yf, jg, dx, dz, is2D);
        double nu_sgs = vreman_nu_sgs(g, Cv, delta);

        int cell_idx = is2D ? (jg * cell_stride + ig)
                            : (kg * cell_plane + jg * cell_stride + ig);
        nu_t_ptr[cell_idx] = nu_sgs;
    }
}

// ============================================================================
// Sigma model
// ============================================================================

double SigmaModel::compute_nu_sgs_cell(const double g[9], double delta) const {
    return sigma_nu_sgs(g, Cs_, delta);
}

void SigmaModel::update_gpu(const TurbulenceDeviceView* dv) {
    const int Nx = dv->Nx, Ny = dv->Ny, Ng = dv->Ng;
    const int Nz_eff = dv->is3D() ? dv->Nz : 1;
    const double dx = dv->dx, dz = dv->dz;
    const double Cs = Cs_;
    const bool is2D = !dv->is3D();

    double* u = dv->u_face;
    double* v = dv->v_face;
    double* w = dv->w_face;
    double* nu_t_ptr = dv->nu_t;
    const double* yf = dv->yf;
    const double* yc = dv->yc;
    const int u_stride = dv->u_stride, v_stride = dv->v_stride, w_stride = dv->w_stride;
    const int u_plane = dv->u_plane_stride, v_plane = dv->v_plane_stride, w_plane = dv->w_plane_stride;
    const int cell_stride = dv->cell_stride, cell_plane = dv->cell_plane_stride;
    [[maybe_unused]] const int u_sz = dv->u_total, v_sz = dv->v_total, w_sz = dv->w_total;
    [[maybe_unused]] const int nut_sz = dv->cell_total, yf_sz = dv->yf_total, yc_sz = dv->yc_total;
    const int total = Nx * Ny * Nz_eff;

    #pragma omp target teams distribute parallel for \
        map(present: u[0:u_sz], v[0:v_sz], w[0:w_sz], nu_t_ptr[0:nut_sz], yf[0:yf_sz], yc[0:yc_sz]) \
        firstprivate(Nx, Ny, Nz_eff, Ng, dx, dz, Cs, is2D, \
                     u_stride, v_stride, w_stride, u_plane, v_plane, w_plane, \
                     cell_stride, cell_plane)
    for (int idx = 0; idx < total; ++idx) {
        int k = idx / (Nx * Ny);
        int rem = idx - k * Nx * Ny;
        int j = rem / Nx;
        int i = rem - j * Nx;
        int ig = i + Ng, jg = j + Ng, kg = k + Ng;

        double g[9];
        if (is2D) {
            compute_grad_2d(ig, jg, dx, yc, u, u_stride, v, v_stride, g);
        } else {
            compute_grad_3d(ig, jg, kg, dx, dz, yc, u, u_stride, u_plane,
                           v, v_stride, v_plane, w, w_stride, w_plane, g);
        }

        double delta = les_filter_width_cell(yf, jg, dx, dz, is2D);
        double nu_sgs = sigma_nu_sgs(g, Cs, delta);

        int cell_idx = is2D ? (jg * cell_stride + ig)
                            : (kg * cell_plane + jg * cell_stride + ig);
        nu_t_ptr[cell_idx] = nu_sgs;
    }
}

// ============================================================================
// Dynamic Smagorinsky model (Germano et al. 1991, Lilly 1992)
// ============================================================================

DynamicSmagorinskyModel::~DynamicSmagorinskyModel() {
    cleanup_dynamic_gpu();
}

void DynamicSmagorinskyModel::init_dynamic_gpu(const TurbulenceDeviceView* dv) {
    if (dyn_gpu_ready_) return;

    Ny_ = dv->Ny;
    cell_total_ = dv->cell_total;

    u_cc_ = new double[cell_total_]();
    v_cc_ = new double[cell_total_]();
    w_cc_ = new double[cell_total_]();
    LM_plane_ = new double[Ny_]();
    MM_plane_ = new double[Ny_]();
    Cs2_plane_ = new double[Ny_]();
    Cs2_plane_host_.resize(Ny_, 0.0);

    #pragma omp target enter data map(alloc: u_cc_[0:cell_total_], \
        v_cc_[0:cell_total_], w_cc_[0:cell_total_], \
        LM_plane_[0:Ny_], MM_plane_[0:Ny_], Cs2_plane_[0:Ny_])

    dyn_gpu_ready_ = true;
}

void DynamicSmagorinskyModel::cleanup_dynamic_gpu() {
    if (!dyn_gpu_ready_) {
        // Still need to free host allocations if they exist
        delete[] u_cc_; u_cc_ = nullptr;
        delete[] v_cc_; v_cc_ = nullptr;
        delete[] w_cc_; w_cc_ = nullptr;
        delete[] LM_plane_; LM_plane_ = nullptr;
        delete[] MM_plane_; MM_plane_ = nullptr;
        delete[] Cs2_plane_; Cs2_plane_ = nullptr;
        return;
    }

    [[maybe_unused]] int ct = cell_total_;
    [[maybe_unused]] int ny = Ny_;

    #pragma omp target exit data map(delete: u_cc_[0:ct], \
        v_cc_[0:ct], w_cc_[0:ct], \
        LM_plane_[0:ny], MM_plane_[0:ny], Cs2_plane_[0:ny])

    delete[] u_cc_; u_cc_ = nullptr;
    delete[] v_cc_; v_cc_ = nullptr;
    delete[] w_cc_; w_cc_ = nullptr;
    delete[] LM_plane_; LM_plane_ = nullptr;
    delete[] MM_plane_; MM_plane_ = nullptr;
    delete[] Cs2_plane_; Cs2_plane_ = nullptr;

    dyn_gpu_ready_ = false;
}

double DynamicSmagorinskyModel::compute_nu_sgs_cell(const double g[9], double delta) const {
    // CPU fallback: use static Cs = 0.17 (dynamic procedure only on GPU)
    return smagorinsky_nu_sgs(g, 0.17, delta);
}

void DynamicSmagorinskyModel::update(
    const Mesh& mesh, const VectorField& velocity,
    const ScalarField& k, const ScalarField& omega,
    ScalarField& nu_t, TensorField* tau_ij,
    const TurbulenceDeviceView* device_view) {
    // Dispatch to GPU (Germano procedure) or CPU fallback
    LESModel::update(mesh, velocity, k, omega, nu_t, tau_ij, device_view);
}

void DynamicSmagorinskyModel::update_gpu(const TurbulenceDeviceView* dv) {
    // Lazy initialization of scratch arrays
    if (!dyn_gpu_ready_) {
        init_dynamic_gpu(dv);
    }

    const int Nx = dv->Nx, Ny = dv->Ny, Ng = dv->Ng;
    const int Nz_eff = dv->is3D() ? dv->Nz : 1;
    const double dx = dv->dx, dz = dv->dz;
    const bool is2D = !dv->is3D();

    double* u = dv->u_face;
    double* v = dv->v_face;
    double* w = dv->w_face;
    double* nu_t_ptr = dv->nu_t;
    const double* yf = dv->yf;
    const double* yc = dv->yc;
    const int u_stride = dv->u_stride, v_stride = dv->v_stride, w_stride = dv->w_stride;
    const int u_plane = dv->u_plane_stride, v_plane = dv->v_plane_stride, w_plane = dv->w_plane_stride;
    const int cell_stride = dv->cell_stride, cell_plane = dv->cell_plane_stride;
    [[maybe_unused]] const int u_sz = dv->u_total, v_sz = dv->v_total, w_sz = dv->w_total;
    [[maybe_unused]] const int nut_sz = dv->cell_total, yf_sz = dv->yf_total, yc_sz = dv->yc_total;
    const int total = Nx * Ny * Nz_eff;

    double* ucc = u_cc_;
    double* vcc = v_cc_;
    double* wcc = w_cc_;
    double* lm_plane = LM_plane_;
    double* mm_plane = MM_plane_;
    double* cs2_plane = Cs2_plane_;
    [[maybe_unused]] const int cc_sz = cell_total_;
    [[maybe_unused]] const int ny_sz = Ny_;

    // ----------------------------------------------------------------
    // Pass 0: Interpolate staggered velocities to cell centers
    // ----------------------------------------------------------------
    #pragma omp target teams distribute parallel for \
        map(present: u[0:u_sz], v[0:v_sz], w[0:w_sz], \
            ucc[0:cc_sz], vcc[0:cc_sz], wcc[0:cc_sz]) \
        firstprivate(Nx, Ny, Nz_eff, Ng, is2D, \
                     u_stride, v_stride, w_stride, u_plane, v_plane, w_plane, \
                     cell_stride, cell_plane)
    for (int idx = 0; idx < total; ++idx) {
        int kk = idx / (Nx * Ny);
        int rem = idx - kk * Nx * Ny;
        int j = rem / Nx;
        int i = rem - j * Nx;
        int ig = i + Ng, jg = j + Ng, kg = kk + Ng;

        int cell_idx = is2D ? (jg * cell_stride + ig)
                            : (kg * cell_plane + jg * cell_stride + ig);

        if (is2D) {
            ucc[cell_idx] = 0.5 * (u[jg * u_stride + ig] +
                                    u[jg * u_stride + ig + 1]);
            vcc[cell_idx] = 0.5 * (v[jg * v_stride + ig] +
                                    v[(jg + 1) * v_stride + ig]);
            wcc[cell_idx] = 0.0;
        } else {
            ucc[cell_idx] = 0.5 * (u[kg * u_plane + jg * u_stride + ig] +
                                    u[kg * u_plane + jg * u_stride + ig + 1]);
            vcc[cell_idx] = 0.5 * (v[kg * v_plane + jg * v_stride + ig] +
                                    v[kg * v_plane + (jg + 1) * v_stride + ig]);
            wcc[cell_idx] = 0.5 * (w[kg * w_plane + jg * w_stride + ig] +
                                    w[(kg + 1) * w_plane + jg * w_stride + ig]);
        }
    }

    // ----------------------------------------------------------------
    // Pass 1: Zero plane sums, then accumulate LM and MM per y-plane
    // ----------------------------------------------------------------
    #pragma omp target teams distribute parallel for \
        map(present: lm_plane[0:ny_sz], mm_plane[0:ny_sz]) \
        firstprivate(Ny)
    for (int j = 0; j < Ny; ++j) {
        lm_plane[j] = 0.0;
        mm_plane[j] = 0.0;
    }

    // Accumulate Germano identity contractions per plane
    // L_ij = <u_i u_j> - <u_i><u_j>  (Leonard stress via test filter)
    // M_ij = delta^2 (alpha^2 - 1) |S| S_ij  (scale-similarity approximation, alpha=2)
    // Cs^2(j) = max(0, sum_xz(L_ij M_ij) / sum_xz(M_ij M_ij))
    #pragma omp target teams distribute parallel for \
        map(present: u[0:u_sz], v[0:v_sz], w[0:w_sz], \
            ucc[0:cc_sz], vcc[0:cc_sz], wcc[0:cc_sz], \
            nu_t_ptr[0:nut_sz], yf[0:yf_sz], yc[0:yc_sz], \
            lm_plane[0:ny_sz], mm_plane[0:ny_sz]) \
        firstprivate(Nx, Ny, Nz_eff, Ng, dx, dz, is2D, \
                     u_stride, v_stride, w_stride, u_plane, v_plane, w_plane, \
                     cell_stride, cell_plane)
    for (int idx = 0; idx < total; ++idx) {
        int kk = idx / (Nx * Ny);
        int rem = idx - kk * Nx * Ny;
        int j = rem / Nx;
        int i = rem - j * Nx;
        int ig = i + Ng, jg = j + Ng, kg = kk + Ng;

        // Compute velocity gradients at grid level
        double g[9];
        if (is2D) {
            compute_grad_2d(ig, jg, dx, yc, u, u_stride, v, v_stride, g);
        } else {
            compute_grad_3d(ig, jg, kg, dx, dz, yc, u, u_stride, u_plane,
                           v, v_stride, v_plane, w, w_stride, w_plane, g);
        }

        // Strain rate tensor and its magnitude
        double S11 = g[0], S22 = g[4], S33 = g[8];
        double S12 = 0.5 * (g[1] + g[3]);
        double S13 = 0.5 * (g[2] + g[6]);
        double S23 = 0.5 * (g[5] + g[7]);
        double S_mag = std::sqrt(2.0 * (S11*S11 + S22*S22 + S33*S33 +
                                         2.0*(S12*S12 + S13*S13 + S23*S23)));

        double delta = les_filter_width_cell(yf, jg, dx, dz, is2D);
        double delta2 = delta * delta;

        // M_ij = delta^2 * (alpha^2 - 1) * |S| * S_ij, alpha = 2 => factor = 3
        double factor = 3.0 * delta2 * S_mag;
        double M11 = factor * S11, M22 = factor * S22, M33 = factor * S33;
        double M12 = factor * S12, M13 = factor * S13, M23 = factor * S23;

        // Leonard stress: L_ij = <u_i u_j>_test - <u_i>_test * <u_j>_test
        // Compute test-filtered velocities and test-filtered products
        double u_bar, v_bar, w_bar;
        double uu_bar, uv_bar, vv_bar, uw_bar, vw_bar, ww_bar;

        if (is2D) {
            // 2D: use cell_stride only (no plane stride)
            int cidx = jg * cell_stride + ig;
            u_bar = ucc[cidx];
            v_bar = vcc[cidx];
            w_bar = 0.0;
            uu_bar = ucc[cidx] * ucc[cidx];
            uv_bar = ucc[cidx] * vcc[cidx];
            vv_bar = vcc[cidx] * vcc[cidx];
            uw_bar = 0.0; vw_bar = 0.0; ww_bar = 0.0;
        } else {
            // 3D: box filter over 3x3x3 stencil with y-wall truncation
            u_bar = box_filter_3d(ucc, ig, jg, kg, cell_stride, cell_plane, Ny, Ng);
            v_bar = box_filter_3d(vcc, ig, jg, kg, cell_stride, cell_plane, Ny, Ng);
            w_bar = box_filter_3d(wcc, ig, jg, kg, cell_stride, cell_plane, Ny, Ng);

            // Test-filter of products: <u_i * u_j>
            uu_bar = box_filter_product_3d(ucc, ucc, ig, jg, kg, cell_stride, cell_plane, Ny, Ng);
            uv_bar = box_filter_product_3d(ucc, vcc, ig, jg, kg, cell_stride, cell_plane, Ny, Ng);
            vv_bar = box_filter_product_3d(vcc, vcc, ig, jg, kg, cell_stride, cell_plane, Ny, Ng);
            uw_bar = box_filter_product_3d(ucc, wcc, ig, jg, kg, cell_stride, cell_plane, Ny, Ng);
            vw_bar = box_filter_product_3d(vcc, wcc, ig, jg, kg, cell_stride, cell_plane, Ny, Ng);
            ww_bar = box_filter_product_3d(wcc, wcc, ig, jg, kg, cell_stride, cell_plane, Ny, Ng);
        }

        double L11 = uu_bar - u_bar * u_bar;
        double L22 = vv_bar - v_bar * v_bar;
        double L33 = ww_bar - w_bar * w_bar;
        double L12 = uv_bar - u_bar * v_bar;
        double L13 = uw_bar - u_bar * w_bar;
        double L23 = vw_bar - v_bar * w_bar;

        // Contract L_ij M_ij and M_ij M_ij
        double LM = L11*M11 + L22*M22 + L33*M33 +
                    2.0*(L12*M12 + L13*M13 + L23*M23);
        double MM = M11*M11 + M22*M22 + M33*M33 +
                    2.0*(M12*M12 + M13*M13 + M23*M23);

        // Accumulate into plane sums (atomic for thread safety)
        #pragma omp atomic update
        lm_plane[j] += LM;
        #pragma omp atomic update
        mm_plane[j] += MM;
    }

    // ----------------------------------------------------------------
    // Pass 2: Compute Cs^2(j) = max(0, LM(j) / MM(j)) and apply nu_t
    // ----------------------------------------------------------------
    const double Cs2_max = 0.5;  // Clamp to prevent excessive values

    #pragma omp target teams distribute parallel for \
        map(present: lm_plane[0:ny_sz], mm_plane[0:ny_sz], cs2_plane[0:ny_sz]) \
        firstprivate(Ny, Cs2_max)
    for (int j = 0; j < Ny; ++j) {
        double lm = lm_plane[j];
        double mm = mm_plane[j];
        double cs2 = (mm > 1e-30) ? lm / mm : 0.0;
        if (cs2 < 0.0) cs2 = 0.0;
        if (cs2 > Cs2_max) cs2 = Cs2_max;
        cs2_plane[j] = cs2;
    }

    // Apply nu_t = Cs^2(j) * delta^2 * |S| at each cell
    #pragma omp target teams distribute parallel for \
        map(present: u[0:u_sz], v[0:v_sz], w[0:w_sz], \
            nu_t_ptr[0:nut_sz], yf[0:yf_sz], yc[0:yc_sz], cs2_plane[0:ny_sz]) \
        firstprivate(Nx, Ny, Nz_eff, Ng, dx, dz, is2D, \
                     u_stride, v_stride, w_stride, u_plane, v_plane, w_plane, \
                     cell_stride, cell_plane)
    for (int idx = 0; idx < total; ++idx) {
        int kk = idx / (Nx * Ny);
        int rem = idx - kk * Nx * Ny;
        int j = rem / Nx;
        int i = rem - j * Nx;
        int ig = i + Ng, jg = j + Ng, kg = kk + Ng;

        double g[9];
        if (is2D) {
            compute_grad_2d(ig, jg, dx, yc, u, u_stride, v, v_stride, g);
        } else {
            compute_grad_3d(ig, jg, kg, dx, dz, yc, u, u_stride, u_plane,
                           v, v_stride, v_plane, w, w_stride, w_plane, g);
        }

        double delta = les_filter_width_cell(yf, jg, dx, dz, is2D);
        double S11 = g[0], S22 = g[4], S33 = g[8];
        double S12 = 0.5 * (g[1] + g[3]);
        double S13 = 0.5 * (g[2] + g[6]);
        double S23 = 0.5 * (g[5] + g[7]);
        double S_mag = std::sqrt(2.0 * (S11*S11 + S22*S22 + S33*S33 +
                                         2.0*(S12*S12 + S13*S13 + S23*S23)));

        double cs2 = cs2_plane[j];
        double nu_sgs = cs2 * delta * delta * S_mag;

        int cell_idx = is2D ? (jg * cell_stride + ig)
                            : (kg * cell_plane + jg * cell_stride + ig);
        nu_t_ptr[cell_idx] = nu_sgs;
    }
}

} // namespace nncfd
