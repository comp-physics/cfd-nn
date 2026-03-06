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
// Dynamic Smagorinsky model
// ============================================================================

double DynamicSmagorinskyModel::compute_nu_sgs_cell(const double g[9], double delta) const {
    return smagorinsky_nu_sgs(g, Cs_dyn_, delta);
}

void DynamicSmagorinskyModel::update(
    const Mesh& mesh, const VectorField& velocity,
    const ScalarField& k, const ScalarField& omega,
    ScalarField& nu_t, TensorField* tau_ij,
    const TurbulenceDeviceView* device_view) {

    static bool warned = false;
    if (!warned) {
        std::cerr << "[LES] DynamicSmagorinsky: using simplified procedure (Cs=0.17 fallback)\n";
        warned = true;
    }

    // Dynamic coefficient: simplified fallback (always Cs=0.17)
    Cs_dyn_ = 0.17;

    // Use the base class update which handles GPU/CPU dispatch
    LESModel::update(mesh, velocity, k, omega, nu_t, tau_ij, device_view);
}

void DynamicSmagorinskyModel::update_gpu(const TurbulenceDeviceView* dv) {
    // Uses Smagorinsky kernel with dynamic Cs
    const int Nx = dv->Nx, Ny = dv->Ny, Ng = dv->Ng;
    const int Nz_eff = dv->is3D() ? dv->Nz : 1;
    const double dx = dv->dx, dz = dv->dz;
    const double Cs = Cs_dyn_;
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

} // namespace nncfd
