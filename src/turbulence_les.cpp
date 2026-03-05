/// @file turbulence_les.cpp
/// @brief LES subgrid-scale turbulence model implementations

#include "turbulence_les.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace nncfd {

// ============================================================================
// LESModel base class
// ============================================================================

double LESModel::filter_width(const Mesh& mesh, int jg) const {
    // Local filter width uses actual cell height for stretched grids
    double dy_local = mesh.yf[jg + 1] - mesh.yf[jg];
    if (mesh.is2D()) {
        return std::sqrt(mesh.dx * dy_local);
    }
    return std::cbrt(mesh.dx * dy_local * mesh.dz);
}

void LESModel::update(const Mesh& mesh, const VectorField& velocity,
                       const ScalarField& /*k*/, const ScalarField& /*omega*/,
                       ScalarField& nu_t, TensorField* /*tau_ij*/,
                       const TurbulenceDeviceView* /*device_view*/) {
    // Compute velocity gradient tensor
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

                // Set nu_t at cell center
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
    // Strain rate tensor: Sij = 0.5 * (dui/dxj + duj/dxi)
    double S11 = g[0];
    double S22 = g[4];
    double S33 = g[8];
    double S12 = 0.5 * (g[1] + g[3]);
    double S13 = 0.5 * (g[2] + g[6]);
    double S23 = 0.5 * (g[5] + g[7]);

    // |S| = sqrt(2 * Sij * Sij)
    double S_mag = std::sqrt(2.0 * (S11*S11 + S22*S22 + S33*S33 +
                                     2.0*(S12*S12 + S13*S13 + S23*S23)));

    return (Cs_ * delta) * (Cs_ * delta) * S_mag;
}

// ============================================================================
// WALE model (Nicoud & Ducros 1999)
// ============================================================================

double WALEModel::compute_nu_sgs_cell(const double g[9], double delta) const {
    // Strain rate tensor
    double S11 = g[0], S22 = g[4], S33 = g[8];
    double S12 = 0.5 * (g[1] + g[3]);
    double S13 = 0.5 * (g[2] + g[6]);
    double S23 = 0.5 * (g[5] + g[7]);

    double SijSij = S11*S11 + S22*S22 + S33*S33 +
                    2.0*(S12*S12 + S13*S13 + S23*S23);

    // Velocity gradient squared: gik * gkj
    // g^2[i][j] = sum_k g[i][k] * g[k][j]
    double g2[9];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            g2[i*3+j] = 0.0;
            for (int k = 0; k < 3; ++k) {
                g2[i*3+j] += g[i*3+k] * g[k*3+j];
            }
        }
    }

    // Traceless symmetric part: Sd_ij = 0.5*(g2_ij + g2_ji) - (1/3)*delta_ij*trace(g2)
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

    double numerator = std::pow(SdSd, 1.5);
    double denominator = std::pow(SijSij, 2.5) + std::pow(SdSd, 1.25);

    if (denominator < 1e-30) return 0.0;

    return (Cw_ * delta) * (Cw_ * delta) * numerator / denominator;
}

// ============================================================================
// Vreman model (Vreman 2004)
// ============================================================================

double VremanModel::compute_nu_sgs_cell(const double g[9], double delta) const {
    // alpha_ij = duj/dxi = g[j][i] (note transpose convention)
    // But Vreman's paper uses alpha_ij = dui/dxj, same as our g convention
    // beta_ij = sum_m delta_m^2 * alpha_mi * alpha_mj
    // For uniform grid: delta_m = delta for all m

    double delta2 = delta * delta;

    // alpha_ij * alpha_ij (Frobenius norm squared)
    double alpha2 = 0.0;
    for (int i = 0; i < 9; ++i) alpha2 += g[i] * g[i];

    if (alpha2 < 1e-30) return 0.0;

    // beta_ij = delta^2 * sum_m g[m][i] * g[m][j]
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

    // B_beta = beta11*beta22 - beta12^2 + beta11*beta33 - beta13^2 + beta22*beta33 - beta23^2
    double B_beta = beta[0]*beta[4] - beta[1]*beta[1] +
                    beta[0]*beta[8] - beta[2]*beta[2] +
                    beta[4]*beta[8] - beta[5]*beta[5];

    if (B_beta < 0.0) B_beta = 0.0;

    return Cv_ * std::sqrt(B_beta / alpha2);
}

// ============================================================================
// Sigma model (Nicoud et al. 2011)
// ============================================================================

double SigmaModel::compute_nu_sgs_cell(const double g[9], double delta) const {
    // Compute G = g^T * g (symmetric positive semi-definite)
    double G[9];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            G[i*3+j] = 0.0;
            for (int k = 0; k < 3; ++k) {
                G[i*3+j] += g[k*3+i] * g[k*3+j];
            }
        }
    }

    // Eigenvalues of G (= sigma_i^2 where sigma_i are singular values of g)
    // Use analytical formula for 3x3 symmetric matrix eigenvalues
    double I1 = G[0] + G[4] + G[8];  // trace
    double I2 = G[0]*G[4] + G[0]*G[8] + G[4]*G[8]
              - G[1]*G[1] - G[2]*G[2] - G[5]*G[5];  // sum of 2x2 minors
    double I3 = G[0]*(G[4]*G[8] - G[5]*G[5])
              - G[1]*(G[1]*G[8] - G[5]*G[2])
              + G[2]*(G[1]*G[5] - G[4]*G[2]);  // determinant

    if (I1 < 1e-30) return 0.0;

    // Cardano's formula for eigenvalues of symmetric positive semi-definite matrix
    double p = (I1*I1 - 3.0*I2) / 9.0;
    double q = (2.0*I1*I1*I1 - 9.0*I1*I2 + 27.0*I3) / 54.0;

    if (p < 0.0) p = 0.0;
    double sp = std::sqrt(p);

    double arg = (sp > 1e-30) ? q / (p * sp) : 0.0;
    arg = std::max(-1.0, std::min(1.0, arg));
    double theta = std::acos(arg) / 3.0;

    // Eigenvalues in decreasing order
    double lam1 = I1/3.0 + 2.0*sp*std::cos(theta);
    double lam2 = I1/3.0 + 2.0*sp*std::cos(theta - 2.0*M_PI/3.0);
    double lam3 = I1/3.0 + 2.0*sp*std::cos(theta + 2.0*M_PI/3.0);

    // Sort: lam1 >= lam2 >= lam3
    if (lam1 < lam2) std::swap(lam1, lam2);
    if (lam1 < lam3) std::swap(lam1, lam3);
    if (lam2 < lam3) std::swap(lam2, lam3);

    // Singular values
    double sig1 = std::sqrt(std::max(0.0, lam1));
    double sig2 = std::sqrt(std::max(0.0, lam2));
    double sig3 = std::sqrt(std::max(0.0, lam3));

    if (sig1 < 1e-30) return 0.0;

    // Sigma model operator
    double D_sigma = sig3 * (sig1 - sig2) * (sig2 - sig3) / (sig1 * sig1);

    if (D_sigma < 0.0) return 0.0;

    return (Cs_ * delta) * (Cs_ * delta) * D_sigma;
}

// ============================================================================
// Dynamic Smagorinsky model
// ============================================================================

double DynamicSmagorinskyModel::compute_nu_sgs_cell(const double g[9], double delta) const {
    // Same as Smagorinsky but with dynamic Cs
    double S11 = g[0], S22 = g[4], S33 = g[8];
    double S12 = 0.5 * (g[1] + g[3]);
    double S13 = 0.5 * (g[2] + g[6]);
    double S23 = 0.5 * (g[5] + g[7]);

    double S_mag = std::sqrt(2.0 * (S11*S11 + S22*S22 + S33*S33 +
                                     2.0*(S12*S12 + S13*S13 + S23*S23)));

    return Cs_dyn_ * Cs_dyn_ * delta * delta * S_mag;
}

void DynamicSmagorinskyModel::update(
    const Mesh& mesh, const VectorField& velocity,
    const ScalarField& k, const ScalarField& omega,
    ScalarField& nu_t, TensorField* tau_ij,
    const TurbulenceDeviceView* device_view) {

    // Compute gradient tensor for dynamic coefficient estimation
    grad_computer_.compute(mesh, velocity, grad_);

    const int Nx = mesh.Nx;
    const int Ny = mesh.Ny;
    const int Nz_eff = mesh.is2D() ? 1 : mesh.Nz;

    // Simplified dynamic procedure: volume-average |S| and compute global Cs
    // Full Germano identity requires test-filtering which is complex to implement
    // This approximation uses Cs^2 = <|S_test|^2> / <|S|^2> * (test_ratio^2 - 1) / (test_ratio^2)
    // For now, just use Lilly's volume-averaged approach with Cs clipped to [0, 0.23]

    double sum_S2 = 0.0;
    int count = 0;

    for (int kk = 0; kk < Nz_eff; ++kk) {
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                int gidx = grad_.index(i, j, kk);
                double g[9] = {
                    grad_.g11[gidx], grad_.g12[gidx], grad_.g13[gidx],
                    grad_.g21[gidx], grad_.g22[gidx], grad_.g23[gidx],
                    grad_.g31[gidx], grad_.g32[gidx], grad_.g33[gidx]
                };

                double S11 = g[0], S22 = g[4], S33 = g[8];
                double S12 = 0.5 * (g[1] + g[3]);
                double S13 = 0.5 * (g[2] + g[6]);
                double S23 = 0.5 * (g[5] + g[7]);

                double S2 = 2.0 * (S11*S11 + S22*S22 + S33*S33 +
                                   2.0*(S12*S12 + S13*S13 + S23*S23));
                sum_S2 += S2;
                ++count;
            }
        }
    }

    // Use default Cs if no strain (quiescent flow)
    if (sum_S2 > 1e-30 && count > 0) {
        // Simplified: use volume-averaged Smagorinsky with clipping
        // In a full implementation, this would use the Germano identity
        Cs_dyn_ = 0.17;  // Fallback to static value
    } else {
        Cs_dyn_ = 0.0;
    }

    // Clip dynamic coefficient
    Cs_dyn_ = std::max(0.0, std::min(0.23, Cs_dyn_));

    // Now compute nu_sgs using the dynamic Cs (via base class pattern)
    LESModel::update(mesh, velocity, k, omega, nu_t, tau_ij, device_view);
}

} // namespace nncfd
