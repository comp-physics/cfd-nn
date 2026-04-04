/// @file turbulence_rsm.cpp
/// @brief Reynolds Stress Model (SSG pressure-strain + omega equation)
///
/// Implements a full 7-equation RSM:
/// - 6 Reynolds stress components R_ij with SSG pressure-strain correlation
/// - omega specific dissipation rate equation
/// - Point-implicit time integration for stability
/// - Realizability enforcement
///
/// References:
/// - Speziale, Sarkar, Gatski (1991) "Modelling the pressure-strain
///   correlation of turbulence: an invariant dynamical systems approach"
/// - Wilcox (2006) "Turbulence Modeling for CFD" — stress-omega model

#include "turbulence_rsm.hpp"
#include "gpu_kernels.hpp"
#include "gpu_kernels.hpp"
#include "timing.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

namespace nncfd {

// ============================================================================
// RSM Cell Kernel — free function for GPU compatibility (nvc++ workaround)
// ============================================================================

#ifdef USE_GPU_OFFLOAD
#pragma omp declare target
#endif

/// Advance Reynolds stresses and omega for a single cell
/// All data passed as arguments (no `this` pointer captured)
inline void rsm_cell_kernel(
    // Cell indices (cell-centered, with ghost layer offset)
    int idx_c, int idx_ip, int idx_im, int idx_jp, int idx_jm,
    // 3D neighbor indices (idx_kp, idx_km for z-neighbors)
    int idx_kp, int idx_km,
    // Velocity gradients (9 components, cell-centered)
    const double* dudx_ptr, const double* dudy_ptr, const double* dudz_ptr,
    const double* dvdx_ptr, const double* dvdy_ptr, const double* dvdz_ptr,
    const double* dwdx_ptr, const double* dwdy_ptr, const double* dwdz_ptr,
    // Current Reynolds stresses
    const double* Rxx, const double* Ryy, const double* Rzz,
    const double* Rxy, const double* Rxz, const double* Ryz,
    // Current omega
    const double* omega_ptr,
    // Previous nu_t for diffusion
    const double* nu_t_ptr,
    // Grid parameters
    [[maybe_unused]] double dx, [[maybe_unused]] double dy, [[maybe_unused]] double dz, double dt,
    double inv_dx2, double inv_dy2, double inv_dz2,
    bool is3D,
    // Model constants
    double nu, double C1, double C1s, double C2, double C3, double C3s,
    double C4, double C5,
    double alpha_omega, double beta_omega, double beta_star, double sigma_omega,
    double sigma_R,
    double k_min, double k_max, double omega_min, double omega_max,
    // Outputs (6 R_ij + omega)
    double& Rxx_new, double& Ryy_new, double& Rzz_new,
    double& Rxy_new, double& Rxz_new, double& Ryz_new,
    double& omega_new_out)
{
    // Read current values
    double rxx = Rxx[idx_c], ryy = Ryy[idx_c], rzz = Rzz[idx_c];
    double rxy = Rxy[idx_c], rxz = Rxz[idx_c], ryz = Ryz[idx_c];
    double om = omega_ptr[idx_c];
    double nu_t_c = nu_t_ptr[idx_c];

    // Enforce positivity on diagonal entries
    rxx = (rxx > k_min) ? rxx : k_min;
    ryy = (ryy > k_min) ? ryy : k_min;
    rzz = (rzz > k_min) ? rzz : k_min;
    om  = (om > omega_min) ? om : omega_min;
    nu_t_c = (nu_t_c > 0.0) ? nu_t_c : 0.0;

    // TKE from trace
    double k = 0.5 * (rxx + ryy + rzz);
    k = (k > k_min) ? k : k_min;

    // Dissipation rate
    double eps = beta_star * k * om;

    // ---- Velocity gradients ----
    double dUdx = dudx_ptr[idx_c], dUdy = dudy_ptr[idx_c], dUdz = dudz_ptr[idx_c];
    double dVdx = dvdx_ptr[idx_c], dVdy = dvdy_ptr[idx_c], dVdz = dvdz_ptr[idx_c];
    double dWdx = dwdx_ptr[idx_c], dWdy = dwdy_ptr[idx_c], dWdz = dwdz_ptr[idx_c];

    if (!is3D) {
        dUdz = 0.0; dVdz = 0.0; dWdx = 0.0; dWdy = 0.0; dWdz = 0.0;
    }

    // ---- Strain rate S_ij and rotation rate W_ij ----
    double Sxx = dUdx;
    double Syy = dVdy;
    double Szz = dWdz;
    double Sxy = 0.5 * (dUdy + dVdx);
    double Sxz = 0.5 * (dUdz + dWdx);
    double Syz = 0.5 * (dVdz + dWdy);

    double Wxy = 0.5 * (dUdy - dVdx);
    double Wxz = 0.5 * (dUdz - dWdx);
    double Wyz = 0.5 * (dVdz - dWdy);

    // ---- Production P_ij (exact, no modeling) ----
    // P_ij = -(R_ik * dU_j/dx_k + R_jk * dU_i/dx_k)
    double Pxx = -2.0 * (rxx * dUdx + rxy * dUdy + rxz * dUdz);
    double Pyy = -2.0 * (rxy * dVdx + ryy * dVdy + ryz * dVdz);
    double Pzz = -2.0 * (rxz * dWdx + ryz * dWdy + rzz * dWdz);
    double Pxy = -(rxx * dVdx + rxy * dVdy + rxz * dVdz
                 + rxy * dUdx + ryy * dUdy + ryz * dUdz);
    double Pxz = -(rxx * dWdx + rxy * dWdy + rxz * dWdz
                 + rxz * dUdx + ryz * dUdy + rzz * dUdz);
    double Pyz = -(rxy * dWdx + ryy * dWdy + ryz * dWdz
                 + rxz * dVdx + ryz * dVdy + rzz * dVdz);

    // P_k = 0.5 * tr(P_ij) = -R_ij * dU_i/dx_j
    double P_k = 0.5 * (Pxx + Pyy + Pzz);

    // ---- Anisotropy tensor b_ij = R_ij/(2k) - delta_ij/3 ----
    double inv_2k = 1.0 / (2.0 * k);
    double bxx = rxx * inv_2k - 1.0/3.0;
    double byy = ryy * inv_2k - 1.0/3.0;
    double bzz = rzz * inv_2k - 1.0/3.0;
    double bxy = rxy * inv_2k;
    double bxz = rxz * inv_2k;
    double byz = ryz * inv_2k;

    // Second invariant II_b = b_ij * b_ji
    double II_b = bxx*bxx + byy*byy + bzz*bzz + 2.0*(bxy*bxy + bxz*bxz + byz*byz);

    // ---- b_ik * b_kj (symmetric) ----
    double bb_xx = bxx*bxx + bxy*bxy + bxz*bxz;
    double bb_yy = bxy*bxy + byy*byy + byz*byz;
    double bb_zz = bxz*bxz + byz*byz + bzz*bzz;
    double bb_xy = bxx*bxy + bxy*byy + bxz*byz;
    double bb_xz = bxx*bxz + bxy*byz + bxz*bzz;
    double bb_yz = bxy*bxz + byy*byz + byz*bzz;

    // ---- b_ik * S_kj + b_jk * S_ki (symmetric) ----
    double bS_xx = 2.0 * (bxx*Sxx + bxy*Sxy + bxz*Sxz);
    double bS_yy = 2.0 * (bxy*Sxy + byy*Syy + byz*Syz);
    double bS_zz = 2.0 * (bxz*Sxz + byz*Syz + bzz*Szz);
    double bS_xy = bxx*Sxy + bxy*Syy + bxz*Syz
                 + bxy*Sxx + byy*Sxy + byz*Sxz;
    double bS_xz = bxx*Sxz + bxy*Syz + bxz*Szz
                 + bxz*Sxx + byz*Sxy + bzz*Sxz;
    double bS_yz = bxy*Sxz + byy*Syz + byz*Szz
                 + bxz*Sxy + byz*Syy + bzz*Syz;

    // b_mn * S_mn (trace of b*S)
    double bS_trace = bxx*Sxx + byy*Syy + bzz*Szz + 2.0*(bxy*Sxy + bxz*Sxz + byz*Syz);

    // ---- b_ik*W_jk + b_jk*W_ik (symmetric combination) ----
    // W is antisymmetric: W_ij = -W_ji, W_ii = 0
    // b_ik*W_jk + b_jk*W_ik
    // Note: W_ij = -W_ji, W_ii = 0
    // W_xy = Wxy, W_xz = Wxz, W_yz = Wyz
    // W_yx = -Wxy, W_zx = -Wxz, W_zy = -Wyz

    // (i=x,j=x): b_xk*W_xk + b_xk*W_xk = 2*sum_k b_xk*W_xk
    //   = 2*(bxx*0 + bxy*Wxy + bxz*Wxz) = 2*(bxy*Wxy + bxz*Wxz)
    double bW_xx2 = 2.0 * (bxy*Wxy + bxz*Wxz);

    // (i=y,j=y): 2*sum_k b_yk*W_yk = 2*(bxy*(-Wxy) + byy*0 + byz*Wyz)
    //   = 2*(-bxy*Wxy + byz*Wyz)
    double bW_yy2 = 2.0 * (-bxy*Wxy + byz*Wyz);

    // (i=z,j=z): 2*sum_k b_zk*W_zk = 2*(bxz*(-Wxz) + byz*(-Wyz) + bzz*0)
    //   = -2*(bxz*Wxz + byz*Wyz)
    double bW_zz2 = -2.0 * (bxz*Wxz + byz*Wyz);

    // (i=x,j=y): b_xk*W_yk + b_yk*W_xk
    //   = bxx*W_yx + bxy*W_yy + bxz*W_yz + byx*W_xx + byy*W_xy + byz*W_xz
    //   = bxx*(-Wxy) + 0 + bxz*Wyz + 0 + byy*Wxy + byz*Wxz
    //   = (byy-bxx)*Wxy + bxz*Wyz + byz*Wxz
    double bW_xy2 = (byy - bxx)*Wxy + bxz*Wyz + byz*Wxz;

    // (i=x,j=z): b_xk*W_zk + b_zk*W_xk
    //   = bxx*W_zx + bxy*W_zy + bxz*W_zz + bxz*W_xx + byz*W_xy + bzz*W_xz
    //   = -bxx*Wxz - bxy*Wyz + 0 + 0 + byz*Wxy + bzz*Wxz
    //   = (bzz-bxx)*Wxz - bxy*Wyz + byz*Wxy
    double bW_xz2 = (bzz - bxx)*Wxz - bxy*Wyz + byz*Wxy;

    // (i=y,j=z): b_yk*W_zk + b_zk*W_yk
    //   = bxy*W_zx + byy*W_zy + byz*W_zz + bxz*W_yx + byz*W_yy + bzz*W_yz
    //   = -bxy*Wxz - byy*Wyz + 0 - bxz*Wxy + 0 + bzz*Wyz
    //   = (bzz-byy)*Wyz - bxy*Wxz - bxz*Wxy
    double bW_yz2 = (bzz - byy)*Wyz - bxy*Wxz - bxz*Wxy;

    // ---- SSG Pressure-Strain Pi_ij ----
    double sqrt_IIb = std::sqrt(II_b);

    // Term 1: -(C1*eps + C1*P_k) * b_ij
    // Term 2: C2 * eps * (b_ik*b_kj - (1/3)*II_b*delta_ij)
    // Term 3: (C3 - C3*sqrt(II_b)) * k * S_ij
    // Term 4: C4 * k * (b_ik*S_jk + b_jk*S_ik - (2/3)*bS_trace*delta_ij)
    // Term 5: C5 * k * (b_ik*W_jk + b_jk*W_ik)

    double Pi_xx = -(C1*eps + C1s*P_k)*bxx
                 + C2*eps*(bb_xx - II_b/3.0)
                 + (C3 - C3s*sqrt_IIb)*k*Sxx
                 + C4*k*(bS_xx - 2.0/3.0*bS_trace)
                 + C5*k*bW_xx2;

    double Pi_yy = -(C1*eps + C1s*P_k)*byy
                 + C2*eps*(bb_yy - II_b/3.0)
                 + (C3 - C3s*sqrt_IIb)*k*Syy
                 + C4*k*(bS_yy - 2.0/3.0*bS_trace)
                 + C5*k*bW_yy2;

    double Pi_zz = -(C1*eps + C1s*P_k)*bzz
                 + C2*eps*(bb_zz - II_b/3.0)
                 + (C3 - C3s*sqrt_IIb)*k*Szz
                 + C4*k*(bS_zz - 2.0/3.0*bS_trace)
                 + C5*k*bW_zz2;

    double Pi_xy = -(C1*eps + C1s*P_k)*bxy
                 + C2*eps*bb_xy
                 + (C3 - C3s*sqrt_IIb)*k*Sxy
                 + C4*k*bS_xy
                 + C5*k*bW_xy2;

    double Pi_xz = -(C1*eps + C1s*P_k)*bxz
                 + C2*eps*bb_xz
                 + (C3 - C3s*sqrt_IIb)*k*Sxz
                 + C4*k*bS_xz
                 + C5*k*bW_xz2;

    double Pi_yz = -(C1*eps + C1s*P_k)*byz
                 + C2*eps*bb_yz
                 + (C3 - C3s*sqrt_IIb)*k*Syz
                 + C4*k*bS_yz
                 + C5*k*bW_yz2;

    // ---- Isotropic dissipation eps_ij = (2/3)*beta_star*k*omega*delta_ij ----
    // Handled implicitly below

    // ---- Diffusion of R_ij: nabla . [(nu + sigma_R * nu_t) * nabla R_ij] ----
    double D_eff = nu + sigma_R * nu_t_c;

    double diff_Rxx = D_eff * ((Rxx[idx_ip] - 2.0*rxx + Rxx[idx_im]) * inv_dx2
                              + (Rxx[idx_jp] - 2.0*rxx + Rxx[idx_jm]) * inv_dy2);
    double diff_Ryy = D_eff * ((Ryy[idx_ip] - 2.0*ryy + Ryy[idx_im]) * inv_dx2
                              + (Ryy[idx_jp] - 2.0*ryy + Ryy[idx_jm]) * inv_dy2);
    double diff_Rzz = D_eff * ((Rzz[idx_ip] - 2.0*rzz + Rzz[idx_im]) * inv_dx2
                              + (Rzz[idx_jp] - 2.0*rzz + Rzz[idx_jm]) * inv_dy2);
    double diff_Rxy = D_eff * ((Rxy[idx_ip] - 2.0*rxy + Rxy[idx_im]) * inv_dx2
                              + (Rxy[idx_jp] - 2.0*rxy + Rxy[idx_jm]) * inv_dy2);
    double diff_Rxz = D_eff * ((Rxz[idx_ip] - 2.0*rxz + Rxz[idx_im]) * inv_dx2
                              + (Rxz[idx_jp] - 2.0*rxz + Rxz[idx_jm]) * inv_dy2);
    double diff_Ryz = D_eff * ((Ryz[idx_ip] - 2.0*ryz + Ryz[idx_im]) * inv_dx2
                              + (Ryz[idx_jp] - 2.0*ryz + Ryz[idx_jm]) * inv_dy2);

    // Add z-diffusion for 3D
    if (is3D) {
        diff_Rxx += D_eff * (Rxx[idx_kp] - 2.0*rxx + Rxx[idx_km]) * inv_dz2;
        diff_Ryy += D_eff * (Ryy[idx_kp] - 2.0*ryy + Ryy[idx_km]) * inv_dz2;
        diff_Rzz += D_eff * (Rzz[idx_kp] - 2.0*rzz + Rzz[idx_km]) * inv_dz2;
        diff_Rxy += D_eff * (Rxy[idx_kp] - 2.0*rxy + Rxy[idx_km]) * inv_dz2;
        diff_Rxz += D_eff * (Rxz[idx_kp] - 2.0*rxz + Rxz[idx_km]) * inv_dz2;
        diff_Ryz += D_eff * (Ryz[idx_kp] - 2.0*ryz + Ryz[idx_km]) * inv_dz2;
    }

    // ---- Point-implicit time advance for R_ij ----
    // dR_ij/dt = P_ij + Pi_ij - eps_ij + diff_ij
    // eps_ij = (2/3)*beta_star*k*omega*delta_ij
    // Implicit destruction: R_new = (R + dt*source) / (1 + dt*sink)
    // For diagonal: sink = (2/3)*beta_star*omega (from dissipation)
    // For off-diagonal: sink = (2/3)*beta_star*omega (same rate, ensures symmetric decay)
    double sink_diag = (2.0/3.0) * beta_star * om;

    double src_xx = Pxx + Pi_xx + diff_Rxx;
    double src_yy = Pyy + Pi_yy + diff_Ryy;
    double src_zz = Pzz + Pi_zz + diff_Rzz;
    double src_xy = Pxy + Pi_xy + diff_Rxy;
    double src_xz = Pxz + Pi_xz + diff_Rxz;
    double src_yz = Pyz + Pi_yz + diff_Ryz;

    double denom = 1.0 + dt * sink_diag;

    Rxx_new = (rxx + dt * src_xx) / denom;
    Ryy_new = (ryy + dt * src_yy) / denom;
    Rzz_new = (rzz + dt * src_zz) / denom;
    Rxy_new = (rxy + dt * src_xy) / denom;
    Rxz_new = (rxz + dt * src_xz) / denom;
    Ryz_new = (ryz + dt * src_yz) / denom;

    // ---- Realizability enforcement ----
    // Diagonal must be positive
    Rxx_new = (Rxx_new > k_min) ? Rxx_new : k_min;
    Ryy_new = (Ryy_new > k_min) ? Ryy_new : k_min;
    Rzz_new = (Rzz_new > k_min) ? Rzz_new : k_min;

    // Schwarz inequality: |R_ij|^2 <= R_ii * R_jj
    double max_xy = std::sqrt(Rxx_new * Ryy_new);
    double max_xz = std::sqrt(Rxx_new * Rzz_new);
    double max_yz = std::sqrt(Ryy_new * Rzz_new);
    if (Rxy_new >  max_xy) Rxy_new =  max_xy;
    if (Rxy_new < -max_xy) Rxy_new = -max_xy;
    if (Rxz_new >  max_xz) Rxz_new =  max_xz;
    if (Rxz_new < -max_xz) Rxz_new = -max_xz;
    if (Ryz_new >  max_yz) Ryz_new =  max_yz;
    if (Ryz_new < -max_yz) Ryz_new = -max_yz;

    // Cap magnitudes
    Rxx_new = (Rxx_new < k_max) ? Rxx_new : k_max;
    Ryy_new = (Ryy_new < k_max) ? Ryy_new : k_max;
    Rzz_new = (Rzz_new < k_max) ? Rzz_new : k_max;

    // ---- Omega equation ----
    // dw/dt = alpha*(w/k)*P_k - beta*w^2 + nabla.[(nu + sigma_w*nu_t)*nabla w]
    double D_omega = nu + sigma_omega * nu_t_c;
    double diff_om = D_omega * ((omega_ptr[idx_ip] - 2.0*om + omega_ptr[idx_im]) * inv_dx2
                               + (omega_ptr[idx_jp] - 2.0*om + omega_ptr[idx_jm]) * inv_dy2);
    if (is3D) {
        diff_om += D_omega * (omega_ptr[idx_kp] - 2.0*om + omega_ptr[idx_km]) * inv_dz2;
    }

    // Production limited to 10*destruction (same as SST)
    double P_k_limited = P_k;
    double P_k_limit = 10.0 * beta_star * k * om;
    P_k_limited = (P_k_limited < P_k_limit) ? P_k_limited : P_k_limit;
    P_k_limited = (P_k_limited > 0.0) ? P_k_limited : 0.0;

    double src_om = alpha_omega * (om / k) * P_k_limited + diff_om;
    double sink_om = beta_omega * om;

    omega_new_out = (om + dt * src_om) / (1.0 + dt * sink_om);
    omega_new_out = (omega_new_out > omega_min) ? omega_new_out : omega_min;
    omega_new_out = (omega_new_out < omega_max) ? omega_new_out : omega_max;
}

#ifdef USE_GPU_OFFLOAD
#pragma omp end declare target
#endif

// ============================================================================
// RSMModel Implementation
// ============================================================================

RSMModel::RSMModel(const RSMConstants& constants)
    : constants_(constants) {}

RSMModel::~RSMModel() {
    cleanup_gpu_buffers();
}

void RSMModel::ensure_initialized(const Mesh& mesh, const ScalarField& k) {
    if (!initialized_) {
        R_xx_ = ScalarField(mesh);
        R_yy_ = ScalarField(mesh);
        R_zz_ = ScalarField(mesh);
        R_xy_ = ScalarField(mesh);
        R_xz_ = ScalarField(mesh);
        R_yz_ = ScalarField(mesh);

        // Initialize from existing k: R_ij = (2k/3)*delta_ij (isotropic)
        const int total = static_cast<int>(k.data().size());
        for (int idx = 0; idx < total; ++idx) {
            double k_val = k.data()[idx];
            k_val = (k_val > constants_.k_min) ? k_val : constants_.k_min;
            double R_diag = (2.0 / 3.0) * k_val;
            R_xx_.data()[idx] = R_diag;
            R_yy_.data()[idx] = R_diag;
            R_zz_.data()[idx] = R_diag;
            R_xy_.data()[idx] = 0.0;
            R_xz_.data()[idx] = 0.0;
            R_yz_.data()[idx] = 0.0;
        }

        initialized_ = true;

        // Allocate GPU flat buffers and sync initial R_ij data
#ifdef USE_GPU_OFFLOAD
        allocate_gpu_buffers(mesh);
        if (buffers_on_gpu_) {
            // Copy isotropic initialization to flat buffers and update GPU
            const int tot = cached_total_cells_;
            for (int idx = 0; idx < tot; ++idx) {
                Rxx_flat_[idx] = R_xx_.data()[idx];
                Ryy_flat_[idx] = R_yy_.data()[idx];
                Rzz_flat_[idx] = R_zz_.data()[idx];
                Rxy_flat_[idx] = R_xy_.data()[idx];
                Rxz_flat_[idx] = R_xz_.data()[idx];
                Ryz_flat_[idx] = R_yz_.data()[idx];
            }
            double* rxx_p = Rxx_flat_.data();
            double* ryy_p = Ryy_flat_.data();
            double* rzz_p = Rzz_flat_.data();
            double* rxy_p = Rxy_flat_.data();
            double* rxz_p = Rxz_flat_.data();
            double* ryz_p = Ryz_flat_.data();
            #pragma omp target update to(rxx_p[0:tot])
            #pragma omp target update to(ryy_p[0:tot])
            #pragma omp target update to(rzz_p[0:tot])
            #pragma omp target update to(rxy_p[0:tot])
            #pragma omp target update to(rxz_p[0:tot])
            #pragma omp target update to(ryz_p[0:tot])
        }
#endif
    }
}

void RSMModel::initialize(const Mesh& mesh, const VectorField& velocity) {
    (void)velocity;
    // Lazy init — actual allocation happens in ensure_initialized
    // which needs k field. GPU buffers allocated there too.
    (void)mesh;
}

void RSMModel::initialize_gpu_buffers(const Mesh& mesh) {
#ifdef USE_GPU_OFFLOAD
    gpu::verify_device_available();
    allocate_gpu_buffers(mesh);
#else
    (void)mesh;
    buffers_on_gpu_ = false;
#endif
}

void RSMModel::allocate_gpu_buffers(const Mesh& mesh) {
#ifdef USE_GPU_OFFLOAD
    const int total = mesh.total_cells();
    if (buffers_on_gpu_ && cached_total_cells_ == total) return;

    free_gpu_buffers();

    Rxx_flat_.resize(total, 0.0);
    Ryy_flat_.resize(total, 0.0);
    Rzz_flat_.resize(total, 0.0);
    Rxy_flat_.resize(total, 0.0);
    Rxz_flat_.resize(total, 0.0);
    Ryz_flat_.resize(total, 0.0);

    double* rxx_p = Rxx_flat_.data();
    double* ryy_p = Ryy_flat_.data();
    double* rzz_p = Rzz_flat_.data();
    double* rxy_p = Rxy_flat_.data();
    double* rxz_p = Rxz_flat_.data();
    double* ryz_p = Ryz_flat_.data();

    #pragma omp target enter data map(to: rxx_p[0:total])
    #pragma omp target enter data map(to: ryy_p[0:total])
    #pragma omp target enter data map(to: rzz_p[0:total])
    #pragma omp target enter data map(to: rxy_p[0:total])
    #pragma omp target enter data map(to: rxz_p[0:total])
    #pragma omp target enter data map(to: ryz_p[0:total])

    cached_total_cells_ = total;
    buffers_on_gpu_ = true;
#else
    (void)mesh;
#endif
}

void RSMModel::free_gpu_buffers() {
#ifdef USE_GPU_OFFLOAD
    if (!buffers_on_gpu_) return;

    const int sz = cached_total_cells_;
    if (sz == 0) return;

    buffers_on_gpu_ = false;

    double* rxx_p = Rxx_flat_.data();
    double* ryy_p = Ryy_flat_.data();
    double* rzz_p = Rzz_flat_.data();
    double* rxy_p = Rxy_flat_.data();
    double* rxz_p = Rxz_flat_.data();
    double* ryz_p = Ryz_flat_.data();

    #pragma omp target exit data map(release: rxx_p[0:sz])
    #pragma omp target exit data map(release: ryy_p[0:sz])
    #pragma omp target exit data map(release: rzz_p[0:sz])
    #pragma omp target exit data map(release: rxy_p[0:sz])
    #pragma omp target exit data map(release: rxz_p[0:sz])
    #pragma omp target exit data map(release: ryz_p[0:sz])

    Rxx_flat_.clear();
    Ryy_flat_.clear();
    Rzz_flat_.clear();
    Rxy_flat_.clear();
    Rxz_flat_.clear();
    Ryz_flat_.clear();
#endif
}

void RSMModel::cleanup_gpu_buffers() {
#ifdef USE_GPU_OFFLOAD
    free_gpu_buffers();
#endif
    buffers_on_gpu_ = false;
}

// ============================================================================
// advance_turbulence — Advance 6 R_ij equations + omega by dt
// ============================================================================

void RSMModel::advance_turbulence(
    const Mesh& mesh,
    const VectorField& velocity,
    double dt,
    ScalarField& k,
    ScalarField& omega,
    const ScalarField& nu_t_prev,
    const TurbulenceDeviceView* device_view)
{
    TIMED_SCOPE("rsm_transport");

    ensure_initialized(mesh, k);

    (void)velocity;  // Gradients come from device_view or are computed separately

    const int Nx = mesh.Nx;
    const int Ny = mesh.Ny;
    const int Nz = mesh.Nz;
    const int Ng = mesh.Nghost;
    const int cell_stride = mesh.total_Nx();
    const int plane_stride = mesh.total_Nx() * mesh.total_Ny();
    const bool is3D = Nz > 1;
    const int n_cells = Nx * Ny * (is3D ? Nz : 1);

    const double dx = mesh.dx;
    const double dy = mesh.dy;
    const double dz = mesh.dz;
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);
    const double inv_dz2 = is3D ? 1.0 / (dz * dz) : 0.0;

    // Copy model constants to locals (nvc++ workaround)
    const double nu_val = nu_;
    const double C1 = constants_.C1;
    const double C1s = constants_.C1_star;
    const double C2 = constants_.C2;
    const double C3 = constants_.C3;
    const double C3s = constants_.C3_star;
    const double C4 = constants_.C4;
    const double C5 = constants_.C5;
    const double alpha_omega = constants_.alpha_omega;
    const double beta_omega = constants_.beta_omega;
    const double beta_star = constants_.beta_star;
    const double sigma_omega = constants_.sigma_omega;
    const double sigma_R = constants_.sigma_R;
    const double k_min = constants_.k_min;
    const double k_max = constants_.k_max;
    const double omega_min = constants_.omega_min;
    const double omega_max = constants_.omega_max;

#ifdef USE_GPU_OFFLOAD
    if (device_view && device_view->is_valid()) {
        // GPU path: operate on solver-owned device-resident data
        // Compute velocity gradients from MAC velocity (other models do this
        // internally; RSM needs them before the transport step)
        [[maybe_unused]] const int cell_total = device_view->cell_total;

        gpu_kernels::compute_gradients_from_mac_gpu(
            device_view->u_face, device_view->v_face,
            is3D ? device_view->w_face : device_view->u_face,
            device_view->dudx, device_view->dudy,
            device_view->dvdx, device_view->dvdy,
            is3D ? device_view->dudz : device_view->dudx,
            is3D ? device_view->dvdz : device_view->dudx,
            is3D ? device_view->dwdx : device_view->dudx,
            is3D ? device_view->dwdy : device_view->dudx,
            is3D ? device_view->dwdz : device_view->dudx,
            Nx, Ny, Nz, Ng,
            device_view->u_stride, device_view->v_stride,
            is3D ? device_view->w_stride : device_view->u_stride,
            device_view->u_plane_stride, device_view->v_plane_stride,
            is3D ? device_view->w_plane_stride : device_view->u_plane_stride,
            cell_stride, plane_stride, cell_total,
            dx, dy, dz);

        // Get device pointers
        const double* dudx_d = device_view->dudx;
        const double* dudy_d = device_view->dudy;
        const double* dvdx_d = device_view->dvdx;
        const double* dvdy_d = device_view->dvdy;

        // 3D gradient pointers are only allocated/mapped for 3D meshes.
        // For 2D, alias them to dudx_d (always mapped) — the kernel zeros
        // these components when !is3D, so the values are never read.
        const double* dudz_d = is3D ? device_view->dudz : dudx_d;
        const double* dvdz_d = is3D ? device_view->dvdz : dudx_d;
        const double* dwdx_d = is3D ? device_view->dwdx : dudx_d;
        const double* dwdy_d = is3D ? device_view->dwdy : dudx_d;
        const double* dwdz_d = is3D ? device_view->dwdz : dudx_d;
        double* k_d = device_view->k;
        double* omega_d = device_view->omega;
        const double* nu_t_d = device_view->nu_t;

        // R_ij are stored in model-owned flat arrays on GPU
        // First, copy from solver's tau_ij to our flat arrays if first step
        // (They get synced back to tau_ij in update())
        double* rxx_d = Rxx_flat_.data();
        double* ryy_d = Ryy_flat_.data();
        double* rzz_d = Rzz_flat_.data();
        double* rxy_d = Rxy_flat_.data();
        double* rxz_d = Rxz_flat_.data();
        double* ryz_d = Ryz_flat_.data();

        [[maybe_unused]] const int rij_sz = cached_total_cells_;

        #pragma omp target teams distribute parallel for \
            map(present: dudx_d[0:cell_total], dudy_d[0:cell_total], \
                         dvdx_d[0:cell_total], dvdy_d[0:cell_total], \
                         dwdx_d[0:cell_total], dwdy_d[0:cell_total], \
                         dudz_d[0:cell_total], dvdz_d[0:cell_total], dwdz_d[0:cell_total], \
                         k_d[0:cell_total], omega_d[0:cell_total], nu_t_d[0:cell_total], \
                         rxx_d[0:rij_sz], ryy_d[0:rij_sz], rzz_d[0:rij_sz], \
                         rxy_d[0:rij_sz], rxz_d[0:rij_sz], ryz_d[0:rij_sz])
        for (int cidx = 0; cidx < n_cells; ++cidx) {
            int i, j, kk;
            if (is3D) {
                kk = cidx / (Nx * Ny);
                int rem = cidx % (Nx * Ny);
                j = rem / Nx;
                i = rem % Nx;
            } else {
                kk = 0; j = cidx / Nx; i = cidx % Nx;
            }
            const int ii = i + Ng, jj = j + Ng, kkk = kk + Ng;

            const int idx_c  = kkk * plane_stride + jj * cell_stride + ii;
            const int idx_ip = kkk * plane_stride + jj * cell_stride + (ii + 1);
            const int idx_im = kkk * plane_stride + jj * cell_stride + (ii - 1);
            const int idx_jp = kkk * plane_stride + (jj + 1) * cell_stride + ii;
            const int idx_jm = kkk * plane_stride + (jj - 1) * cell_stride + ii;
            const int idx_kp = is3D ? (kkk + 1) * plane_stride + jj * cell_stride + ii : idx_c;
            const int idx_km = is3D ? (kkk - 1) * plane_stride + jj * cell_stride + ii : idx_c;

            double rxx_n, ryy_n, rzz_n, rxy_n, rxz_n, ryz_n, om_n;
            rsm_cell_kernel(
                idx_c, idx_ip, idx_im, idx_jp, idx_jm, idx_kp, idx_km,
                dudx_d, dudy_d, dudz_d, dvdx_d, dvdy_d, dvdz_d,
                dwdx_d, dwdy_d, dwdz_d,
                rxx_d, ryy_d, rzz_d, rxy_d, rxz_d, ryz_d,
                omega_d, nu_t_d,
                dx, dy, dz, dt, inv_dx2, inv_dy2, inv_dz2, is3D,
                nu_val, C1, C1s, C2, C3, C3s, C4, C5,
                alpha_omega, beta_omega, beta_star, sigma_omega, sigma_R,
                k_min, k_max, omega_min, omega_max,
                rxx_n, ryy_n, rzz_n, rxy_n, rxz_n, ryz_n, om_n);

            rxx_d[idx_c] = rxx_n;
            ryy_d[idx_c] = ryy_n;
            rzz_d[idx_c] = rzz_n;
            rxy_d[idx_c] = rxy_n;
            rxz_d[idx_c] = rxz_n;
            ryz_d[idx_c] = ryz_n;
            omega_d[idx_c] = om_n;

            // Update k from trace
            k_d[idx_c] = 0.5 * (rxx_n + ryy_n + rzz_n);
        }

        // Apply wall BCs on GPU for omega (same as SST)
        const int total_Nx_bc = cell_stride;
        const double nu_bc = nu_val;
        const double beta1_bc = 0.075;  // Standard k-omega wall BC
        const double omega_max_bc = omega_max;
        const double* wall_ptr = device_view->wall_distance;
        [[maybe_unused]] const int wall_total = device_view->cell_total;

        // Omega wall BC
        #pragma omp target teams distribute parallel for \
            map(present: omega_d[0:cell_total], wall_ptr[0:wall_total])
        for (int i = Ng; i < Nx + Ng; ++i) {
            for (int g = 0; g < Ng; ++g) {
                // Bottom wall
                {
                    int idx_ghost = g * cell_stride + i;
                    int idx_int = Ng * cell_stride + i;
                    double y1 = wall_ptr[idx_int];
                    y1 = (y1 > 1e-10) ? y1 : 1e-10;
                    double ow = 60.0 * nu_bc / (beta1_bc * y1 * y1);
                    ow = (ow < omega_max_bc) ? ow : omega_max_bc;
                    omega_d[idx_ghost] = 2.0 * ow - omega_d[idx_int];
                }
                // Top wall
                {
                    int idx_ghost = (Ny + Ng + g) * cell_stride + i;
                    int idx_int = (Ny + Ng - 1) * cell_stride + i;
                    double y1 = wall_ptr[idx_int];
                    y1 = (y1 > 1e-10) ? y1 : 1e-10;
                    double ow = 60.0 * nu_bc / (beta1_bc * y1 * y1);
                    ow = (ow < omega_max_bc) ? ow : omega_max_bc;
                    omega_d[idx_ghost] = 2.0 * ow - omega_d[idx_int];
                }
            }
        }

        // R_ij wall BCs: R_ii ghost = -R_ii interior (gives 0 at wall)
        //                R_ij ghost = -R_ij interior (off-diag also 0 at wall)
        #pragma omp target teams distribute parallel for \
            map(present: rxx_d[0:rij_sz], ryy_d[0:rij_sz], rzz_d[0:rij_sz], \
                         rxy_d[0:rij_sz], rxz_d[0:rij_sz], ryz_d[0:rij_sz], \
                         k_d[0:cell_total])
        for (int i = 0; i < total_Nx_bc; ++i) {
            for (int g = 0; g < Ng; ++g) {
                // Bottom
                int ig_bot = g * cell_stride + i;
                int ii_bot = Ng * cell_stride + i;
                rxx_d[ig_bot] = -rxx_d[ii_bot];
                ryy_d[ig_bot] = -ryy_d[ii_bot];
                rzz_d[ig_bot] = -rzz_d[ii_bot];
                rxy_d[ig_bot] = -rxy_d[ii_bot];
                rxz_d[ig_bot] = -rxz_d[ii_bot];
                ryz_d[ig_bot] = -ryz_d[ii_bot];
                k_d[ig_bot]   = -k_d[ii_bot];

                // Top
                int ig_top = (Ny + Ng + g) * cell_stride + i;
                int ii_top = (Ny + Ng - 1) * cell_stride + i;
                rxx_d[ig_top] = -rxx_d[ii_top];
                ryy_d[ig_top] = -ryy_d[ii_top];
                rzz_d[ig_top] = -rzz_d[ii_top];
                rxy_d[ig_top] = -rxy_d[ii_top];
                rxz_d[ig_top] = -rxz_d[ii_top];
                ryz_d[ig_top] = -ryz_d[ii_top];
                k_d[ig_top]   = -k_d[ii_top];
            }
        }

        return;
    }
#else
    (void)device_view;
#endif

    // ---- CPU path ----
    // Get raw pointers from ScalarField data
    double* rxx_ptr = R_xx_.data().data();
    double* ryy_ptr = R_yy_.data().data();
    double* rzz_ptr = R_zz_.data().data();
    double* rxy_ptr = R_xy_.data().data();
    double* rxz_ptr = R_xz_.data().data();
    double* ryz_ptr = R_yz_.data().data();
    double* omega_ptr = omega.data().data();
    double* k_ptr = k.data().data();
    const double* nu_t_ptr = nu_t_prev.data().data();

    // Compute velocity gradients from MAC velocity on CPU.
    // Other models compute gradients internally in update(), but RSM
    // needs them in advance_turbulence() for the transport equations.
    const int total_cells = static_cast<int>(k.data().size());
    std::vector<double> grad_dudx(total_cells, 0.0);
    std::vector<double> grad_dudy(total_cells, 0.0);
    std::vector<double> grad_dvdx(total_cells, 0.0);
    std::vector<double> grad_dvdy(total_cells, 0.0);
    std::vector<double> grad_dudz(total_cells, 0.0);
    std::vector<double> grad_dvdz(total_cells, 0.0);
    std::vector<double> grad_dwdx(total_cells, 0.0);
    std::vector<double> grad_dwdy(total_cells, 0.0);
    std::vector<double> grad_dwdz(total_cells, 0.0);

    // Compute gradients from staggered MAC velocity
    const double inv_dx = 1.0 / dx;
    const double inv_dy = 1.0 / dy;
    const double inv_dz = is3D ? 1.0 / dz : 0.0;
    const int u_stride = velocity.u_stride();
    const int v_stride = velocity.v_stride();

    for (int cidx = 0; cidx < n_cells; ++cidx) {
        int il, jl, kl;
        if (is3D) {
            kl = cidx / (Nx * Ny); il = cidx % Nx; jl = (cidx / Nx) % Ny;
        } else {
            il = cidx % Nx; jl = cidx / Nx; kl = 0;
        }
        int ig = il + Ng, jg = jl + Ng, kg = is3D ? kl + Ng : 0;
        int cell = kg * plane_stride + jg * cell_stride + ig;

        // du/dx, du/dy from staggered u
        grad_dudx[cell] = (velocity.u_data()[jg * u_stride + (ig + 1)]
                         - velocity.u_data()[jg * u_stride + ig]) * inv_dx;
        grad_dudy[cell] = 0.5 * ((velocity.u_data()[(jg+1) * u_stride + ig]
                                 + velocity.u_data()[(jg+1) * u_stride + (ig+1)])
                                - (velocity.u_data()[(jg-1) * u_stride + ig]
                                 + velocity.u_data()[(jg-1) * u_stride + (ig+1)])) * 0.5 * inv_dy;
        // dv/dx, dv/dy from staggered v
        grad_dvdx[cell] = 0.5 * ((velocity.v_data()[jg * v_stride + (ig+1)]
                                 + velocity.v_data()[(jg+1) * v_stride + (ig+1)])
                                - (velocity.v_data()[jg * v_stride + (ig-1)]
                                 + velocity.v_data()[(jg+1) * v_stride + (ig-1)])) * 0.5 * inv_dx;
        grad_dvdy[cell] = (velocity.v_data()[(jg + 1) * v_stride + ig]
                         - velocity.v_data()[jg * v_stride + ig]) * inv_dy;
    }

    const double* dudx_ptr = grad_dudx.data();
    const double* dudy_ptr = grad_dudy.data();
    const double* dudz_ptr = grad_dudz.data();
    const double* dvdx_ptr = grad_dvdx.data();
    const double* dvdy_ptr = grad_dvdy.data();
    const double* dvdz_ptr = grad_dvdz.data();
    const double* dwdx_ptr = grad_dwdx.data();
    const double* dwdy_ptr = grad_dwdy.data();
    const double* dwdz_ptr = grad_dwdz.data();

    for (int cidx = 0; cidx < n_cells; ++cidx) {
        int i_loc, j_loc, k_loc;
        if (is3D) {
            k_loc = cidx / (Nx * Ny);
            int rem = cidx % (Nx * Ny);
            j_loc = rem / Nx;
            i_loc = rem % Nx;
        } else {
            k_loc = 0; j_loc = cidx / Nx; i_loc = cidx % Nx;
        }
        const int ii = i_loc + Ng, jj = j_loc + Ng, kkk = k_loc + Ng;

        const int idx_c  = kkk * plane_stride + jj * cell_stride + ii;
        const int idx_ip = kkk * plane_stride + jj * cell_stride + (ii + 1);
        const int idx_im = kkk * plane_stride + jj * cell_stride + (ii - 1);
        const int idx_jp = kkk * plane_stride + (jj + 1) * cell_stride + ii;
        const int idx_jm = kkk * plane_stride + (jj - 1) * cell_stride + ii;
        const int idx_kp = is3D ? (kkk + 1) * plane_stride + jj * cell_stride + ii : idx_c;
        const int idx_km = is3D ? (kkk - 1) * plane_stride + jj * cell_stride + ii : idx_c;

        double rxx_n, ryy_n, rzz_n, rxy_n, rxz_n, ryz_n, om_n;
        rsm_cell_kernel(
            idx_c, idx_ip, idx_im, idx_jp, idx_jm, idx_kp, idx_km,
            dudx_ptr, dudy_ptr, dudz_ptr, dvdx_ptr, dvdy_ptr, dvdz_ptr,
            dwdx_ptr, dwdy_ptr, dwdz_ptr,
            rxx_ptr, ryy_ptr, rzz_ptr, rxy_ptr, rxz_ptr, ryz_ptr,
            omega_ptr, nu_t_ptr,
            dx, dy, dz, dt, inv_dx2, inv_dy2, inv_dz2, is3D,
            nu_val, C1, C1s, C2, C3, C3s, C4, C5,
            alpha_omega, beta_omega, beta_star, sigma_omega, sigma_R,
            k_min, k_max, omega_min, omega_max,
            rxx_n, ryy_n, rzz_n, rxy_n, rxz_n, ryz_n, om_n);

        rxx_ptr[idx_c] = rxx_n;
        ryy_ptr[idx_c] = ryy_n;
        rzz_ptr[idx_c] = rzz_n;
        rxy_ptr[idx_c] = rxy_n;
        rxz_ptr[idx_c] = rxz_n;
        ryz_ptr[idx_c] = ryz_n;
        omega_ptr[idx_c] = om_n;
        k_ptr[idx_c] = 0.5 * (rxx_n + ryy_n + rzz_n);
    }

    // Wall BCs for omega (same as SST)
    const double beta1_bc = 0.075;
    for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
        // Bottom
        int j_first = mesh.j_begin();
        double y1 = std::max(mesh.wall_distance(i, j_first), 1e-10);
        double ow = std::min(60.0 * nu_val / (beta1_bc * y1 * y1), omega_max);
        for (int g = 0; g < Ng; ++g) {
            omega(i, g) = 2.0 * ow - omega(i, j_first);
        }
        // Top
        int j_last = mesh.j_end() - 1;
        y1 = std::max(mesh.wall_distance(i, j_last), 1e-10);
        ow = std::min(60.0 * nu_val / (beta1_bc * y1 * y1), omega_max);
        for (int g = 0; g < Ng; ++g) {
            omega(i, Ny + Ng + g) = 2.0 * ow - omega(i, j_last);
        }
    }

    // R_ij wall BCs
    for (int i = 0; i < mesh.total_Nx(); ++i) {
        for (int g = 0; g < Ng; ++g) {
            // Bottom
            R_xx_(i, g) = -R_xx_(i, Ng);
            R_yy_(i, g) = -R_yy_(i, Ng);
            R_zz_(i, g) = -R_zz_(i, Ng);
            R_xy_(i, g) = -R_xy_(i, Ng);
            R_xz_(i, g) = -R_xz_(i, Ng);
            R_yz_(i, g) = -R_yz_(i, Ng);
            k(i, g) = -k(i, Ng);

            // Top
            R_xx_(i, Ny + Ng + g) = -R_xx_(i, Ny + Ng - 1);
            R_yy_(i, Ny + Ng + g) = -R_yy_(i, Ny + Ng - 1);
            R_zz_(i, Ny + Ng + g) = -R_zz_(i, Ny + Ng - 1);
            R_xy_(i, Ny + Ng + g) = -R_xy_(i, Ny + Ng - 1);
            R_xz_(i, Ny + Ng + g) = -R_xz_(i, Ny + Ng - 1);
            R_yz_(i, Ny + Ng + g) = -R_yz_(i, Ny + Ng - 1);
            k(i, Ny + Ng + g) = -k(i, Ny + Ng - 1);
        }
    }
}

// ============================================================================
// update — Copy R_ij to solver's tau_ij, compute nu_t
// ============================================================================

void RSMModel::update(
    const Mesh& mesh,
    const VectorField& velocity,
    const ScalarField& k,
    const ScalarField& omega,
    ScalarField& nu_t,
    TensorField* tau_ij,
    const TurbulenceDeviceView* device_view)
{
    TIMED_SCOPE("rsm_update");
    (void)velocity;

    ensure_initialized(mesh, k);

    const int Nx = mesh.Nx;
    const int Ny = mesh.Ny;
    const int Nz = mesh.Nz;
    const int Ng = mesh.Nghost;
    const bool is3D = Nz > 1;
    const double beta_star = constants_.beta_star;
    const double k_min = constants_.k_min;
    const double omega_min = constants_.omega_min;

#ifdef USE_GPU_OFFLOAD
    if (device_view && device_view->is_valid()) {
        const int n_cells = Nx * Ny * (is3D ? Nz : 1);
        const int cell_stride = mesh.total_Nx();
        const int plane_stride = mesh.total_Nx() * mesh.total_Ny();
        [[maybe_unused]] const int cell_total = device_view->cell_total;

        double* nu_t_d = device_view->nu_t;
        const double* k_d = device_view->k;
        const double* omega_d = device_view->omega;

        // Copy R_ij to solver's tau_ij
        double* tau_xx_d = device_view->tau_xx;
        double* tau_xy_d = device_view->tau_xy;
        double* tau_xz_d = device_view->tau_xz;
        double* tau_yy_d = device_view->tau_yy;
        double* tau_yz_d = device_view->tau_yz;
        double* tau_zz_d = device_view->tau_zz;

        const double* rxx_d = Rxx_flat_.data();
        const double* ryy_d = Ryy_flat_.data();
        const double* rzz_d = Rzz_flat_.data();
        const double* rxy_d = Rxy_flat_.data();
        const double* rxz_d = Rxz_flat_.data();
        const double* ryz_d = Ryz_flat_.data();
        [[maybe_unused]] const int rij_sz = cached_total_cells_;

        #pragma omp target teams distribute parallel for \
            map(present: nu_t_d[0:cell_total], k_d[0:cell_total], omega_d[0:cell_total], \
                         tau_xx_d[0:cell_total], tau_xy_d[0:cell_total], tau_xz_d[0:cell_total], \
                         tau_yy_d[0:cell_total], tau_yz_d[0:cell_total], tau_zz_d[0:cell_total], \
                         rxx_d[0:rij_sz], ryy_d[0:rij_sz], rzz_d[0:rij_sz], \
                         rxy_d[0:rij_sz], rxz_d[0:rij_sz], ryz_d[0:rij_sz])
        for (int cidx = 0; cidx < n_cells; ++cidx) {
            int i_l, j_l, k_l;
            if (is3D) {
                k_l = cidx / (Nx * Ny);
                int rem = cidx % (Nx * Ny);
                j_l = rem / Nx;
                i_l = rem % Nx;
            } else {
                k_l = 0; j_l = cidx / Nx; i_l = cidx % Nx;
            }
            const int idx = (k_l + Ng) * plane_stride + (j_l + Ng) * cell_stride + (i_l + Ng);

            tau_xx_d[idx] = rxx_d[idx];
            tau_xy_d[idx] = rxy_d[idx];
            tau_xz_d[idx] = rxz_d[idx];
            tau_yy_d[idx] = ryy_d[idx];
            tau_yz_d[idx] = ryz_d[idx];
            tau_zz_d[idx] = rzz_d[idx];

            // nu_t = C_mu * k / omega (for diffusion operator)
            double k_val = k_d[idx];
            double om_val = omega_d[idx];
            k_val = (k_val > k_min) ? k_val : k_min;
            om_val = (om_val > omega_min) ? om_val : omega_min;
            nu_t_d[idx] = beta_star * k_val / om_val;
        }

        return;
    }
#else
    (void)device_view;
#endif

    // CPU path
    for (int kk = (is3D ? mesh.k_begin() : 0); kk < (is3D ? mesh.k_end() : 1); ++kk) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                if (tau_ij) {
                    if (is3D) {
                        tau_ij->xx(i, j, kk) = R_xx_(i, j, kk);
                        tau_ij->xy(i, j, kk) = R_xy_(i, j, kk);
                        tau_ij->xz(i, j, kk) = R_xz_(i, j, kk);
                        tau_ij->yy(i, j, kk) = R_yy_(i, j, kk);
                        tau_ij->yz(i, j, kk) = R_yz_(i, j, kk);
                        tau_ij->zz(i, j, kk) = R_zz_(i, j, kk);
                    } else {
                        tau_ij->xx(i, j) = R_xx_(i, j);
                        tau_ij->xy(i, j) = R_xy_(i, j);
                        tau_ij->yy(i, j) = R_yy_(i, j);
                    }
                }

                double k_val = std::max(k_min, is3D ? k(i, j, kk) : k(i, j));
                double om_val = std::max(omega_min, is3D ? omega(i, j, kk) : omega(i, j));
                double nut = beta_star * k_val / om_val;

                if (is3D) {
                    nu_t(i, j, kk) = nut;
                } else {
                    nu_t(i, j) = nut;
                }
            }
        }
    }
}

} // namespace nncfd
