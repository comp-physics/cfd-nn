#include "gpu_kernels.hpp"
#include "numerics.hpp"
#include "profiling.hpp"
#include <cmath>
#include <algorithm>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

namespace nncfd {
namespace gpu_kernels {

// ============================================================================
// GPU Kernel: Compute cell-centered gradients from staggered MAC grid
// ============================================================================
void compute_gradients_from_mac_gpu(
    const double* u_face,
    const double* v_face,
    const double* w_face,
    double* dudx_cell,
    double* dudy_cell,
    double* dvdx_cell,
    double* dvdy_cell,
    double* dudz_cell,
    double* dvdz_cell,
    double* dwdx_cell,
    double* dwdy_cell,
    double* dwdz_cell,
    int Nx, int Ny, int Nz,
    int Ng,
    double dx, double dy, double dz,
    int u_stride,
    int v_stride,
    int cell_stride,
    int u_plane_stride,
    int v_plane_stride,
    int w_stride,
    int w_plane_stride,
    int cell_plane_stride,
    int u_total_size,
    int v_total_size,
    int w_total_size,
    int cell_total_size,
    const double* dyc,
    int dyc_size)
{
    NVTX_SCOPE_GRADIENT("kernel:gradients_from_mac");

    const double inv_2dx = 1.0 / (2.0 * dx);
    const double inv_2dy = 1.0 / (2.0 * dy);
    [[maybe_unused]] const double inv_2dz = (Nz > 1) ? 1.0 / (2.0 * dz) : 0.0;
    [[maybe_unused]] const double inv_dz = (Nz > 1) ? 1.0 / dz : 0.0;
    const bool use_stretched = (dyc != nullptr && dyc_size > 0);
    const bool is3D = (Nz > 1 && w_face != nullptr);
    const bool has_3d_grads = (dudz_cell != nullptr);  // 3D gradient outputs provided
    const int n_cells_3d = Nx * Ny * Nz;

#ifdef USE_GPU_OFFLOAD
    if (is3D && has_3d_grads && use_stretched) {
        #pragma omp target teams distribute parallel for \
            map(present: u_face[0:u_total_size], v_face[0:v_total_size], \
                         w_face[0:w_total_size], \
                         dudx_cell[0:cell_total_size], dudy_cell[0:cell_total_size], \
                         dvdx_cell[0:cell_total_size], dvdy_cell[0:cell_total_size], \
                         dudz_cell[0:cell_total_size], dvdz_cell[0:cell_total_size], \
                         dwdx_cell[0:cell_total_size], dwdy_cell[0:cell_total_size], \
                         dwdz_cell[0:cell_total_size], \
                         dyc[0:dyc_size])
        for (int idx = 0; idx < n_cells_3d; ++idx) {
            const int ii = idx % Nx;
            const int jj = (idx / Nx) % Ny;
            const int kk = idx / (Nx * Ny);
            const int i = ii + Ng;
            const int j = jj + Ng;
            const int k = kk + Ng;
            const int idx_cell = k * cell_plane_stride + j * cell_stride + i;

            const int u_base = k * u_plane_stride;
            const int v_base = k * v_plane_stride;

            dudx_cell[idx_cell] = (u_face[u_base + j * u_stride + (i + 1)] - u_face[u_base + j * u_stride + (i - 1)]) * inv_2dx;
            dvdx_cell[idx_cell] = (v_face[v_base + j * v_stride + (i + 1)] - v_face[v_base + j * v_stride + (i - 1)]) * inv_2dx;

            double inv_dy_span = 1.0 / (dyc[j] + dyc[j + 1]);
            dudy_cell[idx_cell] = (u_face[u_base + (j + 1) * u_stride + i] - u_face[u_base + (j - 1) * u_stride + i]) * inv_dy_span;
            dvdy_cell[idx_cell] = (v_face[v_base + (j + 1) * v_stride + i] - v_face[v_base + (j - 1) * v_stride + i]) * inv_dy_span;

            // 3D z-gradients: du/dz, dv/dz from central difference on face values
            dudz_cell[idx_cell] = (u_face[(k + 1) * u_plane_stride + j * u_stride + i] - u_face[(k - 1) * u_plane_stride + j * u_stride + i]) * inv_2dz;
            dvdz_cell[idx_cell] = (v_face[(k + 1) * v_plane_stride + j * v_stride + i] - v_face[(k - 1) * v_plane_stride + j * v_stride + i]) * inv_2dz;

            // dw/dx, dw/dy at cell center from central difference on w-face values
            const int w_base = k * w_plane_stride;
            dwdx_cell[idx_cell] = (w_face[w_base + j * w_stride + (i + 1)] - w_face[w_base + j * w_stride + (i - 1)]) * inv_2dx;
            dwdy_cell[idx_cell] = (w_face[w_base + (j + 1) * w_stride + i] - w_face[w_base + (j - 1) * w_stride + i]) * inv_dy_span;

            // dw/dz: w is at z-faces, so dw/dz at cell center = (w[k+1] - w[k]) / dz
            dwdz_cell[idx_cell] = (w_face[(k + 1) * w_plane_stride + j * w_stride + i] - w_face[k * w_plane_stride + j * w_stride + i]) * inv_dz;
        }
    } else if (is3D && has_3d_grads) {
        #pragma omp target teams distribute parallel for \
            map(present: u_face[0:u_total_size], v_face[0:v_total_size], \
                         w_face[0:w_total_size], \
                         dudx_cell[0:cell_total_size], dudy_cell[0:cell_total_size], \
                         dvdx_cell[0:cell_total_size], dvdy_cell[0:cell_total_size], \
                         dudz_cell[0:cell_total_size], dvdz_cell[0:cell_total_size], \
                         dwdx_cell[0:cell_total_size], dwdy_cell[0:cell_total_size], \
                         dwdz_cell[0:cell_total_size])
        for (int idx = 0; idx < n_cells_3d; ++idx) {
            const int ii = idx % Nx;
            const int jj = (idx / Nx) % Ny;
            const int kk = idx / (Nx * Ny);
            const int i = ii + Ng;
            const int j = jj + Ng;
            const int k = kk + Ng;
            const int idx_cell = k * cell_plane_stride + j * cell_stride + i;

            const int u_base = k * u_plane_stride;
            const int v_base = k * v_plane_stride;

            dudx_cell[idx_cell] = (u_face[u_base + j * u_stride + (i + 1)] - u_face[u_base + j * u_stride + (i - 1)]) * inv_2dx;
            dudy_cell[idx_cell] = (u_face[u_base + (j + 1) * u_stride + i] - u_face[u_base + (j - 1) * u_stride + i]) * inv_2dy;
            dvdx_cell[idx_cell] = (v_face[v_base + j * v_stride + (i + 1)] - v_face[v_base + j * v_stride + (i - 1)]) * inv_2dx;
            dvdy_cell[idx_cell] = (v_face[v_base + (j + 1) * v_stride + i] - v_face[v_base + (j - 1) * v_stride + i]) * inv_2dy;

            // 3D z-gradients
            dudz_cell[idx_cell] = (u_face[(k + 1) * u_plane_stride + j * u_stride + i] - u_face[(k - 1) * u_plane_stride + j * u_stride + i]) * inv_2dz;
            dvdz_cell[idx_cell] = (v_face[(k + 1) * v_plane_stride + j * v_stride + i] - v_face[(k - 1) * v_plane_stride + j * v_stride + i]) * inv_2dz;

            const int w_base = k * w_plane_stride;
            dwdx_cell[idx_cell] = (w_face[w_base + j * w_stride + (i + 1)] - w_face[w_base + j * w_stride + (i - 1)]) * inv_2dx;
            dwdy_cell[idx_cell] = (w_face[w_base + (j + 1) * w_stride + i] - w_face[w_base + (j - 1) * w_stride + i]) * inv_2dy;
            dwdz_cell[idx_cell] = (w_face[(k + 1) * w_plane_stride + j * w_stride + i] - w_face[k * w_plane_stride + j * w_stride + i]) * inv_dz;
        }
    } else if (use_stretched) {
        // 2D stretched-y path
        [[maybe_unused]] int w_ts = w_total_size;
        #pragma omp target teams distribute parallel for \
            map(present: u_face[0:u_total_size], v_face[0:v_total_size], \
                         dudx_cell[0:cell_total_size], dudy_cell[0:cell_total_size], \
                         dvdx_cell[0:cell_total_size], dvdy_cell[0:cell_total_size], \
                         dyc[0:dyc_size])
        for (int idx = 0; idx < n_cells_3d; ++idx) {
            const int ii = idx % Nx;
            const int jj = (idx / Nx) % Ny;
            const int kk = idx / (Nx * Ny);
            const int i = ii + Ng;
            const int j = jj + Ng;
            const int k = kk + Ng;
            const int idx_cell = k * cell_plane_stride + j * cell_stride + i;

            const int u_base = k * u_plane_stride;
            const int v_base = k * v_plane_stride;

            dudx_cell[idx_cell] = (u_face[u_base + j * u_stride + (i + 1)] - u_face[u_base + j * u_stride + (i - 1)]) * inv_2dx;
            dvdx_cell[idx_cell] = (v_face[v_base + j * v_stride + (i + 1)] - v_face[v_base + j * v_stride + (i - 1)]) * inv_2dx;

            double inv_dy_span = 1.0 / (dyc[j] + dyc[j + 1]);
            dudy_cell[idx_cell] = (u_face[u_base + (j + 1) * u_stride + i] - u_face[u_base + (j - 1) * u_stride + i]) * inv_dy_span;
            dvdy_cell[idx_cell] = (v_face[v_base + (j + 1) * v_stride + i] - v_face[v_base + (j - 1) * v_stride + i]) * inv_dy_span;
        }
    } else {
        // 2D uniform path
        [[maybe_unused]] int w_ts = w_total_size;
        #pragma omp target teams distribute parallel for \
            map(present: u_face[0:u_total_size], v_face[0:v_total_size], \
                         dudx_cell[0:cell_total_size], dudy_cell[0:cell_total_size], \
                         dvdx_cell[0:cell_total_size], dvdy_cell[0:cell_total_size])
        for (int idx = 0; idx < n_cells_3d; ++idx) {
            const int ii = idx % Nx;
            const int jj = (idx / Nx) % Ny;
            const int kk = idx / (Nx * Ny);
            const int i = ii + Ng;
            const int j = jj + Ng;
            const int k = kk + Ng;
            const int idx_cell = k * cell_plane_stride + j * cell_stride + i;

            const int u_base = k * u_plane_stride;
            const int v_base = k * v_plane_stride;

            dudx_cell[idx_cell] = (u_face[u_base + j * u_stride + (i + 1)] - u_face[u_base + j * u_stride + (i - 1)]) * inv_2dx;
            dudy_cell[idx_cell] = (u_face[u_base + (j + 1) * u_stride + i] - u_face[u_base + (j - 1) * u_stride + i]) * inv_2dy;
            dvdx_cell[idx_cell] = (v_face[v_base + j * v_stride + (i + 1)] - v_face[v_base + j * v_stride + (i - 1)]) * inv_2dx;
            dvdy_cell[idx_cell] = (v_face[v_base + (j + 1) * v_stride + i] - v_face[v_base + (j - 1) * v_stride + i]) * inv_2dy;
        }
    }
#else
    (void)u_total_size; (void)v_total_size; (void)w_total_size; (void)cell_total_size;
    for (int idx = 0; idx < n_cells_3d; ++idx) {
        const int ii = idx % Nx;
        const int jj = (idx / Nx) % Ny;
        const int kk = idx / (Nx * Ny);
        const int i = ii + Ng;
        const int j = jj + Ng;
        // 2D (Nz=1): use plane 0 to match solver's 2D convention
        // 3D (Nz>1): use interior plane at kk + Ng
        const int k = is3D ? (kk + Ng) : 0;
        const int idx_cell = k * cell_plane_stride + j * cell_stride + i;

        const int u_base = k * u_plane_stride;
        const int v_base = k * v_plane_stride;

        double inv_dy_local = inv_2dy;
        if (use_stretched) {
            inv_dy_local = 1.0 / (dyc[j] + dyc[j + 1]);
        }

        dudx_cell[idx_cell] = (u_face[u_base + j * u_stride + (i + 1)] - u_face[u_base + j * u_stride + (i - 1)]) * inv_2dx;
        dudy_cell[idx_cell] = (u_face[u_base + (j + 1) * u_stride + i] - u_face[u_base + (j - 1) * u_stride + i]) * inv_dy_local;
        dvdx_cell[idx_cell] = (v_face[v_base + j * v_stride + (i + 1)] - v_face[v_base + j * v_stride + (i - 1)]) * inv_2dx;
        dvdy_cell[idx_cell] = (v_face[v_base + (j + 1) * v_stride + i] - v_face[v_base + (j - 1) * v_stride + i]) * inv_dy_local;

        if (is3D && dudz_cell != nullptr) {
            dudz_cell[idx_cell] = (u_face[(k + 1) * u_plane_stride + j * u_stride + i] - u_face[(k - 1) * u_plane_stride + j * u_stride + i]) * inv_2dz;
            dvdz_cell[idx_cell] = (v_face[(k + 1) * v_plane_stride + j * v_stride + i] - v_face[(k - 1) * v_plane_stride + j * v_stride + i]) * inv_2dz;

            const int w_base = k * w_plane_stride;
            dwdx_cell[idx_cell] = (w_face[w_base + j * w_stride + (i + 1)] - w_face[w_base + j * w_stride + (i - 1)]) * inv_2dx;
            dwdy_cell[idx_cell] = (w_face[w_base + (j + 1) * w_stride + i] - w_face[w_base + (j - 1) * w_stride + i]) * inv_dy_local;
            dwdz_cell[idx_cell] = (w_face[(k + 1) * w_plane_stride + j * w_stride + i] - w_face[k * w_plane_stride + j * w_stride + i]) * inv_dz;
        }
    }
#endif
}

// ============================================================================
// GPU Kernel: Compute scalar MLP features
// ============================================================================
void compute_pope_invariants_gpu(
    const double* dudx, const double* dudy,
    const double* dvdx, const double* dvdy,
    const double* k, const double* omega,
    const double* u_face, const double* v_face,
    const double* w_face,
    double* features,
    int Nx, int Ny, int Nz, int Ng,
    int cell_stride, int cell_plane_stride,
    int u_stride, int u_plane_stride,
    int v_stride, int v_plane_stride,
    int w_stride, int w_plane_stride,
    int total_cells, int u_total, int v_total, int w_total,
    double dx, double dy, double dz)
{
    NVTX_SCOPE_TURB("kernel:pope_invariants");

    const int n_cells = Nx * Ny * Nz;
    const bool is3D = (Nz > 1 && w_face != nullptr);
    [[maybe_unused]] const double inv_2dx = 1.0 / (2.0 * dx);
    [[maybe_unused]] const double inv_2dy = 1.0 / (2.0 * dy);
    [[maybe_unused]] const double inv_2dz = is3D ? 1.0 / (2.0 * dz) : 0.0;
    [[maybe_unused]] const double inv_dz = is3D ? 1.0 / dz : 0.0;
    [[maybe_unused]] const double C_mu = 0.09;

#ifdef USE_GPU_OFFLOAD
    if (is3D) {
        #pragma omp target teams distribute parallel for \
            map(present: dudx[0:total_cells], dudy[0:total_cells], \
                         dvdx[0:total_cells], dvdy[0:total_cells], \
                         k[0:total_cells], omega[0:total_cells], \
                         u_face[0:u_total], v_face[0:v_total], \
                         w_face[0:w_total], \
                         features[0:(n_cells*5)])
        for (int idx = 0; idx < n_cells; ++idx) {
            const int ii = idx % Nx;
            const int jj = (idx / Nx) % Ny;
            const int kk = idx / (Nx * Ny);
            const int i = ii + Ng;
            const int j = jj + Ng;
            const int kp = kk + Ng;
            const int idx_cell = kp * cell_plane_stride + j * cell_stride + i;

            // Get 2D gradients (already computed by gradient kernel)
            double dudx_v = dudx[idx_cell];
            double dudy_v = dudy[idx_cell];
            double dvdx_v = dvdx[idx_cell];
            double dvdy_v = dvdy[idx_cell];

            // Compute z-gradients inline from staggered velocity fields
            const int u_base_kp = (kp + 1) * u_plane_stride + j * u_stride + i;
            const int u_base_km = (kp - 1) * u_plane_stride + j * u_stride + i;
            double dudz_v = (u_face[u_base_kp] - u_face[u_base_km]) * inv_2dz;

            const int v_base_kp = (kp + 1) * v_plane_stride + j * v_stride + i;
            const int v_base_km = (kp - 1) * v_plane_stride + j * v_stride + i;
            double dvdz_v = (v_face[v_base_kp] - v_face[v_base_km]) * inv_2dz;

            const int w_base = kp * w_plane_stride + j * w_stride;
            double dwdx_v = (w_face[w_base + (i + 1)] - w_face[w_base + (i - 1)]) * inv_2dx;
            double dwdy_v = (w_face[kp * w_plane_stride + (j + 1) * w_stride + i] -
                             w_face[kp * w_plane_stride + (j - 1) * w_stride + i]) * inv_2dy;
            double dwdz_v = (w_face[(kp + 1) * w_plane_stride + j * w_stride + i] -
                             w_face[kp * w_plane_stride + j * w_stride + i]) * inv_dz;

            // Strain rate tensor: S_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
            double Sxx = dudx_v;
            double Syy = dvdy_v;
            double Szz = dwdz_v;
            double Sxy = 0.5 * (dudy_v + dvdx_v);
            double Sxz = 0.5 * (dudz_v + dwdx_v);
            double Syz = 0.5 * (dvdz_v + dwdy_v);

            // Rotation tensor: Omega_ij = 0.5 * (du_i/dx_j - du_j/dx_i)
            double Oxy = 0.5 * (dudy_v - dvdx_v);
            double Oxz = 0.5 * (dudz_v - dwdx_v);
            double Oyz = 0.5 * (dvdz_v - dwdy_v);

            // Non-dimensionalize: S_hat = tau * S, Omega_hat = tau * Omega
            // tau = k / epsilon, epsilon = C_mu * k * omega
            double k_val = k[idx_cell];
            double omega_val = omega[idx_cell];
            double eps = C_mu * k_val * omega_val;
            double k_safe = (k_val > 1e-10) ? k_val : 1e-10;
            double eps_safe = (eps > 1e-20) ? eps : 1e-20;
            double tau = k_safe / eps_safe;

            // S_hat components
            double Sh11 = Sxx * tau, Sh22 = Syy * tau, Sh33 = Szz * tau;
            double Sh12 = Sxy * tau, Sh13 = Sxz * tau, Sh23 = Syz * tau;
            // Omega_hat components (antisymmetric: Oh_ij = -Oh_ji, diag=0)
            double Oh12 = Oxy * tau, Oh13 = Oxz * tau, Oh23 = Oyz * tau;

            // lambda_1 = tr(S_hat^2)
            double lam1 = Sh11*Sh11 + Sh22*Sh22 + Sh33*Sh33
                        + 2.0*(Sh12*Sh12 + Sh13*Sh13 + Sh23*Sh23);

            // lambda_2 = tr(Omega_hat^2)  (Omega antisymmetric => tr(O^2) = -2*sum(Oij^2))
            double lam2 = -2.0*(Oh12*Oh12 + Oh13*Oh13 + Oh23*Oh23);

            // lambda_3 = tr(S_hat^3)
            // (S^3)_ii = sum_j,k S_ij S_jk S_ki
            double S3_11 = Sh11*Sh11*Sh11 + Sh11*Sh12*Sh12 + Sh11*Sh13*Sh13
                         + Sh12*Sh12*Sh11 + Sh12*Sh22*Sh12 + Sh12*Sh23*Sh13
                         + Sh13*Sh13*Sh11 + Sh13*Sh23*Sh12 + Sh13*Sh33*Sh13;
            double S3_22 = Sh12*Sh11*Sh12 + Sh12*Sh12*Sh22 + Sh12*Sh13*Sh23
                         + Sh22*Sh12*Sh12 + Sh22*Sh22*Sh22 + Sh22*Sh23*Sh23
                         + Sh23*Sh13*Sh12 + Sh23*Sh23*Sh22 + Sh23*Sh33*Sh23;
            double S3_33 = Sh13*Sh11*Sh13 + Sh13*Sh12*Sh23 + Sh13*Sh13*Sh33
                         + Sh23*Sh12*Sh13 + Sh23*Sh22*Sh23 + Sh23*Sh23*Sh33
                         + Sh33*Sh13*Sh13 + Sh33*Sh23*Sh23 + Sh33*Sh33*Sh33;
            double lam3 = S3_11 + S3_22 + S3_33;

            // lambda_4 = tr(Omega_hat^2 * S_hat)
            // O^2 is symmetric: (O^2)_ij = sum_k O_ik O_kj
            // O_ij antisymmetric: O_11=O_22=O_33=0, O_21=-O_12, O_31=-O_13, O_32=-O_23
            double O2_11 = -(Oh12*Oh12 + Oh13*Oh13);
            double O2_22 = -(Oh12*Oh12 + Oh23*Oh23);
            double O2_33 = -(Oh13*Oh13 + Oh23*Oh23);
            double O2_12 = -(Oh13*Oh23);  // O_1k*O_k2 = O_13*O_32 = -Oh13*Oh23
            double O2_13 = Oh12*Oh23;     // O_1k*O_k3 = O_12*O_23 = Oh12*Oh23
            double O2_23 = -(Oh12*Oh13);  // O_2k*O_k3 = O_21*O_13 = -Oh12*Oh13
            // tr(O^2 * S) = sum_i (O^2 * S)_ii = sum_i,j O2_ij * S_ji
            double lam4 = O2_11*Sh11 + O2_12*Sh12 + O2_13*Sh13
                        + O2_12*Sh12 + O2_22*Sh22 + O2_23*Sh23
                        + O2_13*Sh13 + O2_23*Sh23 + O2_33*Sh33;

            // lambda_5 = tr(Omega_hat^2 * S_hat^2)
            // S^2 is symmetric: (S^2)_ij = sum_k S_ik S_kj
            double S2_11 = Sh11*Sh11 + Sh12*Sh12 + Sh13*Sh13;
            double S2_22 = Sh12*Sh12 + Sh22*Sh22 + Sh23*Sh23;
            double S2_33 = Sh13*Sh13 + Sh23*Sh23 + Sh33*Sh33;
            double S2_12 = Sh11*Sh12 + Sh12*Sh22 + Sh13*Sh23;
            double S2_13 = Sh11*Sh13 + Sh12*Sh23 + Sh13*Sh33;
            double S2_23 = Sh12*Sh13 + Sh22*Sh23 + Sh23*Sh33;
            // tr(O^2 * S^2) = sum_i,j O2_ij * S2_ji
            double lam5 = O2_11*S2_11 + O2_12*S2_12 + O2_13*S2_13
                        + O2_12*S2_12 + O2_22*S2_22 + O2_23*S2_23
                        + O2_13*S2_13 + O2_23*S2_23 + O2_33*S2_33;

            int feat_base = idx * 5;
            features[feat_base + 0] = lam1;
            features[feat_base + 1] = lam2;
            features[feat_base + 2] = lam3;
            features[feat_base + 3] = lam4;
            features[feat_base + 4] = lam5;
        }
    } else {
        // 2D path (Nz == 1): same as TBNN 2D invariant computation
        #pragma omp target teams distribute parallel for \
            map(present: dudx[0:total_cells], dudy[0:total_cells], \
                         dvdx[0:total_cells], dvdy[0:total_cells], \
                         k[0:total_cells], omega[0:total_cells], \
                         features[0:(n_cells*5)])
        for (int idx = 0; idx < n_cells; ++idx) {
            const int ii = idx % Nx;
            const int jj = idx / Nx;
            const int i = ii + Ng;
            const int j = jj + Ng;
            const int idx_cell = j * cell_stride + i;

            double dudx_v = dudx[idx_cell];
            double dudy_v = dudy[idx_cell];
            double dvdx_v = dvdx[idx_cell];
            double dvdy_v = dvdy[idx_cell];

            double Sxx = dudx_v;
            double Syy = dvdy_v;
            double Sxy = 0.5 * (dudy_v + dvdx_v);
            double Oxy = 0.5 * (dudy_v - dvdx_v);

            // Non-dimensionalize
            double k_val = k[idx_cell];
            double omega_val = omega[idx_cell];
            double eps = C_mu * k_val * omega_val;
            double k_safe = (k_val > 1e-10) ? k_val : 1e-10;
            double eps_safe = (eps > 1e-20) ? eps : 1e-20;
            double tau = k_safe / eps_safe;

            double Sh11 = Sxx * tau, Sh22 = Syy * tau;
            double Sh12 = Sxy * tau;
            double Oh12 = Oxy * tau;

            // lambda_1 = tr(S_hat^2) (2D: Szz=0, Sxz=Syz=0)
            double lam1 = Sh11*Sh11 + Sh22*Sh22 + 2.0*Sh12*Sh12;

            // lambda_2 = tr(Omega_hat^2) (2D: only Oxy nonzero)
            double lam2 = -2.0*Oh12*Oh12;

            // lambda_3 = tr(S_hat^3) (2D: S33=0)
            // In 2D: tr(S^3) = S11^3 + 3*S11*S12^2 + 3*S22*S12^2 + S22^3
            //       = S11*(S11^2+S12^2) + S12*(S11*S12+S12*S22) + ...
            // Compute directly:
            double S3_11 = Sh11*Sh11*Sh11 + Sh12*Sh12*Sh11 + Sh11*Sh12*Sh12 + Sh12*Sh22*Sh12;
            double S3_22 = Sh12*Sh11*Sh12 + Sh22*Sh12*Sh12 + Sh12*Sh12*Sh22 + Sh22*Sh22*Sh22;
            double lam3 = S3_11 + S3_22;

            // lambda_4 = tr(Omega_hat^2 * S_hat) (2D)
            // O^2 in 2D: O2_11 = -Oh12^2, O2_22 = -Oh12^2, O2_12 = 0
            double O2_diag = -Oh12*Oh12;
            double lam4 = O2_diag * Sh11 + O2_diag * Sh22;

            // lambda_5 = tr(Omega_hat^2 * S_hat^2) (2D)
            double S2_11 = Sh11*Sh11 + Sh12*Sh12;
            double S2_22 = Sh12*Sh12 + Sh22*Sh22;
            double lam5 = O2_diag * S2_11 + O2_diag * S2_22;

            int feat_base = idx * 5;
            features[feat_base + 0] = lam1;
            features[feat_base + 1] = lam2;
            features[feat_base + 2] = lam3;
            features[feat_base + 3] = lam4;
            features[feat_base + 4] = lam5;
        }
    }
#else
    (void)dudx; (void)dudy; (void)dvdx; (void)dvdy;
    (void)k; (void)omega;
    (void)u_face; (void)v_face; (void)w_face; (void)features;
    (void)Nx; (void)Ny; (void)Nz; (void)Ng;
    (void)cell_stride; (void)cell_plane_stride;
    (void)u_stride; (void)u_plane_stride;
    (void)v_stride; (void)v_plane_stride;
    (void)w_stride; (void)w_plane_stride;
    (void)total_cells; (void)u_total; (void)v_total; (void)w_total;
    (void)dx; (void)dy; (void)dz;
#endif
}

// ============================================================================
// GPU Kernel: Postprocess MLP outputs to ghosted field
// ============================================================================
void postprocess_mlp_outputs_gpu(
    const double* nn_outputs,
    double* nu_t_field,
    int Nx, int Ny, int Nz, int Ng,
    int stride,
    int plane_stride,
    int total_field_size,
    double nu_t_max)
{
    NVTX_SCOPE_NN("kernel:postprocess_mlp");

    const int n_cells = Nx * Ny * Nz;

#ifdef USE_GPU_OFFLOAD
    #pragma omp target teams distribute parallel for \
        map(present: nn_outputs[0:n_cells], nu_t_field[0:total_field_size])
    for (int idx = 0; idx < n_cells; ++idx) {
        const int ii = idx % Nx;
        const int jj = (idx / Nx) % Ny;
        const int kk = idx / (Nx * Ny);
        const int i = ii + Ng;
        const int j = jj + Ng;
        const int k = kk + Ng;
        const int idx_out = k * plane_stride + j * stride + i;

        // Get NN prediction
        double nu_t_val = nn_outputs[idx];

        // Apply realizability and clipping
        if (nu_t_val != nu_t_val || nu_t_val < 0.0) {  // NaN or negative
            nu_t_val = 0.0;
        }
        if (nu_t_val > nu_t_max) {
            nu_t_val = nu_t_max;
        }

        nu_t_field[idx_out] = nu_t_val;
    }
#else
    (void)nn_outputs; (void)nu_t_field;
    (void)Nx; (void)Ny; (void)Nz; (void)Ng;
    (void)stride; (void)plane_stride; (void)total_field_size; (void)nu_t_max;
#endif
}

// ============================================================================
// GPU Kernel: Compute TBNN features and tensor basis
// ============================================================================
void compute_tbnn_features_gpu(
    const double* dudx, const double* dudy,
    const double* dvdx, const double* dvdy,
    const double* k, const double* omega,
    const double* wall_distance,
    double* features,
    double* basis,
    int Nx, int Ny, int Nz, int Ng,
    int cell_stride, int cell_plane_stride,
    int total_cells,
    double nu, double delta)
{
    NVTX_SCOPE_TURB("kernel:tbnn_features");

    const int n_cells = Nx * Ny * Nz;

#ifdef USE_GPU_OFFLOAD
    const double C_mu = 0.09;

    #pragma omp target teams distribute parallel for \
        map(present: dudx[0:total_cells], dudy[0:total_cells], \
                     dvdx[0:total_cells], dvdy[0:total_cells], \
                     k[0:total_cells], omega[0:total_cells], \
                     wall_distance[0:total_cells], \
                     features[0:(n_cells*5)], basis[0:(n_cells*60)])
    for (int idx = 0; idx < n_cells; ++idx) {
        const int ii = idx % Nx;
        const int jj = (idx / Nx) % Ny;
        const int kk = idx / (Nx * Ny);
        const int i = ii + Ng;
        const int j = jj + Ng;
        const int kp = kk + Ng;
        const int idx_cell = kp * cell_plane_stride + j * cell_stride + i;

        // Get gradients (computed by gradient kernel for all z-planes)
        // Note: only dudx/dudy/dvdx/dvdy available in this kernel signature.
        // z-gradients are zero (will be non-zero when 3D gradient pointers are added).
        double dudx_v = dudx[idx_cell];
        double dudy_v = dudy[idx_cell];
        double dvdx_v = dvdx[idx_cell];
        double dvdy_v = dvdy[idx_cell];

        // Build 3D strain tensor S_ij = 0.5*(du_i/dx_j + du_j/dx_i)
        double Sn[3][3];
        Sn[0][0] = dudx_v;
        Sn[1][1] = dvdy_v;
        Sn[2][2] = 0.0;  // dwdz
        Sn[0][1] = 0.5 * (dudy_v + dvdx_v);
        Sn[0][2] = 0.0;  // 0.5*(dudz + dwdx)
        Sn[1][2] = 0.0;  // 0.5*(dvdz + dwdy)
        Sn[1][0] = Sn[0][1];
        Sn[2][0] = Sn[0][2];
        Sn[2][1] = Sn[1][2];

        // Build 3D rotation tensor O_ij = 0.5*(du_i/dx_j - du_j/dx_i)
        double On[3][3];
        On[0][0] = 0.0; On[1][1] = 0.0; On[2][2] = 0.0;
        On[0][1] = 0.5 * (dudy_v - dvdx_v);
        On[0][2] = 0.0;
        On[1][2] = 0.0;
        On[1][0] = -On[0][1];
        On[2][0] = -On[0][2];
        On[2][1] = -On[1][2];

        // Magnitudes (Frobenius norms)
        double S_mag_sq = 0.0, O_mag_sq = 0.0;
        for (int p = 0; p < 3; ++p)
            for (int q = 0; q < 3; ++q) {
                S_mag_sq += Sn[p][q] * Sn[p][q];
                O_mag_sq += On[p][q] * On[p][q];
            }
        double S_mag = sqrt(S_mag_sq);
        double Omega_mag = sqrt(O_mag_sq);

        // Get k, omega, epsilon
        double k_val = k[idx_cell];
        double omega_val = omega[idx_cell];
        double eps = C_mu * k_val * omega_val;

        double k_safe = (k_val > numerics::K_FLOOR) ? k_val : numerics::K_FLOOR;
        double eps_safe = (eps > numerics::EPS_FLOOR) ? eps : numerics::EPS_FLOOR;
        double tau = k_safe / eps_safe;

        double S_norm = S_mag * tau;
        double Omega_norm = Omega_mag * tau;

        // Normalize S and O by tau
        for (int p = 0; p < 3; ++p)
            for (int q = 0; q < 3; ++q) {
                Sn[p][q] *= tau;
                On[p][q] *= tau;
            }

        // Compute tr(Sn^2) and tr(On^2) for features
        double trSnSn = 0.0, trOnOn = 0.0;
        for (int p = 0; p < 3; ++p)
            for (int q = 0; q < 3; ++q) {
                trSnSn += Sn[p][q] * Sn[q][p];
                trOnOn += On[p][q] * On[q][p];
            }

        // Features (5 values)
        int feat_base = idx * 5;
        features[feat_base + 0] = S_norm * S_norm;
        features[feat_base + 1] = Omega_norm * Omega_norm;
        features[feat_base + 2] = trSnSn;
        features[feat_base + 3] = -trOnOn;
        features[feat_base + 4] = wall_distance[idx_cell] / delta;

        // ========== Pope (1975) 10 tensor basis, 6 symmetric components each ==========
        int basis_base = idx * 60;

        // Precompute matrix products
        double S2[3][3], O2[3][3], SO[3][3], OS[3][3];
        for (int p = 0; p < 3; ++p)
            for (int q = 0; q < 3; ++q) {
                S2[p][q] = 0.0; O2[p][q] = 0.0;
                SO[p][q] = 0.0; OS[p][q] = 0.0;
                for (int r = 0; r < 3; ++r) {
                    S2[p][q] += Sn[p][r] * Sn[r][q];
                    O2[p][q] += On[p][r] * On[r][q];
                    SO[p][q] += Sn[p][r] * On[r][q];
                    OS[p][q] += On[p][r] * Sn[r][q];
                }
            }

        double trS2 = S2[0][0] + S2[1][1] + S2[2][2];
        double trO2 = O2[0][0] + O2[1][1] + O2[2][2];

        // T1 = S
        basis[basis_base + 0] = Sn[0][0]; basis[basis_base + 1] = Sn[0][1]; basis[basis_base + 2] = Sn[0][2];
        basis[basis_base + 3] = Sn[1][1]; basis[basis_base + 4] = Sn[1][2]; basis[basis_base + 5] = Sn[2][2];

        // T2 = SO - OS
        basis[basis_base + 6]  = SO[0][0] - OS[0][0]; basis[basis_base + 7]  = SO[0][1] - OS[0][1]; basis[basis_base + 8]  = SO[0][2] - OS[0][2];
        basis[basis_base + 9]  = SO[1][1] - OS[1][1]; basis[basis_base + 10] = SO[1][2] - OS[1][2]; basis[basis_base + 11] = SO[2][2] - OS[2][2];

        // T3 = S^2 - (1/3)*tr(S^2)*I
        double trS2_3 = trS2 / 3.0;
        basis[basis_base + 12] = S2[0][0] - trS2_3; basis[basis_base + 13] = S2[0][1]; basis[basis_base + 14] = S2[0][2];
        basis[basis_base + 15] = S2[1][1] - trS2_3; basis[basis_base + 16] = S2[1][2]; basis[basis_base + 17] = S2[2][2] - trS2_3;

        // T4 = O^2 - (1/3)*tr(O^2)*I
        double trO2_3 = trO2 / 3.0;
        basis[basis_base + 18] = O2[0][0] - trO2_3; basis[basis_base + 19] = O2[0][1]; basis[basis_base + 20] = O2[0][2];
        basis[basis_base + 21] = O2[1][1] - trO2_3; basis[basis_base + 22] = O2[1][2]; basis[basis_base + 23] = O2[2][2] - trO2_3;

        // T5 = OS^2 - S^2O
        double OS2[3][3], S2O[3][3];
        for (int p = 0; p < 3; ++p)
            for (int q = 0; q < 3; ++q) {
                OS2[p][q] = 0.0; S2O[p][q] = 0.0;
                for (int r = 0; r < 3; ++r) {
                    OS2[p][q] += On[p][r] * S2[r][q];
                    S2O[p][q] += S2[p][r] * On[r][q];
                }
            }
        basis[basis_base + 24] = OS2[0][0] - S2O[0][0]; basis[basis_base + 25] = OS2[0][1] - S2O[0][1]; basis[basis_base + 26] = OS2[0][2] - S2O[0][2];
        basis[basis_base + 27] = OS2[1][1] - S2O[1][1]; basis[basis_base + 28] = OS2[1][2] - S2O[1][2]; basis[basis_base + 29] = OS2[2][2] - S2O[2][2];

        // T6 = O^2*S + S*O^2 - (2/3)*tr(S*O^2)*I
        double O2S[3][3], SO2[3][3];
        for (int p = 0; p < 3; ++p)
            for (int q = 0; q < 3; ++q) {
                O2S[p][q] = 0.0; SO2[p][q] = 0.0;
                for (int r = 0; r < 3; ++r) {
                    O2S[p][q] += O2[p][r] * Sn[r][q];
                    SO2[p][q] += Sn[p][r] * O2[r][q];
                }
            }
        double trSO2 = SO2[0][0] + SO2[1][1] + SO2[2][2];
        double trSO2_23 = 2.0 * trSO2 / 3.0;
        basis[basis_base + 30] = O2S[0][0] + SO2[0][0] - trSO2_23; basis[basis_base + 31] = O2S[0][1] + SO2[0][1]; basis[basis_base + 32] = O2S[0][2] + SO2[0][2];
        basis[basis_base + 33] = O2S[1][1] + SO2[1][1] - trSO2_23; basis[basis_base + 34] = O2S[1][2] + SO2[1][2]; basis[basis_base + 35] = O2S[2][2] + SO2[2][2] - trSO2_23;

        // T7 = O*(S*O^2) - (O^2*S)*O
        double OSO2[3][3], O2SO[3][3];
        for (int p = 0; p < 3; ++p)
            for (int q = 0; q < 3; ++q) {
                OSO2[p][q] = 0.0; O2SO[p][q] = 0.0;
                for (int r = 0; r < 3; ++r) {
                    OSO2[p][q] += On[p][r] * SO2[r][q];
                    O2SO[p][q] += O2S[p][r] * On[r][q];
                }
            }
        basis[basis_base + 36] = OSO2[0][0] - O2SO[0][0]; basis[basis_base + 37] = OSO2[0][1] - O2SO[0][1]; basis[basis_base + 38] = OSO2[0][2] - O2SO[0][2];
        basis[basis_base + 39] = OSO2[1][1] - O2SO[1][1]; basis[basis_base + 40] = OSO2[1][2] - O2SO[1][2]; basis[basis_base + 41] = OSO2[2][2] - O2SO[2][2];

        // T8 = (S*O)*S^2 - (S^2*O)*S
        double SOS2[3][3], S2OS[3][3];
        for (int p = 0; p < 3; ++p)
            for (int q = 0; q < 3; ++q) {
                SOS2[p][q] = 0.0; S2OS[p][q] = 0.0;
                for (int r = 0; r < 3; ++r) {
                    SOS2[p][q] += SO[p][r] * S2[r][q];
                    S2OS[p][q] += S2O[p][r] * Sn[r][q];
                }
            }
        basis[basis_base + 42] = SOS2[0][0] - S2OS[0][0]; basis[basis_base + 43] = SOS2[0][1] - S2OS[0][1]; basis[basis_base + 44] = SOS2[0][2] - S2OS[0][2];
        basis[basis_base + 45] = SOS2[1][1] - S2OS[1][1]; basis[basis_base + 46] = SOS2[1][2] - S2OS[1][2]; basis[basis_base + 47] = SOS2[2][2] - S2OS[2][2];

        // T9 = O^2*S^2 + S^2*O^2 - (2/3)*tr(S^2*O^2)*I
        double O2S2[3][3], S2O2[3][3];
        for (int p = 0; p < 3; ++p)
            for (int q = 0; q < 3; ++q) {
                O2S2[p][q] = 0.0; S2O2[p][q] = 0.0;
                for (int r = 0; r < 3; ++r) {
                    O2S2[p][q] += O2[p][r] * S2[r][q];
                    S2O2[p][q] += S2[p][r] * O2[r][q];
                }
            }
        double trS2O2 = S2O2[0][0] + S2O2[1][1] + S2O2[2][2];
        double trS2O2_23 = 2.0 * trS2O2 / 3.0;
        basis[basis_base + 48] = O2S2[0][0] + S2O2[0][0] - trS2O2_23; basis[basis_base + 49] = O2S2[0][1] + S2O2[0][1]; basis[basis_base + 50] = O2S2[0][2] + S2O2[0][2];
        basis[basis_base + 51] = O2S2[1][1] + S2O2[1][1] - trS2O2_23; basis[basis_base + 52] = O2S2[1][2] + S2O2[1][2]; basis[basis_base + 53] = O2S2[2][2] + S2O2[2][2] - trS2O2_23;

        // T10 = O*(S^2*O^2) - (O^2*S^2)*O
        double OS2O2[3][3], O2S2O[3][3];
        for (int p = 0; p < 3; ++p)
            for (int q = 0; q < 3; ++q) {
                OS2O2[p][q] = 0.0; O2S2O[p][q] = 0.0;
                for (int r = 0; r < 3; ++r) {
                    OS2O2[p][q] += On[p][r] * S2O2[r][q];
                    O2S2O[p][q] += O2S2[p][r] * On[r][q];
                }
            }
        basis[basis_base + 54] = OS2O2[0][0] - O2S2O[0][0]; basis[basis_base + 55] = OS2O2[0][1] - O2S2O[0][1]; basis[basis_base + 56] = OS2O2[0][2] - O2S2O[0][2];
        basis[basis_base + 57] = OS2O2[1][1] - O2S2O[1][1]; basis[basis_base + 58] = OS2O2[1][2] - O2S2O[1][2]; basis[basis_base + 59] = OS2O2[2][2] - O2S2O[2][2];
    }

    (void)nu;
#else
    (void)dudx; (void)dudy; (void)dvdx; (void)dvdy;
    (void)k; (void)omega; (void)wall_distance;
    (void)features; (void)basis;
    (void)Nx; (void)Ny; (void)Nz; (void)Ng;
    (void)cell_stride; (void)cell_plane_stride; (void)total_cells;
    (void)nu; (void)delta;
#endif
}

// ============================================================================
// GPU Kernel: Postprocess NN outputs
// ============================================================================
void postprocess_nn_outputs_gpu(
    const double* nn_outputs,
    const double* basis,
    const double* k,
    const double* dudx, const double* dudy,
    const double* dvdx, const double* dvdy,
    double* nu_t,
    double* tau_xx, double* tau_xy, double* tau_yy,
    int Nx, int Ny, int Nz, int Ng,
    int cell_stride, int cell_plane_stride,
    int total_cells,
    int output_dim,
    double nu_ref)
{
    NVTX_SCOPE_NN("kernel:postprocess_nn");

    const int n_cells = Nx * Ny * Nz;

#ifdef USE_GPU_OFFLOAD
    const int NUM_BASIS = 10;
    const bool compute_tau = (tau_xx != nullptr);

    if (!compute_tau) {
        #pragma omp target teams distribute parallel for \
            map(present: nn_outputs[0:(n_cells*output_dim)], basis[0:(n_cells*60)], \
                         k[0:total_cells], dudx[0:total_cells], dudy[0:total_cells], \
                         dvdx[0:total_cells], dvdy[0:total_cells], nu_t[0:total_cells])
        for (int idx = 0; idx < n_cells; ++idx) {
            const int ii = idx % Nx;
            const int jj = (idx / Nx) % Ny;
            const int kk = idx / (Nx * Ny);
            const int i = ii + Ng;
            const int j = jj + Ng;
            const int kp = kk + Ng;
            const int idx_cell = kp * cell_plane_stride + j * cell_stride + i;

            // Extract G coefficients from NN output
            double G[10] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            const int out_base = idx * output_dim;
            for (int n = 0; n < NUM_BASIS && n < output_dim; ++n) {
                G[n] = nn_outputs[out_base + n];
            }

            const int basis_base = idx * 60;

            // Contract: b_ij = sum_n G[n] * T[n][c]
            // Components: XX=0, XY=1, XZ=2, YY=3, YZ=4, ZZ=5
            double b_xx = 0.0, b_xy = 0.0, b_xz = 0.0;
            double b_yy = 0.0, b_yz = 0.0, b_zz = 0.0;
            for (int n = 0; n < NUM_BASIS; ++n) {
                b_xx += G[n] * basis[basis_base + n*6 + 0];
                b_xy += G[n] * basis[basis_base + n*6 + 1];
                b_xz += G[n] * basis[basis_base + n*6 + 2];
                b_yy += G[n] * basis[basis_base + n*6 + 3];
                b_yz += G[n] * basis[basis_base + n*6 + 4];
                b_zz += G[n] * basis[basis_base + n*6 + 5];
            }

            const double k_val = k[idx_cell];
            const double dudy_v = dudy[idx_cell];
            const double dvdx_v = dvdx[idx_cell];
            const double Sxy = 0.5 * (dudy_v + dvdx_v);

            double nu_t_val = 0.0;
            if (fabs(Sxy) > 1e-10) {
                nu_t_val = fabs(-b_xy * k_val / Sxy);
            } else {
                const double dudx_v = dudx[idx_cell];
                const double dvdy_v = dvdy[idx_cell];
                const double Sxx = dudx_v;
                const double Syy = dvdy_v;
                const double S_mag = sqrt(Sxx*Sxx + Syy*Syy + 2.0*Sxy*Sxy);
                if (S_mag > 1e-10) {
                    const double b_mag = sqrt(b_xx*b_xx + 2.0*b_xy*b_xy + 2.0*b_xz*b_xz
                                            + b_yy*b_yy + 2.0*b_yz*b_yz + b_zz*b_zz);
                    nu_t_val = k_val * b_mag / S_mag;
                }
            }

            const double max_nu_t = 10.0 * nu_ref;
            nu_t_val = (nu_t_val > 0.0) ? nu_t_val : 0.0;
            nu_t_val = (nu_t_val < max_nu_t) ? nu_t_val : max_nu_t;
            if (nu_t_val != nu_t_val || nu_t_val > 1e30) {
                nu_t_val = 0.0;
            }

            nu_t[idx_cell] = nu_t_val;
        }
    } else {
        #pragma omp target teams distribute parallel for \
            map(present: nn_outputs[0:(n_cells*output_dim)], basis[0:(n_cells*60)], \
                         k[0:total_cells], dudx[0:total_cells], dudy[0:total_cells], \
                         dvdx[0:total_cells], dvdy[0:total_cells], nu_t[0:total_cells], \
                         tau_xx[0:total_cells], tau_xy[0:total_cells], tau_yy[0:total_cells])
        for (int idx = 0; idx < n_cells; ++idx) {
            const int ii = idx % Nx;
            const int jj = (idx / Nx) % Ny;
            const int kk = idx / (Nx * Ny);
            const int i = ii + Ng;
            const int j = jj + Ng;
            const int kp = kk + Ng;
            const int idx_cell = kp * cell_plane_stride + j * cell_stride + i;

            double G[10] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            const int out_base = idx * output_dim;
            for (int n = 0; n < NUM_BASIS && n < output_dim; ++n) {
                G[n] = nn_outputs[out_base + n];
            }

            const int basis_base = idx * 60;

            double b_xx = 0.0, b_xy = 0.0, b_xz = 0.0;
            double b_yy = 0.0, b_yz = 0.0, b_zz = 0.0;
            for (int n = 0; n < NUM_BASIS; ++n) {
                b_xx += G[n] * basis[basis_base + n*6 + 0];
                b_xy += G[n] * basis[basis_base + n*6 + 1];
                b_xz += G[n] * basis[basis_base + n*6 + 2];
                b_yy += G[n] * basis[basis_base + n*6 + 3];
                b_yz += G[n] * basis[basis_base + n*6 + 4];
                b_zz += G[n] * basis[basis_base + n*6 + 5];
            }

            const double k_val = k[idx_cell];
            const double k_safe = (k_val > 0.0) ? k_val : 0.0;
            tau_xx[idx_cell] = 2.0 * k_safe * (b_xx + 1.0/3.0);
            tau_xy[idx_cell] = 2.0 * k_safe * b_xy;
            tau_yy[idx_cell] = 2.0 * k_safe * (b_yy + 1.0/3.0);
            // tau_xz, tau_yz, tau_zz will be written when device_view has those pointers (Task 4)

            const double dudy_v = dudy[idx_cell];
            const double dvdx_v = dvdx[idx_cell];
            const double Sxy = 0.5 * (dudy_v + dvdx_v);

            double nu_t_val = 0.0;
            if (fabs(Sxy) > 1e-10) {
                nu_t_val = fabs(-b_xy * k_val / Sxy);
            } else {
                const double dudx_v = dudx[idx_cell];
                const double dvdy_v = dvdy[idx_cell];
                const double Sxx = dudx_v;
                const double Syy = dvdy_v;
                const double S_mag = sqrt(Sxx*Sxx + Syy*Syy + 2.0*Sxy*Sxy);
                if (S_mag > 1e-10) {
                    const double b_mag = sqrt(b_xx*b_xx + 2.0*b_xy*b_xy + 2.0*b_xz*b_xz
                                            + b_yy*b_yy + 2.0*b_yz*b_yz + b_zz*b_zz);
                    nu_t_val = k_val * b_mag / S_mag;
                }
            }

            const double max_nu_t = 10.0 * nu_ref;
            nu_t_val = (nu_t_val > 0.0) ? nu_t_val : 0.0;
            nu_t_val = (nu_t_val < max_nu_t) ? nu_t_val : max_nu_t;
            if (nu_t_val != nu_t_val || nu_t_val > 1e30) {
                nu_t_val = 0.0;
            }

            nu_t[idx_cell] = nu_t_val;
        }
    }
#else
    (void)nn_outputs; (void)basis; (void)k;
    (void)dudx; (void)dudy; (void)dvdx; (void)dvdy;
    (void)nu_t; (void)tau_xx; (void)tau_xy; (void)tau_yy;
    (void)Nx; (void)Ny; (void)Nz; (void)Ng;
    (void)cell_stride; (void)cell_plane_stride; (void)total_cells;
    (void)output_dim; (void)nu_ref;
#endif
}

// ============================================================================
// GPU Kernel: Boussinesq k-omega closure
// ============================================================================
void compute_boussinesq_closure_gpu(
    const double* k,
    const double* omega,
    double* nu_t,
    int Nx, int Ny, int Nz,
    int Ng,
    int stride,
    int cell_plane_stride,
    int total_size,
    double nu,
    double k_min, double omega_min,
    double nu_t_max)
{
    NVTX_SCOPE_CLOSURE("kernel:boussinesq_closure");

#ifdef USE_GPU_OFFLOAD
    const int n_cells = Nx * Ny * Nz;
    const int NxNy = Nx * Ny;

    // CRITICAL: map(present:...) for solver-managed device buffers
    #pragma omp target teams distribute parallel for \
        map(present: k[0:total_size], omega[0:total_size], nu_t[0:total_size])
    for (int idx = 0; idx < n_cells; ++idx) {
        // Convert flat index to (i,j,k) including ghost cells
        const int i = idx % Nx + Ng;
        const int j = (idx / Nx) % Ny + Ng;
        const int kz = (Nz > 1) ? (idx / NxNy + Ng) : 0;
        const int cell_idx = kz * cell_plane_stride + j * stride + i;
        
        // Read k and omega
        double k_val = k[cell_idx];
        double omega_val = omega[cell_idx];
        
        // Clamp to minimum values
        k_val = (k_val > k_min) ? k_val : k_min;
        omega_val = (omega_val > omega_min) ? omega_val : omega_min;
        
        // Boussinesq closure: ν_t = k / ω
        double nu_t_val = k_val / omega_val;
        
        // Realizability constraint: ν_t ≥ 0
        nu_t_val = (nu_t_val > 0.0) ? nu_t_val : 0.0;
        
        // Upper limit: ν_t ≤ nu_t_max * ν
        double max_val = nu_t_max * nu;
        nu_t_val = (nu_t_val < max_val) ? nu_t_val : max_val;
        
        nu_t[cell_idx] = nu_t_val;
    }
#else
    (void)k; (void)omega; (void)nu_t;
    (void)Nx; (void)Ny; (void)Nz; (void)Ng; (void)stride; (void)cell_plane_stride; (void)total_size;
    (void)nu; (void)k_min; (void)omega_min; (void)nu_t_max;
#endif
}

// ============================================================================
// GPU Kernel: SST k-omega closure
// ============================================================================
void compute_sst_closure_gpu(
    const double* k,
    const double* omega,
    const double* dudx,
    const double* dudy,
    const double* dvdx,
    const double* dvdy,
    const double* wall_distance,
    double* nu_t,
    int Nx, int Ny, int Nz,
    int Ng,
    int stride,
    int cell_plane_stride,
    int total_size,
    int wall_dist_size,
    double nu,
    double a1,
    double beta_star,
    double k_min, double omega_min,
    double nu_t_max)
{
    NVTX_SCOPE_CLOSURE("kernel:sst_closure");

#ifdef USE_GPU_OFFLOAD
    const int n_cells = Nx * Ny * Nz;
    const int NxNy = Nx * Ny;

    // CRITICAL: map(present:...) for solver-managed device buffers
    // wall_distance has same layout as k/omega/nu_t (full field with ghosts)
    #pragma omp target teams distribute parallel for \
        map(present: k[0:total_size], omega[0:total_size], \
                     dudx[0:total_size], dudy[0:total_size], \
                     dvdx[0:total_size], dvdy[0:total_size], \
                     wall_distance[0:wall_dist_size], \
                     nu_t[0:total_size])
    for (int idx = 0; idx < n_cells; ++idx) {
        // Convert flat index to (i,j,k) including ghost cells
        // For 2D (Nz==1): use plane 0 (no Ng offset) to match solver's 2D kernels
        const int i = idx % Nx + Ng;
        const int j = (idx / Nx) % Ny + Ng;
        const int kz = (Nz > 1) ? (idx / NxNy + Ng) : 0;
        const int cell_idx = kz * cell_plane_stride + j * stride + i;

        // Read fields (all use stride-based indexing with ghosts)
        double k_val = k[cell_idx];
        double omega_val = omega[cell_idx];
        double y_wall = wall_distance[cell_idx];  // Wall distance uses same indexing as k/omega

        // Clamp to minimum values
        k_val = (k_val > k_min) ? k_val : k_min;
        omega_val = (omega_val > omega_min) ? omega_val : omega_min;
        double y_safe = (y_wall > 1e-10) ? y_wall : 1e-10;

        // Strain rate magnitude from gradients
        double Sxx = dudx[cell_idx];
        double Syy = dvdy[cell_idx];
        double Sxy = 0.5 * (dudy[cell_idx] + dvdx[cell_idx]);
        double S_mag = sqrt(2.0 * (Sxx*Sxx + Syy*Syy + 2.0*Sxy*Sxy));

        // F2 blending function for SST
        // arg2 = max(2√k / (β*ωy), 500ν / (y²ω))
        double sqrt_k = sqrt(k_val);
        double term1 = 2.0 * sqrt_k / (beta_star * omega_val * y_safe);
        double term2 = 500.0 * nu / (y_safe * y_safe * omega_val);
        double arg2 = (term1 > term2) ? term1 : term2;
        double F2 = tanh(arg2 * arg2);

        // SST eddy viscosity: ν_t = a₁k / max(a₁ω, SF₂)
        double denom = a1 * omega_val;
        double SF2 = S_mag * F2;
        denom = (denom > SF2) ? denom : SF2;

        // Prevent division by zero
        denom = (denom > 1e-20) ? denom : 1e-20;

        double nu_t_val = a1 * k_val / denom;

        // Realizability: ν_t ≥ 0
        nu_t_val = (nu_t_val > 0.0) ? nu_t_val : 0.0;

        // Upper limit: ν_t ≤ nu_t_max * ν
        double max_val = nu_t_max * nu;
        nu_t_val = (nu_t_val < max_val) ? nu_t_val : max_val;

        nu_t[cell_idx] = nu_t_val;
    }
#else
    (void)k; (void)omega; (void)dudx; (void)dudy; (void)dvdx; (void)dvdy;
    (void)wall_distance; (void)nu_t;
    (void)Nx; (void)Ny; (void)Nz; (void)Ng; (void)stride; (void)cell_plane_stride;
    (void)total_size; (void)wall_dist_size;
    (void)nu; (void)a1; (void)beta_star;
    (void)k_min; (void)omega_min; (void)nu_t_max;
#endif
}

// ============================================================================
// GPU Kernel: k-omega transport step
// ============================================================================
void komega_transport_step_gpu(
    const double* u, const double* v,
    double* k, double* omega,
    const double* nu_t_prev,
    int Nx, int Ny, int Nz, int Ng,
    int stride,
    int cell_plane_stride,
    int u_stride, int v_stride,
    int u_plane_stride, int v_plane_stride,
    int total_size,
    int vel_u_size, int vel_v_size,
    double dx, double dy, double dt,
    double nu, double sigma_k, double sigma_omega,
    double beta, double beta_star, double alpha,
    double k_min, double k_max,
    double omega_min, double omega_max)
{
    NVTX_SCOPE_TURB("kernel:komega_transport");

#ifdef USE_GPU_OFFLOAD
    const int n_cells = Nx * Ny * Nz;
    const int NxNy = Nx * Ny;
    const double inv_2dx = 1.0 / (2.0 * dx);
    const double inv_2dy = 1.0 / (2.0 * dy);
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);

    // CRITICAL: map(present:...) for all solver-managed device buffers
    #pragma omp target teams distribute parallel for \
        map(present: u[0:vel_u_size], v[0:vel_v_size], \
                     k[0:total_size], omega[0:total_size], \
                     nu_t_prev[0:total_size])
    for (int idx = 0; idx < n_cells; ++idx) {
        // Convert flat index to (i,j,k) including ghost cells
        const int i = idx % Nx + Ng;
        const int j = (idx / Nx) % Ny + Ng;
        const int kk = idx / NxNy;
        const int kz = (Nz > 1) ? (kk + Ng) : 0;
        const int z_cell_off = kz * cell_plane_stride;
        const int z_u_off = kz * u_plane_stride;
        const int z_v_off = kz * v_plane_stride;
        const int cell_idx = z_cell_off + j * stride + i;

        // Read current values (nu_t_prev now uses same ghost+stride layout as k/omega)
        double k_val = k[cell_idx];
        double omega_val = omega[cell_idx];
        double nu_t_val = nu_t_prev[cell_idx];  // Now uses ghost+stride indexing

        // Clamp to valid range
        k_val = (k_val > k_min) ? k_val : k_min;
        omega_val = (omega_val > omega_min) ? omega_val : omega_min;
        nu_t_val = (nu_t_val > 0.0) ? nu_t_val : 0.0;

        // Compute velocity gradients from staggered MAC grid
        // u is at x-faces, v is at y-faces
        const int u_idx_ip = z_u_off + j * u_stride + (i + 1);
        const int u_idx_im = z_u_off + j * u_stride + (i - 1);
        const int u_idx_jp = z_u_off + (j + 1) * u_stride + i;
        const int u_idx_jm = z_u_off + (j - 1) * u_stride + i;

        const int v_idx_ip = z_v_off + j * v_stride + (i + 1);
        const int v_idx_im = z_v_off + j * v_stride + (i - 1);
        const int v_idx_jp = z_v_off + (j + 1) * v_stride + i;
        const int v_idx_jm = z_v_off + (j - 1) * v_stride + i;
        
        double dudx_val = (u[u_idx_ip] - u[u_idx_im]) * inv_2dx;
        double dudy_val = (u[u_idx_jp] - u[u_idx_jm]) * inv_2dy;
        double dvdx_val = (v[v_idx_ip] - v[v_idx_im]) * inv_2dx;
        double dvdy_val = (v[v_idx_jp] - v[v_idx_jm]) * inv_2dy;
        
        // Strain rate magnitude
        double Sxx = dudx_val;
        double Syy = dvdy_val;
        double Sxy = 0.5 * (dudy_val + dvdx_val);
        double S2 = 2.0 * (Sxx*Sxx + Syy*Syy + 2.0*Sxy*Sxy);
        
        // Production term: P_k = 2 ν_t |S|²
        double P_k = 2.0 * nu_t_val * S2;
        
        // Effective diffusivities
        double nu_k = nu + sigma_k * nu_t_val;
        double nu_omega_eff = nu + sigma_omega * nu_t_val;
        
        // Get velocity at cell center (approximate from faces)
        double u_c = 0.5 * (u[z_u_off + j * u_stride + i] + u[z_u_off + j * u_stride + (i+1)]);
        double v_c = 0.5 * (v[z_v_off + j * v_stride + i] + v[z_v_off + (j+1) * v_stride + i]);

        // Advection terms (upwind scheme)
        double adv_k, adv_omega;
        const int k_idx_ip = z_cell_off + j * stride + (i + 1);
        const int k_idx_im = z_cell_off + j * stride + (i - 1);
        const int k_idx_jp = z_cell_off + (j + 1) * stride + i;
        const int k_idx_jm = z_cell_off + (j - 1) * stride + i;
        
        if (u_c >= 0.0) {
            adv_k = u_c * (k_val - k[k_idx_im]) / dx;
            adv_omega = u_c * (omega_val - omega[k_idx_im]) / dx;
        } else {
            adv_k = u_c * (k[k_idx_ip] - k_val) / dx;
            adv_omega = u_c * (omega[k_idx_ip] - omega_val) / dx;
        }
        
        if (v_c >= 0.0) {
            adv_k += v_c * (k_val - k[k_idx_jm]) / dy;
            adv_omega += v_c * (omega_val - omega[k_idx_jm]) / dy;
        } else {
            adv_k += v_c * (k[k_idx_jp] - k_val) / dy;
            adv_omega += v_c * (omega[k_idx_jp] - omega_val) / dy;
        }
        
        // Diffusion terms (central differences)
        double diff_k = nu_k * ((k[k_idx_ip] - 2.0*k_val + k[k_idx_im]) * inv_dx2 +
                                (k[k_idx_jp] - 2.0*k_val + k[k_idx_jm]) * inv_dy2);
        
        double diff_omega = nu_omega_eff * ((omega[k_idx_ip] - 2.0*omega_val + omega[k_idx_im]) * inv_dx2 +
                                            (omega[k_idx_jp] - 2.0*omega_val + omega[k_idx_jm]) * inv_dy2);
        
        // Point-implicit: treat destruction terms implicitly for stability
        double source_k = P_k + diff_k - adv_k;
        double sink_k = beta_star * omega_val;
        double source_omega = alpha * (omega_val / k_val) * P_k + diff_omega - adv_omega;
        double sink_omega = beta * omega_val;

        double k_new = (k_val + dt * source_k) / (1.0 + dt * sink_k);
        double omega_new = (omega_val + dt * source_omega) / (1.0 + dt * sink_omega);
        
        // Clip to valid range
        k_new = (k_new > k_min) ? k_new : k_min;
        k_new = (k_new < k_max) ? k_new : k_max;
        omega_new = (omega_new > omega_min) ? omega_new : omega_min;
        omega_new = (omega_new < omega_max) ? omega_new : omega_max;
        
        // Write back
        k[cell_idx] = k_new;
        omega[cell_idx] = omega_new;
    }
#else
    (void)u; (void)v; (void)k; (void)omega; (void)nu_t_prev;
    (void)Nx; (void)Ny; (void)Nz; (void)Ng; (void)stride; (void)cell_plane_stride;
    (void)u_stride; (void)v_stride; (void)u_plane_stride; (void)v_plane_stride;
    (void)total_size; (void)vel_u_size; (void)vel_v_size;
    (void)dx; (void)dy; (void)dt;
    (void)nu; (void)sigma_k; (void)sigma_omega;
    (void)beta; (void)beta_star; (void)alpha;
    (void)k_min; (void)k_max; (void)omega_min; (void)omega_max;
#endif
}

} // namespace gpu_kernels
} // namespace nncfd

