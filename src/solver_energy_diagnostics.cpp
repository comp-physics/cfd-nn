// Energy balance diagnostics for RANSSolver
// Split from solver.cpp to reduce compilation unit size and avoid nvc++ compiler crash
// Uses GPU kernels with is_device_ptr pattern (same as compute_kinetic_energy_device in master)

#include "solver.hpp"
#include <cmath>
#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#include "gpu_utils.hpp"
#endif

namespace nncfd {

//==============================================================================
// Energy balance diagnostics (GPU implementation)
//==============================================================================

double RANSSolver::compute_kinetic_energy() const {
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Ng = mesh_->Nghost;
    const double dV = mesh_->dx * mesh_->dy * (mesh_->is2D() ? 1.0 : mesh_->dz);

    const int u_stride = velocity_.u_stride();
    const int v_stride = velocity_.v_stride();
    const int u_plane = velocity_.u_plane_stride();
    const int v_plane = velocity_.v_plane_stride();
    const int w_stride = velocity_.w_stride();
    const int w_plane = velocity_.w_plane_stride();

    double ke = 0.0;

#ifdef USE_GPU_OFFLOAD
    if (mesh_->is2D()) {
        const int n_cells = Nx * Ny;
        int device = omp_get_default_device();
        const double* u_dev = static_cast<const double*>(omp_get_mapped_ptr(velocity_u_ptr_, device));
        const double* v_dev = static_cast<const double*>(omp_get_mapped_ptr(velocity_v_ptr_, device));

        #pragma omp target teams distribute parallel for reduction(+:ke) is_device_ptr(u_dev, v_dev) \
            firstprivate(Nx, Ny, Ng, u_stride, v_stride, dV)
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + Ng;
            int j = idx / Nx + Ng;
            double u = 0.5 * (u_dev[j * u_stride + i] + u_dev[j * u_stride + (i + 1)]);
            double v = 0.5 * (v_dev[j * v_stride + i] + v_dev[(j + 1) * v_stride + i]);
            ke += 0.5 * (u * u + v * v) * dV;
        }
    } else {
        const int n_cells = Nx * Ny * Nz;
        int device = omp_get_default_device();
        const double* u_dev = static_cast<const double*>(omp_get_mapped_ptr(velocity_u_ptr_, device));
        const double* v_dev = static_cast<const double*>(omp_get_mapped_ptr(velocity_v_ptr_, device));
        const double* w_dev = static_cast<const double*>(omp_get_mapped_ptr(velocity_w_ptr_, device));

        #pragma omp target teams distribute parallel for reduction(+:ke) is_device_ptr(u_dev, v_dev, w_dev) \
            firstprivate(Nx, Ny, Nz, Ng, u_stride, v_stride, w_stride, u_plane, v_plane, w_plane, dV)
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % Ny + Ng;
            int k = idx / (Nx * Ny) + Ng;
            double u = 0.5 * (u_dev[k * u_plane + j * u_stride + i] + u_dev[k * u_plane + j * u_stride + (i + 1)]);
            double v = 0.5 * (v_dev[k * v_plane + j * v_stride + i] + v_dev[k * v_plane + (j + 1) * v_stride + i]);
            double w = 0.5 * (w_dev[k * w_plane + j * w_stride + i] + w_dev[(k + 1) * w_plane + j * w_stride + i]);
            ke += 0.5 * (u * u + v * v + w * w) * dV;
        }
    }
#else
    // CPU fallback for non-GPU builds
    double* u_ptr = velocity_u_ptr_;
    double* v_ptr = velocity_v_ptr_;
    double* w_ptr = velocity_w_ptr_;

    if (mesh_->is2D()) {
        const int n_cells = Nx * Ny;
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + Ng;
            int j = idx / Nx + Ng;
            double u = 0.5 * (u_ptr[j * u_stride + i] + u_ptr[j * u_stride + (i + 1)]);
            double v = 0.5 * (v_ptr[j * v_stride + i] + v_ptr[(j + 1) * v_stride + i]);
            ke += 0.5 * (u * u + v * v) * dV;
        }
    } else {
        const int n_cells = Nx * Ny * Nz;
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % Ny + Ng;
            int k = idx / (Nx * Ny) + Ng;
            double u = 0.5 * (u_ptr[k * u_plane + j * u_stride + i] + u_ptr[k * u_plane + j * u_stride + (i + 1)]);
            double v = 0.5 * (v_ptr[k * v_plane + j * v_stride + i] + v_ptr[k * v_plane + (j + 1) * v_stride + i]);
            double w = 0.5 * (w_ptr[k * w_plane + j * w_stride + i] + w_ptr[(k + 1) * w_plane + j * w_stride + i]);
            ke += 0.5 * (u * u + v * v + w * w) * dV;
        }
    }
#endif

    return ke;
}

double RANSSolver::compute_bulk_velocity() const {
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Ng = mesh_->Nghost;
    const double dx = mesh_->dx;
    const double dy = mesh_->dy;
    const double dz = mesh_->is2D() ? 1.0 : mesh_->dz;

    const int u_stride = velocity_.u_stride();
    const int u_plane = velocity_.u_plane_stride();

    double sum_u = 0.0;

#ifdef USE_GPU_OFFLOAD
    if (mesh_->is2D()) {
        const int n_cells = Nx * Ny;
        int device = omp_get_default_device();
        const double* u_dev = static_cast<const double*>(omp_get_mapped_ptr(velocity_u_ptr_, device));

        #pragma omp target teams distribute parallel for reduction(+:sum_u) is_device_ptr(u_dev) \
            firstprivate(Nx, Ny, Ng, u_stride)
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + Ng;
            int j = idx / Nx + Ng;
            double u_cc = 0.5 * (u_dev[j * u_stride + i] + u_dev[j * u_stride + (i + 1)]);
            sum_u += u_cc;
        }
    } else {
        const int n_cells = Nx * Ny * Nz;
        int device = omp_get_default_device();
        const double* u_dev = static_cast<const double*>(omp_get_mapped_ptr(velocity_u_ptr_, device));

        #pragma omp target teams distribute parallel for reduction(+:sum_u) is_device_ptr(u_dev) \
            firstprivate(Nx, Ny, Nz, Ng, u_stride, u_plane)
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % Ny + Ng;
            int k = idx / (Nx * Ny) + Ng;
            double u_cc = 0.5 * (u_dev[k * u_plane + j * u_stride + i] + u_dev[k * u_plane + j * u_stride + (i + 1)]);
            sum_u += u_cc;
        }
    }
#else
    double* u_ptr = velocity_u_ptr_;
    if (mesh_->is2D()) {
        const int n_cells = Nx * Ny;
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + Ng;
            int j = idx / Nx + Ng;
            double u_cc = 0.5 * (u_ptr[j * u_stride + i] + u_ptr[j * u_stride + (i + 1)]);
            sum_u += u_cc;
        }
    } else {
        const int n_cells = Nx * Ny * Nz;
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % Ny + Ng;
            int k = idx / (Nx * Ny) + Ng;
            double u_cc = 0.5 * (u_ptr[k * u_plane + j * u_stride + i] + u_ptr[k * u_plane + j * u_stride + (i + 1)]);
            sum_u += u_cc;
        }
    }
#endif

    double volume = (mesh_->x_max - mesh_->x_min) *
                    (mesh_->y_max - mesh_->y_min) *
                    (mesh_->is2D() ? 1.0 : (mesh_->z_max - mesh_->z_min));
    double dV = dx * dy * dz;
    return (sum_u * dV) / volume;
}

double RANSSolver::compute_power_input() const {
    // P_in = integral(f_i * u_i) dV
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Ng = mesh_->Nghost;
    const double dV = mesh_->dx * mesh_->dy * (mesh_->is2D() ? 1.0 : mesh_->dz);
    const double fx = fx_, fy = fy_, fz = fz_;

    const int u_stride = velocity_.u_stride();
    const int v_stride = velocity_.v_stride();
    const int u_plane = velocity_.u_plane_stride();
    const int v_plane = velocity_.v_plane_stride();
    const int w_stride = velocity_.w_stride();
    const int w_plane = velocity_.w_plane_stride();

    double power = 0.0;

#ifdef USE_GPU_OFFLOAD
    if (mesh_->is2D()) {
        const int n_cells = Nx * Ny;
        int device = omp_get_default_device();
        const double* u_dev = static_cast<const double*>(omp_get_mapped_ptr(velocity_u_ptr_, device));
        const double* v_dev = static_cast<const double*>(omp_get_mapped_ptr(velocity_v_ptr_, device));

        #pragma omp target teams distribute parallel for reduction(+:power) is_device_ptr(u_dev, v_dev) \
            firstprivate(Nx, Ny, Ng, u_stride, v_stride, dV, fx, fy)
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + Ng;
            int j = idx / Nx + Ng;
            double u_cc = 0.5 * (u_dev[j * u_stride + i] + u_dev[j * u_stride + (i + 1)]);
            double v_cc = 0.5 * (v_dev[j * v_stride + i] + v_dev[(j + 1) * v_stride + i]);
            power += (fx * u_cc + fy * v_cc) * dV;
        }
    } else {
        const int n_cells = Nx * Ny * Nz;
        int device = omp_get_default_device();
        const double* u_dev = static_cast<const double*>(omp_get_mapped_ptr(velocity_u_ptr_, device));
        const double* v_dev = static_cast<const double*>(omp_get_mapped_ptr(velocity_v_ptr_, device));
        const double* w_dev = static_cast<const double*>(omp_get_mapped_ptr(velocity_w_ptr_, device));

        #pragma omp target teams distribute parallel for reduction(+:power) is_device_ptr(u_dev, v_dev, w_dev) \
            firstprivate(Nx, Ny, Nz, Ng, u_stride, v_stride, w_stride, u_plane, v_plane, w_plane, dV, fx, fy, fz)
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % Ny + Ng;
            int k = idx / (Nx * Ny) + Ng;
            double u_cc = 0.5 * (u_dev[k * u_plane + j * u_stride + i] + u_dev[k * u_plane + j * u_stride + (i + 1)]);
            double v_cc = 0.5 * (v_dev[k * v_plane + j * v_stride + i] + v_dev[k * v_plane + (j + 1) * v_stride + i]);
            double w_cc = 0.5 * (w_dev[k * w_plane + j * w_stride + i] + w_dev[(k + 1) * w_plane + j * w_stride + i]);
            power += (fx * u_cc + fy * v_cc + fz * w_cc) * dV;
        }
    }
#else
    double* u_ptr = velocity_u_ptr_;
    double* v_ptr = velocity_v_ptr_;
    double* w_ptr = velocity_w_ptr_;

    if (mesh_->is2D()) {
        const int n_cells = Nx * Ny;
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + Ng;
            int j = idx / Nx + Ng;
            double u_cc = 0.5 * (u_ptr[j * u_stride + i] + u_ptr[j * u_stride + (i + 1)]);
            double v_cc = 0.5 * (v_ptr[j * v_stride + i] + v_ptr[(j + 1) * v_stride + i]);
            power += (fx * u_cc + fy * v_cc) * dV;
        }
    } else {
        const int n_cells = Nx * Ny * Nz;
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % Ny + Ng;
            int k = idx / (Nx * Ny) + Ng;
            double u_cc = 0.5 * (u_ptr[k * u_plane + j * u_stride + i] + u_ptr[k * u_plane + j * u_stride + (i + 1)]);
            double v_cc = 0.5 * (v_ptr[k * v_plane + j * v_stride + i] + v_ptr[k * v_plane + (j + 1) * v_stride + i]);
            double w_cc = 0.5 * (w_ptr[k * w_plane + j * w_stride + i] + w_ptr[(k + 1) * w_plane + j * w_stride + i]);
            power += (fx * u_cc + fy * v_cc + fz * w_cc) * dV;
        }
    }
#endif

    return power;
}

double RANSSolver::compute_viscous_dissipation() const {
    // epsilon = 2*nu * integral(S_ij * S_ij) dV
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Ng = mesh_->Nghost;
    const double dx = mesh_->dx;
    const double dy = mesh_->dy;
    const double dz = mesh_->is2D() ? 1.0 : mesh_->dz;
    const double dV = dx * dy * dz;
    const double nu = config_.nu;

    const int u_stride = velocity_.u_stride();
    const int v_stride = velocity_.v_stride();
    const int u_plane = velocity_.u_plane_stride();
    const int v_plane = velocity_.v_plane_stride();
    const int w_stride = velocity_.w_stride();
    const int w_plane = velocity_.w_plane_stride();

    double dissipation = 0.0;

#ifdef USE_GPU_OFFLOAD
    if (mesh_->is2D()) {
        const int n_cells = Nx * Ny;
        int device = omp_get_default_device();
        const double* u_dev = static_cast<const double*>(omp_get_mapped_ptr(velocity_u_ptr_, device));
        const double* v_dev = static_cast<const double*>(omp_get_mapped_ptr(velocity_v_ptr_, device));

        #pragma omp target teams distribute parallel for reduction(+:dissipation) is_device_ptr(u_dev, v_dev) \
            firstprivate(Nx, Ny, Ng, u_stride, v_stride, dx, dy, dV, nu)
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + Ng;
            int j = idx / Nx + Ng;

            double dudx = (u_dev[j * u_stride + (i + 1)] - u_dev[j * u_stride + i]) / dx;
            double dvdy = (v_dev[(j + 1) * v_stride + i] - v_dev[j * v_stride + i]) / dy;

            double u_jm = 0.5 * (u_dev[(j - 1) * u_stride + i] + u_dev[(j - 1) * u_stride + (i + 1)]);
            double u_jp = 0.5 * (u_dev[(j + 1) * u_stride + i] + u_dev[(j + 1) * u_stride + (i + 1)]);
            double dudy = (u_jp - u_jm) / (2.0 * dy);

            double v_im = 0.5 * (v_dev[j * v_stride + (i - 1)] + v_dev[(j + 1) * v_stride + (i - 1)]);
            double v_ip = 0.5 * (v_dev[j * v_stride + (i + 1)] + v_dev[(j + 1) * v_stride + (i + 1)]);
            double dvdx = (v_ip - v_im) / (2.0 * dx);

            double two_SijSij = 2.0 * (dudx * dudx + dvdy * dvdy) + (dudy + dvdx) * (dudy + dvdx);
            dissipation += nu * two_SijSij * dV;
        }
    } else {
        const int n_cells = Nx * Ny * Nz;
        int device = omp_get_default_device();
        const double* u_dev = static_cast<const double*>(omp_get_mapped_ptr(velocity_u_ptr_, device));
        const double* v_dev = static_cast<const double*>(omp_get_mapped_ptr(velocity_v_ptr_, device));
        const double* w_dev = static_cast<const double*>(omp_get_mapped_ptr(velocity_w_ptr_, device));

        #pragma omp target teams distribute parallel for reduction(+:dissipation) is_device_ptr(u_dev, v_dev, w_dev) \
            firstprivate(Nx, Ny, Nz, Ng, u_stride, v_stride, w_stride, u_plane, v_plane, w_plane, dx, dy, dz, dV, nu)
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % Ny + Ng;
            int k = idx / (Nx * Ny) + Ng;

            // Diagonal terms
            double dudx = (u_dev[k * u_plane + j * u_stride + (i + 1)] - u_dev[k * u_plane + j * u_stride + i]) / dx;
            double dvdy = (v_dev[k * v_plane + (j + 1) * v_stride + i] - v_dev[k * v_plane + j * v_stride + i]) / dy;
            double dwdz = (w_dev[(k + 1) * w_plane + j * w_stride + i] - w_dev[k * w_plane + j * w_stride + i]) / dz;

            // du/dy, dv/dx
            double u_jm = 0.5 * (u_dev[k * u_plane + (j - 1) * u_stride + i] + u_dev[k * u_plane + (j - 1) * u_stride + (i + 1)]);
            double u_jp = 0.5 * (u_dev[k * u_plane + (j + 1) * u_stride + i] + u_dev[k * u_plane + (j + 1) * u_stride + (i + 1)]);
            double dudy = (u_jp - u_jm) / (2.0 * dy);

            double v_im = 0.5 * (v_dev[k * v_plane + j * v_stride + (i - 1)] + v_dev[k * v_plane + (j + 1) * v_stride + (i - 1)]);
            double v_ip = 0.5 * (v_dev[k * v_plane + j * v_stride + (i + 1)] + v_dev[k * v_plane + (j + 1) * v_stride + (i + 1)]);
            double dvdx = (v_ip - v_im) / (2.0 * dx);

            // du/dz, dw/dx
            double u_km = 0.5 * (u_dev[(k - 1) * u_plane + j * u_stride + i] + u_dev[(k - 1) * u_plane + j * u_stride + (i + 1)]);
            double u_kp = 0.5 * (u_dev[(k + 1) * u_plane + j * u_stride + i] + u_dev[(k + 1) * u_plane + j * u_stride + (i + 1)]);
            double dudz = (u_kp - u_km) / (2.0 * dz);

            double w_im = 0.5 * (w_dev[k * w_plane + j * w_stride + (i - 1)] + w_dev[(k + 1) * w_plane + j * w_stride + (i - 1)]);
            double w_ip = 0.5 * (w_dev[k * w_plane + j * w_stride + (i + 1)] + w_dev[(k + 1) * w_plane + j * w_stride + (i + 1)]);
            double dwdx = (w_ip - w_im) / (2.0 * dx);

            // dv/dz, dw/dy
            double v_km = 0.5 * (v_dev[(k - 1) * v_plane + j * v_stride + i] + v_dev[(k - 1) * v_plane + (j + 1) * v_stride + i]);
            double v_kp = 0.5 * (v_dev[(k + 1) * v_plane + j * v_stride + i] + v_dev[(k + 1) * v_plane + (j + 1) * v_stride + i]);
            double dvdz = (v_kp - v_km) / (2.0 * dz);

            double w_jm = 0.5 * (w_dev[k * w_plane + (j - 1) * w_stride + i] + w_dev[(k + 1) * w_plane + (j - 1) * w_stride + i]);
            double w_jp = 0.5 * (w_dev[k * w_plane + (j + 1) * w_stride + i] + w_dev[(k + 1) * w_plane + (j + 1) * w_stride + i]);
            double dwdy = (w_jp - w_jm) / (2.0 * dy);

            double two_SijSij = 2.0 * (dudx * dudx + dvdy * dvdy + dwdz * dwdz)
                              + (dudy + dvdx) * (dudy + dvdx)
                              + (dudz + dwdx) * (dudz + dwdx)
                              + (dvdz + dwdy) * (dvdz + dwdy);
            dissipation += nu * two_SijSij * dV;
        }
    }
#else
    double* u_ptr = velocity_u_ptr_;
    double* v_ptr = velocity_v_ptr_;
    double* w_ptr = velocity_w_ptr_;

    if (mesh_->is2D()) {
        const int n_cells = Nx * Ny;
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + Ng;
            int j = idx / Nx + Ng;

            double dudx = (u_ptr[j * u_stride + (i + 1)] - u_ptr[j * u_stride + i]) / dx;
            double dvdy = (v_ptr[(j + 1) * v_stride + i] - v_ptr[j * v_stride + i]) / dy;

            double u_jm = 0.5 * (u_ptr[(j - 1) * u_stride + i] + u_ptr[(j - 1) * u_stride + (i + 1)]);
            double u_jp = 0.5 * (u_ptr[(j + 1) * u_stride + i] + u_ptr[(j + 1) * u_stride + (i + 1)]);
            double dudy = (u_jp - u_jm) / (2.0 * dy);

            double v_im = 0.5 * (v_ptr[j * v_stride + (i - 1)] + v_ptr[(j + 1) * v_stride + (i - 1)]);
            double v_ip = 0.5 * (v_ptr[j * v_stride + (i + 1)] + v_ptr[(j + 1) * v_stride + (i + 1)]);
            double dvdx = (v_ip - v_im) / (2.0 * dx);

            double two_SijSij = 2.0 * (dudx * dudx + dvdy * dvdy) + (dudy + dvdx) * (dudy + dvdx);
            dissipation += nu * two_SijSij * dV;
        }
    } else {
        const int n_cells = Nx * Ny * Nz;
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % Ny + Ng;
            int k = idx / (Nx * Ny) + Ng;

            double dudx = (u_ptr[k * u_plane + j * u_stride + (i + 1)] - u_ptr[k * u_plane + j * u_stride + i]) / dx;
            double dvdy = (v_ptr[k * v_plane + (j + 1) * v_stride + i] - v_ptr[k * v_plane + j * v_stride + i]) / dy;
            double dwdz = (w_ptr[(k + 1) * w_plane + j * w_stride + i] - w_ptr[k * w_plane + j * w_stride + i]) / dz;

            double u_jm = 0.5 * (u_ptr[k * u_plane + (j - 1) * u_stride + i] + u_ptr[k * u_plane + (j - 1) * u_stride + (i + 1)]);
            double u_jp = 0.5 * (u_ptr[k * u_plane + (j + 1) * u_stride + i] + u_ptr[k * u_plane + (j + 1) * u_stride + (i + 1)]);
            double dudy = (u_jp - u_jm) / (2.0 * dy);

            double v_im = 0.5 * (v_ptr[k * v_plane + j * v_stride + (i - 1)] + v_ptr[k * v_plane + (j + 1) * v_stride + (i - 1)]);
            double v_ip = 0.5 * (v_ptr[k * v_plane + j * v_stride + (i + 1)] + v_ptr[k * v_plane + (j + 1) * v_stride + (i + 1)]);
            double dvdx = (v_ip - v_im) / (2.0 * dx);

            double u_km = 0.5 * (u_ptr[(k - 1) * u_plane + j * u_stride + i] + u_ptr[(k - 1) * u_plane + j * u_stride + (i + 1)]);
            double u_kp = 0.5 * (u_ptr[(k + 1) * u_plane + j * u_stride + i] + u_ptr[(k + 1) * u_plane + j * u_stride + (i + 1)]);
            double dudz = (u_kp - u_km) / (2.0 * dz);

            double w_im = 0.5 * (w_ptr[k * w_plane + j * w_stride + (i - 1)] + w_ptr[(k + 1) * w_plane + j * w_stride + (i - 1)]);
            double w_ip = 0.5 * (w_ptr[k * w_plane + j * w_stride + (i + 1)] + w_ptr[(k + 1) * w_plane + j * w_stride + (i + 1)]);
            double dwdx = (w_ip - w_im) / (2.0 * dx);

            double v_km = 0.5 * (v_ptr[(k - 1) * v_plane + j * v_stride + i] + v_ptr[(k - 1) * v_plane + (j + 1) * v_stride + i]);
            double v_kp = 0.5 * (v_ptr[(k + 1) * v_plane + j * v_stride + i] + v_ptr[(k + 1) * v_plane + (j + 1) * v_stride + i]);
            double dvdz = (v_kp - v_km) / (2.0 * dz);

            double w_jm = 0.5 * (w_ptr[k * w_plane + (j - 1) * w_stride + i] + w_ptr[(k + 1) * w_plane + (j - 1) * w_stride + i]);
            double w_jp = 0.5 * (w_ptr[k * w_plane + (j + 1) * w_stride + i] + w_ptr[(k + 1) * w_plane + (j + 1) * w_stride + i]);
            double dwdy = (w_jp - w_jm) / (2.0 * dy);

            double two_SijSij = 2.0 * (dudx * dudx + dvdy * dvdy + dwdz * dwdz)
                              + (dudy + dvdx) * (dudy + dvdx)
                              + (dudz + dwdx) * (dudz + dwdx)
                              + (dvdz + dwdy) * (dvdz + dwdy);
            dissipation += nu * two_SijSij * dV;
        }
    }
#endif

    return dissipation;
}

//==============================================================================
// Helper functions for compute_plane_stats GPU kernels
// NOTE: nvc++ 25.5 crashes when GPU pragmas are in functions returning structs.
//       These helper functions work around that compiler bug by taking outputs
//       by reference instead of returning structs.
//==============================================================================

#ifdef USE_GPU_OFFLOAD
namespace {

// NOTE: All function parameters are copied to local variables before use in firstprivate
// to work around nvc++ 25.5 compiler bug (internal compiler error with firstprivate(param))

void plane_stats_mean_2d_gpu(const double* u_dev, const double* v_dev,
                              int Ny_param, int Ng_param, int ig_param,
                              int u_stride_param, int v_stride_param,
                              double& sum_u, double& sum_v) {
    // Copy params to locals for firstprivate (nvc++ 25.5 workaround)
    const int Ny = Ny_param;
    const int Ng = Ng_param;
    const int ig = ig_param;
    const int u_stride = u_stride_param;
    const int v_stride = v_stride_param;

    double local_sum_u = 0.0, local_sum_v = 0.0;
    const int n_iter = Ny;
    #pragma omp target teams distribute parallel for reduction(+:local_sum_u, local_sum_v) \
        is_device_ptr(u_dev, v_dev) firstprivate(n_iter, Ng, ig, u_stride, v_stride)
    for (int j = 0; j < n_iter; ++j) {
        int jg = j + Ng;
        double u = u_dev[jg * u_stride + ig];
        double v = 0.5 * (v_dev[jg * v_stride + ig] + v_dev[(jg + 1) * v_stride + ig]);
        local_sum_u += u;
        local_sum_v += v;
    }
    sum_u = local_sum_u;
    sum_v = local_sum_v;
}

void plane_stats_mean_3d_gpu(const double* u_dev, const double* v_dev, const double* w_dev,
                              int n_points_param, int Ny_param, int Ng_param, int ig_param,
                              int u_stride_param, int v_stride_param, int w_stride_param,
                              int u_plane_param, int v_plane_param, int w_plane_param,
                              double& sum_u, double& sum_v, double& sum_w) {
    // Copy params to locals for firstprivate (nvc++ 25.5 workaround)
    const int n_points = n_points_param;
    const int Ny = Ny_param;
    const int Ng = Ng_param;
    const int ig = ig_param;
    const int u_stride = u_stride_param;
    const int v_stride = v_stride_param;
    const int w_stride = w_stride_param;
    const int u_plane = u_plane_param;
    const int v_plane = v_plane_param;
    const int w_plane = w_plane_param;

    double local_sum_u = 0.0, local_sum_v = 0.0, local_sum_w = 0.0;
    #pragma omp target teams distribute parallel for reduction(+:local_sum_u, local_sum_v, local_sum_w) \
        is_device_ptr(u_dev, v_dev, w_dev) \
        firstprivate(n_points, Ny, Ng, ig, u_stride, v_stride, w_stride, u_plane, v_plane, w_plane)
    for (int idx = 0; idx < n_points; ++idx) {
        int j = idx % Ny + Ng;
        int k = idx / Ny + Ng;
        double u = u_dev[k * u_plane + j * u_stride + ig];
        double v = 0.5 * (v_dev[k * v_plane + j * v_stride + ig] + v_dev[k * v_plane + (j + 1) * v_stride + ig]);
        double w = 0.5 * (w_dev[k * w_plane + j * w_stride + ig] + w_dev[(k + 1) * w_plane + j * w_stride + ig]);
        local_sum_u += u;
        local_sum_v += v;
        local_sum_w += w;
    }
    sum_u = local_sum_u;
    sum_v = local_sum_v;
    sum_w = local_sum_w;
}

void plane_stats_fluct_2d_gpu(const double* u_dev, const double* v_dev,
                               int Ny_param, int Ng_param, int ig_param,
                               int u_stride_param, int v_stride_param,
                               double u_mean_param, double v_mean_param,
                               double& sum_uu, double& sum_vv, double& sum_uv) {
    // Copy params to locals for firstprivate (nvc++ 25.5 workaround)
    const int Ny = Ny_param;
    const int Ng = Ng_param;
    const int ig = ig_param;
    const int u_stride = u_stride_param;
    const int v_stride = v_stride_param;
    const double u_mean = u_mean_param;
    const double v_mean = v_mean_param;

    double local_sum_uu = 0.0, local_sum_vv = 0.0, local_sum_uv = 0.0;
    const int n_iter = Ny;
    #pragma omp target teams distribute parallel for reduction(+:local_sum_uu, local_sum_vv, local_sum_uv) \
        is_device_ptr(u_dev, v_dev) firstprivate(n_iter, Ng, ig, u_stride, v_stride, u_mean, v_mean)
    for (int j = 0; j < n_iter; ++j) {
        int jg = j + Ng;
        double u = u_dev[jg * u_stride + ig];
        double v = 0.5 * (v_dev[jg * v_stride + ig] + v_dev[(jg + 1) * v_stride + ig]);
        double u_prime = u - u_mean;
        double v_prime = v - v_mean;
        local_sum_uu += u_prime * u_prime;
        local_sum_vv += v_prime * v_prime;
        local_sum_uv += u_prime * v_prime;
    }
    sum_uu = local_sum_uu;
    sum_vv = local_sum_vv;
    sum_uv = local_sum_uv;
}

void plane_stats_fluct_3d_gpu(const double* u_dev, const double* v_dev, const double* w_dev,
                               int n_points_param, int Ny_param, int Ng_param, int ig_param,
                               int u_stride_param, int v_stride_param, int w_stride_param,
                               int u_plane_param, int v_plane_param, int w_plane_param,
                               double u_mean_param, double v_mean_param, double w_mean_param,
                               double& sum_uu, double& sum_vv, double& sum_ww, double& sum_uv) {
    // Copy params to locals for firstprivate (nvc++ 25.5 workaround)
    const int n_points = n_points_param;
    const int Ny = Ny_param;
    const int Ng = Ng_param;
    const int ig = ig_param;
    const int u_stride = u_stride_param;
    const int v_stride = v_stride_param;
    const int w_stride = w_stride_param;
    const int u_plane = u_plane_param;
    const int v_plane = v_plane_param;
    const int w_plane = w_plane_param;
    const double u_mean = u_mean_param;
    const double v_mean = v_mean_param;
    const double w_mean = w_mean_param;

    double local_sum_uu = 0.0, local_sum_vv = 0.0, local_sum_ww = 0.0, local_sum_uv = 0.0;
    #pragma omp target teams distribute parallel for \
        reduction(+:local_sum_uu, local_sum_vv, local_sum_ww, local_sum_uv) \
        is_device_ptr(u_dev, v_dev, w_dev) \
        firstprivate(n_points, Ny, Ng, ig, u_stride, v_stride, w_stride, u_plane, v_plane, w_plane, u_mean, v_mean, w_mean)
    for (int idx = 0; idx < n_points; ++idx) {
        int j = idx % Ny + Ng;
        int k = idx / Ny + Ng;
        double u = u_dev[k * u_plane + j * u_stride + ig];
        double v = 0.5 * (v_dev[k * v_plane + j * v_stride + ig] + v_dev[k * v_plane + (j + 1) * v_stride + ig]);
        double w = 0.5 * (w_dev[k * w_plane + j * w_stride + ig] + w_dev[(k + 1) * w_plane + j * w_stride + ig]);
        double u_prime = u - u_mean;
        double v_prime = v - v_mean;
        double w_prime = w - w_mean;
        local_sum_uu += u_prime * u_prime;
        local_sum_vv += v_prime * v_prime;
        local_sum_ww += w_prime * w_prime;
        local_sum_uv += u_prime * v_prime;
    }
    sum_uu = local_sum_uu;
    sum_vv = local_sum_vv;
    sum_ww = local_sum_ww;
    sum_uv = local_sum_uv;
}

} // anonymous namespace
#endif

RANSSolver::PlaneStats RANSSolver::compute_plane_stats(int i_global) const {
    PlaneStats stats = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Ng = mesh_->Nghost;
    const int ig = i_global + Ng;

    const int Nz_loop = mesh_->is2D() ? 1 : Nz;
    const int n_points = Ny * Nz_loop;
    if (n_points == 0) return stats;

    const int u_stride = velocity_.u_stride();
    const int v_stride = velocity_.v_stride();
    const int u_plane = velocity_.u_plane_stride();
    const int v_plane = velocity_.v_plane_stride();
    const int w_stride = velocity_.w_stride();
    const int w_plane = velocity_.w_plane_stride();

    double sum_u = 0.0, sum_v = 0.0, sum_w = 0.0;

#ifdef USE_GPU_OFFLOAD
    int device = omp_get_default_device();
    const double* u_dev = static_cast<const double*>(omp_get_mapped_ptr(velocity_u_ptr_, device));
    const double* v_dev = static_cast<const double*>(omp_get_mapped_ptr(velocity_v_ptr_, device));
    const double* w_dev = static_cast<const double*>(omp_get_mapped_ptr(velocity_w_ptr_, device));

    // First pass: compute means using helper functions (avoid struct-return + GPU pragma crash)
    if (mesh_->is2D()) {
        plane_stats_mean_2d_gpu(u_dev, v_dev, Ny, Ng, ig, u_stride, v_stride, sum_u, sum_v);
    } else {
        plane_stats_mean_3d_gpu(u_dev, v_dev, w_dev, n_points, Ny, Ng, ig,
                                 u_stride, v_stride, w_stride, u_plane, v_plane, w_plane,
                                 sum_u, sum_v, sum_w);
    }
#else
    double* u_ptr = velocity_u_ptr_;
    double* v_ptr = velocity_v_ptr_;
    double* w_ptr = velocity_w_ptr_;

    if (mesh_->is2D()) {
        for (int j = 0; j < Ny; ++j) {
            int jg = j + Ng;
            double u = u_ptr[jg * u_stride + ig];
            double v = 0.5 * (v_ptr[jg * v_stride + ig] + v_ptr[(jg + 1) * v_stride + ig]);
            sum_u += u;
            sum_v += v;
        }
    } else {
        for (int idx = 0; idx < n_points; ++idx) {
            int j = idx % Ny + Ng;
            int k = idx / Ny + Ng;
            double u = u_ptr[k * u_plane + j * u_stride + ig];
            double v = 0.5 * (v_ptr[k * v_plane + j * v_stride + ig] + v_ptr[k * v_plane + (j + 1) * v_stride + ig]);
            double w = 0.5 * (w_ptr[k * w_plane + j * w_stride + ig] + w_ptr[(k + 1) * w_plane + j * w_stride + ig]);
            sum_u += u;
            sum_v += v;
            sum_w += w;
        }
    }
#endif

    stats.u_mean = sum_u / n_points;
    stats.v_mean = sum_v / n_points;
    stats.w_mean = sum_w / n_points;

    // Second pass: fluctuations and Reynolds stress
    double sum_uu = 0.0, sum_vv = 0.0, sum_ww = 0.0, sum_uv = 0.0;
    const double u_mean = stats.u_mean;
    const double v_mean = stats.v_mean;
    const double w_mean = stats.w_mean;

#ifdef USE_GPU_OFFLOAD
    if (mesh_->is2D()) {
        plane_stats_fluct_2d_gpu(u_dev, v_dev, Ny, Ng, ig, u_stride, v_stride,
                                  u_mean, v_mean, sum_uu, sum_vv, sum_uv);
    } else {
        plane_stats_fluct_3d_gpu(u_dev, v_dev, w_dev, n_points, Ny, Ng, ig,
                                  u_stride, v_stride, w_stride, u_plane, v_plane, w_plane,
                                  u_mean, v_mean, w_mean, sum_uu, sum_vv, sum_ww, sum_uv);
    }
#else
    if (mesh_->is2D()) {
        for (int j = 0; j < Ny; ++j) {
            int jg = j + Ng;
            double u = u_ptr[jg * u_stride + ig];
            double v = 0.5 * (v_ptr[jg * v_stride + ig] + v_ptr[(jg + 1) * v_stride + ig]);
            double u_prime = u - u_mean;
            double v_prime = v - v_mean;
            sum_uu += u_prime * u_prime;
            sum_vv += v_prime * v_prime;
            sum_uv += u_prime * v_prime;
        }
    } else {
        for (int idx = 0; idx < n_points; ++idx) {
            int j = idx % Ny + Ng;
            int k = idx / Ny + Ng;
            double u = u_ptr[k * u_plane + j * u_stride + ig];
            double v = 0.5 * (v_ptr[k * v_plane + j * v_stride + ig] + v_ptr[k * v_plane + (j + 1) * v_stride + ig]);
            double w = 0.5 * (w_ptr[k * w_plane + j * w_stride + ig] + w_ptr[(k + 1) * w_plane + j * w_stride + ig]);
            double u_prime = u - u_mean;
            double v_prime = v - v_mean;
            double w_prime = w - w_mean;
            sum_uu += u_prime * u_prime;
            sum_vv += v_prime * v_prime;
            sum_ww += w_prime * w_prime;
            sum_uv += u_prime * v_prime;
        }
    }
#endif

    stats.u_rms = std::sqrt(sum_uu / n_points);
    stats.v_rms = std::sqrt(sum_vv / n_points);
    stats.w_rms = std::sqrt(sum_ww / n_points);
    stats.uv_reynolds = -sum_uv / n_points;  // -<u'v'>

    return stats;
}

} // namespace nncfd
