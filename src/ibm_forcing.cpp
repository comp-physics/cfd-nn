/// @file ibm_forcing.cpp
/// @brief Direct-forcing immersed boundary method implementation

#include "ibm_forcing.hpp"
#include <cmath>
#include <algorithm>
#include <cassert>
#include <iostream>

namespace nncfd {

IBMForcing::IBMForcing(const Mesh& mesh, std::shared_ptr<IBMBody> body)
    : mesh_(&mesh), body_(std::move(body))
{
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = std::max(mesh_->Nz, 1);
    const int Ng = mesh_->Nghost;

    // Band width: ~1.5 grid cells for the forcing region
    double h = std::min(mesh_->dx, mesh_->dy);
    if (!mesh_->is2D()) h = std::min(h, mesh_->dz);
    band_width_ = 1.5 * h;

    // Compute strides for velocity face indexing
    // u: (Nx+1+2Ng) x (Ny+2Ng) x (Nz+2Ng)
    u_stride_ = Nx + 1 + 2 * Ng;
    u_plane_stride_ = u_stride_ * (Ny + 2 * Ng);

    // v: (Nx+2Ng) x (Ny+1+2Ng) x (Nz+2Ng)
    v_stride_ = Nx + 2 * Ng;
    v_plane_stride_ = v_stride_ * (Ny + 1 + 2 * Ng);

    // w: (Nx+2Ng) x (Ny+2Ng) x (Nz+1+2Ng)
    w_stride_ = Nx + 2 * Ng;
    w_plane_stride_ = w_stride_ * (Ny + 2 * Ng);

    // Allocate cell type arrays
    int Nz_eff = mesh_->is2D() ? 1 : Nz;
    int u_total = u_stride_ * (Ny + 2 * Ng) * (Nz_eff + 2 * Ng);
    int v_total = v_stride_ * (Ny + 1 + 2 * Ng) * (Nz_eff + 2 * Ng);
    int w_total = w_stride_ * (Ny + 2 * Ng) * (Nz_eff + 1 + 2 * Ng);

    cell_type_u_.resize(u_total, IBMCellType::Fluid);
    cell_type_v_.resize(v_total, IBMCellType::Fluid);
    if (!mesh_->is2D()) {
        cell_type_w_.resize(w_total, IBMCellType::Fluid);
    }

    classify_cells();
}

IBMCellType IBMForcing::classify_point(double phi) const {
    if (phi > 0.0) return IBMCellType::Fluid;
    if (phi < -band_width_) return IBMCellType::Solid;
    return IBMCellType::Forcing;
}

void IBMForcing::classify_cells() {
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = std::max(mesh_->Nz, 1);
    const int Ng = mesh_->Nghost;
    const bool is2D = mesh_->is2D();

    n_forcing_ = 0;
    n_solid_ = 0;

    // Classify u-faces (located at x_{i+1/2}, y_j, z_k)
    int Nz_eff = is2D ? 1 : Nz;
    for (int k = Ng; k < Nz_eff + Ng; ++k) {
        for (int j = Ng; j < Ny + Ng; ++j) {
            for (int i = Ng; i <= Nx + Ng; ++i) {
                double x = mesh_->xf[i];
                double y = mesh_->y(j);
                double z = is2D ? 0.0 : mesh_->z(k);
                double phi = body_->phi(x, y, z);
                int idx = is2D ? (j * u_stride_ + i) : (k * u_plane_stride_ + j * u_stride_ + i);
                auto ct = classify_point(phi);
                cell_type_u_[idx] = ct;
                if (ct == IBMCellType::Forcing) ++n_forcing_;
                if (ct == IBMCellType::Solid) ++n_solid_;
            }
        }
    }

    // Classify v-faces (located at x_i, y_{j+1/2}, z_k)
    for (int k = Ng; k < Nz_eff + Ng; ++k) {
        for (int j = Ng; j <= Ny + Ng; ++j) {
            for (int i = Ng; i < Nx + Ng; ++i) {
                double x = mesh_->x(i);
                double y = mesh_->yf[j];
                double z = is2D ? 0.0 : mesh_->z(k);
                double phi = body_->phi(x, y, z);
                int idx = is2D ? (j * v_stride_ + i) : (k * v_plane_stride_ + j * v_stride_ + i);
                cell_type_v_[idx] = classify_point(phi);
            }
        }
    }

    // Classify w-faces (3D only, located at x_i, y_j, z_{k+1/2})
    if (!is2D) {
        for (int k = Ng; k <= Nz + Ng; ++k) {
            for (int j = Ng; j < Ny + Ng; ++j) {
                for (int i = Ng; i < Nx + Ng; ++i) {
                    double x = mesh_->x(i);
                    double y = mesh_->y(j);
                    double z = mesh_->zf[k];
                    double phi = body_->phi(x, y, z);
                    int idx = k * w_plane_stride_ + j * w_stride_ + i;
                    cell_type_w_[idx] = classify_point(phi);
                }
            }
        }
    }

    std::cout << "[IBM] Classified cells: " << n_forcing_ << " forcing, "
              << n_solid_ << " solid (band=" << band_width_ << ")\n";
}

void IBMForcing::apply_forcing(VectorField& vel, double dt) {
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = std::max(mesh_->Nz, 1);
    const int Ng = mesh_->Nghost;
    const bool is2D = mesh_->is2D();
    int Nz_eff = is2D ? 1 : Nz;

    (void)dt;  // Direct forcing: set velocity to target (0 for no-slip)

    // Apply forcing to u-faces
    for (int k = Ng; k < Nz_eff + Ng; ++k) {
        for (int j = Ng; j < Ny + Ng; ++j) {
            for (int i = Ng; i <= Nx + Ng; ++i) {
                int idx = is2D ? (j * u_stride_ + i) : (k * u_plane_stride_ + j * u_stride_ + i);
                IBMCellType ct = cell_type_u_[idx];
                if (ct == IBMCellType::Solid) {
                    if (is2D) vel.u(i, j) = 0.0;
                    else vel.u(i, j, k) = 0.0;
                } else if (ct == IBMCellType::Forcing) {
                    double x = mesh_->xf[i];
                    double y = mesh_->y(j);
                    double z = is2D ? 0.0 : mesh_->z(k);
                    double phi = body_->phi(x, y, z);
                    double weight = std::abs(phi) / band_width_;
                    weight = std::max(0.0, std::min(1.0, weight));
                    if (is2D) vel.u(i, j) *= weight;
                    else vel.u(i, j, k) *= weight;
                }
            }
        }
    }

    // Apply forcing to v-faces
    for (int k = Ng; k < Nz_eff + Ng; ++k) {
        for (int j = Ng; j <= Ny + Ng; ++j) {
            for (int i = Ng; i < Nx + Ng; ++i) {
                int idx = is2D ? (j * v_stride_ + i) : (k * v_plane_stride_ + j * v_stride_ + i);
                IBMCellType ct = cell_type_v_[idx];
                if (ct == IBMCellType::Solid) {
                    if (is2D) vel.v(i, j) = 0.0;
                    else vel.v(i, j, k) = 0.0;
                } else if (ct == IBMCellType::Forcing) {
                    double x = mesh_->x(i);
                    double y = mesh_->yf[j];
                    double z = is2D ? 0.0 : mesh_->z(k);
                    double phi = body_->phi(x, y, z);
                    double weight = std::abs(phi) / band_width_;
                    weight = std::max(0.0, std::min(1.0, weight));
                    if (is2D) vel.v(i, j) *= weight;
                    else vel.v(i, j, k) *= weight;
                }
            }
        }
    }

    // Apply forcing to w-faces (3D only)
    if (!is2D) {
        for (int k = Ng; k <= Nz + Ng; ++k) {
            for (int j = Ng; j < Ny + Ng; ++j) {
                for (int i = Ng; i < Nx + Ng; ++i) {
                    int idx = k * w_plane_stride_ + j * w_stride_ + i;
                    IBMCellType ct = cell_type_w_[idx];
                    if (ct == IBMCellType::Solid) {
                        vel.w(i, j, k) = 0.0;
                    } else if (ct == IBMCellType::Forcing) {
                        double x = mesh_->x(i);
                        double y = mesh_->y(j);
                        double z = mesh_->zf[k];
                        double phi = body_->phi(x, y, z);
                        double weight = std::abs(phi) / band_width_;
                        weight = std::max(0.0, std::min(1.0, weight));
                        vel.w(i, j, k) *= weight;
                    }
                }
            }
        }
    }
}

std::tuple<double, double, double> IBMForcing::compute_forces(
    const VectorField& vel, double dt) const
{
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = std::max(mesh_->Nz, 1);
    const int Ng = mesh_->Nghost;
    const bool is2D = mesh_->is2D();
    int Nz_eff = is2D ? 1 : Nz;

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    const double dx = mesh_->dx;
    const double dz = is2D ? 1.0 : mesh_->dz;

    // Sum forcing contributions from u-faces
    for (int k = Ng; k < Nz_eff + Ng; ++k) {
        for (int j = Ng; j < Ny + Ng; ++j) {
            double dy_local = mesh_->yf[j + 1] - mesh_->yf[j];
            double dV = dx * dy_local * dz;
            for (int i = Ng; i <= Nx + Ng; ++i) {
                int idx = is2D ? (j * u_stride_ + i) : (k * u_plane_stride_ + j * u_stride_ + i);
                if (cell_type_u_[idx] == IBMCellType::Forcing ||
                    cell_type_u_[idx] == IBMCellType::Solid) {
                    double u_val = is2D ? vel.u(i, j) : vel.u(i, j, k);
                    Fx -= u_val / dt * dV;
                }
            }
        }
    }

    // Sum forcing contributions from v-faces
    for (int k = Ng; k < Nz_eff + Ng; ++k) {
        for (int j = Ng; j <= Ny + Ng; ++j) {
            double dy_local = mesh_->yc[j] - mesh_->yc[j - 1];
            double dV = dx * dy_local * dz;
            for (int i = Ng; i < Nx + Ng; ++i) {
                int idx = is2D ? (j * v_stride_ + i) : (k * v_plane_stride_ + j * v_stride_ + i);
                if (cell_type_v_[idx] == IBMCellType::Forcing ||
                    cell_type_v_[idx] == IBMCellType::Solid) {
                    double v_val = is2D ? vel.v(i, j) : vel.v(i, j, k);
                    Fy -= v_val / dt * dV;
                }
            }
        }
    }

    // w-faces (3D only)
    if (!is2D) {
        for (int k = Ng; k <= Nz + Ng; ++k) {
            for (int j = Ng; j < Ny + Ng; ++j) {
                double dy_local = mesh_->yf[j + 1] - mesh_->yf[j];
                double dV = dx * dy_local * dz;
                for (int i = Ng; i < Nx + Ng; ++i) {
                    int idx = k * w_plane_stride_ + j * w_stride_ + i;
                    if (cell_type_w_[idx] == IBMCellType::Forcing ||
                        cell_type_w_[idx] == IBMCellType::Solid) {
                        Fz -= vel.w(i, j, k) / dt * dV;
                    }
                }
            }
        }
    }

    return {Fx, Fy, Fz};
}

IBMCellType IBMForcing::cell_type_u(int i, int j, int k) const {
    int idx = mesh_->is2D() ? (j * u_stride_ + i) : (k * u_plane_stride_ + j * u_stride_ + i);
    assert(idx >= 0 && idx < static_cast<int>(cell_type_u_.size()));
    return cell_type_u_[idx];
}

IBMCellType IBMForcing::cell_type_v(int i, int j, int k) const {
    int idx = mesh_->is2D() ? (j * v_stride_ + i) : (k * v_plane_stride_ + j * v_stride_ + i);
    assert(idx >= 0 && idx < static_cast<int>(cell_type_v_.size()));
    return cell_type_v_[idx];
}

IBMCellType IBMForcing::cell_type_w(int i, int j, int k) const {
    int idx = k * w_plane_stride_ + j * w_stride_ + i;
    assert(idx >= 0 && idx < static_cast<int>(cell_type_w_.size()));
    return cell_type_w_[idx];
}

} // namespace nncfd
