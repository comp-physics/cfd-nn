/// @file ibm_forcing.cpp
/// @brief Direct-forcing immersed boundary method implementation (GPU-accelerated)
///
/// GPU strategy: pre-compute weight arrays (0=solid, 0<w<1=forcing, 1=fluid) at
/// classification time, map to GPU, then apply_forcing is a simple element-wise
/// multiply — no virtual function calls or branching on GPU.

#include "ibm_forcing.hpp"
#include <cmath>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <stdexcept>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

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
    u_stride_ = Nx + 1 + 2 * Ng;
    u_plane_stride_ = u_stride_ * (Ny + 2 * Ng);

    v_stride_ = Nx + 2 * Ng;
    v_plane_stride_ = v_stride_ * (Ny + 1 + 2 * Ng);

    w_stride_ = Nx + 2 * Ng;
    w_plane_stride_ = w_stride_ * (Ny + 2 * Ng);

    // Allocate cell type arrays
    int Nz_eff = mesh_->is2D() ? 1 : Nz;
    u_total_ = u_stride_ * (Ny + 2 * Ng) * (Nz_eff + 2 * Ng);
    v_total_ = v_stride_ * (Ny + 1 + 2 * Ng) * (Nz_eff + 2 * Ng);
    w_total_ = w_stride_ * (Ny + 2 * Ng) * (Nz_eff + 1 + 2 * Ng);

    cell_type_u_.resize(u_total_, IBMCellType::Fluid);
    cell_type_v_.resize(v_total_, IBMCellType::Fluid);
    if (!mesh_->is2D()) {
        cell_type_w_.resize(w_total_, IBMCellType::Fluid);
    }

    classify_cells();
    compute_weights();
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

void IBMForcing::compute_weights() {
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = std::max(mesh_->Nz, 1);
    const int Ng = mesh_->Nghost;
    const bool is2D = mesh_->is2D();
    int Nz_eff = is2D ? 1 : Nz;

    // Initialize all weights to 1.0 (fluid = pass-through)
    weight_u_.assign(u_total_, 1.0);
    weight_v_.assign(v_total_, 1.0);
    if (!is2D) weight_w_.assign(w_total_, 1.0);

    // Set weights for u-faces
    for (int k = Ng; k < Nz_eff + Ng; ++k) {
        for (int j = Ng; j < Ny + Ng; ++j) {
            for (int i = Ng; i <= Nx + Ng; ++i) {
                int idx = is2D ? (j * u_stride_ + i) : (k * u_plane_stride_ + j * u_stride_ + i);
                IBMCellType ct = cell_type_u_[idx];
                if (ct == IBMCellType::Solid) {
                    weight_u_[idx] = 0.0;
                } else if (ct == IBMCellType::Forcing) {
                    double x = mesh_->xf[i];
                    double y = mesh_->y(j);
                    double z = is2D ? 0.0 : mesh_->z(k);
                    double phi = body_->phi(x, y, z);
                    double w = std::abs(phi) / band_width_;
                    weight_u_[idx] = std::max(0.0, std::min(1.0, w));
                }
            }
        }
    }

    // Set weights for v-faces
    for (int k = Ng; k < Nz_eff + Ng; ++k) {
        for (int j = Ng; j <= Ny + Ng; ++j) {
            for (int i = Ng; i < Nx + Ng; ++i) {
                int idx = is2D ? (j * v_stride_ + i) : (k * v_plane_stride_ + j * v_stride_ + i);
                IBMCellType ct = cell_type_v_[idx];
                if (ct == IBMCellType::Solid) {
                    weight_v_[idx] = 0.0;
                } else if (ct == IBMCellType::Forcing) {
                    double x = mesh_->x(i);
                    double y = mesh_->yf[j];
                    double z = is2D ? 0.0 : mesh_->z(k);
                    double phi = body_->phi(x, y, z);
                    double w = std::abs(phi) / band_width_;
                    weight_v_[idx] = std::max(0.0, std::min(1.0, w));
                }
            }
        }
    }

    // Set weights for w-faces (3D only)
    if (!is2D) {
        for (int k = Ng; k <= Nz + Ng; ++k) {
            for (int j = Ng; j < Ny + Ng; ++j) {
                for (int i = Ng; i < Nx + Ng; ++i) {
                    int idx = k * w_plane_stride_ + j * w_stride_ + i;
                    IBMCellType ct = cell_type_w_[idx];
                    if (ct == IBMCellType::Solid) {
                        weight_w_[idx] = 0.0;
                    } else if (ct == IBMCellType::Forcing) {
                        double x = mesh_->x(i);
                        double y = mesh_->y(j);
                        double z = mesh_->zf[k];
                        double phi = body_->phi(x, y, z);
                        double w = std::abs(phi) / band_width_;
                        weight_w_[idx] = std::max(0.0, std::min(1.0, w));
                    }
                }
            }
        }
    }

    weight_u_ptr_ = weight_u_.data();
    weight_v_ptr_ = weight_v_.data();
    weight_w_ptr_ = is2D ? nullptr : weight_w_.data();

    // Compute cell-centered solid mask for Poisson RHS masking
    const int cell_stride = mesh_->total_Nx();
    const int cell_plane = cell_stride * mesh_->total_Ny();
    cell_total_ = is2D ? static_cast<size_t>(cell_stride * mesh_->total_Ny())
                       : static_cast<size_t>(cell_plane * mesh_->total_Nz());
    solid_mask_cell_.assign(cell_total_, 1.0);

    for (int k = Ng; k < Nz_eff + Ng; ++k) {
        for (int j = Ng; j < Ny + Ng; ++j) {
            for (int i = Ng; i < Nx + Ng; ++i) {
                double x = mesh_->x(i);
                double y = mesh_->y(j);
                double z = is2D ? 0.0 : mesh_->z(k);
                if (body_->phi(x, y, z) < 0.0) {
                    int idx = is2D ? (j * cell_stride + i) : (k * cell_plane + j * cell_stride + i);
                    solid_mask_cell_[idx] = 0.0;
                }
            }
        }
    }
    solid_mask_cell_ptr_ = solid_mask_cell_.data();
}

void IBMForcing::map_to_gpu() {
    if (gpu_mapped_) return;

    #pragma omp target enter data map(to: weight_u_ptr_[0:u_total_])
    #pragma omp target enter data map(to: weight_v_ptr_[0:v_total_])
    if (weight_w_ptr_ && w_total_ > 0) {
        #pragma omp target enter data map(to: weight_w_ptr_[0:w_total_])
    }
    if (solid_mask_cell_ptr_ && cell_total_ > 0) {
        #pragma omp target enter data map(to: solid_mask_cell_ptr_[0:cell_total_])
    }

    gpu_mapped_ = true;
}

void IBMForcing::unmap_from_gpu() {
    if (!gpu_mapped_) return;

    #pragma omp target exit data map(delete: weight_u_ptr_[0:u_total_])
    #pragma omp target exit data map(delete: weight_v_ptr_[0:v_total_])
    if (weight_w_ptr_ && w_total_ > 0) {
        #pragma omp target exit data map(delete: weight_w_ptr_[0:w_total_])
    }
    if (solid_mask_cell_ptr_ && cell_total_ > 0) {
        #pragma omp target exit data map(delete: solid_mask_cell_ptr_[0:cell_total_])
    }

    gpu_mapped_ = false;
}

void IBMForcing::reset_force_accumulator() {
    last_Fx_ = 0.0;
    last_Fy_ = 0.0;
    last_Fz_ = 0.0;
}

void IBMForcing::apply_forcing(VectorField& vel, double dt) {
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = std::max(mesh_->Nz, 1);
    const int Ng = mesh_->Nghost;
    const bool is2D = mesh_->is2D();
    int Nz_eff = is2D ? 1 : Nz;

    // Accumulate IBM momentum correction into force accumulator (ADD, don't reset).
    // Call reset_force_accumulator() once per step before the first apply_forcing call.
    // Both calls (predictor + corrected velocity) contribute: predictor captures viscous
    // damping, corrected velocity captures pressure drag (u^{n+1} = u*_IBM - dt*grad(p)).
    if (dt > 0.0) {
        const double dx = mesh_->dx;
        const double dz_val = is2D ? 1.0 : mesh_->dz;

        for (int k = Ng; k < Nz_eff + Ng; ++k) {
            for (int j = Ng; j < Ny + Ng; ++j) {
                double dy_local = mesh_->yf[j + 1] - mesh_->yf[j];
                double dV = dx * dy_local * dz_val;
                for (int i = Ng; i <= Nx + Ng; ++i) {
                    int idx = is2D ? (j * u_stride_ + i) : (k * u_plane_stride_ + j * u_stride_ + i);
                    double w = weight_u_[idx];
                    if (w < 1.0) {
                        double u_val = is2D ? vel.u(i, j) : vel.u(i, j, k);
                        last_Fx_ += (1.0 - w) * u_val / dt * dV;
                    }
                }
            }
        }

        for (int k = Ng; k < Nz_eff + Ng; ++k) {
            for (int j = Ng; j <= Ny + Ng; ++j) {
                double dy_local = mesh_->yc[j] - mesh_->yc[j - 1];
                double dV = dx * dy_local * dz_val;
                for (int i = Ng; i < Nx + Ng; ++i) {
                    int idx = is2D ? (j * v_stride_ + i) : (k * v_plane_stride_ + j * v_stride_ + i);
                    double w = weight_v_[idx];
                    if (w < 1.0) {
                        double v_val = is2D ? vel.v(i, j) : vel.v(i, j, k);
                        last_Fy_ += (1.0 - w) * v_val / dt * dV;
                    }
                }
            }
        }

        if (!is2D) {
            for (int k = Ng; k <= Nz + Ng; ++k) {
                for (int j = Ng; j < Ny + Ng; ++j) {
                    double dy_local = mesh_->yf[j + 1] - mesh_->yf[j];
                    double dV = dx * dy_local * dz_val;
                    for (int i = Ng; i < Nx + Ng; ++i) {
                        int idx = k * w_plane_stride_ + j * w_stride_ + i;
                        double w = weight_w_[idx];
                        if (w < 1.0) {
                            last_Fz_ += (1.0 - w) * vel.w(i, j, k) / dt * dV;
                        }
                    }
                }
            }
        }
    }

    // CPU path: element-wise multiply by weight
    for (int k = Ng; k < Nz_eff + Ng; ++k) {
        for (int j = Ng; j < Ny + Ng; ++j) {
            for (int i = Ng; i <= Nx + Ng; ++i) {
                int idx = is2D ? (j * u_stride_ + i) : (k * u_plane_stride_ + j * u_stride_ + i);
                if (is2D) vel.u(i, j) *= weight_u_[idx];
                else vel.u(i, j, k) *= weight_u_[idx];
            }
        }
    }

    for (int k = Ng; k < Nz_eff + Ng; ++k) {
        for (int j = Ng; j <= Ny + Ng; ++j) {
            for (int i = Ng; i < Nx + Ng; ++i) {
                int idx = is2D ? (j * v_stride_ + i) : (k * v_plane_stride_ + j * v_stride_ + i);
                if (is2D) vel.v(i, j) *= weight_v_[idx];
                else vel.v(i, j, k) *= weight_v_[idx];
            }
        }
    }

    if (!is2D) {
        for (int k = Ng; k <= Nz + Ng; ++k) {
            for (int j = Ng; j < Ny + Ng; ++j) {
                for (int i = Ng; i < Nx + Ng; ++i) {
                    int idx = k * w_plane_stride_ + j * w_stride_ + i;
                    vel.w(i, j, k) *= weight_w_[idx];
                }
            }
        }
    }
}

void IBMForcing::apply_forcing_device(double* u_ptr, double* v_ptr, double* w_ptr, double dt) {
    if (!gpu_mapped_)
        throw std::runtime_error("[IBM] apply_forcing_device called before map_to_gpu()");

    double* wu = weight_u_ptr_;
    double* wv = weight_v_ptr_;
    double* ww = weight_w_ptr_;
    const int u_n = static_cast<int>(u_total_);
    const int v_n = static_cast<int>(v_total_);
    const int w_n = static_cast<int>(w_total_);

    // Accumulate IBM momentum correction (ADD to accumulator; caller resets via
    // reset_force_accumulator() once per step). Both calls (predictor + corrected
    // velocity) contribute so the total captures viscous + pressure drag.
    if (dt > 0.0) {
        [[maybe_unused]] const double dV = mesh_->dx * mesh_->dy * (mesh_->is2D() ? 1.0 : mesh_->dz);
        double Fx_acc = 0.0;
        double Fy_acc = 0.0;
        double Fz_acc = 0.0;

        #pragma omp target teams distribute parallel for reduction(+:Fx_acc) \
            map(present: u_ptr[0:u_n], wu[0:u_n])
        for (int i = 0; i < u_n; ++i) {
            Fx_acc += (1.0 - wu[i]) * u_ptr[i];
        }

        #pragma omp target teams distribute parallel for reduction(+:Fy_acc) \
            map(present: v_ptr[0:v_n], wv[0:v_n])
        for (int i = 0; i < v_n; ++i) {
            Fy_acc += (1.0 - wv[i]) * v_ptr[i];
        }

        if (w_ptr && ww && w_n > 0) {
            #pragma omp target teams distribute parallel for reduction(+:Fz_acc) \
                map(present: w_ptr[0:w_n], ww[0:w_n])
            for (int i = 0; i < w_n; ++i) {
                Fz_acc += (1.0 - ww[i]) * w_ptr[i];
            }
        }

        last_Fx_ += Fx_acc / dt * dV;
        last_Fy_ += Fy_acc / dt * dV;
        last_Fz_ += Fz_acc / dt * dV;
    }

    #pragma omp target teams distribute parallel for \
        map(present: u_ptr[0:u_n], wu[0:u_n])
    for (int i = 0; i < u_n; ++i) {
        u_ptr[i] *= wu[i];
    }

    #pragma omp target teams distribute parallel for \
        map(present: v_ptr[0:v_n], wv[0:v_n])
    for (int i = 0; i < v_n; ++i) {
        v_ptr[i] *= wv[i];
    }

    if (w_ptr && ww && w_n > 0) {
        #pragma omp target teams distribute parallel for \
            map(present: w_ptr[0:w_n], ww[0:w_n])
        for (int i = 0; i < w_n; ++i) {
            w_ptr[i] *= ww[i];
        }
    }
}

void IBMForcing::mask_rhs_device(double* rhs_ptr) {
    if (!gpu_mapped_)
        throw std::runtime_error("[IBM] mask_rhs_device called before map_to_gpu()");
    if (!solid_mask_cell_ptr_)
        throw std::runtime_error("[IBM] mask_rhs_device called but solid_mask_cell_ptr_ is null");

    double* mask = solid_mask_cell_ptr_;
    const int n = static_cast<int>(cell_total_);

    #pragma omp target teams distribute parallel for \
        map(present: rhs_ptr[0:n], mask[0:n])
    for (int i = 0; i < n; ++i) {
        rhs_ptr[i] *= mask[i];
    }
}

std::tuple<double, double, double> IBMForcing::compute_forces(
    const VectorField& vel, double dt) const
{
    (void)vel; (void)dt;
    return {last_Fx_, last_Fy_, last_Fz_};
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
