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

    if (n_solid_ == 0 && n_forcing_ > 0) {
        std::cerr << "\n[IBM] WARNING: No solid cells detected — body is under-resolved!\n"
                  << "  The grid spacing (dx=" << mesh_->dx << ", dy=" << mesh_->dy
                  << ") is larger than the body radius.\n"
                  << "  Increase grid resolution so at least a few cells are fully inside the body.\n\n";
    }
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

    // Ghost-cell IBM: override forcing cell weights to 1.0 and compute interpolation
    if (ghost_cell_ibm_) {
        // Set all Forcing cells to weight=1 (bypass weight multiply for these cells)
        for (size_t i = 0; i < weight_u_.size(); ++i) {
            if (cell_type_u_[i] == IBMCellType::Forcing) weight_u_[i] = 1.0;
        }
        for (size_t i = 0; i < weight_v_.size(); ++i) {
            if (cell_type_v_[i] == IBMCellType::Forcing) weight_v_[i] = 1.0;
        }
        if (!is2D) {
            for (size_t i = 0; i < weight_w_.size(); ++i) {
                if (cell_type_w_[i] == IBMCellType::Forcing) weight_w_[i] = 1.0;
            }
        }
        weight_u_ptr_ = weight_u_.data();
        weight_v_ptr_ = weight_v_.data();
        weight_w_ptr_ = is2D ? nullptr : weight_w_.data();

        compute_ghost_cell_interp();
        if (ghost_cell_ibm_) {
            compute_ghost_cell_interp_2nd();
        }
    }
}

void IBMForcing::compute_ghost_cell_interp() {
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = std::max(mesh_->Nz, 1);
    const int Ng = mesh_->Nghost;
    const bool is2D = mesh_->is2D();
    int Nz_eff = is2D ? 1 : Nz;

    ghost_self_u_.clear(); ghost_nbr_u_.clear(); ghost_alpha_u_.clear();
    ghost_self_v_.clear(); ghost_nbr_v_.clear(); ghost_alpha_v_.clear();
    ghost_self_w_.clear(); ghost_nbr_w_.clear(); ghost_alpha_w_.clear();

    // Helper: find a fluid neighbor in direction (di, dj, dk) within ±2 steps
    // Returns flat index of the fluid neighbor, or -1 if not found
    auto find_fluid_nbr = [&](const std::vector<IBMCellType>& ct, int idx, int stride,
                               int i, int j, int k, int di, int dj, int dk,
                               int i_max, int j_max, int k_max,
                               int plane_stride) -> std::pair<int, double> {
        for (int step = 1; step <= 2; ++step) {
            int ni = i + step * di;
            int nj = j + step * dj;
            int nk = k + step * dk;
            if (ni < Ng || ni > i_max || nj < Ng || nj > j_max) return {-1, 0.0};
            if (!is2D && (nk < Ng || nk > k_max)) return {-1, 0.0};
            int nidx = is2D ? (nj * stride + ni) : (nk * plane_stride + nj * stride + ni);
            if (ct[nidx] == IBMCellType::Fluid) {
                // phi at the forcing cell
                double x_f, y_f, z_f;
                // These are approximate — use cell/face centers
                x_f = mesh_->xf[i]; y_f = mesh_->y(j);
                z_f = is2D ? 0.0 : mesh_->z(k);
                double phi_f = std::abs(body_->phi(x_f, y_f, z_f));

                double x_n, y_n, z_n;
                x_n = mesh_->xf[ni]; y_n = mesh_->y(nj);
                z_n = is2D ? 0.0 : mesh_->z(nk);
                double phi_n = body_->phi(x_n, y_n, z_n);

                if (phi_n <= 0.0) continue; // Not actually fluid
                // Ghost-cell: linearly interpolate to u=0 at surface
                // alpha = d_to_surface / (d_to_surface + d_fluid_to_surface)
                // At surface (phi_f→0): alpha→0, u→0 (no-slip)
                double alpha = phi_f / (phi_f + phi_n);
                alpha = std::max(0.0, std::min(1.0, alpha));
                return {nidx, alpha};
            }
        }
        return {-1, 0.0};
    };

    // Process u-faces
    for (int k = Ng; k < Nz_eff + Ng; ++k) {
        for (int j = Ng; j < Ny + Ng; ++j) {
            for (int i = Ng; i <= Nx + Ng; ++i) {
                int idx = is2D ? (j * u_stride_ + i) : (k * u_plane_stride_ + j * u_stride_ + i);
                if (cell_type_u_[idx] != IBMCellType::Forcing) continue;

                // Try each direction, pick the one with a fluid neighbor
                std::pair<int, double> best = {-1, 0.0};

                // +x direction
                auto r = find_fluid_nbr(cell_type_u_, idx, u_stride_, i, j, k, 1, 0, 0,
                                         Nx + Ng, Ny + Ng - 1, Nz_eff + Ng - 1, u_plane_stride_);
                if (r.first >= 0) best = r;

                // -x direction
                if (best.first < 0) {
                    r = find_fluid_nbr(cell_type_u_, idx, u_stride_, i, j, k, -1, 0, 0,
                                       Nx + Ng, Ny + Ng - 1, Nz_eff + Ng - 1, u_plane_stride_);
                    if (r.first >= 0) best = r;
                }

                // +y direction
                if (best.first < 0) {
                    r = find_fluid_nbr(cell_type_u_, idx, u_stride_, i, j, k, 0, 1, 0,
                                       Nx + Ng, Ny + Ng - 1, Nz_eff + Ng - 1, u_plane_stride_);
                    if (r.first >= 0) best = r;
                }

                // -y direction
                if (best.first < 0) {
                    r = find_fluid_nbr(cell_type_u_, idx, u_stride_, i, j, k, 0, -1, 0,
                                       Nx + Ng, Ny + Ng - 1, Nz_eff + Ng - 1, u_plane_stride_);
                    if (r.first >= 0) best = r;
                }

                if (!is2D && best.first < 0) {
                    // +z, -z
                    r = find_fluid_nbr(cell_type_u_, idx, u_stride_, i, j, k, 0, 0, 1,
                                       Nx + Ng, Ny + Ng - 1, Nz + Ng - 1, u_plane_stride_);
                    if (r.first >= 0) best = r;
                    if (best.first < 0) {
                        r = find_fluid_nbr(cell_type_u_, idx, u_stride_, i, j, k, 0, 0, -1,
                                           Nx + Ng, Ny + Ng - 1, Nz + Ng - 1, u_plane_stride_);
                        if (r.first >= 0) best = r;
                    }
                }

                if (best.first >= 0) {
                    ghost_self_u_.push_back(idx);
                    ghost_nbr_u_.push_back(best.first);
                    ghost_alpha_u_.push_back(best.second);
                } else {
                    // No fluid neighbor found — treat as solid (u=0)
                    weight_u_[idx] = 0.0;
                }
            }
        }
    }

    // Process v-faces (similar but with v strides and v cell types)
    for (int k = Ng; k < Nz_eff + Ng; ++k) {
        for (int j = Ng; j <= Ny + Ng; ++j) {
            for (int i = Ng; i < Nx + Ng; ++i) {
                int idx = is2D ? (j * v_stride_ + i) : (k * v_plane_stride_ + j * v_stride_ + i);
                if (cell_type_v_[idx] != IBMCellType::Forcing) continue;

                std::pair<int, double> best = {-1, 0.0};

                // Use v-face coordinates for phi
                auto find_v = [&](int di, int dj, int dk) -> std::pair<int, double> {
                    for (int step = 1; step <= 2; ++step) {
                        int ni = i + step * di, nj = j + step * dj, nk = k + step * dk;
                        if (ni < Ng || ni >= Nx + Ng || nj < Ng || nj > Ny + Ng) return {-1, 0.0};
                        if (!is2D && (nk < Ng || nk >= Nz_eff + Ng)) return {-1, 0.0};
                        int nidx = is2D ? (nj * v_stride_ + ni) : (nk * v_plane_stride_ + nj * v_stride_ + ni);
                        if (cell_type_v_[nidx] == IBMCellType::Fluid) {
                            double phi_f = std::abs(body_->phi(mesh_->x(i), mesh_->yf[j], is2D ? 0.0 : mesh_->z(k)));
                            double phi_n = body_->phi(mesh_->x(ni), mesh_->yf[nj], is2D ? 0.0 : mesh_->z(nk));
                            if (phi_n <= 0.0) continue;
                            double alpha = std::max(0.0, std::min(1.0, phi_f / (phi_f + phi_n)));
                            return {nidx, alpha};
                        }
                    }
                    return {-1, 0.0};
                };

                for (auto [di, dj, dk] : std::vector<std::tuple<int,int,int>>{{1,0,0},{-1,0,0},{0,1,0},{0,-1,0}}) {
                    if (best.first >= 0) break;
                    best = find_v(di, dj, dk);
                }
                if (!is2D && best.first < 0) {
                    for (auto [di, dj, dk] : std::vector<std::tuple<int,int,int>>{{0,0,1},{0,0,-1}}) {
                        if (best.first >= 0) break;
                        best = find_v(di, dj, dk);
                    }
                }

                if (best.first >= 0) {
                    ghost_self_v_.push_back(idx);
                    ghost_nbr_v_.push_back(best.first);
                    ghost_alpha_v_.push_back(best.second);
                } else {
                    weight_v_[idx] = 0.0;
                }
            }
        }
    }

    // Process w-faces (3D only)
    if (!is2D) {
        for (int k = Ng; k <= Nz + Ng; ++k) {
            for (int j = Ng; j < Ny + Ng; ++j) {
                for (int i = Ng; i < Nx + Ng; ++i) {
                    int idx = k * w_plane_stride_ + j * w_stride_ + i;
                    if (cell_type_w_[idx] != IBMCellType::Forcing) continue;

                    std::pair<int, double> best = {-1, 0.0};

                    auto find_w = [&](int di, int dj, int dk) -> std::pair<int, double> {
                        for (int step = 1; step <= 2; ++step) {
                            int ni = i + step * di, nj = j + step * dj, nk = k + step * dk;
                            if (ni < Ng || ni >= Nx + Ng || nj < Ng || nj >= Ny + Ng) return {-1, 0.0};
                            if (nk < Ng || nk > Nz + Ng) return {-1, 0.0};
                            int nidx = nk * w_plane_stride_ + nj * w_stride_ + ni;
                            if (cell_type_w_[nidx] == IBMCellType::Fluid) {
                                double phi_f = std::abs(body_->phi(mesh_->x(i), mesh_->y(j), mesh_->zf[k]));
                                double phi_n = body_->phi(mesh_->x(ni), mesh_->y(nj), mesh_->zf[nk]);
                                if (phi_n <= 0.0) continue;
                                double alpha = std::max(0.0, std::min(1.0, phi_f / (phi_f + phi_n)));
                                return {nidx, alpha};
                            }
                        }
                        return {-1, 0.0};
                    };

                    for (auto [di, dj, dk] : std::vector<std::tuple<int,int,int>>{{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}}) {
                        if (best.first >= 0) break;
                        best = find_w(di, dj, dk);
                    }

                    if (best.first >= 0) {
                        ghost_self_w_.push_back(idx);
                        ghost_nbr_w_.push_back(best.first);
                        ghost_alpha_w_.push_back(best.second);
                    } else {
                        weight_w_[idx] = 0.0;
                    }
                }
            }
        }
    }

    n_ghost_u_ = static_cast<int>(ghost_self_u_.size());
    n_ghost_v_ = static_cast<int>(ghost_self_v_.size());
    n_ghost_w_ = static_cast<int>(ghost_self_w_.size());

    ghost_self_u_ptr_ = ghost_self_u_.data();
    ghost_nbr_u_ptr_ = ghost_nbr_u_.data();
    ghost_alpha_u_ptr_ = ghost_alpha_u_.data();
    ghost_self_v_ptr_ = ghost_self_v_.data();
    ghost_nbr_v_ptr_ = ghost_nbr_v_.data();
    ghost_alpha_v_ptr_ = ghost_alpha_v_.data();
    ghost_self_w_ptr_ = ghost_self_w_.data();
    ghost_nbr_w_ptr_ = ghost_nbr_w_.data();
    ghost_alpha_w_ptr_ = ghost_alpha_w_.data();

    std::cout << "[IBM] Ghost-cell: " << n_ghost_u_ << " u-faces, "
              << n_ghost_v_ << " v-faces, " << n_ghost_w_ << " w-faces\n";
}

void IBMForcing::compute_ghost_cell_interp_2nd() {
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Ng = mesh_->Nghost;
    const bool is2D = mesh_->is2D();

    if (!is2D) {
        // 3D trilinear (8-point) not yet implemented — use first-order fallback
        return;
    }

    // Resize and zero second-order arrays
    gc2_nbr_u_.assign(n_ghost_u_ * GC_STENCIL_SIZE, 0);
    gc2_wt_u_.assign(n_ghost_u_ * GC_STENCIL_SIZE, 0.0);
    gc2_nbr_v_.assign(n_ghost_v_ * GC_STENCIL_SIZE, 0);
    gc2_wt_v_.assign(n_ghost_v_ * GC_STENCIL_SIZE, 0.0);

    // Helper: compute bilinear stencil for mirror point (x_m, y_m) on a face grid.
    // x_coords/y_coords are the coordinate arrays for that face type.
    // Returns true if all 4 stencil cells are Fluid.
    auto build_bilinear = [&](
        double x_m, double y_m,
        const std::vector<IBMCellType>& ct, int stride,
        const double* x_coords, int x_lo, int x_hi,
        const double* y_coords, int y_lo, int y_hi,
        int* nbr_out, double* wt_out) -> bool
    {
        // Find i0: largest index where x_coords[i] <= x_m
        int i0 = -1;
        for (int i = x_lo; i < x_hi; ++i) {
            if (x_coords[i] <= x_m) i0 = i;
            else break;
        }
        // Require 2-cell margin from boundaries for numerical safety
        if (i0 < x_lo + 1 || i0 + 1 >= x_hi) return false;
        int i1 = i0 + 1;

        // Find j0: largest index where y_coords[j] <= y_m
        int j0 = -1;
        for (int j = y_lo; j < y_hi; ++j) {
            if (y_coords[j] <= y_m) j0 = j;
            else break;
        }
        if (j0 < y_lo + 1 || j0 + 1 >= y_hi) return false;
        int j1 = j0 + 1;

        // Bilinear parameters
        double x0 = x_coords[i0], x1 = x_coords[i1];
        double y0 = y_coords[j0], y1 = y_coords[j1];
        double Lx = x1 - x0, Ly = y1 - y0;
        if (Lx < 1e-30 || Ly < 1e-30) return false;

        double tx = std::max(0.0, std::min(1.0, (x_m - x0) / Lx));
        double ty = std::max(0.0, std::min(1.0, (y_m - y0) / Ly));

        // 4 corner flat indices
        int idx00 = j0 * stride + i0;
        int idx10 = j0 * stride + i1;
        int idx01 = j1 * stride + i0;
        int idx11 = j1 * stride + i1;

        // All 4 must be Fluid
        if (idx00 < 0 || idx00 >= static_cast<int>(ct.size()) ||
            idx10 < 0 || idx10 >= static_cast<int>(ct.size()) ||
            idx01 < 0 || idx01 >= static_cast<int>(ct.size()) ||
            idx11 < 0 || idx11 >= static_cast<int>(ct.size()))
            return false;

        if (ct[idx00] != IBMCellType::Fluid || ct[idx10] != IBMCellType::Fluid ||
            ct[idx01] != IBMCellType::Fluid || ct[idx11] != IBMCellType::Fluid)
            return false;

        nbr_out[0] = idx00; wt_out[0] = (1 - tx) * (1 - ty);
        nbr_out[1] = idx10; wt_out[1] = tx * (1 - ty);
        nbr_out[2] = idx01; wt_out[2] = (1 - tx) * ty;
        nbr_out[3] = idx11; wt_out[3] = tx * ty;
        return true;
    };

    int n_valid_u = 0, n_valid_v = 0;

    // Process u-face ghost cells
    for (int g = 0; g < n_ghost_u_; ++g) {
        int idx = ghost_self_u_[g];
        int j = idx / u_stride_;
        int i = idx % u_stride_;

        // u-face position
        double x_g = mesh_->xf[i];
        double y_g = mesh_->y(j);  // yc[j]

        // Mirror point
        auto [x_s, y_s, z_s] = body_->closest_point(x_g, y_g, 0.0);
        double x_m = 2.0 * x_s - x_g;
        double y_m = 2.0 * y_s - y_g;

        // Build bilinear stencil on u-face grid: x=xf, y=yc
        bool ok = build_bilinear(
            x_m, y_m,
            cell_type_u_, u_stride_,
            mesh_->xf.data(), Ng, Nx + Ng,
            mesh_->yc.data(), Ng, Ny + Ng - 1,
            &gc2_nbr_u_[g * GC_STENCIL_SIZE],
            &gc2_wt_u_[g * GC_STENCIL_SIZE]);
        if (ok) n_valid_u++;
    }

    // Process v-face ghost cells
    for (int g = 0; g < n_ghost_v_; ++g) {
        int idx = ghost_self_v_[g];
        int j = idx / v_stride_;
        int i = idx % v_stride_;

        // v-face position
        double x_g = mesh_->xc[i];  // cell center x
        double y_g = mesh_->yf[j];  // face y

        auto [x_s, y_s, z_s] = body_->closest_point(x_g, y_g, 0.0);
        double x_m = 2.0 * x_s - x_g;
        double y_m = 2.0 * y_s - y_g;

        // Build bilinear stencil on v-face grid: x=xc, y=yf
        bool ok = build_bilinear(
            x_m, y_m,
            cell_type_v_, v_stride_,
            mesh_->xc.data(), Ng, Nx + Ng - 1,
            mesh_->yf.data(), Ng, Ny + Ng,
            &gc2_nbr_v_[g * GC_STENCIL_SIZE],
            &gc2_wt_v_[g * GC_STENCIL_SIZE]);
        if (ok) n_valid_v++;
    }

    // Set GPU pointers
    gc2_nbr_u_ptr_ = gc2_nbr_u_.data();
    gc2_wt_u_ptr_ = gc2_wt_u_.data();
    gc2_nbr_v_ptr_ = gc2_nbr_v_.data();
    gc2_wt_v_ptr_ = gc2_wt_v_.data();

    std::cerr << "[IBM] 2nd-order ghost-cell: "
              << n_valid_u << "/" << n_ghost_u_ << " u-faces, "
              << n_valid_v << "/" << n_ghost_v_ << " v-faces valid\n";
}

void IBMForcing::exclude_wall_cells(bool y_lo_wall, bool y_hi_wall,
                                     bool z_lo_wall, bool z_hi_wall) {
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Ng = mesh_->Nghost;
    const bool is2D = mesh_->is2D();
    const int Nz = std::max(mesh_->Nz, 1);
    int Nz_eff = is2D ? 1 : Nz;

    // Set weight=1.0 (no IBM forcing) at wall-adjacent cells where BCs already enforce no-slip.
    // This prevents double-enforcement that causes instability with small penalization eta.
    for (int k = Ng; k < Nz_eff + Ng; ++k) {
        for (int i = Ng; i <= Nx + Ng; ++i) {
            if (y_lo_wall) {
                int idx = is2D ? (Ng * u_stride_ + i) : (k * u_plane_stride_ + Ng * u_stride_ + i);
                weight_u_[idx] = 1.0;
            }
            if (y_hi_wall) {
                int j = Ny + Ng - 1;
                int idx = is2D ? (j * u_stride_ + i) : (k * u_plane_stride_ + j * u_stride_ + i);
                weight_u_[idx] = 1.0;
            }
        }
        for (int i = Ng; i < Nx + Ng; ++i) {
            if (y_lo_wall) {
                int idx0 = is2D ? (Ng * v_stride_ + i) : (k * v_plane_stride_ + Ng * v_stride_ + i);
                int idx1 = is2D ? ((Ng + 1) * v_stride_ + i) : (k * v_plane_stride_ + (Ng + 1) * v_stride_ + i);
                weight_v_[idx0] = 1.0;
                weight_v_[idx1] = 1.0;
            }
            if (y_hi_wall) {
                int j = Ny + Ng;
                int idx0 = is2D ? (j * v_stride_ + i) : (k * v_plane_stride_ + j * v_stride_ + i);
                int idx1 = is2D ? ((j - 1) * v_stride_ + i) : (k * v_plane_stride_ + (j - 1) * v_stride_ + i);
                weight_v_[idx0] = 1.0;
                weight_v_[idx1] = 1.0;
            }
        }
    }

    weight_u_ptr_ = weight_u_.data();
    weight_v_ptr_ = weight_v_.data();
    std::cout << "[IBM] Excluded wall cells from forcing (y_lo=" << y_lo_wall
              << ", y_hi=" << y_hi_wall << ")\n";
}

void IBMForcing::recompute_and_remap() {
    // Unmap old data from GPU if mapped
    if (gpu_mapped_) {
        unmap_from_gpu();
    }
    // Recompute weights (and ghost-cell stencils if enabled)
    compute_weights();
    // Re-map to GPU
    map_to_gpu();
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

    // Ghost-cell arrays
    if (ghost_cell_ibm_ && n_ghost_u_ > 0) {
        #pragma omp target enter data map(to: ghost_self_u_ptr_[0:n_ghost_u_])
        #pragma omp target enter data map(to: ghost_nbr_u_ptr_[0:n_ghost_u_])
        #pragma omp target enter data map(to: ghost_alpha_u_ptr_[0:n_ghost_u_])
    }
    if (ghost_cell_ibm_ && n_ghost_v_ > 0) {
        #pragma omp target enter data map(to: ghost_self_v_ptr_[0:n_ghost_v_])
        #pragma omp target enter data map(to: ghost_nbr_v_ptr_[0:n_ghost_v_])
        #pragma omp target enter data map(to: ghost_alpha_v_ptr_[0:n_ghost_v_])
    }
    if (ghost_cell_ibm_ && n_ghost_w_ > 0) {
        #pragma omp target enter data map(to: ghost_self_w_ptr_[0:n_ghost_w_])
        #pragma omp target enter data map(to: ghost_nbr_w_ptr_[0:n_ghost_w_])
        #pragma omp target enter data map(to: ghost_alpha_w_ptr_[0:n_ghost_w_])
    }

    // Second-order ghost-cell arrays (bilinear stencil)
    if (ghost_cell_ibm_ && !gc2_nbr_u_.empty()) {
        int gc2_u_n = static_cast<int>(gc2_nbr_u_.size());
        int gc2_v_n = static_cast<int>(gc2_nbr_v_.size());
        #pragma omp target enter data map(to: gc2_nbr_u_ptr_[0:gc2_u_n])
        #pragma omp target enter data map(to: gc2_wt_u_ptr_[0:gc2_u_n])
        if (gc2_v_n > 0) {
            #pragma omp target enter data map(to: gc2_nbr_v_ptr_[0:gc2_v_n])
            #pragma omp target enter data map(to: gc2_wt_v_ptr_[0:gc2_v_n])
        }
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

    if (ghost_cell_ibm_ && n_ghost_u_ > 0) {
        #pragma omp target exit data map(delete: ghost_self_u_ptr_[0:n_ghost_u_])
        #pragma omp target exit data map(delete: ghost_nbr_u_ptr_[0:n_ghost_u_])
        #pragma omp target exit data map(delete: ghost_alpha_u_ptr_[0:n_ghost_u_])
    }
    if (ghost_cell_ibm_ && n_ghost_v_ > 0) {
        #pragma omp target exit data map(delete: ghost_self_v_ptr_[0:n_ghost_v_])
        #pragma omp target exit data map(delete: ghost_nbr_v_ptr_[0:n_ghost_v_])
        #pragma omp target exit data map(delete: ghost_alpha_v_ptr_[0:n_ghost_v_])
    }
    if (ghost_cell_ibm_ && n_ghost_w_ > 0) {
        #pragma omp target exit data map(delete: ghost_self_w_ptr_[0:n_ghost_w_])
        #pragma omp target exit data map(delete: ghost_nbr_w_ptr_[0:n_ghost_w_])
        #pragma omp target exit data map(delete: ghost_alpha_w_ptr_[0:n_ghost_w_])
    }

    // Second-order ghost-cell arrays
    if (ghost_cell_ibm_ && !gc2_nbr_u_.empty()) {
        int gc2_u_n = static_cast<int>(gc2_nbr_u_.size());
        int gc2_v_n = static_cast<int>(gc2_nbr_v_.size());
        #pragma omp target exit data map(delete: gc2_nbr_u_ptr_[0:gc2_u_n])
        #pragma omp target exit data map(delete: gc2_wt_u_ptr_[0:gc2_u_n])
        if (gc2_v_n > 0) {
            #pragma omp target exit data map(delete: gc2_nbr_v_ptr_[0:gc2_v_n])
            #pragma omp target exit data map(delete: gc2_wt_v_ptr_[0:gc2_v_n])
        }
    }

    gpu_mapped_ = false;
}

void IBMForcing::reset_force_accumulator() {
    last_Fx_ = 0.0;
    last_Fy_ = 0.0;
    last_Fz_ = 0.0;
}

void IBMForcing::apply_forcing(VectorField& vel, double dt) {
    current_dt_ = dt;  // store for post-correction force accumulation
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = std::max(mesh_->Nz, 1);
    const int Ng = mesh_->Nghost;
    const bool is2D = mesh_->is2D();
    int Nz_eff = is2D ? 1 : Nz;

    // Force accumulation: only when enabled (expensive loops over all cells).
    if (accumulate_forces_ && dt > 0.0) {
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

    // CPU path: semi-implicit volume penalization (same formula as GPU path)
    // When eta < dt, use eta_eff = eta to get strong enforcement.
    // When eta >= dt, dt/eta is small → weak enforcement.
    // For resolution-independent penalization, set eta proportional to dt:
    //   eta_eff = max(penalization_eta_, 0) → dt/eta_eff = dt/eta
    const double dt_over_eta = (penalization_eta_ > 0.0 && dt > 0.0)
                                ? dt / penalization_eta_ : 0.0;

    for (int k = Ng; k < Nz_eff + Ng; ++k) {
        for (int j = Ng; j < Ny + Ng; ++j) {
            for (int i = Ng; i <= Nx + Ng; ++i) {
                int idx = is2D ? (j * u_stride_ + i) : (k * u_plane_stride_ + j * u_stride_ + i);
                double w = weight_u_[idx];
                double factor = (dt_over_eta > 0.0) ? 1.0 / (1.0 + (1.0 - w) * dt_over_eta) : w;
                if (is2D) vel.u(i, j) *= factor;
                else vel.u(i, j, k) *= factor;
            }
        }
    }

    for (int k = Ng; k < Nz_eff + Ng; ++k) {
        for (int j = Ng; j <= Ny + Ng; ++j) {
            for (int i = Ng; i < Nx + Ng; ++i) {
                int idx = is2D ? (j * v_stride_ + i) : (k * v_plane_stride_ + j * v_stride_ + i);
                double w = weight_v_[idx];
                double factor = (dt_over_eta > 0.0) ? 1.0 / (1.0 + (1.0 - w) * dt_over_eta) : w;
                if (is2D) vel.v(i, j) *= factor;
                else vel.v(i, j, k) *= factor;
            }
        }
    }

    if (!is2D) {
        for (int k = Ng; k <= Nz + Ng; ++k) {
            for (int j = Ng; j < Ny + Ng; ++j) {
                for (int i = Ng; i < Nx + Ng; ++i) {
                    int idx = k * w_plane_stride_ + j * w_stride_ + i;
                    double w = weight_w_[idx];
                    double factor = (dt_over_eta > 0.0) ? 1.0 / (1.0 + (1.0 - w) * dt_over_eta) : w;
                    vel.w(i, j, k) *= factor;
                }
            }
        }
    }

    // Ghost-cell mirror is applied separately in apply_ghost_cell() post-correction.
}

void IBMForcing::apply_forcing_device(double* u_ptr, double* v_ptr, double* w_ptr, double dt) {
    if (!gpu_mapped_)
        throw std::runtime_error("[IBM] apply_forcing_device called before map_to_gpu()");

    current_dt_ = dt;  // store for post-correction force accumulation
    double* wu = weight_u_ptr_;
    double* wv = weight_v_ptr_;
    double* ww = weight_w_ptr_;
    const int u_n = static_cast<int>(u_total_);
    const int v_n = static_cast<int>(v_total_);
    const int w_n = static_cast<int>(w_total_);

    // Force accumulation: only when enabled (expensive GPU reductions).
    // Callers enable this at output intervals via set_accumulate_forces(true).
    if (accumulate_forces_ && dt > 0.0) {
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

    // Semi-implicit volume penalization (Angot et al. 1999):
    //   u = u / (1 + (1-w) * dt/eta)
    // Unconditionally stable for any dt/eta. When eta=0: hard forcing u *= w.
    const double dt_over_eta = (penalization_eta_ > 0.0 && dt > 0.0)
                                ? dt / penalization_eta_ : 0.0;

    if (dt_over_eta > 0.0) {
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_n], wu[0:u_n]) firstprivate(dt_over_eta)
        for (int i = 0; i < u_n; ++i) {
            u_ptr[i] /= 1.0 + (1.0 - wu[i]) * dt_over_eta;
        }
        #pragma omp target teams distribute parallel for \
            map(present: v_ptr[0:v_n], wv[0:v_n]) firstprivate(dt_over_eta)
        for (int i = 0; i < v_n; ++i) {
            v_ptr[i] /= 1.0 + (1.0 - wv[i]) * dt_over_eta;
        }
        if (w_ptr && ww && w_n > 0) {
            #pragma omp target teams distribute parallel for \
                map(present: w_ptr[0:w_n], ww[0:w_n]) firstprivate(dt_over_eta)
            for (int i = 0; i < w_n; ++i) {
                w_ptr[i] /= 1.0 + (1.0 - ww[i]) * dt_over_eta;
            }
        }
    } else {
        // Hard forcing fallback (eta=0): u *= w
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

    // Note: ghost-cell scatter (mirror condition) is applied separately in
    // apply_ghost_cell_device() AFTER the Poisson pressure correction.
    // The pre-correction path here only zeroes solid cells via weight multiply.
    // Forcing cells have weight=1.0 (no modification here).

    // Ghost-cell interpolation (pre-correction): u[self] = u[fluid_nbr] * alpha
    // Linearly interpolates velocity to zero at body surface.
    // Accumulate ghost-cell forces: F = Σ (u_before - u_after) / dt * dV
    // This captures the momentum removed by the ghost-cell interpolation.
    if (ghost_cell_ibm_ && n_ghost_u_ > 0) {
        int* su = ghost_self_u_ptr_;
        int* nu_g = ghost_nbr_u_ptr_;
        double* au = ghost_alpha_u_ptr_;
        [[maybe_unused]] const int ngu = n_ghost_u_;

        if (accumulate_forces_ && dt > 0.0) {
            [[maybe_unused]] const double dV = mesh_->dx * mesh_->dy * (mesh_->is2D() ? 1.0 : mesh_->dz);
            double Fx_gc = 0.0;
            #pragma omp target teams distribute parallel for reduction(+:Fx_gc) \
                map(present: u_ptr[0:u_n], su[0:ngu], nu_g[0:ngu], au[0:ngu])
            for (int g = 0; g < ngu; ++g) {
                double u_before = u_ptr[su[g]];
                double u_after = u_ptr[nu_g[g]] * au[g];
                Fx_gc += (u_before - u_after);
            }
            last_Fx_ += Fx_gc / dt * dV;
        }

        // Pre-correction: use FIRST-ORDER only (gentle, keeps div(u*) small)
        // Second-order mirror is too aggressive for pre-correction and causes instability
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_n], su[0:ngu], nu_g[0:ngu], au[0:ngu])
        for (int g = 0; g < ngu; ++g) {
            u_ptr[su[g]] = u_ptr[nu_g[g]] * au[g];
        }

        int* sv = ghost_self_v_ptr_;
        int* nv_g = ghost_nbr_v_ptr_;
        double* av = ghost_alpha_v_ptr_;
        [[maybe_unused]] const int ngv = n_ghost_v_;

        if (accumulate_forces_ && dt > 0.0) {
            [[maybe_unused]] const double dV = mesh_->dx * mesh_->dy * (mesh_->is2D() ? 1.0 : mesh_->dz);
            double Fy_gc = 0.0;
            #pragma omp target teams distribute parallel for reduction(+:Fy_gc) \
                map(present: v_ptr[0:v_n], sv[0:ngv], nv_g[0:ngv], av[0:ngv])
            for (int g = 0; g < ngv; ++g) {
                double v_before = v_ptr[sv[g]];
                double v_after = v_ptr[nv_g[g]] * av[g];
                Fy_gc += (v_before - v_after);
            }
            last_Fy_ += Fy_gc / dt * dV;
        }

        // Pre-correction v: first-order only
        #pragma omp target teams distribute parallel for \
            map(present: v_ptr[0:v_n], sv[0:ngv], nv_g[0:ngv], av[0:ngv])
        for (int g = 0; g < ngv; ++g) {
            v_ptr[sv[g]] = v_ptr[nv_g[g]] * av[g];
        }

        if (w_ptr && n_ghost_w_ > 0) {
            int* sw = ghost_self_w_ptr_;
            int* nw_g = ghost_nbr_w_ptr_;
            double* aw = ghost_alpha_w_ptr_;
            [[maybe_unused]] const int ngw = n_ghost_w_;

            if (accumulate_forces_ && dt > 0.0) {
                [[maybe_unused]] const double dV = mesh_->dx * mesh_->dy * mesh_->dz;
                double Fz_gc = 0.0;
                #pragma omp target teams distribute parallel for reduction(+:Fz_gc) \
                    map(present: w_ptr[0:w_n], sw[0:ngw], nw_g[0:ngw], aw[0:ngw])
                for (int g = 0; g < ngw; ++g) {
                    double w_before = w_ptr[sw[g]];
                    double w_after = w_ptr[nw_g[g]] * aw[g];
                    Fz_gc += (w_before - w_after);
                }
                last_Fz_ += Fz_gc / dt * dV;
            }

            #pragma omp target teams distribute parallel for \
                map(present: w_ptr[0:w_n], sw[0:ngw], nw_g[0:ngw], aw[0:ngw])
            for (int g = 0; g < ngw; ++g) {
                w_ptr[sw[g]] = w_ptr[nw_g[g]] * aw[g];
            }
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

void IBMForcing::apply_ghost_cell(double* u_ptr, double* v_ptr, double* w_ptr) {
    if (!ghost_cell_ibm_) return;

    // Re-enforce ghost-cell interpolation after pressure correction.
    // u_ghost = u_fluid * alpha (positive — interpolates toward zero at surface).
    // The pressure correction may have overwritten these values with grad(p).
    // Accumulate forces: the momentum removed here is part of the total IBM force.
    [[maybe_unused]] const int u_n = static_cast<int>(u_total_);
    [[maybe_unused]] const int v_n = static_cast<int>(v_total_);
    [[maybe_unused]] const int w_n = static_cast<int>(w_total_);

    if (n_ghost_u_ > 0) {
        int* su = ghost_self_u_ptr_;
        int* nu_g = ghost_nbr_u_ptr_;
        double* au = ghost_alpha_u_ptr_;
        int* gc2_n_u = gc2_nbr_u_ptr_;
        double* gc2_w_u = gc2_wt_u_ptr_;
        [[maybe_unused]] const int ngu = n_ghost_u_;
        [[maybe_unused]] const int ngu_s = n_ghost_u_ * GC_STENCIL_SIZE;
        [[maybe_unused]] const int S = GC_STENCIL_SIZE;

        if (accumulate_forces_ && current_dt_ > 0.0) {
            [[maybe_unused]] const double dV = mesh_->dx * mesh_->dy * (mesh_->is2D() ? 1.0 : mesh_->dz);
            double Fx_gc = 0.0;
            #pragma omp target teams distribute parallel for reduction(+:Fx_gc) \
                map(present: u_ptr[0:u_n], su[0:ngu], nu_g[0:ngu], au[0:ngu], \
                             gc2_n_u[0:ngu_s], gc2_w_u[0:ngu_s])
            for (int g = 0; g < ngu; ++g) {
                double wsum = gc2_w_u[g*S] + gc2_w_u[g*S+1] + gc2_w_u[g*S+2] + gc2_w_u[g*S+3];
                double u_after;
                if (wsum > 0.5) {
                    double u_mirror = 0.0;
                    for (int k = 0; k < S; ++k)
                        u_mirror += gc2_w_u[g*S+k] * u_ptr[gc2_n_u[g*S+k]];
                    u_after = u_ptr[nu_g[g]] * au[g];
                } else {
                    u_after = u_ptr[nu_g[g]] * au[g];
                }
                Fx_gc += (u_ptr[su[g]] - u_after);
            }
            last_Fx_ += Fx_gc / current_dt_ * dV;
        }

        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_n], su[0:ngu], nu_g[0:ngu], au[0:ngu], \
                         gc2_n_u[0:ngu_s], gc2_w_u[0:ngu_s])
        for (int g = 0; g < ngu; ++g) {
            double wsum = gc2_w_u[g*S] + gc2_w_u[g*S+1] + gc2_w_u[g*S+2] + gc2_w_u[g*S+3];
            if (wsum > 0.5) {
                double u_mirror = 0.0;
                for (int k = 0; k < S; ++k)
                    u_mirror += gc2_w_u[g*S+k] * u_ptr[gc2_n_u[g*S+k]];
                // Second-order mirror disabled — causes instability in explicit RK3.
                // Fall back to first-order for all ghost cells.
                // TODO: implement implicit or iterative mirror for explicit solvers.
                u_ptr[su[g]] = u_ptr[nu_g[g]] * au[g];
            } else {
                u_ptr[su[g]] = u_ptr[nu_g[g]] * au[g];
            }
        }
    }

    if (n_ghost_v_ > 0) {
        int* sv = ghost_self_v_ptr_;
        int* nv_g = ghost_nbr_v_ptr_;
        double* av = ghost_alpha_v_ptr_;
        int* gc2_n_v = gc2_nbr_v_ptr_;
        double* gc2_w_v = gc2_wt_v_ptr_;
        [[maybe_unused]] const int ngv = n_ghost_v_;
        [[maybe_unused]] const int ngv_s = n_ghost_v_ * GC_STENCIL_SIZE;
        [[maybe_unused]] const int S = GC_STENCIL_SIZE;

        if (accumulate_forces_ && current_dt_ > 0.0) {
            [[maybe_unused]] const double dV = mesh_->dx * mesh_->dy * (mesh_->is2D() ? 1.0 : mesh_->dz);
            double Fy_gc = 0.0;
            #pragma omp target teams distribute parallel for reduction(+:Fy_gc) \
                map(present: v_ptr[0:v_n], sv[0:ngv], nv_g[0:ngv], av[0:ngv], \
                             gc2_n_v[0:ngv_s], gc2_w_v[0:ngv_s])
            for (int g = 0; g < ngv; ++g) {
                double wsum = gc2_w_v[g*S] + gc2_w_v[g*S+1] + gc2_w_v[g*S+2] + gc2_w_v[g*S+3];
                double v_after;
                if (wsum > 0.5) {
                    double v_mirror = 0.0;
                    for (int k = 0; k < S; ++k)
                        v_mirror += gc2_w_v[g*S+k] * v_ptr[gc2_n_v[g*S+k]];
                    v_after = v_ptr[nv_g[g]] * av[g];
                } else {
                    v_after = v_ptr[nv_g[g]] * av[g];
                }
                Fy_gc += (v_ptr[sv[g]] - v_after);
            }
            last_Fy_ += Fy_gc / current_dt_ * dV;
        }

        #pragma omp target teams distribute parallel for \
            map(present: v_ptr[0:v_n], sv[0:ngv], nv_g[0:ngv], av[0:ngv], \
                         gc2_n_v[0:ngv_s], gc2_w_v[0:ngv_s])
        for (int g = 0; g < ngv; ++g) {
            double wsum = gc2_w_v[g*S] + gc2_w_v[g*S+1] + gc2_w_v[g*S+2] + gc2_w_v[g*S+3];
            if (wsum > 0.5) {
                double v_mirror = 0.0;
                for (int k = 0; k < S; ++k)
                    v_mirror += gc2_w_v[g*S+k] * v_ptr[gc2_n_v[g*S+k]];
                v_ptr[sv[g]] = v_ptr[nv_g[g]] * av[g];
            } else {
                v_ptr[sv[g]] = v_ptr[nv_g[g]] * av[g];
            }
        }
    }

    if (w_ptr && n_ghost_w_ > 0) {
        int* sw = ghost_self_w_ptr_;
        int* nw_g = ghost_nbr_w_ptr_;
        double* aw = ghost_alpha_w_ptr_;
        [[maybe_unused]] const int ngw = n_ghost_w_;

        if (accumulate_forces_ && current_dt_ > 0.0) {
            [[maybe_unused]] const double dV = mesh_->dx * mesh_->dy * mesh_->dz;
            double Fz_gc = 0.0;
            #pragma omp target teams distribute parallel for reduction(+:Fz_gc) \
                map(present: w_ptr[0:w_n], sw[0:ngw], nw_g[0:ngw], aw[0:ngw])
            for (int g = 0; g < ngw; ++g) {
                Fz_gc += (w_ptr[sw[g]] - w_ptr[nw_g[g]] * aw[g]);
            }
            last_Fz_ += Fz_gc / current_dt_ * dV;
        }

        #pragma omp target teams distribute parallel for \
            map(present: w_ptr[0:w_n], sw[0:ngw], nw_g[0:ngw], aw[0:ngw])
        for (int g = 0; g < ngw; ++g) {
            w_ptr[sw[g]] = w_ptr[nw_g[g]] * aw[g];
        }
    }
}

} // namespace nncfd
