/// @file solver_time.cpp
/// @brief Time integration methods for RANSSolver
///
/// Uses kernel helper functions from solver_time_kernels_*.cpp to avoid
/// exceeding nvc++ compiler's pragma limits.

#include "solver.hpp"
#include "solver_time_kernels.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace nncfd {

// ============================================================================
// Helper: Identify velocity field for pointer lookup (used by project_velocity)
// ============================================================================

namespace {
enum class VelFieldId { Velocity, VelocityStar, VelocityOld, VelocityRk, Unknown };
} // anonymous namespace

// ============================================================================
// RK Time Integration Methods
// ============================================================================

void RANSSolver::euler_substep(VectorField& vel_in, VectorField& vel_out, double dt) {
    // Compute convective and diffusive terms for vel_in.
    // The compute functions now use get_velocity_ptrs() to properly handle
    // any VectorField (velocity_, velocity_star_, etc.) without swapping.
    compute_convective_term(vel_in, conv_);
    compute_diffusive_term(vel_in, nu_eff_, diff_);

    // Get pointers for vel_in and vel_out
    auto get_ptrs = [this](VectorField& vel, double*& u, double*& v, double*& w) {
        if (&vel == &velocity_) {
            u = velocity_u_ptr_; v = velocity_v_ptr_; w = velocity_w_ptr_;
        } else if (&vel == &velocity_star_) {
            u = velocity_star_u_ptr_; v = velocity_star_v_ptr_; w = velocity_star_w_ptr_;
        } else if (&vel == &velocity_old_) {
            u = velocity_old_u_ptr_; v = velocity_old_v_ptr_; w = velocity_old_w_ptr_;
        } else if (&vel == &velocity_rk_) {
            u = velocity_rk_u_ptr_; v = velocity_rk_v_ptr_; w = velocity_rk_w_ptr_;
        }
    };

    double *u_in, *v_in, *w_in, *u_out, *v_out, *w_out;
    get_ptrs(vel_in, u_in, v_in, w_in);
    get_ptrs(vel_out, u_out, v_out, w_out);

    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Ng = mesh_->Nghost;
    const int u_stride = Nx + 2 * Ng + 1;
    const int v_stride = Nx + 2 * Ng;
    const size_t u_total = vel_in.u_total_size();
    const size_t v_total = vel_in.v_total_size();

    const bool x_periodic = (velocity_bc_.x_lo == VelocityBC::Periodic) &&
                            (velocity_bc_.x_hi == VelocityBC::Periodic);
    const bool y_periodic = (velocity_bc_.y_lo == VelocityBC::Periodic) &&
                            (velocity_bc_.y_hi == VelocityBC::Periodic);

    if (mesh_->is2D()) {
        // INLINE euler_advance with explicit code paths for each case
        // This ensures member pointers are used directly in the pragmas
        const double fx = fx_;
        const double fy = fy_;

        // Case: velocity_ -> velocity_star_
        if (&vel_in == &velocity_ && &vel_out == &velocity_star_) {
            // NVHPC workaround: Use dev_ptr() + is_device_ptr pattern
            const double* u_in_dev = gpu::dev_ptr(velocity_u_ptr_);
            double* u_out_dev = gpu::dev_ptr(velocity_star_u_ptr_);
            const double* conv_u_dev = gpu::dev_ptr(conv_u_ptr_);
            const double* diff_u_dev = gpu::dev_ptr(diff_u_ptr_);
            const double* v_in_dev = gpu::dev_ptr(velocity_v_ptr_);
            double* v_out_dev = gpu::dev_ptr(velocity_star_v_ptr_);
            const double* conv_v_dev = gpu::dev_ptr(conv_v_ptr_);
            const double* diff_v_dev = gpu::dev_ptr(diff_v_ptr_);

            #pragma omp target teams distribute parallel for collapse(2) \
                is_device_ptr(u_in_dev, u_out_dev, conv_u_dev, diff_u_dev)
            for (int j = Ng; j < Ng + Ny; ++j) {
                for (int i = Ng; i <= Ng + Nx; ++i) {
                    int idx = j * u_stride + i;
                    u_out_dev[idx] = u_in_dev[idx] + dt * (-conv_u_dev[idx] + diff_u_dev[idx] + fx);
                }
            }
            #pragma omp target teams distribute parallel for collapse(2) \
                is_device_ptr(v_in_dev, v_out_dev, conv_v_dev, diff_v_dev)
            for (int j = Ng; j <= Ng + Ny; ++j) {
                for (int i = Ng; i < Ng + Nx; ++i) {
                    int idx = j * v_stride + i;
                    v_out_dev[idx] = v_in_dev[idx] + dt * (-conv_v_dev[idx] + diff_v_dev[idx] + fy);
                }
            }

            // Periodicity for velocity_star_
            if (x_periodic) {
                #pragma omp target teams distribute parallel for is_device_ptr(u_out_dev)
                for (int j = Ng; j < Ng + Ny; ++j) {
                    u_out_dev[j * u_stride + Ng + Nx] = u_out_dev[j * u_stride + Ng];
                }
            }
            if (y_periodic) {
                #pragma omp target teams distribute parallel for is_device_ptr(v_out_dev)
                for (int i = Ng; i < Ng + Nx; ++i) {
                    v_out_dev[(Ng + Ny) * v_stride + i] = v_out_dev[Ng * v_stride + i];
                }
            }
        }
        // Case: velocity_star_ -> velocity_
        else if (&vel_in == &velocity_star_ && &vel_out == &velocity_) {
            // NVHPC workaround: Use dev_ptr() + is_device_ptr pattern
            const double* u_in_dev = gpu::dev_ptr(velocity_star_u_ptr_);
            double* u_out_dev = gpu::dev_ptr(velocity_u_ptr_);
            const double* conv_u_dev = gpu::dev_ptr(conv_u_ptr_);
            const double* diff_u_dev = gpu::dev_ptr(diff_u_ptr_);
            const double* v_in_dev = gpu::dev_ptr(velocity_star_v_ptr_);
            double* v_out_dev = gpu::dev_ptr(velocity_v_ptr_);
            const double* conv_v_dev = gpu::dev_ptr(conv_v_ptr_);
            const double* diff_v_dev = gpu::dev_ptr(diff_v_ptr_);

            #pragma omp target teams distribute parallel for collapse(2) \
                is_device_ptr(u_in_dev, u_out_dev, conv_u_dev, diff_u_dev)
            for (int j = Ng; j < Ng + Ny; ++j) {
                for (int i = Ng; i <= Ng + Nx; ++i) {
                    int idx = j * u_stride + i;
                    u_out_dev[idx] = u_in_dev[idx] + dt * (-conv_u_dev[idx] + diff_u_dev[idx] + fx);
                }
            }
            #pragma omp target teams distribute parallel for collapse(2) \
                is_device_ptr(v_in_dev, v_out_dev, conv_v_dev, diff_v_dev)
            for (int j = Ng; j <= Ng + Ny; ++j) {
                for (int i = Ng; i < Ng + Nx; ++i) {
                    int idx = j * v_stride + i;
                    v_out_dev[idx] = v_in_dev[idx] + dt * (-conv_v_dev[idx] + diff_v_dev[idx] + fy);
                }
            }
            // Periodicity for velocity_
            if (x_periodic) {
                #pragma omp target teams distribute parallel for is_device_ptr(u_out_dev)
                for (int j = Ng; j < Ng + Ny; ++j) {
                    u_out_dev[j * u_stride + Ng + Nx] = u_out_dev[j * u_stride + Ng];
                }
            }
            if (y_periodic) {
                #pragma omp target teams distribute parallel for is_device_ptr(v_out_dev)
                for (int i = Ng; i < Ng + Nx; ++i) {
                    v_out_dev[(Ng + Ny) * v_stride + i] = v_out_dev[Ng * v_stride + i];
                }
            }
        }
        // Case: velocity_ -> velocity_ (in-place, for Euler integrator)
        else if (&vel_in == &velocity_ && &vel_out == &velocity_) {
            // NVHPC workaround: Use dev_ptr() + is_device_ptr pattern
            double* u_dev = gpu::dev_ptr(velocity_u_ptr_);
            const double* conv_u_dev = gpu::dev_ptr(conv_u_ptr_);
            const double* diff_u_dev = gpu::dev_ptr(diff_u_ptr_);
            double* v_dev = gpu::dev_ptr(velocity_v_ptr_);
            const double* conv_v_dev = gpu::dev_ptr(conv_v_ptr_);
            const double* diff_v_dev = gpu::dev_ptr(diff_v_ptr_);

            #pragma omp target teams distribute parallel for collapse(2) \
                is_device_ptr(u_dev, conv_u_dev, diff_u_dev)
            for (int j = Ng; j < Ng + Ny; ++j) {
                for (int i = Ng; i <= Ng + Nx; ++i) {
                    int idx = j * u_stride + i;
                    u_dev[idx] = u_dev[idx] + dt * (-conv_u_dev[idx] + diff_u_dev[idx] + fx);
                }
            }
            #pragma omp target teams distribute parallel for collapse(2) \
                is_device_ptr(v_dev, conv_v_dev, diff_v_dev)
            for (int j = Ng; j <= Ng + Ny; ++j) {
                for (int i = Ng; i < Ng + Nx; ++i) {
                    int idx = j * v_stride + i;
                    v_dev[idx] = v_dev[idx] + dt * (-conv_v_dev[idx] + diff_v_dev[idx] + fy);
                }
            }
            // Periodicity for velocity_
            if (x_periodic) {
                #pragma omp target teams distribute parallel for is_device_ptr(u_dev)
                for (int j = Ng; j < Ng + Ny; ++j) {
                    u_dev[j * u_stride + Ng + Nx] = u_dev[j * u_stride + Ng];
                }
            }
            if (y_periodic) {
                #pragma omp target teams distribute parallel for is_device_ptr(v_dev)
                for (int i = Ng; i < Ng + Nx; ++i) {
                    v_dev[(Ng + Ny) * v_stride + i] = v_dev[Ng * v_stride + i];
                }
            }
        }
        else {
            // Fallback for any other combinations (shouldn't happen in normal usage)
            time_kernels::euler_advance_2d(
                u_in, u_out, conv_u_ptr_, diff_u_ptr_,
                v_in, v_out, conv_v_ptr_, diff_v_ptr_,
                Nx, Ny, Ng, u_stride, v_stride, dt, fx_, fy_, u_total, v_total);
            time_kernels::periodic_2d(u_out, v_out, x_periodic, y_periodic,
                                       Nx, Ny, Ng, u_stride, v_stride, u_total, v_total);
        }
    } else {
        // 3D INLINE euler_advance with explicit code paths for each case
        // Use MEMBER pointers directly (NVHPC workaround - function params don't work)
        const int Nz = mesh_->Nz;
        const int u_plane = u_stride * (Ny + 2 * Ng);
        const int v_plane = v_stride * (Ny + 2 * Ng + 1);
        const int w_stride_local = Nx + 2 * Ng;
        const int w_plane = w_stride_local * (Ny + 2 * Ng);
        const size_t w_total = vel_in.w_total_size();
        const bool z_periodic = (velocity_bc_.z_lo == VelocityBC::Periodic) &&
                                (velocity_bc_.z_hi == VelocityBC::Periodic);
        const double fx = fx_;
        const double fy = fy_;
        const double fz = fz_;

        // Case: velocity_ -> velocity_star_
        if (&vel_in == &velocity_ && &vel_out == &velocity_star_) {
            // NVHPC WORKAROUND: Use gpu::dev_ptr to get actual device addresses.
            // Member pointers in target regions get HOST addresses in NVHPC.
            const double* u_in_dev = gpu::dev_ptr(velocity_u_ptr_);
            double* u_out_dev = gpu::dev_ptr(velocity_star_u_ptr_);
            const double* conv_u_dev = gpu::dev_ptr(conv_u_ptr_);
            const double* diff_u_dev = gpu::dev_ptr(diff_u_ptr_);
            const double* v_in_dev = gpu::dev_ptr(velocity_v_ptr_);
            double* v_out_dev = gpu::dev_ptr(velocity_star_v_ptr_);
            const double* conv_v_dev = gpu::dev_ptr(conv_v_ptr_);
            const double* diff_v_dev = gpu::dev_ptr(diff_v_ptr_);
            const double* w_in_dev = gpu::dev_ptr(velocity_w_ptr_);
            double* w_out_dev = gpu::dev_ptr(velocity_star_w_ptr_);
            const double* conv_w_dev = gpu::dev_ptr(conv_w_ptr_);
            const double* diff_w_dev = gpu::dev_ptr(diff_w_ptr_);

            // Euler advance u
            #pragma omp target teams distribute parallel for collapse(3) \
                is_device_ptr(u_in_dev, u_out_dev, conv_u_dev, diff_u_dev)
            for (int k = Ng; k < Ng + Nz; ++k) {
                for (int j = Ng; j < Ng + Ny; ++j) {
                    for (int i = Ng; i <= Ng + Nx; ++i) {
                        int idx = k * u_plane + j * u_stride + i;
                        u_out_dev[idx] = u_in_dev[idx] + dt * (-conv_u_dev[idx] + diff_u_dev[idx] + fx);
                    }
                }
            }
            // Euler advance v
            #pragma omp target teams distribute parallel for collapse(3) \
                is_device_ptr(v_in_dev, v_out_dev, conv_v_dev, diff_v_dev)
            for (int k = Ng; k < Ng + Nz; ++k) {
                for (int j = Ng; j <= Ng + Ny; ++j) {
                    for (int i = Ng; i < Ng + Nx; ++i) {
                        int idx = k * v_plane + j * v_stride + i;
                        v_out_dev[idx] = v_in_dev[idx] + dt * (-conv_v_dev[idx] + diff_v_dev[idx] + fy);
                    }
                }
            }
            // Euler advance w
            #pragma omp target teams distribute parallel for collapse(3) \
                is_device_ptr(w_in_dev, w_out_dev, conv_w_dev, diff_w_dev)
            for (int k = Ng; k <= Ng + Nz; ++k) {
                for (int j = Ng; j < Ng + Ny; ++j) {
                    for (int i = Ng; i < Ng + Nx; ++i) {
                        int idx = k * w_plane + j * w_stride_local + i;
                        w_out_dev[idx] = w_in_dev[idx] + dt * (-conv_w_dev[idx] + diff_w_dev[idx] + fz);
                    }
                }
            }
            // Periodicity for velocity_star_
            if (x_periodic) {
                #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(u_out_dev)
                for (int k = Ng; k < Ng + Nz; ++k) {
                    for (int j = Ng; j < Ng + Ny; ++j) {
                        u_out_dev[k * u_plane + j * u_stride + Ng + Nx] =
                            u_out_dev[k * u_plane + j * u_stride + Ng];
                    }
                }
            }
            if (y_periodic) {
                #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(v_out_dev)
                for (int k = Ng; k < Ng + Nz; ++k) {
                    for (int i = Ng; i < Ng + Nx; ++i) {
                        v_out_dev[k * v_plane + (Ng + Ny) * v_stride + i] =
                            v_out_dev[k * v_plane + Ng * v_stride + i];
                    }
                }
            }
            if (z_periodic) {
                #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(w_out_dev)
                for (int j = Ng; j < Ng + Ny; ++j) {
                    for (int i = Ng; i < Ng + Nx; ++i) {
                        w_out_dev[(Ng + Nz) * w_plane + j * w_stride_local + i] =
                            w_out_dev[Ng * w_plane + j * w_stride_local + i];
                    }
                }
            }
        }
        // Case: velocity_star_ -> velocity_
        else if (&vel_in == &velocity_star_ && &vel_out == &velocity_) {
            // NVHPC WORKAROUND: Use gpu::dev_ptr to get actual device addresses.
            const double* u_in_dev = gpu::dev_ptr(velocity_star_u_ptr_);
            double* u_out_dev = gpu::dev_ptr(velocity_u_ptr_);
            const double* conv_u_dev = gpu::dev_ptr(conv_u_ptr_);
            const double* diff_u_dev = gpu::dev_ptr(diff_u_ptr_);
            const double* v_in_dev = gpu::dev_ptr(velocity_star_v_ptr_);
            double* v_out_dev = gpu::dev_ptr(velocity_v_ptr_);
            const double* conv_v_dev = gpu::dev_ptr(conv_v_ptr_);
            const double* diff_v_dev = gpu::dev_ptr(diff_v_ptr_);
            const double* w_in_dev = gpu::dev_ptr(velocity_star_w_ptr_);
            double* w_out_dev = gpu::dev_ptr(velocity_w_ptr_);
            const double* conv_w_dev = gpu::dev_ptr(conv_w_ptr_);
            const double* diff_w_dev = gpu::dev_ptr(diff_w_ptr_);

            // Euler advance u
            #pragma omp target teams distribute parallel for collapse(3) \
                is_device_ptr(u_in_dev, u_out_dev, conv_u_dev, diff_u_dev)
            for (int k = Ng; k < Ng + Nz; ++k) {
                for (int j = Ng; j < Ng + Ny; ++j) {
                    for (int i = Ng; i <= Ng + Nx; ++i) {
                        int idx = k * u_plane + j * u_stride + i;
                        u_out_dev[idx] = u_in_dev[idx] + dt * (-conv_u_dev[idx] + diff_u_dev[idx] + fx);
                    }
                }
            }
            // Euler advance v
            #pragma omp target teams distribute parallel for collapse(3) \
                is_device_ptr(v_in_dev, v_out_dev, conv_v_dev, diff_v_dev)
            for (int k = Ng; k < Ng + Nz; ++k) {
                for (int j = Ng; j <= Ng + Ny; ++j) {
                    for (int i = Ng; i < Ng + Nx; ++i) {
                        int idx = k * v_plane + j * v_stride + i;
                        v_out_dev[idx] = v_in_dev[idx] + dt * (-conv_v_dev[idx] + diff_v_dev[idx] + fy);
                    }
                }
            }
            // Euler advance w
            #pragma omp target teams distribute parallel for collapse(3) \
                is_device_ptr(w_in_dev, w_out_dev, conv_w_dev, diff_w_dev)
            for (int k = Ng; k <= Ng + Nz; ++k) {
                for (int j = Ng; j < Ng + Ny; ++j) {
                    for (int i = Ng; i < Ng + Nx; ++i) {
                        int idx = k * w_plane + j * w_stride_local + i;
                        w_out_dev[idx] = w_in_dev[idx] + dt * (-conv_w_dev[idx] + diff_w_dev[idx] + fz);
                    }
                }
            }
            // Periodicity for velocity_
            if (x_periodic) {
                #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(u_out_dev)
                for (int k = Ng; k < Ng + Nz; ++k) {
                    for (int j = Ng; j < Ng + Ny; ++j) {
                        u_out_dev[k * u_plane + j * u_stride + Ng + Nx] =
                            u_out_dev[k * u_plane + j * u_stride + Ng];
                    }
                }
            }
            if (y_periodic) {
                #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(v_out_dev)
                for (int k = Ng; k < Ng + Nz; ++k) {
                    for (int i = Ng; i < Ng + Nx; ++i) {
                        v_out_dev[k * v_plane + (Ng + Ny) * v_stride + i] =
                            v_out_dev[k * v_plane + Ng * v_stride + i];
                    }
                }
            }
            if (z_periodic) {
                #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(w_out_dev)
                for (int j = Ng; j < Ng + Ny; ++j) {
                    for (int i = Ng; i < Ng + Nx; ++i) {
                        w_out_dev[(Ng + Nz) * w_plane + j * w_stride_local + i] =
                            w_out_dev[Ng * w_plane + j * w_stride_local + i];
                    }
                }
            }
        }
        else {
            // Fallback for any other combinations (shouldn't happen in normal usage)
            time_kernels::euler_advance_3d(
                u_in, u_out, conv_u_ptr_, diff_u_ptr_,
                v_in, v_out, conv_v_ptr_, diff_v_ptr_,
                w_in, w_out, conv_w_ptr_, diff_w_ptr_,
                Nx, Ny, Nz, Ng, u_stride, v_stride, w_stride_local,
                u_plane, v_plane, w_plane, dt, fx_, fy_, fz_, u_total, v_total, w_total);

            time_kernels::periodic_3d(u_out, v_out, w_out, x_periodic, y_periodic, z_periodic,
                                       Nx, Ny, Nz, Ng, u_stride, v_stride, w_stride_local,
                                       u_plane, v_plane, w_plane, u_total, v_total, w_total);
        }
    }
}

void RANSSolver::project_velocity(VectorField& vel, double dt) {
    VelFieldId vel_id = VelFieldId::Velocity;
    if (&vel == &velocity_star_) vel_id = VelFieldId::VelocityStar;
    else if (&vel == &velocity_old_) vel_id = VelFieldId::VelocityOld;
    else if (&vel == &velocity_rk_) vel_id = VelFieldId::VelocityRk;

    const bool need_swap = (&vel != &velocity_);

    if (need_swap) {
        std::swap(velocity_, vel);
        if (vel_id == VelFieldId::VelocityStar) {
            std::swap(velocity_u_ptr_, velocity_star_u_ptr_);
            std::swap(velocity_v_ptr_, velocity_star_v_ptr_);
            if (!mesh_->is2D()) std::swap(velocity_w_ptr_, velocity_star_w_ptr_);
        } else if (vel_id == VelFieldId::VelocityOld) {
            std::swap(velocity_u_ptr_, velocity_old_u_ptr_);
            std::swap(velocity_v_ptr_, velocity_old_v_ptr_);
            if (!mesh_->is2D()) std::swap(velocity_w_ptr_, velocity_old_w_ptr_);
        } else if (vel_id == VelFieldId::VelocityRk) {
            std::swap(velocity_u_ptr_, velocity_rk_u_ptr_);
            std::swap(velocity_v_ptr_, velocity_rk_v_ptr_);
            if (!mesh_->is2D()) std::swap(velocity_w_ptr_, velocity_rk_w_ptr_);
        }
    }

    apply_velocity_bc();

    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Ng = mesh_->Nghost;
    const bool is_2d = mesh_->is2D();
    const int stride = Nx + 2 * Ng;
    const int plane_stride = stride * (Ny + 2 * Ng);
    const size_t field_total = field_total_size_;

    // INLINE divergence computation to use member pointers directly (NVHPC workaround)
    // compute_divergence(VelocityWhich::Current, div_velocity_);
    const double dx = mesh_->dx;
    const double dy = mesh_->dy;
    const int u_stride_div = velocity_.u_stride();
    const int v_stride_div = velocity_.v_stride();
    const size_t u_total_div = velocity_.u_total_size();
    const size_t v_total_div = velocity_.v_total_size();

    if (is_2d) {
        // Use dev_ptr + is_device_ptr pattern (NVHPC workaround)
        const double* u_dev = gpu::dev_ptr(velocity_u_ptr_);
        const double* v_dev = gpu::dev_ptr(velocity_v_ptr_);
        double* div_dev = gpu::dev_ptr(div_velocity_ptr_);

        // DEBUG: Check velocity data on GPU before divergence computation
        static int div_debug = 0;
        if (div_debug < 3) {
            div_debug++;
            double max_u_before = 0.0;
            int sample_idx_div = (Ng + Ny / 4) * u_stride_div + (Ng + Nx / 4);
            double u_sample_before = 0.0;
            #pragma omp target teams distribute parallel for reduction(max:max_u_before) \
                map(from:u_sample_before) is_device_ptr(u_dev)
            for (size_t i = 0; i < u_total_div; ++i) {
                double val = u_dev[i];
                if (val < 0) val = -val;
                if (val > max_u_before) max_u_before = val;
                if (static_cast<int>(i) == sample_idx_div) u_sample_before = u_dev[i];
            }
            std::cerr << "[project_velocity div DEBUG] velocity_u_ptr_=" << velocity_u_ptr_
                      << " max_u_before=" << max_u_before << " u_sample=" << u_sample_before << "\n";
        }

        const int n_cells = Nx * Ny;
        #pragma omp target teams distribute parallel for is_device_ptr(u_dev, v_dev, div_dev)
        for (int idx = 0; idx < n_cells; ++idx) {
            const int i = idx % Nx + Ng;
            const int j = idx / Nx + Ng;

            const int u_right = j * u_stride_div + (i + 1);
            const int u_left = j * u_stride_div + i;
            const int v_top = (j + 1) * v_stride_div + i;
            const int v_bottom = j * v_stride_div + i;
            const int div_idx = j * stride + i;

            const double dudx_val = (u_dev[u_right] - u_dev[u_left]) / dx;
            const double dvdy_val = (v_dev[v_top] - v_dev[v_bottom]) / dy;
            div_dev[div_idx] = dudx_val + dvdy_val;
        }
    } else {
        // For 3D, use the existing compute_divergence call for now
        compute_divergence(VelocityWhich::Current, div_velocity_);
    }

    // Compute mean divergence and build RHS
    // INLINE using member pointers directly (NVHPC workaround - function params don't work)
    double mean_div = 0.0;
    const double dt_inv = 1.0 / dt;
    const int count = is_2d ? (Nx * Ny) : (Nx * Ny * Nz);

    if (is_2d) {
        // Use dev_ptr + is_device_ptr pattern (NVHPC workaround)
        const double* div_dev = gpu::dev_ptr(div_velocity_ptr_);
        double* rhs_dev = gpu::dev_ptr(rhs_poisson_ptr_);

        // Compute mean divergence
        double sum = 0.0;
        #pragma omp target teams distribute parallel for collapse(2) reduction(+:sum) \
            is_device_ptr(div_dev)
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                int idx = (j + Ng) * stride + (i + Ng);
                sum += div_dev[idx];
            }
        }
        mean_div = (count > 0) ? sum / count : 0.0;

        // Build Poisson RHS
        #pragma omp target teams distribute parallel for collapse(2) \
            is_device_ptr(div_dev, rhs_dev)
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                int idx = (j + Ng) * stride + (i + Ng);
                rhs_dev[idx] = (div_dev[idx] - mean_div) * dt_inv;
            }
        }
    } else {
        // 3D INLINE: Compute mean divergence using gpu::dev_ptr for NVHPC workaround
        // Member pointers in target regions get HOST addresses in NVHPC.
        const double* div_dev = gpu::dev_ptr(div_velocity_ptr_);
        double* rhs_dev = gpu::dev_ptr(rhs_poisson_ptr_);

        double sum = 0.0;
        #pragma omp target teams distribute parallel for collapse(3) reduction(+:sum) is_device_ptr(div_dev)
        for (int k = 0; k < Nz; ++k) {
            for (int j = 0; j < Ny; ++j) {
                for (int i = 0; i < Nx; ++i) {
                    int idx = (k + Ng) * plane_stride + (j + Ng) * stride + (i + Ng);
                    sum += div_dev[idx];
                }
            }
        }
        mean_div = (count > 0) ? sum / count : 0.0;

        // 3D INLINE: Build Poisson RHS
        #pragma omp target teams distribute parallel for collapse(3) is_device_ptr(div_dev, rhs_dev)
        for (int k = 0; k < Nz; ++k) {
            for (int j = 0; j < Ny; ++j) {
                for (int i = 0; i < Nx; ++i) {
                    int idx = (k + Ng) * plane_stride + (j + Ng) * stride + (i + Ng);
                    rhs_dev[idx] = (div_dev[idx] - mean_div) * dt_inv;
                }
            }
        }
    }

    PoissonConfig pcfg;
    pcfg.max_vcycles = config_.poisson_max_vcycles;
    pcfg.tol_rhs = config_.poisson_tol_rhs;
    pcfg.fixed_cycles = config_.poisson_fixed_cycles;

    // DEBUG: Check RHS before Poisson solve
    static int proj_debug = 0;
    if (proj_debug < 5) {
        const double* rhs_dev = gpu::dev_ptr(rhs_poisson_ptr_);
        double max_rhs = 0.0;
        #pragma omp target teams distribute parallel for reduction(max:max_rhs) \
            is_device_ptr(rhs_dev)
        for (size_t idx = 0; idx < field_total_size_; ++idx) {
            double r = rhs_dev[idx];
            if (r < 0) r = -r;
            if (r > max_rhs) max_rhs = r;
        }
        std::cerr << "[project_velocity DEBUG] need_swap=" << need_swap
                  << " mean_div=" << mean_div << " max_rhs=" << max_rhs << "\n";
    }

    // Use solve_device() since our RHS was built on GPU using dev_ptr pattern
    // solve() reads from HOST memory which is stale; solve_device() works with device-resident data
    mg_poisson_solver_.solve_device(rhs_poisson_ptr_, pressure_corr_ptr_, pcfg);

    // DEBUG: Check pressure correction after solve
    if (proj_debug < 5) {
        proj_debug++;
        const double* pcorr_dev = gpu::dev_ptr(pressure_corr_ptr_);
        double max_pcorr = 0.0;
        #pragma omp target teams distribute parallel for reduction(max:max_pcorr) \
            is_device_ptr(pcorr_dev)
        for (size_t idx = 0; idx < field_total_size_; ++idx) {
            double p = pcorr_dev[idx];
            if (p < 0) p = -p;
            if (p > max_pcorr) max_pcorr = p;
        }
        std::cerr << "[project_velocity DEBUG] max_pcorr=" << max_pcorr << "\n";
    }

    // Copy velocity_ to velocity_star_ for correct_velocity()
    // Use dev_ptr + is_device_ptr pattern (NVHPC workaround)
    const int u_stride = Nx + 2 * Ng + 1;
    const int v_stride = Nx + 2 * Ng;

    if (is_2d) {
        // Get device pointers for copy
        const double* u_src_dev = gpu::dev_ptr(velocity_u_ptr_);
        double* u_dst_dev = gpu::dev_ptr(velocity_star_u_ptr_);
        const double* v_src_dev = gpu::dev_ptr(velocity_v_ptr_);
        double* v_dst_dev = gpu::dev_ptr(velocity_star_v_ptr_);

        // Copy u-velocity: velocity_star_u = velocity_u
        #pragma omp target teams distribute parallel for collapse(2) \
            is_device_ptr(u_src_dev, u_dst_dev)
        for (int j = Ng; j < Ng + Ny; ++j) {
            for (int i = Ng; i <= Ng + Nx; ++i) {
                int idx = j * u_stride + i;
                u_dst_dev[idx] = u_src_dev[idx];
            }
        }

        // Copy v-velocity: velocity_star_v = velocity_v
        #pragma omp target teams distribute parallel for collapse(2) \
            is_device_ptr(v_src_dev, v_dst_dev)
        for (int j = Ng; j <= Ng + Ny; ++j) {
            for (int i = Ng; i < Ng + Nx; ++i) {
                int idx = j * v_stride + i;
                v_dst_dev[idx] = v_src_dev[idx];
            }
        }
    } else {
        // 3D INLINE copy: velocity_star_ = velocity_
        // NVHPC WORKAROUND: Use gpu::dev_ptr to get actual device addresses.
        const int u_plane = u_stride * (Ny + 2 * Ng);
        const int v_plane = v_stride * (Ny + 2 * Ng + 1);
        const int w_stride_local = Nx + 2 * Ng;
        const int w_plane = w_stride_local * (Ny + 2 * Ng);

        const double* u_src_dev = gpu::dev_ptr(velocity_u_ptr_);
        double* u_dst_dev = gpu::dev_ptr(velocity_star_u_ptr_);
        const double* v_src_dev = gpu::dev_ptr(velocity_v_ptr_);
        double* v_dst_dev = gpu::dev_ptr(velocity_star_v_ptr_);
        const double* w_src_dev = gpu::dev_ptr(velocity_w_ptr_);
        double* w_dst_dev = gpu::dev_ptr(velocity_star_w_ptr_);

        // Copy u-velocity
        #pragma omp target teams distribute parallel for collapse(3) is_device_ptr(u_src_dev, u_dst_dev)
        for (int k = Ng; k < Ng + Nz; ++k) {
            for (int j = Ng; j < Ng + Ny; ++j) {
                for (int i = Ng; i <= Ng + Nx; ++i) {
                    int idx = k * u_plane + j * u_stride + i;
                    u_dst_dev[idx] = u_src_dev[idx];
                }
            }
        }

        // Copy v-velocity
        #pragma omp target teams distribute parallel for collapse(3) is_device_ptr(v_src_dev, v_dst_dev)
        for (int k = Ng; k < Ng + Nz; ++k) {
            for (int j = Ng; j <= Ng + Ny; ++j) {
                for (int i = Ng; i < Ng + Nx; ++i) {
                    int idx = k * v_plane + j * v_stride + i;
                    v_dst_dev[idx] = v_src_dev[idx];
                }
            }
        }

        // Copy w-velocity
        #pragma omp target teams distribute parallel for collapse(3) is_device_ptr(w_src_dev, w_dst_dev)
        for (int k = Ng; k <= Ng + Nz; ++k) {
            for (int j = Ng; j < Ng + Ny; ++j) {
                for (int i = Ng; i < Ng + Nx; ++i) {
                    int idx = k * w_plane + j * w_stride_local + i;
                    w_dst_dev[idx] = w_src_dev[idx];
                }
            }
        }
    }

    correct_velocity();
    apply_velocity_bc();
    // Note: correct_velocity() already updates pressure internally

    if (need_swap) {
        if (vel_id == VelFieldId::VelocityStar) {
            std::swap(velocity_u_ptr_, velocity_star_u_ptr_);
            std::swap(velocity_v_ptr_, velocity_star_v_ptr_);
            if (!mesh_->is2D()) std::swap(velocity_w_ptr_, velocity_star_w_ptr_);
        } else if (vel_id == VelFieldId::VelocityOld) {
            std::swap(velocity_u_ptr_, velocity_old_u_ptr_);
            std::swap(velocity_v_ptr_, velocity_old_v_ptr_);
            if (!mesh_->is2D()) std::swap(velocity_w_ptr_, velocity_old_w_ptr_);
        } else if (vel_id == VelFieldId::VelocityRk) {
            std::swap(velocity_u_ptr_, velocity_rk_u_ptr_);
            std::swap(velocity_v_ptr_, velocity_rk_v_ptr_);
            if (!mesh_->is2D()) std::swap(velocity_w_ptr_, velocity_rk_w_ptr_);
        }
        std::swap(velocity_, vel);
    }
}

void RANSSolver::ssprk2_step(double dt) {
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Ng = mesh_->Nghost;
    const int u_stride = Nx + 2 * Ng + 1;
    const int v_stride = Nx + 2 * Ng;
    const bool is_2d = mesh_->is2D();
    const size_t u_total = velocity_.u_total_size();
    const size_t v_total = velocity_.v_total_size();

    // DEBUG: Check values BEFORE copy
    static int copy_debug = 0;
    if (copy_debug < 3) {
        const double* u_src_dbg = gpu::dev_ptr(velocity_u_ptr_);
        const double* u_dst_dbg = gpu::dev_ptr(velocity_rk_u_ptr_);
        double u_src_sample = 0.0, u_dst_sample_before = 0.0;
        int ii = Ng + Nx / 4;
        int jj = Ng + Ny / 4;
        int sample_idx = jj * u_stride + ii;
        #pragma omp target map(from: u_src_sample, u_dst_sample_before) \
            is_device_ptr(u_src_dbg, u_dst_dbg)
        {
            u_src_sample = u_src_dbg[sample_idx];
            u_dst_sample_before = u_dst_dbg[sample_idx];
        }
        std::cerr << "[ssprk2 copy DEBUG] BEFORE copy: u_src[" << sample_idx << "]=" << u_src_sample
                  << " u_dst(rk)[" << sample_idx << "]=" << u_dst_sample_before << "\n";
    }

    // Store u^n in velocity_rk_ - use dev_ptr + is_device_ptr pattern (NVHPC workaround)
    // Member pointers inside target regions get HOST addresses with NVHPC.
    if (is_2d) {
        // Get device pointers for copy
        const double* u_src_dev = gpu::dev_ptr(velocity_u_ptr_);
        double* u_dst_dev = gpu::dev_ptr(velocity_rk_u_ptr_);
        const double* v_src_dev = gpu::dev_ptr(velocity_v_ptr_);
        double* v_dst_dev = gpu::dev_ptr(velocity_rk_v_ptr_);

        // Copy u-velocity: velocity_rk_u = velocity_u
        #pragma omp target teams distribute parallel for collapse(2) \
            is_device_ptr(u_src_dev, u_dst_dev)
        for (int j = Ng; j < Ng + Ny; ++j) {
            for (int i = Ng; i <= Ng + Nx; ++i) {
                int idx = j * u_stride + i;
                u_dst_dev[idx] = u_src_dev[idx];
            }
        }

        // Copy v-velocity: velocity_rk_v = velocity_v
        #pragma omp target teams distribute parallel for collapse(2) \
            is_device_ptr(v_src_dev, v_dst_dev)
        for (int j = Ng; j <= Ng + Ny; ++j) {
            for (int i = Ng; i < Ng + Nx; ++i) {
                int idx = j * v_stride + i;
                v_dst_dev[idx] = v_src_dev[idx];
            }
        }

        // DEBUG: Check values AFTER copy
        if (copy_debug < 3) {
            copy_debug++;
            double u_dst_sample_after = 0.0;
            int ii = Ng + Nx / 4;
            int jj = Ng + Ny / 4;
            int sample_idx = jj * u_stride + ii;
            #pragma omp target map(from: u_dst_sample_after) is_device_ptr(u_dst_dev)
            {
                u_dst_sample_after = u_dst_dev[sample_idx];
            }
            std::cerr << "[ssprk2 copy DEBUG] AFTER copy: u_dst(rk)[" << sample_idx << "]=" << u_dst_sample_after << "\n";
        }
    } else {
        // 3D INLINE copy: velocity_rk_ = velocity_
        // Use dev_ptr + is_device_ptr pattern (NVHPC workaround)
        const int Nz = mesh_->Nz;
        const int u_plane = u_stride * (Ny + 2 * Ng);
        const int v_plane = v_stride * (Ny + 2 * Ng + 1);
        const int w_stride_local = Nx + 2 * Ng;
        const int w_plane = w_stride_local * (Ny + 2 * Ng);

        // Get device pointers for copy
        const double* u_src_dev = gpu::dev_ptr(velocity_u_ptr_);
        double* u_dst_dev = gpu::dev_ptr(velocity_rk_u_ptr_);
        const double* v_src_dev = gpu::dev_ptr(velocity_v_ptr_);
        double* v_dst_dev = gpu::dev_ptr(velocity_rk_v_ptr_);
        const double* w_src_dev = gpu::dev_ptr(velocity_w_ptr_);
        double* w_dst_dev = gpu::dev_ptr(velocity_rk_w_ptr_);

        // Copy u-velocity
        #pragma omp target teams distribute parallel for collapse(3) \
            is_device_ptr(u_src_dev, u_dst_dev)
        for (int k = Ng; k < Ng + Nz; ++k) {
            for (int j = Ng; j < Ng + Ny; ++j) {
                for (int i = Ng; i <= Ng + Nx; ++i) {
                    int idx = k * u_plane + j * u_stride + i;
                    u_dst_dev[idx] = u_src_dev[idx];
                }
            }
        }

        // Copy v-velocity
        #pragma omp target teams distribute parallel for collapse(3) \
            is_device_ptr(v_src_dev, v_dst_dev)
        for (int k = Ng; k < Ng + Nz; ++k) {
            for (int j = Ng; j <= Ng + Ny; ++j) {
                for (int i = Ng; i < Ng + Nx; ++i) {
                    int idx = k * v_plane + j * v_stride + i;
                    v_dst_dev[idx] = v_src_dev[idx];
                }
            }
        }

        // Copy w-velocity
        #pragma omp target teams distribute parallel for collapse(3) \
            is_device_ptr(w_src_dev, w_dst_dev)
        for (int k = Ng; k <= Ng + Nz; ++k) {
            for (int j = Ng; j < Ng + Ny; ++j) {
                for (int i = Ng; i < Ng + Nx; ++i) {
                    int idx = k * w_plane + j * w_stride_local + i;
                    w_dst_dev[idx] = w_src_dev[idx];
                }
            }
        }
    }

    // Stage 1: u^(1) = u^n + dt * L(u^n)
    euler_substep(velocity_, velocity_star_, dt);
    project_velocity(velocity_star_, dt);

    // Stage 2: temp = u^(1) + dt * L(u^(1))
    euler_substep(velocity_star_, velocity_, dt);

    // Blend: u^{n+1} = 0.5 * u^n + 0.5 * temp
    // DEBUG: Check values before blend - sample from middle of domain
    static int blend_debug = 0;
    if (blend_debug < 3) {
        const double* u_rk_dbg = gpu::dev_ptr(velocity_rk_u_ptr_);
        const double* u_dbg = gpu::dev_ptr(velocity_u_ptr_);
        double u_rk_sample = 0.0, u_sample = 0.0;
        // Sample from cell (Nx/4, Ny/4) relative to interior
        int ii = Ng + Nx / 4;
        int jj = Ng + Ny / 4;
        int sample_idx = jj * u_stride + ii;
        std::cerr << "[ssprk2 blend DEBUG] Nx=" << Nx << " Ny=" << Ny << " Ng=" << Ng
                  << " u_stride=" << u_stride << " sample_idx=" << sample_idx << "\n";
        #pragma omp target map(from: u_rk_sample, u_sample) \
            is_device_ptr(u_rk_dbg, u_dbg)
        {
            u_rk_sample = u_rk_dbg[sample_idx];
            u_sample = u_dbg[sample_idx];
        }
        std::cerr << "[ssprk2 blend DEBUG] BEFORE: u_rk[" << sample_idx << "]=" << u_rk_sample
                  << " u[" << sample_idx << "]=" << u_sample << "\n";
    }

    if (is_2d) {
        // INLINE blend: velocity_ = 0.5 * velocity_rk_ + 0.5 * velocity_
        // Use dev_ptr + is_device_ptr pattern (NVHPC workaround)
        const double* u_rk_dev = gpu::dev_ptr(velocity_rk_u_ptr_);
        double* u_dev = gpu::dev_ptr(velocity_u_ptr_);
        const double* v_rk_dev = gpu::dev_ptr(velocity_rk_v_ptr_);
        double* v_dev = gpu::dev_ptr(velocity_v_ptr_);

        // Blend u-velocity
        #pragma omp target teams distribute parallel for collapse(2) \
            is_device_ptr(u_rk_dev, u_dev)
        for (int j = Ng; j < Ng + Ny; ++j) {
            for (int i = Ng; i <= Ng + Nx; ++i) {
                int idx = j * u_stride + i;
                u_dev[idx] = 0.5 * u_rk_dev[idx] + 0.5 * u_dev[idx];
            }
        }

        // Blend v-velocity
        #pragma omp target teams distribute parallel for collapse(2) \
            is_device_ptr(v_rk_dev, v_dev)
        for (int j = Ng; j <= Ng + Ny; ++j) {
            for (int i = Ng; i < Ng + Nx; ++i) {
                int idx = j * v_stride + i;
                v_dev[idx] = 0.5 * v_rk_dev[idx] + 0.5 * v_dev[idx];
            }
        }

        // DEBUG: Check values after blend
        if (blend_debug < 3) {
            blend_debug++;
            double u_sample_after = 0.0;
            int ii = Ng + Nx / 4;
            int jj = Ng + Ny / 4;
            int sample_idx = jj * u_stride + ii;
            #pragma omp target map(from: u_sample_after) is_device_ptr(u_dev)
            {
                u_sample_after = u_dev[sample_idx];
            }
            std::cerr << "[ssprk2 blend DEBUG] AFTER: u[" << sample_idx << "]=" << u_sample_after << "\n";
        }
    } else {
        // 3D INLINE blend: velocity_ = 0.5 * velocity_rk_ + 0.5 * velocity_
        // Use dev_ptr + is_device_ptr pattern (NVHPC workaround)
        const int Nz = mesh_->Nz;
        const int u_plane = u_stride * (Ny + 2 * Ng);
        const int v_plane = v_stride * (Ny + 2 * Ng + 1);
        const int w_stride_local = Nx + 2 * Ng;
        const int w_plane = w_stride_local * (Ny + 2 * Ng);

        const double* u_rk_dev = gpu::dev_ptr(velocity_rk_u_ptr_);
        double* u_dev = gpu::dev_ptr(velocity_u_ptr_);
        const double* v_rk_dev = gpu::dev_ptr(velocity_rk_v_ptr_);
        double* v_dev = gpu::dev_ptr(velocity_v_ptr_);
        const double* w_rk_dev = gpu::dev_ptr(velocity_rk_w_ptr_);
        double* w_dev = gpu::dev_ptr(velocity_w_ptr_);

        // Blend u-velocity
        #pragma omp target teams distribute parallel for collapse(3) \
            is_device_ptr(u_rk_dev, u_dev)
        for (int k = Ng; k < Ng + Nz; ++k) {
            for (int j = Ng; j < Ng + Ny; ++j) {
                for (int i = Ng; i <= Ng + Nx; ++i) {
                    int idx = k * u_plane + j * u_stride + i;
                    u_dev[idx] = 0.5 * u_rk_dev[idx] + 0.5 * u_dev[idx];
                }
            }
        }

        // Blend v-velocity
        #pragma omp target teams distribute parallel for collapse(3) \
            is_device_ptr(v_rk_dev, v_dev)
        for (int k = Ng; k < Ng + Nz; ++k) {
            for (int j = Ng; j <= Ng + Ny; ++j) {
                for (int i = Ng; i < Ng + Nx; ++i) {
                    int idx = k * v_plane + j * v_stride + i;
                    v_dev[idx] = 0.5 * v_rk_dev[idx] + 0.5 * v_dev[idx];
                }
            }
        }

        // Blend w-velocity
        #pragma omp target teams distribute parallel for collapse(3) \
            is_device_ptr(w_rk_dev, w_dev)
        for (int k = Ng; k <= Ng + Nz; ++k) {
            for (int j = Ng; j < Ng + Ny; ++j) {
                for (int i = Ng; i < Ng + Nx; ++i) {
                    int idx = k * w_plane + j * w_stride_local + i;
                    w_dev[idx] = 0.5 * w_rk_dev[idx] + 0.5 * w_dev[idx];
                }
            }
        }
    }

    project_velocity(velocity_, dt);

#ifndef NDEBUG
    // Verify pointer consistency at end of RK2 step
    if (velocity_u_ptr_ != velocity_.u_data().data()) {
        std::cerr << "[ssprk2_step] ERROR: velocity_u_ptr_ mismatch!\n"
                  << "  velocity_u_ptr_ = " << velocity_u_ptr_
                  << "  velocity_.u_data().data() = " << velocity_.u_data().data() << "\n";
    }
#endif
}

void RANSSolver::ssprk3_step(double dt) {
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Ng = mesh_->Nghost;
    const int u_stride = Nx + 2 * Ng + 1;
    const int v_stride = Nx + 2 * Ng;
    const bool is_2d = mesh_->is2D();
    const size_t u_total = velocity_.u_total_size();
    const size_t v_total = velocity_.v_total_size();

    // Store u^n in velocity_rk_ - use dev_ptr + is_device_ptr pattern (NVHPC workaround)
    // Member pointers inside target regions get HOST addresses with NVHPC.
    if (is_2d) {
        // Get device pointers for copy
        const double* u_src_dev = gpu::dev_ptr(velocity_u_ptr_);
        double* u_dst_dev = gpu::dev_ptr(velocity_rk_u_ptr_);
        const double* v_src_dev = gpu::dev_ptr(velocity_v_ptr_);
        double* v_dst_dev = gpu::dev_ptr(velocity_rk_v_ptr_);

        // Copy u-velocity: velocity_rk_u = velocity_u
        #pragma omp target teams distribute parallel for collapse(2) \
            is_device_ptr(u_src_dev, u_dst_dev)
        for (int j = Ng; j < Ng + Ny; ++j) {
            for (int i = Ng; i <= Ng + Nx; ++i) {
                int idx = j * u_stride + i;
                u_dst_dev[idx] = u_src_dev[idx];
            }
        }

        // Copy v-velocity: velocity_rk_v = velocity_v
        #pragma omp target teams distribute parallel for collapse(2) \
            is_device_ptr(v_src_dev, v_dst_dev)
        for (int j = Ng; j <= Ng + Ny; ++j) {
            for (int i = Ng; i < Ng + Nx; ++i) {
                int idx = j * v_stride + i;
                v_dst_dev[idx] = v_src_dev[idx];
            }
        }
    } else {
        // 3D INLINE copy: velocity_rk_ = velocity_
        // Use dev_ptr + is_device_ptr pattern (NVHPC workaround)
        const int Nz = mesh_->Nz;
        const int u_plane = u_stride * (Ny + 2 * Ng);
        const int v_plane = v_stride * (Ny + 2 * Ng + 1);
        const int w_stride_local = Nx + 2 * Ng;
        const int w_plane = w_stride_local * (Ny + 2 * Ng);

        // Get device pointers for copy
        const double* u_src_dev = gpu::dev_ptr(velocity_u_ptr_);
        double* u_dst_dev = gpu::dev_ptr(velocity_rk_u_ptr_);
        const double* v_src_dev = gpu::dev_ptr(velocity_v_ptr_);
        double* v_dst_dev = gpu::dev_ptr(velocity_rk_v_ptr_);
        const double* w_src_dev = gpu::dev_ptr(velocity_w_ptr_);
        double* w_dst_dev = gpu::dev_ptr(velocity_rk_w_ptr_);

        // Copy u-velocity
        #pragma omp target teams distribute parallel for collapse(3) \
            is_device_ptr(u_src_dev, u_dst_dev)
        for (int k = Ng; k < Ng + Nz; ++k) {
            for (int j = Ng; j < Ng + Ny; ++j) {
                for (int i = Ng; i <= Ng + Nx; ++i) {
                    int idx = k * u_plane + j * u_stride + i;
                    u_dst_dev[idx] = u_src_dev[idx];
                }
            }
        }

        // Copy v-velocity
        #pragma omp target teams distribute parallel for collapse(3) \
            is_device_ptr(v_src_dev, v_dst_dev)
        for (int k = Ng; k < Ng + Nz; ++k) {
            for (int j = Ng; j <= Ng + Ny; ++j) {
                for (int i = Ng; i < Ng + Nx; ++i) {
                    int idx = k * v_plane + j * v_stride + i;
                    v_dst_dev[idx] = v_src_dev[idx];
                }
            }
        }

        // Copy w-velocity
        #pragma omp target teams distribute parallel for collapse(3) \
            is_device_ptr(w_src_dev, w_dst_dev)
        for (int k = Ng; k <= Ng + Nz; ++k) {
            for (int j = Ng; j < Ng + Ny; ++j) {
                for (int i = Ng; i < Ng + Nx; ++i) {
                    int idx = k * w_plane + j * w_stride_local + i;
                    w_dst_dev[idx] = w_src_dev[idx];
                }
            }
        }
    }

    // Stage 1: u^(1) = u^n + dt * L(u^n)
    euler_substep(velocity_, velocity_star_, dt);
    project_velocity(velocity_star_, dt);

    // Stage 2: u^(2) = 0.75 * u^n + 0.25 * (u^(1) + dt * L(u^(1)))
    euler_substep(velocity_star_, velocity_, dt);

    if (is_2d) {
        // INLINE blend_to: velocity_star_ = 0.75 * velocity_rk_ + 0.25 * velocity_
        // Use dev_ptr + is_device_ptr pattern (NVHPC workaround)
        const double* u_rk_dev = gpu::dev_ptr(velocity_rk_u_ptr_);
        const double* u_dev = gpu::dev_ptr(velocity_u_ptr_);
        double* u_star_dev = gpu::dev_ptr(velocity_star_u_ptr_);
        const double* v_rk_dev = gpu::dev_ptr(velocity_rk_v_ptr_);
        const double* v_dev = gpu::dev_ptr(velocity_v_ptr_);
        double* v_star_dev = gpu::dev_ptr(velocity_star_v_ptr_);

        // Blend u-velocity to velocity_star_
        #pragma omp target teams distribute parallel for collapse(2) \
            is_device_ptr(u_rk_dev, u_dev, u_star_dev)
        for (int j = Ng; j < Ng + Ny; ++j) {
            for (int i = Ng; i <= Ng + Nx; ++i) {
                int idx = j * u_stride + i;
                u_star_dev[idx] = 0.75 * u_rk_dev[idx] + 0.25 * u_dev[idx];
            }
        }

        // Blend v-velocity to velocity_star_
        #pragma omp target teams distribute parallel for collapse(2) \
            is_device_ptr(v_rk_dev, v_dev, v_star_dev)
        for (int j = Ng; j <= Ng + Ny; ++j) {
            for (int i = Ng; i < Ng + Nx; ++i) {
                int idx = j * v_stride + i;
                v_star_dev[idx] = 0.75 * v_rk_dev[idx] + 0.25 * v_dev[idx];
            }
        }
    } else {
        // 3D INLINE blend_to: velocity_star_ = 0.75 * velocity_rk_ + 0.25 * velocity_
        // Use dev_ptr + is_device_ptr pattern (NVHPC workaround)
        const int Nz = mesh_->Nz;
        const int u_plane = u_stride * (Ny + 2 * Ng);
        const int v_plane = v_stride * (Ny + 2 * Ng + 1);
        const int w_stride_local = Nx + 2 * Ng;
        const int w_plane = w_stride_local * (Ny + 2 * Ng);

        const double* u_rk_dev = gpu::dev_ptr(velocity_rk_u_ptr_);
        const double* u_dev = gpu::dev_ptr(velocity_u_ptr_);
        double* u_star_dev = gpu::dev_ptr(velocity_star_u_ptr_);
        const double* v_rk_dev = gpu::dev_ptr(velocity_rk_v_ptr_);
        const double* v_dev = gpu::dev_ptr(velocity_v_ptr_);
        double* v_star_dev = gpu::dev_ptr(velocity_star_v_ptr_);
        const double* w_rk_dev = gpu::dev_ptr(velocity_rk_w_ptr_);
        const double* w_dev = gpu::dev_ptr(velocity_w_ptr_);
        double* w_star_dev = gpu::dev_ptr(velocity_star_w_ptr_);

        // Blend u-velocity to velocity_star_
        #pragma omp target teams distribute parallel for collapse(3) \
            is_device_ptr(u_rk_dev, u_dev, u_star_dev)
        for (int k = Ng; k < Ng + Nz; ++k) {
            for (int j = Ng; j < Ng + Ny; ++j) {
                for (int i = Ng; i <= Ng + Nx; ++i) {
                    int idx = k * u_plane + j * u_stride + i;
                    u_star_dev[idx] = 0.75 * u_rk_dev[idx] + 0.25 * u_dev[idx];
                }
            }
        }

        // Blend v-velocity to velocity_star_
        #pragma omp target teams distribute parallel for collapse(3) \
            is_device_ptr(v_rk_dev, v_dev, v_star_dev)
        for (int k = Ng; k < Ng + Nz; ++k) {
            for (int j = Ng; j <= Ng + Ny; ++j) {
                for (int i = Ng; i < Ng + Nx; ++i) {
                    int idx = k * v_plane + j * v_stride + i;
                    v_star_dev[idx] = 0.75 * v_rk_dev[idx] + 0.25 * v_dev[idx];
                }
            }
        }

        // Blend w-velocity to velocity_star_
        #pragma omp target teams distribute parallel for collapse(3) \
            is_device_ptr(w_rk_dev, w_dev, w_star_dev)
        for (int k = Ng; k <= Ng + Nz; ++k) {
            for (int j = Ng; j < Ng + Ny; ++j) {
                for (int i = Ng; i < Ng + Nx; ++i) {
                    int idx = k * w_plane + j * w_stride_local + i;
                    w_star_dev[idx] = 0.75 * w_rk_dev[idx] + 0.25 * w_dev[idx];
                }
            }
        }
    }

    project_velocity(velocity_star_, dt);

    // Stage 3: u^{n+1} = (1/3) * u^n + (2/3) * (u^(2) + dt * L(u^(2)))
    euler_substep(velocity_star_, velocity_, dt);

    if (is_2d) {
        // INLINE blend: velocity_ = (1/3) * velocity_rk_ + (2/3) * velocity_
        // Use dev_ptr + is_device_ptr pattern (NVHPC workaround)
        const double* u_rk_dev = gpu::dev_ptr(velocity_rk_u_ptr_);
        double* u_dev = gpu::dev_ptr(velocity_u_ptr_);
        const double* v_rk_dev = gpu::dev_ptr(velocity_rk_v_ptr_);
        double* v_dev = gpu::dev_ptr(velocity_v_ptr_);

        // DEBUG: Check values BEFORE final blend
        static int blend_debug = 0;
        if (blend_debug < 3) {
            double max_rk = 0.0, max_vel = 0.0;
            const size_t nu_dbg = u_total;

            #pragma omp target teams distribute parallel for reduction(max:max_rk,max_vel) \
                is_device_ptr(u_rk_dev, u_dev)
            for (size_t i = 0; i < nu_dbg; ++i) {
                double r = std::abs(u_rk_dev[i]);
                double v = std::abs(u_dev[i]);
                if (r > max_rk) max_rk = r;
                if (v > max_vel) max_vel = v;
            }

            std::cerr << "[RK3 BLEND DEBUG] BEFORE: max_rk=" << max_rk << " max_vel=" << max_vel
                      << " velocity_rk_u_ptr_=" << velocity_rk_u_ptr_
                      << " velocity_u_ptr_=" << velocity_u_ptr_ << "\n";
        }

        // Blend u-velocity
        #pragma omp target teams distribute parallel for collapse(2) \
            is_device_ptr(u_rk_dev, u_dev)
        for (int j = Ng; j < Ng + Ny; ++j) {
            for (int i = Ng; i <= Ng + Nx; ++i) {
                int idx = j * u_stride + i;
                u_dev[idx] = (1.0/3.0) * u_rk_dev[idx] + (2.0/3.0) * u_dev[idx];
            }
        }

        // Blend v-velocity
        #pragma omp target teams distribute parallel for collapse(2) \
            is_device_ptr(v_rk_dev, v_dev)
        for (int j = Ng; j <= Ng + Ny; ++j) {
            for (int i = Ng; i < Ng + Nx; ++i) {
                int idx = j * v_stride + i;
                v_dev[idx] = (1.0/3.0) * v_rk_dev[idx] + (2.0/3.0) * v_dev[idx];
            }
        }

        // DEBUG: Check values AFTER final blend
        if (blend_debug < 3) {
            blend_debug++;
            double max_vel_after = 0.0;
            const size_t nu_dbg = u_total;

            #pragma omp target teams distribute parallel for reduction(max:max_vel_after) \
                is_device_ptr(u_dev)
            for (size_t i = 0; i < nu_dbg; ++i) {
                double v = std::abs(u_dev[i]);
                if (v > max_vel_after) max_vel_after = v;
            }

            std::cerr << "[RK3 BLEND DEBUG] AFTER: max_vel=" << max_vel_after << "\n";
        }
    } else {
        // 3D INLINE blend: velocity_ = (1/3) * velocity_rk_ + (2/3) * velocity_
        // Use dev_ptr + is_device_ptr pattern (NVHPC workaround)
        const int Nz = mesh_->Nz;
        const int u_plane = u_stride * (Ny + 2 * Ng);
        const int v_plane = v_stride * (Ny + 2 * Ng + 1);
        const int w_stride_local = Nx + 2 * Ng;
        const int w_plane = w_stride_local * (Ny + 2 * Ng);

        const double* u_rk_dev = gpu::dev_ptr(velocity_rk_u_ptr_);
        double* u_dev = gpu::dev_ptr(velocity_u_ptr_);
        const double* v_rk_dev = gpu::dev_ptr(velocity_rk_v_ptr_);
        double* v_dev = gpu::dev_ptr(velocity_v_ptr_);
        const double* w_rk_dev = gpu::dev_ptr(velocity_rk_w_ptr_);
        double* w_dev = gpu::dev_ptr(velocity_w_ptr_);

        // Blend u-velocity
        #pragma omp target teams distribute parallel for collapse(3) \
            is_device_ptr(u_rk_dev, u_dev)
        for (int k = Ng; k < Ng + Nz; ++k) {
            for (int j = Ng; j < Ng + Ny; ++j) {
                for (int i = Ng; i <= Ng + Nx; ++i) {
                    int idx = k * u_plane + j * u_stride + i;
                    u_dev[idx] = (1.0/3.0) * u_rk_dev[idx] + (2.0/3.0) * u_dev[idx];
                }
            }
        }

        // Blend v-velocity
        #pragma omp target teams distribute parallel for collapse(3) \
            is_device_ptr(v_rk_dev, v_dev)
        for (int k = Ng; k < Ng + Nz; ++k) {
            for (int j = Ng; j <= Ng + Ny; ++j) {
                for (int i = Ng; i < Ng + Nx; ++i) {
                    int idx = k * v_plane + j * v_stride + i;
                    v_dev[idx] = (1.0/3.0) * v_rk_dev[idx] + (2.0/3.0) * v_dev[idx];
                }
            }
        }

        // Blend w-velocity
        #pragma omp target teams distribute parallel for collapse(3) \
            is_device_ptr(w_rk_dev, w_dev)
        for (int k = Ng; k <= Ng + Nz; ++k) {
            for (int j = Ng; j < Ng + Ny; ++j) {
                for (int i = Ng; i < Ng + Nx; ++i) {
                    int idx = k * w_plane + j * w_stride_local + i;
                    w_dev[idx] = (1.0/3.0) * w_rk_dev[idx] + (2.0/3.0) * w_dev[idx];
                }
            }
        }
    }

    project_velocity(velocity_, dt);
    apply_velocity_bc();

    // DEBUG: Verify pointer consistency at end of RK3 step
    if (velocity_u_ptr_ != velocity_.u_data().data()) {
        std::cerr << "[ssprk3_step] ERROR: velocity_u_ptr_ mismatch!\n"
                  << "  velocity_u_ptr_ = " << velocity_u_ptr_
                  << "  velocity_.u_data().data() = " << velocity_.u_data().data() << "\n";
    }

    // DEBUG: Verify GPU data at end of RK3 step
    static int rk3_end_debug = 0;
    if (rk3_end_debug < 3) {
        rk3_end_debug++;
        const double* u_final_dev = gpu::dev_ptr(velocity_u_ptr_);
        double max_u_final = 0.0;
        const size_t nu = u_total;

        #pragma omp target teams distribute parallel for reduction(max:max_u_final) \
            is_device_ptr(u_final_dev)
        for (size_t i = 0; i < nu; ++i) {
            double val = std::abs(u_final_dev[i]);
            if (val > max_u_final) max_u_final = val;
        }

        std::cerr << "[ssprk3_step END] velocity_u_ptr_=" << velocity_u_ptr_
                  << " max_u_final=" << max_u_final << "\n";
    }
}

} // namespace nncfd
