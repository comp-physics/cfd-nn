/// @file solver_time.cpp
/// @brief Time integration methods for RANSSolver
///
/// Uses kernel helper functions from solver_time_kernels_*.cpp to avoid
/// exceeding nvc++ compiler's pragma limits.

#include "solver.hpp"
#include "solver_time_kernels.hpp"
#include "gpu_utils.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstdlib>

namespace nncfd {

// ============================================================================
// Diagnostic: NaN sentinel check with location reporting
// ============================================================================

namespace {

/// Check for NaN in velocity field and return location of first NaN found
/// Returns true if NaN found, sets (nan_i, nan_j, nan_k) to location
bool check_nan_with_location(const double* u_ptr, const double* v_ptr, const double* w_ptr,
                              int Nx, int Ny, int Nz, int Ng,
                              int u_stride, int v_stride, int w_stride,
                              int u_plane, int v_plane, int w_plane,
                              bool is_2d,
                              int& nan_i, int& nan_j, int& nan_k, char& nan_component) {
    nan_i = nan_j = nan_k = -1;
    nan_component = '?';

#ifdef USE_GPU_OFFLOAD
    // GPU reduction to find first NaN location
    const double* u_dev = gpu::dev_ptr(const_cast<double*>(u_ptr));
    const double* v_dev = gpu::dev_ptr(const_cast<double*>(v_ptr));
    const double* w_dev = w_ptr ? gpu::dev_ptr(const_cast<double*>(w_ptr)) : nullptr;

    // First pass: check if any NaN exists
    int has_nan_u = 0, has_nan_v = 0, has_nan_w = 0;

    if (is_2d) {
        #pragma omp target teams distribute parallel for collapse(2) reduction(|:has_nan_u) \
            is_device_ptr(u_dev)
        for (int j = Ng; j < Ny + Ng; ++j) {
            for (int i = Ng; i <= Nx + Ng; ++i) {
                int idx = j * u_stride + i;
                double val = u_dev[idx];
                if (val != val) has_nan_u = 1;  // NaN check
            }
        }
        #pragma omp target teams distribute parallel for collapse(2) reduction(|:has_nan_v) \
            is_device_ptr(v_dev)
        for (int j = Ng; j <= Ny + Ng; ++j) {
            for (int i = Ng; i < Nx + Ng; ++i) {
                int idx = j * v_stride + i;
                double val = v_dev[idx];
                if (val != val) has_nan_v = 1;
            }
        }
    } else {
        #pragma omp target teams distribute parallel for collapse(3) reduction(|:has_nan_u) \
            is_device_ptr(u_dev)
        for (int k = Ng; k < Nz + Ng; ++k) {
            for (int j = Ng; j < Ny + Ng; ++j) {
                for (int i = Ng; i <= Nx + Ng; ++i) {
                    int idx = k * u_plane + j * u_stride + i;
                    double val = u_dev[idx];
                    if (val != val) has_nan_u = 1;
                }
            }
        }
        #pragma omp target teams distribute parallel for collapse(3) reduction(|:has_nan_v) \
            is_device_ptr(v_dev)
        for (int k = Ng; k < Nz + Ng; ++k) {
            for (int j = Ng; j <= Ny + Ng; ++j) {
                for (int i = Ng; i < Nx + Ng; ++i) {
                    int idx = k * v_plane + j * v_stride + i;
                    double val = v_dev[idx];
                    if (val != val) has_nan_v = 1;
                }
            }
        }
        if (w_dev) {
            #pragma omp target teams distribute parallel for collapse(3) reduction(|:has_nan_w) \
                is_device_ptr(w_dev)
            for (int k = Ng; k <= Nz + Ng; ++k) {
                for (int j = Ng; j < Ny + Ng; ++j) {
                    for (int i = Ng; i < Nx + Ng; ++i) {
                        int idx = k * w_plane + j * w_stride + i;
                        double val = w_dev[idx];
                        if (val != val) has_nan_w = 1;
                    }
                }
            }
        }
    }

    if (has_nan_u == 0 && has_nan_v == 0 && has_nan_w == 0) {
        return false;  // No NaN
    }

    // Second pass (CPU): find first NaN location by copying a small amount of data
    // For production, just report which component has NaN
    if (has_nan_u) {
        nan_component = 'u';
        nan_i = nan_j = nan_k = 0;  // Location not pinpointed on GPU
    } else if (has_nan_v) {
        nan_component = 'v';
        nan_i = nan_j = nan_k = 0;
    } else if (has_nan_w) {
        nan_component = 'w';
        nan_i = nan_j = nan_k = 0;
    }
    return true;
#else
    // CPU path: full scan with location
    if (is_2d) {
        for (int j = Ng; j < Ny + Ng; ++j) {
            for (int i = Ng; i <= Nx + Ng; ++i) {
                if (std::isnan(u_ptr[j * u_stride + i])) {
                    nan_i = i - Ng; nan_j = j - Ng; nan_k = 0;
                    nan_component = 'u';
                    return true;
                }
            }
        }
        for (int j = Ng; j <= Ny + Ng; ++j) {
            for (int i = Ng; i < Nx + Ng; ++i) {
                if (std::isnan(v_ptr[j * v_stride + i])) {
                    nan_i = i - Ng; nan_j = j - Ng; nan_k = 0;
                    nan_component = 'v';
                    return true;
                }
            }
        }
    } else {
        for (int k = Ng; k < Nz + Ng; ++k) {
            for (int j = Ng; j < Ny + Ng; ++j) {
                for (int i = Ng; i <= Nx + Ng; ++i) {
                    if (std::isnan(u_ptr[k * u_plane + j * u_stride + i])) {
                        nan_i = i - Ng; nan_j = j - Ng; nan_k = k - Ng;
                        nan_component = 'u';
                        return true;
                    }
                }
            }
        }
        for (int k = Ng; k < Nz + Ng; ++k) {
            for (int j = Ng; j <= Ny + Ng; ++j) {
                for (int i = Ng; i < Nx + Ng; ++i) {
                    if (std::isnan(v_ptr[k * v_plane + j * v_stride + i])) {
                        nan_i = i - Ng; nan_j = j - Ng; nan_k = k - Ng;
                        nan_component = 'v';
                        return true;
                    }
                }
            }
        }
        if (w_ptr) {
            for (int k = Ng; k <= Nz + Ng; ++k) {
                for (int j = Ng; j < Ny + Ng; ++j) {
                    for (int i = Ng; i < Nx + Ng; ++i) {
                        if (std::isnan(w_ptr[k * w_plane + j * w_stride + i])) {
                            nan_i = i - Ng; nan_j = j - Ng; nan_k = k - Ng;
                            nan_component = 'w';
                            return true;
                        }
                    }
                }
            }
        }
    }
    return false;
#endif
}

/// Compute CFL_y_max = max|v| * dt / dy_local(j) for stretched grids
double compute_cfl_y_max(const double* v_ptr,
                          int Nx, int Ny, int Nz, int Ng,
                          int v_stride, int v_plane,
                          bool is_2d, double dt,
                          const std::vector<double>& dyv, double dy_uniform) {
    double cfl_y_max = 0.0;

#ifdef USE_GPU_OFFLOAD
    const double* v_dev = gpu::dev_ptr(const_cast<double*>(v_ptr));
    const double* dyv_ptr = dyv.empty() ? nullptr : dyv.data();
    const double dy_u = dy_uniform;
    const bool use_dyv = !dyv.empty();

    if (is_2d) {
        #pragma omp target teams distribute parallel for collapse(2) reduction(max:cfl_y_max) \
            is_device_ptr(v_dev) map(to: dyv_ptr[:dyv.size()])
        for (int j = Ng; j <= Ny + Ng; ++j) {
            for (int i = Ng; i < Nx + Ng; ++i) {
                int idx = j * v_stride + i;
                double v_abs = v_dev[idx];
                if (v_abs < 0) v_abs = -v_abs;
                double dy_local = use_dyv ? dyv_ptr[j] : dy_u;
                double cfl = v_abs * dt / dy_local;
                if (cfl > cfl_y_max) cfl_y_max = cfl;
            }
        }
    } else {
        #pragma omp target teams distribute parallel for collapse(3) reduction(max:cfl_y_max) \
            is_device_ptr(v_dev) map(to: dyv_ptr[:dyv.size()])
        for (int k = Ng; k < Nz + Ng; ++k) {
            for (int j = Ng; j <= Ny + Ng; ++j) {
                for (int i = Ng; i < Nx + Ng; ++i) {
                    int idx = k * v_plane + j * v_stride + i;
                    double v_abs = v_dev[idx];
                    if (v_abs < 0) v_abs = -v_abs;
                    double dy_local = use_dyv ? dyv_ptr[j] : dy_u;
                    double cfl = v_abs * dt / dy_local;
                    if (cfl > cfl_y_max) cfl_y_max = cfl;
                }
            }
        }
    }
#else
    for (int k = Ng; k < (is_2d ? Ng + 1 : Nz + Ng); ++k) {
        for (int j = Ng; j <= Ny + Ng; ++j) {
            for (int i = Ng; i < Nx + Ng; ++i) {
                int idx = is_2d ? (j * v_stride + i) : (k * v_plane + j * v_stride + i);
                double v_abs = std::abs(v_ptr[idx]);
                double dy_local = dyv.empty() ? dy_uniform : dyv[j];
                double cfl = v_abs * dt / dy_local;
                if (cfl > cfl_y_max) cfl_y_max = cfl;
            }
        }
    }
#endif
    return cfl_y_max;
}

/// Check if step is in diagnostic range (controlled by NAN_DIAG_START/NAN_DIAG_END env vars)
bool in_diagnostic_range(int step) {
    static int diag_start = -1, diag_end = -1;
    static bool initialized = false;
    if (!initialized) {
        const char* start_env = std::getenv("NAN_DIAG_START");
        const char* end_env = std::getenv("NAN_DIAG_END");
        // Disabled by default (avoids GPU sync overhead in tests/production).
        // Set NAN_DIAG_START=180 NAN_DIAG_END=260 to enable for DNS debugging.
        diag_start = start_env ? std::atoi(start_env) : -1;
        diag_end = end_env ? std::atoi(end_env) : -1;
        initialized = true;
    }
    return step >= diag_start && step <= diag_end;
}

} // anonymous namespace

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

    double *u_in = nullptr, *v_in = nullptr, *w_in = nullptr;
    double *u_out = nullptr, *v_out = nullptr, *w_out = nullptr;
    get_ptrs(vel_in, u_in, v_in, w_in);
    get_ptrs(vel_out, u_out, v_out, w_out);

    // Guard against unsupported VectorField inputs
    const bool needs_w = !mesh_->is2D();
    if (!u_in || !v_in || (needs_w && !w_in) ||
        !u_out || !v_out || (needs_w && !w_out)) {
        throw std::invalid_argument(
            "euler_substep: unsupported VectorField (only velocity_, velocity_star_, "
            "velocity_old_, velocity_rk_ are supported)");
    }

    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Ng = mesh_->Nghost;
    const int u_stride = Nx + 2 * Ng + 1;
    const int v_stride = Nx + 2 * Ng;
    [[maybe_unused]] const size_t u_total = vel_in.u_total_size();
    [[maybe_unused]] const size_t v_total = vel_in.v_total_size();

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

    // Use solve_device() for GPU builds (data is device-resident)
    // Use solve() for CPU builds (data is host-resident)
    // IMPORTANT: Dispatch based on selected_solver_ to use FFT when appropriate
    int vcycles = 0;
#ifdef USE_GPU_OFFLOAD
    switch (selected_solver_) {
#ifdef USE_FFT_POISSON
        case PoissonSolverType::FFT:
            if (fft_poisson_solver_) {
                vcycles = fft_poisson_solver_->solve_device(rhs_poisson_ptr_, pressure_corr_ptr_, pcfg);
            }
            break;
        case PoissonSolverType::FFT2D:
            if (fft2d_poisson_solver_) {
                vcycles = fft2d_poisson_solver_->solve_device(rhs_poisson_ptr_, pressure_corr_ptr_, pcfg);
            }
            break;
        case PoissonSolverType::FFT1D:
            if (fft1d_poisson_solver_) {
                vcycles = fft1d_poisson_solver_->solve_device(rhs_poisson_ptr_, pressure_corr_ptr_, pcfg);
            }
            break;
#endif
        case PoissonSolverType::MG:
        default:
            vcycles = mg_poisson_solver_.solve_device(rhs_poisson_ptr_, pressure_corr_ptr_, pcfg);
            break;
    }
#else
    vcycles = mg_poisson_solver_.solve(rhs_poisson_, pressure_correction_, pcfg);
#endif

    // Populate PoissonStats for external access (for RK paths that call project_velocity)
    if (selected_solver_ == PoissonSolverType::MG) {
        poisson_stats_.cycles = vcycles;
        // Determine solve status from MG solver flags
        if (mg_poisson_solver_.used_fixed_cycles()) {
            poisson_stats_.status = PoissonSolveStatus::FixedCycles;
        } else if (mg_poisson_solver_.converged()) {
            poisson_stats_.status = PoissonSolveStatus::ConvergedToTol;
        } else {
            poisson_stats_.status = PoissonSolveStatus::HitMaxCycles;
        }
        poisson_stats_.rhs_norm_l2 = mg_poisson_solver_.rhs_norm_l2();
        poisson_stats_.rhs_norm_inf = mg_poisson_solver_.rhs_norm();
        poisson_stats_.res_norm_l2 = mg_poisson_solver_.residual_l2();
        poisson_stats_.res_norm_inf = mg_poisson_solver_.residual();
        double b_norm = pcfg.use_l2_norm ? poisson_stats_.rhs_norm_l2 : poisson_stats_.rhs_norm_inf;
        double r_norm = pcfg.use_l2_norm ? poisson_stats_.res_norm_l2 : poisson_stats_.res_norm_inf;
        poisson_stats_.res_over_rhs = (b_norm > 1e-30) ? r_norm / b_norm : 0.0;
    }

    // Track Poisson solve stats for benchmarking (enable with POISSON_STATS=1)
    static bool print_stats = (std::getenv("POISSON_STATS") != nullptr);
    static bool print_debug = (std::getenv("POISSON_DEBUG") != nullptr);
    static int poisson_solve_count = 0;
    static int total_vcycles = 0;
    poisson_solve_count++;
    total_vcycles += vcycles;
    if (print_stats) {
        std::cerr << "[Poisson] solve #" << poisson_solve_count << " vcycles=" << vcycles << "\n";
    }

    // Debug: print RHS and solution norms (enable with POISSON_DEBUG=1)
    if (print_debug && poisson_solve_count <= 10) {
        // Compute max|rhs| and max|p| on device
        double rhs_max = 0.0, p_max = 0.0;
        const double* rhs_dev = gpu::dev_ptr(rhs_poisson_ptr_);
        const double* p_dev = gpu::dev_ptr(pressure_corr_ptr_);
        const int n_cells = Nx * Ny * (is_2d ? 1 : Nz);
        #pragma omp target teams distribute parallel for reduction(max:rhs_max,p_max) is_device_ptr(rhs_dev, p_dev)
        for (int idx = 0; idx < n_cells; ++idx) {
            int ii = idx % Nx + Ng;
            int jk = idx / Nx;
            int jj = (is_2d ? jk : jk % Ny) + Ng;
            int kk = (is_2d ? 0 : jk / Ny) + Ng;
            int flat = kk * plane_stride + jj * stride + ii;
            rhs_max = std::max(rhs_max, std::abs(rhs_dev[flat]));
            p_max = std::max(p_max, std::abs(p_dev[flat]));
        }
        // Also get MG residual if using MG
        double mg_res = (selected_solver_ == PoissonSolverType::MG) ? mg_poisson_solver_.residual() : 0.0;
        std::cerr << "[Poisson DEBUG] solve #" << poisson_solve_count
                  << "  max|rhs|=" << rhs_max << "  max|p|=" << p_max
                  << "  residual=" << mg_res << "  vcycles=" << vcycles << "\n";
    }

    // Copy velocity_ to velocity_star_ for correct_velocity()
    // Use dev_ptr + is_device_ptr pattern (NVHPC workaround)
    const int u_stride = Nx + 2 * Ng + 1;
    const int v_stride = Nx + 2 * Ng;

    // Copy velocity_ to velocity_star_ using kernel calls
    if (is_2d) {
        time_kernels::copy_2d_uv(velocity_u_ptr_, velocity_star_u_ptr_,
                                 velocity_v_ptr_, velocity_star_v_ptr_,
                                 Nx, Ny, Ng, u_stride, v_stride,
                                 velocity_.u_total_size(), velocity_.v_total_size());
    } else {
        const int u_plane = u_stride * (Ny + 2 * Ng);
        const int v_plane = v_stride * (Ny + 2 * Ng + 1);
        const int w_stride_local = Nx + 2 * Ng;
        const int w_plane = w_stride_local * (Ny + 2 * Ng);
        time_kernels::copy_3d_uvw(velocity_u_ptr_, velocity_star_u_ptr_,
                                  velocity_v_ptr_, velocity_star_v_ptr_,
                                  velocity_w_ptr_, velocity_star_w_ptr_,
                                  Nx, Ny, Nz, Ng,
                                  u_stride, v_stride, w_stride_local,
                                  u_plane, v_plane, w_plane,
                                  velocity_.u_total_size(), velocity_.v_total_size(),
                                  velocity_.w_total_size());
    }

    correct_velocity();
    apply_velocity_bc();
    // Note: correct_velocity() already updates pressure internally

    // Adaptive projection: if div is too high, run more cycles
    // Track total cycles used across adaptive iterations
    int total_cycles_used = vcycles;

#ifdef USE_GPU_OFFLOAD
    if (config_.adaptive_projection && selected_solver_ == PoissonSolverType::MG) {
        // Compute max|div u| after correction (GPU reduction)
        compute_divergence(VelocityWhich::Current, div_velocity_);

        double max_div = 0.0;
        const double* div_dev = gpu::dev_ptr(div_velocity_ptr_);

        if (is_2d) {
            #pragma omp target teams distribute parallel for collapse(2) \
                is_device_ptr(div_dev) reduction(max: max_div)
            for (int j = Ng; j < Ny + Ng; ++j) {
                for (int i = Ng; i < Nx + Ng; ++i) {
                    int idx = j * stride + i;
                    double d = div_dev[idx];
                    double abs_d = (d >= 0.0) ? d : -d;
                    if (abs_d > max_div) max_div = abs_d;
                }
            }
        } else {
            #pragma omp target teams distribute parallel for collapse(3) \
                is_device_ptr(div_dev) reduction(max: max_div)
            for (int k = Ng; k < Nz + Ng; ++k) {
                for (int j = Ng; j < Ny + Ng; ++j) {
                    for (int i = Ng; i < Nx + Ng; ++i) {
                        int idx = k * plane_stride + j * stride + i;
                        double d = div_dev[idx];
                        double abs_d = (d >= 0.0) ? d : -d;
                        if (abs_d > max_div) max_div = abs_d;
                    }
                }
            }
        }

        // Adaptive iteration: if div > target and haven't hit max cycles, do more
        int adaptive_iter = 0;
        while (max_div > config_.div_target && total_cycles_used < config_.projection_max_cycles) {
            adaptive_iter++;

            // Recompute RHS from current (partially corrected) velocity
            double mean_div_adapt = 0.0;
            double sum_adapt = 0.0;

            if (is_2d) {
                #pragma omp target teams distribute parallel for collapse(2) reduction(+:sum_adapt) \
                    is_device_ptr(div_dev)
                for (int j = 0; j < Ny; ++j) {
                    for (int i = 0; i < Nx; ++i) {
                        int idx = (j + Ng) * stride + (i + Ng);
                        sum_adapt += div_dev[idx];
                    }
                }
            } else {
                #pragma omp target teams distribute parallel for collapse(3) reduction(+:sum_adapt) \
                    is_device_ptr(div_dev)
                for (int k = 0; k < Nz; ++k) {
                    for (int j = 0; j < Ny; ++j) {
                        for (int i = 0; i < Nx; ++i) {
                            int idx = (k + Ng) * plane_stride + (j + Ng) * stride + (i + Ng);
                            sum_adapt += div_dev[idx];
                        }
                    }
                }
            }
            mean_div_adapt = (count > 0) ? sum_adapt / count : 0.0;

            // Build Poisson RHS
            double* rhs_dev_adapt = gpu::dev_ptr(rhs_poisson_ptr_);
            if (is_2d) {
                #pragma omp target teams distribute parallel for collapse(2) \
                    is_device_ptr(div_dev, rhs_dev_adapt)
                for (int j = 0; j < Ny; ++j) {
                    for (int i = 0; i < Nx; ++i) {
                        int idx = (j + Ng) * stride + (i + Ng);
                        rhs_dev_adapt[idx] = (div_dev[idx] - mean_div_adapt) * dt_inv;
                    }
                }
            } else {
                #pragma omp target teams distribute parallel for collapse(3) \
                    is_device_ptr(div_dev, rhs_dev_adapt)
                for (int k = 0; k < Nz; ++k) {
                    for (int j = 0; j < Ny; ++j) {
                        for (int i = 0; i < Nx; ++i) {
                            int idx = (k + Ng) * plane_stride + (j + Ng) * stride + (i + Ng);
                            rhs_dev_adapt[idx] = (div_dev[idx] - mean_div_adapt) * dt_inv;
                        }
                    }
                }
            }

            // Copy current (corrected) velocity to velocity_star_ for incremental correction
            // This ensures correct_velocity() uses the latest corrected velocity as input
            if (is_2d) {
                time_kernels::copy_2d_uv(velocity_u_ptr_, velocity_star_u_ptr_,
                                         velocity_v_ptr_, velocity_star_v_ptr_,
                                         Nx, Ny, Ng, u_stride, v_stride,
                                         velocity_.u_total_size(), velocity_.v_total_size());
            } else {
                const int u_plane = u_stride * (Ny + 2 * Ng);
                const int v_plane = v_stride * (Ny + 2 * Ng + 1);
                const int w_stride_local = Nx + 2 * Ng;
                const int w_plane = w_stride_local * (Ny + 2 * Ng);
                time_kernels::copy_3d_uvw(velocity_u_ptr_, velocity_star_u_ptr_,
                                          velocity_v_ptr_, velocity_star_v_ptr_,
                                          velocity_w_ptr_, velocity_star_w_ptr_,
                                          Nx, Ny, Nz, Ng,
                                          u_stride, v_stride, w_stride_local,
                                          u_plane, v_plane, w_plane,
                                          velocity_.u_total_size(), velocity_.v_total_size(),
                                          velocity_.w_total_size());
            }

            // Solve Poisson with extra cycles
            PoissonConfig pcfg_adapt;
            pcfg_adapt.max_vcycles = config_.projection_extra_chunk;
            pcfg_adapt.tol_rhs = config_.poisson_tol_rhs;
            pcfg_adapt.fixed_cycles = config_.projection_extra_chunk;

            int extra_cycles = mg_poisson_solver_.solve_device(rhs_poisson_ptr_, pressure_corr_ptr_, pcfg_adapt);
            total_cycles_used += extra_cycles;

            // Correct velocity again (now from updated velocity_star_)
            correct_velocity();
            apply_velocity_bc();

            // Recompute divergence
            compute_divergence(VelocityWhich::Current, div_velocity_);

            max_div = 0.0;
            if (is_2d) {
                #pragma omp target teams distribute parallel for collapse(2) \
                    is_device_ptr(div_dev) reduction(max: max_div)
                for (int j = Ng; j < Ny + Ng; ++j) {
                    for (int i = Ng; i < Nx + Ng; ++i) {
                        int idx = j * stride + i;
                        double d = div_dev[idx];
                        double abs_d = (d >= 0.0) ? d : -d;
                        if (abs_d > max_div) max_div = abs_d;
                    }
                }
            } else {
                #pragma omp target teams distribute parallel for collapse(3) \
                    is_device_ptr(div_dev) reduction(max: max_div)
                for (int k = Ng; k < Nz + Ng; ++k) {
                    for (int j = Ng; j < Ny + Ng; ++j) {
                        for (int i = Ng; i < Nx + Ng; ++i) {
                            int idx = k * plane_stride + j * stride + i;
                            double d = div_dev[idx];
                            double abs_d = (d >= 0.0) ? d : -d;
                            if (abs_d > max_div) max_div = abs_d;
                        }
                    }
                }
            }

            // Log adaptive iteration
            static bool adaptive_verbose = (std::getenv("ADAPTIVE_PROJECTION_VERBOSE") != nullptr);
            if (adaptive_verbose) {
                std::cerr << "[Adaptive Projection] iter=" << adaptive_iter
                          << " max|div|=" << max_div << " total_cycles=" << total_cycles_used << "\n";
            }
        }

        // Update stats with total cycles
        poisson_stats_.cycles = total_cycles_used;
    }
#endif

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
    [[maybe_unused]] const int u_stride = Nx + 2 * Ng + 1;
    [[maybe_unused]] const int v_stride = Nx + 2 * Ng;
    const bool is_2d = mesh_->is2D();

    // Store u^n in velocity_rk_ (use kernel calls to avoid inlining)
    if (is_2d) {
        time_kernels::copy_2d_uv(velocity_u_ptr_, velocity_rk_u_ptr_,
                                 velocity_v_ptr_, velocity_rk_v_ptr_,
                                 Nx, Ny, Ng, u_stride, v_stride,
                                 velocity_.u_total_size(), velocity_.v_total_size());
    } else {
        const int Nz = mesh_->Nz;
        const int u_plane = u_stride * (Ny + 2 * Ng);
        const int v_plane = v_stride * (Ny + 2 * Ng + 1);
        const int w_stride_local = Nx + 2 * Ng;
        const int w_plane = w_stride_local * (Ny + 2 * Ng);
        time_kernels::copy_3d_uvw(velocity_u_ptr_, velocity_rk_u_ptr_,
                                  velocity_v_ptr_, velocity_rk_v_ptr_,
                                  velocity_w_ptr_, velocity_rk_w_ptr_,
                                  Nx, Ny, Nz, Ng,
                                  u_stride, v_stride, w_stride_local,
                                  u_plane, v_plane, w_plane,
                                  velocity_.u_total_size(), velocity_.v_total_size(),
                                  velocity_.w_total_size());
    }

    // Stage 1: u^(1) = u^n + dt * L(u^n)
    euler_substep(velocity_, velocity_star_, dt);
    // Apply inlet BC before projection so Poisson sees correct u^*_n = u_bc
    // This ensures ∂φ/∂n = 0 is consistent at the velocity-Dirichlet inflow
    if (use_recycling_) apply_recycling_inlet_bc();
    project_velocity(velocity_star_, dt);

    // Stage 2: temp = u^(1) + dt * L(u^(1))
    euler_substep(velocity_star_, velocity_, dt);

    // Blend: u^{n+1} = 0.5 * u^n + 0.5 * temp
    if (is_2d) {
        time_kernels::blend_2d_uv(velocity_rk_u_ptr_, velocity_u_ptr_,
                                  velocity_rk_v_ptr_, velocity_v_ptr_,
                                  0.5, 0.5,
                                  Nx, Ny, Ng, u_stride, v_stride,
                                  velocity_.u_total_size(), velocity_.v_total_size());
    } else {
        const int Nz = mesh_->Nz;
        const int u_plane = u_stride * (Ny + 2 * Ng);
        const int v_plane = v_stride * (Ny + 2 * Ng + 1);
        const int w_stride_local = Nx + 2 * Ng;
        const int w_plane = w_stride_local * (Ny + 2 * Ng);
        time_kernels::blend_3d_uvw(velocity_rk_u_ptr_, velocity_u_ptr_,
                                   velocity_rk_v_ptr_, velocity_v_ptr_,
                                   velocity_rk_w_ptr_, velocity_w_ptr_,
                                   0.5, 0.5,
                                   Nx, Ny, Nz, Ng,
                                   u_stride, v_stride, w_stride_local,
                                   u_plane, v_plane, w_plane,
                                   velocity_.u_total_size(), velocity_.v_total_size(),
                                   velocity_.w_total_size());
    }

    // Apply inlet BC BEFORE projection so Poisson can make it div-free
    // Sequence: 1) set v,w at inlet, 2) correct u for div-free, 3) optional fringe blend
    if (use_recycling_) {
        apply_recycling_inlet_bc();     // Sets v, w at inlet
        correct_inlet_divergence();     // Computes u_inlet to make first slab div-free
        apply_fringe_blending();        // Optional smoothing
    }
    project_velocity(velocity_, dt);

    // Recycling inflow: extract from projected (div-free) flow and prepare data for next step
    if (use_recycling_) {
        extract_recycle_plane();
        process_recycle_inflow();
        // Note: DO NOT apply BC after projection - it would introduce divergence
    }

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
    [[maybe_unused]] const int u_stride = Nx + 2 * Ng + 1;
    [[maybe_unused]] const int v_stride = Nx + 2 * Ng;
    const bool is_2d = mesh_->is2D();
    const int Nz = is_2d ? 1 : mesh_->Nz;
    const int u_plane = u_stride * (Ny + 2 * Ng);
    const int v_plane = v_stride * (Ny + 2 * Ng + 1);
    const int w_stride_local = Nx + 2 * Ng;
    const int w_plane = w_stride_local * (Ny + 2 * Ng);

    // Check if in diagnostic range for NaN sentinels
    const bool run_diag = in_diagnostic_range(step_count_);

    // Store u^n in velocity_rk_ (use kernel calls to avoid inlining)
    if (is_2d) {
        time_kernels::copy_2d_uv(velocity_u_ptr_, velocity_rk_u_ptr_,
                                 velocity_v_ptr_, velocity_rk_v_ptr_,
                                 Nx, Ny, Ng, u_stride, v_stride,
                                 velocity_.u_total_size(), velocity_.v_total_size());
    } else {
        time_kernels::copy_3d_uvw(velocity_u_ptr_, velocity_rk_u_ptr_,
                                  velocity_v_ptr_, velocity_rk_v_ptr_,
                                  velocity_w_ptr_, velocity_rk_w_ptr_,
                                  Nx, Ny, Nz, Ng,
                                  u_stride, v_stride, w_stride_local,
                                  u_plane, v_plane, w_plane,
                                  velocity_.u_total_size(), velocity_.v_total_size(),
                                  velocity_.w_total_size());
    }

    // Stage 1: u^(1) = u^n + dt * L(u^n)
    euler_substep(velocity_, velocity_star_, dt);

    // SENTINEL A1: Check for NaN after predictor (Stage 1)
    if (run_diag) {
        int nan_i, nan_j, nan_k;
        char nan_comp;
        if (check_nan_with_location(velocity_star_u_ptr_, velocity_star_v_ptr_,
                                    is_2d ? nullptr : velocity_star_w_ptr_,
                                    Nx, Ny, Nz, Ng,
                                    u_stride, v_stride, w_stride_local,
                                    u_plane, v_plane, w_plane, is_2d,
                                    nan_i, nan_j, nan_k, nan_comp)) {
            std::cerr << "[SENTINEL-A1] Step " << step_count_ << " Stage 1 PREDICTOR: NaN in "
                      << nan_comp << " at (" << nan_i << "," << nan_j << "," << nan_k << ")\n";
            // Compute CFL_y for context
            double cfl_y = compute_cfl_y_max(velocity_v_ptr_, Nx, Ny, Nz, Ng,
                                              v_stride, v_plane, is_2d, dt,
                                              mesh_->dyv, mesh_->dy);
            std::cerr << "[SENTINEL-A1] CFL_y_max = " << cfl_y << " (dt=" << dt << ")\n";
            std::abort();
        }
    }

    // Apply inlet BC before projection so Poisson sees correct u^*_n = u_bc
    if (use_recycling_) apply_recycling_inlet_bc();
    project_velocity(velocity_star_, dt);

    // SENTINEL B1: Check for NaN after projection (Stage 1)
    if (run_diag) {
        int nan_i, nan_j, nan_k;
        char nan_comp;
        if (check_nan_with_location(velocity_star_u_ptr_, velocity_star_v_ptr_,
                                    is_2d ? nullptr : velocity_star_w_ptr_,
                                    Nx, Ny, Nz, Ng,
                                    u_stride, v_stride, w_stride_local,
                                    u_plane, v_plane, w_plane, is_2d,
                                    nan_i, nan_j, nan_k, nan_comp)) {
            std::cerr << "[SENTINEL-B1] Step " << step_count_ << " Stage 1 PROJECTION: NaN in "
                      << nan_comp << " at (" << nan_i << "," << nan_j << "," << nan_k << ")\n";
            std::abort();
        }
    }

    // Stage 2: u^(2) = 0.75 * u^n + 0.25 * (u^(1) + dt * L(u^(1)))
    euler_substep(velocity_star_, velocity_, dt);

    // SENTINEL A2: Check for NaN after predictor (Stage 2)
    if (run_diag) {
        int nan_i, nan_j, nan_k;
        char nan_comp;
        if (check_nan_with_location(velocity_u_ptr_, velocity_v_ptr_,
                                    is_2d ? nullptr : velocity_w_ptr_,
                                    Nx, Ny, Nz, Ng,
                                    u_stride, v_stride, w_stride_local,
                                    u_plane, v_plane, w_plane, is_2d,
                                    nan_i, nan_j, nan_k, nan_comp)) {
            std::cerr << "[SENTINEL-A2] Step " << step_count_ << " Stage 2 PREDICTOR: NaN in "
                      << nan_comp << " at (" << nan_i << "," << nan_j << "," << nan_k << ")\n";
            std::abort();
        }
    }

    // Blend to velocity_star_: 0.75 * velocity_rk_ + 0.25 * velocity_
    if (is_2d) {
        time_kernels::blend_to_2d_uv(velocity_rk_u_ptr_, velocity_u_ptr_, velocity_star_u_ptr_,
                                     velocity_rk_v_ptr_, velocity_v_ptr_, velocity_star_v_ptr_,
                                     0.75, 0.25,
                                     Nx, Ny, Ng, u_stride, v_stride,
                                     velocity_.u_total_size(), velocity_.v_total_size());
    } else {
        const int Nz = mesh_->Nz;
        const int u_plane = u_stride * (Ny + 2 * Ng);
        const int v_plane = v_stride * (Ny + 2 * Ng + 1);
        const int w_stride_local = Nx + 2 * Ng;
        const int w_plane = w_stride_local * (Ny + 2 * Ng);
        time_kernels::blend_to_3d_uvw(velocity_rk_u_ptr_, velocity_u_ptr_, velocity_star_u_ptr_,
                                      velocity_rk_v_ptr_, velocity_v_ptr_, velocity_star_v_ptr_,
                                      velocity_rk_w_ptr_, velocity_w_ptr_, velocity_star_w_ptr_,
                                      0.75, 0.25,
                                      Nx, Ny, Nz, Ng,
                                      u_stride, v_stride, w_stride_local,
                                      u_plane, v_plane, w_plane,
                                      velocity_.u_total_size(), velocity_.v_total_size(),
                                      velocity_.w_total_size());
    }

    // Apply inlet BC before projection
    if (use_recycling_) apply_recycling_inlet_bc();
    project_velocity(velocity_star_, dt);

    // Stage 3: u^{n+1} = (1/3) * u^n + (2/3) * (u^(2) + dt * L(u^(2)))
    euler_substep(velocity_star_, velocity_, dt);

    // SENTINEL A3: Check for NaN after predictor (Stage 3)
    if (run_diag) {
        int nan_i, nan_j, nan_k;
        char nan_comp;
        if (check_nan_with_location(velocity_u_ptr_, velocity_v_ptr_,
                                    is_2d ? nullptr : velocity_w_ptr_,
                                    Nx, Ny, Nz, Ng,
                                    u_stride, v_stride, w_stride_local,
                                    u_plane, v_plane, w_plane, is_2d,
                                    nan_i, nan_j, nan_k, nan_comp)) {
            std::cerr << "[SENTINEL-A3] Step " << step_count_ << " Stage 3 PREDICTOR: NaN in "
                      << nan_comp << " at (" << nan_i << "," << nan_j << "," << nan_k << ")\n";
            std::abort();
        }
    }

    // Final blend: velocity_ = (1/3) * velocity_rk_ + (2/3) * velocity_
    if (is_2d) {
        time_kernels::blend_2d_uv(velocity_rk_u_ptr_, velocity_u_ptr_,
                                  velocity_rk_v_ptr_, velocity_v_ptr_,
                                  1.0/3.0, 2.0/3.0,
                                  Nx, Ny, Ng, u_stride, v_stride,
                                  velocity_.u_total_size(), velocity_.v_total_size());
    } else {
        time_kernels::blend_3d_uvw(velocity_rk_u_ptr_, velocity_u_ptr_,
                                   velocity_rk_v_ptr_, velocity_v_ptr_,
                                   velocity_rk_w_ptr_, velocity_w_ptr_,
                                   1.0/3.0, 2.0/3.0,
                                   Nx, Ny, Nz, Ng,
                                   u_stride, v_stride, w_stride_local,
                                   u_plane, v_plane, w_plane,
                                   velocity_.u_total_size(), velocity_.v_total_size(),
                                   velocity_.w_total_size());
    }

    // Apply inlet BC BEFORE projection so Poisson can make it div-free
    // Sequence: 1) set v,w at inlet, 2) correct u for div-free, 3) optional fringe blend
    if (use_recycling_) {
        apply_recycling_inlet_bc();     // Sets v, w at inlet
        correct_inlet_divergence();     // Computes u_inlet to make first slab div-free
        apply_fringe_blending();        // Optional smoothing
    }
    project_velocity(velocity_, dt);

    // SENTINEL B3: Check for NaN after final projection (Stage 3)
    if (run_diag) {
        int nan_i, nan_j, nan_k;
        char nan_comp;
        if (check_nan_with_location(velocity_u_ptr_, velocity_v_ptr_,
                                    is_2d ? nullptr : velocity_w_ptr_,
                                    Nx, Ny, Nz, Ng,
                                    u_stride, v_stride, w_stride_local,
                                    u_plane, v_plane, w_plane, is_2d,
                                    nan_i, nan_j, nan_k, nan_comp)) {
            std::cerr << "[SENTINEL-B3] Step " << step_count_ << " Stage 3 PROJECTION: NaN in "
                      << nan_comp << " at (" << nan_i << "," << nan_j << "," << nan_k << ")\n";
            std::abort();
        }

        // Log CFL_y_max at end of step (diagnostic range only)
        double cfl_y = compute_cfl_y_max(velocity_v_ptr_, Nx, Ny, Nz, Ng,
                                          v_stride, v_plane, is_2d, dt,
                                          mesh_->dyv, mesh_->dy);
        std::cerr << "[CFL_Y] Step " << step_count_ << " CFL_y_max = " << cfl_y << "\n";
    }

    apply_velocity_bc();

    // Recycling inflow: extract from projected (div-free) flow and prepare data for next step
    if (use_recycling_) {
        extract_recycle_plane();
        process_recycle_inflow();
        // Note: DO NOT apply BC after projection - it would introduce divergence
    }

#ifndef NDEBUG
    // Verify pointer consistency at end of RK3 step
    if (velocity_u_ptr_ != velocity_.u_data().data()) {
        std::cerr << "[ssprk3_step] ERROR: velocity_u_ptr_ mismatch!\n"
                  << "  velocity_u_ptr_ = " << velocity_u_ptr_
                  << "  velocity_.u_data().data() = " << velocity_.u_data().data() << "\n";
    }
#endif
}

} // namespace nncfd
