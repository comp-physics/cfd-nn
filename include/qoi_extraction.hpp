#pragma once

#include "turbulence_model.hpp"
#include <vector>
#include <string>

namespace nncfd {
namespace qoi {

/// Extract skin friction Cf(x) along the bottom wall of periodic hills.
/// Pre-computed arrays hill_y[Nx] and j_first_fluid[Nx] identify the wall
/// location and first fluid cell at each x-station.
/// Output: cf_out[Nx] = tau_w / (0.5 * rho * U_ref^2)
void compute_cf_x_device(const double* u_ptr, int u_stride, int u_plane_stride,
                         const double* yf_ptr, int yf_size,
                         const double* hill_y_ptr, const int* j_first_fluid_ptr,
                         double nu, double u_ref,
                         double* cf_out,
                         int Nx, int Ny, int Ng);

/// Extract a velocity profile U(y), V(y) at a given x-station i_station.
/// For 3D, averages over z. Output arrays must be Ny in length.
void extract_velocity_profile_device(
    const double* u_ptr, const double* v_ptr,
    int u_stride, int v_stride,
    int u_plane_stride, int v_plane_stride,
    int i_station, int Nx, int Ny, int Nz, int Ng,
    double* u_profile, double* v_profile);

/// Accumulate running mean on GPU: mean = mean + (inst - mean) / n
/// Both mean and inst arrays must be GPU-resident with size total_size.
void accumulate_running_mean_device(double* mean, const double* inst,
                                    int total_size, int sample_count);

/// Extract a y-z cross-section of all 3 velocity components at x-station i_station.
/// Interpolates staggered velocities to cell centers.
/// Output arrays must be Ny*Nz in length (j-major order).
void extract_cross_section_device(
    const double* u_ptr, const double* v_ptr, const double* w_ptr,
    int u_stride, int v_stride, int w_stride,
    int u_plane_stride, int v_plane_stride, int w_plane_stride,
    int i_station, int Nx, int Ny, int Nz, int Ng,
    double* u_yz, double* v_yz, double* w_yz);

/// Compute wall shear stress along y-walls (top/bottom) for a duct.
/// Output: tau_bot[Nz] at j=Ng, tau_top[Nz] at j=Ny+Ng-1 (x-averaged).
void compute_wall_shear_y_device(
    const double* u_ptr, int u_stride, int u_plane_stride,
    double nu, double dy_wall,
    int Nx, int Ny, int Nz, int Ng,
    double* tau_bot, double* tau_top);

// =========================================================================
// CPU utility functions (post-processing of GPU-extracted data)
// =========================================================================

/// Find separation and reattachment points from Cf(x) array.
/// Returns (x_separation, x_reattachment). If not found, returns (-1, -1).
std::pair<double, double> find_separation_reattachment(
    const double* cf, const double* x_centers, int Nx);

/// Write a 1D profile to file (two-column: coord, value).
void write_profile(const std::string& filename,
                   const double* coord, const double* value, int N,
                   const std::string& header);

/// Write a 2D profile to file (three-column: coord, u, v).
void write_profile_uv(const std::string& filename,
                      const double* coord,
                      const double* u, const double* v, int N,
                      const std::string& header);

/// Write duct cross-section to file (5-column: y, z, u, v, w).
void write_cross_section(const std::string& filename,
                         const double* yc, const double* zc,
                         const double* u, const double* v, const double* w,
                         int Ny, int Nz,
                         const std::string& header);

/// Append a line to a time-series file (forces, etc.)
void append_timeseries(const std::string& filename,
                       int step, double time,
                       const std::vector<double>& values);

} // namespace qoi
} // namespace nncfd
