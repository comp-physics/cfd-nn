#include "qoi_extraction.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace nncfd {
namespace qoi {

// =========================================================================
// Profile Extraction (CPU)
//
// These run on CPU after sync_solution_from_gpu(). QoI extraction is
// infrequent (every qoi_freq steps) and touches small data (profiles
// = O(Ny) or O(Ny*Nz)), so CPU overhead is negligible.
// =========================================================================

void compute_cf_x_device(const double* u_ptr, int u_stride,
                         [[maybe_unused]] int u_plane_stride,
                         const double* yf_ptr, [[maybe_unused]] int yf_size,
                         const double* hill_y_ptr,
                         const int* j_first_fluid_ptr,
                         double nu, double u_ref, double dx,
                         double* cf_out,
                         int Nx, [[maybe_unused]] int Ny, int Ng) {
    const double q_ref = 0.5 * u_ref * u_ref;

    for (int i = 0; i < Nx; ++i) {
        int ig = i + Ng;
        int jf = j_first_fluid_ptr[i];
        double y_wall = hill_y_ptr[i];
        double y_cell = 0.5 * (yf_ptr[jf] + yf_ptr[jf + 1]);
        double dy = y_cell - y_wall;
        double u_cell = 0.5 * (u_ptr[jf * u_stride + ig] +
                                u_ptr[jf * u_stride + ig + 1]);

        // Correct for hill slope: the wall-normal distance is dy * cos(θ),
        // NOT dy. On sloped surfaces, using dy alone underestimates tau_w.
        // dh/dx from central differences (periodic wrap for i=0 and i=Nx-1)
        double dhdx;
        if (i == 0) {
            dhdx = (hill_y_ptr[1] - hill_y_ptr[Nx - 1]) / (2.0 * dx);
        } else if (i == Nx - 1) {
            dhdx = (hill_y_ptr[0] - hill_y_ptr[Nx - 2]) / (2.0 * dx);
        } else {
            dhdx = (hill_y_ptr[i + 1] - hill_y_ptr[i - 1]) / (2.0 * dx);
        }
        double cos_theta = 1.0 / std::sqrt(1.0 + dhdx * dhdx);
        double dy_normal = dy * cos_theta;  // wall-normal distance

        double tau_w = nu * u_cell / dy_normal;
        cf_out[i] = tau_w / q_ref;
    }
}

void extract_velocity_profile_device(
    const double* u_ptr, const double* v_ptr,
    int u_stride, int v_stride,
    int u_plane_stride, int v_plane_stride,
    int i_station, [[maybe_unused]] int Nx, int Ny, int Nz, int Ng,
    double* u_profile, double* v_profile) {

    const int ig = i_station + Ng;

    for (int j = 0; j < Ny; ++j) {
        int jg = j + Ng;
        double u_sum = 0.0, v_sum = 0.0;
        int nz_loop = (Nz > 1) ? Nz : 1;
        for (int k = 0; k < nz_loop; ++k) {
            int kg = k + Ng;
            int u_off = (Nz > 1) ? kg * u_plane_stride : 0;
            int v_off = (Nz > 1) ? kg * v_plane_stride : 0;
            u_sum += 0.5 * (u_ptr[u_off + jg * u_stride + ig] +
                            u_ptr[u_off + jg * u_stride + ig + 1]);
            v_sum += 0.5 * (v_ptr[v_off + jg * v_stride + ig] +
                            v_ptr[v_off + (jg + 1) * v_stride + ig]);
        }
        u_profile[j] = u_sum / nz_loop;
        v_profile[j] = v_sum / nz_loop;
    }
}

void accumulate_running_mean_device(double* mean, const double* inst,
                                    int total_size, int sample_count) {
    const double inv_n = 1.0 / sample_count;
    for (int i = 0; i < total_size; ++i) {
        mean[i] += (inst[i] - mean[i]) * inv_n;
    }
}

void extract_cross_section_device(
    const double* u_ptr, const double* v_ptr, const double* w_ptr,
    int u_stride, int v_stride, int w_stride,
    int u_plane_stride, int v_plane_stride, int w_plane_stride,
    int i_station, [[maybe_unused]] int Nx, int Ny, int Nz, int Ng,
    double* u_yz, double* v_yz, double* w_yz) {

    const int ig = i_station + Ng;

    for (int j = 0; j < Ny; ++j) {
        int jg = j + Ng;
        for (int k = 0; k < Nz; ++k) {
            int kg = k + Ng;
            int idx = j * Nz + k;

            u_yz[idx] = 0.5 * (u_ptr[kg * u_plane_stride + jg * u_stride + ig] +
                                u_ptr[kg * u_plane_stride + jg * u_stride + ig + 1]);
            v_yz[idx] = 0.5 * (v_ptr[kg * v_plane_stride + jg * v_stride + ig] +
                                v_ptr[kg * v_plane_stride + (jg + 1) * v_stride + ig]);
            w_yz[idx] = 0.5 * (w_ptr[kg * w_plane_stride + jg * w_stride + ig] +
                                w_ptr[(kg + 1) * w_plane_stride + jg * w_stride + ig]);
        }
    }
}

void compute_wall_shear_y_device(
    const double* u_ptr, int u_stride, int u_plane_stride,
    double nu, double dy_wall,
    int Nx, [[maybe_unused]] int Ny, int Nz, int Ng,
    double* tau_bot, double* tau_top) {

    const double inv_Nx = 1.0 / Nx;
    const int j_bot = Ng;
    const int j_top = Ny + Ng - 1;

    for (int k = 0; k < Nz; ++k) {
        int kg = k + Ng;
        double sum_bot = 0.0, sum_top = 0.0;
        for (int i = 0; i < Nx; ++i) {
            int ig = i + Ng;
            double u_bot = 0.5 * (u_ptr[kg * u_plane_stride + j_bot * u_stride + ig] +
                                  u_ptr[kg * u_plane_stride + j_bot * u_stride + ig + 1]);
            double u_top = 0.5 * (u_ptr[kg * u_plane_stride + j_top * u_stride + ig] +
                                  u_ptr[kg * u_plane_stride + j_top * u_stride + ig + 1]);
            sum_bot += u_bot;
            sum_top += u_top;
        }
        tau_bot[k] = nu * (sum_bot * inv_Nx) / dy_wall;
        tau_top[k] = -nu * (sum_top * inv_Nx) / dy_wall;
    }
}

// =========================================================================
// CPU Utility Functions
// =========================================================================

std::pair<double, double> find_separation_reattachment(
    const double* cf, const double* x_centers, int Nx) {
    double x_sep = -1.0, x_reattach = -1.0;

    for (int i = 1; i < Nx; ++i) {
        if (cf[i - 1] > 0.0 && cf[i] <= 0.0 && x_sep < 0.0) {
            double frac = cf[i - 1] / (cf[i - 1] - cf[i]);
            x_sep = x_centers[i - 1] + frac * (x_centers[i] - x_centers[i - 1]);
        }
        if (cf[i - 1] < 0.0 && cf[i] >= 0.0 && x_sep > 0.0 && x_reattach < 0.0) {
            double frac = -cf[i - 1] / (cf[i] - cf[i - 1]);
            x_reattach = x_centers[i - 1] + frac * (x_centers[i] - x_centers[i - 1]);
        }
    }
    return {x_sep, x_reattach};
}

void write_profile(const std::string& filename,
                   const double* coord, const double* value, int N,
                   const std::string& header) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "[QoI] Warning: cannot open " << filename << "\n";
        return;
    }
    out << "# " << header << "\n";
    out << std::scientific << std::setprecision(8);
    for (int i = 0; i < N; ++i) {
        out << coord[i] << " " << value[i] << "\n";
    }
}

void write_profile_uv(const std::string& filename,
                      const double* coord,
                      const double* u, const double* v, int N,
                      const std::string& header) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "[QoI] Warning: cannot open " << filename << "\n";
        return;
    }
    out << "# " << header << "\n";
    out << std::scientific << std::setprecision(8);
    for (int i = 0; i < N; ++i) {
        out << coord[i] << " " << u[i] << " " << v[i] << "\n";
    }
}

void write_cross_section(const std::string& filename,
                         const double* yc, const double* zc,
                         const double* u, const double* v, const double* w,
                         int Ny, int Nz,
                         const std::string& header) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "[QoI] Warning: cannot open " << filename << "\n";
        return;
    }
    out << "# " << header << "\n";
    out << std::scientific << std::setprecision(8);
    for (int j = 0; j < Ny; ++j) {
        for (int k = 0; k < Nz; ++k) {
            int idx = j * Nz + k;
            out << yc[j] << " " << zc[k] << " "
                << u[idx] << " " << v[idx] << " " << w[idx] << "\n";
        }
    }
}

void append_timeseries(const std::string& filename,
                       int step, double time,
                       const std::vector<double>& values) {
    std::ofstream out(filename, std::ios::app);
    if (!out) {
        std::cerr << "[QoI] Warning: cannot open " << filename << "\n";
        return;
    }
    out << std::scientific << std::setprecision(8);
    out << step << " " << time;
    for (double val : values) {
        out << " " << val;
    }
    out << "\n";
}

double compute_strouhal(const std::vector<double>& time,
                        const std::vector<double>& cl,
                        double diameter, double U_inf) {
    int N = static_cast<int>(cl.size());
    if (N < 4) return -1.0;

    // Use second half of the time series (skip transient)
    int start = N / 2;

    // Find zero-crossings (positive-going) in Cl
    std::vector<double> crossing_times;
    for (int i = start + 1; i < N; ++i) {
        if (cl[i - 1] < 0.0 && cl[i] >= 0.0) {
            // Linear interpolation for crossing time
            double frac = -cl[i - 1] / (cl[i] - cl[i - 1]);
            double t_cross = time[i - 1] + frac * (time[i] - time[i - 1]);
            crossing_times.push_back(t_cross);
        }
    }

    if (crossing_times.size() < 2) return -1.0;

    // Average period from consecutive crossings
    double total_period = 0.0;
    int n_periods = static_cast<int>(crossing_times.size()) - 1;
    for (int i = 0; i < n_periods; ++i) {
        total_period += crossing_times[i + 1] - crossing_times[i];
    }
    double avg_period = total_period / n_periods;

    double freq = 1.0 / avg_period;
    return freq * diameter / U_inf;
}

double compute_separation_angle_sphere(
    const double* u_ptr, const double* v_ptr,
    int u_stride, int v_stride,
    int u_plane_stride, int v_plane_stride,
    double cx, double cy, double radius, double probe_offset,
    const double* xf, const double* yf, int Nx, int Ny, int Nz, int Ng) {

    double r_probe = radius + probe_offset;
    int n_angles = 360;
    double prev_v_tan = 0.0;
    double sep_angle = -1.0;

    // Mid-plane in z for 3D
    int k_mid = Nz / 2;
    int kg_mid = k_mid + Ng;

    for (int a = 0; a < n_angles; ++a) {
        // Angle from front stagnation point (theta=0 is upstream face)
        double theta = M_PI * a / n_angles;  // 0 to pi
        double px = cx - r_probe * std::cos(theta);  // front is -x direction
        double py = cy + r_probe * std::sin(theta);

        // Find enclosing cell indices
        int ic = -1, jc = -1;
        for (int i = Ng; i < Nx + Ng; ++i) {
            if (xf[i] <= px && px < xf[i + 1]) { ic = i; break; }
        }
        for (int j = Ng; j < Ny + Ng; ++j) {
            if (yf[j] <= py && py < yf[j + 1]) { jc = j; break; }
        }
        if (ic < 0 || jc < 0) continue;

        // Bilinear interpolation of u and v at probe point
        // Simple: use nearest cell-center velocity
        int u_off = (Nz > 1) ? kg_mid * u_plane_stride : 0;
        int v_off = (Nz > 1) ? kg_mid * v_plane_stride : 0;

        double u_local = 0.5 * (u_ptr[u_off + jc * u_stride + ic] +
                                 u_ptr[u_off + jc * u_stride + ic + 1]);
        double v_local = 0.5 * (v_ptr[v_off + jc * v_stride + ic] +
                                 v_ptr[v_off + (jc + 1) * v_stride + ic]);

        // Tangential velocity: v_tan = -u*sin(theta) + v*cos(theta)
        double v_tan = -u_local * std::sin(theta) + v_local * std::cos(theta);

        // Separation: tangential velocity changes sign (positive to negative)
        if (a > 0 && prev_v_tan > 0.0 && v_tan <= 0.0) {
            // Interpolate
            double frac = prev_v_tan / (prev_v_tan - v_tan);
            sep_angle = M_PI * (a - 1 + frac) / n_angles;
            break;
        }
        prev_v_tan = v_tan;
    }

    if (sep_angle < 0.0) return -1.0;  // Not found
    return sep_angle * 180.0 / M_PI;   // Convert to degrees
}

void extract_wake_profile(
    const double* u_ptr, const double* v_ptr,
    int u_stride, int v_stride,
    int u_plane_stride, int v_plane_stride,
    int Nx, int Ny, int Nz, int Ng,
    double x_station, const double* xc, const double* yc,
    const std::string& filename, const std::string& header) {

    // Find closest grid index to x_station
    int i_station = 0;
    double min_dist = 1e30;
    for (int i = 0; i < Nx; ++i) {
        double d = std::abs(xc[i + Ng] - x_station);
        if (d < min_dist) { min_dist = d; i_station = i; }
    }

    std::vector<double> u_prof(Ny), v_prof(Ny);
    extract_velocity_profile_device(
        u_ptr, v_ptr,
        u_stride, v_stride,
        u_plane_stride, v_plane_stride,
        i_station, Nx, Ny, Nz, Ng,
        u_prof.data(), v_prof.data());

    std::vector<double> yc_arr(Ny);
    for (int j = 0; j < Ny; ++j) yc_arr[j] = yc[j + Ng];

    write_profile_uv(filename, yc_arr.data(),
                     u_prof.data(), v_prof.data(), Ny, header);
}

} // namespace qoi
} // namespace nncfd
