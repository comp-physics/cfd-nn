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
                         double nu, double u_ref,
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
        double tau_w = nu * u_cell / dy;
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

} // namespace qoi
} // namespace nncfd
