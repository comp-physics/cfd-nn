#pragma once

/// @file turbulence_device_view.hpp
/// @brief Minimal header for TurbulenceDeviceView — no heavy includes.
///
/// Extracted from turbulence_model.hpp so GPU kernel files can include
/// just this struct without pulling in mesh.hpp/fields.hpp/config.hpp
/// (which expand to 107K+ preprocessed lines and crash nvc++).

namespace nncfd {

struct TurbulenceDeviceView {
    double* u_face = nullptr;
    double* v_face = nullptr;
    double* w_face = nullptr;
    int u_stride = 0, v_stride = 0, w_stride = 0;
    int u_plane_stride = 0, v_plane_stride = 0, w_plane_stride = 0;

    double* k = nullptr;
    double* omega = nullptr;
    double* nu_t = nullptr;
    int cell_stride = 0, cell_plane_stride = 0;

    double* tau_xx = nullptr;
    double* tau_xy = nullptr;
    double* tau_xz = nullptr;
    double* tau_yy = nullptr;
    double* tau_yz = nullptr;
    double* tau_zz = nullptr;

    double* dudx = nullptr;
    double* dudy = nullptr;
    double* dvdx = nullptr;
    double* dvdy = nullptr;
    double* dudz = nullptr;  // 3D gradient components
    double* dvdz = nullptr;
    double* dwdx = nullptr;
    double* dwdy = nullptr;
    double* dwdz = nullptr;

    double* wall_distance = nullptr;

    const double* dyc = nullptr;
    int dyc_size = 0;
    bool is_y_stretched = false;

    const double* yf = nullptr;
    const double* yc = nullptr;

    int Nx = 0, Ny = 0, Nz = 1, Ng = 0;
    double dx = 0.0, dy = 0.0, dz = 1.0, delta = 0.0;

    int u_total = 0, v_total = 0, w_total = 0;
    int cell_total = 0, yf_total = 0, yc_total = 0;

    bool is3D() const { return Nz > 1; }

    bool is_valid() const {
        return (u_face != nullptr && v_face != nullptr && nu_t != nullptr &&
                dudx != nullptr && dudy != nullptr && dvdx != nullptr && dvdy != nullptr &&
                Nx > 0 && Ny > 0);
    }
};

} // namespace nncfd
