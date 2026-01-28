/// @file solver_kernels.hpp
/// @brief Unified CPU/GPU kernels for the RANS solver
///
/// This file contains inline kernel functions that are compiled for both
/// host and device when USE_GPU_OFFLOAD is enabled. These kernels implement:
/// - Boundary condition application (normal and tangential)
/// - Divergence computation
/// - Velocity correction (pressure projection)
/// - Convective term computation (central, skew-symmetric, upwind, O4)
/// - Diffusive term computation
/// - Poisson solver helper kernels
///
/// All kernels operate on raw pointers with explicit strides, making them
/// suitable for use in both CPU loops and GPU parallel regions.

#ifndef NNCFD_SOLVER_KERNELS_HPP
#define NNCFD_SOLVER_KERNELS_HPP

#include "stencil_operators.hpp"

namespace nncfd {
namespace kernels {

// Import stencil operators for use in GPU kernels
using namespace nncfd::stencil;


#ifdef USE_GPU_OFFLOAD
#pragma omp declare target
#endif

// Unified normal velocity BC kernel for staggered grid
// Handles both u-velocity in x-direction and v-velocity in y-direction
// @param idx_fixed     Index along the fixed direction (j for u_bc_x, i for v_bc_y)
// @param g             Ghost layer number (0 to Ng-1)
// @param N_domain      Domain size in the normal direction (Nx for u_bc_x, Ny for v_bc_y)
// @param Ng            Number of ghost cells
// @param stride        Array stride
// @param row_major     If true: linear_idx = fixed * stride + varying (for u_bc_x)
//                      If false: linear_idx = varying * stride + fixed (for v_bc_y)
// @param lo_periodic, lo_noslip  Low boundary condition flags
// @param hi_periodic, hi_noslip  High boundary condition flags
// @param ptr           Pointer to velocity array
inline void apply_normal_bc_staggered(
    int idx_fixed, int g,
    int N_domain, int Ng, int stride,
    bool row_major,
    bool lo_periodic, bool lo_noslip,
    bool hi_periodic, bool hi_noslip,
    double* ptr)
{
    // Helper lambda for computing linear index based on layout
    auto lin_idx = [=](int idx_varying) {
        return row_major ? (idx_fixed * stride + idx_varying)
                         : (idx_varying * stride + idx_fixed);
    };

    // CRITICAL for staggered grid with periodic BCs:
    // The far interior face (Ng+N) IS the near interior face (Ng)
    // They represent the same physical location in a periodic domain
    if (lo_periodic && hi_periodic && g == 0) {
        int idx_lo = Ng;
        int idx_hi = Ng + N_domain;
        ptr[lin_idx(idx_hi)] = ptr[lin_idx(idx_lo)];
    }

    // Low boundary: normal velocity at faces
    if (lo_noslip) {
        // No-penetration: velocity = 0 at wall face
        if (g == 0) {
            ptr[lin_idx(Ng)] = 0.0;
        }
        // Set ghost faces to zero for stencil consistency
        int idx_ghost = Ng - 1 - g;
        ptr[lin_idx(idx_ghost)] = 0.0;
    } else if (lo_periodic) {
        int idx_ghost = Ng - 1 - g;
        int idx_periodic = Ng + N_domain - 1 - g;
        ptr[lin_idx(idx_ghost)] = ptr[lin_idx(idx_periodic)];
    }

    // High boundary
    if (hi_noslip) {
        // No-penetration: velocity = 0 at wall face
        if (g == 0) {
            ptr[lin_idx(Ng + N_domain)] = 0.0;
        }
        // Set ghost faces to zero for stencil consistency
        int idx_ghost = Ng + N_domain + 1 + g;
        ptr[lin_idx(idx_ghost)] = 0.0;
    } else if (hi_periodic) {
        int idx_ghost = Ng + N_domain + 1 + g;  // Ghost on far side (+1 because Ng+N is now interior)
        int idx_periodic = Ng + 1 + g;           // Wrap from near interior
        ptr[lin_idx(idx_ghost)] = ptr[lin_idx(idx_periodic)];
    }
}

// Wrapper for u-velocity BC in x-direction (maintains backward compatibility)
inline void apply_u_bc_x_staggered(
    int j, int g,
    int Nx, int Ng, int u_stride,
    bool x_lo_periodic, bool x_lo_noslip,
    bool x_hi_periodic, bool x_hi_noslip,
    double* u_ptr)
{
    apply_normal_bc_staggered(j, g, Nx, Ng, u_stride, true,
                              x_lo_periodic, x_lo_noslip,
                              x_hi_periodic, x_hi_noslip, u_ptr);
}

// Unified tangential velocity BC kernel for staggered grid
// Handles both u-velocity in y-direction and v-velocity in x-direction
// @param idx_fixed     Index along the fixed direction (i for u_bc_y, j for v_bc_x)
// @param g             Ghost layer number (0 to Ng-1)
// @param N_domain      Domain size in the ghost direction (Ny for u_bc_y, Nx for v_bc_x)
// @param Ng            Number of ghost cells
// @param stride        Array stride
// @param row_major     If true: linear_idx = varying * stride + fixed (for u_bc_y)
//                      If false: linear_idx = fixed * stride + varying (for v_bc_x)
// @param lo_periodic, lo_noslip  Low boundary condition flags
// @param hi_periodic, hi_noslip  High boundary condition flags
// @param ptr           Pointer to velocity array
inline void apply_tangential_bc_staggered(
    int idx_fixed, int g,
    int N_domain, int Ng, int stride,
    bool row_major,
    bool lo_periodic, bool lo_noslip,
    bool hi_periodic, bool hi_noslip,
    double* ptr)
{
    // Compute linear index based on layout
    // row_major: linear_idx = varying * stride + fixed (for u_bc_y)
    // col_major: linear_idx = fixed * stride + varying (for v_bc_x)
    int idx_lo_ghost = Ng - 1 - g;
    int idx_lo_interior = Ng + g;
    int idx_lo_periodic = N_domain + Ng - 1 - g;
    int idx_hi_ghost = N_domain + Ng + g;
    int idx_hi_interior = N_domain + Ng - 1 - g;
    int idx_hi_periodic = Ng + g;

    // Low boundary (bottom for y, left for x)
    if (lo_noslip) {
        int lin_ghost = row_major ? (idx_lo_ghost * stride + idx_fixed) : (idx_fixed * stride + idx_lo_ghost);
        int lin_interior = row_major ? (idx_lo_interior * stride + idx_fixed) : (idx_fixed * stride + idx_lo_interior);
        ptr[lin_ghost] = -ptr[lin_interior];
    } else if (lo_periodic) {
        int lin_ghost = row_major ? (idx_lo_ghost * stride + idx_fixed) : (idx_fixed * stride + idx_lo_ghost);
        int lin_periodic = row_major ? (idx_lo_periodic * stride + idx_fixed) : (idx_fixed * stride + idx_lo_periodic);
        ptr[lin_ghost] = ptr[lin_periodic];
    }

    // High boundary (top for y, right for x)
    if (hi_noslip) {
        int lin_ghost = row_major ? (idx_hi_ghost * stride + idx_fixed) : (idx_fixed * stride + idx_hi_ghost);
        int lin_interior = row_major ? (idx_hi_interior * stride + idx_fixed) : (idx_fixed * stride + idx_hi_interior);
        ptr[lin_ghost] = -ptr[lin_interior];
    } else if (hi_periodic) {
        int lin_ghost = row_major ? (idx_hi_ghost * stride + idx_fixed) : (idx_fixed * stride + idx_hi_ghost);
        int lin_periodic = row_major ? (idx_hi_periodic * stride + idx_fixed) : (idx_fixed * stride + idx_hi_periodic);
        ptr[lin_ghost] = ptr[lin_periodic];
    }
}

// Wrapper for u-velocity BC in y-direction (maintains backward compatibility)
inline void apply_u_bc_y_staggered(
    int i, int g,
    int Ny, int Ng, int u_stride,
    bool y_lo_periodic, bool y_lo_noslip,
    bool y_hi_periodic, bool y_hi_noslip,
    double* u_ptr)
{
    apply_tangential_bc_staggered(i, g, Ny, Ng, u_stride, true,
                                   y_lo_periodic, y_lo_noslip,
                                   y_hi_periodic, y_hi_noslip, u_ptr);
}

// Wrapper for v-velocity BC in x-direction (maintains backward compatibility)
inline void apply_v_bc_x_staggered(
    int j, int g,
    int Nx, int Ng, int v_stride,
    bool x_lo_periodic, bool x_lo_noslip,
    bool x_hi_periodic, bool x_hi_noslip,
    double* v_ptr)
{
    apply_tangential_bc_staggered(j, g, Nx, Ng, v_stride, false,
                                   x_lo_periodic, x_lo_noslip,
                                   x_hi_periodic, x_hi_noslip, v_ptr);
}

// Wrapper for v-velocity BC in y-direction (maintains backward compatibility)
inline void apply_v_bc_y_staggered(
    int i, int g,
    int Ny, int Ng, int v_stride,
    bool y_lo_periodic, bool y_lo_noslip,
    bool y_hi_periodic, bool y_hi_noslip,
    double* v_ptr)
{
    apply_normal_bc_staggered(i, g, Ny, Ng, v_stride, false,
                              y_lo_periodic, y_lo_noslip,
                              y_hi_periodic, y_hi_noslip, v_ptr);
}

// Convective term kernel for a single cell
inline void convective_cell_kernel(
    int cell_idx, int stride, double dx, double dy, bool use_central,
    const double* u_ptr, const double* v_ptr,
    double* conv_u_ptr, double* conv_v_ptr)
{
    double uu = u_ptr[cell_idx];
    double vv = v_ptr[cell_idx];

    double dudx, dudy, dvdx, dvdy;

    if (use_central) {
        // Central differences
        dudx = (u_ptr[cell_idx+1] - u_ptr[cell_idx-1]) / (2.0 * dx);
        dudy = (u_ptr[cell_idx+stride] - u_ptr[cell_idx-stride]) / (2.0 * dy);
        dvdx = (v_ptr[cell_idx+1] - v_ptr[cell_idx-1]) / (2.0 * dx);
        dvdy = (v_ptr[cell_idx+stride] - v_ptr[cell_idx-stride]) / (2.0 * dy);
    } else {
        // First-order upwind
        if (uu >= 0) {
            dudx = (u_ptr[cell_idx] - u_ptr[cell_idx-1]) / dx;
            dvdx = (v_ptr[cell_idx] - v_ptr[cell_idx-1]) / dx;
        } else {
            dudx = (u_ptr[cell_idx+1] - u_ptr[cell_idx]) / dx;
            dvdx = (v_ptr[cell_idx+1] - v_ptr[cell_idx]) / dx;
        }

        if (vv >= 0) {
            dudy = (u_ptr[cell_idx] - u_ptr[cell_idx-stride]) / dy;
            dvdy = (v_ptr[cell_idx] - v_ptr[cell_idx-stride]) / dy;
        } else {
            dudy = (u_ptr[cell_idx+stride] - u_ptr[cell_idx]) / dy;
            dvdy = (v_ptr[cell_idx+stride] - v_ptr[cell_idx]) / dy;
        }
    }

    conv_u_ptr[cell_idx] = uu * dudx + vv * dudy;
    conv_v_ptr[cell_idx] = uu * dvdx + vv * dvdy;
}

// Diffusive term kernel for a single cell
inline void diffusive_cell_kernel(
    int cell_idx, int stride, double dx2, double dy2,
    const double* u_ptr, const double* v_ptr, const double* nu_ptr,
    double* diff_u_ptr, double* diff_v_ptr)
{
    // Face-averaged effective viscosity
    double nu_e = 0.5 * (nu_ptr[cell_idx] + nu_ptr[cell_idx+1]);
    double nu_w = 0.5 * (nu_ptr[cell_idx] + nu_ptr[cell_idx-1]);
    double nu_n = 0.5 * (nu_ptr[cell_idx] + nu_ptr[cell_idx+stride]);
    double nu_s = 0.5 * (nu_ptr[cell_idx] + nu_ptr[cell_idx-stride]);

    // Diffusive flux for u
    double diff_u_x = (nu_e * (u_ptr[cell_idx+1] - u_ptr[cell_idx]) 
                    - nu_w * (u_ptr[cell_idx] - u_ptr[cell_idx-1])) / dx2;
    double diff_u_y = (nu_n * (u_ptr[cell_idx+stride] - u_ptr[cell_idx]) 
                    - nu_s * (u_ptr[cell_idx] - u_ptr[cell_idx-stride])) / dy2;

    // Diffusive flux for v
    double diff_v_x = (nu_e * (v_ptr[cell_idx+1] - v_ptr[cell_idx]) 
                    - nu_w * (v_ptr[cell_idx] - v_ptr[cell_idx-1])) / dx2;
    double diff_v_y = (nu_n * (v_ptr[cell_idx+stride] - v_ptr[cell_idx]) 
                    - nu_s * (v_ptr[cell_idx] - v_ptr[cell_idx-stride])) / dy2;

    diff_u_ptr[cell_idx] = diff_u_x + diff_u_y;
    diff_v_ptr[cell_idx] = diff_v_x + diff_v_y;
}

// Divergence kernel for staggered grid at cell center (i,j)
// u is at x-faces, v is at y-faces
// div(i,j) = (u(i+1,j) - u(i,j))/dx + (v(i,j+1) - v(i,j))/dy
#pragma omp declare target
inline void divergence_cell_kernel_staggered(
    int i, int j, 
    int u_stride, int v_stride, int div_stride,
    double dx, double dy,
    const double* u_ptr, const double* v_ptr,
    double* div_ptr)
{
    // u indices: u(i,j) is at u_ptr[j * u_stride + i]
    // v indices: v(i,j) is at v_ptr[j * v_stride + i]
    const int u_right = j * u_stride + (i + 1);
    const int u_left = j * u_stride + i;
    const int v_top = (j + 1) * v_stride + i;
    const int v_bottom = j * v_stride + i;
    const int div_idx = j * div_stride + i;
    
    double dudx = (u_ptr[u_right] - u_ptr[u_left]) / dx;
    double dvdy = (v_ptr[v_top] - v_ptr[v_bottom]) / dy;
    div_ptr[div_idx] = dudx + dvdy;
}
#pragma omp end declare target

// Velocity correction kernels for staggered grid
// Correct u at x-face (i,j) using pressure at adjacent cell centers
inline void correct_u_face_kernel_staggered(
    int i, int j,
    int u_stride, int p_stride,
    double dx, double dt,
    const double* u_star_ptr, const double* p_corr_ptr,
    double* u_ptr)
{
    // u(i,j) is at x-face between cells (i-1,j) and (i,j)
    const int u_idx = j * u_stride + i;
    const int p_right = j * p_stride + i;
    const int p_left = j * p_stride + (i - 1);
    
    double dp_dx = (p_corr_ptr[p_right] - p_corr_ptr[p_left]) / dx;
    u_ptr[u_idx] = u_star_ptr[u_idx] - dt * dp_dx;
}

// Correct v at y-face (i,j) using pressure at adjacent cell centers
inline void correct_v_face_kernel_staggered(
    int i, int j,
    int v_stride, int p_stride,
    double dy, double dt,
    const double* v_star_ptr, const double* p_corr_ptr,
    double* v_ptr)
{
    // v(i,j) is at y-face between cells (i,j-1) and (i,j)
    const int v_idx = j * v_stride + i;
    const int p_top = j * p_stride + i;
    const int p_bottom = (j - 1) * p_stride + i;
    
    double dp_dy = (p_corr_ptr[p_top] - p_corr_ptr[p_bottom]) / dy;
    v_ptr[v_idx] = v_star_ptr[v_idx] - dt * dp_dy;
}

// Update pressure at cell center (i,j)
inline void update_pressure_kernel(
    int i, int j, int p_stride,
    const double* p_corr_ptr, double* p_ptr)
{
    const int p_idx = j * p_stride + i;
    p_ptr[p_idx] += p_corr_ptr[p_idx];
}

// Convection term for u-momentum at x-face (i,j) - staggered grid
inline void convective_u_face_kernel_staggered(
    int i, int j,
    int u_stride, int v_stride, int conv_stride,
    double dx, double dy, bool use_central,
    const double* u_ptr, const double* v_ptr,
    double* conv_u_ptr)
{
    const int u_idx = j * u_stride + i;
    const double uu = u_ptr[u_idx];
    
    // Interpolate v to x-face (average 4 surrounding v-faces)
    const double v_bl = v_ptr[j * v_stride + (i-1)];
    const double v_br = v_ptr[j * v_stride + i];
    const double v_tl = v_ptr[(j+1) * v_stride + (i-1)];
    const double v_tr = v_ptr[(j+1) * v_stride + i];
    const double vv = 0.25 * (v_bl + v_br + v_tl + v_tr);
    
    double dudx, dudy;
    
    if (use_central) {
        // du/dx at x-face: central difference using neighboring x-faces
        dudx = (u_ptr[j * u_stride + (i+1)] - u_ptr[j * u_stride + (i-1)]) / (2.0 * dx);
        // du/dy at x-face: central difference using vertically adjacent x-faces  
        dudy = (u_ptr[(j+1) * u_stride + i] - u_ptr[(j-1) * u_stride + i]) / (2.0 * dy);
    } else {
        // Upwind du/dx
        if (uu >= 0) {
            dudx = (u_ptr[u_idx] - u_ptr[j * u_stride + (i-1)]) / dx;
        } else {
            dudx = (u_ptr[j * u_stride + (i+1)] - u_ptr[u_idx]) / dx;
        }
        // Upwind du/dy
        if (vv >= 0) {
            dudy = (u_ptr[u_idx] - u_ptr[(j-1) * u_stride + i]) / dy;
        } else {
            dudy = (u_ptr[(j+1) * u_stride + i] - u_ptr[u_idx]) / dy;
        }
    }
    
    const int conv_idx = j * conv_stride + i;
    conv_u_ptr[conv_idx] = uu * dudx + vv * dudy;
}

// Convection term for v-momentum at y-face (i,j) - staggered grid
inline void convective_v_face_kernel_staggered(
    int i, int j,
    int u_stride, int v_stride, int conv_stride,
    double dx, double dy, bool use_central,
    const double* u_ptr, const double* v_ptr,
    double* conv_v_ptr)
{
    const int v_idx = j * v_stride + i;
    const double vv = v_ptr[v_idx];
    
    // Interpolate u to y-face (average 4 surrounding u-faces)
    const double u_bl = u_ptr[(j-1) * u_stride + i];
    const double u_br = u_ptr[(j-1) * u_stride + (i+1)];
    const double u_tl = u_ptr[j * u_stride + i];
    const double u_tr = u_ptr[j * u_stride + (i+1)];
    const double uu = 0.25 * (u_bl + u_br + u_tl + u_tr);
    
    double dvdx, dvdy;
    
    if (use_central) {
        // dv/dx at y-face: central difference using horizontally adjacent y-faces
        dvdx = (v_ptr[j * v_stride + (i+1)] - v_ptr[j * v_stride + (i-1)]) / (2.0 * dx);
        // dv/dy at y-face: central difference using neighboring y-faces
        dvdy = (v_ptr[(j+1) * v_stride + i] - v_ptr[(j-1) * v_stride + i]) / (2.0 * dy);
    } else {
        // Upwind dv/dx
        if (uu >= 0) {
            dvdx = (v_ptr[v_idx] - v_ptr[j * v_stride + (i-1)]) / dx;
        } else {
            dvdx = (v_ptr[j * v_stride + (i+1)] - v_ptr[v_idx]) / dx;
        }
        // Upwind dv/dy
        if (vv >= 0) {
            dvdy = (v_ptr[v_idx] - v_ptr[(j-1) * v_stride + i]) / dy;
        } else {
            dvdy = (v_ptr[(j+1) * v_stride + i] - v_ptr[v_idx]) / dy;
        }
    }
    
    const int conv_idx = j * conv_stride + i;
    conv_v_ptr[conv_idx] = uu * dvdx + vv * dvdy;
}


// Diffusion term for u-momentum at x-face (i,j) - staggered grid
inline void diffusive_u_face_kernel_staggered(
    int i, int j,
    int u_stride, int nu_stride, int diff_stride,
    double dx, double dy,
    const double* u_ptr, const double* nu_ptr,
    double* diff_u_ptr)
{
    const int u_idx = j * u_stride + i;
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    
    // Viscosity at cell centers adjacent to x-face
    const double nu_left = nu_ptr[j * nu_stride + (i-1)];
    const double nu_right = nu_ptr[j * nu_stride + i];

    // Face-averaged viscosity for d2u/dx2 term (east/west faces of u-control-volume)
    const double nu_e = 0.5 * (nu_right + (i+1 < nu_stride ? nu_ptr[j * nu_stride + (i+1)] : nu_right));
    const double nu_w = 0.5 * (nu_left + (i-2 >= 0 ? nu_ptr[j * nu_stride + (i-2)] : nu_left));

    // Face-averaged viscosity for d2u/dy2 term (north/south faces of u-control-volume)
    // These require 4-point averages at the corners of the u-control-volume
    const double nu_n = 0.25 * (nu_ptr[j * nu_stride + (i-1)] + nu_ptr[j * nu_stride + i] +
                                nu_ptr[(j+1) * nu_stride + (i-1)] + nu_ptr[(j+1) * nu_stride + i]);
    const double nu_s = 0.25 * (nu_ptr[(j-1) * nu_stride + (i-1)] + nu_ptr[(j-1) * nu_stride + i] +
                                nu_ptr[j * nu_stride + (i-1)] + nu_ptr[j * nu_stride + i]);
    
    // d2u/dx2 using u at x-faces
    const double d2u_dx2 = (nu_e * (u_ptr[j * u_stride + (i+1)] - u_ptr[u_idx])
                           - nu_w * (u_ptr[u_idx] - u_ptr[j * u_stride + (i-1)])) / dx2;
    
    // d2u/dy2 using u at x-faces
    const double d2u_dy2 = (nu_n * (u_ptr[(j+1) * u_stride + i] - u_ptr[u_idx])
                           - nu_s * (u_ptr[u_idx] - u_ptr[(j-1) * u_stride + i])) / dy2;
    
    const int diff_idx = j * diff_stride + i;
    diff_u_ptr[diff_idx] = d2u_dx2 + d2u_dy2;
}

// Diffusion term for v-momentum at y-face (i,j) - staggered grid
inline void diffusive_v_face_kernel_staggered(
    int i, int j,
    int v_stride, int nu_stride, int diff_stride,
    double dx, double dy,
    const double* v_ptr, const double* nu_ptr,
    double* diff_v_ptr)
{
    const int v_idx = j * v_stride + i;
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    
    // Viscosity at cell centers adjacent to y-face
    const double nu_bottom = nu_ptr[(j-1) * nu_stride + i];
    const double nu_top = nu_ptr[j * nu_stride + i];

    // Face-averaged viscosity for d2v/dx2 term (east/west faces of v-control-volume)
    // These require 4-point averages at the corners of the v-control-volume
    const double nu_e = 0.25 * (nu_ptr[(j-1) * nu_stride + i] + nu_ptr[j * nu_stride + i] +
                                nu_ptr[(j-1) * nu_stride + (i+1)] + nu_ptr[j * nu_stride + (i+1)]);
    const double nu_w = 0.25 * (nu_ptr[(j-1) * nu_stride + (i-1)] + nu_ptr[j * nu_stride + (i-1)] +
                                nu_ptr[(j-1) * nu_stride + i] + nu_ptr[j * nu_stride + i]);

    // Face-averaged viscosity for d2v/dy2 term (north/south faces of v-control-volume)
    const double nu_n = 0.5 * (nu_top + (j+1 < nu_stride ? nu_ptr[(j+1) * nu_stride + i] : nu_top));
    const double nu_s = 0.5 * (nu_bottom + (j-2 >= 0 ? nu_ptr[(j-2) * nu_stride + i] : nu_bottom));
    
    // d2v/dx2 using v at y-faces
    const double d2v_dx2 = (nu_e * (v_ptr[j * v_stride + (i+1)] - v_ptr[v_idx])
                           - nu_w * (v_ptr[v_idx] - v_ptr[j * v_stride + (i-1)])) / dx2;
    
    // d2v/dy2 using v at y-faces
    const double d2v_dy2 = (nu_n * (v_ptr[(j+1) * v_stride + i] - v_ptr[v_idx])
                           - nu_s * (v_ptr[v_idx] - v_ptr[(j-1) * v_stride + i])) / dy2;
    
    const int diff_idx = j * diff_stride + i;
    diff_v_ptr[diff_idx] = d2v_dx2 + d2v_dy2;
}

// ============================================================================
// 3D OPERATOR KERNELS
// ============================================================================

// 3D Divergence: div(i,j,k) = du/dx + dv/dy + dw/dz
#pragma omp declare target
inline void divergence_cell_kernel_staggered_3d(
    int i, int j, int k,
    int u_stride, int u_plane_stride,
    int v_stride, int v_plane_stride,
    int w_stride, int w_plane_stride,
    int div_stride, int div_plane_stride,
    double dx, double dy, double dz,
    const double* u_ptr, const double* v_ptr, const double* w_ptr,
    double* div_ptr)
{
    // u at x-faces, v at y-faces, w at z-faces
    const int u_right = k * u_plane_stride + j * u_stride + (i + 1);
    const int u_left  = k * u_plane_stride + j * u_stride + i;
    const int v_top   = k * v_plane_stride + (j + 1) * v_stride + i;
    const int v_bottom = k * v_plane_stride + j * v_stride + i;
    const int w_front = (k + 1) * w_plane_stride + j * w_stride + i;
    const int w_back  = k * w_plane_stride + j * w_stride + i;
    const int div_idx = k * div_plane_stride + j * div_stride + i;

    double dudx = (u_ptr[u_right] - u_ptr[u_left]) / dx;
    double dvdy = (v_ptr[v_top] - v_ptr[v_bottom]) / dy;
    double dwdz = (w_ptr[w_front] - w_ptr[w_back]) / dz;
    div_ptr[div_idx] = dudx + dvdy + dwdz;
}

// 3D O4 Divergence at cell center (face→center derivatives)
// Uses Dfc_O4 with boundary fallback to O2
// For periodic directions, ghost cells have valid data so O4 is always safe
// Note: For full O4 projection, the Poisson solver would also need O4 Laplacian
inline void divergence_cell_kernel_staggered_O4_3d(
    int i, int j, int k, int Ng, int Nx, int Ny, int Nz,
    int u_stride, int u_plane_stride,
    int v_stride, int v_plane_stride,
    int w_stride, int w_plane_stride,
    int div_stride, int div_plane_stride,
    double dx, double dy, double dz,
    bool periodic_x, bool periodic_y, bool periodic_z,
    const double* u_ptr, const double* v_ptr, const double* w_ptr,
    double* div_ptr)
{
    const int div_idx = k * div_plane_stride + j * div_stride + i;

    // Check if O4 is safe in each direction
    // For periodic: ghost cells have valid data, so O4 is always safe
    // For non-periodic: need faces i-1, i, i+1, i+2 to be valid
    const bool o4_safe_x = (Ng >= 2) && (periodic_x || ((i >= Ng + 1) && (i <= Ng + Nx - 2)));
    const bool o4_safe_y = (Ng >= 2) && (periodic_y || ((j >= Ng + 1) && (j <= Ng + Ny - 2)));
    const bool o4_safe_z = (Ng >= 2) && (periodic_z || ((k >= Ng + 1) && (k <= Ng + Nz - 2)));

    // du/dx at cell center i using u at faces
    double dudx;
    if (o4_safe_x) {
        // O4: Dfc uses faces i-1, i, i+1, i+2
        const int u_im1 = k * u_plane_stride + j * u_stride + (i - 1);
        const int u_i   = k * u_plane_stride + j * u_stride + i;
        const int u_ip1 = k * u_plane_stride + j * u_stride + (i + 1);
        const int u_ip2 = k * u_plane_stride + j * u_stride + (i + 2);
        dudx = Dfc_O4(u_ptr[u_im1], u_ptr[u_i], u_ptr[u_ip1], u_ptr[u_ip2], dx);
    } else {
        // O2 fallback near non-periodic boundaries
        const int u_right = k * u_plane_stride + j * u_stride + (i + 1);
        const int u_left  = k * u_plane_stride + j * u_stride + i;
        dudx = (u_ptr[u_right] - u_ptr[u_left]) / dx;
    }

    // dv/dy at cell center j using v at faces
    double dvdy;
    if (o4_safe_y) {
        const int v_jm1 = k * v_plane_stride + (j - 1) * v_stride + i;
        const int v_j   = k * v_plane_stride + j * v_stride + i;
        const int v_jp1 = k * v_plane_stride + (j + 1) * v_stride + i;
        const int v_jp2 = k * v_plane_stride + (j + 2) * v_stride + i;
        dvdy = Dfc_O4(v_ptr[v_jm1], v_ptr[v_j], v_ptr[v_jp1], v_ptr[v_jp2], dy);
    } else {
        // O2 fallback near non-periodic boundaries
        const int v_top    = k * v_plane_stride + (j + 1) * v_stride + i;
        const int v_bottom = k * v_plane_stride + j * v_stride + i;
        dvdy = (v_ptr[v_top] - v_ptr[v_bottom]) / dy;
    }

    // dw/dz at cell center k using w at faces
    double dwdz;
    if (o4_safe_z) {
        const int w_km1 = (k - 1) * w_plane_stride + j * w_stride + i;
        const int w_k   = k * w_plane_stride + j * w_stride + i;
        const int w_kp1 = (k + 1) * w_plane_stride + j * w_stride + i;
        const int w_kp2 = (k + 2) * w_plane_stride + j * w_stride + i;
        dwdz = Dfc_O4(w_ptr[w_km1], w_ptr[w_k], w_ptr[w_kp1], w_ptr[w_kp2], dz);
    } else {
        // O2 fallback near non-periodic boundaries
        const int w_front = (k + 1) * w_plane_stride + j * w_stride + i;
        const int w_back  = k * w_plane_stride + j * w_stride + i;
        dwdz = (w_ptr[w_front] - w_ptr[w_back]) / dz;
    }

    div_ptr[div_idx] = dudx + dvdy + dwdz;
}

// 3D Velocity correction for u at x-face
inline void correct_u_face_kernel_staggered_3d(
    int i, int j, int k,
    int u_stride, int u_plane_stride,
    int p_stride, int p_plane_stride,
    double dx, double dt,
    const double* u_star_ptr, const double* p_corr_ptr,
    double* u_ptr)
{
    const int u_idx = k * u_plane_stride + j * u_stride + i;
    const int p_right = k * p_plane_stride + j * p_stride + i;
    const int p_left  = k * p_plane_stride + j * p_stride + (i - 1);

    double dp_dx = (p_corr_ptr[p_right] - p_corr_ptr[p_left]) / dx;
    u_ptr[u_idx] = u_star_ptr[u_idx] - dt * dp_dx;
}

// 3D Velocity correction for v at y-face
inline void correct_v_face_kernel_staggered_3d(
    int i, int j, int k,
    int v_stride, int v_plane_stride,
    int p_stride, int p_plane_stride,
    double dy, double dt,
    const double* v_star_ptr, const double* p_corr_ptr,
    double* v_ptr)
{
    const int v_idx = k * v_plane_stride + j * v_stride + i;
    const int p_top    = k * p_plane_stride + j * p_stride + i;
    const int p_bottom = k * p_plane_stride + (j - 1) * p_stride + i;

    double dp_dy = (p_corr_ptr[p_top] - p_corr_ptr[p_bottom]) / dy;
    v_ptr[v_idx] = v_star_ptr[v_idx] - dt * dp_dy;
}

// 3D Velocity correction for w at z-face
inline void correct_w_face_kernel_staggered_3d(
    int i, int j, int k,
    int w_stride, int w_plane_stride,
    int p_stride, int p_plane_stride,
    double dz, double dt,
    const double* w_star_ptr, const double* p_corr_ptr,
    double* w_ptr)
{
    const int w_idx = k * w_plane_stride + j * w_stride + i;
    const int p_front = k * p_plane_stride + j * p_stride + i;
    const int p_back  = (k - 1) * p_plane_stride + j * p_stride + i;

    double dp_dz = (p_corr_ptr[p_front] - p_corr_ptr[p_back]) / dz;
    w_ptr[w_idx] = w_star_ptr[w_idx] - dt * dp_dz;
}

// ============================================================================
// O4 Velocity Correction Kernels (3D) with boundary fallback
// These use Dcf_O4 for center→face pressure gradient in interior,
// falling back to O2 near boundaries where O4 stencil is incomplete.
// ============================================================================

// 3D O4 Velocity correction for u at x-face
// At u-face i, derivative uses cells i-2, i-1, i, i+1
// O4 safe when periodic OR (i >= Ng+2 and i <= Ng+Nx-2)
// For periodic x, ghost cells contain valid periodic data so O4 is always safe
inline void correct_u_face_kernel_staggered_O4_3d(
    int i, int j, int k, int Ng, int Nx,
    int u_stride, int u_plane_stride,
    int p_stride, int p_plane_stride,
    double dx, double dt, bool periodic_x,
    const double* u_star_ptr, const double* p_corr_ptr,
    double* u_ptr)
{
    const int u_idx = k * u_plane_stride + j * u_stride + i;
    const int jj = j;  // y index (unchanged)
    const int kk = k;  // z index (unchanged)

    double dp_dx;

    // Check if O4 stencil is valid in x-direction
    // For periodic: ghost cells have valid data, so O4 is always safe
    // For non-periodic: need cells i-2, i-1, i, i+1 to be interior
    const bool o4_safe = periodic_x || ((i >= Ng + 2) && (i <= Ng + Nx - 2));

    if (o4_safe) {
        // O4: Dcf at face between cells i-1 and i uses cells i-2, i-1, i, i+1
        const int p_im2 = kk * p_plane_stride + jj * p_stride + (i - 2);
        const int p_im1 = kk * p_plane_stride + jj * p_stride + (i - 1);
        const int p_i   = kk * p_plane_stride + jj * p_stride + i;
        const int p_ip1 = kk * p_plane_stride + jj * p_stride + (i + 1);
        dp_dx = Dcf_O4(p_corr_ptr[p_im2], p_corr_ptr[p_im1],
                       p_corr_ptr[p_i], p_corr_ptr[p_ip1], dx);
    } else {
        // O2 fallback near non-periodic boundaries
        const int p_right = kk * p_plane_stride + jj * p_stride + i;
        const int p_left  = kk * p_plane_stride + jj * p_stride + (i - 1);
        dp_dx = (p_corr_ptr[p_right] - p_corr_ptr[p_left]) / dx;
    }

    u_ptr[u_idx] = u_star_ptr[u_idx] - dt * dp_dx;
}

// 3D O4 Velocity correction for v at y-face
// At v-face j, derivative uses cells j-2, j-1, j, j+1
// O4 safe when periodic OR (j >= Ng+2 and j <= Ng+Ny-2)
inline void correct_v_face_kernel_staggered_O4_3d(
    int i, int j, int k, int Ng, int Ny,
    int v_stride, int v_plane_stride,
    int p_stride, int p_plane_stride,
    double dy, double dt, bool periodic_y,
    const double* v_star_ptr, const double* p_corr_ptr,
    double* v_ptr)
{
    const int v_idx = k * v_plane_stride + j * v_stride + i;
    const int ii = i;  // x index (unchanged)
    const int kk = k;  // z index (unchanged)

    double dp_dy;

    // Check if O4 stencil is valid in y-direction
    const bool o4_safe = periodic_y || ((j >= Ng + 2) && (j <= Ng + Ny - 2));

    if (o4_safe) {
        // O4: Dcf at face between cells j-1 and j uses cells j-2, j-1, j, j+1
        const int p_jm2 = kk * p_plane_stride + (j - 2) * p_stride + ii;
        const int p_jm1 = kk * p_plane_stride + (j - 1) * p_stride + ii;
        const int p_j   = kk * p_plane_stride + j * p_stride + ii;
        const int p_jp1 = kk * p_plane_stride + (j + 1) * p_stride + ii;
        dp_dy = Dcf_O4(p_corr_ptr[p_jm2], p_corr_ptr[p_jm1],
                       p_corr_ptr[p_j], p_corr_ptr[p_jp1], dy);
    } else {
        // O2 fallback near non-periodic boundaries
        const int p_top    = kk * p_plane_stride + j * p_stride + ii;
        const int p_bottom = kk * p_plane_stride + (j - 1) * p_stride + ii;
        dp_dy = (p_corr_ptr[p_top] - p_corr_ptr[p_bottom]) / dy;
    }

    v_ptr[v_idx] = v_star_ptr[v_idx] - dt * dp_dy;
}

// 3D O4 Velocity correction for w at z-face
// At w-face k, derivative uses cells k-2, k-1, k, k+1
// O4 safe when periodic OR (k >= Ng+2 and k <= Ng+Nz-2)
inline void correct_w_face_kernel_staggered_O4_3d(
    int i, int j, int k, int Ng, int Nz,
    int w_stride, int w_plane_stride,
    int p_stride, int p_plane_stride,
    double dz, double dt, bool periodic_z,
    const double* w_star_ptr, const double* p_corr_ptr,
    double* w_ptr)
{
    const int w_idx = k * w_plane_stride + j * w_stride + i;
    const int ii = i;  // x index (unchanged)
    const int jj = j;  // y index (unchanged)

    double dp_dz;

    // Check if O4 stencil is valid in z-direction
    const bool o4_safe = periodic_z || ((k >= Ng + 2) && (k <= Ng + Nz - 2));

    if (o4_safe) {
        // O4: Dcf at face between cells k-1 and k uses cells k-2, k-1, k, k+1
        const int p_km2 = (k - 2) * p_plane_stride + jj * p_stride + ii;
        const int p_km1 = (k - 1) * p_plane_stride + jj * p_stride + ii;
        const int p_k   = k * p_plane_stride + jj * p_stride + ii;
        const int p_kp1 = (k + 1) * p_plane_stride + jj * p_stride + ii;
        dp_dz = Dcf_O4(p_corr_ptr[p_km2], p_corr_ptr[p_km1],
                       p_corr_ptr[p_k], p_corr_ptr[p_kp1], dz);
    } else {
        // O2 fallback near non-periodic boundaries
        const int p_front = k * p_plane_stride + jj * p_stride + ii;
        const int p_back  = (k - 1) * p_plane_stride + jj * p_stride + ii;
        dp_dz = (p_corr_ptr[p_front] - p_corr_ptr[p_back]) / dz;
    }

    w_ptr[w_idx] = w_star_ptr[w_idx] - dt * dp_dz;
}

// ============================================================================
// End of O4 Velocity Correction Kernels
// ============================================================================

// 3D Convection term for u-momentum at x-face
inline void convective_u_face_kernel_staggered_3d(
    int i, int j, int k,
    int u_stride, int u_plane_stride,
    int v_stride, int v_plane_stride,
    int w_stride, int w_plane_stride,
    int conv_stride, int conv_plane_stride,
    double dx, double dy, double dz, bool use_central,
    const double* u_ptr, const double* v_ptr, const double* w_ptr,
    double* conv_u_ptr)
{
    const int u_idx = k * u_plane_stride + j * u_stride + i;
    const double uu = u_ptr[u_idx];

    // Interpolate v to x-face (average 4 surrounding v-faces in x-y plane)
    const double v_bl = v_ptr[k * v_plane_stride + j * v_stride + (i-1)];
    const double v_br = v_ptr[k * v_plane_stride + j * v_stride + i];
    const double v_tl = v_ptr[k * v_plane_stride + (j+1) * v_stride + (i-1)];
    const double v_tr = v_ptr[k * v_plane_stride + (j+1) * v_stride + i];
    const double vv = 0.25 * (v_bl + v_br + v_tl + v_tr);

    // Interpolate w to x-face (average 4 surrounding w-faces in x-z plane)
    const double w_bl = w_ptr[k * w_plane_stride + j * w_stride + (i-1)];
    const double w_br = w_ptr[k * w_plane_stride + j * w_stride + i];
    const double w_fl = w_ptr[(k+1) * w_plane_stride + j * w_stride + (i-1)];
    const double w_fr = w_ptr[(k+1) * w_plane_stride + j * w_stride + i];
    const double ww = 0.25 * (w_bl + w_br + w_fl + w_fr);

    double dudx, dudy, dudz;

    if (use_central) {
        dudx = (u_ptr[k * u_plane_stride + j * u_stride + (i+1)] -
                u_ptr[k * u_plane_stride + j * u_stride + (i-1)]) / (2.0 * dx);
        dudy = (u_ptr[k * u_plane_stride + (j+1) * u_stride + i] -
                u_ptr[k * u_plane_stride + (j-1) * u_stride + i]) / (2.0 * dy);
        dudz = (u_ptr[(k+1) * u_plane_stride + j * u_stride + i] -
                u_ptr[(k-1) * u_plane_stride + j * u_stride + i]) / (2.0 * dz);
    } else {
        // Upwind
        if (uu >= 0) {
            dudx = (u_ptr[u_idx] - u_ptr[k * u_plane_stride + j * u_stride + (i-1)]) / dx;
        } else {
            dudx = (u_ptr[k * u_plane_stride + j * u_stride + (i+1)] - u_ptr[u_idx]) / dx;
        }
        if (vv >= 0) {
            dudy = (u_ptr[u_idx] - u_ptr[k * u_plane_stride + (j-1) * u_stride + i]) / dy;
        } else {
            dudy = (u_ptr[k * u_plane_stride + (j+1) * u_stride + i] - u_ptr[u_idx]) / dy;
        }
        if (ww >= 0) {
            dudz = (u_ptr[u_idx] - u_ptr[(k-1) * u_plane_stride + j * u_stride + i]) / dz;
        } else {
            dudz = (u_ptr[(k+1) * u_plane_stride + j * u_stride + i] - u_ptr[u_idx]) / dz;
        }
    }

    const int conv_idx = k * conv_plane_stride + j * conv_stride + i;
    conv_u_ptr[conv_idx] = uu * dudx + vv * dudy + ww * dudz;
}

// 3D Convection term for v-momentum at y-face
inline void convective_v_face_kernel_staggered_3d(
    int i, int j, int k,
    int u_stride, int u_plane_stride,
    int v_stride, int v_plane_stride,
    int w_stride, int w_plane_stride,
    int conv_stride, int conv_plane_stride,
    double dx, double dy, double dz, bool use_central,
    const double* u_ptr, const double* v_ptr, const double* w_ptr,
    double* conv_v_ptr)
{
    const int v_idx = k * v_plane_stride + j * v_stride + i;
    const double vv = v_ptr[v_idx];

    // Interpolate u to y-face (average 4 surrounding u-faces)
    const double u_bl = u_ptr[k * u_plane_stride + (j-1) * u_stride + i];
    const double u_br = u_ptr[k * u_plane_stride + (j-1) * u_stride + (i+1)];
    const double u_tl = u_ptr[k * u_plane_stride + j * u_stride + i];
    const double u_tr = u_ptr[k * u_plane_stride + j * u_stride + (i+1)];
    const double uu = 0.25 * (u_bl + u_br + u_tl + u_tr);

    // Interpolate w to y-face (average 4 surrounding w-faces)
    const double w_bl = w_ptr[k * w_plane_stride + (j-1) * w_stride + i];
    const double w_tl = w_ptr[k * w_plane_stride + j * w_stride + i];
    const double w_bf = w_ptr[(k+1) * w_plane_stride + (j-1) * w_stride + i];
    const double w_tf = w_ptr[(k+1) * w_plane_stride + j * w_stride + i];
    const double ww = 0.25 * (w_bl + w_tl + w_bf + w_tf);

    double dvdx, dvdy, dvdz;

    if (use_central) {
        dvdx = (v_ptr[k * v_plane_stride + j * v_stride + (i+1)] -
                v_ptr[k * v_plane_stride + j * v_stride + (i-1)]) / (2.0 * dx);
        dvdy = (v_ptr[k * v_plane_stride + (j+1) * v_stride + i] -
                v_ptr[k * v_plane_stride + (j-1) * v_stride + i]) / (2.0 * dy);
        dvdz = (v_ptr[(k+1) * v_plane_stride + j * v_stride + i] -
                v_ptr[(k-1) * v_plane_stride + j * v_stride + i]) / (2.0 * dz);
    } else {
        // Upwind
        if (uu >= 0) {
            dvdx = (v_ptr[v_idx] - v_ptr[k * v_plane_stride + j * v_stride + (i-1)]) / dx;
        } else {
            dvdx = (v_ptr[k * v_plane_stride + j * v_stride + (i+1)] - v_ptr[v_idx]) / dx;
        }
        if (vv >= 0) {
            dvdy = (v_ptr[v_idx] - v_ptr[k * v_plane_stride + (j-1) * v_stride + i]) / dy;
        } else {
            dvdy = (v_ptr[k * v_plane_stride + (j+1) * v_stride + i] - v_ptr[v_idx]) / dy;
        }
        if (ww >= 0) {
            dvdz = (v_ptr[v_idx] - v_ptr[(k-1) * v_plane_stride + j * v_stride + i]) / dz;
        } else {
            dvdz = (v_ptr[(k+1) * v_plane_stride + j * v_stride + i] - v_ptr[v_idx]) / dz;
        }
    }

    const int conv_idx = k * conv_plane_stride + j * conv_stride + i;
    conv_v_ptr[conv_idx] = uu * dvdx + vv * dvdy + ww * dvdz;
}

// 3D Convection term for w-momentum at z-face
inline void convective_w_face_kernel_staggered_3d(
    int i, int j, int k,
    int u_stride, int u_plane_stride,
    int v_stride, int v_plane_stride,
    int w_stride, int w_plane_stride,
    int conv_stride, int conv_plane_stride,
    double dx, double dy, double dz, bool use_central,
    const double* u_ptr, const double* v_ptr, const double* w_ptr,
    double* conv_w_ptr)
{
    const int w_idx = k * w_plane_stride + j * w_stride + i;
    const double ww = w_ptr[w_idx];

    // Interpolate u to z-face (average 4 surrounding u-faces)
    const double u_bl = u_ptr[(k-1) * u_plane_stride + j * u_stride + i];
    const double u_br = u_ptr[(k-1) * u_plane_stride + j * u_stride + (i+1)];
    const double u_fl = u_ptr[k * u_plane_stride + j * u_stride + i];
    const double u_fr = u_ptr[k * u_plane_stride + j * u_stride + (i+1)];
    const double uu = 0.25 * (u_bl + u_br + u_fl + u_fr);

    // Interpolate v to z-face (average 4 surrounding v-faces)
    const double v_bl = v_ptr[(k-1) * v_plane_stride + j * v_stride + i];
    const double v_tl = v_ptr[(k-1) * v_plane_stride + (j+1) * v_stride + i];
    const double v_bf = v_ptr[k * v_plane_stride + j * v_stride + i];
    const double v_tf = v_ptr[k * v_plane_stride + (j+1) * v_stride + i];
    const double vv = 0.25 * (v_bl + v_tl + v_bf + v_tf);

    double dwdx, dwdy, dwdz;

    if (use_central) {
        dwdx = (w_ptr[k * w_plane_stride + j * w_stride + (i+1)] -
                w_ptr[k * w_plane_stride + j * w_stride + (i-1)]) / (2.0 * dx);
        dwdy = (w_ptr[k * w_plane_stride + (j+1) * w_stride + i] -
                w_ptr[k * w_plane_stride + (j-1) * w_stride + i]) / (2.0 * dy);
        dwdz = (w_ptr[(k+1) * w_plane_stride + j * w_stride + i] -
                w_ptr[(k-1) * w_plane_stride + j * w_stride + i]) / (2.0 * dz);
    } else {
        // Upwind
        if (uu >= 0) {
            dwdx = (w_ptr[w_idx] - w_ptr[k * w_plane_stride + j * w_stride + (i-1)]) / dx;
        } else {
            dwdx = (w_ptr[k * w_plane_stride + j * w_stride + (i+1)] - w_ptr[w_idx]) / dx;
        }
        if (vv >= 0) {
            dwdy = (w_ptr[w_idx] - w_ptr[k * w_plane_stride + (j-1) * w_stride + i]) / dy;
        } else {
            dwdy = (w_ptr[k * w_plane_stride + (j+1) * w_stride + i] - w_ptr[w_idx]) / dy;
        }
        if (ww >= 0) {
            dwdz = (w_ptr[w_idx] - w_ptr[(k-1) * w_plane_stride + j * w_stride + i]) / dz;
        } else {
            dwdz = (w_ptr[(k+1) * w_plane_stride + j * w_stride + i] - w_ptr[w_idx]) / dz;
        }
    }

    const int conv_idx = k * conv_plane_stride + j * conv_stride + i;
    conv_w_ptr[conv_idx] = uu * dwdx + vv * dwdy + ww * dwdz;
}

// ============================================================================
// O4 CENTRAL ADVECTION KERNELS (4th-order spatial derivatives)
// ============================================================================
// These kernels use O4 same-stagger derivatives for velocity gradients.
// Collocation interpolation remains O2 for now (to be upgraded in follow-up).
// Falls back to O2 near boundaries where O4 stencil is not safe.

// 3D O4 Central Convection for u-momentum at x-face
inline void convective_u_face_kernel_central_O4_3d(
    int i, int j, int k,
    int u_stride, int u_plane_stride,
    int v_stride, int v_plane_stride,
    int w_stride, int w_plane_stride,
    int conv_stride, int conv_plane_stride,
    double dx, double dy, double dz,
    int Ng, int Nx, int Ny, int Nz,
    bool periodic_x, bool periodic_y, bool periodic_z,
    const double* u_ptr, const double* v_ptr, const double* w_ptr,
    double* conv_u_ptr)
{
    const int u_idx = k * u_plane_stride + j * u_stride + i;
    const double uu = u_ptr[u_idx];

    // Interpolate v to x-face (O2: 4-point average in x-y plane)
    const double v_bl = v_ptr[k * v_plane_stride + j * v_stride + (i-1)];
    const double v_br = v_ptr[k * v_plane_stride + j * v_stride + i];
    const double v_tl = v_ptr[k * v_plane_stride + (j+1) * v_stride + (i-1)];
    const double v_tr = v_ptr[k * v_plane_stride + (j+1) * v_stride + i];
    const double vv = 0.25 * (v_bl + v_br + v_tl + v_tr);

    // Interpolate w to x-face (O2: 4-point average in x-z plane)
    const double w_bl = w_ptr[k * w_plane_stride + j * w_stride + (i-1)];
    const double w_br = w_ptr[k * w_plane_stride + j * w_stride + i];
    const double w_fl = w_ptr[(k+1) * w_plane_stride + j * w_stride + (i-1)];
    const double w_fr = w_ptr[(k+1) * w_plane_stride + j * w_stride + i];
    const double ww = 0.25 * (w_bl + w_br + w_fl + w_fr);

    double dudx, dudy, dudz;

    // Check boundary safety for O4 in each direction
    // For periodic: ghost cells have valid data, so O4 is always safe
    // u-faces: x runs from Ng to Ng+Nx (Nx+1 faces), y/z from Ng to Ng+N-1
    const bool safe_x = periodic_x || ((i >= Ng + 2) && (i <= Ng + Nx - 2));
    const bool safe_y = periodic_y || ((j >= Ng + 2) && (j <= Ng + Ny - 3));
    const bool safe_z = periodic_z || ((k >= Ng + 2) && (k <= Ng + Nz - 3));

    // X-derivative: ∂u/∂x at u-face (same-stagger)
    if (safe_x) {
        const double u_im2 = u_ptr[k * u_plane_stride + j * u_stride + (i-2)];
        const double u_im1 = u_ptr[k * u_plane_stride + j * u_stride + (i-1)];
        const double u_ip1 = u_ptr[k * u_plane_stride + j * u_stride + (i+1)];
        const double u_ip2 = u_ptr[k * u_plane_stride + j * u_stride + (i+2)];
        dudx = D_same_O4(u_im2, u_im1, u_ip1, u_ip2, dx);
    } else {
        const double u_im1 = u_ptr[k * u_plane_stride + j * u_stride + (i-1)];
        const double u_ip1 = u_ptr[k * u_plane_stride + j * u_stride + (i+1)];
        dudx = D_same_O2(u_im1, u_ip1, dx);
    }

    // Y-derivative: ∂u/∂y at u-face (same-stagger)
    if (safe_y) {
        const double u_jm2 = u_ptr[k * u_plane_stride + (j-2) * u_stride + i];
        const double u_jm1 = u_ptr[k * u_plane_stride + (j-1) * u_stride + i];
        const double u_jp1 = u_ptr[k * u_plane_stride + (j+1) * u_stride + i];
        const double u_jp2 = u_ptr[k * u_plane_stride + (j+2) * u_stride + i];
        dudy = D_same_O4(u_jm2, u_jm1, u_jp1, u_jp2, dy);
    } else {
        const double u_jm1 = u_ptr[k * u_plane_stride + (j-1) * u_stride + i];
        const double u_jp1 = u_ptr[k * u_plane_stride + (j+1) * u_stride + i];
        dudy = D_same_O2(u_jm1, u_jp1, dy);
    }

    // Z-derivative: ∂u/∂z at u-face (same-stagger)
    if (safe_z) {
        const double u_km2 = u_ptr[(k-2) * u_plane_stride + j * u_stride + i];
        const double u_km1 = u_ptr[(k-1) * u_plane_stride + j * u_stride + i];
        const double u_kp1 = u_ptr[(k+1) * u_plane_stride + j * u_stride + i];
        const double u_kp2 = u_ptr[(k+2) * u_plane_stride + j * u_stride + i];
        dudz = D_same_O4(u_km2, u_km1, u_kp1, u_kp2, dz);
    } else {
        const double u_km1 = u_ptr[(k-1) * u_plane_stride + j * u_stride + i];
        const double u_kp1 = u_ptr[(k+1) * u_plane_stride + j * u_stride + i];
        dudz = D_same_O2(u_km1, u_kp1, dz);
    }

    const int conv_idx = k * conv_plane_stride + j * conv_stride + i;
    conv_u_ptr[conv_idx] = uu * dudx + vv * dudy + ww * dudz;
}

// 3D O4 Central Convection for v-momentum at y-face
inline void convective_v_face_kernel_central_O4_3d(
    int i, int j, int k,
    int u_stride, int u_plane_stride,
    int v_stride, int v_plane_stride,
    int w_stride, int w_plane_stride,
    int conv_stride, int conv_plane_stride,
    double dx, double dy, double dz,
    int Ng, int Nx, int Ny, int Nz,
    bool periodic_x, bool periodic_y, bool periodic_z,
    const double* u_ptr, const double* v_ptr, const double* w_ptr,
    double* conv_v_ptr)
{
    const int v_idx = k * v_plane_stride + j * v_stride + i;
    const double vv = v_ptr[v_idx];

    // Interpolate u to y-face (O2: 4-point average)
    const double u_bl = u_ptr[k * u_plane_stride + (j-1) * u_stride + i];
    const double u_br = u_ptr[k * u_plane_stride + (j-1) * u_stride + (i+1)];
    const double u_tl = u_ptr[k * u_plane_stride + j * u_stride + i];
    const double u_tr = u_ptr[k * u_plane_stride + j * u_stride + (i+1)];
    const double uu = 0.25 * (u_bl + u_br + u_tl + u_tr);

    // Interpolate w to y-face (O2: 4-point average)
    const double w_bl = w_ptr[k * w_plane_stride + (j-1) * w_stride + i];
    const double w_tl = w_ptr[k * w_plane_stride + j * w_stride + i];
    const double w_bf = w_ptr[(k+1) * w_plane_stride + (j-1) * w_stride + i];
    const double w_tf = w_ptr[(k+1) * w_plane_stride + j * w_stride + i];
    const double ww = 0.25 * (w_bl + w_tl + w_bf + w_tf);

    double dvdx, dvdy, dvdz;

    // Boundary safety checks for v at y-faces (periodic-aware)
    const bool safe_x = periodic_x || ((i >= Ng + 2) && (i <= Ng + Nx - 3));
    const bool safe_y = periodic_y || ((j >= Ng + 2) && (j <= Ng + Ny - 2));
    const bool safe_z = periodic_z || ((k >= Ng + 2) && (k <= Ng + Nz - 3));

    // X-derivative
    if (safe_x) {
        const double v_im2 = v_ptr[k * v_plane_stride + j * v_stride + (i-2)];
        const double v_im1 = v_ptr[k * v_plane_stride + j * v_stride + (i-1)];
        const double v_ip1 = v_ptr[k * v_plane_stride + j * v_stride + (i+1)];
        const double v_ip2 = v_ptr[k * v_plane_stride + j * v_stride + (i+2)];
        dvdx = D_same_O4(v_im2, v_im1, v_ip1, v_ip2, dx);
    } else {
        const double v_im1 = v_ptr[k * v_plane_stride + j * v_stride + (i-1)];
        const double v_ip1 = v_ptr[k * v_plane_stride + j * v_stride + (i+1)];
        dvdx = D_same_O2(v_im1, v_ip1, dx);
    }

    // Y-derivative
    if (safe_y) {
        const double v_jm2 = v_ptr[k * v_plane_stride + (j-2) * v_stride + i];
        const double v_jm1 = v_ptr[k * v_plane_stride + (j-1) * v_stride + i];
        const double v_jp1 = v_ptr[k * v_plane_stride + (j+1) * v_stride + i];
        const double v_jp2 = v_ptr[k * v_plane_stride + (j+2) * v_stride + i];
        dvdy = D_same_O4(v_jm2, v_jm1, v_jp1, v_jp2, dy);
    } else {
        const double v_jm1 = v_ptr[k * v_plane_stride + (j-1) * v_stride + i];
        const double v_jp1 = v_ptr[k * v_plane_stride + (j+1) * v_stride + i];
        dvdy = D_same_O2(v_jm1, v_jp1, dy);
    }

    // Z-derivative
    if (safe_z) {
        const double v_km2 = v_ptr[(k-2) * v_plane_stride + j * v_stride + i];
        const double v_km1 = v_ptr[(k-1) * v_plane_stride + j * v_stride + i];
        const double v_kp1 = v_ptr[(k+1) * v_plane_stride + j * v_stride + i];
        const double v_kp2 = v_ptr[(k+2) * v_plane_stride + j * v_stride + i];
        dvdz = D_same_O4(v_km2, v_km1, v_kp1, v_kp2, dz);
    } else {
        const double v_km1 = v_ptr[(k-1) * v_plane_stride + j * v_stride + i];
        const double v_kp1 = v_ptr[(k+1) * v_plane_stride + j * v_stride + i];
        dvdz = D_same_O2(v_km1, v_kp1, dz);
    }

    const int conv_idx = k * conv_plane_stride + j * conv_stride + i;
    conv_v_ptr[conv_idx] = uu * dvdx + vv * dvdy + ww * dvdz;
}

// 3D O4 Central Convection for w-momentum at z-face
inline void convective_w_face_kernel_central_O4_3d(
    int i, int j, int k,
    int u_stride, int u_plane_stride,
    int v_stride, int v_plane_stride,
    int w_stride, int w_plane_stride,
    int conv_stride, int conv_plane_stride,
    double dx, double dy, double dz,
    int Ng, int Nx, int Ny, int Nz,
    bool periodic_x, bool periodic_y, bool periodic_z,
    const double* u_ptr, const double* v_ptr, const double* w_ptr,
    double* conv_w_ptr)
{
    const int w_idx = k * w_plane_stride + j * w_stride + i;
    const double ww = w_ptr[w_idx];

    // Interpolate u to z-face (O2: 4-point average)
    const double u_bl = u_ptr[(k-1) * u_plane_stride + j * u_stride + i];
    const double u_br = u_ptr[(k-1) * u_plane_stride + j * u_stride + (i+1)];
    const double u_fl = u_ptr[k * u_plane_stride + j * u_stride + i];
    const double u_fr = u_ptr[k * u_plane_stride + j * u_stride + (i+1)];
    const double uu = 0.25 * (u_bl + u_br + u_fl + u_fr);

    // Interpolate v to z-face (O2: 4-point average)
    const double v_bl = v_ptr[(k-1) * v_plane_stride + j * v_stride + i];
    const double v_tl = v_ptr[(k-1) * v_plane_stride + (j+1) * v_stride + i];
    const double v_bf = v_ptr[k * v_plane_stride + j * v_stride + i];
    const double v_tf = v_ptr[k * v_plane_stride + (j+1) * v_stride + i];
    const double vv = 0.25 * (v_bl + v_tl + v_bf + v_tf);

    double dwdx, dwdy, dwdz;

    // Boundary safety checks for w at z-faces (periodic-aware)
    const bool safe_x = periodic_x || ((i >= Ng + 2) && (i <= Ng + Nx - 3));
    const bool safe_y = periodic_y || ((j >= Ng + 2) && (j <= Ng + Ny - 3));
    const bool safe_z = periodic_z || ((k >= Ng + 2) && (k <= Ng + Nz - 2));

    // X-derivative
    if (safe_x) {
        const double w_im2 = w_ptr[k * w_plane_stride + j * w_stride + (i-2)];
        const double w_im1 = w_ptr[k * w_plane_stride + j * w_stride + (i-1)];
        const double w_ip1 = w_ptr[k * w_plane_stride + j * w_stride + (i+1)];
        const double w_ip2 = w_ptr[k * w_plane_stride + j * w_stride + (i+2)];
        dwdx = D_same_O4(w_im2, w_im1, w_ip1, w_ip2, dx);
    } else {
        const double w_im1 = w_ptr[k * w_plane_stride + j * w_stride + (i-1)];
        const double w_ip1 = w_ptr[k * w_plane_stride + j * w_stride + (i+1)];
        dwdx = D_same_O2(w_im1, w_ip1, dx);
    }

    // Y-derivative
    if (safe_y) {
        const double w_jm2 = w_ptr[k * w_plane_stride + (j-2) * w_stride + i];
        const double w_jm1 = w_ptr[k * w_plane_stride + (j-1) * w_stride + i];
        const double w_jp1 = w_ptr[k * w_plane_stride + (j+1) * w_stride + i];
        const double w_jp2 = w_ptr[k * w_plane_stride + (j+2) * w_stride + i];
        dwdy = D_same_O4(w_jm2, w_jm1, w_jp1, w_jp2, dy);
    } else {
        const double w_jm1 = w_ptr[k * w_plane_stride + (j-1) * w_stride + i];
        const double w_jp1 = w_ptr[k * w_plane_stride + (j+1) * w_stride + i];
        dwdy = D_same_O2(w_jm1, w_jp1, dy);
    }

    // Z-derivative
    if (safe_z) {
        const double w_km2 = w_ptr[(k-2) * w_plane_stride + j * w_stride + i];
        const double w_km1 = w_ptr[(k-1) * w_plane_stride + j * w_stride + i];
        const double w_kp1 = w_ptr[(k+1) * w_plane_stride + j * w_stride + i];
        const double w_kp2 = w_ptr[(k+2) * w_plane_stride + j * w_stride + i];
        dwdz = D_same_O4(w_km2, w_km1, w_kp1, w_kp2, dz);
    } else {
        const double w_km1 = w_ptr[(k-1) * w_plane_stride + j * w_stride + i];
        const double w_kp1 = w_ptr[(k+1) * w_plane_stride + j * w_stride + i];
        dwdz = D_same_O2(w_km1, w_kp1, dz);
    }

    const int conv_idx = k * conv_plane_stride + j * conv_stride + i;
    conv_w_ptr[conv_idx] = uu * dwdx + vv * dwdy + ww * dwdz;
}

// ============================================================================
// SKEW-SYMMETRIC (SPLIT) ADVECTION KERNELS - Energy Conserving
// ============================================================================
// The skew-symmetric form is: N = 0.5 * (A + C) where
//   A = advective form: u·∇φ
//   C = conservative form: ∇·(u⊗φ)
// This form conserves kinetic energy in the inviscid limit for periodic BCs.

// 2D skew-symmetric convection for u-momentum at x-face (i,j)
inline void convective_u_face_kernel_skew_2d(
    int i, int j,
    int u_stride, int v_stride, int conv_stride,
    double dx, double dy,
    const double* u_ptr, const double* v_ptr,
    double* conv_u_ptr)
{
    const int u_idx = j * u_stride + i;
    const double uu = u_ptr[u_idx];

    // Neighbors for derivatives
    const double u_ip1 = u_ptr[j * u_stride + (i+1)];
    const double u_im1 = u_ptr[j * u_stride + (i-1)];
    const double u_jp1 = u_ptr[(j+1) * u_stride + i];
    const double u_jm1 = u_ptr[(j-1) * u_stride + i];

    // Interpolate v to x-face (average 4 surrounding v-faces)
    const double v_bl = v_ptr[j * v_stride + (i-1)];
    const double v_br = v_ptr[j * v_stride + i];
    const double v_tl = v_ptr[(j+1) * v_stride + (i-1)];
    const double v_tr = v_ptr[(j+1) * v_stride + i];
    const double vv = 0.25 * (v_bl + v_br + v_tl + v_tr);

    // X-direction: skew = 0.5*(advective + conservative)
    // Advective: uu * du/dx
    const double adv_x = uu * (u_ip1 - u_im1) / (2.0 * dx);
    // Conservative: d(uu*u)/dx using interpolated velocities at cell centers
    // u at cell center (i-1,j) ≈ 0.5*(u[i-1] + u[i])
    // u at cell center (i,j) ≈ 0.5*(u[i] + u[i+1])
    const double uu_left = 0.5 * (u_im1 + uu);
    const double uu_right = 0.5 * (uu + u_ip1);
    const double cons_x = (uu_right * uu_right - uu_left * uu_left) / dx;
    const double skew_x = 0.5 * (adv_x + cons_x);

    // Y-direction: skew = 0.5*(advective + conservative)
    // Advective: vv * du/dy
    const double adv_y = vv * (u_jp1 - u_jm1) / (2.0 * dy);
    // Conservative: d(vv*u)/dy using fluxes at edge midpoints
    // vv*u at (i-1/2, j-1/2) and (i-1/2, j+1/2)
    const double vv_bot = 0.5 * (v_bl + v_br);
    const double vv_top = 0.5 * (v_tl + v_tr);
    const double u_bot = 0.5 * (uu + u_jm1);
    const double u_top = 0.5 * (uu + u_jp1);
    const double cons_y = (vv_top * u_top - vv_bot * u_bot) / dy;
    const double skew_y = 0.5 * (adv_y + cons_y);

    const int conv_idx = j * conv_stride + i;
    conv_u_ptr[conv_idx] = skew_x + skew_y;
}

// 2D skew-symmetric convection for v-momentum at y-face (i,j)
inline void convective_v_face_kernel_skew_2d(
    int i, int j,
    int u_stride, int v_stride, int conv_stride,
    double dx, double dy,
    const double* u_ptr, const double* v_ptr,
    double* conv_v_ptr)
{
    const int v_idx = j * v_stride + i;
    const double vv = v_ptr[v_idx];

    // Neighbors for derivatives
    const double v_ip1 = v_ptr[j * v_stride + (i+1)];
    const double v_im1 = v_ptr[j * v_stride + (i-1)];
    const double v_jp1 = v_ptr[(j+1) * v_stride + i];
    const double v_jm1 = v_ptr[(j-1) * v_stride + i];

    // Interpolate u to y-face (average 4 surrounding u-faces)
    const double u_bl = u_ptr[(j-1) * u_stride + i];
    const double u_br = u_ptr[(j-1) * u_stride + (i+1)];
    const double u_tl = u_ptr[j * u_stride + i];
    const double u_tr = u_ptr[j * u_stride + (i+1)];
    const double uu = 0.25 * (u_bl + u_br + u_tl + u_tr);

    // X-direction: skew = 0.5*(advective + conservative)
    const double adv_x = uu * (v_ip1 - v_im1) / (2.0 * dx);
    const double uu_left = 0.5 * (u_bl + u_tl);
    const double uu_right = 0.5 * (u_br + u_tr);
    const double v_left = 0.5 * (v_im1 + vv);
    const double v_right = 0.5 * (vv + v_ip1);
    const double cons_x = (uu_right * v_right - uu_left * v_left) / dx;
    const double skew_x = 0.5 * (adv_x + cons_x);

    // Y-direction: skew = 0.5*(advective + conservative)
    const double adv_y = vv * (v_jp1 - v_jm1) / (2.0 * dy);
    const double vv_bot = 0.5 * (v_jm1 + vv);
    const double vv_top = 0.5 * (vv + v_jp1);
    const double cons_y = (vv_top * vv_top - vv_bot * vv_bot) / dy;
    const double skew_y = 0.5 * (adv_y + cons_y);

    const int conv_idx = j * conv_stride + i;
    conv_v_ptr[conv_idx] = skew_x + skew_y;
}

// 3D skew-symmetric convection for u-momentum at x-face
inline void convective_u_face_kernel_skew_3d(
    int i, int j, int k,
    int u_stride, int u_plane_stride,
    int v_stride, int v_plane_stride,
    int w_stride, int w_plane_stride,
    int conv_stride, int conv_plane_stride,
    double dx, double dy, double dz,
    const double* u_ptr, const double* v_ptr, const double* w_ptr,
    double* conv_u_ptr)
{
    const int u_idx = k * u_plane_stride + j * u_stride + i;
    const double uu = u_ptr[u_idx];

    // Neighbors
    const double u_ip1 = u_ptr[k * u_plane_stride + j * u_stride + (i+1)];
    const double u_im1 = u_ptr[k * u_plane_stride + j * u_stride + (i-1)];
    const double u_jp1 = u_ptr[k * u_plane_stride + (j+1) * u_stride + i];
    const double u_jm1 = u_ptr[k * u_plane_stride + (j-1) * u_stride + i];
    const double u_kp1 = u_ptr[(k+1) * u_plane_stride + j * u_stride + i];
    const double u_km1 = u_ptr[(k-1) * u_plane_stride + j * u_stride + i];

    // Interpolate v to x-face
    const double v_bl = v_ptr[k * v_plane_stride + j * v_stride + (i-1)];
    const double v_br = v_ptr[k * v_plane_stride + j * v_stride + i];
    const double v_tl = v_ptr[k * v_plane_stride + (j+1) * v_stride + (i-1)];
    const double v_tr = v_ptr[k * v_plane_stride + (j+1) * v_stride + i];
    const double vv = 0.25 * (v_bl + v_br + v_tl + v_tr);

    // Interpolate w to x-face
    const double w_bl = w_ptr[k * w_plane_stride + j * w_stride + (i-1)];
    const double w_br = w_ptr[k * w_plane_stride + j * w_stride + i];
    const double w_fl = w_ptr[(k+1) * w_plane_stride + j * w_stride + (i-1)];
    const double w_fr = w_ptr[(k+1) * w_plane_stride + j * w_stride + i];
    const double ww = 0.25 * (w_bl + w_br + w_fl + w_fr);

    // X-direction skew
    const double adv_x = uu * (u_ip1 - u_im1) / (2.0 * dx);
    const double uu_left = 0.5 * (u_im1 + uu);
    const double uu_right = 0.5 * (uu + u_ip1);
    const double cons_x = (uu_right * uu_right - uu_left * uu_left) / dx;
    const double skew_x = 0.5 * (adv_x + cons_x);

    // Y-direction skew
    const double adv_y = vv * (u_jp1 - u_jm1) / (2.0 * dy);
    const double vv_bot = 0.5 * (v_bl + v_br);
    const double vv_top = 0.5 * (v_tl + v_tr);
    const double u_bot = 0.5 * (uu + u_jm1);
    const double u_top = 0.5 * (uu + u_jp1);
    const double cons_y = (vv_top * u_top - vv_bot * u_bot) / dy;
    const double skew_y = 0.5 * (adv_y + cons_y);

    // Z-direction skew
    const double adv_z = ww * (u_kp1 - u_km1) / (2.0 * dz);
    const double ww_back = 0.5 * (w_bl + w_br);
    const double ww_front = 0.5 * (w_fl + w_fr);
    const double u_back = 0.5 * (uu + u_km1);
    const double u_front = 0.5 * (uu + u_kp1);
    const double cons_z = (ww_front * u_front - ww_back * u_back) / dz;
    const double skew_z = 0.5 * (adv_z + cons_z);

    const int conv_idx = k * conv_plane_stride + j * conv_stride + i;
    conv_u_ptr[conv_idx] = skew_x + skew_y + skew_z;
}

// 3D skew-symmetric convection for v-momentum at y-face
inline void convective_v_face_kernel_skew_3d(
    int i, int j, int k,
    int u_stride, int u_plane_stride,
    int v_stride, int v_plane_stride,
    int w_stride, int w_plane_stride,
    int conv_stride, int conv_plane_stride,
    double dx, double dy, double dz,
    const double* u_ptr, const double* v_ptr, const double* w_ptr,
    double* conv_v_ptr)
{
    const int v_idx = k * v_plane_stride + j * v_stride + i;
    const double vv = v_ptr[v_idx];

    // Neighbors
    const double v_ip1 = v_ptr[k * v_plane_stride + j * v_stride + (i+1)];
    const double v_im1 = v_ptr[k * v_plane_stride + j * v_stride + (i-1)];
    const double v_jp1 = v_ptr[k * v_plane_stride + (j+1) * v_stride + i];
    const double v_jm1 = v_ptr[k * v_plane_stride + (j-1) * v_stride + i];
    const double v_kp1 = v_ptr[(k+1) * v_plane_stride + j * v_stride + i];
    const double v_km1 = v_ptr[(k-1) * v_plane_stride + j * v_stride + i];

    // Interpolate u to y-face
    const double u_bl = u_ptr[k * u_plane_stride + (j-1) * u_stride + i];
    const double u_br = u_ptr[k * u_plane_stride + (j-1) * u_stride + (i+1)];
    const double u_tl = u_ptr[k * u_plane_stride + j * u_stride + i];
    const double u_tr = u_ptr[k * u_plane_stride + j * u_stride + (i+1)];
    const double uu = 0.25 * (u_bl + u_br + u_tl + u_tr);

    // Interpolate w to y-face
    const double w_bl = w_ptr[k * w_plane_stride + (j-1) * w_stride + i];
    const double w_tl = w_ptr[k * w_plane_stride + j * w_stride + i];
    const double w_bf = w_ptr[(k+1) * w_plane_stride + (j-1) * w_stride + i];
    const double w_tf = w_ptr[(k+1) * w_plane_stride + j * w_stride + i];
    const double ww = 0.25 * (w_bl + w_tl + w_bf + w_tf);

    // X-direction skew
    const double adv_x = uu * (v_ip1 - v_im1) / (2.0 * dx);
    const double uu_left = 0.5 * (u_bl + u_tl);
    const double uu_right = 0.5 * (u_br + u_tr);
    const double v_left = 0.5 * (v_im1 + vv);
    const double v_right = 0.5 * (vv + v_ip1);
    const double cons_x = (uu_right * v_right - uu_left * v_left) / dx;
    const double skew_x = 0.5 * (adv_x + cons_x);

    // Y-direction skew
    const double adv_y = vv * (v_jp1 - v_jm1) / (2.0 * dy);
    const double vv_bot = 0.5 * (v_jm1 + vv);
    const double vv_top = 0.5 * (vv + v_jp1);
    const double cons_y = (vv_top * vv_top - vv_bot * vv_bot) / dy;
    const double skew_y = 0.5 * (adv_y + cons_y);

    // Z-direction skew
    const double adv_z = ww * (v_kp1 - v_km1) / (2.0 * dz);
    const double ww_back = 0.5 * (w_bl + w_tl);
    const double ww_front = 0.5 * (w_bf + w_tf);
    const double v_back = 0.5 * (vv + v_km1);
    const double v_front = 0.5 * (vv + v_kp1);
    const double cons_z = (ww_front * v_front - ww_back * v_back) / dz;
    const double skew_z = 0.5 * (adv_z + cons_z);

    const int conv_idx = k * conv_plane_stride + j * conv_stride + i;
    conv_v_ptr[conv_idx] = skew_x + skew_y + skew_z;
}

// 3D skew-symmetric convection for w-momentum at z-face
inline void convective_w_face_kernel_skew_3d(
    int i, int j, int k,
    int u_stride, int u_plane_stride,
    int v_stride, int v_plane_stride,
    int w_stride, int w_plane_stride,
    int conv_stride, int conv_plane_stride,
    double dx, double dy, double dz,
    const double* u_ptr, const double* v_ptr, const double* w_ptr,
    double* conv_w_ptr)
{
    const int w_idx = k * w_plane_stride + j * w_stride + i;
    const double ww = w_ptr[w_idx];

    // Neighbors
    const double w_ip1 = w_ptr[k * w_plane_stride + j * w_stride + (i+1)];
    const double w_im1 = w_ptr[k * w_plane_stride + j * w_stride + (i-1)];
    const double w_jp1 = w_ptr[k * w_plane_stride + (j+1) * w_stride + i];
    const double w_jm1 = w_ptr[k * w_plane_stride + (j-1) * w_stride + i];
    const double w_kp1 = w_ptr[(k+1) * w_plane_stride + j * w_stride + i];
    const double w_km1 = w_ptr[(k-1) * w_plane_stride + j * w_stride + i];

    // Interpolate u to z-face
    const double u_bl = u_ptr[(k-1) * u_plane_stride + j * u_stride + i];
    const double u_br = u_ptr[(k-1) * u_plane_stride + j * u_stride + (i+1)];
    const double u_fl = u_ptr[k * u_plane_stride + j * u_stride + i];
    const double u_fr = u_ptr[k * u_plane_stride + j * u_stride + (i+1)];
    const double uu = 0.25 * (u_bl + u_br + u_fl + u_fr);

    // Interpolate v to z-face
    const double v_bl = v_ptr[(k-1) * v_plane_stride + j * v_stride + i];
    const double v_tl = v_ptr[(k-1) * v_plane_stride + (j+1) * v_stride + i];
    const double v_bf = v_ptr[k * v_plane_stride + j * v_stride + i];
    const double v_tf = v_ptr[k * v_plane_stride + (j+1) * v_stride + i];
    const double vv = 0.25 * (v_bl + v_tl + v_bf + v_tf);

    // X-direction skew
    const double adv_x = uu * (w_ip1 - w_im1) / (2.0 * dx);
    const double uu_left = 0.5 * (u_bl + u_fl);
    const double uu_right = 0.5 * (u_br + u_fr);
    const double w_left = 0.5 * (w_im1 + ww);
    const double w_right = 0.5 * (ww + w_ip1);
    const double cons_x = (uu_right * w_right - uu_left * w_left) / dx;
    const double skew_x = 0.5 * (adv_x + cons_x);

    // Y-direction skew
    const double adv_y = vv * (w_jp1 - w_jm1) / (2.0 * dy);
    const double vv_bot = 0.5 * (v_bl + v_bf);
    const double vv_top = 0.5 * (v_tl + v_tf);
    const double w_bot = 0.5 * (ww + w_jm1);
    const double w_top = 0.5 * (ww + w_jp1);
    const double cons_y = (vv_top * w_top - vv_bot * w_bot) / dy;
    const double skew_y = 0.5 * (adv_y + cons_y);

    // Z-direction skew
    const double adv_z = ww * (w_kp1 - w_km1) / (2.0 * dz);
    const double ww_back = 0.5 * (w_km1 + ww);
    const double ww_front = 0.5 * (ww + w_kp1);
    const double cons_z = (ww_front * ww_front - ww_back * ww_back) / dz;
    const double skew_z = 0.5 * (adv_z + cons_z);

    const int conv_idx = k * conv_plane_stride + j * conv_stride + i;
    conv_w_ptr[conv_idx] = skew_x + skew_y + skew_z;
}

// ============================================================================
// O4 SKEW-SYMMETRIC ADVECTION KERNELS - 4th order with boundary fallback
// ============================================================================
// Uses O4 same-stagger derivatives for the advective part where safe,
// falls back to O2 near boundaries.

// 3D O4 skew-symmetric convection for u-momentum at x-face
inline void convective_u_face_kernel_skew_O4_3d(
    int i, int j, int k,
    int u_stride, int u_plane_stride,
    int v_stride, int v_plane_stride,
    int w_stride, int w_plane_stride,
    int conv_stride, int conv_plane_stride,
    double dx, double dy, double dz,
    int Ng, int Nx, int Ny, int Nz,
    bool periodic_x, bool periodic_y, bool periodic_z,
    const double* u_ptr, const double* v_ptr, const double* w_ptr,
    double* conv_u_ptr)
{
    const int u_idx = k * u_plane_stride + j * u_stride + i;
    const double uu = u_ptr[u_idx];

    // Neighbors for O2 (always needed for conservative)
    const double u_ip1 = u_ptr[k * u_plane_stride + j * u_stride + (i+1)];
    const double u_im1 = u_ptr[k * u_plane_stride + j * u_stride + (i-1)];
    const double u_jp1 = u_ptr[k * u_plane_stride + (j+1) * u_stride + i];
    const double u_jm1 = u_ptr[k * u_plane_stride + (j-1) * u_stride + i];
    const double u_kp1 = u_ptr[(k+1) * u_plane_stride + j * u_stride + i];
    const double u_km1 = u_ptr[(k-1) * u_plane_stride + j * u_stride + i];

    // Interpolate v to x-face (O2: 4-point average)
    const double v_bl = v_ptr[k * v_plane_stride + j * v_stride + (i-1)];
    const double v_br = v_ptr[k * v_plane_stride + j * v_stride + i];
    const double v_tl = v_ptr[k * v_plane_stride + (j+1) * v_stride + (i-1)];
    const double v_tr = v_ptr[k * v_plane_stride + (j+1) * v_stride + i];
    const double vv = 0.25 * (v_bl + v_br + v_tl + v_tr);

    // Interpolate w to x-face (O2: 4-point average)
    const double w_bl = w_ptr[k * w_plane_stride + j * w_stride + (i-1)];
    const double w_br = w_ptr[k * w_plane_stride + j * w_stride + i];
    const double w_fl = w_ptr[(k+1) * w_plane_stride + j * w_stride + (i-1)];
    const double w_fr = w_ptr[(k+1) * w_plane_stride + j * w_stride + i];
    const double ww = 0.25 * (w_bl + w_br + w_fl + w_fr);

    // Check boundary safety for O4 in each direction (periodic-aware)
    const bool safe_x = periodic_x || ((i >= Ng + 2) && (i <= Ng + Nx - 2));
    const bool safe_y = periodic_y || ((j >= Ng + 2) && (j <= Ng + Ny - 3));
    const bool safe_z = periodic_z || ((k >= Ng + 2) && (k <= Ng + Nz - 3));

    // X-direction skew = 0.5*(advective + conservative)
    double adv_x;
    if (safe_x) {
        const double u_im2 = u_ptr[k * u_plane_stride + j * u_stride + (i-2)];
        const double u_ip2 = u_ptr[k * u_plane_stride + j * u_stride + (i+2)];
        adv_x = uu * D_same_O4(u_im2, u_im1, u_ip1, u_ip2, dx);
    } else {
        adv_x = uu * D_same_O2(u_im1, u_ip1, dx);
    }
    // Conservative uses O2 flux reconstruction (inherently O2)
    const double uu_left = 0.5 * (u_im1 + uu);
    const double uu_right = 0.5 * (uu + u_ip1);
    const double cons_x = (uu_right * uu_right - uu_left * uu_left) / dx;
    const double skew_x = 0.5 * (adv_x + cons_x);

    // Y-direction skew
    double adv_y;
    if (safe_y) {
        const double u_jm2 = u_ptr[k * u_plane_stride + (j-2) * u_stride + i];
        const double u_jp2 = u_ptr[k * u_plane_stride + (j+2) * u_stride + i];
        adv_y = vv * D_same_O4(u_jm2, u_jm1, u_jp1, u_jp2, dy);
    } else {
        adv_y = vv * D_same_O2(u_jm1, u_jp1, dy);
    }
    const double vv_bot = 0.5 * (v_bl + v_br);
    const double vv_top = 0.5 * (v_tl + v_tr);
    const double u_bot = 0.5 * (uu + u_jm1);
    const double u_top = 0.5 * (uu + u_jp1);
    const double cons_y = (vv_top * u_top - vv_bot * u_bot) / dy;
    const double skew_y = 0.5 * (adv_y + cons_y);

    // Z-direction skew
    double adv_z;
    if (safe_z) {
        const double u_km2 = u_ptr[(k-2) * u_plane_stride + j * u_stride + i];
        const double u_kp2 = u_ptr[(k+2) * u_plane_stride + j * u_stride + i];
        adv_z = ww * D_same_O4(u_km2, u_km1, u_kp1, u_kp2, dz);
    } else {
        adv_z = ww * D_same_O2(u_km1, u_kp1, dz);
    }
    const double ww_back = 0.5 * (w_bl + w_br);
    const double ww_front = 0.5 * (w_fl + w_fr);
    const double u_back = 0.5 * (uu + u_km1);
    const double u_front = 0.5 * (uu + u_kp1);
    const double cons_z = (ww_front * u_front - ww_back * u_back) / dz;
    const double skew_z = 0.5 * (adv_z + cons_z);

    const int conv_idx = k * conv_plane_stride + j * conv_stride + i;
    conv_u_ptr[conv_idx] = skew_x + skew_y + skew_z;
}

// 3D O4 skew-symmetric convection for v-momentum at y-face
inline void convective_v_face_kernel_skew_O4_3d(
    int i, int j, int k,
    int u_stride, int u_plane_stride,
    int v_stride, int v_plane_stride,
    int w_stride, int w_plane_stride,
    int conv_stride, int conv_plane_stride,
    double dx, double dy, double dz,
    int Ng, int Nx, int Ny, int Nz,
    bool periodic_x, bool periodic_y, bool periodic_z,
    const double* u_ptr, const double* v_ptr, const double* w_ptr,
    double* conv_v_ptr)
{
    const int v_idx = k * v_plane_stride + j * v_stride + i;
    const double vv = v_ptr[v_idx];

    // Neighbors
    const double v_ip1 = v_ptr[k * v_plane_stride + j * v_stride + (i+1)];
    const double v_im1 = v_ptr[k * v_plane_stride + j * v_stride + (i-1)];
    const double v_jp1 = v_ptr[k * v_plane_stride + (j+1) * v_stride + i];
    const double v_jm1 = v_ptr[k * v_plane_stride + (j-1) * v_stride + i];
    const double v_kp1 = v_ptr[(k+1) * v_plane_stride + j * v_stride + i];
    const double v_km1 = v_ptr[(k-1) * v_plane_stride + j * v_stride + i];

    // Interpolate u to y-face
    const double u_bl = u_ptr[k * u_plane_stride + (j-1) * u_stride + i];
    const double u_br = u_ptr[k * u_plane_stride + (j-1) * u_stride + (i+1)];
    const double u_tl = u_ptr[k * u_plane_stride + j * u_stride + i];
    const double u_tr = u_ptr[k * u_plane_stride + j * u_stride + (i+1)];
    const double uu = 0.25 * (u_bl + u_br + u_tl + u_tr);

    // Interpolate w to y-face
    const double w_bl = w_ptr[k * w_plane_stride + (j-1) * w_stride + i];
    const double w_tl = w_ptr[k * w_plane_stride + j * w_stride + i];
    const double w_bf = w_ptr[(k+1) * w_plane_stride + (j-1) * w_stride + i];
    const double w_tf = w_ptr[(k+1) * w_plane_stride + j * w_stride + i];
    const double ww = 0.25 * (w_bl + w_tl + w_bf + w_tf);

    // Check boundary safety for O4 (periodic directions always safe)
    const bool safe_x = periodic_x || ((i >= Ng + 2) && (i <= Ng + Nx - 3));
    const bool safe_y = periodic_y || ((j >= Ng + 2) && (j <= Ng + Ny - 2));
    const bool safe_z = periodic_z || ((k >= Ng + 2) && (k <= Ng + Nz - 3));

    // X-direction skew
    double adv_x;
    if (safe_x) {
        const double v_im2 = v_ptr[k * v_plane_stride + j * v_stride + (i-2)];
        const double v_ip2 = v_ptr[k * v_plane_stride + j * v_stride + (i+2)];
        adv_x = uu * D_same_O4(v_im2, v_im1, v_ip1, v_ip2, dx);
    } else {
        adv_x = uu * D_same_O2(v_im1, v_ip1, dx);
    }
    const double uu_left = 0.5 * (u_bl + u_tl);
    const double uu_right = 0.5 * (u_br + u_tr);
    const double v_left = 0.5 * (v_im1 + vv);
    const double v_right = 0.5 * (vv + v_ip1);
    const double cons_x = (uu_right * v_right - uu_left * v_left) / dx;
    const double skew_x = 0.5 * (adv_x + cons_x);

    // Y-direction skew
    double adv_y;
    if (safe_y) {
        const double v_jm2 = v_ptr[k * v_plane_stride + (j-2) * v_stride + i];
        const double v_jp2 = v_ptr[k * v_plane_stride + (j+2) * v_stride + i];
        adv_y = vv * D_same_O4(v_jm2, v_jm1, v_jp1, v_jp2, dy);
    } else {
        adv_y = vv * D_same_O2(v_jm1, v_jp1, dy);
    }
    const double vv_bot = 0.5 * (v_jm1 + vv);
    const double vv_top = 0.5 * (vv + v_jp1);
    const double cons_y = (vv_top * vv_top - vv_bot * vv_bot) / dy;
    const double skew_y = 0.5 * (adv_y + cons_y);

    // Z-direction skew
    double adv_z;
    if (safe_z) {
        const double v_km2 = v_ptr[(k-2) * v_plane_stride + j * v_stride + i];
        const double v_kp2 = v_ptr[(k+2) * v_plane_stride + j * v_stride + i];
        adv_z = ww * D_same_O4(v_km2, v_km1, v_kp1, v_kp2, dz);
    } else {
        adv_z = ww * D_same_O2(v_km1, v_kp1, dz);
    }
    const double ww_back = 0.5 * (w_bl + w_tl);
    const double ww_front = 0.5 * (w_bf + w_tf);
    const double v_back = 0.5 * (vv + v_km1);
    const double v_front = 0.5 * (vv + v_kp1);
    const double cons_z = (ww_front * v_front - ww_back * v_back) / dz;
    const double skew_z = 0.5 * (adv_z + cons_z);

    const int conv_idx = k * conv_plane_stride + j * conv_stride + i;
    conv_v_ptr[conv_idx] = skew_x + skew_y + skew_z;
}

// 3D O4 skew-symmetric convection for w-momentum at z-face
inline void convective_w_face_kernel_skew_O4_3d(
    int i, int j, int k,
    int u_stride, int u_plane_stride,
    int v_stride, int v_plane_stride,
    int w_stride, int w_plane_stride,
    int conv_stride, int conv_plane_stride,
    double dx, double dy, double dz,
    int Ng, int Nx, int Ny, int Nz,
    bool periodic_x, bool periodic_y, bool periodic_z,
    const double* u_ptr, const double* v_ptr, const double* w_ptr,
    double* conv_w_ptr)
{
    const int w_idx = k * w_plane_stride + j * w_stride + i;
    const double ww = w_ptr[w_idx];

    // Neighbors
    const double w_ip1 = w_ptr[k * w_plane_stride + j * w_stride + (i+1)];
    const double w_im1 = w_ptr[k * w_plane_stride + j * w_stride + (i-1)];
    const double w_jp1 = w_ptr[k * w_plane_stride + (j+1) * w_stride + i];
    const double w_jm1 = w_ptr[k * w_plane_stride + (j-1) * w_stride + i];
    const double w_kp1 = w_ptr[(k+1) * w_plane_stride + j * w_stride + i];
    const double w_km1 = w_ptr[(k-1) * w_plane_stride + j * w_stride + i];

    // Interpolate u to z-face
    const double u_bl = u_ptr[(k-1) * u_plane_stride + j * u_stride + i];
    const double u_br = u_ptr[(k-1) * u_plane_stride + j * u_stride + (i+1)];
    const double u_fl = u_ptr[k * u_plane_stride + j * u_stride + i];
    const double u_fr = u_ptr[k * u_plane_stride + j * u_stride + (i+1)];
    const double uu = 0.25 * (u_bl + u_br + u_fl + u_fr);

    // Interpolate v to z-face
    const double v_bl = v_ptr[(k-1) * v_plane_stride + j * v_stride + i];
    const double v_tl = v_ptr[(k-1) * v_plane_stride + (j+1) * v_stride + i];
    const double v_bf = v_ptr[k * v_plane_stride + j * v_stride + i];
    const double v_tf = v_ptr[k * v_plane_stride + (j+1) * v_stride + i];
    const double vv = 0.25 * (v_bl + v_tl + v_bf + v_tf);

    // Check boundary safety for O4 (periodic directions always safe)
    const bool safe_x = periodic_x || ((i >= Ng + 2) && (i <= Ng + Nx - 3));
    const bool safe_y = periodic_y || ((j >= Ng + 2) && (j <= Ng + Ny - 3));
    const bool safe_z = periodic_z || ((k >= Ng + 2) && (k <= Ng + Nz - 2));

    // X-direction skew
    double adv_x;
    if (safe_x) {
        const double w_im2 = w_ptr[k * w_plane_stride + j * w_stride + (i-2)];
        const double w_ip2 = w_ptr[k * w_plane_stride + j * w_stride + (i+2)];
        adv_x = uu * D_same_O4(w_im2, w_im1, w_ip1, w_ip2, dx);
    } else {
        adv_x = uu * D_same_O2(w_im1, w_ip1, dx);
    }
    const double uu_left = 0.5 * (u_bl + u_fl);
    const double uu_right = 0.5 * (u_br + u_fr);
    const double w_left = 0.5 * (w_im1 + ww);
    const double w_right = 0.5 * (ww + w_ip1);
    const double cons_x = (uu_right * w_right - uu_left * w_left) / dx;
    const double skew_x = 0.5 * (adv_x + cons_x);

    // Y-direction skew
    double adv_y;
    if (safe_y) {
        const double w_jm2 = w_ptr[k * w_plane_stride + (j-2) * w_stride + i];
        const double w_jp2 = w_ptr[k * w_plane_stride + (j+2) * w_stride + i];
        adv_y = vv * D_same_O4(w_jm2, w_jm1, w_jp1, w_jp2, dy);
    } else {
        adv_y = vv * D_same_O2(w_jm1, w_jp1, dy);
    }
    const double vv_bot = 0.5 * (v_bl + v_bf);
    const double vv_top = 0.5 * (v_tl + v_tf);
    const double w_bot = 0.5 * (ww + w_jm1);
    const double w_top = 0.5 * (ww + w_jp1);
    const double cons_y = (vv_top * w_top - vv_bot * w_bot) / dy;
    const double skew_y = 0.5 * (adv_y + cons_y);

    // Z-direction skew
    double adv_z;
    if (safe_z) {
        const double w_km2 = w_ptr[(k-2) * w_plane_stride + j * w_stride + i];
        const double w_kp2 = w_ptr[(k+2) * w_plane_stride + j * w_stride + i];
        adv_z = ww * D_same_O4(w_km2, w_km1, w_kp1, w_kp2, dz);
    } else {
        adv_z = ww * D_same_O2(w_km1, w_kp1, dz);
    }
    const double ww_back = 0.5 * (w_km1 + ww);
    const double ww_front = 0.5 * (ww + w_kp1);
    const double cons_z = (ww_front * ww_front - ww_back * ww_back) / dz;
    const double skew_z = 0.5 * (adv_z + cons_z);

    const int conv_idx = k * conv_plane_stride + j * conv_stride + i;
    conv_w_ptr[conv_idx] = skew_x + skew_y + skew_z;
}

// ============================================================================
// 2ND-ORDER UPWIND ADVECTION KERNELS
// ============================================================================
// Uses a linear reconstruction: φ_{i+1/2} = φ_i + 0.5*slope_i (for positive velocity)
// Slope limited for robustness (using minmod here for simplicity)

inline double minmod(double a, double b) {
    if (a * b <= 0.0) return 0.0;
    return (a > 0.0) ? ((a < b) ? a : b) : ((a > b) ? a : b);
}

// 2D 2nd-order upwind convection for u-momentum at x-face
inline void convective_u_face_kernel_upwind2_2d(
    int i, int j,
    int u_stride, int v_stride, int conv_stride,
    double dx, double dy,
    const double* u_ptr, const double* v_ptr,
    double* conv_u_ptr)
{
    const int u_idx = j * u_stride + i;
    const double uu = u_ptr[u_idx];

    // Neighbors
    const double u_ip2 = u_ptr[j * u_stride + (i+2)];
    const double u_ip1 = u_ptr[j * u_stride + (i+1)];
    const double u_im1 = u_ptr[j * u_stride + (i-1)];
    const double u_im2 = u_ptr[j * u_stride + (i-2)];
    const double u_jp2 = u_ptr[(j+2) * u_stride + i];
    const double u_jp1 = u_ptr[(j+1) * u_stride + i];
    const double u_jm1 = u_ptr[(j-1) * u_stride + i];
    const double u_jm2 = u_ptr[(j-2) * u_stride + i];

    // Interpolate v to x-face
    const double v_bl = v_ptr[j * v_stride + (i-1)];
    const double v_br = v_ptr[j * v_stride + i];
    const double v_tl = v_ptr[(j+1) * v_stride + (i-1)];
    const double v_tr = v_ptr[(j+1) * v_stride + i];
    const double vv = 0.25 * (v_bl + v_br + v_tl + v_tr);

    double dudx, dudy;

    // X-direction: 2nd-order upwind with minmod limiter
    if (uu >= 0) {
        // Use values from left: i-2, i-1, i
        double slope = minmod(uu - u_im1, u_im1 - u_im2);
        dudx = (uu - u_im1 + 0.5 * slope) / dx;
    } else {
        // Use values from right: i, i+1, i+2
        double slope = minmod(u_ip1 - uu, u_ip2 - u_ip1);
        dudx = (u_ip1 - uu - 0.5 * slope) / dx;
    }

    // Y-direction: 2nd-order upwind with minmod limiter
    if (vv >= 0) {
        double slope = minmod(uu - u_jm1, u_jm1 - u_jm2);
        dudy = (uu - u_jm1 + 0.5 * slope) / dy;
    } else {
        double slope = minmod(u_jp1 - uu, u_jp2 - u_jp1);
        dudy = (u_jp1 - uu - 0.5 * slope) / dy;
    }

    const int conv_idx = j * conv_stride + i;
    conv_u_ptr[conv_idx] = uu * dudx + vv * dudy;
}

// 2D 2nd-order upwind convection for v-momentum at y-face
inline void convective_v_face_kernel_upwind2_2d(
    int i, int j,
    int u_stride, int v_stride, int conv_stride,
    double dx, double dy,
    const double* u_ptr, const double* v_ptr,
    double* conv_v_ptr)
{
    const int v_idx = j * v_stride + i;
    const double vv = v_ptr[v_idx];

    // Neighbors
    const double v_ip2 = v_ptr[j * v_stride + (i+2)];
    const double v_ip1 = v_ptr[j * v_stride + (i+1)];
    const double v_im1 = v_ptr[j * v_stride + (i-1)];
    const double v_im2 = v_ptr[j * v_stride + (i-2)];
    const double v_jp2 = v_ptr[(j+2) * v_stride + i];
    const double v_jp1 = v_ptr[(j+1) * v_stride + i];
    const double v_jm1 = v_ptr[(j-1) * v_stride + i];
    const double v_jm2 = v_ptr[(j-2) * v_stride + i];

    // Interpolate u to y-face
    const double u_bl = u_ptr[(j-1) * u_stride + i];
    const double u_br = u_ptr[(j-1) * u_stride + (i+1)];
    const double u_tl = u_ptr[j * u_stride + i];
    const double u_tr = u_ptr[j * u_stride + (i+1)];
    const double uu = 0.25 * (u_bl + u_br + u_tl + u_tr);

    double dvdx, dvdy;

    // X-direction
    if (uu >= 0) {
        double slope = minmod(vv - v_im1, v_im1 - v_im2);
        dvdx = (vv - v_im1 + 0.5 * slope) / dx;
    } else {
        double slope = minmod(v_ip1 - vv, v_ip2 - v_ip1);
        dvdx = (v_ip1 - vv - 0.5 * slope) / dx;
    }

    // Y-direction
    if (vv >= 0) {
        double slope = minmod(vv - v_jm1, v_jm1 - v_jm2);
        dvdy = (vv - v_jm1 + 0.5 * slope) / dy;
    } else {
        double slope = minmod(v_jp1 - vv, v_jp2 - v_jp1);
        dvdy = (v_jp1 - vv - 0.5 * slope) / dy;
    }

    const int conv_idx = j * conv_stride + i;
    conv_v_ptr[conv_idx] = uu * dvdx + vv * dvdy;
}

// 3D 2nd-order upwind - simplified version using 1st-order near boundaries
// Full 2nd-order requires Nghost >= 2
inline void convective_u_face_kernel_upwind2_3d(
    int i, int j, int k,
    int u_stride, int u_plane_stride,
    int v_stride, int v_plane_stride,
    int w_stride, int w_plane_stride,
    int conv_stride, int conv_plane_stride,
    double dx, double dy, double dz,
    const double* u_ptr, const double* v_ptr, const double* w_ptr,
    double* conv_u_ptr)
{
    const int u_idx = k * u_plane_stride + j * u_stride + i;
    const double uu = u_ptr[u_idx];

    // Neighbors for 2nd-order stencil
    const double u_ip2 = u_ptr[k * u_plane_stride + j * u_stride + (i+2)];
    const double u_ip1 = u_ptr[k * u_plane_stride + j * u_stride + (i+1)];
    const double u_im1 = u_ptr[k * u_plane_stride + j * u_stride + (i-1)];
    const double u_im2 = u_ptr[k * u_plane_stride + j * u_stride + (i-2)];
    const double u_jp2 = u_ptr[k * u_plane_stride + (j+2) * u_stride + i];
    const double u_jp1 = u_ptr[k * u_plane_stride + (j+1) * u_stride + i];
    const double u_jm1 = u_ptr[k * u_plane_stride + (j-1) * u_stride + i];
    const double u_jm2 = u_ptr[k * u_plane_stride + (j-2) * u_stride + i];
    const double u_kp2 = u_ptr[(k+2) * u_plane_stride + j * u_stride + i];
    const double u_kp1 = u_ptr[(k+1) * u_plane_stride + j * u_stride + i];
    const double u_km1 = u_ptr[(k-1) * u_plane_stride + j * u_stride + i];
    const double u_km2 = u_ptr[(k-2) * u_plane_stride + j * u_stride + i];

    // Interpolate v to x-face
    const double v_bl = v_ptr[k * v_plane_stride + j * v_stride + (i-1)];
    const double v_br = v_ptr[k * v_plane_stride + j * v_stride + i];
    const double v_tl = v_ptr[k * v_plane_stride + (j+1) * v_stride + (i-1)];
    const double v_tr = v_ptr[k * v_plane_stride + (j+1) * v_stride + i];
    const double vv = 0.25 * (v_bl + v_br + v_tl + v_tr);

    // Interpolate w to x-face
    const double w_bl = w_ptr[k * w_plane_stride + j * w_stride + (i-1)];
    const double w_br = w_ptr[k * w_plane_stride + j * w_stride + i];
    const double w_fl = w_ptr[(k+1) * w_plane_stride + j * w_stride + (i-1)];
    const double w_fr = w_ptr[(k+1) * w_plane_stride + j * w_stride + i];
    const double ww = 0.25 * (w_bl + w_br + w_fl + w_fr);

    double dudx, dudy, dudz;

    // X-direction with minmod limiter
    if (uu >= 0) {
        double slope = minmod(uu - u_im1, u_im1 - u_im2);
        dudx = (uu - u_im1 + 0.5 * slope) / dx;
    } else {
        double slope = minmod(u_ip1 - uu, u_ip2 - u_ip1);
        dudx = (u_ip1 - uu - 0.5 * slope) / dx;
    }

    // Y-direction
    if (vv >= 0) {
        double slope = minmod(uu - u_jm1, u_jm1 - u_jm2);
        dudy = (uu - u_jm1 + 0.5 * slope) / dy;
    } else {
        double slope = minmod(u_jp1 - uu, u_jp2 - u_jp1);
        dudy = (u_jp1 - uu - 0.5 * slope) / dy;
    }

    // Z-direction
    if (ww >= 0) {
        double slope = minmod(uu - u_km1, u_km1 - u_km2);
        dudz = (uu - u_km1 + 0.5 * slope) / dz;
    } else {
        double slope = minmod(u_kp1 - uu, u_kp2 - u_kp1);
        dudz = (u_kp1 - uu - 0.5 * slope) / dz;
    }

    const int conv_idx = k * conv_plane_stride + j * conv_stride + i;
    conv_u_ptr[conv_idx] = uu * dudx + vv * dudy + ww * dudz;
}

// 3D 2nd-order upwind for v-momentum (abbreviated - same pattern)
inline void convective_v_face_kernel_upwind2_3d(
    int i, int j, int k,
    int u_stride, int u_plane_stride,
    int v_stride, int v_plane_stride,
    int w_stride, int w_plane_stride,
    int conv_stride, int conv_plane_stride,
    double dx, double dy, double dz,
    const double* u_ptr, const double* v_ptr, const double* w_ptr,
    double* conv_v_ptr)
{
    const int v_idx = k * v_plane_stride + j * v_stride + i;
    const double vv = v_ptr[v_idx];

    // Neighbors
    const double v_ip2 = v_ptr[k * v_plane_stride + j * v_stride + (i+2)];
    const double v_ip1 = v_ptr[k * v_plane_stride + j * v_stride + (i+1)];
    const double v_im1 = v_ptr[k * v_plane_stride + j * v_stride + (i-1)];
    const double v_im2 = v_ptr[k * v_plane_stride + j * v_stride + (i-2)];
    const double v_jp2 = v_ptr[k * v_plane_stride + (j+2) * v_stride + i];
    const double v_jp1 = v_ptr[k * v_plane_stride + (j+1) * v_stride + i];
    const double v_jm1 = v_ptr[k * v_plane_stride + (j-1) * v_stride + i];
    const double v_jm2 = v_ptr[k * v_plane_stride + (j-2) * v_stride + i];
    const double v_kp2 = v_ptr[(k+2) * v_plane_stride + j * v_stride + i];
    const double v_kp1 = v_ptr[(k+1) * v_plane_stride + j * v_stride + i];
    const double v_km1 = v_ptr[(k-1) * v_plane_stride + j * v_stride + i];
    const double v_km2 = v_ptr[(k-2) * v_plane_stride + j * v_stride + i];

    // Interpolate u to y-face
    const double u_bl = u_ptr[k * u_plane_stride + (j-1) * u_stride + i];
    const double u_br = u_ptr[k * u_plane_stride + (j-1) * u_stride + (i+1)];
    const double u_tl = u_ptr[k * u_plane_stride + j * u_stride + i];
    const double u_tr = u_ptr[k * u_plane_stride + j * u_stride + (i+1)];
    const double uu = 0.25 * (u_bl + u_br + u_tl + u_tr);

    // Interpolate w to y-face
    const double w_bl = w_ptr[k * w_plane_stride + (j-1) * w_stride + i];
    const double w_tl = w_ptr[k * w_plane_stride + j * w_stride + i];
    const double w_bf = w_ptr[(k+1) * w_plane_stride + (j-1) * w_stride + i];
    const double w_tf = w_ptr[(k+1) * w_plane_stride + j * w_stride + i];
    const double ww = 0.25 * (w_bl + w_tl + w_bf + w_tf);

    double dvdx, dvdy, dvdz;

    if (uu >= 0) {
        double slope = minmod(vv - v_im1, v_im1 - v_im2);
        dvdx = (vv - v_im1 + 0.5 * slope) / dx;
    } else {
        double slope = minmod(v_ip1 - vv, v_ip2 - v_ip1);
        dvdx = (v_ip1 - vv - 0.5 * slope) / dx;
    }

    if (vv >= 0) {
        double slope = minmod(vv - v_jm1, v_jm1 - v_jm2);
        dvdy = (vv - v_jm1 + 0.5 * slope) / dy;
    } else {
        double slope = minmod(v_jp1 - vv, v_jp2 - v_jp1);
        dvdy = (v_jp1 - vv - 0.5 * slope) / dy;
    }

    if (ww >= 0) {
        double slope = minmod(vv - v_km1, v_km1 - v_km2);
        dvdz = (vv - v_km1 + 0.5 * slope) / dz;
    } else {
        double slope = minmod(v_kp1 - vv, v_kp2 - v_kp1);
        dvdz = (v_kp1 - vv - 0.5 * slope) / dz;
    }

    const int conv_idx = k * conv_plane_stride + j * conv_stride + i;
    conv_v_ptr[conv_idx] = uu * dvdx + vv * dvdy + ww * dvdz;
}

// 3D 2nd-order upwind for w-momentum
inline void convective_w_face_kernel_upwind2_3d(
    int i, int j, int k,
    int u_stride, int u_plane_stride,
    int v_stride, int v_plane_stride,
    int w_stride, int w_plane_stride,
    int conv_stride, int conv_plane_stride,
    double dx, double dy, double dz,
    const double* u_ptr, const double* v_ptr, const double* w_ptr,
    double* conv_w_ptr)
{
    const int w_idx = k * w_plane_stride + j * w_stride + i;
    const double ww = w_ptr[w_idx];

    // Neighbors
    const double w_ip2 = w_ptr[k * w_plane_stride + j * w_stride + (i+2)];
    const double w_ip1 = w_ptr[k * w_plane_stride + j * w_stride + (i+1)];
    const double w_im1 = w_ptr[k * w_plane_stride + j * w_stride + (i-1)];
    const double w_im2 = w_ptr[k * w_plane_stride + j * w_stride + (i-2)];
    const double w_jp2 = w_ptr[k * w_plane_stride + (j+2) * w_stride + i];
    const double w_jp1 = w_ptr[k * w_plane_stride + (j+1) * w_stride + i];
    const double w_jm1 = w_ptr[k * w_plane_stride + (j-1) * w_stride + i];
    const double w_jm2 = w_ptr[k * w_plane_stride + (j-2) * w_stride + i];
    const double w_kp2 = w_ptr[(k+2) * w_plane_stride + j * w_stride + i];
    const double w_kp1 = w_ptr[(k+1) * w_plane_stride + j * w_stride + i];
    const double w_km1 = w_ptr[(k-1) * w_plane_stride + j * w_stride + i];
    const double w_km2 = w_ptr[(k-2) * w_plane_stride + j * w_stride + i];

    // Interpolate u to z-face
    const double u_bl = u_ptr[(k-1) * u_plane_stride + j * u_stride + i];
    const double u_br = u_ptr[(k-1) * u_plane_stride + j * u_stride + (i+1)];
    const double u_fl = u_ptr[k * u_plane_stride + j * u_stride + i];
    const double u_fr = u_ptr[k * u_plane_stride + j * u_stride + (i+1)];
    const double uu = 0.25 * (u_bl + u_br + u_fl + u_fr);

    // Interpolate v to z-face
    const double v_bl = v_ptr[(k-1) * v_plane_stride + j * v_stride + i];
    const double v_tl = v_ptr[(k-1) * v_plane_stride + (j+1) * v_stride + i];
    const double v_bf = v_ptr[k * v_plane_stride + j * v_stride + i];
    const double v_tf = v_ptr[k * v_plane_stride + (j+1) * v_stride + i];
    const double vv = 0.25 * (v_bl + v_tl + v_bf + v_tf);

    double dwdx, dwdy, dwdz;

    if (uu >= 0) {
        double slope = minmod(ww - w_im1, w_im1 - w_im2);
        dwdx = (ww - w_im1 + 0.5 * slope) / dx;
    } else {
        double slope = minmod(w_ip1 - ww, w_ip2 - w_ip1);
        dwdx = (w_ip1 - ww - 0.5 * slope) / dx;
    }

    if (vv >= 0) {
        double slope = minmod(ww - w_jm1, w_jm1 - w_jm2);
        dwdy = (ww - w_jm1 + 0.5 * slope) / dy;
    } else {
        double slope = minmod(w_jp1 - ww, w_jp2 - w_jp1);
        dwdy = (w_jp1 - ww - 0.5 * slope) / dy;
    }

    if (ww >= 0) {
        double slope = minmod(ww - w_km1, w_km1 - w_km2);
        dwdz = (ww - w_km1 + 0.5 * slope) / dz;
    } else {
        double slope = minmod(w_kp1 - ww, w_kp2 - w_kp1);
        dwdz = (w_kp1 - ww - 0.5 * slope) / dz;
    }

    const int conv_idx = k * conv_plane_stride + j * conv_stride + i;
    conv_w_ptr[conv_idx] = uu * dwdx + vv * dwdy + ww * dwdz;
}

// ============================================================================
// END ADVECTION KERNELS
// ============================================================================

// 3D Diffusion term for u-momentum at x-face
inline void diffusive_u_face_kernel_staggered_3d(
    int i, int j, int k,
    int u_stride, int u_plane_stride,
    int nu_stride, int nu_plane_stride,
    int diff_stride, int diff_plane_stride,
    double dx, double dy, double dz,
    const double* u_ptr, const double* nu_ptr,
    double* diff_u_ptr)
{
    const int u_idx = k * u_plane_stride + j * u_stride + i;
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double dz2 = dz * dz;

    // Viscosity at cell centers adjacent to x-face
    const double nu_left  = nu_ptr[k * nu_plane_stride + j * nu_stride + (i-1)];
    const double nu_right = nu_ptr[k * nu_plane_stride + j * nu_stride + i];
    const double nu_avg = 0.5 * (nu_left + nu_right);

    // d2u/dx2
    const double d2u_dx2 = nu_avg * (u_ptr[k * u_plane_stride + j * u_stride + (i+1)]
                                   - 2.0 * u_ptr[u_idx]
                                   + u_ptr[k * u_plane_stride + j * u_stride + (i-1)]) / dx2;

    // d2u/dy2
    const double d2u_dy2 = nu_avg * (u_ptr[k * u_plane_stride + (j+1) * u_stride + i]
                                   - 2.0 * u_ptr[u_idx]
                                   + u_ptr[k * u_plane_stride + (j-1) * u_stride + i]) / dy2;

    // d2u/dz2
    const double d2u_dz2 = nu_avg * (u_ptr[(k+1) * u_plane_stride + j * u_stride + i]
                                   - 2.0 * u_ptr[u_idx]
                                   + u_ptr[(k-1) * u_plane_stride + j * u_stride + i]) / dz2;

    const int diff_idx = k * diff_plane_stride + j * diff_stride + i;
    diff_u_ptr[diff_idx] = d2u_dx2 + d2u_dy2 + d2u_dz2;
}

// 3D Diffusion term for v-momentum at y-face
inline void diffusive_v_face_kernel_staggered_3d(
    int i, int j, int k,
    int v_stride, int v_plane_stride,
    int nu_stride, int nu_plane_stride,
    int diff_stride, int diff_plane_stride,
    double dx, double dy, double dz,
    const double* v_ptr, const double* nu_ptr,
    double* diff_v_ptr)
{
    const int v_idx = k * v_plane_stride + j * v_stride + i;
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double dz2 = dz * dz;

    // Viscosity at cell centers adjacent to y-face
    const double nu_bottom = nu_ptr[k * nu_plane_stride + (j-1) * nu_stride + i];
    const double nu_top    = nu_ptr[k * nu_plane_stride + j * nu_stride + i];
    const double nu_avg = 0.5 * (nu_bottom + nu_top);

    // d2v/dx2
    const double d2v_dx2 = nu_avg * (v_ptr[k * v_plane_stride + j * v_stride + (i+1)]
                                   - 2.0 * v_ptr[v_idx]
                                   + v_ptr[k * v_plane_stride + j * v_stride + (i-1)]) / dx2;

    // d2v/dy2
    const double d2v_dy2 = nu_avg * (v_ptr[k * v_plane_stride + (j+1) * v_stride + i]
                                   - 2.0 * v_ptr[v_idx]
                                   + v_ptr[k * v_plane_stride + (j-1) * v_stride + i]) / dy2;

    // d2v/dz2
    const double d2v_dz2 = nu_avg * (v_ptr[(k+1) * v_plane_stride + j * v_stride + i]
                                   - 2.0 * v_ptr[v_idx]
                                   + v_ptr[(k-1) * v_plane_stride + j * v_stride + i]) / dz2;

    const int diff_idx = k * diff_plane_stride + j * diff_stride + i;
    diff_v_ptr[diff_idx] = d2v_dx2 + d2v_dy2 + d2v_dz2;
}

// 3D Diffusion term for w-momentum at z-face
inline void diffusive_w_face_kernel_staggered_3d(
    int i, int j, int k,
    int w_stride, int w_plane_stride,
    int nu_stride, int nu_plane_stride,
    int diff_stride, int diff_plane_stride,
    double dx, double dy, double dz,
    const double* w_ptr, const double* nu_ptr,
    double* diff_w_ptr)
{
    const int w_idx = k * w_plane_stride + j * w_stride + i;
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double dz2 = dz * dz;

    // Viscosity at cell centers adjacent to z-face
    const double nu_back  = nu_ptr[(k-1) * nu_plane_stride + j * nu_stride + i];
    const double nu_front = nu_ptr[k * nu_plane_stride + j * nu_stride + i];
    const double nu_avg = 0.5 * (nu_back + nu_front);

    // d2w/dx2
    const double d2w_dx2 = nu_avg * (w_ptr[k * w_plane_stride + j * w_stride + (i+1)]
                                   - 2.0 * w_ptr[w_idx]
                                   + w_ptr[k * w_plane_stride + j * w_stride + (i-1)]) / dx2;

    // d2w/dy2
    const double d2w_dy2 = nu_avg * (w_ptr[k * w_plane_stride + (j+1) * w_stride + i]
                                   - 2.0 * w_ptr[w_idx]
                                   + w_ptr[k * w_plane_stride + (j-1) * w_stride + i]) / dy2;

    // d2w/dz2
    const double d2w_dz2 = nu_avg * (w_ptr[(k+1) * w_plane_stride + j * w_stride + i]
                                   - 2.0 * w_ptr[w_idx]
                                   + w_ptr[(k-1) * w_plane_stride + j * w_stride + i]) / dz2;

    const int diff_idx = k * diff_plane_stride + j * diff_stride + i;
    diff_w_ptr[diff_idx] = d2w_dx2 + d2w_dy2 + d2w_dz2;
}

#pragma omp end declare target
// ============================================================================
// END 3D OPERATOR KERNELS
// ============================================================================

// Poisson boundary condition kernel for x-direction
inline void apply_poisson_bc_x_cell(
    int j, int g,
    int Nx, int Ng, int stride,
    int bc_x_lo, int bc_x_hi,  // 0=Dirichlet, 1=Neumann, 2=Periodic
    double dirichlet_val,
    double* p_ptr)
{
    // Left boundary
    int i_ghost = g;
    int i_interior = Ng;
    int i_periodic = Nx + Ng - 1 - g;
    
    int idx_ghost = j * stride + i_ghost;
    int idx_interior = j * stride + i_interior;
    int idx_periodic = j * stride + i_periodic;
    
    if (bc_x_lo == 2) {  // Periodic
        p_ptr[idx_ghost] = p_ptr[idx_periodic];
    } else if (bc_x_lo == 1) {  // Neumann
        p_ptr[idx_ghost] = p_ptr[idx_interior];
    } else {  // Dirichlet
        p_ptr[idx_ghost] = 2.0 * dirichlet_val - p_ptr[idx_interior];
    }
    
    // Right boundary
    i_ghost = Nx + Ng + g;
    i_interior = Nx + Ng - 1;
    i_periodic = Ng + g;
    
    idx_ghost = j * stride + i_ghost;
    idx_interior = j * stride + i_interior;
    idx_periodic = j * stride + i_periodic;
    
    if (bc_x_hi == 2) {  // Periodic
        p_ptr[idx_ghost] = p_ptr[idx_periodic];
    } else if (bc_x_hi == 1) {  // Neumann
        p_ptr[idx_ghost] = p_ptr[idx_interior];
    } else {  // Dirichlet
        p_ptr[idx_ghost] = 2.0 * dirichlet_val - p_ptr[idx_interior];
    }
}

// Poisson boundary condition kernel for y-direction
inline void apply_poisson_bc_y_cell(
    int i, int g,
    int Ny, int Ng, int stride,
    int bc_y_lo, int bc_y_hi,  // 0=Dirichlet, 1=Neumann, 2=Periodic
    double dirichlet_val,
    double* p_ptr)
{
    // Bottom boundary
    int j_ghost = g;
    int j_interior = Ng;
    int j_periodic = Ny + Ng - 1 - g;
    
    int idx_ghost = j_ghost * stride + i;
    int idx_interior = j_interior * stride + i;
    int idx_periodic = j_periodic * stride + i;
    
    if (bc_y_lo == 2) {  // Periodic
        p_ptr[idx_ghost] = p_ptr[idx_periodic];
    } else if (bc_y_lo == 1) {  // Neumann
        p_ptr[idx_ghost] = p_ptr[idx_interior];
    } else {  // Dirichlet
        p_ptr[idx_ghost] = 2.0 * dirichlet_val - p_ptr[idx_interior];
    }
    
    // Top boundary
    j_ghost = Ny + Ng + g;
    j_interior = Ny + Ng - 1;
    j_periodic = Ng + g;
    
    idx_ghost = j_ghost * stride + i;
    idx_interior = j_interior * stride + i;
    idx_periodic = j_periodic * stride + i;
    
    if (bc_y_hi == 2) {  // Periodic
        p_ptr[idx_ghost] = p_ptr[idx_periodic];
    } else if (bc_y_hi == 1) {  // Neumann
        p_ptr[idx_ghost] = p_ptr[idx_interior];
    } else {  // Dirichlet
        p_ptr[idx_ghost] = 2.0 * dirichlet_val - p_ptr[idx_interior];
    }
}

// Red-black SOR Poisson iteration kernel for a single cell
inline void poisson_sor_cell_kernel(
    int cell_idx, int stride,
    double dx2, double dy2, double omega,
    const double* rhs_ptr,
    double* p_ptr)
{
    const double coeff = 2.0 / dx2 + 2.0 / dy2;
    
    double p_old = p_ptr[cell_idx];
    double p_gs = ((p_ptr[cell_idx+1] + p_ptr[cell_idx-1]) / dx2 +
                   (p_ptr[cell_idx+stride] + p_ptr[cell_idx-stride]) / dy2
                   - rhs_ptr[cell_idx]) / coeff;
    p_ptr[cell_idx] = (1.0 - omega) * p_old + omega * p_gs;
}

// Poisson residual kernel for a single cell
inline double poisson_residual_cell_kernel(
    int cell_idx, int stride,
    double dx2, double dy2,
    const double* rhs_ptr,
    const double* p_ptr)
{
    double laplacian = (p_ptr[cell_idx+1] - 2.0*p_ptr[cell_idx] + p_ptr[cell_idx-1]) / dx2
                     + (p_ptr[cell_idx+stride] - 2.0*p_ptr[cell_idx] + p_ptr[cell_idx-stride]) / dy2;
    double res = laplacian - rhs_ptr[cell_idx];
    return (res < 0.0) ? -res : res;  // abs
}

#ifdef USE_GPU_OFFLOAD
#pragma omp end declare target
#endif

}  // namespace kernels
}  // namespace nncfd

#endif  // NNCFD_SOLVER_KERNELS_HPP
