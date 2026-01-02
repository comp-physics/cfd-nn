/// @file solver.cpp
/// @brief Implementation of incompressible RANS solver with fractional-step projection method
///
/// This file implements the RANSSolver class, which solves the Reynolds-Averaged Navier-Stokes
/// equations using a fractional-step projection method. Key features:
/// - Fractional-step time integration (explicit Euler + pressure projection)
/// - Multigrid Poisson solver for pressure correction
/// - Staggered MAC grid discretization (2nd-order central differences)
/// - GPU acceleration via OpenMP target offload
/// - Support for multiple turbulence models (algebraic, transport, EARSM, neural networks)
///
/// The implementation includes unified CPU/GPU kernels that compile for both host and device,
/// ensuring numerical consistency between platforms.

#include "solver.hpp"
#include "timing.hpp"
#include "gpu_utils.hpp"
#include "profiling.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <cassert>
#include <cstring>

#ifdef GPU_PROFILE_TRANSFERS
#include <chrono>
#endif

// Legacy NVTX macros for backward compatibility
// These are kept for existing code; new code should use profiling.hpp macros
#ifdef GPU_PROFILE_KERNELS
#if __has_include(<nvtx3/nvToolsExt.h>)
    #include <nvtx3/nvToolsExt.h>
    #define NVTX_PUSH(name) nvtxRangePushA(name)
    #define NVTX_POP() nvtxRangePop()
#elif __has_include(<nvToolsExt.h>)
    #include <nvToolsExt.h>
    #define NVTX_PUSH(name) nvtxRangePushA(name)
    #define NVTX_POP() nvtxRangePop()
#else
    #define NVTX_PUSH(name)
    #define NVTX_POP()
#endif
#else
#define NVTX_PUSH(name)
#define NVTX_POP()
#endif

namespace nncfd {

// ============================================================================
// Unified CPU/GPU kernels - single source of truth for numerical algorithms
// These kernels are compiled for both host and device when USE_GPU_OFFLOAD is on
// ============================================================================

#ifdef USE_GPU_OFFLOAD
#pragma omp declare target
#endif

// Staggered grid BC kernel for u-velocity (at x-faces) in x-direction
inline void apply_u_bc_x_staggered(
    int j, int g,
    int Nx, int Ng, int u_stride,
    bool x_lo_periodic, bool x_lo_noslip,
    bool x_hi_periodic, bool x_hi_noslip,
    double* u_ptr)
{
    // CRITICAL for staggered grid with periodic BCs:
    // The rightmost interior face (Ng+Nx) IS the leftmost interior face (Ng)
    // They represent the same physical location in a periodic domain
    if (x_lo_periodic && x_hi_periodic && g == 0) {
        // Enforce periodicity: right edge = left edge
        int i_left = Ng;
        int i_right = Ng + Nx;
        u_ptr[j * u_stride + i_right] = u_ptr[j * u_stride + i_left];
    }
    
    // Left boundary: u is the normal velocity at x-faces
    if (x_lo_noslip) {
        // No-penetration: u = 0 at wall face (i = Ng)
        // This is Dirichlet BC for the normal velocity
        if (g == 0) {
            u_ptr[j * u_stride + Ng] = 0.0;
        }
        // Set ghost faces to zero for stencil consistency
        int i_ghost = Ng - 1 - g;
        u_ptr[j * u_stride + i_ghost] = 0.0;
    } else if (x_lo_periodic) {
        int i_ghost = Ng - 1 - g;
        int i_periodic = Ng + Nx - 1 - g;
        u_ptr[j * u_stride + i_ghost] = u_ptr[j * u_stride + i_periodic];
    }
    
    // Right boundary
    if (x_hi_noslip) {
        // No-penetration: u = 0 at wall face (i = Ng + Nx)
        if (g == 0) {
            u_ptr[j * u_stride + (Ng + Nx)] = 0.0;
        }
        // Set ghost faces to zero for stencil consistency
        int i_ghost = Ng + Nx + 1 + g;
        u_ptr[j * u_stride + i_ghost] = 0.0;
    } else if (x_hi_periodic) {
        int i_ghost = Ng + Nx + 1 + g;  // Ghost on right (+1 because Ng+Nx is now interior)
        int i_periodic = Ng + 1 + g;     // Wrap from left interior
        u_ptr[j * u_stride + i_ghost] = u_ptr[j * u_stride + i_periodic];
    }
}

// Staggered grid BC kernel for u-velocity (at x-faces) in y-direction
inline void apply_u_bc_y_staggered(
    int i, int g,
    int Ny, int Ng, int u_stride,
    bool y_lo_periodic, bool y_lo_noslip,
    bool y_hi_periodic, bool y_hi_noslip,
    double* u_ptr)
{
    // Bottom boundary
    if (y_lo_noslip) {
        // No-slip: u at wall faces should be zero, enforce via ghost cells
        int j_ghost = Ng - 1 - g;
        int j_interior = Ng + g;
        u_ptr[j_ghost * u_stride + i] = -u_ptr[j_interior * u_stride + i];
    } else if (y_lo_periodic) {
        int j_ghost = Ng - 1 - g;
        int j_periodic = Ny + Ng - 1 - g;
        u_ptr[j_ghost * u_stride + i] = u_ptr[j_periodic * u_stride + i];
    }
    
    // Top boundary
    if (y_hi_noslip) {
        int j_ghost = Ny + Ng + g;
        int j_interior = Ny + Ng - 1 - g;
        u_ptr[j_ghost * u_stride + i] = -u_ptr[j_interior * u_stride + i];
    } else if (y_hi_periodic) {
        int j_ghost = Ny + Ng + g;
        int j_periodic = Ng + g;
        u_ptr[j_ghost * u_stride + i] = u_ptr[j_periodic * u_stride + i];
    }
}

// Staggered grid BC kernel for v-velocity (at y-faces) in x-direction
inline void apply_v_bc_x_staggered(
    int j, int g,
    int Nx, int Ng, int v_stride,
    bool x_lo_periodic, bool x_lo_noslip,
    bool x_hi_periodic, bool x_hi_noslip,
    double* v_ptr)
{
    // Left boundary
    if (x_lo_noslip) {
        // No-slip: v at wall faces should be zero
        int i_ghost = Ng - 1 - g;
        int i_interior = Ng + g;
        v_ptr[j * v_stride + i_ghost] = -v_ptr[j * v_stride + i_interior];
    } else if (x_lo_periodic) {
        int i_ghost = Ng - 1 - g;
        int i_periodic = Nx + Ng - 1 - g;
        v_ptr[j * v_stride + i_ghost] = v_ptr[j * v_stride + i_periodic];
    }
    
    // Right boundary
    if (x_hi_noslip) {
        int i_ghost = Nx + Ng + g;
        int i_interior = Nx + Ng - 1 - g;
        v_ptr[j * v_stride + i_ghost] = -v_ptr[j * v_stride + i_interior];
    } else if (x_hi_periodic) {
        int i_ghost = Nx + Ng + g;
        int i_periodic = Ng + g;
        v_ptr[j * v_stride + i_ghost] = v_ptr[j * v_stride + i_periodic];
    }
}

// Staggered grid BC kernel for v-velocity (at y-faces) in y-direction
inline void apply_v_bc_y_staggered(
    int i, int g,
    int Ny, int Ng, int v_stride,
    bool y_lo_periodic, bool y_lo_noslip,
    bool y_hi_periodic, bool y_hi_noslip,
    double* v_ptr)
{
    // CRITICAL for staggered grid with periodic BCs:
    // The topmost interior face (Ng+Ny) IS the bottommost interior face (Ng)
    // They represent the same physical location in a periodic domain
    if (y_lo_periodic && y_hi_periodic && g == 0) {
        // Enforce periodicity: top edge = bottom edge
        int j_bottom = Ng;
        int j_top = Ng + Ny;
        v_ptr[j_top * v_stride + i] = v_ptr[j_bottom * v_stride + i];
    }
    
    // No-slip/no-penetration at walls
    // v is stored AT the wall faces, so we set them to zero directly
    // Also set ghost cells to enforce BC in stencils
    if (y_lo_noslip) {
        // Bottom wall is at j = Ng (first y-face)
        if (g == 0) {
            v_ptr[Ng * v_stride + i] = 0.0;
        }
        // Ghost cells below wall
        int j_ghost = Ng - 1 - g;
        v_ptr[j_ghost * v_stride + i] = 0.0;  // Also zero for consistency
    } else if (y_lo_periodic) {
        int j_ghost = Ng - 1 - g;
        int j_periodic = Ng + Ny - 1 - g;
        v_ptr[j_ghost * v_stride + i] = v_ptr[j_periodic * v_stride + i];
    }
    
    if (y_hi_noslip) {
        // Top wall is at j = Ng + Ny (last y-face)
        if (g == 0) {
            v_ptr[(Ng + Ny) * v_stride + i] = 0.0;
        }
        // Ghost cells above wall
        int j_ghost = Ng + Ny + 1 + g;
        v_ptr[j_ghost * v_stride + i] = 0.0;  // Also zero for consistency
    } else if (y_hi_periodic) {
        int j_ghost = Ng + Ny + 1 + g;  // Ghost on top (+1 because Ng+Ny is now interior)
        int j_periodic = Ng + 1 + g;     // Wrap from bottom interior
        v_ptr[j_ghost * v_stride + i] = v_ptr[j_periodic * v_stride + i];
    }
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
    
    // Face-averaged viscosity for d2u/dx2 term
    const double nu_e = 0.5 * (nu_right + (i+1 < nu_stride ? nu_ptr[j * nu_stride + (i+1)] : nu_right));
    const double nu_w = 0.5 * (nu_left + (i-2 >= 0 ? nu_ptr[j * nu_stride + (i-2)] : nu_left));
    const double nu_n = 0.5 * (nu_left + nu_right);
    const double nu_s = 0.5 * (nu_left + nu_right);
    
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
    
    // Face-averaged viscosity
    const double nu_e = 0.5 * (nu_bottom + nu_top);
    const double nu_w = 0.5 * (nu_bottom + nu_top);
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

RANSSolver::RANSSolver(const Mesh& mesh, const Config& config)
    : mesh_(&mesh)
    , config_(config)
    , velocity_(mesh)
    , velocity_star_(mesh)
    , pressure_(mesh)
    , pressure_correction_(mesh)
    , nu_t_(mesh)
    , k_(mesh)
    , omega_(mesh)
    , tau_ij_(mesh)
    , rhs_poisson_(mesh)
    , div_velocity_(mesh)
    , nu_eff_(mesh, config.nu)   // Persistent effective viscosity field
    , conv_(mesh)                 // Persistent convective work field
    , diff_(mesh)                 // Persistent diffusive work field
    , velocity_old_(mesh)         // GPU-resident old velocity for residual
    , dudx_(mesh), dudy_(mesh), dvdx_(mesh), dvdy_(mesh)  // Gradient scratch for turbulence
    , wall_distance_(mesh)        // Precomputed wall distance field
    , poisson_solver_(mesh)
    , mg_poisson_solver_(mesh)
    , use_multigrid_(true)
    , current_dt_(config.dt)
{
    // Precompute wall distance (once, then stays on GPU if enabled)
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            wall_distance_(i, j) = mesh.wall_distance(i, j);
        }
    }
    // Set up Poisson solver BCs (periodic in x, Neumann in y for channel)
    poisson_solver_.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                           PoissonBC::Neumann, PoissonBC::Neumann);
    mg_poisson_solver_.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                              PoissonBC::Neumann, PoissonBC::Neumann);

#ifdef USE_HYPRE
    // Initialize HYPRE PFMG solver
    hypre_poisson_solver_ = std::make_unique<HyprePoissonSolver>(mesh);
    hypre_poisson_solver_->set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                                   PoissonBC::Neumann, PoissonBC::Neumann);
#endif

#ifdef USE_FFT_POISSON
    // Initialize FFT solvers for periodic cases
    // FFT (2D): requires periodic x AND z with uniform spacing
    // FFT1D: requires periodic x OR z (exactly one) with uniform spacing
    bool fft_applicable = false;
    bool fft1d_applicable = false;
    int fft1d_periodic_dir = 0;  // 0 = x periodic, 2 = z periodic

    // Check which FFT solver is applicable (actual BCs set later via set_velocity_bc)
    // For now, assume defaults: periodic x,z - will be updated in set_velocity_bc
    bool periodic_xz = true;  // Default for channel: periodic x/z
    bool uniform_xz = true;   // Default for channel: uniform x/z spacing

    if (!mesh.is2D()) {
        // Try 2D FFT first (periodic x AND z)
        if (periodic_xz && uniform_xz) {
            try {
                fft_poisson_solver_ = std::make_unique<FFTPoissonSolver>(mesh);
                fft_poisson_solver_->set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                                             PoissonBC::Neumann, PoissonBC::Neumann,
                                             PoissonBC::Periodic, PoissonBC::Periodic);
                fft_applicable = true;
            } catch (const std::exception& e) {
                std::cerr << "[Solver] FFT solver initialization failed: " << e.what() << "\n";
            }
        }

        // Also initialize 1D FFT solver (for cases like duct flow)
        // Will be used if 2D FFT becomes incompatible after BC update
        try {
            // Default to x-periodic (duct flow typical case)
            fft1d_poisson_solver_ = std::make_unique<FFT1DPoissonSolver>(mesh, 0);
            fft1d_poisson_solver_->set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                                           PoissonBC::Neumann, PoissonBC::Neumann,
                                           PoissonBC::Neumann, PoissonBC::Neumann);
            fft1d_applicable = true;
            fft1d_periodic_dir = 0;
        } catch (const std::exception& e) {
            std::cerr << "[Solver] FFT1D solver initialization failed: " << e.what() << "\n";
        }
    }
#endif

    // ========================================================================
    // Poisson Solver Auto-Selection
    // Priority: FFT (periodic x+z) → FFT1D (periodic x OR z) → HYPRE → MG
    // ========================================================================
    PoissonSolverType requested = config.poisson_solver;
    std::string selection_reason;

    if (requested == PoissonSolverType::Auto) {
        // Auto-select: FFT > FFT1D > HYPRE > MG
#ifdef USE_FFT_POISSON
        if (fft_applicable) {
            selected_solver_ = PoissonSolverType::FFT;
            selection_reason = "auto: periodic(x,z) + uniform(dx,dz) + 3D";
        } else if (fft1d_applicable) {
            selected_solver_ = PoissonSolverType::FFT1D;
            selection_reason = "auto: periodic(x) + uniform(dx) + 3D (1D FFT)";
        } else
#endif
#ifdef USE_HYPRE
        if (hypre_poisson_solver_) {
            selected_solver_ = PoissonSolverType::HYPRE;
            selection_reason = "auto: FFT not applicable, HYPRE available";
        } else
#endif
        {
            selected_solver_ = PoissonSolverType::MG;
            selection_reason = "auto: fallback to multigrid";
        }
    } else if (requested == PoissonSolverType::FFT) {
#ifdef USE_FFT_POISSON
        if (fft_applicable) {
            selected_solver_ = PoissonSolverType::FFT;
            selection_reason = "explicit: user requested FFT";
        } else {
            std::cerr << "[Solver] Warning: FFT requested but not applicable "
                      << "(requires 3D, periodic x/z, uniform dx/dz). Falling back to ";
            if (fft1d_applicable) {
                selected_solver_ = PoissonSolverType::FFT1D;
                std::cerr << "FFT1D.\n";
                selection_reason = "fallback from FFT: using FFT1D";
            } else
#ifdef USE_HYPRE
            if (hypre_poisson_solver_) {
                selected_solver_ = PoissonSolverType::HYPRE;
                std::cerr << "HYPRE.\n";
                selection_reason = "fallback from FFT: not applicable";
            } else
#endif
            {
                selected_solver_ = PoissonSolverType::MG;
                std::cerr << "MG.\n";
                selection_reason = "fallback from FFT: not applicable";
            }
        }
#else
        std::cerr << "[Solver] Warning: FFT requested but USE_FFT_POISSON not built. ";
#ifdef USE_HYPRE
        selected_solver_ = PoissonSolverType::HYPRE;
        std::cerr << "Using HYPRE.\n";
#else
        selected_solver_ = PoissonSolverType::MG;
        std::cerr << "Using MG.\n";
#endif
        selection_reason = "fallback from FFT: not built";
#endif
    } else if (requested == PoissonSolverType::FFT1D) {
#ifdef USE_FFT_POISSON
        if (fft1d_applicable) {
            selected_solver_ = PoissonSolverType::FFT1D;
            selection_reason = "explicit: user requested FFT1D";
        } else {
            std::cerr << "[Solver] Warning: FFT1D requested but not applicable. ";
            selected_solver_ = PoissonSolverType::MG;
            std::cerr << "Using MG.\n";
            selection_reason = "fallback from FFT1D: not applicable";
        }
#else
        std::cerr << "[Solver] Warning: FFT1D requested but USE_FFT_POISSON not built. ";
        selected_solver_ = PoissonSolverType::MG;
        std::cerr << "Using MG.\n";
        selection_reason = "fallback from FFT1D: not built";
#endif
    } else if (requested == PoissonSolverType::HYPRE) {
#ifdef USE_HYPRE
        if (hypre_poisson_solver_) {
            selected_solver_ = PoissonSolverType::HYPRE;
            selection_reason = "explicit: user requested HYPRE";
        } else {
            std::cerr << "[Solver] Warning: HYPRE initialization failed. Using MG.\n";
            selected_solver_ = PoissonSolverType::MG;
            selection_reason = "fallback from HYPRE: init failed";
        }
#else
        std::cerr << "[Solver] Warning: HYPRE requested but USE_HYPRE not built. Using MG.\n";
        selected_solver_ = PoissonSolverType::MG;
        selection_reason = "fallback from HYPRE: not built";
#endif
    } else {
        // PoissonSolverType::MG
        selected_solver_ = PoissonSolverType::MG;
        selection_reason = "explicit: user requested MG";
    }

    // Log the selection
    const char* solver_name = (selected_solver_ == PoissonSolverType::FFT) ? "FFT" :
                              (selected_solver_ == PoissonSolverType::FFT1D) ? "FFT1D" :
                              (selected_solver_ == PoissonSolverType::HYPRE) ? "HYPRE" : "MG";
    std::cout << "[Poisson] selected=" << solver_name
              << " reason=" << selection_reason
              << " dims=" << mesh.Nx << "x" << mesh.Ny << "x" << mesh.Nz << "\n";

#ifdef USE_GPU_OFFLOAD
    // Fail-fast if GPU offload is enabled but no device is available
    gpu::verify_device_available();
    
    initialize_gpu_buffers();
    // GPU buffers are now mapped and will persist for solver lifetime
    // NOTE: Turbulence models manage their own GPU buffers independently
#endif
}

RANSSolver::~RANSSolver() {
    cleanup_gpu_buffers();  // Safe to call unconditionally (no-op when GPU disabled)
}

void RANSSolver::set_turbulence_model(std::unique_ptr<TurbulenceModel> model) {
    turb_model_ = std::move(model);
    if (turb_model_) {
        turb_model_->set_nu(config_.nu);
        
        // Initialize turbulence model GPU buffers if GPU is available and mesh is initialized
        if (mesh_) {
            turb_model_->initialize_gpu_buffers(*mesh_);
        }
    }
}

void RANSSolver::set_velocity_bc(const VelocityBC& bc) {
    // Validate: periodic BCs must be symmetric (both ends must match)
    if ((bc.x_lo == VelocityBC::Periodic) != (bc.x_hi == VelocityBC::Periodic)) {
        throw std::invalid_argument("Periodic BC in x requires both x_lo and x_hi to be Periodic");
    }
    if ((bc.y_lo == VelocityBC::Periodic) != (bc.y_hi == VelocityBC::Periodic)) {
        throw std::invalid_argument("Periodic BC in y requires both y_lo and y_hi to be Periodic");
    }
    if ((bc.z_lo == VelocityBC::Periodic) != (bc.z_hi == VelocityBC::Periodic)) {
        throw std::invalid_argument("Periodic BC in z requires both z_lo and z_hi to be Periodic");
    }

    velocity_bc_ = bc;

    // Update Poisson BCs based on velocity BCs
    PoissonBC p_x_lo = (bc.x_lo == VelocityBC::Periodic) ? PoissonBC::Periodic : PoissonBC::Neumann;
    PoissonBC p_x_hi = (bc.x_hi == VelocityBC::Periodic) ? PoissonBC::Periodic : PoissonBC::Neumann;
    PoissonBC p_y_lo = (bc.y_lo == VelocityBC::Periodic) ? PoissonBC::Periodic : PoissonBC::Neumann;
    PoissonBC p_y_hi = (bc.y_hi == VelocityBC::Periodic) ? PoissonBC::Periodic : PoissonBC::Neumann;
    PoissonBC p_z_lo = (bc.z_lo == VelocityBC::Periodic) ? PoissonBC::Periodic : PoissonBC::Neumann;
    PoissonBC p_z_hi = (bc.z_hi == VelocityBC::Periodic) ? PoissonBC::Periodic : PoissonBC::Neumann;

    // Store for GPU Poisson solver
    poisson_bc_x_lo_ = p_x_lo;
    poisson_bc_x_hi_ = p_x_hi;
    poisson_bc_y_lo_ = p_y_lo;
    poisson_bc_y_hi_ = p_y_hi;
    poisson_bc_z_lo_ = p_z_lo;
    poisson_bc_z_hi_ = p_z_hi;

    // Set BCs on Poisson solvers - use 3D overload for 3D meshes
    if (!mesh_->is2D()) {
        poisson_solver_.set_bc(p_x_lo, p_x_hi, p_y_lo, p_y_hi, p_z_lo, p_z_hi);
        mg_poisson_solver_.set_bc(p_x_lo, p_x_hi, p_y_lo, p_y_hi, p_z_lo, p_z_hi);
#ifdef USE_HYPRE
        if (hypre_poisson_solver_) {
            hypre_poisson_solver_->set_bc(p_x_lo, p_x_hi, p_y_lo, p_y_hi, p_z_lo, p_z_hi);
        }
#endif
    } else {
        poisson_solver_.set_bc(p_x_lo, p_x_hi, p_y_lo, p_y_hi);
        mg_poisson_solver_.set_bc(p_x_lo, p_x_hi, p_y_lo, p_y_hi);
#ifdef USE_HYPRE
        if (hypre_poisson_solver_) {
            hypre_poisson_solver_->set_bc(p_x_lo, p_x_hi, p_y_lo, p_y_hi);
        }
#endif
    }

    // Re-check FFT/FFT1D applicability after BC update
#ifdef USE_FFT_POISSON
    bool periodic_x = (p_x_lo == PoissonBC::Periodic && p_x_hi == PoissonBC::Periodic);
    bool periodic_z = (p_z_lo == PoissonBC::Periodic && p_z_hi == PoissonBC::Periodic);

    // 2D FFT requires periodic in BOTH x and z
    bool fft_compatible = periodic_x && periodic_z && !mesh_->is2D();

    // 1D FFT requires periodic in EXACTLY ONE of x or z
    bool fft1d_compatible = (periodic_x != periodic_z) && !mesh_->is2D();
    int fft1d_dir = periodic_x ? 0 : 2;  // 0 = x periodic, 2 = z periodic

    if (fft_poisson_solver_) {
        if (fft_compatible) {
            // Update FFT solver BCs
            fft_poisson_solver_->set_bc(p_x_lo, p_x_hi, p_y_lo, p_y_hi, p_z_lo, p_z_hi);
        } else if (selected_solver_ == PoissonSolverType::FFT) {
            // FFT was selected but BCs are now incompatible
            if (fft1d_compatible && fft1d_poisson_solver_) {
                std::cerr << "[Poisson] Warning: FFT solver incompatible with BCs "
                          << "(requires periodic x AND z). Switching to FFT1D.\n";
                selected_solver_ = PoissonSolverType::FFT1D;
            } else {
                std::cerr << "[Poisson] Warning: FFT solver incompatible with BCs "
                          << "(requires periodic x AND z). Switching to MG.\n";
                selected_solver_ = PoissonSolverType::MG;
            }
        }
    }

    if (fft1d_poisson_solver_) {
        if (fft1d_compatible) {
            // Update FFT1D solver BCs
            fft1d_poisson_solver_->set_bc(p_x_lo, p_x_hi, p_y_lo, p_y_hi, p_z_lo, p_z_hi);
        } else if (selected_solver_ == PoissonSolverType::FFT1D) {
            // FFT1D was selected but BCs are now incompatible - switch to MG
            std::cerr << "[Poisson] Warning: FFT1D solver incompatible with BCs "
                      << "(requires periodic in exactly one of x or z). Switching to MG.\n";
            selected_solver_ = PoissonSolverType::MG;
        }
    }
#endif
}

void RANSSolver::set_body_force(double fx, double fy, double fz) {
    fx_ = fx;
    fy_ = fy;
    fz_ = fz;
}

void RANSSolver::initialize(const VectorField& initial_velocity) {
    velocity_ = initial_velocity;
    apply_velocity_bc();
    
    // Initialize k, omega for transport models if not already set
    if (turb_model_ && turb_model_->uses_transport_equations()) {
        // Estimate initial turbulence from velocity magnitude
        double u_max = 0.0;
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                double u = 0.5 * (velocity_.u(i, j) + velocity_.u(i+1, j));
                double v = 0.5 * (velocity_.v(i, j) + velocity_.v(i, j+1));
                double u_mag = std::sqrt(u*u + v*v);
                u_max = std::max(u_max, u_mag);
            }
        }
        // Use reasonable reference velocity - minimum 1% of bulk or 0.01 whichever is larger
        // This ensures k/omega stay above the low-turbulence threshold (1e-8) for EARSM
        double u_ref = std::max(u_max, 0.01);
        double Ti = 0.05;  // 5% turbulence intensity
        double k_init = 1.5 * (u_ref * Ti) * (u_ref * Ti);
        // Ensure k_init is physically meaningful (above low-turb threshold)
        k_init = std::max(k_init, 1e-7);
        
        double omega_init = k_init / (0.09 * config_.nu * 100.0);  // ν_t/ν ≈ 100 initially
        omega_init = std::max(omega_init, 1e-6);  // Ensure omega is also meaningful
        
        k_.fill(k_init);
        omega_.fill(omega_init);
        
        // Set wall values for omega (higher near walls)
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            // Bottom wall
            int j_bot = mesh_->j_begin();
            double y_bot = mesh_->wall_distance(i, j_bot);
            if (y_bot > 1e-10) {
                omega_(i, j_bot) = 10.0 * 6.0 * config_.nu / (0.075 * y_bot * y_bot);
            }
            
            // Top wall
            int j_top = mesh_->j_end() - 1;
            double y_top = mesh_->wall_distance(i, j_top);
            if (y_top > 1e-10) {
                omega_(i, j_top) = 10.0 * 6.0 * config_.nu / (0.075 * y_top * y_top);
            }
        }
    }
    
    if (turb_model_) {
        turb_model_->initialize(*mesh_, velocity_);
    }
    
#ifdef USE_GPU_OFFLOAD
    // Ensure initialized fields are mirrored to device for GPU runs
    sync_to_gpu();
#endif
}

void RANSSolver::initialize_uniform(double u0, double v0) {
    velocity_.fill(u0, v0);
    apply_velocity_bc();
    
    // Initialize k, omega for transport models
    if (turb_model_ && turb_model_->uses_transport_equations()) {
        // Estimate initial turbulence from velocity
        double u_ref = std::max(std::abs(u0), 0.01);
        double Ti = 0.05;  // 5% turbulence intensity
        double k_init = 1.5 * (u_ref * Ti) * (u_ref * Ti);
        // Ensure k_init is physically meaningful (above low-turb threshold)
        k_init = std::max(k_init, 1e-7);
        
        double omega_init = k_init / (0.09 * config_.nu * 100.0);  // ν_t/ν ≈ 100 initially
        omega_init = std::max(omega_init, 1e-6);  // Ensure omega is also meaningful
        
        k_.fill(k_init);
        omega_.fill(omega_init);
        
        // Set wall values for omega (higher near walls)
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            // Bottom wall
            int j_bot = mesh_->j_begin();
            double y_bot = mesh_->wall_distance(i, j_bot);
            omega_(i, j_bot) = 10.0 * 6.0 * config_.nu / (0.075 * y_bot * y_bot);
            
            // Top wall
            int j_top = mesh_->j_end() - 1;
            double y_top = mesh_->wall_distance(i, j_top);
            omega_(i, j_top) = 10.0 * 6.0 * config_.nu / (0.075 * y_top * y_top);
        }
    }
    
    if (turb_model_) {
        turb_model_->initialize(*mesh_, velocity_);
    }
    
#ifdef USE_GPU_OFFLOAD
    // CRITICAL: Sync k_ and omega_ to GPU after CPU-side initialization
    // These were modified at lines 419-420 AFTER initialize_gpu_buffers() ran
    // Without this sync, GPU kernels will use stale zero values instead of proper initial conditions
    if (turb_model_ && turb_model_->uses_transport_equations()) {
        #pragma omp target update to(k_ptr_[0:field_total_size_])
        #pragma omp target update to(omega_ptr_[0:field_total_size_])
    }

    // Also sync velocity to device so GPU path starts from correct ICs
    sync_to_gpu();
#endif
}

void RANSSolver::apply_velocity_bc() {
    NVTX_PUSH("apply_velocity_bc");
    
    // Get unified view (device pointers in GPU build, host pointers in CPU build)
    auto v = get_solver_view();
    
    const int Nx = v.Nx;
    const int Ny = v.Ny;
    const int Ng = v.Ng;
    const int u_stride = v.u_stride;
    const int v_stride = v.v_stride;
    const int u_total_Ny = Ny + 2 * Ng;
    const int v_total_Nx = Nx + 2 * Ng;

    const bool x_lo_periodic = (velocity_bc_.x_lo == VelocityBC::Periodic);
    const bool x_lo_noslip   = (velocity_bc_.x_lo == VelocityBC::NoSlip);
    const bool x_hi_periodic = (velocity_bc_.x_hi == VelocityBC::Periodic);
    const bool x_hi_noslip   = (velocity_bc_.x_hi == VelocityBC::NoSlip);

    const bool y_lo_periodic = (velocity_bc_.y_lo == VelocityBC::Periodic);
    const bool y_lo_noslip   = (velocity_bc_.y_lo == VelocityBC::NoSlip);
    const bool y_hi_periodic = (velocity_bc_.y_hi == VelocityBC::Periodic);
    const bool y_hi_noslip   = (velocity_bc_.y_hi == VelocityBC::NoSlip);

    // Validate that all BCs are supported (Inflow/Outflow not implemented for x/y)
    if (!x_lo_periodic && !x_lo_noslip) {
        throw std::runtime_error("Unsupported velocity BC type for x_lo (only Periodic and NoSlip are implemented)");
    }
    if (!x_hi_periodic && !x_hi_noslip) {
        throw std::runtime_error("Unsupported velocity BC type for x_hi (only Periodic and NoSlip are implemented)");
    }
    if (!y_lo_periodic && !y_lo_noslip) {
        throw std::runtime_error("Unsupported velocity BC type for y_lo (only Periodic and NoSlip are implemented)");
    }
    if (!y_hi_periodic && !y_hi_noslip) {
        throw std::runtime_error("Unsupported velocity BC type for y_hi (only Periodic and NoSlip are implemented)");
    }

    double* u_ptr = v.u_face;
    double* v_ptr = v.v_face;
    [[maybe_unused]] const size_t u_total_size = velocity_.u_total_size();
    [[maybe_unused]] const size_t v_total_size = velocity_.v_total_size();

    // For 3D, apply x/y BCs to ALL z-planes (interior and ghost)
    // This is necessary because the z-BC code assumes x/y BCs are already applied
    // Nz_total = Nz + 2*Ng for 3D, 1 for 2D
    const int Nz_total = mesh_->is2D() ? 1 : (v.Nz + 2*Ng);
    const int u_plane_stride = mesh_->is2D() ? 0 : v.u_plane_stride;
    const int v_plane_stride = mesh_->is2D() ? 0 : v.v_plane_stride;

    // Apply u BCs in x-direction (for all k-planes including ghosts in 3D)
    const int n_u_x_bc = u_total_Ny * Ng * Nz_total;
    #pragma omp target teams distribute parallel for \
        map(present: u_ptr[0:u_total_size]) \
        firstprivate(Nx, Ny, Ng, u_stride, u_plane_stride, Nz_total, x_lo_periodic, x_lo_noslip, x_hi_periodic, x_hi_noslip)
    for (int idx = 0; idx < n_u_x_bc; ++idx) {
        int j = idx % u_total_Ny;
        int g = (idx / u_total_Ny) % Ng;
        int k = idx / (u_total_Ny * Ng);  // k = 0 to Nz_total-1 covers all planes
        double* u_plane_ptr = u_ptr + k * u_plane_stride;
        apply_u_bc_x_staggered(j, g, Nx, Ng, u_stride,
                              x_lo_periodic, x_lo_noslip,
                              x_hi_periodic, x_hi_noslip, u_plane_ptr);
    }

    // Apply u BCs in y-direction (for all k-planes including ghosts in 3D)
    const int n_u_y_bc = (Nx + 1 + 2 * Ng) * Ng * Nz_total;
    #pragma omp target teams distribute parallel for \
        map(present: u_ptr[0:u_total_size]) \
        firstprivate(Nx, Ny, Ng, u_stride, u_plane_stride, Nz_total, y_lo_periodic, y_lo_noslip, y_hi_periodic, y_hi_noslip)
    for (int idx = 0; idx < n_u_y_bc; ++idx) {
        int u_x_size = Nx + 1 + 2 * Ng;
        int i = idx % u_x_size;
        int g = (idx / u_x_size) % Ng;
        int k = idx / (u_x_size * Ng);
        double* u_plane_ptr = u_ptr + k * u_plane_stride;
        apply_u_bc_y_staggered(i, g, Ny, Ng, u_stride,
                              y_lo_periodic, y_lo_noslip,
                              y_hi_periodic, y_hi_noslip, u_plane_ptr);
    }

    // Apply v BCs in x-direction (for all k-planes including ghosts in 3D)
    const int n_v_x_bc = (Ny + 1 + 2 * Ng) * Ng * Nz_total;
    #pragma omp target teams distribute parallel for \
        map(present: v_ptr[0:v_total_size]) \
        firstprivate(Nx, Ny, Ng, v_stride, v_plane_stride, Nz_total, x_lo_periodic, x_lo_noslip, x_hi_periodic, x_hi_noslip)
    for (int idx = 0; idx < n_v_x_bc; ++idx) {
        int v_y_size = Ny + 1 + 2 * Ng;
        int j = idx % v_y_size;
        int g = (idx / v_y_size) % Ng;
        int k = idx / (v_y_size * Ng);
        double* v_plane_ptr = v_ptr + k * v_plane_stride;
        apply_v_bc_x_staggered(j, g, Nx, Ng, v_stride,
                              x_lo_periodic, x_lo_noslip,
                              x_hi_periodic, x_hi_noslip, v_plane_ptr);
    }

    // Apply v BCs in y-direction (for all k-planes including ghosts in 3D)
    const int n_v_y_bc = v_total_Nx * Ng * Nz_total;
    #pragma omp target teams distribute parallel for \
        map(present: v_ptr[0:v_total_size]) \
        firstprivate(Nx, Ny, Ng, v_stride, v_plane_stride, Nz_total, y_lo_periodic, y_lo_noslip, y_hi_periodic, y_hi_noslip)
    for (int idx = 0; idx < n_v_y_bc; ++idx) {
        int i = idx % v_total_Nx;
        int g = (idx / v_total_Nx) % Ng;
        int k = idx / (v_total_Nx * Ng);
        double* v_plane_ptr = v_ptr + k * v_plane_stride;
        apply_v_bc_y_staggered(i, g, Ny, Ng, v_stride,
                              y_lo_periodic, y_lo_noslip,
                              y_hi_periodic, y_hi_noslip, v_plane_ptr);
    }

    // CORNER FIX: For fully periodic domains, apply x-direction BCs again
    // to ensure corner ghosts are correctly wrapped after y-direction BCs modified them
    if (x_lo_periodic && x_hi_periodic) {
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_total_size]) \
            firstprivate(Nx, Ny, Ng, u_stride, u_plane_stride, Nz_total, x_lo_periodic, x_lo_noslip, x_hi_periodic, x_hi_noslip)
        for (int idx = 0; idx < n_u_x_bc; ++idx) {
            int j = idx % u_total_Ny;
            int g = (idx / u_total_Ny) % Ng;
            int k = idx / (u_total_Ny * Ng);
            double* u_plane_ptr = u_ptr + k * u_plane_stride;
            apply_u_bc_x_staggered(j, g, Nx, Ng, u_stride,
                                  x_lo_periodic, x_lo_noslip,
                                  x_hi_periodic, x_hi_noslip, u_plane_ptr);
        }
    }

    if (y_lo_periodic && y_hi_periodic) {
        #pragma omp target teams distribute parallel for \
            map(present: v_ptr[0:v_total_size]) \
            firstprivate(Nx, Ny, Ng, v_stride, v_plane_stride, Nz_total, y_lo_periodic, y_lo_noslip, y_hi_periodic, y_hi_noslip)
        for (int idx = 0; idx < n_v_y_bc; ++idx) {
            int i = idx % v_total_Nx;
            int g = (idx / v_total_Nx) % Ng;
            int k = idx / (v_total_Nx * Ng);
            double* v_plane_ptr = v_ptr + k * v_plane_stride;
            apply_v_bc_y_staggered(i, g, Ny, Ng, v_stride,
                                  y_lo_periodic, y_lo_noslip,
                                  y_hi_periodic, y_hi_noslip, v_plane_ptr);
        }
    }

    // 3D z-direction boundary conditions
    if (!mesh_->is2D()) {
        const int Nz = v.Nz;
        // u_plane_stride and v_plane_stride already defined in outer scope
        const int w_stride = v.w_stride;
        const int w_plane_stride = v.w_plane_stride;
        double* w_ptr = v.w_face;
        [[maybe_unused]] const size_t w_total_size = velocity_.w_total_size();

        const bool z_lo_periodic = (velocity_bc_.z_lo == VelocityBC::Periodic);
        const bool z_hi_periodic = (velocity_bc_.z_hi == VelocityBC::Periodic);
        const bool z_lo_noslip = (velocity_bc_.z_lo == VelocityBC::NoSlip);
        const bool z_hi_noslip = (velocity_bc_.z_hi == VelocityBC::NoSlip);

        // Validate that z-direction BCs are supported (Inflow/Outflow not implemented)
        if (!z_lo_periodic && !z_lo_noslip) {
            throw std::runtime_error("Unsupported velocity BC type for z_lo (only Periodic and NoSlip are implemented)");
        }
        if (!z_hi_periodic && !z_hi_noslip) {
            throw std::runtime_error("Unsupported velocity BC type for z_hi (only Periodic and NoSlip are implemented)");
        }

        // Apply u BCs in z-direction (for all x-faces, all y rows)
        // Each x-face: (Nx+1) i-values, (Ny) j-values, Ng ghost layers at each z-end
        const int n_u_z_bc = (Nx + 1 + 2*Ng) * (Ny + 2*Ng) * Ng;
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_total_size]) \
            firstprivate(Nx, Ny, Ng, Nz, u_stride, u_plane_stride, z_lo_periodic, z_hi_periodic, z_lo_noslip, z_hi_noslip)
        for (int idx = 0; idx < n_u_z_bc; ++idx) {
            int i = idx % (Nx + 1 + 2*Ng);
            int j = (idx / (Nx + 1 + 2*Ng)) % (Ny + 2*Ng);
            int g = idx / ((Nx + 1 + 2*Ng) * (Ny + 2*Ng));
            // z-lo ghost: k = Ng-1-g = Ng-1, Ng-2, ... for g=0,1,...
            // z-hi ghost: k = Ng+Nz+g
            int k_lo = Ng - 1 - g;
            int k_hi = Ng + Nz + g;
            int src_lo = Ng;        // First interior k
            int src_hi = Ng + Nz - 1;  // Last interior k
            int idx_lo = k_lo * u_plane_stride + j * u_stride + i;
            int idx_hi = k_hi * u_plane_stride + j * u_stride + i;
            int idx_src_lo = src_lo * u_plane_stride + j * u_stride + i;
            int idx_src_hi = src_hi * u_plane_stride + j * u_stride + i;

            if (z_lo_periodic && z_hi_periodic) {
                // Periodic: copy from opposite interior boundary
                u_ptr[idx_lo] = u_ptr[(Ng + Nz - 1 - g) * u_plane_stride + j * u_stride + i];
                u_ptr[idx_hi] = u_ptr[(Ng + g) * u_plane_stride + j * u_stride + i];
            } else {
                if (z_lo_noslip) u_ptr[idx_lo] = -u_ptr[idx_src_lo];
                if (z_hi_noslip) u_ptr[idx_hi] = -u_ptr[idx_src_hi];
            }
        }

        // Apply v BCs in z-direction
        const int n_v_z_bc = (Nx + 2*Ng) * (Ny + 1 + 2*Ng) * Ng;
        #pragma omp target teams distribute parallel for \
            map(present: v_ptr[0:v_total_size]) \
            firstprivate(Nx, Ny, Ng, Nz, v_stride, v_plane_stride, z_lo_periodic, z_hi_periodic, z_lo_noslip, z_hi_noslip)
        for (int idx = 0; idx < n_v_z_bc; ++idx) {
            int i = idx % (Nx + 2*Ng);
            int j = (idx / (Nx + 2*Ng)) % (Ny + 1 + 2*Ng);
            int g = idx / ((Nx + 2*Ng) * (Ny + 1 + 2*Ng));
            int k_lo = Ng - 1 - g;
            int k_hi = Ng + Nz + g;
            int src_lo = Ng;
            int src_hi = Ng + Nz - 1;
            int idx_lo = k_lo * v_plane_stride + j * v_stride + i;
            int idx_hi = k_hi * v_plane_stride + j * v_stride + i;
            int idx_src_lo = src_lo * v_plane_stride + j * v_stride + i;
            int idx_src_hi = src_hi * v_plane_stride + j * v_stride + i;

            if (z_lo_periodic && z_hi_periodic) {
                v_ptr[idx_lo] = v_ptr[(Ng + Nz - 1 - g) * v_plane_stride + j * v_stride + i];
                v_ptr[idx_hi] = v_ptr[(Ng + g) * v_plane_stride + j * v_stride + i];
            } else {
                if (z_lo_noslip) v_ptr[idx_lo] = -v_ptr[idx_src_lo];
                if (z_hi_noslip) v_ptr[idx_hi] = -v_ptr[idx_src_hi];
            }
        }

        // Apply w BCs in z-direction (w is at z-faces, so different treatment)
        // For periodic: w at k=Ng and k=Ng+Nz should be same (wrap around)
        const int n_w_z_bc = (Nx + 2*Ng) * (Ny + 2*Ng) * Ng;
        #pragma omp target teams distribute parallel for \
            map(present: w_ptr[0:w_total_size]) \
            firstprivate(Nx, Ny, Ng, Nz, w_stride, w_plane_stride, z_lo_periodic, z_hi_periodic, z_lo_noslip, z_hi_noslip)
        for (int idx = 0; idx < n_w_z_bc; ++idx) {
            int i = idx % (Nx + 2*Ng);
            int j = (idx / (Nx + 2*Ng)) % (Ny + 2*Ng);
            int g = idx / ((Nx + 2*Ng) * (Ny + 2*Ng));
            int k_lo = Ng - 1 - g;
            int k_hi = Ng + Nz + 1 + g;  // Note: Nz+1 z-faces for w
            int idx_lo = k_lo * w_plane_stride + j * w_stride + i;
            int idx_hi = k_hi * w_plane_stride + j * w_stride + i;

            if (z_lo_periodic && z_hi_periodic) {
                // CRITICAL for staggered grid with periodic BCs:
                // The topmost interior face (Ng+Nz) IS the bottommost interior face (Ng)
                // They represent the same physical location in a periodic domain
                if (g == 0) {
                    w_ptr[(Ng + Nz) * w_plane_stride + j * w_stride + i] =
                        w_ptr[Ng * w_plane_stride + j * w_stride + i];
                }
                // For w at z-faces with periodic BC:
                // Ghost at k=Ng-1-g gets value from k=Ng+Nz-1-g (interior near hi)
                // Ghost at k=Ng+Nz+1+g gets value from k=Ng+1+g (interior near lo)
                w_ptr[idx_lo] = w_ptr[(Ng + Nz - 1 - g) * w_plane_stride + j * w_stride + i];
                w_ptr[idx_hi] = w_ptr[(Ng + 1 + g) * w_plane_stride + j * w_stride + i];
            } else {
                // For no-slip: w at boundaries should be zero (normal velocity)
                if (z_lo_noslip) {
                    // w at k=Ng (first interior z-face) = 0 for solid wall
                    if (g == 0) {
                        w_ptr[(Ng) * w_plane_stride + j * w_stride + i] = 0.0;
                    }
                    w_ptr[idx_lo] = -w_ptr[(Ng + g + 1) * w_plane_stride + j * w_stride + i];
                }
                if (z_hi_noslip) {
                    // w at k=Ng+Nz (last interior z-face) = 0 for solid wall
                    if (g == 0) {
                        w_ptr[(Ng + Nz) * w_plane_stride + j * w_stride + i] = 0.0;
                    }
                    w_ptr[idx_hi] = -w_ptr[(Ng + Nz - 1 - g) * w_plane_stride + j * w_stride + i];
                }
            }
        }

        // Apply w BCs in x and y directions
        // w in x-direction
        const int n_w_x_bc = (Ny + 2*Ng) * (Nz + 1 + 2*Ng) * Ng;
        #pragma omp target teams distribute parallel for \
            map(present: w_ptr[0:w_total_size]) \
            firstprivate(Nx, Ng, w_stride, w_plane_stride, x_lo_periodic, x_lo_noslip, x_hi_periodic, x_hi_noslip)
        for (int idx = 0; idx < n_w_x_bc; ++idx) {
            int j = idx % (Ny + 2*Ng);
            int k = (idx / (Ny + 2*Ng)) % (Nz + 1 + 2*Ng);
            int g = idx / ((Ny + 2*Ng) * (Nz + 1 + 2*Ng));
            int i_lo = Ng - 1 - g;
            int i_hi = Ng + Nx + g;
            int idx_lo = k * w_plane_stride + j * w_stride + i_lo;
            int idx_hi = k * w_plane_stride + j * w_stride + i_hi;

            if (x_lo_periodic && x_hi_periodic) {
                w_ptr[idx_lo] = w_ptr[k * w_plane_stride + j * w_stride + (Ng + Nx - 1 - g)];
                w_ptr[idx_hi] = w_ptr[k * w_plane_stride + j * w_stride + (Ng + g)];
            } else {
                if (x_lo_noslip) w_ptr[idx_lo] = -w_ptr[k * w_plane_stride + j * w_stride + (Ng + g)];
                if (x_hi_noslip) w_ptr[idx_hi] = -w_ptr[k * w_plane_stride + j * w_stride + (Ng + Nx - 1 - g)];
            }
        }

        // w in y-direction
        const int n_w_y_bc = (Nx + 2*Ng) * (Nz + 1 + 2*Ng) * Ng;
        #pragma omp target teams distribute parallel for \
            map(present: w_ptr[0:w_total_size]) \
            firstprivate(Ny, Ng, w_stride, w_plane_stride, y_lo_periodic, y_lo_noslip, y_hi_periodic, y_hi_noslip)
        for (int idx = 0; idx < n_w_y_bc; ++idx) {
            int i = idx % (Nx + 2*Ng);
            int k = (idx / (Nx + 2*Ng)) % (Nz + 1 + 2*Ng);
            int g = idx / ((Nx + 2*Ng) * (Nz + 1 + 2*Ng));
            int j_lo = Ng - 1 - g;
            int j_hi = Ng + Ny + g;
            int idx_lo = k * w_plane_stride + j_lo * w_stride + i;
            int idx_hi = k * w_plane_stride + j_hi * w_stride + i;

            if (y_lo_periodic && y_hi_periodic) {
                w_ptr[idx_lo] = w_ptr[k * w_plane_stride + (Ng + Ny - 1 - g) * w_stride + i];
                w_ptr[idx_hi] = w_ptr[k * w_plane_stride + (Ng + g) * w_stride + i];
            } else {
                if (y_lo_noslip) w_ptr[idx_lo] = -w_ptr[k * w_plane_stride + (Ng + g) * w_stride + i];
                if (y_hi_noslip) w_ptr[idx_hi] = -w_ptr[k * w_plane_stride + (Ng + Ny - 1 - g) * w_stride + i];
            }
        }
        // NOTE: x/y BCs for u and v across all k-planes are now handled
        // above (lines 1268-1360) using the staggered kernel functions
        // which properly handle staggered grid periodic BCs.
    }

    NVTX_POP();  // End apply_velocity_bc
}

void RANSSolver::compute_convective_term(const VectorField& vel, VectorField& conv) {
    NVTX_SCOPE_CONVECT("solver:convective_term");

    (void)vel;   // Unused - always operates on velocity_ via view
    (void)conv;  // Unused - always operates on conv_ via view

    // Get unified view
    auto v = get_solver_view();

    const double dx = v.dx;
    const double dy = v.dy;
    const int Nx = v.Nx;
    const int Ny = v.Ny;
    const int Nz = v.Nz;
    const int Ng = v.Ng;
    const int u_stride = v.u_stride;
    const int v_stride = v.v_stride;
    const bool use_central = (config_.convective_scheme == ConvectiveScheme::Central);

    [[maybe_unused]] const size_t u_total_size = velocity_.u_total_size();
    [[maybe_unused]] const size_t v_total_size = velocity_.v_total_size();
    [[maybe_unused]] const size_t w_total_size = velocity_.w_total_size();

    const double* u_ptr      = v.u_face;
    const double* v_ptr      = v.v_face;
    double*       conv_u_ptr = v.conv_u;
    double*       conv_v_ptr = v.conv_v;

    // 3D path
    if (!mesh_->is2D()) {
        const double dz = v.dz;
        const int u_plane_stride = v.u_plane_stride;
        const int v_plane_stride = v.v_plane_stride;
        const int w_stride = v.w_stride;
        const int w_plane_stride = v.w_plane_stride;
        const double* w_ptr = v.w_face;
        double*       conv_w_ptr = v.conv_w;

        // Compute u-momentum convection at x-faces (3D)
        const int n_u_faces = (Nx + 1) * Ny * Nz;
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], w_ptr[0:w_total_size], conv_u_ptr[0:u_total_size]) \
            firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, use_central, Nx, Ny, Ng)
        for (int idx = 0; idx < n_u_faces; ++idx) {
            int i = idx % (Nx + 1) + Ng;
            int j = (idx / (Nx + 1)) % Ny + Ng;
            int k = idx / ((Nx + 1) * Ny) + Ng;

            convective_u_face_kernel_staggered_3d(i, j, k,
                u_stride, u_plane_stride, v_stride, v_plane_stride,
                w_stride, w_plane_stride, u_stride, u_plane_stride,
                dx, dy, dz, use_central, u_ptr, v_ptr, w_ptr, conv_u_ptr);
        }

        // Compute v-momentum convection at y-faces (3D)
        const int n_v_faces = Nx * (Ny + 1) * Nz;
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], w_ptr[0:w_total_size], conv_v_ptr[0:v_total_size]) \
            firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, use_central, Nx, Ny, Ng)
        for (int idx = 0; idx < n_v_faces; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % (Ny + 1) + Ng;
            int k = idx / (Nx * (Ny + 1)) + Ng;

            convective_v_face_kernel_staggered_3d(i, j, k,
                u_stride, u_plane_stride, v_stride, v_plane_stride,
                w_stride, w_plane_stride, v_stride, v_plane_stride,
                dx, dy, dz, use_central, u_ptr, v_ptr, w_ptr, conv_v_ptr);
        }

        // Compute w-momentum convection at z-faces (3D)
        const int n_w_faces = Nx * Ny * (Nz + 1);
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], w_ptr[0:w_total_size], conv_w_ptr[0:w_total_size]) \
            firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, use_central, Nx, Ny, Ng)
        for (int idx = 0; idx < n_w_faces; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % Ny + Ng;
            int k = idx / (Nx * Ny) + Ng;

            convective_w_face_kernel_staggered_3d(i, j, k,
                u_stride, u_plane_stride, v_stride, v_plane_stride,
                w_stride, w_plane_stride, w_stride, w_plane_stride,
                dx, dy, dz, use_central, u_ptr, v_ptr, w_ptr, conv_w_ptr);
        }
        return;
    }

    // 2D path
    // Compute u-momentum convection at x-faces
    const int n_u_faces = (Nx + 1) * Ny;
    #pragma omp target teams distribute parallel for \
        map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], conv_u_ptr[0:u_total_size]) \
        firstprivate(dx, dy, u_stride, v_stride, use_central, Nx, Ng)
    for (int idx = 0; idx < n_u_faces; ++idx) {
        int i_local = idx % (Nx + 1);
        int j_local = idx / (Nx + 1);
        int i = i_local + Ng;
        int j = j_local + Ng;

        convective_u_face_kernel_staggered(i, j, u_stride, v_stride, u_stride, dx, dy, use_central,
                                          u_ptr, v_ptr, conv_u_ptr);
    }

    // Compute v-momentum convection at y-faces
    const int n_v_faces = Nx * (Ny + 1);
    #pragma omp target teams distribute parallel for \
        map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], conv_v_ptr[0:v_total_size]) \
        firstprivate(dx, dy, u_stride, v_stride, use_central, Nx, Ng)
    for (int idx = 0; idx < n_v_faces; ++idx) {
        int i_local = idx % Nx;
        int j_local = idx / Nx;
        int i = i_local + Ng;
        int j = j_local + Ng;

        convective_v_face_kernel_staggered(i, j, u_stride, v_stride, v_stride, dx, dy, use_central,
                                          u_ptr, v_ptr, conv_v_ptr);
    }
}

void RANSSolver::compute_diffusive_term(const VectorField& vel, const ScalarField& nu_eff,
                                        VectorField& diff) {
    NVTX_SCOPE_DIFFUSE("solver:diffusive_term");

    (void)vel;     // Unused - always operates on velocity_ via view
    (void)nu_eff;  // Unused - always operates on nu_eff_ via view
    (void)diff;    // Unused - always operates on diff_ via view

    // Get unified view
    auto v = get_solver_view();

    const double dx = v.dx;
    const double dy = v.dy;
    const int Nx = v.Nx;
    const int Ny = v.Ny;
    const int Nz = v.Nz;
    const int Ng = v.Ng;
    const int u_stride = v.u_stride;
    const int v_stride = v.v_stride;
    const int nu_stride = v.cell_stride;

    [[maybe_unused]] const size_t u_total_size = velocity_.u_total_size();
    [[maybe_unused]] const size_t v_total_size = velocity_.v_total_size();
    [[maybe_unused]] const size_t w_total_size = velocity_.w_total_size();
    [[maybe_unused]] const size_t nu_total_size = field_total_size_;

    const double* u_ptr      = v.u_face;
    const double* v_ptr      = v.v_face;
    const double* nu_ptr     = v.nu_eff;
    double*       diff_u_ptr = v.diff_u;
    double*       diff_v_ptr = v.diff_v;

    // 3D path
    if (!mesh_->is2D()) {
        const double dz = v.dz;
        const int u_plane_stride = v.u_plane_stride;
        const int v_plane_stride = v.v_plane_stride;
        const int w_stride = v.w_stride;
        const int w_plane_stride = v.w_plane_stride;
        const int nu_plane_stride = v.cell_plane_stride;
        const double* w_ptr = v.w_face;
        double*       diff_w_ptr = v.diff_w;

        // Compute u-momentum diffusion at x-faces (3D)
        const int n_u_faces = (Nx + 1) * Ny * Nz;
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_total_size], nu_ptr[0:nu_total_size], diff_u_ptr[0:u_total_size]) \
            firstprivate(dx, dy, dz, u_stride, u_plane_stride, nu_stride, nu_plane_stride, Nx, Ny, Ng)
        for (int idx = 0; idx < n_u_faces; ++idx) {
            int i = idx % (Nx + 1) + Ng;
            int j = (idx / (Nx + 1)) % Ny + Ng;
            int k = idx / ((Nx + 1) * Ny) + Ng;

            diffusive_u_face_kernel_staggered_3d(i, j, k,
                u_stride, u_plane_stride, nu_stride, nu_plane_stride, u_stride, u_plane_stride,
                dx, dy, dz, u_ptr, nu_ptr, diff_u_ptr);
        }

        // Compute v-momentum diffusion at y-faces (3D)
        const int n_v_faces = Nx * (Ny + 1) * Nz;
        #pragma omp target teams distribute parallel for \
            map(present: v_ptr[0:v_total_size], nu_ptr[0:nu_total_size], diff_v_ptr[0:v_total_size]) \
            firstprivate(dx, dy, dz, v_stride, v_plane_stride, nu_stride, nu_plane_stride, Nx, Ny, Ng)
        for (int idx = 0; idx < n_v_faces; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % (Ny + 1) + Ng;
            int k = idx / (Nx * (Ny + 1)) + Ng;

            diffusive_v_face_kernel_staggered_3d(i, j, k,
                v_stride, v_plane_stride, nu_stride, nu_plane_stride, v_stride, v_plane_stride,
                dx, dy, dz, v_ptr, nu_ptr, diff_v_ptr);
        }

        // Compute w-momentum diffusion at z-faces (3D)
        const int n_w_faces = Nx * Ny * (Nz + 1);
        #pragma omp target teams distribute parallel for \
            map(present: w_ptr[0:w_total_size], nu_ptr[0:nu_total_size], diff_w_ptr[0:w_total_size]) \
            firstprivate(dx, dy, dz, w_stride, w_plane_stride, nu_stride, nu_plane_stride, Nx, Ny, Ng)
        for (int idx = 0; idx < n_w_faces; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % Ny + Ng;
            int k = idx / (Nx * Ny) + Ng;

            diffusive_w_face_kernel_staggered_3d(i, j, k,
                w_stride, w_plane_stride, nu_stride, nu_plane_stride, w_stride, w_plane_stride,
                dx, dy, dz, w_ptr, nu_ptr, diff_w_ptr);
        }
        return;
    }

    // 2D path
    // Compute u-momentum diffusion at x-faces
    const int n_u_faces = (Nx + 1) * Ny;
    #pragma omp target teams distribute parallel for \
        map(present: u_ptr[0:u_total_size], nu_ptr[0:nu_total_size], diff_u_ptr[0:u_total_size]) \
        firstprivate(dx, dy, u_stride, nu_stride, Nx, Ng)
    for (int idx = 0; idx < n_u_faces; ++idx) {
        int i_local = idx % (Nx + 1);
        int j_local = idx / (Nx + 1);
        int i = i_local + Ng;
        int j = j_local + Ng;

        diffusive_u_face_kernel_staggered(i, j, u_stride, nu_stride, u_stride, dx, dy,
                                         u_ptr, nu_ptr, diff_u_ptr);
    }

    // Compute v-momentum diffusion at y-faces
    const int n_v_faces = Nx * (Ny + 1);
    #pragma omp target teams distribute parallel for \
        map(present: v_ptr[0:v_total_size], nu_ptr[0:nu_total_size], diff_v_ptr[0:v_total_size]) \
        firstprivate(dx, dy, v_stride, nu_stride, Nx, Ng)
    for (int idx = 0; idx < n_v_faces; ++idx) {
        int i_local = idx % Nx;
        int j_local = idx / Nx;
        int i = i_local + Ng;
        int j = j_local + Ng;

        diffusive_v_face_kernel_staggered(i, j, v_stride, nu_stride, v_stride, dx, dy,
                                         v_ptr, nu_ptr, diff_v_ptr);
    }
}

void RANSSolver::compute_divergence(VelocityWhich which, ScalarField& div) {
    (void)div;  // Unused - always operates on div_velocity_ via view

    // Get unified view
    auto v = get_solver_view();
    auto vel_ptrs = select_face_velocity(v, which);

    const double dx = v.dx;
    const double dy = v.dy;
    const int Nx = v.Nx;
    const int Ny = v.Ny;
    const int Nz = v.Nz;
    const int Ng = v.Ng;
    const int div_stride = v.cell_stride;
    const int u_stride = vel_ptrs.u_stride;
    const int v_stride = vel_ptrs.v_stride;

    [[maybe_unused]] const size_t u_total_size = velocity_.u_total_size();
    [[maybe_unused]] const size_t v_total_size = velocity_.v_total_size();
    [[maybe_unused]] const size_t w_total_size = velocity_.w_total_size();
    [[maybe_unused]] const size_t div_total_size = field_total_size_;

    const double* u_ptr = vel_ptrs.u;
    const double* v_ptr = vel_ptrs.v;
    double* div_ptr = v.div;

    // 3D path
    if (!mesh_->is2D()) {
        const double dz = v.dz;
        const int u_plane_stride = vel_ptrs.u_plane_stride;
        const int v_plane_stride = vel_ptrs.v_plane_stride;
        const int w_stride = vel_ptrs.w_stride;
        const int w_plane_stride = vel_ptrs.w_plane_stride;
        const int div_plane_stride = v.cell_plane_stride;
        const double* w_ptr = vel_ptrs.w;

        const int n_cells = Nx * Ny * Nz;
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], w_ptr[0:w_total_size], div_ptr[0:div_total_size]) \
            firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, div_stride, div_plane_stride, Nx, Ny, Ng)
        for (int idx = 0; idx < n_cells; ++idx) {
            const int i = idx % Nx + Ng;
            const int j = (idx / Nx) % Ny + Ng;
            const int k = idx / (Nx * Ny) + Ng;

            divergence_cell_kernel_staggered_3d(i, j, k,
                u_stride, u_plane_stride, v_stride, v_plane_stride,
                w_stride, w_plane_stride, div_stride, div_plane_stride,
                dx, dy, dz, u_ptr, v_ptr, w_ptr, div_ptr);
        }
        return;
    }

    // 2D path
    const int n_cells = Nx * Ny;

    // Use target data for scalar parameters (NVHPC workaround)
    #pragma omp target data map(to: dx, dy, u_stride, v_stride, div_stride, Nx, Ng)
    {
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], div_ptr[0:div_total_size])
        for (int idx = 0; idx < n_cells; ++idx) {
            const int i = idx % Nx + Ng;  // Cell center i index (with ghosts)
            const int j = idx / Nx + Ng;  // Cell center j index (with ghosts)

            // Fully inlined divergence computation
            const int u_right = j * u_stride + (i + 1);
            const int u_left = j * u_stride + i;
            const int v_top = (j + 1) * v_stride + i;
            const int v_bottom = j * v_stride + i;
            const int div_idx = j * div_stride + i;

            const double dudx = (u_ptr[u_right] - u_ptr[u_left]) / dx;
            const double dvdy = (v_ptr[v_top] - v_ptr[v_bottom]) / dy;
            div_ptr[div_idx] = dudx + dvdy;
        }
    }
}

// Compute max absolute divergence of a velocity field (CPU-side, staggered grid)
[[maybe_unused]] static double compute_max_divergence(const Mesh& mesh, const VectorField& vel) {
    const double dx = mesh.dx;
    const double dy = mesh.dy;
    double max_div = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            // Staggered divergence: div = (u(i+1,j) - u(i,j))/dx + (v(i,j+1) - v(i,j))/dy
            const double du_dx = (vel.u(i+1, j) - vel.u(i, j)) / dx;
            const double dv_dy = (vel.v(i, j+1) - vel.v(i, j)) / dy;
            const double div = du_dx + dv_dy;
            max_div = std::max(max_div, std::abs(div));
        }
    }
    return max_div;
}

void RANSSolver::compute_pressure_gradient(ScalarField& dp_dx, ScalarField& dp_dy) {
    double dx = mesh_->dx;
    double dy = mesh_->dy;
    
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            dp_dx(i, j) = (pressure_(i+1, j) - pressure_(i-1, j)) / (2.0 * dx);
            dp_dy(i, j) = (pressure_(i, j+1) - pressure_(i, j-1)) / (2.0 * dy);
        }
    }
}

void RANSSolver::correct_velocity() {
    NVTX_PUSH("correct_velocity");

    // Get unified view (device pointers in GPU build, host pointers in CPU build)
    auto v = get_solver_view();

    const double dx = v.dx;
    const double dy = v.dy;
    const double dt = v.dt;
    const int Nx = v.Nx;
    const int Ny = v.Ny;
    const int Nz = v.Nz;
    const int Ng = v.Ng;
    const int u_stride = v.u_stride;
    const int v_stride = v.v_stride;
    const int p_stride = v.cell_stride;

    const bool x_periodic = (velocity_bc_.x_lo == VelocityBC::Periodic) &&
                            (velocity_bc_.x_hi == VelocityBC::Periodic);
    const bool y_periodic = (velocity_bc_.y_lo == VelocityBC::Periodic) &&
                            (velocity_bc_.y_hi == VelocityBC::Periodic);
    const bool z_periodic = (velocity_bc_.z_lo == VelocityBC::Periodic) &&
                            (velocity_bc_.z_hi == VelocityBC::Periodic);

    [[maybe_unused]] const size_t u_total_size = velocity_.u_total_size();
    [[maybe_unused]] const size_t v_total_size = velocity_.v_total_size();
    [[maybe_unused]] const size_t w_total_size = velocity_.w_total_size();
    [[maybe_unused]] const size_t p_total_size = field_total_size_;

    const double* u_star_ptr = v.u_star_face;
    const double* v_star_ptr = v.v_star_face;
    const double* p_corr_ptr = v.p_corr;
    double*       u_ptr      = v.u_face;
    double*       v_ptr      = v.v_face;
    double*       p_ptr      = v.p;

    // 3D path
    if (!mesh_->is2D()) {
        const double dz = v.dz;
        const int u_plane_stride = v.u_plane_stride;
        const int v_plane_stride = v.v_plane_stride;
        const int w_stride = v.w_stride;
        const int w_plane_stride = v.w_plane_stride;
        const int p_plane_stride = v.cell_plane_stride;
        const double* w_star_ptr = v.w_star_face;
        double*       w_ptr      = v.w_face;

        // Correct u-velocities at x-faces (3D)
        const int n_u_faces = (Nx + 1) * Ny * Nz;
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_total_size], u_star_ptr[0:u_total_size], p_corr_ptr[0:p_total_size]) \
            firstprivate(dx, dt, u_stride, u_plane_stride, p_stride, p_plane_stride, Nx, Ny, Ng)
        for (int idx = 0; idx < n_u_faces; ++idx) {
            int i = idx % (Nx + 1) + Ng;
            int j = (idx / (Nx + 1)) % Ny + Ng;
            int k = idx / ((Nx + 1) * Ny) + Ng;

            correct_u_face_kernel_staggered_3d(i, j, k,
                u_stride, u_plane_stride, p_stride, p_plane_stride,
                dx, dt, u_star_ptr, p_corr_ptr, u_ptr);
        }

        // Enforce x-periodicity (3D)
        if (x_periodic) {
            const int n_u_periodic = Ny * Nz;
            #pragma omp target teams distribute parallel for \
                map(present: u_ptr[0:u_total_size]) \
                firstprivate(u_stride, u_plane_stride, Nx, Ny, Ng)
            for (int idx = 0; idx < n_u_periodic; ++idx) {
                int j = idx % Ny + Ng;
                int k = idx / Ny + Ng;
                int i_left = Ng;
                int i_right = Ng + Nx;
                int idx_left = k * u_plane_stride + j * u_stride + i_left;
                int idx_right = k * u_plane_stride + j * u_stride + i_right;
                double u_avg = 0.5 * (u_ptr[idx_left] + u_ptr[idx_right]);
                u_ptr[idx_left] = u_avg;
                u_ptr[idx_right] = u_avg;
            }
        }

        // Correct v-velocities at y-faces (3D)
        const int n_v_faces = Nx * (Ny + 1) * Nz;
        #pragma omp target teams distribute parallel for \
            map(present: v_ptr[0:v_total_size], v_star_ptr[0:v_total_size], p_corr_ptr[0:p_total_size]) \
            firstprivate(dy, dt, v_stride, v_plane_stride, p_stride, p_plane_stride, Nx, Ny, Ng)
        for (int idx = 0; idx < n_v_faces; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % (Ny + 1) + Ng;
            int k = idx / (Nx * (Ny + 1)) + Ng;

            correct_v_face_kernel_staggered_3d(i, j, k,
                v_stride, v_plane_stride, p_stride, p_plane_stride,
                dy, dt, v_star_ptr, p_corr_ptr, v_ptr);
        }

        // Enforce y-periodicity (3D)
        if (y_periodic) {
            const int n_v_periodic = Nx * Nz;
            #pragma omp target teams distribute parallel for \
                map(present: v_ptr[0:v_total_size]) \
                firstprivate(v_stride, v_plane_stride, Nx, Ny, Ng)
            for (int idx = 0; idx < n_v_periodic; ++idx) {
                int i = idx % Nx + Ng;
                int k = idx / Nx + Ng;
                int j_bottom = Ng;
                int j_top = Ng + Ny;
                int idx_bottom = k * v_plane_stride + j_bottom * v_stride + i;
                int idx_top = k * v_plane_stride + j_top * v_stride + i;
                double v_avg = 0.5 * (v_ptr[idx_bottom] + v_ptr[idx_top]);
                v_ptr[idx_bottom] = v_avg;
                v_ptr[idx_top] = v_avg;
            }
        }

        // Correct w-velocities at z-faces (3D)
        const int n_w_faces = Nx * Ny * (Nz + 1);
        #pragma omp target teams distribute parallel for \
            map(present: w_ptr[0:w_total_size], w_star_ptr[0:w_total_size], p_corr_ptr[0:p_total_size]) \
            firstprivate(dz, dt, w_stride, w_plane_stride, p_stride, p_plane_stride, Nx, Ny, Ng)
        for (int idx = 0; idx < n_w_faces; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % Ny + Ng;
            int k = idx / (Nx * Ny) + Ng;

            correct_w_face_kernel_staggered_3d(i, j, k,
                w_stride, w_plane_stride, p_stride, p_plane_stride,
                dz, dt, w_star_ptr, p_corr_ptr, w_ptr);
        }

        // Enforce z-periodicity (3D)
        if (z_periodic) {
            const int n_w_periodic = Nx * Ny;
            #pragma omp target teams distribute parallel for \
                map(present: w_ptr[0:w_total_size]) \
                firstprivate(w_stride, w_plane_stride, Nx, Nz, Ng)
            for (int idx = 0; idx < n_w_periodic; ++idx) {
                int i = idx % Nx + Ng;
                int j = idx / Nx + Ng;
                int k_front = Ng;
                int k_back = Ng + Nz;
                int idx_front = k_front * w_plane_stride + j * w_stride + i;
                int idx_back = k_back * w_plane_stride + j * w_stride + i;
                double w_avg = 0.5 * (w_ptr[idx_front] + w_ptr[idx_back]);
                w_ptr[idx_front] = w_avg;
                w_ptr[idx_back] = w_avg;
            }
        }

        // Update pressure at cell centers (3D)
        const int n_cells = Nx * Ny * Nz;
        #pragma omp target teams distribute parallel for \
            map(present: p_ptr[0:p_total_size], p_corr_ptr[0:p_total_size]) \
            firstprivate(p_stride, p_plane_stride, Nx, Ny, Ng)
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % Ny + Ng;
            int k = idx / (Nx * Ny) + Ng;

            int p_idx = k * p_plane_stride + j * p_stride + i;
            p_ptr[p_idx] += p_corr_ptr[p_idx];
        }

        NVTX_POP();
        return;
    }

    // 2D path
    const int n_cells = Nx * Ny;

    // Correct ALL u-velocities at x-faces (including redundant face if periodic)
    const int n_u_faces = (Nx + 1) * Ny;
    #pragma omp target teams distribute parallel for \
        map(present: u_ptr[0:u_total_size], u_star_ptr[0:u_total_size], p_corr_ptr[0:p_total_size]) \
        firstprivate(dx, dt, u_stride, p_stride, Nx, Ng)
    for (int idx = 0; idx < n_u_faces; ++idx) {
        int i_local = idx % (Nx + 1);
        int j_local = idx / (Nx + 1);
        int i = i_local + Ng;
        int j = j_local + Ng;

        correct_u_face_kernel_staggered(i, j, u_stride, p_stride, dx, dt,
                                       u_star_ptr, p_corr_ptr, u_ptr);
    }

    // Enforce exact x-periodicity: average the left and right edge values
    if (x_periodic) {
        const int n_u_periodic = Ny;
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_total_size]) \
            firstprivate(u_stride, Nx, Ng)
        for (int j_local = 0; j_local < n_u_periodic; ++j_local) {
            int j = j_local + Ng;
            int i_left = Ng;
            int i_right = Ng + Nx;
            double u_avg = 0.5 * (u_ptr[j * u_stride + i_left] + u_ptr[j * u_stride + i_right]);
            u_ptr[j * u_stride + i_left] = u_avg;
            u_ptr[j * u_stride + i_right] = u_avg;
        }
    }

    // Correct ALL v-velocities at y-faces (including redundant face if periodic)
    const int n_v_faces = Nx * (Ny + 1);
    #pragma omp target teams distribute parallel for \
        map(present: v_ptr[0:v_total_size], v_star_ptr[0:v_total_size], p_corr_ptr[0:p_total_size]) \
        firstprivate(dy, dt, v_stride, p_stride, Nx, Ng)
    for (int idx = 0; idx < n_v_faces; ++idx) {
        int i_local = idx % Nx;
        int j_local = idx / Nx;
        int i = i_local + Ng;
        int j = j_local + Ng;

        correct_v_face_kernel_staggered(i, j, v_stride, p_stride, dy, dt,
                                       v_star_ptr, p_corr_ptr, v_ptr);
    }

    // Enforce exact y-periodicity: average the bottom and top edge values
    if (y_periodic) {
        const int n_v_periodic = Nx;
        #pragma omp target teams distribute parallel for \
            map(present: v_ptr[0:v_total_size]) \
            firstprivate(v_stride, Ny, Ng)
        for (int i_local = 0; i_local < n_v_periodic; ++i_local) {
            int i = i_local + Ng;
            int j_bottom = Ng;
            int j_top = Ng + Ny;
            double v_avg = 0.5 * (v_ptr[j_bottom * v_stride + i] + v_ptr[j_top * v_stride + i]);
            v_ptr[j_bottom * v_stride + i] = v_avg;
            v_ptr[j_top * v_stride + i] = v_avg;
        }
    }

    // Update pressure at cell centers
    #pragma omp target teams distribute parallel for \
        map(present: p_ptr[0:p_total_size], p_corr_ptr[0:p_total_size]) \
        firstprivate(p_stride, Nx)
    for (int idx = 0; idx < n_cells; ++idx) {
        int i = idx % Nx + Ng;
        int j = idx / Nx + Ng;

        update_pressure_kernel(i, j, p_stride, p_corr_ptr, p_ptr);
    }

    NVTX_POP();  // End correct_velocity
}

double RANSSolver::compute_residual() {
 // Compute residual based on velocity change
    double max_res = 0.0;
    
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            double du = velocity_.u(i, j) - velocity_star_.u(i, j);
            double dv = velocity_.v(i, j) - velocity_star_.v(i, j);
            max_res = std::max(max_res, std::abs(du));
            max_res = std::max(max_res, std::abs(dv));
        }
    }
    
    return max_res;
}

double RANSSolver::step() {
    TIMED_SCOPE("solver_step");
    
    // Store old velocity for convergence check (at face locations for staggered grid)
    const int Ng = mesh_->Nghost;
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    
#ifdef USE_GPU_OFFLOAD
    // GPU path: copy current velocity to velocity_old on device (device-to-device, no H↔D)
    {
    NVTX_PUSH("velocity_copy");
    const size_t u_total_size = velocity_.u_total_size();
    const size_t v_total_size = velocity_.v_total_size();
    const int u_stride = Nx + 2 * Ng + 1;
    const int v_stride = Nx + 2 * Ng;

    if (mesh_->is2D()) {
        // Copy u-velocity device-to-device (2D)
        #pragma omp target teams distribute parallel for collapse(2) \
            map(present: velocity_u_ptr_[0:u_total_size], velocity_old_u_ptr_[0:u_total_size])
        for (int j = Ng; j < Ng + Ny; ++j) {
            for (int i = Ng; i <= Ng + Nx; ++i) {
                const int idx = j * u_stride + i;
                velocity_old_u_ptr_[idx] = velocity_u_ptr_[idx];
            }
        }

        // Copy v-velocity device-to-device (2D)
        #pragma omp target teams distribute parallel for collapse(2) \
            map(present: velocity_v_ptr_[0:v_total_size], velocity_old_v_ptr_[0:v_total_size])
        for (int j = Ng; j <= Ng + Ny; ++j) {
            for (int i = Ng; i < Ng + Nx; ++i) {
                const int idx = j * v_stride + i;
                velocity_old_v_ptr_[idx] = velocity_v_ptr_[idx];
            }
        }
    } else {
        // 3D path - need to copy u, v, AND w
        const int Nz = mesh_->Nz;
        const int u_plane_stride = u_stride * (Ny + 2*Ng);
        const int v_plane_stride = v_stride * (Ny + 2*Ng + 1);
        const size_t w_total_size = velocity_.w_total_size();
        const int w_stride = Nx + 2*Ng;
        const int w_plane_stride = w_stride * (Ny + 2*Ng);

        // Copy u-velocity device-to-device (3D)
        #pragma omp target teams distribute parallel for collapse(3) \
            map(present: velocity_u_ptr_[0:u_total_size], velocity_old_u_ptr_[0:u_total_size])
        for (int k = Ng; k < Ng + Nz; ++k) {
            for (int j = Ng; j < Ng + Ny; ++j) {
                for (int i = Ng; i <= Ng + Nx; ++i) {
                    const int idx = k * u_plane_stride + j * u_stride + i;
                    velocity_old_u_ptr_[idx] = velocity_u_ptr_[idx];
                }
            }
        }

        // Copy v-velocity device-to-device (3D)
        #pragma omp target teams distribute parallel for collapse(3) \
            map(present: velocity_v_ptr_[0:v_total_size], velocity_old_v_ptr_[0:v_total_size])
        for (int k = Ng; k < Ng + Nz; ++k) {
            for (int j = Ng; j <= Ng + Ny; ++j) {
                for (int i = Ng; i < Ng + Nx; ++i) {
                    const int idx = k * v_plane_stride + j * v_stride + i;
                    velocity_old_v_ptr_[idx] = velocity_v_ptr_[idx];
                }
            }
        }

        // Copy w-velocity device-to-device (3D) - THIS WAS MISSING!
        #pragma omp target teams distribute parallel for collapse(3) \
            map(present: velocity_w_ptr_[0:w_total_size], velocity_old_w_ptr_[0:w_total_size])
        for (int k = Ng; k <= Ng + Nz; ++k) {
            for (int j = Ng; j < Ng + Ny; ++j) {
                for (int i = Ng; i < Ng + Nx; ++i) {
                    const int idx = k * w_plane_stride + j * w_stride + i;
                    velocity_old_w_ptr_[idx] = velocity_w_ptr_[idx];
                }
            }
        }
    }
    NVTX_POP();
    }
#else
    // CPU path: use host-side velocity_old_
    if (mesh_->is2D()) {
        for (int j = Ng; j < Ng + Ny; ++j) {
            for (int i = Ng; i <= Ng + Nx; ++i) {
                velocity_old_.u(i, j) = velocity_.u(i, j);
            }
        }
        for (int j = Ng; j <= Ng + Ny; ++j) {
            for (int i = Ng; i < Ng + Nx; ++i) {
                velocity_old_.v(i, j) = velocity_.v(i, j);
            }
        }
    } else {
        // 3D path
        const int Nz = mesh_->Nz;
        for (int k = Ng; k < Ng + Nz; ++k) {
            for (int j = Ng; j < Ng + Ny; ++j) {
                for (int i = Ng; i <= Ng + Nx; ++i) {
                    velocity_old_.u(i, j, k) = velocity_.u(i, j, k);
                }
            }
        }
        for (int k = Ng; k < Ng + Nz; ++k) {
            for (int j = Ng; j <= Ng + Ny; ++j) {
                for (int i = Ng; i < Ng + Nx; ++i) {
                    velocity_old_.v(i, j, k) = velocity_.v(i, j, k);
                }
            }
        }
        for (int k = Ng; k <= Ng + Nz; ++k) {
            for (int j = Ng; j < Ng + Ny; ++j) {
                for (int i = Ng; i < Ng + Nx; ++i) {
                    velocity_old_.w(i, j, k) = velocity_.w(i, j, k);
                }
            }
        }
    }
#endif
    
    // 1a. Advance turbulence transport equations (if model uses them)
    if (turb_model_ && turb_model_->uses_transport_equations()) {
        TIMED_SCOPE("turbulence_transport");
        NVTX_PUSH("turbulence_transport");
        
        // Get device view for GPU-accelerated transport
        const TurbulenceDeviceView* device_view_ptr = nullptr;
#ifdef USE_GPU_OFFLOAD
        TurbulenceDeviceView device_view = get_device_view();
        if (device_view.is_valid()) {
            device_view_ptr = &device_view;
        }
#endif
        
        turb_model_->advance_turbulence(
            *mesh_,
            velocity_,
            current_dt_,
            k_,          // Updated in-place
            omega_,      // Updated in-place
            nu_t_,       // Previous step's nu_t for diffusion coefficients
            device_view_ptr
        );
        NVTX_POP();
        
#ifdef USE_GPU_OFFLOAD
        // CRITICAL FIX: Sync k and omega to GPU after transport equation update
        // ONLY if model didn't use GPU path (models operating on device_view don't need this)
        if (!turb_model_->is_gpu_ready()) {
            #pragma omp target update to(k_ptr_[0:field_total_size_])
            #pragma omp target update to(omega_ptr_[0:field_total_size_])
        }
#endif
    }
    
    // 1b. Update turbulence model (compute nu_t and optional tau_ij)
    if (turb_model_) {
        TIMED_SCOPE("turbulence_update");
        NVTX_PUSH("turbulence_update");
        
        // PHASE 1 GPU OPTIMIZATION: Pass device view if GPU is ready
        const TurbulenceDeviceView* device_view_ptr = nullptr;
#ifdef USE_GPU_OFFLOAD
        TurbulenceDeviceView device_view = get_device_view();
        if (device_view.is_valid()) {
            device_view_ptr = &device_view;
        }
        
        // GPU simulation: enforce device_view validity (host fallback forbidden)
        if (gpu_ready_ && (!device_view_ptr || !device_view_ptr->is_valid())) {
            throw std::runtime_error("GPU simulation requires valid TurbulenceDeviceView - host fallback forbidden");
        }
#endif
        
        turb_model_->update(*mesh_, velocity_, k_, omega_, nu_t_, 
                           turb_model_->provides_reynolds_stresses() ? &tau_ij_ : nullptr,
                           device_view_ptr);
        NVTX_POP();
        
        // CRITICAL FIX: Only sync nu_t to GPU if the model didn't use GPU path
        // Models that use device_view write directly to GPU nu_t and should NOT be overwritten
        // Models that work on CPU (like NN-MLP) write to host nu_t and MUST be synced to GPU
#ifdef USE_GPU_OFFLOAD
        // If device_view was valid and model is GPU-ready, nu_t is already on device
        // Otherwise (CPU path), sync host nu_t to device
        bool model_used_gpu = (device_view_ptr && device_view_ptr->is_valid() && turb_model_->is_gpu_ready());
        if (!model_used_gpu) {
        #pragma omp target update to(nu_t_ptr_[0:field_total_size_])
        }
#endif
    }
    
    // Effective viscosity: nu_eff_ = nu + nu_t (use persistent field)
    // GPU path: compute directly on GPU without CPU fill
#ifdef USE_GPU_OFFLOAD
    if (gpu_ready_) {
        NVTX_PUSH("nu_eff_computation");
        const int Nx = mesh_->Nx;
        const int Ny = mesh_->Ny;
        const int Nz = mesh_->Nz;
        const int Ng = mesh_->Nghost;
        const int stride = Nx + 2 * Ng;
        const int plane_stride = stride * (Ny + 2 * Ng);
        const size_t total_size = field_total_size_;
        const double nu = config_.nu;
        double* nu_eff_ptr = nu_eff_ptr_;
        const double* nu_t_ptr = nu_t_ptr_;
        const bool is_2d = mesh_->is2D();

        if (is_2d) {
            // 2D path
            const int n_cells = Nx * Ny;
            if (turb_model_) {
                #pragma omp target teams distribute parallel for \
                    map(present: nu_eff_ptr[0:total_size]) \
                    map(present: nu_t_ptr[0:total_size]) \
                    firstprivate(nu, stride, Nx, Ng)
                for (int idx = 0; idx < n_cells; ++idx) {
                    int i = idx % Nx + Ng;
                    int j = idx / Nx + Ng;
                    int cell_idx = j * stride + i;
                    nu_eff_ptr[cell_idx] = nu + nu_t_ptr[cell_idx];
                }
            } else {
                #pragma omp target teams distribute parallel for \
                    map(present: nu_eff_ptr[0:total_size]) \
                    firstprivate(nu, stride, Nx, Ng)
                for (int idx = 0; idx < n_cells; ++idx) {
                    int i = idx % Nx + Ng;
                    int j = idx / Nx + Ng;
                    int cell_idx = j * stride + i;
                    nu_eff_ptr[cell_idx] = nu;
                }
            }
        } else {
            // 3D path
            const int n_cells = Nx * Ny * Nz;
            if (turb_model_) {
                #pragma omp target teams distribute parallel for \
                    map(present: nu_eff_ptr[0:total_size]) \
                    map(present: nu_t_ptr[0:total_size]) \
                    firstprivate(nu, stride, plane_stride, Nx, Ny, Ng)
                for (int idx = 0; idx < n_cells; ++idx) {
                    int i = idx % Nx + Ng;
                    int j = (idx / Nx) % Ny + Ng;
                    int k = idx / (Nx * Ny) + Ng;
                    int cell_idx = k * plane_stride + j * stride + i;
                    nu_eff_ptr[cell_idx] = nu + nu_t_ptr[cell_idx];
                }
            } else {
                #pragma omp target teams distribute parallel for \
                    map(present: nu_eff_ptr[0:total_size]) \
                    firstprivate(nu, stride, plane_stride, Nx, Ny, Ng)
                for (int idx = 0; idx < n_cells; ++idx) {
                    int i = idx % Nx + Ng;
                    int j = (idx / Nx) % Ny + Ng;
                    int k = idx / (Nx * Ny) + Ng;
                    int cell_idx = k * plane_stride + j * stride + i;
                    nu_eff_ptr[cell_idx] = nu;
                }
            }
        }
        NVTX_POP();
    } else
#endif
    {
        // CPU path
        nu_eff_.fill(config_.nu);
        if (turb_model_) {
            if (mesh_->is2D()) {
                for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                    for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                        nu_eff_(i, j) = config_.nu + nu_t_(i, j);
                    }
                }
            } else {
                for (int k = mesh_->k_begin(); k < mesh_->k_end(); ++k) {
                    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                            nu_eff_(i, j, k) = config_.nu + nu_t_(i, j, k);
                        }
                    }
                }
            }
        }
    }

    // 2. Compute convective and diffusive terms (use persistent fields)
    {
        TIMED_SCOPE("convective_term");
        NVTX_PUSH("convection");
        compute_convective_term(velocity_, conv_);
        NVTX_POP();
    }
    
    {
        TIMED_SCOPE("diffusive_term");
        NVTX_PUSH("diffusion");
        compute_diffusive_term(velocity_, nu_eff_, diff_);
        NVTX_POP();
    }
    
    // 3. Compute provisional velocity u* (without pressure gradient) at face locations
    // u* = u^n + dt * (-conv + diff + body_force)
    NVTX_PUSH("predictor_step");
    
    // Get unified view (reuse Nx, Ny, Ng from function scope)
    auto v = get_solver_view();
    
    const int u_stride_pred = v.u_stride;
    const int v_stride_pred = v.v_stride;
    const double dt = v.dt;
    const double fx = fx_;
    const double fy = fy_;
    
    const bool x_periodic = (velocity_bc_.x_lo == VelocityBC::Periodic) && 
                            (velocity_bc_.x_hi == VelocityBC::Periodic);
    const bool y_periodic = (velocity_bc_.y_lo == VelocityBC::Periodic) && 
                            (velocity_bc_.y_hi == VelocityBC::Periodic);
    
    [[maybe_unused]] const size_t u_total_size_pred = velocity_.u_total_size();
    [[maybe_unused]] const size_t v_total_size_pred = velocity_.v_total_size();
    
    const double* u_ptr = v.u_face;
    const double* v_ptr = v.v_face;
    double* u_star_ptr = v.u_star_face;
    double* v_star_ptr = v.v_star_face;
    const double* conv_u_ptr = v.conv_u;
    const double* conv_v_ptr = v.conv_v;
    const double* diff_u_ptr = v.diff_u;
    const double* diff_v_ptr = v.diff_v;

    const bool is_2d_pred = mesh_->is2D();
    const int Nz_pred = mesh_->Nz;
    const int Nz_eff_pred = is_2d_pred ? 1 : Nz_pred;
    // Avoid reading uninitialized strides in 2D mode (set to 0 if 2D)
    const int u_plane_stride_pred = is_2d_pred ? 0 : v.u_plane_stride;
    const int v_plane_stride_pred = is_2d_pred ? 0 : v.v_plane_stride;

    // Compute u* at ALL x-faces (including redundant if periodic)
    const int n_u_faces_pred = (Nx + 1) * Ny * Nz_eff_pred;
    #pragma omp target teams distribute parallel for \
        map(present: u_ptr[0:u_total_size_pred], u_star_ptr[0:u_total_size_pred], \
                    conv_u_ptr[0:u_total_size_pred], diff_u_ptr[0:u_total_size_pred]) \
        firstprivate(dt, fx, u_stride_pred, u_plane_stride_pred, Nx, Ny, Nz_eff_pred, Ng, is_2d_pred)
    for (int idx = 0; idx < n_u_faces_pred; ++idx) {
        int i_local = idx % (Nx + 1);
        int j_local = (idx / (Nx + 1)) % Ny;
        int k_local = idx / ((Nx + 1) * Ny);
        int i = i_local + Ng;
        int j = j_local + Ng;
        int k = k_local + Ng;
        int u_idx = is_2d_pred ? (j * u_stride_pred + i)
                               : (k * u_plane_stride_pred + j * u_stride_pred + i);

        u_star_ptr[u_idx] = u_ptr[u_idx] + dt * (-conv_u_ptr[u_idx] + diff_u_ptr[u_idx] + fx);
    }

    // Enforce exact x-periodicity for u*: average left and right edges
    if (x_periodic) {
        const int n_u_periodic = Ny * Nz_eff_pred;
        #pragma omp target teams distribute parallel for \
            map(present: u_star_ptr[0:u_total_size_pred]) \
            firstprivate(u_stride_pred, u_plane_stride_pred, Nx, Ny, Nz_eff_pred, Ng, is_2d_pred)
        for (int idx = 0; idx < n_u_periodic; ++idx) {
            int j_local = idx % Ny;
            int k_local = idx / Ny;
            int j = j_local + Ng;
            int k = k_local + Ng;
            int base = is_2d_pred ? (j * u_stride_pred)
                                  : (k * u_plane_stride_pred + j * u_stride_pred);
            double u_avg = 0.5 * (u_star_ptr[base + Ng] + u_star_ptr[base + (Ng + Nx)]);
            u_star_ptr[base + Ng] = u_avg;
            u_star_ptr[base + (Ng + Nx)] = u_avg;
        }
    }

    // Compute v* at ALL y-faces (including redundant if periodic)
    const int n_v_faces_pred = Nx * (Ny + 1) * Nz_eff_pred;
    #pragma omp target teams distribute parallel for \
        map(present: v_ptr[0:v_total_size_pred], v_star_ptr[0:v_total_size_pred], \
                    conv_v_ptr[0:v_total_size_pred], diff_v_ptr[0:v_total_size_pred]) \
        firstprivate(dt, fy, v_stride_pred, v_plane_stride_pred, Nx, Ny, Nz_eff_pred, Ng, is_2d_pred)
    for (int idx = 0; idx < n_v_faces_pred; ++idx) {
        int i_local = idx % Nx;
        int j_local = (idx / Nx) % (Ny + 1);
        int k_local = idx / (Nx * (Ny + 1));
        int i = i_local + Ng;
        int j = j_local + Ng;
        int k = k_local + Ng;
        int v_idx = is_2d_pred ? (j * v_stride_pred + i)
                               : (k * v_plane_stride_pred + j * v_stride_pred + i);

        v_star_ptr[v_idx] = v_ptr[v_idx] + dt * (-conv_v_ptr[v_idx] + diff_v_ptr[v_idx] + fy);
    }

    // Enforce exact y-periodicity for v*: average bottom and top edges
    if (y_periodic) {
        const int n_v_periodic = Nx * Nz_eff_pred;
        #pragma omp target teams distribute parallel for \
            map(present: v_star_ptr[0:v_total_size_pred]) \
            firstprivate(v_stride_pred, v_plane_stride_pred, Nx, Ny, Nz_eff_pred, Ng, is_2d_pred)
        for (int idx = 0; idx < n_v_periodic; ++idx) {
            int i_local = idx % Nx;
            int k_local = idx / Nx;
            int i = i_local + Ng;
            int k = k_local + Ng;
            int base_lo = is_2d_pred ? (Ng * v_stride_pred + i)
                                     : (k * v_plane_stride_pred + Ng * v_stride_pred + i);
            int base_hi = is_2d_pred ? ((Ng + Ny) * v_stride_pred + i)
                                     : (k * v_plane_stride_pred + (Ng + Ny) * v_stride_pred + i);
            double v_avg = 0.5 * (v_star_ptr[base_lo] + v_star_ptr[base_hi]);
            v_star_ptr[base_lo] = v_avg;
            v_star_ptr[base_hi] = v_avg;
        }
    }

    // 3D: Compute w* at ALL z-faces
    if (!mesh_->is2D()) {
        const int Nz = mesh_->Nz;
        const int w_stride_pred = v.w_stride;
        const int w_plane_stride_pred = v.w_plane_stride;
        const double fz = fz_;
        [[maybe_unused]] const size_t w_total_size_pred = velocity_.w_total_size();

        const double* w_ptr = v.w_face;
        double* w_star_ptr = v.w_star_face;
        const double* conv_w_ptr = v.conv_w;
        const double* diff_w_ptr = v.diff_w;

        const bool z_periodic = (velocity_bc_.z_lo == VelocityBC::Periodic) &&
                                (velocity_bc_.z_hi == VelocityBC::Periodic);

        // Compute w* = w + dt * (-conv_w + diff_w + fz)
        const int n_w_faces_pred = Nx * Ny * (Nz + 1);
        #pragma omp target teams distribute parallel for \
            map(present: w_ptr[0:w_total_size_pred], w_star_ptr[0:w_total_size_pred], \
                        conv_w_ptr[0:w_total_size_pred], diff_w_ptr[0:w_total_size_pred]) \
            firstprivate(dt, fz, w_stride_pred, w_plane_stride_pred, Nx, Ny, Ng)
        for (int idx = 0; idx < n_w_faces_pred; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % Ny + Ng;
            int k = idx / (Nx * Ny) + Ng;
            int w_idx = k * w_plane_stride_pred + j * w_stride_pred + i;

            w_star_ptr[w_idx] = w_ptr[w_idx] + dt * (-conv_w_ptr[w_idx] + diff_w_ptr[w_idx] + fz);
        }

        // Enforce exact z-periodicity for w*: average front and back edges
        if (z_periodic) {
            const int n_w_periodic = Nx * Ny;
            #pragma omp target teams distribute parallel for \
                map(present: w_star_ptr[0:w_total_size_pred]) \
                firstprivate(w_stride_pred, w_plane_stride_pred, Nx, Nz, Ng)
            for (int idx = 0; idx < n_w_periodic; ++idx) {
                int i = idx % Nx + Ng;
                int j = idx / Nx + Ng;
                int idx_back = Ng * w_plane_stride_pred + j * w_stride_pred + i;
                int idx_front = (Ng + Nz) * w_plane_stride_pred + j * w_stride_pred + i;
                double w_avg = 0.5 * (w_star_ptr[idx_back] + w_star_ptr[idx_front]);
                w_star_ptr[idx_back] = w_avg;
                w_star_ptr[idx_front] = w_avg;
            }
        }
    }

    // Apply BCs to provisional velocity (needed for divergence calculation)
    // Temporarily swap velocity_ and velocity_star_ to use apply_velocity_bc
    std::swap(velocity_, velocity_star_);
#ifdef USE_GPU_OFFLOAD
    // CRITICAL: std::swap invalidates GPU pointers - they still point to old memory
    // After swap, velocity_u_ptr_ points to what is now velocity_star_ data!
    // Must swap the pointers too to keep them consistent
    std::swap(velocity_u_ptr_, velocity_star_u_ptr_);
    std::swap(velocity_v_ptr_, velocity_star_v_ptr_);
    if (!mesh_->is2D()) {
        std::swap(velocity_w_ptr_, velocity_star_w_ptr_);
    }
#endif

    // PHASE 1.5 OPTIMIZATION: Skip redundant BC call for fully periodic domains
    // The inline periodic averaging above already handles periodic BCs correctly
    // Only apply BCs if domain has non-periodic boundaries (which need ghost cell updates)
    const bool z_periodic_check = mesh_->is2D() ||
                                  ((velocity_bc_.z_lo == VelocityBC::Periodic) &&
                                   (velocity_bc_.z_hi == VelocityBC::Periodic));
    const bool needs_bc_update = !x_periodic || !y_periodic || !z_periodic_check;
    if (needs_bc_update) {
        apply_velocity_bc();
    }

    std::swap(velocity_, velocity_star_);
#ifdef USE_GPU_OFFLOAD
    // Swap pointers back to restore original mapping
    std::swap(velocity_u_ptr_, velocity_star_u_ptr_);
    std::swap(velocity_v_ptr_, velocity_star_v_ptr_);
    if (!mesh_->is2D()) {
        std::swap(velocity_w_ptr_, velocity_star_w_ptr_);
    }
#endif
    NVTX_POP();  // End predictor_step
    
    // 4. Solve pressure Poisson equation
    // nabla^2p' = (1/dt) nabla*u*
    {
        TIMED_SCOPE("divergence");
        NVTX_PUSH("divergence");
        compute_divergence(VelocityWhich::Star, div_velocity_);
        NVTX_POP();
    }
    
    // Build RHS on GPU and subtract mean divergence to ensure solvability
    // GPU-RESIDENT OPTIMIZATION: Keep all data on device, only transfer scalars
    double mean_div = 0.0;
    
#ifdef USE_GPU_OFFLOAD
    if (gpu_ready_) {
        // GPU-resident path: compute mean divergence on device via reduction
        const int Nx = mesh_->Nx;
        const int Ny = mesh_->Ny;
        const int Nz = mesh_->Nz;
        const int Ng = mesh_->Nghost;
        const int i_begin = mesh_->i_begin();
        const int j_begin = mesh_->j_begin();
        const int k_begin = mesh_->k_begin();
        const int stride = Nx + 2 * Ng;
        const int plane_stride = stride * (Ny + 2 * Ng);
        const bool is_2d = mesh_->is2D();

        double sum_div = 0.0;
        int count = is_2d ? (Nx * Ny) : (Nx * Ny * Nz);

        if (is_2d) {
            // 2D path
            #pragma omp target teams distribute parallel for collapse(2) \
                map(present: div_velocity_ptr_[0:field_total_size_]) \
                reduction(+:sum_div)
            for (int j = 0; j < Ny; ++j) {
                for (int i = 0; i < Nx; ++i) {
                    int ii = i + i_begin;
                    int jj = j + j_begin;
                    int idx = jj * stride + ii;
                    sum_div += div_velocity_ptr_[idx];
                }
            }
        } else {
            // 3D path
            #pragma omp target teams distribute parallel for collapse(3) \
                map(present: div_velocity_ptr_[0:field_total_size_]) \
                reduction(+:sum_div)
            for (int k = 0; k < Nz; ++k) {
                for (int j = 0; j < Ny; ++j) {
                    for (int i = 0; i < Nx; ++i) {
                        int ii = i + i_begin;
                        int jj = j + j_begin;
                        int kk = k + k_begin;
                        int idx = kk * plane_stride + jj * stride + ii;
                        sum_div += div_velocity_ptr_[idx];
                    }
                }
            }
        }

        mean_div = (count > 0) ? sum_div / count : 0.0;

        // Build RHS on GPU: rhs = (div - mean_div) / dt
        const double dt_inv = 1.0 / current_dt_;

        if (is_2d) {
            // 2D path
            #pragma omp target teams distribute parallel for collapse(2) \
                map(present: div_velocity_ptr_[0:field_total_size_], rhs_poisson_ptr_[0:field_total_size_])
            for (int j = 0; j < Ny; ++j) {
                for (int i = 0; i < Nx; ++i) {
                    int ii = i + i_begin;
                    int jj = j + j_begin;
                    int idx = jj * stride + ii;
                    rhs_poisson_ptr_[idx] = (div_velocity_ptr_[idx] - mean_div) * dt_inv;
                }
            }
        } else {
            // 3D path
            #pragma omp target teams distribute parallel for collapse(3) \
                map(present: div_velocity_ptr_[0:field_total_size_], rhs_poisson_ptr_[0:field_total_size_])
            for (int k = 0; k < Nz; ++k) {
                for (int j = 0; j < Ny; ++j) {
                    for (int i = 0; i < Nx; ++i) {
                        int ii = i + i_begin;
                        int jj = j + j_begin;
                        int kk = k + k_begin;
                        int idx = kk * plane_stride + jj * stride + ii;
                        rhs_poisson_ptr_[idx] = (div_velocity_ptr_[idx] - mean_div) * dt_inv;
                    }
                }
            }
        }

        // OPTIMIZATION: Warm-start for Poisson solver (device-resident)
        // Zero pressure correction on device on first iteration only
        if (iter_ == 0) {
            #pragma omp target teams distribute parallel for \
                map(present: pressure_corr_ptr_[0:field_total_size_])
            for (size_t idx = 0; idx < field_total_size_; ++idx) {
                pressure_corr_ptr_[idx] = 0.0;
            }
        }
        // Otherwise, reuse previous solution (already on device, no action needed)

    } else
#endif
    {
        // Host path
        double sum_div = 0.0;
        int count = 0;

        if (mesh_->is2D()) {
            // 2D path
            for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                    double div = div_velocity_(i, j);
                    sum_div += div;
                    ++count;
                }
            }
            mean_div = (count > 0) ? sum_div / count : 0.0;

            // Use multiplication by inverse to match GPU arithmetic exactly
            const double dt_inv = 1.0 / current_dt_;
            for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                    rhs_poisson_(i, j) = (div_velocity_(i, j) - mean_div) * dt_inv;
                }
            }
        } else {
            // 3D path
            const int Nz = mesh_->Nz;
            const int Ng = mesh_->Nghost;
            for (int k = Ng; k < Ng + Nz; ++k) {
                for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                    for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                        double div = div_velocity_(i, j, k);
                        sum_div += div;
                        ++count;
                    }
                }
            }
            mean_div = (count > 0) ? sum_div / count : 0.0;

            // Use multiplication by inverse to match GPU arithmetic exactly
            const double dt_inv = 1.0 / current_dt_;
            for (int k = Ng; k < Ng + Nz; ++k) {
                for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                    for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                        rhs_poisson_(i, j, k) = (div_velocity_(i, j, k) - mean_div) * dt_inv;
                    }
                }
            }
        }

        // Warm-start: zero on first iteration
        if (iter_ == 0) {
            pressure_correction_.fill(0.0);
        }
    }
    
    // 4b. Solve Poisson equation for pressure correction
    {
        TIMED_SCOPE("poisson_solve");
        NVTX_PUSH("poisson_solve");
        
        // CRITICAL: Use relative tolerance for Poisson solver (standard multigrid practice)
        // When turbulence changes effective viscosity, RHS magnitude varies significantly
        // Absolute tolerance would be too strict for small RHS, too loose for large RHS
        double rhs_norm_sq = 0.0;
        int rhs_count = 0;

#ifdef USE_GPU_OFFLOAD
        {
            const int Nx = mesh_->Nx;
            const int Ny = mesh_->Ny;
            const int Nz = mesh_->Nz;
            const int Ng = mesh_->Nghost;
            const int i_begin = mesh_->i_begin();
            const int j_begin = mesh_->j_begin();
            const int k_begin = mesh_->k_begin();
            const int stride = Nx + 2 * Ng;
            const int plane_stride = stride * (Ny + 2 * Ng);
            const bool is_2d = mesh_->is2D();

            if (is_2d) {
                #pragma omp target teams distribute parallel for collapse(2) \
                    map(present: rhs_poisson_ptr_[0:field_total_size_]) \
                    reduction(+:rhs_norm_sq, rhs_count)
                for (int j = 0; j < Ny; ++j) {
                    for (int i = 0; i < Nx; ++i) {
                        int ii = i + i_begin;
                        int jj = j + j_begin;
                        int idx = jj * stride + ii;
                        double rhs_val = rhs_poisson_ptr_[idx];
                        rhs_norm_sq += rhs_val * rhs_val;
                        rhs_count++;
                    }
                }
            } else {
                #pragma omp target teams distribute parallel for collapse(3) \
                    map(present: rhs_poisson_ptr_[0:field_total_size_]) \
                    reduction(+:rhs_norm_sq, rhs_count)
                for (int k = 0; k < Nz; ++k) {
                    for (int j = 0; j < Ny; ++j) {
                        for (int i = 0; i < Nx; ++i) {
                            int ii = i + i_begin;
                            int jj = j + j_begin;
                            int kk = k + k_begin;
                            int idx = kk * plane_stride + jj * stride + ii;
                            double rhs_val = rhs_poisson_ptr_[idx];
                            rhs_norm_sq += rhs_val * rhs_val;
                            rhs_count++;
                        }
                    }
                }
            }
        }
#else
        if (mesh_->is2D()) {
            for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                    double rhs_val = rhs_poisson_(i, j);
                    rhs_norm_sq += rhs_val * rhs_val;
                    rhs_count++;
                }
            }
        } else {
            const int Nz = mesh_->Nz;
            const int Ng = mesh_->Nghost;
            for (int k = Ng; k < Ng + Nz; ++k) {
                for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                    for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                        double rhs_val = rhs_poisson_(i, j, k);
                        rhs_norm_sq += rhs_val * rhs_val;
                        rhs_count++;
                    }
                }
            }
        }
#endif
        
        double rhs_rms = std::sqrt(rhs_norm_sq / std::max(rhs_count, 1));
        
        // Scale tolerance by RHS magnitude (relative convergence)
        // Use max(rhs_rms, 1e-12) to avoid making tolerance too tight for near-zero RHS
        // Also enforce absolute floor to prevent over-solving when near steady state
        double relative_tol = config_.poisson_tol * std::max(rhs_rms, 1e-12);
        double effective_tol = std::max(relative_tol, config_.poisson_abs_tol_floor);
        
        PoissonConfig pcfg;
        pcfg.tol = effective_tol;
        pcfg.max_iter = config_.poisson_max_iter;
        pcfg.omega = config_.poisson_omega;
        pcfg.verbose = false;  // Disable per-cycle output (too verbose)
        
        // Environment variable to enable detailed Poisson cycle diagnostics
        static bool poisson_diagnostics = (std::getenv("NNCFD_POISSON_DIAGNOSTICS") != nullptr);
        static int poisson_diagnostics_interval = []() {
            const char* env = std::getenv("NNCFD_POISSON_DIAGNOSTICS_INTERVAL");
            int v = env ? std::atoi(env) : 1;
            return (v > 0) ? v : 1;
        }();
        
        int cycles = 0;
        double final_residual = 0.0;

        // Dispatch to selected Poisson solver
        // Note: Selection was done at init time; we just execute the selected path here
        static bool solver_logged = false;

#ifdef USE_GPU_OFFLOAD
        if (gpu_ready_) {
            // GPU path based on selected solver
            switch (selected_solver_) {
#ifdef USE_FFT_POISSON
                case PoissonSolverType::FFT:
                    if (fft_poisson_solver_) {
                        if (!solver_logged) {
                            std::cout << "[Poisson] Using FFT solve_device() (cuFFT+cuSPARSE)\n";
                            solver_logged = true;
                        }
                        cycles = fft_poisson_solver_->solve_device(rhs_poisson_ptr_, pressure_corr_ptr_, pcfg);
                        final_residual = fft_poisson_solver_->residual();
                    }
                    break;
                case PoissonSolverType::FFT1D:
                    if (fft1d_poisson_solver_) {
                        if (!solver_logged) {
                            std::cout << "[Poisson] Using FFT1D solve_device() (1D cuFFT + 2D Helmholtz)\n";
                            solver_logged = true;
                        }
                        cycles = fft1d_poisson_solver_->solve_device(rhs_poisson_ptr_, pressure_corr_ptr_, pcfg);
                        final_residual = fft1d_poisson_solver_->residual();
                    }
                    break;
#endif
#ifdef USE_HYPRE
                case PoissonSolverType::HYPRE:
                    if (hypre_poisson_solver_) {
                        if (hypre_poisson_solver_->using_cuda()) {
                            if (!solver_logged) {
                                std::cout << "[Poisson] Using HYPRE solve_device() (CUDA)\n";
                                solver_logged = true;
                            }
                            cycles = hypre_poisson_solver_->solve_device(rhs_poisson_ptr_, pressure_corr_ptr_, pcfg);
                            final_residual = hypre_poisson_solver_->residual();
                        } else {
                            // HYPRE host fallback with GPU staging
                            if (!solver_logged) {
                                std::cout << "[Poisson] Using HYPRE solve() (host, GPU staging)\n";
                                solver_logged = true;
                            }
                            #pragma omp target update from(rhs_poisson_ptr_[0:field_total_size_])
                            std::memcpy(rhs_poisson_.data().data(), rhs_poisson_ptr_, field_total_size_ * sizeof(double));
                            cycles = hypre_poisson_solver_->solve(rhs_poisson_, pressure_correction_, pcfg);
                            final_residual = hypre_poisson_solver_->residual();
                            std::memcpy(pressure_corr_ptr_, pressure_correction_.data().data(), field_total_size_ * sizeof(double));
                            #pragma omp target update to(pressure_corr_ptr_[0:field_total_size_])
                        }
                    }
                    break;
#endif
                case PoissonSolverType::MG:
                default:
                    if (!solver_logged) {
                        std::cout << "[Poisson] Using MG solve_device()\n";
                        solver_logged = true;
                    }
                    cycles = mg_poisson_solver_.solve_device(rhs_poisson_ptr_, pressure_corr_ptr_, pcfg);
                    final_residual = mg_poisson_solver_.residual();
                    break;
            }
        } else
#endif
        {
            // Host path
            if (!solver_logged) {
                std::cout << "[Poisson] Using HOST path\n";
                solver_logged = true;
            }
            if (use_multigrid_) {
                cycles = mg_poisson_solver_.solve(rhs_poisson_, pressure_correction_, pcfg);
                final_residual = mg_poisson_solver_.residual();
            } else {
                cycles = poisson_solver_.solve(rhs_poisson_, pressure_correction_, pcfg);
                final_residual = poisson_solver_.residual();
            }
        }

        // Print cycle count diagnostics if enabled
        if (poisson_diagnostics && (iter_ % poisson_diagnostics_interval == 0)) {
            std::cout << "[Poisson] iter=" << iter_ << " cycles=" << cycles
                      << " residual=" << std::scientific << std::setprecision(15)
                      << final_residual << "\n";
        }
        
        NVTX_POP();
    }
    
    // 5. Correct velocity and pressure
    {
        TIMED_SCOPE("velocity_correction");
        NVTX_PUSH("velocity_correction");
        correct_velocity();
        NVTX_POP();
    }
    
    // 6. Apply boundary conditions
    apply_velocity_bc();
    
    // Note: iter_ is managed by the outer solve loop, don't increment here
    
    // Return max velocity change as convergence criterion (unified view-based)
    auto v_res = get_solver_view();

    [[maybe_unused]] const size_t u_total_size_res = velocity_.u_total_size();
    [[maybe_unused]] const size_t v_total_size_res = velocity_.v_total_size();
    [[maybe_unused]] const size_t w_total_size_res = velocity_.w_total_size();
    const double* u_new_ptr = v_res.u_face;
    const double* v_new_ptr = v_res.v_face;
    const double* u_old_ptr = v_res.u_old_face;
    const double* v_old_ptr = v_res.v_old_face;
    const int u_stride_res = v_res.u_stride;
    const int v_stride_res = v_res.v_stride;
    const int Nz = mesh_->Nz;
    const bool is_2d_res = mesh_->is2D();
    const int Nz_eff = is_2d_res ? 1 : Nz;  // Effective Nz for loop bounds

    // Compute max |u_new - u_old| via reduction
    const int n_u_faces_res = (Nx + 1) * Ny * Nz_eff;
    const int u_plane_stride_res = is_2d_res ? 0 : v_res.u_plane_stride;
    double max_du = 0.0;
    #pragma omp target teams distribute parallel for reduction(max:max_du) \
        map(present: u_new_ptr[0:u_total_size_res], u_old_ptr[0:u_total_size_res]) \
        map(to: Ng, u_stride_res, u_plane_stride_res, Nx, Ny, Nz_eff, is_2d_res)
    for (int idx = 0; idx < n_u_faces_res; ++idx) {
        int i_local = idx % (Nx + 1);
        int j_local = (idx / (Nx + 1)) % Ny;
        int k_local = idx / ((Nx + 1) * Ny);
        int i = i_local + Ng;
        int j = j_local + Ng;
        int k = k_local + Ng;
        int u_idx = is_2d_res ? (j * u_stride_res + i)
                              : (k * u_plane_stride_res + j * u_stride_res + i);
        double du = u_new_ptr[u_idx] - u_old_ptr[u_idx];
        if (du < 0.0) du = -du;
        if (du > max_du) max_du = du;
    }

    // Compute max |v_new - v_old| via reduction
    const int n_v_faces_res = Nx * (Ny + 1) * Nz_eff;
    const int v_plane_stride_res = is_2d_res ? 0 : v_res.v_plane_stride;
    double max_dv = 0.0;
    #pragma omp target teams distribute parallel for reduction(max:max_dv) \
        map(present: v_new_ptr[0:v_total_size_res], v_old_ptr[0:v_total_size_res]) \
        map(to: Ng, v_stride_res, v_plane_stride_res, Nx, Ny, Nz_eff, is_2d_res)
    for (int idx = 0; idx < n_v_faces_res; ++idx) {
        int i_local = idx % Nx;
        int j_local = (idx / Nx) % (Ny + 1);
        int k_local = idx / (Nx * (Ny + 1));
        int i = i_local + Ng;
        int j = j_local + Ng;
        int k = k_local + Ng;
        int v_idx = is_2d_res ? (j * v_stride_res + i)
                              : (k * v_plane_stride_res + j * v_stride_res + i);
        double dv = v_new_ptr[v_idx] - v_old_ptr[v_idx];
        if (dv < 0.0) dv = -dv;
        if (dv > max_dv) max_dv = dv;
    }

    double max_change = (max_du > max_dv) ? max_du : max_dv;

    // For 3D, also check w component
    if (!is_2d_res) {
        const double* w_new_ptr = v_res.w_face;
        const double* w_old_ptr = v_res.w_old_face;
        const int w_stride_res = v_res.w_stride;
        const int w_plane_stride_res = v_res.w_plane_stride;
        const int n_w_faces_res = Nx * Ny * (Nz + 1);
        double max_dw = 0.0;
        #pragma omp target teams distribute parallel for reduction(max:max_dw) \
            map(present: w_new_ptr[0:w_total_size_res], w_old_ptr[0:w_total_size_res]) \
            map(to: Ng, w_stride_res, w_plane_stride_res, Nx, Ny, Nz)
        for (int idx = 0; idx < n_w_faces_res; ++idx) {
            int i_local = idx % Nx;
            int j_local = (idx / Nx) % Ny;
            int k_local = idx / (Nx * Ny);
            int i = i_local + Ng;
            int j = j_local + Ng;
            int k = k_local + Ng;
            int w_idx = k * w_plane_stride_res + j * w_stride_res + i;
            double dw = w_new_ptr[w_idx] - w_old_ptr[w_idx];
            if (dw < 0.0) dw = -dw;
            if (dw > max_dw) max_dw = dw;
        }
        if (max_dw > max_change) max_change = max_dw;
    }

    // NaN/Inf GUARD: Check for numerical stability issues
    // Do this after turbulence update but before next iteration starts
    check_for_nan_inf(step_count_);
    ++step_count_;

    return max_change;
}

std::pair<double, int> RANSSolver::solve_steady() {
    double residual = 1.0;
    
    if (config_.verbose) {
        // Enable line buffering for immediate output visibility (SLURM/redirected stdout)
        std::cout << std::unitbuf;
        
        if (config_.adaptive_dt) {
            std::cout << std::setw(8) << "Iter" 
                      << std::setw(15) << "Residual"
                      << std::setw(15) << "Max |u|"
                      << std::setw(12) << "dt"
                      << std::endl;
        } else {
            std::cout << std::setw(8) << "Iter" 
                      << std::setw(15) << "Residual"
                      << std::setw(15) << "Max |u|"
                      << std::endl;
        }
    }
    
    for (iter_ = 0; iter_ < config_.max_iter; ++iter_) {
        // Update time step if adaptive
        if (config_.adaptive_dt) {
            current_dt_ = compute_adaptive_dt();
        }
        
        residual = step();
        
        if (config_.verbose && (iter_ + 1) % config_.output_freq == 0) {
            double max_vel = velocity_.max_magnitude();
            if (config_.adaptive_dt) {
                std::cout << std::setw(8) << iter_ + 1
                          << std::setw(15) << std::scientific << std::setprecision(3) << residual
                          << std::setw(15) << std::fixed << max_vel
                          << std::setw(12) << std::scientific << std::setprecision(2) << current_dt_
                          << std::endl;  // Flush for immediate visibility
            } else {
                std::cout << std::setw(8) << iter_ + 1
                          << std::setw(15) << std::scientific << std::setprecision(3) << residual
                          << std::setw(15) << std::fixed << max_vel
                          << std::endl;  // Flush for immediate visibility
            }
        }
        
        if (residual < config_.tol) {
            if (config_.verbose) {
                std::cout << "Converged at iteration " << iter_ + 1 
                          << " with residual " << residual << std::endl;
            }
            break;
        }
        
        // Check for divergence
        if (std::isnan(residual) || std::isinf(residual)) {
            if (config_.verbose) {
                std::cerr << "Solver diverged at iteration " << iter_ + 1 << std::endl;
            }
            break;
        }
    }
    
#ifdef USE_GPU_OFFLOAD
    // Sync solution fields after solve completes for backward compatibility
    // This ensures CPU data is up-to-date for tests and diagnostics
    // Note: solve_steady_with_snapshots() handles syncs during I/O instead
    sync_solution_from_gpu();
#endif
    
    return {residual, iter_ + 1};
}

std::pair<double, int> RANSSolver::solve_steady_with_snapshots(
    const std::string& output_prefix,
    int num_snapshots,
    int snapshot_freq) 
{
    // Calculate snapshot frequency if not provided
    if (snapshot_freq < 0 && num_snapshots > 0) {
        snapshot_freq = std::max(1, config_.max_iter / num_snapshots);
    }
    
    if (config_.verbose && !output_prefix.empty()) {
        std::cout << "Will output ";
        if (num_snapshots > 0) {
            std::cout << num_snapshots << " VTK snapshots (every " 
                     << snapshot_freq << " iterations)" << std::endl;
        } else {
            std::cout << "final VTK snapshot only" << std::endl;
        }
    }
    
    double residual = 1.0;
    int snapshot_count = 0;
    
    if (config_.verbose) {
        // Enable line buffering for immediate output visibility
        std::cout << std::unitbuf;
        
        if (config_.adaptive_dt) {
            std::cout << std::setw(8) << "Iter" 
                      << std::setw(15) << "Residual"
                      << std::setw(15) << "Max |u|"
                      << std::setw(12) << "dt"
                      << std::endl;
        } else {
            std::cout << std::setw(8) << "Iter" 
                      << std::setw(15) << "Residual"
                      << std::setw(15) << "Max |u|"
                      << std::endl;
        }
    }
    
    for (iter_ = 0; iter_ < config_.max_iter; ++iter_) {
        // Update time step if adaptive
        if (config_.adaptive_dt) {
            current_dt_ = compute_adaptive_dt();
        }
        
        residual = step();
        
        // Write VTK snapshots at regular intervals
        if (!output_prefix.empty() && num_snapshots > 0 && 
            snapshot_freq > 0 && (iter_ + 1) % snapshot_freq == 0) {
            snapshot_count++;
            std::string vtk_file = output_prefix + "_" + 
                                  std::to_string(snapshot_count) + ".vtk";
            try {
                write_vtk(vtk_file);
                if (config_.verbose) {
                    std::cout << "Wrote snapshot " << snapshot_count 
                             << ": " << vtk_file << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "Warning: Could not write VTK snapshot: " 
                         << e.what() << std::endl;
            }
        }
        
        // Console output
        if (config_.verbose && (iter_ + 1) % config_.output_freq == 0) {
            double max_vel = velocity_.max_magnitude();
            if (config_.adaptive_dt) {
                std::cout << std::setw(8) << iter_ + 1
                          << std::setw(15) << std::scientific << std::setprecision(3) << residual
                          << std::setw(15) << std::fixed << max_vel
                          << std::setw(12) << std::scientific << std::setprecision(2) << current_dt_
                          << std::endl;
            } else {
                std::cout << std::setw(8) << iter_ + 1
                          << std::setw(15) << std::scientific << std::setprecision(3) << residual
                          << std::setw(15) << std::fixed << max_vel
                          << std::endl;
            }
        }
        
        if (residual < config_.tol) {
            if (config_.verbose) {
                std::cout << "Converged at iteration " << iter_ + 1 
                          << " with residual " << residual << std::endl;
            }
            break;
        }
        
        // Check for divergence
        if (std::isnan(residual) || std::isinf(residual)) {
            if (config_.verbose) {
                std::cerr << "Solver diverged at iteration " << iter_ + 1 << std::endl;
            }
            break;
        }
    }
    
    // Write final snapshot if output prefix provided
    if (!output_prefix.empty()) {
        std::string final_file = output_prefix + "_final.vtk";
        try {
            write_vtk(final_file);
            if (config_.verbose) {
                std::cout << "Final VTK output: " << final_file << "\n";
                if (num_snapshots > 0) {
                    std::cout << "Total VTK snapshots: " << snapshot_count + 1 
                             << " (" << snapshot_count << " during + 1 final)\n";
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: Could not write final VTK: " 
                     << e.what() << "\n";
        }
    }
    
#ifdef USE_GPU_OFFLOAD
    // Sync all fields from GPU after solve completes
    // write_vtk() calls sync_from_gpu(), but if no output was written we still need to sync
    if (output_prefix.empty()) {
        sync_from_gpu();
    }
#endif
    
    return {residual, iter_ + 1};
}

double RANSSolver::bulk_velocity() const {
    // Area-averaged streamwise velocity
    double sum = 0.0;
    int count = 0;
    
    [[maybe_unused]] const int Nx = mesh_->Nx;
    [[maybe_unused]] const int Ny = mesh_->Ny;
    [[maybe_unused]] const int Ng = mesh_->Nghost;
    
#ifdef USE_GPU_OFFLOAD
    if (gpu_ready_) {
        // GPU path: compute sum on device, only transfer scalar
        const size_t u_total_size = velocity_.u_total_size();
        const int u_stride = Nx + 2*Ng + 1;
        
        #pragma omp target teams distribute parallel for collapse(2) \
            map(present: velocity_u_ptr_[0:u_total_size]) \
            reduction(+:sum)
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                int ii = i + Ng;
                int jj = j + Ng;
                sum += velocity_u_ptr_[jj * u_stride + ii];
            }
        }
        count = Nx * Ny;
    } else
#endif
    {
        // Host path
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                sum += velocity_.u(i, j);
                ++count;
            }
        }
    }

    return sum / count;
}

double RANSSolver::wall_shear_stress() const {
    // Compute du/dy at the bottom wall
    // Using one-sided difference from first interior cell to wall
    double sum = 0.0;
    int count = 0;
    
    [[maybe_unused]] const int Nx = mesh_->Nx;
    const int Ng = mesh_->Nghost;
    const int j_wall = Ng;  // First interior row
    const double y_cell = mesh_->y(j_wall);
    const double y_wall = mesh_->y_min;
    const double dist = y_cell - y_wall;
    
#ifdef USE_GPU_OFFLOAD
    if (gpu_ready_) {
        // GPU path: compute sum on device, only transfer scalar
        const size_t u_total_size = velocity_.u_total_size();
        const int u_stride = Nx + 2*Ng + 1;
        
        #pragma omp target teams distribute parallel for \
            map(present: velocity_u_ptr_[0:u_total_size]) \
            reduction(+:sum)
        for (int i = 0; i < Nx; ++i) {
            int ii = i + Ng;
            // u at wall is 0 (no-slip), so dudy = u[j_wall] / dist
            double dudy = velocity_u_ptr_[j_wall * u_stride + ii] / dist;
            sum += dudy;
        }
        count = Nx;
    } else
#endif
    {
        // Host path
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            // u at wall is 0 (no-slip)
            double dudy = velocity_.u(i, j_wall) / dist;
            sum += dudy;
            ++count;
        }
    }

    double dudy_avg = sum / count;
    return config_.nu * dudy_avg;  // tau_w = mu * du/dy = rho * nu * du/dy (rho=1)
}

double RANSSolver::friction_velocity() const {
    double tau_w = wall_shear_stress();
    return std::sqrt(std::abs(tau_w));  // u_tau = sqrt(tau_w / rho)
}

double RANSSolver::Re_tau() const {
    double u_tau = friction_velocity();
    double delta = (mesh_->y_max - mesh_->y_min) / 2.0;
    return u_tau * delta / config_.nu;
}

// ============================================================================
// NaN/Inf Guard: Abort immediately on non-finite values
// ============================================================================

void RANSSolver::check_for_nan_inf(int step) const {
    if (!config_.turb_guard_enabled) {
        return;  // Guard disabled in config
    }
    
    // Only check every guard_interval steps (performance)
    if (step % config_.turb_guard_interval != 0) {
        return;
    }
    
    bool all_finite = true;
    const bool has_transport = turb_model_ && turb_model_->uses_transport_equations();
    
#ifdef USE_GPU_OFFLOAD
    // GPU path: Do NaN/Inf check entirely on device, only transfer 1 scalar
    if (gpu_ready_) {
        int has_bad = 0;
        
        const size_t u_total = velocity_.u_total_size();
        const size_t v_total = velocity_.v_total_size();
        const size_t field_total = field_total_size_;
        
        // Check u-velocity (x-faces)
        #pragma omp target teams distribute parallel for \
            map(present: velocity_u_ptr_[0:u_total]) reduction(|: has_bad)
        for (size_t idx = 0; idx < u_total; ++idx) {
            const double x = velocity_u_ptr_[idx];
            // Use manual NaN/Inf check (x != x for NaN, or x-x != 0 for Inf)
            has_bad |= (x != x || (x - x) != 0.0) ? 1 : 0;
        }
        
        // Check v-velocity (y-faces)
        #pragma omp target teams distribute parallel for \
            map(present: velocity_v_ptr_[0:v_total]) reduction(|: has_bad)
        for (size_t idx = 0; idx < v_total; ++idx) {
            const double x = velocity_v_ptr_[idx];
            has_bad |= (x != x || (x - x) != 0.0) ? 1 : 0;
        }
        
        // Check pressure and eddy viscosity (cell-centered)
        #pragma omp target teams distribute parallel for \
            map(present: pressure_ptr_[0:field_total], nu_t_ptr_[0:field_total]) \
            reduction(|: has_bad)
        for (size_t idx = 0; idx < field_total; ++idx) {
            const double p = pressure_ptr_[idx];
            const double nut = nu_t_ptr_[idx];
            has_bad |= (p != p || (p - p) != 0.0 || nut != nut || (nut - nut) != 0.0) ? 1 : 0;
        }
        
        // Check transport variables if turbulence model uses them
        if (has_transport) {
            #pragma omp target teams distribute parallel for \
                map(present: k_ptr_[0:field_total], omega_ptr_[0:field_total]) \
                reduction(|: has_bad)
            for (size_t idx = 0; idx < field_total; ++idx) {
                const double k = k_ptr_[idx];
                const double w = omega_ptr_[idx];
                has_bad |= (k != k || (k - k) != 0.0 || w != w || (w - w) != 0.0) ? 1 : 0;
            }
        }
        
        all_finite = (has_bad == 0);
    } else
#endif
    {
        // CPU path: Check host-side fields directly (no GPU sync needed)
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                // Check velocity, pressure, nu_t
                double u = 0.5 * (velocity_.u(i, j) + velocity_.u(i+1, j));
                double v = 0.5 * (velocity_.v(i, j) + velocity_.v(i, j+1));
                double p = pressure_(i, j);
                double nu_t_val = nu_t_(i, j);
                
                if (!std::isfinite(u) || !std::isfinite(v) || 
                    !std::isfinite(p) || !std::isfinite(nu_t_val)) {
                    all_finite = false;
                    break;
                }
                
                // Check transport variables if applicable
                if (has_transport) {
                    double k_val = k_(i, j);
                    double omega_val = omega_(i, j);
                    
                    if (!std::isfinite(k_val) || !std::isfinite(omega_val)) {
                        all_finite = false;
                        break;
                    }
                }
            }
            if (!all_finite) break;
        }
    }
    
    // Abort immediately on non-finite values
    if (!all_finite) {
        std::cerr << "\n========================================\n";
        std::cerr << "NUMERICAL STABILITY GUARD: NaN/Inf DETECTED\n";
        std::cerr << "========================================\n";
        std::cerr << "Step: " << step << "\n";
        std::cerr << "\nOne or more fields contain NaN or Inf:\n";
        std::cerr << "  - Velocity (u, v)\n";
        std::cerr << "  - Pressure (p)\n";
        std::cerr << "  - Eddy viscosity (nu_t)\n";
        if (has_transport) {
            std::cerr << "  - Transport variables (k, omega)\n";
        }
        std::cerr << "\nThis indicates numerical instability.\n";
        std::cerr << "Aborting to prevent garbage propagation.\n";
        std::cerr << "\nPossible causes:\n";
        std::cerr << "  - Time step too large (reduce dt or enable adaptive_dt)\n";
        std::cerr << "  - Turbulence model incompatible with flow regime\n";
        std::cerr << "  - Mesh resolution insufficient\n";
        std::cerr << "  - Boundary conditions inconsistent\n";
        std::cerr << "========================================\n";
        throw std::runtime_error("NaN/Inf detected in solution fields");
    }
}

void RANSSolver::print_velocity_profile(double x_loc) const {
    // Find i index closest to x_loc
    int i_loc = mesh_->i_begin();
    double min_dist = std::abs(mesh_->x(i_loc) - x_loc);
    
    for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
        double dist = std::abs(mesh_->x(i) - x_loc);
        if (dist < min_dist) {
            min_dist = dist;
            i_loc = i;
        }
    }
    
    std::cout << "\nVelocity profile at x = " << mesh_->x(i_loc) << ":\n";
    std::cout << std::setw(12) << "y" << std::setw(12) << "u" << std::setw(12) << "v" << "\n";
    
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        std::cout << std::setw(12) << std::fixed << std::setprecision(4) << mesh_->y(j)
                  << std::setw(12) << velocity_.u(i_loc, j)
                  << std::setw(12) << velocity_.v(i_loc, j)
                  << "\n";
    }
}

void RANSSolver::write_fields(const std::string& prefix) const {
#ifdef USE_GPU_OFFLOAD
    // Download solution fields from GPU before writing
    const_cast<RANSSolver*>(this)->sync_solution_from_gpu();
    // Transport fields only if turbulence model active
    if (turb_model_ && turb_model_->uses_transport_equations()) {
        const_cast<RANSSolver*>(this)->sync_transport_from_gpu();
    }
#endif
    
    velocity_.write(prefix + "_velocity.dat");
    pressure_.write(prefix + "_pressure.dat");

    if (turb_model_) {
        nu_t_.write(prefix + "_nu_t.dat");
    }
}

double RANSSolver::compute_adaptive_dt() const {
    [[maybe_unused]] const int Nx = mesh_->Nx;
    [[maybe_unused]] const int Ny = mesh_->Ny;
    [[maybe_unused]] const int Ng = mesh_->Nghost;
    const double nu = config_.nu;
    
#ifdef USE_GPU_OFFLOAD
    // GPU path: compute both CFL and diffusive constraints on device with reductions
    // This avoids expensive device→host transfers every iteration
    
    double u_max = 1e-10;
    double nu_eff_max = nu;
    
    const size_t u_total_size = velocity_.u_total_size();
    const size_t v_total_size = velocity_.v_total_size();
    const size_t field_total_size = field_total_size_;
    const int u_stride = Nx + 2*Ng + 1;
    const int v_stride = Nx + 2*Ng;
    const int stride = Nx + 2*Ng;
    
    // Compute max velocity magnitude (for advective CFL) on GPU
    #pragma omp target teams distribute parallel for collapse(2) \
        map(present: velocity_u_ptr_[0:u_total_size], velocity_v_ptr_[0:v_total_size]) \
        reduction(max:u_max)
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            int ii = i + Ng;
            int jj = j + Ng;
            // Interpolate u and v to cell center for staggered grid
            double u_avg = 0.5 * (velocity_u_ptr_[jj * u_stride + ii] + 
                                  velocity_u_ptr_[jj * u_stride + ii + 1]);
            double v_avg = 0.5 * (velocity_v_ptr_[jj * v_stride + ii] + 
                                  velocity_v_ptr_[(jj + 1) * v_stride + ii]);
            double u_mag = sqrt(u_avg*u_avg + v_avg*v_avg);
            if (u_mag > u_max) u_max = u_mag;
        }
    }
    
    // Compute max effective viscosity (for diffusive CFL) on GPU if turbulence active
    if (turb_model_) {
        #pragma omp target teams distribute parallel for collapse(2) \
            map(present: nu_t_ptr_[0:field_total_size]) \
            reduction(max:nu_eff_max)
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                int ii = i + Ng;
                int jj = j + Ng;
                int idx = jj * stride + ii;
                double nu_eff = nu + nu_t_ptr_[idx];
                if (nu_eff > nu_eff_max) nu_eff_max = nu_eff;
            }
        }
    }
    
#else
    // Host path: original host-side computation
    double u_max = 1e-10;
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            double u_mag = velocity_.magnitude(i, j);
            u_max = std::max(u_max, u_mag);
        }
    }
    
    double nu_eff_max = nu;
    if (turb_model_) {
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                nu_eff_max = std::max(nu_eff_max, nu + nu_t_(i, j));
            }
        }
    }
#endif
    
    // Compute time step constraints (same for GPU and CPU)
    double dx_min = std::min(mesh_->dx, mesh_->dy);
    double dt_cfl = config_.CFL_max * dx_min / u_max;
    
    // Diffusive stability: dt < 0.25 * dx² / ν (hard limit from von Neumann analysis)
    // NOTE: Do NOT scale by CFL_max - this is a stability constant, not a tuning parameter
    double dt_diff = 0.25 * dx_min * dx_min / nu_eff_max;
    
    return std::min(dt_cfl, dt_diff);
}

void RANSSolver::write_vtk(const std::string& filename) const {
    // NaN/Inf GUARD: Check before writing output
    // Catch NaNs before they're written to files
    check_for_nan_inf(step_count_);

#ifdef USE_GPU_OFFLOAD
    // Download solution fields from GPU for I/O (only what's needed!)
    const_cast<RANSSolver*>(this)->sync_solution_from_gpu();
    // Transport fields only if they'll be written (turbulence model active)
    if (turb_model_ && turb_model_->uses_transport_equations()) {
        const_cast<RANSSolver*>(this)->sync_transport_from_gpu();
    }
#endif

    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error: Cannot open " << filename << " for writing\n";
        return;
    }

    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const bool is_2d = mesh_->is2D();

    // VTK header
    file << "# vtk DataFile Version 3.0\n";
    file << "RANS simulation output\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_POINTS\n";

    if (is_2d) {
        file << "DIMENSIONS " << Nx << " " << Ny << " 1\n";
        file << "ORIGIN " << mesh_->x_min << " " << mesh_->y_min << " 0\n";
        file << "SPACING " << mesh_->dx << " " << mesh_->dy << " 1\n";
        file << "POINT_DATA " << Nx * Ny << "\n";

        // Velocity vector field (interpolated from staggered grid to cell centers)
        // Use 2-component SCALARS for 2D (VTK VECTORS requires 3 components)
        file << "SCALARS velocity double 2\n";
        file << "LOOKUP_TABLE default\n";
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                double u_center = velocity_.u_center(i, j);
                double v_center = velocity_.v_center(i, j);
                file << u_center << " " << v_center << "\n";
            }
        }

        // Pressure scalar field
        file << "SCALARS pressure double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                file << pressure_(i, j) << "\n";
            }
        }

        // Velocity magnitude
        file << "SCALARS velocity_magnitude double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                file << velocity_.magnitude(i, j) << "\n";
            }
        }

        // Eddy viscosity (if turbulence model is active)
        if (turb_model_) {
            file << "SCALARS nu_t double 1\n";
            file << "LOOKUP_TABLE default\n";
            for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                    file << nu_t_(i, j) << "\n";
                }
            }
        }

        // Individual velocity components as scalars
        file << "SCALARS u double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                file << velocity_.u_center(i, j) << "\n";
            }
        }

        file << "SCALARS v double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                file << velocity_.v_center(i, j) << "\n";
            }
        }

        // Pressure gradients using central differences
        // For periodic BCs: wrap indices; for non-periodic: one-sided at boundaries
        const bool periodic_x = (poisson_bc_x_lo_ == PoissonBC::Periodic);
        const bool periodic_y = (poisson_bc_y_lo_ == PoissonBC::Periodic);

        // Helper lambdas for pressure gradient computation
        auto compute_dpdx_2d = [&](int i, int j) -> double {
            if (periodic_x) {
                int im = (i == mesh_->i_begin()) ? mesh_->i_end() - 1 : i - 1;
                int ip = (i == mesh_->i_end() - 1) ? mesh_->i_begin() : i + 1;
                return (pressure_(ip, j) - pressure_(im, j)) / (2.0 * mesh_->dx);
            } else {
                if (i == mesh_->i_begin()) {
                    return (pressure_(i + 1, j) - pressure_(i, j)) / mesh_->dx;
                } else if (i == mesh_->i_end() - 1) {
                    return (pressure_(i, j) - pressure_(i - 1, j)) / mesh_->dx;
                } else {
                    return (pressure_(i + 1, j) - pressure_(i - 1, j)) / (2.0 * mesh_->dx);
                }
            }
        };

        auto compute_dpdy_2d = [&](int i, int j) -> double {
            if (periodic_y) {
                int jm = (j == mesh_->j_begin()) ? mesh_->j_end() - 1 : j - 1;
                int jp = (j == mesh_->j_end() - 1) ? mesh_->j_begin() : j + 1;
                return (pressure_(i, jp) - pressure_(i, jm)) / (2.0 * mesh_->dy);
            } else {
                if (j == mesh_->j_begin()) {
                    return (pressure_(i, j + 1) - pressure_(i, j)) / mesh_->dy;
                } else if (j == mesh_->j_end() - 1) {
                    return (pressure_(i, j) - pressure_(i, j - 1)) / mesh_->dy;
                } else {
                    return (pressure_(i, j + 1) - pressure_(i, j - 1)) / (2.0 * mesh_->dy);
                }
            }
        };

        // Pressure gradient as 2-component vector
        file << "SCALARS pressure_gradient double 2\n";
        file << "LOOKUP_TABLE default\n";
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                file << compute_dpdx_2d(i, j) << " " << compute_dpdy_2d(i, j) << "\n";
            }
        }

        // Pressure gradient components as scalars
        file << "SCALARS dP_dx double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                file << compute_dpdx_2d(i, j) << "\n";
            }
        }

        file << "SCALARS dP_dy double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                file << compute_dpdy_2d(i, j) << "\n";
            }
        }
    } else {
        // 3D output
        file << "DIMENSIONS " << Nx << " " << Ny << " " << Nz << "\n";
        file << "ORIGIN " << mesh_->x_min << " " << mesh_->y_min << " " << mesh_->z_min << "\n";
        file << "SPACING " << mesh_->dx << " " << mesh_->dy << " " << mesh_->dz << "\n";
        file << "POINT_DATA " << Nx * Ny * Nz << "\n";

        // Velocity vector field (interpolated from staggered grid to cell centers)
        file << "VECTORS velocity double\n";
        for (int k = mesh_->k_begin(); k < mesh_->k_end(); ++k) {
            for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                    double u_center = velocity_.u_center(i, j, k);
                    double v_center = velocity_.v_center(i, j, k);
                    double w_center = velocity_.w_center(i, j, k);
                    file << u_center << " " << v_center << " " << w_center << "\n";
                }
            }
        }

        // Pressure scalar field
        file << "SCALARS pressure double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int k = mesh_->k_begin(); k < mesh_->k_end(); ++k) {
            for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                    file << pressure_(i, j, k) << "\n";
                }
            }
        }

        // Velocity magnitude
        file << "SCALARS velocity_magnitude double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int k = mesh_->k_begin(); k < mesh_->k_end(); ++k) {
            for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                    file << velocity_.magnitude(i, j, k) << "\n";
                }
            }
        }

        // Eddy viscosity (if turbulence model is active)
        if (turb_model_) {
            file << "SCALARS nu_t double 1\n";
            file << "LOOKUP_TABLE default\n";
            for (int k = mesh_->k_begin(); k < mesh_->k_end(); ++k) {
                for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                    for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                        file << nu_t_(i, j, k) << "\n";
                    }
                }
            }
        }

        // Individual velocity components as scalars
        file << "SCALARS u double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int k = mesh_->k_begin(); k < mesh_->k_end(); ++k) {
            for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                    file << velocity_.u_center(i, j, k) << "\n";
                }
            }
        }

        file << "SCALARS v double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int k = mesh_->k_begin(); k < mesh_->k_end(); ++k) {
            for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                    file << velocity_.v_center(i, j, k) << "\n";
                }
            }
        }

        file << "SCALARS w double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int k = mesh_->k_begin(); k < mesh_->k_end(); ++k) {
            for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                    file << velocity_.w_center(i, j, k) << "\n";
                }
            }
        }

        // Pressure gradients using central differences
        // For periodic BCs: wrap indices; for non-periodic: one-sided at boundaries
        const bool periodic_x = (poisson_bc_x_lo_ == PoissonBC::Periodic);
        const bool periodic_y = (poisson_bc_y_lo_ == PoissonBC::Periodic);
        const bool periodic_z = (poisson_bc_z_lo_ == PoissonBC::Periodic);

        // Helper lambdas for pressure gradient computation
        auto compute_dpdx_3d = [&](int i, int j, int k) -> double {
            if (periodic_x) {
                int im = (i == mesh_->i_begin()) ? mesh_->i_end() - 1 : i - 1;
                int ip = (i == mesh_->i_end() - 1) ? mesh_->i_begin() : i + 1;
                return (pressure_(ip, j, k) - pressure_(im, j, k)) / (2.0 * mesh_->dx);
            } else {
                if (i == mesh_->i_begin()) {
                    return (pressure_(i + 1, j, k) - pressure_(i, j, k)) / mesh_->dx;
                } else if (i == mesh_->i_end() - 1) {
                    return (pressure_(i, j, k) - pressure_(i - 1, j, k)) / mesh_->dx;
                } else {
                    return (pressure_(i + 1, j, k) - pressure_(i - 1, j, k)) / (2.0 * mesh_->dx);
                }
            }
        };

        auto compute_dpdy_3d = [&](int i, int j, int k) -> double {
            if (periodic_y) {
                int jm = (j == mesh_->j_begin()) ? mesh_->j_end() - 1 : j - 1;
                int jp = (j == mesh_->j_end() - 1) ? mesh_->j_begin() : j + 1;
                return (pressure_(i, jp, k) - pressure_(i, jm, k)) / (2.0 * mesh_->dy);
            } else {
                if (j == mesh_->j_begin()) {
                    return (pressure_(i, j + 1, k) - pressure_(i, j, k)) / mesh_->dy;
                } else if (j == mesh_->j_end() - 1) {
                    return (pressure_(i, j, k) - pressure_(i, j - 1, k)) / mesh_->dy;
                } else {
                    return (pressure_(i, j + 1, k) - pressure_(i, j - 1, k)) / (2.0 * mesh_->dy);
                }
            }
        };

        auto compute_dpdz_3d = [&](int i, int j, int k) -> double {
            if (periodic_z) {
                int km = (k == mesh_->k_begin()) ? mesh_->k_end() - 1 : k - 1;
                int kp = (k == mesh_->k_end() - 1) ? mesh_->k_begin() : k + 1;
                return (pressure_(i, j, kp) - pressure_(i, j, km)) / (2.0 * mesh_->dz);
            } else {
                if (k == mesh_->k_begin()) {
                    return (pressure_(i, j, k + 1) - pressure_(i, j, k)) / mesh_->dz;
                } else if (k == mesh_->k_end() - 1) {
                    return (pressure_(i, j, k) - pressure_(i, j, k - 1)) / mesh_->dz;
                } else {
                    return (pressure_(i, j, k + 1) - pressure_(i, j, k - 1)) / (2.0 * mesh_->dz);
                }
            }
        };

        // Pressure gradient as vector
        file << "VECTORS pressure_gradient double\n";
        for (int k = mesh_->k_begin(); k < mesh_->k_end(); ++k) {
            for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                    file << compute_dpdx_3d(i, j, k) << " "
                         << compute_dpdy_3d(i, j, k) << " "
                         << compute_dpdz_3d(i, j, k) << "\n";
                }
            }
        }

        // Pressure gradient components as scalars
        file << "SCALARS dP_dx double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int k = mesh_->k_begin(); k < mesh_->k_end(); ++k) {
            for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                    file << compute_dpdx_3d(i, j, k) << "\n";
                }
            }
        }

        file << "SCALARS dP_dy double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int k = mesh_->k_begin(); k < mesh_->k_end(); ++k) {
            for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                    file << compute_dpdy_3d(i, j, k) << "\n";
                }
            }
        }

        file << "SCALARS dP_dz double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int k = mesh_->k_begin(); k < mesh_->k_end(); ++k) {
            for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                    file << compute_dpdz_3d(i, j, k) << "\n";
                }
            }
        }
    }

    file.close();
}

#ifdef USE_GPU_OFFLOAD
void RANSSolver::initialize_gpu_buffers() {
    // Verify GPU is available (throws if not)
    gpu::verify_device_available();
    
    // Get raw pointers to all field data
    field_total_size_ = mesh_->total_cells();  // (Nx + 2*Nghost) * (Ny + 2*Nghost) for cell-centered fields
    
    // Staggered grid: u and v have different sizes
    velocity_u_ptr_ = velocity_.u_data().data();
    velocity_v_ptr_ = velocity_.v_data().data();
    velocity_star_u_ptr_ = velocity_star_.u_data().data();
    velocity_star_v_ptr_ = velocity_star_.v_data().data();
    pressure_ptr_ = pressure_.data().data();
    pressure_corr_ptr_ = pressure_correction_.data().data();
    nu_t_ptr_ = nu_t_.data().data();  // Keep for nu_eff calculation
    nu_eff_ptr_ = nu_eff_.data().data();
    conv_u_ptr_ = conv_.u_data().data();
    conv_v_ptr_ = conv_.v_data().data();
    diff_u_ptr_ = diff_.u_data().data();
    diff_v_ptr_ = diff_.v_data().data();
    rhs_poisson_ptr_ = rhs_poisson_.data().data();
    div_velocity_ptr_ = div_velocity_.data().data();

    // 3D w-velocity fields
    if (!mesh_->is2D()) {
        velocity_w_ptr_ = velocity_.w_data().data();
        velocity_star_w_ptr_ = velocity_star_.w_data().data();
        conv_w_ptr_ = conv_.w_data().data();
        diff_w_ptr_ = diff_.w_data().data();
    }
    // NOTE: k and omega are NOT mapped - turbulence models manage their own GPU copies
    k_ptr_ = k_.data().data();
    omega_ptr_ = omega_.data().data();
    
    // Reynolds stress tensor components (for EARSM/TBNN)
    tau_xx_ptr_ = tau_ij_.xx_data().data();
    tau_xy_ptr_ = tau_ij_.xy_data().data();
    tau_yy_ptr_ = tau_ij_.yy_data().data();
    
    // Gradient scratch buffers for turbulence models
    dudx_ptr_ = dudx_.data().data();
    dudy_ptr_ = dudy_.data().data();
    dvdx_ptr_ = dvdx_.data().data();
    dvdy_ptr_ = dvdy_.data().data();
    wall_distance_ptr_ = wall_distance_.data().data();
    
#ifdef GPU_PROFILE_TRANSFERS
    auto transfer_start = std::chrono::steady_clock::now();
#endif
    
    // Fail fast if no GPU device available (GPU build requires GPU)
    gpu::verify_device_available();
    
    // Map all arrays to GPU device and copy initial values
    // Using map(to:) instead of map(alloc:) to transfer initialized data
    // Data will persist on GPU for entire solver lifetime
    // Staggered grid: u and v have different sizes
    const size_t u_total_size = velocity_.u_total_size();
    const size_t v_total_size = velocity_.v_total_size();

    #pragma omp target enter data map(to: velocity_u_ptr_[0:u_total_size])
    #pragma omp target enter data map(to: velocity_v_ptr_[0:v_total_size])
    #pragma omp target enter data map(to: velocity_star_u_ptr_[0:u_total_size])
    #pragma omp target enter data map(to: velocity_star_v_ptr_[0:v_total_size])
    #pragma omp target enter data map(to: pressure_ptr_[0:field_total_size_])
    #pragma omp target enter data map(to: pressure_corr_ptr_[0:field_total_size_])
    #pragma omp target enter data map(to: nu_t_ptr_[0:field_total_size_])  // Keep for nu_eff = nu + nu_t
    #pragma omp target enter data map(to: nu_eff_ptr_[0:field_total_size_])
    #pragma omp target enter data map(to: conv_u_ptr_[0:u_total_size])
    #pragma omp target enter data map(to: conv_v_ptr_[0:v_total_size])
    #pragma omp target enter data map(to: diff_u_ptr_[0:u_total_size])
    #pragma omp target enter data map(to: diff_v_ptr_[0:v_total_size])
    #pragma omp target enter data map(to: rhs_poisson_ptr_[0:field_total_size_])
    #pragma omp target enter data map(to: div_velocity_ptr_[0:field_total_size_])

    // 3D w-velocity fields
    if (!mesh_->is2D()) {
        const size_t w_total_size = velocity_.w_total_size();
        #pragma omp target enter data map(to: velocity_w_ptr_[0:w_total_size])
        #pragma omp target enter data map(to: velocity_star_w_ptr_[0:w_total_size])
        #pragma omp target enter data map(to: conv_w_ptr_[0:w_total_size])
        #pragma omp target enter data map(to: diff_w_ptr_[0:w_total_size])
    }
    
    // Transport equation fields (k, omega) - needed for EARSM/SST models
    // These are initialized by RANSSolver::initialize() before GPU buffers are set up,
    // so we upload them with map(to:). They'll be updated by transport models on GPU.
    #pragma omp target enter data map(to: k_ptr_[0:field_total_size_])
    #pragma omp target enter data map(to: omega_ptr_[0:field_total_size_])
    
    // Reynolds stress tensor components (alloc - will be computed by EARSM/TBNN on GPU)
    #pragma omp target enter data map(alloc: tau_xx_ptr_[0:field_total_size_])
    #pragma omp target enter data map(alloc: tau_xy_ptr_[0:field_total_size_])
    #pragma omp target enter data map(alloc: tau_yy_ptr_[0:field_total_size_])
    
    // Gradient scratch buffers for turbulence models (to, not alloc - need zero init)
    // These must be initialized to zero to prevent NaN propagation in EARSM models
    // on the first timestep before compute_gradients_from_mac_gpu runs
    #pragma omp target enter data map(to: dudx_ptr_[0:field_total_size_])
    #pragma omp target enter data map(to: dudy_ptr_[0:field_total_size_])
    #pragma omp target enter data map(to: dvdx_ptr_[0:field_total_size_])
    #pragma omp target enter data map(to: dvdy_ptr_[0:field_total_size_])
    #pragma omp target enter data map(to: wall_distance_ptr_[0:field_total_size_])  // Precomputed, upload once
    
    // Allocate device-resident "old velocity" buffers for residual computation
    // Host storage exists but is never used - device copy is authoritative
    // This eliminates per-step H→D upload for residual computation
    velocity_old_u_ptr_ = velocity_old_.u_data().data();
    velocity_old_v_ptr_ = velocity_old_.v_data().data();

    #pragma omp target enter data map(alloc: velocity_old_u_ptr_[0:u_total_size])
    #pragma omp target enter data map(alloc: velocity_old_v_ptr_[0:v_total_size])

    // 3D old velocity
    if (!mesh_->is2D()) {
        velocity_old_w_ptr_ = velocity_old_.w_data().data();
        const size_t w_total_size = velocity_.w_total_size();
        #pragma omp target enter data map(alloc: velocity_old_w_ptr_[0:w_total_size])
    }

    // Zero-initialize device-only arrays to prevent garbage in first residual computation
    // Arrays allocated with map(alloc:) contain garbage until explicitly written
    #pragma omp target teams distribute parallel for map(present: velocity_old_u_ptr_[0:u_total_size])
    for (size_t i = 0; i < u_total_size; ++i) velocity_old_u_ptr_[i] = 0.0;

    #pragma omp target teams distribute parallel for map(present: velocity_old_v_ptr_[0:v_total_size])
    for (size_t i = 0; i < v_total_size; ++i) velocity_old_v_ptr_[i] = 0.0;

    if (!mesh_->is2D()) {
        const size_t w_total_size = velocity_.w_total_size();
        #pragma omp target teams distribute parallel for map(present: velocity_old_w_ptr_[0:w_total_size])
        for (size_t i = 0; i < w_total_size; ++i) velocity_old_w_ptr_[i] = 0.0;
    }

    // Zero-initialize Reynolds stress tensor components
    #pragma omp target teams distribute parallel for map(present: tau_xx_ptr_[0:field_total_size_])
    for (size_t i = 0; i < field_total_size_; ++i) tau_xx_ptr_[i] = 0.0;

    #pragma omp target teams distribute parallel for map(present: tau_xy_ptr_[0:field_total_size_])
    for (size_t i = 0; i < field_total_size_; ++i) tau_xy_ptr_[i] = 0.0;

    #pragma omp target teams distribute parallel for map(present: tau_yy_ptr_[0:field_total_size_])
    for (size_t i = 0; i < field_total_size_; ++i) tau_yy_ptr_[i] = 0.0;

    // Verify mappings succeeded (fail fast if GPU unavailable despite num_devices>0)
    if (!gpu::is_pointer_present(velocity_u_ptr_)) {
        throw std::runtime_error("GPU mapping failed despite device availability");
    }
    
    gpu_ready_ = true;
    
#ifdef GPU_PROFILE_TRANSFERS
    auto transfer_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> transfer_time = transfer_end - transfer_start;
    double mb_transferred = 16 * field_total_size_ * sizeof(double) / 1024.0 / 1024.0;
    double bandwidth = mb_transferred / transfer_time.count();
    (void)mb_transferred;
    (void)bandwidth;
#endif
}

void RANSSolver::cleanup_gpu_buffers() {
    assert(gpu_ready_ && "GPU must be initialized before cleanup");
    
    // Staggered grid sizes
    const size_t u_total_size = velocity_.u_total_size();
    const size_t v_total_size = velocity_.v_total_size();
    
    // Copy final results back from GPU, then free device memory
    // Using map(from:) to get final state back to host
    #pragma omp target exit data map(from: velocity_u_ptr_[0:u_total_size])
    #pragma omp target exit data map(from: velocity_v_ptr_[0:v_total_size])
    #pragma omp target exit data map(from: pressure_ptr_[0:field_total_size_])
    #pragma omp target exit data map(from: nu_t_ptr_[0:field_total_size_])

    // 3D w-velocity results
    if (!mesh_->is2D()) {
        const size_t w_total_size = velocity_.w_total_size();
        #pragma omp target exit data map(from: velocity_w_ptr_[0:w_total_size])
    }
    // k and omega are managed by turbulence model, not unmapped here

    // Delete temporary/work arrays without copying back
    #pragma omp target exit data map(delete: velocity_star_u_ptr_[0:u_total_size])
    #pragma omp target exit data map(delete: velocity_star_v_ptr_[0:v_total_size])
    #pragma omp target exit data map(delete: velocity_old_u_ptr_[0:u_total_size])
    #pragma omp target exit data map(delete: velocity_old_v_ptr_[0:v_total_size])
    #pragma omp target exit data map(delete: pressure_corr_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: nu_eff_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: conv_u_ptr_[0:u_total_size])
    #pragma omp target exit data map(delete: conv_v_ptr_[0:v_total_size])
    #pragma omp target exit data map(delete: diff_u_ptr_[0:u_total_size])
    #pragma omp target exit data map(delete: diff_v_ptr_[0:v_total_size])
    #pragma omp target exit data map(delete: rhs_poisson_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: div_velocity_ptr_[0:field_total_size_])

    // 3D temporary arrays
    if (!mesh_->is2D()) {
        const size_t w_total_size = velocity_.w_total_size();
        #pragma omp target exit data map(delete: velocity_star_w_ptr_[0:w_total_size])
        #pragma omp target exit data map(delete: velocity_old_w_ptr_[0:w_total_size])
        #pragma omp target exit data map(delete: conv_w_ptr_[0:w_total_size])
        #pragma omp target exit data map(delete: diff_w_ptr_[0:w_total_size])
    }
    
    // Delete gradient scratch buffers
    #pragma omp target exit data map(delete: dudx_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: dudy_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: dvdx_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: dvdy_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: wall_distance_ptr_[0:field_total_size_])
    
    // Delete transport fields
    #pragma omp target exit data map(delete: k_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: omega_ptr_[0:field_total_size_])
    
    // Delete Reynolds stress tensor buffers
    #pragma omp target exit data map(delete: tau_xx_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: tau_xy_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: tau_yy_ptr_[0:field_total_size_])
    
    gpu_ready_ = false;
}

void RANSSolver::sync_to_gpu() {
    assert(gpu_ready_ && "GPU must be initialized before sync");
    
    // Update GPU with changed fields (typically after CPU-side modifications)
    // Staggered grid: u and v have different sizes
    const size_t u_total_size = velocity_.u_total_size();
    const size_t v_total_size = velocity_.v_total_size();

    #pragma omp target update to(velocity_u_ptr_[0:u_total_size])
    #pragma omp target update to(velocity_v_ptr_[0:v_total_size])
    #pragma omp target update to(pressure_ptr_[0:field_total_size_])
    #pragma omp target update to(nu_t_ptr_[0:field_total_size_])

    // 3D w-velocity
    if (!mesh_->is2D()) {
        const size_t w_total_size = velocity_.w_total_size();
        #pragma omp target update to(velocity_w_ptr_[0:w_total_size])
    }

    // Upload k and omega if turbulence model uses transport equations
    // These are initialized by RANSSolver::initialize() after GPU buffers are allocated
    if (turb_model_ && turb_model_->uses_transport_equations()) {
        #pragma omp target update to(k_ptr_[0:field_total_size_])
        #pragma omp target update to(omega_ptr_[0:field_total_size_])
    }
}

void RANSSolver::sync_from_gpu() {
    // Legacy sync for backward compatibility - downloads everything
    // Prefer using sync_solution_from_gpu() and sync_transport_from_gpu() selectively
    sync_solution_from_gpu();
    sync_transport_from_gpu();
}

void RANSSolver::sync_solution_from_gpu() {
    assert(gpu_ready_ && "GPU must be initialized before sync");
    
    // Download only primary solution fields needed for I/O/analysis
    // Staggered grid: u and v have different sizes
    const size_t u_total_size = velocity_.u_total_size();
    const size_t v_total_size = velocity_.v_total_size();

    #pragma omp target update from(velocity_u_ptr_[0:u_total_size])
    #pragma omp target update from(velocity_v_ptr_[0:v_total_size])
    #pragma omp target update from(pressure_ptr_[0:field_total_size_])
    #pragma omp target update from(nu_t_ptr_[0:field_total_size_])

    // 3D w-velocity
    if (!mesh_->is2D()) {
        const size_t w_total_size = velocity_.w_total_size();
        #pragma omp target update from(velocity_w_ptr_[0:w_total_size])
    }
}

void RANSSolver::sync_transport_from_gpu() {
    assert(gpu_ready_ && "GPU must be initialized before sync");
    
    // Download transport equation fields (k, omega) only if turbulence model uses them
    // For laminar runs (turb_model = none), this saves hundreds of MB on large grids!
    if (turb_model_ && turb_model_->uses_transport_equations()) {
        #pragma omp target update from(k_ptr_[0:field_total_size_])
        #pragma omp target update from(omega_ptr_[0:field_total_size_])
    }
}

TurbulenceDeviceView RANSSolver::get_device_view() const {
    assert(gpu_ready_ && "GPU must be initialized to get device view");
    
    TurbulenceDeviceView view;
    
    // Velocity field (staggered, solver-owned, persistent on GPU)
    view.u_face = velocity_u_ptr_;
    view.v_face = velocity_v_ptr_;
    view.u_stride = velocity_.u_stride();
    view.v_stride = velocity_.v_stride();
    
    // Turbulence fields (cell-centered)
    view.k = k_ptr_;
    view.omega = omega_ptr_;
    view.nu_t = nu_t_ptr_;
    view.cell_stride = mesh_->total_Nx();  // Stride for cell-centered fields
    
    // Reynolds stress tensor
    view.tau_xx = tau_xx_ptr_;
    view.tau_xy = tau_xy_ptr_;
    view.tau_yy = tau_yy_ptr_;
    
    // Gradient scratch buffers
    view.dudx = dudx_ptr_;
    view.dudy = dudy_ptr_;
    view.dvdx = dvdx_ptr_;
    view.dvdy = dvdy_ptr_;
    
    // Wall distance
    view.wall_distance = wall_distance_ptr_;
    
    // Mesh parameters
    view.Nx = mesh_->Nx;
    view.Ny = mesh_->Ny;
    view.Ng = mesh_->Nghost;
    view.dx = mesh_->dx;
    view.dy = mesh_->dy;
    view.delta = (turb_model_ ? turb_model_->delta() : 1.0);
    
    return view;
}

SolverDeviceView RANSSolver::get_solver_view() const {
    SolverDeviceView view;

#ifdef USE_GPU_OFFLOAD
    assert(gpu_ready_ && "GPU must be initialized to get solver view");

    // GPU path: return device-present pointers
    view.u_face = velocity_u_ptr_;
    view.v_face = velocity_v_ptr_;
    view.u_star_face = velocity_star_u_ptr_;
    view.v_star_face = velocity_star_v_ptr_;
    view.u_old_face = velocity_old_u_ptr_;
    view.v_old_face = velocity_old_v_ptr_;
    view.u_stride = velocity_.u_stride();
    view.v_stride = velocity_.v_stride();

    // Initialize 3D fields to avoid undefined behavior in 2D mode
    view.w_face = nullptr;
    view.w_star_face = nullptr;
    view.w_old_face = nullptr;
    view.w_stride = 0;
    view.u_plane_stride = 0;
    view.v_plane_stride = 0;
    view.w_plane_stride = 0;

    // 3D velocity fields (overwrite defaults if 3D)
    if (!mesh_->is2D()) {
        view.w_face = velocity_w_ptr_;
        view.w_star_face = velocity_star_w_ptr_;
        view.w_old_face = velocity_old_w_ptr_;
        view.w_stride = velocity_.w_stride();
        view.u_plane_stride = velocity_.u_plane_stride();
        view.v_plane_stride = velocity_.v_plane_stride();
        view.w_plane_stride = velocity_.w_plane_stride();
    }

    view.p = pressure_ptr_;
    view.p_corr = pressure_corr_ptr_;
    view.nu_t = nu_t_ptr_;
    view.nu_eff = nu_eff_ptr_;
    view.rhs = rhs_poisson_ptr_;
    view.div = div_velocity_ptr_;
    view.cell_stride = mesh_->total_Nx();
    view.cell_plane_stride = mesh_->total_Nx() * mesh_->total_Ny();

    view.conv_u = conv_u_ptr_;
    view.conv_v = conv_v_ptr_;
    view.diff_u = diff_u_ptr_;
    view.diff_v = diff_v_ptr_;

    // Initialize 3D work arrays to avoid undefined behavior in 2D mode
    view.conv_w = nullptr;
    view.diff_w = nullptr;

    // 3D work arrays (overwrite defaults if 3D)
    if (!mesh_->is2D()) {
        view.conv_w = conv_w_ptr_;
        view.diff_w = diff_w_ptr_;
    }

    view.Nx = mesh_->Nx;
    view.Ny = mesh_->Ny;
    view.Nz = mesh_->Nz;
    view.Ng = mesh_->Nghost;
    view.dx = mesh_->dx;
    view.dy = mesh_->dy;
    view.dz = mesh_->dz;
    view.dt = current_dt_;
#else
    // CPU build: always return host pointers
    view.u_face = const_cast<double*>(velocity_.u_data().data());
    view.v_face = const_cast<double*>(velocity_.v_data().data());
    view.u_star_face = const_cast<double*>(velocity_star_.u_data().data());
    view.v_star_face = const_cast<double*>(velocity_star_.v_data().data());
    view.u_old_face = const_cast<double*>(velocity_old_.u_data().data());
    view.v_old_face = const_cast<double*>(velocity_old_.v_data().data());
    view.u_stride = velocity_.u_stride();
    view.v_stride = velocity_.v_stride();
    
    view.p = const_cast<double*>(pressure_.data().data());
    view.p_corr = const_cast<double*>(pressure_correction_.data().data());
    view.nu_t = const_cast<double*>(nu_t_.data().data());
    view.nu_eff = const_cast<double*>(nu_eff_.data().data());
    view.rhs = const_cast<double*>(rhs_poisson_.data().data());
    view.div = const_cast<double*>(div_velocity_.data().data());
    view.cell_stride = mesh_->total_Nx();
    
    view.conv_u = const_cast<double*>(conv_.u_data().data());
    view.conv_v = const_cast<double*>(conv_.v_data().data());
    view.diff_u = const_cast<double*>(diff_.u_data().data());
    view.diff_v = const_cast<double*>(diff_.v_data().data());
    
    view.Nx = mesh_->Nx;
    view.Ny = mesh_->Ny;
    view.Ng = mesh_->Nghost;
    view.dx = mesh_->dx;
    view.dy = mesh_->dy;
    view.dt = current_dt_;
#endif
    
    return view;
}
#else
// No-op implementations when GPU offloading is disabled
void RANSSolver::initialize_gpu_buffers() {
    gpu_ready_ = false;
}

void RANSSolver::cleanup_gpu_buffers() {
    // No-op
}

void RANSSolver::sync_to_gpu() {
    // No-op
}

void RANSSolver::sync_from_gpu() {
    // No-op
}

void RANSSolver::sync_solution_from_gpu() {
    // No-op
}

void RANSSolver::sync_transport_from_gpu() {
    // No-op
}

TurbulenceDeviceView RANSSolver::get_device_view() const {
    // CPU build: return host pointers (following get_solver_view() pattern)
    TurbulenceDeviceView view;

    // Velocity field (staggered, use same pattern as get_solver_view)
    view.u_face = const_cast<double*>(velocity_.u_data().data());
    view.v_face = const_cast<double*>(velocity_.v_data().data());
    view.u_stride = velocity_.u_stride();
    view.v_stride = velocity_.v_stride();

    // Turbulence fields (cell-centered)
    view.k = const_cast<double*>(k_.data().data());
    view.omega = const_cast<double*>(omega_.data().data());
    view.nu_t = const_cast<double*>(nu_t_.data().data());
    view.cell_stride = mesh_->total_Nx();

    // Reynolds stress tensor
    view.tau_xx = const_cast<double*>(tau_ij_.xx_data().data());
    view.tau_xy = const_cast<double*>(tau_ij_.xy_data().data());
    view.tau_yy = const_cast<double*>(tau_ij_.yy_data().data());

    // Gradient scratch buffers
    view.dudx = const_cast<double*>(dudx_.data().data());
    view.dudy = const_cast<double*>(dudy_.data().data());
    view.dvdx = const_cast<double*>(dvdx_.data().data());
    view.dvdy = const_cast<double*>(dvdy_.data().data());

    // Wall distance
    view.wall_distance = const_cast<double*>(wall_distance_.data().data());

    // Mesh parameters
    view.Nx = mesh_->Nx;
    view.Ny = mesh_->Ny;
    view.Ng = mesh_->Nghost;
    view.dx = mesh_->dx;
    view.dy = mesh_->dy;
    view.delta = (turb_model_ ? turb_model_->delta() : 1.0);

    return view;
}

SolverDeviceView RANSSolver::get_solver_view() const {
    // CPU build: always return host pointers
    SolverDeviceView view;

    view.u_face = const_cast<double*>(velocity_.u_data().data());
    view.v_face = const_cast<double*>(velocity_.v_data().data());
    view.u_star_face = const_cast<double*>(velocity_star_.u_data().data());
    view.v_star_face = const_cast<double*>(velocity_star_.v_data().data());
    view.u_old_face = const_cast<double*>(velocity_old_.u_data().data());
    view.v_old_face = const_cast<double*>(velocity_old_.v_data().data());
    view.u_stride = velocity_.u_stride();
    view.v_stride = velocity_.v_stride();

    // Initialize 3D fields to avoid undefined behavior in 2D mode
    view.w_face = nullptr;
    view.w_star_face = nullptr;
    view.w_old_face = nullptr;
    view.w_stride = 0;
    view.u_plane_stride = 0;
    view.v_plane_stride = 0;
    view.w_plane_stride = 0;

    // 3D velocity fields (overwrite defaults if 3D)
    if (!mesh_->is2D()) {
        view.w_face = const_cast<double*>(velocity_.w_data().data());
        view.w_star_face = const_cast<double*>(velocity_star_.w_data().data());
        view.w_old_face = const_cast<double*>(velocity_old_.w_data().data());
        view.w_stride = velocity_.w_stride();
        view.u_plane_stride = velocity_.u_plane_stride();
        view.v_plane_stride = velocity_.v_plane_stride();
        view.w_plane_stride = velocity_.w_plane_stride();
    }

    view.p = const_cast<double*>(pressure_.data().data());
    view.p_corr = const_cast<double*>(pressure_correction_.data().data());
    view.nu_t = const_cast<double*>(nu_t_.data().data());
    view.nu_eff = const_cast<double*>(nu_eff_.data().data());
    view.rhs = const_cast<double*>(rhs_poisson_.data().data());
    view.div = const_cast<double*>(div_velocity_.data().data());
    view.cell_stride = mesh_->total_Nx();
    view.cell_plane_stride = mesh_->total_Nx() * mesh_->total_Ny();

    view.conv_u = const_cast<double*>(conv_.u_data().data());
    view.conv_v = const_cast<double*>(conv_.v_data().data());
    view.diff_u = const_cast<double*>(diff_.u_data().data());
    view.diff_v = const_cast<double*>(diff_.v_data().data());

    // Initialize 3D work arrays to avoid undefined behavior in 2D mode
    view.conv_w = nullptr;
    view.diff_w = nullptr;

    // 3D work arrays (overwrite defaults if 3D)
    if (!mesh_->is2D()) {
        view.conv_w = const_cast<double*>(conv_.w_data().data());
        view.diff_w = const_cast<double*>(diff_.w_data().data());
    }

    view.Nx = mesh_->Nx;
    view.Ny = mesh_->Ny;
    view.Nz = mesh_->Nz;
    view.Ng = mesh_->Nghost;
    view.dx = mesh_->dx;
    view.dy = mesh_->dy;
    view.dz = mesh_->dz;
    view.dt = current_dt_;

    return view;
}
#endif

} // namespace nncfd

