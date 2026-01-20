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
#include "mpi_check.hpp"
#include "numerics.hpp"
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

// ============================================================================
// Spatial Operator Primitives (O2 and O4)
// These are the fundamental 1D building blocks for all derivative/interpolation ops.
// For stretched grids, multiply result by appropriate metric at output location.
// ============================================================================

// --- O2 Primitives (uniform h) ---

/// Same-stagger centered derivative O2: df/dx at location i (same stagger as input)
/// Stencil: (f[i+1] - f[i-1]) / (2*h)
inline double D_same_O2(double fm1, double fp1, double h) {
    return (fp1 - fm1) / (2.0 * h);
}

/// Center→Face derivative O2: (Dcf p)_{i+1/2} where p is at centers
/// Stencil: (p[i+1] - p[i]) / h  (output at face between i and i+1)
inline double Dcf_O2(double p_i, double p_ip1, double h) {
    return (p_ip1 - p_i) / h;
}

/// Face→Center derivative O2: (Dfc u)_i where u is at faces
/// Stencil: (u[i+1/2] - u[i-1/2]) / h  (output at center i)
inline double Dfc_O2(double u_imh, double u_iph, double h) {
    return (u_iph - u_imh) / h;
}

/// Center→Face interpolation O2: (Icf p)_{i+1/2} where p is at centers
/// Stencil: (p[i] + p[i+1]) / 2
inline double Icf_O2(double p_i, double p_ip1) {
    return 0.5 * (p_i + p_ip1);
}

/// Face→Center interpolation O2: (Ifc u)_i where u is at faces
/// Stencil: (u[i-1/2] + u[i+1/2]) / 2
inline double Ifc_O2(double u_imh, double u_iph) {
    return 0.5 * (u_imh + u_iph);
}

// --- O4 Primitives (uniform h) ---

/// Same-stagger centered derivative O4: df/dx at location i (same stagger as input)
/// Stencil: (-f[i+2] + 8*f[i+1] - 8*f[i-1] + f[i-2]) / (12*h)
inline double D_same_O4(double fm2, double fm1, double fp1, double fp2, double h) {
    return (-fp2 + 8.0*fp1 - 8.0*fm1 + fm2) / (12.0 * h);
}

/// Center→Face derivative O4: (Dcf p)_{i+1/2} where p is at centers
/// Stencil: (p[i-1] - 27*p[i] + 27*p[i+1] - p[i+2]) / (24*h)
/// This is the compact 4th-order stencil for center→face
inline double Dcf_O4(double p_im1, double p_i, double p_ip1, double p_ip2, double h) {
    return (p_im1 - 27.0*p_i + 27.0*p_ip1 - p_ip2) / (24.0 * h);
}

/// Face→Center derivative O4: (Dfc u)_i where u is at faces
/// Stencil: (u[i-3/2] - 27*u[i-1/2] + 27*u[i+1/2] - u[i+3/2]) / (24*h)
/// Faces indexed as: i-3/2 → im3h, i-1/2 → imh, i+1/2 → iph, i+3/2 → ip3h
inline double Dfc_O4(double u_im3h, double u_imh, double u_iph, double u_ip3h, double h) {
    return (u_im3h - 27.0*u_imh + 27.0*u_iph - u_ip3h) / (24.0 * h);
}

/// Center→Face interpolation O4: (Icf p)_{i+1/2} where p is at centers
/// Stencil: (-p[i-1] + 9*p[i] + 9*p[i+1] - p[i+2]) / 16
inline double Icf_O4(double p_im1, double p_i, double p_ip1, double p_ip2) {
    return (-p_im1 + 9.0*p_i + 9.0*p_ip1 - p_ip2) / 16.0;
}

/// Face→Center interpolation O4: (Ifc u)_i where u is at faces
/// Stencil: (-u[i-3/2] + 9*u[i-1/2] + 9*u[i+1/2] - u[i+3/2]) / 16
inline double Ifc_O4(double u_im3h, double u_imh, double u_iph, double u_ip3h) {
    return (-u_im3h + 9.0*u_imh + 9.0*u_iph - u_ip3h) / 16.0;
}

// --- Boundary-safe order selection ---
// These helpers determine if O4 stencil is safe at a given index

/// Check if O4 same-stagger derivative is safe (needs ±2 neighbors)
inline bool is_O4_safe_same(int i, int Ng, int N_interior) {
    // Interior indices run from Ng to Ng+N_interior-1
    // Need i-2 >= Ng and i+2 < Ng+N_interior
    return (i >= Ng + 2) && (i < Ng + N_interior - 2);
}

/// Check if O4 center→face derivative is safe at face i+1/2
/// For Dcf at face i+1/2, we need centers at i-1, i, i+1, i+2
inline bool is_O4_safe_Dcf(int i, int Ng, int N_interior) {
    return (i >= Ng + 1) && (i < Ng + N_interior - 2);
}

/// Check if O4 face→center derivative is safe at center i
/// For Dfc at center i, we need faces at i-3/2, i-1/2, i+1/2, i+3/2
/// In integer indexing: faces i-1, i, i+1, i+2 (if face index = left cell index + 1/2)
inline bool is_O4_safe_Dfc(int i, int Ng, int N_interior) {
    return (i >= Ng + 1) && (i < Ng + N_interior - 1);
}

/// Check if O4 interpolation is safe (same stencil width as derivatives)
inline bool is_O4_safe_interp(int i, int Ng, int N_interior) {
    return (i >= Ng + 1) && (i < Ng + N_interior - 1);
}

// ============================================================================
// End of Spatial Operator Primitives
// ============================================================================

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

// End the declare target block started earlier for unified kernels
#ifdef USE_GPU_OFFLOAD
#pragma omp end declare target
#endif

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
    const bool o4_safe_x = periodic_x || ((i >= Ng + 1) && (i <= Ng + Nx - 2));
    const bool o4_safe_y = periodic_y || ((j >= Ng + 1) && (j <= Ng + Ny - 2));
    const bool o4_safe_z = periodic_z || ((k >= Ng + 1) && (k <= Ng + Nz - 2));

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
    , velocity_rk_(mesh)          // RK work buffer for multi-stage methods
    , dudx_(mesh), dudy_(mesh), dvdx_(mesh), dvdy_(mesh)  // Gradient scratch for turbulence
    , wall_distance_(mesh)        // Precomputed wall distance field
    , poisson_solver_(mesh)
    , mg_poisson_solver_(mesh)
    , use_multigrid_(true)
    , current_dt_(config.dt)
{
    // Check for MPI environment - hard fail for GPU builds, warn for CPU
    // (this code uses GPU parallelism, not MPI distribution)
    // Set NNCFD_ALLOW_MULTI_RANK=1 to override (dangerous)
    enforce_single_rank_gpu("RANSSolver");

    // Validate ghost cell requirements for advection schemes
    // Upwind2 requires Nghost >= 2 due to 5-point stencil (i±2)
    if (config_.convective_scheme == ConvectiveScheme::Upwind2 && mesh.Nghost < 2) {
        std::cerr << "[Solver] WARNING: upwind2 scheme requires Nghost >= 2 but Nghost = "
                  << mesh.Nghost << "\n"
                  << "         Falling back to 1st-order upwind scheme.\n";
        config_.convective_scheme = ConvectiveScheme::Upwind;
    }

    // O4 spatial discretization requires Nghost >= 2 for 5-point stencils
    if (config_.space_order == 4 && mesh.Nghost < 2) {
        std::cerr << "[Solver] WARNING: space_order=4 requires Nghost >= 2 but Nghost = "
                  << mesh.Nghost << "\n"
                  << "         Falling back to space_order=2.\n";
        config_.space_order = 2;
    }

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
    // FFT (2D): requires periodic x AND z with uniform spacing - 3D only
    // FFT2D: for 2D meshes with periodic x, walls y
    // FFT1D: requires periodic x OR z (exactly one) with uniform spacing - 3D only
    bool fft_applicable = false;
    bool fft2d_applicable = false;
    bool fft1d_applicable = false;

    // Check which FFT solver is applicable (actual BCs set later via set_velocity_bc)
    // For now, assume defaults: periodic x,z - will be updated in set_velocity_bc
    bool periodic_xz = true;  // Default for channel: periodic x/z
    bool uniform_xz = true;   // Default for channel: uniform x/z spacing

    if (mesh.is2D()) {
        // 2D mesh: try FFT2D solver (periodic x, non-periodic y)
        try {
            fft2d_poisson_solver_ = std::make_unique<FFT2DPoissonSolver>(mesh);
            fft2d_poisson_solver_->set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                                          PoissonBC::Neumann, PoissonBC::Neumann);
            fft2d_applicable = true;
            std::cout << "[Solver] FFT2D solver initialized for 2D mesh\n";
        } catch (const std::exception& e) {
            std::cerr << "[Solver] FFT2D solver initialization failed: " << e.what() << "\n";
            fft2d_applicable = false;
        }
    } else {
        // 3D mesh: try 2D FFT first (periodic x AND z)
        if (periodic_xz && uniform_xz) {
            try {
                fft_poisson_solver_ = std::make_unique<FFTPoissonSolver>(mesh);
                fft_poisson_solver_->set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                                             PoissonBC::Neumann, PoissonBC::Neumann,
                                             PoissonBC::Periodic, PoissonBC::Periodic);
                // Set space order for O4-consistent eigenvalues when using O4 projection
                fft_poisson_solver_->set_space_order(config_.space_order);
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
        } catch (const std::exception& e) {
            std::cerr << "[Solver] FFT1D solver initialization failed: " << e.what() << "\n";
        }
    }
#endif

    // ========================================================================
    // Poisson Solver Auto-Selection
    // Priority: FFT (3D) → FFT2D (2D mesh) → FFT1D (3D 1-periodic) → HYPRE → MG
    // ========================================================================
    PoissonSolverType requested = config.poisson_solver;

    if (requested == PoissonSolverType::Auto) {
        // Auto-select: FFT > FFT2D > FFT1D > HYPRE > MG
#ifdef USE_FFT_POISSON
        if (fft_applicable) {
            selected_solver_ = PoissonSolverType::FFT;
            selection_reason_ ="auto: periodic(x,z) + uniform(dx,dz) + 3D";
        } else if (fft2d_applicable) {
            selected_solver_ = PoissonSolverType::FFT2D;
            selection_reason_ ="auto: 2D mesh + periodic(x) + uniform(dx)";
        } else if (fft1d_applicable) {
            selected_solver_ = PoissonSolverType::FFT1D;
            selection_reason_ ="auto: periodic(x) + uniform(dx) + 3D (1D FFT)";
        } else
#endif
#ifdef USE_HYPRE
        if (hypre_poisson_solver_) {
            selected_solver_ = PoissonSolverType::HYPRE;
            selection_reason_ ="auto: FFT not applicable, HYPRE available";
        } else
#endif
        {
            selected_solver_ = PoissonSolverType::MG;
            selection_reason_ ="auto: fallback to multigrid";
        }
    } else if (requested == PoissonSolverType::FFT) {
#ifdef USE_FFT_POISSON
        if (fft_applicable) {
            selected_solver_ = PoissonSolverType::FFT;
            selection_reason_ ="explicit: user requested FFT";
        } else {
            std::cerr << "[Solver] Warning: FFT requested but not applicable "
                      << "(requires 3D, periodic x/z, uniform dx/dz). Falling back to ";
            if (fft1d_applicable) {
                selected_solver_ = PoissonSolverType::FFT1D;
                std::cerr << "FFT1D.\n";
                selection_reason_ ="fallback from FFT: using FFT1D";
            } else
#ifdef USE_HYPRE
            if (hypre_poisson_solver_) {
                selected_solver_ = PoissonSolverType::HYPRE;
                std::cerr << "HYPRE.\n";
                selection_reason_ ="fallback from FFT: not applicable";
            } else
#endif
            {
                selected_solver_ = PoissonSolverType::MG;
                std::cerr << "MG.\n";
                selection_reason_ ="fallback from FFT: not applicable";
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
        selection_reason_ ="fallback from FFT: not built";
#endif
    } else if (requested == PoissonSolverType::FFT1D) {
#ifdef USE_FFT_POISSON
        if (fft1d_applicable) {
            selected_solver_ = PoissonSolverType::FFT1D;
            selection_reason_ ="explicit: user requested FFT1D";
        } else {
            std::cerr << "[Solver] Warning: FFT1D requested but not applicable. ";
            selected_solver_ = PoissonSolverType::MG;
            std::cerr << "Using MG.\n";
            selection_reason_ ="fallback from FFT1D: not applicable";
        }
#else
        std::cerr << "[Solver] Warning: FFT1D requested but USE_FFT_POISSON not built. ";
        selected_solver_ = PoissonSolverType::MG;
        std::cerr << "Using MG.\n";
        selection_reason_ ="fallback from FFT1D: not built";
#endif
    } else if (requested == PoissonSolverType::HYPRE) {
#ifdef USE_HYPRE
        if (hypre_poisson_solver_) {
            selected_solver_ = PoissonSolverType::HYPRE;
            selection_reason_ ="explicit: user requested HYPRE";
        } else {
            std::cerr << "[Solver] Warning: HYPRE initialization failed. Using MG.\n";
            selected_solver_ = PoissonSolverType::MG;
            selection_reason_ ="fallback from HYPRE: init failed";
        }
#else
        std::cerr << "[Solver] Warning: HYPRE requested but USE_HYPRE not built. Using MG.\n";
        selected_solver_ = PoissonSolverType::MG;
        selection_reason_ ="fallback from HYPRE: not built";
#endif
    } else {
        // PoissonSolverType::MG
        selected_solver_ = PoissonSolverType::MG;
        selection_reason_ ="explicit: user requested MG";
    }

    // Log the selection
    const char* solver_name = (selected_solver_ == PoissonSolverType::FFT) ? "FFT" :
                              (selected_solver_ == PoissonSolverType::FFT2D) ? "FFT2D" :
                              (selected_solver_ == PoissonSolverType::FFT1D) ? "FFT1D" :
                              (selected_solver_ == PoissonSolverType::HYPRE) ? "HYPRE" : "MG";
    std::cout << "[Poisson] selected=" << solver_name
              << " reason=" << selection_reason_
              << " dims=" << mesh.Nx << "x" << mesh.Ny << "x" << mesh.Nz << "\n";

    // Safety check: O4 spatial order requires Nghost >= 2, but MG is currently ng=1 only
    if (config.space_order == 4 && selected_solver_ == PoissonSolverType::MG) {
        std::cerr << "[Solver] ERROR: space_order=4 requires Nghost >= 2, but MG backend is ng=1 only.\n";
#ifdef USE_HYPRE
        if (hypre_poisson_solver_) {
            std::cerr << "         Falling back to HYPRE.\n";
            selected_solver_ = PoissonSolverType::HYPRE;
            selection_reason_ = "fallback from MG: O4 requires ng>=2";
        } else {
            std::cerr << "         HYPRE not available. Results may be incorrect!\n";
        }
#else
        std::cerr << "         HYPRE not built. Consider using FFT (periodic BCs) or rebuilding with HYPRE.\n";
        std::cerr << "         Results may be incorrect!\n";
#endif
    }

#ifdef USE_GPU_OFFLOAD
    // Fail-fast if GPU offload is enabled but no device is available
    gpu::verify_device_available();
#endif

    // Initialize raw pointers for unified code paths
    // In GPU builds, this also maps data to device; in CPU builds, just sets pointers
    initialize_gpu_buffers();
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

    // FFT2D requires periodic x AND 2D mesh AND non-periodic y
    bool periodic_y = (p_y_lo == PoissonBC::Periodic && p_y_hi == PoissonBC::Periodic);
    bool fft2d_compatible = periodic_x && !periodic_y && mesh_->is2D();

    if (fft2d_poisson_solver_) {
        if (fft2d_compatible) {
            // Update FFT2D solver BCs (y direction can be Neumann or Dirichlet)
            fft2d_poisson_solver_->set_bc(p_x_lo, p_x_hi, p_y_lo, p_y_hi);
        } else if (selected_solver_ == PoissonSolverType::FFT2D) {
            // FFT2D was selected but BCs are now incompatible - switch to MG
            std::cerr << "[Poisson] Warning: FFT2D solver incompatible with BCs "
                      << "(requires periodic x for 2D mesh). Switching to MG.\n";
            selected_solver_ = PoissonSolverType::MG;
        }
    }
#endif

#ifdef USE_HYPRE
    // Check HYPRE compatibility with current BCs
    // KNOWN ISSUE: HYPRE GPU (CUDA) has numerical instability with 2D problems
    // that have periodic Y-direction BCs. Fall back to MG for these cases.
    // The issue manifests as NaN after ~10 time steps.
    // 3D works fine, and 2D with x-periodic + y-walls (channel) works fine.
    // But 2D with y-periodic (spanwise or fully periodic) fails.
    {
        bool hypre_periodic_y = (p_y_lo == PoissonBC::Periodic && p_y_hi == PoissonBC::Periodic);
        bool y_periodic_2d = mesh_->is2D() && hypre_periodic_y;

        if (selected_solver_ == PoissonSolverType::HYPRE && y_periodic_2d) {
#ifdef USE_GPU_OFFLOAD
            // Fall back to MG for 2D with y-periodic on GPU (HYPRE CUDA instability)
            std::cerr << "[Poisson] HYPRE->MG fallback: 2D y-periodic + GPU\n";
            selected_solver_ = PoissonSolverType::MG;
#endif
        }
    }
#endif
}

void RANSSolver::set_body_force(double fx, double fy, double fz) {
    fx_ = fx;
    fy_ = fy;
    fz_ = fz;
}

void RANSSolver::print_solver_info() const {
    std::cout << "\n=== Solver Configuration ===\n";

    // Mesh info
    std::cout << "Mesh: " << mesh_->Nx << " x " << mesh_->Ny;
    if (!mesh_->is2D()) {
        std::cout << " x " << mesh_->Nz << " (3D)";
    } else {
        std::cout << " (2D)";
    }
    std::cout << "\n";

    // Poisson solver selection
    std::cout << "Poisson solver: ";
    switch (selected_solver_) {
        case PoissonSolverType::FFT:
            std::cout << "FFT (2D-FFT in x-z + tridiagonal in y)";
            break;
        case PoissonSolverType::FFT2D:
            std::cout << "FFT2D (1D-FFT in x + tridiagonal in y)";
            break;
        case PoissonSolverType::FFT1D:
            std::cout << "FFT1D (1D-FFT in periodic dir + 2D Helmholtz)";
            break;
        case PoissonSolverType::HYPRE:
            std::cout << "HYPRE PFMG (geometric multigrid)";
            break;
        case PoissonSolverType::MG:
            std::cout << "Native Multigrid (V-cycle)";
            break;
        default:
            std::cout << "Unknown";
    }
    std::cout << "\n";

    std::cout << "Selection reason: " << selection_reason_ << "\n";

    // Poisson solver parameters (MG uses these, others may not)
    if (selected_solver_ == PoissonSolverType::MG) {
        std::cout << "MG params: tol=" << config_.poisson_tol
                  << ", max_vcycles=" << config_.poisson_max_vcycles
                  << ", omega=" << config_.poisson_omega << "\n";
    }

    // Boundary conditions
    std::cout << "Velocity BCs: x=";
    std::cout << (velocity_bc_.x_lo == VelocityBC::Periodic ? "periodic" :
                  velocity_bc_.x_lo == VelocityBC::NoSlip ? "wall" : "inflow/outflow");
    std::cout << ", y=";
    std::cout << (velocity_bc_.y_lo == VelocityBC::Periodic ? "periodic" :
                  velocity_bc_.y_lo == VelocityBC::NoSlip ? "wall" : "inflow/outflow");
    if (!mesh_->is2D()) {
        std::cout << ", z=";
        std::cout << (velocity_bc_.z_lo == VelocityBC::Periodic ? "periodic" :
                      velocity_bc_.z_lo == VelocityBC::NoSlip ? "wall" : "inflow/outflow");
    }
    std::cout << "\n";

    // Turbulence model
    std::cout << "Turbulence: ";
    if (turb_model_) {
        std::cout << turb_model_->name();
    } else {
        std::cout << "None (laminar)";
    }
    std::cout << "\n";

    // Discretization info
    std::cout << "Convective scheme: ";
    switch (config_.convective_scheme) {
        case ConvectiveScheme::Central: std::cout << "Central"; break;
        case ConvectiveScheme::Skew:    std::cout << "Skew-symmetric"; break;
        case ConvectiveScheme::Upwind:  std::cout << "Upwind (1st)"; break;
        case ConvectiveScheme::Upwind2: std::cout << "Upwind (2nd)"; break;
    }
    std::cout << ", space_order=" << config_.space_order << "\n";

    // Build info
    std::cout << "Build features: ";
    std::vector<std::string> features;
#ifdef USE_GPU_OFFLOAD
    features.push_back("GPU");
#else
    features.push_back("CPU");
#endif
#ifdef USE_FFT_POISSON
    features.push_back("FFT");
#endif
#ifdef USE_HYPRE
    features.push_back("HYPRE");
#endif
    for (size_t i = 0; i < features.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << features[i];
    }
    std::cout << "\n";

    std::cout << "============================\n\n";
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
        
        double omega_init = k_init / (numerics::C_MU * config_.nu * 100.0);  // ν_t/ν ≈ 100 initially
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
        
        double omega_init = k_init / (numerics::C_MU * config_.nu * 100.0);  // ν_t/ν ≈ 100 initially
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

    // NVHPC WORKAROUND: Use gpu::dev_ptr + is_device_ptr for correct device addresses
    // Local pointer copies from get_solver_view() get HOST addresses that NVHPC
    // doesn't translate correctly when used with map(present:)
    double* u_dev = gpu::dev_ptr(v.u_face);
    double* v_dev = gpu::dev_ptr(v.v_face);

    // For 3D, apply x/y BCs to ALL z-planes (interior and ghost)
    // This is necessary because the z-BC code assumes x/y BCs are already applied
    // Nz_total = Nz + 2*Ng for 3D, 1 for 2D
    const int Nz_total = mesh_->is2D() ? 1 : (v.Nz + 2*Ng);
    const int u_plane_stride = mesh_->is2D() ? 0 : v.u_plane_stride;
    const int v_plane_stride = mesh_->is2D() ? 0 : v.v_plane_stride;

    // Apply u BCs in x-direction (for all k-planes including ghosts in 3D)
    const int n_u_x_bc = u_total_Ny * Ng * Nz_total;
    #pragma omp target teams distribute parallel for is_device_ptr(u_dev) \
        firstprivate(Nx, Ny, Ng, u_stride, u_plane_stride, Nz_total, x_lo_periodic, x_lo_noslip, x_hi_periodic, x_hi_noslip)
    for (int idx = 0; idx < n_u_x_bc; ++idx) {
        int j = idx % u_total_Ny;
        int g = (idx / u_total_Ny) % Ng;
        int k = idx / (u_total_Ny * Ng);  // k = 0 to Nz_total-1 covers all planes
        double* u_plane_ptr = u_dev + k * u_plane_stride;
        apply_u_bc_x_staggered(j, g, Nx, Ng, u_stride,
                              x_lo_periodic, x_lo_noslip,
                              x_hi_periodic, x_hi_noslip, u_plane_ptr);
    }

    // Apply u BCs in y-direction (for all k-planes including ghosts in 3D)
    const int n_u_y_bc = (Nx + 1 + 2 * Ng) * Ng * Nz_total;
    #pragma omp target teams distribute parallel for is_device_ptr(u_dev) \
        firstprivate(Nx, Ny, Ng, u_stride, u_plane_stride, Nz_total, y_lo_periodic, y_lo_noslip, y_hi_periodic, y_hi_noslip)
    for (int idx = 0; idx < n_u_y_bc; ++idx) {
        int u_x_size = Nx + 1 + 2 * Ng;
        int i = idx % u_x_size;
        int g = (idx / u_x_size) % Ng;
        int k = idx / (u_x_size * Ng);
        double* u_plane_ptr = u_dev + k * u_plane_stride;
        apply_u_bc_y_staggered(i, g, Ny, Ng, u_stride,
                              y_lo_periodic, y_lo_noslip,
                              y_hi_periodic, y_hi_noslip, u_plane_ptr);
    }

    // Apply v BCs in x-direction (for all k-planes including ghosts in 3D)
    const int n_v_x_bc = (Ny + 1 + 2 * Ng) * Ng * Nz_total;
    #pragma omp target teams distribute parallel for is_device_ptr(v_dev) \
        firstprivate(Nx, Ny, Ng, v_stride, v_plane_stride, Nz_total, x_lo_periodic, x_lo_noslip, x_hi_periodic, x_hi_noslip)
    for (int idx = 0; idx < n_v_x_bc; ++idx) {
        int v_y_size = Ny + 1 + 2 * Ng;
        int j = idx % v_y_size;
        int g = (idx / v_y_size) % Ng;
        int k = idx / (v_y_size * Ng);
        double* v_plane_ptr = v_dev + k * v_plane_stride;
        apply_v_bc_x_staggered(j, g, Nx, Ng, v_stride,
                              x_lo_periodic, x_lo_noslip,
                              x_hi_periodic, x_hi_noslip, v_plane_ptr);
    }

    // Apply v BCs in y-direction (for all k-planes including ghosts in 3D)
    const int n_v_y_bc = v_total_Nx * Ng * Nz_total;
    #pragma omp target teams distribute parallel for is_device_ptr(v_dev) \
        firstprivate(Nx, Ny, Ng, v_stride, v_plane_stride, Nz_total, y_lo_periodic, y_lo_noslip, y_hi_periodic, y_hi_noslip)
    for (int idx = 0; idx < n_v_y_bc; ++idx) {
        int i = idx % v_total_Nx;
        int g = (idx / v_total_Nx) % Ng;
        int k = idx / (v_total_Nx * Ng);
        double* v_plane_ptr = v_dev + k * v_plane_stride;
        apply_v_bc_y_staggered(i, g, Ny, Ng, v_stride,
                              y_lo_periodic, y_lo_noslip,
                              y_hi_periodic, y_hi_noslip, v_plane_ptr);
    }

    // CORNER FIX: For fully periodic domains, apply x-direction BCs again
    // to ensure corner ghosts are correctly wrapped after y-direction BCs modified them
    if (x_lo_periodic && x_hi_periodic) {
        #pragma omp target teams distribute parallel for is_device_ptr(u_dev) \
            firstprivate(Nx, Ny, Ng, u_stride, u_plane_stride, Nz_total, x_lo_periodic, x_lo_noslip, x_hi_periodic, x_hi_noslip)
        for (int idx = 0; idx < n_u_x_bc; ++idx) {
            int j = idx % u_total_Ny;
            int g = (idx / u_total_Ny) % Ng;
            int k = idx / (u_total_Ny * Ng);
            double* u_plane_ptr = u_dev + k * u_plane_stride;
            apply_u_bc_x_staggered(j, g, Nx, Ng, u_stride,
                                  x_lo_periodic, x_lo_noslip,
                                  x_hi_periodic, x_hi_noslip, u_plane_ptr);
        }
    }

    if (y_lo_periodic && y_hi_periodic) {
        #pragma omp target teams distribute parallel for is_device_ptr(v_dev) \
            firstprivate(Nx, Ny, Ng, v_stride, v_plane_stride, Nz_total, y_lo_periodic, y_lo_noslip, y_hi_periodic, y_hi_noslip)
        for (int idx = 0; idx < n_v_y_bc; ++idx) {
            int i = idx % v_total_Nx;
            int g = (idx / v_total_Nx) % Ng;
            int k = idx / (v_total_Nx * Ng);
            double* v_plane_ptr = v_dev + k * v_plane_stride;
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
        // NVHPC WORKAROUND: Get device pointer for w-velocity
        double* w_dev = gpu::dev_ptr(v.w_face);

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
        #pragma omp target teams distribute parallel for is_device_ptr(u_dev) \
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
                u_dev[idx_lo] = u_dev[(Ng + Nz - 1 - g) * u_plane_stride + j * u_stride + i];
                u_dev[idx_hi] = u_dev[(Ng + g) * u_plane_stride + j * u_stride + i];
            } else {
                if (z_lo_noslip) u_dev[idx_lo] = -u_dev[idx_src_lo];
                if (z_hi_noslip) u_dev[idx_hi] = -u_dev[idx_src_hi];
            }
        }

        // Apply v BCs in z-direction
        const int n_v_z_bc = (Nx + 2*Ng) * (Ny + 1 + 2*Ng) * Ng;
        #pragma omp target teams distribute parallel for is_device_ptr(v_dev) \
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
                v_dev[idx_lo] = v_dev[(Ng + Nz - 1 - g) * v_plane_stride + j * v_stride + i];
                v_dev[idx_hi] = v_dev[(Ng + g) * v_plane_stride + j * v_stride + i];
            } else {
                if (z_lo_noslip) v_dev[idx_lo] = -v_dev[idx_src_lo];
                if (z_hi_noslip) v_dev[idx_hi] = -v_dev[idx_src_hi];
            }
        }

        // Apply w BCs in z-direction (w is at z-faces, so different treatment)
        // For periodic: w at k=Ng and k=Ng+Nz should be same (wrap around)
        const int n_w_z_bc = (Nx + 2*Ng) * (Ny + 2*Ng) * Ng;
        #pragma omp target teams distribute parallel for is_device_ptr(w_dev) \
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
                    w_dev[(Ng + Nz) * w_plane_stride + j * w_stride + i] =
                        w_dev[Ng * w_plane_stride + j * w_stride + i];
                }
                // For w at z-faces with periodic BC:
                // Ghost at k=Ng-1-g gets value from k=Ng+Nz-1-g (interior near hi)
                // Ghost at k=Ng+Nz+1+g gets value from k=Ng+1+g (interior near lo)
                w_dev[idx_lo] = w_dev[(Ng + Nz - 1 - g) * w_plane_stride + j * w_stride + i];
                w_dev[idx_hi] = w_dev[(Ng + 1 + g) * w_plane_stride + j * w_stride + i];
            } else {
                // For no-slip: w at boundaries should be zero (normal velocity)
                if (z_lo_noslip) {
                    // w at k=Ng (first interior z-face) = 0 for solid wall
                    if (g == 0) {
                        w_dev[(Ng) * w_plane_stride + j * w_stride + i] = 0.0;
                    }
                    w_dev[idx_lo] = -w_dev[(Ng + g + 1) * w_plane_stride + j * w_stride + i];
                }
                if (z_hi_noslip) {
                    // w at k=Ng+Nz (last interior z-face) = 0 for solid wall
                    if (g == 0) {
                        w_dev[(Ng + Nz) * w_plane_stride + j * w_stride + i] = 0.0;
                    }
                    w_dev[idx_hi] = -w_dev[(Ng + Nz - 1 - g) * w_plane_stride + j * w_stride + i];
                }
            }
        }

        // Apply w BCs in x and y directions
        // w in x-direction
        const int n_w_x_bc = (Ny + 2*Ng) * (Nz + 1 + 2*Ng) * Ng;
        #pragma omp target teams distribute parallel for is_device_ptr(w_dev) \
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
                w_dev[idx_lo] = w_dev[k * w_plane_stride + j * w_stride + (Ng + Nx - 1 - g)];
                w_dev[idx_hi] = w_dev[k * w_plane_stride + j * w_stride + (Ng + g)];
            } else {
                if (x_lo_noslip) w_dev[idx_lo] = -w_dev[k * w_plane_stride + j * w_stride + (Ng + g)];
                if (x_hi_noslip) w_dev[idx_hi] = -w_dev[k * w_plane_stride + j * w_stride + (Ng + Nx - 1 - g)];
            }
        }

        // w in y-direction
        const int n_w_y_bc = (Nx + 2*Ng) * (Nz + 1 + 2*Ng) * Ng;
        #pragma omp target teams distribute parallel for is_device_ptr(w_dev) \
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
                w_dev[idx_lo] = w_dev[k * w_plane_stride + (Ng + Ny - 1 - g) * w_stride + i];
                w_dev[idx_hi] = w_dev[k * w_plane_stride + (Ng + g) * w_stride + i];
            } else {
                if (y_lo_noslip) w_dev[idx_lo] = -w_dev[k * w_plane_stride + (Ng + g) * w_stride + i];
                if (y_hi_noslip) w_dev[idx_hi] = -w_dev[k * w_plane_stride + (Ng + Ny - 1 - g) * w_stride + i];
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

    // =========================================================================
    // NVHPC-safe pattern: Use explicit vel/conv arguments with outer-target
    // regions. Pointer aliases are created INSIDE target regions to ensure
    // proper device address translation.
    // =========================================================================

    // Extract host pointers from arguments (these are mapping table keys)
    const double* u_host = vel.u_data().data();
    const double* v_host = vel.v_data().data();
    double* cu_host = conv.u_data().data();
    double* cv_host = conv.v_data().data();

    // Sizes for mapping
    const size_t u_total = vel.u_total_size();
    const size_t v_total = vel.v_total_size();
    const size_t cu_total = conv.u_total_size();
    const size_t cv_total = conv.v_total_size();

    // Strides (vel and conv may have different layouts in principle)
    const int u_stride = vel.u_stride();
    const int v_stride = vel.v_stride();
    const int conv_u_stride = conv.u_stride();
    const int conv_v_stride = conv.v_stride();

#ifndef NDEBUG
    // Sanity check: sizes should match for standard usage
    assert(u_total == cu_total && "vel and conv u-component size mismatch");
    assert(v_total == cv_total && "vel and conv v-component size mismatch");
#endif

    // Mesh parameters
    const double dx = mesh_->dx;
    const double dy = mesh_->dy;
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Ng = mesh_->Nghost;

    // Scheme selection
    const bool use_central = (config_.convective_scheme == ConvectiveScheme::Central);
    const bool use_skew = (config_.convective_scheme == ConvectiveScheme::Skew);
    const bool use_upwind2 = (config_.convective_scheme == ConvectiveScheme::Upwind2);
    const bool use_O4 = (config_.space_order == 4);

    // Periodic flags for O4 boundary safety checks
    const bool x_periodic = (velocity_bc_.x_lo == VelocityBC::Periodic) &&
                            (velocity_bc_.x_hi == VelocityBC::Periodic);
    const bool y_periodic = (velocity_bc_.y_lo == VelocityBC::Periodic) &&
                            (velocity_bc_.y_hi == VelocityBC::Periodic);
    const bool z_periodic = (velocity_bc_.z_lo == VelocityBC::Periodic) &&
                            (velocity_bc_.z_hi == VelocityBC::Periodic);

    // For 3D path compatibility (uses old SolverDeviceView pattern for now)
    [[maybe_unused]] const size_t u_total_size = u_total;
    [[maybe_unused]] const size_t v_total_size = v_total;
    [[maybe_unused]] const size_t w_total_size = vel.w_total_size();
    const double* u_ptr = u_host;
    const double* v_ptr = v_host;
    double* conv_u_ptr = cu_host;
    double* conv_v_ptr = cv_host;

    // 3D path
    if (!mesh_->is2D()) {
        const double dz = mesh_->dz;

        // 3D strides from vel argument
        const int u_plane_stride = vel.u_plane_stride();
        const int v_plane_stride = vel.v_plane_stride();
        const int w_stride = vel.w_stride();
        const int w_plane_stride = vel.w_plane_stride();

        // 3D pointers from arguments
        const double* w_host = vel.w_data().data();
        double* cw_host = conv.w_data().data();

        const int n_u_faces = (Nx + 1) * Ny * Nz;
        const int n_v_faces = Nx * (Ny + 1) * Nz;
        const int n_w_faces = Nx * Ny * (Nz + 1);

        // Conv strides from conv argument (shadow outer scope intentionally)
        const int conv_u_stride = conv.u_stride();
        const int conv_u_plane_stride = conv.u_plane_stride();
        const int conv_v_stride = conv.v_stride();
        const int conv_v_plane_stride = conv.v_plane_stride();
        const int conv_w_stride = conv.w_stride();
        const int conv_w_plane_stride = conv.w_plane_stride();

        // NVHPC WORKAROUND: Use MEMBER pointers directly for device access.
        // The euler_substep function swaps velocity_u_ptr_ etc. before calling this,
        // so these member pointers point to the correct velocity field data.
        // Using function parameter pointers (u_host etc.) doesn't work correctly
        // because NVHPC fails to find the device mapping.
        const double* u_dev = gpu::dev_ptr(velocity_u_ptr_);
        const double* v_dev = gpu::dev_ptr(velocity_v_ptr_);
        const double* w_dev = gpu::dev_ptr(velocity_w_ptr_);
        double* conv_u_dev = gpu::dev_ptr(conv_u_ptr_);
        double* conv_v_dev = gpu::dev_ptr(conv_v_ptr_);
        double* conv_w_dev = gpu::dev_ptr(conv_w_ptr_);

        if (use_skew) {
            // Skew-symmetric (energy-conserving) advection
            #pragma omp target teams distribute parallel for is_device_ptr(u_dev, v_dev, w_dev, conv_u_dev) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, conv_u_stride, conv_u_plane_stride, Nx, Ny, Ng)
            for (int idx = 0; idx < n_u_faces; ++idx) {
                int i = idx % (Nx + 1) + Ng;
                int j = (idx / (Nx + 1)) % Ny + Ng;
                int k = idx / ((Nx + 1) * Ny) + Ng;

                convective_u_face_kernel_skew_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, conv_u_stride, conv_u_plane_stride,
                    dx, dy, dz, u_dev, v_dev, w_dev, conv_u_dev);
            }

            #pragma omp target teams distribute parallel for is_device_ptr(u_dev, v_dev, w_dev, conv_v_dev) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, conv_v_stride, conv_v_plane_stride, Nx, Ny, Ng)
            for (int idx = 0; idx < n_v_faces; ++idx) {
                int i = idx % Nx + Ng;
                int j = (idx / Nx) % (Ny + 1) + Ng;
                int k = idx / (Nx * (Ny + 1)) + Ng;

                convective_v_face_kernel_skew_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, conv_v_stride, conv_v_plane_stride,
                    dx, dy, dz, u_dev, v_dev, w_dev, conv_v_dev);
            }

            #pragma omp target teams distribute parallel for is_device_ptr(u_dev, v_dev, w_dev, conv_w_dev) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, conv_w_stride, conv_w_plane_stride, Nx, Ny, Ng)
            for (int idx = 0; idx < n_w_faces; ++idx) {
                int i = idx % Nx + Ng;
                int j = (idx / Nx) % Ny + Ng;
                int k = idx / (Nx * Ny) + Ng;

                convective_w_face_kernel_skew_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, conv_w_stride, conv_w_plane_stride,
                    dx, dy, dz, u_dev, v_dev, w_dev, conv_w_dev);
            }
        } else if (use_upwind2) {
            // 2nd-order upwind with minmod limiter
            #pragma omp target teams distribute parallel for is_device_ptr(u_dev, v_dev, w_dev, conv_u_dev) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, conv_u_stride, conv_u_plane_stride, Nx, Ny, Ng)
            for (int idx = 0; idx < n_u_faces; ++idx) {
                int i = idx % (Nx + 1) + Ng;
                int j = (idx / (Nx + 1)) % Ny + Ng;
                int k = idx / ((Nx + 1) * Ny) + Ng;

                convective_u_face_kernel_upwind2_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, conv_u_stride, conv_u_plane_stride,
                    dx, dy, dz, u_dev, v_dev, w_dev, conv_u_dev);
            }

            #pragma omp target teams distribute parallel for is_device_ptr(u_dev, v_dev, w_dev, conv_v_dev) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, conv_v_stride, conv_v_plane_stride, Nx, Ny, Ng)
            for (int idx = 0; idx < n_v_faces; ++idx) {
                int i = idx % Nx + Ng;
                int j = (idx / Nx) % (Ny + 1) + Ng;
                int k = idx / (Nx * (Ny + 1)) + Ng;

                convective_v_face_kernel_upwind2_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, conv_v_stride, conv_v_plane_stride,
                    dx, dy, dz, u_dev, v_dev, w_dev, conv_v_dev);
            }

            #pragma omp target teams distribute parallel for is_device_ptr(u_dev, v_dev, w_dev, conv_w_dev) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, conv_w_stride, conv_w_plane_stride, Nx, Ny, Ng)
            for (int idx = 0; idx < n_w_faces; ++idx) {
                int i = idx % Nx + Ng;
                int j = (idx / Nx) % Ny + Ng;
                int k = idx / (Nx * Ny) + Ng;

                convective_w_face_kernel_upwind2_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, conv_w_stride, conv_w_plane_stride,
                    dx, dy, dz, u_dev, v_dev, w_dev, conv_w_dev);
            }
        } else if (use_O4 && use_central) {
            // O4 Central advection (4th-order derivatives, hybrid O4/O2 near boundaries)
            #pragma omp target teams distribute parallel for is_device_ptr(u_dev, v_dev, w_dev, conv_u_dev) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, Ng, Nx, Ny, Nz, x_periodic, y_periodic, z_periodic)
            for (int idx = 0; idx < n_u_faces; ++idx) {
                int i = idx % (Nx + 1) + Ng;
                int j = (idx / (Nx + 1)) % Ny + Ng;
                int k = idx / ((Nx + 1) * Ny) + Ng;

                convective_u_face_kernel_central_O4_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, u_stride, u_plane_stride,
                    dx, dy, dz, Ng, Nx, Ny, Nz, x_periodic, y_periodic, z_periodic,
                    u_dev, v_dev, w_dev, conv_u_dev);
            }

            #pragma omp target teams distribute parallel for is_device_ptr(u_dev, v_dev, w_dev, conv_v_dev) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, Ng, Nx, Ny, Nz, x_periodic, y_periodic, z_periodic)
            for (int idx = 0; idx < n_v_faces; ++idx) {
                int i = idx % Nx + Ng;
                int j = (idx / Nx) % (Ny + 1) + Ng;
                int k = idx / (Nx * (Ny + 1)) + Ng;

                convective_v_face_kernel_central_O4_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, v_stride, v_plane_stride,
                    dx, dy, dz, Ng, Nx, Ny, Nz, x_periodic, y_periodic, z_periodic,
                    u_dev, v_dev, w_dev, conv_v_dev);
            }

            #pragma omp target teams distribute parallel for is_device_ptr(u_dev, v_dev, w_dev, conv_w_dev) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, Ng, Nx, Ny, Nz, x_periodic, y_periodic, z_periodic)
            for (int idx = 0; idx < n_w_faces; ++idx) {
                int i = idx % Nx + Ng;
                int j = (idx / Nx) % Ny + Ng;
                int k = idx / (Nx * Ny) + Ng;

                convective_w_face_kernel_central_O4_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, w_stride, w_plane_stride,
                    dx, dy, dz, Ng, Nx, Ny, Nz, x_periodic, y_periodic, z_periodic,
                    u_dev, v_dev, w_dev, conv_w_dev);
            }
        } else if (use_O4 && use_skew) {
            // O4 Skew-symmetric advection (energy-conserving with 4th-order advective derivatives)
            #pragma omp target teams distribute parallel for is_device_ptr(u_dev, v_dev, w_dev, conv_u_dev) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, Ng, Nx, Ny, Nz, x_periodic, y_periodic, z_periodic)
            for (int idx = 0; idx < n_u_faces; ++idx) {
                int i = idx % (Nx + 1) + Ng;
                int j = (idx / (Nx + 1)) % Ny + Ng;
                int k = idx / ((Nx + 1) * Ny) + Ng;

                convective_u_face_kernel_skew_O4_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, u_stride, u_plane_stride,
                    dx, dy, dz, Ng, Nx, Ny, Nz, x_periodic, y_periodic, z_periodic,
                    u_dev, v_dev, w_dev, conv_u_dev);
            }

            #pragma omp target teams distribute parallel for is_device_ptr(u_dev, v_dev, w_dev, conv_v_dev) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, Ng, Nx, Ny, Nz, x_periodic, y_periodic, z_periodic)
            for (int idx = 0; idx < n_v_faces; ++idx) {
                int i = idx % Nx + Ng;
                int j = (idx / Nx) % (Ny + 1) + Ng;
                int k = idx / (Nx * (Ny + 1)) + Ng;

                convective_v_face_kernel_skew_O4_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, v_stride, v_plane_stride,
                    dx, dy, dz, Ng, Nx, Ny, Nz, x_periodic, y_periodic, z_periodic,
                    u_dev, v_dev, w_dev, conv_v_dev);
            }

            #pragma omp target teams distribute parallel for is_device_ptr(u_dev, v_dev, w_dev, conv_w_dev) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, Ng, Nx, Ny, Nz, x_periodic, y_periodic, z_periodic)
            for (int idx = 0; idx < n_w_faces; ++idx) {
                int i = idx % Nx + Ng;
                int j = (idx / Nx) % Ny + Ng;
                int k = idx / (Nx * Ny) + Ng;

                convective_w_face_kernel_skew_O4_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, w_stride, w_plane_stride,
                    dx, dy, dz, Ng, Nx, Ny, Nz, x_periodic, y_periodic, z_periodic,
                    u_dev, v_dev, w_dev, conv_w_dev);
            }
        } else {
            // Central or 1st-order upwind (O2 path)
            #pragma omp target teams distribute parallel for is_device_ptr(u_dev, v_dev, w_dev, conv_u_dev) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, use_central, Nx, Ny, Ng)
            for (int idx = 0; idx < n_u_faces; ++idx) {
                int i = idx % (Nx + 1) + Ng;
                int j = (idx / (Nx + 1)) % Ny + Ng;
                int k = idx / ((Nx + 1) * Ny) + Ng;

                convective_u_face_kernel_staggered_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, u_stride, u_plane_stride,
                    dx, dy, dz, use_central, u_dev, v_dev, w_dev, conv_u_dev);
            }

            #pragma omp target teams distribute parallel for is_device_ptr(u_dev, v_dev, w_dev, conv_v_dev) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, use_central, Nx, Ny, Ng)
            for (int idx = 0; idx < n_v_faces; ++idx) {
                int i = idx % Nx + Ng;
                int j = (idx / Nx) % (Ny + 1) + Ng;
                int k = idx / (Nx * (Ny + 1)) + Ng;

                convective_v_face_kernel_staggered_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, v_stride, v_plane_stride,
                    dx, dy, dz, use_central, u_dev, v_dev, w_dev, conv_v_dev);
            }

            #pragma omp target teams distribute parallel for is_device_ptr(u_dev, v_dev, w_dev, conv_w_dev) \
                firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, use_central, Nx, Ny, Ng)
            for (int idx = 0; idx < n_w_faces; ++idx) {
                int i = idx % Nx + Ng;
                int j = (idx / Nx) % Ny + Ng;
                int k = idx / (Nx * Ny) + Ng;

                convective_w_face_kernel_staggered_3d(i, j, k,
                    u_stride, u_plane_stride, v_stride, v_plane_stride,
                    w_stride, w_plane_stride, w_stride, w_plane_stride,
                    dx, dy, dz, use_central, u_dev, v_dev, w_dev, conv_w_dev);
            }
        }
        return;
    }

    // 2D path - use aliases that match original pragma pattern
    const int n_u_faces = (Nx + 1) * Ny;
    const int n_v_faces = Nx * (Ny + 1);

#ifdef USE_GPU_OFFLOAD
    // NVHPC WORKAROUND: Get actual device addresses via omp_get_mapped_ptr
    // Local pointer copies in map(present:) get HOST addresses in NVHPC
    int device = omp_get_default_device();
    const double* u_dev = static_cast<const double*>(omp_get_mapped_ptr(const_cast<double*>(u_host), device));
    const double* v_dev = static_cast<const double*>(omp_get_mapped_ptr(const_cast<double*>(v_host), device));
    double* cu_dev = static_cast<double*>(omp_get_mapped_ptr(cu_host, device));
    double* cv_dev = static_cast<double*>(omp_get_mapped_ptr(cv_host, device));
#else
    const double* u_dev = u_host;
    const double* v_dev = v_host;
    double* cu_dev = cu_host;
    double* cv_dev = cv_host;
#endif

    if (use_skew) {
        // Skew-symmetric (energy-conserving) advection - 2D
        #pragma omp target teams distribute parallel for is_device_ptr(u_dev, v_dev, cu_dev) \
            firstprivate(dx, dy, u_stride, v_stride, conv_u_stride, Nx, Ng)
        for (int idx = 0; idx < n_u_faces; ++idx) {
            int i = idx % (Nx + 1) + Ng;
            int j = idx / (Nx + 1) + Ng;

            convective_u_face_kernel_skew_2d(i, j, u_stride, v_stride, conv_u_stride,
                                            dx, dy, u_dev, v_dev, cu_dev);
        }

        #pragma omp target teams distribute parallel for is_device_ptr(u_dev, v_dev, cv_dev) \
            firstprivate(dx, dy, u_stride, v_stride, conv_v_stride, Nx, Ng)
        for (int idx = 0; idx < n_v_faces; ++idx) {
            int i = idx % Nx + Ng;
            int j = idx / Nx + Ng;

            convective_v_face_kernel_skew_2d(i, j, u_stride, v_stride, conv_v_stride,
                                            dx, dy, u_dev, v_dev, cv_dev);
        }
    } else if (use_upwind2) {
        // 2nd-order upwind with minmod limiter - 2D
        #pragma omp target teams distribute parallel for is_device_ptr(u_dev, v_dev, cu_dev) \
            firstprivate(dx, dy, u_stride, v_stride, conv_u_stride, Nx, Ng)
        for (int idx = 0; idx < n_u_faces; ++idx) {
            int i = idx % (Nx + 1) + Ng;
            int j = idx / (Nx + 1) + Ng;

            convective_u_face_kernel_upwind2_2d(i, j, u_stride, v_stride, conv_u_stride,
                                               dx, dy, u_dev, v_dev, cu_dev);
        }

        #pragma omp target teams distribute parallel for is_device_ptr(u_dev, v_dev, cv_dev) \
            firstprivate(dx, dy, u_stride, v_stride, conv_v_stride, Nx, Ng)
        for (int idx = 0; idx < n_v_faces; ++idx) {
            int i = idx % Nx + Ng;
            int j = idx / Nx + Ng;

            convective_v_face_kernel_upwind2_2d(i, j, u_stride, v_stride, conv_v_stride,
                                               dx, dy, u_dev, v_dev, cv_dev);
        }
    } else {
        // Central or 1st-order upwind (original path) - 2D
        // Uses ADVECTIVE form: conv = u*du/dx + v*du/dy
        // NVHPC WORKAROUND: Use device pointers obtained via omp_get_mapped_ptr above

        #pragma omp target teams distribute parallel for is_device_ptr(u_dev, v_dev, cu_dev) \
            firstprivate(dx, dy, u_stride, v_stride, use_central, Nx, Ng)
        for (int idx = 0; idx < n_u_faces; ++idx) {
            int i = idx % (Nx + 1) + Ng;
            int j = idx / (Nx + 1) + Ng;

            // Inline convective_u_face_kernel_staggered (advective form)
            const int u_idx = j * u_stride + i;
            const double uu = u_dev[u_idx];

            // Interpolate v to x-face (average 4 surrounding v-faces)
            const double v_bl = v_dev[j * v_stride + (i-1)];
            const double v_br = v_dev[j * v_stride + i];
            const double v_tl = v_dev[(j+1) * v_stride + (i-1)];
            const double v_tr = v_dev[(j+1) * v_stride + i];
            const double vv = 0.25 * (v_bl + v_br + v_tl + v_tr);

            double dudx, dudy;
            if (use_central) {
                dudx = (u_dev[j * u_stride + (i+1)] - u_dev[j * u_stride + (i-1)]) / (2.0 * dx);
                dudy = (u_dev[(j+1) * u_stride + i] - u_dev[(j-1) * u_stride + i]) / (2.0 * dy);
            } else {
                if (uu >= 0) {
                    dudx = (u_dev[u_idx] - u_dev[j * u_stride + (i-1)]) / dx;
                } else {
                    dudx = (u_dev[j * u_stride + (i+1)] - u_dev[u_idx]) / dx;
                }
                if (vv >= 0) {
                    dudy = (u_dev[u_idx] - u_dev[(j-1) * u_stride + i]) / dy;
                } else {
                    dudy = (u_dev[(j+1) * u_stride + i] - u_dev[u_idx]) / dy;
                }
            }
            cu_dev[u_idx] = uu * dudx + vv * dudy;
        }

        #pragma omp target teams distribute parallel for is_device_ptr(u_dev, v_dev, cv_dev) \
            firstprivate(dx, dy, u_stride, v_stride, use_central, Nx, Ng)
        for (int idx = 0; idx < n_v_faces; ++idx) {
            int i = idx % Nx + Ng;
            int j = idx / Nx + Ng;

            // Inline convective_v_face_kernel_staggered (advective form)
            const int v_idx = j * v_stride + i;
            const double vv = v_dev[v_idx];

            // Interpolate u to y-face (average 4 surrounding u-faces)
            const double u_bl = u_dev[(j-1) * u_stride + i];
            const double u_br = u_dev[(j-1) * u_stride + (i+1)];
            const double u_tl = u_dev[j * u_stride + i];
            const double u_tr = u_dev[j * u_stride + (i+1)];
            const double uu = 0.25 * (u_bl + u_br + u_tl + u_tr);

            double dvdx, dvdy;
            if (use_central) {
                dvdx = (v_dev[j * v_stride + (i+1)] - v_dev[j * v_stride + (i-1)]) / (2.0 * dx);
                dvdy = (v_dev[(j+1) * v_stride + i] - v_dev[(j-1) * v_stride + i]) / (2.0 * dy);
            } else {
                if (uu >= 0) {
                    dvdx = (v_dev[v_idx] - v_dev[j * v_stride + (i-1)]) / dx;
                } else {
                    dvdx = (v_dev[j * v_stride + (i+1)] - v_dev[v_idx]) / dx;
                }
                if (vv >= 0) {
                    dvdy = (v_dev[v_idx] - v_dev[(j-1) * v_stride + i]) / dy;
                } else {
                    dvdy = (v_dev[(j+1) * v_stride + i] - v_dev[v_idx]) / dy;
                }
            }
            cv_dev[v_idx] = uu * dvdx + vv * dvdy;
        }
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

        // NVHPC WORKAROUND: Use MEMBER pointers directly for device access.
        // The euler_substep function swaps velocity_u_ptr_ etc. before calling this,
        // so these member pointers point to the correct velocity field data.
        // Using get_solver_view() pointers doesn't work correctly because NVHPC
        // fails to find the device mapping.
        const double* u_dev = gpu::dev_ptr(velocity_u_ptr_);
        const double* v_dev = gpu::dev_ptr(velocity_v_ptr_);
        const double* w_dev = gpu::dev_ptr(velocity_w_ptr_);
        const double* nu_dev = gpu::dev_ptr(nu_eff_ptr_);
        double* diff_u_dev = gpu::dev_ptr(diff_u_ptr_);
        double* diff_v_dev = gpu::dev_ptr(diff_v_ptr_);
        double* diff_w_dev = gpu::dev_ptr(diff_w_ptr_);

        // Compute u-momentum diffusion at x-faces (3D)
        const int n_u_faces = (Nx + 1) * Ny * Nz;
        #pragma omp target teams distribute parallel for is_device_ptr(u_dev, nu_dev, diff_u_dev) \
            firstprivate(dx, dy, dz, u_stride, u_plane_stride, nu_stride, nu_plane_stride, Nx, Ny, Ng)
        for (int idx = 0; idx < n_u_faces; ++idx) {
            int i = idx % (Nx + 1) + Ng;
            int j = (idx / (Nx + 1)) % Ny + Ng;
            int k = idx / ((Nx + 1) * Ny) + Ng;

            diffusive_u_face_kernel_staggered_3d(i, j, k,
                u_stride, u_plane_stride, nu_stride, nu_plane_stride, u_stride, u_plane_stride,
                dx, dy, dz, u_dev, nu_dev, diff_u_dev);
        }

        // Compute v-momentum diffusion at y-faces (3D)
        const int n_v_faces = Nx * (Ny + 1) * Nz;
        #pragma omp target teams distribute parallel for is_device_ptr(v_dev, nu_dev, diff_v_dev) \
            firstprivate(dx, dy, dz, v_stride, v_plane_stride, nu_stride, nu_plane_stride, Nx, Ny, Ng)
        for (int idx = 0; idx < n_v_faces; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % (Ny + 1) + Ng;
            int k = idx / (Nx * (Ny + 1)) + Ng;

            diffusive_v_face_kernel_staggered_3d(i, j, k,
                v_stride, v_plane_stride, nu_stride, nu_plane_stride, v_stride, v_plane_stride,
                dx, dy, dz, v_dev, nu_dev, diff_v_dev);
        }

        // Compute w-momentum diffusion at z-faces (3D)
        const int n_w_faces = Nx * Ny * (Nz + 1);
        #pragma omp target teams distribute parallel for is_device_ptr(w_dev, nu_dev, diff_w_dev) \
            firstprivate(dx, dy, dz, w_stride, w_plane_stride, nu_stride, nu_plane_stride, Nx, Ny, Ng)
        for (int idx = 0; idx < n_w_faces; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % Ny + Ng;
            int k = idx / (Nx * Ny) + Ng;

            diffusive_w_face_kernel_staggered_3d(i, j, k,
                w_stride, w_plane_stride, nu_stride, nu_plane_stride, w_stride, w_plane_stride,
                dx, dy, dz, w_dev, nu_dev, diff_w_dev);
        }
        return;
    }

    // 2D path
    // NVHPC WORKAROUND: Use omp_get_mapped_ptr to get actual device addresses.
    // Member pointers in target regions get HOST addresses in NVHPC.

#ifdef USE_GPU_OFFLOAD
    int device = omp_get_default_device();
    const double* u_dev = static_cast<const double*>(omp_get_mapped_ptr(velocity_u_ptr_, device));
    const double* v_dev = static_cast<const double*>(omp_get_mapped_ptr(velocity_v_ptr_, device));
    const double* nu_dev = static_cast<const double*>(omp_get_mapped_ptr(nu_eff_ptr_, device));
    double* diff_u_dev = static_cast<double*>(omp_get_mapped_ptr(diff_u_ptr_, device));
    double* diff_v_dev = static_cast<double*>(omp_get_mapped_ptr(diff_v_ptr_, device));
#else
    const double* u_dev = velocity_u_ptr_;
    const double* v_dev = velocity_v_ptr_;
    const double* nu_dev = nu_eff_ptr_;
    double* diff_u_dev = diff_u_ptr_;
    double* diff_v_dev = diff_v_ptr_;
#endif

    // Compute u-momentum diffusion at x-faces
    const int n_u_faces = (Nx + 1) * Ny;
    const int diff_u_stride = u_stride;
    #pragma omp target teams distribute parallel for is_device_ptr(u_dev, nu_dev, diff_u_dev) \
        firstprivate(dx, dy, u_stride, nu_stride, diff_u_stride, Nx, Ng)
    for (int idx = 0; idx < n_u_faces; ++idx) {
        int i_local = idx % (Nx + 1);
        int j_local = idx / (Nx + 1);
        int i = i_local + Ng;
        int j = j_local + Ng;

        // Inline diffusive_u_face_kernel_staggered
        const int u_idx = j * u_stride + i;
        const double dx2 = dx * dx;
        const double dy2 = dy * dy;

        // Viscosity at cell centers adjacent to x-face
        const double nu_left = nu_dev[j * nu_stride + (i-1)];
        const double nu_right = nu_dev[j * nu_stride + i];

        // Face-averaged viscosity for d2u/dx2 term
        const double nu_e = 0.5 * (nu_right + (i+1 < nu_stride ? nu_dev[j * nu_stride + (i+1)] : nu_right));
        const double nu_w = 0.5 * (nu_left + (i-2 >= 0 ? nu_dev[j * nu_stride + (i-2)] : nu_left));
        const double nu_n = 0.5 * (nu_left + nu_right);
        const double nu_s = 0.5 * (nu_left + nu_right);

        // d2u/dx2 using u at x-faces
        const double d2u_dx2 = (nu_e * (u_dev[j * u_stride + (i+1)] - u_dev[u_idx])
                               - nu_w * (u_dev[u_idx] - u_dev[j * u_stride + (i-1)])) / dx2;

        // d2u/dy2 using u at x-faces
        const double d2u_dy2 = (nu_n * (u_dev[(j+1) * u_stride + i] - u_dev[u_idx])
                               - nu_s * (u_dev[u_idx] - u_dev[(j-1) * u_stride + i])) / dy2;

        const int diff_idx = j * diff_u_stride + i;
        diff_u_dev[diff_idx] = d2u_dx2 + d2u_dy2;
    }

    // Compute v-momentum diffusion at y-faces
    const int n_v_faces = Nx * (Ny + 1);
    const int diff_v_stride = v_stride;
    #pragma omp target teams distribute parallel for is_device_ptr(v_dev, nu_dev, diff_v_dev) \
        firstprivate(dx, dy, v_stride, nu_stride, diff_v_stride, Nx, Ng)
    for (int idx = 0; idx < n_v_faces; ++idx) {
        int i_local = idx % Nx;
        int j_local = idx / Nx;
        int i = i_local + Ng;
        int j = j_local + Ng;

        // Inline diffusive_v_face_kernel_staggered
        const int v_idx = j * v_stride + i;
        const double dx2 = dx * dx;
        const double dy2 = dy * dy;

        // Viscosity at cell centers adjacent to y-face
        const double nu_bottom = nu_dev[(j-1) * nu_stride + i];
        const double nu_top = nu_dev[j * nu_stride + i];

        // Face-averaged viscosity
        const double nu_e = 0.5 * (nu_bottom + nu_top);
        const double nu_w = 0.5 * (nu_bottom + nu_top);
        const double nu_n = 0.5 * (nu_top + (j+1 < nu_stride ? nu_dev[(j+1) * nu_stride + i] : nu_top));
        const double nu_s = 0.5 * (nu_bottom + (j-2 >= 0 ? nu_dev[(j-2) * nu_stride + i] : nu_bottom));

        // d2v/dx2 using v at y-faces
        const double d2v_dx2 = (nu_e * (v_dev[j * v_stride + (i+1)] - v_dev[v_idx])
                               - nu_w * (v_dev[v_idx] - v_dev[j * v_stride + (i-1)])) / dx2;

        // d2v/dy2 using v at y-faces
        const double d2v_dy2 = (nu_n * (v_dev[(j+1) * v_stride + i] - v_dev[v_idx])
                               - nu_s * (v_dev[v_idx] - v_dev[(j-1) * v_stride + i])) / dy2;

        const int diff_idx = j * diff_v_stride + i;
        diff_v_dev[diff_idx] = d2v_dx2 + d2v_dy2;
    }
}

void RANSSolver::compute_divergence(VelocityWhich which, ScalarField& div) {
    (void)div;  // Unused - always operates on div_velocity_ via member pointers

    // NVHPC WORKAROUND: Use member pointers directly in target regions.
    // Local pointer copies from get_solver_view() get HOST addresses in NVHPC.

    const double dx = mesh_->dx;
    const double dy = mesh_->dy;
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Ng = mesh_->Nghost;
    const int div_stride = mesh_->total_Nx();

    // O4 spatial discretization for divergence (Dfc_O4)
    const bool use_O4 = (config_.space_order == 4);

    // Periodic flags for O4 boundary handling
    const bool x_periodic = (velocity_bc_.x_lo == VelocityBC::Periodic) &&
                            (velocity_bc_.x_hi == VelocityBC::Periodic);
    const bool y_periodic = (velocity_bc_.y_lo == VelocityBC::Periodic) &&
                            (velocity_bc_.y_hi == VelocityBC::Periodic);
    const bool z_periodic = (velocity_bc_.z_lo == VelocityBC::Periodic) &&
                            (velocity_bc_.z_hi == VelocityBC::Periodic);

    const int u_stride = velocity_.u_stride();
    const int v_stride = velocity_.v_stride();

    const size_t u_total = velocity_.u_total_size();
    const size_t v_total = velocity_.v_total_size();
    const size_t w_total = velocity_.w_total_size();
    const size_t div_total = field_total_size_;

    // 3D path
    if (!mesh_->is2D()) {
        const double dz = mesh_->dz;
        const int u_plane_stride = velocity_.u_plane_stride();
        const int v_plane_stride = velocity_.v_plane_stride();
        const int w_stride = velocity_.w_stride();
        const int w_plane_stride = velocity_.w_plane_stride();
        const int div_plane_stride = mesh_->total_Nx() * mesh_->total_Ny();

        const int n_cells = Nx * Ny * Nz;

        // Select velocity field based on 'which' - NVHPC WORKAROUND: Use gpu::dev_ptr + is_device_ptr
        if (which == VelocityWhich::Current) {
            const double* u_dev = gpu::dev_ptr(velocity_u_ptr_);
            const double* v_dev = gpu::dev_ptr(velocity_v_ptr_);
            const double* w_dev = gpu::dev_ptr(velocity_w_ptr_);
            double* div_dev = gpu::dev_ptr(div_velocity_ptr_);

            if (use_O4) {
                #pragma omp target teams distribute parallel for is_device_ptr(u_dev, v_dev, w_dev, div_dev) \
                    firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, div_stride, div_plane_stride, Nx, Ny, Nz, Ng, x_periodic, y_periodic, z_periodic)
                for (int idx = 0; idx < n_cells; ++idx) {
                    const int i = idx % Nx + Ng;
                    const int j = (idx / Nx) % Ny + Ng;
                    const int k = idx / (Nx * Ny) + Ng;
                    divergence_cell_kernel_staggered_O4_3d(i, j, k, Ng, Nx, Ny, Nz,
                        u_stride, u_plane_stride, v_stride, v_plane_stride,
                        w_stride, w_plane_stride, div_stride, div_plane_stride,
                        dx, dy, dz, x_periodic, y_periodic, z_periodic,
                        u_dev, v_dev, w_dev, div_dev);
                }
            } else {
                #pragma omp target teams distribute parallel for is_device_ptr(u_dev, v_dev, w_dev, div_dev) \
                    firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, div_stride, div_plane_stride, Nx, Ny, Ng)
                for (int idx = 0; idx < n_cells; ++idx) {
                    const int i = idx % Nx + Ng;
                    const int j = (idx / Nx) % Ny + Ng;
                    const int k = idx / (Nx * Ny) + Ng;
                    divergence_cell_kernel_staggered_3d(i, j, k,
                        u_stride, u_plane_stride, v_stride, v_plane_stride,
                        w_stride, w_plane_stride, div_stride, div_plane_stride,
                        dx, dy, dz, u_dev, v_dev, w_dev, div_dev);
                }
            }
        } else {
            // VelocityWhich::Star path
            const double* u_dev = gpu::dev_ptr(velocity_star_u_ptr_);
            const double* v_dev = gpu::dev_ptr(velocity_star_v_ptr_);
            const double* w_dev = gpu::dev_ptr(velocity_star_w_ptr_);
            double* div_dev = gpu::dev_ptr(div_velocity_ptr_);

            if (use_O4) {
                #pragma omp target teams distribute parallel for is_device_ptr(u_dev, v_dev, w_dev, div_dev) \
                    firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, div_stride, div_plane_stride, Nx, Ny, Nz, Ng, x_periodic, y_periodic, z_periodic)
                for (int idx = 0; idx < n_cells; ++idx) {
                    const int i = idx % Nx + Ng;
                    const int j = (idx / Nx) % Ny + Ng;
                    const int k = idx / (Nx * Ny) + Ng;
                    divergence_cell_kernel_staggered_O4_3d(i, j, k, Ng, Nx, Ny, Nz,
                        u_stride, u_plane_stride, v_stride, v_plane_stride,
                        w_stride, w_plane_stride, div_stride, div_plane_stride,
                        dx, dy, dz, x_periodic, y_periodic, z_periodic,
                        u_dev, v_dev, w_dev, div_dev);
                }
            } else {
                #pragma omp target teams distribute parallel for is_device_ptr(u_dev, v_dev, w_dev, div_dev) \
                    firstprivate(dx, dy, dz, u_stride, v_stride, w_stride, u_plane_stride, v_plane_stride, w_plane_stride, div_stride, div_plane_stride, Nx, Ny, Ng)
                for (int idx = 0; idx < n_cells; ++idx) {
                    const int i = idx % Nx + Ng;
                    const int j = (idx / Nx) % Ny + Ng;
                    const int k = idx / (Nx * Ny) + Ng;
                    divergence_cell_kernel_staggered_3d(i, j, k,
                        u_stride, u_plane_stride, v_stride, v_plane_stride,
                        w_stride, w_plane_stride, div_stride, div_plane_stride,
                        dx, dy, dz, u_dev, v_dev, w_dev, div_dev);
                }
            }
        }
        return;
    }

    // 2D path - NVHPC WORKAROUND: Use gpu::dev_ptr + is_device_ptr
    const int n_cells = Nx * Ny;

    if (which == VelocityWhich::Current) {
        const double* u_dev = gpu::dev_ptr(velocity_u_ptr_);
        const double* v_dev = gpu::dev_ptr(velocity_v_ptr_);
        double* div_dev = gpu::dev_ptr(div_velocity_ptr_);

        #pragma omp target teams distribute parallel for is_device_ptr(u_dev, v_dev, div_dev) \
            firstprivate(dx, dy, u_stride, v_stride, div_stride, Nx, Ng)
        for (int idx = 0; idx < n_cells; ++idx) {
            const int i = idx % Nx + Ng;
            const int j = idx / Nx + Ng;

            const int u_right = j * u_stride + (i + 1);
            const int u_left = j * u_stride + i;
            const int v_top = (j + 1) * v_stride + i;
            const int v_bottom = j * v_stride + i;
            const int div_idx = j * div_stride + i;

            const double dudx = (u_dev[u_right] - u_dev[u_left]) / dx;
            const double dvdy = (v_dev[v_top] - v_dev[v_bottom]) / dy;
            div_dev[div_idx] = dudx + dvdy;
        }
    } else {
        // VelocityWhich::Star path - NVHPC WORKAROUND: Use gpu::dev_ptr + is_device_ptr
        const double* u_star_dev = gpu::dev_ptr(velocity_star_u_ptr_);
        const double* v_star_dev = gpu::dev_ptr(velocity_star_v_ptr_);
        double* div_dev = gpu::dev_ptr(div_velocity_ptr_);

        #pragma omp target teams distribute parallel for is_device_ptr(u_star_dev, v_star_dev, div_dev) \
            firstprivate(dx, dy, u_stride, v_stride, div_stride, Nx, Ng)
        for (int idx = 0; idx < n_cells; ++idx) {
            const int i = idx % Nx + Ng;
            const int j = idx / Nx + Ng;

            const int u_right = j * u_stride + (i + 1);
            const int u_left = j * u_stride + i;
            const int v_top = (j + 1) * v_stride + i;
            const int v_bottom = j * v_stride + i;
            const int div_idx = j * div_stride + i;

            const double dudx = (u_star_dev[u_right] - u_star_dev[u_left]) / dx;
            const double dvdy = (v_star_dev[v_top] - v_star_dev[v_bottom]) / dy;
            div_dev[div_idx] = dudx + dvdy;
        }
    }
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

    // O4 spatial discretization for pressure gradient (Dcf_O4)
    const bool use_O4 = (config_.space_order == 4);

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

    // 3D path - use MEMBER pointers directly (NVHPC workaround)
    // Local pointer aliases from get_solver_view() contain HOST addresses that NVHPC
    // doesn't translate correctly when passed to device functions. Use member pointers
    // directly in target regions instead.
    if (!mesh_->is2D()) {
        const double dz = mesh_->dz;
        const int u_plane_stride = velocity_.u_plane_stride();
        const int v_plane_stride = velocity_.v_plane_stride();
        const int w_stride_3d = velocity_.w_stride();
        const int w_plane_stride = velocity_.w_plane_stride();
        const int p_plane_stride = mesh_->total_Nx() * mesh_->total_Ny();

        // Correct u-velocities at x-faces (3D) - O2 pressure gradient, inlined
        // Read from velocity_star_u_ptr_ (predicted u*), write to velocity_u_ptr_ (output u^{n+1})
        // u^{n+1} = u* - dt * dp'/dx
        // NVHPC WORKAROUND: Use gpu::dev_ptr + is_device_ptr for correct device addresses
        const int n_u_faces = (Nx + 1) * Ny * Nz;
        {
            double* u_dev = gpu::dev_ptr(velocity_u_ptr_);
            const double* us_dev = gpu::dev_ptr(velocity_star_u_ptr_);
            const double* pc_dev = gpu::dev_ptr(pressure_corr_ptr_);

            #pragma omp target teams distribute parallel for is_device_ptr(u_dev, us_dev, pc_dev) \
                firstprivate(dx, dt, u_stride, u_plane_stride, p_stride, p_plane_stride, Nx, Ny, Ng)
            for (int idx = 0; idx < n_u_faces; ++idx) {
                int i = idx % (Nx + 1) + Ng;
                int j = (idx / (Nx + 1)) % Ny + Ng;
                int k = idx / ((Nx + 1) * Ny) + Ng;

                // u at x-face (i,j,k) straddles cells (i-1,j,k) and (i,j,k)
                const int u_idx = k * u_plane_stride + j * u_stride + i;
                const int p_right = k * p_plane_stride + j * p_stride + i;
                const int p_left = k * p_plane_stride + j * p_stride + (i - 1);

                double dp_dx = (pc_dev[p_right] - pc_dev[p_left]) / dx;
                u_dev[u_idx] = us_dev[u_idx] - dt * dp_dx;
            }
        }

        // Enforce x-periodicity (3D)
        if (x_periodic) {
            const int n_u_periodic = Ny * Nz;
            double* u_dev = gpu::dev_ptr(velocity_u_ptr_);

            #pragma omp target teams distribute parallel for is_device_ptr(u_dev) \
                firstprivate(u_stride, u_plane_stride, Nx, Ny, Ng)
            for (int idx = 0; idx < n_u_periodic; ++idx) {
                int j = idx % Ny + Ng;
                int k = idx / Ny + Ng;
                int i_left = Ng;
                int i_right = Ng + Nx;
                int idx_left = k * u_plane_stride + j * u_stride + i_left;
                int idx_right = k * u_plane_stride + j * u_stride + i_right;
                double u_avg = 0.5 * (u_dev[idx_left] + u_dev[idx_right]);
                u_dev[idx_left] = u_avg;
                u_dev[idx_right] = u_avg;
            }
        }

        // Correct v-velocities at y-faces (3D) - O2 pressure gradient, inlined
        // Read from velocity_star_v_ptr_ (predicted v*), write to velocity_v_ptr_ (output v^{n+1})
        // v^{n+1} = v* - dt * dp'/dy
        const int n_v_faces = Nx * (Ny + 1) * Nz;
        {
            double* v_dev = gpu::dev_ptr(velocity_v_ptr_);
            const double* vs_dev = gpu::dev_ptr(velocity_star_v_ptr_);
            const double* pc_dev = gpu::dev_ptr(pressure_corr_ptr_);

            #pragma omp target teams distribute parallel for is_device_ptr(v_dev, vs_dev, pc_dev) \
                firstprivate(dy, dt, v_stride, v_plane_stride, p_stride, p_plane_stride, Nx, Ny, Ng)
            for (int idx = 0; idx < n_v_faces; ++idx) {
                int i = idx % Nx + Ng;
                int j = (idx / Nx) % (Ny + 1) + Ng;
                int k = idx / (Nx * (Ny + 1)) + Ng;

                // v at y-face (i,j,k) straddles cells (i,j-1,k) and (i,j,k)
                const int v_idx = k * v_plane_stride + j * v_stride + i;
                const int p_top = k * p_plane_stride + j * p_stride + i;
                const int p_bottom = k * p_plane_stride + (j - 1) * p_stride + i;

                double dp_dy = (pc_dev[p_top] - pc_dev[p_bottom]) / dy;
                v_dev[v_idx] = vs_dev[v_idx] - dt * dp_dy;
            }
        }

        // Enforce y-periodicity (3D)
        if (y_periodic) {
            const int n_v_periodic = Nx * Nz;
            double* v_dev = gpu::dev_ptr(velocity_v_ptr_);

            #pragma omp target teams distribute parallel for is_device_ptr(v_dev) \
                firstprivate(v_stride, v_plane_stride, Nx, Ny, Ng)
            for (int idx = 0; idx < n_v_periodic; ++idx) {
                int i = idx % Nx + Ng;
                int k = idx / Nx + Ng;
                int j_bottom = Ng;
                int j_top = Ng + Ny;
                int idx_bottom = k * v_plane_stride + j_bottom * v_stride + i;
                int idx_top = k * v_plane_stride + j_top * v_stride + i;
                double v_avg = 0.5 * (v_dev[idx_bottom] + v_dev[idx_top]);
                v_dev[idx_bottom] = v_avg;
                v_dev[idx_top] = v_avg;
            }
        }

        // Correct w-velocities at z-faces (3D) - O2 pressure gradient, inlined
        const int n_w_faces = Nx * Ny * (Nz + 1);
        {
            double* w_dev = gpu::dev_ptr(velocity_w_ptr_);
            const double* ws_dev = gpu::dev_ptr(velocity_star_w_ptr_);
            const double* pc_dev = gpu::dev_ptr(pressure_corr_ptr_);

            #pragma omp target teams distribute parallel for is_device_ptr(w_dev, ws_dev, pc_dev) \
                firstprivate(dz, dt, w_stride_3d, w_plane_stride, p_stride, p_plane_stride, Nx, Ny, Ng)
            for (int idx = 0; idx < n_w_faces; ++idx) {
                int i = idx % Nx + Ng;
                int j = (idx / Nx) % Ny + Ng;
                int k = idx / (Nx * Ny) + Ng;

                // w at z-face (i,j,k) straddles cells (i,j,k-1) and (i,j,k)
                const int w_idx = k * w_plane_stride + j * w_stride_3d + i;
                const int p_front = k * p_plane_stride + j * p_stride + i;
                const int p_back = (k - 1) * p_plane_stride + j * p_stride + i;

                double dp_dz = (pc_dev[p_front] - pc_dev[p_back]) / dz;
                w_dev[w_idx] = ws_dev[w_idx] - dt * dp_dz;
            }
        }

        // Enforce z-periodicity (3D)
        if (z_periodic) {
            const int n_w_periodic = Nx * Ny;
            double* w_dev = gpu::dev_ptr(velocity_w_ptr_);

            #pragma omp target teams distribute parallel for is_device_ptr(w_dev) \
                firstprivate(w_stride_3d, w_plane_stride, Nx, Nz, Ng)
            for (int idx = 0; idx < n_w_periodic; ++idx) {
                int i = idx % Nx + Ng;
                int j = idx / Nx + Ng;
                int k_front = Ng;
                int k_back = Ng + Nz;
                int idx_front = k_front * w_plane_stride + j * w_stride_3d + i;
                int idx_back = k_back * w_plane_stride + j * w_stride_3d + i;
                double w_avg = 0.5 * (w_dev[idx_front] + w_dev[idx_back]);
                w_dev[idx_front] = w_avg;
                w_dev[idx_back] = w_avg;
            }
        }

        // Update pressure at cell centers (3D)
        const int n_cells = Nx * Ny * Nz;
        {
            double* p_dev = gpu::dev_ptr(pressure_ptr_);
            const double* pc_dev = gpu::dev_ptr(pressure_corr_ptr_);

            #pragma omp target teams distribute parallel for is_device_ptr(p_dev, pc_dev) \
                firstprivate(p_stride, p_plane_stride, Nx, Ny, Ng)
            for (int idx = 0; idx < n_cells; ++idx) {
                int i = idx % Nx + Ng;
                int j = (idx / Nx) % Ny + Ng;
                int k = idx / (Nx * Ny) + Ng;

                int p_idx = k * p_plane_stride + j * p_stride + i;
                p_dev[p_idx] += pc_dev[p_idx];
            }
        }

        NVTX_POP();
        return;
    }

    // 2D path - NVHPC WORKAROUND: use gpu::dev_ptr for device addresses
    // Use linear indexing to avoid compiler issues with collapse(2)
    const int n_u_faces = (Nx + 1) * Ny;
    const int n_v_faces = Nx * (Ny + 1);
    const int n_cells = Nx * Ny;

    // Get device pointers via gpu::dev_ptr
    double* u_corr_dev = gpu::dev_ptr(velocity_u_ptr_);
    const double* u_star_dev = gpu::dev_ptr(velocity_star_u_ptr_);
    double* v_corr_dev = gpu::dev_ptr(velocity_v_ptr_);
    const double* v_star_dev = gpu::dev_ptr(velocity_star_v_ptr_);
    const double* p_corr_dev = gpu::dev_ptr(pressure_corr_ptr_);
    double* p_dev = gpu::dev_ptr(pressure_ptr_);

    // Correct ALL u-velocities at x-faces (including redundant face if periodic)
    // Read from u_star_dev (predicted u*), write to u_corr_dev (output u^{n+1})
    // u^{n+1} = u* - dt * dp'/dx
    #pragma omp target teams distribute parallel for is_device_ptr(u_corr_dev, u_star_dev, p_corr_dev) \
        firstprivate(dx, dt, u_stride, p_stride, Nx, Ng)
    for (int idx = 0; idx < n_u_faces; ++idx) {
        int i_local = idx % (Nx + 1);
        int j_local = idx / (Nx + 1);
        int i = i_local + Ng;
        int j = j_local + Ng;

        // u(i,j) is at x-face between cells (i-1,j) and (i,j)
        const int u_idx = j * u_stride + i;
        const int p_right = j * p_stride + i;
        const int p_left = j * p_stride + (i - 1);

        double dp_dx = (p_corr_dev[p_right] - p_corr_dev[p_left]) / dx;
        u_corr_dev[u_idx] = u_star_dev[u_idx] - dt * dp_dx;
    }

    // Enforce exact x-periodicity: average the left and right edge values
    if (x_periodic) {
        #pragma omp target teams distribute parallel for is_device_ptr(u_corr_dev) \
            firstprivate(u_stride, Nx, Ng)
        for (int j_local = 0; j_local < Ny; ++j_local) {
            int j = j_local + Ng;
            int i_left = Ng;
            int i_right = Ng + Nx;
            double u_avg = 0.5 * (u_corr_dev[j * u_stride + i_left] + u_corr_dev[j * u_stride + i_right]);
            u_corr_dev[j * u_stride + i_left] = u_avg;
            u_corr_dev[j * u_stride + i_right] = u_avg;
        }
    }

    // Correct ALL v-velocities at y-faces (including redundant face if periodic)
    // Read from v_star_dev (predicted v*), write to v_corr_dev (output v^{n+1})
    // v^{n+1} = v* - dt * dp'/dy
    #pragma omp target teams distribute parallel for is_device_ptr(v_corr_dev, v_star_dev, p_corr_dev) \
        firstprivate(dy, dt, v_stride, p_stride, Nx, Ng)
    for (int idx = 0; idx < n_v_faces; ++idx) {
        int i_local = idx % Nx;
        int j_local = idx / Nx;
        int i = i_local + Ng;
        int j = j_local + Ng;

        // v(i,j) is at y-face between cells (i,j-1) and (i,j)
        const int v_idx = j * v_stride + i;
        const int p_top = j * p_stride + i;
        const int p_bottom = (j - 1) * p_stride + i;

        double dp_dy = (p_corr_dev[p_top] - p_corr_dev[p_bottom]) / dy;
        v_corr_dev[v_idx] = v_star_dev[v_idx] - dt * dp_dy;
    }

    // Enforce exact y-periodicity: average the bottom and top edge values
    if (y_periodic) {
        #pragma omp target teams distribute parallel for is_device_ptr(v_corr_dev) \
            firstprivate(v_stride, Ny, Ng)
        for (int i_local = 0; i_local < Nx; ++i_local) {
            int i = i_local + Ng;
            int j_bottom = Ng;
            int j_top = Ng + Ny;
            double v_avg = 0.5 * (v_corr_dev[j_bottom * v_stride + i] + v_corr_dev[j_top * v_stride + i]);
            v_corr_dev[j_bottom * v_stride + i] = v_avg;
            v_corr_dev[j_top * v_stride + i] = v_avg;
        }
    }

    // Update pressure at cell centers
    #pragma omp target teams distribute parallel for is_device_ptr(p_dev, p_corr_dev) \
        firstprivate(p_stride, Nx, Ng)
    for (int idx = 0; idx < n_cells; ++idx) {
        int i = idx % Nx + Ng;
        int j = idx / Nx + Ng;

        const int p_idx = j * p_stride + i;
        p_dev[p_idx] += p_corr_dev[p_idx];
    }

    NVTX_POP();  // End correct_velocity
}

// Note: correct_velocity(vel_in, vel_out) is implemented in solver_time.cpp
// to keep this file under nvc++ complexity limits

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
    NVTX_SCOPE_SOLVER("time_step");

#ifndef NDEBUG
    // Verify GPU mappings haven't been invalidated by std::vector reallocation
    verify_mapping_integrity();
#endif

    // Store old velocity for convergence check (at face locations for staggered grid)
    const int Ng = mesh_->Nghost;
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    
// Unified CPU/GPU path: copy current velocity to velocity_old using raw pointers
    {
    NVTX_PUSH("velocity_copy");
    [[maybe_unused]] const size_t u_total_size = velocity_.u_total_size();
    [[maybe_unused]] const size_t v_total_size = velocity_.v_total_size();
    const int u_stride = Nx + 2 * Ng + 1;
    const int v_stride = Nx + 2 * Ng;

    if (mesh_->is2D()) {
#ifdef USE_GPU_OFFLOAD
        // NVHPC WORKAROUND: Use gpu::dev_ptr + is_device_ptr for correct device addresses
        const int n_u_copy = (Nx + 1) * Ny;
        const int n_v_copy = Nx * (Ny + 1);

        {
            const double* u_src_dev = gpu::dev_ptr(velocity_u_ptr_);
            double* u_dst_dev = gpu::dev_ptr(velocity_old_u_ptr_);
            const double* v_src_dev = gpu::dev_ptr(velocity_v_ptr_);
            double* v_dst_dev = gpu::dev_ptr(velocity_old_v_ptr_);

            // Copy u-velocity (2D) - linear indexing
            #pragma omp target teams distribute parallel for is_device_ptr(u_src_dev, u_dst_dev) \
                firstprivate(u_stride, Ng, Nx)
            for (int lin = 0; lin < n_u_copy; ++lin) {
                int i_local = lin % (Nx + 1);
                int j_local = lin / (Nx + 1);
                int i = i_local + Ng;
                int j = j_local + Ng;
                const int idx = j * u_stride + i;
                u_dst_dev[idx] = u_src_dev[idx];
            }

            // Copy v-velocity (2D) - linear indexing
            #pragma omp target teams distribute parallel for is_device_ptr(v_src_dev, v_dst_dev) \
                firstprivate(v_stride, Ng, Nx)
            for (int lin = 0; lin < n_v_copy; ++lin) {
                int i_local = lin % Nx;
                int j_local = lin / Nx;
                int i = i_local + Ng;
                int j = j_local + Ng;
                const int idx = j * v_stride + i;
                v_dst_dev[idx] = v_src_dev[idx];
            }
        }
#else
        for (int j = Ng; j < Ng + Ny; ++j) {
            for (int i = Ng; i <= Ng + Nx; ++i) {
                const int idx = j * u_stride + i;
                velocity_old_u_ptr_[idx] = velocity_u_ptr_[idx];
            }
        }
        for (int j = Ng; j <= Ng + Ny; ++j) {
            for (int i = Ng; i < Ng + Nx; ++i) {
                const int idx = j * v_stride + i;
                velocity_old_v_ptr_[idx] = velocity_v_ptr_[idx];
            }
        }
#endif
    } else {
        // 3D path - copy u, v, AND w
        const int Nz = mesh_->Nz;
        const int u_plane_stride = u_stride * (Ny + 2*Ng);
        const int v_plane_stride = v_stride * (Ny + 2*Ng + 1);
        [[maybe_unused]] const size_t w_total_size = velocity_.w_total_size();
        const int w_stride = Nx + 2*Ng;
        const int w_plane_stride = w_stride * (Ny + 2*Ng);

#ifdef USE_GPU_OFFLOAD
        // NVHPC WORKAROUND: Use gpu::dev_ptr + is_device_ptr for correct device addresses
        const int n_u_copy_3d = (Nx + 1) * Ny * Nz;
        const int n_v_copy_3d = Nx * (Ny + 1) * Nz;
        const int n_w_copy_3d = Nx * Ny * (Nz + 1);

        {
            const double* u_src_dev = gpu::dev_ptr(velocity_u_ptr_);
            double* u_dst_dev = gpu::dev_ptr(velocity_old_u_ptr_);
            const double* v_src_dev = gpu::dev_ptr(velocity_v_ptr_);
            double* v_dst_dev = gpu::dev_ptr(velocity_old_v_ptr_);
            const double* w_src_dev = gpu::dev_ptr(velocity_w_ptr_);
            double* w_dst_dev = gpu::dev_ptr(velocity_old_w_ptr_);

            // Copy u-velocity (3D) - linear indexing
            #pragma omp target teams distribute parallel for is_device_ptr(u_src_dev, u_dst_dev) \
                firstprivate(u_stride, u_plane_stride, Ng, Nx, Ny)
            for (int lin = 0; lin < n_u_copy_3d; ++lin) {
                int i_local = lin % (Nx + 1);
                int j_local = (lin / (Nx + 1)) % Ny;
                int k_local = lin / ((Nx + 1) * Ny);
                int i = i_local + Ng;
                int j = j_local + Ng;
                int k = k_local + Ng;
                const int idx = k * u_plane_stride + j * u_stride + i;
                u_dst_dev[idx] = u_src_dev[idx];
            }

            // Copy v-velocity (3D) - linear indexing
            #pragma omp target teams distribute parallel for is_device_ptr(v_src_dev, v_dst_dev) \
                firstprivate(v_stride, v_plane_stride, Ng, Nx, Ny)
            for (int lin = 0; lin < n_v_copy_3d; ++lin) {
                int i_local = lin % Nx;
                int j_local = (lin / Nx) % (Ny + 1);
                int k_local = lin / (Nx * (Ny + 1));
                int i = i_local + Ng;
                int j = j_local + Ng;
                int k = k_local + Ng;
                const int idx = k * v_plane_stride + j * v_stride + i;
                v_dst_dev[idx] = v_src_dev[idx];
            }

            // Copy w-velocity (3D) - linear indexing
            #pragma omp target teams distribute parallel for is_device_ptr(w_src_dev, w_dst_dev) \
                firstprivate(w_stride, w_plane_stride, Ng, Nx, Ny)
            for (int lin = 0; lin < n_w_copy_3d; ++lin) {
                int i_local = lin % Nx;
                int j_local = (lin / Nx) % Ny;
                int k_local = lin / (Nx * Ny);
                int i = i_local + Ng;
                int j = j_local + Ng;
                int k = k_local + Ng;
                const int idx = k * w_plane_stride + j * w_stride + i;
                w_dst_dev[idx] = w_src_dev[idx];
            }
        }
#else
        for (int k = Ng; k < Ng + Nz; ++k) {
            for (int j = Ng; j < Ng + Ny; ++j) {
                for (int i = Ng; i <= Ng + Nx; ++i) {
                    const int idx = k * u_plane_stride + j * u_stride + i;
                    velocity_old_u_ptr_[idx] = velocity_u_ptr_[idx];
                }
            }
        }
        for (int k = Ng; k < Ng + Nz; ++k) {
            for (int j = Ng; j <= Ng + Ny; ++j) {
                for (int i = Ng; i < Ng + Nx; ++i) {
                    const int idx = k * v_plane_stride + j * v_stride + i;
                    velocity_old_v_ptr_[idx] = velocity_v_ptr_[idx];
                }
            }
        }
        for (int k = Ng; k <= Ng + Nz; ++k) {
            for (int j = Ng; j < Ng + Ny; ++j) {
                for (int i = Ng; i < Ng + Nx; ++i) {
                    const int idx = k * w_plane_stride + j * w_stride + i;
                    velocity_old_w_ptr_[idx] = velocity_w_ptr_[idx];
                }
            }
        }
#endif
    }
    NVTX_POP();
    }
    
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
        const bool is_2d = mesh_->is2D();

        // NVHPC WORKAROUND: Use gpu::dev_ptr to get actual device addresses.
        // Local pointer copies of member pointers get HOST addresses in NVHPC.
        double* nu_eff_dev = gpu::dev_ptr(nu_eff_ptr_);
        const double* nu_t_dev = turb_model_ ? gpu::dev_ptr(nu_t_ptr_) : nullptr;

        if (is_2d) {
            // 2D path
            const int n_cells = Nx * Ny;
            if (turb_model_) {
                #pragma omp target teams distribute parallel for \
                    is_device_ptr(nu_eff_dev, nu_t_dev) \
                    firstprivate(nu, stride, Nx, Ng)
                for (int idx = 0; idx < n_cells; ++idx) {
                    int i = idx % Nx + Ng;
                    int j = idx / Nx + Ng;
                    int cell_idx = j * stride + i;
                    nu_eff_dev[cell_idx] = nu + nu_t_dev[cell_idx];
                }
            } else {
                #pragma omp target teams distribute parallel for \
                    is_device_ptr(nu_eff_dev) \
                    firstprivate(nu, stride, Nx, Ng)
                for (int idx = 0; idx < n_cells; ++idx) {
                    int i = idx % Nx + Ng;
                    int j = idx / Nx + Ng;
                    int cell_idx = j * stride + i;
                    nu_eff_dev[cell_idx] = nu;
                }
            }
        } else {
            // 3D path
            const int n_cells = Nx * Ny * Nz;
            if (turb_model_) {
                #pragma omp target teams distribute parallel for \
                    is_device_ptr(nu_eff_dev, nu_t_dev) \
                    firstprivate(nu, stride, plane_stride, Nx, Ny, Ng)
                for (int idx = 0; idx < n_cells; ++idx) {
                    int i = idx % Nx + Ng;
                    int j = (idx / Nx) % Ny + Ng;
                    int k = idx / (Nx * Ny) + Ng;
                    int cell_idx = k * plane_stride + j * stride + i;
                    nu_eff_dev[cell_idx] = nu + nu_t_dev[cell_idx];
                }
            } else {
                #pragma omp target teams distribute parallel for \
                    is_device_ptr(nu_eff_dev) \
                    firstprivate(nu, stride, plane_stride, Nx, Ny, Ng)
                for (int idx = 0; idx < n_cells; ++idx) {
                    int i = idx % Nx + Ng;
                    int j = (idx / Nx) % Ny + Ng;
                    int k = idx / (Nx * Ny) + Ng;
                    int cell_idx = k * plane_stride + j * stride + i;
                    nu_eff_dev[cell_idx] = nu;
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

    // Dispatch to appropriate time integrator
    // RK methods handle their own convective/diffusive terms and projection
    if (config_.time_integrator != TimeIntegrator::Euler) {
        if (config_.time_integrator == TimeIntegrator::RK3) {
            ssprk3_step(current_dt_);
        } else if (config_.time_integrator == TimeIntegrator::RK2) {
            ssprk2_step(current_dt_);
        }

        // Compute residual for RK methods
        // Note: Post-step divergence check and NaN guard are still done below
        // Fall through to residual computation
    } else {
    // =========== Euler time integration path (default) ===========
    // 2. Compute convective and diffusive terms (use persistent fields)
    {
        TIMED_SCOPE("convective_term");
        NVTX_PUSH("convection");
        compute_convective_term(velocity_, conv_);
        NVTX_POP();

        // Convective KE production diagnostic: <u, conv(u)>
        // For skew-symmetric form, this should be ~0 (energy conservative)
        static bool conv_ke_diagnostics = (std::getenv("NNCFD_CONV_KE_DIAGNOSTICS") != nullptr);
        if (conv_ke_diagnostics && (iter_ % 100 == 0)) {
#ifdef USE_GPU_OFFLOAD
            // Need to sync conv_ and velocity_ to host for the diagnostic
            sync_from_gpu();  // Syncs all fields including conv_ and velocity_
#endif
            double dke_conv = compute_convective_ke_production();
            double ke = 0.0;
            const double dV = mesh_->dx * mesh_->dy * (mesh_->is2D() ? 1.0 : mesh_->dz);
            if (mesh_->is2D()) {
                for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                    for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                        double u = 0.5 * (velocity_.u(i, j) + velocity_.u(i+1, j));
                        double v = 0.5 * (velocity_.v(i, j) + velocity_.v(i, j+1));
                        ke += 0.5 * (u*u + v*v) * dV;
                    }
                }
            } else {
                for (int k = mesh_->k_begin(); k < mesh_->k_end(); ++k) {
                    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                            double u = 0.5 * (velocity_.u(i, j, k) + velocity_.u(i+1, j, k));
                            double v = 0.5 * (velocity_.v(i, j, k) + velocity_.v(i, j+1, k));
                            double w = 0.5 * (velocity_.w(i, j, k) + velocity_.w(i, j, k+1));
                            ke += 0.5 * (u*u + v*v + w*w) * dV;
                        }
                    }
                }
            }
            // Normalize by KE to get fractional rate
            double rel_rate = (ke > 1e-30) ? dke_conv / ke : 0.0;
            std::cout << "[Convection] dKE/dt_conv=" << std::scientific << std::setprecision(6)
                      << dke_conv << " (rel=" << rel_rate << "/s)\n";
        }
    }

    {
        TIMED_SCOPE("diffusive_term");
        NVTX_PUSH("diffusion");
        compute_diffusive_term(velocity_, nu_eff_, diff_);
        NVTX_POP();
    }
    
    // 3. Compute provisional velocity u* (without pressure gradient) at face locations
    // u* = u^n + dt * (-conv + diff + body_force)
    // IMPORTANT: Use MEMBER pointers directly inside target regions. Local pointer copies
    // don't work correctly with NVHPC - they copy HOST addresses that fail on device.
    NVTX_PUSH("predictor_step");

    // Get strides and other params (but NOT pointer copies)
    const int u_stride_pred = velocity_.u_stride();
    const int v_stride_pred = velocity_.v_stride();
    const double dt = current_dt_;
    const double fx = fx_;
    const double fy = fy_;

    const bool x_periodic = (velocity_bc_.x_lo == VelocityBC::Periodic) &&
                            (velocity_bc_.x_hi == VelocityBC::Periodic);
    const bool y_periodic = (velocity_bc_.y_lo == VelocityBC::Periodic) &&
                            (velocity_bc_.y_hi == VelocityBC::Periodic);

    [[maybe_unused]] const size_t u_total_size_pred = velocity_.u_total_size();
    [[maybe_unused]] const size_t v_total_size_pred = velocity_.v_total_size();

    const bool is_2d_pred = mesh_->is2D();
    const int Nz_pred = mesh_->Nz;
    const int Nz_eff_pred = is_2d_pred ? 1 : Nz_pred;
    // Avoid reading uninitialized strides in 2D mode (set to 0 if 2D)
    const int u_plane_stride_pred = is_2d_pred ? 0 : velocity_.u_plane_stride();
    const int v_plane_stride_pred = is_2d_pred ? 0 : velocity_.v_plane_stride();

    // Compute u* at ALL x-faces (including redundant if periodic)
    // NVHPC WORKAROUND: Use gpu::dev_ptr for actual device addresses
    const int n_u_faces_pred = (Nx + 1) * Ny * Nz_eff_pred;
    {
        const double* u_dev = gpu::dev_ptr(velocity_u_ptr_);
        double* u_star_dev = gpu::dev_ptr(velocity_star_u_ptr_);
        const double* conv_u_dev = gpu::dev_ptr(conv_u_ptr_);
        const double* diff_u_dev = gpu::dev_ptr(diff_u_ptr_);

        #pragma omp target teams distribute parallel for is_device_ptr(u_dev, u_star_dev, conv_u_dev, diff_u_dev) \
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

            u_star_dev[u_idx] = u_dev[u_idx] + dt * (-conv_u_dev[u_idx] + diff_u_dev[u_idx] + fx);
        }
    }

    // Enforce exact x-periodicity for u*: average left and right edges
    // NVHPC WORKAROUND: use gpu::dev_ptr for device address
    if (x_periodic) {
        const int n_u_periodic = Ny * Nz_eff_pred;
        double* u_star_periodic = gpu::dev_ptr(velocity_star_u_ptr_);
        #pragma omp target teams distribute parallel for is_device_ptr(u_star_periodic) \
            firstprivate(u_stride_pred, u_plane_stride_pred, Nx, Ny, Nz_eff_pred, Ng, is_2d_pred)
        for (int idx = 0; idx < n_u_periodic; ++idx) {
            int j_local = idx % Ny;
            int k_local = idx / Ny;
            int j = j_local + Ng;
            int k = k_local + Ng;
            int base = is_2d_pred ? (j * u_stride_pred)
                                  : (k * u_plane_stride_pred + j * u_stride_pred);
            double u_avg = 0.5 * (u_star_periodic[base + Ng] + u_star_periodic[base + (Ng + Nx)]);
            u_star_periodic[base + Ng] = u_avg;
            u_star_periodic[base + (Ng + Nx)] = u_avg;
        }
    }

    // Compute v* at ALL y-faces (including redundant if periodic)
    // NVHPC WORKAROUND: Use gpu::dev_ptr for actual device addresses
    const int n_v_faces_pred = Nx * (Ny + 1) * Nz_eff_pred;
    {
        const double* v_dev = gpu::dev_ptr(velocity_v_ptr_);
        double* v_star_dev = gpu::dev_ptr(velocity_star_v_ptr_);
        const double* conv_v_dev = gpu::dev_ptr(conv_v_ptr_);
        const double* diff_v_dev = gpu::dev_ptr(diff_v_ptr_);

        #pragma omp target teams distribute parallel for is_device_ptr(v_dev, v_star_dev, conv_v_dev, diff_v_dev) \
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

            v_star_dev[v_idx] = v_dev[v_idx] + dt * (-conv_v_dev[v_idx] + diff_v_dev[v_idx] + fy);
        }
    }

    // Enforce exact y-periodicity for v*: average bottom and top edges
    // NVHPC WORKAROUND: use gpu::dev_ptr for device address
    if (y_periodic) {
        const int n_v_periodic = Nx * Nz_eff_pred;
        double* v_star_periodic = gpu::dev_ptr(velocity_star_v_ptr_);
        #pragma omp target teams distribute parallel for is_device_ptr(v_star_periodic) \
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
            double v_avg = 0.5 * (v_star_periodic[base_lo] + v_star_periodic[base_hi]);
            v_star_periodic[base_lo] = v_avg;
            v_star_periodic[base_hi] = v_avg;
        }
    }

    // 3D: Compute w* at ALL z-faces
    // Use MEMBER pointers directly with nested target data + target pattern
    if (!mesh_->is2D()) {
        const int Nz = mesh_->Nz;
        const int w_stride_pred = velocity_.w_stride();
        const int w_plane_stride_pred = velocity_.w_plane_stride();
        const double fz = fz_;
        [[maybe_unused]] const size_t w_total_size_pred = velocity_.w_total_size();

        const bool z_periodic = (velocity_bc_.z_lo == VelocityBC::Periodic) &&
                                (velocity_bc_.z_hi == VelocityBC::Periodic);

        // Compute w* = w + dt * (-conv_w + diff_w + fz)
        // NVHPC WORKAROUND: Use gpu::dev_ptr + is_device_ptr for correct device addresses
        const int n_w_faces_pred = Nx * Ny * (Nz + 1);
        {
            const double* w_dev = gpu::dev_ptr(velocity_w_ptr_);
            double* w_star_dev = gpu::dev_ptr(velocity_star_w_ptr_);
            const double* conv_w_dev = gpu::dev_ptr(conv_w_ptr_);
            const double* diff_w_dev = gpu::dev_ptr(diff_w_ptr_);

            #pragma omp target teams distribute parallel for is_device_ptr(w_dev, w_star_dev, conv_w_dev, diff_w_dev) \
                firstprivate(dt, fz, w_stride_pred, w_plane_stride_pred, Nx, Ny, Ng)
            for (int idx = 0; idx < n_w_faces_pred; ++idx) {
                int i = idx % Nx + Ng;
                int j = (idx / Nx) % Ny + Ng;
                int k = idx / (Nx * Ny) + Ng;
                int w_idx = k * w_plane_stride_pred + j * w_stride_pred + i;

                w_star_dev[w_idx] = w_dev[w_idx] + dt * (-conv_w_dev[w_idx] + diff_w_dev[w_idx] + fz);
            }
        }

        // Enforce exact z-periodicity for w*: average front and back edges
        if (z_periodic) {
            const int n_w_periodic = Nx * Ny;
            double* w_star_dev = gpu::dev_ptr(velocity_star_w_ptr_);

            #pragma omp target teams distribute parallel for is_device_ptr(w_star_dev) \
                firstprivate(w_stride_pred, w_plane_stride_pred, Nx, Nz, Ng)
            for (int idx = 0; idx < n_w_periodic; ++idx) {
                int i = idx % Nx + Ng;
                int j = idx / Nx + Ng;
                int idx_back = Ng * w_plane_stride_pred + j * w_stride_pred + i;
                int idx_front = (Ng + Nz) * w_plane_stride_pred + j * w_stride_pred + i;
                double w_avg = 0.5 * (w_star_dev[idx_back] + w_star_dev[idx_front]);
                w_star_dev[idx_back] = w_avg;
                w_star_dev[idx_front] = w_avg;
            }
        }
    }

    // NO SWAP! Keep predicted velocity in velocity_star_ / velocity_star_u_ptr_.
    // This avoids NVHPC GPU pointer issues where swapped member pointers confuse
    // the device mapping in target update directives.
    //
    // For fully periodic domains, the inline periodic averaging above handles BCs.
    // For non-periodic domains, we'd need apply_velocity_bc_star() (not implemented yet).
    // TODO: Implement apply_velocity_bc_star() for non-periodic boundary support.
    const bool z_periodic_check = mesh_->is2D() ||
                                  ((velocity_bc_.z_lo == VelocityBC::Periodic) &&
                                   (velocity_bc_.z_hi == VelocityBC::Periodic));
    const bool fully_periodic = x_periodic && y_periodic && z_periodic_check;
    if (!fully_periodic) {
        // Non-periodic domains not yet supported with this GPU path
        // Fall back to swap approach (known broken on GPU, but works on CPU)
        std::swap(velocity_, velocity_star_);
#ifdef USE_GPU_OFFLOAD
        std::swap(velocity_u_ptr_, velocity_star_u_ptr_);
        std::swap(velocity_v_ptr_, velocity_star_v_ptr_);
        if (!mesh_->is2D()) {
            std::swap(velocity_w_ptr_, velocity_star_w_ptr_);
        }
#endif
        apply_velocity_bc();
        std::swap(velocity_, velocity_star_);
#ifdef USE_GPU_OFFLOAD
        std::swap(velocity_u_ptr_, velocity_star_u_ptr_);
        std::swap(velocity_v_ptr_, velocity_star_v_ptr_);
        if (!mesh_->is2D()) {
            std::swap(velocity_w_ptr_, velocity_star_w_ptr_);
        }
#endif
    }
    // After this block: velocity_star_ / velocity_star_u_ptr_ contains u* (predicted)
    //                   velocity_ / velocity_u_ptr_ contains u^n (old)

    NVTX_POP();  // End predictor_step

    // 4. Solve pressure Poisson equation
    // nabla^2p' = (1/dt) nabla*u*
    // For fully periodic: velocity_star_ contains u*, use VelocityWhich::Star
    // For non-periodic: after swap-back, velocity_star_ still contains u*
    {
        TIMED_SCOPE("divergence");
        NVTX_PUSH("divergence");
        compute_divergence(VelocityWhich::Star, div_velocity_);
        NVTX_POP();
    }

    // Build RHS on GPU and subtract mean divergence to ensure solvability
    // GPU-RESIDENT OPTIMIZATION: Keep all data on device, only transfer scalars
    NVTX_PUSH("poisson_rhs_build");
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

        // NVHPC-safe pattern: Use MEMBER pointers directly inside target regions.
        // Local pointer aliases don't work correctly - NVHPC passes HOST addresses
        // instead of translating to device addresses.
        // Use linear indexing (not collapse) to avoid NVHPC ICE.
        const size_t n_field = field_total_size_;

        double sum_div = 0.0;
        const int n_cells = is_2d ? (Nx * Ny) : (Nx * Ny * Nz);

        // NVHPC WORKAROUND: Use gpu::dev_ptr + is_device_ptr for correct device addresses
        const double* div_dev_sum = gpu::dev_ptr(div_velocity_ptr_);

        if (is_2d) {
            // 2D path
            #pragma omp target teams distribute parallel for is_device_ptr(div_dev_sum) \
                reduction(+:sum_div) firstprivate(stride, i_begin, j_begin, Nx)
            for (int lin = 0; lin < n_cells; ++lin) {
                int i = lin % Nx;
                int j = lin / Nx;
                int ii = i + i_begin;
                int jj = j + j_begin;
                int idx = jj * stride + ii;
                sum_div += div_dev_sum[idx];
            }
        } else {
            // 3D path
            #pragma omp target teams distribute parallel for is_device_ptr(div_dev_sum) \
                reduction(+:sum_div) firstprivate(stride, plane_stride, i_begin, j_begin, k_begin, Nx, Ny)
            for (int lin = 0; lin < n_cells; ++lin) {
                int i = lin % Nx;
                int j = (lin / Nx) % Ny;
                int k = lin / (Nx * Ny);
                int ii = i + i_begin;
                int jj = j + j_begin;
                int kk = k + k_begin;
                int idx = kk * plane_stride + jj * stride + ii;
                sum_div += div_dev_sum[idx];
            }
        }

        mean_div = (n_cells > 0) ? sum_div / n_cells : 0.0;

        // Build RHS on GPU: rhs = (div - mean_div) / dt
        const double dt_inv = 1.0 / current_dt_;

        // NVHPC WORKAROUND: Use omp_get_mapped_ptr for actual device addresses
        int device = omp_get_default_device();
        const double* div_dev = static_cast<const double*>(omp_get_mapped_ptr(div_velocity_ptr_, device));
        double* rhs_dev = static_cast<double*>(omp_get_mapped_ptr(rhs_poisson_ptr_, device));

        if (is_2d) {
            // 2D path
            #pragma omp target teams distribute parallel for is_device_ptr(div_dev, rhs_dev) \
                firstprivate(stride, i_begin, j_begin, Nx, mean_div, dt_inv)
            for (int lin = 0; lin < n_cells; ++lin) {
                int i = lin % Nx;
                int j = lin / Nx;
                int ii = i + i_begin;
                int jj = j + j_begin;
                int idx = jj * stride + ii;
                rhs_dev[idx] = (div_dev[idx] - mean_div) * dt_inv;
            }
        } else {
            // 3D path
            #pragma omp target teams distribute parallel for is_device_ptr(div_dev, rhs_dev) \
                firstprivate(stride, plane_stride, i_begin, j_begin, k_begin, Nx, Ny, mean_div, dt_inv)
            for (int lin = 0; lin < n_cells; ++lin) {
                int i = lin % Nx;
                int j = (lin / Nx) % Ny;
                int k = lin / (Nx * Ny);
                int ii = i + i_begin;
                int jj = j + j_begin;
                int kk = k + k_begin;
                int idx = kk * plane_stride + jj * stride + ii;
                rhs_dev[idx] = (div_dev[idx] - mean_div) * dt_inv;
            }
        }

        // OPTIMIZATION: Warm-start for Poisson solver (device-resident)
        // Zero pressure correction on device on first iteration only
        if (iter_ == 0) {
            // NVHPC WORKAROUND: Use gpu::dev_ptr + is_device_ptr for correct device addresses
            double* pcorr_dev = gpu::dev_ptr(pressure_corr_ptr_);
            #pragma omp target teams distribute parallel for is_device_ptr(pcorr_dev)
            for (size_t idx = 0; idx < n_field; ++idx) {
                pcorr_dev[idx] = 0.0;
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
    NVTX_POP();  // End poisson_rhs_build

    // 4b. Solve Poisson equation for pressure correction
    {
        TIMED_SCOPE("poisson_solve");
        NVTX_PUSH("poisson_solve");
        
        // CRITICAL: Use relative tolerance for Poisson solver (standard multigrid practice)
        // When turbulence changes effective viscosity, RHS magnitude varies significantly
        // Absolute tolerance would be too strict for small RHS, too loose for large RHS
        double rhs_norm_sq = 0.0;
        int rhs_count = 0;

// Compute RHS norm for tolerance calculation
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

#ifdef USE_GPU_OFFLOAD
            if (gpu_ready_) {
                // NVHPC WORKAROUND: Use gpu::dev_ptr + is_device_ptr for correct device addresses
                const double* rhs_dev = gpu::dev_ptr(rhs_poisson_ptr_);

                if (mesh_->is2D()) {
                    #pragma omp target teams distribute parallel for collapse(2) is_device_ptr(rhs_dev) \
                        reduction(+:rhs_norm_sq, rhs_count)
                    for (int j = 0; j < Ny; ++j) {
                        for (int i = 0; i < Nx; ++i) {
                            int ii = i + i_begin;
                            int jj = j + j_begin;
                            int idx = jj * stride + ii;
                            double rhs_val = rhs_dev[idx];
                            rhs_norm_sq += rhs_val * rhs_val;
                            rhs_count++;
                        }
                    }
                } else {
                    #pragma omp target teams distribute parallel for collapse(3) is_device_ptr(rhs_dev) \
                        reduction(+:rhs_norm_sq, rhs_count)
                    for (int k = 0; k < Nz; ++k) {
                        for (int j = 0; j < Ny; ++j) {
                            for (int i = 0; i < Nx; ++i) {
                                int ii = i + i_begin;
                                int jj = j + j_begin;
                                int kk = k + k_begin;
                                int idx = kk * plane_stride + jj * stride + ii;
                                double rhs_val = rhs_dev[idx];
                                rhs_norm_sq += rhs_val * rhs_val;
                                rhs_count++;
                            }
                        }
                    }
                }
            } else
#endif
            {
                // CPU path
                if (mesh_->is2D()) {
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
        }
        
        double rhs_rms = std::sqrt(rhs_norm_sq / std::max(rhs_count, 1));
        
        // Configure Poisson solver with robust convergence criteria
        // The MG solver now supports three convergence criteria (any triggers exit):
        //   1. ||r||_∞ ≤ tol_abs  (absolute, usually disabled)
        //   2. ||r||/||b|| ≤ tol_rhs  (RHS-relative, recommended for projection)
        //   3. ||r||/||r0|| ≤ tol_rel  (initial-residual relative, backup)
        PoissonConfig pcfg;
        pcfg.max_vcycles = config_.poisson_max_vcycles;
        pcfg.omega = config_.poisson_omega;
        pcfg.verbose = false;  // Disable per-cycle output (too verbose)

        // New robust tolerance parameters (preferred for MG)
        pcfg.tol_abs = config_.poisson_tol_abs;
        pcfg.tol_rhs = config_.poisson_tol_rhs;
        pcfg.tol_rel = config_.poisson_tol_rel;
        pcfg.check_interval = config_.poisson_check_interval;
        pcfg.use_l2_norm = config_.poisson_use_l2_norm;
        pcfg.linf_safety_factor = config_.poisson_linf_safety;
        pcfg.fixed_cycles = config_.poisson_fixed_cycles;
        pcfg.adaptive_cycles = config_.poisson_adaptive_cycles;
        pcfg.check_after = config_.poisson_check_after;
        pcfg.nu1 = config_.poisson_nu1;
        pcfg.nu2 = config_.poisson_nu2;
        pcfg.chebyshev_degree = config_.poisson_chebyshev_degree;
        pcfg.use_vcycle_graph = config_.poisson_use_vcycle_graph;

        // Legacy tolerance for backward compatibility (non-MG solvers use this)
        double relative_tol = config_.poisson_tol * std::max(rhs_rms, 1e-12);
        double effective_tol = std::max(relative_tol, config_.poisson_abs_tol_floor);
        pcfg.tol = effective_tol;
        
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
                case PoissonSolverType::FFT2D:
                    if (fft2d_poisson_solver_) {
                        if (!solver_logged) {
                            std::cout << "[Poisson] Using FFT2D solve_device() (1D cuFFT + cuSPARSE tridiag)\n";
                            solver_logged = true;
                        }
                        cycles = fft2d_poisson_solver_->solve_device(rhs_poisson_ptr_, pressure_corr_ptr_, pcfg);
                        final_residual = fft2d_poisson_solver_->residual();
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
                      << " residual=" << std::scientific << std::setprecision(6)
                      << final_residual;
            // For MG solver, also print norms and ratios for convergence analysis
            if (selected_solver_ == PoissonSolverType::MG) {
                // Get both L∞ and L2 norms
                double r_inf = mg_poisson_solver_.residual();
                double r_l2 = mg_poisson_solver_.residual_l2();
                double b_inf = mg_poisson_solver_.rhs_norm();
                double b_l2 = mg_poisson_solver_.rhs_norm_l2();
                double r0_inf = mg_poisson_solver_.initial_residual();
                double r0_l2 = mg_poisson_solver_.initial_residual_l2();

                // Show which norm is used for convergence
                const char* norm_type = pcfg.use_l2_norm ? "L2" : "Linf";
                double r_norm = pcfg.use_l2_norm ? r_l2 : r_inf;
                double b_norm = pcfg.use_l2_norm ? b_l2 : b_inf;
                double r0_norm = pcfg.use_l2_norm ? r0_l2 : r0_inf;
                double r_over_b = (b_norm > 1e-30) ? r_norm / b_norm : 0.0;
                double r_over_r0 = (r0_norm > 1e-30) ? r_norm / r0_norm : 0.0;
                std::cout << " [" << norm_type << "] ||b||=" << b_norm
                          << " ||r0||=" << r0_norm
                          << " ||r||/||b||=" << r_over_b
                          << " ||r||/||r0||=" << r_over_r0;
            }
            std::cout << "\n";
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
    // NVHPC WORKAROUND: apply_velocity_bc() uses broken local pointer pattern.
    // Skip for fully periodic domains where no ghost cell updates are needed.
    // correct_velocity() already handles periodic averaging at boundaries.
    {
        const bool x_per = (velocity_bc_.x_lo == VelocityBC::Periodic) &&
                           (velocity_bc_.x_hi == VelocityBC::Periodic);
        const bool y_per = (velocity_bc_.y_lo == VelocityBC::Periodic) &&
                           (velocity_bc_.y_hi == VelocityBC::Periodic);
        const bool z_per = mesh_->is2D() ||
                           ((velocity_bc_.z_lo == VelocityBC::Periodic) &&
                            (velocity_bc_.z_hi == VelocityBC::Periodic));
        const bool fully_per = x_per && y_per && z_per;
        if (!fully_per) {
            apply_velocity_bc();
        }
    }

    } // End Euler time integration path

    // Post-projection divergence check (diagnostic only)
    // This is the actual measure of projection quality: max|div(u^{n+1})|
    {
        static bool div_diagnostics = (std::getenv("NNCFD_POISSON_DIAGNOSTICS") != nullptr);
        static int div_diagnostics_interval = []() {
            const char* env = std::getenv("NNCFD_POISSON_DIAGNOSTICS_INTERVAL");
            int v = env ? std::atoi(env) : 1;
            return (v > 0) ? v : 1;
        }();
        if (div_diagnostics && (iter_ % div_diagnostics_interval == 0)) {
            compute_divergence(VelocityWhich::Current, div_velocity_);  // Divergence of corrected velocity
            double max_div = 0.0;
            double sum_div2 = 0.0;  // For L2 norm
            int n_cells = 0;
            if (mesh_->is2D()) {
                const double dV = mesh_->dx * mesh_->dy;  // Cell volume (uniform)
                for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                    for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                        double d = div_velocity_(i, j);
                        max_div = std::max(max_div, std::abs(d));
                        sum_div2 += d * d * dV;
                        n_cells++;
                    }
                }
            } else {
                const double dV = mesh_->dx * mesh_->dy * mesh_->dz;  // Cell volume
                for (int k = mesh_->k_begin(); k < mesh_->k_end(); ++k) {
                    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                            double d = div_velocity_(i, j, k);
                            max_div = std::max(max_div, std::abs(d));
                            sum_div2 += d * d * dV;
                            n_cells++;
                        }
                    }
                }
            }
            double l2_div = std::sqrt(sum_div2);
            std::cout << "[Projection] ||div(u)||_Linf=" << std::scientific << std::setprecision(6)
                      << max_div << " ||div(u)||_L2=" << l2_div
                      << " dt*Linf=" << current_dt_ * max_div << "\n";
        }
    }
    
    // Note: iter_ is managed by the outer solve loop, don't increment here

    // Return max velocity change as convergence criterion (unified view-based)
    NVTX_PUSH("residual_computation");
    auto v_res = get_solver_view();

    [[maybe_unused]] const size_t u_total_size_res = velocity_.u_total_size();
    [[maybe_unused]] const size_t v_total_size_res = velocity_.v_total_size();
    [[maybe_unused]] const size_t w_total_size_res = velocity_.w_total_size();
    const int u_stride_res = v_res.u_stride;
    const int v_stride_res = v_res.v_stride;
    const int Nz = mesh_->Nz;
    const bool is_2d_res = mesh_->is2D();
    const int Nz_eff = is_2d_res ? 1 : Nz;  // Effective Nz for loop bounds

    // NVHPC WORKAROUND: Use gpu::dev_ptr to get actual device addresses.
    // Local pointer copies from get_solver_view() get HOST addresses in NVHPC.
    const double* u_new_dev = gpu::dev_ptr(v_res.u_face);
    const double* v_new_dev = gpu::dev_ptr(v_res.v_face);
    const double* u_old_dev = gpu::dev_ptr(v_res.u_old_face);
    const double* v_old_dev = gpu::dev_ptr(v_res.v_old_face);

    // Compute max |u_new - u_old| via reduction
    const int n_u_faces_res = (Nx + 1) * Ny * Nz_eff;
    const int u_plane_stride_res = is_2d_res ? 0 : v_res.u_plane_stride;
    double max_du = 0.0;
    #pragma omp target teams distribute parallel for reduction(max:max_du) \
        is_device_ptr(u_new_dev, u_old_dev) \
        firstprivate(Ng, u_stride_res, u_plane_stride_res, Nx, Ny, Nz_eff, is_2d_res)
    for (int idx = 0; idx < n_u_faces_res; ++idx) {
        int i_local = idx % (Nx + 1);
        int j_local = (idx / (Nx + 1)) % Ny;
        int k_local = idx / ((Nx + 1) * Ny);
        int i = i_local + Ng;
        int j = j_local + Ng;
        int k = k_local + Ng;
        int u_idx = is_2d_res ? (j * u_stride_res + i)
                              : (k * u_plane_stride_res + j * u_stride_res + i);
        double du = u_new_dev[u_idx] - u_old_dev[u_idx];
        if (du < 0.0) du = -du;
        if (du > max_du) max_du = du;
    }

    // Compute max |v_new - v_old| via reduction
    const int n_v_faces_res = Nx * (Ny + 1) * Nz_eff;
    const int v_plane_stride_res = is_2d_res ? 0 : v_res.v_plane_stride;
    double max_dv = 0.0;
    #pragma omp target teams distribute parallel for reduction(max:max_dv) \
        is_device_ptr(v_new_dev, v_old_dev) \
        firstprivate(Ng, v_stride_res, v_plane_stride_res, Nx, Ny, Nz_eff, is_2d_res)
    for (int idx = 0; idx < n_v_faces_res; ++idx) {
        int i_local = idx % Nx;
        int j_local = (idx / Nx) % (Ny + 1);
        int k_local = idx / (Nx * (Ny + 1));
        int i = i_local + Ng;
        int j = j_local + Ng;
        int k = k_local + Ng;
        int v_idx = is_2d_res ? (j * v_stride_res + i)
                              : (k * v_plane_stride_res + j * v_stride_res + i);
        double dv = v_new_dev[v_idx] - v_old_dev[v_idx];
        if (dv < 0.0) dv = -dv;
        if (dv > max_dv) max_dv = dv;
    }

    double max_change = (max_du > max_dv) ? max_du : max_dv;

    // For 3D, also check w component
    if (!is_2d_res) {
        const int w_stride_res = v_res.w_stride;
        const int w_plane_stride_res = v_res.w_plane_stride;
        const int n_w_faces_res = Nx * Ny * (Nz + 1);
        // NVHPC WORKAROUND: Use gpu::dev_ptr for w pointers too
        const double* w_new_dev = gpu::dev_ptr(v_res.w_face);
        const double* w_old_dev = gpu::dev_ptr(v_res.w_old_face);
        double max_dw = 0.0;
        #pragma omp target teams distribute parallel for reduction(max:max_dw) \
            is_device_ptr(w_new_dev, w_old_dev) \
            firstprivate(Ng, w_stride_res, w_plane_stride_res, Nx, Ny, Nz)
        for (int idx = 0; idx < n_w_faces_res; ++idx) {
            int i_local = idx % Nx;
            int j_local = (idx / Nx) % Ny;
            int k_local = idx / (Nx * Ny);
            int i = i_local + Ng;
            int j = j_local + Ng;
            int k = k_local + Ng;
            int w_idx = k * w_plane_stride_res + j * w_stride_res + i;
            double dw = w_new_dev[w_idx] - w_old_dev[w_idx];
            if (dw < 0.0) dw = -dw;
            if (dw > max_dw) max_dw = dw;
        }
        if (max_dw > max_change) max_change = max_dw;
    }

    NVTX_POP();  // End residual_computation

    // NaN/Inf GUARD: Check for numerical stability issues
    // Do this after turbulence update but before next iteration starts
    check_for_nan_inf(step_count_);
    ++step_count_;

    return max_change;
}

// ============================================================================
// RK Time Integration Methods
// These methods are implemented in solver_time.cpp to reduce compilation unit
// complexity for GPU builds with nvc++.
// See: euler_substep, project_velocity, ssprk2_step, ssprk3_step
// ============================================================================

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
    
    for (iter_ = 0; iter_ < config_.max_steps; ++iter_) {
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
        snapshot_freq = std::max(1, config_.max_steps / num_snapshots);
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

    // Progress output interval for CI visibility (always enabled)
    const int progress_interval = std::max(1, config_.max_steps / 10);

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

    for (iter_ = 0; iter_ < config_.max_steps; ++iter_) {
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
        
        // Always show progress every ~10% for CI visibility
        if ((iter_ + 1) % progress_interval == 0 || iter_ == 0) {
            std::cout << "    Iter " << std::setw(6) << iter_ + 1 << " / " << config_.max_steps
                      << "  (" << std::setw(3) << (100 * (iter_ + 1) / config_.max_steps) << "%)"
                      << "  residual = " << std::scientific << std::setprecision(3) << residual
                      << std::fixed << "\n" << std::flush;
        } else if (config_.verbose && (iter_ + 1) % config_.output_freq == 0) {
            // Detailed verbose output
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
        const int u_stride = Nx + 2*Ng + 1;

        // NVHPC WORKAROUND: Use gpu::dev_ptr to get actual device addresses.
        // Member pointers in target regions get HOST addresses in NVHPC.
        const double* u_dev = gpu::dev_ptr(velocity_u_ptr_);

        #pragma omp target teams distribute parallel for collapse(2) \
            is_device_ptr(u_dev) \
            reduction(+:sum)
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                int ii = i + Ng;
                int jj = j + Ng;
                sum += u_dev[jj * u_stride + ii];
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
        const int u_stride = Nx + 2*Ng + 1;

        // NVHPC WORKAROUND: Use gpu::dev_ptr to get actual device addresses.
        // Member pointers in target regions get HOST addresses in NVHPC.
        const double* u_dev = gpu::dev_ptr(velocity_u_ptr_);

        #pragma omp target teams distribute parallel for \
            is_device_ptr(u_dev) \
            reduction(+:sum)
        for (int i = 0; i < Nx; ++i) {
            int ii = i + Ng;
            // u at wall is 0 (no-slip), so dudy = u[j_wall] / dist
            double dudy = u_dev[j_wall * u_stride + ii] / dist;
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

double RANSSolver::compute_convective_ke_production() const {
    // Compute <u, conv(u)> = rate of KE change due to advection
    // For skew-symmetric advection with div(u)=0, this should be ~0
    //
    // IMPORTANT: For periodic directions, the last face wraps to the first,
    // so we only sum over Nx (not Nx+1) unique faces to avoid double counting.
    // For non-periodic (wall) directions, all Ny+1 faces are unique.

    double sum = 0.0;
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Ng = mesh_->Nghost;
    const double dV = mesh_->dx * mesh_->dy * (mesh_->is2D() ? 1.0 : mesh_->dz);

    // Determine periodicity from velocity BCs
    const bool x_periodic = (velocity_bc_.x_lo == VelocityBC::Periodic);
    const bool y_periodic = (velocity_bc_.y_lo == VelocityBC::Periodic);
    const bool z_periodic = (velocity_bc_.z_lo == VelocityBC::Periodic);

    // Face counts: periodic dirs have N unique faces, non-periodic have N+1
    const int n_u_x = x_periodic ? Nx : (Nx + 1);  // u-faces in x
    const int n_v_y = y_periodic ? Ny : (Ny + 1);  // v-faces in y
    const int n_w_z = z_periodic ? Nz : (Nz + 1);  // w-faces in z

    if (mesh_->is2D()) {
        // 2D: u-faces + v-faces
        // u-faces: n_u_x faces in x, Ny cells in y
        for (int j = Ng; j < Ng + Ny; ++j) {
            for (int i = Ng; i < Ng + n_u_x; ++i) {
                sum += velocity_.u(i, j) * conv_.u(i, j) * dV;
            }
        }
        // v-faces: Nx cells in x, n_v_y faces in y
        for (int j = Ng; j < Ng + n_v_y; ++j) {
            for (int i = Ng; i < Ng + Nx; ++i) {
                sum += velocity_.v(i, j) * conv_.v(i, j) * dV;
            }
        }
    } else {
        // 3D: u-faces + v-faces + w-faces
        // u-faces: n_u_x in x, Ny in y, Nz in z
        for (int k = Ng; k < Ng + Nz; ++k) {
            for (int j = Ng; j < Ng + Ny; ++j) {
                for (int i = Ng; i < Ng + n_u_x; ++i) {
                    sum += velocity_.u(i, j, k) * conv_.u(i, j, k) * dV;
                }
            }
        }
        // v-faces: Nx in x, n_v_y in y, Nz in z
        for (int k = Ng; k < Ng + Nz; ++k) {
            for (int j = Ng; j < Ng + n_v_y; ++j) {
                for (int i = Ng; i < Ng + Nx; ++i) {
                    sum += velocity_.v(i, j, k) * conv_.v(i, j, k) * dV;
                }
            }
        }
        // w-faces: Nx in x, Ny in y, n_w_z in z
        for (int k = Ng; k < Ng + n_w_z; ++k) {
            for (int j = Ng; j < Ng + Ny; ++j) {
                for (int i = Ng; i < Ng + Nx; ++i) {
                    sum += velocity_.w(i, j, k) * conv_.w(i, j, k) * dV;
                }
            }
        }
    }

    return sum;
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

        // NVHPC WORKAROUND: Use gpu::dev_ptr to get actual device addresses.
        // Local pointer copies of member pointers get HOST addresses in NVHPC.
        const double* u_dev = gpu::dev_ptr(velocity_u_ptr_);
        const double* v_dev = gpu::dev_ptr(velocity_v_ptr_);
        const double* p_dev = gpu::dev_ptr(pressure_ptr_);
        const double* nut_dev = gpu::dev_ptr(nu_t_ptr_);

        // Check u-velocity (x-faces)
        #pragma omp target teams distribute parallel for is_device_ptr(u_dev) reduction(|: has_bad)
        for (size_t idx = 0; idx < u_total; ++idx) {
            const double x = u_dev[idx];
            // Use manual NaN/Inf check (x != x for NaN, or x-x != 0 for Inf)
            has_bad |= (x != x || (x - x) != 0.0) ? 1 : 0;
        }

        // Check v-velocity (y-faces)
        #pragma omp target teams distribute parallel for is_device_ptr(v_dev) reduction(|: has_bad)
        for (size_t idx = 0; idx < v_total; ++idx) {
            const double x = v_dev[idx];
            has_bad |= (x != x || (x - x) != 0.0) ? 1 : 0;
        }

        // Check pressure and eddy viscosity (cell-centered)
        #pragma omp target teams distribute parallel for is_device_ptr(p_dev, nut_dev) reduction(|: has_bad)
        for (size_t idx = 0; idx < field_total; ++idx) {
            const double pval = p_dev[idx];
            const double nutval = nut_dev[idx];
            has_bad |= (pval != pval || (pval - pval) != 0.0 || nutval != nutval || (nutval - nutval) != 0.0) ? 1 : 0;
        }

        // Check transport variables if turbulence model uses them
        if (has_transport) {
            const double* k_dev = gpu::dev_ptr(k_ptr_);
            const double* omega_dev = gpu::dev_ptr(omega_ptr_);
            #pragma omp target teams distribute parallel for is_device_ptr(k_dev, omega_dev) \
                reduction(|: has_bad)
            for (size_t idx = 0; idx < field_total; ++idx) {
                const double kval = k_dev[idx];
                const double wval = omega_dev[idx];
                has_bad |= (kval != kval || (kval - kval) != 0.0 || wval != wval || (wval - wval) != 0.0) ? 1 : 0;
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
    
// Unified CPU/GPU path: compute CFL and diffusive constraints using raw pointers
    double u_max = 1e-10;
    double nu_eff_max = nu;

    [[maybe_unused]] const size_t u_total_size = velocity_.u_total_size();
    [[maybe_unused]] const size_t v_total_size = velocity_.v_total_size();
    [[maybe_unused]] const size_t field_total_size = field_total_size_;
    const int u_stride = Nx + 2*Ng + 1;
    const int v_stride = Nx + 2*Ng;
    const int stride = Nx + 2*Ng;

    if (mesh_->is2D()) {
#ifdef USE_GPU_OFFLOAD
        // NVHPC WORKAROUND: Use gpu::dev_ptr to get actual device addresses.
        // Local pointer copies of member pointers get HOST addresses in NVHPC.
        const double* u_dev = gpu::dev_ptr(velocity_u_ptr_);
        const double* v_dev = gpu::dev_ptr(velocity_v_ptr_);
        const double* nut_dev = turb_model_ ? gpu::dev_ptr(nu_t_ptr_) : nullptr;

        // 2D: Compute max velocity magnitude (for advective CFL)
        #pragma omp target teams distribute parallel for collapse(2) \
            is_device_ptr(u_dev, v_dev) reduction(max:u_max)
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                int ii = i + Ng;
                int jj = j + Ng;
                // Interpolate u and v to cell center for staggered grid
                double u_avg = 0.5 * (u_dev[jj * u_stride + ii] + u_dev[jj * u_stride + ii + 1]);
                double v_avg = 0.5 * (v_dev[jj * v_stride + ii] + v_dev[(jj + 1) * v_stride + ii]);
                double u_mag = sqrt(u_avg*u_avg + v_avg*v_avg);
                if (u_mag > u_max) u_max = u_mag;
            }
        }

        // 2D: Compute max effective viscosity (for diffusive CFL) if turbulence active
        if (turb_model_) {
            #pragma omp target teams distribute parallel for collapse(2) \
                is_device_ptr(nut_dev) reduction(max:nu_eff_max)
            for (int j = 0; j < Ny; ++j) {
                for (int i = 0; i < Nx; ++i) {
                    int ii = i + Ng;
                    int jj = j + Ng;
                    int idx = jj * stride + ii;
                    double nu_eff = nu + nut_dev[idx];
                    if (nu_eff > nu_eff_max) nu_eff_max = nu_eff;
                }
            }
        }
#else
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                int ii = i + Ng;
                int jj = j + Ng;
                double u_avg = 0.5 * (velocity_u_ptr_[jj * u_stride + ii] +
                                      velocity_u_ptr_[jj * u_stride + ii + 1]);
                double v_avg = 0.5 * (velocity_v_ptr_[jj * v_stride + ii] +
                                      velocity_v_ptr_[(jj + 1) * v_stride + ii]);
                double u_mag = sqrt(u_avg*u_avg + v_avg*v_avg);
                if (u_mag > u_max) u_max = u_mag;
            }
        }
        if (turb_model_) {
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
#endif
    } else {
        // 3D case
        const int Nz = mesh_->Nz;
        [[maybe_unused]] const size_t w_total_size = velocity_.w_total_size();
        const int u_plane_stride = u_stride * (Ny + 2*Ng);
        const int v_plane_stride = v_stride * (Ny + 2*Ng + 1);
        const int w_stride = Nx + 2*Ng;
        const int w_plane_stride = w_stride * (Ny + 2*Ng);
        const int plane_stride = stride * (Ny + 2*Ng);

#ifdef USE_GPU_OFFLOAD
        // NVHPC WORKAROUND: Use gpu::dev_ptr to get actual device addresses.
        // Local pointer copies of member pointers get HOST addresses in NVHPC.
        const double* u_dev = gpu::dev_ptr(velocity_u_ptr_);
        const double* v_dev = gpu::dev_ptr(velocity_v_ptr_);
        const double* w_dev = gpu::dev_ptr(velocity_w_ptr_);
        const double* nut_dev = turb_model_ ? gpu::dev_ptr(nu_t_ptr_) : nullptr;

        // 3D: Compute max velocity magnitude (for advective CFL)
        #pragma omp target teams distribute parallel for collapse(3) \
            is_device_ptr(u_dev, v_dev, w_dev) reduction(max:u_max)
        for (int k = 0; k < Nz; ++k) {
            for (int j = 0; j < Ny; ++j) {
                for (int i = 0; i < Nx; ++i) {
                    int ii = i + Ng;
                    int jj = j + Ng;
                    int kk = k + Ng;
                    // Interpolate u, v, w to cell center for staggered grid
                    double u_avg = 0.5 * (u_dev[kk * u_plane_stride + jj * u_stride + ii] +
                                          u_dev[kk * u_plane_stride + jj * u_stride + ii + 1]);
                    double v_avg = 0.5 * (v_dev[kk * v_plane_stride + jj * v_stride + ii] +
                                          v_dev[kk * v_plane_stride + (jj + 1) * v_stride + ii]);
                    double w_avg = 0.5 * (w_dev[kk * w_plane_stride + jj * w_stride + ii] +
                                          w_dev[(kk + 1) * w_plane_stride + jj * w_stride + ii]);
                    double u_mag = sqrt(u_avg*u_avg + v_avg*v_avg + w_avg*w_avg);
                    if (u_mag > u_max) u_max = u_mag;
                }
            }
        }

        // 3D: Compute max effective viscosity (for diffusive CFL) if turbulence active
        if (turb_model_) {
            #pragma omp target teams distribute parallel for collapse(3) \
                is_device_ptr(nut_dev) reduction(max:nu_eff_max)
            for (int k = 0; k < Nz; ++k) {
                for (int j = 0; j < Ny; ++j) {
                    for (int i = 0; i < Nx; ++i) {
                        int ii = i + Ng;
                        int jj = j + Ng;
                        int kk = k + Ng;
                        int idx = kk * plane_stride + jj * stride + ii;
                        double nu_eff = nu + nut_dev[idx];
                        if (nu_eff > nu_eff_max) nu_eff_max = nu_eff;
                    }
                }
            }
        }
#else
        for (int k = 0; k < Nz; ++k) {
            for (int j = 0; j < Ny; ++j) {
                for (int i = 0; i < Nx; ++i) {
                    int ii = i + Ng;
                    int jj = j + Ng;
                    int kk = k + Ng;
                    double u_avg = 0.5 * (velocity_u_ptr_[kk * u_plane_stride + jj * u_stride + ii] +
                                          velocity_u_ptr_[kk * u_plane_stride + jj * u_stride + ii + 1]);
                    double v_avg = 0.5 * (velocity_v_ptr_[kk * v_plane_stride + jj * v_stride + ii] +
                                          velocity_v_ptr_[kk * v_plane_stride + (jj + 1) * v_stride + ii]);
                    double w_avg = 0.5 * (velocity_w_ptr_[kk * w_plane_stride + jj * w_stride + ii] +
                                          velocity_w_ptr_[(kk + 1) * w_plane_stride + jj * w_stride + ii]);
                    double u_mag = sqrt(u_avg*u_avg + v_avg*v_avg + w_avg*w_avg);
                    if (u_mag > u_max) u_max = u_mag;
                }
            }
        }
        if (turb_model_) {
            for (int k = 0; k < Nz; ++k) {
                for (int j = 0; j < Ny; ++j) {
                    for (int i = 0; i < Nx; ++i) {
                        int ii = i + Ng;
                        int jj = j + Ng;
                        int kk = k + Ng;
                        int idx = kk * plane_stride + jj * stride + ii;
                        double nu_eff = nu + nu_t_ptr_[idx];
                        if (nu_eff > nu_eff_max) nu_eff_max = nu_eff;
                    }
                }
            }
        }
#endif
    }

    // Compute time step constraints (same for GPU and CPU)
    double dx_min = mesh_->is2D() ? std::min(mesh_->dx, mesh_->dy)
                                  : std::min({mesh_->dx, mesh_->dy, mesh_->dz});
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

        // Vorticity (omega_z = dv/dx - du/dy) - scalar in 2D
        // NOTE: Uses uniform dx/dy spacing. On stretched meshes, this is an approximation
        // suitable for visualization but not metrically consistent with the solver discretization.
        // Guard: skip vorticity output for degenerate meshes (need >= 2 cells per direction)
        const int nx_2d = mesh_->i_end() - mesh_->i_begin();
        const int ny_2d = mesh_->j_end() - mesh_->j_begin();
        if (nx_2d >= 2 && ny_2d >= 2) {
        auto compute_vorticity_2d = [&](int i, int j) -> double {
            // dv/dx
            double dvdx;
            if (periodic_x) {
                int im = (i == mesh_->i_begin()) ? mesh_->i_end() - 1 : i - 1;
                int ip = (i == mesh_->i_end() - 1) ? mesh_->i_begin() : i + 1;
                dvdx = (velocity_.v_center(ip, j) - velocity_.v_center(im, j)) / (2.0 * mesh_->dx);
            } else {
                if (i == mesh_->i_begin()) {
                    dvdx = (velocity_.v_center(i + 1, j) - velocity_.v_center(i, j)) / mesh_->dx;
                } else if (i == mesh_->i_end() - 1) {
                    dvdx = (velocity_.v_center(i, j) - velocity_.v_center(i - 1, j)) / mesh_->dx;
                } else {
                    dvdx = (velocity_.v_center(i + 1, j) - velocity_.v_center(i - 1, j)) / (2.0 * mesh_->dx);
                }
            }

            // du/dy
            double dudy;
            if (periodic_y) {
                int jm = (j == mesh_->j_begin()) ? mesh_->j_end() - 1 : j - 1;
                int jp = (j == mesh_->j_end() - 1) ? mesh_->j_begin() : j + 1;
                dudy = (velocity_.u_center(i, jp) - velocity_.u_center(i, jm)) / (2.0 * mesh_->dy);
            } else {
                if (j == mesh_->j_begin()) {
                    dudy = (velocity_.u_center(i, j + 1) - velocity_.u_center(i, j)) / mesh_->dy;
                } else if (j == mesh_->j_end() - 1) {
                    dudy = (velocity_.u_center(i, j) - velocity_.u_center(i, j - 1)) / mesh_->dy;
                } else {
                    dudy = (velocity_.u_center(i, j + 1) - velocity_.u_center(i, j - 1)) / (2.0 * mesh_->dy);
                }
            }

            return dvdx - dudy;
        };

        file << "SCALARS vorticity double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                file << compute_vorticity_2d(i, j) << "\n";
            }
        }

        // Vorticity magnitude (same as |omega_z| in 2D)
        file << "SCALARS vorticity_magnitude double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                file << std::abs(compute_vorticity_2d(i, j)) << "\n";
            }
        }
        } // end degenerate mesh guard for 2D vorticity
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

        // Vorticity (vector in 3D):
        //   omega_x = dw/dy - dv/dz
        //   omega_y = du/dz - dw/dx
        //   omega_z = dv/dx - du/dy
        // NOTE: Uses uniform dx/dy/dz spacing. On stretched meshes, this is an approximation
        // suitable for visualization but not metrically consistent with the solver discretization.
        // Guard: skip vorticity/Q-criterion output for degenerate meshes (need >= 2 cells per direction)
        const int nx_3d = mesh_->i_end() - mesh_->i_begin();
        const int ny_3d = mesh_->j_end() - mesh_->j_begin();
        const int nz_3d = mesh_->k_end() - mesh_->k_begin();
        if (nx_3d >= 2 && ny_3d >= 2 && nz_3d >= 2) {

        // Helper lambda for dw/dy
        auto compute_dwdy = [&](int i, int j, int k) -> double {
            if (periodic_y) {
                int jm = (j == mesh_->j_begin()) ? mesh_->j_end() - 1 : j - 1;
                int jp = (j == mesh_->j_end() - 1) ? mesh_->j_begin() : j + 1;
                return (velocity_.w_center(i, jp, k) - velocity_.w_center(i, jm, k)) / (2.0 * mesh_->dy);
            } else {
                if (j == mesh_->j_begin()) {
                    return (velocity_.w_center(i, j + 1, k) - velocity_.w_center(i, j, k)) / mesh_->dy;
                } else if (j == mesh_->j_end() - 1) {
                    return (velocity_.w_center(i, j, k) - velocity_.w_center(i, j - 1, k)) / mesh_->dy;
                } else {
                    return (velocity_.w_center(i, j + 1, k) - velocity_.w_center(i, j - 1, k)) / (2.0 * mesh_->dy);
                }
            }
        };

        // Helper lambda for dv/dz
        auto compute_dvdz = [&](int i, int j, int k) -> double {
            if (periodic_z) {
                int km = (k == mesh_->k_begin()) ? mesh_->k_end() - 1 : k - 1;
                int kp = (k == mesh_->k_end() - 1) ? mesh_->k_begin() : k + 1;
                return (velocity_.v_center(i, j, kp) - velocity_.v_center(i, j, km)) / (2.0 * mesh_->dz);
            } else {
                if (k == mesh_->k_begin()) {
                    return (velocity_.v_center(i, j, k + 1) - velocity_.v_center(i, j, k)) / mesh_->dz;
                } else if (k == mesh_->k_end() - 1) {
                    return (velocity_.v_center(i, j, k) - velocity_.v_center(i, j, k - 1)) / mesh_->dz;
                } else {
                    return (velocity_.v_center(i, j, k + 1) - velocity_.v_center(i, j, k - 1)) / (2.0 * mesh_->dz);
                }
            }
        };

        // Helper lambda for du/dz
        auto compute_dudz = [&](int i, int j, int k) -> double {
            if (periodic_z) {
                int km = (k == mesh_->k_begin()) ? mesh_->k_end() - 1 : k - 1;
                int kp = (k == mesh_->k_end() - 1) ? mesh_->k_begin() : k + 1;
                return (velocity_.u_center(i, j, kp) - velocity_.u_center(i, j, km)) / (2.0 * mesh_->dz);
            } else {
                if (k == mesh_->k_begin()) {
                    return (velocity_.u_center(i, j, k + 1) - velocity_.u_center(i, j, k)) / mesh_->dz;
                } else if (k == mesh_->k_end() - 1) {
                    return (velocity_.u_center(i, j, k) - velocity_.u_center(i, j, k - 1)) / mesh_->dz;
                } else {
                    return (velocity_.u_center(i, j, k + 1) - velocity_.u_center(i, j, k - 1)) / (2.0 * mesh_->dz);
                }
            }
        };

        // Helper lambda for dw/dx
        auto compute_dwdx = [&](int i, int j, int k) -> double {
            if (periodic_x) {
                int im = (i == mesh_->i_begin()) ? mesh_->i_end() - 1 : i - 1;
                int ip = (i == mesh_->i_end() - 1) ? mesh_->i_begin() : i + 1;
                return (velocity_.w_center(ip, j, k) - velocity_.w_center(im, j, k)) / (2.0 * mesh_->dx);
            } else {
                if (i == mesh_->i_begin()) {
                    return (velocity_.w_center(i + 1, j, k) - velocity_.w_center(i, j, k)) / mesh_->dx;
                } else if (i == mesh_->i_end() - 1) {
                    return (velocity_.w_center(i, j, k) - velocity_.w_center(i - 1, j, k)) / mesh_->dx;
                } else {
                    return (velocity_.w_center(i + 1, j, k) - velocity_.w_center(i - 1, j, k)) / (2.0 * mesh_->dx);
                }
            }
        };

        // Helper lambda for dv/dx
        auto compute_dvdx = [&](int i, int j, int k) -> double {
            if (periodic_x) {
                int im = (i == mesh_->i_begin()) ? mesh_->i_end() - 1 : i - 1;
                int ip = (i == mesh_->i_end() - 1) ? mesh_->i_begin() : i + 1;
                return (velocity_.v_center(ip, j, k) - velocity_.v_center(im, j, k)) / (2.0 * mesh_->dx);
            } else {
                if (i == mesh_->i_begin()) {
                    return (velocity_.v_center(i + 1, j, k) - velocity_.v_center(i, j, k)) / mesh_->dx;
                } else if (i == mesh_->i_end() - 1) {
                    return (velocity_.v_center(i, j, k) - velocity_.v_center(i - 1, j, k)) / mesh_->dx;
                } else {
                    return (velocity_.v_center(i + 1, j, k) - velocity_.v_center(i - 1, j, k)) / (2.0 * mesh_->dx);
                }
            }
        };

        // Helper lambda for du/dy
        auto compute_dudy = [&](int i, int j, int k) -> double {
            if (periodic_y) {
                int jm = (j == mesh_->j_begin()) ? mesh_->j_end() - 1 : j - 1;
                int jp = (j == mesh_->j_end() - 1) ? mesh_->j_begin() : j + 1;
                return (velocity_.u_center(i, jp, k) - velocity_.u_center(i, jm, k)) / (2.0 * mesh_->dy);
            } else {
                if (j == mesh_->j_begin()) {
                    return (velocity_.u_center(i, j + 1, k) - velocity_.u_center(i, j, k)) / mesh_->dy;
                } else if (j == mesh_->j_end() - 1) {
                    return (velocity_.u_center(i, j, k) - velocity_.u_center(i, j - 1, k)) / mesh_->dy;
                } else {
                    return (velocity_.u_center(i, j + 1, k) - velocity_.u_center(i, j - 1, k)) / (2.0 * mesh_->dy);
                }
            }
        };

        // Vorticity components
        auto compute_omega_x = [&](int i, int j, int k) -> double {
            return compute_dwdy(i, j, k) - compute_dvdz(i, j, k);
        };

        auto compute_omega_y = [&](int i, int j, int k) -> double {
            return compute_dudz(i, j, k) - compute_dwdx(i, j, k);
        };

        auto compute_omega_z = [&](int i, int j, int k) -> double {
            return compute_dvdx(i, j, k) - compute_dudy(i, j, k);
        };

        // Vorticity vector field
        file << "VECTORS vorticity double\n";
        for (int k = mesh_->k_begin(); k < mesh_->k_end(); ++k) {
            for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                    file << compute_omega_x(i, j, k) << " "
                         << compute_omega_y(i, j, k) << " "
                         << compute_omega_z(i, j, k) << "\n";
                }
            }
        }

        // Vorticity magnitude
        file << "SCALARS vorticity_magnitude double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int k = mesh_->k_begin(); k < mesh_->k_end(); ++k) {
            for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                    double ox = compute_omega_x(i, j, k);
                    double oy = compute_omega_y(i, j, k);
                    double oz = compute_omega_z(i, j, k);
                    file << std::sqrt(ox*ox + oy*oy + oz*oz) << "\n";
                }
            }
        }

        // Q-criterion: Q = 0.5 * (||Omega||^2 - ||S||^2)
        // Positive Q indicates vortex cores (rotation dominates strain)
        // Requires diagonal derivatives (dudx, dvdy, dwdz) in addition to existing off-diagonals
        // NOTE: Uses uniform dx/dy/dz spacing (same assumption as vorticity above).
        // TODO: For large grids, consider precomputing the 3x3 gradient tensor per cell
        //       to avoid redundant lambda calls in the output loop.

        // Helper lambda for du/dx
        auto compute_dudx = [&](int i, int j, int k) -> double {
            if (periodic_x) {
                int im = (i == mesh_->i_begin()) ? mesh_->i_end() - 1 : i - 1;
                int ip = (i == mesh_->i_end() - 1) ? mesh_->i_begin() : i + 1;
                return (velocity_.u_center(ip, j, k) - velocity_.u_center(im, j, k)) / (2.0 * mesh_->dx);
            } else {
                if (i == mesh_->i_begin()) {
                    return (velocity_.u_center(i + 1, j, k) - velocity_.u_center(i, j, k)) / mesh_->dx;
                } else if (i == mesh_->i_end() - 1) {
                    return (velocity_.u_center(i, j, k) - velocity_.u_center(i - 1, j, k)) / mesh_->dx;
                } else {
                    return (velocity_.u_center(i + 1, j, k) - velocity_.u_center(i - 1, j, k)) / (2.0 * mesh_->dx);
                }
            }
        };

        // Helper lambda for dv/dy
        auto compute_dvdy = [&](int i, int j, int k) -> double {
            if (periodic_y) {
                int jm = (j == mesh_->j_begin()) ? mesh_->j_end() - 1 : j - 1;
                int jp = (j == mesh_->j_end() - 1) ? mesh_->j_begin() : j + 1;
                return (velocity_.v_center(i, jp, k) - velocity_.v_center(i, jm, k)) / (2.0 * mesh_->dy);
            } else {
                if (j == mesh_->j_begin()) {
                    return (velocity_.v_center(i, j + 1, k) - velocity_.v_center(i, j, k)) / mesh_->dy;
                } else if (j == mesh_->j_end() - 1) {
                    return (velocity_.v_center(i, j, k) - velocity_.v_center(i, j - 1, k)) / mesh_->dy;
                } else {
                    return (velocity_.v_center(i, j + 1, k) - velocity_.v_center(i, j - 1, k)) / (2.0 * mesh_->dy);
                }
            }
        };

        // Helper lambda for dw/dz
        auto compute_dwdz = [&](int i, int j, int k) -> double {
            if (periodic_z) {
                int km = (k == mesh_->k_begin()) ? mesh_->k_end() - 1 : k - 1;
                int kp = (k == mesh_->k_end() - 1) ? mesh_->k_begin() : k + 1;
                return (velocity_.w_center(i, j, kp) - velocity_.w_center(i, j, km)) / (2.0 * mesh_->dz);
            } else {
                if (k == mesh_->k_begin()) {
                    return (velocity_.w_center(i, j, k + 1) - velocity_.w_center(i, j, k)) / mesh_->dz;
                } else if (k == mesh_->k_end() - 1) {
                    return (velocity_.w_center(i, j, k) - velocity_.w_center(i, j, k - 1)) / mesh_->dz;
                } else {
                    return (velocity_.w_center(i, j, k + 1) - velocity_.w_center(i, j, k - 1)) / (2.0 * mesh_->dz);
                }
            }
        };

        file << "SCALARS Q_criterion double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (int k = mesh_->k_begin(); k < mesh_->k_end(); ++k) {
            for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                    // Velocity gradients (reuse existing lambdas + new diagonal ones)
                    double dudx = compute_dudx(i, j, k);
                    double dudy = compute_dudy(i, j, k);
                    double dudz = compute_dudz(i, j, k);
                    double dvdx = compute_dvdx(i, j, k);
                    double dvdy = compute_dvdy(i, j, k);
                    double dvdz = compute_dvdz(i, j, k);
                    double dwdx = compute_dwdx(i, j, k);
                    double dwdy = compute_dwdy(i, j, k);
                    double dwdz = compute_dwdz(i, j, k);

                    // Strain rate tensor components: S_ij = 0.5*(du_i/dx_j + du_j/dx_i)
                    double Sxx = dudx;
                    double Syy = dvdy;
                    double Szz = dwdz;
                    double Sxy = 0.5 * (dudy + dvdx);
                    double Sxz = 0.5 * (dudz + dwdx);
                    double Syz = 0.5 * (dvdz + dwdy);

                    // Rotation rate tensor components: Omega_ij = 0.5*(du_i/dx_j - du_j/dx_i)
                    double Oxy = 0.5 * (dudy - dvdx);
                    double Oxz = 0.5 * (dudz - dwdx);
                    double Oyz = 0.5 * (dvdz - dwdy);

                    // Squared Frobenius norms
                    double S_sq = Sxx*Sxx + Syy*Syy + Szz*Szz + 2.0*(Sxy*Sxy + Sxz*Sxz + Syz*Syz);
                    double O_sq = 2.0 * (Oxy*Oxy + Oxz*Oxz + Oyz*Oyz);

                    // Q-criterion
                    double Q = 0.5 * (O_sq - S_sq);
                    file << Q << "\n";
                }
            }
        }
        } // end degenerate mesh guard for 3D vorticity/Q-criterion
    }

    file.close();
}

// ============================================================================
// Shared pointer extraction (used by both CPU and GPU paths)
// ============================================================================

void RANSSolver::extract_field_pointers() {
    field_total_size_ = mesh_->total_cells();

    // Staggered grid velocity fields
    velocity_u_ptr_ = velocity_.u_data().data();
    velocity_v_ptr_ = velocity_.v_data().data();
    velocity_star_u_ptr_ = velocity_star_.u_data().data();
    velocity_star_v_ptr_ = velocity_star_.v_data().data();
    velocity_old_u_ptr_ = velocity_old_.u_data().data();
    velocity_old_v_ptr_ = velocity_old_.v_data().data();
    velocity_rk_u_ptr_ = velocity_rk_.u_data().data();
    velocity_rk_v_ptr_ = velocity_rk_.v_data().data();

    // Cell-centered fields
    pressure_ptr_ = pressure_.data().data();
    pressure_corr_ptr_ = pressure_correction_.data().data();
    nu_t_ptr_ = nu_t_.data().data();
    nu_eff_ptr_ = nu_eff_.data().data();
    rhs_poisson_ptr_ = rhs_poisson_.data().data();
    div_velocity_ptr_ = div_velocity_.data().data();

    // Work arrays
    conv_u_ptr_ = conv_.u_data().data();
    conv_v_ptr_ = conv_.v_data().data();
    diff_u_ptr_ = diff_.u_data().data();
    diff_v_ptr_ = diff_.v_data().data();

    // 3D w-velocity fields
    if (!mesh_->is2D()) {
        velocity_w_ptr_ = velocity_.w_data().data();
        velocity_star_w_ptr_ = velocity_star_.w_data().data();
        velocity_old_w_ptr_ = velocity_old_.w_data().data();
        velocity_rk_w_ptr_ = velocity_rk_.w_data().data();
        conv_w_ptr_ = conv_.w_data().data();
        diff_w_ptr_ = diff_.w_data().data();
    }

    // Turbulence transport fields
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
}

#ifndef NDEBUG
/// Verify that field data pointers haven't changed since GPU mapping.
/// Call this at the start of step() to catch accidental std::vector reallocation.
/// IMPORTANT: std::vector reallocation invalidates GPU mappings silently!
void RANSSolver::verify_mapping_integrity() const {
    // Verify velocity fields
    assert(velocity_u_ptr_ == velocity_.u_data().data() &&
           "velocity_.u_data() was reallocated after GPU mapping!");
    assert(velocity_v_ptr_ == velocity_.v_data().data() &&
           "velocity_.v_data() was reallocated after GPU mapping!");
    assert(velocity_star_u_ptr_ == velocity_star_.u_data().data() &&
           "velocity_star_.u_data() was reallocated after GPU mapping!");
    assert(velocity_star_v_ptr_ == velocity_star_.v_data().data() &&
           "velocity_star_.v_data() was reallocated after GPU mapping!");

    // Verify work arrays
    assert(conv_u_ptr_ == conv_.u_data().data() &&
           "conv_.u_data() was reallocated after GPU mapping!");
    assert(conv_v_ptr_ == conv_.v_data().data() &&
           "conv_.v_data() was reallocated after GPU mapping!");
    assert(diff_u_ptr_ == diff_.u_data().data() &&
           "diff_.u_data() was reallocated after GPU mapping!");
    assert(diff_v_ptr_ == diff_.v_data().data() &&
           "diff_.v_data() was reallocated after GPU mapping!");

    // Verify scalar fields
    assert(pressure_ptr_ == pressure_.data().data() &&
           "pressure_.data() was reallocated after GPU mapping!");
    assert(nu_eff_ptr_ == nu_eff_.data().data() &&
           "nu_eff_.data() was reallocated after GPU mapping!");
    assert(div_velocity_ptr_ == div_velocity_.data().data() &&
           "div_velocity_.data() was reallocated after GPU mapping!");

    // 3D fields
    if (!mesh_->is2D()) {
        assert(velocity_w_ptr_ == velocity_.w_data().data() &&
               "velocity_.w_data() was reallocated after GPU mapping!");
        assert(velocity_star_w_ptr_ == velocity_star_.w_data().data() &&
               "velocity_star_.w_data() was reallocated after GPU mapping!");
        assert(conv_w_ptr_ == conv_.w_data().data() &&
               "conv_.w_data() was reallocated after GPU mapping!");
        assert(diff_w_ptr_ == diff_.w_data().data() &&
               "diff_.w_data() was reallocated after GPU mapping!");
    }
}
#endif

#ifdef USE_GPU_OFFLOAD
void RANSSolver::initialize_gpu_buffers() {
    // Verify GPU is available (throws if not)
    gpu::verify_device_available();

    // Extract all raw pointers (shared with CPU path)
    extract_field_pointers();
    
#ifdef GPU_PROFILE_TRANSFERS
    auto transfer_start = std::chrono::steady_clock::now();
#endif
    
    // Fail fast if no GPU device available (GPU build requires GPU)
    gpu::verify_device_available();
    
    // Map all arrays to GPU device and copy initial values
    // Using map(to:) to transfer initialized data, map(alloc:) for device-only buffers
    // Data will persist on GPU for entire solver lifetime
    // Staggered grid: u and v have different sizes
    const size_t u_total_size = velocity_.u_total_size();
    const size_t v_total_size = velocity_.v_total_size();

    // Consolidated GPU buffer mapping - grouping arrays by size and transfer type
    // Group 1: u-component sized arrays (to: transfer initial data)
    #pragma omp target enter data \
        map(to: velocity_u_ptr_[0:u_total_size], \
                velocity_star_u_ptr_[0:u_total_size], \
                conv_u_ptr_[0:u_total_size], \
                diff_u_ptr_[0:u_total_size])

    // Group 2: v-component sized arrays (to: transfer initial data)
    #pragma omp target enter data \
        map(to: velocity_v_ptr_[0:v_total_size], \
                velocity_star_v_ptr_[0:v_total_size], \
                conv_v_ptr_[0:v_total_size], \
                diff_v_ptr_[0:v_total_size])

    // Group 3: field-sized arrays with initial data (to: transfer)
    #pragma omp target enter data \
        map(to: pressure_ptr_[0:field_total_size_], \
                pressure_corr_ptr_[0:field_total_size_], \
                nu_t_ptr_[0:field_total_size_], \
                nu_eff_ptr_[0:field_total_size_], \
                rhs_poisson_ptr_[0:field_total_size_], \
                div_velocity_ptr_[0:field_total_size_], \
                k_ptr_[0:field_total_size_], \
                omega_ptr_[0:field_total_size_])

    // Group 4: gradient buffers (to: need zero init to prevent NaN in EARSM)
    #pragma omp target enter data \
        map(to: dudx_ptr_[0:field_total_size_], \
                dudy_ptr_[0:field_total_size_], \
                dvdx_ptr_[0:field_total_size_], \
                dvdy_ptr_[0:field_total_size_], \
                wall_distance_ptr_[0:field_total_size_])

    // Group 5: device-only arrays (alloc: will be computed on GPU)
    // velocity_old: device-resident for residual computation (host never used)
    // velocity_rk: work buffer for RK time integration stages
    // tau_*: Reynolds stress components computed by EARSM/TBNN
    #pragma omp target enter data \
        map(alloc: velocity_old_u_ptr_[0:u_total_size], \
                   velocity_old_v_ptr_[0:v_total_size], \
                   velocity_rk_u_ptr_[0:u_total_size], \
                   velocity_rk_v_ptr_[0:v_total_size], \
                   tau_xx_ptr_[0:field_total_size_], \
                   tau_xy_ptr_[0:field_total_size_], \
                   tau_yy_ptr_[0:field_total_size_])

    // 3D w-velocity fields
    if (!mesh_->is2D()) {
        const size_t w_total_size = velocity_.w_total_size();
        #pragma omp target enter data \
            map(to: velocity_w_ptr_[0:w_total_size], \
                    velocity_star_w_ptr_[0:w_total_size], \
                    conv_w_ptr_[0:w_total_size], \
                    diff_w_ptr_[0:w_total_size]) \
            map(alloc: velocity_old_w_ptr_[0:w_total_size], \
                       velocity_rk_w_ptr_[0:w_total_size])
    }

    // Zero-initialize device-only arrays to prevent garbage in first residual computation
    // Arrays allocated with map(alloc:) contain garbage until explicitly written
    // NVHPC WORKAROUND: Use gpu::dev_ptr + is_device_ptr for correct device addresses
    {
        double* old_u_dev = gpu::dev_ptr(velocity_old_u_ptr_);
        double* old_v_dev = gpu::dev_ptr(velocity_old_v_ptr_);
        double* rk_u_dev = gpu::dev_ptr(velocity_rk_u_ptr_);
        double* rk_v_dev = gpu::dev_ptr(velocity_rk_v_ptr_);

        #pragma omp target teams distribute parallel for is_device_ptr(old_u_dev)
        for (size_t i = 0; i < u_total_size; ++i) old_u_dev[i] = 0.0;

        #pragma omp target teams distribute parallel for is_device_ptr(old_v_dev)
        for (size_t i = 0; i < v_total_size; ++i) old_v_dev[i] = 0.0;

        #pragma omp target teams distribute parallel for is_device_ptr(rk_u_dev)
        for (size_t i = 0; i < u_total_size; ++i) rk_u_dev[i] = 0.0;

        #pragma omp target teams distribute parallel for is_device_ptr(rk_v_dev)
        for (size_t i = 0; i < v_total_size; ++i) rk_v_dev[i] = 0.0;
    }

    if (!mesh_->is2D()) {
        const size_t w_total_size = velocity_.w_total_size();
        double* old_w_dev = gpu::dev_ptr(velocity_old_w_ptr_);
        double* rk_w_dev = gpu::dev_ptr(velocity_rk_w_ptr_);

        #pragma omp target teams distribute parallel for is_device_ptr(old_w_dev)
        for (size_t i = 0; i < w_total_size; ++i) old_w_dev[i] = 0.0;
        #pragma omp target teams distribute parallel for is_device_ptr(rk_w_dev)
        for (size_t i = 0; i < w_total_size; ++i) rk_w_dev[i] = 0.0;
    }

    // Zero-initialize Reynolds stress tensor components
    {
        double* tau_xx_dev = gpu::dev_ptr(tau_xx_ptr_);
        double* tau_xy_dev = gpu::dev_ptr(tau_xy_ptr_);
        double* tau_yy_dev = gpu::dev_ptr(tau_yy_ptr_);

        #pragma omp target teams distribute parallel for is_device_ptr(tau_xx_dev)
        for (size_t i = 0; i < field_total_size_; ++i) tau_xx_dev[i] = 0.0;

        #pragma omp target teams distribute parallel for is_device_ptr(tau_xy_dev)
        for (size_t i = 0; i < field_total_size_; ++i) tau_xy_dev[i] = 0.0;

        #pragma omp target teams distribute parallel for is_device_ptr(tau_yy_dev)
        for (size_t i = 0; i < field_total_size_; ++i) tau_yy_dev[i] = 0.0;
    }

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

    // Delete temporary/work arrays without copying back
    #pragma omp target exit data map(delete: velocity_star_u_ptr_[0:u_total_size])
    #pragma omp target exit data map(delete: velocity_star_v_ptr_[0:v_total_size])
    #pragma omp target exit data map(delete: velocity_old_u_ptr_[0:u_total_size])
    #pragma omp target exit data map(delete: velocity_old_v_ptr_[0:v_total_size])
    #pragma omp target exit data map(delete: velocity_rk_u_ptr_[0:u_total_size])
    #pragma omp target exit data map(delete: velocity_rk_v_ptr_[0:v_total_size])
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
        #pragma omp target exit data map(delete: velocity_rk_w_ptr_[0:w_total_size])
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

#ifndef NDEBUG
    // Debug: track sync calls to verify "no H↔D during stepping" guarantee
    gpu::increment_sync_counter();
#endif

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
#ifndef NDEBUG
    // Debug: track sync calls to verify "no H↔D during stepping" guarantee
    gpu::increment_sync_counter();
#endif

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

    // Get destination host pointers (current velocity field)
    double* u_host = velocity_.u_data().data();
    double* v_host = velocity_.v_data().data();

    // Get device pointers using omp_get_mapped_ptr
    // This returns the device address corresponding to the host address
    int device = omp_get_default_device();
    int host_dev = omp_get_initial_device();

    void* u_dev = omp_get_mapped_ptr(u_host, device);
    void* v_dev = omp_get_mapped_ptr(v_host, device);

    // FIX: Use omp_target_memcpy for reliable D2H sync (avoids NVHPC target update bugs)
    // This is the same approach used for w-velocity, now applied to u and v as well.
    if (u_dev) {
        omp_target_memcpy(u_host, u_dev,
                         u_total_size * sizeof(double),
                         0, 0, host_dev, device);
    } else {
        #pragma omp target update from(velocity_u_ptr_[0:u_total_size])
    }

    if (v_dev) {
        omp_target_memcpy(v_host, v_dev,
                         v_total_size * sizeof(double),
                         0, 0, host_dev, device);
    } else {
        #pragma omp target update from(velocity_v_ptr_[0:v_total_size])
    }

    // 3D w-velocity
    if (!mesh_->is2D()) {
        const size_t w_total_size = velocity_.w_total_size();
        double* w_host = velocity_.w_data().data();
        void* w_dev = omp_get_mapped_ptr(w_host, device);
        if (w_dev) {
            omp_target_memcpy(w_host, w_dev,
                             w_total_size * sizeof(double),
                             0, 0, host_dev, device);
        } else {
            #pragma omp target update from(velocity_w_ptr_[0:w_total_size])
        }
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

// ============================================================================
// Device-side QOI computation (avoids broken D2H sync in NVHPC)
// ============================================================================

double RANSSolver::compute_kinetic_energy_device() const {
    // Compute KE = 0.5 * integral(u^2 + v^2 + w^2) dV on device
    // For staggered grid: interpolate face velocities to cell centers
    //
    // NVHPC WORKAROUND: Use member pointers directly with nested target data + target.
    // Local pointer copies don't work because NVHPC doesn't translate them correctly.

    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Ng = mesh_->Nghost;
    const double dx = mesh_->dx;
    const double dy = mesh_->dy;
    const double dz = mesh_->is2D() ? 1.0 : mesh_->dz;
    const double dV = dx * dy * dz;

    const int u_stride = velocity_.u_stride();
    const int v_stride = velocity_.v_stride();
    const size_t u_total = velocity_.u_total_size();
    const size_t v_total = velocity_.v_total_size();

    double ke = 0.0;

    if (mesh_->is2D()) {
        const int n_cells = Nx * Ny;

        // NVHPC WORKAROUND: Use omp_get_mapped_ptr for actual device addresses
        int device = omp_get_default_device();
        const double* u_dev = static_cast<const double*>(omp_get_mapped_ptr(velocity_u_ptr_, device));
        const double* v_dev = static_cast<const double*>(omp_get_mapped_ptr(velocity_v_ptr_, device));

        #pragma omp target teams distribute parallel for reduction(+:ke) is_device_ptr(u_dev, v_dev) \
            firstprivate(Nx, Ny, Ng, u_stride, v_stride, dV)
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + Ng;
            int j = idx / Nx + Ng;

            // Interpolate u to cell center: average of left and right faces
            double u = 0.5 * (u_dev[j * u_stride + i] +
                              u_dev[j * u_stride + (i + 1)]);
            // Interpolate v to cell center: average of bottom and top faces
            double v = 0.5 * (v_dev[j * v_stride + i] +
                              v_dev[(j + 1) * v_stride + i]);

            ke += 0.5 * (u * u + v * v) * dV;
        }
    } else {
        const int u_plane = velocity_.u_plane_stride();
        const int v_plane = velocity_.v_plane_stride();
        const int w_stride = velocity_.w_stride();
        const int w_plane = velocity_.w_plane_stride();
        const int n_cells = Nx * Ny * Nz;

        // NVHPC WORKAROUND: Use omp_get_mapped_ptr for actual device addresses
        int device = omp_get_default_device();
        const double* u_dev = static_cast<const double*>(omp_get_mapped_ptr(velocity_u_ptr_, device));
        const double* v_dev = static_cast<const double*>(omp_get_mapped_ptr(velocity_v_ptr_, device));
        const double* w_dev = static_cast<const double*>(omp_get_mapped_ptr(velocity_w_ptr_, device));

        #pragma omp target teams distribute parallel for reduction(+:ke) is_device_ptr(u_dev, v_dev, w_dev) \
            firstprivate(Nx, Ny, Nz, Ng, u_stride, v_stride, w_stride, u_plane, v_plane, w_plane, dV)
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % Ny + Ng;
            int k = idx / (Nx * Ny) + Ng;

            double u = 0.5 * (u_dev[k * u_plane + j * u_stride + i] +
                              u_dev[k * u_plane + j * u_stride + (i + 1)]);
            double v = 0.5 * (v_dev[k * v_plane + j * v_stride + i] +
                              v_dev[k * v_plane + (j + 1) * v_stride + i]);
            double w = 0.5 * (w_dev[k * w_plane + j * w_stride + i] +
                              w_dev[(k + 1) * w_plane + j * w_stride + i]);

            ke += 0.5 * (u * u + v * v + w * w) * dV;
        }
    }

    return ke;
}

double RANSSolver::compute_max_velocity_device() const {
    // Compute max(|u|, |v|, |w|) on device
    // NVHPC WORKAROUND: Use gpu::dev_ptr + is_device_ptr for correct device addresses

    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Ng = mesh_->Nghost;

    const int u_stride = velocity_.u_stride();
    const int v_stride = velocity_.v_stride();

    double max_vel = 0.0;

    // Get device pointers
    const double* u_dev = gpu::dev_ptr(velocity_u_ptr_);
    const double* v_dev = gpu::dev_ptr(velocity_v_ptr_);

    if (mesh_->is2D()) {
        // Check u faces
        const int n_u = (Nx + 1) * Ny;
        #pragma omp target teams distribute parallel for is_device_ptr(u_dev) reduction(max:max_vel) \
            firstprivate(Nx, Ny, Ng, u_stride)
        for (int idx = 0; idx < n_u; ++idx) {
            int i = idx % (Nx + 1) + Ng;
            int j = idx / (Nx + 1) + Ng;
            double val = u_dev[j * u_stride + i];
            if (val < 0) val = -val;
            if (val > max_vel) max_vel = val;
        }

        // Check v faces
        const int n_v = Nx * (Ny + 1);
        #pragma omp target teams distribute parallel for is_device_ptr(v_dev) reduction(max:max_vel) \
            firstprivate(Nx, Ny, Ng, v_stride)
        for (int idx = 0; idx < n_v; ++idx) {
            int i = idx % Nx + Ng;
            int j = idx / Nx + Ng;
            double val = v_dev[j * v_stride + i];
            if (val < 0) val = -val;
            if (val > max_vel) max_vel = val;
        }
    } else {
        const int u_plane = velocity_.u_plane_stride();
        const int v_plane = velocity_.v_plane_stride();
        const int w_stride = velocity_.w_stride();
        const int w_plane = velocity_.w_plane_stride();
        const double* w_dev = gpu::dev_ptr(velocity_w_ptr_);

        // Check u faces
        const int n_u = (Nx + 1) * Ny * Nz;
        #pragma omp target teams distribute parallel for is_device_ptr(u_dev) reduction(max:max_vel) \
            firstprivate(Nx, Ny, Nz, Ng, u_stride, u_plane)
        for (int idx = 0; idx < n_u; ++idx) {
            int i = idx % (Nx + 1) + Ng;
            int j = (idx / (Nx + 1)) % Ny + Ng;
            int k = idx / ((Nx + 1) * Ny) + Ng;
            double val = u_dev[k * u_plane + j * u_stride + i];
            if (val < 0) val = -val;
            if (val > max_vel) max_vel = val;
        }

        // Check v faces
        const int n_v = Nx * (Ny + 1) * Nz;
        #pragma omp target teams distribute parallel for is_device_ptr(v_dev) reduction(max:max_vel) \
            firstprivate(Nx, Ny, Nz, Ng, v_stride, v_plane)
        for (int idx = 0; idx < n_v; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % (Ny + 1) + Ng;
            int k = idx / (Nx * (Ny + 1)) + Ng;
            double val = v_dev[k * v_plane + j * v_stride + i];
            if (val < 0) val = -val;
            if (val > max_vel) max_vel = val;
        }

        // Check w faces
        const int n_w = Nx * Ny * (Nz + 1);
        #pragma omp target teams distribute parallel for is_device_ptr(w_dev) reduction(max:max_vel) \
            firstprivate(Nx, Ny, Nz, Ng, w_stride, w_plane)
        for (int idx = 0; idx < n_w; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % Ny + Ng;
            int k = idx / (Nx * Ny) + Ng;
            double val = w_dev[k * w_plane + j * w_stride + i];
            if (val < 0) val = -val;
            if (val > max_vel) max_vel = val;
        }
    }

    return max_vel;
}

double RANSSolver::compute_divergence_linf_device() const {
    // Compute max|div(u)| on device
    // First compute divergence into div_velocity_, then find max
    // NVHPC WORKAROUND: Use gpu::dev_ptr + is_device_ptr for correct device addresses

    // Note: We need to compute divergence first. Use VelocityWhich::Current
    // This is const but we need to modify div_velocity_ - cast away const temporarily
    auto* self = const_cast<RANSSolver*>(this);
    self->compute_divergence(VelocityWhich::Current, self->div_velocity_);

    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Ng = mesh_->Nghost;
    const int stride = mesh_->total_Nx();
    const int plane_stride = stride * mesh_->total_Ny();

    double max_div = 0.0;

    // Get device pointer
    const double* div_dev = gpu::dev_ptr(div_velocity_ptr_);

    if (mesh_->is2D()) {
        const int n_cells = Nx * Ny;
        #pragma omp target teams distribute parallel for is_device_ptr(div_dev) reduction(max:max_div) \
            firstprivate(Nx, Ny, Ng, stride)
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + Ng;
            int j = idx / Nx + Ng;
            double val = div_dev[j * stride + i];
            if (val < 0) val = -val;
            if (val > max_div) max_div = val;
        }
    } else {
        const int n_cells = Nx * Ny * Nz;
        #pragma omp target teams distribute parallel for is_device_ptr(div_dev) reduction(max:max_div) \
            firstprivate(Nx, Ny, Nz, Ng, stride, plane_stride)
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % Ny + Ng;
            int k = idx / (Nx * Ny) + Ng;
            double val = div_dev[k * plane_stride + j * stride + i];
            if (val < 0) val = -val;
            if (val > max_div) max_div = val;
        }
    }

    return max_div;
}

double RANSSolver::compute_divergence_l2_device() const {
    // Compute L2 norm of divergence on device
    // NVHPC WORKAROUND: Use gpu::dev_ptr + is_device_ptr for correct device addresses

    auto* self = const_cast<RANSSolver*>(this);
    self->compute_divergence(VelocityWhich::Current, self->div_velocity_);

    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Nz = mesh_->Nz;
    const int Ng = mesh_->Nghost;
    const int stride = mesh_->total_Nx();
    const int plane_stride = stride * mesh_->total_Ny();
    const double dx = mesh_->dx;
    const double dy = mesh_->dy;
    const double dz = mesh_->is2D() ? 1.0 : mesh_->dz;
    const double dV = dx * dy * dz;

    double l2_sq = 0.0;

    // Get device pointer
    const double* div_dev = gpu::dev_ptr(div_velocity_ptr_);

    if (mesh_->is2D()) {
        const int n_cells = Nx * Ny;
        #pragma omp target teams distribute parallel for is_device_ptr(div_dev) reduction(+:l2_sq) \
            firstprivate(Nx, Ny, Ng, stride, dV)
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + Ng;
            int j = idx / Nx + Ng;
            double val = div_dev[j * stride + i];
            l2_sq += val * val * dV;
        }
    } else {
        const int n_cells = Nx * Ny * Nz;
        #pragma omp target teams distribute parallel for is_device_ptr(div_dev) reduction(+:l2_sq) \
            firstprivate(Nx, Ny, Nz, Ng, stride, plane_stride, dV)
        for (int idx = 0; idx < n_cells; ++idx) {
            int i = idx % Nx + Ng;
            int j = (idx / Nx) % Ny + Ng;
            int k = idx / (Nx * Ny) + Ng;
            double val = div_dev[k * plane_stride + j * stride + i];
            l2_sq += val * val * dV;
        }
    }

    return std::sqrt(l2_sq);
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
// CPU: Set raw pointers for unified code paths (no GPU mapping)
//
// This function enables the same loop code to work on both CPU and GPU builds.
// In GPU builds, these pointers are mapped to device memory with OpenMP target pragmas.
// In CPU builds, the loops simply use these raw pointers directly (no pragmas applied).
// This unification eliminates divergent CPU/GPU arithmetic and reduces code duplication.
void RANSSolver::initialize_gpu_buffers() {
    extract_field_pointers();
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
    // CPU build: return host pointers (same pattern as GPU version)
    TurbulenceDeviceView view;

    // Velocity field (staggered)
    view.u_face = velocity_u_ptr_;
    view.v_face = velocity_v_ptr_;
    view.u_stride = velocity_.u_stride();
    view.v_stride = velocity_.v_stride();

    // Turbulence fields (cell-centered)
    view.k = k_ptr_;
    view.omega = omega_ptr_;
    view.nu_t = nu_t_ptr_;
    view.cell_stride = mesh_->total_Nx();

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

