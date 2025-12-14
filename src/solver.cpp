#include "solver.hpp"
#include "timing.hpp"
#include "gpu_utils.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <cassert>

#ifdef GPU_PROFILE_TRANSFERS
#include <chrono>
#endif

#ifdef GPU_PROFILE_KERNELS
// Try to use NVTX if available, otherwise use lightweight markers
#if __has_include(<nvtx3/nvToolsExt.h>)
    #include <nvtx3/nvToolsExt.h>
    #define NVTX_PUSH(name) nvtxRangePushA(name)
    #define NVTX_POP() nvtxRangePop()
    #define NVTX_AVAILABLE 1
    
    // RAII scope guard for NVTX ranges (prevents unmatched push/pop on early returns)
    struct NvtxRangeScope {
        explicit NvtxRangeScope(const char* name) { nvtxRangePushA(name); }
        ~NvtxRangeScope() { nvtxRangePop(); }
        NvtxRangeScope(const NvtxRangeScope&) = delete;
        NvtxRangeScope& operator=(const NvtxRangeScope&) = delete;
    };
    #define NVTX_RANGE(name) NvtxRangeScope nvtx_scope_##__LINE__(name)
#else
    // Lightweight markers when NVTX is not available
    #define NVTX_PUSH(name) do { if (false) std::cout << "NVTX: " << name << std::endl; } while(0)
    #define NVTX_POP() do { } while(0)
    #define NVTX_AVAILABLE 0
    struct NvtxRangeScope {
        explicit NvtxRangeScope(const char*) {}
    };
    #define NVTX_RANGE(name) NvtxRangeScope nvtx_scope_##__LINE__(name)
#endif
#else
#define NVTX_PUSH(name)
#define NVTX_POP()
struct NvtxRangeScope {
    explicit NvtxRangeScope(const char*) {}
};
#define NVTX_RANGE(name) NvtxRangeScope nvtx_scope_##__LINE__(name)
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
    bool x_lo_periodic, bool x_hi_periodic,
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
    
    // Now set ghost cells for stencils
    if (x_lo_periodic) {
        int i_ghost = Ng - 1 - g;
        int i_periodic = Ng + Nx - 1 - g;
        u_ptr[j * u_stride + i_ghost] = u_ptr[j * u_stride + i_periodic];
    }
    
    if (x_hi_periodic) {
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

#ifdef USE_GPU_OFFLOAD
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
    velocity_bc_ = bc;
    
    // Update Poisson BCs based on velocity BCs
    PoissonBC p_x_lo = (bc.x_lo == VelocityBC::Periodic) ? PoissonBC::Periodic : PoissonBC::Neumann;
    PoissonBC p_x_hi = (bc.x_hi == VelocityBC::Periodic) ? PoissonBC::Periodic : PoissonBC::Neumann;
    PoissonBC p_y_lo = (bc.y_lo == VelocityBC::Periodic) ? PoissonBC::Periodic : PoissonBC::Neumann;
    PoissonBC p_y_hi = (bc.y_hi == VelocityBC::Periodic) ? PoissonBC::Periodic : PoissonBC::Neumann;
    
    // Store for GPU Poisson solver
    poisson_bc_x_lo_ = p_x_lo;
    poisson_bc_x_hi_ = p_x_hi;
    poisson_bc_y_lo_ = p_y_lo;
    poisson_bc_y_hi_ = p_y_hi;
    
    poisson_solver_.set_bc(p_x_lo, p_x_hi, p_y_lo, p_y_hi);
    mg_poisson_solver_.set_bc(p_x_lo, p_x_hi, p_y_lo, p_y_hi);
}

void RANSSolver::set_body_force(double fx, double fy) {
    fx_ = fx;
    fy_ = fy;
}

void RANSSolver::initialize(const VectorField& initial_velocity) {
    velocity_ = initial_velocity;
    apply_velocity_bc();
    
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
        double omega_init = k_init / (0.09 * config_.nu * 100.0);  // ν_t/ν ≈ 100 initially
        
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
    const int Nx = mesh_->Nx;
    const int Ny = mesh_->Ny;
    const int Ng = mesh_->Nghost;
    const int u_stride = velocity_.u_stride();
    const int v_stride = velocity_.v_stride();
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

#ifdef USE_GPU_OFFLOAD
    if (gpu::should_use_gpu_path()) {
        double* u_ptr = velocity_u_ptr_;
        double* v_ptr = velocity_v_ptr_;
        const size_t u_total_size = velocity_.u_total_size();
        const size_t v_total_size = velocity_.v_total_size();

        // Apply u BCs in x-direction
        const int n_u_x_bc = u_total_Ny * Ng;
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_total_size])
        for (int idx = 0; idx < n_u_x_bc; ++idx) {
            int j = idx / Ng;
            int g = idx % Ng;
            apply_u_bc_x_staggered(j, g, Nx, Ng, u_stride,
                                  x_lo_periodic, x_hi_periodic, u_ptr);
        }

        // Apply u BCs in y-direction
        const int n_u_y_bc = (Nx + 1 + 2 * Ng) * Ng;
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_total_size])
        for (int idx = 0; idx < n_u_y_bc; ++idx) {
            int i = idx / Ng;
            int g = idx % Ng;
            apply_u_bc_y_staggered(i, g, Ny, Ng, u_stride,
                                  y_lo_periodic, y_lo_noslip,
                                  y_hi_periodic, y_hi_noslip, u_ptr);
        }

        // Apply v BCs in x-direction
        const int n_v_x_bc = (Ny + 1 + 2 * Ng) * Ng;
        #pragma omp target teams distribute parallel for \
            map(present: v_ptr[0:v_total_size])
        for (int idx = 0; idx < n_v_x_bc; ++idx) {
            int j = idx / Ng;
            int g = idx % Ng;
            apply_v_bc_x_staggered(j, g, Nx, Ng, v_stride,
                                  x_lo_periodic, x_lo_noslip,
                                  x_hi_periodic, x_hi_noslip, v_ptr);
        }

        // Apply v BCs in y-direction
        const int n_v_y_bc = v_total_Nx * Ng;
        #pragma omp target teams distribute parallel for \
            map(present: v_ptr[0:v_total_size])
        for (int idx = 0; idx < n_v_y_bc; ++idx) {
            int i = idx / Ng;
            int g = idx % Ng;
            apply_v_bc_y_staggered(i, g, Ny, Ng, v_stride,
                                  y_lo_periodic, y_lo_noslip,
                                  y_hi_periodic, y_hi_noslip, v_ptr);
        }
        NVTX_POP();  // End apply_velocity_bc (GPU path)
        return;
    }
#endif

    // CPU path: staggered BCs on host pointers
    double* u_ptr = velocity_.u_data().data();
    double* v_ptr = velocity_.v_data().data();

    // Apply u BCs in x-direction
    for (int j = 0; j < u_total_Ny; ++j) {
        for (int g = 0; g < Ng; ++g) {
            apply_u_bc_x_staggered(j, g, Nx, Ng, u_stride,
                                  x_lo_periodic, x_hi_periodic, u_ptr);
        }
    }

    // Apply u BCs in y-direction
    for (int i = 0; i < Nx + 1 + 2 * Ng; ++i) {
        for (int g = 0; g < Ng; ++g) {
            apply_u_bc_y_staggered(i, g, Ny, Ng, u_stride,
                                  y_lo_periodic, y_lo_noslip,
                                  y_hi_periodic, y_hi_noslip, u_ptr);
        }
    }

    // Apply v BCs in x-direction
    for (int j = 0; j < Ny + 1 + 2 * Ng; ++j) {
        for (int g = 0; g < Ng; ++g) {
            apply_v_bc_x_staggered(j, g, Nx, Ng, v_stride,
                                  x_lo_periodic, x_lo_noslip,
                                  x_hi_periodic, x_hi_noslip, v_ptr);
        }
    }

    // Apply v BCs in y-direction  
    for (int i = 0; i < v_total_Nx; ++i) {
        for (int g = 0; g < Ng; ++g) {
            apply_v_bc_y_staggered(i, g, Ny, Ng, v_stride,
                                  y_lo_periodic, y_lo_noslip,
                                  y_hi_periodic, y_hi_noslip, v_ptr);
        }
    }
    
    // CORNER FIX: For fully periodic domains, apply x-direction BCs again
    // to ensure corner ghosts are correctly wrapped after y-direction BCs modified them
    if (x_lo_periodic && x_hi_periodic) {
        for (int j = 0; j < u_total_Ny; ++j) {
            for (int g = 0; g < Ng; ++g) {
                apply_u_bc_x_staggered(j, g, Nx, Ng, u_stride,
                                      x_lo_periodic, x_hi_periodic, u_ptr);
            }
        }
    }
    
    if (y_lo_periodic && y_hi_periodic) {
        for (int i = 0; i < v_total_Nx; ++i) {
            for (int g = 0; g < Ng; ++g) {
                apply_v_bc_y_staggered(i, g, Ny, Ng, v_stride,
                                      y_lo_periodic, y_lo_noslip,
                                      y_hi_periodic, y_hi_noslip, v_ptr);
            }
        }
    }
    NVTX_POP();  // End apply_velocity_bc
}

void RANSSolver::compute_convective_term(const VectorField& vel, VectorField& conv) {
    const double dx = mesh_->dx;
    const double dy = mesh_->dy;
    [[maybe_unused]] const int Nx = mesh_->Nx;
    [[maybe_unused]] const int Ny = mesh_->Ny;
    const int Ng = mesh_->Nghost;
    const int u_stride = vel.u_stride();
    const int v_stride = vel.v_stride();
    const bool use_central = (config_.convective_scheme == ConvectiveScheme::Central);

#ifdef USE_GPU_OFFLOAD
    // GPU path: staggered convection on GPU
    if (gpu::should_use_gpu_path()) {
        const size_t u_total_size = vel.u_total_size();
        const size_t v_total_size = vel.v_total_size();

        const double* u_ptr      = velocity_u_ptr_;
        const double* v_ptr      = velocity_v_ptr_;
        double*       conv_u_ptr = conv_u_ptr_;
        double*       conv_v_ptr = conv_v_ptr_;

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
        return;
    }
#endif

    // CPU path: staggered convection on CPU
    const double* u_ptr      = vel.u_data().data();
    const double* v_ptr      = vel.v_data().data();
    double*       conv_u_ptr = conv.u_data().data();
    double*       conv_v_ptr = conv.v_data().data();

    // Compute u-momentum convection at x-faces: i in [Ng, Ng+Nx], j in [Ng, Ng+Ny-1]
    for (int j = Ng; j < Ng + Ny; ++j) {
        for (int i = Ng; i <= Ng + Nx; ++i) {
            convective_u_face_kernel_staggered(i, j, u_stride, v_stride, u_stride, dx, dy, use_central,
                                              u_ptr, v_ptr, conv_u_ptr);
        }
    }

    // Compute v-momentum convection at y-faces: i in [Ng, Ng+Nx-1], j in [Ng, Ng+Ny]
    for (int j = Ng; j <= Ng + Ny; ++j) {
        for (int i = Ng; i < Ng + Nx; ++i) {
            convective_v_face_kernel_staggered(i, j, u_stride, v_stride, v_stride, dx, dy, use_central,
                                              u_ptr, v_ptr, conv_v_ptr);
        }
    }
}

void RANSSolver::compute_diffusive_term(const VectorField& vel, const ScalarField& nu_eff, 
                                        VectorField& diff) {
    const double dx = mesh_->dx;
    const double dy = mesh_->dy;
    [[maybe_unused]] const int Nx = mesh_->Nx;
    [[maybe_unused]] const int Ny = mesh_->Ny;
    const int Ng = mesh_->Nghost;
    const int u_stride = vel.u_stride();
    const int v_stride = vel.v_stride();
    const int nu_stride = mesh_->total_Nx();

#ifdef USE_GPU_OFFLOAD
    // GPU path: staggered diffusion on GPU
    if (gpu::should_use_gpu_path()) {
        const size_t u_total_size = vel.u_total_size();
        const size_t v_total_size = vel.v_total_size();
        const size_t nu_total_size = field_total_size_;

        const double* u_ptr      = velocity_u_ptr_;
        const double* v_ptr      = velocity_v_ptr_;
        const double* nu_ptr     = nu_eff_ptr_;
        double*       diff_u_ptr = diff_u_ptr_;
        double*       diff_v_ptr = diff_v_ptr_;

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
        return;
    }
#endif

    // CPU path: staggered diffusion on CPU
    const double* u_ptr      = vel.u_data().data();
    const double* v_ptr      = vel.v_data().data();
    const double* nu_ptr     = nu_eff.data().data();
    double*       diff_u_ptr = diff.u_data().data();
    double*       diff_v_ptr = diff.v_data().data();

    // Compute u-momentum diffusion at x-faces: i in [Ng, Ng+Nx], j in [Ng, Ng+Ny-1]
    for (int j = Ng; j < Ng + Ny; ++j) {
        for (int i = Ng; i <= Ng + Nx; ++i) {
            diffusive_u_face_kernel_staggered(i, j, u_stride, nu_stride, u_stride, dx, dy,
                                             u_ptr, nu_ptr, diff_u_ptr);
        }
    }

    // Compute v-momentum diffusion at y-faces: i in [Ng, Ng+Nx-1], j in [Ng, Ng+Ny]
    for (int j = Ng; j <= Ng + Ny; ++j) {
        for (int i = Ng; i < Ng + Nx; ++i) {
            diffusive_v_face_kernel_staggered(i, j, v_stride, nu_stride, v_stride, dx, dy,
                                             v_ptr, nu_ptr, diff_v_ptr);
        }
    }
}

void RANSSolver::compute_divergence(const VectorField& vel, ScalarField& div) {
    const double dx = mesh_->dx;
    const double dy = mesh_->dy;
    [[maybe_unused]] const int Nx = mesh_->Nx;
    [[maybe_unused]] const int Ny = mesh_->Ny;
    const int div_stride = mesh_->total_Nx();
    const int u_stride = vel.u_stride();
    const int v_stride = vel.v_stride();

#ifdef USE_GPU_OFFLOAD
    // GPU path: staggered divergence on GPU
    if (gpu::should_use_gpu_path()) {
        const int n_cells = Nx * Ny;
        const size_t u_total_size = vel.u_total_size();
        const size_t v_total_size = vel.v_total_size();
        const size_t div_total_size = field_total_size_;

        // Use SolverDeviceView to centralize pointer selection
        auto view = get_solver_view();
        
        // Determine correct GPU pointers based on which VectorField was passed
        // Compare data pointers to identify which field this is
        const double* u_ptr = nullptr;
        const double* v_ptr = nullptr;
        if (vel.u_data().data() == velocity_.u_data().data()) {
            u_ptr = view.u_face;
            v_ptr = view.v_face;
        } else if (vel.u_data().data() == velocity_star_.u_data().data()) {
            u_ptr = view.u_star_face;
            v_ptr = view.v_star_face;
        } else {
            // Unknown field - fall back to CPU
            goto cpu_divergence_fallback;
        }
        
        // Output divergence (always div_velocity_)
        double* div_ptr = view.div;

        // Use target data for scalar parameters (NVHPC workaround)
        #pragma omp target data map(to: dx, dy, u_stride, v_stride, div_stride, Nx)
        {
            #pragma omp target teams distribute parallel for \
                map(present: u_ptr[0:u_total_size], v_ptr[0:v_total_size], div_ptr[0:div_total_size])
            for (int idx = 0; idx < n_cells; ++idx) {
                const int i = idx % Nx + 1;  // Cell center i index (with ghosts)
                const int j = idx / Nx + 1;  // Cell center j index (with ghosts)

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
        return;
    }
    
cpu_divergence_fallback:
#endif
    {
        // CPU path: staggered divergence on CPU
        const double* u_ptr  = vel.u_data().data();
        const double* v_ptr  = vel.v_data().data();
        double*       div_ptr = div.data().data();

        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                divergence_cell_kernel_staggered(i, j, u_stride, v_stride, div_stride, dx, dy,
                                                u_ptr, v_ptr, div_ptr);
            }
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
    const double dx = mesh_->dx;
    const double dy = mesh_->dy;
    const double dt = current_dt_;
    [[maybe_unused]] const int Nx = mesh_->Nx;
    [[maybe_unused]] const int Ny = mesh_->Ny;
    const int p_stride = mesh_->total_Nx();
    const int u_stride = velocity_.u_stride();
    const int v_stride = velocity_.v_stride();
    const int Ng = mesh_->Nghost;
    
    const bool x_periodic = (velocity_bc_.x_lo == VelocityBC::Periodic) && 
                            (velocity_bc_.x_hi == VelocityBC::Periodic);
    const bool y_periodic = (velocity_bc_.y_lo == VelocityBC::Periodic) && 
                            (velocity_bc_.y_hi == VelocityBC::Periodic);

#ifdef USE_GPU_OFFLOAD
    // GPU path: staggered velocity correction on GPU
    if (gpu::should_use_gpu_path()) {
        const int n_cells = Nx * Ny;
        const size_t u_total_size = velocity_.u_total_size();
        const size_t v_total_size = velocity_.v_total_size();
        const size_t p_total_size = field_total_size_;

        const double* u_star_ptr = velocity_star_u_ptr_;
        const double* v_star_ptr = velocity_star_v_ptr_;
        const double* p_corr_ptr = pressure_corr_ptr_;
        double*       u_ptr      = velocity_u_ptr_;
        double*       v_ptr      = velocity_v_ptr_;
        double*       p_ptr      = pressure_ptr_;

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
    } else {
#endif
        // CPU path: staggered velocity correction on CPU
        const double* u_star_ptr = velocity_star_.u_data().data();
        const double* v_star_ptr = velocity_star_.v_data().data();
        const double* p_corr_ptr = pressure_correction_.data().data();
        double*       u_ptr      = velocity_.u_data().data();
        double*       v_ptr      = velocity_.v_data().data();
        double*       p_ptr      = pressure_.data().data();

        // Correct ALL u-velocities at x-faces (including redundant if periodic)
        for (int j = Ng; j < Ng + Ny; ++j) {
            for (int i = Ng; i <= Ng + Nx; ++i) {
                correct_u_face_kernel_staggered(i, j, u_stride, p_stride, dx, dt,
                                               u_star_ptr, p_corr_ptr, u_ptr);
            }
        }
        
        // Enforce exact x-periodicity: average left and right edges
        if (x_periodic) {
            for (int j = Ng; j < Ng + Ny; ++j) {
                double u_avg = 0.5 * (u_ptr[j * u_stride + Ng] + u_ptr[j * u_stride + (Ng + Nx)]);
                u_ptr[j * u_stride + Ng] = u_avg;
                u_ptr[j * u_stride + (Ng + Nx)] = u_avg;
            }
        }

        // Correct ALL v-velocities at y-faces (including redundant if periodic)
        for (int j = Ng; j <= Ng + Ny; ++j) {
            for (int i = Ng; i < Ng + Nx; ++i) {
                correct_v_face_kernel_staggered(i, j, v_stride, p_stride, dy, dt,
                                               v_star_ptr, p_corr_ptr, v_ptr);
            }
        }
        
        // Enforce exact y-periodicity: average bottom and top edges
        if (y_periodic) {
            for (int i = Ng; i < Ng + Nx; ++i) {
                double v_avg = 0.5 * (v_ptr[Ng * v_stride + i] + v_ptr[(Ng + Ny) * v_stride + i]);
                v_ptr[Ng * v_stride + i] = v_avg;
                v_ptr[(Ng + Ny) * v_stride + i] = v_avg;
            }
        }

        // Update pressure at cell centers
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                update_pressure_kernel(i, j, p_stride, p_corr_ptr, p_ptr);
            }
        }
#ifdef USE_GPU_OFFLOAD
    }
#endif
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
    
    // Copy u-velocity device-to-device
    #pragma omp target teams distribute parallel for collapse(2) \
        map(present: velocity_u_ptr_[0:u_total_size], velocity_old_u_ptr_[0:u_total_size])
    for (int j = Ng; j < Ng + Ny; ++j) {
        for (int i = Ng; i <= Ng + Nx; ++i) {
            const int idx = j * u_stride + i;
            velocity_old_u_ptr_[idx] = velocity_u_ptr_[idx];
        }
    }
    
    // Copy v-velocity device-to-device
    #pragma omp target teams distribute parallel for collapse(2) \
        map(present: velocity_v_ptr_[0:v_total_size], velocity_old_v_ptr_[0:v_total_size])
    for (int j = Ng; j <= Ng + Ny; ++j) {
        for (int i = Ng; i < Ng + Nx; ++i) {
            const int idx = j * v_stride + i;
            velocity_old_v_ptr_[idx] = velocity_v_ptr_[idx];
        }
    }
    NVTX_POP();
    }
#else
    // CPU path: use host-side velocity_old_
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
#endif
        
        turb_model_->update(*mesh_, velocity_, k_, omega_, nu_t_, 
                           turb_model_->provides_reynolds_stresses() ? &tau_ij_ : nullptr,
                           device_view_ptr);
        NVTX_POP();
        
        // Turbulence models now manage their own GPU buffers and copy results back to CPU nu_t
        // Sync nu_t to GPU for use in nu_eff computation
#ifdef USE_GPU_OFFLOAD
        #pragma omp target update to(nu_t_ptr_[0:field_total_size_])
#endif
    }
    
    // Effective viscosity: nu_eff_ = nu + nu_t (use persistent field)
    // GPU path: compute directly on GPU without CPU fill
#ifdef USE_GPU_OFFLOAD
    if (mesh_->Nx >= 32 && mesh_->Ny >= 32) {
        NVTX_PUSH("nu_eff_computation");
        const int Nx = mesh_->Nx;
        const int Ny = mesh_->Ny;
        const int n_cells = Nx * Ny;
        const int stride = Nx + 2;
        const size_t total_size = field_total_size_;
        const double nu = config_.nu;
        double* nu_eff_ptr = nu_eff_ptr_;
        const double* nu_t_ptr = nu_t_ptr_;
        
        if (turb_model_) {
            // With turbulence: nu_eff = nu + nu_t
            #pragma omp target teams distribute parallel for \
                map(present: nu_eff_ptr[0:total_size]) \
                map(present: nu_t_ptr[0:total_size]) \
                firstprivate(nu, stride, Nx)
            for (int idx = 0; idx < n_cells; ++idx) {
                int i = idx % Nx + 1;  // +1 for ghost cells
                int j = idx / Nx + 1;
                int cell_idx = j * stride + i;
                nu_eff_ptr[cell_idx] = nu + nu_t_ptr[cell_idx];
            }
        } else {
            // No turbulence: nu_eff = nu (constant)
            #pragma omp target teams distribute parallel for \
                map(present: nu_eff_ptr[0:total_size]) \
                firstprivate(nu, stride, Nx)
            for (int idx = 0; idx < n_cells; ++idx) {
                int i = idx % Nx + 1;  // +1 for ghost cells
                int j = idx / Nx + 1;
                int cell_idx = j * stride + i;
                nu_eff_ptr[cell_idx] = nu;
            }
        }
        NVTX_POP();
    } else
#endif
    {
        // CPU path
        nu_eff_.fill(config_.nu);
        if (turb_model_) {
            for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
                for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                    nu_eff_(i, j) = config_.nu + nu_t_(i, j);
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
    [[maybe_unused]] const int u_stride = velocity_.u_stride();
    [[maybe_unused]] const int v_stride = velocity_.v_stride();
    
    const bool x_periodic = (velocity_bc_.x_lo == VelocityBC::Periodic) && 
                            (velocity_bc_.x_hi == VelocityBC::Periodic);
    const bool y_periodic = (velocity_bc_.y_lo == VelocityBC::Periodic) && 
                            (velocity_bc_.y_hi == VelocityBC::Periodic);
    
#ifdef USE_GPU_OFFLOAD
    if (gpu::should_use_gpu_path()) {
        const size_t u_total_size = velocity_.u_total_size();
        const size_t v_total_size = velocity_.v_total_size();
        const double dt = current_dt_;
        const double fx = fx_;
        const double fy = fy_;
        const double* u_ptr = velocity_u_ptr_;
        const double* v_ptr = velocity_v_ptr_;
        double* u_star_ptr = velocity_star_u_ptr_;
        double* v_star_ptr = velocity_star_v_ptr_;
        const double* conv_u_ptr = conv_u_ptr_;
        const double* conv_v_ptr = conv_v_ptr_;
        const double* diff_u_ptr = diff_u_ptr_;
        const double* diff_v_ptr = diff_v_ptr_;
        
        // Compute u* at ALL x-faces (including redundant if periodic)
        const int n_u_faces = (Nx + 1) * Ny;
        #pragma omp target teams distribute parallel for \
            map(present: u_ptr[0:u_total_size], u_star_ptr[0:u_total_size], \
                        conv_u_ptr[0:u_total_size], diff_u_ptr[0:u_total_size]) \
            firstprivate(dt, fx, u_stride, Nx, Ng)
        for (int idx = 0; idx < n_u_faces; ++idx) {
            int i_local = idx % (Nx + 1);
            int j_local = idx / (Nx + 1);
            int i = i_local + Ng;
            int j = j_local + Ng;
            int u_idx = j * u_stride + i;
            
            u_star_ptr[u_idx] = u_ptr[u_idx] + dt * (-conv_u_ptr[u_idx] + diff_u_ptr[u_idx] + fx);
        }
        
        // Enforce exact x-periodicity for u*: average left and right edges
        if (x_periodic) {
            const int n_u_periodic = Ny;
            #pragma omp target teams distribute parallel for \
                map(present: u_star_ptr[0:u_total_size]) \
                firstprivate(u_stride, Nx, Ng)
            for (int j_local = 0; j_local < n_u_periodic; ++j_local) {
                int j = j_local + Ng;
                double u_avg = 0.5 * (u_star_ptr[j * u_stride + Ng] + u_star_ptr[j * u_stride + (Ng + Nx)]);
                u_star_ptr[j * u_stride + Ng] = u_avg;
                u_star_ptr[j * u_stride + (Ng + Nx)] = u_avg;
            }
        }
        
        // Compute v* at ALL y-faces (including redundant if periodic)
        const int n_v_faces = Nx * (Ny + 1);
        #pragma omp target teams distribute parallel for \
            map(present: v_ptr[0:v_total_size], v_star_ptr[0:v_total_size], \
                        conv_v_ptr[0:v_total_size], diff_v_ptr[0:v_total_size]) \
            firstprivate(dt, fy, v_stride, Nx, Ng)
        for (int idx = 0; idx < n_v_faces; ++idx) {
            int i_local = idx % Nx;
            int j_local = idx / Nx;
            int i = i_local + Ng;
            int j = j_local + Ng;
            int v_idx = j * v_stride + i;
            
            v_star_ptr[v_idx] = v_ptr[v_idx] + dt * (-conv_v_ptr[v_idx] + diff_v_ptr[v_idx] + fy);
        }
        
        // Enforce exact y-periodicity for v*: average bottom and top edges
        if (y_periodic) {
            const int n_v_periodic = Nx;
            #pragma omp target teams distribute parallel for \
                map(present: v_star_ptr[0:v_total_size]) \
                firstprivate(v_stride, Ny, Ng)
            for (int i_local = 0; i_local < n_v_periodic; ++i_local) {
                int i = i_local + Ng;
                double v_avg = 0.5 * (v_star_ptr[Ng * v_stride + i] + v_star_ptr[(Ng + Ny) * v_stride + i]);
                v_star_ptr[Ng * v_stride + i] = v_avg;
                v_star_ptr[(Ng + Ny) * v_stride + i] = v_avg;
            }
        }
    } else
#endif
    {
        // CPU path: compute u* at ALL x-faces (including redundant if periodic)
        for (int j = Ng; j < Ng + Ny; ++j) {
            for (int i = Ng; i <= Ng + Nx; ++i) {
                velocity_star_.u(i, j) = velocity_.u(i, j) + current_dt_ * 
                    (-conv_.u(i, j) + diff_.u(i, j) + fx_);
            }
        }
        
        // Enforce exact x-periodicity for u*: average left and right edges
        if (x_periodic) {
            for (int j = Ng; j < Ng + Ny; ++j) {
                double u_avg = 0.5 * (velocity_star_.u(Ng, j) + velocity_star_.u(Ng + Nx, j));
                velocity_star_.u(Ng, j) = u_avg;
                velocity_star_.u(Ng + Nx, j) = u_avg;
            }
        }
        
        // CPU path: compute v* at ALL y-faces (including redundant if periodic)
        for (int j = Ng; j <= Ng + Ny; ++j) {
            for (int i = Ng; i < Ng + Nx; ++i) {
                velocity_star_.v(i, j) = velocity_.v(i, j) + current_dt_ * 
                    (-conv_.v(i, j) + diff_.v(i, j) + fy_);
            }
        }
        
        // Enforce exact y-periodicity for v*: average bottom and top edges
        if (y_periodic) {
            for (int i = Ng; i < Ng + Nx; ++i) {
                double v_avg = 0.5 * (velocity_star_.v(i, Ng) + velocity_star_.v(i, Ng + Ny));
                velocity_star_.v(i, Ng) = v_avg;
                velocity_star_.v(i, Ng + Ny) = v_avg;
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
#endif
    
    // PHASE 1.5 OPTIMIZATION: Skip redundant BC call for fully periodic domains
    // The inline periodic averaging above already handles periodic BCs correctly
    // Only apply BCs if domain has non-periodic boundaries (which need ghost cell updates)
    const bool needs_bc_update = !x_periodic || !y_periodic;
    if (needs_bc_update) {
        apply_velocity_bc();
    }
    
    std::swap(velocity_, velocity_star_);
#ifdef USE_GPU_OFFLOAD
    // Swap pointers back to restore original mapping
    std::swap(velocity_u_ptr_, velocity_star_u_ptr_);
    std::swap(velocity_v_ptr_, velocity_star_v_ptr_);
#endif
    NVTX_POP();  // End predictor_step
    
    // 4. Solve pressure Poisson equation
    // nabla^2p' = (1/dt) nabla*u*
    {
        TIMED_SCOPE("divergence");
        NVTX_PUSH("divergence");
        compute_divergence(velocity_star_, div_velocity_);
        NVTX_POP();
    }
    
    // Build RHS on GPU and subtract mean divergence to ensure solvability
    // GPU-RESIDENT OPTIMIZATION: Keep all data on device, only transfer scalars
    double mean_div = 0.0;
    
#ifdef USE_GPU_OFFLOAD
    if (mesh_->Nx >= 32 && mesh_->Ny >= 32) {
        // GPU-resident path: compute mean divergence on device via reduction
        const int Nx = mesh_->Nx;
        const int Ny = mesh_->Ny;
        const int i_begin = mesh_->i_begin();
        const int j_begin = mesh_->j_begin();
        const int Nxg = mesh_->Nx + 2;  // Total grid width with ghost cells
        
        double sum_div = 0.0;
        int count = Nx * Ny;
        
        // Compute sum of divergence on GPU (parallel reduction)
        // Note: Variables captured implicitly (NVHPC has issues with firstprivate)
        #pragma omp target teams distribute parallel for collapse(2) \
            map(present: div_velocity_ptr_[0:field_total_size_]) \
            reduction(+:sum_div)
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                int ii = i + i_begin;
                int jj = j + j_begin;
                int idx = ii + jj * Nxg;
                sum_div += div_velocity_ptr_[idx];
            }
        }
        
        mean_div = (count > 0) ? sum_div / count : 0.0;
        
        // Build RHS on GPU: rhs = (div - mean_div) / dt
        const double dt_inv = 1.0 / current_dt_;
        
        // Note: Variables captured implicitly (NVHPC has issues with firstprivate)
        #pragma omp target teams distribute parallel for collapse(2) \
            map(present: div_velocity_ptr_[0:field_total_size_], rhs_poisson_ptr_[0:field_total_size_])
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                int ii = i + i_begin;
                int jj = j + j_begin;
                int idx = ii + jj * Nxg;
                rhs_poisson_ptr_[idx] = (div_velocity_ptr_[idx] - mean_div) * dt_inv;
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
        // CPU fallback path (unchanged)
        double sum_div = 0.0;
        int count = 0;
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                double div = div_velocity_(i, j);
                sum_div += div;
                ++count;
            }
        }
        mean_div = (count > 0) ? sum_div / count : 0.0;
        
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                rhs_poisson_(i, j) = (div_velocity_(i, j) - mean_div) / current_dt_;
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
        PoissonConfig pcfg;
        pcfg.tol = config_.poisson_tol;
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
#ifdef USE_GPU_OFFLOAD
        if (mesh_->Nx >= 32 && mesh_->Ny >= 32 && use_multigrid_) {
            // GPU-RESIDENT PATH: solve directly on device without host staging
            // This eliminates the DtoH/HtoD transfers that happened in the old path
            cycles = mg_poisson_solver_.solve_device(rhs_poisson_ptr_, pressure_corr_ptr_, pcfg);
        } else
#endif
        {
            // CPU fallback path (or GPU with host staging for non-multigrid)
            if (use_multigrid_) {
                cycles = mg_poisson_solver_.solve(rhs_poisson_, pressure_correction_, pcfg);
            } else {
                cycles = poisson_solver_.solve(rhs_poisson_, pressure_correction_, pcfg);
            }
        }
        
        // Print cycle count diagnostics if enabled
        if (poisson_diagnostics && (iter_ % poisson_diagnostics_interval == 0)) {
            std::cout << "[Poisson] iter=" << iter_ << " cycles=" << cycles 
                      << " residual=" << std::scientific << std::setprecision(15) 
                      << mg_poisson_solver_.residual() << "\n";
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
    
    // Return max velocity change as convergence criterion
    double max_change = 0.0;
    
#ifdef USE_GPU_OFFLOAD
    if (gpu::should_use_gpu_path()) {
        // GPU-resident residual: compute max change on GPU via reduction
        // velocity_old is now device-resident - NO H→D upload needed!
        const size_t u_total_size = velocity_.u_total_size();
        const size_t v_total_size = velocity_.v_total_size();
        const double* u_new_ptr = velocity_u_ptr_;
        const double* v_new_ptr = velocity_v_ptr_;
        const double* u_old_ptr = velocity_old_u_ptr_;
        const double* v_old_ptr = velocity_old_v_ptr_;
        const int u_stride = Nx + 2 * Ng + 1;
        const int v_stride = Nx + 2 * Ng;
        
        // Compute max |u_new - u_old| on GPU (both arrays already on device!)
        const int n_u_faces = (Nx + 1) * Ny;
        double max_du = 0.0;
        #pragma omp target teams distribute parallel for reduction(max:max_du) \
            map(present: u_new_ptr[0:u_total_size], u_old_ptr[0:u_total_size]) \
            map(to: Ng, u_stride, Nx)
        for (int idx = 0; idx < n_u_faces; ++idx) {
            int i_local = idx % (Nx + 1);
            int j_local = idx / (Nx + 1);
            int i = i_local + Ng;
            int j = j_local + Ng;
            int u_idx = j * u_stride + i;
            double du = u_new_ptr[u_idx] - u_old_ptr[u_idx];
            if (du < 0.0) du = -du;
            if (du > max_du) max_du = du;
        }
        
        // Compute max |v_new - v_old| on GPU (both arrays already on device!)
        const int n_v_faces = Nx * (Ny + 1);
        double max_dv = 0.0;
        #pragma omp target teams distribute parallel for reduction(max:max_dv) \
            map(present: v_new_ptr[0:v_total_size], v_old_ptr[0:v_total_size]) \
            map(to: Ng, v_stride, Nx)
        for (int idx = 0; idx < n_v_faces; ++idx) {
            int i_local = idx % Nx;
            int j_local = idx / Nx;
            int i = i_local + Ng;
            int j = j_local + Ng;
            int v_idx = j * v_stride + i;
            double dv = v_new_ptr[v_idx] - v_old_ptr[v_idx];
            if (dv < 0.0) dv = -dv;
            if (dv > max_dv) max_dv = dv;
        }
        
        max_change = (max_du > max_dv) ? max_du : max_dv;
    } else
#endif
    {
        // CPU path: compute residual on CPU using velocity_old_
        // Check u-velocity change at x-faces
        for (int j = Ng; j < Ng + Ny; ++j) {
            for (int i = Ng; i <= Ng + Nx; ++i) {
                double du = std::abs(velocity_.u(i, j) - velocity_old_.u(i, j));
                max_change = std::max(max_change, du);
            }
        }
        
        // Check v-velocity change at y-faces
        for (int j = Ng; j <= Ng + Ny; ++j) {
            for (int i = Ng; i < Ng + Nx; ++i) {
                double dv = std::abs(velocity_.v(i, j) - velocity_old_.v(i, j));
                max_change = std::max(max_change, dv);
            }
        }
    }

    return max_change;
}

std::pair<double, int> RANSSolver::solve_steady() {
    double residual = 1.0;
    
    if (config_.verbose) {
        if (config_.adaptive_dt) {
            std::cout << std::setw(8) << "Iter" 
                      << std::setw(15) << "Residual"
                      << std::setw(15) << "Max |u|"
                      << std::setw(12) << "dt"
                      << "\n";
        } else {
            std::cout << std::setw(8) << "Iter" 
                      << std::setw(15) << "Residual"
                      << std::setw(15) << "Max |u|"
                      << "\n";
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
                          << "\n";
            } else {
                std::cout << std::setw(8) << iter_ + 1
                          << std::setw(15) << std::scientific << std::setprecision(3) << residual
                          << std::setw(15) << std::fixed << max_vel
                          << "\n";
            }
        }
        
        if (residual < config_.tol) {
            if (config_.verbose) {
                std::cout << "Converged at iteration " << iter_ + 1 
                          << " with residual " << residual << "\n";
            }
            break;
        }
        
        // Check for divergence
        if (std::isnan(residual) || std::isinf(residual)) {
            if (config_.verbose) {
                std::cerr << "Solver diverged at iteration " << iter_ + 1 << "\n";
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
                     << snapshot_freq << " iterations)\n";
        } else {
            std::cout << "final VTK snapshot only\n";
        }
    }
    
    double residual = 1.0;
    int snapshot_count = 0;
    
    if (config_.verbose) {
        if (config_.adaptive_dt) {
            std::cout << std::setw(8) << "Iter" 
                      << std::setw(15) << "Residual"
                      << std::setw(15) << "Max |u|"
                      << std::setw(12) << "dt"
                      << "\n";
        } else {
            std::cout << std::setw(8) << "Iter" 
                      << std::setw(15) << "Residual"
                      << std::setw(15) << "Max |u|"
                      << "\n";
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
                             << ": " << vtk_file << "\n";
                }
            } catch (const std::exception& e) {
                std::cerr << "Warning: Could not write VTK snapshot: " 
                         << e.what() << "\n";
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
                          << "\n";
            } else {
                std::cout << std::setw(8) << iter_ + 1
                          << std::setw(15) << std::scientific << std::setprecision(3) << residual
                          << std::setw(15) << std::fixed << max_vel
                          << "\n";
            }
        }
        
        if (residual < config_.tol) {
            if (config_.verbose) {
                std::cout << "Converged at iteration " << iter_ + 1 
                          << " with residual " << residual << "\n";
            }
            break;
        }
        
        // Check for divergence
        if (std::isnan(residual) || std::isinf(residual)) {
            if (config_.verbose) {
                std::cerr << "Solver diverged at iteration " << iter_ + 1 << "\n";
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
    
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            sum += velocity_.u(i, j);
            ++count;
        }
    }
    
    return sum / count;
}

double RANSSolver::wall_shear_stress() const {
    // Compute du/dy at the bottom wall
    // Using one-sided difference from first interior cell to wall
    double sum = 0.0;
    int count = 0;
    
    int j = mesh_->j_begin();  // First interior row
    double y_cell = mesh_->y(j);
    double y_wall = mesh_->y_min;
    double dist = y_cell - y_wall;
    
    for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
        // u at wall is 0 (no-slip)
        double dudy = velocity_.u(i, j) / dist;
        sum += dudy;
        ++count;
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
    // CFL condition: dt <= CFL * min(dx, dy) / |u_max|
    double u_max = 1e-10;  // Small value to avoid division by zero
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            // CRITICAL: Use correct magnitude for staggered grid (interpolates to cell center)
            // DO NOT mix u(i,j) and v(i,j) - they're at different physical locations!
            double u_mag = velocity_.magnitude(i, j);
            u_max = std::max(u_max, u_mag);
        }
    }
    double dt_cfl = config_.CFL_max * std::min(mesh_->dx, mesh_->dy) / u_max;
    
    // Diffusion stability: dt <= factor * min(dx^2, dy^2) / nu_eff_max
    double nu_eff_max = config_.nu;
    if (turb_model_) {
        for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
            for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
                nu_eff_max = std::max(nu_eff_max, config_.nu + nu_t_(i, j));
            }
        }
    }
    double dx_min = std::min(mesh_->dx, mesh_->dy);
    double dt_diff = 0.25 * dx_min * dx_min / nu_eff_max;
    
    // Take minimum of both constraints
    return std::min(dt_cfl, dt_diff);
}

void RANSSolver::write_vtk(const std::string& filename) const {
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
    
    int Nx = mesh_->Nx;
    int Ny = mesh_->Ny;
    
    // VTK header
    file << "# vtk DataFile Version 3.0\n";
    file << "RANS simulation output\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_POINTS\n";
    file << "DIMENSIONS " << Nx << " " << Ny << " 1\n";
    file << "ORIGIN " << mesh_->x_min << " " << mesh_->y_min << " 0\n";
    file << "SPACING " << mesh_->dx << " " << mesh_->dy << " 1\n";
    file << "POINT_DATA " << Nx * Ny << "\n";
    
    // Velocity vector field (interpolated from staggered grid to cell centers)
    file << "VECTORS velocity double\n";
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            // Staggered grid: interpolate u and v to cell centers
            double u_center = velocity_.u_center(i, j);
            double v_center = velocity_.v_center(i, j);
            file << u_center << " " << v_center << " 0\n";
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
    
    // Velocity magnitude (computed from cell-centered interpolated values)
    file << "SCALARS velocity_magnitude double 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int j = mesh_->j_begin(); j < mesh_->j_end(); ++j) {
        for (int i = mesh_->i_begin(); i < mesh_->i_end(); ++i) {
            // Staggered grid: use proper magnitude calculation
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
    
    file.close();
}

#ifdef USE_GPU_OFFLOAD
void RANSSolver::initialize_gpu_buffers() {
    // Verify GPU is available (throws if not)
    gpu::verify_device_available();
    
    // Get raw pointers to all field data
    field_total_size_ = (mesh_->Nx + 2) * (mesh_->Ny + 2);  // For cell-centered fields
    
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
    // k and omega are NOT mapped here - turbulence models manage their own GPU copies
    
    // Reynolds stress tensor components (alloc - will be computed by EARSM/TBNN on GPU)
    #pragma omp target enter data map(alloc: tau_xx_ptr_[0:field_total_size_])
    #pragma omp target enter data map(alloc: tau_xy_ptr_[0:field_total_size_])
    #pragma omp target enter data map(alloc: tau_yy_ptr_[0:field_total_size_])
    
    // Gradient scratch buffers for turbulence models (alloc, not to - computed on GPU)
    #pragma omp target enter data map(alloc: dudx_ptr_[0:field_total_size_])
    #pragma omp target enter data map(alloc: dudy_ptr_[0:field_total_size_])
    #pragma omp target enter data map(alloc: dvdx_ptr_[0:field_total_size_])
    #pragma omp target enter data map(alloc: dvdy_ptr_[0:field_total_size_])
    #pragma omp target enter data map(to: wall_distance_ptr_[0:field_total_size_])  // Precomputed, upload once
    
    // Allocate device-resident "old velocity" buffers for residual computation
    // Host storage exists but is never used - device copy is authoritative
    // This eliminates per-step H→D upload for residual computation
    velocity_old_u_ptr_ = velocity_old_.u_data().data();
    velocity_old_v_ptr_ = velocity_old_.v_data().data();
    
    #pragma omp target enter data map(alloc: velocity_old_u_ptr_[0:u_total_size])
    #pragma omp target enter data map(alloc: velocity_old_v_ptr_[0:v_total_size])
    
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
    
    // Delete gradient scratch buffers
    #pragma omp target exit data map(delete: dudx_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: dudy_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: dvdx_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: dvdy_ptr_[0:field_total_size_])
    #pragma omp target exit data map(delete: wall_distance_ptr_[0:field_total_size_])
    
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
    // k and omega are managed by turbulence model independently
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
    
    view.p = pressure_ptr_;
    view.p_corr = pressure_corr_ptr_;
    view.nu_t = nu_t_ptr_;
    view.nu_eff = nu_eff_ptr_;
    view.rhs = rhs_poisson_ptr_;
    view.div = div_velocity_ptr_;
    view.cell_stride = mesh_->total_Nx();
    
    view.conv_u = conv_u_ptr_;
    view.conv_v = conv_v_ptr_;
    view.diff_u = diff_u_ptr_;
    view.diff_v = diff_v_ptr_;
    
    view.Nx = mesh_->Nx;
    view.Ny = mesh_->Ny;
    view.Ng = mesh_->Nghost;
    view.dx = mesh_->dx;
    view.dy = mesh_->dy;
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
    return TurbulenceDeviceView();  // Returns invalid view (all nullptrs)
}
#endif

} // namespace nncfd

