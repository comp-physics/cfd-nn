/// @file stencil_operators.hpp
/// @brief Spatial operator primitives for staggered grid discretization
///
/// These are the fundamental 1D building blocks for all derivative/interpolation ops.
/// For stretched grids, multiply result by appropriate metric at output location.
///
/// Naming convention:
///   - D = derivative, I = interpolation
///   - cf = center→face, fc = face→center, same = same stagger
///   - O2 = 2nd order, O4 = 4th order

#pragma once

namespace nncfd {
namespace stencil {

#ifdef USE_GPU_OFFLOAD
#pragma omp declare target
#endif

// ============================================================================
// O2 Primitives (uniform h)
// ============================================================================

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

// ============================================================================
// O4 Primitives (uniform h)
// ============================================================================

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

// ============================================================================
// Boundary-safe order selection
// These helpers determine if O4 stencil is safe at a given index
// ============================================================================

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

#ifdef USE_GPU_OFFLOAD
#pragma omp end declare target
#endif

}  // namespace stencil
}  // namespace nncfd
