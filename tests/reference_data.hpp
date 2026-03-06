#pragma once
/// @file reference_data.hpp
/// @brief Embedded DNS reference data for CI validation tests
///
/// Sources:
///   MKM: Moser, Kim & Mansour (1999) JFM 399:263-291, Re_tau=180
///   TGV: Brachet et al. (1983) JFM 130:411-452, Re=1600

#include <array>
#include <cmath>

namespace nncfd {
namespace reference {

// ============================================================================
// MKM DNS Channel Flow, Re_tau = 180
// ============================================================================

/// Mean velocity profile U+(y+)
/// 19 points spanning viscous sublayer through channel center
struct MKMPoint { double y_plus; double u_plus; };

constexpr std::array<MKMPoint, 19> mkm_retau180_u_profile = {{
    {0.05,   0.05},
    {0.1,    0.1},
    {0.2,    0.2},
    {0.5,    0.5},
    {1.0,    1.0},
    {2.0,    2.0},
    {5.0,    5.0},
    {8.0,    7.8},
    {10.0,   9.2},
    {15.0,   11.5},
    {20.0,   13.0},
    {30.0,   14.8},
    {50.0,   16.9},
    {70.0,   18.2},
    {100.0,  19.4},
    {120.0,  20.1},
    {140.0,  20.6},
    {160.0,  21.0},
    {180.0,  21.3},
}};

/// Reynolds stress profiles at selected y+ locations
/// Values from MKM Table 7 (Re_tau=180)
/// uu+ = <u'u'>/u_tau^2, vv+ = <v'v'>/u_tau^2, etc.
struct MKMStressPoint { double y_plus; double uu_plus; double vv_plus; double ww_plus; double uv_plus; };

constexpr std::array<MKMStressPoint, 15> mkm_retau180_stresses = {{
    // y+     uu+     vv+     ww+    -uv+
    {1.0,    0.04,   0.0003, 0.008,  0.003},
    {2.0,    0.27,   0.002,  0.04,   0.018},
    {5.0,    1.67,   0.014,  0.24,   0.12},
    {10.0,   4.84,   0.058,  0.79,   0.44},
    {15.0,   6.69,   0.13,   1.27,   0.66},
    {20.0,   7.14,   0.23,   1.54,   0.76},
    {30.0,   6.30,   0.44,   1.72,   0.82},
    {50.0,   4.23,   0.68,   1.52,   0.72},
    {70.0,   3.03,   0.77,   1.25,   0.58},
    {100.0,  1.92,   0.76,   0.92,   0.38},
    {120.0,  1.44,   0.69,   0.73,   0.26},
    {140.0,  1.06,   0.56,   0.56,   0.15},
    {160.0,  0.74,   0.37,   0.40,   0.06},
    {170.0,  0.59,   0.25,   0.31,   0.03},
    {180.0,  0.51,   0.18,   0.26,   0.00},
}};

/// Law of the wall reference for quick checks
/// Viscous sublayer: U+ = y+ (exact for y+ < 5)
/// Log law: U+ = (1/kappa) * ln(y+) + B (approximate for y+ > 30)
constexpr double kappa = 0.41;
constexpr double B = 5.2;

inline double law_of_wall(double y_plus) {
    if (y_plus < 5.0) return y_plus;
    return (1.0 / kappa) * std::log(y_plus) + B;
}

// ============================================================================
// Brachet TGV, Re = 1600
// ============================================================================

/// Peak dissipation rate for TGV at Re=1600
/// Brachet et al. (1983): epsilon_max / (U0^3/L) ~ 0.0127 at t* ~ 9.0
constexpr double tgv_re1600_epsilon_peak = 0.0127;
constexpr double tgv_re1600_t_peak = 9.0;

/// TGV analytical KE at t=0: E0 = 0.125 (for unit amplitude on [0,2pi]^3)
constexpr double tgv_re1600_E0 = 0.125;

// ============================================================================
// Poiseuille flow analytical solution
// ============================================================================

/// Analytical Poiseuille flow: U(y) = -(dp/dx)/(2*nu) * y * (Ly - y)
/// where y is measured from bottom wall, Ly is channel height
inline double poiseuille_velocity(double y, double dp_dx, double nu, double Ly) {
    return -dp_dx / (2.0 * nu) * y * (Ly - y);
}

inline double poiseuille_centerline(double dp_dx, double nu, double Ly) {
    return -dp_dx / (2.0 * nu) * (Ly / 2.0) * (Ly / 2.0);
}

inline double poiseuille_bulk(double dp_dx, double nu, double Ly) {
    return (2.0 / 3.0) * poiseuille_centerline(dp_dx, nu, Ly);
}

} // namespace reference
} // namespace nncfd
