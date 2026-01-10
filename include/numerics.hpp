/// @file numerics.hpp
/// @brief Safe numerical utilities for scientific computing
///
/// Provides:
/// - safe_divide: Division with floor to prevent overflow from small denominators
/// - bounded_ratio: Division with both floor and ceiling
/// - Common floor constants for turbulence quantities
///
/// These utilities address numerical stability risks in turbulence transport
/// models where divisions by k, omega, or wall distance can cause overflow.

#pragma once

#include <cmath>
#include <algorithm>

namespace nncfd {
namespace numerics {

/// Safe division with floor to prevent overflow from small denominators.
/// Returns num / den_safe where |den_safe| >= floor and sign is preserved.
///
/// @param num    Numerator
/// @param den    Denominator (magnitude floored, sign preserved)
/// @param floor  Minimum absolute value for denominator (default 1e-30)
/// @return       num / den_safe (sign-preserving)
///
/// Example:
///   safe_divide(1.0, 0.0)      → 1e30
///   safe_divide(1.0, 1e-40)    → 1e30
///   safe_divide(1.0, 2.0)      → 0.5
///   safe_divide(-1.0, -0.5)    → 2.0
inline double safe_divide(double num, double den, double floor = 1e-30) {
    const double mag = std::max(std::abs(den), floor);
    // Preserve sign of denominator (treat ±0 as +0)
    const double den_safe = (den < 0.0) ? -mag : mag;
    return num / den_safe;
}

/// Bounded ratio: safe division with ceiling on result magnitude.
/// Useful for ratios like omega/k in turbulence models where the ratio
/// itself should be bounded even if both numerator and denominator are valid.
///
/// @param num      Numerator
/// @param den      Denominator
/// @param floor    Minimum absolute denominator (default 1e-30)
/// @param ceiling  Maximum absolute result (default 1e10)
/// @return         Clamped result in [-ceiling, ceiling]
///
/// Example:
///   bounded_ratio(1e5, 1e-8)   → 1e10 (clamped)
///   bounded_ratio(-1e5, 1e-8)  → -1e10 (clamped)
///   bounded_ratio(1.0, 2.0)    → 0.5
inline double bounded_ratio(double num, double den,
                            double floor = 1e-30, double ceiling = 1e10) {
    double ratio = safe_divide(num, den, floor);
    return std::clamp(ratio, -ceiling, ceiling);
}

/// Check if a value is finite (not NaN or Inf)
inline bool is_finite(double val) {
    return std::isfinite(val);
}

/// Check if a value is within a valid range
inline bool in_range(double val, double lo, double hi) {
    return val >= lo && val <= hi;
}

/// Clamp a value to a valid range, returning whether clamping occurred
inline double clamp_with_flag(double val, double lo, double hi, bool& clamped) {
    if (val < lo) { clamped = true; return lo; }
    if (val > hi) { clamped = true; return hi; }
    clamped = false;
    return val;
}

// =============================================================================
// Turbulence model constants
// =============================================================================

/// Standard k-ε/k-ω model constant C_μ
/// Relates turbulent viscosity to k and ε: ν_t = C_μ * k² / ε
/// or equivalently for k-ω: ν_t = k / ω (with C_μ embedded in ω definition)
/// Also used to convert between ε and ω: ε = C_μ * k * ω
constexpr double C_MU = 0.09;

/// von Karman constant for wall-bounded turbulent flows
/// Appears in log-law: u+ = (1/κ) * ln(y+) + B
constexpr double KAPPA = 0.41;

/// van Driest damping constant
/// Used in damping function: f = 1 - exp(-y+/A+)
constexpr double A_PLUS = 26.0;

// =============================================================================
// Common floor/ceiling constants for turbulence quantities
// =============================================================================

/// Minimum turbulent kinetic energy (k) [m²/s²]
/// Prevents division by zero in omega/k ratios
constexpr double K_FLOOR = 1e-10;

/// Minimum specific dissipation rate (omega) [1/s]
/// Prevents division by zero in k/omega ratios
constexpr double OMEGA_FLOOR = 1e-10;

/// Minimum dissipation rate (epsilon) [m²/s³]
/// Prevents division by zero in k/epsilon ratios
constexpr double EPS_FLOOR = 1e-20;

/// Minimum wall distance [m]
/// Prevents division by zero in wall function calculations
constexpr double Y_WALL_FLOOR = 1e-10;

/// Maximum omega/k ratio
/// Prevents numerical blow-up in omega production term
constexpr double OMEGA_OVER_K_MAX = 1e8;

/// Maximum turbulent viscosity ratio (nu_t / nu)
/// Typical physical limit is O(1000) for highly turbulent flows
constexpr double NU_T_RATIO_MAX = 1e6;

} // namespace numerics
} // namespace nncfd
