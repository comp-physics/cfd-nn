/// @file test_utilities.hpp
/// @brief Common test utilities for CPU/GPU comparison and field validation

#pragma once

#include <cmath>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <vector>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

namespace nncfd {
namespace test {

//=============================================================================
// Field Comparison Utilities
//=============================================================================

/// Unified field comparison result structure
/// Tracks max/RMS differences and location of worst error
struct FieldComparison {
    double max_abs_diff = 0.0;
    double max_rel_diff = 0.0;
    double rms_diff = 0.0;
    int worst_i = 0, worst_j = 0, worst_k = 0;
    double ref_at_worst = 0.0;
    double test_at_worst = 0.0;
    int count = 0;

    /// Update comparison with a new point (3D version)
    void update(int i, int j, int k, double ref_val, double test_val) {
        double abs_diff = std::abs(ref_val - test_val);
        double rel_diff = abs_diff / (std::abs(ref_val) + 1e-15);

        rms_diff += abs_diff * abs_diff;
        count++;

        if (abs_diff > max_abs_diff) {
            max_abs_diff = abs_diff;
            max_rel_diff = rel_diff;
            worst_i = i; worst_j = j; worst_k = k;
            ref_at_worst = ref_val;
            test_at_worst = test_val;
        }
    }

    /// Update comparison with a new point (2D version)
    void update(int i, int j, double ref_val, double test_val) {
        update(i, j, 0, ref_val, test_val);
    }

    /// Update comparison without location tracking (simple value comparison)
    void update(double ref_val, double test_val) {
        update(0, 0, 0, ref_val, test_val);
    }

    /// Finalize RMS computation after all updates
    void finalize() {
        if (count > 0) {
            rms_diff = std::sqrt(rms_diff / count);
        }
    }

    /// Print comparison results with optional field name
    void print(const std::string& name = "") const {
        if (!name.empty()) {
            std::cout << "  " << name << ":\n";
            std::cout << "    Max abs diff: " << std::scientific << max_abs_diff << "\n";
            std::cout << "    Max rel diff: " << max_rel_diff << "\n";
            std::cout << "    RMS diff:     " << rms_diff << "\n";
            if (max_abs_diff > 0) {
                std::cout << "    Worst at (" << worst_i << "," << worst_j << "," << worst_k << "): "
                          << "ref=" << ref_at_worst << ", test=" << test_at_worst << "\n";
            }
        } else {
            std::cout << std::scientific << std::setprecision(6);
            std::cout << "  Max absolute difference: " << max_abs_diff << "\n";
            std::cout << "  Max relative difference: " << max_rel_diff << "\n";
            std::cout << "  RMS difference:          " << rms_diff << "\n";
            if (max_abs_diff > 0) {
                std::cout << "  Worst at (" << worst_i << "," << worst_j << "," << worst_k << "): "
                          << "ref=" << ref_at_worst << ", test=" << test_at_worst << "\n";
            }
        }
    }

    /// Check if comparison is within tolerance
    bool within_tolerance(double tol) const {
        return max_abs_diff < tol;
    }

    /// Reset comparison state
    void reset() {
        max_abs_diff = 0.0;
        max_rel_diff = 0.0;
        rms_diff = 0.0;
        worst_i = worst_j = worst_k = 0;
        ref_at_worst = test_at_worst = 0.0;
        count = 0;
    }
};

//=============================================================================
// Tolerance Constants
//=============================================================================

/// CPU/GPU bitwise comparison tolerance
constexpr double BITWISE_TOLERANCE = 1e-10;

/// Minimum expected FP difference (to verify different backends executed)
constexpr double MIN_EXPECTED_DIFF = 1e-14;

//=============================================================================
// Floating-Point Comparison Utilities (rel+abs tolerance)
//=============================================================================

/// Check if two values are close using combined relative and absolute tolerance.
/// Returns true if |a - b| <= atol + rtol * max(|a|, |b|)
/// This is the standard approach used in numpy.isclose() and similar.
inline bool check_close(double a, double b, double rtol = 1e-5, double atol = 1e-8) {
    double diff = std::abs(a - b);
    double scale = std::max(std::abs(a), std::abs(b));
    return diff <= atol + rtol * scale;
}

/// Check if value is close to zero (absolute tolerance only)
inline bool check_near_zero(double val, double atol = 1e-10) {
    return std::abs(val) <= atol;
}

/// Check if value is within relative tolerance of expected
inline bool check_relative(double actual, double expected, double rtol = 1e-5) {
    if (std::abs(expected) < 1e-15) return std::abs(actual) < 1e-15;
    return std::abs(actual - expected) / std::abs(expected) <= rtol;
}

/// Macro for cleaner test assertions with rel+abs tolerance
#define CHECK_CLOSE(a, b) nncfd::test::check_close((a), (b))
#define CHECK_CLOSE_TOL(a, b, rtol, atol) nncfd::test::check_close((a), (b), (rtol), (atol))
#define CHECK_NEAR_ZERO(val) nncfd::test::check_near_zero((val))
#define CHECK_RELATIVE(actual, expected, rtol) nncfd::test::check_relative((actual), (expected), (rtol))

//=============================================================================
// Utility Functions
//=============================================================================

/// Check if a file exists
inline bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

//=============================================================================
// Field Helper Functions
//=============================================================================

/// Compute relative L2 difference between two scalar fields
template<typename MeshT, typename FieldT>
inline double compute_l2_diff(const FieldT& p1, const FieldT& p2, const MeshT& mesh) {
    double diff = 0.0, norm = 0.0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double d = p1(i, j, k) - p2(i, j, k);
                diff += d * d;
                norm += p1(i, j, k) * p1(i, j, k);
            }
        }
    }
    if (norm < 1e-30) norm = 1.0;
    return std::sqrt(diff / norm);
}

/// Compute max absolute difference between two scalar fields
template<typename MeshT, typename FieldT>
inline double compute_max_diff(const FieldT& p1, const FieldT& p2, const MeshT& mesh) {
    double max_diff = 0.0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                max_diff = std::max(max_diff, std::abs(p1(i, j, k) - p2(i, j, k)));
            }
        }
    }
    return max_diff;
}

/// Compute mean of a scalar field over interior cells
template<typename MeshT, typename FieldT>
inline double compute_mean(const FieldT& p, const MeshT& mesh) {
    double sum = 0.0;
    int count = 0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                sum += p(i, j, k);
                ++count;
            }
        }
    }
    if (count == 0) return 0.0;
    return sum / count;
}

/// Subtract mean from a scalar field (pressure gauge normalization)
template<typename MeshT, typename FieldT>
inline void subtract_mean(FieldT& p, const MeshT& mesh) {
    double mean = compute_mean(p, mesh);
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                p(i, j, k) -= mean;
            }
        }
    }
}

/// Compute L2 error against exact solution (3D, with mean subtraction for Neumann)
template<typename MeshT, typename FieldT, typename Solution>
inline double compute_l2_error_3d(const FieldT& p_num, const MeshT& mesh, const Solution& sol) {
    double p_mean = 0.0, exact_mean = 0.0;
    int count = 0;

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                p_mean += p_num(i, j, k);
                exact_mean += sol.p(mesh.x(i), mesh.y(j), mesh.z(k));
                ++count;
            }
        }
    }
    p_mean /= count;
    exact_mean /= count;

    double l2_error = 0.0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double exact = sol.p(mesh.x(i), mesh.y(j), mesh.z(k));
                double diff = (p_num(i, j, k) - p_mean) - (exact - exact_mean);
                l2_error += diff * diff;
            }
        }
    }
    return std::sqrt(l2_error / count);
}

/// Compute L2 error against exact solution (2D, with mean subtraction for Neumann)
template<typename MeshT, typename FieldT, typename Solution>
inline double compute_l2_error_2d(const FieldT& p_num, const MeshT& mesh, const Solution& sol) {
    double p_mean = 0.0, exact_mean = 0.0;
    int count = 0;

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            p_mean += p_num(i, j);
            exact_mean += sol.p(mesh.x(i), mesh.y(j));
            ++count;
        }
    }
    p_mean /= count;
    exact_mean /= count;

    double l2_error = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double exact = sol.p(mesh.x(i), mesh.y(j));
            double diff = (p_num(i, j) - p_mean) - (exact - exact_mean);
            l2_error += diff * diff;
        }
    }
    return std::sqrt(l2_error / count);
}

//=============================================================================
// Unified L2 Error Computation
//=============================================================================

/// Unified L2 error computation against an exact function.
/// Works with ScalarField, supports 2D/3D meshes, optional mean subtraction.
///
/// @param field     Numerical solution field
/// @param mesh      Computational mesh
/// @param exact_fn  Callable: exact_fn(x, y) for 2D, exact_fn(x, y, z) for 3D
/// @param subtract_mean  If true, compare (field - mean(field)) vs (exact - mean(exact))
/// @param relative  If true, return error / ||exact||, else return absolute L2 norm
///
/// Example:
///   auto exact = [](double x, double y) { return std::sin(x) * std::cos(y); };
///   double err = compute_l2_error(p, mesh, exact, true, true);
///
template<typename FieldT, typename MeshT, typename ExactFn>
inline double compute_l2_error(const FieldT& field, const MeshT& mesh, ExactFn&& exact_fn,
                                bool subtract_mean = true, bool relative = true) {
    double field_mean = 0.0, exact_mean = 0.0;
    double error_sq = 0.0, norm_sq = 0.0;
    int count = 0;

    // Determine if 3D based on mesh
    const bool is_3d = (mesh.Nz > 1);

    // First pass: compute means if needed
    if (subtract_mean) {
        if (is_3d) {
            for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
                for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                    for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                        field_mean += field(i, j, k);
                        exact_mean += exact_fn(mesh.x(i), mesh.y(j), mesh.z(k));
                        ++count;
                    }
                }
            }
        } else {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    field_mean += field(i, j);
                    exact_mean += exact_fn(mesh.x(i), mesh.y(j));
                    ++count;
                }
            }
        }
        if (count > 0) {
            field_mean /= count;
            exact_mean /= count;
        }
        count = 0;  // Reset for second pass
    }

    // Second pass: compute error
    if (is_3d) {
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    double f_val = field(i, j, k) - field_mean;
                    double e_val = exact_fn(mesh.x(i), mesh.y(j), mesh.z(k)) - exact_mean;
                    double diff = f_val - e_val;
                    error_sq += diff * diff;
                    norm_sq += e_val * e_val;
                    ++count;
                }
            }
        }
    } else {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double f_val = field(i, j) - field_mean;
                double e_val = exact_fn(mesh.x(i), mesh.y(j)) - exact_mean;
                double diff = f_val - e_val;
                error_sq += diff * diff;
                norm_sq += e_val * e_val;
                ++count;
            }
        }
    }

    if (count == 0) return 0.0;

    double l2_abs = std::sqrt(error_sq / count);
    if (relative && norm_sq > 1e-14) {
        return l2_abs / std::sqrt(norm_sq / count);
    }
    return l2_abs;
}

/// Compute L2 error for velocity u-component against exact function
/// Uses cell-centered interpolation of staggered u-velocity
template<typename VelFieldT, typename MeshT, typename ExactFn>
inline double compute_l2_error_velocity_u(const VelFieldT& vel, const MeshT& mesh,
                                           ExactFn&& u_exact, bool relative = true) {
    double error_sq = 0.0, norm_sq = 0.0;
    int count = 0;

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            // Interpolate u to cell center
            double u_num = 0.5 * (vel.u(i, j) + vel.u(i+1, j));
            double u_ex = u_exact(mesh.x(i), mesh.y(j));
            double diff = u_num - u_ex;
            error_sq += diff * diff;
            norm_sq += u_ex * u_ex;
            ++count;
        }
    }

    if (count == 0) return 0.0;

    double l2_abs = std::sqrt(error_sq / count);
    if (relative && norm_sq > 1e-14) {
        return l2_abs / std::sqrt(norm_sq / count);
    }
    return l2_abs;
}

} // namespace test
} // namespace nncfd

//=============================================================================
// Domain Iteration Macros
//=============================================================================

/// Iterate over interior cells of a 2D mesh
#define FOR_INTERIOR_2D(mesh, i, j) \
    for (int j = (mesh).j_begin(); j < (mesh).j_end(); ++j) \
    for (int i = (mesh).i_begin(); i < (mesh).i_end(); ++i)

/// Iterate over interior cells of a 3D mesh
#define FOR_INTERIOR_3D(mesh, i, j, k) \
    for (int k = (mesh).k_begin(); k < (mesh).k_end(); ++k) \
    for (int j = (mesh).j_begin(); j < (mesh).j_end(); ++j) \
    for (int i = (mesh).i_begin(); i < (mesh).i_end(); ++i)

/// Dimension-agnostic iteration over interior cells (2D or 3D)
/// In 2D, k iterates once (k_begin to k_end is a single iteration).
/// This allows the same loop body to work for both 2D and 3D meshes.
///
/// Usage:
///   FOR_INTERIOR(mesh, i, j, k) {
///       // Body executes for all interior cells
///       // In 2D: k is always 0 (or mesh.Nghost for ghost-offset meshes)
///       // In 3D: k iterates over z-slices
///   }
///
#define FOR_INTERIOR(mesh, i, j, k) \
    for (int k = (mesh).k_begin(); k < (mesh).k_end(); ++k) \
    for (int j = (mesh).j_begin(); j < (mesh).j_end(); ++j) \
    for (int i = (mesh).i_begin(); i < (mesh).i_end(); ++i)

/// Iterate over interior cells with optional stride for cache efficiency
#define FOR_INTERIOR_STRIDED(mesh, i, j, k, i_stride) \
    for (int k = (mesh).k_begin(); k < (mesh).k_end(); ++k) \
    for (int j = (mesh).j_begin(); j < (mesh).j_end(); ++j) \
    for (int i = (mesh).i_begin(); i < (mesh).i_end(); i += (i_stride))

//=============================================================================
// GPU/CPU Test Utilities
//=============================================================================

namespace nncfd {
namespace test {

//=============================================================================
// GPU Availability and Verification (unified interface)
//=============================================================================

namespace gpu {

/// Check if any GPU device is available
inline bool available() {
#ifdef USE_GPU_OFFLOAD
    return omp_get_num_devices() > 0;
#else
    return false;
#endif
}

/// Get number of available GPU devices
inline int device_count() {
#ifdef USE_GPU_OFFLOAD
    return omp_get_num_devices();
#else
    return 0;
#endif
}

/// Verify that code actually executes on GPU (not just compiled for it)
/// Returns true if target region executes on device
inline bool verify_execution() {
#ifdef USE_GPU_OFFLOAD
    if (omp_get_num_devices() == 0) return false;
    int on_device = 0;
    #pragma omp target map(tofrom: on_device)
    { on_device = !omp_is_initial_device(); }
    return on_device != 0;
#else
    return false;
#endif
}

/// Check if this is a GPU build (compile-time check)
inline constexpr bool is_gpu_build() {
#ifdef USE_GPU_OFFLOAD
    return true;
#else
    return false;
#endif
}

/// Get build type string for display
inline const char* build_type_string() {
#ifdef USE_GPU_OFFLOAD
    return "GPU (USE_GPU_OFFLOAD=ON)";
#else
    return "CPU (USE_GPU_OFFLOAD=OFF)";
#endif
}

/// Print GPU configuration info
inline void print_config() {
    std::cout << "Build: " << build_type_string() << "\n";
#ifdef USE_GPU_OFFLOAD
    std::cout << "Devices: " << device_count() << "\n";
    if (available()) {
        std::cout << "GPU execution: " << (verify_execution() ? "YES" : "NO") << "\n";
    }
#endif
}

} // namespace gpu

/// Test case configuration for turbulence model tests
struct TurbulenceTestCase {
    int nx, ny;
    int seed;
};

/// Default test cases for turbulence model testing
inline std::vector<TurbulenceTestCase> default_turbulence_cases() {
    return {{64, 64, 0}, {48, 96, 1}, {63, 97, 2}, {128, 128, 3}};
}

/// Smaller test cases for computationally expensive tests (GEP, NN-MLP)
inline std::vector<TurbulenceTestCase> small_turbulence_cases() {
    return {{64, 64, 0}, {48, 96, 1}, {128, 128, 2}};
}

/// Create a deterministic but non-trivial velocity field for testing
/// Parabolic base profile + sinusoidal + random perturbation
template<typename MeshT, typename VectorFieldT>
inline void create_test_velocity_field(const MeshT& mesh, VectorFieldT& vel, int seed = 0) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-0.1, 0.1);

    FOR_INTERIOR_2D(mesh, i, j) {
        double y = mesh.yc[j];
        double x = mesh.xc[i];

        // Parabolic + perturbation
        double u_base = 4.0 * y * (1.0 - y);
        double v_base = 0.1 * std::sin(2.0 * M_PI * x);

        vel.u(i, j) = u_base + 0.01 * dist(rng);
        vel.v(i, j) = v_base + 0.01 * dist(rng);
    }
}

/// Tolerance check result with combined abs/rel check
struct ToleranceCheck {
    bool passed;
    double abs_diff;
    double rel_diff;

    ToleranceCheck(double abs_d, double rel_d, double tol_abs, double tol_rel)
        : passed(abs_d <= tol_abs || rel_d <= tol_rel), abs_diff(abs_d), rel_diff(rel_d) {}

    void print_result(const std::string& test_name = "") const {
        if (!test_name.empty()) {
            std::cout << "    " << test_name << ": ";
        }
        std::cout << (passed ? "PASSED" : "FAILED") << "\n";
    }
};

/// CPU/GPU comparison tolerances (tight for MAC-consistent paths)
constexpr double GPU_CPU_ABS_TOL = 1e-12;
constexpr double GPU_CPU_REL_TOL = 1e-10;

/// Cross-build comparison tolerances (CPU reference vs GPU with different compiler/rounding)
constexpr double CROSS_BUILD_ABS_TOL = 1e-6;
constexpr double CROSS_BUILD_REL_TOL = 1e-5;

/// Check GPU/CPU consistency with tight tolerances
inline ToleranceCheck check_gpu_cpu_consistency(const FieldComparison& cmp) {
    return ToleranceCheck(cmp.max_abs_diff, cmp.max_rel_diff, GPU_CPU_ABS_TOL, GPU_CPU_REL_TOL);
}

/// Check cross-build consistency with relaxed tolerances
inline ToleranceCheck check_cross_build_consistency(const FieldComparison& cmp) {
    return ToleranceCheck(cmp.max_abs_diff, cmp.max_rel_diff, CROSS_BUILD_ABS_TOL, CROSS_BUILD_REL_TOL);
}

//=============================================================================
// Boundary Condition Pattern Helpers
//=============================================================================

/// Common velocity BC patterns for test setup
/// Reduces duplication in test files where BC setup is repeated 25+ times
enum class BCPattern {
    Channel2D,      ///< Periodic x, NoSlip y (classic 2D channel)
    Channel3D,      ///< Periodic x/z, NoSlip y (3D channel)
    FullyPeriodic,  ///< All directions periodic (e.g., Taylor-Green)
    AllNoSlip,      ///< All walls (e.g., lid-driven cavity)
    Duct,           ///< Periodic x, NoSlip y/z (rectangular duct)
    Pipe            ///< Periodic x, NoSlip y/z (same as Duct for now)
};

/// Get a descriptive name for a BC pattern
inline const char* bc_pattern_name(BCPattern pattern) {
    switch (pattern) {
        case BCPattern::Channel2D:     return "Channel2D";
        case BCPattern::Channel3D:     return "Channel3D";
        case BCPattern::FullyPeriodic: return "FullyPeriodic";
        case BCPattern::AllNoSlip:     return "AllNoSlip";
        case BCPattern::Duct:          return "Duct";
        case BCPattern::Pipe:          return "Pipe";
    }
    return "Unknown";
}

} // namespace test
} // namespace nncfd

// Include solver.hpp for VelocityBC - forward declare to avoid circular includes
// Tests that use create_velocity_bc should include solver.hpp themselves
#include "solver.hpp"

namespace nncfd {
namespace test {

/// Create a VelocityBC struct for a common pattern
/// Usage: solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel2D));
inline VelocityBC create_velocity_bc(BCPattern pattern) {
    VelocityBC bc;

    switch (pattern) {
        case BCPattern::Channel2D:
            bc.x_lo = VelocityBC::Periodic;
            bc.x_hi = VelocityBC::Periodic;
            bc.y_lo = VelocityBC::NoSlip;
            bc.y_hi = VelocityBC::NoSlip;
            bc.z_lo = VelocityBC::Periodic;
            bc.z_hi = VelocityBC::Periodic;
            break;

        case BCPattern::Channel3D:
            bc.x_lo = VelocityBC::Periodic;
            bc.x_hi = VelocityBC::Periodic;
            bc.y_lo = VelocityBC::NoSlip;
            bc.y_hi = VelocityBC::NoSlip;
            bc.z_lo = VelocityBC::Periodic;
            bc.z_hi = VelocityBC::Periodic;
            break;

        case BCPattern::FullyPeriodic:
            bc.x_lo = VelocityBC::Periodic;
            bc.x_hi = VelocityBC::Periodic;
            bc.y_lo = VelocityBC::Periodic;
            bc.y_hi = VelocityBC::Periodic;
            bc.z_lo = VelocityBC::Periodic;
            bc.z_hi = VelocityBC::Periodic;
            break;

        case BCPattern::AllNoSlip:
            bc.x_lo = VelocityBC::NoSlip;
            bc.x_hi = VelocityBC::NoSlip;
            bc.y_lo = VelocityBC::NoSlip;
            bc.y_hi = VelocityBC::NoSlip;
            bc.z_lo = VelocityBC::NoSlip;
            bc.z_hi = VelocityBC::NoSlip;
            break;

        case BCPattern::Duct:
        case BCPattern::Pipe:
            bc.x_lo = VelocityBC::Periodic;
            bc.x_hi = VelocityBC::Periodic;
            bc.y_lo = VelocityBC::NoSlip;
            bc.y_hi = VelocityBC::NoSlip;
            bc.z_lo = VelocityBC::NoSlip;
            bc.z_hi = VelocityBC::NoSlip;
            break;
    }

    return bc;
}

} // namespace test
} // namespace nncfd
