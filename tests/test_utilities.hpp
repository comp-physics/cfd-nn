/// @file test_utilities.hpp
/// @brief Common test utilities for CPU/GPU comparison and field validation

#pragma once

#include <cmath>
#include <cstdlib>
#include <ctime>
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
/// Tracks max/RMS differences, location of worst error, and normalized norms
struct FieldComparison {
    double max_abs_diff = 0.0;
    double max_rel_diff = 0.0;
    double rms_diff = 0.0;
    int worst_i = 0, worst_j = 0, worst_k = 0;
    double ref_at_worst = 0.0;
    double test_at_worst = 0.0;
    int count = 0;

    // For normalized L2/Linf norms
    double sum_sq_diff_ = 0.0;   // Sum of (ref - test)^2
    double sum_sq_ref_ = 0.0;    // Sum of ref^2
    double max_abs_ref_ = 0.0;   // max|ref| for Linf normalization

    /// Update comparison with a new point (3D version)
    void update(int i, int j, int k, double ref_val, double test_val) {
        double abs_diff = std::abs(ref_val - test_val);
        double rel_diff = abs_diff / (std::abs(ref_val) + 1e-15);

        sum_sq_diff_ += abs_diff * abs_diff;
        sum_sq_ref_ += ref_val * ref_val;
        max_abs_ref_ = std::max(max_abs_ref_, std::abs(ref_val));
        rms_diff += abs_diff * abs_diff;  // Will be sqrt'd in finalize()
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

    /// Compute normalized L2 norm: ||test - ref||_2 / (||ref||_2 + eps)
    /// More stable than max relative diff for comparing fields
    double rel_l2(double eps = 1e-30) const {
        double l2_diff = std::sqrt(sum_sq_diff_);
        double l2_ref = std::sqrt(sum_sq_ref_);
        return l2_diff / (l2_ref + eps);
    }

    /// Compute normalized L-infinity norm: ||test - ref||_inf / (||ref||_inf + eps)
    double rel_linf(double eps = 1e-30) const {
        return max_abs_diff / (max_abs_ref_ + eps);
    }

    /// Print comparison results with optional field name
    void print(const std::string& name = "") const {
        if (!name.empty()) {
            std::cout << "  " << name << ":\n";
            std::cout << "    Max abs diff: " << std::scientific << max_abs_diff << "\n";
            std::cout << "    Max rel diff: " << max_rel_diff << "\n";
            std::cout << "    RMS diff:     " << rms_diff << "\n";
            std::cout << "    Rel L2 norm:  " << rel_l2() << "\n";
            std::cout << "    Rel Linf:     " << rel_linf() << "\n";
            if (max_abs_diff > 0) {
                std::cout << "    Worst at (" << worst_i << "," << worst_j << "," << worst_k << "): "
                          << "ref=" << ref_at_worst << ", test=" << test_at_worst << "\n";
            }
        } else {
            std::cout << std::scientific << std::setprecision(6);
            std::cout << "  Max absolute difference: " << max_abs_diff << "\n";
            std::cout << "  Max relative difference: " << max_rel_diff << "\n";
            std::cout << "  RMS difference:          " << rms_diff << "\n";
            std::cout << "  Rel L2 norm:             " << rel_l2() << "\n";
            std::cout << "  Rel Linf norm:           " << rel_linf() << "\n";
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

    /// Check if normalized L2 norm is within tolerance (more robust for CI)
    bool within_rel_l2_tolerance(double tol) const {
        return rel_l2() < tol;
    }

    /// Reset comparison state
    void reset() {
        max_abs_diff = 0.0;
        max_rel_diff = 0.0;
        rms_diff = 0.0;
        worst_i = worst_j = worst_k = 0;
        ref_at_worst = test_at_worst = 0.0;
        count = 0;
        sum_sq_diff_ = 0.0;
        sum_sq_ref_ = 0.0;
        max_abs_ref_ = 0.0;
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

/// Check if OMP_TARGET_OFFLOAD is set to MANDATORY
inline bool is_mandatory_offload() {
    const char* env = std::getenv("OMP_TARGET_OFFLOAD");
    return env && std::string(env) == "MANDATORY";
}

/// GPU canary check: verifies GPU is available when expected
/// Returns true if check passes, false if it fails
/// Call this in GPU tests to catch "running on CPU when expecting GPU" issues
inline bool canary_check() {
#ifdef USE_GPU_OFFLOAD
    const int num_dev = device_count();
    const bool mandatory = is_mandatory_offload();

    if (mandatory && num_dev == 0) {
        std::cerr << "\n========================================\n";
        std::cerr << "GPU CANARY FAILURE\n";
        std::cerr << "========================================\n";
        std::cerr << "OMP_TARGET_OFFLOAD=MANDATORY but no GPU devices available!\n";
        std::cerr << "This means we would silently fall back to CPU.\n";
        std::cerr << "Devices reported by omp_get_num_devices(): " << num_dev << "\n";
        std::cerr << "========================================\n\n";
        return false;
    }

    if (num_dev == 0) {
        std::cout << "[GPU Canary] No devices available (GPU build but CPU execution)\n";
    } else if (!verify_execution()) {
        std::cout << "[GPU Canary] WARNING: Devices reported but verify_execution() failed\n";
    }

    return true;
#else
    // CPU build - always passes
    return true;
#endif
}

//=============================================================================
// GPU Sync Guard for Tests
//=============================================================================

/// Global counter for sync_from_gpu calls (for debugging sync issues)
inline int& sync_call_count() {
    static int count = 0;
    return count;
}

/// Reset sync call counter (call at start of test)
inline void reset_sync_count() {
    sync_call_count() = 0;
}

/// Get number of sync_from_gpu calls since last reset
inline int get_sync_count() {
    return sync_call_count();
}

/// Ensure solver data is on host before reading fields
/// This prevents the common bug of reading stale host data after GPU computation.
/// On CPU builds, this is a no-op.
///
/// Usage:
///     solver.step();
///     gpu::ensure_synced(solver);  // Safe to read solver.velocity() now
///     double div = compute_div(solver.velocity(), mesh);
///
/// @param solver  Any solver object with sync_from_gpu() method
template<typename Solver>
inline void ensure_synced(Solver& solver) {
#ifdef USE_GPU_OFFLOAD
    solver.sync_from_gpu();
    ++sync_call_count();
#else
    (void)solver;  // Suppress unused warning on CPU builds
#endif
}

/// Assert that at least N sync calls were made (for test canaries)
/// Use this to verify that tests properly sync before host reads.
///
/// Usage:
///     gpu::reset_sync_count();
///     // ... test code that should call ensure_synced() ...
///     gpu::assert_synced(1, "divergence computation");  // Fails if no sync
///
inline bool assert_synced(int min_calls, const char* context = "") {
#ifdef USE_GPU_OFFLOAD
    if (sync_call_count() < min_calls) {
        std::cerr << "[GPU Sync Guard] FAILURE: Expected at least " << min_calls
                  << " sync_from_gpu() call(s)";
        if (context && context[0]) {
            std::cerr << " for " << context;
        }
        std::cerr << ", but got " << sync_call_count() << "\n";
        std::cerr << "[GPU Sync Guard] This may cause stale host data to be read!\n";
        return false;
    }
    return true;
#else
    (void)min_calls;
    (void)context;
    return true;  // CPU build - always passes
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

//=============================================================================
// Mesh Factory Functions
//=============================================================================

/// Create a uniform 2D mesh for channel flow (periodic x, walls y)
/// @param Nx  Grid cells in x
/// @param Ny  Grid cells in y
/// @param Lx  Domain length in x (default 2π)
/// @param Ly  Domain height in y (default 2, from -1 to 1)
inline Mesh create_channel_mesh_2d(int Nx, int Ny, double Lx = 2.0 * M_PI, double Ly = 2.0) {
    Mesh mesh;
    mesh.init_uniform(Nx, Ny, 0.0, Lx, -Ly/2, Ly/2);
    return mesh;
}

/// Create a uniform 3D mesh for channel flow (periodic x/z, walls y)
/// @param Nx  Grid cells in x
/// @param Ny  Grid cells in y
/// @param Nz  Grid cells in z
/// @param Lx  Domain length in x (default 2π)
/// @param Ly  Domain height in y (default 2, from -1 to 1)
/// @param Lz  Domain length in z (default π)
inline Mesh create_channel_mesh_3d(int Nx, int Ny, int Nz,
                                    double Lx = 2.0 * M_PI, double Ly = 2.0, double Lz = M_PI) {
    Mesh mesh;
    mesh.init_uniform(Nx, Ny, Nz, 0.0, Lx, -Ly/2, Ly/2, 0.0, Lz);
    return mesh;
}

/// Create a uniform 2D periodic mesh (periodic in both directions)
/// @param N  Grid cells in each direction
/// @param L  Domain size in each direction (default 2π)
inline Mesh create_periodic_mesh_2d(int N, double L = 2.0 * M_PI) {
    Mesh mesh;
    mesh.init_uniform(N, N, 0.0, L, 0.0, L);
    return mesh;
}

/// Create a uniform 3D periodic mesh (periodic in all directions)
/// @param N  Grid cells in each direction
/// @param L  Domain size in each direction (default 2π)
inline Mesh create_periodic_mesh_3d(int N, double L = 2.0 * M_PI) {
    Mesh mesh;
    mesh.init_uniform(N, N, N, 0.0, L, 0.0, L, 0.0, L);
    return mesh;
}

/// Create a uniform 2D unit square mesh [0,1] x [0,1]
/// @param N  Grid cells in each direction
inline Mesh create_unit_square_mesh(int N) {
    Mesh mesh;
    mesh.init_uniform(N, N, 0.0, 1.0, 0.0, 1.0);
    return mesh;
}

/// Create a uniform 3D unit cube mesh [0,1]^3
/// @param N  Grid cells in each direction
inline Mesh create_unit_cube_mesh(int N) {
    Mesh mesh;
    mesh.init_uniform(N, N, N, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    return mesh;
}

/// Create a 2D mesh with custom dimensions
/// @param Nx, Ny  Grid cells
/// @param Lx, Ly  Domain dimensions
inline Mesh create_uniform_mesh_2d(int Nx, int Ny, double Lx, double Ly) {
    Mesh mesh;
    mesh.init_uniform(Nx, Ny, 0.0, Lx, 0.0, Ly);
    return mesh;
}

/// Create a 3D mesh with custom dimensions
/// @param Nx, Ny, Nz  Grid cells
/// @param Lx, Ly, Lz  Domain dimensions
inline Mesh create_uniform_mesh_3d(int Nx, int Ny, int Nz, double Lx, double Ly, double Lz) {
    Mesh mesh;
    mesh.init_uniform(Nx, Ny, Nz, 0.0, Lx, 0.0, Ly, 0.0, Lz);
    return mesh;
}

//=============================================================================
// Config Factory Functions
//=============================================================================

/// Create a laminar flow configuration
/// @param nu  Kinematic viscosity
/// @param dt  Time step
/// @param verbose  Enable verbose output (default false)
inline Config create_laminar_config(double nu, double dt, bool verbose = false) {
    Config config;
    config.nu = nu;
    config.dt = dt;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = verbose;
    return config;
}

/// Create a turbulent flow configuration
/// @param nu  Kinematic viscosity
/// @param dt  Time step
/// @param turb_model  Turbulence model type
/// @param verbose  Enable verbose output (default false)
inline Config create_turbulent_config(double nu, double dt,
                                       TurbulenceModelType turb_model,
                                       bool verbose = false) {
    Config config;
    config.nu = nu;
    config.dt = dt;
    config.turb_model = turb_model;
    config.verbose = verbose;
    return config;
}

/// Create a configuration with adaptive time stepping
/// @param nu  Kinematic viscosity
/// @param CFL_max  Maximum CFL number for stability
/// @param verbose  Enable verbose output (default false)
inline Config create_adaptive_config(double nu, double CFL_max = 0.5, bool verbose = false) {
    Config config;
    config.nu = nu;
    config.dt = 0.001;  // Initial guess, will be overridden by adaptive dt
    config.CFL_max = CFL_max;
    config.adaptive_dt = true;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = verbose;
    return config;
}

/// Create a minimal test configuration (low verbosity, fast)
/// @param nu  Kinematic viscosity (default 0.01)
/// @param dt  Time step (default 0.001)
inline Config create_test_config(double nu = 0.01, double dt = 0.001) {
    Config config;
    config.nu = nu;
    config.dt = dt;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;
    return config;
}

//=============================================================================
// Solver Setup Helper
//=============================================================================

/// Configure a solver with standard settings for testing
/// @param solver  RANSSolver reference to configure
/// @param bc_pattern  Boundary condition pattern
/// @param u_init  Initial u-velocity (default 0.0)
/// @param v_init  Initial v-velocity (default 0.0)
inline void setup_solver_for_test(RANSSolver& solver, BCPattern bc_pattern,
                                   double u_init = 0.0, double v_init = 0.0) {
    solver.set_velocity_bc(create_velocity_bc(bc_pattern));
    solver.initialize_uniform(u_init, v_init);
#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif
}

//=============================================================================
// Unified Test Solver Factory
//=============================================================================

/// Complete test solver bundle: mesh + config + solver in one struct.
/// Eliminates 15-20 lines of boilerplate per test function.
///
/// Usage:
///   auto ts = make_test_solver(32, 32, BCPattern::Channel2D);
///   ts.solver->step();
///   // Access: ts.mesh, ts.config, ts.solver
struct TestSolver {
    Mesh mesh;
    Config config;
    std::unique_ptr<RANSSolver> solver;

    /// Get reference to solver (convenience)
    RANSSolver& operator*() { return *solver; }
    RANSSolver* operator->() { return solver.get(); }
};

/// Create a complete 2D test solver setup in one call.
/// Replaces ~15 lines of mesh/config/solver/BC setup boilerplate.
///
/// @param Nx, Ny  Grid dimensions
/// @param bc  Boundary condition pattern (default Channel2D)
/// @param nu  Kinematic viscosity (default 0.01)
/// @param dt  Time step (default 0.001)
/// @param turb  Turbulence model (default None)
/// @param u_init, v_init  Initial velocity (default 0, 0)
///
/// Example:
///   auto ts = make_test_solver(64, 64);  // Channel2D, laminar
///   ts->step();
///
inline TestSolver make_test_solver(int Nx, int Ny,
                                    BCPattern bc = BCPattern::Channel2D,
                                    double nu = 0.01,
                                    double dt = 0.001,
                                    TurbulenceModelType turb = TurbulenceModelType::None,
                                    double u_init = 0.0,
                                    double v_init = 0.0) {
    TestSolver ts;
    ts.mesh.init_uniform(Nx, Ny, 0.0, 2.0 * M_PI, 0.0, 2.0);
    ts.config.nu = nu;
    ts.config.dt = dt;
    ts.config.turb_model = turb;
    ts.config.verbose = false;
    ts.solver = std::make_unique<RANSSolver>(ts.mesh, ts.config);
    ts.solver->set_velocity_bc(create_velocity_bc(bc));
    ts.solver->initialize_uniform(u_init, v_init);
#ifdef USE_GPU_OFFLOAD
    ts.solver->sync_to_gpu();
#endif
    return ts;
}

/// Create a 2D test solver with custom domain bounds
inline TestSolver make_test_solver_domain(int Nx, int Ny,
                                           double x_min, double x_max,
                                           double y_min, double y_max,
                                           BCPattern bc = BCPattern::Channel2D,
                                           double nu = 0.01,
                                           double dt = 0.001) {
    TestSolver ts;
    ts.mesh.init_uniform(Nx, Ny, x_min, x_max, y_min, y_max);
    ts.config.nu = nu;
    ts.config.dt = dt;
    ts.config.turb_model = TurbulenceModelType::None;
    ts.config.verbose = false;
    ts.solver = std::make_unique<RANSSolver>(ts.mesh, ts.config);
    ts.solver->set_velocity_bc(create_velocity_bc(bc));
    ts.solver->initialize_uniform(0.0, 0.0);
#ifdef USE_GPU_OFFLOAD
    ts.solver->sync_to_gpu();
#endif
    return ts;
}

/// Create a 3D test solver setup
inline TestSolver make_test_solver_3d(int Nx, int Ny, int Nz,
                                       BCPattern bc = BCPattern::Channel3D,
                                       double nu = 0.01,
                                       double dt = 0.001,
                                       TurbulenceModelType turb = TurbulenceModelType::None) {
    TestSolver ts;
    ts.mesh.init_uniform(Nx, Ny, Nz, 0.0, 2.0 * M_PI, 0.0, 2.0, 0.0, M_PI);
    ts.config.nu = nu;
    ts.config.dt = dt;
    ts.config.turb_model = turb;
    ts.config.verbose = false;
    ts.solver = std::make_unique<RANSSolver>(ts.mesh, ts.config);
    ts.solver->set_velocity_bc(create_velocity_bc(bc));
    ts.solver->initialize_uniform(0.0, 0.0);
#ifdef USE_GPU_OFFLOAD
    ts.solver->sync_to_gpu();
#endif
    return ts;
}

/// Create a 3D test solver with custom domain bounds
inline TestSolver make_test_solver_3d_domain(int Nx, int Ny, int Nz,
                                              double x_min, double x_max,
                                              double y_min, double y_max,
                                              double z_min, double z_max,
                                              BCPattern bc = BCPattern::Channel3D,
                                              double nu = 0.01,
                                              double dt = 0.001) {
    TestSolver ts;
    ts.mesh.init_uniform(Nx, Ny, Nz, x_min, x_max, y_min, y_max, z_min, z_max);
    ts.config.nu = nu;
    ts.config.dt = dt;
    ts.config.turb_model = TurbulenceModelType::None;
    ts.config.verbose = false;
    ts.solver = std::make_unique<RANSSolver>(ts.mesh, ts.config);
    ts.solver->set_velocity_bc(create_velocity_bc(bc));
    ts.solver->initialize_uniform(0.0, 0.0);
#ifdef USE_GPU_OFFLOAD
    ts.solver->sync_to_gpu();
#endif
    return ts;
}

//=============================================================================
// Field Norm Utilities
//=============================================================================

/// Compute L2 norm of a scalar field over interior cells
template<typename FieldT, typename MeshT>
inline double l2_norm(const FieldT& f, const MeshT& mesh) {
    double sum = 0.0;
    int count = 0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                sum += f(i, j, k) * f(i, j, k);
                ++count;
            }
        }
    }
    return std::sqrt(sum / std::max(1, count));
}

/// Compute L-infinity (max) norm of a scalar field over interior cells
template<typename FieldT, typename MeshT>
inline double linf_norm(const FieldT& f, const MeshT& mesh) {
    double max_val = 0.0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                max_val = std::max(max_val, std::abs(f(i, j, k)));
            }
        }
    }
    return max_val;
}

/// Compute L2 difference between two scalar fields
template<typename FieldT, typename MeshT>
inline double l2_diff(const FieldT& a, const FieldT& b, const MeshT& mesh) {
    double sum = 0.0;
    int count = 0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double d = a(i, j, k) - b(i, j, k);
                sum += d * d;
                ++count;
            }
        }
    }
    return std::sqrt(sum / std::max(1, count));
}

/// Compute mean value of a scalar field over interior cells
template<typename FieldT, typename MeshT>
inline double mean_value(const FieldT& f, const MeshT& mesh) {
    double sum = 0.0;
    int count = 0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                sum += f(i, j, k);
                ++count;
            }
        }
    }
    return sum / std::max(1, count);
}

/// Remove mean from a scalar field (in-place)
template<typename FieldT, typename MeshT>
inline void remove_mean(FieldT& f, const MeshT& mesh) {
    double m = mean_value(f, mesh);
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                f(i, j, k) -= m;
            }
        }
    }
}

//=============================================================================
// Physics Test Utilities
//=============================================================================

/// Initialize Taylor-Green vortex (MAC grid: u at x-faces, v at y-faces)
inline void init_taylor_green(RANSSolver& solver, const Mesh& mesh) {
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            solver.velocity().u(i, j) = std::sin(mesh.xf[i]) * std::cos(mesh.y(j));
        }
    }
    for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            solver.velocity().v(i, j) = -std::cos(mesh.x(i)) * std::sin(mesh.yf[j]);
        }
    }
}

/// Compute kinetic energy: 0.5 * integral(u^2 + v^2) dx dy
inline double compute_kinetic_energy(const Mesh& mesh, const VectorField& vel) {
    double KE = 0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double u = 0.5 * (vel.u(i, j) + vel.u(i+1, j));
            double v = 0.5 * (vel.v(i, j) + vel.v(i, j+1));
            KE += 0.5 * (u*u + v*v) * mesh.dx * mesh.dy;
        }
    }
    return KE;
}

//=============================================================================
// CI Metrics and JSON Artifact Support
//=============================================================================

// Forward declarations for helper functions used by CIMetrics
inline std::string get_git_sha();
inline std::string get_build_type_string();
inline std::string get_gpu_name();

/// Metrics collected for CI artifact output
/// Designed to be minimal and stable (avoid frequent schema changes)
struct CIMetrics {
    // Build info
    std::string git_sha;
    std::string build_type;       // "CPU" or "GPU"
    std::string precision;        // "double" or "float"
    std::string gpu_name;
    int compute_capability = 0;

    // Test identification
    std::string test_name;
    std::string timestamp;        // ISO 8601 format

    // Timing
    double wall_time_seconds = 0.0;
    int num_iterations = 0;

    // TGV invariants
    double tgv_2d_div_max = 0.0;
    double tgv_2d_E_final = 0.0;
    double tgv_3d_div_max = 0.0;
    double tgv_3d_E_final = 0.0;

    // CPU/GPU comparison (per-field)
    double u_rel_l2 = 0.0;
    double u_rel_linf = 0.0;
    double v_rel_l2 = 0.0;
    double v_rel_linf = 0.0;
    double p_rel_l2 = 0.0;
    double p_rel_linf = 0.0;

    // Optional turbulence fields
    double k_rel_l2 = 0.0;
    double omega_rel_l2 = 0.0;
    double nu_t_rel_l2 = 0.0;

    /// Populate build info from environment
    void populate_from_environment() {
        git_sha = get_git_sha();
        build_type = get_build_type_string();
        gpu_name = get_gpu_name();
#ifdef NNCFD_USE_FLOAT
        precision = "float";
#else
        precision = "double";
#endif
        // Timestamp in ISO 8601 format
        std::time_t now = std::time(nullptr);
        char buf[32];
        std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
        timestamp = buf;
    }

    /// Write metrics to JSON file
    /// @param filename Path to output JSON file (e.g., "artifacts/metrics.json")
    void write_json(const std::string& filename) const {
        std::ofstream ofs(filename);
        if (!ofs) {
            std::cerr << "[CIMetrics] Warning: Could not open " << filename << " for writing\n";
            return;
        }

        ofs << std::scientific << std::setprecision(12);
        ofs << "{\n";
        ofs << "  \"git_sha\": \"" << git_sha << "\",\n";
        ofs << "  \"build_type\": \"" << build_type << "\",\n";
        ofs << "  \"precision\": \"" << precision << "\",\n";
        ofs << "  \"gpu_name\": \"" << gpu_name << "\",\n";
        ofs << "  \"compute_capability\": " << compute_capability << ",\n";
        ofs << "  \"test_name\": \"" << test_name << "\",\n";
        ofs << "  \"timestamp\": \"" << timestamp << "\",\n";
        ofs << "  \"wall_time_seconds\": " << wall_time_seconds << ",\n";
        ofs << "  \"num_iterations\": " << num_iterations << ",\n";
        ofs << "  \"tgv_2d\": {\n";
        ofs << "    \"div_max\": " << tgv_2d_div_max << ",\n";
        ofs << "    \"E_final\": " << tgv_2d_E_final << "\n";
        ofs << "  },\n";
        ofs << "  \"tgv_3d\": {\n";
        ofs << "    \"div_max\": " << tgv_3d_div_max << ",\n";
        ofs << "    \"E_final\": " << tgv_3d_E_final << "\n";
        ofs << "  },\n";
        ofs << "  \"cpu_gpu_diff\": {\n";
        ofs << "    \"u_rel_l2\": " << u_rel_l2 << ",\n";
        ofs << "    \"u_rel_linf\": " << u_rel_linf << ",\n";
        ofs << "    \"v_rel_l2\": " << v_rel_l2 << ",\n";
        ofs << "    \"v_rel_linf\": " << v_rel_linf << ",\n";
        ofs << "    \"p_rel_l2\": " << p_rel_l2 << ",\n";
        ofs << "    \"p_rel_linf\": " << p_rel_linf << ",\n";
        ofs << "    \"k_rel_l2\": " << k_rel_l2 << ",\n";
        ofs << "    \"omega_rel_l2\": " << omega_rel_l2 << ",\n";
        ofs << "    \"nu_t_rel_l2\": " << nu_t_rel_l2 << "\n";
        ofs << "  }\n";
        ofs << "}\n";

        ofs.close();
        std::cout << "[CIMetrics] Wrote metrics to " << filename << "\n";
    }

    /// Print metrics summary to stdout
    void print_summary() const {
        std::cout << "\n--- CI Metrics Summary ---\n";
        std::cout << "  Build: " << build_type << " (" << precision << ")";
        if (!gpu_name.empty()) {
            std::cout << " GPU: " << gpu_name << ", CC " << compute_capability;
        }
        std::cout << "\n";
        if (!git_sha.empty()) {
            std::cout << "  Git SHA: " << git_sha << "\n";
        }
        if (!timestamp.empty()) {
            std::cout << "  Timestamp: " << timestamp << "\n";
        }
        if (wall_time_seconds > 0) {
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "  Wall time: " << wall_time_seconds << "s";
            if (num_iterations > 0) {
                std::cout << " (" << num_iterations << " iters)";
            }
            std::cout << "\n";
        }
        std::cout << std::scientific << std::setprecision(2);
        if (tgv_2d_div_max > 0 || tgv_2d_E_final > 0) {
            std::cout << "  TGV 2D: div_max=" << tgv_2d_div_max << ", E_final=" << tgv_2d_E_final << "\n";
        }
        if (tgv_3d_div_max > 0 || tgv_3d_E_final > 0) {
            std::cout << "  TGV 3D: div_max=" << tgv_3d_div_max << ", E_final=" << tgv_3d_E_final << "\n";
        }
        if (u_rel_l2 > 0) {
            std::cout << "  CPU/GPU u: rel_l2=" << u_rel_l2 << ", rel_linf=" << u_rel_linf << "\n";
        }
        if (v_rel_l2 > 0) {
            std::cout << "  CPU/GPU v: rel_l2=" << v_rel_l2 << ", rel_linf=" << v_rel_linf << "\n";
        }
        if (p_rel_l2 > 0) {
            std::cout << "  CPU/GPU p: rel_l2=" << p_rel_l2 << ", rel_linf=" << p_rel_linf << "\n";
        }
        std::cout << "\n";
    }
};

/// Get git SHA from environment or .git directory
/// Returns empty string if not available
inline std::string get_git_sha() {
    // Try environment variable first (set by CI)
    const char* sha = std::getenv("GIT_SHA");
    if (sha) return std::string(sha);

    // Try GITHUB_SHA (GitHub Actions)
    sha = std::getenv("GITHUB_SHA");
    if (sha) return std::string(sha);

    return "";
}

/// Get build type string
inline std::string get_build_type_string() {
#ifdef USE_GPU_OFFLOAD
    return "GPU";
#else
    return "CPU";
#endif
}

/// Get GPU name if available
inline std::string get_gpu_name() {
#ifdef USE_GPU_OFFLOAD
    // On NVIDIA, could use nvidia-smi or CUDA API
    // For now, return generic
    if (gpu::available()) {
        return "GPU_" + std::to_string(gpu::device_count()) + "_devices";
    }
#endif
    return "";
}

} // namespace test
} // namespace nncfd
