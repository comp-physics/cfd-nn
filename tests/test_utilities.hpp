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

//=============================================================================
// GPU/CPU Test Utilities
//=============================================================================

namespace nncfd {
namespace test {

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

} // namespace test
} // namespace nncfd
