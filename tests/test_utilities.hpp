/// @file test_utilities.hpp
/// @brief Common test utilities for CPU/GPU comparison, field validation, and iteration helpers
///
/// This header consolidates duplicated test code from:
///   - test_cpu_gpu_bitwise.cpp (ComparisonResult)
///   - test_poisson_cpu_gpu_3d.cpp (ComparisonResult)
///   - test_hypre_validation.cpp (ComparisonResult)
///   - test_cpu_gpu_consistency.cpp (FieldComparison)

#pragma once

#include <cmath>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>

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
// Tolerance Configuration
//=============================================================================

/// GPU vs CPU tolerance - relaxed for GPU smoke tests
inline double gpu_error_tolerance() {
#ifdef USE_GPU_OFFLOAD
    return 0.05;  // 5% for GPU (fast smoke test)
#else
    return 0.03;  // 3% for CPU (stricter validation)
#endif
}

/// Maximum iterations for steady-state tests
inline int steady_max_iter() {
#ifdef USE_GPU_OFFLOAD
    return 120;   // Fast GPU smoke test
#else
    return 3000;  // Full CPU convergence
#endif
}

/// Poiseuille flow error limit
inline double poiseuille_error_limit() {
#ifdef USE_GPU_OFFLOAD
    return 0.05;  // 5% for GPU (120 iters with analytical init)
#else
    return 0.03;  // 3% for CPU (3000 iters, near steady state)
#endif
}

/// Steady-state residual limit
inline double steady_residual_limit() {
#ifdef USE_GPU_OFFLOAD
    return 5e-3;  // Relaxed for fast GPU test
#else
    return 1e-4;  // Strict for CPU validation
#endif
}

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

/// GPU synchronization helper (no-op on CPU builds)
template<typename Solver>
inline void sync_to_gpu_if_available([[maybe_unused]] Solver& solver) {
#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif
}

/// GPU synchronization from GPU to host
template<typename Solver>
inline void sync_from_gpu_if_available([[maybe_unused]] Solver& solver) {
#ifdef USE_GPU_OFFLOAD
    solver.sync_from_gpu();
#endif
}

//=============================================================================
// Domain Iteration Macros
//=============================================================================

} // namespace test
} // namespace nncfd

/// Iterate over interior cells of a 2D mesh
/// Usage: FOR_INTERIOR_2D(mesh, i, j) { ... }
#define FOR_INTERIOR_2D(mesh, i, j) \
    for (int j = (mesh).j_begin(); j < (mesh).j_end(); ++j) \
    for (int i = (mesh).i_begin(); i < (mesh).i_end(); ++i)

/// Iterate over interior cells of a 3D mesh
/// Usage: FOR_INTERIOR_3D(mesh, i, j, k) { ... }
#define FOR_INTERIOR_3D(mesh, i, j, k) \
    for (int k = (mesh).k_begin(); k < (mesh).k_end(); ++k) \
    for (int j = (mesh).j_begin(); j < (mesh).j_end(); ++j) \
    for (int i = (mesh).i_begin(); i < (mesh).i_end(); ++i)

/// Iterate over all cells including ghosts (2D)
/// Usage: FOR_ALL_2D(mesh, i, j) { ... }
#define FOR_ALL_2D(mesh, i, j) \
    for (int j = 0; j < (mesh).Ny_total(); ++j) \
    for (int i = 0; i < (mesh).Nx_total(); ++i)

/// Iterate over all cells including ghosts (3D)
/// Usage: FOR_ALL_3D(mesh, i, j, k) { ... }
#define FOR_ALL_3D(mesh, i, j, k) \
    for (int k = 0; k < (mesh).Nz_total(); ++k) \
    for (int j = 0; j < (mesh).Ny_total(); ++j) \
    for (int i = 0; i < (mesh).Nx_total(); ++i)

/// Iterate over u-velocity staggered points (2D interior)
#define FOR_U_INTERIOR_2D(mesh, i, j) \
    for (int j = (mesh).j_begin(); j < (mesh).j_end(); ++j) \
    for (int i = (mesh).i_begin(); i <= (mesh).i_end(); ++i)

/// Iterate over v-velocity staggered points (2D interior)
#define FOR_V_INTERIOR_2D(mesh, i, j) \
    for (int j = (mesh).j_begin(); j <= (mesh).j_end(); ++j) \
    for (int i = (mesh).i_begin(); i < (mesh).i_end(); ++i)

/// Iterate over u-velocity staggered points (3D interior)
#define FOR_U_INTERIOR_3D(mesh, i, j, k) \
    for (int k = (mesh).k_begin(); k < (mesh).k_end(); ++k) \
    for (int j = (mesh).j_begin(); j < (mesh).j_end(); ++j) \
    for (int i = (mesh).i_begin(); i <= (mesh).i_end(); ++i)

/// Iterate over v-velocity staggered points (3D interior)
#define FOR_V_INTERIOR_3D(mesh, i, j, k) \
    for (int k = (mesh).k_begin(); k < (mesh).k_end(); ++k) \
    for (int j = (mesh).j_begin(); j <= (mesh).j_end(); ++j) \
    for (int i = (mesh).i_begin(); i < (mesh).i_end(); ++i)

/// Iterate over w-velocity staggered points (3D interior)
#define FOR_W_INTERIOR_3D(mesh, i, j, k) \
    for (int k = (mesh).k_begin(); k <= (mesh).k_end(); ++k) \
    for (int j = (mesh).j_begin(); j < (mesh).j_end(); ++j) \
    for (int i = (mesh).i_begin(); i < (mesh).i_end(); ++i)
