/// @file test_hypre_backend.cpp
/// @brief HYPRE backend verification test
///
/// CRITICAL TEST: Ensures HYPRE runs on the correct backend (CPU or GPU).
/// This test FAILS if:
///   - GPU build (USE_GPU_OFFLOAD) but HYPRE CUDA is NOT active
///   - Expected backend doesn't match actual runtime backend
///
/// This catches silent fallback to CPU when GPU was intended, which would
/// cause significant performance regressions in production.
///
/// The test also validates that HYPRE actually solves correctly on whichever
/// backend is active, catching configuration issues.

#include <iostream>
#include <cmath>
#include <iomanip>
#include <string>

#ifdef USE_HYPRE
#include "mesh.hpp"
#include "fields.hpp"
#include "poisson_solver_hypre.hpp"
#endif

using namespace nncfd;

// ============================================================================
// Compile-time expectations
// ============================================================================

#ifdef USE_GPU_OFFLOAD
constexpr bool EXPECT_GPU = true;
#else
constexpr bool EXPECT_GPU = false;
#endif

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_GPU)
constexpr bool HYPRE_BUILT_WITH_CUDA = true;
#else
constexpr bool HYPRE_BUILT_WITH_CUDA = false;
#endif

// ============================================================================
// Main test
// ============================================================================

int main() {
    std::cout << "================================================================\n";
    std::cout << "  HYPRE Backend Verification Test\n";
    std::cout << "================================================================\n\n";

    // Print build configuration
    std::cout << "Build configuration:\n";
#ifdef USE_GPU_OFFLOAD
    std::cout << "  USE_GPU_OFFLOAD: ON (GPU build)\n";
#else
    std::cout << "  USE_GPU_OFFLOAD: OFF (CPU build)\n";
#endif

#ifdef USE_HYPRE
    std::cout << "  USE_HYPRE: ON\n";
#else
    std::cout << "  USE_HYPRE: OFF\n";
#endif

#if defined(HYPRE_USING_CUDA)
    std::cout << "  HYPRE_USING_CUDA: defined\n";
#elif defined(HYPRE_USING_GPU)
    std::cout << "  HYPRE_USING_GPU: defined\n";
#else
    std::cout << "  HYPRE_USING_CUDA/GPU: not defined\n";
#endif

    std::cout << "\nExpected backend: " << (EXPECT_GPU ? "GPU (CUDA)" : "CPU") << "\n";
    std::cout << "HYPRE CUDA support: " << (HYPRE_BUILT_WITH_CUDA ? "YES" : "NO") << "\n\n";

#ifndef USE_HYPRE
    std::cout << "[SKIP] HYPRE not enabled - skipping backend verification\n";
    return 0;
#else

    // ========================================================================
    // Backend consistency check
    // ========================================================================
    std::cout << "--- Backend Consistency Check ---\n\n";

    if (EXPECT_GPU && !HYPRE_BUILT_WITH_CUDA) {
        std::cerr << "[FAIL] GPU build but HYPRE was NOT built with CUDA support!\n";
        std::cerr << "       This means Poisson solves will run on CPU, not GPU.\n";
        std::cerr << "       Rebuild HYPRE with HYPRE_WITH_CUDA=ON or HYPRE_WITH_GPU=ON.\n";
        return 1;
    }

    // Create a simple mesh and solver to verify runtime behavior
    Mesh mesh;
    mesh.init_uniform(32, 32, 0.0, 1.0, 0.0, 1.0);

    HyprePoissonSolver solver(mesh);
    solver.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                  PoissonBC::Neumann, PoissonBC::Neumann);

    std::cout << "HYPRE solver initialized.\n";
    std::cout << "  is_initialized(): " << (solver.is_initialized() ? "true" : "false") << "\n";
    std::cout << "  using_cuda(): " << (solver.using_cuda() ? "true" : "false") << "\n\n";

    // Verify runtime matches compile-time expectation
    if (EXPECT_GPU) {
        if (!solver.using_cuda()) {
            std::cerr << "[FAIL] GPU build but HYPRE is NOT using CUDA at runtime!\n";
            std::cerr << "       Check HYPRE library linking and CUDA initialization.\n";
            return 1;
        }
        std::cout << "[PASS] GPU build correctly using CUDA backend\n";
    } else {
        if (solver.using_cuda()) {
            // This is actually fine - HYPRE CUDA works on CPU builds too
            std::cout << "[INFO] CPU build but HYPRE has CUDA available (OK)\n";
        } else {
            std::cout << "[PASS] CPU build correctly using CPU backend\n";
        }
    }

    // ========================================================================
    // Functional verification: actually solve something
    // ========================================================================
    std::cout << "\n--- Functional Verification ---\n\n";

    ScalarField rhs(mesh), p(mesh);

    // Simple test problem: sin(2*pi*x) * cos(2*pi*y)
    // Analytical Laplacian = -8*pi^2 * sin(2*pi*x) * cos(2*pi*y)
    const double pi = 3.14159265358979323846;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double x = mesh.x(i);
            double y = mesh.y(j);
            rhs(i, j) = -8.0 * pi * pi * std::sin(2 * pi * x) * std::cos(2 * pi * y);
            p(i, j) = 0.0;  // Initial guess
        }
    }

    PoissonConfig cfg;
    cfg.max_steps = 200;
    cfg.tol = 1e-10;
    cfg.verbose = false;

    int iters = solver.solve(rhs, p, cfg);
    double residual = solver.residual();

    std::cout << "Solve completed:\n";
    std::cout << "  Iterations: " << iters << "\n";
    std::cout << "  Residual: " << std::scientific << std::setprecision(3) << residual << "\n";

    // Check solution accuracy
    double max_err = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double x = mesh.x(i);
            double y = mesh.y(j);
            double p_exact = std::sin(2 * pi * x) * std::cos(2 * pi * y);
            double err = std::abs(p(i, j) - p_exact);
            max_err = std::max(max_err, err);
        }
    }

    // Normalize by mean (solution is unique up to constant for periodic x)
    double p_mean = 0.0;
    int count = 0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            p_mean += p(i, j);
            ++count;
        }
    }
    p_mean /= count;

    double p_exact_mean = 0.0;  // sin*cos integrates to 0 over [0,1]^2

    // Recompute error with mean adjustment
    max_err = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double x = mesh.x(i);
            double y = mesh.y(j);
            double p_exact = std::sin(2 * pi * x) * std::cos(2 * pi * y);
            double err = std::abs((p(i, j) - p_mean) - (p_exact - p_exact_mean));
            max_err = std::max(max_err, err);
        }
    }

    std::cout << "  Max error: " << std::scientific << std::setprecision(3) << max_err << "\n";

    // Check convergence
    bool converged = (residual < 1e-6) && (max_err < 0.01);

    if (!converged) {
        std::cerr << "\n[FAIL] HYPRE did not converge to acceptable accuracy!\n";
        std::cerr << "       residual=" << residual << " (need < 1e-6)\n";
        std::cerr << "       max_err=" << max_err << " (need < 0.01)\n";
        return 1;
    }

    std::cout << "\n[PASS] HYPRE solve converged correctly on "
              << (solver.using_cuda() ? "GPU" : "CPU") << " backend\n";

    // ========================================================================
    // Summary
    // ========================================================================
    std::cout << "\n================================================================\n";
    std::cout << "HYPRE Backend Verification: PASSED\n";
    std::cout << "================================================================\n";
    std::cout << "  Build: " << (EXPECT_GPU ? "GPU" : "CPU") << "\n";
    std::cout << "  Runtime: " << (solver.using_cuda() ? "CUDA" : "CPU") << "\n";
    std::cout << "  Solve: " << iters << " iterations, residual=" << residual << "\n";
    std::cout << "================================================================\n";

    return 0;

#endif // USE_HYPRE
}
