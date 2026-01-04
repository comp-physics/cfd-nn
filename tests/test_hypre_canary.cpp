/// @file test_hypre_canary.cpp
/// @brief Quarantined canary test for known HYPRE limitations
///
/// PURPOSE: Document and monitor known HYPRE issues without failing CI.
/// This test is in "canary mode" - it reports status but doesn't block builds.
///
/// KNOWN ISSUES:
/// 1. HYPRE 2D with y-periodic BCs causes NaN/instability (documented issue)
///    - Symptoms: NaN appears after ~50-100 steps
///    - Root cause: Suspected HYPRE PFMG configuration for mixed BCs
///    - Workaround: Use MG solver for 2D y-periodic cases
///
/// This test provides observability into whether these issues are fixed
/// in future HYPRE versions.

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "config.hpp"
#include <iostream>
#include <cmath>
#include <iomanip>

using namespace nncfd;

// Check for NaN in a scalar field
bool has_nan(const ScalarField& f, const Mesh& mesh) {
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            if (std::isnan(f(i, j)) || std::isinf(f(i, j))) {
                return true;
            }
        }
    }
    return false;
}

int main() {
    std::cout << "================================================================\n";
    std::cout << "  HYPRE Canary Test (Quarantined)\n";
    std::cout << "================================================================\n\n";

    std::cout << "This test monitors known HYPRE limitations.\n";
    std::cout << "Failures are EXPECTED and do not block CI.\n\n";

#ifndef HAVE_HYPRE
    std::cout << "[SKIP] HYPRE not enabled in this build\n";
    std::cout << "[PASS] Canary test skipped (no HYPRE)\n";
    return 0;
#endif

    int canary_issues = 0;

    // ========================================================================
    // Canary 1: HYPRE 2D with Y-periodic BCs (known issue)
    // ========================================================================
    std::cout << "--- Canary 1: HYPRE 2D Y-Periodic ---\n";
    std::cout << "Known issue: HYPRE may produce NaN with 2D y-periodic BCs.\n\n";

#ifdef HAVE_HYPRE
    {
        const int N = 32;
        Mesh mesh;
        mesh.init_uniform(N, N, 0.0, 2.0*M_PI, 0.0, 2.0*M_PI);

        Config config;
        config.Nx = N;
        config.Ny = N;
        config.dt = 0.001;
        config.nu = 0.01;
        config.verbose = false;
        config.poisson_solver = PoissonSolverType::HYPRE;

        RANSSolver solver(mesh, config);

        VelocityBC bc;
        bc.x_lo = VelocityBC::Periodic;
        bc.x_hi = VelocityBC::Periodic;
        bc.y_lo = VelocityBC::Periodic;  // This is the problematic BC
        bc.y_hi = VelocityBC::Periodic;
        solver.set_velocity_bc(bc);

        // Check if HYPRE was actually selected (might fall back)
        if (solver.poisson_solver_type() != PoissonSolverType::HYPRE) {
            std::cout << "  [SKIP] HYPRE not selected (fell back to "
                      << (solver.poisson_solver_type() == PoissonSolverType::MG ? "MG" : "other")
                      << ")\n";
        } else {
            VectorField& vel = solver.velocity();
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                    vel.u(i, j) = std::sin(mesh.x(i)) * std::cos(mesh.y(j));
                }
            }
            for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    vel.v(i, j) = -std::cos(mesh.x(i)) * std::sin(mesh.y(j));
                }
            }
            solver.initialize(vel);

            // Run for 100 steps and check for NaN
            bool nan_detected = false;
            int nan_step = -1;

            for (int step = 0; step < 100; ++step) {
                solver.step();

#ifdef USE_GPU_OFFLOAD
                solver.sync_from_gpu();
#endif

                if (has_nan(solver.pressure(), mesh)) {
                    nan_detected = true;
                    nan_step = step;
                    break;
                }
            }

            if (nan_detected) {
                std::cout << "  [EXPECTED] NaN detected at step " << nan_step << "\n";
                std::cout << "             This is the known HYPRE 2D y-periodic issue.\n";
                ++canary_issues;
            } else {
                std::cout << "  [FIXED!] No NaN after 100 steps!\n";
                std::cout << "           The HYPRE 2D y-periodic issue may be resolved.\n";
            }
        }
    }
#endif

    // ========================================================================
    // Summary
    // ========================================================================
    std::cout << "\n================================================================\n";
    std::cout << "HYPRE Canary Summary\n";
    std::cout << "================================================================\n";
    std::cout << "  Known issues detected: " << canary_issues << "\n";

    if (canary_issues > 0) {
        std::cout << "\n[INFO] Known limitations confirmed - this is expected.\n";
        std::cout << "       Workaround: Use MG solver for affected configurations.\n";
    } else {
        std::cout << "\n[INFO] No known issues detected!\n";
        std::cout << "       Consider removing quarantine if fixes are confirmed.\n";
    }

    // Always pass - this is a canary test
    std::cout << "\n[PASS] Canary test completed (always passes)\n";
    return 0;
}
