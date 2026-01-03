/// @file test_poisson_selection.cpp
/// @brief Solver-selection state machine unit test
///
/// CRITICAL TEST: Validates the Poisson solver selection logic and fallback chains.
/// Without this test, selection logic can drift silently - a change might make
/// FFT unavailable and accidentally skip to MG instead of FFT1D.
///
/// Tests 10+ scenarios covering:
///   - Auto-selection priority: FFT → FFT1D → HYPRE → MG
///   - BC-triggered reselection after set_velocity_bc()
///   - HYPRE 2D y-periodic GPU fallback
///   - Explicit solver requests with unavailable solvers
///   - 2D vs 3D dimension constraints

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "config.hpp"
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <functional>

using namespace nncfd;

// Test case structure
struct SelectionTestCase {
    std::string name;
    int Nx, Ny, Nz;  // Grid size (Nz=1 for 2D)
    PoissonSolverType requested;
    VelocityBC::Type x_lo, x_hi, y_lo, y_hi, z_lo, z_hi;
    PoissonSolverType expected_cpu;   // Expected on CPU build
    PoissonSolverType expected_gpu;   // Expected on GPU build
    std::string expected_message;     // Optional: grep-able log message
};

const char* solver_name(PoissonSolverType t) {
    switch (t) {
        case PoissonSolverType::Auto: return "Auto";
        case PoissonSolverType::FFT: return "FFT";
        case PoissonSolverType::FFT1D: return "FFT1D";
        case PoissonSolverType::HYPRE: return "HYPRE";
        case PoissonSolverType::MG: return "MG";
        default: return "Unknown";
    }
}

// Run a selection test and return pass/fail
bool run_selection_test(const SelectionTestCase& tc, bool& is_gpu) {
    // Determine if this is a GPU build
#ifdef USE_GPU_OFFLOAD
    is_gpu = true;
#else
    is_gpu = false;
#endif

    PoissonSolverType expected = is_gpu ? tc.expected_gpu : tc.expected_cpu;

    // Create mesh
    Mesh mesh;
    if (tc.Nz == 1) {
        mesh.init_uniform(tc.Nx, tc.Ny, 0.0, 2.0*M_PI, 0.0, 2.0);
    } else {
        mesh.init_uniform(tc.Nx, tc.Ny, tc.Nz, 0.0, 2.0*M_PI, 0.0, 2.0, 0.0, 2.0*M_PI);
    }

    // Create config
    Config config;
    config.Nx = tc.Nx;
    config.Ny = tc.Ny;
    config.Nz = tc.Nz;
    config.dt = 0.001;
    config.max_iter = 1;
    config.nu = 1.0;
    config.poisson_solver = tc.requested;

    // Create solver
    RANSSolver solver(mesh, config);

    // Set BCs
    VelocityBC bc;
    bc.x_lo = tc.x_lo;
    bc.x_hi = tc.x_hi;
    bc.y_lo = tc.y_lo;
    bc.y_hi = tc.y_hi;
    bc.z_lo = tc.z_lo;
    bc.z_hi = tc.z_hi;
    solver.set_velocity_bc(bc);

    // Check selection
    PoissonSolverType actual = solver.poisson_solver_type();

    bool passed = (actual == expected);

    std::cout << "  " << tc.name << ": "
              << (passed ? "[PASS]" : "[FAIL]")
              << " expected=" << solver_name(expected)
              << " actual=" << solver_name(actual);

    if (!passed) {
        std::cout << " (requested=" << solver_name(tc.requested) << ")";
    }
    std::cout << "\n";

    return passed;
}

int main() {
    std::cout << "================================================================\n";
    std::cout << "  Poisson Solver Selection State Machine Test\n";
    std::cout << "================================================================\n\n";

    // Build info
#ifdef USE_GPU_OFFLOAD
    std::cout << "Build: GPU (USE_GPU_OFFLOAD=ON)\n";
#else
    std::cout << "Build: CPU (USE_GPU_OFFLOAD=OFF)\n";
#endif
#ifdef USE_HYPRE
    std::cout << "HYPRE: enabled\n";
#else
    std::cout << "HYPRE: disabled\n";
#endif
#ifdef USE_FFT_POISSON
    std::cout << "FFT: enabled\n";
#else
    std::cout << "FFT: disabled\n";
#endif
    std::cout << "\n";

    // ========================================================================
    // Define test cases
    // ========================================================================

    std::vector<SelectionTestCase> tests;

    // --- 3D Cases: FFT Selection ---
    // Test 1: Channel flow (periodic x,z, walls y) -> FFT on GPU, HYPRE/MG on CPU
    tests.push_back({
        "3D_channel_auto",
        32, 32, 32,
        PoissonSolverType::Auto,
        VelocityBC::Periodic, VelocityBC::Periodic,  // x
        VelocityBC::NoSlip, VelocityBC::NoSlip,      // y
        VelocityBC::Periodic, VelocityBC::Periodic,  // z
#ifdef USE_FFT_POISSON
        // CPU: no FFT, fall to HYPRE or MG
#ifdef USE_HYPRE
        PoissonSolverType::HYPRE,
#else
        PoissonSolverType::MG,
#endif
        PoissonSolverType::FFT,  // GPU: FFT available
#else
#ifdef USE_HYPRE
        PoissonSolverType::HYPRE,
        PoissonSolverType::HYPRE,
#else
        PoissonSolverType::MG,
        PoissonSolverType::MG,
#endif
#endif
        ""
    });

    // Test 2: Duct flow (periodic x, walls yz) -> FFT1D on GPU
    tests.push_back({
        "3D_duct_x_periodic_auto",
        32, 32, 32,
        PoissonSolverType::Auto,
        VelocityBC::Periodic, VelocityBC::Periodic,  // x: periodic
        VelocityBC::NoSlip, VelocityBC::NoSlip,          // y: walls
        VelocityBC::NoSlip, VelocityBC::NoSlip,          // z: walls
#ifdef USE_FFT_POISSON
#ifdef USE_HYPRE
        PoissonSolverType::HYPRE,  // CPU: no FFT1D
#else
        PoissonSolverType::MG,
#endif
        PoissonSolverType::FFT1D,  // GPU: FFT1D
#else
#ifdef USE_HYPRE
        PoissonSolverType::HYPRE,
        PoissonSolverType::HYPRE,
#else
        PoissonSolverType::MG,
        PoissonSolverType::MG,
#endif
#endif
        ""
    });

    // Test 3: Alternate duct (walls x, walls y, periodic z)
    tests.push_back({
        "3D_duct_z_periodic_auto",
        32, 32, 32,
        PoissonSolverType::Auto,
        VelocityBC::NoSlip, VelocityBC::NoSlip,              // x: walls
        VelocityBC::NoSlip, VelocityBC::NoSlip,              // y: walls
        VelocityBC::Periodic, VelocityBC::Periodic,      // z: periodic
#ifdef USE_FFT_POISSON
#ifdef USE_HYPRE
        PoissonSolverType::HYPRE,
#else
        PoissonSolverType::MG,
#endif
        PoissonSolverType::FFT1D,
#else
#ifdef USE_HYPRE
        PoissonSolverType::HYPRE,
        PoissonSolverType::HYPRE,
#else
        PoissonSolverType::MG,
        PoissonSolverType::MG,
#endif
#endif
        ""
    });

    // Test 4: All walls (no periodicity) -> HYPRE or MG
    tests.push_back({
        "3D_cavity_all_walls_auto",
        32, 32, 32,
        PoissonSolverType::Auto,
        VelocityBC::NoSlip, VelocityBC::NoSlip,
        VelocityBC::NoSlip, VelocityBC::NoSlip,
        VelocityBC::NoSlip, VelocityBC::NoSlip,
#ifdef USE_HYPRE
        PoissonSolverType::HYPRE,
        PoissonSolverType::HYPRE,
#else
        PoissonSolverType::MG,
        PoissonSolverType::MG,
#endif
        ""
    });

    // Test 5: Explicit FFT request with incompatible BCs -> fallback
    tests.push_back({
        "3D_explicit_fft_incompatible",
        32, 32, 32,
        PoissonSolverType::FFT,  // Explicitly request FFT
        VelocityBC::Periodic, VelocityBC::Periodic,  // x: periodic
        VelocityBC::NoSlip, VelocityBC::NoSlip,          // y: walls
        VelocityBC::NoSlip, VelocityBC::NoSlip,          // z: walls (need periodic for FFT!)
#ifdef USE_FFT_POISSON
#ifdef USE_HYPRE
        PoissonSolverType::HYPRE,  // CPU fallback
#else
        PoissonSolverType::MG,
#endif
        PoissonSolverType::FFT1D,  // GPU: falls back to FFT1D
#else
        PoissonSolverType::MG,
        PoissonSolverType::MG,
#endif
        ""
    });

    // --- 2D Cases ---
    // Test 6: 2D channel (periodic x, walls y)
    tests.push_back({
        "2D_channel_auto",
        64, 64, 1,
        PoissonSolverType::Auto,
        VelocityBC::Periodic, VelocityBC::Periodic,
        VelocityBC::NoSlip, VelocityBC::NoSlip,
        VelocityBC::Periodic, VelocityBC::Periodic,  // ignored for 2D
#ifdef USE_HYPRE
        PoissonSolverType::HYPRE,
        PoissonSolverType::HYPRE,
#else
        PoissonSolverType::MG,
        PoissonSolverType::MG,
#endif
        ""
    });

    // Test 7: 2D y-periodic on GPU -> HYPRE fallback to MG
    tests.push_back({
        "2D_y_periodic_hypre_gpu_fallback",
        64, 64, 1,
        PoissonSolverType::HYPRE,  // Explicitly request HYPRE
        VelocityBC::NoSlip, VelocityBC::NoSlip,
        VelocityBC::Periodic, VelocityBC::Periodic,  // y: periodic -> triggers fallback
        VelocityBC::Periodic, VelocityBC::Periodic,
#ifdef USE_HYPRE
        PoissonSolverType::HYPRE,  // CPU: HYPRE works
        PoissonSolverType::MG,     // GPU: fallback to MG (CUDA instability)
#else
        PoissonSolverType::MG,
        PoissonSolverType::MG,
#endif
        "[Poisson] HYPRE->MG fallback: 2D y-periodic + GPU"
    });

    // Test 8: 2D fully periodic (x and y)
    tests.push_back({
        "2D_fully_periodic_auto",
        64, 64, 1,
        PoissonSolverType::Auto,
        VelocityBC::Periodic, VelocityBC::Periodic,
        VelocityBC::Periodic, VelocityBC::Periodic,
        VelocityBC::Periodic, VelocityBC::Periodic,
#ifdef USE_HYPRE
        PoissonSolverType::HYPRE,  // CPU: HYPRE
        PoissonSolverType::MG,     // GPU: y-periodic triggers fallback
#else
        PoissonSolverType::MG,
        PoissonSolverType::MG,
#endif
        ""
    });

    // Test 9: Explicit MG request (should always work)
    tests.push_back({
        "3D_explicit_mg_request",
        32, 32, 32,
        PoissonSolverType::MG,
        VelocityBC::Periodic, VelocityBC::Periodic,
        VelocityBC::NoSlip, VelocityBC::NoSlip,
        VelocityBC::Periodic, VelocityBC::Periodic,
        PoissonSolverType::MG,
        PoissonSolverType::MG,
        ""
    });

    // Test 10: Explicit HYPRE request
    tests.push_back({
        "3D_explicit_hypre_request",
        32, 32, 32,
        PoissonSolverType::HYPRE,
        VelocityBC::Periodic, VelocityBC::Periodic,
        VelocityBC::NoSlip, VelocityBC::NoSlip,
        VelocityBC::Periodic, VelocityBC::Periodic,
#ifdef USE_HYPRE
        PoissonSolverType::HYPRE,
        PoissonSolverType::HYPRE,
#else
        PoissonSolverType::MG,  // Fallback when HYPRE not built
        PoissonSolverType::MG,
#endif
        ""
    });

    // ========================================================================
    // Run tests
    // ========================================================================

    std::cout << "--- Running " << tests.size() << " selection tests ---\n\n";

    int passed = 0;
    int failed = 0;
    bool is_gpu = false;

    for (const auto& tc : tests) {
        if (run_selection_test(tc, is_gpu)) {
            ++passed;
        } else {
            ++failed;
        }
    }

    // ========================================================================
    // Summary
    // ========================================================================

    std::cout << "\n================================================================\n";
    std::cout << "Selection Test Summary (" << (is_gpu ? "GPU" : "CPU") << " build)\n";
    std::cout << "================================================================\n";
    std::cout << "  Passed: " << passed << "/" << tests.size() << "\n";
    std::cout << "  Failed: " << failed << "/" << tests.size() << "\n";

    if (failed == 0) {
        std::cout << "\n[PASS] All solver selection tests passed\n";
        return 0;
    } else {
        std::cout << "\n[FAIL] " << failed << " solver selection test(s) failed\n";
        std::cout << "       This indicates selection logic has drifted!\n";
        return 1;
    }
}
