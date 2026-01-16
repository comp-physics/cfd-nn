/// @file test_solver_selection.cpp
/// @brief Solver selection matrix test
///
/// Verifies that auto-selection logic picks the correct Poisson solver for
/// different mesh/BC combinations. This catches "someone changed defaults"
/// regressions that can silently degrade performance or correctness.
///
/// Test cases:
///   - 2D periodic x + wall y -> expect FFT2D (if available) or MG
///   - 2D nonperiodic -> expect MG
///   - 3D singly periodic (x periodic, y/z walls) -> expect FFT1D (if available) or MG
///   - 3D fully nonperiodic -> expect MG or HYPRE
///   - Explicit override honored for each solver type

#include "test_harness.hpp"
#include "mesh.hpp"
#include "solver.hpp"
#include "config.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

using namespace nncfd;
using namespace nncfd::test::harness;

// ============================================================================
// Test case definition
// ============================================================================

struct SolverSelectCase {
    std::string name;
    int Nx, Ny, Nz;
    VelocityBC::Type x_bc, y_bc, z_bc;
    PoissonSolverType requested;  // What we ask for (Auto or explicit)
    std::string expected_cpu;     // Expected on CPU build
    std::string expected_gpu;     // Expected on GPU build
};

// Helper to get solver name string from PoissonSolverType enum
static std::string get_solver_name(const RANSSolver& solver) {
    switch (solver.poisson_solver_type()) {
        case PoissonSolverType::HYPRE: return "HYPRE";
        case PoissonSolverType::FFT:   return "FFT";
        case PoissonSolverType::FFT1D: return "FFT1D";
        case PoissonSolverType::FFT2D: return "FFT2D";
        case PoissonSolverType::MG:    return "MG";
        case PoissonSolverType::Auto:  return "Auto";  // Should not happen after init
        default:                       return "Unknown";
    }
}

// ============================================================================
// Run a single selection test
// ============================================================================

static bool run_selection_test(const SolverSelectCase& tc) {
    // Create mesh
    Mesh mesh;
    if (tc.Nz == 1) {
        mesh.init_uniform(tc.Nx, tc.Ny, 0.0, 2.0*M_PI, 0.0, 2.0);
    } else {
        mesh.init_uniform(tc.Nx, tc.Ny, tc.Nz, 0.0, 2.0*M_PI, 0.0, 2.0, 0.0, 2.0*M_PI);
    }

    // Create config
    Config config;
    config.nu = 0.01;
    config.dt = 0.001;
    config.poisson_solver = tc.requested;
    config.verbose = false;

    // Create solver
    RANSSolver solver(mesh, config);

    // Set BCs
    VelocityBC bc;
    bc.x_lo = tc.x_bc; bc.x_hi = tc.x_bc;
    bc.y_lo = tc.y_bc; bc.y_hi = tc.y_bc;
    bc.z_lo = tc.z_bc; bc.z_hi = tc.z_bc;
    solver.set_velocity_bc(bc);

    // Get selected solver
    std::string selected = get_solver_name(solver);

    // Determine expected based on build type
#ifdef USE_GPU_OFFLOAD
    std::string expected = tc.expected_gpu;
    std::string build = "GPU";
#else
    std::string expected = tc.expected_cpu;
    std::string build = "CPU";
#endif

    // Check if selected matches expected
    // Allow multiple valid options separated by |
    bool passed = false;
    std::string expected_options = expected;
    size_t pos = 0;
    while ((pos = expected_options.find('|')) != std::string::npos) {
        std::string opt = expected_options.substr(0, pos);
        if (selected == opt) {
            passed = true;
            break;
        }
        expected_options = expected_options.substr(pos + 1);
    }
    if (!passed && selected == expected_options) {
        passed = true;
    }

    // Print result
    std::cout << "  " << std::left << std::setw(30) << tc.name
              << " selected=" << std::setw(6) << selected
              << " expected=" << std::setw(12) << expected
              << " [" << (passed ? "PASS" : "FAIL") << "]\n";

    // Emit QoI for trend tracking
    std::cout << "QOI_JSON: {\"test\":\"solver_select\""
              << ",\"case\":\"" << tc.name << "\""
              << ",\"selected\":\"" << selected << "\""
              << ",\"expected\":\"" << expected << "\""
              << ",\"build\":\"" << build << "\""
              << "}\n";

    return passed;
}

// ============================================================================
// Main test
// ============================================================================

void test_solver_selection_matrix() {
    std::cout << "\n--- Solver Selection Matrix ---\n\n";

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

    // Define test cases
    // Format: name, Nx, Ny, Nz, x_bc, y_bc, z_bc, requested, expected_cpu, expected_gpu
    std::vector<SolverSelectCase> cases;

    // Auto-selection cases (the important ones)

    // 2D channel: periodic x, wall y -> FFT2D on GPU, MG on CPU
    cases.push_back({
        "2D_channel_auto",
        64, 64, 1,
        VelocityBC::Periodic, VelocityBC::NoSlip, VelocityBC::Periodic,
        PoissonSolverType::Auto,
        "MG",          // CPU: no FFT
        "FFT2D|MG"     // GPU: FFT2D if available, else MG
    });

    // 2D box: all walls -> MG always
    cases.push_back({
        "2D_box_auto",
        64, 64, 1,
        VelocityBC::NoSlip, VelocityBC::NoSlip, VelocityBC::NoSlip,
        PoissonSolverType::Auto,
        "MG",
        "MG"
    });

    // 3D channel: periodic x/z, wall y -> FFT (2D FFT in x-z) on GPU, MG on CPU
    // Note: FFT (not FFT1D) is used when BOTH x AND z are periodic
    cases.push_back({
        "3D_channel_auto",
        32, 32, 32,
        VelocityBC::Periodic, VelocityBC::NoSlip, VelocityBC::Periodic,
        PoissonSolverType::Auto,
        "MG|HYPRE",        // CPU: MG or HYPRE
        "FFT|FFT1D|MG"     // GPU: FFT (2D) if x+z periodic, else FFT1D, else MG
    });

    // 3D duct: periodic x, wall y/z -> FFT1D on GPU, MG on CPU
    cases.push_back({
        "3D_duct_auto",
        32, 32, 32,
        VelocityBC::Periodic, VelocityBC::NoSlip, VelocityBC::NoSlip,
        PoissonSolverType::Auto,
        "MG|HYPRE",    // CPU: MG or HYPRE
        "FFT1D|MG"     // GPU: FFT1D if available, else MG
    });

    // 3D box: all walls -> MG or HYPRE
    cases.push_back({
        "3D_box_auto",
        32, 32, 32,
        VelocityBC::NoSlip, VelocityBC::NoSlip, VelocityBC::NoSlip,
        PoissonSolverType::Auto,
        "MG|HYPRE",
        "MG|HYPRE"
    });

    // 3D fully periodic -> FFT on GPU, MG on CPU
    cases.push_back({
        "3D_periodic_auto",
        32, 32, 32,
        VelocityBC::Periodic, VelocityBC::Periodic, VelocityBC::Periodic,
        PoissonSolverType::Auto,
        "MG",          // CPU: MG (no FFT)
        "FFT|MG"       // GPU: FFT if available
    });

    // Explicit override cases - verify we honor explicit requests

    // Force MG regardless of BCs
    cases.push_back({
        "3D_channel_force_MG",
        32, 32, 32,
        VelocityBC::Periodic, VelocityBC::NoSlip, VelocityBC::Periodic,
        PoissonSolverType::MG,
        "MG",
        "MG"
    });

#ifdef USE_HYPRE
    // Force HYPRE if available
    cases.push_back({
        "3D_channel_force_HYPRE",
        32, 32, 32,
        VelocityBC::Periodic, VelocityBC::NoSlip, VelocityBC::Periodic,
        PoissonSolverType::HYPRE,
        "HYPRE",
        "HYPRE"
    });
#endif

    // Run all tests
    int passed = 0, failed = 0;
    for (const auto& tc : cases) {
        if (run_selection_test(tc)) {
            ++passed;
        } else {
            ++failed;
        }
    }

    std::cout << "\n";
    record("Solver selection matrix", failed == 0,
           "(" + std::to_string(passed) + "/" + std::to_string(passed + failed) + " cases)");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    return run("Solver Selection Tests", []() {
        test_solver_selection_matrix();
    });
}
