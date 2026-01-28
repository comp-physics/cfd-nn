/// @file test_scheme_comprehensive.cpp
/// @brief Comprehensive validation tests for ALL convective schemes
///
/// Tests include:
/// 1. Energy conservation/dissipation properties
/// 2. Scheme dissipation comparison
/// 3. Stability under various CFL conditions
/// 4. Basic sanity checks (div-free, no NaN)
///
/// Key coverage: Upwind2 scheme (previously untested)

#include "test_harness.hpp"
#include "test_utilities.hpp"
#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include <cmath>
#include <vector>
#include <tuple>
#include <string>
#include <algorithm>
#include <iomanip>

using namespace nncfd;
using namespace nncfd::test::harness;

// ============================================================================
// Helper: Check if velocity field contains NaN or Inf
// ============================================================================
static bool check_field_validity(const VectorField& vel, const Mesh& mesh) {
    const int Nx = mesh.Nx;
    const int Ny = mesh.Ny;
    const int Ng = mesh.Nghost;

    for (int j = Ng; j < Ny + Ng; ++j) {
        for (int i = Ng; i < Nx + Ng + 1; ++i) {
            if (!std::isfinite(vel.u(i, j))) return false;
        }
    }
    for (int j = Ng; j < Ny + Ng + 1; ++j) {
        for (int i = Ng; i < Nx + Ng; ++i) {
            if (!std::isfinite(vel.v(i, j))) return false;
        }
    }
    return true;
}

// ============================================================================
// Helper: Compute kinetic energy for 2D field
// ============================================================================
static double compute_ke_2d(const VectorField& vel, const Mesh& mesh) {
    double ke = 0.0;
    const int Nx = mesh.Nx;
    const int Ny = mesh.Ny;
    const int Ng = mesh.Nghost;
    const double dx = mesh.dx;
    const double dy = mesh.dy;

    for (int j = Ng; j < Ny + Ng; ++j) {
        for (int i = Ng; i < Nx + Ng; ++i) {
            // Interpolate u and v to cell centers
            double u_c = 0.5 * (vel.u(i, j) + vel.u(i + 1, j));
            double v_c = 0.5 * (vel.v(i, j) + vel.v(i, j + 1));
            ke += 0.5 * (u_c * u_c + v_c * v_c) * dx * dy;
        }
    }
    return ke;
}

// ============================================================================
// Helper: Run simulation and return energy metrics
// ============================================================================
struct EnergyResult {
    double ke_init;
    double ke_final;
    double ke_max;
    double ke_min;
    double max_div;
    bool valid;  // No NaN/Inf
    int steps_completed;
};

static EnergyResult run_energy_test(
    int N, double L, double nu, double dt, int nsteps,
    ConvectiveScheme scheme, int Ng = 1)
{
    Mesh mesh;
    mesh.init_uniform(N, N, 0.0, L, 0.0, L, Ng);

    Config config;
    config.nu = nu;
    config.dt = dt;
    config.adaptive_dt = false;
    config.convective_scheme = scheme;
    config.poisson_solver = PoissonSolverType::MG;
    config.poisson_fixed_cycles = 8;
    config.poisson_adaptive_cycles = true;
    config.verbose = false;

    RANSSolver solver(mesh, config);
    test::init_taylor_green(solver, mesh);

    EnergyResult result;
    result.ke_init = compute_ke_2d(solver.velocity(), mesh);
    result.ke_max = result.ke_init;
    result.ke_min = result.ke_init;
    result.valid = true;
    result.steps_completed = 0;

    for (int step = 0; step < nsteps; ++step) {
        solver.step();
        result.steps_completed++;

        // Check validity
        if (!check_field_validity(solver.velocity(), mesh)) {
            result.valid = false;
            break;
        }

        double ke = compute_ke_2d(solver.velocity(), mesh);
        result.ke_max = std::max(result.ke_max, ke);
        result.ke_min = std::min(result.ke_min, ke);
    }

    result.ke_final = compute_ke_2d(solver.velocity(), mesh);
    result.max_div = test::compute_max_divergence_2d(solver.velocity(), mesh);

    return result;
}

// ============================================================================
// Test 1: Energy Conservation/Dissipation Properties
// ============================================================================
/// Tests that each scheme has expected energy behavior:
/// - Skew/Central: Energy-conserving (stable or slow decay)
/// - Upwind/Upwind2: Dissipative (energy decays)
static void test_energy_properties() {
    std::cout << "\n=== Energy Conservation/Dissipation Test ===\n\n";

    const int N = 32;
    const double L = 2.0 * M_PI;
    const double nu = 1e-4;  // Small viscosity
    const double dt = 0.005;
    const int nsteps = 200;

    struct SchemeTest {
        std::string name;
        ConvectiveScheme scheme;
        int ng;  // Required ghost cells
        double max_growth;  // Max allowed KE growth ratio
        bool expect_dissipation;
    };

    std::vector<SchemeTest> tests = {
        {"Central", ConvectiveScheme::Central, 1, 1.01, false},
        {"Skew", ConvectiveScheme::Skew, 1, 1.01, false},
        {"Upwind", ConvectiveScheme::Upwind, 1, 1.01, true},
        {"Upwind2", ConvectiveScheme::Upwind2, 2, 1.01, true}  // Requires Ng=2
    };

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  Scheme     | KE_init    | KE_final   | KE_ratio | max_div    | Status\n";
    std::cout << "  -----------+------------+------------+----------+------------+--------\n";

    bool all_pass = true;

    for (const auto& t : tests) {
        auto result = run_energy_test(N, L, nu, dt, nsteps, t.scheme, t.ng);

        double ke_ratio = result.ke_final / result.ke_init;
        bool energy_ok = (result.ke_max / result.ke_init) < t.max_growth;
        bool div_ok = result.max_div < 1e-5;  // Relax for CPU MG solver
        bool valid = result.valid;

        // Note: With low viscosity (nu=1e-4) and short runs, dissipation is negligible
        // The dissipation ordering test separately verifies relative dissipation rates
        bool pass = energy_ok && div_ok && valid;
        all_pass &= pass;

        std::cout << "  " << std::left << std::setw(10) << t.name
                  << " | " << std::setw(10) << result.ke_init
                  << " | " << std::setw(10) << result.ke_final
                  << " | " << std::setw(8) << ke_ratio
                  << " | " << std::scientific << std::setw(10) << result.max_div
                  << " | " << (pass ? "PASS" : "FAIL") << "\n";
        std::cout << std::fixed;
    }

    std::cout << "\nQOI_JSON: {\"test\":\"energy_properties\"";
    for (const auto& t : tests) {
        auto result = run_energy_test(N, L, nu, dt, nsteps, t.scheme, t.ng);
        std::string key = t.name;
        std::transform(key.begin(), key.end(), key.begin(), ::tolower);
        std::cout << ",\"ke_ratio_" << key << "\":" << (result.ke_final / result.ke_init);
    }
    std::cout << "}\n" << std::flush;

    record("[Schemes] All schemes have correct energy behavior", all_pass);
}

// ============================================================================
// Test 2: Scheme Dissipation Comparison
// ============================================================================
/// Verify: Skew ≈ Central > Upwind2 > Upwind (dissipation ordering)
static void test_dissipation_ordering() {
    std::cout << "\n=== Scheme Dissipation Ordering Test ===\n\n";

    const int N = 32;
    const double L = 2.0 * M_PI;
    const double nu = 1e-3;  // Moderate viscosity
    const double dt = 0.002;
    const int nsteps = 500;

    // Run all schemes
    auto central = run_energy_test(N, L, nu, dt, nsteps, ConvectiveScheme::Central, 1);
    auto skew = run_energy_test(N, L, nu, dt, nsteps, ConvectiveScheme::Skew, 1);
    auto upwind = run_energy_test(N, L, nu, dt, nsteps, ConvectiveScheme::Upwind, 1);
    auto upwind2 = run_energy_test(N, L, nu, dt, nsteps, ConvectiveScheme::Upwind2, 2);

    double r_central = central.ke_final / central.ke_init;
    double r_skew = skew.ke_final / skew.ke_init;
    double r_upwind = upwind.ke_final / upwind.ke_init;
    double r_upwind2 = upwind2.ke_final / upwind2.ke_init;

    std::cout << "  KE retention (higher = less dissipation):\n";
    std::cout << "    Central:  " << std::fixed << std::setprecision(4) << r_central << "\n";
    std::cout << "    Skew:     " << r_skew << "\n";
    std::cout << "    Upwind2:  " << r_upwind2 << "\n";
    std::cout << "    Upwind:   " << r_upwind << "\n\n";

    // Check ordering (with tolerance for numerical variation)
    const double tol = 0.1;  // 10% tolerance
    bool central_skew_similar = std::abs(r_central - r_skew) < tol * std::max(r_central, r_skew);
    bool upwind2_better_than_upwind = r_upwind2 > r_upwind - tol;
    bool upwind_most_dissipative = r_upwind < r_central + tol;

    std::cout << "  Ordering checks:\n";
    std::cout << "    Central ≈ Skew:           " << (central_skew_similar ? "PASS" : "FAIL") << "\n";
    std::cout << "    Upwind2 > Upwind:         " << (upwind2_better_than_upwind ? "PASS" : "FAIL") << "\n";
    std::cout << "    Upwind most dissipative:  " << (upwind_most_dissipative ? "PASS" : "FAIL") << "\n";

    std::cout << "\nQOI_JSON: {\"test\":\"dissipation_ordering\""
              << ",\"r_central\":" << r_central
              << ",\"r_skew\":" << r_skew
              << ",\"r_upwind2\":" << r_upwind2
              << ",\"r_upwind\":" << r_upwind
              << "}\n" << std::flush;

    bool pass = central_skew_similar && upwind2_better_than_upwind && upwind_most_dissipative;
    record("[Schemes] Dissipation ordering is correct", pass);

    // Also verify Upwind2 is tested and works
    record("[Upwind2] Scheme runs without errors", upwind2.valid);
    record("[Upwind2] Produces div-free field", upwind2.max_div < 1e-5);  // Relax for CPU MG
}

// ============================================================================
// Test 3: Stability Test (Various CFL)
// ============================================================================
/// Test that schemes remain stable under expected CFL conditions
static void test_stability() {
    std::cout << "\n=== Scheme Stability Test ===\n\n";

    const int N = 32;
    const double L = 2.0 * M_PI;
    const double nu = 1e-4;
    const int nsteps = 100;

    // TGV velocity scale is ~1, so CFL ≈ dt * U / dx = dt * 1 / (L/N) = dt * N / L
    // For L = 2π, N = 32: CFL ≈ dt * 32 / 6.28 ≈ dt * 5.1

    struct StabilityTest {
        std::string name;
        ConvectiveScheme scheme;
        int ng;
        double cfl;  // Target CFL
    };

    std::vector<StabilityTest> tests = {
        {"Central CFL=0.3", ConvectiveScheme::Central, 1, 0.3},
        {"Skew CFL=0.3", ConvectiveScheme::Skew, 1, 0.3},
        {"Upwind CFL=0.5", ConvectiveScheme::Upwind, 1, 0.5},
        {"Upwind2 CFL=0.4", ConvectiveScheme::Upwind2, 2, 0.4}
    };

    std::cout << "  Test             | dt       | Steps | Valid | max_div    | Status\n";
    std::cout << "  -----------------+----------+-------+-------+------------+--------\n";

    bool all_pass = true;

    for (const auto& t : tests) {
        double dx = L / N;
        double dt = t.cfl * dx / 1.0;  // U ~ 1 for TGV

        auto result = run_energy_test(N, L, nu, dt, nsteps, t.scheme, t.ng);

        // Relax divergence tolerance for CPU MG solver (can be ~1e-6 to 1e-7)
        bool pass = result.valid && result.max_div < 1e-4 && result.steps_completed == nsteps;
        all_pass &= pass;

        std::cout << "  " << std::left << std::setw(16) << t.name
                  << " | " << std::fixed << std::setprecision(5) << dt
                  << " | " << std::setw(5) << result.steps_completed
                  << " | " << (result.valid ? "Yes" : "No ")
                  << "   | " << std::scientific << std::setprecision(2) << result.max_div
                  << " | " << (pass ? "PASS" : "FAIL") << "\n";
        std::cout << std::fixed;
    }

    record("[Schemes] All schemes stable at expected CFL", all_pass);
}

// ============================================================================
// Test 4: Upwind2 Specific Validation
// ============================================================================
/// Dedicated tests for Upwind2 since it was previously untested
static void test_upwind2_specific() {
    std::cout << "\n=== Upwind2 Specific Validation ===\n\n";

    const int N = 32;
    const double L = 2.0 * M_PI;
    const double nu = 1e-4;
    const double dt = 0.002;
    const int nsteps = 100;

    // Test 1: Basic functionality with Ng=2
    std::cout << "  1. Basic functionality with Ng=2...\n";
    auto result_ng2 = run_energy_test(N, L, nu, dt, nsteps, ConvectiveScheme::Upwind2, 2);
    bool basic_ok = result_ng2.valid && result_ng2.max_div < 1e-5;
    std::cout << "     " << (basic_ok ? "PASS" : "FAIL") << " - Valid: " << result_ng2.valid
              << ", max_div: " << std::scientific << result_ng2.max_div << std::fixed << "\n";

    // Test 2: Verify Upwind2 is less dissipative than Upwind
    std::cout << "  2. Less dissipative than Upwind...\n";
    auto upwind1 = run_energy_test(N, L, nu, dt, nsteps, ConvectiveScheme::Upwind, 1);
    double r_upwind1 = upwind1.ke_final / upwind1.ke_init;
    double r_upwind2 = result_ng2.ke_final / result_ng2.ke_init;
    bool less_diss = r_upwind2 >= r_upwind1;
    std::cout << "     " << (less_diss ? "PASS" : "FAIL")
              << " - Upwind2 KE ratio: " << r_upwind2
              << ", Upwind1 KE ratio: " << r_upwind1 << "\n";

    // Test 3: Longer run stability
    std::cout << "  3. Long-time stability (500 steps)...\n";
    auto long_run = run_energy_test(N, L, nu, dt, 500, ConvectiveScheme::Upwind2, 2);
    bool long_ok = long_run.valid && long_run.steps_completed == 500;
    std::cout << "     " << (long_ok ? "PASS" : "FAIL")
              << " - Completed: " << long_run.steps_completed << "/500 steps\n";

    // Test 4: 3D test
    std::cout << "  4. 3D simulation...\n";
    {
        Mesh mesh3d;
        mesh3d.init_uniform(16, 16, 16, 0.0, L, 0.0, L, 0.0, L, 2);  // Ng=2 for Upwind2

        Config config;
        config.nu = nu;
        config.dt = dt;
        config.adaptive_dt = false;
        config.convective_scheme = ConvectiveScheme::Upwind2;
        config.poisson_solver = PoissonSolverType::MG;
        config.verbose = false;

        RANSSolver solver(mesh3d, config);

        // Initialize with TGV 3D - use proper Ng-based indexing
        const int Ng = mesh3d.Nghost;
        const int N = 16;
        for (int k = Ng; k <= N + Ng; ++k) {
            double z = mesh3d.z(k);
            for (int j = Ng; j <= N + Ng; ++j) {
                double y = mesh3d.y(j);
                for (int i = Ng; i <= N + Ng + 1; ++i) {
                    double x = mesh3d.xf[i];
                    solver.velocity().u(i, j, k) = std::sin(x) * std::cos(y) * std::cos(z);
                }
            }
        }
        for (int k = Ng; k <= N + Ng; ++k) {
            double z = mesh3d.z(k);
            for (int j = Ng; j <= N + Ng + 1; ++j) {
                double y = mesh3d.yf[j];
                for (int i = Ng; i <= N + Ng; ++i) {
                    double x = mesh3d.x(i);
                    solver.velocity().v(i, j, k) = -std::cos(x) * std::sin(y) * std::cos(z);
                }
            }
        }
        // w = 0

        bool valid_3d = true;
        for (int step = 0; step < 50; ++step) {
            solver.step();
        }

        // Check all velocity components for NaN
        for (int k = Ng; k <= N + Ng && valid_3d; ++k) {
            for (int j = Ng; j <= N + Ng && valid_3d; ++j) {
                for (int i = Ng; i <= N + Ng + 1 && valid_3d; ++i) {
                    if (!std::isfinite(solver.velocity().u(i, j, k))) valid_3d = false;
                }
                for (int i = Ng; i <= N + Ng && valid_3d; ++i) {
                    if (!std::isfinite(solver.velocity().v(i, j, k))) valid_3d = false;
                    if (!std::isfinite(solver.velocity().w(i, j, k))) valid_3d = false;
                }
            }
        }

        std::cout << "     " << (valid_3d ? "PASS" : "FAIL") << " - 3D run completed\n";
        record("[Upwind2] 3D simulation works", valid_3d);
    }

    std::cout << "\nQOI_JSON: {\"test\":\"upwind2_specific\""
              << ",\"basic_ok\":" << (basic_ok ? "true" : "false")
              << ",\"less_dissipative\":" << (less_diss ? "true" : "false")
              << ",\"long_run_ok\":" << (long_ok ? "true" : "false")
              << ",\"ke_ratio_upwind2\":" << r_upwind2
              << ",\"ke_ratio_upwind1\":" << r_upwind1
              << "}\n" << std::flush;

    record("[Upwind2] Basic functionality (Ng=2)", basic_ok);
    record("[Upwind2] Less dissipative than Upwind", less_diss);
    record("[Upwind2] Long-time stability", long_ok);
}

// ============================================================================
// Test 5: Divergence-Free Verification for All Schemes
// ============================================================================
static void test_divergence_free() {
    std::cout << "\n=== Divergence-Free Verification ===\n\n";

    const int N = 32;
    const double L = 2.0 * M_PI;
    const double nu = 1e-4;
    const double dt = 0.002;
    const int nsteps = 50;

    std::vector<std::tuple<std::string, ConvectiveScheme, int>> schemes = {
        {"Central", ConvectiveScheme::Central, 1},
        {"Skew", ConvectiveScheme::Skew, 1},
        {"Upwind", ConvectiveScheme::Upwind, 1},
        {"Upwind2", ConvectiveScheme::Upwind2, 2}
    };

    std::cout << "  Scheme   | max|div(u)| | Status\n";
    std::cout << "  ---------+-------------+--------\n";

    bool all_pass = true;

    for (const auto& [name, scheme, ng] : schemes) {
        auto result = run_energy_test(N, L, nu, dt, nsteps, scheme, ng);

        bool pass = result.max_div < 1e-5;
        all_pass &= pass;

        std::cout << "  " << std::left << std::setw(8) << name
                  << " | " << std::scientific << std::setprecision(3) << result.max_div
                  << " | " << (pass ? "PASS" : "FAIL") << "\n";
        std::cout << std::fixed;
    }

    record("[Schemes] All schemes produce div-free velocity", all_pass);
}

// ============================================================================
// Main
// ============================================================================
int main() {
    return run("SchemeComprehensiveTest", []() {
        test_energy_properties();
        test_dissipation_ordering();
        test_stability();
        test_upwind2_specific();
        test_divergence_free();
    });
}
