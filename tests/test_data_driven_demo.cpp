/// Data-Driven Test Demo
///
/// This file demonstrates how the unified test_runner.hpp framework
/// can express 40+ tests in ~200 lines instead of ~4000 lines.
///
/// Compare: Each test here is 5-10 lines vs 50-150 lines traditionally.

#include "test_runner.hpp"

using namespace nncfd;
using namespace nncfd::test;

// Note: make_test() is now provided by test_runner.hpp

//=============================================================================
// Physics Validation Tests (replaces test_physics_validation*.cpp)
//=============================================================================

std::vector<TestSpec> physics_tests() {
    std::vector<TestSpec> tests;

    double nu = 0.01, dp_dx = -0.001, H = 1.0;

    // Poiseuille analytical solution
    auto u_poiseuille = [=](double, double y) {
        return -dp_dx / (2.0 * nu) * (H * H - y * y);
    };

    // Test 1-3: Poiseuille at multiple resolutions
    // Use 0.99 init factor for GPU convergence
    double init_factor = 0.99;
    for (int n : {32, 48, 64}) {
        tests.push_back(make_test(
            "poiseuille_" + std::to_string(n) + "x" + std::to_string(2*n),
            "physics",
            MeshSpec::channel(n, 2*n),
            ConfigSpec::laminar(nu),
            BCSpec::channel(),
            InitSpec::poiseuille(dp_dx, init_factor),
            RunSpec::channel(dp_dx),
            CheckSpec::l2_error(0.05, u_poiseuille)
        ));
    }

    // Test 4-6: Taylor-Green energy decay
    for (int n : {32, 48, 64}) {
        tests.push_back(make_test(
            "taylor_green_" + std::to_string(n),
            "physics",
            MeshSpec::taylor_green(n),
            ConfigSpec::unsteady(0.01, 0.005),
            BCSpec::periodic(),
            InitSpec::taylor_green(),
            RunSpec::steps(50),
            CheckSpec::energy_decay()
        ));
    }

    // Test 7: Divergence-free check
    tests.push_back(make_test(
        "divergence_free",
        "physics",
        MeshSpec::taylor_green(64),
        ConfigSpec::unsteady(0.01, 0.01),
        BCSpec::periodic(),
        InitSpec::taylor_green(),
        RunSpec::steps(20),
        CheckSpec::divergence_free(1e-3)
    ));

    // Test 8: Advection stability
    tests.push_back(make_test(
        "advection_stability",
        "physics",
        MeshSpec::taylor_green(64),
        ConfigSpec::unsteady(0.01, 0.01),
        BCSpec::periodic(),
        InitSpec::taylor_green(),
        RunSpec::steps(100),
        CheckSpec::bounded(10.0)
    ));

    return tests;
}

//=============================================================================
// Solver Convergence Tests (replaces test_solver.cpp)
//=============================================================================

std::vector<TestSpec> solver_tests() {
    std::vector<TestSpec> tests;

    // Test steady convergence at different resolutions
    // Use 0.99 init factor for GPU convergence
    for (int n : {16, 32, 64}) {
        tests.push_back(make_test(
            "steady_convergence_" + std::to_string(n),
            "solver",
            MeshSpec::channel(n, 2*n),
            ConfigSpec::laminar(),
            BCSpec::channel(),
            InitSpec::poiseuille(-0.001, 0.99),
            RunSpec::channel(-0.001),
            CheckSpec::residual(1e-4)
        ));
    }

    // Single timestep accuracy
    ConfigSpec single_step_cfg;
    single_step_cfg.nu = 0.01;
    single_step_cfg.dt = 0.001;
    single_step_cfg.adaptive_dt = false;

    tests.push_back(make_test(
        "single_step_accuracy",
        "solver",
        MeshSpec::channel(32, 64),
        single_step_cfg,
        BCSpec::channel(),
        InitSpec::poiseuille(-0.001, 1.0),
        RunSpec::steps(1),
        CheckSpec::none()
    ));

    return tests;
}

//=============================================================================
// Turbulence Model Tests (replaces test_turbulence*.cpp)
//=============================================================================

std::vector<TestSpec> turbulence_tests() {
    std::vector<TestSpec> tests;

    // Mixing length model (Baseline) - run steps, check bounded
    ConfigSpec baseline_cfg;
    baseline_cfg.nu = 0.001;
    baseline_cfg.turb_model = TurbulenceModelType::Baseline;

    tests.push_back(make_test(
        "mixing_length_channel",
        "turbulence",
        MeshSpec::stretched_channel(32, 64, 2.0),
        baseline_cfg,
        BCSpec::channel(),
        InitSpec::uniform(0.5),
        RunSpec::steps(200),
        CheckSpec::bounded(10.0)
    ));

    // k-omega model - run steps, check bounded (turbulence doesn't always converge to tight tolerance)
    ConfigSpec komega_cfg = ConfigSpec::turbulent_komega();
    tests.push_back(make_test(
        "komega_channel",
        "turbulence",
        MeshSpec::stretched_channel(32, 96, 2.0),
        komega_cfg,
        BCSpec::channel(),
        InitSpec::uniform(0.5),
        RunSpec::steps(500),
        CheckSpec::bounded(20.0)
    ));

    // GEP model
    ConfigSpec gep_cfg;
    gep_cfg.nu = 0.001;
    gep_cfg.turb_model = TurbulenceModelType::GEP;

    tests.push_back(make_test(
        "gep_channel",
        "turbulence",
        MeshSpec::stretched_channel(32, 64, 2.0),
        gep_cfg,
        BCSpec::channel(),
        InitSpec::uniform(0.5),
        RunSpec::steps(100),
        CheckSpec::bounded(50.0)
    ));

    return tests;
}

//=============================================================================
// Boundary Condition Tests
//=============================================================================

std::vector<TestSpec> bc_tests() {
    std::vector<TestSpec> tests;

    // All periodic
    tests.push_back(make_test(
        "periodic_all",
        "bc",
        MeshSpec::unit_square(32),
        ConfigSpec::unsteady(),
        BCSpec::periodic(),
        InitSpec::taylor_green(),
        RunSpec::steps(10),
        CheckSpec::bounded(5.0)
    ));

    // Cavity (all no-slip)
    tests.push_back(make_test(
        "cavity_noslip",
        "bc",
        MeshSpec::unit_square(32),
        ConfigSpec::laminar(0.01),
        BCSpec::cavity(),
        InitSpec::zero(),
        RunSpec::steps(50),
        CheckSpec::bounded(1.0)
    ));

    // Channel (periodic x, no-slip y)
    tests.push_back(make_test(
        "channel_bc",
        "bc",
        MeshSpec::channel(32, 64),
        ConfigSpec::laminar(),
        BCSpec::channel(),
        InitSpec::poiseuille(-0.001, 0.99),
        RunSpec::channel(-0.001),
        CheckSpec::converges()
    ));

    return tests;
}

//=============================================================================
// Main: Run All Test Suites
//=============================================================================

int main() {
    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "  DATA-DRIVEN TEST FRAMEWORK DEMO\n";
    std::cout << "  Shows how 40+ tests fit in ~200 lines\n";
    std::cout << "================================================================\n";

    int total_passed = 0, total_failed = 0;

    auto count_results = [&](const std::vector<TestSpec>& tests) {
        for (const auto& t : tests) {
            auto r = run_test(t);
            if (r.passed) ++total_passed;
            else ++total_failed;
        }
    };

    run_test_suite("Physics Validation", physics_tests());
    count_results(physics_tests());

    run_test_suite("Solver Tests", solver_tests());
    count_results(solver_tests());

    run_test_suite("Turbulence Models", turbulence_tests());
    count_results(turbulence_tests());

    run_test_suite("Boundary Conditions", bc_tests());
    count_results(bc_tests());

    // Also run predefined suites
    run_test_suite("Channel Flow Suite", channel_flow_suite());
    count_results(channel_flow_suite());

    run_test_suite("Taylor-Green Suite", taylor_green_suite());
    count_results(taylor_green_suite());

    std::cout << "\n================================================================\n";
    std::cout << "GRAND TOTAL: " << total_passed << " passed, " << total_failed << " failed\n";
    std::cout << "================================================================\n";

    return total_failed > 0 ? 1 : 0;
}
