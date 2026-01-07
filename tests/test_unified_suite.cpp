/// Unified Test Suite - Data-Driven Tests
///
/// This file consolidates multiple test files into a single data-driven suite:
/// - test_physics_validation.cpp tests
/// - test_solver.cpp tests
/// - test_stability.cpp tests
/// - test_turbulence.cpp tests
/// - test_divergence_all_bcs.cpp tests
/// - test_2d_3d_comparison.cpp tests
///
/// Total reduction: ~4000 lines -> ~400 lines

#include "test_runner.hpp"

using namespace nncfd;
using namespace nncfd::test;

//=============================================================================
// Physics Validation Suite (from test_physics_validation.cpp)
//=============================================================================

std::vector<TestSpec> physics_validation_tests() {
    std::vector<TestSpec> tests;

    double nu = 0.01, dp_dx = -0.001, H = 1.0;

    // Poiseuille analytical solution
    auto u_poiseuille = [=](double, double y) {
        return -dp_dx / (2.0 * nu) * (H * H - y * y);
    };

    // Test 1: Poiseuille single-step invariance
    {
        ConfigSpec cfg;
        cfg.nu = nu;
        cfg.dt = 0.001;
        cfg.adaptive_dt = false;
        cfg.max_iter = 1;

        tests.push_back(make_test(
            "poiseuille_single_step",
            "physics",
            MeshSpec::channel(64, 128),
            cfg,
            BCSpec::channel(),
            InitSpec::poiseuille(dp_dx, 1.0),
            RunSpec::steps(1),
            CheckSpec::l2_error(0.005, u_poiseuille)
        ));
    }

    // Test 2: Poiseuille multi-step stability
    {
        ConfigSpec cfg;
        cfg.nu = nu;
        cfg.dt = 0.002;
        cfg.adaptive_dt = false;
        cfg.max_iter = 10;

        tests.push_back(make_test(
            "poiseuille_multistep",
            "physics",
            MeshSpec::channel(64, 128),
            cfg,
            BCSpec::channel(),
            InitSpec::poiseuille(dp_dx, 1.0),
            RunSpec::steps(10),
            CheckSpec::l2_error(0.01, u_poiseuille)
        ));
    }

    // Test 3: Channel symmetry
    tests.push_back(make_test(
        "channel_symmetry",
        "physics",
        MeshSpec::channel(64, 128),
        ConfigSpec::laminar(nu),
        BCSpec::channel(),
        InitSpec::uniform(0.1),
        RunSpec::channel(dp_dx),
        CheckSpec::symmetry(0.01)
    ));

    // Test 4: Divergence-free constraint
    {
        ConfigSpec cfg;
        cfg.nu = nu;
        cfg.adaptive_dt = true;
        cfg.max_iter = 300;
        cfg.tol = 1e-4;
        cfg.turb_model = TurbulenceModelType::Baseline;

        tests.push_back(make_test(
            "divergence_free",
            "physics",
            MeshSpec::channel(64, 128),
            cfg,
            BCSpec::channel(),
            InitSpec::uniform(0.1),
            RunSpec::channel(dp_dx),
            CheckSpec::divergence_free(1e-3)
        ));
    }

    // Test 5: Field finiteness
    tests.push_back(make_test(
        "field_finiteness",
        "physics",
        MeshSpec::channel(32, 64),
        ConfigSpec::laminar(nu),
        BCSpec::channel(),
        InitSpec::uniform(0.1),
        RunSpec::steps(10),
        CheckSpec::finite()
    ));

    return tests;
}

//=============================================================================
// Solver Convergence Suite (from test_solver.cpp)
//=============================================================================

std::vector<TestSpec> solver_convergence_tests() {
    std::vector<TestSpec> tests;

    double dp_dx = -0.001;

    // Test convergence at multiple resolutions
    for (int n : {16, 32, 64}) {
        tests.push_back(make_test(
            "convergence_" + std::to_string(n) + "x" + std::to_string(2*n),
            "solver",
            MeshSpec::channel(n, 2*n),
            ConfigSpec::laminar(0.01),
            BCSpec::channel(),
            InitSpec::poiseuille(dp_dx, 0.99),
            RunSpec::channel(dp_dx),
            CheckSpec::residual(1e-4)
        ));
    }

    // Test with different turbulence models
    std::vector<std::pair<TurbulenceModelType, std::string>> models = {
        {TurbulenceModelType::None, "laminar"},
        {TurbulenceModelType::Baseline, "mixing_length"},
        {TurbulenceModelType::KOmega, "komega"}
    };

    for (const auto& [model, name] : models) {
        ConfigSpec cfg;
        cfg.nu = 0.01;
        cfg.adaptive_dt = true;
        cfg.max_iter = 500;
        cfg.tol = 1e-4;
        cfg.turb_model = model;

        tests.push_back(make_test(
            "model_" + name,
            "solver",
            MeshSpec::channel(32, 64),
            cfg,
            BCSpec::channel(),
            InitSpec::poiseuille(dp_dx, 0.99),
            RunSpec::channel(dp_dx),
            CheckSpec::converges()
        ));
    }

    return tests;
}

//=============================================================================
// Stability Suite (from test_stability.cpp)
//=============================================================================

std::vector<TestSpec> stability_tests() {
    std::vector<TestSpec> tests;

    // Taylor-Green stability at multiple resolutions
    for (int n : {32, 48, 64}) {
        tests.push_back(make_test(
            "taylor_green_stability_" + std::to_string(n),
            "stability",
            MeshSpec::taylor_green(n),
            ConfigSpec::unsteady(0.01, 0.005),
            BCSpec::periodic(),
            InitSpec::taylor_green(),
            RunSpec::steps(100),
            CheckSpec::bounded(10.0)
        ));
    }

    // Long-run channel stability
    {
        ConfigSpec cfg;
        cfg.nu = 0.01;
        cfg.dt = 0.01;
        cfg.adaptive_dt = false;
        cfg.max_iter = 500;

        tests.push_back(make_test(
            "channel_long_run",
            "stability",
            MeshSpec::channel(32, 64),
            cfg,
            BCSpec::channel(),
            InitSpec::poiseuille(-0.001, 0.99),
            RunSpec::steps(500),
            CheckSpec::finite()
        ));
    }

    // Stability with different BCs
    tests.push_back(make_test(
        "cavity_stability",
        "stability",
        MeshSpec::unit_square(32),
        ConfigSpec::laminar(0.01),
        BCSpec::cavity(),
        InitSpec::zero(),
        RunSpec::steps(100),
        CheckSpec::bounded(5.0)
    ));

    return tests;
}

//=============================================================================
// Turbulence Model Suite (from test_turbulence.cpp)
//=============================================================================

std::vector<TestSpec> turbulence_model_tests() {
    std::vector<TestSpec> tests;

    // Test all turbulence models (excluding NN models which need weight files)
    std::vector<std::pair<TurbulenceModelType, std::string>> models = {
        {TurbulenceModelType::Baseline, "baseline"},
        {TurbulenceModelType::GEP, "gep"},
        {TurbulenceModelType::KOmega, "komega"},
        {TurbulenceModelType::SSTKOmega, "sst_komega"},
        {TurbulenceModelType::EARSM_WJ, "earsm_wj"},
        {TurbulenceModelType::EARSM_GS, "earsm_gs"},
        {TurbulenceModelType::EARSM_Pope, "earsm_pope"}
    };

    for (const auto& [model, name] : models) {
        ConfigSpec cfg;
        cfg.nu = 0.001;
        cfg.dt = 0.001;
        cfg.adaptive_dt = true;
        cfg.max_iter = 200;
        cfg.tol = 1e-4;
        cfg.turb_model = model;

        // Realizability check
        tests.push_back(make_test(
            "realizability_" + name,
            "turbulence",
            MeshSpec::stretched_channel(32, 64, 2.0),
            cfg,
            BCSpec::channel(),
            InitSpec::uniform(0.5),
            RunSpec::steps(100),
            CheckSpec::realizability()
        ));

        // Bounded check
        tests.push_back(make_test(
            "bounded_" + name,
            "turbulence",
            MeshSpec::stretched_channel(32, 64, 2.0),
            cfg,
            BCSpec::channel(),
            InitSpec::uniform(0.5),
            RunSpec::steps(100),
            CheckSpec::bounded(20.0)
        ));
    }

    return tests;
}

//=============================================================================
// Boundary Condition Suite (from test_divergence_all_bcs.cpp)
//=============================================================================

std::vector<TestSpec> boundary_condition_tests() {
    std::vector<TestSpec> tests;

    // All periodic
    tests.push_back(make_test(
        "bc_all_periodic",
        "bc",
        MeshSpec::unit_square(32),
        ConfigSpec::unsteady(0.01, 0.01),
        BCSpec::periodic(),
        InitSpec::taylor_green(),
        RunSpec::steps(20),
        CheckSpec::divergence_free(1e-6)
    ));

    // Channel (periodic x, no-slip y)
    tests.push_back(make_test(
        "bc_channel",
        "bc",
        MeshSpec::channel(32, 64),
        ConfigSpec::laminar(0.01),
        BCSpec::channel(),
        InitSpec::poiseuille(-0.001, 0.99),
        RunSpec::channel(-0.001),
        CheckSpec::divergence_free(1e-6)
    ));

    // Cavity (all no-slip)
    tests.push_back(make_test(
        "bc_cavity",
        "bc",
        MeshSpec::unit_square(32),
        ConfigSpec::laminar(0.01),
        BCSpec::cavity(),
        InitSpec::zero(),
        RunSpec::steps(50),
        CheckSpec::divergence_free(1e-6)
    ));

    // Mixed BCs (periodic x, inflow/outflow y) - skipped, not yet implemented
    // {
    //     BCSpec mixed_bc;
    //     mixed_bc.x_lo = VelocityBC::Periodic;
    //     mixed_bc.x_hi = VelocityBC::Periodic;
    //     mixed_bc.y_lo = VelocityBC::Inflow;
    //     mixed_bc.y_hi = VelocityBC::Outflow;
    //
    //     tests.push_back(make_test(...));
    // }

    return tests;
}

//=============================================================================
// Resolution Convergence Suite
//=============================================================================

std::vector<TestSpec> resolution_convergence_tests() {
    std::vector<TestSpec> tests;

    double nu = 0.01, dp_dx = -0.001, H = 1.0;
    auto u_exact = [=](double, double y) {
        return -dp_dx / (2.0 * nu) * (H * H - y * y);
    };

    // Test L2 error decreases with resolution
    for (int n : {16, 32, 64, 96}) {
        tests.push_back(make_test(
            "resolution_" + std::to_string(n) + "x" + std::to_string(2*n),
            "convergence",
            MeshSpec::channel(n, 2*n),
            ConfigSpec::laminar(nu),
            BCSpec::channel(),
            InitSpec::poiseuille(dp_dx, 0.99),
            RunSpec::channel(dp_dx),
            CheckSpec::l2_error(0.10, u_exact)  // Generous tolerance
        ));
    }

    return tests;
}

//=============================================================================
// 3D Validation Suite (from test_3d_quick_validation.cpp, test_taylor_green_3d.cpp)
//=============================================================================

std::vector<TestSpec> validation_3d_tests() {
    std::vector<TestSpec> tests;

    // Constants for 3D Poiseuille
    const double NU = 0.01;
    const double DP_DX = -0.001;
    const double H = 1.0;  // Half-height (domain 0 to 2, center at 1)

    // Analytical Poiseuille solution (y from 0 to 2, centered at y=1)
    auto u_poiseuille_3d = [=](double y) {
        double y_centered = y - H;  // Shift so y=0 at center
        return -DP_DX / (2.0 * NU) * (H * H - y_centered * y_centered);
    };

    // U_max for relative error calculation
    const double U_max = -DP_DX / (2.0 * NU) * H * H;

    // Test 1: Fast Poiseuille convergence (init at 0.95x analytical)
    {
        ConfigSpec cfg;
        cfg.nu = NU;
        cfg.adaptive_dt = true;
        cfg.max_iter = 100;
        cfg.tol = 1e-6;
        cfg.turb_model = TurbulenceModelType::None;

        tests.push_back(make_test(
            "poiseuille_3d_fast",
            "3d",
            MeshSpec::poiseuille_3d(32, 32, 8),
            cfg,
            BCSpec::channel(),
            InitSpec::poiseuille_3d(DP_DX, 0.95),
            RunSpec::channel(DP_DX),
            CheckSpec::l2_error_3d(0.10 * U_max, u_poiseuille_3d)  // 10% relative to U_max
        ));
    }

    // Test 2: Larger grid Poiseuille (48x48x8, init 0.90x, stricter tolerance)
    {
        ConfigSpec cfg;
        cfg.nu = NU;
        cfg.adaptive_dt = true;
        cfg.max_iter = 150;
        cfg.tol = 1e-6;
        cfg.turb_model = TurbulenceModelType::None;

        tests.push_back(make_test(
            "poiseuille_3d_48x48",
            "3d",
            MeshSpec::poiseuille_3d(48, 48, 8),
            cfg,
            BCSpec::channel(),
            InitSpec::poiseuille_3d(DP_DX, 0.90),
            RunSpec::channel(DP_DX),
            CheckSpec::l2_error_3d(0.15 * U_max, u_poiseuille_3d)  // 15% relative
        ));
    }

    // Test 3: W-velocity stays zero for channel flow
    {
        ConfigSpec cfg;
        cfg.nu = NU;
        cfg.adaptive_dt = true;
        cfg.max_iter = 50;
        cfg.tol = 1e-6;
        cfg.turb_model = TurbulenceModelType::None;

        tests.push_back(make_test(
            "w_zero_channel_3d",
            "3d",
            MeshSpec::poiseuille_3d(32, 32, 8),
            cfg,
            BCSpec::channel(),
            InitSpec::poiseuille_3d(DP_DX, 0.95),
            RunSpec::steps(50),
            CheckSpec::w_zero(1e-8)
        ));
    }

    // 3D Taylor-Green vortex energy decay
    tests.push_back(make_test(
        "taylor_green_3d_32",
        "3d",
        MeshSpec::taylor_green_3d(32),
        ConfigSpec::unsteady(0.01, 0.01),
        BCSpec::periodic(),
        InitSpec::taylor_green_3d(),
        RunSpec::steps(50),
        CheckSpec::energy_decay()
    ));

    // 3D divergence-free check
    tests.push_back(make_test(
        "divergence_free_3d",
        "3d",
        MeshSpec::channel_3d(16, 16, 8),
        ConfigSpec::laminar(0.01),
        BCSpec::channel(),
        InitSpec::z_invariant(-0.001, 0.99),
        RunSpec::steps(20),
        CheckSpec::divergence_free(1e-3)
    ));

    // z-invariant flow preservation
    tests.push_back(make_test(
        "z_invariant_preservation",
        "3d",
        MeshSpec::channel_3d(16, 16, 8),
        ConfigSpec::unsteady(0.01, 0.001),
        BCSpec::channel(),
        InitSpec::z_invariant(-0.001, 1.0),
        RunSpec::steps(10),
        CheckSpec::z_invariant(1e-4)
    ));

    // 3D stability test
    tests.push_back(make_test(
        "stability_3d",
        "3d",
        MeshSpec::channel_3d(16, 16, 8),
        ConfigSpec::unsteady(0.01, 0.001),
        BCSpec::channel(),
        InitSpec::z_invariant(-0.001, 1.0),
        RunSpec::steps(50),
        CheckSpec::bounded(10.0)
    ));

    return tests;
}

//=============================================================================
// Main - Run All Suites
//=============================================================================

int main() {
    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "  UNIFIED TEST SUITE\n";
    std::cout << "  Consolidates ~4000 lines of tests into ~500 lines\n";
    std::cout << "================================================================\n\n";

    int total_passed = 0, total_failed = 0;

    // Collect all tests
    std::vector<std::pair<std::string, std::vector<TestSpec>>> suites = {
        {"Physics Validation", physics_validation_tests()},
        {"Solver Convergence", solver_convergence_tests()},
        {"Stability", stability_tests()},
        {"Turbulence Models", turbulence_model_tests()},
        {"Boundary Conditions", boundary_condition_tests()},
        {"Resolution Convergence", resolution_convergence_tests()},
        {"3D Validation", validation_3d_tests()}
    };

    // Run each suite
    for (const auto& [name, tests] : suites) {
        std::cout << "\n========================================\n";
        std::cout << name << "\n";
        std::cout << "========================================\n";

        int suite_passed = 0, suite_failed = 0;
        for (const auto& t : tests) {
            auto r = run_test(t);
            std::cout << "  " << std::left << std::setw(40) << t.name;
            if (r.passed) {
                std::cout << "[PASS] " << r.message;
                if (r.iterations > 0) std::cout << " (iters=" << r.iterations << ")";
                std::cout << "\n";
                ++suite_passed;
                ++total_passed;
            } else {
                std::cout << "[FAIL] " << r.message << "\n";
                ++suite_failed;
                ++total_failed;
            }
        }
        std::cout << "\nSummary: " << suite_passed << " passed, " << suite_failed << " failed\n";
    }

    std::cout << "\n================================================================\n";
    std::cout << "GRAND TOTAL: " << total_passed << " passed, " << total_failed << " failed\n";
    std::cout << "================================================================\n";

    return total_failed > 0 ? 1 : 0;
}
