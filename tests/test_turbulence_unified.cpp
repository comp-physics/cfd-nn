/// Unified Turbulence Model Tests
/// Tests smoke, realizability, EARSM trace-free, guard functionality, golden regression, features

#include "test_harness.hpp"
#include "test_utilities.hpp"
#include "features.hpp"
#include "turbulence_model.hpp"
#include "turbulence_baseline.hpp"
#include "turbulence_gep.hpp"
#include "turbulence_earsm.hpp"
#include <limits>

using namespace nncfd;
using nncfd::test::harness::record;
using nncfd::test::BCPattern;
using nncfd::test::create_velocity_bc;
using nncfd::test::file_exists;

static std::string resolve_nn_path(const std::string& subdir) {
    for (const auto& prefix : {"data/models/", "../data/models/"}) {
        std::string path = prefix + subdir;
        if (file_exists(path + "/layer0_W.txt")) return path;
    }
    return "";
}

static std::string model_name(TurbulenceModelType type) {
    switch (type) {
        case TurbulenceModelType::None: return "Laminar";
        case TurbulenceModelType::Baseline: return "Baseline";
        case TurbulenceModelType::GEP: return "GEP";
        case TurbulenceModelType::NNMLP: return "NN-MLP";
        case TurbulenceModelType::NNTBNN: return "NN-TBNN";
        case TurbulenceModelType::SSTKOmega: return "SST k-omega";
        case TurbulenceModelType::KOmega: return "k-omega";
        case TurbulenceModelType::EARSM_WJ: return "EARSM-WJ";
        case TurbulenceModelType::EARSM_GS: return "EARSM-GS";
        case TurbulenceModelType::EARSM_Pope: return "EARSM-Pope";
        default: return "Unknown";
    }
}

static bool is_transport_model(TurbulenceModelType type) {
    return type == TurbulenceModelType::SSTKOmega || type == TurbulenceModelType::KOmega ||
           type == TurbulenceModelType::EARSM_WJ || type == TurbulenceModelType::EARSM_GS ||
           type == TurbulenceModelType::EARSM_Pope;
}

//=============================================================================
// Section 1: Smoke Tests
//=============================================================================

struct SmokeResult { bool passed = false, skipped = false; std::string message; };

static SmokeResult run_smoke_test(TurbulenceModelType type, int num_iter = 100) {
    SmokeResult result;

    std::string nn_path;
    if (type == TurbulenceModelType::NNMLP) {
        nn_path = resolve_nn_path("mlp_channel_caseholdout");
        if (nn_path.empty()) { result.skipped = true; result.message = "MLP weights not found"; return result; }
    } else if (type == TurbulenceModelType::NNTBNN) {
        nn_path = resolve_nn_path("tbnn_channel_caseholdout");
        if (nn_path.empty()) { result.skipped = true; result.message = "TBNN weights not found"; return result; }
    }

    try {
        Mesh mesh;
        mesh.init_uniform(16, 32, 0.0, 2.0, -1.0, 1.0);

        Config config;
        config.nu = 0.001;
        config.dt = 0.001;
        config.turb_model = type;
        config.verbose = false;
        config.turb_guard_enabled = true;
        if (!nn_path.empty()) { config.nn_weights_path = nn_path; config.nn_scaling_path = nn_path; }

        RANSSolver solver(mesh, config);
        solver.set_body_force(0.001, 0.0);
        solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel2D));
        if (type != TurbulenceModelType::None)
            solver.set_turbulence_model(create_turbulence_model(type, nn_path, nn_path));

        solver.initialize_uniform(1.0, 0.0);
        FOR_INTERIOR_2D(mesh, i, j) {
            solver.velocity().u(i, j) = 0.1 * (1.0 - mesh.y(j) * mesh.y(j));
        }
        solver.velocity().u(mesh.i_end(), mesh.j_begin()) = 0.0;  // Staggered
        solver.sync_to_gpu();

        for (int step = 0; step < num_iter; ++step) solver.step();
        solver.sync_from_gpu();

        const auto& vel = solver.velocity();
        const auto& nu_t = solver.nu_t();

        FOR_INTERIOR_2D(mesh, i, j) {
            if (!std::isfinite(vel.u(i, j)) || !std::isfinite(vel.v(i, j)))
                { result.message = "NaN/Inf in velocity"; return result; }
            if (!std::isfinite(nu_t(i, j))) { result.message = "NaN/Inf in nu_t"; return result; }
            if (nu_t(i, j) < 0.0) { result.message = "Negative nu_t"; return result; }
        }

        if (is_transport_model(type)) {
            const auto& k = solver.k();
            const auto& omega = solver.omega();
            FOR_INTERIOR_2D(mesh, i, j) {
                if (!std::isfinite(k(i, j)) || k(i, j) < 1e-12) { result.message = "Invalid k"; return result; }
                if (!std::isfinite(omega(i, j)) || omega(i, j) < 1e-12) { result.message = "Invalid omega"; return result; }
            }
        }

        result.passed = true;
        result.message = "OK";
    } catch (const std::exception& e) {
        result.message = std::string("Exception: ") + e.what();
    }
    return result;
}

static void test_smoke_all_models() {
    std::vector<TurbulenceModelType> models = {
        TurbulenceModelType::None, TurbulenceModelType::Baseline,
        TurbulenceModelType::GEP, TurbulenceModelType::SSTKOmega,
        TurbulenceModelType::KOmega, TurbulenceModelType::EARSM_WJ,
        TurbulenceModelType::EARSM_GS, TurbulenceModelType::EARSM_Pope,
        TurbulenceModelType::NNMLP, TurbulenceModelType::NNTBNN
    };
    for (auto type : models) {
        auto result = run_smoke_test(type);
        record(("Smoke: " + model_name(type)).c_str(), result.passed, result.skipped);
    }
}

//=============================================================================
// Section 2: Transport Realizability
//=============================================================================

static void test_transport_realizability() {
    std::vector<TurbulenceModelType> transport_models = {
        TurbulenceModelType::SSTKOmega, TurbulenceModelType::KOmega,
        TurbulenceModelType::EARSM_WJ, TurbulenceModelType::EARSM_GS,
        TurbulenceModelType::EARSM_Pope
    };
    for (auto type : transport_models) {
        auto result = run_smoke_test(type, 500);
        record(("Realizability: " + model_name(type)).c_str(), result.passed, result.skipped);
    }
}

//=============================================================================
// Section 3: EARSM Trace-Free Constraint
//=============================================================================

static bool test_tensor_basis_trace_free() {
    std::vector<VelocityGradient> test_cases = {
        {0.0, 1.0, 0.0, 0.0}, {0.5, 0.5, -0.5, -0.5},
        {0.3, 0.7, -0.2, -0.3}, {2.0, 0.0, 0.0, -2.0}
    };

    for (const auto& grad : test_cases) {
        std::array<std::array<double, 3>, TensorBasis::NUM_BASIS> basis;
        TensorBasis::compute(grad, 0.1, 0.01, basis);
        for (int n = 0; n < TensorBasis::NUM_BASIS; ++n)
            if (std::abs(basis[n][0] + basis[n][2]) > 1e-10) return false;
    }
    return true;
}

static bool test_anisotropy_construction_trace_free() {
    std::vector<std::array<double, TensorBasis::NUM_BASIS>> G_cases = {
        {-0.1, 0.0, 0.0, 0.0}, {-0.1, 0.05, 0.0, 0.0},
        {-0.1, 0.05, 0.02, 0.0}, {-0.3, 0.1, 0.08, 0.0}
    };
    std::vector<VelocityGradient> grad_cases = {
        {0.0, 1.0, 0.0, 0.0}, {0.5, 0.5, -0.5, -0.5}, {1.0, 0.5, -0.3, -1.0}
    };

    for (const auto& grad : grad_cases) {
        std::array<std::array<double, 3>, TensorBasis::NUM_BASIS> basis;
        TensorBasis::compute(grad, 0.1, 0.01, basis);
        for (const auto& G : G_cases) {
            double b_xx, b_xy, b_yy;
            TensorBasis::construct_anisotropy(G, basis, b_xx, b_xy, b_yy);
            if (std::abs(b_xx + b_yy) > 1e-10) return false;
        }
    }
    return true;
}

static bool test_earsm_closures_trace_free() {
    Mesh mesh;
    mesh.init_uniform(8, 16, 0.0, 1.0, -1.0, 1.0);

    VectorField vel(mesh);
    for (int j = 0; j < mesh.total_Ny(); ++j)
        for (int i = 0; i < mesh.total_Nx(); ++i) { vel.u(i, j) = mesh.y(j); vel.v(i, j) = 0.0; }

    ScalarField k(mesh, 0.1), omega(mesh, 10.0), nu_t(mesh);

    std::vector<EARSMType> types = {
        EARSMType::WallinJohansson2000, EARSMType::GatskiSpeziale1993, EARSMType::Pope1975
    };

    for (auto type : types) {
        TensorField tau_ij(mesh);
        SSTWithEARSM model(type);
        model.set_nu(0.001);
        model.set_delta(1.0);
        model.initialize(mesh, vel);
        model.update(mesh, vel, k, omega, nu_t, &tau_ij);

        FOR_INTERIOR_2D(mesh, i, j) {
            if (k(i, j) < 1e-10) continue;
            double b_trace = tau_ij.trace(i, j) / (2.0 * k(i, j)) - 2.0/3.0;
            if (std::abs(b_trace) > 1e-10) return false;
        }
    }
    return true;
}

static void test_earsm_trace_free() {
    record("Tensor basis trace-free", test_tensor_basis_trace_free());
    record("Anisotropy construction trace-free", test_anisotropy_construction_trace_free());
    record("EARSM closures trace-free", test_earsm_closures_trace_free());
}

//=============================================================================
// Section 4: Guard Functionality
//=============================================================================

static bool test_guard_allows_normal_operation() {
    Mesh mesh;
    mesh.init_uniform(32, 64, 0.0, 2.0, -1.0, 1.0);

    Config config;
    config.nu = 0.01; config.dt = 5e-4;
    config.turb_model = TurbulenceModelType::SSTKOmega;
    config.turb_guard_enabled = true; config.verbose = false;

    RANSSolver solver(mesh, config);
    solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel2D));
    solver.set_turbulence_model(create_turbulence_model(TurbulenceModelType::SSTKOmega));
    solver.set_body_force(-0.001, 0.0);
    solver.initialize_uniform(0.5, 0.0);

    try { for (int i = 0; i < 100; ++i) solver.step(); return true; }
    catch (...) { return false; }
}

static bool test_guard_detects_nan() {
    Mesh mesh;
    mesh.init_uniform(16, 32, 0.0, 1.0, -0.5, 0.5);

    Config config;
    config.nu = 0.01; config.dt = 1e-3;
    config.turb_model = TurbulenceModelType::None;
    config.turb_guard_enabled = true; config.turb_guard_interval = 1;
    config.verbose = false;

    RANSSolver solver(mesh, config);
    solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel2D));
    solver.initialize_uniform(1.0, 0.0);

    for (int i = 0; i < 5; ++i) solver.step();

    solver.velocity().u(mesh.Nx/2, mesh.Ny/2) = std::numeric_limits<double>::quiet_NaN();
    solver.sync_to_gpu();

    try {
        solver.check_for_nan_inf(5);
        return false;
    } catch (const std::runtime_error& e) {
        std::string msg(e.what());
        return msg.find("NaN") != std::string::npos || msg.find("NUMERICAL") != std::string::npos;
    }
}

static void test_guard_functionality() {
    record("Guard allows normal operation", test_guard_allows_normal_operation());
    record("Guard detects injected NaN", test_guard_detects_nan());
}

//=============================================================================
// Section 5: Golden Regression Tests
//=============================================================================

namespace golden {
    constexpr double LAMINAR_U_MEAN = 6.6739e-01, LAMINAR_U_MAX = 9.9942e-01;
    constexpr double BASELINE_U_MEAN = 6.6631e-01, BASELINE_U_MAX = 9.9876e-01;
    constexpr double TOLERANCE = 0.01;
}

struct VelStats { double u_mean, u_max; };

static VelStats compute_vel_stats(const RANSSolver& solver, const Mesh& mesh) {
    VelStats s{0.0, -1e30};
    int count = 0;
    const auto& vel = solver.velocity();
    FOR_INTERIOR_2D(mesh, i, j) {
        double u = 0.5 * (vel.u(i, j) + vel.u(i+1, j));
        s.u_mean += u;
        s.u_max = std::max(s.u_max, u);
        ++count;
    }
    if (count > 0) s.u_mean /= count;
    return s;
}

static VelStats run_golden_model(TurbulenceModelType type, const Mesh& mesh, int nsteps) {
    Config config;
    config.dt = 0.001; config.nu = 0.001; config.turb_model = type; config.verbose = false;

    RANSSolver solver(mesh, config);
    solver.set_turbulence_model(create_turbulence_model(type));
    solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel2D));

    FOR_INTERIOR_2D(mesh, i, j) {
        double y_norm = (mesh.y(j) - mesh.y_min) / (mesh.y_max - mesh.y_min);
        solver.velocity().u(i, j) = 4.0 * y_norm * (1.0 - y_norm);
    }
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        double y_norm = (mesh.y(j) - mesh.y_min) / (mesh.y_max - mesh.y_min);
        solver.velocity().u(mesh.i_end(), j) = 4.0 * y_norm * (1.0 - y_norm);
    }
    solver.initialize(solver.velocity());
    solver.set_body_force(0.01, 0.0, 0.0);

    for (int step = 0; step < nsteps; ++step) solver.step();
    solver.sync_from_gpu();
    return compute_vel_stats(solver, mesh);
}

static void test_golden_regression() {
    Mesh mesh;
    mesh.init_uniform(32, 32, 0.0, 2.0 * M_PI, 0.0, 2.0);

    auto laminar = run_golden_model(TurbulenceModelType::None, mesh, 50);
    auto baseline = run_golden_model(TurbulenceModelType::Baseline, mesh, 50);

    auto check = [](const VelStats& s, double exp_mean, double exp_max) {
        return std::abs(s.u_mean - exp_mean) / std::abs(exp_mean) < golden::TOLERANCE &&
               std::abs(s.u_max - exp_max) / std::abs(exp_max) < golden::TOLERANCE;
    };

    record("Golden: Laminar", check(laminar, golden::LAMINAR_U_MEAN, golden::LAMINAR_U_MAX));
    record("Golden: Baseline", check(baseline, golden::BASELINE_U_MEAN, golden::BASELINE_U_MAX));
}

//=============================================================================
// Section 6: Feature Computation
//=============================================================================

static bool test_feature_computer_batch() {
    Mesh mesh;
    mesh.init_uniform(16, 16, 0.0, 2.0, -1.0, 1.0);

    VectorField vel(mesh);
    for (int j = 0; j < mesh.total_Ny(); ++j)
        for (int i = 0; i < mesh.total_Nx(); ++i) { vel.u(i, j) = 2.0 * mesh.y(j); vel.v(i, j) = 0.0; }

    ScalarField k(mesh, 0.1), omega(mesh, 1.0);
    FeatureComputer fc(mesh);
    fc.set_reference(0.001, 1.0, 1.0);

    std::vector<Features> scalar_features;
    fc.compute_scalar_features(vel, k, omega, scalar_features);
    if (static_cast<int>(scalar_features.size()) != mesh.Nx * mesh.Ny) return false;
    for (const auto& feat : scalar_features)
        for (int n = 0; n < feat.size(); ++n)
            if (!std::isfinite(feat[n])) return false;

    std::vector<Features> tbnn_features;
    std::vector<std::array<std::array<double, 3>, TensorBasis::NUM_BASIS>> basis;
    fc.compute_tbnn_features(vel, k, omega, tbnn_features, basis);
    return static_cast<int>(tbnn_features.size()) == mesh.Nx * mesh.Ny;
}

static void test_feature_computation() {
    record("Feature computer batch", test_feature_computer_batch());
}

//=============================================================================
// Main
//=============================================================================

int main() {
    using namespace nncfd::test::harness;
    return run_sections("Unified Turbulence Model Tests", {
        {"Smoke Tests (all models, 100 steps)", test_smoke_all_models},
        {"Transport Realizability (500 steps)", test_transport_realizability},
        {"EARSM Trace-Free Constraint", test_earsm_trace_free},
        {"Guard Functionality", test_guard_functionality},
        {"Golden Regression Tests", test_golden_regression},
        {"Feature Computation", test_feature_computation}
    });
}
