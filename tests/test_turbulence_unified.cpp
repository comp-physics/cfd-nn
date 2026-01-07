/// Unified Turbulence Model Tests
/// Consolidates: test_turbulence_features, test_all_turbulence_models_smoke,
///               test_turbulence_guard, test_transport_realizability,
///               test_earsm_trace_free, test_turbulence_golden
///
/// Test sections:
/// 1. Smoke tests - all 10 models run without NaN/Inf
/// 2. Realizability - transport models maintain k>0, omega>0, nu_t>=0
/// 3. EARSM trace-free - anisotropy tensor satisfies b_xx + b_yy = 0
/// 4. Guard functionality - NaN/Inf detection works
/// 5. Golden regression - velocity statistics match reference
/// 6. Feature computation - batch feature computation works

#include "mesh.hpp"
#include "fields.hpp"
#include "features.hpp"
#include "solver.hpp"
#include "config.hpp"
#include "turbulence_model.hpp"
#include "turbulence_baseline.hpp"
#include "turbulence_gep.hpp"
#include "turbulence_earsm.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <limits>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

using namespace nncfd;

//=============================================================================
// Test Framework
//=============================================================================

static int g_passed = 0, g_failed = 0, g_skipped = 0;

static void record(const char* name, bool pass, bool skip = false) {
    std::cout << "  " << std::left << std::setw(50) << name;
    if (skip) { std::cout << "[SKIP]\n"; ++g_skipped; }
    else if (pass) { std::cout << "[PASS]\n"; ++g_passed; }
    else { std::cout << "[FAIL]\n"; ++g_failed; }
}

static bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

static std::string resolve_nn_path(const std::string& subdir) {
    std::string path = "data/models/" + subdir;
    if (file_exists(path + "/layer0_W.txt")) return path;
    path = "../data/models/" + subdir;
    if (file_exists(path + "/layer0_W.txt")) return path;
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
    return type == TurbulenceModelType::SSTKOmega ||
           type == TurbulenceModelType::KOmega ||
           type == TurbulenceModelType::EARSM_WJ ||
           type == TurbulenceModelType::EARSM_GS ||
           type == TurbulenceModelType::EARSM_Pope;
}

//=============================================================================
// Section 1: Smoke Tests (all models, 100 steps)
//=============================================================================

struct SmokeResult {
    bool passed = false;
    bool skipped = false;
    std::string message;
};

static SmokeResult run_smoke_test(TurbulenceModelType type, int num_steps = 100) {
    SmokeResult result;

    // Check NN weights availability
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
        if (!nn_path.empty()) {
            config.nn_weights_path = nn_path;
            config.nn_scaling_path = nn_path;
        }

        RANSSolver solver(mesh, config);
        solver.set_body_force(0.001, 0.0);

        VelocityBC bc;
        bc.x_lo = VelocityBC::Periodic;
        bc.x_hi = VelocityBC::Periodic;
        bc.y_lo = VelocityBC::NoSlip;
        bc.y_hi = VelocityBC::NoSlip;
        solver.set_velocity_bc(bc);

        if (type != TurbulenceModelType::None) {
            solver.set_turbulence_model(create_turbulence_model(type, nn_path, nn_path));
        }

        solver.initialize_uniform(1.0, 0.0);
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            double y = mesh.y(j);
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                solver.velocity().u(i, j) = 0.1 * (1.0 - y * y);
            }
        }
        solver.sync_to_gpu();

        for (int step = 0; step < num_steps; ++step) {
            solver.step();
        }
        solver.sync_from_gpu();

        // Validate fields
        const auto& vel = solver.velocity();
        const auto& nu_t = solver.nu_t();

        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                if (!std::isfinite(vel.u(i, j)) || !std::isfinite(vel.v(i, j))) {
                    result.message = "NaN/Inf in velocity"; return result;
                }
                if (!std::isfinite(nu_t(i, j))) {
                    result.message = "NaN/Inf in nu_t"; return result;
                }
                if (nu_t(i, j) < 0.0) {
                    result.message = "Negative nu_t"; return result;
                }
            }
        }

        if (is_transport_model(type)) {
            const auto& k = solver.k();
            const auto& omega = solver.omega();
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    if (!std::isfinite(k(i, j)) || k(i, j) < 1e-12) {
                        result.message = "Invalid k"; return result;
                    }
                    if (!std::isfinite(omega(i, j)) || omega(i, j) < 1e-12) {
                        result.message = "Invalid omega"; return result;
                    }
                }
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
    std::cout << "\n--- Smoke Tests (all models, 100 steps) ---\n\n";

    std::vector<TurbulenceModelType> models = {
        TurbulenceModelType::None, TurbulenceModelType::Baseline,
        TurbulenceModelType::GEP, TurbulenceModelType::SSTKOmega,
        TurbulenceModelType::KOmega, TurbulenceModelType::EARSM_WJ,
        TurbulenceModelType::EARSM_GS, TurbulenceModelType::EARSM_Pope,
        TurbulenceModelType::NNMLP, TurbulenceModelType::NNTBNN
    };

    for (auto type : models) {
        std::string name = "Smoke: " + model_name(type);
        auto result = run_smoke_test(type);
        record(name.c_str(), result.passed, result.skipped);
    }
}

//=============================================================================
// Section 2: Transport Realizability (500 steps)
//=============================================================================

static void test_transport_realizability() {
    std::cout << "\n--- Transport Realizability (500 steps) ---\n\n";

    std::vector<TurbulenceModelType> transport_models = {
        TurbulenceModelType::SSTKOmega, TurbulenceModelType::KOmega,
        TurbulenceModelType::EARSM_WJ, TurbulenceModelType::EARSM_GS,
        TurbulenceModelType::EARSM_Pope
    };

    for (auto type : transport_models) {
        std::string name = "Realizability: " + model_name(type);
        auto result = run_smoke_test(type, 500);
        record(name.c_str(), result.passed, result.skipped);
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

    const double tol = 1e-10;
    for (const auto& grad : test_cases) {
        std::array<std::array<double, 3>, TensorBasis::NUM_BASIS> basis;
        TensorBasis::compute(grad, 0.1, 0.01, basis);

        for (int n = 0; n < TensorBasis::NUM_BASIS; ++n) {
            double trace = basis[n][0] + basis[n][2];
            if (std::abs(trace) > tol) return false;
        }
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

    const double tol = 1e-10;
    for (const auto& grad : grad_cases) {
        std::array<std::array<double, 3>, TensorBasis::NUM_BASIS> basis;
        TensorBasis::compute(grad, 0.1, 0.01, basis);

        for (const auto& G : G_cases) {
            double b_xx, b_xy, b_yy;
            TensorBasis::construct_anisotropy(G, basis, b_xx, b_xy, b_yy);
            if (std::abs(b_xx + b_yy) > tol) return false;
        }
    }
    return true;
}

static bool test_earsm_closures_trace_free() {
    Mesh mesh;
    mesh.init_uniform(8, 16, 0.0, 1.0, -1.0, 1.0);

    VectorField vel(mesh);
    for (int j = 0; j < mesh.total_Ny(); ++j) {
        for (int i = 0; i < mesh.total_Nx(); ++i) {
            vel.u(i, j) = mesh.y(j);
            vel.v(i, j) = 0.0;
        }
    }

    ScalarField k(mesh, 0.1), omega(mesh, 10.0), nu_t(mesh);

    std::vector<EARSMType> types = {
        EARSMType::WallinJohansson2000, EARSMType::GatskiSpeziale1993, EARSMType::Pope1975
    };

    const double tol = 1e-10;
    for (auto type : types) {
        TensorField tau_ij(mesh);  // Fresh field for each model iteration
        SSTWithEARSM model(type);
        model.set_nu(0.001);
        model.set_delta(1.0);
        model.initialize(mesh, vel);
        model.update(mesh, vel, k, omega, nu_t, &tau_ij);

        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                if (k(i, j) < 1e-10) continue;
                double b_trace = tau_ij.trace(i, j) / (2.0 * k(i, j)) - 2.0/3.0;
                if (std::abs(b_trace) > tol) return false;
            }
        }
    }
    return true;
}

static void test_earsm_trace_free() {
    std::cout << "\n--- EARSM Trace-Free Constraint ---\n\n";

    record("Tensor basis trace-free", test_tensor_basis_trace_free());
    record("Anisotropy construction trace-free", test_anisotropy_construction_trace_free());
    record("EARSM closures trace-free", test_earsm_closures_trace_free());
}

//=============================================================================
// Section 4: Guard Functionality (NaN Detection)
//=============================================================================

static bool test_guard_allows_normal_operation() {
    Mesh mesh;
    mesh.init_uniform(32, 64, 0.0, 2.0, -1.0, 1.0);

    Config config;
    config.nu = 0.01;
    config.dt = 5e-4;
    config.turb_model = TurbulenceModelType::SSTKOmega;
    config.turb_guard_enabled = true;
    config.verbose = false;

    RANSSolver solver(mesh, config);
    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic; bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip; bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);
    solver.set_turbulence_model(create_turbulence_model(TurbulenceModelType::SSTKOmega));
    solver.set_body_force(-0.001, 0.0);
    solver.initialize_uniform(0.5, 0.0);

    try {
        for (int i = 0; i < 100; ++i) solver.step();
        return true;
    } catch (...) {
        return false;
    }
}

static bool test_guard_detects_nan() {
    Mesh mesh;
    mesh.init_uniform(16, 32, 0.0, 1.0, -0.5, 0.5);

    Config config;
    config.nu = 0.01;
    config.dt = 1e-3;
    config.turb_model = TurbulenceModelType::None;
    config.turb_guard_enabled = true;
    config.turb_guard_interval = 1;
    config.verbose = false;

    RANSSolver solver(mesh, config);
    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic; bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip; bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);
    solver.initialize_uniform(1.0, 0.0);

    for (int i = 0; i < 5; ++i) solver.step();

    // Inject NaN
    solver.velocity().u(mesh.Nx/2, mesh.Ny/2) = std::numeric_limits<double>::quiet_NaN();
#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    try {
        solver.check_for_nan_inf(5);
        return false;  // Should have thrown
    } catch (const std::runtime_error& e) {
        std::string msg(e.what());
        return msg.find("NaN") != std::string::npos || msg.find("NUMERICAL") != std::string::npos;
    }
}

static void test_guard_functionality() {
    std::cout << "\n--- Guard Functionality ---\n\n";

    record("Guard allows normal operation", test_guard_allows_normal_operation());
    record("Guard detects injected NaN", test_guard_detects_nan());
}

//=============================================================================
// Section 5: Golden Regression Tests
//=============================================================================

namespace golden {
    constexpr double LAMINAR_U_MEAN = 6.6739e-01;
    constexpr double LAMINAR_U_MAX  = 9.9942e-01;
    constexpr double BASELINE_U_MEAN = 6.6631e-01;
    constexpr double BASELINE_U_MAX  = 9.9876e-01;
    constexpr double TOLERANCE = 0.01;
}

struct VelStats { double u_mean, u_max; };

static VelStats compute_vel_stats(const RANSSolver& solver, const Mesh& mesh) {
    VelStats s{0.0, -1e30};
    int count = 0;
    const auto& vel = solver.velocity();
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double u = 0.5 * (vel.u(i, j) + vel.u(i+1, j));
            s.u_mean += u;
            s.u_max = std::max(s.u_max, u);
            ++count;
        }
    }
    if (count > 0) s.u_mean /= count;
    return s;
}

static VelStats run_golden_model(TurbulenceModelType type, const Mesh& mesh, int nsteps) {
    Config config;
    config.dt = 0.001;
    config.nu = 0.001;
    config.turb_model = type;
    config.verbose = false;

    RANSSolver solver(mesh, config);
    solver.set_turbulence_model(create_turbulence_model(type));

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic; bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip; bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);

    auto& vel = solver.velocity();
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            double y = mesh.y(j);
            double y_norm = (y - mesh.y_min) / (mesh.y_max - mesh.y_min);
            vel.u(i, j) = 4.0 * y_norm * (1.0 - y_norm);
        }
    }
    solver.initialize(vel);
    solver.set_body_force(0.01, 0.0, 0.0);

    for (int step = 0; step < nsteps; ++step) solver.step();
    solver.sync_from_gpu();

    return compute_vel_stats(solver, mesh);
}

static bool check_golden(const VelStats& actual, double exp_mean, double exp_max) {
    double err_mean = std::abs(actual.u_mean - exp_mean) / std::abs(exp_mean);
    double err_max = std::abs(actual.u_max - exp_max) / std::abs(exp_max);
    return err_mean < golden::TOLERANCE && err_max < golden::TOLERANCE;
}

static void test_golden_regression() {
    std::cout << "\n--- Golden Regression Tests ---\n\n";

    Mesh mesh;
    mesh.init_uniform(32, 32, 0.0, 2.0 * M_PI, 0.0, 2.0);
    const int nsteps = 50;

    auto laminar = run_golden_model(TurbulenceModelType::None, mesh, nsteps);
    auto baseline = run_golden_model(TurbulenceModelType::Baseline, mesh, nsteps);

    record("Golden: Laminar", check_golden(laminar, golden::LAMINAR_U_MEAN, golden::LAMINAR_U_MAX));
    record("Golden: Baseline", check_golden(baseline, golden::BASELINE_U_MEAN, golden::BASELINE_U_MAX));
}

//=============================================================================
// Section 6: Feature Computation
//=============================================================================

static bool test_feature_computer_batch() {
    Mesh mesh;
    mesh.init_uniform(16, 16, 0.0, 2.0, -1.0, 1.0);

    VectorField vel(mesh);
    for (int j = 0; j < mesh.total_Ny(); ++j) {
        for (int i = 0; i < mesh.total_Nx(); ++i) {
            vel.u(i, j) = 2.0 * mesh.y(j);
            vel.v(i, j) = 0.0;
        }
    }

    ScalarField k(mesh, 0.1), omega(mesh, 1.0);
    FeatureComputer fc(mesh);
    fc.set_reference(0.001, 1.0, 1.0);

    std::vector<Features> scalar_features;
    fc.compute_scalar_features(vel, k, omega, scalar_features);

    if (static_cast<int>(scalar_features.size()) != mesh.Nx * mesh.Ny) return false;

    for (const auto& feat : scalar_features) {
        for (int n = 0; n < feat.size(); ++n) {
            if (!std::isfinite(feat[n])) return false;
        }
    }

    std::vector<Features> tbnn_features;
    std::vector<std::array<std::array<double, 3>, TensorBasis::NUM_BASIS>> basis;
    fc.compute_tbnn_features(vel, k, omega, tbnn_features, basis);

    if (static_cast<int>(tbnn_features.size()) != mesh.Nx * mesh.Ny) return false;

    return true;
}

static void test_feature_computation() {
    std::cout << "\n--- Feature Computation ---\n\n";
    record("Feature computer batch", test_feature_computer_batch());
}

//=============================================================================
// Main
//=============================================================================

int main() {
    std::cout << "================================================================\n";
    std::cout << "  Unified Turbulence Model Tests\n";
    std::cout << "================================================================\n";
#ifdef USE_GPU_OFFLOAD
    std::cout << "Build: GPU (USE_GPU_OFFLOAD=ON)\n";
#else
    std::cout << "Build: CPU (USE_GPU_OFFLOAD=OFF)\n";
#endif

    test_smoke_all_models();
    test_transport_realizability();
    test_earsm_trace_free();
    test_guard_functionality();
    test_golden_regression();
    test_feature_computation();

    std::cout << "\n================================================================\n";
    std::cout << "Summary: " << g_passed << " passed, " << g_failed << " failed, "
              << g_skipped << " skipped\n";
    std::cout << "================================================================\n";

    return g_failed > 0 ? 1 : 0;
}
