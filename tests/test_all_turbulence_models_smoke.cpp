/// All Turbulence Models Smoke Test
/// Tests that all 10 turbulence models can run 100 steps without crashing or producing NaN/Inf
///
/// Models tested:
/// - None (laminar)
/// - Baseline (mixing length)
/// - GEP (gene expression programming)
/// - SSTKOmega, KOmega (transport models)
/// - EARSM_WJ, EARSM_GS, EARSM_Pope (explicit algebraic Reynolds stress)
/// - NNMLP, NNTBNN (neural network models)

#include "solver.hpp"
#include "mesh.hpp"
#include "config.hpp"
#include "turbulence_baseline.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

using namespace nncfd;

// Helper to check if a file exists
bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

// Get model name for display
std::string model_name(TurbulenceModelType type) {
    switch (type) {
        case TurbulenceModelType::None: return "None (Laminar)";
        case TurbulenceModelType::Baseline: return "Baseline (Mixing Length)";
        case TurbulenceModelType::GEP: return "GEP";
        case TurbulenceModelType::NNMLP: return "NN-MLP";
        case TurbulenceModelType::NNTBNN: return "NN-TBNN";
        case TurbulenceModelType::SSTKOmega: return "SST k-omega";
        case TurbulenceModelType::KOmega: return "k-omega";
        case TurbulenceModelType::EARSM_WJ: return "EARSM (Wallin-Johansson)";
        case TurbulenceModelType::EARSM_GS: return "EARSM (Gatski-Speziale)";
        case TurbulenceModelType::EARSM_Pope: return "EARSM (Pope)";
        default: return "Unknown";
    }
}

// Check if a model requires NN weights
bool requires_nn_weights(TurbulenceModelType type) {
    return type == TurbulenceModelType::NNMLP || type == TurbulenceModelType::NNTBNN;
}

// Check if model uses transport equations (k, omega)
bool uses_transport(TurbulenceModelType type) {
    return type == TurbulenceModelType::SSTKOmega ||
           type == TurbulenceModelType::KOmega ||
           type == TurbulenceModelType::EARSM_WJ ||
           type == TurbulenceModelType::EARSM_GS ||
           type == TurbulenceModelType::EARSM_Pope;
}

struct TestResult {
    bool passed;
    bool skipped;
    std::string message;
};

// Test a single turbulence model
TestResult test_model(TurbulenceModelType type) {
    TestResult result{false, false, ""};

    // Check for NN weights availability
    std::string nn_path;
    if (type == TurbulenceModelType::NNMLP) {
        nn_path = "data/models/mlp_channel_caseholdout";
        if (!file_exists(nn_path + "/layer0_W.txt")) {
            nn_path = "../data/models/mlp_channel_caseholdout";
            if (!file_exists(nn_path + "/layer0_W.txt")) {
                result.skipped = true;
                result.message = "MLP weights not found";
                return result;
            }
        }
    } else if (type == TurbulenceModelType::NNTBNN) {
        nn_path = "data/models/tbnn_channel_caseholdout";
        if (!file_exists(nn_path + "/layer0_W.txt")) {
            nn_path = "../data/models/tbnn_channel_caseholdout";
            if (!file_exists(nn_path + "/layer0_W.txt")) {
                result.skipped = true;
                result.message = "TBNN weights not found";
                return result;
            }
        }
    }

    try {
        // Setup: 16x32 channel
        Mesh mesh;
        mesh.init_uniform(16, 32, 0.0, 2.0, -1.0, 1.0);

        Config config;
        config.nu = 0.001;
        config.dt = 0.001;
        config.adaptive_dt = false;
        config.max_iter = 100;
        config.tol = 1e-6;
        config.turb_model = type;
        config.verbose = false;
        config.turb_guard_enabled = true;
        config.turb_guard_interval = 10;

        // Set NN paths if needed
        if (!nn_path.empty()) {
            config.nn_weights_path = nn_path;
            config.nn_scaling_path = nn_path;
        }

        RANSSolver solver(mesh, config);
        solver.set_body_force(0.001, 0.0);

        // Channel flow BCs
        VelocityBC bc;
        bc.x_lo = VelocityBC::Periodic;
        bc.x_hi = VelocityBC::Periodic;
        bc.y_lo = VelocityBC::NoSlip;
        bc.y_hi = VelocityBC::NoSlip;
        solver.set_velocity_bc(bc);

        // Create and set turbulence model (must be done before initialize)
        if (type != TurbulenceModelType::None) {
            auto model = create_turbulence_model(type, nn_path, nn_path);
            solver.set_turbulence_model(std::move(model));
        }

        // Initialize uniformly first (this sets up k/omega for transport models)
        solver.initialize_uniform(1.0, 0.0);

        // Then modify to Poiseuille-like profile
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            double y = mesh.y(j);
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                solver.velocity().u(i, j) = 0.1 * (1.0 - y * y);
            }
        }

        solver.sync_to_gpu();

        // Run 100 steps
        for (int step = 0; step < 100; ++step) {
            solver.step();
        }

        solver.sync_from_gpu();

        // Validate fields
        const VectorField& vel = solver.velocity();
        const ScalarField& nu_t = solver.nu_t();

        bool all_finite = true;
        bool nu_t_positive = true;
        bool k_positive = true;
        bool omega_positive = true;

        // Check velocity and nu_t
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                if (!std::isfinite(vel.u(i, j)) || !std::isfinite(vel.v(i, j))) {
                    all_finite = false;
                }
                if (!std::isfinite(nu_t(i, j))) {
                    all_finite = false;
                }
                if (nu_t(i, j) < 0.0) {
                    nu_t_positive = false;
                }
            }
        }

        // Check k and omega for transport models
        // Note: Transport models use k_min = 1e-10, omega_min = 1e-10 as floors
        const double k_min_tolerance = 1e-12;
        const double omega_min_tolerance = 1e-12;

        if (uses_transport(type)) {
            const ScalarField& k = solver.k();
            const ScalarField& omega = solver.omega();

            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    if (!std::isfinite(k(i, j)) || !std::isfinite(omega(i, j))) {
                        all_finite = false;
                    }
                    if (k(i, j) < k_min_tolerance) {
                        k_positive = false;
                    }
                    if (omega(i, j) < omega_min_tolerance) {
                        omega_positive = false;
                    }
                }
            }
        }

        // Determine result
        if (!all_finite) {
            result.message = "NaN/Inf detected in fields";
        } else if (!nu_t_positive) {
            result.message = "Negative nu_t detected";
        } else if (uses_transport(type) && !k_positive) {
            result.message = "Non-positive k detected";
        } else if (uses_transport(type) && !omega_positive) {
            result.message = "Non-positive omega detected";
        } else {
            result.passed = true;
            result.message = "All checks passed";
        }

    } catch (const std::exception& e) {
        result.message = std::string("Exception: ") + e.what();
    } catch (...) {
        result.message = "Unknown exception";
    }

    return result;
}

int main() {
    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "  ALL TURBULENCE MODELS SMOKE TEST\n";
    std::cout << "================================================================\n";
    std::cout << "Testing all 10 turbulence models with 100 timesteps each\n";
    std::cout << "Validates: No NaN/Inf, nu_t >= 0, k > 0, omega > 0\n\n";

    // List of all models to test
    std::vector<TurbulenceModelType> models = {
        TurbulenceModelType::None,
        TurbulenceModelType::Baseline,
        TurbulenceModelType::GEP,
        TurbulenceModelType::SSTKOmega,
        TurbulenceModelType::KOmega,
        TurbulenceModelType::EARSM_WJ,
        TurbulenceModelType::EARSM_GS,
        TurbulenceModelType::EARSM_Pope,
        TurbulenceModelType::NNMLP,
        TurbulenceModelType::NNTBNN
    };

    int passed = 0;
    int skipped = 0;
    int failed = 0;

    std::cout << std::left << std::setw(35) << "Model"
              << std::setw(10) << "Status"
              << "Details\n";
    std::cout << std::string(70, '-') << "\n";

    for (auto type : models) {
        std::string name = model_name(type);
        std::cout << std::left << std::setw(35) << name << std::flush;

        TestResult result = test_model(type);

        if (result.skipped) {
            std::cout << std::setw(10) << "SKIP" << result.message << "\n";
            skipped++;
        } else if (result.passed) {
            std::cout << std::setw(10) << "PASS" << result.message << "\n";
            passed++;
        } else {
            std::cout << std::setw(10) << "FAIL" << result.message << "\n";
            failed++;
        }
    }

    std::cout << std::string(70, '-') << "\n";

    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "SUMMARY\n";
    std::cout << "================================================================\n";
    std::cout << "Passed:  " << passed << "/" << models.size() << "\n";
    std::cout << "Skipped: " << skipped << "/" << models.size() << "\n";
    std::cout << "Failed:  " << failed << "/" << models.size() << "\n\n";

    if (failed == 0) {
        std::cout << "[SUCCESS] All tested models passed!\n";
        if (skipped > 0) {
            std::cout << "Note: " << skipped << " model(s) skipped due to missing weights\n";
        }
        std::cout << "================================================================\n\n";
        return 0;
    } else {
        std::cout << "[FAILURE] " << failed << " model(s) failed\n";
        std::cout << "================================================================\n\n";
        return 1;
    }
}
