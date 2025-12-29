/// Transport Equation Realizability Test
/// Verifies that transport turbulence models maintain physical realizability constraints
/// over long simulations:
///   - k > 0 (turbulent kinetic energy must be positive)
///   - omega > 0 (specific dissipation must be positive)
///   - nu_t >= 0 (eddy viscosity must be non-negative)
///   - All fields finite (no NaN/Inf)

#include "solver.hpp"
#include "mesh.hpp"
#include "config.hpp"
#include "turbulence_baseline.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <vector>

using namespace nncfd;

// Get model name for display
std::string model_name(TurbulenceModelType type) {
    switch (type) {
        case TurbulenceModelType::SSTKOmega: return "SST k-omega";
        case TurbulenceModelType::KOmega: return "k-omega";
        case TurbulenceModelType::EARSM_WJ: return "EARSM (Wallin-Johansson)";
        case TurbulenceModelType::EARSM_GS: return "EARSM (Gatski-Speziale)";
        case TurbulenceModelType::EARSM_Pope: return "EARSM (Pope)";
        default: return "Unknown";
    }
}

struct RealizabilityResult {
    bool passed;
    int failure_step;
    std::string failure_reason;
    double k_min;
    double omega_min;
    double nu_t_min;
};

// Test realizability for a single model
RealizabilityResult test_model_realizability(TurbulenceModelType type, int num_steps, int check_interval) {
    RealizabilityResult result{true, -1, "", 1e20, 1e20, 1e20};

    // Tolerance for numerical realizability (transport models clip at k_min=1e-10)
    const double k_tol = 1e-12;
    const double omega_tol = 1e-12;
    const double nu_t_tol = -1e-15;  // Allow tiny negative due to floating point

    // Setup: 16x32 channel flow
    Mesh mesh;
    mesh.init_uniform(16, 32, 0.0, 2.0, -1.0, 1.0);

    Config config;
    config.nu = 0.001;
    config.dt = 0.001;
    config.adaptive_dt = false;
    config.turb_model = type;
    config.verbose = false;
    config.turb_guard_enabled = true;
    config.turb_guard_interval = 10;

    RANSSolver solver(mesh, config);
    solver.set_body_force(0.001, 0.0);

    // Channel flow BCs
    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);

    // Create and set turbulence model
    auto model = create_turbulence_model(type);
    solver.set_turbulence_model(std::move(model));

    // Initialize
    solver.initialize_uniform(1.0, 0.0);
    solver.sync_to_gpu();

    // Run simulation with periodic realizability checks
    for (int step = 0; step < num_steps; ++step) {
        try {
            solver.step();
        } catch (const std::exception& e) {
            result.passed = false;
            result.failure_step = step;
            result.failure_reason = std::string("Exception: ") + e.what();
            return result;
        } catch (...) {
            result.passed = false;
            result.failure_step = step;
            result.failure_reason = "Unknown exception";
            return result;
        }

        // Check realizability at intervals
        if ((step + 1) % check_interval == 0) {
            solver.sync_from_gpu();

            const ScalarField& k = solver.k();
            const ScalarField& omega = solver.omega();
            const ScalarField& nu_t = solver.nu_t();

            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    double k_val = k(i, j);
                    double omega_val = omega(i, j);
                    double nu_t_val = nu_t(i, j);

                    // Track minimum values
                    result.k_min = std::min(result.k_min, k_val);
                    result.omega_min = std::min(result.omega_min, omega_val);
                    result.nu_t_min = std::min(result.nu_t_min, nu_t_val);

                    // Check for NaN/Inf
                    if (!std::isfinite(k_val)) {
                        result.passed = false;
                        result.failure_step = step + 1;
                        result.failure_reason = "NaN/Inf in k field";
                        return result;
                    }
                    if (!std::isfinite(omega_val)) {
                        result.passed = false;
                        result.failure_step = step + 1;
                        result.failure_reason = "NaN/Inf in omega field";
                        return result;
                    }
                    if (!std::isfinite(nu_t_val)) {
                        result.passed = false;
                        result.failure_step = step + 1;
                        result.failure_reason = "NaN/Inf in nu_t field";
                        return result;
                    }

                    // Check realizability constraints
                    if (k_val < k_tol) {
                        result.passed = false;
                        result.failure_step = step + 1;
                        result.failure_reason = "k <= 0 (non-positive TKE)";
                        return result;
                    }
                    if (omega_val < omega_tol) {
                        result.passed = false;
                        result.failure_step = step + 1;
                        result.failure_reason = "omega <= 0 (non-positive dissipation)";
                        return result;
                    }
                    if (nu_t_val < nu_t_tol) {
                        result.passed = false;
                        result.failure_step = step + 1;
                        result.failure_reason = "nu_t < 0 (negative eddy viscosity)";
                        return result;
                    }
                }
            }
        }
    }

    return result;
}

int main() {
    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "  TRANSPORT EQUATION REALIZABILITY TEST\n";
    std::cout << "================================================================\n";
    std::cout << "Tests transport models over 500 steps with realizability checks\n";
    std::cout << "Validates: k > 0, omega > 0, nu_t >= 0, finite values\n\n";

    // Transport models to test
    std::vector<TurbulenceModelType> models = {
        TurbulenceModelType::SSTKOmega,
        TurbulenceModelType::KOmega,
        TurbulenceModelType::EARSM_WJ,
        TurbulenceModelType::EARSM_GS,
        TurbulenceModelType::EARSM_Pope
    };

    const int num_steps = 500;
    const int check_interval = 50;

    int passed = 0;
    int failed = 0;

    std::cout << std::left << std::setw(30) << "Model"
              << std::setw(10) << "Status"
              << std::setw(15) << "k_min"
              << std::setw(15) << "omega_min"
              << std::setw(15) << "nu_t_min"
              << "\n";
    std::cout << std::string(85, '-') << "\n";

    for (auto type : models) {
        std::string name = model_name(type);
        std::cout << std::left << std::setw(30) << name << std::flush;

        RealizabilityResult result = test_model_realizability(type, num_steps, check_interval);

        if (result.passed) {
            std::cout << std::setw(10) << "PASS"
                      << std::scientific << std::setprecision(2)
                      << std::setw(15) << result.k_min
                      << std::setw(15) << result.omega_min
                      << std::setw(15) << result.nu_t_min
                      << "\n";
            passed++;
        } else {
            std::cout << std::setw(10) << "FAIL"
                      << "Step " << result.failure_step << ": " << result.failure_reason
                      << "\n";
            failed++;
        }
    }

    std::cout << std::string(85, '-') << "\n";

    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "SUMMARY\n";
    std::cout << "================================================================\n";
    std::cout << "Passed:  " << passed << "/" << models.size() << "\n";
    std::cout << "Failed:  " << failed << "/" << models.size() << "\n\n";

    if (failed == 0) {
        std::cout << "[SUCCESS] All transport models maintain realizability!\n";
        std::cout << "Verified over " << num_steps << " timesteps with checks every "
                  << check_interval << " steps\n";
        std::cout << "================================================================\n\n";
        return 0;
    } else {
        std::cout << "[FAILURE] " << failed << " model(s) violated realizability\n";
        std::cout << "================================================================\n\n";
        return 1;
    }
}
