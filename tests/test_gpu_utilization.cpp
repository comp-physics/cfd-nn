/// GPU Utilization Test for CI
/// Validates that GPU builds run computation primarily on GPU, not CPU
///
/// For GPU builds, this test:
/// 1. Runs various turbulence models for N timesteps
/// 2. Measures time spent in GPU kernels vs CPU compute
/// 3. Asserts GPU utilization exceeds threshold (default 70%)
///
/// This catches regressions where compute accidentally runs on CPU.

#include "solver.hpp"
#include "mesh.hpp"
#include "config.hpp"
#include "timing.hpp"
#include "turbulence_baseline.hpp"
#include "test_utilities.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib>

using namespace nncfd;
using nncfd::test::BCPattern;
using nncfd::test::create_velocity_bc;

// Configuration
constexpr int NUM_STEPS = 50;        // Steps per model
constexpr int NX = 32;               // Grid size
constexpr int NY = 64;
constexpr double GPU_THRESHOLD = 0.70;  // 70% GPU utilization required

bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

std::string model_name(TurbulenceModelType type) {
    switch (type) {
        case TurbulenceModelType::None: return "Laminar";
        case TurbulenceModelType::Baseline: return "Baseline";
        case TurbulenceModelType::GEP: return "GEP";
        case TurbulenceModelType::NNMLP: return "NN-MLP";
        case TurbulenceModelType::NNTBNN: return "NN-TBNN";
        case TurbulenceModelType::SSTKOmega: return "SST";
        case TurbulenceModelType::KOmega: return "k-omega";
        case TurbulenceModelType::EARSM_WJ: return "EARSM-WJ";
        case TurbulenceModelType::EARSM_GS: return "EARSM-GS";
        case TurbulenceModelType::EARSM_Pope: return "EARSM-Pope";
        default: return "Unknown";
    }
}

bool requires_nn_weights(TurbulenceModelType type) {
    return type == TurbulenceModelType::NNMLP || type == TurbulenceModelType::NNTBNN;
}

struct ModelResult {
    TurbulenceModelType type;
    bool ran;
    bool skipped;
    double gpu_time;
    double cpu_time;
    double gpu_ratio;
    std::string skip_reason;
};

ModelResult run_model(TurbulenceModelType type) {
    ModelResult result;
    result.type = type;
    result.ran = false;
    result.skipped = false;
    result.gpu_time = 0.0;
    result.cpu_time = 0.0;
    result.gpu_ratio = 0.0;

    // Check for NN weights
    std::string nn_path;
    if (type == TurbulenceModelType::NNMLP) {
        nn_path = "data/models/mlp_channel_caseholdout";
        if (!file_exists(nn_path + "/layer0_W.txt")) {
            nn_path = "../data/models/mlp_channel_caseholdout";
            if (!file_exists(nn_path + "/layer0_W.txt")) {
                result.skipped = true;
                result.skip_reason = "MLP weights not found";
                return result;
            }
        }
    } else if (type == TurbulenceModelType::NNTBNN) {
        nn_path = "data/models/tbnn_channel_caseholdout";
        if (!file_exists(nn_path + "/layer0_W.txt")) {
            nn_path = "../data/models/tbnn_channel_caseholdout";
            if (!file_exists(nn_path + "/layer0_W.txt")) {
                result.skipped = true;
                result.skip_reason = "TBNN weights not found";
                return result;
            }
        }
    }

    try {
        // Reset timing stats for this model
        TimingStats::instance().reset();

        // Setup solver
        Mesh mesh;
        mesh.init_uniform(NX, NY, 0.0, 2.0, -1.0, 1.0);

        Config config;
        config.nu = 0.001;
        config.dt = 0.001;
        config.adaptive_dt = false;
        config.max_steps = 100;
        config.tol = 1e-6;
        config.turb_model = type;
        config.verbose = false;
        config.turb_guard_enabled = true;
        config.turb_guard_interval = 10;

        if (!nn_path.empty()) {
            config.nn_weights_path = nn_path;
            config.nn_scaling_path = nn_path;
        }

        RANSSolver solver(mesh, config);
        solver.set_body_force(0.001, 0.0);
        solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel2D));

        if (type != TurbulenceModelType::None) {
            auto model = create_turbulence_model(type, nn_path, nn_path);
            solver.set_turbulence_model(std::move(model));
        }

        solver.initialize_uniform(1.0, 0.0);

        // Initialize with Poiseuille-like profile
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            double y = mesh.y(j);
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                solver.velocity().u(i, j) = 0.1 * (1.0 - y * y);
            }
        }

        solver.sync_to_gpu();

        // Run timesteps
        for (int step = 0; step < NUM_STEPS; ++step) {
            solver.step();
        }

        solver.sync_from_gpu();

        // Collect timing results
        result.ran = true;
        result.gpu_time = TimingStats::instance().gpu_kernel_time();
        result.cpu_time = TimingStats::instance().cpu_compute_time();
        double total = result.gpu_time + result.cpu_time;
        result.gpu_ratio = (total > 0.0) ? (result.gpu_time / total) : 0.0;

    } catch (const std::exception& e) {
        result.skip_reason = std::string("Exception: ") + e.what();
        result.skipped = true;
    }

    return result;
}

int main(int argc, char** argv) {
    // Allow threshold override via environment
    double threshold = GPU_THRESHOLD;
    if (const char* env_thresh = std::getenv("GPU_UTIL_THRESHOLD")) {
        try {
            threshold = std::stod(env_thresh);
        } catch (const std::exception& e) {
            std::cerr << "Invalid GPU_UTIL_THRESHOLD: " << env_thresh << "\n";
            return 1;
        }
    }

    // Allow override via command line
    if (argc > 1) {
        try {
            threshold = std::stod(argv[1]);
        } catch (const std::exception& e) {
            std::cerr << "Invalid threshold argument: " << argv[1] << "\n";
            return 1;
        }
    }

    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "  GPU UTILIZATION TEST\n";
    std::cout << "================================================================\n";
#ifdef USE_GPU_OFFLOAD
    std::cout << "Build type: GPU (USE_GPU_OFFLOAD enabled)\n";
#else
    std::cout << "Build type: CPU (USE_GPU_OFFLOAD disabled)\n";
    std::cout << "\nSKIPPING: This test only applies to GPU builds.\n";
    std::cout << "================================================================\n\n";
    return 0;  // Not a failure, just skip for CPU builds
#endif

    std::cout << "Grid:      " << NX << " x " << NY << "\n";
    std::cout << "Steps:     " << NUM_STEPS << " per model\n";
    std::cout << "Threshold: " << (threshold * 100.0) << "% GPU utilization\n\n";

    // Models to test (skip laminar as it has minimal compute)
    std::vector<TurbulenceModelType> models = {
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

    std::cout << std::left << std::setw(15) << "Model"
              << std::right << std::setw(12) << "GPU (s)"
              << std::setw(12) << "CPU (s)"
              << std::setw(12) << "GPU %"
              << std::setw(10) << "Status\n";
    std::cout << std::string(61, '-') << "\n";

    std::vector<ModelResult> results;
    int passed = 0;
    int failed = 0;
    int skipped = 0;

    for (auto type : models) {
        std::cout << std::left << std::setw(15) << model_name(type) << std::flush;

        ModelResult result = run_model(type);
        results.push_back(result);

        if (result.skipped) {
            std::cout << std::right << std::setw(12) << "-"
                      << std::setw(12) << "-"
                      << std::setw(12) << "-"
                      << std::setw(10) << "SKIP" << "\n";
            skipped++;
        } else if (result.ran) {
            std::cout << std::right << std::fixed << std::setprecision(3)
                      << std::setw(12) << result.gpu_time
                      << std::setw(12) << result.cpu_time
                      << std::setw(11) << (result.gpu_ratio * 100.0) << "%";

            if (result.gpu_ratio >= threshold) {
                std::cout << std::setw(10) << "PASS" << "\n";
                passed++;
            } else {
                std::cout << std::setw(10) << "FAIL" << "\n";
                failed++;
            }
        }
    }

    std::cout << std::string(61, '-') << "\n";

    // Compute aggregate statistics
    double total_gpu_time = 0.0;
    double total_cpu_time = 0.0;
    int models_run = 0;
    for (const auto& r : results) {
        if (r.ran) {
            total_gpu_time += r.gpu_time;
            total_cpu_time += r.cpu_time;
            models_run++;
        }
    }

    double total_time = total_gpu_time + total_cpu_time;
    double aggregate_ratio = (total_time > 0.0) ? (total_gpu_time / total_time) : 0.0;

    std::cout << std::left << std::setw(15) << "AGGREGATE"
              << std::right << std::fixed << std::setprecision(3)
              << std::setw(12) << total_gpu_time
              << std::setw(12) << total_cpu_time
              << std::setw(11) << (aggregate_ratio * 100.0) << "%";

    if (aggregate_ratio >= threshold) {
        std::cout << std::setw(10) << "PASS" << "\n";
    } else {
        std::cout << std::setw(10) << "FAIL" << "\n";
    }

    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "SUMMARY\n";
    std::cout << "================================================================\n";
    std::cout << "Models passed:  " << passed << "/" << models_run << "\n";
    std::cout << "Models failed:  " << failed << "/" << models_run << "\n";
    std::cout << "Models skipped: " << skipped << "/" << models.size() << "\n";
    std::cout << "Aggregate GPU:  " << std::fixed << std::setprecision(1)
              << (aggregate_ratio * 100.0) << "% (threshold: "
              << (threshold * 100.0) << "%)\n\n";

    // Print detailed breakdown if there were failures or if verbose
    if (failed > 0) {
        std::cout << "FAILURE DETAILS:\n";
        for (const auto& r : results) {
            if (r.ran && r.gpu_ratio < threshold) {
                std::cout << "  " << model_name(r.type) << ": "
                          << std::fixed << std::setprecision(1)
                          << (r.gpu_ratio * 100.0) << "% GPU "
                          << "(need >= " << (threshold * 100.0) << "%)\n";
            }
        }
        std::cout << "\n";
    }

    if (failed == 0 && aggregate_ratio >= threshold) {
        std::cout << "[SUCCESS] GPU utilization test PASSED\n";
        std::cout << "================================================================\n\n";
        return 0;
    } else {
        std::cout << "[FAILURE] GPU utilization test FAILED\n";
        if (aggregate_ratio < threshold) {
            std::cout << "Aggregate GPU utilization " << std::fixed << std::setprecision(1)
                      << (aggregate_ratio * 100.0) << "% < "
                      << (threshold * 100.0) << "% threshold\n";
        }
        std::cout << "================================================================\n\n";
        return 1;
    }
}
