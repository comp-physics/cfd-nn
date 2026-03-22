/// Test all NN turbulence models through the full solver pipeline (CPU and GPU).
/// Verifies that each model loads, runs 5 solver steps, and produces finite results.
/// Models tested: MLP, MLP-Large, TBNN, PI-TBNN, TBRF (1-tree).

#include "mesh.hpp"
#include "fields.hpp"
#include "config.hpp"
#include "solver.hpp"
#include "turbulence_baseline.hpp"
#include "turbulence_nn_mlp.hpp"
#include "turbulence_nn_tbnn.hpp"
#include "turbulence_nn_tbrf.hpp"
#include "test_harness.hpp"
#include <iostream>
#include <cmath>
#include <memory>

using namespace nncfd;
using nncfd::test::harness::record;

namespace {

std::string resolve_model_dir(const std::string& name) {
    // Try relative and absolute paths
    std::vector<std::string> paths = {
        "data/models/" + name,
        "../data/models/" + name,
        "../../data/models/" + name,
    };
    for (const auto& p : paths) {
        std::ifstream f(p + "/metadata.json");
        if (f.good()) return p;
        // TBRF uses trees.bin instead of metadata.json
        std::ifstream f2(p + "/trees.bin");
        if (f2.good()) return p;
    }
    return "";
}

/// Run a model through the solver for a few steps and check results
bool run_model_through_solver(
    const std::string& model_name,
    TurbulenceModelType model_type,
    const std::string& nn_preset)
{
    // Small 2D channel (fast)
    Mesh mesh;
    mesh.init_uniform(16, 16, 0.0, 2.0, -1.0, 1.0);

    Config config;
    config.nu = 0.01;
    config.dp_dx = -1.0;
    config.dt = 0.001;
    config.max_steps = 5;
    config.turb_model = model_type;
    config.nn_weights_path = nn_preset;
    config.nn_scaling_path = nn_preset;

    RANSSolver solver(mesh, config);
    solver.set_body_force(1.0, 0.0);

    // Create and set turbulence model
    auto turb_model = create_turbulence_model(
        model_type, nn_preset, nn_preset, 0.0, 0.0);
    if (!turb_model) {
        std::cerr << "  " << model_name << ": failed to create model\n";
        return false;
    }
    turb_model->set_nu(config.nu);

    if (auto* mlp = dynamic_cast<TurbulenceNNMLP*>(turb_model.get())) {
        mlp->set_delta(1.0);
    }
    if (auto* tbnn = dynamic_cast<TurbulenceNNTBNN*>(turb_model.get())) {
        tbnn->set_delta(1.0);
    }

    solver.set_turbulence_model(std::move(turb_model));

    // Initialize with Poiseuille-like profile
    VectorField init_vel(mesh);
    for (int j = 0; j < mesh.total_Ny(); ++j) {
        double y = mesh.y(j);
        double u = std::max(0.0, 1.0 - y * y);
        for (int i = 0; i < mesh.total_Nx(); ++i) {
            init_vel.u(i, j) = u;
        }
    }
    solver.initialize(init_vel);

    // Step the solver
    bool ok = true;
    for (int step = 0; step < 5; ++step) {
        double residual = solver.step();
        if (!std::isfinite(residual)) {
            std::cerr << "  " << model_name << ": NaN residual at step " << step << "\n";
            ok = false;
            break;
        }
    }

    return ok;
}

} // anonymous namespace

void test_model(const char* name, TurbulenceModelType type, const char* preset) {
    std::string path = resolve_model_dir(preset);
    if (path.empty()) {
        record(name, true, true);  // skip if weights not found
        return;
    }
    bool pass = run_model_through_solver(name, type, path);
    record(name, pass);
}

int main() {
    using nncfd::test::harness::run_sections;

    return run_sections("NNModels", {
        {"NN-MLP", []() {
            test_model("NN-MLP", TurbulenceModelType::NNMLP, "mlp_paper");
        }},
        {"NN-MLP-Large", []() {
            test_model("NN-MLP-Large", TurbulenceModelType::NNMLP, "mlp_large_paper");
        }},
        {"NN-TBNN", []() {
            test_model("NN-TBNN", TurbulenceModelType::NNTBNN, "tbnn_paper");
        }},
        {"NN-PI-TBNN", []() {
            test_model("NN-PI-TBNN", TurbulenceModelType::NNTBNN, "pi_tbnn_paper");
        }},
        {"NN-TBRF-1t", []() {
            test_model("NN-TBRF-1t", TurbulenceModelType::NNTBRF, "tbrf_1t_paper");
        }},
    });
}
