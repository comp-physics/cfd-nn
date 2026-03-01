/// @file test_rans_channel_validation.cpp
/// @brief RANS channel flow validation: all 10 models vs MKM DNS Re_tau=180
///
/// For each turbulence model:
///   1. Run ~800 steps on 48x48 stretched channel
///   2. Compute x-averaged U+(y+) profile
///   3. Compare against embedded MKM DNS data (L2 error)
///   4. Check profile shape (no-slip, monotonic, centerline max)
///   5. Check nu_t is reasonable

#include "test_harness.hpp"
#include "test_utilities.hpp"
#include "reference_data.hpp"
#include "solver.hpp"
#include "config.hpp"
#include "turbulence_model.hpp"
#include <cmath>
#include <vector>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <string>

using namespace nncfd;
using namespace nncfd::test;
using nncfd::test::harness::record;

// ============================================================================
// Helper: resolve NN model path
// ============================================================================
static std::string resolve_nn_path(const std::string& subdir) {
    for (const auto& prefix : {"data/models/", "../data/models/"}) {
        std::string path = std::string(prefix) + subdir;
        // Check for layer0_W.txt as existence indicator
        if (nncfd::test::file_exists(path + "/layer0_W.txt")) return path;
    }
    return "";
}

// ============================================================================
// Helper: compute x-averaged U profile in wall units
// ============================================================================
struct WallUnitProfile {
    std::vector<double> y_plus;
    std::vector<double> u_plus;
    double u_tau;
    double re_tau;
};

static WallUnitProfile compute_u_plus_profile(const RANSSolver& solver, const Mesh& mesh,
                                                double nu, double dp_dx) {
    WallUnitProfile result;

    // u_tau from body force: tau_w = delta * |dp/dx|, u_tau = sqrt(tau_w)
    double delta = (mesh.y_max - mesh.y_min) / 2.0;
    double tau_w = delta * std::abs(dp_dx);
    result.u_tau = std::sqrt(tau_w);
    result.re_tau = result.u_tau * delta / nu;

    int Ny = mesh.Ny;
    result.y_plus.resize(Ny);
    result.u_plus.resize(Ny);

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        // x-average of u at cell centers
        double u_sum = 0.0;
        int count = 0;
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            u_sum += 0.5 * (solver.velocity().u(i, j) + solver.velocity().u(i + 1, j));
            ++count;
        }
        double U = u_sum / count;

        int j_idx = j - mesh.j_begin();
        // Distance from nearest wall
        double y_center = mesh.y(j);
        double y_from_wall = std::min(y_center - mesh.y_min, mesh.y_max - y_center);
        result.y_plus[j_idx] = y_from_wall * result.u_tau / nu;
        result.u_plus[j_idx] = U / result.u_tau;
    }

    return result;
}

// ============================================================================
// Helper: compute L2 error vs MKM reference
// ============================================================================
static double compute_mkm_l2_error(const WallUnitProfile& profile) {
    const auto& ref = reference::mkm_retau180_u_profile;
    double err_sq = 0.0;
    double ref_sq = 0.0;
    int n_compared = 0;

    // For each reference point, find closest y+ in computed profile (bottom half only)
    int half_ny = static_cast<int>(profile.y_plus.size()) / 2;

    for (const auto& rp : ref) {
        // Find nearest y+ in computed profile (use bottom half: y+ increasing from wall)
        double best_dist = 1e30;
        double u_interp = 0.0;

        for (int j = 0; j < half_ny; ++j) {
            double dist = std::abs(profile.y_plus[j] - rp.y_plus);
            if (dist < best_dist) {
                best_dist = dist;
                u_interp = profile.u_plus[j];
            }
        }

        // Only compare if we found a reasonably close match (within 20% of target y+)
        if (best_dist < 0.2 * rp.y_plus + 1.0) {
            double diff = u_interp - rp.u_plus;
            err_sq += diff * diff;
            ref_sq += rp.u_plus * rp.u_plus;
            ++n_compared;
        }
    }

    if (n_compared < 5 || ref_sq < 1e-30) return 1.0;  // Not enough points
    return std::sqrt(err_sq / ref_sq);
}

// ============================================================================
// Helper: run one model and collect results
// ============================================================================
struct ModelResult {
    std::string name;
    double l2_error;
    double u_tau;
    double re_tau;
    double max_nut;
    bool no_slip;
    bool monotonic;
    bool symmetric;
    bool stable;
    bool ran_ok;
};

static ModelResult run_model(TurbulenceModelType type, const std::string& model_name,
                              const std::string& nn_path = "") {
    ModelResult result;
    result.name = model_name;
    result.ran_ok = false;

    const int Nx = 48, Ny = 48;
    const double Lx = 2.0 * M_PI;
    const double Ly = 2.0;   // y in [-1, 1]
    const double nu = 1.0 / 180.0;
    const double dp_dx = -1.0;
    const double beta = 2.0;
    const int nsteps = 800;

    try {
        Mesh mesh;
        mesh.init_stretched_y(Nx, Ny, 0.0, Lx, -Ly / 2, Ly / 2,
                              Mesh::tanh_stretching(beta));

        Config config;
        config.nu = nu;
        config.dt = 0.001;
        config.adaptive_dt = true;
        config.CFL_max = 0.5;
        config.turb_model = type;
        config.verbose = false;

        RANSSolver solver(mesh, config);

        // Create and set turbulence model
        auto turb_model = create_turbulence_model(type, nn_path, nn_path);
        solver.set_turbulence_model(std::move(turb_model));

        solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel2D));
        solver.set_body_force(-dp_dx, 0.0);
        solver.initialize_uniform(0.1, 0.0);
        solver.sync_to_gpu();

        // Run
        for (int step = 0; step < nsteps; ++step) {
            solver.step();
        }
        solver.sync_from_gpu();

        // Compute profile
        auto profile = compute_u_plus_profile(solver, mesh, nu, dp_dx);
        result.u_tau = profile.u_tau;
        result.re_tau = profile.re_tau;

        // L2 error vs MKM
        result.l2_error = compute_mkm_l2_error(profile);

        // No-slip check: U at first and last cells should be small
        result.no_slip = (std::abs(profile.u_plus.front()) < 2.0 &&
                          std::abs(profile.u_plus.back()) < 2.0);

        // Monotonic check (bottom half: wall to center)
        int half = Ny / 2;
        result.monotonic = true;
        for (int j = 1; j < half - 1; ++j) {
            if (profile.u_plus[j] < profile.u_plus[j - 1] - 0.5) {
                result.monotonic = false;
                break;
            }
        }

        // Symmetry: U(j) ~ U(Ny-1-j)
        result.symmetric = true;
        for (int j = 0; j < half; ++j) {
            double diff = std::abs(profile.u_plus[j] - profile.u_plus[Ny - 1 - j]);
            if (diff > 1.0) {  // Allow 1 wall unit of asymmetry
                result.symmetric = false;
                break;
            }
        }

        // Max nu_t
        result.max_nut = 0.0;
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                result.max_nut = std::max(result.max_nut, solver.nu_t()(i, j));
            }
        }

        result.stable = std::isfinite(result.l2_error) && result.l2_error < 1.0;
        result.ran_ok = true;

    } catch (const std::exception& e) {
        std::cerr << "  [ERROR] " << model_name << " failed: " << e.what() << "\n";
        result.l2_error = 1.0;
        result.max_nut = 0.0;
        result.no_slip = false;
        result.monotonic = false;
        result.symmetric = false;
        result.stable = false;
    }

    return result;
}

// ============================================================================
// Section 1: Algebraic models (None, Baseline, GEP)
// ============================================================================
void test_algebraic_models() {
    std::cout << "\n--- Algebraic Models vs MKM ---\n\n";

    // None (DNS-like, no model)
    auto none = run_model(TurbulenceModelType::None, "None");
    std::cout << "  None:     L2=" << std::fixed << std::setprecision(3) << none.l2_error
              << " u_tau=" << std::setprecision(4) << none.u_tau << "\n";
    record("None: ran without error", none.ran_ok);
    record("None: profile stable", none.stable);

    // Baseline (mixing length)
    auto baseline = run_model(TurbulenceModelType::Baseline, "Baseline");
    std::cout << "  Baseline: L2=" << std::fixed << std::setprecision(3) << baseline.l2_error
              << " u_tau=" << std::setprecision(4) << baseline.u_tau
              << " max_nut=" << std::scientific << baseline.max_nut << "\n";
    record("Baseline: L2 error < 25%", baseline.l2_error < 0.25);
    record("Baseline: no-slip", baseline.no_slip);
    record("Baseline: monotonic", baseline.monotonic);
    record("Baseline: nu_t > 0", baseline.max_nut > 0.0);

    // GEP
    auto gep = run_model(TurbulenceModelType::GEP, "GEP");
    std::cout << "  GEP:      L2=" << std::fixed << std::setprecision(3) << gep.l2_error
              << " u_tau=" << std::setprecision(4) << gep.u_tau
              << " max_nut=" << std::scientific << gep.max_nut << "\n";
    record("GEP: L2 error < 30%", gep.l2_error < 0.30);
    record("GEP: profile stable", gep.stable);

    std::cout << "\n";
}

// ============================================================================
// Section 2: Transport models (SST, KOmega)
// ============================================================================
void test_transport_models() {
    std::cout << "\n--- Transport Models vs MKM ---\n\n";

    auto sst = run_model(TurbulenceModelType::SSTKOmega, "SST k-omega");
    std::cout << "  SST:      L2=" << std::fixed << std::setprecision(3) << sst.l2_error
              << " u_tau=" << std::setprecision(4) << sst.u_tau
              << " max_nut=" << std::scientific << sst.max_nut << "\n";
    record("SST: L2 error < 25%", sst.l2_error < 0.25);
    record("SST: no-slip", sst.no_slip);
    record("SST: monotonic", sst.monotonic);
    record("SST: nu_t > 0", sst.max_nut > 0.0);

    auto komega = run_model(TurbulenceModelType::KOmega, "k-omega");
    std::cout << "  k-omega:  L2=" << std::fixed << std::setprecision(3) << komega.l2_error
              << " u_tau=" << std::setprecision(4) << komega.u_tau
              << " max_nut=" << std::scientific << komega.max_nut << "\n";
    record("k-omega: L2 error < 30%", komega.l2_error < 0.30);
    record("k-omega: profile stable", komega.stable);

    std::cout << "\n";
}

// ============================================================================
// Section 3: EARSM models
// ============================================================================
void test_earsm_models() {
    std::cout << "\n--- EARSM Models vs MKM ---\n\n";

    auto wj = run_model(TurbulenceModelType::EARSM_WJ, "EARSM-WJ");
    std::cout << "  EARSM-WJ: L2=" << std::fixed << std::setprecision(3) << wj.l2_error
              << " u_tau=" << std::setprecision(4) << wj.u_tau << "\n";
    record("EARSM-WJ: L2 error < 30%", wj.l2_error < 0.30);
    record("EARSM-WJ: stable", wj.stable);

    auto gs = run_model(TurbulenceModelType::EARSM_GS, "EARSM-GS");
    std::cout << "  EARSM-GS: L2=" << std::fixed << std::setprecision(3) << gs.l2_error
              << " u_tau=" << std::setprecision(4) << gs.u_tau << "\n";
    record("EARSM-GS: L2 error < 30%", gs.l2_error < 0.30);
    record("EARSM-GS: stable", gs.stable);

    auto pope = run_model(TurbulenceModelType::EARSM_Pope, "EARSM-Pope");
    std::cout << "  EARSM-Pope: L2=" << std::fixed << std::setprecision(3) << pope.l2_error
              << " u_tau=" << std::setprecision(4) << pope.u_tau << "\n";
    record("EARSM-Pope: L2 error < 30%", pope.l2_error < 0.30);
    record("EARSM-Pope: stable", pope.stable);

    std::cout << "\n";
}

// ============================================================================
// Section 4: Neural network models
// ============================================================================
void test_nn_models() {
    std::cout << "\n--- Neural Network Models vs MKM ---\n\n";

    // MLP
    std::string mlp_path = resolve_nn_path("mlp_channel_caseholdout");
    if (mlp_path.empty()) {
        std::cout << "  [SKIP] MLP weights not found\n";
        record("MLP: weights found", false, true);  // skip
    } else {
        auto mlp = run_model(TurbulenceModelType::NNMLP, "NN-MLP", mlp_path);
        std::cout << "  NN-MLP:   L2=" << std::fixed << std::setprecision(3) << mlp.l2_error
                  << " u_tau=" << std::setprecision(4) << mlp.u_tau
                  << " max_nut=" << std::scientific << mlp.max_nut << "\n";
        record("MLP: L2 error < 30%", mlp.l2_error < 0.30);
        record("MLP: stable", mlp.stable);
    }

    // TBNN
    std::string tbnn_path = resolve_nn_path("tbnn_channel_caseholdout");
    if (tbnn_path.empty()) {
        std::cout << "  [SKIP] TBNN weights not found\n";
        record("TBNN: weights found", false, true);  // skip
    } else {
        auto tbnn = run_model(TurbulenceModelType::NNTBNN, "NN-TBNN", tbnn_path);
        std::cout << "  NN-TBNN:  L2=" << std::fixed << std::setprecision(3) << tbnn.l2_error
                  << " u_tau=" << std::setprecision(4) << tbnn.u_tau << "\n";
        record("TBNN: L2 error < 30%", tbnn.l2_error < 0.30);
        record("TBNN: stable", tbnn.stable);
    }

    std::cout << "\n";
}

// ============================================================================
// Section 5: Cross-model comparison summary
// ============================================================================
void test_model_comparison() {
    std::cout << "\n--- Model Comparison Summary ---\n\n";

    // Run all models and print table
    struct Entry { std::string name; TurbulenceModelType type; std::string nn; };
    std::vector<Entry> models = {
        {"None",       TurbulenceModelType::None, ""},
        {"Baseline",   TurbulenceModelType::Baseline, ""},
        {"GEP",        TurbulenceModelType::GEP, ""},
        {"SST",        TurbulenceModelType::SSTKOmega, ""},
        {"k-omega",    TurbulenceModelType::KOmega, ""},
        {"EARSM-WJ",   TurbulenceModelType::EARSM_WJ, ""},
        {"EARSM-GS",   TurbulenceModelType::EARSM_GS, ""},
        {"EARSM-Pope", TurbulenceModelType::EARSM_Pope, ""},
    };

    // Add NN models if weights found
    std::string mlp_path = resolve_nn_path("mlp_channel_caseholdout");
    if (!mlp_path.empty()) models.push_back({"NN-MLP", TurbulenceModelType::NNMLP, mlp_path});
    std::string tbnn_path = resolve_nn_path("tbnn_channel_caseholdout");
    if (!tbnn_path.empty()) models.push_back({"NN-TBNN", TurbulenceModelType::NNTBNN, tbnn_path});

    std::cout << std::left << std::setw(14) << "  Model"
              << std::right << std::setw(8) << "L2_err"
              << std::setw(10) << "max_nut"
              << std::setw(8) << "shape" << "\n";
    std::cout << "  " << std::string(38, '-') << "\n";

    int n_ran = 0;
    int n_stable = 0;

    for (const auto& m : models) {
        auto r = run_model(m.type, m.name, m.nn);
        if (r.ran_ok) {
            ++n_ran;
            if (r.stable) ++n_stable;
        }

        std::string shape = (r.no_slip && r.monotonic && r.symmetric) ? "OK" : "WARN";
        std::cout << "  " << std::left << std::setw(14) << m.name
                  << std::right << std::setw(7) << std::fixed << std::setprecision(3) << r.l2_error
                  << std::setw(10) << std::scientific << std::setprecision(1) << r.max_nut
                  << std::setw(8) << shape << "\n";
    }

    std::cout << "\n  Models ran: " << n_ran << "/" << models.size()
              << ", stable: " << n_stable << "/" << n_ran << "\n\n";

    record("All models ran", n_ran == static_cast<int>(models.size()));
    record(">=80% models stable", n_stable >= static_cast<int>(0.8 * n_ran));
}

// ============================================================================
// Main
// ============================================================================
int main() {
    return harness::run_sections("RANSChannelValidation", {
        {"Algebraic models", test_algebraic_models},
        {"Transport models", test_transport_models},
        {"EARSM models", test_earsm_models},
        {"Neural network models", test_nn_models},
        {"Model comparison summary", test_model_comparison},
    });
}
