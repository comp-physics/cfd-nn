/// Cross-Geometry Turbulence Model Tests
/// Tests all 10 turbulence closures across 3D channel, duct, and TGV geometries.
/// Complements test_turbulence_unified.cpp (2D channel only) with 3D coverage.

#include "test_harness.hpp"
#include "test_utilities.hpp"
#include "turbulence_model.hpp"
#include "turbulence_baseline.hpp"
#include "turbulence_gep.hpp"
#include "turbulence_earsm.hpp"
#include <cmath>

using namespace nncfd;
using nncfd::test::harness::record;
using nncfd::test::BCPattern;
using nncfd::test::create_velocity_bc;

// ============================================================================
// Helpers (shared across sections)
// ============================================================================

static std::string resolve_nn_path(const std::string& subdir) {
    for (const auto& prefix : {"data/models/", "../data/models/"}) {
        std::string path = prefix + subdir;
        if (nncfd::test::file_exists(path + "/layer0_W.txt")) return path;
    }
    return "";
}

static std::string model_name(TurbulenceModelType type) {
    switch (type) {
        case TurbulenceModelType::None:       return "Laminar";
        case TurbulenceModelType::Baseline:   return "Baseline";
        case TurbulenceModelType::GEP:        return "GEP";
        case TurbulenceModelType::NNMLP:      return "NN-MLP";
        case TurbulenceModelType::NNTBNN:     return "NN-TBNN";
        case TurbulenceModelType::SSTKOmega:  return "SST k-omega";
        case TurbulenceModelType::KOmega:     return "k-omega";
        case TurbulenceModelType::EARSM_WJ:   return "EARSM-WJ";
        case TurbulenceModelType::EARSM_GS:   return "EARSM-GS";
        case TurbulenceModelType::EARSM_Pope: return "EARSM-Pope";
        default: return "Unknown";
    }
}

static bool is_transport_model(TurbulenceModelType type) {
    return type == TurbulenceModelType::SSTKOmega || type == TurbulenceModelType::KOmega ||
           type == TurbulenceModelType::EARSM_WJ || type == TurbulenceModelType::EARSM_GS ||
           type == TurbulenceModelType::EARSM_Pope;
}

static bool is_nn_model(TurbulenceModelType type) {
    return type == TurbulenceModelType::NNMLP || type == TurbulenceModelType::NNTBNN;
}

static const std::vector<TurbulenceModelType> ALL_MODELS = {
    TurbulenceModelType::None, TurbulenceModelType::Baseline,
    TurbulenceModelType::GEP, TurbulenceModelType::SSTKOmega,
    TurbulenceModelType::KOmega, TurbulenceModelType::EARSM_WJ,
    TurbulenceModelType::EARSM_GS, TurbulenceModelType::EARSM_Pope,
    TurbulenceModelType::NNMLP, TurbulenceModelType::NNTBNN
};

struct SmokeResult { bool passed = false, skipped = false; std::string message; };

/// Run a 3D smoke test for a single model on a given geometry
static SmokeResult run_3d_smoke(TurbulenceModelType type, const Mesh& mesh,
                                 BCPattern bc, int num_steps,
                                 double body_force_x = 0.001) {
    SmokeResult result;

    std::string nn_path;
    if (type == TurbulenceModelType::NNMLP) {
        nn_path = resolve_nn_path("mlp_paper");
        if (nn_path.empty()) { result.skipped = true; result.message = "MLP weights not found"; return result; }
    } else if (type == TurbulenceModelType::NNTBNN) {
        nn_path = resolve_nn_path("tbnn_paper");
        if (nn_path.empty()) { result.skipped = true; result.message = "TBNN weights not found"; return result; }
    }

    try {
        Config config;
        config.nu = 0.001;
        config.dt = 0.001;
        config.turb_model = type;
        config.verbose = false;
        config.turb_guard_enabled = true;
        if (!nn_path.empty()) { config.nn_weights_path = nn_path; config.nn_scaling_path = nn_path; }

        RANSSolver solver(mesh, config);
        solver.set_body_force(body_force_x, 0.0, 0.0);
        solver.set_velocity_bc(create_velocity_bc(bc));
        if (type != TurbulenceModelType::None)
            solver.set_turbulence_model(create_turbulence_model(type, nn_path, nn_path));

        // initialize_uniform sets k/omega for transport models
        solver.initialize_uniform(1.0, 0.0);

        // Override with a simple profile
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k)
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j)
                for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                    double y_norm = (mesh.y(j) - mesh.y_min) / (mesh.y_max - mesh.y_min);
                    solver.velocity().u(i, j, k) = 0.1 * 4.0 * y_norm * (1.0 - y_norm);
                }
        solver.sync_to_gpu();

        for (int step = 0; step < num_steps; ++step) solver.step();
        solver.sync_from_gpu();

        const auto& vel = solver.velocity();
        const auto& nu_t = solver.nu_t();

        FOR_INTERIOR_3D(mesh, i, j, k) {
            if (!std::isfinite(vel.u(i, j, k)) || !std::isfinite(vel.v(i, j, k)) ||
                !std::isfinite(vel.w(i, j, k)))
                { result.message = "NaN/Inf in velocity"; return result; }
            if (std::abs(vel.u(i, j, k)) > 100.0 || std::abs(vel.v(i, j, k)) > 100.0 ||
                std::abs(vel.w(i, j, k)) > 100.0)
                { result.message = "Velocity exceeds bound (|u|>100)"; return result; }
            if (!std::isfinite(nu_t(i, j, k)) || nu_t(i, j, k) < 0.0)
                { result.message = "Invalid nu_t"; return result; }
        }

        if (is_transport_model(type)) {
            const auto& k_field = solver.k();
            const auto& omega_field = solver.omega();
            FOR_INTERIOR_3D(mesh, i, j, k) {
                if (!std::isfinite(k_field(i, j, k)) || k_field(i, j, k) < 1e-12)
                    { result.message = "Invalid k"; return result; }
                if (!std::isfinite(omega_field(i, j, k)) || omega_field(i, j, k) < 1e-12)
                    { result.message = "Invalid omega"; return result; }
            }
        }

        // Check divergence
        double max_div = 0.0;
        FOR_INTERIOR_3D(mesh, i, j, k) {
            double dudx = (vel.u(i+1,j,k) - vel.u(i,j,k)) / mesh.dx;
            double dvdy = (vel.v(i,j+1,k) - vel.v(i,j,k)) / mesh.dy;
            double dwdz = (vel.w(i,j,k+1) - vel.w(i,j,k)) / mesh.dz;
            max_div = std::max(max_div, std::abs(dudx + dvdy + dwdz));
        }
        if (max_div > 1e-4) {
            result.message = "Divergence too large: " + std::to_string(max_div);
            return result;
        }

        result.passed = true;
        result.message = "OK";
    } catch (const std::exception& e) {
        result.message = std::string("Exception: ") + e.what();
    }
    return result;
}

// ============================================================================
// Section 1: 3D Channel Smoke — all 10 models, 50 steps
// ============================================================================

static void test_3d_channel_smoke() {
    Mesh mesh;
    mesh.init_uniform(16, 16, 16, 0.0, 2.0, -1.0, 1.0, 0.0, 2.0);

    for (auto type : ALL_MODELS) {
        auto result = run_3d_smoke(type, mesh, BCPattern::Channel3D, 50);
        record(("3D-Channel: " + model_name(type)).c_str(), result.passed, result.skipped);
        if (!result.passed && !result.skipped)
            std::cerr << "  FAIL detail: " << result.message << "\n";
    }
}

// ============================================================================
// Section 2: 3D Duct Smoke — all 10 models, 50 steps
// ============================================================================

static void test_3d_duct_smoke() {
    Mesh mesh;
    mesh.init_uniform(16, 16, 16, 0.0, 2.0, -1.0, 1.0, -1.0, 1.0);

    for (auto type : ALL_MODELS) {
        auto result = run_3d_smoke(type, mesh, BCPattern::Duct, 50);
        record(("3D-Duct: " + model_name(type)).c_str(), result.passed, result.skipped);
        if (!result.passed && !result.skipped)
            std::cerr << "  FAIL detail: " << result.message << "\n";
    }
}

// ============================================================================
// Section 3: 3D TGV Smoke — 8 models (skip NN), 50 steps
// ============================================================================

static void test_3d_tgv_smoke() {
    double L = 2.0 * M_PI;
    Mesh mesh;
    mesh.init_uniform(16, 16, 16, 0.0, L, 0.0, L, 0.0, L);

    for (auto type : ALL_MODELS) {
        // Skip NN models — trained on channel, meaningless on wall-free flow
        if (is_nn_model(type)) continue;

        std::string label = "3D-TGV: " + model_name(type);

        try {
            Config config;
            config.nu = 0.01;
            config.dt = 0.005;
            config.turb_model = type;
            config.verbose = false;
            config.turb_guard_enabled = true;

            RANSSolver solver(mesh, config);
            solver.set_velocity_bc(create_velocity_bc(BCPattern::FullyPeriodic));
            if (type != TurbulenceModelType::None)
                solver.set_turbulence_model(create_turbulence_model(type));
            solver.initialize_uniform(0.0, 0.0);

            // TGV initial condition (overwrites velocity from initialize_uniform)
            for (int k = mesh.k_begin(); k < mesh.k_end(); ++k)
                for (int j = mesh.j_begin(); j < mesh.j_end(); ++j)
                    for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i)
                        solver.velocity().u(i, j, k) =
                            std::sin(mesh.xf[i]) * std::cos(mesh.y(j)) * std::cos(mesh.z(k));

            for (int k = mesh.k_begin(); k < mesh.k_end(); ++k)
                for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j)
                    for (int i = mesh.i_begin(); i < mesh.i_end(); ++i)
                        solver.velocity().v(i, j, k) =
                            -std::cos(mesh.x(i)) * std::sin(mesh.yf[j]) * std::cos(mesh.z(k));

            // w = 0 (already initialized)
            solver.sync_to_gpu();

            double E_initial = solver.compute_kinetic_energy();

            for (int step = 0; step < 50; ++step) solver.step();
            solver.sync_from_gpu();

            double E_final = solver.compute_kinetic_energy();

            // Check: no NaN/Inf, bounded velocity
            bool ok = true;
            FOR_INTERIOR_3D(mesh, i, j, k) {
                if (!std::isfinite(solver.velocity().u(i, j, k)) ||
                    !std::isfinite(solver.velocity().v(i, j, k)) ||
                    !std::isfinite(solver.velocity().w(i, j, k))) {
                    ok = false;
                    break;
                }
            }

            // Energy should decay (unforced viscous TGV has no energy input)
            ok = ok && (E_final <= E_initial * 1.01);  // Small tolerance for numerics

            record(label.c_str(), ok);
            if (!ok)
                std::cerr << "  E_initial=" << E_initial << ", E_final=" << E_final << "\n";
        } catch (const std::exception& e) {
            record(label.c_str(), false);
            std::cerr << "  Exception: " << e.what() << "\n";
        }
    }
}

// ============================================================================
// Section 4: DNS Combo — turb_model=None + trip + filter, 150 steps
// ============================================================================

static void test_dns_combo() {
    Mesh mesh;
    mesh.init_uniform(16, 16, 16, 0.0, 2.0, -1.0, 1.0, 0.0, 2.0);

    try {
        Config config;
        config.nu = 0.001;
        config.turb_model = TurbulenceModelType::None;
        config.verbose = false;
        config.convective_scheme = ConvectiveScheme::Skew;
        config.time_integrator = TimeIntegrator::RK3;
        config.adaptive_dt = true;
        config.CFL_max = 0.3;
        config.dt = 0.001;

        // Trip forcing
        config.trip_enabled = true;
        config.trip_amplitude = 1.0;
        config.trip_duration = 0.10;
        config.trip_force_w = true;
        config.trip_w_scale = 2.0;

        // Velocity filter
        config.filter_strength = 0.03;
        config.filter_interval = 2;

        RANSSolver solver(mesh, config);
        solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel3D));
        solver.set_body_force(0.001, 0.0, 0.0);

        // Poiseuille + perturbation
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k)
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j)
                for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                    double y = mesh.y(j);
                    solver.velocity().u(i, j, k) = 0.5 * (1.0 - y * y);
                }
        solver.sync_to_gpu();

        for (int step = 0; step < 150; ++step) solver.step();
        solver.sync_from_gpu();

        // Check: no NaN, bounded velocity, divergence
        bool ok = true;
        double max_vel = 0.0;
        const auto& vel = solver.velocity();
        FOR_INTERIOR_3D(mesh, i, j, k) {
            if (!std::isfinite(vel.u(i, j, k)) || !std::isfinite(vel.v(i, j, k)) ||
                !std::isfinite(vel.w(i, j, k))) {
                ok = false;
                break;
            }
            max_vel = std::max(max_vel, std::abs(vel.u(i, j, k)));
            max_vel = std::max(max_vel, std::abs(vel.v(i, j, k)));
            max_vel = std::max(max_vel, std::abs(vel.w(i, j, k)));
        }

        double max_div = 0.0;
        FOR_INTERIOR_3D(mesh, i, j, k) {
            double dudx = (vel.u(i+1,j,k) - vel.u(i,j,k)) / mesh.dx;
            double dvdy = (vel.v(i,j+1,k) - vel.v(i,j,k)) / mesh.dy;
            double dwdz = (vel.w(i,j,k+1) - vel.w(i,j,k)) / mesh.dz;
            max_div = std::max(max_div, std::abs(dudx + dvdy + dwdz));
        }

        record("DNS combo: no NaN", ok);
        record("DNS combo: velocity bounded", ok && max_vel < 100.0);
        record("DNS combo: divergence", ok && max_div < 1e-4);
    } catch (const std::exception& e) {
        record("DNS combo: no NaN", false);
        record("DNS combo: velocity bounded", false);
        record("DNS combo: divergence", false);
        std::cerr << "  DNS combo exception: " << e.what() << "\n";
    }
}

// ============================================================================
// Section 5: Transport Realizability 3D — 5 transport models, 200 steps
// ============================================================================

static void test_transport_realizability_3d() {
    Mesh mesh;
    mesh.init_uniform(16, 16, 16, 0.0, 2.0, -1.0, 1.0, 0.0, 2.0);

    std::vector<TurbulenceModelType> transport_models = {
        TurbulenceModelType::SSTKOmega, TurbulenceModelType::KOmega,
        TurbulenceModelType::EARSM_WJ, TurbulenceModelType::EARSM_GS,
        TurbulenceModelType::EARSM_Pope
    };

    for (auto type : transport_models) {
        auto result = run_3d_smoke(type, mesh, BCPattern::Channel3D, 200);
        record(("Realizability-3D: " + model_name(type)).c_str(), result.passed, result.skipped);
        if (!result.passed && !result.skipped)
            std::cerr << "  FAIL detail: " << result.message << "\n";
    }
}

// ============================================================================
// Section 6: EARSM Trace-Free — 3 EARSM models, direct model update
// ============================================================================

static void test_earsm_trace_free() {
    // EARSM implementation uses a 2D tensor basis. Verify trace-free property
    // with a different mesh/IC than test_turbulence_unified (Poiseuille profile).
    Mesh mesh;
    mesh.init_uniform(16, 16, 0.0, 2.0, -1.0, 1.0);

    VectorField vel(mesh);
    for (int j = 0; j < mesh.total_Ny(); ++j)
        for (int i = 0; i < mesh.total_Nx(); ++i) {
            vel.u(i, j) = 0.5 * (1.0 - mesh.y(j) * mesh.y(j));  // Poiseuille
            vel.v(i, j) = 0.0;
        }

    ScalarField k_field(mesh, 0.05), omega_field(mesh, 20.0), nu_t(mesh);

    std::vector<EARSMType> types = {
        EARSMType::WallinJohansson2000, EARSMType::GatskiSpeziale1993, EARSMType::Pope1975
    };
    std::vector<std::string> names = {"EARSM-WJ", "EARSM-GS", "EARSM-Pope"};

    for (size_t idx = 0; idx < types.size(); ++idx) {
        std::string label = "Trace-free EARSM: " + names[idx];
        try {
            TensorField tau_ij(mesh);
            SSTWithEARSM model(types[idx]);
            model.set_nu(0.001);
            model.set_delta(1.0);
            model.initialize(mesh, vel);
            model.update(mesh, vel, k_field, omega_field, nu_t, &tau_ij);

            double max_trace_err = 0.0;
            FOR_INTERIOR_2D(mesh, i, j) {
                if (k_field(i, j) < 1e-10) continue;
                // Full 3D trace: xx + yy + zz (deviatoric uses 1/3 in 3D basis)
                double tau_trace = tau_ij.xx(i, j) + tau_ij.yy(i, j) + tau_ij.zz(i, j, 0);
                double b_trace = tau_trace / (2.0 * k_field(i, j)) - 1.0;
                max_trace_err = std::max(max_trace_err, std::abs(b_trace));
            }

            record(label.c_str(), max_trace_err < 1e-8);
            if (max_trace_err >= 1e-8)
                std::cerr << "  " << names[idx] << " max trace error: " << max_trace_err << "\n";
        } catch (const std::exception& e) {
            record(label.c_str(), false);
            std::cerr << "  " << names[idx] << " exception: " << e.what() << "\n";
        }
    }
}

// ============================================================================
// Section 7: Cross-Geometry Consistency — Baseline + SST across 3 geometries
// ============================================================================

static void test_cross_geometry_consistency() {
    std::vector<TurbulenceModelType> models = {
        TurbulenceModelType::Baseline, TurbulenceModelType::SSTKOmega
    };

    struct GeomSetup {
        std::string name;
        BCPattern bc;
        double x_min, x_max, y_min, y_max, z_min, z_max;
    };

    std::vector<GeomSetup> geometries = {
        {"Channel", BCPattern::Channel3D, 0.0, 2.0, -1.0, 1.0, 0.0, 2.0},
        {"Duct",    BCPattern::Duct,      0.0, 2.0, -1.0, 1.0, -1.0, 1.0},
        {"TGV",     BCPattern::FullyPeriodic, 0.0, 2.0*M_PI, 0.0, 2.0*M_PI, 0.0, 2.0*M_PI}
    };

    for (auto type : models) {
        for (const auto& geom : geometries) {
            Mesh mesh;
            mesh.init_uniform(16, 16, 16, geom.x_min, geom.x_max,
                              geom.y_min, geom.y_max, geom.z_min, geom.z_max);

            std::string label = "Consistency: " + model_name(type) + " / " + geom.name;

            try {
                Config config;
                config.nu = 0.001;
                config.dt = 0.001;
                config.turb_model = type;
                config.verbose = false;

                RANSSolver solver(mesh, config);
                solver.set_velocity_bc(create_velocity_bc(geom.bc));
                solver.set_turbulence_model(create_turbulence_model(type));
                solver.initialize_uniform(0.1, 0.0);

                if (geom.bc != BCPattern::FullyPeriodic) {
                    solver.set_body_force(0.001, 0.0, 0.0);
                    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k)
                        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j)
                            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                                double y_norm = (mesh.y(j) - mesh.y_min) / (mesh.y_max - mesh.y_min);
                                solver.velocity().u(i, j, k) = 0.1 * 4.0 * y_norm * (1.0 - y_norm);
                            }
                } else {
                    // TGV init
                    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k)
                        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j)
                            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i)
                                solver.velocity().u(i, j, k) =
                                    0.1 * std::sin(mesh.xf[i]) * std::cos(mesh.y(j));
                }
                solver.sync_to_gpu();

                for (int step = 0; step < 50; ++step) solver.step();
                solver.sync_from_gpu();

                // Check nu_t is finite and non-negative everywhere
                bool ok = true;
                const auto& nut = solver.nu_t();
                FOR_INTERIOR_3D(mesh, i, j, k) {
                    if (!std::isfinite(nut(i, j, k)) || nut(i, j, k) < 0.0) {
                        ok = false;
                        break;
                    }
                }

                record(label.c_str(), ok);
            } catch (const std::exception& e) {
                record(label.c_str(), false);
                std::cerr << "  Exception: " << e.what() << "\n";
            }
        }
    }
}

// ============================================================================
// Section 8: Model Ordering — turbulent models diffuse more than laminar
// ============================================================================

/// Run a model on 2D channel and return (u_max, mean_nu_t, bulk_velocity).
/// Uses 2D mesh because turbulence model CPU paths are 2D-only.
struct ModelMetrics {
    double u_max = 0.0;
    double mean_nu_t = 0.0;
    double bulk_u = 0.0;
    bool ok = false;
};

static ModelMetrics run_model_metrics(TurbulenceModelType type, int num_steps = 100) {
    ModelMetrics m;
    Mesh mesh;
    mesh.init_uniform(16, 32, 0.0, 2.0, -1.0, 1.0);

    try {
        Config config;
        config.nu = 0.001;
        config.dt = 0.001;
        config.turb_model = type;
        config.verbose = false;

        RANSSolver solver(mesh, config);
        solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel2D));
        solver.set_body_force(0.01, 0.0);
        if (type != TurbulenceModelType::None)
            solver.set_turbulence_model(create_turbulence_model(type));
        solver.initialize_uniform(1.0, 0.0);

        // Poiseuille IC
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j)
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                double y = mesh.y(j);
                solver.velocity().u(i, j) = 0.5 * (1.0 - y * y);
            }
        solver.sync_to_gpu();

        for (int step = 0; step < num_steps; ++step) solver.step();
        solver.sync_from_gpu();

        const auto& vel = solver.velocity();
        const auto& nu_t = solver.nu_t();
        double sum_u = 0.0, sum_nut = 0.0;
        int count = 0;

        FOR_INTERIOR_2D(mesh, i, j) {
            double u_cc = 0.5 * (vel.u(i, j) + vel.u(i + 1, j));
            m.u_max = std::max(m.u_max, u_cc);
            sum_u += u_cc;
            sum_nut += nu_t(i, j);
            ++count;
        }
        m.bulk_u = sum_u / count;
        m.mean_nu_t = sum_nut / count;
        m.ok = std::isfinite(m.u_max) && std::isfinite(m.bulk_u);
    } catch (const std::exception& e) {
        m.ok = false;
        std::cerr << "  run_model_metrics(" << model_name(type) << ") exception: "
                  << e.what() << "\n";
    }
    return m;
}

static void test_model_ordering() {
    auto laminar  = run_model_metrics(TurbulenceModelType::None);
    auto baseline = run_model_metrics(TurbulenceModelType::Baseline);
    auto sst      = run_model_metrics(TurbulenceModelType::SSTKOmega);

    record("Ordering: all runs succeeded",
           laminar.ok && baseline.ok && sst.ok);

    if (!laminar.ok || !baseline.ok || !sst.ok) return;

    // Turbulence adds diffusion → flattens profile → lower u_max
    record("Ordering: Baseline u_max < Laminar u_max",
           baseline.u_max < laminar.u_max);
    std::cerr << "  Laminar u_max=" << laminar.u_max
              << ", Baseline u_max=" << baseline.u_max
              << ", SST u_max=" << sst.u_max << "\n";

    // Turbulent models should have nonzero eddy viscosity
    record("Ordering: Laminar nu_t == 0", laminar.mean_nu_t < 1e-15);
    record("Ordering: Baseline nu_t > 0", baseline.mean_nu_t > 1e-10);
    record("Ordering: SST nu_t > 0",      sst.mean_nu_t > 1e-10);

    std::cerr << "  Laminar nu_t=" << laminar.mean_nu_t
              << ", Baseline nu_t=" << baseline.mean_nu_t
              << ", SST nu_t=" << sst.mean_nu_t << "\n";
}

// ============================================================================
// Section 9: Transport Profile Shape — k/omega/nu_t spatial structure
// ============================================================================

static void test_transport_profile_shape() {
    // Use 2D mesh — turbulence model CPU paths are 2D-only
    try {
        Mesh mesh;
        mesh.init_uniform(16, 32, 0.0, 2.0, -1.0, 1.0);

        Config config;
        config.nu = 0.001;
        config.dt = 0.001;
        config.turb_model = TurbulenceModelType::SSTKOmega;
        config.verbose = false;

        RANSSolver solver(mesh, config);
        solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel2D));
        solver.set_body_force(0.01, 0.0);
        solver.set_turbulence_model(create_turbulence_model(TurbulenceModelType::SSTKOmega));
        solver.initialize_uniform(1.0, 0.0);

        // Poiseuille IC
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j)
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                double y = mesh.y(j);
                solver.velocity().u(i, j) = 0.5 * (1.0 - y * y);
            }
        solver.sync_to_gpu();

        for (int step = 0; step < 200; ++step) solver.step();
        solver.sync_from_gpu();

        const auto& k_field = solver.k();
        const auto& omega_field = solver.omega();
        const auto& nu_t_field = solver.nu_t();

        // Build x-averaged profiles: k(y), omega(y), nu_t(y)
        int Ny = mesh.Ny;
        std::vector<double> k_prof(Ny, 0.0), omega_prof(Ny, 0.0), nut_prof(Ny, 0.0);
        for (int jj = 0; jj < Ny; ++jj) {
            int j = mesh.j_begin() + jj;
            double sum_k = 0.0, sum_om = 0.0, sum_nut = 0.0;
            int cnt = 0;
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                sum_k   += k_field(i, j);
                sum_om  += omega_field(i, j);
                sum_nut += nu_t_field(i, j);
                ++cnt;
            }
            k_prof[jj]     = sum_k / cnt;
            omega_prof[jj]  = sum_om / cnt;
            nut_prof[jj]    = sum_nut / cnt;
        }

        // Check 1: k > 0 everywhere in interior
        bool k_positive = true;
        for (int jj = 0; jj < Ny; ++jj)
            if (k_prof[jj] < 1e-12) k_positive = false;
        record("Profile: k > 0 throughout", k_positive);

        // Check 2: omega at walls > omega at center
        double omega_wall = std::max(omega_prof[0], omega_prof[Ny - 1]);
        double omega_center = omega_prof[Ny / 2];
        record("Profile: omega_wall > omega_center",
               omega_wall > omega_center);
        std::cerr << "  omega_wall=" << omega_wall
                  << ", omega_center=" << omega_center << "\n";

        // Check 3: nu_t at walls should be small relative to interior
        double nut_wall = std::min(nut_prof[0], nut_prof[Ny - 1]);
        double nut_max = *std::max_element(nut_prof.begin(), nut_prof.end());
        record("Profile: nu_t_wall < nu_t_max",
               nut_wall < nut_max);
        record("Profile: nu_t_wall < 50% of nu_t_max",
               nut_wall < 0.5 * nut_max);
        std::cerr << "  nu_t_wall=" << nut_wall << ", nu_t_max=" << nut_max << "\n";

        // Check 4: velocity profile is roughly parabolic (u_center > u_wall)
        const auto& vel = solver.velocity();
        double u_center = 0.0, u_wall_val = 0.0;
        int j_center = mesh.j_begin() + Ny / 2;
        int j_wall = mesh.j_begin();
        int ucnt = 0;
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            u_center += 0.5 * (vel.u(i, j_center) + vel.u(i + 1, j_center));
            u_wall_val += 0.5 * (vel.u(i, j_wall) + vel.u(i + 1, j_wall));
            ++ucnt;
        }
        u_center /= ucnt;
        u_wall_val /= ucnt;
        record("Profile: u_center > u_wall", u_center > u_wall_val);

        // Check 5: k profile should have peak away from walls (not monotonic)
        int k_peak_idx = static_cast<int>(
            std::max_element(k_prof.begin(), k_prof.end()) - k_prof.begin());
        bool k_peak_interior = (k_peak_idx > 0) && (k_peak_idx < Ny - 1);
        record("Profile: k peaks in interior (not at wall)", k_peak_interior);
        std::cerr << "  k peak at y-index " << k_peak_idx << "/" << Ny << "\n";
    } catch (const std::exception& e) {
        std::cerr << "  Transport profile exception: " << e.what() << "\n";
        record("Profile: k > 0 throughout", false);
        record("Profile: omega_wall > omega_center", false);
        record("Profile: nu_t_wall < nu_t_max", false);
        record("Profile: nu_t_wall < 50% of nu_t_max", false);
        record("Profile: u_center > u_wall", false);
        record("Profile: k peaks in interior (not at wall)", false);
    }
}

// ============================================================================
// Section 10: TGV Energy Decay Ordering — more viscosity = faster decay
// ============================================================================

static void test_tgv_energy_ordering() {
    try {
        double L = 2.0 * M_PI;
        Mesh mesh;
        mesh.init_uniform(16, 16, 16, 0.0, L, 0.0, L, 0.0, L);

        auto run_tgv = [&](TurbulenceModelType type) -> double {
            Config config;
            config.nu = 0.01;
            config.dt = 0.005;
            config.turb_model = type;
            config.verbose = false;

            RANSSolver solver(mesh, config);
            solver.set_velocity_bc(create_velocity_bc(BCPattern::FullyPeriodic));
            if (type != TurbulenceModelType::None)
                solver.set_turbulence_model(create_turbulence_model(type));
            solver.initialize_uniform(0.0, 0.0);

            // TGV IC (overwrites velocity from initialize_uniform)
            for (int k = mesh.k_begin(); k < mesh.k_end(); ++k)
                for (int j = mesh.j_begin(); j < mesh.j_end(); ++j)
                    for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i)
                        solver.velocity().u(i, j, k) =
                            std::sin(mesh.xf[i]) * std::cos(mesh.y(j)) * std::cos(mesh.z(k));
            for (int k = mesh.k_begin(); k < mesh.k_end(); ++k)
                for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j)
                    for (int i = mesh.i_begin(); i < mesh.i_end(); ++i)
                        solver.velocity().v(i, j, k) =
                            -std::cos(mesh.x(i)) * std::sin(mesh.yf[j]) * std::cos(mesh.z(k));
            solver.sync_to_gpu();

            for (int step = 0; step < 100; ++step) solver.step();
            solver.sync_from_gpu();
            return solver.compute_kinetic_energy();
        };

        double E_laminar  = run_tgv(TurbulenceModelType::None);
        double E_baseline = run_tgv(TurbulenceModelType::Baseline);
        double E_sst      = run_tgv(TurbulenceModelType::SSTKOmega);

        std::cerr << "  TGV E_final: Laminar=" << E_laminar
                  << ", Baseline=" << E_baseline
                  << ", SST=" << E_sst << "\n";

        // More effective viscosity → faster energy decay → lower final energy
        // Baseline adds mixing-length nu_t → should decay faster than laminar
        record("TGV ordering: E_baseline <= E_laminar",
               E_baseline <= E_laminar * 1.01);  // 1% tolerance

        // All energies should be positive and finite
        record("TGV ordering: all energies valid",
               std::isfinite(E_laminar) && std::isfinite(E_baseline) &&
               std::isfinite(E_sst) && E_laminar > 0 && E_baseline > 0 && E_sst > 0);
    } catch (const std::exception& e) {
        std::cerr << "  TGV energy ordering exception: " << e.what() << "\n";
        record("TGV ordering: E_baseline <= E_laminar", false);
        record("TGV ordering: all energies valid", false);
    }
}

// ============================================================================
// Section 11: EARSM produces anisotropic stresses
// ============================================================================

static void test_earsm_anisotropy() {
    // EARSM should produce nonzero normal stress differences (tau_xx != tau_yy)
    // in shear flow, which Boussinesq-based models cannot.
    try {
        Mesh mesh;
        mesh.init_uniform(16, 32, 0.0, 2.0, -1.0, 1.0);

        VectorField vel(mesh);
        for (int j = 0; j < mesh.total_Ny(); ++j)
            for (int i = 0; i < mesh.total_Nx(); ++i) {
                vel.u(i, j) = 1.0 - mesh.y(j) * mesh.y(j);  // Strong Poiseuille
                vel.v(i, j) = 0.0;
            }

        ScalarField k_field(mesh, 0.1), omega_field(mesh, 10.0), nu_t_field(mesh);

        TensorField tau_earsm(mesh);
        SSTWithEARSM earsm(EARSMType::WallinJohansson2000);
        earsm.set_nu(0.001);
        earsm.set_delta(1.0);
        earsm.initialize(mesh, vel);
        earsm.update(mesh, vel, k_field, omega_field, nu_t_field, &tau_earsm);

        // In Boussinesq: tau_xx - tau_yy = -2*nu_t*(S_11 - S_22) = 0 for channel
        // since S_11 = du/dx = 0 and S_22 = dv/dy = 0. But EARSM gives nonzero.
        double max_anisotropy = 0.0;
        double max_tau_xy = 0.0;
        FOR_INTERIOR_2D(mesh, i, j) {
            double aniso = std::abs(tau_earsm.xx(i, j) - tau_earsm.yy(i, j));
            max_anisotropy = std::max(max_anisotropy, aniso);
            max_tau_xy = std::max(max_tau_xy, std::abs(tau_earsm.xy(i, j)));
        }

        record("EARSM: nonzero tau_xy in shear flow", max_tau_xy > 1e-10);
        record("EARSM: normal stress anisotropy (non-Boussinesq)",
               max_anisotropy > 1e-10);
        std::cerr << "  max|tau_xx - tau_yy|=" << max_anisotropy
                  << ", max|tau_xy|=" << max_tau_xy << "\n";

        // The anisotropy should be physically meaningful relative to 2k
        // |b_xx - b_yy| = |tau_xx - tau_yy| / (2k) should be O(0.01-1)
        double k_val = k_field(mesh.i_begin() + mesh.Nx / 2, mesh.j_begin() + mesh.Ny / 2);
        double b_aniso = max_anisotropy / (2.0 * k_val + 1e-15);
        record("EARSM: anisotropy magnitude plausible (b_aniso > 0.01)",
               b_aniso > 0.01);
        std::cerr << "  b_aniso (normalized)=" << b_aniso << "\n";
    } catch (const std::exception& e) {
        std::cerr << "  EARSM anisotropy exception: " << e.what() << "\n";
        record("EARSM: nonzero tau_xy in shear flow", false);
        record("EARSM: normal stress anisotropy (non-Boussinesq)", false);
        record("EARSM: anisotropy magnitude plausible (b_aniso > 0.01)", false);
    }
}

// ============================================================================
// Main
// ============================================================================

int main() {
    using namespace nncfd::test::harness;
    return run_sections("Cross-Geometry Turbulence Model Tests", {
        {"3D Channel Smoke (all models, 50 steps)",        test_3d_channel_smoke},
        {"3D Duct Smoke (all models, 50 steps)",           test_3d_duct_smoke},
        {"3D TGV Smoke (8 models, 50 steps)",              test_3d_tgv_smoke},
        {"DNS Combo (trip + filter, 150 steps)",           test_dns_combo},
        {"Transport Realizability 3D (200 steps)",         test_transport_realizability_3d},
        {"EARSM Trace-Free",                               test_earsm_trace_free},
        {"Cross-Geometry Consistency (2 models x 3 geom)", test_cross_geometry_consistency},
        {"Model Ordering (turbulent > laminar mixing)",    test_model_ordering},
        {"Transport Profile Shape (k/omega/nu_t)",         test_transport_profile_shape},
        {"TGV Energy Decay Ordering",                      test_tgv_energy_ordering},
        {"EARSM Anisotropy (non-Boussinesq)",              test_earsm_anisotropy}
    });
}
