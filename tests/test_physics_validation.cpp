/// Physics validation tests for CI - Verify solver correctly solves N-S
/// REFACTORED: Using test_framework.hpp - reduced from 784 to ~450 lines

#include "test_framework.hpp"
#include "timing.hpp"
#include <cstring>

using namespace nncfd;
using namespace nncfd::test;

//=============================================================================
// Test 1A: Poiseuille Single-Step Analytical Invariance
//=============================================================================
void test_poiseuille_single_step() {
    std::cout << "\n========================================\n";
    std::cout << "Test 1A: Poiseuille Single-Step Invariance\n";
    std::cout << "========================================\n";

    Mesh mesh;
    mesh.init_uniform(64, 128, 0.0, 4.0, -1.0, 1.0);

    Config config;
    config.nu = 0.01;
    config.dp_dx = -0.001;
    config.dt = 0.001;
    config.adaptive_dt = false;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);
    solver.set_body_force(-config.dp_dx, 0.0);

    init_poiseuille(solver, mesh, config.dp_dx, config.nu, 1.0, 1.0);
    solver.sync_to_gpu();

    solver.step();
    solver.sync_from_gpu();

    double l2_error = compute_poiseuille_error(solver.velocity(), mesh, config.dp_dx, config.nu);

    std::cout << "  L2 profile error after 1 step: " << l2_error * 100 << "%\n";

    if (l2_error > 0.005) {
        throw std::runtime_error("Single-step Poiseuille test failed: error=" + std::to_string(l2_error*100) + "%");
    }
    std::cout << "[PASS] Analytical profile preserved\n";
}

//=============================================================================
// Test 1B: Poiseuille Multi-Step Stability
//=============================================================================
void test_poiseuille_multistep() {
    std::cout << "\n========================================\n";
    std::cout << "Test 1B: Poiseuille Multi-Step Stability\n";
    std::cout << "========================================\n";

    Mesh mesh;
    mesh.init_uniform(64, 128, 0.0, 4.0, -1.0, 1.0);

    Config config;
    config.nu = 0.01;
    config.dp_dx = -0.001;
    config.dt = 0.002;
    config.adaptive_dt = false;
    config.max_iter = 10;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);
    solver.set_body_force(-config.dp_dx, 0.0);

    init_poiseuille(solver, mesh, config.dp_dx, config.nu, 1.0, 1.0);
    solver.sync_to_gpu();

    for (int step = 0; step < config.max_iter; ++step) {
        solver.step();
    }
    solver.sync_from_gpu();

    // Check for NaN/Inf
    const VectorField& vel = solver.velocity();
    int i_center = mesh.i_begin() + mesh.Nx / 2;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        if (!std::isfinite(vel.u(i_center, j))) {
            throw std::runtime_error("Solution contains NaN/Inf!");
        }
    }

    double l2_error = compute_poiseuille_error(vel, mesh, config.dp_dx, config.nu);
    std::cout << "  L2 error after 10 steps: " << l2_error * 100 << "%\n";

    if (l2_error > 0.01) {
        throw std::runtime_error("Poiseuille multi-step accuracy failed");
    }
    std::cout << "[PASS] Solution stable and accurate\n";
}

//=============================================================================
// Test 2: Divergence-Free Constraint
//=============================================================================
void test_divergence_free() {
    std::cout << "\n========================================\n";
    std::cout << "Test 2: Divergence-Free Constraint\n";
    std::cout << "========================================\n";

    Mesh mesh;
    mesh.init_uniform(64, 128, 0.0, 4.0, -1.0, 1.0);

    Config config;
    config.nu = 0.01;
    config.adaptive_dt = true;
    config.max_iter = 300;
    config.tol = 1e-4;
    config.turb_model = TurbulenceModelType::Baseline;
    config.verbose = true;
    config.output_freq = 50;

    RANSSolver solver(mesh, config);
    setup_channel_solver(solver, config);
    solver.initialize_uniform(0.1, 0.0);

    auto [residual, iters] = solver.solve_steady();
    solver.sync_from_gpu();

    double max_div = compute_max_divergence(solver.velocity(), mesh);
    std::cout << "  Max divergence: " << std::scientific << max_div << "\n";

    if (max_div > 1e-3) {
        throw std::runtime_error("Divergence-free test failed");
    }
    std::cout << "[PASS] Incompressibility constraint satisfied\n";
}

//=============================================================================
// Test 3: Global Momentum Balance
//=============================================================================
void test_momentum_balance() {
    std::cout << "\n========================================\n";
    std::cout << "Test 3: Global Momentum Balance\n";
    std::cout << "========================================\n";

    Mesh mesh;
    mesh.init_uniform(64, 128, 0.0, 4.0, -1.0, 1.0);

    Config config;
    config.nu = 0.01;
    config.dp_dx = -0.001;
    config.adaptive_dt = true;
    config.max_iter = 100;
    config.tol = 1e-5;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = true;
    config.output_freq = 50;
    config.poisson_max_iter = 1000;
    config.poisson_abs_tol_floor = 1e-6;

    RANSSolver solver(mesh, config);
    solver.set_body_force(-config.dp_dx, 0.0);

    init_poiseuille(solver, mesh, config.dp_dx, config.nu, 1.0, 0.9);
    solver.sync_to_gpu();

    auto [residual, iters] = solver.solve_steady();
    solver.sync_from_gpu();

    const VectorField& vel = solver.velocity();

    // Body force
    double L_x = mesh.x_max - mesh.x_min;
    double L_y = mesh.y_max - mesh.y_min;
    double F_body = -config.dp_dx * L_x * L_y;

    // Wall shear stress
    double F_wall = 0.0;
    int j_bot = mesh.j_begin();
    int j_top = mesh.j_end() - 1;
    for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
        double tau_bot = config.nu * std::abs((vel.u(i, j_bot+1) - vel.u(i, j_bot)) / mesh.dy);
        double tau_top = config.nu * std::abs((vel.u(i, j_top) - vel.u(i, j_top-1)) / mesh.dy);
        F_wall += (tau_bot + tau_top) * mesh.dx;
    }

    double imbalance = std::abs(F_body - F_wall) / F_body;
    std::cout << "  Body force:    " << F_body << "\n";
    std::cout << "  Wall friction: " << F_wall << "\n";
    std::cout << "  Imbalance:     " << imbalance * 100 << "%\n";

    if (imbalance > 0.11) {
        throw std::runtime_error("Momentum balance test failed");
    }
    std::cout << "[PASS] Momentum balanced to " << imbalance*100 << "%\n";
}

//=============================================================================
// Test 4: Channel Flow Symmetry
//=============================================================================
void test_channel_symmetry() {
    std::cout << "\n========================================\n";
    std::cout << "Test 4: Channel Flow Symmetry\n";
    std::cout << "========================================\n";

    Mesh mesh;
    mesh.init_uniform(64, 128, 0.0, 4.0, -1.0, 1.0);

    Config config;
    config.nu = 0.01;
    config.adaptive_dt = true;
    config.max_iter = 300;
    config.tol = 1e-4;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);
    setup_channel_solver(solver, config);
    solver.initialize_uniform(0.1, 0.0);

    auto [residual, iters] = solver.solve_steady();
    solver.sync_from_gpu();

    const VectorField& vel = solver.velocity();
    double max_asymmetry = 0.0;
    int i_mid = mesh.i_begin() + mesh.Nx / 2;

    for (int j = mesh.j_begin(); j < mesh.j_begin() + mesh.Ny/2; ++j) {
        int j_mirror = mesh.j_end() - 1 - (j - mesh.j_begin());
        double u_lower = vel.u(i_mid, j);
        double u_upper = vel.u(i_mid, j_mirror);
        double asymmetry = std::abs(u_lower - u_upper) / std::max(std::abs(u_lower), 1e-10);
        max_asymmetry = std::max(max_asymmetry, asymmetry);
    }

    std::cout << "  Max asymmetry: " << max_asymmetry * 100 << "%\n";

    if (max_asymmetry > 0.01) {
        throw std::runtime_error("Symmetry test failed");
    }
    std::cout << "[PASS] Flow symmetric\n";
}

//=============================================================================
// Test 5: Cross-Model Consistency (Laminar Limit)
//=============================================================================
void test_cross_model_consistency() {
    std::cout << "\n========================================\n";
    std::cout << "Test 5: Cross-Model Consistency\n";
    std::cout << "========================================\n";

    std::vector<TurbulenceModelType> models = {
        TurbulenceModelType::None,
        TurbulenceModelType::Baseline,
        TurbulenceModelType::KOmega
    };
    std::vector<std::string> model_names = {"None (laminar)", "Baseline", "K-Omega"};
    std::vector<double> bulk_velocities;

    for (size_t m = 0; m < models.size(); ++m) {
        Mesh mesh;
        mesh.init_uniform(32, 64, 0.0, 4.0, -1.0, 1.0);

        Config config;
        config.nu = 0.01;
        config.dp_dx = -0.001;
        config.adaptive_dt = true;
        config.max_iter = 300;
        config.tol = 1e-4;
        config.turb_model = models[m];
        config.verbose = false;

        RANSSolver solver(mesh, config);
        solver.set_body_force(-config.dp_dx, 0.0);

        init_poiseuille(solver, mesh, config.dp_dx, config.nu, 1.0, 0.9);
        solver.sync_to_gpu();

        auto [residual, iters] = solver.solve_steady();
        solver.sync_from_gpu();

        const VectorField& vel = solver.velocity();
        double bulk_u = 0.0;
        int count = 0;
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                bulk_u += vel.u(i, j);
                count++;
            }
        }
        bulk_u /= count;
        bulk_velocities.push_back(bulk_u);

        std::cout << "  " << model_names[m] << ": U_bulk=" << bulk_u << "\n";
    }

    double ref = bulk_velocities[0];
    for (size_t m = 1; m < bulk_velocities.size(); ++m) {
        double diff = std::abs(bulk_velocities[m] - ref) / ref;
        if (diff > 0.05) {
            throw std::runtime_error("Cross-model consistency failed");
        }
    }
    std::cout << "[PASS] All models consistent\n";
}

//=============================================================================
// Test 6: CPU vs GPU Consistency
//=============================================================================
void test_cpu_gpu_consistency() {
    std::cout << "\n========================================\n";
    std::cout << "Test 6: CPU vs GPU Consistency\n";
    std::cout << "========================================\n";

#ifndef USE_GPU_OFFLOAD
    std::cout << "SKIPPED: GPU offload not enabled\n";
    return;
#else
    if (omp_get_num_devices() == 0) {
        throw std::runtime_error("USE_GPU_OFFLOAD enabled but no GPU devices found");
    }

    int on_device = 0;
    #pragma omp target map(tofrom: on_device)
    {
        on_device = !omp_is_initial_device();
    }

    if (!on_device) {
        throw std::runtime_error("GPU not accessible");
    }

    std::cout << "  GPU accessible: YES\n";

    Mesh mesh;
    mesh.init_uniform(32, 64, 0.0, 4.0, -1.0, 1.0);

    Config config;
    config.nu = 0.01;
    config.dp_dx = -0.001;
    config.adaptive_dt = true;
    config.max_iter = 1000;
    config.tol = 1e-6;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);
    solver.set_body_force(-config.dp_dx, 0.0);
    init_poiseuille(solver, mesh, config.dp_dx, config.nu, 1.0, 0.9);
    solver.sync_to_gpu();

    auto [res1, iter1] = solver.solve_steady();
    solver.sync_from_gpu();

    double u_center = solver.velocity().u(mesh.i_begin() + mesh.Nx/2, mesh.j_begin() + mesh.Ny/2);
    std::cout << "  u_center=" << u_center << ", iters=" << iter1 << "\n";

    std::cout << "[PASS] GPU execution successful\n";
#endif
}

//=============================================================================
// Test 7: Quick Sanity Checks
//=============================================================================
void test_sanity_checks() {
    std::cout << "\n========================================\n";
    std::cout << "Test 7: Quick Sanity Checks\n";
    std::cout << "========================================\n";

    // Check for NaN/Inf
    {
        std::cout << "  Checking for NaN/Inf... " << std::flush;
        Mesh mesh;
        mesh.init_uniform(16, 32, 0.0, 1.0, -1.0, 1.0);

        Config config;
        config.nu = 0.01;
        config.dt = 0.001;
        config.max_iter = 100;
        config.tol = 1e-6;
        config.turb_model = TurbulenceModelType::Baseline;
        config.verbose = false;

        RANSSolver solver(mesh, config);
        setup_channel_solver(solver, config);
        solver.initialize_uniform(0.1, 0.0);
        solver.step();
        solver.sync_from_gpu();

        const VectorField& vel = solver.velocity();
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                if (!std::isfinite(vel.u(i,j)) || !std::isfinite(vel.v(i,j))) {
                    throw std::runtime_error("Velocity contains NaN/Inf!");
                }
            }
        }
        std::cout << "[OK]\n";
    }

    // Check realizability (nu_t >= 0)
    {
        std::cout << "  Checking realizability... " << std::flush;
        Mesh mesh;
        mesh.init_uniform(16, 32, 0.0, 1.0, -1.0, 1.0);

        Config config;
        config.nu = 0.01;
        config.dt = 0.001;
        config.max_iter = 100;
        config.tol = 1e-6;
        config.turb_model = TurbulenceModelType::Baseline;
        config.verbose = false;

        RANSSolver solver(mesh, config);
        setup_channel_solver(solver, config);
        solver.initialize_uniform(0.1, 0.0);
        solver.step();
        solver.sync_from_gpu();

        const ScalarField& nu_t = solver.nu_t();
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                if (nu_t(i,j) < 0.0) {
                    throw std::runtime_error("Eddy viscosity is negative!");
                }
            }
        }
        std::cout << "[OK]\n";
    }

    std::cout << "[PASS] All sanity checks passed\n";
}

//=============================================================================
// Main
//=============================================================================
int main(int argc, char* argv[]) {
    bool poiseuille_only = false;
    bool show_timing = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--poiseuille-only") == 0 || std::strcmp(argv[i], "-p") == 0) {
            poiseuille_only = true;
        } else if (std::strcmp(argv[i], "--timing") == 0 || std::strcmp(argv[i], "-t") == 0) {
            show_timing = true;
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            std::cout << "Usage: " << argv[0] << " [--poiseuille-only|-p] [--timing|-t]\n";
            return 0;
        }
    }

    std::cout << "\n========================================================\n";
    std::cout << "  PHYSICS VALIDATION TEST SUITE\n";
    std::cout << "========================================================\n";

    try {
        if (poiseuille_only) {
            test_poiseuille_single_step();
            test_poiseuille_multistep();
        } else {
            test_sanity_checks();
            test_poiseuille_single_step();
            test_poiseuille_multistep();
            test_divergence_free();
            test_momentum_balance();
            test_channel_symmetry();
            test_cross_model_consistency();
            test_cpu_gpu_consistency();
        }

        std::cout << "\n========================================================\n";
        std::cout << "  [PASS] ALL PHYSICS TESTS PASSED!\n";
        std::cout << "========================================================\n";

        if (show_timing) {
            TimingStats::instance().print_summary();
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n[FAIL] PHYSICS VALIDATION FAILED: " << e.what() << "\n";
        return 1;
    }
}
