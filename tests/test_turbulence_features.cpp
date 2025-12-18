/// Turbulence model feature tests
/// 
/// Tests that exercise turbulence model computation paths:
/// - EARSM Re_t-based blending (nonlinear terms engage)
/// - Model response to nontrivial velocity gradients
/// - Feature computation consistency
/// - Backend verification (CPU in CPU builds, GPU in GPU builds)

#include "mesh.hpp"
#include "fields.hpp"
#include "features.hpp"
#include "turbulence_model.hpp"
#include "turbulence_baseline.hpp"
#include "turbulence_gep.hpp"
#include "turbulence_earsm.hpp"
#include "solver.hpp"
#include "config.hpp"
#include <iostream>
#include <cmath>
#include <cassert>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

using namespace nncfd;

void test_earsm_ret_blending() {
    std::cout << "Testing EARSM Re_t-based blending... ";
    
    // Setup: shear flow with fixed omega, varying k to sweep Re_t
    Mesh mesh;
    mesh.init_uniform(16, 16, 0.0, 2.0, -1.0, 1.0);
    
    const double gamma = 2.0;  // Shear rate
    const double nu = 0.01;
    const double omega_fixed = 10.0;
    
    // Create shear flow: u = γy, v = 0
    VectorField vel(mesh);
    for (int j = 0; j < mesh.total_Ny(); ++j) {
        for (int i = 0; i < mesh.total_Nx(); ++i) {
            vel.u(i, j) = gamma * mesh.y(j);
            vel.v(i, j) = 0.0;
        }
    }
    
    // Test Pope EARSM (simplest model with guaranteed nonlinear terms)
    auto pope_model = std::make_unique<PopeQuadraticEARSM>();
    pope_model->set_nu(nu);
    pope_model->set_delta(1.0);
    
    int i = mesh.Nx/2;
    int j = mesh.Ny/2;
    
    [[maybe_unused]] double b_xy_low = 0.0;
    [[maybe_unused]] double b_xy_high = 0.0;
    
    // Case 1: Low k → Low Re_t → nonlinear terms should fade
    {
        ScalarField k_low(mesh, 0.0001);  // Very low TKE
        ScalarField omega_low(mesh, omega_fixed);
        ScalarField nu_t_low(mesh);
        TensorField tau_low(mesh);
        
        pope_model->compute_nu_t(mesh, vel, k_low, omega_low, nu_t_low, &tau_low);
        
        // Compute anisotropy: b_xy = tau_xy / (2*k)
        const double tau_xy_low_val = tau_low.xy(i, j);
        const double k_val_low = k_low(i, j);
        b_xy_low = tau_xy_low_val / (2.0 * k_val_low);
        
        // At low Re_t, anisotropy should be small (approaching linear Boussinesq)
        assert(std::isfinite(b_xy_low));
        assert(std::abs(b_xy_low) < 10.0);  // Reasonable bound
    }
    
    // Case 2: High k → High Re_t → nonlinear terms should engage
    {
        ScalarField k_high(mesh, 1.0);  // High TKE
        ScalarField omega_high(mesh, omega_fixed);
        ScalarField nu_t_high(mesh);
        TensorField tau_high(mesh);
        
        pope_model->compute_nu_t(mesh, vel, k_high, omega_high, nu_t_high, &tau_high);
        
        const double tau_xy_high_val = tau_high.xy(i, j);
        const double k_val_high = k_high(i, j);
        b_xy_high = tau_xy_high_val / (2.0 * k_val_high);
        
        assert(std::isfinite(b_xy_high));
        assert(std::abs(b_xy_high) < 10.0);
    }
    
    // The key test: anisotropy should DIFFER between low/high Re_t
    // (If it doesn't, nonlinear terms aren't engaging)
    // This test verifies the blending mechanism is active
    assert(std::abs(b_xy_low - b_xy_high) > 1e-6);
    
    std::cout << "PASSED (Re_t blending active)\n";
}

void test_baseline_responds_to_shear() {
    std::cout << "Testing Baseline model responds to shear... ";
    
    Mesh mesh;
    mesh.init_uniform(32, 32, 0.0, 2.0, -1.0, 1.0);
    
    const double gamma = 3.0;
    
    // Shear flow
    VectorField vel(mesh);
    for (int j = 0; j < mesh.total_Ny(); ++j) {
        for (int i = 0; i < mesh.total_Nx(); ++i) {
            vel.u(i, j) = gamma * mesh.y(j);
            vel.v(i, j) = 0.0;
        }
    }
    
    auto baseline = std::make_unique<MixingLengthModel>();
    baseline->set_nu(0.01);
    baseline->set_delta(1.0);
    
    ScalarField k(mesh, 0.1);
    ScalarField omega(mesh, 1.0);
    ScalarField nu_t(mesh);
    
    baseline->update(mesh, vel, k, omega, nu_t);
    
    // Check nu_t in the interior (away from walls)
    int i_mid = mesh.Nx/2;
    int j_mid = mesh.Ny/2;
    
    double nu_t_val = nu_t(i_mid, j_mid);
    
    // Should be finite, non-negative, and nonzero for shear flow away from walls
    assert(std::isfinite(nu_t_val));
    assert(nu_t_val >= 0.0);
    
    // Near the center of the channel, with shear, nu_t should be positive
    // (not testing exact value, just that it responds)
    double wall_dist = mesh.wall_distance(i_mid, j_mid);
    if (wall_dist > 0.2) {  // Sufficiently far from wall
        assert(nu_t_val > 0.0);
    }
    
    std::cout << "PASSED (nu_t=" << nu_t_val << " at y=" << mesh.y(j_mid) << ")\n";
}

void test_gep_responds_to_shear() {
    std::cout << "Testing GEP model responds to shear... ";
    
    Mesh mesh;
    mesh.init_uniform(32, 32, 0.0, 2.0, -1.0, 1.0);
    
    const double gamma = 3.0;
    
    VectorField vel(mesh);
    for (int j = 0; j < mesh.total_Ny(); ++j) {
        for (int i = 0; i < mesh.total_Nx(); ++i) {
            vel.u(i, j) = gamma * mesh.y(j);
            vel.v(i, j) = 0.0;
        }
    }
    
    auto gep = std::make_unique<TurbulenceGEP>();
    gep->set_nu(0.01);
    gep->set_u_ref(1.0);
    gep->set_delta(1.0);
    gep->initialize(mesh, vel);
    
    ScalarField k(mesh, 0.1);
    ScalarField omega(mesh, 1.0);
    ScalarField nu_t(mesh);
    
    gep->update(mesh, vel, k, omega, nu_t);
    
    int i_mid = mesh.Nx/2;
    int j_mid = mesh.Ny/2;
    double nu_t_val = nu_t(i_mid, j_mid);
    
    assert(std::isfinite(nu_t_val));
    assert(nu_t_val >= 0.0);
    
    double wall_dist = mesh.wall_distance(i_mid, j_mid);
    if (wall_dist > 0.2) {
        assert(nu_t_val > 0.0);
    }
    
    std::cout << "PASSED (nu_t=" << nu_t_val << ")\n";
}

void test_earsm_wallin_johansson_shear() {
    std::cout << "Testing Wallin-Johansson EARSM with shear... ";
    
    Mesh mesh;
    mesh.init_uniform(32, 32, 0.0, 2.0, -1.0, 1.0);
    
    const double gamma = 2.0;
    
    VectorField vel(mesh);
    for (int j = 0; j < mesh.total_Ny(); ++j) {
        for (int i = 0; i < mesh.total_Nx(); ++i) {
            vel.u(i, j) = gamma * mesh.y(j);
            vel.v(i, j) = 0.0;
        }
    }
    
    auto wj = std::make_unique<WallinJohanssonEARSM>();
    wj->set_nu(0.01);
    wj->set_delta(1.0);
    
    ScalarField k(mesh, 0.1);
    ScalarField omega(mesh, 10.0);
    ScalarField nu_t(mesh);
    TensorField tau(mesh);
    
    wj->compute_nu_t(mesh, vel, k, omega, nu_t, &tau);
    
    int i_mid = mesh.Nx/2;
    int j_mid = mesh.Ny/2;
    
    double nu_t_val = nu_t(i_mid, j_mid);
    double tau_xy_val = tau.xy(i_mid, j_mid);
    
    // Basic sanity checks
    assert(std::isfinite(nu_t_val));
    assert(std::isfinite(tau_xy_val));
    assert(nu_t_val >= 0.0);
    
    // For shear flow with positive strain, tau_xy should be nonzero
    assert(std::abs(tau_xy_val) > 1e-10);
    
    std::cout << "PASSED (nu_t=" << nu_t_val << ", tau_xy=" << tau_xy_val << ")\n";
}

void test_earsm_gatski_speziale_shear() {
    std::cout << "Testing Gatski-Speziale EARSM with shear... ";
    
    Mesh mesh;
    mesh.init_uniform(32, 32, 0.0, 2.0, -1.0, 1.0);
    
    const double gamma = 2.0;
    
    VectorField vel(mesh);
    for (int j = 0; j < mesh.total_Ny(); ++j) {
        for (int i = 0; i < mesh.total_Nx(); ++i) {
            vel.u(i, j) = gamma * mesh.y(j);
            vel.v(i, j) = 0.0;
        }
    }
    
    auto gs = std::make_unique<GatskiSpezialeEARSM>();
    gs->set_nu(0.01);
    gs->set_delta(1.0);
    
    ScalarField k(mesh, 0.1);
    ScalarField omega(mesh, 10.0);
    ScalarField nu_t(mesh);
    TensorField tau(mesh);
    
    gs->compute_nu_t(mesh, vel, k, omega, nu_t, &tau);
    
    int i_mid = mesh.Nx/2;
    int j_mid = mesh.Ny/2;
    
    double nu_t_val = nu_t(i_mid, j_mid);
    double tau_xy_val = tau.xy(i_mid, j_mid);
    
    assert(std::isfinite(nu_t_val));
    assert(std::isfinite(tau_xy_val));
    assert(nu_t_val >= 0.0);
    assert(std::abs(tau_xy_val) > 1e-10);
    
    std::cout << "PASSED (nu_t=" << nu_t_val << ", tau_xy=" << tau_xy_val << ")\n";
}

void test_earsm_pope_quadratic_shear() {
    std::cout << "Testing Pope quadratic model with shear... ";
    
    Mesh mesh;
    mesh.init_uniform(32, 32, 0.0, 2.0, -1.0, 1.0);
    
    const double gamma = 2.0;
    
    VectorField vel(mesh);
    for (int j = 0; j < mesh.total_Ny(); ++j) {
        for (int i = 0; i < mesh.total_Nx(); ++i) {
            vel.u(i, j) = gamma * mesh.y(j);
            vel.v(i, j) = 0.0;
        }
    }
    
    auto pope = std::make_unique<PopeQuadraticEARSM>();
    pope->set_nu(0.01);
    pope->set_delta(1.0);
    
    ScalarField k(mesh, 0.1);
    ScalarField omega(mesh, 10.0);
    ScalarField nu_t(mesh);
    TensorField tau(mesh);
    
    pope->compute_nu_t(mesh, vel, k, omega, nu_t, &tau);
    
    int i_mid = mesh.Nx/2;
    int j_mid = mesh.Ny/2;
    
    double nu_t_val = nu_t(i_mid, j_mid);
    [[maybe_unused]] double tau_xy_val = tau.xy(i_mid, j_mid);
    double tau_xx_val = tau.xx(i_mid, j_mid);
    double tau_yy_val = tau.yy(i_mid, j_mid);
    
    assert(std::isfinite(nu_t_val));
    assert(std::isfinite(tau_xy_val));
    assert(std::isfinite(tau_xx_val));
    assert(std::isfinite(tau_yy_val));
    assert(nu_t_val >= 0.0);
    
    // Anisotropy check: for shear, tau_xx != tau_yy (anisotropic)
    double anisotropy = std::abs(tau_xx_val - tau_yy_val);
    assert(anisotropy > 1e-12);  // Should have some anisotropy
    
    std::cout << "PASSED (nu_t=" << nu_t_val << ", anisotropy=" << anisotropy << ")\n";
}

void test_feature_computer_batch() {
    std::cout << "Testing FeatureComputer batch computation... ";
    
    Mesh mesh;
    mesh.init_uniform(16, 16, 0.0, 2.0, -1.0, 1.0);
    
    const double gamma = 2.0;
    
    VectorField vel(mesh);
    for (int j = 0; j < mesh.total_Ny(); ++j) {
        for (int i = 0; i < mesh.total_Nx(); ++i) {
            vel.u(i, j) = gamma * mesh.y(j);
            vel.v(i, j) = 0.0;
        }
    }
    
    ScalarField k(mesh, 0.1);
    ScalarField omega(mesh, 1.0);
    
    FeatureComputer fc(mesh);
    fc.set_reference(0.001, 1.0, 1.0);
    
    // Test scalar features
    std::vector<Features> scalar_features;
    fc.compute_scalar_features(vel, k, omega, scalar_features);
    
    int n_interior = mesh.Nx * mesh.Ny;
    assert(static_cast<int>(scalar_features.size()) == n_interior);
    
    // All features should be finite
    for (const auto& feat : scalar_features) {
        for (int n = 0; n < feat.size(); ++n) {
            assert(std::isfinite(feat[n]));
        }
    }
    
    // Test TBNN features
    std::vector<Features> tbnn_features;
    std::vector<std::array<std::array<double, 3>, TensorBasis::NUM_BASIS>> basis;
    fc.compute_tbnn_features(vel, k, omega, tbnn_features, basis);
    
    assert(static_cast<int>(tbnn_features.size()) == n_interior);
    assert(static_cast<int>(basis.size()) == n_interior);
    
    // All features and basis tensors should be finite
    for (int idx = 0; idx < n_interior; ++idx) {
        for (int n = 0; n < tbnn_features[idx].size(); ++n) {
            assert(std::isfinite(tbnn_features[idx][n]));
        }
        for (int b = 0; b < TensorBasis::NUM_BASIS; ++b) {
            for (int c = 0; c < 3; ++c) {
                assert(std::isfinite(basis[idx][b][c]));
            }
        }
    }
    
    std::cout << "PASSED (" << n_interior << " cells processed)\n";
}

void test_realizability_constraints() {
    std::cout << "Testing realizability constraints (nu_t >= 0)... ";
    
    Mesh mesh;
    mesh.init_uniform(32, 32, 0.0, 2.0, -1.0, 1.0);
    
    // Create various velocity fields
    const double gamma = 2.0;
    
    VectorField vel(mesh);
    for (int j = 0; j < mesh.total_Ny(); ++j) {
        for (int i = 0; i < mesh.total_Nx(); ++i) {
            vel.u(i, j) = gamma * mesh.y(j);
            vel.v(i, j) = 0.0;
        }
    }
    
    ScalarField k(mesh, 0.1);
    ScalarField omega(mesh, 10.0);
    ScalarField nu_t(mesh);
    
    // Test all EARSM models for realizability
    std::vector<std::unique_ptr<EARSMClosure>> models;
    models.push_back(std::make_unique<WallinJohanssonEARSM>());
    models.push_back(std::make_unique<GatskiSpezialeEARSM>());
    models.push_back(std::make_unique<PopeQuadraticEARSM>());
    
    for (auto& model : models) {
        model->set_nu(0.01);
        model->set_delta(1.0);
        
        model->compute_nu_t(mesh, vel, k, omega, nu_t);
        
        // Check all cells
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                [[maybe_unused]] double nu_t_val = nu_t(i, j);
                
                // Realizability: nu_t >= 0, finite
                assert(std::isfinite(nu_t_val));
                assert(nu_t_val >= 0.0);
            }
        }
    }
    
    std::cout << "PASSED (all models satisfy nu_t >= 0)\n";
}

void test_solver_backend_execution() {
#ifdef USE_GPU_OFFLOAD
    std::cout << "Testing solver backend execution (GPU)... ";
    
    int num_devices = omp_get_num_devices();
    if (num_devices == 0) {
        std::cout << "SKIPPED (no GPU devices)\n";
        return;
    }
#else
    std::cout << "Testing solver backend execution (CPU)... ";
#endif
    
    // Run a short simulation with Baseline turbulence model
    Mesh mesh;
    mesh.init_uniform(16, 32, 0.0, 2.0, -1.0, 1.0);
    
    Config config;
    config.nu = 0.01;
    config.dt = 1e-3;
    config.turb_model = TurbulenceModelType::Baseline;
    config.verbose = false;
    
    RANSSolver solver(mesh, config);
    
    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);
    
    auto turb_model = create_turbulence_model(TurbulenceModelType::Baseline, "", "");
    solver.set_turbulence_model(std::move(turb_model));
    
    solver.set_body_force(-0.001, 0.0);
    solver.initialize_uniform(0.5, 0.0);
    
    // Run 20 steps
    for (int i = 0; i < 20; ++i) {
        solver.step();
    }
    
    // Verify results are finite and reasonable
    const auto& nu_t = solver.nu_t();
    const auto& vel = solver.velocity();
    
    double max_nu_t = 0.0;
    double max_u = 0.0;
    
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            assert(std::isfinite(nu_t(i, j)));
            assert(std::isfinite(vel.u(i, j)));
            assert(std::isfinite(vel.v(i, j)));
            max_nu_t = std::max(max_nu_t, nu_t(i, j));
            max_u = std::max(max_u, std::abs(vel.u(i, j)));
        }
    }
    
    assert(max_nu_t >= 0.0);  // Realizability
    assert(max_u > 0.0);      // Flow is actually moving
    
#ifdef USE_GPU_OFFLOAD
    std::cout << "PASSED (GPU backend verified)\n";
#else
    std::cout << "PASSED (CPU backend verified)\n";
#endif
}

int main() {
    std::cout << "\n========================================\n";
    std::cout << "  TURBULENCE MODEL FEATURE TESTS\n";
    std::cout << "========================================\n";
    std::cout << "Purpose: Verify turbulence models\n";
    std::cout << "         respond correctly to nontrivial\n";
    std::cout << "         velocity gradients and exercise\n";
    std::cout << "         nonlinear feature paths\n";
#ifdef USE_GPU_OFFLOAD
    std::cout << "Backend: GPU\n";
#else
    std::cout << "Backend: CPU\n";
#endif
    std::cout << "========================================\n\n";
    
    // EARSM-specific tests
    test_earsm_ret_blending();
    test_earsm_wallin_johansson_shear();
    test_earsm_gatski_speziale_shear();
    test_earsm_pope_quadratic_shear();
    
    // Algebraic model tests
    test_baseline_responds_to_shear();
    test_gep_responds_to_shear();
    
    // Batch computation tests
    test_feature_computer_batch();
    
    // Realizability tests
    test_realizability_constraints();
    
    // Backend execution test (solver-driven)
    test_solver_backend_execution();
    
    std::cout << "\n========================================\n";
    std::cout << "[SUCCESS] All turbulence feature tests passed!\n";
    std::cout << "========================================\n";
    return 0;
}

