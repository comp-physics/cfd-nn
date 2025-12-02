/// Unit tests for turbulence models

#include "mesh.hpp"
#include "fields.hpp"
#include "turbulence_model.hpp"
#include "turbulence_baseline.hpp"
#include "turbulence_gep.hpp"
#include "turbulence_nn_mlp.hpp"
#include "turbulence_nn_tbnn.hpp"
#include "turbulence_transport.hpp"
#include "turbulence_earsm.hpp"
#include <iostream>
#include <cmath>
#include <cassert>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

using namespace nncfd;

void test_baseline_model() {
    std::cout << "Testing baseline mixing length model... ";
    
    Mesh mesh;
    mesh.init_uniform(16, 32, 0.0, 2.0, -1.0, 1.0);
    
    // Simple shear flow
    VectorField vel(mesh);
    for (int j = 0; j < mesh.total_Ny(); ++j) {
        for (int i = 0; i < mesh.total_Nx(); ++i) {
            vel.u(i, j) = mesh.y(j);
            vel.v(i, j) = 0.0;
        }
    }
    
    ScalarField k(mesh, 0.0);
    ScalarField omega(mesh, 0.0);
    ScalarField nu_t(mesh);
    
    MixingLengthModel model;
    model.set_nu(0.001);
    model.set_delta(1.0);
    model.update(mesh, vel, k, omega, nu_t);
    
    // Check nu_t is positive and bounded
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            assert(nu_t(i, j) >= 0.0);
            assert(std::isfinite(nu_t(i, j)));
            assert(nu_t(i, j) < 10.0);  // Reasonable upper bound
        }
    }
    
    std::cout << "PASSED\n";
}

void test_gep_model() {
    std::cout << "Testing GEP model... ";
    
    Mesh mesh;
    mesh.init_uniform(16, 32, 0.0, 2.0, -1.0, 1.0);
    
    VectorField vel(mesh, 1.0, 0.0);
    ScalarField k(mesh, 0.0);
    ScalarField omega(mesh, 0.0);
    ScalarField nu_t(mesh);
    
    TurbulenceGEP model;
    model.set_nu(0.001);
    model.update(mesh, vel, k, omega, nu_t);
    
    // Check validity
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            assert(nu_t(i, j) >= 0.0);
            assert(std::isfinite(nu_t(i, j)));
        }
    }
    
    std::cout << "PASSED\n";
}

void test_nn_mlp_model() {
    std::cout << "Testing NN-MLP model... ";
    
    Mesh mesh;
    mesh.init_uniform(8, 16, 0.0, 1.0, -1.0, 1.0);
    
    VectorField vel(mesh, 1.0, 0.0);
    ScalarField k(mesh, 0.01);
    ScalarField omega(mesh, 1.0);
    ScalarField nu_t(mesh);
    
    TurbulenceNNMLP model;
    model.set_nu(0.001);
    
    try {
        model.load("../data/models/test_mlp", "../data/models/test_mlp");
        
#ifdef USE_GPU_OFFLOAD
        // Upload to GPU if available
        if (omp_get_num_devices() > 0) {
            model.upload_to_gpu();
            std::cout << "[GPU mode] ";
        }
#endif
        
        model.update(mesh, vel, k, omega, nu_t);
        
        // Check all values are finite and positive
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                assert(std::isfinite(nu_t(i, j)));
                assert(nu_t(i, j) >= 0.0);
            }
        }
        
        std::cout << "PASSED\n";
    } catch (const std::exception& e) {
        std::cout << "SKIPPED (model not found)\n";
    }
}

void test_nn_tbnn_model() {
    std::cout << "Testing NN-TBNN model... ";
    
    Mesh mesh;
    mesh.init_uniform(8, 16, 0.0, 1.0, -1.0, 1.0);
    
    VectorField vel(mesh, 1.0, 0.0);
    ScalarField k(mesh, 0.01);
    ScalarField omega(mesh, 1.0);
    ScalarField nu_t(mesh);
    
    TurbulenceNNTBNN model;
    model.set_nu(0.001);
    model.set_delta(1.0);
    model.set_u_ref(1.0);
    
    try {
        model.load("../data/models/test_tbnn", "../data/models/test_tbnn");
        
#ifdef USE_GPU_OFFLOAD
        // Upload to GPU if available
        if (omp_get_num_devices() > 0) {
            model.upload_to_gpu();
            std::cout << "[GPU mode] ";
        }
#endif
        
        model.update(mesh, vel, k, omega, nu_t);
        
        // Check validity
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                assert(std::isfinite(nu_t(i, j)));
                assert(nu_t(i, j) >= 0.0);
            }
        }
        
        std::cout << "PASSED\n";
    } catch (const std::exception& e) {
        std::cout << "SKIPPED (model not found)\n";
    }
}

void test_sst_komega_transport() {
    std::cout << "Testing SST k-omega transport model... ";
    
    Mesh mesh;
    mesh.init_uniform(16, 32, 0.0, 2.0, -1.0, 1.0);
    
    // Simple shear flow (Couette-like)
    VectorField vel(mesh);
    for (int j = 0; j < mesh.total_Ny(); ++j) {
        for (int i = 0; i < mesh.total_Nx(); ++i) {
            double y = mesh.y(j);
            vel.u(i, j) = 0.5 * (y + 1.0);  // Linear profile
            vel.v(i, j) = 0.0;
        }
    }
    
    // Initial turbulence fields
    ScalarField k(mesh, 0.01);
    ScalarField omega(mesh, 100.0);
    ScalarField nu_t(mesh, 0.0);
    
    SSTKOmegaTransport model;
    model.set_nu(0.001);
    model.set_delta(1.0);
    model.initialize(mesh, vel);
    
    // Check that it's a transport model
    assert(model.uses_transport_equations());
    assert(model.name() == "SSTKOmega");
    
    // Take a few transport steps
    double dt = 0.001;
    for (int step = 0; step < 5; ++step) {
        model.advance_turbulence(mesh, vel, dt, k, omega, nu_t);
        model.update(mesh, vel, k, omega, nu_t);
    }
    
    // Check validity of results
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            assert(k(i, j) > 0.0);
            assert(omega(i, j) > 0.0);
            assert(nu_t(i, j) >= 0.0);
            assert(std::isfinite(k(i, j)));
            assert(std::isfinite(omega(i, j)));
            assert(std::isfinite(nu_t(i, j)));
        }
    }
    
    std::cout << "PASSED\n";
}

void test_komega_transport() {
    std::cout << "Testing standard k-omega transport model... ";
    
    Mesh mesh;
    mesh.init_uniform(16, 32, 0.0, 2.0, -1.0, 1.0);
    
    VectorField vel(mesh, 1.0, 0.0);
    ScalarField k(mesh, 0.01);
    ScalarField omega(mesh, 100.0);
    ScalarField nu_t(mesh, 0.0);
    
    KOmegaTransport model;
    model.set_nu(0.001);
    model.initialize(mesh, vel);
    
    assert(model.uses_transport_equations());
    assert(model.name() == "KOmega");
    
    // Take a transport step
    double dt = 0.001;
    model.advance_turbulence(mesh, vel, dt, k, omega, nu_t);
    model.update(mesh, vel, k, omega, nu_t);
    
    // Check validity
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            assert(std::isfinite(k(i, j)));
            assert(std::isfinite(omega(i, j)));
            assert(std::isfinite(nu_t(i, j)));
        }
    }
    
    std::cout << "PASSED\n";
}

void test_wallin_johansson_earsm() {
    std::cout << "Testing Wallin-Johansson EARSM... ";
    
    Mesh mesh;
    mesh.init_uniform(16, 32, 0.0, 2.0, -1.0, 1.0);
    
    // Shear flow
    VectorField vel(mesh);
    for (int j = 0; j < mesh.total_Ny(); ++j) {
        for (int i = 0; i < mesh.total_Nx(); ++i) {
            double y = mesh.y(j);
            vel.u(i, j) = y;
            vel.v(i, j) = 0.0;
        }
    }
    
    ScalarField k(mesh, 0.01);
    ScalarField omega(mesh, 10.0);
    ScalarField nu_t(mesh);
    TensorField tau_ij(mesh);
    
    WallinJohanssonEARSM model;
    model.set_nu(0.001);
    model.set_delta(1.0);
    
    assert(model.provides_reynolds_stresses());
    assert(model.name() == "WJ-EARSM");
    
    model.compute_nu_t(mesh, vel, k, omega, nu_t, &tau_ij);
    
    // Check validity
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            assert(std::isfinite(nu_t(i, j)));
            assert(nu_t(i, j) >= 0.0);
            assert(std::isfinite(tau_ij.xx(i, j)));
            assert(std::isfinite(tau_ij.xy(i, j)));
            assert(std::isfinite(tau_ij.yy(i, j)));
        }
    }
    
    std::cout << "PASSED\n";
}

void test_gatski_speziale_earsm() {
    std::cout << "Testing Gatski-Speziale EARSM... ";
    
    Mesh mesh;
    mesh.init_uniform(16, 32, 0.0, 2.0, -1.0, 1.0);
    
    VectorField vel(mesh, 1.0, 0.0);
    ScalarField k(mesh, 0.01);
    ScalarField omega(mesh, 10.0);
    ScalarField nu_t(mesh);
    
    GatskiSpezialeEARSM model;
    model.set_nu(0.001);
    model.set_delta(1.0);
    
    assert(model.name() == "GS-EARSM");
    
    model.compute_nu_t(mesh, vel, k, omega, nu_t);
    
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            assert(std::isfinite(nu_t(i, j)));
            assert(nu_t(i, j) >= 0.0);
        }
    }
    
    std::cout << "PASSED\n";
}

void test_pope_quadratic_earsm() {
    std::cout << "Testing Pope quadratic EARSM... ";
    
    Mesh mesh;
    mesh.init_uniform(16, 32, 0.0, 2.0, -1.0, 1.0);
    
    VectorField vel(mesh, 1.0, 0.0);
    ScalarField k(mesh, 0.01);
    ScalarField omega(mesh, 10.0);
    ScalarField nu_t(mesh);
    
    PopeQuadraticEARSM model;
    model.set_nu(0.001);
    
    assert(model.name() == "Pope-Quadratic");
    
    model.compute_nu_t(mesh, vel, k, omega, nu_t);
    
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            assert(std::isfinite(nu_t(i, j)));
            assert(nu_t(i, j) >= 0.0);
        }
    }
    
    std::cout << "PASSED\n";
}

void test_sst_with_earsm() {
    std::cout << "Testing SST + EARSM combined model... ";
    
    Mesh mesh;
    mesh.init_uniform(16, 32, 0.0, 2.0, -1.0, 1.0);
    
    VectorField vel(mesh);
    for (int j = 0; j < mesh.total_Ny(); ++j) {
        for (int i = 0; i < mesh.total_Nx(); ++i) {
            vel.u(i, j) = 0.5 * (mesh.y(j) + 1.0);
            vel.v(i, j) = 0.0;
        }
    }
    
    ScalarField k(mesh, 0.01);
    ScalarField omega(mesh, 100.0);
    ScalarField nu_t(mesh, 0.0);
    TensorField tau_ij(mesh);
    
    SSTWithEARSM model(EARSMType::WallinJohansson2000);
    model.set_nu(0.001);
    model.set_delta(1.0);
    model.initialize(mesh, vel);
    
    assert(model.uses_transport_equations());
    assert(model.provides_reynolds_stresses());
    
    // Take transport steps
    double dt = 0.001;
    for (int step = 0; step < 3; ++step) {
        model.advance_turbulence(mesh, vel, dt, k, omega, nu_t);
        model.update(mesh, vel, k, omega, nu_t, &tau_ij);
    }
    
    // Check validity
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            assert(std::isfinite(k(i, j)));
            assert(std::isfinite(omega(i, j)));
            assert(std::isfinite(nu_t(i, j)));
            assert(std::isfinite(tau_ij.xx(i, j)));
            assert(std::isfinite(tau_ij.xy(i, j)));
            assert(std::isfinite(tau_ij.yy(i, j)));
        }
    }
    
    std::cout << "PASSED\n";
}

void test_factory_functions() {
    std::cout << "Testing turbulence model factory functions... ";
    
    // Test transport model factory
    auto sst = create_transport_model("SST");
    assert(sst != nullptr);
    assert(sst->uses_transport_equations());
    
    auto komega = create_transport_model("KOmega");
    assert(komega != nullptr);
    
    // Test EARSM closure factory
    auto wj = create_earsm_closure("WJ");
    assert(wj != nullptr);
    assert(wj->name() == "WJ-EARSM");
    
    auto gs = create_earsm_closure("GS");
    assert(gs != nullptr);
    assert(gs->name() == "GS-EARSM");
    
    auto pope = create_earsm_closure("Pope");
    assert(pope != nullptr);
    
    // Test main factory with new model types
    auto sst_model = create_turbulence_model(TurbulenceModelType::SSTKOmega);
    assert(sst_model != nullptr);
    assert(sst_model->uses_transport_equations());
    
    auto earsm_wj = create_turbulence_model(TurbulenceModelType::EARSM_WJ);
    assert(earsm_wj != nullptr);
    assert(earsm_wj->uses_transport_equations());
    assert(earsm_wj->provides_reynolds_stresses());
    
    std::cout << "PASSED\n";
}

int main() {
    std::cout << "=== Turbulence Model Tests ===\n\n";
    
    // Original tests
    test_baseline_model();
    test_gep_model();
    test_nn_mlp_model();
    test_nn_tbnn_model();
    
    // New transport model tests
    std::cout << "\n--- Transport Model Tests ---\n";
    test_sst_komega_transport();
    test_komega_transport();
    
    // EARSM tests
    std::cout << "\n--- EARSM Tests ---\n";
    test_wallin_johansson_earsm();
    test_gatski_speziale_earsm();
    test_pope_quadratic_earsm();
    test_sst_with_earsm();
    
    // Factory tests
    std::cout << "\n--- Factory Tests ---\n";
    test_factory_functions();
    
    std::cout << "\nAll turbulence model tests completed!\n";
    return 0;
}

