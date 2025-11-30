/// Unit tests for turbulence models

#include "mesh.hpp"
#include "fields.hpp"
#include "turbulence_model.hpp"
#include "turbulence_baseline.hpp"
#include "turbulence_gep.hpp"
#include "turbulence_nn_mlp.hpp"
#include "turbulence_nn_tbnn.hpp"
#include <iostream>
#include <cmath>
#include <cassert>

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

int main() {
    std::cout << "=== Turbulence Model Tests ===\n\n";
    
    test_baseline_model();
    test_gep_model();
    test_nn_mlp_model();
    test_nn_tbnn_model();
    
    std::cout << "\nAll turbulence model tests completed!\n";
    return 0;
}

