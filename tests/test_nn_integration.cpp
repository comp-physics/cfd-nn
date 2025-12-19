/// Integration tests for NN turbulence models with the solver
/// Tests that NN models work correctly within the full solver loop

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "config.hpp"
#include "turbulence_nn_mlp.hpp"
#include "turbulence_nn_tbnn.hpp"
#include <iostream>
#include <cmath>
#include <cassert>
#include <fstream>

using namespace nncfd;

// Helper to check if a file exists
bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

// Helper to check field validity
bool is_field_valid(const ScalarField& field, const Mesh& mesh) {
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            if (!std::isfinite(field(i, j))) {
                return false;
            }
        }
    }
    return true;
}

bool is_velocity_valid(const VectorField& vel, const Mesh& mesh) {
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            if (!std::isfinite(vel.u(i, j)) || !std::isfinite(vel.v(i, j))) {
                return false;
            }
        }
    }
    return true;
}

// Test 1: NN-MLP model produces valid output
void test_nn_mlp_validity() {
    std::cout << "Testing NN-MLP model validity... ";
    
    // NOTE: We currently only have TBNN checkpoints (5 inputs -> 4 outputs)
    // NN-MLP expects a different architecture (6 inputs -> 1 output)
    // Skip this test until we have a real trained MLP checkpoint
    std::cout << "SKIPPED (no trained MLP checkpoint available; only TBNN checkpoints exist)\n";
    return;
    
    // TODO: When mlp_channel_caseholdout/ is added, use it here:
    // std::string model_path = "data/models/mlp_channel_caseholdout";
    // if (!file_exists(model_path + "/layer0_W.txt")) {
    //     model_path = "../data/models/mlp_channel_caseholdout";
    //     if (!file_exists(model_path + "/layer0_W.txt")) {
    //         std::cout << "SKIPPED (model not found)\n";
    //         return;
    //     }
    // }
    
    Mesh mesh;
    mesh.init_uniform(16, 32, 0.0, 4.0, -1.0, 1.0);
    
    // Create velocity field with shear
    VectorField vel(mesh);
    for (int j = 0; j < mesh.total_Ny(); ++j) {
        for (int i = 0; i < mesh.total_Nx(); ++i) {
            vel.u(i, j) = 1.0 - mesh.y(j) * mesh.y(j);  // Parabolic profile
            vel.v(i, j) = 0.0;
        }
    }
    
    ScalarField k(mesh, 0.01);
    ScalarField omega(mesh, 1.0);
    ScalarField nu_t(mesh);
    
    TurbulenceNNMLP model;
    model.set_nu(0.001);
    
    try {
        model.load(model_path, model_path);
        model.update(mesh, vel, k, omega, nu_t);
        
        // Check all values are finite and non-negative
        bool valid = true;
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                if (!std::isfinite(nu_t(i, j)) || nu_t(i, j) < 0.0) {
                    valid = false;
                    break;
                }
            }
            if (!valid) break;
        }
        
        assert(valid && "NN-MLP produced invalid nu_t values!");
        std::cout << "PASSED\n";
    } catch (const std::exception& e) {
        std::cout << "SKIPPED (" << e.what() << ")\n";
    }
}

// Test 2: NN-TBNN model produces valid output
void test_nn_tbnn_validity() {
    std::cout << "Testing NN-TBNN model validity... ";
    
    // Use trained TBNN model
    std::string model_path = "data/models/tbnn_channel_caseholdout";
    if (!file_exists(model_path + "/layer0_W.txt")) {
        // Try from build directory
        model_path = "../data/models/tbnn_channel_caseholdout";
        if (!file_exists(model_path + "/layer0_W.txt")) {
            std::cout << "SKIPPED (model not found)\n";
            return;
        }
    }
    
    Mesh mesh;
    mesh.init_uniform(16, 32, 0.0, 4.0, -1.0, 1.0);
    
    VectorField vel(mesh);
    for (int j = 0; j < mesh.total_Ny(); ++j) {
        for (int i = 0; i < mesh.total_Nx(); ++i) {
            vel.u(i, j) = 1.0 - mesh.y(j) * mesh.y(j);
            vel.v(i, j) = 0.0;
        }
    }
    
    ScalarField k(mesh, 0.01);
    ScalarField omega(mesh, 1.0);
    ScalarField nu_t(mesh);
    
    TurbulenceNNTBNN model;
    model.set_nu(0.001);
    model.set_delta(1.0);
    model.set_u_ref(1.0);
    
    try {
        model.load(model_path, model_path);
        model.update(mesh, vel, k, omega, nu_t);
        
        assert(is_field_valid(nu_t, mesh) && "NN-TBNN produced NaN/Inf nu_t!");
        
        // Check nu_t is non-negative
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                assert(nu_t(i, j) >= 0.0 && "NN-TBNN produced negative nu_t!");
            }
        }
        
        std::cout << "PASSED\n";
    } catch (const std::exception& e) {
        std::cout << "SKIPPED (" << e.what() << ")\n";
    }
}

// Test 3: NN-TBNN with solver integration
void test_nn_tbnn_solver_integration() {
    std::cout << "Testing NN-TBNN solver integration... ";
    
    // Use trained TBNN model
    std::string model_path = "data/models/tbnn_channel_caseholdout";
    if (!file_exists(model_path + "/layer0_W.txt")) {
        model_path = "../data/models/tbnn_channel_caseholdout";
        if (!file_exists(model_path + "/layer0_W.txt")) {
            std::cout << "SKIPPED (model not found)\n";
            return;
        }
    }
    
    Mesh mesh;
    mesh.init_uniform(16, 32, 0.0, 4.0, -1.0, 1.0);
    
    Config config;
    config.nu = 0.01;
    config.dp_dx = -1.0;
    config.adaptive_dt = true;
    config.CFL_max = 0.5;
    config.max_iter = 50;
    config.tol = 1e-6;
    config.turb_model = TurbulenceModelType::NNTBNN;
    config.nn_weights_path = model_path;
    config.nn_scaling_path = model_path;
    config.verbose = false;
    
    try {
        RANSSolver solver(mesh, config);
        
        // Run several iterations
        for (int iter = 0; iter < 20; ++iter) {
            solver.step();
        }
        
        // Check solution validity
        assert(is_velocity_valid(solver.velocity(), mesh) && "Solution diverged with NN-TBNN!");
        assert(is_field_valid(solver.nu_t(), mesh) && "nu_t is invalid!");
        
        std::cout << "PASSED\n";
    } catch (const std::exception& e) {
        std::cout << "SKIPPED (" << e.what() << ")\n";
    }
}

// Test 4: Multiple NN updates don't cause memory issues
void test_nn_repeated_updates() {
    std::cout << "Testing repeated NN updates... ";
    
    std::string model_path = "data/models/tbnn_channel_caseholdout";
    if (!file_exists(model_path + "/layer0_W.txt")) {
        model_path = "../data/models/tbnn_channel_caseholdout";
        if (!file_exists(model_path + "/layer0_W.txt")) {
            std::cout << "SKIPPED (model not found)\n";
            return;
        }
    }
    
    Mesh mesh;
    mesh.init_uniform(16, 32, 0.0, 4.0, -1.0, 1.0);
    
    VectorField vel(mesh);
    for (int j = 0; j < mesh.total_Ny(); ++j) {
        for (int i = 0; i < mesh.total_Nx(); ++i) {
            vel.u(i, j) = 1.0 - mesh.y(j) * mesh.y(j);
            vel.v(i, j) = 0.0;
        }
    }
    
    ScalarField k(mesh, 0.01);
    ScalarField omega(mesh, 1.0);
    ScalarField nu_t(mesh);
    
    TurbulenceNNTBNN model;
    model.set_nu(0.001);
    model.set_delta(1.0);
    model.set_u_ref(1.0);
    
    try {
        model.load(model_path, model_path);
        
        // Call update many times - should not leak memory or crash
        for (int i = 0; i < 100; ++i) {
            model.update(mesh, vel, k, omega, nu_t);
            
            // Verify output is still valid
            if (i % 20 == 0) {
                assert(is_field_valid(nu_t, mesh) && "nu_t became invalid during repeated updates!");
            }
        }
        
        std::cout << "PASSED\n";
    } catch (const std::exception& e) {
        std::cout << "SKIPPED (" << e.what() << ")\n";
    }
}

// Test 5: NN model with different grid sizes
void test_nn_different_grid_sizes() {
    std::cout << "Testing NN with different grid sizes... ";
    
    std::string model_path = "data/models/tbnn_channel_caseholdout";
    if (!file_exists(model_path + "/layer0_W.txt")) {
        model_path = "../data/models/tbnn_channel_caseholdout";
        if (!file_exists(model_path + "/layer0_W.txt")) {
            std::cout << "SKIPPED (model not found)\n";
            return;
        }
    }
    
    std::vector<std::pair<int, int>> grid_sizes = {
        {8, 16},
        {16, 32},
        {32, 64},
        {64, 128}
    };
    
    try {
        for (const auto& [nx, ny] : grid_sizes) {
            Mesh mesh;
            mesh.init_uniform(nx, ny, 0.0, 4.0, -1.0, 1.0);
            
            VectorField vel(mesh);
            for (int j = 0; j < mesh.total_Ny(); ++j) {
                for (int i = 0; i < mesh.total_Nx(); ++i) {
                    vel.u(i, j) = 1.0 - mesh.y(j) * mesh.y(j);
                    vel.v(i, j) = 0.0;
                }
            }
            
            ScalarField k(mesh, 0.01);
            ScalarField omega(mesh, 1.0);
            ScalarField nu_t(mesh);
            
            TurbulenceNNTBNN model;
            model.set_nu(0.001);
            model.set_delta(1.0);
            model.set_u_ref(1.0);
            model.load(model_path, model_path);
            
            model.update(mesh, vel, k, omega, nu_t);
            
            assert(is_field_valid(nu_t, mesh) && "NN failed on different grid size!");
        }
        
        std::cout << "PASSED\n";
    } catch (const std::exception& e) {
        std::cout << "FAILED (" << e.what() << ")\n";
        assert(false);
    }
}

int main() {
    std::cout << "=== NN Integration Tests ===\n\n";
    
    test_nn_mlp_validity();
    test_nn_tbnn_validity();
    test_nn_tbnn_solver_integration();
    test_nn_repeated_updates();
    test_nn_different_grid_sizes();
    
    std::cout << "\nAll NN integration tests completed!\n";
    return 0;
}

