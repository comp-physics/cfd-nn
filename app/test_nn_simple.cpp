/// Simple test program to isolate NN-MLP issues

#include "mesh.hpp"
#include "fields.hpp"
#include "nn_core.hpp"
#include "features.hpp"
#include "turbulence_nn_mlp.hpp"

#include <iostream>
#include <iomanip>

using namespace nncfd;

int main() {
    std::cout << "=== Testing NN Components ===\n\n";
    
    // Test 1: Load MLP weights
    std::cout << "Test 1: Loading MLP weights\n";
    try {
        MLP mlp;
        mlp.load_weights("../data");
        std::cout << "  [OK] MLP loaded successfully\n";
        std::cout << "  Input dim: " << mlp.input_dim() << "\n";
        std::cout << "  Output dim: " << mlp.output_dim() << "\n";
        std::cout << "  Num layers: " << mlp.num_layers() << "\n";
        
        // Test forward pass with simple input
        std::vector<double> test_input = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        std::vector<double> output = mlp.forward(test_input);
        std::cout << "  Test forward pass:\n";
        std::cout << "    Input: [";
        for (size_t i = 0; i < test_input.size(); ++i) {
            std::cout << test_input[i];
            if (i < test_input.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
        std::cout << "    Output: " << output[0] << "\n";
        
        if (std::isnan(output[0]) || std::isinf(output[0])) {
            std::cout << "  [FAIL] ERROR: Output is NaN or inf!\n";
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cout << "  [FAIL] Error loading MLP: " << e.what() << "\n";
        return 1;
    }
    
    // Test 2: Create simple mesh and compute features
    std::cout << "\nTest 2: Creating mesh and computing features\n";
    Mesh mesh;
    mesh.init_uniform(8, 8, 0.0, 1.0, -1.0, 1.0);
    std::cout << "  [OK] Mesh created: " << mesh.Nx << "x" << mesh.Ny << "\n";
    
    VectorField velocity(mesh, 1.0, 0.0);
    ScalarField k(mesh, 0.01);
    ScalarField omega(mesh, 1.0);
    
    std::cout << "  [OK] Fields created\n";
    
    // Test feature computation
    FeatureComputer feat_comp(mesh);
    feat_comp.set_reference(0.1, 1.0, 1.0);
    
    std::vector<Features> features;
    feat_comp.compute_scalar_features(velocity, k, omega, features);
    
    std::cout << "  [OK] Features computed: " << features.size() << " cells\n";
    
    if (!features.empty()) {
        std::cout << "  Sample feature vector (cell 0):\n";
        for (int i = 0; i < std::min(6, (int)features[0].size()); ++i) {
            std::cout << "    [" << i << "]: " << features[0][i];
            if (std::isnan(features[0][i]) || std::isinf(features[0][i])) {
                std::cout << " <- NaN/inf detected!";
            }
            std::cout << "\n";
        }
    }
    
    // Test 3: Full TurbulenceNNMLP
    std::cout << "\nTest 3: Testing TurbulenceNNMLP\n";
    try {
        TurbulenceNNMLP turb_model;
        turb_model.load("../data", "../data");
        turb_model.set_nu(0.1);
        turb_model.set_delta(1.0);
        turb_model.set_u_ref(1.0);
        turb_model.set_nu_t_max(0.1);
        
        std::cout << "  [OK] Turbulence model loaded\n";
        
        ScalarField nu_t(mesh);
        turb_model.update(mesh, velocity, k, omega, nu_t);
        
        std::cout << "  [OK] Turbulence model updated\n";
        
        // Check nu_t values
        double min_nut = 1e10;
        double max_nut = -1e10;
        int nan_count = 0;
        int inf_count = 0;
        
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double val = nu_t(i, j);
                if (std::isnan(val)) {
                    nan_count++;
                } else if (std::isinf(val)) {
                    inf_count++;
                } else {
                    min_nut = std::min(min_nut, val);
                    max_nut = std::max(max_nut, val);
                }
            }
        }
        
        std::cout << "  nu_t statistics:\n";
        std::cout << "    Min: " << min_nut << "\n";
        std::cout << "    Max: " << max_nut << "\n";
        std::cout << "    NaN count: " << nan_count << "\n";
        std::cout << "    Inf count: " << inf_count << "\n";
        
        if (nan_count > 0 || inf_count > 0) {
            std::cout << "  [FAIL] ERROR: NaN or inf values detected in nu_t!\n";
            
            // Print first few nu_t values for debugging
            std::cout << "  First few nu_t values:\n";
            int count = 0;
            for (int j = mesh.j_begin(); j < mesh.j_end() && count < 5; ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end() && count < 5; ++i) {
                    std::cout << "    (" << i << "," << j << "): " << nu_t(i, j) << "\n";
                    count++;
                }
            }
            return 1;
        }
        
        std::cout << "  [OK] All nu_t values are finite\n";
        
    } catch (const std::exception& e) {
        std::cout << "  [FAIL] Error: " << e.what() << "\n";
        return 1;
    }
    
    std::cout << "\n=== All tests passed! ===\n";
    return 0;
}

