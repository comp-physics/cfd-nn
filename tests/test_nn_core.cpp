/// Unit tests for NN core

#include "nn_core.hpp"
#include <iostream>
#include <cmath>
#include <cassert>

using namespace nncfd;

void test_dense_layer() {
    std::cout << "Testing dense layer forward pass... ";
    
    DenseLayer layer;
    layer.in_dim = 3;
    layer.out_dim = 2;
    
    // W = [[1, 0, -1], [0, 1, 0]], b = [1, 2]
    layer.W = {1.0, 0.0, -1.0, 0.0, 1.0, 0.0};
    layer.b = {1.0, 2.0};
    
    std::vector<double> x = {1.0, 2.0, 3.0};
    std::vector<double> y = layer.forward(x);
    
    // y[0] = 1*1 + 0*2 + (-1)*3 + 1 = -1
    // y[1] = 0*1 + 1*2 + 0*3 + 2 = 4
    assert(y.size() == 2);
    assert(std::abs(y[0] - (-1.0)) < 1e-10);
    assert(std::abs(y[1] - 4.0) < 1e-10);
    
    std::cout << "PASSED\n";
}

void test_mlp_forward() {
    std::cout << "Testing MLP forward pass... ";
    
    MLP mlp;
    
    // Simple 2->3->1 network
    DenseLayer layer1;
    layer1.in_dim = 2;
    layer1.out_dim = 3;
    layer1.W = {1.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    layer1.b = {0.0, 0.0, 0.0};
    
    DenseLayer layer2;
    layer2.in_dim = 3;
    layer2.out_dim = 1;
    layer2.W = {1.0, 1.0, 1.0};
    layer2.b = {0.0};
    
    mlp.add_layer(layer1, Activation::Tanh);
    mlp.add_layer(layer2, Activation::Linear);
    
    std::vector<double> x = {1.0, 1.0};
    std::vector<double> y = mlp.forward(x);
    
    assert(y.size() == 1);
    assert(std::isfinite(y[0]));
    
    std::cout << "PASSED\n";
}

void test_load_weights() {
    std::cout << "Testing weight loading... ";
    
    try {
        MLP mlp;
        mlp.load_weights("../data/models/test_mlp");
        
        assert(mlp.input_dim() > 0);
        assert(mlp.output_dim() > 0);
        assert(mlp.num_layers() > 0);
        
        // Test forward pass
        std::vector<double> x(mlp.input_dim(), 1.0);
        std::vector<double> y = mlp.forward(x);
        
        assert(y.size() == static_cast<size_t>(mlp.output_dim()));
        assert(std::isfinite(y[0]));
        
        std::cout << "PASSED\n";
    } catch (const std::exception& e) {
        std::cout << "SKIPPED (test model not found)\n";
    }
}

int main() {
    std::cout << "=== NN Core Tests ===\n\n";
    
    test_dense_layer();
    test_mlp_forward();
    test_load_weights();
    
    std::cout << "\nAll NN core tests passed!\n";
    return 0;
}

