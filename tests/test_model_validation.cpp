/// @file test_model_validation.cpp
/// @brief Test that model loading validates weights/biases for NaN/Inf

#include "nn_core.hpp"
#include <cassert>
#include <iostream>
#include <fstream>
#include <cmath>
#include <stdexcept>
#include <filesystem>

namespace fs = std::filesystem;

void write_test_file(const std::string& filename, const std::string& content) {
    std::ofstream f(filename);
    f << content;
    f.close();
}

void test_valid_weights() {
    std::cout << "\n=== Test: Valid weights should load successfully ===\n";
    
    fs::create_directories("test_model_valid");
    
    // Write valid weight/bias files
    write_test_file("test_model_valid/layer0_W.txt", "0.1 0.2\n0.3 0.4\n");
    write_test_file("test_model_valid/layer0_b.txt", "0.5\n0.6\n");
    
    try {
        nncfd::MLP mlp;
        mlp.load_weights("test_model_valid");
        
        assert(mlp.num_layers() == 1);
        assert(mlp.input_dim() == 2);
        assert(mlp.output_dim() == 2);
        
        std::cout << "✓ Valid weights loaded successfully\n";
    } catch (const std::exception& e) {
        std::cerr << "✗ FAILED: Valid weights rejected: " << e.what() << "\n";
        throw;
    }
    
    fs::remove_all("test_model_valid");
}

void test_nan_in_weights() {
    std::cout << "\n=== Test: NaN in weights should be rejected ===\n";
    
    fs::create_directories("test_model_nan_w");
    
    // Write weight file with NaN
    write_test_file("test_model_nan_w/layer0_W.txt", "0.1 nan\n0.3 0.4\n");
    write_test_file("test_model_nan_w/layer0_b.txt", "0.5\n0.6\n");
    
    try {
        nncfd::MLP mlp;
        mlp.load_weights("test_model_nan_w");
        
        std::cerr << "✗ FAILED: NaN in weights was not detected!\n";
        fs::remove_all("test_model_nan_w");
        throw std::runtime_error("NaN validation failed");
    } catch (const std::runtime_error& e) {
        std::string msg(e.what());
        if (msg.find("NaN") != std::string::npos || msg.find("Invalid") != std::string::npos) {
            std::cout << "✓ NaN in weights correctly rejected\n";
            std::cout << "  Error message: " << e.what() << "\n";
        } else {
            std::cerr << "✗ FAILED: Wrong error message: " << e.what() << "\n";
            fs::remove_all("test_model_nan_w");
            throw;
        }
    }
    
    fs::remove_all("test_model_nan_w");
}

void test_inf_in_biases() {
    std::cout << "\n=== Test: Inf in biases should be rejected ===\n";
    
    fs::create_directories("test_model_inf_b");
    
    // Write bias file with Inf
    write_test_file("test_model_inf_b/layer0_W.txt", "0.1 0.2\n0.3 0.4\n");
    write_test_file("test_model_inf_b/layer0_b.txt", "0.5\ninf\n");
    
    try {
        nncfd::MLP mlp;
        mlp.load_weights("test_model_inf_b");
        
        std::cerr << "✗ FAILED: Inf in biases was not detected!\n";
        fs::remove_all("test_model_inf_b");
        throw std::runtime_error("Inf validation failed");
    } catch (const std::runtime_error& e) {
        std::string msg(e.what());
        if (msg.find("Inf") != std::string::npos || msg.find("Invalid") != std::string::npos) {
            std::cout << "✓ Inf in biases correctly rejected\n";
            std::cout << "  Error message: " << e.what() << "\n";
        } else {
            std::cerr << "✗ FAILED: Wrong error message: " << e.what() << "\n";
            fs::remove_all("test_model_inf_b");
            throw;
        }
    }
    
    fs::remove_all("test_model_inf_b");
}

void test_negative_inf_in_weights() {
    std::cout << "\n=== Test: -Inf in weights should be rejected ===\n";
    
    fs::create_directories("test_model_neginf");
    
    // Write weight file with -Inf
    write_test_file("test_model_neginf/layer0_W.txt", "0.1 0.2\n-inf 0.4\n");
    write_test_file("test_model_neginf/layer0_b.txt", "0.5\n0.6\n");
    
    try {
        nncfd::MLP mlp;
        mlp.load_weights("test_model_neginf");
        
        std::cerr << "✗ FAILED: -Inf in weights was not detected!\n";
        fs::remove_all("test_model_neginf");
        throw std::runtime_error("-Inf validation failed");
    } catch (const std::runtime_error& e) {
        std::string msg(e.what());
        if (msg.find("Inf") != std::string::npos || msg.find("Invalid") != std::string::npos) {
            std::cout << "✓ -Inf in weights correctly rejected\n";
            std::cout << "  Error message: " << e.what() << "\n";
        } else {
            std::cerr << "✗ FAILED: Wrong error message: " << e.what() << "\n";
            fs::remove_all("test_model_neginf");
            throw;
        }
    }
    
    fs::remove_all("test_model_neginf");
}

void test_valid_normalization() {
    std::cout << "\n=== Test: Valid normalization should load successfully ===\n";
    
    fs::create_directories("test_model_norm_valid");
    
    write_test_file("test_model_norm_valid/layer0_W.txt", "0.1 0.2\n0.3 0.4\n");
    write_test_file("test_model_norm_valid/layer0_b.txt", "0.5\n0.6\n");
    write_test_file("test_model_norm_valid/input_means.txt", "0.1\n0.2\n");
    write_test_file("test_model_norm_valid/input_stds.txt", "1.0\n2.0\n");
    
    try {
        nncfd::MLP mlp;
        mlp.load_weights("test_model_norm_valid");
        mlp.load_scaling("test_model_norm_valid/input_means.txt",
                        "test_model_norm_valid/input_stds.txt");
        
        assert(mlp.has_scaling());
        std::cout << "✓ Valid normalization loaded successfully\n";
    } catch (const std::exception& e) {
        std::cerr << "✗ FAILED: Valid normalization rejected: " << e.what() << "\n";
        fs::remove_all("test_model_norm_valid");
        throw;
    }
    
    fs::remove_all("test_model_norm_valid");
}

void test_inf_in_normalization() {
    std::cout << "\n=== Test: Inf in normalization should be rejected ===\n";
    
    fs::create_directories("test_model_norm_inf");
    
    write_test_file("test_model_norm_inf/layer0_W.txt", "0.1 0.2\n0.3 0.4\n");
    write_test_file("test_model_norm_inf/layer0_b.txt", "0.5\n0.6\n");
    write_test_file("test_model_norm_inf/input_means.txt", "0.1\n0.2\n");
    write_test_file("test_model_norm_inf/input_stds.txt", "1.0\ninf\n");
    
    try {
        nncfd::MLP mlp;
        mlp.load_weights("test_model_norm_inf");
        mlp.load_scaling("test_model_norm_inf/input_means.txt",
                        "test_model_norm_inf/input_stds.txt");
        
        std::cerr << "✗ FAILED: Inf in normalization was not detected!\n";
        fs::remove_all("test_model_norm_inf");
        throw std::runtime_error("Normalization validation failed");
    } catch (const std::runtime_error& e) {
        std::string msg(e.what());
        if (msg.find("not finite") != std::string::npos || 
            msg.find("Invalid") != std::string::npos) {
            std::cout << "✓ Inf in normalization correctly rejected\n";
        } else {
            std::cerr << "✗ FAILED: Wrong error message: " << e.what() << "\n";
            fs::remove_all("test_model_norm_inf");
            throw;
        }
    }
    
    fs::remove_all("test_model_norm_inf");
}

int main() {
    std::cout << "========================================\n";
    std::cout << "Model Validation Tests\n";
    std::cout << "========================================\n";
    
    try {
        test_valid_weights();
        test_nan_in_weights();
        test_inf_in_biases();
        test_negative_inf_in_weights();
        test_valid_normalization();
        test_inf_in_normalization();
        
        std::cout << "\n========================================\n";
        std::cout << "✓ ALL TESTS PASSED\n";
        std::cout << "========================================\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n========================================\n";
        std::cerr << "✗ TEST SUITE FAILED\n";
        std::cerr << "========================================\n";
        return 1;
    }
}

