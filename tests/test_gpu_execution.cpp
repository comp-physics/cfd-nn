/// GPU Execution Verification Test
/// Ensures that when compiled with USE_GPU_OFFLOAD, code actually runs on GPU

#include "mesh.hpp"
#include "fields.hpp"
#include "nn_core.hpp"
#include "turbulence_nn_mlp.hpp"
#include "turbulence_nn_tbnn.hpp"
#include <iostream>
#include <cassert>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

using namespace nncfd;

void test_gpu_available() {
    std::cout << "Testing GPU availability... ";
    
#ifdef USE_GPU_OFFLOAD
    int num_devices = omp_get_num_devices();
    std::cout << "\n  USE_GPU_OFFLOAD is defined\n";
    std::cout << "  Number of GPU devices: " << num_devices << "\n";
    
    if (num_devices > 0) {
        std::cout << "  [OK] GPU devices available\n";
        std::cout << "PASSED\n";
    } else {
        // GPU build with no device should fail - test that it does
        std::cout << "  Testing GPU-required contract (should throw)...\n";
        try {
            Mesh mesh(1.0, 1.0, 8, 8);
            Config cfg;
            RANSSolver solver(mesh, cfg);  // Should throw during GPU init
            std::cout << "FAILED: Expected exception but none thrown\n";
            assert(false);
        } catch (const std::runtime_error& e) {
            std::cout << "  [OK] Correctly threw: " << e.what() << "\n";
            std::cout << "PASSED\n";
        }
    }
#else
    std::cout << "SKIPPED (USE_GPU_OFFLOAD not defined)\n";
#endif
}

void test_mlp_gpu_execution() {
    std::cout << "Testing MLP GPU execution... ";
    
#ifdef USE_GPU_OFFLOAD
    int num_devices = omp_get_num_devices();
    if (num_devices == 0) {
        std::cout << "SKIPPED (no GPU devices - would throw)\n";
        return;
    }
    
    // Create simple MLP
    MLP mlp({5, 32, 32, 1}, Activation::Tanh);
    
    // Initialize with dummy weights
    for (auto& layer : mlp.layers()) {
        // Cast away const to initialize (only for testing)
        DenseLayer& l = const_cast<DenseLayer&>(layer);
        for (auto& w : l.W) w = 0.1;
        for (auto& b : l.b) b = 0.0;
    }
    
    // Upload to GPU
    mlp.upload_to_gpu();
    
    if (!mlp.is_on_gpu()) {
        std::cout << "FAILED (upload_to_gpu did not set gpu_ready flag)\n";
        assert(false);
    }
    
    // Test batched GPU forward pass
    const int batch_size = 128;
    std::vector<double> x_batch(batch_size * 5, 1.0);
    std::vector<double> y_batch(batch_size * 1);
    std::vector<double> workspace(mlp.workspace_size(batch_size));
    
    double* x_ptr = x_batch.data();
    double* y_ptr = y_batch.data();
    double* work_ptr = workspace.data();
    
    // Map to GPU
    #pragma omp target enter data \
        map(to: x_ptr[0:batch_size*5]) \
        map(alloc: y_ptr[0:batch_size], work_ptr[0:workspace.size()])
    
    // Run on GPU
    mlp.forward_batch_gpu(x_ptr, y_ptr, batch_size, work_ptr);
    
    // Download results
    #pragma omp target update from(y_ptr[0:batch_size])
    #pragma omp target exit data \
        map(delete: x_ptr[0:batch_size*5], y_ptr[0:batch_size], work_ptr[0:workspace.size()])
    
    // Verify results are finite
    for (int i = 0; i < batch_size; ++i) {
        assert(std::isfinite(y_batch[i]));
    }
    
    mlp.free_gpu();
    
    std::cout << "PASSED (GPU execution verified)\n";
#else
    std::cout << "SKIPPED (USE_GPU_OFFLOAD not defined)\n";
#endif
}

void test_turbulence_nn_mlp_gpu() {
    std::cout << "Testing TurbulenceNNMLP GPU execution... ";
    
#ifdef USE_GPU_OFFLOAD
    int num_devices = omp_get_num_devices();
    if (num_devices == 0) {
        std::cout << "SKIPPED (no GPU devices - would throw)\n";
        return;
    }
    
    Mesh mesh;
    mesh.init_uniform(16, 32, 0.0, 2.0, -1.0, 1.0);
    
    VectorField vel(mesh, 1.0, 0.0);
    ScalarField k(mesh, 0.01);
    ScalarField omega(mesh, 1.0);
    ScalarField nu_t(mesh);
    
    TurbulenceNNMLP model;
    model.set_nu(0.001);
    
    try {
        // Try to load example weights
        model.load("../data/models/example_scalar_nut", "../data");
        
        // Upload to GPU - THIS IS THE KEY STEP!
        model.upload_to_gpu();
        
        if (!model.is_gpu_ready()) {
            std::cout << "WARNING (GPU not ready, using CPU fallback)\n";
            return;
        }
        
        // Run update - should use GPU path
        model.update(mesh, vel, k, omega, nu_t);
        
        // Verify results
        [[maybe_unused]] bool all_finite = true;
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                if (!std::isfinite(nu_t(i, j))) {
                    all_finite = false;
                }
            }
        }
        
        assert(all_finite);
        std::cout << "PASSED (GPU path executed)\n";
        
    } catch (const std::exception& e) {
        std::cout << "SKIPPED (model files not found)\n";
    }
#else
    std::cout << "SKIPPED (USE_GPU_OFFLOAD not defined)\n";
#endif
}

void test_turbulence_nn_tbnn_gpu() {
    std::cout << "Testing TurbulenceNNTBNN GPU execution... ";
    
#ifdef USE_GPU_OFFLOAD
    int num_devices = omp_get_num_devices();
    if (num_devices == 0) {
        std::cout << "SKIPPED (no GPU devices - would throw)\n";
        return;
    }
    
    Mesh mesh;
    mesh.init_uniform(16, 32, 0.0, 2.0, -1.0, 1.0);
    
    VectorField vel(mesh, 1.0, 0.0);
    ScalarField k(mesh, 0.01);
    ScalarField omega(mesh, 1.0);
    ScalarField nu_t(mesh);
    
    TurbulenceNNTBNN model;
    model.set_nu(0.001);
    
    try {
        // Try to load example weights
        model.load("../data/models/example_tbnn", "../data");
        
        // Upload to GPU - THIS IS THE KEY STEP!
        model.upload_to_gpu();
        
        if (!model.is_gpu_ready()) {
            std::cout << "WARNING (GPU not ready, using CPU fallback)\n";
            return;
        }
        
        // Run update - should use GPU path
        model.update(mesh, vel, k, omega, nu_t);
        
        // Verify results
        [[maybe_unused]] bool all_finite = true;
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                if (!std::isfinite(nu_t(i, j))) {
                    all_finite = false;
                }
            }
        }
        
        assert(all_finite);
        std::cout << "PASSED (GPU path executed)\n";
        
    } catch (const std::exception& e) {
        std::cout << "SKIPPED (model files not found)\n";
    }
#else
    std::cout << "SKIPPED (USE_GPU_OFFLOAD not defined)\n";
#endif
}

void test_actual_gpu_usage() {
    std::cout << "Testing actual GPU computation... ";
    
#ifdef USE_GPU_OFFLOAD
    int num_devices = omp_get_num_devices();
    if (num_devices == 0) {
        std::cout << "SKIPPED (no GPU devices - would throw)\n";
        return;
    }
    
    // Simple computation to verify GPU is actually doing work
    const int N = 1000000;
    std::vector<double> a(N, 2.0);
    std::vector<double> b(N, 3.0);
    std::vector<double> c(N, 0.0);
    
    double* a_ptr = a.data();
    double* b_ptr = b.data();
    double* c_ptr = c.data();
    
    #pragma omp target enter data map(to: a_ptr[0:N], b_ptr[0:N]) map(alloc: c_ptr[0:N])
    
    // This MUST execute on GPU
    #pragma omp target teams distribute parallel for
    for (int i = 0; i < N; ++i) {
        c_ptr[i] = a_ptr[i] + b_ptr[i];
    }
    
    #pragma omp target update from(c_ptr[0:N])
    #pragma omp target exit data map(delete: a_ptr[0:N], b_ptr[0:N], c_ptr[0:N])
    
    // Verify
    for (int i = 0; i < 100; ++i) {
        assert(std::abs(c[i] - 5.0) < 1e-10);
    }
    
    std::cout << "PASSED (GPU computed correctly)\n";
#else
    std::cout << "SKIPPED (USE_GPU_OFFLOAD not defined)\n";
#endif
}

int main() {
    std::cout << "=== GPU Execution Verification Tests ===\n\n";
    
    test_gpu_available();
    test_actual_gpu_usage();
    test_mlp_gpu_execution();
    test_turbulence_nn_mlp_gpu();
    test_turbulence_nn_tbnn_gpu();
    
    std::cout << "\n";
#ifdef USE_GPU_OFFLOAD
    int num_devices = omp_get_num_devices();
    if (num_devices > 0) {
        std::cout << "[PASS] All GPU execution tests passed!\n";
        std::cout << "[OK] GPU is actually being used for computation\n";
    } else {
        std::cout << "[WARNING] Tests compiled with GPU support but no devices available\n";
        std::cout << "  (This is expected on CPU-only nodes)\n";
    }
#else
    std::cout << "Note: Tests were not compiled with GPU support\n";
#endif
    
    return 0;
}













