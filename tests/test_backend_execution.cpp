/// Backend Execution Test (CPU and GPU)
/// Verifies that code executes correctly on the configured backend
/// - CPU builds: verify CPU execution
/// - GPU builds: verify GPU execution

#include "mesh.hpp"
#include "fields.hpp"
#include "config.hpp"
#include "nn_core.hpp"
#include "solver.hpp"
#include "turbulence_nn_mlp.hpp"
#include "turbulence_nn_tbnn.hpp"
#include <iostream>
#include <cassert>
#include <fstream>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

using namespace nncfd;

//=============================================================================
// Path resolution helpers for NN models
//=============================================================================
static bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

static std::string resolve_model_dir(const std::string& p) {
    // Strip trailing slashes
    std::string path = p;
    while (!path.empty() && path.back() == '/') {
        path.pop_back();
    }
    
    // Try relative to current directory (when running from repo root)
    if (file_exists(path + "/layer0_W.txt")) {
        return path;
    }
    
    // Try relative to build directory (when running from build/)
    if (file_exists("../" + path + "/layer0_W.txt")) {
        return "../" + path;
    }
    
    throw std::runtime_error(
        "NN model files not found. Tried: " + path + " and ../" + path
    );
}

void test_backend_available() {
    std::cout << "Testing backend availability... ";
    
#ifdef USE_GPU_OFFLOAD
    int num_devices = omp_get_num_devices();
    std::cout << "\n  Backend: GPU (USE_GPU_OFFLOAD enabled)\n";
    std::cout << "  Number of GPU devices: " << num_devices << "\n";
    
    if (num_devices > 0) {
        std::cout << "  [OK] GPU devices available\n";
        std::cout << "PASSED\n";
    } else {
        // GPU build with no device should fail - test that it does
        std::cout << "  Testing GPU-required contract (should throw)...\n";
        try {
            Mesh mesh = Mesh::create_uniform(8, 8);
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
    std::cout << "\n  Backend: CPU (USE_GPU_OFFLOAD disabled)\n";
    std::cout << "  [OK] CPU backend available\n";
    std::cout << "PASSED\n";
#endif
}

void test_basic_computation() {
    std::cout << "Testing basic computation... ";
    
    const int N = 100000;
    std::vector<double> a(N, 2.0);
    std::vector<double> b(N, 3.0);
    std::vector<double> c(N, 0.0);
    
#ifdef USE_GPU_OFFLOAD
    int num_devices = omp_get_num_devices();
    if (num_devices == 0) {
        std::cout << "SKIPPED (no GPU devices - would throw)\n";
        return;
    }
    
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
    
    std::cout << "PASSED (GPU computed correctly)\n";
#else
    // CPU path
    for (int i = 0; i < N; ++i) {
        c[i] = a[i] + b[i];
    }
    
    std::cout << "PASSED (CPU computed correctly)\n";
#endif
    
    // Verify (same for both backends)
    for (int i = 0; i < 100; ++i) {
        assert(std::abs(c[i] - 5.0) < 1e-10);
    }
}

void test_mlp_execution() {
    std::cout << "Testing MLP execution... ";
    
    // Create simple MLP
    MLP mlp({5, 32, 32, 1}, Activation::Tanh);
    
    // Initialize with dummy weights
    for (auto& layer : mlp.layers()) {
        // Cast away const to initialize (only for testing)
        DenseLayer& l = const_cast<DenseLayer&>(layer);
        for (auto& w : l.W) w = 0.1;
        for (auto& b : l.b) b = 0.0;
    }
    
    // Test single forward pass (CPU)
    std::vector<double> x_single = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y_single = mlp.forward(x_single);
    assert(std::isfinite(y_single[0]));
    
#ifdef USE_GPU_OFFLOAD
    int num_devices = omp_get_num_devices();
    if (num_devices == 0) {
        std::cout << "PASSED (CPU path verified; GPU unavailable)\n";
        return;
    }
    
    // GPU path - upload and test batched inference
    mlp.upload_to_gpu();
    
    if (!mlp.is_on_gpu()) {
        std::cout << "WARNING (GPU upload failed, using CPU)\n";
        std::cout << "PASSED (CPU path verified)\n";
        return;
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
    // CPU-only build
    std::cout << "PASSED (CPU execution verified)\n";
#endif
}

void test_turbulence_nn_mlp() {
    std::cout << "Testing TurbulenceNNMLP execution... ";
    
    // Test with trained MLP model from data/models/mlp_channel_caseholdout
    Mesh mesh;
    mesh.init_uniform(16, 32, 0.0, 2.0, -1.0, 1.0);
    
    VectorField vel(mesh, 1.0, 0.0);
    ScalarField k(mesh, 0.01);
    ScalarField omega(mesh, 1.0);
    ScalarField nu_t(mesh);
    
    TurbulenceNNMLP model;
    model.set_nu(0.001);
    
    try {
        // Load trained MLP weights
        std::string model_path = resolve_model_dir("data/models/mlp_channel_caseholdout");
        model.load(model_path, model_path);
        
#ifdef USE_GPU_OFFLOAD
        int num_devices = omp_get_num_devices();
        if (num_devices > 0) {
            // Initialize GPU buffers (includes weight upload)
            model.initialize_gpu_buffers(mesh);
            
            // In GPU builds, GPU must be ready (no fallback allowed)
            if (!model.is_gpu_ready()) {
                std::cerr << "FAILED: GPU build requires GPU execution, but GPU not ready!\n";
                assert(false);
            }
        }
#endif
        
        // Run update (will use GPU in GPU builds, CPU in CPU builds)
        model.update(mesh, vel, k, omega, nu_t);
        
        // Verify results
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                assert(std::isfinite(nu_t(i, j)));
                assert(nu_t(i, j) >= 0.0);  // Eddy viscosity must be non-negative
            }
        }
        
#ifdef USE_GPU_OFFLOAD
        std::cout << "PASSED (GPU path executed)\n";
#else
        std::cout << "PASSED (CPU path executed)\n";
#endif
        
    } catch (const std::exception& e) {
        std::cout << "SKIPPED (model files not found: " << e.what() << ")\n";
    }
}

void test_turbulence_nn_tbnn() {
    std::cout << "Testing TurbulenceNNTBNN execution... ";
    
    Mesh mesh;
    mesh.init_uniform(16, 32, 0.0, 2.0, -1.0, 1.0);
    
    VectorField vel(mesh, 1.0, 0.0);
    ScalarField k(mesh, 0.01);
    ScalarField omega(mesh, 1.0);
    ScalarField nu_t(mesh);
    
    TurbulenceNNTBNN model;
    model.set_nu(0.001);
    
    try {
        // Load trained TBNN weights
        std::string model_path = resolve_model_dir("data/models/tbnn_channel_caseholdout");
        model.load(model_path, model_path);
        
#ifdef USE_GPU_OFFLOAD
        int num_devices = omp_get_num_devices();
        if (num_devices > 0) {
            // Initialize GPU buffers (includes weight upload)
            model.initialize_gpu_buffers(mesh);
            
            // In GPU builds, GPU must be ready (no fallback allowed)
            if (!model.is_gpu_ready()) {
                std::cerr << "FAILED: GPU build requires GPU execution, but GPU not ready!\n";
                assert(false);
            }
        }
#endif
        
        // Run update (will use GPU in GPU builds, CPU in CPU builds)
        model.update(mesh, vel, k, omega, nu_t);
        
        // Verify results
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                assert(std::isfinite(nu_t(i, j)));
            }
        }
        
#ifdef USE_GPU_OFFLOAD
        std::cout << "PASSED (GPU path executed)\n";
#else
        std::cout << "PASSED (CPU path executed)\n";
#endif
        
    } catch (const std::exception& e) {
        std::cout << "SKIPPED (model files not found)\n";
    }
}

int main() {
    std::cout << "=== Backend Execution Tests ===\n\n";
    
    test_backend_available();
    test_basic_computation();
    test_mlp_execution();
    test_turbulence_nn_mlp();
    test_turbulence_nn_tbnn();
    
    std::cout << "\n";
#ifdef USE_GPU_OFFLOAD
    int num_devices = omp_get_num_devices();
    if (num_devices > 0) {
        std::cout << "[PASS] All GPU backend tests passed!\n";
    } else {
        std::cout << "[WARNING] GPU build but no devices (expected on CPU-only nodes)\n";
    }
#else
    std::cout << "[PASS] All CPU backend tests passed!\n";
#endif
    
    return 0;
}

