/// Unified Backend Tests
/// Consolidates test_backend_execution.cpp and test_backend_canary.cpp
///
/// Tests:
/// 1. Backend availability (CPU or GPU devices present)
/// 2. Basic computation verification
/// 3. Canary test - verifies CPU/GPU produce different FP results (detects false coverage)
/// 4. NN model execution (MLP, TBNN)

#include "mesh.hpp"
#include "fields.hpp"
#include "config.hpp"
#include "nn_core.hpp"
#include "solver.hpp"
#include "turbulence_nn_mlp.hpp"
#include "turbulence_nn_tbnn.hpp"
#include "test_harness.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <vector>
#include <fstream>
#include <cassert>

using namespace nncfd;
using nncfd::test::file_exists;
using nncfd::test::harness::record;
namespace test_gpu = nncfd::test::gpu;

static std::string resolve_model_dir(const std::string& p) {
    std::string path = p;
    while (!path.empty() && path.back() == '/') path.pop_back();
    if (file_exists(path + "/layer0_W.txt")) return path;
    if (file_exists("../" + path + "/layer0_W.txt")) return "../" + path;
    return "";
}

// LCG for deterministic pseudo-random values
static double generate_value(int idx) {
    constexpr uint64_t a = 1103515245, c = 12345, m = 1ULL << 31;
    uint64_t seed = (static_cast<uint64_t>(idx) * a + c) % m;
    seed = (seed * a + c) % m;
    double val = (static_cast<double>(seed) / m) * 2.0 - 1.0;
    return val * std::pow(10.0, (idx % 10) - 5);
}

//=============================================================================
// Test 1: Backend Availability
//=============================================================================

bool test_backend_available() {
    if (test_gpu::is_gpu_build()) {
        if (test_gpu::available()) {
            record("Backend available (GPU)", true);
            return true;
        } else {
            record("Backend available (GPU build, no devices)", true);
            return false;  // No GPU devices
        }
    } else {
        record("Backend available (CPU)", true);
        return true;
    }
}

//=============================================================================
// Test 2: Basic Computation
//=============================================================================

void test_basic_computation(bool gpu_available) {
    (void)gpu_available;  // Used only in GPU builds
    const int N = 10000;
    std::vector<double> a(N, 2.0), b(N, 3.0), c(N, 0.0);

#ifdef USE_GPU_OFFLOAD
    if (!gpu_available) {
        record("Basic computation", true, true);
        return;
    }
    double* a_ptr = a.data();
    double* b_ptr = b.data();
    double* c_ptr = c.data();

    #pragma omp target enter data map(to: a_ptr[0:N], b_ptr[0:N]) map(alloc: c_ptr[0:N])
    #pragma omp target teams distribute parallel for
    for (int i = 0; i < N; ++i) c_ptr[i] = a_ptr[i] + b_ptr[i];
    #pragma omp target update from(c_ptr[0:N])
    #pragma omp target exit data map(delete: a_ptr[0:N], b_ptr[0:N], c_ptr[0:N])
#else
    for (int i = 0; i < N; ++i) c[i] = a[i] + b[i];
#endif

    bool pass = true;
    for (int i = 0; i < 100; ++i) {
        if (std::abs(c[i] - 5.0) > 1e-10) pass = false;
    }
    record("Basic computation", pass);
}

//=============================================================================
// Test 3: Canary Test (FP Non-Associativity)
//=============================================================================

void test_canary(bool gpu_available) {
    (void)gpu_available;  // Used only in GPU builds
#ifdef USE_GPU_OFFLOAD
    if (!gpu_available) {
        record("Canary (CPU/GPU FP difference)", true, true);
        return;
    }

    constexpr int N = 100000;
    constexpr double TOLERANCE = 1e-6;
    constexpr double MIN_DIFF = 1e-14;

    // CPU sequential sum
    double cpu_sum = 0.0;
    for (int i = 0; i < N; ++i) cpu_sum += generate_value(i);

    // GPU parallel sum
    double gpu_sum = 0.0;
    #pragma omp target teams distribute parallel for reduction(+:gpu_sum)
    for (int i = 0; i < N; ++i) gpu_sum += generate_value(i);

    double abs_diff = std::abs(cpu_sum - gpu_sum);
    double rel_diff = abs_diff / (std::abs(cpu_sum) + 1e-15);

    // Results should be within tolerance but NOT identical
    bool pass = (rel_diff < TOLERANCE) && (abs_diff > MIN_DIFF);
    record("Canary (CPU/GPU FP difference)", pass);
#else
    // CPU-only build - just verify sequential sum works
    constexpr int N = 100000;
    double sum = 0.0;
    for (int i = 0; i < N; ++i) sum += generate_value(i);
    record("Canary (CPU sequential sum)", std::isfinite(sum));
#endif
}

//=============================================================================
// Test 4: MLP Execution
//=============================================================================

void test_mlp_execution(bool gpu_available) {
    (void)gpu_available;  // Used only in GPU builds
    MLP mlp({5, 16, 1}, Activation::Tanh);
    for (auto& layer : mlp.layers()) {
        DenseLayer& l = const_cast<DenseLayer&>(layer);
        for (auto& w : l.W) w = 0.1;
        for (auto& b : l.b) b = 0.0;
    }

    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = mlp.forward(x);

    bool pass = (y.size() == 1) && std::isfinite(y[0]);

#ifdef USE_GPU_OFFLOAD
    if (gpu_available) {
        mlp.sync_weights_to_gpu();
        if (mlp.is_on_gpu()) {
            const int batch = 32;
            std::vector<double> xb(batch * 5, 1.0), yb(batch);
            std::vector<double> work(mlp.workspace_size(batch));
            double *xp = xb.data(), *yp = yb.data(), *wp = work.data();
            size_t ws = work.size();

            #pragma omp target enter data map(to: xp[0:batch*5]) map(alloc: yp[0:batch], wp[0:ws])
            mlp.forward_batch_gpu(xp, yp, batch, wp);
            #pragma omp target update from(yp[0:batch])
            #pragma omp target exit data map(delete: xp[0:batch*5], yp[0:batch], wp[0:ws])

            for (int i = 0; i < batch && pass; ++i) {
                if (!std::isfinite(yb[i])) pass = false;
            }
            mlp.free_gpu();
        }
    }
#endif
    record("MLP execution", pass);
}

//=============================================================================
// Test 5: Turbulence NN Models
//=============================================================================

void test_turbulence_nn(bool gpu_available) {
    (void)gpu_available;  // Used only in GPU builds
    Mesh mesh;
    mesh.init_uniform(8, 16, 0.0, 1.0, 0.0, 1.0);
    VectorField vel(mesh, 0.5, 0.0);
    ScalarField k(mesh, 0.01), omega(mesh, 1.0), nu_t(mesh);

    // Test MLP
    // Note: Direct model testing on GPU requires full solver context for device_view setup.
    // This test validates CPU path; GPU path is validated by test_turbulence_unified via solver.
    std::string mlp_path = resolve_model_dir("data/models/mlp_channel_caseholdout");
    if (mlp_path.empty()) {
        record("TurbulenceNNMLP", true, true);
    } else {
#ifdef USE_GPU_OFFLOAD
        // GPU builds: Skip direct model test - GPU pipeline requires solver-managed device_view.
        // Full GPU NN testing is done in test_turbulence_unified via RANSSolver.
        (void)mesh; (void)vel; (void)k; (void)omega; (void)nu_t;
        record("TurbulenceNNMLP (GPU: via solver)", true, true);
#else
        TurbulenceNNMLP model;
        model.set_nu(0.001);
        model.load(mlp_path, mlp_path);
        model.update(mesh, vel, k, omega, nu_t);

        bool pass = true;
        for (int j = mesh.j_begin(); j < mesh.j_end() && pass; ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end() && pass; ++i) {
                if (!std::isfinite(nu_t(i, j)) || nu_t(i, j) < 0) pass = false;
            }
        }
        record("TurbulenceNNMLP", pass);
#endif
    }

    // Test TBNN
    std::string tbnn_path = resolve_model_dir("data/models/tbnn_channel_caseholdout");
    if (tbnn_path.empty()) {
        record("TurbulenceNNTBNN", true, true);
    } else {
#ifdef USE_GPU_OFFLOAD
        // GPU builds: Skip direct model test - GPU pipeline requires solver-managed device_view.
        record("TurbulenceNNTBNN (GPU: via solver)", true, true);
#else
        TurbulenceNNTBNN model;
        model.set_nu(0.001);
        model.load(tbnn_path, tbnn_path);
        model.update(mesh, vel, k, omega, nu_t);

        bool pass = true;
        for (int j = mesh.j_begin(); j < mesh.j_end() && pass; ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end() && pass; ++i) {
                if (!std::isfinite(nu_t(i, j))) pass = false;
            }
        }
        record("TurbulenceNNTBNN", pass);
#endif
    }
}

//=============================================================================
// Main
//=============================================================================

int main() {
    using namespace nncfd::test::harness;

    return run("Unified Backend Tests", []() {
        bool gpu_avail = test_backend_available();
        test_basic_computation(gpu_avail);
        test_canary(gpu_avail);
        test_mlp_execution(gpu_avail);
        test_turbulence_nn(gpu_avail);
    });
}
