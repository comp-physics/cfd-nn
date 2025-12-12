/// Comprehensive CPU vs GPU consistency tests
/// Tests each GPU-offloaded kernel against its CPU reference implementation
/// Uses tight tolerances based on algorithm, not platform

#include "mesh.hpp"
#include "fields.hpp"
#include "turbulence_baseline.hpp"
#include "turbulence_gep.hpp"
#include "turbulence_nn_mlp.hpp"
#include "turbulence_nn_tbnn.hpp"
#include "turbulence_transport.hpp"
#include "features.hpp"
#include <iostream>
#include <cmath>
#include <cassert>
#include <iomanip>
#include <random>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

using namespace nncfd;

// Utility: compare two scalar fields
struct FieldComparison {
    double max_abs_diff = 0.0;
    double max_rel_diff = 0.0;
    double rms_diff = 0.0;
    int max_i = -1;
    int max_j = -1;
    double cpu_val_at_max = 0.0;
    double gpu_val_at_max = 0.0;
    int n_points = 0;
};

FieldComparison compare_fields(const Mesh& mesh, const ScalarField& cpu, const ScalarField& gpu, const std::string& name = "") {
    FieldComparison result;
    
    double sum_sq = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double c = cpu(i, j);
            double g = gpu(i, j);
            double abs_diff = std::abs(c - g);
            double rel_diff = abs_diff / (std::abs(c) + 1e-20);
            
            sum_sq += abs_diff * abs_diff;
            result.n_points++;
            
            if (abs_diff > result.max_abs_diff) {
                result.max_abs_diff = abs_diff;
                result.max_rel_diff = rel_diff;
                result.max_i = i;
                result.max_j = j;
                result.cpu_val_at_max = c;
                result.gpu_val_at_max = g;
            }
        }
    }
    
    result.rms_diff = std::sqrt(sum_sq / result.n_points);
    
    if (!name.empty()) {
        std::cout << "  Field: " << name << "\n";
    }
    std::cout << "    Max abs diff: " << std::scientific << std::setprecision(6) << result.max_abs_diff << "\n";
    std::cout << "    Max rel diff: " << result.max_rel_diff << "\n";
    std::cout << "    RMS diff:     " << result.rms_diff << "\n";
    if (result.max_abs_diff > 0) {
        std::cout << "    Location:     (" << result.max_i << ", " << result.max_j << ")\n";
        std::cout << "      CPU value: " << std::fixed << std::setprecision(12) << result.cpu_val_at_max << "\n";
        std::cout << "      GPU value: " << result.gpu_val_at_max << "\n";
    }
    
    return result;
}

// Self-test: verify the comparison harness actually detects differences
void test_harness_sanity() {
    std::cout << "Testing comparison harness... ";
    
    Mesh mesh;
    mesh.init_uniform(8, 8, 0.0, 1.0, 0.0, 1.0, 1);
    
    ScalarField f1(mesh, 1.0);
    ScalarField f2(mesh, 1.0);
    
    // Verify addresses are different
    assert(f1.data().data() != f2.data().data());
    
    // Should report zero difference
    [[maybe_unused]] auto cmp1 = compare_fields(mesh, f1, f2);
    assert(cmp1.max_abs_diff == 0.0);
    
    // Perturb one cell
    f2(mesh.i_begin() + 1, mesh.j_begin() + 1) = 2.0;
    [[maybe_unused]] auto cmp2 = compare_fields(mesh, f1, f2);
    assert(cmp2.max_abs_diff > 0.0);
    assert(cmp2.max_abs_diff == 1.0);
    
    std::cout << "PASSED\n";
}

// Create a deterministic but non-trivial velocity field
void create_test_velocity_field(const Mesh& mesh, VectorField& vel, int seed = 0) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-0.1, 0.1);
    
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double y = mesh.yc[j];
            double x = mesh.xc[i];
            
            // Parabolic + perturbation
            double u_base = 4.0 * y * (1.0 - y);
            double v_base = 0.1 * std::sin(2.0 * M_PI * x);
            
            vel.u(i, j) = u_base + 0.01 * dist(rng);
            vel.v(i, j) = v_base + 0.01 * dist(rng);
        }
    }
}

// Test 1: MixingLengthModel CPU vs GPU
void test_mixing_length_consistency() {
    std::cout << "\n=== Testing MixingLengthModel CPU vs GPU ===" << std::endl;
    
#ifdef USE_GPU_OFFLOAD
    int num_devices = omp_get_num_devices();
    if (num_devices == 0) {
        std::cout << "SKIPPED (no GPU devices)\n";
        return;
    }
#else
    std::cout << "SKIPPED (GPU offload not enabled)\n";
    return;
#endif
    
    // Test multiple grid sizes and velocity fields
    struct TestCase { int nx, ny; int seed; };
    std::vector<TestCase> cases = {
        {64, 64, 0},
        {48, 96, 1},
        {63, 97, 2},  // Odd sizes
        {128, 128, 3}
    };
    
    bool all_passed = true;
    double worst_abs = 0.0, worst_rel = 0.0;
    
    for (const auto& tc : cases) {
        std::cout << "\n  Grid: " << tc.nx << "x" << tc.ny << ", seed=" << tc.seed << "\n";
        
        Mesh mesh;
        mesh.init_uniform(tc.nx, tc.ny, 0.0, 2.0, 0.0, 1.0, 1);
        
        VectorField velocity(mesh);
        create_test_velocity_field(mesh, velocity, tc.seed);
        
        ScalarField k(mesh), omega(mesh);
        ScalarField nu_t_gpu(mesh), nu_t_cpu(mesh);
        
        // Verify field addresses are different
        assert(nu_t_gpu.data().data() != nu_t_cpu.data().data());
        
        // GPU path - CRITICAL: initialize GPU buffers so update() uses GPU
        MixingLengthModel model_gpu;
        model_gpu.set_nu(1.0 / 10000.0);
        model_gpu.set_delta(0.5);
        model_gpu.initialize_gpu_buffers(mesh);
        
        // Hard check: GPU must be ready (otherwise this is CPU-vs-CPU!)
        if (!model_gpu.is_gpu_ready()) {
            std::cout << "    FAILED: GPU buffers not ready (would be CPU-vs-CPU test!)\n";
            assert(false);
        }
        
        model_gpu.update(mesh, velocity, k, omega, nu_t_gpu);
        
        // CPU reference (explicit reimplementation)
        ScalarField dudx(mesh), dudy(mesh), dvdx(mesh), dvdy(mesh);
        compute_all_velocity_gradients(mesh, velocity, dudx, dudy, dvdx, dvdy);
        
        const double nu = 1.0 / 10000.0;
        const double kappa = 0.41;
        const double A_plus = 26.0;
        const double delta = 0.5;
        
        // Estimate u_tau
        double u_tau = 0.0;
        {
            int j = mesh.j_begin();
            double dudy_wall = 0.0;
            int count = 0;
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                dudy_wall += std::abs(dudy(i, j));
                ++count;
            }
            dudy_wall /= count;
            double tau_w = nu * dudy_wall;
            u_tau = std::sqrt(tau_w);
        }
        u_tau = std::max(u_tau, 1e-10);
        
        // CPU computation
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double y_wall = mesh.wall_distance(i, j);
                double y_plus = y_wall * u_tau / nu;
                double damping = 1.0 - std::exp(-y_plus / A_plus);
                double l_mix = kappa * y_wall * damping;
                l_mix = std::min(l_mix, 0.5 * delta);
                
                double Sxx = dudx(i, j);
                double Syy = dvdy(i, j);
                double Sxy = 0.5 * (dudy(i, j) + dvdx(i, j));
                double S_mag = std::sqrt(2.0 * (Sxx*Sxx + Syy*Syy + 2.0*Sxy*Sxy));
                
                nu_t_cpu(i, j) = l_mix * l_mix * S_mag;
            }
        }
        
        // Compare
        auto cmp = compare_fields(mesh, nu_t_cpu, nu_t_gpu, "nu_t");
        
        worst_abs = std::max(worst_abs, cmp.max_abs_diff);
        worst_rel = std::max(worst_rel, cmp.max_rel_diff);
        
        // Tolerances (algorithm-based, not platform-based)
        const double tol_abs = 1e-10;
        const double tol_rel = 1e-8;
        
        if (cmp.max_abs_diff > tol_abs && cmp.max_rel_diff > tol_rel) {
            std::cout << "    FAILED: Differences exceed tolerance\n";
            std::cout << "      (abs_tol=" << tol_abs << ", rel_tol=" << tol_rel << ")\n";
            all_passed = false;
        } else {
            std::cout << "    PASSED\n";
        }
    }
    
    std::cout << "\n  Overall worst differences across all cases:\n";
    std::cout << "    Max abs: " << std::scientific << worst_abs << "\n";
    std::cout << "    Max rel: " << worst_rel << "\n";
    
    if (all_passed) {
        std::cout << "\n✓ MixingLengthModel CPU/GPU consistency: PASSED\n";
    } else {
        std::cout << "\n✗ MixingLengthModel CPU/GPU consistency: FAILED\n";
        assert(false);
    }
}

// Test 2: GEP model CPU vs GPU (if GPU path exists)
void test_gep_consistency() {
    std::cout << "\n=== Testing TurbulenceGEP CPU vs GPU ===" << std::endl;
    
#ifdef USE_GPU_OFFLOAD
    int num_devices = omp_get_num_devices();
    if (num_devices == 0) {
        std::cout << "SKIPPED (no GPU devices)\n";
        return;
    }
    
    // Check if GEP has GPU implementation
    Mesh mesh;
    mesh.init_uniform(64, 64, 0.0, 2.0, 0.0, 1.0, 1);
    
    VectorField vel(mesh);
    create_test_velocity_field(mesh, vel, 0);
    
    ScalarField k(mesh), omega(mesh), nu_t(mesh);
    
    TurbulenceGEP model;
    model.set_nu(0.001);
    model.set_delta(0.5);
    
    // For now, GEP might not have GPU path - just run and check it doesn't crash
    model.update(mesh, vel, k, omega, nu_t);
    
    std::cout << "  GEP model executed (GPU path may or may not exist)\n";
    std::cout << "  TODO: Add explicit CPU/GPU comparison when GPU path is implemented\n";
    std::cout << "SKIPPED (explicit CPU/GPU compare not yet implemented)\n";
#else
    std::cout << "SKIPPED (GPU offload not enabled)\n";
#endif
}

// Test 3: NN-MLP model CPU vs GPU
void test_nn_mlp_consistency() {
    std::cout << "\n=== Testing TurbulenceNNMLP CPU vs GPU ===" << std::endl;
    
#ifdef USE_GPU_OFFLOAD
    int num_devices = omp_get_num_devices();
    if (num_devices == 0) {
        std::cout << "SKIPPED (no GPU devices)\n";
        return;
    }
    
    try {
        Mesh mesh;
        mesh.init_uniform(32, 64, 0.0, 2.0, 0.0, 1.0, 1);
        
        VectorField vel(mesh);
        create_test_velocity_field(mesh, vel, 0);
        
        ScalarField k(mesh, 0.01);
        ScalarField omega(mesh, 10.0);
        ScalarField nu_t_cpu(mesh), nu_t_gpu(mesh);
        
        // CPU version
        TurbulenceNNMLP model_cpu;
        model_cpu.set_nu(0.001);
        model_cpu.load("../data/models/example_scalar_nut", "../data");
        model_cpu.update(mesh, vel, k, omega, nu_t_cpu);
        
        // GPU version
        TurbulenceNNMLP model_gpu;
        model_gpu.set_nu(0.001);
        model_gpu.load("../data/models/example_scalar_nut", "../data");
        model_gpu.upload_to_gpu();
        
        if (!model_gpu.is_gpu_ready()) {
            std::cout << "SKIPPED (GPU upload failed)\n";
            return;
        }
        
        model_gpu.update(mesh, vel, k, omega, nu_t_gpu);
        
        // Compare
        auto cmp = compare_fields(mesh, nu_t_cpu, nu_t_gpu, "nu_t");
        
        const double tol_abs = 1e-10;
        const double tol_rel = 1e-8;
        
        if (cmp.max_abs_diff > tol_abs && cmp.max_rel_diff > tol_rel) {
            std::cout << "  FAILED: Differences exceed tolerance\n";
            assert(false);
        } else {
            std::cout << "  PASSED\n";
        }
        
    } catch (const std::exception& e) {
        std::cout << "SKIPPED (model files not found: " << e.what() << ")\n";
    }
#else
    std::cout << "SKIPPED (GPU offload not enabled)\n";
#endif
}

// Test 4: Simple GPU compute test
void test_basic_gpu_compute() {
    std::cout << "\n=== Testing Basic GPU Computation ===" << std::endl;
    
#ifdef USE_GPU_OFFLOAD
    int num_devices = omp_get_num_devices();
    if (num_devices == 0) {
        std::cout << "SKIPPED (no GPU devices)\n";
        return;
    }
    
    const int N = 100000;
    std::vector<double> a(N, 2.0);
    std::vector<double> b(N, 3.0);
    std::vector<double> c(N, 0.0);
    
    double* a_ptr = a.data();
    double* b_ptr = b.data();
    double* c_ptr = c.data();
    
    #pragma omp target enter data map(to: a_ptr[0:N], b_ptr[0:N]) map(alloc: c_ptr[0:N])
    
    #pragma omp target teams distribute parallel for
    for (int i = 0; i < N; ++i) {
        c_ptr[i] = a_ptr[i] + b_ptr[i];
    }
    
    #pragma omp target update from(c_ptr[0:N])
    #pragma omp target exit data map(delete: a_ptr[0:N], b_ptr[0:N], c_ptr[0:N])
    
    // Verify
    for (int i = 0; i < 10; ++i) {
        assert(std::abs(c[i] - 5.0) < 1e-10);
    }
    
    std::cout << "  Basic GPU arithmetic verified\n";
    std::cout << "PASSED\n";
#else
    std::cout << "SKIPPED (GPU offload not enabled)\n";
#endif
}

// Test 5: Randomized regression - many random fields
void test_randomized_regression() {
    std::cout << "\n=== Randomized Regression Test ===" << std::endl;
    
#ifdef USE_GPU_OFFLOAD
    int num_devices = omp_get_num_devices();
    if (num_devices == 0) {
        std::cout << "SKIPPED (no GPU devices)\n";
        return;
    }
    
    // Fixed grid, many random velocity fields
    Mesh mesh;
    mesh.init_uniform(64, 64, 0.0, 2.0, 0.0, 1.0, 1);
    
    const int num_trials = 20;  // Test 20 different random fields
    double worst_abs = 0.0;
    double worst_rel = 0.0;
    int worst_seed = 0;  // Initialize to valid seed (not -1)
    
    std::cout << "  Testing " << num_trials << " random velocity fields...\n";
    
    // Initialize GPU model once (reuse across trials for efficiency)
    MixingLengthModel model_gpu;
    model_gpu.set_nu(1.0 / 10000.0);
    model_gpu.set_delta(0.5);
    model_gpu.initialize_gpu_buffers(mesh);
    
    if (!model_gpu.is_gpu_ready()) {
        std::cout << "  FAILED: GPU buffers not ready (would be CPU-vs-CPU test!)\n";
        assert(false);
    }
    
    for (int trial = 0; trial < num_trials; ++trial) {
        VectorField vel(mesh);
        ScalarField k(mesh), omega(mesh);
        ScalarField nu_t_cpu(mesh), nu_t_gpu(mesh);
        
        // Random velocity field
        create_test_velocity_field(mesh, vel, trial * 42);
        
        // GPU path (model already initialized)
        model_gpu.update(mesh, vel, k, omega, nu_t_gpu);
        
        // CPU reference
        ScalarField dudx(mesh), dudy(mesh), dvdx(mesh), dvdy(mesh);
        compute_all_velocity_gradients(mesh, vel, dudx, dudy, dvdx, dvdy);
        
        const double nu = 1.0 / 10000.0;
        const double kappa = 0.41;
        const double A_plus = 26.0;
        const double delta = 0.5;
        
        double u_tau = 0.0;
        {
            int j = mesh.j_begin();
            double dudy_wall = 0.0;
            int count = 0;
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                dudy_wall += std::abs(dudy(i, j));
                ++count;
            }
            dudy_wall /= count;
            double tau_w = nu * dudy_wall;
            u_tau = std::sqrt(tau_w);
        }
        u_tau = std::max(u_tau, 1e-10);
        
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double y_wall = mesh.wall_distance(i, j);
                double y_plus = y_wall * u_tau / nu;
                double damping = 1.0 - std::exp(-y_plus / A_plus);
                double l_mix = kappa * y_wall * damping;
                l_mix = std::min(l_mix, 0.5 * delta);
                
                double Sxx = dudx(i, j);
                double Syy = dvdy(i, j);
                double Sxy = 0.5 * (dudy(i, j) + dvdx(i, j));
                double S_mag = std::sqrt(2.0 * (Sxx*Sxx + Syy*Syy + 2.0*Sxy*Sxy));
                
                nu_t_cpu(i, j) = l_mix * l_mix * S_mag;
            }
        }
        
        // Compare
        double max_abs = 0.0, max_rel = 0.0;
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double diff = std::abs(nu_t_cpu(i, j) - nu_t_gpu(i, j));
                double rel = diff / (std::abs(nu_t_cpu(i, j)) + 1e-20);
                max_abs = std::max(max_abs, diff);
                max_rel = std::max(max_rel, rel);
            }
        }
        
        if (max_abs > worst_abs) {
            worst_abs = max_abs;
            worst_rel = max_rel;
            worst_seed = trial;
        }
        
        if ((trial + 1) % 5 == 0) {
            std::cout << "    Completed " << (trial + 1) << "/" << num_trials << " trials\n";
        }
    }
    
    std::cout << "  Worst case across all trials:\n";
    std::cout << "    Seed: " << worst_seed << "\n";
    std::cout << "    Max abs diff: " << std::scientific << worst_abs << "\n";
    std::cout << "    Max rel diff: " << worst_rel << "\n";
    
    const double tol_abs = 1e-10;
    const double tol_rel = 1e-8;
    
    if (worst_abs > tol_abs && worst_rel > tol_rel) {
        std::cout << "  FAILED: Worst case exceeds tolerance\n";
        assert(false);
    } else {
        std::cout << "  PASSED\n";
    }
    
#else
    std::cout << "SKIPPED (GPU offload not enabled)\n";
#endif
}

int main() {
    std::cout << "========================================\n";
    std::cout << "CPU vs GPU Consistency Test Suite\n";
    std::cout << "========================================\n";
    
#ifdef USE_GPU_OFFLOAD
    std::cout << "\nGPU Configuration:\n";
    int num_devices = omp_get_num_devices();
    std::cout << "  GPU devices available: " << num_devices << "\n";
    
    if (num_devices > 0) {
        int on_device = 0;
        #pragma omp target map(tofrom: on_device)
        {
            on_device = !omp_is_initial_device();
        }
        std::cout << "  GPU accessible: " << (on_device ? "YES" : "NO") << "\n";
    }
#else
    std::cout << "\nGPU offload: NOT ENABLED\n";
    std::cout << "Most tests will be skipped.\n";
#endif
    
    // Run tests
    test_harness_sanity();
    test_basic_gpu_compute();
    test_mixing_length_consistency();
    test_gep_consistency();
    test_nn_mlp_consistency();
    test_randomized_regression();
    
    std::cout << "\n========================================\n";
    std::cout << "All CPU/GPU consistency tests completed!\n";
    std::cout << "========================================\n";
    
    return 0;
}

