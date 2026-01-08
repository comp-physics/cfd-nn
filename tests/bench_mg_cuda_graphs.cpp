/// @file bench_mg_cuda_graphs.cpp
/// @brief Benchmark MG solver with/without CUDA Graphs on large 3D grids
/// Uses solve_device() with persistent GPU data for clean profiling

#include <iostream>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <vector>
#include "mesh.hpp"
#include "fields.hpp"
#include "poisson_solver_multigrid.hpp"

using namespace nncfd;
using namespace std::chrono;

void benchmark_grid(int N, int trials, int vcycles, bool use_fixed_cycles) {
    const double L = 2 * M_PI;

    Mesh mesh;
    mesh.init_uniform(N, N, N, 0.0, L, 0.0, L, 0.0, L);

    // Total size with ghost cells
    const size_t total_size = static_cast<size_t>(N + 2) * (N + 2) * (N + 2);

    // Allocate host arrays (will be mapped to device)
    std::vector<double> rhs_host(total_size, 0.0);
    std::vector<double> p_host(total_size, 0.0);

    // Initialize RHS with sin pattern (known solution)
    const int Ng = 1;
    const int stride = N + 2;
    const int plane_stride = stride * (N + 2);
    for (int k = Ng; k < N + Ng; ++k) {
        double z = (k - Ng + 0.5) * L / N;
        for (int j = Ng; j < N + Ng; ++j) {
            double y = (j - Ng + 0.5) * L / N;
            for (int i = Ng; i < N + Ng; ++i) {
                double x = (i - Ng + 0.5) * L / N;
                int idx = k * plane_stride + j * stride + i;
                rhs_host[idx] = -3.0 * std::sin(x) * std::sin(y) * std::sin(z);
            }
        }
    }

    // Get raw pointers for OpenMP target mapping
    double* rhs_ptr = rhs_host.data();
    double* p_ptr = p_host.data();

    // Map data to device ONCE (persistent for all solves)
    #pragma omp target enter data map(to: rhs_ptr[0:total_size], p_ptr[0:total_size])

    MultigridPoissonSolver mg(mesh);
    mg.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
              PoissonBC::Periodic, PoissonBC::Periodic,
              PoissonBC::Periodic, PoissonBC::Periodic);

    PoissonConfig cfg;
    if (use_fixed_cycles) {
        cfg.fixed_cycles = vcycles;  // Fixed mode: no convergence checks
    } else {
        cfg.tol = 1e-10;  // Convergence mode
        cfg.max_iter = vcycles;
        cfg.check_interval = 1;
    }

    // Warmup (2 solves) - zero initial guess on device
    for (int w = 0; w < 2; ++w) {
        // Zero p on device
        #pragma omp target teams distribute parallel for map(present, alloc: p_ptr[0:total_size])
        for (size_t idx = 0; idx < total_size; ++idx) {
            p_ptr[idx] = 0.0;
        }
        mg.solve_device(rhs_ptr, p_ptr, cfg);
    }

    // Benchmark
    std::vector<double> times;
    times.reserve(trials);

    for (int t = 0; t < trials; ++t) {
        // Zero p on device
        #pragma omp target teams distribute parallel for map(present, alloc: p_ptr[0:total_size])
        for (size_t idx = 0; idx < total_size; ++idx) {
            p_ptr[idx] = 0.0;
        }

        auto start = high_resolution_clock::now();
        int iters = mg.solve_device(rhs_ptr, p_ptr, cfg);
        auto end = high_resolution_clock::now();
        (void)iters;  // Suppress unused warning
        double ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        times.push_back(ms);
    }

    // Unmap data from device
    #pragma omp target exit data map(delete: rhs_ptr[0:total_size], p_ptr[0:total_size])

    // Compute stats
    double sum = 0, min_t = times[0], max_t = times[0];
    for (double t : times) {
        sum += t;
        min_t = std::min(min_t, t);
        max_t = std::max(max_t, t);
    }
    double avg = sum / trials;

    // Grid info
    double cells_M = (N * N * N) / 1e6;

    std::cout << std::setw(4) << N << "³  "
              << std::fixed << std::setprecision(2)
              << std::setw(6) << cells_M << "M cells  "
              << std::setw(8) << avg << " ms avg  "
              << std::setw(8) << min_t << " ms min  "
              << std::setw(8) << max_t << " ms max"
              << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "=== MG Solver Benchmark (solve_device path) ===" << std::endl;

    // Check if CUDA Graphs are enabled
    const char* env = std::getenv("MG_USE_CUDA_GRAPHS");
    bool cuda_graphs = (env && (std::string(env) == "1"));
    std::cout << "CUDA Graphs: " << (cuda_graphs ? "ENABLED" : "DISABLED") << std::endl;

    // Check for fixed_cycles mode
    const char* fixed_env = std::getenv("MG_FIXED_CYCLES");
    bool use_fixed = (fixed_env && std::string(fixed_env) == "1");
    std::cout << "Mode: " << (use_fixed ? "FIXED_CYCLES (no convergence checks)" : "CONVERGENCE (with tolerance checks)") << std::endl;
    std::cout << "Path: solve_device() with persistent GPU data (D-to-D only)" << std::endl;
    std::cout << std::endl;

    const int trials = 10;
    const int vcycles = 8;  // 8 V-cycles is typical for projection

    std::cout << "Trials: " << trials << ", V-cycles: " << vcycles << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    // Test various grid sizes
    benchmark_grid(64, trials, vcycles, use_fixed);
    benchmark_grid(96, trials, vcycles, use_fixed);
    benchmark_grid(128, trials, vcycles, use_fixed);
    benchmark_grid(160, trials, vcycles, use_fixed);
    benchmark_grid(192, trials, vcycles, use_fixed);

    std::cout << std::string(80, '-') << std::endl;

    return 0;
}
