/// @file bench_mg_cuda_graphs.cpp
/// @brief Benchmark MG solver with/without CUDA Graphs on large 3D grids

#include <iostream>
#include <chrono>
#include <cmath>
#include <iomanip>
#include "mesh.hpp"
#include "fields.hpp"
#include "poisson_solver_multigrid.hpp"

using namespace nncfd;
using namespace std::chrono;

void benchmark_grid(int N, int trials, int vcycles) {
    const double L = 2 * M_PI;
    
    Mesh mesh;
    mesh.init_uniform(N, N, N, 0.0, L, 0.0, L, 0.0, L);
    
    ScalarField rhs(mesh);
    ScalarField p(mesh);
    
    // Initialize with sin pattern (known solution)
    for (int k = 1; k <= N; ++k) {
        double z = (k - 0.5) * L / N;
        for (int j = 1; j <= N; ++j) {
            double y = (j - 0.5) * L / N;
            for (int i = 1; i <= N; ++i) {
                double x = (i - 0.5) * L / N;
                rhs(i, j, k) = -3.0 * std::sin(x) * std::sin(y) * std::sin(z);
            }
        }
    }
    
    MultigridPoissonSolver mg(mesh);
    mg.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
              PoissonBC::Periodic, PoissonBC::Periodic,
              PoissonBC::Periodic, PoissonBC::Periodic);
    
    PoissonConfig cfg;
    cfg.tol = 1e-10;  // Tight tolerance to ensure consistent iterations
    cfg.max_iter = vcycles;
    cfg.check_interval = 1;
    
    // Warmup (2 solves)
    for (int w = 0; w < 2; ++w) {
        p.fill(0);
        mg.solve(rhs, p, cfg);
    }
    
    // Benchmark
    std::vector<double> times;
    times.reserve(trials);
    
    for (int t = 0; t < trials; ++t) {
        p.fill(0);
        auto start = high_resolution_clock::now();
        int iters = mg.solve(rhs, p, cfg);
        auto end = high_resolution_clock::now();
        double ms = duration_cast<microseconds>(end - start).count() / 1000.0;
        times.push_back(ms);
    }
    
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
              << std::setw(8) << max_t << " ms max  "
              << "res=" << std::scientific << std::setprecision(2) << mg.residual()
              << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "=== MG Solver Benchmark ===" << std::endl;
    
    // Check if CUDA Graphs are enabled
    const char* env = std::getenv("MG_USE_CUDA_GRAPHS");
    bool cuda_graphs = (env && (std::string(env) == "1"));
    std::cout << "CUDA Graphs: " << (cuda_graphs ? "ENABLED" : "DISABLED") << std::endl;
    std::cout << std::endl;
    
    const int trials = 10;
    const int vcycles = 20;
    
    std::cout << "Trials: " << trials << ", V-cycles: " << vcycles << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    // Test various grid sizes
    benchmark_grid(64, trials, vcycles);
    benchmark_grid(96, trials, vcycles);
    benchmark_grid(128, trials, vcycles);
    benchmark_grid(160, trials, vcycles);
    benchmark_grid(192, trials, vcycles);
    
    std::cout << std::string(80, '-') << std::endl;
    
    return 0;
}
