#include <iostream>
#include <chrono>
#include <cmath>
#include <iomanip>
#include "mesh.hpp"
#include "fields.hpp"
#include "poisson_solver_multigrid.hpp"

using namespace nncfd;
using namespace std::chrono;

int main() {
    const int N = 256;
    const double L = 2 * M_PI;
    
    Mesh mesh;
    mesh.init_uniform(N, N, N, 0.0, L, 0.0, L, 0.0, L);
    
    ScalarField rhs(mesh);
    ScalarField p(mesh);
    
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
    cfg.tol = 1e-10;
    cfg.max_iter = 20;
    cfg.check_interval = 1;
    
    // Warmup
    p.fill(0);
    mg.solve(rhs, p, cfg);
    
    // Benchmark
    const int trials = 5;
    double total = 0;
    for (int t = 0; t < trials; ++t) {
        p.fill(0);
        auto start = high_resolution_clock::now();
        mg.solve(rhs, p, cfg);
        auto end = high_resolution_clock::now();
        total += duration_cast<microseconds>(end - start).count() / 1000.0;
    }
    
    double cells_M = (N * N * N) / 1e6;
    std::cout << " 256³   " << std::fixed << std::setprecision(1) << cells_M 
              << "M cells  " << std::setprecision(2) << (total/trials) << " ms avg" << std::endl;
    
    return 0;
}
