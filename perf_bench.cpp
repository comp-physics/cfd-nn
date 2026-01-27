/// @file perf_bench.cpp
/// @brief Simple performance benchmark for branch comparison
/// Usage: ./perf_bench [grid_size] [num_steps]

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>

using namespace nncfd;
using namespace std::chrono;

int main(int argc, char** argv) {
    // Parse arguments
    int N = 64;          // Default grid size
    int nsteps = 100;    // Default number of steps

    if (argc > 1) N = std::atoi(argv[1]);
    if (argc > 2) nsteps = std::atoi(argv[2]);

    std::cout << "=== Performance Benchmark ===" << std::endl;
    std::cout << "Grid: " << N << "³ | Steps: " << nsteps << std::endl;

    // Create 3D mesh (periodic in x,z, walls in y - channel flow)
    const double Lx = 2.0 * M_PI;
    const double Ly = 2.0;
    const double Lz = M_PI;

    Mesh mesh;
    mesh.init_uniform(N, N, N, 0.0, Lx, 0.0, Ly, 0.0, Lz);

    // Configure solver - minimal settings, no I/O
    Config config;
    config.nu = 1e-4;
    config.dt = 1e-4;
    config.adaptive_dt = false;
    config.max_steps = nsteps;
    config.verbose = false;
    config.output_freq = 0;
    config.write_fields = false;
    config.num_snapshots = 0;
    config.turb_model = TurbulenceModelType::None;
    config.poisson_solver = PoissonSolverType::MG;

    // Create solver
    RANSSolver solver(mesh, config);

    // Set BCs: periodic x,z, no-slip y (channel)
    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    bc.z_lo = VelocityBC::Periodic;
    bc.z_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    // Initialize with Taylor-Green-like vortex
    const double U0 = 1.0;
    for (int k = 1; k <= N; ++k) {
        double z = mesh.z(k);
        for (int j = 1; j <= N; ++j) {
            double y = mesh.y(j);
            for (int i = 1; i <= N + 1; ++i) {
                double x = mesh.xf[i];
                solver.velocity().u(i, j, k) = U0 * std::sin(x) * std::cos(y) * std::cos(z);
            }
        }
    }
    for (int k = 1; k <= N; ++k) {
        double z = mesh.z(k);
        for (int j = 1; j <= N + 1; ++j) {
            double y = mesh.yf[j];
            for (int i = 1; i <= N; ++i) {
                double x = mesh.x(i);
                solver.velocity().v(i, j, k) = -U0 * std::cos(x) * std::sin(y) * std::cos(z);
            }
        }
    }
    // w = 0

    // Warmup (1 step)
    solver.step();

    // Timed run
    auto start = high_resolution_clock::now();
    for (int i = 1; i < nsteps; ++i) {
        solver.step();
    }
    auto end = high_resolution_clock::now();

    double elapsed_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
    double ms_per_step = elapsed_ms / (nsteps - 1);
    double mcells_per_sec = (static_cast<double>(N) * N * N) / (ms_per_step * 1000.0);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Total time:  " << elapsed_ms << " ms" << std::endl;
    std::cout << "Per step:    " << std::setprecision(3) << ms_per_step << " ms" << std::endl;
    std::cout << "Throughput:  " << std::setprecision(2) << mcells_per_sec << " Mcells/s" << std::endl;

    return 0;
}
