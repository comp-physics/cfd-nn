/// @file bench_fft_vs_mg.cpp
/// @brief Compare FFT vs MG Poisson solver performance on periodic grids
///
/// Uses full solver integration to compare real-world performance

#include <iostream>
#include <chrono>
#include <cmath>
#include <iomanip>
#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"

using namespace nncfd;
using namespace std::chrono;

/// Run Taylor-Green for a few steps with specified solver and return avg Poisson time
double benchmark_solver(int N, PoissonSolverType solver_type, int num_iter, bool fixed_cycles) {
    const double L = 2 * M_PI;
    const double Re = 100.0;
    const double V0 = 1.0;
    const double nu = V0 / Re;

    Mesh mesh;
    mesh.init_uniform(N, N, N, 0.0, L, 0.0, L, 0.0, L);

    Config config;
    config.Nx = config.Ny = config.Nz = N;
    config.nu = nu;
    config.dt = 0.01;
    config.poisson_solver = solver_type;
    config.verbose = false;
    if (fixed_cycles) {
        config.poisson_fixed_cycles = 8;
    }

    RANSSolver solver(mesh, config);

    // Periodic BCs
    VelocityBC bc;
    bc.x_lo = bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = bc.y_hi = VelocityBC::Periodic;
    bc.z_lo = bc.z_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    // Initialize Taylor-Green
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                double x = mesh.xf[i];
                double y = mesh.y(j);
                double z = mesh.z(k);
                solver.velocity().u(i, j, k) = V0 * std::sin(x) * std::cos(y) * std::cos(z);
            }
        }
    }
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double x = mesh.x(i);
                double y = mesh.yf[j];
                double z = mesh.z(k);
                solver.velocity().v(i, j, k) = -V0 * std::cos(x) * std::sin(y) * std::cos(z);
            }
        }
    }

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    // Warmup
    for (int w = 0; w < 2; ++w) {
        solver.step();
    }

    // Time the steps
    auto start = high_resolution_clock::now();
    for (int s = 0; s < num_iter; ++s) {
        solver.step();
    }
    auto end = high_resolution_clock::now();

    double total_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
    return total_ms / num_iter;
}

int main() {
    std::cout << "=== FFT vs MG Performance (Periodic 3D Taylor-Green) ===\n\n";
    std::cout << "Measuring full timestep cost (includes advection, diffusion, projection)\n";
    std::cout << "FFT: cuFFT (direct)  |  MG: fixed 8 V-cycles\n\n";

    const int num_iter = 10;

    std::cout << std::string(85, '-') << std::endl;
    std::cout << std::setw(6) << "Grid" << std::setw(10) << "Cells"
              << std::setw(15) << "FFT (ms)" << std::setw(15) << "MG (ms)"
              << std::setw(12) << "Ratio" << std::endl;
    std::cout << std::string(85, '-') << std::endl;

    for (int N : {64, 96, 128, 160, 192}) {
        double fft_time = benchmark_solver(N, PoissonSolverType::FFT, num_iter, false);
        double mg_time = benchmark_solver(N, PoissonSolverType::MG, num_iter, true);

        double cells_M = (N * N * N) / 1e6;
        double ratio = mg_time / fft_time;

        std::cout << std::setw(4) << N << "^3"
                  << std::fixed << std::setprecision(2)
                  << std::setw(8) << cells_M << "M"
                  << std::setw(15) << std::setprecision(2) << fft_time
                  << std::setw(15) << mg_time
                  << std::setw(12) << std::setprecision(2) << ratio << "x"
                  << std::endl;
    }

    std::cout << std::string(85, '-') << std::endl;
    std::cout << "\nNote: ratio > 1 means MG is slower than FFT for full timestep\n";
    std::cout << "For non-periodic BCs, only MG is applicable (FFT requires periodicity)\n";

    return 0;
}
