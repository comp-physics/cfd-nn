/// @file bench_fft_stretched_timing.cpp
/// @brief Benchmark FFT vs MG Poisson solver timing on stretched-y grids.
/// Reports wall-clock time per step and speedup ratio.

#include "test_harness.hpp"
#include "test_utilities.hpp"
#include "solver.hpp"
#include "config.hpp"
#include "mesh.hpp"
#include "timing.hpp"
#include <cmath>
#include <chrono>
#include <iomanip>

using namespace nncfd;
using nncfd::test::BCPattern;
using nncfd::test::create_velocity_bc;

static void benchmark_solver(const char* label, PoissonSolverType solver_type,
                              int Nx, int Ny, int Nz, int nsteps, double beta) {
    Mesh mesh;
    mesh.init_stretched_y(Nx, Ny, Nz, 0.0, 2*M_PI, 0.0, 2.0, 0.0, 2*M_PI,
                          Mesh::tanh_stretching(beta));

    Config cfg;
    cfg.Nx = Nx; cfg.Ny = Ny; cfg.Nz = Nz;
    cfg.dt = 0.0001;
    cfg.max_steps = nsteps;
    cfg.nu = 0.01;
    cfg.dp_dx = -1.0;
    cfg.poisson_solver = solver_type;
    cfg.stretch_y = true;
    cfg.verbose = false;
    cfg.perf_mode = true;

    RANSSolver solver(mesh, cfg);
    solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel3D));
    solver.set_body_force(1.0, 0.0, 0.0);
    solver.initialize_uniform(1.0, 0.0);

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    // Warmup
    for (int i = 0; i < 3; ++i) solver.step();

    // Timed steps
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < nsteps; ++i) solver.step();
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    double ms_per_step = elapsed / nsteps * 1000.0;

    std::cout << "  " << label << ": " << std::fixed << std::setprecision(2)
              << ms_per_step << " ms/step (" << nsteps << " steps, "
              << Nx << "x" << Ny << "x" << Nz << ", beta=" << beta << ")\n";
}

int main() {
    std::cout << "================================================================\n";
    std::cout << "  FFT vs MG Benchmark (Stretched Y)\n";
    std::cout << "================================================================\n\n";

#ifdef USE_GPU_OFFLOAD
    std::cout << "Build: GPU\n\n";
#else
    std::cout << "Build: CPU\n\n";
#endif

    // Small grid: quick comparison
    std::cout << "--- 32x32x32, beta=2.0 ---\n";
    benchmark_solver("MG  ", PoissonSolverType::MG, 32, 32, 32, 20, 2.0);
    benchmark_solver("FFT ", PoissonSolverType::FFT, 32, 32, 32, 20, 2.0);

    // Medium grid: more representative
    std::cout << "\n--- 64x64x64, beta=2.0 ---\n";
    benchmark_solver("MG  ", PoissonSolverType::MG, 64, 64, 64, 10, 2.0);
    benchmark_solver("FFT ", PoissonSolverType::FFT, 64, 64, 64, 10, 2.0);

    // DNS-scale: 128x64x128 (like production)
    std::cout << "\n--- 128x64x128, beta=2.0 ---\n";
    benchmark_solver("MG  ", PoissonSolverType::MG, 128, 64, 128, 5, 2.0);
    benchmark_solver("FFT ", PoissonSolverType::FFT, 128, 64, 128, 5, 2.0);

    std::cout << "\nDone.\n";
    return 0;
}
