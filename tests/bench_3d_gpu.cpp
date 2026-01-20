/// @file bench_3d_gpu.cpp
/// @brief 3D GPU performance benchmark - TGV with no I/O

#include "solver.hpp"
#include "config.hpp"
#include "mesh.hpp"
#include "gpu_utils.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>

using namespace nncfd;

void init_tgv_3d(RANSSolver& solver, const Mesh& mesh) {
    auto& vel = solver.velocity();
    for (int k = 0; k <= mesh.Nz; ++k) {
        for (int j = 0; j <= mesh.Ny; ++j) {
            for (int i = 0; i <= mesh.Nx + 1; ++i) {
                double x = (i - 0.5) * mesh.dx;
                double y = j * mesh.dy;
                double z = k * mesh.dz;
                vel.u(i, j, k) = std::sin(x) * std::cos(y) * std::cos(z);
            }
        }
    }
    for (int k = 0; k <= mesh.Nz; ++k) {
        for (int j = 0; j <= mesh.Ny + 1; ++j) {
            for (int i = 0; i <= mesh.Nx; ++i) {
                double x = i * mesh.dx;
                double y = (j - 0.5) * mesh.dy;
                double z = k * mesh.dz;
                vel.v(i, j, k) = -std::cos(x) * std::sin(y) * std::cos(z);
            }
        }
    }
    for (int k = 0; k <= mesh.Nz + 1; ++k) {
        for (int j = 0; j <= mesh.Ny; ++j) {
            for (int i = 0; i <= mesh.Nx; ++i) {
                vel.w(i, j, k) = 0.0;
            }
        }
    }
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? std::atoi(argv[1]) : 64;
    int nsteps = (argc > 2) ? std::atoi(argv[2]) : 100;

    std::cout << "=========================================\n";
    std::cout << "  3D GPU Performance Benchmark\n";
    std::cout << "=========================================\n";
    std::cout << "Grid: " << N << "x" << N << "x" << N << "\n";
    std::cout << "Steps: " << nsteps << "\n";
#ifdef USE_GPU_OFFLOAD
    std::cout << "Build: GPU\n";
    std::cout << "Devices: " << gpu::num_devices() << "\n";
#else
    std::cout << "Build: CPU\n";
#endif
    std::cout << "\n";

    const double L = 2.0 * M_PI;
    Mesh mesh;
    mesh.init_uniform(N, N, N, 0.0, L, 0.0, L, 0.0, L);

    Config config;
    config.nu = 1e-3;
    config.dt = 0.005;
    config.adaptive_dt = false;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = bc.y_hi = VelocityBC::Periodic;
    bc.z_lo = bc.z_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    init_tgv_3d(solver, mesh);
    solver.sync_to_gpu();

    std::cout << "Warming up (5 steps)...\n";
    for (int i = 0; i < 5; ++i) solver.step();

    std::cout << "Running " << nsteps << " timed steps...\n";
    auto t_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < nsteps; ++i) solver.step();
#ifdef USE_GPU_OFFLOAD
    gpu::sync();
#endif
    auto t_end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    double ms_per_step = elapsed_ms / nsteps;
    double cells = static_cast<double>(N) * N * N;
    double mcups = (cells * (1000.0 / ms_per_step)) / 1e6;

    std::cout << "\n=========================================\n";
    std::cout << "  Results\n";
    std::cout << "=========================================\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Time per step:  " << ms_per_step << " ms\n";
    std::cout << "Throughput:     " << mcups << " Mcells/s\n";
    std::cout << "=========================================\n";
    std::cout << "\nBENCH_JSON: {\"grid\":" << N << ",\"steps\":" << nsteps
              << ",\"ms_per_step\":" << ms_per_step << ",\"mcups\":" << mcups << "}\n";
    return 0;
}
