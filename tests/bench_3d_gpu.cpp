/// @file bench_3d_gpu.cpp
/// @brief 3D GPU performance benchmark - TGV with no I/O

#include "solver.hpp"
#include "config.hpp"
#include "mesh.hpp"
#include "gpu_utils.hpp"
#include "test_utilities.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>

using namespace nncfd;
using nncfd::test::BCPattern;
using nncfd::test::create_velocity_bc;

void init_tgv_3d(RANSSolver& solver, const Mesh& mesh) {
    auto& vel = solver.velocity();
    // Use mesh bounds to properly account for ghost cells
    // u at x-faces: use xf for x, cell centers for y/z
    for (int k = mesh.k_begin(); k <= mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i <= mesh.i_end() + 1; ++i) {
                double x = mesh.xf[i];
                double y = mesh.y(j);
                double z = mesh.z(k);
                vel.u(i, j, k) = std::sin(x) * std::cos(y) * std::cos(z);
            }
        }
    }
    // v at y-faces: use cell centers for x/z, yf for y
    for (int k = mesh.k_begin(); k <= mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j <= mesh.j_end() + 1; ++j) {
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                double x = mesh.x(i);
                double y = mesh.yf[j];
                double z = mesh.z(k);
                vel.v(i, j, k) = -std::cos(x) * std::sin(y) * std::cos(z);
            }
        }
    }
    // w at z-faces: use cell centers for x/y, zf for z
    for (int k = mesh.k_begin(); k <= mesh.k_end() + 1; ++k) {
        for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                vel.w(i, j, k) = 0.0;
            }
        }
    }
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? std::atoi(argv[1]) : 64;
    int nsteps = (argc > 2) ? std::atoi(argv[2]) : 100;
    std::string ti_name = (argc > 3) ? argv[3] : "euler";

    TimeIntegrator ti = TimeIntegrator::Euler;
    if (ti_name == "rk2") ti = TimeIntegrator::RK2;
    else if (ti_name == "rk3") ti = TimeIntegrator::RK3;

    std::cout << "=========================================\n";
    std::cout << "  3D GPU Performance Benchmark\n";
    std::cout << "=========================================\n";
    std::cout << "Grid: " << N << "x" << N << "x" << N << "\n";
    std::cout << "Steps: " << nsteps << "\n";
    std::cout << "Integrator: " << ti_name << "\n";
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
    config.time_integrator = ti;

    RANSSolver solver(mesh, config);
    solver.set_velocity_bc(create_velocity_bc(BCPattern::FullyPeriodic));

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
