// Quick MG benchmark - channel BCs force MG solver
#include "solver.hpp"
#include "config.hpp"
#include "mesh.hpp"
#include "gpu_utils.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace nncfd;

int main(int argc, char** argv) {
    int N = (argc > 1) ? std::atoi(argv[1]) : 64;
    int nsteps = (argc > 2) ? std::atoi(argv[2]) : 50;

    std::cout << "=========================================\n";
    std::cout << "  3D GPU MG Benchmark (Channel BCs)\n";
    std::cout << "=========================================\n";
    std::cout << "Grid: " << N << "x" << N << "x" << N << "\n";
    std::cout << "Steps: " << nsteps << "\n";
#ifdef USE_GPU_OFFLOAD
    std::cout << "Build: GPU\n";
#else
    std::cout << "Build: CPU\n";
#endif
    std::cout << "\n";

    Mesh mesh;
    mesh.init_uniform(N, N, N, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    Config config;
    config.nu = 1e-3;
    config.dt = 0.001;
    config.adaptive_dt = false;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    // Channel BCs (wall y, periodic x,z) - forces MG solver
    VelocityBC bc;
    bc.x_lo = bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = bc.y_hi = VelocityBC::NoSlip;
    bc.z_lo = bc.z_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);
    solver.set_body_force(1.0, 0.0, 0.0);

    // Initialize with small perturbation
    auto& vel = solver.velocity();
    for (int k = 0; k <= mesh.Nz; ++k)
        for (int j = 0; j <= mesh.Ny; ++j)
            for (int i = 0; i <= mesh.Nx + 1; ++i)
                vel.u(i, j, k) = 0.01 * std::sin(j * 3.14159 / mesh.Ny);
    
    solver.sync_to_gpu();

    std::cout << "Warming up (3 steps)...\n";
    for (int i = 0; i < 3; ++i) solver.step();

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
    std::cout << "  Results (MG Solver)\n";
    std::cout << "=========================================\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Time per step:  " << ms_per_step << " ms\n";
    std::cout << "Throughput:     " << mcups << " Mcells/s\n";
    std::cout << "=========================================\n";
    std::cout << "\nBENCH_MG_JSON: {\"grid\":" << N << ",\"steps\":" << nsteps
              << ",\"ms_per_step\":" << ms_per_step << ",\"mcups\":" << mcups << "}\n";
    return 0;
}
