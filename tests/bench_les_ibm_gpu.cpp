/// @file bench_les_ibm_gpu.cpp
/// @brief Benchmark: LES + IBM GPU performance on 3D problem
///
/// Runs a 3D channel with LES (Smagorinsky) and IBM (cylinder) for a handful
/// of steps, reporting per-step timing with GPU kernel breakdown.
///
/// Usage: ./bench_les_ibm_gpu [Nx] [Ny] [Nz] [nsteps]
///   Defaults: 128 64 128 20

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "config.hpp"
#include "timing.hpp"
#include "turbulence_les.hpp"
#include "ibm_geometry.hpp"
#include "ibm_forcing.hpp"
#include "decomposition.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <cstdlib>

using namespace nncfd;

int main(int argc, char** argv) {
    int Nx = 128, Ny = 64, Nz = 128;
    int nsteps = 20;
    int warmup = 3;

    if (argc > 1) Nx = std::atoi(argv[1]);
    if (argc > 2) Ny = std::atoi(argv[2]);
    if (argc > 3) Nz = std::atoi(argv[3]);
    if (argc > 4) nsteps = std::atoi(argv[4]);

    double Lx = 2.0 * M_PI;
    double Ly = 2.0;
    double Lz = M_PI;

    std::cout << "================================================================\n";
    std::cout << "  LES + IBM GPU Benchmark (3D)\n";
    std::cout << "================================================================\n";
#ifdef USE_GPU_OFFLOAD
    std::cout << "Build: GPU (USE_GPU_OFFLOAD=ON)\n";
#else
    std::cout << "Build: CPU (USE_GPU_OFFLOAD=OFF)\n";
#endif
    std::cout << "Grid:  " << Nx << " x " << Ny << " x " << Nz
              << " = " << (Nx * Ny * Nz) << " cells\n";
    std::cout << "Steps: " << warmup << " warmup + " << nsteps << " timed\n";
    std::cout << "Memory: ~" << std::fixed << std::setprecision(1)
              << (Nx * Ny * Nz * 8.0 * 15 / 1e9) << " GB (estimated)\n\n";

    // ========================================================================
    // Test 1: LES channel (Smagorinsky, no IBM)
    // ========================================================================
    {
        std::cout << "--- Test 1: Smagorinsky LES channel (no IBM) ---\n";
        TimingStats::instance().reset();

        Mesh mesh;
        mesh.init_uniform(Nx, Ny, Nz, 0.0, Lx, -Ly/2, Ly/2, 0.0, Lz);

        Config config;
        config.Nx = Nx; config.Ny = Ny; config.Nz = Nz;
        config.nu = 0.001;
        config.dt = 0.001;
        config.adaptive_dt = false;
        config.turb_model = TurbulenceModelType::Smagorinsky;
        config.verbose = false;
        config.poisson_solver = PoissonSolverType::MG;

        Decomposition decomp(Nz);
        RANSSolver solver(mesh, config);
        solver.set_decomposition(&decomp);

        VelocityBC bc;
        bc.x_lo = VelocityBC::Periodic; bc.x_hi = VelocityBC::Periodic;
        bc.y_lo = VelocityBC::NoSlip;   bc.y_hi = VelocityBC::NoSlip;
        bc.z_lo = VelocityBC::Periodic; bc.z_hi = VelocityBC::Periodic;
        solver.set_velocity_bc(bc);
        solver.set_body_force(0.001, 0.0, 0.0);

        auto turb = create_turbulence_model(TurbulenceModelType::Smagorinsky);
        turb->set_nu(config.nu);
        solver.set_turbulence_model(std::move(turb));
        solver.initialize_uniform(1.0, 0.0);

        // Warmup
        for (int s = 0; s < warmup; ++s) solver.step();

        // Timed
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int s = 0; s < nsteps; ++s) solver.step();
        auto t1 = std::chrono::high_resolution_clock::now();

        double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double per_step = total_ms / nsteps;

        std::cout << "  Total: " << std::fixed << std::setprecision(1)
                  << total_ms << " ms (" << nsteps << " steps)\n";
        std::cout << "  Per step: " << std::setprecision(2) << per_step << " ms\n";
        std::cout << "  Throughput: " << std::setprecision(1)
                  << (Nx * Ny * Nz / per_step / 1e3) << " Mcells/s\n\n";

        TimingStats::instance().print_summary();
        std::cout << "\n";
    }

    // ========================================================================
    // Test 2: LES channel + IBM cylinder
    // ========================================================================
    {
        std::cout << "--- Test 2: Smagorinsky LES + IBM cylinder ---\n";
        TimingStats::instance().reset();

        Mesh mesh;
        mesh.init_uniform(Nx, Ny, Nz, 0.0, Lx, -Ly/2, Ly/2, 0.0, Lz);

        Config config;
        config.Nx = Nx; config.Ny = Ny; config.Nz = Nz;
        config.nu = 0.001;
        config.dt = 0.001;
        config.adaptive_dt = false;
        config.turb_model = TurbulenceModelType::Smagorinsky;
        config.verbose = false;
        config.poisson_solver = PoissonSolverType::MG;

        Decomposition decomp(Nz);
        RANSSolver solver(mesh, config);
        solver.set_decomposition(&decomp);

        VelocityBC bc;
        bc.x_lo = VelocityBC::Periodic; bc.x_hi = VelocityBC::Periodic;
        bc.y_lo = VelocityBC::Periodic; bc.y_hi = VelocityBC::Periodic;
        bc.z_lo = VelocityBC::Periodic; bc.z_hi = VelocityBC::Periodic;
        solver.set_velocity_bc(bc);
        solver.set_body_force(0.001, 0.0, 0.0);

        auto turb = create_turbulence_model(TurbulenceModelType::Smagorinsky);
        turb->set_nu(config.nu);
        solver.set_turbulence_model(std::move(turb));
        solver.initialize_uniform(1.0, 0.0);

        // IBM cylinder at domain center
        auto body = std::make_shared<CylinderBody>(Lx / 2, 0.0, 0.3);
        IBMForcing ibm(mesh, body);
        solver.set_ibm_forcing(&ibm);

        std::cout << "  IBM: " << ibm.num_forcing_cells() << " forcing, "
                  << ibm.num_solid_cells() << " solid cells\n";

        // Warmup
        for (int s = 0; s < warmup; ++s) solver.step();

        // Timed
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int s = 0; s < nsteps; ++s) solver.step();
        auto t1 = std::chrono::high_resolution_clock::now();

        double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double per_step = total_ms / nsteps;

        std::cout << "  Total: " << std::fixed << std::setprecision(1)
                  << total_ms << " ms (" << nsteps << " steps)\n";
        std::cout << "  Per step: " << std::setprecision(2) << per_step << " ms\n";
        std::cout << "  Throughput: " << std::setprecision(1)
                  << (Nx * Ny * Nz / per_step / 1e3) << " Mcells/s\n\n";

        TimingStats::instance().print_summary();
        std::cout << "\n";
    }

    // ========================================================================
    // Test 3: Laminar baseline (no turbulence model, no IBM)
    // ========================================================================
    {
        std::cout << "--- Test 3: Laminar baseline (no LES, no IBM) ---\n";
        TimingStats::instance().reset();

        Mesh mesh;
        mesh.init_uniform(Nx, Ny, Nz, 0.0, Lx, -Ly/2, Ly/2, 0.0, Lz);

        Config config;
        config.Nx = Nx; config.Ny = Ny; config.Nz = Nz;
        config.nu = 0.001;
        config.dt = 0.001;
        config.adaptive_dt = false;
        config.turb_model = TurbulenceModelType::None;
        config.verbose = false;
        config.poisson_solver = PoissonSolverType::MG;

        Decomposition decomp(Nz);
        RANSSolver solver(mesh, config);
        solver.set_decomposition(&decomp);

        VelocityBC bc;
        bc.x_lo = VelocityBC::Periodic; bc.x_hi = VelocityBC::Periodic;
        bc.y_lo = VelocityBC::NoSlip;   bc.y_hi = VelocityBC::NoSlip;
        bc.z_lo = VelocityBC::Periodic; bc.z_hi = VelocityBC::Periodic;
        solver.set_velocity_bc(bc);
        solver.set_body_force(0.001, 0.0, 0.0);
        solver.initialize_uniform(1.0, 0.0);

        // Warmup
        for (int s = 0; s < warmup; ++s) solver.step();

        // Timed
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int s = 0; s < nsteps; ++s) solver.step();
        auto t1 = std::chrono::high_resolution_clock::now();

        double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double per_step = total_ms / nsteps;

        std::cout << "  Total: " << std::fixed << std::setprecision(1)
                  << total_ms << " ms (" << nsteps << " steps)\n";
        std::cout << "  Per step: " << std::setprecision(2) << per_step << " ms\n";
        std::cout << "  Throughput: " << std::setprecision(1)
                  << (Nx * Ny * Nz / per_step / 1e3) << " Mcells/s\n\n";

        TimingStats::instance().print_summary();
        std::cout << "\n";
    }

    std::cout << "================================================================\n";
    std::cout << "  Benchmark complete\n";
    std::cout << "================================================================\n";

    return 0;
}
