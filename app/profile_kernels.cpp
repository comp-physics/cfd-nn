/// @file profile_kernels.cpp
/// @brief Comprehensive kernel-level profiling for NVTX analysis
///
/// This profiler exercises all GPU kernels to generate granular NVTX traces.
/// Run with nsys:
///   nsys profile -t nvtx,cuda --stats=true -o profile_report ./profile_kernels
///
/// Build with:
///   cmake -DUSE_GPU_OFFLOAD=ON -DGPU_PROFILE_KERNELS=ON ..

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "timing.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <memory>

#ifdef GPU_PROFILE_KERNELS
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define NVTX_MARK_PHASE(name) nvtxMarkA(name)
#else
#define NVTX_MARK_PHASE(name)
#endif
#else
#define NVTX_MARK_PHASE(name)
#endif

using namespace nncfd;

// Run N solver steps with a specific turbulence model
void profile_turbulence_model(const std::string& name, TurbulenceModelType type,
                               int nsteps, int Nx, int Ny) {
    std::cout << "\n=== Profiling: " << name << " (" << nsteps << " steps, "
              << Nx << "x" << Ny << " grid) ===\n";

    NVTX_MARK_PHASE(name.c_str());

    Mesh mesh;
    mesh.init_uniform(Nx, Ny, 0.0, 4.0, -1.0, 1.0);

    Config config;
    config.nu = 0.01;
    config.dp_dx = -0.001;
    config.dt = 0.001;
    config.adaptive_dt = false;  // Fixed dt for consistent profiling
    config.max_steps = nsteps;
    config.tol = 1e-8;
    config.turb_model = type;
    config.verbose = false;

    RANSSolver solver(mesh, config);
    solver.set_body_force(-config.dp_dx, 0.0);

    // Initialize with parabolic profile
    double H = 1.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        double y = mesh.y(j);
        double u = -config.dp_dx / (2.0 * config.nu) * (H * H - y * y);
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            solver.velocity().u(i, j) = 0.8 * u;  // 80% of analytical
        }
    }

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    // Run steps
    TimingStats::instance().reset();

    for (int step = 0; step < nsteps; ++step) {
        double residual = solver.step();
        if ((step + 1) % 10 == 0 || step == nsteps - 1) {
            std::cout << "  Step " << std::setw(4) << (step + 1)
                      << ", residual = " << std::scientific << std::setprecision(3)
                      << residual << std::endl;
        }
    }

    std::cout << "  Completed " << nsteps << " steps for " << name << std::endl;
    TimingStats::instance().print_summary();
}

// Profile Poisson solver specifically (many iterations)
void profile_poisson_solver(int nsteps, int Nx, int Ny) {
    std::cout << "\n=== Profiling: Poisson Solver Focus (" << nsteps << " steps) ===\n";

    NVTX_MARK_PHASE("poisson_focus");

    Mesh mesh;
    mesh.init_uniform(Nx, Ny, 0.0, 4.0, -1.0, 1.0);

    Config config;
    config.nu = 0.01;
    config.dp_dx = -0.001;
    config.dt = 0.001;
    config.adaptive_dt = false;
    config.max_steps = nsteps;
    config.poisson_max_vcycles = 50;  // More Poisson iterations for profiling
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);
    solver.set_body_force(-config.dp_dx, 0.0);
    solver.initialize_uniform(0.1, 0.0);

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    TimingStats::instance().reset();

    for (int step = 0; step < nsteps; ++step) {
        solver.step();
    }

    std::cout << "  Completed " << nsteps << " steps (Poisson focus)\n";
    TimingStats::instance().print_summary();
}

// Profile 3D solver
void profile_3d_solver(int nsteps, int Nx, int Ny, int Nz) {
    std::cout << "\n=== Profiling: 3D Solver (" << nsteps << " steps, "
              << Nx << "x" << Ny << "x" << Nz << " grid) ===\n";

    NVTX_MARK_PHASE("3d_solver");

    Mesh mesh;
    mesh.init_uniform(Nx, Ny, Nz, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0);

    Config config;
    config.nu = 0.01;
    config.dt = 0.001;
    config.adaptive_dt = false;
    config.max_steps = nsteps;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);
    solver.initialize_uniform(0.1, 0.0);  // Initialize u, v (w is set to 0)

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    TimingStats::instance().reset();

    for (int step = 0; step < nsteps; ++step) {
        double residual = solver.step();
        if ((step + 1) % 5 == 0 || step == nsteps - 1) {
            std::cout << "  Step " << std::setw(4) << (step + 1)
                      << ", residual = " << std::scientific << std::setprecision(3)
                      << residual << std::endl;
        }
    }

    std::cout << "  Completed " << nsteps << " steps (3D)\n";
    TimingStats::instance().print_summary();
}

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n";
    std::cerr << "Options:\n";
    std::cerr << "  -n STEPS   Number of timesteps per case (default: 20)\n";
    std::cerr << "  -g NxN     Grid size (default: 64x64)\n";
    std::cerr << "  --all      Run all turbulence models\n";
    std::cerr << "  --laminar  Run laminar only\n";
    std::cerr << "  --turb     Run turbulence models only\n";
    std::cerr << "  --poisson  Run Poisson-focused test\n";
    std::cerr << "  --3d       Run 3D solver test\n";
    std::cerr << "  --help     Show this help\n";
}

int main(int argc, char** argv) {
    std::cout << "========================================\n";
    std::cout << "   Kernel-Level Performance Profiler\n";
    std::cout << "========================================\n\n";

    int nsteps = 20;
    int Nx = 64, Ny = 64, Nz = 32;
    bool run_all = false;
    bool run_laminar = false;
    bool run_turb = false;
    bool run_poisson = false;
    bool run_3d = false;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-n" && i + 1 < argc) {
            nsteps = std::atoi(argv[++i]);
        } else if (arg == "-g" && i + 1 < argc) {
            Nx = Ny = std::atoi(argv[++i]);
        } else if (arg == "--all") {
            run_all = true;
        } else if (arg == "--laminar") {
            run_laminar = true;
        } else if (arg == "--turb") {
            run_turb = true;
        } else if (arg == "--poisson") {
            run_poisson = true;
        } else if (arg == "--3d") {
            run_3d = true;
        } else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    // Default: run all if nothing specified
    if (!run_all && !run_laminar && !run_turb && !run_poisson && !run_3d) {
        run_all = true;
    }

    std::cout << "Configuration:\n";
    std::cout << "  Steps per case: " << nsteps << "\n";
    std::cout << "  2D Grid: " << Nx << "x" << Ny << "\n";
    std::cout << "  3D Grid: " << Nx << "x" << Ny << "x" << Nz << "\n";

#ifdef USE_GPU_OFFLOAD
    std::cout << "  GPU offload: ENABLED\n";
#else
    std::cout << "  GPU offload: DISABLED\n";
#endif

#ifdef GPU_PROFILE_KERNELS
    std::cout << "  NVTX profiling: ENABLED\n";
#else
    std::cout << "  NVTX profiling: DISABLED\n";
#endif

    std::cout << std::endl;

    // Run selected tests
    if (run_all || run_laminar) {
        profile_turbulence_model("Laminar", TurbulenceModelType::None, nsteps, Nx, Ny);
    }

    if (run_all || run_turb) {
        profile_turbulence_model("Baseline k-omega", TurbulenceModelType::Baseline, nsteps, Nx, Ny);
        profile_turbulence_model("GEP", TurbulenceModelType::GEP, nsteps, Nx, Ny);
        profile_turbulence_model("SST k-omega", TurbulenceModelType::SSTKOmega, nsteps, Nx, Ny);
        profile_turbulence_model("k-omega", TurbulenceModelType::KOmega, nsteps, Nx, Ny);
        profile_turbulence_model("EARSM-WJ", TurbulenceModelType::EARSM_WJ, nsteps, Nx, Ny);
    }

    if (run_all || run_poisson) {
        profile_poisson_solver(nsteps * 2, Nx, Ny);
    }

    if (run_all || run_3d) {
        profile_3d_solver(nsteps, Nx/2, Ny/2, Nz);  // Smaller 3D grid
    }

    std::cout << "\n========================================\n";
    std::cout << "   Profiling Complete\n";
    std::cout << "========================================\n";

    std::cout << "\nTo analyze with NVIDIA Nsight Systems:\n";
    std::cout << "  nsys profile -t nvtx,cuda --stats=true -o profile ./profile_kernels\n";
    std::cout << "  nsys-ui profile.qdrep\n";

    return 0;
}
