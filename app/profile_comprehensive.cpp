/// @file profile_comprehensive.cpp
/// @brief Comprehensive profiling across BCs, Poisson solvers, and turbulence models
///
/// Runs 128³ grids with various configurations for NVTX profiling.
/// Usage: nsys profile --stats=true -t cuda,nvtx ./profile_comprehensive
///
/// Configurations tested:
/// - BCs: All-periodic, Channel (walls y), Duct (walls y+z)
/// - Poisson: MG, MG+Graph, FFT (periodic only), HYPRE
/// - Turbulence: None (laminar), Baseline (Smagorinsky), SST k-omega

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "profiling.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include <cstdlib>

using namespace nncfd;
using namespace std::chrono;

// ============================================================================
// Configuration structures
// ============================================================================

struct ProfileConfig {
    std::string name;
    int Nx, Ny, Nz;

    // Velocity BCs
    VelocityBC::Type bc_x_lo, bc_x_hi;
    VelocityBC::Type bc_y_lo, bc_y_hi;
    VelocityBC::Type bc_z_lo, bc_z_hi;

    // Solver type
    PoissonSolverType poisson_solver;
    bool use_vcycle_graph;  // MG_USE_VCYCLE_GRAPH

    // Turbulence
    TurbulenceModelType turb_model;

    // Run params
    int nsteps;
};

// ============================================================================
// Helper functions
// ============================================================================

std::string bc_to_string(VelocityBC::Type bc) {
    switch (bc) {
        case VelocityBC::Periodic: return "P";
        case VelocityBC::NoSlip: return "W";
        case VelocityBC::Inflow: return "I";
        case VelocityBC::Outflow: return "O";
        default: return "?";
    }
}

std::string poisson_to_string(PoissonSolverType solver, bool graph) {
    std::string base;
    switch (solver) {
        case PoissonSolverType::MG: base = "MG"; break;
        case PoissonSolverType::FFT: base = "FFT"; break;
        case PoissonSolverType::FFT1D: base = "FFT1D"; break;
        case PoissonSolverType::FFT2D: base = "FFT2D"; break;
        case PoissonSolverType::HYPRE: base = "HYPRE"; break;
        case PoissonSolverType::Auto: base = "Auto"; break;
        default: base = "?";
    }
    if (solver == PoissonSolverType::MG && graph) {
        base += "+Graph";
    }
    return base;
}

std::string turb_to_string(TurbulenceModelType turb) {
    switch (turb) {
        case TurbulenceModelType::None: return "Laminar";
        case TurbulenceModelType::Baseline: return "Smagorinsky";
        case TurbulenceModelType::SSTKOmega: return "SST-kw";
        case TurbulenceModelType::KOmega: return "k-w";
        case TurbulenceModelType::GEP: return "GEP";
        default: return "Other";
    }
}

void print_config(const ProfileConfig& cfg) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Profile: " << cfg.name << "\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "  Grid: " << cfg.Nx << "x" << cfg.Ny << "x" << cfg.Nz << "\n";
    std::cout << "  BCs: x=" << bc_to_string(cfg.bc_x_lo) << bc_to_string(cfg.bc_x_hi)
              << " y=" << bc_to_string(cfg.bc_y_lo) << bc_to_string(cfg.bc_y_hi)
              << " z=" << bc_to_string(cfg.bc_z_lo) << bc_to_string(cfg.bc_z_hi) << "\n";
    std::cout << "  Poisson: " << poisson_to_string(cfg.poisson_solver, cfg.use_vcycle_graph) << "\n";
    std::cout << "  Turbulence: " << turb_to_string(cfg.turb_model) << "\n";
    std::cout << "  Steps: " << cfg.nsteps << "\n";
    std::cout << std::string(70, '-') << "\n";
}

// ============================================================================
// Run a single profile configuration
// ============================================================================

void run_profile(const ProfileConfig& cfg) {
    NVTX_SCOPE_SOLVER(cfg.name.c_str());

    print_config(cfg);

    // Set environment for MG graph mode
    if (cfg.use_vcycle_graph) {
        setenv("MG_USE_VCYCLE_GRAPH", "1", 1);
    } else {
        unsetenv("MG_USE_VCYCLE_GRAPH");
    }

    // Create mesh - cubic domain for simplicity
    const double L = 2.0 * M_PI;

    Mesh mesh;
    mesh.init_uniform(cfg.Nx, cfg.Ny, cfg.Nz, 0.0, L, 0.0, L, 0.0, L);

    // Configure solver
    Config config;
    config.nu = 1e-4;  // Re ~ 10000 based on unit velocity
    config.dt = 1e-4;
    config.adaptive_dt = false;  // Fixed dt for consistent profiling
    config.max_iter = cfg.nsteps;
    config.tol = 1e-6;
    config.turb_model = cfg.turb_model;
    config.poisson_solver = cfg.poisson_solver;
    config.verbose = false;
    config.output_freq = 0;  // No console spam
    config.write_fields = false;  // No VTK output
    config.num_snapshots = 0;

    // Use fixed V-cycle count to enable CUDA Graph path for MG+Graph configs
    config.poisson_fixed_cycles = 10;  // 10 V-cycles per solve (no convergence check)

    // Create solver
    RANSSolver solver(mesh, config);

    // Set velocity BCs
    VelocityBC bc;
    bc.x_lo = cfg.bc_x_lo;
    bc.x_hi = cfg.bc_x_hi;
    bc.y_lo = cfg.bc_y_lo;
    bc.y_hi = cfg.bc_y_hi;
    bc.z_lo = cfg.bc_z_lo;
    bc.z_hi = cfg.bc_z_hi;
    solver.set_velocity_bc(bc);

    // Initialize with Taylor-Green-like vortex (divergence-free)
    const double U0 = 1.0;
    for (int k = 1; k <= cfg.Nz; ++k) {
        double z = mesh.z(k);
        for (int j = 1; j <= cfg.Ny; ++j) {
            double y = mesh.y(j);
            for (int i = 1; i <= cfg.Nx + 1; ++i) {
                double x = mesh.xf[i];
                solver.velocity().u(i, j, k) = U0 * std::sin(x) * std::cos(y) * std::cos(z);
            }
        }
    }
    for (int k = 1; k <= cfg.Nz; ++k) {
        double z = mesh.z(k);
        for (int j = 1; j <= cfg.Ny + 1; ++j) {
            double y = mesh.yf[j];
            for (int i = 1; i <= cfg.Nx; ++i) {
                double x = mesh.x(i);
                solver.velocity().v(i, j, k) = -U0 * std::cos(x) * std::sin(y) * std::cos(z);
            }
        }
    }
    // w stays zero

    // Sync to GPU
#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    // Warmup step (triggers JIT, graph capture, etc.)
    {
        NVTX_SCOPE_SOLVER("warmup");
        solver.step();
    }

    // Timed run
    auto start = high_resolution_clock::now();
    {
        NVTX_SCOPE_SOLVER("timed_steps");
        for (int i = 1; i < cfg.nsteps; ++i) {
            solver.step();
        }
    }
    auto end = high_resolution_clock::now();

    double elapsed_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
    double ms_per_step = elapsed_ms / (cfg.nsteps - 1);

    std::cout << "\n  Results:\n";
    std::cout << "    Total time: " << std::fixed << std::setprecision(2) << elapsed_ms << " ms\n";
    std::cout << "    Per step:   " << std::setprecision(3) << ms_per_step << " ms\n";
    std::cout << "    Throughput: " << std::setprecision(2)
              << (cfg.Nx * cfg.Ny * cfg.Nz) / (ms_per_step * 1000.0) << " Mcells/s\n";
}

// ============================================================================
// Main: Define and run all configurations
// ============================================================================

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    std::cout << "======================================================================\n";
    std::cout << "  Comprehensive NVTX Profiling Suite\n";
    std::cout << "======================================================================\n";

    const int N = 128;
    const int nsteps = 10;

    std::cout << "  Grid: " << N << "³ | Steps: " << nsteps << " | No I/O\n";
    std::cout << "======================================================================\n";

    // Build configuration matrix
    std::vector<ProfileConfig> configs;

    // ========================================================================
    // 1. BC variations with MG solver (laminar)
    // ========================================================================

    // All periodic (Taylor-Green decay)
    configs.push_back({
        "AllPeriodic_MG_Laminar", N, N, N,
        VelocityBC::Periodic, VelocityBC::Periodic,
        VelocityBC::Periodic, VelocityBC::Periodic,
        VelocityBC::Periodic, VelocityBC::Periodic,
        PoissonSolverType::MG, false,
        TurbulenceModelType::None, nsteps
    });

    // All periodic with MG+Graph
    configs.push_back({
        "AllPeriodic_MG+Graph_Laminar", N, N, N,
        VelocityBC::Periodic, VelocityBC::Periodic,
        VelocityBC::Periodic, VelocityBC::Periodic,
        VelocityBC::Periodic, VelocityBC::Periodic,
        PoissonSolverType::MG, true,
        TurbulenceModelType::None, nsteps
    });

    // Channel: Periodic x/z, NoSlip walls y
    configs.push_back({
        "Channel_MG_Laminar", N, N, N,
        VelocityBC::Periodic, VelocityBC::Periodic,
        VelocityBC::NoSlip, VelocityBC::NoSlip,
        VelocityBC::Periodic, VelocityBC::Periodic,
        PoissonSolverType::MG, false,
        TurbulenceModelType::None, nsteps
    });

    // Channel with MG+Graph
    configs.push_back({
        "Channel_MG+Graph_Laminar", N, N, N,
        VelocityBC::Periodic, VelocityBC::Periodic,
        VelocityBC::NoSlip, VelocityBC::NoSlip,
        VelocityBC::Periodic, VelocityBC::Periodic,
        PoissonSolverType::MG, true,
        TurbulenceModelType::None, nsteps
    });

    // Duct: Periodic x, NoSlip walls y and z
    configs.push_back({
        "Duct_MG_Laminar", N, N, N,
        VelocityBC::Periodic, VelocityBC::Periodic,
        VelocityBC::NoSlip, VelocityBC::NoSlip,
        VelocityBC::NoSlip, VelocityBC::NoSlip,
        PoissonSolverType::MG, false,
        TurbulenceModelType::None, nsteps
    });

    // Duct with MG+Graph
    configs.push_back({
        "Duct_MG+Graph_Laminar", N, N, N,
        VelocityBC::Periodic, VelocityBC::Periodic,
        VelocityBC::NoSlip, VelocityBC::NoSlip,
        VelocityBC::NoSlip, VelocityBC::NoSlip,
        PoissonSolverType::MG, true,
        TurbulenceModelType::None, nsteps
    });

    // ========================================================================
    // 2. Poisson solver variations (all-periodic for FFT compatibility)
    // ========================================================================

    // FFT (requires all-periodic)
    configs.push_back({
        "AllPeriodic_FFT_Laminar", N, N, N,
        VelocityBC::Periodic, VelocityBC::Periodic,
        VelocityBC::Periodic, VelocityBC::Periodic,
        VelocityBC::Periodic, VelocityBC::Periodic,
        PoissonSolverType::FFT, false,
        TurbulenceModelType::None, nsteps
    });

    // FFT1D (channel compatible - periodic x/z)
    configs.push_back({
        "Channel_FFT1D_Laminar", N, N, N,
        VelocityBC::Periodic, VelocityBC::Periodic,
        VelocityBC::NoSlip, VelocityBC::NoSlip,
        VelocityBC::Periodic, VelocityBC::Periodic,
        PoissonSolverType::FFT1D, false,
        TurbulenceModelType::None, nsteps
    });

    // HYPRE (channel BCs)
    configs.push_back({
        "Channel_HYPRE_Laminar", N, N, N,
        VelocityBC::Periodic, VelocityBC::Periodic,
        VelocityBC::NoSlip, VelocityBC::NoSlip,
        VelocityBC::Periodic, VelocityBC::Periodic,
        PoissonSolverType::HYPRE, false,
        TurbulenceModelType::None, nsteps
    });

    // ========================================================================
    // 3. Turbulence model variations (channel BCs, MG+Graph)
    // ========================================================================

    // Smagorinsky LES
    configs.push_back({
        "Channel_MG+Graph_Smagorinsky", N, N, N,
        VelocityBC::Periodic, VelocityBC::Periodic,
        VelocityBC::NoSlip, VelocityBC::NoSlip,
        VelocityBC::Periodic, VelocityBC::Periodic,
        PoissonSolverType::MG, true,
        TurbulenceModelType::Baseline, nsteps
    });

    // SST k-omega RANS
    configs.push_back({
        "Channel_MG+Graph_SST-kw", N, N, N,
        VelocityBC::Periodic, VelocityBC::Periodic,
        VelocityBC::NoSlip, VelocityBC::NoSlip,
        VelocityBC::Periodic, VelocityBC::Periodic,
        PoissonSolverType::MG, true,
        TurbulenceModelType::SSTKOmega, nsteps
    });

    // k-omega (standard)
    configs.push_back({
        "Channel_MG+Graph_kw", N, N, N,
        VelocityBC::Periodic, VelocityBC::Periodic,
        VelocityBC::NoSlip, VelocityBC::NoSlip,
        VelocityBC::Periodic, VelocityBC::Periodic,
        PoissonSolverType::MG, true,
        TurbulenceModelType::KOmega, nsteps
    });

    // ========================================================================
    // Run all configurations
    // ========================================================================

    std::cout << "\nTotal configurations: " << configs.size() << "\n";

    for (size_t i = 0; i < configs.size(); ++i) {
        std::cout << "\n>>> Running " << (i + 1) << "/" << configs.size() << "\n";
        try {
            run_profile(configs[i]);
        } catch (const std::exception& e) {
            std::cerr << "  ERROR: " << e.what() << "\n";
            std::cerr << "  Skipping this configuration.\n";
        }
    }

    std::cout << "\n======================================================================\n";
    std::cout << "  Profiling Complete\n";
    std::cout << "======================================================================\n";

    return 0;
}
