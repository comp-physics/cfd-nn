/// Profiling driver for solver performance analysis
/// Mimics test_solver.cpp setup but runs limited steps for manageable nsys profiles

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>

using namespace nncfd;

// Helper: Initialize velocity with analytical Poiseuille profile
// (Same as test_solver.cpp)
void initialize_poiseuille_profile(RANSSolver& solver, const Mesh& mesh, 
                                   double dp_dx, double nu, double scale = 0.9) {
    double H = 1.0;  // Half-height of channel
    
    // Set u-velocity at x-faces (staggered grid)
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        double y = mesh.y(j);
        double u_analytical = -dp_dx / (2.0 * nu) * (H * H - y * y);
        
        // Apply to all x-faces at this y
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            solver.velocity().u(i, j) = scale * u_analytical;
        }
    }
    
    // v-velocity stays zero (no cross-flow in Poiseuille)
    for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            solver.velocity().v(i, j) = 0.0;
        }
    }
}

void profile_laminar_poiseuille(int nsteps) {
    std::cout << "=== Profiling: Laminar Poiseuille (" << nsteps << " steps) ===\n";
    
    // EXACT same setup as test_solver.cpp test_laminar_poiseuille()
    Mesh mesh;
    mesh.init_uniform(32, 64, 0.0, 4.0, -1.0, 1.0);
    
    Config config;
    config.nu = 0.01;
    config.dp_dx = -0.001;
    config.adaptive_dt = true;
    config.max_iter = nsteps;  // LIMITED for profiling
    config.tol = 1e-8;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = true;  // Show progress
    config.output_freq = 5;  // Print every 5 steps
    
    RANSSolver solver(mesh, config);
    solver.set_body_force(-config.dp_dx, 0.0);
    
    // Initialize close to solution (same as test)
    initialize_poiseuille_profile(solver, mesh, config.dp_dx, config.nu, 0.9);
    
#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif
    
    std::cout << "Running " << nsteps << " solver steps...\n";
    
    // Run limited steady iterations (NOT thousands)
    auto [residual, iters] = solver.solve_steady();
    
    std::cout << "\nCompleted " << iters << " iterations\n";
    std::cout << "Final residual: " << std::scientific << residual << "\n";
}

void profile_divergence_free(int nsteps) {
    std::cout << "\n=== Profiling: Divergence-Free Test (" << nsteps << " steps) ===\n";
    
    // EXACT same setup as test_solver.cpp test_divergence_free()
    Mesh mesh;
    mesh.init_uniform(32, 64, 0.0, 4.0, -1.0, 1.0);
    
    Config config;
    config.nu = 0.01;
    config.dp_dx = -0.001;
    config.adaptive_dt = true;
    config.max_iter = nsteps;  // LIMITED
    config.tol = 1e-7;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = true;
    config.output_freq = 5;
    
    RANSSolver solver(mesh, config);
    solver.set_body_force(-config.dp_dx, 0.0);
    solver.initialize_uniform(0.01, 0.0);
    
#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif
    
    std::cout << "Running " << nsteps << " solver steps...\n";
    
    // Run limited steps (matches test but fewer iterations)
    for (int step = 0; step < nsteps; ++step) {
        double residual = solver.step();
        if ((step + 1) % 5 == 0) {
            std::cout << "Step " << step + 1 << ", residual = " << std::scientific << residual << "\n";
        }
    }
    
    std::cout << "\nCompleted " << nsteps << " steps\n";
}

void profile_mass_conservation(int nsteps) {
    std::cout << "\n=== Profiling: Mass Conservation (" << nsteps << " steps) ===\n";
    
    // EXACT same setup as test_solver.cpp test_mass_conservation()
    Mesh mesh;
    mesh.init_uniform(32, 64, 0.0, 4.0, -1.0, 1.0);
    
    Config config;
    config.nu = 0.01;
    config.dp_dx = -0.001;
    config.adaptive_dt = true;
    config.max_iter = nsteps;
    config.tol = 1e-6;
    config.verbose = true;
    config.output_freq = 5;
    
    RANSSolver solver(mesh, config);
    solver.initialize_uniform(0.1, 0.0);
    solver.set_body_force(-config.dp_dx, 0.0);
    
#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif
    
    std::cout << "Running " << nsteps << " solver steps...\n";
    
    // Run limited steps
    for (int step = 0; step < nsteps; ++step) {
        double residual = solver.step();
        if ((step + 1) % 5 == 0) {
            std::cout << "Step " << step + 1 << ", residual = " << std::scientific << residual << "\n";
        }
    }
    
    std::cout << "\nCompleted " << nsteps << " steps\n";
}

int main(int argc, char** argv) {
    std::cout << "====================================\n";
    std::cout << "   Solver Performance Profiling\n";
    std::cout << "====================================\n\n";
    
    int nsteps = 20;  // Default: 20 steps for manageable profile
    
    // Parse command line
    if (argc > 1) {
        nsteps = std::atoi(argv[1]);
        if (nsteps <= 0 || nsteps > 1000) {
            std::cerr << "Usage: " << argv[0] << " [nsteps]\n";
            std::cerr << "nsteps must be between 1 and 1000 (default: 20)\n";
            return 1;
        }
    }
    
    std::cout << "Profiling configuration:\n";
    std::cout << "  Steps per case: " << nsteps << "\n";
    std::cout << "  Mesh: 32x64 cells\n";
    std::cout << "  Physics: Laminar channel flow\n\n";
    
#ifdef USE_GPU_OFFLOAD
    std::cout << "GPU offload: ENABLED\n\n";
#else
    std::cout << "GPU offload: DISABLED\n\n";
#endif
    
    // Run profiling cases (mimicking test_solver.cpp)
    profile_laminar_poiseuille(nsteps);
    profile_divergence_free(nsteps);
    profile_mass_conservation(nsteps);
    
    std::cout << "\n====================================\n";
    std::cout << "   Profiling Complete\n";
    std::cout << "====================================\n";
    
    return 0;
}

