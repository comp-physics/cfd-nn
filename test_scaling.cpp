#include "mesh.hpp"
#include "solver.hpp"
#include "config.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace nncfd;

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <Nx> <Ny>\n";
        return 1;
    }
    
    int Nx = std::atoi(argv[1]);
    int Ny = std::atoi(argv[2]);
    
    // Create mesh
    Mesh mesh;
    mesh.init_uniform(Nx, Ny, 0.0, 10.0, -1.0, 1.0);
    
    // Setup configuration
    Config config;
    config.nu = 0.01;
    config.dt = 0.001;
    config.dp_dx = -0.001;
    config.adaptive_dt = false;
    config.max_iter = 10000;
    config.tol = 1e-6;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;
    
    // Create solver
    RANSSolver solver(mesh, config);
    
    // Set body force
    solver.set_body_force(-config.dp_dx, 0.0);
    
    // Initialize velocity field (laminar)
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            solver.velocity().u(i, j) = 0.1;
        }
    }
    for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            solver.velocity().v(i, j) = 0.0;
        }
    }
    
#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Mesh: " << Nx << "×" << Ny << " (" << (Nx*Ny) << " cells)\n";
    std::cout << "Running 10 time steps...\n" << std::flush;
    
    // Warm-up step
    solver.step();
    
    // Timed steps
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 10; ++i) {
        solver.step();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    double time_per_step = elapsed.count() / 10.0;
    double total_time = elapsed.count();
    
    std::cout << "Total time:    " << total_time << " s\n";
    std::cout << "Time per step: " << time_per_step << " s\n";
    std::cout << "Steps/sec:     " << (10.0 / total_time) << "\n";
    
    return 0;
}
