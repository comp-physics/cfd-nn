/// HYPRE PFMG solver integration test
///
/// NOTE: HYPRE PFMG with CUDA backend requires the solve_device() path
/// which uses OMP-mapped device arrays. The ScalarField-based solve()
/// method has known issues in CUDA mode.
///
/// For comprehensive HYPRE testing, use the channel application:
///   ./channel --use_hypre --Nx 64 --Ny 128 --Nz 64 --max_iter 100 --model baseline
///
/// This test verifies basic HYPRE initialization and configuration.

#ifdef USE_HYPRE

#include "mesh.hpp"
#include "poisson_solver_hypre.hpp"
#include <iostream>

using namespace nncfd;

int main() {
    std::cout << "=== HYPRE PFMG Integration Test ===\n\n";

    // Test 1: Basic initialization
    std::cout << "Test 1: HYPRE solver initialization... ";
    try {
        Mesh mesh;
        mesh.init_uniform(16, 16, 16, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

        HyprePoissonSolver solver(mesh);

        if (solver.using_cuda()) {
            std::cout << "PASSED (CUDA backend)\n";
        } else {
            std::cout << "PASSED (HOST backend)\n";
        }
    } catch (const std::exception& e) {
        std::cout << "FAILED: " << e.what() << "\n";
        return 1;
    }

    // Test 2: BC configuration
    std::cout << "Test 2: Boundary condition setup... ";
    try {
        Mesh mesh;
        mesh.init_uniform(16, 16, 16, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

        HyprePoissonSolver solver(mesh);

        // Channel flow BCs (primary use case)
        solver.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,   // x
                      PoissonBC::Neumann, PoissonBC::Neumann,     // y
                      PoissonBC::Periodic, PoissonBC::Periodic);  // z

        std::cout << "PASSED\n";
    } catch (const std::exception& e) {
        std::cout << "FAILED: " << e.what() << "\n";
        return 1;
    }

    // Test 3: 2D initialization
    std::cout << "Test 3: 2D mesh support... ";
    try {
        Mesh mesh;
        mesh.init_uniform(32, 32, 0.0, 1.0, 0.0, 1.0);

        HyprePoissonSolver solver(mesh);
        solver.set_bc(PoissonBC::Neumann, PoissonBC::Neumann,
                      PoissonBC::Neumann, PoissonBC::Neumann);

        std::cout << "PASSED\n";
    } catch (const std::exception& e) {
        std::cout << "FAILED: " << e.what() << "\n";
        return 1;
    }

    std::cout << "\n=== All HYPRE integration tests passed ===\n";
    std::cout << "\nFor solve validation, run:\n";
    std::cout << "  ./channel --use_hypre --Nx 32 --Ny 64 --Nz 32 --max_iter 20\n";

    return 0;
}

#else

#include <iostream>

int main() {
    std::cout << "HYPRE support not enabled. Rebuild with -DUSE_HYPRE=ON\n";
    return 0;
}

#endif
