#include "mesh.hpp"
#include "fields.hpp"
#include "poisson_solver_multigrid.hpp"
#include <iostream>
#include <cmath>
#include <iomanip>

using namespace nncfd;

// Test that 3D Poisson solver gives identical results on CPU and GPU
int main() {
    std::cout << "=== 3D Poisson Solver CPU vs GPU Comparison ===" << std::endl;

    const int NX = 32;
    const int NY = 32;
    const int NZ = 4;
    const double LX = 1.0;
    const double LY = 1.0;
    const double LZ = 1.0;

    // Create mesh
    Mesh mesh;
    mesh.init_uniform(NX, NY, NZ, 0.0, LX, 0.0, LY, 0.0, LZ);

    // Create RHS with known pattern
    ScalarField rhs(mesh, 0.0);
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double x = mesh.x(i);
                double y = mesh.y(j);
                double z = mesh.z(k);
                // Simple forcing term (compatible with periodic BCs)
                rhs(i, j, k) = std::sin(2.0 * M_PI * x) * std::sin(2.0 * M_PI * y);
            }
        }
    }

    // Solve on CPU (force CPU path)
    std::cout << "\nSolving on CPU..." << std::endl;
    ScalarField p_cpu(mesh, 0.0);

    MultigridPoissonSolver solver_cpu(mesh);
    solver_cpu.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                     PoissonBC::Periodic, PoissonBC::Periodic,
                     PoissonBC::Periodic, PoissonBC::Periodic);

    PoissonConfig cfg;
    cfg.tol = 1e-8;
    cfg.max_iter = 100;

    // Force CPU path by not calling sync_level_to_gpu
    int iter_cpu = solver_cpu.solve(rhs, p_cpu, cfg);
    std::cout << "  CPU iterations: " << iter_cpu << std::endl;
    std::cout << "  CPU residual: " << solver_cpu.residual() << std::endl;

#ifdef USE_GPU_OFFLOAD
    // Solve on GPU
    std::cout << "\nSolving on GPU..." << std::endl;
    ScalarField p_gpu(mesh, 0.0);

    MultigridPoissonSolver solver_gpu(mesh);
    solver_gpu.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                     PoissonBC::Periodic, PoissonBC::Periodic,
                     PoissonBC::Periodic, PoissonBC::Periodic);

    // Force GPU path by syncing to GPU
    solver_gpu.sync_level_to_gpu(0);

    int iter_gpu = solver_gpu.solve(rhs, p_gpu, cfg);
    std::cout << "  GPU iterations: " << iter_gpu << std::endl;
    std::cout << "  GPU residual: " << solver_gpu.residual() << std::endl;

    // Compare solutions point-by-point
    std::cout << "\nComparing solutions..." << std::endl;

    double max_abs_diff = 0.0;
    double max_rel_diff = 0.0;
    double sum_sq_diff = 0.0;
    int count = 0;

    int worst_i = 0, worst_j = 0, worst_k = 0;

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double val_cpu = p_cpu(i, j, k);
                double val_gpu = p_gpu(i, j, k);
                double abs_diff = std::abs(val_cpu - val_gpu);
                double rel_diff = abs_diff / (std::abs(val_cpu) + 1e-15);

                sum_sq_diff += abs_diff * abs_diff;
                ++count;

                if (abs_diff > max_abs_diff) {
                    max_abs_diff = abs_diff;
                    max_rel_diff = rel_diff;
                    worst_i = i;
                    worst_j = j;
                    worst_k = k;
                }
            }
        }
    }

    double rms_diff = std::sqrt(sum_sq_diff / count);

    std::cout << std::scientific << std::setprecision(6);
    std::cout << "  Max absolute difference: " << max_abs_diff << std::endl;
    std::cout << "  Max relative difference: " << max_rel_diff << std::endl;
    std::cout << "  RMS difference: " << rms_diff << std::endl;
    std::cout << "  Worst point: (" << worst_i << ", " << worst_j << ", " << worst_k << ")" << std::endl;
    std::cout << "    CPU value: " << p_cpu(worst_i, worst_j, worst_k) << std::endl;
    std::cout << "    GPU value: " << p_gpu(worst_i, worst_j, worst_k) << std::endl;

    // Check sample points across all z-planes
    std::cout << "\nSample points across z-planes:" << std::endl;
    int mid_i = mesh.i_begin() + NX/2;
    int mid_j = mesh.j_begin() + NY/2;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        double val_cpu = p_cpu(mid_i, mid_j, k);
        double val_gpu = p_gpu(mid_i, mid_j, k);
        std::cout << "  z-plane " << k << ": CPU=" << val_cpu
                  << ", GPU=" << val_gpu
                  << ", diff=" << (val_cpu - val_gpu) << std::endl;
    }

    // Pass criteria: RMS difference should be < 1e-10 (numerical precision)
    bool passed = (rms_diff < 1e-10 && iter_cpu == iter_gpu);

    std::cout << "\n";
    if (passed) {
        std::cout << "[SUCCESS] CPU and GPU solutions match within tolerance" << std::endl;
        return 0;
    } else {
        std::cout << "[FAILURE] CPU and GPU solutions differ significantly!" << std::endl;
        if (iter_cpu != iter_gpu) {
            std::cout << "  Iteration count mismatch: CPU=" << iter_cpu << " GPU=" << iter_gpu << std::endl;
        }
        return 1;
    }
#else
    std::cout << "\n[SKIPPED] GPU offload not enabled" << std::endl;
    return 0;
#endif
}
