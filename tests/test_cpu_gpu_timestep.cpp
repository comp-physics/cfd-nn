#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "config.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace nncfd;

// Compare two scalar fields
double compute_max_diff(const ScalarField& a, const ScalarField& b, const Mesh& mesh) {
    double max_diff = 0.0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double diff = std::abs(a(i, j, k) - b(i, j, k));
                max_diff = std::max(max_diff, diff);
            }
        }
    }
    return max_diff;
}

// Compare two velocity fields
std::pair<double, double> compute_velocity_diff(const VectorField& a, const VectorField& b, const Mesh& mesh) {
    double max_u_diff = 0.0;
    double max_v_diff = 0.0;

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                double u_diff = std::abs(a.u(i, j, k) - b.u(i, j, k));
                max_u_diff = std::max(max_u_diff, u_diff);
            }
        }
    }

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double v_diff = std::abs(a.v(i, j, k) - b.v(i, j, k));
                max_v_diff = std::max(max_v_diff, v_diff);
            }
        }
    }

    return {max_u_diff, max_v_diff};
}

// Initialize Poiseuille flow
void init_poiseuille_3d(RANSSolver& solver, const Mesh& mesh, double dp_dx, double nu) {
    const double LY = 2.0;
    double H = LY / 2.0;
    double y_mid = LY / 2.0;

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            double y = mesh.y(j) - y_mid;
            double u_analytical = -dp_dx / (2.0 * nu) * (H * H - y * y);

            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                solver.velocity().u(i, j, k) = 0.9 * u_analytical;
            }
        }
    }

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                solver.velocity().v(i, j, k) = 0.0;
            }
        }
    }

    if (!mesh.is2D()) {
        for (int k = mesh.k_begin(); k <= mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    solver.velocity().w(i, j, k) = 0.0;
                }
            }
        }
    }
}

int main() {
    std::cout << "=== CPU vs GPU Timestep Comparison ===" << std::endl;

    const int NX = 32;
    const int NY = 32;
    const int NZ = 4;
    const double LX = 4.0;
    const double LY = 2.0;
    const double LZ = 1.0;
    const double NU = 0.01;
    const double DP_DX = -0.001;
    const int NUM_STEPS = 10;  // Check first 10 timesteps

    // Create mesh
    Mesh mesh;
    mesh.init_uniform(NX, NY, NZ, 0.0, LX, 0.0, LY, 0.0, LZ);

    std::cout << "\nMesh: " << NX << "x" << NY << "x" << NZ << std::endl;
    std::cout << "Domain: " << LX << "x" << LY << "x" << LZ << std::endl;

    // CPU solver
    std::cout << "\nInitializing CPU solver..." << std::endl;
    Config config_cpu;
    config_cpu.nu = NU;
    config_cpu.dp_dx = DP_DX;
    config_cpu.adaptive_dt = true;
    config_cpu.tol = 1e-6;
    config_cpu.turb_model = TurbulenceModelType::None;
    config_cpu.verbose = false;

    RANSSolver solver_cpu(mesh, config_cpu);
    solver_cpu.set_body_force(-DP_DX, 0.0, 0.0);
    init_poiseuille_3d(solver_cpu, mesh, DP_DX, NU);

    // GPU solver
    std::cout << "Initializing GPU solver..." << std::endl;
    Config config_gpu;
    config_gpu.nu = NU;
    config_gpu.dp_dx = DP_DX;
    config_gpu.adaptive_dt = true;
    config_gpu.tol = 1e-6;
    config_gpu.turb_model = TurbulenceModelType::None;
    config_gpu.verbose = false;

    RANSSolver solver_gpu(mesh, config_gpu);
    solver_gpu.set_body_force(-DP_DX, 0.0, 0.0);
    init_poiseuille_3d(solver_gpu, mesh, DP_DX, NU);

#ifdef USE_GPU_OFFLOAD
    std::cout << "Syncing both solvers to device..." << std::endl;
    solver_cpu.sync_to_gpu();  // CRITICAL: Upload initial conditions to GPU for solver_cpu too!
    solver_gpu.sync_to_gpu();
#endif

    // Compare initial conditions
    std::cout << "\n=== Initial Conditions ===" << std::endl;
    auto [u0_diff, v0_diff] = compute_velocity_diff(solver_cpu.velocity(), solver_gpu.velocity(), mesh);
    std::cout << std::scientific << std::setprecision(2);
    std::cout << "Initial u difference: " << u0_diff << std::endl;
    std::cout << "Initial v difference: " << v0_diff << std::endl;

    if (u0_diff > 1e-14 || v0_diff > 1e-14) {
        std::cout << "[WARNING] Initial conditions differ!" << std::endl;
    }

    // Run timesteps and compare
    std::cout << "\n=== Timestep-by-Timestep Comparison ===" << std::endl;
    std::cout << std::setw(6) << "Step"
              << std::setw(15) << "CPU Res"
              << std::setw(15) << "GPU Res"
              << std::setw(15) << "u Diff"
              << std::setw(15) << "v Diff"
              << std::setw(15) << "p Diff"
              << std::endl;

    bool diverged = false;
    int diverge_step = -1;

    for (int step = 0; step < NUM_STEPS; ++step) {
        // CPU step
        double res_cpu = solver_cpu.step();

#ifdef USE_GPU_OFFLOAD
        // GPU step
        double res_gpu = solver_gpu.step();

        // Sync BOTH solvers' GPU results back for comparison
        solver_cpu.sync_solution_from_gpu();
        solver_gpu.sync_solution_from_gpu();

        // Compare results
        auto [u_diff, v_diff] = compute_velocity_diff(solver_cpu.velocity(), solver_gpu.velocity(), mesh);
        double p_diff = compute_max_diff(solver_cpu.pressure(), solver_gpu.pressure(), mesh);

        std::cout << std::setw(6) << step + 1
                  << std::setw(15) << res_cpu
                  << std::setw(15) << res_gpu
                  << std::setw(15) << u_diff
                  << std::setw(15) << v_diff
                  << std::setw(15) << p_diff
                  << std::endl;

        // Check for divergence
        if (u_diff > 1e-10 || v_diff > 1e-10 || p_diff > 1e-10) {
            if (!diverged) {
                diverged = true;
                diverge_step = step + 1;
                std::cout << "\n[DIVERGENCE DETECTED at step " << diverge_step << "]" << std::endl;

                // Print detailed diagnostics
                std::cout << "\nDetailed comparison at divergence point:" << std::endl;

                // Find worst point
                double worst_u_diff = 0.0;
                int worst_i = 0, worst_j = 0, worst_k = 0;

                for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
                    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                            double diff = std::abs(solver_cpu.velocity().u(i, j, k) -
                                                  solver_gpu.velocity().u(i, j, k));
                            if (diff > worst_u_diff) {
                                worst_u_diff = diff;
                                worst_i = i;
                                worst_j = j;
                                worst_k = k;
                            }
                        }
                    }
                }

                std::cout << "  Worst u point: (" << worst_i << ", " << worst_j << ", " << worst_k << ")" << std::endl;
                std::cout << "    CPU u: " << solver_cpu.velocity().u(worst_i, worst_j, worst_k) << std::endl;
                std::cout << "    GPU u: " << solver_gpu.velocity().u(worst_i, worst_j, worst_k) << std::endl;
                std::cout << "    Difference: " << worst_u_diff << std::endl;

                // Sample across z-planes
                std::cout << "\n  u-velocity across z-planes at center:" << std::endl;
                int mid_i = mesh.i_begin() + NX/2;
                int mid_j = mesh.j_begin() + NY/2;
                for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
                    double u_cpu = solver_cpu.velocity().u(mid_i, mid_j, k);
                    double u_gpu = solver_gpu.velocity().u(mid_i, mid_j, k);
                    std::cout << "    z=" << k << ": CPU=" << u_cpu
                              << ", GPU=" << u_gpu
                              << ", diff=" << (u_cpu - u_gpu) << std::endl;
                }
            }
        }
#else
        std::cout << std::setw(6) << step + 1
                  << std::setw(15) << res_cpu
                  << std::setw(15) << "N/A"
                  << std::setw(15) << "N/A"
                  << std::setw(15) << "N/A"
                  << std::setw(15) << "N/A"
                  << std::endl;
#endif
    }

#ifdef USE_GPU_OFFLOAD
    std::cout << "\n=== Summary ===" << std::endl;
    if (!diverged) {
        std::cout << "[SUCCESS] CPU and GPU remain synchronized for " << NUM_STEPS << " steps" << std::endl;
        return 0;
    } else {
        std::cout << "[FAILURE] CPU and GPU diverged at step " << diverge_step << std::endl;
        return 1;
    }
#else
    std::cout << "\n[SKIPPED] GPU offload not enabled" << std::endl;
    return 0;
#endif
}
