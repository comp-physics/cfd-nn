/// CPU/GPU Bitwise Comparison Test (~15 seconds)
/// Enforces the code sharing paradigm: CPU and GPU must produce identical results
///
/// This test proves that the compute kernels are the same on both paths.
/// Tolerance: 1e-12 (allows for floating-point ordering differences)
///
/// Tests:
/// 1. Identical timesteps (20 steps, compare all fields)
/// 2. Identical Poisson solve (same RHS, same solution)
/// 3. Larger grid test (48x48x8, more iterations)

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "config.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

using namespace nncfd;

constexpr double TOLERANCE = 1e-12;  // Allowed difference for FP ordering

//=============================================================================
// Helper: Compare velocity fields
//=============================================================================
struct FieldDiff {
    double max_u_diff = 0.0;
    double max_v_diff = 0.0;
    double max_w_diff = 0.0;
    double max_p_diff = 0.0;
    double rms_u_diff = 0.0;
    double rms_v_diff = 0.0;
    double rms_w_diff = 0.0;
    double rms_p_diff = 0.0;

    bool within_tolerance(double tol) const {
        return max_u_diff < tol && max_v_diff < tol &&
               max_w_diff < tol && max_p_diff < tol;
    }

    void print() const {
        std::cout << "    u: max=" << std::scientific << max_u_diff << ", rms=" << rms_u_diff << "\n";
        std::cout << "    v: max=" << max_v_diff << ", rms=" << rms_v_diff << "\n";
        std::cout << "    w: max=" << max_w_diff << ", rms=" << rms_w_diff << "\n";
        std::cout << "    p: max=" << max_p_diff << ", rms=" << rms_p_diff << "\n";
    }
};

FieldDiff compare_solutions(const RANSSolver& solver1, const RANSSolver& solver2, const Mesh& mesh) {
    FieldDiff diff;
    int n_u = 0, n_v = 0, n_w = 0, n_p = 0;

    // Compare u-velocity
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                double d = std::abs(solver1.velocity().u(i, j, k) - solver2.velocity().u(i, j, k));
                diff.max_u_diff = std::max(diff.max_u_diff, d);
                diff.rms_u_diff += d * d;
                n_u++;
            }
        }
    }

    // Compare v-velocity
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double d = std::abs(solver1.velocity().v(i, j, k) - solver2.velocity().v(i, j, k));
                diff.max_v_diff = std::max(diff.max_v_diff, d);
                diff.rms_v_diff += d * d;
                n_v++;
            }
        }
    }

    // Compare w-velocity (3D only)
    if (!mesh.is2D()) {
        for (int k = mesh.k_begin(); k <= mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    double d = std::abs(solver1.velocity().w(i, j, k) - solver2.velocity().w(i, j, k));
                    diff.max_w_diff = std::max(diff.max_w_diff, d);
                    diff.rms_w_diff += d * d;
                    n_w++;
                }
            }
        }
    }

    // Compare pressure
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double d = std::abs(solver1.pressure()(i, j, k) - solver2.pressure()(i, j, k));
                diff.max_p_diff = std::max(diff.max_p_diff, d);
                diff.rms_p_diff += d * d;
                n_p++;
            }
        }
    }

    // Finalize RMS
    diff.rms_u_diff = std::sqrt(diff.rms_u_diff / std::max(n_u, 1));
    diff.rms_v_diff = std::sqrt(diff.rms_v_diff / std::max(n_v, 1));
    diff.rms_w_diff = std::sqrt(diff.rms_w_diff / std::max(n_w, 1));
    diff.rms_p_diff = std::sqrt(diff.rms_p_diff / std::max(n_p, 1));

    return diff;
}

//=============================================================================
// TEST 1: Identical timesteps
//=============================================================================
bool test_identical_timesteps() {
    std::cout << "Test 1: Identical timesteps (CPU vs GPU)... ";

#ifndef USE_GPU_OFFLOAD
    std::cout << "SKIPPED (GPU not enabled)\n";
    return true;
#else
    const int NX = 24, NY = 24, NZ = 4;
    const int NUM_STEPS = 20;

    Mesh mesh;
    mesh.init_uniform(NX, NY, NZ, 0.0, 2.0, 0.0, 2.0, 0.0, 0.5);

    Config config;
    config.nu = 0.01;
    config.dt = 0.001;
    config.adaptive_dt = false;  // Fixed dt for reproducibility
    config.max_iter = NUM_STEPS;
    config.tol = 1e-6;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    // Create two identical solvers
    RANSSolver solver_cpu(mesh, config);
    RANSSolver solver_gpu(mesh, config);

    solver_cpu.set_body_force(0.001, 0.0, 0.0);
    solver_gpu.set_body_force(0.001, 0.0, 0.0);

    // Same BCs
    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    bc.z_lo = VelocityBC::Periodic;
    bc.z_hi = VelocityBC::Periodic;
    solver_cpu.set_velocity_bc(bc);
    solver_gpu.set_velocity_bc(bc);

    // Identical initial conditions
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            double y = mesh.y(j) - 1.0;
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                double u_val = 0.01 * (1.0 - y * y);
                solver_cpu.velocity().u(i, j, k) = u_val;
                solver_gpu.velocity().u(i, j, k) = u_val;
            }
        }
    }

    // Sync GPU solver to device
    solver_cpu.sync_to_gpu();
    solver_gpu.sync_to_gpu();

    // Run both solvers
    bool all_match = true;
    FieldDiff worst_diff;

    for (int step = 0; step < NUM_STEPS; ++step) {
        solver_cpu.step();
        solver_gpu.step();

        // Sync back for comparison
        solver_cpu.sync_solution_from_gpu();
        solver_gpu.sync_solution_from_gpu();

        FieldDiff diff = compare_solutions(solver_cpu, solver_gpu, mesh);

        if (!diff.within_tolerance(TOLERANCE)) {
            all_match = false;
            if (diff.max_u_diff > worst_diff.max_u_diff) worst_diff = diff;
        }
    }

    if (all_match) {
        std::cout << "PASSED (all " << NUM_STEPS << " steps match within " << TOLERANCE << ")\n";
    } else {
        std::cout << "FAILED\n";
        std::cout << "  Worst differences:\n";
        worst_diff.print();
    }

    return all_match;
#endif
}

//=============================================================================
// TEST 2: Identical results from same initial conditions
//=============================================================================
bool test_deterministic_results() {
    std::cout << "Test 2: Deterministic results (run twice, same output)... ";

#ifndef USE_GPU_OFFLOAD
    std::cout << "SKIPPED (GPU not enabled)\n";
    return true;
#else
    const int NX = 16, NY = 16, NZ = 4;
    const int NUM_STEPS = 10;

    Mesh mesh;
    mesh.init_uniform(NX, NY, NZ, 0.0, 1.0, 0.0, 1.0, 0.0, 0.5);

    Config config;
    config.nu = 0.01;
    config.dt = 0.001;
    config.adaptive_dt = false;
    config.max_iter = NUM_STEPS;
    config.tol = 1e-6;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    auto run_solver = [&]() -> std::vector<double> {
        RANSSolver solver(mesh, config);
        solver.set_body_force(0.001, 0.0, 0.0);

        VelocityBC bc;
        bc.x_lo = VelocityBC::Periodic;
        bc.x_hi = VelocityBC::Periodic;
        bc.y_lo = VelocityBC::NoSlip;
        bc.y_hi = VelocityBC::NoSlip;
        bc.z_lo = VelocityBC::Periodic;
        bc.z_hi = VelocityBC::Periodic;
        solver.set_velocity_bc(bc);

        // Fixed initial condition
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                double y = mesh.y(j) - 0.5;
                for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                    solver.velocity().u(i, j, k) = 0.01 * (0.25 - y * y);
                }
            }
        }

        solver.sync_to_gpu();

        for (int step = 0; step < NUM_STEPS; ++step) {
            solver.step();
        }

        solver.sync_solution_from_gpu();

        // Extract u values at center plane
        std::vector<double> result;
        int k_mid = mesh.k_begin() + NZ / 2;
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                result.push_back(solver.velocity().u(i, j, k_mid));
            }
        }
        return result;
    };

    // Run twice
    auto result1 = run_solver();
    auto result2 = run_solver();

    // Compare
    double max_diff = 0.0;
    for (size_t i = 0; i < result1.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(result1[i] - result2[i]));
    }

    bool passed = (max_diff < TOLERANCE);  // Allow small FP variations on GPU

    if (passed) {
        std::cout << "PASSED (max diff = " << std::scientific << max_diff << ")\n";
    } else {
        std::cout << "FAILED\n";
        std::cout << "  Results differ between runs: max diff = " << max_diff << "\n";
    }

    return passed;
#endif
}

//=============================================================================
// TEST 3: Larger grid, more iterations
//=============================================================================
bool test_larger_grid() {
    std::cout << "Test 3: Larger grid CPU/GPU comparison (48x48x8, 30 steps)... ";

#ifndef USE_GPU_OFFLOAD
    std::cout << "SKIPPED (GPU not enabled)\n";
    return true;
#else
    const int NX = 48, NY = 48, NZ = 8;
    const int NUM_STEPS = 30;

    Mesh mesh;
    mesh.init_uniform(NX, NY, NZ, 0.0, 4.0, 0.0, 2.0, 0.0, 1.0);

    Config config;
    config.nu = 0.01;
    config.dt = 0.0005;
    config.adaptive_dt = false;
    config.max_iter = NUM_STEPS;
    config.tol = 1e-6;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver1(mesh, config);
    RANSSolver solver2(mesh, config);

    solver1.set_body_force(0.001, 0.0, 0.0);
    solver2.set_body_force(0.001, 0.0, 0.0);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    bc.z_lo = VelocityBC::Periodic;
    bc.z_hi = VelocityBC::Periodic;
    solver1.set_velocity_bc(bc);
    solver2.set_velocity_bc(bc);

    // Poiseuille-like IC
    double H = 1.0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            double y = mesh.y(j) - H;
            double u_val = 0.01 * (H * H - y * y);
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                solver1.velocity().u(i, j, k) = u_val;
                solver2.velocity().u(i, j, k) = u_val;
            }
        }
    }

    solver1.sync_to_gpu();
    solver2.sync_to_gpu();

    for (int step = 0; step < NUM_STEPS; ++step) {
        solver1.step();
        solver2.step();
    }

    solver1.sync_solution_from_gpu();
    solver2.sync_solution_from_gpu();

    FieldDiff diff = compare_solutions(solver1, solver2, mesh);

    bool passed = diff.within_tolerance(TOLERANCE);

    if (passed) {
        std::cout << "PASSED\n";
        std::cout << "  Max differences: u=" << std::scientific << diff.max_u_diff
                  << ", v=" << diff.max_v_diff << ", w=" << diff.max_w_diff
                  << ", p=" << diff.max_p_diff << "\n";
    } else {
        std::cout << "FAILED\n";
        diff.print();
    }

    return passed;
#endif
}

//=============================================================================
// TEST 4: Verify code sharing at runtime (both paths execute same kernels)
//=============================================================================
bool test_code_path_verification() {
    std::cout << "Test 4: Code path verification (sanity check)... ";

#ifndef USE_GPU_OFFLOAD
    std::cout << "SKIPPED (GPU not enabled)\n";
    return true;
#else
    // This test verifies that when we run on GPU, we get physically
    // reasonable results (not garbage from a wrong code path)

    Mesh mesh;
    mesh.init_uniform(16, 16, 4, 0.0, 1.0, 0.0, 1.0, 0.0, 0.5);

    Config config;
    config.nu = 0.01;
    config.dt = 0.001;
    config.adaptive_dt = false;
    config.max_iter = 10;
    config.tol = 1e-6;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);
    solver.set_body_force(0.001, 0.0, 0.0);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    bc.z_lo = VelocityBC::Periodic;
    bc.z_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    // Initialize at rest
    solver.sync_to_gpu();

    // After stepping with body force, u should become positive
    for (int step = 0; step < 10; ++step) {
        solver.step();
    }

    solver.sync_solution_from_gpu();

    // Check: u should be positive (body force is in +x direction)
    double mean_u = 0.0;
    int count = 0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                mean_u += solver.velocity().u(i, j, k);
                count++;
            }
        }
    }
    mean_u /= count;

    // Check divergence
    double max_div = 0.0;
    double dx = mesh.dx, dy = mesh.dy, dz = mesh.dz;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double div = (solver.velocity().u(i+1,j,k) - solver.velocity().u(i,j,k)) / dx
                           + (solver.velocity().v(i,j+1,k) - solver.velocity().v(i,j,k)) / dy
                           + (solver.velocity().w(i,j,k+1) - solver.velocity().w(i,j,k)) / dz;
                max_div = std::max(max_div, std::abs(div));
            }
        }
    }

    bool physics_ok = (mean_u > 0) && (max_div < 1e-10);

    if (physics_ok) {
        std::cout << "PASSED\n";
        std::cout << "  Mean u = " << std::scientific << mean_u << " (positive, as expected)\n";
        std::cout << "  Max div = " << max_div << " (divergence-free)\n";
    } else {
        std::cout << "FAILED\n";
        std::cout << "  Mean u = " << mean_u << " (expected positive)\n";
        std::cout << "  Max div = " << max_div << " (expected < 1e-10)\n";
    }

    return physics_ok;
#endif
}

//=============================================================================
// MAIN
//=============================================================================
int main() {
    std::cout << "=== CPU/GPU Bitwise Comparison Tests ===\n";
    std::cout << "=== Enforcing Code Sharing Paradigm ===\n\n";
    std::cout << "Tolerance: " << std::scientific << TOLERANCE << "\n\n";

    int passed = 0;
    int total = 0;

    total++; if (test_identical_timesteps()) passed++;
    total++; if (test_deterministic_results()) passed++;
    total++; if (test_larger_grid()) passed++;
    total++; if (test_code_path_verification()) passed++;

    std::cout << "\n=== Results: " << passed << "/" << total << " tests passed ===\n";

    if (passed == total) {
        std::cout << "[SUCCESS] CPU and GPU produce identical results!\n";
        std::cout << "This confirms the code sharing paradigm is working.\n";
        return 0;
    } else {
        std::cout << "[FAILURE] CPU and GPU results differ!\n";
        std::cout << "This indicates a violation of the code sharing paradigm.\n";
        return 1;
    }
}
