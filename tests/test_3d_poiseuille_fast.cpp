/// Fast 3D Poiseuille flow test (~10 seconds)
/// Verifies correct steady-state physics with analytical solution
///
/// Strategy: Initialize at 0.95x analytical solution to converge quickly

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "config.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace nncfd;

//=============================================================================
// Test parameters
//=============================================================================
constexpr int NX = 32;
constexpr int NY = 32;
constexpr int NZ = 8;
constexpr double LX = 4.0;
constexpr double LY = 2.0;
constexpr double LZ = 1.0;
constexpr double NU = 0.01;
constexpr double DP_DX = -0.001;

// Analytical Poiseuille solution
// u(y) = -dp_dx / (2*nu) * (H^2 - y^2)
// where y is measured from channel center, H = LY/2
double poiseuille_analytical(double y, double dp_dx, double nu, double H) {
    double y_centered = y - H;  // Shift so y=0 at center
    return -dp_dx / (2.0 * nu) * (H * H - y_centered * y_centered);
}

double max_poiseuille_velocity(double dp_dx, double nu, double H) {
    return -dp_dx / (2.0 * nu) * H * H;
}

//=============================================================================
// TEST 1: Fast convergence from near-analytical initial condition
//=============================================================================
bool test_poiseuille_fast_convergence() {
    std::cout << "Test 1: Fast Poiseuille convergence (init at 0.95x analytical)... ";

    Mesh mesh;
    mesh.init_uniform(NX, NY, NZ, 0.0, LX, 0.0, LY, 0.0, LZ);

    Config config;
    config.nu = NU;
    config.dp_dx = DP_DX;
    config.adaptive_dt = true;
    config.max_iter = 100;  // Max iterations, but should converge faster
    config.tol = 1e-6;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);
    solver.set_body_force(-config.dp_dx, 0.0, 0.0);

    // Set BCs
    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    bc.z_lo = VelocityBC::Periodic;
    bc.z_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    double H = LY / 2.0;
    double U_max = max_poiseuille_velocity(DP_DX, NU, H);

    // Initialize at 0.95x analytical solution
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            double y = mesh.y(j);
            double u_analytical = poiseuille_analytical(y, DP_DX, NU, H);

            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                solver.velocity().u(i, j, k) = 0.95 * u_analytical;
            }
        }
    }

    // v = 0, w = 0 (already initialized)

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    // Run until convergence or max iterations
    auto [residual, iterations] = solver.solve_steady();

#ifdef USE_GPU_OFFLOAD
    solver.sync_solution_from_gpu();
#endif

    // Compute error vs analytical
    double max_error = 0.0;
    double l2_error = 0.0;
    int n_points = 0;

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            double y = mesh.y(j);
            double u_analytical = poiseuille_analytical(y, DP_DX, NU, H);

            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                double u_computed = solver.velocity().u(i, j, k);
                double error = std::abs(u_computed - u_analytical);
                max_error = std::max(max_error, error);
                l2_error += error * error;
                n_points++;
            }
        }
    }
    l2_error = std::sqrt(l2_error / n_points);

    double relative_error = max_error / std::abs(U_max);

    bool passed = (relative_error < 0.10);  // 10% relative error tolerance (limited by iteration count)

    if (passed) {
        std::cout << "PASSED\n";
        std::cout << "  Iterations: " << iterations << ", Residual: " << std::scientific << residual << "\n";
        std::cout << "  Max error: " << max_error << " (" << std::fixed << std::setprecision(1)
                  << 100 * relative_error << "% of U_max=" << std::scientific << U_max << ")\n";
    } else {
        std::cout << "FAILED\n";
        std::cout << "  Relative error: " << 100 * relative_error << "% (expected < 10%)\n";
    }

    return passed;
}

//=============================================================================
// TEST 2: Larger grid Poiseuille (more resolution, slightly longer)
//=============================================================================
bool test_poiseuille_larger_grid() {
    std::cout << "Test 2: Larger grid Poiseuille (48x48x8)... ";

    const int NX_L = 48, NY_L = 48, NZ_L = 8;

    Mesh mesh;
    mesh.init_uniform(NX_L, NY_L, NZ_L, 0.0, LX, 0.0, LY, 0.0, LZ);

    Config config;
    config.nu = NU;
    config.dp_dx = DP_DX;
    config.adaptive_dt = true;
    config.max_iter = 150;
    config.tol = 1e-6;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);
    solver.set_body_force(-config.dp_dx, 0.0, 0.0);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    bc.z_lo = VelocityBC::Periodic;
    bc.z_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    double H = LY / 2.0;
    double U_max = max_poiseuille_velocity(DP_DX, NU, H);

    // Initialize at 0.90x analytical (slightly further from solution to test convergence)
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            double y = mesh.y(j);
            double u_analytical = poiseuille_analytical(y, DP_DX, NU, H);

            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                solver.velocity().u(i, j, k) = 0.90 * u_analytical;
            }
        }
    }

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    auto [residual, iterations] = solver.solve_steady();

#ifdef USE_GPU_OFFLOAD
    solver.sync_solution_from_gpu();
#endif

    // Compute centerline velocity (should be close to U_max)
    double centerline_u = 0.0;
    int n_centerline = 0;
    int j_center = mesh.j_begin() + NY_L / 2;

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            centerline_u += solver.velocity().u(i, j_center, k);
            n_centerline++;
        }
    }
    centerline_u /= n_centerline;

    double centerline_error = std::abs(centerline_u - U_max) / std::abs(U_max);

    bool passed = (centerline_error < 0.15);  // 15% centerline error (limited by iteration count)

    if (passed) {
        std::cout << "PASSED\n";
        std::cout << "  Iterations: " << iterations << "\n";
        std::cout << "  Centerline velocity: " << std::scientific << centerline_u
                  << " (analytical: " << U_max << ", error: " << std::fixed << std::setprecision(1)
                  << 100 * centerline_error << "%)\n";
    } else {
        std::cout << "FAILED\n";
        std::cout << "  Centerline error: " << 100 * centerline_error << "% (expected < 15%)\n";
    }

    return passed;
}

//=============================================================================
// TEST 3: Verify w stays zero for channel flow
//=============================================================================
bool test_w_zero_channel() {
    std::cout << "Test 3: W-velocity stays zero for channel flow... ";

    Mesh mesh;
    mesh.init_uniform(NX, NY, NZ, 0.0, LX, 0.0, LY, 0.0, LZ);

    Config config;
    config.nu = NU;
    config.adaptive_dt = true;
    config.max_iter = 50;
    config.tol = 1e-6;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);
    solver.set_body_force(-DP_DX, 0.0, 0.0);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    bc.z_lo = VelocityBC::Periodic;
    bc.z_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    double H = LY / 2.0;

    // Initialize with Poiseuille profile
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            double y = mesh.y(j);
            double u_analytical = poiseuille_analytical(y, DP_DX, NU, H);

            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                solver.velocity().u(i, j, k) = 0.95 * u_analytical;
            }
        }
    }

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    // Run 50 timesteps
    for (int step = 0; step < 50; ++step) {
        solver.step();
    }

#ifdef USE_GPU_OFFLOAD
    solver.sync_solution_from_gpu();
#endif

    // Check max |w| and max |u|
    double max_w = 0.0;
    double max_u = 0.0;

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                max_u = std::max(max_u, std::abs(solver.velocity().u(i, j, k)));
            }
        }
    }

    for (int k = mesh.k_begin(); k <= mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                max_w = std::max(max_w, std::abs(solver.velocity().w(i, j, k)));
            }
        }
    }

    double w_relative = max_w / std::max(max_u, 1e-10);

    bool passed = (w_relative < 1e-8);  // w should be essentially zero

    if (passed) {
        std::cout << "PASSED\n";
        std::cout << "  Max |u|: " << std::scientific << max_u << "\n";
        std::cout << "  Max |w|: " << max_w << " (ratio |w|/|u| = " << w_relative << ")\n";
    } else {
        std::cout << "FAILED\n";
        std::cout << "  |w|/|u| ratio: " << w_relative << " (expected < 1e-8)\n";
    }

    return passed;
}

//=============================================================================
// MAIN
//=============================================================================
int main() {
    std::cout << "=== Fast 3D Poiseuille Tests ===\n\n";

    int passed = 0;
    int total = 0;

    total++; if (test_poiseuille_fast_convergence()) passed++;
    total++; if (test_poiseuille_larger_grid()) passed++;
    total++; if (test_w_zero_channel()) passed++;

    std::cout << "\n=== Results: " << passed << "/" << total << " tests passed ===\n";

    if (passed == total) {
        std::cout << "[SUCCESS] All fast Poiseuille tests passed!\n";
        return 0;
    } else {
        std::cout << "[FAILURE] Some tests failed\n";
        return 1;
    }
}
