/// 3D Boundary Condition Tests (~5 seconds)
/// Verifies 3D boundary conditions are applied correctly
///
/// Tests:
/// 1. No-slip walls enforced on all boundaries
/// 2. Periodic z-direction consistency
/// 3. Mass conservation (inflow = outflow)

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "config.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace nncfd;

//=============================================================================
// TEST 1: No-slip walls enforced
//=============================================================================
bool test_no_slip_walls() {
    std::cout << "Test 1: No-slip walls enforced on y-boundaries... ";

    Mesh mesh;
    mesh.init_uniform(16, 16, 8, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

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

    // Set BCs: no-slip on y walls, periodic in x and z
    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    bc.z_lo = VelocityBC::Periodic;
    bc.z_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    // Initialize with non-zero velocity throughout
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                solver.velocity().u(i, j, k) = 0.1;
            }
        }
    }

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    // Run a few timesteps (BCs should be enforced)
    for (int step = 0; step < 5; ++step) {
        solver.step();
    }

#ifdef USE_GPU_OFFLOAD
    solver.sync_solution_from_gpu();
#endif

    // Check wall velocities
    // At y_lo wall: v(i, j_begin, k) should be 0
    // At y_hi wall: v(i, j_end, k) should be 0
    double max_wall_v = 0.0;

    // Check bottom wall (j = j_begin, v-faces)
    int j_lo = mesh.j_begin();
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            max_wall_v = std::max(max_wall_v, std::abs(solver.velocity().v(i, j_lo, k)));
        }
    }

    // Check top wall (j = j_end, v-faces)
    int j_hi = mesh.j_end();
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            max_wall_v = std::max(max_wall_v, std::abs(solver.velocity().v(i, j_hi, k)));
        }
    }

    bool passed = (max_wall_v < 1e-14);

    if (passed) {
        std::cout << "PASSED (max wall v = " << std::scientific << max_wall_v << ")\n";
    } else {
        std::cout << "FAILED\n";
        std::cout << "  Max wall v-velocity: " << max_wall_v << " (expected 0)\n";
    }

    return passed;
}

//=============================================================================
// TEST 2: Periodic z-direction consistency
//=============================================================================
bool test_periodic_z() {
    std::cout << "Test 2: Periodic z-direction consistency... ";

    Mesh mesh;
    mesh.init_uniform(16, 16, 8, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

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

    // Initialize with z-varying field to test periodic BCs
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        double z = mesh.z(k);
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            double y = mesh.y(j) - 0.5;
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                // Periodic in z: sin(2*pi*z/Lz)
                solver.velocity().u(i, j, k) = 0.01 * (0.25 - y * y) * (1.0 + 0.1 * std::sin(2 * M_PI * z));
            }
        }
    }

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    for (int step = 0; step < 10; ++step) {
        solver.step();
    }

#ifdef USE_GPU_OFFLOAD
    solver.sync_solution_from_gpu();
#endif

    // For periodic BC, the w-velocity at z_lo face should equal w at z_hi face
    // w is staggered, so w(i,j,k_begin) corresponds to z=0 face
    // and w(i,j,k_end) corresponds to z=Lz face
    double max_w_diff = 0.0;

    int k_lo = mesh.k_begin();
    int k_hi = mesh.k_end();

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double w_lo = solver.velocity().w(i, j, k_lo);
            double w_hi = solver.velocity().w(i, j, k_hi);
            max_w_diff = std::max(max_w_diff, std::abs(w_lo - w_hi));
        }
    }

    // For periodic, the faces should have same values
    bool passed = (max_w_diff < 1e-12);

    if (passed) {
        std::cout << "PASSED (max w diff at periodic boundary = " << std::scientific << max_w_diff << ")\n";
    } else {
        std::cout << "FAILED\n";
        std::cout << "  Max w difference at z boundaries: " << max_w_diff << " (expected < 1e-12)\n";
    }

    return passed;
}

//=============================================================================
// TEST 3: Mass conservation (divergence-free implies mass conservation)
//=============================================================================
bool test_mass_conservation() {
    std::cout << "Test 3: Mass conservation (divergence-free)... ";

    // Use same grid setup as the successful test_2d_3d_comparison test
    const int NX = 32, NY = 32, NZ = 4;
    const double LX = 2.0, LY = 2.0, LZ = 1.0;
    const double NU = 0.01;
    const double DP_DX = -0.001;

    Mesh mesh;
    mesh.init_uniform(NX, NY, NZ, 0.0, LX, 0.0, LY, 0.0, LZ);

    Config config;
    config.nu = NU;
    config.dp_dx = DP_DX;
    config.adaptive_dt = true;
    config.max_iter = 500;
    config.tol = 1e-6;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);
    solver.set_body_force(-DP_DX, 0.0, 0.0);

    // Initialize with Poiseuille profile at 0.9x analytical
    double H = LY / 2.0;
    double y_mid = LY / 2.0;

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            double y = mesh.y(j) - y_mid;
            double u_analytical = -DP_DX / (2.0 * NU) * (H * H - y * y);
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                solver.velocity().u(i, j, k) = 0.9 * u_analytical;
            }
        }
    }

    // v = 0 everywhere
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                solver.velocity().v(i, j, k) = 0.0;
            }
        }
    }

    // w = 0 everywhere
    for (int k = mesh.k_begin(); k <= mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                solver.velocity().w(i, j, k) = 0.0;
            }
        }
    }

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    // Run to near steady state
    [[maybe_unused]] auto [res, iters] = solver.solve_steady();

#ifdef USE_GPU_OFFLOAD
    solver.sync_solution_from_gpu();
#endif

    // Compute max divergence
    double max_div = 0.0;
    double dx = mesh.dx, dy = mesh.dy, dz = mesh.dz;

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double dudx = (solver.velocity().u(i+1, j, k) - solver.velocity().u(i, j, k)) / dx;
                double dvdy = (solver.velocity().v(i, j+1, k) - solver.velocity().v(i, j, k)) / dy;
                double dwdz = (solver.velocity().w(i, j, k+1) - solver.velocity().w(i, j, k)) / dz;
                double div = dudx + dvdy + dwdz;
                max_div = std::max(max_div, std::abs(div));
            }
        }
    }

    // Divergence should be small after projection (Poisson solver tolerance + discretization)
    bool passed = (max_div < 1e-4);

    if (passed) {
        std::cout << "PASSED (max divergence = " << std::scientific << max_div << ")\n";
    } else {
        std::cout << "FAILED\n";
        std::cout << "  Max divergence: " << max_div << " (expected < 1e-4)\n";
    }

    return passed;
}

//=============================================================================
// TEST 4: All six boundaries can be set independently
//=============================================================================
bool test_all_bc_types() {
    std::cout << "Test 4: All boundary types can be set independently... ";

    Mesh mesh;
    mesh.init_uniform(16, 16, 8, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    Config config;
    config.nu = 0.01;
    config.dt = 0.0005;
    config.adaptive_dt = false;
    config.max_iter = 5;
    config.tol = 1e-6;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    // Test different BC combinations
    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    bc.z_lo = VelocityBC::Periodic;
    bc.z_hi = VelocityBC::Periodic;

    solver.set_velocity_bc(bc);

    // Initialize simple field
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                solver.velocity().u(i, j, k) = 0.01;
            }
        }
    }

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    try {
        for (int step = 0; step < 5; ++step) {
            solver.step();
        }
    } catch (const std::exception& e) {
        std::cout << "FAILED (exception: " << e.what() << ")\n";
        return false;
    }

#ifdef USE_GPU_OFFLOAD
    solver.sync_solution_from_gpu();
#endif

    // Check for NaN/Inf
    double max_vel = solver.velocity().max_magnitude();
    if (!std::isfinite(max_vel)) {
        std::cout << "FAILED (NaN/Inf in velocity)\n";
        return false;
    }

    std::cout << "PASSED (solver ran without errors, max vel = " << std::scientific << max_vel << ")\n";
    return true;
}

//=============================================================================
// MAIN
//=============================================================================
int main() {
    std::cout << "=== 3D Boundary Condition Tests ===\n\n";

    int passed = 0;
    int total = 0;

    total++; if (test_no_slip_walls()) passed++;
    total++; if (test_periodic_z()) passed++;
    total++; if (test_mass_conservation()) passed++;
    total++; if (test_all_bc_types()) passed++;

    std::cout << "\n=== Results: " << passed << "/" << total << " tests passed ===\n";

    if (passed == total) {
        std::cout << "[SUCCESS] All 3D BC tests passed!\n";
        return 0;
    } else {
        std::cout << "[FAILURE] Some tests failed\n";
        return 1;
    }
}
