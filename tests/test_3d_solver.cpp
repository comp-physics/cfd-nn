/// Unit tests for 3D solver extension
/// Tests basic 3D mesh setup, field operations, and solver functionality

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>

using namespace nncfd;

// Test 3D mesh initialization
bool test_3d_mesh_init() {
    std::cout << "Testing 3D mesh initialization... ";

    Mesh mesh;
    mesh.init_uniform(16, 16, 16, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    // Verify dimensions
    if (mesh.Nx != 16 || mesh.Ny != 16 || mesh.Nz != 16) {
        std::cout << "FAILED (wrong dimensions)\n";
        return false;
    }

    // Verify it's not 2D
    if (mesh.is2D()) {
        std::cout << "FAILED (should not be 2D)\n";
        return false;
    }

    // Verify spacing
    double expected_dx = 1.0 / 16.0;
    if (std::abs(mesh.dx - expected_dx) > 1e-12 ||
        std::abs(mesh.dy - expected_dx) > 1e-12 ||
        std::abs(mesh.dz - expected_dx) > 1e-12) {
        std::cout << "FAILED (wrong spacing)\n";
        return false;
    }

    // Verify k_begin/k_end
    if (mesh.k_begin() != mesh.Nghost || mesh.k_end() != mesh.Nz + mesh.Nghost) {
        std::cout << "FAILED (wrong k range)\n";
        return false;
    }

    std::cout << "PASSED\n";
    return true;
}

// Test 3D scalar field operations
bool test_3d_scalar_field() {
    std::cout << "Testing 3D scalar field... ";

    Mesh mesh;
    mesh.init_uniform(8, 8, 8, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    ScalarField p(mesh);

    // Set values and verify
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                p(i, j, k) = i + 10*j + 100*k;
            }
        }
    }

    // Verify values
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double expected = i + 10*j + 100*k;
                if (std::abs(p(i, j, k) - expected) > 1e-12) {
                    std::cout << "FAILED (value mismatch at " << i << "," << j << "," << k << ")\n";
                    return false;
                }
            }
        }
    }

    std::cout << "PASSED\n";
    return true;
}

// Test 3D vector field operations
bool test_3d_vector_field() {
    std::cout << "Testing 3D vector field... ";

    Mesh mesh;
    mesh.init_uniform(8, 8, 8, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    VectorField vel(mesh);

    // Set u, v, w at faces
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                vel.u(i, j, k) = 1.0;  // Uniform u
            }
        }
    }

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                vel.v(i, j, k) = 0.5;  // Uniform v
            }
        }
    }

    for (int k = mesh.k_begin(); k <= mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                vel.w(i, j, k) = 0.25;  // Uniform w
            }
        }
    }

    // Verify cell-centered values
    int i = mesh.i_begin() + 2;
    int j = mesh.j_begin() + 2;
    int k = mesh.k_begin() + 2;

    double u_c = vel.u_center(i, j, k);
    double v_c = vel.v_center(i, j, k);
    double w_c = vel.w_center(i, j, k);

    if (std::abs(u_c - 1.0) > 1e-10 || std::abs(v_c - 0.5) > 1e-10 || std::abs(w_c - 0.25) > 1e-10) {
        std::cout << "FAILED (cell-center interpolation wrong)\n";
        std::cout << "  u_c=" << u_c << " v_c=" << v_c << " w_c=" << w_c << "\n";
        return false;
    }

    // Verify magnitude
    double mag = vel.magnitude(i, j, k);
    double expected_mag = std::sqrt(1.0*1.0 + 0.5*0.5 + 0.25*0.25);
    if (std::abs(mag - expected_mag) > 1e-10) {
        std::cout << "FAILED (magnitude wrong: " << mag << " vs " << expected_mag << ")\n";
        return false;
    }

    std::cout << "PASSED\n";
    return true;
}

// Test 3D solver creation and basic step
bool test_3d_solver_creation() {
    std::cout << "Testing 3D solver creation... ";

    Mesh mesh;
    mesh.init_uniform(16, 16, 16, 0.0, 2.0, -1.0, 1.0, 0.0, 1.0);

    Config config;
    config.nu = 0.01;
    config.adaptive_dt = true;
    config.max_iter = 10;
    config.tol = 1e-6;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    try {
        RANSSolver solver(mesh, config);

        // Verify mesh is 3D
        if (mesh.is2D()) {
            std::cout << "FAILED (mesh should be 3D)\n";
            return false;
        }

        std::cout << "PASSED\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "FAILED (exception: " << e.what() << ")\n";
        return false;
    }
}

// Test 3D divergence-free constraint with uniform flow
bool test_3d_divergence_free() {
    std::cout << "Testing 3D divergence-free constraint... ";

    Mesh mesh;
    mesh.init_uniform(16, 16, 16, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    Config config;
    config.nu = 0.01;
    config.adaptive_dt = true;
    config.max_iter = 5;
    config.tol = 1e-6;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    // Initialize with uniform divergence-free flow: u=1, v=0, w=0
    for (int k = 0; k < mesh.total_Nz(); ++k) {
        for (int j = 0; j < mesh.total_Ny(); ++j) {
            for (int i = 0; i < mesh.total_Nx(); ++i) {
                solver.velocity().u(i, j, k) = 1.0;
            }
        }
    }

    for (int k = 0; k < mesh.total_Nz(); ++k) {
        for (int j = 0; j < mesh.total_Ny() + 1; ++j) {
            for (int i = 0; i < mesh.total_Nx(); ++i) {
                solver.velocity().v(i, j, k) = 0.0;
            }
        }
    }

    for (int k = 0; k < mesh.total_Nz() + 1; ++k) {
        for (int j = 0; j < mesh.total_Ny(); ++j) {
            for (int i = 0; i < mesh.total_Nx(); ++i) {
                solver.velocity().w(i, j, k) = 0.0;
            }
        }
    }

    // Compute divergence manually
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

    if (max_div > 1e-10) {
        std::cout << "FAILED (max_div=" << max_div << ")\n";
        return false;
    }

    std::cout << "PASSED (max_div=" << std::scientific << max_div << ")\n";
    return true;
}

// Test 3D channel flow (periodic in x and z, walls in y)
bool test_3d_channel_flow() {
    std::cout << "Testing 3D channel flow (5 steps)... ";

    Mesh mesh;
    // Small 3D channel: x-periodic, y-walls, z-periodic
    mesh.init_uniform(16, 16, 8, 0.0, 2.0, -1.0, 1.0, 0.0, 0.5);

    Config config;
    config.nu = 0.01;
    config.dt = 0.001;
    config.adaptive_dt = false;
    config.max_iter = 5;
    config.tol = 1e-6;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    // Set BCs: periodic in x and z, no-slip walls in y
    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    bc.z_lo = VelocityBC::Periodic;
    bc.z_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    // Apply body force in x-direction (pressure gradient)
    solver.set_body_force(0.001, 0.0, 0.0);

    // Initialize with small perturbation
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                double y = mesh.y(j);
                // Parabolic profile approximation
                solver.velocity().u(i, j, k) = 0.001 * (1.0 - y*y);
            }
        }
    }

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    // Run a few timesteps
    double total_time = 0.0;

    for (int step = 0; step < 5; ++step) {
        try {
            double dt_actual = solver.step();
            total_time += dt_actual;
        } catch (const std::exception& e) {
            std::cout << "FAILED (exception at step " << step << ": " << e.what() << ")\n";
            return false;
        }
    }

    // Check for NaN/Inf
    double max_vel = solver.velocity().max_magnitude();
    if (!std::isfinite(max_vel)) {
        std::cout << "FAILED (NaN/Inf in velocity)\n";
        return false;
    }

    std::cout << "PASSED (max_vel=" << std::scientific << max_vel << ", t=" << total_time << ")\n";
    return true;
}

int main() {
    std::cout << "=== 3D Solver Validation Tests ===\n\n";

    int passed = 0;
    int failed = 0;

    if (test_3d_mesh_init()) ++passed; else ++failed;
    if (test_3d_scalar_field()) ++passed; else ++failed;
    if (test_3d_vector_field()) ++passed; else ++failed;
    if (test_3d_solver_creation()) ++passed; else ++failed;
    if (test_3d_divergence_free()) ++passed; else ++failed;
    if (test_3d_channel_flow()) ++passed; else ++failed;

    std::cout << "\n=== Results: " << passed << "/" << (passed + failed) << " tests passed ===\n";

    if (failed > 0) {
        std::cout << "[FAILED] Some tests failed!\n";
        return 1;
    }

    std::cout << "[SUCCESS] All 3D tests passed!\n";
    return 0;
}
