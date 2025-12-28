/// 3D W-Velocity Tests (~5 seconds)
/// Tests the w-velocity component (unique to 3D)
///
/// Tests:
/// 1. W-velocity field storage and indexing
/// 2. W-contribution to divergence
/// 3. Pressure gradient in z-direction
/// 4. W-velocity boundary conditions

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "config.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace nncfd;

//=============================================================================
// TEST 1: W-velocity field storage and indexing
//=============================================================================
bool test_w_storage() {
    std::cout << "Test 1: W-velocity storage and indexing... ";

    Mesh mesh;
    mesh.init_uniform(8, 8, 8, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    VectorField vel(mesh);

    // Set w = i + 10*j + 100*k at each z-face
    for (int k = mesh.k_begin(); k <= mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                vel.w(i, j, k) = static_cast<double>(i + 10 * j + 100 * k);
            }
        }
    }

    // Verify values read back correctly
    double max_error = 0.0;
    for (int k = mesh.k_begin(); k <= mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double expected = static_cast<double>(i + 10 * j + 100 * k);
                double actual = vel.w(i, j, k);
                max_error = std::max(max_error, std::abs(actual - expected));
            }
        }
    }

    bool passed = (max_error < 1e-14);

    if (passed) {
        std::cout << "PASSED\n";
    } else {
        std::cout << "FAILED (max error = " << max_error << ")\n";
    }

    return passed;
}

//=============================================================================
// TEST 2: W-velocity staggering (z-face locations)
//=============================================================================
bool test_w_staggering() {
    std::cout << "Test 2: W-velocity staggering (z-face locations)... ";

    Mesh mesh;
    mesh.init_uniform(4, 4, 4, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    // Verify w is at z-faces (Nz+1 faces for Nz cells)
    // For Nz=4 interior cells, we have 5 z-faces
    // k_begin() to k_end() inclusive should give 5 values

    int num_w_faces = 0;
    for (int k = mesh.k_begin(); k <= mesh.k_end(); ++k) {
        num_w_faces++;
    }

    int expected_faces = mesh.Nz + 1;  // Nz cells have Nz+1 faces

    bool passed = (num_w_faces == expected_faces);

    if (passed) {
        std::cout << "PASSED (w has " << num_w_faces << " z-faces for " << mesh.Nz << " cells)\n";
    } else {
        std::cout << "FAILED\n";
        std::cout << "  Expected " << expected_faces << " z-faces, got " << num_w_faces << "\n";
    }

    return passed;
}

//=============================================================================
// TEST 3: W contribution to divergence
//=============================================================================
bool test_w_divergence_contribution() {
    std::cout << "Test 3: W contribution to divergence... ";

    Mesh mesh;
    mesh.init_uniform(8, 8, 8, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    VectorField vel(mesh);

    // Set u = 0, v = 0, w = z (linear in z)
    // dw/dz = 1, so divergence should be 1 everywhere
    for (int k = mesh.k_begin(); k <= mesh.k_end(); ++k) {
        double z = mesh.zf[k];
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                vel.w(i, j, k) = z;
            }
        }
    }

    // Compute divergence
    double max_error = 0.0;
    double expected_div = 1.0;
    double dz = mesh.dz;

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double dwdz = (vel.w(i, j, k + 1) - vel.w(i, j, k)) / dz;
                // For this test, du/dx = dv/dy = 0
                double div = dwdz;
                max_error = std::max(max_error, std::abs(div - expected_div));
            }
        }
    }

    bool passed = (max_error < 1e-10);

    if (passed) {
        std::cout << "PASSED (max divergence error = " << std::scientific << max_error << ")\n";
    } else {
        std::cout << "FAILED\n";
        std::cout << "  Max divergence error: " << max_error << "\n";
    }

    return passed;
}

//=============================================================================
// TEST 4: Pressure gradient in z-direction affects w
//=============================================================================
bool test_pressure_gradient_z() {
    std::cout << "Test 4: Pressure gradient in z affects w... ";

    Mesh mesh;
    mesh.init_uniform(8, 8, 8, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    Config config;
    config.nu = 0.01;
    config.dt = 0.001;
    config.adaptive_dt = false;
    config.max_iter = 5;
    config.tol = 1e-6;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    // Apply body force in z-direction
    solver.set_body_force(0.0, 0.0, 0.001);

    // Set BCs
    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::Periodic;
    bc.y_hi = VelocityBC::Periodic;
    bc.z_lo = VelocityBC::Periodic;
    bc.z_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    // Run a few timesteps
    for (int step = 0; step < 5; ++step) {
        solver.step();
    }

#ifdef USE_GPU_OFFLOAD
    solver.sync_solution_from_gpu();
#endif

    // W should have become positive due to body force in +z direction
    double mean_w = 0.0;
    int count = 0;

    for (int k = mesh.k_begin(); k <= mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                mean_w += solver.velocity().w(i, j, k);
                count++;
            }
        }
    }
    mean_w /= count;

    bool passed = (mean_w > 0);

    if (passed) {
        std::cout << "PASSED (mean w = " << std::scientific << mean_w << " > 0)\n";
    } else {
        std::cout << "FAILED\n";
        std::cout << "  Mean w = " << mean_w << " (expected > 0 due to +z body force)\n";
    }

    return passed;
}

//=============================================================================
// TEST 5: W-velocity boundary conditions (no-slip and periodic)
//=============================================================================
bool test_w_boundary_conditions() {
    std::cout << "Test 5: W-velocity boundary conditions... ";

    Mesh mesh;
    mesh.init_uniform(8, 8, 8, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    Config config;
    config.nu = 0.01;
    config.dt = 0.001;
    config.adaptive_dt = false;
    config.max_iter = 10;
    config.tol = 1e-6;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);
    solver.set_body_force(0.001, 0.001, 0.001);

    // Set BCs with no-slip on z-boundaries
    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::Periodic;
    bc.y_hi = VelocityBC::Periodic;
    bc.z_lo = VelocityBC::NoSlip;
    bc.z_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);

    // Initialize with non-zero w
    for (int k = mesh.k_begin(); k <= mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                solver.velocity().w(i, j, k) = 0.1;
            }
        }
    }

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    // Run timesteps
    for (int step = 0; step < 10; ++step) {
        solver.step();
    }

#ifdef USE_GPU_OFFLOAD
    solver.sync_solution_from_gpu();
#endif

    // Check w at z-boundaries (should be zero for no-slip)
    double max_w_boundary = 0.0;

    // z_lo boundary
    int k_lo = mesh.k_begin();
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            max_w_boundary = std::max(max_w_boundary, std::abs(solver.velocity().w(i, j, k_lo)));
        }
    }

    // z_hi boundary
    int k_hi = mesh.k_end();
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            max_w_boundary = std::max(max_w_boundary, std::abs(solver.velocity().w(i, j, k_hi)));
        }
    }

    bool passed = (max_w_boundary < 1e-10);

    if (passed) {
        std::cout << "PASSED (max w at walls = " << std::scientific << max_w_boundary << ")\n";
    } else {
        std::cout << "FAILED\n";
        std::cout << "  Max w at no-slip walls: " << max_w_boundary << " (expected ~0)\n";
    }

    return passed;
}

//=============================================================================
// TEST 6: W-velocity cell-center interpolation
//=============================================================================
bool test_w_center_interpolation() {
    std::cout << "Test 6: W-velocity cell-center interpolation... ";

    Mesh mesh;
    mesh.init_uniform(8, 8, 8, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    VectorField vel(mesh);

    // Set w = z at faces
    for (int k = mesh.k_begin(); k <= mesh.k_end(); ++k) {
        double z = mesh.zf[k];
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                vel.w(i, j, k) = z;
            }
        }
    }

    // Cell-center w should be average of top and bottom faces
    double max_error = 0.0;

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        double z_center = mesh.z(k);  // Cell center z-coordinate

        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double w_center = vel.w_center(i, j, k);
                double expected = z_center;  // Since w = z, w at center = z_center

                max_error = std::max(max_error, std::abs(w_center - expected));
            }
        }
    }

    bool passed = (max_error < 1e-10);

    if (passed) {
        std::cout << "PASSED (max interpolation error = " << std::scientific << max_error << ")\n";
    } else {
        std::cout << "FAILED\n";
        std::cout << "  Max interpolation error: " << max_error << "\n";
    }

    return passed;
}

//=============================================================================
// MAIN
//=============================================================================
int main() {
    std::cout << "=== 3D W-Velocity Tests ===\n\n";

    int passed = 0;
    int total = 0;

    total++; if (test_w_storage()) passed++;
    total++; if (test_w_staggering()) passed++;
    total++; if (test_w_divergence_contribution()) passed++;
    total++; if (test_pressure_gradient_z()) passed++;
    total++; if (test_w_boundary_conditions()) passed++;
    total++; if (test_w_center_interpolation()) passed++;

    std::cout << "\n=== Results: " << passed << "/" << total << " tests passed ===\n";

    if (passed == total) {
        std::cout << "[SUCCESS] All w-velocity tests passed!\n";
        return 0;
    } else {
        std::cout << "[FAILURE] Some tests failed\n";
        return 1;
    }
}
