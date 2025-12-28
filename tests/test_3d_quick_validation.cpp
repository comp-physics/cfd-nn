/// Fast 3D validation tests (~5 seconds total)
/// Quick smoke tests that verify basic 3D functionality
///
/// Tests:
/// 1. Divergence-free after projection (1s)
/// 2. Z-invariant flow preservation (2s)
/// 3. Degenerate 3D (Nz=1) matches 2D behavior (2s)

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "config.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

using namespace nncfd;

//=============================================================================
// Helper functions
//=============================================================================

double compute_max_divergence_3d(const VectorField& vel, const Mesh& mesh) {
    double max_div = 0.0;
    double dx = mesh.dx, dy = mesh.dy, dz = mesh.dz;

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double dudx = (vel.u(i+1, j, k) - vel.u(i, j, k)) / dx;
                double dvdy = (vel.v(i, j+1, k) - vel.v(i, j, k)) / dy;
                double dwdz = (vel.w(i, j, k+1) - vel.w(i, j, k)) / dz;
                double div = dudx + dvdy + dwdz;
                max_div = std::max(max_div, std::abs(div));
            }
        }
    }
    return max_div;
}

// Extract u-velocity at a specific z-plane
std::vector<double> extract_u_plane(const VectorField& vel, const Mesh& mesh, int k) {
    std::vector<double> u_vals;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            u_vals.push_back(vel.u(i, j, k));
        }
    }
    return u_vals;
}

double compute_max_diff(const std::vector<double>& a, const std::vector<double>& b) {
    double max_diff = 0.0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
    }
    return max_diff;
}

//=============================================================================
// TEST 1: Divergence-free after projection
//=============================================================================
bool test_divergence_free() {
    std::cout << "Test 1: Divergence-free after projection... ";

    // Small 3D grid, run to steady state
    Mesh mesh;
    mesh.init_uniform(16, 16, 4, 0.0, 1.0, 0.0, 1.0, 0.0, 0.5);

    Config config;
    config.nu = 0.01;
    config.adaptive_dt = true;
    config.max_iter = 50;  // Enough iterations to approach steady state
    config.tol = 1e-6;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);
    solver.set_body_force(0.001, 0.0, 0.0);

    // Set BCs for channel flow
    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    bc.z_lo = VelocityBC::Periodic;
    bc.z_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    // Initialize with Poiseuille-like profile (nearly divergence-free from start)
    double H = 0.5;  // half channel height
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            double y = mesh.y(j) - H;
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                solver.velocity().u(i, j, k) = 0.01 * (H * H - y * y);
            }
        }
    }

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    // Run to steady state
    [[maybe_unused]] auto [res, iters] = solver.solve_steady();

#ifdef USE_GPU_OFFLOAD
    solver.sync_solution_from_gpu();
#endif

    double max_div_after = compute_max_divergence_3d(solver.velocity(), mesh);

    // Check divergence is small (Poisson solver tolerance ~1e-6 produces div ~1e-4)
    bool passed = (max_div_after < 1e-3);

    if (passed) {
        std::cout << "PASSED (div=" << std::scientific << max_div_after << ")\n";
    } else {
        std::cout << "FAILED\n";
        std::cout << "  Divergence after " << iters << " iterations: " << max_div_after << " (expected < 1e-3)\n";
    }

    return passed;
}

//=============================================================================
// TEST 2: Z-invariant flow stays z-invariant
//=============================================================================
bool test_z_invariant_preservation() {
    std::cout << "Test 2: Z-invariant flow preservation... ";

    // 3D grid with 8 z-planes
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

    // Set BCs: periodic in x and z, no-slip in y
    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    bc.z_lo = VelocityBC::Periodic;
    bc.z_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    // Initialize with z-invariant Poiseuille-like profile
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            double y = mesh.y(j) - 0.5;  // center at y=0.5
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                solver.velocity().u(i, j, k) = 0.01 * (0.25 - y * y);
            }
        }
    }

    // v = 0, w = 0 everywhere (already default)

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    // Run 10 timesteps
    for (int step = 0; step < 10; ++step) {
        solver.step();
    }

#ifdef USE_GPU_OFFLOAD
    solver.sync_solution_from_gpu();
#endif

    // Compare all z-planes to first z-plane
    auto u_plane0 = extract_u_plane(solver.velocity(), mesh, mesh.k_begin());
    double max_z_variation = 0.0;

    for (int k = mesh.k_begin() + 1; k < mesh.k_end(); ++k) {
        auto u_plane_k = extract_u_plane(solver.velocity(), mesh, k);
        double diff = compute_max_diff(u_plane0, u_plane_k);
        max_z_variation = std::max(max_z_variation, diff);
    }

    // All z-planes should be identical within numerical precision
    // Allow some tolerance due to iterative solver and floating point accumulation
    bool passed = (max_z_variation < 1e-4);

    if (passed) {
        std::cout << "PASSED (max z-variation=" << std::scientific << max_z_variation << ")\n";
    } else {
        std::cout << "FAILED\n";
        std::cout << "  Max z-variation: " << max_z_variation << " (expected < 1e-4)\n";
    }

    return passed;
}

//=============================================================================
// TEST 3: Degenerate 3D (Nz=1) matches 2D behavior
//=============================================================================
bool test_degenerate_3d() {
    std::cout << "Test 3: Degenerate 3D (Nz=1) matches 2D... ";

    const int NX = 16, NY = 16;
    const double LX = 1.0, LY = 1.0;

    // --- Run 2D solver ---
    Mesh mesh_2d;
    mesh_2d.init_uniform(NX, NY, 0.0, LX, 0.0, LY);

    Config config;
    config.nu = 0.01;
    config.dt = 0.001;
    config.adaptive_dt = false;
    config.max_iter = 20;
    config.tol = 1e-6;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver_2d(mesh_2d, config);
    solver_2d.set_body_force(0.001, 0.0);

    // Initialize with simple profile
    for (int j = mesh_2d.j_begin(); j < mesh_2d.j_end(); ++j) {
        double y = mesh_2d.y(j) - 0.5;
        for (int i = mesh_2d.i_begin(); i <= mesh_2d.i_end(); ++i) {
            solver_2d.velocity().u(i, j) = 0.01 * (0.25 - y * y);
        }
    }

#ifdef USE_GPU_OFFLOAD
    solver_2d.sync_to_gpu();
#endif

    for (int step = 0; step < 20; ++step) {
        solver_2d.step();
    }

#ifdef USE_GPU_OFFLOAD
    solver_2d.sync_solution_from_gpu();
#endif

    // --- Run 3D solver with Nz=1 (degenerate case) ---
    Mesh mesh_3d;
    mesh_3d.init_uniform(NX, NY, 1, 0.0, LX, 0.0, LY, 0.0, 0.1);

    RANSSolver solver_3d(mesh_3d, config);
    solver_3d.set_body_force(0.001, 0.0, 0.0);

    // Initialize with same profile (use 2D accessors for Nz=1 which is treated as 2D)
    for (int j = mesh_3d.j_begin(); j < mesh_3d.j_end(); ++j) {
        double y = mesh_3d.y(j) - 0.5;
        for (int i = mesh_3d.i_begin(); i <= mesh_3d.i_end(); ++i) {
            solver_3d.velocity().u(i, j) = 0.01 * (0.25 - y * y);
        }
    }

#ifdef USE_GPU_OFFLOAD
    solver_3d.sync_to_gpu();
#endif

    for (int step = 0; step < 20; ++step) {
        solver_3d.step();
    }

#ifdef USE_GPU_OFFLOAD
    solver_3d.sync_solution_from_gpu();
#endif

    // Compare results
    double max_u_diff = 0.0;
    for (int j = mesh_2d.j_begin(); j < mesh_2d.j_end(); ++j) {
        for (int i = mesh_2d.i_begin(); i <= mesh_2d.i_end(); ++i) {
            double u_2d = solver_2d.velocity().u(i, j);
            double u_3d = solver_3d.velocity().u(i, j);  // 2D accessor for Nz=1
            max_u_diff = std::max(max_u_diff, std::abs(u_2d - u_3d));
        }
    }

    // Should match exactly (or very close) since Nz=1 uses 2D code paths
    bool passed = (max_u_diff < 1e-12);

    if (passed) {
        std::cout << "PASSED (max diff=" << std::scientific << max_u_diff << ")\n";
    } else {
        std::cout << "FAILED\n";
        std::cout << "  Max u difference: " << max_u_diff << " (expected < 1e-12)\n";
    }

    return passed;
}

//=============================================================================
// MAIN
//=============================================================================
int main() {
    std::cout << "=== Fast 3D Validation Tests ===\n\n";

    int passed = 0;
    int total = 0;

    total++; if (test_divergence_free()) passed++;
    total++; if (test_z_invariant_preservation()) passed++;
    total++; if (test_degenerate_3d()) passed++;

    std::cout << "\n=== Results: " << passed << "/" << total << " tests passed ===\n";

    if (passed == total) {
        std::cout << "[SUCCESS] All quick 3D validation tests passed!\n";
        return 0;
    } else {
        std::cout << "[FAILURE] Some tests failed\n";
        return 1;
    }
}
