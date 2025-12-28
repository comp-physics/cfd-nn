/// 2D vs 3D Solver Comparison Tests
/// Validates that 3D solver produces correct results by comparing to 2D reference
///
/// Test Strategy:
/// 1. Degenerate case: 3D with Nz=1 should exactly match 2D
/// 2. Z-invariant case: 3D with Nz=8 and z-invariant flow should match 2D at every z-plane

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <vector>

using namespace nncfd;

// Test parameters
constexpr int NX = 32;
constexpr int NY = 32;
constexpr double LX = 2.0;
constexpr double LY = 2.0;
constexpr double LZ = 1.0;
constexpr double NU = 0.01;
constexpr double DP_DX = -0.001;
constexpr int MAX_ITER = 500;

// Helper: compute L2 norm of difference between two arrays
double compute_l2_error(const std::vector<double>& a, const std::vector<double>& b) {
    assert(a.size() == b.size());
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum / a.size());
}

// Helper: compute max absolute difference
double compute_linf_error(const std::vector<double>& a, const std::vector<double>& b) {
    assert(a.size() == b.size());
    double max_diff = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
    }
    return max_diff;
}

// Helper: extract u velocity from 2D solver at interior points
std::vector<double> extract_2d_u(const RANSSolver& solver, const Mesh& mesh) {
    std::vector<double> u_vals;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            u_vals.push_back(solver.velocity().u(i, j));
        }
    }
    return u_vals;
}

// Helper: extract v velocity from 2D solver at interior points
std::vector<double> extract_2d_v(const RANSSolver& solver, const Mesh& mesh) {
    std::vector<double> v_vals;
    for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            v_vals.push_back(solver.velocity().v(i, j));
        }
    }
    return v_vals;
}

// Helper: extract u velocity from 3D solver at z=k_mid slice
std::vector<double> extract_3d_u_slice(const RANSSolver& solver, const Mesh& mesh, int k) {
    std::vector<double> u_vals;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            u_vals.push_back(solver.velocity().u(i, j, k));
        }
    }
    return u_vals;
}

// Helper: extract v velocity from 3D solver at z=k_mid slice
std::vector<double> extract_3d_v_slice(const RANSSolver& solver, const Mesh& mesh, int k) {
    std::vector<double> v_vals;
    for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            v_vals.push_back(solver.velocity().v(i, j, k));
        }
    }
    return v_vals;
}

// Helper: compute max divergence for 2D solver
double compute_max_div_2d(const RANSSolver& solver, const Mesh& mesh) {
    double max_div = 0.0;
    double dx = mesh.dx;
    double dy = mesh.dy;

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double div = (solver.velocity().u(i+1, j) - solver.velocity().u(i, j)) / dx
                       + (solver.velocity().v(i, j+1) - solver.velocity().v(i, j)) / dy;
            max_div = std::max(max_div, std::abs(div));
        }
    }
    return max_div;
}

// Helper: compute max divergence for 3D solver (at specific k-plane for degenerate case)
double compute_max_div_3d_slice(const RANSSolver& solver, const Mesh& mesh, int k) {
    double max_div = 0.0;
    double dx = mesh.dx;
    double dy = mesh.dy;
    // For 2D-like divergence at specific k, only use du/dx and dv/dy
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double div = (solver.velocity().u(i+1, j, k) - solver.velocity().u(i, j, k)) / dx
                       + (solver.velocity().v(i, j+1, k) - solver.velocity().v(i, j, k)) / dy;
            max_div = std::max(max_div, std::abs(div));
        }
    }
    return max_div;
}

// Helper: compute max divergence for 3D solver (full 3D)
double compute_max_div_3d(const RANSSolver& solver, const Mesh& mesh) {
    double max_div = 0.0;
    double dx = mesh.dx;
    double dy = mesh.dy;
    double dz = mesh.dz;

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double div = (solver.velocity().u(i+1, j, k) - solver.velocity().u(i, j, k)) / dx
                           + (solver.velocity().v(i, j+1, k) - solver.velocity().v(i, j, k)) / dy
                           + (solver.velocity().w(i, j, k+1) - solver.velocity().w(i, j, k)) / dz;
                max_div = std::max(max_div, std::abs(div));
            }
        }
    }
    return max_div;
}

// Helper: initialize Poiseuille profile for 2D
void init_poiseuille_2d(RANSSolver& solver, const Mesh& mesh, double dp_dx, double nu) {
    double H = LY / 2.0;
    double y_mid = LY / 2.0;

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        double y = mesh.y(j) - y_mid;  // Shift so y=0 at center
        double u_analytical = -dp_dx / (2.0 * nu) * (H * H - y * y);

        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            solver.velocity().u(i, j) = 0.9 * u_analytical;
        }
    }

    for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            solver.velocity().v(i, j) = 0.0;
        }
    }
}

// Helper: initialize Poiseuille profile for 3D (z-invariant)
void init_poiseuille_3d(RANSSolver& solver, const Mesh& mesh, double dp_dx, double nu) {
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
}

//=============================================================================
// TEST 1: Degenerate case - 3D with Nz=1 should exactly match 2D
//=============================================================================
bool test_degenerate_nz1() {
    std::cout << "Test 1: Degenerate case (Nz=1 vs 2D)...\n";

    // ---- Run 2D solver ----
    std::cout << "  Running 2D solver... ";
    Mesh mesh_2d;
    mesh_2d.init_uniform(NX, NY, 0.0, LX, 0.0, LY);

    Config config_2d;
    config_2d.nu = NU;
    config_2d.dp_dx = DP_DX;
    config_2d.adaptive_dt = true;
    config_2d.max_iter = MAX_ITER;
    config_2d.tol = 1e-6;
    config_2d.turb_model = TurbulenceModelType::None;
    config_2d.verbose = false;

    RANSSolver solver_2d(mesh_2d, config_2d);
    solver_2d.set_body_force(-config_2d.dp_dx, 0.0);
    init_poiseuille_2d(solver_2d, mesh_2d, config_2d.dp_dx, config_2d.nu);

#ifdef USE_GPU_OFFLOAD
    solver_2d.sync_to_gpu();  // Upload initial conditions to GPU
#endif

    auto [res_2d, iter_2d] = solver_2d.solve_steady();
    std::cout << "done (iters=" << iter_2d << ", res=" << std::scientific << res_2d << ")\n";

    // ---- Run 3D solver with Nz=1 ----
    std::cout << "  Running 3D solver (Nz=1)... ";
    Mesh mesh_3d;
    mesh_3d.init_uniform(NX, NY, 1, 0.0, LX, 0.0, LY, 0.0, LZ);

    Config config_3d;
    config_3d.nu = NU;
    config_3d.dp_dx = DP_DX;
    config_3d.adaptive_dt = true;
    config_3d.max_iter = MAX_ITER;
    config_3d.tol = 1e-6;
    config_3d.turb_model = TurbulenceModelType::None;
    config_3d.verbose = false;

    RANSSolver solver_3d(mesh_3d, config_3d);
    solver_3d.set_body_force(-config_3d.dp_dx, 0.0, 0.0);
    // For Nz=1 (degenerate case), is2D() returns true so 2D code paths are used
    // 2D accessors u(i,j) map to k=0 plane for backward compatibility
    // We must use 2D initialization which writes to k=0 via 2D accessors
    init_poiseuille_2d(solver_3d, mesh_3d, config_3d.dp_dx, config_3d.nu);

#ifdef USE_GPU_OFFLOAD
    solver_3d.sync_to_gpu();  // Upload initial conditions to GPU
#endif

    auto [res_3d, iter_3d] = solver_3d.solve_steady();
    std::cout << "done (iters=" << iter_3d << ", res=" << std::scientific << res_3d << ")\n";

    // ---- Compare results ----
    std::cout << "  Comparing results...\n";

    auto u_2d = extract_2d_u(solver_2d, mesh_2d);
    auto v_2d = extract_2d_v(solver_2d, mesh_2d);
    // For Nz=1, 2D accessors use k=0 plane (by design for backward compatibility)
    const int k_slice = 0;
    auto u_3d = extract_3d_u_slice(solver_3d, mesh_3d, k_slice);
    auto v_3d = extract_3d_v_slice(solver_3d, mesh_3d, k_slice);

    double u_l2_err = compute_l2_error(u_2d, u_3d);
    double u_linf_err = compute_linf_error(u_2d, u_3d);
    double v_l2_err = compute_l2_error(v_2d, v_3d);
    double v_linf_err = compute_linf_error(v_2d, v_3d);

    double div_2d = compute_max_div_2d(solver_2d, mesh_2d);
    // For Nz=1, use slice version at k=0 to match 2D behavior
    double div_3d = mesh_3d.is2D() ? compute_max_div_3d_slice(solver_3d, mesh_3d, 0)
                                   : compute_max_div_3d(solver_3d, mesh_3d);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "    u L2 error:   " << std::scientific << u_l2_err << "\n";
    std::cout << "    u Linf error: " << std::scientific << u_linf_err << "\n";
    std::cout << "    v L2 error:   " << std::scientific << v_l2_err << "\n";
    std::cout << "    v Linf error: " << std::scientific << v_linf_err << "\n";
    std::cout << "    2D max div:   " << std::scientific << div_2d << "\n";
    std::cout << "    3D max div:   " << std::scientific << div_3d << "\n";
    std::cout << "    Iter diff:    " << std::abs(iter_2d - iter_3d) << " ("
              << std::fixed << std::setprecision(1)
              << 100.0 * std::abs(iter_2d - iter_3d) / std::max(iter_2d, iter_3d) << "%)\n";

    // Check pass criteria
    bool passed = true;
    if (u_l2_err > 1e-8) {
        std::cout << "  FAILED: u L2 error too large (> 1e-8)\n";
        passed = false;
    }
    if (v_l2_err > 1e-8) {
        std::cout << "  FAILED: v L2 error too large (> 1e-8)\n";
        passed = false;
    }
    if (div_3d > 1e-8) {
        std::cout << "  FAILED: 3D divergence too large (> 1e-8)\n";
        passed = false;
    }

    if (passed) {
        std::cout << "  PASSED\n";
    }
    return passed;
}

//=============================================================================
// TEST 2: Z-invariant flow - 3D with Nz=4 should match 2D at every z-plane
//=============================================================================
bool test_z_invariant_poiseuille() {
    std::cout << "\nTest 2: Z-invariant Poiseuille (Nz=4 vs 2D)...\n";

    constexpr int NZ = 4;  // Use moderate z resolution to test 3D properly

    // ---- Run 2D solver ----
    std::cout << "  Running 2D solver... ";
    Mesh mesh_2d;
    mesh_2d.init_uniform(NX, NY, 0.0, LX, 0.0, LY);

    Config config_2d;
    config_2d.nu = NU;
    config_2d.dp_dx = DP_DX;
    config_2d.adaptive_dt = true;
    config_2d.max_iter = MAX_ITER;
    config_2d.tol = 1e-6;
    config_2d.turb_model = TurbulenceModelType::None;
    config_2d.verbose = false;

    RANSSolver solver_2d(mesh_2d, config_2d);
    solver_2d.set_body_force(-config_2d.dp_dx, 0.0);
    init_poiseuille_2d(solver_2d, mesh_2d, config_2d.dp_dx, config_2d.nu);

#ifdef USE_GPU_OFFLOAD
    solver_2d.sync_to_gpu();  // Upload initial conditions to GPU
#endif

    auto [res_2d, iter_2d] = solver_2d.solve_steady();
    std::cout << "done (iters=" << iter_2d << ", res=" << std::scientific << res_2d << ")\n";

    // ---- Run 3D solver with Nz=8 ----
    std::cout << "  Running 3D solver (Nz=" << NZ << ")... ";
    Mesh mesh_3d;
    mesh_3d.init_uniform(NX, NY, NZ, 0.0, LX, 0.0, LY, 0.0, LZ);

    Config config_3d;
    config_3d.nu = NU;
    config_3d.dp_dx = DP_DX;
    config_3d.adaptive_dt = true;
    config_3d.max_iter = MAX_ITER;
    config_3d.tol = 1e-6;
    config_3d.turb_model = TurbulenceModelType::None;
    config_3d.verbose = false;

    RANSSolver solver_3d(mesh_3d, config_3d);
    solver_3d.set_body_force(-config_3d.dp_dx, 0.0, 0.0);
    init_poiseuille_3d(solver_3d, mesh_3d, config_3d.dp_dx, config_3d.nu);

#ifdef USE_GPU_OFFLOAD
    solver_3d.sync_to_gpu();
#endif

    auto [res_3d, iter_3d] = solver_3d.solve_steady();
    std::cout << "done (iters=" << iter_3d << ", res=" << std::scientific << res_3d << ")\n";

    // ---- Compare each z-plane to 2D ----
    std::cout << "  Comparing z-planes to 2D reference...\n";

    auto u_2d = extract_2d_u(solver_2d, mesh_2d);
    auto v_2d = extract_2d_v(solver_2d, mesh_2d);

    double max_u_err = 0.0;
    double max_v_err = 0.0;

    for (int k = mesh_3d.k_begin(); k < mesh_3d.k_end(); ++k) {
        auto u_3d = extract_3d_u_slice(solver_3d, mesh_3d, k);
        auto v_3d = extract_3d_v_slice(solver_3d, mesh_3d, k);

        double u_err = compute_l2_error(u_2d, u_3d);
        double v_err = compute_l2_error(v_2d, v_3d);

        max_u_err = std::max(max_u_err, u_err);
        max_v_err = std::max(max_v_err, v_err);

        std::cout << "    z-plane " << (k - mesh_3d.k_begin()) << ": u_err="
                  << std::scientific << u_err << ", v_err=" << v_err << "\n";
    }

    double div_3d = compute_max_div_3d(solver_3d, mesh_3d);

    std::cout << "    Max u error across all planes: " << std::scientific << max_u_err << "\n";
    std::cout << "    Max v error across all planes: " << std::scientific << max_v_err << "\n";
    std::cout << "    3D max divergence: " << std::scientific << div_3d << "\n";

    // Check z-invariance: all z-planes should be identical
    std::cout << "  Checking z-invariance (all planes should be identical)...\n";

    auto u_plane0 = extract_3d_u_slice(solver_3d, mesh_3d, mesh_3d.k_begin());
    double max_z_variation = 0.0;

    for (int k = mesh_3d.k_begin() + 1; k < mesh_3d.k_end(); ++k) {
        auto u_plane_k = extract_3d_u_slice(solver_3d, mesh_3d, k);
        double z_err = compute_linf_error(u_plane0, u_plane_k);
        max_z_variation = std::max(max_z_variation, z_err);
    }
    std::cout << "    Max z-variation: " << std::scientific << max_z_variation << "\n";

    // Check pass criteria
    // Tolerances are realistic for time-stepping CFD to steady state:
    // - u error < 1e-3 (within 2% of max velocity ~ 0.05)
    // - divergence < 1e-4 (reasonably incompressible)
    // - z-variation < 1e-5 (z-invariance preserved)
    bool passed = true;
    if (max_u_err > 1e-3) {
        std::cout << "  FAILED: u error too large (> 1e-3)\n";
        passed = false;
    }
    if (max_v_err > 1e-6) {
        std::cout << "  FAILED: v error too large (> 1e-6)\n";
        passed = false;
    }
    if (div_3d > 1e-4) {
        std::cout << "  FAILED: 3D divergence too large (> 1e-4)\n";
        passed = false;
    }
    if (max_z_variation > 1e-5) {
        std::cout << "  FAILED: z-variation too large (> 1e-5)\n";
        passed = false;
    }

    if (passed) {
        std::cout << "  PASSED\n";
    }
    return passed;
}

//=============================================================================
// TEST 3: Verify w stays zero for z-invariant flow
//=============================================================================
bool test_w_stays_zero() {
    std::cout << "\nTest 3: Verify w velocity stays zero for z-invariant flow...\n";

    constexpr int NZ = 4;

    Mesh mesh;
    mesh.init_uniform(NX, NY, NZ, 0.0, LX, 0.0, LY, 0.0, LZ);

    Config config;
    config.nu = NU;
    config.dp_dx = DP_DX;
    config.adaptive_dt = true;
    config.max_iter = MAX_ITER;
    config.tol = 1e-6;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);
    solver.set_body_force(-config.dp_dx, 0.0, 0.0);
    init_poiseuille_3d(solver, mesh, config.dp_dx, config.nu);

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    std::cout << "  Running 3D solver (Nz=" << NZ << ")... ";
    auto [res, iter] = solver.solve_steady();
    std::cout << "done (iters=" << iter << ", res=" << std::scientific << res << ")\n";

    // Check max |w| and max |u|
    double max_w = 0.0;
    double max_u = 0.0;
    for (int k = mesh.k_begin(); k <= mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                max_u = std::max(max_u, std::abs(solver.velocity().u(i, j, k)));
            }
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                max_w = std::max(max_w, std::abs(solver.velocity().w(i, j, k)));
            }
        }
    }

    std::cout << "  Max |u|: " << std::scientific << max_u << "\n";
    std::cout << "  Max |w|: " << std::scientific << max_w << "\n";

    bool passed = (max_w < 1e-10);
    if (passed) {
        std::cout << "  PASSED\n";
    } else {
        std::cout << "  FAILED: w should be ~0 for z-invariant flow\n";
    }
    return passed;
}

//=============================================================================
// MAIN
//=============================================================================
int main() {
    std::cout << "=== 2D vs 3D Solver Comparison Tests ===\n\n";

    int passed = 0;
    int total = 0;

    total++; if (test_degenerate_nz1()) passed++;
    total++; if (test_z_invariant_poiseuille()) passed++;
    total++; if (test_w_stays_zero()) passed++;

    std::cout << "\n=== Results: " << passed << "/" << total << " tests passed ===\n";

    if (passed == total) {
        std::cout << "[SUCCESS] 3D solver matches 2D reference!\n";
        return 0;
    } else {
        std::cout << "[FAILURE] 3D solver does not match 2D\n";
        return 1;
    }
}
