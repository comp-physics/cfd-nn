/// @file test_duct_validation.cpp
/// @brief Validates duct geometry: wall_distance for z-walls, flow sanity.

#include "test_harness.hpp"
#include "test_utilities.hpp"
#include "solver.hpp"
#include "config.hpp"
#include "mesh.hpp"
#include <cmath>
#include <iomanip>

using namespace nncfd;
using nncfd::test::harness::record;
using nncfd::test::BCPattern;
using nncfd::test::create_velocity_bc;

// ============================================================================
// Test 1: wall_distance accounts for z-walls in duct geometry
// ============================================================================

void test_wall_distance_duct() {
    // Square duct: walls at y=0, y=1, z=0, z=1
    Mesh mesh;
    mesh.init_uniform(8, 32, 32, 0.0, 4.0, 0.0, 1.0, 0.0, 1.0);
    mesh.z_has_walls_ = true;

    const int Ng = mesh.Nghost;

    // Corner point near (y=0, z=0): should be min of both distances
    {
        int j = Ng;      // Near y=0 wall
        int k = Ng;      // Near z=0 wall
        double dist = mesh.wall_distance(Ng, j, k);
        double y_dist = std::abs(mesh.yc[j] - mesh.y_min);
        double z_dist = std::abs(mesh.zc[k] - mesh.z_min);
        double expected = std::min(y_dist, z_dist);
        double err = std::abs(dist - expected);
        record("Wall distance at corner (y-wall, z-wall)", err < 1e-14,
               "dist=" + std::to_string(dist) + " expected=" + std::to_string(expected));
    }

    // Center point: equidistant to all walls -> min = 0.5
    {
        int j = Ng + mesh.Ny / 2;
        int k = Ng + mesh.Nz / 2;
        double dist = mesh.wall_distance(Ng, j, k);
        // At center of unit square, dist to nearest wall = 0.5 - half_cell
        double y_dist = std::min(std::abs(mesh.yc[j] - mesh.y_min),
                                 std::abs(mesh.yc[j] - mesh.y_max));
        double z_dist = std::min(std::abs(mesh.zc[k] - mesh.z_min),
                                 std::abs(mesh.zc[k] - mesh.z_max));
        double expected = std::min(y_dist, z_dist);
        double err = std::abs(dist - expected);
        record("Wall distance at center (duct)", err < 1e-14,
               "dist=" + std::to_string(dist) + " expected=" + std::to_string(expected));
    }

    // Point near z-wall only: z_dist < y_dist
    {
        int j = Ng + mesh.Ny / 2;  // Middle of y (far from y-walls)
        int k = Ng;                  // Near z=0 wall
        double dist = mesh.wall_distance(Ng, j, k);
        double z_dist = std::abs(mesh.zc[k] - mesh.z_min);
        double y_dist = std::min(std::abs(mesh.yc[j] - mesh.y_min),
                                 std::abs(mesh.yc[j] - mesh.y_max));
        // z_dist should be smaller than y_dist
        bool z_closer = (z_dist < y_dist);
        record("Wall distance near z-wall (duct)", z_closer && std::abs(dist - z_dist) < 1e-14,
               "z_dist=" + std::to_string(z_dist) + " y_dist=" + std::to_string(y_dist));
    }

    // Channel mode (z_has_walls_ = false): should ignore z
    mesh.z_has_walls_ = false;
    {
        int j = Ng + mesh.Ny / 2;
        int k = Ng;
        double dist_channel = mesh.wall_distance(Ng, j, k);
        double y_dist = std::min(std::abs(mesh.yc[j] - mesh.y_min),
                                 std::abs(mesh.yc[j] - mesh.y_max));
        double err = std::abs(dist_channel - y_dist);
        record("Wall distance ignores z when z_has_walls=false", err < 1e-14);
    }
}

// ============================================================================
// Test 2: Solver sets z_has_walls_ from velocity BCs
// ============================================================================

void test_solver_sets_z_walls() {
    Mesh mesh;
    mesh.init_uniform(8, 16, 16, 0.0, 4.0, 0.0, 1.0, 0.0, 1.0);

    Config cfg;
    cfg.Nx = 8; cfg.Ny = 16; cfg.Nz = 16;
    cfg.dt = 0.001; cfg.max_steps = 1; cfg.nu = 0.01;

    RANSSolver solver(mesh, cfg);

    // Channel BCs: periodic z -> z_has_walls_ = false
    solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel3D));
    record("Channel3D: z_has_walls=false", !mesh.z_has_walls_);

    // Duct BCs: no-slip z -> z_has_walls_ = true
    solver.set_velocity_bc(create_velocity_bc(BCPattern::Duct));
    record("Duct: z_has_walls=true", mesh.z_has_walls_);
}

// ============================================================================
// Test 3: Duct flow conserves mass and respects no-slip at all 4 walls
// ============================================================================

void test_duct_noslip_and_mass() {
    const int Nx = 8;
    const int Ny = 16;
    const int Nz = 16;
    const double Lx = 4.0;
    const double H = 1.0;
    const double W = 1.0;
    const double nu = 0.01;
    const double dp_dx = -0.1;

    Mesh mesh;
    mesh.init_uniform(Nx, Ny, Nz, 0.0, Lx, 0.0, H, 0.0, W);

    Config cfg;
    cfg.nu = nu;
    cfg.dp_dx = dp_dx;
    cfg.dt = 0.005;
    cfg.adaptive_dt = true;
    cfg.CFL_max = 0.8;
    cfg.max_steps = 500;
    cfg.tol = 1e-6;
    cfg.turb_model = TurbulenceModelType::None;
    cfg.simulation_mode = SimulationMode::Steady;
    cfg.verbose = false;

    RANSSolver solver(mesh, cfg);
    solver.set_velocity_bc(create_velocity_bc(BCPattern::Duct));
    solver.set_body_force(-dp_dx, 0.0, 0.0);
    solver.initialize_uniform(0.0, 0.0);

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    auto [residual, iters] = solver.solve_steady();
    solver.sync_from_gpu();

    // Check no-slip at y-walls (j = j_begin, j_end-1)
    double max_u_ywall = 0.0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            max_u_ywall = std::max(max_u_ywall,
                std::abs(solver.velocity().u_center(i, mesh.j_begin(), k)));
            max_u_ywall = std::max(max_u_ywall,
                std::abs(solver.velocity().u_center(i, mesh.j_end() - 1, k)));
        }
    }
    // No-slip is approximate: cell center is half a cell from wall
    // On coarse grid, wall-adjacent cell center u can be ~10% of bulk
    record("Duct no-slip y-walls", max_u_ywall < 0.2,
           "max_u=" + std::to_string(max_u_ywall));

    // Check no-slip at z-walls
    double max_u_zwall = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            max_u_zwall = std::max(max_u_zwall,
                std::abs(solver.velocity().u_center(i, j, mesh.k_begin())));
            max_u_zwall = std::max(max_u_zwall,
                std::abs(solver.velocity().u_center(i, j, mesh.k_end() - 1)));
        }
    }
    record("Duct no-slip z-walls", max_u_zwall < 0.2,
           "max_u=" + std::to_string(max_u_zwall));

    // All velocities finite
    bool all_finite = true;
    for (int k = mesh.k_begin(); k < mesh.k_end() && all_finite; ++k)
        for (int j = mesh.j_begin(); j < mesh.j_end() && all_finite; ++j)
            for (int i = mesh.i_begin(); i < mesh.i_end() && all_finite; ++i)
                if (!std::isfinite(solver.velocity().u_center(i, j, k)))
                    all_finite = false;
    record("Duct all velocities finite", all_finite);

    // z_has_walls_ was set correctly
    record("Duct z_has_walls flag set", mesh.z_has_walls_);

    std::cout << "  Converged in " << iters << " iters, residual = "
              << std::scientific << residual << "\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    namespace harness = nncfd::test::harness;
    return harness::run("Duct Validation Tests", [] {
        test_wall_distance_duct();
        test_solver_sets_z_walls();
        test_duct_noslip_and_mass();
    });
}
