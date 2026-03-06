/// @file test_duct_poiseuille.cpp
/// @brief Validates duct geometry against analytical laminar duct flow reference.
///
/// For a square duct with side a, driven by dp/dx, the laminar solution is a
/// double Fourier series. Key analytical facts for square cross-section:
///   - U_bulk_plates = |dp_dx| * a^2 / (12 * nu)    (parallel plates reference)
///   - U_bulk_duct / U_bulk_plates ≈ 0.4217          (series correction)
///   - f * Re_Dh = 56.91                             (friction factor)
///   - Velocity profile has 4-fold symmetry: u(y,z) = u(H-y,z) = u(y,W-z)

#include "test_harness.hpp"
#include "test_utilities.hpp"
#include "solver.hpp"
#include "config.hpp"
#include "mesh.hpp"
#include <cmath>
#include <vector>
#include <iomanip>

using namespace nncfd;
using namespace nncfd::test;
using nncfd::test::harness::record;

// ============================================================================
// Test 1: Laminar duct flow (3D) — bulk velocity and profile shape
// ============================================================================

void test_laminar_duct() {
    std::cout << "  Setting up laminar square duct flow...\n";

    // Square duct: periodic x, no-slip y and z
    // H = W = 1.0 (side length a = 1.0)
    const int Nx = 8;
    const int Ny = 32;
    const int Nz = 32;
    const double Lx = 4.0;
    const double H = 1.0;   // y extent [0, H]
    const double W = 1.0;   // z extent [0, W]
    const double nu = 0.01;
    const double dp_dx = -0.1;   // drives flow in +x

    Mesh mesh;
    mesh.init_uniform(Nx, Ny, Nz, 0.0, Lx, 0.0, H, 0.0, W);

    Config cfg;
    cfg.nu = nu;
    cfg.dp_dx = dp_dx;
    cfg.dt = 0.005;
    cfg.adaptive_dt = true;
    cfg.CFL_max = 0.8;
    cfg.max_steps = 3000;
    cfg.tol = 1e-8;
    cfg.turb_model = TurbulenceModelType::None;
    cfg.simulation_mode = SimulationMode::Steady;
    cfg.verbose = false;

    RANSSolver solver(mesh, cfg);
    solver.set_velocity_bc(create_velocity_bc(BCPattern::Duct));
    solver.set_body_force(-dp_dx, 0.0, 0.0);   // fx = |dp_dx|
    solver.initialize_uniform(0.0, 0.0);

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    // Run to steady state
    auto [residual, iters] = solver.solve_steady();
    solver.sync_from_gpu();

    std::cout << "  Converged in " << iters << " iterations, residual = "
              << std::scientific << std::setprecision(3) << residual << "\n";

    // -- Check 1: All velocities are finite --
    bool all_finite = true;
    for (int k = mesh.k_begin(); k < mesh.k_end() && all_finite; ++k)
        for (int j = mesh.j_begin(); j < mesh.j_end() && all_finite; ++j)
            for (int i = mesh.i_begin(); i < mesh.i_end() && all_finite; ++i)
                if (!std::isfinite(solver.velocity().u_center(i, j, k)))
                    all_finite = false;
    record("All velocities finite", all_finite);

    // -- Check 2: No-slip at y walls --
    // Near-wall u should be much smaller than center u
    double max_u_ywall = 0.0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k)
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            // u at first interior row next to y=0 wall
            double u_lo = std::abs(solver.velocity().u_center(i, mesh.j_begin(), k));
            // u at last interior row next to y=H wall
            double u_hi = std::abs(solver.velocity().u_center(i, mesh.j_end() - 1, k));
            max_u_ywall = std::max(max_u_ywall, std::max(u_lo, u_hi));
        }

    // -- Check 3: No-slip at z walls --
    double max_u_zwall = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j)
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double u_lo = std::abs(solver.velocity().u_center(i, j, mesh.k_begin()));
            double u_hi = std::abs(solver.velocity().u_center(i, j, mesh.k_end() - 1));
            max_u_zwall = std::max(max_u_zwall, std::max(u_lo, u_hi));
        }

    // -- Check 4: Maximum velocity at center --
    int j_mid = mesh.j_begin() + Ny / 2;
    int k_mid = mesh.k_begin() + Nz / 2;
    double u_center = 0.0;
    for (int i = mesh.i_begin(); i < mesh.i_end(); ++i)
        u_center += solver.velocity().u_center(i, j_mid, k_mid);
    u_center /= Nx;   // average over x (should be uniform in periodic x)

    double u_max_global = 0.0;
    int j_max = 0, k_max = 0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k)
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            double u_avg_x = 0.0;
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i)
                u_avg_x += solver.velocity().u_center(i, j, k);
            u_avg_x /= Nx;
            if (u_avg_x > u_max_global) {
                u_max_global = u_avg_x;
                j_max = j;
                k_max = k;
            }
        }

    // Max velocity should be near the center of the cross-section
    bool max_near_center = (std::abs(j_max - j_mid) <= 2) && (std::abs(k_max - k_mid) <= 2);
    record("Max velocity near duct center", max_near_center);

    // Near-wall velocity should be much less than center velocity
    bool noslip_y = (max_u_ywall < 0.5 * u_max_global);
    bool noslip_z = (max_u_zwall < 0.5 * u_max_global);
    record("No-slip enforced at y walls", noslip_y);
    record("No-slip enforced at z walls", noslip_z);

    // -- Check 5: Bulk velocity vs analytical --
    // U_bulk_plates = |dp_dx| * H^2 / (12 * nu) for infinite parallel plates at y=0,H
    double U_bulk_plates = std::abs(dp_dx) * H * H / (12.0 * nu);
    // For square duct: U_bulk_duct / U_bulk_plates ≈ 0.4217

    // Compute numerical bulk velocity (volume average of u)
    double u_sum = 0.0;
    int cell_count = 0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k)
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j)
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                u_sum += solver.velocity().u_center(i, j, k);
                cell_count++;
            }
    double U_bulk_num = u_sum / cell_count;

    double ratio = U_bulk_num / U_bulk_plates;
    std::cout << "  U_bulk_plates = " << std::fixed << std::setprecision(6) << U_bulk_plates << "\n";
    std::cout << "  U_bulk_duct   = " << U_bulk_num << "\n";
    std::cout << "  Ratio (analytical ≈ 0.4217) = " << ratio << "\n";

    // Accept if ratio is in a reasonable range (coarse grid, so allow generous tolerance)
    bool ratio_ok = (ratio > 0.30 && ratio < 0.55);
    record("Bulk velocity ratio vs plates (expect ~0.42)", ratio_ok);

    // -- Check 6: Flow is positive (driven by body force in +x) --
    bool flow_positive = (U_bulk_num > 0.0);
    record("Flow driven in correct direction", flow_positive);

    // -- Check 7: Transverse velocities are small --
    double max_v = 0.0, max_w = 0.0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k)
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j)
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                max_v = std::max(max_v, std::abs(solver.velocity().v_center(i, j, k)));
                max_w = std::max(max_w, std::abs(solver.velocity().w_center(i, j, k)));
            }

    bool transverse_small = (max_v < 0.01 * u_max_global) && (max_w < 0.01 * u_max_global);
    record("Transverse velocities small (v,w << u)", transverse_small);

    std::cout << "  max|v|/u_max = " << std::scientific << max_v / u_max_global
              << ", max|w|/u_max = " << max_w / u_max_global << "\n";
}

// ============================================================================
// Test 2: Duct symmetry test — 4-fold symmetry of square duct
// ============================================================================

void test_duct_symmetry() {
    std::cout << "  Setting up duct symmetry test...\n";

    const int Nx = 8;
    const int Ny = 32;
    const int Nz = 32;
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
    cfg.max_steps = 3000;
    cfg.tol = 1e-8;
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

    std::cout << "  Converged in " << iters << " iterations, residual = "
              << std::scientific << std::setprecision(3) << residual << "\n";

    // Build x-averaged u(j, k) profile for symmetry checks
    // Average u over all x-locations (should be uniform for periodic x)
    std::vector<double> u_jk(Ny * Nz, 0.0);
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k)
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            double u_avg = 0.0;
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i)
                u_avg += solver.velocity().u_center(i, j, k);
            u_avg /= Nx;
            int jj = j - mesh.j_begin();
            int kk = k - mesh.k_begin();
            u_jk[kk * Ny + jj] = u_avg;
        }

    // Find max velocity for relative error
    double u_max = 0.0;
    for (int idx = 0; idx < Ny * Nz; ++idx)
        u_max = std::max(u_max, std::abs(u_jk[idx]));

    // Check y-symmetry: u(j, k) ≈ u(Ny-1-j, k)
    double max_y_asym = 0.0;
    for (int kk = 0; kk < Nz; ++kk)
        for (int jj = 0; jj < Ny / 2; ++jj) {
            double u1 = u_jk[kk * Ny + jj];
            double u2 = u_jk[kk * Ny + (Ny - 1 - jj)];
            double diff = std::abs(u1 - u2);
            max_y_asym = std::max(max_y_asym, diff);
        }
    double y_asym_rel = max_y_asym / (u_max + 1e-30);
    std::cout << "  Y-symmetry max relative error: " << std::scientific << y_asym_rel << "\n";
    record("Y-symmetry: u(y,z) ≈ u(H-y,z)", y_asym_rel < 0.05);

    // Check z-symmetry: u(j, k) ≈ u(j, Nz-1-k)
    double max_z_asym = 0.0;
    for (int kk = 0; kk < Nz / 2; ++kk)
        for (int jj = 0; jj < Ny; ++jj) {
            double u1 = u_jk[kk * Ny + jj];
            double u2 = u_jk[(Nz - 1 - kk) * Ny + jj];
            double diff = std::abs(u1 - u2);
            max_z_asym = std::max(max_z_asym, diff);
        }
    double z_asym_rel = max_z_asym / (u_max + 1e-30);
    std::cout << "  Z-symmetry max relative error: " << std::scientific << z_asym_rel << "\n";
    record("Z-symmetry: u(y,z) ≈ u(y,W-z)", z_asym_rel < 0.05);

    // Check diagonal symmetry (since H = W, square duct): u(j,k) ≈ u(k,j)
    // This only holds when Ny == Nz and H == W
    double max_diag_asym = 0.0;
    for (int kk = 0; kk < Nz; ++kk)
        for (int jj = 0; jj < Ny; ++jj) {
            double u1 = u_jk[kk * Ny + jj];
            double u2 = u_jk[jj * Ny + kk];  // swap j and k
            double diff = std::abs(u1 - u2);
            max_diag_asym = std::max(max_diag_asym, diff);
        }
    double diag_asym_rel = max_diag_asym / (u_max + 1e-30);
    std::cout << "  Diagonal symmetry max relative error: " << std::scientific << diag_asym_rel << "\n";
    record("Diagonal symmetry: u(y,z) ≈ u(z,y)", diag_asym_rel < 0.05);
}

// ============================================================================
// Test 3: Duct with SST turbulence model — stability check
// ============================================================================

void test_duct_sst() {
    std::cout << "  Setting up duct with SST model...\n";

    const int Nx = 8;
    const int Ny = 16;
    const int Nz = 16;
    const double Lx = 4.0;
    const double H = 1.0;
    const double W = 1.0;
    const double nu = 0.001;
    const double dp_dx = -0.1;

    Mesh mesh;
    mesh.init_uniform(Nx, Ny, Nz, 0.0, Lx, 0.0, H, 0.0, W);

    Config cfg;
    cfg.nu = nu;
    cfg.dp_dx = dp_dx;
    cfg.dt = 0.001;
    cfg.adaptive_dt = true;
    cfg.CFL_max = 0.5;
    cfg.max_steps = 500;
    cfg.tol = 1e-6;
    cfg.turb_model = TurbulenceModelType::SSTKOmega;
    cfg.verbose = false;

    RANSSolver solver(mesh, cfg);
    solver.set_velocity_bc(create_velocity_bc(BCPattern::Duct));
    solver.set_body_force(-dp_dx, 0.0, 0.0);
    solver.initialize_uniform(0.0, 0.0);

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    // Run 200 steps — just checking stability, not accuracy
    const int nsteps = 200;
    bool stable = true;
    for (int step = 0; step < nsteps && stable; ++step) {
        double res = solver.step();
        if (!std::isfinite(res)) {
            std::cerr << "  SST duct NaN at step " << step << "\n";
            stable = false;
        }
    }
    solver.sync_from_gpu();

    record("SST duct stable for 200 steps", stable);

    if (!stable) return;

    // Check all velocities are finite
    bool all_finite = true;
    double max_vel = 0.0;
    for (int k = mesh.k_begin(); k < mesh.k_end() && all_finite; ++k)
        for (int j = mesh.j_begin(); j < mesh.j_end() && all_finite; ++j)
            for (int i = mesh.i_begin(); i < mesh.i_end() && all_finite; ++i) {
                double uc = solver.velocity().u_center(i, j, k);
                if (!std::isfinite(uc)) {
                    all_finite = false;
                } else {
                    max_vel = std::max(max_vel, std::abs(uc));
                }
            }
    record("SST duct: all velocities finite", all_finite);

    // Velocity should be bounded (not blown up)
    bool bounded = (max_vel < 100.0);
    std::cout << "  SST duct max|u| = " << std::scientific << max_vel << "\n";
    record("SST duct: velocity bounded", bounded);

    // Check nu_t is non-negative
    double min_nut = 1e30, max_nut = 0.0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k)
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j)
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double nt = solver.nu_t()(i, j, k);
                if (std::isfinite(nt)) {
                    min_nut = std::min(min_nut, nt);
                    max_nut = std::max(max_nut, nt);
                }
            }
    std::cout << "  SST duct nu_t range: [" << std::scientific << min_nut
              << ", " << max_nut << "]\n";
    record("SST duct: nu_t non-negative", min_nut >= 0.0);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    return nncfd::test::harness::run_sections("DuctPoiseuille", {
        {"Laminar duct flow", test_laminar_duct},
        {"Duct symmetry", test_duct_symmetry},
        {"Duct with SST model", test_duct_sst},
    });
}
