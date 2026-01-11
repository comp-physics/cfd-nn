/// @file test_poiseuille_steady.cpp
/// @brief Decisive analytic steady-state Poiseuille flow test
///
/// PURPOSE: Nails BCs, forcing sign, nu, diffusion stencil, projection coupling
/// with interpretable exact-solution comparison.
///
/// SETUP:
///   - 2D channel, periodic x, no-slip walls y
///   - Laminar (no turbulence), no perturbations
///   - Fixed dp/dx body force
///   - Run until truly steady: relL2(u^{n+1}-u^n) < 1e-10 for 50 consecutive steps
///
/// VALIDATES THREE INDEPENDENT THINGS:
///   1. relL2(u(y) - u_exact(y)) < 1e-3 (profile accuracy)
///   2. |U_bulk - U_bulk_exact| / U_bulk_exact < 1e-3 (integrated quantity)
///   3. |tau_w - tau_w_exact| / |tau_w_exact| < 1e-2 (derivative - noisier)
///
/// EXACT SOLUTION (Poiseuille flow between parallel plates):
///   u(y) = (dp/dx) / (2*nu) * y * (H - y)  [for y in [0, H]]
///   U_bulk = (dp/dx) * H^2 / (12*nu)
///   tau_w = nu * du/dy|_wall = (dp/dx) * H / 2
///
/// EMITS QOI:
///   poiseuille_steady: relL2_profile, U_bulk_err, tau_w_err, iters_to_steady

#include "test_harness.hpp"
#include "test_utilities.hpp"
#include "mesh.hpp"
#include "solver.hpp"
#include "config.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <vector>

using namespace nncfd;
using namespace nncfd::test;
using nncfd::test::harness::record;

// ============================================================================
// Exact Poiseuille solution
// ============================================================================
struct PoiseuilleExact {
    double dp_dx;   // pressure gradient (negative for flow in +x)
    double nu;      // kinematic viscosity
    double H;       // channel height
    double y_min;   // lower wall y-coordinate

    // Exact velocity profile: u(y) = -(dp/dx)/(2*nu) * (y-y_min) * (H - (y-y_min))
    double u(double y) const {
        double y_rel = y - y_min;
        return (-dp_dx / (2.0 * nu)) * y_rel * (H - y_rel);
    }

    // Exact bulk velocity: U_bulk = -(dp/dx) * H^2 / (12*nu)
    double U_bulk() const {
        return (-dp_dx) * H * H / (12.0 * nu);
    }

    // Exact wall shear stress: tau_w = nu * du/dy|_wall = -(dp/dx) * H / 2
    // (magnitude, positive for flow in +x direction)
    double tau_w() const {
        return (-dp_dx) * H / 2.0;
    }

    // Exact du/dy at lower wall (y = y_min)
    double dudy_lower() const {
        return (-dp_dx / (2.0 * nu)) * H;
    }

    // Exact du/dy at upper wall (y = y_min + H)
    double dudy_upper() const {
        return -(-dp_dx / (2.0 * nu)) * H;
    }
};

// ============================================================================
// Compute relative L2 error of u-velocity profile against exact
// ============================================================================
static double compute_profile_relL2(const VectorField& vel, const Mesh& mesh,
                                     const PoiseuilleExact& exact) {
    double error_sq = 0.0;
    double norm_sq = 0.0;

    // Sample at cell centers, average u from faces
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        double y = mesh.yc[j];
        double u_exact = exact.u(y);

        // Average over x (should be uniform for steady Poiseuille)
        double u_avg = 0.0;
        int count = 0;
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            // Interpolate u to cell center
            u_avg += 0.5 * (vel.u(i, j) + vel.u(i+1, j));
            count++;
        }
        u_avg /= count;

        double diff = u_avg - u_exact;
        error_sq += diff * diff * mesh.dy;
        norm_sq += u_exact * u_exact * mesh.dy;
    }

    return (norm_sq > 1e-30) ? std::sqrt(error_sq / norm_sq) : std::sqrt(error_sq);
}

// ============================================================================
// Compute bulk velocity (volume-averaged u)
// ============================================================================
static double compute_bulk_velocity(const VectorField& vel, const Mesh& mesh) {
    double u_sum = 0.0;
    double vol = 0.0;

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double u_cell = 0.5 * (vel.u(i, j) + vel.u(i+1, j));
            u_sum += u_cell * mesh.dx * mesh.dy;
            vol += mesh.dx * mesh.dy;
        }
    }

    return u_sum / vol;
}

// ============================================================================
// Compute wall shear stress (nu * du/dy at walls)
// ============================================================================
static double compute_wall_shear(const VectorField& vel, const Mesh& mesh, double nu) {
    // Compute du/dy at lower wall using one-sided difference
    // At j = j_begin (first interior cell), wall is at j_begin - 0.5
    // u_wall = 0 (no-slip), so du/dy ≈ (u_cell - 0) / (dy/2)

    double tau_sum = 0.0;
    int count = 0;

    int j_lo = mesh.j_begin();
    for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
        double u_cell = 0.5 * (vel.u(i, j_lo) + vel.u(i+1, j_lo));
        // Distance from wall to cell center is dy/2
        double dudy = u_cell / (mesh.dy / 2.0);
        tau_sum += nu * dudy;
        count++;
    }

    return tau_sum / count;
}

// ============================================================================
// Compute relative L2 norm of velocity change between steps
// ============================================================================
static double compute_velocity_change_relL2(const VectorField& v_new,
                                             const VectorField& v_old,
                                             const Mesh& mesh) {
    double diff_sq = 0.0;
    double norm_sq = 0.0;

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            double du = v_new.u(i, j) - v_old.u(i, j);
            diff_sq += du * du;
            norm_sq += v_new.u(i, j) * v_new.u(i, j);
        }
    }

    return (norm_sq > 1e-30) ? std::sqrt(diff_sq / norm_sq) : std::sqrt(diff_sq);
}

// ============================================================================
// Copy velocity field
// ============================================================================
static void copy_velocity(VectorField& dst, const VectorField& src, const Mesh& mesh) {
    for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            dst.u(i, j) = src.u(i, j);
            dst.v(i, j) = src.v(i, j);
        }
    }
}

// ============================================================================
// Result structure
// ============================================================================
struct PoiseuilleResult {
    bool converged;
    int iters_to_steady;
    double relL2_profile;
    double U_bulk_num;
    double U_bulk_exact;
    double U_bulk_err;
    double tau_w_num;
    double tau_w_exact;
    double tau_w_err;
    bool profile_ok;
    bool bulk_ok;
    bool shear_ok;
};

// ============================================================================
// Run Poiseuille steady-state test
// ============================================================================
PoiseuilleResult run_poiseuille_steady() {
    PoiseuilleResult result;
    result.converged = false;

    // Grid and physical parameters
    // Higher viscosity = faster diffusion = faster convergence to steady state
    // Finer grid = lower truncation error for strict tolerances
    const int NX = 32;
    const int NY = 64;
    const double Lx = 2.0;
    const double Ly = 1.0;
    const double nu = 0.1;              // Higher viscosity for faster convergence
    const double dp_dx = -0.1;          // Stronger forcing, scaled with nu

    // Convergence parameters
    // Strict tolerances for a proper physics validation test
    const double steady_tol = 1e-10;      // relL2 change threshold
    const int consecutive_required = 50;   // Must meet criterion this many times
    const int max_iters = 50000;           // Safety limit

    Mesh mesh;
    mesh.init_uniform(NX, NY, 0.0, Lx, 0.0, Ly);

    PoiseuilleExact exact;
    exact.dp_dx = dp_dx;
    exact.nu = nu;
    exact.H = Ly;
    exact.y_min = 0.0;

    Config config;
    config.nu = nu;
    config.dt = 0.001;
    config.adaptive_dt = true;
    config.CFL_max = 0.5;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);

    // Body force = -dp/dx (pressure gradient drives flow)
    solver.set_body_force(-dp_dx, 0.0);

    // Initialize with partial solution (helps convergence)
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        double y = mesh.yc[j];
        double u_init = 0.5 * exact.u(y);  // Start at 50% of exact
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            solver.velocity().u(i, j) = u_init;
        }
    }

    solver.sync_to_gpu();

    // Temporary storage for previous velocity
    VectorField v_old(mesh);

    int consecutive_steady = 0;
    int iter = 0;

    // Time-step until steady
    for (iter = 0; iter < max_iters; ++iter) {
        // Save current velocity
        solver.sync_from_gpu();
        copy_velocity(v_old, solver.velocity(), mesh);

        // Take one step
        solver.step();
        solver.sync_from_gpu();

        // Check convergence
        double rel_change = compute_velocity_change_relL2(solver.velocity(), v_old, mesh);

        if (rel_change < steady_tol) {
            consecutive_steady++;
            if (consecutive_steady >= consecutive_required) {
                result.converged = true;
                break;
            }
        } else {
            consecutive_steady = 0;
        }

        // Progress output every 5000 steps
        if (iter > 0 && iter % 5000 == 0) {
            std::cout << "    iter " << iter << ": rel_change = " << std::scientific
                      << rel_change << ", consecutive = " << consecutive_steady << "\n";
        }
    }

    result.iters_to_steady = iter;

    // Compute metrics
    solver.sync_from_gpu();

    result.relL2_profile = compute_profile_relL2(solver.velocity(), mesh, exact);
    result.U_bulk_num = compute_bulk_velocity(solver.velocity(), mesh);
    result.U_bulk_exact = exact.U_bulk();
    result.U_bulk_err = std::abs(result.U_bulk_num - result.U_bulk_exact) / result.U_bulk_exact;

    result.tau_w_num = compute_wall_shear(solver.velocity(), mesh, nu);
    result.tau_w_exact = exact.tau_w();
    result.tau_w_err = std::abs(result.tau_w_num - result.tau_w_exact) / std::abs(result.tau_w_exact);

    // Pass/fail criteria - strict for a proper physics validation
    result.profile_ok = result.relL2_profile < 1e-3;  // 0.1% profile error
    result.bulk_ok = result.U_bulk_err < 1e-3;        // 0.1% bulk velocity error
    result.shear_ok = result.tau_w_err < 1e-2;        // 1% wall shear (derivative is noisier)

    return result;
}

// ============================================================================
// Emit QoI for CI tracking
// ============================================================================
static void emit_qoi_poiseuille(const PoiseuilleResult& r) {
    std::cout << "QOI_JSON: {\"test\":\"poiseuille_steady\""
              << ",\"relL2_profile\":" << harness::json_double(r.relL2_profile)
              << ",\"U_bulk_err\":" << harness::json_double(r.U_bulk_err)
              << ",\"tau_w_err\":" << harness::json_double(r.tau_w_err)
              << ",\"iters_to_steady\":" << r.iters_to_steady
              << ",\"converged\":" << (r.converged ? "true" : "false")
              << "}\n" << std::flush;
}

// ============================================================================
// Main test function
// ============================================================================
void test_poiseuille_steady() {
    std::cout << "\n--- Poiseuille Steady-State Test ---\n\n";
    std::cout << "  Running 2D channel flow to steady state...\n";
    std::cout << "  Criterion: relL2(u^{n+1}-u^n) < 1e-10 for 50 consecutive steps\n\n";

    PoiseuilleResult r = run_poiseuille_steady();

    // Print results
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  Converged:        " << (r.converged ? "yes" : "NO") << "\n";
    std::cout << "  Iters to steady:  " << r.iters_to_steady << "\n\n";

    std::cout << "  Profile validation:\n";
    std::cout << "    relL2(u - u_exact):  " << std::scientific << r.relL2_profile
              << " (limit: 1e-3) " << (r.profile_ok ? "[OK]" : "[FAIL]") << "\n";

    std::cout << "\n  Bulk velocity:\n";
    std::cout << "    U_bulk (numerical): " << std::scientific << r.U_bulk_num << "\n";
    std::cout << "    U_bulk (exact):     " << r.U_bulk_exact << "\n";
    std::cout << "    Relative error:     " << r.U_bulk_err
              << " (limit: 1e-3) " << (r.bulk_ok ? "[OK]" : "[FAIL]") << "\n";

    std::cout << "\n  Wall shear stress:\n";
    std::cout << "    tau_w (numerical):  " << std::scientific << r.tau_w_num << "\n";
    std::cout << "    tau_w (exact):      " << r.tau_w_exact << "\n";
    std::cout << "    Relative error:     " << r.tau_w_err
              << " (limit: 1e-2) " << (r.shear_ok ? "[OK]" : "[FAIL]") << "\n\n";

    // Emit QoI
    emit_qoi_poiseuille(r);

    // Record results
    record("Reached steady state", r.converged);
    record("Profile accuracy (relL2 < 1e-3)", r.profile_ok);
    record("Bulk velocity accuracy (err < 1e-3)", r.bulk_ok);
    record("Wall shear accuracy (err < 1e-2)", r.shear_ok);
}

// ============================================================================
// Main
// ============================================================================
int main() {
    return harness::run("Poiseuille Steady-State Test", []() {
        test_poiseuille_steady();
    });
}
