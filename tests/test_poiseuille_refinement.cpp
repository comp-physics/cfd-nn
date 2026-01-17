/// @file test_poiseuille_refinement.cpp
/// @brief Poiseuille flow grid refinement convergence test
///
/// PURPOSE: MMS tests general convergence, but this tests BC + forcing +
/// projection + viscosity together with a real flow that has an exact solution.
///
/// SETUP:
///   - Run Poiseuille steady at two resolutions (NY = 32, 64)
///   - Compare profile error at each resolution
///   - Check monotone improvement and approximate order
///
/// VALIDATES:
///   1. Error decreases with resolution (err_fine < 0.9 * err_coarse)
///   2. Observed convergence rate > 1.5 (expect ~2 for 2nd order)
///
/// CATCHES:
///   - Something "works" only at one resolution
///   - BC discretization errors and stencil bugs
///   - Order-reducing bugs in boundary treatment
///
/// EMITS QOI:
///   poiseuille_refine: err32, err64, rate

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
// Exact Poiseuille solution (copied from test_poiseuille_steady.cpp)
// ============================================================================
struct PoiseuilleExact {
    double dp_dx;
    double nu;
    double H;
    double y_min;

    double u(double y) const {
        double y_rel = y - y_min;
        return (-dp_dx / (2.0 * nu)) * y_rel * (H - y_rel);
    }
};

// ============================================================================
// Compute relative L2 error of u-velocity profile against exact
// ============================================================================
static double compute_profile_relL2(const VectorField& vel, const Mesh& mesh,
                                     const PoiseuilleExact& exact) {
    double error_sq = 0.0;
    double norm_sq = 0.0;

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        double y = mesh.yc[j];
        double u_exact = exact.u(y);

        double u_avg = 0.0;
        int count = 0;
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
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
// Compute relative L2 norm of velocity change
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
// Run Poiseuille flow at a specific resolution and return profile error
// ============================================================================
static double run_poiseuille_at_resolution(int NY, const PoiseuilleExact& exact,
                                            double nu, double dp_dx,
                                            int& iters_out, bool& converged_out) {
    // Grid: keep aspect ratio roughly 2:1
    const int NX = NY / 2;
    const double Lx = 2.0;
    const double Ly = 1.0;

    // Convergence parameters - relaxed for faster testing
    const double steady_tol = 1e-8;
    const int consecutive_required = 20;
    const int max_iters = 100000;

    Mesh mesh;
    mesh.init_uniform(NX, NY, 0.0, Lx, 0.0, Ly);

    // dt scales with dy^2 for diffusive stability
    double dy = 1.0 / NY;
    double dt_base = 0.001;  // Base dt for NY=32
    double dt = dt_base * (dy * dy) / ((1.0/32.0) * (1.0/32.0));  // Scale as dy^2
    dt = std::min(dt, dt_base);  // Cap at base

    Config config;
    config.nu = nu;
    config.dt = dt;
    config.adaptive_dt = true;
    config.CFL_max = 0.3;  // More conservative CFL
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);
    solver.set_body_force(-dp_dx, 0.0);

    // Initialize with partial solution
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        double y = mesh.yc[j];
        double u_init = 0.5 * exact.u(y);
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            solver.velocity().u(i, j) = u_init;
        }
    }

    solver.sync_to_gpu();

    VectorField v_old(mesh);
    int consecutive_steady = 0;
    int iter = 0;
    converged_out = false;

    for (iter = 0; iter < max_iters; ++iter) {
        solver.sync_from_gpu();
        copy_velocity(v_old, solver.velocity(), mesh);
        solver.step();
        solver.sync_from_gpu();

        double rel_change = compute_velocity_change_relL2(solver.velocity(), v_old, mesh);

        if (rel_change < steady_tol) {
            consecutive_steady++;
            if (consecutive_steady >= consecutive_required) {
                converged_out = true;
                break;
            }
        } else {
            consecutive_steady = 0;
        }
    }

    iters_out = iter;

    solver.sync_from_gpu();
    return compute_profile_relL2(solver.velocity(), mesh, exact);
}

// ============================================================================
// Result structure
// ============================================================================
struct RefinementResult {
    double err32;
    double err64;
    int iters32;
    int iters64;
    bool conv32;
    bool conv64;
    double rate;
    bool monotone;
    bool rate_ok;
};

// ============================================================================
// Run refinement test
// ============================================================================
RefinementResult run_refinement_test() {
    RefinementResult result;

    // Physical parameters
    const double nu = 0.1;
    const double dp_dx = -0.1;

    PoiseuilleExact exact;
    exact.dp_dx = dp_dx;
    exact.nu = nu;
    exact.H = 1.0;
    exact.y_min = 0.0;

    std::cout << "    Running NY=32..." << std::flush;
    result.err32 = run_poiseuille_at_resolution(32, exact, nu, dp_dx, result.iters32, result.conv32);
    std::cout << " err=" << std::scientific << std::setprecision(2) << result.err32 << "\n";

    std::cout << "    Running NY=64..." << std::flush;
    result.err64 = run_poiseuille_at_resolution(64, exact, nu, dp_dx, result.iters64, result.conv64);
    std::cout << " err=" << result.err64 << "\n";

    // Check monotone improvement (at least 10% reduction)
    result.monotone = result.err64 < 0.9 * result.err32;

    // Compute observed convergence rate: rate = log(err32 / err64) / log(2)
    if (result.err64 > 1e-15 && result.err32 > 1e-15) {
        result.rate = std::log(result.err32 / result.err64) / std::log(2.0);
    } else {
        result.rate = 0.0;
    }

    // Rate should be > 1.5 (expect ~2 for 2nd order scheme)
    result.rate_ok = result.rate > 1.5;

    return result;
}

// ============================================================================
// Emit QoI for CI tracking
// ============================================================================
static void emit_qoi_refinement(const RefinementResult& r) {
    std::cout << "QOI_JSON: {\"test\":\"poiseuille_refine\""
              << ",\"err32\":" << harness::json_double(r.err32)
              << ",\"err64\":" << harness::json_double(r.err64)
              << ",\"rate\":" << harness::json_double(r.rate)
              << "}\n" << std::flush;
}

// ============================================================================
// Main test function
// ============================================================================
void test_poiseuille_refinement() {
    std::cout << "\n--- Poiseuille Refinement Test ---\n\n";
    std::cout << "  Testing convergence with grid refinement (NY = 32, 64)\n\n";

    RefinementResult r = run_refinement_test();

    // Print results
    std::cout << std::scientific << std::setprecision(4);
    std::cout << "\n  Errors:\n";
    std::cout << "    NY=32: err = " << r.err32 << " (iters: " << r.iters32
              << ", conv: " << (r.conv32 ? "yes" : "no") << ")\n";
    std::cout << "    NY=64: err = " << r.err64 << " (iters: " << r.iters64
              << ", conv: " << (r.conv64 ? "yes" : "no") << ")\n";

    std::cout << "\n  Convergence:\n";
    std::cout << "    Rate: " << std::fixed << std::setprecision(2) << r.rate
              << " (expect > 1.5 for 2nd order)\n";
    std::cout << "    err64 < 0.9*err32: " << (r.monotone ? "[OK]" : "[FAIL]") << "\n";
    std::cout << "    Rate > 1.5:        " << (r.rate_ok ? "[OK]" : "[FAIL]") << "\n\n";

    // Emit QoI
    emit_qoi_refinement(r);

    // Record results
    record("Both resolutions converged", r.conv32 && r.conv64);
    record("Error decreases with refinement (>10%)", r.monotone);
    record("Convergence rate > 1.5", r.rate_ok);
}

// ============================================================================
// Main
// ============================================================================
int main() {
    return harness::run("Poiseuille Refinement Test", []() {
        test_poiseuille_refinement();
    });
}
