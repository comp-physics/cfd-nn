/// @file test_projection_effectiveness.cpp
/// @brief Targeted projection effectiveness test using deliberately non-solenoidal field
///
/// PURPOSE: Verify that the pressure projection actually works by starting from
/// a velocity field that is KNOWN to be highly non-divergence-free, then checking
/// that projection reduces divergence by many orders of magnitude.
///
/// SETUP:
///   - Construct u* = grad(phi) for a smooth scalar phi
///   - This guarantees div(u*) = Lap(phi) which is large and non-zero
///   - Apply one projection step (pressure solve + velocity correction)
///   - Check divergence reduction
///
/// VALIDATES:
///   1. div_before is "large" (test is non-trivial)
///   2. div_after < 1e-4 (projection works, limited by iterative solver tolerance)
///   3. reduction_ratio = div_after/div_before < 1e-4 (4+ orders reduction)
///
/// CATCHES:
///   - Pressure solve not actually coupled
///   - Velocity correction applied with wrong sign
///   - Divergence operator mismatch with gradient operator
///
/// EMITS QOI:
///   projection_effectiveness: div_before, div_after, reduction_ratio

#include "test_harness.hpp"
#include "test_utilities.hpp"
#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "config.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>

using namespace nncfd;
using namespace nncfd::test;
using nncfd::test::harness::record;

// ============================================================================
// Construct u* = grad(phi) where phi = sin(kx*x)*sin(ky*y)
// This gives: div(u*) = Lap(phi) = -(kx^2 + ky^2)*sin(kx*x)*sin(ky*y)
// which is highly non-zero
// ============================================================================
static void construct_gradient_velocity(VectorField& vel, const Mesh& mesh) {
    double kx = 2.0 * M_PI / (mesh.x_max - mesh.x_min);
    double ky = 2.0 * M_PI / (mesh.y_max - mesh.y_min);

    // u = d(phi)/dx at u-faces
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        double y = mesh.yc[j];
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            double x = mesh.xf[i];
            // phi = sin(kx*x)*sin(ky*y)
            // d(phi)/dx = kx*cos(kx*x)*sin(ky*y)
            vel.u(i, j) = kx * std::cos(kx * x) * std::sin(ky * y);
        }
    }

    // v = d(phi)/dy at v-faces
    for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
        double y = mesh.yf[j];
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double x = mesh.xc[i];
            // d(phi)/dy = ky*sin(kx*x)*cos(ky*y)
            vel.v(i, j) = ky * std::sin(kx * x) * std::cos(ky * y);
        }
    }
}

// ============================================================================
// Compute max divergence
// ============================================================================
static double compute_max_div(const VectorField& v, const Mesh& mesh) {
    double max_div = 0.0;

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double dudx = (v.u(i+1, j) - v.u(i, j)) / mesh.dx;
            double dvdy = (v.v(i, j+1) - v.v(i, j)) / mesh.dy;
            max_div = std::max(max_div, std::abs(dudx + dvdy));
        }
    }

    return max_div;
}

// ============================================================================
// Compute L2 norm of divergence
// ============================================================================
static double compute_L2_div(const VectorField& v, const Mesh& mesh) {
    double div_sq = 0.0;

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double dudx = (v.u(i+1, j) - v.u(i, j)) / mesh.dx;
            double dvdy = (v.v(i, j+1) - v.v(i, j)) / mesh.dy;
            double div = dudx + dvdy;
            div_sq += div * div * mesh.dx * mesh.dy;
        }
    }

    return std::sqrt(div_sq);
}

// ============================================================================
// Result structure
// ============================================================================
struct ProjectionResult {
    double div_before_max;
    double div_before_L2;
    double div_after_max;
    double div_after_L2;
    double reduction_ratio_max;
    double reduction_ratio_L2;
    bool div_before_large;      // Test is non-trivial
    bool div_after_small;       // Projection works
    bool reduction_significant; // Significant reduction achieved
};

// ============================================================================
// Run projection effectiveness test
// ============================================================================
ProjectionResult run_projection_effectiveness_test() {
    ProjectionResult result;

    // Grid parameters
    const int NX = 32;
    const int NY = 32;
    const double Lx = 2.0 * M_PI;
    const double Ly = 2.0 * M_PI;

    Mesh mesh;
    mesh.init_uniform(NX, NY, 0.0, Lx, 0.0, Ly);

    Config config;
    config.nu = 0.01;
    config.dt = 0.01;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;
    config.adaptive_dt = false;

    RANSSolver solver(mesh, config);

    // Fully periodic BCs (simplest for projection test)
    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::Periodic;
    bc.y_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    // Construct u* = grad(phi) - highly non-solenoidal
    construct_gradient_velocity(solver.velocity(), mesh);

    // Compute divergence before projection
    result.div_before_max = compute_max_div(solver.velocity(), mesh);
    result.div_before_L2 = compute_L2_div(solver.velocity(), mesh);

    // Run one solver step (includes projection)
    solver.sync_to_gpu();
    solver.step();
    solver.sync_from_gpu();

    // Compute divergence after projection
    result.div_after_max = compute_max_div(solver.velocity(), mesh);
    result.div_after_L2 = compute_L2_div(solver.velocity(), mesh);

    // Compute reduction ratios
    result.reduction_ratio_max = (result.div_before_max > 1e-15) ?
                                  result.div_after_max / result.div_before_max : 0.0;
    result.reduction_ratio_L2 = (result.div_before_L2 > 1e-15) ?
                                 result.div_after_L2 / result.div_before_L2 : 0.0;

    // Pass criteria
    result.div_before_large = result.div_before_max > 0.1;  // Non-trivial starting field
    result.div_after_small = result.div_after_max < 1e-4;   // Projection effective (MG solver tolerance)
    result.reduction_significant = result.reduction_ratio_max < 1e-4;  // 4+ orders reduction

    return result;
}

// ============================================================================
// Emit QoI for CI tracking
// ============================================================================
static void emit_qoi_projection(const ProjectionResult& r) {
    std::cout << "QOI_JSON: {\"test\":\"projection_effectiveness\""
              << ",\"div_before_max\":" << harness::json_double(r.div_before_max)
              << ",\"div_before_L2\":" << harness::json_double(r.div_before_L2)
              << ",\"div_after_max\":" << harness::json_double(r.div_after_max)
              << ",\"div_after_L2\":" << harness::json_double(r.div_after_L2)
              << ",\"reduction_ratio_max\":" << harness::json_double(r.reduction_ratio_max)
              << ",\"reduction_ratio_L2\":" << harness::json_double(r.reduction_ratio_L2)
              << "}\n" << std::flush;
}

// ============================================================================
// Main test function
// ============================================================================
void test_projection_effectiveness() {
    std::cout << "\n--- Projection Effectiveness Test ---\n\n";
    std::cout << "  Starting from u* = grad(phi), which has large divergence.\n";
    std::cout << "  Checking that projection reduces divergence by >4 orders.\n\n";

    ProjectionResult r = run_projection_effectiveness_test();

    // Print results
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "  Divergence before projection:\n";
    std::cout << "    max|div|: " << r.div_before_max << " (expect > 0.1)\n";
    std::cout << "    L2(div):  " << r.div_before_L2 << "\n\n";

    std::cout << "  Divergence after projection:\n";
    std::cout << "    max|div|: " << r.div_after_max << " (limit: 1e-4)\n";
    std::cout << "    L2(div):  " << r.div_after_L2 << "\n\n";

    std::cout << "  Reduction ratios:\n";
    std::cout << "    max: " << r.reduction_ratio_max << " (limit: 1e-4)\n";
    std::cout << "    L2:  " << r.reduction_ratio_L2 << "\n\n";

    std::cout << "  Tests:\n";
    std::cout << "    Non-trivial starting field:  " << (r.div_before_large ? "[OK]" : "[FAIL]") << "\n";
    std::cout << "    Projection effective:        " << (r.div_after_small ? "[OK]" : "[FAIL]") << "\n";
    std::cout << "    Significant reduction:       " << (r.reduction_significant ? "[OK]" : "[FAIL]") << "\n\n";

    // Emit QoI
    emit_qoi_projection(r);

    // Record results
    record("Starting field has large divergence (>0.1)", r.div_before_large);
    record("Projected divergence < 1e-4", r.div_after_small);
    record("Divergence reduced by >4 orders", r.reduction_significant);
}

// ============================================================================
// Main
// ============================================================================
int main() {
    return harness::run("Projection Effectiveness Test", []() {
        test_projection_effectiveness();
    });
}
