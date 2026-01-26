/// @file test_operator_consistency.cpp
/// @brief Operator consistency tests for MAC grid (catch classic index bugs)
///
/// PURPOSE: Validate the math structure the solver relies on, without "physics".
/// These catch mismatched staggering, missing dx factors, wrong boundary handling.
///
/// TESTS:
/// 3A) Discrete adjointness: <grad(p), u> ≈ -<p, div(u)>
/// 3B) Projection makes velocity divergence-free: |div(u)| < tol after step
/// 3C) Divergence reduces with projection: div_after << div_before
///
/// EMITS QOI:
///   op_adjoint: rel_mismatch
///   projection_divfree: div_after, div_before
///   divergence_reduction: reduction_factor

#include "test_harness.hpp"
#include "test_utilities.hpp"
#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "config.hpp"
#include <cmath>
#include <random>
#include <iostream>
#include <iomanip>

using namespace nncfd;
using namespace nncfd::test;
using nncfd::test::harness::record;
using nncfd::test::create_velocity_bc;
using nncfd::test::BCPattern;

// ============================================================================
// Helper: Generate smooth random field (for testing operators)
// ============================================================================
static void generate_smooth_pressure(ScalarField& p, const Mesh& mesh, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // Use low-frequency sinusoids for smoothness
    double kx = 2.0 * M_PI / (mesh.x_max - mesh.x_min);
    double ky = 2.0 * M_PI / (mesh.y_max - mesh.y_min);

    double ax = dist(rng), bx = dist(rng);
    double ay = dist(rng), by = dist(rng);

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double x = mesh.xc[i];
            double y = mesh.yc[j];
            p(i, j) = ax * std::sin(kx * x) + bx * std::cos(kx * x)
                    + ay * std::sin(ky * y) + by * std::cos(ky * y);
        }
    }
}

static void generate_smooth_velocity(VectorField& v, const Mesh& mesh, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    double kx = 2.0 * M_PI / (mesh.x_max - mesh.x_min);
    double ky = 2.0 * M_PI / (mesh.y_max - mesh.y_min);

    double au = dist(rng), bu = dist(rng);
    double av = dist(rng), bv = dist(rng);

    // Generate non-divergence-free velocity field
    // u = sin(kx*x) * sin(ky*y), v = sin(kx*x) * sin(ky*y)
    // div = kx*cos(kx*x)*sin(ky*y) + ky*sin(kx*x)*cos(ky*y) != 0

    // u at x-faces
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            double x = mesh.xf[i];
            double y = mesh.yc[j];
            v.u(i, j) = au * std::sin(kx * x) * std::sin(ky * y)
                      + bu * std::cos(kx * x);
        }
    }

    // v at y-faces
    for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double x = mesh.xc[i];
            double y = mesh.yf[j];
            v.v(i, j) = av * std::sin(kx * x) * std::sin(ky * y)
                      + bv * std::cos(ky * y);
        }
    }
}

// ============================================================================
// Compute <grad(p), u> = sum over faces of (dp/dx * u + dp/dy * v) * dV
// ============================================================================
static double compute_gradp_dot_u(const ScalarField& p, const VectorField& u,
                                   const Mesh& mesh) {
    double result = 0.0;

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            // dp/dx at cell center (approximation)
            double dpdx = (p(i+1, j) - p(i-1, j)) / (2.0 * mesh.dx);
            // dp/dy at cell center
            double dpdy = (p(i, j+1) - p(i, j-1)) / (2.0 * mesh.dy);

            // u, v at cell center
            double u_cc = 0.5 * (u.u(i, j) + u.u(i+1, j));
            double v_cc = 0.5 * (u.v(i, j) + u.v(i, j+1));

            result += (dpdx * u_cc + dpdy * v_cc) * mesh.dx * mesh.dy;
        }
    }

    return result;
}

// ============================================================================
// Compute <p, div(u)> = sum over cells of p * div(u) * dV
// ============================================================================
static double compute_p_dot_divu(const ScalarField& p, const VectorField& u,
                                  const Mesh& mesh) {
    double result = 0.0;

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double dudx = (u.u(i+1, j) - u.u(i, j)) / mesh.dx;
            double dvdy = (u.v(i, j+1) - u.v(i, j)) / mesh.dy;
            double div = dudx + dvdy;

            result += p(i, j) * div * mesh.dx * mesh.dy;
        }
    }

    return result;
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
// Test 3A: Discrete adjointness <grad(p), u> ≈ -<p, div(u)>
// ============================================================================
struct AdjointResult {
    double gradp_u;
    double p_divu;
    double rel_mismatch;
    bool passed;
};

AdjointResult test_adjointness() {
    AdjointResult result;

    const int N = 32;
    Mesh mesh;
    mesh.init_uniform(N, N, 0.0, 2.0*M_PI, 0.0, 2.0*M_PI);

    ScalarField p(mesh);
    VectorField u(mesh);

    generate_smooth_pressure(p, mesh, 12345);
    generate_smooth_velocity(u, mesh, 67890);

    result.gradp_u = compute_gradp_dot_u(p, u, mesh);
    result.p_divu = compute_p_dot_divu(p, u, mesh);

    // Check: <grad(p), u> ≈ -<p, div(u)>
    double sum = result.gradp_u + result.p_divu;  // Should be ~0
    double scale = std::max(std::abs(result.gradp_u), std::abs(result.p_divu));
    scale = std::max(scale, 1e-10);

    result.rel_mismatch = std::abs(sum) / scale;
    result.passed = result.rel_mismatch < 0.1;  // Allow some boundary effects

    return result;
}

// ============================================================================
// Test 3B: Projection makes velocity divergence-free
// ============================================================================
struct DivFreeResult {
    double div_before;
    double div_after;
    bool passed;
};

DivFreeResult test_projection_divfree() {
    DivFreeResult result;

    const int N = 32;
    Mesh mesh;
    mesh.init_uniform(N, N, 0.0, 2.0*M_PI, 0.0, 2.0*M_PI);

    Config config;
    config.nu = 0.01;
    config.dt = 0.01;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;
    config.adaptive_dt = false;

    RANSSolver solver(mesh, config);
    solver.set_velocity_bc(create_velocity_bc(BCPattern::FullyPeriodic));

    // Generate non-div-free velocity
    generate_smooth_velocity(solver.velocity(), mesh, 11111);

    // Compute divergence before step
    result.div_before = compute_max_div(solver.velocity(), mesh);

    // Run one step (includes projection)
    solver.sync_to_gpu();
    solver.step();
    solver.sync_from_gpu();

    // Compute divergence after step
    result.div_after = compute_max_div(solver.velocity(), mesh);

    // After projection, divergence should be very small
    // Allow 1e-4 for iterative solver tolerance (depending on Poisson solver settings)
    result.passed = result.div_after < 1e-4;

    return result;
}

// ============================================================================
// Test 3C: Divergence reduces significantly with projection
// ============================================================================
struct DivReductionResult {
    double div_before;
    double div_after;
    double reduction_factor;
    bool passed;
};

DivReductionResult test_divergence_reduction() {
    DivReductionResult result;

    const int N = 32;
    Mesh mesh;
    mesh.init_uniform(N, N, 0.0, 2.0*M_PI, 0.0, 2.0*M_PI);

    Config config;
    config.nu = 0.01;
    config.dt = 0.001;  // Small dt to minimize advection effects
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;
    config.adaptive_dt = false;

    RANSSolver solver(mesh, config);
    solver.set_velocity_bc(create_velocity_bc(BCPattern::FullyPeriodic));

    // Generate highly non-div-free velocity (large divergence)
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        double y = mesh.yc[j];
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            double x = mesh.xf[i];
            // u = sin(x) * sin(y) - not divergence-free
            solver.velocity().u(i, j) = 0.1 * std::sin(x) * std::sin(y);
        }
    }
    for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
        double y = mesh.yf[j];
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double x = mesh.xc[i];
            // v = sin(x) * sin(y) - not divergence-free
            solver.velocity().v(i, j) = 0.1 * std::sin(x) * std::sin(y);
        }
    }

    // Compute divergence before
    result.div_before = compute_max_div(solver.velocity(), mesh);

    // Run one step
    solver.sync_to_gpu();
    solver.step();
    solver.sync_from_gpu();

    // Compute divergence after
    result.div_after = compute_max_div(solver.velocity(), mesh);

    // Compute reduction factor
    result.reduction_factor = (result.div_before > 1e-15) ?
                              result.div_after / result.div_before : 0.0;

    // Divergence should reduce by at least 4 orders of magnitude
    result.passed = (result.reduction_factor < 1e-4) || (result.div_after < 1e-10);

    return result;
}

// ============================================================================
// Emit QoI functions
// ============================================================================
static void emit_qoi_adjoint(const AdjointResult& r) {
    std::cout << "QOI_JSON: {\"test\":\"op_adjoint\""
              << ",\"rel_mismatch\":" << harness::json_double(r.rel_mismatch)
              << ",\"gradp_u\":" << harness::json_double(r.gradp_u)
              << ",\"p_divu\":" << harness::json_double(r.p_divu)
              << "}\n" << std::flush;
}

static void emit_qoi_divfree(const DivFreeResult& r) {
    std::cout << "QOI_JSON: {\"test\":\"projection_divfree\""
              << ",\"div_before\":" << harness::json_double(r.div_before)
              << ",\"div_after\":" << harness::json_double(r.div_after)
              << "}\n" << std::flush;
}

static void emit_qoi_reduction(const DivReductionResult& r) {
    std::cout << "QOI_JSON: {\"test\":\"divergence_reduction\""
              << ",\"div_before\":" << harness::json_double(r.div_before)
              << ",\"div_after\":" << harness::json_double(r.div_after)
              << ",\"reduction_factor\":" << harness::json_double(r.reduction_factor)
              << "}\n" << std::flush;
}

// ============================================================================
// Main test functions
// ============================================================================
void run_all_operator_tests() {
    std::cout << "\n--- Operator Consistency Tests ---\n";

    // Test 3A: Adjointness
    std::cout << "\n=== Test 3A: Discrete Adjointness ===\n";
    std::cout << "  Checking: <grad(p), u> ≈ -<p, div(u)>\n\n";

    AdjointResult adj = test_adjointness();
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "  <grad(p), u>:  " << adj.gradp_u << "\n";
    std::cout << "  <p, div(u)>:   " << adj.p_divu << "\n";
    std::cout << "  Sum (should~0):" << (adj.gradp_u + adj.p_divu) << "\n";
    std::cout << "  Rel mismatch:  " << adj.rel_mismatch << "\n";
    std::cout << "  Result:        " << (adj.passed ? "[PASS]" : "[FAIL]") << "\n";
    emit_qoi_adjoint(adj);
    record("Adjointness <grad(p),u> ≈ -<p,div(u)>", adj.passed);

    // Test 3B: Projection makes velocity divergence-free
    std::cout << "\n=== Test 3B: Projection Divergence-Free ===\n";
    std::cout << "  Checking: |div(u)| < 1e-4 after step\n\n";

    DivFreeResult divfree = test_projection_divfree();
    std::cout << "  div before step: " << divfree.div_before << "\n";
    std::cout << "  div after step:  " << divfree.div_after << " (limit: 1e-4)\n";
    std::cout << "  Result:          " << (divfree.passed ? "[PASS]" : "[FAIL]") << "\n";
    emit_qoi_divfree(divfree);
    record("Projection makes div(u) < 1e-4", divfree.passed);

    // Test 3C: Divergence reduces significantly
    std::cout << "\n=== Test 3C: Divergence Reduction ===\n";
    std::cout << "  Checking: div reduces by >4 orders of magnitude\n\n";

    DivReductionResult redux = test_divergence_reduction();
    std::cout << "  div before:      " << redux.div_before << "\n";
    std::cout << "  div after:       " << redux.div_after << "\n";
    std::cout << "  reduction factor:" << redux.reduction_factor << " (limit: 1e-4)\n";
    std::cout << "  Result:          " << (redux.passed ? "[PASS]" : "[FAIL]") << "\n";
    emit_qoi_reduction(redux);
    record("Divergence reduces >4 orders", redux.passed);
}

// ============================================================================
// Main
// ============================================================================
int main() {
    return harness::run("Operator Consistency Tests", []() {
        run_all_operator_tests();
    });
}
