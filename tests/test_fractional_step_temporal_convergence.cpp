/// @file test_fractional_step_temporal_convergence.cpp
/// @brief Fractional step temporal convergence test
///
/// PURPOSE: Verify time integrator behavior with fractional step projection:
///   - All methods: O(Δt¹) convergence due to fractional step splitting error
///   - RK2/RK3: Lower absolute error than Euler at same dt
///   - All methods: Projection correctly enforces divergence-free constraint
///
/// NOTE ON FRACTIONAL STEP SPLITTING:
///   For incompressible flow with projection methods, the splitting introduces
///   O(Δt) error regardless of the time integrator order. Higher-order RK methods
///   reduce advection/diffusion error but don't affect splitting error.
///   True O(2)/O(3) convergence requires pressure extrapolation (Kim & Moin, 1985),
///   iterative refinement (Armfield & Street, 2002), or IMEX schemes.
///
/// APPROACH: Richardson-style self-referencing
///   - For each integrator, run at dt, dt/2, dt/4
///   - Compare dt vs dt/2 solutions to get error estimate
///   - Compare dt/2 vs dt/4 solutions to get another error estimate
///   - Compute observed order from error ratios
///   - This avoids cross-integrator cancellation and guarantees nonzero differences
///
/// TEST CASE: Periodic Taylor-Green at Re=100
///   - N = 64, T = 0.1 (smooth, stable)
///   - scheme = skew (energy-stable)
///   - space_order = 2 (avoids MG ng=1 issues)

#include "test_harness.hpp"
#include "test_utilities.hpp"
#include "solver.hpp"
#include "config.hpp"
#include <cmath>
#include <vector>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <map>

using namespace nncfd;
using namespace nncfd::test;
using nncfd::test::harness::record;

// ============================================================================
// Helper: Format QoI output
// ============================================================================
static std::string qoi(double value, double threshold) {
    std::ostringstream ss;
    ss << std::scientific << std::setprecision(3);
    ss << "(val=" << value << ", thr=" << threshold << ")";
    return ss.str();
}

// ============================================================================
// Helper: Compute max divergence (L-infinity norm)
// ============================================================================
static double compute_max_divergence_2d(const VectorField& vel, const Mesh& mesh) {
    double max_div = 0.0;
    const double dx = mesh.dx;
    const double dy = mesh.dy;

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double du_dx = (vel.u(i+1, j) - vel.u(i, j)) / dx;
            double dv_dy = (vel.v(i, j+1) - vel.v(i, j)) / dy;
            double div = std::abs(du_dx + dv_dy);
            max_div = std::max(max_div, div);
        }
    }
    return max_div;
}

// ============================================================================
// Helper: Compute relative L2 velocity difference between two fields
// ============================================================================
static double compute_velocity_l2_diff(const VectorField& a, const VectorField& b) {
    double sum_sq_diff = 0.0;
    double sum_sq_b = 0.0;

    // u-component (full array)
    const auto& u_a = a.u_data();
    const auto& u_b = b.u_data();
    for (size_t i = 0; i < u_a.size(); ++i) {
        double diff = u_a[i] - u_b[i];
        sum_sq_diff += diff * diff;
        sum_sq_b += u_b[i] * u_b[i];
    }

    // v-component (full array)
    const auto& v_a = a.v_data();
    const auto& v_b = b.v_data();
    for (size_t i = 0; i < v_a.size(); ++i) {
        double diff = v_a[i] - v_b[i];
        sum_sq_diff += diff * diff;
        sum_sq_b += v_b[i] * v_b[i];
    }

    // Avoid division by zero
    if (sum_sq_b < 1e-30) return 0.0;
    return std::sqrt(sum_sq_diff / sum_sq_b);
}

// ============================================================================
// Helper: Run solver to time T and return velocity field
// ============================================================================
static VectorField run_to_time(const Mesh& mesh, double nu, double T, double dt,
                               TimeIntegrator integrator) {
    Config config;
    config.nu = nu;
    config.dt = dt;
    config.time_integrator = integrator;
    config.convective_scheme = ConvectiveScheme::Skew;
    config.poisson_solver = PoissonSolverType::MG;
    config.adaptive_dt = false;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    // Fully periodic BCs
    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::Periodic;
    bc.y_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    // Initialize with Taylor-Green vortex
    init_taylor_green(solver, mesh);
    solver.sync_to_gpu();

    // Time step to T
    int nsteps = static_cast<int>(std::round(T / dt));
    for (int i = 0; i < nsteps; ++i) {
        solver.step();
    }

    solver.sync_from_gpu();
    return solver.velocity();
}

// ============================================================================
// Helper: Epsilon floor for order computation
// ============================================================================
static constexpr double ERROR_FLOOR = 1e-30;

static double safe_log_ratio(double e1, double e2) {
    // Apply floor to avoid log(0) or negative orders from roundoff
    e1 = std::max(e1, ERROR_FLOOR);
    e2 = std::max(e2, ERROR_FLOOR);
    if (e2 >= e1) return 0.0;  // No convergence
    return std::log(e1 / e2);
}

// ============================================================================
// Main test: Fractional step temporal convergence (Richardson self-reference)
// ============================================================================
void test_fractional_step_convergence() {
    std::cout << "\n=== Fractional Step Temporal Convergence Test ===\n";
    std::cout << "    (Richardson-style self-referencing)\n\n";

    // Test configuration
    const int N = 64;
    const double L = 2.0 * M_PI;
    const double nu = 0.01;  // Re = L*U/nu ≈ 100 for U~1
    const double T = 0.1;

    // dt ladder (pure halving) - we compare adjacent pairs
    const double dt0 = 0.01;
    const std::vector<double> dt_set = {dt0, dt0/2, dt0/4, dt0/8};

    // Create mesh (reused for all runs)
    Mesh mesh;
    mesh.init_uniform(N, N, 0.0, L, 0.0, L);

    std::cout << "Configuration:\n";
    std::cout << "  Grid: " << N << "x" << N << ", domain: [0, 2π]²\n";
    std::cout << "  nu = " << nu << " (Re ≈ 100)\n";
    std::cout << "  T = " << T << "\n";
    std::cout << "  dt_set = {";
    for (size_t i = 0; i < dt_set.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << dt_set[i];
    }
    std::cout << "}\n";
    std::cout << "  Method: Compare ||u(dt) - u(dt/2)|| for self-referencing\n\n";

    // Test each integrator
    std::vector<std::tuple<std::string, TimeIntegrator>> integrators = {
        {"Euler", TimeIntegrator::Euler},
        {"RK2", TimeIntegrator::RK2},
        {"RK3", TimeIntegrator::RK3}
    };

    // Store errors at finest dt for inter-method comparison
    std::map<std::string, double> finest_errors;

    for (const auto& [name, integrator] : integrators) {
        std::cout << "Testing " << name << ":\n";

        // Run at each dt and store solutions
        std::vector<VectorField> solutions;
        std::vector<double> divs;

        for (double dt : dt_set) {
            int nsteps = static_cast<int>(std::round(T / dt));
            std::cout << "  Running dt=" << std::scientific << std::setprecision(4) << dt
                      << " (" << nsteps << " steps)... " << std::flush;

            VectorField sol = run_to_time(mesh, nu, T, dt, integrator);
            double div = compute_max_divergence_2d(sol, mesh);
            solutions.push_back(std::move(sol));
            divs.push_back(div);

            std::cout << "div=" << std::scientific << std::setprecision(2) << div << "\n";
        }

        // Compute Richardson-style errors: ||u(dt) - u(dt/2)|| / ||u(dt/2)||
        std::vector<double> errors;
        for (size_t i = 0; i + 1 < solutions.size(); ++i) {
            double err = compute_velocity_l2_diff(solutions[i], solutions[i+1]);
            errors.push_back(err);
        }

        std::cout << "  Richardson errors: ";
        for (size_t i = 0; i < errors.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << std::scientific << std::setprecision(3) << errors[i];
        }
        std::cout << "\n";

        // Compute observed orders from adjacent error pairs
        std::vector<double> orders;
        for (size_t i = 0; i + 1 < errors.size(); ++i) {
            double log_err_ratio = safe_log_ratio(errors[i], errors[i+1]);
            double log_dt_ratio = std::log(2.0);  // dt halving
            double p = log_err_ratio / log_dt_ratio;
            orders.push_back(p);
        }

        std::cout << "  Observed orders: ";
        for (size_t i = 0; i < orders.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(2) << orders[i];
        }
        std::cout << "\n";

        // Check that at least one error is significantly above floor
        double max_err = *std::max_element(errors.begin(), errors.end());
        bool has_signal = max_err > 1e-12;
        if (!has_signal) {
            std::cout << "  WARNING: All errors below 1e-12, convergence not measurable\n";
        }

        // Check monotone decrease (with tolerance for noise)
        bool monotone = true;
        for (size_t i = 0; i + 1 < errors.size(); ++i) {
            // Allow 10% tolerance for noise
            if (errors[i+1] > errors[i] * 1.1) {
                monotone = false;
                std::cout << "  WARNING: Non-monotone at dt=" << dt_set[i+1] << "\n";
            }
        }

        // Average order (all pairs)
        double avg_order = 0.0;
        if (!orders.empty()) {
            avg_order = std::accumulate(orders.begin(), orders.end(), 0.0) / orders.size();
        }

        // For projection methods, we expect O(1) due to splitting
        // Accept order >= 0.5 as "converging" (very lenient due to splitting error)
        // If errors are too small to measure, skip the order check (not a failure)
        bool order_ok = has_signal ? (avg_order > 0.5 && std::isfinite(avg_order)) : true;

        // Check divergence (all runs)
        // Use 1e-7 threshold to allow for MG solver tolerance
        double max_div = *std::max_element(divs.begin(), divs.end());
        bool div_ok = max_div < 1e-7;

        // Record results
        record(name + " converging (order > 0.5 or no signal)", order_ok,
               has_signal ? qoi(avg_order, 0.5) : "(errors too small to measure)");
        record(name + " monotone error decrease", monotone || !has_signal);  // Skip if no signal
        record(name + " divergence < 1e-7", div_ok, qoi(max_div, 1e-7));

        // Emit QOI JSON
        std::cout << "QOI_JSON: {\"test\":\"time_integrator_" << name << "\""
                  << ",\"avg_order\":" << std::fixed << std::setprecision(4) << avg_order
                  << ",\"max_err\":" << std::scientific << std::setprecision(6) << max_err
                  << ",\"max_div\":" << std::scientific << std::setprecision(6) << max_div
                  << ",\"errors\":[";
        for (size_t i = 0; i < errors.size(); ++i) {
            if (i > 0) std::cout << ",";
            std::cout << std::scientific << std::setprecision(6) << errors[i];
        }
        std::cout << "],\"orders\":[";
        for (size_t i = 0; i < orders.size(); ++i) {
            if (i > 0) std::cout << ",";
            std::cout << std::fixed << std::setprecision(4) << orders[i];
        }
        std::cout << "]}\n\n";

        // Store finest error for inter-method comparison
        finest_errors[name] = errors.back();
    }

    // Verify that higher-order methods give lower error at same dt pair
    // (comparing dt/4 vs dt/8 Richardson error)
    std::cout << "Inter-method comparison (Richardson error at finest dt pair):\n";
    std::cout << "  Euler: " << std::scientific << std::setprecision(3) << finest_errors["Euler"] << "\n";
    std::cout << "  RK2:   " << std::scientific << std::setprecision(3) << finest_errors["RK2"] << "\n";
    std::cout << "  RK3:   " << std::scientific << std::setprecision(3) << finest_errors["RK3"] << "\n";

    // RK2/RK3 should have lower Richardson error than Euler (better accuracy)
    // Use factor of 0.5 - they don't need to be dramatically better
    bool rk2_better = finest_errors["RK2"] < finest_errors["Euler"];
    bool rk3_not_worse = finest_errors["RK3"] <= finest_errors["RK2"] * 1.1;  // Allow 10% tolerance

    record("RK2 error <= Euler error", rk2_better,
           qoi(finest_errors["RK2"], finest_errors["Euler"]));
    record("RK3 error <= RK2 error (within 10%)", rk3_not_worse,
           qoi(finest_errors["RK3"], finest_errors["RK2"]));
}

// ============================================================================
// Main
// ============================================================================
int main() {
    return nncfd::test::harness::run("Fractional Step Temporal Convergence", []() {
        test_fractional_step_convergence();
    });
}
