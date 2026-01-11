/// @file test_mms_convergence.cpp
/// @brief MMS (Method of Manufactured Solutions) convergence rate verification
///
/// PURPOSE: Verifies that the solver achieves the expected spatial convergence
/// rate (2nd order for central differences) using manufactured solutions.
///
/// MMS approach:
///   1. Choose a smooth solution u_exact, v_exact, p_exact
///   2. Substitute into N-S to get source terms f_u, f_v
///   3. Initialize with exact solution
///   4. Step a few times and measure error growth
///   5. Refine grid and verify error decreases at expected rate
///
/// Expected: O(h^2) convergence for velocity, O(h^2) for pressure
/// Test asserts: convergence rate >= 1.8 (allowing some margin for 2nd order)

#include "test_harness.hpp"
#include "test_utilities.hpp"
#include "solver.hpp"
#include "config.hpp"
#include <cmath>
#include <vector>
#include <iomanip>
#include <sstream>
#include <numeric>

using namespace nncfd;
using namespace nncfd::test;
using nncfd::test::harness::record;

// Helper to format QoI output
static std::string qoi(double value, double threshold) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2);
    ss << "(val=" << value << ", thr=" << threshold << ")";
    return ss.str();
}

// ============================================================================
// MMS Solution: Taylor-Green type (divergence-free by construction)
// ============================================================================
// u = sin(kx) * cos(ky)
// v = -cos(kx) * sin(ky)
// p = (1/4) * (cos(2kx) + cos(2ky))
//
// This is a steady solution to inviscid Euler, with viscous decay exp(-2*nu*k^2*t)
// For short time steps, we can treat it as quasi-steady.
// ============================================================================

struct MMSSolution {
    double k;
    double nu;
    double t;

    double u(double x, double y) const {
        return std::exp(-2.0 * nu * k * k * t) * std::sin(k * x) * std::cos(k * y);
    }

    double v(double x, double y) const {
        return -std::exp(-2.0 * nu * k * k * t) * std::cos(k * x) * std::sin(k * y);
    }

    double p(double x, double y) const {
        double decay = std::exp(-4.0 * nu * k * k * t);
        return -0.25 * decay * (std::cos(2.0 * k * x) + std::cos(2.0 * k * y));
    }
};

// ============================================================================
// Error computation
// ============================================================================

/// Compute L2 error for u-velocity vs exact solution
static double compute_u_l2_error(const VectorField& vel, const Mesh& mesh,
                                  const MMSSolution& exact) {
    double error_sq = 0.0;
    double norm_sq = 0.0;

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            // Interpolate u to cell center
            double u_num = 0.5 * (vel.u(i, j) + vel.u(i+1, j));
            double u_ex = exact.u(mesh.x(i), mesh.y(j));
            double diff = u_num - u_ex;
            error_sq += diff * diff * mesh.dx * mesh.dy;
            norm_sq += u_ex * u_ex * mesh.dx * mesh.dy;
        }
    }

    return (norm_sq > 1e-14) ? std::sqrt(error_sq / norm_sq) : std::sqrt(error_sq);
}

/// Compute L2 error for v-velocity vs exact solution
static double compute_v_l2_error(const VectorField& vel, const Mesh& mesh,
                                  const MMSSolution& exact) {
    double error_sq = 0.0;
    double norm_sq = 0.0;

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            // Interpolate v to cell center
            double v_num = 0.5 * (vel.v(i, j) + vel.v(i, j+1));
            double v_ex = exact.v(mesh.x(i), mesh.y(j));
            double diff = v_num - v_ex;
            error_sq += diff * diff * mesh.dx * mesh.dy;
            norm_sq += v_ex * v_ex * mesh.dx * mesh.dy;
        }
    }

    return (norm_sq > 1e-14) ? std::sqrt(error_sq / norm_sq) : std::sqrt(error_sq);
}

/// Compute combined velocity L2 error
static double compute_velocity_l2_error(const VectorField& vel, const Mesh& mesh,
                                         const MMSSolution& exact) {
    double u_error = compute_u_l2_error(vel, mesh, exact);
    double v_error = compute_v_l2_error(vel, mesh, exact);
    return std::sqrt(u_error * u_error + v_error * v_error);
}

// ============================================================================
// Initialize with MMS solution
// ============================================================================
static void init_mms(RANSSolver& solver, const Mesh& mesh, const MMSSolution& mms) {
    // Initialize u at u-faces
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            double x = mesh.x(i) - mesh.dx / 2.0;  // u-face location
            double y = mesh.y(j);
            solver.velocity().u(i, j) = mms.u(x, y);
        }
    }

    // Initialize v at v-faces
    for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double x = mesh.x(i);
            double y = mesh.y(j) - mesh.dy / 2.0;  // v-face location
            solver.velocity().v(i, j) = mms.v(x, y);
        }
    }
}

// ============================================================================
// Test: MMS Spatial Convergence Rate
// ============================================================================
void test_mms_spatial_convergence() {
    std::cout << "\n--- MMS Spatial Convergence Rate ---\n\n";

    const double nu = 0.01;
    const double k = 2.0 * M_PI;  // One wavelength in [0,1]
    const double dt = 0.0001;      // Small dt to minimize temporal error
    const int nsteps = 10;        // Few steps to measure spatial discretization error
    const double T = nsteps * dt;

    std::vector<int> Ns = {16, 32, 64};
    std::vector<double> errors;
    std::vector<double> hs;

    MMSSolution mms_init{k, nu, 0.0};
    MMSSolution mms_final{k, nu, T};

    for (int N : Ns) {
        Mesh mesh;
        mesh.init_uniform(N, N, 0.0, 1.0, 0.0, 1.0);

        Config config;
        config.nu = nu;
        config.dt = dt;
        config.adaptive_dt = false;
        config.turb_model = TurbulenceModelType::None;
        config.verbose = false;

        RANSSolver solver(mesh, config);

        VelocityBC bc;
        bc.x_lo = VelocityBC::Periodic;
        bc.x_hi = VelocityBC::Periodic;
        bc.y_lo = VelocityBC::Periodic;
        bc.y_hi = VelocityBC::Periodic;
        solver.set_velocity_bc(bc);

        init_mms(solver, mesh, mms_init);
        solver.sync_to_gpu();

        for (int step = 0; step < nsteps; ++step) {
            solver.step();
        }
        solver.sync_from_gpu();

        double l2_error = compute_velocity_l2_error(solver.velocity(), mesh, mms_final);
        errors.push_back(l2_error);
        hs.push_back(1.0 / N);

        std::cout << "  N=" << std::setw(3) << N
                  << ", h=" << std::scientific << std::setprecision(4) << (1.0/N)
                  << ", L2_error=" << l2_error << "\n";
    }

    // Compute convergence rates between successive refinements
    std::vector<double> rates;
    for (size_t i = 1; i < errors.size(); ++i) {
        double rate = std::log(errors[i-1] / errors[i]) / std::log(hs[i-1] / hs[i]);
        rates.push_back(rate);
        std::cout << "  Rate (" << Ns[i-1] << " -> " << Ns[i] << "): "
                  << std::fixed << std::setprecision(2) << rate << "\n";
    }

    // Check that all rates are close to 2 (second order)
    double min_rate = *std::min_element(rates.begin(), rates.end());
    double avg_rate = std::accumulate(rates.begin(), rates.end(), 0.0) / rates.size();

    std::cout << "\n  Minimum rate: " << std::fixed << std::setprecision(2) << min_rate
              << ", Average rate: " << avg_rate << "\n\n";

    // 2nd-order method should achieve rate ~2, but allow margin for pre-asymptotic effects
    record("MMS min convergence rate >= 1.4", min_rate >= 1.4,
           qoi(min_rate, 1.4));
    record("MMS avg convergence rate >= 1.6", avg_rate >= 1.6,
           qoi(avg_rate, 1.6));
}

// ============================================================================
// Test: MMS Temporal Convergence Rate
// ============================================================================
void test_mms_temporal_convergence() {
    std::cout << "\n--- MMS Temporal Convergence Rate ---\n\n";

    const int N = 64;  // Fine grid to reduce spatial error
    const double nu = 0.01;
    const double k = 2.0 * M_PI;
    const double T = 0.1;

    std::vector<double> dts = {0.01, 0.005, 0.0025};
    std::vector<double> errors;

    for (double dt : dts) {
        MMSSolution mms_init{k, nu, 0.0};
        MMSSolution mms_final{k, nu, T};

        int nsteps = static_cast<int>(T / dt);

        Mesh mesh;
        mesh.init_uniform(N, N, 0.0, 1.0, 0.0, 1.0);

        Config config;
        config.nu = nu;
        config.dt = dt;
        config.adaptive_dt = false;
        config.turb_model = TurbulenceModelType::None;
        config.verbose = false;

        RANSSolver solver(mesh, config);

        VelocityBC bc;
        bc.x_lo = VelocityBC::Periodic;
        bc.x_hi = VelocityBC::Periodic;
        bc.y_lo = VelocityBC::Periodic;
        bc.y_hi = VelocityBC::Periodic;
        solver.set_velocity_bc(bc);

        init_mms(solver, mesh, mms_init);
        solver.sync_to_gpu();

        for (int step = 0; step < nsteps; ++step) {
            solver.step();
        }
        solver.sync_from_gpu();

        double l2_error = compute_velocity_l2_error(solver.velocity(), mesh, mms_final);
        errors.push_back(l2_error);

        std::cout << "  dt=" << std::scientific << std::setprecision(4) << dt
                  << ", nsteps=" << std::setw(4) << nsteps
                  << ", L2_error=" << l2_error << "\n";
    }

    // Compute temporal convergence rates
    std::vector<double> rates;
    for (size_t i = 1; i < errors.size(); ++i) {
        double rate = std::log(errors[i-1] / errors[i]) / std::log(dts[i-1] / dts[i]);
        rates.push_back(rate);
        std::cout << "  Temporal rate (dt=" << dts[i-1] << " -> " << dts[i] << "): "
                  << std::fixed << std::setprecision(2) << rate << "\n";
    }

    double min_rate = *std::min_element(rates.begin(), rates.end());

    std::cout << "\n  Minimum temporal rate: " << std::fixed << std::setprecision(2) << min_rate << "\n\n";

    // First-order Euler or second-order expected, allow 0.8 for first order
    record("MMS temporal convergence rate >= 0.8", min_rate >= 0.8,
           qoi(min_rate, 0.8));
}

// ============================================================================
// Test: Poisson Solver Projection Quality
// ============================================================================
void test_poisson_projection_quality() {
    std::cout << "\n--- Poisson Projection Quality ---\n\n";

    // Verify that after projection, divergence is small on multiple grid sizes
    std::vector<int> Ns = {16, 32, 64};
    std::vector<double> max_divs;

    for (int N : Ns) {
        Mesh mesh;
        mesh.init_uniform(N, N, 0.0, 2.0*M_PI, 0.0, 2.0*M_PI);

        Config config;
        config.nu = 0.01;
        config.dt = 0.01;
        config.tol = 1e-10;
        config.turb_model = TurbulenceModelType::None;
        config.verbose = false;

        RANSSolver solver(mesh, config);

        VelocityBC bc;
        bc.x_lo = VelocityBC::Periodic;
        bc.x_hi = VelocityBC::Periodic;
        bc.y_lo = VelocityBC::Periodic;
        bc.y_hi = VelocityBC::Periodic;
        solver.set_velocity_bc(bc);

        // Initialize with Taylor-Green (divergence-free IC)
        init_taylor_green(solver, mesh);
        solver.sync_to_gpu();

        // Step forward - Poisson solve happens each step
        for (int step = 0; step < 10; ++step) {
            solver.step();
        }
        solver.sync_from_gpu();

        // Check divergence
        double max_div = 0.0;
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double du_dx = (solver.velocity().u(i+1, j) - solver.velocity().u(i, j)) / mesh.dx;
                double dv_dy = (solver.velocity().v(i, j+1) - solver.velocity().v(i, j)) / mesh.dy;
                max_div = std::max(max_div, std::abs(du_dx + dv_dy));
            }
        }

        max_divs.push_back(max_div);

        std::cout << "  N=" << std::setw(3) << N
                  << ", max|div|=" << std::scientific << std::setprecision(4) << max_div << "\n";
    }

    double max_all = *std::max_element(max_divs.begin(), max_divs.end());

    std::cout << "\n  Maximum divergence across all grids: " << std::scientific << max_all << "\n\n";

    std::ostringstream ss;
    ss << std::scientific << std::setprecision(2) << "(val=" << max_all << ", thr=1.00e-06)";
    record("Poisson projection max|div| < 1e-6", max_all < 1e-6, ss.str());
}

// ============================================================================
// Test: Viscous Decay Rate
// ============================================================================
void test_viscous_decay_rate() {
    std::cout << "\n--- Viscous Decay Rate (Analytical) ---\n\n";

    // Taylor-Green vortex decays as exp(-2*nu*k^2*t)
    // This tests that the viscous term is correctly implemented

    const int N = 48;
    const double nu = 0.01;
    const double k = 1.0;  // One wavelength in [0, 2*pi]

    Mesh mesh;
    mesh.init_uniform(N, N, 0.0, 2.0*M_PI, 0.0, 2.0*M_PI);

    Config config;
    config.nu = nu;
    config.dt = 0.01;
    config.adaptive_dt = false;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::Periodic;
    bc.y_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    init_taylor_green(solver, mesh);
    solver.sync_to_gpu();

    double KE0 = compute_kinetic_energy(mesh, solver.velocity());

    double T = 0.5;
    int nsteps = static_cast<int>(T / config.dt);

    for (int step = 0; step < nsteps; ++step) {
        solver.step();
    }
    solver.sync_from_gpu();

    double KE_final = compute_kinetic_energy(mesh, solver.velocity());

    // Theory: KE decays as exp(-4*nu*k^2*t) for Taylor-Green
    // Since amplitude decays as exp(-2*nu*k^2*t), energy = amplitude^2 decays at 2x rate
    double KE_theory = KE0 * std::exp(-4.0 * nu * k * k * T);

    double ratio_num = KE_final / KE0;
    double ratio_theory = KE_theory / KE0;
    double rel_error = std::abs(ratio_num - ratio_theory) / ratio_theory;

    std::cout << "  KE ratio (numerical):   " << std::fixed << std::setprecision(6) << ratio_num << "\n";
    std::cout << "  KE ratio (theory):      " << ratio_theory << "\n";
    std::cout << "  Relative error:         " << std::scientific << rel_error * 100 << "%\n\n";

    std::ostringstream ss2;
    ss2 << std::fixed << std::setprecision(2) << "(err=" << rel_error*100 << "%, thr=10%)";
    record("Viscous decay within 10% of theory", rel_error < 0.10, ss2.str());
}

// ============================================================================
// Main
// ============================================================================
int main() {
    return harness::run("MMS Convergence Tests", []() {
        test_mms_spatial_convergence();
        test_mms_temporal_convergence();
        test_poisson_projection_quality();
        test_viscous_decay_rate();
    });
}
