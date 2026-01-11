/// @file test_energy_budget_channel.cpp
/// @brief Global energy/work budget identity test for forced channel
///
/// PURPOSE: One of the highest-trust CI checks. If any core piece is wrong
/// (signs, operators, scaling, pressure projection inconsistency), this blows up.
///
/// PHYSICS: For a forced incompressible flow, the KE balance is:
///   dKE/dt = power_in - dissipation
/// where:
///   KE = 0.5 * integral(|u|^2 dV)
///   power_in = integral(u * f dV)  [body force work]
///   dissipation = nu * integral(|grad(u)|^2 dV)
///
/// TEST: Run a short forced channel and check:
///   budget_err = |dKE/dt - (power_in - dissipation)| / max(|power_in|, |dissipation|, eps)
///   Require budget_err < 1e-2
///
/// EMITS QOI:
///   energy_budget: budget_err, power_in, dissipation, dKEdt

#include "test_harness.hpp"
#include "test_utilities.hpp"
#include "mesh.hpp"
#include "solver.hpp"
#include "config.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>

using namespace nncfd;
using namespace nncfd::test;
using nncfd::test::harness::record;

// ============================================================================
// Compute kinetic energy: KE = 0.5 * integral(|u|^2 dV)
// ============================================================================
static double compute_KE(const VectorField& vel, const Mesh& mesh) {
    double ke = 0.0;

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            // Interpolate velocities to cell center
            double u = 0.5 * (vel.u(i, j) + vel.u(i+1, j));
            double v = 0.5 * (vel.v(i, j) + vel.v(i, j+1));
            ke += 0.5 * (u*u + v*v) * mesh.dx * mesh.dy;
        }
    }

    return ke;
}

// ============================================================================
// Compute power input from body force: P_in = integral(u * f dV)
// For f = (fx, 0), this is integral(u * fx dV)
// ============================================================================
static double compute_power_in(const VectorField& vel, const Mesh& mesh, double fx) {
    double power = 0.0;

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double u = 0.5 * (vel.u(i, j) + vel.u(i+1, j));
            power += u * fx * mesh.dx * mesh.dy;
        }
    }

    return power;
}

// ============================================================================
// Compute dissipation: D = nu * integral(|grad(u)|^2 dV)
// |grad(u)|^2 = (du/dx)^2 + (du/dy)^2 + (dv/dx)^2 + (dv/dy)^2
// ============================================================================
static double compute_dissipation(const VectorField& vel, const Mesh& mesh, double nu) {
    double diss = 0.0;

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            // du/dx at cell center
            double dudx = (vel.u(i+1, j) - vel.u(i, j)) / mesh.dx;

            // du/dy at cell center (average from faces)
            double dudy_lo = (vel.u(i, j) - vel.u(i, j-1)) / mesh.dy;
            double dudy_hi = (vel.u(i, j+1) - vel.u(i, j)) / mesh.dy;
            double dudy = 0.5 * (dudy_lo + dudy_hi);
            // Handle boundaries
            if (j == mesh.j_begin()) {
                // No-slip: u at wall is 0
                dudy = (vel.u(i, j) - 0.0) / (0.5 * mesh.dy);
            } else if (j == mesh.j_end() - 1) {
                dudy = (0.0 - vel.u(i, j)) / (0.5 * mesh.dy);
            }

            // dv/dx at cell center
            double dvdx_lo = (vel.v(i, j) - vel.v(i-1, j)) / mesh.dx;
            double dvdx_hi = (vel.v(i+1, j) - vel.v(i, j)) / mesh.dx;
            double dvdx = 0.5 * (dvdx_lo + dvdx_hi);

            // dv/dy at cell center
            double dvdy = (vel.v(i, j+1) - vel.v(i, j)) / mesh.dy;

            double grad_sq = dudx*dudx + dudy*dudy + dvdx*dvdx + dvdy*dvdy;
            diss += nu * grad_sq * mesh.dx * mesh.dy;
        }
    }

    return diss;
}

// ============================================================================
// Result structure
// ============================================================================
struct EnergyBudgetResult {
    double KE_start;
    double KE_end;
    double dKEdt;
    double power_in;
    double dissipation;
    double budget_err;
    bool passed;
};

// ============================================================================
// Run energy budget test
// ============================================================================
EnergyBudgetResult run_energy_budget_test() {
    EnergyBudgetResult result;

    // Grid and physical parameters
    const int NX = 32;
    const int NY = 32;
    const double Lx = 2.0 * M_PI;
    const double Ly = 2.0;
    const double nu = 0.01;
    const double fx = 0.01;  // Body force in x-direction
    const double dt = 0.005;
    const int nsteps = 100;

    Mesh mesh;
    mesh.init_uniform(NX, NY, 0.0, Lx, 0.0, Ly);

    Config config;
    config.nu = nu;
    config.dt = dt;
    config.adaptive_dt = false;  // Fixed dt for clean budget calculation
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);
    solver.set_body_force(fx, 0.0);

    // Initialize with small perturbation (not zero, not steady)
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        double y = mesh.yc[j];
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            double x = mesh.xf[i];
            // Small sinusoidal perturbation
            solver.velocity().u(i, j) = 0.1 * std::sin(2.0 * M_PI * y / Ly);
        }
    }

    solver.sync_to_gpu();

    // Compute initial KE
    solver.sync_from_gpu();
    result.KE_start = compute_KE(solver.velocity(), mesh);

    // Accumulate power input and dissipation over the run
    double total_power_in = 0.0;
    double total_dissipation = 0.0;

    for (int step = 0; step < nsteps; ++step) {
        solver.sync_from_gpu();

        // Compute instantaneous rates (averaged over step)
        double P_in = compute_power_in(solver.velocity(), mesh, fx);
        double D = compute_dissipation(solver.velocity(), mesh, nu);

        total_power_in += P_in * dt;
        total_dissipation += D * dt;

        solver.step();
    }

    // Compute final KE
    solver.sync_from_gpu();
    result.KE_end = compute_KE(solver.velocity(), mesh);

    // Budget quantities
    double delta_KE = result.KE_end - result.KE_start;
    double T_total = nsteps * dt;

    result.dKEdt = delta_KE / T_total;
    result.power_in = total_power_in / T_total;
    result.dissipation = total_dissipation / T_total;

    // Budget error: |dKE/dt - (P_in - D)| / max(|P_in|, |D|, eps)
    double budget_imbalance = std::abs(result.dKEdt - (result.power_in - result.dissipation));
    double scale = std::max(std::abs(result.power_in), std::abs(result.dissipation));
    scale = std::max(scale, 1e-10);

    result.budget_err = budget_imbalance / scale;

    // Pass criterion
    result.passed = result.budget_err < 1e-2;

    return result;
}

// ============================================================================
// Emit QoI for CI tracking
// ============================================================================
static void emit_qoi_energy_budget(const EnergyBudgetResult& r) {
    std::cout << "QOI_JSON: {\"test\":\"energy_budget\""
              << ",\"budget_err\":" << harness::json_double(r.budget_err)
              << ",\"power_in\":" << harness::json_double(r.power_in)
              << ",\"dissipation\":" << harness::json_double(r.dissipation)
              << ",\"dKEdt\":" << harness::json_double(r.dKEdt)
              << "}\n" << std::flush;
}

// ============================================================================
// Main test function
// ============================================================================
void test_energy_budget() {
    std::cout << "\n--- Energy Budget Test (Forced Channel) ---\n\n";
    std::cout << "  Checking KE balance: dKE/dt = power_in - dissipation\n\n";

    EnergyBudgetResult r = run_energy_budget_test();

    // Print results
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "  KE (start):    " << r.KE_start << "\n";
    std::cout << "  KE (end):      " << r.KE_end << "\n";
    std::cout << "  dKE/dt:        " << r.dKEdt << "\n\n";

    std::cout << "  Power input:   " << r.power_in << "\n";
    std::cout << "  Dissipation:   " << r.dissipation << "\n";
    std::cout << "  P_in - D:      " << (r.power_in - r.dissipation) << "\n\n";

    std::cout << "  Budget error:  " << r.budget_err << " (limit: 1e-2)\n";
    std::cout << "  Result:        " << (r.passed ? "[PASS]" : "[FAIL]") << "\n\n";

    // Emit QoI
    emit_qoi_energy_budget(r);

    // Record results
    record("Energy budget balanced (err < 1e-2)", r.passed);
    record("Power input positive", r.power_in > 0);
    record("Dissipation positive", r.dissipation > 0);
}

// ============================================================================
// Main
// ============================================================================
int main() {
    return harness::run("Energy Budget Test", []() {
        test_energy_budget();
    });
}
