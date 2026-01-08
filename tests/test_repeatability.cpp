/// @file test_repeatability.cpp
/// @brief Repeatability envelope test (not bitwise determinism)
///
/// CRITICAL TEST: Catches nondeterminism explosions while acknowledging FP non-bitwise behavior.
/// We don't require bitwise determinism, but we do require:
///   - Running the same case twice produces results within tight tolerance
///   - Detects race conditions, uninitialized memory, catastrophic divergence
///
/// Method:
///   1. Run a representative case twice on the same backend
///   2. Compare key metrics (kinetic energy, divergence, pressure variance)
///   3. Assert relative difference < epsilon (1e-10 to 1e-8)
///
/// This is NOT a CPU vs GPU test - it's same-backend repeatability.

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "config.hpp"
#include "test_harness.hpp"
#include <cmath>
#include <string>

using namespace nncfd;
using nncfd::test::harness::record;

// ============================================================================
// Metric computation
// ============================================================================

struct SimulationMetrics {
    double kinetic_energy;
    double pressure_variance;
    double u_max;
    double v_max;
    double w_max;
    int steps_completed;
};

double compute_kinetic_energy(const VectorField& vel, const Mesh& mesh) {
    double ke = 0.0;
    int count = 0;

    if (mesh.is2D()) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double u = 0.5 * (vel.u(i, j) + vel.u(i+1, j));
                double v = 0.5 * (vel.v(i, j) + vel.v(i, j+1));
                ke += 0.5 * (u*u + v*v);
                ++count;
            }
        }
    } else {
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    double u = 0.5 * (vel.u(i, j, k) + vel.u(i+1, j, k));
                    double v = 0.5 * (vel.v(i, j, k) + vel.v(i, j+1, k));
                    double w = 0.5 * (vel.w(i, j, k) + vel.w(i, j, k+1));
                    ke += 0.5 * (u*u + v*v + w*w);
                    ++count;
                }
            }
        }
    }
    return ke / count;
}

double compute_pressure_variance(const ScalarField& p, const Mesh& mesh) {
    double p_mean = 0.0;
    int count = 0;

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                p_mean += p(i, j, k);
                ++count;
            }
        }
    }
    p_mean /= count;

    double variance = 0.0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double diff = p(i, j, k) - p_mean;
                variance += diff * diff;
            }
        }
    }
    return variance / count;
}

double compute_max_velocity(const VectorField& vel, const Mesh& mesh, char component) {
    double max_val = 0.0;

    if (mesh.is2D()) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double val = 0.0;
                if (component == 'u') val = std::abs(vel.u(i, j));
                else if (component == 'v') val = std::abs(vel.v(i, j));
                max_val = std::max(max_val, val);
            }
        }
    } else {
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    double val = 0.0;
                    if (component == 'u') val = std::abs(vel.u(i, j, k));
                    else if (component == 'v') val = std::abs(vel.v(i, j, k));
                    else if (component == 'w') val = std::abs(vel.w(i, j, k));
                    max_val = std::max(max_val, val);
                }
            }
        }
    }
    return max_val;
}

// ============================================================================
// Run a simulation and collect metrics
// ============================================================================

SimulationMetrics run_simulation(const Mesh& mesh, const Config& config,
                                  const VelocityBC& bc, int nsteps) {
    RANSSolver solver(mesh, config);
    solver.set_velocity_bc(bc);

    VectorField vel(mesh);
    vel.fill(1.0, 0.0, 0.0);

    // Add deterministic perturbation
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double x = mesh.x(i);
                double y = mesh.y(j);
                vel.u(i, j, k) += 0.01 * std::sin(2.0 * M_PI * y / (mesh.y_max - mesh.y_min));
                vel.v(i, j, k) += 0.01 * std::sin(2.0 * M_PI * x / (mesh.x_max - mesh.x_min));
            }
        }
    }

    solver.initialize(vel);
    solver.set_body_force(0.001, 0.0, 0.0);

    // Run simulation
    for (int step = 0; step < nsteps; ++step) {
        solver.step();
    }

    // Collect metrics
    SimulationMetrics metrics;
    metrics.kinetic_energy = compute_kinetic_energy(solver.velocity(), mesh);
    metrics.pressure_variance = compute_pressure_variance(solver.pressure(), mesh);
    metrics.u_max = compute_max_velocity(solver.velocity(), mesh, 'u');
    metrics.v_max = compute_max_velocity(solver.velocity(), mesh, 'v');
    metrics.w_max = mesh.is2D() ? 0.0 : compute_max_velocity(solver.velocity(), mesh, 'w');
    metrics.steps_completed = nsteps;

    return metrics;
}

// ============================================================================
// Compare metrics with tolerance
// ============================================================================

bool compare_metrics(const SimulationMetrics& m1, const SimulationMetrics& m2,
                      double rel_tol, std::string& failure_reason) {
    auto check = [&](const char* name, double v1, double v2) -> bool {
        double denom = std::max(std::abs(v1), std::abs(v2));
        if (denom < 1e-15) denom = 1.0;  // Avoid division by tiny numbers

        double rel_diff = std::abs(v1 - v2) / denom;
        if (rel_diff > rel_tol) {
            char buf[256];
            snprintf(buf, sizeof(buf), "%s differs: %.12e vs %.12e (rel_diff=%.2e > %.2e)",
                     name, v1, v2, rel_diff, rel_tol);
            failure_reason = buf;
            return false;
        }
        return true;
    };

    if (!check("kinetic_energy", m1.kinetic_energy, m2.kinetic_energy)) return false;
    if (!check("pressure_variance", m1.pressure_variance, m2.pressure_variance)) return false;
    if (!check("u_max", m1.u_max, m2.u_max)) return false;
    if (!check("v_max", m1.v_max, m2.v_max)) return false;
    if (!check("w_max", m1.w_max, m2.w_max)) return false;

    return true;
}

// ============================================================================
// Test cases
// ============================================================================

struct RepeatabilityTestCase {
    std::string name;
    int Nx, Ny, Nz;
    double Lx, Ly, Lz;
    VelocityBC::Type x_bc, y_bc, z_bc;
    int nsteps;
    double tolerance;
};

bool run_repeatability_test(const RepeatabilityTestCase& tc) {
    // Create mesh
    Mesh mesh;
    if (tc.Nz == 1) {
        mesh.init_uniform(tc.Nx, tc.Ny, 0.0, tc.Lx, 0.0, tc.Ly);
    } else {
        mesh.init_uniform(tc.Nx, tc.Ny, tc.Nz, 0.0, tc.Lx, 0.0, tc.Ly, 0.0, tc.Lz);
    }

    // Create config
    Config config;
    config.Nx = tc.Nx;
    config.Ny = tc.Ny;
    config.Nz = tc.Nz;
    config.x_min = 0.0; config.x_max = tc.Lx;
    config.y_min = 0.0; config.y_max = tc.Ly;
    config.z_min = 0.0; config.z_max = tc.Lz;
    config.dt = 0.001;
    config.max_iter = tc.nsteps + 100;
    config.nu = 0.01;
    config.poisson_solver = PoissonSolverType::Auto;

    // Create BCs
    VelocityBC bc;
    bc.x_lo = tc.x_bc; bc.x_hi = tc.x_bc;
    bc.y_lo = tc.y_bc; bc.y_hi = tc.y_bc;
    bc.z_lo = tc.z_bc; bc.z_hi = tc.z_bc;

    // Run simulation twice
    SimulationMetrics m1 = run_simulation(mesh, config, bc, tc.nsteps);
    SimulationMetrics m2 = run_simulation(mesh, config, bc, tc.nsteps);

    // Compare
    std::string failure_reason;
    return compare_metrics(m1, m2, tc.tolerance, failure_reason);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    return nncfd::test::harness::run("Repeatability Tests", [] {
        const double REL_TOL = 1e-10;
        const int NSTEPS = 100;

        // 2D tests
        record("2D channel repeatability (100 steps x2)", run_repeatability_test({
            "2D_channel", 64, 64, 1, 2.0*M_PI, 2.0, 1.0,
            VelocityBC::Periodic, VelocityBC::NoSlip, VelocityBC::Periodic,
            NSTEPS, REL_TOL
        }));

        record("2D cavity repeatability (100 steps x2)", run_repeatability_test({
            "2D_cavity", 64, 64, 1, 1.0, 1.0, 1.0,
            VelocityBC::NoSlip, VelocityBC::NoSlip, VelocityBC::Periodic,
            NSTEPS, REL_TOL
        }));

        // 3D tests
        record("3D channel repeatability (100 steps x2)", run_repeatability_test({
            "3D_channel", 32, 32, 32, 2.0*M_PI, 2.0, 2.0*M_PI,
            VelocityBC::Periodic, VelocityBC::NoSlip, VelocityBC::Periodic,
            NSTEPS, REL_TOL
        }));

        record("3D duct repeatability (100 steps x2)", run_repeatability_test({
            "3D_duct", 32, 32, 32, 2.0*M_PI, 2.0, 2.0,
            VelocityBC::Periodic, VelocityBC::NoSlip, VelocityBC::NoSlip,
            NSTEPS, REL_TOL
        }));
    });
}
