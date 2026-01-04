/// @file test_projection_invariants.cpp
/// @brief Projection method invariants test for full time-stepper coupling
///
/// CRITICAL TEST: Validates that the solver correctly couples Poisson solver
/// selection with time integration. Catches cases where:
///   - Poisson solver works in isolation but breaks in projection method
///   - Solver switching happens unexpectedly mid-run
///   - Divergence accumulates over time
///   - Kinetic energy blows up
///
/// Method:
///   1. Run 50-100 steps on small 2D and 3D cases
///   2. At each step, check invariants:
///      - Divergence remains bounded
///      - Kinetic energy is finite and doesn't blow up
///      - Pressure is finite
///      - Solver selection remains stable
///
/// This test complements endurance_stability (which focuses on NaN detection)
/// by checking physical invariants.

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "config.hpp"
#include <iostream>
#include <cmath>
#include <iomanip>
#include <vector>
#include <string>

using namespace nncfd;

// ============================================================================
// Helper functions
// ============================================================================

double compute_max_divergence(const VectorField& vel, const Mesh& mesh) {
    double max_div = 0.0;

    if (mesh.is2D()) {
        double dx = (mesh.x_max - mesh.x_min) / mesh.Nx;
        double dy = (mesh.y_max - mesh.y_min) / mesh.Ny;

        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double du_dx = (vel.u(i+1, j) - vel.u(i, j)) / dx;
                double dv_dy = (vel.v(i, j+1) - vel.v(i, j)) / dy;
                double div = std::abs(du_dx + dv_dy);
                max_div = std::max(max_div, div);
            }
        }
    } else {
        double dx = (mesh.x_max - mesh.x_min) / mesh.Nx;
        double dy = (mesh.y_max - mesh.y_min) / mesh.Ny;
        double dz = (mesh.z_max - mesh.z_min) / mesh.Nz;

        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    double du_dx = (vel.u(i+1, j, k) - vel.u(i, j, k)) / dx;
                    double dv_dy = (vel.v(i, j+1, k) - vel.v(i, j, k)) / dy;
                    double dw_dz = (vel.w(i, j, k+1) - vel.w(i, j, k)) / dz;
                    double div = std::abs(du_dx + dv_dy + dw_dz);
                    max_div = std::max(max_div, div);
                }
            }
        }
    }
    return max_div;
}

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

double compute_max_pressure(const ScalarField& p, const Mesh& mesh) {
    double max_p = 0.0;

    if (mesh.is2D()) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                max_p = std::max(max_p, std::abs(p(i, j)));
            }
        }
    } else {
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    max_p = std::max(max_p, std::abs(p(i, j, k)));
                }
            }
        }
    }
    return max_p;
}

bool is_finite(double x) {
    return std::isfinite(x) && !std::isnan(x);
}

// ============================================================================
// Test case structure
// ============================================================================

struct InvariantTestCase {
    std::string name;
    int Nx, Ny, Nz;
    double Lx, Ly, Lz;
    VelocityBC::Type x_bc, y_bc, z_bc;
    int nsteps;
    double div_bound;      // Maximum allowed divergence
    double ke_growth_max;  // Maximum allowed KE growth factor
    bool zero_init;        // Initialize with zero velocity (for enclosed cavities)
};

struct InvariantTestResult {
    bool passed;
    int steps_completed;
    double initial_ke;
    double final_ke;
    double max_div_seen;
    double max_pressure_seen;
    std::string initial_solver;
    std::string final_solver;
    std::string failure_reason;
};

// ============================================================================
// Run test
// ============================================================================

InvariantTestResult run_invariant_test(const InvariantTestCase& tc) {
    InvariantTestResult result;
    result.passed = true;
    result.steps_completed = 0;
    result.failure_reason = "";

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
    config.verbose = false;

    RANSSolver solver(mesh, config);

    // Set BCs
    VelocityBC bc;
    bc.x_lo = tc.x_bc; bc.x_hi = tc.x_bc;
    bc.y_lo = tc.y_bc; bc.y_hi = tc.y_bc;
    bc.z_lo = tc.z_bc; bc.z_hi = tc.z_bc;
    solver.set_velocity_bc(bc);

    // Initialize velocity field
    VectorField vel(mesh);
    if (tc.zero_init) {
        // For enclosed cavities, start from rest
        vel.fill(0.0, 0.0, 0.0);
    } else {
        // For channels/ducts with periodic streamwise, use uniform + perturbation
        vel.fill(1.0, 0.0, 0.0);
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    double x = mesh.x(i);
                    double y = mesh.y(j);
                    vel.u(i, j, k) += 0.01 * std::sin(2.0 * M_PI * y / tc.Ly);
                    vel.v(i, j, k) += 0.01 * std::sin(2.0 * M_PI * x / tc.Lx);
                }
            }
        }
    }
    solver.initialize(vel);
    solver.set_body_force(0.001, 0.0, 0.0);

    // Record initial state
    result.initial_ke = compute_kinetic_energy(solver.velocity(), mesh);
    result.max_div_seen = compute_max_divergence(solver.velocity(), mesh);
    result.max_pressure_seen = compute_max_pressure(solver.pressure(), mesh);

    // Record initial solver selection
    PoissonSolverType initial_type = solver.poisson_solver_type();
    result.initial_solver = solver.selection_reason();
    result.final_solver = result.initial_solver;

    // Run simulation and check invariants
    for (int step = 0; step < tc.nsteps; ++step) {
        solver.step();
        ++result.steps_completed;

        // Check divergence
        double div = compute_max_divergence(solver.velocity(), mesh);
        result.max_div_seen = std::max(result.max_div_seen, div);
        if (div > tc.div_bound) {
            result.passed = false;
            result.failure_reason = "divergence exceeded bound: " + std::to_string(div);
            break;
        }

        // Check kinetic energy
        double ke = compute_kinetic_energy(solver.velocity(), mesh);
        if (!is_finite(ke)) {
            result.passed = false;
            result.failure_reason = "kinetic energy is NaN/Inf";
            break;
        }
        // For zero-init cases, check absolute KE bound instead of growth ratio
        double ke_ref = std::max(result.initial_ke, 1e-10);  // Avoid div by zero
        if (ke > ke_ref * tc.ke_growth_max && ke > 1.0) {
            // Only fail if both relative growth AND absolute KE are large
            result.passed = false;
            result.failure_reason = "kinetic energy grew too fast: " +
                std::to_string(ke / ke_ref) + "x";
            break;
        }
        result.final_ke = ke;

        // Check pressure
        double max_p = compute_max_pressure(solver.pressure(), mesh);
        result.max_pressure_seen = std::max(result.max_pressure_seen, max_p);
        if (!is_finite(max_p)) {
            result.passed = false;
            result.failure_reason = "pressure is NaN/Inf";
            break;
        }

        // Check solver stability (no unexpected switching)
        PoissonSolverType current_type = solver.poisson_solver_type();
        if (current_type != initial_type) {
            result.passed = false;
            result.final_solver = solver.selection_reason();
            result.failure_reason = "solver switched mid-run";
            break;
        }
    }

    return result;
}

void print_result(const std::string& name, const InvariantTestResult& r) {
    std::cout << "  " << name << ": ";

    if (r.passed) {
        std::cout << "[PASS] ";
    } else {
        std::cout << "[FAIL] ";
    }

    std::cout << "steps=" << r.steps_completed
              << " div=" << std::scientific << std::setprecision(2) << r.max_div_seen;

    // Handle zero-init cases where initial_ke is 0
    if (r.initial_ke > 1e-15) {
        std::cout << " ke=" << std::fixed << std::setprecision(2)
                  << r.final_ke / r.initial_ke << "x";
    } else {
        std::cout << " ke=" << std::scientific << std::setprecision(2) << r.final_ke;
    }

    if (!r.passed) {
        std::cout << " (" << r.failure_reason << ")";
    }
    std::cout << "\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "================================================================\n";
    std::cout << "  Projection Method Invariants Test\n";
    std::cout << "================================================================\n\n";

#ifdef USE_GPU_OFFLOAD
    std::cout << "Build: GPU (USE_GPU_OFFLOAD=ON)\n";
#else
    std::cout << "Build: CPU (USE_GPU_OFFLOAD=OFF)\n";
#endif
#ifdef USE_HYPRE
    std::cout << "HYPRE: enabled\n";
#else
    std::cout << "HYPRE: disabled\n";
#endif
    std::cout << "\n";

    std::cout << "Testing projection method invariants over time:\n";
    std::cout << "  - Divergence remains bounded\n";
    std::cout << "  - Kinetic energy stays finite\n";
    std::cout << "  - Pressure stays finite\n";
    std::cout << "  - Solver selection stays stable\n\n";

    // Test cases: small grids, moderate step counts
    // Divergence bound: scales with grid spacing (coarser grids allow more error)
    //   - 2D 64^2: bound=0.1 (fine enough for strict check)
    //   - 3D 32^3: bound=0.3 (coarser grid, more discretization error, dx≈0.2)
    // KE growth max: 10x (should not blow up, but allow some transient growth)
    std::vector<InvariantTestCase> tests = {
        // 2D channel: periodic x, no-slip y walls
        {"2D_channel", 64, 64, 1, 2.0*M_PI, 2.0, 1.0,
         VelocityBC::Periodic, VelocityBC::NoSlip, VelocityBC::Periodic,
         50, 0.1, 10.0, false},

        // 2D cavity: all walls, start from rest
        {"2D_cavity", 64, 64, 1, 1.0, 1.0, 1.0,
         VelocityBC::NoSlip, VelocityBC::NoSlip, VelocityBC::Periodic,
         50, 0.1, 10.0, true},

        // 3D channel (small): periodic x/z, no-slip y
        // Relaxed div bound due to coarser grid (dx≈0.2 means O(dx) discretization error)
        {"3D_channel", 32, 32, 32, 2.0*M_PI, 2.0, 2.0*M_PI,
         VelocityBC::Periodic, VelocityBC::NoSlip, VelocityBC::Periodic,
         50, 0.3, 10.0, false},

        // 3D duct: periodic x, no-slip y/z walls
        {"3D_duct", 32, 32, 32, 2.0*M_PI, 2.0, 2.0,
         VelocityBC::Periodic, VelocityBC::NoSlip, VelocityBC::NoSlip,
         50, 0.3, 10.0, false},
    };

    std::cout << "--- Running " << tests.size() << " invariant tests ---\n\n";

    int passed = 0, failed = 0;

    for (const auto& tc : tests) {
        InvariantTestResult r = run_invariant_test(tc);
        print_result(tc.name, r);
        r.passed ? ++passed : ++failed;
    }

    // Summary
    std::cout << "\n================================================================\n";
    std::cout << "Projection Invariants Summary\n";
    std::cout << "================================================================\n";
    std::cout << "  Passed: " << passed << "/" << (passed + failed) << "\n";
    std::cout << "  Failed: " << failed << "/" << (passed + failed) << "\n";

    if (failed == 0) {
        std::cout << "\n[PASS] All projection invariant tests passed\n";
        std::cout << "       Solver coupling with time-stepper is correct\n";
        return 0;
    } else {
        std::cout << "\n[FAIL] " << failed << " projection invariant test(s) failed\n";
        std::cout << "       Check solver coupling with time integration!\n";
        return 1;
    }
}
