/// @file test_residual_consistency.cpp
/// @brief Poisson solver sanity check - validates pressure field is reasonable
///
/// SANITY TEST: Validates that Poisson solver produces non-trivial, finite
/// pressure fields. This is a basic smoke test, not a full residual check.
///
/// What this test checks:
///   - Pressure field is non-zero after projection
///   - L(p) = d²p/dx² + d²p/dy² is finite and reasonable
///   - Solution doesn't blow up or contain NaN
///
/// NOTE: This does NOT compute the true residual ||L(p) - rhs|| because the
/// intermediate RHS (div(u*)/dt) is internal to RANSSolver. For true residual
/// validation, use test_poisson_manufactured.cpp which uses known analytic RHS.

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

// Apply discrete 2D Laplacian operator: L(p) = (p_{i+1} - 2p_i + p_{i-1})/dx^2 + ...
void apply_laplacian_2d(const ScalarField& p, ScalarField& Lp, const Mesh& mesh) {
    double dx = mesh.dx;
    double dy = mesh.dy;
    double dx2_inv = 1.0 / (dx * dx);
    double dy2_inv = 1.0 / (dy * dy);

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double d2p_dx2 = (p(i+1, j) - 2.0*p(i, j) + p(i-1, j)) * dx2_inv;
            double d2p_dy2 = (p(i, j+1) - 2.0*p(i, j) + p(i, j-1)) * dy2_inv;
            Lp(i, j) = d2p_dx2 + d2p_dy2;
        }
    }
}

// Apply discrete 3D Laplacian operator
void apply_laplacian_3d(const ScalarField& p, ScalarField& Lp, const Mesh& mesh) {
    double dx = mesh.dx;
    double dy = mesh.dy;
    double dz = mesh.dz;
    double dx2_inv = 1.0 / (dx * dx);
    double dy2_inv = 1.0 / (dy * dy);
    double dz2_inv = 1.0 / (dz * dz);

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double d2p_dx2 = (p(i+1, j, k) - 2.0*p(i, j, k) + p(i-1, j, k)) * dx2_inv;
                double d2p_dy2 = (p(i, j+1, k) - 2.0*p(i, j, k) + p(i, j-1, k)) * dy2_inv;
                double d2p_dz2 = (p(i, j, k+1) - 2.0*p(i, j, k) + p(i, j, k-1)) * dz2_inv;
                Lp(i, j, k) = d2p_dx2 + d2p_dy2 + d2p_dz2;
            }
        }
    }
}

// Compute L2 norm
double l2_norm(const ScalarField& f, const Mesh& mesh) {
    double sum_sq = 0.0;
    int count = 0;
    if (mesh.is2D()) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                sum_sq += f(i, j) * f(i, j);
                ++count;
            }
        }
    } else {
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    sum_sq += f(i, j, k) * f(i, j, k);
                    ++count;
                }
            }
        }
    }
    return std::sqrt(sum_sq / count);
}

// Run a single residual test and return relative residual
// Returns -1 on failure
double run_residual_test_2d([[maybe_unused]] const std::string& name, int Nx, int Ny,
                            VelocityBC::Type xbc, VelocityBC::Type ybc,
                            PoissonSolverType solver_type) {
    Mesh mesh;
    mesh.init_uniform(Nx, Ny, 0.0, 2.0*M_PI, 0.0, 2.0*M_PI);

    Config config;
    config.Nx = Nx;
    config.Ny = Ny;
    config.Nz = 1;
    config.x_min = 0.0; config.x_max = 2.0*M_PI;
    config.y_min = 0.0; config.y_max = 2.0*M_PI;
    config.dt = 0.001;
    config.nu = 0.01;
    config.poisson_solver = solver_type;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = xbc; bc.x_hi = xbc;
    bc.y_lo = ybc; bc.y_hi = ybc;
    solver.set_velocity_bc(bc);

    // Initialize with divergent velocity to create Poisson problem
    VectorField vel(mesh);
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            double x = mesh.x(i);
            double y = mesh.y(j);
            vel.u(i, j) = std::sin(x) * std::cos(y);
        }
    }
    for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double x = mesh.x(i);
            double y = mesh.y(j);
            vel.v(i, j) = -std::cos(x) * std::sin(y) * 0.5;  // Non-zero divergence
        }
    }
    solver.initialize(vel);

    // Run one step to solve Poisson
    solver.step();

    // Compute residual: L(p) - rhs (where rhs = div(u*)/dt)
    // After projection, the pressure should satisfy the discrete Poisson equation
    const ScalarField& p = solver.pressure();

    // Compute L(p)
    ScalarField Lp(mesh);
    Lp.fill(0.0);
    apply_laplacian_2d(p, Lp, mesh);

    // For a well-posed Poisson solve, |L(p)| should be comparable to |rhs|
    // We check that the solution is non-trivial and consistent
    double Lp_norm = l2_norm(Lp, mesh);
    double p_norm = l2_norm(p, mesh);

    if (p_norm < 1e-15) {
        return -1.0;  // Trivial solution - failure
    }

    // Check that L(p) is reasonable (should be O(1) for our problem)
    if (!std::isfinite(Lp_norm) || Lp_norm > 1e10) {
        return -1.0;
    }

    return Lp_norm;  // Return the discrete Laplacian norm as a sanity check
}

int main() {
    std::cout << "================================================================\n";
    std::cout << "  Discrete Residual Consistency Test\n";
    std::cout << "================================================================\n\n";

#ifdef USE_GPU_OFFLOAD
    std::cout << "Build: GPU (USE_GPU_OFFLOAD=ON)\n";
#else
    std::cout << "Build: CPU (USE_GPU_OFFLOAD=OFF)\n";
#endif
    std::cout << "\n";

    std::cout << "Sanity check: Poisson solver produces reasonable pressure fields.\n";
    std::cout << "Checking: p is non-zero and L(p) is finite after projection.\n\n";

    int passed = 0, failed = 0;

    // Test 1: MG 2D periodic
    std::cout << "--- Test 1: MG 2D Periodic ---\n";
    {
        double result = run_residual_test_2d("MG_2D_periodic", 64, 64,
                                              VelocityBC::Periodic, VelocityBC::Periodic,
                                              PoissonSolverType::MG);
        if (result > 0 && result < 1e6) {
            std::cout << "  [PASS] |L(p)| = " << std::scientific << result << "\n";
            ++passed;
        } else {
            std::cout << "  [FAIL] Invalid solution\n";
            ++failed;
        }
    }

    // Test 2: MG 2D channel
    std::cout << "\n--- Test 2: MG 2D Channel ---\n";
    {
        double result = run_residual_test_2d("MG_2D_channel", 64, 64,
                                              VelocityBC::Periodic, VelocityBC::NoSlip,
                                              PoissonSolverType::MG);
        if (result > 0 && result < 1e6) {
            std::cout << "  [PASS] |L(p)| = " << std::scientific << result << "\n";
            ++passed;
        } else {
            std::cout << "  [FAIL] Invalid solution\n";
            ++failed;
        }
    }

    // Summary
    std::cout << "\n================================================================\n";
    std::cout << "Poisson Sanity Check Summary\n";
    std::cout << "================================================================\n";
    std::cout << "  Passed:  " << passed << "/" << (passed + failed) << "\n";
    std::cout << "  Failed:  " << failed << "/" << (passed + failed) << "\n";

    if (failed == 0) {
        std::cout << "\n[PASS] All sanity checks passed\n";
        std::cout << "       Pressure fields are non-trivial and finite\n";
        return 0;
    } else {
        std::cout << "\n[FAIL] " << failed << " sanity check(s) failed\n";
        return 1;
    }
}
