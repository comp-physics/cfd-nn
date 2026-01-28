/// @file test_mg_ramp_diagnostic.cpp
/// @brief Diagnose where the MG solver introduces the linear ramp mode
///
/// For periodic Poisson with RHS=0, the solution should be constant.
/// This test verifies the MG solver handles this case correctly.

#include <iostream>
#include <iomanip>
#include <cmath>
#include "poisson_solver_multigrid.hpp"
#include "mesh.hpp"
#include "fields.hpp"

using namespace nncfd;

/// Compute mean dp/dx over the interior
double compute_mean_dpdx(const ScalarField& p, const Mesh& mesh) {
    double sum_dpdx = 0.0;
    int count = 0;

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end() - 1; ++i) {
            double dpdx = (p(i+1, j) - p(i, j)) / mesh.dx;
            sum_dpdx += dpdx;
            count++;
        }
    }

    return (count > 0) ? sum_dpdx / count : 0.0;
}

/// Compute pressure range
std::pair<double, double> compute_pressure_range(const ScalarField& p, const Mesh& mesh) {
    double p_min = 1e30, p_max = -1e30;

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double val = p(i, j);
            p_min = std::min(p_min, val);
            p_max = std::max(p_max, val);
        }
    }

    return {p_min, p_max};
}

/// Compute mean pressure
double compute_mean_pressure(const ScalarField& p, const Mesh& mesh) {
    double sum = 0.0;
    int count = 0;

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            sum += p(i, j);
            count++;
        }
    }

    return (count > 0) ? sum / count : 0.0;
}

int main() {
    std::cout << "================================================================\n";
    std::cout << "  MG Ramp Mode Diagnostic\n";
    std::cout << "================================================================\n\n";

    const int N = 8;  // Grid size (same as flow solver test)

    Mesh mesh;
    mesh.init_uniform(N, N, 0.0, 2.0*M_PI, 0.0, 2.0*M_PI);

    std::cout << "Grid: " << N << "x" << N << "\n";
    std::cout << "Domain: [0, 2π] x [0, 2π]\n";
    std::cout << "dx = " << mesh.dx << "\n\n";

    // Create fields
    ScalarField rhs(mesh);
    ScalarField p(mesh);

    // Set RHS = 0 everywhere (including all ghost cells for Ng >= 2)
    const int Ng = mesh.Nghost;
    for (int j = 1 - Ng; j <= mesh.Ny + Ng; ++j) {
        for (int i = 1 - Ng; i <= mesh.Nx + Ng; ++i) {
            rhs(i, j) = 0.0;
        }
    }

    // Set initial guess = 0
    for (int j = 1 - Ng; j <= mesh.Ny + Ng; ++j) {
        for (int i = 1 - Ng; i <= mesh.Nx + Ng; ++i) {
            p(i, j) = 0.0;
        }
    }

    std::cout << "Initial conditions:\n";
    std::cout << "  RHS = 0 everywhere\n";
    std::cout << "  Initial guess = 0 everywhere\n\n";

    // Create MG solver with periodic BCs
    MultigridPoissonSolver solver(mesh);
    solver.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                  PoissonBC::Periodic, PoissonBC::Periodic);

    PoissonConfig cfg;
    cfg.tol = 1e-12;
    cfg.max_vcycles = 50;
    // Test both modes
    bool test_fixed_cycle = true;
    if (test_fixed_cycle) {
        cfg.fixed_cycles = 8;  // Match flow solver default
        cfg.adaptive_cycles = true;
        cfg.tol_rhs = 1e-6;
        std::cout << "Mode: Fixed-cycle (8 cycles, adaptive checking)\n";
    } else {
        std::cout << "Mode: Convergence-based (max " << cfg.max_vcycles << " V-cycles)\n";
    }

    std::cout << "BCs: All Periodic\n";
    std::cout << "Tolerance: " << cfg.tol << "\n\n";

    // Check initial state
    auto [p_min_init, p_max_init] = compute_pressure_range(p, mesh);
    double dpdx_init = compute_mean_dpdx(p, mesh);

    std::cout << "Before solve:\n";
    std::cout << "  p range: [" << p_min_init << ", " << p_max_init << "]\n";
    std::cout << "  mean(dp/dx): " << std::scientific << dpdx_init << "\n\n";

    // Solve
    std::cout << "=== Running MG solve ===\n\n";

    int iter = solver.solve(rhs, p, cfg);

    std::cout << "Solve completed in " << iter << " iterations\n";
    std::cout << "Final residual: " << solver.residual() << "\n\n";

    // Check result
    auto [p_min_final, p_max_final] = compute_pressure_range(p, mesh);
    double dpdx_final = compute_mean_dpdx(p, mesh);
    double p_mean = compute_mean_pressure(p, mesh);

    std::cout << "After solve:\n";
    std::cout << "  p range: [" << std::scientific << std::setprecision(4)
              << p_min_final << ", " << p_max_final << "]\n";
    std::cout << "  p spread (max-min): " << (p_max_final - p_min_final) << "\n";
    std::cout << "  mean(p): " << p_mean << "\n";
    std::cout << "  mean(dp/dx): " << dpdx_final << "\n\n";

    // Check individual rows to see if dp/dx varies
    std::cout << "=== dp/dx by row (should all be ~0 for periodic) ===\n\n";

    for (int j = mesh.j_begin(); j < std::min(mesh.j_begin() + 4, mesh.j_end()); ++j) {
        double row_dpdx = 0.0;
        int count = 0;
        for (int i = mesh.i_begin(); i < mesh.i_end() - 1; ++i) {
            row_dpdx += (p(i+1, j) - p(i, j)) / mesh.dx;
            count++;
        }
        row_dpdx /= count;
        std::cout << "  Row j=" << j << ": mean(dp/dx) = " << row_dpdx << "\n";
    }

    // Sample the actual pressure values across one row
    int j_sample = mesh.j_begin() + mesh.Ny / 2;
    std::cout << "\n=== Pressure values across row j=" << j_sample << " ===\n";
    std::cout << "  (Should be constant for periodic with RHS=0)\n\n";

    for (int i = mesh.i_begin(); i < std::min(mesh.i_begin() + 8, mesh.i_end()); ++i) {
        std::cout << "  p(" << i << ") = " << std::fixed << std::setprecision(8)
                  << p(i, j_sample) << "\n";
    }
    std::cout << "  ...\n";
    for (int i = std::max(mesh.i_begin(), mesh.i_end() - 4); i < mesh.i_end(); ++i) {
        std::cout << "  p(" << i << ") = " << p(i, j_sample) << "\n";
    }

    // Check periodicity: left and right edges should be consistent
    double p_left = p(mesh.i_begin(), j_sample);
    double p_right = p(mesh.i_end() - 1, j_sample);
    std::cout << "\nPeriodicity check:\n";
    std::cout << "  p_left (i=" << mesh.i_begin() << ") = " << std::scientific << p_left << "\n";
    std::cout << "  p_right (i=" << (mesh.i_end()-1) << ") = " << p_right << "\n";
    std::cout << "  Difference: " << (p_right - p_left) << "\n";

    // Check ghost cells
    std::cout << "\nGhost cell check:\n";
    std::cout << "  Left ghost p(" << (mesh.i_begin()-1) << ") = " << p(mesh.i_begin()-1, j_sample) << "\n";
    std::cout << "  Should equal p(" << (mesh.i_end()-1) << ") = " << p(mesh.i_end()-1, j_sample) << "\n";
    std::cout << "  Right ghost p(" << mesh.i_end() << ") = " << p(mesh.i_end(), j_sample) << "\n";
    std::cout << "  Should equal p(" << mesh.i_begin() << ") = " << p(mesh.i_begin(), j_sample) << "\n";

    // Check dp/dx at boundaries (should be same as interior for periodic)
    double dpdx_left = (p(mesh.i_begin(), j_sample) - p(mesh.i_begin()-1, j_sample)) / mesh.dx;
    double dpdx_mid = (p(mesh.i_begin() + mesh.Nx/2 + 1, j_sample) - p(mesh.i_begin() + mesh.Nx/2, j_sample)) / mesh.dx;
    double dpdx_right = (p(mesh.i_end(), j_sample) - p(mesh.i_end()-1, j_sample)) / mesh.dx;

    std::cout << "\ndp/dx at different x-locations:\n";
    std::cout << "  Left boundary: " << dpdx_left << "\n";
    std::cout << "  Middle: " << dpdx_mid << "\n";
    std::cout << "  Right boundary: " << dpdx_right << "\n";

    std::cout << "\n=== Analysis ===\n\n";

    double spread = p_max_final - p_min_final;
    double tol = 1e-8;

    if (spread < tol && std::abs(dpdx_final) < tol) {
        std::cout << "PASS: Solution is constant (as expected for periodic RHS=0)\n";
        return 0;
    } else {
        std::cout << "FAIL: Solution has non-constant structure\n";
        std::cout << "  Pressure spread: " << spread << " (expected < " << tol << ")\n";
        std::cout << "  Mean dp/dx: " << dpdx_final << " (expected ~0)\n\n";

        // Check if it's a linear ramp
        // For p = ax + c, the spread would be approximately a * L
        double implied_slope = (p_max_final - p_min_final) / (2.0 * M_PI);
        std::cout << "  If linear ramp, implied slope a = " << implied_slope << "\n";
        std::cout << "  This would give dp/dx = " << implied_slope << "\n";
        std::cout << "  Actual mean(dp/dx) = " << dpdx_final << "\n\n";

        if (std::abs(implied_slope - dpdx_final) / (std::abs(dpdx_final) + 1e-15) < 0.1) {
            std::cout << "  *** CONFIRMED: Linear ramp mode present in solution ***\n\n";
            std::cout << "  The MG solver is not properly eliminating the ramp mode.\n";
            std::cout << "  For periodic Laplacian, p=ax is NOT in the nullspace\n";
            std::cout << "  (since ∇²(ax)=0 but p(0)≠p(L) violates periodicity).\n";
            std::cout << "  This suggests a bug in periodic BC enforcement.\n";
        }

        return 1;
    }
}
