/// @file test_poisson_manufactured.cpp
/// @brief Manufactured-solution Poisson solver correctness test
///
/// CRITICAL TEST: Validates Poisson solvers produce CORRECT results, not just stable ones.
/// Tests all available solver backends with analytic solutions to catch:
///   - Sign errors, BC mishandling, stencil regressions
///   - Wrong scaling with dx/dy/dz
///   - Silent GPU changes that produce wrong answers
///
/// Method:
///   1. Pick analytic p(x,y,z) compatible with BCs
///   2. Compute RHS f = ∇²p analytically
///   3. Solve ∇²p = f numerically
///   4. Compare recovered p to analytic p (L2/L∞ norms)
///   5. Verify 2nd-order convergence with grid refinement
///
/// This catches "solver runs and is wrong" - stability tests alone miss this.

#include "mesh.hpp"
#include "fields.hpp"
#include "poisson_solver.hpp"
#include "poisson_solver_multigrid.hpp"
#ifdef USE_HYPRE
#include "poisson_solver_hypre.hpp"
#endif
// NOTE: FFT solver tests are in test_poisson_fft_manufactured.cpp (GPU-only)
#include <iostream>
#include <cmath>
#include <iomanip>
#include <vector>
#include <string>
#include <functional>

using namespace nncfd;

// ============================================================================
// Manufactured Solutions
// ============================================================================

// Solution for periodic x,z + Neumann y (channel flow BCs)
// p = sin(2πx/Lx) * cos(πy/Ly) * sin(2πz/Lz)
// ∇²p = -[(2π/Lx)² + (π/Ly)² + (2π/Lz)²] * p
struct ChannelSolution {
    double Lx, Ly, Lz;
    double kx, ky, kz;
    double lap_coeff;

    ChannelSolution(double lx, double ly, double lz)
        : Lx(lx), Ly(ly), Lz(lz) {
        kx = 2.0 * M_PI / Lx;
        ky = M_PI / Ly;  // cos for Neumann-compatible
        kz = 2.0 * M_PI / Lz;
        lap_coeff = -(kx*kx + ky*ky + kz*kz);
    }

    double p(double x, double y, double z) const {
        return std::sin(kx * x) * std::cos(ky * y) * std::sin(kz * z);
    }

    double rhs(double x, double y, double z) const {
        return lap_coeff * p(x, y, z);
    }
};

// Solution for periodic x + Neumann yz (duct flow BCs for FFT1D)
// p = sin(2πx/Lx) * cos(πy/Ly) * cos(πz/Lz)
struct DuctSolution {
    double Lx, Ly, Lz;
    double kx, ky, kz;
    double lap_coeff;

    DuctSolution(double lx, double ly, double lz)
        : Lx(lx), Ly(ly), Lz(lz) {
        kx = 2.0 * M_PI / Lx;
        ky = M_PI / Ly;
        kz = M_PI / Lz;
        lap_coeff = -(kx*kx + ky*ky + kz*kz);
    }

    double p(double x, double y, double z) const {
        return std::sin(kx * x) * std::cos(ky * y) * std::cos(kz * z);
    }

    double rhs(double x, double y, double z) const {
        return lap_coeff * p(x, y, z);
    }
};

// Solution for fully periodic (Taylor-Green like)
// p = sin(2πx/Lx) * sin(2πy/Ly) * sin(2πz/Lz)
struct PeriodicSolution {
    double Lx, Ly, Lz;
    double kx, ky, kz;
    double lap_coeff;

    PeriodicSolution(double lx, double ly, double lz)
        : Lx(lx), Ly(ly), Lz(lz) {
        kx = 2.0 * M_PI / Lx;
        ky = 2.0 * M_PI / Ly;
        kz = 2.0 * M_PI / Lz;
        lap_coeff = -(kx*kx + ky*ky + kz*kz);
    }

    double p(double x, double y, double z) const {
        return std::sin(kx * x) * std::sin(ky * y) * std::sin(kz * z);
    }

    double rhs(double x, double y, double z) const {
        return lap_coeff * p(x, y, z);
    }
};

// Solution for 2D periodic (x) + Neumann (y) - 2D channel
// p = sin(2πx/Lx) * cos(πy/Ly)
struct Channel2DSolution {
    double Lx, Ly;
    double kx, ky;
    double lap_coeff;

    Channel2DSolution(double lx, double ly)
        : Lx(lx), Ly(ly) {
        kx = 2.0 * M_PI / Lx;
        ky = M_PI / Ly;
        lap_coeff = -(kx*kx + ky*ky);
    }

    double p(double x, double y) const {
        return std::sin(kx * x) * std::cos(ky * y);
    }

    double rhs(double x, double y) const {
        return lap_coeff * p(x, y);
    }
};

// ============================================================================
// Error computation
// ============================================================================

template<typename Solution>
double compute_l2_error_3d(const ScalarField& p_num, const Mesh& mesh, const Solution& sol) {
    // Compute means (pressure determined up to constant)
    double p_mean = 0.0, exact_mean = 0.0;
    int count = 0;

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                p_mean += p_num(i, j, k);
                exact_mean += sol.p(mesh.x(i), mesh.y(j), mesh.z(k));
                ++count;
            }
        }
    }
    p_mean /= count;
    exact_mean /= count;

    // Compute L2 error
    double l2_error = 0.0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double exact = sol.p(mesh.x(i), mesh.y(j), mesh.z(k));
                double diff = (p_num(i, j, k) - p_mean) - (exact - exact_mean);
                l2_error += diff * diff;
            }
        }
    }
    return std::sqrt(l2_error / count);
}

double compute_l2_error_2d(const ScalarField& p_num, const Mesh& mesh, const Channel2DSolution& sol) {
    double p_mean = 0.0, exact_mean = 0.0;
    int count = 0;

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            p_mean += p_num(i, j);
            exact_mean += sol.p(mesh.x(i), mesh.y(j));
            ++count;
        }
    }
    p_mean /= count;
    exact_mean /= count;

    double l2_error = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double exact = sol.p(mesh.x(i), mesh.y(j));
            double diff = (p_num(i, j) - p_mean) - (exact - exact_mean);
            l2_error += diff * diff;
        }
    }
    return std::sqrt(l2_error / count);
}

// ============================================================================
// Test result structure
// ============================================================================

struct ConvergenceResult {
    std::string solver_name;
    std::string bc_config;
    std::vector<int> grid_sizes;
    std::vector<double> errors;
    double convergence_rate = 0.0;
    bool passed = false;
    std::string message;
};

// ============================================================================
// Solver-specific tests
// ============================================================================

// Test MG solver with manufactured solution
ConvergenceResult test_mg_convergence_3d(const std::string& bc_config) {
    ConvergenceResult result;
    result.solver_name = "MG";
    result.bc_config = bc_config;

    std::vector<int> Ns = {32, 64};
    const double Lx = 2.0 * M_PI;
    const double Ly = 2.0;
    const double Lz = 2.0 * M_PI;

    ChannelSolution sol(Lx, Ly, Lz);

    for (int N : Ns) {
        Mesh mesh;
        mesh.init_uniform(N, N, N, 0.0, Lx, 0.0, Ly, 0.0, Lz);

        ScalarField rhs(mesh);
        ScalarField p(mesh, 0.0);

        // Set RHS from manufactured solution
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    rhs(i, j, k) = sol.rhs(mesh.x(i), mesh.y(j), mesh.z(k));
                }
            }
        }

        MultigridPoissonSolver solver(mesh);
        solver.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                      PoissonBC::Neumann, PoissonBC::Neumann,
                      PoissonBC::Periodic, PoissonBC::Periodic);

        PoissonConfig cfg;
        cfg.tol = 1e-10;
        cfg.max_iter = 50;

        solver.solve(rhs, p, cfg);

        double err = compute_l2_error_3d(p, mesh, sol);
        result.grid_sizes.push_back(N);
        result.errors.push_back(err);
    }

    // Compute convergence rate
    if (result.errors.size() >= 2) {
        result.convergence_rate = std::log2(result.errors[0] / result.errors[1]);
        result.passed = (result.convergence_rate > 1.5 && result.convergence_rate < 2.5);
        result.message = result.passed ? "2nd-order convergence" : "convergence rate out of range";
    } else {
        result.passed = false;
        result.message = "insufficient data";
    }

    return result;
}

// Test MG solver in 2D
ConvergenceResult test_mg_convergence_2d() {
    ConvergenceResult result;
    result.solver_name = "MG";
    result.bc_config = "2D_channel_periodic_x_neumann_y";

    std::vector<int> Ns = {32, 64};
    const double Lx = 2.0 * M_PI;
    const double Ly = 2.0;

    Channel2DSolution sol(Lx, Ly);

    for (int N : Ns) {
        Mesh mesh;
        mesh.init_uniform(N, N, 0.0, Lx, 0.0, Ly);

        ScalarField rhs(mesh);
        ScalarField p(mesh, 0.0);

        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                rhs(i, j) = sol.rhs(mesh.x(i), mesh.y(j));
            }
        }

        MultigridPoissonSolver solver(mesh);
        solver.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                      PoissonBC::Neumann, PoissonBC::Neumann);

        PoissonConfig cfg;
        cfg.tol = 1e-10;
        cfg.max_iter = 50;

        solver.solve(rhs, p, cfg);

        double err = compute_l2_error_2d(p, mesh, sol);
        result.grid_sizes.push_back(N);
        result.errors.push_back(err);
    }

    if (result.errors.size() >= 2) {
        result.convergence_rate = std::log2(result.errors[0] / result.errors[1]);
        result.passed = (result.convergence_rate > 1.5 && result.convergence_rate < 2.5);
        result.message = result.passed ? "2nd-order convergence" : "convergence rate out of range";
    } else {
        result.passed = false;
        result.message = "insufficient data";
    }

    return result;
}

#ifdef USE_HYPRE
// Test HYPRE solver with manufactured solution
ConvergenceResult test_hypre_convergence_3d() {
    ConvergenceResult result;
    result.solver_name = "HYPRE";
    result.bc_config = "3D_channel_periodic_xz_neumann_y";

    std::vector<int> Ns = {32, 64};
    const double Lx = 2.0 * M_PI;
    const double Ly = 2.0;
    const double Lz = 2.0 * M_PI;

    ChannelSolution sol(Lx, Ly, Lz);

    for (int N : Ns) {
        Mesh mesh;
        mesh.init_uniform(N, N, N, 0.0, Lx, 0.0, Ly, 0.0, Lz);

        ScalarField rhs(mesh);
        ScalarField p(mesh, 0.0);

        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    rhs(i, j, k) = sol.rhs(mesh.x(i), mesh.y(j), mesh.z(k));
                }
            }
        }

        HyprePoissonSolver solver(mesh);
        solver.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                      PoissonBC::Neumann, PoissonBC::Neumann,
                      PoissonBC::Periodic, PoissonBC::Periodic);

        PoissonConfig cfg;
        cfg.tol = 1e-10;
        cfg.max_iter = 50;

        solver.solve(rhs, p, cfg);

        double err = compute_l2_error_3d(p, mesh, sol);
        result.grid_sizes.push_back(N);
        result.errors.push_back(err);
    }

    if (result.errors.size() >= 2) {
        result.convergence_rate = std::log2(result.errors[0] / result.errors[1]);
        result.passed = (result.convergence_rate > 1.5 && result.convergence_rate < 2.5);
        result.message = result.passed ? "2nd-order convergence" : "convergence rate out of range";
    } else {
        result.passed = false;
        result.message = "insufficient data";
    }

    return result;
}

ConvergenceResult test_hypre_convergence_2d() {
    ConvergenceResult result;
    result.solver_name = "HYPRE";
    result.bc_config = "2D_channel_periodic_x_neumann_y";

    std::vector<int> Ns = {32, 64};
    const double Lx = 2.0 * M_PI;
    const double Ly = 2.0;

    Channel2DSolution sol(Lx, Ly);

    for (int N : Ns) {
        Mesh mesh;
        mesh.init_uniform(N, N, 0.0, Lx, 0.0, Ly);

        ScalarField rhs(mesh);
        ScalarField p(mesh, 0.0);

        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                rhs(i, j) = sol.rhs(mesh.x(i), mesh.y(j));
            }
        }

        HyprePoissonSolver solver(mesh);
        solver.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                      PoissonBC::Neumann, PoissonBC::Neumann);

        PoissonConfig cfg;
        cfg.tol = 1e-10;
        cfg.max_iter = 50;

        solver.solve(rhs, p, cfg);

        double err = compute_l2_error_2d(p, mesh, sol);
        result.grid_sizes.push_back(N);
        result.errors.push_back(err);
    }

    if (result.errors.size() >= 2) {
        result.convergence_rate = std::log2(result.errors[0] / result.errors[1]);
        result.passed = (result.convergence_rate > 1.5 && result.convergence_rate < 2.5);
        result.message = result.passed ? "2nd-order convergence" : "convergence rate out of range";
    } else {
        result.passed = false;
        result.message = "insufficient data";
    }

    return result;
}
#endif

// NOTE: FFT/FFT1D tests are in test_poisson_fft_manufactured.cpp
// They use solve_device() and require GPU + device pointer setup.

// ============================================================================
// Main
// ============================================================================

void print_result(const ConvergenceResult& r) {
    std::cout << "  " << r.solver_name << " [" << r.bc_config << "]: ";

    if (r.passed) {
        std::cout << "[PASS] ";
    } else {
        std::cout << "[FAIL] ";
    }

    // Print errors at each grid size
    for (size_t i = 0; i < r.grid_sizes.size(); ++i) {
        std::cout << "N=" << r.grid_sizes[i] << ":err=" << std::scientific
                  << std::setprecision(2) << r.errors[i];
        if (i < r.grid_sizes.size() - 1) std::cout << ", ";
    }

    std::cout << " rate=" << std::fixed << std::setprecision(2)
              << r.convergence_rate << " (" << r.message << ")\n";
}

int main() {
    std::cout << "================================================================\n";
    std::cout << "  Manufactured Solution Poisson Solver Correctness Test\n";
    std::cout << "================================================================\n\n";

    // Build info
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
#ifdef USE_FFT_POISSON
    std::cout << "FFT: enabled\n";
#else
    std::cout << "FFT: disabled\n";
#endif
    std::cout << "\n";

    std::vector<ConvergenceResult> results;
    int passed = 0, failed = 0;

    // ========================================================================
    // MG Tests (always available)
    // ========================================================================
    std::cout << "--- Multigrid Solver Tests ---\n";

    results.push_back(test_mg_convergence_3d("3D_channel_periodic_xz_neumann_y"));
    print_result(results.back());
    results.back().passed ? ++passed : ++failed;

    results.push_back(test_mg_convergence_2d());
    print_result(results.back());
    results.back().passed ? ++passed : ++failed;

    // ========================================================================
    // HYPRE Tests (if available)
    // ========================================================================
#ifdef USE_HYPRE
    std::cout << "\n--- HYPRE Solver Tests ---\n";

    results.push_back(test_hypre_convergence_3d());
    print_result(results.back());
    results.back().passed ? ++passed : ++failed;

    results.push_back(test_hypre_convergence_2d());
    print_result(results.back());
    results.back().passed ? ++passed : ++failed;
#endif

    // NOTE: FFT tests are in test_poisson_fft_manufactured.cpp (GPU-only, uses solve_device())

    // ========================================================================
    // Summary
    // ========================================================================
    std::cout << "\n================================================================\n";
    std::cout << "Manufactured Solution Test Summary\n";
    std::cout << "================================================================\n";
    std::cout << "  Passed: " << passed << "/" << (passed + failed) << "\n";
    std::cout << "  Failed: " << failed << "/" << (passed + failed) << "\n";

    if (failed == 0) {
        std::cout << "\n[PASS] All solvers produce correct results with 2nd-order convergence\n";
        return 0;
    } else {
        std::cout << "\n[FAIL] " << failed << " solver(s) failed correctness check\n";
        std::cout << "       This indicates a regression in solver accuracy!\n";
        return 1;
    }
}
