/// @file test_poisson_dirichlet_mixed.cpp
/// @brief Dirichlet and mixed-BC Poisson solver validation test
///
/// CRITICAL TEST: Validates solvers handle Dirichlet and mixed BCs correctly.
/// These configurations are weakly tested elsewhere but expose:
///   - Gauge/nullspace handling bugs (Dirichlet removes the nullspace)
///   - Boundary flux errors
///   - BC mishandling at corners
///
/// Tests:
///   1. Pure Dirichlet 3D cube - known analytic solution
///   2. Mixed BC (periodic x, Dirichlet y, Neumann z) - representative production case
///   3. Pure Dirichlet 2D square
///
/// For each, we use manufactured solutions and verify 2nd-order convergence.

#include "mesh.hpp"
#include "fields.hpp"
#include "poisson_solver_multigrid.hpp"
#include "test_fixtures.hpp"
#include "test_utilities.hpp"
#ifdef USE_HYPRE
#include "poisson_solver_hypre.hpp"
#endif
#include <iostream>
#include <cmath>
#include <iomanip>
#include <vector>
#include <string>

using namespace nncfd;
using nncfd::test::DirichletSolution3D;
using nncfd::test::DirichletSolution2D;
using nncfd::test::MixedBCSolution3D;
using nncfd::test::compute_l2_error_3d;
using nncfd::test::compute_l2_error_2d;

// Manufactured solutions imported from test_fixtures.hpp:
// - DirichletSolution3D: pure Dirichlet (p=0 at all boundaries)
// - DirichletSolution2D: 2D pure Dirichlet
// - MixedBCSolution3D: periodic x, Dirichlet y, Neumann z

// Error computation imported from test_utilities.hpp:
// - compute_l2_error_3d(p_num, mesh, sol) - with mean subtraction
// - compute_l2_error_2d(p_num, mesh, sol) - with mean subtraction

// For pure Dirichlet, no mean subtraction needed (solution is unique)
// Use local wrapper that skips mean subtraction
template<typename Solution>
double compute_l2_error_dirichlet_3d(const ScalarField& p_num, const Mesh& mesh, const Solution& sol) {
    double l2_error = 0.0;
    int count = 0;

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double exact = sol.p(mesh.x(i), mesh.y(j), mesh.z(k));
                double diff = p_num(i, j, k) - exact;
                l2_error += diff * diff;
                ++count;
            }
        }
    }
    return std::sqrt(l2_error / count);
}

template<typename Solution>
double compute_l2_error_dirichlet_2d(const ScalarField& p_num, const Mesh& mesh, const Solution& sol) {
    double l2_error = 0.0;
    int count = 0;

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double exact = sol.p(mesh.x(i), mesh.y(j));
            double diff = p_num(i, j) - exact;
            l2_error += diff * diff;
            ++count;
        }
    }
    return std::sqrt(l2_error / count);
}

// For mixed BC with periodic direction, use compute_l2_error_3d which includes mean subtraction

// ============================================================================
// Test result structure
// ============================================================================

struct TestResult {
    std::string solver_name;
    std::string bc_config;
    std::vector<int> grid_sizes;
    std::vector<double> errors;
    double convergence_rate = 0.0;
    bool passed = false;
    std::string message;
};

void print_result(const TestResult& r) {
    std::cout << "  " << r.solver_name << " [" << r.bc_config << "]: ";

    if (r.passed) {
        std::cout << "[PASS] ";
    } else {
        std::cout << "[FAIL] ";
    }

    for (size_t i = 0; i < r.grid_sizes.size(); ++i) {
        std::cout << "N=" << r.grid_sizes[i] << ":err=" << std::scientific
                  << std::setprecision(2) << r.errors[i];
        if (i < r.grid_sizes.size() - 1) std::cout << ", ";
    }

    std::cout << " rate=" << std::fixed << std::setprecision(2)
              << r.convergence_rate << " (" << r.message << ")\n";
}

// ============================================================================
// MG Tests
// ============================================================================

TestResult test_mg_dirichlet_3d() {
    TestResult result;
    result.solver_name = "MG";
    result.bc_config = "3D_pure_dirichlet";

    std::vector<int> Ns = {32, 64};
    const double Lx = 1.0, Ly = 1.0, Lz = 1.0;

    DirichletSolution3D sol(Lx, Ly, Lz);

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

        MultigridPoissonSolver solver(mesh);
        solver.set_bc(PoissonBC::Dirichlet, PoissonBC::Dirichlet,
                      PoissonBC::Dirichlet, PoissonBC::Dirichlet,
                      PoissonBC::Dirichlet, PoissonBC::Dirichlet);

        PoissonConfig cfg;
        cfg.tol = 1e-10;
        cfg.max_iter = 50;

        solver.solve(rhs, p, cfg);

        double err = compute_l2_error_dirichlet_3d(p, mesh, sol);
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

TestResult test_mg_dirichlet_2d() {
    TestResult result;
    result.solver_name = "MG";
    result.bc_config = "2D_pure_dirichlet";

    std::vector<int> Ns = {32, 64};
    const double Lx = 1.0, Ly = 1.0;

    DirichletSolution2D sol(Lx, Ly);

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
        solver.set_bc(PoissonBC::Dirichlet, PoissonBC::Dirichlet,
                      PoissonBC::Dirichlet, PoissonBC::Dirichlet);

        PoissonConfig cfg;
        cfg.tol = 1e-10;
        cfg.max_iter = 50;

        solver.solve(rhs, p, cfg);

        double err = compute_l2_error_dirichlet_2d(p, mesh, sol);
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

TestResult test_mg_mixed_bc() {
    TestResult result;
    result.solver_name = "MG";
    result.bc_config = "3D_mixed_periodic_dirichlet_neumann";

    std::vector<int> Ns = {32, 64};
    const double Lx = 2.0 * M_PI, Ly = 1.0, Lz = 1.0;

    MixedBCSolution3D sol(Lx, Ly, Lz);

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

        MultigridPoissonSolver solver(mesh);
        solver.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,    // x: periodic
                      PoissonBC::Dirichlet, PoissonBC::Dirichlet,  // y: Dirichlet
                      PoissonBC::Neumann, PoissonBC::Neumann);     // z: Neumann

        PoissonConfig cfg;
        cfg.tol = 1e-10;
        cfg.max_iter = 50;

        solver.solve(rhs, p, cfg);

        // Mixed BC with periodic direction needs mean subtraction
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

// ============================================================================
// HYPRE Tests
// ============================================================================

#ifdef USE_HYPRE
TestResult test_hypre_dirichlet_3d() {
    TestResult result;
    result.solver_name = "HYPRE";
    result.bc_config = "3D_pure_dirichlet";

    std::vector<int> Ns = {32, 64};
    const double Lx = 1.0, Ly = 1.0, Lz = 1.0;

    DirichletSolution3D sol(Lx, Ly, Lz);

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
        solver.set_bc(PoissonBC::Dirichlet, PoissonBC::Dirichlet,
                      PoissonBC::Dirichlet, PoissonBC::Dirichlet,
                      PoissonBC::Dirichlet, PoissonBC::Dirichlet);

        PoissonConfig cfg;
        cfg.tol = 1e-10;
        cfg.max_iter = 50;

        solver.solve(rhs, p, cfg);

        // Pure Dirichlet: no mean subtraction needed
        double err = compute_l2_error_dirichlet_3d(p, mesh, sol);
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

TestResult test_hypre_dirichlet_2d() {
    TestResult result;
    result.solver_name = "HYPRE";
    result.bc_config = "2D_pure_dirichlet";

    std::vector<int> Ns = {32, 64};
    const double Lx = 1.0, Ly = 1.0;

    DirichletSolution2D sol(Lx, Ly);

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
        solver.set_bc(PoissonBC::Dirichlet, PoissonBC::Dirichlet,
                      PoissonBC::Dirichlet, PoissonBC::Dirichlet);

        PoissonConfig cfg;
        cfg.tol = 1e-10;
        cfg.max_iter = 50;

        solver.solve(rhs, p, cfg);

        // Pure Dirichlet: no mean subtraction needed
        double err = compute_l2_error_dirichlet_2d(p, mesh, sol);
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

TestResult test_hypre_mixed_bc() {
    TestResult result;
    result.solver_name = "HYPRE";
    result.bc_config = "3D_mixed_periodic_dirichlet_neumann";

    std::vector<int> Ns = {32, 64};
    const double Lx = 2.0 * M_PI, Ly = 1.0, Lz = 1.0;

    MixedBCSolution3D sol(Lx, Ly, Lz);

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
                      PoissonBC::Dirichlet, PoissonBC::Dirichlet,
                      PoissonBC::Neumann, PoissonBC::Neumann);

        PoissonConfig cfg;
        cfg.tol = 1e-10;
        cfg.max_iter = 50;

        solver.solve(rhs, p, cfg);

        // Mixed BC with periodic direction needs mean subtraction
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
#endif

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "================================================================\n";
    std::cout << "  Dirichlet and Mixed-BC Poisson Solver Validation Test\n";
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

    int passed = 0, failed = 0;

    // ========================================================================
    // MG Tests
    // ========================================================================
    std::cout << "--- Multigrid Solver Tests ---\n";

    TestResult r = test_mg_dirichlet_3d();
    print_result(r);
    r.passed ? ++passed : ++failed;

    r = test_mg_dirichlet_2d();
    print_result(r);
    r.passed ? ++passed : ++failed;

    r = test_mg_mixed_bc();
    print_result(r);
    r.passed ? ++passed : ++failed;

    // ========================================================================
    // HYPRE Tests
    // ========================================================================
#ifdef USE_HYPRE
    std::cout << "\n--- HYPRE Solver Tests ---\n";

    r = test_hypre_dirichlet_3d();
    print_result(r);
    r.passed ? ++passed : ++failed;

    r = test_hypre_dirichlet_2d();
    print_result(r);
    r.passed ? ++passed : ++failed;

    r = test_hypre_mixed_bc();
    print_result(r);
    r.passed ? ++passed : ++failed;
#endif

    // ========================================================================
    // Summary
    // ========================================================================
    std::cout << "\n================================================================\n";
    std::cout << "Dirichlet/Mixed-BC Test Summary\n";
    std::cout << "================================================================\n";
    std::cout << "  Passed: " << passed << "/" << (passed + failed) << "\n";
    std::cout << "  Failed: " << failed << "/" << (passed + failed) << "\n";

    if (failed == 0) {
        std::cout << "\n[PASS] All Dirichlet/mixed-BC solves correct with 2nd-order convergence\n";
        return 0;
    } else {
        std::cout << "\n[FAIL] " << failed << " solver(s) failed Dirichlet/mixed-BC correctness\n";
        std::cout << "       This indicates BC handling or gauge issues!\n";
        return 1;
    }
}
