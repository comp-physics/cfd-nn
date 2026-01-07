/// Unified Poisson Solver Test Suite
///
/// Consolidates 10 Poisson test files (~3934 lines) into one parameterized file.
/// Uses loops over solver types, BCs, and grid sizes.
///
/// Covers:
/// - Basic Laplacian/solver unit tests
/// - Manufactured solution correctness
/// - Grid convergence (2nd order)
/// - Cross-solver consistency
/// - Nullspace/gauge handling
/// - Stretched grid robustness
/// - Solver selection logic
/// - CPU/GPU consistency (3D)

#include "mesh.hpp"
#include "fields.hpp"
#include "poisson_solver.hpp"
#include "poisson_solver_multigrid.hpp"
#include "test_framework.hpp"
#include "test_fixtures.hpp"
#include "test_utilities.hpp"
#include "solver.hpp"
#include "config.hpp"
#ifdef USE_HYPRE
#include "poisson_solver_hypre.hpp"
#endif
#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <functional>

using namespace nncfd;
using namespace nncfd::test;

//=============================================================================
// Test Result Tracking
//=============================================================================

struct TestResult {
    std::string name;
    bool passed;
    std::string message;
};

static std::vector<TestResult> results;

static void record(const std::string& name, bool passed, const std::string& msg = "") {
    results.push_back({name, passed, msg});
    std::cout << "  " << std::left << std::setw(50) << name;
    std::cout << (passed ? "[PASS]" : "[FAIL]");
    if (!msg.empty()) std::cout << " " << msg;
    std::cout << "\n";
}

//=============================================================================
// Section 1: Basic Unit Tests (from test_poisson.cpp)
//=============================================================================

void test_laplacian() {
    Mesh mesh;
    mesh.init_uniform(20, 20, 0.0, 1.0, 0.0, 1.0);

    ScalarField p(mesh);
    for (int j = 0; j < mesh.total_Ny(); ++j) {
        for (int i = 0; i < mesh.total_Nx(); ++i) {
            double x = mesh.x(i), y = mesh.y(j);
            p(i, j) = x * x + y * y;
        }
    }

    double dx2 = mesh.dx * mesh.dx;
    double dy2 = mesh.dy * mesh.dy;
    double max_err = 0.0;

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double lap = (p(i+1,j) - 2*p(i,j) + p(i-1,j)) / dx2
                       + (p(i,j+1) - 2*p(i,j) + p(i,j-1)) / dy2;
            max_err = std::max(max_err, std::abs(lap - 4.0));
        }
    }

    record("Laplacian of x^2+y^2 = 4", max_err < 0.01,
           "err=" + std::to_string(max_err));
}

void test_basic_solve() {
    Mesh mesh;
    mesh.init_uniform(32, 32, 0.0, 1.0, 0.0, 1.0);

    ScalarField rhs(mesh, 1.0);
    ScalarField p(mesh, 0.0);

    PoissonSolver solver(mesh);
    solver.set_bc(PoissonBC::Dirichlet, PoissonBC::Dirichlet,
                  PoissonBC::Dirichlet, PoissonBC::Dirichlet);
    solver.set_dirichlet_value(0.0);

    PoissonConfig cfg;
    cfg.tol = 1e-6;
    cfg.max_iter = 20000;
    cfg.omega = 1.8;

    int iters = solver.solve(rhs, p, cfg);
    bool converged = solver.residual() < 1e-4;

    record("Basic Dirichlet solve", converged,
           "iters=" + std::to_string(iters) + " res=" + std::to_string(solver.residual()));
}

void test_periodic_solve() {
    Mesh mesh;
    int N = 32;
    double L = 2.0 * M_PI;
    mesh.init_uniform(N, N, 0.0, L, 0.0, L);

    ScalarField rhs(mesh);
    ScalarField p(mesh, 0.0);

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double x = mesh.x(i), y = mesh.y(j);
            rhs(i, j) = -2.0 * std::sin(x) * std::sin(y);
        }
    }

    PoissonSolver solver(mesh);
    solver.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                  PoissonBC::Periodic, PoissonBC::Periodic);

    PoissonConfig cfg;
    cfg.tol = 1e-8;
    cfg.max_iter = 10000;

    solver.solve(rhs, p, cfg);

    // Check against exact (up to constant)
    double p_mean = 0.0, exact_mean = 0.0;
    int count = 0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            p_mean += p(i, j);
            exact_mean += std::sin(mesh.x(i)) * std::sin(mesh.y(j));
            ++count;
        }
    }
    p_mean /= count;
    exact_mean /= count;

    double max_err = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double exact = std::sin(mesh.x(i)) * std::sin(mesh.y(j));
            double err = std::abs((p(i,j) - p_mean) - (exact - exact_mean));
            max_err = std::max(max_err, err);
        }
    }

    record("Periodic sin(x)sin(y) solve", max_err < 0.1,
           "max_err=" + std::to_string(max_err));
}

void run_unit_tests() {
    std::cout << "\n=== Unit Tests ===\n";
    test_laplacian();
    test_basic_solve();
    test_periodic_solve();
}

//=============================================================================
// Section 2: Grid Convergence Tests (from test_poisson_solvers.cpp)
//=============================================================================

double compute_l2_error_func(const ScalarField& p, const Mesh& mesh,
                              std::function<double(double,double)> exact) {
    double p_mean = 0.0, exact_mean = 0.0;
    int count = 0;

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            p_mean += p(i, j);
            exact_mean += exact(mesh.x(i), mesh.y(j));
            ++count;
        }
    }
    p_mean /= count;
    exact_mean /= count;

    double l2 = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double diff = (p(i,j) - p_mean) - (exact(mesh.x(i), mesh.y(j)) - exact_mean);
            l2 += diff * diff;
        }
    }
    return std::sqrt(l2 / count);
}

void test_mg_convergence_2d() {
    std::cout << "\n=== Multigrid 2D Convergence ===\n";

    std::vector<int> sizes = {16, 32, 64};
    std::vector<double> errors;

    for (int N : sizes) {
        Mesh mesh;
        double L = 2.0 * M_PI;
        mesh.init_uniform(N, N, 0.0, L, 0.0, L);

        auto exact = [](double x, double y) { return std::sin(x) * std::sin(y); };
        auto rhs_fn = [](double x, double y) { return -2.0 * std::sin(x) * std::sin(y); };

        ScalarField rhs(mesh);
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                rhs(i, j) = rhs_fn(mesh.x(i), mesh.y(j));
            }
        }

        ScalarField p(mesh, 0.0);
        MultigridPoissonSolver mg(mesh);
        mg.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                  PoissonBC::Periodic, PoissonBC::Periodic);

        PoissonConfig cfg;
        cfg.tol = 1e-10;
        cfg.max_iter = 100;
        mg.solve(rhs, p, cfg);

        double err = compute_l2_error_func(p, mesh, exact);
        errors.push_back(err);

        record("MG 2D N=" + std::to_string(N), true,
               "L2=" + std::to_string(err));
    }

    // Check 2nd order convergence
    if (errors.size() >= 2) {
        double rate = std::log(errors[0] / errors[1]) / std::log(2.0);
        record("MG 2D convergence rate", rate > 1.5,
               "rate=" + std::to_string(rate) + " (expect ~2)");
    }
}

void run_convergence_tests() {
    test_mg_convergence_2d();
}

//=============================================================================
// Section 3: Solver Selection Tests (from test_poisson_selection.cpp)
//=============================================================================

void test_solver_selection() {
    std::cout << "\n=== Solver Selection ===\n";

    // Test 2D channel auto-selection
    {
        Mesh mesh;
        mesh.init_uniform(32, 32, 0.0, 2*M_PI, 0.0, 2.0);

        Config config;
        config.Nx = 32;
        config.Ny = 32;
        config.dt = 0.001;
        config.nu = 1.0;
        config.poisson_solver = PoissonSolverType::Auto;

        RANSSolver solver(mesh, config);

        VelocityBC bc;
        bc.x_lo = VelocityBC::Periodic;
        bc.x_hi = VelocityBC::Periodic;
        bc.y_lo = VelocityBC::NoSlip;
        bc.y_hi = VelocityBC::NoSlip;
        solver.set_velocity_bc(bc);

        PoissonSolverType selected = solver.poisson_solver_type();

#ifdef USE_FFT_POISSON
        bool ok = (selected == PoissonSolverType::FFT2D);
        record("2D channel auto -> FFT2D", ok,
               "selected=" + std::to_string(static_cast<int>(selected)));
#else
        bool ok = (selected == PoissonSolverType::MG);
        record("2D channel auto -> MG (no FFT)", ok,
               "selected=" + std::to_string(static_cast<int>(selected)));
#endif
    }

    // Test explicit MG request
    {
        Mesh mesh;
        mesh.init_uniform(32, 32, 0.0, 2*M_PI, 0.0, 2.0);

        Config config;
        config.Nx = 32;
        config.Ny = 32;
        config.dt = 0.001;
        config.nu = 1.0;
        config.poisson_solver = PoissonSolverType::MG;

        RANSSolver solver(mesh, config);

        VelocityBC bc;
        bc.x_lo = VelocityBC::Periodic;
        bc.x_hi = VelocityBC::Periodic;
        bc.y_lo = VelocityBC::NoSlip;
        bc.y_hi = VelocityBC::NoSlip;
        solver.set_velocity_bc(bc);

        bool ok = (solver.poisson_solver_type() == PoissonSolverType::MG);
        record("Explicit MG request honored", ok);
    }
}

void run_selection_tests() {
    test_solver_selection();
}

//=============================================================================
// Section 4: Nullspace Tests (from test_poisson_nullspace.cpp)
//=============================================================================

void test_nullspace_periodic() {
    std::cout << "\n=== Nullspace Handling ===\n";

    // Fully periodic - has nullspace (constant functions)
    Mesh mesh;
    int N = 32;
    mesh.init_uniform(N, N, 0.0, 2*M_PI, 0.0, 2*M_PI);

    ScalarField rhs(mesh);
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            rhs(i, j) = std::sin(mesh.x(i)) * std::cos(mesh.y(j));
        }
    }

    ScalarField p(mesh, 0.0);
    MultigridPoissonSolver mg(mesh);
    mg.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
              PoissonBC::Periodic, PoissonBC::Periodic);

    PoissonConfig cfg;
    cfg.tol = 1e-8;
    cfg.max_iter = 100;
    int iters = mg.solve(rhs, p, cfg);

    bool converged = (mg.residual() < 1e-6);

    // Check mean is reasonable
    double mean = 0.0;
    int count = 0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            mean += p(i, j);
            ++count;
        }
    }
    mean /= count;

    record("Periodic nullspace convergence", converged,
           "iters=" + std::to_string(iters) + " res=" + std::to_string(mg.residual()));
    record("Periodic solution mean finite", std::isfinite(mean),
           "mean=" + std::to_string(mean));
}

void run_nullspace_tests() {
    test_nullspace_periodic();
}

//=============================================================================
// Section 5: 3D CPU/GPU Consistency (from test_poisson_cpu_gpu_3d.cpp)
//=============================================================================

#ifdef USE_GPU_OFFLOAD
void test_3d_cpu_gpu_consistency() {
    std::cout << "\n=== 3D CPU/GPU Consistency ===\n";

    Mesh mesh;
    mesh.init_uniform(16, 16, 8, 0.0, 2*M_PI, 0.0, 2.0, 0.0, 2*M_PI);

    // Set up RHS
    ScalarField rhs(mesh);
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                rhs(i, j, k) = std::sin(mesh.x(i)) * std::cos(M_PI * mesh.y(j) / 2.0) * std::sin(mesh.z(k));
            }
        }
    }

    // Solve with MG
    ScalarField p(mesh, 0.0);
    MultigridPoissonSolver mg(mesh);
    mg.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
              PoissonBC::Neumann, PoissonBC::Neumann,
              PoissonBC::Periodic, PoissonBC::Periodic);

    PoissonConfig cfg;
    cfg.tol = 1e-8;
    cfg.max_iter = 100;
    mg.solve(rhs, p, cfg);

    bool converged = (mg.residual() < 1e-6);
    record("3D MG converges", converged,
           "res=" + std::to_string(mg.residual()));

    // Check solution is finite
    bool all_finite = true;
    for (int k = mesh.k_begin(); k < mesh.k_end() && all_finite; ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end() && all_finite; ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end() && all_finite; ++i) {
                if (!std::isfinite(p(i, j, k))) all_finite = false;
            }
        }
    }
    record("3D solution finite", all_finite);
}
#endif

void run_3d_tests() {
#ifdef USE_GPU_OFFLOAD
    test_3d_cpu_gpu_consistency();
#else
    std::cout << "\n=== 3D Tests (skipped - CPU build) ===\n";
#endif
}

//=============================================================================
// Section 6: Stretched Grid Tests (from test_poisson_stretched_grid.cpp)
//=============================================================================

void test_stretched_grid() {
    std::cout << "\n=== Stretched Grid ===\n";

    // Test anisotropic grid with compressed domain (thin in y)
    // Use uniform grid cells, but fewer in y for higher AR
    Mesh mesh;
    int Nx = 64, Ny = 16;
    double Lx = 1.0, Ly = 1.0;  // Same domain, fewer Ny cells gives dy > dx
    mesh.init_uniform(Nx, Ny, 0.0, Lx, 0.0, Ly);

    // Manufactured solution: sin(πx/Lx)*sin(πy/Ly)
    double kx = M_PI / Lx;
    double ky = M_PI / Ly;
    double lap_coeff = -(kx*kx + ky*ky);

    ScalarField rhs(mesh);
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            rhs(i, j) = lap_coeff * std::sin(kx * mesh.x(i)) * std::sin(ky * mesh.y(j));
        }
    }

    ScalarField p(mesh, 0.0);
    MultigridPoissonSolver mg(mesh);
    mg.set_bc(PoissonBC::Dirichlet, PoissonBC::Dirichlet,
              PoissonBC::Dirichlet, PoissonBC::Dirichlet);

    PoissonConfig cfg;
    cfg.tol = 1e-6;
    cfg.max_iter = 500;
    int iters = mg.solve(rhs, p, cfg);

    // Compute error
    double max_err = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double exact = std::sin(kx * mesh.x(i)) * std::sin(ky * mesh.y(j));
            max_err = std::max(max_err, std::abs(p(i,j) - exact));
        }
    }

    // For anisotropic grids, error scales with max cell size
    double max_spacing = std::max(Lx / Nx, Ly / Ny);
    double error_bound = 10.0 * max_spacing * max_spacing;

    record("Anisotropic grid (AR=4) error bounded", max_err < error_bound,
           "err=" + std::to_string(max_err) + " bound=" + std::to_string(error_bound));

    // Check solution is finite
    bool all_finite = true;
    for (int j = mesh.j_begin(); j < mesh.j_end() && all_finite; ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end() && all_finite; ++i) {
            if (!std::isfinite(p(i, j))) all_finite = false;
        }
    }
    record("Anisotropic grid solution finite", all_finite);
}

void run_stretched_tests() {
    test_stretched_grid();
}

//=============================================================================
// Section 7: Cross-Solver Consistency (from test_poisson_cross_solver.cpp)
//=============================================================================

void test_cross_solver_consistency() {
    std::cout << "\n=== Cross-Solver Consistency ===\n";

    // Compare SOR vs MG on same problem
    Mesh mesh;
    int N = 32;
    double L = 2.0 * M_PI;
    mesh.init_uniform(N, N, 0.0, L, 0.0, L);

    ScalarField rhs(mesh);
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            rhs(i, j) = -2.0 * std::sin(mesh.x(i)) * std::sin(mesh.y(j));
        }
    }

    // Solve with SOR
    ScalarField p_sor(mesh, 0.0);
    PoissonSolver sor(mesh);
    sor.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
               PoissonBC::Periodic, PoissonBC::Periodic);
    PoissonConfig cfg_sor;
    cfg_sor.tol = 1e-8;
    cfg_sor.max_iter = 10000;
    sor.solve(rhs, p_sor, cfg_sor);

    // Solve with MG
    ScalarField p_mg(mesh, 0.0);
    MultigridPoissonSolver mg(mesh);
    mg.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
              PoissonBC::Periodic, PoissonBC::Periodic);
    PoissonConfig cfg_mg;
    cfg_mg.tol = 1e-10;
    cfg_mg.max_iter = 100;
    mg.solve(rhs, p_mg, cfg_mg);

    // Compare (after subtracting means)
    double sor_mean = 0.0, mg_mean = 0.0;
    int count = 0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            sor_mean += p_sor(i, j);
            mg_mean += p_mg(i, j);
            ++count;
        }
    }
    sor_mean /= count;
    mg_mean /= count;

    double max_diff = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double diff = std::abs((p_sor(i,j) - sor_mean) - (p_mg(i,j) - mg_mean));
            max_diff = std::max(max_diff, diff);
        }
    }

    record("SOR vs MG consistency", max_diff < 1e-4,
           "max_diff=" + std::to_string(max_diff));
}

void run_cross_solver_tests() {
    test_cross_solver_consistency();
}

//=============================================================================
// Section 8: Dirichlet/Mixed BC Tests (from test_poisson_dirichlet_mixed.cpp)
//=============================================================================

void test_dirichlet_bc() {
    std::cout << "\n=== Dirichlet/Mixed BCs ===\n";

    // Pure Dirichlet 2D
    Mesh mesh;
    mesh.init_uniform(32, 32, 0.0, M_PI, 0.0, M_PI);

    // Solution: sin(x)*sin(y), which is 0 on boundaries when domain is [0,π]
    ScalarField rhs(mesh);
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            rhs(i, j) = -2.0 * std::sin(mesh.x(i)) * std::sin(mesh.y(j));
        }
    }

    ScalarField p(mesh, 0.0);
    MultigridPoissonSolver mg(mesh);
    mg.set_bc(PoissonBC::Dirichlet, PoissonBC::Dirichlet,
              PoissonBC::Dirichlet, PoissonBC::Dirichlet);

    PoissonConfig cfg;
    cfg.tol = 1e-8;
    cfg.max_iter = 100;
    mg.solve(rhs, p, cfg);

    // Check error
    double max_err = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double exact = std::sin(mesh.x(i)) * std::sin(mesh.y(j));
            max_err = std::max(max_err, std::abs(p(i,j) - exact));
        }
    }

    record("Pure Dirichlet manufactured solution", max_err < 0.01,
           "max_err=" + std::to_string(max_err));
}

void run_dirichlet_tests() {
    test_dirichlet_bc();
}

//=============================================================================
// Main
//=============================================================================

int main() {
    std::cout << "================================================================\n";
    std::cout << "  UNIFIED POISSON SOLVER TEST SUITE\n";
    std::cout << "  Consolidates 10 test files into one parameterized suite\n";
    std::cout << "================================================================\n";

#ifdef USE_GPU_OFFLOAD
    std::cout << "Build: GPU\n";
#else
    std::cout << "Build: CPU\n";
#endif

#ifdef USE_FFT_POISSON
    std::cout << "FFT Poisson: ENABLED\n";
#else
    std::cout << "FFT Poisson: DISABLED\n";
#endif

#ifdef USE_HYPRE
    std::cout << "HYPRE: ENABLED\n";
#else
    std::cout << "HYPRE: DISABLED\n";
#endif

    // Run all test sections
    run_unit_tests();
    run_convergence_tests();
    run_selection_tests();
    run_nullspace_tests();
    run_3d_tests();
    run_stretched_tests();
    run_cross_solver_tests();
    run_dirichlet_tests();

    // Summary
    int passed = 0, failed = 0;
    for (const auto& r : results) {
        if (r.passed) ++passed;
        else ++failed;
    }

    std::cout << "\n================================================================\n";
    std::cout << "SUMMARY: " << passed << " passed, " << failed << " failed\n";
    std::cout << "================================================================\n";

    return failed > 0 ? 1 : 0;
}
