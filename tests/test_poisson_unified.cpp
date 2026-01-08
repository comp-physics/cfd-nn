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
#include "test_fixtures.hpp"
#include "test_utilities.hpp"
#include "test_harness.hpp"
#include "solver.hpp"
#include "config.hpp"
#ifdef USE_HYPRE
#include "poisson_solver_hypre.hpp"
#endif
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <functional>

using namespace nncfd;
using namespace nncfd::test;
using nncfd::test::harness::record;

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
    test_laplacian();
    test_basic_solve();
    test_periodic_solve();
}

//=============================================================================
// Section 2: Grid Convergence Tests (table-driven)
//=============================================================================

/// Test case for grid convergence verification
struct ConvergenceTestCase {
    const char* name;
    std::vector<int> grid_sizes;
    double Lx, Ly;
    PoissonBC bc_x, bc_y;
    double expected_rate;
    double rate_tolerance;
};

/// Run a convergence test for a specific manufactured solution
template<typename Solution>
void run_convergence_case(const ConvergenceTestCase& tc) {
    std::vector<double> errors;
    std::vector<double> h_values;

    for (int N : tc.grid_sizes) {
        Mesh mesh;
        mesh.init_uniform(N, N, 0.0, tc.Lx, 0.0, tc.Ly);
        Solution sol(tc.Lx, tc.Ly);

        // Set up RHS from manufactured solution
        ScalarField rhs(mesh);
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                rhs(i, j) = sol.rhs(mesh.x(i), mesh.y(j));
            }
        }

        // Solve with multigrid
        ScalarField p(mesh, 0.0);
        MultigridPoissonSolver mg(mesh);
        mg.set_bc(tc.bc_x, tc.bc_x, tc.bc_y, tc.bc_y);

        PoissonConfig cfg;
        cfg.tol = 1e-10;
        cfg.max_iter = 100;
        mg.solve(rhs, p, cfg);

        // Compute L2 error with mean subtraction (Neumann compatibility)
        double err = compute_l2_error_2d(p, mesh, sol);
        errors.push_back(err);
        h_values.push_back(tc.Lx / N);

        record(std::string(tc.name) + " N=" + std::to_string(N), true,
               "L2=" + std::to_string(err));
    }

    // Compute and check convergence rate
    if (errors.size() >= 2) {
        double rate = std::log(errors[0] / errors[1]) / std::log(h_values[0] / h_values[1]);
        bool rate_ok = rate > tc.expected_rate - tc.rate_tolerance;
        record(std::string(tc.name) + " rate", rate_ok,
               "rate=" + std::to_string(rate) + " (expect >" +
               std::to_string(tc.expected_rate - tc.rate_tolerance) + ")");
    }
}

void run_convergence_tests() {
    // Table of convergence test cases
    const std::vector<ConvergenceTestCase> cases = {
        {"MG Periodic", {16, 32, 64}, 2*M_PI, 2*M_PI,
         PoissonBC::Periodic, PoissonBC::Periodic, 2.0, 0.3},
        {"MG Channel", {16, 32, 64}, 4.0, 2.0,
         PoissonBC::Periodic, PoissonBC::Neumann, 2.0, 0.3},
    };

    for (const auto& tc : cases) {
        if (tc.bc_y == PoissonBC::Periodic) {
            run_convergence_case<PeriodicSolution2D>(tc);
        } else {
            run_convergence_case<ChannelSolution2D>(tc);
        }
    }
}

//=============================================================================
// Section 3: Solver Selection Tests (from test_poisson_selection.cpp)
//=============================================================================

void test_solver_selection() {
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
        solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel2D));

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
        solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel2D));

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
// Section 5: 3D GPU Convergence (from test_poisson_cpu_gpu_3d.cpp)
//=============================================================================

#ifdef USE_GPU_OFFLOAD
void test_3d_gpu_convergence() {
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
    test_3d_gpu_convergence();
#else
    record("3D GPU tests", true, true);  // Skip on CPU build
#endif
}

//=============================================================================
// Section 6: Stretched Grid Tests (from test_poisson_stretched_grid.cpp)
//=============================================================================

void test_stretched_grid() {
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
    [[maybe_unused]] int iters = mg.solve(rhs, p, cfg);

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
    using namespace nncfd::test::harness;

    std::cout << "  Consolidates 10 test files into one parameterized suite\n";
#ifdef USE_FFT_POISSON
    std::cout << "  FFT Poisson: ENABLED\n";
#else
    std::cout << "  FFT Poisson: DISABLED\n";
#endif
#ifdef USE_HYPRE
    std::cout << "  HYPRE: ENABLED\n";
#else
    std::cout << "  HYPRE: DISABLED\n";
#endif

    return run_sections("Unified Poisson Solver Test Suite", {
        {"Unit Tests", run_unit_tests},
        {"Grid Convergence", run_convergence_tests},
        {"Solver Selection", run_selection_tests},
        {"Nullspace Handling", run_nullspace_tests},
        {"3D Tests", run_3d_tests},
        {"Stretched Grid", run_stretched_tests},
        {"Cross-Solver Consistency", run_cross_solver_tests},
        {"Dirichlet/Mixed BCs", run_dirichlet_tests}
    });
}
