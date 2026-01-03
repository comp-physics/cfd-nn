/// @file test_poisson_cross_solver.cpp
/// @brief Cross-solver consistency test for Poisson solvers
///
/// CRITICAL TEST: Different Poisson solvers (FFT, FFT1D, HYPRE, MG) should
/// produce equivalent solutions for the same problem. This test catches:
///   - Discretization mismatches between solvers
///   - BC handling differences
///   - Scale factor or sign errors
///
/// Method:
///   1. Run the same problem with all available solvers
///   2. Compare solutions pairwise
///   3. Assert relative difference < tolerance
///
/// Note: Uses manufactured solutions where the exact answer is known.

#include "mesh.hpp"
#include "fields.hpp"
#include "poisson_solver_multigrid.hpp"
#ifdef USE_HYPRE
#include "poisson_solver_hypre.hpp"
#endif
#ifdef USE_FFT_POISSON
#include "poisson_solver_fft.hpp"
#include "poisson_solver_fft1d.hpp"
#endif
#include <iostream>
#include <cmath>
#include <iomanip>
#include <vector>
#include <string>
#include <memory>

using namespace nncfd;

// ============================================================================
// Manufactured solutions
// ============================================================================

// Fully periodic solution: sin(x)*sin(y) on [0, 2π]^2
struct PeriodicSolution2D {
    static double p(double x, double y) {
        return std::sin(x) * std::sin(y);
    }
    static double rhs(double x, double y) {
        return -2.0 * std::sin(x) * std::sin(y);  // -∆p
    }
};

// Fully periodic 3D: sin(x)*sin(y)*sin(z) on [0, 2π]^3
struct PeriodicSolution3D {
    static double p(double x, double y, double z) {
        return std::sin(x) * std::sin(y) * std::sin(z);
    }
    static double rhs(double x, double y, double z) {
        return -3.0 * std::sin(x) * std::sin(y) * std::sin(z);  // -∆p
    }
};

// Channel-like: periodic x/z, Neumann y
struct ChannelSolution3D {
    static double p(double x, double y, double z, double Ly) {
        // cos(πy/Ly) has zero normal derivative at y=0 and y=Ly
        return std::sin(x) * std::cos(M_PI * y / Ly) * std::sin(z);
    }
    static double rhs(double x, double y, double z, double Ly) {
        double ky = M_PI / Ly;
        return -(2.0 + ky*ky) * std::sin(x) * std::cos(M_PI * y / Ly) * std::sin(z);
    }
};

// ============================================================================
// Helper functions
// ============================================================================

double compute_l2_diff(const ScalarField& p1, const ScalarField& p2, const Mesh& mesh) {
    double diff = 0.0;
    double norm = 0.0;
    int count = 0;

    if (mesh.is2D()) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double d = p1(i, j) - p2(i, j);
                diff += d * d;
                norm += p1(i, j) * p1(i, j);
                ++count;
            }
        }
    } else {
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    double d = p1(i, j, k) - p2(i, j, k);
                    diff += d * d;
                    norm += p1(i, j, k) * p1(i, j, k);
                    ++count;
                }
            }
        }
    }

    if (norm < 1e-30) norm = 1.0;  // Avoid division by zero
    return std::sqrt(diff / norm);
}

double compute_max_diff(const ScalarField& p1, const ScalarField& p2, const Mesh& mesh) {
    double max_diff = 0.0;

    if (mesh.is2D()) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double d = std::abs(p1(i, j) - p2(i, j));
                max_diff = std::max(max_diff, d);
            }
        }
    } else {
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    double d = std::abs(p1(i, j, k) - p2(i, j, k));
                    max_diff = std::max(max_diff, d);
                }
            }
        }
    }
    return max_diff;
}

void subtract_mean(ScalarField& p, const Mesh& mesh) {
    double sum = 0.0;
    int count = 0;

    if (mesh.is2D()) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                sum += p(i, j);
                ++count;
            }
        }
        double mean = sum / count;
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                p(i, j) -= mean;
            }
        }
    } else {
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    sum += p(i, j, k);
                    ++count;
                }
            }
        }
        double mean = sum / count;
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    p(i, j, k) -= mean;
                }
            }
        }
    }
}

// ============================================================================
// Test: Fully periodic 2D comparison
// ============================================================================

bool test_periodic_2d() {
    std::cout << "\n  Fully Periodic 2D (all available solvers):\n";

    const int N = 64;
    const double L = 2.0 * M_PI;

    Mesh mesh;
    mesh.init_uniform(N, N, 0.0, L, 0.0, L);

    // Setup RHS
    ScalarField rhs(mesh);
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            rhs(i, j) = PeriodicSolution2D::rhs(mesh.x(i), mesh.y(j));
        }
    }

    PoissonConfig cfg;
    cfg.tol = 1e-8;
    cfg.max_iter = 500;

    std::vector<std::pair<std::string, ScalarField>> solutions;

    // MG solver (always available)
    {
        ScalarField p_mg(mesh, 0.0);
        MultigridPoissonSolver mg(mesh);
        mg.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                  PoissonBC::Periodic, PoissonBC::Periodic);
        mg.solve(rhs, p_mg, cfg);
        subtract_mean(p_mg, mesh);  // Normalize gauge
        solutions.push_back({"MG", p_mg});
        std::cout << "    MG: solved\n";
    }

#ifdef USE_HYPRE
    // HYPRE solver
    {
        ScalarField p_hypre(mesh, 0.0);
        HyprePoissonSolver hypre(mesh);
        hypre.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                     PoissonBC::Periodic, PoissonBC::Periodic);
        hypre.solve(rhs, p_hypre, cfg);
        subtract_mean(p_hypre, mesh);
        solutions.push_back({"HYPRE", p_hypre});
        std::cout << "    HYPRE: solved\n";
    }
#endif

#ifdef USE_FFT_POISSON
    // FFT solver (GPU only, requires fully periodic)
    {
        ScalarField p_fft(mesh, 0.0);
        FFTPoissonSolver fft(mesh);
        // FFT assumes fully periodic
        fft.solve(rhs, p_fft, cfg);
        subtract_mean(p_fft, mesh);
        solutions.push_back({"FFT", p_fft});
        std::cout << "    FFT: solved\n";
    }
#endif

    // Compare all pairs
    bool all_pass = true;
    const double TOL = 1e-4;  // Discretization differences allowed

    for (size_t i = 0; i < solutions.size(); ++i) {
        for (size_t j = i + 1; j < solutions.size(); ++j) {
            double rel_diff = compute_l2_diff(solutions[i].second, solutions[j].second, mesh);
            double max_diff = compute_max_diff(solutions[i].second, solutions[j].second, mesh);

            bool pass = (rel_diff < TOL);
            all_pass = all_pass && pass;

            std::cout << "    " << solutions[i].first << " vs " << solutions[j].first
                      << ": rel=" << std::scientific << std::setprecision(2) << rel_diff
                      << " max=" << max_diff << " ";
            std::cout << (pass ? "[OK]" : "[MISMATCH]") << "\n";
        }
    }

    return all_pass;
}

// ============================================================================
// Test: Fully periodic 3D comparison
// ============================================================================

bool test_periodic_3d() {
    std::cout << "\n  Fully Periodic 3D (all available solvers):\n";

    const int N = 32;
    const double L = 2.0 * M_PI;

    Mesh mesh;
    mesh.init_uniform(N, N, N, 0.0, L, 0.0, L, 0.0, L);

    ScalarField rhs(mesh);
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                rhs(i, j, k) = PeriodicSolution3D::rhs(mesh.x(i), mesh.y(j), mesh.z(k));
            }
        }
    }

    PoissonConfig cfg;
    cfg.tol = 1e-8;
    cfg.max_iter = 500;

    std::vector<std::pair<std::string, ScalarField>> solutions;

    // MG
    {
        ScalarField p(mesh, 0.0);
        MultigridPoissonSolver solver(mesh);
        solver.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                      PoissonBC::Periodic, PoissonBC::Periodic,
                      PoissonBC::Periodic, PoissonBC::Periodic);
        solver.solve(rhs, p, cfg);
        subtract_mean(p, mesh);
        solutions.push_back({"MG", p});
        std::cout << "    MG: solved\n";
    }

#ifdef USE_HYPRE
    {
        ScalarField p(mesh, 0.0);
        HyprePoissonSolver solver(mesh);
        solver.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                      PoissonBC::Periodic, PoissonBC::Periodic,
                      PoissonBC::Periodic, PoissonBC::Periodic);
        solver.solve(rhs, p, cfg);
        subtract_mean(p, mesh);
        solutions.push_back({"HYPRE", p});
        std::cout << "    HYPRE: solved\n";
    }
#endif

#ifdef USE_FFT_POISSON
    {
        ScalarField p(mesh, 0.0);
        FFTPoissonSolver solver(mesh);
        solver.solve(rhs, p, cfg);
        subtract_mean(p, mesh);
        solutions.push_back({"FFT", p});
        std::cout << "    FFT: solved\n";
    }
#endif

    // Compare
    bool all_pass = true;
    const double TOL = 1e-4;

    for (size_t i = 0; i < solutions.size(); ++i) {
        for (size_t j = i + 1; j < solutions.size(); ++j) {
            double rel_diff = compute_l2_diff(solutions[i].second, solutions[j].second, mesh);
            double max_diff = compute_max_diff(solutions[i].second, solutions[j].second, mesh);

            bool pass = (rel_diff < TOL);
            all_pass = all_pass && pass;

            std::cout << "    " << solutions[i].first << " vs " << solutions[j].first
                      << ": rel=" << std::scientific << std::setprecision(2) << rel_diff
                      << " max=" << max_diff << " ";
            std::cout << (pass ? "[OK]" : "[MISMATCH]") << "\n";
        }
    }

    return all_pass;
}

// ============================================================================
// Test: Channel-like 3D (periodic x/z, Neumann y) - MG vs HYPRE
// ============================================================================

bool test_channel_3d() {
    std::cout << "\n  Channel 3D (periodic x/z, Neumann y):\n";

    const int N = 32;
    const double Lx = 2.0 * M_PI;
    const double Ly = 2.0;
    const double Lz = 2.0 * M_PI;

    Mesh mesh;
    mesh.init_uniform(N, N, N, 0.0, Lx, 0.0, Ly, 0.0, Lz);

    ScalarField rhs(mesh);
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                rhs(i, j, k) = ChannelSolution3D::rhs(mesh.x(i), mesh.y(j), mesh.z(k), Ly);
            }
        }
    }

    PoissonConfig cfg;
    cfg.tol = 1e-8;
    cfg.max_iter = 500;

    std::vector<std::pair<std::string, ScalarField>> solutions;

    // MG
    {
        ScalarField p(mesh, 0.0);
        MultigridPoissonSolver solver(mesh);
        solver.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,   // x
                      PoissonBC::Neumann, PoissonBC::Neumann,     // y
                      PoissonBC::Periodic, PoissonBC::Periodic);  // z
        solver.solve(rhs, p, cfg);
        subtract_mean(p, mesh);
        solutions.push_back({"MG", p});
        std::cout << "    MG: solved\n";
    }

#ifdef USE_HYPRE
    {
        ScalarField p(mesh, 0.0);
        HyprePoissonSolver solver(mesh);
        solver.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                      PoissonBC::Neumann, PoissonBC::Neumann,
                      PoissonBC::Periodic, PoissonBC::Periodic);
        solver.solve(rhs, p, cfg);
        subtract_mean(p, mesh);
        solutions.push_back({"HYPRE", p});
        std::cout << "    HYPRE: solved\n";
    }
#endif

#ifdef USE_FFT_POISSON
    // FFT1D can handle periodic x/z with non-periodic y (tridiagonal in y)
    {
        ScalarField p(mesh, 0.0);
        FFT1DPoissonSolver solver(mesh);
        // FFT1D: periodic in x/z, tridiagonal in y
        solver.set_bc_y(PoissonBC::Neumann, PoissonBC::Neumann);
        solver.solve(rhs, p, cfg);
        subtract_mean(p, mesh);
        solutions.push_back({"FFT1D", p});
        std::cout << "    FFT1D: solved\n";
    }
#endif

    // Compare
    bool all_pass = true;
    const double TOL = 1e-4;

    for (size_t i = 0; i < solutions.size(); ++i) {
        for (size_t j = i + 1; j < solutions.size(); ++j) {
            double rel_diff = compute_l2_diff(solutions[i].second, solutions[j].second, mesh);
            double max_diff = compute_max_diff(solutions[i].second, solutions[j].second, mesh);

            bool pass = (rel_diff < TOL);
            all_pass = all_pass && pass;

            std::cout << "    " << solutions[i].first << " vs " << solutions[j].first
                      << ": rel=" << std::scientific << std::setprecision(2) << rel_diff
                      << " max=" << max_diff << " ";
            std::cout << (pass ? "[OK]" : "[MISMATCH]") << "\n";
        }
    }

    return all_pass;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "================================================================\n";
    std::cout << "  Cross-Solver Consistency Test\n";
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
#ifdef USE_FFT_POISSON
    std::cout << "FFT: enabled\n";
#else
    std::cout << "FFT: disabled (GPU only)\n";
#endif

    std::cout << "\nComparing solutions from different Poisson solvers.\n";
    std::cout << "All solvers should produce equivalent results for the same problem.\n";

    int passed = 0, failed = 0;

    // Test cases
    std::vector<std::pair<std::string, bool(*)()>> tests = {
        {"Periodic 2D", test_periodic_2d},
        {"Periodic 3D", test_periodic_3d},
        {"Channel 3D", test_channel_3d},
    };

    for (const auto& [name, test_fn] : tests) {
        bool ok = test_fn();
        if (ok) {
            std::cout << "  => " << name << ": [PASS]\n";
            ++passed;
        } else {
            std::cout << "  => " << name << ": [FAIL]\n";
            ++failed;
        }
    }

    // Summary
    std::cout << "\n================================================================\n";
    std::cout << "Cross-Solver Consistency Summary\n";
    std::cout << "================================================================\n";
    std::cout << "  Passed: " << passed << "/" << (passed + failed) << "\n";
    std::cout << "  Failed: " << failed << "/" << (passed + failed) << "\n";

    if (failed == 0) {
        std::cout << "\n[PASS] All cross-solver consistency tests passed\n";
        return 0;
    } else {
        std::cout << "\n[FAIL] " << failed << " cross-solver test(s) failed\n";
        std::cout << "       Solvers producing different solutions for the same problem!\n";
        return 1;
    }
}
