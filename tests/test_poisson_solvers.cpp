/// Comprehensive tests for Poisson solvers (SOR and Multigrid) in 2D and 3D
/// Uses grid convergence testing to verify 2nd-order accuracy
///
/// REFACTORED: Using test_framework.hpp - reduced from 467 lines to ~80 lines

#include "test_framework.hpp"
#include <cstdlib>

using namespace nncfd;
using namespace nncfd::test;

int main() {
    std::cout << "=== Poisson Solver Convergence Tests ===\n";
    std::cout << "Verifying 2nd-order accuracy via grid refinement\n\n";

    int passed = 0, total = 0;

    auto check = [&](const std::string& name, const ConvergenceResult& r) {
        std::cout << std::left << std::setw(40) << name;
        r.print();
        if (r.passed) ++passed;
        ++total;
    };

    // Manufactured solution: p = sin(x)*sin(y) or sin(x)*sin(y)*sin(z)
    SinSolution sol_2d(1, 1, 0);
    SinSolution sol_3d(1, 1, 1);

    std::cout << "--- 2D Grid Convergence ---\n";

    check("2D SOR (N=16 -> N=32)",
          run_poisson_convergence({16, 32}, sol_2d, TestPoissonSolver::SOR, false));

    check("2D Multigrid (N=32 -> N=64)",
          run_poisson_convergence({32, 64}, sol_2d, TestPoissonSolver::Multigrid, false));

    std::cout << "\n--- 3D Grid Convergence ---\n";

    // Note: 3D SOR is slow (requires 200K iterations for tight tolerance)
    // Skip if QUICK_TEST environment variable is set
    const char* quick = std::getenv("QUICK_TEST");
    if (!quick) {
        check("3D SOR (N=8 -> N=16)",
              run_poisson_convergence({8, 16}, sol_3d, TestPoissonSolver::SOR, true));
    } else {
        std::cout << std::left << std::setw(40) << "3D SOR (N=8 -> N=16)"
                  << "SKIPPED (QUICK_TEST)\n";
    }

    check("3D Multigrid (N=16 -> N=32)",
          run_poisson_convergence({16, 32}, sol_3d, TestPoissonSolver::Multigrid, true));

    // Solver consistency tests (SOR vs Multigrid should give same answer)
    std::cout << "\n--- Solver Consistency ---\n";

    auto check_consistency = [&](const std::string& name, int N, bool is_3d) {
        // Skip 3D SOR tests in quick mode
        if (is_3d && quick) {
            std::cout << std::left << std::setw(40) << name
                      << "SKIPPED (QUICK_TEST)\n";
            return;
        }
        auto r1 = run_poisson_convergence({N}, is_3d ? sol_3d : sol_2d,
                                          TestPoissonSolver::SOR, is_3d);
        auto r2 = run_poisson_convergence({N}, is_3d ? sol_3d : sol_2d,
                                          TestPoissonSolver::Multigrid, is_3d);
        double diff = std::abs(r1.errors[0] - r2.errors[0]);
        bool ok = diff < 1e-4;
        std::cout << std::left << std::setw(40) << name
                  << (ok ? "PASSED" : "FAILED")
                  << " (diff=" << std::scientific << diff << ")\n";
        if (ok) ++passed;
        ++total;
    };

    check_consistency("2D SOR vs Multigrid (N=32)", 32, false);
    check_consistency("3D SOR vs Multigrid (N=16)", 16, true);

    std::cout << "\n=== Results: " << passed << "/" << total << " tests passed ===\n";

    if (passed == total) {
        std::cout << "[SUCCESS] All Poisson solver convergence tests passed!\n";
        return 0;
    } else {
        std::cout << "[FAILURE] Some tests failed!\n";
        return 1;
    }
}
