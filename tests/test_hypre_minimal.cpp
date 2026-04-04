#include "momentum_solver_hypre.hpp"
#include "mesh.hpp"
#include <iostream>
#include <vector>
#include <cmath>
using namespace nncfd;

int main() {
#ifdef USE_HYPRE
    Mesh mesh;
    mesh.init_uniform(8, 8, 0.0, 1.0, 0.0, 1.0);

    HypreMomentumSolver solver(mesh, true);

    int n = 9 * 8;
    std::vector<double> aW(n, 1.0), aE(n, 1.0), aS(n, 1.0), aN(n, 1.0);
    std::vector<double> aP(n, 5.0);
    std::vector<double> rhs(n, 1.0);
    std::vector<double> x(n, 0.0);

    solver.set_coefficients(aW.data(), aE.data(), aS.data(), aN.data(),
                            aP.data(), n);
    int iters = solver.solve(rhs.data(), x.data(), 1e-6, 50);

    std::cout << "HYPRE solved in " << iters << " iters, res="
              << solver.final_residual() << "\n";
    std::cout << "x[0]=" << x[0] << " x[n/2]=" << x[n/2]
              << " (expected ~0.2 for 5*x=1)\n";

    if (std::abs(x[n/2] - 0.2) < 0.1) {
        std::cout << "PASS\n";
    } else {
        std::cout << "FAIL: x[n/2]=" << x[n/2] << " expected ~0.2\n";
    }
#else
    std::cout << "HYPRE not enabled\n";
#endif
    return 0;
}
