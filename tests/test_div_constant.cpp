/// @file test_div_constant.cpp
/// @brief Check discrete divergence of constant velocity field

#include <iostream>
#include <iomanip>
#include <cmath>
#include "mesh.hpp"
#include "fields.hpp"

using namespace nncfd;

int main() {
    std::cout << "================================================================\n";
    std::cout << "  Discrete Divergence Check for Constant Velocity\n";
    std::cout << "================================================================\n\n";

    const int N = 8;
    const double u_const = 1.5;
    const double v_const = 0.75;

    Mesh mesh;
    mesh.init_uniform(N, N, 0.0, 2.0*M_PI, 0.0, 2.0*M_PI);

    VectorField velocity(mesh);
    ScalarField divergence(mesh);

    // Set constant velocity everywhere (including ghost cells)
    for (int j = 0; j <= mesh.Ny + 1; ++j) {
        for (int i = 0; i <= mesh.Nx + 1; ++i) {
            velocity.u(i, j) = u_const;
            velocity.v(i, j) = v_const;
        }
    }

    // Compute discrete divergence at cell centers
    // div = (u[i+1,j] - u[i,j])/dx + (v[i,j+1] - v[i,j])/dy
    double max_div = 0.0;
    double sum_div = 0.0;
    int count = 0;

    std::cout << "Discrete divergence at each cell:\n\n";
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        std::cout << "j=" << j << ": ";
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double dudx = (velocity.u(i+1, j) - velocity.u(i, j)) / mesh.dx;
            double dvdy = (velocity.v(i, j+1) - velocity.v(i, j)) / mesh.dy;
            double div = dudx + dvdy;

            divergence(i, j) = div;
            max_div = std::max(max_div, std::abs(div));
            sum_div += div;
            count++;

            std::cout << std::setw(12) << std::scientific << std::setprecision(3) << div;
        }
        std::cout << "\n";
    }

    double mean_div = sum_div / count;

    // Fail fast if divergence is non-zero
    const double tol = 1e-12;
    if (max_div > tol || std::abs(mean_div) > tol) {
        std::cerr << "\n[FAIL] Divergence too large: max=" << max_div
                  << " mean=" << mean_div << " (tol=" << tol << ")\n";
        return 1;
    }

    std::cout << "\nMax |div|: " << max_div << "\n";
    std::cout << "Mean div: " << mean_div << "\n";
    std::cout << "Sum div: " << sum_div << "\n\n";

    // Now check the u and v values to make sure they're actually constant
    std::cout << "Sample u values at j=" << mesh.j_begin() << ":\n";
    for (int i = mesh.i_begin(); i <= mesh.i_end() + 1; ++i) {
        std::cout << "  u(" << i << ")=" << std::fixed << std::setprecision(10)
                  << velocity.u(i, mesh.j_begin()) << "\n";
    }

    std::cout << "\nSample v values at i=" << mesh.i_begin() << ":\n";
    for (int j = mesh.j_begin(); j <= mesh.j_end() + 1; ++j) {
        std::cout << "  v(" << j << ")=" << std::fixed << std::setprecision(10)
                  << velocity.v(mesh.i_begin(), j) << "\n";
    }

    // Check if ghost cells are set correctly
    std::cout << "\nGhost cell check:\n";
    std::cout << "  u(0, 1) = " << velocity.u(0, 1) << " (ghost)\n";
    std::cout << "  u(1, 1) = " << velocity.u(1, 1) << " (interior)\n";
    std::cout << "  u(Nx+1, 1) = " << velocity.u(mesh.Nx+1, 1) << " (ghost)\n";

    return 0;
}
