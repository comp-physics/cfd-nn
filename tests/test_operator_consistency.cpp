// Test A: Verify D(G(φ)) = L(φ) for stretched y-grid
// This tests discrete operator consistency required for projection step
#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include "mesh.hpp"

using namespace nncfd;

// Compute gradient in y at face j: (phi[j] - phi[j-1]) / dyc[j]
double grad_y_at_face(const std::vector<double>& phi, int j, int stride,
                       const std::vector<double>& dyc, double dy_uniform, bool stretched) {
    double phi_top = phi[j * stride];
    double phi_bot = phi[(j-1) * stride];
    double spacing = stretched ? dyc[j] : dy_uniform;
    return (phi_top - phi_bot) / spacing;
}

// Compute divergence of gradient in y at cell j: (grad[j+1] - grad[j]) / dyv[j]
double div_grad_y(const std::vector<double>& phi, int j, int stride,
                   const std::vector<double>& dyc, const std::vector<double>& dyv,
                   double dy_uniform, bool stretched) {
    double grad_top = grad_y_at_face(phi, j+1, stride, dyc, dy_uniform, stretched);
    double grad_bot = grad_y_at_face(phi, j, stride, dyc, dy_uniform, stretched);
    double cell_height = stretched ? dyv[j] : dy_uniform;
    return (grad_top - grad_bot) / cell_height;
}

// Compute Laplacian in y using aS, aP, aN coefficients
double laplacian_y(const std::vector<double>& phi, int j, int stride,
                    const std::vector<double>& aS, const std::vector<double>& aP,
                    const std::vector<double>& aN, double dy_uniform, bool stretched) {
    double phi_j = phi[j * stride];
    double phi_jm1 = phi[(j-1) * stride];
    double phi_jp1 = phi[(j+1) * stride];

    if (stretched) {
        return aS[j] * phi_jm1 + aP[j] * phi_j + aN[j] * phi_jp1;
    } else {
        double dy2 = dy_uniform * dy_uniform;
        return (phi_jp1 - 2.0 * phi_j + phi_jm1) / dy2;
    }
}

int main() {
    std::cout << "=== Test A: D(G(phi)) vs L(phi) Operator Consistency ===\n\n";

    // Test parameters
    const int Ny = 32;
    const int Ng = 2;
    const double y_min = -1.0, y_max = 1.0;
    const double beta = 2.0;  // Stretching parameter

    // Create stretched mesh
    Mesh mesh;
    mesh.init_stretched_y(1, Ny, y_min, y_max, y_min, y_max, Mesh::tanh_stretching(beta), Ng);

    int total_ny = mesh.total_Ny();
    int stride = 1;  // 1D test (only y direction)

    std::cout << "Grid: Ny=" << Ny << ", Ng=" << Ng << ", total_ny=" << total_ny << "\n";
    std::cout << "Stretching: beta=" << beta << "\n";
    std::cout << "y_stretched: " << (mesh.is_y_stretched() ? "YES" : "NO") << "\n\n";

    // Print some y-metrics for verification
    std::cout << "Y-metric samples:\n";
    std::cout << "  dyv[Ng]=" << mesh.dyv[Ng] << " (first interior cell)\n";
    std::cout << "  dyv[Ng+Ny/2]=" << mesh.dyv[Ng + Ny/2] << " (mid-channel)\n";
    std::cout << "  dyv[Ng+Ny-1]=" << mesh.dyv[Ng + Ny - 1] << " (last interior cell)\n";
    std::cout << "  dyc[Ng]=" << mesh.dyc[Ng] << "\n";
    std::cout << "  dyc[Ng+Ny/2]=" << mesh.dyc[Ng + Ny/2] << "\n";
    std::cout << "  yLap_aS[Ng]=" << mesh.yLap_aS[Ng] << "\n";
    std::cout << "  yLap_aN[Ng]=" << mesh.yLap_aN[Ng] << "\n";
    std::cout << "  yLap_aP[Ng]=" << mesh.yLap_aP[Ng] << " (should be -(aS+aN)="
              << -(mesh.yLap_aS[Ng] + mesh.yLap_aN[Ng]) << ")\n\n";

    // Create random test field (interior only, zero at boundaries)
    std::vector<double> phi(total_ny, 0.0);
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int j = Ng; j < Ng + Ny; ++j) {
        phi[j] = dist(rng);
    }
    // Set ghost cells to satisfy Neumann BC (zero gradient)
    phi[Ng - 1] = phi[Ng];
    phi[Ng - 2] = phi[Ng + 1];
    phi[Ng + Ny] = phi[Ng + Ny - 1];
    phi[Ng + Ny + 1] = phi[Ng + Ny - 2];

    // Compute D(G(phi)) and L(phi) for interior cells
    double max_error = 0.0;
    double l2_error = 0.0;
    double l2_norm_L = 0.0;
    int error_count = 0;

    std::cout << "Comparing D(G(phi)) vs L(phi) for interior cells:\n";
    std::cout << "  j     D(G(phi))      L(phi)       Error\n";
    std::cout << "  ---   ---------   ---------   ---------\n";

    for (int j = Ng; j < Ng + Ny; ++j) {
        double dg = div_grad_y(phi, j, stride, mesh.dyc, mesh.dyv, mesh.dy, true);
        double L = laplacian_y(phi, j, stride, mesh.yLap_aS, mesh.yLap_aP, mesh.yLap_aN, mesh.dy, true);

        double error = std::abs(dg - L);
        max_error = std::max(max_error, error);
        l2_error += error * error;
        l2_norm_L += L * L;

        if (error > 1e-10) {
            error_count++;
            if (error_count <= 5) {
                std::cout << "  " << j << "   " << dg << "   " << L << "   " << error << "\n";
            }
        }
    }

    l2_error = std::sqrt(l2_error);
    l2_norm_L = std::sqrt(l2_norm_L);
    double rel_error = (l2_norm_L > 1e-30) ? l2_error / l2_norm_L : l2_error;

    std::cout << "\n=== Results ===\n";
    std::cout << "Max |D(G(phi)) - L(phi)|: " << max_error << "\n";
    std::cout << "L2 error:             " << l2_error << "\n";
    std::cout << "L2 norm of L(phi):      " << l2_norm_L << "\n";
    std::cout << "Relative error:       " << rel_error << "\n";
    std::cout << "Cells with error > 1e-10: " << error_count << "/" << Ny << "\n\n";

    // Test verdict
    bool passed = (rel_error < 1e-10);
    std::cout << "=== " << (passed ? "PASS" : "FAIL") << " ===\n";
    std::cout << "D(G(phi)) = L(phi) consistency: " << (passed ? "YES" : "NO") << "\n";

    if (!passed) {
        std::cout << "\nDiagnosis: The discrete operators are NOT consistent.\n";
        std::cout << "Check the formulas for dyc, dyv, aS, aN, aP.\n";
    }

    return passed ? 0 : 1;
}
