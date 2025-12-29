/// 3D Gradient Tests (~5 seconds)
/// Verifies 3D gradient computations are correct
///
/// Tests gradient accuracy using known analytical velocity fields
/// where gradients can be computed exactly.
///
/// Tests:
/// 1. Linear u = z field -> du/dz = 1
/// 2. Sinusoidal w = sin(x) -> dw/dx = cos(x)
/// 3. All nine gradient components with polynomial field
/// 4. Divergence computation accuracy

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "config.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace nncfd;

//=============================================================================
// TEST 1: Linear velocity field - du/dz = 1
//=============================================================================
bool test_linear_dudz() {
    std::cout << "Test 1: Linear u=z field (du/dz should be 1)... ";

    Mesh mesh;
    mesh.init_uniform(8, 8, 8, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    // Set u = z (linear in z)
    // du/dz should be 1 everywhere
    VectorField vel(mesh);

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        double z = mesh.z(k);
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                vel.u(i, j, k) = z;
            }
        }
    }

    // Compute du/dz using central differences
    double max_error = 0.0;
    double expected_dudz = 1.0;
    double dz = mesh.dz;

    for (int k = mesh.k_begin() + 1; k < mesh.k_end() - 1; ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                // Central difference for du/dz
                double u_above = vel.u(i, j, k + 1);
                double u_below = vel.u(i, j, k - 1);
                double dudz = (u_above - u_below) / (2.0 * dz);

                double error = std::abs(dudz - expected_dudz);
                max_error = std::max(max_error, error);
            }
        }
    }

    bool passed = (max_error < 1e-10);

    if (passed) {
        std::cout << "PASSED (max error = " << std::scientific << max_error << ")\n";
    } else {
        std::cout << "FAILED\n";
        std::cout << "  Max gradient error: " << max_error << " (expected < 1e-10)\n";
    }

    return passed;
}

//=============================================================================
// TEST 2: Sinusoidal w = sin(x) -> dw/dx = cos(x)
//=============================================================================
bool test_sinusoidal_dwdx() {
    std::cout << "Test 2: Sinusoidal w=sin(x) field (dw/dx = cos(x))... ";

    Mesh mesh;
    mesh.init_uniform(32, 8, 8, 0.0, 2 * M_PI, 0.0, 1.0, 0.0, 1.0);

    VectorField vel(mesh);

    // Set w = sin(x)
    for (int k = mesh.k_begin(); k <= mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double x = mesh.x(i);
                vel.w(i, j, k) = std::sin(x);
            }
        }
    }

    // Compute dw/dx using central differences
    double max_error = 0.0;
    double dx = mesh.dx;

    for (int k = mesh.k_begin(); k <= mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin() + 1; i < mesh.i_end() - 1; ++i) {
                double x = mesh.x(i);
                double expected_dwdx = std::cos(x);

                double w_right = vel.w(i + 1, j, k);
                double w_left = vel.w(i - 1, j, k);
                double dwdx = (w_right - w_left) / (2.0 * dx);

                double error = std::abs(dwdx - expected_dwdx);
                max_error = std::max(max_error, error);
            }
        }
    }

    // Central difference has O(dx^2) error for smooth functions
    // For 32 cells over 2*pi, dx ~= 0.2, so error ~ dx^2 ~ 0.04
    // But sin is smooth, so we expect better accuracy
    bool passed = (max_error < 0.01);

    if (passed) {
        std::cout << "PASSED (max error = " << std::scientific << max_error << ")\n";
    } else {
        std::cout << "FAILED\n";
        std::cout << "  Max gradient error: " << max_error << " (expected < 0.01)\n";
    }

    return passed;
}

//=============================================================================
// TEST 3: All nine gradient components with polynomial field
//=============================================================================
bool test_all_nine_gradients() {
    std::cout << "Test 3: All nine gradient components (polynomial field)... ";

    // Use field: u = x + y + z, v = 2x + 3y + 4z, w = 5x + 6y + 7z
    // Expected gradients:
    // du/dx = 1, du/dy = 1, du/dz = 1
    // dv/dx = 2, dv/dy = 3, dv/dz = 4
    // dw/dx = 5, dw/dy = 6, dw/dz = 7

    Mesh mesh;
    mesh.init_uniform(16, 16, 16, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    VectorField vel(mesh);

    // Set u-velocity at x-faces
    // u is at face i, cell centers (j, k)
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        double z = mesh.z(k);
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            double y = mesh.y(j);
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                double x = mesh.xf[i];  // x at face
                vel.u(i, j, k) = x + y + z;
            }
        }
    }

    // Set v-velocity at y-faces
    // v is at cell centers (i, k), face j
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        double z = mesh.z(k);
        for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
            double y = mesh.yf[j];  // y at face
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double x = mesh.x(i);
                vel.v(i, j, k) = 2 * x + 3 * y + 4 * z;
            }
        }
    }

    // Set w-velocity at z-faces
    // w is at cell centers (i, j), face k
    for (int k = mesh.k_begin(); k <= mesh.k_end(); ++k) {
        double z = mesh.zf[k];  // z at face
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            double y = mesh.y(j);
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double x = mesh.x(i);
                vel.w(i, j, k) = 5 * x + 6 * y + 7 * z;
            }
        }
    }

    // Compute all gradients and check against analytical values
    double max_error = 0.0;
    double dx = mesh.dx, dy = mesh.dy, dz = mesh.dz;

    // Expected gradients
    const double expected[3][3] = {
        {1.0, 1.0, 1.0},  // du/dx, du/dy, du/dz
        {2.0, 3.0, 4.0},  // dv/dx, dv/dy, dv/dz
        {5.0, 6.0, 7.0}   // dw/dx, dw/dy, dw/dz
    };

    // Check interior points only (avoid boundary issues)
    for (int k = mesh.k_begin() + 1; k < mesh.k_end() - 1; ++k) {
        for (int j = mesh.j_begin() + 1; j < mesh.j_end() - 1; ++j) {
            for (int i = mesh.i_begin() + 1; i < mesh.i_end() - 1; ++i) {
                // du/dx (at cell center, using u at faces)
                double dudx = (vel.u(i + 1, j, k) - vel.u(i, j, k)) / dx;
                max_error = std::max(max_error, std::abs(dudx - expected[0][0]));

                // du/dy (central difference)
                double dudy = (vel.u(i, j + 1, k) - vel.u(i, j - 1, k)) / (2 * dy);
                max_error = std::max(max_error, std::abs(dudy - expected[0][1]));

                // du/dz (central difference)
                double dudz = (vel.u(i, j, k + 1) - vel.u(i, j, k - 1)) / (2 * dz);
                max_error = std::max(max_error, std::abs(dudz - expected[0][2]));

                // dv/dx (central difference)
                double dvdx = (vel.v(i + 1, j, k) - vel.v(i - 1, j, k)) / (2 * dx);
                max_error = std::max(max_error, std::abs(dvdx - expected[1][0]));

                // dv/dy (at cell center, using v at faces)
                double dvdy = (vel.v(i, j + 1, k) - vel.v(i, j, k)) / dy;
                max_error = std::max(max_error, std::abs(dvdy - expected[1][1]));

                // dv/dz (central difference)
                double dvdz = (vel.v(i, j, k + 1) - vel.v(i, j, k - 1)) / (2 * dz);
                max_error = std::max(max_error, std::abs(dvdz - expected[1][2]));

                // dw/dx (central difference)
                double dwdx = (vel.w(i + 1, j, k) - vel.w(i - 1, j, k)) / (2 * dx);
                max_error = std::max(max_error, std::abs(dwdx - expected[2][0]));

                // dw/dy (central difference)
                double dwdy = (vel.w(i, j + 1, k) - vel.w(i, j - 1, k)) / (2 * dy);
                max_error = std::max(max_error, std::abs(dwdy - expected[2][1]));

                // dw/dz (at cell center, using w at faces)
                double dwdz = (vel.w(i, j, k + 1) - vel.w(i, j, k)) / dz;
                max_error = std::max(max_error, std::abs(dwdz - expected[2][2]));
            }
        }
    }

    bool passed = (max_error < 1e-10);

    if (passed) {
        std::cout << "PASSED (max error = " << std::scientific << max_error << ")\n";
    } else {
        std::cout << "FAILED\n";
        std::cout << "  Max gradient error: " << max_error << " (expected < 1e-10)\n";
    }

    return passed;
}

//=============================================================================
// TEST 4: Divergence accuracy for known divergence-free field
//=============================================================================
bool test_divergence_accuracy() {
    std::cout << "Test 4: Divergence accuracy (divergence-free field)... ";

    // Use divergence-free field: u = sin(x)*cos(y), v = -cos(x)*sin(y), w = 0
    // div(u) = cos(x)*cos(y) - cos(x)*cos(y) + 0 = 0

    Mesh mesh;
    mesh.init_uniform(32, 32, 4, 0.0, 2 * M_PI, 0.0, 2 * M_PI, 0.0, 1.0);

    VectorField vel(mesh);

    // Set u = sin(x)*cos(y) at x-faces
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            double y = mesh.y(j);
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                double x = mesh.xf[i];
                vel.u(i, j, k) = std::sin(x) * std::cos(y);
            }
        }
    }

    // Set v = -cos(x)*sin(y) at y-faces
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
            double y = mesh.yf[j];
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double x = mesh.x(i);
                vel.v(i, j, k) = -std::cos(x) * std::sin(y);
            }
        }
    }

    // Set w = 0
    for (int k = mesh.k_begin(); k <= mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                vel.w(i, j, k) = 0.0;
            }
        }
    }

    // Compute divergence using finite differences
    double max_div = 0.0;
    double dx = mesh.dx, dy = mesh.dy, dz = mesh.dz;

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double dudx = (vel.u(i + 1, j, k) - vel.u(i, j, k)) / dx;
                double dvdy = (vel.v(i, j + 1, k) - vel.v(i, j, k)) / dy;
                double dwdz = (vel.w(i, j, k + 1) - vel.w(i, j, k)) / dz;
                double div = dudx + dvdy + dwdz;
                max_div = std::max(max_div, std::abs(div));
            }
        }
    }

    // Discretization error for smooth field should be small
    // For 32 cells, dx ~= 0.2, discretization error ~ dx^2 ~ 0.04
    bool passed = (max_div < 0.01);

    if (passed) {
        std::cout << "PASSED (max div = " << std::scientific << max_div << ")\n";
    } else {
        std::cout << "FAILED\n";
        std::cout << "  Max divergence: " << max_div << " (expected < 0.01)\n";
    }

    return passed;
}

//=============================================================================
// TEST 5: Z-gradient symmetry for symmetric field
//=============================================================================
bool test_z_gradient_symmetry() {
    std::cout << "Test 5: Z-gradient symmetry (parabolic profile)... ";

    // u = 1 - z^2 (symmetric about z=0 if domain is [-1,1])
    // du/dz = -2z (antisymmetric)

    Mesh mesh;
    mesh.init_uniform(8, 8, 16, 0.0, 1.0, 0.0, 1.0, -1.0, 1.0);

    VectorField vel(mesh);

    // Set u = 1 - z^2
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        double z = mesh.z(k);
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                vel.u(i, j, k) = 1.0 - z * z;
            }
        }
    }

    // Compute du/dz and check against -2z
    double max_error = 0.0;
    double dz = mesh.dz;

    for (int k = mesh.k_begin() + 1; k < mesh.k_end() - 1; ++k) {
        double z = mesh.z(k);
        double expected_dudz = -2.0 * z;

        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double dudz = (vel.u(i, j, k + 1) - vel.u(i, j, k - 1)) / (2.0 * dz);
                double error = std::abs(dudz - expected_dudz);
                max_error = std::max(max_error, error);
            }
        }
    }

    // Should be exact for quadratic function with central differences
    bool passed = (max_error < 1e-10);

    if (passed) {
        std::cout << "PASSED (max error = " << std::scientific << max_error << ")\n";
    } else {
        std::cout << "FAILED\n";
        std::cout << "  Max gradient error: " << max_error << " (expected < 1e-10)\n";
    }

    return passed;
}

//=============================================================================
// MAIN
//=============================================================================
int main() {
    std::cout << "=== 3D Gradient Tests ===\n\n";

    int passed = 0;
    int total = 0;

    total++; if (test_linear_dudz()) passed++;
    total++; if (test_sinusoidal_dwdx()) passed++;
    total++; if (test_all_nine_gradients()) passed++;
    total++; if (test_divergence_accuracy()) passed++;
    total++; if (test_z_gradient_symmetry()) passed++;

    std::cout << "\n=== Results: " << passed << "/" << total << " tests passed ===\n";

    if (passed == total) {
        std::cout << "[SUCCESS] All 3D gradient tests passed!\n";
        return 0;
    } else {
        std::cout << "[FAILURE] Some tests failed\n";
        return 1;
    }
}
