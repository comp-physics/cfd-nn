/// @file test_velocity_gradient.cpp
/// @brief Tests for velocity gradient tensor computation
///
/// Test coverage:
///   1. Linear u field: du/dx exact to machine precision
///   2. Linear v field: dv/dy exact to machine precision
///   3. Pure shear: du/dy and dv/dx from linear cross-terms
///   4. 3D: dw/dz from linear w field
///   5. Continuity: du/dx + dv/dy + dw/dz = 0 for divergence-free field

#include "velocity_gradient.hpp"
#include "mesh.hpp"
#include "fields.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace nncfd;

#define CHECK(cond, msg) do { if (!(cond)) throw std::runtime_error(msg); } while(0)

void test_dudx_linear() {
    // u = 2*x → du/dx = 2 (exact for 2nd-order FD on linear field)
    const int Nx = 16, Ny = 16;
    Mesh mesh;
    mesh.init_uniform(Nx, Ny, 0.0, 1.0, 0.0, 1.0);

    VectorField vel(mesh);
    const int Ng = mesh.Nghost;

    // Set u = 2*x at x-faces
    for (int j = 0; j < Ny + 2 * Ng; ++j) {
        for (int i = 0; i < Nx + 1 + 2 * Ng; ++i) {
            vel.u(i, j) = 2.0 * mesh.xf[i];
        }
    }

    VelocityGradient grad_comp;
    VelocityGradientTensor grad;
    grad_comp.compute(mesh, vel, grad);

    double max_err = 0.0;
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            int idx = grad.index(i, j, 0);
            max_err = std::max(max_err, std::abs(grad.g11[idx] - 2.0));
        }
    }

    CHECK(max_err < 1e-12, "du/dx must be exact for linear u field");
    std::cout << "PASS: du/dx linear (max_err=" << max_err << ")" << std::endl;
}

void test_dvdy_linear() {
    // v = 3*y → dv/dy = 3
    const int Nx = 16, Ny = 16;
    Mesh mesh;
    mesh.init_uniform(Nx, Ny, 0.0, 1.0, 0.0, 1.0);

    VectorField vel(mesh);
    const int Ng = mesh.Nghost;

    for (int j = 0; j < Ny + 1 + 2 * Ng; ++j) {
        for (int i = 0; i < Nx + 2 * Ng; ++i) {
            vel.v(i, j) = 3.0 * mesh.yf[j];
        }
    }

    VelocityGradient grad_comp;
    VelocityGradientTensor grad;
    grad_comp.compute(mesh, vel, grad);

    double max_err = 0.0;
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            int idx = grad.index(i, j, 0);
            max_err = std::max(max_err, std::abs(grad.g22[idx] - 3.0));
        }
    }

    CHECK(max_err < 1e-12, "dv/dy must be exact for linear v field");
    std::cout << "PASS: dv/dy linear (max_err=" << max_err << ")" << std::endl;
}

void test_pure_shear() {
    // u = S*y, v = 0 → du/dy = S, all other gradients ≈ 0
    const double S = 5.0;
    const int Nx = 16, Ny = 16;
    Mesh mesh;
    mesh.init_uniform(Nx, Ny, 0.0, 1.0, 0.0, 1.0);

    VectorField vel(mesh);
    const int Ng = mesh.Nghost;

    // u = S*y at x-faces (u lives at (xf[i], y[j]))
    for (int j = 0; j < Ny + 2 * Ng; ++j) {
        double y = mesh.y(j);
        for (int i = 0; i < Nx + 1 + 2 * Ng; ++i) {
            vel.u(i, j) = S * y;
        }
    }

    VelocityGradient grad_comp;
    VelocityGradientTensor grad;
    grad_comp.compute(mesh, vel, grad);

    double max_err_dudy = 0.0;
    double max_err_dudx = 0.0;
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            int idx = grad.index(i, j, 0);
            max_err_dudy = std::max(max_err_dudy, std::abs(grad.g12[idx] - S));
            max_err_dudx = std::max(max_err_dudx, std::abs(grad.g11[idx]));
        }
    }

    CHECK(max_err_dudy < 1e-12, "du/dy must be exact for u = S*y");
    CHECK(max_err_dudx < 1e-12, "du/dx must be zero for u = S*y");
    std::cout << "PASS: Pure shear (dudy_err=" << max_err_dudy
              << ", dudx_err=" << max_err_dudx << ")" << std::endl;
}

void test_3d_dwdz() {
    // w = 4*z → dw/dz = 4
    const int Nx = 8, Ny = 8, Nz = 8;
    Mesh mesh;
    mesh.init_uniform(Nx, Ny, Nz, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    VectorField vel(mesh);
    const int Ng = mesh.Nghost;

    // w = 4*z at z-faces
    for (int k = 0; k < Nz + 1 + 2 * Ng; ++k) {
        for (int j = 0; j < Ny + 2 * Ng; ++j) {
            for (int i = 0; i < Nx + 2 * Ng; ++i) {
                vel.w(i, j, k) = 4.0 * mesh.zf[k];
            }
        }
    }

    VelocityGradient grad_comp;
    VelocityGradientTensor grad;
    grad_comp.compute(mesh, vel, grad);

    double max_err = 0.0;
    for (int k = 0; k < Nz; ++k) {
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                int idx = grad.index(i, j, k);
                max_err = std::max(max_err, std::abs(grad.g33[idx] - 4.0));
            }
        }
    }

    CHECK(max_err < 1e-12, "dw/dz must be exact for linear w field");
    std::cout << "PASS: 3D dw/dz linear (max_err=" << max_err << ")" << std::endl;
}

void test_divergence_free() {
    // u = sin(2πx)cos(2πy), v = -cos(2πx)sin(2πy) → div = 0
    // du/dx + dv/dy should be near zero
    const int Nx = 32, Ny = 32;
    Mesh mesh;
    mesh.init_uniform(Nx, Ny, 0.0, 1.0, 0.0, 1.0);

    VectorField vel(mesh);
    const int Ng = mesh.Nghost;
    const double pi2 = 2.0 * M_PI;

    for (int j = 0; j < Ny + 2 * Ng; ++j) {
        double y = mesh.y(j);
        for (int i = 0; i < Nx + 1 + 2 * Ng; ++i) {
            double x = mesh.xf[i];
            vel.u(i, j) = std::sin(pi2 * x) * std::cos(pi2 * y);
        }
    }
    for (int j = 0; j < Ny + 1 + 2 * Ng; ++j) {
        double y = mesh.yf[j];
        for (int i = 0; i < Nx + 2 * Ng; ++i) {
            double x = mesh.x(i);
            vel.v(i, j) = -std::cos(pi2 * x) * std::sin(pi2 * y);
        }
    }

    VelocityGradient grad_comp;
    VelocityGradientTensor grad;
    grad_comp.compute(mesh, vel, grad);

    double max_div = 0.0;
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            int idx = grad.index(i, j, 0);
            double div = grad.g11[idx] + grad.g22[idx];
            max_div = std::max(max_div, std::abs(div));
        }
    }

    // 2nd-order FD on sinusoidal field: error ~ (pi*dx)^2
    CHECK(max_div < 0.1, "Divergence should be small for div-free field");
    std::cout << "PASS: Divergence-free check (max_div=" << max_div << ")" << std::endl;
}

int main() {
    test_dudx_linear();
    test_dvdy_linear();
    test_pure_shear();
    test_3d_dwdz();
    test_divergence_free();

    std::cout << "\nAll velocity gradient tests PASSED" << std::endl;
    return 0;
}
