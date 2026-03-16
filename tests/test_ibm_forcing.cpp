/// @file test_ibm_forcing.cpp
/// @brief Tests for IBM direct-forcing module
///
/// Test coverage:
///   1. Cell classification: cylinder in 2D, correct Fluid/Forcing/Solid counts
///   2. Direct forcing: velocity inside body goes to zero
///   3. Velocity outside body preserved
///   4. Force computation: nonzero drag for flow past cylinder

#include "ibm_forcing.hpp"
#include "ibm_geometry.hpp"
#include "mesh.hpp"
#include "fields.hpp"
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>

using namespace nncfd;

#define CHECK(cond, msg) do { if (!(cond)) throw std::runtime_error(msg); } while(0)

void test_cell_classification() {
    const int Nx = 32, Ny = 32;
    Mesh mesh;
    mesh.init_uniform(Nx, Ny, 0.0, 4.0, -2.0, 2.0);

    auto body = std::make_shared<CylinderBody>(2.0, 0.0, 0.5);
    IBMForcing ibm(mesh, body);

    CHECK(ibm.num_forcing_cells() > 0, "Should have forcing cells");
    CHECK(ibm.num_solid_cells() > 0, "Should have solid cells");

    const int Ng = mesh.Nghost;
    int i_center = Ng + (int)(2.0 / mesh.dx);
    int j_center = Ng + (int)((0.0 + 2.0) / mesh.dy);

    CHECK(ibm.cell_type_u(i_center, j_center) == IBMCellType::Solid,
          "Center of cylinder should be Solid");

    CHECK(ibm.cell_type_u(Ng, j_center) == IBMCellType::Fluid,
          "Far from cylinder should be Fluid");

    std::cout << "PASS: Cell classification (" << ibm.num_forcing_cells()
              << " forcing, " << ibm.num_solid_cells() << " solid)" << std::endl;
}

void test_forcing_zeroes_velocity() {
    const int Nx = 32, Ny = 32;
    Mesh mesh;
    mesh.init_uniform(Nx, Ny, 0.0, 4.0, -2.0, 2.0);

    auto body = std::make_shared<CylinderBody>(2.0, 0.0, 0.5);
    IBMForcing ibm(mesh, body);

    VectorField vel(mesh);
    const int Ng = mesh.Nghost;
    for (int j = 0; j < Ny + 2*Ng; ++j) {
        for (int i = 0; i < Nx + 1 + 2*Ng; ++i) {
            vel.u(i, j) = 1.0;
        }
    }
    for (int j = 0; j < Ny + 1 + 2*Ng; ++j) {
        for (int i = 0; i < Nx + 2*Ng; ++i) {
            vel.v(i, j) = 0.0;
        }
    }

    ibm.apply_forcing(vel, 0.001);

    int i_center = Ng + (int)(2.0 / mesh.dx);
    int j_center = Ng + (int)((0.0 + 2.0) / mesh.dy);
    CHECK(std::abs(vel.u(i_center, j_center)) < 1e-10,
          "Velocity inside body should be zero");

    CHECK(std::abs(vel.u(Ng, j_center) - 1.0) < 1e-10,
          "Velocity far from body should be unchanged");

    std::cout << "PASS: Forcing zeroes velocity inside body" << std::endl;
}

void test_forcing_preserves_exterior() {
    const int Nx = 32, Ny = 32;
    Mesh mesh;
    mesh.init_uniform(Nx, Ny, 0.0, 4.0, -2.0, 2.0);

    auto body = std::make_shared<CylinderBody>(2.0, 0.0, 0.3);
    IBMForcing ibm(mesh, body);

    VectorField vel(mesh);
    const int Ng = mesh.Nghost;
    for (int j = 0; j < Ny + 2*Ng; ++j) {
        for (int i = 0; i < Nx + 1 + 2*Ng; ++i) {
            vel.u(i, j) = 1.0;
        }
    }
    for (int j = 0; j < Ny + 1 + 2*Ng; ++j) {
        for (int i = 0; i < Nx + 2*Ng; ++i) {
            vel.v(i, j) = 0.5;
        }
    }

    ibm.apply_forcing(vel, 0.001);

    int n_unchanged = 0;
    int n_total = 0;
    for (int j = Ng; j < Ny + Ng; ++j) {
        for (int i = Ng; i < Nx + Ng; ++i) {
            double x = mesh.xf[i];
            double y = mesh.y(j);
            if (body->phi(x, y, 0.0) > 0.0) {
                n_total++;
                if (std::abs(vel.u(i, j) - 1.0) < 1e-14) {
                    n_unchanged++;
                }
            }
        }
    }

    double ratio = static_cast<double>(n_unchanged) / n_total;
    CHECK(ratio > 0.9, "Most exterior cells should be unchanged");

    std::cout << "PASS: Forcing preserves exterior (" << n_unchanged
              << "/" << n_total << " unchanged)" << std::endl;
}

void test_force_computation() {
    const int Nx = 32, Ny = 32;
    Mesh mesh;
    mesh.init_uniform(Nx, Ny, 0.0, 4.0, -2.0, 2.0);

    auto body = std::make_shared<CylinderBody>(2.0, 0.0, 0.5);
    IBMForcing ibm(mesh, body);

    VectorField vel(mesh);
    const int Ng = mesh.Nghost;
    for (int j = 0; j < Ny + 2*Ng; ++j) {
        for (int i = 0; i < Nx + 1 + 2*Ng; ++i) {
            vel.u(i, j) = 1.0;
        }
    }

    // reset_force_accumulator() then apply_forcing(dt>0) accumulates forces.
    // compute_forces returns the cached values.
    const double dt = 0.001;
    ibm.set_accumulate_forces(true);
    ibm.reset_force_accumulator();
    ibm.apply_forcing(vel, dt);
    auto [Fx, Fy, Fz] = ibm.compute_forces(vel, dt);

    CHECK(std::abs(Fx) > 0.0, "Force must be nonzero for flow through body");

    // For symmetric flow past cylinder, lift should be ~0
    CHECK(std::abs(Fy) < std::abs(Fx) * 0.1,
          "Lift should be small for symmetric flow");

    std::cout << "PASS: Force computation (Fx=" << Fx << ", Fy=" << Fy << ")" << std::endl;
}

int main() {
    test_cell_classification();
    test_forcing_zeroes_velocity();
    test_forcing_preserves_exterior();
    test_force_computation();

    std::cout << "\nAll IBM forcing tests PASSED" << std::endl;
    return 0;
}
