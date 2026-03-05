/// @file test_checkpoint.cpp
/// @brief Tests for HDF5 checkpoint/restart
///
/// Test coverage:
///   1. Write and read checkpoint: verify round-trip preserves data
///   2. Mesh dimension mismatch detection

#include "checkpoint.hpp"
#include "mesh.hpp"
#include "fields.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <cstdio>

using namespace nncfd;

#define CHECK(cond, msg) do { if (!(cond)) throw std::runtime_error(msg); } while(0)

#ifdef USE_HDF5

void test_roundtrip() {
    const int Nx = 8, Ny = 8;
    Mesh mesh;
    mesh.init_uniform(Nx, Ny, 0.0, 1.0, 0.0, 1.0);

    VectorField vel(mesh);
    ScalarField pressure(mesh);
    const int Ng = mesh.Nghost;

    // Fill with known data
    for (int j = 0; j < Ny + 2 * Ng; ++j) {
        for (int i = 0; i < Nx + 1 + 2 * Ng; ++i) {
            vel.u(i, j) = std::sin(i * 0.1) * std::cos(j * 0.2);
        }
    }
    for (int j = 0; j < Ny + 1 + 2 * Ng; ++j) {
        for (int i = 0; i < Nx + 2 * Ng; ++i) {
            vel.v(i, j) = std::cos(i * 0.3) * std::sin(j * 0.4);
        }
    }
    for (int j = 0; j < Ny + 2 * Ng; ++j) {
        for (int i = 0; i < Nx + 2 * Ng; ++i) {
            pressure(i, j) = i * 0.1 + j * 0.2;
        }
    }

    int step = 42;
    double time = 3.14;
    double dt = 0.001;

    std::string filename = "/tmp/test_checkpoint.h5";

    // Write
    write_checkpoint(filename, mesh, vel, pressure, step, time, dt);

    // Read into fresh fields
    VectorField vel2(mesh);
    ScalarField pressure2(mesh);
    int step2 = 0;
    double time2 = 0.0, dt2 = 0.0;

    bool ok = read_checkpoint(filename, mesh, vel2, pressure2, step2, time2, dt2);
    CHECK(ok, "read_checkpoint should succeed");
    CHECK(step2 == step, "Step mismatch");
    CHECK(std::abs(time2 - time) < 1e-15, "Time mismatch");
    CHECK(std::abs(dt2 - dt) < 1e-15, "dt mismatch");

    // Verify velocity data matches
    double max_err_u = 0.0;
    for (int j = 0; j < Ny + 2 * Ng; ++j) {
        for (int i = 0; i < Nx + 1 + 2 * Ng; ++i) {
            max_err_u = std::max(max_err_u, std::abs(vel.u(i, j) - vel2.u(i, j)));
        }
    }
    CHECK(max_err_u < 1e-15, "Velocity u round-trip error too large");

    double max_err_v = 0.0;
    for (int j = 0; j < Ny + 1 + 2 * Ng; ++j) {
        for (int i = 0; i < Nx + 2 * Ng; ++i) {
            max_err_v = std::max(max_err_v, std::abs(vel.v(i, j) - vel2.v(i, j)));
        }
    }
    CHECK(max_err_v < 1e-15, "Velocity v round-trip error too large");

    double max_err_p = 0.0;
    for (int j = 0; j < Ny + 2 * Ng; ++j) {
        for (int i = 0; i < Nx + 2 * Ng; ++i) {
            max_err_p = std::max(max_err_p, std::abs(pressure(i, j) - pressure2(i, j)));
        }
    }
    CHECK(max_err_p < 1e-15, "Pressure round-trip error too large");

    // Cleanup
    std::remove(filename.c_str());

    std::cout << "PASS: Checkpoint round-trip (u_err=" << max_err_u
              << ", v_err=" << max_err_v << ", p_err=" << max_err_p << ")" << std::endl;
}

void test_mesh_mismatch() {
    const int Nx = 8, Ny = 8;
    Mesh mesh;
    mesh.init_uniform(Nx, Ny, 0.0, 1.0, 0.0, 1.0);

    VectorField vel(mesh);
    ScalarField pressure(mesh);
    std::string filename = "/tmp/test_checkpoint_mismatch.h5";

    write_checkpoint(filename, mesh, vel, pressure, 0, 0.0, 0.001);

    // Try to load with different mesh
    Mesh mesh2;
    mesh2.init_uniform(16, 16, 0.0, 1.0, 0.0, 1.0);
    VectorField vel2(mesh2);
    ScalarField pressure2(mesh2);
    int step;
    double time, dt;

    bool ok = read_checkpoint(filename, mesh2, vel2, pressure2, step, time, dt);
    CHECK(!ok, "Should fail for mesh dimension mismatch");

    std::remove(filename.c_str());
    std::cout << "PASS: Mesh mismatch detection" << std::endl;
}

#endif // USE_HDF5

int main() {
#ifdef USE_HDF5
    test_roundtrip();
    test_mesh_mismatch();
    std::cout << "\nAll checkpoint tests PASSED" << std::endl;
#else
    std::cout << "SKIP: HDF5 not available, checkpoint tests skipped" << std::endl;
#endif
    return 0;
}
