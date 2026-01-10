/// Unit tests for Mesh class

#include "mesh.hpp"
#include "fields.hpp"
#include "test_harness.hpp"
#include <iostream>
#include <cmath>
#include <cassert>

using namespace nncfd;
using nncfd::test::harness::record;

void test_uniform_mesh() {
    Mesh mesh;
    mesh.init_uniform(10, 20, 0.0, 1.0, -1.0, 1.0);

    bool pass = (mesh.Nx == 10) && (mesh.Ny == 20);
    pass = pass && (std::abs(mesh.dx - 0.1) < 1e-10);
    pass = pass && (std::abs(mesh.dy - 0.1) < 1e-10);
    pass = pass && (mesh.total_Nx() == 12);  // 10 + 2 ghost
    pass = pass && (mesh.total_Ny() == 22);  // 20 + 2 ghost

    // Check indexing
    int idx = mesh.index(5, 10);
    int i, j;
    mesh.inv_index(idx, i, j);
    pass = pass && (i == 5 && j == 10);

    // Check interior detection
    pass = pass && mesh.isInterior(5, 10);
    pass = pass && !mesh.isInterior(0, 10);
    pass = pass && !mesh.isInterior(11, 10);

    // Check cell centers
    pass = pass && (std::abs(mesh.x(1) - 0.05) < 1e-10);

    record("Uniform mesh", pass);
}

void test_stretched_mesh() {
    Mesh mesh;
    mesh.init_stretched_y(10, 20, 0.0, 1.0, -1.0, 1.0,
                          Mesh::tanh_stretching(2.0));

    bool pass = (mesh.Nx == 10) && (mesh.Ny == 20);

    // Stretched mesh should have smaller dy near walls
    double dy_wall = mesh.dyv[mesh.Nghost];  // First interior cell
    double dy_center = mesh.dyv[mesh.Nghost + mesh.Ny/2];  // Center
    pass = pass && (dy_wall < dy_center);  // Finer at wall

    record("Stretched mesh", pass);
}

void test_wall_distance() {
    Mesh mesh;
    mesh.init_uniform(10, 20, 0.0, 1.0, -1.0, 1.0);

    double dist_bottom = mesh.wall_distance(5, mesh.j_begin());
    double dist_top = mesh.wall_distance(5, mesh.j_end() - 1);
    double dist_center = mesh.wall_distance(5, mesh.j_begin() + mesh.Ny/2);

    bool pass = (dist_bottom < 0.1) && (dist_top < 0.1) && (dist_center > 0.9);
    record("Wall distance", pass);
}

void test_scalar_field() {
    Mesh mesh;
    mesh.init_uniform(10, 10, 0.0, 1.0, 0.0, 1.0);

    ScalarField f(mesh, 1.0);
    bool pass = std::abs(f(5, 5) - 1.0) < 1e-10;

    f(5, 5) = 2.0;
    pass = pass && (std::abs(f(5, 5) - 2.0) < 1e-10);

    f.fill(3.0);
    pass = pass && (std::abs(f(3, 3) - 3.0) < 1e-10);

    record("Scalar field", pass);
}

void test_vector_field() {
    Mesh mesh;
    mesh.init_uniform(10, 10, 0.0, 1.0, 0.0, 1.0);

    VectorField v(mesh, 1.0, 2.0);
    bool pass = (std::abs(v.u(5, 5) - 1.0) < 1e-10);
    pass = pass && (std::abs(v.v(5, 5) - 2.0) < 1e-10);

    double mag = v.magnitude(5, 5);
    pass = pass && (std::abs(mag - std::sqrt(5.0)) < 1e-10);

    record("Vector field", pass);
}

int main() {
    return nncfd::test::harness::run("Mesh and Fields Tests", [] {
        test_uniform_mesh();
        test_stretched_mesh();
        test_wall_distance();
        test_scalar_field();
        test_vector_field();
    });
}


