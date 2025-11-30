/// Unit tests for Mesh class

#include "mesh.hpp"
#include "fields.hpp"
#include <iostream>
#include <cmath>
#include <cassert>

using namespace nncfd;

void test_uniform_mesh() {
    std::cout << "Testing uniform mesh... ";
    
    Mesh mesh;
    mesh.init_uniform(10, 20, 0.0, 1.0, -1.0, 1.0);
    
    assert(mesh.Nx == 10);
    assert(mesh.Ny == 20);
    assert(std::abs(mesh.dx - 0.1) < 1e-10);
    assert(std::abs(mesh.dy - 0.1) < 1e-10);
    
    // Check interior cell count
    assert(mesh.total_Nx() == 12);  // 10 + 2 ghost
    assert(mesh.total_Ny() == 22);  // 20 + 2 ghost
    
    // Check indexing
    int idx = mesh.index(5, 10);
    int i, j;
    mesh.inv_index(idx, i, j);
    assert(i == 5 && j == 10);
    
    // Check interior detection
    assert(mesh.isInterior(5, 10));
    assert(!mesh.isInterior(0, 10));
    assert(!mesh.isInterior(11, 10));
    
    // Check cell centers
    assert(std::abs(mesh.x(1) - 0.05) < 1e-10);  // First interior cell at ghost index 1
    
    std::cout << "PASSED\n";
}

void test_stretched_mesh() {
    std::cout << "Testing stretched mesh... ";
    
    Mesh mesh;
    mesh.init_stretched_y(10, 20, 0.0, 1.0, -1.0, 1.0, 
                          Mesh::tanh_stretching(2.0));
    
    assert(mesh.Nx == 10);
    assert(mesh.Ny == 20);
    
    // Stretched mesh should have smaller dy near walls
    double dy_wall = mesh.dyv[mesh.Nghost];  // First interior cell
    double dy_center = mesh.dyv[mesh.Nghost + mesh.Ny/2];  // Center
    
    assert(dy_wall < dy_center);  // Finer at wall
    
    std::cout << "PASSED\n";
}

void test_wall_distance() {
    std::cout << "Testing wall distance... ";
    
    Mesh mesh;
    mesh.init_uniform(10, 20, 0.0, 1.0, -1.0, 1.0);
    
    // Cell at bottom wall should have small wall distance
    double dist_bottom = mesh.wall_distance(5, mesh.j_begin());
    double dist_top = mesh.wall_distance(5, mesh.j_end() - 1);
    double dist_center = mesh.wall_distance(5, mesh.j_begin() + mesh.Ny/2);
    
    assert(dist_bottom < 0.1);
    assert(dist_top < 0.1);
    assert(dist_center > 0.9);  // Near centerline
    
    std::cout << "PASSED\n";
}

void test_scalar_field() {
    std::cout << "Testing scalar field... ";
    
    Mesh mesh;
    mesh.init_uniform(10, 10, 0.0, 1.0, 0.0, 1.0);
    
    ScalarField f(mesh, 1.0);
    
    // Check initial value
    assert(std::abs(f(5, 5) - 1.0) < 1e-10);
    
    // Modify and check
    f(5, 5) = 2.0;
    assert(std::abs(f(5, 5) - 2.0) < 1e-10);
    
    // Check fill
    f.fill(3.0);
    assert(std::abs(f(3, 3) - 3.0) < 1e-10);
    
    std::cout << "PASSED\n";
}

void test_vector_field() {
    std::cout << "Testing vector field... ";
    
    Mesh mesh;
    mesh.init_uniform(10, 10, 0.0, 1.0, 0.0, 1.0);
    
    VectorField v(mesh, 1.0, 2.0);
    
    assert(std::abs(v.u(5, 5) - 1.0) < 1e-10);
    assert(std::abs(v.v(5, 5) - 2.0) < 1e-10);
    
    // Check magnitude
    double mag = v.magnitude(5, 5);
    assert(std::abs(mag - std::sqrt(5.0)) < 1e-10);
    
    std::cout << "PASSED\n";
}

int main() {
    std::cout << "=== Mesh and Fields Tests ===\n\n";
    
    test_uniform_mesh();
    test_stretched_mesh();
    test_wall_distance();
    test_scalar_field();
    test_vector_field();
    
    std::cout << "\nAll tests PASSED!\n";
    return 0;
}


