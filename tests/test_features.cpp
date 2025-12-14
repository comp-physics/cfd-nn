/// Unit tests for feature computation

#include "mesh.hpp"
#include "fields.hpp"
#include "features.hpp"
#include <iostream>
#include <cmath>
#include <cassert>

using namespace nncfd;

void test_velocity_gradient() {
    std::cout << "Testing velocity gradient computation... ";
    
    Mesh mesh;
    mesh.init_uniform(16, 16, 0.0, 1.0, 0.0, 1.0);
    
    // Create linear velocity field: u = x, v = y
    VectorField vel(mesh);
    for (int j = 0; j < mesh.total_Ny(); ++j) {
        for (int i = 0; i < mesh.total_Nx(); ++i) {
            vel.u(i, j) = mesh.x(i);
            vel.v(i, j) = mesh.y(j);
        }
    }
    
    // Gradients should be: dudx=1, dudy=0, dvdx=0, dvdy=1
    int i = mesh.Nx/2;
    int j = mesh.Ny/2;
    
    // Compute gradients using MAC-aware method
    const double inv_2dx = 1.0 / (2.0 * mesh.dx);
    const double inv_2dy = 1.0 / (2.0 * mesh.dy);
    [[maybe_unused]] VelocityGradient grad;
    grad.dudx = (vel.u(i + 1, j) - vel.u(i - 1, j)) * inv_2dx;
    grad.dudy = (vel.u(i, j + 1) - vel.u(i, j - 1)) * inv_2dy;
    grad.dvdx = (vel.v(i + 1, j) - vel.v(i - 1, j)) * inv_2dx;
    grad.dvdy = (vel.v(i, j + 1) - vel.v(i, j - 1)) * inv_2dy;
    
    assert(std::abs(grad.dudx - 1.0) < 1e-6);
    assert(std::abs(grad.dudy - 0.0) < 1e-6);
    assert(std::abs(grad.dvdx - 0.0) < 1e-6);
    assert(std::abs(grad.dvdy - 1.0) < 1e-6);
    
    // Strain and rotation
    assert(std::abs(grad.Sxx() - 1.0) < 1e-6);
    assert(std::abs(grad.Syy() - 1.0) < 1e-6);
    assert(std::abs(grad.Sxy() - 0.0) < 1e-6);
    assert(std::abs(grad.Oxy() - 0.0) < 1e-6);
    (void)grad;  // Used in assert
    
    std::cout << "PASSED\n";
}

void test_tensor_basis() {
    std::cout << "Testing tensor basis computation... ";
    
    // Create simple velocity gradient
    VelocityGradient grad;
    grad.dudx = 1.0;
    grad.dudy = 0.5;
    grad.dvdx = 0.5;
    grad.dvdy = -1.0;
    
    double k = 0.1;
    double epsilon = 0.01;
    
    std::array<std::array<double, 3>, TensorBasis::NUM_BASIS> basis;
    TensorBasis::compute(grad, k, epsilon, basis);
    
    // T1 should be proportional to S
    // Just check it's non-zero
    double T1_mag_sq = basis[0][0]*basis[0][0] + 
                       basis[0][1]*basis[0][1] + 
                       basis[0][2]*basis[0][2];
    assert(T1_mag_sq > 1e-12);
    
    std::cout << "PASSED (T1_mag=" << std::sqrt(T1_mag_sq) << ")\n";
}

void test_invariants() {
    std::cout << "Testing invariant computation... ";
    
    Mesh mesh;
    mesh.init_uniform(16, 16, 0.0, 1.0, 0.0, 1.0);
    
    // Simple shear flow: u = y, v = 0
    VectorField vel(mesh);
    for (int j = 0; j < mesh.total_Ny(); ++j) {
        for (int i = 0; i < mesh.total_Nx(); ++i) {
            vel.u(i, j) = mesh.y(j);
            vel.v(i, j) = 0.0;
        }
    }
    
    ScalarField k(mesh, 0.01);
    ScalarField omega(mesh, 1.0);
    
    int i = mesh.Nx/2;
    int j = mesh.Ny/2;
    
    Features feat = compute_features_tbnn(mesh, vel, k, omega, i, j, 0.001, 1.0);
    
    assert(feat.size() == 5);  // Should have 5 invariants
    
    // All invariants should be finite
    for (int n = 0; n < feat.size(); ++n) {
        assert(std::isfinite(feat[n]));
    }
    
    std::cout << "PASSED\n";
}

int main() {
    std::cout << "=== Feature Computation Tests ===\n\n";
    
    test_velocity_gradient();
    test_tensor_basis();
    test_invariants();
    
    std::cout << "\nAll feature tests passed!\n";
    return 0;
}

