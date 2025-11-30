/// Unit tests for Poisson solver

#include "mesh.hpp"
#include "fields.hpp"
#include "poisson_solver.hpp"
#include <iostream>
#include <cmath>
#include <cassert>

using namespace nncfd;

void test_laplacian() {
    std::cout << "Testing Laplacian... ";
    
    Mesh mesh;
    mesh.init_uniform(20, 20, 0.0, 1.0, 0.0, 1.0);
    
    // Create a quadratic field p = x^2 + y^2
    // Laplacian should be 4
    ScalarField p(mesh);
    
    for (int j = 0; j < mesh.total_Ny(); ++j) {
        for (int i = 0; i < mesh.total_Nx(); ++i) {
            double x = mesh.x(i);
            double y = mesh.y(j);
            p(i, j) = x * x + y * y;
        }
    }
    
    // Check Laplacian at interior points
    double dx2 = mesh.dx * mesh.dx;
    double dy2 = mesh.dy * mesh.dy;
    
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double laplacian = (p(i+1, j) - 2*p(i, j) + p(i-1, j)) / dx2
                             + (p(i, j+1) - 2*p(i, j) + p(i, j-1)) / dy2;
            
            // Should be 4 for p = x^2 + y^2
            assert(std::abs(laplacian - 4.0) < 0.01);
            (void)laplacian;  // Used in assert
        }
    }
    
    std::cout << "PASSED\n";
}

void test_poisson_constant_rhs() {
    std::cout << "Testing Poisson with constant RHS... ";
    
    Mesh mesh;
    mesh.init_uniform(32, 32, 0.0, 1.0, 0.0, 1.0);
    
    // Solve nabla^2p = 1 with Dirichlet BC p = 0
    ScalarField rhs(mesh, 1.0);
    ScalarField p(mesh, 0.0);
    
    PoissonSolver solver(mesh);
    solver.set_bc(PoissonBC::Dirichlet, PoissonBC::Dirichlet,
                  PoissonBC::Dirichlet, PoissonBC::Dirichlet);
    solver.set_dirichlet_value(0.0);
    
    PoissonConfig cfg;
    cfg.tol = 1e-8;
    cfg.max_iter = 10000;
    cfg.omega = 1.8;
    
    int iters = solver.solve(rhs, p, cfg);
    
    std::cout << "(iters=" << iters << ", res=" << solver.residual() << ") ";
    
    // Check that solution is reasonable (positive in interior)
    bool positive_interior = true;
    for (int j = mesh.j_begin() + 1; j < mesh.j_end() - 1; ++j) {
        for (int i = mesh.i_begin() + 1; i < mesh.i_end() - 1; ++i) {
            if (p(i, j) < 0) {
                positive_interior = false;
            }
        }
    }
    
    assert(positive_interior);
    (void)positive_interior;  // Used in assert
    assert(solver.residual() < 1e-6);
    
    std::cout << "PASSED\n";
}

void test_poisson_periodic() {
    std::cout << "Testing Poisson with periodic BC... ";
    
    Mesh mesh;
    int N = 32;
    double L = 2.0 * M_PI;
    mesh.init_uniform(N, N, 0.0, L, 0.0, L);
    
    // Solve nabla^2p = -sin(x) * sin(y)
    // Exact solution: p = sin(x) * sin(y) / 2
    ScalarField rhs(mesh);
    ScalarField p(mesh, 0.0);
    
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double x = mesh.x(i);
            double y = mesh.y(j);
            rhs(i, j) = -2.0 * std::sin(x) * std::sin(y);  // Laplacian of sin(x)*sin(y)
        }
    }
    
    PoissonSolver solver(mesh);
    solver.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                  PoissonBC::Periodic, PoissonBC::Periodic);
    
    PoissonConfig cfg;
    cfg.tol = 1e-8;
    cfg.max_iter = 10000;
    cfg.omega = 1.7;
    
    int iters = solver.solve(rhs, p, cfg);
    
    std::cout << "(iters=" << iters << ", res=" << solver.residual() << ") ";
    
    // Check against exact solution (up to constant)
    // Subtract mean from both numerical and exact
    double p_mean = 0.0;
    double p_exact_mean = 0.0;
    int count = 0;
    
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double x = mesh.x(i);
            double y = mesh.y(j);
            p_mean += p(i, j);
            p_exact_mean += std::sin(x) * std::sin(y);
            ++count;
        }
    }
    p_mean /= count;
    p_exact_mean /= count;
    
    double max_error = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double x = mesh.x(i);
            double y = mesh.y(j);
            double p_exact = std::sin(x) * std::sin(y);
            double error = std::abs((p(i, j) - p_mean) - (p_exact - p_exact_mean));
            max_error = std::max(max_error, error);
        }
    }
    
    std::cout << "(max_err=" << max_error << ") ";
    
    assert(max_error < 0.1);  // Allow some discretization error
    
    std::cout << "PASSED\n";
}

void test_poisson_channel_bc() {
    std::cout << "Testing Poisson with channel-like BC (periodic x, Neumann y)... ";
    
    Mesh mesh;
    mesh.init_uniform(32, 32, 0.0, 2*M_PI, -1.0, 1.0);
    
    // Uniform RHS (like divergence-free correction)
    ScalarField rhs(mesh, 0.0);
    
    // Small perturbation
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            rhs(i, j) = 0.1 * std::sin(mesh.x(i));
        }
    }
    
    ScalarField p(mesh, 0.0);
    
    PoissonSolver solver(mesh);
    solver.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                  PoissonBC::Neumann, PoissonBC::Neumann);
    
    PoissonConfig cfg;
    cfg.tol = 1e-8;
    cfg.max_iter = 5000;
    cfg.omega = 1.7;
    
    int iters = solver.solve(rhs, p, cfg);
    
    std::cout << "(iters=" << iters << ", res=" << solver.residual() << ") ";
    
    assert(solver.residual() < 1e-6);
    
    std::cout << "PASSED\n";
}

int main() {
    std::cout << "=== Poisson Solver Tests ===\n\n";
    
    test_laplacian();
    test_poisson_constant_rhs();
    test_poisson_periodic();
    test_poisson_channel_bc();
    
    std::cout << "\nAll tests PASSED!\n";
    return 0;
}


