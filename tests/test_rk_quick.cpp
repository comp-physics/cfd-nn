// Quick RK2/RK3 verification test
#include "solver.hpp"
#include "config.hpp"
#include "mesh.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>

using namespace nncfd;

void test_integrator(TimeIntegrator integrator, const char* name) {
    Mesh mesh;
    mesh.init_uniform(32, 32, 0.0, 2*M_PI, 0.0, 2*M_PI);
    
    Config config;
    config.nu = 0.01;
    config.dt = 0.01;
    config.adaptive_dt = false;
    config.time_integrator = integrator;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;
    
    RANSSolver solver(mesh, config);
    
    // TGV initial condition
    auto& vel = solver.velocity();
    for (int j = 0; j <= mesh.Ny; ++j) {
        double y = mesh.y(j);
        for (int i = 0; i <= mesh.Nx + 1; ++i) {
            double x = (i <= mesh.Nx) ? mesh.x(i) : mesh.x(mesh.Nx) + mesh.dx;
            vel.u(i, j) = std::cos(x) * std::sin(y);
        }
    }
    for (int j = 0; j <= mesh.Ny + 1; ++j) {
        double y = (j <= mesh.Ny) ? mesh.y(j) : mesh.y(mesh.Ny) + mesh.dy;
        for (int i = 0; i <= mesh.Nx; ++i) {
            double x = mesh.x(i);
            vel.v(i, j) = -std::sin(x) * std::cos(y);
        }
    }
    
    // Compute initial KE
    double ke_init = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double u = 0.5 * (vel.u(i,j) + vel.u(i+1,j));
            double v = 0.5 * (vel.v(i,j) + vel.v(i,j+1));
            ke_init += 0.5 * (u*u + v*v);
        }
    }
    ke_init *= mesh.dx * mesh.dy;
    
    solver.sync_to_gpu();
    
    // Step 50 times
    for (int i = 0; i < 50; ++i) {
        solver.step();
    }
    
    solver.sync_from_gpu();
    
    // Compute final KE
    double ke_final = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double u = 0.5 * (vel.u(i,j) + vel.u(i+1,j));
            double v = 0.5 * (vel.v(i,j) + vel.v(i,j+1));
            ke_final += 0.5 * (u*u + v*v);
        }
    }
    ke_final *= mesh.dx * mesh.dy;
    
    double ke_ratio = ke_final / ke_init;
    bool decaying = ke_ratio < 1.0 && ke_ratio > 0.5;  // Should decay but not explode
    
    std::cout << std::fixed << std::setprecision(4);
    std::cout << name << ": KE_init=" << ke_init << " KE_final=" << ke_final 
              << " ratio=" << ke_ratio << " => " << (decaying ? "OK" : "FAIL") << "\n";
}

int main() {
    std::cout << "=== Quick RK Verification (50 steps of 2D TGV) ===\n";
    test_integrator(TimeIntegrator::Euler, "Euler");
    test_integrator(TimeIntegrator::RK2, "RK2  ");
    test_integrator(TimeIntegrator::RK3, "RK3  ");
    return 0;
}
