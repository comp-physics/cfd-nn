/// Test Poisson solver with x-dependent perturbed channel flow
/// This creates non-zero divergence to validate Poisson residual monitoring

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "config.hpp"
#include "timing.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>

using namespace nncfd;

int main(int argc, char** argv) {
    std::cout << "=== Perturbed Channel Flow - Poisson Solver Test ===\n\n";
    
    // Parse configuration
    Config config;
    config.parse_args(argc, argv);
    config.print();
    
    // Create mesh
    Mesh mesh;
    mesh.init_uniform(config.Nx, config.Ny,
                      config.x_min, config.x_max,
                      config.y_min, config.y_max);
    
    std::cout << "Mesh: " << mesh.Nx << " x " << mesh.Ny << " cells\n";
    std::cout << "dx = " << mesh.dx << ", dy = " << mesh.dy << "\n\n";
    
    // Create solver
    RANSSolver solver(mesh, config);
    
    // Set boundary conditions
    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);
    
    // Set body force
    solver.set_body_force(-config.dp_dx, 0.0);
    
    // Initialize with 2D perturbation: u = u0 * (1 + 0.1*sin(2*x)*cos(pi*y))
    // This creates x-dependent flow → non-zero divergence initially
    VectorField vel_init(mesh);
    const double H = (config.y_max - config.y_min) / 2.0;
    const double u0 = -config.dp_dx * H * H / (2.0 * config.nu) * 0.1;  // 10% of Poiseuille max
    const int Ng = mesh.Nghost;
    
    std::cout << "Initializing with 2D perturbation: u = u0*(1 + 0.1*sin(2*x)*cos(pi*y))\n";
    std::cout << "Base velocity u0 = " << u0 << "\n\n";
    
    // Initialize u at x-faces
    for (int j = Ng; j < Ng + mesh.Ny; ++j) {
        for (int i = Ng; i <= Ng + mesh.Nx; ++i) {
            double x = mesh.x(i - Ng) + 0.5 * mesh.dx;  // x-face location
            double y = mesh.y(j - Ng);
            double y_norm = y / H;
            double pert = 0.1 * std::sin(2.0 * x) * std::cos(M_PI * y_norm);
            vel_init.u(i, j) = u0 * (1.0 + pert);
        }
    }
    
    // Initialize v at y-faces (small perturbation for continuity)
    for (int j = Ng; j <= Ng + mesh.Ny; ++j) {
        for (int i = Ng; i < Ng + mesh.Nx; ++i) {
            double x = mesh.x(i - Ng);
            double y = mesh.y(j - Ng) + 0.5 * mesh.dy;  // y-face location
            double y_norm = y / H;
            // v chosen to approximately satisfy continuity initially
            double pert = 0.05 * std::cos(2.0 * x) * std::sin(M_PI * y_norm);
            vel_init.v(i, j) = u0 * pert;
        }
    }
    
    solver.initialize(vel_init);
    
    // Solve to steady state
    ScopedTimer total_timer("Total simulation", true);
    auto [residual, iterations] = solver.solve_steady();
    total_timer.stop();
    
    // Sync solution from GPU for diagnostics (since no output files are written)
    // Note: If write_vtk/write_fields were called, they would sync automatically
#ifdef USE_GPU_OFFLOAD
    solver.sync_solution_from_gpu();
#endif
    
    // Results
    std::cout << "\n=== Results ===\n";
    std::cout << "Final residual: " << std::scientific << residual << "\n";
    std::cout << "Iterations: " << iterations << "\n";
    std::cout << "Converged: " << (residual < config.tol ? "YES" : "NO") << "\n";
    std::cout << "Bulk velocity: " << std::fixed << std::setprecision(6) << solver.bulk_velocity() << "\n";
    
    // Print timing summary
    TimingStats::instance().print_summary();
    
    return 0;
}

