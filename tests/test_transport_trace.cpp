/// Minimal test to trace CPU vs GPU transport differences
/// Runs 1 step and outputs detailed values at a specific cell

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "config.hpp"
#include "turbulence_model.hpp"
#include <iostream>
#include <iomanip>

using namespace nncfd;

int main() {
    std::cout << std::setprecision(15) << std::scientific;
    std::cout << "=== Transport Trace Test ===\n\n";

#ifdef USE_GPU_OFFLOAD
    std::cout << "Build: GPU (USE_GPU_OFFLOAD=ON)\n\n";
#else
    std::cout << "Build: CPU (USE_GPU_OFFLOAD=OFF)\n\n";
#endif

    // Same setup as golden test
    Mesh mesh;
    mesh.init_uniform(32, 32, 0.0, 2.0 * 3.14159265358979, -1.0, 1.0);

    Config config;
    config.nu = 0.001;
    config.dt = 0.001;
    config.adaptive_dt = false;
    config.max_iter = 100;
    config.tol = 1e-6;
    config.turb_model = TurbulenceModelType::SSTKOmega;
    config.verbose = false;
    config.turb_guard_enabled = true;
    config.turb_guard_interval = 5;

    RANSSolver solver(mesh, config);
    solver.set_body_force(0.001, 0.0);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);

    auto model = create_turbulence_model(TurbulenceModelType::SSTKOmega, "", "");
    solver.set_turbulence_model(std::move(model));
    solver.initialize_uniform(1.0, 0.0);

    // Set initial velocity profile
    VectorField& vel = solver.velocity();
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        double y = mesh.y(j);
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            vel.u(i, j) = 0.1 * (1.0 - y * y);
        }
    }

    // Sample cell for tracing (32, 2) - where max omega diff occurs
    int trace_i = 32, trace_j = 2;

    std::cout << "Initial state at (" << trace_i << "," << trace_j << "):\n";
    std::cout << "  k     = " << solver.k()(trace_i, trace_j) << "\n";
    std::cout << "  omega = " << solver.omega()(trace_i, trace_j) << "\n";
    std::cout << "  nu_t  = " << solver.nu_t()(trace_i, trace_j) << "\n";
    std::cout << "  u     = " << solver.velocity().u(trace_i, trace_j) << "\n";
    std::cout << "  v     = " << solver.velocity().v(trace_i, trace_j) << "\n";
    std::cout << "  y     = " << mesh.y(trace_j) << "\n";
    std::cout << "  wall_dist = " << mesh.wall_distance(trace_i, trace_j) << "\n\n";

    solver.sync_to_gpu();

    // Run just 1 step
    std::cout << "Running 1 time step...\n";
    solver.step();

    solver.sync_solution_from_gpu();

    std::cout << "\nAfter 1 step at (" << trace_i << "," << trace_j << "):\n";
    std::cout << "  k     = " << solver.k()(trace_i, trace_j) << "\n";
    std::cout << "  omega = " << solver.omega()(trace_i, trace_j) << "\n";
    std::cout << "  nu_t  = " << solver.nu_t()(trace_i, trace_j) << "\n\n";

    // Run 9 more steps to match golden test
    std::cout << "Running 9 more steps...\n";
    for (int step = 0; step < 9; ++step) {
        solver.step();
    }

    solver.sync_solution_from_gpu();

    std::cout << "\nAfter 10 steps at (" << trace_i << "," << trace_j << "):\n";
    std::cout << "  k     = " << solver.k()(trace_i, trace_j) << "\n";
    std::cout << "  omega = " << solver.omega()(trace_i, trace_j) << "\n";
    std::cout << "  nu_t  = " << solver.nu_t()(trace_i, trace_j) << "\n\n";

    // Also check a few other cells
    std::cout << "Sample omega values after 10 steps:\n";
    for (int j = 1; j <= 3; ++j) {
        for (int i = 30; i <= 32; ++i) {
            std::cout << "  omega(" << i << "," << j << ") = " << solver.omega()(i, j) << "\n";
        }
    }

    return 0;
}
