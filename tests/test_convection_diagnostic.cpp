/// @file test_convection_diagnostic.cpp
/// @brief Diagnose spurious velocities: check convection and pressure separately

#include <iostream>
#include <iomanip>
#include <cmath>
#include "solver.hpp"
#include "mesh.hpp"

using namespace nncfd;

// Check if convection alone creates spurious v for u=sin(y), v=0
void test_convection_only() {
    std::cout << "\n=== Convection Term Diagnostic for u=sin(y), v=0 ===\n\n";

    const int N = 32;
    Mesh mesh;
    mesh.init_uniform(N, N, 0.0, 2.0*M_PI, 0.0, 2.0*M_PI);

    Config config;
    config.nu = 0.0;  // ZERO viscosity - pure convection
    config.dt = 0.01;
    config.adaptive_dt = false;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::Periodic;
    bc.y_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    // Shear flow: u = sin(y), v = 0 (divergence-free, no convection)
    for (int j = 0; j <= mesh.Ny + 1; ++j) {
        double y = mesh.y(j);
        for (int i = 0; i <= mesh.Nx + 1; ++i) {
            solver.velocity().u(i, j) = std::sin(y);
            solver.velocity().v(i, j) = 0.0;
        }
    }

    // Check initial divergence
    double max_div_init = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double div = (solver.velocity().u(i+1, j) - solver.velocity().u(i, j)) / mesh.dx
                       + (solver.velocity().v(i, j+1) - solver.velocity().v(i, j)) / mesh.dy;
            max_div_init = std::max(max_div_init, std::abs(div));
        }
    }
    std::cout << "  Initial max|div|: " << std::scientific << max_div_init << "\n";

    solver.sync_to_gpu();

    // Run ONE step only to see initial effect
    solver.step();

    solver.sync_from_gpu();

    // Check what v became
    double max_v_after1 = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            max_v_after1 = std::max(max_v_after1, std::abs(solver.velocity().v(i, j)));
        }
    }
    std::cout << "  After 1 step: max|v| = " << max_v_after1 << "\n";

    // Check divergence after step
    double max_div_after1 = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double div = (solver.velocity().u(i+1, j) - solver.velocity().u(i, j)) / mesh.dx
                       + (solver.velocity().v(i, j+1) - solver.velocity().v(i, j)) / mesh.dy;
            max_div_after1 = std::max(max_div_after1, std::abs(div));
        }
    }
    std::cout << "  After 1 step: max|div| = " << max_div_after1 << "\n\n";

    // Check pattern of v - is it structured or random?
    std::cout << "  V-velocity sample at j=N/2, k=const:\n  ";
    int j_mid = mesh.Ny / 2 + mesh.j_begin();
    for (int i = mesh.i_begin(); i < std::min(mesh.i_begin() + 8, mesh.i_end()); ++i) {
        std::cout << std::setw(10) << std::scientific << std::setprecision(2)
                  << solver.velocity().v(i, j_mid);
    }
    std::cout << " ...\n";
}

// Check if constant velocity really has zero convection/diffusion
void test_constant_velocity_terms() {
    std::cout << "\n=== Constant Velocity Term Check ===\n\n";

    const int N = 32;
    const double u_const = 1.5;
    const double v_const = 0.75;

    Mesh mesh;
    mesh.init_uniform(N, N, 0.0, 2.0*M_PI, 0.0, 2.0*M_PI);

    Config config;
    config.nu = 0.01;
    config.dt = 0.01;
    config.adaptive_dt = false;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::Periodic;
    bc.y_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    // Constant velocity everywhere
    for (int j = 0; j <= mesh.Ny + 1; ++j) {
        for (int i = 0; i <= mesh.Nx + 1; ++i) {
            solver.velocity().u(i, j) = u_const;
            solver.velocity().v(i, j) = v_const;
        }
    }

    // Check initial divergence
    double max_div = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double div = (solver.velocity().u(i+1, j) - solver.velocity().u(i, j)) / mesh.dx
                       + (solver.velocity().v(i, j+1) - solver.velocity().v(i, j)) / mesh.dy;
            max_div = std::max(max_div, std::abs(div));
        }
    }
    std::cout << "  Initial max|div|: " << std::scientific << max_div << "\n";

    solver.sync_to_gpu();

    // Run ONE step
    solver.step();

    solver.sync_from_gpu();

    // Check drift after one step
    double max_u_drift = 0.0, max_v_drift = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            max_u_drift = std::max(max_u_drift, std::abs(solver.velocity().u(i, j) - u_const));
        }
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            max_v_drift = std::max(max_v_drift, std::abs(solver.velocity().v(i, j) - v_const));
        }
    }
    std::cout << "  After 1 step: max|u-u0| = " << max_u_drift << "\n";
    std::cout << "  After 1 step: max|v-v0| = " << max_v_drift << "\n";

    // Check divergence after step
    double max_div_after = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double div = (solver.velocity().u(i+1, j) - solver.velocity().u(i, j)) / mesh.dx
                       + (solver.velocity().v(i, j+1) - solver.velocity().v(i, j)) / mesh.dy;
            max_div_after = std::max(max_div_after, std::abs(div));
        }
    }
    std::cout << "  After 1 step: max|div| = " << max_div_after << "\n\n";

    // Sample the drift pattern
    std::cout << "  U-drift sample at j=N/2:\n  ";
    int j_mid = mesh.Ny / 2 + mesh.j_begin();
    for (int i = mesh.i_begin(); i < std::min(mesh.i_begin() + 8, mesh.i_end()); ++i) {
        std::cout << std::setw(10) << std::scientific << std::setprecision(2)
                  << (solver.velocity().u(i, j_mid) - u_const);
    }
    std::cout << " ...\n";
}

// Check if pressure projection is the source of spurious v
void test_projection_diagnostic() {
    std::cout << "\n=== Pressure Projection Diagnostic ===\n\n";

    const int N = 32;
    Mesh mesh;
    mesh.init_uniform(N, N, 0.0, 2.0*M_PI, 0.0, 2.0*M_PI);

    Config config;
    config.nu = 0.01;
    config.dt = 0.01;
    config.adaptive_dt = false;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::Periodic;
    bc.y_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    // TGV: u = sin(x)cos(y), v = -cos(x)sin(y) (exactly div-free)
    for (int j = 0; j <= mesh.Ny + 1; ++j) {
        for (int i = 0; i <= mesh.Nx + 1; ++i) {
            // u at x-face (i, j) uses xf[i] and y[j]
            double xf = mesh.xf[i];
            double yc = mesh.y(j);
            solver.velocity().u(i, j) = std::sin(xf) * std::cos(yc);

            // v at y-face (i, j) uses x[i] and yf[j]
            double xc = mesh.x(i);
            double yf = mesh.yf[j];
            solver.velocity().v(i, j) = -std::cos(xc) * std::sin(yf);
        }
    }

    // Check initial divergence
    double max_div_init = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double div = (solver.velocity().u(i+1, j) - solver.velocity().u(i, j)) / mesh.dx
                       + (solver.velocity().v(i, j+1) - solver.velocity().v(i, j)) / mesh.dy;
            max_div_init = std::max(max_div_init, std::abs(div));
        }
    }
    std::cout << "  TGV initial max|div|: " << std::scientific << max_div_init << "\n";

    solver.sync_to_gpu();

    // Run a few steps
    for (int step = 0; step < 10; ++step) {
        solver.step();
    }

    solver.sync_from_gpu();

    // Check divergence after projection
    double max_div_after = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double div = (solver.velocity().u(i+1, j) - solver.velocity().u(i, j)) / mesh.dx
                       + (solver.velocity().v(i, j+1) - solver.velocity().v(i, j)) / mesh.dy;
            max_div_after = std::max(max_div_after, std::abs(div));
        }
    }
    std::cout << "  After 10 steps: max|div| = " << max_div_after << "\n";

    // Check symmetry
    double u_mean = 0.0, v_mean = 0.0;
    int count = 0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            u_mean += 0.5 * (solver.velocity().u(i, j) + solver.velocity().u(i+1, j));
            v_mean += 0.5 * (solver.velocity().v(i, j) + solver.velocity().v(i, j+1));
            count++;
        }
    }
    u_mean /= count;
    v_mean /= count;
    std::cout << "  Mean u: " << u_mean << ", Mean v: " << v_mean << "\n";
}

int main() {
    std::cout << "================================================================\n";
    std::cout << "  Convection/Projection Diagnostics\n";
    std::cout << "================================================================\n";

#ifdef USE_GPU_OFFLOAD
    std::cout << "\nBuild: GPU (USE_GPU_OFFLOAD=ON)\n";
#else
    std::cout << "\nBuild: CPU (USE_GPU_OFFLOAD=OFF)\n";
#endif

    test_convection_only();
    test_constant_velocity_terms();
    test_projection_diagnostic();

    return 0;
}
