/// @file test_step_trace.cpp
/// @brief Trace ALL intermediate values during a step for constant velocity
///
/// For constant velocity, conv=diff=0, so u*=u, div(u*)=0, RHS=0, p'=0.

#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include "solver.hpp"
#include "mesh.hpp"
#include "test_utilities.hpp"

using namespace nncfd;
using nncfd::test::create_velocity_bc;
using nncfd::test::BCPattern;

int main() {
    std::cout << "================================================================\n";
    std::cout << "  Step Trace for Constant Velocity - GPU Path\n";
    std::cout << "================================================================\n\n";

    const int N = 8;
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
    config.convective_scheme = ConvectiveScheme::Skew;
    config.poisson_fixed_cycles = 8;

    RANSSolver solver(mesh, config);
    solver.set_velocity_bc(create_velocity_bc(BCPattern::FullyPeriodic));
    solver.set_body_force(0.0, 0.0);

    std::cout << "Setup:\n";
    std::cout << "  Grid: " << N << "x" << N << "\n";
    std::cout << "  Domain: [0, 2pi]^2, dx = " << mesh.dx << "\n";
    std::cout << "  nu = " << config.nu << ", dt = " << config.dt << "\n";
    std::cout << "  Velocity: u = " << u_const << ", v = " << v_const << "\n\n";

    // Set constant velocity everywhere (including ALL ghost cells)
    const int Ng = mesh.Nghost;
    for (int j = 0; j <= mesh.Ny + 2*Ng - 1; ++j) {
        for (int i = 0; i <= mesh.Nx + 2*Ng; ++i) {  // u has Nx+1 faces
            solver.velocity().u(i, j) = u_const;
        }
    }
    for (int j = 0; j <= mesh.Ny + 2*Ng; ++j) {  // v has Ny+1 faces
        for (int i = 0; i <= mesh.Nx + 2*Ng - 1; ++i) {
            solver.velocity().v(i, j) = v_const;
        }
    }

    std::cout << "1. Initial velocity:\n";
    std::cout << "   u values at j=" << mesh.j_begin() << ":\n";
    for (int i = mesh.i_begin(); i <= mesh.i_end() + 1; ++i) {
        std::cout << "     u(" << i << ") = " << solver.velocity().u(i, mesh.j_begin()) << "\n";
    }

    // Verify periodicity in ghost cells
    int j_mid = mesh.j_begin() + N/2;
    std::cout << "\n   Ghost cell check at j=" << j_mid << ":\n";
    std::cout << "     u(ghost_left=" << (Ng-1) << ") = " << solver.velocity().u(Ng-1, j_mid) << "\n";
    std::cout << "     u(interior_left=" << Ng << ") = " << solver.velocity().u(Ng, j_mid) << "\n";
    std::cout << "     u(interior_right=" << (Ng+N) << ") = " << solver.velocity().u(Ng+N, j_mid) << "\n";
    std::cout << "     u(ghost_right=" << (Ng+N+1) << ") = " << solver.velocity().u(Ng+N+1, j_mid) << "\n";

    // Compute initial divergence manually
    std::cout << "\n2. Initial divergence (computed manually):\n";
    double max_div = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double dudx = (solver.velocity().u(i+1, j) - solver.velocity().u(i, j)) / mesh.dx;
            double dvdy = (solver.velocity().v(i, j+1) - solver.velocity().v(i, j)) / mesh.dy;
            double div = dudx + dvdy;
            max_div = std::max(max_div, std::fabs(div));
        }
    }
    std::cout << "   max|div(u)| = " << std::scientific << max_div << "\n";

    // Run ONE step on GPU
    std::cout << "\n3. Running ONE step (GPU path)...\n\n";
    solver.sync_to_gpu();
    solver.step();
    solver.sync_from_gpu();

    // Get access to internal fields via solver view
    // NOTE: In GPU builds, conv_u/diff_u etc. are device pointers NOT synced by sync_from_gpu()
    // so we can only access them in CPU builds
    auto view = solver.get_solver_view();

#ifndef USE_GPU_OFFLOAD
    std::cout << "4. Convection terms:\n";
    double max_conv_u = 0, max_conv_v = 0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            int idx = j * view.u_stride + i;
            max_conv_u = std::max(max_conv_u, std::fabs(view.conv_u[idx]));
        }
    }
    for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            int idx = j * view.v_stride + i;
            max_conv_v = std::max(max_conv_v, std::fabs(view.conv_v[idx]));
        }
    }
    std::cout << "   max|conv_u| = " << std::scientific << max_conv_u << "\n";
    std::cout << "   max|conv_v| = " << max_conv_v << "\n";

    std::cout << "\n5. Diffusion terms:\n";
    double max_diff_u = 0, max_diff_v = 0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            int idx = j * view.u_stride + i;
            max_diff_u = std::max(max_diff_u, std::fabs(view.diff_u[idx]));
        }
    }
    for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            int idx = j * view.v_stride + i;
            max_diff_v = std::max(max_diff_v, std::fabs(view.diff_v[idx]));
        }
    }
    std::cout << "   max|diff_u| = " << std::scientific << max_diff_u << "\n";
    std::cout << "   max|diff_v| = " << max_diff_v << "\n";
#else
    std::cout << "4. Convection terms: [skipped - GPU work arrays not synced]\n";
    std::cout << "5. Diffusion terms: [skipped - GPU work arrays not synced]\n";
#endif

    std::cout << "\n6. Poisson RHS (= div(u*)/dt):\n";
    double max_rhs = 0, rhs_sum = 0;
    int rhs_count = 0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            int idx = j * view.cell_stride + i;
            double rhs = view.rhs[idx];
            max_rhs = std::max(max_rhs, std::fabs(rhs));
            rhs_sum += rhs;
            rhs_count++;
        }
    }
    std::cout << "   max|RHS| = " << std::scientific << max_rhs << "\n";
    std::cout << "   sum(RHS) = " << rhs_sum << "\n";

    // Sample RHS values
    std::cout << "   RHS values at j=" << j_mid << ":\n";
    for (int i = mesh.i_begin(); i < std::min(mesh.i_begin() + 4, mesh.i_end()); ++i) {
        int idx = j_mid * view.cell_stride + i;
        std::cout << "     RHS(" << i << ") = " << view.rhs[idx] << "\n";
    }

    std::cout << "\n7. Pressure correction:\n";
    const auto& p_corr = solver.pressure_correction();
    double p_min = 1e30, p_max = -1e30;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double p = p_corr(i, j);
            p_min = std::min(p_min, p);
            p_max = std::max(p_max, p);
        }
    }
    std::cout << "   p_corr range: [" << std::scientific << p_min << ", " << p_max << "]\n";
    std::cout << "   p_corr spread: " << (p_max - p_min) << "\n";

    // Sample p_corr values
    std::cout << "   p_corr at j=" << j_mid << ":\n";
    for (int i = mesh.i_begin(); i < std::min(mesh.i_begin() + 4, mesh.i_end()); ++i) {
        std::cout << "     p(" << i << ") = " << std::fixed << std::setprecision(8) << p_corr(i, j_mid) << "\n";
    }

    std::cout << "\n8. Final velocity:\n";
    double u_drift_max = 0, v_drift_max = 0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            u_drift_max = std::max(u_drift_max, std::fabs(solver.velocity().u(i, j) - u_const));
        }
    }
    for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            v_drift_max = std::max(v_drift_max, std::fabs(solver.velocity().v(i, j) - v_const));
        }
    }
    std::cout << "   max|u_drift| = " << std::scientific << u_drift_max << "\n";
    std::cout << "   max|v_drift| = " << v_drift_max << "\n";

    std::cout << "\n=== Analysis ===\n\n";

    bool pass = true;
    double tol = 1e-10;

#ifndef USE_GPU_OFFLOAD
    if (max_conv_u > tol || max_conv_v > tol) {
        std::cout << "ISSUE: Non-zero convection terms\n";
        pass = false;
    }
    if (max_diff_u > tol || max_diff_v > tol) {
        std::cout << "ISSUE: Non-zero diffusion terms\n";
        pass = false;
    }
#endif
    if (max_rhs > tol) {
        std::cout << "ISSUE: Non-zero Poisson RHS\n";
        pass = false;
    }
    if (p_max - p_min > tol) {
        std::cout << "ISSUE: Non-constant pressure correction\n";
        pass = false;
    }
    if (u_drift_max > tol || v_drift_max > tol) {
        std::cout << "ISSUE: Velocity drift\n";
        pass = false;
    }

    if (pass) {
        std::cout << "PASS: All values as expected for constant velocity\n";
        return 0;
    } else {
        std::cout << "\nFAIL: Unexpected values detected\n";
        return 1;
    }
}
