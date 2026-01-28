/// @file test_invariant_scaling.cpp
/// @brief Diagnostic: check if invariant violations scale correctly with dt/grid
///
/// If errors scale as O(dt) or O(h^2), they're discretization errors.
/// If they don't scale, there's likely a bug in the kernels.

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include "solver.hpp"
#include "mesh.hpp"
#include "test_utilities.hpp"

using namespace nncfd;
using nncfd::test::create_velocity_bc;
using nncfd::test::BCPattern;

// Test z-invariance scaling with grid size
void test_z_invariance_scaling() {
    std::cout << "\n=== Z-Invariance Scaling with Grid Size ===\n";
    std::cout << "Expected: O(h^2) scaling for 2nd-order scheme\n\n";

    std::vector<int> grids = {8, 16, 32};
    std::vector<double> z_vars;

    for (int N : grids) {
        int Nz = N / 2;  // Keep aspect ratio

        Mesh mesh;
        mesh.init_uniform(N, N, Nz, 0.0, 2.0*M_PI, 0.0, 2.0*M_PI, 0.0, 2.0*M_PI);

        Config config;
        config.nu = 0.001;
        config.dt = 0.005;
        config.adaptive_dt = false;
        config.turb_model = TurbulenceModelType::None;
        config.verbose = false;

        RANSSolver solver(mesh, config);
        solver.set_velocity_bc(create_velocity_bc(BCPattern::FullyPeriodic));

        // 2D extruded TGV IC (z-independent)
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                    double x = mesh.xf[i];
                    double y = mesh.y(j);
                    solver.velocity().u(i, j, k) = std::sin(x) * std::cos(y);
                }
            }
        }
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    double x = mesh.x(i);
                    double y = mesh.yf[j];
                    solver.velocity().v(i, j, k) = -std::cos(x) * std::sin(y);
                }
            }
        }
        for (int k = mesh.k_begin(); k <= mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    solver.velocity().w(i, j, k) = 0.0;
                }
            }
        }

        solver.sync_to_gpu();

        // Run 10 steps
        for (int step = 0; step < 10; ++step) {
            solver.step();
        }

        solver.sync_from_gpu();

        // Measure z-variation
        double max_z_var = 0.0;
        int k_ref = mesh.k_begin();
        for (int k = mesh.k_begin() + 1; k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    double u_ref = 0.5 * (solver.velocity().u(i, j, k_ref) + solver.velocity().u(i+1, j, k_ref));
                    double u_k = 0.5 * (solver.velocity().u(i, j, k) + solver.velocity().u(i+1, j, k));
                    max_z_var = std::max(max_z_var, std::abs(u_ref - u_k));
                }
            }
        }

        z_vars.push_back(max_z_var);
        double h = 2.0 * M_PI / N;
        std::cout << "  N=" << std::setw(2) << N << ", h=" << std::scientific << std::setprecision(3) << h
                  << ", z_var=" << max_z_var << "\n";
    }

    // Check scaling
    std::cout << "\nScaling ratios (should be ~4 for O(h^2)):\n";
    for (size_t i = 1; i < z_vars.size(); ++i) {
        double ratio = z_vars[i-1] / z_vars[i];
        std::cout << "  z_var[N=" << grids[i-1] << "] / z_var[N=" << grids[i] << "] = "
                  << std::fixed << std::setprecision(2) << ratio << "\n";
    }
}

// Test spurious v scaling with dt
void test_spurious_v_scaling_dt() {
    std::cout << "\n=== Spurious V Scaling with dt ===\n";
    std::cout << "Expected: O(dt) or O(dt^2) for time discretization error\n\n";

    const int N = 32;
    std::vector<double> dts = {0.02, 0.01, 0.005};
    std::vector<double> spurious_vs;

    for (double dt : dts) {
        Mesh mesh;
        mesh.init_uniform(N, N, 0.0, 2.0*M_PI, 0.0, 2.0*M_PI);

        Config config;
        config.nu = 0.01;
        config.dt = dt;
        config.adaptive_dt = false;
        config.turb_model = TurbulenceModelType::None;
        config.verbose = false;

        RANSSolver solver(mesh, config);

        solver.set_velocity_bc(create_velocity_bc(BCPattern::FullyPeriodic));

        // Shear flow: u = sin(y), v = 0
        for (int j = 0; j <= mesh.Ny + 1; ++j) {
            double y = mesh.y(j);
            for (int i = 0; i <= mesh.Nx + 1; ++i) {
                solver.velocity().u(i, j) = std::sin(y);
                solver.velocity().v(i, j) = 0.0;
            }
        }

        solver.sync_to_gpu();

        // Run to same physical time T=0.5
        int nsteps = static_cast<int>(0.5 / dt);
        for (int step = 0; step < nsteps; ++step) {
            solver.step();
        }

        solver.sync_from_gpu();

        // Measure spurious v
        double max_u = 0.0, max_v = 0.0;
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                max_u = std::max(max_u, std::abs(solver.velocity().u(i, j)));
            }
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                max_v = std::max(max_v, std::abs(solver.velocity().v(i, j)));
            }
        }

        double v_over_u = max_v / (max_u + 1e-30);
        spurious_vs.push_back(v_over_u);

        std::cout << "  dt=" << std::scientific << std::setprecision(3) << dt
                  << ", nsteps=" << std::setw(3) << nsteps
                  << ", max|v|/max|u|=" << v_over_u << "\n";
    }

    std::cout << "\nScaling ratios (should be ~2 for O(dt), ~4 for O(dt^2)):\n";
    for (size_t i = 1; i < spurious_vs.size(); ++i) {
        double ratio = spurious_vs[i-1] / spurious_vs[i];
        std::cout << "  v/u[dt=" << dts[i-1] << "] / v/u[dt=" << dts[i] << "] = "
                  << std::fixed << std::setprecision(2) << ratio << "\n";
    }
}

// Test spurious v scaling with grid
void test_spurious_v_scaling_grid() {
    std::cout << "\n=== Spurious V Scaling with Grid Size ===\n";
    std::cout << "Expected: O(h^2) for spatial discretization error\n\n";

    std::vector<int> grids = {16, 32, 64};
    std::vector<double> spurious_vs;

    for (int N : grids) {
        Mesh mesh;
        mesh.init_uniform(N, N, 0.0, 2.0*M_PI, 0.0, 2.0*M_PI);

        Config config;
        config.nu = 0.01;
        config.dt = 0.005;  // Small dt to minimize time error
        config.adaptive_dt = false;
        config.turb_model = TurbulenceModelType::None;
        config.verbose = false;

        RANSSolver solver(mesh, config);

        solver.set_velocity_bc(create_velocity_bc(BCPattern::FullyPeriodic));

        // Shear flow: u = sin(y), v = 0
        for (int j = 0; j <= mesh.Ny + 1; ++j) {
            double y = mesh.y(j);
            for (int i = 0; i <= mesh.Nx + 1; ++i) {
                solver.velocity().u(i, j) = std::sin(y);
                solver.velocity().v(i, j) = 0.0;
            }
        }

        solver.sync_to_gpu();

        // Run 50 steps
        for (int step = 0; step < 50; ++step) {
            solver.step();
        }

        solver.sync_from_gpu();

        // Measure spurious v
        double max_u = 0.0, max_v = 0.0;
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                max_u = std::max(max_u, std::abs(solver.velocity().u(i, j)));
            }
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                max_v = std::max(max_v, std::abs(solver.velocity().v(i, j)));
            }
        }

        double v_over_u = max_v / (max_u + 1e-30);
        spurious_vs.push_back(v_over_u);

        double h = 2.0 * M_PI / N;
        std::cout << "  N=" << std::setw(2) << N << ", h=" << std::scientific << std::setprecision(3) << h
                  << ", max|v|/max|u|=" << v_over_u << "\n";
    }

    std::cout << "\nScaling ratios (should be ~4 for O(h^2)):\n";
    for (size_t i = 1; i < spurious_vs.size(); ++i) {
        double ratio = spurious_vs[i-1] / spurious_vs[i];
        std::cout << "  v/u[N=" << grids[i-1] << "] / v/u[N=" << grids[i] << "] = "
                  << std::fixed << std::setprecision(2) << ratio << "\n";
    }
}

// Test constant velocity drift scaling
void test_const_vel_scaling() {
    std::cout << "\n=== Constant Velocity Drift Scaling ===\n";
    std::cout << "Expected: Should stay O(machine eps) or O(solver tol)\n\n";

    std::vector<int> grids = {16, 32, 64};

    for (int N : grids) {
        Mesh mesh;
        mesh.init_uniform(N, N, 0.0, 2.0*M_PI, 0.0, 2.0*M_PI);

        Config config;
        config.nu = 0.01;
        config.dt = 0.01;
        config.adaptive_dt = false;
        config.turb_model = TurbulenceModelType::None;
        config.verbose = false;

        RANSSolver solver(mesh, config);
        solver.set_velocity_bc(create_velocity_bc(BCPattern::FullyPeriodic));

        // Constant velocity IC
        const double u_const = 1.5, v_const = 0.5;
        for (int j = 0; j <= mesh.Ny + 1; ++j) {
            for (int i = 0; i <= mesh.Nx + 1; ++i) {
                solver.velocity().u(i, j) = u_const;
                solver.velocity().v(i, j) = v_const;
            }
        }

        solver.sync_to_gpu();

        // Run 50 steps
        for (int step = 0; step < 50; ++step) {
            solver.step();
        }

        solver.sync_from_gpu();

        // Measure drift
        double max_u_drift = 0.0, max_v_drift = 0.0;
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                max_u_drift = std::max(max_u_drift, std::abs(solver.velocity().u(i, j) - u_const));
            }
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                max_v_drift = std::max(max_v_drift, std::abs(solver.velocity().v(i, j) - v_const));
            }
        }

        std::cout << "  N=" << std::setw(2) << N
                  << ", max|u-u0|=" << std::scientific << std::setprecision(3) << max_u_drift
                  << ", max|v-v0|=" << max_v_drift << "\n";
    }
}

int main() {
    std::cout << "================================================================\n";
    std::cout << "  Invariant Violation Scaling Diagnostics\n";
    std::cout << "================================================================\n";

#ifdef USE_GPU_OFFLOAD
    std::cout << "\nBuild: GPU (USE_GPU_OFFLOAD=ON)\n";
#else
    std::cout << "\nBuild: CPU (USE_GPU_OFFLOAD=OFF)\n";
#endif

    test_z_invariance_scaling();
    test_spurious_v_scaling_dt();
    test_spurious_v_scaling_grid();
    test_const_vel_scaling();

    std::cout << "\n================================================================\n";
    std::cout << "  Interpretation Guide\n";
    std::cout << "================================================================\n";
    std::cout << "If errors scale as expected (O(h^2), O(dt)), they're discretization errors.\n";
    std::cout << "If errors DON'T scale or scale incorrectly, there's likely a bug.\n";
    std::cout << "If errors are O(1) regardless of resolution, check BCs/kernels.\n";

    return 0;
}
