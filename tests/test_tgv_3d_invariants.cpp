/// @file test_tgv_3d_invariants.cpp
/// @brief 3D Taylor-Green vortex invariant tests for CI
///
/// PURPOSE: "3D Canary" - catches 3D-only indexing/strides/halo plane sync
/// and GPU memory layout issues that won't show up in 2D tests.
///
/// Verifies physical invariants:
///   1. Divergence-free: max|div(u)| should remain small after projection
///   2. Energy monotonicity: For nu > 0, kinetic energy should not increase
///   3. Symmetry: mean(u), mean(v), mean(w) should remain near zero
///
/// Test cases:
///   - 16x16x16 grid, 100 steps (medium, runs in Release CI)
///   - Fully periodic domain [0, 2π]³
///   - Classic TGV: u=sin(x)cos(y)cos(z), v=-cos(x)sin(y)cos(z), w=0

#include "test_harness.hpp"
#include "test_utilities.hpp"
#include "solver.hpp"
#include "config.hpp"
#include <cmath>
#include <vector>
#include <iomanip>
#include <algorithm>

using namespace nncfd;
using namespace nncfd::test;
using nncfd::test::harness::record;

// ============================================================================
// Helper: Compute max divergence for 3D (L-infinity norm)
// ============================================================================
static double compute_max_divergence_3d(const VectorField& vel, const Mesh& mesh) {
    double max_div = 0.0;
    double dx = mesh.dx;
    double dy = mesh.dy;
    double dz = mesh.dz;

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                // MAC grid: u at x-faces, v at y-faces, w at z-faces
                double du_dx = (vel.u(i+1, j, k) - vel.u(i, j, k)) / dx;
                double dv_dy = (vel.v(i, j+1, k) - vel.v(i, j, k)) / dy;
                double dw_dz = (vel.w(i, j, k+1) - vel.w(i, j, k)) / dz;
                double div = std::abs(du_dx + dv_dy + dw_dz);
                max_div = std::max(max_div, div);
            }
        }
    }
    return max_div;
}

// ============================================================================
// Helper: Compute kinetic energy for 3D
// ============================================================================
static double compute_kinetic_energy_3d(const VectorField& vel, const Mesh& mesh) {
    double KE = 0.0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                // Interpolate to cell centers
                double u = 0.5 * (vel.u(i, j, k) + vel.u(i+1, j, k));
                double v = 0.5 * (vel.v(i, j, k) + vel.v(i, j+1, k));
                double w = 0.5 * (vel.w(i, j, k) + vel.w(i, j, k+1));
                KE += 0.5 * (u*u + v*v + w*w) * mesh.dx * mesh.dy * mesh.dz;
            }
        }
    }
    return KE;
}

// ============================================================================
// Helper: Compute mean velocity components for 3D
// ============================================================================
static void compute_mean_velocity_3d(const VectorField& vel, const Mesh& mesh,
                                     double& u_mean, double& v_mean, double& w_mean) {
    double u_sum = 0.0, v_sum = 0.0, w_sum = 0.0;
    int count = 0;

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                u_sum += 0.5 * (vel.u(i, j, k) + vel.u(i+1, j, k));
                v_sum += 0.5 * (vel.v(i, j, k) + vel.v(i, j+1, k));
                w_sum += 0.5 * (vel.w(i, j, k) + vel.w(i, j, k+1));
                ++count;
            }
        }
    }

    if (count > 0) {
        u_mean = u_sum / count;
        v_mean = v_sum / count;
        w_mean = w_sum / count;
    } else {
        u_mean = v_mean = w_mean = 0.0;
    }
}

// ============================================================================
// Helper: Initialize 3D Taylor-Green vortex
// ============================================================================
static void init_taylor_green_3d(RANSSolver& solver, const Mesh& mesh) {
    // Classic 3D TGV: u = sin(x)cos(y)cos(z), v = -cos(x)sin(y)cos(z), w = 0
    // This is divergence-free: du/dx + dv/dy + dw/dz = cos cos cos - cos cos cos + 0 = 0

    // u at x-faces
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                double x = mesh.xf[i];  // x-face location
                double y = mesh.y(j);   // cell center y
                double z = mesh.z(k);   // cell center z
                solver.velocity().u(i, j, k) = std::sin(x) * std::cos(y) * std::cos(z);
            }
        }
    }

    // v at y-faces
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double x = mesh.x(i);   // cell center x
                double y = mesh.yf[j];  // y-face location
                double z = mesh.z(k);   // cell center z
                solver.velocity().v(i, j, k) = -std::cos(x) * std::sin(y) * std::cos(z);
            }
        }
    }

    // w at z-faces (zero for classic TGV)
    for (int k = mesh.k_begin(); k <= mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                solver.velocity().w(i, j, k) = 0.0;
            }
        }
    }
}

// ============================================================================
// Test: 3D Taylor-Green Vortex Invariants
// ============================================================================
void test_tgv_3d_invariants() {
    std::cout << "\n--- 3D Taylor-Green Vortex Invariants ---\n\n";

    // Configuration: 16^3 periodic, nu=1e-3, 100 steps
    const int N = 16;
    const int nsteps = 100;
    const double nu = 1e-3;
    const double dt_max = 5e-3;  // Cap dt for determinism
    const double L = 2.0 * M_PI;

    // Thresholds
    const double div_threshold = 1e-6;
    const double energy_growth_tol = 1e-12;
    const double mean_vel_threshold = 1e-12;

    // Setup mesh and config
    Mesh mesh;
    mesh.init_uniform(N, N, N, 0.0, L, 0.0, L, 0.0, L);

    Config config;
    config.nu = nu;
    config.dt = dt_max;
    config.adaptive_dt = false;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    // Fully periodic BCs
    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::Periodic;
    bc.y_hi = VelocityBC::Periodic;
    bc.z_lo = VelocityBC::Periodic;
    bc.z_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    // Initialize with 3D Taylor-Green vortex
    init_taylor_green_3d(solver, mesh);
    solver.sync_to_gpu();

    // Track metrics
    double E_prev = compute_kinetic_energy_3d(solver.velocity(), mesh);
    double max_div_observed = 0.0;
    bool energy_monotonic = true;
    int energy_violation_step = -1;

    std::vector<double> energy_history;
    energy_history.push_back(E_prev);

    // Run simulation, check invariants every step
    for (int step = 1; step <= nsteps; ++step) {
        solver.step();
        solver.sync_from_gpu();

        // Compute divergence
        double div = compute_max_divergence_3d(solver.velocity(), mesh);
        max_div_observed = std::max(max_div_observed, div);

        // Compute energy
        double E_curr = compute_kinetic_energy_3d(solver.velocity(), mesh);
        energy_history.push_back(E_curr);

        // Check energy monotonicity
        if (E_curr > E_prev * (1.0 + energy_growth_tol)) {
            if (energy_monotonic) {
                energy_monotonic = false;
                energy_violation_step = step;
            }
        }

        E_prev = E_curr;
    }

    // Compute final mean velocities (should be ~0 for symmetric IC)
    double u_mean, v_mean, w_mean;
    compute_mean_velocity_3d(solver.velocity(), mesh, u_mean, v_mean, w_mean);

    // Print diagnostic info
    std::cout << "  Grid: " << N << "^3, steps: " << nsteps
              << ", nu: " << nu << ", dt: " << dt_max << "\n";
    std::cout << "  max_div_Linf observed: " << std::scientific << std::setprecision(2)
              << max_div_observed << " (threshold: " << div_threshold << ")\n";
    std::cout << "  KE decay: " << std::fixed << std::setprecision(4)
              << energy_history.back() / energy_history.front()
              << " (initial: " << std::scientific << energy_history.front() << ")\n";
    std::cout << "  Mean velocities: |u|=" << std::abs(u_mean)
              << ", |v|=" << std::abs(v_mean)
              << ", |w|=" << std::abs(w_mean) << "\n";

    if (!energy_monotonic) {
        std::cout << "  [WARN] Energy grew at step " << energy_violation_step << "\n";
    }
    std::cout << "\n";

    // Record test results
    record("3D Divergence-free (max|div| < 1e-6)", max_div_observed < div_threshold);
    record("3D Energy monotonicity (E non-increasing)", energy_monotonic);
    record("3D Energy bounded (final KE finite)", std::isfinite(energy_history.back()));
    record("3D Symmetry (|mean(u)| < 1e-12)", std::abs(u_mean) < mean_vel_threshold);
    record("3D Symmetry (|mean(v)| < 1e-12)", std::abs(v_mean) < mean_vel_threshold);
    record("3D Symmetry (|mean(w)| < 1e-12)", std::abs(w_mean) < mean_vel_threshold);
}

// ============================================================================
// Test: Initial divergence for 3D (verify init is div-free)
// ============================================================================
void test_tgv_3d_initial_divergence() {
    std::cout << "\n--- 3D Initial Divergence Check ---\n\n";

    const int N = 16;
    Mesh mesh;
    mesh.init_uniform(N, N, N, 0.0, 2.0*M_PI, 0.0, 2.0*M_PI, 0.0, 2.0*M_PI);

    Config config;
    config.nu = 0.01;
    config.dt = 0.001;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::Periodic;
    bc.y_hi = VelocityBC::Periodic;
    bc.z_lo = VelocityBC::Periodic;
    bc.z_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    init_taylor_green_3d(solver, mesh);

    // Check divergence BEFORE any time stepping
    double initial_div = compute_max_divergence_3d(solver.velocity(), mesh);

    std::cout << "  Initial max|div|: " << std::scientific << initial_div << "\n\n";

    // 3D TGV should be analytically divergence-free
    record("3D Initial field divergence-free (< 1e-10)", initial_div < 1e-10);
}

// ============================================================================
// Helper: Initialize 2D-extruded TGV (z-independent) for stride testing
// ============================================================================
static void init_taylor_green_2d_extruded(RANSSolver& solver, const Mesh& mesh) {
    // 2D TGV extruded to 3D: u = sin(x)cos(y), v = -cos(x)sin(y), w = 0
    // This is z-independent and should remain so after time-stepping

    // u at x-faces
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                double x = mesh.xf[i];
                double y = mesh.y(j);
                solver.velocity().u(i, j, k) = std::sin(x) * std::cos(y);
            }
        }
    }

    // v at y-faces
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double x = mesh.x(i);
                double y = mesh.yf[j];
                solver.velocity().v(i, j, k) = -std::cos(x) * std::sin(y);
            }
        }
    }

    // w at z-faces (zero)
    for (int k = mesh.k_begin(); k <= mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                solver.velocity().w(i, j, k) = 0.0;
            }
        }
    }
}

// ============================================================================
// Test: 3D stride/indexing verification
// ============================================================================
void test_tgv_3d_stride_verification() {
    std::cout << "\n--- 3D Stride/Indexing Verification ---\n\n";

    // This test verifies that 3D memory layout and indexing is correct
    // by using a z-independent IC (2D TGV extruded) and checking
    // that the solution remains z-invariant after time stepping

    const int Nx = 16, Ny = 16, Nz = 8;
    const int nsteps = 10;

    Mesh mesh;
    mesh.init_uniform(Nx, Ny, Nz, 0.0, 2.0*M_PI, 0.0, 2.0*M_PI, 0.0, 2.0*M_PI);

    Config config;
    config.nu = 0.01;
    config.dt = 0.005;
    config.adaptive_dt = false;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::Periodic;
    bc.y_hi = VelocityBC::Periodic;
    bc.z_lo = VelocityBC::Periodic;
    bc.z_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    // Use z-independent IC for stride verification
    init_taylor_green_2d_extruded(solver, mesh);
    solver.sync_to_gpu();

    for (int step = 0; step < nsteps; ++step) {
        solver.step();
    }
    solver.sync_from_gpu();

    // Check z-invariance: compare slices at different k values
    double max_z_variation = 0.0;
    int k_ref = mesh.k_begin();

    for (int k = mesh.k_begin() + 1; k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double u_ref = 0.5 * (solver.velocity().u(i, j, k_ref) + solver.velocity().u(i+1, j, k_ref));
                double u_k = 0.5 * (solver.velocity().u(i, j, k) + solver.velocity().u(i+1, j, k));
                max_z_variation = std::max(max_z_variation, std::abs(u_ref - u_k));

                double v_ref = 0.5 * (solver.velocity().v(i, j, k_ref) + solver.velocity().v(i, j+1, k_ref));
                double v_k = 0.5 * (solver.velocity().v(i, j, k) + solver.velocity().v(i, j+1, k));
                max_z_variation = std::max(max_z_variation, std::abs(v_ref - v_k));
            }
        }
    }

    std::cout << "  Max z-variation: " << std::scientific << max_z_variation << "\n\n";

    // Z-variation should remain small. Allow for numerical effects in 3D
    // (FP accumulation, slight z-coupling through Poisson solver)
    // Threshold 1e-4 catches gross stride/indexing bugs while tolerating FP effects
    record("3D z-invariance preserved (< 1e-4)", max_z_variation < 1e-4);
}

// ============================================================================
// Main
// ============================================================================
int main() {
    return harness::run("3D Taylor-Green Vortex Invariants", []() {
        test_tgv_3d_initial_divergence();
        test_tgv_3d_invariants();
        test_tgv_3d_stride_verification();
    });
}
