/// Unit tests for solver error handling and recovery
///
/// Tests failure modes and edge cases:
/// - Poisson solver non-convergence behavior
/// - NaN detection and field identification
/// - Singular system handling (pure Neumann/Periodic)
/// - Turbulence realizability constraints

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "poisson_solver.hpp"
#include "turbulence_model.hpp"
#include <iostream>
#include <cmath>
#include <cassert>
#include <limits>

using namespace nncfd;

// ============================================================================
// Poisson Solver Error Handling Tests
// ============================================================================

void test_poisson_limited_iterations() {
    std::cout << "Testing Poisson solver with limited iterations... ";

    Mesh mesh;
    mesh.init_uniform(32, 32, 0.0, 1.0, 0.0, 1.0);

    // Create a problem that won't converge in 1 iteration
    ScalarField rhs(mesh);
    ScalarField p(mesh, 0.0);

    // Non-trivial RHS
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double x = mesh.x(i);
            double y = mesh.y(j);
            rhs(i, j) = std::sin(M_PI * x) * std::sin(M_PI * y);
        }
    }

    PoissonSolver solver(mesh);
    solver.set_bc(PoissonBC::Dirichlet, PoissonBC::Dirichlet,
                  PoissonBC::Dirichlet, PoissonBC::Dirichlet);
    solver.set_dirichlet_value(0.0);

    PoissonConfig cfg;
    cfg.tol = 1e-12;      // Very tight tolerance
    cfg.max_iter = 1;     // Only 1 iteration - won't converge
    cfg.omega = 1.5;

    int iters = solver.solve(rhs, p, cfg);

    // Should return after max_iter iterations
    assert(iters == 1);

    // Residual should be high (not converged)
    double residual = solver.residual();
    assert(residual > 1e-6);  // Definitely not converged

    // But solution should still be finite (no NaN)
    bool all_finite = true;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            if (!std::isfinite(p(i, j))) {
                all_finite = false;
                break;
            }
        }
    }
    assert(all_finite);

    std::cout << "PASSED (iters=" << iters << ", res=" << residual << ")\n";
}

void test_poisson_singular_neumann() {
    std::cout << "Testing Poisson with pure Neumann BCs (singular)... ";

    Mesh mesh;
    mesh.init_uniform(32, 32, 0.0, 1.0, 0.0, 1.0);

    // Pure Neumann problem - singular, solution defined up to constant
    ScalarField rhs(mesh, 0.0);
    ScalarField p(mesh, 0.0);

    // RHS must have zero mean for compatibility
    // Using zero RHS satisfies this automatically

    PoissonSolver solver(mesh);
    solver.set_bc(PoissonBC::Neumann, PoissonBC::Neumann,
                  PoissonBC::Neumann, PoissonBC::Neumann);

    PoissonConfig cfg;
    cfg.tol = 1e-6;
    cfg.max_iter = 1000;
    cfg.omega = 1.5;

    int iters = solver.solve(rhs, p, cfg);

    // Should converge (mean subtraction handles singular system)
    assert(solver.residual() < 1e-4);

    // Solution should be nearly constant (since RHS=0)
    double p_mean = 0.0;
    int count = 0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            p_mean += p(i, j);
            count++;
        }
    }
    p_mean /= count;

    // Variance should be near zero
    double variance = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double diff = p(i, j) - p_mean;
            variance += diff * diff;
        }
    }
    variance /= count;

    assert(variance < 1e-8);  // Solution is nearly constant

    std::cout << "PASSED (iters=" << iters << ", variance=" << variance << ")\n";
}

void test_poisson_singular_periodic() {
    std::cout << "Testing Poisson with pure Periodic BCs (singular)... ";

    Mesh mesh;
    int N = 32;
    double L = 2.0 * M_PI;
    mesh.init_uniform(N, N, 0.0, L, 0.0, L);

    // Periodic problem with zero-mean RHS
    ScalarField rhs(mesh);
    ScalarField p(mesh, 0.0);

    // sin(x)*sin(y) has zero mean over [0,2π]²
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double x = mesh.x(i);
            double y = mesh.y(j);
            rhs(i, j) = -2.0 * std::sin(x) * std::sin(y);
        }
    }

    PoissonSolver solver(mesh);
    solver.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                  PoissonBC::Periodic, PoissonBC::Periodic);

    PoissonConfig cfg;
    cfg.tol = 1e-6;
    cfg.max_iter = 5000;
    cfg.omega = 1.7;

    int iters = solver.solve(rhs, p, cfg);

    // Should converge
    assert(solver.residual() < 1e-4);

    // Check against analytical solution (up to constant)
    double p_mean = 0.0, exact_mean = 0.0;
    int count = 0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double x = mesh.x(i);
            double y = mesh.y(j);
            p_mean += p(i, j);
            exact_mean += std::sin(x) * std::sin(y);
            count++;
        }
    }
    p_mean /= count;
    exact_mean /= count;

    double max_error = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double x = mesh.x(i);
            double y = mesh.y(j);
            double p_exact = std::sin(x) * std::sin(y);
            double error = std::abs((p(i, j) - p_mean) - (p_exact - exact_mean));
            max_error = std::max(max_error, error);
        }
    }

    assert(max_error < 0.1);

    std::cout << "PASSED (iters=" << iters << ", max_err=" << max_error << ")\n";
}

// ============================================================================
// NaN Detection Tests
// ============================================================================

void test_nan_detection_velocity() {
    std::cout << "Testing NaN detection in velocity field... ";

    Mesh mesh;
    mesh.init_uniform(16, 32, 0.0, 1.0, -0.5, 0.5);

    Config config;
    config.nu = 0.01;
    config.dt = 1e-3;
    config.turb_model = TurbulenceModelType::None;
    config.turb_guard_enabled = true;
    config.turb_guard_interval = 1;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);

    solver.initialize_uniform(1.0, 0.0);

    // Inject NaN into u-velocity
    solver.velocity().u(mesh.Nx/2, mesh.Ny/2) = std::numeric_limits<double>::quiet_NaN();

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    bool detected = false;
    try {
        solver.check_for_nan_inf(1);
    } catch (const std::runtime_error& e) {
        std::string msg(e.what());
        if (msg.find("NaN") != std::string::npos || msg.find("NUMERICAL") != std::string::npos) {
            detected = true;
        }
    }

    assert(detected);
    std::cout << "PASSED\n";
}

void test_inf_detection_pressure() {
    std::cout << "Testing Inf detection in pressure field... ";

    Mesh mesh;
    mesh.init_uniform(16, 32, 0.0, 1.0, -0.5, 0.5);

    Config config;
    config.nu = 0.01;
    config.dt = 1e-3;
    config.turb_model = TurbulenceModelType::None;
    config.turb_guard_enabled = true;
    config.turb_guard_interval = 1;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);

    solver.initialize_uniform(1.0, 0.0);

    // Inject Inf into pressure
    solver.pressure()(mesh.Nx/2, mesh.Ny/2) = std::numeric_limits<double>::infinity();

#ifdef USE_GPU_OFFLOAD
    solver.sync_to_gpu();
#endif

    bool detected = false;
    try {
        solver.check_for_nan_inf(1);
    } catch (const std::runtime_error& e) {
        std::string msg(e.what());
        if (msg.find("Inf") != std::string::npos || msg.find("NUMERICAL") != std::string::npos) {
            detected = true;
        }
    }

    assert(detected);
    std::cout << "PASSED\n";
}

// ============================================================================
// Turbulence Realizability Tests
// ============================================================================

void test_realizability_k_positive() {
    std::cout << "Testing k remains positive during simulation... ";

    Mesh mesh;
    mesh.init_uniform(32, 64, 0.0, 2.0, -1.0, 1.0);

    Config config;
    config.nu = 0.001;
    config.dt = 1e-4;
    config.turb_model = TurbulenceModelType::KOmega;
    config.turb_guard_enabled = true;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);

    auto turb_model = create_turbulence_model(TurbulenceModelType::KOmega, "", "");
    solver.set_turbulence_model(std::move(turb_model));

    solver.set_body_force(-0.001, 0.0);
    solver.initialize_uniform(0.5, 0.0);

    // Run some steps
    for (int i = 0; i < 50; ++i) {
        solver.step();
    }

    // Check k is positive everywhere
    const ScalarField& k = solver.k();
    bool all_positive = true;
    double k_min = 1e10;

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            if (k(i, j) < 0.0) {
                all_positive = false;
            }
            k_min = std::min(k_min, k(i, j));
        }
    }

    assert(all_positive);
    std::cout << "PASSED (k_min=" << k_min << ")\n";
}

void test_realizability_omega_positive() {
    std::cout << "Testing omega remains positive during simulation... ";

    Mesh mesh;
    mesh.init_uniform(32, 64, 0.0, 2.0, -1.0, 1.0);

    Config config;
    config.nu = 0.001;
    config.dt = 1e-4;
    config.turb_model = TurbulenceModelType::KOmega;
    config.turb_guard_enabled = true;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);

    auto turb_model = create_turbulence_model(TurbulenceModelType::KOmega, "", "");
    solver.set_turbulence_model(std::move(turb_model));

    solver.set_body_force(-0.001, 0.0);
    solver.initialize_uniform(0.5, 0.0);

    // Run some steps
    for (int i = 0; i < 50; ++i) {
        solver.step();
    }

    // Check omega is positive everywhere
    const ScalarField& omega = solver.omega();
    bool all_positive = true;
    double omega_min = 1e10;

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            if (omega(i, j) < 0.0) {
                all_positive = false;
            }
            omega_min = std::min(omega_min, omega(i, j));
        }
    }

    assert(all_positive);
    std::cout << "PASSED (omega_min=" << omega_min << ")\n";
}

void test_nu_t_bounded() {
    std::cout << "Testing nu_t is bounded by nu_t_max... ";

    Mesh mesh;
    mesh.init_uniform(32, 64, 0.0, 2.0, -1.0, 1.0);

    Config config;
    config.nu = 0.001;
    config.dt = 1e-4;
    config.turb_model = TurbulenceModelType::SSTKOmega;
    config.nu_t_max = 0.5;  // Set explicit bound
    config.turb_guard_enabled = true;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);

    auto turb_model = create_turbulence_model(TurbulenceModelType::SSTKOmega, "", "");
    solver.set_turbulence_model(std::move(turb_model));

    solver.set_body_force(-0.001, 0.0);
    solver.initialize_uniform(0.5, 0.0);

    // Run some steps
    for (int i = 0; i < 50; ++i) {
        solver.step();
    }

    // Check nu_t is bounded
    const ScalarField& nu_t = solver.nu_t();
    bool all_bounded = true;
    double nu_t_max_actual = 0.0;

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            if (nu_t(i, j) > config.nu_t_max * 1.01) {  // 1% tolerance
                all_bounded = false;
            }
            nu_t_max_actual = std::max(nu_t_max_actual, nu_t(i, j));
        }
    }

    assert(all_bounded);
    std::cout << "PASSED (nu_t_max=" << nu_t_max_actual << ")\n";
}

// ============================================================================
// Edge Case Tests
// ============================================================================

void test_zero_velocity_field() {
    std::cout << "Testing solver with zero initial velocity... ";

    Mesh mesh;
    mesh.init_uniform(32, 32, 0.0, 1.0, 0.0, 1.0);

    Config config;
    config.nu = 0.01;
    config.dt = 0.001;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);

    // Initialize with zero velocity
    solver.initialize_uniform(0.0, 0.0);

    // Should not crash
    for (int i = 0; i < 10; ++i) {
        solver.step();
    }

    // Velocity should remain zero (no forcing)
    const VectorField& vel = solver.velocity();
    double max_vel = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            max_vel = std::max(max_vel, std::abs(vel.u(i, j)));
            max_vel = std::max(max_vel, std::abs(vel.v(i, j)));
        }
    }

    assert(max_vel < 1e-10);
    std::cout << "PASSED (max_vel=" << max_vel << ")\n";
}

void test_very_small_dt() {
    std::cout << "Testing solver with very small dt... ";

    Mesh mesh;
    mesh.init_uniform(16, 16, 0.0, 1.0, 0.0, 1.0);

    Config config;
    config.nu = 0.01;
    config.dt = 1e-8;  // Very small
    config.adaptive_dt = false;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);

    solver.set_body_force(-0.001, 0.0);
    solver.initialize_uniform(1.0, 0.0);

    // Should not crash or produce NaN
    for (int i = 0; i < 10; ++i) {
        solver.step();
    }

    const VectorField& vel = solver.velocity();
    bool all_finite = true;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            if (!std::isfinite(vel.u(i, j)) || !std::isfinite(vel.v(i, j))) {
                all_finite = false;
            }
        }
    }

    assert(all_finite);
    std::cout << "PASSED\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== Error Recovery Tests ===\n\n";

    // Poisson solver tests
    test_poisson_limited_iterations();
    test_poisson_singular_neumann();
    test_poisson_singular_periodic();

    // NaN detection tests
    test_nan_detection_velocity();
    test_inf_detection_pressure();

    // Turbulence realizability tests
    test_realizability_k_positive();
    test_realizability_omega_positive();
    test_nu_t_bounded();

    // Edge case tests
    test_zero_velocity_field();
    test_very_small_dt();

    std::cout << "\nAll tests PASSED!\n";
    return 0;
}
