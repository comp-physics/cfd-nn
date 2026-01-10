/// Unit tests for solver error handling and recovery

#include "test_harness.hpp"
#include "test_utilities.hpp"
#include "poisson_solver.hpp"
#include "turbulence_model.hpp"
#include <cmath>
#include <stdexcept>
#include <limits>

using namespace nncfd;
using nncfd::test::harness::record;
using nncfd::test::BCPattern;
using nncfd::test::create_velocity_bc;
using nncfd::test::make_test_solver_domain;

// ============================================================================
// Poisson Solver Error Handling Tests
// ============================================================================

bool test_poisson_limited_iterations() {
    Mesh mesh;
    mesh.init_uniform(32, 32, 0.0, 1.0, 0.0, 1.0);

    ScalarField rhs(mesh), p(mesh, 0.0);
    FOR_INTERIOR_2D(mesh, i, j) {
        rhs(i, j) = std::sin(M_PI * mesh.x(i)) * std::sin(M_PI * mesh.y(j));
    }

    PoissonSolver solver(mesh);
    solver.set_bc(PoissonBC::Dirichlet, PoissonBC::Dirichlet,
                  PoissonBC::Dirichlet, PoissonBC::Dirichlet);
    solver.set_dirichlet_value(0.0);

    PoissonConfig cfg{1e-12, 1, 1.5};  // Very tight tol, 1 iteration
    int iters = solver.solve(rhs, p, cfg);

    if (iters > cfg.max_steps + 1) return false;
    if (solver.residual() <= 1e-6) return false;

    FOR_INTERIOR_2D(mesh, i, j) {
        if (!std::isfinite(p(i, j))) return false;
    }
    return true;
}

bool test_poisson_singular_neumann() {
    Mesh mesh;
    mesh.init_uniform(32, 32, 0.0, 1.0, 0.0, 1.0);

    ScalarField rhs(mesh, 0.0), p(mesh, 0.0);

    PoissonSolver solver(mesh);
    solver.set_bc(PoissonBC::Neumann, PoissonBC::Neumann,
                  PoissonBC::Neumann, PoissonBC::Neumann);

    PoissonConfig cfg{1e-6, 1000, 1.5};
    solver.solve(rhs, p, cfg);

    if (solver.residual() >= 1e-4) return false;

    double p_mean = nncfd::test::compute_mean(p, mesh);
    double variance = 0.0;
    int count = 0;
    FOR_INTERIOR_2D(mesh, i, j) {
        double diff = p(i, j) - p_mean;
        variance += diff * diff;
        count++;
    }
    return (variance / count) < 1e-8;
}

bool test_poisson_singular_periodic() {
    int N = 32;
    double L = 2.0 * M_PI;
    Mesh mesh;
    mesh.init_uniform(N, N, 0.0, L, 0.0, L);

    ScalarField rhs(mesh), p(mesh, 0.0);
    FOR_INTERIOR_2D(mesh, i, j) {
        rhs(i, j) = -2.0 * std::sin(mesh.x(i)) * std::sin(mesh.y(j));
    }

    PoissonSolver solver(mesh);
    solver.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                  PoissonBC::Periodic, PoissonBC::Periodic);

    PoissonConfig cfg{1e-6, 5000, 1.7};
    solver.solve(rhs, p, cfg);

    if (solver.residual() >= 1e-4) return false;

    // Check against analytical solution (up to constant)
    double p_mean = nncfd::test::compute_mean(p, mesh);
    double exact_mean = 0.0, count = 0, err_sq = 0.0;
    FOR_INTERIOR_2D(mesh, i, j) {
        exact_mean += std::sin(mesh.x(i)) * std::sin(mesh.y(j));
        ++count;
    }
    exact_mean /= count;
    FOR_INTERIOR_2D(mesh, i, j) {
        double diff = (p(i, j) - p_mean) - (std::sin(mesh.x(i)) * std::sin(mesh.y(j)) - exact_mean);
        err_sq += diff * diff;
    }
    return std::sqrt(err_sq / count) < 0.1;
}

// ============================================================================
// NaN Detection Tests
// ============================================================================

bool test_nan_detection_velocity() {
    nncfd::test::TestSolver ts;
    ts.mesh.init_uniform(16, 32, 0.0, 1.0, -0.5, 0.5);
    ts.config.nu = 0.01;
    ts.config.dt = 0.001;
    ts.config.turb_model = TurbulenceModelType::None;
    ts.config.turb_guard_enabled = true;
    ts.config.turb_guard_interval = 1;
    ts.config.verbose = false;
    ts.solver = std::make_unique<RANSSolver>(ts.mesh, ts.config);
    ts.solver->set_velocity_bc(create_velocity_bc(BCPattern::Channel2D));
    ts->initialize_uniform(1.0, 0.0);

    // Inject NaN
    ts->velocity().u(ts.mesh.Nx/2, ts.mesh.Ny/2) = std::numeric_limits<double>::quiet_NaN();
    ts->sync_to_gpu();

    try {
        ts->check_for_nan_inf(1);
    } catch (const std::runtime_error& e) {
        std::string msg(e.what());
        return msg.find("NaN") != std::string::npos || msg.find("NUMERICAL") != std::string::npos;
    }
    return false;
}

bool test_inf_detection_pressure() {
    nncfd::test::TestSolver ts;
    ts.mesh.init_uniform(16, 32, 0.0, 1.0, -0.5, 0.5);
    ts.config.nu = 0.01;
    ts.config.dt = 0.001;
    ts.config.turb_model = TurbulenceModelType::None;
    ts.config.turb_guard_enabled = true;
    ts.config.turb_guard_interval = 1;
    ts.config.verbose = false;
    ts.solver = std::make_unique<RANSSolver>(ts.mesh, ts.config);
    ts.solver->set_velocity_bc(create_velocity_bc(BCPattern::Channel2D));
    ts->initialize_uniform(1.0, 0.0);

    // Inject Inf
    ts->pressure()(ts.mesh.Nx/2, ts.mesh.Ny/2) = std::numeric_limits<double>::infinity();
    ts->sync_to_gpu();

    try {
        ts->check_for_nan_inf(1);
    } catch (const std::runtime_error& e) {
        std::string msg(e.what());
        return msg.find("Inf") != std::string::npos || msg.find("NUMERICAL") != std::string::npos;
    }
    return false;
}

// ============================================================================
// Turbulence Realizability Tests
// ============================================================================

static nncfd::test::TestSolver make_turb_solver(TurbulenceModelType model, double nu_t_max = -1.0) {
    nncfd::test::TestSolver ts;
    ts.mesh.init_uniform(32, 64, 0.0, 2.0, -1.0, 1.0);
    ts.config.nu = 0.001;
    ts.config.dt = 1e-4;
    ts.config.turb_model = model;
    ts.config.turb_guard_enabled = true;
    ts.config.verbose = false;
    if (nu_t_max > 0.0) ts.config.nu_t_max = nu_t_max;
    ts.solver = std::make_unique<RANSSolver>(ts.mesh, ts.config);
    ts.solver->set_velocity_bc(create_velocity_bc(BCPattern::Channel2D));
    auto turb = create_turbulence_model(model, "", "");
    ts.solver->set_turbulence_model(std::move(turb));
    ts.solver->set_body_force(-0.001, 0.0);
    ts.solver->initialize_uniform(0.5, 0.0);
    ts.solver->sync_to_gpu();
    return ts;
}

bool test_realizability_k_positive() {
    auto ts = make_turb_solver(TurbulenceModelType::KOmega);
    for (int i = 0; i < 50; ++i) ts->step();
    ts->sync_from_gpu();

    const ScalarField& k = ts->k();
    FOR_INTERIOR_2D(ts.mesh, i, j) {
        if (k(i, j) < 0.0) return false;
    }
    return true;
}

bool test_realizability_omega_positive() {
    auto ts = make_turb_solver(TurbulenceModelType::KOmega);
    for (int i = 0; i < 50; ++i) ts->step();
    ts->sync_from_gpu();

    const ScalarField& omega = ts->omega();
    FOR_INTERIOR_2D(ts.mesh, i, j) {
        if (omega(i, j) < 0.0) return false;
    }
    return true;
}

bool test_nu_t_bounded() {
    const double nu_t_max = 0.5;
    auto ts = make_turb_solver(TurbulenceModelType::SSTKOmega, nu_t_max);

    for (int i = 0; i < 50; ++i) ts->step();
    ts->sync_from_gpu();

    const ScalarField& nu_t = ts->nu_t();
    FOR_INTERIOR_2D(ts.mesh, i, j) {
        if (nu_t(i, j) > nu_t_max * 1.01) return false;
    }
    return true;
}

// ============================================================================
// Edge Case Tests
// ============================================================================

bool test_zero_velocity_field() {
    auto ts = make_test_solver_domain(32, 32, 0.0, 1.0, 0.0, 1.0);

    for (int i = 0; i < 10; ++i) ts->step();
    ts->sync_from_gpu();

    const VectorField& vel = ts->velocity();
    double max_vel = 0.0;
    FOR_INTERIOR_2D(ts.mesh, i, j) {
        max_vel = std::max(max_vel, std::abs(vel.u(i, j)));
        max_vel = std::max(max_vel, std::abs(vel.v(i, j)));
    }
    return max_vel < 1e-10;
}

bool test_very_small_dt() {
    nncfd::test::TestSolver ts;
    ts.mesh.init_uniform(16, 16, 0.0, 1.0, 0.0, 1.0);
    ts.config.nu = 0.01;
    ts.config.dt = 1e-8;  // Very small
    ts.config.adaptive_dt = false;
    ts.config.turb_model = TurbulenceModelType::None;
    ts.config.verbose = false;
    ts.solver = std::make_unique<RANSSolver>(ts.mesh, ts.config);
    ts.solver->set_velocity_bc(create_velocity_bc(BCPattern::Channel2D));
    ts.solver->set_body_force(-0.001, 0.0);
    ts.solver->initialize_uniform(1.0, 0.0);
    ts.solver->sync_to_gpu();

    for (int i = 0; i < 10; ++i) ts->step();
    ts->sync_from_gpu();

    const VectorField& vel = ts->velocity();
    FOR_INTERIOR_2D(ts.mesh, i, j) {
        if (!std::isfinite(vel.u(i, j)) || !std::isfinite(vel.v(i, j))) return false;
    }
    return true;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    return nncfd::test::harness::run("Error Recovery Tests", [] {
        record("Poisson limited iterations", test_poisson_limited_iterations());
        record("Poisson singular Neumann", test_poisson_singular_neumann());
        record("Poisson singular Periodic", test_poisson_singular_periodic());
        record("NaN detection velocity", test_nan_detection_velocity());
        record("Inf detection pressure", test_inf_detection_pressure());
        record("Realizability k positive", test_realizability_k_positive());
        record("Realizability omega positive", test_realizability_omega_positive());
        record("nu_t bounded", test_nu_t_bounded());
        record("Zero velocity field", test_zero_velocity_field());
        record("Very small dt", test_very_small_dt());
    });
}
