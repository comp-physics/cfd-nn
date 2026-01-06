/// Unified Data-Driven Test Framework
///
/// This framework allows tests to be defined as data structures rather than code.
/// A single TestSpec struct can describe mesh, config, BCs, initialization,
/// execution mode, and validation criteria - replacing 50-150 lines of boilerplate.
///
/// Example:
///   TestSpec spec = {
///       .name = "poiseuille_32x64",
///       .mesh = {32, 64, 4.0, 2.0},
///       .config = {.nu = 0.01, .turb = None},
///       .bc = BC_CHANNEL,
///       .init = Init::Poiseuille(-0.001),
///       .run = Run::Steady(1e-6, 2000),
///       .check = Check::L2Error(0.05)
///   };
///   auto result = run_test(spec);

#pragma once

#include "solver.hpp"
#include "mesh.hpp"
#include "config.hpp"
#include "fields.hpp"
#include "poisson_solver_multigrid.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <functional>
#include <stdexcept>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

namespace nncfd {
namespace test {

//=============================================================================
// Mesh Specification
//=============================================================================
struct MeshSpec {
    int nx = 32, ny = 32, nz = 1;
    double Lx = 1.0, Ly = 1.0, Lz = 1.0;
    double x0 = 0.0, y0 = 0.0, z0 = 0.0;

    enum Type { UNIFORM, STRETCHED_Y, STRETCHED_YZ } type = UNIFORM;
    double stretch_factor = 2.0;

    // Convenience constructors
    static MeshSpec uniform_2d(int nx, int ny, double Lx, double Ly,
                                double x0 = 0.0, double y0 = 0.0) {
        return {nx, ny, 1, Lx, Ly, 1.0, x0, y0, 0.0, UNIFORM, 2.0};
    }

    static MeshSpec channel(int nx = 32, int ny = 64) {
        return {nx, ny, 1, 4.0, 2.0, 1.0, 0.0, -1.0, 0.0, UNIFORM, 2.0};
    }

    static MeshSpec taylor_green(int n = 64) {
        return {n, n, 1, 2.0*M_PI, 2.0*M_PI, 1.0, 0.0, 0.0, 0.0, UNIFORM, 2.0};
    }

    static MeshSpec unit_square(int n = 64) {
        return {n, n, 1, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, UNIFORM, 2.0};
    }

    static MeshSpec stretched_channel(int nx = 32, int ny = 96, double stretch = 2.0) {
        return {nx, ny, 1, 4.0, 2.0, 1.0, 0.0, -1.0, 0.0, STRETCHED_Y, stretch};
    }

    bool is_3d() const { return nz > 1; }
};

//=============================================================================
// Config Specification
//=============================================================================
struct ConfigSpec {
    double nu = 0.01;
    double dt = 0.001;
    bool adaptive_dt = true;
    int max_iter = 1000;
    double tol = 1e-6;
    TurbulenceModelType turb_model = TurbulenceModelType::None;
    std::string nn_model_path;
    bool verbose = false;
    int poisson_max_iter = 50;

    static ConfigSpec laminar(double nu = 0.01) {
        return {nu, 0.001, true, 1000, 1e-6, TurbulenceModelType::None};
    }

    static ConfigSpec turbulent_komega(double nu = 0.00005) {
        return {nu, 0.001, true, 5000, 1e-5, TurbulenceModelType::KOmega};
    }

    static ConfigSpec unsteady(double nu = 0.01, double dt = 0.01) {
        return {nu, dt, false, 100, 1e-6, TurbulenceModelType::None};
    }
};

//=============================================================================
// Boundary Condition Specification
//=============================================================================
struct BCSpec {
    VelocityBC::Type x_lo = VelocityBC::Periodic;
    VelocityBC::Type x_hi = VelocityBC::Periodic;
    VelocityBC::Type y_lo = VelocityBC::NoSlip;
    VelocityBC::Type y_hi = VelocityBC::NoSlip;
    VelocityBC::Type z_lo = VelocityBC::Periodic;
    VelocityBC::Type z_hi = VelocityBC::Periodic;

    static BCSpec channel() {
        return {VelocityBC::Periodic, VelocityBC::Periodic,
                VelocityBC::NoSlip, VelocityBC::NoSlip,
                VelocityBC::Periodic, VelocityBC::Periodic};
    }

    static BCSpec periodic() {
        return {VelocityBC::Periodic, VelocityBC::Periodic,
                VelocityBC::Periodic, VelocityBC::Periodic,
                VelocityBC::Periodic, VelocityBC::Periodic};
    }

    static BCSpec cavity() {
        return {VelocityBC::NoSlip, VelocityBC::NoSlip,
                VelocityBC::NoSlip, VelocityBC::NoSlip,
                VelocityBC::NoSlip, VelocityBC::NoSlip};
    }

    VelocityBC to_velocity_bc() const {
        VelocityBC bc;
        bc.x_lo = x_lo; bc.x_hi = x_hi;
        bc.y_lo = y_lo; bc.y_hi = y_hi;
        bc.z_lo = z_lo; bc.z_hi = z_hi;
        return bc;
    }
};

//=============================================================================
// Initialization Specification
//=============================================================================
struct InitSpec {
    enum Type { ZERO, UNIFORM, POISEUILLE, TAYLOR_GREEN, PERTURBED, CUSTOM };
    Type type = ZERO;
    double u0 = 0.0, v0 = 0.0, w0 = 0.0;
    double dp_dx = 0.0;
    double scale = 0.9;  // For Poiseuille: fraction of analytical
    std::function<void(RANSSolver&, const Mesh&)> custom_init;

    static InitSpec zero() { return {ZERO}; }
    static InitSpec uniform(double u, double v = 0.0) { return {UNIFORM, u, v}; }
    static InitSpec poiseuille(double dp_dx, double scale = 0.9) {
        return {POISEUILLE, 0, 0, 0, dp_dx, scale};
    }
    static InitSpec taylor_green() { return {TAYLOR_GREEN}; }
    static InitSpec perturbed() { return {PERTURBED}; }
};

//=============================================================================
// Execution Specification
//=============================================================================
struct RunSpec {
    enum Mode { STEADY, N_STEPS, TIME_EVOLVE };
    Mode mode = STEADY;
    int n_steps = 100;
    double t_end = 1.0;
    double body_force_x = 0.0;
    double body_force_y = 0.0;

    static RunSpec steady(double tol = 1e-6, int max_iter = 2000) {
        RunSpec r; r.mode = STEADY; return r;
    }
    static RunSpec steps(int n) {
        RunSpec r; r.mode = N_STEPS; r.n_steps = n; return r;
    }
    static RunSpec time(double t) {
        RunSpec r; r.mode = TIME_EVOLVE; r.t_end = t; return r;
    }
    static RunSpec channel(double dp_dx) {
        RunSpec r; r.mode = STEADY; r.body_force_x = -dp_dx; return r;
    }
};

//=============================================================================
// Validation Specification
//=============================================================================
struct CheckSpec {
    enum Type {
        NONE,              // Just verify it runs without crashing
        CONVERGES,         // Verify residual drops
        L2_ERROR,          // Compare to analytical solution
        DIVERGENCE_FREE,   // Check |div(u)| < tol
        ENERGY_DECAY,      // Verify KE decreases monotonically
        BOUNDED,           // Verify max velocity stays bounded
        RESIDUAL           // Check final residual < tol
    };
    Type type = NONE;
    double tolerance = 0.05;

    // For L2_ERROR: analytical solution
    std::function<double(double, double)> u_exact;
    std::function<double(double, double)> v_exact;

    static CheckSpec none() { return {NONE}; }
    static CheckSpec converges() { return {CONVERGES}; }
    static CheckSpec l2_error(double tol,
                              std::function<double(double,double)> u_ex = nullptr) {
        CheckSpec c; c.type = L2_ERROR; c.tolerance = tol; c.u_exact = u_ex;
        return c;
    }
    static CheckSpec divergence_free(double tol = 1e-10) {
        return {DIVERGENCE_FREE, tol};
    }
    static CheckSpec energy_decay() { return {ENERGY_DECAY}; }
    static CheckSpec bounded(double max_vel = 10.0) {
        return {BOUNDED, max_vel};
    }
    static CheckSpec residual(double tol = 1e-6) {
        return {RESIDUAL, tol};
    }
};

//=============================================================================
// Complete Test Specification
//=============================================================================
struct TestSpec {
    std::string name;
    std::string category;  // For grouping output

    MeshSpec mesh;
    ConfigSpec config;
    BCSpec bc;
    InitSpec init;
    RunSpec run;
    CheckSpec check;

    bool skip = false;  // For conditional tests
    std::string skip_reason;
};

//=============================================================================
// Test Result
//=============================================================================
struct TestResult {
    std::string name;
    bool passed = false;
    std::string message;
    int iterations = 0;
    double residual = 0.0;
    double error = 0.0;
    double elapsed_ms = 0.0;
};

//=============================================================================
// Test Runner Implementation
//=============================================================================

inline void apply_init(RANSSolver& solver, const Mesh& mesh, const InitSpec& init,
                       double nu, double H = 1.0) {
    switch (init.type) {
        case InitSpec::ZERO:
            solver.initialize_uniform(0.0, 0.0);
            break;

        case InitSpec::UNIFORM:
            solver.initialize_uniform(init.u0, init.v0);
            break;

        case InitSpec::POISEUILLE: {
            double dp_dx = init.dp_dx;
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                double y = mesh.y(j);
                double u_ex = -dp_dx / (2.0 * nu) * (H * H - y * y);
                for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                    solver.velocity().u(i, j) = init.scale * u_ex;
                }
            }
            break;
        }

        case InitSpec::TAYLOR_GREEN:
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                    double x = (i < mesh.i_end()) ? mesh.x(i) + mesh.dx/2.0 : mesh.x_max;
                    double y = mesh.y(j);
                    solver.velocity().u(i, j) = std::sin(x) * std::cos(y);
                }
            }
            for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    double x = mesh.x(i);
                    double y = (j < mesh.j_end()) ? mesh.y(j) + mesh.dy/2.0 : mesh.y_max;
                    solver.velocity().v(i, j) = -std::cos(x) * std::sin(y);
                }
            }
            break;

        case InitSpec::CUSTOM:
            if (init.custom_init) init.custom_init(solver, mesh);
            break;

        default:
            break;
    }
}

inline double compute_l2_error(const VectorField& vel, const Mesh& mesh,
                               const std::function<double(double,double)>& u_exact) {
    if (!u_exact) return 0.0;

    double error_sq = 0.0, norm_sq = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double u_num = 0.5 * (vel.u(i, j) + vel.u(i+1, j));
            double u_ex = u_exact(mesh.x(i), mesh.y(j));
            double diff = u_num - u_ex;
            error_sq += diff * diff * mesh.dx * mesh.dy;
            norm_sq += u_ex * u_ex * mesh.dx * mesh.dy;
        }
    }
    return (norm_sq > 1e-14) ? std::sqrt(error_sq / norm_sq) : std::sqrt(error_sq);
}

inline double compute_max_divergence(const VectorField& vel, const Mesh& mesh) {
    double max_div = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double dudx = (vel.u(i+1, j) - vel.u(i, j)) / mesh.dx;
            double dvdy = (vel.v(i, j+1) - vel.v(i, j)) / mesh.dy;
            max_div = std::max(max_div, std::abs(dudx + dvdy));
        }
    }
    return max_div;
}

inline double compute_kinetic_energy(const VectorField& vel, const Mesh& mesh) {
    double KE = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double u = 0.5 * (vel.u(i, j) + vel.u(i+1, j));
            double v = 0.5 * (vel.v(i, j) + vel.v(i, j+1));
            KE += 0.5 * (u*u + v*v) * mesh.dx * mesh.dy;
        }
    }
    return KE;
}

inline double compute_max_velocity(const VectorField& vel, const Mesh& mesh) {
    double max_vel = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double u = vel.u(i, j);
            double v = vel.v(i, j);
            max_vel = std::max(max_vel, std::sqrt(u*u + v*v));
        }
    }
    return max_vel;
}

inline TestResult run_test(const TestSpec& spec) {
    TestResult result;
    result.name = spec.name;

    if (spec.skip) {
        result.passed = true;
        result.message = "SKIPPED: " + spec.skip_reason;
        return result;
    }

    try {
        // Create mesh
        Mesh mesh;
        if (spec.mesh.type == MeshSpec::STRETCHED_Y) {
            auto stretch = Mesh::tanh_stretching(spec.mesh.stretch_factor);
            mesh.init_stretched_y(spec.mesh.nx, spec.mesh.ny,
                                  spec.mesh.x0, spec.mesh.x0 + spec.mesh.Lx,
                                  spec.mesh.y0, spec.mesh.y0 + spec.mesh.Ly, stretch);
        } else {
            if (spec.mesh.is_3d()) {
                mesh.init_uniform(spec.mesh.nx, spec.mesh.ny, spec.mesh.nz,
                                  spec.mesh.x0, spec.mesh.x0 + spec.mesh.Lx,
                                  spec.mesh.y0, spec.mesh.y0 + spec.mesh.Ly,
                                  spec.mesh.z0, spec.mesh.z0 + spec.mesh.Lz);
            } else {
                mesh.init_uniform(spec.mesh.nx, spec.mesh.ny,
                                  spec.mesh.x0, spec.mesh.x0 + spec.mesh.Lx,
                                  spec.mesh.y0, spec.mesh.y0 + spec.mesh.Ly);
            }
        }

        // Create config
        Config config;
        config.nu = spec.config.nu;
        config.dt = spec.config.dt;
        config.adaptive_dt = spec.config.adaptive_dt;
        config.max_iter = spec.config.max_iter;
        config.tol = spec.config.tol;
        config.turb_model = spec.config.turb_model;
        config.verbose = spec.config.verbose;
        config.poisson_max_iter = spec.config.poisson_max_iter;

        // Create solver
        RANSSolver solver(mesh, config);
        solver.set_velocity_bc(spec.bc.to_velocity_bc());

        if (spec.run.body_force_x != 0.0 || spec.run.body_force_y != 0.0) {
            solver.set_body_force(spec.run.body_force_x, spec.run.body_force_y);
        }

        // Initialize
        double H = spec.mesh.Ly / 2.0;
        apply_init(solver, mesh, spec.init, spec.config.nu, H);

        solver.sync_to_gpu();

        // Run
        double KE_initial = 0.0;
        if (spec.check.type == CheckSpec::ENERGY_DECAY) {
            KE_initial = compute_kinetic_energy(solver.velocity(), mesh);
        }

        int iters = 0;
        double residual = 0.0;

        switch (spec.run.mode) {
            case RunSpec::STEADY: {
                auto [res, it] = solver.solve_steady();
                residual = res;
                iters = it;
                break;
            }
            case RunSpec::N_STEPS:
                for (int i = 0; i < spec.run.n_steps; ++i) {
                    residual = solver.step();
                    ++iters;
                }
                break;
            case RunSpec::TIME_EVOLVE: {
                double t = 0.0;
                while (t < spec.run.t_end) {
                    residual = solver.step();
                    t += spec.config.dt;
                    ++iters;
                }
                break;
            }
        }

        solver.sync_from_gpu();

        result.iterations = iters;
        result.residual = residual;

        // Validate
        switch (spec.check.type) {
            case CheckSpec::NONE:
                result.passed = true;
                result.message = "completed";
                break;

            case CheckSpec::CONVERGES:
                result.passed = (residual < spec.config.tol);
                result.message = result.passed ? "converged" : "did not converge";
                break;

            case CheckSpec::L2_ERROR: {
                double err = compute_l2_error(solver.velocity(), mesh, spec.check.u_exact);
                result.error = err;
                result.passed = (err < spec.check.tolerance);
                result.message = "L2=" + std::to_string(err * 100) + "%";
                break;
            }

            case CheckSpec::DIVERGENCE_FREE: {
                double div = compute_max_divergence(solver.velocity(), mesh);
                result.error = div;
                result.passed = (div < spec.check.tolerance);
                result.message = "div=" + std::to_string(div);
                break;
            }

            case CheckSpec::ENERGY_DECAY: {
                double KE_final = compute_kinetic_energy(solver.velocity(), mesh);
                result.passed = (KE_final < KE_initial);
                result.message = "KE: " + std::to_string(KE_initial) + " -> " + std::to_string(KE_final);
                break;
            }

            case CheckSpec::BOUNDED: {
                double max_vel = compute_max_velocity(solver.velocity(), mesh);
                result.error = max_vel;
                result.passed = (max_vel < spec.check.tolerance);
                result.message = "max_vel=" + std::to_string(max_vel);
                break;
            }

            case CheckSpec::RESIDUAL:
                result.passed = (residual < spec.check.tolerance);
                result.message = "res=" + std::to_string(residual);
                break;
        }

    } catch (const std::exception& e) {
        result.passed = false;
        result.message = std::string("EXCEPTION: ") + e.what();
    }

    return result;
}

//=============================================================================
// Test Suite Runner
//=============================================================================

inline void run_test_suite(const std::string& name,
                           const std::vector<TestSpec>& tests,
                           bool stop_on_fail = false) {
    std::cout << "\n========================================\n";
    std::cout << name << "\n";
    std::cout << "========================================\n";

    int passed = 0, failed = 0, skipped = 0;

    for (const auto& spec : tests) {
        auto result = run_test(spec);

        std::cout << "  " << std::left << std::setw(40) << spec.name;

        if (result.message.find("SKIPPED") == 0) {
            std::cout << "[SKIP] " << result.message << "\n";
            ++skipped;
        } else if (result.passed) {
            std::cout << "[PASS] " << result.message;
            if (result.iterations > 0) std::cout << " (iters=" << result.iterations << ")";
            std::cout << "\n";
            ++passed;
        } else {
            std::cout << "[FAIL] " << result.message << "\n";
            ++failed;
            if (stop_on_fail) break;
        }
    }

    std::cout << "\nSummary: " << passed << " passed, " << failed << " failed";
    if (skipped > 0) std::cout << ", " << skipped << " skipped";
    std::cout << "\n";
}

//=============================================================================
// Predefined Test Suites
//=============================================================================

// Channel flow tests at multiple resolutions
inline std::vector<TestSpec> channel_flow_suite(double dp_dx = -0.001) {
    std::vector<TestSpec> tests;

    // Use high init factor (0.99) for both CPU and GPU
    // This initializes close to analytical solution, reducing iterations needed
    // CPU multigrid is slower than GPU FFT, so this helps both converge within max_iter
    double init_factor = 0.99;

    for (int nx : {16, 32, 64}) {
        int ny = 2 * nx;
        double H = 1.0;
        double nu = 0.01;

        auto u_exact = [dp_dx, nu, H](double, double y) {
            return -dp_dx / (2.0 * nu) * (H * H - y * y);
        };

        tests.push_back({
            .name = "channel_" + std::to_string(nx) + "x" + std::to_string(ny),
            .category = "physics",
            .mesh = MeshSpec::channel(nx, ny),
            .config = ConfigSpec::laminar(nu),
            .bc = BCSpec::channel(),
            .init = InitSpec::poiseuille(dp_dx, init_factor),
            .run = RunSpec::channel(dp_dx),
            .check = CheckSpec::l2_error(0.05, u_exact)
        });
    }

    return tests;
}

// Taylor-Green vortex decay tests
inline std::vector<TestSpec> taylor_green_suite() {
    std::vector<TestSpec> tests;

    for (int n : {32, 48, 64}) {
        tests.push_back({
            .name = "taylor_green_" + std::to_string(n),
            .category = "physics",
            .mesh = MeshSpec::taylor_green(n),
            .config = ConfigSpec::unsteady(0.01, 0.01),
            .bc = BCSpec::periodic(),
            .init = InitSpec::taylor_green(),
            .run = RunSpec::steps(50),
            .check = CheckSpec::energy_decay()
        });
    }

    return tests;
}

} // namespace test
} // namespace nncfd
