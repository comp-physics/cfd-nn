/// @file test_framework.hpp
/// @brief Unified testing framework for NNCFD
///
/// This framework dramatically reduces test code by providing:
/// 1. Pre-configured mesh/solver/BC presets
/// 2. Manufactured solutions with analytical RHS
/// 3. Reusable test runners for common patterns
/// 4. Standardized result types and assertions
///
/// A typical test file goes from 400+ lines to 50-100 lines.

#pragma once

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "config.hpp"
#include "poisson_solver.hpp"
#include "poisson_solver_multigrid.hpp"
#include "test_fixtures.hpp"  // Include manufactured solutions
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <functional>
#include <stdexcept>
#include <string>

namespace nncfd {
namespace test {

//=============================================================================
// Configuration Presets
//=============================================================================

/// Mesh configuration preset
struct MeshPreset {
    int nx, ny, nz;
    double x_min, x_max, y_min, y_max, z_min, z_max;

    Mesh create() const {
        Mesh m;
        if (nz <= 1) {
            m.init_uniform(nx, ny, x_min, x_max, y_min, y_max);
        } else {
            m.init_uniform(nx, ny, nz, x_min, x_max, y_min, y_max, z_min, z_max);
        }
        return m;
    }

    bool is_3d() const { return nz > 1; }
};

/// Common mesh presets
namespace meshes {
    inline MeshPreset periodic_2d(int n, double L = 2*M_PI) {
        return {n, n, 1, 0, L, 0, L, 0, 0};
    }
    inline MeshPreset channel_2d(int nx = 32, int ny = 64) {
        return {nx, ny, 1, 0, 4, 0, 1, 0, 0};
    }
    inline MeshPreset periodic_3d(int n, double L = 2*M_PI) {
        return {n, n, n, 0, L, 0, L, 0, L};
    }
    inline MeshPreset channel_3d(int nx = 16, int ny = 32, int nz = 8) {
        return {nx, ny, nz, 0, 4, 0, 1, 0, 2};
    }
    inline MeshPreset duct_3d(int nx = 16, int ny = 32, int nz = 32) {
        return {nx, ny, nz, 0, 4, 0, 1, 0, 1};
    }
}

/// Solver configuration
struct SolverPreset {
    double nu = 0.01;
    double dt = 0.01;
    int max_iter = 1000;
    double tol = 1e-6;
    bool adaptive_dt = false;
    TurbulenceModelType turb = TurbulenceModelType::None;

    Config to_config() const {
        Config c;
        c.nu = nu;
        c.dt = dt;
        c.max_iter = max_iter;
        c.tol = tol;
        c.adaptive_dt = adaptive_dt;
        c.turb_model = turb;
        c.verbose = false;
        return c;
    }
};

/// Common solver presets
namespace solvers {
    inline SolverPreset laminar(double nu = 0.01) {
        return {nu, 0.01, 2000, 1e-6, false, TurbulenceModelType::None};
    }
    inline SolverPreset fast_laminar(double nu = 0.01) {
        return {nu, 0.01, 500, 1e-5, false, TurbulenceModelType::None};
    }
    inline SolverPreset turbulent_komega() {
        return {0.001, 0.001, 5000, 1e-6, true, TurbulenceModelType::KOmega};
    }
}

/// Boundary condition configuration
struct BCPreset {
    VelocityBC::Type x_lo = VelocityBC::Periodic;
    VelocityBC::Type x_hi = VelocityBC::Periodic;
    VelocityBC::Type y_lo = VelocityBC::Periodic;
    VelocityBC::Type y_hi = VelocityBC::Periodic;
    VelocityBC::Type z_lo = VelocityBC::Periodic;
    VelocityBC::Type z_hi = VelocityBC::Periodic;

    VelocityBC to_velocity_bc() const {
        VelocityBC bc;
        bc.x_lo = x_lo; bc.x_hi = x_hi;
        bc.y_lo = y_lo; bc.y_hi = y_hi;
        bc.z_lo = z_lo; bc.z_hi = z_hi;
        return bc;
    }
};

/// Common BC presets
namespace bcs {
    inline BCPreset periodic_2d() {
        return {VelocityBC::Periodic, VelocityBC::Periodic,
                VelocityBC::Periodic, VelocityBC::Periodic};
    }
    inline BCPreset channel_2d() {
        return {VelocityBC::Periodic, VelocityBC::Periodic,
                VelocityBC::NoSlip, VelocityBC::NoSlip};
    }
    inline BCPreset channel_3d() {
        return {VelocityBC::Periodic, VelocityBC::Periodic,
                VelocityBC::NoSlip, VelocityBC::NoSlip,
                VelocityBC::Periodic, VelocityBC::Periodic};
    }
}

//=============================================================================
// Manufactured Solutions
//=============================================================================

/// Base class for manufactured solutions
struct Solution {
    virtual ~Solution() = default;
    virtual double p(double x, double y, double z = 0) const = 0;
    virtual double rhs(double x, double y, double z = 0) const = 0;
    virtual double u(double x, double y, double z = 0) const { return 0; }
    virtual double v(double x, double y, double z = 0) const { return 0; }
    virtual double w(double x, double y, double z = 0) const { return 0; }
};

/// Sinusoidal solution: p = sin(kx*x) * sin(ky*y) * sin(kz*z)
struct SinSolution : Solution {
    double kx, ky, kz;

    SinSolution(double kx_ = 1, double ky_ = 1, double kz_ = 0)
        : kx(kx_), ky(ky_), kz(kz_) {}

    double p(double x, double y, double z = 0) const override {
        double val = std::sin(kx * x) * std::sin(ky * y);
        if (kz > 0) val *= std::sin(kz * z);
        return val;
    }

    double rhs(double x, double y, double z = 0) const override {
        double lap = -(kx*kx + ky*ky + (kz > 0 ? kz*kz : 0));
        return lap * p(x, y, z);
    }
};

/// Poiseuille flow: u(y) = (dp/dx)/(2*nu) * y * (H - y)
struct PoiseuilleSolution : Solution {
    double dp_dx, nu, H, y_min;

    PoiseuilleSolution(double dp_dx_ = -0.01, double nu_ = 0.01,
                       double H_ = 1.0, double y_min_ = 0.0)
        : dp_dx(dp_dx_), nu(nu_), H(H_), y_min(y_min_) {}

    double p(double x, double, double) const override { return dp_dx * x; }
    double rhs(double, double, double) const override { return 0; }

    double u(double, double y, double) const override {
        double y_rel = y - y_min;
        return (-dp_dx / (2.0 * nu)) * y_rel * (H - y_rel);
    }
};

/// Taylor-Green vortex (2D)
struct TaylorGreen2D : Solution {
    double L;
    TaylorGreen2D(double L_ = 2*M_PI) : L(L_) {}

    double p(double x, double y, double) const override {
        return 0.25 * (std::cos(2*x) + std::cos(2*y));
    }
    double rhs(double, double, double) const override { return 0; }
    double u(double x, double y, double) const override {
        return std::sin(x) * std::cos(y);
    }
    double v(double x, double y, double) const override {
        return -std::cos(x) * std::sin(y);
    }
};

//=============================================================================
// Result Types
//=============================================================================

struct ConvergenceResult {
    bool passed = false;
    std::vector<double> errors;
    std::vector<int> sizes;
    double rate = 0;
    std::string message;

    void print(const std::string& name = "") const {
        if (!name.empty()) std::cout << name << ": ";
        std::cout << (passed ? "PASSED" : "FAILED")
                  << " (rate=" << std::fixed << std::setprecision(2) << rate << ")\n";
        for (size_t i = 0; i < errors.size(); ++i) {
            std::cout << "  N=" << sizes[i] << ": error="
                      << std::scientific << errors[i] << "\n";
        }
    }
};

struct SteadyStateResult {
    bool passed = false;
    double l2_error = 0;
    int iterations = 0;
    double residual = 0;
    std::string message;

    void print(const std::string& name = "") const {
        if (!name.empty()) std::cout << name << ": ";
        std::cout << (passed ? "PASSED" : "FAILED")
                  << " (error=" << std::scientific << l2_error * 100 << "%, "
                  << "iters=" << iterations << ")\n";
    }
};

struct ComparisonResult {
    bool passed = false;
    double max_diff = 0;
    double rms_diff = 0;
    std::string field_name;
    std::string message;

    void print() const {
        std::cout << field_name << ": " << (passed ? "PASS" : "FAIL")
                  << " (max=" << std::scientific << max_diff
                  << ", rms=" << rms_diff << ")\n";
    }
};

//=============================================================================
// Test Runners
//=============================================================================

/// Compute L2 error with mean subtraction (for Neumann problems)
template<typename FieldT>
inline double compute_l2_error(const FieldT& p_num, const Mesh& mesh,
                               const Solution& sol) {
    double p_mean = 0, exact_mean = 0;
    int count = 0;

    if (mesh.is2D()) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                p_mean += p_num(i, j);
                exact_mean += sol.p(mesh.x(i), mesh.y(j));
                ++count;
            }
        }
    } else {
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    p_mean += p_num(i, j, k);
                    exact_mean += sol.p(mesh.x(i), mesh.y(j), mesh.z(k));
                    ++count;
                }
            }
        }
    }
    p_mean /= count;
    exact_mean /= count;

    double l2_error = 0;
    if (mesh.is2D()) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double exact = sol.p(mesh.x(i), mesh.y(j));
                double diff = (p_num(i, j) - p_mean) - (exact - exact_mean);
                l2_error += diff * diff;
            }
        }
    } else {
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    double exact = sol.p(mesh.x(i), mesh.y(j), mesh.z(k));
                    double diff = (p_num(i, j, k) - p_mean) - (exact - exact_mean);
                    l2_error += diff * diff;
                }
            }
        }
    }
    return std::sqrt(l2_error / count);
}

/// Run Poisson convergence study
enum class TestPoissonSolver { SOR, Multigrid };

inline ConvergenceResult run_poisson_convergence(
    const std::vector<int>& sizes,
    const Solution& sol,
    TestPoissonSolver solver_type,
    bool is_3d = false,
    double L = 2*M_PI,
    double expected_rate = 2.0,
    double rate_tolerance = 0.5)
{
    ConvergenceResult result;
    result.sizes = sizes;

    for (int N : sizes) {
        Mesh mesh;
        if (is_3d) {
            mesh.init_uniform(N, N, N, 0, L, 0, L, 0, L);
        } else {
            mesh.init_uniform(N, N, 0, L, 0, L);
        }

        ScalarField rhs(mesh), p(mesh, 0.0);

        // Set RHS from manufactured solution
        if (is_3d) {
            for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
                for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                    for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                        rhs(i, j, k) = sol.rhs(mesh.x(i), mesh.y(j), mesh.z(k));
                    }
                }
            }
        } else {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    rhs(i, j) = sol.rhs(mesh.x(i), mesh.y(j));
                }
            }
        }

        PoissonConfig cfg;
        cfg.tol = 1e-10;
        // SOR needs many more iterations than multigrid, especially in 3D
        if (solver_type == TestPoissonSolver::SOR) {
            cfg.max_iter = is_3d ? 200000 : 50000;
            cfg.omega = 1.7;  // Over-relaxation for faster convergence
        } else {
            cfg.max_iter = is_3d ? 200 : 100;
        }

        if (solver_type == TestPoissonSolver::SOR) {
            PoissonSolver solver(mesh);
            solver.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                         PoissonBC::Periodic, PoissonBC::Periodic);
            solver.solve(rhs, p, cfg);
        } else {
            MultigridPoissonSolver solver(mesh);
            solver.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                         PoissonBC::Periodic, PoissonBC::Periodic);
            solver.solve(rhs, p, cfg);
        }

        result.errors.push_back(compute_l2_error(p, mesh, sol));
    }

    // Compute convergence rate
    if (result.errors.size() >= 2) {
        result.rate = std::log2(result.errors[0] / result.errors[1]);
    }

    result.passed = (result.rate > expected_rate - rate_tolerance &&
                     result.rate < expected_rate + rate_tolerance);
    result.message = result.passed ? "PASSED" : "FAILED";

    return result;
}

/// Poisson BC configuration for flexible testing
struct PoissonBCConfig {
    PoissonBC x_lo = PoissonBC::Periodic, x_hi = PoissonBC::Periodic;
    PoissonBC y_lo = PoissonBC::Periodic, y_hi = PoissonBC::Periodic;
    PoissonBC z_lo = PoissonBC::Periodic, z_hi = PoissonBC::Periodic;

    static PoissonBCConfig periodic() {
        return {PoissonBC::Periodic, PoissonBC::Periodic,
                PoissonBC::Periodic, PoissonBC::Periodic,
                PoissonBC::Periodic, PoissonBC::Periodic};
    }
    static PoissonBCConfig channel() {  // periodic x/z, Neumann y
        return {PoissonBC::Periodic, PoissonBC::Periodic,
                PoissonBC::Neumann, PoissonBC::Neumann,
                PoissonBC::Periodic, PoissonBC::Periodic};
    }
    static PoissonBCConfig duct() {  // periodic x, Neumann y/z
        return {PoissonBC::Periodic, PoissonBC::Periodic,
                PoissonBC::Neumann, PoissonBC::Neumann,
                PoissonBC::Neumann, PoissonBC::Neumann};
    }
    static PoissonBCConfig channel_2d() {  // periodic x, Neumann y
        return {PoissonBC::Periodic, PoissonBC::Periodic,
                PoissonBC::Neumann, PoissonBC::Neumann};
    }
};

/// Domain configuration for Poisson tests
struct DomainConfig {
    double Lx, Ly, Lz;
    bool is_3d;

    static DomainConfig periodic_cube(double L = 2*M_PI) {
        return {L, L, L, true};
    }
    static DomainConfig channel_3d(double Lx = 2*M_PI, double Ly = 2.0, double Lz = 2*M_PI) {
        return {Lx, Ly, Lz, true};
    }
    static DomainConfig channel_2d(double Lx = 2*M_PI, double Ly = 2.0) {
        return {Lx, Ly, 0, false};
    }
};

/// Flexible Poisson convergence test with configurable BCs and domain
/// Works with manufactured solutions from test_fixtures.hpp
template<typename ManufacturedSol>
inline ConvergenceResult run_poisson_convergence_flex(
    const std::vector<int>& sizes,
    const ManufacturedSol& sol,
    TestPoissonSolver solver_type,
    const DomainConfig& domain,
    const PoissonBCConfig& bc,
    double expected_rate = 2.0,
    double rate_tolerance = 0.5)
{
    ConvergenceResult result;
    result.sizes = sizes;

    for (int N : sizes) {
        Mesh mesh;
        if (domain.is_3d) {
            mesh.init_uniform(N, N, N, 0, domain.Lx, 0, domain.Ly, 0, domain.Lz);
        } else {
            mesh.init_uniform(N, N, 0, domain.Lx, 0, domain.Ly);
        }

        ScalarField rhs(mesh), p(mesh, 0.0);

        // Set RHS from manufactured solution
        if (domain.is_3d) {
            for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
                for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                    for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                        rhs(i, j, k) = sol.rhs(mesh.x(i), mesh.y(j), mesh.z(k));
                    }
                }
            }
        } else {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    rhs(i, j) = sol.rhs(mesh.x(i), mesh.y(j));
                }
            }
        }

        PoissonConfig cfg;
        cfg.tol = 1e-10;
        cfg.max_iter = (solver_type == TestPoissonSolver::SOR) ? 50000 : 50;

        if (solver_type == TestPoissonSolver::SOR) {
            PoissonSolver solver(mesh);
            if (domain.is_3d) {
                solver.set_bc(bc.x_lo, bc.x_hi, bc.y_lo, bc.y_hi, bc.z_lo, bc.z_hi);
            } else {
                solver.set_bc(bc.x_lo, bc.x_hi, bc.y_lo, bc.y_hi);
            }
            solver.solve(rhs, p, cfg);
        } else {
            MultigridPoissonSolver solver(mesh);
            if (domain.is_3d) {
                solver.set_bc(bc.x_lo, bc.x_hi, bc.y_lo, bc.y_hi, bc.z_lo, bc.z_hi);
            } else {
                solver.set_bc(bc.x_lo, bc.x_hi, bc.y_lo, bc.y_hi);
            }
            solver.solve(rhs, p, cfg);
        }

        // Compute error with mean subtraction
        double p_mean = 0, exact_mean = 0;
        int count = 0;
        if (domain.is_3d) {
            for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
                for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                    for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                        p_mean += p(i, j, k);
                        exact_mean += sol.p(mesh.x(i), mesh.y(j), mesh.z(k));
                        ++count;
                    }
                }
            }
        } else {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    p_mean += p(i, j);
                    exact_mean += sol.p(mesh.x(i), mesh.y(j));
                    ++count;
                }
            }
        }
        p_mean /= count;
        exact_mean /= count;

        double l2_error = 0;
        if (domain.is_3d) {
            for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
                for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                    for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                        double exact = sol.p(mesh.x(i), mesh.y(j), mesh.z(k));
                        double diff = (p(i, j, k) - p_mean) - (exact - exact_mean);
                        l2_error += diff * diff;
                    }
                }
            }
        } else {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    double exact = sol.p(mesh.x(i), mesh.y(j));
                    double diff = (p(i, j) - p_mean) - (exact - exact_mean);
                    l2_error += diff * diff;
                }
            }
        }
        result.errors.push_back(std::sqrt(l2_error / count));
    }

    if (result.errors.size() >= 2) {
        result.rate = std::log2(result.errors[0] / result.errors[1]);
    }
    result.passed = (result.rate > expected_rate - rate_tolerance &&
                     result.rate < expected_rate + rate_tolerance);
    result.message = result.passed ? "PASSED" : "FAILED";

    return result;
}

/// Run steady-state flow test
inline SteadyStateResult run_steady_flow(
    const MeshPreset& mesh_cfg,
    const SolverPreset& solver_cfg,
    const BCPreset& bc_cfg,
    const Solution& exact,
    double tolerance,
    double body_force_x = 0,
    double body_force_y = 0)
{
    SteadyStateResult result;

    Mesh mesh = mesh_cfg.create();
    Config config = solver_cfg.to_config();
    RANSSolver solver(mesh, config);
    solver.set_velocity_bc(bc_cfg.to_velocity_bc());

    if (body_force_x != 0 || body_force_y != 0) {
        solver.set_body_force(body_force_x, body_force_y);
    }

    // Initialize near exact solution for fast convergence
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            solver.velocity().u(i, j) = 0.9 * exact.u(mesh.x(i), mesh.y(j));
        }
    }
    for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            solver.velocity().v(i, j) = 0.9 * exact.v(mesh.x(i), mesh.y(j));
        }
    }

    solver.sync_to_gpu();
    auto [residual, iters] = solver.solve_steady();
    solver.sync_from_gpu();

    // Compute L2 error in u-velocity
    double error_sq = 0, norm_sq = 0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double u_num = 0.5 * (solver.velocity().u(i, j) + solver.velocity().u(i+1, j));
            double u_ex = exact.u(mesh.x(i), mesh.y(j));
            error_sq += (u_num - u_ex) * (u_num - u_ex);
            norm_sq += u_ex * u_ex;
        }
    }
    result.l2_error = std::sqrt(error_sq / norm_sq);
    result.iterations = iters;
    result.residual = residual;
    result.passed = result.l2_error < tolerance;
    result.message = result.passed ? "PASSED" : "FAILED";

    return result;
}

/// Initialize Taylor-Green vortex
inline void init_taylor_green(RANSSolver& solver, const Mesh& mesh) {
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            double x = (i < mesh.i_end()) ? mesh.x(i) + mesh.dx/2.0 : mesh.x_max;
            solver.velocity().u(i, j) = std::sin(x) * std::cos(mesh.y(j));
        }
    }
    for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double y = (j < mesh.j_end()) ? mesh.y(j) + mesh.dy/2.0 : mesh.y_max;
            solver.velocity().v(i, j) = -std::cos(mesh.x(i)) * std::sin(y);
        }
    }
}

/// Compute kinetic energy
inline double compute_kinetic_energy(const Mesh& mesh, const VectorField& vel) {
    double KE = 0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double u = 0.5 * (vel.u(i, j) + vel.u(i+1, j));
            double v = 0.5 * (vel.v(i, j) + vel.v(i, j+1));
            KE += 0.5 * (u*u + v*v) * mesh.dx * mesh.dy;
        }
    }
    return KE;
}

//=============================================================================
// Assertions
//=============================================================================

inline void ASSERT_PASS(bool condition, const std::string& msg = "") {
    if (!condition) {
        throw std::runtime_error("ASSERTION FAILED: " + msg);
    }
}

inline void ASSERT_RATE(const ConvergenceResult& r, double expected = 2.0,
                        double margin = 0.5) {
    ASSERT_PASS(r.rate > expected - margin && r.rate < expected + margin,
                "Convergence rate " + std::to_string(r.rate) +
                " not in [" + std::to_string(expected - margin) + ", " +
                std::to_string(expected + margin) + "]");
}

inline void ASSERT_ERROR(const SteadyStateResult& r, double max_error) {
    ASSERT_PASS(r.l2_error < max_error,
                "L2 error " + std::to_string(r.l2_error) +
                " exceeds " + std::to_string(max_error));
}

//=============================================================================
// Common Flow Initialization Helpers
//=============================================================================

/// Initialize analytical Poiseuille profile for fast convergence
/// Profile: u(y) = -dp_dx/(2*nu) * (H² - y²) where H = half-height
inline void init_poiseuille(RANSSolver& solver, const Mesh& mesh,
                            double dp_dx, double nu, double H = 1.0, double scale = 0.9) {
    // Set u-velocity at x-faces (staggered grid)
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        double y = mesh.y(j);
        double u_analytical = -dp_dx / (2.0 * nu) * (H * H - y * y);
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            solver.velocity().u(i, j) = scale * u_analytical;
        }
    }
    // v-velocity stays zero
    for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            solver.velocity().v(i, j) = 0.0;
        }
    }
}

/// Compute L2 error of u-velocity profile vs analytical Poiseuille
inline double compute_poiseuille_error(const VectorField& vel, const Mesh& mesh,
                                       double dp_dx, double nu, double H = 1.0) {
    double l2_error_sq = 0.0, l2_norm_sq = 0.0;
    int i_center = mesh.i_begin() + mesh.Nx / 2;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        double y = mesh.y(j);
        double u_num = vel.u(i_center, j);
        double u_exact = -dp_dx / (2.0 * nu) * (H * H - y * y);
        double error = u_num - u_exact;
        l2_error_sq += error * error;
        l2_norm_sq += u_exact * u_exact;
    }
    return std::sqrt(l2_error_sq / l2_norm_sq);
}

/// Compute maximum divergence |∂u/∂x + ∂v/∂y|
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

//=============================================================================
// Platform-Specific Tolerance Helpers
//=============================================================================

/// Get steady-state iteration limit based on build type
inline int steady_max_iter() {
#ifdef USE_GPU_OFFLOAD
    return 120;   // Fast GPU smoke test
#else
    return 3000;  // Full CPU convergence
#endif
}

/// Get Poiseuille error limit based on build type
inline double poiseuille_error_limit() {
#ifdef USE_GPU_OFFLOAD
    return 0.05;  // 5% for GPU (120 iters)
#else
    return 0.03;  // 3% for CPU (3000 iters)
#endif
}

/// Get steady-state residual limit based on build type
inline double steady_residual_limit() {
#ifdef USE_GPU_OFFLOAD
    return 5e-3;  // Relaxed for fast GPU test
#else
    return 1e-4;  // Strict for CPU validation
#endif
}

//=============================================================================
// Common Mesh and Config Factory Functions
//=============================================================================

/// Create channel mesh (periodic x, walls y)
inline Mesh create_channel_mesh(int nx = 64, int ny = 128,
                                double Lx = 4.0, double Ly = 2.0) {
    Mesh mesh;
    mesh.init_uniform(nx, ny, 0.0, Lx, -Ly/2, Ly/2);  // y in [-1, 1]
    return mesh;
}

/// Create basic channel flow config
inline Config create_channel_config(double nu = 0.01, double dp_dx = -0.001,
                                    double dt = 0.01, int max_iter = 0) {
    Config config;
    config.nu = nu;
    config.dp_dx = dp_dx;
    config.dt = dt;
    config.adaptive_dt = false;
    config.max_iter = (max_iter > 0) ? max_iter : steady_max_iter();
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;
    return config;
}

/// Setup solver with channel BCs and body force
inline void setup_channel_solver(RANSSolver& solver, const Config& config) {
    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    solver.set_velocity_bc(bc);
    solver.set_body_force(-config.dp_dx, 0.0);
}

} // namespace test
} // namespace nncfd
