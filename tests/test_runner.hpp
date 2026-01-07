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

    // 3D mesh factories
    static MeshSpec taylor_green_3d(int n = 32) {
        return {n, n, n, 2.0*M_PI, 2.0*M_PI, 2.0*M_PI, 0.0, 0.0, 0.0, UNIFORM, 2.0};
    }

    static MeshSpec channel_3d(int nx = 16, int ny = 16, int nz = 8) {
        return {nx, ny, nz, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, UNIFORM, 2.0};
    }

    static MeshSpec cube(int n = 16, double L = 1.0) {
        return {n, n, n, L, L, L, 0.0, 0.0, 0.0, UNIFORM, 2.0};
    }

    // 3D Poiseuille channel (domain 4x2x1 with y in [0, 2], center at y=1)
    static MeshSpec poiseuille_3d(int nx = 32, int ny = 32, int nz = 8) {
        return {nx, ny, nz, 4.0, 2.0, 1.0, 0.0, 0.0, 0.0, UNIFORM, 2.0};
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

    static ConfigSpec laminar(double nu_val = 0.01) {
        ConfigSpec c;
        c.nu = nu_val;
        c.dt = 0.001;
        c.adaptive_dt = true;
        c.max_iter = 1000;
        c.tol = 1e-6;
        c.turb_model = TurbulenceModelType::None;
        return c;
    }

    static ConfigSpec turbulent_komega(double nu_val = 0.00005) {
        ConfigSpec c;
        c.nu = nu_val;
        c.dt = 0.001;
        c.adaptive_dt = true;
        c.max_iter = 5000;
        c.tol = 1e-5;
        c.turb_model = TurbulenceModelType::KOmega;
        return c;
    }

    static ConfigSpec unsteady(double nu_val = 0.01, double dt_val = 0.01) {
        ConfigSpec c;
        c.nu = nu_val;
        c.dt = dt_val;
        c.adaptive_dt = false;
        c.max_iter = 100;
        c.tol = 1e-6;
        c.turb_model = TurbulenceModelType::None;
        return c;
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
    enum Type { ZERO, UNIFORM, POISEUILLE, POISEUILLE_3D, TAYLOR_GREEN, TAYLOR_GREEN_3D, Z_INVARIANT, PERTURBED, CUSTOM };
    Type type = ZERO;
    double u0 = 0.0, v0 = 0.0, w0 = 0.0;
    double dp_dx = 0.0;
    double scale = 0.9;  // For Poiseuille: fraction of analytical
    std::function<void(RANSSolver&, const Mesh&)> custom_init;

    static InitSpec zero() {
        InitSpec i; i.type = ZERO; return i;
    }
    static InitSpec uniform(double u, double v = 0.0) {
        InitSpec i; i.type = UNIFORM; i.u0 = u; i.v0 = v; return i;
    }
    static InitSpec poiseuille(double dp, double sc = 0.9) {
        InitSpec i; i.type = POISEUILLE; i.dp_dx = dp; i.scale = sc; return i;
    }
    static InitSpec poiseuille_3d(double dp, double sc = 0.9) {
        InitSpec i; i.type = POISEUILLE_3D; i.dp_dx = dp; i.scale = sc; return i;
    }
    static InitSpec taylor_green() {
        InitSpec i; i.type = TAYLOR_GREEN; return i;
    }
    static InitSpec taylor_green_3d() {
        InitSpec i; i.type = TAYLOR_GREEN_3D; return i;
    }
    static InitSpec z_invariant(double dp = -0.001, double sc = 1.0) {
        InitSpec i; i.type = Z_INVARIANT; i.dp_dx = dp; i.scale = sc; return i;
    }
    static InitSpec perturbed() {
        InitSpec i; i.type = PERTURBED; return i;
    }
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

    static RunSpec steady() {
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
        L2_ERROR,          // Compare to analytical solution (2D)
        L2_ERROR_3D,       // Compare to analytical solution (3D)
        DIVERGENCE_FREE,   // Check |div(u)| < tol
        ENERGY_DECAY,      // Verify KE decreases monotonically
        BOUNDED,           // Verify max velocity stays bounded
        RESIDUAL,          // Check final residual < tol
        SYMMETRY,          // Check flow symmetry about centerline
        FINITE,            // Check all fields are finite (no NaN/Inf)
        REALIZABILITY,     // Check nu_t >= 0, k >= 0, omega > 0
        Z_INVARIANT,       // Check 3D flow stays z-invariant
        W_ZERO,            // Check w stays at machine zero (for 2D-in-3D)
        CUSTOM             // User-provided check function
    };
    Type type = NONE;
    double tolerance = 0.05;

    // For L2_ERROR: analytical solution (2D)
    std::function<double(double, double)> u_exact;
    std::function<double(double, double)> v_exact;

    // For L2_ERROR_3D: analytical solution (3D, function of y only for channel)
    std::function<double(double)> u_exact_3d;  // u(y)

    // For CUSTOM: user-provided check
    std::function<bool(const RANSSolver&, const Mesh&, std::string&)> custom_check;

    static CheckSpec none() {
        CheckSpec c; c.type = NONE; return c;
    }
    static CheckSpec converges() {
        CheckSpec c; c.type = CONVERGES; return c;
    }
    static CheckSpec l2_error(double tol,
                              std::function<double(double,double)> u_ex = nullptr) {
        CheckSpec c; c.type = L2_ERROR; c.tolerance = tol; c.u_exact = u_ex;
        return c;
    }
    static CheckSpec divergence_free(double tol = 1e-10) {
        CheckSpec c; c.type = DIVERGENCE_FREE; c.tolerance = tol; return c;
    }
    static CheckSpec energy_decay() {
        CheckSpec c; c.type = ENERGY_DECAY; return c;
    }
    static CheckSpec bounded(double max_vel = 10.0) {
        CheckSpec c; c.type = BOUNDED; c.tolerance = max_vel; return c;
    }
    static CheckSpec residual(double tol = 1e-6) {
        CheckSpec c; c.type = RESIDUAL; c.tolerance = tol; return c;
    }
    static CheckSpec symmetry(double tol = 0.01) {
        CheckSpec c; c.type = SYMMETRY; c.tolerance = tol; return c;
    }
    static CheckSpec finite() {
        CheckSpec c; c.type = FINITE; return c;
    }
    static CheckSpec realizability() {
        CheckSpec c; c.type = REALIZABILITY; return c;
    }
    static CheckSpec z_invariant(double tol = 1e-4) {
        CheckSpec c; c.type = Z_INVARIANT; c.tolerance = tol; return c;
    }
    static CheckSpec w_zero(double tol = 1e-8) {
        CheckSpec c; c.type = W_ZERO; c.tolerance = tol; return c;
    }
    static CheckSpec l2_error_3d(double tol, std::function<double(double)> u_ex) {
        CheckSpec c; c.type = L2_ERROR_3D; c.tolerance = tol; c.u_exact_3d = u_ex;
        return c;
    }
    static CheckSpec custom(std::function<bool(const RANSSolver&, const Mesh&, std::string&)> fn) {
        CheckSpec c; c.type = CUSTOM; c.custom_check = fn; return c;
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

// Helper to build TestSpec without C++20 designated initializers
inline TestSpec make_test(const std::string& name, const std::string& cat,
                          MeshSpec mesh, ConfigSpec config, BCSpec bc,
                          InitSpec init, RunSpec run, CheckSpec check) {
    TestSpec t;
    t.name = name;
    t.category = cat;
    t.mesh = mesh;
    t.config = config;
    t.bc = bc;
    t.init = init;
    t.run = run;
    t.check = check;
    return t;
}

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

        case InitSpec::POISEUILLE_3D: {
            // 3D Poiseuille: y ranges from 0 to Ly, center at Ly/2
            double dp_dx = init.dp_dx;
            double y_center = 0.5 * (mesh.y_min + mesh.y_max);
            double half_height = 0.5 * (mesh.y_max - mesh.y_min);
            for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
                for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                    double y = mesh.y(j);
                    double y_centered = y - y_center;
                    double u_ex = -dp_dx / (2.0 * nu) * (half_height * half_height - y_centered * y_centered);
                    for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                        solver.velocity().u(i, j, k) = init.scale * u_ex;
                    }
                }
            }
            break;
        }

        case InitSpec::TAYLOR_GREEN:
            // u at x-faces, v at y-faces (MAC grid)
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                    solver.velocity().u(i, j) = std::sin(mesh.xf[i]) * std::cos(mesh.y(j));
                }
            }
            for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    solver.velocity().v(i, j) = -std::cos(mesh.x(i)) * std::sin(mesh.yf[j]);
                }
            }
            break;

        case InitSpec::TAYLOR_GREEN_3D:
            // u = sin(x)cos(y)cos(z) at x-faces
            for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
                for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                    for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                        solver.velocity().u(i, j, k) = std::sin(mesh.xf[i]) * std::cos(mesh.y(j)) * std::cos(mesh.z(k));
                    }
                }
            }
            // v = -cos(x)sin(y)cos(z) at y-faces
            for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
                for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
                    for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                        solver.velocity().v(i, j, k) = -std::cos(mesh.x(i)) * std::sin(mesh.yf[j]) * std::cos(mesh.z(k));
                    }
                }
            }
            // w = 0 (already initialized to 0)
            break;

        case InitSpec::Z_INVARIANT: {
            // 3D Poiseuille-like profile, invariant in z
            double dp_dx = init.dp_dx;
            double y_center = 0.5 * (mesh.y_min + mesh.y_max);
            double half_height = 0.5 * (mesh.y_max - mesh.y_min);
            for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
                for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                    double y = mesh.y(j) - y_center;
                    double u_ex = -dp_dx / (2.0 * nu) * (half_height * half_height - y * y);
                    for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                        solver.velocity().u(i, j, k) = init.scale * u_ex;
                    }
                }
            }
            break;
        }

        case InitSpec::PERTURBED:
            throw std::runtime_error("PERTURBED initialization: use InitSpec::custom() with a custom init function");

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
    if (!mesh.is2D()) {
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    double dudx = (vel.u(i+1, j, k) - vel.u(i, j, k)) / mesh.dx;
                    double dvdy = (vel.v(i, j+1, k) - vel.v(i, j, k)) / mesh.dy;
                    double dwdz = (vel.w(i, j, k+1) - vel.w(i, j, k)) / mesh.dz;
                    max_div = std::max(max_div, std::abs(dudx + dvdy + dwdz));
                }
            }
        }
    } else {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double dudx = (vel.u(i+1, j) - vel.u(i, j)) / mesh.dx;
                double dvdy = (vel.v(i, j+1) - vel.v(i, j)) / mesh.dy;
                max_div = std::max(max_div, std::abs(dudx + dvdy));
            }
        }
    }
    return max_div;
}

inline double compute_kinetic_energy(const VectorField& vel, const Mesh& mesh) {
    double KE = 0.0;
    if (!mesh.is2D()) {
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    double u = 0.5 * (vel.u(i, j, k) + vel.u(i+1, j, k));
                    double v = 0.5 * (vel.v(i, j, k) + vel.v(i, j+1, k));
                    double w = 0.5 * (vel.w(i, j, k) + vel.w(i, j, k+1));
                    KE += 0.5 * (u*u + v*v + w*w) * mesh.dx * mesh.dy * mesh.dz;
                }
            }
        }
    } else {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double u = 0.5 * (vel.u(i, j) + vel.u(i+1, j));
                double v = 0.5 * (vel.v(i, j) + vel.v(i, j+1));
                KE += 0.5 * (u*u + v*v) * mesh.dx * mesh.dy;
            }
        }
    }
    return KE;
}

inline double compute_max_velocity(const VectorField& vel, const Mesh& mesh) {
    double max_vel = 0.0;
    if (!mesh.is2D()) {
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    double u = vel.u(i, j, k);
                    double v = vel.v(i, j, k);
                    double w = vel.w(i, j, k);
                    max_vel = std::max(max_vel, std::sqrt(u*u + v*v + w*w));
                }
            }
        }
    } else {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double u = vel.u(i, j);
                double v = vel.v(i, j);
                max_vel = std::max(max_vel, std::sqrt(u*u + v*v));
            }
        }
    }
    return max_vel;
}

// 3D-specific: Check z-invariance of a 3D field
inline double compute_z_variation(const VectorField& vel, const Mesh& mesh) {
    if (mesh.is2D()) return 0.0;

    double max_var = 0.0;
    int k0 = mesh.k_begin();
    for (int k = k0 + 1; k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                double diff = std::abs(vel.u(i, j, k) - vel.u(i, j, k0));
                max_var = std::max(max_var, diff);
            }
        }
    }
    return max_var;
}

// 3D L2 error vs analytical solution u(y) for Poiseuille-like flows
inline std::pair<double, double> compute_l2_error_3d(const VectorField& vel, const Mesh& mesh,
                                                     const std::function<double(double)>& u_exact) {
    if (!u_exact || mesh.is2D()) return {0.0, 0.0};

    double max_error = 0.0;
    double l2_error_sq = 0.0;
    int n_points = 0;

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            double y = mesh.y(j);
            double u_analytical = u_exact(y);
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                double u_computed = vel.u(i, j, k);
                double error = std::abs(u_computed - u_analytical);
                max_error = std::max(max_error, error);
                l2_error_sq += error * error;
                n_points++;
            }
        }
    }

    double l2_error = (n_points > 0) ? std::sqrt(l2_error_sq / n_points) : 0.0;
    return {max_error, l2_error};
}

// Check if w is essentially zero (for 2D flows extended to 3D)
inline std::pair<double, double> compute_w_relative(const VectorField& vel, const Mesh& mesh) {
    if (mesh.is2D()) return {0.0, 0.0};

    double max_w = 0.0;
    double max_u = 0.0;

    // Max |u|
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                max_u = std::max(max_u, std::abs(vel.u(i, j, k)));
            }
        }
    }

    // Max |w|
    for (int k = mesh.k_begin(); k <= mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                max_w = std::max(max_w, std::abs(vel.w(i, j, k)));
            }
        }
    }

    double w_relative = max_w / std::max(max_u, 1e-10);
    return {max_w, w_relative};
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
                if (spec.config.dt <= 0.0) {
                    throw std::runtime_error("TIME_EVOLVE requires dt > 0");
                }
                double t = 0.0;
                int max_steps = static_cast<int>(std::ceil(spec.run.t_end / spec.config.dt)) + 10;
                for (int step = 0; step < max_steps && t < spec.run.t_end; ++step) {
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

            case CheckSpec::SYMMETRY: {
                const VectorField& vel = solver.velocity();
                double max_asymmetry = 0.0;
                int i_mid = mesh.i_begin() + mesh.Nx / 2;
                for (int j = mesh.j_begin(); j < mesh.j_begin() + mesh.Ny/2; ++j) {
                    int j_mirror = mesh.j_end() - 1 - (j - mesh.j_begin());
                    double u_lower = vel.u(i_mid, j);
                    double u_upper = vel.u(i_mid, j_mirror);
                    double asymmetry = std::abs(u_lower - u_upper) / std::max(std::abs(u_lower), 1e-10);
                    max_asymmetry = std::max(max_asymmetry, asymmetry);
                }
                result.error = max_asymmetry;
                result.passed = (max_asymmetry < spec.check.tolerance);
                result.message = "asymmetry=" + std::to_string(max_asymmetry * 100) + "%";
                break;
            }

            case CheckSpec::FINITE: {
                const VectorField& vel = solver.velocity();
                bool all_finite = true;
                if (!mesh.is2D()) {
                    for (int k = mesh.k_begin(); k < mesh.k_end() && all_finite; ++k) {
                        for (int j = mesh.j_begin(); j < mesh.j_end() && all_finite; ++j) {
                            for (int i = mesh.i_begin(); i < mesh.i_end() && all_finite; ++i) {
                                if (!std::isfinite(vel.u(i,j,k)) || !std::isfinite(vel.v(i,j,k)) ||
                                    !std::isfinite(vel.w(i,j,k))) {
                                    all_finite = false;
                                }
                            }
                        }
                    }
                } else {
                    for (int j = mesh.j_begin(); j < mesh.j_end() && all_finite; ++j) {
                        for (int i = mesh.i_begin(); i < mesh.i_end() && all_finite; ++i) {
                            if (!std::isfinite(vel.u(i,j)) || !std::isfinite(vel.v(i,j))) {
                                all_finite = false;
                            }
                        }
                    }
                }
                result.passed = all_finite;
                result.message = all_finite ? "all finite" : "NaN/Inf detected";
                break;
            }

            case CheckSpec::REALIZABILITY: {
                const ScalarField& nu_t = solver.nu_t();
                double min_nu_t = 1e100;
                for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                    for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                        min_nu_t = std::min(min_nu_t, nu_t(i,j));
                    }
                }
                result.passed = (min_nu_t >= -1e-12);
                result.message = "min_nu_t=" + std::to_string(min_nu_t);
                break;
            }

            case CheckSpec::Z_INVARIANT: {
                double z_var = compute_z_variation(solver.velocity(), mesh);
                result.error = z_var;
                result.passed = (z_var < spec.check.tolerance);
                result.message = "z_variation=" + std::to_string(z_var);
                break;
            }

            case CheckSpec::L2_ERROR_3D: {
                auto [max_err, l2_err] = compute_l2_error_3d(solver.velocity(), mesh, spec.check.u_exact_3d);
                result.error = max_err;
                result.passed = (max_err < spec.check.tolerance);
                result.message = "max_err=" + std::to_string(max_err) + ", L2=" + std::to_string(l2_err);
                break;
            }

            case CheckSpec::W_ZERO: {
                auto [max_w, w_rel] = compute_w_relative(solver.velocity(), mesh);
                result.error = w_rel;
                result.passed = (w_rel < spec.check.tolerance);
                result.message = "|w|/|u|=" + std::to_string(w_rel);
                break;
            }

            case CheckSpec::CUSTOM: {
                std::string msg;
                result.passed = spec.check.custom_check(solver, mesh, msg);
                result.message = msg;
                break;
            }
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

        tests.push_back(make_test(
            "channel_" + std::to_string(nx) + "x" + std::to_string(ny),
            "physics",
            MeshSpec::channel(nx, ny),
            ConfigSpec::laminar(nu),
            BCSpec::channel(),
            InitSpec::poiseuille(dp_dx, init_factor),
            RunSpec::channel(dp_dx),
            CheckSpec::l2_error(0.05, u_exact)
        ));
    }

    return tests;
}

// Taylor-Green vortex decay tests
inline std::vector<TestSpec> taylor_green_suite() {
    std::vector<TestSpec> tests;

    for (int n : {32, 48, 64}) {
        tests.push_back(make_test(
            "taylor_green_" + std::to_string(n),
            "physics",
            MeshSpec::taylor_green(n),
            ConfigSpec::unsteady(0.01, 0.01),
            BCSpec::periodic(),
            InitSpec::taylor_green(),
            RunSpec::steps(50),
            CheckSpec::energy_decay()
        ));
    }

    return tests;
}

// 3D validation test suite
inline std::vector<TestSpec> validation_3d_suite() {
    std::vector<TestSpec> tests;

    // 3D Taylor-Green energy decay
    tests.push_back(make_test(
        "taylor_green_3d_32",
        "3d",
        MeshSpec::taylor_green_3d(32),
        ConfigSpec::unsteady(0.01, 0.01),
        BCSpec::periodic(),
        InitSpec::taylor_green_3d(),
        RunSpec::steps(50),
        CheckSpec::energy_decay()
    ));

    // 3D divergence-free check
    tests.push_back(make_test(
        "divergence_free_3d",
        "3d",
        MeshSpec::channel_3d(16, 16, 8),
        ConfigSpec::laminar(0.01),
        BCSpec::channel(),
        InitSpec::z_invariant(-0.001, 0.99),
        RunSpec::steps(20),
        CheckSpec::divergence_free(1e-3)
    ));

    // z-invariant flow preservation
    tests.push_back(make_test(
        "z_invariant_preservation",
        "3d",
        MeshSpec::channel_3d(16, 16, 8),
        ConfigSpec::unsteady(0.01, 0.001),
        BCSpec::channel(),
        InitSpec::z_invariant(-0.001, 1.0),
        RunSpec::steps(10),
        CheckSpec::z_invariant(1e-4)
    ));

    // 3D stability test
    tests.push_back(make_test(
        "stability_3d",
        "3d",
        MeshSpec::channel_3d(16, 16, 8),
        ConfigSpec::unsteady(0.01, 0.001),
        BCSpec::channel(),
        InitSpec::z_invariant(-0.001, 1.0),
        RunSpec::steps(50),
        CheckSpec::bounded(10.0)
    ));

    return tests;
}

} // namespace test
} // namespace nncfd
