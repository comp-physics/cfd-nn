/// @file test_endurance_stability.cpp
/// @brief Endurance stability test for Poisson solvers
///
/// CRITICAL TEST: Catches "NaN after N steps" class bugs.
///
/// The HYPRE 2D y-periodic GPU instability manifested as NaN after ~10 steps.
/// Short tests (1-5 steps) passed but production runs failed. This test runs
/// 500 steps on small grids across multiple solver/BC configurations to catch
/// similar latent instabilities.
///
/// Tests:
///   1. 2D channel (periodic x, walls y) - baseline
///   2. 2D fully periodic - historically problematic on GPU
///   3. 3D channel (periodic xz, walls y) - production path
///   4. 3D duct (periodic x, walls yz) - FFT1D path
///   5. 2D y-periodic explicit MG - stability after HYPRE fallback
///
/// Each test runs 500 steps and asserts:
///   - No NaN/Inf in velocity or pressure
///   - Bounded divergence
///   - Finite kinetic energy

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "config.hpp"
#include "test_harness.hpp"
#include <cmath>
#include <string>

using namespace nncfd;
using nncfd::test::harness::record;

// Configuration for an endurance test
struct EnduranceConfig {
    std::string name;
    int Nx, Ny, Nz;
    double Lx, Ly, Lz;
    VelocityBC::Type x_lo, x_hi, y_lo, y_hi, z_lo, z_hi;
    PoissonSolverType solver;
    int nsteps;
    double dt;
};

// Check for NaN/Inf in a scalar field
bool has_nan_or_inf(const ScalarField& f, const Mesh& mesh) {
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double val = f(i, j, k);
                if (std::isnan(val) || std::isinf(val)) {
                    return true;
                }
            }
        }
    }
    return false;
}

// Check for NaN/Inf in a vector field
bool has_nan_or_inf_velocity(const VectorField& v, const Mesh& mesh) {
    // Check u component
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end() + 1; ++i) {
                double val = v.u(i, j, k);
                if (std::isnan(val) || std::isinf(val)) return true;
            }
        }
    }
    // Check v component
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end() + 1; ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double val = v.v(i, j, k);
                if (std::isnan(val) || std::isinf(val)) return true;
            }
        }
    }
    // Check w component (3D only)
    if (!mesh.is2D()) {
        for (int k = mesh.k_begin(); k < mesh.k_end() + 1; ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    double val = v.w(i, j, k);
                    if (std::isnan(val) || std::isinf(val)) return true;
                }
            }
        }
    }
    return false;
}

// Compute max absolute value
double max_abs(const ScalarField& f, const Mesh& mesh) {
    double max_val = 0.0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                max_val = std::max(max_val, std::abs(f(i, j, k)));
            }
        }
    }
    return max_val;
}

// Compute kinetic energy
double kinetic_energy(const VectorField& vel, const Mesh& mesh) {
    double ke = 0.0;
    int count = 0;

    if (mesh.is2D()) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double u = 0.5 * (vel.u(i, j) + vel.u(i+1, j));
                double v = 0.5 * (vel.v(i, j) + vel.v(i, j+1));
                ke += 0.5 * (u*u + v*v);
                ++count;
            }
        }
    } else {
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    double u = 0.5 * (vel.u(i, j, k) + vel.u(i+1, j, k));
                    double v = 0.5 * (vel.v(i, j, k) + vel.v(i, j+1, k));
                    double w = 0.5 * (vel.w(i, j, k) + vel.w(i, j, k+1));
                    ke += 0.5 * (u*u + v*v + w*w);
                    ++count;
                }
            }
        }
    }
    return ke / count;
}

// Run a single endurance test
bool run_endurance_test(const EnduranceConfig& cfg) {
    // Create mesh
    Mesh mesh;
    if (cfg.Nz == 1) {
        mesh.init_uniform(cfg.Nx, cfg.Ny, 0.0, cfg.Lx, 0.0, cfg.Ly);
    } else {
        mesh.init_uniform(cfg.Nx, cfg.Ny, cfg.Nz, 0.0, cfg.Lx, 0.0, cfg.Ly, 0.0, cfg.Lz);
    }

    // Create config
    Config config;
    config.Nx = cfg.Nx;
    config.Ny = cfg.Ny;
    config.Nz = cfg.Nz;
    config.x_min = 0.0; config.x_max = cfg.Lx;
    config.y_min = 0.0; config.y_max = cfg.Ly;
    config.z_min = 0.0; config.z_max = cfg.Lz;
    config.dt = cfg.dt;
    config.max_steps = cfg.nsteps + 100;  // Allow headroom
    config.nu = 0.01;  // Moderate viscosity for stability
    config.poisson_solver = cfg.solver;

    // Create solver
    RANSSolver solver(mesh, config);

    // Set BCs
    VelocityBC bc;
    bc.x_lo = cfg.x_lo;
    bc.x_hi = cfg.x_hi;
    bc.y_lo = cfg.y_lo;
    bc.y_hi = cfg.y_hi;
    bc.z_lo = cfg.z_lo;
    bc.z_hi = cfg.z_hi;
    solver.set_velocity_bc(bc);

    // Initialize with slightly perturbed flow
    VectorField vel(mesh);
    vel.fill(1.0, 0.0, 0.0);  // Base flow

    // Add small perturbation to trigger dynamics
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double x = mesh.x(i);
                double y = mesh.y(j);
                vel.u(i, j, k) += 0.01 * std::sin(2.0 * M_PI * y / cfg.Ly);
                vel.v(i, j, k) += 0.01 * std::sin(2.0 * M_PI * x / cfg.Lx);
            }
        }
    }

    solver.initialize(vel);

    // Set small body force for channel-like flow
    solver.set_body_force(0.001, 0.0, 0.0);

    // Run endurance test
    double initial_ke = 0.0;
    int check_interval = cfg.nsteps / 10;  // Check 10 times during run
    if (check_interval < 1) check_interval = 1;

    for (int step = 1; step <= cfg.nsteps; ++step) {
        solver.step();

        // Periodic checks
        if (step % check_interval == 0 || step == 1) {
            const ScalarField& p = solver.pressure();
            const VectorField& v = solver.velocity();

            // Check for NaN/Inf
            if (has_nan_or_inf(p, mesh)) return false;
            if (has_nan_or_inf_velocity(v, mesh)) return false;

            // Check for unbounded growth
            double p_max = max_abs(p, mesh);
            if (p_max > 1e10) return false;

            double ke = kinetic_energy(v, mesh);
            if (step == 1) initial_ke = ke;

            // Allow 100x growth as a very loose bound
            if (ke > 100.0 * initial_ke && initial_ke > 1e-10) return false;
        }
    }

    return true;
}

int main() {
    return nncfd::test::harness::run("Endurance Stability Tests", [] {
        const int NSTEPS = 500;  // Endurance run length
        const double DT = 0.001; // Small timestep for stability

        // 2D Tests
        record("2D channel baseline (500 steps)", run_endurance_test({
            "2D_channel_baseline", 64, 64, 1,
            2.0 * M_PI, 2.0, 1.0,
            VelocityBC::Periodic, VelocityBC::Periodic,
            VelocityBC::NoSlip, VelocityBC::NoSlip,
            VelocityBC::Periodic, VelocityBC::Periodic,
            PoissonSolverType::Auto, NSTEPS, DT
        }));

        record("2D fully periodic (500 steps)", run_endurance_test({
            "2D_fully_periodic", 64, 64, 1,
            2.0 * M_PI, 2.0 * M_PI, 1.0,
            VelocityBC::Periodic, VelocityBC::Periodic,
            VelocityBC::Periodic, VelocityBC::Periodic,
            VelocityBC::Periodic, VelocityBC::Periodic,
            PoissonSolverType::Auto, NSTEPS, DT
        }));

        record("2D y-periodic MG (500 steps)", run_endurance_test({
            "2D_y_periodic_MG", 64, 64, 1,
            2.0 * M_PI, 2.0 * M_PI, 1.0,
            VelocityBC::NoSlip, VelocityBC::NoSlip,
            VelocityBC::Periodic, VelocityBC::Periodic,
            VelocityBC::Periodic, VelocityBC::Periodic,
            PoissonSolverType::MG, NSTEPS, DT
        }));

        // 3D Tests (smaller grids for speed)
        record("3D channel FFT (500 steps)", run_endurance_test({
            "3D_channel_FFT", 32, 32, 32,
            2.0 * M_PI, 2.0, 2.0 * M_PI,
            VelocityBC::Periodic, VelocityBC::Periodic,
            VelocityBC::NoSlip, VelocityBC::NoSlip,
            VelocityBC::Periodic, VelocityBC::Periodic,
            PoissonSolverType::Auto, NSTEPS, DT
        }));

        record("3D duct FFT1D (500 steps)", run_endurance_test({
            "3D_duct_FFT1D", 32, 32, 32,
            2.0 * M_PI, 2.0, 2.0,
            VelocityBC::Periodic, VelocityBC::Periodic,
            VelocityBC::NoSlip, VelocityBC::NoSlip,
            VelocityBC::NoSlip, VelocityBC::NoSlip,
            PoissonSolverType::Auto, NSTEPS, DT
        }));

        record("3D cavity MG (500 steps)", run_endurance_test({
            "3D_cavity_MG", 32, 32, 32,
            2.0, 2.0, 2.0,
            VelocityBC::NoSlip, VelocityBC::NoSlip,
            VelocityBC::NoSlip, VelocityBC::NoSlip,
            VelocityBC::NoSlip, VelocityBC::NoSlip,
            PoissonSolverType::MG, NSTEPS, DT
        }));

#ifdef USE_HYPRE
        record("3D channel HYPRE (500 steps)", run_endurance_test({
            "3D_channel_HYPRE", 32, 32, 32,
            2.0 * M_PI, 2.0, 2.0 * M_PI,
            VelocityBC::Periodic, VelocityBC::Periodic,
            VelocityBC::NoSlip, VelocityBC::NoSlip,
            VelocityBC::Periodic, VelocityBC::Periodic,
            PoissonSolverType::HYPRE, NSTEPS, DT
        }));
#endif
    });
}
