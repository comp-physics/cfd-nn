/// @file test_les_tgv_dissipation.cpp
/// @brief Physics validation: LES SGS models add dissipation in TGV flow
///
/// An SGS model must increase energy dissipation relative to no model.
/// For the same IC and time steps, E_les < E_nomodel at all tested models.
/// Uses 3D Taylor-Green vortex on 16^3 grid.

#include "test_utilities.hpp"
#include "solver.hpp"
#include "config.hpp"
#include "turbulence_les.hpp"
#include "turbulence_model.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <iomanip>

using namespace nncfd;
using nncfd::test::BCPattern;
using nncfd::test::create_velocity_bc;

#define CHECK(cond, msg) do { if (!(cond)) throw std::runtime_error(msg); } while(0)

// ============================================================================
// TGV initial condition
// ============================================================================
static void init_tgv_3d(RANSSolver& solver, const Mesh& mesh) {
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                double x = mesh.xf[i];
                double y = mesh.y(j);
                double z = mesh.z(k);
                solver.velocity().u(i, j, k) = std::sin(x) * std::cos(y) * std::cos(z);
            }
        }
    }
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double x = mesh.x(i);
                double y = mesh.yf[j];
                double z = mesh.z(k);
                solver.velocity().v(i, j, k) = -std::cos(x) * std::sin(y) * std::cos(z);
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
}

// ============================================================================
// Run TGV for nsteps and return final mean kinetic energy per cell
// ============================================================================
static double run_tgv(TurbulenceModelType model_type, int nsteps,
                      double nu, double dt) {
    const int N = 16;
    const double L = 2.0 * M_PI;

    Mesh mesh;
    mesh.init_uniform(N, N, N, 0.0, L, 0.0, L, 0.0, L);

    Config config;
    config.nu = nu;
    config.dt = dt;
    config.adaptive_dt = false;
    config.turb_model = model_type;
    config.verbose = false;

    RANSSolver solver(mesh, config);
    solver.set_velocity_bc(create_velocity_bc(BCPattern::FullyPeriodic));

    // For LES models, create and attach the turbulence model explicitly
    if (model_type != TurbulenceModelType::None) {
        auto turb_model = create_turbulence_model(model_type);
        turb_model->set_nu(nu);
        solver.set_turbulence_model(std::move(turb_model));
    }

    init_tgv_3d(solver, mesh);
    solver.sync_to_gpu();

    for (int step = 0; step < nsteps; ++step) {
        double res = solver.step();
        if (!std::isfinite(res)) {
            throw std::runtime_error("Simulation blew up at step " + std::to_string(step));
        }
    }
    solver.sync_from_gpu();

    // Mean kinetic energy per cell
    double ke = 0.0;
    int count = 0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double uc = 0.5 * (solver.velocity().u(i, j, k) + solver.velocity().u(i+1, j, k));
                double vc = 0.5 * (solver.velocity().v(i, j, k) + solver.velocity().v(i, j+1, k));
                double wc = 0.5 * (solver.velocity().w(i, j, k) + solver.velocity().w(i, j, k+1));
                ke += 0.5 * (uc*uc + vc*vc + wc*wc);
                count++;
            }
        }
    }
    ke /= count;
    return ke;
}

// ============================================================================
// Compute initial energy for reference
// ============================================================================
static double compute_initial_energy() {
    const int N = 16;
    const double L = 2.0 * M_PI;
    const double nu = 1e-3;
    const double dt = 4e-3;

    Mesh mesh;
    mesh.init_uniform(N, N, N, 0.0, L, 0.0, L, 0.0, L);

    Config config;
    config.nu = nu;
    config.dt = dt;
    config.adaptive_dt = false;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);
    solver.set_velocity_bc(create_velocity_bc(BCPattern::FullyPeriodic));

    init_tgv_3d(solver, mesh);
    // Do NOT sync to GPU — compute from CPU fields directly

    double ke = 0.0;
    int count = 0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double uc = 0.5 * (solver.velocity().u(i, j, k) + solver.velocity().u(i+1, j, k));
                double vc = 0.5 * (solver.velocity().v(i, j, k) + solver.velocity().v(i, j+1, k));
                double wc = 0.5 * (solver.velocity().w(i, j, k) + solver.velocity().w(i, j, k+1));
                ke += 0.5 * (uc*uc + vc*vc + wc*wc);
                count++;
            }
        }
    }
    ke /= count;
    return ke;
}

// ============================================================================
// Main test
// ============================================================================
int main() {
    try {
        const int nsteps = 150;
        const double nu = 1e-3;
        const double dt = 4e-3;

        std::cout << "LES TGV Dissipation Test\n";
        std::cout << "  N=16, nsteps=" << nsteps << ", nu=" << nu << ", dt=" << dt << "\n\n";

        double E_initial = compute_initial_energy();
        std::cout << "  E_initial   = " << std::scientific << std::setprecision(6)
                  << E_initial << "\n";

        double E_nomodel = run_tgv(TurbulenceModelType::None, nsteps, nu, dt);
        double E_smag    = run_tgv(TurbulenceModelType::Smagorinsky, nsteps, nu, dt);
        double E_wale    = run_tgv(TurbulenceModelType::WALE, nsteps, nu, dt);
        double E_vreman  = run_tgv(TurbulenceModelType::Vreman, nsteps, nu, dt);
        double E_sigma   = run_tgv(TurbulenceModelType::Sigma, nsteps, nu, dt);

        std::cout << "\n  Model         E_final       E/E_nomodel\n";
        std::cout << "  No model      " << std::scientific << std::setprecision(6)
                  << E_nomodel << "  " << std::fixed << std::setprecision(6)
                  << 1.0 << "\n";
        std::cout << "  Smagorinsky   " << std::scientific << std::setprecision(6)
                  << E_smag << "  " << std::fixed << std::setprecision(6)
                  << E_smag / E_nomodel << "\n";
        std::cout << "  WALE          " << std::scientific << std::setprecision(6)
                  << E_wale << "  " << std::fixed << std::setprecision(6)
                  << E_wale / E_nomodel << "\n";
        std::cout << "  Vreman        " << std::scientific << std::setprecision(6)
                  << E_vreman << "  " << std::fixed << std::setprecision(6)
                  << E_vreman / E_nomodel << "\n";
        std::cout << "  Sigma         " << std::scientific << std::setprecision(6)
                  << E_sigma << "  " << std::fixed << std::setprecision(6)
                  << E_sigma / E_nomodel << "\n\n";

        // Basic sanity: all energies positive (no blow-up)
        CHECK(E_nomodel > 0.0, "No-model TGV: energy must be positive");
        CHECK(E_smag > 0.0,    "Smagorinsky TGV: energy must be positive");
        CHECK(E_wale > 0.0,    "WALE TGV: energy must be positive");
        CHECK(E_vreman > 0.0,  "Vreman TGV: energy must be positive");
        CHECK(E_sigma > 0.0,   "Sigma TGV: energy must be positive");

        // Pure viscous dissipation must also reduce energy
        CHECK(E_nomodel < E_initial, "No-model: viscous dissipation must reduce energy");

        // SGS models must dissipate at least 0.1% more than no-model
        CHECK(E_smag   < E_nomodel * 0.999,
              "Smagorinsky: must dissipate more energy than no-model (E_smag >= E_nomodel*0.999)");
        CHECK(E_wale   < E_nomodel * 0.999,
              "WALE: must dissipate more energy than no-model (E_wale >= E_nomodel*0.999)");
        CHECK(E_vreman < E_nomodel * 0.999,
              "Vreman: must dissipate more energy than no-model (E_vreman >= E_nomodel*0.999)");
        CHECK(E_sigma  < E_nomodel * 0.999,
              "Sigma: must dissipate more energy than no-model (E_sigma >= E_nomodel*0.999)");

        std::cout << "PASS: All LES SGS models add dissipation in TGV flow\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
}
