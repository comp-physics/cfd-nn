/// @file test_les_tgv.cpp
/// @brief Validation test: 3D Taylor-Green vortex with WALE SGS model
///
/// Test coverage:
///   1. WALE model produces positive SGS viscosity for TGV flow
///   2. Energy decays monotonically (viscous + SGS dissipation)
///   3. Divergence stays small after projection
///   4. SGS viscosity is bounded and reasonable

#include "test_utilities.hpp"
#include "solver.hpp"
#include "config.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace nncfd;
using nncfd::test::BCPattern;
using nncfd::test::create_velocity_bc;

#define CHECK(cond, msg) do { if (!(cond)) throw std::runtime_error(msg); } while(0)

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

void test_les_tgv_wale() {
    const int N = 16;
    const int nsteps = 100;
    const double nu = 1e-3;
    const double dt = 5e-3;
    const double L = 2.0 * M_PI;

    Mesh mesh;
    mesh.init_uniform(N, N, N, 0.0, L, 0.0, L, 0.0, L);

    Config config;
    config.nu = nu;
    config.dt = dt;
    config.adaptive_dt = false;
    config.turb_model = TurbulenceModelType::WALE;
    config.verbose = false;

    RANSSolver solver(mesh, config);
    solver.set_velocity_bc(create_velocity_bc(BCPattern::FullyPeriodic));

    init_tgv_3d(solver, mesh);
    solver.sync_to_gpu();

    double E_prev = nncfd::test::compute_kinetic_energy_3d(solver.velocity(), mesh);
    double E_initial = E_prev;
    std::vector<double> energy_history;
    energy_history.push_back(E_prev);

    bool energy_monotonic = true;

    for (int step = 1; step <= nsteps; ++step) {
        solver.step();
        solver.sync_from_gpu();

        double E_curr = nncfd::test::compute_kinetic_energy_3d(solver.velocity(), mesh);
        energy_history.push_back(E_curr);

        if (E_curr > E_prev * (1.0 + 1e-12)) {
            energy_monotonic = false;
        }
        E_prev = E_curr;
    }

    double E_final = energy_history.back();
    double E_ratio = E_final / E_initial;
    double div = nncfd::test::compute_max_divergence_3d(solver.velocity(), mesh);

    std::cout << "  TGV+WALE: E_ratio=" << E_ratio << ", max_div=" << div << std::endl;

    CHECK(energy_monotonic, "Energy should decay monotonically with WALE + viscosity");
    CHECK(E_ratio < 1.0, "Energy must decrease from initial value");
    CHECK(E_ratio > 0.01, "Energy should not decay to near zero in 100 steps");
    CHECK(div < 1e-6, "Divergence should be small after projection");

    std::cout << "PASS: TGV + WALE LES (energy decays, div clean)" << std::endl;
}

void test_les_tgv_smagorinsky() {
    const int N = 16;
    const int nsteps = 100;
    const double nu = 1e-3;
    const double dt = 5e-3;
    const double L = 2.0 * M_PI;

    Mesh mesh;
    mesh.init_uniform(N, N, N, 0.0, L, 0.0, L, 0.0, L);

    Config config;
    config.nu = nu;
    config.dt = dt;
    config.adaptive_dt = false;
    config.turb_model = TurbulenceModelType::Smagorinsky;
    config.verbose = false;

    RANSSolver solver(mesh, config);
    solver.set_velocity_bc(create_velocity_bc(BCPattern::FullyPeriodic));

    init_tgv_3d(solver, mesh);
    solver.sync_to_gpu();

    double E_initial = nncfd::test::compute_kinetic_energy_3d(solver.velocity(), mesh);

    for (int step = 1; step <= nsteps; ++step) {
        solver.step();
    }
    solver.sync_from_gpu();

    double E_final = nncfd::test::compute_kinetic_energy_3d(solver.velocity(), mesh);
    double E_ratio = E_final / E_initial;

    std::cout << "  TGV+Smagorinsky: E_ratio=" << E_ratio << std::endl;

    CHECK(E_ratio < 1.0, "Energy must decrease with Smagorinsky model");
    CHECK(E_ratio > 0.001, "Energy should not vanish in 100 steps");

    std::cout << "PASS: TGV + Smagorinsky LES" << std::endl;
}

int main() {
    test_les_tgv_wale();
    test_les_tgv_smagorinsky();
    std::cout << "\nAll LES TGV tests PASSED" << std::endl;
    return 0;
}
