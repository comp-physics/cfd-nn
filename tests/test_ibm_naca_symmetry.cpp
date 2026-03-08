/// @file test_ibm_naca_symmetry.cpp
/// @brief Physics validation: NACA 0012 symmetry — zero lift at AoA=0
///
/// A symmetric airfoil at zero angle of attack must produce Cl=0 by symmetry.
/// Tests IBM geometry correctness and force integration consistency.

#include "test_utilities.hpp"
#include "ibm_forcing.hpp"
#include "ibm_geometry.hpp"
#include "solver.hpp"
#include "config.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace nncfd;
using nncfd::test::BCPattern;
using nncfd::test::create_velocity_bc;

#define CHECK(cond, msg) do { if (!(cond)) throw std::runtime_error(msg); } while(0)

/// NACA 0012 at AoA=0 must produce zero lift by symmetry
void test_naca_symmetry() {
    // Physical parameters
    const double U_inf = 1.0;
    const double chord = 1.0;
    const double aoa_rad = 0.0;
    const double nu = 0.01;   // Re = U*c/nu = 100

    // Domain: [0,16] x [-6,6], airfoil leading edge at (3,0)
    const double x_min = 0.0,  x_max = 16.0;
    const double y_min = -6.0, y_max = 6.0;
    const int Nx = 80, Ny = 60;

    // Airfoil leading edge
    const double x_le = 3.0, y_le = 0.0;

    // Time stepping
    const double dt = 0.005;
    const int nsteps = 600;
    const int avg_start = nsteps - 200;  // average over last 200 steps

    std::cout << "=== NACA 0012 Symmetry (AoA=0) ===" << std::endl;
    std::cout << "  Domain: [" << x_min << "," << x_max << "] x ["
              << y_min << "," << y_max << "]" << std::endl;
    std::cout << "  Grid: " << Nx << " x " << Ny << std::endl;
    std::cout << "  chord=" << chord << ", AoA=0, nu=" << nu
              << ", Re=" << (U_inf * chord / nu) << std::endl;

    Mesh mesh;
    mesh.init_uniform(Nx, Ny, x_min, x_max, y_min, y_max);

    Config config;
    config.nu = nu;
    config.dt = dt;
    config.adaptive_dt = false;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);
    solver.set_velocity_bc(create_velocity_bc(BCPattern::FullyPeriodic));

    // Initialize uniform inflow
    solver.initialize_uniform(U_inf, 0.0);

    // NACA 0012 IBM
    auto body = std::make_shared<NACABody>(x_le, y_le, chord, aoa_rad, "0012");
    IBMForcing ibm(mesh, body);
    solver.set_ibm_forcing(&ibm);

    CHECK(ibm.num_forcing_cells() > 0, "Must have IBM forcing cells for NACA 0012");
    CHECK(ibm.num_solid_cells() > 0, "Must have IBM solid cells for NACA 0012");

    std::cout << "  IBM: " << ibm.num_forcing_cells() << " forcing, "
              << ibm.num_solid_cells() << " solid cells" << std::endl;

    // Reference area: chord * 1 (span = 1 for 2D)
    const double A_ref = chord * 1.0;
    const double q_inf = 0.5 * U_inf * U_inf;

    // Time integration + force accumulation
    double sum_Cd = 0.0, sum_Cl = 0.0;
    int n_avg = 0;
    bool stable = true;

    for (int step = 1; step <= nsteps; ++step) {
        solver.step();

        // Force measurement during averaging window
        if (step > avg_start) {
            solver.sync_from_gpu();
            auto [Fx, Fy, Fz] = ibm.compute_forces(solver.velocity(), dt);
            double Cd = Fx / (q_inf * A_ref);
            double Cl = Fy / (q_inf * A_ref);
            sum_Cd += Cd;
            sum_Cl += Cl;
            ++n_avg;
        }

        // Blow-up check every 200 steps
        if (step % 200 == 0) {
            solver.sync_from_gpu();

            double u_max = 0.0;
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                    u_max = std::max(u_max, std::abs(solver.velocity().u(i, j)));
                }
            }

            if (u_max > 100.0 || std::isnan(u_max)) {
                stable = false;
                std::cout << "  Blow-up at step " << step << ": u_max=" << u_max << std::endl;
                break;
            }
            std::cout << "  Step " << step << ": u_max=" << u_max << std::endl;
        }
    }

    CHECK(stable, "Simulation blew up — unstable");

    // Compute averaged coefficients
    if (n_avg == 0) {
        throw std::runtime_error("No averaging steps collected");
    }
    const double Cd_mean = sum_Cd / n_avg;
    const double Cl_mean = sum_Cl / n_avg;

    std::cout << "  Averaged over " << n_avg << " steps:" << std::endl;
    std::cout << "    Cl_mean = " << Cl_mean << " (expected ~0 by symmetry)" << std::endl;
    std::cout << "    Cd_mean = " << Cd_mean << " (some drag expected)" << std::endl;

    // Primary symmetry check: |Cl| < 0.05
    if (std::abs(Cl_mean) >= 0.05) {
        std::cout << "FAIL: |Cl_mean|=" << std::abs(Cl_mean)
                  << " exceeds 0.05 — NACA 0012 at AoA=0 must have Cl=0 by symmetry" << std::endl;
        throw std::runtime_error("|Cl| too large for symmetric airfoil at AoA=0");
    }

    // Sanity check: some drag must be present (IBM should impede flow)
    if (std::abs(Cd_mean) <= 0.0) {
        std::cout << "FAIL: Cd_mean=" << Cd_mean
                  << " — zero drag is unphysical for flow past an airfoil" << std::endl;
        throw std::runtime_error("Cd=0 is unphysical");
    }

    std::cout << "PASS: NACA 0012 symmetry (Cl=" << Cl_mean
              << ", Cd=" << Cd_mean << ")" << std::endl;
}

int main() {
    try {
        test_naca_symmetry();
        std::cout << "\nAll NACA symmetry tests PASSED" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << std::endl;
        return 1;
    }
}
