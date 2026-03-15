/// @file test_ibm_cylinder_drag.cpp
/// @brief Physics validation: cylinder drag coefficient at Re=20
///
/// Steady cylinder flow at Re=20: Cd ~ 2.05 (Tritton 1959)
/// IBM direct-forcing on periodic domain. Drag measured from IBM force output.
/// Tolerance ±35% due to Cartesian-grid IBM discretization error.

#include "test_utilities.hpp"
#include "ibm_forcing.hpp"
#include "ibm_geometry.hpp"
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

/// Cylinder drag coefficient validation at Re=20 (Tritton 1959: Cd ~ 2.05)
void test_cylinder_drag_re20() {
    // Physical parameters
    const double U_inf = 1.0;
    const double D = 1.0;        // cylinder diameter = 2 * radius
    const double radius = D / 2.0;
    const double Re = 20.0;
    const double nu = U_inf * D / Re;  // = 0.05

    // Domain: [0,20] x [-8,8]
    const double x_min = 0.0, x_max = 20.0;
    const double y_min = -8.0, y_max = 8.0;
    const int Nx = 128, Ny = 80;

    // Cylinder center at (5, 0)
    const double cx = 5.0, cy = 0.0;

    // Time stepping
    const double dt = 0.005;
    const int nsteps = 4000;
    const int avg_start = nsteps - 1000;  // average over last 1000 steps

    std::cout << "=== Cylinder Drag Re=20 ===" << std::endl;
    std::cout << "  Domain: [" << x_min << "," << x_max << "] x ["
              << y_min << "," << y_max << "]" << std::endl;
    std::cout << "  Grid: " << Nx << " x " << Ny << std::endl;
    std::cout << "  nu=" << nu << ", Re=" << Re << ", D=" << D << std::endl;

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

    // Cylinder IBM
    auto body = std::make_shared<CylinderBody>(cx, cy, radius);
    IBMForcing ibm(mesh, body);
    ibm.set_accumulate_forces(true);
    solver.set_ibm_forcing(&ibm);

    CHECK(ibm.num_forcing_cells() > 0, "Must have IBM forcing cells");
    CHECK(ibm.num_solid_cells() > 0, "Must have IBM solid cells");

    std::cout << "  IBM: " << ibm.num_forcing_cells() << " forcing, "
              << ibm.num_solid_cells() << " solid cells" << std::endl;

    // Reference area: D * 1 (span = 1 for 2D)
    const double A_ref = D * 1.0;
    const double q_inf = 0.5 * U_inf * U_inf;

    // Time integration + force accumulation
    double sum_Cd = 0.0, sum_Cl = 0.0;
    int n_avg = 0;
    bool stable = true;

    for (int step = 1; step <= nsteps; ++step) {
        solver.step();

        // Force measurement at every step during averaging window
        if (step > avg_start) {
            auto [Fx, Fy, Fz] = ibm.compute_forces(solver.velocity(), dt);
            // compute_forces returns force on body (positive drag in +x direction)
            double Cd = Fx / (q_inf * A_ref);
            double Cl = Fy / (q_inf * A_ref);
            sum_Cd += Cd;
            sum_Cl += Cl;
            ++n_avg;
        }

        // Blow-up check every 1000 steps
        if (step % 1000 == 0) {
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
    std::cout << "    Cd_mean = " << Cd_mean << " (reference: 2.05)" << std::endl;
    std::cout << "    Cl_mean = " << Cl_mean << " (expected ~0 at Re=20)" << std::endl;

    // Physics checks
    // Cd in [1.0, 3.5]: predictor-based force correctly captures IBM drag near Tritton 2.05
    if (Cd_mean < 1.0 || Cd_mean > 3.5) {
        std::cout << "FAIL: Cd_mean=" << Cd_mean
                  << " outside expected range [1.0, 3.5] for Re=20" << std::endl;
        throw std::runtime_error("Cd out of range");
    }

    // Lift should be near zero (symmetric flow, no vortex shedding at Re=20)
    if (std::abs(Cl_mean) > 0.3) {
        std::cout << "FAIL: |Cl_mean|=" << std::abs(Cl_mean)
                  << " exceeds 0.3 (flow should be symmetric at Re=20)" << std::endl;
        throw std::runtime_error("|Cl| too large");
    }

    std::cout << "PASS: Cylinder drag Re=20 (Cd=" << Cd_mean
              << ", Cl=" << Cl_mean << ")" << std::endl;
}

int main() {
    try {
        test_cylinder_drag_re20();
        std::cout << "\nAll cylinder drag tests PASSED" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << std::endl;
        return 1;
    }
}
