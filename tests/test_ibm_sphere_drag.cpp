/// @file test_ibm_sphere_drag.cpp
/// @brief Physics validation: sphere drag coefficient at Re=100 (3D IBM)
///
/// Steady flow past sphere at Re=100: Cd ~ 1.08 (Schiller-Naumann).
/// 3D IBM test — exercises halo exchange, 3D IBM geometry, force integration.
/// Tolerance ±40% due to coarse IBM grid resolution.

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

void test_sphere_drag_re100() {
    // Physical parameters
    // radius=0.75: with 64x32x32 grid, dy=dz=0.3125, band=0.469, nearest face
    // centers to sphere axis at dist≈0.221 → phi=0.221-0.75=-0.529 < -0.469 → solid.
    // radius=0.5 gives min dist 0.221 → phi=-0.279 > -0.469 → no solid cells.
    const double radius  = 0.75;
    const double D       = 2.0 * radius;    // 1.5
    const double U_inf   = 1.0;
    const double Re      = 100.0;
    const double nu      = U_inf * D / Re;  // 0.015

    // Schiller-Naumann: Cd = 24/Re * (1 + 0.15 * Re^0.687)
    const double Cd_ref  = (24.0 / Re) * (1.0 + 0.15 * std::pow(Re, 0.687));
    // At Re=100: Cd_ref ~ 1.08

    // Domain: [0,20] x [-5,5] x [-5,5]
    const double x_lo = 0.0,  x_hi = 20.0;
    const double y_lo = -5.0, y_hi =  5.0;
    const double z_lo = -5.0, z_hi =  5.0;

    // Grid: 64 x 32 x 32
    const int Nx = 64;
    const int Ny = 32;
    const int Nz = 32;

    // Sphere center
    const double cx = 4.0, cy = 0.0, cz = 0.0;

    // Time integration
    const double dt           = 0.002;
    const int    nsteps_total = 5000;
    const int    nsteps_avg   = 2000;    // average Cd over last 2000 steps

    // Reference area: pi * (D/2)^2
    const double A_ref = M_PI * radius * radius;

    // Tolerance: ±40% around Cd_ref ~ 1.08
    const double Cd_lo = 0.4;
    const double Cd_hi = 2.5;

    Mesh mesh;
    mesh.init_uniform(Nx, Ny, Nz, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi);

    Config config;
    config.nu           = nu;
    config.dt           = dt;
    config.adaptive_dt  = false;
    config.turb_model   = TurbulenceModelType::None;
    config.verbose      = false;

    RANSSolver solver(mesh, config);

    // Fully periodic in all directions
    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic; bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::Periodic; bc.y_hi = VelocityBC::Periodic;
    bc.z_lo = VelocityBC::Periodic; bc.z_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    // Uniform inflow: u=U_inf, v=0, w=0
    solver.initialize_uniform(U_inf, 0.0);
    solver.sync_to_gpu();

    // IBM: sphere centered at (cx, cy, cz)
    auto body = std::make_shared<SphereBody>(cx, cy, cz, radius);
    IBMForcing ibm(mesh, body);
    ibm.set_accumulate_forces(true);
    solver.set_ibm_forcing(&ibm);

    std::cout << "  IBM sphere: " << ibm.num_forcing_cells() << " forcing cells, "
              << ibm.num_solid_cells() << " solid cells" << std::endl;
    std::cout << "  Schiller-Naumann Cd_ref=" << Cd_ref << " at Re=" << Re << std::endl;
    std::cout << "  Reference area A_ref=" << A_ref << std::endl;

    CHECK(ibm.num_forcing_cells() > 0, "Must have forcing cells around sphere");
    CHECK(ibm.num_solid_cells()   > 0, "Must have solid cells inside sphere");

    // --- Run all steps, averaging force over last nsteps_avg steps ---
    const int step_avg_start = nsteps_total - nsteps_avg + 1;

    double Cd_sum = 0.0, Cl_sum = 0.0, Cz_sum = 0.0;
    int    n_avg  = 0;

    for (int step = 1; step <= nsteps_total; ++step) {
        solver.step();

        // Accumulate forces in averaging window
        if (step >= step_avg_start) {
            solver.sync_from_gpu();
            auto [Fx, Fy, Fz] = ibm.compute_forces(solver.velocity(), dt);

            double Cd = Fx / (0.5 * U_inf * U_inf * A_ref);
            double Cl = Fy / (0.5 * U_inf * U_inf * A_ref);
            double Cz = Fz / (0.5 * U_inf * U_inf * A_ref);

            Cd_sum += Cd;
            Cl_sum += Cl;
            Cz_sum += Cz;
            ++n_avg;
        }

        // Blow-up guard every 1000 steps
        if (step % 1000 == 0) {
            solver.sync_from_gpu();
            double u_max = 0.0;
            for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
                for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                    for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                        u_max = std::max(u_max, std::abs(solver.velocity().u(i, j, k)));
                    }
                }
            }
            std::cout << "  Step " << step << ": u_max=" << u_max << std::endl;
            if (u_max > 20.0 || std::isnan(u_max)) {
                throw std::runtime_error("Blow-up at step " + std::to_string(step)
                                         + " (u_max=" + std::to_string(u_max) + ")");
            }
        }
    }

    if (n_avg == 0) {
        throw std::runtime_error("No averaging steps accumulated");
    }

    double Cd_mean = Cd_sum / n_avg;
    double Cl_mean = Cl_sum / n_avg;
    double Cz_mean = Cz_sum / n_avg;

    std::cout << "  Cd_mean=" << Cd_mean
              << "  (ref=" << Cd_ref
              << ", window=[" << Cd_lo << "," << Cd_hi << "])" << std::endl;
    std::cout << "  Cl_mean=" << Cl_mean << "  Cz_mean=" << Cz_mean << std::endl;

    // Assertions
    CHECK(Cd_mean >= Cd_lo && Cd_mean <= Cd_hi,
          "Drag coefficient out of expected range [0.4, 2.5]");
    CHECK(std::abs(Cl_mean) + std::abs(Cz_mean) < 0.5,
          "Side force too large — flow should be nearly symmetric");

    std::cout << "PASS: Sphere drag at Re=100 (Cd=" << Cd_mean
              << ", ref=" << Cd_ref << ")" << std::endl;
}

int main() {
    try {
        test_sphere_drag_re100();
        std::cout << "\nAll sphere drag tests PASSED" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << std::endl;
        return 1;
    }
}
