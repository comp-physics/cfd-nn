/// @file test_ibm_strouhal.cpp
/// @brief Physics validation: cylinder Strouhal number at Re=100
///
/// Vortex shedding at Re=100: St = f*D/U_inf ~ 0.165 (Williamson 1989).
/// Measure lift oscillation frequency after shedding is established.
/// Tolerance ±20% due to IBM grid resolution and periodic domain effects.

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

void test_cylinder_strouhal() {
    // Physical parameters
    const double D       = 1.0;
    const double U_inf   = 1.0;
    const double Re      = 100.0;
    const double nu      = U_inf * D / Re;   // 0.01

    // Domain: [0,20] x [-8,8]
    const double Lx      = 20.0;
    const double y_lo    = -8.0;
    const double y_hi    =  8.0;
    const double Ly      = y_hi - y_lo;

    // Grid: 96 x 64
    const int Nx = 96;
    const int Ny = 64;

    // Cylinder center and radius
    const double cx      = 5.0;
    const double cy      = 0.0;
    const double radius  = 0.5;   // D/2

    // Time integration
    const double dt            = 0.005;
    const int    nsteps_total  = 8000;
    const int    nsteps_trans  = 2000;   // discard transient
    const int    nsteps_anal   = nsteps_total - nsteps_trans;

    // Reference values
    const double St_ref        = 0.165;
    const double St_lo         = 0.130;
    const double St_hi         = 0.200;
    const int    min_crossings = 3;

    Mesh mesh;
    mesh.init_uniform(Nx, Ny, 0.0, Lx, y_lo, y_hi);

    Config config;
    config.nu           = nu;
    config.dt           = dt;
    config.adaptive_dt  = false;
    config.turb_model   = TurbulenceModelType::None;
    config.verbose      = false;

    RANSSolver solver(mesh, config);
    solver.set_velocity_bc(create_velocity_bc(BCPattern::FullyPeriodic));

    // Uniform inflow initialization
    const int Ng = mesh.Nghost;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            solver.velocity().u(i, j) = U_inf;
        }
    }
    solver.sync_to_gpu();

    // IBM: cylinder at (cx, cy) with radius D/2
    auto body = std::make_shared<CylinderBody>(cx, cy, radius);
    IBMForcing ibm(mesh, body);
    solver.set_ibm_forcing(&ibm);

    std::cout << "  IBM cylinder: " << ibm.num_forcing_cells() << " forcing cells, "
              << ibm.num_solid_cells() << " solid cells" << std::endl;
    CHECK(ibm.num_forcing_cells() > 0, "Must have forcing cells around cylinder");
    CHECK(ibm.num_solid_cells()   > 0, "Must have solid cells inside cylinder");

    // Reference area for 2D drag/lift (per unit span, D as reference length)
    const double A_ref = D;   // 2D: force coefficient uses diameter as reference length

    // --- Transient phase ---
    for (int step = 1; step <= nsteps_trans; ++step) {
        solver.step();

        if (step % 1000 == 0) {
            solver.sync_from_gpu();
            double u_max = 0.0;
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                    u_max = std::max(u_max, std::abs(solver.velocity().u(i, j)));
                }
            }
            std::cout << "  Transient step " << step << ": u_max=" << u_max << std::endl;
            if (u_max > 10.0 || std::isnan(u_max)) {
                throw std::runtime_error("Blow-up during transient (step " + std::to_string(step) + ")");
            }
        }
    }

    // --- Analysis phase: collect Cl and Cd ---
    // Sync every 10 steps to avoid excessive GPU→CPU transfers (6000 steps total).
    // This gives 600 force samples — more than sufficient to measure shedding frequency.
    const int sync_interval = 10;
    std::vector<double> Cl_signal;
    Cl_signal.reserve(nsteps_anal / sync_interval + 1);
    double Cd_sum = 0.0;
    int    Cd_count = 0;

    for (int step = 1; step <= nsteps_anal; ++step) {
        solver.step();

        if (step % sync_interval == 0) {
            // Sync GPU→CPU for force measurement (compute_forces reads CPU-side velocity)
            solver.sync_from_gpu();
            auto [Fx, Fy, Fz] = ibm.compute_forces(solver.velocity(), dt);

            double Cl = Fy / (0.5 * U_inf * U_inf * A_ref);
            double Cd = Fx / (0.5 * U_inf * U_inf * A_ref);

            Cl_signal.push_back(Cl);
            Cd_sum   += Cd;
            ++Cd_count;

            if (step % 1000 == 0) {
                double u_max = 0.0;
                for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                    for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                        u_max = std::max(u_max, std::abs(solver.velocity().u(i, j)));
                    }
                }
                std::cout << "  Analysis step " << step << ": u_max=" << u_max
                          << "  Cl=" << Cl << "  Cd=" << Cd << std::endl;
                if (u_max > 10.0 || std::isnan(u_max)) {
                    throw std::runtime_error("Blow-up during analysis (step " + std::to_string(step) + ")");
                }
            }
        }
    }

    // --- Strouhal number measurement via zero-crossings ---
    int num_upward_crossings = 0;
    for (int i = 1; i < static_cast<int>(Cl_signal.size()); ++i) {
        if (Cl_signal[i - 1] < 0.0 && Cl_signal[i] >= 0.0) {
            ++num_upward_crossings;
        }
    }

    double total_time  = nsteps_anal * dt;           // physical time of analysis window
    double frequency   = (num_upward_crossings > 0)
                         ? (static_cast<double>(num_upward_crossings) / total_time)
                         : 0.0;
    // Note: Cl_signal sampled every sync_interval steps, but total_time is the full
    // analysis window, so frequency is correctly computed in physical time.
    // St = f * D / U_inf  (D=1, U_inf=1)
    double St          = frequency * D / U_inf;
    double Cd_mean     = (Cd_count > 0) ? (Cd_sum / Cd_count) : 0.0;

    // Cl amplitude: max of |Cl| over analysis window
    double Cl_amp = 0.0;
    for (double cl : Cl_signal) {
        Cl_amp = std::max(Cl_amp, std::abs(cl));
    }

    std::cout << "  Strouhal number:  St=" << St
              << "  (ref=" << St_ref << ", window=[" << St_lo << "," << St_hi << "])" << std::endl;
    std::cout << "  Drag coefficient: Cd_mean=" << Cd_mean << std::endl;
    std::cout << "  Lift amplitude:   Cl_amp=" << Cl_amp << std::endl;
    std::cout << "  Zero-crossings:   " << num_upward_crossings
              << "  (need >= " << min_crossings << ")" << std::endl;

    // Assertions
    CHECK(num_upward_crossings >= min_crossings,
          "Not enough zero-crossings — vortex shedding not established");
    CHECK(St >= St_lo && St <= St_hi,
          "Strouhal number out of range (±20% around 0.165)");
    CHECK(Cd_mean >= 0.7 && Cd_mean <= 2.5,
          "Drag coefficient out of expected range [0.7, 2.5]");

    std::cout << "PASS: Cylinder Strouhal number at Re=100 (St=" << St << ")" << std::endl;
}

int main() {
    try {
        test_cylinder_strouhal();
        std::cout << "\nAll Strouhal tests PASSED" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << std::endl;
        return 1;
    }
}
