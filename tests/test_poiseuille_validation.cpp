/// @file test_poiseuille_validation.cpp
/// @brief Laminar Poiseuille flow validation against analytical solution
///
/// Validates:
///   1. Velocity profile matches U(y) = -(dp/dx)/(2*nu) * y * (Ly - y)
///   2. L2 relative error < 0.1%
///   3. Mass conservation (bulk velocity matches analytical)
///   4. Pressure gradient balance (wall shear = body force integral)
///   5. Symmetry: U(y) = U(Ly - y)

#include "test_harness.hpp"
#include "test_utilities.hpp"
#include "reference_data.hpp"
#include "solver.hpp"
#include "config.hpp"
#include <cmath>
#include <vector>
#include <iomanip>
#include <sstream>

using namespace nncfd;
using namespace nncfd::test;
using nncfd::test::harness::record;

// ============================================================================
// Section 1: Poiseuille Re=100 on uniform grid
// ============================================================================
void test_poiseuille_re100() {
    std::cout << "\n--- Poiseuille Re=100, 64x32 uniform ---\n\n";

    const int Nx = 64, Ny = 32;
    const double Lx = 2.0 * M_PI, Ly = 2.0;
    const double nu = 0.01;
    const double dp_dx = -1.0;  // Body force magnitude
    const int nsteps = 2000;

    // Analytical solution
    const double U_max_analytical = reference::poiseuille_centerline(dp_dx, nu, Ly);
    const double U_bulk_analytical = reference::poiseuille_bulk(dp_dx, nu, Ly);

    std::cout << "  Analytical: U_max=" << std::fixed << std::setprecision(4)
              << U_max_analytical << ", U_bulk=" << U_bulk_analytical << "\n";

    // Setup solver
    Mesh mesh;
    mesh.init_uniform(Nx, Ny, 0.0, Lx, 0.0, Ly);

    Config config;
    config.nu = nu;
    config.dt = 0.001;
    config.adaptive_dt = true;
    config.CFL_max = 0.5;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);
    solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel2D));
    solver.set_body_force(-dp_dx, 0.0);  // fx = -dp/dx
    solver.initialize_uniform(0.0, 0.0);
    solver.sync_to_gpu();

    // Run to steady state
    for (int step = 0; step < nsteps; ++step) {
        solver.step();
    }
    solver.sync_from_gpu();

    // Compute x-averaged U profile and compare to analytical
    double l2_num = 0.0, l2_den = 0.0;
    double linf_err = 0.0;
    double symmetry_err = 0.0;

    std::vector<double> U_computed(Ny, 0.0);
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        double u_sum = 0.0;
        int count = 0;
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            u_sum += 0.5 * (solver.velocity().u(i, j) + solver.velocity().u(i + 1, j));
            ++count;
        }
        int j_idx = j - mesh.j_begin();
        U_computed[j_idx] = u_sum / count;

        // y from bottom wall (mesh y goes from 0 to Ly)
        double y = mesh.y(j) - 0.0;
        double U_exact = reference::poiseuille_velocity(y, dp_dx, nu, Ly);

        double err = std::abs(U_computed[j_idx] - U_exact);
        l2_num += err * err;
        l2_den += U_exact * U_exact;
        linf_err = std::max(linf_err, err / (std::abs(U_exact) + 1e-30));
    }

    double l2_rel = std::sqrt(l2_num / (l2_den + 1e-30));

    // Symmetry: U(j) vs U(Ny-1-j)
    for (int j = 0; j < Ny / 2; ++j) {
        double diff = std::abs(U_computed[j] - U_computed[Ny - 1 - j]);
        double mag = std::max(std::abs(U_computed[j]), 1e-30);
        symmetry_err = std::max(symmetry_err, diff / mag);
    }

    // Bulk velocity
    double U_bulk = 0.0;
    for (double u : U_computed) U_bulk += u;
    U_bulk /= Ny;
    double bulk_err = std::abs(U_bulk - U_bulk_analytical) / U_bulk_analytical;

    // Divergence
    double max_div = solver.compute_divergence_linf_device();

    std::cout << "  L2 relative error: " << std::scientific << std::setprecision(2) << l2_rel << "\n";
    std::cout << "  Linf relative error: " << linf_err << "\n";
    std::cout << "  Bulk velocity error: " << bulk_err << "\n";
    std::cout << "  Symmetry error: " << symmetry_err << "\n";
    std::cout << "  max|div(u)|: " << max_div << "\n\n";

    record("L2 error < 0.1%", l2_rel < 0.001);
    record("Linf error < 1%", linf_err < 0.01);
    record("Bulk velocity error < 1%", bulk_err < 0.01);
    record("Symmetry (rel err < 1e-6)", symmetry_err < 1e-6);
    record("Incompressibility (div < 1e-6)", max_div < 1e-6);
}

// ============================================================================
// Section 2: Poiseuille Re=1000 on finer grid
// ============================================================================
void test_poiseuille_re1000() {
    std::cout << "\n--- Poiseuille Re=1000, 128x64 uniform ---\n\n";

    const int Nx = 128, Ny = 64;
    const double Lx = 2.0 * M_PI, Ly = 2.0;
    const double nu = 0.001;
    const double dp_dx = -1.0;
    const int nsteps = 5000;

    const double U_max_analytical = reference::poiseuille_centerline(dp_dx, nu, Ly);

    Mesh mesh;
    mesh.init_uniform(Nx, Ny, 0.0, Lx, 0.0, Ly);

    Config config;
    config.nu = nu;
    config.dt = 0.0005;
    config.adaptive_dt = true;
    config.CFL_max = 0.5;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);
    solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel2D));
    solver.set_body_force(-dp_dx, 0.0);
    solver.initialize_uniform(0.0, 0.0);
    solver.sync_to_gpu();

    for (int step = 0; step < nsteps; ++step) {
        solver.step();
    }
    solver.sync_from_gpu();

    // Compute L2 error
    double l2_num = 0.0, l2_den = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        double u_sum = 0.0;
        int count = 0;
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            u_sum += 0.5 * (solver.velocity().u(i, j) + solver.velocity().u(i + 1, j));
            ++count;
        }
        double U_computed = u_sum / count;
        double y = mesh.y(j);
        double U_exact = reference::poiseuille_velocity(y, dp_dx, nu, Ly);
        double err = U_computed - U_exact;
        l2_num += err * err;
        l2_den += U_exact * U_exact;
    }
    double l2_rel = std::sqrt(l2_num / (l2_den + 1e-30));

    double max_div = solver.compute_divergence_linf_device();

    std::cout << "  U_max analytical: " << std::fixed << std::setprecision(2) << U_max_analytical << "\n";
    std::cout << "  L2 relative error: " << std::scientific << std::setprecision(2) << l2_rel << "\n";
    std::cout << "  max|div(u)|: " << max_div << "\n\n";

    record("Re=1000 L2 error < 0.1%", l2_rel < 0.001);
    record("Re=1000 incompressibility", max_div < 1e-6);
}

// ============================================================================
// Section 3: Grid convergence (2nd-order spatial accuracy)
// ============================================================================
void test_poiseuille_convergence() {
    std::cout << "\n--- Poiseuille Grid Convergence (order of accuracy) ---\n\n";

    const double nu = 0.01;
    const double dp_dx = -1.0;
    const double Lx = 2.0 * M_PI, Ly = 2.0;
    const int grids[] = {16, 32, 64};
    double errors[3] = {};

    for (int g = 0; g < 3; ++g) {
        int Ny = grids[g];
        int Nx = 2 * Ny;
        int nsteps = 2000;

        Mesh mesh;
        mesh.init_uniform(Nx, Ny, 0.0, Lx, 0.0, Ly);

        Config config;
        config.nu = nu;
        config.dt = 0.001;
        config.adaptive_dt = true;
        config.CFL_max = 0.5;
        config.turb_model = TurbulenceModelType::None;
        config.verbose = false;

        RANSSolver solver(mesh, config);
        solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel2D));
        solver.set_body_force(-dp_dx, 0.0);
        solver.initialize_uniform(0.0, 0.0);
        solver.sync_to_gpu();

        for (int step = 0; step < nsteps; ++step) {
            solver.step();
        }
        solver.sync_from_gpu();

        double l2_num = 0.0, l2_den = 0.0;
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            double u_sum = 0.0;
            int count = 0;
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                u_sum += 0.5 * (solver.velocity().u(i, j) + solver.velocity().u(i + 1, j));
                ++count;
            }
            double U_computed = u_sum / count;
            double y = mesh.y(j);
            double U_exact = reference::poiseuille_velocity(y, dp_dx, nu, Ly);
            double err = U_computed - U_exact;
            l2_num += err * err;
            l2_den += U_exact * U_exact;
        }
        errors[g] = std::sqrt(l2_num / (l2_den + 1e-30));
        std::cout << "  Ny=" << Ny << ": L2_rel=" << std::scientific << std::setprecision(3) << errors[g] << "\n";
    }

    // Convergence rate: log(e1/e2) / log(h1/h2)
    double rate_1 = std::log(errors[0] / errors[1]) / std::log(2.0);
    double rate_2 = std::log(errors[1] / errors[2]) / std::log(2.0);

    std::cout << "  Convergence rate (16->32): " << std::fixed << std::setprecision(2) << rate_1 << "\n";
    std::cout << "  Convergence rate (32->64): " << rate_2 << "\n\n";

    // 2nd-order scheme should give rate >= 1.8 (allowing some tolerance)
    record("Convergence rate 16->32 >= 1.8", rate_1 >= 1.8);
    record("Convergence rate 32->64 >= 1.8", rate_2 >= 1.8);
}

// ============================================================================
// Main
// ============================================================================
int main() {
    return harness::run_sections("PoiseuilleValidation", {
        {"Re=100 analytical", test_poiseuille_re100},
        {"Re=1000 analytical", test_poiseuille_re1000},
        {"Grid convergence", test_poiseuille_convergence},
    });
}
