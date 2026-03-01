/// @file test_poiseuille_validation.cpp
/// @brief Laminar Poiseuille flow validation against analytical solution
///
/// Validates:
///   1. Velocity profile matches U(y) = -(dp/dx)/(2*nu) * y * (H - y)
///   2. L2 relative error < 5% (grid-dependent discretization error)
///   3. Mass conservation (bulk velocity matches analytical)
///   4. Symmetry: U(y) = U(H - y)
///   5. Grid convergence rate (2nd order)

#include "test_harness.hpp"
#include "test_utilities.hpp"
#include "reference_data.hpp"
#include "solver.hpp"
#include "config.hpp"
#include <cmath>
#include <vector>
#include <iomanip>

using namespace nncfd;
using namespace nncfd::test;
using nncfd::test::harness::record;

// ============================================================================
// Helper: compute profile errors vs analytical Poiseuille solution
// ============================================================================
namespace {

struct ProfileErrors {
    double l2_rel;
    double linf_rel;
    double bulk_err;
    double symmetry_err;
    double max_div;
};

ProfileErrors compute_poiseuille_errors(RANSSolver& solver, const Mesh& mesh,
                                         double dp_dx, double nu) {
    const double H = mesh.y_max - mesh.y_min;
    const double U_bulk_analytical = reference::poiseuille_bulk(dp_dx, nu, H);

    double l2_num = 0.0, l2_den = 0.0;
    double linf_err = 0.0;
    double symmetry_err = 0.0;

    int ny_int = mesh.j_end() - mesh.j_begin();
    std::vector<double> U_computed(ny_int, 0.0);
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        double u_sum = 0.0;
        int count = 0;
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            u_sum += 0.5 * (solver.velocity().u(i, j) + solver.velocity().u(i + 1, j));
            ++count;
        }
        int j_idx = j - mesh.j_begin();
        U_computed[j_idx] = u_sum / count;

        double y_rel = mesh.y(j) - mesh.y_min;
        double U_exact = reference::poiseuille_velocity(y_rel, dp_dx, nu, H);

        double err = std::abs(U_computed[j_idx] - U_exact);
        l2_num += err * err;
        l2_den += U_exact * U_exact;
        linf_err = std::max(linf_err, err / (std::abs(U_exact) + 1e-30));
    }

    double l2_rel = std::sqrt(l2_num / (l2_den + 1e-30));

    // Symmetry: U(j) vs U(ny_int-1-j)
    for (int j = 0; j < ny_int / 2; ++j) {
        double diff = std::abs(U_computed[j] - U_computed[ny_int - 1 - j]);
        double mag = std::max(std::abs(U_computed[j]), 1e-30);
        symmetry_err = std::max(symmetry_err, diff / mag);
    }

    // Bulk velocity
    double U_bulk = 0.0;
    for (double u : U_computed) U_bulk += u;
    U_bulk /= ny_int;
    double bulk_err = std::abs(U_bulk - U_bulk_analytical) / (std::abs(U_bulk_analytical) + 1e-30);

    double max_div = compute_max_divergence_2d(solver.velocity(), mesh);

    return {l2_rel, linf_err, bulk_err, symmetry_err, max_div};
}

/// Initialize velocity to fraction of analytical Poiseuille profile
void init_poiseuille(RANSSolver& solver, const Mesh& mesh,
                     double dp_dx, double nu, double fraction) {
    const double H = mesh.y_max - mesh.y_min;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        double y_rel = mesh.y(j) - mesh.y_min;
        double u_init = fraction * reference::poiseuille_velocity(y_rel, dp_dx, nu, H);
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            solver.velocity().u(i, j) = u_init;
        }
    }
}

/// Configure a Config object for Poiseuille steady-state solving
void configure_poiseuille(Config& config, double nu, int max_steps) {
    config.nu = nu;
    config.dt = 0.005;
    config.adaptive_dt = true;
    config.CFL_max = 0.5;
    config.max_steps = max_steps;
    config.tol = 1e-10;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;
}

} // anonymous namespace

// ============================================================================
// Section 1: Poiseuille on 32x64 grid
// ============================================================================
void test_poiseuille_32x64() {
    std::cout << "\n--- Poiseuille 32x64, dp/dx=-0.01, nu=0.01, solve_steady ---\n\n";

    const double H = 1.0;
    const double nu = 0.01;
    const double dp_dx = -0.01;

    const double U_max = reference::poiseuille_centerline(dp_dx, nu, H);
    const double U_bulk = reference::poiseuille_bulk(dp_dx, nu, H);
    std::cout << "  Analytical: U_max=" << std::fixed << std::setprecision(4)
              << U_max << ", U_bulk=" << U_bulk << "\n";

    Mesh mesh;
    mesh.init_uniform(32, 64, 0.0, 4.0, 0.0, H);
    Config config;
    configure_poiseuille(config, nu, 20000);
    RANSSolver solver(mesh, config);
    solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel2D));
    solver.set_body_force(-dp_dx, 0.0);
    init_poiseuille(solver, mesh, dp_dx, nu, 0.9);
    solver.sync_to_gpu();

    auto [residual, iters] = solver.solve_steady();
    solver.sync_from_gpu();

    std::cout << "  Converged: residual=" << std::scientific << std::setprecision(2)
              << residual << " in " << iters << " steps\n";

    auto err = compute_poiseuille_errors(solver, mesh, dp_dx, nu);

    std::cout << "  L2 relative error: " << std::scientific << err.l2_rel << "\n";
    std::cout << "  Linf relative error: " << err.linf_rel << "\n";
    std::cout << "  Bulk velocity error: " << err.bulk_err << "\n";
    std::cout << "  Symmetry error: " << err.symmetry_err << "\n";
    std::cout << "  max|div(u)|: " << err.max_div << "\n\n";

    record("L2 error < 1%", err.l2_rel < 0.01);
    record("Linf error < 5%", err.linf_rel < 0.05);
    record("Bulk velocity error < 1%", err.bulk_err < 0.01);
    record("Symmetry (rel err < 1e-6)", err.symmetry_err < 1e-6);
    record("Incompressibility (div < 1e-6)", err.max_div < 1e-6);
}

// ============================================================================
// Section 2: Poiseuille on finer 64x128 grid
// ============================================================================
void test_poiseuille_64x128() {
    std::cout << "\n--- Poiseuille 64x128, dp/dx=-0.01, nu=0.01, solve_steady ---\n\n";

    const double H = 1.0;
    const double nu = 0.01;
    const double dp_dx = -0.01;

    const double U_max = reference::poiseuille_centerline(dp_dx, nu, H);
    std::cout << "  Analytical U_max: " << std::fixed << std::setprecision(4) << U_max << "\n";

    Mesh mesh;
    mesh.init_uniform(64, 128, 0.0, 4.0, 0.0, H);
    Config config;
    configure_poiseuille(config, nu, 20000);
    RANSSolver solver(mesh, config);
    solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel2D));
    solver.set_body_force(-dp_dx, 0.0);
    init_poiseuille(solver, mesh, dp_dx, nu, 0.9);
    solver.sync_to_gpu();

    auto [residual, iters] = solver.solve_steady();
    solver.sync_from_gpu();

    std::cout << "  Converged: residual=" << std::scientific << std::setprecision(2)
              << residual << " in " << iters << " steps\n";

    auto err = compute_poiseuille_errors(solver, mesh, dp_dx, nu);

    std::cout << "  L2 relative error: " << std::scientific << err.l2_rel << "\n";
    std::cout << "  max|div(u)|: " << err.max_div << "\n\n";

    // Finer grid should give smaller error
    record("Fine-grid L2 error < 2%", err.l2_rel < 0.02);
    record("Fine-grid incompressibility", err.max_div < 1e-6);
}

// ============================================================================
// Section 3: Grid convergence (2nd-order spatial accuracy)
// ============================================================================
void test_poiseuille_convergence() {
    std::cout << "\n--- Poiseuille Grid Convergence (order of accuracy) ---\n\n";

    const double nu = 0.01;
    const double dp_dx = -0.01;
    const double H = 1.0;
    const int grids[] = {16, 32, 64};
    double errors[3] = {};

    for (int g = 0; g < 3; ++g) {
        int Ny = grids[g];
        int Nx = Ny / 2;

        Mesh mesh;
        mesh.init_uniform(Nx, Ny, 0.0, 4.0, 0.0, H);
        Config config;
        configure_poiseuille(config, nu, 30000);
        RANSSolver solver(mesh, config);
        solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel2D));
        solver.set_body_force(-dp_dx, 0.0);
        init_poiseuille(solver, mesh, dp_dx, nu, 0.9);
        solver.sync_to_gpu();

        auto [residual, iters] = solver.solve_steady();
        solver.sync_from_gpu();

        auto err = compute_poiseuille_errors(solver, mesh, dp_dx, nu);
        errors[g] = err.l2_rel;
        std::cout << "  Ny=" << Ny << ": L2_rel=" << std::scientific << std::setprecision(3)
                  << errors[g] << " (iters=" << iters << ", res=" << residual << ")\n";
    }

    // Convergence rate: log(e1/e2) / log(h1/h2)
    double rate_1 = std::log(errors[0] / errors[1]) / std::log(2.0);
    double rate_2 = std::log(errors[1] / errors[2]) / std::log(2.0);

    std::cout << "  Convergence rate (16->32): " << std::fixed << std::setprecision(2) << rate_1 << "\n";
    std::cout << "  Convergence rate (32->64): " << rate_2 << "\n\n";

    // 2nd-order scheme should give rate >= 1.5 (allowing tolerance for BCs)
    record("Convergence rate 16->32 >= 1.5", rate_1 >= 1.5);
    record("Convergence rate 32->64 >= 1.5", rate_2 >= 1.5);
}

// ============================================================================
// Main
// ============================================================================
int main() {
    return harness::run_sections("PoiseuilleValidation", {
        {"32x64 analytical", test_poiseuille_32x64},
        {"64x128 analytical", test_poiseuille_64x128},
        {"Grid convergence", test_poiseuille_convergence},
    });
}
