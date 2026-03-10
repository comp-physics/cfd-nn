/// @file test_stretched_gradient.cpp
/// @brief Verify du/dy gradient accuracy on stretched (non-uniform) y-grids
///
/// Validates:
///   1. du/dy of cos(pi*y/2) on 2D stretched grid matches analytical
///   2. du/dy of Poiseuille profile on 2D stretched grid matches analytical
///   3. 3D stretched gradient matches 2D (z-independence)
///   4. Grid convergence order ~2 for gradient computation

#include "test_harness.hpp"
#include "test_utilities.hpp"
#include "solver.hpp"
#include "config.hpp"
#include "mesh.hpp"
#include <cmath>
#include <vector>
#include <iomanip>

using namespace nncfd;
using namespace nncfd::test;
using nncfd::test::harness::record;

// ============================================================================
// Helper: compute L2 relative error of du/dy on a 2D stretched grid
// ============================================================================
namespace {

/// Set u-field to a known profile u(y) on a 2D mesh.
/// u lives at x-faces (i+1/2, j), where j corresponds to cell-center y = yc[j].
void set_u_profile_2d(VectorField& vel, const Mesh& mesh,
                      std::function<double(double)> u_of_y) {
    for (int j = 0; j < mesh.total_Ny(); ++j) {
        double y = mesh.yc[j];
        double u_val = u_of_y(y);
        // u is stored at x-faces: indices run from 0 to Nx+2*Ng (i.e. Nx+1 interior faces + ghost)
        for (int i = 0; i < mesh.Nx + 1 + 2 * mesh.Nghost; ++i) {
            vel.u(i, j) = u_val;
        }
    }
    // Set v to zero everywhere
    for (int j = 0; j < mesh.Ny + 1 + 2 * mesh.Nghost; ++j) {
        for (int i = 0; i < mesh.total_Nx(); ++i) {
            vel.v(i, j) = 0.0;
        }
    }
}

/// Set u-field to a known profile u(y) on a 3D mesh.
void set_u_profile_3d(VectorField& vel, const Mesh& mesh,
                      std::function<double(double)> u_of_y) {
    for (int k = 0; k < mesh.total_Nz(); ++k) {
        for (int j = 0; j < mesh.total_Ny(); ++j) {
            double y = mesh.yc[j];
            double u_val = u_of_y(y);
            for (int i = 0; i < mesh.Nx + 1 + 2 * mesh.Nghost; ++i) {
                vel.u(i, j, k) = u_val;
            }
        }
    }
    // Set v and w to zero
    for (int k = 0; k < mesh.total_Nz(); ++k) {
        for (int j = 0; j < mesh.Ny + 1 + 2 * mesh.Nghost; ++j) {
            for (int i = 0; i < mesh.total_Nx(); ++i) {
                vel.v(i, j, k) = 0.0;
            }
        }
    }
    for (int k = 0; k < mesh.Nz + 1 + 2 * mesh.Nghost; ++k) {
        for (int j = 0; j < mesh.total_Ny(); ++j) {
            for (int i = 0; i < mesh.total_Nx(); ++i) {
                vel.w(i, j, k) = 0.0;
            }
        }
    }
}

/// Compute du/dy L2 relative error on a 2D stretched grid.
/// du/dy is computed at y-face j (between cell j-1 and cell j):
///   dudy_num(j) = (u_avg[j] - u_avg[j-1]) / dyc[j]
/// where u_avg[j] averages u over x at cell row j, and
/// dyc[j] = yc[j] - yc[j-1] is the center-to-center spacing.
/// The analytical du/dy is evaluated at yf[j] (the y-face location).
double compute_dudy_l2_error_2d(const VectorField& vel, const Mesh& mesh,
                                 std::function<double(double)> dudy_exact) {
    // Average u over x for each cell row j (interior only)
    int ny_total = mesh.total_Ny();
    std::vector<double> u_avg(ny_total, 0.0);

    for (int j = 0; j < ny_total; ++j) {
        double u_sum = 0.0;
        int count = 0;
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            u_sum += vel.u(i, j);
            ++count;
        }
        u_avg[j] = u_sum / count;
    }

    // Compute du/dy at interior y-faces and compare to analytical
    // Interior faces: j_begin() to j_end() gives cell indices.
    // du/dy at face between cell j-1 and cell j: for j in [j_begin(), j_end()]
    // That gives Ny interior faces (between the Ny interior cells, plus boundary faces).
    // We evaluate at faces j_begin() to j_end() (Ny faces between Ny cells + 1 boundary).
    // Actually, the gradient at face j is between cells j-1 and j.
    // Interior faces that have valid cells on both sides: j in [j_begin(), j_end()]
    double l2_num = 0.0;
    double l2_den = 0.0;

    for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
        // dyc[j] = yc[j] - yc[j-1]
        double dyc_j = mesh.yc[j] - mesh.yc[j - 1];
        double dudy_num = (u_avg[j] - u_avg[j - 1]) / dyc_j;
        double y_face = mesh.yf[j];
        double dudy_an = dudy_exact(y_face);

        double err = dudy_num - dudy_an;
        l2_num += err * err;
        l2_den += dudy_an * dudy_an;
    }

    return std::sqrt(l2_num / (l2_den + 1e-30));
}

/// Compute du/dy L2 relative error on a 3D stretched grid (averages over x and z).
double compute_dudy_l2_error_3d(const VectorField& vel, const Mesh& mesh,
                                 std::function<double(double)> dudy_exact) {
    int ny_total = mesh.total_Ny();
    std::vector<double> u_avg(ny_total, 0.0);

    for (int j = 0; j < ny_total; ++j) {
        double u_sum = 0.0;
        int count = 0;
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                u_sum += vel.u(i, j, k);
                ++count;
            }
        }
        u_avg[j] = u_sum / count;
    }

    double l2_num = 0.0;
    double l2_den = 0.0;

    for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
        double dyc_j = mesh.yc[j] - mesh.yc[j - 1];
        double dudy_num = (u_avg[j] - u_avg[j - 1]) / dyc_j;
        double y_face = mesh.yf[j];
        double dudy_an = dudy_exact(y_face);

        double err = dudy_num - dudy_an;
        l2_num += err * err;
        l2_den += dudy_an * dudy_an;
    }

    return std::sqrt(l2_num / (l2_den + 1e-30));
}

} // anonymous namespace

// ============================================================================
// Test 1: du/dy of cos(pi*y/2) on 2D stretched grid
// ============================================================================
static double l2_error_cos_2d_Ny64 = 0.0;  // saved for 3D comparison

void test_cos_gradient_2d() {
    std::cout << "\n  Setting up 2D stretched grid (32 x 64, beta=2.0)\n";

    Mesh mesh;
    mesh.init_stretched_y(32, 64, 0.0, 4.0 * M_PI, -1.0, 1.0,
                          Mesh::tanh_stretching(2.0));

    Config config;
    config.nu = 0.01;
    config.dt = 0.001;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);
    solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel2D));
    solver.set_body_force(1.0, 0.0);
    solver.initialize_uniform(0.0, 0.0);

    // Set u(y) = cos(pi*y/2) for y in [-1, 1]
    auto u_cos = [](double y) { return std::cos(M_PI * y / 2.0); };
    set_u_profile_2d(solver.velocity(), mesh, u_cos);

    // Analytical: du/dy = -pi/2 * sin(pi*y/2)
    auto dudy_cos = [](double y) { return -M_PI / 2.0 * std::sin(M_PI * y / 2.0); };

    double l2_err = compute_dudy_l2_error_2d(solver.velocity(), mesh, dudy_cos);
    l2_error_cos_2d_Ny64 = l2_err;

    std::cout << "  cos(pi*y/2) du/dy L2 relative error: "
              << std::scientific << std::setprecision(4) << l2_err << "\n";

    record("2D cos gradient L2 error < 5%", l2_err < 0.05);
}

// ============================================================================
// Test 2: du/dy of Poiseuille profile on 2D stretched grid
// ============================================================================
void test_poiseuille_gradient_2d() {
    std::cout << "\n  Setting up 2D stretched grid (32 x 64, beta=2.0)\n";

    const double nu = 0.01;
    const double dp_dx = -1.0;

    Mesh mesh;
    mesh.init_stretched_y(32, 64, 0.0, 4.0 * M_PI, -1.0, 1.0,
                          Mesh::tanh_stretching(2.0));

    Config config;
    config.nu = nu;
    config.dt = 0.001;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);
    solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel2D));
    solver.set_body_force(-dp_dx, 0.0);
    solver.initialize_uniform(0.0, 0.0);

    // Poiseuille: u(y) = |dp_dx| / (2*nu) * (1 - y^2) for y in [-1, 1]
    double coeff = std::abs(dp_dx) / (2.0 * nu);
    auto u_pois = [coeff](double y) { return coeff * (1.0 - y * y); };
    set_u_profile_2d(solver.velocity(), mesh, u_pois);

    // Analytical: du/dy = |dp_dx| / (2*nu) * (-2*y) = -|dp_dx| * y / nu
    double grad_coeff = std::abs(dp_dx) / nu;
    auto dudy_pois = [grad_coeff](double y) { return -grad_coeff * y; };

    double l2_err = compute_dudy_l2_error_2d(solver.velocity(), mesh, dudy_pois);

    std::cout << "  Poiseuille du/dy L2 relative error: "
              << std::scientific << std::setprecision(4) << l2_err << "\n";

    record("2D Poiseuille gradient L2 error < 5%", l2_err < 0.05);
}

// ============================================================================
// Test 3: du/dy of cos(pi*y/2) on 3D stretched grid (should match 2D)
// ============================================================================
void test_cos_gradient_3d() {
    std::cout << "\n  Setting up 3D stretched grid (16 x 64 x 16, beta=2.0)\n";

    Mesh mesh;
    mesh.init_stretched_y(16, 64, 16, 0.0, 4.0 * M_PI, -1.0, 1.0,
                          0.0, 2.0 * M_PI, Mesh::tanh_stretching(2.0));

    Config config;
    config.nu = 0.01;
    config.dt = 0.001;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;

    RANSSolver solver(mesh, config);
    solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel3D));
    solver.set_body_force(1.0, 0.0, 0.0);
    solver.initialize_uniform(0.0, 0.0);

    // Set u(y) = cos(pi*y/2) for y in [-1, 1]
    auto u_cos = [](double y) { return std::cos(M_PI * y / 2.0); };
    set_u_profile_3d(solver.velocity(), mesh, u_cos);

    // Analytical: du/dy = -pi/2 * sin(pi*y/2)
    auto dudy_cos = [](double y) { return -M_PI / 2.0 * std::sin(M_PI * y / 2.0); };

    double l2_3d = compute_dudy_l2_error_3d(solver.velocity(), mesh, dudy_cos);
    double l2_2d = l2_error_cos_2d_Ny64;

    std::cout << "  3D cos(pi*y/2) du/dy L2 relative error: "
              << std::scientific << std::setprecision(4) << l2_3d << "\n";
    std::cout << "  2D reference L2 error: " << l2_2d << "\n";

    double rel_diff = std::abs(l2_3d - l2_2d) / (l2_2d + 1e-30);
    std::cout << "  Relative difference 3D vs 2D: "
              << std::fixed << std::setprecision(2) << rel_diff * 100.0 << "%\n";

    record("3D cos gradient L2 error < 5%", l2_3d < 0.05);
    record("3D gradient matches 2D (within 10%)", rel_diff < 0.10);
}

// ============================================================================
// Test 4: Grid convergence of du/dy (expect ~2nd order)
// ============================================================================
void test_gradient_convergence() {
    std::cout << "\n  Grid convergence study for du/dy on stretched grids\n\n";

    const int ny_grids[] = {32, 64, 128};
    double errors[3] = {};

    auto u_cos = [](double y) { return std::cos(M_PI * y / 2.0); };
    auto dudy_cos = [](double y) { return -M_PI / 2.0 * std::sin(M_PI * y / 2.0); };

    for (int g = 0; g < 3; ++g) {
        int Ny = ny_grids[g];
        int Nx = Ny / 2;

        Mesh mesh;
        mesh.init_stretched_y(Nx, Ny, 0.0, 4.0 * M_PI, -1.0, 1.0,
                              Mesh::tanh_stretching(2.0));

        Config config;
        config.nu = 0.01;
        config.dt = 0.001;
        config.turb_model = TurbulenceModelType::None;
        config.verbose = false;

        RANSSolver solver(mesh, config);
        solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel2D));
        solver.set_body_force(1.0, 0.0);
        solver.initialize_uniform(0.0, 0.0);

        set_u_profile_2d(solver.velocity(), mesh, u_cos);
        errors[g] = compute_dudy_l2_error_2d(solver.velocity(), mesh, dudy_cos);

        std::cout << "  Ny=" << std::setw(4) << Ny
                  << ": L2 error = " << std::scientific << std::setprecision(4)
                  << errors[g] << "\n";
    }

    // Convergence rate: log(e1/e2) / log(h1/h2)
    // All grids double in Ny, so h ratio = 2
    double rate_1 = std::log(errors[0] / errors[1]) / std::log(2.0);
    double rate_2 = std::log(errors[1] / errors[2]) / std::log(2.0);
    double avg_rate = 0.5 * (rate_1 + rate_2);

    std::cout << "\n  Convergence rate (32->64):  " << std::fixed << std::setprecision(2) << rate_1 << "\n";
    std::cout << "  Convergence rate (64->128): " << rate_2 << "\n";
    std::cout << "  Average convergence order:  " << avg_rate << "\n";

    record("Gradient convergence order > 1.5", avg_rate > 1.5);
}

// ============================================================================
// Main
// ============================================================================
int main() {
    return harness::run_sections("StretchedGradient", {
        {"2D cosine gradient", test_cos_gradient_2d},
        {"2D Poiseuille gradient", test_poiseuille_gradient_2d},
        {"3D cosine gradient", test_cos_gradient_3d},
        {"Gradient convergence", test_gradient_convergence},
    });
}
