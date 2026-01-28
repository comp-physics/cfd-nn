/// @file test_scheme_combinations.cpp
/// @brief Test specific scheme combinations that users are likely to use
///
/// Tests include:
/// 1. DNS Configuration: O4 + Skew + RK3
/// 2. LES Configuration: O4 + Skew + RK2
/// 3. RANS Configuration: O2 + Upwind2 + Euler
/// 4. Robust Configuration: O2 + Central + RK2
/// 5. Budget DNS: O2 + Skew + RK3
///
/// Key coverage: Scheme combination tests (previously untested)

#include "test_harness.hpp"
#include "test_utilities.hpp"
#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include <cmath>
#include <vector>
#include <string>
#include <iomanip>

using namespace nncfd;
using namespace nncfd::test::harness;

// ============================================================================
// Helper: Compute kinetic energy for 2D field
// ============================================================================
static double compute_ke_2d(const VectorField& vel, const Mesh& mesh) {
    double ke = 0.0;
    const int Nx = mesh.Nx;
    const int Ny = mesh.Ny;
    const int Ng = mesh.Nghost;
    const double dx = mesh.dx;
    const double dy = mesh.dy;

    for (int j = Ng; j < Ny + Ng; ++j) {
        for (int i = Ng; i < Nx + Ng; ++i) {
            double u_c = 0.5 * (vel.u(i, j) + vel.u(i + 1, j));
            double v_c = 0.5 * (vel.v(i, j) + vel.v(i, j + 1));
            ke += 0.5 * (u_c * u_c + v_c * v_c) * dx * dy;
        }
    }
    return ke;
}

// ============================================================================
// Helper: Compute kinetic energy for 3D field
// ============================================================================
static double compute_ke_3d(const VectorField& vel, const Mesh& mesh) {
    double ke = 0.0;
    const int Nx = mesh.Nx;
    const int Ny = mesh.Ny;
    const int Nz = mesh.Nz;
    const int Ng = mesh.Nghost;
    const double cell_vol = mesh.dx * mesh.dy * mesh.dz;

    for (int k = Ng; k < Nz + Ng; ++k) {
        for (int j = Ng; j < Ny + Ng; ++j) {
            for (int i = Ng; i < Nx + Ng; ++i) {
                double u_c = 0.5 * (vel.u(i, j, k) + vel.u(i + 1, j, k));
                double v_c = 0.5 * (vel.v(i, j, k) + vel.v(i, j + 1, k));
                double w_c = 0.5 * (vel.w(i, j, k) + vel.w(i, j, k + 1));
                ke += 0.5 * (u_c * u_c + v_c * v_c + w_c * w_c) * cell_vol;
            }
        }
    }
    return ke;
}

// ============================================================================
// Helper: Check if velocity field contains NaN or Inf
// ============================================================================
static bool check_field_validity_2d(const VectorField& vel, const Mesh& mesh) {
    const int Nx = mesh.Nx;
    const int Ny = mesh.Ny;
    const int Ng = mesh.Nghost;

    for (int j = Ng; j < Ny + Ng; ++j) {
        for (int i = Ng; i < Nx + Ng + 1; ++i) {
            if (!std::isfinite(vel.u(i, j))) return false;
        }
    }
    for (int j = Ng; j < Ny + Ng + 1; ++j) {
        for (int i = Ng; i < Nx + Ng; ++i) {
            if (!std::isfinite(vel.v(i, j))) return false;
        }
    }
    return true;
}

static bool check_field_validity_3d(const VectorField& vel, const Mesh& mesh) {
    const int Nx = mesh.Nx;
    const int Ny = mesh.Ny;
    const int Nz = mesh.Nz;
    const int Ng = mesh.Nghost;

    // Check u component
    for (int k = Ng; k < Nz + Ng; ++k) {
        for (int j = Ng; j < Ny + Ng; ++j) {
            for (int i = Ng; i < Nx + Ng + 1; ++i) {
                if (!std::isfinite(vel.u(i, j, k))) return false;
            }
        }
    }
    // Check v component
    for (int k = Ng; k < Nz + Ng; ++k) {
        for (int j = Ng; j < Ny + Ng + 1; ++j) {
            for (int i = Ng; i < Nx + Ng; ++i) {
                if (!std::isfinite(vel.v(i, j, k))) return false;
            }
        }
    }
    // Check w component
    for (int k = Ng; k < Nz + Ng + 1; ++k) {
        for (int j = Ng; j < Ny + Ng; ++j) {
            for (int i = Ng; i < Nx + Ng; ++i) {
                if (!std::isfinite(vel.w(i, j, k))) return false;
            }
        }
    }
    return true;
}

// ============================================================================
// Test 1: DNS Configuration - O4 + Skew + RK3
// ============================================================================
/// High-fidelity DNS configuration for accuracy
static void test_dns_configuration() {
    std::cout << "\n=== DNS Configuration: O4 + Skew + RK3 ===\n\n";

    const int N = 32;
    const double L = 2.0 * M_PI;
    const double nu = 1e-4;
    const double dt = 0.002;
    const int nsteps = 200;

    std::cout << "  2D TGV at N=" << N << ", " << nsteps << " steps...\n";

    Mesh mesh;
    mesh.init_uniform(N, N, 0.0, L, 0.0, L, 2);  // Ng=2 for O4

    Config config;
    config.nu = nu;
    config.dt = dt;
    config.adaptive_dt = false;
    config.space_order = 4;
    config.convective_scheme = ConvectiveScheme::Skew;
    config.time_integrator = TimeIntegrator::RK3;
    config.poisson_solver = PoissonSolverType::FFT;
    config.verbose = false;

    RANSSolver solver(mesh, config);
    test::init_taylor_green(solver, mesh);

    double ke_init = compute_ke_2d(solver.velocity(), mesh);
    double ke_max = ke_init;
    bool valid = true;
    int steps_completed = 0;

    for (int step = 0; step < nsteps && valid; ++step) {
        solver.step();
        steps_completed++;
        if (!check_field_validity_2d(solver.velocity(), mesh)) {
            valid = false;
            break;
        }
        double ke = compute_ke_2d(solver.velocity(), mesh);
        ke_max = std::max(ke_max, ke);
    }

    double ke_final = compute_ke_2d(solver.velocity(), mesh);
    double max_div = test::compute_max_divergence_2d(solver.velocity(), mesh);

    std::cout << "  Results:\n";
    std::cout << "    Steps:       " << steps_completed << "/" << nsteps << "\n";
    std::cout << "    KE ratio:    " << std::fixed << std::setprecision(4) << (ke_final / ke_init) << "\n";
    std::cout << "    KE max/init: " << (ke_max / ke_init) << "\n";
    std::cout << "    max|div|:    " << std::scientific << max_div << "\n";

    std::cout << "\nQOI_JSON: {\"test\":\"dns_o4_skew_rk3\""
              << ",\"ke_ratio\":" << (ke_final / ke_init)
              << ",\"max_div\":" << max_div
              << "}\n" << std::flush;

    bool stable = valid && steps_completed == nsteps;
    bool energy_ok = (ke_max / ke_init) < 1.01;
    bool div_ok = max_div < 1e-8;

    record("[DNS O4+Skew+RK3] Simulation stable", stable);
    record("[DNS O4+Skew+RK3] Energy conserving", energy_ok);
    record("[DNS O4+Skew+RK3] Divergence-free", div_ok);
}

// ============================================================================
// Test 2: LES Configuration - O4 + Skew + RK2
// ============================================================================
/// Practical LES configuration (slightly cheaper than DNS)
static void test_les_configuration() {
    std::cout << "\n=== LES Configuration: O4 + Skew + RK2 ===\n\n";

    const int N = 32;
    const double L = 2.0 * M_PI;
    const double nu = 1e-4;
    const double dt = 0.002;
    const int nsteps = 200;

    std::cout << "  2D TGV at N=" << N << ", " << nsteps << " steps...\n";

    Mesh mesh;
    mesh.init_uniform(N, N, 0.0, L, 0.0, L, 2);  // Ng=2 for O4

    Config config;
    config.nu = nu;
    config.dt = dt;
    config.adaptive_dt = false;
    config.space_order = 4;
    config.convective_scheme = ConvectiveScheme::Skew;
    config.time_integrator = TimeIntegrator::RK2;
    config.poisson_solver = PoissonSolverType::FFT;
    config.verbose = false;

    RANSSolver solver(mesh, config);
    test::init_taylor_green(solver, mesh);

    double ke_init = compute_ke_2d(solver.velocity(), mesh);
    bool valid = true;
    int steps_completed = 0;

    for (int step = 0; step < nsteps && valid; ++step) {
        solver.step();
        steps_completed++;
        if (!check_field_validity_2d(solver.velocity(), mesh)) {
            valid = false;
        }
    }

    double ke_final = compute_ke_2d(solver.velocity(), mesh);
    double max_div = test::compute_max_divergence_2d(solver.velocity(), mesh);

    std::cout << "  Results:\n";
    std::cout << "    Steps:       " << steps_completed << "/" << nsteps << "\n";
    std::cout << "    KE ratio:    " << std::fixed << std::setprecision(4) << (ke_final / ke_init) << "\n";
    std::cout << "    max|div|:    " << std::scientific << max_div << "\n";

    std::cout << "\nQOI_JSON: {\"test\":\"les_o4_skew_rk2\""
              << ",\"ke_ratio\":" << (ke_final / ke_init)
              << ",\"max_div\":" << max_div
              << "}\n" << std::flush;

    bool stable = valid && steps_completed == nsteps;
    bool div_ok = max_div < 1e-8;

    record("[LES O4+Skew+RK2] Simulation stable", stable);
    record("[LES O4+Skew+RK2] Divergence-free", div_ok);
}

// ============================================================================
// Test 3: RANS Configuration - O2 + Upwind2 + Euler
// ============================================================================
/// Robust RANS configuration (dissipative but stable)
static void test_rans_configuration() {
    std::cout << "\n=== RANS Configuration: O2 + Upwind2 + Euler ===\n\n";

    const int N = 32;
    const double L = 2.0 * M_PI;
    const double nu = 1e-3;  // Higher viscosity for RANS-like behavior
    const double dt = 0.002;
    const int nsteps = 200;

    std::cout << "  2D TGV at N=" << N << ", " << nsteps << " steps...\n";

    Mesh mesh;
    mesh.init_uniform(N, N, 0.0, L, 0.0, L, 2);  // Ng=2 for Upwind2

    Config config;
    config.nu = nu;
    config.dt = dt;
    config.adaptive_dt = false;
    config.space_order = 2;
    config.convective_scheme = ConvectiveScheme::Upwind2;
    config.time_integrator = TimeIntegrator::Euler;
    config.poisson_solver = PoissonSolverType::MG;
    config.poisson_fixed_cycles = 8;
    config.verbose = false;

    RANSSolver solver(mesh, config);
    test::init_taylor_green(solver, mesh);

    double ke_init = compute_ke_2d(solver.velocity(), mesh);
    bool valid = true;
    int steps_completed = 0;

    for (int step = 0; step < nsteps && valid; ++step) {
        solver.step();
        steps_completed++;
        if (!check_field_validity_2d(solver.velocity(), mesh)) {
            valid = false;
        }
    }

    double ke_final = compute_ke_2d(solver.velocity(), mesh);
    double max_div = test::compute_max_divergence_2d(solver.velocity(), mesh);
    double ke_ratio = ke_final / ke_init;

    std::cout << "  Results:\n";
    std::cout << "    Steps:       " << steps_completed << "/" << nsteps << "\n";
    std::cout << "    KE ratio:    " << std::fixed << std::setprecision(4) << ke_ratio << "\n";
    std::cout << "    max|div|:    " << std::scientific << max_div << "\n";

    std::cout << "\nQOI_JSON: {\"test\":\"rans_o2_upwind2_euler\""
              << ",\"ke_ratio\":" << ke_ratio
              << ",\"max_div\":" << max_div
              << "}\n" << std::flush;

    bool stable = valid && steps_completed == nsteps;
    bool div_ok = max_div < 1e-6;
    // Note: Upwind2 is dissipative but with low nu and short runs the effect is minimal
    // The key check is that the combination runs stably without blowing up
    record("[RANS O2+Upwind2+Euler] Simulation stable", stable);
    record("[RANS O2+Upwind2+Euler] Divergence-free", div_ok);
}

// ============================================================================
// Test 4: Robust Configuration - O2 + Central + RK2
// ============================================================================
/// Balanced accuracy/stability configuration
static void test_robust_configuration() {
    std::cout << "\n=== Robust Configuration: O2 + Central + RK2 ===\n\n";

    const int N = 32;
    const double L = 2.0 * M_PI;
    const double nu = 1e-4;
    const double dt = 0.002;
    const int nsteps = 500;

    std::cout << "  2D TGV at N=" << N << ", " << nsteps << " steps...\n";

    Mesh mesh;
    mesh.init_uniform(N, N, 0.0, L, 0.0, L, 1);  // Ng=1 for O2

    Config config;
    config.nu = nu;
    config.dt = dt;
    config.adaptive_dt = false;
    config.space_order = 2;
    config.convective_scheme = ConvectiveScheme::Central;
    config.time_integrator = TimeIntegrator::RK2;
    config.poisson_solver = PoissonSolverType::MG;
    config.poisson_fixed_cycles = 8;
    config.verbose = false;

    RANSSolver solver(mesh, config);
    test::init_taylor_green(solver, mesh);

    double ke_init = compute_ke_2d(solver.velocity(), mesh);
    bool valid = true;
    int steps_completed = 0;

    for (int step = 0; step < nsteps && valid; ++step) {
        solver.step();
        steps_completed++;
        if (!check_field_validity_2d(solver.velocity(), mesh)) {
            valid = false;
        }
    }

    double ke_final = compute_ke_2d(solver.velocity(), mesh);
    double max_div = test::compute_max_divergence_2d(solver.velocity(), mesh);

    std::cout << "  Results:\n";
    std::cout << "    Steps:       " << steps_completed << "/" << nsteps << "\n";
    std::cout << "    KE ratio:    " << std::fixed << std::setprecision(4) << (ke_final / ke_init) << "\n";
    std::cout << "    max|div|:    " << std::scientific << max_div << "\n";

    std::cout << "\nQOI_JSON: {\"test\":\"robust_o2_central_rk2\""
              << ",\"ke_ratio\":" << (ke_final / ke_init)
              << ",\"max_div\":" << max_div
              << "}\n" << std::flush;

    bool stable = valid && steps_completed == nsteps;
    bool div_ok = max_div < 1e-6;

    record("[Robust O2+Central+RK2] Long-time stable (500 steps)", stable);
    record("[Robust O2+Central+RK2] Divergence-free", div_ok);
}

// ============================================================================
// Test 5: Budget DNS - O2 + Skew + RK3
// ============================================================================
/// Budget DNS configuration (Skew for energy conservation, O2 for speed)
static void test_budget_dns() {
    std::cout << "\n=== Budget DNS: O2 + Skew + RK3 ===\n\n";

    const int N = 32;
    const double L = 2.0 * M_PI;
    const double nu = 1e-4;
    const double dt = 0.002;
    const int nsteps = 200;

    std::cout << "  2D TGV at N=" << N << ", " << nsteps << " steps...\n";

    Mesh mesh;
    mesh.init_uniform(N, N, 0.0, L, 0.0, L, 1);  // Ng=1 for O2

    Config config;
    config.nu = nu;
    config.dt = dt;
    config.adaptive_dt = false;
    config.space_order = 2;
    config.convective_scheme = ConvectiveScheme::Skew;
    config.time_integrator = TimeIntegrator::RK3;
    config.poisson_solver = PoissonSolverType::MG;
    config.poisson_fixed_cycles = 8;
    config.verbose = false;

    RANSSolver solver(mesh, config);
    test::init_taylor_green(solver, mesh);

    double ke_init = compute_ke_2d(solver.velocity(), mesh);
    double ke_max = ke_init;
    bool valid = true;
    int steps_completed = 0;

    for (int step = 0; step < nsteps && valid; ++step) {
        solver.step();
        steps_completed++;
        if (!check_field_validity_2d(solver.velocity(), mesh)) {
            valid = false;
            break;
        }
        double ke = compute_ke_2d(solver.velocity(), mesh);
        ke_max = std::max(ke_max, ke);
    }

    double ke_final = compute_ke_2d(solver.velocity(), mesh);
    double max_div = test::compute_max_divergence_2d(solver.velocity(), mesh);

    std::cout << "  Results:\n";
    std::cout << "    Steps:       " << steps_completed << "/" << nsteps << "\n";
    std::cout << "    KE ratio:    " << std::fixed << std::setprecision(4) << (ke_final / ke_init) << "\n";
    std::cout << "    KE max/init: " << (ke_max / ke_init) << "\n";
    std::cout << "    max|div|:    " << std::scientific << max_div << "\n";

    std::cout << "\nQOI_JSON: {\"test\":\"budget_dns_o2_skew_rk3\""
              << ",\"ke_ratio\":" << (ke_final / ke_init)
              << ",\"max_div\":" << max_div
              << "}\n" << std::flush;

    bool stable = valid && steps_completed == nsteps;
    bool energy_ok = (ke_max / ke_init) < 1.01;  // Skew should prevent growth
    bool div_ok = max_div < 1e-6;

    record("[Budget DNS O2+Skew+RK3] Simulation stable", stable);
    record("[Budget DNS O2+Skew+RK3] Energy conserving", energy_ok);
    record("[Budget DNS O2+Skew+RK3] Divergence-free", div_ok);
}

// ============================================================================
// Test 6: 3D DNS Configuration
// ============================================================================
/// 3D DNS with O4 + Skew + RK3
static void test_3d_dns() {
    std::cout << "\n=== 3D DNS: O4 + Skew + RK3 ===\n\n";

    const int N = 16;
    const double L = 2.0 * M_PI;
    const double nu = 1e-3;
    const double dt = 0.005;
    const int nsteps = 50;

    std::cout << "  3D TGV at N=" << N << "³, " << nsteps << " steps...\n";

    Mesh mesh;
    mesh.init_uniform(N, N, N, 0.0, L, 0.0, L, 0.0, L, 2);  // Ng=2 for O4

    Config config;
    config.nu = nu;
    config.dt = dt;
    config.adaptive_dt = false;
    config.space_order = 4;
    config.convective_scheme = ConvectiveScheme::Skew;
    config.time_integrator = TimeIntegrator::RK3;
    config.poisson_solver = PoissonSolverType::FFT;
    config.verbose = false;

    RANSSolver solver(mesh, config);

    // Initialize 3D TGV (w=0 by default from VectorField constructor)
    // Use < bounds consistent with check_field_validity_3d
    const int Ng = mesh.Nghost;
    // u on x-faces
    for (int k = Ng; k < N + Ng; ++k) {
        double z = mesh.z(k);
        for (int j = Ng; j < N + Ng; ++j) {
            double y = mesh.y(j);
            for (int i = Ng; i < N + Ng + 1; ++i) {
                double x = mesh.xf[i];
                solver.velocity().u(i, j, k) = std::sin(x) * std::cos(y) * std::cos(z);
            }
        }
    }
    // v on y-faces
    for (int k = Ng; k < N + Ng; ++k) {
        double z = mesh.z(k);
        for (int j = Ng; j < N + Ng + 1; ++j) {
            double y = mesh.yf[j];
            for (int i = Ng; i < N + Ng; ++i) {
                double x = mesh.x(i);
                solver.velocity().v(i, j, k) = -std::cos(x) * std::sin(y) * std::cos(z);
            }
        }
    }
    // w = 0 initially (already zero from VectorField init)

    double ke_init = compute_ke_3d(solver.velocity(), mesh);
    bool valid = true;
    int steps_completed = 0;

    for (int step = 0; step < nsteps && valid; ++step) {
        solver.step();
        steps_completed++;
        if (!check_field_validity_3d(solver.velocity(), mesh)) {
            valid = false;
        }
    }

    double ke_final = compute_ke_3d(solver.velocity(), mesh);

    std::cout << "  Results:\n";
    std::cout << "    Steps:       " << steps_completed << "/" << nsteps << "\n";
    std::cout << "    KE ratio:    " << std::fixed << std::setprecision(4) << (ke_final / ke_init) << "\n";

    std::cout << "\nQOI_JSON: {\"test\":\"3d_dns_o4_skew_rk3\""
              << ",\"ke_ratio\":" << (ke_final / ke_init)
              << "}\n" << std::flush;

    bool stable = valid && steps_completed == nsteps;
    // 3D TGV decays due to viscosity
    bool energy_ok = (ke_final / ke_init) > 0.1 && (ke_final / ke_init) < 1.1;

    record("[3D DNS O4+Skew+RK3] Simulation stable", stable);
    record("[3D DNS O4+Skew+RK3] Energy bounded", energy_ok);
}

// ============================================================================
// Test 7: Mixed Upwind Comparison
// ============================================================================
/// Compare Upwind vs Upwind2 in otherwise identical configs
static void test_upwind_comparison() {
    std::cout << "\n=== Upwind vs Upwind2 Comparison ===\n\n";

    const int N = 32;
    const double L = 2.0 * M_PI;
    const double nu = 1e-3;
    const double dt = 0.002;
    const int nsteps = 300;

    double ke_init, ke_upwind1, ke_upwind2;

    // Upwind (1st order)
    {
        Mesh mesh;
        mesh.init_uniform(N, N, 0.0, L, 0.0, L, 1);

        Config config;
        config.nu = nu;
        config.dt = dt;
        config.adaptive_dt = false;
        config.convective_scheme = ConvectiveScheme::Upwind;
        config.time_integrator = TimeIntegrator::RK2;
        config.poisson_solver = PoissonSolverType::MG;
        config.verbose = false;

        RANSSolver solver(mesh, config);
        test::init_taylor_green(solver, mesh);
        ke_init = compute_ke_2d(solver.velocity(), mesh);

        for (int step = 0; step < nsteps; ++step) {
            solver.step();
        }
        ke_upwind1 = compute_ke_2d(solver.velocity(), mesh);
    }

    // Upwind2 (2nd order)
    {
        Mesh mesh;
        mesh.init_uniform(N, N, 0.0, L, 0.0, L, 2);  // Ng=2

        Config config;
        config.nu = nu;
        config.dt = dt;
        config.adaptive_dt = false;
        config.convective_scheme = ConvectiveScheme::Upwind2;
        config.time_integrator = TimeIntegrator::RK2;
        config.poisson_solver = PoissonSolverType::MG;
        config.verbose = false;

        RANSSolver solver(mesh, config);
        test::init_taylor_green(solver, mesh);

        for (int step = 0; step < nsteps; ++step) {
            solver.step();
        }
        ke_upwind2 = compute_ke_2d(solver.velocity(), mesh);
    }

    double r_upwind1 = ke_upwind1 / ke_init;
    double r_upwind2 = ke_upwind2 / ke_init;

    std::cout << "  Results after " << nsteps << " steps:\n";
    std::cout << "    Upwind  KE ratio: " << std::fixed << std::setprecision(4) << r_upwind1 << "\n";
    std::cout << "    Upwind2 KE ratio: " << r_upwind2 << "\n";
    std::cout << "    Upwind2 retains " << (r_upwind2 / r_upwind1) << "x more KE\n";

    std::cout << "\nQOI_JSON: {\"test\":\"upwind_comparison\""
              << ",\"r_upwind1\":" << r_upwind1
              << ",\"r_upwind2\":" << r_upwind2
              << "}\n" << std::flush;

    // Upwind2 should retain more energy (less dissipative)
    bool upwind2_better = r_upwind2 >= r_upwind1 * 0.99;  // Allow small tolerance

    record("[Upwind2] Less dissipative than Upwind", upwind2_better);
}

// ============================================================================
// Main
// ============================================================================
int main() {
    return run("SchemeCombinationsTest", []() {
        test_dns_configuration();
        test_les_configuration();
        test_rans_configuration();
        test_robust_configuration();
        test_budget_dns();
        test_3d_dns();
        test_upwind_comparison();
    });
}
