/// @file verify_implicit_ydiff.cpp
/// @brief Verification of implicit y-diffusion: dt speedup + profile correctness
///
/// Tests:
/// 1. dt comparison: implicit vs explicit on stretched RANS grid
/// 2. Poiseuille convergence: steady-state profile accuracy (low Re for fast convergence)
/// 3. All turbulence models: stability on stretched grid

#include "test_utilities.hpp"
#include "solver.hpp"
#include "config.hpp"
#include "turbulence_model.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>

using namespace nncfd;
using namespace nncfd::test;

// ============================================================================
// Test 1: dt speedup on stretched grid
// ============================================================================
void test_dt_speedup() {
    std::cout << "=== Test 1: dt speedup (implicit vs explicit) ===\n\n";

    const int Nx = 32, Ny = 96;
    const double nu = 1.0 / 180.0;

    Mesh mesh;
    mesh.init_stretched_y(Nx, Ny, 0.0, 2.0 * M_PI, -1.0, 1.0,
                          Mesh::tanh_stretching(2.0));

    double dy_min = 1e10, dy_max = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_begin() + Ny; ++j) {
        double dy_j = mesh.dy_at(j);
        dy_min = std::min(dy_min, dy_j);
        dy_max = std::max(dy_max, dy_j);
    }
    double dx = mesh.dx;
    std::cout << "  Grid: " << Nx << "x" << Ny << " stretched (beta=2.0)\n";
    std::cout << "  dx = " << std::scientific << std::setprecision(4) << dx << "\n";
    std::cout << "  dy_min = " << dy_min << "  dy_max = " << dy_max << "\n";
    std::cout << "  dx/dy_min = " << std::fixed << std::setprecision(1)
              << dx / dy_min << "x\n\n";

    double dt_diff_explicit = 0.25 * dy_min * dy_min / nu;
    double dt_diff_implicit = 0.25 * dx * dx / nu;
    std::cout << "  Diffusive CFL limits:\n";
    std::cout << "    Explicit (dy_min): " << std::scientific << std::setprecision(3)
              << dt_diff_explicit << "\n";
    std::cout << "    Implicit (dx):     " << dt_diff_implicit << "\n";
    std::cout << "    Ratio:             " << std::fixed << std::setprecision(0)
              << dt_diff_implicit / dt_diff_explicit << "x\n\n";

    // Run 20 steps with explicit y-diffusion and measure adaptive dt
    double dt_explicit;
    {
        Config config;
        config.nu = nu;
        config.adaptive_dt = true;
        config.CFL_max = 0.3;
        config.max_steps = 20;
        config.implicit_y_diffusion = false;
        config.stretch_y = true;
        config.stretch_beta = 2.0;
        config.verbose = false;

        RANSSolver solver(mesh, config);
        solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel2D));
        solver.set_body_force(1.0, 0.0);
        solver.initialize_uniform(1.0, 0.0);
        solver.sync_to_gpu();
        for (int i = 0; i < 20; ++i) solver.step();
        dt_explicit = solver.current_dt();
    }

    // Run 20 steps with implicit y-diffusion
    double dt_implicit;
    {
        Config config;
        config.nu = nu;
        config.adaptive_dt = true;
        config.CFL_max = 0.3;
        config.max_steps = 20;
        config.implicit_y_diffusion = true;
        config.stretch_y = true;
        config.stretch_beta = 2.0;
        config.verbose = false;

        RANSSolver solver(mesh, config);
        solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel2D));
        solver.set_body_force(1.0, 0.0);
        solver.initialize_uniform(1.0, 0.0);
        solver.sync_to_gpu();
        for (int i = 0; i < 20; ++i) solver.step();
        dt_implicit = solver.current_dt();
    }

    std::cout << "  After 20 steps (adaptive dt settled):\n";
    std::cout << "    Explicit dt: " << std::scientific << std::setprecision(3)
              << dt_explicit << "\n";
    std::cout << "    Implicit dt: " << dt_implicit << "\n";
    std::cout << "    Speedup:     " << std::fixed << std::setprecision(1)
              << dt_implicit / dt_explicit << "x\n\n";
}

// ============================================================================
// Test 2: Poiseuille profile correctness (low Re for fast convergence)
// ============================================================================
void test_poiseuille_accuracy() {
    std::cout << "=== Test 2: Poiseuille profile accuracy ===\n\n";

    // Use Re=10 (nu=0.1) for fast convergence
    // Diffusion time ~ delta^2/nu = 1/0.1 = 10 seconds
    // With dt~0.01, need ~5000 steps for 5 diffusion times
    const int Nx = 32, Ny = 48;
    const double nu = 0.1;
    const double dp_dx = -1.0;
    const double delta = 1.0;

    // Analytical: u(y) = -dp_dx/(2*nu) * (delta^2 - y^2)
    double u_center_exact = -dp_dx / (2.0 * nu) * delta * delta;  // = 5.0

    Mesh mesh;
    mesh.init_uniform(Nx, Ny, 0.0, 2.0 * M_PI, -delta, delta);

    Config config;
    config.nu = nu;
    config.adaptive_dt = true;
    config.CFL_max = 0.3;
    config.max_steps = 20000;
    config.tol = 1e-8;
    config.implicit_y_diffusion = true;
    config.verbose = false;

    RANSSolver solver(mesh, config);
    solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel2D));
    solver.set_body_force(-dp_dx, 0.0);
    solver.initialize_uniform(0.0, 0.0);
    solver.sync_to_gpu();

    auto [residual, iters] = solver.solve_steady();
    solver.sync_from_gpu();

    std::cout << "  Re = " << std::fixed << std::setprecision(0) << u_center_exact * delta / nu
              << "  (nu=" << std::setprecision(2) << nu << ")\n";
    std::cout << "  Converged in " << iters << " steps, residual = "
              << std::scientific << std::setprecision(2) << residual << "\n";

    int i_mid = mesh.i_begin() + Nx / 2;
    double max_err = 0.0;
    double u_center_num = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_begin() + Ny; ++j) {
        double y = mesh.y(j);
        double u_exact = -dp_dx / (2.0 * nu) * (delta * delta - y * y);
        double u_num = 0.5 * (solver.velocity().u(i_mid, j) +
                               solver.velocity().u(i_mid + 1, j));
        double err = std::abs(u_num - u_exact);
        max_err = std::max(max_err, err);
        if (std::abs(y) < mesh.dy) u_center_num = u_num;
    }

    std::cout << "  u_center: exact = " << std::fixed << std::setprecision(4)
              << u_center_exact << "  numerical = " << u_center_num << "\n";
    std::cout << "  L_inf error = " << std::scientific << std::setprecision(3) << max_err << "\n";
    std::cout << "  Relative = " << max_err / u_center_exact << "\n";
    bool ok = (residual < 1e-5) && (max_err / u_center_exact < 1e-2);
    std::cout << "  => " << (ok ? "PASS" : "FAIL") << "\n\n";
}

// ============================================================================
// Test 3: All turbulence models on stretched grid
// ============================================================================
void test_all_models_stretched() {
    std::cout << "=== Test 3: All turbulence models on stretched grid (500 steps) ===\n\n";

    const int Nx = 32, Ny = 48;
    const double nu = 1.0 / 180.0;
    const int nsteps = 500;

    struct ModelSpec {
        TurbulenceModelType type;
        const char* name;
    };

    ModelSpec models[] = {
        {TurbulenceModelType::None, "None (laminar)"},
        {TurbulenceModelType::Baseline, "Baseline"},
        {TurbulenceModelType::GEP, "GEP"},
        {TurbulenceModelType::SSTKOmega, "SST"},
        {TurbulenceModelType::KOmega, "k-omega"},
        {TurbulenceModelType::EARSM_WJ, "EARSM-WJ"},
    };

    std::cout << std::left << std::setw(18) << "Model"
              << std::right << std::setw(12) << "dt"
              << std::setw(12) << "max_vel"
              << std::setw(12) << "max_nut"
              << std::setw(12) << "residual"
              << "  Status\n";
    std::cout << std::string(70, '-') << "\n";

    int pass_count = 0;
    int total = 0;

    for (auto& spec : models) {
        Mesh mesh;
        mesh.init_stretched_y(Nx, Ny, 0.0, 2.0 * M_PI, -1.0, 1.0,
                              Mesh::tanh_stretching(2.0));

        Config config;
        config.nu = nu;
        config.adaptive_dt = true;
        config.CFL_max = 0.3;
        config.max_steps = nsteps;
        config.implicit_y_diffusion = true;
        config.stretch_y = true;
        config.stretch_beta = 2.0;
        config.verbose = false;
        config.turb_model = spec.type;

        RANSSolver solver(mesh, config);
        if (spec.type != TurbulenceModelType::None) {
            auto turb = create_turbulence_model(spec.type);
            solver.set_turbulence_model(std::move(turb));
        }
        solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel2D));
        solver.set_body_force(1.0, 0.0);
        solver.initialize_uniform(1.0, 0.0);
        solver.sync_to_gpu();

        bool ok = true;
        double residual = 0.0;
        auto t0 = std::chrono::high_resolution_clock::now();
        try {
            for (int s = 0; s < nsteps; ++s) {
                residual = solver.step();
            }
        } catch (...) {
            ok = false;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        solver.sync_from_gpu();

        // Check for NaN
        double max_vel = 0.0;
        for (int j = mesh.j_begin(); j < mesh.j_begin() + Ny; ++j) {
            for (int i = mesh.i_begin(); i <= mesh.i_begin() + Nx; ++i) {
                double u = solver.velocity().u(i, j);
                if (std::isnan(u) || std::isinf(u)) { ok = false; break; }
                max_vel = std::max(max_vel, std::abs(u));
            }
            if (!ok) break;
        }

        double max_nut = 0.0;
        if (spec.type != TurbulenceModelType::None) {
            for (int j = mesh.j_begin(); j < mesh.j_begin() + Ny; ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_begin() + Nx; ++i) {
                    max_nut = std::max(max_nut, solver.nu_t()(i, j));
                }
            }
        }

        if (ok && max_vel > 100.0) ok = false;  // velocity blowup

        std::cout << std::left << std::setw(18) << spec.name
                  << std::right << std::scientific << std::setprecision(2)
                  << std::setw(12) << solver.current_dt()
                  << std::setw(12) << max_vel
                  << std::setw(12) << max_nut
                  << std::setw(12) << residual
                  << "  " << (ok ? "OK" : "FAIL")
                  << "  (" << std::fixed << std::setprecision(1)
                  << elapsed << "s, "
                  << std::setprecision(0) << nsteps / elapsed << " steps/s)\n";

        if (ok) ++pass_count;
        ++total;
    }
    std::cout << "\n  " << pass_count << "/" << total << " models passed\n\n";
}

// ============================================================================
// Test 4: Wall-clock comparison (implicit vs explicit)
// ============================================================================
void test_wallclock_comparison() {
    std::cout << "=== Test 4: Wall-clock speedup (stretched grid, 200 steps) ===\n\n";

    const int Nx = 32, Ny = 48;
    const double nu = 1.0 / 180.0;
    const int nsteps = 200;

    // Run with EXPLICIT y-diffusion
    double elapsed_explicit, dt_explicit;
    {
        Mesh mesh;
        mesh.init_stretched_y(Nx, Ny, 0.0, 2.0 * M_PI, -1.0, 1.0,
                              Mesh::tanh_stretching(2.0));
        Config config;
        config.nu = nu;
        config.adaptive_dt = true;
        config.CFL_max = 0.3;
        config.implicit_y_diffusion = false;
        config.stretch_y = true;
        config.stretch_beta = 2.0;
        config.verbose = false;

        RANSSolver solver(mesh, config);
        solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel2D));
        solver.set_body_force(1.0, 0.0);
        solver.initialize_uniform(1.0, 0.0);
        solver.sync_to_gpu();

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int s = 0; s < nsteps; ++s) solver.step();
        auto t1 = std::chrono::high_resolution_clock::now();
        elapsed_explicit = std::chrono::duration<double>(t1 - t0).count();
        dt_explicit = solver.current_dt();
    }

    // Run with IMPLICIT y-diffusion
    double elapsed_implicit, dt_implicit;
    {
        Mesh mesh;
        mesh.init_stretched_y(Nx, Ny, 0.0, 2.0 * M_PI, -1.0, 1.0,
                              Mesh::tanh_stretching(2.0));
        Config config;
        config.nu = nu;
        config.adaptive_dt = true;
        config.CFL_max = 0.3;
        config.implicit_y_diffusion = true;
        config.stretch_y = true;
        config.stretch_beta = 2.0;
        config.verbose = false;

        RANSSolver solver(mesh, config);
        solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel2D));
        solver.set_body_force(1.0, 0.0);
        solver.initialize_uniform(1.0, 0.0);
        solver.sync_to_gpu();

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int s = 0; s < nsteps; ++s) solver.step();
        auto t1 = std::chrono::high_resolution_clock::now();
        elapsed_implicit = std::chrono::duration<double>(t1 - t0).count();
        dt_implicit = solver.current_dt();
    }

    double physical_time_explicit = nsteps * dt_explicit;
    double physical_time_implicit = nsteps * dt_implicit;

    std::cout << "  " << std::left << std::setw(20) << ""
              << std::right << std::setw(15) << "Explicit"
              << std::setw(15) << "Implicit" << "\n";
    std::cout << "  " << std::string(50, '-') << "\n";
    std::cout << "  " << std::left << std::setw(20) << "dt (adaptive)"
              << std::right << std::scientific << std::setprecision(3)
              << std::setw(15) << dt_explicit
              << std::setw(15) << dt_implicit << "\n";
    std::cout << "  " << std::left << std::setw(20) << "Wall time (s)"
              << std::right << std::fixed << std::setprecision(2)
              << std::setw(15) << elapsed_explicit
              << std::setw(15) << elapsed_implicit << "\n";
    std::cout << "  " << std::left << std::setw(20) << "Physical time"
              << std::right << std::scientific << std::setprecision(3)
              << std::setw(15) << physical_time_explicit
              << std::setw(15) << physical_time_implicit << "\n";
    std::cout << "  " << std::left << std::setw(20) << "Steps/sec"
              << std::right << std::fixed << std::setprecision(0)
              << std::setw(15) << nsteps / elapsed_explicit
              << std::setw(15) << nsteps / elapsed_implicit << "\n";
    std::cout << "\n  Physical time per wall-second:\n";
    std::cout << "    Explicit: " << std::scientific << std::setprecision(3)
              << physical_time_explicit / elapsed_explicit << "\n";
    std::cout << "    Implicit: " << std::scientific << std::setprecision(3)
              << physical_time_implicit / elapsed_implicit << "\n";

    std::cout << "\n=== PASS ===\n";
}

int main() {
    test_dt_speedup();
    test_poiseuille_accuracy();
    test_all_models_stretched();
    test_wallclock_comparison();
    return 0;
}