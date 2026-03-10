/// @file test_fft_stretched.cpp
/// @brief Validates FFT Poisson solvers on stretched y-grids.
///
/// Tests that FFT/FFT2D with variable-coefficient tridiagonal solves match
/// multigrid reference solutions on tanh-stretched grids (D·G=L consistency).

#include "test_harness.hpp"
#include "test_utilities.hpp"
#include "solver.hpp"
#include "config.hpp"
#include "mesh.hpp"
#include <cmath>
#include <iomanip>

using namespace nncfd;
using nncfd::test::harness::record;
using nncfd::test::BCPattern;
using nncfd::test::create_velocity_bc;
using nncfd::test::l2_norm;
using nncfd::test::l2_diff;
using nncfd::test::remove_mean;

static bool fft_available() {
#if defined(USE_GPU_OFFLOAD) && defined(USE_FFT_POISSON)
    return true;
#else
    return false;
#endif
}

// ============================================================================
// Test 1: FFT2D vs MG on 2D stretched-y channel
// ============================================================================

void test_fft2d_stretched_vs_mg() {
    if (!fft_available()) {
        record("FFT2D stretched vs MG (2D)", true, true);
        return;
    }

#if defined(USE_GPU_OFFLOAD) && defined(USE_FFT_POISSON)
    const int Nx = 32;
    const int Ny = 32;
    const double Lx = 2.0 * M_PI;
    const double Ly = 2.0;
    const double beta = 2.0;  // Tanh stretching parameter

    // Stretched mesh
    Mesh mesh;
    mesh.init_stretched_y(Nx, Ny, 0.0, Lx, 0.0, Ly,
                          Mesh::tanh_stretching(beta));

    auto bc = create_velocity_bc(BCPattern::Channel2D);

    // MG reference
    Config cfg_mg;
    cfg_mg.Nx = Nx; cfg_mg.Ny = Ny;
    cfg_mg.dt = 0.001; cfg_mg.max_steps = 1; cfg_mg.nu = 0.01;
    cfg_mg.poisson_solver = PoissonSolverType::MG;
    cfg_mg.stretch_y = true;

    RANSSolver solver_mg(mesh, cfg_mg);
    solver_mg.set_velocity_bc(bc);
    VectorField vel_mg(mesh);
    vel_mg.fill(1.0, 0.0);
    solver_mg.initialize(vel_mg);
    solver_mg.step();
    solver_mg.sync_from_gpu();

    ScalarField p_mg(mesh);
    FOR_INTERIOR_2D(mesh, i, j) { p_mg(i, j) = solver_mg.pressure()(i, j); }

    // FFT2D
    Config cfg_fft = cfg_mg;
    cfg_fft.poisson_solver = PoissonSolverType::FFT2D;

    RANSSolver solver_fft(mesh, cfg_fft);
    solver_fft.set_velocity_bc(bc);

    bool fft2d_selected = (solver_fft.poisson_solver_type() == PoissonSolverType::FFT2D);
    record("[FFT2D stretched] Solver selected for 2D stretched mesh", fft2d_selected);
    if (!fft2d_selected) {
        std::cerr << "  Got: " << static_cast<int>(solver_fft.poisson_solver_type()) << "\n";
        record("FFT2D stretched vs MG (2D)", true, true);
        return;
    }

    VectorField vel_fft(mesh);
    vel_fft.fill(1.0, 0.0);
    solver_fft.initialize(vel_fft);
    solver_fft.step();
    solver_fft.sync_from_gpu();

    ScalarField p_fft(mesh);
    FOR_INTERIOR_2D(mesh, i, j) { p_fft(i, j) = solver_fft.pressure()(i, j); }

    remove_mean(p_mg, mesh);
    remove_mean(p_fft, mesh);

    double ref_norm = l2_norm(p_mg, mesh);
    double diff = l2_diff(p_fft, p_mg, mesh);
    double rel_diff = (ref_norm > 1e-15) ? diff / ref_norm : diff;

    std::cout << "  FFT2D vs MG relative diff: " << std::scientific << rel_diff << "\n";
    record("FFT2D stretched vs MG (2D)", rel_diff < 0.1);
#endif
}

// ============================================================================
// Test 2: FFT vs MG on 3D stretched-y channel
// ============================================================================

void test_fft_stretched_vs_mg_3d() {
    if (!fft_available()) {
        record("FFT stretched vs MG (3D)", true, true);
        return;
    }

#if defined(USE_GPU_OFFLOAD) && defined(USE_FFT_POISSON)
    const int Nx = 32;
    const int Ny = 32;
    const int Nz = 32;
    const double Lx = 2.0 * M_PI;
    const double Ly = 2.0;
    const double Lz = 2.0 * M_PI;
    const double beta = 2.0;

    Mesh mesh;
    mesh.init_stretched_y(Nx, Ny, Nz, 0.0, Lx, 0.0, Ly, 0.0, Lz,
                          Mesh::tanh_stretching(beta));

    auto bc = create_velocity_bc(BCPattern::Channel3D);

    // MG reference
    Config cfg_mg;
    cfg_mg.Nx = Nx; cfg_mg.Ny = Ny; cfg_mg.Nz = Nz;
    cfg_mg.dt = 0.001; cfg_mg.max_steps = 1; cfg_mg.nu = 0.01;
    cfg_mg.poisson_solver = PoissonSolverType::MG;
    cfg_mg.stretch_y = true;

    RANSSolver solver_mg(mesh, cfg_mg);
    solver_mg.set_velocity_bc(bc);
    VectorField vel_mg(mesh);
    vel_mg.fill(1.0, 0.0, 0.0);
    solver_mg.initialize(vel_mg);
    solver_mg.step();
    solver_mg.sync_from_gpu();

    ScalarField p_mg(mesh);
    FOR_INTERIOR_3D(mesh, i, j, k) { p_mg(i, j, k) = solver_mg.pressure()(i, j, k); }

    // FFT
    Config cfg_fft = cfg_mg;
    cfg_fft.poisson_solver = PoissonSolverType::FFT;

    RANSSolver solver_fft(mesh, cfg_fft);
    solver_fft.set_velocity_bc(bc);

    bool fft_selected = (solver_fft.poisson_solver_type() == PoissonSolverType::FFT);
    record("[FFT stretched] Solver selected for 3D stretched mesh", fft_selected);
    if (!fft_selected) {
        std::cerr << "  Got: " << static_cast<int>(solver_fft.poisson_solver_type()) << "\n";
        record("FFT stretched vs MG (3D)", true, true);
        return;
    }

    VectorField vel_fft(mesh);
    vel_fft.fill(1.0, 0.0, 0.0);
    solver_fft.initialize(vel_fft);
    solver_fft.step();
    solver_fft.sync_from_gpu();

    ScalarField p_fft(mesh);
    FOR_INTERIOR_3D(mesh, i, j, k) { p_fft(i, j, k) = solver_fft.pressure()(i, j, k); }

    remove_mean(p_mg, mesh);
    remove_mean(p_fft, mesh);

    double ref_norm = l2_norm(p_mg, mesh);
    double diff = l2_diff(p_fft, p_mg, mesh);
    double rel_diff = (ref_norm > 1e-15) ? diff / ref_norm : diff;

    std::cout << "  FFT vs MG relative diff: " << std::scientific << rel_diff << "\n";
    record("FFT stretched vs MG (3D)", rel_diff < 0.1);
#endif
}

// ============================================================================
// Test 3: Auto-selection picks FFT for stretched-y (not MG)
// ============================================================================

void test_auto_selection_stretched() {
    if (!fft_available()) {
        record("Auto-select FFT for stretched-y", true, true);
        return;
    }

#if defined(USE_GPU_OFFLOAD) && defined(USE_FFT_POISSON)
    // 2D stretched
    {
        Mesh mesh;
        mesh.init_stretched_y(32, 32, 0.0, 2*M_PI, 0.0, 2.0,
                              Mesh::tanh_stretching(2.0));

        Config cfg;
        cfg.Nx = 32; cfg.Ny = 32;
        cfg.dt = 0.001; cfg.max_steps = 1; cfg.nu = 0.01;
        cfg.stretch_y = true;
        cfg.poisson_solver = PoissonSolverType::Auto;

        RANSSolver solver(mesh, cfg);
        solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel2D));

        bool got_fft2d = (solver.poisson_solver_type() == PoissonSolverType::FFT2D);
        record("Auto-select FFT2D for 2D stretched-y", got_fft2d);
    }

    // 3D stretched
    {
        Mesh mesh;
        mesh.init_stretched_y(32, 32, 32, 0.0, 2*M_PI, 0.0, 2.0, 0.0, 2*M_PI,
                              Mesh::tanh_stretching(2.0));

        Config cfg;
        cfg.Nx = 32; cfg.Ny = 32; cfg.Nz = 32;
        cfg.dt = 0.001; cfg.max_steps = 1; cfg.nu = 0.01;
        cfg.stretch_y = true;
        cfg.poisson_solver = PoissonSolverType::Auto;

        RANSSolver solver(mesh, cfg);
        solver.set_velocity_bc(create_velocity_bc(BCPattern::Channel3D));

        bool got_fft = (solver.poisson_solver_type() == PoissonSolverType::FFT);
        record("Auto-select FFT for 3D stretched-y", got_fft);
    }
#endif
}

// ============================================================================
// Test 4: FFT1D not selected for stretched-y (internal MG uses uniform dy)
// ============================================================================

void test_fft1d_guard_stretched() {
    if (!fft_available()) {
        record("FFT1D guard for stretched-y", true, true);
        return;
    }

#if defined(USE_GPU_OFFLOAD) && defined(USE_FFT_POISSON)
    Mesh mesh;
    mesh.init_stretched_y(32, 32, 32, 0.0, 2*M_PI, 0.0, 2.0, 0.0, 2.0,
                          Mesh::tanh_stretching(2.0));

    Config cfg;
    cfg.Nx = 32; cfg.Ny = 32; cfg.Nz = 32;
    cfg.dt = 0.001; cfg.max_steps = 1; cfg.nu = 0.01;
    cfg.stretch_y = true;
    cfg.poisson_solver = PoissonSolverType::Auto;

    RANSSolver solver(mesh, cfg);
    solver.set_velocity_bc(create_velocity_bc(BCPattern::Duct));

    // For duct (periodic x, walls y+z), FFT1D would normally be selected
    // but should be skipped for stretched y
    bool not_fft1d = (solver.poisson_solver_type() != PoissonSolverType::FFT1D);
    record("FFT1D not selected for stretched-y duct", not_fft1d);
#endif
}

// ============================================================================
// Main
// ============================================================================

int main() {
    namespace harness = nncfd::test::harness;
    return harness::run("FFT Stretched-Y Tests", [] {
        std::cout << "Build: "
#ifdef USE_GPU_OFFLOAD
                  << "GPU, FFT: "
#ifdef USE_FFT_POISSON
                  << "enabled\n";
#else
                  << "disabled\n";
#endif
#else
                  << "CPU (FFT not available)\n";
#endif
        std::cout << "\n";

        test_fft2d_stretched_vs_mg();
        test_fft_stretched_vs_mg_3d();
        test_auto_selection_stretched();
        test_fft1d_guard_stretched();
    });
}
