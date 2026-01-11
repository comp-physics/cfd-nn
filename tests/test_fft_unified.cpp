/// Unified FFT Poisson Solver Tests
/// Tests FFT solver selection, FFT vs MG reference, grid convergence

#include "test_harness.hpp"
#include "test_utilities.hpp"
#include "poisson_solver.hpp"
#include "timing.hpp"
#include <cmath>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace nncfd;
using nncfd::test::harness::record;
using nncfd::test::BCPattern;
using nncfd::test::create_velocity_bc;
using nncfd::test::l2_norm;
using nncfd::test::l2_diff;
using nncfd::test::linf_norm;
using nncfd::test::mean_value;
using nncfd::test::remove_mean;

static bool fft_available() {
#if defined(USE_GPU_OFFLOAD) && defined(USE_FFT_POISSON)
    return true;
#else
    return false;
#endif
}

//=============================================================================
// Test 1: FFT1D Solver Selection
//=============================================================================

void test_fft1d_selection() {
    if (!fft_available()) {
        record("FFT1D solver selection", true, true);
        return;
    }

#if defined(USE_GPU_OFFLOAD) && defined(USE_FFT_POISSON)
    Mesh mesh;
    mesh.init_uniform(32, 32, 32, 0.0, 2*M_PI, 0.0, 2.0, 0.0, 2.0);

    Config cfg;
    cfg.Nx = 32; cfg.Ny = 32; cfg.Nz = 32;
    cfg.dt = 0.001; cfg.max_steps = 1; cfg.nu = 1.0;
    cfg.poisson_solver = PoissonSolverType::FFT1D;

    RANSSolver solver(mesh, cfg);
    solver.set_velocity_bc(create_velocity_bc(BCPattern::Duct));

    record("FFT1D solver selection", solver.poisson_solver_type() == PoissonSolverType::FFT1D);
#endif
}

//=============================================================================
// Test 2: FFT vs MG Reference (3D Periodic)
//=============================================================================

void test_fft_vs_mg_periodic() {
    if (!fft_available()) {
        record("FFT vs MG (3D periodic)", true, true);
        return;
    }

#if defined(USE_GPU_OFFLOAD) && defined(USE_FFT_POISSON)
    const int N = 32;
    const double L = 2.0 * M_PI;

    Mesh mesh;
    mesh.init_uniform(N, N, N, 0.0, L, 0.0, L, 0.0, L);
    auto bc = create_velocity_bc(BCPattern::FullyPeriodic);

    auto init_velocity = [&](VectorField& vel) {
        FOR_INTERIOR_3D(mesh, i, j, k) {
            vel.u(i, j, k) = std::sin(2*M_PI*mesh.x(i)/L) *
                            std::cos(2*M_PI*mesh.y(j)/L) *
                            std::cos(2*M_PI*mesh.z(k)/L);
        }
        // u at i_end
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k)
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j)
                vel.u(mesh.i_end(), j, k) = std::sin(2*M_PI*mesh.x(mesh.i_end())/L) *
                                            std::cos(2*M_PI*mesh.y(j)/L) *
                                            std::cos(2*M_PI*mesh.z(k)/L);
    };

    // MG reference
    Config cfg_mg;
    cfg_mg.Nx = N; cfg_mg.Ny = N; cfg_mg.Nz = N;
    cfg_mg.dt = 0.001; cfg_mg.max_steps = 1; cfg_mg.nu = 0.01;
    cfg_mg.poisson_solver = PoissonSolverType::MG;

    RANSSolver solver_mg(mesh, cfg_mg);
    solver_mg.set_velocity_bc(bc);
    VectorField vel_mg(mesh);
    init_velocity(vel_mg);
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

    if (solver_fft.poisson_solver_type() != PoissonSolverType::FFT) {
        record("FFT vs MG (3D periodic)", true, true);
        return;
    }

    VectorField vel_fft(mesh);
    init_velocity(vel_fft);
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

    record("FFT vs MG (3D periodic)", rel_diff < 0.1);
#endif
}

//=============================================================================
// Test 3: FFT1D vs MG Reference (3D Channel)
//=============================================================================

void test_fft1d_vs_mg_channel() {
    if (!fft_available()) {
        record("FFT1D vs MG (3D channel)", true, true);
        return;
    }

#if defined(USE_GPU_OFFLOAD) && defined(USE_FFT_POISSON)
    const int N = 32;
    Mesh mesh;
    mesh.init_uniform(N, N, N, 0.0, 2*M_PI, 0.0, 2.0, 0.0, 2*M_PI);
    auto bc = create_velocity_bc(BCPattern::Channel3D);

    Config cfg_mg;
    cfg_mg.Nx = N; cfg_mg.Ny = N; cfg_mg.Nz = N;
    cfg_mg.dt = 0.001; cfg_mg.max_steps = 1; cfg_mg.nu = 0.01;
    cfg_mg.poisson_solver = PoissonSolverType::MG;

    RANSSolver solver_mg(mesh, cfg_mg);
    solver_mg.set_velocity_bc(bc);
    VectorField vel(mesh);
    vel.fill(1.0, 0.0, 0.0);
    solver_mg.initialize(vel);
    solver_mg.step();

    ScalarField p_mg(mesh);
    FOR_INTERIOR_3D(mesh, i, j, k) { p_mg(i, j, k) = solver_mg.pressure()(i, j, k); }

    Config cfg_fft = cfg_mg;
    cfg_fft.poisson_solver = PoissonSolverType::FFT1D;

    RANSSolver solver_fft(mesh, cfg_fft);
    solver_fft.set_velocity_bc(bc);

    if (solver_fft.poisson_solver_type() != PoissonSolverType::FFT1D) {
        record("FFT1D vs MG (3D channel)", true, true);
        return;
    }

    VectorField vel2(mesh);
    vel2.fill(1.0, 0.0, 0.0);
    solver_fft.initialize(vel2);
    solver_fft.step();
    solver_fft.sync_from_gpu();

    ScalarField p_fft(mesh);
    FOR_INTERIOR_3D(mesh, i, j, k) { p_fft(i, j, k) = solver_fft.pressure()(i, j, k); }

    remove_mean(p_mg, mesh);
    remove_mean(p_fft, mesh);

    double ref_norm = l2_norm(p_mg, mesh);
    double diff = l2_diff(p_fft, p_mg, mesh);
    double rel_diff = (ref_norm > 1e-15) ? diff / ref_norm : diff;

    record("FFT1D vs MG (3D channel)", rel_diff < 0.15);
#endif
}

//=============================================================================
// Test 4: FFT1D vs MG Reference (3D Duct)
//=============================================================================

void test_fft1d_vs_mg_duct() {
    if (!fft_available()) {
        record("FFT1D vs MG (3D duct)", true, true);
        return;
    }

#if defined(USE_GPU_OFFLOAD) && defined(USE_FFT_POISSON)
    const int N = 32;
    Mesh mesh;
    mesh.init_uniform(N, N, N, 0.0, 2*M_PI, 0.0, 2.0, 0.0, 2.0);
    auto bc = create_velocity_bc(BCPattern::Duct);

    Config cfg_mg;
    cfg_mg.Nx = N; cfg_mg.Ny = N; cfg_mg.Nz = N;
    cfg_mg.dt = 0.001; cfg_mg.max_steps = 1; cfg_mg.nu = 0.01;
    cfg_mg.poisson_solver = PoissonSolverType::MG;

    RANSSolver solver_mg(mesh, cfg_mg);
    solver_mg.set_velocity_bc(bc);
    VectorField vel(mesh);
    vel.fill(1.0, 0.0, 0.0);
    solver_mg.initialize(vel);
    solver_mg.step();

    ScalarField p_mg(mesh);
    FOR_INTERIOR_3D(mesh, i, j, k) { p_mg(i, j, k) = solver_mg.pressure()(i, j, k); }

    Config cfg_fft = cfg_mg;
    cfg_fft.poisson_solver = PoissonSolverType::FFT1D;

    RANSSolver solver_fft(mesh, cfg_fft);
    solver_fft.set_velocity_bc(bc);

    if (solver_fft.poisson_solver_type() != PoissonSolverType::FFT1D) {
        record("FFT1D vs MG (3D duct)", true, true);
        return;
    }

    VectorField vel2(mesh);
    vel2.fill(1.0, 0.0, 0.0);
    solver_fft.initialize(vel2);
    solver_fft.step();
    solver_fft.sync_from_gpu();

    ScalarField p_fft(mesh);
    FOR_INTERIOR_3D(mesh, i, j, k) { p_fft(i, j, k) = solver_fft.pressure()(i, j, k); }

    remove_mean(p_mg, mesh);
    remove_mean(p_fft, mesh);

    double ref_norm = l2_norm(p_mg, mesh);
    double diff = l2_diff(p_fft, p_mg, mesh);
    double rel_diff = (ref_norm > 1e-15) ? diff / ref_norm : diff;

    record("FFT1D vs MG (3D duct)", rel_diff < 0.15);
#endif
}

//=============================================================================
// Test 5: FFT2D vs MG (2D Channel)
//=============================================================================

void test_fft2d_vs_mg_channel() {
#ifndef USE_GPU_OFFLOAD
    record("FFT2D vs MG (2D channel)", true, true);
    return;
#else
    const int Nx = 32, Ny = 32;
    const double Lx = 2.0 * M_PI, Ly = 2.0;

    Mesh mesh;
    mesh.init_uniform(Nx, Ny, 0.0, Lx, 0.0, Ly);
    auto bc = create_velocity_bc(BCPattern::Channel2D);

    auto init_velocity = [&](VectorField& vel) {
        FOR_INTERIOR_2D(mesh, i, j) {
            vel.u(i, j) = std::sin(mesh.x(i)) * std::cos(M_PI * mesh.y(j) / Ly);
        }
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j)
            vel.u(mesh.i_end(), j) = std::sin(mesh.x(mesh.i_end())) * std::cos(M_PI * mesh.y(j) / Ly);
    };

    // MG reference
    Config cfg_mg;
    cfg_mg.Nx = Nx; cfg_mg.Ny = Ny;
    cfg_mg.dt = 0.001; cfg_mg.max_steps = 1; cfg_mg.nu = 0.01;
    cfg_mg.poisson_solver = PoissonSolverType::MG;

    RANSSolver solver_mg(mesh, cfg_mg);
    solver_mg.set_velocity_bc(bc);
    VectorField vel(mesh);
    init_velocity(vel);
    solver_mg.initialize(vel);
    solver_mg.step();

    double mg_max = linf_norm(solver_mg.pressure(), mesh);

    // FFT
    Config cfg_fft = cfg_mg;
    cfg_fft.poisson_solver = PoissonSolverType::FFT;

    RANSSolver solver_fft(mesh, cfg_fft);
    solver_fft.set_velocity_bc(bc);

    if (solver_fft.poisson_solver_type() == PoissonSolverType::MG) {
        record("FFT2D vs MG (2D channel)", true, true);
        return;
    }

    VectorField vel2(mesh);
    init_velocity(vel2);
    solver_fft.initialize(vel2);
    solver_fft.step();
    solver_fft.sync_from_gpu();

    double fft_max = linf_norm(solver_fft.pressure(), mesh);

    bool pass = (mg_max > 1e-10 && fft_max > 1e-10);
    if (pass && mg_max > 1e-10) {
        double ratio = fft_max / mg_max;
        pass = (ratio > 0.1 && ratio < 10.0);
    }
    record("FFT2D vs MG (2D channel)", pass);
#endif
}

//=============================================================================
// Test 6: FFT1D Correctness (pressure stays finite)
//=============================================================================

void test_fft1d_correctness() {
    if (!fft_available()) {
        record("FFT1D correctness (finite pressure)", true, true);
        return;
    }

#if defined(USE_GPU_OFFLOAD) && defined(USE_FFT_POISSON)
    const int N = 64;
    Mesh mesh;
    mesh.init_uniform(N, N, N, 0.0, 2*M_PI, 0.0, 2.0, 0.0, 2.0);

    Config cfg;
    cfg.Nx = N; cfg.Ny = N; cfg.Nz = N;
    cfg.dt = 0.001; cfg.max_steps = 1; cfg.nu = 1.0;
    cfg.poisson_solver = PoissonSolverType::FFT1D;

    RANSSolver solver(mesh, cfg);
    solver.set_velocity_bc(create_velocity_bc(BCPattern::Duct));

    if (solver.poisson_solver_type() != PoissonSolverType::FFT1D) {
        record("FFT1D correctness (finite pressure)", true, true);
        return;
    }

    VectorField vel(mesh);
    vel.fill(1.0, 0.0, 0.0);
    solver.initialize(vel);
    solver.step();
    solver.sync_from_gpu();

    double p_max = linf_norm(solver.pressure(), mesh);
    record("FFT1D correctness (finite pressure)", std::isfinite(p_max) && p_max < 1e10);
#endif
}

//=============================================================================
// Test 7: FFT1D Grid Convergence
//=============================================================================

void test_fft1d_grid_convergence() {
    if (!fft_available()) {
        record("FFT1D grid convergence", true, true);
        return;
    }

#if defined(USE_GPU_OFFLOAD) && defined(USE_FFT_POISSON)
    std::vector<int> Ns = {16, 32};
    std::vector<double> norms;

    for (int N : Ns) {
        Mesh mesh;
        mesh.init_uniform(N, N, N, 0.0, 2*M_PI, 0.0, 2.0, 0.0, 2.0);

        Config cfg;
        cfg.Nx = N; cfg.Ny = N; cfg.Nz = N;
        cfg.dt = 0.001; cfg.max_steps = 1; cfg.nu = 0.01;
        cfg.poisson_solver = PoissonSolverType::FFT1D;
        cfg.dp_dx = -1.0;  // Add pressure gradient to drive flow

        RANSSolver solver(mesh, cfg);
        solver.set_velocity_bc(create_velocity_bc(BCPattern::Duct));

        if (solver.poisson_solver_type() != PoissonSolverType::FFT1D) continue;

        // Use non-uniform velocity to create non-zero divergence
        VectorField vel(mesh);
        FOR_INTERIOR_3D(mesh, i, j, k) {
            double y = mesh.y(j);
            double z = mesh.z(k);
            // Parabolic profile in y and z (duct flow approximation)
            vel.u(i, j, k) = (1.0 - (y - 1.0)*(y - 1.0)) * (1.0 - (z - 1.0)*(z - 1.0));
        }
        solver.initialize(vel);

        for (int step = 0; step < 5; ++step) solver.step();
        solver.sync_from_gpu();

        norms.push_back(linf_norm(solver.pressure(), mesh));
    }

    bool pass = (norms.size() >= 2);
    if (pass) {
        double ratio = norms[0] / (norms[1] + 1e-15);
        pass = (ratio > 0.1 && ratio < 10.0);
    }
    record("FFT1D grid convergence", pass);
#endif
}

//=============================================================================
// Test 8: 2D Pack/Unpack Identity (indexing check)
//=============================================================================

void test_2d_indexing() {
    const int Nx = 16, Ny = 16;
    Mesh mesh;
    mesh.init_uniform(Nx, Ny, 0.0, 2*M_PI, 0.0, 2.0);

    ScalarField input(mesh);
    FOR_INTERIOR_2D(mesh, i, j) {
        input(i, j) = (j - mesh.j_begin()) * Nx + (i - mesh.i_begin()) + 1.0;
    }

    double max_err = 0.0;
    const int Ng = mesh.Nghost;
    const int Nx_full = Nx + 2 * Ng;

    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            size_t idx = static_cast<size_t>(j + Ng) * Nx_full + (i + Ng);
            double val = input.data()[idx];
            double expected = j * Nx + i + 1.0;
            max_err = std::max(max_err, std::abs(val - expected));
        }
    }

    record("2D indexing pack/unpack identity", max_err < 1e-10);
}

//=============================================================================
// Benchmark: FFT1D Performance
//=============================================================================

void benchmark_fft1d_performance() {
#if defined(USE_GPU_OFFLOAD) && defined(USE_FFT_POISSON)
    std::cout << "\n=== FFT1D Performance Benchmark ===\n";
    std::cout << "Grid\tms/step\t\tPoisson(ms)\tFraction\n";

    std::vector<int> sizes = {64, 96, 128, 192};

    for (int N : sizes) {
        Mesh mesh;
        mesh.init_uniform(N, N, N, 0.0, 2*M_PI, -1.0, 1.0, -1.0, 1.0);

        Config cfg;
        cfg.Nx = N; cfg.Ny = N; cfg.Nz = N;
        cfg.dt = 0.001; cfg.max_steps = 100; cfg.nu = 0.0001;
        cfg.poisson_solver = PoissonSolverType::FFT1D;
        cfg.dp_dx = -0.0002;

        RANSSolver solver(mesh, cfg);
        solver.set_velocity_bc(create_velocity_bc(BCPattern::Duct));

        if (solver.poisson_solver_type() != PoissonSolverType::FFT1D) {
            std::cout << "N=" << N << ": FFT1D not available\n";
            continue;
        }

        VectorField vel(mesh);
        vel.fill(0.0, 0.0, 0.0);
        solver.initialize(vel);

        // Warm up
        for (int i = 0; i < 5; ++i) solver.step();

        // Reset timing stats
        TimingStats::instance().reset();

        // Time 50 steps
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 50; ++i) solver.step();
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double poisson_ms = TimingStats::instance().total("poisson_solve") * 1000.0;
        double fraction = poisson_ms / ms;

        std::cout << N << "^3\t" << std::fixed << std::setprecision(3) << ms/50
                  << "\t\t" << poisson_ms/50
                  << "\t\t" << std::setprecision(1) << fraction*100 << "%\n";
    }
    std::cout << "\n";
#endif
}

//=============================================================================
// Main
//=============================================================================

int main() {
    namespace harness = nncfd::test::harness;
    return harness::run("Unified FFT Poisson Solver Tests", [] {
        std::cout << "Build: "
#ifdef USE_GPU_OFFLOAD
                  << "GPU (USE_GPU_OFFLOAD=ON), FFT: "
#ifdef USE_FFT_POISSON
                  << "enabled\n";
#else
                  << "disabled\n";
#endif
#else
                  << "CPU (USE_GPU_OFFLOAD=OFF), FFT: not available\n";
#endif
        std::cout << "\n";

        test_fft1d_selection();
        test_fft_vs_mg_periodic();
        test_fft1d_vs_mg_channel();
        test_fft1d_vs_mg_duct();
        test_fft2d_vs_mg_channel();
        test_fft1d_correctness();
        test_fft1d_grid_convergence();
        test_2d_indexing();

        // Performance benchmark
        benchmark_fft1d_performance();

        const auto& c = harness::counters();
        if (c.skipped > 0 && c.passed == 0 && c.failed == 0) {
            std::cout << "\nNote: All tests skipped (FFT requires GPU build with cuFFT)\n";
        }
    });
}
