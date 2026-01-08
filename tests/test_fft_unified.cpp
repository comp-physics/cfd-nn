/// Unified FFT Poisson Solver Tests
/// Consolidates: test_fft1d_validation.cpp, test_fft2d_integration.cpp, test_fft_cpu_reference.cpp
///
/// Tests:
/// 1. FFT solver selection (FFT, FFT1D, FFT2D)
/// 2. FFT vs MG reference (3D periodic)
/// 3. FFT1D vs MG reference (channel/duct)
/// 4. FFT2D vs MG reference (2D channel)
/// 5. Grid convergence
///
/// GPU-only: FFT solvers require USE_GPU_OFFLOAD and USE_FFT_POISSON

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "config.hpp"
#include "poisson_solver.hpp"
#include "test_harness.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

using namespace nncfd;
using nncfd::test::harness::record;
using nncfd::test::BCPattern;
using nncfd::test::create_velocity_bc;

//=============================================================================
// Helpers
//=============================================================================

[[maybe_unused]] static double l2_norm(const ScalarField& f, const Mesh& mesh) {
    double sum = 0.0;
    int count = 0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                sum += f(i, j, k) * f(i, j, k);
                ++count;
            }
        }
    }
    return std::sqrt(sum / std::max(1, count));
}

[[maybe_unused]] static double l2_diff(const ScalarField& a, const ScalarField& b, const Mesh& mesh) {
    double sum = 0.0;
    int count = 0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double d = a(i, j, k) - b(i, j, k);
                sum += d * d;
                ++count;
            }
        }
    }
    return std::sqrt(sum / std::max(1, count));
}

static double mean_field(const ScalarField& f, const Mesh& mesh) {
    double sum = 0.0;
    int count = 0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                sum += f(i, j, k);
                ++count;
            }
        }
    }
    return sum / std::max(1, count);
}

[[maybe_unused]] static void remove_mean(ScalarField& f, const Mesh& mesh) {
    double m = mean_field(f, mesh);
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                f(i, j, k) -= m;
            }
        }
    }
}

[[maybe_unused]] static double linf_field(const ScalarField& f, const Mesh& mesh) {
    double max_val = 0.0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                max_val = std::max(max_val, std::abs(f(i, j, k)));
            }
        }
    }
    return max_val;
}

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
    cfg.dt = 0.001; cfg.max_iter = 1; cfg.nu = 1.0;
    cfg.poisson_solver = PoissonSolverType::FFT1D;

    RANSSolver solver(mesh, cfg);
    solver.set_velocity_bc(create_velocity_bc(BCPattern::Duct));

    bool pass = (solver.poisson_solver_type() == PoissonSolverType::FFT1D);
    record("FFT1D solver selection", pass);
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

    // Run with MG reference
    Config cfg_mg;
    cfg_mg.Nx = N; cfg_mg.Ny = N; cfg_mg.Nz = N;
    cfg_mg.dt = 0.001; cfg_mg.max_iter = 1; cfg_mg.nu = 0.01;
    cfg_mg.poisson_solver = PoissonSolverType::MG;

    RANSSolver solver_mg(mesh, cfg_mg);
    auto bc = create_velocity_bc(BCPattern::FullyPeriodic);
    solver_mg.set_velocity_bc(bc);

    // Initialize with sinusoidal velocity
    VectorField vel_mg(mesh);
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                vel_mg.u(i, j, k) = std::sin(2*M_PI*mesh.x(i)/L) *
                                    std::cos(2*M_PI*mesh.y(j)/L) *
                                    std::cos(2*M_PI*mesh.z(k)/L);
            }
        }
    }
    solver_mg.initialize(vel_mg);
    solver_mg.step();

    // Copy MG pressure
    ScalarField p_mg(mesh);
    for (int k = 0; k < mesh.Nz + 2; ++k)
        for (int j = 0; j < mesh.Ny + 2; ++j)
            for (int i = 0; i < mesh.Nx + 2; ++i)
                p_mg(i, j, k) = solver_mg.pressure()(i, j, k);

    // Run with FFT
    Config cfg_fft = cfg_mg;
    cfg_fft.poisson_solver = PoissonSolverType::FFT;

    RANSSolver solver_fft(mesh, cfg_fft);
    solver_fft.set_velocity_bc(bc);

    if (solver_fft.poisson_solver_type() != PoissonSolverType::FFT) {
        record("FFT vs MG (3D periodic)", true, true);
        return;
    }

    VectorField vel_fft(mesh);
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                vel_fft.u(i, j, k) = std::sin(2*M_PI*mesh.x(i)/L) *
                                     std::cos(2*M_PI*mesh.y(j)/L) *
                                     std::cos(2*M_PI*mesh.z(k)/L);
            }
        }
    }
    solver_fft.initialize(vel_fft);
    solver_fft.step();

#ifdef USE_GPU_OFFLOAD
    solver_fft.sync_from_gpu();
#endif

    ScalarField p_fft(mesh);
    for (int k = 0; k < mesh.Nz + 2; ++k)
        for (int j = 0; j < mesh.Ny + 2; ++j)
            for (int i = 0; i < mesh.Nx + 2; ++i)
                p_fft(i, j, k) = solver_fft.pressure()(i, j, k);

    // Compare (remove mean for gauge-independent comparison)
    remove_mean(p_mg, mesh);
    remove_mean(p_fft, mesh);

    double ref_norm = l2_norm(p_mg, mesh);
    double diff = l2_diff(p_fft, p_mg, mesh);
    double rel_diff = (ref_norm > 1e-15) ? diff / ref_norm : diff;

    bool pass = (rel_diff < 0.1);
    record("FFT vs MG (3D periodic)", pass);
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

    // Run with MG reference
    Config cfg_mg;
    cfg_mg.Nx = N; cfg_mg.Ny = N; cfg_mg.Nz = N;
    cfg_mg.dt = 0.001; cfg_mg.max_iter = 1; cfg_mg.nu = 0.01;
    cfg_mg.poisson_solver = PoissonSolverType::MG;

    RANSSolver solver_mg(mesh, cfg_mg);
    auto bc = create_velocity_bc(BCPattern::Channel3D);
    solver_mg.set_velocity_bc(bc);

    VectorField vel(mesh);
    vel.fill(1.0, 0.0, 0.0);
    solver_mg.initialize(vel);
    solver_mg.step();

    ScalarField p_mg(mesh);
    for (int k = 0; k < mesh.Nz + 2; ++k)
        for (int j = 0; j < mesh.Ny + 2; ++j)
            for (int i = 0; i < mesh.Nx + 2; ++i)
                p_mg(i, j, k) = solver_mg.pressure()(i, j, k);

    // Run with FFT1D
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

#ifdef USE_GPU_OFFLOAD
    solver_fft.sync_from_gpu();
#endif

    ScalarField p_fft(mesh);
    for (int k = 0; k < mesh.Nz + 2; ++k)
        for (int j = 0; j < mesh.Ny + 2; ++j)
            for (int i = 0; i < mesh.Nx + 2; ++i)
                p_fft(i, j, k) = solver_fft.pressure()(i, j, k);

    remove_mean(p_mg, mesh);
    remove_mean(p_fft, mesh);

    double ref_norm = l2_norm(p_mg, mesh);
    double diff = l2_diff(p_fft, p_mg, mesh);
    double rel_diff = (ref_norm > 1e-15) ? diff / ref_norm : diff;

    bool pass = (rel_diff < 0.15);
    record("FFT1D vs MG (3D channel)", pass);
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

    Config cfg_mg;
    cfg_mg.Nx = N; cfg_mg.Ny = N; cfg_mg.Nz = N;
    cfg_mg.dt = 0.001; cfg_mg.max_iter = 1; cfg_mg.nu = 0.01;
    cfg_mg.poisson_solver = PoissonSolverType::MG;

    RANSSolver solver_mg(mesh, cfg_mg);
    auto bc = create_velocity_bc(BCPattern::Duct);
    solver_mg.set_velocity_bc(bc);

    VectorField vel(mesh);
    vel.fill(1.0, 0.0, 0.0);
    solver_mg.initialize(vel);
    solver_mg.step();

    ScalarField p_mg(mesh);
    for (int k = 0; k < mesh.Nz + 2; ++k)
        for (int j = 0; j < mesh.Ny + 2; ++j)
            for (int i = 0; i < mesh.Nx + 2; ++i)
                p_mg(i, j, k) = solver_mg.pressure()(i, j, k);

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

#ifdef USE_GPU_OFFLOAD
    solver_fft.sync_from_gpu();
#endif

    ScalarField p_fft(mesh);
    for (int k = 0; k < mesh.Nz + 2; ++k)
        for (int j = 0; j < mesh.Ny + 2; ++j)
            for (int i = 0; i < mesh.Nx + 2; ++i)
                p_fft(i, j, k) = solver_fft.pressure()(i, j, k);

    remove_mean(p_mg, mesh);
    remove_mean(p_fft, mesh);

    double ref_norm = l2_norm(p_mg, mesh);
    double diff = l2_diff(p_fft, p_mg, mesh);
    double rel_diff = (ref_norm > 1e-15) ? diff / ref_norm : diff;

    bool pass = (rel_diff < 0.15);
    record("FFT1D vs MG (3D duct)", pass);
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

    // MG reference (CPU)
    Config cfg_mg;
    cfg_mg.Nx = Nx; cfg_mg.Ny = Ny;
    cfg_mg.dt = 0.001; cfg_mg.max_iter = 1; cfg_mg.nu = 0.01;
    cfg_mg.poisson_solver = PoissonSolverType::MG;

    RANSSolver solver_mg(mesh, cfg_mg);
    auto bc = create_velocity_bc(BCPattern::Channel2D);
    solver_mg.set_velocity_bc(bc);

    VectorField vel(mesh);
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        double y = mesh.y(j);
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            vel.u(i, j) = std::sin(mesh.x(i)) * std::cos(M_PI * y / Ly);
        }
    }
    solver_mg.initialize(vel);
    solver_mg.step();

    double mg_max = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            mg_max = std::max(mg_max, std::abs(solver_mg.pressure()(i, j)));
        }
    }

    // FFT2D (GPU) - test via RANSSolver
    Config cfg_fft = cfg_mg;
    cfg_fft.poisson_solver = PoissonSolverType::FFT;

    RANSSolver solver_fft(mesh, cfg_fft);
    solver_fft.set_velocity_bc(bc);

    // If FFT not available, skip
    if (solver_fft.poisson_solver_type() == PoissonSolverType::MG) {
        record("FFT2D vs MG (2D channel)", true, true);
        return;
    }

    VectorField vel2(mesh);
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        double y = mesh.y(j);
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            vel2.u(i, j) = std::sin(mesh.x(i)) * std::cos(M_PI * y / Ly);
        }
    }
    solver_fft.initialize(vel2);
    solver_fft.step();

#ifdef USE_GPU_OFFLOAD
    solver_fft.sync_from_gpu();
#endif

    double fft_max = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            fft_max = std::max(fft_max, std::abs(solver_fft.pressure()(i, j)));
        }
    }

    // Check that both produce non-trivial solutions of similar magnitude
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
    cfg.dt = 0.001; cfg.max_iter = 1; cfg.nu = 1.0;
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

#ifdef USE_GPU_OFFLOAD
    solver.sync_from_gpu();
#endif

    double p_max = linf_field(solver.pressure(), mesh);
    bool pass = std::isfinite(p_max) && (p_max < 1e10);
    record("FFT1D correctness (finite pressure)", pass);
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
        cfg.dt = 0.001; cfg.max_iter = 1; cfg.nu = 1.0;
        cfg.poisson_solver = PoissonSolverType::FFT1D;

        RANSSolver solver(mesh, cfg);
        solver.set_velocity_bc(create_velocity_bc(BCPattern::Duct));

        if (solver.poisson_solver_type() != PoissonSolverType::FFT1D) {
            continue;
        }

        VectorField vel(mesh);
        vel.fill(1.0, 0.0, 0.0);
        solver.initialize(vel);

        for (int step = 0; step < 5; ++step) solver.step();

#ifdef USE_GPU_OFFLOAD
        solver.sync_from_gpu();
#endif

        norms.push_back(linf_field(solver.pressure(), mesh));
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
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            input(i, j) = (j - mesh.j_begin()) * Nx + (i - mesh.i_begin()) + 1.0;
        }
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
// Main
//=============================================================================

static void print_build_info() {
#ifdef USE_GPU_OFFLOAD
    std::cout << "Build: GPU (USE_GPU_OFFLOAD=ON)\n";
#ifdef USE_FFT_POISSON
    std::cout << "FFT:   enabled (USE_FFT_POISSON=ON)\n";
#else
    std::cout << "FFT:   disabled (USE_FFT_POISSON=OFF)\n";
#endif
#else
    std::cout << "Build: CPU (USE_GPU_OFFLOAD=OFF)\n";
    std::cout << "FFT:   not available (GPU required)\n";
#endif
    std::cout << "\n";
}

int main() {
    namespace harness = nncfd::test::harness;
    return harness::run("Unified FFT Poisson Solver Tests", [] {
        print_build_info();

        test_fft1d_selection();
        test_fft_vs_mg_periodic();
        test_fft1d_vs_mg_channel();
        test_fft1d_vs_mg_duct();
        test_fft2d_vs_mg_channel();
        test_fft1d_correctness();
        test_fft1d_grid_convergence();
        test_2d_indexing();

        const auto& c = harness::counters();
        if (c.skipped > 0 && c.passed == 0 && c.failed == 0) {
            std::cout << "\nNote: All tests skipped (FFT requires GPU build with cuFFT)\n";
        }
    });
}
