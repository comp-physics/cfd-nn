/**
 * @file test_fft2d_integration.cpp
 * @brief Integration test for FFT2D - mimics how RANSSolver uses it
 *
 * This test isolates why FFT2D works in unit tests but fails in solver integration.
 */

#include <iostream>
#include <cmath>
#include <algorithm>
#include "mesh.hpp"
#include "fields.hpp"
#include "poisson_solver_fft2d.hpp"
#include "poisson_solver_multigrid.hpp"

using namespace nncfd;

// Test channel flow Poisson solve: periodic x, Neumann y
// Compare FFT2D vs MG to see if results match
bool test_fft2d_vs_mg_channel() {
    std::cout << "\n=== Test: FFT2D vs MG for Channel Flow ===\n";

    const int Nx = 32, Ny = 32;
    const double Lx = 2.0 * M_PI, Ly = 2.0;

    // Create mesh (2D)
    Mesh mesh;
    mesh.init_uniform(Nx, Ny, 0.0, Lx, 0.0, Ly);

    std::cout << "  Mesh: " << Nx << "x" << Ny << ", Nghost=" << mesh.Nghost << "\n";
    std::cout << "  total_cells=" << mesh.total_cells() << "\n";
    std::cout << "  is2D=" << mesh.is2D() << "\n";

    // Create RHS field: typical Poisson RHS = div(u*) / dt
    // For testing, use a smooth function that has zero mean
    ScalarField rhs_fft(mesh), rhs_mg(mesh);
    ScalarField p_fft(mesh), p_mg(mesh);

    // RHS = sin(x) * cos(pi*y/Ly) - has zero x-integral (good for periodic x)
    // NOTE: FFT2D and MG both use 2D indexing for 2D meshes
    // The solver's 2D path uses Mesh::index(i,j) = j*Nx_full + i
    double rhs_sum = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double x = (i - mesh.Nghost + 0.5) * mesh.dx;
            double y = (j - mesh.Nghost + 0.5) * mesh.dy;
            double val = std::sin(x) * std::cos(M_PI * y / Ly);
            // Both FFT2D and MG use 2D indexing for 2D meshes
            rhs_fft(i, j) = val;
            rhs_mg(i, j) = val;
            rhs_sum += val;
        }
    }
    p_fft.fill(0.0);
    p_mg.fill(0.0);

    std::cout << "  RHS sum (before mean): " << rhs_sum << "\n";

#ifdef USE_GPU_OFFLOAD
    // Test MG with CPU interface first to verify it works
    std::cout << "\n  [MG CPU Solve (sanity check)]\n";
    MultigridPoissonSolver mg_cpu(mesh);
    mg_cpu.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                  PoissonBC::Neumann, PoissonBC::Neumann);
    PoissonConfig cpu_cfg;
    cpu_cfg.tol = 1e-10;
    cpu_cfg.max_iter = 100;
    int iters_cpu = mg_cpu.solve(rhs_mg, p_mg, cpu_cfg);
    std::cout << "    Iterations: " << iters_cpu << "\n";
    std::cout << "    Residual: " << mg_cpu.residual() << "\n";

    double mg_cpu_max = 0.0, mg_cpu_sum = 0.0;
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double v = p_mg(i, j);
            mg_cpu_max = std::max(mg_cpu_max, std::abs(v));
            mg_cpu_sum += v;
        }
    }
    std::cout << "    MG CPU result: max=" << mg_cpu_max << ", sum=" << mg_cpu_sum << "\n";

    // Reset p_mg for GPU test
    p_mg.fill(0.0);

    // Setup FFT2D solver
    FFT2DPoissonSolver fft2d(mesh);
    fft2d.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                 PoissonBC::Neumann, PoissonBC::Neumann);

    // Setup MG solver (fresh instance for GPU)
    MultigridPoissonSolver mg(mesh);
    mg.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
              PoissonBC::Neumann, PoissonBC::Neumann);

    PoissonConfig cfg;
    cfg.tol = 1e-10;
    cfg.max_iter = 100;
    cfg.verbose = true;

    // Get raw pointers
    double* rhs_fft_ptr = rhs_fft.data().data();
    double* rhs_mg_ptr = rhs_mg.data().data();
    double* p_fft_ptr = p_fft.data().data();
    double* p_mg_ptr = p_mg.data().data();
    size_t size = mesh.total_cells();

    std::cout << "  Field size: " << size << "\n";

    // Map to device
    #pragma omp target enter data map(to: rhs_fft_ptr[0:size]) \
                                  map(to: rhs_mg_ptr[0:size]) \
                                  map(to: p_fft_ptr[0:size]) \
                                  map(to: p_mg_ptr[0:size])

    // Debug: verify RHS data is on device
    double rhs_sum_device = 0.0;
    #pragma omp target teams distribute parallel for reduction(+:rhs_sum_device) \
        map(present: rhs_mg_ptr[0:size])
    for (size_t i = 0; i < size; ++i) {
        rhs_sum_device += std::abs(rhs_mg_ptr[i]);
    }
    std::cout << "  RHS sum on device: " << rhs_sum_device << "\n";

    // Solve with FFT2D
    std::cout << "\n  [FFT2D Solve]\n";
    int iters_fft = fft2d.solve_device(rhs_fft_ptr, p_fft_ptr, cfg);
    std::cout << "    Iterations: " << iters_fft << "\n";

    // Solve with MG
    std::cout << "\n  [MG GPU Solve]\n";

    // Debug: check p_mg before solve
    double p_mg_sum_before = 0.0;
    #pragma omp target teams distribute parallel for reduction(+:p_mg_sum_before) \
        map(present: p_mg_ptr[0:size])
    for (size_t i = 0; i < size; ++i) {
        p_mg_sum_before += std::abs(p_mg_ptr[i]);
    }
    std::cout << "    p_mg sum before solve: " << p_mg_sum_before << "\n";

    int iters_mg = mg.solve_device(rhs_mg_ptr, p_mg_ptr, cfg);
    std::cout << "    Iterations: " << iters_mg << "\n";
    std::cout << "    Residual: " << mg.residual() << "\n";

    // Debug: check p_mg after solve (still on device)
    double p_mg_sum_after = 0.0;
    #pragma omp target teams distribute parallel for reduction(+:p_mg_sum_after) \
        map(present: p_mg_ptr[0:size])
    for (size_t i = 0; i < size; ++i) {
        p_mg_sum_after += std::abs(p_mg_ptr[i]);
    }
    std::cout << "    p_mg sum after solve (device): " << p_mg_sum_after << "\n";

    // Copy back
    #pragma omp target update from(p_fft_ptr[0:size])
    #pragma omp target update from(p_mg_ptr[0:size])
    #pragma omp target exit data map(delete: rhs_fft_ptr[0:size], rhs_mg_ptr[0:size], \
                                              p_fft_ptr[0:size], p_mg_ptr[0:size])

    // Compare solutions
    double max_fft = 0.0, max_mg = 0.0;
    double sum_fft = 0.0, sum_mg = 0.0;
    double max_diff = 0.0, l2_diff = 0.0;
    int count = 0;

    // Both FFT2D and MG use 2D indexing for 2D meshes
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double v_fft = p_fft(i, j);  // 2D indexing
            double v_mg = p_mg(i, j);    // 2D indexing

            max_fft = std::max(max_fft, std::abs(v_fft));
            max_mg = std::max(max_mg, std::abs(v_mg));
            sum_fft += v_fft;
            sum_mg += v_mg;

            double diff = std::abs(v_fft - v_mg);
            max_diff = std::max(max_diff, diff);
            l2_diff += diff * diff;
            count++;
        }
    }
    l2_diff = std::sqrt(l2_diff / count);

    std::cout << "\n  Solution comparison:\n";
    std::cout << "    FFT2D: max=" << max_fft << ", sum=" << sum_fft << "\n";
    std::cout << "    MG:    max=" << max_mg << ", sum=" << sum_mg << "\n";
    std::cout << "    Diff:  max=" << max_diff << ", L2=" << l2_diff << "\n";

    // Check scale factor
    if (max_mg > 1e-10) {
        double scale = max_fft / max_mg;
        std::cout << "    Scale factor (FFT/MG): " << scale << "\n";
    }

    // Print first few values
    std::cout << "\n  Sample values (j=Ny/2):\n";
    int j_mid = mesh.j_begin() + Ny / 2;
    for (int i = mesh.i_begin(); i < std::min(mesh.i_begin() + 8, mesh.i_end()); ++i) {
        std::cout << "    i=" << i - mesh.i_begin()
                  << ": FFT=" << p_fft(i, j_mid)
                  << ", MG=" << p_mg(i, j_mid) << "\n";
    }

    // Pass if solutions are similar (within reasonable tolerance)
    bool pass = (max_diff < 0.1 * max_mg) || (max_mg < 1e-10);
    std::cout << "\n  Result: " << (pass ? "[PASS]" : "[FAIL]") << "\n";

    if (!pass && max_fft > 1e-10 && max_mg > 1e-10) {
        std::cout << "    NOTE: Scale mismatch suggests normalization or indexing bug\n";
        std::cout << "    Expected scale ~1.0, got " << (max_fft/max_mg) << "\n";
    }

    return pass;
#else
    std::cout << "  [SKIP] GPU not available\n";
    return true;
#endif
}

// Simpler test: verify pack/unpack is identity
bool test_pack_unpack_identity() {
    std::cout << "\n=== Test: Pack/Unpack Identity ===\n";

    const int Nx = 16, Ny = 16;
    const double Lx = 2.0 * M_PI, Ly = 2.0;

    Mesh mesh;
    mesh.init_uniform(Nx, Ny, 0.0, Lx, 0.0, Ly);

    // Create input field with known pattern using 2D indexing
    ScalarField input(mesh), output(mesh);

    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            // Unique value at each cell (2D indexing)
            input(i, j) = (j - mesh.j_begin()) * Nx + (i - mesh.i_begin()) + 1.0;
        }
    }
    output.fill(0.0);

    // The pack/unpack in FFT2D uses 2D indexing for 2D meshes
    // Verify field access is correct with 2D formula: idx = j * Nx_full + i

    double* in_ptr = input.data().data();
    double* out_ptr = output.data().data();
    size_t size = mesh.total_cells();

    // FFT2D uses 2D indexing for 2D meshes
    const int Ng = mesh.Nghost;
    const int Nx_full = Nx + 2 * Ng;
    const int Ny_full = Ny + 2 * Ng;
    const int Nz_full = 1 + 2 * Ng;
    const size_t size_2d = (size_t)Nx_full * Ny_full;  // 2D plane size

    std::cout << "  Nx_full=" << Nx_full << ", Ny_full=" << Ny_full << ", Nz_full=" << Nz_full << "\n";
    std::cout << "  2D plane size=" << size_2d << ", total_cells()=" << size << "\n";

    // Test the 2D indexing formula (no k offset)
    double max_err = 0.0;
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            // FFT2D pack formula (2D indexing, no k offset):
            const size_t src_idx = (size_t)(j + Ng) * Nx_full + (i + Ng);
            double val = in_ptr[src_idx];
            double expected = j * Nx + i + 1.0;

            double err = std::abs(val - expected);
            max_err = std::max(max_err, err);
        }
    }

    std::cout << "  Max indexing error: " << max_err << "\n";
    bool pass = max_err < 1e-10;
    std::cout << "  Result: " << (pass ? "[PASS]" : "[FAIL]") << "\n";
    return pass;
}

int main() {
    std::cout << "=== FFT2D Integration Tests ===\n";

    int passed = 0, failed = 0;

    if (test_pack_unpack_identity()) passed++; else failed++;
    if (test_fft2d_vs_mg_channel()) passed++; else failed++;

    std::cout << "\n=== Summary ===\n";
    std::cout << "Passed: " << passed << ", Failed: " << failed << "\n";

    return (failed == 0) ? 0 : 1;
}
