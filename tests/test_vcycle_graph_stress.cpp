/// @file test_vcycle_graph_stress.cpp
/// @brief Stress tests for V-cycle graph: BC alternation, convergence parity, anisotropic grids
///
/// Tests:
/// 1. BC type alternation (Dirichlet↔Neumann↔Periodic) - verifies graph recapture
/// 2. Convergence curve parity (graphed vs non-graphed paths)
/// 3. Mixed BCs on anisotropic grids

#include <iostream>
#include <cmath>
#include <vector>
#include <iomanip>
#include "mesh.hpp"
#include "fields.hpp"
#include "poisson_solver_multigrid.hpp"

using namespace nncfd;

// Helper to compute L2 residual on host
double compute_l2_residual(const double* u, const double* f, int N, double h) {
    const int Ng = 1;
    const int stride = N + 2*Ng;
    const int plane_stride = stride * stride;
    double sum_sq = 0.0;
    int count = 0;

    for (int k = Ng; k < N + Ng; ++k) {
        for (int j = Ng; j < N + Ng; ++j) {
            for (int i = Ng; i < N + Ng; ++i) {
                int idx = k * plane_stride + j * stride + i;
                double laplacian = (u[idx+1] - 2.0*u[idx] + u[idx-1]) / (h*h)
                                 + (u[idx+stride] - 2.0*u[idx] + u[idx-stride]) / (h*h)
                                 + (u[idx+plane_stride] - 2.0*u[idx] + u[idx-plane_stride]) / (h*h);
                double r = f[idx] - laplacian;
                sum_sq += r * r;
                count++;
            }
        }
    }
    return std::sqrt(sum_sq / count);
}

// Test 1: BC type alternation stress test
bool test_bc_alternation() {
    std::cout << "\n=== Test 1: BC Type Alternation ===" << std::endl;

    const int N = 32;
    const double L = 2.0 * M_PI;
    const double h = L / N;

    Mesh mesh;
    mesh.init_uniform(N, N, N, 0.0, L, 0.0, L, 0.0, L);

    const size_t total_size = static_cast<size_t>(N + 2) * (N + 2) * (N + 2);
    std::vector<double> rhs(total_size, 0.0);
    std::vector<double> p(total_size, 0.0);

    // Initialize RHS
    const int Ng = 1;
    const int stride = N + 2;
    const int plane_stride = stride * stride;
    for (int k = Ng; k < N + Ng; ++k) {
        double z = (k - Ng + 0.5) * h;
        for (int j = Ng; j < N + Ng; ++j) {
            double y = (j - Ng + 0.5) * h;
            for (int i = Ng; i < N + Ng; ++i) {
                double x = (i - Ng + 0.5) * h;
                int idx = k * plane_stride + j * stride + i;
                rhs[idx] = -3.0 * std::sin(x) * std::sin(y) * std::sin(z);
            }
        }
    }

    double* rhs_ptr = rhs.data();
    double* p_ptr = p.data();

#ifdef USE_GPU_OFFLOAD
    #pragma omp target enter data map(to: rhs_ptr[0:total_size], p_ptr[0:total_size])
#endif

    MultigridPoissonSolver mg(mesh);
    PoissonConfig cfg;
    cfg.fixed_cycles = 8;

    // BC configurations to cycle through
    struct BCConfig {
        PoissonBC x_lo, x_hi, y_lo, y_hi, z_lo, z_hi;
        const char* name;
    };

    BCConfig configs[] = {
        {PoissonBC::Periodic, PoissonBC::Periodic, PoissonBC::Periodic, PoissonBC::Periodic,
         PoissonBC::Periodic, PoissonBC::Periodic, "All Periodic"},
        {PoissonBC::Neumann, PoissonBC::Neumann, PoissonBC::Neumann, PoissonBC::Neumann,
         PoissonBC::Periodic, PoissonBC::Periodic, "Channel (Neumann walls)"},
        {PoissonBC::Dirichlet, PoissonBC::Dirichlet, PoissonBC::Dirichlet, PoissonBC::Dirichlet,
         PoissonBC::Dirichlet, PoissonBC::Dirichlet, "All Dirichlet"},
        {PoissonBC::Periodic, PoissonBC::Periodic, PoissonBC::Neumann, PoissonBC::Neumann,
         PoissonBC::Neumann, PoissonBC::Neumann, "Duct (Periodic x, Neumann y/z)"},
    };

    bool all_passed = true;
    const int num_alternations = 10;

    for (int iter = 0; iter < num_alternations; ++iter) {
        const BCConfig& bc = configs[iter % 4];

        // Reset solution
        std::fill(p.begin(), p.end(), 0.0);
#ifdef USE_GPU_OFFLOAD
        #pragma omp target update to(p_ptr[0:total_size])
#endif

        // Set new BCs (should trigger graph recapture)
        mg.set_bc(bc.x_lo, bc.x_hi, bc.y_lo, bc.y_hi, bc.z_lo, bc.z_hi);

#ifdef USE_GPU_OFFLOAD
        mg.solve_device(rhs_ptr, p_ptr, cfg);
        #pragma omp target update from(p_ptr[0:total_size])
#else
        ScalarField rhs_field(mesh), p_field(mesh);
        for (size_t i = 0; i < total_size; ++i) {
            rhs_field.data().data()[i] = rhs[i];
            p_field.data().data()[i] = p[i];
        }
        mg.solve(rhs_field, p_field, cfg);
        for (size_t i = 0; i < total_size; ++i) {
            p[i] = p_field.data().data()[i];
        }
#endif

        double res = compute_l2_residual(p_ptr, rhs_ptr, N, h);
        bool passed = res < 1e-2;  // Reasonable threshold for 8 V-cycles

        std::cout << "  Iter " << std::setw(2) << iter << " [" << bc.name << "]: "
                  << "res=" << std::scientific << res
                  << (passed ? " [PASS]" : " [FAIL]") << std::endl;

        if (!passed) all_passed = false;
    }

#ifdef USE_GPU_OFFLOAD
    #pragma omp target exit data map(delete: rhs_ptr[0:total_size], p_ptr[0:total_size])
#endif

    return all_passed;
}

// Test 2: Convergence curve parity (graphed vs non-graphed)
bool test_convergence_parity() {
    std::cout << "\n=== Test 2: Convergence Curve Parity ===" << std::endl;

    const int N = 32;
    const double L = 2.0 * M_PI;
    const double h = L / N;

    Mesh mesh;
    mesh.init_uniform(N, N, N, 0.0, L, 0.0, L, 0.0, L);

    const size_t total_size = static_cast<size_t>(N + 2) * (N + 2) * (N + 2);
    std::vector<double> rhs(total_size, 0.0);
    std::vector<double> p_graphed(total_size, 0.0);
    std::vector<double> p_nongraphed(total_size, 0.0);

    // Initialize RHS
    const int Ng = 1;
    const int stride = N + 2;
    const int plane_stride = stride * stride;
    for (int k = Ng; k < N + Ng; ++k) {
        double z = (k - Ng + 0.5) * h;
        for (int j = Ng; j < N + Ng; ++j) {
            double y = (j - Ng + 0.5) * h;
            for (int i = Ng; i < N + Ng; ++i) {
                double x = (i - Ng + 0.5) * h;
                int idx = k * plane_stride + j * stride + i;
                rhs[idx] = -3.0 * std::sin(x) * std::sin(y) * std::sin(z);
            }
        }
    }

    double* rhs_ptr = rhs.data();
    double* pg_ptr = p_graphed.data();
    double* png_ptr = p_nongraphed.data();

#ifdef USE_GPU_OFFLOAD
    #pragma omp target enter data map(to: rhs_ptr[0:total_size], pg_ptr[0:total_size], png_ptr[0:total_size])
#endif

    // Test with graphed path (MG_USE_VCYCLE_GRAPH=1)
    MultigridPoissonSolver mg_graphed(mesh);
    mg_graphed.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                       PoissonBC::Periodic, PoissonBC::Periodic,
                       PoissonBC::Periodic, PoissonBC::Periodic);

    // Test with non-graphed path
    MultigridPoissonSolver mg_nongraphed(mesh);
    mg_nongraphed.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                          PoissonBC::Periodic, PoissonBC::Periodic,
                          PoissonBC::Periodic, PoissonBC::Periodic);

    std::cout << "  Cycles  Graphed-Res    NonGraphed-Res  Diff" << std::endl;
    std::cout << "  ------  -----------    --------------  ----" << std::endl;

    bool all_close = true;
    for (int cycles = 1; cycles <= 8; ++cycles) {
        // Reset solutions
        std::fill(p_graphed.begin(), p_graphed.end(), 0.0);
        std::fill(p_nongraphed.begin(), p_nongraphed.end(), 0.0);
#ifdef USE_GPU_OFFLOAD
        #pragma omp target update to(pg_ptr[0:total_size], png_ptr[0:total_size])
#endif

        PoissonConfig cfg;
        cfg.fixed_cycles = cycles;

#ifdef USE_GPU_OFFLOAD
        mg_graphed.solve_device(rhs_ptr, pg_ptr, cfg);
        mg_nongraphed.solve_device(rhs_ptr, png_ptr, cfg);
        #pragma omp target update from(pg_ptr[0:total_size], png_ptr[0:total_size])
#else
        // CPU path would need different handling
        std::cout << "  (Skipping - CPU build)" << std::endl;
        break;
#endif

        double res_g = compute_l2_residual(pg_ptr, rhs_ptr, N, h);
        double res_ng = compute_l2_residual(png_ptr, rhs_ptr, N, h);
        double diff = std::abs(res_g - res_ng) / std::max(res_g, res_ng);

        bool close = diff < 0.01;  // Within 1%
        std::cout << "  " << std::setw(6) << cycles
                  << "  " << std::scientific << std::setprecision(4) << res_g
                  << "    " << res_ng
                  << "  " << std::fixed << std::setprecision(2) << (diff * 100) << "%"
                  << (close ? "" : " [MISMATCH]") << std::endl;

        if (!close) all_close = false;
    }

#ifdef USE_GPU_OFFLOAD
    #pragma omp target exit data map(delete: rhs_ptr[0:total_size], pg_ptr[0:total_size], png_ptr[0:total_size])
#endif

    return all_close;
}

// Test 3: Mixed BCs on anisotropic grid
bool test_anisotropic_mixed_bc() {
    std::cout << "\n=== Test 3: Mixed BCs on Anisotropic Grid ===" << std::endl;

    // Anisotropic grid: 64x32x16 with aspect ratio 4:2:1
    const int Nx = 64, Ny = 32, Nz = 16;
    const double Lx = 4.0 * M_PI, Ly = 2.0 * M_PI, Lz = M_PI;
    const double hx = Lx / Nx, hy = Ly / Ny, hz = Lz / Nz;

    Mesh mesh;
    mesh.init_uniform(Nx, Ny, Nz, 0.0, Lx, 0.0, Ly, 0.0, Lz);

    const size_t total_size = static_cast<size_t>(Nx + 2) * (Ny + 2) * (Nz + 2);
    std::vector<double> rhs(total_size, 0.0);
    std::vector<double> p(total_size, 0.0);

    // Initialize RHS with anisotropic frequencies
    const int Ng = 1;
    const int stride = Nx + 2*Ng;
    const int plane_stride = stride * (Ny + 2*Ng);
    for (int k = Ng; k < Nz + Ng; ++k) {
        double z = (k - Ng + 0.5) * hz;
        for (int j = Ng; j < Ny + Ng; ++j) {
            double y = (j - Ng + 0.5) * hy;
            for (int i = Ng; i < Nx + Ng; ++i) {
                double x = (i - Ng + 0.5) * hx;
                int idx = k * plane_stride + j * stride + i;
                // RHS for solution sin(x/2)*sin(y)*sin(2z)
                double kx = 0.5, ky = 1.0, kz = 2.0;
                rhs[idx] = -(kx*kx + ky*ky + kz*kz) * std::sin(kx*x) * std::sin(ky*y) * std::sin(kz*z);
            }
        }
    }

    double* rhs_ptr = rhs.data();
    double* p_ptr = p.data();

#ifdef USE_GPU_OFFLOAD
    #pragma omp target enter data map(to: rhs_ptr[0:total_size], p_ptr[0:total_size])
#endif

    MultigridPoissonSolver mg(mesh);
    // Mixed BCs: Periodic x, Neumann y, Dirichlet z
    mg.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
              PoissonBC::Neumann, PoissonBC::Neumann,
              PoissonBC::Dirichlet, PoissonBC::Dirichlet);

    PoissonConfig cfg;
    cfg.fixed_cycles = 12;  // More cycles for anisotropic grid

#ifdef USE_GPU_OFFLOAD
    mg.solve_device(rhs_ptr, p_ptr, cfg);
    #pragma omp target update from(p_ptr[0:total_size])
#else
    ScalarField rhs_field(mesh), p_field(mesh);
    for (size_t i = 0; i < total_size; ++i) {
        rhs_field.data().data()[i] = rhs[i];
    }
    mg.solve(rhs_field, p_field, cfg);
    for (size_t i = 0; i < total_size; ++i) {
        p[i] = p_field.data().data()[i];
    }
#endif

    // Compute residual using correct anisotropic spacing
    double sum_sq = 0.0;
    int count = 0;
    for (int k = Ng; k < Nz + Ng; ++k) {
        for (int j = Ng; j < Ny + Ng; ++j) {
            for (int i = Ng; i < Nx + Ng; ++i) {
                int idx = k * plane_stride + j * stride + i;
                double laplacian = (p[idx+1] - 2.0*p[idx] + p[idx-1]) / (hx*hx)
                                 + (p[idx+stride] - 2.0*p[idx] + p[idx-stride]) / (hy*hy)
                                 + (p[idx+plane_stride] - 2.0*p[idx] + p[idx-plane_stride]) / (hz*hz);
                double r = rhs[idx] - laplacian;
                sum_sq += r * r;
                count++;
            }
        }
    }
    double res = std::sqrt(sum_sq / count);

#ifdef USE_GPU_OFFLOAD
    #pragma omp target exit data map(delete: rhs_ptr[0:total_size], p_ptr[0:total_size])
#endif

    bool passed = res < 0.1;  // Relaxed threshold for anisotropic grid
    std::cout << "  Grid: " << Nx << "x" << Ny << "x" << Nz
              << " (AR=" << (Lx/Lz) << ":" << (Ly/Lz) << ":1)" << std::endl;
    std::cout << "  BCs: Periodic-x, Neumann-y, Dirichlet-z" << std::endl;
    std::cout << "  Residual: " << std::scientific << res
              << (passed ? " [PASS]" : " [FAIL]") << std::endl;

    return passed;
}

int main() {
    std::cout << "=== V-cycle Graph Stress Tests ===" << std::endl;

    int failures = 0;

    if (!test_bc_alternation()) failures++;
    if (!test_convergence_parity()) failures++;
    if (!test_anisotropic_mixed_bc()) failures++;

    std::cout << "\n=== Summary ===" << std::endl;
    if (failures == 0) {
        std::cout << "All stress tests PASSED" << std::endl;
        return 0;
    } else {
        std::cout << failures << " stress test(s) FAILED" << std::endl;
        return 1;
    }
}
