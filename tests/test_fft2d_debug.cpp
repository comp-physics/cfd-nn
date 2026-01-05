/**
 * @file test_fft2d_debug.cpp
 * @brief Debug test for FFT2D Poisson solver - compares GPU vs CPU reference
 *
 * This test isolates FFT2D bugs by comparing against a simple CPU reference:
 * 1. CPU: 1D FFT in x + Thomas algorithm for tridiagonal in y
 * 2. GPU: FFT2DPoissonSolver
 *
 * Run with small grid (16x16) to easily inspect intermediate values.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <algorithm>
#include <iomanip>
#include "mesh.hpp"
#include "fields.hpp"
#include "poisson_solver_fft2d.hpp"

using namespace nncfd;

// ============================================================================
// CPU Reference Implementation
// ============================================================================

// Simple 1D FFT using direct DFT (for small N, correctness over speed)
void cpu_fft_1d(const std::vector<double>& in, std::vector<std::complex<double>>& out, int N) {
    int N_modes = N / 2 + 1;
    out.resize(N_modes);

    for (int m = 0; m < N_modes; ++m) {
        std::complex<double> sum(0.0, 0.0);
        for (int i = 0; i < N; ++i) {
            double theta = -2.0 * M_PI * m * i / N;
            sum += in[i] * std::complex<double>(std::cos(theta), std::sin(theta));
        }
        out[m] = sum;
    }
}

// Inverse 1D FFT (C2R)
void cpu_ifft_1d(const std::vector<std::complex<double>>& in, std::vector<double>& out, int N) {
    int N_modes = N / 2 + 1;
    out.resize(N);

    for (int i = 0; i < N; ++i) {
        double sum = 0.0;
        for (int m = 0; m < N_modes; ++m) {
            double theta = 2.0 * M_PI * m * i / N;
            std::complex<double> exp_factor(std::cos(theta), std::sin(theta));
            std::complex<double> contrib = in[m] * exp_factor;

            // For R2C FFT, modes 1 to N/2-1 have conjugate pairs
            if (m == 0 || m == N / 2) {
                sum += contrib.real();
            } else {
                sum += 2.0 * contrib.real();  // Account for conjugate symmetry
            }
        }
        out[i] = sum / N;  // Normalization
    }
}

// Thomas algorithm for tridiagonal system: Ax = b
// A is tridiagonal with lower=a, diagonal=d, upper=c
void thomas_solve(const std::vector<double>& a,
                  const std::vector<double>& d,
                  const std::vector<double>& c,
                  const std::vector<std::complex<double>>& b,
                  std::vector<std::complex<double>>& x) {
    int n = b.size();
    x.resize(n);

    // Forward elimination
    std::vector<double> c_prime(n);
    std::vector<std::complex<double>> d_prime(n);

    c_prime[0] = c[0] / d[0];
    d_prime[0] = b[0] / d[0];

    for (int i = 1; i < n; ++i) {
        double denom = d[i] - a[i] * c_prime[i-1];
        if (i < n - 1) {
            c_prime[i] = c[i] / denom;
        }
        d_prime[i] = (b[i] - a[i] * d_prime[i-1]) / denom;
    }

    // Back substitution
    x[n-1] = d_prime[n-1];
    for (int i = n - 2; i >= 0; --i) {
        x[i] = d_prime[i] - c_prime[i] * x[i+1];
    }
}

// CPU reference solver: 1D FFT in x + Thomas for each mode
void cpu_poisson_2d_reference(
    const std::vector<double>& rhs,  // Nx * Ny row-major
    std::vector<double>& p,
    int Nx, int Ny,
    double dx, double dy,
    bool neumann_y_lo, bool neumann_y_hi)
{
    int N_modes = Nx / 2 + 1;

    // Step 1: Compute eigenvalues for x-direction
    std::vector<double> lambda_x(N_modes);
    for (int m = 0; m < N_modes; ++m) {
        double theta = 2.0 * M_PI * m / Nx;
        lambda_x[m] = (2.0 - 2.0 * std::cos(theta)) / (dx * dx);
    }

    // Step 2: Subtract mean from RHS (for Neumann-Neumann case)
    std::vector<double> rhs_centered = rhs;
    double sum = 0.0;
    for (double v : rhs) sum += v;
    double mean = sum / (Nx * Ny);
    for (double& v : rhs_centered) v -= mean;

    // Step 3: FFT each row (y=const)
    // rhs_hat[m][j] = FFT of rhs[:, j]
    std::vector<std::vector<std::complex<double>>> rhs_hat(N_modes, std::vector<std::complex<double>>(Ny));

    for (int j = 0; j < Ny; ++j) {
        std::vector<double> row(Nx);
        for (int i = 0; i < Nx; ++i) {
            row[i] = rhs_centered[j * Nx + i];
        }
        std::vector<std::complex<double>> row_hat;
        cpu_fft_1d(row, row_hat, Nx);
        for (int m = 0; m < N_modes; ++m) {
            rhs_hat[m][j] = row_hat[m];
        }
    }

    // Step 4: Solve tridiagonal for each mode
    // (d²/dy² - λ_x[m]) p_hat = rhs_hat
    // Discretized: (p_{j-1} - 2*p_j + p_{j+1})/dy² - λ_x*p_j = rhs_hat_j
    // Rearranged: a*p_{j-1} + d*p_j + c*p_{j+1} = rhs_hat_j
    // where a = c = 1/dy², d = -2/dy² - λ_x

    double ay = 1.0 / (dy * dy);
    std::vector<std::vector<std::complex<double>>> p_hat(N_modes, std::vector<std::complex<double>>(Ny));

    for (int m = 0; m < N_modes; ++m) {
        std::vector<double> a_vec(Ny), d_vec(Ny), c_vec(Ny);

        // Solving: (d²/dy² - λ_x) p = rhs
        // Discretized: (p_{j-1} - 2p_j + p_{j+1})/dy² - λ_x*p_j = rhs_j
        // As tridiagonal: a*p_{j-1} + d*p_j + c*p_{j+1} = rhs_j
        // where a = c = 1/dy², d = -2/dy² - λ_x

        for (int j = 0; j < Ny; ++j) {
            // Default interior stencil
            a_vec[j] = ay;  // lower diagonal (1/dy²)
            c_vec[j] = ay;  // upper diagonal (1/dy²)
            d_vec[j] = -2.0 * ay - lambda_x[m];  // main diagonal
        }

        // Apply Neumann BC: ghost = interior, so p_{-1} = p_0 and p_N = p_{N-1}
        // At j=0: a*p_{-1} + d*p_0 + c*p_1 = rhs_0
        //         a*p_0 + d*p_0 + c*p_1 = rhs_0  (Neumann: p_{-1} = p_0)
        //         (a+d)*p_0 + c*p_1 = rhs_0
        // So: a_new[0] = 0, d_new[0] = a + d = ay + (-2ay - λ) = -ay - λ
        if (neumann_y_lo) {
            a_vec[0] = 0.0;
            d_vec[0] = -ay - lambda_x[m];  // (a + d) combined
        }
        if (neumann_y_hi) {
            c_vec[Ny-1] = 0.0;
            d_vec[Ny-1] = -ay - lambda_x[m];  // (c + d) combined
        }

        // Handle zero mode singularity (m=0 has lambda_x=0)
        // For pure Neumann, the system is singular. Pin p_hat[0][0] = 0.
        if (m == 0) {
            a_vec[0] = 0.0;
            d_vec[0] = 1.0;
            c_vec[0] = 0.0;
            rhs_hat[0][0] = std::complex<double>(0.0, 0.0);
        }

        thomas_solve(a_vec, d_vec, c_vec, rhs_hat[m], p_hat[m]);
    }

    // Step 5: Inverse FFT each row
    p.resize(Nx * Ny, 0.0);
    for (int j = 0; j < Ny; ++j) {
        std::vector<std::complex<double>> col_hat(N_modes);
        for (int m = 0; m < N_modes; ++m) {
            col_hat[m] = p_hat[m][j];
        }
        std::vector<double> row;
        cpu_ifft_1d(col_hat, row, Nx);
        for (int i = 0; i < Nx; ++i) {
            p[j * Nx + i] = row[i];
        }
    }
}

// ============================================================================
// Test Functions
// ============================================================================

void print_array_2d(const std::string& name, const std::vector<double>& arr, int Nx, int Ny) {
    std::cout << name << " (" << Nx << "x" << Ny << "):\n";
    for (int j = 0; j < std::min(Ny, 8); ++j) {
        std::cout << "  j=" << j << ": ";
        for (int i = 0; i < std::min(Nx, 8); ++i) {
            std::cout << std::setw(10) << std::setprecision(4) << arr[j * Nx + i] << " ";
        }
        if (Nx > 8) std::cout << "...";
        std::cout << "\n";
    }
    if (Ny > 8) std::cout << "  ...\n";
}

bool test_cpu_reference_only() {
    std::cout << "\n=== Test 1: CPU Reference Sanity Check ===\n";

    const int Nx = 16, Ny = 16;
    const double Lx = 2.0 * M_PI, Ly = 2.0;
    const double dx = Lx / Nx, dy = Ly / Ny;

    // Create manufactured solution: p = sin(x) * cos(pi*y/Ly)
    // Laplacian: -sin(x)*cos(pi*y/Ly) - sin(x)*(pi/Ly)^2*cos(pi*y/Ly)
    //          = -sin(x)*cos(pi*y/Ly) * (1 + (pi/Ly)^2)
    std::vector<double> p_exact(Nx * Ny);
    std::vector<double> rhs(Nx * Ny);

    double coeff = 1.0 + (M_PI / Ly) * (M_PI / Ly);
    for (int j = 0; j < Ny; ++j) {
        double y = (j + 0.5) * dy - Ly / 2;  // Cell centers, y ∈ [-1, 1]
        for (int i = 0; i < Nx; ++i) {
            double x = (i + 0.5) * dx;
            p_exact[j * Nx + i] = std::sin(x) * std::cos(M_PI * y / Ly);
            rhs[j * Nx + i] = -coeff * p_exact[j * Nx + i];
        }
    }

    // Solve with CPU reference
    std::vector<double> p_cpu;
    cpu_poisson_2d_reference(rhs, p_cpu, Nx, Ny, dx, dy, true, true);

    // Compare
    double max_err = 0.0, l2_err = 0.0;
    for (int i = 0; i < Nx * Ny; ++i) {
        double err = std::abs(p_cpu[i] - p_exact[i]);
        max_err = std::max(max_err, err);
        l2_err += err * err;
    }
    l2_err = std::sqrt(l2_err / (Nx * Ny));

    std::cout << "  Grid: " << Nx << "x" << Ny << "\n";
    std::cout << "  L2 error:  " << std::scientific << l2_err << "\n";
    std::cout << "  Max error: " << std::scientific << max_err << "\n";

    bool pass = (max_err < 0.1);  // Expect O(h²) discretization error
    std::cout << "  Result: " << (pass ? "[PASS]" : "[FAIL]") << "\n";
    return pass;
}

#ifdef USE_GPU_OFFLOAD
bool test_fft2d_vs_cpu() {
    std::cout << "\n=== Test 2: FFT2D vs CPU Reference ===\n";

    const int Nx = 16, Ny = 16;
    const double Lx = 2.0 * M_PI, Ly = 2.0;

    // Create mesh
    Mesh mesh;
    mesh.init_uniform(Nx, Ny, 0.0, Lx, -Ly/2, Ly/2);

    // Create manufactured RHS
    ScalarField rhs_field(mesh), p_field(mesh);

    double coeff = 1.0 + (M_PI / Ly) * (M_PI / Ly);
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double x = (i - 0.5) * mesh.dx;
            double y = -Ly/2 + (j - 0.5) * mesh.dy;
            rhs_field(i, j, 1) = -coeff * std::sin(x) * std::cos(M_PI * y / Ly);
        }
    }
    p_field.fill(0.0);

    // Solve with FFT2D
    FFT2DPoissonSolver solver(mesh);
    solver.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                  PoissonBC::Neumann, PoissonBC::Neumann);

    PoissonConfig cfg;
    cfg.verbose = true;

    // Map data to device
    double* rhs_ptr = rhs_field.data().data();
    double* p_ptr = p_field.data().data();
    size_t size = rhs_field.data().size();

    #pragma omp target enter data map(to: rhs_ptr[0:size]) map(alloc: p_ptr[0:size])
    #pragma omp target update to(p_ptr[0:size])

    int iters = solver.solve_device(rhs_ptr, p_ptr, cfg);

    #pragma omp target update from(p_ptr[0:size])
    #pragma omp target exit data map(delete: rhs_ptr[0:size], p_ptr[0:size])

    std::cout << "  FFT2D iterations: " << iters << "\n";

    // Extract GPU solution to flat array
    std::vector<double> p_gpu(Nx * Ny);
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            p_gpu[j * Nx + i] = p_field(i + 1, j + 1, 1);
        }
    }

    // Solve with CPU reference
    std::vector<double> rhs_flat(Nx * Ny);
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            rhs_flat[j * Nx + i] = rhs_field(i + 1, j + 1, 1);
        }
    }

    std::vector<double> p_cpu;
    cpu_poisson_2d_reference(rhs_flat, p_cpu, Nx, Ny, mesh.dx, mesh.dy, true, true);

    // Check if GPU solution is all zeros (major bug indicator)
    double gpu_sum = 0.0, gpu_max = 0.0;
    for (int i = 0; i < Nx * Ny; ++i) {
        gpu_sum += std::abs(p_gpu[i]);
        gpu_max = std::max(gpu_max, std::abs(p_gpu[i]));
    }
    std::cout << "  GPU solution stats: sum=" << gpu_sum << ", max=" << gpu_max << "\n";
    if (gpu_max < 1e-10) {
        std::cout << "  [BUG] GPU solution is all zeros! FFT2D not producing output.\n";
    }

    // Compare GPU vs CPU
    double max_diff = 0.0, l2_diff = 0.0;
    for (int i = 0; i < Nx * Ny; ++i) {
        double diff = std::abs(p_gpu[i] - p_cpu[i]);
        max_diff = std::max(max_diff, diff);
        l2_diff += diff * diff;
    }
    l2_diff = std::sqrt(l2_diff / (Nx * Ny));

    std::cout << "  L2 diff (GPU vs CPU):  " << std::scientific << l2_diff << "\n";
    std::cout << "  Max diff (GPU vs CPU): " << std::scientific << max_diff << "\n";

    if (max_diff > 1e-6) {
        std::cout << "\n  Detailed comparison (first 8x8):\n";
        std::cout << "  GPU solution:\n";
        print_array_2d("    p_gpu", p_gpu, Nx, Ny);
        std::cout << "  CPU solution:\n";
        print_array_2d("    p_cpu", p_cpu, Nx, Ny);
    }

    bool pass = (max_diff < 1e-4);  // Should match closely
    std::cout << "  Result: " << (pass ? "[PASS]" : "[FAIL]") << "\n";
    return pass;
}
#endif

int main() {
    std::cout << "=== FFT2D Debug Tests ===\n";
    std::cout << "Goal: Isolate FFT2D bugs by comparison with CPU reference\n";

    int passed = 0, failed = 0;

    if (test_cpu_reference_only()) passed++; else failed++;

#ifdef USE_GPU_OFFLOAD
    if (test_fft2d_vs_cpu()) passed++; else failed++;
#else
    std::cout << "\n[SKIP] GPU tests (USE_GPU_OFFLOAD not defined)\n";
#endif

    std::cout << "\n=== Summary ===\n";
    std::cout << "Passed: " << passed << ", Failed: " << failed << "\n";

    return (failed == 0) ? 0 : 1;
}
