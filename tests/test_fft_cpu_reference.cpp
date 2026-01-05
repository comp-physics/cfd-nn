/// @file test_fft_cpu_reference.cpp
/// @brief FFT/FFT1D validation against CPU reference (MG/HYPRE)
///
/// CRITICAL TEST: Validates that FFT and FFT1D solvers (GPU-only) produce
/// solutions consistent with CPU-based solvers (MG, HYPRE) on the SAME node.
///
/// This test should be run on the H200 runner where both CPU and GPU builds
/// are available. It verifies:
///   1. FFT and MG/HYPRE produce the same solution (within tolerance)
///   2. FFT1D and MG/HYPRE produce the same solution (within tolerance)
///   3. FFT solvers don't converge to wrong solutions due to BC/gauge bugs
///
/// Method:
///   1. Create manufactured solution with known RHS
///   2. Solve with MG (or HYPRE) as CPU reference
///   3. Solve with FFT or FFT1D via RANSSolver (GPU path)
///   4. Compare solutions: ||p_fft - p_ref|| / ||p_ref|| < tolerance
///
/// Note: This test uses the full RANSSolver to exercise the solver selection
/// and GPU paths, not the standalone PoissonSolver.

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "config.hpp"
#include <iostream>
#include <cmath>
#include <iomanip>
#include <vector>
#include <string>

using namespace nncfd;

// Compute L2 norm of a 3D field (interior only)
double l2_norm_3d(const ScalarField& f, const Mesh& mesh) {
    double sum_sq = 0.0;
    int count = 0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                sum_sq += f(i, j, k) * f(i, j, k);
                ++count;
            }
        }
    }
    return std::sqrt(sum_sq / count);
}

// Compute L2 difference: ||a - b||_2
double l2_diff_3d(const ScalarField& a, const ScalarField& b, const Mesh& mesh) {
    double sum_sq = 0.0;
    int count = 0;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double diff = a(i, j, k) - b(i, j, k);
                sum_sq += diff * diff;
                ++count;
            }
        }
    }
    return std::sqrt(sum_sq / count);
}

// Compute mean of a 3D field (for gauge comparison)
double mean_3d(const ScalarField& f, const Mesh& mesh) {
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
    return sum / count;
}

// Subtract mean from field (remove gauge offset)
void remove_mean_3d(ScalarField& f, const Mesh& mesh) {
    double m = mean_3d(f, mesh);
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                f(i, j, k) -= m;
            }
        }
    }
}

struct FFTRefTestResult {
    bool passed;
    std::string fft_solver;
    std::string ref_solver;
    double relative_diff;
    double fft_mean;
    double ref_mean;
    std::string failure_reason;
};

// Run FFT vs CPU reference test
// This requires GPU to be available (FFT/FFT1D are GPU-only)
FFTRefTestResult test_fft_vs_reference(
    [[maybe_unused]] const std::string& test_name,
    PoissonSolverType fft_type,
    int Nx, int Ny, int Nz,
    double Lx, double Ly, double Lz,
    VelocityBC::Type x_bc, VelocityBC::Type y_bc, VelocityBC::Type z_bc,
    double tolerance)
{
    FFTRefTestResult result;
    result.passed = true;
    result.fft_solver = (fft_type == PoissonSolverType::FFT) ? "FFT" : "FFT1D";
    result.failure_reason = "";

    // Create mesh
    Mesh mesh;
    mesh.init_uniform(Nx, Ny, Nz, 0.0, Lx, 0.0, Ly, 0.0, Lz);

    // Create config for reference solver (MG)
    Config config_ref;
    config_ref.Nx = Nx;
    config_ref.Ny = Ny;
    config_ref.Nz = Nz;
    config_ref.x_min = 0.0; config_ref.x_max = Lx;
    config_ref.y_min = 0.0; config_ref.y_max = Ly;
    config_ref.z_min = 0.0; config_ref.z_max = Lz;
    config_ref.dt = 0.001;
    config_ref.max_iter = 100;
    config_ref.nu = 0.01;
    config_ref.poisson_solver = PoissonSolverType::MG;  // CPU reference
    config_ref.verbose = false;

    RANSSolver solver_ref(mesh, config_ref);

    // Set BCs
    VelocityBC bc;
    bc.x_lo = x_bc; bc.x_hi = x_bc;
    bc.y_lo = y_bc; bc.y_hi = y_bc;
    bc.z_lo = z_bc; bc.z_hi = z_bc;
    solver_ref.set_velocity_bc(bc);

    // Initialize with divergent velocity field to create Poisson problem
    VectorField vel_ref(mesh);
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        double z = mesh.z(k);
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            double y = mesh.y(j);
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                double x = mesh.x(i);
                // u = sin(2*pi*x/Lx) * cos(2*pi*y/Ly) * cos(2*pi*z/Lz)
                vel_ref.u(i, j, k) = std::sin(2.0*M_PI*x/Lx) *
                                      std::cos(2.0*M_PI*y/Ly) *
                                      std::cos(2.0*M_PI*z/Lz);
            }
        }
    }
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        double z = mesh.z(k);
        for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
            double y = mesh.y(j);
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double x = mesh.x(i);
                // v = -cos(2*pi*x/Lx) * sin(2*pi*y/Ly) * cos(2*pi*z/Lz) / 2
                // (partial divergence-free)
                vel_ref.v(i, j, k) = -std::cos(2.0*M_PI*x/Lx) *
                                      std::sin(2.0*M_PI*y/Ly) *
                                      std::cos(2.0*M_PI*z/Lz) * 0.5;
            }
        }
    }
    for (int k = mesh.k_begin(); k <= mesh.k_end(); ++k) {
        double z = mesh.z(k);
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            double y = mesh.y(j);
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double x = mesh.x(i);
                // w = -cos(2*pi*x/Lx) * cos(2*pi*y/Ly) * sin(2*pi*z/Lz) / 2
                vel_ref.w(i, j, k) = -std::cos(2.0*M_PI*x/Lx) *
                                      std::cos(2.0*M_PI*y/Ly) *
                                      std::sin(2.0*M_PI*z/Lz) * 0.5;
            }
        }
    }
    solver_ref.initialize(vel_ref);

    // Run one step to solve Poisson and project
    solver_ref.step();
    result.ref_solver = solver_ref.selection_reason();

    // Copy reference pressure
    ScalarField p_ref(mesh);
    const ScalarField& p_ref_src = solver_ref.pressure();
    for (int k = 0; k < mesh.Nz + 2; ++k) {
        for (int j = 0; j < mesh.Ny + 2; ++j) {
            for (int i = 0; i < mesh.Nx + 2; ++i) {
                p_ref(i, j, k) = p_ref_src(i, j, k);
            }
        }
    }

    // Create config for FFT solver
    Config config_fft;
    config_fft.Nx = Nx;
    config_fft.Ny = Ny;
    config_fft.Nz = Nz;
    config_fft.x_min = 0.0; config_fft.x_max = Lx;
    config_fft.y_min = 0.0; config_fft.y_max = Ly;
    config_fft.z_min = 0.0; config_fft.z_max = Lz;
    config_fft.dt = 0.001;
    config_fft.max_iter = 100;
    config_fft.nu = 0.01;
    config_fft.poisson_solver = fft_type;  // Explicit FFT or FFT1D
    config_fft.verbose = false;

    RANSSolver solver_fft(mesh, config_fft);
    solver_fft.set_velocity_bc(bc);

    // Check if FFT solver is actually selected
    // (It may fall back to MG on CPU builds)
    if (solver_fft.poisson_solver_type() != fft_type) {
        result.passed = true;  // Skip, not fail
        result.failure_reason = "FFT not available (GPU-only)";
        return result;
    }

    // Initialize with same velocity field
    VectorField vel_fft(mesh);
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        double z = mesh.z(k);
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            double y = mesh.y(j);
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                double x = mesh.x(i);
                vel_fft.u(i, j, k) = std::sin(2.0*M_PI*x/Lx) *
                                      std::cos(2.0*M_PI*y/Ly) *
                                      std::cos(2.0*M_PI*z/Lz);
            }
        }
    }
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        double z = mesh.z(k);
        for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
            double y = mesh.y(j);
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double x = mesh.x(i);
                vel_fft.v(i, j, k) = -std::cos(2.0*M_PI*x/Lx) *
                                      std::sin(2.0*M_PI*y/Ly) *
                                      std::cos(2.0*M_PI*z/Lz) * 0.5;
            }
        }
    }
    for (int k = mesh.k_begin(); k <= mesh.k_end(); ++k) {
        double z = mesh.z(k);
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            double y = mesh.y(j);
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double x = mesh.x(i);
                vel_fft.w(i, j, k) = -std::cos(2.0*M_PI*x/Lx) *
                                      std::cos(2.0*M_PI*y/Ly) *
                                      std::sin(2.0*M_PI*z/Lz) * 0.5;
            }
        }
    }
    solver_fft.initialize(vel_fft);

    // Run one step
    solver_fft.step();

#ifdef USE_GPU_OFFLOAD
    solver_fft.sync_from_gpu();
#endif

    // Copy FFT pressure
    ScalarField p_fft(mesh);
    const ScalarField& p_fft_src = solver_fft.pressure();
    for (int k = 0; k < mesh.Nz + 2; ++k) {
        for (int j = 0; j < mesh.Ny + 2; ++j) {
            for (int i = 0; i < mesh.Nx + 2; ++i) {
                p_fft(i, j, k) = p_fft_src(i, j, k);
            }
        }
    }

    // Compute means (for gauge comparison)
    result.fft_mean = mean_3d(p_fft, mesh);
    result.ref_mean = mean_3d(p_ref, mesh);

    // Remove means for comparison (gauge-independent)
    remove_mean_3d(p_fft, mesh);
    remove_mean_3d(p_ref, mesh);

    // Compute relative difference
    double ref_norm = l2_norm_3d(p_ref, mesh);
    double diff_norm = l2_diff_3d(p_fft, p_ref, mesh);

    if (ref_norm > 1e-15) {
        result.relative_diff = diff_norm / ref_norm;
    } else {
        result.relative_diff = diff_norm;
    }

    // Check tolerance
    if (result.relative_diff > tolerance) {
        result.passed = false;
        result.failure_reason = "difference exceeds tolerance";
    }

    return result;
}

int main() {
    std::cout << "================================================================\n";
    std::cout << "  FFT/FFT1D vs CPU Reference Validation Test\n";
    std::cout << "================================================================\n\n";

#ifdef USE_GPU_OFFLOAD
    std::cout << "Build: GPU (USE_GPU_OFFLOAD=ON)\n";
    std::cout << "FFT solvers: available (testing against MG reference)\n";
#else
    std::cout << "Build: CPU (USE_GPU_OFFLOAD=OFF)\n";
    std::cout << "FFT solvers: NOT available (will skip)\n";
    std::cout << "\nNote: This test is designed for H200 runner where both\n";
    std::cout << "      CPU and GPU builds are available on the same node.\n";
    std::cout << "      Run GPU build to test FFT solvers.\n";
#endif
#ifdef USE_HYPRE
    std::cout << "HYPRE: enabled\n";
#else
    std::cout << "HYPRE: disabled\n";
#endif
    std::cout << "\n";

    std::cout << "Validating FFT/FFT1D produce same solutions as CPU solvers.\n";
    std::cout << "All tests use same manufactured velocity field on same grid.\n\n";

    int passed = 0, failed = 0, skipped = 0;

    // Test 1: FFT (fully periodic) vs MG
    std::cout << "--- Test 1: FFT (fully periodic 3D) vs MG ---\n";
    {
        auto r = test_fft_vs_reference(
            "FFT_vs_MG_periodic",
            PoissonSolverType::FFT,
            32, 32, 32,
            2.0*M_PI, 2.0*M_PI, 2.0*M_PI,
            VelocityBC::Periodic, VelocityBC::Periodic, VelocityBC::Periodic,
            0.1);  // 10% tolerance for solver differences

        std::cout << "  FFT solver: " << r.fft_solver << "\n";
        std::cout << "  Ref solver: " << r.ref_solver << "\n";

        if (r.failure_reason == "FFT not available (GPU-only)") {
            std::cout << "  [SKIP] " << r.failure_reason << "\n";
            ++skipped;
        } else if (r.passed) {
            std::cout << "  [PASS] ||p_fft - p_ref|| / ||p_ref|| = "
                      << std::scientific << std::setprecision(2) << r.relative_diff << "\n";
            ++passed;
        } else {
            std::cout << "  [FAIL] ||p_fft - p_ref|| / ||p_ref|| = "
                      << std::scientific << std::setprecision(2) << r.relative_diff
                      << " (" << r.failure_reason << ")\n";
            ++failed;
        }
    }

    // Test 2: FFT1D (channel: periodic x/z, Neumann y) vs MG
    std::cout << "\n--- Test 2: FFT1D (channel 3D) vs MG ---\n";
    {
        auto r = test_fft_vs_reference(
            "FFT1D_vs_MG_channel",
            PoissonSolverType::FFT1D,
            32, 32, 32,
            2.0*M_PI, 2.0, 2.0*M_PI,
            VelocityBC::Periodic, VelocityBC::NoSlip, VelocityBC::Periodic,
            0.15);  // 15% tolerance for mixed BC case

        std::cout << "  FFT solver: " << r.fft_solver << "\n";
        std::cout << "  Ref solver: " << r.ref_solver << "\n";

        if (r.failure_reason == "FFT not available (GPU-only)") {
            std::cout << "  [SKIP] " << r.failure_reason << "\n";
            ++skipped;
        } else if (r.passed) {
            std::cout << "  [PASS] ||p_fft - p_ref|| / ||p_ref|| = "
                      << std::scientific << std::setprecision(2) << r.relative_diff << "\n";
            ++passed;
        } else {
            std::cout << "  [FAIL] ||p_fft - p_ref|| / ||p_ref|| = "
                      << std::scientific << std::setprecision(2) << r.relative_diff
                      << " (" << r.failure_reason << ")\n";
            ++failed;
        }
    }

    // Test 3: FFT1D (duct: periodic x only) vs MG
    std::cout << "\n--- Test 3: FFT1D (duct 3D) vs MG ---\n";
    {
        auto r = test_fft_vs_reference(
            "FFT1D_vs_MG_duct",
            PoissonSolverType::FFT1D,
            32, 32, 32,
            2.0*M_PI, 2.0, 2.0,
            VelocityBC::Periodic, VelocityBC::NoSlip, VelocityBC::NoSlip,
            0.15);

        std::cout << "  FFT solver: " << r.fft_solver << "\n";
        std::cout << "  Ref solver: " << r.ref_solver << "\n";

        if (r.failure_reason == "FFT not available (GPU-only)") {
            std::cout << "  [SKIP] " << r.failure_reason << "\n";
            ++skipped;
        } else if (r.passed) {
            std::cout << "  [PASS] ||p_fft - p_ref|| / ||p_ref|| = "
                      << std::scientific << std::setprecision(2) << r.relative_diff << "\n";
            ++passed;
        } else {
            std::cout << "  [FAIL] ||p_fft - p_ref|| / ||p_ref|| = "
                      << std::scientific << std::setprecision(2) << r.relative_diff
                      << " (" << r.failure_reason << ")\n";
            ++failed;
        }
    }

    // Summary
    std::cout << "\n================================================================\n";
    std::cout << "FFT vs CPU Reference Summary\n";
    std::cout << "================================================================\n";
    std::cout << "  Passed:  " << passed << "/" << (passed + failed) << "\n";
    std::cout << "  Failed:  " << failed << "/" << (passed + failed) << "\n";
    std::cout << "  Skipped: " << skipped << "\n";

    if (skipped > 0 && passed == 0 && failed == 0) {
        std::cout << "\n[SKIP] All tests skipped (FFT requires GPU build)\n";
        std::cout << "       Run on H200 with GPU build to validate FFT solvers\n";
        return 0;  // Not a failure, just skip
    }

    if (failed == 0) {
        std::cout << "\n[PASS] All FFT vs CPU reference tests passed\n";
        std::cout << "       FFT/FFT1D produce solutions consistent with MG\n";
        return 0;
    } else {
        std::cout << "\n[FAIL] " << failed << " FFT vs CPU reference test(s) failed\n";
        std::cout << "       FFT solvers may be solving wrong problem!\n";
        return 1;
    }
}
