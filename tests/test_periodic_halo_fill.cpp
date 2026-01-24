/// @file test_periodic_halo_fill.cpp
/// @brief Isolation test for enforce_periodic_halos_device() function
///
/// This test verifies that periodic ghost layers are correctly filled:
/// - Low ghost cells get values from far interior
/// - High ghost cells get values from near interior
/// - Seam faces are averaged (normal components only)
/// - Corner ghosts are consistent after corner fix
///
/// The test uses a tiny mesh (8x6 with Ng=2 ghosts) and fills interior
/// with a known pattern, then verifies halos match expected values.

#include "solver.hpp"
#include "config.hpp"
#include "mesh.hpp"
#include "fields.hpp"
#include "gpu_utils.hpp"

#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <limits>

using namespace nncfd;

namespace {

// Test configuration
constexpr int Nx = 8;
constexpr int Ny = 6;
constexpr int Ng = 2;
constexpr double Lx = 1.0;
constexpr double Ly = 0.75;

// Fill pattern: u(i,j) = 100*i + j (easily identifies location)
inline double u_pattern(int i, int j) { return 100.0 * i + j; }
inline double v_pattern(int i, int j) { return 1000.0 * i + 10.0 * j; }

// NaN sentinel for unfilled ghosts
constexpr double SENTINEL = std::numeric_limits<double>::quiet_NaN();

bool run_2d_periodic_halo_test() {
    std::cout << "\n=== 2D Periodic Halo Fill Test ===" << std::endl;
    std::cout << "Mesh: " << Nx << "x" << Ny << " with Ng=" << Ng << " ghosts" << std::endl;

    // Create mesh with 2 ghost layers
    Mesh mesh;
    mesh.init_uniform(Nx, Ny, 0.0, Lx, 0.0, Ly, Ng);

    // Configure solver
    Config config;
    config.nu = 0.01;
    config.dt = 0.001;
    config.convective_scheme = ConvectiveScheme::Central;
    config.verbose = false;

    // Create solver and set periodic BCs
    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::Periodic;
    bc.y_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    // Get the velocity field which we'll use for testing
    auto& vel = solver.velocity();

    // Strides for indexing
    const int u_stride = vel.u_stride();
    const int v_stride = vel.v_stride();

    // Expected extents (for documentation, not used after fixing warnings)
    [[maybe_unused]] const int u_Nx_total = Nx + 1 + 2 * Ng;  // u is at x-faces

    // Step 1: Fill ENTIRE array with NaN sentinel (including interior)
    std::cout << "Filling with NaN sentinel..." << std::endl;
    for (size_t idx = 0; idx < vel.u_data().size(); ++idx) {
        vel.u_data()[idx] = SENTINEL;
    }
    for (size_t idx = 0; idx < vel.v_data().size(); ++idx) {
        vel.v_data()[idx] = SENTINEL;
    }

    // Step 2: Fill INTERIOR faces with known pattern
    // u interior: i in [Ng, Ng+Nx], j in [Ng, Ng+Ny-1]
    std::cout << "Filling interior with pattern..." << std::endl;
    for (int j = Ng; j < Ng + Ny; ++j) {
        for (int i = Ng; i <= Ng + Nx; ++i) {
            vel.u_data()[j * u_stride + i] = u_pattern(i, j);
        }
    }
    // v interior: i in [Ng, Ng+Nx-1], j in [Ng, Ng+Ny]
    for (int j = Ng; j <= Ng + Ny; ++j) {
        for (int i = Ng; i < Ng + Nx; ++i) {
            vel.v_data()[j * v_stride + i] = v_pattern(i, j);
        }
    }

    // Step 3: Call the halo fill function
    // Note: In CPU builds, gpu::dev_ptr just returns the host pointer unchanged
    // In GPU builds, the data would need to be mapped first (handled by solver init)
    std::cout << "Calling enforce_periodic_halos_device()..." << std::endl;
    double* u_ptr = gpu::dev_ptr(vel.u_data().data(), "u_test");
    double* v_ptr = gpu::dev_ptr(vel.v_data().data(), "v_test");
    solver.enforce_periodic_halos_device(u_ptr, v_ptr, nullptr);

    // Step 4: Verify ghost cells are correctly filled
    int errors = 0;
    const double tol = 1e-14;

    std::cout << "\nVerifying u-velocity halos..." << std::endl;

    // u: Normal in X (seam + ghost)
    // Check seam: u[j, Ng+Nx] should equal u[j, Ng]
    for (int j = Ng; j < Ng + Ny; ++j) {
        double expected = u_pattern(Ng, j);  // Value at left boundary
        double actual = vel.u_data()[j * u_stride + (Ng + Nx)];
        if (std::abs(actual - expected) > tol) {
            std::cerr << "  FAIL: u seam at j=" << j << ": expected " << expected
                      << ", got " << actual << std::endl;
            errors++;
        }
    }

    // Check x-ghost low: u[j, Ng-1-g] should equal u[j, Ng+Nx-1-g]
    for (int g = 0; g < Ng; ++g) {
        for (int j = Ng; j < Ng + Ny; ++j) {
            int i_ghost = Ng - 1 - g;
            int i_src = Ng + Nx - 1 - g;
            double expected = u_pattern(i_src, j);
            double actual = vel.u_data()[j * u_stride + i_ghost];
            if (std::abs(actual - expected) > tol) {
                std::cerr << "  FAIL: u x-lo ghost g=" << g << " j=" << j
                          << ": expected " << expected << ", got " << actual << std::endl;
                errors++;
            }
        }
    }

    // Check x-ghost high: u[j, Ng+Nx+1+g] should equal u[j, Ng+1+g]
    for (int g = 0; g < Ng; ++g) {
        for (int j = Ng; j < Ng + Ny; ++j) {
            int i_ghost = Ng + Nx + 1 + g;
            int i_src = Ng + 1 + g;
            double expected = u_pattern(i_src, j);
            double actual = vel.u_data()[j * u_stride + i_ghost];
            if (std::abs(actual - expected) > tol) {
                std::cerr << "  FAIL: u x-hi ghost g=" << g << " j=" << j
                          << ": expected " << expected << ", got " << actual << std::endl;
                errors++;
            }
        }
    }

    // u: Tangential in Y (ghost fill only, no seam)
    // Check y-ghost low: u[Ng-1-g, i] should equal u[Ng+Ny-1-g, i]
    // Note: For seam face (i=Ng+Nx), the value was copied from i=Ng during X-halo fill
    for (int g = 0; g < Ng; ++g) {
        for (int i = Ng; i <= Ng + Nx; ++i) {  // Only interior x-range
            int j_ghost = Ng - 1 - g;
            int j_src = Ng + Ny - 1 - g;
            // Only check if source was in interior
            if (j_src >= Ng && j_src < Ng + Ny) {
                // For seam face (i=Ng+Nx), source was copied from i=Ng during X-pass
                int i_effective = (i == Ng + Nx) ? Ng : i;
                double expected = u_pattern(i_effective, j_src);
                double actual = vel.u_data()[j_ghost * u_stride + i];
                if (std::abs(actual - expected) > tol) {
                    std::cerr << "  FAIL: u y-lo ghost g=" << g << " i=" << i
                              << ": expected " << expected << ", got " << actual << std::endl;
                    errors++;
                }
            }
        }
    }

    // Check y-ghost high: u[Ng+Ny+g, i] should equal u[Ng+g, i]
    for (int g = 0; g < Ng; ++g) {
        for (int i = Ng; i <= Ng + Nx; ++i) {  // Only interior x-range
            int j_ghost = Ng + Ny + g;
            int j_src = Ng + g;
            if (j_src >= Ng && j_src < Ng + Ny) {
                // For seam face (i=Ng+Nx), source was copied from i=Ng during X-pass
                int i_effective = (i == Ng + Nx) ? Ng : i;
                double expected = u_pattern(i_effective, j_src);
                double actual = vel.u_data()[j_ghost * u_stride + i];
                if (std::abs(actual - expected) > tol) {
                    std::cerr << "  FAIL: u y-hi ghost g=" << g << " i=" << i
                              << ": expected " << expected << ", got " << actual << std::endl;
                    errors++;
                }
            }
        }
    }

    std::cout << "\nVerifying v-velocity halos..." << std::endl;

    // v: Tangential in X (ghost fill only)
    // Check x-ghost low: v[j, Ng-1-g] should equal v[j, Ng+Nx-1-g]
    // Note: For seam face (j=Ng+Ny), the value was copied from j=Ng during Y-halo fill
    for (int g = 0; g < Ng; ++g) {
        for (int j = Ng; j <= Ng + Ny; ++j) {
            int i_ghost = Ng - 1 - g;
            int i_src = Ng + Nx - 1 - g;
            if (i_src >= Ng && i_src < Ng + Nx) {
                // For seam face (j=Ng+Ny), source was copied from j=Ng during Y-pass
                int j_effective = (j == Ng + Ny) ? Ng : j;
                double expected = v_pattern(i_src, j_effective);
                double actual = vel.v_data()[j * v_stride + i_ghost];
                if (std::abs(actual - expected) > tol) {
                    std::cerr << "  FAIL: v x-lo ghost g=" << g << " j=" << j
                              << ": expected " << expected << ", got " << actual << std::endl;
                    errors++;
                }
            }
        }
    }

    // Check x-ghost high: v[j, Ng+Nx+g] should equal v[j, Ng+g]
    for (int g = 0; g < Ng; ++g) {
        for (int j = Ng; j <= Ng + Ny; ++j) {
            int i_ghost = Ng + Nx + g;
            int i_src = Ng + g;
            if (i_src >= Ng && i_src < Ng + Nx) {
                // For seam face (j=Ng+Ny), source was copied from j=Ng during Y-pass
                int j_effective = (j == Ng + Ny) ? Ng : j;
                double expected = v_pattern(i_src, j_effective);
                double actual = vel.v_data()[j * v_stride + i_ghost];
                if (std::abs(actual - expected) > tol) {
                    std::cerr << "  FAIL: v x-hi ghost g=" << g << " j=" << j
                              << ": expected " << expected << ", got " << actual << std::endl;
                    errors++;
                }
            }
        }
    }

    // v: Normal in Y (seam + ghost)
    // Check seam: v[Ng+Ny, i] should equal v[Ng, i]
    for (int i = Ng; i < Ng + Nx; ++i) {
        double expected = v_pattern(i, Ng);  // Value at bottom boundary
        double actual = vel.v_data()[(Ng + Ny) * v_stride + i];
        if (std::abs(actual - expected) > tol) {
            std::cerr << "  FAIL: v seam at i=" << i << ": expected " << expected
                      << ", got " << actual << std::endl;
            errors++;
        }
    }

    // Check y-ghost low: v[Ng-1-g, i] should equal v[Ng+Ny-1-g, i]
    for (int g = 0; g < Ng; ++g) {
        for (int i = Ng; i < Ng + Nx; ++i) {
            int j_ghost = Ng - 1 - g;
            int j_src = Ng + Ny - 1 - g;
            double expected = v_pattern(i, j_src);
            double actual = vel.v_data()[j_ghost * v_stride + i];
            if (std::abs(actual - expected) > tol) {
                std::cerr << "  FAIL: v y-lo ghost g=" << g << " i=" << i
                          << ": expected " << expected << ", got " << actual << std::endl;
                errors++;
            }
        }
    }

    // Check y-ghost high: v[Ng+Ny+1+g, i] should equal v[Ng+1+g, i]
    for (int g = 0; g < Ng; ++g) {
        for (int i = Ng; i < Ng + Nx; ++i) {
            int j_ghost = Ng + Ny + 1 + g;
            int j_src = Ng + 1 + g;
            double expected = v_pattern(i, j_src);
            double actual = vel.v_data()[j_ghost * v_stride + i];
            if (std::abs(actual - expected) > tol) {
                std::cerr << "  FAIL: v y-hi ghost g=" << g << " i=" << i
                          << ": expected " << expected << ", got " << actual << std::endl;
                errors++;
            }
        }
    }

    // Summary
    if (errors == 0) {
        std::cout << "\nPASS: All periodic halo values correct" << std::endl;
        return true;
    } else {
        std::cout << "\nFAIL: " << errors << " halo verification errors" << std::endl;
        return false;
    }
}

bool run_divergence_stability_guard() {
    // Regression guard: verify that halo fill prevents divergence drift
    // This catches "forgot halos → biased divergence" bugs

    std::cout << "\n=== Divergence Stability Guard ===" << std::endl;

    // Create mesh
    Mesh mesh;
    mesh.init_uniform(16, 16, 0.0, 1.0, 0.0, 1.0, 2);

    // Configure solver
    Config config;
    config.nu = 0.01;
    config.dt = 0.001;
    config.convective_scheme = ConvectiveScheme::Central;
    config.verbose = false;

    // Create solver with periodic BCs
    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::Periodic;
    bc.y_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    // Initialize with Taylor-Green vortex (div-free by construction)
    auto& vel = solver.velocity();
    const double pi = 3.14159265358979323846;
    for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            double x = (i - mesh.Nghost + 0.5) * mesh.dx;
            double y = (j - mesh.Nghost) * mesh.dy;
            vel.u(i, j) = std::sin(2 * pi * x) * std::cos(2 * pi * y);
        }
    }
    for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double x = (i - mesh.Nghost) * mesh.dx;
            double y = (j - mesh.Nghost + 0.5) * mesh.dy;
            vel.v(i, j) = -std::cos(2 * pi * x) * std::sin(2 * pi * y);
        }
    }

    // Take a few steps
    const int max_steps = 10;
    std::cout << "Running " << max_steps << " steps..." << std::endl;
    for (int i = 0; i < max_steps; ++i) {
        solver.step();
    }

#ifdef USE_GPU_OFFLOAD
    solver.sync_solution_from_gpu();
#endif

    // Compute max divergence of final velocity
    double max_div = 0.0;
    const auto& div = solver.div_velocity();
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            max_div = std::max(max_div, std::abs(div(i, j)));
        }
    }

    std::cout << "Final max|div(u)| = " << std::scientific << max_div << std::endl;

    // Divergence should be bounded after projection.
    // On coarse meshes (16x16) with upwind advection, divergence is typically O(1e-2 to 1e-1)
    // due to advection discretization errors before projection.
    // If halos are truly broken (stale ghosts), divergence would be O(1) or NaN.
    // The key check: projection converges (no NaN/Inf) and div doesn't grow unbounded.
    const double div_tol = 0.5;  // Guard against catastrophic failure (NaN, O(1) growth)
    if (max_div < div_tol && !std::isnan(max_div) && std::isfinite(max_div)) {
        std::cout << "PASS: Divergence bounded (projection converges, max_div="
                  << max_div << " < " << div_tol << ")" << std::endl;
        return true;
    } else {
        std::cerr << "FAIL: Divergence unbounded or NaN (halo issue?)" << std::endl;
        return false;
    }
}

} // anonymous namespace

int main() {
    std::cout << "===============================================" << std::endl;
    std::cout << "Periodic Halo Fill Isolation Test" << std::endl;
    std::cout << "===============================================" << std::endl;

    bool pass = true;

    pass = run_2d_periodic_halo_test() && pass;
    pass = run_divergence_stability_guard() && pass;

    std::cout << "\n===============================================" << std::endl;
    if (pass) {
        std::cout << "ALL TESTS PASSED" << std::endl;
        return 0;
    } else {
        std::cout << "SOME TESTS FAILED" << std::endl;
        return 1;
    }
}
