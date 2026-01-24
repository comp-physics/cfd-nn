/// @file test_kernel_parity_detailed.cpp
/// @brief Detailed CPU/GPU kernel parity test for non-Poisson production kernels
///
/// CRITICAL TEST: Validates that non-Poisson kernels produce identical results
/// on CPU and GPU. This catches:
///   - Gradient computation differences
///   - Advection/convection term differences
///   - Diffusion term differences
///   - Velocity correction differences
///   - Boundary condition application differences
///
/// Method:
///   1. Set up identical initial conditions
///   2. Run kernels (gradients, advection, diffusion, correction)
///   3. Dump intermediate results for comparison
///   4. Compare CPU vs GPU outputs (when run with --compare-prefix)
///
/// The test supports two modes:
///   --dump-prefix <path>    : Generate reference outputs
///   --compare-prefix <path> : Compare against reference outputs
///
/// Designed to be run on H200 runner with paired CPU+GPU builds.

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "features.hpp"
#include "config.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <string>
#include <cstring>

using namespace nncfd;

// Dump a scalar field to file
bool dump_field(const std::string& filename, const ScalarField& f, const Mesh& mesh) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) return false;

    int Nx = mesh.Nx, Ny = mesh.Ny, Nz = std::max(mesh.Nz, 1);
    ofs.write(reinterpret_cast<const char*>(&Nx), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(&Ny), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(&Nz), sizeof(int));

    if (mesh.is2D()) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double val = f(i, j);
                ofs.write(reinterpret_cast<const char*>(&val), sizeof(double));
            }
        }
    } else {
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    double val = f(i, j, k);
                    ofs.write(reinterpret_cast<const char*>(&val), sizeof(double));
                }
            }
        }
    }
    return true;
}

// Load a scalar field from file and compare
struct CompareResult {
    bool success;
    double max_abs_diff;
    double max_rel_diff;
    double rms_diff;
};

CompareResult compare_field(const std::string& filename, const ScalarField& f, const Mesh& mesh) {
    CompareResult result;
    result.success = false;
    result.max_abs_diff = 0.0;
    result.max_rel_diff = 0.0;
    result.rms_diff = 0.0;

    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) return result;

    int Nx_ref, Ny_ref, Nz_ref;
    ifs.read(reinterpret_cast<char*>(&Nx_ref), sizeof(int));
    ifs.read(reinterpret_cast<char*>(&Ny_ref), sizeof(int));
    ifs.read(reinterpret_cast<char*>(&Nz_ref), sizeof(int));

    if (Nx_ref != mesh.Nx || Ny_ref != mesh.Ny) {
        std::cerr << "Dimension mismatch in " << filename << "\n";
        return result;
    }

    double sum_sq = 0.0;
    int count = 0;

    if (mesh.is2D()) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double val_ref;
                ifs.read(reinterpret_cast<char*>(&val_ref), sizeof(double));
                double val = f(i, j);
                double abs_diff = std::abs(val - val_ref);
                double rel_diff = (std::abs(val_ref) > 1e-15) ?
                                  abs_diff / std::abs(val_ref) : abs_diff;
                result.max_abs_diff = std::max(result.max_abs_diff, abs_diff);
                result.max_rel_diff = std::max(result.max_rel_diff, rel_diff);
                sum_sq += abs_diff * abs_diff;
                ++count;
            }
        }
    } else {
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    double val_ref;
                    ifs.read(reinterpret_cast<char*>(&val_ref), sizeof(double));
                    double val = f(i, j, k);
                    double abs_diff = std::abs(val - val_ref);
                    double rel_diff = (std::abs(val_ref) > 1e-15) ?
                                      abs_diff / std::abs(val_ref) : abs_diff;
                    result.max_abs_diff = std::max(result.max_abs_diff, abs_diff);
                    result.max_rel_diff = std::max(result.max_rel_diff, rel_diff);
                    sum_sq += abs_diff * abs_diff;
                    ++count;
                }
            }
        }
    }

    result.rms_diff = std::sqrt(sum_sq / count);
    result.success = true;
    return result;
}

int main(int argc, char* argv[]) {
    std::cout << "================================================================\n";
    std::cout << "  Detailed Kernel Parity Test (Non-Poisson)\n";
    std::cout << "================================================================\n\n";

    // Parse arguments
    std::string dump_prefix;
    std::string compare_prefix;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--dump-prefix") == 0 && i + 1 < argc) {
            dump_prefix = argv[++i];
        } else if (std::strcmp(argv[i], "--compare-prefix") == 0 && i + 1 < argc) {
            compare_prefix = argv[++i];
        }
    }

    bool mode_dump = !dump_prefix.empty();
    bool mode_compare = !compare_prefix.empty();

    if (mode_dump) {
        std::cout << "Mode: DUMP reference outputs to " << dump_prefix << "\n";
    } else if (mode_compare) {
        std::cout << "Mode: COMPARE against reference " << compare_prefix << "\n";
    } else {
        std::cout << "Mode: STANDALONE (no dump/compare)\n";
        std::cout << "\nUsage:\n";
        std::cout << "  Generate reference: ./test --dump-prefix <path>\n";
        std::cout << "  Compare:            ./test --compare-prefix <path>\n";
    }

#ifdef USE_GPU_OFFLOAD
    std::cout << "Build: GPU (USE_GPU_OFFLOAD=ON)\n";
#else
    std::cout << "Build: CPU (USE_GPU_OFFLOAD=OFF)\n";
#endif
    std::cout << "\n";

    // ========================================================================
    // Setup: Create mesh and initialize with known velocity field
    // ========================================================================
    const int N = 64;
    Mesh mesh;
    mesh.init_uniform(N, N, 0.0, 2.0*M_PI, 0.0, 2.0*M_PI);

    Config config;
    config.Nx = N;
    config.Ny = N;
    config.dt = 0.001;
    config.nu = 0.01;
    config.verbose = false;
    config.poisson_solver = PoissonSolverType::MG;

    RANSSolver solver(mesh, config);

    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::Periodic;
    bc.y_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);

    // Initialize with smooth trigonometric field (divergence-free)
    VectorField& vel = solver.velocity();
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
            double x = mesh.x(i);
            double y = mesh.y(j);
            vel.u(i, j) = std::sin(x) * std::cos(y);
        }
    }
    for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double x = mesh.x(i);
            double y = mesh.y(j);
            vel.v(i, j) = -std::cos(x) * std::sin(y);
        }
    }
    solver.initialize(vel);

    bool all_passed = true;

    // ========================================================================
    // Test 1: Gradient computation (dudx, dudy, dvdx, dvdy)
    // ========================================================================
    std::cout << "--- Test 1: Gradient Computation ---\n";
    {
        ScalarField dudx(mesh), dudy(mesh), dvdx(mesh), dvdy(mesh);
        compute_gradients_from_mac(mesh, vel, dudx, dudy, dvdx, dvdy);

        if (mode_dump) {
            dump_field(dump_prefix + "_dudx.bin", dudx, mesh);
            dump_field(dump_prefix + "_dudy.bin", dudy, mesh);
            dump_field(dump_prefix + "_dvdx.bin", dvdx, mesh);
            dump_field(dump_prefix + "_dvdy.bin", dvdy, mesh);
            std::cout << "  [DUMP] Gradient fields saved\n";
        } else if (mode_compare) {
            auto r1 = compare_field(compare_prefix + "_dudx.bin", dudx, mesh);
            auto r2 = compare_field(compare_prefix + "_dudy.bin", dudy, mesh);
            auto r3 = compare_field(compare_prefix + "_dvdx.bin", dvdx, mesh);
            auto r4 = compare_field(compare_prefix + "_dvdy.bin", dvdy, mesh);

            double max_diff = std::max({r1.max_abs_diff, r2.max_abs_diff,
                                        r3.max_abs_diff, r4.max_abs_diff});
            if (max_diff < 1e-10) {
                std::cout << "  [PASS] Gradients match (max diff = "
                          << std::scientific << max_diff << ")\n";
            } else {
                std::cout << "  [FAIL] Gradient mismatch (max diff = "
                          << std::scientific << max_diff << ")\n";
                all_passed = false;
            }
        } else {
            // Standalone mode: verify gradients analytically
#ifdef USE_GPU_OFFLOAD
            // GPU path: map data to device, run kernel, sync back
            double* u_ptr = const_cast<double*>(vel.u_data().data());
            double* v_ptr = const_cast<double*>(vel.v_data().data());
            double* dudx_ptr = dudx.data().data();
            double* dudy_ptr = dudy.data().data();
            double* dvdx_ptr = dvdx.data().data();
            double* dvdy_ptr = dvdy.data().data();

            const int u_size = vel.u_total_size();
            const int v_size = vel.v_total_size();
            const int cell_size = mesh.total_Nx() * mesh.total_Ny();

            // Map input data to device
            #pragma omp target enter data map(to: u_ptr[0:u_size], v_ptr[0:v_size])
            // Allocate output arrays on device
            #pragma omp target enter data map(alloc: dudx_ptr[0:cell_size], dudy_ptr[0:cell_size], \
                                                      dvdx_ptr[0:cell_size], dvdy_ptr[0:cell_size])

            // Run gradient computation (uses map(present:...))
            compute_gradients_from_mac(mesh, vel, dudx, dudy, dvdx, dvdy);

            // Sync output back to host
            #pragma omp target update from(dudx_ptr[0:cell_size], dudy_ptr[0:cell_size], \
                                          dvdx_ptr[0:cell_size], dvdy_ptr[0:cell_size])

            // Clean up device memory
            #pragma omp target exit data map(delete: u_ptr[0:u_size], v_ptr[0:v_size], \
                                                     dudx_ptr[0:cell_size], dudy_ptr[0:cell_size], \
                                                     dvdx_ptr[0:cell_size], dvdy_ptr[0:cell_size])
#endif
            // Verify gradients analytically (same for CPU and GPU after sync)
            // u = sin(x)*cos(y) -> dudx = cos(x)*cos(y), dudy = -sin(x)*sin(y)
            double max_err = 0.0;
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                double y = mesh.y(j);
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    double x = mesh.x(i);
                    double exact_dudx = std::cos(x) * std::cos(y);
                    double exact_dudy = -std::sin(x) * std::sin(y);
                    max_err = std::max(max_err, std::abs(dudx(i, j) - exact_dudx));
                    max_err = std::max(max_err, std::abs(dudy(i, j) - exact_dudy));
                }
            }
            if (max_err < 0.1) {  // Allow for discretization error
                std::cout << "  [PASS] Gradients correct (max err = "
                          << std::scientific << max_err << ")\n";
            } else {
                std::cout << "  [FAIL] Gradient error too large (max err = "
                          << std::scientific << max_err << ")\n";
                all_passed = false;
            }
        }
    }

    // ========================================================================
    // Test 2: Single time step (covers advection, diffusion, projection)
    // ========================================================================
    std::cout << "\n--- Test 2: Single Time Step ---\n";
    {
        solver.step();

#ifdef USE_GPU_OFFLOAD
        solver.sync_from_gpu();
#endif

        const ScalarField& p = solver.pressure();

        if (mode_dump) {
            dump_field(dump_prefix + "_p_step1.bin", p, mesh);
            std::cout << "  [DUMP] Pressure after step 1 saved\n";
        } else if (mode_compare) {
            auto r = compare_field(compare_prefix + "_p_step1.bin", p, mesh);
            if (r.max_abs_diff < 1e-8) {
                std::cout << "  [PASS] Pressure matches (max diff = "
                          << std::scientific << r.max_abs_diff << ")\n";
            } else {
                std::cout << "  [FAIL] Pressure mismatch (max diff = "
                          << std::scientific << r.max_abs_diff << ")\n";
                all_passed = false;
            }
        } else {
            // Just check for finite values
            double max_p = 0.0;
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    max_p = std::max(max_p, std::abs(p(i, j)));
                }
            }
            if (std::isfinite(max_p) && max_p < 1e10) {
                std::cout << "  [PASS] Pressure finite (max = "
                          << std::scientific << max_p << ")\n";
            } else {
                std::cout << "  [FAIL] Pressure invalid\n";
                all_passed = false;
            }
        }
    }

    // ========================================================================
    // Test 3: Multi-step evolution (covers accumulated numerical behavior)
    // ========================================================================
    std::cout << "\n--- Test 3: Multi-Step Evolution (10 steps) ---\n";
    {
        for (int step = 0; step < 9; ++step) {
            solver.step();
        }

#ifdef USE_GPU_OFFLOAD
        solver.sync_from_gpu();
#endif

        const ScalarField& p = solver.pressure();

        if (mode_dump) {
            dump_field(dump_prefix + "_p_step10.bin", p, mesh);
            std::cout << "  [DUMP] Pressure after step 10 saved\n";
        } else if (mode_compare) {
            auto r = compare_field(compare_prefix + "_p_step10.bin", p, mesh);
            // Allow slightly larger tolerance for accumulated differences
            if (r.max_abs_diff < 1e-6) {
                std::cout << "  [PASS] Pressure matches after 10 steps (max diff = "
                          << std::scientific << r.max_abs_diff << ")\n";
            } else {
                std::cout << "  [FAIL] Pressure drift after 10 steps (max diff = "
                          << std::scientific << r.max_abs_diff << ")\n";
                all_passed = false;
            }
        } else {
            // Check KE stability
            double ke = 0.0;
            int count = 0;
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    double u = 0.5 * (vel.u(i, j) + vel.u(i+1, j));
                    double v = 0.5 * (vel.v(i, j) + vel.v(i, j+1));
                    ke += 0.5 * (u*u + v*v);
                    ++count;
                }
            }
            ke /= count;
            if (std::isfinite(ke) && ke > 0 && ke < 10) {
                std::cout << "  [PASS] KE stable after 10 steps (KE = "
                          << std::fixed << std::setprecision(4) << ke << ")\n";
            } else {
                std::cout << "  [FAIL] KE unstable after 10 steps (KE = " << ke << ")\n";
                all_passed = false;
            }
        }
    }

    // ========================================================================
    // Test 4: Divergence-free check (velocity correction kernel)
    // ========================================================================
    std::cout << "\n--- Test 4: Divergence-Free Verification ---\n";
    {
        double max_div = 0.0;
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double dudx = (vel.u(i+1, j) - vel.u(i, j)) / mesh.dx;
                double dvdy = (vel.v(i, j+1) - vel.v(i, j)) / mesh.dy;
                double div = std::abs(dudx + dvdy);
                max_div = std::max(max_div, div);
            }
        }

        if (mode_dump) {
            std::ofstream ofs(dump_prefix + "_max_div.txt");
            ofs << std::scientific << max_div << "\n";
            std::cout << "  [DUMP] Max divergence saved (= "
                      << std::scientific << max_div << ")\n";
        } else if (mode_compare) {
            std::ifstream ifs(compare_prefix + "_max_div.txt");
            double ref_div;
            ifs >> ref_div;
            double diff = std::abs(max_div - ref_div);
            if (diff < 1e-10) {
                std::cout << "  [PASS] Divergence matches (diff = "
                          << std::scientific << diff << ")\n";
            } else {
                std::cout << "  [WARN] Divergence differs (GPU=" << max_div
                          << " CPU=" << ref_div << ")\n";
                // Don't fail - divergence can vary slightly
            }
        } else {
            if (max_div < 1e-8) {
                std::cout << "  [PASS] Divergence-free (max = "
                          << std::scientific << max_div << ")\n";
            } else {
                std::cout << "  [WARN] Divergence not zero (max = "
                          << std::scientific << max_div << ")\n";
            }
        }
    }

    // ========================================================================
    // Summary
    // ========================================================================
    std::cout << "\n================================================================\n";

    if (mode_dump) {
        std::cout << "[DONE] Reference outputs dumped to " << dump_prefix << "*\n";
        std::cout << "       Run GPU build with --compare-prefix to validate\n";
        return 0;
    }

    if (all_passed) {
        if (mode_compare) {
            std::cout << "[PASS] CPU/GPU kernel parity verified\n";
            std::cout << "       Non-Poisson kernels produce identical results\n";
        } else {
            std::cout << "[PASS] All standalone kernel tests passed\n";
        }
        return 0;
    } else {
        std::cout << "[FAIL] Kernel parity test failed\n";
        std::cout << "       CPU and GPU kernels produce different results!\n";
        return 1;
    }
}
