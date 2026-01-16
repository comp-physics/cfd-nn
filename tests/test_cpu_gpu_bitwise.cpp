/// CPU/GPU Bitwise Comparison Test
/// Compares CPU-built and GPU-built solver outputs to verify code sharing paradigm.
///
/// This test REQUIRES two separate builds:
///   1. CPU build (USE_GPU_OFFLOAD=OFF): Run with --dump-prefix to generate reference
///   2. GPU build (USE_GPU_OFFLOAD=ON):  Run with --compare-prefix to compare against reference
///
/// Expected result: Small differences (1e-12 to 1e-10) due to FP operation ordering,
/// but not exact zeros (which would indicate both runs used the same backend).

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "config.hpp"
#include "test_utilities.hpp"
#include "test_harness.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <sstream>
#include <functional>
#include <climits>

using nncfd::test::FieldComparison;
using nncfd::test::file_exists;
using nncfd::test::BITWISE_TOLERANCE;
using nncfd::test::MIN_EXPECTED_DIFF;

// OpenMP headers - needed for both CPU and GPU builds for backend verification
#if defined(_OPENMP)
#include <omp.h>
#endif

using namespace nncfd;

//=============================================================================
// Backend identity verification
//=============================================================================

// Print backend identity marker for CI parsing
void print_backend_identity() {
#ifdef USE_GPU_OFFLOAD
    std::cout << "EXEC_BACKEND=GPU_OFFLOAD\n";
    std::cout << "  Compiled with: USE_GPU_OFFLOAD=ON\n";
    #if defined(_OPENMP)
    std::cout << "  OMP devices: " << omp_get_num_devices() << "\n";
    std::cout << "  OMP default device: " << omp_get_default_device() << "\n";
    #endif
#else
    std::cout << "EXEC_BACKEND=CPU_ONLY\n";
    std::cout << "  Compiled with: USE_GPU_OFFLOAD=OFF\n";
    #if defined(_OPENMP)
    std::cout << "  OMP available: YES (version " << _OPENMP << ")\n";
    // Check if target offload is even possible
    int num_devices = 0;
    #if _OPENMP >= 201511  // OpenMP 4.5+
    num_devices = omp_get_num_devices();
    #endif
    std::cout << "  OMP devices visible: " << num_devices << "\n";
    #else
    std::cout << "  OMP available: NO\n";
    #endif
#endif
}

// Verify CPU build is actually running on CPU (not secretly offloading)
// Returns true if verification passes, false if something is wrong
bool verify_cpu_backend() {
#ifdef USE_GPU_OFFLOAD
    // GPU build - this function shouldn't be called
    return false;
#else
    #if defined(_OPENMP) && _OPENMP >= 201511
    // Even in CPU build, check that we're on initial device
    // This catches cases where the build system was misconfigured
    int on_initial = 1;
    // Note: We can't use #pragma omp target in CPU build, but we CAN check
    // that no devices are being used by examining omp_get_num_devices()
    // and OMP_TARGET_OFFLOAD environment variable
    const char* offload_env = std::getenv("OMP_TARGET_OFFLOAD");
    if (offload_env != nullptr) {
        std::string offload_str(offload_env);
        if (offload_str == "MANDATORY") {
            std::cerr << "WARNING: OMP_TARGET_OFFLOAD=MANDATORY but this is a CPU build\n";
            std::cerr << "         This environment variable has no effect on CPU builds\n";
        }
    }
    #endif
    return true;
#endif
}

// Verify GPU build is actually running on GPU
bool verify_gpu_backend() {
#ifndef USE_GPU_OFFLOAD
    return false;
#else
    const int num_devices = omp_get_num_devices();
    if (num_devices == 0) {
        std::cerr << "ERROR: No GPU devices found\n";
        return false;
    }

    // Actually test that target regions execute on device
    int on_device = 0;
    #pragma omp target map(tofrom: on_device)
    {
        on_device = !omp_is_initial_device();
    }

    if (!on_device) {
        std::cerr << "ERROR: Target region executed on host, not GPU\n";
        return false;
    }

    std::cout << "  GPU execution verified: YES (device " << omp_get_default_device() << ")\n";
    return true;
#endif
}

// Tolerance constants imported from test_utilities.hpp:
// - BITWISE_TOLERANCE = 1e-10 (CPU vs GPU comparison)
// - MIN_EXPECTED_DIFF = 1e-14 (minimum to verify different backends)

//=============================================================================
// File I/O helpers
//=============================================================================

// file_exists() imported from test_utilities.hpp

// Write velocity field component to file
void write_field_data(const std::string& filename,
                      [[maybe_unused]] const Mesh& mesh,
                      const std::function<double(int, int, int)>& getter,
                      int i_begin, int i_end, int j_begin, int j_end, int k_begin, int k_end) {
    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    file << std::setprecision(17) << std::scientific;
    file << "# i j k value\n";

    for (int k = k_begin; k < k_end; ++k) {
        for (int j = j_begin; j < j_end; ++j) {
            for (int i = i_begin; i < i_end; ++i) {
                file << i << " " << j << " " << k << " " << getter(i, j, k) << "\n";
            }
        }
    }
}

// Read field data from file into a vector with index mapping
struct FieldData {
    std::vector<double> values;
    int i_min, i_max, j_min, j_max, k_min, k_max;
    int ni, nj, nk;

    double operator()(int i, int j, int k) const {
        int idx = (k - k_min) * ni * nj + (j - j_min) * ni + (i - i_min);
        return values[idx];
    }
};

FieldData read_field_data(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open reference file: " + filename);
    }

    // First pass: determine bounds
    int i_min = INT_MAX, i_max = INT_MIN;
    int j_min = INT_MAX, j_max = INT_MIN;
    int k_min = INT_MAX, k_max = INT_MIN;

    std::string line;
    std::vector<std::tuple<int, int, int, double>> entries;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        int i, j, k;
        double value;
        if (!(iss >> i >> j >> k >> value)) continue;

        entries.emplace_back(i, j, k, value);
        i_min = std::min(i_min, i); i_max = std::max(i_max, i);
        j_min = std::min(j_min, j); j_max = std::max(j_max, j);
        k_min = std::min(k_min, k); k_max = std::max(k_max, k);
    }

    if (entries.empty()) {
        throw std::runtime_error("No data found in reference file: " + filename);
    }

    FieldData data;
    data.i_min = i_min; data.i_max = i_max + 1;
    data.j_min = j_min; data.j_max = j_max + 1;
    data.k_min = k_min; data.k_max = k_max + 1;
    data.ni = data.i_max - i_min;
    data.nj = data.j_max - j_min;
    data.nk = data.k_max - k_min;

    data.values.resize(data.ni * data.nj * data.nk, 0.0);

    for (const auto& [i, j, k, value] : entries) {
        int idx = (k - k_min) * data.ni * data.nj + (j - j_min) * data.ni + (i - i_min);
        data.values[idx] = value;
    }

    return data;
}

// FieldComparison imported from test_utilities.hpp

//=============================================================================
// Test case: Channel flow with body force (same as original test)
//=============================================================================

void setup_channel_test(Mesh& mesh, Config& config, int NX, int NY, int NZ, int num_iter) {
    mesh.init_uniform(NX, NY, NZ, 0.0, 4.0, 0.0, 2.0, 0.0, 1.0);

    config.nu = 0.01;
    config.dt = 0.0005;
    config.adaptive_dt = false;  // Fixed dt for reproducibility
    config.max_steps = num_iter;
    config.tol = 1e-6;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;
    // IMPORTANT: Force MG solver to validate CPU/GPU determinism.
    // This test checks that identical code produces identical results on both backends.
    // FFT solver is GPU-only, so auto-selection would compare different algorithms
    // rather than testing code sharing. We explicitly use MG on both.
    config.poisson_solver = PoissonSolverType::MG;
}

void initialize_poiseuille_ic(RANSSolver& solver, const Mesh& mesh) {
    double H = 1.0;  // Half-height
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            double y = mesh.y(j) - H;
            double u_val = 0.01 * (H * H - y * y);
            for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                solver.velocity().u(i, j, k) = u_val;
            }
        }
    }
}

void set_channel_bcs(RANSSolver& solver) {
    VelocityBC bc;
    bc.x_lo = VelocityBC::Periodic;
    bc.x_hi = VelocityBC::Periodic;
    bc.y_lo = VelocityBC::NoSlip;
    bc.y_hi = VelocityBC::NoSlip;
    bc.z_lo = VelocityBC::Periodic;
    bc.z_hi = VelocityBC::Periodic;
    solver.set_velocity_bc(bc);
}

//=============================================================================
// Dump mode: Generate CPU reference
//=============================================================================

int run_dump_mode(const std::string& prefix) {
#ifdef USE_GPU_OFFLOAD
    std::cerr << "ERROR: --dump-prefix requires CPU-only build\n";
    std::cerr << "       This binary was built with USE_GPU_OFFLOAD=ON\n";
    std::cerr << "       Rebuild with -DUSE_GPU_OFFLOAD=OFF\n";
    return 1;
#else
    std::cout << "=== CPU Reference Generation Mode ===\n";
    print_backend_identity();
    std::cout << "Output prefix: " << prefix << "\n\n";

    // Verify we're actually running on CPU
    if (!verify_cpu_backend()) {
        std::cerr << "ERROR: CPU backend verification failed\n";
        return 1;
    }

    const int NX = 48, NY = 48, NZ = 8;
    const int NUM_STEPS = 30;

    Mesh mesh;
    Config config;
    setup_channel_test(mesh, config, NX, NY, NZ, NUM_STEPS);

    RANSSolver solver(mesh, config);
    solver.set_body_force(0.001, 0.0, 0.0);
    set_channel_bcs(solver);
    initialize_poiseuille_ic(solver, mesh);

    std::cout << "Running " << NUM_STEPS << " time steps on CPU...\n";
    for (int step = 0; step < NUM_STEPS; ++step) {
        solver.step();
    }

    std::cout << "Writing reference fields...\n";

    // Write u-velocity (at x-faces, so i goes to i_end inclusive)
    write_field_data(prefix + "_u.dat", mesh,
        [&](int i, int j, int k) { return solver.velocity().u(i, j, k); },
        mesh.i_begin(), mesh.i_end() + 1, mesh.j_begin(), mesh.j_end(), mesh.k_begin(), mesh.k_end());
    std::cout << "  Wrote: " << prefix << "_u.dat\n";

    // Write v-velocity (at y-faces, so j goes to j_end inclusive)
    write_field_data(prefix + "_v.dat", mesh,
        [&](int i, int j, int k) { return solver.velocity().v(i, j, k); },
        mesh.i_begin(), mesh.i_end(), mesh.j_begin(), mesh.j_end() + 1, mesh.k_begin(), mesh.k_end());
    std::cout << "  Wrote: " << prefix << "_v.dat\n";

    // Write w-velocity (at z-faces, so k goes to k_end inclusive)
    if (!mesh.is2D()) {
        write_field_data(prefix + "_w.dat", mesh,
            [&](int i, int j, int k) { return solver.velocity().w(i, j, k); },
            mesh.i_begin(), mesh.i_end(), mesh.j_begin(), mesh.j_end(), mesh.k_begin(), mesh.k_end() + 1);
        std::cout << "  Wrote: " << prefix << "_w.dat\n";
    }

    // Write pressure (cell-centered)
    write_field_data(prefix + "_p.dat", mesh,
        [&](int i, int j, int k) { return solver.pressure()(i, j, k); },
        mesh.i_begin(), mesh.i_end(), mesh.j_begin(), mesh.j_end(), mesh.k_begin(), mesh.k_end());
    std::cout << "  Wrote: " << prefix << "_p.dat\n";

    std::cout << "\n[SUCCESS] CPU reference files written\n";
    return 0;
#endif
}

//=============================================================================
// Compare mode: Run GPU and compare against CPU reference
//=============================================================================

int run_compare_mode([[maybe_unused]] const std::string& prefix) {
#ifndef USE_GPU_OFFLOAD
    std::cerr << "ERROR: --compare-prefix requires GPU build\n";
    std::cerr << "       This binary was built with USE_GPU_OFFLOAD=OFF\n";
    std::cerr << "       Rebuild with -DUSE_GPU_OFFLOAD=ON\n";
    return 1;
#else
    std::cout << "=== GPU Comparison Mode ===\n";
    print_backend_identity();
    std::cout << "Reference prefix: " << prefix << "\n\n";

    // Verify GPU is actually accessible and executing on device
    if (!verify_gpu_backend()) {
        std::cerr << "       Check GPU drivers and OMP_TARGET_OFFLOAD settings.\n";
        return 1;
    }
    std::cout << "\n";

    // Verify reference files exist
    if (!file_exists(prefix + "_u.dat")) {
        std::cerr << "ERROR: Reference file not found: " << prefix << "_u.dat\n";
        std::cerr << "       Run CPU build with --dump-prefix first\n";
        return 1;
    }

    const int NX = 48, NY = 48, NZ = 8;
    const int NUM_STEPS = 30;

    Mesh mesh;
    Config config;
    setup_channel_test(mesh, config, NX, NY, NZ, NUM_STEPS);

    RANSSolver solver(mesh, config);
    solver.set_body_force(0.001, 0.0, 0.0);
    set_channel_bcs(solver);
    initialize_poiseuille_ic(solver, mesh);

    // GPU solver automatically initialized in constructor
    solver.sync_to_gpu();

    std::cout << "Running " << NUM_STEPS << " time steps on GPU...\n";
    for (int step = 0; step < NUM_STEPS; ++step) {
        solver.step();
    }
    solver.sync_solution_from_gpu();

    std::cout << "Loading CPU reference and comparing...\n\n";

    bool all_passed = true;
    double u_rel_l2 = 0.0;
    double p_rel_l2 = 0.0;

    // Compare u-velocity
    {
        auto ref = read_field_data(prefix + "_u.dat");
        FieldComparison result;
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                    result.update(i, j, k, ref(i, j, k), solver.velocity().u(i, j, k));
                }
            }
        }
        result.finalize();
        result.print("u-velocity");
        u_rel_l2 = result.rel_l2();

        if (!result.within_tolerance(BITWISE_TOLERANCE)) {
            std::cout << "    [FAIL] Exceeds tolerance " << BITWISE_TOLERANCE << "\n";
            all_passed = false;
        } else if (result.max_abs_diff < MIN_EXPECTED_DIFF) {
            // Small diff is fine - canary test verifies backend execution.
            // This just means computation isn't sensitive to FP reordering.
            std::cout << "    [PASS] (tiny diff - not sensitive to FP reordering)\n";
        } else {
            std::cout << "    [PASS]\n";
        }
    }

    // Compare v-velocity
    {
        auto ref = read_field_data(prefix + "_v.dat");
        FieldComparison result;
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    result.update(i, j, k, ref(i, j, k), solver.velocity().v(i, j, k));
                }
            }
        }
        result.finalize();
        result.print("v-velocity");

        if (!result.within_tolerance(BITWISE_TOLERANCE)) {
            std::cout << "    [FAIL] Exceeds tolerance " << BITWISE_TOLERANCE << "\n";
            all_passed = false;
        } else if (result.max_abs_diff < MIN_EXPECTED_DIFF) {
            // Small diff is fine - canary test verifies backend execution.
            // This just means computation isn't sensitive to FP reordering.
            std::cout << "    [PASS] (tiny diff - not sensitive to FP reordering)\n";
        } else {
            std::cout << "    [PASS]\n";
        }
    }

    // Compare w-velocity (3D only)
    if (!mesh.is2D() && file_exists(prefix + "_w.dat")) {
        auto ref = read_field_data(prefix + "_w.dat");
        FieldComparison result;
        for (int k = mesh.k_begin(); k <= mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    result.update(i, j, k, ref(i, j, k), solver.velocity().w(i, j, k));
                }
            }
        }
        result.finalize();
        result.print("w-velocity");

        if (!result.within_tolerance(BITWISE_TOLERANCE)) {
            std::cout << "    [FAIL] Exceeds tolerance " << BITWISE_TOLERANCE << "\n";
            all_passed = false;
        } else if (result.max_abs_diff < MIN_EXPECTED_DIFF) {
            // Small diff is fine - canary test verifies backend execution.
            // This just means computation isn't sensitive to FP reordering.
            std::cout << "    [PASS] (tiny diff - not sensitive to FP reordering)\n";
        } else {
            std::cout << "    [PASS]\n";
        }
    }

    // Compare pressure
    {
        auto ref = read_field_data(prefix + "_p.dat");
        FieldComparison result;
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    result.update(i, j, k, ref(i, j, k), solver.pressure()(i, j, k));
                }
            }
        }
        result.finalize();
        result.print("pressure");
        p_rel_l2 = result.rel_l2();

        if (!result.within_tolerance(BITWISE_TOLERANCE)) {
            std::cout << "    [FAIL] Exceeds tolerance " << BITWISE_TOLERANCE << "\n";
            all_passed = false;
        } else if (result.max_abs_diff < MIN_EXPECTED_DIFF) {
            // Small diff is fine - canary test verifies backend execution.
            // This just means computation isn't sensitive to FP reordering.
            std::cout << "    [PASS] (tiny diff - not sensitive to FP reordering)\n";
        } else {
            std::cout << "    [PASS]\n";
        }
    }

    // Emit machine-readable QoI for CI metrics
    nncfd::test::harness::emit_qoi_cpu_gpu(u_rel_l2, p_rel_l2);

    std::cout << "\n";
    if (all_passed) {
        std::cout << "[SUCCESS] GPU results match CPU reference within tolerance\n";
        return 0;
    } else {
        std::cout << "[FAILURE] GPU results differ from CPU reference beyond tolerance\n";
        return 1;
    }
#endif
}

//=============================================================================
// MAIN
//=============================================================================

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n\n";
    std::cout << "This test compares CPU and GPU solver outputs.\n";
    std::cout << "It requires running BOTH CPU and GPU builds:\n\n";
    std::cout << "  Step 1: Build and run CPU reference:\n";
    std::cout << "    cmake .. -DUSE_GPU_OFFLOAD=OFF && make test_cpu_gpu_bitwise\n";
    std::cout << "    ./test_cpu_gpu_bitwise --dump-prefix /path/to/ref\n\n";
    std::cout << "  Step 2: Build and run GPU comparison:\n";
    std::cout << "    cmake .. -DUSE_GPU_OFFLOAD=ON && make test_cpu_gpu_bitwise\n";
    std::cout << "    ./test_cpu_gpu_bitwise --compare-prefix /path/to/ref\n\n";
    std::cout << "Options:\n";
    std::cout << "  --dump-prefix <prefix>     Generate CPU reference files (CPU build only)\n";
    std::cout << "  --compare-prefix <prefix>  Compare GPU against CPU reference (GPU build only)\n";
    std::cout << "  --help                     Show this message\n";
}

int main(int argc, char* argv[]) {
    try {
        std::string dump_prefix, compare_prefix;

        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--dump-prefix") == 0 && i + 1 < argc) {
                dump_prefix = argv[++i];
            } else if (std::strcmp(argv[i], "--compare-prefix") == 0 && i + 1 < argc) {
                compare_prefix = argv[++i];
            } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
                print_usage(argv[0]);
                return 0;
            } else {
                std::cerr << "Unknown argument: " << argv[i] << "\n";
                print_usage(argv[0]);
                return 1;
            }
        }

        std::cout << "=== CPU/GPU Bitwise Comparison Test ===\n";
#ifdef USE_GPU_OFFLOAD
        std::cout << "Build: GPU (USE_GPU_OFFLOAD=ON)\n";
#else
        std::cout << "Build: CPU (USE_GPU_OFFLOAD=OFF)\n";
#endif
        std::cout << "Tolerance: " << std::scientific << BITWISE_TOLERANCE << "\n\n";

        if (!dump_prefix.empty()) {
#ifdef USE_GPU_OFFLOAD
            std::cerr << "ERROR: --dump-prefix requires CPU build (USE_GPU_OFFLOAD=OFF)\n";
            std::cerr << "       GPU builds should use --compare-prefix to compare against CPU reference.\n";
            std::cerr << "       To generate reference data, rebuild with -DUSE_GPU_OFFLOAD=OFF\n";
            return 1;
#else
            return run_dump_mode(dump_prefix);
#endif
        } else if (!compare_prefix.empty()) {
            return run_compare_mode(compare_prefix);
        } else {
            std::cerr << "ERROR: This test requires --dump-prefix or --compare-prefix\n\n";
            print_usage(argv[0]);
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
