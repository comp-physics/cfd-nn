/// 3D Poisson Solver CPU vs GPU Comparison Test
/// Compares CPU-built and GPU-built Poisson solver outputs.
///
/// This test REQUIRES two separate builds:
///   1. CPU build (USE_GPU_OFFLOAD=OFF): Run with --dump-prefix to generate reference
///   2. GPU build (USE_GPU_OFFLOAD=ON):  Run with --compare-prefix to compare against reference
///
/// Expected result: Small differences (1e-12 to 1e-10) due to FP operation ordering,
/// but not exact zeros (which would indicate both runs used the same backend).

#include "mesh.hpp"
#include "fields.hpp"
#include "poisson_solver_multigrid.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <cstring>
#include <vector>
#include <climits>

#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif

using namespace nncfd;

// Tolerance for CPU vs GPU comparison
constexpr double TOLERANCE = 1e-10;

// Minimum expected difference - if below this, CPU and GPU may be running same code path
// Machine epsilon for double is ~2.2e-16, so any real FP difference should exceed this
constexpr double MIN_EXPECTED_DIFF = 1e-14;

//=============================================================================
// File I/O helpers
//=============================================================================

bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

// Write scalar field to file
void write_scalar_field(const std::string& filename, const ScalarField& field, const Mesh& mesh) {
    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    file << std::setprecision(17) << std::scientific;
    file << "# i j k value\n";

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                file << i << " " << j << " " << k << " " << field(i, j, k) << "\n";
            }
        }
    }
}

// Read scalar field data from file
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

//=============================================================================
// Comparison helper
//=============================================================================

struct ComparisonResult {
    double max_abs_diff = 0.0;
    double max_rel_diff = 0.0;
    double rms_diff = 0.0;
    int worst_i = 0, worst_j = 0, worst_k = 0;
    double ref_at_worst = 0.0;
    double gpu_at_worst = 0.0;
    int count = 0;

    void update(int i, int j, int k, double ref_val, double gpu_val) {
        double abs_diff = std::abs(ref_val - gpu_val);
        double rel_diff = abs_diff / (std::abs(ref_val) + 1e-15);

        rms_diff += abs_diff * abs_diff;
        count++;

        if (abs_diff > max_abs_diff) {
            max_abs_diff = abs_diff;
            max_rel_diff = rel_diff;
            worst_i = i; worst_j = j; worst_k = k;
            ref_at_worst = ref_val;
            gpu_at_worst = gpu_val;
        }
    }

    void finalize() {
        if (count > 0) {
            rms_diff = std::sqrt(rms_diff / count);
        }
    }

    void print() const {
        std::cout << std::scientific << std::setprecision(6);
        std::cout << "  Max absolute difference: " << max_abs_diff << "\n";
        std::cout << "  Max relative difference: " << max_rel_diff << "\n";
        std::cout << "  RMS difference:          " << rms_diff << "\n";
        if (max_abs_diff > 0) {
            std::cout << "  Worst at (" << worst_i << "," << worst_j << "," << worst_k << "): "
                      << "CPU=" << ref_at_worst << ", GPU=" << gpu_at_worst << "\n";
        }
    }

    bool within_tolerance(double tol) const {
        return max_abs_diff < tol;
    }
};

//=============================================================================
// Test parameters
//=============================================================================

const int NX = 32;
const int NY = 32;
const int NZ = 4;
const double LX = 1.0;
const double LY = 1.0;
const double LZ = 1.0;

void setup_rhs(ScalarField& rhs, const Mesh& mesh) {
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                double x = mesh.x(i);
                double y = mesh.y(j);
                // Simple forcing term (compatible with periodic BCs)
                rhs(i, j, k) = std::sin(2.0 * M_PI * x) * std::sin(2.0 * M_PI * y);
            }
        }
    }
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
    std::cout << "Output prefix: " << prefix << "\n\n";

    // Create mesh
    Mesh mesh;
    mesh.init_uniform(NX, NY, NZ, 0.0, LX, 0.0, LY, 0.0, LZ);

    // Create RHS
    ScalarField rhs(mesh, 0.0);
    setup_rhs(rhs, mesh);

    // Create solver and solution field
    ScalarField pressure(mesh, 0.0);
    MultigridPoissonSolver solver(mesh);
    solver.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                  PoissonBC::Periodic, PoissonBC::Periodic,
                  PoissonBC::Periodic, PoissonBC::Periodic);

    PoissonConfig cfg;
    cfg.tol = 1e-8;
    cfg.max_iter = 100;

    std::cout << "Solving Poisson equation on CPU...\n";
    int iterations = solver.solve(rhs, pressure, cfg);
    double residual = solver.residual();

    std::cout << "  Iterations: " << iterations << "\n";
    std::cout << "  Residual:   " << std::scientific << residual << "\n";

    // Write solution
    std::cout << "Writing reference solution...\n";
    write_scalar_field(prefix + "_pressure.dat", pressure, mesh);
    std::cout << "  Wrote: " << prefix << "_pressure.dat\n";

    // Write metadata
    std::ofstream meta(prefix + "_meta.dat");
    meta << "iterations " << iterations << "\n";
    meta << "residual " << std::setprecision(17) << residual << "\n";
    meta << "NX " << NX << "\n";
    meta << "NY " << NY << "\n";
    meta << "NZ " << NZ << "\n";
    meta.close();
    std::cout << "  Wrote: " << prefix << "_meta.dat\n";

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
    std::cout << "Reference prefix: " << prefix << "\n\n";

    // Verify GPU is actually accessible (not just compiled with offload)
    const int num_devices = omp_get_num_devices();
    std::cout << "GPU devices available: " << num_devices << "\n";
    if (num_devices == 0) {
        std::cerr << "ERROR: No GPU devices found. Cannot run GPU comparison.\n";
        return 1;
    }

    // Verify target regions actually execute on GPU (not host fallback)
    int on_device = 0;
    #pragma omp target map(tofrom: on_device)
    {
        on_device = !omp_is_initial_device();
    }
    if (!on_device) {
        std::cerr << "ERROR: Target region executed on host, not GPU.\n";
        std::cerr << "       Check GPU drivers and OMP_TARGET_OFFLOAD settings.\n";
        return 1;
    }
    std::cout << "GPU execution verified: YES\n\n";

    // Verify reference files exist
    if (!file_exists(prefix + "_pressure.dat")) {
        std::cerr << "ERROR: Reference file not found: " << prefix << "_pressure.dat\n";
        std::cerr << "       Run CPU build with --dump-prefix first\n";
        return 1;
    }

    // Create mesh
    Mesh mesh;
    mesh.init_uniform(NX, NY, NZ, 0.0, LX, 0.0, LY, 0.0, LZ);

    // Create RHS (same as CPU)
    ScalarField rhs(mesh, 0.0);
    setup_rhs(rhs, mesh);

    // Create solver and solution field
    ScalarField pressure(mesh, 0.0);
    MultigridPoissonSolver solver(mesh);
    solver.set_bc(PoissonBC::Periodic, PoissonBC::Periodic,
                  PoissonBC::Periodic, PoissonBC::Periodic,
                  PoissonBC::Periodic, PoissonBC::Periodic);

    PoissonConfig cfg;
    cfg.tol = 1e-8;
    cfg.max_iter = 100;

    // GPU solver initialized in constructor, sync_to_gpu called in solve()
    std::cout << "Solving Poisson equation on GPU...\n";
    int iterations = solver.solve(rhs, pressure, cfg);
    double residual = solver.residual();

    std::cout << "  Iterations: " << iterations << "\n";
    std::cout << "  Residual:   " << std::scientific << residual << "\n";

    // Load CPU reference and compare
    std::cout << "\nLoading CPU reference and comparing...\n\n";

    auto ref = read_field_data(prefix + "_pressure.dat");
    ComparisonResult result;

    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
            for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                result.update(i, j, k, ref(i, j, k), pressure(i, j, k));
            }
        }
    }
    result.finalize();
    result.print();

    // Show sample points across z-planes
    std::cout << "\nSample points across z-planes (center):\n";
    int mid_i = mesh.i_begin() + NX/2;
    int mid_j = mesh.j_begin() + NY/2;
    for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
        double val_cpu = ref(mid_i, mid_j, k);
        double val_gpu = pressure(mid_i, mid_j, k);
        std::cout << "  z-plane " << k << ": CPU=" << std::scientific << val_cpu
                  << ", GPU=" << val_gpu
                  << ", diff=" << (val_cpu - val_gpu) << "\n";
    }

    std::cout << "\n";
    if (!result.within_tolerance(TOLERANCE)) {
        std::cout << "[FAILURE] GPU results differ from CPU reference beyond tolerance " << TOLERANCE << "\n";
        return 1;
    } else if (result.max_abs_diff < MIN_EXPECTED_DIFF) {
        std::cout << "[WARN] Suspiciously small diff (" << result.max_abs_diff
                  << " < " << MIN_EXPECTED_DIFF << ") - possibly same backend?\n";
        std::cout << "[SUCCESS] Results match within tolerance\n";
        return 0;
    } else {
        std::cout << "[SUCCESS] GPU results match CPU reference within tolerance\n";
        return 0;
    }
#endif
}

//=============================================================================
// MAIN
//=============================================================================

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n\n";
    std::cout << "This test compares CPU and GPU Poisson solver outputs.\n";
    std::cout << "It requires running BOTH CPU and GPU builds:\n\n";
    std::cout << "  Step 1: Build and run CPU reference:\n";
    std::cout << "    cmake .. -DUSE_GPU_OFFLOAD=OFF && make test_poisson_cpu_gpu_3d\n";
    std::cout << "    ./test_poisson_cpu_gpu_3d --dump-prefix /path/to/ref\n\n";
    std::cout << "  Step 2: Build and run GPU comparison:\n";
    std::cout << "    cmake .. -DUSE_GPU_OFFLOAD=ON && make test_poisson_cpu_gpu_3d\n";
    std::cout << "    ./test_poisson_cpu_gpu_3d --compare-prefix /path/to/ref\n\n";
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

        std::cout << "=== 3D Poisson Solver CPU vs GPU Comparison ===\n";
#ifdef USE_GPU_OFFLOAD
        std::cout << "Build: GPU (USE_GPU_OFFLOAD=ON)\n";
#else
        std::cout << "Build: CPU (USE_GPU_OFFLOAD=OFF)\n";
#endif
        std::cout << "Tolerance: " << std::scientific << TOLERANCE << "\n\n";

        if (!dump_prefix.empty()) {
            return run_dump_mode(dump_prefix);
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
