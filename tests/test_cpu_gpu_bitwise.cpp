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
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <cstring>
#include <vector>
#include <sstream>
#include <functional>
#include <climits>

using namespace nncfd;

// Tolerance for CPU vs GPU comparison
// Should see small FP differences due to different instruction ordering, FMA, etc.
constexpr double TOLERANCE = 1e-10;

//=============================================================================
// File I/O helpers
//=============================================================================

bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

// Write velocity field component to file
void write_field_data(const std::string& filename,
                      const Mesh& mesh,
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

//=============================================================================
// Comparison helpers
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

    void print(const std::string& name) const {
        std::cout << "  " << name << ":\n";
        std::cout << "    Max abs diff: " << std::scientific << max_abs_diff << "\n";
        std::cout << "    Max rel diff: " << max_rel_diff << "\n";
        std::cout << "    RMS diff:     " << rms_diff << "\n";
        if (max_abs_diff > 0) {
            std::cout << "    Worst at (" << worst_i << "," << worst_j << "," << worst_k << "): "
                      << "CPU=" << ref_at_worst << ", GPU=" << gpu_at_worst << "\n";
        }
    }

    bool within_tolerance(double tol) const {
        return max_abs_diff < tol;
    }
};

//=============================================================================
// Test case: Channel flow with body force (same as original test)
//=============================================================================

void setup_channel_test(Mesh& mesh, Config& config, int NX, int NY, int NZ, int num_steps) {
    mesh.init_uniform(NX, NY, NZ, 0.0, 4.0, 0.0, 2.0, 0.0, 1.0);

    config.nu = 0.01;
    config.dt = 0.0005;
    config.adaptive_dt = false;  // Fixed dt for reproducibility
    config.max_iter = num_steps;
    config.tol = 1e-6;
    config.turb_model = TurbulenceModelType::None;
    config.verbose = false;
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
    std::cout << "Output prefix: " << prefix << "\n\n";

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

int run_compare_mode(const std::string& prefix) {
#ifndef USE_GPU_OFFLOAD
    std::cerr << "ERROR: --compare-prefix requires GPU build\n";
    std::cerr << "       This binary was built with USE_GPU_OFFLOAD=OFF\n";
    std::cerr << "       Rebuild with -DUSE_GPU_OFFLOAD=ON\n";
    return 1;
#else
    std::cout << "=== GPU Comparison Mode ===\n";
    std::cout << "Reference prefix: " << prefix << "\n\n";

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

    // Compare u-velocity
    {
        auto ref = read_field_data(prefix + "_u.dat");
        ComparisonResult result;
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i <= mesh.i_end(); ++i) {
                    result.update(i, j, k, ref(i, j, k), solver.velocity().u(i, j, k));
                }
            }
        }
        result.finalize();
        result.print("u-velocity");

        if (!result.within_tolerance(TOLERANCE)) {
            std::cout << "    [FAIL] Exceeds tolerance " << TOLERANCE << "\n";
            all_passed = false;
        } else if (result.max_abs_diff == 0.0) {
            std::cout << "    [WARN] Exact match - possibly comparing same backend?\n";
        } else {
            std::cout << "    [PASS]\n";
        }
    }

    // Compare v-velocity
    {
        auto ref = read_field_data(prefix + "_v.dat");
        ComparisonResult result;
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j <= mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    result.update(i, j, k, ref(i, j, k), solver.velocity().v(i, j, k));
                }
            }
        }
        result.finalize();
        result.print("v-velocity");

        if (!result.within_tolerance(TOLERANCE)) {
            std::cout << "    [FAIL] Exceeds tolerance " << TOLERANCE << "\n";
            all_passed = false;
        } else if (result.max_abs_diff == 0.0) {
            std::cout << "    [WARN] Exact match - possibly comparing same backend?\n";
        } else {
            std::cout << "    [PASS]\n";
        }
    }

    // Compare w-velocity (3D only)
    if (!mesh.is2D() && file_exists(prefix + "_w.dat")) {
        auto ref = read_field_data(prefix + "_w.dat");
        ComparisonResult result;
        for (int k = mesh.k_begin(); k <= mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    result.update(i, j, k, ref(i, j, k), solver.velocity().w(i, j, k));
                }
            }
        }
        result.finalize();
        result.print("w-velocity");

        if (!result.within_tolerance(TOLERANCE)) {
            std::cout << "    [FAIL] Exceeds tolerance " << TOLERANCE << "\n";
            all_passed = false;
        } else if (result.max_abs_diff == 0.0) {
            std::cout << "    [WARN] Exact match - possibly comparing same backend?\n";
        } else {
            std::cout << "    [PASS]\n";
        }
    }

    // Compare pressure
    {
        auto ref = read_field_data(prefix + "_p.dat");
        ComparisonResult result;
        for (int k = mesh.k_begin(); k < mesh.k_end(); ++k) {
            for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
                for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
                    result.update(i, j, k, ref(i, j, k), solver.pressure()(i, j, k));
                }
            }
        }
        result.finalize();
        result.print("pressure");

        if (!result.within_tolerance(TOLERANCE)) {
            std::cout << "    [FAIL] Exceeds tolerance " << TOLERANCE << "\n";
            all_passed = false;
        } else if (result.max_abs_diff == 0.0) {
            std::cout << "    [WARN] Exact match - possibly comparing same backend?\n";
        } else {
            std::cout << "    [PASS]\n";
        }
    }

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
}
