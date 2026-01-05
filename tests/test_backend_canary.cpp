/// Backend Canary Test
/// ====================
/// This test MUST produce different floating-point results on CPU vs GPU.
/// If results are bitwise identical, it indicates the same backend executed both runs.
///
/// The test uses a non-associative reduction (floating-point sum) over many values.
/// Due to different reduction tree orderings, CPU (sequential) and GPU (parallel) will
/// produce slightly different results (~1e-10 to 1e-8 relative difference).
///
/// SUCCESS criteria:
///   - Results within tolerance (1e-6) - algorithms are equivalent
///   - Results differ by more than MIN_EXPECTED_DIFF (1e-14) - different backends
///
/// FAILURE if:
///   - Results exceed tolerance - algorithmic bug
///   - Results too similar (< 1e-14) - same backend executed both (false coverage)

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <vector>
#include <fstream>
#include <cstring>
#include <cstdlib>

#if defined(_OPENMP)
#include <omp.h>
#endif

// Number of elements for reduction - must be large enough to see FP ordering effects
constexpr int N = 1000000;

// Tolerance for "same algorithm" - results should be within this
constexpr double TOLERANCE = 1e-6;

// Minimum expected difference between CPU and GPU due to FP non-associativity
// If diff is smaller than this, backends are probably the same
constexpr double MIN_EXPECTED_DIFF = 1e-14;

// Generate deterministic pseudo-random values (same on both CPU and GPU)
// Uses simple LCG to avoid library differences
double generate_value(int idx) {
    // LCG parameters (same as glibc)
    constexpr uint64_t a = 1103515245;
    constexpr uint64_t c = 12345;
    constexpr uint64_t m = 1ULL << 31;

    uint64_t seed = static_cast<uint64_t>(idx) * a + c;
    seed = (seed * a + c) % m;

    // Map to [-1, 1] range with varying magnitudes to amplify FP effects
    double val = (static_cast<double>(seed) / m) * 2.0 - 1.0;

    // Add some variation in magnitude to make reduction order matter more
    int exp_mod = (idx % 10) - 5;
    return val * std::pow(10.0, exp_mod);
}

// CPU sequential sum (deterministic ordering)
double cpu_sequential_sum() {
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        sum += generate_value(i);
    }
    return sum;
}

#ifdef USE_GPU_OFFLOAD
// GPU parallel reduction (different ordering due to parallel tree reduction)
double gpu_parallel_sum() {
    double sum = 0.0;

    // OpenMP target teams reduction - uses parallel tree reduction on GPU
    #pragma omp target teams distribute parallel for reduction(+:sum)
    for (int i = 0; i < N; ++i) {
        sum += generate_value(i);
    }

    return sum;
}
#endif

void print_backend_info() {
#ifdef USE_GPU_OFFLOAD
    std::cout << "EXEC_BACKEND=GPU_OFFLOAD\n";
    #if defined(_OPENMP)
    std::cout << "  OMP devices: " << omp_get_num_devices() << "\n";
    #endif
#else
    std::cout << "EXEC_BACKEND=CPU_ONLY\n";
#endif
}

bool verify_gpu_available() {
#ifndef USE_GPU_OFFLOAD
    return false;
#else
    if (omp_get_num_devices() == 0) {
        std::cerr << "ERROR: No GPU devices available\n";
        return false;
    }

    int on_device = 0;
    #pragma omp target map(tofrom: on_device)
    {
        on_device = !omp_is_initial_device();
    }

    if (!on_device) {
        std::cerr << "ERROR: Target region executed on host, not GPU\n";
        return false;
    }

    return true;
#endif
}

//=============================================================================
// Dump mode: Generate CPU reference sum
//=============================================================================

int run_dump_mode(const std::string& filename) {
#ifdef USE_GPU_OFFLOAD
    (void)filename;  // Suppress unused parameter warning
    std::cerr << "ERROR: --dump requires CPU build\n";
    return 1;
#else
    std::cout << "=== CPU Reference Generation ===\n";
    print_backend_info();

    double cpu_sum = cpu_sequential_sum();
    std::cout << "CPU sequential sum: " << std::setprecision(17) << cpu_sum << "\n";

    // Write to file
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "ERROR: Cannot write to " << filename << "\n";
        return 1;
    }
    out << std::setprecision(17) << cpu_sum << "\n";
    std::cout << "Reference written to: " << filename << "\n";

    return 0;
#endif
}

//=============================================================================
// Compare mode: Run GPU and compare against CPU reference
//=============================================================================

int run_compare_mode(const std::string& filename) {
#ifndef USE_GPU_OFFLOAD
    (void)filename;  // Suppress unused parameter warning
    std::cerr << "ERROR: --compare requires GPU build\n";
    return 1;
#else
    std::cout << "=== GPU Comparison Mode (Canary Test) ===\n";
    print_backend_info();

    if (!verify_gpu_available()) {
        return 1;
    }

    // Read CPU reference
    std::ifstream in(filename);
    if (!in) {
        std::cerr << "ERROR: Cannot read reference file: " << filename << "\n";
        std::cerr << "       Run CPU build with --dump first\n";
        return 1;
    }

    double cpu_sum;
    in >> cpu_sum;
    std::cout << "CPU reference sum:  " << std::setprecision(17) << cpu_sum << "\n";

    // Run GPU reduction
    double gpu_sum = gpu_parallel_sum();
    std::cout << "GPU parallel sum:   " << std::setprecision(17) << gpu_sum << "\n";

    // Compute difference
    double abs_diff = std::abs(cpu_sum - gpu_sum);
    double rel_diff = abs_diff / (std::abs(cpu_sum) + 1e-15);

    std::cout << "\nComparison:\n";
    std::cout << "  Absolute diff: " << std::scientific << abs_diff << "\n";
    std::cout << "  Relative diff: " << rel_diff << "\n";

    // Check results
    bool passed = true;

    // Check 1: Results should be within tolerance (same algorithm)
    if (rel_diff > TOLERANCE) {
        std::cerr << "\n[FAIL] Results differ too much (rel_diff=" << rel_diff
                  << " > tolerance=" << TOLERANCE << ")\n";
        std::cerr << "       This indicates an algorithmic bug, not just FP ordering.\n";
        passed = false;
    }

    // Check 2: Results should NOT be identical (different backends)
    if (abs_diff < MIN_EXPECTED_DIFF) {
        std::cerr << "\n[FAIL] Results suspiciously identical (diff=" << abs_diff
                  << " < " << MIN_EXPECTED_DIFF << ")\n";
        std::cerr << "       This indicates CPU and GPU ran the SAME code path!\n";
        std::cerr << "       The parity test may be giving false coverage.\n";
        std::cerr << "\n       Possible causes:\n";
        std::cerr << "       1. CPU reference was generated by GPU build\n";
        std::cerr << "       2. GPU is falling back to host execution\n";
        std::cerr << "       3. Build system misconfiguration\n";
        passed = false;
    }

    if (passed) {
        std::cout << "\n[PASS] Canary test confirms different backends executed\n";
        std::cout << "       CPU and GPU results differ by " << abs_diff << "\n";
        std::cout << "       This is expected FP non-associativity from parallel reduction.\n";
        return 0;
    } else {
        return 1;
    }
#endif
}

//=============================================================================
// Standalone mode: Run both CPU and GPU in same binary (GPU build only)
//=============================================================================

int run_standalone_mode() {
#ifndef USE_GPU_OFFLOAD
    std::cout << "=== Standalone Mode (CPU only) ===\n";
    print_backend_info();
    std::cout << "\nThis test requires GPU build for meaningful comparison.\n";
    std::cout << "In CPU-only mode, we just verify the sequential sum works.\n\n";

    double cpu_sum = cpu_sequential_sum();
    std::cout << "CPU sequential sum: " << std::setprecision(17) << cpu_sum << "\n";
    std::cout << "\n[PASS] CPU-only mode completed (no GPU comparison possible)\n";
    return 0;
#else
    std::cout << "=== Standalone Canary Test ===\n";
    print_backend_info();

    if (!verify_gpu_available()) {
        return 1;
    }
    std::cout << "\n";

    // Run CPU sequential sum (even in GPU build, this is sequential on host)
    double cpu_sum = cpu_sequential_sum();
    std::cout << "CPU sequential sum: " << std::setprecision(17) << cpu_sum << "\n";

    // Run GPU parallel sum
    double gpu_sum = gpu_parallel_sum();
    std::cout << "GPU parallel sum:   " << std::setprecision(17) << gpu_sum << "\n";

    // Compute difference
    double abs_diff = std::abs(cpu_sum - gpu_sum);
    double rel_diff = abs_diff / (std::abs(cpu_sum) + 1e-15);

    std::cout << "\nComparison:\n";
    std::cout << "  Absolute diff: " << std::scientific << abs_diff << "\n";
    std::cout << "  Relative diff: " << rel_diff << "\n";

    // In standalone mode, we EXPECT a difference because:
    // - cpu_sequential_sum runs on host (sequential)
    // - gpu_parallel_sum runs on device (parallel reduction)

    if (rel_diff > TOLERANCE) {
        std::cerr << "\n[FAIL] Results differ too much - algorithmic bug\n";
        return 1;
    }

    if (abs_diff < MIN_EXPECTED_DIFF) {
        // In GPU build standalone mode, this should NEVER happen
        // because we're explicitly comparing host sequential vs device parallel
        std::cerr << "\n[FAIL] Results identical - GPU reduction may not be running on device\n";
        return 1;
    }

    std::cout << "\n[PASS] Standalone canary confirms GPU is executing parallel reduction\n";
    std::cout << "       Different FP ordering produced expected difference: " << abs_diff << "\n";
    return 0;
#endif
}

//=============================================================================
// Main
//=============================================================================

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n\n";
    std::cout << "Backend Canary Test - verifies CPU and GPU produce different FP results\n\n";
    std::cout << "Options:\n";
    std::cout << "  --dump <file>      Generate CPU reference (CPU build only)\n";
    std::cout << "  --compare <file>   Compare GPU against CPU reference (GPU build only)\n";
    std::cout << "  (no args)          Standalone mode - run both in same binary\n";
    std::cout << "  --help             Show this message\n";
}

int main(int argc, char* argv[]) {
    try {
        std::string dump_file, compare_file;

        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
                dump_file = argv[++i];
            } else if (std::strcmp(argv[i], "--compare") == 0 && i + 1 < argc) {
                compare_file = argv[++i];
            } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
                print_usage(argv[0]);
                return 0;
            } else {
                std::cerr << "Unknown argument: " << argv[i] << "\n";
                print_usage(argv[0]);
                return 1;
            }
        }

        if (!dump_file.empty()) {
            return run_dump_mode(dump_file);
        } else if (!compare_file.empty()) {
            return run_compare_mode(compare_file);
        } else {
            // Standalone mode - most useful for quick verification
            return run_standalone_mode();
        }

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
