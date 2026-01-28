/// @file test_gpu_mapping_canary.cpp
/// @brief Fast GPU mapping canary test - validates dev_ptr() + is_device_ptr pattern
///
/// This is a "smoke test" that catches GPU pointer mapping failures early:
///   1. Allocates a small array and maps to device
///   2. Gets device pointer via gpu::dev_ptr()
///   3. Launches a trivial kernel with is_device_ptr
///   4. Reads back result and validates
///
/// If this test fails, ALL GPU kernels in the codebase are suspect.
/// If it passes, the fundamental mapping mechanism works correctly.
///
/// Expected runtime: < 1 second (just kernel launch overhead)

#include "test_harness.hpp"
#include "gpu_utils.hpp"
#include <vector>
#include <cmath>

using namespace nncfd;
using namespace nncfd::test::harness;

/// Sentinel value that kernel writes (unlikely to exist by accident)
static constexpr double SENTINEL_VALUE = 314159.265358979;

//=============================================================================
// Test: Basic mapping + write + readback
//=============================================================================
static void test_basic_mapping() {
    const int N = 10;
    std::vector<double> data(N, 0.0);
    double* data_ptr = data.data();

#ifdef USE_GPU_OFFLOAD
    // Map data to device using raw pointer (required for NVHPC)
    #pragma omp target enter data map(to: data_ptr[0:N])

    // Get device pointer using the required pattern
    double* data_dev = gpu::dev_ptr(data_ptr);

    // Verify dev_ptr returned non-null
    record("dev_ptr returns non-null", data_dev != nullptr);

    // Launch kernel that writes sentinel value using is_device_ptr
    #pragma omp target teams distribute parallel for is_device_ptr(data_dev)
    for (int i = 0; i < N; ++i) {
        data_dev[i] = SENTINEL_VALUE;
    }

    // Read back to host
    #pragma omp target update from(data_ptr[0:N])

    // Unmap
    #pragma omp target exit data map(delete: data_ptr[0:N])

    // Validate all values are sentinel
    bool all_correct = true;
    for (int i = 0; i < N; ++i) {
        if (std::abs(data[i] - SENTINEL_VALUE) > 1e-10) {
            all_correct = false;
            std::cerr << "  data[" << i << "] = " << data[i]
                      << " (expected " << SENTINEL_VALUE << ")\n";
        }
    }
    record("Kernel wrote correct values via is_device_ptr", all_correct);
#else
    // CPU build - trivial pass
    (void)data_ptr;
    record("dev_ptr returns non-null", true);
    record("Kernel wrote correct values via is_device_ptr", true, true); // skip
#endif
}

//=============================================================================
// Test: Reduction via is_device_ptr
//=============================================================================
static void test_reduction_mapping() {
    const int N = 1000;
    std::vector<double> data(N);
    double* data_ptr = data.data();

    // Initialize with known values
    for (int i = 0; i < N; ++i) {
        data[i] = 1.0;  // Sum should be N
    }

#ifdef USE_GPU_OFFLOAD
    // Map data to device
    #pragma omp target enter data map(to: data_ptr[0:N])

    // Get device pointer
    double* data_dev = gpu::dev_ptr(data_ptr);

    // Compute sum on device using reduction
    double sum = 0.0;
    #pragma omp target teams distribute parallel for is_device_ptr(data_dev) \
        reduction(+:sum)
    for (int i = 0; i < N; ++i) {
        sum += data_dev[i];
    }

    // Unmap
    #pragma omp target exit data map(delete: data_ptr[0:N])

    // Validate sum
    bool sum_correct = std::abs(sum - static_cast<double>(N)) < 1e-10;
    record("Reduction via is_device_ptr correct", sum_correct);

    if (!sum_correct) {
        std::cerr << "  sum = " << sum << " (expected " << N << ")\n";
    }
#else
    (void)data_ptr;
    record("Reduction via is_device_ptr correct", true, true); // skip
#endif
}

//=============================================================================
// Test: Multiple pointers with is_device_ptr
//=============================================================================
static void test_multiple_pointers() {
    const int N = 100;
    std::vector<double> a(N, 1.0);
    std::vector<double> b(N, 2.0);
    std::vector<double> c(N, 0.0);
    double* a_ptr = a.data();
    double* b_ptr = b.data();
    double* c_ptr = c.data();

#ifdef USE_GPU_OFFLOAD
    // Map all arrays
    #pragma omp target enter data map(to: a_ptr[0:N], b_ptr[0:N]) map(alloc: c_ptr[0:N])

    // Get device pointers for all three
    double* a_dev = gpu::dev_ptr(a_ptr);
    double* b_dev = gpu::dev_ptr(b_ptr);
    double* c_dev = gpu::dev_ptr(c_ptr);

    record("Multiple dev_ptr calls succeed",
           a_dev != nullptr && b_dev != nullptr && c_dev != nullptr);

    // Kernel: c = a + b
    #pragma omp target teams distribute parallel for \
        is_device_ptr(a_dev, b_dev, c_dev)
    for (int i = 0; i < N; ++i) {
        c_dev[i] = a_dev[i] + b_dev[i];
    }

    // Read back c
    #pragma omp target update from(c_ptr[0:N])

    // Unmap
    #pragma omp target exit data map(delete: a_ptr[0:N], b_ptr[0:N], c_ptr[0:N])

    // Validate c[i] = 3.0 for all i
    bool all_correct = true;
    for (int i = 0; i < N; ++i) {
        if (std::abs(c[i] - 3.0) > 1e-10) {
            all_correct = false;
            std::cerr << "  c[" << i << "] = " << c[i] << " (expected 3.0)\n";
            break;
        }
    }
    record("Multi-pointer kernel correct", all_correct);
#else
    (void)a_ptr; (void)b_ptr; (void)c_ptr;
    record("Multiple dev_ptr calls succeed", true, true);
    record("Multi-pointer kernel correct", true, true);
#endif
}

//=============================================================================
// Test: Verify get_device_ptr returns nullptr for unmapped pointer
//=============================================================================
static void test_unmapped_detection() {
#ifdef USE_GPU_OFFLOAD
    // This tests that get_device_ptr correctly returns nullptr for unmapped pointers
    // We can't call dev_ptr on unmapped data (it aborts), so we use get_device_ptr

    std::vector<double> unmapped_data(10, 0.0);
    double* unmapped_ptr = unmapped_data.data();
    // DO NOT map this data

    // get_device_ptr (not dev_ptr) returns nullptr for unmapped pointers
    double* result = gpu::get_device_ptr(unmapped_ptr);
    record("get_device_ptr returns nullptr for unmapped", result == nullptr);
#else
    record("get_device_ptr returns nullptr for unmapped", true, true);
#endif
}

//=============================================================================
// Test: Verify mapping survives multiple kernel launches
//=============================================================================
static void test_persistent_mapping() {
    const int N = 50;
    std::vector<double> data(N, 0.0);
    double* data_ptr = data.data();

#ifdef USE_GPU_OFFLOAD
    // Map once
    #pragma omp target enter data map(to: data_ptr[0:N])

    double* data_dev = gpu::dev_ptr(data_ptr);

    // Launch multiple kernels on same mapping
    for (int k = 0; k < 5; ++k) {
        #pragma omp target teams distribute parallel for is_device_ptr(data_dev)
        for (int i = 0; i < N; ++i) {
            data_dev[i] += 1.0;
        }
    }

    // Read back
    #pragma omp target update from(data_ptr[0:N])

    // Unmap
    #pragma omp target exit data map(delete: data_ptr[0:N])

    // Each element should be 5.0 (5 iterations of +=1)
    bool all_correct = true;
    for (int i = 0; i < N; ++i) {
        if (std::abs(data[i] - 5.0) > 1e-10) {
            all_correct = false;
            std::cerr << "  data[" << i << "] = " << data[i] << " (expected 5.0)\n";
            break;
        }
    }
    record("Persistent mapping survives multiple kernels", all_correct);
#else
    (void)data_ptr;
    record("Persistent mapping survives multiple kernels", true, true);
#endif
}

//=============================================================================
// Test: Verify kernel actually executes on device (not host fallback)
//=============================================================================
static void test_device_execution() {
#ifdef USE_GPU_OFFLOAD
    // This test catches the case where OpenMP silently falls back to host execution
    // omp_is_initial_device() returns 1 on host, 0 on device
    int executed_on_device = 0;

    #pragma omp target map(from: executed_on_device)
    {
        // Inside target region: omp_is_initial_device() should be 0 (false)
        executed_on_device = (omp_is_initial_device() == 0) ? 1 : 0;
    }

    record("Kernel executed on device (not host fallback)", executed_on_device == 1);

    if (executed_on_device != 1) {
        std::cerr << "  CRITICAL: Target region ran on HOST, not device!\n";
        std::cerr << "  This means GPU offloading is silently disabled.\n";
        std::cerr << "  Check OMP_TARGET_OFFLOAD and GPU availability.\n";
    }
#else
    record("Kernel executed on device (not host fallback)", true, true);
#endif
}

//=============================================================================
// Main
//=============================================================================

int main() {
    std::cout << "================================================================\n";
    std::cout << "  GPU Mapping Canary Test\n";
    std::cout << "================================================================\n\n";

    std::cout << "This test validates the gpu::dev_ptr() + is_device_ptr pattern.\n";
    std::cout << "If ANY test fails, ALL GPU kernels in the codebase are suspect.\n\n";

    // Use run_sections which handles print_config() and canary_check() automatically
    return run_sections("GPU Mapping Canary", {
        {"Device Execution", test_device_execution},
        {"Basic Mapping", test_basic_mapping},
        {"Reduction Mapping", test_reduction_mapping},
        {"Multiple Pointers", test_multiple_pointers},
        {"Unmapped Detection", test_unmapped_detection},
        {"Persistent Mapping", test_persistent_mapping}
    });
}
