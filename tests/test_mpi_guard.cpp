/// @file test_mpi_guard.cpp
/// @brief Unit tests for MPI guard policy logic
///
/// Tests the pure function check_mpi_guard_policy() which determines whether
/// to exit, warn, or allow based on:
///   - world_size (number of MPI ranks)
///   - NNCFD_ALLOW_MULTI_RANK env var
///   - GPU vs CPU build
///
/// This test does NOT require MPI runtime - it tests the decision logic directly.

#include "mpi_check.hpp"
#include <iostream>
#include <string>
#include <vector>

using namespace nncfd;

// Test helper
struct TestCase {
    std::string name;
    int world_size;
    const char* allow_override;
    bool is_gpu_build;
    // Expected results
    bool expect_should_exit;
    bool expect_is_multi_rank;
    bool expect_is_override;
};

bool run_test(const TestCase& tc) {
    MpiGuardResult result = check_mpi_guard_policy(tc.world_size, tc.allow_override, tc.is_gpu_build);

    bool pass = true;

    if (result.should_exit != tc.expect_should_exit) {
        std::cerr << "  FAIL: should_exit=" << result.should_exit
                  << " expected=" << tc.expect_should_exit << "\n";
        pass = false;
    }
    if (result.is_multi_rank != tc.expect_is_multi_rank) {
        std::cerr << "  FAIL: is_multi_rank=" << result.is_multi_rank
                  << " expected=" << tc.expect_is_multi_rank << "\n";
        pass = false;
    }
    if (result.is_override != tc.expect_is_override) {
        std::cerr << "  FAIL: is_override=" << result.is_override
                  << " expected=" << tc.expect_is_override << "\n";
        pass = false;
    }

    std::cout << "  " << tc.name << ": " << (pass ? "[PASS]" : "[FAIL]") << "\n";
    return pass;
}

int main() {
    std::cout << "================================================================\n";
    std::cout << "  MPI Guard Policy Unit Tests\n";
    std::cout << "================================================================\n\n";

    std::cout << "Testing check_mpi_guard_policy() decision logic:\n\n";

    std::vector<TestCase> tests = {
        // =====================================================================
        // Single-rank or non-MPI cases (should never exit)
        // =====================================================================
        {"non-MPI (world_size=0), GPU build",
         0, nullptr, true,
         false, false, false},

        {"single-rank (world_size=1), GPU build",
         1, nullptr, true,
         false, false, false},

        {"non-MPI (world_size=0), CPU build",
         0, nullptr, false,
         false, false, false},

        {"single-rank (world_size=1), CPU build",
         1, nullptr, false,
         false, false, false},

        // =====================================================================
        // Multi-rank GPU build (should exit unless override)
        // =====================================================================
        {"multi-rank (world_size=2), GPU build, no override",
         2, nullptr, true,
         true, true, false},  // should_exit=true

        {"multi-rank (world_size=4), GPU build, no override",
         4, nullptr, true,
         true, true, false},  // should_exit=true

        {"multi-rank (world_size=2), GPU build, override=1",
         2, "1", true,
         false, true, true},  // should_exit=false (override)

        {"multi-rank (world_size=2), GPU build, override=true",
         2, "true", true,
         false, true, true},  // should_exit=false (override)

        {"multi-rank (world_size=2), GPU build, override=TRUE",
         2, "TRUE", true,
         false, true, true},  // should_exit=false (override)

        {"multi-rank (world_size=2), GPU build, override=0 (not valid)",
         2, "0", true,
         true, true, false},  // should_exit=true (0 is not a valid override)

        {"multi-rank (world_size=2), GPU build, override=yes (not valid)",
         2, "yes", true,
         true, true, false},  // should_exit=true (yes is not recognized)

        // =====================================================================
        // Multi-rank CPU build (warn only, never exit)
        // =====================================================================
        {"multi-rank (world_size=2), CPU build, no override",
         2, nullptr, false,
         false, true, false},  // should_exit=false (CPU builds just warn)

        {"multi-rank (world_size=4), CPU build, no override",
         4, nullptr, false,
         false, true, false},  // should_exit=false

        {"multi-rank (world_size=2), CPU build, override=1",
         2, "1", false,
         false, true, true},  // should_exit=false (already allowed, but override noted)

        // =====================================================================
        // Edge cases
        // =====================================================================
        {"large world_size=128, GPU build, no override",
         128, nullptr, true,
         true, true, false},  // should_exit=true

        {"large world_size=128, GPU build, override=1",
         128, "1", true,
         false, true, true},  // should_exit=false (override)
    };

    int passed = 0, failed = 0;
    for (const auto& tc : tests) {
        if (run_test(tc)) {
            ++passed;
        } else {
            ++failed;
        }
    }

    std::cout << "\n================================================================\n";
    std::cout << "MPI Guard Policy Summary\n";
    std::cout << "================================================================\n";
    std::cout << "  Passed: " << passed << "/" << (passed + failed) << "\n";
    std::cout << "  Failed: " << failed << "/" << (passed + failed) << "\n";

    if (failed == 0) {
        std::cout << "\n[PASS] All MPI guard policy tests passed\n";
        return 0;
    } else {
        std::cout << "\n[FAIL] " << failed << " MPI guard policy test(s) failed\n";
        return 1;
    }
}
