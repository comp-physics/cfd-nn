/// @file test_harness.hpp
/// @brief Unified test harness for NNCFD test suite
///
/// Provides:
/// - Global test result tracking (passed/failed/skipped)
/// - Standard record() function for test results
/// - Suite runner with summary printing
/// - Automatic GPU configuration printing
///
/// Usage:
///   #include "test_harness.hpp"
///   using namespace nncfd::test;
///
///   void my_tests() {
///       harness::record("Test 1", true);
///       harness::record("Test 2", false);
///       harness::record("Test 3", true, true);  // skipped
///   }
///
///   int main() {
///       return harness::run("My Suite", my_tests);
///   }

#pragma once

#include "test_utilities.hpp"
#include <iostream>
#include <iomanip>
#include <string>
#include <functional>

namespace nncfd {
namespace test {
namespace harness {

//=============================================================================
// Global Test State
//=============================================================================

/// Thread-local test counters (safe for parallel test execution)
struct TestCounters {
    int passed = 0;
    int failed = 0;
    int skipped = 0;

    void reset() { passed = failed = skipped = 0; }

    int total() const { return passed + failed + skipped; }

    bool all_passed() const { return failed == 0; }
};

/// Global test counters (initialized on first use)
inline TestCounters& counters() {
    static TestCounters c;
    return c;
}

//=============================================================================
// Test Recording
//=============================================================================

/// Record a test result with optional skip flag
/// @param name  Test name (displayed left-aligned)
/// @param pass  Whether test passed (ignored if skip=true)
/// @param skip  Whether test was skipped
/// @param width Column width for name alignment (default 50)
inline void record(const char* name, bool pass, bool skip = false, int width = 50) {
    std::cout << "  " << std::left << std::setw(width) << name;
    if (skip) {
        std::cout << "[SKIP]\n";
        ++counters().skipped;
    } else if (pass) {
        std::cout << "[PASS]\n";
        ++counters().passed;
    } else {
        std::cout << "[FAIL]\n";
        ++counters().failed;
    }
}

/// Record a test result with a message
inline void record(const char* name, bool pass, const std::string& msg, bool skip = false, int width = 50) {
    std::cout << "  " << std::left << std::setw(width) << name;
    if (skip) {
        std::cout << "[SKIP]";
        ++counters().skipped;
    } else if (pass) {
        std::cout << "[PASS]";
        ++counters().passed;
    } else {
        std::cout << "[FAIL]";
        ++counters().failed;
    }
    if (!msg.empty()) {
        std::cout << " " << msg;
    }
    std::cout << "\n";
}

/// String overload for convenience
inline void record(const std::string& name, bool pass, bool skip = false) {
    record(name.c_str(), pass, skip);
}

/// String name + debug message overload
inline void record(const std::string& name, bool pass, const std::string& msg) {
    record(name.c_str(), pass, msg, false);
}

//=============================================================================
// Suite Runner
//=============================================================================

/// Print a suite header with title
inline void print_header(const char* suite_name) {
    std::cout << "================================================================\n";
    std::cout << "  " << suite_name << "\n";
    std::cout << "================================================================\n\n";
}

/// Print GPU configuration info
inline void print_gpu_config() {
    gpu::print_config();
    std::cout << "\n";
}

/// Print test summary and return exit code
inline int print_summary() {
    const auto& c = counters();
    std::cout << "\n================================================================\n";
    std::cout << "Summary: " << c.passed << " passed, " << c.failed << " failed";
    if (c.skipped > 0) {
        std::cout << ", " << c.skipped << " skipped";
    }
    std::cout << "\n================================================================\n";

    return c.failed > 0 ? 1 : 0;
}

/// Run a test suite with automatic header, GPU config, and summary
/// @param suite_name  Name of the test suite
/// @param test_fn     Function containing all tests (calls record() internally)
/// @return Exit code (0 = all passed, 1 = failures)
inline int run(const char* suite_name, std::function<void()> test_fn) {
    counters().reset();
    print_header(suite_name);
    print_gpu_config();
    test_fn();
    return print_summary();
}

/// Run multiple test sections with headers
inline int run_sections(const char* suite_name,
                        std::initializer_list<std::pair<const char*, std::function<void()>>> sections) {
    counters().reset();
    print_header(suite_name);
    print_gpu_config();

    for (const auto& [section_name, section_fn] : sections) {
        std::cout << "\n--- " << section_name << " ---\n\n";
        section_fn();
    }

    return print_summary();
}

//=============================================================================
// Assertion Helpers
//=============================================================================

/// Assert that a condition is true, recording result
inline void assert_true(const char* name, bool condition) {
    record(name, condition);
}

/// Assert that two values are close (using check_close from test_utilities.hpp)
inline void assert_close(const char* name, double a, double b,
                         double rtol = 1e-5, double atol = 1e-8) {
    bool pass = check_close(a, b, rtol, atol);
    record(name, pass);
}

/// Assert that a value is near zero
inline void assert_near_zero(const char* name, double val, double atol = 1e-10) {
    bool pass = check_near_zero(val, atol);
    record(name, pass);
}

/// Assert that all values in a range are finite
template<typename Iterator>
inline void assert_all_finite(const char* name, Iterator begin, Iterator end) {
    bool all_finite = true;
    for (auto it = begin; it != end && all_finite; ++it) {
        if (!std::isfinite(*it)) all_finite = false;
    }
    record(name, all_finite);
}

} // namespace harness
} // namespace test
} // namespace nncfd
