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
#include <sstream>
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

/// Run a test suite with automatic header, GPU config, canary check, and summary
/// @param suite_name  Name of the test suite
/// @param test_fn     Function containing all tests (calls record() internally)
/// @return Exit code (0 = all passed, 1 = failures)
inline int run(const char* suite_name, std::function<void()> test_fn) {
    counters().reset();
    print_header(suite_name);
    print_gpu_config();

    // GPU canary: fail fast if MANDATORY but no devices
    if (!gpu::canary_check()) {
        std::cerr << "GPU canary check failed - aborting test suite\n";
        return 1;
    }

    test_fn();
    return print_summary();
}

/// Run multiple test sections with headers
inline int run_sections(const char* suite_name,
                        std::initializer_list<std::pair<const char*, std::function<void()>>> sections) {
    counters().reset();
    print_header(suite_name);
    print_gpu_config();

    // GPU canary: fail fast if MANDATORY but no devices
    if (!gpu::canary_check()) {
        std::cerr << "GPU canary check failed - aborting test suite\n";
        return 1;
    }

    for (const auto& [section_name, section_fn] : sections) {
        std::cout << "\n--- " << section_name << " ---\n\n";
        section_fn();
    }

    return print_summary();
}

//=============================================================================
// Debug Dump Infrastructure
//=============================================================================

/// Rolling window entry for time history
struct StepSnapshot {
    int step = 0;
    double E = 0.0;
    double max_div = 0.0;
    double max_u = 0.0;
    int poisson_iters = 0;
};

/// Simulation diagnostic state for failure debugging
struct SimDiagnostics {
    // Grid configuration
    int Nx = 0, Ny = 0, Nz = 0;
    double dt = 0.0;
    std::string bcs = "";
    std::string poisson_solver = "";
    int poisson_iters = 0;

    // Flow state
    double max_u = 0.0, max_v = 0.0, max_w = 0.0;
    double E_initial = 0.0, E_final = 0.0;
    double max_div = 0.0;
    int max_div_i = 0, max_div_j = 0, max_div_k = 0;

    // RANS-specific (optional)
    double min_k = 1e30, min_omega = 1e30;
    double min_nut = 1e30, max_nut = 0.0;
    double max_nut_over_nu = 0.0;
    double U_bulk = 0.0, tau_w = 0.0;

    // Rolling window: last 5 steps for debugging "when did it go wrong"
    static constexpr int WINDOW_SIZE = 5;
    StepSnapshot history[WINDOW_SIZE];
    int history_idx = 0;
    int history_count = 0;

    /// Record a step snapshot (call each step during simulation)
    void record_step(int step, double E, double div, double u_max, int p_iters = 0) {
        history[history_idx] = {step, E, div, u_max, p_iters};
        history_idx = (history_idx + 1) % WINDOW_SIZE;
        if (history_count < WINDOW_SIZE) ++history_count;
    }

    /// Print all diagnostics to stderr for failure debugging
    void dump(const char* test_name) const {
        std::cerr << "\n======== FAILURE DIAGNOSTICS: " << test_name << " ========\n";
        std::cerr << "Grid: " << Nx << "x" << Ny;
        if (Nz > 1) std::cerr << "x" << Nz;
        std::cerr << ", dt=" << std::scientific << std::setprecision(3) << dt << "\n";
        std::cerr << "BCs: " << bcs << "\n";
        std::cerr << "Poisson: " << poisson_solver;
        if (poisson_iters > 0) std::cerr << " (" << poisson_iters << " iters)";
        std::cerr << "\n";
        std::cerr << "max|u|=" << max_u << ", max|v|=" << max_v;
        if (max_w > 0) std::cerr << ", max|w|=" << max_w;
        std::cerr << "\n";
        std::cerr << "E(t0)=" << E_initial << ", E(tf)=" << E_final << "\n";
        std::cerr << "max_div=" << max_div << " at (" << max_div_i << ","
                  << max_div_j;
        if (Nz > 1) std::cerr << "," << max_div_k;
        std::cerr << ")\n";

        // Print rolling window history (most recent steps)
        if (history_count > 0) {
            std::cerr << "--- Last " << history_count << " steps ---\n";
            std::cerr << std::setprecision(6);
            for (int i = 0; i < history_count; ++i) {
                int idx = (history_idx - history_count + i + WINDOW_SIZE) % WINDOW_SIZE;
                const auto& s = history[idx];
                std::cerr << "  step " << std::setw(4) << s.step
                          << ": E=" << std::scientific << s.E
                          << ", div=" << s.max_div
                          << ", |u|=" << s.max_u;
                if (s.poisson_iters > 0) std::cerr << ", p_iters=" << s.poisson_iters;
                std::cerr << "\n";
            }
        }

        // RANS diagnostics (only print if populated)
        if (min_k < 1e20) {
            std::cerr << "--- RANS diagnostics ---\n";
            std::cerr << "min(k)=" << min_k << ", min(omega)=" << min_omega << "\n";
            std::cerr << "min(nu_t)=" << min_nut << ", max(nu_t)=" << max_nut << "\n";
            std::cerr << "max(nu_t/nu)=" << max_nut_over_nu << "\n";
            std::cerr << "U_bulk=" << U_bulk << ", tau_w=" << tau_w << "\n";
        }
        std::cerr << "========================================================\n\n";
    }
};

/// Global diagnostics for current test (set before checks)
inline SimDiagnostics& current_diagnostics() {
    static SimDiagnostics diag;
    return diag;
}

/// Reset diagnostics for new test
inline void reset_diagnostics() {
    current_diagnostics() = SimDiagnostics{};
}

//=============================================================================
// CHECK_OR_DUMP Macro
//=============================================================================

/// Check condition, dump diagnostics on failure, and record result
/// Usage: CHECK_OR_DUMP("Energy monotonic", E_final <= E_initial, diag);
#define CHECK_OR_DUMP(name, condition, diagnostics) \
    do { \
        bool _pass = (condition); \
        if (!_pass) { \
            (diagnostics).dump(name); \
        } \
        nncfd::test::harness::record(name, _pass); \
    } while(0)

/// Version that also captures the actual vs expected values
#define CHECK_OR_DUMP_VAL(name, condition, actual, threshold, diagnostics) \
    do { \
        bool _pass = (condition); \
        if (!_pass) { \
            (diagnostics).dump(name); \
            std::cerr << "  Actual: " << std::scientific << (actual) \
                      << ", Threshold: " << (threshold) << "\n"; \
        } \
        std::ostringstream _msg; \
        _msg << std::scientific << std::setprecision(2); \
        _msg << "(val=" << (actual) << ", thr=" << (threshold) << ")"; \
        nncfd::test::harness::record(name, _pass, _msg.str()); \
    } while(0)

//=============================================================================
// QOI_JSON Emission (machine-readable metrics for CI parsing)
//=============================================================================

/// Emit a single QOI value as JSON line (for ci.sh to parse)
/// Format: QOI_JSON: {"test":"name","key":value,...}
/// This is more robust than regex-parsing human-readable output.

/// Helper to emit a double in scientific notation for JSON
/// Uses 15 significant digits to preserve full double precision for trend analysis
inline std::string json_double(double val) {
    if (!std::isfinite(val)) return "null";
    std::ostringstream ss;
    ss << std::scientific << std::setprecision(15) << val;
    return ss.str();
}

/// Emit QOI JSON for TGV 2D invariants
/// Keys use nondimensional names (ke_final not ke_final_J) since tests use nondim values
inline void emit_qoi_tgv_2d(double div_Linf, double ke_final, double ke_ratio,
                             double const_vel_Linf = -1.0) {
    std::cout << "QOI_JSON: {\"test\":\"tgv_2d\""
              << ",\"div_Linf\":" << json_double(div_Linf)
              << ",\"ke_final\":" << json_double(ke_final)
              << ",\"ke_ratio\":" << json_double(ke_ratio);
    if (const_vel_Linf >= 0.0) {
        std::cout << ",\"const_vel_Linf\":" << json_double(const_vel_Linf);
    }
    std::cout << "}\n" << std::flush;
}

/// Emit QOI JSON for TGV 3D invariants
inline void emit_qoi_tgv_3d(double div_Linf, double ke_final = -1.0) {
    std::cout << "QOI_JSON: {\"test\":\"tgv_3d\""
              << ",\"div_Linf\":" << json_double(div_Linf);
    if (ke_final >= 0.0) {
        std::cout << ",\"ke_final\":" << json_double(ke_final);
    }
    std::cout << "}\n" << std::flush;
}

/// Emit QOI JSON for repeatability test
inline void emit_qoi_repeatability(double ke_rel_diff, double u_rel_L2 = -1.0) {
    std::cout << "QOI_JSON: {\"test\":\"repeatability\""
              << ",\"ke_rel_diff\":" << json_double(ke_rel_diff);
    if (u_rel_L2 >= 0.0) {
        std::cout << ",\"u_rel_L2\":" << json_double(u_rel_L2);
    }
    std::cout << "}\n" << std::flush;
}

/// Emit QOI JSON for CPU/GPU bitwise comparison
inline void emit_qoi_cpu_gpu(double u_rel_L2, double p_rel_L2) {
    std::cout << "QOI_JSON: {\"test\":\"cpu_gpu\""
              << ",\"u_rel_L2\":" << json_double(u_rel_L2)
              << ",\"p_rel_L2\":" << json_double(p_rel_L2)
              << "}\n" << std::flush;
}

/// Emit QOI JSON for HYPRE vs MG comparison
/// Primary metrics: div_mg/div_hypre (incompressibility), gradp_relL2 (pressure gradient match)
/// Secondary metrics: u_rel_L2 (velocity), p_prime_rel_L2 (mean-removed pressure)
inline void emit_qoi_hypre(double p_prime_rel_L2, double u_rel_L2,
                            double mean_p_mg, double mean_p_hypre,
                            double div_mg, double div_hypre, double gradp_relL2) {
    std::cout << "QOI_JSON: {\"test\":\"hypre_vs_mg\""
              << ",\"p_prime_rel_L2\":" << json_double(p_prime_rel_L2)
              << ",\"u_rel_L2\":" << json_double(u_rel_L2)
              << ",\"mean_p_mg\":" << json_double(mean_p_mg)
              << ",\"mean_p_hypre\":" << json_double(mean_p_hypre)
              << ",\"div_mg\":" << json_double(div_mg)
              << ",\"div_hypre\":" << json_double(div_hypre)
              << ",\"gradp_relL2\":" << json_double(gradp_relL2)
              << "}\n" << std::flush;
}

/// Emit QOI JSON for MMS convergence test
inline void emit_qoi_mms(double spatial_order, double u_L2_error = -1.0) {
    std::cout << "QOI_JSON: {\"test\":\"mms\""
              << ",\"spatial_order\":" << json_double(spatial_order);
    if (u_L2_error >= 0.0) {
        std::cout << ",\"u_L2_error\":" << json_double(u_L2_error);
    }
    std::cout << "}\n" << std::flush;
}

/// Emit QOI JSON for RANS channel sanity
inline void emit_qoi_rans_channel(double u_bulk, double nut_ratio_max,
                                   double k_min = -1.0, double omega_min = -1.0) {
    std::cout << "QOI_JSON: {\"test\":\"rans_channel\""
              << ",\"u_bulk\":" << json_double(u_bulk)
              << ",\"nut_ratio_max\":" << json_double(nut_ratio_max);
    if (k_min >= 0.0) {
        std::cout << ",\"k_min\":" << json_double(k_min);
    }
    if (omega_min >= 0.0) {
        std::cout << ",\"omega_min\":" << json_double(omega_min);
    }
    std::cout << "}\n" << std::flush;
}

/// Emit QOI JSON for Fourier mode invariance test
inline void emit_qoi_fourier_mode(double ke_ratio, double max_v_over_max_u) {
    std::cout << "QOI_JSON: {\"test\":\"fourier_mode\""
              << ",\"ke_ratio\":" << json_double(ke_ratio)
              << ",\"max_v_over_max_u\":" << json_double(max_v_over_max_u)
              << "}\n" << std::flush;
}

/// Emit QOI JSON for performance gates
/// Each gate emits its own line; ci.sh aggregates into perf_gate nested map
/// Includes warmup/timed steps so baseline comparisons account for methodology changes
inline void emit_qoi_perf(const std::string& case_name, double ms_per_step,
                           double threshold_ms, int warmup_steps = 0, int timed_steps = 0) {
    std::cout << "QOI_JSON: {\"test\":\"perf_gate\""
              << ",\"case\":\"" << case_name << "\""
              << ",\"ms_per_step\":" << json_double(ms_per_step)
              << ",\"threshold_ms\":" << json_double(threshold_ms);
    if (warmup_steps > 0) {
        std::cout << ",\"warmup_steps\":" << warmup_steps;
    }
    if (timed_steps > 0) {
        std::cout << ",\"timed_steps\":" << timed_steps;
    }
    std::cout << "}\n" << std::flush;
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
