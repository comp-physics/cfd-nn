/// @file test_numerics.cpp
/// @brief Unit tests for numerics.hpp safe division utilities

#include "numerics.hpp"
#include "test_harness.hpp"
#include <cmath>
#include <limits>

using namespace nncfd;
using namespace nncfd::test::harness;
using namespace nncfd::numerics;

//=============================================================================
// safe_divide tests
//=============================================================================

static void test_safe_divide() {
    // Normal division
    assert_close("safe_divide normal", safe_divide(1.0, 2.0), 0.5);
    assert_close("safe_divide negative num", safe_divide(-1.0, 2.0), -0.5);
    // Note: safe_divide preserves the sign of the denominator
    // This matches standard division behavior: 1/-2 = -0.5, -1/-2 = 0.5
    assert_close("safe_divide negative den", safe_divide(1.0, -2.0), -0.5);
    assert_close("safe_divide both negative", safe_divide(-1.0, -2.0), 0.5);

    // Division by zero - should use floor
    assert_close("safe_divide by zero", safe_divide(1.0, 0.0), 1e30, 1e25, 0.0);
    assert_close("safe_divide by tiny", safe_divide(1.0, 1e-40), 1e30, 1e25, 0.0);

    // Custom floor
    assert_close("safe_divide custom floor", safe_divide(1.0, 0.0, 1e-10), 1e10, 1e5, 0.0);

    // Large values
    assert_close("safe_divide large num", safe_divide(1e20, 2.0), 5e19, 1e14, 0.0);
}

//=============================================================================
// bounded_ratio tests
//=============================================================================

static void test_bounded_ratio() {
    // Normal ratios
    assert_close("bounded_ratio normal", bounded_ratio(1.0, 2.0), 0.5);

    // Should be clamped to ceiling
    double result = bounded_ratio(1e15, 1.0);
    record("bounded_ratio clamped to ceiling", result == 1e10);

    // Negative clamping
    result = bounded_ratio(-1e15, 1.0);
    record("bounded_ratio clamped negative", result == -1e10);

    // Custom floor and ceiling
    result = bounded_ratio(100.0, 0.0, 1e-10, 50.0);
    record("bounded_ratio custom ceiling", result == 50.0);

    // Small denominator
    result = bounded_ratio(1.0, 1e-15);
    record("bounded_ratio small den clamped", result == 1e10);
}

//=============================================================================
// is_finite tests
//=============================================================================

static void test_is_finite() {
    record("is_finite normal", is_finite(1.0));
    record("is_finite zero", is_finite(0.0));
    record("is_finite negative", is_finite(-1e10));
    record("is_finite NaN", !is_finite(std::numeric_limits<double>::quiet_NaN()));
    record("is_finite +Inf", !is_finite(std::numeric_limits<double>::infinity()));
    record("is_finite -Inf", !is_finite(-std::numeric_limits<double>::infinity()));
}

//=============================================================================
// in_range tests
//=============================================================================

static void test_in_range() {
    record("in_range inside", in_range(0.5, 0.0, 1.0));
    record("in_range at lo", in_range(0.0, 0.0, 1.0));
    record("in_range at hi", in_range(1.0, 0.0, 1.0));
    record("in_range below", !in_range(-0.1, 0.0, 1.0));
    record("in_range above", !in_range(1.1, 0.0, 1.0));
}

//=============================================================================
// clamp_with_flag tests
//=============================================================================

static void test_clamp_with_flag() {
    bool clamped;

    double result = clamp_with_flag(0.5, 0.0, 1.0, clamped);
    record("clamp_with_flag no clamp value", std::abs(result - 0.5) < 1e-15);
    record("clamp_with_flag no clamp flag", !clamped);

    result = clamp_with_flag(-0.5, 0.0, 1.0, clamped);
    record("clamp_with_flag lo clamp value", std::abs(result - 0.0) < 1e-15);
    record("clamp_with_flag lo clamp flag", clamped);

    result = clamp_with_flag(1.5, 0.0, 1.0, clamped);
    record("clamp_with_flag hi clamp value", std::abs(result - 1.0) < 1e-15);
    record("clamp_with_flag hi clamp flag", clamped);
}

//=============================================================================
// Constants tests
//=============================================================================

static void test_constants() {
    record("K_FLOOR positive", K_FLOOR > 0);
    record("K_FLOOR small", K_FLOOR < 1e-5);
    record("OMEGA_FLOOR positive", OMEGA_FLOOR > 0);
    record("Y_WALL_FLOOR positive", Y_WALL_FLOOR > 0);
    record("OMEGA_OVER_K_MAX large", OMEGA_OVER_K_MAX > 1e6);
    record("NU_T_RATIO_MAX reasonable", NU_T_RATIO_MAX >= 1000);
}

//=============================================================================
// Integration test: turbulence-like computation
//=============================================================================

static void test_turbulence_scenario() {
    // Simulate omega production term: alpha * (omega/k) * P_k
    // When k is very small, this can blow up

    double k_values[] = {1.0, 0.1, 1e-5, 1e-10, 1e-15, 0.0};
    double omega = 100.0;
    double alpha = 0.44;
    double P_k = 0.1;

    bool all_finite = true;
    bool all_bounded = true;

    for (double k : k_values) {
        // Without safe ratio (would blow up for small k)
        double omega_over_k = bounded_ratio(omega, k, K_FLOOR, OMEGA_OVER_K_MAX);
        double production = alpha * omega_over_k * P_k;

        if (!is_finite(production)) all_finite = false;
        if (std::abs(production) > 1e12) all_bounded = false;
    }

    record("turbulence scenario all finite", all_finite);
    record("turbulence scenario all bounded", all_bounded);
}

//=============================================================================
// Main
//=============================================================================

int main() {
    return run_sections("Numerics Utilities Tests", {
        {"safe_divide", test_safe_divide},
        {"bounded_ratio", test_bounded_ratio},
        {"is_finite", test_is_finite},
        {"in_range", test_in_range},
        {"clamp_with_flag", test_clamp_with_flag},
        {"Constants", test_constants},
        {"Turbulence Scenario", test_turbulence_scenario}
    });
}
