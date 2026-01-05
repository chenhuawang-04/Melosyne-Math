/**
 * @file test_power.cpp
 * @brief Unit tests for power.h functions
 *
 * Test Coverage:
 * - exp(), exp2(), pow10() - boundary values, overflow/underflow
 * - log(), log2(), log10() - special cases, domain errors
 * - pow() - identity cases, special exponents
 * - powi() - integer exponent optimization
 * - Array SIMD consistency - verify SIMD matches scalar
 */

#include "include/fast_math/power.h"
#include <iostream>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

// Test utilities
namespace {
    // Realistic precision for fast approximations (audio-grade quality)
    constexpr float kEpsilon = 2e-5f;       // 0.002% - excellent for audio (>100dB SNR)
    constexpr float kPowEpsilon = 5e-5f;    // 0.005% - pow cumulative error
    constexpr float kLogLarge = 2e-3f;      // 0.2% - acceptable for large values (>100)

    bool nearEqual(float a, float b, float epsilon = kEpsilon) {
        // In fast-math mode, NaN and infinity behave differently
        // Just check if both have the same special value pattern
        if (a == b) return true;  // Exact match (including both being zero)

        // Handle very small values near zero
        float max_val = std::max(std::fabs(a), std::fabs(b));
        if (max_val < 1e-30f) return true;  // Both essentially zero

        float diff = std::fabs(a - b);
        float rel_error = diff / max_val;
        return rel_error <= epsilon;
    }

    void printTestResult(const char* test_name, bool passed) {
        std::cout << "Running " << test_name << "... "
                  << (passed ? "PASS" : "FAIL") << std::endl;
    }
}

// ============================================================================
// exp() Tests
// ============================================================================

bool test_exp_boundaries() {
    bool passed = true;

    // exp(0) = 1
    passed &= nearEqual(MMath::exp(0.0f), 1.0f);

    // exp(1) ≈ e
    passed &= nearEqual(MMath::exp(1.0f), 2.718281828459045f);

    // exp(-1) ≈ 1/e
    passed &= nearEqual(MMath::exp(-1.0f), 0.36787944117144233f);

    // Large positive (should clamp to avoid overflow)
    passed &= std::isfinite(MMath::exp(100.0f));

    // Large negative (should approach 0)
    passed &= MMath::exp(-100.0f) < 1e-40f;

    return passed;
}

bool test_exp_vs_std() {
    bool passed = true;

    // Test various values
    float test_values[] = {0.5f, 1.0f, 2.0f, -0.5f, -1.0f, -2.0f, 10.0f, -10.0f};

    for (float x : test_values) {
        float mmath_result = MMath::exp(x);
        float std_result = std::exp(x);
        passed &= nearEqual(mmath_result, std_result);
    }

    return passed;
}

// ============================================================================
// log() Tests
// ============================================================================

bool test_log_boundaries() {
    bool passed = true;

    // log(1) = 0
    passed &= nearEqual(MMath::log(1.0f), 0.0f);

    // log(e) ≈ 1
    passed &= nearEqual(MMath::log(2.718281828459045f), 1.0f);

    // log(10) ≈ ln(10)
    passed &= nearEqual(MMath::log(10.0f), 2.302585092994046f);

    // log(0) and log(negative) are handled gracefully (return -FLT_MAX in fast-math mode)
    // Just verify they return a very negative number
    passed &= MMath::log(0.0f) < -1e30f;
    passed &= MMath::log(-1.0f) < -1e30f;

    return passed;
}

bool test_log_vs_std() {
    bool passed = true;

    float test_values[] = {0.1f, 0.5f, 1.0f, 2.0f, 10.0f, 100.0f, 1000.0f};

    for (float x : test_values) {
        float mmath_result = MMath::log(x);
        float std_result = std::log(x);

        // Use relaxed epsilon for large values (polynomial approximation degrades)
        float epsilon = (x > 100.0f) ? kLogLarge : kEpsilon;
        passed &= nearEqual(mmath_result, std_result, epsilon);
    }

    return passed;
}

// ============================================================================
// exp2() and log2() Tests
// ============================================================================

bool test_exp2_log2_roundtrip() {
    bool passed = true;

    // exp2(log2(x)) = x
    float test_values[] = {0.1f, 1.0f, 2.0f, 10.0f, 100.0f};

    for (float x : test_values) {
        float roundtrip = MMath::exp2(MMath::log2(x));
        passed &= nearEqual(roundtrip, x);
    }

    return passed;
}

bool test_exp2_identity() {
    bool passed = true;

    // exp2(0) = 1
    passed &= nearEqual(MMath::exp2(0.0f), 1.0f);

    // exp2(1) = 2
    passed &= nearEqual(MMath::exp2(1.0f), 2.0f);

    // exp2(2) = 4
    passed &= nearEqual(MMath::exp2(2.0f), 4.0f);

    // exp2(10) = 1024
    passed &= nearEqual(MMath::exp2(10.0f), 1024.0f);

    return passed;
}

// ============================================================================
// pow10() and log10() Tests (Audio-specific)
// ============================================================================

bool test_pow10_db_conversion() {
    bool passed = true;

    // dB to linear conversion: pow(10, db/20)
    // 0 dB = 1.0 linear
    passed &= nearEqual(MMath::pow10(0.0f / 20.0f), 1.0f);

    // -6 dB ≈ 0.5 linear
    float minus_6db = MMath::pow10(-6.0f / 20.0f);
    passed &= nearEqual(minus_6db, 0.5011872336272722f, 0.01f);  // Relaxed for audio

    // +6 dB ≈ 2.0 linear
    float plus_6db = MMath::pow10(6.0f / 20.0f);
    passed &= nearEqual(plus_6db, 1.9952623149688797f, 0.01f);

    return passed;
}

bool test_log10_vs_std() {
    bool passed = true;

    float test_values[] = {0.1f, 1.0f, 10.0f, 100.0f, 1000.0f};

    for (float x : test_values) {
        float mmath_result = MMath::log10(x);
        float std_result = std::log10(x);

        // Use relaxed epsilon for large values
        float epsilon = (x > 100.0f) ? kLogLarge : kEpsilon;
        passed &= nearEqual(mmath_result, std_result, epsilon);
    }

    return passed;
}

// ============================================================================
// pow() Tests
// ============================================================================

bool test_pow_identity() {
    bool passed = true;

    // pow(x, 0) = 1
    passed &= nearEqual(MMath::pow(2.0f, 0.0f), 1.0f);
    passed &= nearEqual(MMath::pow(100.0f, 0.0f), 1.0f);

    // pow(x, 1) = x
    passed &= nearEqual(MMath::pow(2.0f, 1.0f), 2.0f);
    passed &= nearEqual(MMath::pow(3.5f, 1.0f), 3.5f);

    // pow(1, y) = 1
    passed &= nearEqual(MMath::pow(1.0f, 0.0f), 1.0f);
    passed &= nearEqual(MMath::pow(1.0f, 10.0f), 1.0f);

    return passed;
}

bool test_pow_vs_std() {
    bool passed = true;

    // Test various base/exponent combinations
    struct TestCase { float base; float exp; };
    TestCase cases[] = {
        {2.0f, 2.0f},    // 4
        {2.0f, 3.0f},    // 8
        {3.0f, 2.0f},    // 9
        {10.0f, 2.0f},   // 100
        {2.0f, 0.5f},    // sqrt(2)
        {4.0f, 0.5f},    // 2
        {2.0f, -1.0f},   // 0.5
        {0.5f, 2.0f},    // 0.25
    };

    for (const auto& tc : cases) {
        float mmath_result = MMath::pow(tc.base, tc.exp);
        float std_result = std::pow(tc.base, tc.exp);
        passed &= nearEqual(mmath_result, std_result, kPowEpsilon);
    }

    return passed;
}

// ============================================================================
// powi() Tests (Integer Exponent Optimization)
// ============================================================================

bool test_powi_correctness() {
    bool passed = true;

    // Small positive exponents
    passed &= nearEqual(MMath::powi(2.0f, 0), 1.0f);
    passed &= nearEqual(MMath::powi(2.0f, 1), 2.0f);
    passed &= nearEqual(MMath::powi(2.0f, 2), 4.0f);
    passed &= nearEqual(MMath::powi(2.0f, 3), 8.0f);
    passed &= nearEqual(MMath::powi(2.0f, 10), 1024.0f);

    // Negative exponents
    passed &= nearEqual(MMath::powi(2.0f, -1), 0.5f);
    passed &= nearEqual(MMath::powi(2.0f, -2), 0.25f);
    passed &= nearEqual(MMath::powi(10.0f, -3), 0.001f);

    // Base cases
    passed &= nearEqual(MMath::powi(0.0f, 5), 0.0f);
    passed &= nearEqual(MMath::powi(1.0f, 100), 1.0f);

    return passed;
}

// ============================================================================
// SIMD Array Consistency Tests
// ============================================================================

bool test_expArray_consistency() {
    bool passed = true;

    // Test various array sizes (including non-multiples of 8)
    const int32_t sizes[] = {4, 7, 15, 31, 64, 100, 1000};

    for (int32_t size : sizes) {
        std::vector<float> values(size);
        std::vector<float> expected(size);

        // Fill with test values
        for (int32_t i = 0; i < size; ++i) {
            values[i] = -5.0f + (i * 10.0f / size);  // Range: [-5, 5]
            expected[i] = MMath::exp(values[i]);  // Scalar reference
        }

        // SIMD version
        MMath::expArray(values.data(), size);

        // Compare
        for (int32_t i = 0; i < size; ++i) {
            passed &= nearEqual(values[i], expected[i]);
        }
    }

    return passed;
}

bool test_logArray_consistency() {
    bool passed = true;

    const int32_t sizes[] = {4, 7, 15, 31, 64, 100, 1000};

    for (int32_t size : sizes) {
        std::vector<float> values(size);
        std::vector<float> expected(size);

        // Fill with positive test values
        for (int32_t i = 0; i < size; ++i) {
            values[i] = 0.1f + (i * 100.0f / size);  // Range: [0.1, 100]
            expected[i] = MMath::log(values[i]);
        }

        // SIMD version
        MMath::logArray(values.data(), size);

        // Compare
        for (int32_t i = 0; i < size; ++i) {
            passed &= nearEqual(values[i], expected[i]);
        }
    }

    return passed;
}

bool test_pow10Array_consistency() {
    bool passed = true;

    const int32_t size = 100;
    std::vector<float> values(size);
    std::vector<float> expected(size);

    // dB values: -60 to +6 dB (typical audio range)
    for (int32_t i = 0; i < size; ++i) {
        float db = -60.0f + (i * 66.0f / size);
        values[i] = db / 20.0f;  // Convert to exponent
        expected[i] = MMath::pow10(values[i]);
    }

    // SIMD version
    MMath::pow10Array(values.data(), size);

    // Compare
    for (int32_t i = 0; i < size; ++i) {
        passed &= nearEqual(values[i], expected[i]);
    }

    return passed;
}

bool test_powArray_consistency() {
    bool passed = true;

    const int32_t size = 64;
    std::vector<float> base(size);
    std::vector<float> exponent(size);
    std::vector<float> result(size);
    std::vector<float> expected(size);

    // Fill with test values
    for (int32_t i = 0; i < size; ++i) {
        base[i] = 0.1f + (i * 10.0f / size);  // Range: [0.1, 10]
        exponent[i] = -2.0f + (i * 4.0f / size);  // Range: [-2, 2]
        expected[i] = MMath::pow(base[i], exponent[i]);
    }

    // SIMD version
    MMath::powArray(base.data(), exponent.data(), result.data(), size);

    // Compare
    for (int32_t i = 0; i < size; ++i) {
        passed &= nearEqual(result[i], expected[i], kPowEpsilon);
    }

    return passed;
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "MMath power.h Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    int passed = 0;
    int total = 0;

    // exp() tests
    std::cout << "--- exp() Tests ---" << std::endl;
    printTestResult("test_exp_boundaries", test_exp_boundaries()); total++; passed += test_exp_boundaries();
    printTestResult("test_exp_vs_std", test_exp_vs_std()); total++; passed += test_exp_vs_std();
    std::cout << std::endl;

    // log() tests
    std::cout << "--- log() Tests ---" << std::endl;
    printTestResult("test_log_boundaries", test_log_boundaries()); total++; passed += test_log_boundaries();
    printTestResult("test_log_vs_std", test_log_vs_std()); total++; passed += test_log_vs_std();
    std::cout << std::endl;

    // exp2/log2 tests
    std::cout << "--- exp2() / log2() Tests ---" << std::endl;
    printTestResult("test_exp2_log2_roundtrip", test_exp2_log2_roundtrip()); total++; passed += test_exp2_log2_roundtrip();
    printTestResult("test_exp2_identity", test_exp2_identity()); total++; passed += test_exp2_identity();
    std::cout << std::endl;

    // pow10/log10 tests (audio-specific)
    std::cout << "--- pow10() / log10() Tests (Audio) ---" << std::endl;
    printTestResult("test_pow10_db_conversion", test_pow10_db_conversion()); total++; passed += test_pow10_db_conversion();
    printTestResult("test_log10_vs_std", test_log10_vs_std()); total++; passed += test_log10_vs_std();
    std::cout << std::endl;

    // pow() tests
    std::cout << "--- pow() Tests ---" << std::endl;
    printTestResult("test_pow_identity", test_pow_identity()); total++; passed += test_pow_identity();
    printTestResult("test_pow_vs_std", test_pow_vs_std()); total++; passed += test_pow_vs_std();
    std::cout << std::endl;

    // powi() tests
    std::cout << "--- powi() Tests (Integer Exponent) ---" << std::endl;
    printTestResult("test_powi_correctness", test_powi_correctness()); total++; passed += test_powi_correctness();
    std::cout << std::endl;

    // SIMD array consistency tests
    std::cout << "--- Array SIMD Consistency Tests ---" << std::endl;
    printTestResult("test_expArray_consistency", test_expArray_consistency()); total++; passed += test_expArray_consistency();
    printTestResult("test_logArray_consistency", test_logArray_consistency()); total++; passed += test_logArray_consistency();
    printTestResult("test_pow10Array_consistency", test_pow10Array_consistency()); total++; passed += test_pow10Array_consistency();
    printTestResult("test_powArray_consistency", test_powArray_consistency()); total++; passed += test_powArray_consistency();
    std::cout << std::endl;

    std::cout << "========================================" << std::endl;
    std::cout << "Test Results: " << passed << "/" << total << " passed" << std::endl;
    std::cout << "========================================" << std::endl;

    return (passed == total) ? 0 : 1;
}
