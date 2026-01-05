/**
 * @file test_sqrt.cpp
 * @brief Comprehensive unit tests for sqrt.h functions
 *
 * Test Coverage:
 * 1. Correctness: Normal values, edge cases, special values (0, Inf, NaN)
 * 2. Precision: Relative error < 0.01% for inverseSqrt/reciprocal
 * 3. Consistency: Scalar vs Array results
 * 4. Comparison: MMath vs std::, theoretical values
 */

#include "include/fast_math/sqrt.h"
#include <iostream>
#include <cmath>
#include <limits>
#include <vector>
#include <random>
#include <iomanip>

// Test framework
#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "FAIL: " << msg << " at line " << __LINE__ << std::endl; \
            return false; \
        } \
    } while(0)

#define RUN_TEST(test_func) \
    do { \
        std::cout << "Running " << #test_func << "... "; \
        if (test_func()) { \
            std::cout << "PASS" << std::endl; \
            pass_count++; \
        } else { \
            std::cout << "FAIL" << std::endl; \
            fail_count++; \
        } \
        total_count++; \
    } while(0)

using namespace MMath;

// Constants for testing
constexpr float EPSILON = 1e-5f;
constexpr float PRECISION_THRESHOLD = 0.0001f;  // 0.01% relative error
constexpr float NAN_VALUE = std::numeric_limits<float>::quiet_NaN();
constexpr float INF_VALUE = std::numeric_limits<float>::infinity();

// ============================================================================
// Helper Functions
// ============================================================================

bool floatEqual(float a, float b, float epsilon = EPSILON) {
    if (std::isnan(a) && std::isnan(b)) return true;
    if (std::isinf(a) && std::isinf(b) && (a > 0) == (b > 0)) return true;
    return std::abs(a - b) <= epsilon;
}

float relativeError(float computed, float expected) {
    if (expected == 0.0f) return std::abs(computed);
    return std::abs((computed - expected) / expected);
}

// ============================================================================
// sqrt() Tests
// ============================================================================

bool test_sqrt_normal_values() {
    TEST_ASSERT(floatEqual(MMath::sqrt(0.0f), 0.0f), "sqrt(0) should be 0");
    TEST_ASSERT(floatEqual(MMath::sqrt(1.0f), 1.0f), "sqrt(1) should be 1");
    TEST_ASSERT(floatEqual(MMath::sqrt(4.0f), 2.0f), "sqrt(4) should be 2");
    TEST_ASSERT(floatEqual(MMath::sqrt(9.0f), 3.0f), "sqrt(9) should be 3");
    TEST_ASSERT(floatEqual(MMath::sqrt(2.0f), 1.41421356f, 1e-6f), "sqrt(2) should be ~1.414");
    return true;
}

bool test_sqrt_edge_cases() {
    TEST_ASSERT(std::isnan(MMath::sqrt(-1.0f)), "sqrt(-1) should be NaN");
    TEST_ASSERT(std::isinf(MMath::sqrt(INF_VALUE)), "sqrt(inf) should be inf");
    TEST_ASSERT(std::isnan(MMath::sqrt(NAN_VALUE)), "sqrt(NaN) should be NaN");
    return true;
}

bool test_sqrt_vs_std() {
    std::vector<float> test_values = {
        0.0f, 0.001f, 0.1f, 1.0f, 2.0f, 3.0f, 4.0f,
        10.0f, 100.0f, 1000.0f, 1e6f
    };

    for (float x : test_values) {
        float mmath_result = MMath::sqrt(x);
        float std_result = std::sqrt(x);

        TEST_ASSERT(floatEqual(mmath_result, std_result, 1e-6f),
                    "sqrt result differs from std::sqrt");
    }
    return true;
}

// ============================================================================
// inverseSqrt() Tests
// ============================================================================

bool test_inverseSqrt_normal_values() {
    // Test known values
    TEST_ASSERT(floatEqual(inverseSqrt(1.0f), 1.0f, 1e-4f), "inverseSqrt(1) should be ~1");
    TEST_ASSERT(floatEqual(inverseSqrt(4.0f), 0.5f, 1e-4f), "inverseSqrt(4) should be ~0.5");
    TEST_ASSERT(floatEqual(inverseSqrt(9.0f), 0.333333f, 1e-3f), "inverseSqrt(9) should be ~0.333");
    TEST_ASSERT(floatEqual(inverseSqrt(0.25f), 2.0f, 1e-4f), "inverseSqrt(0.25) should be ~2");
    return true;
}

bool test_inverseSqrt_precision() {
    // Test precision against 1/std::sqrt over large range
    std::vector<float> test_values = {
        0.001f, 0.01f, 0.1f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        10.0f, 100.0f, 1000.0f, 10000.0f
    };

    for (float x : test_values) {
        float mmath_result = inverseSqrt(x);
        float expected = 1.0f / std::sqrt(x);
        float rel_error = relativeError(mmath_result, expected);

        TEST_ASSERT(rel_error < PRECISION_THRESHOLD,
                    "inverseSqrt relative error > 0.01%");
    }
    return true;
}

bool test_inverseSqrt_random_values() {
    // Test 10000 random values
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.001f, 10000.0f);

    int error_count = 0;
    float max_error = 0.0f;

    for (int i = 0; i < 10000; ++i) {
        float x = dist(rng);
        float mmath_result = inverseSqrt(x);
        float expected = 1.0f / std::sqrt(x);
        float rel_error = relativeError(mmath_result, expected);

        if (rel_error > PRECISION_THRESHOLD) {
            error_count++;
        }
        max_error = std::max(max_error, rel_error);
    }

    std::cout << "\n  inverseSqrt random test: max relative error = "
              << std::scientific << max_error
              << ", errors > 0.01% = " << error_count << "/10000";

    TEST_ASSERT(error_count < 10, "Too many inverseSqrt precision errors");
    TEST_ASSERT(max_error < 0.001f, "inverseSqrt max error > 0.1%");
    return true;
}

// ============================================================================
// reciprocal() Tests
// ============================================================================

bool test_reciprocal_normal_values() {
    TEST_ASSERT(floatEqual(reciprocal(1.0f), 1.0f, 1e-4f), "reciprocal(1) should be ~1");
    TEST_ASSERT(floatEqual(reciprocal(2.0f), 0.5f, 1e-4f), "reciprocal(2) should be ~0.5");
    TEST_ASSERT(floatEqual(reciprocal(4.0f), 0.25f, 1e-4f), "reciprocal(4) should be ~0.25");
    TEST_ASSERT(floatEqual(reciprocal(0.5f), 2.0f, 1e-4f), "reciprocal(0.5) should be ~2");
    TEST_ASSERT(floatEqual(reciprocal(0.1f), 10.0f, 1e-3f), "reciprocal(0.1) should be ~10");
    return true;
}

bool test_reciprocal_precision() {
    std::vector<float> test_values = {
        0.001f, 0.01f, 0.1f, 0.5f, 1.0f, 2.0f, 3.0f, 4.0f,
        10.0f, 100.0f, 1000.0f
    };

    for (float x : test_values) {
        float mmath_result = reciprocal(x);
        float expected = 1.0f / x;
        float rel_error = relativeError(mmath_result, expected);

        TEST_ASSERT(rel_error < PRECISION_THRESHOLD,
                    "reciprocal relative error > 0.01%");
    }
    return true;
}

bool test_reciprocal_random_values() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.001f, 10000.0f);

    int error_count = 0;
    float max_error = 0.0f;

    for (int i = 0; i < 10000; ++i) {
        float x = dist(rng);
        float mmath_result = reciprocal(x);
        float expected = 1.0f / x;
        float rel_error = relativeError(mmath_result, expected);

        if (rel_error > PRECISION_THRESHOLD) {
            error_count++;
        }
        max_error = std::max(max_error, rel_error);
    }

    std::cout << "\n  reciprocal random test: max relative error = "
              << std::scientific << max_error
              << ", errors > 0.01% = " << error_count << "/10000";

    TEST_ASSERT(error_count < 10, "Too many reciprocal precision errors");
    TEST_ASSERT(max_error < 0.001f, "reciprocal max error > 0.1%");
    return true;
}

// ============================================================================
// Array Tests (SIMD Consistency)
// ============================================================================

bool test_sqrtArray_consistency() {
    constexpr int N = 1000;
    std::vector<float> values(N), expected(N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.001f, 1000.0f);

    for (int i = 0; i < N; ++i) {
        values[i] = dist(rng);
        expected[i] = std::sqrt(values[i]);
    }

    sqrtArray(values.data(), N);

    for (int i = 0; i < N; ++i) {
        TEST_ASSERT(floatEqual(values[i], expected[i], 1e-6f),
                    "sqrtArray result differs from std::sqrt");
    }
    return true;
}

bool test_inverseSqrtArray_consistency() {
    constexpr int N = 1000;
    std::vector<float> values(N), expected(N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.001f, 1000.0f);

    for (int i = 0; i < N; ++i) {
        float x = dist(rng);
        values[i] = x;
        expected[i] = 1.0f / std::sqrt(x);
    }

    inverseSqrtArray(values.data(), N);

    int error_count = 0;
    for (int i = 0; i < N; ++i) {
        float rel_error = relativeError(values[i], expected[i]);
        if (rel_error > PRECISION_THRESHOLD) {
            error_count++;
        }
    }

    TEST_ASSERT(error_count < 10, "Too many inverseSqrtArray precision errors");
    return true;
}

bool test_reciprocalArray_consistency() {
    constexpr int N = 1000;
    std::vector<float> values(N), expected(N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.001f, 1000.0f);

    for (int i = 0; i < N; ++i) {
        float x = dist(rng);
        values[i] = x;
        expected[i] = 1.0f / x;
    }

    reciprocalArray(values.data(), N);

    int error_count = 0;
    for (int i = 0; i < N; ++i) {
        float rel_error = relativeError(values[i], expected[i]);
        if (rel_error > PRECISION_THRESHOLD) {
            error_count++;
        }
    }

    TEST_ASSERT(error_count < 10, "Too many reciprocalArray precision errors");
    return true;
}

bool test_array_odd_sizes() {
    // Test sizes that don't align with SIMD (not multiple of 4 or 8)
    for (int size : {1, 3, 5, 7, 9, 13, 17}) {
        std::vector<float> values(size);
        for (int i = 0; i < size; ++i) {
            values[i] = static_cast<float>(i + 1);
        }

        // Test sqrtArray
        auto sqrt_test = values;
        sqrtArray(sqrt_test.data(), size);
        for (int i = 0; i < size; ++i) {
            float expected = std::sqrt(static_cast<float>(i + 1));
            TEST_ASSERT(floatEqual(sqrt_test[i], expected, 1e-6f),
                        "sqrtArray fails on odd-sized array");
        }

        // Test inverseSqrtArray
        auto inverseSqrt_test = values;
        inverseSqrtArray(inverseSqrt_test.data(), size);
        for (int i = 0; i < size; ++i) {
            float expected = 1.0f / std::sqrt(static_cast<float>(i + 1));
            float rel_error = relativeError(inverseSqrt_test[i], expected);
            TEST_ASSERT(rel_error < PRECISION_THRESHOLD,
                        "inverseSqrtArray fails on odd-sized array");
        }
    }
    return true;
}

// ============================================================================
// Utility Function Tests
// ============================================================================

bool test_normalizeFactor() {
    // Test normalization factor (inverseSqrt of length squared)
    TEST_ASSERT(floatEqual(normalizeFactor(25.0f), 0.2f, 1e-4f),
                "normalizeFactor(25) should be ~0.2");
    TEST_ASSERT(floatEqual(normalizeFactor(1.0f), 1.0f, 1e-4f),
                "normalizeFactor(1) should be ~1");
    return true;
}

bool test_isSafeForDivision() {
    TEST_ASSERT(isSafeForDivision(1.0f), "1.0 is safe for division");
    TEST_ASSERT(isSafeForDivision(0.01f), "0.01 is safe for division");
    TEST_ASSERT(!isSafeForDivision(1e-9f), "1e-9 is unsafe for division (default epsilon)");
    TEST_ASSERT(!isSafeForDivision(0.0f), "0.0 is unsafe for division");
    return true;
}

bool test_safeReciprocal() {
    TEST_ASSERT(floatEqual(safeReciprocal(2.0f), 0.5f, 1e-4f), "safeReciprocal(2) should be ~0.5");
    TEST_ASSERT(floatEqual(safeReciprocal(1e-9f), 0.0f), "safeReciprocal(small) should return fallback");
    TEST_ASSERT(floatEqual(safeReciprocal(1e-9f, 1.0f), 1.0f), "safeReciprocal should use custom fallback");
    return true;
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    int total_count = 0;
    int pass_count = 0;
    int fail_count = 0;

    std::cout << "========================================" << std::endl;
    std::cout << "MMath sqrt.h Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << "\n--- sqrt() Tests ---" << std::endl;
    RUN_TEST(test_sqrt_normal_values);
    RUN_TEST(test_sqrt_edge_cases);
    RUN_TEST(test_sqrt_vs_std);

    std::cout << "\n--- inverseSqrt() Tests ---" << std::endl;
    RUN_TEST(test_inverseSqrt_normal_values);
    RUN_TEST(test_inverseSqrt_precision);
    RUN_TEST(test_inverseSqrt_random_values);

    std::cout << "\n--- reciprocal() Tests ---" << std::endl;
    RUN_TEST(test_reciprocal_normal_values);
    RUN_TEST(test_reciprocal_precision);
    RUN_TEST(test_reciprocal_random_values);

    std::cout << "\n--- Array SIMD Consistency Tests ---" << std::endl;
    RUN_TEST(test_sqrtArray_consistency);
    RUN_TEST(test_inverseSqrtArray_consistency);
    RUN_TEST(test_reciprocalArray_consistency);
    RUN_TEST(test_array_odd_sizes);

    std::cout << "\n--- Utility Function Tests ---" << std::endl;
    RUN_TEST(test_normalizeFactor);
    RUN_TEST(test_isSafeForDivision);
    RUN_TEST(test_safeReciprocal);

    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Results: " << pass_count << "/" << total_count << " passed";
    if (fail_count > 0) {
        std::cout << ", " << fail_count << " FAILED";
    }
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;

    return fail_count == 0 ? 0 : 1;
}
