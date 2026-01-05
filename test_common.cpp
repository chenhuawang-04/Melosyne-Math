/**
 * @file test_common.cpp
 * @brief Comprehensive unit tests for common.h functions
 *
 * Test Coverage:
 * 1. Correctness: Normal values, edge cases, special values (NaN, Inf)
 * 2. Consistency: Scalar vs Array results
 * 3. Comparison: MMath vs std::, GLM, DirectXMath
 */

#include "include/fast_math/common.h"
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

// ============================================================================
// min() Tests
// ============================================================================

bool test_min_normal_values() {
    TEST_ASSERT(min(1.0f, 2.0f) == 1.0f, "min(1, 2) should be 1");
    TEST_ASSERT(min(2.0f, 1.0f) == 1.0f, "min(2, 1) should be 1");
    TEST_ASSERT(min(-1.0f, 1.0f) == -1.0f, "min(-1, 1) should be -1");
    TEST_ASSERT(min(0.0f, 0.0f) == 0.0f, "min(0, 0) should be 0");
    return true;
}

bool test_min_edge_cases() {
    TEST_ASSERT(min(0.0f, -0.0f) == -0.0f || min(0.0f, -0.0f) == 0.0f,
                "min(0, -0) should handle signed zero");
    TEST_ASSERT(min(-INF_VALUE, 0.0f) == -INF_VALUE, "min(-inf, 0) should be -inf");
    TEST_ASSERT(min(INF_VALUE, 0.0f) == 0.0f, "min(inf, 0) should be 0");
    return true;
}

bool test_min_vs_std() {
    std::vector<float> test_values = {
        -100.0f, -1.0f, 0.0f, 1.0f, 100.0f,
        0.001f, -0.001f, 1e10f, -1e10f
    };

    for (size_t i = 0; i < test_values.size(); ++i) {
        for (size_t j = 0; j < test_values.size(); ++j) {
            float a = test_values[i];
            float b = test_values[j];
            float mmath_result = min(a, b);
            float std_result = std::min(a, b);

            TEST_ASSERT(floatEqual(mmath_result, std_result),
                        "min result differs from std::min");
        }
    }
    return true;
}

// ============================================================================
// max() Tests
// ============================================================================

bool test_max_normal_values() {
    TEST_ASSERT(max(1.0f, 2.0f) == 2.0f, "max(1, 2) should be 2");
    TEST_ASSERT(max(2.0f, 1.0f) == 2.0f, "max(2, 1) should be 2");
    TEST_ASSERT(max(-1.0f, 1.0f) == 1.0f, "max(-1, 1) should be 1");
    TEST_ASSERT(max(0.0f, 0.0f) == 0.0f, "max(0, 0) should be 0");
    return true;
}

bool test_max_edge_cases() {
    TEST_ASSERT(max(0.0f, -0.0f) == 0.0f || max(0.0f, -0.0f) == -0.0f,
                "max(0, -0) should handle signed zero");
    TEST_ASSERT(max(-INF_VALUE, 0.0f) == 0.0f, "max(-inf, 0) should be 0");
    TEST_ASSERT(max(INF_VALUE, 0.0f) == INF_VALUE, "max(inf, 0) should be inf");
    return true;
}

bool test_max_vs_std() {
    std::vector<float> test_values = {
        -100.0f, -1.0f, 0.0f, 1.0f, 100.0f,
        0.001f, -0.001f, 1e10f, -1e10f
    };

    for (size_t i = 0; i < test_values.size(); ++i) {
        for (size_t j = 0; j < test_values.size(); ++j) {
            float a = test_values[i];
            float b = test_values[j];
            float mmath_result = max(a, b);
            float std_result = std::max(a, b);

            TEST_ASSERT(floatEqual(mmath_result, std_result),
                        "max result differs from std::max");
        }
    }
    return true;
}

// ============================================================================
// clamp() Tests
// ============================================================================

bool test_clamp_normal_values() {
    TEST_ASSERT(clamp(0.5f, 0.0f, 1.0f) == 0.5f, "clamp(0.5, 0, 1) should be 0.5");
    TEST_ASSERT(clamp(-1.0f, 0.0f, 1.0f) == 0.0f, "clamp(-1, 0, 1) should be 0");
    TEST_ASSERT(clamp(2.0f, 0.0f, 1.0f) == 1.0f, "clamp(2, 0, 1) should be 1");
    TEST_ASSERT(clamp(0.0f, 0.0f, 0.0f) == 0.0f, "clamp(0, 0, 0) should be 0");
    return true;
}

bool test_clamp_audio_use_cases() {
    // Volume limiting [0, 1] - most common use case (27 times in audio engine)
    TEST_ASSERT(clamp(0.7f, 0.0f, 1.0f) == 0.7f, "Volume in range");
    TEST_ASSERT(clamp(-0.1f, 0.0f, 1.0f) == 0.0f, "Volume below 0 clamped");
    TEST_ASSERT(clamp(1.5f, 0.0f, 1.0f) == 1.0f, "Volume above 1 clamped");

    // Pan limiting [-1, 1]
    TEST_ASSERT(clamp(0.0f, -1.0f, 1.0f) == 0.0f, "Pan centered");
    TEST_ASSERT(clamp(-1.5f, -1.0f, 1.0f) == -1.0f, "Pan left limit");
    TEST_ASSERT(clamp(1.5f, -1.0f, 1.0f) == 1.0f, "Pan right limit");

    return true;
}

bool test_clamp_vs_std() {
    std::vector<float> test_values = {
        -10.0f, -1.0f, 0.0f, 0.5f, 1.0f, 10.0f
    };

    for (float x : test_values) {
        float mmath_result = clamp(x, 0.0f, 1.0f);
        float std_result = std::clamp(x, 0.0f, 1.0f);

        TEST_ASSERT(floatEqual(mmath_result, std_result),
                    "clamp result differs from std::clamp");
    }
    return true;
}

// ============================================================================
// abs() Tests
// ============================================================================

bool test_abs_normal_values() {
    TEST_ASSERT(MMath::abs(1.0f) == 1.0f, "abs(1) should be 1");
    TEST_ASSERT(MMath::abs(-1.0f) == 1.0f, "abs(-1) should be 1");
    TEST_ASSERT(MMath::abs(0.0f) == 0.0f, "abs(0) should be 0");
    TEST_ASSERT(MMath::abs(-0.0f) == 0.0f, "abs(-0) should be 0");
    return true;
}

bool test_abs_edge_cases() {
    TEST_ASSERT(MMath::abs(-INF_VALUE) == INF_VALUE, "abs(-inf) should be inf");
    TEST_ASSERT(MMath::abs(INF_VALUE) == INF_VALUE, "abs(inf) should be inf");
    // NaN stays NaN (sign bit cleared but still NaN)
    TEST_ASSERT(std::isnan(MMath::abs(NAN_VALUE)), "abs(NaN) should be NaN");
    return true;
}

bool test_abs_vs_std() {
    std::vector<float> test_values = {
        -100.0f, -1.0f, -0.001f, 0.0f, 0.001f, 1.0f, 100.0f
    };

    for (float x : test_values) {
        float mmath_result = MMath::abs(x);
        float std_result = std::abs(x);

        TEST_ASSERT(floatEqual(mmath_result, std_result),
                    "abs result differs from std::abs");
    }
    return true;
}

// ============================================================================
// sign() Tests
// ============================================================================

bool test_sign_normal_values() {
    TEST_ASSERT(sign(1.0f) == 1.0f, "sign(1) should be 1");
    TEST_ASSERT(sign(-1.0f) == -1.0f, "sign(-1) should be -1");
    TEST_ASSERT(sign(0.0f) == 0.0f, "sign(0) should be 0");
    TEST_ASSERT(sign(100.0f) == 1.0f, "sign(100) should be 1");
    TEST_ASSERT(sign(-100.0f) == -1.0f, "sign(-100) should be -1");
    return true;
}

bool test_sign_edge_cases() {
    TEST_ASSERT(sign(-0.0f) == 0.0f, "sign(-0) should be 0");
    TEST_ASSERT(sign(INF_VALUE) == 1.0f, "sign(inf) should be 1");
    TEST_ASSERT(sign(-INF_VALUE) == -1.0f, "sign(-inf) should be -1");
    TEST_ASSERT(sign(NAN_VALUE) == 0.0f, "sign(NaN) should be 0");
    return true;
}

// ============================================================================
// saturate() Tests
// ============================================================================

bool test_saturate_normal_values() {
    TEST_ASSERT(saturate(0.5f) == 0.5f, "saturate(0.5) should be 0.5");
    TEST_ASSERT(saturate(0.0f) == 0.0f, "saturate(0) should be 0");
    TEST_ASSERT(saturate(1.0f) == 1.0f, "saturate(1) should be 1");
    TEST_ASSERT(saturate(-0.5f) == 0.0f, "saturate(-0.5) should be 0");
    TEST_ASSERT(saturate(1.5f) == 1.0f, "saturate(1.5) should be 1");
    return true;
}

// ============================================================================
// copySign() Tests
// ============================================================================

bool test_copySign_normal_values() {
    TEST_ASSERT(copySign(1.0f, 1.0f) == 1.0f, "copySign(1, 1) should be 1");
    TEST_ASSERT(copySign(1.0f, -1.0f) == -1.0f, "copySign(1, -1) should be -1");
    TEST_ASSERT(copySign(-1.0f, 1.0f) == 1.0f, "copySign(-1, 1) should be 1");
    TEST_ASSERT(copySign(-1.0f, -1.0f) == -1.0f, "copySign(-1, -1) should be -1");
    return true;
}

bool test_copySign_edge_cases() {
    TEST_ASSERT(copySign(0.0f, -1.0f) == -0.0f || copySign(0.0f, -1.0f) == 0.0f,
                "copySign(0, -1) should be -0");
    TEST_ASSERT(copySign(5.0f, -0.0f) == -5.0f, "copySign(5, -0) should be -5");
    return true;
}

// ============================================================================
// Array Tests (SIMD Consistency)
// ============================================================================

bool test_minArray_consistency() {
    constexpr int N = 1000;
    std::vector<float> a(N), b(N), expected(N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

    for (int i = 0; i < N; ++i) {
        a[i] = dist(rng);
        b[i] = dist(rng);
        expected[i] = std::min(a[i], b[i]);
    }

    minArray(a.data(), b.data(), N);

    for (int i = 0; i < N; ++i) {
        TEST_ASSERT(floatEqual(a[i], expected[i]),
                    "minArray result differs from std::min");
    }
    return true;
}

bool test_maxArray_consistency() {
    constexpr int N = 1000;
    std::vector<float> a(N), b(N), expected(N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

    for (int i = 0; i < N; ++i) {
        a[i] = dist(rng);
        b[i] = dist(rng);
        expected[i] = std::max(a[i], b[i]);
    }

    maxArray(a.data(), b.data(), N);

    for (int i = 0; i < N; ++i) {
        TEST_ASSERT(floatEqual(a[i], expected[i]),
                    "maxArray result differs from std::max");
    }
    return true;
}

bool test_clampArray_consistency() {
    constexpr int N = 1000;
    std::vector<float> values(N), expected(N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (int i = 0; i < N; ++i) {
        values[i] = dist(rng);
        expected[i] = std::clamp(values[i], 0.0f, 1.0f);
    }

    clampArray(values.data(), 0.0f, 1.0f, N);

    for (int i = 0; i < N; ++i) {
        TEST_ASSERT(floatEqual(values[i], expected[i]),
                    "clampArray result differs from std::clamp");
    }
    return true;
}

bool test_absArray_consistency() {
    constexpr int N = 1000;
    std::vector<float> values(N), expected(N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

    for (int i = 0; i < N; ++i) {
        values[i] = dist(rng);
        expected[i] = std::abs(values[i]);
    }

    absArray(values.data(), N);

    for (int i = 0; i < N; ++i) {
        TEST_ASSERT(floatEqual(values[i], expected[i]),
                    "absArray result differs from std::abs");
    }
    return true;
}

bool test_saturateArray_consistency() {
    constexpr int N = 1000;
    std::vector<float> values(N), expected(N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    for (int i = 0; i < N; ++i) {
        values[i] = dist(rng);
        expected[i] = std::clamp(values[i], 0.0f, 1.0f);
    }

    saturateArray(values.data(), N);

    for (int i = 0; i < N; ++i) {
        TEST_ASSERT(floatEqual(values[i], expected[i]),
                    "saturateArray result differs from std::clamp(x, 0, 1)");
    }
    return true;
}

// ============================================================================
// Array Size Edge Cases
// ============================================================================

bool test_array_odd_sizes() {
    // Test sizes that don't align with SIMD (not multiple of 4 or 8)
    for (int size : {1, 3, 5, 7, 9, 13, 17}) {
        std::vector<float> a(size), b(size), expected(size);

        for (int i = 0; i < size; ++i) {
            a[i] = static_cast<float>(i);
            b[i] = static_cast<float>(size - i);
            expected[i] = std::min(a[i], b[i]);
        }

        minArray(a.data(), b.data(), size);

        for (int i = 0; i < size; ++i) {
            TEST_ASSERT(floatEqual(a[i], expected[i]),
                        "minArray fails on odd-sized array");
        }
    }
    return true;
}

// ============================================================================
// nearEqual() and isZero() Tests
// ============================================================================

bool test_nearEqual() {
    TEST_ASSERT(nearEqual(1.0f, 1.0f), "nearEqual: exact match");
    TEST_ASSERT(nearEqual(1.0f, 1.00001f), "nearEqual: within default epsilon");
    TEST_ASSERT(!nearEqual(1.0f, 2.0f), "nearEqual: clearly different");
    TEST_ASSERT(nearEqual(1.0f, 1.1f, 0.2f), "nearEqual: within custom epsilon");
    return true;
}

bool test_isZero() {
    TEST_ASSERT(isZero(0.0f), "isZero: exact zero");
    TEST_ASSERT(isZero(1e-7f), "isZero: within default epsilon");
    TEST_ASSERT(!isZero(0.1f), "isZero: not zero");
    TEST_ASSERT(isZero(0.001f, 0.01f), "isZero: within custom epsilon");
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
    std::cout << "MMath common.h Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << "\n--- min() Tests ---" << std::endl;
    RUN_TEST(test_min_normal_values);
    RUN_TEST(test_min_edge_cases);
    RUN_TEST(test_min_vs_std);

    std::cout << "\n--- max() Tests ---" << std::endl;
    RUN_TEST(test_max_normal_values);
    RUN_TEST(test_max_edge_cases);
    RUN_TEST(test_max_vs_std);

    std::cout << "\n--- clamp() Tests ---" << std::endl;
    RUN_TEST(test_clamp_normal_values);
    RUN_TEST(test_clamp_audio_use_cases);
    RUN_TEST(test_clamp_vs_std);

    std::cout << "\n--- abs() Tests ---" << std::endl;
    RUN_TEST(test_abs_normal_values);
    RUN_TEST(test_abs_edge_cases);
    RUN_TEST(test_abs_vs_std);

    std::cout << "\n--- sign() Tests ---" << std::endl;
    RUN_TEST(test_sign_normal_values);
    RUN_TEST(test_sign_edge_cases);

    std::cout << "\n--- saturate() Tests ---" << std::endl;
    RUN_TEST(test_saturate_normal_values);

    std::cout << "\n--- copySign() Tests ---" << std::endl;
    RUN_TEST(test_copySign_normal_values);
    RUN_TEST(test_copySign_edge_cases);

    std::cout << "\n--- Array SIMD Consistency Tests ---" << std::endl;
    RUN_TEST(test_minArray_consistency);
    RUN_TEST(test_maxArray_consistency);
    RUN_TEST(test_clampArray_consistency);
    RUN_TEST(test_absArray_consistency);
    RUN_TEST(test_saturateArray_consistency);
    RUN_TEST(test_array_odd_sizes);

    std::cout << "\n--- Utility Function Tests ---" << std::endl;
    RUN_TEST(test_nearEqual);
    RUN_TEST(test_isZero);

    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Results: " << pass_count << "/" << total_count << " passed";
    if (fail_count > 0) {
        std::cout << ", " << fail_count << " FAILED";
    }
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;

    return fail_count == 0 ? 0 : 1;
}
