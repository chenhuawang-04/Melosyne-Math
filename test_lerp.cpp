/**
 * @file test_lerp.cpp
 * @brief Comprehensive unit tests for lerp.h functions
 *
 * Test Coverage:
 * 1. Correctness: Boundary values (t=0, t=0.5, t=1), edge cases
 * 2. Mathematical properties: lerp(a,b,0)=a, lerp(a,b,1)=b, midpoint
 * 3. Consistency: Scalar vs Array results
 * 4. Smoothstep: Derivatives at boundaries, S-curve shape
 */

#include "include/fast_math/lerp.h"
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

// ============================================================================
// Helper Functions
// ============================================================================

bool floatEqual(float a, float b, float epsilon = EPSILON) {
    return std::abs(a - b) <= epsilon;
}

// ============================================================================
// lerp() Tests
// ============================================================================

bool test_lerp_boundaries() {
    // Test t=0 returns a
    TEST_ASSERT(floatEqual(lerp(10.0f, 20.0f, 0.0f), 10.0f), "lerp(a,b,0) should equal a");

    // Test t=1 returns b
    TEST_ASSERT(floatEqual(lerp(10.0f, 20.0f, 1.0f), 20.0f), "lerp(a,b,1) should equal b");

    // Test t=0.5 returns midpoint
    TEST_ASSERT(floatEqual(lerp(10.0f, 20.0f, 0.5f), 15.0f), "lerp(a,b,0.5) should be midpoint");

    return true;
}

bool test_lerp_linearity() {
    // Test linear progression
    float a = 0.0f, b = 100.0f;

    TEST_ASSERT(floatEqual(lerp(a, b, 0.25f), 25.0f), "lerp should be linear");
    TEST_ASSERT(floatEqual(lerp(a, b, 0.75f), 75.0f), "lerp should be linear");

    // Test negative range
    TEST_ASSERT(floatEqual(lerp(-10.0f, 10.0f, 0.5f), 0.0f), "lerp should handle negative values");

    return true;
}

bool test_lerp_extrapolation() {
    // Test t < 0 (extrapolation below)
    TEST_ASSERT(floatEqual(lerp(10.0f, 20.0f, -0.5f), 5.0f), "lerp should extrapolate for t<0");

    // Test t > 1 (extrapolation above)
    TEST_ASSERT(floatEqual(lerp(10.0f, 20.0f, 1.5f), 25.0f), "lerp should extrapolate for t>1");

    return true;
}

bool test_lerp_identity_cases() {
    // Test lerp(a, a, t) = a for any t
    TEST_ASSERT(floatEqual(lerp(5.0f, 5.0f, 0.0f), 5.0f), "lerp(a,a,0) = a");
    TEST_ASSERT(floatEqual(lerp(5.0f, 5.0f, 0.5f), 5.0f), "lerp(a,a,0.5) = a");
    TEST_ASSERT(floatEqual(lerp(5.0f, 5.0f, 1.0f), 5.0f), "lerp(a,a,1) = a");

    return true;
}

// ============================================================================
// inverseLerp() Tests
// ============================================================================

bool test_inverseLerp_boundaries() {
    // inverseLerp(a, b, a) = 0
    TEST_ASSERT(floatEqual(inverseLerp(10.0f, 20.0f, 10.0f), 0.0f),
                "inverseLerp(a,b,a) should be 0");

    // inverseLerp(a, b, b) = 1
    TEST_ASSERT(floatEqual(inverseLerp(10.0f, 20.0f, 20.0f), 1.0f),
                "inverseLerp(a,b,b) should be 1");

    // inverseLerp(a, b, midpoint) = 0.5
    TEST_ASSERT(floatEqual(inverseLerp(10.0f, 20.0f, 15.0f), 0.5f),
                "inverseLerp should return 0.5 for midpoint");

    return true;
}

bool test_inverseLerp_roundtrip() {
    // Test lerp(a, b, inverseLerp(a, b, value)) ≈ value
    float a = 10.0f, b = 50.0f;

    for (float value = a; value <= b; value += 5.0f) {
        float t = inverseLerp(a, b, value);
        float reconstructed = lerp(a, b, t);
        TEST_ASSERT(floatEqual(reconstructed, value, 1e-4f),
                    "lerp/inverseLerp roundtrip should be identity");
    }

    return true;
}

bool test_inverseLerp_degenerate() {
    // Test a == b (degenerate case)
    TEST_ASSERT(floatEqual(inverseLerp(5.0f, 5.0f, 5.0f), 0.0f),
                "inverseLerp should return 0 for degenerate case");

    return true;
}

// ============================================================================
// remap() Tests
// ============================================================================

bool test_remap_basic() {
    // Test MIDI note (0-127) to frequency scaling example
    // Simplified: remap 60 from [0, 127] to [0, 1]
    TEST_ASSERT(floatEqual(remap(60.0f, 0.0f, 127.0f, 0.0f, 1.0f), 60.0f/127.0f, 1e-4f),
                "remap should correctly map ranges");

    // Test remapping [0, 10] to [100, 200]
    TEST_ASSERT(floatEqual(remap(5.0f, 0.0f, 10.0f, 100.0f, 200.0f), 150.0f),
                "remap should map midpoint to midpoint");

    return true;
}

bool test_remap_negative_ranges() {
    // Test remapping with negative values
    TEST_ASSERT(floatEqual(remap(0.0f, -1.0f, 1.0f, 0.0f, 100.0f), 50.0f),
                "remap should handle negative input ranges");

    return true;
}

// ============================================================================
// smoothstep() Tests
// ============================================================================

bool test_smoothstep_boundaries() {
    // smoothstep should return 0 for t <= edge0
    TEST_ASSERT(floatEqual(smoothstep(0.0f, 1.0f, -0.5f), 0.0f),
                "smoothstep should return 0 for t < edge0");
    TEST_ASSERT(floatEqual(smoothstep(0.0f, 1.0f, 0.0f), 0.0f),
                "smoothstep should return 0 for t = edge0");

    // smoothstep should return 1 for t >= edge1
    TEST_ASSERT(floatEqual(smoothstep(0.0f, 1.0f, 1.0f), 1.0f),
                "smoothstep should return 1 for t = edge1");
    TEST_ASSERT(floatEqual(smoothstep(0.0f, 1.0f, 1.5f), 1.0f),
                "smoothstep should return 1 for t > edge1");

    return true;
}

bool test_smoothstep_midpoint() {
    // smoothstep(0, 1, 0.5) should be 0.5 (S-curve passes through midpoint)
    TEST_ASSERT(floatEqual(smoothstep(0.0f, 1.0f, 0.5f), 0.5f),
                "smoothstep should pass through midpoint");

    return true;
}

bool test_smoothstep_monotonic() {
    // Test that smoothstep is monotonically increasing
    std::vector<float> values;
    for (float t = 0.0f; t <= 1.0f; t += 0.1f) {
        values.push_back(smoothstep(0.0f, 1.0f, t));
    }

    for (size_t i = 1; i < values.size(); ++i) {
        TEST_ASSERT(values[i] >= values[i-1], "smoothstep should be monotonically increasing");
    }

    return true;
}

// ============================================================================
// smootherstep() Tests
// ============================================================================

bool test_smootherstep_boundaries() {
    TEST_ASSERT(floatEqual(smootherstep(0.0f, 1.0f, 0.0f), 0.0f),
                "smootherstep(0,1,0) = 0");
    TEST_ASSERT(floatEqual(smootherstep(0.0f, 1.0f, 1.0f), 1.0f),
                "smootherstep(0,1,1) = 1");
    TEST_ASSERT(floatEqual(smootherstep(0.0f, 1.0f, 0.5f), 0.5f),
                "smootherstep(0,1,0.5) = 0.5");

    return true;
}

bool test_smootherstep_smoother_than_smoothstep() {
    // At t=0.25, smootherstep should be lower than smoothstep (smoother acceleration)
    float smooth = smoothstep(0.0f, 1.0f, 0.25f);
    float smoother = smootherstep(0.0f, 1.0f, 0.25f);

    TEST_ASSERT(smoother < smooth, "smootherstep should be smoother at t=0.25");

    // At t=0.75, smootherstep should be higher than smoothstep (smoother deceleration)
    smooth = smoothstep(0.0f, 1.0f, 0.75f);
    smoother = smootherstep(0.0f, 1.0f, 0.75f);

    TEST_ASSERT(smoother > smooth, "smootherstep should be smoother at t=0.75");

    return true;
}

// ============================================================================
// lerpClamped() Tests
// ============================================================================

bool test_lerpClamped() {
    // Test t < 0 is clamped
    TEST_ASSERT(floatEqual(lerpClamped(10.0f, 20.0f, -0.5f), 10.0f),
                "lerpClamped should clamp t<0 to 0");

    // Test t > 1 is clamped
    TEST_ASSERT(floatEqual(lerpClamped(10.0f, 20.0f, 1.5f), 20.0f),
                "lerpClamped should clamp t>1 to 1");

    // Test normal range
    TEST_ASSERT(floatEqual(lerpClamped(10.0f, 20.0f, 0.5f), 15.0f),
                "lerpClamped should work normally for t in [0,1]");

    return true;
}

// ============================================================================
// Array Tests (SIMD Consistency)
// ============================================================================

bool test_lerpArray_consistency() {
    constexpr int N = 1000;
    std::vector<float> a(N), b(N), result(N), expected(N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

    for (int i = 0; i < N; ++i) {
        a[i] = dist(rng);
        b[i] = dist(rng);
    }

    float t = 0.3f;

    // Compute with SIMD
    lerpArray(a.data(), b.data(), t, result.data(), N);

    // Compute expected with scalar
    for (int i = 0; i < N; ++i) {
        expected[i] = lerp(a[i], b[i], t);
    }

    // Compare
    for (int i = 0; i < N; ++i) {
        TEST_ASSERT(floatEqual(result[i], expected[i], 1e-4f),
                    "lerpArray result differs from scalar lerp");
    }

    return true;
}

bool test_smoothstepArray_consistency() {
    constexpr int N = 1000;
    std::vector<float> values(N), expected(N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 2.0f);

    for (int i = 0; i < N; ++i) {
        values[i] = dist(rng);
        expected[i] = smoothstep(0.0f, 1.0f, values[i]);
    }

    smoothstepArray(0.0f, 1.0f, values.data(), N);

    for (int i = 0; i < N; ++i) {
        TEST_ASSERT(floatEqual(values[i], expected[i], 1e-4f),
                    "smoothstepArray result differs from scalar smoothstep");
    }

    return true;
}

bool test_array_odd_sizes() {
    // Test sizes that don't align with SIMD (not multiple of 4 or 8)
    for (int size : {1, 3, 5, 7, 9, 13, 17}) {
        std::vector<float> a(size), b(size), result(size);

        for (int i = 0; i < size; ++i) {
            a[i] = static_cast<float>(i);
            b[i] = static_cast<float>(i + 10);
        }

        lerpArray(a.data(), b.data(), 0.5f, result.data(), size);

        for (int i = 0; i < size; ++i) {
            float expected = lerp(a[i], b[i], 0.5f);
            TEST_ASSERT(floatEqual(result[i], expected),
                        "lerpArray fails on odd-sized array");
        }
    }
    return true;
}

// ============================================================================
// Utility Function Tests
// ============================================================================

bool test_isInRange() {
    TEST_ASSERT(isInRange(5.0f, 0.0f, 10.0f), "5 should be in [0, 10]");
    TEST_ASSERT(!isInRange(-1.0f, 0.0f, 10.0f), "-1 should not be in [0, 10]");
    TEST_ASSERT(!isInRange(11.0f, 0.0f, 10.0f), "11 should not be in [0, 10]");
    TEST_ASSERT(isInRange(0.0f, 0.0f, 10.0f), "0 should be in [0, 10]");
    TEST_ASSERT(isInRange(10.0f, 0.0f, 10.0f), "10 should be in [0, 10]");

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
    std::cout << "MMath lerp.h Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << "\n--- lerp() Tests ---" << std::endl;
    RUN_TEST(test_lerp_boundaries);
    RUN_TEST(test_lerp_linearity);
    RUN_TEST(test_lerp_extrapolation);
    RUN_TEST(test_lerp_identity_cases);

    std::cout << "\n--- inverseLerp() Tests ---" << std::endl;
    RUN_TEST(test_inverseLerp_boundaries);
    RUN_TEST(test_inverseLerp_roundtrip);
    RUN_TEST(test_inverseLerp_degenerate);

    std::cout << "\n--- remap() Tests ---" << std::endl;
    RUN_TEST(test_remap_basic);
    RUN_TEST(test_remap_negative_ranges);

    std::cout << "\n--- smoothstep() Tests ---" << std::endl;
    RUN_TEST(test_smoothstep_boundaries);
    RUN_TEST(test_smoothstep_midpoint);
    RUN_TEST(test_smoothstep_monotonic);

    std::cout << "\n--- smootherstep() Tests ---" << std::endl;
    RUN_TEST(test_smootherstep_boundaries);
    RUN_TEST(test_smootherstep_smoother_than_smoothstep);

    std::cout << "\n--- lerpClamped() Tests ---" << std::endl;
    RUN_TEST(test_lerpClamped);

    std::cout << "\n--- Array SIMD Consistency Tests ---" << std::endl;
    RUN_TEST(test_lerpArray_consistency);
    RUN_TEST(test_smoothstepArray_consistency);
    RUN_TEST(test_array_odd_sizes);

    std::cout << "\n--- Utility Function Tests ---" << std::endl;
    RUN_TEST(test_isInRange);

    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Results: " << pass_count << "/" << total_count << " passed";
    if (fail_count > 0) {
        std::cout << ", " << fail_count << " FAILED";
    }
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;

    return fail_count == 0 ? 0 : 1;
}
