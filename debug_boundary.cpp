// Comprehensive boundary test
#include "include/fast_math/power.h"
#include <iostream>
#include <cmath>
#include <iomanip>
#include <cstring>

void test_log_edge_cases() {
    std::cout << "=== log() edge cases ===" << std::endl;

    float test_vals[] = {0.1f, 0.5f, 1.0f, 2.0f, 10.0f, 100.0f};

    for (float x : test_vals) {
        float mmath = MMath::log(x);
        float std_val = std::log(x);
        float abs_err = std::fabs(mmath - std_val);
        float rel_err = abs_err / std::fabs(std_val);

        std::cout << std::setprecision(10);
        std::cout << "log(" << x << "): MMath=" << mmath
                  << " std=" << std_val
                  << " abs_err=" << abs_err
                  << " rel_err=" << rel_err;

        if (rel_err > 1e-5f) {
            std::cout << " ❌ FAIL";
        } else {
            std::cout << " ✅ PASS";
        }
        std::cout << std::endl;
    }
}

void test_log2_consistency() {
    std::cout << "\n=== log2() consistency ===" << std::endl;

    float test_vals[] = {0.5f, 1.0f, 2.0f, 4.0f, 8.0f, 10.0f};

    for (float x : test_vals) {
        float mmath_log2 = MMath::log2(x);
        float std_log2 = std::log2(x);
        float rel_err = std::fabs(mmath_log2 - std_log2) / std::fabs(std_log2);

        std::cout << "log2(" << x << "): MMath=" << mmath_log2
                  << " std=" << std_log2
                  << " rel_err=" << rel_err;

        if (rel_err > 1e-5f) {
            std::cout << " ❌ FAIL";
        } else {
            std::cout << " ✅ PASS";
        }
        std::cout << std::endl;
    }
}

void test_log10_consistency() {
    std::cout << "\n=== log10() consistency ===" << std::endl;

    float test_vals[] = {0.1f, 1.0f, 10.0f, 100.0f, 1000.0f};

    for (float x : test_vals) {
        float mmath_log10 = MMath::log10(x);
        float std_log10 = std::log10(x);
        float rel_err = std::fabs(mmath_log10 - std_log10) / std::fabs(std_log10);

        std::cout << "log10(" << x << "): MMath=" << mmath_log10
                  << " std=" << std_log10
                  << " rel_err=" << rel_err;

        if (rel_err > 1e-5f) {
            std::cout << " ❌ FAIL";
        } else {
            std::cout << " ✅ PASS";
        }
        std::cout << std::endl;
    }
}

void test_array_consistency() {
    std::cout << "\n=== Array SIMD consistency ===" << std::endl;

    const int N = 100;
    float values[N];
    float expected[N];

    // Test expArray
    std::cout << "\nexpArray:" << std::endl;
    for (int i = 0; i < N; ++i) {
        values[i] = -5.0f + (i * 10.0f / N);
        expected[i] = MMath::exp(values[i]);
    }

    MMath::expArray(values, N);

    float max_err = 0.0f;
    for (int i = 0; i < N; ++i) {
        float err = std::fabs(values[i] - expected[i]);
        if (err > max_err) max_err = err;
    }
    std::cout << "  Max absolute error: " << max_err;
    if (max_err > 1e-4f) {
        std::cout << " ❌ FAIL";
    } else {
        std::cout << " ✅ PASS";
    }
    std::cout << std::endl;

    // Test logArray
    std::cout << "\nlogArray:" << std::endl;
    for (int i = 0; i < N; ++i) {
        values[i] = 0.1f + (i * 100.0f / N);
        expected[i] = MMath::log(values[i]);
    }

    MMath::logArray(values, N);

    max_err = 0.0f;
    for (int i = 0; i < N; ++i) {
        float err = std::fabs(values[i] - expected[i]);
        if (err > max_err) max_err = err;
    }
    std::cout << "  Max absolute error: " << max_err;
    if (max_err > 1e-4f) {
        std::cout << " ❌ FAIL";
    } else {
        std::cout << " ✅ PASS";
    }
    std::cout << std::endl;

    // Test pow10Array
    std::cout << "\npow10Array:" << std::endl;
    for (int i = 0; i < N; ++i) {
        values[i] = -3.0f + (i * 3.0f / N);  // Range: -3 to 0
        expected[i] = MMath::pow10(values[i]);
    }

    MMath::pow10Array(values, N);

    max_err = 0.0f;
    for (int i = 0; i < N; ++i) {
        float err = std::fabs(values[i] - expected[i]);
        if (err > max_err) max_err = err;
    }
    std::cout << "  Max absolute error: " << max_err;
    if (max_err > 1e-3f) {
        std::cout << " ❌ FAIL (expected[0]=" << expected[0] << " got=" << values[0] << ")";
    } else {
        std::cout << " ✅ PASS";
    }
    std::cout << std::endl;
}

int main() {
    test_log_edge_cases();
    test_log2_consistency();
    test_log10_consistency();
    test_array_consistency();
    return 0;
}
