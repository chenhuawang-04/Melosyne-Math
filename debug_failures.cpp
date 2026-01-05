// Debug remaining failures
#include "include/fast_math/power.h"
#include <iostream>
#include <cmath>
#include <iomanip>
#include <vector>

void test_log_large_values() {
    std::cout << "=== log() large values ===" << std::endl;
    std::cout << std::setprecision(10);

    float vals[] = {0.1f, 1.0f, 10.0f, 100.0f, 1000.0f, 10000.0f};

    for (float x : vals) {
        float mmath = MMath::log(x);
        float std_val = std::log(x);
        float rel_err = std::fabs(mmath - std_val) / std::fabs(std_val);

        std::cout << "log(" << x << "): MMath=" << mmath
                  << " std=" << std_val
                  << " rel_err=" << rel_err;

        if (rel_err > 1e-5f) std::cout << " ❌";
        else std::cout << " ✅";
        std::cout << std::endl;
    }
}

void test_exp2_log2_roundtrip() {
    std::cout << "\n=== exp2/log2 roundtrip ===" << std::endl;

    float vals[] = {0.1f, 1.0f, 2.0f, 10.0f, 100.0f};

    for (float x : vals) {
        float roundtrip = MMath::exp2(MMath::log2(x));
        float err = std::fabs(roundtrip - x) / x;

        std::cout << "exp2(log2(" << x << ")) = " << roundtrip
                  << " (expected " << x << ")"
                  << " rel_err=" << err;

        if (err > 1e-5f) std::cout << " ❌";
        else std::cout << " ✅";
        std::cout << std::endl;
    }
}

void test_powArray() {
    std::cout << "\n=== powArray consistency ===" << std::endl;

    const int N = 10;
    std::vector<float> base(N);
    std::vector<float> exp(N);
    std::vector<float> result(N);
    std::vector<float> expected(N);

    for (int i = 0; i < N; ++i) {
        base[i] = 0.5f + i * 1.0f;
        exp[i] = -1.0f + i * 0.3f;
        expected[i] = MMath::pow(base[i], exp[i]);
    }

    MMath::powArray(base.data(), exp.data(), result.data(), N);

    float max_err = 0.0f;
    for (int i = 0; i < N; ++i) {
        float err = std::fabs(result[i] - expected[i]);
        if (err > max_err) max_err = err;

        std::cout << "pow(" << base[i] << ", " << exp[i] << "): "
                  << "SIMD=" << result[i] << " scalar=" << expected[i]
                  << " err=" << err << std::endl;
    }

    std::cout << "Max error: " << max_err;
    if (max_err > 1e-4f) std::cout << " ❌";
    else std::cout << " ✅";
    std::cout << std::endl;
}

int main() {
    test_log_large_values();
    test_exp2_log2_roundtrip();
    test_powArray();
    return 0;
}
