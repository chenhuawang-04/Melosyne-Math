// Debug powArray failure
#include "include/fast_math/power.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

int main() {
    constexpr float kPowEpsilon = 5e-5f;

    const int size = 64;
    std::vector<float> base(size);
    std::vector<float> exponent(size);
    std::vector<float> result(size);
    std::vector<float> expected(size);

    // Fill with test values
    for (int i = 0; i < size; ++i) {
        base[i] = 0.1f + (i * 10.0f / size);  // Range: [0.1, 10]
        exponent[i] = -2.0f + (i * 4.0f / size);  // Range: [-2, 2]
        expected[i] = MMath::pow(base[i], exponent[i]);
    }

    // SIMD version
    MMath::powArray(base.data(), exponent.data(), result.data(), size);

    // Compare
    std::cout << std::setprecision(10);
    std::cout << "Checking powArray consistency..." << std::endl;

    int failures = 0;
    for (int i = 0; i < size; ++i) {
        float max_val = std::max(std::fabs(result[i]), std::fabs(expected[i]));
        float rel_err = (max_val < 1e-30f) ? 0.0f : std::fabs(result[i] - expected[i]) / max_val;

        if (rel_err > kPowEpsilon) {
            std::cout << "[" << i << "] pow(" << base[i] << ", " << exponent[i] << "): "
                      << "SIMD=" << result[i] << " scalar=" << expected[i]
                      << " rel_err=" << rel_err << " ❌ FAIL" << std::endl;
            failures++;
        }
    }

    if (failures == 0) {
        std::cout << "All tests passed! ✅" << std::endl;
    } else {
        std::cout << "Failures: " << failures << "/" << size << std::endl;
    }

    return (failures == 0) ? 0 : 1;
}
