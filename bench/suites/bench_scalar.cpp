#include "bench/framework/benchmark_framework.h"

#include "fast_math/common.h"
#include "fast_math/lerp.h"
#include "fast_math/power.h"
#include "fast_math/sqrt.h"
#include "fast_math/trig.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

std::vector<float> makeSignal(int n_, float start_) {
    std::vector<float> v(n_);
    for (int i = 0; i < n_; ++i) {
        v[i] = start_ + 0.00137f * static_cast<float>(i % 997) - 0.75f;
    }
    return v;
}

} // namespace

FM_BENCH(ScalarMath, SingleValueKernel) {
    constexpr int n = 8192;
    const std::vector<float> a = makeSignal(n, -0.2f);
    const std::vector<float> b = makeSignal(n, 0.5f);

    FmBench::runComparisonCase(
        "single-value mixed math kernel",
        static_cast<std::size_t>(n),
        {
            {"fast_math", true, [&]() {
                 double acc = 0.0;
                 for (int i = 0; i < n; ++i) {
                     const float x = a[i];
                     const float y = b[i];
                     const float v1 = MMath::clamp(MMath::lerp(x, y, 0.37f), -2.0f, 2.0f);
                     const float v2 = MMath::sqrt(MMath::abs(x) + 0.01f);
                     const float v3 = MMath::log(MMath::abs(y) + 1.01f);
                     const float v4 = MMath::sin(v1) + MMath::cos(v2);
                     acc += static_cast<double>(v1 + v2 + v3 + v4);
                 }
                 FmBench::consume(acc);
             }},
            {"std_math", true, [&]() {
                 double acc = 0.0;
                 for (int i = 0; i < n; ++i) {
                     const float x = a[i];
                     const float y = b[i];
                     const float v1 = std::clamp(x + 0.37f * (y - x), -2.0f, 2.0f);
                     const float v2 = std::sqrt(std::fabs(x) + 0.01f);
                     const float v3 = std::log(std::fabs(y) + 1.01f);
                     const float v4 = std::sin(v1) + std::cos(v2);
                     acc += static_cast<double>(v1 + v2 + v3 + v4);
                 }
                 FmBench::consume(acc);
             }},
        },
        options.config);
}

FM_BENCH(ScalarMath, BatchArrayKernel) {
    constexpr int n = 16384;
    const std::vector<float> base1 = makeSignal(n, -1.0f);
    // Keep this strictly positive to avoid domain errors in sqrt().
    const std::vector<float> base2 = makeSignal(n, 1.0f);
    const std::vector<float> angles = makeSignal(n, -3.14f);

    std::vector<float> x_fast(n), y_fast(n), sin_fast(n), cos_fast(n);
    std::vector<float> x_std(n), y_std(n), sin_std(n), cos_std(n);

    FmBench::runComparisonCase(
        "batch array kernel (clamp/min/sqrt/exp/sincos)",
        static_cast<std::size_t>(n * 5),
        {
            {"fast_math", true, [&]() {
                 x_fast = base1;
                 y_fast = base2;

                 MMath::minArray(x_fast.data(), y_fast.data(), n);
                 MMath::clampArray(x_fast.data(), -1.5f, 1.5f, n);
                 MMath::sqrtArray(y_fast.data(), n);
                 MMath::expArray(y_fast.data(), n);
                 MMath::sincosArray(angles.data(), sin_fast.data(), cos_fast.data(), n);

                 double acc = 0.0;
                 for (int i = 0; i < n; ++i) {
                     acc += x_fast[i] + y_fast[i] + sin_fast[i] + cos_fast[i];
                 }
                 FmBench::consume(acc);
             }},
            {"std_math", true, [&]() {
                 x_std = base1;
                 y_std = base2;

                 for (int i = 0; i < n; ++i) x_std[i] = std::min(x_std[i], y_std[i]);
                 for (int i = 0; i < n; ++i) x_std[i] = std::clamp(x_std[i], -1.5f, 1.5f);
                 for (int i = 0; i < n; ++i) y_std[i] = std::sqrt(y_std[i]);
                 for (int i = 0; i < n; ++i) y_std[i] = std::exp(y_std[i]);
                 for (int i = 0; i < n; ++i) {
                     sin_std[i] = std::sin(angles[i]);
                     cos_std[i] = std::cos(angles[i]);
                 }

                 double acc = 0.0;
                 for (int i = 0; i < n; ++i) {
                     acc += x_std[i] + y_std[i] + sin_std[i] + cos_std[i];
                 }
                 FmBench::consume(acc);
             }},
        },
        options.config);
}
