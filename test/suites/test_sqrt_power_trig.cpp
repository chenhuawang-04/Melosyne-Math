#include "test/framework/test_framework.h"
#include "test/framework/test_utils.h"

#include "fast_math/power.h"
#include "fast_math/sqrt.h"
#include "fast_math/trig.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

FM_TEST(Sqrt, ScalarVsReference) {
    fmtest::Rng rng(0x7A11B00BU);

    for (int i = 0; i < 8000; ++i) {
        const float x = rng.uniform(1e-4f, 5e4f);
        const float s = MMath::sqrt(x);
        const float inv = MMath::inverseSqrt(x);
        const float rcp = MMath::reciprocal(x);

        FM_REQUIRE_NEAR(s, std::sqrt(x), 1e-4f);
        FM_REQUIRE_NEAR(inv, 1.0f / std::sqrt(x), 2e-3f);
        FM_REQUIRE_NEAR(rcp, 1.0f / x, 2e-3f);

        FM_REQUIRE_NEAR(s * inv, 1.0f, 2e-3f);
        FM_REQUIRE_NEAR(x * rcp, 1.0f, 2e-3f);

        FM_REQUIRE(MMath::isSafeForDivision(x));
        FM_REQUIRE_NEAR(MMath::normalizeFactor(x), inv, 2e-6f);
    }

    FM_REQUIRE_NEAR(MMath::safeReciprocal(0.0f, 123.0f), 123.0f, 1e-6f);
    FM_REQUIRE_NEAR(MMath::safeReciprocal(1e-12f, 7.0f), 7.0f, 1e-6f);
}

FM_TEST(Sqrt, ArrayOpsConsistency) {
    fmtest::Rng rng(0xABCD1234U);

    for (int n : {1, 3, 8, 17, 31, 64, 129}) {
        std::vector<float> values(n), ref(n);

        for (int i = 0; i < n; ++i) {
            values[i] = rng.uniform(1e-3f, 2e3f);
            ref[i] = std::sqrt(values[i]);
        }
        MMath::sqrtArray(values.data(), n);
        for (int i = 0; i < n; ++i) FM_REQUIRE_NEAR(values[i], ref[i], 1e-4f);

        for (int i = 0; i < n; ++i) {
            values[i] = rng.uniform(1e-3f, 2e3f);
            ref[i] = 1.0f / std::sqrt(values[i]);
        }
        MMath::inverseSqrtArray(values.data(), n);
        for (int i = 0; i < n; ++i) FM_REQUIRE_NEAR(values[i], ref[i], 2e-3f);

        for (int i = 0; i < n; ++i) {
            values[i] = rng.uniform(1e-3f, 2e3f);
            ref[i] = 1.0f / values[i];
        }
        MMath::reciprocalArray(values.data(), n);
        for (int i = 0; i < n; ++i) FM_REQUIRE_NEAR(values[i], ref[i], 2e-3f);
    }
}

FM_TEST(Power, ScalarVsReference) {
    fmtest::Rng rng(0xCAFEBABEU);

    for (int i = 0; i < 5000; ++i) {
        const float x = rng.uniform(0.05f, 20.0f);
        const float y = rng.uniform(-3.0f, 3.0f);
        const float t = rng.uniform(-4.0f, 4.0f);

        FM_REQUIRE(fmtest::near_rel(MMath::exp(t), std::exp(t), 2e-2f, 3e-2f));
        FM_REQUIRE_NEAR(MMath::exp2(t), std::exp2(t), 8e-3f);
        FM_REQUIRE(fmtest::near_rel(MMath::pow10(t), std::pow(10.0f, t), 2e-2f, 2e-2f));

        FM_REQUIRE(fmtest::near_rel(MMath::log(x), std::log(x), 2e-2f, 3e-2f));
        FM_REQUIRE(fmtest::near_rel(MMath::log2(x), std::log2(x), 2e-2f, 3e-2f));
        FM_REQUIRE(fmtest::near_rel(MMath::log10(x), std::log10(x), 2e-2f, 3e-2f));

        FM_REQUIRE(fmtest::near_rel(MMath::pow(x, y), std::pow(x, y), 5e-2f, 2e-2f));

        const int n = rng.uniform_int(-8, 8);
        FM_REQUIRE(fmtest::near_rel(MMath::powi(x, n), std::pow(x, static_cast<float>(n)), 2e-5f, 2e-3f));
    }

#if !defined(__FAST_MATH__)
    const float neg_pow = MMath::pow(-1.0f, 0.5f);
    uint32_t bits = 0;
    std::memcpy(&bits, &neg_pow, sizeof(bits));
    const bool is_nan_bits = ((bits & 0x7F800000u) == 0x7F800000u) && ((bits & 0x007FFFFFu) != 0u);
    FM_REQUIRE(is_nan_bits);
#else
    // With -ffast-math, NaN propagation/observation is not guaranteed by the compiler.
    FM_REQUIRE(true);
#endif
}

FM_TEST(Power, ArrayOpsConsistency) {
    fmtest::Rng rng(0x00FEED77U);

    for (int n : {1, 7, 16, 29, 64, 121}) {
        std::vector<float> values(n), ref(n), base(n), expo(n), out(n);

        for (int i = 0; i < n; ++i) {
            values[i] = rng.uniform(-3.0f, 3.0f);
            ref[i] = std::exp(values[i]);
        }
        MMath::expArray(values.data(), n);
        for (int i = 0; i < n; ++i) FM_REQUIRE(fmtest::near_rel(values[i], ref[i], 2e-2f, 3e-2f));

        for (int i = 0; i < n; ++i) {
            values[i] = rng.uniform(0.1f, 20.0f);
            ref[i] = std::log(values[i]);
        }
        MMath::logArray(values.data(), n);
        for (int i = 0; i < n; ++i) FM_REQUIRE(fmtest::near_rel(values[i], ref[i], 2e-2f, 3e-2f));

        for (int i = 0; i < n; ++i) {
            base[i] = rng.uniform(0.2f, 10.0f);
            expo[i] = rng.uniform(-2.0f, 2.0f);
            ref[i] = std::pow(base[i], expo[i]);
        }
        MMath::powArray(base.data(), expo.data(), out.data(), n);
        for (int i = 0; i < n; ++i) FM_REQUIRE(fmtest::near_rel(out[i], ref[i], 5e-2f, 2e-2f));
    }
}

FM_TEST(Power, ArrayEdgeCaseContracts) {
    const float pos_inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();

    const float base[] = {-2.0f, -2.0f, 0.0f, -2.0f, pos_inf, 4.0f};
    const float expo[] = {4.0f, 3.0f, -1.0f, 0.5f, 2.0f, nan};
    float out[6]{};

    MMath::powArray(base, expo, out, 6);

    FM_REQUIRE_NEAR(out[0], 16.0f, 1e-5f);
    FM_REQUIRE_NEAR(out[1], -8.0f, 1e-5f);
    FM_REQUIRE(std::isinf(out[2]));
    FM_REQUIRE(std::isnan(out[3]));
    FM_REQUIRE(std::isinf(out[4]));
    FM_REQUIRE(std::isnan(out[5]));
}

FM_TEST(Power, EdgeCaseContracts) {
    FM_REQUIRE_NEAR(MMath::pow(0.0f, 3.0f), 0.0f, 1e-6f);
    FM_REQUIRE_NEAR(MMath::pow(-2.0f, 4.0f), 16.0f, 1e-5f);
    FM_REQUIRE_NEAR(MMath::pow(-2.0f, 3.0f), -8.0f, 1e-5f);

    const float zero_neg = MMath::pow(0.0f, -1.0f);
    FM_REQUIRE(std::isinf(zero_neg));

    const float neg_fractional = MMath::pow(-2.0f, 0.5f);
    FM_REQUIRE(std::isnan(neg_fractional));

    FM_REQUIRE_NEAR(MMath::powi(2.0f, std::numeric_limits<int32_t>::min()), 0.0f, 1e-6f);
    FM_REQUIRE(std::isinf(MMath::powi(0.0f, -2)));
    FM_REQUIRE_NEAR(MMath::powi(-2.0f, 5), -32.0f, 1e-5f);

    FM_REQUIRE(std::isinf(MMath::log(0.0f)));
    FM_REQUIRE(MMath::log(0.0f) < 0.0f);
    FM_REQUIRE(std::isnan(MMath::log(-1.0f)));
}

FM_TEST(Trig, ScalarAndBatchConsistency) {
    fmtest::Rng rng(0x13579BDFU);

    for (int i = 0; i < 3000; ++i) {
        const float angle = rng.uniform(-50.0f, 50.0f);
        FM_REQUIRE_NEAR(MMath::sin(angle), std::sin(angle), 3e-4f);
        FM_REQUIRE_NEAR(MMath::cos(angle), std::cos(angle), 3e-4f);

        MMath::SinCos sc{};
        MMath::sincos(MMath::Angle{angle}, &sc);
        FM_REQUIRE_NEAR(sc.sin, std::sin(angle), 3e-4f);
        FM_REQUIRE_NEAR(sc.cos, std::cos(angle), 3e-4f);
    }

    for (int n : {1, 5, 8, 17, 33, 65, 257}) {
        std::vector<float> angles(n), out_s(n), out_c(n);
        for (float& a : angles) a = rng.uniform(-100.0f, 100.0f);

        MMath::sincosArray(angles.data(), out_s.data(), out_c.data(), n);
        for (int i = 0; i < n; ++i) {
            FM_REQUIRE_NEAR(out_s[i], std::sin(angles[i]), 3e-4f);
            FM_REQUIRE_NEAR(out_c[i], std::cos(angles[i]), 3e-4f);
        }

        std::fill(out_s.begin(), out_s.end(), 0.0f);
        std::fill(out_c.begin(), out_c.end(), 0.0f);

        MMath::sinArray(angles.data(), out_s.data(), n);
        MMath::cosArray(angles.data(), out_c.data(), n);
        for (int i = 0; i < n; ++i) {
            FM_REQUIRE_NEAR(out_s[i], std::sin(angles[i]), 3e-4f);
            FM_REQUIRE_NEAR(out_c[i], std::cos(angles[i]), 3e-4f);
        }
    }
}
