#include "test/framework/test_framework.h"
#include "test/framework/test_utils.h"

#include "fast_math/common.h"
#include "fast_math/lerp.h"

#include <algorithm>
#include <cmath>
#include <vector>

FM_TEST(Common, ExactCases) {
    FM_REQUIRE_NEAR(MMath::min(2.0f, -3.0f), -3.0f, 1e-6f);
    FM_REQUIRE_NEAR(MMath::max(2.0f, -3.0f), 2.0f, 1e-6f);
    FM_REQUIRE_NEAR(MMath::clamp(5.0f, -1.0f, 1.0f), 1.0f, 1e-6f);
    FM_REQUIRE_NEAR(MMath::clamp(-5.0f, -1.0f, 1.0f), -1.0f, 1e-6f);
    FM_REQUIRE_NEAR(MMath::abs(-42.5f), 42.5f, 1e-6f);
    FM_REQUIRE_NEAR(MMath::sign(-10.0f), -1.0f, 1e-6f);
    FM_REQUIRE_NEAR(MMath::sign(0.0f), 0.0f, 1e-6f);
    FM_REQUIRE_NEAR(MMath::sign(10.0f), 1.0f, 1e-6f);
    FM_REQUIRE_NEAR(MMath::saturate(-0.5f), 0.0f, 1e-6f);
    FM_REQUIRE_NEAR(MMath::saturate(1.5f), 1.0f, 1e-6f);
    FM_REQUIRE_NEAR(MMath::copySign(2.0f, -1.0f), -2.0f, 1e-6f);

    FM_REQUIRE(MMath::nearEqual(1.0f, 1.0f + 1e-6f, 1e-5f));
    FM_REQUIRE(!MMath::nearEqual(1.0f, 1.2f, 1e-3f));
    FM_REQUIRE(MMath::isZero(1e-7f, 1e-6f));
    FM_REQUIRE(!MMath::isZero(1e-2f, 1e-6f));
}

FM_TEST(Common, RandomAgainstReference) {
    FmTest::Rng rng(0x12345678U);

    for (int i = 0; i < 5000; ++i) {
        const float a = rng.uniform(-200.0f, 200.0f);
        const float b = rng.uniform(-200.0f, 200.0f);
        const float lo = std::min(a, b);
        const float hi = std::max(a, b);
        const float x = rng.uniform(-300.0f, 300.0f);

        FM_REQUIRE_NEAR(MMath::min(a, b), std::min(a, b), 1e-6f);
        FM_REQUIRE_NEAR(MMath::max(a, b), std::max(a, b), 1e-6f);
        FM_REQUIRE_NEAR(MMath::clamp(x, lo, hi), std::clamp(x, lo, hi), 1e-6f);
        FM_REQUIRE_NEAR(MMath::abs(x), std::fabs(x), 1e-6f);
        FM_REQUIRE_NEAR(MMath::copySign(a, b), std::copysign(a, b), 1e-6f);

        const float clamped = MMath::clamp(x, lo, hi);
        FM_REQUIRE(clamped >= lo - 1e-6f && clamped <= hi + 1e-6f);
    }
}

FM_TEST(Common, ArrayOpsConsistency) {
    FmTest::Rng rng(0xBADC0DEU);

    for (int n : {1, 3, 4, 7, 8, 15, 16, 33, 64, 97}) {
        std::vector<float> a(n), b(n), expected(n);
        for (int i = 0; i < n; ++i) {
            a[i] = rng.uniform(-20.0f, 20.0f);
            b[i] = rng.uniform(-20.0f, 20.0f);
        }

        expected = a;
        for (int i = 0; i < n; ++i) expected[i] = std::min(expected[i], b[i]);
        MMath::minArray(a.data(), b.data(), n);
        for (int i = 0; i < n; ++i) FM_REQUIRE_NEAR(a[i], expected[i], 1e-6f);

        for (int i = 0; i < n; ++i) {
            a[i] = rng.uniform(-20.0f, 20.0f);
            expected[i] = std::clamp(a[i], -3.5f, 2.25f);
        }
        MMath::clampArray(a.data(), -3.5f, 2.25f, n);
        for (int i = 0; i < n; ++i) FM_REQUIRE_NEAR(a[i], expected[i], 1e-6f);

        for (int i = 0; i < n; ++i) {
            a[i] = rng.uniform(-2.0f, 2.0f);
            expected[i] = std::clamp(a[i], 0.0f, 1.0f);
        }
        MMath::saturateArray(a.data(), n);
        for (int i = 0; i < n; ++i) FM_REQUIRE_NEAR(a[i], expected[i], 1e-6f);
    }
}

FM_TEST(Lerp, ScalarRoundTripAndRanges) {
    FmTest::Rng rng(0xDEADBEEFU);

    for (int i = 0; i < 6000; ++i) {
        const float a = rng.uniform(-50.0f, 50.0f);
        const float b = rng.uniform(-50.0f, 50.0f);
        const float t = rng.uniform(-2.0f, 2.0f);

        const float v = MMath::lerp(a, b, t);
        FM_REQUIRE_NEAR(v, a + t * (b - a), 1e-5f);

        if (std::fabs(b - a) > 1e-4f) {
            const float inv = MMath::inverseLerp(a, b, v);
            FM_REQUIRE_NEAR(inv, t, 2e-4f);
        }

        const float remapped = MMath::remap(v, a, b, -1.0f, 1.0f);
        const float ref = -1.0f + (v - a) / (b - a) * 2.0f;
        if (std::fabs(b - a) > 1e-4f) {
            FM_REQUIRE_NEAR(remapped, ref, 3e-4f);
        }

        const float lc = MMath::lerpClamped(a, b, t);
        const float lo = std::min(a, b);
        const float hi = std::max(a, b);
        FM_REQUIRE(lc >= lo - 1e-5f && lc <= hi + 1e-5f);

        const float s = MMath::lerpStepped(0.0f, 10.0f, t, 11);
        FM_REQUIRE(s >= -1e-4f && s <= 10.0f + 1e-4f);
    }
}

FM_TEST(Lerp, SmoothFunctionsMonotonicity) {
    float prev_smooth = -1.0f;
    float prev_smoother = -1.0f;

    for (int i = 0; i <= 1000; ++i) {
        const float t = static_cast<float>(i) / 1000.0f;
        const float s1 = MMath::smoothstep(0.0f, 1.0f, t);
        const float s2 = MMath::smootherstep(0.0f, 1.0f, t);

        FM_REQUIRE(s1 >= prev_smooth - 1e-6f);
        FM_REQUIRE(s2 >= prev_smoother - 1e-6f);
        FM_REQUIRE(s1 >= -1e-6f && s1 <= 1.0f + 1e-6f);
        FM_REQUIRE(s2 >= -1e-6f && s2 <= 1.0f + 1e-6f);

        prev_smooth = s1;
        prev_smoother = s2;
    }

    FM_REQUIRE_NEAR(MMath::smoothstep(0.0f, 1.0f, 0.0f), 0.0f, 1e-6f);
    FM_REQUIRE_NEAR(MMath::smoothstep(0.0f, 1.0f, 1.0f), 1.0f, 1e-6f);
    FM_REQUIRE_NEAR(MMath::smootherstep(0.0f, 1.0f, 0.0f), 0.0f, 1e-6f);
    FM_REQUIRE_NEAR(MMath::smootherstep(0.0f, 1.0f, 1.0f), 1.0f, 1e-6f);
}

FM_TEST(Lerp, ArrayOpsConsistency) {
    FmTest::Rng rng(0x77889911U);

    for (int n : {1, 5, 8, 17, 32, 65, 128}) {
        std::vector<float> a(n), b(n), out(n), ref(n);
        for (int i = 0; i < n; ++i) {
            a[i] = rng.uniform(-10.0f, 10.0f);
            b[i] = rng.uniform(-10.0f, 10.0f);
            ref[i] = MMath::lerp(a[i], b[i], 0.37f);
        }

        MMath::lerpArray(a.data(), b.data(), 0.37f, out.data(), n);
        for (int i = 0; i < n; ++i) FM_REQUIRE_NEAR(out[i], ref[i], 1e-5f);

        for (int i = 0; i < n; ++i) {
            out[i] = rng.uniform(-2.0f, 2.0f);
            ref[i] = MMath::smoothstep(-1.0f, 1.0f, out[i]);
        }
        MMath::smoothstepArray(-1.0f, 1.0f, out.data(), n);
        for (int i = 0; i < n; ++i) FM_REQUIRE_NEAR(out[i], ref[i], 2e-5f);

        for (int i = 0; i < n; ++i) {
            a[i] = rng.uniform(-5.0f, 5.0f);
            ref[i] = MMath::remap(a[i], -5.0f, 5.0f, 0.0f, 1.0f);
        }
        MMath::remapArray(a.data(), -5.0f, 5.0f, 0.0f, 1.0f, out.data(), n);
        for (int i = 0; i < n; ++i) FM_REQUIRE_NEAR(out[i], ref[i], 1e-5f);
    }
}
