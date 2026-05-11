#include "test/framework/test_framework.h"
#include "test/framework/test_utils.h"

#include "fast_math/vec2.h"
#include "fast_math/vec3.h"
#include "fast_math/vec3_simd.h"
#include "fast_math/vec4.h"

#include <cmath>
#include <vector>

FM_TEST(Vec2, ScalarAndArrayOps) {
    fmtest::Rng rng(0x42424242U);

    for (int i = 0; i < 4000; ++i) {
        MMath::Vec2 a{rng.uniform(-10.0f, 10.0f), rng.uniform(-10.0f, 10.0f)};
        MMath::Vec2 b{rng.uniform(-10.0f, 10.0f), rng.uniform(-10.0f, 10.0f)};

        MMath::Vec2 add = MMath::vec2Add(a, b);
        FM_REQUIRE_VEC2_NEAR(add, (MMath::Vec2{a.x + b.x, a.y + b.y}), 1e-6f);

        MMath::Vec2 sub = MMath::vec2Sub(a, b);
        FM_REQUIRE_VEC2_NEAR(sub, (MMath::Vec2{a.x - b.x, a.y - b.y}), 1e-6f);

        FM_REQUIRE_NEAR(MMath::vec2Dot(a, b), a.x * b.x + a.y * b.y, 1e-5f);

        const float len = MMath::vec2Length(a);
        FM_REQUIRE_NEAR(len * len, MMath::vec2LengthSquared(a), 1e-4f);

        if (MMath::vec2LengthSquared(a) > 1e-4f) {
            MMath::Vec2 n = MMath::vec2Normalize(a);
            FM_REQUIRE_NEAR(MMath::vec2Length(n), 1.0f, 1e-4f);
        }

        MMath::Vec2 p = MMath::vec2Project(a, b);
        MMath::Vec2 r = MMath::vec2Sub(a, p);
        FM_REQUIRE_NEAR(MMath::vec2Dot(r, b), 0.0f, 3e-3f);
    }

    for (int n : {1, 3, 8, 17, 33, 64, 130}) {
        std::vector<MMath::Vec2> a(n), b(n), out(n);
        std::vector<float> dots(n), ref_dots(n);

        for (int i = 0; i < n; ++i) {
            a[i] = MMath::Vec2{rng.uniform(-5.0f, 5.0f), rng.uniform(-5.0f, 5.0f)};
            b[i] = MMath::Vec2{rng.uniform(-5.0f, 5.0f), rng.uniform(-5.0f, 5.0f)};
        }

        MMath::vec2AddArray(a.data(), b.data(), out.data(), n);
        for (int i = 0; i < n; ++i) {
            FM_REQUIRE_VEC2_NEAR(out[i], MMath::vec2Add(a[i], b[i]), 1e-6f);
        }

        MMath::vec2DotArray(a.data(), b.data(), dots.data(), n);
        for (int i = 0; i < n; ++i) {
            ref_dots[i] = MMath::vec2Dot(a[i], b[i]);
            FM_REQUIRE_NEAR(dots[i], ref_dots[i], 1e-4f);
        }
    }
}

FM_TEST(Vec3, GeometryInvariants) {
    fmtest::Rng rng(0xF00DFACEU);

    for (int i = 0; i < 6000; ++i) {
        const MMath::Vec3 a = fmtest::random_vec3(rng, -10.0f, 10.0f);
        const MMath::Vec3 b = fmtest::random_vec3(rng, -10.0f, 10.0f);

        const MMath::Vec3 c = MMath::vec3Cross(a, b);
        FM_REQUIRE_NEAR(MMath::vec3Dot(c, a), 0.0f, 3e-3f);
        FM_REQUIRE_NEAR(MMath::vec3Dot(c, b), 0.0f, 3e-3f);

        const MMath::Vec3 p = MMath::vec3Project(a, b);
        const MMath::Vec3 r = MMath::vec3Reject(a, b);
        FM_REQUIRE_VEC3_NEAR(MMath::vec3Add(p, r), a, 2e-3f);

        if (MMath::vec3LengthSquared(a) > 1e-4f) {
            const MMath::Vec3 n = MMath::vec3Normalize(a);
            FM_REQUIRE_NEAR(MMath::vec3Length(n), 1.0f, 2e-4f);
            FM_REQUIRE(MMath::vec3IsNormalized(n));

            const MMath::Vec3 nf = MMath::vec3NormalizeFast(a);
            FM_REQUIRE_NEAR(MMath::vec3Length(nf), 1.0f, 2e-3f);
        }

        MMath::Vec3 t{}, bit{};
        const MMath::Vec3 dir = MMath::vec3Add(a, MMath::Vec3{0.7f, -0.2f, 0.4f});
        if (MMath::vec3LengthSquared(dir) > 1e-4f) {
            MMath::vec3OrthonormalBasis(dir, t, bit);
            FM_REQUIRE_NEAR(MMath::vec3Dot(t, bit), 0.0f, 3e-3f);
            FM_REQUIRE_NEAR(MMath::vec3Length(t), 1.0f, 3e-3f);
            FM_REQUIRE_NEAR(MMath::vec3Length(bit), 1.0f, 3e-3f);
        }
    }
}

FM_TEST(Vec3, BatchSimdConsistency) {
    fmtest::Rng rng(0x1BADB002U);

    for (int n : {1, 3, 4, 7, 16, 31, 64, 129}) {
        std::vector<MMath::Vec3> a(n), b(n), out(n), ref(n);
        std::vector<float> out_f(n), ref_f(n);

        for (int i = 0; i < n; ++i) {
            a[i] = fmtest::random_vec3(rng, -4.0f, 4.0f);
            b[i] = fmtest::random_vec3(rng, -4.0f, 4.0f);
        }

        MMath::vec3AddBatchSimd(a.data(), b.data(), out.data(), n);
        for (int i = 0; i < n; ++i) {
            ref[i] = MMath::vec3Add(a[i], b[i]);
            FM_REQUIRE_VEC3_NEAR(out[i], ref[i], 1e-5f);
        }

        MMath::vec3DotBatchSimd(a.data(), b.data(), out_f.data(), n);
        for (int i = 0; i < n; ++i) {
            ref_f[i] = MMath::vec3Dot(a[i], b[i]);
            FM_REQUIRE_NEAR(out_f[i], ref_f[i], 1e-4f);
        }

        MMath::vec3CrossBatchSimd(a.data(), b.data(), out.data(), n);
        for (int i = 0; i < n; ++i) {
            ref[i] = MMath::vec3Cross(a[i], b[i]);
            FM_REQUIRE_VEC3_NEAR(out[i], ref[i], 1e-4f);
        }

        for (int i = 0; i < n; ++i) {
            a[i] = fmtest::random_vec3(rng, 0.1f, 2.0f);
        }
        MMath::vec3ScaleBatchSimd(a.data(), 1.75f, out.data(), n);
        for (int i = 0; i < n; ++i) {
            ref[i] = MMath::vec3Scale(a[i], 1.75f);
            FM_REQUIRE_VEC3_NEAR(out[i], ref[i], 1e-5f);
        }
    }
}

FM_TEST(Vec4, CoreOpsAndNormalization) {
    fmtest::Rng rng(0x10203040U);

    for (int i = 0; i < 5000; ++i) {
        const MMath::Vec4 a = fmtest::random_vec4(rng, -8.0f, 8.0f);
        const MMath::Vec4 b = fmtest::random_vec4(rng, -8.0f, 8.0f);

        FM_REQUIRE_VEC4_NEAR(
            MMath::vec4Add(a, b),
            (MMath::Vec4{a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w}),
            1e-6f);
        FM_REQUIRE_VEC4_NEAR(
            MMath::vec4Sub(a, b),
            (MMath::Vec4{a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w}),
            1e-6f);

        const float dot = MMath::vec4Dot(a, b);
        FM_REQUIRE_NEAR(dot, a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w, 1e-4f);

        const float len = MMath::vec4Length(a);
        FM_REQUIRE_NEAR(len * len, MMath::vec4LengthSquared(a), 3e-4f);

        if (MMath::vec4LengthSquared(a) > 1e-4f) {
            const MMath::Vec4 n = MMath::vec4Normalize(a);
            FM_REQUIRE_NEAR(MMath::vec4Length(n), 1.0f, 3e-4f);
            FM_REQUIRE(MMath::vec4NearEqual(MMath::vec4NormalizeSafe(a), n, 1e-3f));
        }

        const MMath::Vec4 clamped = MMath::vec4Clamp(a, MMath::vec4Splat(-1.0f), MMath::vec4Splat(1.0f));
        FM_REQUIRE(clamped.x >= -1.0f && clamped.x <= 1.0f);
        FM_REQUIRE(clamped.y >= -1.0f && clamped.y <= 1.0f);
        FM_REQUIRE(clamped.z >= -1.0f && clamped.z <= 1.0f);
        FM_REQUIRE(clamped.w >= -1.0f && clamped.w <= 1.0f);
    }

    FM_REQUIRE(MMath::vec4Equal(MMath::vec4Zero(), MMath::Vec4{0, 0, 0, 0}));
    FM_REQUIRE(MMath::vec4Equal(MMath::vec4UnitX(), MMath::Vec4{1, 0, 0, 0}));
    FM_REQUIRE(MMath::vec4Equal(MMath::vec4UnitY(), MMath::Vec4{0, 1, 0, 0}));
    FM_REQUIRE(MMath::vec4Equal(MMath::vec4UnitZ(), MMath::Vec4{0, 0, 1, 0}));
    FM_REQUIRE(MMath::vec4Equal(MMath::vec4UnitW(), MMath::Vec4{0, 0, 0, 1}));
}
