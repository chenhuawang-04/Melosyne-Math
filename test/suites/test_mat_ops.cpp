#include "test/framework/test_framework.h"
#include "test/framework/test_utils.h"

#include "fast_math/mat3.h"
#include "fast_math/mat4.h"

#include <cmath>

namespace {

float ndc_depth(const MMath::Vec4& clip) {
    return clip.z / clip.w;
}

} // namespace

FM_TEST(Mat3, CoreOperationsAndInverse) {
    const MMath::Mat3 id = MMath::mat3Identity();
    const MMath::Mat3 zero = MMath::mat3Zero();

    FM_REQUIRE(MMath::mat3Equal(MMath::mat3Multiply(id, id), id));
    FM_REQUIRE(MMath::mat3Equal(MMath::mat3Multiply(zero, id), zero));

    MMath::Vec2 t{2.0f, -3.0f};
    MMath::Vec2 s{1.5f, 0.75f};
    const float r = 0.37f;

    MMath::Mat3 trs = MMath::mat3FromTrs(t, r, s);
    const auto inv_checked = MMath::mat3InverseAffineChecked(trs);
    FM_REQUIRE(inv_checked.invertible);
    MMath::Mat3 inv = inv_checked.value;
    MMath::Mat3 should_id = MMath::mat3MultiplyAffine(trs, inv);

    FM_REQUIRE_MAT3_NEAR(should_id, id, 3e-4f);

    MMath::Vec2 p{3.5f, -1.25f};
    MMath::Vec2 tp = MMath::mat3TransformPoint(trs, p);
    MMath::Vec2 back = MMath::mat3TransformPoint(inv, tp);
    FM_REQUIRE_VEC2_NEAR(back, p, 3e-4f);

    MMath::Mat3 tr = MMath::mat3Transpose(trs);
    MMath::Mat3 tt = MMath::mat3Transpose(tr);
    FM_REQUIRE_MAT3_NEAR(tt, trs, 1e-6f);
}

FM_TEST(Mat3, BatchInterfaces) {
    fmtest::Rng rng(0x44556677U);

    constexpr int n = 64;
    MMath::Vec2 translations[n];
    float rotations[n];
    MMath::Vec2 scales[n];
    MMath::Mat3 out[n];
    MMath::Mat3 ref[n];

    for (int i = 0; i < n; ++i) {
        translations[i] = MMath::Vec2{rng.uniform(-5.0f, 5.0f), rng.uniform(-5.0f, 5.0f)};
        rotations[i] = rng.uniform(-3.14f, 3.14f);
        scales[i] = MMath::Vec2{rng.uniform(0.5f, 2.0f), rng.uniform(0.5f, 2.0f)};
        ref[i] = MMath::mat3FromTrs(translations[i], rotations[i], scales[i]);
    }

    MMath::mat3FromTrsArray(translations, rotations, scales, out, n);
    for (int i = 0; i < n; ++i) {
        FM_REQUIRE_MAT3_NEAR(out[i], ref[i], 1e-4f);
    }
}

FM_TEST(Mat3, SingularInverseReportsFailure) {
    const MMath::Mat3 singular{
        1.0f, 2.0f, 3.0f,
        2.0f, 4.0f, 6.0f,
        0.0f, 0.0f, 0.0f
    };

    MMath::Mat3 out = MMath::mat3Zero();
    FM_REQUIRE(!MMath::mat3TryInverse(singular, &out));
    FM_REQUIRE(!MMath::mat3InverseChecked(singular).invertible);

    const MMath::Mat3 affine_singular{
        1.0f, 2.0f, 3.0f,
        2.0f, 4.0f, 5.0f,
        0.0f, 0.0f, 1.0f
    };
    FM_REQUIRE(!MMath::mat3TryInverseAffine(affine_singular, &out));
    FM_REQUIRE(!MMath::mat3InverseAffineChecked(affine_singular).invertible);
}

FM_TEST(Mat4, CoreAlgebraAndTransforms) {
    fmtest::Rng rng(0x76543210U);

    const MMath::Mat4 id = MMath::mat4Identity();
    FM_REQUIRE_MAT4_NEAR(MMath::mat4Multiply(id, id), id, 1e-6f);

    for (int i = 0; i < 2000; ++i) {
        MMath::Vec4 v = fmtest::random_vec4(rng, -5.0f, 5.0f);
        MMath::Mat4 t = MMath::mat4Translation(
            rng.uniform(-4.0f, 4.0f),
            rng.uniform(-4.0f, 4.0f),
            rng.uniform(-4.0f, 4.0f));
        MMath::Mat4 s = MMath::mat4Scale(
            rng.uniform(0.5f, 2.0f),
            rng.uniform(0.5f, 2.0f),
            rng.uniform(0.5f, 2.0f));
        MMath::Mat4 r = MMath::mat4RotationZ(rng.uniform(-3.14f, 3.14f));

        MMath::Mat4 m = MMath::mat4Multiply(MMath::mat4Multiply(t, r), s);
        const auto inv_checked = MMath::mat4InverseChecked(m);
        FM_REQUIRE(inv_checked.invertible);
        MMath::Mat4 inv = inv_checked.value;
        MMath::Mat4 should_id = MMath::mat4Multiply(m, inv);
        FM_REQUIRE_MAT4_NEAR(should_id, id, 2e-2f);

        MMath::Vec4 mv = MMath::mat4MultiplyVec4(m, v);
        MMath::Vec4 back = MMath::mat4MultiplyVec4(inv, mv);
        FM_REQUIRE_VEC4_NEAR(back, v, 2e-2f);

        MMath::Mat4 tt = MMath::mat4Transpose(MMath::mat4Transpose(m));
        FM_REQUIRE_MAT4_NEAR(tt, m, 1e-5f);
    }
}

FM_TEST(Mat4, ExplicitConventionHelpersAgreeWithCanonicalDefaults) {
    const MMath::Vec4 eye{0.0f, 1.0f, 5.0f, 1.0f};
    const MMath::Vec4 target{0.0f, 0.0f, 0.0f, 1.0f};
    const MMath::Vec4 up{0.0f, 1.0f, 0.0f, 0.0f};

    FM_REQUIRE_MAT4_NEAR(
        MMath::mat4Perspective(1.1f, 16.0f / 9.0f, 0.1f, 250.0f),
        MMath::mat4PerspectiveRH(1.1f, 16.0f / 9.0f, 0.1f, 250.0f),
        1e-6f);
    FM_REQUIRE_MAT4_NEAR(
        MMath::mat4Ortho(-2.0f, 2.0f, -1.5f, 1.5f, 0.1f, 250.0f),
        MMath::mat4OrthoRH(-2.0f, 2.0f, -1.5f, 1.5f, 0.1f, 250.0f),
        1e-6f);
    FM_REQUIRE_MAT4_NEAR(
        MMath::mat4LookAt(eye, target, up),
        MMath::mat4LookAtRH(eye, target, up),
        1e-6f);
}

FM_TEST(Mat4, ViewAndProjectionSanity) {
    const MMath::Vec4 eye{0.0f, 0.0f, 5.0f, 1.0f};
    const MMath::Vec4 target{0.0f, 0.0f, 0.0f, 1.0f};
    const MMath::Vec4 up{0.0f, 1.0f, 0.0f, 0.0f};

    const MMath::Mat4 view = MMath::mat4LookAtRH(eye, target, up);
    const MMath::Vec4 eye_view = MMath::mat4MultiplyVec4(view, eye);

    FM_REQUIRE_NEAR(eye_view.x, 0.0f, 2e-3f);
    FM_REQUIRE_NEAR(eye_view.y, 0.0f, 2e-3f);
    FM_REQUIRE_NEAR(eye_view.z, 0.0f, 2e-2f);

    const MMath::Mat4 proj = MMath::mat4PerspectiveRH(3.1415926f / 3.0f, 16.0f / 9.0f, 0.1f, 100.0f);
    FM_REQUIRE(std::isfinite(proj.m[0]));
    FM_REQUIRE(std::isfinite(proj.m[5]));
    FM_REQUIRE(std::isfinite(proj.m[10]));
}

FM_TEST(Mat4, SingularInverseReportsFailure) {
    const MMath::Mat4 singular{{
        1.0f, 2.0f, 3.0f, 4.0f,
        2.0f, 4.0f, 6.0f, 8.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 1.0f
    }};

    MMath::Mat4 out = MMath::mat4Identity();
    FM_REQUIRE(!MMath::mat4TryInverse(singular, &out));
    FM_REQUIRE(!MMath::mat4InverseChecked(singular).invertible);
}

FM_TEST(Mat4, DepthRangeMappingZeroToOne) {
    constexpr float kNear = 0.1f;
    constexpr float kFar = 100.0f;
    constexpr float kFovY = 3.1415926f / 3.0f;
    constexpr float kAspect = 16.0f / 9.0f;

    const MMath::Mat4 proj_rh = MMath::mat4PerspectiveRH(kFovY, kAspect, kNear, kFar);
    const MMath::Mat4 proj_lh = MMath::mat4PerspectiveLH(kFovY, kAspect, kNear, kFar);

    const MMath::Vec4 rh_near_view{0.0f, 0.0f, -kNear, 1.0f};
    const MMath::Vec4 rh_far_view{0.0f, 0.0f, -kFar, 1.0f};
    const MMath::Vec4 lh_near_view{0.0f, 0.0f, kNear, 1.0f};
    const MMath::Vec4 lh_far_view{0.0f, 0.0f, kFar, 1.0f};

    FM_REQUIRE_NEAR(ndc_depth(MMath::mat4MultiplyVec4(proj_rh, rh_near_view)), 0.0f, 1e-4f);
    FM_REQUIRE_NEAR(ndc_depth(MMath::mat4MultiplyVec4(proj_rh, rh_far_view)), 1.0f, 1e-4f);
    FM_REQUIRE_NEAR(ndc_depth(MMath::mat4MultiplyVec4(proj_lh, lh_near_view)), 0.0f, 1e-4f);
    FM_REQUIRE_NEAR(ndc_depth(MMath::mat4MultiplyVec4(proj_lh, lh_far_view)), 1.0f, 1e-4f);

    const MMath::Mat4 ortho_rh = MMath::mat4OrthoRH(-2.0f, 2.0f, -1.0f, 1.0f, kNear, kFar);
    const MMath::Mat4 ortho_lh = MMath::mat4OrthoLH(-2.0f, 2.0f, -1.0f, 1.0f, kNear, kFar);
    FM_REQUIRE_NEAR(ndc_depth(MMath::mat4MultiplyVec4(ortho_rh, rh_near_view)), 0.0f, 1e-5f);
    FM_REQUIRE_NEAR(ndc_depth(MMath::mat4MultiplyVec4(ortho_rh, rh_far_view)), 1.0f, 1e-5f);
    FM_REQUIRE_NEAR(ndc_depth(MMath::mat4MultiplyVec4(ortho_lh, lh_near_view)), 0.0f, 1e-5f);
    FM_REQUIRE_NEAR(ndc_depth(MMath::mat4MultiplyVec4(ortho_lh, lh_far_view)), 1.0f, 1e-5f);
}
