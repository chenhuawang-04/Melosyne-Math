#include "test/framework/test_framework.h"
#include "test/framework/test_utils.h"

#include "fast_math/mat3.h"
#include "fast_math/mat4.h"
#include "fast_math/mat4_d3d.h"

#include <cmath>

namespace {

MMath::Mat4 mat4_from_d3d(const MMath::D3D::Mat4& in) {
    MMath::Mat4 out{};
    for (int i = 0; i < 16; ++i) out.m[i] = in.m[i];
    return out;
}

MMath::D3D::Mat4 d3d_from_mat4(const MMath::Mat4& in) {
    MMath::D3D::Mat4 out{};
    for (int i = 0; i < 16; ++i) out.m[i] = in.m[i];
    return out;
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
    MMath::Mat3 inv = MMath::mat3InverseAffine(trs);
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

FM_TEST(Mat4, CoreAlgebraAndTransforms) {
    fmtest::Rng rng(0x76543210U);

    const MMath::Mat4 id = MMath::mat4Identity();
    FM_REQUIRE_MAT4_NEAR(MMath::mat4Mul(id, id), id, 1e-6f);

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

        MMath::Mat4 m = MMath::mat4Mul(MMath::mat4Mul(t, r), s);
        MMath::Mat4 inv = MMath::mat4Inverse(m);
        MMath::Mat4 should_id = MMath::mat4Mul(m, inv);
        FM_REQUIRE_MAT4_NEAR(should_id, id, 2e-2f);

        MMath::Vec4 mv = MMath::mat4MulVec4(m, v);
        MMath::Vec4 back = MMath::mat4MulVec4(inv, mv);
        FM_REQUIRE_VEC4_NEAR(back, v, 2e-2f);

        MMath::Mat4 tt = MMath::mat4Transpose(MMath::mat4Transpose(m));
        FM_REQUIRE_MAT4_NEAR(tt, m, 1e-5f);
    }
}

FM_TEST(Mat4D3D, ParityWithGenericMat4ForSharedOps) {
    fmtest::Rng rng(0x99112233U);

    for (int i = 0; i < 1500; ++i) {
        const MMath::Mat4 a = MMath::mat4Mul(
            MMath::mat4Translation(rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-2, 2)),
            MMath::mat4Scale(rng.uniform(0.5f, 2.0f), rng.uniform(0.5f, 2.0f), rng.uniform(0.5f, 2.0f)));
        const MMath::Mat4 b = MMath::mat4Mul(
            MMath::mat4RotationX(rng.uniform(-3.14f, 3.14f)),
            MMath::mat4RotationY(rng.uniform(-3.14f, 3.14f)));
        const MMath::Vec4 v = fmtest::random_vec4(rng, -3.0f, 3.0f);

        const MMath::D3D::Mat4 ad = d3d_from_mat4(a);
        const MMath::D3D::Mat4 bd = d3d_from_mat4(b);

        const MMath::Mat4 mul_fast = MMath::mat4Mul(a, b);
        const MMath::Mat4 mul_d3d = mat4_from_d3d(MMath::D3D::mat4Mul(ad, bd));
        FM_REQUIRE_MAT4_NEAR(mul_d3d, mul_fast, 1e-4f);

        const MMath::Vec4 mv_fast = MMath::mat4MulVec4(a, v);
        const MMath::Vec4 mv_d3d = MMath::D3D::mat4MulVec4(ad, v);
        FM_REQUIRE_VEC4_NEAR(mv_d3d, mv_fast, 1e-4f);

        const MMath::Mat4 tr_fast = MMath::mat4Transpose(a);
        const MMath::Mat4 tr_d3d = mat4_from_d3d(MMath::D3D::mat4Transpose(ad));
        FM_REQUIRE_MAT4_NEAR(tr_d3d, tr_fast, 1e-5f);
    }
}

FM_TEST(Mat4D3D, LookAtAndProjectionSanity) {
    const MMath::Vec4 eye{0.0f, 0.0f, 5.0f, 1.0f};
    const MMath::Vec4 target{0.0f, 0.0f, 0.0f, 1.0f};
    const MMath::Vec4 up{0.0f, 1.0f, 0.0f, 0.0f};

    const MMath::D3D::Mat4 view = MMath::D3D::mat4LookAtRH(eye, target, up);
    const MMath::Vec4 eye_view = MMath::D3D::mat4MulVec4(view, eye);

    FM_REQUIRE_NEAR(eye_view.x, 0.0f, 2e-3f);
    FM_REQUIRE_NEAR(eye_view.y, 0.0f, 2e-3f);
    FM_REQUIRE_NEAR(eye_view.z, 0.0f, 2e-2f);

    const MMath::D3D::Mat4 proj = MMath::D3D::mat4PerspectiveD3D_RH(3.1415926f / 3.0f, 16.0f / 9.0f, 0.1f, 100.0f);
    FM_REQUIRE(std::isfinite(proj.m[0]));
    FM_REQUIRE(std::isfinite(proj.m[5]));
    FM_REQUIRE(std::isfinite(proj.m[10]));
}
