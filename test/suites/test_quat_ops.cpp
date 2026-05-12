#include "test/framework/test_framework.h"
#include "test/framework/test_utils.h"

#include "fast_math/quat.h"

#include <cmath>

#if FM_HAVE_GLM
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#endif

#if FM_HAVE_EIGEN
#include <Eigen/Dense>
#include <Eigen/Geometry>
#endif

#if FM_HAVE_DIRECTXMATH
#include <DirectXMath.h>
#endif

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kHalfPi = 1.57079632679489661923f;

// ============================================================================
// Helpers
// ============================================================================

MMath::Quat randomUnitQuat(FmTest::Rng& rng_) {
    MMath::Quat q{
        rng_.uniform(-1.0f, 1.0f),
        rng_.uniform(-1.0f, 1.0f),
        rng_.uniform(-1.0f, 1.0f),
        rng_.uniform(-1.0f, 1.0f)
    };
    const float len_sq = MMath::quatLengthSquared(q);
    if (len_sq < 1e-6f) {
        return MMath::quatIdentity();
    }
    return MMath::quatNormalize(q);
}

MMath::Vec3 randomUnitVec3(FmTest::Rng& rng_) {
    MMath::Vec3 v{
        rng_.uniform(-1.0f, 1.0f),
        rng_.uniform(-1.0f, 1.0f),
        rng_.uniform(-1.0f, 1.0f)
    };
    if (MMath::vec3LengthSquared(v) < 1e-6f) {
        return MMath::Vec3{1.0f, 0.0f, 0.0f};
    }
    return MMath::vec3Normalize(v);
}

void requireQuatNearOrient(const MMath::Quat& a_, const MMath::Quat& b_, float eps_) {
    // Quaternions are orientation-equivalent under sign flip.
    FM_REQUIRE(MMath::quatNearEqual(a_, b_, eps_, /*orientation_aware=*/true));
}

} // namespace

// ============================================================================
// Tier 1: Identity / length / normalize / integrate
// ============================================================================

FM_TEST(Quat, IdentityIsUnit) {
    const MMath::Quat id = MMath::quatIdentity();
    FM_REQUIRE(id.x == 0.0f);
    FM_REQUIRE(id.y == 0.0f);
    FM_REQUIRE(id.z == 0.0f);
    FM_REQUIRE(id.w == 1.0f);
    FM_REQUIRE_NEAR(MMath::quatLengthSquared(id), 1.0f, 1e-7f);
    FM_REQUIRE_NEAR(MMath::quatLength(id), 1.0f, 1e-6f);
}

FM_TEST(Quat, NormalizeProducesUnitLength) {
    FmTest::Rng rng(0xA1B2C3D4U);
    for (int i = 0; i < 2000; ++i) {
        MMath::Quat q{
            rng.uniform(-5.0f, 5.0f),
            rng.uniform(-5.0f, 5.0f),
            rng.uniform(-5.0f, 5.0f),
            rng.uniform(-5.0f, 5.0f)
        };
        if (MMath::quatLengthSquared(q) < 1e-4f) continue;
        const MMath::Quat n = MMath::quatNormalize(q);
        FM_REQUIRE_NEAR(MMath::quatLength(n), 1.0f, 3e-4f);
    }
}

FM_TEST(Quat, NormalizeSafeReturnsIdentityForZero) {
    const MMath::Quat zero{0.0f, 0.0f, 0.0f, 0.0f};
    const MMath::Quat n = MMath::quatNormalizeSafe(zero);
    FM_REQUIRE_NEAR(n.w, 1.0f, 1e-7f);
    FM_REQUIRE_NEAR(n.x, 0.0f, 1e-7f);
    FM_REQUIRE_NEAR(n.y, 0.0f, 1e-7f);
    FM_REQUIRE_NEAR(n.z, 0.0f, 1e-7f);
}

FM_TEST(Quat, IntegrateZeroOmegaIsIdentity) {
    FmTest::Rng rng(0xCAFEF00DU);
    for (int i = 0; i < 200; ++i) {
        const MMath::Quat q = randomUnitQuat(rng);
        const MMath::Quat q2 = MMath::quatIntegrate(q, MMath::Vec3{0.0f, 0.0f, 0.0f}, 0.016f);
        requireQuatNearOrient(q2, q, 1e-5f);
    }
}

FM_TEST(Quat, IntegrateMatchesAxisAngleForConstantSpin) {
    // Spinning around the Z axis at omega rad/s for total time T should
    // be equivalent to applying quatFromAxisAngle(Z, omega*T) on top of
    // the initial orientation, within first-order integration error.
    const MMath::Vec3 omega{0.0f, 0.0f, 1.0f};   // rad/s
    const float dt = 1.0f / 4096.0f;             // small step keeps Euler error tiny
    const int steps = 1024;
    const float total_time = dt * static_cast<float>(steps);

    MMath::Quat q = MMath::quatIdentity();
    for (int i = 0; i < steps; ++i) {
        q = MMath::quatIntegrate(q, omega, dt);
    }

    const MMath::Quat expected = MMath::quatFromAxisAngle(
        MMath::Vec3{0.0f, 0.0f, 1.0f}, total_time);
    requireQuatNearOrient(q, expected, 5e-3f);
}

FM_TEST(Quat, IntegrateVec4OverloadMatchesVec3) {
    FmTest::Rng rng(0x55AA55AAU);
    for (int i = 0; i < 200; ++i) {
        const MMath::Quat q = randomUnitQuat(rng);
        const MMath::Vec3 w{
            rng.uniform(-3.0f, 3.0f),
            rng.uniform(-3.0f, 3.0f),
            rng.uniform(-3.0f, 3.0f)
        };
        const MMath::Vec4 w4{w.x, w.y, w.z, rng.uniform(-10.0f, 10.0f)};  // w lane garbage
        const float dt = rng.uniform(1e-4f, 0.05f);

        const MMath::Quat a = MMath::quatIntegrate(q, w, dt);
        const MMath::Quat b = MMath::quatIntegrate(q, w4, dt);
        FM_REQUIRE_NEAR(a.x, b.x, 1e-6f);
        FM_REQUIRE_NEAR(a.y, b.y, 1e-6f);
        FM_REQUIRE_NEAR(a.z, b.z, 1e-6f);
        FM_REQUIRE_NEAR(a.w, b.w, 1e-6f);
    }
}

// ============================================================================
// Tier 2: Multiply / conjugate / inverse / dot / rotate
// ============================================================================

FM_TEST(Quat, MultiplyIdentityIsIdentity) {
    FmTest::Rng rng(0x12345678U);
    for (int i = 0; i < 500; ++i) {
        const MMath::Quat q = randomUnitQuat(rng);
        const MMath::Quat id = MMath::quatIdentity();
        const MMath::Quat l = MMath::quatMultiply(q, id);
        const MMath::Quat r = MMath::quatMultiply(id, q);
        FM_REQUIRE_NEAR(l.x, q.x, 1e-6f); FM_REQUIRE_NEAR(l.y, q.y, 1e-6f);
        FM_REQUIRE_NEAR(l.z, q.z, 1e-6f); FM_REQUIRE_NEAR(l.w, q.w, 1e-6f);
        FM_REQUIRE_NEAR(r.x, q.x, 1e-6f); FM_REQUIRE_NEAR(r.y, q.y, 1e-6f);
        FM_REQUIRE_NEAR(r.z, q.z, 1e-6f); FM_REQUIRE_NEAR(r.w, q.w, 1e-6f);
    }
}

FM_TEST(Quat, MultiplyCompositionOrder) {
    // Locked semantics: quatMultiply(a, b) applies b first, then a.
    // So rotating v by (a*b) must equal rotating v first by b then by a.
    FmTest::Rng rng(0x9F9F9F9FU);
    for (int i = 0; i < 500; ++i) {
        const MMath::Quat a = randomUnitQuat(rng);
        const MMath::Quat b = randomUnitQuat(rng);
        const MMath::Vec3 v{
            rng.uniform(-3.0f, 3.0f),
            rng.uniform(-3.0f, 3.0f),
            rng.uniform(-3.0f, 3.0f)
        };

        const MMath::Vec3 composed = MMath::quatRotateVec3(MMath::quatMultiply(a, b), v);
        const MMath::Vec3 stepwise = MMath::quatRotateVec3(a, MMath::quatRotateVec3(b, v));
        FM_REQUIRE_VEC3_NEAR(composed, stepwise, 5e-5f);
    }
}

FM_TEST(Quat, ConjugateAndInverse) {
    FmTest::Rng rng(0xDEADBEEFU);
    for (int i = 0; i < 500; ++i) {
        const MMath::Quat q = randomUnitQuat(rng);
        // For a unit quaternion, conjugate == inverse.
        const MMath::Quat c = MMath::quatConjugate(q);
        const MMath::Quat inv = MMath::quatInverse(q);
        FM_REQUIRE_NEAR(c.x, inv.x, 5e-6f);
        FM_REQUIRE_NEAR(c.y, inv.y, 5e-6f);
        FM_REQUIRE_NEAR(c.z, inv.z, 5e-6f);
        FM_REQUIRE_NEAR(c.w, inv.w, 5e-6f);

        // q * conj(q) = identity for unit q.
        const MMath::Quat r = MMath::quatMultiply(q, c);
        FM_REQUIRE_NEAR(r.x, 0.0f, 1e-5f);
        FM_REQUIRE_NEAR(r.y, 0.0f, 1e-5f);
        FM_REQUIRE_NEAR(r.z, 0.0f, 1e-5f);
        FM_REQUIRE_NEAR(r.w, 1.0f, 1e-5f);
    }
}

FM_TEST(Quat, InverseOfNonUnit) {
    // Inverse must satisfy q * inv(q) = identity even when |q| != 1.
    FmTest::Rng rng(0x77777777U);
    for (int i = 0; i < 200; ++i) {
        MMath::Quat q{
            rng.uniform(-3.0f, 3.0f),
            rng.uniform(-3.0f, 3.0f),
            rng.uniform(-3.0f, 3.0f),
            rng.uniform(-3.0f, 3.0f)
        };
        if (MMath::quatLengthSquared(q) < 1e-3f) continue;
        const MMath::Quat inv = MMath::quatInverse(q);
        const MMath::Quat id = MMath::quatMultiply(q, inv);
        FM_REQUIRE_NEAR(id.x, 0.0f, 3e-4f);
        FM_REQUIRE_NEAR(id.y, 0.0f, 3e-4f);
        FM_REQUIRE_NEAR(id.z, 0.0f, 3e-4f);
        FM_REQUIRE_NEAR(id.w, 1.0f, 3e-4f);
    }
}

FM_TEST(Quat, RotateVec3PreservesLength) {
    FmTest::Rng rng(0x12345ABCU);
    for (int i = 0; i < 500; ++i) {
        const MMath::Quat q = randomUnitQuat(rng);
        const MMath::Vec3 v{
            rng.uniform(-10.0f, 10.0f),
            rng.uniform(-10.0f, 10.0f),
            rng.uniform(-10.0f, 10.0f)
        };
        const MMath::Vec3 r = MMath::quatRotateVec3(q, v);
        FM_REQUIRE_NEAR(MMath::vec3Length(r), MMath::vec3Length(v), 2e-4f);
    }
}

FM_TEST(Quat, RotateVec3MatchesAxisAngle) {
    // Rotating +X by 90 degrees about +Z must give +Y.
    const MMath::Quat q = MMath::quatFromAxisAngle(MMath::Vec3{0.0f, 0.0f, 1.0f}, kHalfPi);
    const MMath::Vec3 r = MMath::quatRotateVec3(q, MMath::Vec3{1.0f, 0.0f, 0.0f});
    FM_REQUIRE_VEC3_NEAR(r, (MMath::Vec3{0.0f, 1.0f, 0.0f}), 5e-6f);

    // Rotating +Y by 90 degrees about +X must give +Z.
    const MMath::Quat qx = MMath::quatFromAxisAngle(MMath::Vec3{1.0f, 0.0f, 0.0f}, kHalfPi);
    const MMath::Vec3 rx = MMath::quatRotateVec3(qx, MMath::Vec3{0.0f, 1.0f, 0.0f});
    FM_REQUIRE_VEC3_NEAR(rx, (MMath::Vec3{0.0f, 0.0f, 1.0f}), 5e-6f);
}

FM_TEST(Quat, RotateVec4PreservesWLane) {
    FmTest::Rng rng(0xBADF00DDU);
    for (int i = 0; i < 200; ++i) {
        const MMath::Quat q = randomUnitQuat(rng);
        const MMath::Vec4 v{
            rng.uniform(-5.0f, 5.0f),
            rng.uniform(-5.0f, 5.0f),
            rng.uniform(-5.0f, 5.0f),
            rng.uniform(-5.0f, 5.0f)
        };
        const MMath::Vec4 r = MMath::quatRotateVec4(q, v);
        FM_REQUIRE_NEAR(r.w, v.w, 1e-6f);

        const MMath::Vec3 r3 = MMath::quatRotateVec3(q, MMath::Vec3{v.x, v.y, v.z});
        FM_REQUIRE_NEAR(r.x, r3.x, 1e-6f);
        FM_REQUIRE_NEAR(r.y, r3.y, 1e-6f);
        FM_REQUIRE_NEAR(r.z, r3.z, 1e-6f);
    }
}

FM_TEST(Quat, DotIsSymmetric) {
    FmTest::Rng rng(0x4242);
    for (int i = 0; i < 200; ++i) {
        const MMath::Quat a = randomUnitQuat(rng);
        const MMath::Quat b = randomUnitQuat(rng);
        FM_REQUIRE_NEAR(MMath::quatDot(a, b), MMath::quatDot(b, a), 1e-7f);
    }
}

// ============================================================================
// Tier 3: Matrix conversion / axis-angle / slerp / nlerp / Euler
// ============================================================================

FM_TEST(Quat, ToMat3MatchesRotateVec3) {
    FmTest::Rng rng(0x33333333U);
    for (int i = 0; i < 300; ++i) {
        const MMath::Quat q = randomUnitQuat(rng);
        const MMath::Mat3 m = MMath::quatToMat3(q);
        const MMath::Vec3 v{
            rng.uniform(-5.0f, 5.0f),
            rng.uniform(-5.0f, 5.0f),
            rng.uniform(-5.0f, 5.0f)
        };

        // Mat3 is row-major and we are working with column vectors (M*v).
        const MMath::Vec3 mv{
            m.m00 * v.x + m.m01 * v.y + m.m02 * v.z,
            m.m10 * v.x + m.m11 * v.y + m.m12 * v.z,
            m.m20 * v.x + m.m21 * v.y + m.m22 * v.z
        };
        const MMath::Vec3 qv = MMath::quatRotateVec3(q, v);
        FM_REQUIRE_VEC3_NEAR(mv, qv, 5e-5f);
    }
}

FM_TEST(Quat, ToMat3IsOrthogonal) {
    FmTest::Rng rng(0x44444444U);
    for (int i = 0; i < 200; ++i) {
        const MMath::Quat q = randomUnitQuat(rng);
        const MMath::Mat3 m = MMath::quatToMat3(q);
        // Columns must be orthonormal: c_i . c_j = delta_ij
        const float c00 = m.m00 * m.m00 + m.m10 * m.m10 + m.m20 * m.m20;
        const float c11 = m.m01 * m.m01 + m.m11 * m.m11 + m.m21 * m.m21;
        const float c22 = m.m02 * m.m02 + m.m12 * m.m12 + m.m22 * m.m22;
        const float c01 = m.m00 * m.m01 + m.m10 * m.m11 + m.m20 * m.m21;
        const float c02 = m.m00 * m.m02 + m.m10 * m.m12 + m.m20 * m.m22;
        const float c12 = m.m01 * m.m02 + m.m11 * m.m12 + m.m21 * m.m22;
        FM_REQUIRE_NEAR(c00, 1.0f, 5e-5f);
        FM_REQUIRE_NEAR(c11, 1.0f, 5e-5f);
        FM_REQUIRE_NEAR(c22, 1.0f, 5e-5f);
        FM_REQUIRE_NEAR(c01, 0.0f, 5e-5f);
        FM_REQUIRE_NEAR(c02, 0.0f, 5e-5f);
        FM_REQUIRE_NEAR(c12, 0.0f, 5e-5f);
    }
}

FM_TEST(Quat, ToMat4AgreesWithLegacyMat4FromQuat) {
    // quatToMat4(Quat) must produce the same bytes as the existing
    // mat4FromQuat(Vec4) entry point, since Quat and Vec4 share layout.
    FmTest::Rng rng(0x55555555U);
    for (int i = 0; i < 200; ++i) {
        const MMath::Quat q = randomUnitQuat(rng);
        const MMath::Vec4 q4{q.x, q.y, q.z, q.w};
        const MMath::Mat4 a = MMath::quatToMat4(q);
        const MMath::Mat4 b = MMath::mat4FromQuat(q4);
        FM_REQUIRE_MAT4_NEAR(a, b, 1e-6f);
    }
}

FM_TEST(Quat, AxisAngleRoundTrip) {
    FmTest::Rng rng(0xABCDEF01U);
    for (int i = 0; i < 300; ++i) {
        const MMath::Vec3 axis = randomUnitVec3(rng);
        const float angle = rng.uniform(-kPi + 0.01f, kPi - 0.01f);  // avoid +/- pi degeneracy

        const MMath::Quat q = MMath::quatFromAxisAngle(axis, angle);
        FM_REQUIRE_NEAR(MMath::quatLength(q), 1.0f, 1e-6f);

        MMath::Vec3 axis_out;
        float angle_out;
        MMath::quatToAxisAngle(q, axis_out, angle_out);

        // axis and (axis,angle) may emerge with sign flipped relative to input
        // when q.w extraction normalizes the angle into [0, 2*pi). Compare via
        // the constructed rotation acting on a probe vector.
        const MMath::Vec3 probe{1.0f, 0.0f, 0.0f};
        const MMath::Vec3 r1 = MMath::quatRotateVec3(q, probe);
        const MMath::Quat q2 = MMath::quatFromAxisAngle(axis_out, angle_out);
        const MMath::Vec3 r2 = MMath::quatRotateVec3(q2, probe);
        FM_REQUIRE_VEC3_NEAR(r1, r2, 3e-5f);
    }
}

FM_TEST(Quat, AxisAngleSafeHandlesZeroAxis) {
    MMath::Quat q = MMath::quatFromAxisAngleSafe(MMath::Vec3{0.0f, 0.0f, 0.0f}, 1.0f);
    FM_REQUIRE_NEAR(q.w, 1.0f, 1e-7f);
    FM_REQUIRE_NEAR(q.x, 0.0f, 1e-7f);
    FM_REQUIRE_NEAR(q.y, 0.0f, 1e-7f);
    FM_REQUIRE_NEAR(q.z, 0.0f, 1e-7f);
}

FM_TEST(Quat, SlerpEndpoints) {
    FmTest::Rng rng(0xC0FFEE42U);
    for (int i = 0; i < 200; ++i) {
        const MMath::Quat a = randomUnitQuat(rng);
        const MMath::Quat b = randomUnitQuat(rng);
        const MMath::Quat at0 = MMath::quatSlerp(a, b, 0.0f);
        const MMath::Quat at1 = MMath::quatSlerp(a, b, 1.0f);
        requireQuatNearOrient(at0, a, 5e-5f);
        requireQuatNearOrient(at1, b, 5e-5f);
    }
}

FM_TEST(Quat, SlerpMidpointBisectsAngle) {
    // Slerp at t=0.5 must produce a quaternion whose rotation angle from a
    // equals its angle from b (within fp tolerance).
    FmTest::Rng rng(0xBADCAFE0U);
    for (int i = 0; i < 200; ++i) {
        const MMath::Quat a = randomUnitQuat(rng);
        const MMath::Quat b = randomUnitQuat(rng);
        const MMath::Quat m = MMath::quatSlerp(a, b, 0.5f);
        FM_REQUIRE_NEAR(MMath::quatLength(m), 1.0f, 5e-5f);

        const float d_am = std::fabs(MMath::quatDot(a, m));
        const float d_bm = std::fabs(MMath::quatDot(b, m));
        FM_REQUIRE_NEAR(d_am, d_bm, 5e-5f);
    }
}

FM_TEST(Quat, SlerpDegeneratesToLerpNearParallel) {
    const MMath::Quat a = MMath::quatIdentity();
    const MMath::Quat b = MMath::quatFromAxisAngle(MMath::Vec3{0.0f, 0.0f, 1.0f}, 1e-4f);
    const MMath::Quat m = MMath::quatSlerp(a, b, 0.5f);
    FM_REQUIRE_NEAR(MMath::quatLength(m), 1.0f, 1e-5f);
}

FM_TEST(Quat, NlerpEndpoints) {
    FmTest::Rng rng(0xDEAFBEACU);
    for (int i = 0; i < 200; ++i) {
        const MMath::Quat a = randomUnitQuat(rng);
        const MMath::Quat b = randomUnitQuat(rng);
        const MMath::Quat at0 = MMath::quatNlerp(a, b, 0.0f);
        const MMath::Quat at1 = MMath::quatNlerp(a, b, 1.0f);
        requireQuatNearOrient(at0, a, 5e-5f);
        requireQuatNearOrient(at1, b, 5e-5f);
        FM_REQUIRE_NEAR(MMath::quatLength(at0), 1.0f, 5e-5f);
        FM_REQUIRE_NEAR(MMath::quatLength(at1), 1.0f, 5e-5f);
    }
}

FM_TEST(Quat, EulerRoundTrip) {
    // Stay well clear of the gimbal-lock pitch = +/- pi/2 boundary.
    FmTest::Rng rng(0x10101010U);
    for (int i = 0; i < 200; ++i) {
        const float yaw   = rng.uniform(-kPi + 0.05f, kPi - 0.05f);
        const float pitch = rng.uniform(-kHalfPi + 0.05f, kHalfPi - 0.05f);
        const float roll  = rng.uniform(-kPi + 0.05f, kPi - 0.05f);

        const MMath::Quat q = MMath::quatFromEuler(yaw, pitch, roll);
        FM_REQUIRE_NEAR(MMath::quatLength(q), 1.0f, 1e-5f);

        const MMath::Vec3 e = MMath::quatToEuler(q);  // (roll, pitch, yaw)
        FM_REQUIRE_NEAR(e.x, roll,  1e-3f);
        FM_REQUIRE_NEAR(e.y, pitch, 1e-3f);
        FM_REQUIRE_NEAR(e.z, yaw,   1e-3f);
    }
}

// ============================================================================
// Tier 4: Convenience ops
// ============================================================================

FM_TEST(Quat, ComponentWiseArithmetic) {
    const MMath::Quat a{1.0f, 2.0f, 3.0f, 4.0f};
    const MMath::Quat b{0.5f, 1.5f, 2.5f, 3.5f};

    const MMath::Quat s = MMath::quatAdd(a, b);
    FM_REQUIRE_NEAR(s.x, 1.5f, 1e-7f);
    FM_REQUIRE_NEAR(s.w, 7.5f, 1e-7f);

    const MMath::Quat d = MMath::quatSub(a, b);
    FM_REQUIRE_NEAR(d.x, 0.5f, 1e-7f);
    FM_REQUIRE_NEAR(d.w, 0.5f, 1e-7f);

    const MMath::Quat sc = MMath::quatScale(a, 2.0f);
    FM_REQUIRE_NEAR(sc.x, 2.0f, 1e-7f);
    FM_REQUIRE_NEAR(sc.w, 8.0f, 1e-7f);

    const MMath::Quat n = MMath::quatNegate(a);
    FM_REQUIRE_NEAR(n.x, -1.0f, 1e-7f);
    FM_REQUIRE_NEAR(n.w, -4.0f, 1e-7f);
}

FM_TEST(Quat, NormalizeFastApproximatesNormalize) {
    FmTest::Rng rng(0xFA57FA57U);
    for (int i = 0; i < 200; ++i) {
        const MMath::Quat q{
            rng.uniform(-3.0f, 3.0f),
            rng.uniform(-3.0f, 3.0f),
            rng.uniform(-3.0f, 3.0f),
            rng.uniform(-3.0f, 3.0f)
        };
        if (MMath::quatLengthSquared(q) < 1e-3f) continue;
        const MMath::Quat fast = MMath::quatNormalizeFast(q);
        // ~11-bit rsqrt seed: tolerate ~5e-4 absolute error per component.
        FM_REQUIRE_NEAR(MMath::quatLength(fast), 1.0f, 2e-3f);
    }
}

// ============================================================================
// Batch (SIMD) APIs
// ============================================================================

FM_TEST(QuatBatch, MultiplyArrayMatchesScalar) {
    FmTest::Rng rng(0x9A9A9A9AU);
    for (int n : {1, 4, 7, 16, 33, 128}) {
        std::vector<MMath::Quat> a(n), b(n), out(n);
        for (int i = 0; i < n; ++i) { a[i] = randomUnitQuat(rng); b[i] = randomUnitQuat(rng); }

        MMath::quatMultiplyArray(a.data(), b.data(), out.data(), n);
        for (int i = 0; i < n; ++i) {
            const MMath::Quat ref = MMath::quatMultiply(a[i], b[i]);
            FM_REQUIRE_NEAR(out[i].x, ref.x, 1e-6f);
            FM_REQUIRE_NEAR(out[i].y, ref.y, 1e-6f);
            FM_REQUIRE_NEAR(out[i].z, ref.z, 1e-6f);
            FM_REQUIRE_NEAR(out[i].w, ref.w, 1e-6f);
        }
    }
}

FM_TEST(QuatBatch, IntegrateArrayMatchesScalar) {
    FmTest::Rng rng(0x71717171U);
    constexpr int n = 64;
    std::vector<MMath::Quat> q(n), out(n);
    std::vector<MMath::Vec3> w(n);
    for (int i = 0; i < n; ++i) {
        q[i] = randomUnitQuat(rng);
        w[i] = MMath::Vec3{rng.uniform(-2.0f, 2.0f), rng.uniform(-2.0f, 2.0f), rng.uniform(-2.0f, 2.0f)};
    }
    const float dt = 1.0f / 240.0f;
    MMath::quatIntegrateArray(q.data(), w.data(), dt, out.data(), n);
    for (int i = 0; i < n; ++i) {
        const MMath::Quat ref = MMath::quatIntegrate(q[i], w[i], dt);
        FM_REQUIRE_NEAR(out[i].x, ref.x, 1e-6f);
        FM_REQUIRE_NEAR(out[i].y, ref.y, 1e-6f);
        FM_REQUIRE_NEAR(out[i].z, ref.z, 1e-6f);
        FM_REQUIRE_NEAR(out[i].w, ref.w, 1e-6f);
    }
}

FM_TEST(QuatBatch, NormalizeSafeArrayPreservesIdentityOnZero) {
    std::vector<MMath::Quat> q = {
        MMath::Quat{0.0f, 0.0f, 0.0f, 0.0f},
        MMath::Quat{1.0f, 0.0f, 0.0f, 0.0f},
        MMath::Quat{0.0f, 0.0f, 0.0f, 2.0f},
    };
    MMath::quatNormalizeSafeArray(q.data(), static_cast<int32_t>(q.size()));
    // Zero-length entry -> identity.
    FM_REQUIRE_NEAR(q[0].w, 1.0f, 1e-7f);
    // Others normalized.
    FM_REQUIRE_NEAR(MMath::quatLength(q[1]), 1.0f, 1e-5f);
    FM_REQUIRE_NEAR(MMath::quatLength(q[2]), 1.0f, 1e-5f);
}

// ============================================================================
// Cross-library parity (gated on third-party availability)
// ============================================================================

#if FM_HAVE_GLM

FM_TEST(QuatExternal, MultiplyMatchesGlm) {
    FmTest::Rng rng(0xC1AAAAAAU);
    for (int i = 0; i < 500; ++i) {
        const MMath::Quat a = randomUnitQuat(rng);
        const MMath::Quat b = randomUnitQuat(rng);

        const MMath::Quat r = MMath::quatMultiply(a, b);

        // GLM stores quaternions as (w, x, y, z) and uses Hamilton convention.
        const glm::quat ga(a.w, a.x, a.y, a.z);
        const glm::quat gb(b.w, b.x, b.y, b.z);
        const glm::quat gr = ga * gb;

        FM_REQUIRE_NEAR(r.x, gr.x, 5e-6f);
        FM_REQUIRE_NEAR(r.y, gr.y, 5e-6f);
        FM_REQUIRE_NEAR(r.z, gr.z, 5e-6f);
        FM_REQUIRE_NEAR(r.w, gr.w, 5e-6f);
    }
}

FM_TEST(QuatExternal, RotateVec3MatchesGlm) {
    FmTest::Rng rng(0xC1BBBBBBU);
    for (int i = 0; i < 500; ++i) {
        const MMath::Quat q = randomUnitQuat(rng);
        const MMath::Vec3 v{
            rng.uniform(-5.0f, 5.0f),
            rng.uniform(-5.0f, 5.0f),
            rng.uniform(-5.0f, 5.0f)
        };

        const MMath::Vec3 r = MMath::quatRotateVec3(q, v);

        const glm::quat gq(q.w, q.x, q.y, q.z);
        const glm::vec3 gr = gq * glm::vec3(v.x, v.y, v.z);

        FM_REQUIRE_NEAR(r.x, gr.x, 2e-5f);
        FM_REQUIRE_NEAR(r.y, gr.y, 2e-5f);
        FM_REQUIRE_NEAR(r.z, gr.z, 2e-5f);
    }
}

FM_TEST(QuatExternal, ToMat4MatchesGlm) {
    FmTest::Rng rng(0xC1CCCCCCU);
    for (int i = 0; i < 200; ++i) {
        const MMath::Quat q = randomUnitQuat(rng);
        const MMath::Mat4 m = MMath::quatToMat4(q);

        const glm::quat gq(q.w, q.x, q.y, q.z);
        const glm::mat4 gm = glm::mat4_cast(gq);

        // Both column-major; compare element-by-element.
        for (int c = 0; c < 4; ++c) {
            for (int r = 0; r < 4; ++r) {
                FM_REQUIRE_NEAR(m.m[c * 4 + r], gm[c][r], 5e-6f);
            }
        }
    }
}

#endif // FM_HAVE_GLM

#if FM_HAVE_EIGEN

FM_TEST(QuatExternal, MultiplyMatchesEigen) {
    FmTest::Rng rng(0xE1AAAAAAU);
    for (int i = 0; i < 500; ++i) {
        const MMath::Quat a = randomUnitQuat(rng);
        const MMath::Quat b = randomUnitQuat(rng);

        const MMath::Quat r = MMath::quatMultiply(a, b);

        // Eigen::Quaternionf ctor takes (w, x, y, z); also Hamilton convention.
        const Eigen::Quaternionf ea(a.w, a.x, a.y, a.z);
        const Eigen::Quaternionf eb(b.w, b.x, b.y, b.z);
        const Eigen::Quaternionf er = ea * eb;

        FM_REQUIRE_NEAR(r.x, er.x(), 5e-6f);
        FM_REQUIRE_NEAR(r.y, er.y(), 5e-6f);
        FM_REQUIRE_NEAR(r.z, er.z(), 5e-6f);
        FM_REQUIRE_NEAR(r.w, er.w(), 5e-6f);
    }
}

FM_TEST(QuatExternal, RotateVec3MatchesEigen) {
    FmTest::Rng rng(0xE1BBBBBBU);
    for (int i = 0; i < 500; ++i) {
        const MMath::Quat q = randomUnitQuat(rng);
        const MMath::Vec3 v{
            rng.uniform(-5.0f, 5.0f),
            rng.uniform(-5.0f, 5.0f),
            rng.uniform(-5.0f, 5.0f)
        };

        const MMath::Vec3 r = MMath::quatRotateVec3(q, v);

        const Eigen::Quaternionf eq(q.w, q.x, q.y, q.z);
        const Eigen::Vector3f ev(v.x, v.y, v.z);
        const Eigen::Vector3f er = eq * ev;

        FM_REQUIRE_NEAR(r.x, er.x(), 2e-5f);
        FM_REQUIRE_NEAR(r.y, er.y(), 2e-5f);
        FM_REQUIRE_NEAR(r.z, er.z(), 2e-5f);
    }
}

#endif // FM_HAVE_EIGEN

#if FM_HAVE_DIRECTXMATH

FM_TEST(QuatExternal, MultiplyMatchesDirectXMath) {
    using namespace DirectX;
    FmTest::Rng rng(0xD1AAAAAAU);
    for (int i = 0; i < 500; ++i) {
        const MMath::Quat a = randomUnitQuat(rng);
        const MMath::Quat b = randomUnitQuat(rng);

        const MMath::Quat r = MMath::quatMultiply(a, b);

        // DirectXMath stores quaternions as (x, y, z, w); XMQuaternionMultiply
        // returns Q1 * Q2 meaning "rotate by Q1 first then by Q2" — the opposite
        // of our convention. To match our quatMultiply(a, b) = "b first then a",
        // we pass them swapped.
        const XMVECTOR da = XMVectorSet(a.x, a.y, a.z, a.w);
        const XMVECTOR db = XMVectorSet(b.x, b.y, b.z, b.w);
        const XMVECTOR dr = XMQuaternionMultiply(db, da);  // intentionally swapped
        XMFLOAT4 out;
        XMStoreFloat4(&out, dr);

        FM_REQUIRE_NEAR(r.x, out.x, 1e-5f);
        FM_REQUIRE_NEAR(r.y, out.y, 1e-5f);
        FM_REQUIRE_NEAR(r.z, out.z, 1e-5f);
        FM_REQUIRE_NEAR(r.w, out.w, 1e-5f);
    }
}

#endif // FM_HAVE_DIRECTXMATH
