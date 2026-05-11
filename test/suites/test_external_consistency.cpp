#include "test/framework/test_framework.h"
#include "test/framework/test_utils.h"

#include "fast_math/mat4.h"
#include "fast_math/vec3.h"

#include <cmath>

#if FM_HAVE_GLM
#include <glm/glm.hpp>
#endif

#if FM_HAVE_EIGEN
#include <Eigen/Dense>
#endif

#if FM_HAVE_DIRECTXMATH
#include <DirectXMath.h>
#endif

namespace {

#if FM_HAVE_GLM
inline glm::vec3 to_glm(const MMath::Vec3& v) { return glm::vec3(v.x, v.y, v.z); }
inline glm::vec4 to_glm4(const MMath::Vec4& v) { return glm::vec4(v.x, v.y, v.z, v.w); }
inline glm::mat4 to_glm(const MMath::Mat4& m) {
    glm::mat4 out(1.0f);
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            out[c][r] = m.m[c * 4 + r];
        }
    }
    return out;
}
#endif

#if FM_HAVE_EIGEN
inline Eigen::Vector3f to_eigen(const MMath::Vec3& v) { return Eigen::Vector3f(v.x, v.y, v.z); }
inline Eigen::Vector4f to_eigen4(const MMath::Vec4& v) { return Eigen::Vector4f(v.x, v.y, v.z, v.w); }
inline Eigen::Matrix4f to_eigen(const MMath::Mat4& m) {
    Eigen::Matrix4f out;
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            out(r, c) = m.m[c * 4 + r];
        }
    }
    return out;
}
#endif

} // namespace

FM_TEST(External, Vector3Consistency) {
    FmTest::Rng rng(0x5A5A5A5AU);

    for (int i = 0; i < 3000; ++i) {
        MMath::Vec3 a = FmTest::randomVec3(rng, -10.0f, 10.0f);
        MMath::Vec3 b = FmTest::randomVec3(rng, -10.0f, 10.0f);

        const float dot_fast = MMath::vec3Dot(a, b);
        const MMath::Vec3 cross_fast = MMath::vec3Cross(a, b);
#if !FM_HAVE_GLM && !FM_HAVE_EIGEN && !FM_HAVE_DIRECTXMATH
        static_cast<void>(dot_fast);
        static_cast<void>(cross_fast);
#endif
        if (MMath::vec3LengthSquared(a) > 1e-4f) {
            const MMath::Vec3 n_fast = MMath::vec3Normalize(a);
            FM_REQUIRE_NEAR(MMath::vec3Length(n_fast), 1.0f, 2e-4f);
        }

#if FM_HAVE_GLM
        const glm::vec3 ga = to_glm(a);
        const glm::vec3 gb = to_glm(b);
        FM_REQUIRE_NEAR(dot_fast, glm::dot(ga, gb), 6e-5f);

        const glm::vec3 gc = glm::cross(ga, gb);
        FM_REQUIRE_VEC3_NEAR(cross_fast, (MMath::Vec3{gc.x, gc.y, gc.z}), 2e-5f);

        if (glm::dot(ga, ga) > 1e-4f) {
            const glm::vec3 gn = glm::normalize(ga);
            FM_REQUIRE_NEAR(gn.x, MMath::vec3Normalize(a).x, 2e-4f);
            FM_REQUIRE_NEAR(gn.y, MMath::vec3Normalize(a).y, 2e-4f);
            FM_REQUIRE_NEAR(gn.z, MMath::vec3Normalize(a).z, 2e-4f);
        }
#endif

#if FM_HAVE_EIGEN
        const Eigen::Vector3f ea = to_eigen(a);
        const Eigen::Vector3f eb = to_eigen(b);
        FM_REQUIRE_NEAR(dot_fast, ea.dot(eb), 6e-5f);

        const Eigen::Vector3f ec = ea.cross(eb);
        FM_REQUIRE_VEC3_NEAR(cross_fast, (MMath::Vec3{ec.x(), ec.y(), ec.z()}), 2e-5f);
#endif

#if FM_HAVE_DIRECTXMATH
        using namespace DirectX;
        const XMVECTOR da = XMVectorSet(a.x, a.y, a.z, 0.0f);
        const XMVECTOR db = XMVectorSet(b.x, b.y, b.z, 0.0f);

        const float ddot = XMVectorGetX(XMVector3Dot(da, db));
        FM_REQUIRE_NEAR(dot_fast, ddot, 6e-5f);

        XMVECTOR dc = XMVector3Cross(da, db);
        FM_REQUIRE_VEC3_NEAR(
            cross_fast,
            (MMath::Vec3{XMVectorGetX(dc), XMVectorGetY(dc), XMVectorGetZ(dc)}),
            2e-5f);
#endif
    }
}

FM_TEST(External, Matrix4ConsistencyGlmEigen) {
    FmTest::Rng rng(0x01010A0AU);

    for (int i = 0; i < 1500; ++i) {
        MMath::Mat4 a = FmTest::randomMat4(rng, -2.0f, 2.0f);
        MMath::Mat4 b = FmTest::randomMat4(rng, -2.0f, 2.0f);
        MMath::Vec4 v = FmTest::randomVec4(rng, -2.0f, 2.0f);

        const MMath::Mat4 m_fast = MMath::mat4Multiply(a, b);
        const MMath::Vec4 v_fast = MMath::mat4MultiplyVec4(a, v);
#if !FM_HAVE_GLM && !FM_HAVE_EIGEN
        static_cast<void>(m_fast);
        static_cast<void>(v_fast);
#endif

#if FM_HAVE_GLM
        const glm::mat4 ga = to_glm(a);
        const glm::mat4 gb = to_glm(b);
        const glm::mat4 gm = ga * gb;

        MMath::Mat4 m_ref{};
        for (int c = 0; c < 4; ++c) {
            for (int r = 0; r < 4; ++r) {
                m_ref.m[c * 4 + r] = gm[c][r];
            }
        }
        FM_REQUIRE_MAT4_NEAR(m_fast, m_ref, 2e-4f);

        const glm::vec4 gv = ga * to_glm4(v);
        FM_REQUIRE_VEC4_NEAR(v_fast, (MMath::Vec4{gv.x, gv.y, gv.z, gv.w}), 2e-4f);
#endif

#if FM_HAVE_EIGEN
        const Eigen::Matrix4f ea = to_eigen(a);
        const Eigen::Matrix4f eb = to_eigen(b);
        const Eigen::Matrix4f em = ea * eb;

        MMath::Mat4 m_ref_eigen{};
        for (int c = 0; c < 4; ++c) {
            for (int r = 0; r < 4; ++r) {
                m_ref_eigen.m[c * 4 + r] = em(r, c);
            }
        }
        FM_REQUIRE_MAT4_NEAR(m_fast, m_ref_eigen, 2e-4f);

        const Eigen::Vector4f ev = ea * to_eigen4(v);
        FM_REQUIRE_VEC4_NEAR(v_fast, (MMath::Vec4{ev.x(), ev.y(), ev.z(), ev.w()}), 2e-4f);
#endif

#if !FM_HAVE_GLM && !FM_HAVE_EIGEN
        FM_REQUIRE(true);
#endif
    }
}
