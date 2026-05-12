#include "bench/framework/benchmark_framework.h"
#include "bench/adapters/external_backends.h"

#include "fast_math/quat.h"

#include <random>
#include <vector>

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

class Rng {
public:
    explicit Rng(uint32_t seed_) : eng(seed_) {}
    float uniform(float lo_, float hi_) {
        std::uniform_real_distribution<float> dist(lo_, hi_);
        return dist(eng);
    }
private:
    std::mt19937 eng;
};

MMath::Quat randomUnitQuat(Rng& rng_) {
    MMath::Quat q{
        rng_.uniform(-1.0f, 1.0f),
        rng_.uniform(-1.0f, 1.0f),
        rng_.uniform(-1.0f, 1.0f),
        rng_.uniform(-1.0f, 1.0f)
    };
    const float len_sq = MMath::quatLengthSquared(q);
    if (len_sq < 1e-4f) {
        return MMath::quatIdentity();
    }
    return MMath::quatNormalize(q);
}

std::vector<MMath::Quat> makeQuats(int n_, uint32_t seed_) {
    Rng rng(seed_);
    std::vector<MMath::Quat> out(n_);
    for (auto& q : out) q = randomUnitQuat(rng);
    return out;
}

std::vector<MMath::Vec3> makeOmegas(int n_, uint32_t seed_) {
    Rng rng(seed_);
    std::vector<MMath::Vec3> out(n_);
    for (auto& w : out) {
        w = MMath::Vec3{rng.uniform(-5.0f, 5.0f), rng.uniform(-5.0f, 5.0f), rng.uniform(-5.0f, 5.0f)};
    }
    return out;
}

std::vector<MMath::Vec3> makeVec3s(int n_, uint32_t seed_) {
    Rng rng(seed_);
    std::vector<MMath::Vec3> out(n_);
    for (auto& v : out) {
        v = MMath::Vec3{rng.uniform(-10.0f, 10.0f), rng.uniform(-10.0f, 10.0f), rng.uniform(-10.0f, 10.0f)};
    }
    return out;
}

} // namespace

// ============================================================================
// quatMultiply: pairwise Hamilton product
// ============================================================================

FM_BENCH(Quat, MultiplyThroughput) {
    constexpr int n = 8192;
    const auto a = makeQuats(n, 0x11111111U);
    const auto b = makeQuats(n, 0x22222222U);

#if FM_HAVE_GLM
    std::vector<glm::quat> ga(n), gb(n);
    for (int i = 0; i < n; ++i) {
        ga[i] = glm::quat(a[i].w, a[i].x, a[i].y, a[i].z);
        gb[i] = glm::quat(b[i].w, b[i].x, b[i].y, b[i].z);
    }
#endif

#if FM_HAVE_EIGEN
    std::vector<Eigen::Quaternionf> ea(n), eb(n);
    for (int i = 0; i < n; ++i) {
        ea[i] = Eigen::Quaternionf(a[i].w, a[i].x, a[i].y, a[i].z);
        eb[i] = Eigen::Quaternionf(b[i].w, b[i].x, b[i].y, b[i].z);
    }
#endif

#if FM_HAVE_DIRECTXMATH
    std::vector<DirectX::XMFLOAT4> da(n), db(n);
    for (int i = 0; i < n; ++i) {
        da[i] = DirectX::XMFLOAT4(a[i].x, a[i].y, a[i].z, a[i].w);
        db[i] = DirectX::XMFLOAT4(b[i].x, b[i].y, b[i].z, b[i].w);
    }
#endif

    FmBench::runComparisonCase(
        "quat multiply (Hamilton)",
        static_cast<std::size_t>(n),
        {
            {"fast_math", true, [&]() {
                 double acc = 0.0;
                 for (int i = 0; i < n; ++i) {
                     auto r = MMath::quatMultiply(a[i], b[i]);
                     acc += r.x + r.y + r.z + r.w;
                 }
                 FmBench::consume(acc);
             }},
#if FM_HAVE_GLM
            {"glm", true, [&]() {
                 double acc = 0.0;
                 for (int i = 0; i < n; ++i) {
                     glm::quat r = ga[i] * gb[i];
                     acc += r.x + r.y + r.z + r.w;
                 }
                 FmBench::consume(acc);
             }},
#endif
#if FM_HAVE_EIGEN
            {"eigen", true, [&]() {
                 double acc = 0.0;
                 for (int i = 0; i < n; ++i) {
                     Eigen::Quaternionf r = ea[i] * eb[i];
                     acc += r.x() + r.y() + r.z() + r.w();
                 }
                 FmBench::consume(acc);
             }},
#endif
#if FM_HAVE_DIRECTXMATH
            {"directxmath", true, [&]() {
                 using namespace DirectX;
                 double acc = 0.0;
                 for (int i = 0; i < n; ++i) {
                     XMVECTOR va = XMLoadFloat4(&da[i]);
                     XMVECTOR vb = XMLoadFloat4(&db[i]);
                     XMVECTOR r = XMQuaternionMultiply(vb, va);  // matches our composition order
                     XMFLOAT4 out; XMStoreFloat4(&out, r);
                     acc += out.x + out.y + out.z + out.w;
                 }
                 FmBench::consume(acc);
             }},
#endif
        },
        options.config);
}

// ============================================================================
// quatNormalize: throughput of unit-length normalization
// ============================================================================

FM_BENCH(Quat, NormalizeThroughput) {
    constexpr int n = 8192;
    // Use unnormalized quaternions so normalization is meaningful.
    Rng rng(0x33333333U);
    std::vector<MMath::Quat> q(n);
    for (auto& v : q) {
        v = MMath::Quat{rng.uniform(-3.0f, 3.0f), rng.uniform(-3.0f, 3.0f),
                        rng.uniform(-3.0f, 3.0f), rng.uniform(-3.0f, 3.0f)};
    }

#if FM_HAVE_GLM
    std::vector<glm::quat> gq(n);
    for (int i = 0; i < n; ++i) gq[i] = glm::quat(q[i].w, q[i].x, q[i].y, q[i].z);
#endif

    FmBench::runComparisonCase(
        "quat normalize",
        static_cast<std::size_t>(n),
        {
            {"fast_math", true, [&]() {
                 double acc = 0.0;
                 for (int i = 0; i < n; ++i) {
                     auto r = MMath::quatNormalize(q[i]);
                     acc += r.x + r.y + r.z + r.w;
                 }
                 FmBench::consume(acc);
             }},
            {"fast_math_fast", true, [&]() {
                 double acc = 0.0;
                 for (int i = 0; i < n; ++i) {
                     auto r = MMath::quatNormalizeFast(q[i]);
                     acc += r.x + r.y + r.z + r.w;
                 }
                 FmBench::consume(acc);
             }},
#if FM_HAVE_GLM
            {"glm", true, [&]() {
                 double acc = 0.0;
                 for (int i = 0; i < n; ++i) {
                     auto r = glm::normalize(gq[i]);
                     acc += r.x + r.y + r.z + r.w;
                 }
                 FmBench::consume(acc);
             }},
#endif
        },
        options.config);
}

// ============================================================================
// quatIntegrate: physics hot path
// ============================================================================

FM_BENCH(Quat, IntegrateThroughput) {
    constexpr int n = 8192;
    const auto q = makeQuats(n, 0x44444444U);
    const auto omega = makeOmegas(n, 0x55555555U);
    const float dt = 1.0f / 240.0f;

#if FM_HAVE_GLM
    std::vector<glm::quat> gq(n);
    std::vector<glm::vec3> gw(n);
    for (int i = 0; i < n; ++i) {
        gq[i] = glm::quat(q[i].w, q[i].x, q[i].y, q[i].z);
        gw[i] = glm::vec3(omega[i].x, omega[i].y, omega[i].z);
    }
#endif

    FmBench::runComparisonCase(
        "quat integrate (one Euler step + renormalize)",
        static_cast<std::size_t>(n),
        {
            {"fast_math", true, [&]() {
                 double acc = 0.0;
                 for (int i = 0; i < n; ++i) {
                     auto r = MMath::quatIntegrate(q[i], omega[i], dt);
                     acc += r.x + r.y + r.z + r.w;
                 }
                 FmBench::consume(acc);
             }},
#if FM_HAVE_GLM
            {"glm_manual", true, [&]() {
                 double acc = 0.0;
                 for (int i = 0; i < n; ++i) {
                     // Equivalent open-coded integration with the same convention.
                     const glm::quat& qq = gq[i];
                     const glm::vec3& w = gw[i];
                     glm::quat omega_q(0.0f, w.x, w.y, w.z);
                     glm::quat dq = omega_q * qq;
                     glm::quat raw(
                         qq.w + 0.5f * dt * dq.w,
                         qq.x + 0.5f * dt * dq.x,
                         qq.y + 0.5f * dt * dq.y,
                         qq.z + 0.5f * dt * dq.z);
                     glm::quat r = glm::normalize(raw);
                     acc += r.x + r.y + r.z + r.w;
                 }
                 FmBench::consume(acc);
             }},
#endif
        },
        options.config);
}

// ============================================================================
// quatRotateVec3: typical asset/animation hot path
// ============================================================================

FM_BENCH(Quat, RotateVec3Throughput) {
    constexpr int n = 8192;
    const auto q = makeQuats(n, 0x66666666U);
    const auto v = makeVec3s(n, 0x77777777U);

#if FM_HAVE_GLM
    std::vector<glm::quat> gq(n);
    std::vector<glm::vec3> gv(n);
    for (int i = 0; i < n; ++i) {
        gq[i] = glm::quat(q[i].w, q[i].x, q[i].y, q[i].z);
        gv[i] = glm::vec3(v[i].x, v[i].y, v[i].z);
    }
#endif

#if FM_HAVE_EIGEN
    std::vector<Eigen::Quaternionf> eq(n);
    std::vector<Eigen::Vector3f> ev(n);
    for (int i = 0; i < n; ++i) {
        eq[i] = Eigen::Quaternionf(q[i].w, q[i].x, q[i].y, q[i].z);
        ev[i] = Eigen::Vector3f(v[i].x, v[i].y, v[i].z);
    }
#endif

    FmBench::runComparisonCase(
        "quat rotate Vec3",
        static_cast<std::size_t>(n),
        {
            {"fast_math", true, [&]() {
                 double acc = 0.0;
                 for (int i = 0; i < n; ++i) {
                     auto r = MMath::quatRotateVec3(q[i], v[i]);
                     acc += r.x + r.y + r.z;
                 }
                 FmBench::consume(acc);
             }},
#if FM_HAVE_GLM
            {"glm", true, [&]() {
                 double acc = 0.0;
                 for (int i = 0; i < n; ++i) {
                     glm::vec3 r = gq[i] * gv[i];
                     acc += r.x + r.y + r.z;
                 }
                 FmBench::consume(acc);
             }},
#endif
#if FM_HAVE_EIGEN
            {"eigen", true, [&]() {
                 double acc = 0.0;
                 for (int i = 0; i < n; ++i) {
                     Eigen::Vector3f r = eq[i] * ev[i];
                     acc += r.x() + r.y() + r.z();
                 }
                 FmBench::consume(acc);
             }},
#endif
        },
        options.config);
}

// ============================================================================
// quatToMat3 / quatToMat4: matrix-stream upload throughput
// ============================================================================

FM_BENCH(Quat, ToMat3Throughput) {
    constexpr int n = 8192;
    const auto q = makeQuats(n, 0x88888888U);

    FmBench::runComparisonCase(
        "quat -> Mat3",
        static_cast<std::size_t>(n),
        {
            {"fast_math", true, [&]() {
                 double acc = 0.0;
                 for (int i = 0; i < n; ++i) {
                     auto m = MMath::quatToMat3(q[i]);
                     acc += m.m00 + m.m11 + m.m22;
                 }
                 FmBench::consume(acc);
             }},
        },
        options.config);
}

FM_BENCH(Quat, ToMat4Throughput) {
    constexpr int n = 8192;
    const auto q = makeQuats(n, 0x99999999U);

#if FM_HAVE_GLM
    std::vector<glm::quat> gq(n);
    for (int i = 0; i < n; ++i) gq[i] = glm::quat(q[i].w, q[i].x, q[i].y, q[i].z);
#endif

    FmBench::runComparisonCase(
        "quat -> Mat4",
        static_cast<std::size_t>(n),
        {
            {"fast_math", true, [&]() {
                 double acc = 0.0;
                 for (int i = 0; i < n; ++i) {
                     auto m = MMath::quatToMat4(q[i]);
                     acc += m.m[0] + m.m[5] + m.m[10];
                 }
                 FmBench::consume(acc);
             }},
#if FM_HAVE_GLM
            {"glm", true, [&]() {
                 double acc = 0.0;
                 for (int i = 0; i < n; ++i) {
                     glm::mat4 m = glm::mat4_cast(gq[i]);
                     acc += m[0][0] + m[1][1] + m[2][2];
                 }
                 FmBench::consume(acc);
             }},
#endif
        },
        options.config);
}

// ============================================================================
// quatSlerp: animation blending
// ============================================================================

FM_BENCH(Quat, SlerpThroughput) {
    constexpr int n = 4096;
    const auto a = makeQuats(n, 0xAAAAAAAAU);
    const auto b = makeQuats(n, 0xBBBBBBBBU);

#if FM_HAVE_GLM
    std::vector<glm::quat> ga(n), gb(n);
    for (int i = 0; i < n; ++i) {
        ga[i] = glm::quat(a[i].w, a[i].x, a[i].y, a[i].z);
        gb[i] = glm::quat(b[i].w, b[i].x, b[i].y, b[i].z);
    }
#endif

    FmBench::runComparisonCase(
        "quat slerp t=0.37",
        static_cast<std::size_t>(n),
        {
            {"fast_math", true, [&]() {
                 double acc = 0.0;
                 for (int i = 0; i < n; ++i) {
                     auto r = MMath::quatSlerp(a[i], b[i], 0.37f);
                     acc += r.x + r.y + r.z + r.w;
                 }
                 FmBench::consume(acc);
             }},
            {"fast_math_nlerp", true, [&]() {
                 double acc = 0.0;
                 for (int i = 0; i < n; ++i) {
                     auto r = MMath::quatNlerp(a[i], b[i], 0.37f);
                     acc += r.x + r.y + r.z + r.w;
                 }
                 FmBench::consume(acc);
             }},
#if FM_HAVE_GLM
            {"glm", true, [&]() {
                 double acc = 0.0;
                 for (int i = 0; i < n; ++i) {
                     glm::quat r = glm::slerp(ga[i], gb[i], 0.37f);
                     acc += r.x + r.y + r.z + r.w;
                 }
                 FmBench::consume(acc);
             }},
#endif
        },
        options.config);
}

// ============================================================================
// Batch integrate (physics tick simulation)
// ============================================================================

FM_BENCH(Quat, IntegrateBatchThroughput) {
    constexpr int n = 16384;
    const auto q = makeQuats(n, 0xC0DEC0DEU);
    const auto omega = makeOmegas(n, 0xFEEDFEEDU);
    const float dt = 1.0f / 240.0f;

    std::vector<MMath::Quat> out(n);

    FmBench::runComparisonCase(
        "quat integrate array (physics tick)",
        static_cast<std::size_t>(n),
        {
            {"fast_math_array", true, [&]() {
                 MMath::quatIntegrateArray(q.data(), omega.data(), dt, out.data(), n);
                 double acc = 0.0;
                 for (int i = 0; i < n; ++i) acc += out[i].w;
                 FmBench::consume(acc);
             }},
            {"fast_math_scalar_loop", true, [&]() {
                 for (int i = 0; i < n; ++i) {
                     out[i] = MMath::quatIntegrate(q[i], omega[i], dt);
                 }
                 double acc = 0.0;
                 for (int i = 0; i < n; ++i) acc += out[i].w;
                 FmBench::consume(acc);
             }},
        },
        options.config);
}
