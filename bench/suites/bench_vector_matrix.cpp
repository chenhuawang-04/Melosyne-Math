#include "bench/framework/benchmark_framework.h"
#include "bench/adapters/external_backends.h"

#include "fast_math/mat4.h"
#include "fast_math/vec3.h"
#include "fast_math/vec4.h"

#include <cmath>
#include <random>
#include <vector>

namespace {

class Rng {
public:
    explicit Rng(uint32_t seed) : eng_(seed) {}
    float uniform(float lo, float hi) {
        std::uniform_real_distribution<float> dist(lo, hi);
        return dist(eng_);
    }

private:
    std::mt19937 eng_;
};

Rng make_rng() {
    return Rng(0x2468ACE0U);
}

std::vector<MMath::Vec3> make_vec3_data(int n, Rng& rng) {
    std::vector<MMath::Vec3> out(n);
    for (auto& v : out) {
        v = MMath::Vec3{rng.uniform(-10.0f, 10.0f), rng.uniform(-10.0f, 10.0f), rng.uniform(-10.0f, 10.0f)};
    }
    return out;
}

std::vector<MMath::Vec4> make_vec4_data(int n, Rng& rng) {
    std::vector<MMath::Vec4> out(n);
    for (auto& v : out) {
        v = MMath::Vec4{
            rng.uniform(-10.0f, 10.0f),
            rng.uniform(-10.0f, 10.0f),
            rng.uniform(-10.0f, 10.0f),
            rng.uniform(-10.0f, 10.0f)};
    }
    return out;
}

std::vector<MMath::Mat4> make_mat4_data(int n, Rng& rng) {
    std::vector<MMath::Mat4> out(n);
    for (auto& m : out) {
        m = MMath::mat4Mul(
            MMath::mat4Translation(rng.uniform(-5.0f, 5.0f), rng.uniform(-5.0f, 5.0f), rng.uniform(-5.0f, 5.0f)),
            MMath::mat4Mul(
                MMath::mat4RotationX(rng.uniform(-3.14f, 3.14f)),
                MMath::mat4Scale(rng.uniform(0.7f, 1.8f), rng.uniform(0.7f, 1.8f), rng.uniform(0.7f, 1.8f))));
    }
    return out;
}

} // namespace

FM_BENCH(Vector, Vec3DotCrossNormalize) {
    constexpr int N = 16384;
    auto rng = make_rng();

    const auto a = make_vec3_data(N, rng);
    const auto b = make_vec3_data(N, rng);

#if FM_HAVE_GLM
    std::vector<glm::vec3> ga(N), gb(N);
    for (int i = 0; i < N; ++i) {
        ga[i] = fmbench::adapters::to_glm(a[i]);
        gb[i] = fmbench::adapters::to_glm(b[i]);
    }
#endif

#if FM_HAVE_EIGEN
    std::vector<Eigen::Vector3f> ea(N), eb(N);
    for (int i = 0; i < N; ++i) {
        ea[i] = fmbench::adapters::to_eigen(a[i]);
        eb[i] = fmbench::adapters::to_eigen(b[i]);
    }
#endif

#if FM_HAVE_DIRECTXMATH
    std::vector<DirectX::XMFLOAT3> da(N), db(N);
    for (int i = 0; i < N; ++i) {
        da[i] = DirectX::XMFLOAT3(a[i].x, a[i].y, a[i].z);
        db[i] = DirectX::XMFLOAT3(b[i].x, b[i].y, b[i].z);
    }
#endif

    fmbench::run_comparison_case(
        "vec3 dot+cross+normalize",
        static_cast<std::size_t>(N * 3),
        {
            {"fast_math", true, [&]() {
                 double acc = 0.0;
                 for (int i = 0; i < N; ++i) {
                     acc += MMath::vec3Dot(a[i], b[i]);
                     auto c = MMath::vec3Cross(a[i], b[i]);
                     auto n = MMath::vec3Normalize(MMath::vec3Add(a[i], MMath::Vec3{0.1f, 0.2f, 0.3f}));
                     acc += c.x + c.y + c.z + n.x + n.y + n.z;
                 }
                 fmbench::consume(acc);
             }},
#if FM_HAVE_GLM
            {"glm", true, [&]() {
                 double acc = 0.0;
                 for (int i = 0; i < N; ++i) {
                     acc += glm::dot(ga[i], gb[i]);
                     glm::vec3 c = glm::cross(ga[i], gb[i]);
                     glm::vec3 n = glm::normalize(ga[i] + glm::vec3(0.1f, 0.2f, 0.3f));
                     acc += c.x + c.y + c.z + n.x + n.y + n.z;
                 }
                 fmbench::consume(acc);
             }},
#else
            {"glm", false, {}},
#endif
#if FM_HAVE_EIGEN
            {"eigen", true, [&]() {
                 double acc = 0.0;
                 for (int i = 0; i < N; ++i) {
                     acc += ea[i].dot(eb[i]);
                     Eigen::Vector3f c = ea[i].cross(eb[i]);
                     Eigen::Vector3f n = (ea[i] + Eigen::Vector3f(0.1f, 0.2f, 0.3f)).normalized();
                     acc += c.x() + c.y() + c.z() + n.x() + n.y() + n.z();
                 }
                 fmbench::consume(acc);
             }},
#else
            {"eigen", false, {}},
#endif
#if FM_HAVE_DIRECTXMATH
            {"directxmath", true, [&]() {
                 using namespace DirectX;
                 double acc = 0.0;
                 for (int i = 0; i < N; ++i) {
                     XMVECTOR va = XMLoadFloat3(&da[i]);
                     XMVECTOR vb = XMLoadFloat3(&db[i]);
                     acc += XMVectorGetX(XMVector3Dot(va, vb));
                     XMVECTOR c = XMVector3Cross(va, vb);
                     XMVECTOR n = XMVector3Normalize(XMVectorAdd(va, XMVectorSet(0.1f, 0.2f, 0.3f, 0.0f)));
                     acc += XMVectorGetX(c) + XMVectorGetY(c) + XMVectorGetZ(c) +
                            XMVectorGetX(n) + XMVectorGetY(n) + XMVectorGetZ(n);
                 }
                 fmbench::consume(acc);
             }},
#else
            {"directxmath", false, {}},
#endif
        },
        options.config);
}

FM_BENCH(Matrix, Mat4MulMulVecTransposeInverse) {
    auto rng = make_rng();

    constexpr int N_MUL = 2048;
    constexpr int N_INV = 256;

    const auto a = make_mat4_data(N_MUL, rng);
    const auto b = make_mat4_data(N_MUL, rng);
    const auto v = make_vec4_data(N_MUL, rng);

#if FM_HAVE_GLM
    std::vector<glm::mat4> ga(N_MUL), gb(N_MUL);
    std::vector<glm::vec4> gv(N_MUL);
    for (int i = 0; i < N_MUL; ++i) {
        ga[i] = fmbench::adapters::to_glm(a[i]);
        gb[i] = fmbench::adapters::to_glm(b[i]);
        gv[i] = fmbench::adapters::to_glm(v[i]);
    }
#endif

#if FM_HAVE_EIGEN
    std::vector<Eigen::Matrix4f> ea(N_MUL), eb(N_MUL);
    std::vector<Eigen::Vector4f> ev(N_MUL);
    for (int i = 0; i < N_MUL; ++i) {
        ea[i] = fmbench::adapters::to_eigen(a[i]);
        eb[i] = fmbench::adapters::to_eigen(b[i]);
        ev[i] = fmbench::adapters::to_eigen(v[i]);
    }
#endif

#if FM_HAVE_DIRECTXMATH
    std::vector<DirectX::XMFLOAT4X4> da(N_MUL), db(N_MUL);
    std::vector<DirectX::XMFLOAT4> dv(N_MUL);
    for (int i = 0; i < N_MUL; ++i) {
        // XMMatrix uses row-major constructors; benchmark only cares about throughput.
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                da[i].m[r][c] = a[i].m[c * 4 + r];
                db[i].m[r][c] = b[i].m[c * 4 + r];
            }
        }
        dv[i] = DirectX::XMFLOAT4(v[i].x, v[i].y, v[i].z, v[i].w);
    }
#endif

    fmbench::run_comparison_case(
        "mat4 mul + mulVec + transpose + inverse",
        static_cast<std::size_t>(N_MUL * 3 + N_INV),
        {
            {"fast_math", true, [&]() {
                 double acc = 0.0;
                 for (int i = 0; i < N_MUL; ++i) {
                     auto m = MMath::mat4Mul(a[i], b[i]);
                     auto tv = MMath::mat4MulVec4(m, v[i]);
                     auto t = MMath::mat4Transpose(m);
                     acc += tv.x + tv.y + tv.z + tv.w + t.m[0] + t.m[5] + t.m[10] + t.m[15];
                 }
                 for (int i = 0; i < N_INV; ++i) {
                     auto inv = MMath::mat4Inverse(a[i]);
                     acc += inv.m[0] + inv.m[5] + inv.m[10] + inv.m[15];
                 }
                 fmbench::consume(acc);
             }},
#if FM_HAVE_GLM
            {"glm", true, [&]() {
                 double acc = 0.0;
                 for (int i = 0; i < N_MUL; ++i) {
                     glm::mat4 m = ga[i] * gb[i];
                     glm::vec4 tv = m * gv[i];
                     glm::mat4 t = glm::transpose(m);
                     acc += tv.x + tv.y + tv.z + tv.w + t[0][0] + t[1][1] + t[2][2] + t[3][3];
                 }
                 for (int i = 0; i < N_INV; ++i) {
                     glm::mat4 inv = glm::inverse(ga[i]);
                     acc += inv[0][0] + inv[1][1] + inv[2][2] + inv[3][3];
                 }
                 fmbench::consume(acc);
             }},
#else
            {"glm", false, {}},
#endif
#if FM_HAVE_EIGEN
            {"eigen", true, [&]() {
                 double acc = 0.0;
                 for (int i = 0; i < N_MUL; ++i) {
                     Eigen::Matrix4f m = ea[i] * eb[i];
                     Eigen::Vector4f tv = m * ev[i];
                     Eigen::Matrix4f t = m.transpose();
                     acc += tv.x() + tv.y() + tv.z() + tv.w() + t(0, 0) + t(1, 1) + t(2, 2) + t(3, 3);
                 }
                 for (int i = 0; i < N_INV; ++i) {
                     Eigen::Matrix4f inv = ea[i].inverse();
                     acc += inv(0, 0) + inv(1, 1) + inv(2, 2) + inv(3, 3);
                 }
                 fmbench::consume(acc);
             }},
#else
            {"eigen", false, {}},
#endif
#if FM_HAVE_DIRECTXMATH
            {"directxmath", true, [&]() {
                 using namespace DirectX;
                 double acc = 0.0;
                 for (int i = 0; i < N_MUL; ++i) {
                     XMMATRIX ma = XMLoadFloat4x4(&da[i]);
                     XMMATRIX mb = XMLoadFloat4x4(&db[i]);
                     XMMATRIX m = XMMatrixMultiply(ma, mb);
                     XMVECTOR vv = XMLoadFloat4(&dv[i]);
                     XMVECTOR tv = XMVector4Transform(vv, m);
                     XMMATRIX t = XMMatrixTranspose(m);
                     acc += XMVectorGetX(tv) + XMVectorGetY(tv) + XMVectorGetZ(tv) + XMVectorGetW(tv);
                     acc += XMVectorGetX(t.r[0]) + XMVectorGetY(t.r[1]) + XMVectorGetZ(t.r[2]) + XMVectorGetW(t.r[3]);
                 }
                 for (int i = 0; i < N_INV; ++i) {
                     XMMATRIX ma = XMLoadFloat4x4(&da[i]);
                     XMVECTOR det;
                     XMMATRIX inv = XMMatrixInverse(&det, ma);
                     acc += XMVectorGetX(inv.r[0]) + XMVectorGetY(inv.r[1]) + XMVectorGetZ(inv.r[2]) + XMVectorGetW(inv.r[3]);
                 }
                 fmbench::consume(acc);
             }},
#else
            {"directxmath", false, {}},
#endif
        },
        options.config);
}

FM_BENCH(Matrix, LookAtConstruction) {
    constexpr int N = 4096;
    auto rng = make_rng();

    std::vector<MMath::Vec4> eyes(N), targets(N), ups(N);
    for (int i = 0; i < N; ++i) {
        eyes[i] = MMath::Vec4{rng.uniform(-8.0f, 8.0f), rng.uniform(-8.0f, 8.0f), rng.uniform(2.0f, 15.0f), 1.0f};
        targets[i] = MMath::Vec4{rng.uniform(-1.0f, 1.0f), rng.uniform(-1.0f, 1.0f), rng.uniform(-1.0f, 1.0f), 1.0f};
        ups[i] = MMath::Vec4{0.0f, 1.0f, 0.0f, 0.0f};
    }

#if FM_HAVE_GLM
    std::vector<glm::vec3> geyes(N), gtargets(N), gups(N);
    for (int i = 0; i < N; ++i) {
        geyes[i] = glm::vec3(eyes[i].x, eyes[i].y, eyes[i].z);
        gtargets[i] = glm::vec3(targets[i].x, targets[i].y, targets[i].z);
        gups[i] = glm::vec3(0.0f, 1.0f, 0.0f);
    }
#endif

#if FM_HAVE_EIGEN
    std::vector<Eigen::Vector3f> eeyes(N), etargets(N);
    for (int i = 0; i < N; ++i) {
        eeyes[i] = Eigen::Vector3f(eyes[i].x, eyes[i].y, eyes[i].z);
        etargets[i] = Eigen::Vector3f(targets[i].x, targets[i].y, targets[i].z);
    }
#endif

#if FM_HAVE_DIRECTXMATH
    std::vector<DirectX::XMFLOAT3> deyes(N), dtargets(N);
    for (int i = 0; i < N; ++i) {
        deyes[i] = DirectX::XMFLOAT3(eyes[i].x, eyes[i].y, eyes[i].z);
        dtargets[i] = DirectX::XMFLOAT3(targets[i].x, targets[i].y, targets[i].z);
    }
#endif

    fmbench::run_comparison_case(
        "lookAt construction",
        static_cast<std::size_t>(N),
        {
            {"fast_math", true, [&]() {
                 double acc = 0.0;
                 for (int i = 0; i < N; ++i) {
                     const MMath::Mat4 view = MMath::mat4LookAt(eyes[i], targets[i], ups[i]);
                     acc += view.m[0] + view.m[5] + view.m[10];
                 }
                 fmbench::consume(acc);
             }},
#if FM_HAVE_GLM
            {"glm", true, [&]() {
                 double acc = 0.0;
                 for (int i = 0; i < N; ++i) {
                     const glm::mat4 view = glm::lookAt(geyes[i], gtargets[i], gups[i]);
                     acc += view[0][0] + view[1][1] + view[2][2];
                 }
                 fmbench::consume(acc);
             }},
#else
            {"glm", false, {}},
#endif
#if FM_HAVE_EIGEN
            {"eigen", true, [&]() {
                 auto make_view = [](const Eigen::Vector3f& eye, const Eigen::Vector3f& target) {
                     Eigen::Vector3f f = (target - eye).normalized();
                     Eigen::Vector3f up(0.0f, 1.0f, 0.0f);
                     Eigen::Vector3f s = f.cross(up).normalized();
                     Eigen::Vector3f u = s.cross(f);
                     Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
                     m.block<1, 3>(0, 0) = s.transpose();
                     m.block<1, 3>(1, 0) = u.transpose();
                     m.block<1, 3>(2, 0) = (-f).transpose();
                     m(0, 3) = -s.dot(eye);
                     m(1, 3) = -u.dot(eye);
                     m(2, 3) = f.dot(eye);
                     return m;
                 };

                 double acc = 0.0;
                 for (int i = 0; i < N; ++i) {
                     const Eigen::Matrix4f view = make_view(eeyes[i], etargets[i]);
                     acc += view(0, 0) + view(1, 1) + view(2, 2);
                 }
                 fmbench::consume(acc);
             }},
#else
            {"eigen", false, {}},
#endif
#if FM_HAVE_DIRECTXMATH
            {"directxmath", true, [&]() {
                 using namespace DirectX;
                 double acc = 0.0;
                 for (int i = 0; i < N; ++i) {
                     const XMVECTOR eye = XMLoadFloat3(&deyes[i]);
                     const XMVECTOR target = XMLoadFloat3(&dtargets[i]);
                     const XMVECTOR up = XMVectorSet(0.f, 1.f, 0.f, 0.f);
                     const XMMATRIX view = XMMatrixLookAtRH(eye, target, up);
                     acc += XMVectorGetX(view.r[0]) + XMVectorGetY(view.r[1]) + XMVectorGetZ(view.r[2]);
                 }
                 fmbench::consume(acc);
             }},
#else
            {"directxmath", false, {}},
#endif
        },
        options.config);
}

FM_BENCH(Matrix, PerspectiveConstruction) {
    constexpr int N = 4096;
    auto rng = make_rng();

    std::vector<float> fovy(N), aspect(N), near_z(N), far_z(N);
    for (int i = 0; i < N; ++i) {
        fovy[i] = rng.uniform(0.7f, 1.6f);
        aspect[i] = rng.uniform(1.2f, 2.4f);
        near_z[i] = rng.uniform(0.05f, 0.5f);
        far_z[i] = near_z[i] + rng.uniform(100.0f, 1500.0f);
    }

    fmbench::run_comparison_case(
        "perspective construction",
        static_cast<std::size_t>(N),
        {
            {"fast_math", true, [&]() {
                 double acc = 0.0;
                 for (int i = 0; i < N; ++i) {
                     const MMath::Mat4 proj = MMath::mat4Perspective(fovy[i], aspect[i], near_z[i], far_z[i]);
                     acc += proj.m[0] + proj.m[5] + proj.m[10];
                 }
                 fmbench::consume(acc);
             }},
#if FM_HAVE_GLM
            {"glm", true, [&]() {
                 double acc = 0.0;
                 for (int i = 0; i < N; ++i) {
                     const glm::mat4 proj = glm::perspectiveRH_ZO(fovy[i], aspect[i], near_z[i], far_z[i]);
                     acc += proj[0][0] + proj[1][1] + proj[2][2];
                 }
                 fmbench::consume(acc);
             }},
#else
            {"glm", false, {}},
#endif
#if FM_HAVE_EIGEN
            {"eigen", true, [&]() {
                 auto make_proj = [](float in_fovy, float in_aspect, float in_near, float in_far) {
                     const float y_scale = 1.0f / std::tan(in_fovy * 0.5f);
                     const float x_scale = y_scale / in_aspect;
                     Eigen::Matrix4f p = Eigen::Matrix4f::Zero();
                     p(0, 0) = x_scale;
                     p(1, 1) = y_scale;
                     p(2, 2) = in_far / (in_near - in_far);
                     p(2, 3) = (in_far * in_near) / (in_near - in_far);
                     p(3, 2) = -1.0f;
                     return p;
                 };

                 double acc = 0.0;
                 for (int i = 0; i < N; ++i) {
                     const Eigen::Matrix4f proj = make_proj(fovy[i], aspect[i], near_z[i], far_z[i]);
                     acc += proj(0, 0) + proj(1, 1) + proj(2, 2);
                 }
                 fmbench::consume(acc);
             }},
#else
            {"eigen", false, {}},
#endif
#if FM_HAVE_DIRECTXMATH
            {"directxmath", true, [&]() {
                 using namespace DirectX;
                 double acc = 0.0;
                 for (int i = 0; i < N; ++i) {
                     const XMMATRIX proj = XMMatrixPerspectiveFovRH(fovy[i], aspect[i], near_z[i], far_z[i]);
                     acc += XMVectorGetX(proj.r[0]) + XMVectorGetY(proj.r[1]) + XMVectorGetZ(proj.r[2]);
                 }
                 fmbench::consume(acc);
             }},
#else
            {"directxmath", false, {}},
#endif
        },
        options.config);
}
