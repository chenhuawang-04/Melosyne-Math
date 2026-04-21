#include "bench/framework/benchmark_framework.h"
#include "bench/adapters/external_backends.h"

#include "fast_math/aabb2.h"
#include "fast_math/aabb3.h"
#include "fast_math/bit_ops.h"
#include "fast_math/mat3.h"
#include "fast_math/mat4.h"

#include <random>
#include <vector>

namespace {

struct Rng {
    explicit Rng(uint32_t seed) : eng(seed) {}
    float uniform(float lo, float hi) {
        std::uniform_real_distribution<float> dist(lo, hi);
        return dist(eng);
    }
    std::mt19937 eng;
};

struct Aabb2Ref {
    float minx, miny, maxx, maxy;
};

struct Aabb3Ref {
    float minx, miny, minz, maxx, maxy, maxz;
};

inline void transform_point_scalar_ref(
    const MMath::Mat4& m,
    float x,
    float y,
    float z,
    float& out_x,
    float& out_y,
    float& out_z) noexcept {
    out_x = m.m[0] * x + m.m[4] * y + m.m[8] * z + m.m[12];
    out_y = m.m[1] * x + m.m[5] * y + m.m[9] * z + m.m[13];
    out_z = m.m[2] * x + m.m[6] * y + m.m[10] * z + m.m[14];
}

Aabb3Ref transform_aabb3_scalar_ref(const Aabb3Ref& box, const MMath::Mat4& m) noexcept {
    const float xs[2] = {box.minx, box.maxx};
    const float ys[2] = {box.miny, box.maxy};
    const float zs[2] = {box.minz, box.maxz};

    Aabb3Ref out{};
    bool first = true;

    for (float x : xs) {
        for (float y : ys) {
            for (float z : zs) {
                float tx = 0.0f, ty = 0.0f, tz = 0.0f;
                transform_point_scalar_ref(m, x, y, z, tx, ty, tz);
                if (first) {
                    out = Aabb3Ref{tx, ty, tz, tx, ty, tz};
                    first = false;
                } else {
                    out.minx = (tx < out.minx) ? tx : out.minx;
                    out.miny = (ty < out.miny) ? ty : out.miny;
                    out.minz = (tz < out.minz) ? tz : out.minz;
                    out.maxx = (tx > out.maxx) ? tx : out.maxx;
                    out.maxy = (ty > out.maxy) ? ty : out.maxy;
                    out.maxz = (tz > out.maxz) ? tz : out.maxz;
                }
            }
        }
    }

    return out;
}

} // namespace

FM_BENCH(Geometry, Aabb2ContainsMerge) {
    constexpr int N = 32768;
    Rng rng(0x0F0F0F0FU);

    std::vector<MMath::Aabb2> boxes(N);
    std::vector<MMath::Vec2> pts(N);
    std::vector<Aabb2Ref> refs(N);

    for (int i = 0; i < N; ++i) {
        const float x0 = rng.uniform(-50.0f, 10.0f);
        const float y0 = rng.uniform(-50.0f, 10.0f);
        const float x1 = x0 + rng.uniform(0.1f, 50.0f);
        const float y1 = y0 + rng.uniform(0.1f, 50.0f);
        boxes[i] = MMath::aabb2FromMinMax(MMath::Vec2{x0, y0}, MMath::Vec2{x1, y1});
        pts[i] = MMath::Vec2{rng.uniform(-60.0f, 60.0f), rng.uniform(-60.0f, 60.0f)};
        refs[i] = Aabb2Ref{x0, y0, x1, y1};
    }

    fmbench::run_comparison_case(
        "aabb2 contains + merge",
        static_cast<std::size_t>(N * 2),
        {
            {"fast_math", true, [&]() {
                 double acc = 0.0;
                 MMath::Aabb2 merged = boxes[0];
                 for (int i = 0; i < N; ++i) {
                     acc += MMath::aabb2ContainsPointAndMergeInPlace(merged, boxes[i], pts[i]) ? 1.0 : 0.0;
                 }
                 acc += MMath::aabb2Area(merged);
                 fmbench::consume(acc);
             }},
#if FM_HAVE_GLM
            {"glm", true, [&]() {
                 double acc = 0.0;
                 glm::vec2 mn(refs[0].minx, refs[0].miny);
                 glm::vec2 mx(refs[0].maxx, refs[0].maxy);
                 for (int i = 0; i < N; ++i) {
                     glm::vec2 p(pts[i].x, pts[i].y);
                     glm::vec2 bmn(refs[i].minx, refs[i].miny);
                     glm::vec2 bmx(refs[i].maxx, refs[i].maxy);
                     const bool c = p.x >= bmn.x && p.x <= bmx.x && p.y >= bmn.y && p.y <= bmx.y;
                     acc += c ? 1.0 : 0.0;
                     mn = glm::min(mn, bmn);
                     mx = glm::max(mx, bmx);
                 }
                 acc += (mx.x - mn.x) * (mx.y - mn.y);
                 fmbench::consume(acc);
             }},
#else
            {"glm", false, {}},
#endif
#if FM_HAVE_EIGEN
            {"eigen", true, [&]() {
                 double acc = 0.0;
                 Eigen::Vector2f mn(refs[0].minx, refs[0].miny);
                 Eigen::Vector2f mx(refs[0].maxx, refs[0].maxy);
                 for (int i = 0; i < N; ++i) {
                     Eigen::Vector2f p(pts[i].x, pts[i].y);
                     Eigen::Vector2f bmn(refs[i].minx, refs[i].miny);
                     Eigen::Vector2f bmx(refs[i].maxx, refs[i].maxy);
                     const bool c = (p.array() >= bmn.array()).all() && (p.array() <= bmx.array()).all();
                     acc += c ? 1.0 : 0.0;
                     mn = mn.cwiseMin(bmn);
                     mx = mx.cwiseMax(bmx);
                 }
                 acc += (mx.x() - mn.x()) * (mx.y() - mn.y());
                 fmbench::consume(acc);
             }},
#else
            {"eigen", false, {}},
#endif
#if FM_HAVE_DIRECTXMATH
            {"directxmath", true, [&]() {
                 using namespace DirectX;
                 double acc = 0.0;
                 XMVECTOR mn = XMVectorSet(refs[0].minx, refs[0].miny, 0.0f, 0.0f);
                 XMVECTOR mx = XMVectorSet(refs[0].maxx, refs[0].maxy, 0.0f, 0.0f);
                 for (int i = 0; i < N; ++i) {
                     XMVECTOR p = XMVectorSet(pts[i].x, pts[i].y, 0.0f, 0.0f);
                     XMVECTOR bmn = XMVectorSet(refs[i].minx, refs[i].miny, 0.0f, 0.0f);
                     XMVECTOR bmx = XMVectorSet(refs[i].maxx, refs[i].maxy, 0.0f, 0.0f);
                     XMVECTOR c1 = XMVectorGreaterOrEqual(p, bmn);
                     XMVECTOR c2 = XMVectorLessOrEqual(p, bmx);
                     XMVECTOR c = XMVectorAndInt(c1, c2);
                     const bool inside = XMVectorGetX(c) != 0.0f && XMVectorGetY(c) != 0.0f;
                     acc += inside ? 1.0 : 0.0;
                     mn = XMVectorMin(mn, bmn);
                     mx = XMVectorMax(mx, bmx);
                 }
                 const float w = XMVectorGetX(mx) - XMVectorGetX(mn);
                 const float h = XMVectorGetY(mx) - XMVectorGetY(mn);
                 acc += w * h;
                 fmbench::consume(acc);
             }},
#else
            {"directxmath", false, {}},
#endif
        },
        options.config);
}

FM_BENCH(Geometry, Aabb2ContainsOnly) {
    constexpr int N = 32768;
    Rng rng(0x0F0F0F0FU);

    std::vector<MMath::Aabb2> boxes(N);
    std::vector<MMath::Vec2> pts(N);
    std::vector<Aabb2Ref> refs(N);

    for (int i = 0; i < N; ++i) {
        const float x0 = rng.uniform(-50.0f, 10.0f);
        const float y0 = rng.uniform(-50.0f, 10.0f);
        const float x1 = x0 + rng.uniform(0.1f, 50.0f);
        const float y1 = y0 + rng.uniform(0.1f, 50.0f);
        boxes[i] = MMath::aabb2FromMinMax(MMath::Vec2{x0, y0}, MMath::Vec2{x1, y1});
        pts[i] = MMath::Vec2{rng.uniform(-60.0f, 60.0f), rng.uniform(-60.0f, 60.0f)};
        refs[i] = Aabb2Ref{x0, y0, x1, y1};
    }

    fmbench::run_comparison_case(
        "aabb2 contains only",
        static_cast<std::size_t>(N),
        {
            {"fast_math", true, [&]() {
                 double acc = 0.0;
                 for (int i = 0; i < N; ++i) {
                     acc += MMath::aabb2ContainsPoint(boxes[i], pts[i]) ? 1.0 : 0.0;
                 }
                 fmbench::consume(acc);
             }},
#if FM_HAVE_GLM
            {"glm", true, [&]() {
                 double acc = 0.0;
                 for (int i = 0; i < N; ++i) {
                     glm::vec2 p(pts[i].x, pts[i].y);
                     glm::vec2 bmn(refs[i].minx, refs[i].miny);
                     glm::vec2 bmx(refs[i].maxx, refs[i].maxy);
                     const bool c = p.x >= bmn.x && p.x <= bmx.x && p.y >= bmn.y && p.y <= bmx.y;
                     acc += c ? 1.0 : 0.0;
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
                     Eigen::Vector2f p(pts[i].x, pts[i].y);
                     Eigen::Vector2f bmn(refs[i].minx, refs[i].miny);
                     Eigen::Vector2f bmx(refs[i].maxx, refs[i].maxy);
                     const bool c = (p.array() >= bmn.array()).all() && (p.array() <= bmx.array()).all();
                     acc += c ? 1.0 : 0.0;
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
                     XMVECTOR p = XMVectorSet(pts[i].x, pts[i].y, 0.0f, 0.0f);
                     XMVECTOR bmn = XMVectorSet(refs[i].minx, refs[i].miny, 0.0f, 0.0f);
                     XMVECTOR bmx = XMVectorSet(refs[i].maxx, refs[i].maxy, 0.0f, 0.0f);
                     XMVECTOR c1 = XMVectorGreaterOrEqual(p, bmn);
                     XMVECTOR c2 = XMVectorLessOrEqual(p, bmx);
                     XMVECTOR c = XMVectorAndInt(c1, c2);
                     acc += (XMVectorGetX(c) != 0.0f && XMVectorGetY(c) != 0.0f) ? 1.0 : 0.0;
                 }
                 fmbench::consume(acc);
             }},
#else
            {"directxmath", false, {}},
#endif
        },
        options.config);
}

FM_BENCH(Geometry, Aabb2MergeOnly) {
    constexpr int N = 32768;
    Rng rng(0x0F0F0F0FU);

    std::vector<MMath::Aabb2> boxes(N);
    std::vector<Aabb2Ref> refs(N);

    for (int i = 0; i < N; ++i) {
        const float x0 = rng.uniform(-50.0f, 10.0f);
        const float y0 = rng.uniform(-50.0f, 10.0f);
        const float x1 = x0 + rng.uniform(0.1f, 50.0f);
        const float y1 = y0 + rng.uniform(0.1f, 50.0f);
        boxes[i] = MMath::aabb2FromMinMax(MMath::Vec2{x0, y0}, MMath::Vec2{x1, y1});
        refs[i] = Aabb2Ref{x0, y0, x1, y1};
    }

    fmbench::run_comparison_case(
        "aabb2 merge only",
        static_cast<std::size_t>(N),
        {
            {"fast_math", true, [&]() {
                 MMath::Aabb2 merged = boxes[0];
                 for (int i = 1; i < N; ++i) {
                     MMath::aabb2MergeInPlace(merged, boxes[i]);
                 }
                 const double area = static_cast<double>(MMath::aabb2Area(merged));
                 fmbench::consume(area);
             }},
#if FM_HAVE_GLM
            {"glm", true, [&]() {
                 glm::vec2 mn(refs[0].minx, refs[0].miny);
                 glm::vec2 mx(refs[0].maxx, refs[0].maxy);
                 for (int i = 1; i < N; ++i) {
                     glm::vec2 bmn(refs[i].minx, refs[i].miny);
                     glm::vec2 bmx(refs[i].maxx, refs[i].maxy);
                     mn = glm::min(mn, bmn);
                     mx = glm::max(mx, bmx);
                 }
                 fmbench::consume(static_cast<double>((mx.x - mn.x) * (mx.y - mn.y)));
             }},
#else
            {"glm", false, {}},
#endif
#if FM_HAVE_EIGEN
            {"eigen", true, [&]() {
                 Eigen::Vector2f mn(refs[0].minx, refs[0].miny);
                 Eigen::Vector2f mx(refs[0].maxx, refs[0].maxy);
                 for (int i = 1; i < N; ++i) {
                     Eigen::Vector2f bmn(refs[i].minx, refs[i].miny);
                     Eigen::Vector2f bmx(refs[i].maxx, refs[i].maxy);
                     mn = mn.cwiseMin(bmn);
                     mx = mx.cwiseMax(bmx);
                 }
                 fmbench::consume(static_cast<double>((mx.x() - mn.x()) * (mx.y() - mn.y())));
             }},
#else
            {"eigen", false, {}},
#endif
#if FM_HAVE_DIRECTXMATH
            {"directxmath", true, [&]() {
                 using namespace DirectX;
                 XMVECTOR mn = XMVectorSet(refs[0].minx, refs[0].miny, 0.0f, 0.0f);
                 XMVECTOR mx = XMVectorSet(refs[0].maxx, refs[0].maxy, 0.0f, 0.0f);
                 for (int i = 1; i < N; ++i) {
                     XMVECTOR bmn = XMVectorSet(refs[i].minx, refs[i].miny, 0.0f, 0.0f);
                     XMVECTOR bmx = XMVectorSet(refs[i].maxx, refs[i].maxy, 0.0f, 0.0f);
                     mn = XMVectorMin(mn, bmn);
                     mx = XMVectorMax(mx, bmx);
                 }
                 const float w = XMVectorGetX(mx) - XMVectorGetX(mn);
                 const float h = XMVectorGetY(mx) - XMVectorGetY(mn);
                 fmbench::consume(static_cast<double>(w * h));
             }},
#else
            {"directxmath", false, {}},
#endif
        },
        options.config);
}

FM_BENCH(Geometry, Aabb3OverlapTransform) {
    constexpr int N = 16384;
    Rng rng(0x22223333U);

    std::vector<MMath::Aabb3> a(N), b(N);
    std::vector<MMath::Mat4> tr(N);
    std::vector<Aabb3Ref> ra(N), rb(N);

    for (int i = 0; i < N; ++i) {
        float ax0 = rng.uniform(-40.0f, 10.0f), ay0 = rng.uniform(-40.0f, 10.0f), az0 = rng.uniform(-40.0f, 10.0f);
        float ax1 = ax0 + rng.uniform(0.1f, 40.0f), ay1 = ay0 + rng.uniform(0.1f, 40.0f), az1 = az0 + rng.uniform(0.1f, 40.0f);
        float bx0 = rng.uniform(-40.0f, 10.0f), by0 = rng.uniform(-40.0f, 10.0f), bz0 = rng.uniform(-40.0f, 10.0f);
        float bx1 = bx0 + rng.uniform(0.1f, 40.0f), by1 = by0 + rng.uniform(0.1f, 40.0f), bz1 = bz0 + rng.uniform(0.1f, 40.0f);

        a[i] = MMath::aabb3FromMinMax(MMath::Vec3{ax0, ay0, az0}, MMath::Vec3{ax1, ay1, az1});
        b[i] = MMath::aabb3FromMinMax(MMath::Vec3{bx0, by0, bz0}, MMath::Vec3{bx1, by1, bz1});
        tr[i] = MMath::mat4Translation(rng.uniform(-3.0f, 3.0f), rng.uniform(-3.0f, 3.0f), rng.uniform(-3.0f, 3.0f));
        ra[i] = Aabb3Ref{ax0, ay0, az0, ax1, ay1, az1};
        rb[i] = Aabb3Ref{bx0, by0, bz0, bx1, by1, bz1};
    }

    fmbench::run_comparison_case(
        "aabb3 overlap + transform",
        static_cast<std::size_t>(N * 2),
        {
            {"fast_math", true, [&]() {
                 double acc = 0.0;
                 for (int i = 0; i < N; ++i) {
                     acc += MMath::aabb3Overlap(a[i], b[i]) ? 1.0 : 0.0;
                     auto t = MMath::aabb3Transform(a[i], tr[i]);
                     acc += MMath::aabb3Volume(t) * 1e-6;
                 }
                 fmbench::consume(acc);
             }},
            {"scalar_ref", true, [&]() {
                 double acc = 0.0;
                 for (int i = 0; i < N; ++i) {
                     const bool overlap = !(ra[i].maxx < rb[i].minx || rb[i].maxx < ra[i].minx ||
                                            ra[i].maxy < rb[i].miny || rb[i].maxy < ra[i].miny ||
                                            ra[i].maxz < rb[i].minz || rb[i].maxz < ra[i].minz);
                     acc += overlap ? 1.0 : 0.0;

                     const Aabb3Ref ta = transform_aabb3_scalar_ref(ra[i], tr[i]);
                     const float w = ta.maxx - ta.minx;
                     const float h = ta.maxy - ta.miny;
                     const float d = ta.maxz - ta.minz;
                     acc += (w * h * d) * 1e-6;
                 }
                 fmbench::consume(acc);
             }},
        },
        options.config);
}

FM_BENCH(BitOps, ThroughputScalarVsOptimized) {
    using namespace Melosyne;
    using namespace Melosyne::BitOps;

    constexpr int N = 4096;
    constexpr int ROUNDS = 2048;

    BitSet<N> a{}, b{};
    for (int i = 0; i < N; i += 3) set(a, i);
    for (int i = 0; i < N; i += 5) set(b, i);

    fmbench::run_comparison_case(
        "bitset and+count+rank kernel",
        static_cast<std::size_t>(ROUNDS * 3),
        {
            {"scalar", true, [&]() {
                 double acc = 0.0;
                 BitSet<N> x = a;
                 for (int r = 0; r < ROUNDS; ++r) {
                     bitwiseAnd(x, b);
                     acc += static_cast<double>(popcount(x));
                     acc += static_cast<double>(rank(x, (r * 17) % N));
                     flipRange(x, (r * 13) % N, ((r * 13) % N) + 7);
                 }
                 fmbench::consume(acc);
             }},
#if defined(__AVX2__) || defined(__SSE4_1__)
            {"optimized", true, [&]() {
                 double acc = 0.0;
                 BitSet<N> x = a;
                 for (int r = 0; r < ROUNDS; ++r) {
                     bitwiseAndOptimized(x, b);
                     acc += static_cast<double>(popcountOptimized(x));
                     acc += static_cast<double>(rankOptimized(x, (r * 17) % N));
                     flipRangeOptimized(x, (r * 13) % N, ((r * 13) % N) + 7);
                 }
                 fmbench::consume(acc);
             }},
#else
            {"optimized", false, {}},
#endif
        },
        options.config);
}
