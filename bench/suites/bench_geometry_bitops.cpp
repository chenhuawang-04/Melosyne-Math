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
    explicit Rng(uint32_t seed_) : eng(seed_) {}
    float uniform(float lo_, float hi_) {
        std::uniform_real_distribution<float> dist(lo_, hi_);
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

inline void transformPointScalarRef(
    const MMath::Mat4& m_,
    float x_,
    float y_,
    float z_,
    float& out_x_,
    float& out_y_,
    float& out_z_) noexcept {
    out_x_ = m_.m[0] * x_ + m_.m[4] * y_ + m_.m[8] * z_ + m_.m[12];
    out_y_ = m_.m[1] * x_ + m_.m[5] * y_ + m_.m[9] * z_ + m_.m[13];
    out_z_ = m_.m[2] * x_ + m_.m[6] * y_ + m_.m[10] * z_ + m_.m[14];
}

Aabb3Ref transformAabb3ScalarRef(const Aabb3Ref& box_, const MMath::Mat4& m_) noexcept {
    const float xs[2] = {box_.minx, box_.maxx};
    const float ys[2] = {box_.miny, box_.maxy};
    const float zs[2] = {box_.minz, box_.maxz};

    Aabb3Ref out{};
    bool first = true;

    for (float x : xs) {
        for (float y : ys) {
            for (float z : zs) {
                float tx = 0.0f, ty = 0.0f, tz = 0.0f;
                transformPointScalarRef(m_, x, y, z, tx, ty, tz);
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
    constexpr int n = 32768;
    Rng rng(0x0F0F0F0FU);

    std::vector<MMath::Aabb2> boxes(n);
    std::vector<MMath::Vec2> pts(n);
    std::vector<Aabb2Ref> refs(n);

    for (int i = 0; i < n; ++i) {
        const float x0 = rng.uniform(-50.0f, 10.0f);
        const float y0 = rng.uniform(-50.0f, 10.0f);
        const float x1 = x0 + rng.uniform(0.1f, 50.0f);
        const float y1 = y0 + rng.uniform(0.1f, 50.0f);
        boxes[i] = MMath::aabb2FromMinMax(MMath::Vec2{x0, y0}, MMath::Vec2{x1, y1});
        pts[i] = MMath::Vec2{rng.uniform(-60.0f, 60.0f), rng.uniform(-60.0f, 60.0f)};
        refs[i] = Aabb2Ref{x0, y0, x1, y1};
    }

    FmBench::runComparisonCase(
        "aabb2 contains + merge",
        static_cast<std::size_t>(n * 2),
        {
            {"fast_math", true, [&]() {
                 double acc = 0.0;
                 MMath::Aabb2 merged = boxes[0];
                 for (int i = 0; i < n; ++i) {
                     acc += MMath::aabb2ContainsPointAndMergeInPlace(merged, boxes[i], pts[i]) ? 1.0 : 0.0;
                 }
                 acc += MMath::aabb2Area(merged);
                 FmBench::consume(acc);
             }},
#if FM_HAVE_GLM
            {"glm", true, [&]() {
                 double acc = 0.0;
                 glm::vec2 mn(refs[0].minx, refs[0].miny);
                 glm::vec2 mx(refs[0].maxx, refs[0].maxy);
                 for (int i = 0; i < n; ++i) {
                     glm::vec2 p(pts[i].x, pts[i].y);
                     glm::vec2 bmn(refs[i].minx, refs[i].miny);
                     glm::vec2 bmx(refs[i].maxx, refs[i].maxy);
                     const bool c = p.x >= bmn.x && p.x <= bmx.x && p.y >= bmn.y && p.y <= bmx.y;
                     acc += c ? 1.0 : 0.0;
                     mn = glm::min(mn, bmn);
                     mx = glm::max(mx, bmx);
                 }
                 acc += (mx.x - mn.x) * (mx.y - mn.y);
                 FmBench::consume(acc);
             }},
#else
            {"glm", false, {}},
#endif
#if FM_HAVE_EIGEN
            {"eigen", true, [&]() {
                 double acc = 0.0;
                 Eigen::Vector2f mn(refs[0].minx, refs[0].miny);
                 Eigen::Vector2f mx(refs[0].maxx, refs[0].maxy);
                 for (int i = 0; i < n; ++i) {
                     Eigen::Vector2f p(pts[i].x, pts[i].y);
                     Eigen::Vector2f bmn(refs[i].minx, refs[i].miny);
                     Eigen::Vector2f bmx(refs[i].maxx, refs[i].maxy);
                     const bool c = (p.array() >= bmn.array()).all() && (p.array() <= bmx.array()).all();
                     acc += c ? 1.0 : 0.0;
                     mn = mn.cwiseMin(bmn);
                     mx = mx.cwiseMax(bmx);
                 }
                 acc += (mx.x() - mn.x()) * (mx.y() - mn.y());
                 FmBench::consume(acc);
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
                 for (int i = 0; i < n; ++i) {
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
                 FmBench::consume(acc);
             }},
#else
            {"directxmath", false, {}},
#endif
        },
        options.config);
}

FM_BENCH(Geometry, Aabb2ContainsOnly) {
    constexpr int n = 32768;
    Rng rng(0x0F0F0F0FU);

    std::vector<MMath::Aabb2> boxes(n);
    std::vector<MMath::Vec2> pts(n);
    std::vector<Aabb2Ref> refs(n);

    for (int i = 0; i < n; ++i) {
        const float x0 = rng.uniform(-50.0f, 10.0f);
        const float y0 = rng.uniform(-50.0f, 10.0f);
        const float x1 = x0 + rng.uniform(0.1f, 50.0f);
        const float y1 = y0 + rng.uniform(0.1f, 50.0f);
        boxes[i] = MMath::aabb2FromMinMax(MMath::Vec2{x0, y0}, MMath::Vec2{x1, y1});
        pts[i] = MMath::Vec2{rng.uniform(-60.0f, 60.0f), rng.uniform(-60.0f, 60.0f)};
        refs[i] = Aabb2Ref{x0, y0, x1, y1};
    }

    FmBench::runComparisonCase(
        "aabb2 contains only",
        static_cast<std::size_t>(n),
        {
            {"fast_math", true, [&]() {
                 double acc = 0.0;
                 for (int i = 0; i < n; ++i) {
                     acc += MMath::aabb2ContainsPoint(boxes[i], pts[i]) ? 1.0 : 0.0;
                 }
                 FmBench::consume(acc);
             }},
#if FM_HAVE_GLM
            {"glm", true, [&]() {
                 double acc = 0.0;
                 for (int i = 0; i < n; ++i) {
                     glm::vec2 p(pts[i].x, pts[i].y);
                     glm::vec2 bmn(refs[i].minx, refs[i].miny);
                     glm::vec2 bmx(refs[i].maxx, refs[i].maxy);
                     const bool c = p.x >= bmn.x && p.x <= bmx.x && p.y >= bmn.y && p.y <= bmx.y;
                     acc += c ? 1.0 : 0.0;
                 }
                 FmBench::consume(acc);
             }},
#else
            {"glm", false, {}},
#endif
#if FM_HAVE_EIGEN
            {"eigen", true, [&]() {
                 double acc = 0.0;
                 for (int i = 0; i < n; ++i) {
                     Eigen::Vector2f p(pts[i].x, pts[i].y);
                     Eigen::Vector2f bmn(refs[i].minx, refs[i].miny);
                     Eigen::Vector2f bmx(refs[i].maxx, refs[i].maxy);
                     const bool c = (p.array() >= bmn.array()).all() && (p.array() <= bmx.array()).all();
                     acc += c ? 1.0 : 0.0;
                 }
                 FmBench::consume(acc);
             }},
#else
            {"eigen", false, {}},
#endif
#if FM_HAVE_DIRECTXMATH
            {"directxmath", true, [&]() {
                 using namespace DirectX;
                 double acc = 0.0;
                 for (int i = 0; i < n; ++i) {
                     XMVECTOR p = XMVectorSet(pts[i].x, pts[i].y, 0.0f, 0.0f);
                     XMVECTOR bmn = XMVectorSet(refs[i].minx, refs[i].miny, 0.0f, 0.0f);
                     XMVECTOR bmx = XMVectorSet(refs[i].maxx, refs[i].maxy, 0.0f, 0.0f);
                     XMVECTOR c1 = XMVectorGreaterOrEqual(p, bmn);
                     XMVECTOR c2 = XMVectorLessOrEqual(p, bmx);
                     XMVECTOR c = XMVectorAndInt(c1, c2);
                     acc += (XMVectorGetX(c) != 0.0f && XMVectorGetY(c) != 0.0f) ? 1.0 : 0.0;
                 }
                 FmBench::consume(acc);
             }},
#else
            {"directxmath", false, {}},
#endif
        },
        options.config);
}

FM_BENCH(Geometry, Aabb2MergeOnly) {
    constexpr int n = 32768;
    Rng rng(0x0F0F0F0FU);

    std::vector<MMath::Aabb2> boxes(n);
    std::vector<Aabb2Ref> refs(n);

    for (int i = 0; i < n; ++i) {
        const float x0 = rng.uniform(-50.0f, 10.0f);
        const float y0 = rng.uniform(-50.0f, 10.0f);
        const float x1 = x0 + rng.uniform(0.1f, 50.0f);
        const float y1 = y0 + rng.uniform(0.1f, 50.0f);
        boxes[i] = MMath::aabb2FromMinMax(MMath::Vec2{x0, y0}, MMath::Vec2{x1, y1});
        refs[i] = Aabb2Ref{x0, y0, x1, y1};
    }

    FmBench::runComparisonCase(
        "aabb2 merge only",
        static_cast<std::size_t>(n),
        {
            {"fast_math", true, [&]() {
                 MMath::Aabb2 merged = boxes[0];
                 for (int i = 1; i < n; ++i) {
                     MMath::aabb2MergeInPlace(merged, boxes[i]);
                 }
                 const double area = static_cast<double>(MMath::aabb2Area(merged));
                 FmBench::consume(area);
             }},
#if FM_HAVE_GLM
            {"glm", true, [&]() {
                 glm::vec2 mn(refs[0].minx, refs[0].miny);
                 glm::vec2 mx(refs[0].maxx, refs[0].maxy);
                 for (int i = 1; i < n; ++i) {
                     glm::vec2 bmn(refs[i].minx, refs[i].miny);
                     glm::vec2 bmx(refs[i].maxx, refs[i].maxy);
                     mn = glm::min(mn, bmn);
                     mx = glm::max(mx, bmx);
                 }
                FmBench::consume(static_cast<double>((mx.x - mn.x) * (mx.y - mn.y)));
             }},
#else
            {"glm", false, {}},
#endif
#if FM_HAVE_EIGEN
            {"eigen", true, [&]() {
                 Eigen::Vector2f mn(refs[0].minx, refs[0].miny);
                 Eigen::Vector2f mx(refs[0].maxx, refs[0].maxy);
                 for (int i = 1; i < n; ++i) {
                     Eigen::Vector2f bmn(refs[i].minx, refs[i].miny);
                     Eigen::Vector2f bmx(refs[i].maxx, refs[i].maxy);
                     mn = mn.cwiseMin(bmn);
                     mx = mx.cwiseMax(bmx);
                 }
                 FmBench::consume(static_cast<double>((mx.x() - mn.x()) * (mx.y() - mn.y())));
             }},
#else
            {"eigen", false, {}},
#endif
#if FM_HAVE_DIRECTXMATH
            {"directxmath", true, [&]() {
                 using namespace DirectX;
                 XMVECTOR mn = XMVectorSet(refs[0].minx, refs[0].miny, 0.0f, 0.0f);
                 XMVECTOR mx = XMVectorSet(refs[0].maxx, refs[0].maxy, 0.0f, 0.0f);
                 for (int i = 1; i < n; ++i) {
                     XMVECTOR bmn = XMVectorSet(refs[i].minx, refs[i].miny, 0.0f, 0.0f);
                     XMVECTOR bmx = XMVectorSet(refs[i].maxx, refs[i].maxy, 0.0f, 0.0f);
                     mn = XMVectorMin(mn, bmn);
                     mx = XMVectorMax(mx, bmx);
                 }
                 const float w = XMVectorGetX(mx) - XMVectorGetX(mn);
                 const float h = XMVectorGetY(mx) - XMVectorGetY(mn);
                 FmBench::consume(static_cast<double>(w * h));
             }},
#else
            {"directxmath", false, {}},
#endif
        },
        options.config);
}

FM_BENCH(Geometry, Aabb3OverlapTransform) {
    constexpr int n = 16384;
    Rng rng(0x22223333U);

    std::vector<MMath::Aabb3> a(n), b(n);
    std::vector<MMath::Mat4> tr(n);
    std::vector<Aabb3Ref> ra(n), rb(n);

    for (int i = 0; i < n; ++i) {
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

    FmBench::runComparisonCase(
        "aabb3 overlap + transform",
        static_cast<std::size_t>(n * 2),
        {
            {"fast_math", true, [&]() {
                 double acc = 0.0;
                 for (int i = 0; i < n; ++i) {
                     acc += MMath::aabb3Overlap(a[i], b[i]) ? 1.0 : 0.0;
                     auto t = MMath::aabb3Transform(a[i], tr[i]);
                     acc += MMath::aabb3Volume(t) * 1e-6;
                 }
                 FmBench::consume(acc);
             }},
            {"scalar_ref", true, [&]() {
                 double acc = 0.0;
                 for (int i = 0; i < n; ++i) {
                     const bool overlap = !(ra[i].maxx < rb[i].minx || rb[i].maxx < ra[i].minx ||
                                            ra[i].maxy < rb[i].miny || rb[i].maxy < ra[i].miny ||
                                            ra[i].maxz < rb[i].minz || rb[i].maxz < ra[i].minz);
                     acc += overlap ? 1.0 : 0.0;

                     const Aabb3Ref ta = transformAabb3ScalarRef(ra[i], tr[i]);
                     const float w = ta.maxx - ta.minx;
                     const float h = ta.maxy - ta.miny;
                     const float d = ta.maxz - ta.minz;
                     acc += (w * h * d) * 1e-6;
                 }
                 FmBench::consume(acc);
             }},
        },
        options.config);
}

FM_BENCH(BitOps, ThroughputScalarVsOptimized) {
    using namespace MMath;
    using namespace MMath::BitOps;

    constexpr std::size_t n = 4096;
    constexpr int rounds = 2048;

    BitSet<n> a{}, b{};
    for (std::size_t i = 0; i < n; i += 3) set(a, i);
    for (std::size_t i = 0; i < n; i += 5) set(b, i);

    FmBench::runComparisonCase(
        "bitset and+count+rank kernel",
        static_cast<std::size_t>(rounds * 3),
        {
            {"scalar", true, [&]() {
                 double acc = 0.0;
                 BitSet<n> x = a;
                 for (int r = 0; r < rounds; ++r) {
                     bitwiseAnd(x, b);
                     acc += static_cast<double>(popcount(x));
                     acc += static_cast<double>(rank(x, static_cast<std::size_t>((r * 17) % static_cast<int>(n))));
                     flipRange(
                         x,
                         static_cast<std::size_t>((r * 13) % static_cast<int>(n)),
                         static_cast<std::size_t>(((r * 13) % static_cast<int>(n)) + 7));
                 }
                 FmBench::consume(acc);
             }},
            {"scalar_fused", true, [&]() {
                 double acc = 0.0;
                 BitSet<n> x = a;
                 for (int r = 0; r < rounds; ++r) {
                     acc += static_cast<double>(bitwiseAndCount(x, b));
                     acc += static_cast<double>(rank(x, static_cast<std::size_t>((r * 17) % static_cast<int>(n))));
                     flipRange(
                         x,
                         static_cast<std::size_t>((r * 13) % static_cast<int>(n)),
                         static_cast<std::size_t>(((r * 13) % static_cast<int>(n)) + 7));
                 }
                 FmBench::consume(acc);
             }},
#if defined(__AVX2__) || defined(__SSE4_1__)
            {"optimized", true, [&]() {
                 double acc = 0.0;
                 BitSet<n> x = a;
                 for (int r = 0; r < rounds; ++r) {
                     acc += static_cast<double>(bitwiseAndCountOptimized(x, b));
                     acc += static_cast<double>(rankOptimized(x, static_cast<std::size_t>((r * 17) % static_cast<int>(n))));
                     flipRangeOptimized(
                         x,
                         static_cast<std::size_t>((r * 13) % static_cast<int>(n)),
                         static_cast<std::size_t>(((r * 13) % static_cast<int>(n)) + 7));
                 }
                 FmBench::consume(acc);
             }},
#else
            {"optimized", false, {}},
#endif
        },
        options.config);
}
