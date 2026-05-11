#include "test/framework/test_framework.h"
#include "test/framework/test_utils.h"

#include "fast_math/aabb2.h"
#include "fast_math/aabb3.h"

#include <algorithm>
#include <array>

namespace {

bool ref_contains_2d(const MMath::Aabb2& a, const MMath::Vec2& p) {
    auto mn = MMath::aabb2Min(a);
    auto mx = MMath::aabb2Max(a);
    return p.x >= mn.x && p.x <= mx.x && p.y >= mn.y && p.y <= mx.y;
}

bool ref_overlap_3d(const MMath::Aabb3& a, const MMath::Aabb3& b) {
    auto amin = MMath::aabb3Min(a);
    auto amax = MMath::aabb3Max(a);
    auto bmin = MMath::aabb3Min(b);
    auto bmax = MMath::aabb3Max(b);
    return !(amax.x < bmin.x || bmax.x < amin.x ||
             amax.y < bmin.y || bmax.y < amin.y ||
             amax.z < bmin.z || bmax.z < amin.z);
}

} // namespace

FM_TEST(Aabb2, BasicProperties) {
    const MMath::Aabb2 box = MMath::aabb2FromMinMax(MMath::Vec2{-1.0f, -2.0f}, MMath::Vec2{3.0f, 4.0f});

    FM_REQUIRE(MMath::aabb2IsValid(box));
    FM_REQUIRE_VEC2_NEAR(MMath::aabb2Min(box), (MMath::Vec2{-1.0f, -2.0f}), 1e-6f);
    FM_REQUIRE_VEC2_NEAR(MMath::aabb2Max(box), (MMath::Vec2{3.0f, 4.0f}), 1e-6f);
    FM_REQUIRE_VEC2_NEAR(MMath::aabb2Center(box), (MMath::Vec2{1.0f, 1.0f}), 1e-6f);
    FM_REQUIRE_VEC2_NEAR(MMath::aabb2Extents(box), (MMath::Vec2{2.0f, 3.0f}), 1e-6f);

    FM_REQUIRE_NEAR(MMath::aabb2Width(box), 4.0f, 1e-6f);
    FM_REQUIRE_NEAR(MMath::aabb2Height(box), 6.0f, 1e-6f);
    FM_REQUIRE_NEAR(MMath::aabb2Area(box), 24.0f, 1e-6f);
    FM_REQUIRE_NEAR(MMath::aabb2Perimeter(box), 20.0f, 1e-6f);

    FM_REQUIRE(MMath::aabb2ContainsPoint(box, MMath::Vec2{0.0f, 0.0f}));
    FM_REQUIRE(!MMath::aabb2ContainsPoint(box, MMath::Vec2{-2.0f, 0.0f}));
}

FM_TEST(Aabb2, RandomizedContainmentAndMerge) {
    fmtest::Rng rng(0xABCDEF01U);

    for (int i = 0; i < 5000; ++i) {
        MMath::Vec2 a{rng.uniform(-20.0f, 0.0f), rng.uniform(-20.0f, 0.0f)};
        MMath::Vec2 b{rng.uniform(0.0f, 20.0f), rng.uniform(0.0f, 20.0f)};
        MMath::Aabb2 box = MMath::aabb2FromMinMax(a, b);

        MMath::Vec2 p{rng.uniform(-30.0f, 30.0f), rng.uniform(-30.0f, 30.0f)};
        FM_REQUIRE(MMath::aabb2ContainsPoint(box, p) == ref_contains_2d(box, p));

        MMath::Vec2 c{rng.uniform(-20.0f, 0.0f), rng.uniform(-20.0f, 0.0f)};
        MMath::Vec2 d{rng.uniform(0.0f, 20.0f), rng.uniform(0.0f, 20.0f)};
        MMath::Aabb2 box2 = MMath::aabb2FromMinMax(c, d);

        MMath::Aabb2 uni = MMath::aabb2Merge(box, box2);
        FM_REQUIRE(MMath::aabb2Contains(uni, box));
        FM_REQUIRE(MMath::aabb2Contains(uni, box2));

        MMath::Aabb2 uni_inplace = box;
        MMath::aabb2MergeInPlace(uni_inplace, box2);
        FM_REQUIRE(MMath::aabb2Equal(uni, uni_inplace));

        MMath::Aabb2 fused_merged = box;
        const bool fused_inside = MMath::aabb2ContainsPointAndMergeInPlace(fused_merged, box2, p);
        const bool separate_inside = MMath::aabb2ContainsPoint(box2, p);
        FM_REQUIRE(fused_inside == separate_inside);
        FM_REQUIRE(MMath::aabb2Equal(fused_merged, MMath::aabb2Merge(box, box2)));

        MMath::Aabb2 inter = MMath::aabb2Intersect(box, box2);
        if (MMath::aabb2Intersects(box, box2)) {
            FM_REQUIRE(MMath::aabb2IsValid(inter));
            FM_REQUIRE(MMath::aabb2Contains(box, inter) || MMath::aabb2Contains(box2, inter) ||
                       MMath::aabb2Area(inter) >= 0.0f);
        }
    }
}

FM_TEST(Aabb2, TransformTranslationInvariant) {
    const MMath::Aabb2 box = MMath::aabb2FromMinMax(MMath::Vec2{-1.0f, -2.0f}, MMath::Vec2{3.0f, 4.0f});
    const MMath::Mat3 tr = MMath::mat3FromTranslation(MMath::Vec2{10.0f, -6.0f});

    const MMath::Aabb2 moved = MMath::aabb2Transform(box, tr);

    FM_REQUIRE_VEC2_NEAR(MMath::aabb2Min(moved), (MMath::Vec2{9.0f, -8.0f}), 1e-5f);
    FM_REQUIRE_VEC2_NEAR(MMath::aabb2Max(moved), (MMath::Vec2{13.0f, -2.0f}), 1e-5f);
}

FM_TEST(Aabb3, BasicPropertiesAndIntersection) {
    const MMath::Aabb3 box = MMath::aabb3FromMinMax(MMath::Vec3{-1.0f, -2.0f, -3.0f}, MMath::Vec3{3.0f, 4.0f, 5.0f});

    FM_REQUIRE(MMath::aabb3IsValid(box));
    FM_REQUIRE_VEC3_NEAR(MMath::aabb3Min(box), (MMath::Vec3{-1.0f, -2.0f, -3.0f}), 1e-6f);
    FM_REQUIRE_VEC3_NEAR(MMath::aabb3Max(box), (MMath::Vec3{3.0f, 4.0f, 5.0f}), 1e-6f);
    FM_REQUIRE_VEC3_NEAR(MMath::aabb3Center(box), (MMath::Vec3{1.0f, 1.0f, 1.0f}), 1e-6f);

    FM_REQUIRE_NEAR(MMath::aabb3Volume(box), 4.0f * 6.0f * 8.0f, 1e-6f);
    FM_REQUIRE_NEAR(MMath::aabb3SurfaceArea(box), 2.0f * (4.0f * 6.0f + 6.0f * 8.0f + 4.0f * 8.0f), 1e-5f);

    FM_REQUIRE(MMath::aabb3Contains(box, MMath::Vec3{0.0f, 0.0f, 0.0f}));
    FM_REQUIRE(!MMath::aabb3Contains(box, MMath::Vec3{10.0f, 0.0f, 0.0f}));

    const MMath::Aabb3 b2 = MMath::aabb3FromMinMax(MMath::Vec3{2.0f, 3.0f, 4.0f}, MMath::Vec3{6.0f, 7.0f, 8.0f});
    FM_REQUIRE(MMath::aabb3Overlap(box, b2));

    const MMath::Aabb3 inter = MMath::aabb3Intersect(box, b2);
    FM_REQUIRE_VEC3_NEAR(MMath::aabb3Min(inter), (MMath::Vec3{2.0f, 3.0f, 4.0f}), 1e-6f);
    FM_REQUIRE_VEC3_NEAR(MMath::aabb3Max(inter), (MMath::Vec3{3.0f, 4.0f, 5.0f}), 1e-6f);
}

FM_TEST(Aabb3, RandomizedOverlapAndUnion) {
    fmtest::Rng rng(0x01020304U);

    for (int i = 0; i < 4000; ++i) {
        MMath::Vec3 a{rng.uniform(-15.0f, 0.0f), rng.uniform(-15.0f, 0.0f), rng.uniform(-15.0f, 0.0f)};
        MMath::Vec3 b{rng.uniform(0.0f, 15.0f), rng.uniform(0.0f, 15.0f), rng.uniform(0.0f, 15.0f)};
        MMath::Aabb3 x = MMath::aabb3FromMinMax(a, b);

        MMath::Vec3 c{rng.uniform(-15.0f, 0.0f), rng.uniform(-15.0f, 0.0f), rng.uniform(-15.0f, 0.0f)};
        MMath::Vec3 d{rng.uniform(0.0f, 15.0f), rng.uniform(0.0f, 15.0f), rng.uniform(0.0f, 15.0f)};
        MMath::Aabb3 y = MMath::aabb3FromMinMax(c, d);

        FM_REQUIRE(MMath::aabb3Overlap(x, y) == ref_overlap_3d(x, y));

        MMath::Aabb3 uni = MMath::aabb3Union(x, y);
        FM_REQUIRE(MMath::aabb3ContainsAabb(uni, x));
        FM_REQUIRE(MMath::aabb3ContainsAabb(uni, y));

        MMath::Vec3 p = fmtest::randomVec3(rng, -20.0f, 20.0f);
        MMath::Aabb3 expanded = MMath::aabb3ExpandToPoint(x, p);
        FM_REQUIRE(MMath::aabb3Contains(expanded, p));
    }
}

FM_TEST(Aabb3, TransformTranslationInvariant) {
    const MMath::Aabb3 box = MMath::aabb3FromMinMax(MMath::Vec3{-1.0f, -2.0f, -3.0f}, MMath::Vec3{3.0f, 4.0f, 5.0f});
    const MMath::Mat4 t = MMath::mat4Translation(10.0f, -6.0f, 2.0f);

    const MMath::Aabb3 moved = MMath::aabb3Transform(box, t);

    FM_REQUIRE_VEC3_NEAR(MMath::aabb3Min(moved), (MMath::Vec3{9.0f, -8.0f, -1.0f}), 1e-5f);
    FM_REQUIRE_VEC3_NEAR(MMath::aabb3Max(moved), (MMath::Vec3{13.0f, -2.0f, 7.0f}), 1e-5f);
}
