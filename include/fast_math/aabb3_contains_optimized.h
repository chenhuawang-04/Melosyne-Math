/**
 * @file aabb3_contains_optimized.h
 * @brief aabb3Contains优化实验 - 深度分析DirectXMath策略
 *
 * DirectXMath分析：
 * - 使用Center+Extents表示（对称测试）
 * - Point参数在XMVECTOR中（寄存器传递）
 * - 核心：XMVector3InBounds(point-center, extents)
 * - 测试：-extents <= (point-center) <= extents
 *
 * 优化策略：
 * 1. 使用_mm_mul_ps取负（模仿DirectXMath）
 * 2. 优化load策略（连续内存访问）
 * 3. 减少依赖链长度
 */

#pragma once
#include "fast_math/types.h"
#include "fast_math/vec3.h"
#include <smmintrin.h>  // SSE4.1

namespace MMath {

// ============================================================================
// 优化版本1：使用_mm_mul_ps取负（模仿DirectXMath）
// ============================================================================

MMATH_FORCE_INLINE bool aabb3Contains_v1(Aabb3 aabb_, Vec3 point_) noexcept {
#if defined(__SSE4_1__)
    // DirectXMath uses _mm_mul_ps for negation
    static const __m128 neg_one = _mm_set1_ps(-1.0f);

    __m128 v_min = _mm_set_ps(0.0f, aabb_.minz, aabb_.miny, aabb_.minx);
    __m128 v_neg_max = _mm_set_ps(0.0f, aabb_.neg_maxz, aabb_.neg_maxy, aabb_.neg_maxx);
    __m128 v_point = _mm_set_ps(0.0f, point_.z, point_.y, point_.x);

    // Convert neg_max to max using mul (DirectXMath style)
    __m128 v_max = _mm_mul_ps(v_neg_max, neg_one);

    // Test: min <= point <= max
    __m128 cmp_min = _mm_cmple_ps(v_min, v_point);
    __m128 cmp_max = _mm_cmple_ps(v_point, v_max);
    __m128 cmp_all = _mm_and_ps(cmp_min, cmp_max);

    return (_mm_movemask_ps(cmp_all) & 0x7) == 0x7;
#else
    return point_.x >= aabb_.minx && point_.x <= -aabb_.neg_maxx &&
           point_.y >= aabb_.miny && point_.y <= -aabb_.neg_maxy &&
           point_.z >= aabb_.minz && point_.z <= -aabb_.neg_maxz;
#endif
}

// ============================================================================
// 优化版本2：模仿DirectXMath的InBounds策略
// ============================================================================

MMATH_FORCE_INLINE bool aabb3Contains_v2(Aabb3 aabb_, Vec3 point_) noexcept {
#if defined(__SSE4_1__)
    // Strategy: Compute center and extents on-the-fly, use InBounds test
    // center = (min + max) / 2 = (min - neg_max) / 2
    // extents = (max - min) / 2 = (-neg_max - min) / 2

    __m128 v_min = _mm_set_ps(0.0f, aabb_.minz, aabb_.miny, aabb_.minx);
    __m128 v_neg_max = _mm_set_ps(0.0f, aabb_.neg_maxz, aabb_.neg_maxy, aabb_.neg_maxx);
    __m128 v_point = _mm_set_ps(0.0f, point_.z, point_.y, point_.x);

    // Compute center = (min - neg_max) * 0.5
    __m128 center = _mm_mul_ps(_mm_sub_ps(v_min, v_neg_max), _mm_set1_ps(0.5f));

    // Compute extents = (-neg_max - min) * 0.5 = (max - min) * 0.5
    __m128 v_max = _mm_mul_ps(v_neg_max, _mm_set1_ps(-1.0f));
    __m128 extents = _mm_mul_ps(_mm_sub_ps(v_max, v_min), _mm_set1_ps(0.5f));

    // Test: abs(point - center) <= extents (DirectXMath InBounds)
    __m128 diff = _mm_sub_ps(v_point, center);

    // V <= extents
    __m128 cmp1 = _mm_cmple_ps(diff, extents);
    // -extents <= V (i.e., V >= -extents)
    __m128 neg_extents = _mm_mul_ps(extents, _mm_set1_ps(-1.0f));
    __m128 cmp2 = _mm_cmple_ps(neg_extents, diff);

    __m128 cmp_all = _mm_and_ps(cmp1, cmp2);
    return (_mm_movemask_ps(cmp_all) & 0x7) == 0x7;
#else
    return point_.x >= aabb_.minx && point_.x <= -aabb_.neg_maxx &&
           point_.y >= aabb_.miny && point_.y <= -aabb_.neg_maxy &&
           point_.z >= aabb_.minz && point_.z <= -aabb_.neg_maxz;
#endif
}

// ============================================================================
// 优化版本3：使用unaligned load减少指令
// ============================================================================

MMATH_FORCE_INLINE bool aabb3Contains_v3(Aabb3 aabb_, Vec3 point_) noexcept {
#if defined(__SSE4_1__)
    // Try to reduce load overhead by loading 4 floats at once
    __m128 v_min_plus = _mm_loadu_ps(&aabb_.minx);  // minx, miny, minz, neg_maxx
    __m128 v_remaining = _mm_set_ps(0.0f, 0.0f, aabb_.neg_maxz, aabb_.neg_maxy);
    __m128 v_point = _mm_set_ps(0.0f, point_.z, point_.y, point_.x);

    // Extract components
    __m128 v_min = _mm_blend_ps(v_min_plus, _mm_setzero_ps(), 0x8);  // Keep xyz, zero w

    // Reconstruct neg_max
    __m128 v_neg_max = _mm_shuffle_ps(v_min_plus, v_remaining, _MM_SHUFFLE(0, 1, 3, 3));
    v_neg_max = _mm_shuffle_ps(v_neg_max, v_neg_max, _MM_SHUFFLE(2, 1, 0, 3));
    v_neg_max = _mm_blend_ps(v_neg_max, _mm_setzero_ps(), 0x8);

    // Convert neg_max to max
    __m128 v_max = _mm_mul_ps(v_neg_max, _mm_set1_ps(-1.0f));

    // Test
    __m128 cmp_min = _mm_cmple_ps(v_min, v_point);
    __m128 cmp_max = _mm_cmple_ps(v_point, v_max);
    __m128 cmp_all = _mm_and_ps(cmp_min, cmp_max);

    return (_mm_movemask_ps(cmp_all) & 0x7) == 0x7;
#else
    return point_.x >= aabb_.minx && point_.x <= -aabb_.neg_maxx &&
           point_.y >= aabb_.miny && point_.y <= -aabb_.neg_maxy &&
           point_.z >= aabb_.minz && point_.z <= -aabb_.neg_maxz;
#endif
}

// ============================================================================
// 优化版本4：使用_mm_cmpge_ps减少逻辑反转
// ============================================================================

MMATH_FORCE_INLINE bool aabb3Contains_v4(Aabb3 aabb_, Vec3 point_) noexcept {
#if defined(__SSE4_1__)
    __m128 v_min = _mm_set_ps(0.0f, aabb_.minz, aabb_.miny, aabb_.minx);
    __m128 v_neg_max = _mm_set_ps(0.0f, aabb_.neg_maxz, aabb_.neg_maxy, aabb_.neg_maxx);
    __m128 v_point = _mm_set_ps(0.0f, point_.z, point_.y, point_.x);

    // Convert neg_max to max
    __m128 v_max = _mm_sub_ps(_mm_setzero_ps(), v_neg_max);

    // Use _mm_cmpge_ps to match comparison direction
    __m128 cmp_ge_min = _mm_cmpge_ps(v_point, v_min);  // point >= min
    __m128 cmp_le_max = _mm_cmple_ps(v_point, v_max);  // point <= max
    __m128 cmp_all = _mm_and_ps(cmp_ge_min, cmp_le_max);

    return (_mm_movemask_ps(cmp_all) & 0x7) == 0x7;
#else
    return point_.x >= aabb_.minx && point_.x <= -aabb_.neg_maxx &&
           point_.y >= aabb_.miny && point_.y <= -aabb_.neg_maxy &&
           point_.z >= aabb_.minz && point_.z <= -aabb_.neg_maxz;
#endif
}

// ============================================================================
// 当前版本（baseline）
// ============================================================================

MMATH_FORCE_INLINE bool aabb3Contains_baseline(Aabb3 aabb_, Vec3 point_) noexcept {
#if defined(__SSE4_1__)
    __m128 v_aabb_min = _mm_setr_ps(aabb_.minx, aabb_.miny, aabb_.minz, 0.0f);
    __m128 v_aabb_max_neg = _mm_setr_ps(aabb_.neg_maxx, aabb_.neg_maxy, aabb_.neg_maxz, 0.0f);
    __m128 v_point = _mm_setr_ps(point_.x, point_.y, point_.z, 0.0f);
    __m128 v_point_neg = _mm_sub_ps(_mm_setzero_ps(), v_point);

    __m128 cmp_min = _mm_cmple_ps(v_aabb_min, v_point);
    __m128 cmp_max = _mm_cmple_ps(v_aabb_max_neg, v_point_neg);
    __m128 cmp_all = _mm_and_ps(cmp_min, cmp_max);

    int mask = _mm_movemask_ps(cmp_all);
    return (mask & 0x7) == 0x7;
#else
    return point_.x >= aabb_.minx && point_.x <= -aabb_.neg_maxx &&
           point_.y >= aabb_.miny && point_.y <= -aabb_.neg_maxy &&
           point_.z >= aabb_.minz && point_.z <= -aabb_.neg_maxz;
#endif
}

} // namespace MMath
