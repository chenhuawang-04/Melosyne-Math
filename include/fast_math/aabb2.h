/**
 * @file aabb2.h
 * @brief 2D Axis-Aligned Bounding Box (AABB) operations
 *
 * Design Philosophy:
 * - Strict POD structure (NO member functions)
 * - Internal optimization: stores -maxx, -maxy for SIMD efficiency
 * - User-friendly API: accessors hide internal representation
 * - Single-instruction Union, 3-instruction Overlap test
 *
 * Key Optimization:
 * By storing maximum coordinates as negative values, union operations
 * become pure min operations, and overlap tests are highly optimized.
 *
 * Reference: https://mtsamis.com/blog/23_11_07_aabbs
 */

#pragma once

#include "types.h"
#include "vec2.h"
#include "mat3.h"
#include <cmath>
#include <algorithm>
#include <limits>

#if defined(__SSE4_1__)
#include <smmintrin.h>
#endif

namespace MMath {

// ============================================================================
// Constructors (Free Functions)
// ============================================================================

/**
 * @brief Create AABB from min and max points
 */
MMATH_FORCE_INLINE Aabb2 aabb2FromMinMax(Vec2 min_, Vec2 max_) noexcept {
    return Aabb2{ min_.x, min_.y, -max_.x, -max_.y };
}

/**
 * @brief Create AABB from center and half-extents
 */
MMATH_FORCE_INLINE Aabb2 aabb2FromCenterExtents(Vec2 center_, Vec2 extents_) noexcept {
    return Aabb2{
        center_.x - extents_.x,  // minx
        center_.y - extents_.y,  // miny
        -(center_.x + extents_.x),  // -maxx
        -(center_.y + extents_.y)   // -maxy
    };
}

/**
 * @brief Create AABB from a single point (degenerate box)
 */
MMATH_FORCE_INLINE Aabb2 aabb2FromPoint(Vec2 point_) noexcept {
    return Aabb2{ point_.x, point_.y, -point_.x, -point_.y };
}

/**
 * @brief Create AABB from array of points
 */
inline Aabb2 aabb2FromPoints(const Vec2* MMATH_RESTRICT points_, int32_t count_) noexcept {
    if (count_ == 0) {
        return Aabb2{ 0.0f, 0.0f, 0.0f, 0.0f };
    }

    float minx = points_[0].x;
    float miny = points_[0].y;
    float maxx = points_[0].x;
    float maxy = points_[0].y;

    for (int32_t i = 1; i < count_; ++i) {
        minx = std::min(minx, points_[i].x);
        miny = std::min(miny, points_[i].y);
        maxx = std::max(maxx, points_[i].x);
        maxy = std::max(maxy, points_[i].y);
    }

    return Aabb2{ minx, miny, -maxx, -maxy };
}

/**
 * @brief Create empty AABB (invalid, used as initial value for expansion)
 */
MMATH_FORCE_INLINE Aabb2 aabb2Empty() noexcept {
    constexpr float large = 1e30f;  // Use large value instead of infinity (compatible with -ffast-math)
    return Aabb2{ large, large, large, large };  // min=large, max=-large
}

/**
 * @brief Check if AABB is valid (min <= max)
 */
MMATH_FORCE_INLINE bool aabb2IsValid(Aabb2 aabb_) noexcept {
    return aabb_.minx <= -aabb_.neg_maxx && aabb_.miny <= -aabb_.neg_maxy;
}

// ============================================================================
// Accessors (Free Functions)
// ============================================================================

/**
 * @brief Get minimum corner
 */
MMATH_FORCE_INLINE Vec2 aabb2Min(Aabb2 aabb_) noexcept {
    return Vec2{ aabb_.minx, aabb_.miny };
}

/**
 * @brief Get maximum corner
 */
MMATH_FORCE_INLINE Vec2 aabb2Max(Aabb2 aabb_) noexcept {
    return Vec2{ -aabb_.neg_maxx, -aabb_.neg_maxy };
}

/**
 * @brief Get center point
 */
MMATH_FORCE_INLINE Vec2 aabb2Center(Aabb2 aabb_) noexcept {
    return Vec2{
        (aabb_.minx - aabb_.neg_maxx) * 0.5f,
        (aabb_.miny - aabb_.neg_maxy) * 0.5f
    };
}

/**
 * @brief Get half-extents (distance from center to edge)
 */
MMATH_FORCE_INLINE Vec2 aabb2Extents(Aabb2 aabb_) noexcept {
    return Vec2{
        (-aabb_.neg_maxx - aabb_.minx) * 0.5f,
        (-aabb_.neg_maxy - aabb_.miny) * 0.5f
    };
}

/**
 * @brief Get full size (width, height)
 */
MMATH_FORCE_INLINE Vec2 aabb2Size(Aabb2 aabb_) noexcept {
    return Vec2{
        -aabb_.neg_maxx - aabb_.minx,
        -aabb_.neg_maxy - aabb_.miny
    };
}

/**
 * @brief Get width
 */
MMATH_FORCE_INLINE float aabb2Width(Aabb2 aabb_) noexcept {
    return -aabb_.neg_maxx - aabb_.minx;
}

/**
 * @brief Get height
 */
MMATH_FORCE_INLINE float aabb2Height(Aabb2 aabb_) noexcept {
    return -aabb_.neg_maxy - aabb_.miny;
}

/**
 * @brief Get area
 */
MMATH_FORCE_INLINE float aabb2Area(Aabb2 aabb_) noexcept {
    float w = -aabb_.neg_maxx - aabb_.minx;
    float h = -aabb_.neg_maxy - aabb_.miny;
    return w * h;
}

/**
 * @brief Get perimeter
 */
MMATH_FORCE_INLINE float aabb2Perimeter(Aabb2 aabb_) noexcept {
    float w = -aabb_.neg_maxx - aabb_.minx;
    float h = -aabb_.neg_maxy - aabb_.miny;
    return 2.0f * (w + h);
}

// ============================================================================
// Geometric Tests
// ============================================================================

/**
 * @brief Test if AABB contains a point
 */
MMATH_FORCE_INLINE bool aabb2ContainsPoint(Aabb2 aabb_, Vec2 point_) noexcept {
#if defined(__SSE4_1__)
    __m128 v_aabb = _mm_loadu_ps(&aabb_.minx);  // [minx, miny, neg_maxx, neg_maxy]
    __m128 v_point = _mm_setr_ps(point_.x, point_.y, -point_.x, -point_.y);
    __m128 cmp = _mm_cmple_ps(v_aabb, v_point);  // aabb <= point
    return _mm_test_all_ones(_mm_castps_si128(cmp));
#else
    return point_.x >= aabb_.minx && point_.x <= -aabb_.neg_maxx &&
           point_.y >= aabb_.miny && point_.y <= -aabb_.neg_maxy;
#endif
}

/**
 * @brief Test if AABB 'a_' contains AABB 'b_'
 */
MMATH_FORCE_INLINE bool aabb2Contains(Aabb2 a_, Aabb2 b_) noexcept {
#if defined(__SSE4_1__)
    __m128 va = _mm_loadu_ps(&a_.minx);
    __m128 vb = _mm_loadu_ps(&b_.minx);
    __m128 cmp = _mm_cmple_ps(va, vb);  // a.min <= b.min && a.neg_max <= b.neg_max
    return _mm_test_all_ones(_mm_castps_si128(cmp));
#else
    return a_.minx <= b_.minx && a_.miny <= b_.miny &&
           a_.neg_maxx <= b_.neg_maxx && a_.neg_maxy <= b_.neg_maxy;
#endif
}

/**
 * @brief Test if two AABBs overlap (mtsamis optimization)
 *
 * Reference: https://mtsamis.com/blog/23_11_07_aabbs
 * Optimized to 3 SSE instructions
 */
MMATH_FORCE_INLINE bool aabb2Intersects(Aabb2 a_, Aabb2 b_) noexcept {
#if defined(__SSE4_1__)
    __m128 va = _mm_loadu_ps(&a_.minx);  // [minx, miny, neg_maxx, neg_maxy]
    __m128 vb = _mm_loadu_ps(&b_.minx);

    // Negate b to convert -maxx, -maxy back to maxx, maxy
    vb = _mm_sub_ps(_mm_setzero_ps(), vb);

    // Shuffle a to swap min and max: [neg_maxx, neg_maxy, minx, miny]
    va = _mm_shuffle_ps(va, va, _MM_SHUFFLE(1, 0, 3, 2));

    // Compare: a <= b (all components must be true for overlap)
    __m128 cmp = _mm_cmple_ps(va, vb);
    return _mm_test_all_ones(_mm_castps_si128(cmp));
#else
    // Standard overlap test
    return a_.minx <= -b_.neg_maxx && -a_.neg_maxx >= b_.minx &&
           a_.miny <= -b_.neg_maxy && -a_.neg_maxy >= b_.miny;
#endif
}

/**
 * @brief Compute overlap area (returns 0 if no overlap)
 */
MMATH_FORCE_INLINE float aabb2OverlapArea(Aabb2 a_, Aabb2 b_) noexcept {
    float minx = std::max(a_.minx, b_.minx);
    float miny = std::max(a_.miny, b_.miny);
    float maxx = std::min(-a_.neg_maxx, -b_.neg_maxx);
    float maxy = std::min(-a_.neg_maxy, -b_.neg_maxy);

    float w = std::max(0.0f, maxx - minx);
    float h = std::max(0.0f, maxy - miny);
    return w * h;
}

// ============================================================================
// Operations
// ============================================================================

/**
 * @brief Merge (Union) two AABBs - Single SSE instruction
 *
 * Reference: https://mtsamis.com/blog/23_11_07_aabbs
 * Thanks to negative max storage, this is just min(a, b)
 */
MMATH_FORCE_INLINE Aabb2 aabb2Merge(Aabb2 a_, Aabb2 b_) noexcept {
#if defined(__SSE4_1__)
    __m128 va = _mm_loadu_ps(&a_.minx);
    __m128 vb = _mm_loadu_ps(&b_.minx);
    __m128 result = _mm_min_ps(va, vb);  // Single instruction!

    Aabb2 out;
    _mm_storeu_ps(&out.minx, result);
    return out;
#else
    return Aabb2{
        std::min(a_.minx, b_.minx),
        std::min(a_.miny, b_.miny),
        std::min(a_.neg_maxx, b_.neg_maxx),
        std::min(a_.neg_maxy, b_.neg_maxy)
    };
#endif
}

/**
 * @brief Intersect two AABBs (may produce invalid AABB if no overlap)
 */
MMATH_FORCE_INLINE Aabb2 aabb2Intersect(Aabb2 a_, Aabb2 b_) noexcept {
#if defined(__SSE4_1__)
    __m128 va = _mm_loadu_ps(&a_.minx);
    __m128 vb = _mm_loadu_ps(&b_.minx);
    __m128 result = _mm_max_ps(va, vb);

    Aabb2 out;
    _mm_storeu_ps(&out.minx, result);
    return out;
#else
    return Aabb2{
        std::max(a_.minx, b_.minx),
        std::max(a_.miny, b_.miny),
        std::max(a_.neg_maxx, b_.neg_maxx),
        std::max(a_.neg_maxy, b_.neg_maxy)
    };
#endif
}

/**
 * @brief Expand AABB by a uniform amount on all sides
 */
MMATH_FORCE_INLINE Aabb2 aabb2Expand(Aabb2 aabb_, float amount_) noexcept {
    return Aabb2{
        aabb_.minx - amount_,
        aabb_.miny - amount_,
        aabb_.neg_maxx - amount_,  // Remember: storing -maxx, so subtract
        aabb_.neg_maxy - amount_
    };
}

/**
 * @brief Expand AABB to include a point
 */
MMATH_FORCE_INLINE Aabb2 aabb2ExpandToPoint(Aabb2 aabb_, Vec2 point_) noexcept {
    return aabb2Merge(aabb_, aabb2FromPoint(point_));
}

/**
 * @brief Expand AABB to include another AABB (same as merge)
 */
MMATH_FORCE_INLINE Aabb2 aabb2ExpandToAabb(Aabb2 a_, Aabb2 b_) noexcept {
    return aabb2Merge(a_, b_);
}

/**
 * @brief Transform AABB by a 2D affine matrix
 *
 * Strategy: Transform all 4 corners, then recompute min/max
 */
inline Aabb2 aabb2Transform(Aabb2 aabb_, Mat3 matrix_) noexcept {
    // Get actual min/max coordinates
    float minx = aabb_.minx;
    float miny = aabb_.miny;
    float maxx = -aabb_.neg_maxx;
    float maxy = -aabb_.neg_maxy;

    // Transform 4 corners
    Vec2 corners[4] = {
        mat3TransformPoint(matrix_, {minx, miny}),
        mat3TransformPoint(matrix_, {maxx, miny}),
        mat3TransformPoint(matrix_, {minx, maxy}),
        mat3TransformPoint(matrix_, {maxx, maxy})
    };

    // Compute new min/max
    return aabb2FromPoints(corners, 4);
}

/**
 * @brief Equality comparison
 */
MMATH_FORCE_INLINE bool aabb2Equal(Aabb2 a_, Aabb2 b_) noexcept {
    return a_.minx == b_.minx && a_.miny == b_.miny &&
           a_.neg_maxx == b_.neg_maxx && a_.neg_maxy == b_.neg_maxy;
}

/**
 * @brief Near-equality comparison with epsilon
 */
MMATH_FORCE_INLINE bool aabb2NearEqual(Aabb2 a_, Aabb2 b_, float epsilon_) noexcept {
    auto is_near_ = [epsilon_](float x, float y) {
        float d = x - y;
        return (d < 0 ? -d : d) <= epsilon_;
    };
    return is_near_(a_.minx, b_.minx) && is_near_(a_.miny, b_.miny) &&
           is_near_(a_.neg_maxx, b_.neg_maxx) && is_near_(a_.neg_maxy, b_.neg_maxy);
}

} // namespace MMath

// Include SIMD batch processing implementations
#include "aabb2_simd.h"

namespace MMath {

// ============================================================================
// Batch Processing (SIMD dispatch)
// ============================================================================

/**
 * @brief Batch overlap test (one reference AABB vs many candidates)
 */
inline void aabb2IntersectsArray(
    Aabb2 reference_,
    const Aabb2* MMATH_RESTRICT candidates_,
    bool* MMATH_RESTRICT results_,
    int32_t count_
) noexcept {
#if defined(__AVX2__) && defined(__FMA__)
    detail::aabb2IntersectsArraySimd(reference_, candidates_, results_, count_);
#else
    for (int32_t i = 0; i < count_; ++i) {
        results_[i] = aabb2Intersects(reference_, candidates_[i]);
    }
#endif
}

/**
 * @brief Batch merge (pairwise)
 */
inline void aabb2MergeArray(
    const Aabb2* MMATH_RESTRICT a_,
    const Aabb2* MMATH_RESTRICT b_,
    Aabb2* MMATH_RESTRICT result_,
    int32_t count_
) noexcept {
#if defined(__AVX2__) && defined(__FMA__)
    detail::aabb2MergeArraySimd(a_, b_, result_, count_);
#else
    for (int32_t i = 0; i < count_; ++i) {
        result_[i] = aabb2Merge(a_[i], b_[i]);
    }
#endif
}

/**
 * @brief Batch contains point test (one AABB vs many points)
 */
inline void aabb2ContainsPointsArray(
    Aabb2 aabb_,
    const Vec2* MMATH_RESTRICT points_,
    bool* MMATH_RESTRICT results_,
    int32_t count_
) noexcept {
#if defined(__AVX2__) && defined(__FMA__)
    detail::aabb2ContainsPointsArraySimd(aabb_, points_, results_, count_);
#else
    for (int32_t i = 0; i < count_; ++i) {
        results_[i] = aabb2ContainsPoint(aabb_, points_[i]);
    }
#endif
}

} // namespace MMath
