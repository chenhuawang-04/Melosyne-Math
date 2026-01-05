/**
 * @file aabb2_simd.h
 * @brief AVX2 batch processing for AABB2 operations
 *
 * Optimization Strategy:
 * - Process 8 AABBs at once with AVX2
 * - Leverage negative max storage for minimal shuffle overhead
 * - Preprocess loop-invariant reference AABBs
 */

#pragma once

#include <cmath>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace MMath {
namespace detail {

#if defined(__AVX2__) && defined(__FMA__)

// ============================================================================
// AVX2 Batch Operations
// ============================================================================

/**
 * @brief Batch overlap test (8 AABBs at once)
 *
 * Reference: https://mtsamis.com/blog/23_11_07_aabbs
 * Optimized to preprocess reference AABB outside the loop
 */
inline void aabb2IntersectsArraySimd(
    Aabb2 reference_,
    const Aabb2* MMATH_RESTRICT candidates_,
    bool* MMATH_RESTRICT results_,
    int32_t count_
) noexcept {
    int32_t i = 0;
    const int32_t simd_count = (count_ / 8) * 8;

    // Preprocess reference (loop-invariant)
    __m128 ref_128 = _mm_loadu_ps(&reference_.minx);  // [minx, miny, neg_maxx, neg_maxy]
    __m128 ref_neg = _mm_sub_ps(_mm_setzero_ps(), ref_128);  // Negate
    __m128 ref_shuffled = _mm_shuffle_ps(ref_neg, ref_neg, _MM_SHUFFLE(1, 0, 3, 2));  // Swap min/max
    __m256 ref_256 = _mm256_broadcast_ps(&ref_shuffled);  // Broadcast to AVX2

    for (; i < simd_count; i += 8) {
        // Load 8 candidate AABBs (32 floats total)
        // AABB0: [minx0, miny0, neg_maxx0, neg_maxy0]
        // AABB1: [minx1, miny1, neg_maxx1, neg_maxy1]
        // ...

        // Strategy: Load in chunks and deinterleave
        __m256 aabb0_1 = _mm256_loadu_ps((const float*)&candidates_[i + 0]);  // AABB 0-1
        __m256 aabb2_3 = _mm256_loadu_ps((const float*)&candidates_[i + 2]);  // AABB 2-3
        __m256 aabb4_5 = _mm256_loadu_ps((const float*)&candidates_[i + 4]);  // AABB 4-5
        __m256 aabb6_7 = _mm256_loadu_ps((const float*)&candidates_[i + 6]);  // AABB 6-7

        // Process 2 AABBs at a time (each __m256 contains 2 AABBs)
        auto test_pair = [&ref_256](__m256 pair) -> int {
            __m256 cmp = _mm256_cmp_ps(pair, ref_256, _CMP_LE_OQ);
            int mask = _mm256_movemask_ps(cmp);
            // Check if all 4 components of each AABB are <= (2 AABBs, 4 components each)
            bool aabb0_overlap = ((mask & 0x0F) == 0x0F);  // Bits 0-3
            bool aabb1_overlap = ((mask & 0xF0) == 0xF0);  // Bits 4-7
            return (aabb0_overlap ? 1 : 0) | (aabb1_overlap ? 2 : 0);
        };

        int result0_1 = test_pair(aabb0_1);
        int result2_3 = test_pair(aabb2_3);
        int result4_5 = test_pair(aabb4_5);
        int result6_7 = test_pair(aabb6_7);

        results_[i + 0] = (result0_1 & 1) != 0;
        results_[i + 1] = (result0_1 & 2) != 0;
        results_[i + 2] = (result2_3 & 1) != 0;
        results_[i + 3] = (result2_3 & 2) != 0;
        results_[i + 4] = (result4_5 & 1) != 0;
        results_[i + 5] = (result4_5 & 2) != 0;
        results_[i + 6] = (result6_7 & 1) != 0;
        results_[i + 7] = (result6_7 & 2) != 0;
    }

    // Handle remainder
    for (; i < count_; ++i) {
        results_[i] = aabb2Intersects(reference_, candidates_[i]);
    }
}

/**
 * @brief Batch merge (8 pairs at once)
 */
inline void aabb2MergeArraySimd(
    const Aabb2* MMATH_RESTRICT a_,
    const Aabb2* MMATH_RESTRICT b_,
    Aabb2* MMATH_RESTRICT result_,
    int32_t count_
) noexcept {
    int32_t i = 0;
    const int32_t simd_count = (count_ / 8) * 8;

    for (; i < simd_count; i += 8) {
        // Load and merge 8 pairs (2 AABBs per __m256)
        __m256 a01 = _mm256_loadu_ps((const float*)&a_[i + 0]);
        __m256 b01 = _mm256_loadu_ps((const float*)&b_[i + 0]);
        __m256 r01 = _mm256_min_ps(a01, b01);  // Merge AABB 0-1

        __m256 a23 = _mm256_loadu_ps((const float*)&a_[i + 2]);
        __m256 b23 = _mm256_loadu_ps((const float*)&b_[i + 2]);
        __m256 r23 = _mm256_min_ps(a23, b23);  // Merge AABB 2-3

        __m256 a45 = _mm256_loadu_ps((const float*)&a_[i + 4]);
        __m256 b45 = _mm256_loadu_ps((const float*)&b_[i + 4]);
        __m256 r45 = _mm256_min_ps(a45, b45);  // Merge AABB 4-5

        __m256 a67 = _mm256_loadu_ps((const float*)&a_[i + 6]);
        __m256 b67 = _mm256_loadu_ps((const float*)&b_[i + 6]);
        __m256 r67 = _mm256_min_ps(a67, b67);  // Merge AABB 6-7

        _mm256_storeu_ps((float*)&result_[i + 0], r01);
        _mm256_storeu_ps((float*)&result_[i + 2], r23);
        _mm256_storeu_ps((float*)&result_[i + 4], r45);
        _mm256_storeu_ps((float*)&result_[i + 6], r67);
    }

    // Handle remainder
    for (; i < count_; ++i) {
        result_[i] = aabb2Merge(a_[i], b_[i]);
    }
}

/**
 * @brief Batch contains point test (8 points at once)
 */
inline void aabb2ContainsPointsArraySimd(
    Aabb2 aabb_,
    const Vec2* MMATH_RESTRICT points_,
    bool* MMATH_RESTRICT results_,
    int32_t count_
) noexcept {
    int32_t i = 0;
    const int32_t simd_count = (count_ / 8) * 8;

    // Broadcast AABB to AVX2 registers
    __m256 aabb_minx = _mm256_set1_ps(aabb_.minx);
    __m256 aabb_miny = _mm256_set1_ps(aabb_.miny);
    __m256 aabb_maxx = _mm256_set1_ps(-aabb_.neg_maxx);
    __m256 aabb_maxy = _mm256_set1_ps(-aabb_.neg_maxy);

    for (; i < simd_count; i += 8) {
        // Load 8 points (need to deinterleave x and y)
        alignas(32) float temp_x[8], temp_y[8];
        for (int j = 0; j < 8; ++j) {
            temp_x[j] = points_[i + j].x;
            temp_y[j] = points_[i + j].y;
        }

        __m256 px = _mm256_load_ps(temp_x);
        __m256 py = _mm256_load_ps(temp_y);

        // Test: minx <= px <= maxx && miny <= py <= maxy
        __m256 cmp_x_min = _mm256_cmp_ps(aabb_minx, px, _CMP_LE_OQ);
        __m256 cmp_x_max = _mm256_cmp_ps(px, aabb_maxx, _CMP_LE_OQ);
        __m256 cmp_y_min = _mm256_cmp_ps(aabb_miny, py, _CMP_LE_OQ);
        __m256 cmp_y_max = _mm256_cmp_ps(py, aabb_maxy, _CMP_LE_OQ);

        __m256 cmp_x = _mm256_and_ps(cmp_x_min, cmp_x_max);
        __m256 cmp_y = _mm256_and_ps(cmp_y_min, cmp_y_max);
        __m256 cmp_all = _mm256_and_ps(cmp_x, cmp_y);

        int mask = _mm256_movemask_ps(cmp_all);
        for (int j = 0; j < 8; ++j) {
            results_[i + j] = (mask & (1 << j)) != 0;
        }
    }

    // Handle remainder
    for (; i < count_; ++i) {
        results_[i] = aabb2ContainsPoint(aabb_, points_[i]);
    }
}

#endif // AVX2

} // namespace detail
} // namespace MMath
