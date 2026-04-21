/**
 * @file vec2_simd.h
 * @brief Hand-written SIMD Vec2 batch processing (AOS deinterleave optimization)
 *
 * Optimization Strategy:
 * - Deinterleave AOS Vec2 to SoA using AVX2 shuffle/unpack
 * - Process 8 Vec2s at once with AVX2
 * - Re-interleave results back to AOS
 *
 * References:
 * - AOS SIMD Vectorization: https://arxiv.org/abs/1806.05713
 * - Deinterleave with AVX2: https://stackoverflow.com/questions/35871639/how-to-deinterleave-image-channel-in-sse
 * - Intel Unpack Intrinsics: https://www.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-intel-advanced-vector-extensions/intrinsics-for-unpack-and-interleave-operations.html
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
// AVX2 Deinterleave/Interleave Helpers
// ============================================================================

/**
 * @brief Deinterleave 8 Vec2s from AOS to SoA
 *
 * Input:  [x0,y0, x1,y1, x2,y2, x3,y3] [x4,y4, x5,y5, x6,y6, x7,y7]
 * Output: [x0,x1,x2,x3,x4,x5,x6,x7] [y0,y1,y2,y3,y4,y5,y6,y7]
 */
MMATH_FORCE_INLINE void deinterleave8Vec2(
    const Vec2* MMATH_RESTRICT input_,
    __m256& out_x_,
    __m256& out_y_
) noexcept {
    // Load 8 Vec2s (16 floats)
    __m256 v0 = _mm256_loadu_ps(reinterpret_cast<const float*>(input_ + 0));  // x0,y0,x1,y1, x2,y2,x3,y3
    __m256 v1 = _mm256_loadu_ps(reinterpret_cast<const float*>(input_ + 4));  // x4,y4,x5,y5, x6,y6,x7,y7

    // Shuffle to separate x and y within 128-bit lanes
    // unpacklo: x0,x1,y0,y1, x4,x5,y4,y5
    // unpackhi: x2,x3,y2,y3, x6,x7,y6,y7
    __m256 xy01 = _mm256_shuffle_ps(v0, v1, _MM_SHUFFLE(2, 0, 2, 0));  // x0,x1,x4,x5, x2,x3,x6,x7
    __m256 xy23 = _mm256_shuffle_ps(v0, v1, _MM_SHUFFLE(3, 1, 3, 1));  // y0,y1,y4,y5, y2,y3,y6,y7

    // Reorder lanes to contiguous x/y vectors:
    // xy01 = [x0,x1,x4,x5, x2,x3,x6,x7]
    // xy23 = [y0,y1,y4,y5, y2,y3,y6,y7]
    // order -> [0,1,4,5,2,3,6,7]
    const __m256i order = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
    out_x_ = _mm256_permutevar8x32_ps(xy01, order);
    out_y_ = _mm256_permutevar8x32_ps(xy23, order);
}

/**
 * @brief Interleave 8 Vec2s from SoA to AOS
 */
MMATH_FORCE_INLINE void interleave8Vec2(
    __m256 x_,
    __m256 y_,
    Vec2* MMATH_RESTRICT output_
) noexcept {
    // Interleave x and y
    __m256 xy_low = _mm256_unpacklo_ps(x_, y_);   // x0,y0,x1,y1, x4,y4,x5,y5
    __m256 xy_high = _mm256_unpackhi_ps(x_, y_);  // x2,y2,x3,y3, x6,y6,x7,y7

    // Permute to correct order
    __m256 v0 = _mm256_permute2f128_ps(xy_low, xy_high, 0x20);   // x0,y0,x1,y1, x2,y2,x3,y3
    __m256 v1 = _mm256_permute2f128_ps(xy_low, xy_high, 0x31);   // x4,y4,x5,y5, x6,y6,x7,y7

    // Store
    _mm256_storeu_ps(reinterpret_cast<float*>(output_ + 0), v0);
    _mm256_storeu_ps(reinterpret_cast<float*>(output_ + 4), v1);
}

// ============================================================================
// AVX2 Vec2 Batch Operations
// ============================================================================

/**
 * @brief Batch Vec2 add with AVX2 deinterleave optimization
 */
inline void vec2AddSimd(
    const Vec2* MMATH_RESTRICT a_,
    const Vec2* MMATH_RESTRICT b_,
    Vec2* MMATH_RESTRICT result_,
    int32_t count_
) noexcept {
    int32_t i = 0;
    const int32_t simd_count = (count_ / 8) * 8;

    // Process 8 Vec2s at a time
    for (; i < simd_count; i += 8) {
        __m256 ax, ay, bx, by;
        deinterleave8Vec2(a_ + i, ax, ay);
        deinterleave8Vec2(b_ + i, bx, by);

        __m256 rx = _mm256_add_ps(ax, bx);
        __m256 ry = _mm256_add_ps(ay, by);

        interleave8Vec2(rx, ry, result_ + i);
    }

    // Handle remainder
    for (; i < count_; ++i) {
        result_[i] = vec2Add(a_[i], b_[i]);
    }
}

/**
 * @brief Batch Vec2 dot product with AVX2
 */
inline void vec2DotSimd(
    const Vec2* MMATH_RESTRICT a_,
    const Vec2* MMATH_RESTRICT b_,
    float* MMATH_RESTRICT result_,
    int32_t count_
) noexcept {
    int32_t i = 0;
    const int32_t simd_count = (count_ / 8) * 8;

    for (; i < simd_count; i += 8) {
        __m256 ax, ay, bx, by;
        deinterleave8Vec2(a_ + i, ax, ay);
        deinterleave8Vec2(b_ + i, bx, by);

        // dot = ax*bx + ay*by
        __m256 dot = _mm256_fmadd_ps(ax, bx, _mm256_mul_ps(ay, by));

        _mm256_storeu_ps(result_ + i, dot);
    }

    // Handle remainder
    for (; i < count_; ++i) {
        result_[i] = vec2Dot(a_[i], b_[i]);
    }
}

/**
 * @brief Batch Vec2 normalize with AVX2
 */
inline void vec2NormalizeSimd(
    const Vec2* MMATH_RESTRICT input_,
    Vec2* MMATH_RESTRICT result_,
    int32_t count_
) noexcept {
    int32_t i = 0;
    const int32_t simd_count = (count_ / 8) * 8;

    const __m256 epsilon = _mm256_set1_ps(1e-8f);

    for (; i < simd_count; i += 8) {
        __m256 vx, vy;
        deinterleave8Vec2(input_ + i, vx, vy);

        // len_sq = x*x + y*y
        __m256 len_sq = _mm256_fmadd_ps(vx, vx, _mm256_mul_ps(vy, vy));

        // inv_len = 1 / sqrt(len_sq)
        __m256 inv_len = _mm256_rsqrt_ps(len_sq);  // Approximate

        // Newton-Raphson refinement: inv_len = inv_len * (1.5 - 0.5 * len_sq * inv_len * inv_len)
        __m256 half = _mm256_set1_ps(0.5f);
        __m256 three_half = _mm256_set1_ps(1.5f);
        __m256 inv_len_sq = _mm256_mul_ps(inv_len, inv_len);
        __m256 temp = _mm256_mul_ps(half, _mm256_mul_ps(len_sq, inv_len_sq));
        inv_len = _mm256_mul_ps(inv_len, _mm256_sub_ps(three_half, temp));

        // Clamp to avoid division by zero
        __m256 is_valid = _mm256_cmp_ps(len_sq, epsilon, _CMP_GT_OQ);
        inv_len = _mm256_and_ps(inv_len, is_valid);

        // result = v * inv_len
        __m256 rx = _mm256_mul_ps(vx, inv_len);
        __m256 ry = _mm256_mul_ps(vy, inv_len);

        interleave8Vec2(rx, ry, result_ + i);
    }

    // Handle remainder
    for (; i < count_; ++i) {
        result_[i] = vec2Normalize(input_[i]);
    }
}

/**
 * @brief Batch Vec2 fast normalize with AVX2 (no refinement)
 */
inline void vec2NormalizeFastSimd(
    const Vec2* MMATH_RESTRICT input_,
    Vec2* MMATH_RESTRICT result_,
    int32_t count_
) noexcept {
    int32_t i = 0;
    const int32_t simd_count = (count_ / 8) * 8;

    const __m256 epsilon = _mm256_set1_ps(1e-8f);

    for (; i < simd_count; i += 8) {
        __m256 vx, vy;
        deinterleave8Vec2(input_ + i, vx, vy);

        // len_sq = x*x + y*y
        __m256 len_sq = _mm256_fmadd_ps(vx, vx, _mm256_mul_ps(vy, vy));

        // inv_len = rsqrt(len_sq) - approximate, no refinement
        __m256 inv_len = _mm256_rsqrt_ps(len_sq);

        // Clamp to avoid division by zero
        __m256 is_valid = _mm256_cmp_ps(len_sq, epsilon, _CMP_GT_OQ);
        inv_len = _mm256_and_ps(inv_len, is_valid);

        // result = v * inv_len
        __m256 rx = _mm256_mul_ps(vx, inv_len);
        __m256 ry = _mm256_mul_ps(vy, inv_len);

        interleave8Vec2(rx, ry, result_ + i);
    }

    // Handle remainder
    for (; i < count_; ++i) {
        result_[i] = vec2NormalizeFast(input_[i]);
    }
}

#endif // AVX2

} // namespace detail
} // namespace MMath
