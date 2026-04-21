module;

#include "fast_math/config_macros.h"
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <limits>
#include <algorithm>
#include <bit>
#include <iterator>
#include <type_traits>
#include <memory>
#include <memory_resource>
#include <cfloat>
#include <immintrin.h>
#include <smmintrin.h>

export module fast_math.lerp_simd;

export import fast_math.lerp;
export import fast_math.types;

export {
/**
 * @file lerp_simd.h
 * @brief SIMD batch processing for interpolation functions
 *
 * Architecture:
 * - AVX2 + FMA path: Process 8 floats per iteration with fused operations
 * - SSE4.1 + FMA path: Process 4 floats per iteration with fused operations
 * - SSE4.1 path: Process 4 floats per iteration (standard mul+add)
 * - Scalar fallback: Process remaining elements
 *
 * Performance:
 * - lerpArray: ~1.5-2x speedup with FMA vs standard
 * - smoothstepArray: ~4-8x speedup vs scalar loop
 * - Batch processing amortizes loop overhead
 *
 * Use cases (from audio engine analysis):
 * - Fractional delay line interpolation (read between samples)
 * - Crossfade between audio buffers
 * - Parameter smoothing over time
 */



#if defined(__AVX2__) && defined(__FMA__)
#elif defined(__AVX2__)
#elif defined(__SSE4_1__) && defined(__FMA__)
#elif defined(__SSE4_1__)
#endif

namespace MMath {
namespace detail {

// ============================================================================
// SIMD Implementation Functions (Internal)
// ============================================================================

/**
 * @brief SIMD lerp for arrays (internal implementation)
 *
 * Formula: result[i] = a[i] + t * (b[i] - a[i])
 * With FMA: result[i] = fma(t, b[i] - a[i], a[i])
 *
 * Optimizations:
 * - Loop unrolling (process 2x SIMD registers per iteration)
 * - Software prefetching for large arrays
 * - Streaming stores for large arrays to avoid cache pollution
 */
inline void lerpSimd(
    const float* MMATH_RESTRICT a_,
    const float* MMATH_RESTRICT b_,
    float t_,
    float* MMATH_RESTRICT result_,
    int32_t count_
) noexcept {
    int32_t i = 0;
    const bool use_streaming = count_ > 16384;  // Use streaming stores for >64KB

#if defined(MMATH_HAS_AVX2_FMA)
    // AVX2 + FMA path with loop unrolling: Process 16 floats per iteration
    const __m256 vt = _mm256_set1_ps(t_);

    // Unrolled loop: 2x AVX2 registers (16 floats) per iteration
    const int32_t unroll_count = (count_ / 16) * 16;
    for (; i < unroll_count; i += 16) {
        // Prefetch next iteration's data (optimized distance: 32 floats = 128 bytes)
        _mm_prefetch(reinterpret_cast<const char*>(a_ + i + 32), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(b_ + i + 32), _MM_HINT_T0);

        // Load all data first (better instruction scheduling, hides load latency)
        __m256 va0 = _mm256_loadu_ps(a_ + i);
        __m256 va1 = _mm256_loadu_ps(a_ + i + 8);
        __m256 vb0 = _mm256_loadu_ps(b_ + i);
        __m256 vb1 = _mm256_loadu_ps(b_ + i + 8);

        // Compute both differences (independent operations, better ILP)
        __m256 diff0 = _mm256_sub_ps(vb0, va0);
        __m256 diff1 = _mm256_sub_ps(vb1, va1);

        // Compute both FMAs (independent operations, better ILP)
        __m256 result0 = _mm256_fmadd_ps(vt, diff0, va0);
        __m256 result1 = _mm256_fmadd_ps(vt, diff1, va1);

        // Store results
        if (use_streaming) {
            _mm256_stream_ps(result_ + i, result0);
            _mm256_stream_ps(result_ + i + 8, result1);
        } else {
            _mm256_storeu_ps(result_ + i, result0);
            _mm256_storeu_ps(result_ + i + 8, result1);
        }
    }

    // Remainder: single AVX2 register (8 floats)
    const int32_t avx2_count = (count_ / 8) * 8;
    for (; i < avx2_count; i += 8) {
        __m256 va = _mm256_loadu_ps(a_ + i);
        __m256 vb = _mm256_loadu_ps(b_ + i);
        __m256 diff = _mm256_sub_ps(vb, va);
        __m256 result = _mm256_fmadd_ps(vt, diff, va);
        _mm256_storeu_ps(result_ + i, result);
    }

    // Ensure streaming stores complete
    if (use_streaming && i > 0) {
        _mm_sfence();
    }
#elif defined(MMATH_HAS_SSE_FMA)
    // SSE + FMA path with loop unrolling: Process 8 floats per iteration
    const __m128 vt_sse = _mm_set1_ps(t_);

    // Unrolled loop: 2x SSE registers (8 floats) per iteration
    const int32_t unroll_count = (count_ / 8) * 8;
    for (; i < unroll_count; i += 8) {
        // Prefetch (optimized distance)
        _mm_prefetch(reinterpret_cast<const char*>(a_ + i + 16), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(b_ + i + 16), _MM_HINT_T0);

        // Load all data first (better instruction scheduling)
        __m128 va0 = _mm_loadu_ps(a_ + i);
        __m128 va1 = _mm_loadu_ps(a_ + i + 4);
        __m128 vb0 = _mm_loadu_ps(b_ + i);
        __m128 vb1 = _mm_loadu_ps(b_ + i + 4);

        // Compute both differences (independent operations)
        __m128 diff0 = _mm_sub_ps(vb0, va0);
        __m128 diff1 = _mm_sub_ps(vb1, va1);

        // Compute both FMAs (independent operations)
        __m128 result0 = _mm_fmadd_ps(vt_sse, diff0, va0);
        __m128 result1 = _mm_fmadd_ps(vt_sse, diff1, va1);

        // Store
        if (use_streaming) {
            _mm_stream_ps(result_ + i, result0);
            _mm_stream_ps(result_ + i + 4, result1);
        } else {
            _mm_storeu_ps(result_ + i, result0);
            _mm_storeu_ps(result_ + i + 4, result1);
        }
    }

    // Remainder: single SSE register (4 floats)
    const int32_t sse_count = (count_ / 4) * 4;
    for (; i < sse_count; i += 4) {
        __m128 va = _mm_loadu_ps(a_ + i);
        __m128 vb = _mm_loadu_ps(b_ + i);
        __m128 diff = _mm_sub_ps(vb, va);
        __m128 result = _mm_fmadd_ps(vt_sse, diff, va);
        _mm_storeu_ps(result_ + i, result);
    }

    if (use_streaming && i > 0) {
        _mm_sfence();
    }
#elif defined(__SSE4_1__)
    // SSE4.1 path (no FMA) with loop unrolling
    const __m128 vt_sse = _mm_set1_ps(t_);

    const int32_t unroll_count = (count_ / 8) * 8;
    for (; i < unroll_count; i += 8) {
        // Prefetch (optimized distance)
        _mm_prefetch(reinterpret_cast<const char*>(a_ + i + 16), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(b_ + i + 16), _MM_HINT_T0);

        // Load all data first (better instruction scheduling)
        __m128 va0 = _mm_loadu_ps(a_ + i);
        __m128 va1 = _mm_loadu_ps(a_ + i + 4);
        __m128 vb0 = _mm_loadu_ps(b_ + i);
        __m128 vb1 = _mm_loadu_ps(b_ + i + 4);

        // Compute both differences (independent operations)
        __m128 diff0 = _mm_sub_ps(vb0, va0);
        __m128 diff1 = _mm_sub_ps(vb1, va1);

        // Compute both scaled values (independent operations)
        __m128 scaled0 = _mm_mul_ps(vt_sse, diff0);
        __m128 scaled1 = _mm_mul_ps(vt_sse, diff1);

        // Compute both results (independent operations)
        __m128 result0 = _mm_add_ps(va0, scaled0);
        __m128 result1 = _mm_add_ps(va1, scaled1);

        if (use_streaming) {
            _mm_stream_ps(result_ + i, result0);
            _mm_stream_ps(result_ + i + 4, result1);
        } else {
            _mm_storeu_ps(result_ + i, result0);
            _mm_storeu_ps(result_ + i + 4, result1);
        }
    }

    const int32_t sse_count = (count_ / 4) * 4;
    for (; i < sse_count; i += 4) {
        __m128 va = _mm_loadu_ps(a_ + i);
        __m128 vb = _mm_loadu_ps(b_ + i);
        __m128 diff = _mm_sub_ps(vb, va);
        __m128 scaled = _mm_mul_ps(vt_sse, diff);
        __m128 result = _mm_add_ps(va, scaled);
        _mm_storeu_ps(result_ + i, result);
    }

    if (use_streaming && i > 0) {
        _mm_sfence();
    }
#endif

    // Scalar fallback
    for (; i < count_; ++i) {
        result_[i] = lerp(a_[i], b_[i], t_);
    }
}

/**
 * @brief SIMD smoothstep for arrays (internal implementation)
 *
 * Formula: x²(3 - 2x) where x = clamp((values[i] - edge0) / (edge1 - edge0), 0, 1)
 */
inline void smoothstepSimd(
    float edge0_,
    float edge1_,
    float* MMATH_RESTRICT values_,
    int32_t count_
) noexcept {
    int32_t i = 0;

    float inv_range = 1.0f / (edge1_ - edge0_);

#if defined(__AVX2__)
    const __m256 vedge0 = _mm256_set1_ps(edge0_);
    const __m256 vinv_range = _mm256_set1_ps(inv_range);
    const __m256 vzero = _mm256_setzero_ps();
    const __m256 vone = _mm256_set1_ps(1.0f);
    const __m256 vthree = _mm256_set1_ps(3.0f);
    const __m256 vtwo = _mm256_set1_ps(2.0f);

    const int32_t avx2_count = (count_ / 8) * 8;
    for (; i < avx2_count; i += 8) {
        __m256 vt = _mm256_loadu_ps(values_ + i);

        // Normalize: x = (t - edge0) / (edge1 - edge0)
        vt = _mm256_sub_ps(vt, vedge0);
        vt = _mm256_mul_ps(vt, vinv_range);

        // Clamp to [0, 1]
        vt = _mm256_max_ps(vt, vzero);
        vt = _mm256_min_ps(vt, vone);

        // Compute: x * x * (3 - 2*x)
        __m256 x2 = _mm256_mul_ps(vt, vt);  // x²
        __m256 two_x = _mm256_mul_ps(vtwo, vt);  // 2x
        __m256 term = _mm256_sub_ps(vthree, two_x);  // 3 - 2x
        __m256 result = _mm256_mul_ps(x2, term);  // x²(3 - 2x)

        _mm256_storeu_ps(values_ + i, result);
    }
#endif

#if defined(__SSE4_1__)
    const __m128 vedge0_sse = _mm_set1_ps(edge0_);
    const __m128 vinv_range_sse = _mm_set1_ps(inv_range);
    const __m128 vzero_sse = _mm_setzero_ps();
    const __m128 vone_sse = _mm_set1_ps(1.0f);
    const __m128 vthree_sse = _mm_set1_ps(3.0f);
    const __m128 vtwo_sse = _mm_set1_ps(2.0f);

    const int32_t sse_count = (count_ / 4) * 4;
    for (; i < sse_count; i += 4) {
        __m128 vt = _mm_loadu_ps(values_ + i);

        // Normalize
        vt = _mm_sub_ps(vt, vedge0_sse);
        vt = _mm_mul_ps(vt, vinv_range_sse);

        // Clamp
        vt = _mm_max_ps(vt, vzero_sse);
        vt = _mm_min_ps(vt, vone_sse);

        // Polynomial
        __m128 x2 = _mm_mul_ps(vt, vt);
        __m128 two_x = _mm_mul_ps(vtwo_sse, vt);
        __m128 term = _mm_sub_ps(vthree_sse, two_x);
        __m128 result = _mm_mul_ps(x2, term);

        _mm_storeu_ps(values_ + i, result);
    }
#endif

    // Scalar fallback
    for (; i < count_; ++i) {
        values_[i] = smoothstep(edge0_, edge1_, values_[i]);
    }
}

/**
 * @brief SIMD smootherstep for arrays (internal implementation)
 *
 * Formula: x³(x(x*6 - 15) + 10)
 */
inline void smootherstepSimd(
    float edge0_,
    float edge1_,
    float* MMATH_RESTRICT values_,
    int32_t count_
) noexcept {
    int32_t i = 0;

    float inv_range = 1.0f / (edge1_ - edge0_);

#if defined(__AVX2__)
    const __m256 vedge0 = _mm256_set1_ps(edge0_);
    const __m256 vinv_range = _mm256_set1_ps(inv_range);
    const __m256 vzero = _mm256_setzero_ps();
    const __m256 vone = _mm256_set1_ps(1.0f);
    const __m256 vsix = _mm256_set1_ps(6.0f);
    const __m256 vfifteen = _mm256_set1_ps(15.0f);
    const __m256 vten = _mm256_set1_ps(10.0f);

    const int32_t avx2_count = (count_ / 8) * 8;
    for (; i < avx2_count; i += 8) {
        __m256 vt = _mm256_loadu_ps(values_ + i);

        // Normalize
        vt = _mm256_sub_ps(vt, vedge0);
        vt = _mm256_mul_ps(vt, vinv_range);

        // Clamp
        vt = _mm256_max_ps(vt, vzero);
        vt = _mm256_min_ps(vt, vone);

        // Horner's method: x * x * x * (x * (x * 6 - 15) + 10)
        __m256 inner = _mm256_mul_ps(vt, vsix);  // x * 6
        inner = _mm256_sub_ps(inner, vfifteen);  // x * 6 - 15
        inner = _mm256_mul_ps(vt, inner);  // x * (x * 6 - 15)
        inner = _mm256_add_ps(inner, vten);  // x * (x * 6 - 15) + 10

        __m256 x3 = _mm256_mul_ps(vt, vt);  // x²
        x3 = _mm256_mul_ps(x3, vt);  // x³

        __m256 result = _mm256_mul_ps(x3, inner);  // x³ * (...)

        _mm256_storeu_ps(values_ + i, result);
    }
#endif

#if defined(__SSE4_1__)
    const __m128 vedge0_sse = _mm_set1_ps(edge0_);
    const __m128 vinv_range_sse = _mm_set1_ps(inv_range);
    const __m128 vzero_sse = _mm_setzero_ps();
    const __m128 vone_sse = _mm_set1_ps(1.0f);
    const __m128 vsix_sse = _mm_set1_ps(6.0f);
    const __m128 vfifteen_sse = _mm_set1_ps(15.0f);
    const __m128 vten_sse = _mm_set1_ps(10.0f);

    const int32_t sse_count = (count_ / 4) * 4;
    for (; i < sse_count; i += 4) {
        __m128 vt = _mm_loadu_ps(values_ + i);

        // Normalize
        vt = _mm_sub_ps(vt, vedge0_sse);
        vt = _mm_mul_ps(vt, vinv_range_sse);

        // Clamp
        vt = _mm_max_ps(vt, vzero_sse);
        vt = _mm_min_ps(vt, vone_sse);

        // Horner's method
        __m128 inner = _mm_mul_ps(vt, vsix_sse);
        inner = _mm_sub_ps(inner, vfifteen_sse);
        inner = _mm_mul_ps(vt, inner);
        inner = _mm_add_ps(inner, vten_sse);

        __m128 x3 = _mm_mul_ps(vt, vt);
        x3 = _mm_mul_ps(x3, vt);

        __m128 result = _mm_mul_ps(x3, inner);

        _mm_storeu_ps(values_ + i, result);
    }
#endif

    // Scalar fallback
    for (; i < count_; ++i) {
        values_[i] = smootherstep(edge0_, edge1_, values_[i]);
    }
}

/**
 * @brief SIMD remap for arrays (internal implementation)
 */
inline void remapSimd(
    const float* MMATH_RESTRICT values_,
    float in_min_,
    float in_max_,
    float out_min_,
    float out_max_,
    float* MMATH_RESTRICT result_,
    int32_t count_
) noexcept {
    int32_t i = 0;

    float in_range = in_max_ - in_min_;
    float out_range = out_max_ - out_min_;

    if (MMath::abs(in_range) < 1e-8f) {
        // Degenerate case: fill with out_min
        for (i = 0; i < count_; ++i) {
            result_[i] = out_min_;
        }
        return;
    }

    float scale = out_range / in_range;

#if defined(__AVX2__)
    const __m256 vin_min = _mm256_set1_ps(in_min_);
    const __m256 vscale = _mm256_set1_ps(scale);
    const __m256 vout_min = _mm256_set1_ps(out_min_);

    const int32_t avx2_count = (count_ / 8) * 8;
    for (; i < avx2_count; i += 8) {
        __m256 v = _mm256_loadu_ps(values_ + i);
        v = _mm256_sub_ps(v, vin_min);  // value - in_min
        v = _mm256_mul_ps(v, vscale);  // * scale
        v = _mm256_add_ps(v, vout_min);  // + out_min
        _mm256_storeu_ps(result_ + i, v);
    }
#endif

#if defined(__SSE4_1__)
    const __m128 vin_min_sse = _mm_set1_ps(in_min_);
    const __m128 vscale_sse = _mm_set1_ps(scale);
    const __m128 vout_min_sse = _mm_set1_ps(out_min_);

    const int32_t sse_count = (count_ / 4) * 4;
    for (; i < sse_count; i += 4) {
        __m128 v = _mm_loadu_ps(values_ + i);
        v = _mm_sub_ps(v, vin_min_sse);
        v = _mm_mul_ps(v, vscale_sse);
        v = _mm_add_ps(v, vout_min_sse);
        _mm_storeu_ps(result_ + i, v);
    }
#endif

    // Scalar fallback
    for (; i < count_; ++i) {
        result_[i] = remap(values_[i], in_min_, in_max_, out_min_, out_max_);
    }
}

} // namespace detail

// ============================================================================
// Public Array API
// ============================================================================

/**
 * @brief Linear interpolation for arrays (in-place on result)
 *
 * Formula: result[i] = a[i] + t * (b[i] - a[i])
 *
 * @param a_ First input array
 * @param b_ Second input array
 * @param t_ Interpolation parameter (constant for all elements)
 * @param result_ Output array (can be same as a_ or b_ for in-place operation)
 * @param count_ Number of elements
 *
 * Use case: Crossfade between two audio buffers
 * Example: lerpArray(buffer_a, buffer_b, fade_amount, output, 4096)
 *
 * Performance: ~1.5-2x faster than scalar loop with FMA
 */
inline void lerpArray(
    const float* MMATH_RESTRICT a_,
    const float* MMATH_RESTRICT b_,
    float t_,
    float* MMATH_RESTRICT result_,
    int32_t count_
) noexcept {
    detail::lerpSimd(a_, b_, t_, result_, count_);
}

/**
 * @brief Smooth Hermite interpolation for arrays (in-place)
 *
 * Applies smoothstep to each element of the array.
 *
 * @param edge0_ Lower edge
 * @param edge1_ Upper edge
 * @param values_ Input/output array (modified in-place)
 * @param count_ Number of elements
 *
 * Use case: Smooth fade curve for volume envelope
 *
 * Performance: ~4-8x faster than scalar loop
 */
inline void smoothstepArray(
    float edge0_,
    float edge1_,
    float* MMATH_RESTRICT values_,
    int32_t count_
) noexcept {
    detail::smoothstepSimd(edge0_, edge1_, values_, count_);
}

/**
 * @brief Smoother Hermite interpolation for arrays (in-place)
 *
 * Applies smootherstep to each element of the array.
 *
 * @param edge0_ Lower edge
 * @param edge1_ Upper edge
 * @param values_ Input/output array (modified in-place)
 * @param count_ Number of elements
 *
 * Performance: ~4-8x faster than scalar loop
 */
inline void smootherstepArray(
    float edge0_,
    float edge1_,
    float* MMATH_RESTRICT values_,
    int32_t count_
) noexcept {
    detail::smootherstepSimd(edge0_, edge1_, values_, count_);
}

/**
 * @brief Remap array values from one range to another
 *
 * Maps each value from [in_min, in_max] to [out_min, out_max].
 *
 * @param values_ Input array
 * @param in_min_ Input range minimum
 * @param in_max_ Input range maximum
 * @param out_min_ Output range minimum
 * @param out_max_ Output range maximum
 * @param result_ Output array
 * @param count_ Number of elements
 *
 * Use case: Normalize sensor readings array to [0, 1]
 */
inline void remapArray(
    const float* MMATH_RESTRICT values_,
    float in_min_,
    float in_max_,
    float out_min_,
    float out_max_,
    float* MMATH_RESTRICT result_,
    int32_t count_
) noexcept {
    detail::remapSimd(values_, in_min_, in_max_, out_min_, out_max_, result_, count_);
}

} // namespace MMath
}
