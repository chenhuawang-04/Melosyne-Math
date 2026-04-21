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

export module fast_math.sqrt_simd;

export import fast_math.sqrt;
export import fast_math.types;

export {
/**
 * @file sqrt_simd.h
 * @brief SIMD batch processing for square root functions
 *
 * Architecture:
 * - AVX2 path: Process 8 floats per iteration
 * - SSE4.1 path: Process 4 floats per iteration
 * - Scalar fallback: Process remaining elements
 *
 * Performance:
 * - sqrtArray: Same as std::sqrt loop (hardware instruction)
 * - inverseSqrtArray: ~2-3x speedup vs 1/std::sqrt loop
 * - reciprocalArray: ~2x speedup vs 1/x loop (on older CPUs)
 *
 * Use cases (from audio engine analysis):
 * - Vector normalization for large arrays
 * - Distance calculations for spatial audio
 * - Batch lighting attenuation
 */



#if defined(__AVX2__)
#elif defined(__SSE4_1__)
#endif

namespace MMath {
namespace detail {

// ============================================================================
// SIMD Implementation Functions (Internal)
// ============================================================================

/**
 * @brief SIMD sqrt for arrays (internal implementation)
 *
 * Uses hardware sqrtps/sqrtss instructions (exact).
 */
inline void sqrtSimd(
    float* MMATH_RESTRICT values_,
    int32_t count_
) noexcept {
    int32_t i = 0;

#if defined(__AVX2__)
    // AVX2 path: Process 8 floats at a time
    const int32_t avx2_count = (count_ / 8) * 8;
    for (; i < avx2_count; i += 8) {
        __m256 v = _mm256_loadu_ps(values_ + i);
        __m256 result = _mm256_sqrt_ps(v);
        _mm256_storeu_ps(values_ + i, result);
    }
#endif

#if defined(__SSE4_1__)
    // SSE4.1 path: Process 4 floats at a time
    const int32_t sse_count = (count_ / 4) * 4;
    for (; i < sse_count; i += 4) {
        __m128 v = _mm_loadu_ps(values_ + i);
        __m128 result = _mm_sqrt_ps(v);
        _mm_storeu_ps(values_ + i, result);
    }
#endif

    // Scalar fallback: Process remaining elements
    for (; i < count_; ++i) {
        values_[i] = MMath::sqrt(values_[i]);
    }
}

/**
 * @brief SIMD inverse sqrt for arrays (internal implementation)
 *
 * Uses rsqrt + Newton-Raphson refinement:
 * - Initial: y0 = rsqrt(x) (11-12 bit accuracy)
 * - Refined: y1 = 0.5 * y0 * (3.0 - x*y0*y0) (~23 bit accuracy)
 */
inline void inverseSqrtSimd(
    float* MMATH_RESTRICT values_,
    int32_t count_
) noexcept {
    int32_t i = 0;

#if defined(__AVX2__)
    // AVX2 path: Process 8 floats at a time
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 three = _mm256_set1_ps(3.0f);

    const int32_t avx2_count = (count_ / 8) * 8;
    for (; i < avx2_count; i += 8) {
        __m256 x = _mm256_loadu_ps(values_ + i);
        __m256 y0 = _mm256_rsqrt_ps(x);  // Initial approximation

        // Newton-Raphson: y1 = 0.5 * y0 * (3.0 - x*y0*y0)
        __m256 y0_squared = _mm256_mul_ps(y0, y0);
        __m256 x_y0_squared = _mm256_mul_ps(x, y0_squared);
        __m256 term = _mm256_sub_ps(three, x_y0_squared);
        __m256 y1 = _mm256_mul_ps(y0, term);
        y1 = _mm256_mul_ps(half, y1);

        _mm256_storeu_ps(values_ + i, y1);
    }
#endif

#if defined(__SSE4_1__)
    // SSE4.1 path: Process 4 floats at a time
    const __m128 half_sse = _mm_set1_ps(0.5f);
    const __m128 three_sse = _mm_set1_ps(3.0f);

    const int32_t sse_count = (count_ / 4) * 4;
    for (; i < sse_count; i += 4) {
        __m128 x = _mm_loadu_ps(values_ + i);
        __m128 y0 = _mm_rsqrt_ps(x);  // Initial approximation

        // Newton-Raphson: y1 = 0.5 * y0 * (3.0 - x*y0*y0)
        __m128 y0_squared = _mm_mul_ps(y0, y0);
        __m128 x_y0_squared = _mm_mul_ps(x, y0_squared);
        __m128 term = _mm_sub_ps(three_sse, x_y0_squared);
        __m128 y1 = _mm_mul_ps(y0, term);
        y1 = _mm_mul_ps(half_sse, y1);

        _mm_storeu_ps(values_ + i, y1);
    }
#endif

    // Scalar fallback
    for (; i < count_; ++i) {
        values_[i] = MMath::inverseSqrt(values_[i]);
    }
}

/**
 * @brief SIMD reciprocal for arrays (internal implementation)
 *
 * Uses rcp + Newton-Raphson refinement:
 * - Initial: y0 = rcp(x) (11-12 bit accuracy)
 * - Refined: y1 = y0 * (2.0 - x*y0) (~23 bit accuracy)
 */
inline void reciprocalSimd(
    float* MMATH_RESTRICT values_,
    int32_t count_
) noexcept {
    int32_t i = 0;

#if defined(__AVX2__)
    // AVX2 path: Process 8 floats at a time
    const __m256 two = _mm256_set1_ps(2.0f);

    const int32_t avx2_count = (count_ / 8) * 8;
    for (; i < avx2_count; i += 8) {
        __m256 x = _mm256_loadu_ps(values_ + i);
        __m256 y0 = _mm256_rcp_ps(x);  // Initial approximation

        // Newton-Raphson: y1 = y0 * (2.0 - x*y0)
        __m256 x_y0 = _mm256_mul_ps(x, y0);
        __m256 term = _mm256_sub_ps(two, x_y0);
        __m256 y1 = _mm256_mul_ps(y0, term);

        _mm256_storeu_ps(values_ + i, y1);
    }
#endif

#if defined(__SSE4_1__)
    // SSE4.1 path: Process 4 floats at a time
    const __m128 two_sse = _mm_set1_ps(2.0f);

    const int32_t sse_count = (count_ / 4) * 4;
    for (; i < sse_count; i += 4) {
        __m128 x = _mm_loadu_ps(values_ + i);
        __m128 y0 = _mm_rcp_ps(x);  // Initial approximation

        // Newton-Raphson: y1 = y0 * (2.0 - x*y0)
        __m128 x_y0 = _mm_mul_ps(x, y0);
        __m128 term = _mm_sub_ps(two_sse, x_y0);
        __m128 y1 = _mm_mul_ps(y0, term);

        _mm_storeu_ps(values_ + i, y1);
    }
#endif

    // Scalar fallback
    for (; i < count_; ++i) {
        values_[i] = MMath::reciprocal(values_[i]);
    }
}

} // namespace detail

// ============================================================================
// Public Array API
// ============================================================================

/**
 * @brief Square root of array elements (in-place)
 *
 * Formula: values[i] = sqrt(values[i])
 *
 * @param values_ Input/output array (modified in-place)
 * @param count_ Number of elements
 *
 * Use case: Distance calculations, magnitude computation
 * Example: sqrtArray(distance_squared, num_points) for spatial audio
 *
 * Performance: Same as std::sqrt loop (hardware instruction)
 */
inline void sqrtArray(
    float* MMATH_RESTRICT values_,
    int32_t count_
) noexcept {
    detail::sqrtSimd(values_, count_);
}

/**
 * @brief Inverse square root of array elements (in-place)
 *
 * Formula: values[i] = 1/sqrt(values[i])
 *
 * @param values_ Input/output array (modified in-place)
 * @param count_ Number of elements
 *
 * Use case: Vector normalization, lighting attenuation
 * Example: inverseSqrtArray(length_squared, num_vectors) for batch normalization
 *
 * Performance: ~2-3x faster than 1/std::sqrt loop
 * Accuracy: Relative error < 0.01% (Newton-Raphson refined)
 */
inline void inverseSqrtArray(
    float* MMATH_RESTRICT values_,
    int32_t count_
) noexcept {
    detail::inverseSqrtSimd(values_, count_);
}

/**
 * @brief Reciprocal of array elements (in-place)
 *
 * Formula: values[i] = 1/values[i]
 *
 * @param values_ Input/output array (modified in-place)
 * @param count_ Number of elements
 *
 * Use case: Division optimization in tight loops
 * Example: reciprocalArray(denominators, count) then multiply instead of divide
 *
 * Performance: ~2x faster than 1/x loop on older CPUs (Sandy Bridge, Haswell)
 * Note: On modern CPUs (Skylake+), hardware division is fast; profile before using
 * Accuracy: Relative error < 0.01% (Newton-Raphson refined)
 */
inline void reciprocalArray(
    float* MMATH_RESTRICT values_,
    int32_t count_
) noexcept {
    detail::reciprocalSimd(values_, count_);
}

} // namespace MMath
}
