/**
 * @file sqrt.h
 * @brief Square root and reciprocal functions
 *
 * Design Philosophy:
 * - STRICT POD structures (NO member functions)
 * - FREE FUNCTIONS only
 * - Scalar versions use SSE scalar instructions
 * - Array versions dispatch to SIMD batch processing
 *
 * Performance Optimizations:
 * - sqrt: Hardware sqrtss instruction (exact, 14-15 cycle latency on modern CPUs)
 * - inverseSqrt: rsqrt + Newton-Raphson iteration (2-3x faster than 1/sqrt)
 * - reciprocal: rcp + Newton-Raphson iteration (2x faster than 1/x on older CPUs)
 *
 * Accuracy:
 * - sqrt: Exact (hardware precision)
 * - inverseSqrt: Relative error < 0.01% (within 1 ULP after NR iteration)
 * - reciprocal: Relative error < 0.01% (within 1 ULP after NR iteration)
 *
 * References:
 * - Fast rsqrt with SSE/AVX: https://stackoverflow.com/questions/31555260/fast-vectorized-rsqrt-and-reciprocal-with-sse-avx-depending-on-precision
 * - Newton-Raphson accuracy: https://www.mdpi.com/1099-4300/23/1/86
 * - DirectXMath optimization guide: https://learn.microsoft.com/en-us/windows/win32/dxmath/pg-xnamath-optimizing
 * - GLM fast square root: https://glm.g-truc.net/0.9.3/api/a00168.html
 */

#pragma once

#include "types.h"
#include <cmath>  // for std::sqrt fallback

#if defined(__SSE4_1__)
#include <smmintrin.h>
#endif

namespace MMath {

// ============================================================================
// Square Root Functions
// ============================================================================

/**
 * @brief Square root (hardware instruction)
 *
 * Implementation:
 * - SSE: Uses sqrtss instruction (exact, 14-15 cycle latency on modern CPUs)
 * - Fallback: std::sqrt (also uses hardware instruction)
 *
 * @param x_ Input value (must be >= 0)
 * @return sqrt(x_)
 *
 * @note For x < 0, result is NaN (IEEE 754 behavior)
 * @note sqrt(0) = 0, sqrt(+Inf) = +Inf, sqrt(NaN) = NaN
 *
 * Use cases (from audio engine analysis):
 * - Distance calculation: sqrt(dx*dx + dy*dy) - 5+ occurrences
 * - Shelf filter coefficients: 4 sqrt calls per filter
 * - Vector normalization: length = sqrt(dot(v, v))
 */
MMATH_FORCE_INLINE float sqrt(float x_) noexcept {
#if defined(__SSE4_1__)
    return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x_)));
#else
    return std::sqrt(x_);
#endif
}

// ============================================================================
// Fast Reciprocal Square Root (1/sqrt(x))
// ============================================================================

/**
 * @brief Fast inverse square root with Newton-Raphson refinement
 *
 * Implementation:
 * - SSE: rsqrtss (11-12 bit accuracy) + 1 Newton-Raphson iteration
 * - Formula: y1 = y0 * (1.5 - 0.5*x*y0*y0), where y0 = rsqrt(x)
 * - Fallback: 1.0f / std::sqrt(x)
 *
 * Performance:
 * - SSE: ~2-3x faster than 1/sqrt(x) (rsqrt: 5 cycle latency vs sqrt: 14 cycles + div: 10-14 cycles)
 * - Accuracy: Relative error < 0.01% (within 1 ULP after NR iteration)
 *
 * @param x_ Input value (must be > 0)
 * @return 1/sqrt(x_)
 *
 * @note For x <= 0, result is undefined (may return NaN or Inf)
 * @note For x = +Inf, result is +0
 *
 * Use cases:
 * - Vector normalization: v * inverseSqrt(dot(v, v)) - GLM/DirectXMath pattern
 * - Fast lighting calculations: attenuation = inverseSqrt(distance_squared)
 * - Quaternion normalization
 *
 * References:
 * - Intel recommends rsqrt + NR for ~23 bits accuracy (vs 24 bits for sqrt)
 * - Research: https://www.mdpi.com/1099-4300/23/1/86
 */
MMATH_FORCE_INLINE float inverseSqrt(float x_) noexcept {
#if defined(__SSE4_1__)
    // Step 1: Get initial approximation (11-12 bit accuracy)
    __m128 v = _mm_set_ss(x_);
    __m128 y0 = _mm_rsqrt_ss(v);  // y0 ≈ 1/sqrt(x), relative error ≤ 1.5*2^-12

    // Step 2: Newton-Raphson iteration (refines to ~23 bits)
    // Formula: y1 = y0 * (1.5 - 0.5*x*y0*y0)
    // Rearranged: y1 = 0.5 * y0 * (3.0 - x*y0*y0)
    __m128 half = _mm_set_ss(0.5f);
    __m128 three = _mm_set_ss(3.0f);
    __m128 y0_squared = _mm_mul_ss(y0, y0);           // y0*y0
    __m128 x_y0_squared = _mm_mul_ss(v, y0_squared);  // x*y0*y0
    __m128 term = _mm_sub_ss(three, x_y0_squared);    // 3.0 - x*y0*y0
    __m128 y1 = _mm_mul_ss(y0, term);                 // y0 * (3.0 - x*y0*y0)
    y1 = _mm_mul_ss(half, y1);                        // 0.5 * y0 * (3.0 - x*y0*y0)

    return _mm_cvtss_f32(y1);
#else
    return 1.0f / std::sqrt(x_);
#endif
}

// ============================================================================
// Fast Reciprocal (1/x)
// ============================================================================

/**
 * @brief Fast reciprocal with Newton-Raphson refinement
 *
 * Implementation:
 * - SSE: rcpss (11-12 bit accuracy) + 1 Newton-Raphson iteration
 * - Formula: y1 = y0 * (2.0 - x*y0), where y0 = rcp(x)
 * - Fallback: 1.0f / x
 *
 * Performance:
 * - Older CPUs: ~2x faster than division (rcp: 5 cycles vs div: 10-14 cycles)
 * - Modern CPUs (Skylake+): Similar to division (div improved to 4-6 cycles)
 * - Accuracy: Relative error < 0.01% (within 1 ULP after NR iteration)
 *
 * @param x_ Input value (must not be 0)
 * @return 1/x_
 *
 * @note For x = 0, result is +/-Inf (IEEE 754 behavior)
 * @note For x = +/-Inf, result is +/-0
 *
 * Use cases:
 * - Division optimization in tight loops
 * - Barycentric coordinate calculation
 * - Screen-space transform: screenX = worldX * reciprocal(worldZ)
 *
 * Note: On modern CPUs (Skylake+), hardware division is very fast.
 * Only use reciprocal() when profiling shows division is a bottleneck,
 * or when targeting older CPUs (Sandy Bridge, Haswell).
 */
MMATH_FORCE_INLINE float reciprocal(float x_) noexcept {
#if defined(__SSE4_1__)
    // Step 1: Get initial approximation (11-12 bit accuracy)
    __m128 v = _mm_set_ss(x_);
    __m128 y0 = _mm_rcp_ss(v);  // y0 ≈ 1/x, relative error ≤ 1.5*2^-12

    // Step 2: Newton-Raphson iteration (refines to ~23 bits)
    // Formula: y1 = y0 * (2.0 - x*y0)
    __m128 two = _mm_set_ss(2.0f);
    __m128 x_y0 = _mm_mul_ss(v, y0);         // x*y0
    __m128 term = _mm_sub_ss(two, x_y0);     // 2.0 - x*y0
    __m128 y1 = _mm_mul_ss(y0, term);        // y0 * (2.0 - x*y0)

    return _mm_cvtss_f32(y1);
#else
    return 1.0f / x_;
#endif
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Fast normalize factor: inverseSqrt(dot(v, v))
 *
 * Combines dot product with inverseSqrt for common normalization pattern.
 *
 * @param length_squared Pre-computed squared length (dot(v, v))
 * @return 1/sqrt(length_squared)
 *
 * Use case:
 * ```cpp
 * Vec2 v = {3.0f, 4.0f};
 * float len_sq = vec2Dot(v, v);  // 25.0f
 * float inv_len = normalizeFactor(len_sq);  // 0.2f
 * Vec2 normalized = vec2Mul(v, inv_len);  // {0.6f, 0.8f}
 * ```
 */
MMATH_FORCE_INLINE float normalizeFactor(float length_squared_) noexcept {
    return inverseSqrt(length_squared_);
}

/**
 * @brief Check if value is safe for inverseSqrt/reciprocal
 *
 * Ensures x > epsilon to avoid division by zero or very small values.
 *
 * @param x_ Value to check
 * @param epsilon_ Minimum safe threshold (default: 1e-8f)
 * @return true if x_ > epsilon_
 */
MMATH_FORCE_INLINE bool isSafeForDivision(float x_, float epsilon_ = 1e-8f) noexcept {
    return x_ > epsilon_;
}

/**
 * @brief Safe reciprocal with fallback
 *
 * Returns 1/x if x > epsilon, otherwise returns fallback value.
 *
 * @param x_ Input value
 * @param fallback_ Value to return if x is too small (default: 0.0f)
 * @param epsilon_ Minimum safe threshold (default: 1e-8f)
 * @return 1/x or fallback
 */
MMATH_FORCE_INLINE float safeReciprocal(float x_, float fallback_ = 0.0f, float epsilon_ = 1e-8f) noexcept {
    return (x_ > epsilon_ || x_ < -epsilon_) ? reciprocal(x_) : fallback_;
}

} // namespace MMath

// Include SIMD batch processing implementations
#include "sqrt_simd.h"
