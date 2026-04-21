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

export module fast_math.power_simd;

export import fast_math.types;
export import fast_math.power;

export {
/**
 * @file power_simd.h
 * @brief SIMD batch processing for exponential and power functions
 *
 * Architecture:
 * - AVX2 path: Process 8 floats per iteration
 * - SSE4.1 path: Process 4 floats per iteration
 * - Scalar fallback: Process remaining elements
 *
 * Performance:
 * - expArray: ~4-6x speedup vs scalar loop
 * - logArray: ~3-5x speedup vs scalar loop
 * - powArray: ~5-8x speedup vs scalar loop
 *
 * Use cases (from audio engine analysis):
 * - Batch dB to linear conversion (pow10Array)
 * - Exponential decay envelopes (expArray)
 * - Distance attenuation calculations (powArray)
 */



#if defined(__AVX2__)
#elif defined(__SSE4_1__)
#endif

namespace MMath {
namespace detail {

// ============================================================================
// SIMD Constants
// ============================================================================

#if defined(__AVX2__)
    inline const __m256 kLn2_256 = _mm256_set1_ps(0.693147180559945309417232121458176568f);
    inline const __m256 kLog2E_256 = _mm256_set1_ps(1.44269504088896340735992468100189214f);
    inline const __m256 kLn10_256 = _mm256_set1_ps(2.30258509299404568401799145468436421f);
    inline const __m256 kLog10E_256 = _mm256_set1_ps(0.434294481903251827651128918916605082f);

    inline const __m256 kExpMaxInput_256 = _mm256_set1_ps(88.376f);
    inline const __m256 kExpMinInput_256 = _mm256_set1_ps(-87.336f);

    inline const __m256 kOne_256 = _mm256_set1_ps(1.0f);
    inline const __m256 kHalf_256 = _mm256_set1_ps(0.5f);
    inline const __m256 kZero_256 = _mm256_setzero_ps();

    // exp polynomial coefficients
    inline const __m256 kExpC1_256 = _mm256_set1_ps(1.0f);
    inline const __m256 kExpC2_256 = _mm256_set1_ps(0.5f);
    inline const __m256 kExpC3_256 = _mm256_set1_ps(0.166666666666666851703837234430154323f);
    inline const __m256 kExpC4_256 = _mm256_set1_ps(0.0416666666666665970564002484798622411f);
    inline const __m256 kExpC5_256 = _mm256_set1_ps(0.00833333333333328885369231554654299327f);

    // log polynomial coefficients
    inline const __m256 kLogP0_256 = _mm256_set1_ps(7.0376836292E-2f);
    inline const __m256 kLogP1_256 = _mm256_set1_ps(-1.1514610310E-1f);
    inline const __m256 kLogP2_256 = _mm256_set1_ps(1.1676998740E-1f);
    inline const __m256 kLogP3_256 = _mm256_set1_ps(-1.2420140846E-1f);
    inline const __m256 kLogP4_256 = _mm256_set1_ps(1.4249322787E-1f);
    inline const __m256 kLogP5_256 = _mm256_set1_ps(-1.6668057665E-1f);
    inline const __m256 kLogP6_256 = _mm256_set1_ps(2.0000714765E-1f);
    inline const __m256 kLogP7_256 = _mm256_set1_ps(-2.4999993993E-1f);
    inline const __m256 kLogP8_256 = _mm256_set1_ps(3.3333331174E-1f);

    inline const __m256 kSqrt2Over2_256 = _mm256_set1_ps(0.707106781186547524400844362104849039f);

    inline const __m256i kExpBias_256i = _mm256_set1_epi32(127);
    inline const __m256i k23_256i = _mm256_set1_epi32(23);
    inline const __m256i k0x7FFFFF_256i = _mm256_set1_epi32(0x007FFFFF);
    inline const __m256i k0x3F800000_256i = _mm256_set1_epi32(0x3F800000);
    inline const __m256i k0xFF_256i = _mm256_set1_epi32(0xFF);
#endif

#if defined(__SSE4_1__)
    inline const __m128 kLn2_128 = _mm_set1_ps(0.693147180559945309417232121458176568f);
    inline const __m128 kLog2E_128 = _mm_set1_ps(1.44269504088896340735992468100189214f);
    inline const __m128 kLn10_128 = _mm_set1_ps(2.30258509299404568401799145468436421f);
    inline const __m128 kLog10E_128 = _mm_set1_ps(0.434294481903251827651128918916605082f);

    inline const __m128 kExpMaxInput_128 = _mm_set1_ps(88.376f);
    inline const __m128 kExpMinInput_128 = _mm_set1_ps(-87.336f);

    inline const __m128 kOne_128 = _mm_set1_ps(1.0f);
    inline const __m128 kHalf_128 = _mm_set1_ps(0.5f);
    inline const __m128 kZero_128 = _mm_setzero_ps();

    // exp polynomial coefficients
    inline const __m128 kExpC1_128 = _mm_set1_ps(1.0f);
    inline const __m128 kExpC2_128 = _mm_set1_ps(0.5f);
    inline const __m128 kExpC3_128 = _mm_set1_ps(0.166666666666666851703837234430154323f);
    inline const __m128 kExpC4_128 = _mm_set1_ps(0.0416666666666665970564002484798622411f);
    inline const __m128 kExpC5_128 = _mm_set1_ps(0.00833333333333328885369231554654299327f);

    // log polynomial coefficients
    inline const __m128 kLogP0_128 = _mm_set1_ps(7.0376836292E-2f);
    inline const __m128 kLogP1_128 = _mm_set1_ps(-1.1514610310E-1f);
    inline const __m128 kLogP2_128 = _mm_set1_ps(1.1676998740E-1f);
    inline const __m128 kLogP3_128 = _mm_set1_ps(-1.2420140846E-1f);
    inline const __m128 kLogP4_128 = _mm_set1_ps(1.4249322787E-1f);
    inline const __m128 kLogP5_128 = _mm_set1_ps(-1.6668057665E-1f);
    inline const __m128 kLogP6_128 = _mm_set1_ps(2.0000714765E-1f);
    inline const __m128 kLogP7_128 = _mm_set1_ps(-2.4999993993E-1f);
    inline const __m128 kLogP8_128 = _mm_set1_ps(3.3333331174E-1f);

    inline const __m128 kSqrt2Over2_128 = _mm_set1_ps(0.707106781186547524400844362104849039f);

    inline const __m128i kExpBias_128i = _mm_set1_epi32(127);
    inline const __m128i k23_128i = _mm_set1_epi32(23);
    inline const __m128i k0x7FFFFF_128i = _mm_set1_epi32(0x007FFFFF);
    inline const __m128i k0x3F800000_128i = _mm_set1_epi32(0x3F800000);
    inline const __m128i k0xFF_128i = _mm_set1_epi32(0xFF);
#endif

// ============================================================================
// SIMD exp Implementation (Internal)
// ============================================================================

/**
 * @brief SIMD exp for arrays (internal implementation)
 *
 * Algorithm: Range reduction + polynomial approximation
 */
inline void expSimd(
    float* MMATH_RESTRICT values_,
    int32_t count_
) noexcept {
    int32_t i = 0;

#if defined(__AVX2__)
    // AVX2 path: Process 8 floats at a time
    const int32_t avx2_count = (count_ / 8) * 8;
    for (; i < avx2_count; i += 8) {
        __m256 x = _mm256_loadu_ps(values_ + i);

        // Clamp input
        x = _mm256_min_ps(x, kExpMaxInput_256);
        x = _mm256_max_ps(x, kExpMinInput_256);

        // Range reduction: x = k*ln(2) + r
        __m256 fx = _mm256_mul_ps(x, kLog2E_256);
        fx = _mm256_add_ps(fx, kHalf_256);
        fx = _mm256_floor_ps(fx);  // k = floor(x * log2(e) + 0.5)

        __m256 r = _mm256_sub_ps(x, _mm256_mul_ps(fx, kLn2_256));  // r = x - k*ln(2)

        // Polynomial approximation: exp(r) using Horner's method
        // exp(r) ≈ 1 + r + r²/2! + r³/3! + r⁴/4! + r⁵/5!
        // = 1 + r*(1 + r*(1/2 + r*(1/6 + r*(1/24 + r/120))))
        __m256 poly = kExpC5_256;  // 1/120
        poly = _mm256_add_ps(kExpC4_256, _mm256_mul_ps(r, poly));  // 1/24 + r*poly
        poly = _mm256_add_ps(kExpC3_256, _mm256_mul_ps(r, poly));  // 1/6 + r*poly
        poly = _mm256_add_ps(kExpC2_256, _mm256_mul_ps(r, poly));  // 1/2 + r*poly (FIXED: was r2)
        poly = _mm256_add_ps(kExpC1_256, _mm256_mul_ps(r, poly));  // 1 + r*poly
        poly = _mm256_add_ps(kOne_256, _mm256_mul_ps(r, poly));    // 1 + r*(...)

        // Reconstruction: exp(x) = 2^k * exp(r)
        // Convert k to integer and compute 2^k via bit manipulation
        __m256i ki = _mm256_cvtps_epi32(fx);
        ki = _mm256_add_epi32(ki, kExpBias_256i);  // Add IEEE 754 exponent bias (127)
        ki = _mm256_slli_epi32(ki, 23);  // Shift to exponent field position
        __m256 pow2 = _mm256_castsi256_ps(ki);

        __m256 result = _mm256_mul_ps(poly, pow2);
        _mm256_storeu_ps(values_ + i, result);
    }
#endif

#if defined(__SSE4_1__)
    // SSE4.1 path: Process 4 floats at a time
    const int32_t sse_count = (count_ / 4) * 4;
    for (; i < sse_count; i += 4) {
        __m128 x = _mm_loadu_ps(values_ + i);

        // Clamp input
        x = _mm_min_ps(x, kExpMaxInput_128);
        x = _mm_max_ps(x, kExpMinInput_128);

        // Range reduction: x = k*ln(2) + r
        __m128 fx = _mm_mul_ps(x, kLog2E_128);
        fx = _mm_add_ps(fx, kHalf_128);
        fx = _mm_floor_ps(fx);  // k = floor(x * log2(e) + 0.5)

        __m128 r = _mm_sub_ps(x, _mm_mul_ps(fx, kLn2_128));  // r = x - k*ln(2)

        // Polynomial approximation
        __m128 poly = kExpC5_128;  // 1/120
        poly = _mm_add_ps(kExpC4_128, _mm_mul_ps(r, poly));
        poly = _mm_add_ps(kExpC3_128, _mm_mul_ps(r, poly));
        poly = _mm_add_ps(kExpC2_128, _mm_mul_ps(r, poly));  // FIXED: was r2
        poly = _mm_add_ps(kExpC1_128, _mm_mul_ps(r, poly));
        poly = _mm_add_ps(kOne_128, _mm_mul_ps(r, poly));

        // Reconstruction: 2^k * exp(r)
        __m128i ki = _mm_cvtps_epi32(fx);
        ki = _mm_add_epi32(ki, kExpBias_128i);
        ki = _mm_slli_epi32(ki, 23);
        __m128 pow2 = _mm_castsi128_ps(ki);

        __m128 result = _mm_mul_ps(poly, pow2);
        _mm_storeu_ps(values_ + i, result);
    }
#endif

    // Scalar fallback
    for (; i < count_; ++i) {
        values_[i] = MMath::exp(values_[i]);
    }
}

// ============================================================================
// SIMD log Implementation (Internal)
// ============================================================================

/**
 * @brief SIMD log for arrays (internal implementation)
 *
 * Algorithm: Extract exponent + polynomial approximation
 */
inline void logSimd(
    float* MMATH_RESTRICT values_,
    int32_t count_
) noexcept {
    int32_t i = 0;

#if defined(__AVX2__)
    // AVX2 path: Process 8 floats at a time
    const int32_t avx2_count = (count_ / 8) * 8;
    for (; i < avx2_count; i += 8) {
        __m256 x = _mm256_loadu_ps(values_ + i);

        // Extract exponent and mantissa
        __m256i bits = _mm256_castps_si256(x);
        __m256i exponent_bits = _mm256_srli_epi32(bits, 23);
        exponent_bits = _mm256_and_si256(exponent_bits, k0xFF_256i);
        __m256i exponent_i = _mm256_sub_epi32(exponent_bits, kExpBias_256i);
        __m256 exponent_f = _mm256_cvtepi32_ps(exponent_i);

        // Extract mantissa (force exponent to 0)
        __m256i mantissa_bits = _mm256_and_si256(bits, k0x7FFFFF_256i);
        mantissa_bits = _mm256_or_si256(mantissa_bits, k0x3F800000_256i);
        __m256 mantissa = _mm256_castsi256_ps(mantissa_bits);

        // Range adjustment: if mantissa < sqrt(2)/2, multiply by 2 and decrement exponent
        __m256 mask = _mm256_cmp_ps(mantissa, kSqrt2Over2_256, _CMP_LT_OQ);
        mantissa = _mm256_blendv_ps(mantissa, _mm256_mul_ps(mantissa, _mm256_set1_ps(2.0f)), mask);
        exponent_f = _mm256_blendv_ps(exponent_f, _mm256_sub_ps(exponent_f, kOne_256), mask);

        // Transform to y = m - 1
        __m256 y = _mm256_sub_ps(mantissa, kOne_256);
        __m256 z = _mm256_mul_ps(y, y);

        // Polynomial approximation using Horner's method
        __m256 poly = kLogP0_256;
        poly = _mm256_add_ps(kLogP1_256, _mm256_mul_ps(y, poly));
        poly = _mm256_add_ps(kLogP2_256, _mm256_mul_ps(y, poly));
        poly = _mm256_add_ps(kLogP3_256, _mm256_mul_ps(y, poly));
        poly = _mm256_add_ps(kLogP4_256, _mm256_mul_ps(y, poly));
        poly = _mm256_add_ps(kLogP5_256, _mm256_mul_ps(y, poly));
        poly = _mm256_add_ps(kLogP6_256, _mm256_mul_ps(y, poly));
        poly = _mm256_add_ps(kLogP7_256, _mm256_mul_ps(y, poly));
        poly = _mm256_add_ps(kLogP8_256, _mm256_mul_ps(y, poly));
        poly = _mm256_mul_ps(y, _mm256_mul_ps(poly, z));

        // Add linear term: -0.5*z + y
        poly = _mm256_add_ps(poly, _mm256_mul_ps(_mm256_set1_ps(-0.5f), z));
        poly = _mm256_add_ps(poly, y);

        // Reconstruction: log(x) = e*ln(2) + poly
        __m256 result = _mm256_add_ps(_mm256_mul_ps(exponent_f, kLn2_256), poly);
        _mm256_storeu_ps(values_ + i, result);
    }
#endif

#if defined(__SSE4_1__)
    // SSE4.1 path: Process 4 floats at a time
    const int32_t sse_count = (count_ / 4) * 4;
    for (; i < sse_count; i += 4) {
        __m128 x = _mm_loadu_ps(values_ + i);

        // Extract exponent and mantissa
        __m128i bits = _mm_castps_si128(x);
        __m128i exponent_bits = _mm_srli_epi32(bits, 23);
        exponent_bits = _mm_and_si128(exponent_bits, k0xFF_128i);
        __m128i exponent_i = _mm_sub_epi32(exponent_bits, kExpBias_128i);
        __m128 exponent_f = _mm_cvtepi32_ps(exponent_i);

        // Extract mantissa
        __m128i mantissa_bits = _mm_and_si128(bits, k0x7FFFFF_128i);
        mantissa_bits = _mm_or_si128(mantissa_bits, k0x3F800000_128i);
        __m128 mantissa = _mm_castsi128_ps(mantissa_bits);

        // Range adjustment
        __m128 mask = _mm_cmplt_ps(mantissa, kSqrt2Over2_128);
        mantissa = _mm_blendv_ps(mantissa, _mm_mul_ps(mantissa, _mm_set1_ps(2.0f)), mask);
        exponent_f = _mm_blendv_ps(exponent_f, _mm_sub_ps(exponent_f, kOne_128), mask);

        // Transform and polynomial
        __m128 y = _mm_sub_ps(mantissa, kOne_128);
        __m128 z = _mm_mul_ps(y, y);

        __m128 poly = kLogP0_128;
        poly = _mm_add_ps(kLogP1_128, _mm_mul_ps(y, poly));
        poly = _mm_add_ps(kLogP2_128, _mm_mul_ps(y, poly));
        poly = _mm_add_ps(kLogP3_128, _mm_mul_ps(y, poly));
        poly = _mm_add_ps(kLogP4_128, _mm_mul_ps(y, poly));
        poly = _mm_add_ps(kLogP5_128, _mm_mul_ps(y, poly));
        poly = _mm_add_ps(kLogP6_128, _mm_mul_ps(y, poly));
        poly = _mm_add_ps(kLogP7_128, _mm_mul_ps(y, poly));
        poly = _mm_add_ps(kLogP8_128, _mm_mul_ps(y, poly));
        poly = _mm_mul_ps(y, _mm_mul_ps(poly, z));

        poly = _mm_add_ps(poly, _mm_mul_ps(_mm_set1_ps(-0.5f), z));
        poly = _mm_add_ps(poly, y);

        // Reconstruction
        __m128 result = _mm_add_ps(_mm_mul_ps(exponent_f, kLn2_128), poly);
        _mm_storeu_ps(values_ + i, result);
    }
#endif

    // Scalar fallback
    for (; i < count_; ++i) {
        values_[i] = MMath::log(values_[i]);
    }
}

// ============================================================================
// Helper SIMD Functions
// ============================================================================

/**
 * @brief SIMD exp2 for arrays (internal)
 */
inline void exp2Simd(float* MMATH_RESTRICT values_, int32_t count_) noexcept {
    // exp2(x) = exp(x * ln(2))
    int32_t i = 0;

#if defined(__AVX2__)
    const int32_t avx2_count = (count_ / 8) * 8;
    for (; i < avx2_count; i += 8) {
        __m256 x = _mm256_loadu_ps(values_ + i);
        x = _mm256_mul_ps(x, kLn2_256);
        _mm256_storeu_ps(values_ + i, x);
    }
#endif

#if defined(__SSE4_1__)
    const int32_t sse_count = (count_ / 4) * 4;
    for (; i < sse_count; i += 4) {
        __m128 x = _mm_loadu_ps(values_ + i);
        x = _mm_mul_ps(x, kLn2_128);
        _mm_storeu_ps(values_ + i, x);
    }
#endif

    for (; i < count_; ++i) {
        values_[i] *= detail::kLn2;
    }

    expSimd(values_, count_);
}

/**
 * @brief SIMD pow10 for arrays (internal)
 */
inline void pow10Simd(float* MMATH_RESTRICT values_, int32_t count_) noexcept {
    // pow10(x) = exp(x * ln(10))
    int32_t i = 0;

#if defined(__AVX2__)
    const int32_t avx2_count = (count_ / 8) * 8;
    for (; i < avx2_count; i += 8) {
        __m256 x = _mm256_loadu_ps(values_ + i);
        x = _mm256_mul_ps(x, kLn10_256);
        _mm256_storeu_ps(values_ + i, x);
    }
#endif

#if defined(__SSE4_1__)
    const int32_t sse_count = (count_ / 4) * 4;
    for (; i < sse_count; i += 4) {
        __m128 x = _mm_loadu_ps(values_ + i);
        x = _mm_mul_ps(x, kLn10_128);
        _mm_storeu_ps(values_ + i, x);
    }
#endif

    for (; i < count_; ++i) {
        values_[i] *= detail::kLn10;
    }

    expSimd(values_, count_);
}

/**
 * @brief SIMD log2 for arrays (internal)
 */
inline void log2Simd(float* MMATH_RESTRICT values_, int32_t count_) noexcept {
    logSimd(values_, count_);

    // Multiply by log2(e) = 1/ln(2)
    int32_t i = 0;

#if defined(__AVX2__)
    const int32_t avx2_count = (count_ / 8) * 8;
    for (; i < avx2_count; i += 8) {
        __m256 x = _mm256_loadu_ps(values_ + i);
        x = _mm256_mul_ps(x, kLog2E_256);
        _mm256_storeu_ps(values_ + i, x);
    }
#endif

#if defined(__SSE4_1__)
    const int32_t sse_count = (count_ / 4) * 4;
    for (; i < sse_count; i += 4) {
        __m128 x = _mm_loadu_ps(values_ + i);
        x = _mm_mul_ps(x, kLog2E_128);
        _mm_storeu_ps(values_ + i, x);
    }
#endif

    for (; i < count_; ++i) {
        values_[i] *= detail::kLog2E;
    }
}

/**
 * @brief SIMD log10 for arrays (internal)
 */
inline void log10Simd(float* MMATH_RESTRICT values_, int32_t count_) noexcept {
    logSimd(values_, count_);

    // Multiply by log10(e) = 1/ln(10)
    int32_t i = 0;

#if defined(__AVX2__)
    const int32_t avx2_count = (count_ / 8) * 8;
    for (; i < avx2_count; i += 8) {
        __m256 x = _mm256_loadu_ps(values_ + i);
        x = _mm256_mul_ps(x, kLog10E_256);
        _mm256_storeu_ps(values_ + i, x);
    }
#endif

#if defined(__SSE4_1__)
    const int32_t sse_count = (count_ / 4) * 4;
    for (; i < sse_count; i += 4) {
        __m128 x = _mm_loadu_ps(values_ + i);
        x = _mm_mul_ps(x, kLog10E_128);
        _mm_storeu_ps(values_ + i, x);
    }
#endif

    for (; i < count_; ++i) {
        values_[i] *= detail::kLog10E;
    }
}

/**
 * @brief SIMD pow for arrays (internal)
 *
 * Computes result[i] = base[i]^exponent[i]
 */
inline void powSimd(
    const float* MMATH_RESTRICT base_,
    const float* MMATH_RESTRICT exponent_,
    float* MMATH_RESTRICT result_,
    int32_t count_
) noexcept {
    // pow(x, y) = exp(y * log(x))
    // First compute log(base)
    for (int32_t i = 0; i < count_; ++i) {
        result_[i] = base_[i];
    }
    logSimd(result_, count_);

    // Multiply by exponent (with special case handling)
    int32_t i = 0;

#if defined(__AVX2__)
    const __m256 vone = _mm256_set1_ps(1.0f);
    const int32_t avx2_count = (count_ / 8) * 8;
    for (; i < avx2_count; i += 8) {
        __m256 log_x = _mm256_loadu_ps(result_ + i);
        __m256 y = _mm256_loadu_ps(exponent_ + i);
        __m256 base_val = _mm256_loadu_ps(base_ + i);

        // If y == 1, use base directly to avoid log+exp error accumulation
        __m256 product = _mm256_mul_ps(y, log_x);
        __m256 mask = _mm256_cmp_ps(y, vone, _CMP_EQ_OQ);  // y == 1.0f
        product = _mm256_blendv_ps(product, _mm256_setzero_ps(), mask);  // Set to 0 if y==1 (exp(0)=1)

        _mm256_storeu_ps(result_ + i, product);
    }
#endif

#if defined(__SSE4_1__)
    const __m128 vone_sse = _mm_set1_ps(1.0f);
    const int32_t sse_count = (count_ / 4) * 4;
    for (; i < sse_count; i += 4) {
        __m128 log_x = _mm_loadu_ps(result_ + i);
        __m128 y = _mm_loadu_ps(exponent_ + i);

        __m128 product = _mm_mul_ps(y, log_x);
        __m128 mask = _mm_cmpeq_ps(y, vone_sse);  // y == 1.0f
        product = _mm_blendv_ps(product, _mm_setzero_ps(), mask);  // Set to 0 if y==1

        _mm_storeu_ps(result_ + i, product);
    }
#endif

    // Scalar fallback with special case handling
    for (; i < count_; ++i) {
        if (exponent_[i] == 1.0f) {
            result_[i] = 0.0f;  // exp(0) = 1, will be multiplied by base later
        } else {
            result_[i] *= exponent_[i];
        }
    }

    // Compute exp(y * log(x))
    expSimd(result_, count_);

    // For y==1 cases, replace with original base value
    for (i = 0; i < count_; ++i) {
        if (exponent_[i] == 1.0f) {
            result_[i] = base_[i];
        }
    }
}

} // namespace detail

// ============================================================================
// Public Array API
// ============================================================================

/**
 * @brief Natural exponential for arrays (in-place)
 *
 * Formula: values[i] = e^values[i]
 *
 * @param values_ Input/output array (modified in-place)
 * @param count_ Number of elements
 *
 * Use case: Exponential decay envelopes
 * Example: expArray(decay_values, num_samples)
 *
 * Performance: ~4-6x faster than scalar loop
 */
inline void expArray(
    float* MMATH_RESTRICT values_,
    int32_t count_
) noexcept {
    detail::expSimd(values_, count_);
}

/**
 * @brief Base-2 exponential for arrays (in-place)
 *
 * Formula: values[i] = 2^values[i]
 *
 * @param values_ Input/output array
 * @param count_ Number of elements
 */
inline void exp2Array(
    float* MMATH_RESTRICT values_,
    int32_t count_
) noexcept {
    detail::exp2Simd(values_, count_);
}

/**
 * @brief Base-10 exponential for arrays (in-place)
 *
 * Formula: values[i] = 10^values[i]
 *
 * @param values_ Input/output array
 * @param count_ Number of elements
 *
 * Use case: dB to linear conversion for audio
 * Example: pow10Array(gain_db_array / 20.0f, count)
 *
 * Performance: ~4-6x faster than scalar loop
 */
inline void pow10Array(
    float* MMATH_RESTRICT values_,
    int32_t count_
) noexcept {
    detail::pow10Simd(values_, count_);
}

/**
 * @brief Natural logarithm for arrays (in-place)
 *
 * Formula: values[i] = ln(values[i])
 *
 * @param values_ Input/output array
 * @param count_ Number of elements
 *
 * Use case: Linear to dB conversion
 * Example: logArray(linear_gains, count); multiply by 20*log10(e)
 *
 * Performance: ~3-5x faster than scalar loop
 */
inline void logArray(
    float* MMATH_RESTRICT values_,
    int32_t count_
) noexcept {
    detail::logSimd(values_, count_);
}

/**
 * @brief Base-2 logarithm for arrays (in-place)
 *
 * Formula: values[i] = log2(values[i])
 *
 * @param values_ Input/output array
 * @param count_ Number of elements
 */
inline void log2Array(
    float* MMATH_RESTRICT values_,
    int32_t count_
) noexcept {
    detail::log2Simd(values_, count_);
}

/**
 * @brief Base-10 logarithm for arrays (in-place)
 *
 * Formula: values[i] = log10(values[i])
 *
 * @param values_ Input/output array
 * @param count_ Number of elements
 *
 * Use case: Linear to dB conversion
 * Example: log10Array(linear_gains, count); multiply by 20
 */
inline void log10Array(
    float* MMATH_RESTRICT values_,
    int32_t count_
) noexcept {
    detail::log10Simd(values_, count_);
}

/**
 * @brief Power function for arrays
 *
 * Formula: result[i] = base[i]^exponent[i]
 *
 * @param base_ Base values array
 * @param exponent_ Exponent values array
 * @param result_ Output array
 * @param count_ Number of elements
 *
 * Use case: Distance attenuation array
 * Example: powArray(distances, rolloff_factors, attenuations, count)
 *
 * Performance: ~5-8x faster than scalar loop
 */
inline void powArray(
    const float* MMATH_RESTRICT base_,
    const float* MMATH_RESTRICT exponent_,
    float* MMATH_RESTRICT result_,
    int32_t count_
) noexcept {
    detail::powSimd(base_, exponent_, result_, count_);
}

} // namespace MMath
}
