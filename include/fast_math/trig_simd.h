/**
 * @file trig_simd.h
 * @brief Hand-written SIMD trigonometric functions (DirectXMath style)
 *
 * Optimizations:
 * - Hand-written SSE4.1/AVX2 intrinsics (NO auto-vectorization)
 * - FMA instructions for maximum ILP
 * - Branchless design using SIMD blends
 * - Constant shuffle optimization
 * - DirectXMath 11/10-degree minimax polynomials
 */

#pragma once

#include "types.h"
#include <cmath>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#elif defined(__SSE4_1__)
#include <smmintrin.h>
#endif

namespace MMath {

// ============================================================================
// Constants (DirectXMath)
// ============================================================================

namespace detail {

// Sine 11-degree minimax coefficients
alignas(32) constexpr float SIN_COEFF[5] = {
    -0.16666667f,      // c0: x³
    +0.0083333310f,    // c1: x⁵
    -0.00019840874f,   // c2: x⁷
    +2.7525562e-06f,   // c3: x⁹
    -2.3889859e-08f    // c4: x¹¹
};

// Cosine 10-degree minimax coefficients
alignas(32) constexpr float COS_COEFF[5] = {
    -0.5f,             // c0: x²
    +0.041666638f,     // c1: x⁴
    -0.0013888378f,    // c2: x⁶
    +2.4760495e-05f,   // c3: x⁸
    -2.6051615e-07f    // c4: x¹⁰
};

constexpr float PI = 3.141592653589793f;
constexpr float TWO_PI = 6.283185307179586f;
constexpr float HALF_PI = 1.570796326794897f;
constexpr float INV_TWO_PI = 0.159154943091895f;

} // namespace detail

// ============================================================================
// AVX2 Implementation (8 floats at once)
// ============================================================================

#if defined(__AVX2__) && defined(__FMA__)

/**
 * @brief Compute sin/cos for 8 angles using AVX2 + FMA
 *
 * Algorithm:
 * 1. Range reduction to [-π, π]
 * 2. Map to [-π/2, π/2] with sign tracking (branchless)
 * 3. Evaluate polynomials using FMA Horner scheme
 *
 * Performance: ~2-3 cycles per angle (8 angles in ~20 cycles)
 */
MMATH_FORCE_INLINE void sincosAvx2(const AnglesAvx2* MMATH_RESTRICT angles_,
                                   SinCosAvx2* MMATH_RESTRICT result_) noexcept {
    // Load angles
    __m256 x = _mm256_load_ps(angles_->angles);

    // Step 1: Range reduction to [-π, π]
    const __m256 two_pi = _mm256_set1_ps(detail::TWO_PI);
    const __m256 inv_two_pi = _mm256_set1_ps(detail::INV_TWO_PI);
    __m256 quotient = _mm256_mul_ps(x, inv_two_pi);
    quotient = _mm256_round_ps(quotient, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    x = _mm256_fnmadd_ps(quotient, two_pi, x);  // x - quotient * 2π

    // Step 2: Map to [-π/2, π/2] (branchless, DirectXMath style)
    const __m256 pi = _mm256_set1_ps(detail::PI);
    const __m256 half_pi = _mm256_set1_ps(detail::HALF_PI);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 neg_one = _mm256_set1_ps(-1.0f);
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);  // 0x80000000

    // Extract sign of x
    __m256 x_sign = _mm256_and_ps(x, sign_mask);
    // c = π (with sign of x)
    __m256 c = _mm256_or_ps(pi, x_sign);
    // abs(x)
    __m256 abs_x = _mm256_andnot_ps(sign_mask, x);
    // reflected = c - x
    __m256 reflected = _mm256_sub_ps(c, x);
    // comp = (abs_x <= π/2)
    __m256 comp = _mm256_cmp_ps(abs_x, half_pi, _CMP_LE_OQ);
    // x_mapped = comp ? x : reflected
    x = _mm256_blendv_ps(reflected, x, comp);
    // cos_sign = comp ? 1.0 : -1.0
    __m256 cos_sign = _mm256_blendv_ps(neg_one, one, comp);

    // Step 3: Polynomial evaluation using FMA
    __m256 x2 = _mm256_mul_ps(x, x);  // x²

    // Load sine coefficients (broadcast)
    const __m256 sin_c4 = _mm256_set1_ps(detail::SIN_COEFF[4]);
    const __m256 sin_c3 = _mm256_set1_ps(detail::SIN_COEFF[3]);
    const __m256 sin_c2 = _mm256_set1_ps(detail::SIN_COEFF[2]);
    const __m256 sin_c1 = _mm256_set1_ps(detail::SIN_COEFF[1]);
    const __m256 sin_c0 = _mm256_set1_ps(detail::SIN_COEFF[0]);

    // Sin polynomial: x * (1 + c0*x² + c1*x⁴ + ... + c4*x¹⁰)
    __m256 sin_poly = sin_c4;
    sin_poly = _mm256_fmadd_ps(sin_poly, x2, sin_c3);
    sin_poly = _mm256_fmadd_ps(sin_poly, x2, sin_c2);
    sin_poly = _mm256_fmadd_ps(sin_poly, x2, sin_c1);
    sin_poly = _mm256_fmadd_ps(sin_poly, x2, sin_c0);
    sin_poly = _mm256_fmadd_ps(sin_poly, x2, one);
    __m256 sin_result = _mm256_mul_ps(sin_poly, x);

    // Load cosine coefficients (broadcast)
    const __m256 cos_c4 = _mm256_set1_ps(detail::COS_COEFF[4]);
    const __m256 cos_c3 = _mm256_set1_ps(detail::COS_COEFF[3]);
    const __m256 cos_c2 = _mm256_set1_ps(detail::COS_COEFF[2]);
    const __m256 cos_c1 = _mm256_set1_ps(detail::COS_COEFF[1]);
    const __m256 cos_c0 = _mm256_set1_ps(detail::COS_COEFF[0]);

    // Cos polynomial: 1 + c0*x² + c1*x⁴ + ... + c4*x¹⁰
    __m256 cos_poly = cos_c4;
    cos_poly = _mm256_fmadd_ps(cos_poly, x2, cos_c3);
    cos_poly = _mm256_fmadd_ps(cos_poly, x2, cos_c2);
    cos_poly = _mm256_fmadd_ps(cos_poly, x2, cos_c1);
    cos_poly = _mm256_fmadd_ps(cos_poly, x2, cos_c0);
    cos_poly = _mm256_fmadd_ps(cos_poly, x2, one);
    __m256 cos_result = _mm256_mul_ps(cos_poly, cos_sign);

    // Store results
    _mm256_store_ps(result_->sins, sin_result);
    _mm256_store_ps(result_->coss, cos_result);
}

#endif // AVX2

// ============================================================================
// SSE4.1 Implementation (4 floats at once)
// ============================================================================

#if defined(__SSE4_1__)

MMATH_FORCE_INLINE void sincosSse(const AnglesSse* MMATH_RESTRICT angles_,
                                  SinCosSse* MMATH_RESTRICT result_) noexcept {
    // Load angles
    __m128 x = _mm_load_ps(angles_->angles);

    // Step 1: Range reduction to [-π, π]
    const __m128 two_pi = _mm_set1_ps(detail::TWO_PI);
    const __m128 inv_two_pi = _mm_set1_ps(detail::INV_TWO_PI);
    __m128 quotient = _mm_mul_ps(x, inv_two_pi);
    quotient = _mm_round_ps(quotient, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    x = _mm_sub_ps(x, _mm_mul_ps(quotient, two_pi));

    // Step 2: Map to [-π/2, π/2] (branchless)
    const __m128 pi = _mm_set1_ps(detail::PI);
    const __m128 half_pi = _mm_set1_ps(detail::HALF_PI);
    const __m128 one = _mm_set1_ps(1.0f);
    const __m128 neg_one = _mm_set1_ps(-1.0f);
    const __m128 sign_mask = _mm_set1_ps(-0.0f);

    __m128 x_sign = _mm_and_ps(x, sign_mask);
    __m128 c = _mm_or_ps(pi, x_sign);
    __m128 abs_x = _mm_andnot_ps(sign_mask, x);
    __m128 reflected = _mm_sub_ps(c, x);
    __m128 comp = _mm_cmple_ps(abs_x, half_pi);

    x = _mm_blendv_ps(reflected, x, comp);
    __m128 cos_sign = _mm_blendv_ps(neg_one, one, comp);

    // Step 3: Polynomial evaluation
    __m128 x2 = _mm_mul_ps(x, x);

    // Sine (Horner with FMA if available)
    const __m128 sin_c4 = _mm_set1_ps(detail::SIN_COEFF[4]);
    const __m128 sin_c3 = _mm_set1_ps(detail::SIN_COEFF[3]);
    const __m128 sin_c2 = _mm_set1_ps(detail::SIN_COEFF[2]);
    const __m128 sin_c1 = _mm_set1_ps(detail::SIN_COEFF[1]);
    const __m128 sin_c0 = _mm_set1_ps(detail::SIN_COEFF[0]);

    __m128 sin_poly = sin_c4;
#if defined(__FMA__)
    sin_poly = _mm_fmadd_ps(sin_poly, x2, sin_c3);
    sin_poly = _mm_fmadd_ps(sin_poly, x2, sin_c2);
    sin_poly = _mm_fmadd_ps(sin_poly, x2, sin_c1);
    sin_poly = _mm_fmadd_ps(sin_poly, x2, sin_c0);
    sin_poly = _mm_fmadd_ps(sin_poly, x2, one);
#else
    sin_poly = _mm_add_ps(_mm_mul_ps(sin_poly, x2), sin_c3);
    sin_poly = _mm_add_ps(_mm_mul_ps(sin_poly, x2), sin_c2);
    sin_poly = _mm_add_ps(_mm_mul_ps(sin_poly, x2), sin_c1);
    sin_poly = _mm_add_ps(_mm_mul_ps(sin_poly, x2), sin_c0);
    sin_poly = _mm_add_ps(_mm_mul_ps(sin_poly, x2), one);
#endif
    __m128 sin_result = _mm_mul_ps(sin_poly, x);

    // Cosine
    const __m128 cos_c4 = _mm_set1_ps(detail::COS_COEFF[4]);
    const __m128 cos_c3 = _mm_set1_ps(detail::COS_COEFF[3]);
    const __m128 cos_c2 = _mm_set1_ps(detail::COS_COEFF[2]);
    const __m128 cos_c1 = _mm_set1_ps(detail::COS_COEFF[1]);
    const __m128 cos_c0 = _mm_set1_ps(detail::COS_COEFF[0]);

    __m128 cos_poly = cos_c4;
#if defined(__FMA__)
    cos_poly = _mm_fmadd_ps(cos_poly, x2, cos_c3);
    cos_poly = _mm_fmadd_ps(cos_poly, x2, cos_c2);
    cos_poly = _mm_fmadd_ps(cos_poly, x2, cos_c1);
    cos_poly = _mm_fmadd_ps(cos_poly, x2, cos_c0);
    cos_poly = _mm_fmadd_ps(cos_poly, x2, one);
#else
    cos_poly = _mm_add_ps(_mm_mul_ps(cos_poly, x2), cos_c3);
    cos_poly = _mm_add_ps(_mm_mul_ps(cos_poly, x2), cos_c2);
    cos_poly = _mm_add_ps(_mm_mul_ps(cos_poly, x2), cos_c1);
    cos_poly = _mm_add_ps(_mm_mul_ps(cos_poly, x2), cos_c0);
    cos_poly = _mm_add_ps(_mm_mul_ps(cos_poly, x2), one);
#endif
    __m128 cos_result = _mm_mul_ps(cos_poly, cos_sign);

    // Store
    _mm_store_ps(result_->sins, sin_result);
    _mm_store_ps(result_->coss, cos_result);
}

#endif // SSE4.1

// ============================================================================
// Scalar Implementation (fallback)
// ============================================================================

/**
 * @brief Scalar sin/cos (same algorithm as SIMD)
 */
MMATH_FORCE_INLINE void sincosScalar(Angle angle_, SinCos* MMATH_RESTRICT result_) noexcept {
    float x = angle_.value;

    // Range reduction
    float quotient = x * detail::INV_TWO_PI;
    quotient = std::round(quotient);
    x = x - quotient * detail::TWO_PI;

    // Map to [-π/2, π/2]
    const uint32_t sign_mask_bits = 0x80000000u;
    uint32_t x_bits;
    std::memcpy(&x_bits, &x, sizeof(float));
    uint32_t x_sign_bits = x_bits & sign_mask_bits;

    uint32_t pi_bits;
    std::memcpy(&pi_bits, &detail::PI, sizeof(float));
    uint32_t c_bits = pi_bits | x_sign_bits;
    float c;
    std::memcpy(&c, &c_bits, sizeof(float));

    uint32_t abs_x_bits = x_bits & ~sign_mask_bits;
    float abs_x;
    std::memcpy(&abs_x, &abs_x_bits, sizeof(float));

    float reflected = c - x;
    float x_mapped = (abs_x <= detail::HALF_PI) ? x : reflected;
    float cos_sign = (abs_x <= detail::HALF_PI) ? 1.0f : -1.0f;

    // Polynomial evaluation
    float x2 = x_mapped * x_mapped;

    float sin_poly = detail::SIN_COEFF[4];
    sin_poly = sin_poly * x2 + detail::SIN_COEFF[3];
    sin_poly = sin_poly * x2 + detail::SIN_COEFF[2];
    sin_poly = sin_poly * x2 + detail::SIN_COEFF[1];
    sin_poly = sin_poly * x2 + detail::SIN_COEFF[0];
    sin_poly = sin_poly * x2 + 1.0f;
    result_->sin = x_mapped * sin_poly;

    float cos_poly = detail::COS_COEFF[4];
    cos_poly = cos_poly * x2 + detail::COS_COEFF[3];
    cos_poly = cos_poly * x2 + detail::COS_COEFF[2];
    cos_poly = cos_poly * x2 + detail::COS_COEFF[1];
    cos_poly = cos_poly * x2 + detail::COS_COEFF[0];
    cos_poly = cos_poly * x2 + 1.0f;
    result_->cos = cos_poly * cos_sign;
}

} // namespace MMath
