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

export module fast_math.mat3_simd;

export import fast_math.types;

export {
/**
 * @file mat3_simd.h
 * @brief Hand-written SIMD Mat3 batch processing (AOS deinterleave optimization)
 *
 * Optimization Strategy:
 * - Deinterleave AOS Mat3 to temporary SoA using AVX2 shuffle/unpack
 * - Process 8 Mat3s at once with AVX2 + FMA
 * - Re-interleave results back to AOS
 * - Affine transform optimization: skip third row computation
 *
 * Key Challenge: 9 floats (36 bytes) doesn't align with AVX2 (32 bytes)
 * Solution: Load in chunks and use complex shuffle patterns
 *
 * References:
 * - Vec2 deinterleave success (1.17x faster dot product)
 * - perf_math SoA batch processing approach
 * - Intel intrinsics guide for optimal shuffle patterns
 */



#if defined(__AVX2__) && defined(__FMA__)
#endif

namespace MMath {
namespace detail {

MMATH_FORCE_INLINE Mat3 mat3FromTrsScalar(Vec2 translation_, float rotation_radians_, Vec2 scale_) noexcept {
    float c = std::cos(rotation_radians_);
    float s = std::sin(rotation_radians_);
    return Mat3{
        scale_.x * c, -scale_.y * s, translation_.x,
        scale_.x * s,  scale_.y * c, translation_.y,
        0.0f, 0.0f, 1.0f
    };
}

MMATH_FORCE_INLINE Mat3 mat3MultiplyAffineScalar(Mat3 a_, Mat3 b_) noexcept {
    return Mat3{
        a_.m00 * b_.m00 + a_.m01 * b_.m10,
        a_.m00 * b_.m01 + a_.m01 * b_.m11,
        a_.m00 * b_.m02 + a_.m01 * b_.m12 + a_.m02,

        a_.m10 * b_.m00 + a_.m11 * b_.m10,
        a_.m10 * b_.m01 + a_.m11 * b_.m11,
        a_.m10 * b_.m02 + a_.m11 * b_.m12 + a_.m12,

        0.0f, 0.0f, 1.0f
    };
}

MMATH_FORCE_INLINE Vec2 mat3TransformPointScalar(Mat3 m_, Vec2 point_) noexcept {
    return Vec2{
        m_.m00 * point_.x + m_.m01 * point_.y + m_.m02,
        m_.m10 * point_.x + m_.m11 * point_.y + m_.m12
    };
}

#if defined(__AVX2__) && defined(__FMA__)

// ============================================================================
// AVX2 Deinterleave/Interleave Helpers for Mat3
// ============================================================================

/**
 * @brief Deinterleave 8 Mat3s from AOS to SoA (affine optimization)
 *
 * Uses direct memory access - compiler will optimize this well
 * Simpler and often faster than complex shuffle patterns for non-power-of-2 strides
 */
MMATH_FORCE_INLINE void deinterleave8Mat3Affine(
    const Mat3* MMATH_RESTRICT input_,
    __m256& m00_, __m256& m01_, __m256& m02_,
    __m256& m10_, __m256& m11_, __m256& m12_
) noexcept {
    // Direct extraction - let compiler vectorize this
    alignas(32) float arr_m00[8], arr_m01[8], arr_m02[8];
    alignas(32) float arr_m10[8], arr_m11[8], arr_m12[8];

    // Unroll manually for better performance
    arr_m00[0] = input_[0].m00; arr_m01[0] = input_[0].m01; arr_m02[0] = input_[0].m02;
    arr_m10[0] = input_[0].m10; arr_m11[0] = input_[0].m11; arr_m12[0] = input_[0].m12;

    arr_m00[1] = input_[1].m00; arr_m01[1] = input_[1].m01; arr_m02[1] = input_[1].m02;
    arr_m10[1] = input_[1].m10; arr_m11[1] = input_[1].m11; arr_m12[1] = input_[1].m12;

    arr_m00[2] = input_[2].m00; arr_m01[2] = input_[2].m01; arr_m02[2] = input_[2].m02;
    arr_m10[2] = input_[2].m10; arr_m11[2] = input_[2].m11; arr_m12[2] = input_[2].m12;

    arr_m00[3] = input_[3].m00; arr_m01[3] = input_[3].m01; arr_m02[3] = input_[3].m02;
    arr_m10[3] = input_[3].m10; arr_m11[3] = input_[3].m11; arr_m12[3] = input_[3].m12;

    arr_m00[4] = input_[4].m00; arr_m01[4] = input_[4].m01; arr_m02[4] = input_[4].m02;
    arr_m10[4] = input_[4].m10; arr_m11[4] = input_[4].m11; arr_m12[4] = input_[4].m12;

    arr_m00[5] = input_[5].m00; arr_m01[5] = input_[5].m01; arr_m02[5] = input_[5].m02;
    arr_m10[5] = input_[5].m10; arr_m11[5] = input_[5].m11; arr_m12[5] = input_[5].m12;

    arr_m00[6] = input_[6].m00; arr_m01[6] = input_[6].m01; arr_m02[6] = input_[6].m02;
    arr_m10[6] = input_[6].m10; arr_m11[6] = input_[6].m11; arr_m12[6] = input_[6].m12;

    arr_m00[7] = input_[7].m00; arr_m01[7] = input_[7].m01; arr_m02[7] = input_[7].m02;
    arr_m10[7] = input_[7].m10; arr_m11[7] = input_[7].m11; arr_m12[7] = input_[7].m12;

    m00_ = _mm256_load_ps(arr_m00);
    m01_ = _mm256_load_ps(arr_m01);
    m02_ = _mm256_load_ps(arr_m02);
    m10_ = _mm256_load_ps(arr_m10);
    m11_ = _mm256_load_ps(arr_m11);
    m12_ = _mm256_load_ps(arr_m12);
}

/**
 * @brief Interleave 6 __m256 vectors back to 8 Mat3s (affine) using sequential stores
 *
 * Note: AVX2 doesn't have scatter, so we extract and store sequentially
 * This is still faster than the loop version due to fewer operations
 */
MMATH_FORCE_INLINE void interleave8Mat3Affine(
    __m256 m00_, __m256 m01_, __m256 m02_,
    __m256 m10_, __m256 m11_, __m256 m12_,
    Mat3* MMATH_RESTRICT output_
) noexcept {
    alignas(32) float temp_m00[8], temp_m01[8], temp_m02[8];
    alignas(32) float temp_m10[8], temp_m11[8], temp_m12[8];

    _mm256_store_ps(temp_m00, m00_);
    _mm256_store_ps(temp_m01, m01_);
    _mm256_store_ps(temp_m02, m02_);
    _mm256_store_ps(temp_m10, m10_);
    _mm256_store_ps(temp_m11, m11_);
    _mm256_store_ps(temp_m12, m12_);

    // Unroll the loop for better performance
    output_[0].m00 = temp_m00[0]; output_[0].m01 = temp_m01[0]; output_[0].m02 = temp_m02[0];
    output_[0].m10 = temp_m10[0]; output_[0].m11 = temp_m11[0]; output_[0].m12 = temp_m12[0];
    output_[0].m20 = 0.0f; output_[0].m21 = 0.0f; output_[0].m22 = 1.0f;

    output_[1].m00 = temp_m00[1]; output_[1].m01 = temp_m01[1]; output_[1].m02 = temp_m02[1];
    output_[1].m10 = temp_m10[1]; output_[1].m11 = temp_m11[1]; output_[1].m12 = temp_m12[1];
    output_[1].m20 = 0.0f; output_[1].m21 = 0.0f; output_[1].m22 = 1.0f;

    output_[2].m00 = temp_m00[2]; output_[2].m01 = temp_m01[2]; output_[2].m02 = temp_m02[2];
    output_[2].m10 = temp_m10[2]; output_[2].m11 = temp_m11[2]; output_[2].m12 = temp_m12[2];
    output_[2].m20 = 0.0f; output_[2].m21 = 0.0f; output_[2].m22 = 1.0f;

    output_[3].m00 = temp_m00[3]; output_[3].m01 = temp_m01[3]; output_[3].m02 = temp_m02[3];
    output_[3].m10 = temp_m10[3]; output_[3].m11 = temp_m11[3]; output_[3].m12 = temp_m12[3];
    output_[3].m20 = 0.0f; output_[3].m21 = 0.0f; output_[3].m22 = 1.0f;

    output_[4].m00 = temp_m00[4]; output_[4].m01 = temp_m01[4]; output_[4].m02 = temp_m02[4];
    output_[4].m10 = temp_m10[4]; output_[4].m11 = temp_m11[4]; output_[4].m12 = temp_m12[4];
    output_[4].m20 = 0.0f; output_[4].m21 = 0.0f; output_[4].m22 = 1.0f;

    output_[5].m00 = temp_m00[5]; output_[5].m01 = temp_m01[5]; output_[5].m02 = temp_m02[5];
    output_[5].m10 = temp_m10[5]; output_[5].m11 = temp_m11[5]; output_[5].m12 = temp_m12[5];
    output_[5].m20 = 0.0f; output_[5].m21 = 0.0f; output_[5].m22 = 1.0f;

    output_[6].m00 = temp_m00[6]; output_[6].m01 = temp_m01[6]; output_[6].m02 = temp_m02[6];
    output_[6].m10 = temp_m10[6]; output_[6].m11 = temp_m11[6]; output_[6].m12 = temp_m12[6];
    output_[6].m20 = 0.0f; output_[6].m21 = 0.0f; output_[6].m22 = 1.0f;

    output_[7].m00 = temp_m00[7]; output_[7].m01 = temp_m01[7]; output_[7].m02 = temp_m02[7];
    output_[7].m10 = temp_m10[7]; output_[7].m11 = temp_m11[7]; output_[7].m12 = temp_m12[7];
    output_[7].m20 = 0.0f; output_[7].m21 = 0.0f; output_[7].m22 = 1.0f;
}

// ============================================================================
// AVX2 Mat3 Batch Operations
// ============================================================================

/**
 * @brief Batch TRS construction with AVX2 + inline SIMD sincos
 */
inline void mat3FromTrsSimd(
    const Vec2* MMATH_RESTRICT translations_,
    const float* MMATH_RESTRICT rotations_,
    const Vec2* MMATH_RESTRICT scales_,
    Mat3* MMATH_RESTRICT result_,
    int32_t count_
) noexcept {
    int32_t i = 0;
    const int32_t simd_count = (count_ / 8) * 8;

    for (; i < simd_count; i += 8) {
        // Load rotations
        __m256 angles = _mm256_loadu_ps(&rotations_[i]);

        // ================================================================
        // Inline SIMD sincos (DirectXMath MiniMax polynomials)
        // ================================================================
        const __m256 two_pi = _mm256_set1_ps(6.283185307179586f);
        const __m256 inv_two_pi = _mm256_set1_ps(0.159154943091895f);
        const __m256 pi = _mm256_set1_ps(3.141592653589793f);
        const __m256 half_pi = _mm256_set1_ps(1.570796326794897f);
        const __m256 one = _mm256_set1_ps(1.0f);
        const __m256 neg_one = _mm256_set1_ps(-1.0f);
        const __m256 sign_mask = _mm256_set1_ps(-0.0f);

        // Range reduction to [-π, π]
        __m256 quotient = _mm256_mul_ps(angles, inv_two_pi);
        quotient = _mm256_round_ps(quotient, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256 x = _mm256_fnmadd_ps(quotient, two_pi, angles);

        // Map to [-π/2, π/2]
        __m256 x_sign = _mm256_and_ps(x, sign_mask);
        __m256 c = _mm256_or_ps(pi, x_sign);
        __m256 abs_x = _mm256_andnot_ps(sign_mask, x);
        __m256 reflected = _mm256_sub_ps(c, x);
        __m256 comp = _mm256_cmp_ps(abs_x, half_pi, _CMP_LE_OQ);
        x = _mm256_blendv_ps(reflected, x, comp);
        __m256 cos_sign = _mm256_blendv_ps(neg_one, one, comp);

        __m256 x2 = _mm256_mul_ps(x, x);

        // Sine polynomial
        const __m256 sin_c0 = _mm256_set1_ps(-0.16666667f);
        const __m256 sin_c1 = _mm256_set1_ps(+0.0083333310f);
        const __m256 sin_c2 = _mm256_set1_ps(-0.00019840874f);
        const __m256 sin_c3 = _mm256_set1_ps(+2.7525562e-06f);
        const __m256 sin_c4 = _mm256_set1_ps(-2.3889859e-08f);

        __m256 sin_result = _mm256_fmadd_ps(sin_c4, x2, sin_c3);
        sin_result = _mm256_fmadd_ps(sin_result, x2, sin_c2);
        sin_result = _mm256_fmadd_ps(sin_result, x2, sin_c1);
        sin_result = _mm256_fmadd_ps(sin_result, x2, sin_c0);
        sin_result = _mm256_fmadd_ps(sin_result, x2, one);
        sin_result = _mm256_mul_ps(sin_result, x);

        // Cosine polynomial
        const __m256 cos_c0 = _mm256_set1_ps(-0.5f);
        const __m256 cos_c1 = _mm256_set1_ps(+0.041666638f);
        const __m256 cos_c2 = _mm256_set1_ps(-0.0013888378f);
        const __m256 cos_c3 = _mm256_set1_ps(+2.4760495e-05f);
        const __m256 cos_c4 = _mm256_set1_ps(-2.6051615e-07f);

        __m256 cos_result = _mm256_fmadd_ps(cos_c4, x2, cos_c3);
        cos_result = _mm256_fmadd_ps(cos_result, x2, cos_c2);
        cos_result = _mm256_fmadd_ps(cos_result, x2, cos_c1);
        cos_result = _mm256_fmadd_ps(cos_result, x2, cos_c0);
        cos_result = _mm256_fmadd_ps(cos_result, x2, one);
        cos_result = _mm256_mul_ps(cos_result, cos_sign);

        // Load scales (deinterleave Vec2 array)
        __m256 sx, sy;
        {
            alignas(32) float temp_sx[8], temp_sy[8];
            for (int j = 0; j < 8; ++j) {
                temp_sx[j] = scales_[i + j].x;
                temp_sy[j] = scales_[i + j].y;
            }
            sx = _mm256_load_ps(temp_sx);
            sy = _mm256_load_ps(temp_sy);
        }

        // Load translations
        __m256 tx, ty;
        {
            alignas(32) float temp_tx[8], temp_ty[8];
            for (int j = 0; j < 8; ++j) {
                temp_tx[j] = translations_[i + j].x;
                temp_ty[j] = translations_[i + j].y;
            }
            tx = _mm256_load_ps(temp_tx);
            ty = _mm256_load_ps(temp_ty);
        }

        // Compute matrix elements
        // m00 = sx * cos, m01 = -sy * sin, m02 = tx
        // m10 = sx * sin, m11 = sy * cos, m12 = ty
        __m256 m00 = _mm256_mul_ps(sx, cos_result);
        __m256 m01 = _mm256_mul_ps(_mm256_mul_ps(sy, sin_result), neg_one);
        __m256 m10 = _mm256_mul_ps(sx, sin_result);
        __m256 m11 = _mm256_mul_ps(sy, cos_result);

        // Interleave back to AOS
        interleave8Mat3Affine(m00, m01, tx, m10, m11, ty, result_ + i);
    }

    // Handle remainder
    for (; i < count_; ++i) {
        result_[i] = mat3FromTrsScalar(translations_[i], rotations_[i], scales_[i]);
    }
}

/**
 * @brief Batch affine matrix multiplication with AVX2
 */
inline void mat3MultiplyAffineSimd(
    const Mat3* MMATH_RESTRICT a_,
    const Mat3* MMATH_RESTRICT b_,
    Mat3* MMATH_RESTRICT result_,
    int32_t count_
) noexcept {
    int32_t i = 0;
    const int32_t simd_count = (count_ / 8) * 8;

    for (; i < simd_count; i += 8) {
        __m256 a00, a01, a02, a10, a11, a12;
        __m256 b00, b01, b02, b10, b11, b12;

        deinterleave8Mat3Affine(a_ + i, a00, a01, a02, a10, a11, a12);
        deinterleave8Mat3Affine(b_ + i, b00, b01, b02, b10, b11, b12);

        // Affine multiplication (6 operations)
        // r00 = a00*b00 + a01*b10
        // r01 = a00*b01 + a01*b11
        // r02 = a00*b02 + a01*b12 + a02
        // r10 = a10*b00 + a11*b10
        // r11 = a10*b01 + a11*b11
        // r12 = a10*b02 + a11*b12 + a12

        __m256 r00 = _mm256_mul_ps(a00, b00);
        r00 = _mm256_fmadd_ps(a01, b10, r00);

        __m256 r01 = _mm256_mul_ps(a00, b01);
        r01 = _mm256_fmadd_ps(a01, b11, r01);

        __m256 r02 = _mm256_mul_ps(a00, b02);
        r02 = _mm256_fmadd_ps(a01, b12, r02);
        r02 = _mm256_add_ps(r02, a02);

        __m256 r10 = _mm256_mul_ps(a10, b00);
        r10 = _mm256_fmadd_ps(a11, b10, r10);

        __m256 r11 = _mm256_mul_ps(a10, b01);
        r11 = _mm256_fmadd_ps(a11, b11, r11);

        __m256 r12 = _mm256_mul_ps(a10, b02);
        r12 = _mm256_fmadd_ps(a11, b12, r12);
        r12 = _mm256_add_ps(r12, a12);

        interleave8Mat3Affine(r00, r01, r02, r10, r11, r12, result_ + i);
    }

    // Handle remainder
    for (; i < count_; ++i) {
        result_[i] = mat3MultiplyAffineScalar(a_[i], b_[i]);
    }
}

/**
 * @brief Batch point transformation (each matrix transforms its corresponding point)
 */
inline void mat3TransformPointsSimd(
    const Mat3* MMATH_RESTRICT matrices_,
    const Vec2* MMATH_RESTRICT points_,
    Vec2* MMATH_RESTRICT result_,
    int32_t count_
) noexcept {
    int32_t i = 0;
    const int32_t simd_count = (count_ / 8) * 8;

    for (; i < simd_count; i += 8) {
        __m256 m00, m01, m02, m10, m11, m12;
        deinterleave8Mat3Affine(matrices_ + i, m00, m01, m02, m10, m11, m12);

        // Load points
        __m256 px, py;
        {
            alignas(32) float temp_px[8], temp_py[8];
            for (int j = 0; j < 8; ++j) {
                temp_px[j] = points_[i + j].x;
                temp_py[j] = points_[i + j].y;
            }
            px = _mm256_load_ps(temp_px);
            py = _mm256_load_ps(temp_py);
        }

        // rx = m00*px + m01*py + m02
        // ry = m10*px + m11*py + m12
        __m256 rx = _mm256_mul_ps(m00, px);
        rx = _mm256_fmadd_ps(m01, py, rx);
        rx = _mm256_add_ps(rx, m02);

        __m256 ry = _mm256_mul_ps(m10, px);
        ry = _mm256_fmadd_ps(m11, py, ry);
        ry = _mm256_add_ps(ry, m12);

        // Store results
        alignas(32) float temp_rx[8], temp_ry[8];
        _mm256_store_ps(temp_rx, rx);
        _mm256_store_ps(temp_ry, ry);

        for (int j = 0; j < 8; ++j) {
            result_[i + j].x = temp_rx[j];
            result_[i + j].y = temp_ry[j];
        }
    }

    // Handle remainder
    for (; i < count_; ++i) {
        result_[i] = mat3TransformPointScalar(matrices_[i], points_[i]);
    }
}

/**
 * @brief Transform multiple points by a SINGLE matrix
 * More efficient when broadcasting one matrix to many points
 */
inline void mat3TransformPointsSingleSimd(
    Mat3 matrix_,
    const Vec2* MMATH_RESTRICT points_,
    Vec2* MMATH_RESTRICT result_,
    int32_t count_
) noexcept {
    int32_t i = 0;
    const int32_t simd_count = (count_ / 8) * 8;

    // Broadcast matrix elements
    __m256 m00 = _mm256_set1_ps(matrix_.m00);
    __m256 m01 = _mm256_set1_ps(matrix_.m01);
    __m256 m02 = _mm256_set1_ps(matrix_.m02);
    __m256 m10 = _mm256_set1_ps(matrix_.m10);
    __m256 m11 = _mm256_set1_ps(matrix_.m11);
    __m256 m12 = _mm256_set1_ps(matrix_.m12);

    for (; i < simd_count; i += 8) {
        // Load 8 points
        __m256 px, py;
        {
            alignas(32) float temp_px[8], temp_py[8];
            for (int j = 0; j < 8; ++j) {
                temp_px[j] = points_[i + j].x;
                temp_py[j] = points_[i + j].y;
            }
            px = _mm256_load_ps(temp_px);
            py = _mm256_load_ps(temp_py);
        }

        // Transform
        __m256 rx = _mm256_mul_ps(m00, px);
        rx = _mm256_fmadd_ps(m01, py, rx);
        rx = _mm256_add_ps(rx, m02);

        __m256 ry = _mm256_mul_ps(m10, px);
        ry = _mm256_fmadd_ps(m11, py, ry);
        ry = _mm256_add_ps(ry, m12);

        // Store results
        alignas(32) float temp_rx[8], temp_ry[8];
        _mm256_store_ps(temp_rx, rx);
        _mm256_store_ps(temp_ry, ry);

        for (int j = 0; j < 8; ++j) {
            result_[i + j].x = temp_rx[j];
            result_[i + j].y = temp_ry[j];
        }
    }

    // Handle remainder
    for (; i < count_; ++i) {
        result_[i] = mat3TransformPointScalar(matrix_, points_[i]);
    }
}

#endif // AVX2

} // namespace detail
} // namespace MMath
}
