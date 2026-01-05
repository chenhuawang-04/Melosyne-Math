/**
 * @file mat3.h
 * @brief High-performance 3x3 matrix library (Strict POD + Hand-written SIMD)
 *
 * Design Philosophy:
 * - STRICT POD structures (NO member functions)
 * - Hand-written AVX2/SSE4.1 SIMD intrinsics
 * - Optimized for 2D affine transforms
 * - Row-major layout for intuitive operations
 * - Compatible with DirectXMath/GLM performance
 *
 * Matrix Layout (Row-Major):
 * [m00  m01  m02]   [sx*cos  -sy*sin  tx]
 * [m10  m11  m12] = [sx*sin   sy*cos  ty]
 * [m20  m21  m22]   [0        0       1 ] (affine)
 *
 * References:
 * - DirectXMath matrix operations
 * - perf_math SoA approach for batch performance
 * - GLM for API design
 */

#pragma once

#include "types.h"
#include "vec2.h"
#include <cmath>
#include <cstring>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#elif defined(__SSE4_1__)
#include <smmintrin.h>
#endif

namespace MMath {

// ============================================================================
// Scalar Mat3 Operations (POD + Free Functions)
// ============================================================================

/**
 * @brief Identity matrix
 */
MMATH_FORCE_INLINE Mat3 mat3Identity() noexcept {
    return Mat3{
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
    };
}

/**
 * @brief Zero matrix
 */
MMATH_FORCE_INLINE Mat3 mat3Zero() noexcept {
    return Mat3{
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f
    };
}

/**
 * @brief Create translation matrix
 */
MMATH_FORCE_INLINE Mat3 mat3FromTranslation(Vec2 translation_) noexcept {
    return Mat3{
        1.0f, 0.0f, translation_.x,
        0.0f, 1.0f, translation_.y,
        0.0f, 0.0f, 1.0f
    };
}

/**
 * @brief Create rotation matrix
 */
MMATH_FORCE_INLINE Mat3 mat3FromRotation(float angle_radians_) noexcept {
    float c = std::cos(angle_radians_);
    float s = std::sin(angle_radians_);
    return Mat3{
        c,    -s,   0.0f,
        s,     c,   0.0f,
        0.0f, 0.0f, 1.0f
    };
}

/**
 * @brief Create scale matrix
 */
MMATH_FORCE_INLINE Mat3 mat3FromScale(Vec2 scale_) noexcept {
    return Mat3{
        scale_.x, 0.0f,     0.0f,
        0.0f,     scale_.y, 0.0f,
        0.0f,     0.0f,     1.0f
    };
}

/**
 * @brief Create TRS (Translation-Rotation-Scale) matrix
 */
MMATH_FORCE_INLINE Mat3 mat3FromTrs(Vec2 translation_, float rotation_radians_, Vec2 scale_) noexcept {
    float c = std::cos(rotation_radians_);
    float s = std::sin(rotation_radians_);
    return Mat3{
        scale_.x * c, -scale_.y * s, translation_.x,
        scale_.x * s,  scale_.y * c, translation_.y,
        0.0f,         0.0f,        1.0f
    };
}

/**
 * @brief Full 3x3 matrix multiplication
 */
MMATH_FORCE_INLINE Mat3 mat3Multiply(Mat3 a_, Mat3 b_) noexcept {
    return Mat3{
        a_.m00 * b_.m00 + a_.m01 * b_.m10 + a_.m02 * b_.m20,
        a_.m00 * b_.m01 + a_.m01 * b_.m11 + a_.m02 * b_.m21,
        a_.m00 * b_.m02 + a_.m01 * b_.m12 + a_.m02 * b_.m22,

        a_.m10 * b_.m00 + a_.m11 * b_.m10 + a_.m12 * b_.m20,
        a_.m10 * b_.m01 + a_.m11 * b_.m11 + a_.m12 * b_.m21,
        a_.m10 * b_.m02 + a_.m11 * b_.m12 + a_.m12 * b_.m22,

        a_.m20 * b_.m00 + a_.m21 * b_.m10 + a_.m22 * b_.m20,
        a_.m20 * b_.m01 + a_.m21 * b_.m11 + a_.m22 * b_.m21,
        a_.m20 * b_.m02 + a_.m21 * b_.m12 + a_.m22 * b_.m22
    };
}

/**
 * @brief Affine matrix multiplication (optimized for 2D transforms)
 * Assumes last row is [0, 0, 1]
 */
MMATH_FORCE_INLINE Mat3 mat3MultiplyAffine(Mat3 a_, Mat3 b_) noexcept {
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

/**
 * @brief Transform point (with translation)
 */
MMATH_FORCE_INLINE Vec2 mat3TransformPoint(Mat3 m_, Vec2 point_) noexcept {
    return Vec2{
        m_.m00 * point_.x + m_.m01 * point_.y + m_.m02,
        m_.m10 * point_.x + m_.m11 * point_.y + m_.m12
    };
}

/**
 * @brief Transform vector (no translation)
 */
MMATH_FORCE_INLINE Vec2 mat3TransformVector(Mat3 m_, Vec2 vector_) noexcept {
    return Vec2{
        m_.m00 * vector_.x + m_.m01 * vector_.y,
        m_.m10 * vector_.x + m_.m11 * vector_.y
    };
}

/**
 * @brief Calculate determinant
 */
MMATH_FORCE_INLINE float mat3Determinant(Mat3 m_) noexcept {
    return m_.m00 * (m_.m11 * m_.m22 - m_.m12 * m_.m21)
         - m_.m01 * (m_.m10 * m_.m22 - m_.m12 * m_.m20)
         + m_.m02 * (m_.m10 * m_.m21 - m_.m11 * m_.m20);
}

/**
 * @brief Full matrix inverse
 */
MMATH_FORCE_INLINE Mat3 mat3Inverse(Mat3 m_) noexcept {
    float c00 = m_.m11 * m_.m22 - m_.m12 * m_.m21;
    float c01 = m_.m12 * m_.m20 - m_.m10 * m_.m22;
    float c02 = m_.m10 * m_.m21 - m_.m11 * m_.m20;

    float c10 = m_.m02 * m_.m21 - m_.m01 * m_.m22;
    float c11 = m_.m00 * m_.m22 - m_.m02 * m_.m20;
    float c12 = m_.m01 * m_.m20 - m_.m00 * m_.m21;

    float c20 = m_.m01 * m_.m12 - m_.m02 * m_.m11;
    float c21 = m_.m02 * m_.m10 - m_.m00 * m_.m12;
    float c22 = m_.m00 * m_.m11 - m_.m01 * m_.m10;

    float det = m_.m00 * c00 + m_.m01 * c01 + m_.m02 * c02;

    if (det * det < 1e-12f) {
        return mat3Identity();
    }

    float inv_det = 1.0f / det;

    return Mat3{
        c00 * inv_det, c10 * inv_det, c20 * inv_det,
        c01 * inv_det, c11 * inv_det, c21 * inv_det,
        c02 * inv_det, c12 * inv_det, c22 * inv_det
    };
}

/**
 * @brief Affine matrix inverse (optimized)
 */
MMATH_FORCE_INLINE Mat3 mat3InverseAffine(Mat3 m_) noexcept {
    float det = m_.m00 * m_.m11 - m_.m01 * m_.m10;

    if (det * det < 1e-12f) {
        return mat3Identity();
    }

    float inv_det = 1.0f / det;

    float r00 =  m_.m11 * inv_det;
    float r01 = -m_.m01 * inv_det;
    float r10 = -m_.m10 * inv_det;
    float r11 =  m_.m00 * inv_det;

    float tx = -(r00 * m_.m02 + r01 * m_.m12);
    float ty = -(r10 * m_.m02 + r11 * m_.m12);

    return Mat3{
        r00, r01, tx,
        r10, r11, ty,
        0.0f, 0.0f, 1.0f
    };
}

/**
 * @brief Matrix transpose
 */
MMATH_FORCE_INLINE Mat3 mat3Transpose(Mat3 m_) noexcept {
    return Mat3{
        m_.m00, m_.m10, m_.m20,
        m_.m01, m_.m11, m_.m21,
        m_.m02, m_.m12, m_.m22
    };
}

/**
 * @brief Equality comparison
 */
MMATH_FORCE_INLINE bool mat3Equal(Mat3 a_, Mat3 b_) noexcept {
    return a_.m00 == b_.m00 && a_.m01 == b_.m01 && a_.m02 == b_.m02 &&
           a_.m10 == b_.m10 && a_.m11 == b_.m11 && a_.m12 == b_.m12 &&
           a_.m20 == b_.m20 && a_.m21 == b_.m21 && a_.m22 == b_.m22;
}

/**
 * @brief Near-equality comparison with epsilon
 */
MMATH_FORCE_INLINE bool mat3NearEqual(Mat3 a_, Mat3 b_, float epsilon_) noexcept {
    auto is_near_ = [epsilon_](float x, float y) {
        float d = x - y;
        return (d < 0 ? -d : d) <= epsilon_;
    };
    return is_near_(a_.m00, b_.m00) && is_near_(a_.m01, b_.m01) && is_near_(a_.m02, b_.m02) &&
           is_near_(a_.m10, b_.m10) && is_near_(a_.m11, b_.m11) && is_near_(a_.m12, b_.m12) &&
           is_near_(a_.m20, b_.m20) && is_near_(a_.m21, b_.m21) && is_near_(a_.m22, b_.m22);
}

} // namespace MMath (close before including mat3_simd.h)

// Include SIMD implementations after closing namespace
#include "mat3_simd.h"

namespace MMath {

// ============================================================================
// Batch Processing (Optimized SIMD dispatch)
// ============================================================================

/**
 * @brief Batch TRS construction (SIMD optimized)
 *
 * @param translations_ Translation vectors (array of Vec2)
 * @param rotations_ Rotation angles in radians (array of float)
 * @param scales_ Scale vectors (array of Vec2)
 * @param result_ Output array
 * @param count_ Number of matrices
 */
inline void mat3FromTrsArray(const Vec2* MMATH_RESTRICT translations_,
                             const float* MMATH_RESTRICT rotations_,
                             const Vec2* MMATH_RESTRICT scales_,
                             Mat3* MMATH_RESTRICT result_,
                             int32_t count_) noexcept {
#if defined(__AVX2__) && defined(__FMA__)
    detail::mat3FromTrsSimd(translations_, rotations_, scales_, result_, count_);
#else
    for (int32_t i = 0; i < count_; ++i) {
        result_[i] = mat3FromTrs(translations_[i], rotations_[i], scales_[i]);
    }
#endif
}

/**
 * @brief Batch affine matrix multiplication (SIMD optimized)
 */
inline void mat3MultiplyAffineArray(const Mat3* MMATH_RESTRICT a_,
                                    const Mat3* MMATH_RESTRICT b_,
                                    Mat3* MMATH_RESTRICT result_,
                                    int32_t count_) noexcept {
#if defined(__AVX2__) && defined(__FMA__)
    detail::mat3MultiplyAffineSimd(a_, b_, result_, count_);
#else
    for (int32_t i = 0; i < count_; ++i) {
        result_[i] = mat3MultiplyAffine(a_[i], b_[i]);
    }
#endif
}

/**
 * @brief Batch point transformation (SIMD optimized)
 */
inline void mat3TransformPointsArray(const Mat3* MMATH_RESTRICT matrices_,
                                     const Vec2* MMATH_RESTRICT points_,
                                     Vec2* MMATH_RESTRICT result_,
                                     int32_t count_) noexcept {
#if defined(__AVX2__) && defined(__FMA__)
    detail::mat3TransformPointsSimd(matrices_, points_, result_, count_);
#else
    for (int32_t i = 0; i < count_; ++i) {
        result_[i] = mat3TransformPoint(matrices_[i], points_[i]);
    }
#endif
}

/**
 * @brief Transform multiple points by a single matrix (SIMD optimized)
 */
inline void mat3TransformPointsSingle(Mat3 matrix_,
                                      const Vec2* MMATH_RESTRICT points_,
                                      Vec2* MMATH_RESTRICT result_,
                                      int32_t count_) noexcept {
#if defined(__AVX2__) && defined(__FMA__)
    detail::mat3TransformPointsSingleSimd(matrix_, points_, result_, count_);
#else
    for (int32_t i = 0; i < count_; ++i) {
        result_[i] = mat3TransformPoint(matrix_, points_[i]);
    }
#endif
}

} // namespace MMath
