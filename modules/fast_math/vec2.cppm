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

export module fast_math.vec2;

export import fast_math.types;
export import fast_math.vec2_simd;

export {
/**
 * @file vec2.h
 * @brief High-performance 2D vector library (Strict POD + Hand-written SIMD)
 *
 * Design Philosophy:
 * - STRICT POD structures (NO member functions)
 * - Hand-written AVX2/SSE4.1 SIMD intrinsics
 * - Optimized for batch processing with deinterleave
 * - Compatible with DirectXMath/GLM performance
 *
 * References:
 * - DirectXMath XMVector2* functions
 * - SIMD Dot Product Optimization: https://github.com/segfaultscribe/SIMD-Dot-product-Optimization
 * - Fast SIMD practices: https://lemire.me/blog/2018/07/05/how-quickly-can-you-compute-the-dot-product-between-two-large-vectors/
 * - AOS Vectorization: https://arxiv.org/abs/1806.05713
 */



#if defined(__AVX2__) && defined(__FMA__)
#elif defined(__SSE4_1__)
#endif

namespace MMath {

// ============================================================================
// Scalar Vec2 Operations (POD + Free Functions)
// ============================================================================

/**
 * @brief Add two vectors
 */
MMATH_FORCE_INLINE Vec2 vec2Add(Vec2 a_, Vec2 b_) noexcept {
    return Vec2{ a_.x + b_.x, a_.y + b_.y };
}

/**
 * @brief Subtract two vectors
 */
MMATH_FORCE_INLINE Vec2 vec2Sub(Vec2 a_, Vec2 b_) noexcept {
    return Vec2{ a_.x - b_.x, a_.y - b_.y };
}

/**
 * @brief Component-wise multiplication
 */
MMATH_FORCE_INLINE Vec2 vec2Mul(Vec2 a_, Vec2 b_) noexcept {
    return Vec2{ a_.x * b_.x, a_.y * b_.y };
}

/**
 * @brief Component-wise division
 */
MMATH_FORCE_INLINE Vec2 vec2Div(Vec2 a_, Vec2 b_) noexcept {
    return Vec2{ a_.x / b_.x, a_.y / b_.y };
}

/**
 * @brief Scale vector by scalar
 */
MMATH_FORCE_INLINE Vec2 vec2Scale(Vec2 v_, float s_) noexcept {
    return Vec2{ v_.x * s_, v_.y * s_ };
}

/**
 * @brief Negate vector
 */
MMATH_FORCE_INLINE Vec2 vec2Negate(Vec2 v_) noexcept {
    return Vec2{ -v_.x, -v_.y };
}

/**
 * @brief Dot product
 */
MMATH_FORCE_INLINE float vec2Dot(Vec2 a_, Vec2 b_) noexcept {
    return a_.x * b_.x + a_.y * b_.y;
}

/**
 * @brief Squared length (avoids sqrt)
 */
MMATH_FORCE_INLINE float vec2LengthSquared(Vec2 v_) noexcept {
    return v_.x * v_.x + v_.y * v_.y;
}

/**
 * @brief Length
 */
MMATH_FORCE_INLINE float vec2Length(Vec2 v_) noexcept {
    return std::sqrt(vec2LengthSquared(v_));
}

/**
 * @brief Normalize vector
 */
MMATH_FORCE_INLINE Vec2 vec2Normalize(Vec2 v_) noexcept {
    float len = vec2Length(v_);
    float inv_len = (len > 0.0f) ? (1.0f / len) : 0.0f;
    return vec2Scale(v_, inv_len);
}

/**
 * @brief Fast normalize using approximate reciprocal square root (~0.1% error)
 */
MMATH_FORCE_INLINE Vec2 vec2NormalizeFast(Vec2 v_) noexcept {
#if defined(__AVX2__) || defined(__SSE__)
    float len_sq = vec2LengthSquared(v_);
    __m128 vlen_sq = _mm_set_ss(len_sq);
    __m128 rsqrt = _mm_rsqrt_ss(vlen_sq);  // Approximate 1/sqrt
    float inv_len = _mm_cvtss_f32(rsqrt);
    return vec2Scale(v_, inv_len);
#else
    return vec2Normalize(v_);  // Fallback
#endif
}

/**
 * @brief 2D cross product (returns scalar z-component)
 */
MMATH_FORCE_INLINE float vec2Cross(Vec2 a_, Vec2 b_) noexcept {
    return a_.x * b_.y - a_.y * b_.x;
}

/**
 * @brief Perpendicular vector (90° counter-clockwise)
 */
MMATH_FORCE_INLINE Vec2 vec2Perpendicular(Vec2 v_) noexcept {
    return Vec2{ -v_.y, v_.x };
}

/**
 * @brief Perpendicular vector (90° clockwise)
 */
MMATH_FORCE_INLINE Vec2 vec2PerpendicularCw(Vec2 v_) noexcept {
    return Vec2{ v_.y, -v_.x };
}

/**
 * @brief Linear interpolation
 */
MMATH_FORCE_INLINE Vec2 vec2Lerp(Vec2 a_, Vec2 b_, float t_) noexcept {
    return Vec2{
        a_.x + (b_.x - a_.x) * t_,
        a_.y + (b_.y - a_.y) * t_
    };
}

/**
 * @brief Component-wise minimum
 */
MMATH_FORCE_INLINE Vec2 vec2Min(Vec2 a_, Vec2 b_) noexcept {
    return Vec2{
        (a_.x < b_.x) ? a_.x : b_.x,
        (a_.y < b_.y) ? a_.y : b_.y
    };
}

/**
 * @brief Component-wise maximum
 */
MMATH_FORCE_INLINE Vec2 vec2Max(Vec2 a_, Vec2 b_) noexcept {
    return Vec2{
        (a_.x > b_.x) ? a_.x : b_.x,
        (a_.y > b_.y) ? a_.y : b_.y
    };
}

/**
 * @brief Reflect vector around normal
 * Formula: v - 2 * dot(v, n) * n
 */
MMATH_FORCE_INLINE Vec2 vec2Reflect(Vec2 v_, Vec2 n_) noexcept {
    float d = vec2Dot(v_, n_);
    return Vec2{
        v_.x - 2.0f * d * n_.x,
        v_.y - 2.0f * d * n_.y
    };
}

/**
 * @brief Project vector a onto b
 * Formula: dot(a, b) / dot(b, b) * b
 */
MMATH_FORCE_INLINE Vec2 vec2Project(Vec2 a_, Vec2 b_) noexcept {
    float d = vec2Dot(a_, b_);
    float b_len_sq = vec2LengthSquared(b_);
    float scale = (b_len_sq > 0.0f) ? (d / b_len_sq) : 0.0f;
    return vec2Scale(b_, scale);
}

/**
 * @brief Squared distance between two points
 */
MMATH_FORCE_INLINE float vec2DistanceSquared(Vec2 a_, Vec2 b_) noexcept {
    Vec2 diff = vec2Sub(a_, b_);
    return vec2LengthSquared(diff);
}

/**
 * @brief Distance between two points
 */
MMATH_FORCE_INLINE float vec2Distance(Vec2 a_, Vec2 b_) noexcept {
    return std::sqrt(vec2DistanceSquared(a_, b_));
}

/**
 * @brief Equality comparison
 */
MMATH_FORCE_INLINE bool vec2Equal(Vec2 a_, Vec2 b_) noexcept {
    return (a_.x == b_.x) && (a_.y == b_.y);
}

/**
 * @brief Near-equality comparison with epsilon
 */
MMATH_FORCE_INLINE bool vec2NearEqual(Vec2 a_, Vec2 b_, float epsilon_) noexcept {
    float dx = a_.x - b_.x;
    float dy = a_.y - b_.y;
    return (dx * dx + dy * dy) < (epsilon_ * epsilon_);
}

} // namespace MMath (close before including vec2_simd.h)

// Include SIMD implementations after closing namespace

namespace MMath {

// ============================================================================
// Batch Processing (Optimized SIMD dispatch)
// ============================================================================

/**
 * @brief Batch add (SIMD optimized)
 *
 * @param a_ Input vectors A (array of Vec2)
 * @param b_ Input vectors B (array of Vec2)
 * @param result_ Output array
 * @param count_ Number of vectors
 */
inline void vec2AddArray(const Vec2* MMATH_RESTRICT a_,
                         const Vec2* MMATH_RESTRICT b_,
                         Vec2* MMATH_RESTRICT result_,
                         int32_t count_) noexcept {
#if defined(__AVX2__) && defined(__FMA__)
    detail::vec2AddSimd(a_, b_, result_, count_);
#else
    for (int32_t i = 0; i < count_; ++i) {
        result_[i] = vec2Add(a_[i], b_[i]);
    }
#endif
}

/**
 * @brief Batch dot product (SIMD optimized)
 */
inline void vec2DotArray(const Vec2* MMATH_RESTRICT a_,
                         const Vec2* MMATH_RESTRICT b_,
                         float* MMATH_RESTRICT result_,
                         int32_t count_) noexcept {
#if defined(__AVX2__) && defined(__FMA__)
    detail::vec2DotSimd(a_, b_, result_, count_);
#else
    for (int32_t i = 0; i < count_; ++i) {
        result_[i] = vec2Dot(a_[i], b_[i]);
    }
#endif
}

/**
 * @brief Batch normalize (SIMD optimized)
 */
inline void vec2NormalizeArray(const Vec2* MMATH_RESTRICT input_,
                               Vec2* MMATH_RESTRICT result_,
                               int32_t count_) noexcept {
#if defined(__AVX2__) && defined(__FMA__)
    detail::vec2NormalizeSimd(input_, result_, count_);
#else
    for (int32_t i = 0; i < count_; ++i) {
        result_[i] = vec2Normalize(input_[i]);
    }
#endif
}

/**
 * @brief Batch fast normalize (SIMD optimized)
 */
inline void vec2NormalizeFastArray(const Vec2* MMATH_RESTRICT input_,
                                   Vec2* MMATH_RESTRICT result_,
                                   int32_t count_) noexcept {
#if defined(__AVX2__) && defined(__FMA__)
    detail::vec2NormalizeFastSimd(input_, result_, count_);
#else
    for (int32_t i = 0; i < count_; ++i) {
        result_[i] = vec2NormalizeFast(input_[i]);
    }
#endif
}

} // namespace MMath
}
