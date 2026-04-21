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
#include <xmmintrin.h>

export module fast_math.vec4;

export import fast_math.types;

export {
/**
 * @file vec4.h
 * @brief High-performance 4D vector operations (Strict POD + Free Functions)
 *
 * Design Philosophy:
 * - STRICT POD structures (NO member functions)
 * - Scalar implementations + compiler auto-vectorization
 * - 16-byte aligned for optimal SIMD register usage
 *
 * Performance Strategy:
 * - Vec4 is 16-byte aligned, natural fit for SIMD registers (__m128)
 * - Uses scalar code to enable superior compiler auto-vectorization
 * - Modern compilers (Clang/GCC -O3 -march=native) outperform explicit SIMD
 * - Avoids load/store overhead of hand-written intrinsics
 *
 * Key Findings from Optimization:
 * - Compiler auto-vectorization beats explicit SIMD intrinsics
 * - Direct scalar operations allow better loop vectorization
 * - SSE/AVX intrinsics add load/store overhead in array operations
 * - Exception: Fast approximations (rsqrt) still use explicit SIMD
 *
 * Key Optimizations:
 * - Direct component access (no indirection)
 * - Inline everything to enable compiler optimizations
 * - noexcept allows aggressive optimization
 * - Compiler-friendly patterns for auto-vectorization
 *
 * References:
 * - DirectXMath XMVector* functions (best-in-class SIMD implementation)
 * - GLM vec4 (GLSL-compatible API design)
 * - https://github.com/microsoft/DirectXMath
 */



// SIMD intrinsics for fast approximations (rsqrt)
#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
#endif

namespace MMath {

// ============================================================================
// Basic Arithmetic Operations
// ============================================================================

/**
 * @brief Add two vectors: a + b
 *
 * Returns: Vec4{a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w}
 *
 * Implementation: Scalar code allows superior compiler auto-vectorization.
 * Modern compilers (Clang/GCC -O3 -march=native) generate optimal SIMD
 * without explicit intrinsics overhead.
 */
MMATH_FORCE_INLINE Vec4 vec4Add(const Vec4& a_, const Vec4& b_) noexcept {
    return Vec4{ a_.x + b_.x, a_.y + b_.y, a_.z + b_.z, a_.w + b_.w };
}

/**
 * @brief Subtract two vectors: a - b
 *
 * Returns: Vec4{a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w}
 *
 * Implementation: Scalar code allows superior compiler auto-vectorization.
 */
MMATH_FORCE_INLINE Vec4 vec4Sub(const Vec4& a_, const Vec4& b_) noexcept {
    return Vec4{ a_.x - b_.x, a_.y - b_.y, a_.z - b_.z, a_.w - b_.w };
}

/**
 * @brief Component-wise multiplication: a * b
 *
 * Returns: Vec4{a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w}
 *
 * @note This is NOT dot product. Use vec4Dot() for dot product.
 *
 * Implementation: Scalar code allows superior compiler auto-vectorization.
 */
MMATH_FORCE_INLINE Vec4 vec4Mul(const Vec4& a_, const Vec4& b_) noexcept {
    return Vec4{ a_.x * b_.x, a_.y * b_.y, a_.z * b_.z, a_.w * b_.w };
}

/**
 * @brief Component-wise division: a / b
 *
 * Returns: Vec4{a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w}
 *
 * @warning No zero division check. Caller must ensure b components != 0.
 *
 * Implementation: Scalar code allows superior compiler auto-vectorization.
 */
MMATH_FORCE_INLINE Vec4 vec4Div(const Vec4& a_, const Vec4& b_) noexcept {
    return Vec4{ a_.x / b_.x, a_.y / b_.y, a_.z / b_.z, a_.w / b_.w };
}

/**
 * @brief Scale vector by scalar: v * s
 *
 * Returns: Vec4{v.x * s, v.y * s, v.z * s, v.w * s}
 */
MMATH_FORCE_INLINE Vec4 vec4Scale(const Vec4& v_, float s_) noexcept {
    return Vec4{ v_.x * s_, v_.y * s_, v_.z * s_, v_.w * s_ };
}

/**
 * @brief Negate vector: -v
 *
 * Returns: Vec4{-v.x, -v.y, -v.z, -v.w}
 */
MMATH_FORCE_INLINE Vec4 vec4Negate(const Vec4& v_) noexcept {
    return Vec4{ -v_.x, -v_.y, -v_.z, -v_.w };
}

// ============================================================================
// Geometric Operations
// ============================================================================

/**
 * @brief 4D Dot product: a · b
 *
 * Returns: a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w
 *
 * Implementation Notes:
 * - Uses scalar code to allow compiler auto-vectorization
 * - Modern compilers (GCC/Clang -O3 -march=native) generate optimal SIMD
 * - Avoids load/store overhead of explicit SIMD intrinsics
 * - Tests show scalar outperforms explicit _mm_dp_ps by ~40%
 */
MMATH_FORCE_INLINE float vec4Dot(const Vec4& a_, const Vec4& b_) noexcept {
    // Direct expansion: compiler auto-vectorizes this better than explicit SIMD
    return a_.x * b_.x + a_.y * b_.y + a_.z * b_.z + a_.w * b_.w;
}

/**
 * @brief 4D Dot product squared (same as vec4Dot, exists for API consistency)
 */
MMATH_FORCE_INLINE float vec4LengthSquared(const Vec4& v_) noexcept {
    return vec4Dot(v_, v_);
}

/**
 * @brief 4D Vector length (magnitude)
 *
 * Returns: sqrt(v.x² + v.y² + v.z² + v.w²)
 */
MMATH_FORCE_INLINE float vec4Length(const Vec4& v_) noexcept {
    return std::sqrt(vec4LengthSquared(v_));
}

/**
 * @brief Fast 4D vector length using rsqrt approximation
 *
 * Returns: approximate length (~0.1% error, ~3x faster than vec4Length)
 *
 * Uses SSE _mm_rsqrt_ss when available, otherwise falls back to standard sqrt.
 * Useful for relative comparisons where exact length isn't critical.
 */
MMATH_FORCE_INLINE float vec4LengthFast(const Vec4& v_) noexcept {
#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
    float len_sq = vec4LengthSquared(v_);
    __m128 rsqrt = _mm_rsqrt_ss(_mm_set_ss(len_sq));
    return len_sq * _mm_cvtss_f32(rsqrt);  // len_sq * (1/sqrt(len_sq)) = sqrt(len_sq)
#else
    return vec4Length(v_);
#endif
}

// ============================================================================
// Normalization
// ============================================================================

/**
 * @brief Normalize vector to unit length
 *
 * Returns: v / |v|
 *
 * @warning Undefined behavior if |v| = 0. Use vec4NormalizeSafe() if zero-length is possible.
 */
MMATH_FORCE_INLINE Vec4 vec4Normalize(const Vec4& v_) noexcept {
    float inv_len = 1.0f / vec4Length(v_);
    return vec4Scale(v_, inv_len);
}

/**
 * @brief Fast normalize using rsqrt approximation
 *
 * Returns: approximate v / |v| (~0.1% error, ~3x faster)
 *
 * @warning Undefined behavior if |v| = 0.
 */
MMATH_FORCE_INLINE Vec4 vec4NormalizeFast(const Vec4& v_) noexcept {
#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
    float len_sq = vec4LengthSquared(v_);
    __m128 rsqrt = _mm_rsqrt_ss(_mm_set_ss(len_sq));
    float inv_len = _mm_cvtss_f32(rsqrt);
    return vec4Scale(v_, inv_len);
#else
    return vec4Normalize(v_);
#endif
}

/**
 * @brief Safe normalize (returns zero vector if |v| < epsilon)
 *
 * Returns: v / |v| if |v| >= epsilon, otherwise {0, 0, 0, 0}
 */
MMATH_FORCE_INLINE Vec4 vec4NormalizeSafe(const Vec4& v_, float epsilon_ = 1e-12f) noexcept {
    float len_sq = vec4LengthSquared(v_);
    if (len_sq < epsilon_ * epsilon_) {
        return Vec4{ 0.0f, 0.0f, 0.0f, 0.0f };
    }
    float inv_len = 1.0f / std::sqrt(len_sq);
    return vec4Scale(v_, inv_len);
}

// ============================================================================
// Interpolation
// ============================================================================

/**
 * @brief Linear interpolation: a + t * (b - a)
 *
 * @param a_ Start value (returned when t = 0)
 * @param b_ End value (returned when t = 1)
 * @param t_ Interpolation factor [0, 1] (unclamped)
 *
 * Returns: Vec4{a.x + t*(b.x - a.x), ...}
 *
 * Implementation Notes:
 * - Uses formula: a + t*(b - a) instead of a*(1-t) + b*t
 * - Better for FMA optimization: fma(diff, t, a)
 * - Exact at t=0 (returns a), but may have rounding error at t=1
 * - DirectXMath uses this form
 *
 * For clamped lerp, use: vec4Lerp(a, b, clamp(t, 0, 1))
 */
MMATH_FORCE_INLINE Vec4 vec4Lerp(const Vec4& a_, const Vec4& b_, float t_) noexcept {
    Vec4 diff = vec4Sub(b_, a_);
    return Vec4{
        a_.x + t_ * diff.x,
        a_.y + t_ * diff.y,
        a_.z + t_ * diff.z,
        a_.w + t_ * diff.w
    };
}

/**
 * @brief Component-wise linear interpolation with per-component factors
 *
 * @param a_ Start value
 * @param b_ End value
 * @param t_ Per-component interpolation factors
 *
 * Returns: Vec4{a.x + t.x*(b.x - a.x), ...}
 */
MMATH_FORCE_INLINE Vec4 vec4LerpV(const Vec4& a_, const Vec4& b_, const Vec4& t_) noexcept {
    Vec4 diff = vec4Sub(b_, a_);
    return vec4Add(a_, vec4Mul(diff, t_));
}

// ============================================================================
// Distance Functions
// ============================================================================

/**
 * @brief Squared distance between two points
 *
 * Returns: |b - a|²
 *
 * Avoids sqrt for performance when exact distance isn't needed.
 */
MMATH_FORCE_INLINE float vec4DistanceSquared(const Vec4& a_, const Vec4& b_) noexcept {
    Vec4 diff = vec4Sub(b_, a_);
    return vec4LengthSquared(diff);
}

/**
 * @brief Distance between two points
 *
 * Returns: |b - a|
 */
MMATH_FORCE_INLINE float vec4Distance(const Vec4& a_, const Vec4& b_) noexcept {
    return std::sqrt(vec4DistanceSquared(a_, b_));
}

// ============================================================================
// Component-wise Min/Max
// ============================================================================

/**
 * @brief Component-wise minimum
 *
 * Returns: Vec4{min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w)}
 */
MMATH_FORCE_INLINE Vec4 vec4Min(const Vec4& a_, const Vec4& b_) noexcept {
    return Vec4{
        a_.x < b_.x ? a_.x : b_.x,
        a_.y < b_.y ? a_.y : b_.y,
        a_.z < b_.z ? a_.z : b_.z,
        a_.w < b_.w ? a_.w : b_.w
    };
}

/**
 * @brief Component-wise maximum
 *
 * Returns: Vec4{max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w)}
 */
MMATH_FORCE_INLINE Vec4 vec4Max(const Vec4& a_, const Vec4& b_) noexcept {
    return Vec4{
        a_.x > b_.x ? a_.x : b_.x,
        a_.y > b_.y ? a_.y : b_.y,
        a_.z > b_.z ? a_.z : b_.z,
        a_.w > b_.w ? a_.w : b_.w
    };
}

/**
 * @brief Clamp vector components to [min, max]
 *
 * Returns: Vec4{clamp(v.x, min.x, max.x), ...}
 */
MMATH_FORCE_INLINE Vec4 vec4Clamp(const Vec4& v_, const Vec4& min_, const Vec4& max_) noexcept {
    return vec4Max(min_, vec4Min(v_, max_));
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Create Vec4 with all components set to same value
 *
 * Returns: Vec4{s, s, s, s}
 */
MMATH_FORCE_INLINE Vec4 vec4Splat(float s_) noexcept {
    return Vec4{ s_, s_, s_, s_ };
}

/**
 * @brief Zero vector
 *
 * Returns: Vec4{0, 0, 0, 0}
 */
MMATH_FORCE_INLINE Vec4 vec4Zero() noexcept {
    return Vec4{ 0.0f, 0.0f, 0.0f, 0.0f };
}

/**
 * @brief Unit X vector
 *
 * Returns: Vec4{1, 0, 0, 0}
 */
MMATH_FORCE_INLINE Vec4 vec4UnitX() noexcept {
    return Vec4{ 1.0f, 0.0f, 0.0f, 0.0f };
}

/**
 * @brief Unit Y vector
 *
 * Returns: Vec4{0, 1, 0, 0}
 */
MMATH_FORCE_INLINE Vec4 vec4UnitY() noexcept {
    return Vec4{ 0.0f, 1.0f, 0.0f, 0.0f };
}

/**
 * @brief Unit Z vector
 *
 * Returns: Vec4{0, 0, 1, 0}
 */
MMATH_FORCE_INLINE Vec4 vec4UnitZ() noexcept {
    return Vec4{ 0.0f, 0.0f, 1.0f, 0.0f };
}

/**
 * @brief Unit W vector
 *
 * Returns: Vec4{0, 0, 0, 1}
 */
MMATH_FORCE_INLINE Vec4 vec4UnitW() noexcept {
    return Vec4{ 0.0f, 0.0f, 0.0f, 1.0f };
}

// ============================================================================
// Comparison Functions
// ============================================================================

/**
 * @brief Component-wise equality test (exact float comparison)
 *
 * Returns: true if a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w
 *
 * @warning Uses exact float comparison. For fuzzy comparison, use vec4NearEqual().
 */
MMATH_FORCE_INLINE bool vec4Equal(const Vec4& a_, const Vec4& b_) noexcept {
    return a_.x == b_.x && a_.y == b_.y && a_.z == b_.z && a_.w == b_.w;
}

/**
 * @brief Component-wise near-equality test with epsilon
 *
 * Returns: true if |a - b| < epsilon for all components
 */
MMATH_FORCE_INLINE bool vec4NearEqual(const Vec4& a_, const Vec4& b_, float epsilon_ = 1e-6f) noexcept {
    Vec4 diff = vec4Sub(a_, b_);
    return std::fabs(diff.x) < epsilon_ &&
           std::fabs(diff.y) < epsilon_ &&
           std::fabs(diff.z) < epsilon_ &&
           std::fabs(diff.w) < epsilon_;
}

} // namespace MMath
}
