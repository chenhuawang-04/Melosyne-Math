/**
 * @file vec3.h
 * @brief High-performance 3D vector operations (Strict POD + Free Functions)
 *
 * Design Philosophy:
 * - STRICT POD structures (NO member functions)
 * - Scalar implementations optimized for single operations
 * - Compiler auto-vectorization friendly code patterns
 * - Optional SIMD batch processing in vec3_simd.h
 *
 * Performance Strategy:
 * - Vec3 is 12 bytes (not 16-byte aligned), so single SIMD ops have load/store overhead
 * - For single operations: scalar code often equals or beats explicit SIMD
 * - For batch operations: use vec3*Array() functions with SIMD
 *
 * Key Optimizations:
 * - vec3NormalizeFast: uses rsqrt approximation (~0.1% error, 3x faster)
 * - vec3LengthSquared: avoids sqrt when possible
 * - Inline everything to enable compiler optimizations
 *
 * References:
 * - DirectXMath XMVector3* (performance champion in benchmarks)
 * - GLM geometric.h SIMD implementations
 * - https://github.com/microsoft/DirectXMath
 */

#pragma once

#include "types.h"
#include <cmath>

#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
#include <xmmintrin.h>  // SSE
#endif

#if defined(__SSE4_1__) || defined(__AVX__)
#include <smmintrin.h>  // SSE4.1
#endif

namespace MMath {

// ============================================================================
// Basic Arithmetic Operations
// ============================================================================

/**
 * @brief Add two vectors: a + b
 */
MMATH_FORCE_INLINE Vec3 vec3Add(Vec3 a_, Vec3 b_) noexcept {
    return Vec3{ a_.x + b_.x, a_.y + b_.y, a_.z + b_.z };
}

/**
 * @brief Subtract two vectors: a - b
 */
MMATH_FORCE_INLINE Vec3 vec3Sub(Vec3 a_, Vec3 b_) noexcept {
    return Vec3{ a_.x - b_.x, a_.y - b_.y, a_.z - b_.z };
}

/**
 * @brief Component-wise multiplication: a * b
 */
MMATH_FORCE_INLINE Vec3 vec3Mul(Vec3 a_, Vec3 b_) noexcept {
    return Vec3{ a_.x * b_.x, a_.y * b_.y, a_.z * b_.z };
}

/**
 * @brief Component-wise division: a / b
 */
MMATH_FORCE_INLINE Vec3 vec3Div(Vec3 a_, Vec3 b_) noexcept {
    return Vec3{ a_.x / b_.x, a_.y / b_.y, a_.z / b_.z };
}

/**
 * @brief Scale vector by scalar: v * s
 */
MMATH_FORCE_INLINE Vec3 vec3Scale(Vec3 v_, float s_) noexcept {
    return Vec3{ v_.x * s_, v_.y * s_, v_.z * s_ };
}

/**
 * @brief Negate vector: -v
 */
MMATH_FORCE_INLINE Vec3 vec3Negate(Vec3 v_) noexcept {
    return Vec3{ -v_.x, -v_.y, -v_.z };
}

// ============================================================================
// Geometric Operations
// ============================================================================

/**
 * @brief Dot product: a · b
 *
 * Performance: 3 mul + 2 add = 5 ops
 * Compiler typically generates optimal scalar code
 */
MMATH_FORCE_INLINE float vec3Dot(Vec3 a_, Vec3 b_) noexcept {
    return a_.x * b_.x + a_.y * b_.y + a_.z * b_.z;
}

/**
 * @brief Cross product: a × b
 *
 * Formula: (a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x)
 * Performance: 6 mul + 3 sub = 9 ops
 */
MMATH_FORCE_INLINE Vec3 vec3Cross(Vec3 a_, Vec3 b_) noexcept {
    return Vec3{
        a_.y * b_.z - a_.z * b_.y,
        a_.z * b_.x - a_.x * b_.z,
        a_.x * b_.y - a_.y * b_.x
    };
}

/**
 * @brief Squared length: |v|²
 *
 * Use this when you only need to compare lengths (avoids sqrt)
 */
MMATH_FORCE_INLINE float vec3LengthSquared(Vec3 v_) noexcept {
    return v_.x * v_.x + v_.y * v_.y + v_.z * v_.z;
}

/**
 * @brief Length (magnitude): |v|
 */
MMATH_FORCE_INLINE float vec3Length(Vec3 v_) noexcept {
    return std::sqrt(vec3LengthSquared(v_));
}

/**
 * @brief Normalize vector to unit length
 *
 * Returns zero vector if input length is zero (safe division)
 */
MMATH_FORCE_INLINE Vec3 vec3Normalize(Vec3 v_) noexcept {
    float len_sq = vec3LengthSquared(v_);
    if (len_sq > 0.0f) {
        float inv_len = 1.0f / std::sqrt(len_sq);
        return Vec3{ v_.x * inv_len, v_.y * inv_len, v_.z * inv_len };
    }
    return Vec3{ 0.0f, 0.0f, 0.0f };
}

/**
 * @brief Fast normalize using approximate reciprocal square root
 *
 * Uses SSE rsqrt instruction (~0.1% relative error, ~3x faster)
 * Suitable for graphics where exact precision isn't critical
 *
 * WARNING: No zero-length check for maximum performance
 */
MMATH_FORCE_INLINE Vec3 vec3NormalizeFast(Vec3 v_) noexcept {
#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
    float len_sq = vec3LengthSquared(v_);
    __m128 vlen_sq = _mm_set_ss(len_sq);
    __m128 rsqrt = _mm_rsqrt_ss(vlen_sq);
    float inv_len = _mm_cvtss_f32(rsqrt);
    return Vec3{ v_.x * inv_len, v_.y * inv_len, v_.z * inv_len };
#else
    return vec3Normalize(v_);
#endif
}

/**
 * @brief Fast normalize with one Newton-Raphson iteration
 *
 * Better precision than vec3NormalizeFast (~0.001% error)
 * Still faster than full sqrt+div
 */
MMATH_FORCE_INLINE Vec3 vec3NormalizePrecise(Vec3 v_) noexcept {
#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
    float len_sq = vec3LengthSquared(v_);
    __m128 vlen_sq = _mm_set_ss(len_sq);
    __m128 rsqrt = _mm_rsqrt_ss(vlen_sq);
    // Newton-Raphson iteration: rsqrt = rsqrt * (1.5 - 0.5 * x * rsqrt * rsqrt)
    __m128 half = _mm_set_ss(0.5f);
    __m128 three_half = _mm_set_ss(1.5f);
    __m128 rsqrt_sq = _mm_mul_ss(rsqrt, rsqrt);
    __m128 half_x = _mm_mul_ss(half, vlen_sq);
    __m128 term = _mm_mul_ss(half_x, rsqrt_sq);
    __m128 factor = _mm_sub_ss(three_half, term);
    rsqrt = _mm_mul_ss(rsqrt, factor);
    float inv_len = _mm_cvtss_f32(rsqrt);
    return Vec3{ v_.x * inv_len, v_.y * inv_len, v_.z * inv_len };
#else
    return vec3Normalize(v_);
#endif
}

// ============================================================================
// Distance Operations
// ============================================================================

/**
 * @brief Squared distance between two points
 *
 * Use when comparing distances (avoids sqrt)
 */
MMATH_FORCE_INLINE float vec3DistanceSquared(Vec3 a_, Vec3 b_) noexcept {
    float dx = a_.x - b_.x;
    float dy = a_.y - b_.y;
    float dz = a_.z - b_.z;
    return dx * dx + dy * dy + dz * dz;
}

/**
 * @brief Distance between two points
 */
MMATH_FORCE_INLINE float vec3Distance(Vec3 a_, Vec3 b_) noexcept {
    return std::sqrt(vec3DistanceSquared(a_, b_));
}

// ============================================================================
// Interpolation and Clamping
// ============================================================================

/**
 * @brief Linear interpolation: a + (b - a) * t
 *
 * @param t_ Interpolation factor [0, 1]
 */
MMATH_FORCE_INLINE Vec3 vec3Lerp(Vec3 a_, Vec3 b_, float t_) noexcept {
    return Vec3{
        a_.x + (b_.x - a_.x) * t_,
        a_.y + (b_.y - a_.y) * t_,
        a_.z + (b_.z - a_.z) * t_
    };
}

/**
 * @brief Component-wise minimum
 */
MMATH_FORCE_INLINE Vec3 vec3Min(Vec3 a_, Vec3 b_) noexcept {
    return Vec3{
        (a_.x < b_.x) ? a_.x : b_.x,
        (a_.y < b_.y) ? a_.y : b_.y,
        (a_.z < b_.z) ? a_.z : b_.z
    };
}

/**
 * @brief Component-wise maximum
 */
MMATH_FORCE_INLINE Vec3 vec3Max(Vec3 a_, Vec3 b_) noexcept {
    return Vec3{
        (a_.x > b_.x) ? a_.x : b_.x,
        (a_.y > b_.y) ? a_.y : b_.y,
        (a_.z > b_.z) ? a_.z : b_.z
    };
}

/**
 * @brief Clamp each component to [min, max]
 */
MMATH_FORCE_INLINE Vec3 vec3Clamp(Vec3 v_, Vec3 min_, Vec3 max_) noexcept {
    return vec3Max(min_, vec3Min(v_, max_));
}

// ============================================================================
// Reflection and Projection
// ============================================================================

/**
 * @brief Reflect vector around normal
 *
 * Formula: v - 2 * dot(v, n) * n
 * @param v_ Incident vector
 * @param n_ Normal vector (should be normalized)
 */
MMATH_FORCE_INLINE Vec3 vec3Reflect(Vec3 v_, Vec3 n_) noexcept {
    float d = vec3Dot(v_, n_);
    float d2 = d + d;  // 2 * dot
    return Vec3{
        v_.x - d2 * n_.x,
        v_.y - d2 * n_.y,
        v_.z - d2 * n_.z
    };
}

/**
 * @brief Project vector a onto b
 *
 * Formula: (dot(a, b) / dot(b, b)) * b
 */
MMATH_FORCE_INLINE Vec3 vec3Project(Vec3 a_, Vec3 b_) noexcept {
    float b_len_sq = vec3LengthSquared(b_);
    if (b_len_sq > 0.0f) {
        float scale = vec3Dot(a_, b_) / b_len_sq;
        return vec3Scale(b_, scale);
    }
    return Vec3{ 0.0f, 0.0f, 0.0f };
}

/**
 * @brief Reject vector a from b (perpendicular component)
 *
 * Formula: a - project(a, b)
 */
MMATH_FORCE_INLINE Vec3 vec3Reject(Vec3 a_, Vec3 b_) noexcept {
    return vec3Sub(a_, vec3Project(a_, b_));
}

// ============================================================================
// Comparison Operations
// ============================================================================

/**
 * @brief Exact equality comparison
 */
MMATH_FORCE_INLINE bool vec3Equal(Vec3 a_, Vec3 b_) noexcept {
    return (a_.x == b_.x) && (a_.y == b_.y) && (a_.z == b_.z);
}

/**
 * @brief Near-equality comparison with epsilon
 */
MMATH_FORCE_INLINE bool vec3NearEqual(Vec3 a_, Vec3 b_, float epsilon_) noexcept {
    float dx = a_.x - b_.x;
    float dy = a_.y - b_.y;
    float dz = a_.z - b_.z;
    return (dx * dx + dy * dy + dz * dz) < (epsilon_ * epsilon_);
}

/**
 * @brief Check if vector is zero (or near-zero)
 */
MMATH_FORCE_INLINE bool vec3IsZero(Vec3 v_, float epsilon_ = 1e-6f) noexcept {
    return vec3LengthSquared(v_) < (epsilon_ * epsilon_);
}

/**
 * @brief Check if vector is normalized (unit length)
 */
MMATH_FORCE_INLINE bool vec3IsNormalized(Vec3 v_, float epsilon_ = 1e-5f) noexcept {
    float len_sq = vec3LengthSquared(v_);
    return std::abs(len_sq - 1.0f) < epsilon_;
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Create Vec3 from scalar (all components equal)
 */
MMATH_FORCE_INLINE Vec3 vec3Splat(float s_) noexcept {
    return Vec3{ s_, s_, s_ };
}

/**
 * @brief Create Vec3 from individual components
 */
MMATH_FORCE_INLINE Vec3 vec3Set(float x_, float y_, float z_) noexcept {
    return Vec3{ x_, y_, z_ };
}

/**
 * @brief Zero vector constant
 */
MMATH_FORCE_INLINE Vec3 vec3Zero() noexcept {
    return Vec3{ 0.0f, 0.0f, 0.0f };
}

/**
 * @brief Unit X axis (1, 0, 0)
 */
MMATH_FORCE_INLINE Vec3 vec3UnitX() noexcept {
    return Vec3{ 1.0f, 0.0f, 0.0f };
}

/**
 * @brief Unit Y axis (0, 1, 0)
 */
MMATH_FORCE_INLINE Vec3 vec3UnitY() noexcept {
    return Vec3{ 0.0f, 1.0f, 0.0f };
}

/**
 * @brief Unit Z axis (0, 0, 1)
 */
MMATH_FORCE_INLINE Vec3 vec3UnitZ() noexcept {
    return Vec3{ 0.0f, 0.0f, 1.0f };
}

/**
 * @brief Absolute value of each component
 */
MMATH_FORCE_INLINE Vec3 vec3Abs(Vec3 v_) noexcept {
    return Vec3{
        std::abs(v_.x),
        std::abs(v_.y),
        std::abs(v_.z)
    };
}

/**
 * @brief Get the component with maximum absolute value
 */
MMATH_FORCE_INLINE float vec3MaxComponent(Vec3 v_) noexcept {
    float m = (v_.x > v_.y) ? v_.x : v_.y;
    return (m > v_.z) ? m : v_.z;
}

/**
 * @brief Get the component with minimum absolute value
 */
MMATH_FORCE_INLINE float vec3MinComponent(Vec3 v_) noexcept {
    float m = (v_.x < v_.y) ? v_.x : v_.y;
    return (m < v_.z) ? m : v_.z;
}

// ============================================================================
// Advanced Operations
// ============================================================================

/**
 * @brief Compute angle between two vectors (in radians)
 *
 * @return Angle in range [0, PI]
 */
MMATH_FORCE_INLINE float vec3Angle(Vec3 a_, Vec3 b_) noexcept {
    float len_a = vec3Length(a_);
    float len_b = vec3Length(b_);
    if (len_a > 0.0f && len_b > 0.0f) {
        float cos_angle = vec3Dot(a_, b_) / (len_a * len_b);
        // Clamp to [-1, 1] to handle floating point errors
        cos_angle = (cos_angle < -1.0f) ? -1.0f : ((cos_angle > 1.0f) ? 1.0f : cos_angle);
        return std::acos(cos_angle);
    }
    return 0.0f;
}

/**
 * @brief Triple scalar product: a · (b × c)
 *
 * Also known as scalar triple product or box product.
 * Returns the signed volume of the parallelepiped formed by the three vectors.
 */
MMATH_FORCE_INLINE float vec3TripleScalar(Vec3 a_, Vec3 b_, Vec3 c_) noexcept {
    return vec3Dot(a_, vec3Cross(b_, c_));
}

/**
 * @brief Compute orthonormal basis from a single direction
 *
 * Given a direction vector, compute two perpendicular vectors
 * Uses the method from "Building an Orthonormal Basis, Revisited" (Duff et al., 2017)
 *
 * @param dir_ Input direction (will be normalized)
 * @param tangent_ Output tangent vector
 * @param bitangent_ Output bitangent vector
 */
inline void vec3OrthonormalBasis(Vec3 dir_, Vec3& tangent_, Vec3& bitangent_) noexcept {
    Vec3 n = vec3Normalize(dir_);

    // Choose the axis most perpendicular to n
    float sign = (n.z >= 0.0f) ? 1.0f : -1.0f;
    float a = -1.0f / (sign + n.z);
    float b = n.x * n.y * a;

    tangent_ = Vec3{ 1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x };
    bitangent_ = Vec3{ b, sign + n.y * n.y * a, -n.y };
}

} // namespace MMath

// ============================================================================
// Batch Processing (SIMD optimized) - Include after namespace close
// ============================================================================
// TODO: #include "vec3_simd.h" for batch array operations
