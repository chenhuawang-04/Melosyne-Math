/**
 * @file types.h
 * @brief Strict POD data structures (NO member functions)
 *
 * Design Philosophy:
 * - STRICT POD (Plain Old Data) - C-compatible
 * - NO constructors, NO destructors, NO member functions
 * - Manipulated ONLY by free functions
 * - Cache-line aligned for SIMD performance
 */

#pragma once

#include <cstdint>
#include <cstring>

#if defined(_MSC_VER)
    #define MMATH_ALIGN(x) __declspec(align(x))
    #define MMATH_FORCE_INLINE __forceinline
    #define MMATH_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
    #define MMATH_ALIGN(x) __attribute__((aligned(x)))
    #define MMATH_FORCE_INLINE __attribute__((always_inline)) inline
    #define MMATH_RESTRICT __restrict__
#else
    #define MMATH_ALIGN(x)
    #define MMATH_FORCE_INLINE inline
    #define MMATH_RESTRICT
#endif

namespace MMath {

// ============================================================================
// Strict POD Structures
// ============================================================================

/**
 * @brief 2D vector (POD, 8 bytes)
 */
struct Vec2 {
    float x;
    float y;
};

/**
 * @brief Single angle (POD)
 */
struct Angle {
    float value;
};

/**
 * @brief Single sin/cos pair (POD)
 */
struct SinCos {
    float sin;
    float cos;
};

/**
 * @brief 3x3 matrix (POD, 36 bytes, row-major)
 *
 * Layout:
 * [m00  m01  m02]   [sx*cos  -sy*sin  tx]
 * [m10  m11  m12] = [sx*sin   sy*cos  ty]
 * [m20  m21  m22]   [0        0       1 ] (for affine 2D transforms)
 */
struct Mat3 {
    float m00, m01, m02;  // Row 0
    float m10, m11, m12;  // Row 1
    float m20, m21, m22;  // Row 2
};

/**
 * @brief 2D Axis-Aligned Bounding Box (POD, 16 bytes)
 *
 * Internal Optimization: Stores -maxx, -maxy instead of positive values
 * This enables single-instruction Union and 3-instruction Overlap test
 *
 * Reference: https://mtsamis.com/blog/23_11_07_aabbs
 *
 * Users should use aabb2FromMinMax() constructor and aabb2Min/Max()
 * accessors to avoid confusion with the internal representation.
 */
struct Aabb2 {
    float minx, miny;         // Minimum coordinates
    float neg_maxx, neg_maxy; // Stored as -maxx, -maxy (SIMD optimization)
};

/**
 * @brief Batch of angles (plain array, POD-compatible)
 */
struct AngleBatch {
    const float* MMATH_RESTRICT data;  // Pointer to angle array
    int32_t count;                    // Number of elements
};

/**
 * @brief Batch of sin/cos pairs (separate arrays, POD-compatible)
 */
struct SinCosBatch {
    float* MMATH_RESTRICT sin_data;    // Output sine array
    float* MMATH_RESTRICT cos_data;    // Output cosine array
    int32_t count;                    // Number of elements
};

// ============================================================================
// SIMD-aligned structures (for internal processing)
// ============================================================================

#if defined(__AVX2__) && defined(__FMA__)

/**
 * @brief 8 angles packed for AVX2 (POD, 32-byte aligned)
 */
struct MMATH_ALIGN(32) AnglesAvx2 {
    float angles[8];
};

/**
 * @brief 8 sin/cos pairs for AVX2 (POD, 32-byte aligned)
 */
struct MMATH_ALIGN(32) SinCosAvx2 {
    float sins[8];
    float coss[8];
};

#endif // AVX2

#if defined(__SSE4_1__)

/**
 * @brief 4 angles packed for SSE4.1 (POD, 16-byte aligned)
 */
struct MMATH_ALIGN(16) AnglesSse {
    float angles[4];
};

/**
 * @brief 4 sin/cos pairs for SSE4.1 (POD, 16-byte aligned)
 */
struct MMATH_ALIGN(16) SinCosSse {
    float sins[4];
    float coss[4];
};

#endif // SSE4.1

} // namespace MMath
