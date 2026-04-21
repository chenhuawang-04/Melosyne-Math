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

export module fast_math.types;

export {
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
 * @brief 3D vector (POD, 12 bytes)
 */
struct Vec3 {
    float x;
    float y;
    float z;
};

/**
 * @brief 4D vector (POD, 16 bytes, SIMD-aligned)
 *
 * Also used for homogeneous coordinates (x, y, z, w)
 * For positions: w = 1.0
 * For directions: w = 0.0
 */
struct MMATH_ALIGN(16) Vec4 {
    float x;
    float y;
    float z;
    float w;
};

/**
 * @brief Quaternion (POD, 16 bytes, SIMD-aligned)
 *
 * Format: (x, y, z, w) where w is the real part
 * Identity quaternion: {0, 0, 0, 1}
 */
struct MMATH_ALIGN(16) Quat {
    float x;  // imaginary i
    float y;  // imaginary j
    float z;  // imaginary k
    float w;  // real (scalar)
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
 * Layout (ROW-MAJOR for CPU-side 2D transforms):
 * [m00  m01  m02]   [sx*cos  -sy*sin  tx]
 * [m10  m11  m12] = [sx*sin   sy*cos  ty]
 * [m20  m21  m22]   [0        0       1 ] (for affine 2D transforms)
 *
 * @note Row-major layout differs from Mat4 (column-major).
 *       Mat3 is optimized for CPU-side 2D calculations.
 *       Mat4 is optimized for GPU upload (OpenGL/Vulkan/DirectX).
 */
struct Mat3 {
    float m00, m01, m02;  // Row 0
    float m10, m11, m12;  // Row 1
    float m20, m21, m22;  // Row 2
};

/**
 * @brief 4x4 matrix (POD, 64 bytes, column-major, GPU-compatible)
 *
 * Layout (COLUMN-MAJOR for GPU upload):
 * m[col * 4 + row]
 *
 * | m[0] m[4] m[8]  m[12] |
 * | m[1] m[5] m[9]  m[13] |
 * | m[2] m[6] m[10] m[14] |
 * | m[3] m[7] m[11] m[15] |
 *
 * For affine transforms:
 * - Columns 0-2: rotation/scale
 * - Column 3: translation (m[12], m[13], m[14])
 *
 * Identity: {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1}
 *
 * @note Column-major layout differs from Mat3 (row-major).
 *       Mat4 is optimized for GPU upload (OpenGL/Vulkan/DirectX).
 *       Mat3 is optimized for CPU-side 2D calculations.
 */
struct MMATH_ALIGN(16) Mat4 {
    float m[16];
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
 * @brief 3D Axis-Aligned Bounding Box (POD, 24 bytes)
 *
 * Internal Optimization: Stores -max instead of positive values
 * This enables single-instruction Union and efficient Overlap test
 *
 * Users should use aabb3FromMinMax() constructor and aabb3Min/Max()
 * accessors to avoid confusion with the internal representation.
 */
struct Aabb3 {
    float minx, miny, minz;          // Minimum coordinates
    float neg_maxx, neg_maxy, neg_maxz; // Stored as -max (SIMD optimization)
};

// ============================================================================
// Integer Vectors
// ============================================================================

/**
 * @brief 2D integer vector (POD, 8 bytes)
 */
struct Vec2i {
    int32_t x;
    int32_t y;
};

/**
 * @brief 3D integer vector (POD, 12 bytes)
 */
struct Vec3i {
    int32_t x;
    int32_t y;
    int32_t z;
};

/**
 * @brief 4D integer vector (POD, 16 bytes, SIMD-aligned)
 */
struct MMATH_ALIGN(16) Vec4i {
    int32_t x;
    int32_t y;
    int32_t z;
    int32_t w;
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
}
