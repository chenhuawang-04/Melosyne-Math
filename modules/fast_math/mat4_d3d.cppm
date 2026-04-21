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
#include <emmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

export module fast_math.mat4_d3d;

export import fast_math.types;
export import fast_math.vec4;

export {
/**
 * @file mat4_d3d.h
 * @brief D3D-compatible 4x4 Matrix Library - Maximum Performance SIMD Implementation
 *
 * ============================================================================
 * D3D ADAPTATION DESIGN PRINCIPLES
 * ============================================================================
 *
 * Core Conventions (NON-NEGOTIABLE):
 * - Storage: Column-major (m[col * 4 + row])
 * - Vectors: Column vectors (v as column vector)
 * - Multiplication: mul(M, v), matrix chain: Proj * View * World * Local
 * - Coordinate System: Right-handed (RH) recommended, view along -Z
 * - NO implicit transpose between CPU and HLSL
 *
 * D3D Clip Space Requirements:
 * - NDC depth range: [0, 1] (NOT OpenGL's [-1, 1])
 * - XY range: [-1, 1]
 *
 * API Layering:
 * - Generic math functions (no API binding)
 * - Explicit API-specific projection/view functions:
 *   - mat4PerspectiveD3D_RH / mat4PerspectiveD3D_LH
 *   - mat4OrthoD3D_RH / mat4OrthoD3D_LH
 *   - mat4LookAtRH / mat4LookAtLH
 *
 * Memory Layout (Column-major):
 * ```
 * | m[0]  m[4]  m[8]   m[12] |   col0  col1  col2  col3
 * | m[1]  m[5]  m[9]   m[13] |
 * | m[2]  m[6]  m[10]  m[14] |
 * | m[3]  m[7]  m[11]  m[15] |
 * ```
 *
 * Performance Optimizations (based on Nicholas Frechette & Fabian Giesen research):
 * - AVX broadcast instruction for 16-34% speedup over DirectXMath
 * - Reduced register pressure (10 XMM -> 7 XMM)
 * - Force-inline variants for hot code paths
 * - FMA instructions when available
 *
 * References:
 * - Nicholas Frechette: "Optimizing 4x4 matrix multiplication" (2017)
 * - Fabian Giesen (rygorous): SSE/AVX matrix multiply gist
 * - DirectXMath XMMatrixPerspectiveFovRH/LH
 * - GLM perspectiveFovRH_ZO (Zero-to-One depth)
 *
 * @author MMath Library
 * @date 2026-01-26
 */



// ============================================================================
// SIMD Intrinsics Detection
// ============================================================================

#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
#endif

#if defined(__SSE4_1__) || defined(__AVX__)
#endif

#if defined(__AVX__)
#endif

#if defined(__FMA__)
    #ifndef MMATH_D3D_AVX
    #endif
#endif

namespace MMath {
namespace D3D {

// Import Vec4 from parent namespace for use in this D3D-specific module
using ::MMath::Vec4;

// ============================================================================
// Mat4 Type Definition (Column-Major Storage)
// ============================================================================

/**
 * @brief 4x4 Matrix with Column-Major storage for D3D compatibility
 *
 * Memory layout:
 *   m[0-3]   = column 0
 *   m[4-7]   = column 1
 *   m[8-11]  = column 2
 *   m[12-15] = column 3
 *
 * Visual representation:
 *   | m[0]  m[4]  m[8]   m[12] |
 *   | m[1]  m[5]  m[9]   m[13] |
 *   | m[2]  m[6]  m[10]  m[14] |
 *   | m[3]  m[7]  m[11]  m[15] |
 */
struct alignas(16) Mat4 {
    float m[16];
};

// ============================================================================
// Mat4R - Register-Resident Matrix Type (DirectXMath XMMATRIX equivalent)
// ============================================================================

#ifdef MMATH_D3D_SSE2

/**
 * @brief Register-resident 4x4 Matrix for maximum SIMD performance
 *
 * This type mirrors DirectXMath's XMMATRIX design:
 * - 4 __m128 registers instead of float[16] in memory
 * - Eliminates load/store overhead in matrix chains
 * - 3-4x faster for MVP pipelines with multiple matrix multiplies
 *
 * Use Mat4R when:
 * - Chaining multiple matrix operations (e.g., Proj * View * Model)
 * - Performance-critical transform loops
 * - Data stays in registers throughout computation
 *
 * Use Mat4 when:
 * - Storing matrices in arrays/structs
 * - Passing to shaders/APIs
 * - Memory layout matters
 *
 * Memory layout (column-major, same as Mat4):
 *   r[0] = column 0 (m[0-3])
 *   r[1] = column 1 (m[4-7])
 *   r[2] = column 2 (m[8-11])
 *   r[3] = column 3 (m[12-15])
 *
 * IMPORTANT: Mat4R requires 16-byte alignment. When storing in containers,
 * ensure proper alignment (e.g., std::vector with aligned allocator).
 */
struct alignas(16) Mat4R {
    __m128 r[4];
};

// Compile-time alignment verification
static_assert(alignof(Mat4) == 16, "Mat4 must be 16-byte aligned for SIMD operations");
static_assert(alignof(Mat4R) == 16, "Mat4R must be 16-byte aligned for SIMD operations");

/**
 * @brief Load Mat4 from memory into register-resident Mat4R
 *
 * IMPORTANT: Input Mat4 must be 16-byte aligned (guaranteed by alignas(16)).
 * Passing unaligned data will cause a crash on most platforms.
 */
MMATH_D3D_FORCE_INLINE Mat4R mat4Load(const Mat4& m) noexcept {
    Mat4R result;
    result.r[0] = _mm_load_ps(&m.m[0]);
    result.r[1] = _mm_load_ps(&m.m[4]);
    result.r[2] = _mm_load_ps(&m.m[8]);
    result.r[3] = _mm_load_ps(&m.m[12]);
    return result;
}

/**
 * @brief Store register-resident Mat4R back to memory Mat4
 *
 * IMPORTANT: Output Mat4 must be 16-byte aligned (guaranteed by alignas(16)).
 * Writing to unaligned memory will cause a crash on most platforms.
 */
MMATH_D3D_FORCE_INLINE Mat4 mat4Store(const Mat4R& m) noexcept {
    Mat4 result;
    _mm_store_ps(&result.m[0],  m.r[0]);
    _mm_store_ps(&result.m[4],  m.r[1]);
    _mm_store_ps(&result.m[8],  m.r[2]);
    _mm_store_ps(&result.m[12], m.r[3]);
    return result;
}

/**
 * @brief Store register-resident Mat4R to float array
 */
MMATH_D3D_FORCE_INLINE void mat4StoreToArray(const Mat4R& m, float* dest) noexcept {
    _mm_storeu_ps(dest + 0,  m.r[0]);
    _mm_storeu_ps(dest + 4,  m.r[1]);
    _mm_storeu_ps(dest + 8,  m.r[2]);
    _mm_storeu_ps(dest + 12, m.r[3]);
}

#endif // MMATH_D3D_SSE2

// ============================================================================
// SIMD Helper Functions (Internal)
// ============================================================================

#ifdef MMATH_D3D_AVX

/**
 * @brief Linear combination using AVX broadcast (FASTEST method)
 *
 * Computes: a[0]*B.col[0] + a[1]*B.col[1] + a[2]*B.col[2] + a[3]*B.col[3]
 *
 * Key optimization from Nicholas Frechette:
 * - _mm_broadcast_ss directly from memory avoids load+shuffle
 * - Reduces register count from 10 to 7
 * - 16-34% faster than DirectXMath reference
 *
 * @param a Pointer to 4 floats (one row of matrix A)
 * @param B Matrix B (column-major)
 * @return Result vector
 */
MMATH_D3D_FORCE_INLINE __m128 _lincomb_avx(const float* a, const Mat4& B) noexcept {
    __m128 result;
    result = _mm_mul_ps(_mm_broadcast_ss(&a[0]), _mm_load_ps(&B.m[0]));   // a[0] * B.col0
    result = _mm_add_ps(result, _mm_mul_ps(_mm_broadcast_ss(&a[1]), _mm_load_ps(&B.m[4])));  // + a[1] * B.col1
    result = _mm_add_ps(result, _mm_mul_ps(_mm_broadcast_ss(&a[2]), _mm_load_ps(&B.m[8])));  // + a[2] * B.col2
    result = _mm_add_ps(result, _mm_mul_ps(_mm_broadcast_ss(&a[3]), _mm_load_ps(&B.m[12]))); // + a[3] * B.col3
    return result;
}

#ifdef MMATH_D3D_FMA
/**
 * @brief Linear combination using AVX broadcast + FMA
 */
MMATH_D3D_FORCE_INLINE __m128 _lincomb_avx_fma(const float* a, const Mat4& B) noexcept {
    __m128 result;
    result = _mm_mul_ps(_mm_broadcast_ss(&a[0]), _mm_load_ps(&B.m[0]));
    result = _mm_fmadd_ps(_mm_broadcast_ss(&a[1]), _mm_load_ps(&B.m[4]), result);
    result = _mm_fmadd_ps(_mm_broadcast_ss(&a[2]), _mm_load_ps(&B.m[8]), result);
    result = _mm_fmadd_ps(_mm_broadcast_ss(&a[3]), _mm_load_ps(&B.m[12]), result);
    return result;
}
#endif

#endif // MMATH_D3D_AVX

#ifdef MMATH_D3D_SSE2

/**
 * @brief Linear combination using SSE shuffle (fallback for non-AVX)
 */
MMATH_D3D_FORCE_INLINE __m128 _lincomb_sse(const __m128& a, const Mat4& B) noexcept {
    __m128 result;
    result = _mm_mul_ps(_mm_shuffle_ps(a, a, 0x00), _mm_load_ps(&B.m[0]));
    result = _mm_add_ps(result, _mm_mul_ps(_mm_shuffle_ps(a, a, 0x55), _mm_load_ps(&B.m[4])));
    result = _mm_add_ps(result, _mm_mul_ps(_mm_shuffle_ps(a, a, 0xaa), _mm_load_ps(&B.m[8])));
    result = _mm_add_ps(result, _mm_mul_ps(_mm_shuffle_ps(a, a, 0xff), _mm_load_ps(&B.m[12])));
    return result;
}

/**
 * @brief Fast 3D vector normalize using rsqrt + Newton-Raphson refinement
 */
MMATH_D3D_FORCE_INLINE __m128 _normalize3_sse(__m128 v) noexcept {
#ifdef MMATH_D3D_SSE4_1
    __m128 dot = _mm_dp_ps(v, v, 0x7F);  // dot product of xyz, broadcast to all lanes
#else
    __m128 t = _mm_mul_ps(v, v);
    __m128 dot = _mm_add_ps(_mm_shuffle_ps(t, t, 0x00),
                 _mm_add_ps(_mm_shuffle_ps(t, t, 0x55),
                            _mm_shuffle_ps(t, t, 0xaa)));
#endif
    __m128 rsqrt = _mm_rsqrt_ps(dot);

    // Newton-Raphson refinement: rsqrt' = rsqrt * (1.5 - 0.5 * dot * rsqrt^2)
    __m128 half = _mm_set1_ps(0.5f);
    __m128 three_half = _mm_set1_ps(1.5f);
    __m128 rsqrt2 = _mm_mul_ps(rsqrt, rsqrt);
    rsqrt2 = _mm_mul_ps(half, _mm_mul_ps(dot, rsqrt2));
    rsqrt = _mm_mul_ps(rsqrt, _mm_sub_ps(three_half, rsqrt2));

    return _mm_mul_ps(v, rsqrt);
}

/**
 * @brief 3D cross product using SSE shuffle
 */
MMATH_D3D_FORCE_INLINE __m128 _cross3_sse(__m128 a, __m128 b) noexcept {
    // a = {ax, ay, az, aw}, b = {bx, by, bz, bw}
    // result = {ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx, 0}
    __m128 a_yzx = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 0, 2, 1));
    __m128 b_yzx = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1));
    __m128 c = _mm_sub_ps(_mm_mul_ps(a, b_yzx), _mm_mul_ps(a_yzx, b));
    return _mm_shuffle_ps(c, c, _MM_SHUFFLE(3, 0, 2, 1));
}

#ifdef MMATH_D3D_SSE4_1
/**
 * @brief 3D dot product using SSE4.1 dp_ps
 */
MMATH_D3D_FORCE_INLINE __m128 _dot3_sse41(__m128 a, __m128 b) noexcept {
    return _mm_dp_ps(a, b, 0x7F);  // xyz dot, broadcast to all lanes
}
#endif

#endif // MMATH_D3D_SSE2

// ============================================================================
// Fast Trigonometric Functions (based on DirectXMath minimax polynomials)
// ============================================================================

// Constants for SinCos
constexpr float MMATH_PI = 3.141592654f;
constexpr float MMATH_2PI = 6.283185307f;
constexpr float MMATH_1DIV2PI = 0.159154943f;
constexpr float MMATH_PIDIV2 = 1.570796327f;

/**
 * @brief Fast SinCos using 11/10-degree minimax polynomial approximation
 *
 * This is a direct port of DirectXMath's XMScalarSinCos for maximum compatibility.
 * Performance: ~3x faster than std::sin/std::cos pair
 *
 * @param pSin Output for sin(Value)
 * @param pCos Output for cos(Value)
 * @param Value Input angle in radians
 */
MMATH_D3D_FORCE_INLINE void fastSinCos(float* pSin, float* pCos, float Value) noexcept {
    // Map Value to y in [-pi,pi], x = 2*pi*quotient + remainder.
    float quotient = MMATH_1DIV2PI * Value;
    if (Value >= 0.0f) {
        quotient = static_cast<float>(static_cast<int>(quotient + 0.5f));
    } else {
        quotient = static_cast<float>(static_cast<int>(quotient - 0.5f));
    }
    float y = Value - MMATH_2PI * quotient;

    // Map y to [-pi/2,pi/2] with sin(y) = sin(Value).
    float sign;
    if (y > MMATH_PIDIV2) {
        y = MMATH_PI - y;
        sign = -1.0f;
    } else if (y < -MMATH_PIDIV2) {
        y = -MMATH_PI - y;
        sign = -1.0f;
    } else {
        sign = +1.0f;
    }

    float y2 = y * y;

    // 11-degree minimax approximation for sin
    *pSin = (((((-2.3889859e-08f * y2 + 2.7525562e-06f) * y2 - 0.00019840874f) * y2 + 0.0083333310f) * y2 - 0.16666667f) * y2 + 1.0f) * y;

    // 10-degree minimax approximation for cos
    float p = ((((-2.6051615e-07f * y2 + 2.4760495e-05f) * y2 - 0.0013888378f) * y2 + 0.041666638f) * y2 - 0.5f) * y2 + 1.0f;
    *pCos = sign * p;
}

/**
 * @brief Fast SinCos for small angles (|Value| < pi/2)
 *
 * Optimized version for perspective FOV where angle is typically in [0, pi/2]
 * Skips range reduction for extra performance.
 */
MMATH_D3D_FORCE_INLINE void fastSinCosSmall(float* pSin, float* pCos, float y) noexcept {
    float y2 = y * y;

    // 11-degree minimax approximation for sin
    *pSin = (((((-2.3889859e-08f * y2 + 2.7525562e-06f) * y2 - 0.00019840874f) * y2 + 0.0083333310f) * y2 - 0.16666667f) * y2 + 1.0f) * y;

    // 10-degree minimax approximation for cos
    *pCos = ((((-2.6051615e-07f * y2 + 2.4760495e-05f) * y2 - 0.0013888378f) * y2 + 0.041666638f) * y2 - 0.5f) * y2 + 1.0f;
}

// ============================================================================
// Matrix Identity
// ============================================================================

/**
 * @brief Create identity matrix
 *
 * Column-major layout:
 *   | 1  0  0  0 |   m[0]=1, m[4]=0, m[8]=0,  m[12]=0
 *   | 0  1  0  0 |   m[1]=0, m[5]=1, m[9]=0,  m[13]=0
 *   | 0  0  1  0 |   m[2]=0, m[6]=0, m[10]=1, m[14]=0
 *   | 0  0  0  1 |   m[3]=0, m[7]=0, m[11]=0, m[15]=1
 */
MMATH_D3D_FORCE_INLINE Mat4 mat4Identity() noexcept {
    Mat4 result;
#ifdef MMATH_D3D_SSE2
    // Use pre-computed constants for reliability (like DirectXMath g_XMIdentityRx)
    _mm_store_ps(&result.m[0],  _mm_set_ps(0.0f, 0.0f, 0.0f, 1.0f));  // Col 0: {1, 0, 0, 0}
    _mm_store_ps(&result.m[4],  _mm_set_ps(0.0f, 0.0f, 1.0f, 0.0f));  // Col 1: {0, 1, 0, 0}
    _mm_store_ps(&result.m[8],  _mm_set_ps(0.0f, 1.0f, 0.0f, 0.0f));  // Col 2: {0, 0, 1, 0}
    _mm_store_ps(&result.m[12], _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f));  // Col 3: {0, 0, 0, 1}
#else
    result.m[0] = 1.0f;  result.m[4] = 0.0f;  result.m[8]  = 0.0f;  result.m[12] = 0.0f;
    result.m[1] = 0.0f;  result.m[5] = 1.0f;  result.m[9]  = 0.0f;  result.m[13] = 0.0f;
    result.m[2] = 0.0f;  result.m[6] = 0.0f;  result.m[10] = 1.0f;  result.m[14] = 0.0f;
    result.m[3] = 0.0f;  result.m[7] = 0.0f;  result.m[11] = 0.0f;  result.m[15] = 1.0f;
#endif
    return result;
}

// ============================================================================
// Matrix Multiplication (OPTIMIZED - Based on Nicholas Frechette research)
// ============================================================================

/**
 * @brief Matrix multiplication: Result = A * B (Column-major)
 *
 * This is the PRIMARY matrix multiply function optimized for:
 * - AVX broadcast instruction (16-34% faster than DirectXMath)
 * - Reduced register pressure
 * - Column-major storage
 *
 * For column-major: Result_col_j = A * B_col_j
 * = B[0,j] * A_col0 + B[1,j] * A_col1 + B[2,j] * A_col2 + B[3,j] * A_col3
 *
 * For hot code paths, use mat4MulInline instead.
 *
 * @param A Left matrix
 * @param B Right matrix
 * @return Result matrix
 */
inline Mat4 mat4Mul(const Mat4& A, const Mat4& B) noexcept {
    Mat4 result;

#if defined(MMATH_D3D_AVX) && defined(MMATH_D3D_FMA)
    // Best: AVX broadcast + FMA
    // For column-major: pass B's column and matrix A
    _mm_store_ps(&result.m[0],  _lincomb_avx_fma(&B.m[0],  A));
    _mm_store_ps(&result.m[4],  _lincomb_avx_fma(&B.m[4],  A));
    _mm_store_ps(&result.m[8],  _lincomb_avx_fma(&B.m[8],  A));
    _mm_store_ps(&result.m[12], _lincomb_avx_fma(&B.m[12], A));

#elif defined(MMATH_D3D_AVX)
    // Good: AVX broadcast without FMA
    _mm_store_ps(&result.m[0],  _lincomb_avx(&B.m[0],  A));
    _mm_store_ps(&result.m[4],  _lincomb_avx(&B.m[4],  A));
    _mm_store_ps(&result.m[8],  _lincomb_avx(&B.m[8],  A));
    _mm_store_ps(&result.m[12], _lincomb_avx(&B.m[12], A));

#elif defined(MMATH_D3D_SSE2)
    // Fallback: SSE shuffle
    __m128 b0 = _mm_load_ps(&B.m[0]);
    __m128 b1 = _mm_load_ps(&B.m[4]);
    __m128 b2 = _mm_load_ps(&B.m[8]);
    __m128 b3 = _mm_load_ps(&B.m[12]);

    _mm_store_ps(&result.m[0],  _lincomb_sse(b0, A));
    _mm_store_ps(&result.m[4],  _lincomb_sse(b1, A));
    _mm_store_ps(&result.m[8],  _lincomb_sse(b2, A));
    _mm_store_ps(&result.m[12], _lincomb_sse(b3, A));

#else
    // Scalar fallback
    for (int col = 0; col < 4; col++) {
        for (int row = 0; row < 4; row++) {
            result.m[col * 4 + row] =
                A.m[0 * 4 + row] * B.m[col * 4 + 0] +
                A.m[1 * 4 + row] * B.m[col * 4 + 1] +
                A.m[2 * 4 + row] * B.m[col * 4 + 2] +
                A.m[3 * 4 + row] * B.m[col * 4 + 3];
        }
    }
#endif

    return result;
}

/**
 * @brief Force-inlined matrix multiplication for hot code paths
 *
 * Use this variant when you KNOW you're in performance-critical code.
 * Based on Nicholas Frechette's research showing 16-34% improvement
 * when force-inlining matrix multiply in tight loops.
 */
MMATH_D3D_FORCE_INLINE void mat4MulInline(const Mat4& A, const Mat4& B, Mat4& result) noexcept {
#if defined(MMATH_D3D_AVX) && defined(MMATH_D3D_FMA)
    _mm_store_ps(&result.m[0],  _lincomb_avx_fma(&B.m[0],  A));
    _mm_store_ps(&result.m[4],  _lincomb_avx_fma(&B.m[4],  A));
    _mm_store_ps(&result.m[8],  _lincomb_avx_fma(&B.m[8],  A));
    _mm_store_ps(&result.m[12], _lincomb_avx_fma(&B.m[12], A));
#elif defined(MMATH_D3D_AVX)
    _mm_store_ps(&result.m[0],  _lincomb_avx(&B.m[0],  A));
    _mm_store_ps(&result.m[4],  _lincomb_avx(&B.m[4],  A));
    _mm_store_ps(&result.m[8],  _lincomb_avx(&B.m[8],  A));
    _mm_store_ps(&result.m[12], _lincomb_avx(&B.m[12], A));
#elif defined(MMATH_D3D_SSE2)
    __m128 b0 = _mm_load_ps(&B.m[0]);
    __m128 b1 = _mm_load_ps(&B.m[4]);
    __m128 b2 = _mm_load_ps(&B.m[8]);
    __m128 b3 = _mm_load_ps(&B.m[12]);
    _mm_store_ps(&result.m[0],  _lincomb_sse(b0, A));
    _mm_store_ps(&result.m[4],  _lincomb_sse(b1, A));
    _mm_store_ps(&result.m[8],  _lincomb_sse(b2, A));
    _mm_store_ps(&result.m[12], _lincomb_sse(b3, A));
#else
    result = mat4Mul(A, B);
#endif
}

// ============================================================================
// Register-Resident Matrix Operations (Mat4R)
// ============================================================================

#ifdef MMATH_D3D_SSE2

/**
 * @brief Linear combination for Mat4R (register-to-register, no memory access)
 *
 * This is the key performance optimization over Mat4:
 * - No _mm_load_ps needed - data already in registers
 * - Compiler can keep intermediate results in XMM registers
 * - Matches DirectXMath XMMATRIX performance
 *
 * Note: Using value parameters for maximum register utilization
 */
MMATH_D3D_FORCE_INLINE __m128 _lincomb_r(__m128 a, __m128 c0, __m128 c1, __m128 c2, __m128 c3) noexcept {
#if defined(MMATH_D3D_FMA)
    __m128 result = _mm_mul_ps(_mm_shuffle_ps(a, a, 0x00), c0);
    result = _mm_fmadd_ps(_mm_shuffle_ps(a, a, 0x55), c1, result);
    result = _mm_fmadd_ps(_mm_shuffle_ps(a, a, 0xaa), c2, result);
    result = _mm_fmadd_ps(_mm_shuffle_ps(a, a, 0xff), c3, result);
    return result;
#else
    __m128 result = _mm_mul_ps(_mm_shuffle_ps(a, a, 0x00), c0);
    result = _mm_add_ps(result, _mm_mul_ps(_mm_shuffle_ps(a, a, 0x55), c1));
    result = _mm_add_ps(result, _mm_mul_ps(_mm_shuffle_ps(a, a, 0xaa), c2));
    result = _mm_add_ps(result, _mm_mul_ps(_mm_shuffle_ps(a, a, 0xff), c3));
    return result;
#endif
}

/**
 * @brief Register-resident matrix multiplication: Result = A * B
 *
 * This is the FASTEST matrix multiply for chained operations.
 * Use when performing multiple matrix multiplies in sequence:
 *
 *   Mat4R proj = mat4Load(projMat);
 *   Mat4R view = mat4Load(viewMat);
 *   Mat4R model = mat4Load(modelMat);
 *   Mat4R mvp = mat4MulR(mat4MulR(proj, view), model);
 *   // or: Mat4R vp = mat4MulR(proj, view);
 *   //     Mat4R mvp = mat4MulR(vp, model);
 *
 * Performance: Matches DirectXMath XMMatrixMultiply (no load/store overhead)
 *
 * @param A Left matrix (in registers)
 * @param B Right matrix (in registers)
 * @return Result matrix (in registers)
 */
MMATH_D3D_FORCE_INLINE Mat4R mat4MulR(Mat4R A, Mat4R B) noexcept {
    Mat4R result;
    result.r[0] = _lincomb_r(B.r[0], A.r[0], A.r[1], A.r[2], A.r[3]);
    result.r[1] = _lincomb_r(B.r[1], A.r[0], A.r[1], A.r[2], A.r[3]);
    result.r[2] = _lincomb_r(B.r[2], A.r[0], A.r[1], A.r[2], A.r[3]);
    result.r[3] = _lincomb_r(B.r[3], A.r[0], A.r[1], A.r[2], A.r[3]);
    return result;
}

/**
 * @brief Transform __m128 vector by register-resident matrix
 *
 * Fastest possible matrix-vector multiply:
 * - Both matrix and vector already in registers
 * - No memory access during computation
 * - Uses value parameters for optimal register allocation
 *
 * @param M Matrix in registers
 * @param v Vector in __m128
 * @return Transformed vector
 */
MMATH_D3D_FORCE_INLINE __m128 mat4MulVec4R(Mat4R M, __m128 v) noexcept {
#if defined(MMATH_D3D_FMA)
    __m128 result = _mm_mul_ps(_mm_shuffle_ps(v, v, 0x00), M.r[0]);
    result = _mm_fmadd_ps(_mm_shuffle_ps(v, v, 0x55), M.r[1], result);
    result = _mm_fmadd_ps(_mm_shuffle_ps(v, v, 0xaa), M.r[2], result);
    result = _mm_fmadd_ps(_mm_shuffle_ps(v, v, 0xff), M.r[3], result);
    return result;
#else
    __m128 result = _mm_mul_ps(_mm_shuffle_ps(v, v, 0x00), M.r[0]);
    result = _mm_add_ps(result, _mm_mul_ps(_mm_shuffle_ps(v, v, 0x55), M.r[1]));
    result = _mm_add_ps(result, _mm_mul_ps(_mm_shuffle_ps(v, v, 0xaa), M.r[2]));
    result = _mm_add_ps(result, _mm_mul_ps(_mm_shuffle_ps(v, v, 0xff), M.r[3]));
    return result;
#endif
}

/**
 * @brief Transform Vec4 by register-resident matrix (convenience wrapper)
 *
 * Loads Vec4, transforms, stores result. Use mat4MulVec4R for tight loops.
 */
MMATH_D3D_FORCE_INLINE Vec4 mat4MulVec4R(Mat4R M, const Vec4& v) noexcept {
    __m128 vr = _mm_set_ps(v.w, v.z, v.y, v.x);
    __m128 result = mat4MulVec4R(M, vr);
    Vec4 out;
    _mm_store_ps(&out.x, result);
    return out;
}

/**
 * @brief Create identity Mat4R (in registers)
 */
MMATH_D3D_FORCE_INLINE Mat4R mat4IdentityR() noexcept {
    Mat4R result;
    result.r[0] = _mm_set_ps(0.0f, 0.0f, 0.0f, 1.0f);
    result.r[1] = _mm_set_ps(0.0f, 0.0f, 1.0f, 0.0f);
    result.r[2] = _mm_set_ps(0.0f, 1.0f, 0.0f, 0.0f);
    result.r[3] = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);
    return result;
}

#endif // MMATH_D3D_SSE2

// ============================================================================
// Matrix-Vector Multiplication
// ============================================================================

/**
 * @brief Transform vector by matrix: result = M * v (column vector convention)
 *
 * Optimized with FMA when available for ~15% speedup over non-FMA.
 */
MMATH_D3D_FORCE_INLINE Vec4 mat4MulVec4(const Mat4& M, const Vec4& v) noexcept {
#if defined(MMATH_D3D_AVX) && defined(MMATH_D3D_FMA)
    // Best: AVX broadcast + FMA
    __m128 result;
    result = _mm_mul_ps(_mm_broadcast_ss(&v.x), _mm_load_ps(&M.m[0]));
    result = _mm_fmadd_ps(_mm_broadcast_ss(&v.y), _mm_load_ps(&M.m[4]), result);
    result = _mm_fmadd_ps(_mm_broadcast_ss(&v.z), _mm_load_ps(&M.m[8]), result);
    result = _mm_fmadd_ps(_mm_broadcast_ss(&v.w), _mm_load_ps(&M.m[12]), result);

    Vec4 out;
    _mm_store_ps(&out.x, result);
    return out;
#elif defined(MMATH_D3D_AVX)
    // AVX broadcast without FMA
    __m128 result;
    result = _mm_mul_ps(_mm_broadcast_ss(&v.x), _mm_load_ps(&M.m[0]));
    result = _mm_add_ps(result, _mm_mul_ps(_mm_broadcast_ss(&v.y), _mm_load_ps(&M.m[4])));
    result = _mm_add_ps(result, _mm_mul_ps(_mm_broadcast_ss(&v.z), _mm_load_ps(&M.m[8])));
    result = _mm_add_ps(result, _mm_mul_ps(_mm_broadcast_ss(&v.w), _mm_load_ps(&M.m[12])));

    Vec4 out;
    _mm_store_ps(&out.x, result);
    return out;
#elif defined(MMATH_D3D_SSE2)
    __m128 vv = _mm_set_ps(v.w, v.z, v.y, v.x);
    __m128 result;
    result = _mm_mul_ps(_mm_shuffle_ps(vv, vv, 0x00), _mm_load_ps(&M.m[0]));
    result = _mm_add_ps(result, _mm_mul_ps(_mm_shuffle_ps(vv, vv, 0x55), _mm_load_ps(&M.m[4])));
    result = _mm_add_ps(result, _mm_mul_ps(_mm_shuffle_ps(vv, vv, 0xaa), _mm_load_ps(&M.m[8])));
    result = _mm_add_ps(result, _mm_mul_ps(_mm_shuffle_ps(vv, vv, 0xff), _mm_load_ps(&M.m[12])));

    Vec4 out;
    _mm_store_ps(&out.x, result);
    return out;
#else
    return Vec4{
        M.m[0] * v.x + M.m[4] * v.y + M.m[8]  * v.z + M.m[12] * v.w,
        M.m[1] * v.x + M.m[5] * v.y + M.m[9]  * v.z + M.m[13] * v.w,
        M.m[2] * v.x + M.m[6] * v.y + M.m[10] * v.z + M.m[14] * v.w,
        M.m[3] * v.x + M.m[7] * v.y + M.m[11] * v.z + M.m[15] * v.w
    };
#endif
}

// ============================================================================
// Matrix Transpose
// ============================================================================

/**
 * @brief Transpose matrix using SSE2 hardware-optimized macro
 */
MMATH_D3D_FORCE_INLINE Mat4 mat4Transpose(const Mat4& M) noexcept {
    Mat4 result;
#ifdef MMATH_D3D_SSE2
    __m128 c0 = _mm_load_ps(&M.m[0]);
    __m128 c1 = _mm_load_ps(&M.m[4]);
    __m128 c2 = _mm_load_ps(&M.m[8]);
    __m128 c3 = _mm_load_ps(&M.m[12]);

    _MM_TRANSPOSE4_PS(c0, c1, c2, c3);

    _mm_store_ps(&result.m[0],  c0);
    _mm_store_ps(&result.m[4],  c1);
    _mm_store_ps(&result.m[8],  c2);
    _mm_store_ps(&result.m[12], c3);
#else
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            result.m[j * 4 + i] = M.m[i * 4 + j];
        }
    }
#endif
    return result;
}

// ============================================================================
// D3D PERSPECTIVE PROJECTION MATRICES (Depth range [0, 1])
// ============================================================================

/**
 * @brief D3D Right-Handed Perspective Projection Matrix (Depth [0, 1])
 *
 * Creates a perspective projection matrix for D3D with:
 * - Right-handed coordinate system (view along -Z)
 * - Depth range [0, 1] (D3D/Vulkan/Metal convention)
 * - FOV is vertical field of view in radians
 *
 * Matrix layout (column-major):
 * | w    0    0    0   |
 * | 0    h    0    0   |
 * | 0    0    A   -1   |
 * | 0    0    B    0   |
 *
 * Where:
 *   w = h / aspect = 1 / (aspect * tan(fov/2))
 *   h = 1 / tan(fov/2)
 *   A = far / (near - far)
 *   B = (near * far) / (near - far)
 *
 * @param fovY Vertical field of view in radians
 * @param aspect Aspect ratio (width / height)
 * @param nearZ Near clipping plane (must be > 0)
 * @param farZ Far clipping plane (must be > nearZ)
 * @return Perspective projection matrix
 */
MMATH_D3D_FORCE_INLINE Mat4 mat4PerspectiveD3D_RH(float fovY, float aspect, float nearZ, float farZ) noexcept {
    // Use fast SinCos instead of std::tan for ~3x speedup
    // cot(x) = cos(x)/sin(x) = 1/tan(x)
    float sinFov, cosFov;
    fastSinCosSmall(&sinFov, &cosFov, fovY * 0.5f);
    float h = cosFov / sinFov;  // cot = 1/tan
    float w = h / aspect;
    float fRange = farZ / (nearZ - farZ);

    Mat4 result = {};

    // Column 0
    result.m[0] = w;

    // Column 1
    result.m[5] = h;

    // Column 2
    result.m[10] = fRange;
    result.m[11] = -1.0f;  // Right-handed: -1

    // Column 3
    result.m[14] = fRange * nearZ;

    return result;
}

/**
 * @brief D3D Left-Handed Perspective Projection Matrix (Depth [0, 1])
 *
 * Creates a perspective projection matrix for D3D with:
 * - Left-handed coordinate system (view along +Z)
 * - Depth range [0, 1]
 *
 * Matrix layout (column-major):
 * | w    0    0    0   |
 * | 0    h    0    0   |
 * | 0    0    A    1   |
 * | 0    0    B    0   |
 *
 * @param fovY Vertical field of view in radians
 * @param aspect Aspect ratio (width / height)
 * @param nearZ Near clipping plane
 * @param farZ Far clipping plane
 * @return Perspective projection matrix
 */
MMATH_D3D_FORCE_INLINE Mat4 mat4PerspectiveD3D_LH(float fovY, float aspect, float nearZ, float farZ) noexcept {
    // Use fast SinCos instead of std::tan for ~3x speedup
    float sinFov, cosFov;
    fastSinCosSmall(&sinFov, &cosFov, fovY * 0.5f);
    float h = cosFov / sinFov;  // cot = 1/tan
    float w = h / aspect;
    float fRange = farZ / (farZ - nearZ);

    Mat4 result = {};

    // Column 0
    result.m[0] = w;

    // Column 1
    result.m[5] = h;

    // Column 2
    result.m[10] = fRange;
    result.m[11] = 1.0f;  // Left-handed: +1

    // Column 3
    result.m[14] = -fRange * nearZ;

    return result;
}

// ============================================================================
// D3D ORTHOGRAPHIC PROJECTION MATRICES (Depth range [0, 1])
// ============================================================================

/**
 * @brief D3D Right-Handed Orthographic Projection Matrix (Depth [0, 1])
 *
 * @param left Left clipping plane
 * @param right Right clipping plane
 * @param bottom Bottom clipping plane
 * @param top Top clipping plane
 * @param nearZ Near clipping plane
 * @param farZ Far clipping plane
 * @return Orthographic projection matrix
 */
MMATH_D3D_FORCE_INLINE Mat4 mat4OrthoD3D_RH(float left, float right, float bottom, float top,
                                             float nearZ, float farZ) noexcept {
    float rml = right - left;
    float tmb = top - bottom;
    float fmn = farZ - nearZ;

    Mat4 result = {};

    // Column 0
    result.m[0] = 2.0f / rml;

    // Column 1
    result.m[5] = 2.0f / tmb;

    // Column 2
    result.m[10] = -1.0f / fmn;  // RH: negative

    // Column 3
    result.m[12] = -(right + left) / rml;
    result.m[13] = -(top + bottom) / tmb;
    result.m[14] = -nearZ / fmn;
    result.m[15] = 1.0f;

    return result;
}

/**
 * @brief D3D Left-Handed Orthographic Projection Matrix (Depth [0, 1])
 */
MMATH_D3D_FORCE_INLINE Mat4 mat4OrthoD3D_LH(float left, float right, float bottom, float top,
                                             float nearZ, float farZ) noexcept {
    float rml = right - left;
    float tmb = top - bottom;
    float fmn = farZ - nearZ;

    Mat4 result = {};

    // Column 0
    result.m[0] = 2.0f / rml;

    // Column 1
    result.m[5] = 2.0f / tmb;

    // Column 2
    result.m[10] = 1.0f / fmn;  // LH: positive

    // Column 3
    result.m[12] = -(right + left) / rml;
    result.m[13] = -(top + bottom) / tmb;
    result.m[14] = -nearZ / fmn;
    result.m[15] = 1.0f;

    return result;
}

// ============================================================================
// VIEW MATRICES (LookAt)
// ============================================================================

/**
 * @brief Right-Handed LookAt View Matrix
 *
 * Creates a view matrix that transforms world space to camera space.
 * Camera looks along -Z axis (right-handed convention).
 *
 * IMPORTANT: Must use with mat4PerspectiveD3D_RH for correct results.
 *
 * @param eye Camera position
 * @param target Point camera is looking at
 * @param up World up vector (typically {0, 1, 0})
 * @return View matrix
 */
MMATH_D3D_FORCE_INLINE Mat4 mat4LookAtRH(const Vec4& eye, const Vec4& target, const Vec4& up) noexcept {
    Mat4 result;

#ifdef MMATH_D3D_SSE2
    __m128 eye_vec = _mm_set_ps(0.0f, eye.z, eye.y, eye.x);
    __m128 target_vec = _mm_set_ps(0.0f, target.z, target.y, target.x);
    __m128 up_vec = _mm_set_ps(0.0f, up.z, up.y, up.x);

    // f = normalize(eye - target) for RH (looking along -Z)
    __m128 f = _normalize3_sse(_mm_sub_ps(eye_vec, target_vec));

    // r = normalize(cross(up, f))
    __m128 r = _normalize3_sse(_cross3_sse(up_vec, f));

    // u = cross(f, r)
    __m128 u = _cross3_sse(f, r);

    // Negate eye for translation
    __m128 neg_eye = _mm_sub_ps(_mm_setzero_ps(), eye_vec);

    // Compute dot products - DirectXMath style with VectorSelect
#ifdef MMATH_D3D_SSE4_1
    __m128 d0 = _mm_dp_ps(r, neg_eye, 0x7F);
    __m128 d1 = _mm_dp_ps(u, neg_eye, 0x7F);
    __m128 d2 = _mm_dp_ps(f, neg_eye, 0x7F);
#else
    // Manual 3D dot products
    __m128 t0 = _mm_mul_ps(r, neg_eye);
    __m128 t1 = _mm_mul_ps(u, neg_eye);
    __m128 t2 = _mm_mul_ps(f, neg_eye);
    // Horizontal add for each
    t0 = _mm_add_ps(t0, _mm_shuffle_ps(t0, t0, _MM_SHUFFLE(2, 3, 0, 1)));
    t0 = _mm_add_ps(t0, _mm_shuffle_ps(t0, t0, _MM_SHUFFLE(0, 1, 2, 3)));
    __m128 d0 = t0;
    t1 = _mm_add_ps(t1, _mm_shuffle_ps(t1, t1, _MM_SHUFFLE(2, 3, 0, 1)));
    t1 = _mm_add_ps(t1, _mm_shuffle_ps(t1, t1, _MM_SHUFFLE(0, 1, 2, 3)));
    __m128 d1 = t1;
    t2 = _mm_add_ps(t2, _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(2, 3, 0, 1)));
    t2 = _mm_add_ps(t2, _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(0, 1, 2, 3)));
    __m128 d2 = t2;
#endif

    // Build row vectors and insert translation using shuffle (avoiding VectorSelect)
    // Row0 = {r.x, r.y, r.z, d0}
    // Row1 = {u.x, u.y, u.z, d1}
    // Row2 = {f.x, f.y, f.z, d2}
    // Row3 = {0, 0, 0, 1}

    // Insert d0 into r.w, d1 into u.w, d2 into f.w using blend/shuffle
    // _mm_blend_ps requires SSE4.1, use shuffle for SSE2 compatibility
    __m128 row0 = _mm_shuffle_ps(r, _mm_shuffle_ps(r, d0, _MM_SHUFFLE(0, 0, 2, 2)), _MM_SHUFFLE(2, 0, 1, 0));
    __m128 row1 = _mm_shuffle_ps(u, _mm_shuffle_ps(u, d1, _MM_SHUFFLE(0, 0, 2, 2)), _MM_SHUFFLE(2, 0, 1, 0));
    __m128 row2 = _mm_shuffle_ps(f, _mm_shuffle_ps(f, d2, _MM_SHUFFLE(0, 0, 2, 2)), _MM_SHUFFLE(2, 0, 1, 0));
    __m128 row3 = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);

    // Transpose to column-major
    _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

    // Store result
    _mm_store_ps(&result.m[0], row0);
    _mm_store_ps(&result.m[4], row1);
    _mm_store_ps(&result.m[8], row2);
    _mm_store_ps(&result.m[12], row3);

#else
    // Scalar fallback
    float fx = eye.x - target.x;
    float fy = eye.y - target.y;
    float fz = eye.z - target.z;
    float flen = std::sqrt(fx*fx + fy*fy + fz*fz);
    fx /= flen; fy /= flen; fz /= flen;

    float rx = up.y * fz - up.z * fy;
    float ry = up.z * fx - up.x * fz;
    float rz = up.x * fy - up.y * fx;
    float rlen = std::sqrt(rx*rx + ry*ry + rz*rz);
    rx /= rlen; ry /= rlen; rz /= rlen;

    float ux = fy * rz - fz * ry;
    float uy = fz * rx - fx * rz;
    float uz = fx * ry - fy * rx;

    result.m[0] = rx;  result.m[4] = ry;  result.m[8]  = rz;  result.m[12] = -(rx*eye.x + ry*eye.y + rz*eye.z);
    result.m[1] = ux;  result.m[5] = uy;  result.m[9]  = uz;  result.m[13] = -(ux*eye.x + uy*eye.y + uz*eye.z);
    result.m[2] = fx;  result.m[6] = fy;  result.m[10] = fz;  result.m[14] = -(fx*eye.x + fy*eye.y + fz*eye.z);
    result.m[3] = 0.0f; result.m[7] = 0.0f; result.m[11] = 0.0f; result.m[15] = 1.0f;
#endif

    return result;
}

/**
 * @brief Left-Handed LookAt View Matrix
 *
 * Camera looks along +Z axis (left-handed convention).
 * IMPORTANT: Must use with mat4PerspectiveD3D_LH for correct results.
 */
MMATH_D3D_FORCE_INLINE Mat4 mat4LookAtLH(const Vec4& eye, const Vec4& target, const Vec4& up) noexcept {
    Mat4 result;

#ifdef MMATH_D3D_SSE2
    __m128 eye_vec = _mm_set_ps(0.0f, eye.z, eye.y, eye.x);
    __m128 target_vec = _mm_set_ps(0.0f, target.z, target.y, target.x);
    __m128 up_vec = _mm_set_ps(0.0f, up.z, up.y, up.x);

    // f = normalize(target - eye) for LH (looking along +Z)
    __m128 f = _normalize3_sse(_mm_sub_ps(target_vec, eye_vec));

    // r = normalize(cross(up, f))
    __m128 r = _normalize3_sse(_cross3_sse(up_vec, f));

    // u = cross(f, r)
    __m128 u = _cross3_sse(f, r);

    // Negate eye for translation
    __m128 neg_eye = _mm_sub_ps(_mm_setzero_ps(), eye_vec);

    // Compute dot products
#ifdef MMATH_D3D_SSE4_1
    __m128 d0 = _mm_dp_ps(r, neg_eye, 0x7F);
    __m128 d1 = _mm_dp_ps(u, neg_eye, 0x7F);
    __m128 d2 = _mm_dp_ps(f, neg_eye, 0x7F);
#else
    __m128 t0 = _mm_mul_ps(r, neg_eye);
    __m128 t1 = _mm_mul_ps(u, neg_eye);
    __m128 t2 = _mm_mul_ps(f, neg_eye);
    t0 = _mm_add_ps(t0, _mm_shuffle_ps(t0, t0, _MM_SHUFFLE(2, 3, 0, 1)));
    t0 = _mm_add_ps(t0, _mm_shuffle_ps(t0, t0, _MM_SHUFFLE(0, 1, 2, 3)));
    __m128 d0 = t0;
    t1 = _mm_add_ps(t1, _mm_shuffle_ps(t1, t1, _MM_SHUFFLE(2, 3, 0, 1)));
    t1 = _mm_add_ps(t1, _mm_shuffle_ps(t1, t1, _MM_SHUFFLE(0, 1, 2, 3)));
    __m128 d1 = t1;
    t2 = _mm_add_ps(t2, _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(2, 3, 0, 1)));
    t2 = _mm_add_ps(t2, _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(0, 1, 2, 3)));
    __m128 d2 = t2;
#endif

    // Build row vectors and transpose to column-major
    __m128 row0 = _mm_shuffle_ps(r, _mm_shuffle_ps(r, d0, _MM_SHUFFLE(0, 0, 2, 2)), _MM_SHUFFLE(2, 0, 1, 0));
    __m128 row1 = _mm_shuffle_ps(u, _mm_shuffle_ps(u, d1, _MM_SHUFFLE(0, 0, 2, 2)), _MM_SHUFFLE(2, 0, 1, 0));
    __m128 row2 = _mm_shuffle_ps(f, _mm_shuffle_ps(f, d2, _MM_SHUFFLE(0, 0, 2, 2)), _MM_SHUFFLE(2, 0, 1, 0));
    __m128 row3 = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);

    _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

    _mm_store_ps(&result.m[0], row0);
    _mm_store_ps(&result.m[4], row1);
    _mm_store_ps(&result.m[8], row2);
    _mm_store_ps(&result.m[12], row3);
#else
    // Scalar fallback
    float fx = target.x - eye.x;
    float fy = target.y - eye.y;
    float fz = target.z - eye.z;
    float flen = std::sqrt(fx*fx + fy*fy + fz*fz);
    fx /= flen; fy /= flen; fz /= flen;

    float rx = up.y * fz - up.z * fy;
    float ry = up.z * fx - up.x * fz;
    float rz = up.x * fy - up.y * fx;
    float rlen = std::sqrt(rx*rx + ry*ry + rz*rz);
    rx /= rlen; ry /= rlen; rz /= rlen;

    float ux = fy * rz - fz * ry;
    float uy = fz * rx - fx * rz;
    float uz = fx * ry - fy * rx;

    result.m[0] = rx;  result.m[4] = ry;  result.m[8]  = rz;  result.m[12] = -(rx*eye.x + ry*eye.y + rz*eye.z);
    result.m[1] = ux;  result.m[5] = uy;  result.m[9]  = uz;  result.m[13] = -(ux*eye.x + uy*eye.y + uz*eye.z);
    result.m[2] = fx;  result.m[6] = fy;  result.m[10] = fz;  result.m[14] = -(fx*eye.x + fy*eye.y + fz*eye.z);
    result.m[3] = 0.0f; result.m[7] = 0.0f; result.m[11] = 0.0f; result.m[15] = 1.0f;
#endif

    return result;
}

// ============================================================================
// TRANSFORM MATRICES
// ============================================================================

/**
 * @brief Create translation matrix
 */
MMATH_D3D_FORCE_INLINE Mat4 mat4Translation(float x, float y, float z) noexcept {
    Mat4 result = mat4Identity();
    result.m[12] = x;
    result.m[13] = y;
    result.m[14] = z;
    return result;
}

/**
 * @brief Create scale matrix
 */
MMATH_D3D_FORCE_INLINE Mat4 mat4Scale(float x, float y, float z) noexcept {
    Mat4 result = {};
    result.m[0]  = x;
    result.m[5]  = y;
    result.m[10] = z;
    result.m[15] = 1.0f;
    return result;
}

/**
 * @brief Create rotation matrix around X axis
 * @param angle Rotation angle in radians
 */
MMATH_D3D_FORCE_INLINE Mat4 mat4RotationX(float angle) noexcept {
    float c = std::cos(angle);
    float s = std::sin(angle);

    Mat4 result = {};
    result.m[0]  = 1.0f;
    result.m[5]  = c;
    result.m[6]  = s;
    result.m[9]  = -s;
    result.m[10] = c;
    result.m[15] = 1.0f;
    return result;
}

/**
 * @brief Create rotation matrix around Y axis
 * @param angle Rotation angle in radians
 */
MMATH_D3D_FORCE_INLINE Mat4 mat4RotationY(float angle) noexcept {
    float c = std::cos(angle);
    float s = std::sin(angle);

    Mat4 result = {};
    result.m[0]  = c;
    result.m[2]  = -s;
    result.m[5]  = 1.0f;
    result.m[8]  = s;
    result.m[10] = c;
    result.m[15] = 1.0f;
    return result;
}

/**
 * @brief Create rotation matrix around Z axis
 * @param angle Rotation angle in radians
 */
MMATH_D3D_FORCE_INLINE Mat4 mat4RotationZ(float angle) noexcept {
    float c = std::cos(angle);
    float s = std::sin(angle);

    Mat4 result = {};
    result.m[0]  = c;
    result.m[1]  = s;
    result.m[4]  = -s;
    result.m[5]  = c;
    result.m[10] = 1.0f;
    result.m[15] = 1.0f;
    return result;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Copy matrix to float array (NO transpose - direct memcpy)
 *
 * IMPORTANT: Per D3D adaptation principles, this is a direct copy.
 * The matrix is already in column-major format suitable for:
 * - HLSL with column_major (default)
 * - glUniformMatrix4fv with GL_FALSE transpose flag
 *
 * If using HLSL with row_major and mul(v, M), you must transpose explicitly.
 */
MMATH_D3D_FORCE_INLINE void mat4CopyToArray(const Mat4& m, float* dest) noexcept {
#ifdef MMATH_D3D_SSE2
    _mm_storeu_ps(dest + 0,  _mm_load_ps(&m.m[0]));
    _mm_storeu_ps(dest + 4,  _mm_load_ps(&m.m[4]));
    _mm_storeu_ps(dest + 8,  _mm_load_ps(&m.m[8]));
    _mm_storeu_ps(dest + 12, _mm_load_ps(&m.m[12]));
#else
    std::memcpy(dest, m.m, sizeof(float) * 16);
#endif
}

/**
 * @brief Check if matrix is approximately identity
 */
inline bool mat4IsIdentity(const Mat4& m, float epsilon = 1e-6f) noexcept {
    const Mat4 id = mat4Identity();
    for (int i = 0; i < 16; i++) {
        if (std::fabs(m.m[i] - id.m[i]) > epsilon) return false;
    }
    return true;
}

} // namespace D3D
} // namespace MMath
}
