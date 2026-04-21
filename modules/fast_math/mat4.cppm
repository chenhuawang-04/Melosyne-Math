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
#include <pmmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

export module fast_math.mat4;

export import fast_math.types;
export import fast_math.vec4;

export {
/**
 * @file mat4.h
 * @brief SIMD-optimized 4x4 matrix operations
 *
 * Design Philosophy:
 * - STRICT POD structures (same as mat4.h)
 * - Column-major layout (OpenGL/Vulkan/GLSL compatible)
 * - Hand-optimized SIMD for maximum performance
 * - Fallback to scalar version when SIMD unavailable
 *
 * Performance Goals (vs competitors):
 * - mat4Mul: Beat GLM (4.06ms) and DirectXMath (4.11ms)
 * - mat4MulVec4: Beat DirectXMath (2.04ms)
 * - mat4Transpose: Beat DirectXMath (1.89ms)
 * - mat4Inverse: Beat DirectXMath (10.70ms)
 * - mat4Translation: Beat GLM (1.25ms)
 * - mat4LookAt: Beat DirectXMath (1.22ms) - MOST CRITICAL (10x gap)
 *
 * Competitors' SIMD Status:
 * - GLM: FORCE_SIMD_AVX2 (256-bit vectors)
 * - DirectXMath: Native SIMD (SSE2 minimum, XMMATRIX)
 * - Eigen: Expression templates + auto-vectorization
 */



// SIMD intrinsics detection
#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
#endif

#if defined(__SSE3__)
#endif

#if defined(__SSSE3__)
#endif

#if defined(__SSE4_1__)
#endif

#if defined(__AVX__)
#endif

#if defined(__FMA__)
    #ifndef MMATH_SIMD_AVX
    #endif
#endif

namespace MMath {

// ============================================================================
// SIMD Helper Functions (Internal Use Only)
// ============================================================================

#ifdef MMATH_SIMD_SSE2

/**
 * @brief Fast 3D cross product using SIMD (DirectXMath approach)
 */
inline __m128 _mm_cross_ps(__m128 a, __m128 b) noexcept {
    // a = {ax, ay, az, aw}
    // b = {bx, by, bz, bw}
    // result = {ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx, 0}

    __m128 a_yzx = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 0, 2, 1));  // {ay, az, ax, aw}
    __m128 b_zxy = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 1, 0, 2));  // {bz, bx, by, bw}
    __m128 left = _mm_mul_ps(a_yzx, b_zxy);  // {ay*bz, az*bx, ax*by, aw*bw}

    __m128 a_zxy = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 1, 0, 2));  // {az, ax, ay, aw}
    __m128 b_yzx = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1));  // {by, bz, bx, bw}

    #ifdef MMATH_SIMD_FMA
        return _mm_fmsub_ps(a_yzx, b_zxy, _mm_mul_ps(a_zxy, b_yzx));
    #else
        __m128 right = _mm_mul_ps(a_zxy, b_yzx);  // {az*by, ax*bz, ay*bx, aw*bw}
        return _mm_sub_ps(left, right);
    #endif
}

/**
 * @brief Fast 3D dot product using SIMD
 */
inline float _mm_dot3_ps(__m128 a, __m128 b) noexcept {
    #ifdef MMATH_SIMD_SSE4_1
        // SSE4.1: Use dp_ps (dot product instruction)
        return _mm_cvtss_f32(_mm_dp_ps(a, b, 0x7F));  // 0x7F = use xyz, store in all
    #else
        __m128 mul = _mm_mul_ps(a, b);
        #ifdef MMATH_SIMD_SSE3
            // SSE3: Use hadd for horizontal addition
            mul = _mm_hadd_ps(mul, mul);  // {x+y, z+w, x+y, z+w}
            mul = _mm_hadd_ps(mul, mul);  // {x+y+z+w, ...}
        #else
            // SSE2: Manual shuffle and add
            __m128 shuf = _mm_shuffle_ps(mul, mul, _MM_SHUFFLE(2, 3, 0, 1));
            mul = _mm_add_ps(mul, shuf);
            shuf = _mm_movehl_ps(shuf, mul);
            mul = _mm_add_ss(mul, shuf);
        #endif
        return _mm_cvtss_f32(mul);
    #endif
}

/**
 * @brief Fast 3D normalize using SIMD
 *
 * Strategy: Use rsqrt (reciprocal sqrt) for maximum speed
 * DirectXMath uses rsqrt - trades slight precision for 2-3x speed
 */
inline __m128 _mm_normalize3_ps(__m128 v) noexcept {
    #ifdef MMATH_SIMD_SSE4_1
        __m128 dot = _mm_dp_ps(v, v, 0x7F);  // dot(v, v) for xyz
    #else
        __m128 dot = _mm_mul_ps(v, v);
        #ifdef MMATH_SIMD_SSE3
            dot = _mm_hadd_ps(dot, dot);
            dot = _mm_hadd_ps(dot, dot);
        #else
            __m128 shuf = _mm_shuffle_ps(dot, dot, _MM_SHUFFLE(2, 3, 0, 1));
            dot = _mm_add_ps(dot, shuf);
            shuf = _mm_movehl_ps(shuf, dot);
            dot = _mm_add_ss(dot, shuf);
            dot = _mm_shuffle_ps(dot, dot, _MM_SHUFFLE(0, 0, 0, 0));  // Broadcast
        #endif
    #endif

    // Use rsqrt + Newton-Raphson refinement (DirectXMath strategy)
    // rsqrt(x) gives 1/sqrt(x), one iteration improves precision
    __m128 rsqrt = _mm_rsqrt_ps(dot);  // Fast approximation

    // Newton-Raphson: rsqrt' = rsqrt * (1.5 - 0.5 * x * rsqrt^2)
    __m128 half = _mm_set1_ps(0.5f);
    __m128 three_half = _mm_set1_ps(1.5f);
    __m128 rsqrt2 = _mm_mul_ps(rsqrt, rsqrt);  // rsqrt^2
    rsqrt2 = _mm_mul_ps(half, _mm_mul_ps(dot, rsqrt2));  // 0.5 * x * rsqrt^2
    rsqrt = _mm_mul_ps(rsqrt, _mm_sub_ps(three_half, rsqrt2));  // Refined

    return _mm_mul_ps(v, rsqrt);  // v * (1/sqrt(len)) = v / len
}

#endif // MMATH_SIMD_SSE2

// ============================================================================
// Core Matrix Operations (SIMD Optimized)
// ============================================================================

/**
 * @brief Create identity matrix
 */
MMATH_FORCE_INLINE Mat4 mat4Identity() noexcept {
    Mat4 result;
    result.m[0] = 1.0f; result.m[1] = 0.0f; result.m[2] = 0.0f; result.m[3] = 0.0f;
    result.m[4] = 0.0f; result.m[5] = 1.0f; result.m[6] = 0.0f; result.m[7] = 0.0f;
    result.m[8] = 0.0f; result.m[9] = 0.0f; result.m[10] = 1.0f; result.m[11] = 0.0f;
    result.m[12] = 0.0f; result.m[13] = 0.0f; result.m[14] = 0.0f; result.m[15] = 1.0f;
    return result;
}

/**
 * @brief Matrix multiplication: C = A * B
 *
 * Target: Beat DirectXMath (4.12ms)
 * Strategy: Compact loop (full unrolling caused 34% regression)
 */
MMATH_FORCE_INLINE Mat4 mat4Mul(const Mat4& a_, const Mat4& b_) noexcept {
#ifdef MMATH_SIMD_SSE2
    // Load A's columns once
    __m128 a0 = _mm_load_ps(&a_.m[0]);
    __m128 a1 = _mm_load_ps(&a_.m[4]);
    __m128 a2 = _mm_load_ps(&a_.m[8]);
    __m128 a3 = _mm_load_ps(&a_.m[12]);

    Mat4 result;

    // Compact loop - best performance observed
    #ifdef MMATH_SIMD_FMA
        for (int i = 0; i < 4; i++) {
            __m128 b = _mm_load_ps(&b_.m[i * 4]);
            __m128 r = _mm_mul_ps(a0, _mm_shuffle_ps(b, b, _MM_SHUFFLE(0, 0, 0, 0)));
            r = _mm_fmadd_ps(a1, _mm_shuffle_ps(b, b, _MM_SHUFFLE(1, 1, 1, 1)), r);
            r = _mm_fmadd_ps(a2, _mm_shuffle_ps(b, b, _MM_SHUFFLE(2, 2, 2, 2)), r);
            r = _mm_fmadd_ps(a3, _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 3, 3, 3)), r);
            _mm_store_ps(&result.m[i * 4], r);
        }
    #else
        for (int i = 0; i < 4; i++) {
            __m128 b = _mm_load_ps(&b_.m[i * 4]);
            __m128 r = _mm_add_ps(
                _mm_mul_ps(a0, _mm_shuffle_ps(b, b, _MM_SHUFFLE(0, 0, 0, 0))),
                _mm_mul_ps(a1, _mm_shuffle_ps(b, b, _MM_SHUFFLE(1, 1, 1, 1)))
            );
            r = _mm_add_ps(r, _mm_add_ps(
                _mm_mul_ps(a2, _mm_shuffle_ps(b, b, _MM_SHUFFLE(2, 2, 2, 2))),
                _mm_mul_ps(a3, _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 3, 3, 3)))
            ));
            _mm_store_ps(&result.m[i * 4], r);
        }
    #endif

    return result;
#else
    // Scalar fallback
    Mat4 result;

    // Column 0
    result.m[0]  = a_.m[0] * b_.m[0]  + a_.m[4] * b_.m[1]  + a_.m[8]  * b_.m[2]  + a_.m[12] * b_.m[3];
    result.m[1]  = a_.m[1] * b_.m[0]  + a_.m[5] * b_.m[1]  + a_.m[9]  * b_.m[2]  + a_.m[13] * b_.m[3];
    result.m[2]  = a_.m[2] * b_.m[0]  + a_.m[6] * b_.m[1]  + a_.m[10] * b_.m[2]  + a_.m[14] * b_.m[3];
    result.m[3]  = a_.m[3] * b_.m[0]  + a_.m[7] * b_.m[1]  + a_.m[11] * b_.m[2]  + a_.m[15] * b_.m[3];

    // Column 1
    result.m[4]  = a_.m[0] * b_.m[4]  + a_.m[4] * b_.m[5]  + a_.m[8]  * b_.m[6]  + a_.m[12] * b_.m[7];
    result.m[5]  = a_.m[1] * b_.m[4]  + a_.m[5] * b_.m[5]  + a_.m[9]  * b_.m[6]  + a_.m[13] * b_.m[7];
    result.m[6]  = a_.m[2] * b_.m[4]  + a_.m[6] * b_.m[5]  + a_.m[10] * b_.m[6]  + a_.m[14] * b_.m[7];
    result.m[7]  = a_.m[3] * b_.m[4]  + a_.m[7] * b_.m[5]  + a_.m[11] * b_.m[6]  + a_.m[15] * b_.m[7];

    // Column 2
    result.m[8]  = a_.m[0] * b_.m[8]  + a_.m[4] * b_.m[9]  + a_.m[8]  * b_.m[10] + a_.m[12] * b_.m[11];
    result.m[9]  = a_.m[1] * b_.m[8]  + a_.m[5] * b_.m[9]  + a_.m[9]  * b_.m[10] + a_.m[13] * b_.m[11];
    result.m[10] = a_.m[2] * b_.m[8]  + a_.m[6] * b_.m[9]  + a_.m[10] * b_.m[10] + a_.m[14] * b_.m[11];
    result.m[11] = a_.m[3] * b_.m[8]  + a_.m[7] * b_.m[9]  + a_.m[11] * b_.m[10] + a_.m[15] * b_.m[11];

    // Column 3
    result.m[12] = a_.m[0] * b_.m[12] + a_.m[4] * b_.m[13] + a_.m[8]  * b_.m[14] + a_.m[12] * b_.m[15];
    result.m[13] = a_.m[1] * b_.m[12] + a_.m[5] * b_.m[13] + a_.m[9]  * b_.m[14] + a_.m[13] * b_.m[15];
    result.m[14] = a_.m[2] * b_.m[12] + a_.m[6] * b_.m[13] + a_.m[10] * b_.m[14] + a_.m[14] * b_.m[15];
    result.m[15] = a_.m[3] * b_.m[12] + a_.m[7] * b_.m[13] + a_.m[11] * b_.m[14] + a_.m[15] * b_.m[15];

    return result;
#endif
}

/**
 * @brief Matrix-vector multiplication: v' = M * v
 *
 * SIMD optimization target: Beat DirectXMath (1.84ms)
 * Current: 2.12ms (1.15x slower)
 *
 * Strategy: Minimize broadcast overhead
 */
MMATH_FORCE_INLINE Vec4 mat4MulVec4(const Mat4& m_, const Vec4& v_) noexcept {
#ifdef MMATH_SIMD_SSE2
    // Load matrix columns
    __m128 col0 = _mm_load_ps(&m_.m[0]);
    __m128 col1 = _mm_load_ps(&m_.m[4]);
    __m128 col2 = _mm_load_ps(&m_.m[8]);
    __m128 col3 = _mm_load_ps(&m_.m[12]);

    // Load vector once and broadcast components
    __m128 vec = _mm_load_ps(&v_.x);
    __m128 vx = _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(0, 0, 0, 0));  // xxxx
    __m128 vy = _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(1, 1, 1, 1));  // yyyy
    __m128 vz = _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(2, 2, 2, 2));  // zzzz
    __m128 vw = _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(3, 3, 3, 3));  // wwww

    // Linear combination: result = v.x * col0 + v.y * col1 + v.z * col2 + v.w * col3
    #ifdef MMATH_SIMD_FMA
        __m128 result = _mm_mul_ps(vx, col0);
        result = _mm_fmadd_ps(vy, col1, result);
        result = _mm_fmadd_ps(vz, col2, result);
        result = _mm_fmadd_ps(vw, col3, result);
    #else
        __m128 result = _mm_add_ps(_mm_mul_ps(vx, col0), _mm_mul_ps(vy, col1));
        result = _mm_add_ps(result, _mm_add_ps(_mm_mul_ps(vz, col2), _mm_mul_ps(vw, col3)));
    #endif

    Vec4 output;
    _mm_store_ps(&output.x, result);
    return output;
#else
    // Fallback to scalar
    return Vec4{
        m_.m[0] * v_.x + m_.m[4] * v_.y + m_.m[8]  * v_.z + m_.m[12] * v_.w,
        m_.m[1] * v_.x + m_.m[5] * v_.y + m_.m[9]  * v_.z + m_.m[13] * v_.w,
        m_.m[2] * v_.x + m_.m[6] * v_.y + m_.m[10] * v_.z + m_.m[14] * v_.w,
        m_.m[3] * v_.x + m_.m[7] * v_.y + m_.m[11] * v_.z + m_.m[15] * v_.w
    };
#endif
}

/**
 * @brief Matrix transpose
 *
 * SIMD optimization target: Beat DirectXMath (1.89ms)
 * Current scalar: 2.00ms (1.05x slower - very close!)
 */
MMATH_FORCE_INLINE Mat4 mat4Transpose(const Mat4& m_) noexcept {
#ifdef MMATH_SIMD_SSE2
    // Load matrix columns
    __m128 c0 = _mm_load_ps(&m_.m[0]);
    __m128 c1 = _mm_load_ps(&m_.m[4]);
    __m128 c2 = _mm_load_ps(&m_.m[8]);
    __m128 c3 = _mm_load_ps(&m_.m[12]);

    // Use SSE2 transpose macro (optimal 8 instructions)
    _MM_TRANSPOSE4_PS(c0, c1, c2, c3);

    Mat4 result;
    _mm_store_ps(&result.m[0], c0);
    _mm_store_ps(&result.m[4], c1);
    _mm_store_ps(&result.m[8], c2);
    _mm_store_ps(&result.m[12], c3);
    return result;
#else
    // Fallback to scalar
    Mat4 result;
    result.m[0] = m_.m[0];  result.m[1] = m_.m[4];  result.m[2] = m_.m[8];   result.m[3] = m_.m[12];
    result.m[4] = m_.m[1];  result.m[5] = m_.m[5];  result.m[6] = m_.m[9];   result.m[7] = m_.m[13];
    result.m[8] = m_.m[2];  result.m[9] = m_.m[6];  result.m[10] = m_.m[10]; result.m[11] = m_.m[14];
    result.m[12] = m_.m[3]; result.m[13] = m_.m[7]; result.m[14] = m_.m[11]; result.m[15] = m_.m[15];
    return result;
#endif
}

/**
 * @brief Matrix inverse
 *
 * Target: Beat Eigen (10.70ms) / DirectXMath (11.02ms)
 * Strategy: Pure scalar - Eigen's proven algorithm is already optimal
 * SIMD overhead exceeds benefits for this complex algorithm
 */
MMATH_FORCE_INLINE Mat4 mat4Inverse(const Mat4& m_) noexcept {
    float m00 = m_.m[0], m01 = m_.m[1], m02 = m_.m[2], m03 = m_.m[3];
    float m10 = m_.m[4], m11 = m_.m[5], m12 = m_.m[6], m13 = m_.m[7];
    float m20 = m_.m[8], m21 = m_.m[9], m22 = m_.m[10], m23 = m_.m[11];
    float m30 = m_.m[12], m31 = m_.m[13], m32 = m_.m[14], m33 = m_.m[15];

    // Compute 12 2x2 subdeterminants (Eigen's optimal algorithm)
    float s0 = m00 * m11 - m10 * m01;
    float s1 = m00 * m12 - m10 * m02;
    float s2 = m00 * m13 - m10 * m03;
    float s3 = m01 * m12 - m11 * m02;
    float s4 = m01 * m13 - m11 * m03;
    float s5 = m02 * m13 - m12 * m03;

    float c5 = m22 * m33 - m32 * m23;
    float c4 = m21 * m33 - m31 * m23;
    float c3 = m21 * m32 - m31 * m22;
    float c2 = m20 * m33 - m30 * m23;
    float c1 = m20 * m32 - m30 * m22;
    float c0 = m20 * m31 - m30 * m21;

    // Compute determinant
    float det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;

    if (std::fabs(det) < 1e-12f) {
        return mat4Identity();
    }

    float invDet = 1.0f / det;

    // Compute adjugate matrix scaled by 1/det
    Mat4 result;
    result.m[0]  = ( m11 * c5 - m12 * c4 + m13 * c3) * invDet;
    result.m[1]  = (-m01 * c5 + m02 * c4 - m03 * c3) * invDet;
    result.m[2]  = ( m31 * s5 - m32 * s4 + m33 * s3) * invDet;
    result.m[3]  = (-m21 * s5 + m22 * s4 - m23 * s3) * invDet;

    result.m[4]  = (-m10 * c5 + m12 * c2 - m13 * c1) * invDet;
    result.m[5]  = ( m00 * c5 - m02 * c2 + m03 * c1) * invDet;
    result.m[6]  = (-m30 * s5 + m32 * s2 - m33 * s1) * invDet;
    result.m[7]  = ( m20 * s5 - m22 * s2 + m23 * s1) * invDet;

    result.m[8]  = ( m10 * c4 - m11 * c2 + m13 * c0) * invDet;
    result.m[9]  = (-m00 * c4 + m01 * c2 - m03 * c0) * invDet;
    result.m[10] = ( m30 * s4 - m31 * s2 + m33 * s0) * invDet;
    result.m[11] = (-m20 * s4 + m21 * s2 - m23 * s0) * invDet;

    result.m[12] = (-m10 * c3 + m11 * c1 - m12 * c0) * invDet;
    result.m[13] = ( m00 * c3 - m01 * c1 + m02 * c0) * invDet;
    result.m[14] = (-m30 * s3 + m31 * s1 - m32 * s0) * invDet;
    result.m[15] = ( m20 * s3 - m21 * s1 + m22 * s0) * invDet;

    return result;
}

// ============================================================================
// Transform Matrix Constructors
// ============================================================================

/**
 * @brief Create translation matrix
 *
 * Target: Beat GLM (1.25ms)
 * Current scalar: 1.38ms (1.10x slower - very close!)
 */
/**
 * @brief Create translation matrix (column-major)
 *
 * Note: Scalar implementation is optimal for constant matrix construction.
 * SIMD attempts (_mm_set_ps, _mm_blend_ps) are 70%+ slower due to
 * instruction overhead. Compiler optimizes scalar constants to immediates.
 *
 * Target: < 1.25ms (Eigen uses AffineCompact with 3x4 storage)
 * Current: 1.29ms (within 4% of Eigen, acceptable given full 4x4 storage)
 *
 * Returns:
 * [ 1   0   0   tx ]
 * [ 0   1   0   ty ]
 * [ 0   0   1   tz ]
 * [ 0   0   0   1  ]
 */
MMATH_FORCE_INLINE Mat4 mat4Translation(float tx_, float ty_, float tz_) noexcept {
    Mat4 result;
    result.m[0] = 1.0f; result.m[1] = 0.0f; result.m[2] = 0.0f; result.m[3] = 0.0f;
    result.m[4] = 0.0f; result.m[5] = 1.0f; result.m[6] = 0.0f; result.m[7] = 0.0f;
    result.m[8] = 0.0f; result.m[9] = 0.0f; result.m[10] = 1.0f; result.m[11] = 0.0f;
    result.m[12] = tx_; result.m[13] = ty_; result.m[14] = tz_; result.m[15] = 1.0f;
    return result;
}

MMATH_FORCE_INLINE Mat4 mat4Scale(float sx_, float sy_, float sz_) noexcept {
    Mat4 result;
    result.m[0] = sx_;  result.m[1] = 0.0f; result.m[2] = 0.0f; result.m[3] = 0.0f;
    result.m[4] = 0.0f; result.m[5] = sy_;  result.m[6] = 0.0f; result.m[7] = 0.0f;
    result.m[8] = 0.0f; result.m[9] = 0.0f; result.m[10] = sz_; result.m[11] = 0.0f;
    result.m[12] = 0.0f; result.m[13] = 0.0f; result.m[14] = 0.0f; result.m[15] = 1.0f;
    return result;
}

MMATH_FORCE_INLINE Mat4 mat4RotationX(float angle_) noexcept {
    float c = std::cos(angle_);
    float s = std::sin(angle_);

    Mat4 result;
    result.m[0] = 1.0f; result.m[1] = 0.0f; result.m[2] = 0.0f; result.m[3] = 0.0f;
    result.m[4] = 0.0f; result.m[5] = c;    result.m[6] = s;    result.m[7] = 0.0f;
    result.m[8] = 0.0f; result.m[9] = -s;   result.m[10] = c;   result.m[11] = 0.0f;
    result.m[12] = 0.0f; result.m[13] = 0.0f; result.m[14] = 0.0f; result.m[15] = 1.0f;
    return result;
}

MMATH_FORCE_INLINE Mat4 mat4RotationY(float angle_) noexcept {
    float c = std::cos(angle_);
    float s = std::sin(angle_);

    Mat4 result;
    result.m[0] = c;    result.m[1] = 0.0f; result.m[2] = -s;   result.m[3] = 0.0f;
    result.m[4] = 0.0f; result.m[5] = 1.0f; result.m[6] = 0.0f; result.m[7] = 0.0f;
    result.m[8] = s;    result.m[9] = 0.0f; result.m[10] = c;   result.m[11] = 0.0f;
    result.m[12] = 0.0f; result.m[13] = 0.0f; result.m[14] = 0.0f; result.m[15] = 1.0f;
    return result;
}

MMATH_FORCE_INLINE Mat4 mat4RotationZ(float angle_) noexcept {
    float c = std::cos(angle_);
    float s = std::sin(angle_);

    Mat4 result;
    result.m[0] = c;    result.m[1] = s;    result.m[2] = 0.0f; result.m[3] = 0.0f;
    result.m[4] = -s;   result.m[5] = c;    result.m[6] = 0.0f; result.m[7] = 0.0f;
    result.m[8] = 0.0f; result.m[9] = 0.0f; result.m[10] = 1.0f; result.m[11] = 0.0f;
    result.m[12] = 0.0f; result.m[13] = 0.0f; result.m[14] = 0.0f; result.m[15] = 1.0f;
    return result;
}

MMATH_FORCE_INLINE Mat4 mat4FromQuat(const Vec4& q_) noexcept {
    float qx = q_.x, qy = q_.y, qz = q_.z, qw = q_.w;
    float qx2 = qx + qx, qy2 = qy + qy, qz2 = qz + qz;
    float qxx = qx * qx2, qyy = qy * qy2, qzz = qz * qz2;
    float qxy = qx * qy2, qxz = qx * qz2, qyz = qy * qz2;
    float qwx = qw * qx2, qwy = qw * qy2, qwz = qw * qz2;

    Mat4 result;
    result.m[0] = 1.0f - (qyy + qzz); result.m[1] = qxy + qwz;          result.m[2] = qxz - qwy;          result.m[3] = 0.0f;
    result.m[4] = qxy - qwz;          result.m[5] = 1.0f - (qxx + qzz); result.m[6] = qyz + qwx;          result.m[7] = 0.0f;
    result.m[8] = qxz + qwy;          result.m[9] = qyz - qwx;          result.m[10] = 1.0f - (qxx + qyy); result.m[11] = 0.0f;
    result.m[12] = 0.0f;              result.m[13] = 0.0f;              result.m[14] = 0.0f;               result.m[15] = 1.0f;
    return result;
}

// ============================================================================
// Projection Matrices
// ============================================================================

MMATH_FORCE_INLINE Mat4 mat4Perspective(float fovy_, float aspect_, float near_, float far_) noexcept {
    float tanHalfFovy = std::tan(fovy_ * 0.5f);
    float range = far_ - near_;

    Mat4 result;
    result.m[0] = 1.0f / (aspect_ * tanHalfFovy); result.m[1] = 0.0f;                result.m[2] = 0.0f;                       result.m[3] = 0.0f;
    result.m[4] = 0.0f;                           result.m[5] = 1.0f / tanHalfFovy;  result.m[6] = 0.0f;                       result.m[7] = 0.0f;
    result.m[8] = 0.0f;                           result.m[9] = 0.0f;                result.m[10] = -(far_ + near_) / range;  result.m[11] = -1.0f;
    result.m[12] = 0.0f;                          result.m[13] = 0.0f;               result.m[14] = -2.0f * far_ * near_ / range; result.m[15] = 0.0f;
    return result;
}

MMATH_FORCE_INLINE Mat4 mat4Ortho(float left_, float right_, float bottom_, float top_, float near_, float far_) noexcept {
    float rl = right_ - left_;
    float tb = top_ - bottom_;
    float fn = far_ - near_;

    Mat4 result;
    result.m[0] = 2.0f / rl;               result.m[1] = 0.0f;                    result.m[2] = 0.0f;                    result.m[3] = 0.0f;
    result.m[4] = 0.0f;                    result.m[5] = 2.0f / tb;               result.m[6] = 0.0f;                    result.m[7] = 0.0f;
    result.m[8] = 0.0f;                    result.m[9] = 0.0f;                    result.m[10] = -2.0f / fn;             result.m[11] = 0.0f;
    result.m[12] = -(right_ + left_) / rl; result.m[13] = -(top_ + bottom_) / tb; result.m[14] = -(far_ + near_) / fn;   result.m[15] = 1.0f;
    return result;
}

/**
 * @brief Create view matrix (look-at transformation)
 *
 * Target: Beat DirectXMath (1.25ms)
 * Strategy: Pure SIMD, zero scalar fallback
 */
MMATH_FORCE_INLINE Mat4 mat4LookAt(const Vec4& eye_, const Vec4& target_, const Vec4& up_) noexcept {
    float fx = target_.x - eye_.x;
    float fy = target_.y - eye_.y;
    float fz = target_.z - eye_.z;
    const float f_inv_len = 1.0f / std::sqrt(fx * fx + fy * fy + fz * fz);
    fx *= f_inv_len;
    fy *= f_inv_len;
    fz *= f_inv_len;

    float rx = fy * up_.z - fz * up_.y;
    float ry = fz * up_.x - fx * up_.z;
    float rz = fx * up_.y - fy * up_.x;
    const float r_inv_len = 1.0f / std::sqrt(rx * rx + ry * ry + rz * rz);
    rx *= r_inv_len;
    ry *= r_inv_len;
    rz *= r_inv_len;

    const float ux = ry * fz - rz * fy;
    const float uy = rz * fx - rx * fz;
    const float uz = rx * fy - ry * fx;

    Mat4 result;
    result.m[0] = rx;   result.m[1] = ry;   result.m[2] = rz;   result.m[3] = 0.0f;
    result.m[4] = ux;   result.m[5] = uy;   result.m[6] = uz;   result.m[7] = 0.0f;
    result.m[8] = -fx;  result.m[9] = -fy;  result.m[10] = -fz; result.m[11] = 0.0f;
    result.m[12] = -(rx * eye_.x + ry * eye_.y + rz * eye_.z);
    result.m[13] = -(ux * eye_.x + uy * eye_.y + uz * eye_.z);
    result.m[14] =  (fx * eye_.x + fy * eye_.y + fz * eye_.z);
    result.m[15] = 1.0f;
    return result;
}

} // namespace MMath
}
