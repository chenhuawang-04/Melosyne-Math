// Shared Mat4 implementation for header/module parity.

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

    __m128 a_zxy = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 1, 0, 2));  // {az, ax, ay, aw}
    __m128 b_yzx = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1));  // {by, bz, bx, bw}

    #ifdef MMATH_SIMD_FMA
        return _mm_fmsub_ps(a_yzx, b_zxy, _mm_mul_ps(a_zxy, b_yzx));
    #else
        __m128 left = _mm_mul_ps(a_yzx, b_zxy);  // {ay*bz, az*bx, ax*by, aw*bw}
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
MMATH_FORCE_INLINE Mat4 mat4Multiply(const Mat4& a_, const Mat4& b_) noexcept {
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
 * @brief Deprecated compatibility alias for 4x4 matrix multiplication
 */
MMATH_DEPRECATED("Use mat4Multiply() as the canonical API spelling.")
MMATH_FORCE_INLINE Mat4 mat4Mul(const Mat4& a_, const Mat4& b_) noexcept {
    return mat4Multiply(a_, b_);
}

/**
 * @brief Matrix-vector multiplication: v' = M * v
 *
 * SIMD optimization target: Beat DirectXMath (1.84ms)
 * Current: 2.12ms (1.15x slower)
 *
 * Strategy: Minimize broadcast overhead
 */
MMATH_FORCE_INLINE Vec4 mat4MultiplyVec4(const Mat4& m_, const Vec4& v_) noexcept {
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
 * @brief Deprecated compatibility alias for matrix-vector multiplication
 */
MMATH_DEPRECATED("Use mat4MultiplyVec4() as the canonical API spelling.")
MMATH_FORCE_INLINE Vec4 mat4MulVec4(const Mat4& m_, const Vec4& v_) noexcept {
    return mat4MultiplyVec4(m_, v_);
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

#include "mat4_checked_inverse.inl"

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
// Projection Matrices (Canonical: RH + Zero-to-One depth)
// ============================================================================

/**
 * @brief Canonical right-handed perspective projection (Depth [0, 1])
 *
 * This is the library's default Mat4 projection convention and matches the
 * zero-to-one clip-space depth used by Direct3D/Vulkan/Metal style pipelines.
 */
MMATH_FORCE_INLINE Mat4 mat4Perspective(float fovy_, float aspect_, float near_, float far_) noexcept {
    float tan_half_fovy = std::tan(fovy_ * 0.5f);
    float h = 1.0f / tan_half_fovy;
    float w = h / aspect_;
    float f_range = far_ / (near_ - far_);

    Mat4 result{};
    result.m[0] = w;
    result.m[5] = h;
    result.m[10] = f_range;
    result.m[11] = -1.0f;
    result.m[14] = f_range * near_;
    return result;
}

/**
 * @brief Explicit right-handed perspective projection (Depth [0, 1])
 */
MMATH_FORCE_INLINE Mat4 mat4PerspectiveRH(float fovy_, float aspect_, float near_, float far_) noexcept {
    return mat4Perspective(fovy_, aspect_, near_, far_);
}

/**
 * @brief Explicit left-handed perspective projection (Depth [0, 1])
 */
MMATH_FORCE_INLINE Mat4 mat4PerspectiveLH(float fovy_, float aspect_, float near_, float far_) noexcept {
    float tan_half_fovy = std::tan(fovy_ * 0.5f);
    float h = 1.0f / tan_half_fovy;
    float w = h / aspect_;
    float f_range = far_ / (far_ - near_);

    Mat4 result{};
    result.m[0] = w;
    result.m[5] = h;
    result.m[10] = f_range;
    result.m[11] = 1.0f;
    result.m[14] = -f_range * near_;
    return result;
}

/**
 * @brief Canonical right-handed orthographic projection (Depth [0, 1])
 */
MMATH_FORCE_INLINE Mat4 mat4Ortho(float left_, float right_, float bottom_, float top_, float near_, float far_) noexcept {
    float rl = right_ - left_;
    float tb = top_ - bottom_;
    float fn = far_ - near_;

    Mat4 result{};
    result.m[0] = 2.0f / rl;
    result.m[5] = 2.0f / tb;
    result.m[10] = -1.0f / fn;
    result.m[12] = -(right_ + left_) / rl;
    result.m[13] = -(top_ + bottom_) / tb;
    result.m[14] = -near_ / fn;
    result.m[15] = 1.0f;
    return result;
}

/**
 * @brief Explicit right-handed orthographic projection (Depth [0, 1])
 */
MMATH_FORCE_INLINE Mat4 mat4OrthoRH(float left_, float right_, float bottom_, float top_, float near_, float far_) noexcept {
    return mat4Ortho(left_, right_, bottom_, top_, near_, far_);
}

/**
 * @brief Explicit left-handed orthographic projection (Depth [0, 1])
 */
MMATH_FORCE_INLINE Mat4 mat4OrthoLH(float left_, float right_, float bottom_, float top_, float near_, float far_) noexcept {
    float rl = right_ - left_;
    float tb = top_ - bottom_;
    float fn = far_ - near_;

    Mat4 result{};
    result.m[0] = 2.0f / rl;
    result.m[5] = 2.0f / tb;
    result.m[10] = 1.0f / fn;
    result.m[12] = -(right_ + left_) / rl;
    result.m[13] = -(top_ + bottom_) / tb;
    result.m[14] = -near_ / fn;
    result.m[15] = 1.0f;
    return result;
}

// ============================================================================
// View Matrices
// ============================================================================

/**
 * @brief Canonical right-handed look-at view matrix
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

/**
 * @brief Explicit right-handed look-at view matrix
 */
MMATH_FORCE_INLINE Mat4 mat4LookAtRH(const Vec4& eye_, const Vec4& target_, const Vec4& up_) noexcept {
    return mat4LookAt(eye_, target_, up_);
}

/**
 * @brief Explicit left-handed look-at view matrix
 */
MMATH_FORCE_INLINE Mat4 mat4LookAtLH(const Vec4& eye_, const Vec4& target_, const Vec4& up_) noexcept {
    float fx = target_.x - eye_.x;
    float fy = target_.y - eye_.y;
    float fz = target_.z - eye_.z;
    const float f_inv_len = 1.0f / std::sqrt(fx * fx + fy * fy + fz * fz);
    fx *= f_inv_len;
    fy *= f_inv_len;
    fz *= f_inv_len;

    float rx = up_.y * fz - up_.z * fy;
    float ry = up_.z * fx - up_.x * fz;
    float rz = up_.x * fy - up_.y * fx;
    const float r_inv_len = 1.0f / std::sqrt(rx * rx + ry * ry + rz * rz);
    rx *= r_inv_len;
    ry *= r_inv_len;
    rz *= r_inv_len;

    const float ux = fy * rz - fz * ry;
    const float uy = fz * rx - fx * rz;
    const float uz = fx * ry - fy * rx;

    Mat4 result;
    result.m[0] = rx;  result.m[1] = ry;  result.m[2] = rz;  result.m[3] = 0.0f;
    result.m[4] = ux;  result.m[5] = uy;  result.m[6] = uz;  result.m[7] = 0.0f;
    result.m[8] = fx;  result.m[9] = fy;  result.m[10] = fz; result.m[11] = 0.0f;
    result.m[12] = -(rx * eye_.x + ry * eye_.y + rz * eye_.z);
    result.m[13] = -(ux * eye_.x + uy * eye_.y + uz * eye_.z);
    result.m[14] = -(fx * eye_.x + fy * eye_.y + fz * eye_.z);
    result.m[15] = 1.0f;
    return result;
}

// ============================================================================
// Utility Helpers
// ============================================================================

/**
 * @brief Copy matrix to float array without transposition
 */
MMATH_FORCE_INLINE void mat4CopyToArray(const Mat4& m_, float* dest_) noexcept {
    std::memcpy(dest_, m_.m, sizeof(float) * 16);
}

/**
 * @brief Check if a matrix is approximately identity
 */
[[nodiscard]] MMATH_FORCE_INLINE bool mat4IsIdentity(const Mat4& m_, float epsilon_ = 1e-6f) noexcept {
    const Mat4 id = mat4Identity();
    for (int i = 0; i < 16; ++i) {
        if (std::fabs(m_.m[i] - id.m[i]) > epsilon_) {
            return false;
        }
    }
    return true;
}
