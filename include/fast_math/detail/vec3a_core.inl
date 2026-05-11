// Shared Vec3A SIMD core implementation for header/module parity.


// ============================================================================
// Vec3A - 16-byte Aligned Vec3 for SIMD
// ============================================================================

/**
 * @brief 16-byte aligned 3D vector for SIMD operations
 *
 * Uses __m128 for storage. The 4th component (w) is always 0.
 * This wastes 25% memory but enables efficient SIMD operations.
 *
 * Memory layout: [x, y, z, 0] (16 bytes, 16-byte aligned)
 */
struct MMATH_ALIGN(16) Vec3A {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#endif
    union {
        struct { float x, y, z, _pad; };
#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
        __m128 data;
#endif
        float v[4];
    };
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
};

// ============================================================================
// Vec3A Creation and Conversion
// ============================================================================

/**
 * @brief Create Vec3A from components
 */
MMATH_FORCE_INLINE Vec3A vec3ASet(float x_, float y_, float z_) noexcept {
    Vec3A result;
#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
    result.data = _mm_set_ps(0.0f, z_, y_, x_);
#else
    result.x = x_;
    result.y = y_;
    result.z = z_;
    result._pad = 0.0f;
#endif
    return result;
}

/**
 * @brief Create Vec3A with all components equal
 */
MMATH_FORCE_INLINE Vec3A vec3ASplat(float s_) noexcept {
    Vec3A result;
#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
    result.data = _mm_set_ps(0.0f, s_, s_, s_);
#else
    result.x = result.y = result.z = s_;
    result._pad = 0.0f;
#endif
    return result;
}

/**
 * @brief Create zero Vec3A
 */
MMATH_FORCE_INLINE Vec3A vec3AZero() noexcept {
    Vec3A result;
#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
    result.data = _mm_setzero_ps();
#else
    result.x = result.y = result.z = result._pad = 0.0f;
#endif
    return result;
}

/**
 * @brief Convert Vec3 to Vec3A
 */
MMATH_FORCE_INLINE Vec3A vec3ToVec3A(Vec3 v_) noexcept {
    return vec3ASet(v_.x, v_.y, v_.z);
}

/**
 * @brief Convert Vec3A to Vec3
 */
MMATH_FORCE_INLINE Vec3 vec3AToVec3(Vec3A v_) noexcept {
    return Vec3{ v_.x, v_.y, v_.z };
}

/**
 * @brief Load Vec3A from __m128 (for internal use)
 */
#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
MMATH_FORCE_INLINE Vec3A vec3AFromM128(__m128 m_) noexcept {
    Vec3A result;
    result.data = m_;
    return result;
}
#endif

// ============================================================================
// Vec3A Basic Arithmetic (SIMD optimized)
// ============================================================================

/**
 * @brief Add two Vec3A
 */
MMATH_FORCE_INLINE Vec3A vec3AAdd(Vec3A a_, Vec3A b_) noexcept {
#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
    Vec3A result;
    result.data = _mm_add_ps(a_.data, b_.data);
    return result;
#else
    return vec3ASet(a_.x + b_.x, a_.y + b_.y, a_.z + b_.z);
#endif
}

/**
 * @brief Subtract two Vec3A
 */
MMATH_FORCE_INLINE Vec3A vec3ASub(Vec3A a_, Vec3A b_) noexcept {
#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
    Vec3A result;
    result.data = _mm_sub_ps(a_.data, b_.data);
    return result;
#else
    return vec3ASet(a_.x - b_.x, a_.y - b_.y, a_.z - b_.z);
#endif
}

/**
 * @brief Component-wise multiply
 */
MMATH_FORCE_INLINE Vec3A vec3AMul(Vec3A a_, Vec3A b_) noexcept {
#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
    Vec3A result;
    result.data = _mm_mul_ps(a_.data, b_.data);
    return result;
#else
    return vec3ASet(a_.x * b_.x, a_.y * b_.y, a_.z * b_.z);
#endif
}

/**
 * @brief Component-wise divide
 */
MMATH_FORCE_INLINE Vec3A vec3ADiv(Vec3A a_, Vec3A b_) noexcept {
#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
    Vec3A result;
    result.data = _mm_div_ps(a_.data, b_.data);
    return result;
#else
    return vec3ASet(a_.x / b_.x, a_.y / b_.y, a_.z / b_.z);
#endif
}

/**
 * @brief Scale Vec3A by scalar
 */
MMATH_FORCE_INLINE Vec3A vec3AScale(Vec3A v_, float s_) noexcept {
#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
    Vec3A result;
    __m128 scalar = _mm_set1_ps(s_);
    result.data = _mm_mul_ps(v_.data, scalar);
    return result;
#else
    return vec3ASet(v_.x * s_, v_.y * s_, v_.z * s_);
#endif
}

/**
 * @brief Negate Vec3A
 */
MMATH_FORCE_INLINE Vec3A vec3ANegate(Vec3A v_) noexcept {
#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
    Vec3A result;
    result.data = _mm_sub_ps(_mm_setzero_ps(), v_.data);
    return result;
#else
    return vec3ASet(-v_.x, -v_.y, -v_.z);
#endif
}

// ============================================================================
// Vec3A Geometric Operations (SIMD optimized)
// ============================================================================

/**
 * @brief Dot product (returns float)
 *
 * Uses SSE4.1 _mm_dp_ps when available, otherwise shuffle+add
 */
MMATH_FORCE_INLINE float vec3ADot(Vec3A a_, Vec3A b_) noexcept {
#if defined(__SSE4_1__)
    // SSE4.1: dp_ps with mask 0x71 = xyz only, result in lowest lane
    __m128 dp = _mm_dp_ps(a_.data, b_.data, 0x71);
    return _mm_cvtss_f32(dp);
#elif defined(__SSE3__)
    // SSE3: mul + hadd + hadd
    __m128 mul = _mm_mul_ps(a_.data, b_.data);
    // Mask out w component
    mul = _mm_and_ps(mul, _mm_castsi128_ps(_mm_set_epi32(0, -1, -1, -1)));
    __m128 hadd1 = _mm_hadd_ps(mul, mul);
    __m128 hadd2 = _mm_hadd_ps(hadd1, hadd1);
    return _mm_cvtss_f32(hadd2);
#elif defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
    // SSE: mul + shuffle + add
    __m128 mul = _mm_mul_ps(a_.data, b_.data);
    // x + y
    __m128 shuf = _mm_shuffle_ps(mul, mul, _MM_SHUFFLE(1, 1, 1, 1));
    __m128 sum = _mm_add_ss(mul, shuf);
    // + z
    shuf = _mm_shuffle_ps(mul, mul, _MM_SHUFFLE(2, 2, 2, 2));
    sum = _mm_add_ss(sum, shuf);
    return _mm_cvtss_f32(sum);
#else
    return a_.x * b_.x + a_.y * b_.y + a_.z * b_.z;
#endif
}

/**
 * @brief Dot product (returns Vec3A with result broadcast to all lanes)
 */
MMATH_FORCE_INLINE Vec3A vec3ADotV(Vec3A a_, Vec3A b_) noexcept {
#if defined(__SSE4_1__)
    Vec3A result;
    result.data = _mm_dp_ps(a_.data, b_.data, 0x7F);
    return result;
#elif defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
    float d = vec3ADot(a_, b_);
    return vec3ASplat(d);
#else
    float d = a_.x * b_.x + a_.y * b_.y + a_.z * b_.z;
    return vec3ASplat(d);
#endif
}

/**
 * @brief Cross product
 */
MMATH_FORCE_INLINE Vec3A vec3ACross(Vec3A a_, Vec3A b_) noexcept {
#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
    // a = [ax, ay, az, 0]
    // b = [bx, by, bz, 0]
    // result = [ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx, 0]

    // Shuffle to get: [ay, az, ax, 0]
    __m128 a_yzx = _mm_shuffle_ps(a_.data, a_.data, _MM_SHUFFLE(3, 0, 2, 1));
    // Shuffle to get: [bz, bx, by, 0]
    __m128 b_zxy = _mm_shuffle_ps(b_.data, b_.data, _MM_SHUFFLE(3, 1, 0, 2));
    // Shuffle to get: [az, ax, ay, 0]
    __m128 a_zxy = _mm_shuffle_ps(a_.data, a_.data, _MM_SHUFFLE(3, 1, 0, 2));
    // Shuffle to get: [by, bz, bx, 0]
    __m128 b_yzx = _mm_shuffle_ps(b_.data, b_.data, _MM_SHUFFLE(3, 0, 2, 1));

    // (ay*bz, az*bx, ax*by, 0)
    __m128 mul1 = _mm_mul_ps(a_yzx, b_zxy);
    // (az*by, ax*bz, ay*bx, 0)
    __m128 mul2 = _mm_mul_ps(a_zxy, b_yzx);

    Vec3A result;
    result.data = _mm_sub_ps(mul1, mul2);
    return result;
#else
    return vec3ASet(
        a_.y * b_.z - a_.z * b_.y,
        a_.z * b_.x - a_.x * b_.z,
        a_.x * b_.y - a_.y * b_.x
    );
#endif
}

/**
 * @brief Length squared
 */
MMATH_FORCE_INLINE float vec3ALengthSquared(Vec3A v_) noexcept {
    return vec3ADot(v_, v_);
}

/**
 * @brief Length (magnitude)
 */
MMATH_FORCE_INLINE float vec3ALength(Vec3A v_) noexcept {
#if defined(__SSE4_1__)
    __m128 len_sq = _mm_dp_ps(v_.data, v_.data, 0x71);
    __m128 len = _mm_sqrt_ss(len_sq);
    return _mm_cvtss_f32(len);
#elif defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
    float len_sq = vec3ADot(v_, v_);
    return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(len_sq)));
#else
    return std::sqrt(vec3ALengthSquared(v_));
#endif
}

/**
 * @brief Normalize to unit length
 */
MMATH_FORCE_INLINE Vec3A vec3ANormalize(Vec3A v_) noexcept {
#if defined(__SSE4_1__)
    // Get length squared broadcast to all lanes
    __m128 len_sq = _mm_dp_ps(v_.data, v_.data, 0x7F);
    // sqrt
    __m128 len = _mm_sqrt_ps(len_sq);
    // divide
    Vec3A result;
    result.data = _mm_div_ps(v_.data, len);
    // Zero out w
    result._pad = 0.0f;
    return result;
#elif defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
    float len_sq = vec3ADot(v_, v_);
    float inv_len = 1.0f / std::sqrt(len_sq);
    return vec3AScale(v_, inv_len);
#else
    float len_sq = vec3ALengthSquared(v_);
    float inv_len = 1.0f / std::sqrt(len_sq);
    return vec3ASet(v_.x * inv_len, v_.y * inv_len, v_.z * inv_len);
#endif
}

/**
 * @brief Fast normalize using rsqrt approximation (~0.1% error)
 */
MMATH_FORCE_INLINE Vec3A vec3ANormalizeFast(Vec3A v_) noexcept {
#if defined(__SSE4_1__)
    __m128 len_sq = _mm_dp_ps(v_.data, v_.data, 0x7F);
    __m128 inv_len = _mm_rsqrt_ps(len_sq);
    Vec3A result;
    result.data = _mm_mul_ps(v_.data, inv_len);
    result._pad = 0.0f;
    return result;
#elif defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
    float len_sq = vec3ADot(v_, v_);
    __m128 inv_len = _mm_rsqrt_ss(_mm_set_ss(len_sq));
    float inv_len_f = _mm_cvtss_f32(inv_len);
    return vec3AScale(v_, inv_len_f);
#else
    float len_sq = vec3ALengthSquared(v_);
    float inv_len = 1.0f / std::sqrt(len_sq);
    return vec3ASet(v_.x * inv_len, v_.y * inv_len, v_.z * inv_len);
#endif
}

/**
 * @brief Normalize with Newton-Raphson refinement (higher precision than fast)
 */
MMATH_FORCE_INLINE Vec3A vec3ANormalizePrecise(Vec3A v_) noexcept {
#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
    float len_sq = vec3ADot(v_, v_);
    __m128 x = _mm_set_ss(len_sq);
    __m128 rsqrt = _mm_rsqrt_ss(x);

    // Newton-Raphson: rsqrt = rsqrt * (1.5 - 0.5 * x * rsqrt * rsqrt)
    __m128 half = _mm_set_ss(0.5f);
    __m128 three_half = _mm_set_ss(1.5f);
    __m128 rsqrt_sq = _mm_mul_ss(rsqrt, rsqrt);
    __m128 x_half = _mm_mul_ss(x, half);
    __m128 tmp = _mm_mul_ss(x_half, rsqrt_sq);
    tmp = _mm_sub_ss(three_half, tmp);
    rsqrt = _mm_mul_ss(rsqrt, tmp);

    float inv_len = _mm_cvtss_f32(rsqrt);
    return vec3AScale(v_, inv_len);
#else
    return vec3ANormalize(v_);
#endif
}

// ============================================================================
// Vec3A Distance and Interpolation
// ============================================================================

/**
 * @brief Distance squared between two points
 */
MMATH_FORCE_INLINE float vec3ADistanceSquared(Vec3A a_, Vec3A b_) noexcept {
    Vec3A diff = vec3ASub(a_, b_);
    return vec3ADot(diff, diff);
}

/**
 * @brief Distance between two points
 */
MMATH_FORCE_INLINE float vec3ADistance(Vec3A a_, Vec3A b_) noexcept {
#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
    float dist_sq = vec3ADistanceSquared(a_, b_);
    return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(dist_sq)));
#else
    return std::sqrt(vec3ADistanceSquared(a_, b_));
#endif
}

/**
 * @brief Linear interpolation
 */
MMATH_FORCE_INLINE Vec3A vec3ALerp(Vec3A a_, Vec3A b_, float t_) noexcept {
#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
    __m128 t = _mm_set1_ps(t_);
    __m128 diff = _mm_sub_ps(b_.data, a_.data);
    Vec3A result;
    result.data = _mm_add_ps(a_.data, _mm_mul_ps(diff, t));
    result._pad = 0.0f;
    return result;
#else
    float omt = 1.0f - t_;
    return vec3ASet(
        a_.x * omt + b_.x * t_,
        a_.y * omt + b_.y * t_,
        a_.z * omt + b_.z * t_
    );
#endif
}

/**
 * @brief Component-wise minimum
 */
MMATH_FORCE_INLINE Vec3A vec3AMin(Vec3A a_, Vec3A b_) noexcept {
#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
    Vec3A result;
    result.data = _mm_min_ps(a_.data, b_.data);
    return result;
#else
    return vec3ASet(
        a_.x < b_.x ? a_.x : b_.x,
        a_.y < b_.y ? a_.y : b_.y,
        a_.z < b_.z ? a_.z : b_.z
    );
#endif
}

/**
 * @brief Component-wise maximum
 */
MMATH_FORCE_INLINE Vec3A vec3AMax(Vec3A a_, Vec3A b_) noexcept {
#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
    Vec3A result;
    result.data = _mm_max_ps(a_.data, b_.data);
    return result;
#else
    return vec3ASet(
        a_.x > b_.x ? a_.x : b_.x,
        a_.y > b_.y ? a_.y : b_.y,
        a_.z > b_.z ? a_.z : b_.z
    );
#endif
}

/**
 * @brief Clamp components to range
 */
MMATH_FORCE_INLINE Vec3A vec3AClamp(Vec3A v_, Vec3A min_, Vec3A max_) noexcept {
    return vec3AMax(min_, vec3AMin(v_, max_));
}

/**
 * @brief Reflect vector around normal
 */
MMATH_FORCE_INLINE Vec3A vec3AReflect(Vec3A v_, Vec3A n_) noexcept {
    // v - 2 * dot(v, n) * n
    float d = vec3ADot(v_, n_);
    float d2 = d + d;
    return vec3ASub(v_, vec3AScale(n_, d2));
}

