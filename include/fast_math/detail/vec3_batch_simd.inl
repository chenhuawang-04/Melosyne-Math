// Shared Vec3 batch SIMD implementation for header/module parity.

// ============================================================================
// Batch Processing - Optimized SIMD for Vec3 arrays
// ============================================================================

// SSE version (processes 4 Vec3s at once) - Most efficient for 3-channel data
#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)

namespace detail {

/**
 * @brief Deinterleave 4 Vec3s from AOS to SOA using pure SSE shuffles
 *
 * Input (AOS):  [x0,y0,z0,x1, y1,z1,x2,y2, z2,x3,y3,z3] (12 floats)
 * Output (SOA): [x0,x1,x2,x3], [y0,y1,y2,y3], [z0,z1,z2,z3]
 *
 * Uses only 3 loads + 9 shuffles - optimal for 3-channel deinterleave
 */
MMATH_FORCE_INLINE void deinterleave4Vec3(
    const Vec3* MMATH_RESTRICT input_,
    __m128& out_x_,
    __m128& out_y_,
    __m128& out_z_
) noexcept {
    const float* ptr = reinterpret_cast<const float*>(input_);

    // Load 12 floats as 3 x __m128
    __m128 v0 = _mm_loadu_ps(ptr + 0);  // [x0,y0,z0,x1]
    __m128 v1 = _mm_loadu_ps(ptr + 4);  // [y1,z1,x2,y2]
    __m128 v2 = _mm_loadu_ps(ptr + 8);  // [z2,x3,y3,z3]

    // Extract x = [x0,x1,x2,x3]
    // x0=v0[0], x1=v0[3], x2=v1[2], x3=v2[1]
    __m128 t0 = _mm_shuffle_ps(v0, v0, _MM_SHUFFLE(3, 3, 0, 0));  // [x0,x0,x1,x1]
    __m128 t1 = _mm_shuffle_ps(v1, v2, _MM_SHUFFLE(1, 1, 2, 2));  // [x2,x2,x3,x3]
    out_x_ = _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(2, 0, 2, 0));     // [x0,x1,x2,x3]

    // Extract y = [y0,y1,y2,y3]
    // y0=v0[1], y1=v1[0], y2=v1[3], y3=v2[2]
    t0 = _mm_shuffle_ps(v0, v1, _MM_SHUFFLE(0, 0, 1, 1));  // [y0,y0,y1,y1]
    t1 = _mm_shuffle_ps(v1, v2, _MM_SHUFFLE(2, 2, 3, 3));  // [y2,y2,y3,y3]
    out_y_ = _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(2, 0, 2, 0));  // [y0,y1,y2,y3]

    // Extract z = [z0,z1,z2,z3]
    // z0=v0[2], z1=v1[1], z2=v2[0], z3=v2[3]
    t0 = _mm_shuffle_ps(v0, v1, _MM_SHUFFLE(1, 1, 2, 2));  // [z0,z0,z1,z1]
    t1 = _mm_shuffle_ps(v2, v2, _MM_SHUFFLE(3, 3, 0, 0));  // [z2,z2,z3,z3]
    out_z_ = _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(2, 0, 2, 0));  // [z0,z1,z2,z3]
}

/**
 * @brief Interleave 4 Vec3s from SOA to AOS using pure SSE shuffles
 *
 * Input (SOA): [x0,x1,x2,x3], [y0,y1,y2,y3], [z0,z1,z2,z3]
 * Output (AOS): [x0,y0,z0,x1, y1,z1,x2,y2, z2,x3,y3,z3]
 *
 * Uses only 9 shuffles + 3 stores
 */
MMATH_FORCE_INLINE void interleave4Vec3(
    __m128 x_,
    __m128 y_,
    __m128 z_,
    Vec3* MMATH_RESTRICT output_
) noexcept {
    float* ptr = reinterpret_cast<float*>(output_);

    // Interleave x and y
    __m128 xy_lo = _mm_unpacklo_ps(x_, y_);  // [x0,y0,x1,y1]
    __m128 xy_hi = _mm_unpackhi_ps(x_, y_);  // [x2,y2,x3,y3]

    // v0 = [x0,y0,z0,x1]
    __m128 t0 = _mm_shuffle_ps(z_, xy_lo, _MM_SHUFFLE(2, 2, 0, 0));  // [z0,z0,x1,x1]
    __m128 v0 = _mm_shuffle_ps(xy_lo, t0, _MM_SHUFFLE(2, 0, 1, 0));  // [x0,y0,z0,x1]

    // v1 = [y1,z1,x2,y2]
    __m128 t1 = _mm_shuffle_ps(xy_lo, z_, _MM_SHUFFLE(1, 1, 3, 3));  // [y1,y1,z1,z1]
    __m128 v1 = _mm_shuffle_ps(t1, xy_hi, _MM_SHUFFLE(1, 0, 2, 0));  // [y1,z1,x2,y2]

    // v2 = [z2,x3,y3,z3]
    __m128 t2 = _mm_shuffle_ps(z_, xy_hi, _MM_SHUFFLE(3, 2, 3, 2));  // [z2,z3,x3,y3]
    __m128 v2 = _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(1, 3, 2, 0));     // [z2,x3,y3,z3]

    _mm_storeu_ps(ptr + 0, v0);
    _mm_storeu_ps(ptr + 4, v1);
    _mm_storeu_ps(ptr + 8, v2);
}

} // namespace detail

#endif // SSE

// AVX2 version (processes 8 Vec3s using 2x SSE)
#if defined(__AVX2__) && defined(__FMA__)

namespace detail {

/**
 * @brief Deinterleave 8 Vec3s from AOS to SOA
 *
 * Uses 2x SSE deinterleave - more efficient than gather on most CPUs
 */
MMATH_FORCE_INLINE void deinterleave8Vec3(
    const Vec3* MMATH_RESTRICT input_,
    __m256& out_x_,
    __m256& out_y_,
    __m256& out_z_
) noexcept {
    __m128 x_lo, y_lo, z_lo, x_hi, y_hi, z_hi;

    // Process first 4 Vec3s
    deinterleave4Vec3(input_, x_lo, y_lo, z_lo);
    // Process second 4 Vec3s
    deinterleave4Vec3(input_ + 4, x_hi, y_hi, z_hi);

    // Combine into 256-bit registers
    out_x_ = _mm256_set_m128(x_hi, x_lo);
    out_y_ = _mm256_set_m128(y_hi, y_lo);
    out_z_ = _mm256_set_m128(z_hi, z_lo);
}

/**
 * @brief Interleave 8 Vec3s from SOA to AOS
 */
MMATH_FORCE_INLINE void interleave8Vec3(
    __m256 x_,
    __m256 y_,
    __m256 z_,
    Vec3* MMATH_RESTRICT output_
) noexcept {
    // Extract 128-bit halves
    __m128 x_lo = _mm256_castps256_ps128(x_);
    __m128 x_hi = _mm256_extractf128_ps(x_, 1);
    __m128 y_lo = _mm256_castps256_ps128(y_);
    __m128 y_hi = _mm256_extractf128_ps(y_, 1);
    __m128 z_lo = _mm256_castps256_ps128(z_);
    __m128 z_hi = _mm256_extractf128_ps(z_, 1);

    // Interleave using SSE version
    interleave4Vec3(x_lo, y_lo, z_lo, output_);
    interleave4Vec3(x_hi, y_hi, z_hi, output_ + 4);
}

} // namespace detail

/**
 * @brief Batch Vec3 add with AVX2
 */
inline void vec3AddBatchSimd(
    const Vec3* MMATH_RESTRICT a_,
    const Vec3* MMATH_RESTRICT b_,
    Vec3* MMATH_RESTRICT result_,
    int32_t count_
) noexcept {
    int32_t i = 0;
    const int32_t simd_count = (count_ / 8) * 8;

    for (; i < simd_count; i += 8) {
        __m256 ax, ay, az, bx, by, bz;
        detail::deinterleave8Vec3(a_ + i, ax, ay, az);
        detail::deinterleave8Vec3(b_ + i, bx, by, bz);

        __m256 rx = _mm256_add_ps(ax, bx);
        __m256 ry = _mm256_add_ps(ay, by);
        __m256 rz = _mm256_add_ps(az, bz);

        detail::interleave8Vec3(rx, ry, rz, result_ + i);
    }

    // Handle remainder with SSE (4 at a time)
    const int32_t sse_count = ((count_ - i) / 4) * 4 + i;
    for (; i < sse_count; i += 4) {
        __m128 ax, ay, az, bx, by, bz;
        detail::deinterleave4Vec3(a_ + i, ax, ay, az);
        detail::deinterleave4Vec3(b_ + i, bx, by, bz);

        __m128 rx = _mm_add_ps(ax, bx);
        __m128 ry = _mm_add_ps(ay, by);
        __m128 rz = _mm_add_ps(az, bz);

        detail::interleave4Vec3(rx, ry, rz, result_ + i);
    }

    // Handle final remainder
    for (; i < count_; ++i) {
        result_[i] = vec3Add(a_[i], b_[i]);
    }
}

/**
 * @brief Batch Vec3 dot product with AVX2
 */
inline void vec3DotBatchSimd(
    const Vec3* MMATH_RESTRICT a_,
    const Vec3* MMATH_RESTRICT b_,
    float* MMATH_RESTRICT result_,
    int32_t count_
) noexcept {
    int32_t i = 0;
    const int32_t simd_count = (count_ / 8) * 8;

    for (; i < simd_count; i += 8) {
        __m256 ax, ay, az, bx, by, bz;
        detail::deinterleave8Vec3(a_ + i, ax, ay, az);
        detail::deinterleave8Vec3(b_ + i, bx, by, bz);

        // dot = ax*bx + ay*by + az*bz
        __m256 dot = _mm256_fmadd_ps(ax, bx,
                     _mm256_fmadd_ps(ay, by,
                     _mm256_mul_ps(az, bz)));

        _mm256_storeu_ps(result_ + i, dot);
    }

    // Handle remainder with SSE
    const int32_t sse_count = ((count_ - i) / 4) * 4 + i;
    for (; i < sse_count; i += 4) {
        __m128 ax, ay, az, bx, by, bz;
        detail::deinterleave4Vec3(a_ + i, ax, ay, az);
        detail::deinterleave4Vec3(b_ + i, bx, by, bz);

        // dot = ax*bx + ay*by + az*bz (no FMA in SSE-only)
        __m128 dot = _mm_add_ps(_mm_add_ps(_mm_mul_ps(ax, bx), _mm_mul_ps(ay, by)),
                                _mm_mul_ps(az, bz));
        _mm_storeu_ps(result_ + i, dot);
    }

    // Handle final remainder
    for (; i < count_; ++i) {
        result_[i] = vec3Dot(a_[i], b_[i]);
    }
}

/**
 * @brief Batch Vec3 cross product with AVX2
 */
inline void vec3CrossBatchSimd(
    const Vec3* MMATH_RESTRICT a_,
    const Vec3* MMATH_RESTRICT b_,
    Vec3* MMATH_RESTRICT result_,
    int32_t count_
) noexcept {
    int32_t i = 0;
    const int32_t simd_count = (count_ / 8) * 8;

    for (; i < simd_count; i += 8) {
        __m256 ax, ay, az, bx, by, bz;
        detail::deinterleave8Vec3(a_ + i, ax, ay, az);
        detail::deinterleave8Vec3(b_ + i, bx, by, bz);

        // cross.x = ay*bz - az*by
        // cross.y = az*bx - ax*bz
        // cross.z = ax*by - ay*bx
        __m256 rx = _mm256_fmsub_ps(ay, bz, _mm256_mul_ps(az, by));
        __m256 ry = _mm256_fmsub_ps(az, bx, _mm256_mul_ps(ax, bz));
        __m256 rz = _mm256_fmsub_ps(ax, by, _mm256_mul_ps(ay, bx));

        detail::interleave8Vec3(rx, ry, rz, result_ + i);
    }

    // Handle remainder with SSE
    const int32_t sse_count = ((count_ - i) / 4) * 4 + i;
    for (; i < sse_count; i += 4) {
        __m128 ax, ay, az, bx, by, bz;
        detail::deinterleave4Vec3(a_ + i, ax, ay, az);
        detail::deinterleave4Vec3(b_ + i, bx, by, bz);

        __m128 rx = _mm_sub_ps(_mm_mul_ps(ay, bz), _mm_mul_ps(az, by));
        __m128 ry = _mm_sub_ps(_mm_mul_ps(az, bx), _mm_mul_ps(ax, bz));
        __m128 rz = _mm_sub_ps(_mm_mul_ps(ax, by), _mm_mul_ps(ay, bx));

        detail::interleave4Vec3(rx, ry, rz, result_ + i);
    }

    // Handle final remainder
    for (; i < count_; ++i) {
        result_[i] = vec3Cross(a_[i], b_[i]);
    }
}

/**
 * @brief Batch Vec3 normalize with AVX2
 */
inline void vec3NormalizeBatchSimd(
    const Vec3* MMATH_RESTRICT input_,
    Vec3* MMATH_RESTRICT result_,
    int32_t count_
) noexcept {
    int32_t i = 0;
    const int32_t simd_count = (count_ / 8) * 8;
    const __m256 epsilon = _mm256_set1_ps(1e-12f);

    for (; i < simd_count; i += 8) {
        __m256 vx, vy, vz;
        detail::deinterleave8Vec3(input_ + i, vx, vy, vz);

        // len_sq = x*x + y*y + z*z
        __m256 len_sq = _mm256_fmadd_ps(vx, vx,
                        _mm256_fmadd_ps(vy, vy,
                        _mm256_mul_ps(vz, vz)));

        // inv_len = 1 / sqrt(len_sq) with Newton-Raphson
        __m256 inv_len = _mm256_rsqrt_ps(len_sq);

        // Newton-Raphson refinement
        __m256 half = _mm256_set1_ps(0.5f);
        __m256 three_half = _mm256_set1_ps(1.5f);
        __m256 inv_len_sq = _mm256_mul_ps(inv_len, inv_len);
        __m256 temp = _mm256_mul_ps(half, _mm256_mul_ps(len_sq, inv_len_sq));
        inv_len = _mm256_mul_ps(inv_len, _mm256_sub_ps(three_half, temp));

        // Clamp for zero vectors
        __m256 is_valid = _mm256_cmp_ps(len_sq, epsilon, _CMP_GT_OQ);
        inv_len = _mm256_and_ps(inv_len, is_valid);

        // result = v * inv_len
        __m256 rx = _mm256_mul_ps(vx, inv_len);
        __m256 ry = _mm256_mul_ps(vy, inv_len);
        __m256 rz = _mm256_mul_ps(vz, inv_len);

        detail::interleave8Vec3(rx, ry, rz, result_ + i);
    }

    // Handle remainder with SSE
    const int32_t sse_count = ((count_ - i) / 4) * 4 + i;
    const __m128 eps4 = _mm_set1_ps(1e-12f);
    for (; i < sse_count; i += 4) {
        __m128 vx, vy, vz;
        detail::deinterleave4Vec3(input_ + i, vx, vy, vz);

        __m128 len_sq = _mm_add_ps(_mm_add_ps(_mm_mul_ps(vx, vx), _mm_mul_ps(vy, vy)),
                                   _mm_mul_ps(vz, vz));

        __m128 inv_len = _mm_rsqrt_ps(len_sq);

        // Newton-Raphson refinement
        __m128 half4 = _mm_set1_ps(0.5f);
        __m128 three_half4 = _mm_set1_ps(1.5f);
        __m128 inv_len_sq4 = _mm_mul_ps(inv_len, inv_len);
        __m128 temp4 = _mm_mul_ps(half4, _mm_mul_ps(len_sq, inv_len_sq4));
        inv_len = _mm_mul_ps(inv_len, _mm_sub_ps(three_half4, temp4));

        // Clamp for zero vectors
        __m128 is_valid4 = _mm_cmpgt_ps(len_sq, eps4);
        inv_len = _mm_and_ps(inv_len, is_valid4);

        __m128 rx = _mm_mul_ps(vx, inv_len);
        __m128 ry = _mm_mul_ps(vy, inv_len);
        __m128 rz = _mm_mul_ps(vz, inv_len);

        detail::interleave4Vec3(rx, ry, rz, result_ + i);
    }

    // Handle final remainder
    for (; i < count_; ++i) {
        result_[i] = vec3Normalize(input_[i]);
    }
}

/**
 * @brief Batch Vec3 length with AVX2
 */
inline void vec3LengthBatchSimd(
    const Vec3* MMATH_RESTRICT input_,
    float* MMATH_RESTRICT result_,
    int32_t count_
) noexcept {
    int32_t i = 0;
    const int32_t simd_count = (count_ / 8) * 8;

    for (; i < simd_count; i += 8) {
        __m256 vx, vy, vz;
        detail::deinterleave8Vec3(input_ + i, vx, vy, vz);

        // len_sq = x*x + y*y + z*z
        __m256 len_sq = _mm256_fmadd_ps(vx, vx,
                        _mm256_fmadd_ps(vy, vy,
                        _mm256_mul_ps(vz, vz)));

        __m256 len = _mm256_sqrt_ps(len_sq);
        _mm256_storeu_ps(result_ + i, len);
    }

    // Handle remainder with SSE
    const int32_t sse_count = ((count_ - i) / 4) * 4 + i;
    for (; i < sse_count; i += 4) {
        __m128 vx, vy, vz;
        detail::deinterleave4Vec3(input_ + i, vx, vy, vz);

        __m128 len_sq = _mm_add_ps(_mm_add_ps(_mm_mul_ps(vx, vx), _mm_mul_ps(vy, vy)),
                                   _mm_mul_ps(vz, vz));
        __m128 len = _mm_sqrt_ps(len_sq);
        _mm_storeu_ps(result_ + i, len);
    }

    // Handle final remainder
    for (; i < count_; ++i) {
        result_[i] = vec3Length(input_[i]);
    }
}

/**
 * @brief Batch Vec3 scale with AVX2
 */
inline void vec3ScaleBatchSimd(
    const Vec3* MMATH_RESTRICT input_,
    float scalar_,
    Vec3* MMATH_RESTRICT result_,
    int32_t count_
) noexcept {
    int32_t i = 0;
    const int32_t simd_count = (count_ / 8) * 8;
    const __m256 s = _mm256_set1_ps(scalar_);

    for (; i < simd_count; i += 8) {
        __m256 vx, vy, vz;
        detail::deinterleave8Vec3(input_ + i, vx, vy, vz);

        __m256 rx = _mm256_mul_ps(vx, s);
        __m256 ry = _mm256_mul_ps(vy, s);
        __m256 rz = _mm256_mul_ps(vz, s);

        detail::interleave8Vec3(rx, ry, rz, result_ + i);
    }

    // Handle remainder with SSE
    const int32_t sse_count = ((count_ - i) / 4) * 4 + i;
    const __m128 s4 = _mm_set1_ps(scalar_);
    for (; i < sse_count; i += 4) {
        __m128 vx, vy, vz;
        detail::deinterleave4Vec3(input_ + i, vx, vy, vz);

        __m128 rx = _mm_mul_ps(vx, s4);
        __m128 ry = _mm_mul_ps(vy, s4);
        __m128 rz = _mm_mul_ps(vz, s4);

        detail::interleave4Vec3(rx, ry, rz, result_ + i);
    }

    // Handle final remainder
    for (; i < count_; ++i) {
        result_[i] = vec3Scale(input_[i], scalar_);
    }
}

#elif defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)

// SSE-only batch operations (processes 4 Vec3s at a time)

inline void vec3AddBatchSimd(
    const Vec3* MMATH_RESTRICT a_,
    const Vec3* MMATH_RESTRICT b_,
    Vec3* MMATH_RESTRICT result_,
    int32_t count_
) noexcept {
    int32_t i = 0;
    const int32_t simd_count = (count_ / 4) * 4;

    for (; i < simd_count; i += 4) {
        __m128 ax, ay, az, bx, by, bz;
        detail::deinterleave4Vec3(a_ + i, ax, ay, az);
        detail::deinterleave4Vec3(b_ + i, bx, by, bz);

        __m128 rx = _mm_add_ps(ax, bx);
        __m128 ry = _mm_add_ps(ay, by);
        __m128 rz = _mm_add_ps(az, bz);

        detail::interleave4Vec3(rx, ry, rz, result_ + i);
    }

    for (; i < count_; ++i) {
        result_[i] = vec3Add(a_[i], b_[i]);
    }
}

inline void vec3DotBatchSimd(
    const Vec3* MMATH_RESTRICT a_,
    const Vec3* MMATH_RESTRICT b_,
    float* MMATH_RESTRICT result_,
    int32_t count_
) noexcept {
    int32_t i = 0;
    const int32_t simd_count = (count_ / 4) * 4;

    for (; i < simd_count; i += 4) {
        __m128 ax, ay, az, bx, by, bz;
        detail::deinterleave4Vec3(a_ + i, ax, ay, az);
        detail::deinterleave4Vec3(b_ + i, bx, by, bz);

        __m128 dot = _mm_add_ps(_mm_add_ps(_mm_mul_ps(ax, bx), _mm_mul_ps(ay, by)),
                                _mm_mul_ps(az, bz));
        _mm_storeu_ps(result_ + i, dot);
    }

    for (; i < count_; ++i) {
        result_[i] = vec3Dot(a_[i], b_[i]);
    }
}

inline void vec3CrossBatchSimd(
    const Vec3* MMATH_RESTRICT a_,
    const Vec3* MMATH_RESTRICT b_,
    Vec3* MMATH_RESTRICT result_,
    int32_t count_
) noexcept {
    int32_t i = 0;
    const int32_t simd_count = (count_ / 4) * 4;

    for (; i < simd_count; i += 4) {
        __m128 ax, ay, az, bx, by, bz;
        detail::deinterleave4Vec3(a_ + i, ax, ay, az);
        detail::deinterleave4Vec3(b_ + i, bx, by, bz);

        __m128 rx = _mm_sub_ps(_mm_mul_ps(ay, bz), _mm_mul_ps(az, by));
        __m128 ry = _mm_sub_ps(_mm_mul_ps(az, bx), _mm_mul_ps(ax, bz));
        __m128 rz = _mm_sub_ps(_mm_mul_ps(ax, by), _mm_mul_ps(ay, bx));

        detail::interleave4Vec3(rx, ry, rz, result_ + i);
    }

    for (; i < count_; ++i) {
        result_[i] = vec3Cross(a_[i], b_[i]);
    }
}

inline void vec3NormalizeBatchSimd(
    const Vec3* MMATH_RESTRICT input_,
    Vec3* MMATH_RESTRICT result_,
    int32_t count_
) noexcept {
    int32_t i = 0;
    const int32_t simd_count = (count_ / 4) * 4;
    const __m128 epsilon = _mm_set1_ps(1e-12f);

    for (; i < simd_count; i += 4) {
        __m128 vx, vy, vz;
        detail::deinterleave4Vec3(input_ + i, vx, vy, vz);

        __m128 len_sq = _mm_add_ps(_mm_add_ps(_mm_mul_ps(vx, vx), _mm_mul_ps(vy, vy)),
                                   _mm_mul_ps(vz, vz));

        __m128 inv_len = _mm_rsqrt_ps(len_sq);

        // Newton-Raphson refinement
        __m128 half = _mm_set1_ps(0.5f);
        __m128 three_half = _mm_set1_ps(1.5f);
        __m128 inv_len_sq = _mm_mul_ps(inv_len, inv_len);
        __m128 temp = _mm_mul_ps(half, _mm_mul_ps(len_sq, inv_len_sq));
        inv_len = _mm_mul_ps(inv_len, _mm_sub_ps(three_half, temp));

        // Clamp for zero vectors
        __m128 is_valid = _mm_cmpgt_ps(len_sq, epsilon);
        inv_len = _mm_and_ps(inv_len, is_valid);

        __m128 rx = _mm_mul_ps(vx, inv_len);
        __m128 ry = _mm_mul_ps(vy, inv_len);
        __m128 rz = _mm_mul_ps(vz, inv_len);

        detail::interleave4Vec3(rx, ry, rz, result_ + i);
    }

    for (; i < count_; ++i) {
        result_[i] = vec3Normalize(input_[i]);
    }
}

inline void vec3LengthBatchSimd(
    const Vec3* MMATH_RESTRICT input_,
    float* MMATH_RESTRICT result_,
    int32_t count_
) noexcept {
    int32_t i = 0;
    const int32_t simd_count = (count_ / 4) * 4;

    for (; i < simd_count; i += 4) {
        __m128 vx, vy, vz;
        detail::deinterleave4Vec3(input_ + i, vx, vy, vz);

        __m128 len_sq = _mm_add_ps(_mm_add_ps(_mm_mul_ps(vx, vx), _mm_mul_ps(vy, vy)),
                                   _mm_mul_ps(vz, vz));
        __m128 len = _mm_sqrt_ps(len_sq);
        _mm_storeu_ps(result_ + i, len);
    }

    for (; i < count_; ++i) {
        result_[i] = vec3Length(input_[i]);
    }
}

inline void vec3ScaleBatchSimd(
    const Vec3* MMATH_RESTRICT input_,
    float scalar_,
    Vec3* MMATH_RESTRICT result_,
    int32_t count_
) noexcept {
    int32_t i = 0;
    const int32_t simd_count = (count_ / 4) * 4;
    const __m128 s = _mm_set1_ps(scalar_);

    for (; i < simd_count; i += 4) {
        __m128 vx, vy, vz;
        detail::deinterleave4Vec3(input_ + i, vx, vy, vz);

        __m128 rx = _mm_mul_ps(vx, s);
        __m128 ry = _mm_mul_ps(vy, s);
        __m128 rz = _mm_mul_ps(vz, s);

        detail::interleave4Vec3(rx, ry, rz, result_ + i);
    }

    for (; i < count_; ++i) {
        result_[i] = vec3Scale(input_[i], scalar_);
    }
}

#else // Fallback scalar batch operations

inline void vec3AddBatchSimd(
    const Vec3* MMATH_RESTRICT a_,
    const Vec3* MMATH_RESTRICT b_,
    Vec3* MMATH_RESTRICT result_,
    int32_t count_
) noexcept {
    for (int32_t i = 0; i < count_; ++i) {
        result_[i] = vec3Add(a_[i], b_[i]);
    }
}

inline void vec3DotBatchSimd(
    const Vec3* MMATH_RESTRICT a_,
    const Vec3* MMATH_RESTRICT b_,
    float* MMATH_RESTRICT result_,
    int32_t count_
) noexcept {
    for (int32_t i = 0; i < count_; ++i) {
        result_[i] = vec3Dot(a_[i], b_[i]);
    }
}

inline void vec3CrossBatchSimd(
    const Vec3* MMATH_RESTRICT a_,
    const Vec3* MMATH_RESTRICT b_,
    Vec3* MMATH_RESTRICT result_,
    int32_t count_
) noexcept {
    for (int32_t i = 0; i < count_; ++i) {
        result_[i] = vec3Cross(a_[i], b_[i]);
    }
}

inline void vec3NormalizeBatchSimd(
    const Vec3* MMATH_RESTRICT input_,
    Vec3* MMATH_RESTRICT result_,
    int32_t count_
) noexcept {
    for (int32_t i = 0; i < count_; ++i) {
        result_[i] = vec3Normalize(input_[i]);
    }
}

inline void vec3LengthBatchSimd(
    const Vec3* MMATH_RESTRICT input_,
    float* MMATH_RESTRICT result_,
    int32_t count_
) noexcept {
    for (int32_t i = 0; i < count_; ++i) {
        result_[i] = vec3Length(input_[i]);
    }
}

inline void vec3ScaleBatchSimd(
    const Vec3* MMATH_RESTRICT input_,
    float scalar_,
    Vec3* MMATH_RESTRICT result_,
    int32_t count_
) noexcept {
    for (int32_t i = 0; i < count_; ++i) {
        result_[i] = vec3Scale(input_[i], scalar_);
    }
}

#endif // AVX2

