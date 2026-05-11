// Shared common SIMD batch/detail implementation for header/module parity.

namespace detail {

MMATH_FORCE_INLINE float minScalarFallback(float a_, float b_) noexcept {
    return (a_ < b_) ? a_ : b_;
}

MMATH_FORCE_INLINE float maxScalarFallback(float a_, float b_) noexcept {
    return (a_ > b_) ? a_ : b_;
}

MMATH_FORCE_INLINE float clampScalarFallback(float x_, float min_val_, float max_val_) noexcept {
    float temp = (x_ < max_val_) ? x_ : max_val_;
    return (temp > min_val_) ? temp : min_val_;
}

MMATH_FORCE_INLINE float absScalarFallback(float x_) noexcept {
    return std::fabs(x_);
}

MMATH_FORCE_INLINE float saturateScalarFallback(float x_) noexcept {
    return clampScalarFallback(x_, 0.0f, 1.0f);
}

// ============================================================================
// SIMD Implementation Functions (Internal)
// ============================================================================

/**
 * @brief SIMD min for arrays (internal implementation)
 */
inline void minSimd(
    float* MMATH_RESTRICT a_,
    const float* MMATH_RESTRICT b_,
    int32_t count_
) noexcept {
    int32_t i = 0;

#if defined(__AVX2__)
    // AVX2 path: Process 8 floats at a time
    const int32_t avx2_count = (count_ / 8) * 8;
    for (; i < avx2_count; i += 8) {
        __m256 va = _mm256_loadu_ps(a_ + i);
        __m256 vb = _mm256_loadu_ps(b_ + i);
        __m256 result = _mm256_min_ps(va, vb);
        _mm256_storeu_ps(a_ + i, result);
    }
#endif

#if defined(__SSE4_1__)
    // SSE4.1 path: Process 4 floats at a time
    const int32_t sse_count = (count_ / 4) * 4;
    for (; i < sse_count; i += 4) {
        __m128 va = _mm_loadu_ps(a_ + i);
        __m128 vb = _mm_loadu_ps(b_ + i);
        __m128 result = _mm_min_ps(va, vb);
        _mm_storeu_ps(a_ + i, result);
    }
#endif

    // Scalar fallback: Process remaining elements
    for (; i < count_; ++i) {
        a_[i] = minScalarFallback(a_[i], b_[i]);
    }
}

/**
 * @brief SIMD max for arrays (internal implementation)
 */
inline void maxSimd(
    float* MMATH_RESTRICT a_,
    const float* MMATH_RESTRICT b_,
    int32_t count_
) noexcept {
    int32_t i = 0;

#if defined(__AVX2__)
    const int32_t avx2_count = (count_ / 8) * 8;
    for (; i < avx2_count; i += 8) {
        __m256 va = _mm256_loadu_ps(a_ + i);
        __m256 vb = _mm256_loadu_ps(b_ + i);
        __m256 result = _mm256_max_ps(va, vb);
        _mm256_storeu_ps(a_ + i, result);
    }
#endif

#if defined(__SSE4_1__)
    const int32_t sse_count = (count_ / 4) * 4;
    for (; i < sse_count; i += 4) {
        __m128 va = _mm_loadu_ps(a_ + i);
        __m128 vb = _mm_loadu_ps(b_ + i);
        __m128 result = _mm_max_ps(va, vb);
        _mm_storeu_ps(a_ + i, result);
    }
#endif

    for (; i < count_; ++i) {
        a_[i] = maxScalarFallback(a_[i], b_[i]);
    }
}

/**
 * @brief SIMD clamp for arrays (internal implementation)
 *
 * Implementation: Chained min/max (GLM style)
 * Formula: min(max(x, min_val), max_val)
 */
inline void clampSimd(
    float* MMATH_RESTRICT values_,
    float min_val_,
    float max_val_,
    int32_t count_
) noexcept {
    int32_t i = 0;

#if defined(__AVX2__)
    // Broadcast min/max to all lanes
    __m256 vmin = _mm256_set1_ps(min_val_);
    __m256 vmax = _mm256_set1_ps(max_val_);

    const int32_t avx2_count = (count_ / 8) * 8;
    for (; i < avx2_count; i += 8) {
        __m256 v = _mm256_loadu_ps(values_ + i);
        v = _mm256_max_ps(v, vmin);  // v = max(v, min)
        v = _mm256_min_ps(v, vmax);  // v = min(v, max)
        _mm256_storeu_ps(values_ + i, v);
    }
#endif

#if defined(__SSE4_1__)
    __m128 vmin_sse = _mm_set1_ps(min_val_);
    __m128 vmax_sse = _mm_set1_ps(max_val_);

    const int32_t sse_count = (count_ / 4) * 4;
    for (; i < sse_count; i += 4) {
        __m128 v = _mm_loadu_ps(values_ + i);
        v = _mm_max_ps(v, vmin_sse);
        v = _mm_min_ps(v, vmax_sse);
        _mm_storeu_ps(values_ + i, v);
    }
#endif

    for (; i < count_; ++i) {
        values_[i] = clampScalarFallback(values_[i], min_val_, max_val_);
    }
}

/**
 * @brief SIMD abs for arrays (internal implementation)
 *
 * Implementation: AND with 0x7FFFFFFF mask to clear sign bit
 * - AVX2: Process 8 floats with vandps
 * - SSE4.1: Process 4 floats with andps
 */
inline void absSimd(
    float* MMATH_RESTRICT values_,
    int32_t count_
) noexcept {
    int32_t i = 0;

#if defined(__AVX2__)
    // Create mask: 0x7FFFFFFF for all 8 lanes
    __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

    const int32_t avx2_count = (count_ / 8) * 8;
    for (; i < avx2_count; i += 8) {
        __m256 v = _mm256_loadu_ps(values_ + i);
        v = _mm256_and_ps(v, mask);  // Clear sign bit
        _mm256_storeu_ps(values_ + i, v);
    }
#endif

#if defined(__SSE4_1__)
    __m128 mask_sse = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));

    const int32_t sse_count = (count_ / 4) * 4;
    for (; i < sse_count; i += 4) {
        __m128 v = _mm_loadu_ps(values_ + i);
        v = _mm_and_ps(v, mask_sse);
        _mm_storeu_ps(values_ + i, v);
    }
#endif

    for (; i < count_; ++i) {
        values_[i] = absScalarFallback(values_[i]);
    }
}

/**
 * @brief SIMD saturate for arrays (internal implementation)
 *
 * Optimized version of clamp(x, 0, 1)
 * Uses constant 0.0f and 1.0f
 */
inline void saturateSimd(
    float* MMATH_RESTRICT values_,
    int32_t count_
) noexcept {
    int32_t i = 0;

#if defined(__AVX2__)
    __m256 vzero = _mm256_setzero_ps();
    __m256 vone = _mm256_set1_ps(1.0f);

    const int32_t avx2_count = (count_ / 8) * 8;
    for (; i < avx2_count; i += 8) {
        __m256 v = _mm256_loadu_ps(values_ + i);
        v = _mm256_max_ps(v, vzero);
        v = _mm256_min_ps(v, vone);
        _mm256_storeu_ps(values_ + i, v);
    }
#endif

#if defined(__SSE4_1__)
    __m128 vzero_sse = _mm_setzero_ps();
    __m128 vone_sse = _mm_set1_ps(1.0f);

    const int32_t sse_count = (count_ / 4) * 4;
    for (; i < sse_count; i += 4) {
        __m128 v = _mm_loadu_ps(values_ + i);
        v = _mm_max_ps(v, vzero_sse);
        v = _mm_min_ps(v, vone_sse);
        _mm_storeu_ps(values_ + i, v);
    }
#endif

    for (; i < count_; ++i) {
        values_[i] = saturateScalarFallback(values_[i]);
    }
}

} // namespace detail

// ============================================================================
// Public Array API
// ============================================================================

/**
 * @brief Element-wise minimum of two arrays (in-place)
 *
 * Formula: a[i] = min(a[i], b[i])
 *
 * @param a_ Input/output array (modified in-place)
 * @param b_ Second input array
 * @param count_ Number of elements
 *
 * Use case: Mixing buffer size calculation (audio engine, 15 uses)
 * Example: frames_to_process = min(frames_available, buffer_size)
 */
inline void minArray(
    float* MMATH_RESTRICT a_,
    const float* MMATH_RESTRICT b_,
    int32_t count_
) noexcept {
    detail::minSimd(a_, b_, count_);
}

/**
 * @brief Element-wise maximum of two arrays (in-place)
 *
 * Formula: a[i] = max(a[i], b[i])
 *
 * @param a_ Input/output array (modified in-place)
 * @param b_ Second input array
 * @param count_ Number of elements
 *
 * Use case: Parameter validation (audio engine, 7 uses)
 * Example: distance = max(distance, min_distance)
 */
inline void maxArray(
    float* MMATH_RESTRICT a_,
    const float* MMATH_RESTRICT b_,
    int32_t count_
) noexcept {
    detail::maxSimd(a_, b_, count_);
}

/**
 * @brief Clamp array elements to [min_val, max_val] range (in-place)
 *
 * Formula: values[i] = clamp(values[i], min_val, max_val)
 *
 * @param values_ Input/output array (modified in-place)
 * @param min_val_ Minimum bound
 * @param max_val_ Maximum bound
 * @param count_ Number of elements
 *
 * Use case: Volume/pan limiting (audio engine, 27 uses!)
 * Examples:
 * - clampArray(volumes, 0.0f, 1.0f, 48000) // 1 second of audio
 * - clampArray(pan_values, -1.0f, 1.0f, num_sources)
 * - clampArray(soft_clip_output, -1.0f, 1.0f, buffer_size)
 */
inline void clampArray(
    float* MMATH_RESTRICT values_,
    float min_val_,
    float max_val_,
    int32_t count_
) noexcept {
    detail::clampSimd(values_, min_val_, max_val_, count_);
}

/**
 * @brief Absolute value of array elements (in-place)
 *
 * Formula: values[i] = |values[i]|
 *
 * @param values_ Input/output array (modified in-place)
 * @param count_ Number of elements
 *
 * Use case: Peak detection, magnitude calculation (audio engine)
 * Example: absArray(audio_buffer, 48000) for peak metering
 */
inline void absArray(
    float* MMATH_RESTRICT values_,
    int32_t count_
) noexcept {
    detail::absSimd(values_, count_);
}

/**
 * @brief Saturate array elements to [0, 1] range (in-place)
 *
 * Formula: values[i] = clamp(values[i], 0, 1)
 *
 * @param values_ Input/output array (modified in-place)
 * @param count_ Number of elements
 *
 * Use case: Color/alpha clamping, normalized parameters
 */
inline void saturateArray(
    float* MMATH_RESTRICT values_,
    int32_t count_
) noexcept {
    detail::saturateSimd(values_, count_);
}
