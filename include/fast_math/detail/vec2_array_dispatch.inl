// Shared Vec2 array dispatch implementation for header/module parity.

// ============================================================================
// Batch Processing (Optimized SIMD dispatch)
// ============================================================================

/**
 * @brief Batch add (SIMD optimized)
 *
 * @param a_ Input vectors A (array of Vec2)
 * @param b_ Input vectors B (array of Vec2)
 * @param result_ Output array
 * @param count_ Number of vectors
 */
inline void vec2AddArray(const Vec2* MMATH_RESTRICT a_,
                         const Vec2* MMATH_RESTRICT b_,
                         Vec2* MMATH_RESTRICT result_,
                         int32_t count_) noexcept {
#if defined(__AVX2__) && defined(__FMA__)
    detail::vec2AddSimd(a_, b_, result_, count_);
#else
    for (int32_t i = 0; i < count_; ++i) {
        result_[i] = vec2Add(a_[i], b_[i]);
    }
#endif
}

/**
 * @brief Batch dot product (SIMD optimized)
 */
inline void vec2DotArray(const Vec2* MMATH_RESTRICT a_,
                         const Vec2* MMATH_RESTRICT b_,
                         float* MMATH_RESTRICT result_,
                         int32_t count_) noexcept {
#if defined(__AVX2__) && defined(__FMA__)
    detail::vec2DotSimd(a_, b_, result_, count_);
#else
    for (int32_t i = 0; i < count_; ++i) {
        result_[i] = vec2Dot(a_[i], b_[i]);
    }
#endif
}

/**
 * @brief Batch normalize (SIMD optimized)
 */
inline void vec2NormalizeArray(const Vec2* MMATH_RESTRICT input_,
                               Vec2* MMATH_RESTRICT result_,
                               int32_t count_) noexcept {
#if defined(__AVX2__) && defined(__FMA__)
    detail::vec2NormalizeSimd(input_, result_, count_);
#else
    for (int32_t i = 0; i < count_; ++i) {
        result_[i] = vec2Normalize(input_[i]);
    }
#endif
}

/**
 * @brief Batch fast normalize (SIMD optimized)
 */
inline void vec2NormalizeFastArray(const Vec2* MMATH_RESTRICT input_,
                                   Vec2* MMATH_RESTRICT result_,
                                   int32_t count_) noexcept {
#if defined(__AVX2__) && defined(__FMA__)
    detail::vec2NormalizeFastSimd(input_, result_, count_);
#else
    for (int32_t i = 0; i < count_; ++i) {
        result_[i] = vec2NormalizeFast(input_[i]);
    }
#endif
}
