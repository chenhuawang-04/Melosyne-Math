// Shared Aabb2 batch dispatch implementation for header/module parity.

// ============================================================================
// Batch Processing (SIMD dispatch)
// ============================================================================

/**
 * @brief Batch overlap test (one reference AABB vs many candidates)
 */
inline void aabb2IntersectsArray(
    Aabb2 reference_,
    const Aabb2* MMATH_RESTRICT candidates_,
    bool* MMATH_RESTRICT results_,
    int32_t count_
) noexcept {
#if defined(__AVX2__) && defined(__FMA__)
    detail::aabb2IntersectsArraySimd(reference_, candidates_, results_, count_);
#else
    for (int32_t i = 0; i < count_; ++i) {
        results_[i] = aabb2Intersects(reference_, candidates_[i]);
    }
#endif
}

/**
 * @brief Batch merge (pairwise)
 */
inline void aabb2MergeArray(
    const Aabb2* MMATH_RESTRICT a_,
    const Aabb2* MMATH_RESTRICT b_,
    Aabb2* MMATH_RESTRICT result_,
    int32_t count_
) noexcept {
#if defined(__AVX2__) && defined(__FMA__)
    detail::aabb2MergeArraySimd(a_, b_, result_, count_);
#else
    for (int32_t i = 0; i < count_; ++i) {
        result_[i] = aabb2Merge(a_[i], b_[i]);
    }
#endif
}

/**
 * @brief Batch contains point test (one AABB vs many points)
 */
inline void aabb2ContainsPointsArray(
    Aabb2 aabb_,
    const Vec2* MMATH_RESTRICT points_,
    bool* MMATH_RESTRICT results_,
    int32_t count_
) noexcept {
#if defined(__AVX2__) && defined(__FMA__)
    detail::aabb2ContainsPointsArraySimd(aabb_, points_, results_, count_);
#else
    for (int32_t i = 0; i < count_; ++i) {
        results_[i] = aabb2ContainsPoint(aabb_, points_[i]);
    }
#endif
}
