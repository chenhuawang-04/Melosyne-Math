// Shared Mat3 batch dispatch implementation for header/module parity.

// ============================================================================
// Batch Processing (Optimized SIMD dispatch)
// ============================================================================

/**
 * @brief Batch TRS construction (SIMD optimized)
 *
 * @param translations_ Translation vectors (array of Vec2)
 * @param rotations_ Rotation angles in radians (array of float)
 * @param scales_ Scale vectors (array of Vec2)
 * @param result_ Output array
 * @param count_ Number of matrices
 */
inline void mat3FromTrsArray(const Vec2* MMATH_RESTRICT translations_,
                             const float* MMATH_RESTRICT rotations_,
                             const Vec2* MMATH_RESTRICT scales_,
                             Mat3* MMATH_RESTRICT result_,
                             int32_t count_) noexcept {
#if defined(__AVX2__) && defined(__FMA__)
    detail::mat3FromTrsSimd(translations_, rotations_, scales_, result_, count_);
#else
    for (int32_t i = 0; i < count_; ++i) {
        result_[i] = mat3FromTrs(translations_[i], rotations_[i], scales_[i]);
    }
#endif
}

/**
 * @brief Batch affine matrix multiplication (SIMD optimized)
 */
inline void mat3MultiplyAffineArray(const Mat3* MMATH_RESTRICT a_,
                                    const Mat3* MMATH_RESTRICT b_,
                                    Mat3* MMATH_RESTRICT result_,
                                    int32_t count_) noexcept {
#if defined(__AVX2__) && defined(__FMA__)
    detail::mat3MultiplyAffineSimd(a_, b_, result_, count_);
#else
    for (int32_t i = 0; i < count_; ++i) {
        result_[i] = mat3MultiplyAffine(a_[i], b_[i]);
    }
#endif
}

/**
 * @brief Batch point transformation (SIMD optimized)
 */
inline void mat3TransformPointsArray(const Mat3* MMATH_RESTRICT matrices_,
                                     const Vec2* MMATH_RESTRICT points_,
                                     Vec2* MMATH_RESTRICT result_,
                                     int32_t count_) noexcept {
#if defined(__AVX2__) && defined(__FMA__)
    detail::mat3TransformPointsSimd(matrices_, points_, result_, count_);
#else
    for (int32_t i = 0; i < count_; ++i) {
        result_[i] = mat3TransformPoint(matrices_[i], points_[i]);
    }
#endif
}

/**
 * @brief Transform multiple points by a single matrix (SIMD optimized)
 */
inline void mat3TransformPointsSingle(Mat3 matrix_,
                                      const Vec2* MMATH_RESTRICT points_,
                                      Vec2* MMATH_RESTRICT result_,
                                      int32_t count_) noexcept {
#if defined(__AVX2__) && defined(__FMA__)
    detail::mat3TransformPointsSingleSimd(matrix_, points_, result_, count_);
#else
    for (int32_t i = 0; i < count_; ++i) {
        result_[i] = mat3TransformPoint(matrix_, points_[i]);
    }
#endif
}
