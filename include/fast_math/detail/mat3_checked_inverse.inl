/**
 * @brief Try to invert a full 3x3 matrix
 *
 * @param m_ Input matrix
 * @param out_ Output pointer receiving the inverse on success
 * @param epsilon_ Determinant threshold used to classify singular matrices
 * @return true when the matrix is invertible, false otherwise
 */
[[nodiscard]] MMATH_FORCE_INLINE bool mat3TryInverse(Mat3 m_, Mat3* out_, float epsilon_ = 1e-6f) noexcept {
    if (out_ == nullptr) {
        return false;
    }

    float c00 = m_.m11 * m_.m22 - m_.m12 * m_.m21;
    float c01 = m_.m12 * m_.m20 - m_.m10 * m_.m22;
    float c02 = m_.m10 * m_.m21 - m_.m11 * m_.m20;

    float c10 = m_.m02 * m_.m21 - m_.m01 * m_.m22;
    float c11 = m_.m00 * m_.m22 - m_.m02 * m_.m20;
    float c12 = m_.m01 * m_.m20 - m_.m00 * m_.m21;

    float c20 = m_.m01 * m_.m12 - m_.m02 * m_.m11;
    float c21 = m_.m02 * m_.m10 - m_.m00 * m_.m12;
    float c22 = m_.m00 * m_.m11 - m_.m01 * m_.m10;

    float det = m_.m00 * c00 + m_.m01 * c01 + m_.m02 * c02;

    if (std::fabs(det) <= epsilon_) {
        return false;
    }

    float inv_det = 1.0f / det;

    *out_ = Mat3{
        c00 * inv_det, c10 * inv_det, c20 * inv_det,
        c01 * inv_det, c11 * inv_det, c21 * inv_det,
        c02 * inv_det, c12 * inv_det, c22 * inv_det
    };
    return true;
}

/**
 * @brief Checked full matrix inverse
 */
[[nodiscard]] MMATH_FORCE_INLINE Mat3InverseResult mat3InverseChecked(Mat3 m_, float epsilon_ = 1e-6f) noexcept {
    Mat3InverseResult result{mat3Identity(), false};
    result.invertible = mat3TryInverse(m_, &result.value, epsilon_);
    return result;
}

/**
 * @brief Full matrix inverse (deprecated legacy compatibility wrapper)
 *
 * Returns identity when the matrix is singular. New call sites should prefer
 * mat3TryInverse() or mat3InverseChecked() for an explicit failure signal.
 */
MMATH_DEPRECATED("Use mat3TryInverse() or mat3InverseChecked() to handle singular matrices explicitly.")
MMATH_FORCE_INLINE Mat3 mat3Inverse(Mat3 m_) noexcept {
    Mat3 result = mat3Identity();
    (void)mat3TryInverse(m_, &result);
    return result;
}

/**
 * @brief Try to invert an affine 2D matrix
 */
[[nodiscard]] MMATH_FORCE_INLINE bool mat3TryInverseAffine(Mat3 m_, Mat3* out_, float epsilon_ = 1e-6f) noexcept {
    if (out_ == nullptr) {
        return false;
    }

    float det = m_.m00 * m_.m11 - m_.m01 * m_.m10;

    if (std::fabs(det) <= epsilon_) {
        return false;
    }

    float inv_det = 1.0f / det;

    float r00 =  m_.m11 * inv_det;
    float r01 = -m_.m01 * inv_det;
    float r10 = -m_.m10 * inv_det;
    float r11 =  m_.m00 * inv_det;

    float tx = -(r00 * m_.m02 + r01 * m_.m12);
    float ty = -(r10 * m_.m02 + r11 * m_.m12);

    *out_ = Mat3{
        r00, r01, tx,
        r10, r11, ty,
        0.0f, 0.0f, 1.0f
    };
    return true;
}

/**
 * @brief Checked affine matrix inverse
 */
[[nodiscard]] MMATH_FORCE_INLINE Mat3InverseResult mat3InverseAffineChecked(Mat3 m_, float epsilon_ = 1e-6f) noexcept {
    Mat3InverseResult result{mat3Identity(), false};
    result.invertible = mat3TryInverseAffine(m_, &result.value, epsilon_);
    return result;
}

/**
 * @brief Affine matrix inverse (deprecated legacy compatibility wrapper)
 *
 * Returns identity when the matrix is singular. New call sites should prefer
 * mat3TryInverseAffine() or mat3InverseAffineChecked().
 */
MMATH_DEPRECATED("Use mat3TryInverseAffine() or mat3InverseAffineChecked() to handle singular matrices explicitly.")
MMATH_FORCE_INLINE Mat3 mat3InverseAffine(Mat3 m_) noexcept {
    Mat3 result = mat3Identity();
    (void)mat3TryInverseAffine(m_, &result);
    return result;
}
