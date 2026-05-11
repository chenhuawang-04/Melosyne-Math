/**
 * @brief Try to invert a 4x4 matrix
 *
 * Target: Beat Eigen (10.70ms) / DirectXMath (11.02ms)
 * Strategy: Pure scalar - Eigen's proven algorithm is already optimal
 * SIMD overhead exceeds benefits for this complex algorithm
 */
[[nodiscard]] MMATH_FORCE_INLINE bool mat4TryInverse(const Mat4& m_, Mat4* out_, float epsilon_ = 1e-6f) noexcept {
    if (out_ == nullptr) {
        return false;
    }

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

    if (std::fabs(det) <= epsilon_) {
        return false;
    }

    float invDet = 1.0f / det;

    // Compute adjugate matrix scaled by 1/det
    out_->m[0]  = ( m11 * c5 - m12 * c4 + m13 * c3) * invDet;
    out_->m[1]  = (-m01 * c5 + m02 * c4 - m03 * c3) * invDet;
    out_->m[2]  = ( m31 * s5 - m32 * s4 + m33 * s3) * invDet;
    out_->m[3]  = (-m21 * s5 + m22 * s4 - m23 * s3) * invDet;

    out_->m[4]  = (-m10 * c5 + m12 * c2 - m13 * c1) * invDet;
    out_->m[5]  = ( m00 * c5 - m02 * c2 + m03 * c1) * invDet;
    out_->m[6]  = (-m30 * s5 + m32 * s2 - m33 * s1) * invDet;
    out_->m[7]  = ( m20 * s5 - m22 * s2 + m23 * s1) * invDet;

    out_->m[8]  = ( m10 * c4 - m11 * c2 + m13 * c0) * invDet;
    out_->m[9]  = (-m00 * c4 + m01 * c2 - m03 * c0) * invDet;
    out_->m[10] = ( m30 * s4 - m31 * s2 + m33 * s0) * invDet;
    out_->m[11] = (-m20 * s4 + m21 * s2 - m23 * s0) * invDet;

    out_->m[12] = (-m10 * c3 + m11 * c1 - m12 * c0) * invDet;
    out_->m[13] = ( m00 * c3 - m01 * c1 + m02 * c0) * invDet;
    out_->m[14] = (-m30 * s3 + m31 * s1 - m32 * s0) * invDet;
    out_->m[15] = ( m20 * s3 - m21 * s1 + m22 * s0) * invDet;
    return true;
}

/**
 * @brief Checked 4x4 inverse
 */
[[nodiscard]] MMATH_FORCE_INLINE Mat4InverseResult mat4InverseChecked(const Mat4& m_, float epsilon_ = 1e-6f) noexcept {
    Mat4InverseResult result{mat4Identity(), false};
    result.invertible = mat4TryInverse(m_, &result.value, epsilon_);
    return result;
}

/**
 * @brief Matrix inverse (deprecated legacy compatibility wrapper)
 *
 * Returns identity when the matrix is singular. New call sites should prefer
 * mat4TryInverse() or mat4InverseChecked() for an explicit failure signal.
 */
MMATH_DEPRECATED("Use mat4TryInverse() or mat4InverseChecked() to handle singular matrices explicitly.")
MMATH_FORCE_INLINE Mat4 mat4Inverse(const Mat4& m_) noexcept {
    Mat4 result = mat4Identity();
    (void)mat4TryInverse(m_, &result);
    return result;
}
