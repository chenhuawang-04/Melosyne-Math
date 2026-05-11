// Shared Mat3 scalar/core implementation for header/module parity.

// ============================================================================
// Scalar Mat3 Operations (POD + Free Functions)
// ============================================================================

/**
 * @brief Identity matrix
 */
MMATH_FORCE_INLINE Mat3 mat3Identity() noexcept {
    return Mat3{
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
    };
}

/**
 * @brief Zero matrix
 */
MMATH_FORCE_INLINE Mat3 mat3Zero() noexcept {
    return Mat3{
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f
    };
}

/**
 * @brief Create translation matrix
 */
MMATH_FORCE_INLINE Mat3 mat3FromTranslation(Vec2 translation_) noexcept {
    return Mat3{
        1.0f, 0.0f, translation_.x,
        0.0f, 1.0f, translation_.y,
        0.0f, 0.0f, 1.0f
    };
}

/**
 * @brief Create rotation matrix
 */
MMATH_FORCE_INLINE Mat3 mat3FromRotation(float angle_radians_) noexcept {
    float c = std::cos(angle_radians_);
    float s = std::sin(angle_radians_);
    return Mat3{
        c,    -s,   0.0f,
        s,     c,   0.0f,
        0.0f, 0.0f, 1.0f
    };
}

/**
 * @brief Create scale matrix
 */
MMATH_FORCE_INLINE Mat3 mat3FromScale(Vec2 scale_) noexcept {
    return Mat3{
        scale_.x, 0.0f,     0.0f,
        0.0f,     scale_.y, 0.0f,
        0.0f,     0.0f,     1.0f
    };
}

/**
 * @brief Create TRS (Translation-Rotation-Scale) matrix
 */
MMATH_FORCE_INLINE Mat3 mat3FromTrs(Vec2 translation_, float rotation_radians_, Vec2 scale_) noexcept {
    float c = std::cos(rotation_radians_);
    float s = std::sin(rotation_radians_);
    return Mat3{
        scale_.x * c, -scale_.y * s, translation_.x,
        scale_.x * s,  scale_.y * c, translation_.y,
        0.0f,         0.0f,        1.0f
    };
}

/**
 * @brief Full 3x3 matrix multiplication
 */
MMATH_FORCE_INLINE Mat3 mat3Multiply(Mat3 a_, Mat3 b_) noexcept {
    return Mat3{
        a_.m00 * b_.m00 + a_.m01 * b_.m10 + a_.m02 * b_.m20,
        a_.m00 * b_.m01 + a_.m01 * b_.m11 + a_.m02 * b_.m21,
        a_.m00 * b_.m02 + a_.m01 * b_.m12 + a_.m02 * b_.m22,

        a_.m10 * b_.m00 + a_.m11 * b_.m10 + a_.m12 * b_.m20,
        a_.m10 * b_.m01 + a_.m11 * b_.m11 + a_.m12 * b_.m21,
        a_.m10 * b_.m02 + a_.m11 * b_.m12 + a_.m12 * b_.m22,

        a_.m20 * b_.m00 + a_.m21 * b_.m10 + a_.m22 * b_.m20,
        a_.m20 * b_.m01 + a_.m21 * b_.m11 + a_.m22 * b_.m21,
        a_.m20 * b_.m02 + a_.m21 * b_.m12 + a_.m22 * b_.m22
    };
}

/**
 * @brief Deprecated compatibility alias for 3x3 matrix multiplication
 *
 * mat3Multiply() is the canonical API spelling.
 */
MMATH_DEPRECATED("Use mat3Multiply() as the canonical API spelling.")
MMATH_FORCE_INLINE Mat3 mat3Mul(Mat3 a_, Mat3 b_) noexcept {
    return mat3Multiply(a_, b_);
}

/**
 * @brief Affine matrix multiplication (optimized for 2D transforms)
 * Assumes last row is [0, 0, 1]
 */
MMATH_FORCE_INLINE Mat3 mat3MultiplyAffine(Mat3 a_, Mat3 b_) noexcept {
    return Mat3{
        a_.m00 * b_.m00 + a_.m01 * b_.m10,
        a_.m00 * b_.m01 + a_.m01 * b_.m11,
        a_.m00 * b_.m02 + a_.m01 * b_.m12 + a_.m02,

        a_.m10 * b_.m00 + a_.m11 * b_.m10,
        a_.m10 * b_.m01 + a_.m11 * b_.m11,
        a_.m10 * b_.m02 + a_.m11 * b_.m12 + a_.m12,

        0.0f, 0.0f, 1.0f
    };
}

/**
 * @brief Deprecated compatibility alias for affine 3x3 multiplication
 */
MMATH_DEPRECATED("Use mat3MultiplyAffine() as the canonical API spelling.")
MMATH_FORCE_INLINE Mat3 mat3MulAffine(Mat3 a_, Mat3 b_) noexcept {
    return mat3MultiplyAffine(a_, b_);
}

/**
 * @brief Transform point (with translation)
 */
MMATH_FORCE_INLINE Vec2 mat3TransformPoint(Mat3 m_, Vec2 point_) noexcept {
    return Vec2{
        m_.m00 * point_.x + m_.m01 * point_.y + m_.m02,
        m_.m10 * point_.x + m_.m11 * point_.y + m_.m12
    };
}

/**
 * @brief Transform vector (no translation)
 */
MMATH_FORCE_INLINE Vec2 mat3TransformVector(Mat3 m_, Vec2 vector_) noexcept {
    return Vec2{
        m_.m00 * vector_.x + m_.m01 * vector_.y,
        m_.m10 * vector_.x + m_.m11 * vector_.y
    };
}

/**
 * @brief Calculate determinant
 */
MMATH_FORCE_INLINE float mat3Determinant(Mat3 m_) noexcept {
    return m_.m00 * (m_.m11 * m_.m22 - m_.m12 * m_.m21)
         - m_.m01 * (m_.m10 * m_.m22 - m_.m12 * m_.m20)
         + m_.m02 * (m_.m10 * m_.m21 - m_.m11 * m_.m20);
}

#include "mat3_checked_inverse.inl"

/**
 * @brief Matrix transpose
 */
MMATH_FORCE_INLINE Mat3 mat3Transpose(Mat3 m_) noexcept {
    return Mat3{
        m_.m00, m_.m10, m_.m20,
        m_.m01, m_.m11, m_.m21,
        m_.m02, m_.m12, m_.m22
    };
}

/**
 * @brief Equality comparison
 */
MMATH_FORCE_INLINE bool mat3Equal(Mat3 a_, Mat3 b_) noexcept {
    return a_.m00 == b_.m00 && a_.m01 == b_.m01 && a_.m02 == b_.m02 &&
           a_.m10 == b_.m10 && a_.m11 == b_.m11 && a_.m12 == b_.m12 &&
           a_.m20 == b_.m20 && a_.m21 == b_.m21 && a_.m22 == b_.m22;
}

/**
 * @brief Near-equality comparison with epsilon
 */
MMATH_FORCE_INLINE bool mat3NearEqual(Mat3 a_, Mat3 b_, float epsilon_) noexcept {
    auto is_near_ = [epsilon_](float x, float y) {
        float d = x - y;
        return (d < 0 ? -d : d) <= epsilon_;
    };
    return is_near_(a_.m00, b_.m00) && is_near_(a_.m01, b_.m01) && is_near_(a_.m02, b_.m02) &&
           is_near_(a_.m10, b_.m10) && is_near_(a_.m11, b_.m11) && is_near_(a_.m12, b_.m12) &&
           is_near_(a_.m20, b_.m20) && is_near_(a_.m21, b_.m21) && is_near_(a_.m22, b_.m22);
}
