// Shared Vec2 scalar core implementation for header/module parity.

// ============================================================================
// Scalar Vec2 Operations (POD + Free Functions)
// ============================================================================

/**
 * @brief Add two vectors
 */
MMATH_FORCE_INLINE Vec2 vec2Add(Vec2 a_, Vec2 b_) noexcept {
    return Vec2{ a_.x + b_.x, a_.y + b_.y };
}

/**
 * @brief Subtract two vectors
 */
MMATH_FORCE_INLINE Vec2 vec2Sub(Vec2 a_, Vec2 b_) noexcept {
    return Vec2{ a_.x - b_.x, a_.y - b_.y };
}

/**
 * @brief Component-wise multiplication
 */
MMATH_FORCE_INLINE Vec2 vec2Mul(Vec2 a_, Vec2 b_) noexcept {
    return Vec2{ a_.x * b_.x, a_.y * b_.y };
}

/**
 * @brief Component-wise division
 */
MMATH_FORCE_INLINE Vec2 vec2Div(Vec2 a_, Vec2 b_) noexcept {
    return Vec2{ a_.x / b_.x, a_.y / b_.y };
}

/**
 * @brief Scale vector by scalar
 */
MMATH_FORCE_INLINE Vec2 vec2Scale(Vec2 v_, float s_) noexcept {
    return Vec2{ v_.x * s_, v_.y * s_ };
}

/**
 * @brief Negate vector
 */
MMATH_FORCE_INLINE Vec2 vec2Negate(Vec2 v_) noexcept {
    return Vec2{ -v_.x, -v_.y };
}

/**
 * @brief Dot product
 */
MMATH_FORCE_INLINE float vec2Dot(Vec2 a_, Vec2 b_) noexcept {
    return a_.x * b_.x + a_.y * b_.y;
}

/**
 * @brief Squared length (avoids sqrt)
 */
MMATH_FORCE_INLINE float vec2LengthSquared(Vec2 v_) noexcept {
    return v_.x * v_.x + v_.y * v_.y;
}

/**
 * @brief Length
 */
MMATH_FORCE_INLINE float vec2Length(Vec2 v_) noexcept {
    return std::sqrt(vec2LengthSquared(v_));
}

/**
 * @brief Normalize vector
 */
MMATH_FORCE_INLINE Vec2 vec2Normalize(Vec2 v_) noexcept {
    float len = vec2Length(v_);
    float inv_len = (len > 0.0f) ? (1.0f / len) : 0.0f;
    return vec2Scale(v_, inv_len);
}

/**
 * @brief Fast normalize using approximate reciprocal square root (~0.1% error)
 */
MMATH_FORCE_INLINE Vec2 vec2NormalizeFast(Vec2 v_) noexcept {
#if defined(__AVX2__) || defined(__SSE__)
    float len_sq = vec2LengthSquared(v_);
    __m128 vlen_sq = _mm_set_ss(len_sq);
    __m128 rsqrt = _mm_rsqrt_ss(vlen_sq);  // Approximate 1/sqrt
    float inv_len = _mm_cvtss_f32(rsqrt);
    return vec2Scale(v_, inv_len);
#else
    return vec2Normalize(v_);  // Fallback
#endif
}

/**
 * @brief 2D cross product (returns scalar z-component)
 */
MMATH_FORCE_INLINE float vec2Cross(Vec2 a_, Vec2 b_) noexcept {
    return a_.x * b_.y - a_.y * b_.x;
}

/**
 * @brief Perpendicular vector (90掳 counter-clockwise)
 */
MMATH_FORCE_INLINE Vec2 vec2Perpendicular(Vec2 v_) noexcept {
    return Vec2{ -v_.y, v_.x };
}

/**
 * @brief Perpendicular vector (90掳 clockwise)
 */
MMATH_FORCE_INLINE Vec2 vec2PerpendicularCw(Vec2 v_) noexcept {
    return Vec2{ v_.y, -v_.x };
}

/**
 * @brief Linear interpolation
 */
MMATH_FORCE_INLINE Vec2 vec2Lerp(Vec2 a_, Vec2 b_, float t_) noexcept {
    return Vec2{
        a_.x + (b_.x - a_.x) * t_,
        a_.y + (b_.y - a_.y) * t_
    };
}

/**
 * @brief Component-wise minimum
 */
MMATH_FORCE_INLINE Vec2 vec2Min(Vec2 a_, Vec2 b_) noexcept {
    return Vec2{
        (a_.x < b_.x) ? a_.x : b_.x,
        (a_.y < b_.y) ? a_.y : b_.y
    };
}

/**
 * @brief Component-wise maximum
 */
MMATH_FORCE_INLINE Vec2 vec2Max(Vec2 a_, Vec2 b_) noexcept {
    return Vec2{
        (a_.x > b_.x) ? a_.x : b_.x,
        (a_.y > b_.y) ? a_.y : b_.y
    };
}

/**
 * @brief Reflect vector around normal
 * Formula: v - 2 * dot(v, n) * n
 */
MMATH_FORCE_INLINE Vec2 vec2Reflect(Vec2 v_, Vec2 n_) noexcept {
    float d = vec2Dot(v_, n_);
    return Vec2{
        v_.x - 2.0f * d * n_.x,
        v_.y - 2.0f * d * n_.y
    };
}

/**
 * @brief Project vector a onto b
 * Formula: dot(a, b) / dot(b, b) * b
 */
MMATH_FORCE_INLINE Vec2 vec2Project(Vec2 a_, Vec2 b_) noexcept {
    float d = vec2Dot(a_, b_);
    float b_len_sq = vec2LengthSquared(b_);
    float scale = (b_len_sq > 0.0f) ? (d / b_len_sq) : 0.0f;
    return vec2Scale(b_, scale);
}

/**
 * @brief Squared distance between two points
 */
MMATH_FORCE_INLINE float vec2DistanceSquared(Vec2 a_, Vec2 b_) noexcept {
    Vec2 diff = vec2Sub(a_, b_);
    return vec2LengthSquared(diff);
}

/**
 * @brief Distance between two points
 */
MMATH_FORCE_INLINE float vec2Distance(Vec2 a_, Vec2 b_) noexcept {
    return std::sqrt(vec2DistanceSquared(a_, b_));
}

/**
 * @brief Equality comparison
 */
MMATH_FORCE_INLINE bool vec2Equal(Vec2 a_, Vec2 b_) noexcept {
    return (a_.x == b_.x) && (a_.y == b_.y);
}

/**
 * @brief Near-equality comparison with epsilon
 */
MMATH_FORCE_INLINE bool vec2NearEqual(Vec2 a_, Vec2 b_, float epsilon_) noexcept {
    float dx = a_.x - b_.x;
    float dy = a_.y - b_.y;
    return (dx * dx + dy * dy) < (epsilon_ * epsilon_);
}
