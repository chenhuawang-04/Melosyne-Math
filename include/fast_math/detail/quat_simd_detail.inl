// Shared Quat SIMD batch/detail implementation for header/module parity.
//
// Strategy
// --------
// Single-quaternion operations stay scalar in quat_core.inl; modern compilers
// vectorize the explicit Hamilton/normalize/rotate formulas to ~5-6 SSE ops
// with -O3 -march=native, matching or beating hand-rolled intrinsics for
// isolated operations.
//
// What this file adds is batch (array-style) processing where the win is real:
//
//   - quatMultiplyArray:      out[i] = quatMultiply(a[i], b[i])
//   - quatNormalizeArray:     in-place; tolerates non-unit input
//   - quatRotateVec3Array:    out[i] = quatRotateVec3(q[i], v[i])
//   - quatIntegrateArray:     out[i] = quatIntegrate(q[i], omega[i], dt)
//
// All entry points use Quat in AoS form (the project's canonical storage) and
// rely on compiler auto-vectorization, which behaves well on the 16-byte
// aligned Quat layout. Future hand-tuned AVX2 SoA paths can be slotted in
// behind the same public API without ABI churn.

namespace detail {

// Internal scalar building blocks (kept here so the array variants do not
// have a transitive dependency on quat_core.inl when included alone). They
// are deliberately tiny so the optimizer inlines them away.

MMATH_FORCE_INLINE Quat quatMulScalar(const Quat& a_, const Quat& b_) noexcept {
    return Quat{
        a_.w * b_.x + a_.x * b_.w + a_.y * b_.z - a_.z * b_.y,
        a_.w * b_.y - a_.x * b_.z + a_.y * b_.w + a_.z * b_.x,
        a_.w * b_.z + a_.x * b_.y - a_.y * b_.x + a_.z * b_.w,
        a_.w * b_.w - a_.x * b_.x - a_.y * b_.y - a_.z * b_.z
    };
}

MMATH_FORCE_INLINE Quat quatNormalizeScalar(const Quat& q_) noexcept {
    const float len_sq = q_.x * q_.x + q_.y * q_.y + q_.z * q_.z + q_.w * q_.w;
    const float inv_len = MMath::inverseSqrt(len_sq);
    return Quat{q_.x * inv_len, q_.y * inv_len, q_.z * inv_len, q_.w * inv_len};
}

MMATH_FORCE_INLINE Vec3 quatRotateVec3Scalar(const Quat& q_, const Vec3& v_) noexcept {
    const float tx = 2.0f * (q_.y * v_.z - q_.z * v_.y);
    const float ty = 2.0f * (q_.z * v_.x - q_.x * v_.z);
    const float tz = 2.0f * (q_.x * v_.y - q_.y * v_.x);
    const float cx = q_.y * tz - q_.z * ty;
    const float cy = q_.z * tx - q_.x * tz;
    const float cz = q_.x * ty - q_.y * tx;
    return Vec3{
        v_.x + q_.w * tx + cx,
        v_.y + q_.w * ty + cy,
        v_.z + q_.w * tz + cz
    };
}

} // namespace detail

// ============================================================================
// Batch APIs (AoS in/out, compiler auto-vectorization)
// ============================================================================

/**
 * @brief Per-element Hamilton product: out[i] = a[i] * b[i].
 *
 * The arrays may alias (out == a or out == b is supported).
 */
inline void quatMultiplyArray(
    const Quat* MMATH_RESTRICT a_,
    const Quat* MMATH_RESTRICT b_,
    Quat* MMATH_RESTRICT out_,
    int32_t count_
) noexcept {
    for (int32_t i = 0; i < count_; ++i) {
        out_[i] = detail::quatMulScalar(a_[i], b_[i]);
    }
}

/**
 * @brief In-place batch normalize. Uses MMath::inverseSqrt; tolerates non-unit input.
 *
 * @warning Behaviour on zero-length quaternions is undefined (matches the
 *          single-element quatNormalize). Use quatNormalizeSafeArray() if
 *          the input may contain zero quaternions.
 */
inline void quatNormalizeArray(Quat* MMATH_RESTRICT q_, int32_t count_) noexcept {
    for (int32_t i = 0; i < count_; ++i) {
        q_[i] = detail::quatNormalizeScalar(q_[i]);
    }
}

/**
 * @brief In-place batch normalize with zero-length guard.
 *
 * Quaternions with length squared below epsilon^2 are replaced by identity.
 */
inline void quatNormalizeSafeArray(Quat* MMATH_RESTRICT q_, int32_t count_, float epsilon_ = 1e-12f) noexcept {
    const float thresh = epsilon_ * epsilon_;
    for (int32_t i = 0; i < count_; ++i) {
        const Quat& s = q_[i];
        const float len_sq = s.x * s.x + s.y * s.y + s.z * s.z + s.w * s.w;
        if (len_sq < thresh) {
            q_[i] = Quat{0.0f, 0.0f, 0.0f, 1.0f};
        } else {
            const float inv = MMath::inverseSqrt(len_sq);
            q_[i] = Quat{s.x * inv, s.y * inv, s.z * inv, s.w * inv};
        }
    }
}

/**
 * @brief Per-element vector rotation: out[i] = quatRotateVec3(q[i], v[i]).
 *
 * Assumes each q[i] is unit-length. Arrays may alias.
 */
inline void quatRotateVec3Array(
    const Quat* MMATH_RESTRICT q_,
    const Vec3* MMATH_RESTRICT v_,
    Vec3* MMATH_RESTRICT out_,
    int32_t count_
) noexcept {
    for (int32_t i = 0; i < count_; ++i) {
        out_[i] = detail::quatRotateVec3Scalar(q_[i], v_[i]);
    }
}

/**
 * @brief Per-element world-frame integration:
 *        out[i] = normalize(q[i] + 0.5*dt*(omega_quat[i] * q[i]))
 *
 * @param q_     Current orientations (count_ entries)
 * @param omega_ World-frame angular velocities (count_ entries, rad/s)
 * @param dt_    Time step (seconds), shared across the batch
 * @param out_   Output orientations (count_ entries). May alias @p q_.
 */
inline void quatIntegrateArray(
    const Quat* MMATH_RESTRICT q_,
    const Vec3* MMATH_RESTRICT omega_,
    float dt_,
    Quat* MMATH_RESTRICT out_,
    int32_t count_
) noexcept {
    const float h = 0.5f * dt_;
    for (int32_t i = 0; i < count_; ++i) {
        const Quat& q = q_[i];
        const Vec3& w = omega_[i];
        const float px =  w.x * q.w + w.y * q.z - w.z * q.y;
        const float py = -w.x * q.z + w.y * q.w + w.z * q.x;
        const float pz =  w.x * q.y - w.y * q.x + w.z * q.w;
        const float pw = -(w.x * q.x + w.y * q.y + w.z * q.z);
        const Quat raw{
            q.x + h * px,
            q.y + h * py,
            q.z + h * pz,
            q.w + h * pw
        };
        out_[i] = detail::quatNormalizeScalar(raw);
    }
}

/**
 * @brief Vec4 overload of quatIntegrateArray: omega.w is ignored.
 *
 * Provided for SoA/Vec4-column storage layouts so callers do not have to
 * repack data on the hot path.
 */
inline void quatIntegrateArray(
    const Quat* MMATH_RESTRICT q_,
    const Vec4* MMATH_RESTRICT omega_,
    float dt_,
    Quat* MMATH_RESTRICT out_,
    int32_t count_
) noexcept {
    const float h = 0.5f * dt_;
    for (int32_t i = 0; i < count_; ++i) {
        const Quat& q = q_[i];
        const Vec4& w = omega_[i];
        const float px =  w.x * q.w + w.y * q.z - w.z * q.y;
        const float py = -w.x * q.z + w.y * q.w + w.z * q.x;
        const float pz =  w.x * q.y - w.y * q.x + w.z * q.w;
        const float pw = -(w.x * q.x + w.y * q.y + w.z * q.z);
        const Quat raw{
            q.x + h * px,
            q.y + h * py,
            q.z + h * pz,
            q.w + h * pw
        };
        out_[i] = detail::quatNormalizeScalar(raw);
    }
}
