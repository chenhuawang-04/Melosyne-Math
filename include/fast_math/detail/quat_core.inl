// Shared Quat scalar/core implementation for header/module parity.
//
// Tier 1 + 2 + 4 scalar operations. Tier 3 (Mat3/Mat4 conversion, axis-angle,
// slerp, Euler) lives in quat_convert.inl so this file does not have to know
// about Mat3/Mat4.
//
// Locked Conventions
// ------------------
//   layout       : Quat{x, y, z, w}, w is the scalar/real part
//   identity     : {0, 0, 0, 1}
//   convention   : Hamilton, right-handed
//   composition  : quatMultiply(a, b) applies b first then a, so
//                  quatRotateVec3(quatMultiply(a, b), v) ==
//                  quatRotateVec3(a, quatRotateVec3(b, v))
//   angles       : radians
//
// Deterministic Profile
// ---------------------
// Scalar operations in this file deliberately avoid std::sqrt and raw
// _mm_rsqrt_ss; the canonical normalization uses MMath::inverseSqrt and
// MMath::sqrt (provided by sqrt.h). Hot-path expressions are written with
// explicit parentheses so /fp:precise and -fno-fast-math builds reproduce
// bit-identical results.

// ============================================================================
// Tier 1: Foundations
// ============================================================================

/**
 * @brief Identity quaternion {0, 0, 0, 1}
 *
 * constexpr so it can appear in constant expressions and static initializers.
 */
constexpr MMATH_FORCE_INLINE Quat quatIdentity() noexcept {
    return Quat{0.0f, 0.0f, 0.0f, 1.0f};
}

/**
 * @brief Squared length |q|^2 = x^2 + y^2 + z^2 + w^2
 *
 * Compute first; defer sqrt/inverseSqrt to caller. Hot-path safe.
 */
MMATH_FORCE_INLINE float quatLengthSquared(const Quat& q_) noexcept {
    return q_.x * q_.x + q_.y * q_.y + q_.z * q_.z + q_.w * q_.w;
}

/**
 * @brief Normalize quaternion to unit length using MMath::inverseSqrt.
 *
 * Deterministic-Profile compliant: routes through MMath::inverseSqrt
 * (rsqrt + 1 Newton-Raphson step inside sqrt.h), never std::sqrt or raw rsqrt.
 *
 * @warning Undefined behavior if |q| is zero. Use quatNormalizeSafe() if the
 *          quaternion may be zero.
 */
MMATH_FORCE_INLINE Quat quatNormalize(const Quat& q_) noexcept {
    const float len_sq = quatLengthSquared(q_);
    const float inv_len = MMath::inverseSqrt(len_sq);
    return Quat{q_.x * inv_len, q_.y * inv_len, q_.z * inv_len, q_.w * inv_len};
}

/**
 * @brief Safe normalize: returns identity if |q|^2 < epsilon^2.
 *
 * Branch is predictable for typical workloads (always non-zero on hot path)
 * and provides a sane fallback for degenerate inputs.
 */
MMATH_FORCE_INLINE Quat quatNormalizeSafe(const Quat& q_, float epsilon_ = 1e-12f) noexcept {
    const float len_sq = quatLengthSquared(q_);
    if (len_sq < epsilon_ * epsilon_) {
        return quatIdentity();
    }
    const float inv_len = MMath::inverseSqrt(len_sq);
    return Quat{q_.x * inv_len, q_.y * inv_len, q_.z * inv_len, q_.w * inv_len};
}

/**
 * @brief First-order quaternion integration with world-frame angular velocity.
 *
 * Convention: q_dot = 0.5 * (omega_quat * q) where omega_quat = (omega, 0).
 * Result is renormalized to absorb integration drift.
 *
 *   q_new = normalize(q + (dt * 0.5) * (omega_quat * q))
 *
 * Expanded for the Hamilton product (apply-b-first convention) without any
 * intermediate Quat allocation. Designed to be the single hottest function
 * in physics inner loops.
 *
 * @param q_     Current orientation (assumed near-unit; will be renormalized)
 * @param omega_ Body-to-world angular velocity (rad/s)
 * @param dt_    Time step (seconds)
 */
MMATH_FORCE_INLINE Quat quatIntegrate(const Quat& q_, const Vec3& omega_, float dt_) noexcept {
    // p = omega_quat * q, where omega_quat = (omega.x, omega.y, omega.z, 0)
    // Hamilton product with a.w = 0 reduces to:
    //   p.x =  omega.x * q.w + omega.y * q.z - omega.z * q.y
    //   p.y = -omega.x * q.z + omega.y * q.w + omega.z * q.x
    //   p.z =  omega.x * q.y - omega.y * q.x + omega.z * q.w
    //   p.w = -(omega.x * q.x + omega.y * q.y + omega.z * q.z)
    const float px =  omega_.x * q_.w + omega_.y * q_.z - omega_.z * q_.y;
    const float py = -omega_.x * q_.z + omega_.y * q_.w + omega_.z * q_.x;
    const float pz =  omega_.x * q_.y - omega_.y * q_.x + omega_.z * q_.w;
    const float pw = -(omega_.x * q_.x + omega_.y * q_.y + omega_.z * q_.z);

    const float h = 0.5f * dt_;
    const Quat raw{
        q_.x + h * px,
        q_.y + h * py,
        q_.z + h * pz,
        q_.w + h * pw
    };
    return quatNormalize(raw);
}

/**
 * @brief Vec4 overload: angular velocity carried in (x, y, z); w is ignored.
 *
 * Provided so PhysicsCenter's SoA / Vec4 column storage does not have to
 * repack data into Vec3 on the hot path.
 */
MMATH_FORCE_INLINE Quat quatIntegrate(const Quat& q_, const Vec4& omega_, float dt_) noexcept {
    const float px =  omega_.x * q_.w + omega_.y * q_.z - omega_.z * q_.y;
    const float py = -omega_.x * q_.z + omega_.y * q_.w + omega_.z * q_.x;
    const float pz =  omega_.x * q_.y - omega_.y * q_.x + omega_.z * q_.w;
    const float pw = -(omega_.x * q_.x + omega_.y * q_.y + omega_.z * q_.z);

    const float h = 0.5f * dt_;
    const Quat raw{
        q_.x + h * px,
        q_.y + h * py,
        q_.z + h * pz,
        q_.w + h * pw
    };
    return quatNormalize(raw);
}

// ============================================================================
// Tier 2: Core algebra
// ============================================================================

/**
 * @brief Hamilton product: r = a * b
 *
 * Composition semantics: r = a*b means "apply b first, then a", i.e.
 *   quatRotateVec3(quatMultiply(a, b), v) ==
 *   quatRotateVec3(a, quatRotateVec3(b, v))
 *
 * 16 muls + 12 adds/subs. Modern compilers vectorize this to ~5 SSE ops with
 * -O3 -march=native; the explicit scalar form lets the deterministic build
 * stay reproducible while the throughput build still gets vectorized.
 */
MMATH_FORCE_INLINE Quat quatMultiply(const Quat& a_, const Quat& b_) noexcept {
    return Quat{
        a_.w * b_.x + a_.x * b_.w + a_.y * b_.z - a_.z * b_.y,
        a_.w * b_.y - a_.x * b_.z + a_.y * b_.w + a_.z * b_.x,
        a_.w * b_.z + a_.x * b_.y - a_.y * b_.x + a_.z * b_.w,
        a_.w * b_.w - a_.x * b_.x - a_.y * b_.y - a_.z * b_.z
    };
}

/**
 * @brief Conjugate q* = (-x, -y, -z, w)
 *
 * For a unit quaternion this is the inverse rotation.
 */
constexpr MMATH_FORCE_INLINE Quat quatConjugate(const Quat& q_) noexcept {
    return Quat{-q_.x, -q_.y, -q_.z, q_.w};
}

/**
 * @brief Inverse: conjugate / |q|^2.
 *
 * For unit quaternions, prefer quatConjugate() (same result, no divide).
 * @warning Undefined if |q| is zero.
 */
MMATH_FORCE_INLINE Quat quatInverse(const Quat& q_) noexcept {
    const float len_sq = quatLengthSquared(q_);
    const float inv = 1.0f / len_sq;
    return Quat{-q_.x * inv, -q_.y * inv, -q_.z * inv, q_.w * inv};
}

/**
 * @brief Dot product a . b
 */
constexpr MMATH_FORCE_INLINE float quatDot(const Quat& a_, const Quat& b_) noexcept {
    return a_.x * b_.x + a_.y * b_.y + a_.z * b_.z + a_.w * b_.w;
}

/**
 * @brief Rotate a Vec3 by a unit quaternion using Fabian Giesen's 3-cross form.
 *
 *   t = 2 * cross(q.xyz, v)
 *   v' = v + q.w * t + cross(q.xyz, t)
 *
 * Equivalent to q * (v, 0) * q* but with only 18 muls and 12 adds, no
 * quaternion multiply, no branches.
 *
 * @note Assumes q is unit-length. For non-unit q, normalize first.
 */
MMATH_FORCE_INLINE Vec3 quatRotateVec3(const Quat& q_, const Vec3& v_) noexcept {
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

/**
 * @brief Rotate the (x, y, z) lanes of a Vec4 by a unit quaternion.
 *
 * The w lane is passed through unchanged. Provided for PhysicsCenter's SoA /
 * Vec4 column layout so calls do not have to repack data.
 */
MMATH_FORCE_INLINE Vec4 quatRotateVec4(const Quat& q_, const Vec4& v_) noexcept {
    const float tx = 2.0f * (q_.y * v_.z - q_.z * v_.y);
    const float ty = 2.0f * (q_.z * v_.x - q_.x * v_.z);
    const float tz = 2.0f * (q_.x * v_.y - q_.y * v_.x);

    const float cx = q_.y * tz - q_.z * ty;
    const float cy = q_.z * tx - q_.x * tz;
    const float cz = q_.x * ty - q_.y * tx;

    return Vec4{
        v_.x + q_.w * tx + cx,
        v_.y + q_.w * ty + cy,
        v_.z + q_.w * tz + cz,
        v_.w
    };
}

// ============================================================================
// Tier 4: Convenience and component-wise ops
// ============================================================================

/**
 * @brief Component-wise add (treats quaternions as 4-vectors).
 *
 * @note Quaternion addition is *not* rotation composition (use quatMultiply
 *       for that). Useful for blending tangent-space increments such as
 *       q + 0.5*dt*omega_quat*q in user-written integrators.
 */
constexpr MMATH_FORCE_INLINE Quat quatAdd(const Quat& a_, const Quat& b_) noexcept {
    return Quat{a_.x + b_.x, a_.y + b_.y, a_.z + b_.z, a_.w + b_.w};
}

constexpr MMATH_FORCE_INLINE Quat quatSub(const Quat& a_, const Quat& b_) noexcept {
    return Quat{a_.x - b_.x, a_.y - b_.y, a_.z - b_.z, a_.w - b_.w};
}

constexpr MMATH_FORCE_INLINE Quat quatScale(const Quat& q_, float s_) noexcept {
    return Quat{q_.x * s_, q_.y * s_, q_.z * s_, q_.w * s_};
}

constexpr MMATH_FORCE_INLINE Quat quatNegate(const Quat& q_) noexcept {
    return Quat{-q_.x, -q_.y, -q_.z, -q_.w};
}

/**
 * @brief Length |q| = sqrt(x^2 + y^2 + z^2 + w^2) using MMath::sqrt.
 */
MMATH_FORCE_INLINE float quatLength(const Quat& q_) noexcept {
    return MMath::sqrt(quatLengthSquared(q_));
}

/**
 * @brief Fast normalize using raw rsqrt (no Newton refinement).
 *
 * About 11-12 bit accuracy; ~2x throughput vs quatNormalize on older CPUs.
 * Not Deterministic-Profile compliant: do not use in deterministic builds.
 *
 * @warning Undefined behavior if |q| is zero.
 */
MMATH_FORCE_INLINE Quat quatNormalizeFast(const Quat& q_) noexcept {
#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
    const float len_sq = quatLengthSquared(q_);
    const __m128 inv = _mm_rsqrt_ss(_mm_set_ss(len_sq));
    const float inv_len = _mm_cvtss_f32(inv);
    return Quat{q_.x * inv_len, q_.y * inv_len, q_.z * inv_len, q_.w * inv_len};
#else
    return quatNormalize(q_);
#endif
}

// ============================================================================
// Comparison helpers
// ============================================================================

/**
 * @brief Exact equality (component-wise, no epsilon).
 */
constexpr MMATH_FORCE_INLINE bool quatEqual(const Quat& a_, const Quat& b_) noexcept {
    return a_.x == b_.x && a_.y == b_.y && a_.z == b_.z && a_.w == b_.w;
}

/**
 * @brief Near-equality with absolute epsilon, accounting for q == -q.
 *
 * In rotation-space q and -q represent the same orientation, so callers that
 * care about orientation rather than raw values should pass
 * @c orientation_aware=true (default).
 */
inline bool quatNearEqual(
    const Quat& a_,
    const Quat& b_,
    float epsilon_ = 1e-5f,
    bool orientation_aware_ = true
) noexcept {
    auto same = [&](const Quat& l_, const Quat& r_) {
        const float dx = l_.x - r_.x;
        const float dy = l_.y - r_.y;
        const float dz = l_.z - r_.z;
        const float dw = l_.w - r_.w;
        const float diff_sq = dx * dx + dy * dy + dz * dz + dw * dw;
        return diff_sq <= epsilon_ * epsilon_;
    };
    if (same(a_, b_)) {
        return true;
    }
    if (!orientation_aware_) {
        return false;
    }
    return same(a_, quatNegate(b_));
}
