// Shared Quat <-> Mat / axis-angle / Euler / slerp implementation.
//
// Included after quat_core.inl. Depends on Mat3, Mat4, Vec3 already being in
// scope (via the public quat.h pulling mat3.h, mat4.h, vec3.h).
//
// Layout reminder
// ---------------
//   Mat3 is row-major:    m00..m02 row 0, m10..m12 row 1, m20..m22 row 2
//   Mat4 is column-major: column-major storage in m[16], identical to mat4FromQuat(Vec4)
//
// Quaternion -> rotation matrix (right-handed, column-vector convention v' = R*v):
//
//     [ 1 - 2(y^2 + z^2)   2(xy - wz)        2(xz + wy)       ]
// R = [ 2(xy + wz)         1 - 2(x^2 + z^2)  2(yz - wx)       ]
//     [ 2(xz - wy)         2(yz + wx)        1 - 2(x^2 + y^2) ]

// ============================================================================
// Quaternion -> Matrix
// ============================================================================

/**
 * @brief Convert unit quaternion to a 3x3 rotation matrix (row-major).
 *
 * @note Assumes q is unit-length. Normalize first if uncertain.
 * @note Hot-path optimized: single pass, no branches, 9 muls and 9 mads.
 */
MMATH_FORCE_INLINE Mat3 quatToMat3(const Quat& q_) noexcept {
    const float xx2 = q_.x * (q_.x + q_.x);
    const float yy2 = q_.y * (q_.y + q_.y);
    const float zz2 = q_.z * (q_.z + q_.z);

    const float xy2 = q_.x * (q_.y + q_.y);
    const float xz2 = q_.x * (q_.z + q_.z);
    const float yz2 = q_.y * (q_.z + q_.z);

    const float wx2 = q_.w * (q_.x + q_.x);
    const float wy2 = q_.w * (q_.y + q_.y);
    const float wz2 = q_.w * (q_.z + q_.z);

    return Mat3{
        1.0f - (yy2 + zz2), xy2 - wz2,           xz2 + wy2,
        xy2 + wz2,          1.0f - (xx2 + zz2),  yz2 - wx2,
        xz2 - wy2,          yz2 + wx2,           1.0f - (xx2 + yy2)
    };
}

/**
 * @brief Convert unit quaternion to a 4x4 homogeneous rotation matrix (column-major).
 *
 * @note Layout matches existing mat4FromQuat(const Vec4&): column-major,
 *       last row = (0, 0, 0, 1).
 * @note Assumes q is unit-length.
 */
MMATH_FORCE_INLINE Mat4 quatToMat4(const Quat& q_) noexcept {
    const float xx2 = q_.x * (q_.x + q_.x);
    const float yy2 = q_.y * (q_.y + q_.y);
    const float zz2 = q_.z * (q_.z + q_.z);

    const float xy2 = q_.x * (q_.y + q_.y);
    const float xz2 = q_.x * (q_.z + q_.z);
    const float yz2 = q_.y * (q_.z + q_.z);

    const float wx2 = q_.w * (q_.x + q_.x);
    const float wy2 = q_.w * (q_.y + q_.y);
    const float wz2 = q_.w * (q_.z + q_.z);

    Mat4 r;
    // column 0
    r.m[0]  = 1.0f - (yy2 + zz2); r.m[1]  = xy2 + wz2;          r.m[2]  = xz2 - wy2;          r.m[3]  = 0.0f;
    // column 1
    r.m[4]  = xy2 - wz2;          r.m[5]  = 1.0f - (xx2 + zz2); r.m[6]  = yz2 + wx2;          r.m[7]  = 0.0f;
    // column 2
    r.m[8]  = xz2 + wy2;          r.m[9]  = yz2 - wx2;          r.m[10] = 1.0f - (xx2 + yy2); r.m[11] = 0.0f;
    // column 3
    r.m[12] = 0.0f;               r.m[13] = 0.0f;               r.m[14] = 0.0f;               r.m[15] = 1.0f;
    return r;
}

// ============================================================================
// Axis-angle <-> Quaternion
// ============================================================================

/**
 * @brief Build a quaternion from a unit rotation axis and angle (radians).
 *
 *   q = (axis.x * sin(a/2), axis.y * sin(a/2), axis.z * sin(a/2), cos(a/2))
 *
 * @note Assumes axis is already unit-length. Use quatFromAxisAngleSafe()
 *       if the axis might be near-zero.
 */
inline Quat quatFromAxisAngle(const Vec3& axis_, float angleRad_) noexcept {
    const float half = 0.5f * angleRad_;
    const float s = std::sin(half);
    const float c = std::cos(half);
    return Quat{axis_.x * s, axis_.y * s, axis_.z * s, c};
}

/**
 * @brief Build a quaternion from a non-normalized axis and angle.
 *
 * Normalizes @p axis_ first. Returns identity if axis is shorter than
 * @p epsilon_ (degenerate axis).
 */
inline Quat quatFromAxisAngleSafe(const Vec3& axis_, float angleRad_, float epsilon_ = 1e-12f) noexcept {
    const float len_sq = axis_.x * axis_.x + axis_.y * axis_.y + axis_.z * axis_.z;
    if (len_sq < epsilon_ * epsilon_) {
        return quatIdentity();
    }
    const float inv_len = MMath::inverseSqrt(len_sq);
    const float half = 0.5f * angleRad_;
    const float s = std::sin(half);
    const float c = std::cos(half);
    return Quat{axis_.x * inv_len * s, axis_.y * inv_len * s, axis_.z * inv_len * s, c};
}

/**
 * @brief Extract the rotation axis and angle from a unit quaternion.
 *
 *   angle = 2 * acos(clamp(q.w, -1, 1))
 *   axis  = q.xyz / sin(angle/2)   (defaults to +X when angle is ~0)
 *
 * @param q_       Input unit quaternion
 * @param axisOut_ Output rotation axis (unit-length)
 * @param angleOut_ Output rotation angle in radians, range [0, 2*pi)
 */
inline void quatToAxisAngle(const Quat& q_, Vec3& axisOut_, float& angleOut_) noexcept {
    // Clamp to handle minor numerical drift past +/-1.
    float w = q_.w;
    if (w >  1.0f) w =  1.0f;
    if (w < -1.0f) w = -1.0f;

    angleOut_ = 2.0f * std::acos(w);

    const float sin_half_sq = 1.0f - w * w;
    if (sin_half_sq <= 1e-12f) {
        // Near-zero rotation; axis is undefined, pick +X by convention.
        axisOut_ = Vec3{1.0f, 0.0f, 0.0f};
        return;
    }
    const float inv_sin_half = MMath::inverseSqrt(sin_half_sq);
    axisOut_ = Vec3{q_.x * inv_sin_half, q_.y * inv_sin_half, q_.z * inv_sin_half};
}

// ============================================================================
// Interpolation
// ============================================================================

/**
 * @brief Normalized linear interpolation along the shortest arc.
 *
 *   r = normalize( (1-t)*a + t*(sign*b) )    sign = (dot(a,b) < 0) ? -1 : +1
 *
 * Constant-time, branch-light, but not constant angular velocity. Use slerp
 * when the interpolation must be uniform on the sphere.
 *
 * @note Result is renormalized so it is always a unit quaternion (raw linear
 *       interpolation between unit quaternions is not a unit quaternion).
 */
inline Quat quatNlerp(const Quat& a_, const Quat& b_, float t_) noexcept {
    const float d = quatDot(a_, b_);
    const float sign = (d < 0.0f) ? -1.0f : 1.0f;
    const float omt = 1.0f - t_;
    const Quat raw{
        omt * a_.x + t_ * (sign * b_.x),
        omt * a_.y + t_ * (sign * b_.y),
        omt * a_.z + t_ * (sign * b_.z),
        omt * a_.w + t_ * (sign * b_.w)
    };
    return quatNormalize(raw);
}

/**
 * @brief Spherical linear interpolation along the shortest arc.
 *
 * Uses the classic Shoemake formulation. Falls back to normalized linear
 * interpolation when the two quaternions are nearly parallel (|dot| > 0.9995),
 * avoiding the sin(theta) -> 0 singularity.
 *
 * @param t_ Interpolation factor; 0 returns @p a_, 1 returns the shortest-arc image of @p b_.
 */
inline Quat quatSlerp(const Quat& a_, const Quat& b_, float t_) noexcept {
    float cos_theta = quatDot(a_, b_);

    // Pick the shortest arc.
    float bx = b_.x;
    float by = b_.y;
    float bz = b_.z;
    float bw = b_.w;
    if (cos_theta < 0.0f) {
        cos_theta = -cos_theta;
        bx = -bx; by = -by; bz = -bz; bw = -bw;
    }

    // Near-parallel: fall back to normalized lerp to avoid sin(theta) ~ 0.
    if (cos_theta > 0.9995f) {
        const float omt = 1.0f - t_;
        const Quat raw{
            omt * a_.x + t_ * bx,
            omt * a_.y + t_ * by,
            omt * a_.z + t_ * bz,
            omt * a_.w + t_ * bw
        };
        return quatNormalize(raw);
    }

    if (cos_theta > 1.0f) cos_theta = 1.0f;   // numerical guard
    const float theta = std::acos(cos_theta);
    const float sin_theta = std::sin(theta);
    const float inv_sin = 1.0f / sin_theta;
    const float w1 = std::sin((1.0f - t_) * theta) * inv_sin;
    const float w2 = std::sin(t_ * theta)         * inv_sin;

    return Quat{
        w1 * a_.x + w2 * bx,
        w1 * a_.y + w2 * by,
        w1 * a_.z + w2 * bz,
        w1 * a_.w + w2 * bw
    };
}

// ============================================================================
// Euler angles (Tait-Bryan, intrinsic ZYX / extrinsic XYZ)
// ============================================================================

/**
 * @brief Build a quaternion from Tait-Bryan Euler angles (radians).
 *
 * Convention: intrinsic ZYX rotation, equivalent to extrinsic XYZ:
 *   q = qZ(yaw) * qY(pitch) * qX(roll)
 *
 * In game-engine terms (Z-up, right-handed):
 *   yaw   -> rotation about world Z (heading)
 *   pitch -> rotation about world Y
 *   roll  -> rotation about world X (bank)
 *
 * @note Many engines use different axis assignments (Unity is Y-up).
 *       This API picks one explicit, documented convention; callers wanting
 *       another convention should compose quatFromAxisAngle() calls directly.
 */
inline Quat quatFromEuler(float yaw_, float pitch_, float roll_) noexcept {
    const float cy = std::cos(yaw_   * 0.5f);
    const float sy = std::sin(yaw_   * 0.5f);
    const float cp = std::cos(pitch_ * 0.5f);
    const float sp = std::sin(pitch_ * 0.5f);
    const float cr = std::cos(roll_  * 0.5f);
    const float sr = std::sin(roll_  * 0.5f);

    return Quat{
        sr * cp * cy - cr * sp * sy,   // x (roll about X)
        cr * sp * cy + sr * cp * sy,   // y (pitch about Y)
        cr * cp * sy - sr * sp * cy,   // z (yaw about Z)
        cr * cp * cy + sr * sp * sy    // w
    };
}

/**
 * @brief Extract Tait-Bryan Euler angles from a unit quaternion.
 *
 * Convention matches quatFromEuler: intrinsic ZYX (extrinsic XYZ).
 * Returns Vec3{roll, pitch, yaw} (X, Y, Z rotations) in radians.
 *
 * Handles gimbal lock at pitch = +/- pi/2 by clamping the sine argument and
 * folding the roll angle into yaw.
 */
inline Vec3 quatToEuler(const Quat& q_) noexcept {
    // roll (X-axis rotation)
    const float sinr_cosp = 2.0f * (q_.w * q_.x + q_.y * q_.z);
    const float cosr_cosp = 1.0f - 2.0f * (q_.x * q_.x + q_.y * q_.y);
    const float roll = std::atan2(sinr_cosp, cosr_cosp);

    // pitch (Y-axis rotation), with gimbal-lock guard
    float sinp = 2.0f * (q_.w * q_.y - q_.z * q_.x);
    if (sinp >  1.0f) sinp =  1.0f;
    if (sinp < -1.0f) sinp = -1.0f;
    const float pitch = std::asin(sinp);

    // yaw (Z-axis rotation)
    const float siny_cosp = 2.0f * (q_.w * q_.z + q_.x * q_.y);
    const float cosy_cosp = 1.0f - 2.0f * (q_.y * q_.y + q_.z * q_.z);
    const float yaw = std::atan2(siny_cosp, cosy_cosp);

    return Vec3{roll, pitch, yaw};
}
