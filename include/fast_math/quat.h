/**
 * @file quat.h
 * @brief High-performance quaternion operations (Strict POD + Free Functions)
 *
 * Locked Conventions
 * ------------------
 *   - layout       : Quat{x, y, z, w}, w is the scalar/real part
 *   - identity     : {0, 0, 0, 1}
 *   - convention   : Hamilton, right-handed
 *   - composition  : quatMultiply(a, b) applies b first then a, so
 *                    quatRotateVec3(quatMultiply(a, b), v) ==
 *                    quatRotateVec3(a, quatRotateVec3(b, v))
 *   - angles       : radians
 *
 * Design Philosophy
 * -----------------
 *   - STRICT POD structures (no member functions)
 *   - FREE FUNCTIONS only, free of allocation and exceptions
 *   - Hot-path operations (quatIntegrate, quatNormalize, quatMultiply,
 *     quatRotateVec3, quatToMat3) marked MMATH_FORCE_INLINE + noexcept,
 *     branch-light, and Deterministic-Profile compliant.
 *   - Deterministic Profile: scalar paths route through MMath::sqrt and
 *     MMath::inverseSqrt; std::sqrt and raw _mm_rsqrt_ss are not used in
 *     the canonical formulas. Quaternion expressions are explicitly
 *     parenthesised so /fp:precise and -fno-fast-math builds preserve bit
 *     identity between runs and translation units.
 *
 * Tiers
 * -----
 *   Tier 1 - foundations:     quatIdentity, quatLengthSquared, quatNormalize,
 *                             quatNormalizeSafe, quatIntegrate (Vec3 + Vec4)
 *   Tier 2 - core algebra:    quatMultiply, quatConjugate, quatInverse,
 *                             quatDot, quatRotateVec3, quatRotateVec4
 *   Tier 3 - conversions:     quatToMat3, quatToMat4, quatFromAxisAngle(Safe),
 *                             quatToAxisAngle, quatSlerp, quatNlerp,
 *                             quatFromEuler, quatToEuler
 *   Tier 4 - convenience:     quatAdd, quatSub, quatScale, quatNegate,
 *                             quatLength, quatNormalizeFast
 *
 * References
 * ----------
 *   - K. Shoemake, "Animating Rotation with Quaternion Curves", SIGGRAPH 1985
 *   - F. Giesen, "A faster quaternion-vector multiplication"
 *     https://fgiesen.wordpress.com/2019/02/09/rotating-a-single-vector-using-a-quaternion/
 *   - D. Baraff, A. Witkin, "Physically Based Modeling: Principles and Practice",
 *     SIGGRAPH 1997 Course Notes (rigid-body integration with q_dot = 0.5*omega*q)
 *   - DirectXMath XMQuaternion* and GLM glm::quat for cross-validation
 */

#pragma once

#include "types.h"
#include "vec3.h"
#include "vec4.h"
#include "mat3.h"
#include "mat4.h"
#include "sqrt.h"
#include <cmath>

#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
#include <xmmintrin.h>
#endif

namespace MMath {
#include <fast_math/detail/quat_core.inl>
#include <fast_math/detail/quat_convert.inl>
} // namespace MMath

// SIMD batch processing and throughput-oriented helpers
#include "quat_simd.h"
