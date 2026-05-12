/**
 * @file quat_simd.h
 * @brief Batch / throughput-oriented quaternion operations.
 *
 * Single-quaternion operations live in quat.h (scalar formulas, marked
 * MMATH_FORCE_INLINE) because modern compilers auto-vectorize them better
 * than hand-rolled intrinsics on the 16-byte aligned Quat layout.
 *
 * This file exposes the *array* / batch variants where the throughput win
 * is real (physics tick, animation blending, GPU upload paths):
 *
 *   - quatMultiplyArray
 *   - quatNormalizeArray / quatNormalizeSafeArray
 *   - quatRotateVec3Array
 *   - quatIntegrateArray (Vec3 / Vec4 angular velocity overloads)
 *
 * Includes quat_simd_detail.inl which currently relies on compiler
 * auto-vectorization. Hand-rolled SoA AVX2/SSE2/NEON paths can be added
 * behind the same public API without touching call sites.
 */

#pragma once

#include "types.h"
#include "sqrt.h"
#include <cstdint>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#elif defined(__SSE4_1__)
#include <smmintrin.h>
#elif defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
#include <xmmintrin.h>
#endif

namespace MMath {
#include <fast_math/detail/quat_simd_detail.inl>
} // namespace MMath
