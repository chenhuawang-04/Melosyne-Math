/**
 * @file mat3.h
 * @brief High-performance 3x3 matrix library (Strict POD + Hand-written SIMD)
 *
 * Design Philosophy:
 * - STRICT POD structures (NO member functions)
 * - Hand-written AVX2/SSE4.1 SIMD intrinsics
 * - Optimized for 2D affine transforms
 * - Row-major layout for intuitive operations
 * - Compatible with DirectXMath/GLM performance
 *
 * Matrix Layout (Row-Major):
 * [m00  m01  m02]   [sx*cos  -sy*sin  tx]
 * [m10  m11  m12] = [sx*sin   sy*cos  ty]
 * [m20  m21  m22]   [0        0       1 ] (affine)
 *
 * References:
 * - DirectXMath matrix operations
 * - perf_math SoA approach for batch performance
 * - GLM for API design
 */

#pragma once

#include "types.h"
#include "vec2.h"
#include <cmath>
#include <cstring>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#elif defined(__SSE4_1__)
#include <smmintrin.h>
#endif

namespace MMath {
#include <fast_math/detail/mat3_core.inl>
} // namespace MMath

// Include SIMD implementations after closing namespace
#include "mat3_simd.h"

namespace MMath {
#include <fast_math/detail/mat3_batch.inl>
} // namespace MMath
