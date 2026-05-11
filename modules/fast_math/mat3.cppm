module;

#include "fast_math/config_macros.h"
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <limits>
#include <algorithm>
#include <bit>
#include <iterator>
#include <type_traits>
#include <memory>
#include <memory_resource>
#include <cfloat>
#include <immintrin.h>
#include <smmintrin.h>

export module fast_math.mat3;

export import fast_math.types;
export import fast_math.vec2;
export import fast_math.mat3_simd;

export {
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



#if defined(__AVX2__) && defined(__FMA__)
#elif defined(__SSE4_1__)
#endif

namespace MMath {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winclude-angled-in-module-purview"
#endif
#include "fast_math/detail/mat3_core.inl"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
} // namespace MMath (close before including mat3_simd.h)

// SIMD batch helpers are provided by imported fast_math.mat3_simd.

namespace MMath {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winclude-angled-in-module-purview"
#endif
#include "fast_math/detail/mat3_batch.inl"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
} // namespace MMath
}
