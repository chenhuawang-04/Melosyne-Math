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
#include <smmintrin.h>

export module fast_math.aabb2;

export import fast_math.types;
export import fast_math.vec2;
export import fast_math.mat3;
export import fast_math.aabb2_simd;

export {
/**
 * @file aabb2.h
 * @brief 2D Axis-Aligned Bounding Box (AABB) operations
 *
 * Design Philosophy:
 * - Strict POD structure (NO member functions)
 * - Internal optimization: stores -maxx, -maxy for SIMD efficiency
 * - User-friendly API: accessors hide internal representation
 * - Single-instruction Union, 3-instruction Overlap test
 *
 * Key Optimization:
 * By storing maximum coordinates as negative values, union operations
 * become pure min operations, and overlap tests are highly optimized.
 *
 * Reference: https://mtsamis.com/blog/23_11_07_aabbs
 */



#if defined(__SSE4_1__)
#endif

namespace MMath {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winclude-angled-in-module-purview"
#endif
#include "fast_math/detail/aabb2_core.inl"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
} // namespace MMath

// SIMD batch helpers are provided by imported fast_math.aabb2_simd.
namespace MMath {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winclude-angled-in-module-purview"
#endif
#include "fast_math/detail/aabb2_batch.inl"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
} // namespace MMath
}
