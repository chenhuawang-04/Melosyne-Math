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

export module fast_math.aabb3;

export import fast_math.types;
export import fast_math.vec3;
export import fast_math.vec4;
export import fast_math.mat4;

export {
/**
 * @file aabb3.h
 * @brief 3D Axis-Aligned Bounding Box (AABB) operations
 *
 * Design Philosophy:
 * - Strict POD structure (NO member functions)
 * - Internal optimization: stores -maxx, -maxy, -maxz for SIMD efficiency
 * - User-friendly API: accessors hide internal representation
 * - Single-instruction Union, optimized Overlap test
 *
 * Key Optimization:
 * By storing maximum coordinates as negative values, union operations
 * become pure min operations, and overlap tests are highly optimized.
 *
 * Reference: https://mtsamis.com/blog/23_11_07_aabbs (2D version)
 * Extended to 3D with same optimization strategy
 */



#if defined(__SSE4_1__)
#endif

namespace MMath {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winclude-angled-in-module-purview"
#endif
#include "fast_math/detail/aabb3_core.inl"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
} // namespace MMath
}
