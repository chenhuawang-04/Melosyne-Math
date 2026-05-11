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

#pragma once

#include "types.h"
#include "vec2.h"
#include "mat3.h"
#include <cmath>
#include <algorithm>
#include <limits>

#if defined(__SSE4_1__)
#include <smmintrin.h>
#endif

namespace MMath {
#include <fast_math/detail/aabb2_core.inl>
} // namespace MMath

// Include SIMD batch processing implementations
#include "aabb2_simd.h"

namespace MMath {
#include <fast_math/detail/aabb2_batch.inl>
} // namespace MMath
