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

#pragma once

#include "types.h"
#include "vec3.h"
#include "vec4.h"
#include "mat4.h"
#include <cmath>
#include <algorithm>
#include <limits>

#if defined(__SSE4_1__)
#include <smmintrin.h>
#endif

namespace MMath {
#include <fast_math/detail/aabb3_core.inl>
} // namespace MMath
