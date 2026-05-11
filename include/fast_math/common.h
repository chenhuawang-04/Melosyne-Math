/**
 * @file common.h
 * @brief Common mathematical functions (min, max, clamp, abs, sign)
 *
 * Design Philosophy:
 * - STRICT POD structures (NO member functions)
 * - FREE FUNCTIONS only
 * - Scalar versions use SIMD scalar instructions (branch-free)
 * - Array versions dispatch to SIMD batch processing
 *
 * Performance Optimizations:
 * - min/max/clamp: Branch-free SSE scalar instructions (minss/maxss)
 * - abs: Bit manipulation (AND with 0x7FFFFFFF mask to clear sign bit)
 * - sign: Simple comparison (compiler optimizes to branchless code)
 *
 * References:
 * - GLM common functions: https://github.com/g-truc/glm/blob/master/glm/detail/func_common.inl
 * - DirectXMath optimization: https://learn.microsoft.com/en-us/windows/win32/dxmath/pg-xnamath-optimizing
 * - SIMD bit tricks: https://hackaday.com/2024/12/22/faster-integer-division-with-floating-point/
 */

#pragma once

#include "types.h"
#include <cstring>  // for memcpy (type punning)
#include <cmath>    // for fabs, copysign

#if defined(__SSE4_1__)
#include <smmintrin.h>
#endif

namespace MMath {
#include <fast_math/detail/common_core.inl>
} // namespace MMath

// Include SIMD batch processing implementations
#include "common_simd.h"
