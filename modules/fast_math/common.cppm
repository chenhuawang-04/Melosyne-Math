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

export module fast_math.common;

export import fast_math.types;
export import fast_math.common_simd;

export {
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



#if defined(__SSE4_1__)
#endif

namespace MMath {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winclude-angled-in-module-purview"
#endif
#include "fast_math/detail/common_core.inl"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
} // namespace MMath

// Include SIMD batch processing implementations
}
