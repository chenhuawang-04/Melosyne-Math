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
#include <xmmintrin.h>
#include <smmintrin.h>

export module fast_math.quat;

export import fast_math.types;
export import fast_math.vec3;
export import fast_math.vec4;
export import fast_math.mat3;
export import fast_math.mat4;
export import fast_math.sqrt;
export import fast_math.quat_simd;

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-WTU-local-entity-exposure"
#endif

export {
/**
 * @file quat.h
 * @brief High-performance quaternion operations (Strict POD + Free Functions)
 *
 * Locked Conventions
 * ------------------
 *   - layout       : Quat{x, y, z, w}, w is the scalar/real part
 *   - identity     : {0, 0, 0, 1}
 *   - convention   : Hamilton, right-handed
 *   - composition  : quatMultiply(a, b) applies b first then a
 *   - angles       : radians
 *
 * Module mirror of include/fast_math/quat.h. See that header for the full
 * doxygen reference; this file simply re-exports the same scalar+conversion
 * detail headers inside the module's namespace.
 */

#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
#endif

namespace MMath {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winclude-angled-in-module-purview"
#endif
#include "fast_math/detail/quat_core.inl"
#include "fast_math/detail/quat_convert.inl"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
} // namespace MMath

// SIMD batch helpers are provided by imported fast_math.quat_simd.
}
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
