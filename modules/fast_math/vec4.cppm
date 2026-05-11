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

export module fast_math.vec4;

export import fast_math.types;

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-WTU-local-entity-exposure"
#endif

export {
/**
 * @file vec4.h
 * @brief High-performance 4D vector operations (Strict POD + Free Functions)
 */

#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
#endif

namespace MMath {

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winclude-angled-in-module-purview"
#endif
#include "fast_math/detail/vec4_core.inl"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

} // namespace MMath
}
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
