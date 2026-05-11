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

export module fast_math.vec2;

export import fast_math.types;
export import fast_math.vec2_simd;

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-WTU-local-entity-exposure"
#endif

export {
/**
 * @file vec2.h
 * @brief High-performance 2D vector library (Strict POD + Hand-written SIMD)
 */

#if defined(__AVX2__) && defined(__FMA__)
#elif defined(__SSE4_1__)
#endif

namespace MMath {

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winclude-angled-in-module-purview"
#endif
#include "fast_math/detail/vec2_core.inl"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

} // namespace MMath (close before including vec2_simd.h)

namespace MMath {

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winclude-angled-in-module-purview"
#endif
#include "fast_math/detail/vec2_array_dispatch.inl"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

} // namespace MMath
}
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
