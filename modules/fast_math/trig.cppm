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

export module fast_math.trig;

export import fast_math.types;
export import fast_math.trig_simd;

export {
/**
 * @file trig.h
 * @brief High-level trigonometry API (operates on POD structures)
 *
 * Design:
 * - Operates on STRICT POD structures
 * - FREE FUNCTIONS only (NO member functions)
 * - Automatic SIMD dispatch (AVX2 > SSE4.1 > Scalar)
 * - Maximum performance batch processing
 */



namespace MMath {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winclude-angled-in-module-purview"
#endif
#include "fast_math/detail/trig_core.inl"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
} // namespace MMath
}
