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

export module fast_math.quat_simd;

export import fast_math.types;
export import fast_math.sqrt;

export {
/**
 * @file quat_simd.h
 * @brief Batch / throughput-oriented quaternion operations.
 *
 * Module mirror of include/fast_math/quat_simd.h.
 */

#if defined(__AVX2__) && defined(__FMA__)
#elif defined(__SSE4_1__)
#elif defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
#endif

namespace MMath {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winclude-angled-in-module-purview"
#endif
#include "fast_math/detail/quat_simd_detail.inl"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
} // namespace MMath
}
