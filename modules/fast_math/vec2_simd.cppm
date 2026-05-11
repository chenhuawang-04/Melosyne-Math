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

export module fast_math.vec2_simd;

export import fast_math.types;

export {
/**
 * @file vec2_simd.h
 * @brief Hand-written SIMD Vec2 batch processing (AOS deinterleave optimization)
 *
 * Optimization Strategy:
 * - Deinterleave AOS Vec2 to SoA using AVX2 shuffle/unpack
 * - Process 8 Vec2s at once with AVX2
 * - Re-interleave results back to AOS
 *
 * References:
 * - AOS SIMD Vectorization: https://arxiv.org/abs/1806.05713
 * - Deinterleave with AVX2: https://stackoverflow.com/questions/35871639/how-to-deinterleave-image-channel-in-sse
 * - Intel Unpack Intrinsics: https://www.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-intel-advanced-vector-extensions/intrinsics-for-unpack-and-interleave-operations.html
 */



#if defined(__AVX2__) && defined(__FMA__)
#endif

namespace MMath {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winclude-angled-in-module-purview"
#endif
#include "fast_math/detail/vec2_simd_detail.inl"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
} // namespace MMath
}
