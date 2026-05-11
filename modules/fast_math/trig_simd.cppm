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

export module fast_math.trig_simd;

export import fast_math.types;

export {
/**
 * @file trig_simd.h
 * @brief Hand-written SIMD trigonometric functions (DirectXMath style)
 *
 * Optimizations:
 * - Hand-written SSE4.1/AVX2 intrinsics (NO auto-vectorization)
 * - FMA instructions for maximum ILP
 * - Branchless design using SIMD blends
 * - Constant shuffle optimization
 * - DirectXMath 11/10-degree minimax polynomials
 */



#if defined(__AVX2__) && defined(__FMA__)
#elif defined(__SSE4_1__)
#endif

namespace MMath {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winclude-angled-in-module-purview"
#endif
#include "fast_math/detail/trig_simd_detail.inl"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
} // namespace MMath
}
