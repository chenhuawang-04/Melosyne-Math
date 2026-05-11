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

export module fast_math.aabb2_simd;

export import fast_math.types;

export {
/**
 * @file aabb2_simd.h
 * @brief AVX2 batch processing for AABB2 operations
 *
 * Optimization Strategy:
 * - Process 8 AABBs at once with AVX2
 * - Leverage negative max storage for minimal shuffle overhead
 * - Preprocess loop-invariant reference AABBs
 */



#if defined(__AVX2__) && defined(__FMA__)
#endif

namespace MMath {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winclude-angled-in-module-purview"
#endif
#include "fast_math/detail/aabb2_simd_detail.inl"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
} // namespace MMath
}
