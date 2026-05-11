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

export module fast_math.common_simd;

export import fast_math.types;

export {
/**
 * @file common_simd.h
 * @brief SIMD batch processing for common functions
 *
 * Architecture:
 * - AVX2 path: Process 8 floats per iteration
 * - SSE4.1 path: Process 4 floats per iteration
 * - Scalar fallback: Process remaining elements
 *
 * Performance:
 * - AVX2: ~8x speedup vs scalar loop
 * - SSE4.1: ~4x speedup vs scalar loop
 *
 * Use cases (from audio engine analysis):
 * - Volume limiting for 48000 samples: clampArray()
 * - Peak detection: absArray()
 * - Mixing buffer size calculation: minArray()
 */



#if defined(__AVX2__)
#elif defined(__SSE4_1__)
#endif

namespace MMath {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winclude-angled-in-module-purview"
#endif
#include "fast_math/detail/common_simd_detail.inl"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
} // namespace MMath
}
