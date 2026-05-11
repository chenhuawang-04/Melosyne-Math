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

export module fast_math.sqrt_simd;

export import fast_math.types;

export {
/**
 * @file sqrt_simd.h
 * @brief SIMD batch processing for square root functions
 *
 * Architecture:
 * - AVX2 path: Process 8 floats per iteration
 * - SSE4.1 path: Process 4 floats per iteration
 * - Scalar fallback: Process remaining elements
 *
 * Performance:
 * - sqrtArray: Same as std::sqrt loop (hardware instruction)
 * - inverseSqrtArray: ~2-3x speedup vs 1/std::sqrt loop
 * - reciprocalArray: ~2x speedup vs 1/x loop (on older CPUs)
 *
 * Use cases (from audio engine analysis):
 * - Vector normalization for large arrays
 * - Distance calculations for spatial audio
 * - Batch lighting attenuation
 */



#if defined(__AVX2__)
#elif defined(__SSE4_1__)
#endif

namespace MMath {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winclude-angled-in-module-purview"
#endif
#include "fast_math/detail/sqrt_simd_detail.inl"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
} // namespace MMath
}
