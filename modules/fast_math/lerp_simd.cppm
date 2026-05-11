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

export module fast_math.lerp_simd;

export import fast_math.types;

export {
/**
 * @file lerp_simd.h
 * @brief SIMD batch processing for interpolation functions
 *
 * Architecture:
 * - AVX2 + FMA path: Process 8 floats per iteration with fused operations
 * - SSE4.1 + FMA path: Process 4 floats per iteration with fused operations
 * - SSE4.1 path: Process 4 floats per iteration (standard mul+add)
 * - Scalar fallback: Process remaining elements
 *
 * Performance:
 * - lerpArray: ~1.5-2x speedup with FMA vs standard
 * - smoothstepArray: ~4-8x speedup vs scalar loop
 * - Batch processing amortizes loop overhead
 *
 * Use cases (from audio engine analysis):
 * - Fractional delay line interpolation (read between samples)
 * - Crossfade between audio buffers
 * - Parameter smoothing over time
 */



#if defined(__AVX2__) && defined(__FMA__)
#elif defined(__AVX2__)
#elif defined(__SSE4_1__) && defined(__FMA__)
#elif defined(__SSE4_1__)
#endif

namespace MMath {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winclude-angled-in-module-purview"
#endif
#include "fast_math/detail/lerp_simd_detail.inl"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
} // namespace MMath
}
