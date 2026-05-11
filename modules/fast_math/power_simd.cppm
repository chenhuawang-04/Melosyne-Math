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

export module fast_math.power_simd;

export import fast_math.types;
export import fast_math.power;

export {
/**
 * @file power_simd.h
 * @brief SIMD batch processing for exponential and power functions
 *
 * Architecture:
 * - AVX2 path: Process 8 floats per iteration
 * - SSE4.1 path: Process 4 floats per iteration
 * - Scalar fallback: Process remaining elements
 *
 * Performance:
 * - expArray: ~4-6x speedup vs scalar loop
 * - logArray: ~3-5x speedup vs scalar loop
 * - powArray: ~5-8x speedup vs scalar loop
 *
 * Use cases (from audio engine analysis):
 * - Batch dB to linear conversion (pow10Array)
 * - Exponential decay envelopes (expArray)
 * - Distance attenuation calculations (powArray)
 */



#if defined(__AVX2__)
#elif defined(__SSE4_1__)
#endif

namespace MMath {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winclude-angled-in-module-purview"
#endif
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 5244)
#endif
#include "fast_math/detail/power_simd.inl"
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
} // namespace MMath
}
