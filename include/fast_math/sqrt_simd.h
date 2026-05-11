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

#pragma once

#include "types.h"
#include <cmath>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__SSE4_1__)
#include <smmintrin.h>
#endif

namespace MMath {
#include <fast_math/detail/sqrt_simd_detail.inl>
} // namespace MMath
