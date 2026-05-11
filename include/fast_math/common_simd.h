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

#pragma once

#include "types.h"
#include <cmath>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__SSE4_1__)
#include <smmintrin.h>
#endif

namespace MMath {
#include <fast_math/detail/common_simd_detail.inl>
} // namespace MMath
