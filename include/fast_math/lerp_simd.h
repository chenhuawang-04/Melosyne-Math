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

#pragma once

#include "types.h"
#include "config_macros.h"

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#elif defined(__SSE4_1__) && defined(__FMA__)
#include <immintrin.h>
#elif defined(__SSE4_1__)
#include <smmintrin.h>
#endif

namespace MMath {
#include <fast_math/detail/lerp_simd_detail.inl>
} // namespace MMath
