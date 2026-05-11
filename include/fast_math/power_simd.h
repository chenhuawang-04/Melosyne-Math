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

#pragma once

#include "types.h"
#include "power.h"  // For scalar fallback

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__SSE4_1__)
#include <smmintrin.h>
#endif

namespace MMath {
#include <fast_math/detail/power_simd.inl>
} // namespace MMath
