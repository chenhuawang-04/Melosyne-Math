/**
 * @file mat3_simd.h
 * @brief Hand-written SIMD Mat3 batch processing (AOS deinterleave optimization)
 *
 * Optimization Strategy:
 * - Deinterleave AOS Mat3 to temporary SoA using AVX2 shuffle/unpack
 * - Process 8 Mat3s at once with AVX2 + FMA
 * - Re-interleave results back to AOS
 * - Affine transform optimization: skip third row computation
 *
 * Key Challenge: 9 floats (36 bytes) doesn't align with AVX2 (32 bytes)
 * Solution: Load in chunks and use complex shuffle patterns
 *
 * References:
 * - Vec2 deinterleave success (1.17x faster dot product)
 * - perf_math SoA batch processing approach
 * - Intel intrinsics guide for optimal shuffle patterns
 */

#pragma once

#include "types.h"
#include <cmath>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace MMath {
#include <fast_math/detail/mat3_simd_detail.inl>
} // namespace MMath
