/**
 * @file vec2_simd.h
 * @brief Hand-written SIMD Vec2 batch processing (AOS deinterleave optimization)
 *
 * Optimization Strategy:
 * - Deinterleave AOS Vec2 to SoA using AVX2 shuffle/unpack
 * - Process 8 Vec2s at once with AVX2
 * - Re-interleave results back to AOS
 *
 * References:
 * - AOS SIMD Vectorization: https://arxiv.org/abs/1806.05713
 * - Deinterleave with AVX2: https://stackoverflow.com/questions/35871639/how-to-deinterleave-image-channel-in-sse
 * - Intel Unpack Intrinsics: https://www.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-intel-advanced-vector-extensions/intrinsics-for-unpack-and-interleave-operations.html
 */

#pragma once

#include "types.h"
#include <cmath>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace MMath {
#include <fast_math/detail/vec2_simd_detail.inl>
} // namespace MMath
