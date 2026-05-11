/**
 * @file trig_simd.h
 * @brief Hand-written SIMD trigonometric functions (DirectXMath style)
 *
 * Optimizations:
 * - Hand-written SSE4.1/AVX2 intrinsics (NO auto-vectorization)
 * - FMA instructions for maximum ILP
 * - Branchless design using SIMD blends
 * - Constant shuffle optimization
 * - DirectXMath 11/10-degree minimax polynomials
 */

#pragma once

#include "types.h"
#include <cmath>
#include <cstring>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#elif defined(__SSE4_1__)
#include <smmintrin.h>
#endif

namespace MMath {
#include <fast_math/detail/trig_simd_detail.inl>
} // namespace MMath
