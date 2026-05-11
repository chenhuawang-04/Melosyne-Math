/**
 * @file vec2.h
 * @brief High-performance 2D vector library (Strict POD + Hand-written SIMD)
 *
 * Design Philosophy:
 * - STRICT POD structures (NO member functions)
 * - Hand-written AVX2/SSE4.1 SIMD intrinsics
 * - Optimized for batch processing with deinterleave
 * - Compatible with DirectXMath/GLM performance
 *
 * References:
 * - DirectXMath XMVector2* functions
 * - SIMD Dot Product Optimization: https://github.com/segfaultscribe/SIMD-Dot-product-Optimization
 * - Fast SIMD practices: https://lemire.me/blog/2018/07/05/how-quickly-can-you-compute-the-dot-product-between-two-large-vectors/
 * - AOS Vectorization: https://arxiv.org/abs/1806.05713
 */

#pragma once

#include "types.h"
#include <cmath>
#include <cstring>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#elif defined(__SSE4_1__)
#include <smmintrin.h>
#elif defined(__SSE__)
#include <xmmintrin.h>
#endif

namespace MMath {

#include <fast_math/detail/vec2_core.inl>

} // namespace MMath (close before including vec2_simd.h)

#include "vec2_simd.h"

namespace MMath {

#include <fast_math/detail/vec2_array_dispatch.inl>

} // namespace MMath
