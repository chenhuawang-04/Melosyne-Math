/**
 * @file aabb2_simd.h
 * @brief AVX2 batch processing for AABB2 operations
 *
 * Optimization Strategy:
 * - Process 8 AABBs at once with AVX2
 * - Leverage negative max storage for minimal shuffle overhead
 * - Preprocess loop-invariant reference AABBs
 */

#pragma once

#include "types.h"
#include <cmath>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace MMath {
#include <fast_math/detail/aabb2_simd_detail.inl>
} // namespace MMath
