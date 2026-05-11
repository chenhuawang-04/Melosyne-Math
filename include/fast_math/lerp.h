/**
 * @file lerp.h
 * @brief Linear interpolation and smooth curve functions
 *
 * Design Philosophy:
 * - STRICT POD structures (NO member functions)
 * - FREE FUNCTIONS only
 * - Scalar versions use FMA (Fused Multiply-Add) optimization
 * - Array versions dispatch to SIMD batch processing
 *
 * Performance Optimizations:
 * - lerp: FMA-optimized formula (5-30% faster than standard)
 * - smoothstep/smootherstep: Horner's method for polynomial evaluation
 * - All functions designed for branch-free execution
 *
 * Accuracy:
 * - FMA provides single rounding step (better precision than separate mul+add)
 * - Numerical stability for edge cases (t=0, t=1)
 *
 * References:
 * - NVIDIA FMA-optimized lerp: https://developer.nvidia.com/blog/lerp-faster-cuda/
 * - Ryg's linear interpolation analysis: https://fgiesen.wordpress.com/2012/08/15/linear-interpolation-past-present-and-future/
 * - GLM mix implementation: https://github.com/g-truc/glm/blob/master/glm/detail/func_common.inl
 * - Smoothstep mathematics: https://en.wikipedia.org/wiki/Smoothstep
 */

#pragma once

#include "types.h"
#include "common.h"  // for clamp
#include <cmath>     // for std::floor

#if defined(__FMA__)
#include <immintrin.h>
#endif

#if defined(__SSE4_1__)
#include <smmintrin.h>
#endif

namespace MMath {
#include <fast_math/detail/lerp_core.inl>
} // namespace MMath

// Include SIMD batch processing implementations
#include "lerp_simd.h"
