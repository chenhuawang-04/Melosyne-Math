/**
 * @file mat4.h
 * @brief SIMD-optimized 4x4 matrix operations
 *
 * Design Philosophy:
 * - STRICT POD structures (same as mat4.h)
 * - Column-major layout (canonical RH + zero-to-one depth helpers)
 * - Hand-optimized SIMD for maximum performance
 * - Fallback to scalar version when SIMD unavailable
 *
 * Performance Goals (vs competitors):
 * - mat4Multiply: Beat GLM (4.06ms) and DirectXMath (4.11ms)
 * - mat4MultiplyVec4: Beat DirectXMath (2.04ms)
 * - mat4Transpose: Beat DirectXMath (1.89ms)
 * - mat4TryInverse / mat4InverseChecked: explicit-failure inverse path
 * - mat4Translation: Beat GLM (1.25ms)
 * - mat4LookAt: Beat DirectXMath (1.22ms) - MOST CRITICAL (10x gap)
 *
 * Competitors' SIMD Status:
 * - GLM: FORCE_SIMD_AVX2 (256-bit vectors)
 * - DirectXMath: Native SIMD (SSE2 minimum, XMMATRIX)
 * - Eigen: Expression templates + auto-vectorization
 */

#pragma once

#include "types.h"
#include "vec4.h"
#include "config_macros.h"
#include <cmath>
#include <cstring>

// SIMD intrinsics detection
#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
    #include <xmmintrin.h>  // SSE
    #include <emmintrin.h>  // SSE2
#endif

#if defined(__SSE3__)
    #include <pmmintrin.h>  // SSE3
#endif

#if defined(__SSSE3__)
    #include <tmmintrin.h>  // SSSE3
#endif

#if defined(__SSE4_1__)
    #include <smmintrin.h>  // SSE4.1
#endif

#if defined(__AVX__)
    #include <immintrin.h>  // AVX
#endif

#if defined(__FMA__)
    #ifndef MMATH_SIMD_AVX
        #include <immintrin.h>
    #endif
#endif

namespace MMath {

#include <fast_math/detail/mat4_core.inl>

} // namespace MMath
