/**
 * @file vec4.h
 * @brief High-performance 4D vector operations (Strict POD + Free Functions)
 *
 * Design Philosophy:
 * - STRICT POD structures (NO member functions)
 * - Scalar implementations + compiler auto-vectorization
 * - 16-byte aligned for optimal SIMD register usage
 *
 * Performance Strategy:
 * - Vec4 is 16-byte aligned, natural fit for SIMD registers (__m128)
 * - Uses scalar code to enable superior compiler auto-vectorization
 * - Modern compilers (Clang/GCC -O3 -march=native) outperform explicit SIMD
 * - Avoids load/store overhead of hand-written intrinsics
 *
 * References:
 * - DirectXMath XMVector* functions (best-in-class SIMD implementation)
 * - GLM vec4 (GLSL-compatible API design)
 * - https://github.com/microsoft/DirectXMath
 */

#pragma once

#include "types.h"
#include <cmath>

#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
#include <xmmintrin.h>
#endif

namespace MMath {

#include <fast_math/detail/vec4_core.inl>

} // namespace MMath
