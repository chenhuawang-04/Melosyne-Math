/**
 * @file vec3.h
 * @brief High-performance 3D vector operations (Strict POD + Free Functions)
 *
 * Design Philosophy:
 * - STRICT POD structures (NO member functions)
 * - Scalar implementations optimized for single operations
 * - Compiler auto-vectorization friendly code patterns
 * - Optional SIMD batch processing in vec3_simd.h
 *
 * Performance Strategy:
 * - Vec3 is 12 bytes (not 16-byte aligned), so single SIMD ops have load/store overhead
 * - For single operations: scalar code often equals or beats explicit SIMD
 * - For batch operations: use vec3*Array() functions with SIMD
 *
 * Key Optimizations:
 * - vec3NormalizeFast: uses rsqrt + Newton refinement for throughput-oriented normalization
 * - vec3LengthSquared: avoids sqrt when possible
 * - Inline everything to enable compiler optimizations
 *
 * References:
 * - DirectXMath XMVector3* (performance champion in benchmarks)
 * - GLM geometric.h SIMD implementations
 * - https://github.com/microsoft/DirectXMath
 */

#pragma once

#include "types.h"
#include <cmath>

#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
#include <xmmintrin.h>
#endif

namespace MMath {
#include <fast_math/detail/vec3_core.inl>
} // namespace MMath

// ============================================================================
// Batch Processing (SIMD optimized) - Include after namespace close
// ============================================================================
// TODO: #include "vec3_simd.h" for batch array operations
