module;

#include "fast_math/config_macros.h"
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <limits>
#include <algorithm>
#include <bit>
#include <iterator>
#include <type_traits>
#include <memory>
#include <memory_resource>
#include <cfloat>
#include <xmmintrin.h>

export module fast_math.vec3;

export import fast_math.types;

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-WTU-local-entity-exposure"
#endif

export {
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



#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
#endif

namespace MMath {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winclude-angled-in-module-purview"
#endif
#include "fast_math/detail/vec3_core.inl"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
} // namespace MMath

// ============================================================================
// Batch Processing (SIMD optimized) - Include after namespace close
// ============================================================================
// TODO: #include "vec3_simd.h" for batch array operations
}
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
