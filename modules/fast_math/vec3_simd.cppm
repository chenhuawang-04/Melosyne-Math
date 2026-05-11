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
#include <emmintrin.h>
#include <pmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

export module fast_math.vec3_simd;

export import fast_math.types;
export import fast_math.vec3;

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-WTU-local-entity-exposure"
#endif

export {
/**
 * @file vec3_simd.h
 * @brief SIMD-optimized 3D vector operations
 *
 * This file provides two levels of SIMD optimization:
 *
 * 1. Vec3A (16-byte aligned Vec3) - Single-value SIMD operations
 *    - Uses __m128 for storage, enabling direct SIMD operations
 *    - 25% wasted space (4th lane unused) but much faster for SIMD ops
 *    - Best for: transform pipelines, physics calculations
 *
 * 2. Batch processing APIs - Vertical SIMD on Vec3 arrays
 *    - Uses AoS-to-SoA deinterleaving for maximum throughput
 *    - Processes 8 Vec3s at once with AVX2, 4 with SSE
 *    - Best for: particle systems, vertex processing, mass operations
 *
 * Design Philosophy:
 * - Strict POD structures (NO member functions)
 * - Compatible with existing Vec3 via conversion functions
 * - Fallback scalar implementations when SIMD not available
 *
 * References:
 * - DirectXMath XMVector3* functions
 * - GLM simd/geometric.h
 * - glam-rs Vec3A implementation
 * - "Optimising path tracing with SIMD" by bitshifter
 */



#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
#endif

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#endif

#if defined(__SSE3__)
#endif

#if defined(__SSE4_1__)
#endif

#if defined(__AVX__)
#endif

namespace MMath {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winclude-angled-in-module-purview"
#endif
#include "fast_math/detail/vec3a_core.inl"
#include "fast_math/detail/vec3_batch_simd.inl"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
} // namespace MMath
}
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
