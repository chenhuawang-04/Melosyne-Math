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
#include <immintrin.h>

namespace Melosyne {
namespace BitOps {
#include "fast_math/detail/bit_ops_dispatch_policy.inl"
} // namespace BitOps
} // namespace Melosyne

export module fast_math.bit_ops_core_simd;

export import fast_math.bitset_view;
export import fast_math.bit_ops_core;

export {
/**
 * @file bit_ops_core_simd.h
 * @brief SIMD-optimized bitwise operations (AVX2/SSE4.1)
 *
 * Design Philosophy:
 * - AVX2/SSE4.1 intrinsics for batch processing
 * - Adaptive strategy: Scalar for small, SIMD for large
 * - In `detail` namespace (internal implementation)
 * - Public API automatically dispatches to SIMD when beneficial
 *
 * Performance:
 * - AVX2: Process 256 bits (4 words) per iteration
 * - AVX2 is reserved for wider bitsets where setup cost is amortized
 * - Aligned memory access remains preferred when available
 *
 * Architecture:
 * - Dispatch thresholds are centralized in `detail/bit_ops_dispatch_policy.inl`
 * - Core bitwise ops currently switch at >= 128 words
 * - Compiler auto-vectorization continues to handle the scalar path
 */



#if defined(__AVX2__)
#endif

namespace Melosyne {
namespace BitOps {
#include "fast_math/detail/bit_ops_core_simd_detail.inl"
} // namespace BitOps
} // namespace Melosyne
}
