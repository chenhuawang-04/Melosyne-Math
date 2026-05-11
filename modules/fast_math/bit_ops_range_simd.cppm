module;

#include "fast_math/config_macros.h"
#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace Melosyne {
namespace BitOps {
#include "fast_math/detail/bit_ops_dispatch_policy.inl"
} // namespace BitOps
} // namespace Melosyne

export module fast_math.bit_ops_range_simd;

export import fast_math.bit_ops_range;
export import fast_math.bit_ops_count_simd;

export {
/**
 * @file bit_ops_range_simd.h
 * @brief SIMD-optimized bit range operations (AVX2)
 *
 * Design Philosophy:
 * - Shared header/module implementation lives in `detail/bit_ops_range_simd_detail.inl`
 * - AVX2 is reserved for bulk range spans with enough full 64-bit words
 * - Dispatch thresholds are centralized in `detail/bit_ops_dispatch_policy.inl`
 */

#if defined(__AVX2__)
#endif

namespace Melosyne {
namespace BitOps {
#include "fast_math/detail/bit_ops_range_simd_detail.inl"
} // namespace BitOps
} // namespace Melosyne
}
