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

export module fast_math.bit_ops_count_simd;

export import fast_math.bitset_view;
export import fast_math.bit_ops_core;
export import fast_math.bit_ops_count;

export {
/**
 * @file bit_ops_count_simd.h
 * @brief SIMD-optimized bit counting operations (AVX2)
 *
 * Design Philosophy:
 * - Shared header/module implementation lives in `detail/bit_ops_count_simd_detail.inl`
 * - AVX2 accelerates wide counting kernels while preserving scalar fallbacks
 * - Dispatch thresholds are centralized in `detail/bit_ops_dispatch_policy.inl`
 */

#if defined(__AVX2__)
#endif

namespace Melosyne {
namespace BitOps {
#include "fast_math/detail/bit_ops_count_simd_detail.inl"
} // namespace BitOps
} // namespace Melosyne
}
