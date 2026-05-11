/**
 * @file bit_ops_range_simd.h
 * @brief SIMD-optimized bit range operations (AVX2)
 *
 * Design Philosophy:
 * - Shared header/module implementation lives in `detail/bit_ops_range_simd_detail.inl`
 * - AVX2 is reserved for bulk range spans with enough full 64-bit words
 * - Dispatch thresholds are centralized in `detail/bit_ops_dispatch_policy.inl`
 *
 * Optimized Operations:
 * - setRangeOptimized / resetRangeOptimized / flipRangeOptimized
 * - popcountRangeOptimized
 */

#pragma once

#include "config_macros.h"
#include "bit_ops_range.h"
#include "bit_ops_count_simd.h"

#include <algorithm>
#include <cstdint>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace Melosyne {
namespace BitOps {
#include "detail/bit_ops_dispatch_policy.inl"
#include <fast_math/detail/bit_ops_range_simd_detail.inl>
} // namespace BitOps
} // namespace Melosyne

#include <fast_math/detail/bitops_namespace_alias.inl>
