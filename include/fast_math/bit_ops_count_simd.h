/**
 * @file bit_ops_count_simd.h
 * @brief SIMD-optimized bit counting operations (AVX2)
 *
 * Design Philosophy:
 * - Shared header/module implementation lives in `detail/bit_ops_count_simd_detail.inl`
 * - AVX2 accelerates wide counting kernels while preserving scalar fallbacks
 * - Dispatch thresholds are centralized in `detail/bit_ops_dispatch_policy.inl`
 *
 * Optimized Operations:
 * - popcountOptimized
 * - bitwiseAndCountOptimized
 * - hammingDistanceOptimized
 * - allOptimized
 * - unionCountOptimized / intersectionCountOptimized
 * - jaccardSimilarityOptimized / diceCoefficientOptimized
 */

#pragma once

#include "config_macros.h"
#include "bitset_view.h"
#include "bit_ops_core.h"
#include "bit_ops_count.h"

#include <algorithm>
#include <bit>
#include <cstdint>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace Melosyne {
namespace BitOps {
#include "detail/bit_ops_dispatch_policy.inl"
#include <fast_math/detail/bit_ops_count_simd_detail.inl>
} // namespace BitOps
} // namespace Melosyne

#include <fast_math/detail/bitops_namespace_alias.inl>
