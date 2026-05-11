/**
 * @file bit_ops_scan_simd.h
 * @brief SIMD-optimized bit scanning operations (limited optimizations)
 *
 * Design Philosophy:
 * - Most scan operations are inherently sequential (early-exit)
 * - SIMD benefits are limited compared to scalar TZCNT/LZCNT
 * - SIMD optimizations focus on select() and rank() operations
 * - For findFirst/findLast/findNext/findPrev, scalar is optimal
 *
 * SIMD-Optimized Operations:
 * - selectSimd: Parallel popcount accumulation with AVX2
 * - rankSimd: Parallel popcount for full words with AVX2
 *
 * Operations WITHOUT SIMD optimization (scalar is faster):
 * - findFirst/findLast: Single TZCNT/LZCNT is 1-cycle, SIMD overhead > benefit
 * - findNext/findPrev: Early-exit advantage, SIMD cannot help
 * - findFirstZero/findLastZero: Same as find operations
 *
 * Performance:
 * - select (SIMD): ~1.5x faster for large bitsets with many set bits
 * - rank (SIMD): ~2x faster for large positions (many full words to count)
 * - Other operations: Scalar version is optimal
 *
 * References:
 * - Zhou et al. "Fast Parallel Computation of Rank and Select"
 * - Vigna "Broadword Implementation of Rank/Select Queries"
 */

#pragma once

#include "config_macros.h"
#include "bitset_view.h"
#include "bit_ops_scan.h"
#include "bit_ops_count_simd.h"  // For popcount256
#include <bit>
#include <cstdint>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace Melosyne {
namespace BitOps {
#include "detail/bit_ops_dispatch_policy.inl"
#include <fast_math/detail/bit_ops_scan_simd_detail.inl>
// ============================================================================
// Note on other operations
// ============================================================================

/*
 * The following operations do NOT have SIMD versions because scalar is optimal:
 *
 * - findFirst/findLast:
 *   TZCNT/LZCNT are 1-cycle instructions
 *   SIMD overhead (loading, masking, movemask) > scalar benefit
 *
 * - findNext/findPrev:
 *   Early-exit is the key optimization
 *   SIMD processes full vectors, cannot early-exit mid-vector
 *
 * - findFirstZero/findLastZero:
 *   Same reasoning as findFirst/findLast
 *
 * For these operations, always use the scalar versions from bit_ops_scan.h
 *
 * Example:
 * @code
 * // These are already optimal (scalar)
 * std::size_t first = findFirst(bs);    // 1-cycle TZCNT
 * std::size_t last = findLast(bs);      // 1-cycle LZCNT
 * std::size_t next = findNext(bs, pos); // Early-exit search
 * @endcode
 */

} // namespace BitOps
} // namespace Melosyne

#include <fast_math/detail/bitops_namespace_alias.inl>
