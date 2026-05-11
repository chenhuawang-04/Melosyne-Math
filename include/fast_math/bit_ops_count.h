/**
 * @file bit_ops_count.h
 * @brief Scalar bit counting and statistical operations (no SIMD intrinsics)
 *
 * Design Philosophy:
 * - FREE FUNCTIONS only (no methods)
 * - Scalar versions using std::popcount and simple loops
 * - Compiler may auto-vectorize loops
 * - For SIMD-optimized versions, use bit_ops_count_simd.h
 *
 * Implemented Operations:
 * - popcount: Count set bits (uses hardware POPCNT instruction via std::popcount)
 * - bitwiseAndCount: Fused in-place AND + count kernel
 * - hammingDistance: Count differing bits between two bitsets
 * - any/none/all: Query if bits are set
 * - unionCount/intersectionCount: Set operation cardinality
 * - jaccardSimilarity/diceCoefficient: Similarity metrics
 * - isSubsetOf/intersects: Set relationship tests
 *
 * Performance:
 * - Suitable for small/medium bitsets
 * - std::popcount uses hardware POPCNT (SSE4.2+)
 * - Early-exit optimization for query operations
 *
 * Usage:
 * @code
 * BitSet<256> a = {}, b = {};
 * std::size_t count = BitOps::popcount(a);
 * bool equal_bits = (BitOps::hammingDistance(a, b) == 0);
 * @endcode
 */

#pragma once

#include "bitset_view.h"
#include <bit>
#include <cstdint>
#include <algorithm>

namespace Melosyne {
namespace BitOps {
#include <fast_math/detail/bit_ops_count_core.inl>
} // namespace BitOps
} // namespace Melosyne

#include <fast_math/detail/bitops_namespace_alias.inl>
