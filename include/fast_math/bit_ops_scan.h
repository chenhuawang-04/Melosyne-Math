/**
 * @file bit_ops_scan.h
 * @brief Scalar bit scanning and position finding operations
 *
 * Design Philosophy:
 * - FREE FUNCTIONS only (no methods)
 * - Uses hardware TZCNT/LZCNT instructions (BMI1)
 * - std::countr_zero/countl_zero compile to single-cycle instructions
 * - Early-exit optimizations for search operations
 *
 * Implemented Operations:
 * - findFirst/findLast: Find first/last set bit
 * - findNext/findPrev: Find next/previous set bit after/before position
 * - findFirstZero/findLastZero: Find zero bits
 * - countTrailingZeros/countLeadingZeros: Count leading/trailing zeros
 * - select: Find k-th set bit (0-indexed)
 * - rank: Count set bits up to position
 * - BitPositionIterator: Range-based for loop support
 *
 * Performance:
 * - TZCNT/LZCNT: Single-cycle latency on modern CPUs (Haswell+)
 * - Early-exit: Best-case O(1), worst-case O(n)
 * - Iterator: Zero-overhead abstraction
 *
 * Usage:
 * @code
 * BitSet<256> bs = {};
 * std::size_t pos = BitOps::findFirst(bs);
 *
 * // Range-based for loop
 * for (std::size_t pos : BitOps::BitPositionIterator(bs)) {
 *     // Process each set bit
 * }
 * @endcode
 *
 * References:
 * - BMI1 Instructions: https://www.intel.com/content/www/us/en/docs/intrinsics-guide
 * - Bit Twiddling Hacks: https://graphics.stanford.edu/~seander/bithacks.html
 */

#pragma once

#include "bitset_view.h"
#include <bit>
#include <cstdint>
#include <iterator>

namespace Melosyne {
namespace BitOps {
#include <fast_math/detail/bit_ops_scan_core.inl>
} // namespace BitOps
} // namespace Melosyne

#include <fast_math/detail/bitops_namespace_alias.inl>
