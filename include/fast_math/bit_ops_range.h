/**
 * @file bit_ops_range.h
 * @brief Scalar bit range operations (no SIMD intrinsics)
 *
 * Design Philosophy:
 * - FREE FUNCTIONS only (no methods)
 * - Word-aligned range optimization (set entire words at once)
 * - SWAR masks for partial words
 * - Simple loops for maximum portability
 *
 * Implemented Operations:
 * - setRange/resetRange/flipRange: Modify bits in range [start, end)
 * - testAll/testAny/testNone: Query if all/any/none bits are set in range
 * - popcountRange: Count set bits in range
 *
 * Optimization Techniques:
 * - Handle partial first/last words with bit masks
 * - Process full words in middle with direct assignment
 * - Up to 64x faster than bit-by-bit loops
 *
 * Performance:
 * - Single word range: O(1) - single mask operation
 * - Multi-word range: O(n) where n = number of words in range
 * - Best case: Aligned range, all full words
 *
 * References:
 * - Bit Twiddling Hacks: https://graphics.stanford.edu/~seander/bithacks.html
 * - SWAR techniques: https://aggregate.org/MAGIC/
 */

#pragma once

#include "bitset_view.h"
#include <bit>
#include <cstdint>
#include <algorithm>

namespace Melosyne {
namespace BitOps {

// ============================================================================
// Helper: Create bit mask for range [start, end) within a word
// ============================================================================

/**
 * @brief Create bit mask for range [start_bit, end_bit) within a 64-bit word
 * @param start_bit Starting bit position (inclusive)
 * @param end_bit Ending bit position (exclusive)
 * @return Mask with bits [start_bit, end_bit) set to 1
 *
 * Examples:
 * - createRangeMask(0, 4) = 0b1111 (bits 0-3 set)
 * - createRangeMask(2, 5) = 0b11100 (bits 2-4 set)
 * - createRangeMask(0, 64) = 0xFFFFFFFFFFFFFFFF (all bits set)
 *
 * Edge cases:
 * - start_bit >= 64: returns 0
 * - end_bit > 64: clamps to 64
 * - start_bit >= end_bit: returns 0
 */
[[nodiscard]] inline constexpr uint64_t createRangeMask(
    std::size_t start_bit, std::size_t end_bit) noexcept {

    if (start_bit >= 64) return 0;
    if (end_bit > 64) end_bit = 64;
    if (start_bit >= end_bit) return 0;

    std::size_t length = end_bit - start_bit;
    if (length >= 64) return ~uint64_t{0};

    return ((uint64_t{1} << length) - 1) << start_bit;
}

// ============================================================================
// Set range: set all bits in [start, end) to 1
// ============================================================================

/**
 * @brief Set all bits in range [start, end) to 1 (scalar version)
 * @param view Bitset view (mutable)
 * @param start Starting bit index (inclusive)
 * @param end Ending bit index (exclusive)
 *
 * Algorithm:
 * 1. Handle single-word range with mask
 * 2. Set partial first word (if start not word-aligned)
 * 3. Set full middle words to 0xFFFFFFFFFFFFFFFF
 * 4. Set partial last word (if end not word-aligned)
 *
 * Performance:
 * - O(1) for single-word range
 * - O(n) for multi-word range, where n = number of words
 * - ~64x faster than bit-by-bit loop
 *
 * Example:
 * @code
 * BitSet<256> bs = {};
 * BitOps::setRange(bs, 10, 100);  // Sets bits 10-99
 * @endcode
 */
inline void setRange(BitSetView view, std::size_t start, std::size_t end) noexcept {
    if (start >= end || start >= view.bit_count) return;
    if (end > view.bit_count) end = view.bit_count;

    std::size_t start_word = start / 64;
    std::size_t end_word = (end - 1) / 64; // Inclusive
    std::size_t start_bit = start % 64;
    std::size_t end_bit = end % 64;

    if (start_word == end_word) {
        // Range within single word
        uint64_t mask = createRangeMask(start_bit, end_bit == 0 ? 64 : end_bit);
        view.data[start_word] |= mask;
        return;
    }

    // Set partial first word
    if (start_bit != 0) {
        uint64_t mask = ~uint64_t{0} << start_bit;
        view.data[start_word] |= mask;
        ++start_word;
    }

    // Set complete middle words
    for (std::size_t i = start_word; i < end_word; ++i) {
        view.data[i] = ~uint64_t{0};
    }

    // Set partial last word
    if (end_bit != 0) {
        uint64_t mask = (uint64_t{1} << end_bit) - 1;
        view.data[end_word] |= mask;
    } else if (end_word < view.word_count) {
        view.data[end_word] = ~uint64_t{0};
    }
}

// ============================================================================
// Reset range: set all bits in [start, end) to 0
// ============================================================================

/**
 * @brief Clear all bits in range [start, end) to 0 (scalar version)
 * @param view Bitset view (mutable)
 * @param start Starting bit index (inclusive)
 * @param end Ending bit index (exclusive)
 *
 * Algorithm: Same as setRange, but uses AND with inverted masks
 *
 * Performance: O(1) for single word, O(n) for n words in range
 */
inline void resetRange(BitSetView view, std::size_t start, std::size_t end) noexcept {
    if (start >= end || start >= view.bit_count) return;
    if (end > view.bit_count) end = view.bit_count;

    std::size_t start_word = start / 64;
    std::size_t end_word = (end - 1) / 64;
    std::size_t start_bit = start % 64;
    std::size_t end_bit = end % 64;

    if (start_word == end_word) {
        // Range within single word
        uint64_t mask = createRangeMask(start_bit, end_bit == 0 ? 64 : end_bit);
        view.data[start_word] &= ~mask;
        return;
    }

    // Clear partial first word
    if (start_bit != 0) {
        uint64_t mask = ~uint64_t{0} << start_bit;
        view.data[start_word] &= ~mask;
        ++start_word;
    }

    // Clear complete middle words
    for (std::size_t i = start_word; i < end_word; ++i) {
        view.data[i] = 0;
    }

    // Clear partial last word
    if (end_bit != 0) {
        uint64_t mask = (uint64_t{1} << end_bit) - 1;
        view.data[end_word] &= ~mask;
    } else if (end_word < view.word_count) {
        view.data[end_word] = 0;
    }
}

// ============================================================================
// Flip range: toggle all bits in [start, end)
// ============================================================================

/**
 * @brief Flip (toggle) all bits in range [start, end) (scalar version)
 * @param view Bitset view (mutable)
 * @param start Starting bit index (inclusive)
 * @param end Ending bit index (exclusive)
 *
 * Algorithm: Same as setRange, but uses XOR instead of OR
 *
 * Performance: O(1) for single word, O(n) for n words in range
 */
inline void flipRange(BitSetView view, std::size_t start, std::size_t end) noexcept {
    if (start >= end || start >= view.bit_count) return;
    if (end > view.bit_count) end = view.bit_count;

    std::size_t start_word = start / 64;
    std::size_t end_word = (end - 1) / 64;
    std::size_t start_bit = start % 64;
    std::size_t end_bit = end % 64;

    if (start_word == end_word) {
        uint64_t mask = createRangeMask(start_bit, end_bit == 0 ? 64 : end_bit);
        view.data[start_word] ^= mask;
        return;
    }

    // Flip partial first word
    if (start_bit != 0) {
        uint64_t mask = ~uint64_t{0} << start_bit;
        view.data[start_word] ^= mask;
        ++start_word;
    }

    // Flip complete middle words
    for (std::size_t i = start_word; i < end_word; ++i) {
        view.data[i] = ~view.data[i];
    }

    // Flip partial last word
    if (end_bit != 0) {
        uint64_t mask = (uint64_t{1} << end_bit) - 1;
        view.data[end_word] ^= mask;
    } else if (end_word < view.word_count) {
        view.data[end_word] = ~view.data[end_word];
    }
}

// ============================================================================
// Test range: check if all/any/none bits in [start, end) are set
// ============================================================================

/**
 * @brief Test if all bits in range [start, end) are set to 1 (scalar version)
 * @param view Bitset view (read-only)
 * @param start Starting bit index (inclusive)
 * @param end Ending bit index (exclusive)
 * @return true if all bits in range are 1, false otherwise
 *
 * Algorithm:
 * - For each word segment, check (word & mask) == mask
 * - Early exit on first non-matching word
 *
 * Performance:
 * - Best case: O(1) - first word fails test
 * - Worst case: O(n) - all bits are set
 */
[[nodiscard]] inline bool testAll(
    ConstBitSetView view, std::size_t start, std::size_t end) noexcept {

    if (start >= end || start >= view.bit_count) return true;
    if (end > view.bit_count) end = view.bit_count;

    std::size_t start_word = start / 64;
    std::size_t end_word = (end - 1) / 64;
    std::size_t start_bit = start % 64;
    std::size_t end_bit = end % 64;

    if (start_word == end_word) {
        uint64_t mask = createRangeMask(start_bit, end_bit == 0 ? 64 : end_bit);
        return (view.data[start_word] & mask) == mask;
    }

    // Check partial first word
    if (start_bit != 0) {
        uint64_t mask = ~uint64_t{0} << start_bit;
        if ((view.data[start_word] & mask) != mask) return false;
        ++start_word;
    }

    // Check complete middle words
    for (std::size_t i = start_word; i < end_word; ++i) {
        if (view.data[i] != ~uint64_t{0}) return false;
    }

    // Check partial last word
    if (end_bit != 0) {
        uint64_t mask = (uint64_t{1} << end_bit) - 1;
        return (view.data[end_word] & mask) == mask;
    }

    return view.data[end_word] == ~uint64_t{0};
}

/**
 * @brief Test if any bit in range [start, end) is set to 1 (scalar version)
 * @param view Bitset view (read-only)
 * @param start Starting bit index (inclusive)
 * @param end Ending bit index (exclusive)
 * @return true if at least one bit in range is 1, false otherwise
 *
 * Performance:
 * - Best case: O(1) - first word has set bit
 * - Worst case: O(n) - no bits set (must check all words)
 */
[[nodiscard]] inline bool testAny(
    ConstBitSetView view, std::size_t start, std::size_t end) noexcept {

    if (start >= end || start >= view.bit_count) return false;
    if (end > view.bit_count) end = view.bit_count;

    std::size_t start_word = start / 64;
    std::size_t end_word = (end - 1) / 64;
    std::size_t start_bit = start % 64;
    std::size_t end_bit = end % 64;

    if (start_word == end_word) {
        uint64_t mask = createRangeMask(start_bit, end_bit == 0 ? 64 : end_bit);
        return (view.data[start_word] & mask) != 0;
    }

    // Check partial first word
    if (start_bit != 0) {
        uint64_t mask = ~uint64_t{0} << start_bit;
        if ((view.data[start_word] & mask) != 0) return true;
        ++start_word;
    }

    // Check complete middle words
    for (std::size_t i = start_word; i < end_word; ++i) {
        if (view.data[i] != 0) return true;
    }

    // Check partial last word
    if (end_bit != 0) {
        uint64_t mask = (uint64_t{1} << end_bit) - 1;
        return (view.data[end_word] & mask) != 0;
    }

    return view.data[end_word] != 0;
}

/**
 * @brief Test if no bits in range [start, end) are set (scalar version)
 * @param view Bitset view (read-only)
 * @param start Starting bit index (inclusive)
 * @param end Ending bit index (exclusive)
 * @return true if all bits in range are 0, false otherwise
 */
[[nodiscard]] inline bool testNone(
    ConstBitSetView view, std::size_t start, std::size_t end) noexcept {
    return !testAny(view, start, end);
}

// ============================================================================
// Count set bits in range [start, end)
// ============================================================================

/**
 * @brief Count number of set bits in range [start, end) (scalar version)
 * @param view Bitset view (read-only)
 * @param start Starting bit index (inclusive)
 * @param end Ending bit index (exclusive)
 * @return Number of bits set to 1 in the range
 *
 * Algorithm:
 * - Mask and popcount partial first word
 * - Popcount full middle words
 * - Mask and popcount partial last word
 *
 * Performance:
 * - Uses hardware POPCNT instruction (std::popcount)
 * - O(n) where n = number of words in range
 * - ~3 cycles per word on modern CPUs
 *
 * Example:
 * @code
 * BitSet<256> bs = {};
 * setRange(bs, 0, 100);
 * std::size_t count = popcountRange(bs, 0, 100);  // Returns 100
 * @endcode
 */
[[nodiscard]] inline std::size_t popcountRange(
    ConstBitSetView view, std::size_t start, std::size_t end) noexcept {

    if (start >= end || start >= view.bit_count) return 0;
    if (end > view.bit_count) end = view.bit_count;

    std::size_t start_word = start / 64;
    std::size_t end_word = (end - 1) / 64;
    std::size_t start_bit = start % 64;
    std::size_t end_bit = end % 64;
    std::size_t count = 0;

    if (start_word == end_word) {
        uint64_t mask = createRangeMask(start_bit, end_bit == 0 ? 64 : end_bit);
        return std::popcount(view.data[start_word] & mask);
    }

    // Count partial first word
    if (start_bit != 0) {
        uint64_t mask = ~uint64_t{0} << start_bit;
        count += std::popcount(view.data[start_word] & mask);
        ++start_word;
    }

    // Count complete middle words
    for (std::size_t i = start_word; i < end_word; ++i) {
        count += std::popcount(view.data[i]);
    }

    // Count partial last word
    if (end_bit != 0) {
        uint64_t mask = (uint64_t{1} << end_bit) - 1;
        count += std::popcount(view.data[end_word] & mask);
    } else {
        count += std::popcount(view.data[end_word]);
    }

    return count;
}

} // namespace BitOps
} // namespace Melosyne
