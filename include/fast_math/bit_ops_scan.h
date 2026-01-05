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

// ============================================================================
// Find first set bit (forward scan from LSB)
// ============================================================================

/**
 * @brief Find the first (lowest) set bit in a bitset
 * @param view Bitset view (read-only)
 * @return Index of first set bit, or view.bit_count if not found
 *
 * Algorithm:
 * - Linear scan through words until non-zero word found
 * - Use TZCNT (std::countr_zero) on non-zero word
 *
 * Performance:
 * - TZCNT: 1 cycle latency (Haswell+)
 * - Best case: O(1) - first word has set bit
 * - Worst case: O(n) - only last word has set bit
 *
 * Example:
 * @code
 * BitSet<256> bs = {};
 * bs.words[0] = 0b10100;  // Bits 2 and 4 set
 * findFirst(bs);  // Returns 2
 * @endcode
 */
[[nodiscard]] inline std::size_t findFirst(ConstBitSetView view) noexcept {
    for (std::size_t i = 0; i < view.word_count; ++i) {
        if (view.data[i] != 0) {
            return i * 64 + std::countr_zero(view.data[i]);
        }
    }
    return view.bit_count;
}

/**
 * @brief Find the last (highest) set bit in a bitset
 * @param view Bitset view (read-only)
 * @return Index of last set bit, or view.bit_count if not found
 *
 * Algorithm:
 * - Reverse scan through words from MSB
 * - Use LZCNT (std::countl_zero) on non-zero word
 *
 * Performance:
 * - LZCNT: 1 cycle latency (Haswell+)
 * - Best case: O(1) - last word has set bit
 * - Worst case: O(n) - only first word has set bit
 */
[[nodiscard]] inline std::size_t findLast(ConstBitSetView view) noexcept {
    for (std::size_t i = view.word_count; i > 0; --i) {
        std::size_t idx = i - 1;
        if (view.data[idx] != 0) {
            std::size_t leading_zeros = std::countl_zero(view.data[idx]);
            return idx * 64 + (63 - leading_zeros);
        }
    }
    return view.bit_count;
}

/**
 * @brief Find next set bit after given position
 * @param view Bitset view
 * @param pos Starting position (search begins at pos + 1)
 * @return Index of next set bit, or view.bit_count if not found
 *
 * Algorithm:
 * 1. Mask out bits <= pos in current word
 * 2. If remaining bits in word, use TZCNT
 * 3. Otherwise, scan subsequent words
 *
 * Use case: Iterate through set bits
 * @code
 * for (size_t pos = findFirst(bs); pos < bs.size(); pos = findNext(bs, pos)) {
 *     // Process bit at pos
 * }
 * @endcode
 */
[[nodiscard]] inline std::size_t findNext(
    ConstBitSetView view, std::size_t pos) noexcept {

    ++pos;
    if (pos >= view.bit_count) return view.bit_count;

    std::size_t word_idx = pos / 64;
    std::size_t bit_idx = pos % 64;

    // Check remaining bits in current word
    uint64_t word = view.data[word_idx] & (~uint64_t{0} << bit_idx);
    if (word != 0) {
        std::size_t result = word_idx * 64 + std::countr_zero(word);
        return result < view.bit_count ? result : view.bit_count;
    }

    // Search remaining words
    for (++word_idx; word_idx < view.word_count; ++word_idx) {
        if (view.data[word_idx] != 0) {
            std::size_t result = word_idx * 64 + std::countr_zero(view.data[word_idx]);
            return result < view.bit_count ? result : view.bit_count;
        }
    }

    return view.bit_count;
}

/**
 * @brief Find previous set bit before given position
 * @param view Bitset view
 * @param pos Starting position (search begins at pos - 1)
 * @return Index of previous set bit, or view.bit_count if not found
 *
 * Algorithm:
 * 1. Mask out bits >= pos in current word
 * 2. If remaining bits in word, use LZCNT
 * 3. Otherwise, scan previous words in reverse
 */
[[nodiscard]] inline std::size_t findPrev(
    ConstBitSetView view, std::size_t pos) noexcept {

    if (pos == 0 || pos > view.bit_count) return view.bit_count;

    --pos;
    std::size_t word_idx = pos / 64;
    std::size_t bit_idx = pos % 64;

    // Check bits in current word up to bit_idx
    uint64_t mask = (uint64_t{1} << (bit_idx + 1)) - 1;
    uint64_t word = view.data[word_idx] & mask;
    if (word != 0) {
        return word_idx * 64 + (63 - std::countl_zero(word));
    }

    // Search previous words
    for (std::size_t i = word_idx; i > 0; --i) {
        std::size_t idx = i - 1;
        if (view.data[idx] != 0) {
            return idx * 64 + (63 - std::countl_zero(view.data[idx]));
        }
    }

    return view.bit_count;
}

// ============================================================================
// Find zero bits (inverse operations)
// ============================================================================

/**
 * @brief Find the first (lowest) zero bit
 * @param view Bitset view
 * @return Index of first zero bit, or view.bit_count if all bits are set
 *
 * Algorithm:
 * - Scan for first word != 0xFFFFFFFFFFFFFFFF
 * - Use TZCNT on inverted word: countr_zero(~word)
 */
[[nodiscard]] inline std::size_t findFirstZero(ConstBitSetView view) noexcept {
    for (std::size_t i = 0; i < view.word_count; ++i) {
        if (view.data[i] != ~uint64_t{0}) {
            std::size_t pos = i * 64 + std::countr_zero(~view.data[i]);
            return pos < view.bit_count ? pos : view.bit_count;
        }
    }
    return view.bit_count;
}

/**
 * @brief Find the last (highest) zero bit
 * @param view Bitset view
 * @return Index of last zero bit, or view.bit_count if all bits are set
 */
[[nodiscard]] inline std::size_t findLastZero(ConstBitSetView view) noexcept {
    for (std::size_t i = view.word_count; i > 0; --i) {
        std::size_t idx = i - 1;
        if (view.data[idx] != ~uint64_t{0}) {
            std::size_t leading_zeros = std::countl_zero(~view.data[idx]);
            std::size_t pos = idx * 64 + (63 - leading_zeros);
            return pos < view.bit_count ? pos : view.bit_count;
        }
    }
    return view.bit_count;
}

// ============================================================================
// Count leading/trailing zeros
// ============================================================================

/**
 * @brief Count trailing zeros (position of first set bit)
 * @param view Bitset view
 * @return Position of first set bit, or SIZE_MAX if all bits are zero
 *
 * Note: Returns SIZE_MAX for empty bitset (different from findFirst)
 */
[[nodiscard]] inline std::size_t countTrailingZeros(ConstBitSetView view) noexcept {
    std::size_t pos = findFirst(view);
    return (pos == view.bit_count) ? SIZE_MAX : pos;
}

/**
 * @brief Count leading zeros from highest possible position
 * @param view Bitset view
 * @return Number of leading zeros, or view.bit_count if all bits are zero
 */
[[nodiscard]] inline std::size_t countLeadingZeros(ConstBitSetView view) noexcept {
    std::size_t pos = findLast(view);
    if (pos == view.bit_count) return view.bit_count;
    return view.bit_count - pos - 1;
}

// ============================================================================
// Select operation: Find k-th set bit (0-indexed)
// ============================================================================

/**
 * @brief Find the k-th set bit (0-indexed)
 * @param view Bitset view
 * @param k Index of set bit to find (0 = first set bit, 1 = second, etc.)
 * @return Position of k-th set bit, or view.bit_count if k >= popcount(view)
 *
 * Algorithm:
 * - Accumulate popcount until we reach target k
 * - Scan bit-by-bit within target word
 *
 * Performance:
 * - O(n) where n = number of words
 * - For large bitsets with many set bits, consider SIMD version
 *
 * Example:
 * @code
 * BitSet<256> bs = {};
 * bs.words[0] = 0b10101;  // Bits 0, 2, 4 set
 * select(bs, 0);  // Returns 0
 * select(bs, 1);  // Returns 2
 * select(bs, 2);  // Returns 4
 * @endcode
 */
[[nodiscard]] inline std::size_t select(
    ConstBitSetView view, std::size_t k) noexcept {

    std::size_t count = 0;
    for (std::size_t i = 0; i < view.word_count; ++i) {
        uint64_t word = view.data[i];
        std::size_t word_popcount = std::popcount(word);

        if (count + word_popcount > k) {
            // The k-th bit is in this word
            std::size_t target = k - count;

            // Find the target-th set bit in word
            for (std::size_t j = 0; j < 64; ++j) {
                if (word & (uint64_t{1} << j)) {
                    if (target == 0) {
                        return i * 64 + j;
                    }
                    --target;
                }
            }
        }

        count += word_popcount;
    }

    return view.bit_count;
}

/**
 * @brief Rank operation: Count set bits up to position (exclusive)
 * @param view Bitset view
 * @param pos Position (exclusive upper bound)
 * @return Number of set bits in [0, pos)
 *
 * Algorithm:
 * - Popcount full words before pos
 * - Mask and popcount partial word containing pos
 *
 * Use case: Succinct data structures, rank-select queries
 *
 * Complexity: O(n) where n = pos / 64
 */
[[nodiscard]] inline std::size_t rank(
    ConstBitSetView view, std::size_t pos) noexcept {

    if (pos == 0) return 0;
    if (pos >= view.bit_count) pos = view.bit_count;

    std::size_t word_idx = pos / 64;
    std::size_t bit_idx = pos % 64;
    std::size_t count = 0;

    // Count full words
    for (std::size_t i = 0; i < word_idx; ++i) {
        count += std::popcount(view.data[i]);
    }

    // Count partial word
    if (bit_idx > 0) {
        uint64_t mask = (uint64_t{1} << bit_idx) - 1;
        count += std::popcount(view.data[word_idx] & mask);
    }

    return count;
}

// ============================================================================
// Bit position iterator (for range-based for loops)
// ============================================================================

/**
 * @brief Iterator over set bit positions
 *
 * Enables range-based for loops over set bits:
 * @code
 * for (std::size_t pos : BitOps::BitPositionIterator(bitset)) {
 *     // pos is the index of a set bit
 * }
 * @endcode
 *
 * Implementation:
 * - Forward iterator using findNext()
 * - Zero-overhead abstraction (inline + optimization)
 * - Compatible with STL algorithms
 */
class BitPositionIterator {
public:
    class iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using pointer = const std::size_t*;
        using reference = std::size_t;

        iterator(ConstBitSetView view, std::size_t pos) noexcept
            : view_(view), pos_(pos) {}

        reference operator*() const noexcept { return pos_; }

        iterator& operator++() noexcept {
            pos_ = findNext(view_, pos_);
            return *this;
        }

        iterator operator++(int) noexcept {
            iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        bool operator==(const iterator& other) const noexcept {
            return pos_ == other.pos_;
        }

        bool operator!=(const iterator& other) const noexcept {
            return pos_ != other.pos_;
        }

    private:
        ConstBitSetView view_;
        std::size_t pos_;
    };

    explicit BitPositionIterator(ConstBitSetView view) noexcept
        : view_(view) {}

    iterator begin() const noexcept {
        return iterator(view_, findFirst(view_));
    }

    iterator end() const noexcept {
        return iterator(view_, view_.bit_count);
    }

private:
    ConstBitSetView view_;
};

} // namespace BitOps
} // namespace Melosyne
