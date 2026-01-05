/**
 * @file bit_ops_core.h
 * @brief Scalar bitwise logical operations (no SIMD intrinsics)
 *
 * Design Philosophy:
 * - FREE FUNCTIONS only (no methods)
 * - Scalar versions for single bitset operations
 * - Simple loops (compiler auto-vectorization)
 * - For batch/array operations, use bit_ops_core_simd.h
 *
 * Performance:
 * - Suitable for small bitsets (<= 256 bits)
 * - Compiler may auto-vectorize loops
 * - Zero overhead compared to member functions
 *
 * Usage:
 * @code
 * BitSet<256> a = {}, b = {};
 * bitOpsAnd(a, b);  // a &= b (scalar version)
 * @endcode
 */

#pragma once

#include "bitset_view.h"
#include <cstring>
#include <algorithm>

namespace Melosyne {
namespace BitOps {

// ============================================================================
// Single Bit Operations
// ============================================================================

/**
 * @brief Set a single bit to 1
 * @param view Bitset view
 * @param pos Bit position (0-based)
 */
inline void set(BitSetView view, std::size_t pos) noexcept {
    view.data[pos / 64] |= (uint64_t{1} << (pos % 64));
}

/**
 * @brief Clear a single bit to 0
 * @param view Bitset view
 * @param pos Bit position (0-based)
 */
inline void reset(BitSetView view, std::size_t pos) noexcept {
    view.data[pos / 64] &= ~(uint64_t{1} << (pos % 64));
}

/**
 * @brief Flip a single bit
 * @param view Bitset view
 * @param pos Bit position (0-based)
 */
inline void flip(BitSetView view, std::size_t pos) noexcept {
    view.data[pos / 64] ^= (uint64_t{1} << (pos % 64));
}

/**
 * @brief Test a single bit
 * @param view Bitset view
 * @param pos Bit position (0-based)
 * @return true if bit is set, false otherwise
 */
[[nodiscard]] inline bool test(ConstBitSetView view, std::size_t pos) noexcept {
    return (view.data[pos / 64] >> (pos % 64)) & 1;
}

// ============================================================================
// Bulk Bitwise Operations
// ============================================================================

/**
 * @brief Bitwise AND: dst &= src (scalar version)
 * @param dst Destination bitset (modified in-place)
 * @param src Source bitset (read-only)
 * @note Scalar loop - suitable for small bitsets
 */
inline void bitwiseAnd(BitSetView dst, ConstBitSetView src) noexcept {
    std::size_t min_words = std::min(dst.word_count, src.word_count);
    
    for (std::size_t i = 0; i < min_words; ++i) {
        dst.data[i] &= src.data[i];
    }
    
    for (std::size_t i = min_words; i < dst.word_count; ++i) {
        dst.data[i] = 0;
    }
}

/**
 * @brief Bitwise OR: dst |= src (scalar version)
 */
inline void bitwiseOr(BitSetView dst, ConstBitSetView src) noexcept {
    std::size_t min_words = std::min(dst.word_count, src.word_count);
    
    for (std::size_t i = 0; i < min_words; ++i) {
        dst.data[i] |= src.data[i];
    }
}

/**
 * @brief Bitwise XOR: dst ^= src (scalar version)
 */
inline void bitwiseXor(BitSetView dst, ConstBitSetView src) noexcept {
    std::size_t min_words = std::min(dst.word_count, src.word_count);
    
    for (std::size_t i = 0; i < min_words; ++i) {
        dst.data[i] ^= src.data[i];
    }
}

/**
 * @brief Bitwise NOT: dst = ~dst (scalar version)
 */
inline void bitwiseNot(BitSetView dst) noexcept {
    for (std::size_t i = 0; i < dst.word_count; ++i) {
        dst.data[i] = ~dst.data[i];
    }
    
    // Clear unused bits
    if (dst.bit_count > 0) {
        std::size_t unused_bits = dst.word_count * 64 - dst.bit_count;
        if (unused_bits > 0) {
            dst.data[dst.word_count - 1] &= ~(~uint64_t{0} << (64 - unused_bits));
        }
    }
}

/**
 * @brief Bitwise AND-NOT: dst &= ~src (scalar version)
 */
inline void bitwiseAndNot(BitSetView dst, ConstBitSetView src) noexcept {
    std::size_t min_words = std::min(dst.word_count, src.word_count);
    
    for (std::size_t i = 0; i < min_words; ++i) {
        dst.data[i] &= ~src.data[i];
    }
}

/**
 * @brief Set all bits to 1 (scalar version)
 */
inline void setAll(BitSetView view) noexcept {
    for (std::size_t i = 0; i < view.word_count; ++i) {
        view.data[i] = ~uint64_t{0};
    }
    
    if (view.bit_count > 0) {
        std::size_t unused_bits = view.word_count * 64 - view.bit_count;
        if (unused_bits > 0) {
            view.data[view.word_count - 1] &= ~(~uint64_t{0} << (64 - unused_bits));
        }
    }
}

/**
 * @brief Reset all bits to 0 (scalar version)
 */
inline void resetAll(BitSetView view) noexcept {
    for (std::size_t i = 0; i < view.word_count; ++i) {
        view.data[i] = 0;
    }
}

/**
 * @brief Flip all bits (scalar version)
 */
inline void flipAll(BitSetView view) noexcept {
    bitwiseNot(view);
}

/**
 * @brief Copy bitset: dst = src (scalar version)
 */
inline void copy(BitSetView dst, ConstBitSetView src) noexcept {
    std::size_t min_words = std::min(dst.word_count, src.word_count);
    
    for (std::size_t i = 0; i < min_words; ++i) {
        dst.data[i] = src.data[i];
    }
    
    for (std::size_t i = min_words; i < dst.word_count; ++i) {
        dst.data[i] = 0;
    }
}

/**
 * @brief Test equality: a == b (scalar version)
 */
[[nodiscard]] inline bool equal(ConstBitSetView a, ConstBitSetView b) noexcept {
    if (a.bit_count != b.bit_count) {
        return false;
    }
    
    for (std::size_t i = 0; i < a.word_count; ++i) {
        if (a.data[i] != b.data[i]) {
            return false;
        }
    }
    
    return true;
}

} // namespace BitOps
} // namespace Melosyne
