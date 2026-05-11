// Shared scalar core bit operations for header/module parity.

namespace detail {

[[nodiscard]] inline constexpr uint64_t coreLowBitsMask(std::size_t bit_count) noexcept {
    if (bit_count == 0) return 0;
    if (bit_count >= 64) return ~uint64_t{0};
    return (uint64_t{1} << bit_count) - 1;
}

inline void coreMaskUnusedBits(BitSetView view) noexcept {
    if (view.word_count == 0) return;
    const std::size_t remainder = view.bit_count % 64;
    if (remainder != 0) {
        view.data[view.word_count - 1] &= coreLowBitsMask(remainder);
    }
}

} // namespace detail

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
    const std::size_t min_words = std::min(dst.word_count, src.word_count);

    for (std::size_t i = 0; i < min_words; ++i) {
        dst.data[i] &= src.data[i];
    }

    for (std::size_t i = min_words; i < dst.word_count; ++i) {
        dst.data[i] = 0;
    }

    detail::coreMaskUnusedBits(dst);
}

/**
 * @brief Bitwise OR: dst |= src (scalar version)
 */
inline void bitwiseOr(BitSetView dst, ConstBitSetView src) noexcept {
    const std::size_t min_words = std::min(dst.word_count, src.word_count);

    for (std::size_t i = 0; i < min_words; ++i) {
        dst.data[i] |= src.data[i];
    }

    detail::coreMaskUnusedBits(dst);
}

/**
 * @brief Bitwise XOR: dst ^= src (scalar version)
 */
inline void bitwiseXor(BitSetView dst, ConstBitSetView src) noexcept {
    const std::size_t min_words = std::min(dst.word_count, src.word_count);

    for (std::size_t i = 0; i < min_words; ++i) {
        dst.data[i] ^= src.data[i];
    }

    detail::coreMaskUnusedBits(dst);
}

/**
 * @brief Bitwise NOT: dst = ~dst (scalar version)
 */
inline void bitwiseNot(BitSetView dst) noexcept {
    for (std::size_t i = 0; i < dst.word_count; ++i) {
        dst.data[i] = ~dst.data[i];
    }

    detail::coreMaskUnusedBits(dst);
}

/**
 * @brief Bitwise AND-NOT: dst &= ~src (scalar version)
 */
inline void bitwiseAndNot(BitSetView dst, ConstBitSetView src) noexcept {
    const std::size_t min_words = std::min(dst.word_count, src.word_count);

    for (std::size_t i = 0; i < min_words; ++i) {
        dst.data[i] &= ~src.data[i];
    }

    detail::coreMaskUnusedBits(dst);
}

/**
 * @brief Set all bits to 1 (scalar version)
 */
inline void setAll(BitSetView view) noexcept {
    for (std::size_t i = 0; i < view.word_count; ++i) {
        view.data[i] = ~uint64_t{0};
    }

    detail::coreMaskUnusedBits(view);
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
    const std::size_t min_words = std::min(dst.word_count, src.word_count);

    for (std::size_t i = 0; i < min_words; ++i) {
        dst.data[i] = src.data[i];
    }

    for (std::size_t i = min_words; i < dst.word_count; ++i) {
        dst.data[i] = 0;
    }

    detail::coreMaskUnusedBits(dst);
}

/**
 * @brief Test equality: a == b (scalar version)
 */
[[nodiscard]] inline bool equal(ConstBitSetView a, ConstBitSetView b) noexcept {
    if (a.bit_count != b.bit_count) {
        return false;
    }

    const std::size_t full_words = a.bit_count / 64;
    for (std::size_t i = 0; i < full_words; ++i) {
        if (a.data[i] != b.data[i]) {
            return false;
        }
    }

    const std::size_t remainder = a.bit_count % 64;
    if (remainder != 0) {
        const uint64_t mask = detail::coreLowBitsMask(remainder);
        return (a.data[full_words] & mask) == (b.data[full_words] & mask);
    }

    return true;
}
