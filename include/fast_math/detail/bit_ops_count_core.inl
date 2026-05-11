
namespace detail {

[[nodiscard]] constexpr uint64_t countLowBitsMask(std::size_t bits) noexcept {
    if (bits == 0) {
        return 0;
    }
    if (bits >= 64) {
        return ~uint64_t{0};
    }
    return (uint64_t{1} << bits) - 1;
}

} // namespace detail

// ============================================================================
// Population count (count number of set bits)
// ============================================================================

/**
 * @brief Count the number of set bits in a bitset (scalar version)
 * @param view Bitset view (read-only)
 * @return Number of bits set to 1
 *
 * Algorithm:
 * - Uses std::popcount which compiles to hardware POPCNT instruction
 * - Simple loop over all words
 * - Compiler may auto-vectorize for better performance
 *
 * Performance:
 * - Hardware POPCNT: 3 cycle latency, 1 cycle throughput
 * - Suitable for bitsets of any size
 * - For very large bitsets (> 32KB), SIMD version may be faster
 *
 * Complexity: O(n) where n = number of words
 */
[[nodiscard]] inline std::size_t popcount(ConstBitSetView view) noexcept {
    std::size_t total = 0;

    for (std::size_t i = 0; i < view.word_count; ++i) {
        total += std::popcount(view.data[i]);
    }

    return total;
}

/**
 * @brief In-place bitwise AND followed by population count in a single pass
 * @param dst Destination bitset, overwritten with (dst AND src)
 * @param src Source bitset
 * @return Number of set bits in the updated destination
 *
 * This fused scalar kernel avoids a second full memory pass compared to:
 * - bitwiseAnd(dst, src)
 * - popcount(dst)
 */
[[nodiscard]] inline std::size_t bitwiseAndCount(
    BitSetView dst, ConstBitSetView src) noexcept {

    const std::size_t min_words = std::min(dst.word_count, src.word_count);
    const std::size_t remainder_bits = dst.bit_count % 64;
    const bool has_partial_tail = remainder_bits != 0 && min_words == dst.word_count;
    const std::size_t scalar_words = has_partial_tail ? (min_words - 1) : min_words;
    std::size_t count = 0;

    for (std::size_t i = 0; i < scalar_words; ++i) {
        const uint64_t value = dst.data[i] & src.data[i];
        dst.data[i] = value;
        count += std::popcount(value);
    }

    if (has_partial_tail) {
        const std::size_t tail_index = dst.word_count - 1;
        const uint64_t value = (dst.data[tail_index] & src.data[tail_index]) &
            detail::countLowBitsMask(remainder_bits);
        dst.data[tail_index] = value;
        count += std::popcount(value);
    }

    for (std::size_t i = min_words; i < dst.word_count; ++i) {
        dst.data[i] = 0;
    }

    return count;
}
// ============================================================================
// Hamming distance (count differing bits)
// ============================================================================

/**
 * @brief Calculate Hamming distance between two bitsets (scalar version)
 * @param a First bitset
 * @param b Second bitset
 * @return Number of bit positions where a and b differ
 *
 * Algorithm:
 * - Hamming distance = popcount(a XOR b)
 * - Handles different-sized bitsets correctly
 *
 * Performance:
 * - One XOR + one POPCNT per word
 * - ~6-8 cycles per word on modern CPUs
 *
 * Use cases:
 * - Similarity measurement
 * - Error detection/correction
 * - Fuzzy matching
 *
 * Complexity: O(n) where n = max(a.words, b.words)
 */
[[nodiscard]] inline std::size_t hammingDistance(
    ConstBitSetView a, ConstBitSetView b) noexcept {

    std::size_t min_words = std::min(a.word_count, b.word_count);
    std::size_t distance = 0;

    // XOR + popcount for common words
    for (std::size_t i = 0; i < min_words; ++i) {
        distance += std::popcount(a.data[i] ^ b.data[i]);
    }

    // Count remaining bits in longer bitset
    for (std::size_t i = min_words; i < a.word_count; ++i) {
        distance += std::popcount(a.data[i]);
    }
    for (std::size_t i = min_words; i < b.word_count; ++i) {
        distance += std::popcount(b.data[i]);
    }

    return distance;
}

// ============================================================================
// Query operations (any/none/all)
// ============================================================================

/**
 * @brief Check if any bit is set (scalar version)
 * @param view Bitset view
 * @return true if at least one bit is set, false if all bits are 0
 *
 * Algorithm:
 * - Simple linear search with early exit
 * - Returns immediately upon finding first non-zero word
 *
 * Performance:
 * - Best case: O(1) - first word is non-zero
 * - Worst case: O(n) - all words are zero
 * - Avg case: O(n/2) for randomly distributed bits
 *
 * Note: SIMD is NOT faster for this operation due to early-exit advantage
 */
[[nodiscard]] inline bool any(ConstBitSetView view) noexcept {
    for (std::size_t i = 0; i < view.word_count; ++i) {
        if (view.data[i] != 0) return true;
    }
    return false;
}

/**
 * @brief Check if no bits are set (scalar version)
 * @param view Bitset view
 * @return true if all bits are 0, false otherwise
 */
[[nodiscard]] inline bool none(ConstBitSetView view) noexcept {
    return !any(view);
}

/**
 * @brief Check if all bits are set (scalar version)
 * @param view Bitset view
 * @return true if all bits are 1 (within bit_count), false otherwise
 *
 * Algorithm:
 * - Check complete words: must be 0xFFFFFFFFFFFFFFFF
 * - Check partial last word: mask to actual bit count
 * - Early exit on first non-full word
 *
 * Performance:
 * - Best case: O(1) - first word is not all 1s
 * - Worst case: O(n) - all bits are 1
 */
[[nodiscard]] inline bool all(ConstBitSetView view) noexcept {
    if (view.word_count == 0) return true;

    std::size_t complete_words = view.bit_count / 64;

    // Check all complete words
    for (std::size_t i = 0; i < complete_words; ++i) {
        if (view.data[i] != ~uint64_t{0}) return false;
    }

    // Check partial last word if any
    std::size_t remaining_bits = view.bit_count % 64;
    if (remaining_bits > 0) {
        uint64_t mask = (~uint64_t{0}) >> (64 - remaining_bits);
        return (view.data[complete_words] & mask) == mask;
    }

    return true;
}

// ============================================================================
// Set operations with cardinality counting
// ============================================================================

/**
 * @brief Count bits in union (A | B) - scalar version
 * @param a First bitset
 * @param b Second bitset
 * @return Number of bits set in (a OR b)
 *
 * Algorithm:
 * - For each word: popcount(a[i] | b[i])
 * - Add remaining words from longer bitset
 *
 * Use cases:
 * - Set union cardinality
 * - Combined coverage analysis
 */
[[nodiscard]] inline std::size_t unionCount(
    ConstBitSetView a, ConstBitSetView b) noexcept {

    std::size_t min_words = std::min(a.word_count, b.word_count);
    std::size_t count = 0;

    for (std::size_t i = 0; i < min_words; ++i) {
        count += std::popcount(a.data[i] | b.data[i]);
    }

    for (std::size_t i = min_words; i < a.word_count; ++i) {
        count += std::popcount(a.data[i]);
    }
    for (std::size_t i = min_words; i < b.word_count; ++i) {
        count += std::popcount(b.data[i]);
    }

    return count;
}

/**
 * @brief Count bits in intersection (A & B) - scalar version
 * @param a First bitset
 * @param b Second bitset
 * @return Number of bits set in (a AND b)
 *
 * Use cases:
 * - Set intersection cardinality
 * - Common features analysis
 */
[[nodiscard]] inline std::size_t intersectionCount(
    ConstBitSetView a, ConstBitSetView b) noexcept {

    std::size_t min_words = std::min(a.word_count, b.word_count);
    std::size_t count = 0;

    for (std::size_t i = 0; i < min_words; ++i) {
        count += std::popcount(a.data[i] & b.data[i]);
    }

    return count;
}

// ============================================================================
// Similarity metrics
// ============================================================================

/**
 * @brief Jaccard similarity coefficient: |A AND B| / |A OR B|
 * @param a First bitset
 * @param b Second bitset
 * @return Similarity in [0.0, 1.0] where 1.0 means identical sets
 *
 * Properties:
 * - jaccard(A, A) = 1.0 (identical)
 * - jaccard(A, B) = 0.0 (disjoint sets)
 * - Symmetric: jaccard(A, B) = jaccard(B, A)
 *
 * Use cases:
 * - Document similarity
 * - Collaborative filtering
 * - Feature matching
 */
[[nodiscard]] inline float jaccardSimilarity(
    ConstBitSetView a, ConstBitSetView b) noexcept {

    std::size_t intersection = intersectionCount(a, b);
    std::size_t union_size = unionCount(a, b);

    if (union_size == 0) return 1.0f; // Both empty
    return static_cast<float>(intersection) / static_cast<float>(union_size);
}

/**
 * @brief Dice coefficient (Sorensen-Dice): 2 * |A AND B| / (|A| + |B|)
 * @param a First bitset
 * @param b Second bitset
 * @return Coefficient in [0.0, 1.0]
 *
 * Properties:
 * - dice(A, A) = 1.0
 * - dice(A, B) = 0.0 if disjoint
 * - More weight to overlap than Jaccard
 *
 * Use cases:
 * - Image segmentation overlap
 * - Medical imaging similarity
 */
[[nodiscard]] inline float diceCoefficient(
    ConstBitSetView a, ConstBitSetView b) noexcept {

    std::size_t intersection = intersectionCount(a, b);
    std::size_t count_a = popcount(a);
    std::size_t count_b = popcount(b);

    std::size_t sum = count_a + count_b;
    if (sum == 0) return 1.0f; // Both empty

    return (2.0f * static_cast<float>(intersection)) / static_cast<float>(sum);
}

// ============================================================================
// Set relationship tests
// ============================================================================

/**
 * @brief Test if A is a subset of B (A subseteq B) - scalar version
 * @param a First bitset (potential subset)
 * @param b Second bitset (potential superset)
 * @return true if every bit set in A is also set in B
 *
 * Algorithm:
 * - For each word: (a[i] & ~b[i]) must be 0
 * - If a is longer, remaining words must be 0
 * - Early exit on first violation
 *
 * Performance:
 * - Best case: O(1) - first word violates subset property
 * - Worst case: O(n) - A is actually a subset of B
 */
[[nodiscard]] inline bool isSubsetOf(ConstBitSetView a, ConstBitSetView b) noexcept {
    std::size_t min_words = std::min(a.word_count, b.word_count);

    // Check common words: (a & ~b) must be 0
    for (std::size_t i = 0; i < min_words; ++i) {
        if ((a.data[i] & ~b.data[i]) != 0) return false;
    }

    // If a is longer, remaining words must be 0
    for (std::size_t i = min_words; i < a.word_count; ++i) {
        if (a.data[i] != 0) return false;
    }

    return true;
}

/**
 * @brief Test if A and B have any common set bits (A AND B != empty) - scalar version
 * @param a First bitset
 * @param b Second bitset
 * @return true if (a AND b) has at least one set bit
 *
 * Algorithm:
 * - For each word: (a[i] & b[i]) != 0
 * - Early exit upon finding first common bit
 *
 * Performance:
 * - Best case: O(1) - first word has common bits
 * - Worst case: O(n) - no common bits (disjoint sets)
 */
[[nodiscard]] inline bool intersects(ConstBitSetView a, ConstBitSetView b) noexcept {
    std::size_t min_words = std::min(a.word_count, b.word_count);

    for (std::size_t i = 0; i < min_words; ++i) {
        if ((a.data[i] & b.data[i]) != 0) return true;
    }

    return false;
}

