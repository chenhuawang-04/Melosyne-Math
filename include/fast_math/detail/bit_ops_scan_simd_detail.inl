// Shared SIMD scan helpers and optimized dispatch wrappers for header/module parity.

namespace detail {


#if defined(__AVX2__)

/**
 * @brief SIMD-accelerated select (find k-th set bit)
 * @param view Bitset view
 * @param k Index of set bit to find (0-indexed)
 * @return Position of k-th set bit, or view.bit_count if not found
 *
 * Algorithm:
 * - Use AVX2 popcount to accumulate counts quickly
 * - Once target word is found, use scalar bit-by-bit search
 *
 * Performance:
 * - ~1.5x faster than scalar for large bitsets
 * - Speedup increases with bitset size (more parallelism)
 */
inline std::size_t selectSimd(ConstBitSetView view, std::size_t k) noexcept {
    std::size_t count = 0;
    std::size_t i = 0;
    const bool has_partial_tail = view.word_count != 0 && (view.bit_count % 64) != 0;
    const std::size_t simd_word_count = has_partial_tail ? view.word_count - 1 : view.word_count;
    const std::size_t avx2_words = simd_word_count & ~std::size_t{3};

    for (; i < avx2_words; i += 4) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(view.data + i));
        std::size_t batch_popcount = popcount256(v);

        if (count + batch_popcount > k) {
            for (std::size_t j = i; j < i + 4; ++j) {
                const uint64_t word = view.data[j];
                const std::size_t word_popcount = std::popcount(word);

                if (count + word_popcount > k) {
                    std::size_t target = k - count;

                    for (std::size_t bit = 0; bit < 64; ++bit) {
                        if (word & (uint64_t{1} << bit)) {
                            if (target == 0) {
                                return j * 64 + bit;
                            }
                            --target;
                        }
                    }
                }

                count += word_popcount;
            }
        }

        count += batch_popcount;
    }

    for (; i < view.word_count; ++i) {
        const uint64_t word = scanMaskedWord(view, i);
        const std::size_t word_popcount = std::popcount(word);

        if (count + word_popcount > k) {
            std::size_t target = k - count;

            for (std::size_t bit = 0; bit < 64; ++bit) {
                if (word & (uint64_t{1} << bit)) {
                    if (target == 0) {
                        return i * 64 + bit;
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
 * @brief SIMD-accelerated rank (count set bits up to position)
 * @param view Bitset view
 * @param pos Position (exclusive upper bound)
 * @return Number of set bits in [0, pos)
 *
 * Algorithm:
 * - Use AVX2 popcount for full words (parallel counting)
 * - Use scalar popcount for partial word
 *
 * Performance:
 * - ~2x faster than scalar for large positions
 * - Speedup proportional to number of full words
 */
inline std::size_t rankSimd(ConstBitSetView view, std::size_t pos) noexcept {
    if (pos == 0) return 0;
    if (pos >= view.bit_count) pos = view.bit_count;

    const std::size_t word_idx = pos / 64;
    const std::size_t bit_idx = pos % 64;
    std::size_t count = 0;

    std::size_t i = 0;
    const std::size_t avx2_words = word_idx & ~std::size_t{3};

    for (; i < avx2_words; i += 4) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(view.data + i));
        count += popcount256(v);
    }

    for (; i < word_idx; ++i) {
        count += std::popcount(view.data[i]);
    }

    if (bit_idx > 0) {
        const uint64_t mask = scanMaskExclusive(bit_idx);
        count += std::popcount(scanMaskedWord(view, word_idx) & mask);
    }

    return count;
}

#endif // __AVX2__

} // namespace detail

// ============================================================================
// Public API with automatic SIMD dispatch
// ============================================================================

/**
 * @brief Select with automatic SIMD dispatch
 * @param view Bitset view
 * @param k Index of set bit to find
 * @return Position of k-th set bit
 * @note Automatically uses AVX2 for larger bitsets (dispatch policy: >= 128 words)
 */
BITOPS_FORCEINLINE std::size_t selectOptimized(ConstBitSetView view, std::size_t k) noexcept {
#if defined(__AVX2__)
    if (view.word_count >= kSelectSimdMinWords) {
        return detail::selectSimd(view, k);
    }
#endif
    return select(view, k);
}

/**
 * @brief Rank with automatic SIMD dispatch
 * @param view Bitset view
 * @param pos Position (exclusive upper bound)
 * @return Number of set bits in [0, pos)
 * @note Automatically uses AVX2 when pos spans many words
 */
BITOPS_FORCEINLINE std::size_t rankOptimized(ConstBitSetView view, std::size_t pos) noexcept {
#if defined(__AVX2__)
    if (view.word_count >= kRankSimdMinWords && (pos / 64) >= kRankSimdMinPosWords) {
        return detail::rankSimd(view, pos);
    }
#endif
    return rank(view, pos);
}

template<std::size_t N>
BITOPS_FORCEINLINE std::size_t rankOptimized(const BitSet<N>& view, std::size_t pos) noexcept {
#if defined(__AVX2__)
    constexpr std::size_t kWords = (N + 63) / 64;
    if constexpr (kWords >= kRankSimdMinWords) {
        if ((pos >> 6) >= kRankSimdMinPosWords) {
            return detail::rankSimd(ConstBitSetView(view), pos);
        }
    }
#endif
    return rank(ConstBitSetView(view), pos);
}
