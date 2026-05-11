// Shared SIMD range operations for header/module parity.

namespace detail {


#if defined(__AVX2__)

/**
 * @brief SIMD-accelerated setRange (set bits to 1) using AVX2
 */
inline void setRangeSimd(BitSetView view, std::size_t start, std::size_t end) noexcept {
    if (start >= end || start >= view.bit_count) return;
    if (end > view.bit_count) end = view.bit_count;

    std::size_t start_word = start / 64;
    std::size_t end_word = (end - 1) / 64;
    const std::size_t start_bit = start % 64;
    const std::size_t end_bit = end % 64;

    if (start_word == end_word) {
        const uint64_t mask = createRangeMask(start_bit, end_bit == 0 ? 64 : end_bit);
        view.data[start_word] |= mask;
        return;
    }

    if (start_bit != 0) {
        const uint64_t mask = ~uint64_t{0} << start_bit;
        view.data[start_word] |= mask;
        ++start_word;
    }

    const std::size_t full_words = end_word - start_word;
    if (full_words >= 4) {
        const __m256i ones = _mm256_set1_epi64x(-1);
        std::size_t i = start_word;
        const std::size_t avx2_end = start_word + (full_words & ~std::size_t{3});

        for (; i < avx2_end; i += 4) {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(view.data + i), ones);
        }
        start_word = i;
    }

    for (std::size_t i = start_word; i < end_word; ++i) {
        view.data[i] = ~uint64_t{0};
    }

    if (end_bit != 0) {
        const uint64_t mask = (uint64_t{1} << end_bit) - 1;
        view.data[end_word] |= mask;
    } else if (end_word < view.word_count) {
        view.data[end_word] = ~uint64_t{0};
    }
}

/**
 * @brief SIMD-accelerated resetRange (clear bits to 0) using AVX2
 */
inline void resetRangeSimd(BitSetView view, std::size_t start, std::size_t end) noexcept {
    if (start >= end || start >= view.bit_count) return;
    if (end > view.bit_count) end = view.bit_count;

    std::size_t start_word = start / 64;
    std::size_t end_word = (end - 1) / 64;
    const std::size_t start_bit = start % 64;
    const std::size_t end_bit = end % 64;

    if (start_word == end_word) {
        const uint64_t mask = createRangeMask(start_bit, end_bit == 0 ? 64 : end_bit);
        view.data[start_word] &= ~mask;
        return;
    }

    if (start_bit != 0) {
        const uint64_t mask = ~uint64_t{0} << start_bit;
        view.data[start_word] &= ~mask;
        ++start_word;
    }

    const std::size_t full_words = end_word - start_word;
    if (full_words >= 4) {
        const __m256i zeros = _mm256_setzero_si256();
        std::size_t i = start_word;
        const std::size_t avx2_end = start_word + (full_words & ~std::size_t{3});

        for (; i < avx2_end; i += 4) {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(view.data + i), zeros);
        }
        start_word = i;
    }

    for (std::size_t i = start_word; i < end_word; ++i) {
        view.data[i] = 0;
    }

    if (end_bit != 0) {
        const uint64_t mask = (uint64_t{1} << end_bit) - 1;
        view.data[end_word] &= ~mask;
    } else if (end_word < view.word_count) {
        view.data[end_word] = 0;
    }
}

/**
 * @brief SIMD-accelerated flipRange (toggle bits) using AVX2
 */
inline void flipRangeSimd(BitSetView view, std::size_t start, std::size_t end) noexcept {
    if (start >= end || start >= view.bit_count) return;
    if (end > view.bit_count) end = view.bit_count;

    std::size_t start_word = start / 64;
    std::size_t end_word = (end - 1) / 64;
    const std::size_t start_bit = start % 64;
    const std::size_t end_bit = end % 64;

    if (start_word == end_word) {
        const uint64_t mask = createRangeMask(start_bit, end_bit == 0 ? 64 : end_bit);
        view.data[start_word] ^= mask;
        return;
    }

    if (start_bit != 0) {
        const uint64_t mask = ~uint64_t{0} << start_bit;
        view.data[start_word] ^= mask;
        ++start_word;
    }

    const std::size_t full_words = end_word - start_word;
    if (full_words >= 4) {
        const __m256i ones = _mm256_set1_epi64x(-1);
        std::size_t i = start_word;
        const std::size_t avx2_end = start_word + (full_words & ~std::size_t{3});

        for (; i < avx2_end; i += 4) {
            const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(view.data + i));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(view.data + i), _mm256_xor_si256(v, ones));
        }
        start_word = i;
    }

    for (std::size_t i = start_word; i < end_word; ++i) {
        view.data[i] = ~view.data[i];
    }

    if (end_bit != 0) {
        const uint64_t mask = (uint64_t{1} << end_bit) - 1;
        view.data[end_word] ^= mask;
    } else if (end_word < view.word_count) {
        view.data[end_word] = ~view.data[end_word];
    }
}

/**
 * @brief SIMD-accelerated popcountRange using AVX2 popcount
 */
[[nodiscard]] inline std::size_t popcountRangeSimd(
    ConstBitSetView view, std::size_t start, std::size_t end) noexcept {

    if (start >= end || start >= view.bit_count) return 0;
    if (end > view.bit_count) end = view.bit_count;

    std::size_t start_word = start / 64;
    const std::size_t end_word = (end - 1) / 64;
    const std::size_t start_bit = start % 64;
    const std::size_t end_bit = end % 64;
    std::size_t count = 0;

    if (start_word == end_word) {
        const uint64_t mask = createRangeMask(start_bit, end_bit == 0 ? 64 : end_bit);
        return std::popcount(view.data[start_word] & mask);
    }

    if (start_bit != 0) {
        const uint64_t mask = ~uint64_t{0} << start_bit;
        count += std::popcount(view.data[start_word] & mask);
        ++start_word;
    }

    const std::size_t full_words = end_word - start_word;
    if (full_words >= 4) {
        std::size_t i = start_word;
        const std::size_t avx2_end = start_word + (full_words & ~std::size_t{3});

        for (; i < avx2_end; i += 4) {
            const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(view.data + i));
            count += popcount256(v);
        }
        start_word = i;
    }

    for (std::size_t i = start_word; i < end_word; ++i) {
        count += std::popcount(view.data[i]);
    }

    if (end_bit != 0) {
        const uint64_t mask = (uint64_t{1} << end_bit) - 1;
        count += std::popcount(view.data[end_word] & mask);
    } else {
        count += std::popcount(view.data[end_word]);
    }

    return count;
}

#endif // __AVX2__

} // namespace detail

// ============================================================================
// Public API with automatic SIMD dispatch
// ============================================================================

/**
 * @brief setRange with automatic SIMD dispatch
 * @param view Bitset view
 * @param start Starting bit index
 * @param end Ending bit index
 * @note Automatically uses AVX2 for large ranges (>= 4 full words)
 */
BITOPS_FORCEINLINE void setRangeOptimized(
    BitSetView view, std::size_t start, std::size_t end) noexcept {
#if defined(__AVX2__)
    if (start < end && start < view.bit_count) {
        const std::size_t start_word = (start + 63) / 64;
        const std::size_t end_word = end / 64;
        if (end_word > start_word && (end_word - start_word) >= kRangeSimdMinFullWords) {
            detail::setRangeSimd(view, start, end);
            return;
        }
    }
#endif
    setRange(view, start, end);
}

/**
 * @brief resetRange with automatic SIMD dispatch
 * @param view Bitset view
 * @param start Starting bit index
 * @param end Ending bit index
 * @note Automatically uses AVX2 for large ranges
 */
BITOPS_FORCEINLINE void resetRangeOptimized(
    BitSetView view, std::size_t start, std::size_t end) noexcept {
#if defined(__AVX2__)
    if (start < end && start < view.bit_count) {
        const std::size_t start_word = (start + 63) / 64;
        const std::size_t end_word = end / 64;
        if (end_word > start_word && (end_word - start_word) >= kRangeSimdMinFullWords) {
            detail::resetRangeSimd(view, start, end);
            return;
        }
    }
#endif
    resetRange(view, start, end);
}

/**
 * @brief flipRange with automatic SIMD dispatch
 * @param view Bitset view
 * @param start Starting bit index
 * @param end Ending bit index
 * @note Automatically uses AVX2 for large ranges
 */
BITOPS_FORCEINLINE void flipRangeOptimized(
    BitSetView view, std::size_t start, std::size_t end) noexcept {
#if defined(__AVX2__)
    if (end > start && (end - start) <= 256) {
        flipRange(view, start, end);
        return;
    }

    if (start < end && start < view.bit_count) {
        const std::size_t start_word = (start + 63) / 64;
        const std::size_t end_word = end / 64;
        if (end_word > start_word && (end_word - start_word) >= kRangeSimdMinFullWords) {
            detail::flipRangeSimd(view, start, end);
            return;
        }
    }
#endif
    flipRange(view, start, end);
}

template<std::size_t N>
BITOPS_FORCEINLINE void flipRangeOptimized(
    BitSet<N>& view, std::size_t start, std::size_t end) noexcept {
#if defined(__AVX2__)
    constexpr std::size_t kWords = (N + 63) / 64;
    if constexpr (kWords >= kRangeSimdMinFullWords) {
        if (end > start && (end - start) <= 256) {
            flipRange(BitSetView(view), start, end);
            return;
        }
        if (start < end && start < N) {
            const std::size_t start_word = (start + 63) / 64;
            const std::size_t end_word = end / 64;
            if (end_word > start_word && (end_word - start_word) >= kRangeSimdMinFullWords) {
                detail::flipRangeSimd(BitSetView(view), start, end);
                return;
            }
        }
    }
#endif
    flipRange(BitSetView(view), start, end);
}

/**
 * @brief popcountRange with automatic SIMD dispatch
 * @param view Bitset view
 * @param start Starting bit index
 * @param end Ending bit index
 * @return Number of set bits in range
 * @note Automatically uses AVX2 popcount for large ranges
 */
[[nodiscard]] BITOPS_FORCEINLINE std::size_t popcountRangeOptimized(
    ConstBitSetView view, std::size_t start, std::size_t end) noexcept {
#if defined(__AVX2__)
    if (start < end && start < view.bit_count) {
        const std::size_t start_word = (start + 63) / 64;
        const std::size_t end_word = end / 64;
        if (end_word > start_word && (end_word - start_word) >= kRangeSimdMinFullWords) {
            return detail::popcountRangeSimd(view, start, end);
        }
    }
#endif
    return popcountRange(view, start, end);
}
