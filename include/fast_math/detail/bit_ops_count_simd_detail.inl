// Shared SIMD bit counting operations for header/module parity.

namespace detail {


#if defined(__AVX2__)

// ============================================================================
// Harley-Seal Algorithm: AVX2 Vectorized Popcount
// ============================================================================

/**
 * @brief Carry-Save Adder (CSA) - Core primitive for Harley-Seal
 * @param h High bits output (carry)
 * @param l Low bits output (sum)
 * @param a First input
 * @param b Second input
 * @param c Third input
 */
inline void csa256(__m256i& h, __m256i& l, __m256i a, __m256i b, __m256i c) noexcept {
    __m256i u = _mm256_xor_si256(a, b);
    h = _mm256_or_si256(_mm256_and_si256(a, b), _mm256_and_si256(u, c));
    l = _mm256_xor_si256(u, c);
}

/**
 * @brief AVX2 popcount for 256-bit register (Harley-Seal + vertical sum)
 * @param v 256-bit input vector
 * @return Total number of set bits in v
 */
inline uint64_t popcount256(__m256i v) noexcept {
    const __m256i lookup = _mm256_setr_epi8(
        0, 1, 1, 2,
        1, 2, 2, 3,
        1, 2, 2, 3,
        2, 3, 3, 4,
        0, 1, 1, 2,
        1, 2, 2, 3,
        1, 2, 2, 3,
        2, 3, 3, 4
    );

    const __m256i low_mask = _mm256_set1_epi8(0x0f);

    const __m256i lo = _mm256_and_si256(v, low_mask);
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);

    const __m256i popcnt_lo = _mm256_shuffle_epi8(lookup, lo);
    const __m256i popcnt_hi = _mm256_shuffle_epi8(lookup, hi);
    const __m256i total = _mm256_add_epi8(popcnt_lo, popcnt_hi);

    const __m256i sum64 = _mm256_sad_epu8(total, _mm256_setzero_si256());
    const __m128i sum128 = _mm_add_epi64(
        _mm256_castsi256_si128(sum64),
        _mm256_extracti128_si256(sum64, 1)
    );

    return static_cast<uint64_t>(_mm_extract_epi64(sum128, 0))
         + static_cast<uint64_t>(_mm_extract_epi64(sum128, 1));
}

BITOPS_FORCEINLINE bool countIsAligned32(const void* ptr_) noexcept {
    return (reinterpret_cast<std::uintptr_t>(ptr_) & 31u) == 0u;
}

/**
 * @brief SIMD popcount with adaptive strategy
 * @param view Bitset view
 * @return Number of set bits
 */
inline std::size_t popcountSimd(ConstBitSetView view) noexcept {
    std::size_t total = 0;
    std::size_t i = 0;
    const bool has_partial_tail = view.word_count != 0 && (view.bit_count % 64) != 0;
    const std::size_t simd_word_count = has_partial_tail ? view.word_count - 1 : view.word_count;
    const std::size_t avx2_words = simd_word_count & ~std::size_t{3};

    const bool aligned = countIsAligned32(view.data);
    if (aligned) {
        for (; i < avx2_words; i += 4) {
            const __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i*>(view.data + i));
            total += popcount256(v);
        }
    } else {
        for (; i < avx2_words; i += 4) {
            const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(view.data + i));
            total += popcount256(v);
        }
    }

    for (; i < view.word_count; ++i) {
        uint64_t word = view.data[i];
        if (i + 1 == view.word_count && has_partial_tail) {
            const std::size_t remainder = view.bit_count % 64;
            word &= (uint64_t{1} << remainder) - 1;
        }
        total += std::popcount(word);
    }

    return total;
}

/**
 * @brief Fused AND + popcount for in-place masking workloads (SIMD)
 */
inline std::size_t bitwiseAndCountSimd(
    BitSetView dst, ConstBitSetView src) noexcept {

    const std::size_t min_words = std::min(dst.word_count, src.word_count);
    const std::size_t remainder_bits = dst.bit_count % 64;
    const bool has_partial_dst_tail = remainder_bits != 0 && min_words == dst.word_count;
    const std::size_t simd_word_count = has_partial_dst_tail ? (min_words - 1) : min_words;
    const std::size_t avx2_words = simd_word_count & ~std::size_t{3};
    std::size_t count = 0;
    std::size_t i = 0;

    const bool aligned = countIsAligned32(dst.data) && countIsAligned32(src.data);
    if (aligned) {
        for (; i < avx2_words; i += 4) {
            const __m256i va = _mm256_load_si256(reinterpret_cast<const __m256i*>(dst.data + i));
            const __m256i vb = _mm256_load_si256(reinterpret_cast<const __m256i*>(src.data + i));
            const __m256i and_result = _mm256_and_si256(va, vb);
            _mm256_store_si256(reinterpret_cast<__m256i*>(dst.data + i), and_result);
            count += popcount256(and_result);
        }
    } else {
        for (; i < avx2_words; i += 4) {
            const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst.data + i));
            const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src.data + i));
            const __m256i and_result = _mm256_and_si256(va, vb);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst.data + i), and_result);
            count += popcount256(and_result);
        }
    }

    for (; i < simd_word_count; ++i) {
        const uint64_t value = dst.data[i] & src.data[i];
        dst.data[i] = value;
        count += std::popcount(value);
    }

    if (has_partial_dst_tail) {
        const std::size_t tail_index = dst.word_count - 1;
        const uint64_t value = (dst.data[tail_index] & src.data[tail_index]) &
            detail::coreLowBitsMask(remainder_bits);
        dst.data[tail_index] = value;
        count += std::popcount(value);
    }

    for (std::size_t j = min_words; j < dst.word_count; ++j) {
        dst.data[j] = 0;
    }

    return count;
}

/**
 * @brief Fused XOR + popcount for Hamming distance (SIMD)
 */
inline std::size_t hammingDistanceSimd(
    ConstBitSetView a, ConstBitSetView b) noexcept {

    const std::size_t min_words = std::min(a.word_count, b.word_count);
    std::size_t distance = 0;

    std::size_t i = 0;
    const std::size_t avx2_words = min_words & ~std::size_t{3};

    for (; i < avx2_words; i += 4) {
        const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a.data + i));
        const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b.data + i));
        const __m256i xor_result = _mm256_xor_si256(va, vb);
        distance += popcount256(xor_result);
    }

    for (; i < min_words; ++i) {
        uint64_t aw = a.data[i];
        uint64_t bw = b.data[i];
        const bool a_last = i + 1 == a.word_count && (a.bit_count % 64) != 0;
        const bool b_last = i + 1 == b.word_count && (b.bit_count % 64) != 0;
        if (a_last) {
            aw &= detail::coreLowBitsMask(a.bit_count % 64);
        }
        if (b_last) {
            bw &= detail::coreLowBitsMask(b.bit_count % 64);
        }
        distance += std::popcount(aw ^ bw);
    }

    for (std::size_t j = min_words; j < a.word_count; ++j) {
        uint64_t word = a.data[j];
        if (j + 1 == a.word_count && (a.bit_count % 64) != 0) {
            word &= detail::coreLowBitsMask(a.bit_count % 64);
        }
        distance += std::popcount(word);
    }
    for (std::size_t j = min_words; j < b.word_count; ++j) {
        uint64_t word = b.data[j];
        if (j + 1 == b.word_count && (b.bit_count % 64) != 0) {
            word &= detail::coreLowBitsMask(b.bit_count % 64);
        }
        distance += std::popcount(word);
    }

    return distance;
}

/**
 * @brief SIMD all() check with AVX2
 * @param view Bitset view
 * @return true if all bits are set
 */
inline bool allSimd(ConstBitSetView view) noexcept {
    if (view.word_count == 0) return true;

    const std::size_t complete_words = view.bit_count / 64;

    std::size_t i = 0;
    const std::size_t avx2_words = complete_words & ~std::size_t{3};
    const __m256i ones = _mm256_set1_epi64x(-1);

    const bool aligned = countIsAligned32(view.data);
    if (aligned) {
        for (; i < avx2_words; i += 4) {
            const __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i*>(view.data + i));
            const __m256i cmp = _mm256_cmpeq_epi64(v, ones);
            if (_mm256_movemask_epi8(cmp) != -1) return false;
        }
    } else {
        for (; i < avx2_words; i += 4) {
            const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(view.data + i));
            const __m256i cmp = _mm256_cmpeq_epi64(v, ones);
            if (_mm256_movemask_epi8(cmp) != -1) return false;
        }
    }

    for (; i < complete_words; ++i) {
        if (view.data[i] != ~uint64_t{0}) return false;
    }

    const std::size_t remaining_bits = view.bit_count % 64;
    if (remaining_bits > 0) {
        const uint64_t mask = detail::coreLowBitsMask(remaining_bits);
        return (view.data[complete_words] & mask) == mask;
    }

    return true;
}

/**
 * @brief Fused OR + popcount for union count (SIMD)
 */
inline std::size_t unionCountSimd(
    ConstBitSetView a, ConstBitSetView b) noexcept {

    const std::size_t min_words = std::min(a.word_count, b.word_count);
    std::size_t count = 0;

    std::size_t i = 0;
    const std::size_t avx2_words = min_words & ~std::size_t{3};

    for (; i < avx2_words; i += 4) {
        const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a.data + i));
        const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b.data + i));
        count += popcount256(_mm256_or_si256(va, vb));
    }

    for (; i < min_words; ++i) {
        uint64_t aw = a.data[i];
        uint64_t bw = b.data[i];
        const bool a_last = i + 1 == a.word_count && (a.bit_count % 64) != 0;
        const bool b_last = i + 1 == b.word_count && (b.bit_count % 64) != 0;
        if (a_last) {
            aw &= detail::coreLowBitsMask(a.bit_count % 64);
        }
        if (b_last) {
            bw &= detail::coreLowBitsMask(b.bit_count % 64);
        }
        count += std::popcount(aw | bw);
    }

    for (std::size_t j = min_words; j < a.word_count; ++j) {
        uint64_t word = a.data[j];
        if (j + 1 == a.word_count && (a.bit_count % 64) != 0) {
            word &= detail::coreLowBitsMask(a.bit_count % 64);
        }
        count += std::popcount(word);
    }
    for (std::size_t j = min_words; j < b.word_count; ++j) {
        uint64_t word = b.data[j];
        if (j + 1 == b.word_count && (b.bit_count % 64) != 0) {
            word &= detail::coreLowBitsMask(b.bit_count % 64);
        }
        count += std::popcount(word);
    }

    return count;
}

/**
 * @brief Fused AND + popcount for intersection count (SIMD)
 */
inline std::size_t intersectionCountSimd(
    ConstBitSetView a, ConstBitSetView b) noexcept {

    const std::size_t min_words = std::min(a.word_count, b.word_count);
    std::size_t count = 0;

    std::size_t i = 0;
    const std::size_t avx2_words = min_words & ~std::size_t{3};

    for (; i < avx2_words; i += 4) {
        const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a.data + i));
        const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b.data + i));
        count += popcount256(_mm256_and_si256(va, vb));
    }

    for (; i < min_words; ++i) {
        uint64_t aw = a.data[i];
        uint64_t bw = b.data[i];
        const bool a_last = i + 1 == a.word_count && (a.bit_count % 64) != 0;
        const bool b_last = i + 1 == b.word_count && (b.bit_count % 64) != 0;
        if (a_last) {
            aw &= detail::coreLowBitsMask(a.bit_count % 64);
        }
        if (b_last) {
            bw &= detail::coreLowBitsMask(b.bit_count % 64);
        }
        count += std::popcount(aw & bw);
    }

    return count;
}

#endif // __AVX2__

} // namespace detail

// ============================================================================
// Public API with automatic SIMD dispatch
// ============================================================================

/**
 * @brief In-place bitwise AND + popcount with automatic SIMD dispatch
 */
BITOPS_FORCEINLINE std::size_t bitwiseAndCountOptimized(
    BitSetView dst, ConstBitSetView src) noexcept {
#if defined(__AVX2__)
    if (std::min(dst.word_count, src.word_count) >= kBitwiseAndCountSimdMinWords) {
        return detail::bitwiseAndCountSimd(dst, src);
    }
#endif
    return bitwiseAndCount(dst, src);
}

template<std::size_t DstBits, std::size_t SrcBits>
BITOPS_FORCEINLINE std::size_t bitwiseAndCountOptimized(
    BitSet<DstBits>& dst, const BitSet<SrcBits>& src) noexcept {
#if defined(__AVX2__)
    constexpr std::size_t kDstWords = (DstBits + 63) / 64;
    constexpr std::size_t kSrcWords = (SrcBits + 63) / 64;
    if constexpr (std::min(kDstWords, kSrcWords) >= kBitwiseAndCountSimdMinWords) {
        return detail::bitwiseAndCountSimd(BitSetView(dst), ConstBitSetView(src));
    }
#endif
    return bitwiseAndCount(BitSetView(dst), ConstBitSetView(src));
}

/**
 * @brief popcount with optimal strategy selection
 * @param view Bitset view
 * @return Number of set bits
 * @note std::popcount is a single POPCNT instruction - already optimal
 */
BITOPS_FORCEINLINE std::size_t popcountOptimized(ConstBitSetView view) noexcept {
#if defined(__AVX2__)
    if (view.word_count >= kPopcountSimdMinWords) {
        return detail::popcountSimd(view);
    }
#endif
    return popcount(view);
}

template<std::size_t N>
BITOPS_FORCEINLINE std::size_t popcountOptimized(const BitSet<N>& view) noexcept {
#if defined(__AVX2__)
    constexpr std::size_t kWords = (N + 63) / 64;
    if constexpr (kWords >= kPopcountSimdMinWords) {
        return detail::popcountSimd(ConstBitSetView(view));
    }
#endif
    return popcount(ConstBitSetView(view));
}

/**
 * @brief Hamming distance with automatic SIMD dispatch
 */
inline std::size_t hammingDistanceOptimized(
    ConstBitSetView a, ConstBitSetView b) noexcept {
#if defined(__AVX2__)
    if (std::min(a.word_count, b.word_count) >= kFusedCountSimdMinWords) {
        return detail::hammingDistanceSimd(a, b);
    }
#endif
    return hammingDistance(a, b);
}

/**
 * @brief all() with automatic SIMD dispatch
 */
inline bool allOptimized(ConstBitSetView view) noexcept {
#if defined(__AVX2__)
    if (view.word_count >= kFusedCountSimdMinWords) {
        return detail::allSimd(view);
    }
#endif
    return all(view);
}

/**
 * @brief unionCount with automatic SIMD dispatch
 */
inline std::size_t unionCountOptimized(
    ConstBitSetView a, ConstBitSetView b) noexcept {
#if defined(__AVX2__)
    if (std::min(a.word_count, b.word_count) >= kFusedCountSimdMinWords) {
        return detail::unionCountSimd(a, b);
    }
#endif
    return unionCount(a, b);
}

/**
 * @brief intersectionCount with automatic SIMD dispatch
 */
inline std::size_t intersectionCountOptimized(
    ConstBitSetView a, ConstBitSetView b) noexcept {
#if defined(__AVX2__)
    if (std::min(a.word_count, b.word_count) >= kFusedCountSimdMinWords) {
        return detail::intersectionCountSimd(a, b);
    }
#endif
    return intersectionCount(a, b);
}

inline float jaccardSimilarityOptimized(
    ConstBitSetView a, ConstBitSetView b) noexcept {
    const std::size_t intersection = intersectionCountOptimized(a, b);
    const std::size_t union_size = unionCountOptimized(a, b);
    if (union_size == 0) return 1.0f;
    return static_cast<float>(intersection) / static_cast<float>(union_size);
}

inline float diceCoefficientOptimized(
    ConstBitSetView a, ConstBitSetView b) noexcept {
    const std::size_t intersection = intersectionCountOptimized(a, b);
    const std::size_t count_a = popcountOptimized(a);
    const std::size_t count_b = popcountOptimized(b);
    const std::size_t sum = count_a + count_b;
    if (sum == 0) return 1.0f;
    return (2.0f * static_cast<float>(intersection)) / static_cast<float>(sum);
}
