// Shared SIMD advanced bit operation helpers and optimized wrappers for header/module parity.

namespace detail {


#if defined(__AVX2__)

/**
 * @brief SIMD-accelerated reverse using AVX2 block swaps
 * @param view Bitset view (mutable)
 *
 * Algorithm:
 * - Reverse bits within each word with scalar SWAR (already optimal)
 * - Reverse word order in 4-word blocks with AVX2
 * - Finish any middle remainder scalar
 */
inline void reverseSimd(BitSetView view) noexcept {
    if (view.word_count == 0) return;

    advancedMaskUnusedBits(view);

    for (std::size_t i = 0; i < view.word_count; ++i) {
        view.data[i] = reverseBits64(view.data[i]);
    }

    std::size_t left = 0;
    std::size_t right = view.word_count;

    while (right - left >= 8) {
        __m256i left_block =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(view.data + left));
        __m256i right_block =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(view.data + right - 4));

        left_block = _mm256_permute4x64_epi64(left_block, 0b00011011);
        right_block = _mm256_permute4x64_epi64(right_block, 0b00011011);

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(view.data + left), right_block);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(view.data + right - 4), left_block);

        left += 4;
        right -= 4;
    }

    while (left < right) {
        --right;
        if (left >= right) break;
        std::swap(view.data[left], view.data[right]);
        ++left;
    }

    const std::size_t unused_bits = view.word_count * 64 - view.bit_count;
    if (unused_bits > 0) {
        for (std::size_t i = 0; i < view.word_count - 1; ++i) {
            view.data[i] = (view.data[i] >> unused_bits) |
                          (view.data[i + 1] << (64 - unused_bits));
        }
        view.data[view.word_count - 1] >>= unused_bits;
    }

    advancedMaskUnusedBits(view);
}

/**
 * @brief SIMD-accelerated byte swap using AVX2
 * @param view Bitset view (mutable)
 *
 * Algorithm:
 * - Use AVX2 shuffle to byte-swap 4 words (32 bytes) at once
 * - Fallback to scalar BSWAP for remainder
 */
inline void byteSwapSimd(BitSetView view) noexcept {
    std::size_t i = 0;
    const std::size_t avx2_words = view.word_count & ~std::size_t{3};

    const __m256i shuffle_mask = _mm256_setr_epi8(
        7, 6, 5, 4, 3, 2, 1, 0,
        15, 14, 13, 12, 11, 10, 9, 8,
        7, 6, 5, 4, 3, 2, 1, 0,
        15, 14, 13, 12, 11, 10, 9, 8
    );

    for (; i < avx2_words; i += 4) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(view.data + i));
        v = _mm256_shuffle_epi8(v, shuffle_mask);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(view.data + i), v);
    }

    for (; i < view.word_count; ++i) {
        view.data[i] = byteswap64(view.data[i]);
    }
}

/**
 * @brief Validated rotate implementation used by optimized wrappers
 * @param view Bitset view (mutable)
 * @param shift Number of positions to rotate left
 */
inline void rotateLeftSimd(BitSetView view, std::size_t shift) noexcept {
    advancedRotateLeftBuffered(view, shift);
}

inline void rotateRightSimd(BitSetView view, std::size_t shift) noexcept {
    if (shift == 0 || view.bit_count == 0) return;
    shift %= view.bit_count;
    if (shift == 0) return;

    advancedRotateLeftBuffered(view, view.bit_count - shift);
}

#endif // __AVX2__

} // namespace detail

// ============================================================================
// Public API with automatic SIMD dispatch
// ============================================================================

/**
 * @brief reverse with automatic SIMD dispatch
 * @param view Bitset view
 * @note Automatically uses AVX2 for larger bitsets (dispatch policy: >= 64 words)
 */
BITOPS_FORCEINLINE void reverseOptimized(BitSetView view) noexcept {
#if defined(__AVX2__)
    if (view.word_count >= kReverseSimdMinWords) {
        detail::reverseSimd(view);
        return;
    }
#endif
    reverse(view);
}

/**
 * @brief byteSwap with automatic SIMD dispatch
 * @param view Bitset view
 * @note Automatically uses AVX2 once the byte-swap span is wide enough (dispatch policy: >= 8 words)
 */
inline void byteSwapOptimized(BitSetView view) noexcept {
#if defined(__AVX2__)
    if (view.word_count >= kByteSwapSimdMinWords) {
        detail::byteSwapSimd(view);
        return;
    }
#endif
    byteSwap(view);
}

/**
 * @brief rotateLeft with automatic SIMD dispatch
 * @param view Bitset view
 * @param shift Number of positions to rotate
 */
inline void rotateLeftOptimized(BitSetView view, std::size_t shift) noexcept {
#if defined(__AVX2__)
    if (view.word_count >= kRotateSimdMinWords) {
        detail::rotateLeftSimd(view, shift);
        return;
    }
#endif
    rotateLeft(view, shift);
}

/**
 * @brief rotateRight with automatic SIMD dispatch
 * @param view Bitset view
 * @param shift Number of positions to rotate
 */
inline void rotateRightOptimized(BitSetView view, std::size_t shift) noexcept {
#if defined(__AVX2__)
    if (view.word_count >= kRotateSimdMinWords) {
        detail::rotateRightSimd(view, shift);
        return;
    }
#endif
    rotateRight(view, shift);
}
