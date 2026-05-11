// Shared SIMD core bit operations for header/module parity.

namespace detail {


#if defined(__AVX2__)

static constexpr std::size_t kAvx2Words = 4;
static constexpr std::size_t kAvx2Bytes = 32;

BITOPS_FORCEINLINE bool isAligned32(const void* ptr_) noexcept {
    return (reinterpret_cast<std::uintptr_t>(ptr_) & (kAvx2Bytes - 1)) == 0;
}

inline void bitwiseAndSimd(BitSetView dst, ConstBitSetView src) noexcept {
    const std::size_t min_words = std::min(dst.word_count, src.word_count);
    const std::size_t simd_words = min_words & ~(kAvx2Words - 1);

    std::size_t i = 0;
    const bool aligned = isAligned32(dst.data) && isAligned32(src.data);

    if (aligned) {
        for (; i < simd_words; i += kAvx2Words) {
            const __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(dst.data + i));
            const __m256i b = _mm256_load_si256(reinterpret_cast<const __m256i*>(src.data + i));
            _mm256_store_si256(reinterpret_cast<__m256i*>(dst.data + i), _mm256_and_si256(a, b));
        }
    } else {
        for (; i < simd_words; i += kAvx2Words) {
            const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst.data + i));
            const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src.data + i));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst.data + i), _mm256_and_si256(a, b));
        }
    }

    for (; i < min_words; ++i) {
        dst.data[i] &= src.data[i];
    }
    for (; i < dst.word_count; ++i) {
        dst.data[i] = 0;
    }

    coreMaskUnusedBits(dst);
}

inline void bitwiseOrSimd(BitSetView dst, ConstBitSetView src) noexcept {
    const std::size_t min_words = std::min(dst.word_count, src.word_count);
    const std::size_t simd_words = min_words & ~(kAvx2Words - 1);

    std::size_t i = 0;
    const bool aligned = isAligned32(dst.data) && isAligned32(src.data);

    if (aligned) {
        for (; i < simd_words; i += kAvx2Words) {
            const __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(dst.data + i));
            const __m256i b = _mm256_load_si256(reinterpret_cast<const __m256i*>(src.data + i));
            _mm256_store_si256(reinterpret_cast<__m256i*>(dst.data + i), _mm256_or_si256(a, b));
        }
    } else {
        for (; i < simd_words; i += kAvx2Words) {
            const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst.data + i));
            const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src.data + i));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst.data + i), _mm256_or_si256(a, b));
        }
    }

    for (; i < min_words; ++i) {
        dst.data[i] |= src.data[i];
    }

    coreMaskUnusedBits(dst);
}

inline void bitwiseXorSimd(BitSetView dst, ConstBitSetView src) noexcept {
    const std::size_t min_words = std::min(dst.word_count, src.word_count);
    const std::size_t simd_words = min_words & ~(kAvx2Words - 1);

    std::size_t i = 0;
    const bool aligned = isAligned32(dst.data) && isAligned32(src.data);

    if (aligned) {
        for (; i < simd_words; i += kAvx2Words) {
            const __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(dst.data + i));
            const __m256i b = _mm256_load_si256(reinterpret_cast<const __m256i*>(src.data + i));
            _mm256_store_si256(reinterpret_cast<__m256i*>(dst.data + i), _mm256_xor_si256(a, b));
        }
    } else {
        for (; i < simd_words; i += kAvx2Words) {
            const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst.data + i));
            const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src.data + i));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst.data + i), _mm256_xor_si256(a, b));
        }
    }

    for (; i < min_words; ++i) {
        dst.data[i] ^= src.data[i];
    }

    coreMaskUnusedBits(dst);
}

inline void bitwiseNotSimd(BitSetView dst) noexcept {
    const std::size_t simd_words = dst.word_count & ~(kAvx2Words - 1);
    const __m256i ones = _mm256_set1_epi64x(-1);
    const bool aligned = isAligned32(dst.data);

    std::size_t i = 0;
    if (aligned) {
        for (; i < simd_words; i += kAvx2Words) {
            const __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i*>(dst.data + i));
            _mm256_store_si256(reinterpret_cast<__m256i*>(dst.data + i), _mm256_xor_si256(v, ones));
        }
    } else {
        for (; i < simd_words; i += kAvx2Words) {
            const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst.data + i));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst.data + i), _mm256_xor_si256(v, ones));
        }
    }

    for (; i < dst.word_count; ++i) {
        dst.data[i] = ~dst.data[i];
    }

    coreMaskUnusedBits(dst);
}

inline void bitwiseAndNotSimd(BitSetView dst, ConstBitSetView src) noexcept {
    const std::size_t min_words = std::min(dst.word_count, src.word_count);
    const std::size_t simd_words = min_words & ~(kAvx2Words - 1);

    std::size_t i = 0;
    const bool aligned = isAligned32(dst.data) && isAligned32(src.data);

    if (aligned) {
        for (; i < simd_words; i += kAvx2Words) {
            const __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(dst.data + i));
            const __m256i b = _mm256_load_si256(reinterpret_cast<const __m256i*>(src.data + i));
            _mm256_store_si256(reinterpret_cast<__m256i*>(dst.data + i), _mm256_andnot_si256(b, a));
        }
    } else {
        for (; i < simd_words; i += kAvx2Words) {
            const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst.data + i));
            const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src.data + i));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst.data + i), _mm256_andnot_si256(b, a));
        }
    }

    for (; i < min_words; ++i) {
        dst.data[i] &= ~src.data[i];
    }

    coreMaskUnusedBits(dst);
}

inline void copySimd(BitSetView dst, ConstBitSetView src) noexcept {
    const std::size_t min_words = std::min(dst.word_count, src.word_count);
    const std::size_t simd_words = min_words & ~(kAvx2Words - 1);

    std::size_t i = 0;
    const bool aligned = isAligned32(dst.data) && isAligned32(src.data);

    if (aligned) {
        for (; i < simd_words; i += kAvx2Words) {
            const __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i*>(src.data + i));
            _mm256_store_si256(reinterpret_cast<__m256i*>(dst.data + i), v);
        }
    } else {
        for (; i < simd_words; i += kAvx2Words) {
            const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src.data + i));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst.data + i), v);
        }
    }

    for (; i < min_words; ++i) {
        dst.data[i] = src.data[i];
    }
    for (; i < dst.word_count; ++i) {
        dst.data[i] = 0;
    }

    coreMaskUnusedBits(dst);
}

[[nodiscard]] inline bool equalSimd(ConstBitSetView a, ConstBitSetView b) noexcept {
    if (a.bit_count != b.bit_count) {
        return false;
    }

    const std::size_t full_words = a.bit_count / 64;
    const std::size_t simd_words = full_words & ~(kAvx2Words - 1);

    std::size_t i = 0;
    const bool aligned = isAligned32(a.data) && isAligned32(b.data);

    if (aligned) {
        for (; i < simd_words; i += kAvx2Words) {
            const __m256i av = _mm256_load_si256(reinterpret_cast<const __m256i*>(a.data + i));
            const __m256i bv = _mm256_load_si256(reinterpret_cast<const __m256i*>(b.data + i));
            const __m256i cmp = _mm256_cmpeq_epi64(av, bv);
            if (_mm256_movemask_epi8(cmp) != -1) {
                return false;
            }
        }
    } else {
        for (; i < simd_words; i += kAvx2Words) {
            const __m256i av = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a.data + i));
            const __m256i bv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b.data + i));
            const __m256i cmp = _mm256_cmpeq_epi64(av, bv);
            if (_mm256_movemask_epi8(cmp) != -1) {
                return false;
            }
        }
    }

    for (; i < full_words; ++i) {
        if (a.data[i] != b.data[i]) {
            return false;
        }
    }

    const std::size_t remainder = a.bit_count % 64;
    if (remainder != 0) {
        const uint64_t mask = coreLowBitsMask(remainder);
        return (a.data[full_words] & mask) == (b.data[full_words] & mask);
    }

    return true;
}

#endif // __AVX2__

} // namespace detail

// ============================================================================
// Public optimized API
// ============================================================================

BITOPS_FORCEINLINE void bitwiseAndOptimized(BitSetView dst, ConstBitSetView src) noexcept {
#if defined(__AVX2__)
    if (std::min(dst.word_count, src.word_count) >= kBitwiseSimdMinWords) {
        detail::bitwiseAndSimd(dst, src);
        return;
    }
#endif
    bitwiseAnd(dst, src);
}

BITOPS_FORCEINLINE void bitwiseOrOptimized(BitSetView dst, ConstBitSetView src) noexcept {
#if defined(__AVX2__)
    if (std::min(dst.word_count, src.word_count) >= kBitwiseSimdMinWords) {
        detail::bitwiseOrSimd(dst, src);
        return;
    }
#endif
    bitwiseOr(dst, src);
}

BITOPS_FORCEINLINE void bitwiseXorOptimized(BitSetView dst, ConstBitSetView src) noexcept {
#if defined(__AVX2__)
    if (std::min(dst.word_count, src.word_count) >= kBitwiseSimdMinWords) {
        detail::bitwiseXorSimd(dst, src);
        return;
    }
#endif
    bitwiseXor(dst, src);
}

BITOPS_FORCEINLINE void bitwiseNotOptimized(BitSetView dst) noexcept {
#if defined(__AVX2__)
    if (dst.word_count >= kBitwiseSimdMinWords) {
        detail::bitwiseNotSimd(dst);
        return;
    }
#endif
    bitwiseNot(dst);
}

BITOPS_FORCEINLINE void bitwiseAndNotOptimized(BitSetView dst, ConstBitSetView src) noexcept {
#if defined(__AVX2__)
    if (std::min(dst.word_count, src.word_count) >= kBitwiseSimdMinWords) {
        detail::bitwiseAndNotSimd(dst, src);
        return;
    }
#endif
    bitwiseAndNot(dst, src);
}

BITOPS_FORCEINLINE void copyOptimized(BitSetView dst, ConstBitSetView src) noexcept {
#if defined(__AVX2__)
    if (std::min(dst.word_count, src.word_count) >= kBitwiseSimdMinWords) {
        detail::copySimd(dst, src);
        return;
    }
#endif
    copy(dst, src);
}

[[nodiscard]] BITOPS_FORCEINLINE bool equalOptimized(ConstBitSetView a, ConstBitSetView b) noexcept {
#if defined(__AVX2__)
    if (std::min(a.word_count, b.word_count) >= kBitwiseSimdMinWords) {
        return detail::equalSimd(a, b);
    }
#endif
    return equal(a, b);
}

template<std::size_t DstBits, std::size_t SrcBits>
BITOPS_FORCEINLINE void bitwiseAndOptimized(BitSet<DstBits>& dst, const BitSet<SrcBits>& src) noexcept {
#if defined(__AVX2__)
    constexpr std::size_t kDstWords = (DstBits + 63) / 64;
    constexpr std::size_t kSrcWords = (SrcBits + 63) / 64;
    if constexpr (std::min(kDstWords, kSrcWords) >= kBitwiseSimdMinWords) {
        detail::bitwiseAndSimd(BitSetView(dst), ConstBitSetView(src));
        return;
    }
#endif
    bitwiseAnd(BitSetView(dst), ConstBitSetView(src));
}

template<std::size_t DstBits, std::size_t SrcBits>
BITOPS_FORCEINLINE void bitwiseOrOptimized(BitSet<DstBits>& dst, const BitSet<SrcBits>& src) noexcept {
#if defined(__AVX2__)
    constexpr std::size_t kDstWords = (DstBits + 63) / 64;
    constexpr std::size_t kSrcWords = (SrcBits + 63) / 64;
    if constexpr (std::min(kDstWords, kSrcWords) >= kBitwiseSimdMinWords) {
        detail::bitwiseOrSimd(BitSetView(dst), ConstBitSetView(src));
        return;
    }
#endif
    bitwiseOr(BitSetView(dst), ConstBitSetView(src));
}

template<std::size_t DstBits, std::size_t SrcBits>
BITOPS_FORCEINLINE void bitwiseXorOptimized(BitSet<DstBits>& dst, const BitSet<SrcBits>& src) noexcept {
#if defined(__AVX2__)
    constexpr std::size_t kDstWords = (DstBits + 63) / 64;
    constexpr std::size_t kSrcWords = (SrcBits + 63) / 64;
    if constexpr (std::min(kDstWords, kSrcWords) >= kBitwiseSimdMinWords) {
        detail::bitwiseXorSimd(BitSetView(dst), ConstBitSetView(src));
        return;
    }
#endif
    bitwiseXor(BitSetView(dst), ConstBitSetView(src));
}

template<std::size_t Bits>
BITOPS_FORCEINLINE void bitwiseNotOptimized(BitSet<Bits>& dst) noexcept {
#if defined(__AVX2__)
    constexpr std::size_t kWords = (Bits + 63) / 64;
    if constexpr (kWords >= kBitwiseSimdMinWords) {
        detail::bitwiseNotSimd(BitSetView(dst));
        return;
    }
#endif
    bitwiseNot(BitSetView(dst));
}

template<std::size_t DstBits, std::size_t SrcBits>
BITOPS_FORCEINLINE void bitwiseAndNotOptimized(BitSet<DstBits>& dst, const BitSet<SrcBits>& src) noexcept {
#if defined(__AVX2__)
    constexpr std::size_t kDstWords = (DstBits + 63) / 64;
    constexpr std::size_t kSrcWords = (SrcBits + 63) / 64;
    if constexpr (std::min(kDstWords, kSrcWords) >= kBitwiseSimdMinWords) {
        detail::bitwiseAndNotSimd(BitSetView(dst), ConstBitSetView(src));
        return;
    }
#endif
    bitwiseAndNot(BitSetView(dst), ConstBitSetView(src));
}

template<std::size_t DstBits, std::size_t SrcBits>
BITOPS_FORCEINLINE void copyOptimized(BitSet<DstBits>& dst, const BitSet<SrcBits>& src) noexcept {
#if defined(__AVX2__)
    constexpr std::size_t kDstWords = (DstBits + 63) / 64;
    constexpr std::size_t kSrcWords = (SrcBits + 63) / 64;
    if constexpr (std::min(kDstWords, kSrcWords) >= kBitwiseSimdMinWords) {
        detail::copySimd(BitSetView(dst), ConstBitSetView(src));
        return;
    }
#endif
    copy(BitSetView(dst), ConstBitSetView(src));
}

template<std::size_t ABits, std::size_t BBits>
[[nodiscard]] BITOPS_FORCEINLINE bool equalOptimized(const BitSet<ABits>& a, const BitSet<BBits>& b) noexcept {
#if defined(__AVX2__)
    constexpr std::size_t kAWords = (ABits + 63) / 64;
    constexpr std::size_t kBWords = (BBits + 63) / 64;
    if constexpr (ABits == BBits && kAWords >= kBitwiseSimdMinWords && kBWords >= kBitwiseSimdMinWords) {
        return detail::equalSimd(ConstBitSetView(a), ConstBitSetView(b));
    }
#endif
    return equal(ConstBitSetView(a), ConstBitSetView(b));
}
