/**
 * @file bit_ops_core_simd.h
 * @brief SIMD-optimized core bitwise operations (AVX2)
 */

#pragma once

#include "config_macros.h"
#include "bitset_view.h"
#include "bit_ops_core.h"

#include <algorithm>
#include <cstdint>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace Melosyne {
namespace BitOps {
namespace detail {

#if defined(__AVX2__)

static constexpr std::size_t kAvx2Words = 4;  // 4 x 64-bit words = 256 bits
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
}

[[nodiscard]] inline bool equalSimd(ConstBitSetView a, ConstBitSetView b) noexcept {
    if (a.bit_count != b.bit_count) {
        return false;
    }

    const std::size_t simd_words = a.word_count & ~(kAvx2Words - 1);
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

    for (; i < a.word_count; ++i) {
        if (a.data[i] != b.data[i]) {
            return false;
        }
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
    if (std::min(dst.word_count, src.word_count) >= 256) {
        detail::bitwiseAndSimd(dst, src);
        return;
    }
#endif
    bitwiseAnd(dst, src);
}

BITOPS_FORCEINLINE void bitwiseOrOptimized(BitSetView dst, ConstBitSetView src) noexcept {
#if defined(__AVX2__)
    if (std::min(dst.word_count, src.word_count) >= 256) {
        detail::bitwiseOrSimd(dst, src);
        return;
    }
#endif
    bitwiseOr(dst, src);
}

BITOPS_FORCEINLINE void bitwiseXorOptimized(BitSetView dst, ConstBitSetView src) noexcept {
#if defined(__AVX2__)
    if (std::min(dst.word_count, src.word_count) >= 256) {
        detail::bitwiseXorSimd(dst, src);
        return;
    }
#endif
    bitwiseXor(dst, src);
}

BITOPS_FORCEINLINE void copyOptimized(BitSetView dst, ConstBitSetView src) noexcept {
#if defined(__AVX2__)
    if (std::min(dst.word_count, src.word_count) >= 256) {
        detail::copySimd(dst, src);
        return;
    }
#endif
    copy(dst, src);
}

[[nodiscard]] BITOPS_FORCEINLINE bool equalOptimized(ConstBitSetView a, ConstBitSetView b) noexcept {
#if defined(__AVX2__)
    if (a.word_count >= 256) {
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
    if constexpr (std::min(kDstWords, kSrcWords) >= 256) {
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
    if constexpr (std::min(kDstWords, kSrcWords) >= 256) {
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
    if constexpr (std::min(kDstWords, kSrcWords) >= 256) {
        detail::bitwiseXorSimd(BitSetView(dst), ConstBitSetView(src));
        return;
    }
#endif
    bitwiseXor(BitSetView(dst), ConstBitSetView(src));
}

template<std::size_t DstBits, std::size_t SrcBits>
BITOPS_FORCEINLINE void copyOptimized(BitSet<DstBits>& dst, const BitSet<SrcBits>& src) noexcept {
#if defined(__AVX2__)
    constexpr std::size_t kDstWords = (DstBits + 63) / 64;
    constexpr std::size_t kSrcWords = (SrcBits + 63) / 64;
    if constexpr (std::min(kDstWords, kSrcWords) >= 256) {
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
    if constexpr (ABits == BBits && kAWords >= 256) {
        return detail::equalSimd(ConstBitSetView(a), ConstBitSetView(b));
    }
#endif
    return equal(ConstBitSetView(a), ConstBitSetView(b));
}

} // namespace BitOps
} // namespace Melosyne
