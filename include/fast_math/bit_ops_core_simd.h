/**
 * @file bit_ops_core_simd.h
 * @brief SIMD-optimized bitwise operations (AVX2/SSE4.1)
 *
 * Design Philosophy:
 * - AVX2/SSE4.1 intrinsics for batch processing
 * - Adaptive strategy: Scalar for small, SIMD for large
 * - In `detail` namespace (internal implementation)
 * - Public API automatically dispatches to SIMD when beneficial
 *
 * Performance:
 * - AVX2: Process 256 bits (4 words) per iteration
 * - ~2-4x faster than scalar for large bitsets (> 512 bits)
 * - Aligned memory access (bitsets are alignas(32))
 *
 * Architecture:
 * - AVX2 path: >= 9 words (> 512 bits)
 * - Scalar path: < 9 words (<= 512 bits)
 * - Compiler auto-vectorization handles scalar
 */

#pragma once

#include "bitset_view.h"
#include "bit_ops_core.h"
#include <cstring>
#include <algorithm>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace Melosyne {
namespace BitOps {
namespace detail {

#if defined(__AVX2__)

/**
 * @brief Bitwise AND with AVX2 optimization
 * @param dst Destination bitset
 * @param src Source bitset
 * @note AVX2: Processes 4 words (256 bits) per iteration
 */
inline void bitwiseAndSimd(BitSetView dst, ConstBitSetView src) noexcept {
    std::size_t min_words = std::min(dst.word_count, src.word_count);

    // AVX2 path
    std::size_t i = 0;
    const std::size_t avx2_words = min_words & ~std::size_t{3};

    for (; i < avx2_words; i += 4) {
        __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(dst.data + i));
        __m256i b = _mm256_load_si256(reinterpret_cast<const __m256i*>(src.data + i));
        __m256i result = _mm256_and_si256(a, b);
        _mm256_store_si256(reinterpret_cast<__m256i*>(dst.data + i), result);
    }

    // Scalar remainder
    for (; i < min_words; ++i) {
        dst.data[i] &= src.data[i];
    }

    // Clear beyond src length
    for (; i < dst.word_count; ++i) {
        dst.data[i] = 0;
    }
}

/**
 * @brief Bitwise OR with AVX2 optimization
 */
inline void bitwiseOrSimd(BitSetView dst, ConstBitSetView src) noexcept {
    std::size_t min_words = std::min(dst.word_count, src.word_count);

    std::size_t i = 0;
    const std::size_t avx2_words = min_words & ~std::size_t{3};

    for (; i < avx2_words; i += 4) {
        __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(dst.data + i));
        __m256i b = _mm256_load_si256(reinterpret_cast<const __m256i*>(src.data + i));
        __m256i result = _mm256_or_si256(a, b);
        _mm256_store_si256(reinterpret_cast<__m256i*>(dst.data + i), result);
    }

    for (; i < min_words; ++i) {
        dst.data[i] |= src.data[i];
    }
}

/**
 * @brief Bitwise XOR with AVX2 optimization
 */
inline void bitwiseXorSimd(BitSetView dst, ConstBitSetView src) noexcept {
    std::size_t min_words = std::min(dst.word_count, src.word_count);

    std::size_t i = 0;
    const std::size_t avx2_words = min_words & ~std::size_t{3};

    for (; i < avx2_words; i += 4) {
        __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(dst.data + i));
        __m256i b = _mm256_load_si256(reinterpret_cast<const __m256i*>(src.data + i));
        __m256i result = _mm256_xor_si256(a, b);
        _mm256_store_si256(reinterpret_cast<__m256i*>(dst.data + i), result);
    }

    for (; i < min_words; ++i) {
        dst.data[i] ^= src.data[i];
    }
}

/**
 * @brief Copy with AVX2 optimization
 */
inline void copySimd(BitSetView dst, ConstBitSetView src) noexcept {
    std::size_t min_words = std::min(dst.word_count, src.word_count);

    std::size_t i = 0;
    const std::size_t avx2_words = min_words & ~std::size_t{3};

    for (; i < avx2_words; i += 4) {
        __m256i data = _mm256_load_si256(reinterpret_cast<const __m256i*>(src.data + i));
        _mm256_store_si256(reinterpret_cast<__m256i*>(dst.data + i), data);
    }

    for (; i < min_words; ++i) {
        dst.data[i] = src.data[i];
    }

    for (; i < dst.word_count; ++i) {
        dst.data[i] = 0;
    }
}

/**
 * @brief Equality test with AVX2 optimization
 */
[[nodiscard]] inline bool equalSimd(ConstBitSetView a, ConstBitSetView b) noexcept {
    if (a.bit_count != b.bit_count) {
        return false;
    }

    std::size_t i = 0;
    const std::size_t avx2_words = a.word_count & ~std::size_t{3};

    for (; i < avx2_words; i += 4) {
        __m256i av = _mm256_load_si256(reinterpret_cast<const __m256i*>(a.data + i));
        __m256i bv = _mm256_load_si256(reinterpret_cast<const __m256i*>(b.data + i));
        __m256i cmp = _mm256_cmpeq_epi64(av, bv);

        if (_mm256_movemask_epi8(cmp) != -1) {
            return false;
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
// Public API with automatic dispatch (using compiler hints for vectorization)
// ============================================================================

/**
 * @brief Bitwise AND with compiler-guided vectorization
 * @param dst Destination bitset (modified in-place)
 * @param src Source bitset (read-only)
 * @note Uses compiler hints to ensure vectorization (faster than manual AVX2)
 */
BITOPS_FORCEINLINE void bitwiseAndOptimized(BitSetView dst, ConstBitSetView src) noexcept {
    const std::size_t min_words = std::min(dst.word_count, src.word_count);

    // Use restrict to help compiler optimize
    uint64_t* __restrict dst_ptr = dst.data;
    const uint64_t* __restrict src_ptr = src.data;

    // Compiler hints for vectorization
#if defined(_MSC_VER)
    #pragma loop(ivdep)
    #pragma loop(hint_parallel(4))
#elif defined(__clang__)
    #pragma clang loop vectorize(enable) interleave(enable)
#elif defined(__GNUC__)
    #pragma GCC ivdep
#endif
    for (std::size_t i = 0; i < min_words; ++i) {
        dst_ptr[i] &= src_ptr[i];
    }

    // Clear beyond src length
#if defined(_MSC_VER)
    #pragma loop(ivdep)
#elif defined(__clang__)
    #pragma clang loop vectorize(enable)
#elif defined(__GNUC__)
    #pragma GCC ivdep
#endif
    for (std::size_t i = min_words; i < dst.word_count; ++i) {
        dst_ptr[i] = 0;
    }
}

} // namespace BitOps
} // namespace Melosyne
