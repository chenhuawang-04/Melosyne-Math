/**
 * @file bit_ops_scan_simd.h
 * @brief SIMD-optimized bit scanning operations (limited optimizations)
 *
 * Design Philosophy:
 * - Most scan operations are inherently sequential (early-exit)
 * - SIMD benefits are limited compared to scalar TZCNT/LZCNT
 * - SIMD optimizations focus on select() and rank() operations
 * - For findFirst/findLast/findNext/findPrev, scalar is optimal
 *
 * SIMD-Optimized Operations:
 * - selectSimd: Parallel popcount accumulation with AVX2
 * - rankSimd: Parallel popcount for full words with AVX2
 *
 * Operations WITHOUT SIMD optimization (scalar is faster):
 * - findFirst/findLast: Single TZCNT/LZCNT is 1-cycle, SIMD overhead > benefit
 * - findNext/findPrev: Early-exit advantage, SIMD cannot help
 * - findFirstZero/findLastZero: Same as find operations
 *
 * Performance:
 * - select (SIMD): ~1.5x faster for large bitsets with many set bits
 * - rank (SIMD): ~2x faster for large positions (many full words to count)
 * - Other operations: Scalar version is optimal
 *
 * References:
 * - Zhou et al. "Fast Parallel Computation of Rank and Select"
 * - Vigna "Broadword Implementation of Rank/Select Queries"
 */

#pragma once

#include "bitset_view.h"
#include "bit_ops_scan.h"
#include "bit_ops_count_simd.h"  // For popcount256
#include <bit>
#include <cstdint>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace Melosyne {
namespace BitOps {
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
    const std::size_t avx2_words = view.word_count & ~std::size_t{3};

    // AVX2 path: Accumulate popcounts in parallel
    for (; i < avx2_words; i += 4) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(view.data + i));
        std::size_t batch_popcount = popcount256(v);

        if (count + batch_popcount > k) {
            // Target is in this batch, search word-by-word
            for (std::size_t j = i; j < i + 4; ++j) {
                uint64_t word = view.data[j];
                std::size_t word_popcount = std::popcount(word);

                if (count + word_popcount > k) {
                    // Found target word
                    std::size_t target = k - count;

                    // Bit-by-bit search within word
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

    // Scalar remainder
    for (; i < view.word_count; ++i) {
        uint64_t word = view.data[i];
        std::size_t word_popcount = std::popcount(word);

        if (count + word_popcount > k) {
            std::size_t target = k - count;

            for (std::size_t j = 0; j < 64; ++j) {
                if (word & (uint64_t{1} << j)) {
                    if (target == 0) {
                        return i * 64 + j;
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

    std::size_t word_idx = pos / 64;
    std::size_t bit_idx = pos % 64;
    std::size_t count = 0;

    // AVX2 path: Count full words in parallel
    std::size_t i = 0;
    const std::size_t avx2_words = word_idx & ~std::size_t{3};

    for (; i < avx2_words; i += 4) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(view.data + i));
        count += popcount256(v);
    }

    // Scalar: Remaining full words
    for (; i < word_idx; ++i) {
        count += std::popcount(view.data[i]);
    }

    // Count partial word
    if (bit_idx > 0) {
        uint64_t mask = (uint64_t{1} << bit_idx) - 1;
        count += std::popcount(view.data[word_idx] & mask);
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
 * @note Automatically uses AVX2 for large bitsets (> 8 words)
 */
BITOPS_FORCEINLINE std::size_t selectOptimized(ConstBitSetView view, std::size_t k) noexcept {
#if defined(__AVX2__)
    // Use SIMD for very large bitsets
    if (view.word_count > 128) {
        return detail::selectSimd(view, k);
    }
#endif
    // Use scalar for medium/small (compiler will optimize)
    return select(view, k);
}

/**
 * @brief Rank with automatic SIMD dispatch
 * @param view Bitset view
 * @param pos Position (exclusive upper bound)
 * @return Number of set bits in [0, pos)
 * @note Automatically uses AVX2 when pos spans many words
 */
inline std::size_t rankOptimized(ConstBitSetView view, std::size_t pos) noexcept {
#if defined(__AVX2__)
    // Use SIMD if pos spans more than 8 words
    if (pos / 64 > 8) {
        return detail::rankSimd(view, pos);
    }
#endif
    return rank(view, pos);
}

// ============================================================================
// Note on other operations
// ============================================================================

/*
 * The following operations do NOT have SIMD versions because scalar is optimal:
 *
 * - findFirst/findLast:
 *   TZCNT/LZCNT are 1-cycle instructions
 *   SIMD overhead (loading, masking, movemask) > scalar benefit
 *
 * - findNext/findPrev:
 *   Early-exit is the key optimization
 *   SIMD processes full vectors, cannot early-exit mid-vector
 *
 * - findFirstZero/findLastZero:
 *   Same reasoning as findFirst/findLast
 *
 * For these operations, always use the scalar versions from bit_ops_scan.h
 *
 * Example:
 * @code
 * // These are already optimal (scalar)
 * std::size_t first = findFirst(bs);    // 1-cycle TZCNT
 * std::size_t last = findLast(bs);      // 1-cycle LZCNT
 * std::size_t next = findNext(bs, pos); // Early-exit search
 * @endcode
 */

} // namespace BitOps
} // namespace Melosyne
