module;

#include "fast_math/config_macros.h"
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <limits>
#include <algorithm>
#include <bit>
#include <iterator>
#include <type_traits>
#include <memory>
#include <memory_resource>
#include <cfloat>
#include <immintrin.h>

namespace Melosyne {
namespace BitOps {
#include "fast_math/detail/bit_ops_dispatch_policy.inl"
} // namespace BitOps
} // namespace Melosyne

export module fast_math.bit_ops_advanced_simd;

export import fast_math.bit_ops_advanced;

export {
/**
 * @file bit_ops_advanced_simd.h
 * @brief SIMD-optimized advanced bit manipulation (limited optimizations)
 *
 * Design Philosophy:
 * - Most advanced operations are inherently scalar (SWAR, BMI2)
 * - SIMD optimizations focus on bulk bitset operations
 * - AVX2 for parallel word processing where beneficial
 * - In `detail` namespace (internal SIMD implementations)
 *
 * SIMD-Optimized Operations:
 * - reverseSimd: Parallel reverseBits64 + bulk word swap
 * - byteSwapSimd: Parallel BSWAP using AVX2 shuffle
 * - rotateLeftSimd: Bulk data movement with AVX2
 *
 * Operations WITHOUT SIMD optimization (scalar is optimal):
 * - reverseBits64/32/16/8: SWAR on single word, already parallel
 * - pext/pdep: BMI2 single instruction (3 cycles)
 * - parity: Requires XOR reduction, not parallelizable
 * - Gray code: Single-word SWAR operations
 * - Morton code: Single-word bit interleaving
 * - scatter/gather: Random access pattern, SIMD cannot help
 *
 * Performance:
 * - reverse (SIMD): used only for genuinely large bitsets
 * - byteSwap (SIMD): enabled earlier because the shuffle path is cheap
 * - rotate (SIMD): reserved for larger memory-bound workloads
 *
 * References:
 * - Intel Intrinsics Guide: https://www.intel.com/content/www/us/en/docs/intrinsics-guide
 * - SWAR Magic: https://aggregate.org/MAGIC/
 */



#if defined(__AVX2__)
#endif

namespace Melosyne {
namespace BitOps {
#include "fast_math/detail/bit_ops_advanced_simd_detail.inl"
// ============================================================================
// Note on other operations
// ============================================================================

/*
 * The following operations do NOT have SIMD versions because scalar is optimal:
 *
 * - reverseBits64/32/16/8:
 *   SWAR algorithm is already maximally parallel within a word
 *   AVX2 cannot improve single-word operations
 *
 * - pext/pdep (BMI2):
 *   Hardware instruction is 3 cycles, cannot be beat
 *   Software fallback is O(popcount), bit-serial by nature
 *
 * - parity:
 *   Requires XOR reduction across all words
 *   SIMD XOR saves operations but needs horizontal reduction
 *   Overhead > benefit for typical bitset sizes
 *
 * - Gray code conversion:
 *   Single-word SWAR operations, inherently parallel
 *
 * - Morton code (interleave/deinterleave):
 *   Single-word bit manipulation, SWAR is optimal
 *
 * - nextPermutation (Gosper's hack):
 *   Bit-serial algorithm, not parallelizable
 *
 * - scatter/gather:
 *   Random access pattern, SIMD gather/scatter not beneficial
 *   (requires AVX512 for efficient gather, still slower than scalar)
 *
 * For these operations, always use the scalar versions from bit_ops_advanced.h
 *
 * Example:
 * @code
 * // These are already optimal (scalar)
 * uint64_t reversed = reverseBits64(value);  // SWAR is parallel
 * uint64_t extracted = pext64(src, mask);    // BMI2 single instruction
 * bool odd_parity = parity(bs);              // XOR reduction
 * uint64_t morton = interleaveBits32(x, y);  // SWAR bit interleave
 * @endcode
 */

} // namespace BitOps
} // namespace Melosyne
}
