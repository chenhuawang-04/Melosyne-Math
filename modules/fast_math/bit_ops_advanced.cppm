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
#include <cstdlib>
#include <immintrin.h>
#include <stdlib.h>

export module fast_math.bit_ops_advanced;

export import fast_math.bitset_view;

export {
/**
 * @file bit_ops_advanced.h
 * @brief Scalar advanced bit manipulation algorithms (no SIMD intrinsics)
 *
 * Design Philosophy:
 * - FREE FUNCTIONS only (no methods)
 * - SWAR (SIMD Within A Register) techniques for parallel bit operations
 * - BMI2 intrinsics (PEXT/PDEP) with portable software fallbacks
 * - Constexpr support for compile-time evaluation where possible
 *
 * Implemented Operations:
 * - Bit reversal: SWAR algorithm for parallel bit swapping
 * - Rotation: Circular shift (left/right)
 * - Byte swap: Endianness conversion
 * - Parity: XOR of all bits (odd/even)
 * - PEXT/PDEP: Parallel bit extract/deposit (BMI2)
 * - Gray code: Binary ↔ Gray code conversion
 * - Morton code: 2D space-filling curve (Z-order)
 * - Permutation: Next combination with same popcount (Gosper's hack)
 * - Scatter/Gather: Distribute/collect bits by index
 *
 * Performance:
 * - SWAR operations: ~5-10 cycles per word
 * - BMI2 PEXT/PDEP: 3 cycles latency (Haswell+), software fallback O(popcount(mask))
 * - Byte swap: 1 cycle (BSWAP instruction)
 *
 * References:
 * - Bit Twiddling Hacks: https://graphics.stanford.edu/~seander/bithacks.html
 * - SWAR Magic: https://aggregate.org/MAGIC/
 * - Bit Permutations: https://programming.sirrida.de/bit_perm.html
 */



#if defined(__BMI2__)
#endif

#if defined(_MSC_VER)
#endif

namespace Melosyne {
namespace BitOps {
#include "fast_math/detail/bit_ops_advanced_core.inl"
} // namespace BitOps
} // namespace Melosyne
}
