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

export module fast_math.bit_ops_range;

export import fast_math.bitset_view;

export {
/**
 * @file bit_ops_range.h
 * @brief Scalar bit range operations (no SIMD intrinsics)
 *
 * Design Philosophy:
 * - FREE FUNCTIONS only (no methods)
 * - Word-aligned range optimization (set entire words at once)
 * - SWAR masks for partial words
 * - Simple loops for maximum portability
 *
 * Implemented Operations:
 * - setRange/resetRange/flipRange: Modify bits in range [start, end)
 * - testAll/testAny/testNone: Query if all/any/none bits are set in range
 * - popcountRange: Count set bits in range
 *
 * Optimization Techniques:
 * - Handle partial first/last words with bit masks
 * - Process full words in middle with direct assignment
 * - Up to 64x faster than bit-by-bit loops
 *
 * Performance:
 * - Single word range: O(1) - single mask operation
 * - Multi-word range: O(n) where n = number of words in range
 * - Best case: Aligned range, all full words
 *
 * References:
 * - Bit Twiddling Hacks: https://graphics.stanford.edu/~seander/bithacks.html
 * - SWAR techniques: https://aggregate.org/MAGIC/
 */



namespace Melosyne {
namespace BitOps {
#include "fast_math/detail/bit_ops_range_core.inl"
} // namespace BitOps
} // namespace Melosyne
}
