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

export module fast_math.bitset_static;

export {
/**
 * @file bitset_static.h
 * @brief Compile-time fixed-size bitset (STRICT POD, NO member functions)
 *
 * Design Philosophy:
 * - STRICT POD (Plain Old Data) - C-compatible
 * - NO constructors, NO destructors, NO member functions
 * - Manipulated ONLY by free functions in bit_ops.h and bit_ops_simd.h
 * - Cache-line aligned for SIMD performance (32-byte alignment)
 *
 * POD guarantees:
 * - Can be memcpy'd safely
 * - Can be used in unions
 * - Compatible with C interfaces  
 * - Trivially copyable and standard layout
 *
 * Usage:
 * @code
 * BitSet<256> bs = {};  // Zero-initialize (C-style)
 * bitOpsSet(bs, 42);    // Set bit 42 using free function
 * bool is_set = bitOpsTest(bs, 42);  // Test bit 42
 * @endcode
 */




namespace Melosyne {

/**
 * @brief Compile-time fixed-size bitset (STRICT POD, N bits)
 * @tparam N Number of bits (must be > 0)
 *
 * Memory layout:
 * - Stack-allocated array of uint64_t words
 * - Size: ((N + 63) / 64) * 8 bytes
 * - Alignment: 32 bytes (for AVX2 SIMD operations)
 *
 * Example sizes:
 * - BitSet<64>:   8 bytes (1 word)
 * - BitSet<256>: 32 bytes (4 words)
 * - BitSet<512>: 64 bytes (8 words)
 *
 * POD verification:
 * - static_assert(std::is_trivial_v<BitSet<256>>)
 * - static_assert(std::is_standard_layout_v<BitSet<256>>)
 */
template<std::size_t N>
struct alignas(32) BitSet {
    static_assert(N > 0, "BitSet size must be positive");

    static constexpr std::size_t kBits = N;   ///< Total bits
    static constexpr std::size_t kWords = (N + 63) / 64;  ///< Number of words
    static constexpr std::size_t kBytes = kWords * 8;     ///< Total bytes

    uint64_t words[kWords];  ///< Word array (pure data)
};

// ============================================================================
// POD verification (compile-time checks)
// ============================================================================

static_assert(std::is_trivial_v<BitSet<64>>,   "BitSet<64> must be trivial");
static_assert(std::is_trivial_v<BitSet<256>>,  "BitSet<256> must be trivial");
static_assert(std::is_trivial_v<BitSet<512>>,  "BitSet<512> must be trivial");
static_assert(std::is_trivial_v<BitSet<1024>>, "BitSet<1024> must be trivial");

static_assert(std::is_standard_layout_v<BitSet<64>>,   "BitSet<64> must be standard layout");
static_assert(std::is_standard_layout_v<BitSet<256>>,  "BitSet<256> must be standard layout");
static_assert(std::is_standard_layout_v<BitSet<512>>,  "BitSet<512> must be standard layout");
static_assert(std::is_standard_layout_v<BitSet<1024>>, "BitSet<1024> must be standard layout");

} // namespace Melosyne
}
