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

// ============================================================================
// Byte swap helpers (compiler intrinsics)
// ============================================================================

namespace detail {

/**
 * @brief Byte-swap 64-bit integer (compiler intrinsic)
 * @param v Input value
 * @return Byte-swapped value
 *
 * Compiles to BSWAP instruction (1 cycle latency)
 */
inline uint64_t byteswap64(uint64_t v) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_bswap64(v);
#elif defined(_MSC_VER)
    return _byteswap_uint64(v);
#else
    // Portable fallback
    return ((v & 0xFF00000000000000ULL) >> 56) |
           ((v & 0x00FF000000000000ULL) >> 40) |
           ((v & 0x0000FF0000000000ULL) >> 24) |
           ((v & 0x000000FF00000000ULL) >> 8)  |
           ((v & 0x00000000FF000000ULL) << 8)  |
           ((v & 0x0000000000FF0000ULL) << 24) |
           ((v & 0x000000000000FF00ULL) << 40) |
           ((v & 0x00000000000000FFULL) << 56);
#endif
}

inline uint32_t byteswap32(uint32_t v) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_bswap32(v);
#elif defined(_MSC_VER)
    return _byteswap_ulong(v);
#else
    return ((v & 0xFF000000U) >> 24) |
           ((v & 0x00FF0000U) >> 8)  |
           ((v & 0x0000FF00U) << 8)  |
           ((v & 0x000000FFU) << 24);
#endif
}

inline uint16_t byteswap16(uint16_t v) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_bswap16(v);
#elif defined(_MSC_VER)
    return _byteswap_ushort(v);
#else
    return ((v & 0xFF00U) >> 8) | ((v & 0x00FFU) << 8);
#endif
}

} // namespace detail

// ============================================================================
// Bit reversal - SWAR algorithm (progressively swap larger groups)
// ============================================================================

/**
 * @brief Reverse bits in 64-bit word using SWAR algorithm
 * @param v Input value
 * @return Bit-reversed value
 *
 * Algorithm: Progressively swap larger groups of bits
 * 1. Swap odd/even bits (1-bit groups)
 * 2. Swap 2-bit groups
 * 3. Swap 4-bit groups (nibbles)
 * 4. Swap bytes (use BSWAP)
 *
 * Performance: ~5 cycles on modern CPUs
 *
 * Example:
 * @code
 * reverseBits64(0x0123456789ABCDEF) = 0xF7B3D591E6A2C480
 * @endcode
 *
 * Reference: https://graphics.stanford.edu/~seander/bithacks.html#ReverseParallel
 */
[[nodiscard]] inline uint64_t reverseBits64(uint64_t v) noexcept {
    // Swap odd/even bits
    v = ((v >> 1) & 0x5555555555555555ULL) | ((v & 0x5555555555555555ULL) << 1);
    // Swap consecutive pairs
    v = ((v >> 2) & 0x3333333333333333ULL) | ((v & 0x3333333333333333ULL) << 2);
    // Swap nibbles
    v = ((v >> 4) & 0x0F0F0F0F0F0F0F0FULL) | ((v & 0x0F0F0F0F0F0F0F0FULL) << 4);
    // Swap bytes (BSWAP)
    v = detail::byteswap64(v);
    return v;
}

[[nodiscard]] inline uint32_t reverseBits32(uint32_t v) noexcept {
    v = ((v >> 1) & 0x55555555U) | ((v & 0x55555555U) << 1);
    v = ((v >> 2) & 0x33333333U) | ((v & 0x33333333U) << 2);
    v = ((v >> 4) & 0x0F0F0F0FU) | ((v & 0x0F0F0F0FU) << 4);
    v = detail::byteswap32(v);
    return v;
}

[[nodiscard]] inline uint16_t reverseBits16(uint16_t v) noexcept {
    v = ((v >> 1) & 0x5555U) | ((v & 0x5555U) << 1);
    v = ((v >> 2) & 0x3333U) | ((v & 0x3333U) << 2);
    v = ((v >> 4) & 0x0F0FU) | ((v & 0x0F0FU) << 4);
    v = detail::byteswap16(v);
    return v;
}

[[nodiscard]] inline constexpr uint8_t reverseBits8(uint8_t v) noexcept {
    v = ((v >> 1) & 0x55U) | ((v & 0x55U) << 1);
    v = ((v >> 2) & 0x33U) | ((v & 0x33U) << 2);
    v = ((v >> 4) & 0x0FU) | ((v & 0x0FU) << 4);
    return v;
}

/**
 * @brief Reverse entire bitset (bit order and word order) (scalar version)
 * @param view Bitset view (mutable)
 *
 * Algorithm:
 * 1. Reverse bits within each word (using reverseBits64)
 * 2. Reverse word order
 * 3. Shift to align with bit_count (if not multiple of 64)
 *
 * Performance: O(n) where n = number of words
 *
 * Example:
 * @code
 * BitSet<128> bs = {};
 * bs.words[0] = 0x0F;  // Bits 0-3 set
 * reverse(bs);
 * // Now bs.words[1] has bits 60-63 set (mirrored position)
 * @endcode
 */
inline void reverse(BitSetView view) noexcept {
    if (view.word_count == 0) return;

    // Reverse each word in-place
    for (std::size_t i = 0; i < view.word_count; ++i) {
        view.data[i] = reverseBits64(view.data[i]);
    }

    // Reverse word order
    for (std::size_t i = 0; i < view.word_count / 2; ++i) {
        std::swap(view.data[i], view.data[view.word_count - 1 - i]);
    }

    // Shift to align with bit_count (if not a multiple of 64)
    std::size_t unused_bits = view.word_count * 64 - view.bit_count;
    if (unused_bits > 0 && view.word_count > 0) {
        // Right shift entire bitset by unused_bits
        for (std::size_t i = 0; i < view.word_count - 1; ++i) {
            view.data[i] = (view.data[i] >> unused_bits) |
                          (view.data[i + 1] << (64 - unused_bits));
        }
        view.data[view.word_count - 1] >>= unused_bits;
    }
}

// ============================================================================
// Rotate operations (circular shift)
// ============================================================================

/**
 * @brief Rotate bitset left (circular shift) (scalar version)
 * @param view Bitset view (mutable)
 * @param shift Number of positions to rotate left
 *
 * Algorithm:
 * - Treats bitset as circular buffer
 * - Handles word-level and bit-level shifts
 * - Uses temporary buffer to avoid overwriting
 *
 * Performance: O(n) where n = number of words
 *
 * Example:
 * @code
 * BitSet<128> bs = {};
 * bs.words[0] = 0x0F;  // Bits 0-3 set
 * rotateLeft(bs, 4);
 * // Now bs.words[0] = 0xF0 (bits 4-7 set)
 * @endcode
 */
inline void rotateLeft(BitSetView view, std::size_t shift) noexcept {
    if (shift == 0 || view.bit_count == 0) return;
    shift %= view.bit_count; // Handle shift > bit_count
    if (shift == 0) return;

    std::size_t word_shift = shift / 64;
    std::size_t bit_shift = shift % 64;

    // Allocate temp buffer for rotated data
    uint64_t* temp = new uint64_t[view.word_count];
    std::memcpy(temp, view.data, view.word_count * sizeof(uint64_t));

    // Rotate
    for (std::size_t i = 0; i < view.word_count; ++i) {
        std::size_t src_idx = (i + view.word_count - word_shift) % view.word_count;
        std::size_t next_idx = (src_idx + view.word_count - 1) % view.word_count;

        if (bit_shift == 0) {
            view.data[i] = temp[src_idx];
        } else {
            view.data[i] = (temp[src_idx] << bit_shift) |
                          (temp[next_idx] >> (64 - bit_shift));
        }
    }

    delete[] temp;
}

/**
 * @brief Rotate bitset right (circular shift) (scalar version)
 * @param view Bitset view (mutable)
 * @param shift Number of positions to rotate right
 *
 * Implementation: Right rotation = left rotation by (bit_count - shift)
 */
inline void rotateRight(BitSetView view, std::size_t shift) noexcept {
    if (shift == 0 || view.bit_count == 0) return;
    shift %= view.bit_count;
    if (shift == 0) return;

    rotateLeft(view, view.bit_count - shift);
}

// ============================================================================
// Byte swap (endianness conversion)
// ============================================================================

/**
 * @brief Swap byte order of all words in bitset (scalar version)
 * @param view Bitset view (mutable)
 *
 * Use case: Endianness conversion (little ↔ big endian)
 *
 * Performance: O(n) where n = number of words, 1 cycle per word (BSWAP)
 */
inline void byteSwap(BitSetView view) noexcept {
    for (std::size_t i = 0; i < view.word_count; ++i) {
        view.data[i] = detail::byteswap64(view.data[i]);
    }
}

// ============================================================================
// Parity calculation (XOR of all bits)
// ============================================================================

/**
 * @brief Calculate parity of bitset (XOR of all bits) (scalar version)
 * @param view Bitset view (read-only)
 * @return true if odd number of bits are set, false if even
 *
 * Algorithm:
 * 1. XOR all words together
 * 2. Reduce 64-bit result to 1 bit using parallel XOR (SWAR)
 *
 * Performance: O(n) XOR operations, then O(log 64) = 6 shifts
 *
 * Use cases:
 * - Error detection (even/odd parity)
 * - Cryptographic checksums
 */
[[nodiscard]] inline bool parity(ConstBitSetView view) noexcept {
    uint64_t xor_all = 0;

    for (std::size_t i = 0; i < view.word_count; ++i) {
        xor_all ^= view.data[i];
    }

    // Reduce 64 bits to 1 bit using parallel XOR (SWAR)
    xor_all ^= xor_all >> 32;
    xor_all ^= xor_all >> 16;
    xor_all ^= xor_all >> 8;
    xor_all ^= xor_all >> 4;
    xor_all ^= xor_all >> 2;
    xor_all ^= xor_all >> 1;

    return (xor_all & 1) != 0;
}

// ============================================================================
// Parallel bit extract (PEXT) - BMI2 instruction
// ============================================================================

/**
 * @brief Extract bits from 'src' at positions where 'mask' has 1s
 * @param src Source value
 * @param mask Bit mask specifying which bits to extract
 * @return Compacted bits (extracted bits moved to contiguous low positions)
 *
 * Algorithm (hardware BMI2):
 * - PEXT instruction: 3 cycles latency (Haswell+)
 *
 * Algorithm (software fallback):
 * - Iterate through mask bits, extract corresponding src bits
 * - O(popcount(mask)) complexity
 *
 * Example:
 * @code
 * pext64(0b10110010, 0b11110000) = 0b00001011
 * // Extracts bits at positions 4-7 -> compacted to bits 0-3
 * @endcode
 *
 * Use cases:
 * - Bit field extraction
 * - Parallel bit gather
 * - Subset selection
 */
#if defined(__BMI2__)

[[nodiscard]] inline uint64_t pext64(uint64_t src, uint64_t mask) noexcept {
    return _pext_u64(src, mask);
}

[[nodiscard]] inline uint32_t pext32(uint32_t src, uint32_t mask) noexcept {
    return _pext_u32(src, mask);
}

#else

// Software fallback (slower but portable)
[[nodiscard]] inline uint64_t pext64(uint64_t src, uint64_t mask) noexcept {
    uint64_t result = 0;
    uint64_t bit_pos = 0;

    while (mask != 0) {
        // Find next set bit in mask
        std::size_t idx = std::countr_zero(mask);
        // Extract corresponding bit from src
        if (src & (uint64_t{1} << idx)) {
            result |= (uint64_t{1} << bit_pos);
        }
        ++bit_pos;
        // Clear this bit from mask
        mask &= mask - 1; // Clear lowest set bit
    }

    return result;
}

[[nodiscard]] inline uint32_t pext32(uint32_t src, uint32_t mask) noexcept {
    uint32_t result = 0;
    uint32_t bit_pos = 0;

    while (mask != 0) {
        std::size_t idx = std::countr_zero(mask);
        if (src & (uint32_t{1} << idx)) {
            result |= (uint32_t{1} << bit_pos);
        }
        ++bit_pos;
        mask &= mask - 1;
    }

    return result;
}

#endif

// ============================================================================
// Parallel bit deposit (PDEP) - BMI2 instruction
// ============================================================================

/**
 * @brief Deposit bits from 'src' to positions where 'mask' has 1s
 * @param src Source value (low bits)
 * @param mask Bit mask specifying where to deposit bits
 * @return Result with src bits deposited at mask positions
 *
 * Algorithm: Inverse of PEXT
 * - Hardware: PDEP instruction (3 cycles)
 * - Software: O(popcount(mask))
 *
 * Example:
 * @code
 * pdep64(0b00001011, 0b11110000) = 0b10110000
 * // Deposits bits 0-3 from src -> to positions 4-7 (where mask is set)
 * @endcode
 *
 * Use cases:
 * - Bit field insertion
 * - Parallel bit scatter
 * - Subset expansion
 */
#if defined(__BMI2__)

[[nodiscard]] inline uint64_t pdep64(uint64_t src, uint64_t mask) noexcept {
    return _pdep_u64(src, mask);
}

[[nodiscard]] inline uint32_t pdep32(uint32_t src, uint32_t mask) noexcept {
    return _pdep_u32(src, mask);
}

#else

// Software fallback
[[nodiscard]] inline uint64_t pdep64(uint64_t src, uint64_t mask) noexcept {
    uint64_t result = 0;
    uint64_t src_bit = 0;

    while (mask != 0) {
        std::size_t idx = std::countr_zero(mask);
        if (src & (uint64_t{1} << src_bit)) {
            result |= (uint64_t{1} << idx);
        }
        ++src_bit;
        mask &= mask - 1;
    }

    return result;
}

[[nodiscard]] inline uint32_t pdep32(uint32_t src, uint32_t mask) noexcept {
    uint32_t result = 0;
    uint32_t src_bit = 0;

    while (mask != 0) {
        std::size_t idx = std::countr_zero(mask);
        if (src & (uint32_t{1} << src_bit)) {
            result |= (uint32_t{1} << idx);
        }
        ++src_bit;
        mask &= mask - 1;
    }

    return result;
}

#endif

// ============================================================================
// Gray code conversion
// ============================================================================

/**
 * @brief Convert binary to Gray code
 * @param binary Binary value
 * @return Gray code value
 *
 * Algorithm: gray = binary XOR (binary >> 1)
 *
 * Property: Adjacent Gray codes differ by exactly 1 bit
 *
 * Use cases:
 * - Rotary encoders (minimize errors)
 * - Genetic algorithms (smooth mutations)
 * - Error correction
 */
[[nodiscard]] inline constexpr uint64_t binaryToGray(uint64_t binary) noexcept {
    return binary ^ (binary >> 1);
}

/**
 * @brief Convert Gray code to binary
 * @param gray Gray code value
 * @return Binary value
 *
 * Algorithm: Parallel prefix XOR (SWAR technique)
 */
[[nodiscard]] inline constexpr uint64_t grayToBinary(uint64_t gray) noexcept {
    // Parallel prefix XOR
    gray ^= gray >> 32;
    gray ^= gray >> 16;
    gray ^= gray >> 8;
    gray ^= gray >> 4;
    gray ^= gray >> 2;
    gray ^= gray >> 1;
    return gray;
}

// ============================================================================
// Morton code (Z-order curve) - interleave bits for spatial indexing
// ============================================================================

/**
 * @brief Interleave bits of x and y to create 2D Morton code
 * @param x X coordinate (32-bit)
 * @param y Y coordinate (32-bit)
 * @return 64-bit Morton code (Z-order curve index)
 *
 * Algorithm: Expand each coordinate to 64 bits with gaps, then interleave
 *
 * Use cases:
 * - Quadtree indexing
 * - Spatial databases
 * - Cache-coherent 2D iteration
 *
 * Example:
 * @code
 * interleaveBits32(0b101, 0b110) = 0b110011
 * // x bits at even positions, y bits at odd positions
 * @endcode
 */
[[nodiscard]] inline constexpr uint64_t interleaveBits32(uint32_t x, uint32_t y) noexcept {
    // Expand x and y to 64-bit with gaps
    auto expand = [](uint32_t v) -> uint64_t {
        uint64_t x = v;
        x = (x | (x << 16)) & 0x0000FFFF0000FFFFULL;
        x = (x | (x << 8))  & 0x00FF00FF00FF00FFULL;
        x = (x | (x << 4))  & 0x0F0F0F0F0F0F0F0FULL;
        x = (x | (x << 2))  & 0x3333333333333333ULL;
        x = (x | (x << 1))  & 0x5555555555555555ULL;
        return x;
    };

    return expand(x) | (expand(y) << 1);
}

/**
 * @brief Deinterleave Morton code back to x, y coordinates
 * @param morton 64-bit Morton code
 * @param x Output X coordinate
 * @param y Output Y coordinate
 */
inline constexpr void deinterleaveBits(uint64_t morton, uint32_t& x, uint32_t& y) noexcept {
    auto compact = [](uint64_t v) -> uint32_t {
        v = v & 0x5555555555555555ULL;
        v = (v | (v >> 1))  & 0x3333333333333333ULL;
        v = (v | (v >> 2))  & 0x0F0F0F0F0F0F0F0FULL;
        v = (v | (v >> 4))  & 0x00FF00FF00FF00FFULL;
        v = (v | (v >> 8))  & 0x0000FFFF0000FFFFULL;
        v = (v | (v >> 16)) & 0x00000000FFFFFFFFULL;
        return static_cast<uint32_t>(v);
    };

    x = compact(morton);
    y = compact(morton >> 1);
}

// ============================================================================
// Next/Previous permutation (for iterating combinations)
// ============================================================================

/**
 * @brief Find next number with same popcount (Gosper's hack)
 * @param v Current value
 * @return Next value with same number of set bits
 *
 * Use case: Iterate all k-bit combinations
 *
 * Example:
 * @code
 * // Iterate all 3-bit combinations of 5 bits
 * for (uint64_t v = 0b00111; v < 32; v = nextPermutation(v)) {
 *     // Process combination
 * }
 * // Generates: 00111, 01011, 01101, 01110, 10011, 10101, 10110, 11001, 11010, 11100
 * @endcode
 *
 * Reference: Gosper's hack (HAKMEM 175)
 */
[[nodiscard]] inline uint64_t nextPermutation(uint64_t v) noexcept {
    uint64_t t = v | (v - 1);
    return (t + 1) | (((~t & -~t) - 1) >> (std::countr_zero(v) + 1));
}

// ============================================================================
// Bit scatter/gather operations
// ============================================================================

/**
 * @brief Scatter: Distribute bits from 'src' to positions specified by 'indices'
 * @param dst Destination bitset
 * @param src Source bits (64-bit)
 * @param indices Array of bit positions
 * @param count Number of indices
 *
 * Algorithm: For each index, set dst[indices[i]] = bit i of src
 *
 * Performance: O(count)
 */
inline void scatterBits(BitSetView dst, uint64_t src,
                       const std::size_t* indices, std::size_t count) noexcept {
    for (std::size_t i = 0; i < count; ++i) {
        std::size_t pos = indices[i];
        if (pos >= dst.bit_count) continue;

        if (src & (uint64_t{1} << i)) {
            dst.data[pos / 64] |= (uint64_t{1} << (pos % 64));
        } else {
            dst.data[pos / 64] &= ~(uint64_t{1} << (pos % 64));
        }
    }
}

/**
 * @brief Gather: Collect bits from positions specified by 'indices'
 * @param src Source bitset
 * @param indices Array of bit positions
 * @param count Number of indices (max 64)
 * @return 64-bit value with gathered bits
 *
 * Algorithm: result[i] = src[indices[i]]
 *
 * Performance: O(count)
 */
[[nodiscard]] inline uint64_t gatherBits(ConstBitSetView src,
                                          const std::size_t* indices,
                                          std::size_t count) noexcept {
    uint64_t result = 0;

    for (std::size_t i = 0; i < count && i < 64; ++i) {
        std::size_t pos = indices[i];
        if (pos >= src.bit_count) continue;

        if (src.data[pos / 64] & (uint64_t{1} << (pos % 64))) {
            result |= (uint64_t{1} << i);
        }
    }

    return result;
}

} // namespace BitOps
} // namespace Melosyne
}
