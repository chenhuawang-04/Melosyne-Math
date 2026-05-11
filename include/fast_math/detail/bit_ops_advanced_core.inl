// Shared advanced bit manipulation implementation for header/module parity.

// ============================================================================
// Byte swap helpers (compiler intrinsics)
// ============================================================================

namespace detail {

[[nodiscard]] inline constexpr uint64_t advancedLowBitsMask(std::size_t bit_count) noexcept {
    if (bit_count == 0) return 0;
    if (bit_count >= 64) return ~uint64_t{0};
    return (uint64_t{1} << bit_count) - 1;
}

[[nodiscard]] inline constexpr uint64_t advancedLastWordMask(std::size_t bit_count) noexcept {
    const std::size_t remainder = bit_count % 64;
    return remainder == 0 ? ~uint64_t{0} : advancedLowBitsMask(remainder);
}

inline void advancedMaskUnusedBits(BitSetView view) noexcept {
    if (view.word_count == 0) return;
    if ((view.bit_count % 64) != 0) {
        view.data[view.word_count - 1] &= advancedLastWordMask(view.bit_count);
    }
}

[[nodiscard]] inline uint64_t advancedLoadWordMasked(
    const uint64_t* data,
    std::size_t word_count,
    std::size_t bit_count,
    std::size_t word_idx) noexcept {

    if (word_idx >= word_count) return 0;

    uint64_t word = data[word_idx];
    if (word_idx + 1 == word_count) {
        word &= advancedLastWordMask(bit_count);
    }
    return word;
}

[[nodiscard]] inline uint64_t advancedExtractLinearBits(
    const uint64_t* data,
    std::size_t word_count,
    std::size_t bit_count,
    std::size_t start_bit,
    std::size_t length) noexcept {

    if (length == 0 || start_bit >= bit_count) return 0;

    const std::size_t word_idx = start_bit / 64;
    const std::size_t bit_idx = start_bit % 64;

    uint64_t result = advancedLoadWordMasked(data, word_count, bit_count, word_idx) >> bit_idx;
    if (bit_idx != 0 && length > (64 - bit_idx)) {
        result |= advancedLoadWordMasked(data, word_count, bit_count, word_idx + 1)
               << (64 - bit_idx);
    }

    if (length < 64) {
        result &= advancedLowBitsMask(length);
    }
    return result;
}

[[nodiscard]] inline uint64_t advancedReadRotatedWord(
    const uint64_t* data,
    std::size_t word_count,
    std::size_t bit_count,
    std::size_t start_bit) noexcept {

    if (bit_count == 0) return 0;

    start_bit %= bit_count;
    const std::size_t first_len = std::min<std::size_t>(64, bit_count - start_bit);

    uint64_t result = advancedExtractLinearBits(
        data, word_count, bit_count, start_bit, first_len);

    if (first_len < 64) {
        const std::size_t second_len = std::min<std::size_t>(64 - first_len, bit_count);
        result |= advancedExtractLinearBits(
            data, word_count, bit_count, 0, second_len) << first_len;
    }

    return result;
}

inline void advancedRotateLeftBuffered(BitSetView view, std::size_t shift) noexcept {
    if (shift == 0 || view.bit_count == 0) return;
    shift %= view.bit_count;
    if (shift == 0) return;

    void* raw = std::malloc(view.word_count * sizeof(uint64_t));
    if (raw == nullptr) {
        return;
    }

    uint64_t* temp = static_cast<uint64_t*>(raw);
    std::memcpy(temp, view.data, view.word_count * sizeof(uint64_t));

    for (std::size_t word_idx = 0; word_idx < view.word_count; ++word_idx) {
        const std::size_t dst_start = word_idx * 64;
        const std::size_t src_start = (dst_start + view.bit_count - shift) % view.bit_count;
        view.data[word_idx] = advancedReadRotatedWord(
            temp, view.word_count, view.bit_count, src_start);
    }

    advancedMaskUnusedBits(view);
    std::free(raw);
}

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
    v = ((v >> 1) & 0x5555555555555555ULL) | ((v & 0x5555555555555555ULL) << 1);
    v = ((v >> 2) & 0x3333333333333333ULL) | ((v & 0x3333333333333333ULL) << 2);
    v = ((v >> 4) & 0x0F0F0F0F0F0F0F0FULL) | ((v & 0x0F0F0F0F0F0F0F0FULL) << 4);
    return detail::byteswap64(v);
}

[[nodiscard]] inline uint32_t reverseBits32(uint32_t v) noexcept {
    v = ((v >> 1) & 0x55555555U) | ((v & 0x55555555U) << 1);
    v = ((v >> 2) & 0x33333333U) | ((v & 0x33333333U) << 2);
    v = ((v >> 4) & 0x0F0F0F0FU) | ((v & 0x0F0F0F0FU) << 4);
    return detail::byteswap32(v);
}

[[nodiscard]] inline uint16_t reverseBits16(uint16_t v) noexcept {
    v = ((v >> 1) & 0x5555U) | ((v & 0x5555U) << 1);
    v = ((v >> 2) & 0x3333U) | ((v & 0x3333U) << 2);
    v = ((v >> 4) & 0x0F0FU) | ((v & 0x0F0FU) << 4);
    return detail::byteswap16(v);
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

    detail::advancedMaskUnusedBits(view);

    for (std::size_t i = 0; i < view.word_count; ++i) {
        view.data[i] = reverseBits64(view.data[i]);
    }

    for (std::size_t i = 0; i < view.word_count / 2; ++i) {
        std::swap(view.data[i], view.data[view.word_count - 1 - i]);
    }

    const std::size_t unused_bits = view.word_count * 64 - view.bit_count;
    if (unused_bits > 0) {
        for (std::size_t i = 0; i < view.word_count - 1; ++i) {
            view.data[i] = (view.data[i] >> unused_bits) |
                          (view.data[i + 1] << (64 - unused_bits));
        }
        view.data[view.word_count - 1] >>= unused_bits;
    }

    detail::advancedMaskUnusedBits(view);
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
 * - Treat valid bit_count bits as a circular buffer
 * - Uses a buffered word-wise remap to avoid padding-bit contamination
 *
 * Performance: O(n) where n = number of words
 */
inline void rotateLeft(BitSetView view, std::size_t shift) noexcept {
    detail::advancedRotateLeftBuffered(view, shift);
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
        xor_all ^= detail::advancedLoadWordMasked(
            view.data, view.word_count, view.bit_count, i);
    }

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

[[nodiscard]] inline uint64_t pext64(uint64_t src, uint64_t mask) noexcept {
    uint64_t result = 0;
    uint64_t bit_pos = 0;

    while (mask != 0) {
        const std::size_t idx = std::countr_zero(mask);
        if (src & (uint64_t{1} << idx)) {
            result |= (uint64_t{1} << bit_pos);
        }
        ++bit_pos;
        mask &= mask - 1;
    }

    return result;
}

[[nodiscard]] inline uint32_t pext32(uint32_t src, uint32_t mask) noexcept {
    uint32_t result = 0;
    uint32_t bit_pos = 0;

    while (mask != 0) {
        const std::size_t idx = std::countr_zero(mask);
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

[[nodiscard]] inline uint64_t pdep64(uint64_t src, uint64_t mask) noexcept {
    uint64_t result = 0;
    uint64_t src_bit = 0;

    while (mask != 0) {
        const std::size_t idx = std::countr_zero(mask);
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
        const std::size_t idx = std::countr_zero(mask);
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
    auto expand = [](uint32_t v) -> uint64_t {
        uint64_t expanded = v;
        expanded = (expanded | (expanded << 16)) & 0x0000FFFF0000FFFFULL;
        expanded = (expanded | (expanded << 8))  & 0x00FF00FF00FF00FFULL;
        expanded = (expanded | (expanded << 4))  & 0x0F0F0F0F0F0F0F0FULL;
        expanded = (expanded | (expanded << 2))  & 0x3333333333333333ULL;
        expanded = (expanded | (expanded << 1))  & 0x5555555555555555ULL;
        return expanded;
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
 * @return Next value with same number of set bits, or 0 on overflow/sentinel
 *
 * Use case: Iterate all k-bit combinations
 *
 * Example:
 * @code
 * // Iterate all 3-bit combinations of 5 bits
 * for (uint64_t v = 0b00111; v < 32; v = nextPermutation(v)) {
 *     // Process combination
 * }
 * @endcode
 *
 * Reference: Gosper's hack (HAKMEM 175)
 */
[[nodiscard]] inline uint64_t nextPermutation(uint64_t v) noexcept {
    if (v == 0) return 0;

    const uint64_t c = v & (~v + 1);
    const uint64_t r = v + c;
    if (r == 0) return 0;

    return (((r ^ v) >> 2) / c) | r;
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
    for (std::size_t i = 0; i < count && i < 64; ++i) {
        const std::size_t pos = indices[i];
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
        const std::size_t pos = indices[i];
        if (pos >= src.bit_count) continue;

        if (src.data[pos / 64] & (uint64_t{1} << (pos % 64))) {
            result |= (uint64_t{1} << i);
        }
    }

    return result;
}
