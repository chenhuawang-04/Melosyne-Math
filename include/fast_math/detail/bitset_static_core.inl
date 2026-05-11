// Shared BitSet<N> definition for header/module parity.

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

    static constexpr std::size_t kBits = N;                 ///< Total bits
    static constexpr std::size_t kWords = (N + 63) / 64;    ///< Number of words
    static constexpr std::size_t kBytes = kWords * 8;       ///< Total bytes

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
