// Shared BitSetView / ConstBitSetView definitions for header/module parity.

// ============================================================================
// BitSetView - Mutable view over bitset data
// ============================================================================

/**
 * @brief Mutable view over bitset data (similar to std::span)
 *
 * Provides a non-owning view over a bitset's underlying word array.
 * Enables algorithms to work uniformly with BitSet<N> and DynamicBitSet.
 *
 * Memory Layout (24 bytes):
 * - uint64_t* data      : 8 bytes (pointer to words)
 * - size_t bit_count    : 8 bytes (total bits)
 * - size_t word_count   : 8 bytes (number of 64-bit words)
 *
 * Pass-by-value semantics:
 * - Fits in 3 registers (RDI, RSI, RDX on x86-64)
 * - No heap allocation
 * - Compiler fully inlines operations
 */
struct BitSetView {
    uint64_t* data;           ///< Pointer to word array
    std::size_t bit_count;    ///< Total number of bits
    std::size_t word_count;   ///< Number of 64-bit words

    /**
     * @brief Default constructor - creates empty view
     */
    constexpr BitSetView() noexcept
        : data(nullptr), bit_count(0), word_count(0) {}

    /**
     * @brief Construct view from raw pointer and bit count
     * @param data_ Pointer to uint64_t word array
     * @param bits_ Number of bits in the bitset
     */
    constexpr BitSetView(uint64_t* data_, std::size_t bits_) noexcept
        : data(data_), bit_count(bits_), word_count((bits_ + 63) / 64) {}

    /**
     * @brief Implicit construction from BitSet<N>
     * @param bs Reference to compile-time fixed-size bitset
     * @note Implicit conversion enables seamless use in algorithms
     */
    template<std::size_t N>
    constexpr BitSetView(BitSet<N>& bs) noexcept
        : data(bs.words), bit_count(N), word_count(BitSet<N>::kWords) {}

    /**
     * @brief Implicit construction from DynamicBitSet
     * @param bs Reference to runtime-sized bitset
     * @note Implementation defined after DynamicBitSet is complete.
     */
    BitSetView(DynamicBitSet& bs) noexcept;

    /**
     * @brief Get total number of bits
     * @return Number of bits in the viewed bitset
     */
    [[nodiscard]] constexpr std::size_t size() const noexcept {
        return bit_count;
    }

    /**
     * @brief Get number of 64-bit words
     * @return Number of uint64_t words
     */
    [[nodiscard]] constexpr std::size_t words() const noexcept {
        return word_count;
    }

    /**
     * @brief Check if view is empty
     * @return true if bit_count == 0
     */
    [[nodiscard]] constexpr bool empty() const noexcept {
        return bit_count == 0;
    }

    /**
     * @brief Create a subview (slice) of the bitset
     * @param offset_bits Starting bit position
     * @param length_bits Number of bits in the subview
     * @return New BitSetView representing the subrange
     * @note For best performance, offset_bits should be word-aligned (multiple of 64)
     */
    [[nodiscard]] constexpr BitSetView subview(
        std::size_t offset_bits,
        std::size_t length_bits) noexcept {

        std::size_t word_offset = offset_bits / 64;
        return BitSetView(data + word_offset, length_bits);
    }
};

// ============================================================================
// ConstBitSetView - Immutable view over bitset data
// ============================================================================

/**
 * @brief Immutable (const) view over bitset data
 *
 * Read-only variant of BitSetView. Automatically constructed from:
 * - const BitSet<N>&
 * - const DynamicBitSet&
 * - BitSetView (mutable -> immutable conversion)
 *
 * Same memory layout as BitSetView (24 bytes), but with const pointer.
 */
struct ConstBitSetView {
    const uint64_t* data;     ///< Const pointer to word array
    std::size_t bit_count;    ///< Total number of bits
    std::size_t word_count;   ///< Number of 64-bit words

    /**
     * @brief Default constructor - creates empty view
     */
    constexpr ConstBitSetView() noexcept
        : data(nullptr), bit_count(0), word_count(0) {}

    /**
     * @brief Construct view from raw const pointer and bit count
     * @param data_ Const pointer to uint64_t word array
     * @param bits_ Number of bits in the bitset
     */
    constexpr ConstBitSetView(const uint64_t* data_, std::size_t bits_) noexcept
        : data(data_), bit_count(bits_), word_count((bits_ + 63) / 64) {}

    /**
     * @brief Implicit construction from const BitSet<N>
     * @param bs Const reference to compile-time fixed-size bitset
     */
    template<std::size_t N>
    constexpr ConstBitSetView(const BitSet<N>& bs) noexcept
        : data(bs.words), bit_count(N), word_count(BitSet<N>::kWords) {}

    /**
     * @brief Implicit construction from const DynamicBitSet
     * @param bs Const reference to runtime-sized bitset
     * @note Implementation defined after DynamicBitSet is complete.
     */
    ConstBitSetView(const DynamicBitSet& bs) noexcept;

    /**
     * @brief Implicit construction from mutable BitSetView
     * @param view Mutable view (automatically converts to const)
     * @note Allows passing mutable views to functions expecting const views
     */
    constexpr ConstBitSetView(BitSetView view) noexcept
        : data(view.data), bit_count(view.bit_count),
          word_count(view.word_count) {}

    /**
     * @brief Get total number of bits
     * @return Number of bits in the viewed bitset
     */
    [[nodiscard]] constexpr std::size_t size() const noexcept {
        return bit_count;
    }

    /**
     * @brief Get number of 64-bit words
     * @return Number of uint64_t words
     */
    [[nodiscard]] constexpr std::size_t words() const noexcept {
        return word_count;
    }

    /**
     * @brief Check if view is empty
     * @return true if bit_count == 0
     */
    [[nodiscard]] constexpr bool empty() const noexcept {
        return bit_count == 0;
    }

    /**
     * @brief Create a const subview (slice) of the bitset
     * @param offset_bits Starting bit position
     * @param length_bits Number of bits in the subview
     * @return New ConstBitSetView representing the subrange
     * @note For best performance, offset_bits should be word-aligned (multiple of 64)
     */
    [[nodiscard]] constexpr ConstBitSetView subview(
        std::size_t offset_bits,
        std::size_t length_bits) const noexcept {

        std::size_t word_offset = offset_bits / 64;
        return ConstBitSetView(data + word_offset, length_bits);
    }
};
