// Shared DynamicBitSet-backed BitSetView constructors for header/module parity.

/**
 * @brief BitSetView constructor implementation for DynamicBitSet
 */
inline BitSetView::BitSetView(DynamicBitSet& bs) noexcept
    : data(bs.data()), bit_count(bs.size()),
      word_count(bs.wordCount()) {}

/**
 * @brief ConstBitSetView constructor implementation for DynamicBitSet
 */
inline ConstBitSetView::ConstBitSetView(const DynamicBitSet& bs) noexcept
    : data(bs.data()), bit_count(bs.size()),
      word_count(bs.wordCount()) {}
