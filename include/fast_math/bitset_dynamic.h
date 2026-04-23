#pragma once

#include "config_macros.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory_resource>

namespace Melosyne {

/**
 * @brief Runtime-sized bitset with small-object optimization and PMR allocator support.
 *
 * Design goals:
 * - Keep data container lightweight; algorithms live in BitOps free functions.
 * - Avoid heap traffic for small bitsets (<= 256 bits).
 * - Keep SIMD-friendly alignment for remote storage (32-byte aligned).
 * - Reuse allocated capacity in assignment/resize paths whenever possible.
 */
class DynamicBitSet {
public:
    static constexpr std::size_t kWordBits = 64;
    static constexpr std::size_t kSimdAlignment = 32;
    static constexpr std::size_t kLocalWords = 4; // 256 bits
    static constexpr std::size_t kLocalBits = kLocalWords * kWordBits;

    // =========================================================================
    // Construction / assignment
    // =========================================================================

    DynamicBitSet() noexcept
        : bit_count_(0),
          word_count_(0),
          capacity_words_(kLocalWords),
          memory_resource_(std::pmr::get_default_resource()),
          is_local_(true),
          remote_alignment_(alignof(uint64_t)) {
        std::memset(local_words_, 0, sizeof(local_words_));
    }

    explicit DynamicBitSet(std::size_t num_bits_)
        : DynamicBitSet(num_bits_, std::pmr::get_default_resource()) {}

    DynamicBitSet(std::size_t num_bits_, std::pmr::memory_resource* mr_)
        : bit_count_(num_bits_),
          word_count_(wordsForBits(num_bits_)),
          capacity_words_(kLocalWords),
          memory_resource_(sanitizeResource(mr_)),
          is_local_(true),
          remote_alignment_(alignof(uint64_t)) {
        std::memset(local_words_, 0, sizeof(local_words_));

        if (word_count_ > kLocalWords) {
            is_local_ = false;
            capacity_words_ = word_count_;
            remote_words_ = allocateWords(capacity_words_, requiredAlignment(capacity_words_));
            std::memset(remote_words_, 0, capacity_words_ * sizeof(uint64_t));
        }
    }

    DynamicBitSet(const DynamicBitSet& other_)
        : bit_count_(other_.bit_count_),
          word_count_(other_.word_count_),
          capacity_words_(other_.word_count_ > kLocalWords ? other_.word_count_ : kLocalWords),
          memory_resource_(other_.memory_resource_),
          is_local_(other_.word_count_ <= kLocalWords),
          remote_alignment_(other_.remote_alignment_) {
        if (is_local_) {
            remote_alignment_ = alignof(uint64_t);
            std::memcpy(local_words_, other_.local_words_, sizeof(local_words_));
        } else {
            remote_words_ = allocateWords(capacity_words_, requiredAlignment(capacity_words_));
            std::memcpy(remote_words_, other_.remote_words_, word_count_ * sizeof(uint64_t));
        }
    }

    DynamicBitSet(DynamicBitSet&& other_) noexcept
        : bit_count_(other_.bit_count_),
          word_count_(other_.word_count_),
          capacity_words_(other_.capacity_words_),
          memory_resource_(other_.memory_resource_),
          is_local_(other_.is_local_),
          remote_alignment_(other_.remote_alignment_) {
        if (is_local_) {
            std::memcpy(local_words_, other_.local_words_, sizeof(local_words_));
        } else {
            remote_words_ = other_.remote_words_;
            other_.remote_words_ = nullptr;
        }
        other_.resetToDefaultState();
    }

    ~DynamicBitSet() {
        releaseRemoteIfAny();
    }

    DynamicBitSet& operator=(const DynamicBitSet& other_) {
        if (this == &other_) {
            return *this;
        }

        const std::size_t other_words = other_.word_count_;
        const bool other_local = other_words <= kLocalWords;

        if (other_local) {
            if (!is_local_) {
                releaseRemoteIfAny();
                is_local_ = true;
                capacity_words_ = kLocalWords;
            }
            memory_resource_ = other_.memory_resource_;
            remote_alignment_ = alignof(uint64_t);
            bit_count_ = other_.bit_count_;
            word_count_ = other_words;
            std::memcpy(local_words_, other_.local_words_, sizeof(local_words_));
            return *this;
        }

        // Remote target. Reuse existing remote allocation when possible.
        const std::size_t required_align = requiredAlignment(other_words);
        const bool same_resource = (memory_resource_ == other_.memory_resource_);
        const bool can_reuse_remote = (!is_local_) && same_resource &&
                                      (capacity_words_ >= other_words) &&
                                      (remote_alignment_ >= required_align);

        if (!can_reuse_remote) {
            releaseRemoteIfAny();
            memory_resource_ = other_.memory_resource_;
            is_local_ = false;
            capacity_words_ = other_words;
            remote_words_ = allocateWords(capacity_words_, required_align);
        }

        bit_count_ = other_.bit_count_;
        word_count_ = other_words;
        std::memcpy(remote_words_, other_.remote_words_, other_words * sizeof(uint64_t));
        clearUnusedBits();
        return *this;
    }

    DynamicBitSet& operator=(DynamicBitSet&& other_) noexcept {
        if (this == &other_) {
            return *this;
        }

        releaseRemoteIfAny();

        bit_count_ = other_.bit_count_;
        word_count_ = other_.word_count_;
        capacity_words_ = other_.capacity_words_;
        memory_resource_ = other_.memory_resource_;
        is_local_ = other_.is_local_;
        remote_alignment_ = other_.remote_alignment_;

        if (is_local_) {
            std::memcpy(local_words_, other_.local_words_, sizeof(local_words_));
        } else {
            remote_words_ = other_.remote_words_;
            other_.remote_words_ = nullptr;
        }

        other_.resetToDefaultState();
        return *this;
    }

    void swap(DynamicBitSet& other_) noexcept {
        if (this == &other_) {
            return;
        }

        DynamicBitSet tmp(std::move(*this));
        *this = std::move(other_);
        other_ = std::move(tmp);
    }

    // =========================================================================
    // Capacity management
    // =========================================================================

    void reserveBits(std::size_t bits_) {
        reserveWords(wordsForBits(bits_));
    }

    void resize(std::size_t bits_, bool value_ = false) {
        const std::size_t new_words = wordsForBits(bits_);
        const std::size_t old_bits = bit_count_;
        const std::size_t old_words = word_count_;

        reserveWords(new_words);

        bit_count_ = bits_;
        word_count_ = new_words;

        uint64_t* words = getWords();

        if (new_words > old_words) {
            std::memset(words + old_words, value_ ? 0xFF : 0x00, (new_words - old_words) * sizeof(uint64_t));
        }

        if (bits_ > old_bits && value_) {
            for (std::size_t i = old_bits; i < bits_; ++i) {
                words[i / kWordBits] |= (uint64_t{1} << (i % kWordBits));
            }
        }

        clearUnusedBits();
    }

    void shrinkToFit() {
        if (is_local_) {
            return;
        }

        if (word_count_ <= kLocalWords) {
            alignas(kSimdAlignment) uint64_t tmp[kLocalWords] = {};
            std::memcpy(tmp, remote_words_, word_count_ * sizeof(uint64_t));
            releaseRemoteIfAny();
            is_local_ = true;
            capacity_words_ = kLocalWords;
            remote_alignment_ = alignof(uint64_t);
            std::memcpy(local_words_, tmp, sizeof(local_words_));
            return;
        }

        if (capacity_words_ == word_count_) {
            return;
        }

        const std::size_t align = requiredAlignment(word_count_);
        uint64_t* new_words = allocateWords(word_count_, align);
        std::memcpy(new_words, remote_words_, word_count_ * sizeof(uint64_t));
        memory_resource_->deallocate(
            remote_words_,
            capacity_words_ * sizeof(uint64_t),
            remote_alignment_);
        remote_words_ = new_words;
        capacity_words_ = word_count_;
        remote_alignment_ = align;
    }

    // =========================================================================
    // Bit access
    // =========================================================================

    [[nodiscard]] MMATH_FORCE_INLINE bool test(std::size_t pos_) const noexcept {
        const uint64_t* words = getWords();
        return (words[pos_ / kWordBits] >> (pos_ % kWordBits)) & uint64_t{1};
    }

    MMATH_FORCE_INLINE DynamicBitSet& set(std::size_t pos_) noexcept {
        uint64_t* words = getWords();
        words[pos_ / kWordBits] |= (uint64_t{1} << (pos_ % kWordBits));
        return *this;
    }

    MMATH_FORCE_INLINE DynamicBitSet& reset(std::size_t pos_) noexcept {
        uint64_t* words = getWords();
        words[pos_ / kWordBits] &= ~(uint64_t{1} << (pos_ % kWordBits));
        return *this;
    }

    MMATH_FORCE_INLINE DynamicBitSet& flip(std::size_t pos_) noexcept {
        uint64_t* words = getWords();
        words[pos_ / kWordBits] ^= (uint64_t{1} << (pos_ % kWordBits));
        return *this;
    }

    DynamicBitSet& setAll() noexcept {
        if (word_count_ == 0) {
            return *this;
        }
        std::memset(getWords(), 0xFF, word_count_ * sizeof(uint64_t));
        clearUnusedBits();
        return *this;
    }

    DynamicBitSet& resetAll() noexcept {
        if (word_count_ == 0) {
            return *this;
        }
        std::memset(getWords(), 0x00, word_count_ * sizeof(uint64_t));
        return *this;
    }

    DynamicBitSet& flipAll() noexcept {
        uint64_t* words = getWords();
        for (std::size_t i = 0; i < word_count_; ++i) {
            words[i] = ~words[i];
        }
        clearUnusedBits();
        return *this;
    }

    // =========================================================================
    // Comparison
    // =========================================================================

    [[nodiscard]] bool operator==(const DynamicBitSet& rhs_) const noexcept {
        if (bit_count_ != rhs_.bit_count_) {
            return false;
        }
        const uint64_t* words = getWords();
        const uint64_t* rhs_words = rhs_.getWords();
        for (std::size_t i = 0; i < word_count_; ++i) {
            if (words[i] != rhs_words[i]) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] bool operator!=(const DynamicBitSet& rhs_) const noexcept {
        return !(*this == rhs_);
    }

    // =========================================================================
    // Query
    // =========================================================================

    [[nodiscard]] MMATH_FORCE_INLINE std::size_t size() const noexcept {
        return bit_count_;
    }

    [[nodiscard]] MMATH_FORCE_INLINE std::size_t wordCount() const noexcept {
        return word_count_;
    }

    [[nodiscard]] MMATH_FORCE_INLINE std::size_t capacity() const noexcept {
        return capacity_words_ * kWordBits;
    }

    [[nodiscard]] MMATH_FORCE_INLINE std::size_t capacityWords() const noexcept {
        return capacity_words_;
    }

    [[nodiscard]] MMATH_FORCE_INLINE bool isSso() const noexcept {
        return is_local_;
    }

    [[nodiscard]] MMATH_FORCE_INLINE bool isAligned32() const noexcept {
        const auto addr = reinterpret_cast<std::uintptr_t>(getWords());
        return (addr % kSimdAlignment) == 0;
    }

    [[nodiscard]] std::size_t memoryUsage() const noexcept {
        if (is_local_) {
            return sizeof(local_words_);
        }
        return capacity_words_ * sizeof(uint64_t);
    }

    [[nodiscard]] MMATH_FORCE_INLINE std::pmr::memory_resource* getMemoryResource() const noexcept {
        return memory_resource_;
    }

    // =========================================================================
    // Raw data access
    // =========================================================================

    [[nodiscard]] MMATH_FORCE_INLINE const uint64_t* data() const noexcept {
        return getWords();
    }

    [[nodiscard]] MMATH_FORCE_INLINE uint64_t* data() noexcept {
        return getWords();
    }

private:
    static constexpr std::size_t wordsForBits(std::size_t bits_) noexcept {
        return (bits_ + (kWordBits - 1)) / kWordBits;
    }

    static std::pmr::memory_resource* sanitizeResource(std::pmr::memory_resource* mr_) noexcept {
        return mr_ ? mr_ : std::pmr::get_default_resource();
    }

    static constexpr std::size_t requiredAlignment(std::size_t words_) noexcept {
        return (words_ >= 256) ? kSimdAlignment : alignof(uint64_t);
    }

    uint64_t* allocateWords(std::size_t count_words_, std::size_t alignment_) {
        if (count_words_ == 0) {
            return nullptr;
        }
        remote_alignment_ = alignment_;
        return static_cast<uint64_t*>(
            memory_resource_->allocate(count_words_ * sizeof(uint64_t), alignment_));
    }

    void releaseRemoteIfAny() noexcept {
        if (!is_local_ && remote_words_) {
            memory_resource_->deallocate(
                remote_words_,
                capacity_words_ * sizeof(uint64_t),
                remote_alignment_);
            remote_words_ = nullptr;
        }
    }

    void resetToDefaultState() noexcept {
        bit_count_ = 0;
        word_count_ = 0;
        capacity_words_ = kLocalWords;
        memory_resource_ = std::pmr::get_default_resource();
        is_local_ = true;
        remote_alignment_ = alignof(uint64_t);
        std::memset(local_words_, 0, sizeof(local_words_));
    }

    void reserveWords(std::size_t requested_words_) {
        if (requested_words_ <= capacity_words_) {
            return;
        }

        // Geometric growth for repeated resize/assignment heavy workloads.
        std::size_t new_capacity = std::max(requested_words_, capacity_words_ * 2);
        if (new_capacity < kLocalWords) {
            new_capacity = kLocalWords;
        }

        const std::size_t new_alignment = requiredAlignment(new_capacity);
        uint64_t* new_words = allocateWords(new_capacity, new_alignment);
        uint64_t* old_words = getWords();
        if (word_count_ > 0) {
            std::memcpy(new_words, old_words, word_count_ * sizeof(uint64_t));
        }
        if (new_capacity > word_count_) {
            std::memset(new_words + word_count_, 0, (new_capacity - word_count_) * sizeof(uint64_t));
        }

        if (!is_local_ && remote_words_) {
            memory_resource_->deallocate(
                remote_words_,
                capacity_words_ * sizeof(uint64_t),
                remote_alignment_);
        }

        remote_words_ = new_words;
        is_local_ = false;
        capacity_words_ = new_capacity;
        remote_alignment_ = new_alignment;
    }

    MMATH_FORCE_INLINE uint64_t* getWords() noexcept {
        return is_local_ ? local_words_ : remote_words_;
    }

    MMATH_FORCE_INLINE const uint64_t* getWords() const noexcept {
        return is_local_ ? local_words_ : remote_words_;
    }

    void clearUnusedBits() noexcept {
        if (bit_count_ == 0 || word_count_ == 0) {
            return;
        }
        const std::size_t used_bits_last = bit_count_ % kWordBits;
        if (used_bits_last == 0) {
            return;
        }
        const uint64_t mask = (uint64_t{1} << used_bits_last) - 1;
        uint64_t* words = getWords();
        words[word_count_ - 1] &= mask;
    }

    // layout metadata
    std::size_t bit_count_;
    std::size_t word_count_;
    std::size_t capacity_words_;
    std::pmr::memory_resource* memory_resource_;
    bool is_local_;
    std::size_t remote_alignment_;

    union {
        alignas(kSimdAlignment) uint64_t local_words_[kLocalWords];
        uint64_t* remote_words_;
    };
};

} // namespace Melosyne
